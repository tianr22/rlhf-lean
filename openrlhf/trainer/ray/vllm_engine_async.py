import re
import os
import ray
import math
import mistune
import logging
import asyncio
import aiohttp

from bs4 import BeautifulSoup
from typing import List, Dict
from loguru import logger
from typing import Dict
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential, before_sleep_log
from dataclasses import dataclass, field
from .vllm_engine import BaseLLMRayActor
from urllib.parse import urljoin, urlparse, urlunparse

from colorama import Fore

DEBUG = True


@ray.remote
class AgentInstance:
    def __init__(self, agent_func_path):
        if agent_func_path.endswith(".py"):
            import importlib.util

            spec = importlib.util.spec_from_file_location("step", agent_func_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            self.agent_step = agent_module.step
        else:
            raise ValueError("Agent path must be a Python file")

    async def step(self, observation, action, label, **kwargs):
        return await self.agent_step(observation, action, label, **kwargs)


@ray.remote
def get_tokenize_text_len(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0])


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        self.agent_func_path = kwargs.pop("agent_func_path")

        # Initialize super class
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()

        os.environ["VLLM_USE_V1"] = "1"
        import vllm

        assert vllm.__version__ > "0.8.5", "Asyn VLLM version must be greater than 0.8.5"

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

        self.lean_client = await Lean4Client.create(base_url="http://localhost:12332")

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def compile_lean(self, lean_code, allow_sorry=False):
        """
        Returns (error, passed, messages)
        """
        # analyze compiler output
        lean_results = await self.lean_client.async_verify(
            [{"code": lean_code, "custom_id": f"{id(lean_code)}"}],
            30,
        )
        lean_result = lean_results["results"][0]
        if not lean_result["error"] is None:
            if DEBUG:
                print(Fore.RED + f"compiler error {lean_result["error"]}")
            return True, False, None
        passed = True
        if not allow_sorry and "sorry" in lean_code:
            passed = False
        messages = lean_result["response"].get("messages", [])
        for message in messages:
            if message.get("severity") == "error":
                passed = False
        return False, passed, messages

    async def add_request(self, sampling_params, prompts, labels, max_length, hf_tokenizer=None, max_steps=10000):
        # Initialize observations and actions for the current prompt
        observation = prompts[0]
        reward = 0
        score = 0
        action_ranges = []
        passed = False
        next_prompt = ""

        # Next sampling budget
        observation_tokens_len = len(
            hf_tokenizer(observation, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        )
        sampling_params.max_tokens = max_length - observation_tokens_len
        if sampling_params.max_tokens <= 0:
            # No budget to generate, stop
            reward = 0
            score = 0
            action_ranges.append((len(observation), len(observation)))
            passed = False
            next_prompt = prompts[0]
        else:
            # rollout llm
            request_output = await self.generate_async(observation, sampling_params)
            action = request_output.outputs[0].text
            observation += action
            action_ranges.append((len(observation), len(observation) + len(action)))
            # analyze generated lean code
            _thought, lean_code = parse_response(action)
            if lean_code is None:
                if DEBUG:
                    print(Fore.RED + "lean code is none")
                reward = 0.0
                score = 0.0
                passed = False
                next_prompt = prompts[0]
                heuristic = -math.inf
            else:
                # compile the code and get the reward
                error, passed, messages = await self.compile_lean(lean_code, allow_sorry=False)
                if error:
                    # compiler error triggered
                    if DEBUG:
                        print(Fore.RED + "error, reward = 0")
                    reward = 0.0
                    score = 0.0
                    passed = False
                    next_prompt = prompts[0]
                    heuristic = -math.inf
                elif passed:
                    # code is good
                    if DEBUG:
                        print(Fore.RED + "passed, reward = 1")
                    reward = 1.0
                    score = 1.0
                    next_prompt = ""
                    heuristic = 0
                else:
                    # code is bad
                    if DEBUG:
                        print(Fore.RED + "failed, reward = 0")
                    reward = 0.0
                    score = 0.0
                    code_with_feedback = Trial(lean_code, messages).__str__()
                    next_prompt = replace_first_lean4_code(prompts[0], code_with_feedback)
                    heuristic = -len(messages)

        final_response = {
            "prompt": prompts[0],
            "label": labels[0],
            "observation": observation,  # observation = (prompt + action)
            "reward": reward,
            "scores": score,
            "extra_logs": {},
            "action_ranges": action_ranges,
            "pass": passed,
            "next_prompt": next_prompt,
            "heuristic": heuristic,
        }
        return final_response

    async def generate_async(self, prompts, sampling_params):
        from vllm.utils import random_uuid

        request_id = random_uuid()
        results_generator = self.llm.generate(prompts, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    async def get_responses(self):
        """
        Synchronously get all completed agent results from the queue.
        Waits for all tasks to complete before returning results.
        Returns: List of all completed agent results.
        """
        # Get all results from the queue
        results = []
        while not self.result_queue.empty():
            try:
                results.append(await self.result_queue.get())
            except asyncio.QueueEmpty:
                break
        return results


class Lean4Client(object):
    """Client for interacting with the Lean 4 verification server.

    This client handles communication with a Lean 4 server for verifying proofs
    and retrieving results. It handles authentication, connection testing,
    and provides methods for synchronous and asynchronous verification.
    """

    def __init__(self, base_url, api_key=None, disable_cache=False) -> None:
        """Initialize the Lean4Client.

        Args:
            base_url (str): Base URL of the Lean 4 server.
            api_key (str, optional): API key for authentication. If None, will try
                to load from LEANSERVER_API_KEY environment variable. Defaults to None.
            disable_cache (bool, optional): Whether to disable result and header caching. Defaults to False.

        Raises:
            Exception: If the Lean server cannot be connected to or is unavailable.
        """
        self.url = base_url
        if api_key is None:
            api_key = os.getenv("LEANSERVER_API_KEY")

        self.api_key = api_key
        self.disable_cache = disable_cache

    @classmethod
    async def create(cls, base_url, api_key=None, disable_cache=False):
        """异步工厂方法创建 Lean4Client 实例"""
        client = cls(base_url, api_key, disable_cache)
        await client._test_connection()
        return client

    async def async_verify(self, codes, timeout, infotree_type=None):
        """verify the proof code and get result

        Args:
            codes (list): The list of lena 4 code to verify.
                Each code is a dict of:
                    - code: The lena 4 code to verify.
                    - custom_id: The custom id of the proof.
            timeout (int): The timeout in seconds.

        Returns:
            response (dict): The response from the server.
                It contains a  key results, which is a list of dictionaries.
                Each dictionary contains the following keys:
                    - code: The custom id of the proof.
                    - error: A string with the error message from the lean server.
                    - response: A dictionary with the response from the LEAN REPL.

        Example:
            >>> client.one_pass_verify("import Mathlib\n\nexample : 2 = 2 := rfl", timeout=60)
            {'results': [{'code': 'test_connection', 'error': None, 'response': {'env': 0}}]}
        """
        json_data = {
            "codes": codes,
            "timeout": timeout,
            "infotree_type": infotree_type,
            "disable_cache": self.disable_cache,
        }
        response = await self._query("post", "/verify", json_data)
        return response

    async def _query(
        self,
        method: str,
        endpoint: str,
        json_data: dict | None = None,
        n_retries: int = 3,
    ) -> dict:
        """
        One single method for sending all requests, with retry behavior controlled by the caller.

        Args:
            method: The HTTP method to use (e.g., "get", "post").
            endpoint: The endpoint to call.
            json_data: The data to send in the request.
            n_retries: Number of retry attempts.

        Returns:
            response: The response from the server.
        """

        # Create retry decorator with dynamic n_retries
        @retry(
            stop=stop_after_attempt(n_retries),  # Dynamic retries based on the caller's argument
            wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
            before_sleep=before_sleep_log(logger, logging.ERROR),  # Optional logging of each retry
        )
        async def query_with_retries(url):
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            # Create a session with trust_env set to True
            if DEBUG:
                print(Fore.RED + "lean client create session")
            async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=3600)) as session:
                async with session.request(
                    method,
                    self._ensure_url_has_scheme(str(urljoin(url, endpoint))),
                    headers=headers,
                    json=json_data,  # Directly send the JSON data
                ) as response:
                    # Get the response body asynchronously and parse it as JSON
                    res = await response.json()

            return res

        # Call the query function with retries
        return await query_with_retries(self.url)

    def _ensure_url_has_scheme(self, url, default_scheme="http"):
        """Ensure URL has a scheme (http/https) prefix.

        Args:
            url (str): The URL to check and potentially modify.
            default_scheme (str, optional): The scheme to add if none exists. Defaults to "http".

        Returns:
            str: URL with a scheme.
        """
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse(f"{default_scheme}://{url}")
        return urlunparse(parsed)

    async def _test_connection(self):
        """Test the connection to the Lean server.

        Sends a simple GET request to the root endpoint to verify
        that the server is available and responsive.

        Raises:
            Exception: If the server cannot be connected to or returns a non-ok status.

        Returns:
            bool: True if connection test passed.
        """
        try:
            if DEBUG:
                print(Fore.RED + "lean client test connection")
            response = await self._query("get", "/")
        except RetryError:
            raise Exception(f"The lean server {self.url} cannot be connected.")

        if response.get("status") != "ok":
            raise Exception(f"The lean server {self.url} cannot be available. {response}")


class CodeLine:
    def __init__(self, line_str, line_number):
        self.line_str = line_str
        self.line_number = line_number
        self.feedback_list = []  # may have multiple feedbacks for single line
        self.add_token_list = {}  # key is the column number, value is the token

    def add_single_feedback(self, info, severity=None):
        self.feedback_list.append([info, severity])

    def add_special_token(self, column_number, token_type):
        token = ""
        if token_type == "info_start":
            token = "<info>"
        elif token_type == "info_end":
            token = "</info>"
        elif token_type == "error_start":
            token = "<error>"
        elif token_type == "error_end":
            token = "</error>"
        else:
            raise ValueError(f"Unknown token type: {token_type}")

        if column_number not in self.add_token_list:
            self.add_token_list[column_number] = token
        else:
            self.add_token_list[column_number] += token

    def __str__(self):
        # sort the token list by column number(key) by decreasing order
        # so that we can insert the token in the correct position
        sorted_token_list = sorted(self.add_token_list.items(), key=lambda x: x[0], reverse=True)
        for column_number, token in sorted_token_list:
            # insert the token at the column number
            self.line_str = self.line_str[:column_number] + token + self.line_str[column_number:]
        string = self.line_str + "\n"
        for feedback in self.feedback_list:
            string += f"<{feedback[1]}_message>\n{feedback[0]}\n</{feedback[1]}_message>\n"
        return string


@dataclass
class Trial:
    code: str = None
    messages: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trial":
        return cls(
            code=data.get("code"),
            messages=data.get("messages", []),
        )

    def _convert_to_codeline(self, str) -> list[CodeLine]:
        """
        Convert a string to a list of CodeLine objects.
        The string is split by new lines, and each line is assigned a line number.
        """
        # TODO may have some problems when '\n' is in the middle of a line
        # currently we ignore this case
        lines = str.split("\n")
        code_lines = []
        for i, line in enumerate(lines):
            code_line = CodeLine(line, i)  # line number starts from 0
            code_lines.append(code_line)
        return code_lines

    def _add_feedback(self, messages, codeline_list: list[CodeLine], add_inner_token: bool = True):
        """
        Add feedback to the list of CodeLine objects.
        The feedback is added to the corresponding line number.
        message has the following structure:
            {
                "data": str,
                "endPos.line": int,
                "endPos.column": int,
                "pos.line": int,
                "pos.column": int,
                "severity": str,
            }
        """
        for message in messages:
            # get the line number from the message
            # currently we add the feedback to endline of the message
            # TODO the actual line number starts from 1, the actual column number starts from 0
            if message["severity"] != "error":
                continue
            start_line = message["pos"]["line"] - 1
            start_column = message["pos"]["column"]
            if message["endPos"] is None:
                # "error: expected token"
                end_line = message["pos"]["line"] - 1
                end_column = message["pos"]["column"]
            else:
                end_line = message["endPos"]["line"] - 1
                end_column = message["endPos"]["column"]
            # get the feedback from the message
            feedback = message["data"]
            # add the feedback to the corresponding CodeLine object
            codeline_list[end_line].add_single_feedback(feedback, message["severity"])
            # add special tokens to the corresponding CodeLine object
            if add_inner_token:
                codeline_list[end_line].add_special_token(end_column, message["severity"] + "_end")
                codeline_list[start_line].add_special_token(start_column, message["severity"] + "_start")
        return codeline_list

    def __str__(self, add_inner_token: bool = True):
        """
        Generate code with feedback.
        The feedback is added to the corresponding line number.
        """
        # convert the string to a list of CodeLine objects
        assert self.code is not None
        codeline_list = self._convert_to_codeline(self.code)
        # add feedback to the list of CodeLine objects
        codeline_list = self._add_feedback(self.messages, codeline_list, add_inner_token)
        # generate the final code string with feedback
        final_code = "\n```lean4\n"
        for code_line in codeline_list:
            final_code += str(code_line)
        # remove last line
        final_code = final_code[:-1]
        final_code += "\n```\n"
        return final_code


def parse_response(response: str) -> tuple[str, str]:
    """
    Returns (thought, code)
    """
    try:
        # translate markdown into html
        markdown = mistune.create_markdown()
        html = markdown(response)
        # extract code
        soup = BeautifulSoup(html, "html.parser")
        code_blocks = []
        for code in soup.find_all("code"):
            classes = code.get("class", [])
            if any(cls.startswith("language-lean") for cls in classes):
                code_blocks.append(code.get_text())
        if len(code_blocks) > 0:
            lean_code = code_blocks[-1].strip()
        else:
            lean_code = None
        # extract thought
        thought_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if thought_match:
            thought_content = thought_match.group(1).strip()
        else:
            thought_content = None
        return thought_content, lean_code
    except:
        return None, None


def replace_first_lean4_code(original: str, new_code: str) -> str:
    """
    替换字符串中第一个 Lean4 代码块的内容
    """
    # 匹配第一个 Lean4 代码块的正则表达式
    pattern = r"```(?:lean4|language-lean4)(.*?```)"

    # 使用非贪婪匹配，确保匹配第一个代码块
    match = re.search(pattern, original, re.DOTALL)

    if not match:
        if DEBUG:
            print(Fore.RED + f"error forming next prompt {original}")
        return original  # 未找到代码块时返回原字符串

    # 计算代码块的起止位置
    start_pos = match.start()
    end_pos = match.end()

    # 提取代码块前的文本
    prefix = original[:start_pos]

    # 提取代码块后的文本
    suffix = original[end_pos:]

    # 构建新代码块（保留原语言标识）
    new_block = f"\n{new_code.strip()}\n"

    # 组合新字符串
    return prefix + new_block + suffix
