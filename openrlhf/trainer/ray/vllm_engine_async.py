import re
import os
import ray
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

    async def add_requests(self, sampling_params, prompts, labels, max_length, hf_tokenizer=None, max_steps=10000):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the step function.
        Results are streamed back as each agent completes its execution.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of prompts to process
            labels: List of labels corresponding to prompts
            max_steps: Maximum number of interaction steps
        """

        # Create semaphore to control concurrent task execution
        NUM_TASKS = os.environ.get("OPENRLHF_ASYNC_NUM_TASKS", 128)
        semaphore = asyncio.Semaphore(NUM_TASKS)

        # TODO: lean agent
        async def execute_agent(prompt, label, sampling_params):
            async with semaphore:
                # Create a unique agent instance for this prompt
                agent_instance = AgentInstance.remote(self.agent_func_path)

                # Initialize observations and actions for the current prompt
                observation = prompt
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
                # No budget to generate, stop
                if sampling_params.max_tokens <= 0:
                    reward = 0
                    score = 0
                    action_ranges.append((len(observation), len(observation)))
                    passed = False
                    next_prompt = prompt
                else:
                    request_output = await self.generate_async(observation, sampling_params)
                    action = request_output.outputs[0].text
                    observation += action

                    action_ranges.append((len(observation), len(observation) + len(action)))
                    # analyze generated lean code
                    _, lean_code = parse_response(action)
                    if lean_code is None:
                        reward = 0
                        score = 0
                        passed = False
                        next_prompt = prompt
                    else:
                        # compile the code and get the reward
                        _, passed, messages = compile_lean(lean_code, allow_sorry=False)
                        if passed:
                            reward = 1.0
                            score = 1.0
                            next_prompt = ""
                        else:
                            reward = 0.0
                            score = 0.0
                            code_with_feedback = Trial(lean_code, messages).__str__()
                            next_prompt = replace_first_lean4_code(prompt, code_with_feedback)

                ray.kill(agent_instance)

                # Store the final response when agent execution is complete
                final_response = {
                    "prompt": prompt,
                    "label": label,
                    "observation": observation,  # observation = (prompt + action)
                    "reward": reward,
                    "scores": score,
                    "extra_logs": {},
                    "action_ranges": action_ranges,
                    "pass": passed,
                    "next_prompt": next_prompt,
                }
                await self.result_queue.put(final_response)

        # Create and start tasks for all agent executions with controlled concurrency
        import copy

        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(execute_agent(prompt, label, copy.deepcopy(sampling_params)))

        # Run the async code using the class's event loop
        await asyncio.gather(*tasks)

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
    def __init__(self, base_url, api_key=None, disable_cache=False) -> None:
        self.url = base_url
        if api_key is None:
            api_key = os.getenv("LEANSERVER_API_KEY")

        self.api_key = api_key
        self.disable_cache = disable_cache

        self._test_connection()

    def verify(self, codes, timeout, infotree_type=None):
        return asyncio.run(self.async_verify(codes, timeout, infotree_type))

    async def async_verify(self, codes, timeout, infotree_type=None):
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
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse(f"{default_scheme}://{url}")
        return urlunparse(parsed)

    def _test_connection(self):
        try:
            response = asyncio.run(self._query("get", "/"))
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


def compile_lean(lean_code, allow_sorry=False):
    """
    Returns (error, passed, messages)
    """
    lean_client = Lean4Client(base_url="115.182.62.193:12332")
    lean_results = lean_client.verify(
        [{"proof": lean_code, "custom_id": f"{id(lean_code)}"}],
        30,
    )
    # analyze compiler output
    lean_result = lean_results["results"][0]
    if not lean_result["error"] is None:
        return True, False, None
    passed = True
    if not allow_sorry and "sorry" in lean_code:
        passed = False
    messages = lean_result["response"].get("messages", [])
    for message in messages:
        if message.get("severity") == "error":
            passed = False
    return False, passed, messages


def replace_first_lean4_code(original: str, new_code: str) -> str:
    """
    替换字符串中第一个 Lean4 代码块的内容
    """
    # 匹配第一个 Lean4 代码块的正则表达式
    pattern = r"```(?:lean4|language-lean4)(.*?```)"

    # 使用非贪婪匹配，确保匹配第一个代码块
    match = re.search(pattern, original, re.DOTALL)

    if not match:
        return original  # 未找到代码块时返回原字符串

    # 计算代码块的起止位置
    start_pos = match.start()
    end_pos = match.end()

    # 提取代码块前的文本
    prefix = original[:start_pos]

    # 提取代码块后的文本
    suffix = original[end_pos:]

    # 构建新代码块（保留原语言标识）
    new_block = f"```lean4\n{new_code.strip()}\n```"

    # 组合新字符串
    return prefix + new_block + suffix
