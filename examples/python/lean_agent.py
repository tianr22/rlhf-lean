import re
import os
import torch
import mistune
import logging
import asyncio
import aiohttp

from typing import List, Dict
from loguru import logger
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential, before_sleep_log
from urllib.parse import urljoin, urlparse, urlunparse
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from typing import Any, Dict

# Global states for the environment
step_idx = 0


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
    response: str = None
    code: str = None
    thought: str = None
    messages: List[Dict] = field(default_factory=list)
    sorry: bool = False

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "code": self.code,
            "thought": self.thought,
            "messages": self.messages,
            "sorry": self.sorry,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trial":
        return cls(
            response=data.get("response"),
            code=data.get("code"),
            thought=data.get("thought"),
            messages=data.get("messages", []),
            sorry=data.get("sorry", False),
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
        final_code = ""
        if self.thought is not None:
            final_code += f"<think>\n{self.thought}\n</think>\n"
        final_code += "\n```lean4\n"
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


def analyze_action(action):
    """
    Returns (thought, lean_code, compiler_output, reward, finished)
    """
    thought, lean_code = parse_response(action)
    if lean_code is None:
        # no lean code found
        reward = 0
        finished = True
    else:
        if "sorry" in lean_code:
            # sorry is not allowed
            reward = 0
            finished = True
        else:
            # compile the lean code
            lean_client = Lean4Client("http://127.0.0.1:12332")
            lean_result = lean_client.verify(
                [{"proof": lean_code, "custom_id": f"{id(lean_code)}"}],
                30,
            )[0]
            passed = True
            messages = lean_result["response"].get("messages", [])
            for message in messages:
                if message.get("severity") == "error":
                    passed = False
            if passed:
                reward = 1
                finished = True
            else:
                reward = 0
                finished = False
    return thought, lean_code, lean_result, reward, finished


def next_state(init_prompt, thought, code, compiler_output):
    observation = init_prompt
    observation += (
        "\nYou have tried the following solution but failed. Compiler feedbacks are inserted into the lean code.\n"
    )
    trial = Trial(response=None, code=code, thought=thought, messages=compiler_output.get("messages", []))
    observation += f"\n{trial.__str__(add_inner_token=False)}\n"
    observation += "\nPlease first analyze the compiler outputs, then generate a new solution."
    return observation


async def step(observation, action, label, **kwargs) -> Dict[str, Any]:
    global step_idx
    print(f"step_idx: {step_idx}")

    # compile the output
    thought, lean_code, lean_result, reward, finished = analyze_action(action)

    if step_idx == 1:
        return {
            "rewards": reward,
            "scores": reward,
            "next_observation": "",
            "done": True,
            "sampling_params": kwargs.get("sampling_params", None),
            "extra_logs": {"dummy_scores": reward},
        }
    else:
        assert step_idx == 0
        step_idx = 1
        next_observation = next_state(observation, thought, lean_code, lean_result)
        return {
            "rewards": reward,
            "scores": reward,
            "next_observation": next_observation,
            "done": True,
            "sampling_params": kwargs.get("sampling_params", None),
            "extra_logs": {"dummy_scores": reward},
        }
