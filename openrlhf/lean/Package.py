from dataclasses import dataclass, field
from typing import Dict, List

from kimina_client import KiminaClient

INSERT_COMPILER_INFO = True


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
        sorted_token_list = sorted(
            self.add_token_list.items(), key=lambda x: x[0], reverse=True
        )
        for column_number, token in sorted_token_list:
            # insert the token at the column number
            self.line_str = (
                self.line_str[:column_number] + token + self.line_str[column_number:]
            )
        string = self.line_str + "\n"
        for feedback in self.feedback_list:
            string += f"<{feedback[1]}>\n{feedback[0]}\n</{feedback[1]}>\n"
        return string


@dataclass
class Trial:
    response: str = None
    code: str = None
    thought: str = None
    messages: List[Dict] = field(default_factory=list)
    sorry: bool = False
    error: str = None
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "code": self.code,
            "thought": self.thought,
            "messages": self.messages,
            "sorry": self.sorry,
            "error": self.error,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trial":
        return cls(
            response=data.get("response"),
            code=data.get("code"),
            thought=data.get("thought"),
            messages=data.get("messages", []),
            sorry=data.get("sorry", False),
            error=data.get("error", ""),
            passed=data.get("passed", False),
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

    def _add_feedback(
        self, messages, codeline_list: list[CodeLine], add_inner_token: bool = True
    ):
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
            start_line = message["pos"]["line"] - 1
            start_column = message["pos"]["column"]
            if not "endPos" in message.keys() or not message["endPos"]:
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
                codeline_list[end_line].add_special_token(
                    end_column, message["severity"] + "_end"
                )
                codeline_list[start_line].add_special_token(
                    start_column, message["severity"] + "_start"
                )
        return codeline_list

    def __str__(self, add_inner_token: bool = False):
        """
        Generate code with feedback.
        The feedback is added to the corresponding line number.
        """
        # convert the string to a list of CodeLine objects
        assert self.code is not None
        codeline_list = self._convert_to_codeline(self.code)

        # remove duplicate messages
        unique_messages = []
        seen_messages = set()
        for message in self.messages:
            message_tuple = (
                message["data"],
                message["pos"]["line"],
                message["pos"]["column"],
                message.get("endPos", message["pos"]).get("line", None),
                message.get("endPos", message["pos"]).get("column", None),
                message["severity"],
            )
            if message_tuple not in seen_messages:
                seen_messages.add(message_tuple)
                unique_messages.append(message)
        self.messages = unique_messages

        # add feedback to the list of CodeLine objects
        codeline_list = self._add_feedback(
            self.messages, codeline_list, add_inner_token
        )
        # generate the final code string with feedback
        final_code = ""
        final_code += "\n```lean4\n"
        for code_line in codeline_list:
            final_code += str(code_line)
        # remove last line
        final_code = final_code[:-1]
        final_code += "\n```\n"
        return final_code


@dataclass
class Package:
    problem_id: int
    unique_id: int
    informal: str
    formal: str
    passed: bool
    # tree
    root_score: float
    trials: List[Trial] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    cluster: List[int] = field(default_factory=list)

    def reset(self):
        self.passed = False
        self.trials = []
        self.parents = []
        self.scores = []

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "unique_id": self.unique_id,
            "informal": self.informal,
            "formal": self.formal,
            "passed": self.passed,
            "root_score": self.root_score,
            "trials": [trial.to_dict() for trial in self.trials],
            "parents": self.parents,
            "scores": self.scores,
            "cluster": self.cluster,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Package":
        return cls(
            problem_id=data["problem_id"],
            unique_id=data["unique_id"],
            informal=data["informal"],
            formal=data["formal"],
            passed=data.get("passed", False),
            root_score=data.get("root_score", 0),
            trials=[
                Trial.from_dict(trial_data) for trial_data in data.get("trials", [])
            ],
            parents=data.get("parents", []),
            scores=data.get("scores", []),
            cluster=data.get("cluster", []),
        )

    def add_trial(self, trial: Trial, parent: int, score: float):
        self.trials.append(trial)
        self.parents.append(parent)
        self.scores.append(score)

    def cluster_trial(self):
        """Cluster the last trial into existing clusters.
        If it does not belong to any existing cluster, create a new cluster."""

        def compare_compiler_messages(messages1, messages2):
            if len(messages1) != len(messages2):
                return False
            for m1, m2 in zip(messages1, messages2):
                if m1 != m2:
                    return False
            return True

        assert len(self.trials) > 0
        if self.trials[-1].passed:
            self.cluster[-1] = -1  # passed trial
            return
        if len(self.trials) <= 1:
            self.cluster[-1] = 1  # first trial
            return
        last_trial = self.trials[-1]
        for i in range(len(self.trials) - 1):
            if compare_compiler_messages(last_trial.messages, self.trials[i].messages):
                self.cluster[-1] = self.cluster[i]
                return
        self.cluster[-1] = max(self.cluster) + 1

    def generate_prompt(
        self, trial_idx: int, model_type: str, add_inner_token: bool = False
    ) -> str:
        """
        Generate the prompt for the problem.
        If no feedback is provided, generate initial prompt.
        """
        if model_type == "Kimina":
            if trial_idx == -1:
                # generate initial prompt
                prompt = "Think about and solve the following problem step by step in Lean 4.\n"
                if self.informal:
                    prompt += f"\n{self.informal}\n"
                prompt += f"\n```lean4\n{self.formal}\n```\n"
            else:
                # generate feedback prompt based on last trial
                if INSERT_COMPILER_INFO:
                    # insert compiler info inside
                    prompt = f"Think about and solve the following problem step by step in Lean 4.\n"
                    if not self.informal is None:
                        prompt += f"\n{self.informal}\n"
                    prompt += self.trials[trial_idx].__str__(
                        add_inner_token=add_inner_token
                    )
                else:
                    # append compiler info
                    prompt = f"Think about and solve the following problem step by step in Lean 4.\n"
                    if not self.informal is None:
                        prompt += f"\n{self.informal}\n"
                    prompt += f"\n```lean4\n{self.trials[trial_idx].code}\n```\n"
                    compiler_messages = ""
                    for message in self.trials[trial_idx].messages:
                        message_str = f"\n{message["severity"]} (line {message["pos"]["line"]}, column {message["pos"]["column"]}):\n{message["data"]}\n"
                        compiler_messages += message_str
                    prompt += f"\n<error>\n{compiler_messages}\n</error>"
        elif model_type == "Goedel":
            if trial_idx == -1:
                prompt = f"""
Complete the following Lean 4 code:

```lean4
{self.formal}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()
            else:
                prompt = f"""
Fix the following Lean 4 code:

{self.trials[trial_idx].__str__(add_inner_token=add_inner_token)}

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan analyzing compiler errors and proof steps.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return prompt


if __name__ == "__main__":
    # test the function
    code_str = r"""import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that for any natural number $n$, 12 divides $4^{n+1} + 20$.-/
theorem induction_12dvd4expnp1p20 (n : ℕ) : 12 ∣ 4 ^ (n + 1) + 20 := by
  induction n with
  | zero =>
    use 2
    norm_num
  | succ k ih =>
    have h1 : 4 ^ (k + 2) + 20 = 4 * 4 ^ (k + 1) + 20 := by ring
    rw [h1]
    omega
""".strip()

    client = KiminaClient(api_url="http://localhost:12334")
    result = client.check(code_str)
    messages = result.results[0].response.get("messages", [])
    trial = Trial(response="", code=code_str, thought="", messages=messages)

    package = Package(
        problem_id=0,
        unique_id=0,
        informal=None,
        formal=None,
        passed=False,
        root_score=0,
        trials=[trial],
        parents=[-1],
        scores=[0],
        cluster=[0],
    )
    print(package.generate_prompt(0, "Kimina"))
