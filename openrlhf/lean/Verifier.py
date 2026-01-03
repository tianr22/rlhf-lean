from kimina_client import KiminaClient
from openrlhf.lean.Package import Package, Trial
import mistune
from bs4 import BeautifulSoup
import re

def parse_response(response: str, model_type: str) -> tuple[str, str]:
    """
    Extract the last lean code block from raw response.
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
        if model_type == "Kimina":
            thought_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if thought_match:
                thought_content = thought_match.group(1).strip()
            else:
                thought_content = None
        elif model_type == "Goedel":
            parts = response.split("### Complete Lean 4 Proof")
            thought_content = parts[0] if len(parts) > 1 else None
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return thought_content, lean_code
    except:
        return None, None


class LeanVerifier:
    def __init__(
        self,
        lean_server_url,
        model_type,
    ):
        self.lean_server_url = lean_server_url
        self.model_type = model_type
        self.lean_client = KiminaClient(lean_server_url)

    def verify(self, package: Package):
        # verify the last trial of the package
        assert len(package.trials) > 0
        trial = package.trials[-1]
        assert isinstance(trial, Trial)
        if trial.error:
            return -1e10
        
        response = trial.response
        thought, code = parse_response(response, model_type=self.model_type)
        if self.model_type == "Goedel":
            if thought is None:
                trial.passed = False
                trial.error = "no thought"
                return -1e10
        trial.thought = thought

        if code is None:
            trial.passed = False
            trial.error = "no lean code"
            return -1e10
        if code == "":
            trial.passed = False
            trial.error = "empty lean code"
            return -1e10

        if self.model_type == "Goedel":
            header = """import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"""
            code = header + code
        trial.code = code

        if "sorry" in code:
            trial.passed = False
            trial.sorry = True
            trial.error = "sorry is not allowed"
            return -1e10

        if not "theorem" in code:
            trial.passed = False
            trial.sorry = True
            trial.error = "no theorem found"
            return -1e10
        
        try:
            # send to Lean server
            lean_results = self.lean_client.check(code, show_progress=False)
            result = lean_results.results[0]

            # check for explicit error in result
            if result.error is not None:
                trial.passed = False
                trial.error = result.error
                return -1e10
            
            # check for errors in messages
            trial.messages = result.response.get("messages", [])
            trial.passed = True
            for message in trial.messages:
                if message.get("severity") == "error":
                    trial.passed = False
            if trial.passed:
                package.passed = True

            if trial.passed:
                return 1.0
            else:
                return None
            
        except Exception as e:
                trial.passed = False
                trial.error = f"{type(e).__name__}: {str(e)}"
                return -1e10