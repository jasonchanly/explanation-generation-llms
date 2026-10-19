import os
import re
from pathlib import Path

from jinja2 import Template

PROMPT_BASE_DIR = Path("prompts") if "src" not in os.getcwd() else Path("../prompts")

PROMPT_TEMPLATES = {}
for prompt_template_fpath in Path(PROMPT_BASE_DIR).rglob("*.jinja2"):
    prompt_name = str(prompt_template_fpath).split("prompts/")[-1].replace(".jinja2", "")
    with open(prompt_template_fpath) as f:
        PROMPT_TEMPLATES[prompt_name] = Template(f.read())

PROMPT_TEMPLATE_INFO = {
    "judge/truth-binary": {
        "grade_pattern": re.compile(r"<answer>.*(?P<answer>true|false|undetermined).*</answer>", re.DOTALL),
        "possible_answers": ("true", "false", "undetermined"), },
    "judge/nli": {
        "grade_pattern": re.compile(r"(?P<answer>entailment|neutral|contradiction)", re.DOTALL),
        "possible_answers": ("entailment", "neutral", "contradiction"), },

}
