import subprocess
import os

import pynvml
from omegaconf import OmegaConf
from dotenv import load_dotenv


MAX_NUM_SEQS_BY_MODEL = {
    "Qwen/Qwen3.5-0.8B": 1024,
    "Qwen/Qwen3.5-2B": 512,
    "Qwen/Qwen3.5-4B": 512,
    "Qwen/Qwen3.5-9B": 256,
    "Qwen/Qwen3.5-27B": 128,
    "google/gemma-4-E2B-it": 512,
    "google/gemma-4-E4B-it": 512,
    "google/gemma-4-12B-it": 256,
    "google/gemma-4-26B-A4B-it": 128,
    "google/gemma-4-31B-it": 128
}


def register_custom_omegaconf_resolvers():
    OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))
    OmegaConf.register_new_resolver("get_max_num_seqs", lambda x: MAX_NUM_SEQS_BY_MODEL.get(x, 256))

def get_gpu_map():
    """
    Returns a list of GPU indices (as str) as they are currently seen by the system.
    """
    cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    output = subprocess.check_output(cmd).decode("utf-8").strip().split("\n")
    return [line.strip() for line in output]


def get_gpu_vram_usage_gb(gpu_index: int = 0) -> float:
    """Returns the current VRAM usage in GB for a specific GPU."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.used / 1024 / 1024 / 1024


def get_api_key(model_provider="vllm"):
    load_dotenv()
    return os.getenv(f"{model_provider.upper()}_API_KEY", None)


TASK_CRITERIA = {
    "reconciliatory_explanation_effectiveness": {
        "prompt_template_name": "judge/nli",
        "prompt_template_args": {"premise": "{p} {e}", "hypothesis": "{h}"},
        "target_answer_values": ("entailment",),
    },

    "reconciliatory_explanation_coherence": {
        "prompt_template_name": "judge/nli",
        "prompt_template_args": {"premise": "{p}", "hypothesis": "{e}"},
        "target_answer_values": ("entailment", "neutral"),
    },
}

if __name__ == "__main__":
    pass
