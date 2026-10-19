import os
import signal
import subprocess
import time
import atexit
import socket

import requests


CUSTOM_REASONING_CONFIGS = {
    "qwen": '{"reasoning_start_str": "<think>", "reasoning_end_str": "\\n\\nI have to give the solution based on the reasoning directly now.</think>"}',
    "gemma": '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "\\n\\nI have to give the solution based on the reasoning directly now.<channel|>"}',
}


def get_model_server(model: str, server_kwargs=None, **kwargs):
    API_PREFIXES = ("gpt-", "o1-", "o3-", "claude-", "gemini-", "openrouter/")

    if any(model.lower().startswith(prefix) for prefix in API_PREFIXES):
        return APIWrapperServer(model=model, server_kwargs=server_kwargs, **kwargs)

    else:
        return VLLMServer(model=model, server_kwargs=server_kwargs, **kwargs)


class VLLMServer:
    HOST = "127.0.0.1"
    DEFAULT_PORT = 8005  # 8005 to avoid port clashes

    def __init__(
            self,
            model: str,
            allocated_gpus=("0",),
            startup_timeout: int = 7200,
            server_kwargs: dict = None,
    ):
        self.model = model
        self.startup_timeout = startup_timeout
        self.proc = None
        self.server_kwargs = server_kwargs or {}

        if "Qwen3." in model:
            print("Adding reasoning-parser qwen3", flush=True)
            self.server_kwargs["reasoning-parser"] = "qwen3"
        elif "gemma-4" in model:
            print("Adding reasoning-parser gemma4", flush=True)
            self.server_kwargs["reasoning-parser"] = "gemma4"

        atexit.register(self.stop)
        self.allocated_gpus = ",".join(allocated_gpus)
        self.api_key = "NOT USED"
        self.model_provider = "vllm"

        self.port = self.DEFAULT_PORT
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.HOST, self.port))
                    break  # Port is available
                except OSError:
                    self.port += 1  # Port in use, try next one

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    @property
    def base_url(self):
        return f"http://{self.HOST}:{self.port}"

    @property
    def api_base(self):
        return f"{self.base_url}/v1"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False  # let the exception propagate normally

    def start(self):
        current_env = os.environ.copy()
        current_env["CUDA_VISIBLE_DEVICES"] = self.allocated_gpus
        current_env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        # debugging
        # current_env["PYTHONUNBUFFERED"] = "1"
        # current_env["VLLM_LOGGING_LEVEL"] = "DEBUG"

        if self.proc is not None:
            raise RuntimeError("Server already running")

        self.proc = subprocess.Popen(self._build_cmd(),
                                     preexec_fn=os.setsid,
                                     env=current_env,
                                     )

        self._wait_until_ready()
        return self.api_base

    def _handle_signal(self, signum, frame):
        self.stop()
        # Re-raise so the process exits with the right code
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = None
            return

        pgid = os.getpgid(self.proc.pid)

        try:
            os.killpg(pgid, signal.SIGINT)
            self.proc.wait(timeout=15)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(pgid, signal.SIGTERM)
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    self.proc.wait()
                except ProcessLookupError:
                    pass
        finally:
            self.proc = None

    def _wait_until_ready(self):
        deadline = time.time() + self.startup_timeout

        while time.time() < deadline:

            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self.proc.returncode}"
                )

            try:
                r = requests.get(
                    f"{self.base_url}/health",
                    timeout=2,
                )

                if r.status_code == 200:
                    return

            except requests.RequestException:
                pass

            time.sleep(2)

        raise TimeoutError(
            f"vLLM did not become healthy within "
            f"{self.startup_timeout} seconds"
        )

    def _build_cmd(self):
        # python code snippet injected directly into the subprocess
        # clean up to address
        # [rank0]:[W1208 14:45:45.628528669 ProcessGroupNCCL.cpp:1496]
        # Warning: WARNING: destroy_process_group() was not called before program exit,
        # which can leak resources. For more info,
        # please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

        cmd = ["vllm", "serve", self.model, "--host", self.HOST, "--port", str(self.port),
               "--tensor-parallel-size", str(len(self.allocated_gpus.split(","))),
               "--disable-uvicorn-access-log", "--shutdown-timeout", "30",
               # debugging
               # "--enable-log-requests" # , "--enable-log-outputs",
               ]

        for key, value in self.server_kwargs.items():
            if key == "add_custom_reasoning_end_str":
                custom_flag = f"--reasoning-config"
                custom_value = None
                for model_type, custom_config in CUSTOM_REASONING_CONFIGS.items():
                    if model_type in self.model.lower():
                        custom_value = custom_config
                        break
                if not custom_value:
                    print(f"Warning: no custom reasoning config defined for {self.model}, defaulting to qwen reasoning config - this may cause unexpected behaviour", flush=True)
                    custom_value = CUSTOM_REASONING_CONFIGS["qwen"]

                cmd.extend([custom_flag, custom_value])
            else:
                print(f"Adding {key}: {value}", flush=True)
                flag = f"--{key}"
                if isinstance(value, bool) and value:
                    if value:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(value)])

        return cmd


class APIWrapperServer:
    def __init__(
            self,
            model: str = "gpt-5.4-mini-2026-03-17",
            model_provider: str = "openai",
            server_kwargs: dict = None,
            **kwargs
    ):
        """
        Bare basic 'server' wrapper when not hosting locally but calling models via paid APIs
        :param model:
        :param model_kwargs:
        :param kwargs:
        """
        self.model = model
        self.server_kwargs = server_kwargs or {}
        self.model_provider = model_provider
        atexit.register(self.stop)

    @property
    def base_url(self):
        return "https://api.openai.com"

    @property
    def api_base(self):
        return f"{self.base_url}/v1"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self):
        return self.api_base

    def stop(self):
        pass

    def _wait_until_ready(self) -> None:
        pass

