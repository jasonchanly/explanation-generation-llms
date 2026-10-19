import os
import gc
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose
import wandb

from main_eval import run_eval
import rewards
from model_server import get_model_server
import utils

logging.getLogger("weave").setLevel(logging.WARNING)
logging.getLogger("weave.trace.weave_client").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

def run_grpo(cfg):
    wandb.init(project=cfg.project_name,
               config=OmegaConf.to_container(cfg, resolve=True),
               job_type="train",
               resume="allow",
               reinit="create_new",
               mode="offline",
               )

    gpu_map = utils.get_gpu_map()
    assert len(gpu_map) > 1, "At least 2 GPUs required (1 for judge model; 1 for trainer+rollout colocate)"
    judge_model_name = cfg.judge_model.model_path or cfg.judge_model.model_name
    train_model_name = cfg.model.model_path or cfg.model.model_name
    judge_server = get_model_server(model=judge_model_name,
                                    allocated_gpus=(gpu_map[-1],),
                                    server_kwargs=OmegaConf.to_container(cfg.judge_model.server_args, resolve=True))

    judge_server.start()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_map[:-1])

    from trl import GRPOTrainer
    from trl import GRPOConfig
    from openai import AsyncOpenAI

    from dataset_loader import GenericDataset

    judge_client = AsyncOpenAI(base_url=judge_server.api_base, api_key=utils.get_api_key(judge_server.model_provider))

    grpo_config_args = OmegaConf.to_container(cfg.grpo_config_args, resolve=True)
    reformatted_generation_kwargs = {}
    for k, v in grpo_config_args.get("generation_kwargs", {}).items():
        if isinstance(v, dict):
            for v_sub_key, v_sub_val in v.items():
                reformatted_generation_kwargs[v_sub_key] = v_sub_val
        else:
            reformatted_generation_kwargs[k] = v

    if "chat_template_kwargs" in list(reformatted_generation_kwargs.keys()):
        del reformatted_generation_kwargs["chat_template_kwargs"]

    grpo_config_args["generation_kwargs"] = reformatted_generation_kwargs

    training_args = GRPOConfig(**grpo_config_args)
    train_dataset = GenericDataset(**cfg.train_dataset).to_hf_dataset(**cfg.prompt_template)

    reward_functions = []
    for reward_func_name, reward_func_args in cfg.reward_funcs.items():
        if "judge" in reward_func_name:
            reward_functions.append(
                rewards.REWARDS_REGISTER[reward_func_name](**OmegaConf.to_container(reward_func_args, resolve=True) |
                                                             {"judge_client": judge_client,
                                                              "judge_model_name": judge_model_name}))
        else:
            reward_functions.append(rewards.REWARDS_REGISTER[reward_func_name](**reward_func_args))

    # monkey patch to insert vllm server-level arguments (i.e. reasoning_parser)
    # that neither GRPOTrainer nor GRPOCOnfig exposes
    if "vllm" not in sys.modules:
        import vllm
        original_vllm_init = vllm.LLM.__init__

        def patched_vllm_init(self, *args, **kwargs):
            kwargs.setdefault("reasoning_parser", "qwen3" if "Qwen" in cfg.model.model_name else "gemma4")
            # any other missing engine args can also be passed here
            return original_vllm_init(self, *args, **kwargs)

        vllm.LLM.__init__ = patched_vllm_init

    trainer = GRPOTrainer(
        model=train_model_name,
        args=training_args,
        reward_funcs=reward_functions,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()
    judge_server.stop()
    print("Training complete - model saved.", flush=True)

    # VLLM CLEANUP
    import torch
    if hasattr(trainer, "callback_handler") and trainer.callback_handler:
        print("RECOGNIZED: callback_handler")
        for cb in trainer.callback_handler.callbacks:
            if hasattr(cb, "_llm"):
                cb._llm = None

    if hasattr(trainer, "optimizer") and trainer.optimizer:
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.optimizer.state.clear()
        trainer.optimizer = None

    # shut down vllm engine
    vllm_gen = getattr(trainer, "vllm_generation", None)
    if vllm_gen and hasattr(vllm_gen, "llm"):
        print("RECOGNIZED: trl_trainer.vllm_generation.llm")
        llm_engine = getattr(vllm_gen.llm, "llm_engine", None)
        if llm_engine and hasattr(llm_engine, "engine_core"):
            print("RECOGNIZED: trl_trainer.vllm_generation.llm.engine_core")

            if hasattr(llm_engine.engine_core, "shutdown"):
                print("RECOGNIZED: trl_trainer.vllm_generation.llm.engine_core.shutdown")
                try:
                    llm_engine.engine_core.shutdown()
                    print("engine_core.shutdown() succeeded")
                except Exception:
                    print("FAILED: engine_core.shutdown()")
                    pass

    # destroy vLLM distributed process groups
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        print("FAILED: Destroy distributed environment")
        pass

    if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
        trainer.accelerator.free_memory()

    # detach key sub-objects from trainer to break circular references
    attrs_to_clear = ["vllm_generation", "model", "model_wrapped", "ref_model", "optimizer", "lr_scheduler",
                      "accelerator", "callback_handler", "reward_funcs", "_buffered_inputs", "reward_weights", ]

    for attr in attrs_to_clear:
        if hasattr(trainer, attr):
            print(f"RECOGNIZED: {attr}", flush=True)
            setattr(trainer, attr, None)

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    final_model_path = Path(cfg.grpo_config_args.output_dir)

    checkpoint_paths = sorted(
        list(final_model_path.glob("checkpoint-*")), key=lambda p: int(p.name.split("-")[-1])
    )

    return checkpoint_paths + [final_model_path]


@hydra.main(version_base=None, config_path="../configs", config_name="training_grpo_config.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    gpu_map = utils.get_gpu_map()
    print("Before training - GPU memory")
    for gpu in gpu_map:
        print(f"GPU {gpu} memory current usage: {utils.get_gpu_vram_usage_gb(int(gpu))}", flush=True)
    trained_model_fpaths = run_grpo(cfg)

    print("After training and cleanup - GPU memory")
    for gpu in gpu_map:
        print(f"GPU {gpu} memory current usage: {utils.get_gpu_vram_usage_gb(int(gpu))}", flush=True)

    if not cfg.post_train_eval:
        return

    print("Evaluating model post-training", flush=True)
    eval_model_fpaths = {"last": [trained_model_fpaths[-1]], "all": trained_model_fpaths}[cfg.post_train_eval]
    for trained_model_fpath in eval_model_fpaths:
        eval_cfg = compose(
            config_name="reg_post_training_eval",
            overrides=[
                f"model/generation_params=vllm_default_instruct",
                f"model.model_path={trained_model_fpath}",
                f"model.model_name={cfg.model.model_name}",
                f"wandb_training_run_id={wandb.run.id}",
                f"log_subdir={cfg.log_subdir}",
                f"model.server_args.gpu-memory-utilization=0.5",
                f"model.server_args.max-num-seqs={utils.MAX_NUM_SEQS_BY_MODEL.get(cfg.model.model_name, 256)}",
            ]
        )
        run_eval(eval_cfg)
    print("Post-trained model evaluated.", flush=True)


if __name__ == "__main__":
    load_dotenv()
    utils.register_custom_omegaconf_resolvers()
    main()
