import os
from datetime import datetime

from dotenv import load_dotenv
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from inspect_flow import FlowSpec, FlowOptions, FlowTask
from inspect_flow.api import run
from inspect_ai.util import AdaptiveConcurrency

from model_server import get_model_server
import utils


def task_requires_judge(task):
    if any(_keyword in task for _keyword in ["counterexample", "contradiction", "harveyai_lab"]):
        return True
    else:
        return False


def task_requires_sandbox(task):
    if any(_keyword in task for _keyword in ["harveyai_lab"]):
        return True
    else:
        return False


def run_eval(cfg):
    # late import to workaround connection error issues -
    # model server at port 8006 whereas inspectai would still call model at port 8005
    from inspect_ai.model import get_model, GenerateConfig
    gpu_map = utils.get_gpu_map()
    print(f"[{datetime.now().strftime('%Y%m%d-%H:%M:%S')}] Initialising wandb...", flush=True)

    wandb.init(project=cfg.project_name, resume="allow", config=OmegaConf.to_container(cfg, resolve=True),
               group=f"{cfg.wandb_training_run_id or 'vanilla_models'}",
               reinit="finish_previous",
               mode="offline",
               job_type="eval")

    eval_model_name = cfg.model.model_path or cfg.model.model_name  # path takes precedent if specified
    print(f"[{datetime.now().strftime('%Y%m%d-%H:%M:%S')}] Initialising model...", flush=True)
    model_server = get_model_server(model=eval_model_name,
                                    allocated_gpus=(gpu_map[0],),
                                    server_kwargs=OmegaConf.to_container(cfg.model.server_args, resolve=True))
    model_server.start()
    print(f"[{datetime.now().strftime('%Y%m%d-%H:%M:%S')}] Model initialised.", flush=True)

    # issue #1 - temporary workaround - inspectai get_model does not respect base_url when specified as function argument
    # - it is calling 8000 even when model served at 8001, resulting in Connection Error
    if model_server.model_provider == "vllm":
        os.environ["VLLM_BASE_URL"] = model_server.api_base

    model = get_model(
        model=f"{model_server.model_provider}/{eval_model_name}",
        # see issue #1 above, base_url is no longer specified in get_model
        # - this is to avoid inspectai hashing base_url + a particular port as part of config,
        # so that eval retries would not re-run if port is different
        # - update: reinstating base_url - as ConnectionError persists even though api_base changes
        base_url=model_server.api_base,
        api_key=utils.get_api_key(model_server.model_provider),
        # https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig
        config=GenerateConfig(
            adaptive_connections=AdaptiveConcurrency(min=10,
                                                     max=OmegaConf.to_container(cfg.model.server_args, resolve=True).get(
                                                         "max-num-seqs", None),
                                                     start=128),
            **OmegaConf.to_container(cfg.model.generation_params, resolve=True),
        ))

    # check if cfg.tasks has some tasks that require a judge model
    judge_model = None
    judge_model_server = None

    tasks_requiring_judge = {task: task_requires_judge(task) for task in cfg.tasks.keys()}

    if (not cfg.model_scoring_later) and (any(tasks_requiring_judge.values())):
        # if so instantiate the judge model server
        judge_model_name = cfg.judge_model.model_path or cfg.judge_model.model_name

        judge_model_server = get_model_server(model=judge_model_name,
                                              allocated_gpus=(gpu_map[-1],),
                                              server_kwargs=OmegaConf.to_container(cfg.judge_model.server_args,
                                                                                   resolve=True))
        judge_model_server.start()
        judge_model = get_model(
            model=f"{judge_model_server.model_provider}/{judge_model_name}",
            base_url=judge_model_server.api_base,
            api_key=utils.get_api_key(judge_model_server.model_provider),
            # https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig
            config=GenerateConfig(
                adaptive_connections=OmegaConf.to_container(cfg.judge_model.server_args, resolve=True).get("max-num-seqs",
                                                                                                           None),
                **OmegaConf.to_container(cfg.judge_model.generation_params)),
        )

    model_roles = {"grader": judge_model} if judge_model else {}

    tasks = []

    for task_name, param_groups in cfg.tasks.items():
        if not param_groups:
            tasks.append(FlowTask(name=task_name, model=model, model_roles=model_roles,
                                  sandbox="podman" if task_requires_sandbox(task_name) else None,
                                  ))
        else:
            for param_group in param_groups:
                tasks.append(FlowTask(name=task_name, model=model, model_roles=model_roles,
                                      sandbox="podman" if task_requires_sandbox(task_name) else None,
                                      args=OmegaConf.to_container(param_group, resolve=True),
                                      ))

    print("Listing initialised tasks")
    for task_i, task in enumerate(tasks):
        print(f"{task_i}: {task.name} -- {task.args}")

    run_results = []
    if not cfg.model_scoring_later:
        spec = FlowSpec(
            store=".inspect_flow_store",
            log_dir=f"{os.getenv('INSPECT_LOG_DIR')}/{cfg.log_subdir}",
            # https://meridianlabs-ai.github.io/inspect_flow/reference/inspect_flow.html#flowoptions
            options=FlowOptions(log_realtime=False, log_buffer=2048, log_dir_allow_dirty=True, display="none",
                                score=not cfg.model_scoring_later,
                                metadata=OmegaConf.to_container(cfg, resolve=True)),
            tasks=tasks,
        )
        judge_model_server.stop()
        run_results.append(run_eval(spec))

    else:
        # defer model grading - other tasks that don't need model-grading don't need to be deferred
        if [eval_task for eval_task in tasks if not tasks_requiring_judge[eval_task.name]]:
            spec_score_now = FlowSpec(
                store=None,
                log_dir=f"{os.getenv('INSPECT_LOG_DIR')}/{cfg.log_subdir}",
                # https://meridianlabs-ai.github.io/inspect_flow/reference/inspect_flow.html#flowoptions
                options=FlowOptions(log_realtime=False, log_buffer=2048, log_dir_allow_dirty=True, display="none",
                                    score=True,
                                    metadata=OmegaConf.to_container(cfg, resolve=True)),
                tasks=[eval_task for eval_task in tasks if not tasks_requiring_judge[eval_task.name]],
            )
            run_results.append(run(spec=spec_score_now))

        if [eval_task for eval_task in tasks if tasks_requiring_judge[eval_task.name]]:
            spec_score_later = FlowSpec(
                store=".inspect_flow_store",
                log_dir=f"{os.getenv('INSPECT_LOG_DIR')}/{cfg.log_subdir}",
                # https://meridianlabs-ai.github.io/inspect_flow/reference/inspect_flow.html#flowoptions
                options=FlowOptions(log_realtime=False, log_buffer=2048, log_dir_allow_dirty=True, display="none",
                                    score=False,
                                    metadata=OmegaConf.to_container(cfg, resolve=True)),
                tasks=[eval_task for eval_task in tasks if tasks_requiring_judge[eval_task.name]],
            )

            run_results.append(run(spec=spec_score_later))

        if not run_results:
            raise ValueError("No tasks processed.")

    # https://github.com/meridianlabs-ai/inspect_flow/blob/ffa0e8545d3e4c0ccf86720a0e3ecd8770ecab75/src/inspect_flow/_api/api.py#L76

    model_server.stop()
    print("RUN COMPLETE")

    return run_results


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(
        f"Calculated max-model-len: {OmegaConf.to_container(cfg.model.server_args, resolve=True).get('max-model-len', None)}")
    run_eval(cfg)


if __name__ == '__main__':
    load_dotenv()

    print(f"[{datetime.now().strftime('%Y%m%d-%H:%M:%S')}] Initialising hydra-core...", flush=True)
    utils.register_custom_omegaconf_resolvers()
    main()
