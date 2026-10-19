import os
from dotenv import load_dotenv
import inspect
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from model_server import get_model_server

from huggingface_hub import hffs

from inspect_ai import score
from inspect_ai.model import get_model, GenerateConfig
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.util import registry_create

import utils

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def run_scoring(cfg):
    gpu_map = utils.get_gpu_map()

    judge_model_name = cfg.judge_model.model_path or cfg.judge_model.model_name

    judge_model_server = get_model_server(model=judge_model_name,
                                          allocated_gpus=(gpu_map[-1],),
                                          server_kwargs=OmegaConf.to_container(cfg.judge_model.server_args, resolve=True))
    judge_model_server.start()

    # see issue #1 in main_eval.py for this workaround
    if judge_model_server.model_provider == "vllm":
        os.environ["VLLM_BASE_URL"] = judge_model_server.api_base

    judge_model = get_model(
        model=f"{judge_model_server.model_provider}/{judge_model_name}",
        base_url=judge_model_server.api_base,
        api_key=utils.get_api_key(judge_model_server.model_provider),
        # https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig
        config=GenerateConfig(adaptive_connections=OmegaConf.to_container(cfg.judge_model.server_args, resolve=True).get("max-num-seqs", None),
                              **OmegaConf.to_container(cfg.judge_model.generation_params, resolve=True)),
    )

    print(f"Running judge model {judge_model_name} with config: {judge_model.config}", flush=True)

    full_log_dirpath = f"{os.getenv('INSPECT_LOG_DIR')}/{cfg.log_subdir}"

    # to work around quirk that hffs.glob will return filenames with "hf://" dropped
    logs_to_be_scored = [f"hf://{file}" for file in hffs.glob(f"{full_log_dirpath}/**/*.eval")
                         ]

    for log_to_be_scored_fpath in logs_to_be_scored:

        eval_log = read_eval_log(log_to_be_scored_fpath)
        # check if model grading is needed e.g. "model_graded_qa"
        if not any(["model_" in eval_scorer.name for eval_scorer in eval_log.eval.scorers]):
            continue  # no model scoring required

        rescorers = []
        for eval_scorer in eval_log.eval.scorers:
            rescorers.append(
                registry_create("scorer", name=eval_scorer.name,
                                **eval_scorer.options)
            )

        print(f"List of scorers: {rescorers}", flush=True)
        for rescorer in rescorers:
            print(rescorer)
            print(inspect.getclosurevars(rescorer), flush=True)

        scored_log = score(eval_log,
                           scorers=rescorers,
                           model_roles={"grader": judge_model},
                           # ignore squiggly yellow lines - updated inspectai support model_roles in score
                           action="append",
                           display="none")

        scored_log.eval.scorers[-1].metadata = {"judge_model_name": judge_model_name,
                                                "judge_model_generation_params":
                                                    OmegaConf.to_container(cfg.judge_model.generation_params, resolve=True)}

        write_eval_log(scored_log, log_to_be_scored_fpath)  # overwriting original log with new scores added

    judge_model_server.stop()

    print("POST-EVAL SCORING COMPLETE")


@hydra.main(version_base=None, config_path="../configs", config_name="deferred_scoring_config.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    run_scoring(cfg)


if __name__ == '__main__':
    load_dotenv()
    utils.register_custom_omegaconf_resolvers()
    main()
