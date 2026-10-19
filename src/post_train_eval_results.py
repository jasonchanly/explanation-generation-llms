import argparse
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
import pandas as pd
from inspect_ai.analysis import evals_df, samples_df, SampleScores, SampleSummary

def reparse_answers(samples_df_eval_fpath, verbose=False):
    """
    Patch to recognise MCQ answers missed by inspectai's built in choice() solver
    :param samples_df_eval_fpath:
    :param verbose:
    :return:
    """
    print(f"Processing {samples_df_eval_fpath}")
    df = samples_df(samples_df_eval_fpath, columns=SampleSummary+SampleScores)

    output_as_dict = df.to_dict(orient='records')

    correct_count = 0
    incorrect_count = 0
    patch_parsed_count = 0
    remain_unparsed_count = 0

    unparsed = []

    patch_regex = re.compile(r"^\(?(?P<response>[A-Z])(\)|\.|$)")
    try:
        for i, entry in enumerate(output_as_dict):
            if entry["score_choice"] == "C":
                correct_count += 1
            elif entry["score_choice_answer"] is None:
                # meaning existing regex failed to detect an answer (ANSWER: [A,B,C,D...])
                answer_found = patch_regex.search(entry["score_choice_explanation"].strip())
                if not answer_found:
                    incorrect_count += 1
                    remain_unparsed_count += 1
                    unparsed.append(entry["score_choice_explanation"])
                elif answer_found["response"] == entry["target"]:
                    correct_count += 1
                    patch_parsed_count += 1
                else:
                    incorrect_count += 1
                    patch_parsed_count += 1
            else:
                incorrect_count += 1
    except KeyError:
        print(f"Answers not in accepted format for reparsing (must be MCQ) - skipping: {samples_df_eval_fpath}")
        return {
            "correct_count": 0, "incorrect_count": 0, "patch_parsed_count": 0, "remain_unparsed_count": 0,
            "example_unparsed": None, "accuracy": 0,
        }

    if verbose:
        print("EXAMPLE ENTRIES of unparsed_answers")
        for i, entry in enumerate(unparsed[:10]):
            print(f"---ENTRY {i} ---")
            print(entry[:100])
        print(f"---END OF ENTRIES---")
        print(f"Score: {correct_count/(correct_count + incorrect_count)}")
        print(f"Total number of instances: {correct_count+incorrect_count}")
        print(f"Patch regex recognised n new answers: {patch_parsed_count}")
        print(f"Answers remain unparsed: {remain_unparsed_count}")

    return {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "patch_parsed_count": patch_parsed_count,
        "remain_unparsed_count": remain_unparsed_count,
        "example_unparsed": None if not unparsed else unparsed[:10],
        "accuracy": correct_count / (correct_count + incorrect_count),
    }


def parse_runs_df(evals_fpath):

    stage1_evals_df = evals_df(evals_fpath)
    # skip unsuccessful or pending runs for now
    stage1_evals_df = stage1_evals_df[stage1_evals_df["status"] == "success"].copy()

    search_reobj = re.compile(r"Qwen_(?P<model_name>.+)_seed_(?P<seed>.+)_beta_(?P<beta>.+)_lr_(?P<lr>.+)_epshigh_(?P<epshigh>.+)")

    model_attributes = [search_reobj.search(model_name.replace("/None", "")) for model_name in stage1_evals_df["model"]]
    model_attributes = [{f"model_seed": int(model_name["seed"]),
                         "model_beta": float(model_name["beta"]),
                         "model_epshigh": float(model_name["epshigh"])} for model_name in model_attributes]

    df = pd.concat([stage1_evals_df, pd.DataFrame(model_attributes)], axis=1)

    patched_results_list = []
    for i, log_fpath in enumerate(df["log"]):
        print(f"Processing log file {i+1} of {len(df)}")
        patched_results_list.append(reparse_answers(log_fpath))

    patched_results = pd.DataFrame(patched_results_list)

    df = pd.concat([df[["task_name", "model_seed", "model_beta", "model_epshigh"]].copy(), patched_results[["accuracy","correct_count","incorrect_count","patch_parsed_count", "remain_unparsed_count"]].copy()], axis=1)

    return df.groupby(by=["task_name", "model_beta", "model_epshigh"]).mean()


def get_baseline_results(baseline_evals_fpath):
    output_fpath = Path(f"{baseline_evals_fpath}/baseline_results.csv")
    baseline_evals_df = evals_df(baseline_evals_fpath)

    targeted_tasks = ["musr", "eval_tasks.py@true_detective"]
    baseline_evals_df = baseline_evals_df[baseline_evals_df["status"] == "success"].copy()
    baseline_evals_df = baseline_evals_df[baseline_evals_df["model"] == "vllm/Qwen/Qwen3.5-2B"].copy()

    baseline_evals_df = baseline_evals_df[baseline_evals_df["task_name"].isin(targeted_tasks)].copy()

    baseline_evals_df["model_is_thinking"] = [json.loads(gen_args)["extra_body"]["chat_template_kwargs"]["enable_thinking"] for gen_args in baseline_evals_df["model_generate_config"]]

    patched_results_list = []
    for i, log_fpath in enumerate(baseline_evals_df["log"]):
        print(f"Processing log file {i+1} of {len(baseline_evals_df)}")
        patched_results_list.append(reparse_answers(log_fpath))

    patched_results = pd.DataFrame(patched_results_list)

    df = pd.concat([baseline_evals_df.reset_index(), patched_results], axis=1)
    df = df[["task_name","model","model_is_thinking","accuracy"]].copy()
    df.to_csv(output_fpath, index=False)
    return df


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_evals_fpath", type=str, default=f"reg_training/baseline")
    parser.add_argument("--evals_fpath", type=str, default=f"reg_training/dapo")

    args = parser.parse_args()
    print("Getting baseline results", flush=True)
    baseline_results = get_baseline_results(f"{os.getenv('INSPECT_LOG_DIR')}/{args.baseline_evals_fpath}")
    print("Baseline results obtained", flush=True)
    results = parse_runs_df(evals_fpath=f"{os.getenv('INSPECT_LOG_DIR')}/{args.evals_fpath}")
    results.to_csv(f"{os.getenv('INSPECT_LOG_DIR')}/{args.evals_fpath}/post_train_eval_results.csv")