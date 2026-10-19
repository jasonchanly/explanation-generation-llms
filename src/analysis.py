import argparse
import os
import json
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from inspect_ai.analysis import evals_df, samples_df, SampleScores, SampleSummary

POST_ANALYSIS_CANDIDATE_JUDGES = ("vllm/google/gemma-4-31B-it",)

def export_dataset_ids_for_sampling(log_subdir_stage0, in_scope_judges, output_fpath, shuffle=False):
    stage0_evals_df = evals_df(log_subdir_stage0)
    stage0_mapping = {}
    stage0_evals_df = stage0_evals_df[
        stage0_evals_df["task_arg_dataset_args"] == '{"dset_split": "mnli"}'].copy().reset_index()
    ids_for_sampling = set()

    for i, entry in enumerate(stage0_evals_df["log"]):
        judge_model = stage0_evals_df.at[i, "model"]
        if stage0_mapping.get(judge_model, None):
            print(f"Warning: NLI judge model {judge_model} has duplicate entries in stage0 log_dir")
            continue
        elif judge_model not in in_scope_judges:
            print(f"Ignoring judgments of {judge_model}")
            continue

        stage0_sample_level_df = samples_df(entry, columns=SampleSummary + SampleScores)

        # only count if judge model had predicted the original P / H as contradictory in the first place
        filtered_ids = stage0_sample_level_df[(stage0_sample_level_df["metadata_label"] == "contradiction")
                                              & (stage0_sample_level_df["score_nli_label_matcher"] == "C")]["metadata_id"]

        if not ids_for_sampling:
            ids_for_sampling = set(filtered_ids)
        else:
            ids_for_sampling = ids_for_sampling.intersection(filtered_ids)

    ids_for_sampling = list(ids_for_sampling)
    if shuffle:
        np_rng = np.random.default_rng(42)
        np_rng.shuffle(ids_for_sampling)

    pd.DataFrame({"selected_sample_id": ids_for_sampling}).to_csv(f"{log_subdir_stage0}/{output_fpath}", index=False)

    return ids_for_sampling


def glm_analysis(df):
    """
    Correlations of
    - Difficulty vs explanation success.
    - Reasoning length vs explanation success (thinking mode only)

    Uses logistic GLMs with standard errors clustered by problem.
    """

    required = {
        "problem_id", "model", "mode",
        "difficulty", "solved", "reasoning_tokens"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = df.copy()
    for col in ["difficulty", "solved", "reasoning_tokens"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=["problem_id", "model", "mode", "difficulty", "solved"]
    )

    if not data["solved"].isin([0, 1]).all():
        raise ValueError("'solved' must contain only 0 and 1.")

    def fit_glm(formula, subset):
        return smf.glm(
            formula=formula,
            data=subset,
            family=sm.families.Binomial()
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": subset["problem_id"]}
        )

    def extract_result(result, term, scale=1):
        beta = result.params[term]
        ci_low, ci_high = result.conf_int().loc[term]

        return {
            "coefficient": float(beta),
            "odds_ratio": float(np.exp(beta * scale)),
            "ci_95_or": [
                float(np.exp(ci_low * scale)),
                float(np.exp(ci_high * scale))
            ],
            "p_value": float(result.pvalues[term]),
        }

    # difficulty -> success, controlling for mode and model
    difficulty_model = fit_glm(
        "solved ~ difficulty + C(mode) + C(model)", data
    )

    difficulty_result = extract_result(
        difficulty_model, "difficulty", scale=0.1
    )
    difficulty_result["odds_ratio_per_0.1_difficulty"] = (
        difficulty_result.pop("odds_ratio")
    )
    difficulty_result["ci_95_or_per_0.1"] = (
        difficulty_result.pop("ci_95_or")
    )
    difficulty_result.update({
        "n_rows": int(len(data)),
        "n_problems": int(data["problem_id"].nunique())
    })

    # reasoning length -> success, thinking mode only
    thinking = data[
        (data["mode"].astype(str).str.lower() == "thinking")
        & data["reasoning_tokens"].notna()
        & (data["reasoning_tokens"] > 0)
    ].copy()

    if len(thinking):
        thinking["log2_reasoning_tokens"] = np.log2(
            thinking["reasoning_tokens"]
        )

        reasoning_model = fit_glm(
            "solved ~ log2_reasoning_tokens + difficulty + C(model)",
            thinking
        )

        reasoning_result = extract_result(
            reasoning_model, "log2_reasoning_tokens"
        )
        reasoning_result["odds_ratio_per_doubling"] = (
            reasoning_result.pop("odds_ratio")
        )
        reasoning_result["ci_95_or_per_doubling"] = (
            reasoning_result.pop("ci_95_or")
        )
        reasoning_result.update({
            "n_rows": int(len(thinking)),
            "n_problems": int(thinking["problem_id"].nunique())
        })
    else:
        reasoning_result = None

    # descriptive Spearman correlations
    problem_summary = data.groupby("problem_id").agg(
        difficulty=("difficulty", "first"),
        solve_rate=("solved", "mean")
    )

    rho_d, p_d = spearmanr(
        problem_summary["difficulty"],
        problem_summary["solve_rate"]
    )

    if len(thinking):
        rho_r, p_r = spearmanr(
            thinking["reasoning_tokens"], thinking["solved"]
        )
    else:
        rho_r, p_r = np.nan, np.nan

    return {
        "difficulty_to_solved": difficulty_result,
        "reasoning_length_to_solved": reasoning_result,
        "descriptive_spearman": {
            "difficulty_vs_problem_solve_rate": {
                "rho": float(rho_d),
                "p_value": float(p_d)
            },
            "reasoning_tokens_vs_solved": {
                "rho": float(rho_r),
                "p_value": float(p_r)
            }
        }
    }

def thresholded_stage0_nli(log_subdir):
    output_fpath = "ANALYSIS_thresholded_nli.csv"

    output_entries = []
    thresholds = [0.0] + [np.round(x, 2) for x in np.arange(0.6, 1.0, 0.05)]

    df = evals_df(log_subdir)

    for i, entry in enumerate(df["log"]):
        sample_level_df = samples_df(entry, columns=SampleSummary + SampleScores)

        if "metadata_label_weight" not in sample_level_df.columns:
            output_entries.append({
                "model": df.at[i, "model"],
                "model_args": df.at[i, "model_args"],
                "model_generate_config": df.at[i, "model_generate_config"],
                "task_arg_dataset_args": df.at[i, "task_arg_dataset_args"],
                "threshold": 0.0,
                "support": len(sample_level_df),
                "accuracy": sample_level_df["score_nli_label_matcher"].value_counts(normalize=True)["C"],
            })
            continue

        for weight_threshold in thresholds:
            filtered_df = sample_level_df[sample_level_df["metadata_label_weight"] >= weight_threshold].copy()
            filtered_accuracy = filtered_df["score_nli_label_matcher"].value_counts(normalize=True)["C"]

            output_entries.append({
                "model": df.at[i, "model"],
                "model_args": df.at[i, "model_args"],
                "model_generate_config": df.at[i, "model_generate_config"],
                "task_arg_dataset_args": df.at[i, "task_arg_dataset_args"],
                "threshold": weight_threshold,
                "support": len(filtered_df),
                "accuracy": filtered_accuracy,
            })

    output_df = pd.DataFrame(output_entries)
    output_df.to_csv(f"{log_subdir}/{output_fpath}", index=False)


def stage1_explain(log_subdir_stage0, log_subdir_stage1):
    output_fpath = "ANALYSIS_explain_w_bootstrapping.csv"
    stage0_evals_df = evals_df(log_subdir_stage0)
    # each entry model should be unique -
    # get a dictionary of {judge_model_name: log_fpath) from the evals df
    stage0_mapping = {}
    stage0_evals_df = stage0_evals_df[stage0_evals_df["task_arg_dataset_args"] == '{"dset_split": "mnli"}'].copy().reset_index()
    print(stage0_evals_df.head(3))

    for i, entry in enumerate(stage0_evals_df["log"]):
        if stage0_mapping.get(stage0_evals_df.at[i, "model"], None):
            print(f"Warning: model {stage0_evals_df.at[i, 'model']} has duplicate entries in stage0 log_dir")
            continue

        stage0_sample_level_df = samples_df(entry, columns=SampleSummary + SampleScores)

        # only count if judge model had predicted the original P / H as contradictory in the first place
        filtered_ids = stage0_sample_level_df[(stage0_sample_level_df["metadata_label"] == "contradiction")
                                             & (stage0_sample_level_df["score_nli_label_matcher"] == "C")]["metadata_id"]

        stage0_mapping[stage0_evals_df.at[i, "model"].replace("vllm/", "").replace("openai/", "")] = filtered_ids

    stage1_evals_df = evals_df(log_subdir_stage1)

    output_entries = []
    for i, entry in enumerate(stage1_evals_df["log"]):
        if "reconcile_contradictions" not in stage1_evals_df.at[i, "task_name"]:
            continue

        model_generate_config = json.loads(stage1_evals_df.at[i, "model_generate_config"])

        if model_generate_config.get("reasoning_effort", None):
            model_is_reasoning = model_generate_config["reasoning_effort"] != "none"
        else:
            model_is_reasoning = model_generate_config["extra_body"]["chat_template_kwargs"]["enable_thinking"]

        stage1_sample_level_df = samples_df(entry, columns=SampleSummary + SampleScores)
        grade_columns = []
        for col in stage1_sample_level_df.columns:
            if f"{col}_metadata" in stage1_sample_level_df.columns and f"{col}_answer" in stage1_sample_level_df.columns:
                grade_columns.append(col)

        grader_models_and_scores = {json.loads(stage1_sample_level_df[f"{score_col}_metadata"][0])["grader_model"]: score_col
                                    for score_col in grade_columns}

        stage1_sample_level_df.set_index("metadata_id", drop=False, inplace=True)

        # only counting score column where judge model is the one doing the scoring
        for grader_model_name, score_col in grader_models_and_scores.items():
            # only counting rows where judge model had predicted a P/H contradiction in the first place
            filtered_df = stage1_sample_level_df.loc[stage0_mapping[grader_model_name]].copy()
            assert len(filtered_df) > 0

            scores_df = pd.DataFrame([json.loads(score_dict) for score_dict in filtered_df[score_col] if not pd.isna(score_dict)])
            print(f"n scores recognised: {len(scores_df)}", flush=True)

            scores_col_names = ["reconciliatory_explanation_effectiveness", "reconciliatory_explanation_coherence", "reconciliatory_explanation_effectiveness__and__reconciliatory_explanation_coherence"]
            # bootstrapping to get 95% confidence interval
            n_boot = 1000
            bootstrapped_means = {score_col_name: [] for score_col_name in scores_col_names}
            for boot_i in range(n_boot):
                bootstrap_sampled_df = scores_df.sample(n=len(scores_df), replace=True, random_state=boot_i)
                for score_col_name in scores_col_names:
                    bootstrapped_means[score_col_name].append(bootstrap_sampled_df[score_col_name].mean())

            ci_dict = {f"{col_name}_{percentile_def}": percentile_val for col_name, col_means in bootstrapped_means.items()
                       for percentile_def, percentile_val in zip(
                    ["lower", "upper"], np.percentile(col_means, [2.5, 97.5])
                )
                       }

            output_entries.append(
                {
                    "explanation_model": stage1_evals_df.at[i, "model"],
                    "explanation_model_is_reasoning": model_is_reasoning,
                    "judge_model": grader_model_name,
                    **scores_df.mean().to_dict(),
                    **ci_dict,
                }
            )

    output_df = pd.DataFrame(output_entries)
    output_df.to_csv(f"{log_subdir_stage1}/{output_fpath}", index=False)


def run_glm_analysis(ids_to_analyse, log_subdir_stage1):
    stage1_evals_df = evals_df(log_subdir_stage1)

    metadata_c_weight_list = []
    expl_success_list = []

    # for log res
    ids_list = []
    model_is_thinking_list = []
    model_name_list = []
    n_reasoning_tokens_list = []

    reasoning_tokens_usage_tally = []
    for i, entry in enumerate(stage1_evals_df["log"]):
        explanation_model = stage1_evals_df.at[i, "model"]

        model_generate_config = json.loads(stage1_evals_df.at[i, "model_generate_config"])
        if model_generate_config.get("reasoning_effort", None):
            model_is_reasoning = model_generate_config["reasoning_effort"] != "none"
        else:
            model_is_reasoning = model_generate_config["extra_body"]["chat_template_kwargs"]["enable_thinking"]

        stage1_sample_level_df = samples_df(entry, columns=SampleSummary + SampleScores)

        stage1_sample_level_df = stage1_sample_level_df[stage1_sample_level_df["metadata_id"].isin(ids_to_analyse)].copy()
        if len(stage1_sample_level_df) == 0:
            print(f"No samples available: {entry}")
            continue

        stage1_sample_level_df["expl_is_successful"] = [json.loads(x)["reconciliatory_explanation_effectiveness__and__reconciliatory_explanation_coherence"] for x in stage1_sample_level_df["score_model_graded_by_tf"]]
        stage1_sample_level_df["output_token_count"] = [list(json.loads(x).values())[0]["output_tokens"] for x in stage1_sample_level_df["model_usage"]]

        metadata_c_weight_list.extend(list(stage1_sample_level_df["metadata_c_weight"]))
        expl_success_list.extend(list(stage1_sample_level_df["expl_is_successful"]))

        # for log res
        ids_list.extend(list(stage1_sample_level_df["metadata_id"]))
        model_name_list.extend([explanation_model]*len(stage1_sample_level_df))
        model_is_thinking_list.extend(["thinking" if model_is_reasoning else "non-thinking"]*len(stage1_sample_level_df))
        n_reasoning_tokens_list.extend([n for n in stage1_sample_level_df["output_token_count"]])

        if model_is_reasoning:
            reasoning_tokens_usage_tally.append({
                "model_name": explanation_model,
                "overall_avg_output_token_count": stage1_sample_level_df["output_token_count"].mean(),
                "successful_expls_avg_output_token_count": stage1_sample_level_df[stage1_sample_level_df["expl_is_successful"] == 1]["output_token_count"].mean(),
                "unsuccessful_expls_avg_output_token_count":
                    stage1_sample_level_df[stage1_sample_level_df["expl_is_successful"] == 0]["output_token_count"].mean(),
                "overall_std_output_token_count": stage1_sample_level_df["output_token_count"].std(),
                "successful_expls_std_output_token_count":
                    stage1_sample_level_df[stage1_sample_level_df["expl_is_successful"] == 1]["output_token_count"].std(),
                "unsuccessful_expls_std_output_token_count":
                    stage1_sample_level_df[stage1_sample_level_df["expl_is_successful"] == 0]["output_token_count"].std(),
            })

    # basic usage tally
    pd.DataFrame(reasoning_tokens_usage_tally).to_csv(f"{log_subdir_stage1}/ANALYSIS_reasoning_tokens_usage_tally.csv", index=False)

    logres_df = pd.DataFrame({
        "problem_id": ids_list,
        "model": model_name_list,
        "mode": model_is_thinking_list,
        "difficulty": metadata_c_weight_list,
        "solved": expl_success_list,
        "reasoning_tokens": n_reasoning_tokens_list,
    })

    logres_results = glm_analysis(logres_df)

    with open(f"{log_subdir_stage1}/ANALYSIS_logres_reasoninglen_vs_cweight_vs_explsuccess.json", "w", encoding="utf-8") as f:
        json.dump(logres_results, f, indent=4, ensure_ascii=False)

    print("Analysis completed")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_subdir_stage0", type=str, default=f"reg_stage0")
    parser.add_argument("--log_subdir_stage1", type=str, default=f"reg_stage1")
    parser.add_argument("--parse_stage0", action="store_true")
    parser.add_argument("--parse_stage1", action="store_true")
    parser.add_argument("--run_glm_analysis", action="store_true")

    args = parser.parse_args()

    if args.log_subdir_stage0 and args.parse_stage0:
        thresholded_stage0_nli(f"{os.getenv('INSPECT_LOG_DIR')}/{args.log_subdir_stage0}")

    if args.log_subdir_stage0 and args.log_subdir_stage1 and args.parse_stage1:
        stage1_explain(f"{os.getenv('INSPECT_LOG_DIR')}/{args.log_subdir_stage0}",
                       f"{os.getenv('INSPECT_LOG_DIR')}/{args.log_subdir_stage1}")

    if args.run_glm_analysis:
        fpath = Path(f"{os.getenv('INSPECT_LOG_DIR')}/{args.log_subdir_stage0}/ANALYSIS_ids_sorted_gemma4-331b-pred-con.csv")
        if not fpath.exists():
            included_ids = export_dataset_ids_for_sampling(args.log_subdir_stage0,
                                                           in_scope_judges=POST_ANALYSIS_CANDIDATE_JUDGES,
                                                           output_fpath=fpath)["selected_sample_id"]
        else:
            included_ids = pd.read_csv(fpath)["selected_sample_id"]

        run_glm_analysis(included_ids, args.log_subdir_stage1)


if __name__ == "__main__":
    main()
