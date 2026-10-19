#!/bin/sh

# test on manually written adversarial explanations
LOG_SUBDIR_CONTROL_MANUAL_WRITTEN="${1:-reg_control_manually_written_explanations}"

python src/main_eval.py --config-name=reg_stage0_extra_for_judge_control_manual_explanations --log_subdir="$LOG_SUBDIR_CONTROL_MANUAL_WRITTEN"

# test on LLM-generated adversarial explanations
LOG_SUBDIR_CONTROL_LLM_WRITTEN="${2:-reg_control_llm_written_explanations}"

python src/main_eval.py --config-name=reg_stage1_extra_for_judge_control --log_subdir="$LOG_SUBDIR_CONTROL_LLM_WRITTEN"

python src/deferred_scoring.py --config-name=deferred_scoring log_subdir="$LOG_SUBDIR_CONTROL_LLM_WRITTEN" judge_model=vllm_model judge_model.model_name=google/gemma-4-31B-it model/generation_params@judge_model.generation_params=vllm_default_instruct

python src/analysis.py --parse_stage1 --log_subdir_stage1="$LOG_SUBDIR_CONTROL_LLM_WRITTEN"