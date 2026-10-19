#!/bin/sh

LOG_SUBDIR_STAGE0="${1:-reg_stage0}"
LOG_SUBDIR_STAGE1="${2:-reg_stage1}"

# nli judge results
python src/main_eval.py --config-name=reg_stage0 -m log_subdir="$LOG_SUBDIR_STAGE0" model=vllm_model model.model_name=$(paste -sd, model_lists/reg_judges/candidate_judges_hybrid_models.txt) model/generation_params=vllm_default_instruct
python src/main_eval.py --config-name=reg_stage0 -m log_subdir="$LOG_SUBDIR_STAGE0" model=openai_model model.model_name=$(paste -sd, model_lists/reg_judges/candidate_judges_openai_models.txt) model.generation_params.reasoning_effort=none
python src/main_eval.py --config-name=reg_stage0 -m log_subdir="$LOG_SUBDIR_STAGE0" model=vllm_model model.model_name=$(paste -sd, model_lists/reg_judges/candidate_judges_hybrid_models_small.txt) model/generation_params=vllm_default_instruct

# explanation generation
python src/main_eval.py --config-name=reg_stage1 -m log_subdir="$LOG_SUBDIR_STAGE1" model=vllm_model model.model_name=$(paste -sd, model_lists/explanation_models/hybrid_models.txt) model/generation_params=vllm_default_instruct,vllm_default_thinking
python src/main_eval.py --config-name=reg_stage1 -m log_subdir="$LOG_SUBDIR_STAGE1" model=openai_model model.model_name=$(paste -sd, model_lists/explanation_models/openai_models.txt) model.generation_params.reasoning_effort=none,medium

# evaluate explanations
python src/deferred_scoring.py --config-name=deferred_scoring log_subdir="$LOG_SUBDIR_STAGE1" judge_model=vllm_model judge_model.model_name=google/gemma-4-31B-it model/generation_params@judge_model.generation_params=vllm_default_instruct

python src/analysis.py --parse_stage0 --parse_stage1 --run_glm_analysis