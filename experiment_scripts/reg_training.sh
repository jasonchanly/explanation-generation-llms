#!/bin/bash


# this is a workaround for hydra-core multirun - to resolve the issue of hydra-core joblib launcher not recognising multiprocessing as a backend when specified
# joblib + multiprocessing backend is needed to resolve issue that CUDA mem allocator can only be called once per process
# and multiprocessing backend launches separate child process for each hydra run in a sweep

BASE_LOG_DIR="reg_training"
LOG_SUBDIR="${1:-dapo}"
MODEL_NAME="${2:-Qwen/Qwen3.5-2B}"

# baseline performance, for comparison
python src/main_eval.py --config-name=reg_post_training_eval -m log_subdir="${BASE_LOG_DIR}/baseline" model.model_name="$MODEL_NAME" model/generation_params=vllm_default_instruct,vllm_default_thinking

SAFE_MODEL_NAME="${MODEL_NAME//\//_}"

# Define parameter ranges
SEEDS=(41 42 43)
BETAS=(0.0)
LEARNING_RATES=(1e-6)
EPSILON_HIGHS=(0.2 0.28 0.36)

for seed in "${SEEDS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for epshigh in "${EPSILON_HIGHS[@]}"; do
        TARGET_DIR="${BASE_LOG_DIR}/${LOG_SUBDIR}/${SAFE_MODEL_NAME}_seed_${seed}_beta_${beta}_lr_${lr}_epshigh_${epshigh}"

        if [ -d "$TARGET_DIR" ]; then
          echo "=== Skipping run | Directory already exists: ${TARGET_DIR} ==="
          continue
        fi

        echo "=== Starting run | seed: ${seed} | beta: ${beta} | lr: ${lr} | epsilon_high: ${epshigh}==="

        python src/grpo_colocate.py \
          --config-name=reg_grpo_training \
          log_subdir="${TARGET_DIR}" \
          model.model_name="${MODEL_NAME}" \
          judge_model.model_name=google/gemma-4-31B-it \
          grpo_config_args.max_steps=-1 \
          grpo_config_args.seed="${seed}" \
          grpo_config_args.beta="${beta}" \
          grpo_config_args.learning_rate="${lr}" \
          grpo_config_args.epsilon_high="${epshigh}"
      done
    done
  done
done

python src/post_train_eval_results.py