#!/bin/bash
cd ~/opencompass

export CUDA_VISIBLE_DEVICES=$1
export HF_ENDPOINT=https://hf-mirror.com

# Names of config files defined under:
# opencompass/opencompass/configs/datasets/a_gsm8k

# Examples: zero-shot and zero-shot-fixed
datasets=(
    "gsm8k_gen_zo_v2"
    "gsm8k_gen_zo_v2_fixed"
    "gsm8k_gen_original_v2"
    "gsm8k_gen_original_v2_fixed"
)

# Names of config files defined under:
# opencompass/opencompass/configs/models/a_test_models_instruct
models=(
    "Qwen2_5_7B_base_vllm_4gpus"
    "Qwen2_5_72B_base_vllm_4gpus"
)

datasets_str=$(IFS=" " ; echo "${datasets[*]}")
models_str=$(IFS=" " ; echo "${models[*]}")

echo "Starting inference with vLLM"

VLLM_WORKER_MULTIPROC_METHOD=spawn \
python run.py \
    --datasets $datasets_str \
    --models $models_str \
    --max-num-workers 1 \
    --max-workers-per-gpu 1 \
    --generation-kwargs seed=42 \
    --work-dir ~/opencompass/math_gsm
    # --reuse 20250630_011117 \
    # --mode eval \

echo "Inference finished"