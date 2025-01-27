#!/bin/bash

# GPU Configuration
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
gpuid=${1:-0}  

# Model Configuration
MODEL_CONFIGS=(
    "mistralai/Mistral-7B-Instruct-v0.1"
    "meta-llama/Llama-2-7b-chat-hf"
    "Yukang/LongAlpaca-7B-16k"
    "lmsys/vicuna-7b-v1.5-16k"
)
model_name="${MODEL_CONFIGS[1]}"  
# Sequence Parameters
pos_interval=500
begin_pos=50
seq_length=128
passkey_length=6

# Dataset Parameters
DATASET_CONFIG=(
    --dataset 'passkey_retrieval'
    --split 'test'
    --dataset_folder './synthetic_tasks'
    --num_gen_example 200
    --max_data_num 200
    --max_generation_length 10
)



echo "Running with configuration:"
echo "Model: $model_name"
echo "GPU ID: $gpuid"
echo "Sequence length: $seq_length"
echo "Passkey length: $passkey_length"

for s_len in $seq_length; do
    for p_len in $passkey_length; do
        for ((i_p=begin_pos; i_p<=s_len; i_p+=pos_interval)); do
            echo "Processing: seq length=$s_len, passkey length=$p_len, position=$i_p"
            
            python syn_data/create_synthetic_set.py \
                "${DATASET_CONFIG[@]}" \
                --insert_position $i_p \
                --interval $pos_interval \
                --seq_length $s_len \
                --passkey_length $p_len \
                -m "$model_name"
        done
    done
done


