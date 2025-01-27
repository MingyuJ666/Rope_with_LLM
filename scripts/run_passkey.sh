export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512' 

#model_name='mistralai/Mistral-7B-Instruct-v0.1'
#model_name='meta-llama/Llama-2-7b-chat-hf'
#model_name='Yukang/LongAlpaca-7B-16k'

#model_name='lmsys/vicuna-7b-v1.5-16k'
pos_interval=500
begin_pos=100
gpuid=$1

#16000 12000 8000 4000
# 8 36 64

model_name="meta-llama/Llama-2-7b-chat-hf"
for s_len in 101
do
    for p_len in 3
    do
        for ((i_p=$begin_pos; i_p<=s_len; i_p+=$pos_interval))
        do
            echo "seq length:  $s_len"
            python syn_data/create_synthetic_set.py\
                --dataset 'passkey_retrieval' \
                --num_gen_example 100  \
                --insert_position $i_p \
                --interval $pos_interval \
                --seq_length $s_len \
                --split 'test' \
                --dataset_folder './synthetic_tasks' \
                --max_generation_length 10 \
                --max_data_num 100 \
                --passkey_length $p_len \
                -m $model_name
        done
    done
done


