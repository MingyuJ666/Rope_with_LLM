# This is all the models we will use
# language model: meta-llama/Meta-Llama-3-8B-Instruct ,
# meta-llama/Llama-2-7b-chat-hf, 'mistralai/Mistral-7B-Instruct-v0.3',
# Qwen/Qwen2.5-7B-Instruct,google/gemma-2-9b-it, facebook/opt-2.7b, ai21labs/Jamba-v0.1
# "Qwen/Qwen2-VL-2B-Instruct"
# "EleutherAI/gpt-neox-20b"
# "EleutherAI/gpt-neo-1.3B"
# "gpt2"
pattern="save_attn"
round=2

# save_attn_map
# save_attn_map
CUDA_VISIBLE_DEVICES=1 python llm_example_save_attn.py \
    --model_name gpt2\
    --pattern "$pattern" \
    --round "$round" \
    #2>&1 | tee ./imdb_destroy.log
