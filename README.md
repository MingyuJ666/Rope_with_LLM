<h1 align="center">  🍊 Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding </h1>
<h3 align="center">  A comprehensive analysis of striped massive values appearing in attention Q and K matrices are mainly responsible for contextual knowledge understanding. </h3>

<p align="center">
  📃 <a href="https://arxiv.org/abs/2306.08018" target="_blank">Paper</a> 



## 🆕 News
- \[**Feb 2025**\] We submit the paper [Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding](https://github.com/zjunlp/MolGen), the striped massive values appearing in attention Q and K matrices are mainly responsible for contextual knowledge understanding, and this phenomenon originates from RoPE's effects on low-frequency channels.




## 📝 How to run the Code

<h3 id="3-1"> 🤗 1. Environment Setting </h3>

```
>> conda create -n myenv python=3.9
>> conda activate myenv
>> pip install -r requirements.txt
```

<h3 id="3-2"> 📊 2. Passkey Retrieve Data Synthesis</h3>

```
>> bash scripts/run_passkey.sh 
```
#### Passkey Retrieval Data Synthesis Parameters：

\[**Core Parameters:**\] seq_length=128: Total length of the generated text sequence (must be ≥ 101)
begin_pos=50: Starting position for password insertion, passkey_length=6: Length of the password to be inserted

\[**Data Generation Controls:**\] num_gen_example=200: Number of examples to generate, max_data_num=200: Maximum number of examples in the final dataset
Note: To adjust the dataset size, both num_gen_example and max_data_num should be set to the same value. For example, to generate 300 examples, set both parameters to 300.

⚠️ Important: Setting seq_length below 101 will result in an error.

```
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
```
<h3 id="3-3"> 3. 🎯 Get the Embedding Vector in different LLMs </h3>

```
>>  sh scripts/save_attn_map.sh 
```

Step 1, pattern="save_attn", select a language model like meta-llama/Llama-2-7b-chat-hf

```shell
>> CUDA_VISIBLE_DEVICES=0 python llm_example_save_attn.py \
    --model_name meta-llama/Llama-2-7b-chat-hf\
    --pattern "$pattern" \
    --round "$round" \
```
Step 2: How to choose layers in different LLM

For example, **Llama**

Find modeling_llama.py and search the code below. Then you can change the code: ```if GLOBAL_L == 1 or GLOBAL_L == 2 or GLOBAL_L == 10:```, choose the layer ```GLOBAL_L``` you want.
```
global GLOBAL_L
        
        head_set = range(32)
        if GLOBAL_L in range(32):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            

            if GLOBAL_L == 1 or GLOBAL_L == 2 or GLOBAL_L == 10:
                print(query_states.shape)
                torch.save(query_states, f"{save_dir}/q_merged_attn_weights_layer{GLOBAL_L}.pt")
                torch.save(key_states, f"{save_dir}/k_merged_attn_weights_layer{GLOBAL_L}.pt")
                torch.save(value_states, f"{save_dir}/v_merged_attn_weights_layer{GLOBAL_L}.pt")
                
```

Step 3: Use attn.ipynb or appendix_result/run.ipynb to show the result.
![architect](image/fig1.png)
