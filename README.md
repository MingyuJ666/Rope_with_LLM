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


```shell
>> git clone https://github.com/zjunlp/Mol-Instruction
>> cd demo
```

Step 1, install Gradio by running：`pip install gradio`. 

Step 2, specify the parameters in the [generate.sh](./demo/generate.sh) file.

```shell
>> CUDA_VISIBLE_DEVICES=0 python generate.py \
    --CLI False\
    --protein False\
    --load_8bit \
    --base_model $BASE_MODEL_PATH \
    --share_gradio True\
    --lora_weights $FINETUNED_MODEL_PATH \
```

For models fine-tuned on *molecule-oriented* and *biomolecular text* instructions, please set `$FINETUNED_MODEL_PATH` to `'zjunlp/llama-molinst-molecule-7b'` or `'zjunlp/llama-molinst-biotext-7b'`.

For the model fine-tuned on *protein-oriented* instructions, you need to perform additional steps as described in [this folder](https://github.com/zjunlp/Mol-Instructions/tree/main/demo).

Step 3, run the [generate.sh](./demo/generate.sh) file in the repository： 

```shell
>> sh generate.sh
```

We offer two methods: the first one is command-line interaction, and the second one is web-based interaction, which provides greater flexibility. 

1. Use the following command to enter **web-based interaction**:
```shell
>> python generate.py
```
  The program will run a web server and output an address. Open the output address in a browser to use it.

2. Use the following command to enter **command-line interaction**:
```shell
>> python generate.py --CLI True
```
  The disadvantage is the inability to dynamically change decoding parameters.

<p align="center">
  <img alt="Demo" src=fig/gradio_interface_gif.gif style="width: 700px; height: 340px;"/>
</p>

<h3 id="3-3"> 💡 3.3 Quantitative Experiments</h3>

To investigate whether Mol-Instructions can enhance LLM’s understanding of biomolecules, we conduct the following quantitative experiments. 
For detailed experimental settings and analysis, please refer to our [paper](https://arxiv.org/pdf/2306.08018.pdf). Please refer to the [evaluation code](https://github.com/zjunlp/Mol-Instructions/tree/main/evaluation) to conduct the same experiments.

## 🧪 Molecular generation tasks
