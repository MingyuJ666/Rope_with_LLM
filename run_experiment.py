# transfromers version 4.32.0
import warnings
import argparse
import os
import pandas as pd

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from datasets import load_dataset
from modify_utils import update_path
import torch
import json
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM
from modeling_mistral import MistralForCausalLM
from modeling_qwen2 import Qwen2ForCausalLM
from modeling_gemma2 import Gemma2ForCausalLM
import re
from datasets import load_dataset

from awq import AutoAWQForCausalLM

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from llmcompressor.transformers import SparseAutoModelForCausalLM

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def extract_last_answer(text):
    try:

        # matches = re.findall(r'\d+(?:,\d+)*', text)
        matches = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)
        matches = [float(match.replace(",", "")) for match in matches]

        # matches = [int(match.replace(",", "")) for match in matches]

        last_answer = matches[-1] if matches else None
    except ValueError as e:
        print(f"Error: {e}")
        last_answer = None
    return last_answer


def contains_long_number(text, length=10):
    return bool(re.search(r"\d{" + str(length + 1) + r",}", text))


def convert_to_question(sentence):
    # Lowercase the first character if it's uppercase
    if sentence[0].isupper():
        sentence = sentence[0].lower() + sentence[1:]

    # Locate 'is' and build the question
    parts = sentence.split()
    is_index = parts.index("is")  # Find the index of 'is'

    if is_index != -1:
        # Move 'is' to the beginning and rearrange the parts
        parts.pop(is_index)
        question = "Is " + " ".join(parts)

        # Replace the final period with a question mark
        if question.endswith("."):
            question = question[:-1] + "?"
        else:
            question += "?"
        return question

    return sentence  # Return the original if 'is' is not found


def check_sentence(sentence):
    sentence_lower = sentence.lower()

    positive_keywords = ["yes", "true", "correct", "right"]
    negative_keywords = ["no", "false", "incorrect", "wrong"]

    if any(keyword in sentence_lower for keyword in positive_keywords):
        return "True"
    elif any(keyword in sentence_lower for keyword in negative_keywords):
        return "False"
    else:
        return sentence

def run_gemma(prompt, tokenizer, model, max_new_tokens):
    chat = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_text = re.sub(r"user[\s\S]*?model", "", answer)
    return cleaned_text


def run_qwen(system_prompt, prompts, tokenizer, model):

    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": prompt},
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    # generated_ids = [
    #     output_ids[len(input_ids) :]
    #     for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    # answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # return answer
    texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    print(texts)
    # breakpoint()

    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    # breakpoint()

    # print(generated_ids.shape)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # breakpoint()

    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return answers


def run_mistral(system_prompt, prompts, tokenizer, model):
    texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    # print(texts)
    # breakpoint()

    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    # breakpoint()

    # print(generated_ids.shape)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # breakpoint()

    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return answers


def run_llama(system_prompt, prompts, tokenizer, model):
    texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    print(texts)
    # breakpoint()

    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    # breakpoint()

    # print(generated_ids.shape)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # breakpoint()

    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return answers


def run_llama_awq(system_prompt, prompt, tokenizer, model, sampling_params):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenized=False)

    outputs = model.generate([prompt], sampling_params)
    # breakpoint()
    return outputs


def load_model_and_tokenizer(model_name, model_path, quantized=None):

    if "Llama" in model_name:

        if quantized == "awq":
            model_path = "/common/users/km1558/huggingface/llama3-8b-awq"
            model = AutoAWQForCausalLM.from_pretrained(
                model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
            )
            model = model.to("cuda:0")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

            # model = AutoAWQForCausalLM.from_pretrained(
            #     model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
            # )
            # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # # Save quantized model
            # model.quantize(tokenizer, quant_config=quant_config)

            # model.save_quantized("/common/users/km1558/huggingface/llama3-8b-awq")
            # tokenizer.save_pretrained("/common/users/km1558/huggingface/llama3-8b-awq")

            # breakpoint()

            # Quantize
            # model.quantize(tokenizer, quant_config=quant_config)
            # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

            # # Create an LLM.
            # model = LLM(model=model_path, quantization="AWQ")
            # Generate texts from the prompts. The output is a list of RequestOutput objects
            # that contain the prompt, generated text, and other information.
            # outputs = llm.generate(prompts, sampling_params)

        elif quantized == "smooth_quant":
            # model = SparseAutoModelForCausalLM.from_pretrained(
            #     model_path,
            #     device_map="auto",
            #     torch_dtype=torch.float16,
            # )
            # tokenizer = AutoTokenizer.from_pretrained(model_path)

            # MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
            model = SparseAutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype="auto",
            )
            # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                        
            
            # model.save_pretrained("/common/users/km1558/huggingface/llama3-8b-gptq")
            # breakpoint()
            from datasets import load_dataset

            NUM_CALIBRATION_SAMPLES = 512
            MAX_SEQUENCE_LENGTH = 2048

            # Load and preprocess the dataset
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

            def preprocess(example):
                return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
            ds = ds.map(preprocess)

            def tokenize(sample):
                return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
            ds = ds.map(tokenize, remove_columns=ds.column_names)

            from llmcompressor.transformers import oneshot
            from llmcompressor.modifiers.quantization import GPTQModifier
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

            # Configure the quantization algorithms
            recipe = [
                SmoothQuantModifier(smoothing_strength=0.8),
                # GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
            ]

            # Apply quantization
            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=MAX_SEQUENCE_LENGTH,
                num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            )

            # Save the compressed model
            SAVE_DIR = "/common/users/km1558/huggingface/llama3-8b-smooth-quant"
            model.save_pretrained(SAVE_DIR, save_compressed=True)
            # tokenizer.save_pretrained(SAVE_DIR)

            # breakpoint()

        else:
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                cache_dir=None,
                device_map="auto",
                torch_dtype=torch.float16,
                use_flash_attention_2=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        tokenizer.pad_token = tokenizer.eos_token

    elif "Qwen2" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Quantize
        # model.quantize(tokenizer, quant_config=quant_config)
        # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        # # Create an LLM.
        # model = LLM(model=model_path, quantization="AWQ")
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        # outputs = llm.generate(prompts, sampling_params)

    elif "mistral" in model_name:
        # model = MistralForCausalLM.from_pretrained(
        #     model_path,
        #     cache_dir=None,
        #     device_map="auto",
        #     torch_dtype=torch.float16,
        #     use_flash_attention_2=True,
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # print("we use mistral")

        model = AutoAWQForCausalLM.from_pretrained(
            model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
        )
        model = model.to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }

        model = AutoAWQForCausalLM.from_pretrained(
            model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Save quantized model
        model.quantize(tokenizer, quant_config=quant_config)

        # model.save_quantized("/common/users/km1558/huggingface/qwen2-8b-awq")
        # tokenizer.save_pretrained("/common/users/km1558/huggingface/qwen2-8b-awq")

        breakpoint()

    elif "gemma" in model_name:
        if quantized in ["awq", "gptq"]:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        
        elif quantized == "smooth_quant":
            # MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
            model_path = "/common/users/km1558/huggingface/gemma2-9b-smooth-quant"
            model = SparseAutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype="auto",
            )
            # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                        
            
            # model.save_pretrained("/common/users/km1558/huggingface/llama3-8b-gptq")
            # breakpoint()
            # from datasets import load_dataset

            # NUM_CALIBRATION_SAMPLES = 512
            # MAX_SEQUENCE_LENGTH = 2048

            # # Load and preprocess the dataset
            # ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            # ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

            # def preprocess(example):
            #     return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
            # ds = ds.map(preprocess)

            # def tokenize(sample):
            #     return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
            # ds = ds.map(tokenize, remove_columns=ds.column_names)

            # from llmcompressor.transformers import oneshot
            # from llmcompressor.modifiers.quantization import GPTQModifier
            # from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

            # # Configure the quantization algorithms
            # recipe = [
            #     SmoothQuantModifier(smoothing_strength=0.8),
            #     # GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
            # ]

            # # Apply quantization
            # oneshot(
            #     model=model,
            #     dataset=ds,
            #     recipe=recipe,
            #     max_seq_length=MAX_SEQUENCE_LENGTH,
            #     num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            # )

            # # Save the compressed model
            # SAVE_DIR = "/common/users/km1558/huggingface/gemma2-9b-smooth-quant"
            # model.save_pretrained(SAVE_DIR, save_compressed=True)
            # breakpoint()
        
        else:
            model = Gemma2ForCausalLM.from_pretrained(
                model_path,
                cache_dir=None,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def run_city(model, tokenizer, dataset, batch_size):
    pass


def run_math(model, tokenizer, dataset, batch_size):
    pass


def run_imdb(model, tokenizer, dataset, batch_size):
    pass


def run_tof(model, tokenizer, dataset, batch_size):
    pass


def run_long(model, tokenizer, dataset, batch_size):
    pass


def main(args):
    model_name = args.model_name
    model_path = model_name
    quantized = args.quantized
    batch_size = args.batch_size

    model, tokenizer = load_model_and_tokenizer(model_name, model_path, quantized)

    model.eval()

    if args.pattern == "city":

        file = "datasets/cities.csv"
        data = pd.read_csv(file)
        update_path(f"saved_attn_scores/{model_name}")
        num_correct = 0
        # data_list = zip(data["statement"], data["label"])
        k = 0
        
        # breakpoint()
        data_size = len(data)
        
        batch_idx = data_size // args.batch_size + 1
        
        for idx in range(batch_idx):
            items = data[
                idx * args.batch_size : min((idx + 1) * args.batch_size, len(data))
            ]
            
            statements = data["statement"][idx * args.batch_size : min((idx + 1) * args.batch_size, len(data))].tolist()
            
            labels = data["label"][idx * args.batch_size : min((idx + 1) * args.batch_size, len(data))].tolist()
            
            # questions = []
            # labels = []
            # for item in items:
            #     statement, label = item[0], item[1]
            #     question = convert_to_question(statement)
            # # print(question)
            #     questions.append(question)
            #     labels.append(label)
            
            questions = [convert_to_question(statement) for statement in statements]

            if "Qwen" in model_path:
                system_prompt = "You are a helpful geography expert who can help me. Answer ''Yes'' or ''No''."
                pred_answers = run_qwen(system_prompt, questions, tokenizer, model)

            elif "gemma" in model_path:
                max_new_tokens = 128
                prompt = question
                answer = run_gemma(prompt, tokenizer, model, max_new_tokens)
                print(answer)

            elif "Llama" in model_path:
                system_prompt = "You are a helpful geography expert who can help me. Answer ''Yes'' or ''No''."

                # if quantized:
                #     answer = run_llama_awq(system_prompt, question, tokenizer,model,sampling_params)
                # else:
                pred_answers = run_llama(system_prompt, questions, tokenizer, model)

            elif "mistral" in model_path:
                system_prompt = "You are a helpful geography expert who can help me. Answer ''Yes'' or ''No''."
                answer = run_mistral(system_prompt, question, tokenizer, model)
                
            print("Original pred answers: ", pred_answers)
            

            pred_answers = [check_sentence(pred_answer) for pred_answer in pred_answers]
            
            print("Pred answers: ", pred_answers)
            
            print("GT answers: ", labels)

            for i in range(batch_size):
                pred_answer = pred_answers[i]
                label = labels[i]
                
                gt = "True" if label == 1 else "False"

                if gt == pred_answer:
                    print("Correct")
                    num_correct += 1
                else:
                    print("Incorrect")
                k += 1

            print(f"Current accuracy: {num_correct}/{k}")
            
            if k == 1000:
                break

        print(f"Accuracy: {num_correct}/1000")

    elif args.pattern == "math":
        data = load_dataset("gsm8k", "main")["test"]
        update_path(f"saved_attn_scores/{model_name}")
        num_samples = len(data)

        num_correct = 0
        num = 0

        batch_idx = len(data) // args.batch_size + 1
        for idx in range(batch_idx):
            items = data[
                idx * args.batch_size : min((idx + 1) * args.batch_size, len(data))
            ]
            # print(items)
            # breakpoint()
            # question = i['question']
            # answers = i['answer']
            questions = items["question"]
            answers = items["answer"]

            if "Qwen" in model_path:
                system_prompt = "You are a helpful math expert who can help me. Answer the question and put the final answer at the end of the sentence."
                pred_answers = run_qwen(system_prompt, questions, tokenizer, model)

            if "gemma" in model_path:
                max_new_tokens = 500
                prompt = question
                answer = run_gemma(system_prompt, questions, tokenizer, model)

            elif "Llama" in model_path:
                max_new_tokens = 500
                system_prompt = "You are a helpful math expert who can help me. Answer the question and put the final answer at the end of the sentence."
                pred_answers = run_llama(system_prompt, questions, tokenizer, model)

            elif "mistral" in model_path:
                max_new_tokens = 500
                system_prompt = "You are a helpful math expert who can help me. Answer the question and put the final answer at the end of the sentence."
                pred_answers = run_mistral(system_prompt, questions, tokenizer, model)

            # else:
            #     messages = [
            #         {
            #             "role": "system",
            #             "content": "You are a helpful math expert who can help me. Answer the question and put the final answer at the end of the sentence.",
            #         },
            #         {"role": "user", "content": question},
            #     ]
            #     input_ids = tokenizer.apply_chat_template(
            #         messages, add_generation_prompt=True, return_tensors="pt"
            #     ).to(model.device)
            #     terminators = [
            #         tokenizer.eos_token_id,
            #         tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            #     ]
            #     tokens = model.generate(
            #         input_ids, max_length=400, eos_token_id=terminators
            #     )
            #     answer = tokenizer.decode(
            #         tokens[0][input_ids.shape[1] :], skip_special_tokens=True
            #     )

            # print(question)

            # print("---------------------------")
            print("Original pred answers: ", pred_answers)
            # breakpoint()
            pred_answers = [
                extract_last_answer(pred_answer) for pred_answer in pred_answers
            ]
            print("Pred answers: ", pred_answers)

            answers = [extract_last_answer(answer) for answer in answers]
            print("gt: ", answers)

            # breakpoint()

            for i in range(batch_size):
                pred_answer = pred_answers[i]
                answer = answers[i]
                if pred_answer is not None and answer is not None:
                    if float(pred_answer) == float(answer):
                        print("Correct")
                        num_correct += 1
                    else:
                        print("Incorrect")
                else:
                    print("Error: One of the values is None")

                num = num + 1

            if num == 1000:
                break

            print(
                f"Current accuracy: {num_correct / num} {num_correct}/{num}",
            )

        print(f"Accuracy: {num_correct} / 1000")

    elif args.pattern == "imdb":
        data = load_dataset("stanfordnlp/imdb")
        update_path(f"saved_attn_scores/{model_name}")

        num_correct = 0
        num = 0
        for i in data["train"]:
            question = i["text"]
            answers = i["label"]
            if "gemma" in model_path:
                prompt = question + " Judge the statement Negative or Positive"
                max_new_tokens = 300
                answer = run_gemma(prompt, tokenizer, model, max_new_tokens)

            print("---------------------------")
            print("answer: ", answer)

            if answers == 0:
                gt = "Negative"
            else:
                gt = "Positive"
            print("gt: ", gt)

            if gt.lower() in answer.lower():
                print("Correct")
                num_correct += 1
            else:
                print("Incorrect")
            num = num + 1
            if num == 1000:
                break

        print(f"Accuracy: {num_correct}/1000")

    elif args.pattern == "tof":
        print("6666666666666666")
        file = "datasets/true_false_dataset.json"
        with open(file, "r") as file:
            data = json.load(file)
        questions_data = data["data"]

        update_path(f"saved_attn_scores/{model_name}")
        num_correct = 0

        k = 0

        for i in questions_data:

            question = i["question"]
            label = i["answer"]
            print(question)
            if "Qwen" in model_path:
                prompt = question
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful expert who can help me. Answer the question and put the final answer at the end of the sentence.",
                    },
                    {"role": "user", "content": prompt},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                generated_ids = model.generate(**model_inputs, max_new_tokens=512)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]
                answer = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

            else:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a knowledge expert who can help me. Please answer ''Yes'' or ''No''.",
                    },
                    {"role": "user", "content": question},
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)

                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                tokens = model.generate(
                    input_ids, max_length=100, eos_token_id=terminators
                )
                answer = tokenizer.decode(
                    tokens[0][input_ids.shape[1] :], skip_special_tokens=True
                )

            print("---------------------------")

            if label:
                gt = "Yes"
            else:
                gt = "No"

            print(answer)
            print("gt: ", gt)

            if gt in answer:
                print("Correct")
                num_correct += 1
            else:
                print("Incorrect")

            print("---------------------------")
            k += 1
            if k == 200:
                break

        print(f"Accuracy: {num_correct}/200")

    elif args.pattern == "long":
        file = "/common/home/mj939/atten_exp/synthetic_tasks/passkey_retrieval/LlamaTokenizer_1024_100_500_len_48/test.jsonl"
        data = pd.read_json(file, lines=True)

        update_path(f"saved_attn_scores/{model_name}")
        num_correct = 0
        num_correct_retrive = 0

        list = data["input"]
        k = 0
        for statement in list:
            question = statement

            if "Qwen" in model_path:
                prompt = question + " What is the passkey?"
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful expert who can help me. Answer the question and put the final answer at the end of the sentence.",
                    },
                    {"role": "user", "content": prompt},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                generated_ids = model.generate(**model_inputs, max_new_tokens=512)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]
                answer = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

            elif "gemma" in model_path:
                prompt = question
                max_new_tokens = 2000
                answer = run_gemma(prompt, tokenizer, model, max_new_tokens)

            print("---------------------------")
            print("answer", answer)

            numbers = re.findall(r"\d+", question)

            label = str(numbers[0])

            print("groudtruth", label)

            if label in answer:
                print("Correct")
                num_correct += 1
            else:
                print("Incorrect")

            k += 1
            if k == 50:
                break

        print(f"Accuracy: {num_correct} / 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--context_len", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--log_scale_base", type=int, required=True)
    parser.add_argument("--pattern", type=str, required=True, help="dataset")
    parser.add_argument("--quantized", choices=[None, "awq", "smooth_quant", "gptq"])
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
