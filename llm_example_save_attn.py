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
import torch.nn as nn
import json
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM
from modeling_mistral import MistralForCausalLM
from modeling_qwen2 import Qwen2ForCausalLM
from modeling_gemma2 import Gemma2ForCausalLM
from modeling_opt import OPTForCausalLM
from modeling_jamba import JambaForCausalLM
from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
import re
from datasets import load_dataset
import csv
import numpy as np
from run_datasets import run_cities, run_aqua, run_math, run_imdb, run_sports, run_art, run_tech, run_cele, run_gemma, run_llama, run_qwen, run_opt, run_mistral, run_knowledge
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def run_jamba(prompt, tokenizer, model):
    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Ensure the model is on the correct device

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    outputs = model.generate(input_ids, max_new_tokens=216)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response


def run_qwen2vl(processor, model):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/common/home/mj939/atten_exp/transformer.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    

def main(args):
    print(args)

    model_name = args.model_name
    model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if 'llama' in model_name:
        model = LlamaForCausalLM.from_pretrained(model_path, 
                                                cache_dir=None, 
                                                device_map='auto', 
                                                torch_dtype=torch.float16, 
                                                use_flash_attention_2=True )
        
    elif 'Qwen/Qwen2.5-7B-Instruct' in model_name:
        model = Qwen2ForCausalLM.from_pretrained(model_path, 
                                                    cache_dir=None, 
                                                    device_map='auto', 
                                                    torch_dtype=torch.float16, 
                                                    use_flash_attention_2=True)
       
    elif 'gemma' in model_name:
        model = Gemma2ForCausalLM.from_pretrained(model_path, 
                                                    cache_dir=None, 
                                                    device_map='auto', 
                                                    torch_dtype=torch.float16,)
    elif 'opt' in model_name:
    
        model = OPTForCausalLM.from_pretrained(model_path, 
                                                    cache_dir=None, 
                                                    device_map='auto', 
                                                    torch_dtype=torch.float16, 
                                                    attn_implementation="flash_attention_2")
        
    elif 'Jamba' in model_name:
        model = JambaForCausalLM.from_pretrained(model_path, 
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             attn_implementation="flash_attention_2",
                                             use_mamba_kernels=False)
        
    elif 'mistral' in model_name:
        model = MistralForCausalLM.from_pretrained(model_path, 
                                                    cache_dir=None, 
                                                    device_map='auto', 
                                                    torch_dtype=torch.float16, 
                                                    use_flash_attention_2=True)
        
    elif 'Qwen2-VL' in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, 
                                                                torch_dtype=torch.float16,
                                                                device_map="auto",
                                                                use_flash_attention_2=True)
        
        processor = AutoProcessor.from_pretrained(model_path)
        



    model.eval()
    if args.pattern == "city":
        file = 'datasets/cities.csv'
        data = pd.read_csv(file)
        update_path(f"saved_attn_scores/{model_name}")
        accuracies = []
        for i in range(3):
            accuracy = run_cities(model_path, tokenizer, model, data)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies) * 100
        std_deviation = np.std(accuracies) * 100

        fluctuation = f"({std_deviation:+.1f}%)"  # Display fluctuation with + or - sign
        print(f"Accuracy: {mean_accuracy:.1f}% {fluctuation}")
        
    elif args.pattern == "knowledge_conflict":
        file = 'datasets/cities.csv'
        data = pd.read_csv(file)
        update_path(f"saved_attn_scores/{model_name}")
        accuracy = run_knowledge(model_path, tokenizer, model, data)


        
    
    elif args.pattern == "save_attn":
        data = load_dataset("deepmind/aqua_rat", "raw")['test']        
        update_path(f"saved_attn_scores/{model_name}")
        num_correct = 0
        
        for i in data:

            question = '1+1=?'
            if 'gemma' in model_path:
                    max_new_tokens = 1000
                    prompt =  question 
                    print(prompt)
                    answer = run_gemma(prompt,tokenizer,model,max_new_tokens)
                    
            elif 'llama' in model_path:
                    max_new_tokens = 1200
                    prompt =  question + ' Choose the correct answer from the options: '
                    print(prompt)
                    system_prompt = "You are a helpful math expert who can help me. Put the final option and answer at the end of the sentence. Do not show other incorrect options."
                    answer = run_llama(system_prompt, prompt, tokenizer,model,max_new_tokens)
                    
            elif 'opt' in model_path:
                    print(model)
                    exit()
                    
                    prompt =  question 
                    print(prompt)
                    answer = run_opt(prompt,tokenizer,model)
                    print(answer)
            
            elif 'Jamba' in model_path:
                    prompt =  question 
                    print(prompt)
                    answer = run_jamba(prompt,tokenizer,model)
                    print(answer)
                    
            elif 'Qwen/Qwen2.5-7B-Instruct' in model_path:
                    prompt =  question 
                    max_new_tokens = 1200
                    system_prompt = "You are a helpful math expert who can help me. Put the final option and answer at the end of the sentence. Do not show other incorrect options."
                    print(prompt)
                    answer = run_qwen(system_prompt, prompt, tokenizer,model,max_new_tokens)
                    print(answer)
                    
            elif 'mistral' in model_path:
                    prompt =  question
                    max_new_tokens = 1000
                    system_prompt = "You are a helpful math expert who can help me. Put the final option and answer at the end of the sentence. Do not show other incorrect options."
                    answer = run_mistral(system_prompt, prompt, tokenizer,model,max_new_tokens)
            
            elif 'Qwen2-VL' in model_path:
                   run_qwen2vl(processor, model)      
            exit()
                    
           
                
                
 
             
    
    elif args.pattern == "aqua": 
       
        data = load_dataset("deepmind/aqua_rat", "raw")['test']
        update_path(f"saved_attn_scores/{model_name}")
        accuracies = []
        for i in range(args.round):
            accuracy, ppl = run_aqua(model_path, tokenizer, model, data)
            accuracies.append(accuracy)

        # Calculate the average and standard deviation
        mean_accuracy = np.mean(accuracies) * 100
        std_deviation = np.std(accuracies) * 100

        # Print the results in the desired format
        fluctuation = f"({std_deviation:+.2f}%)"  
        print(f"Accuracy: {mean_accuracy:.1f}% {fluctuation}")
        
                        

    
    elif args.pattern == "math":
        data = load_dataset('gsm8k', 'main')['test']
        update_path(f"saved_attn_scores/{model_name}")
        accuracies = []
        for i in range(args.round):
            accuracy,ppl = run_math(model_path, tokenizer, model, data)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies) * 100
        std_deviation = np.std(accuracies) * 100
        
        fluctuation = f"({std_deviation:+.2f}%)"  # Display fluctuation with + or - sign
        print(accuracies)
        print(f"Accuracy: {mean_accuracy:.1f}% {fluctuation}")
       
        
    
    elif args.pattern == "imdb":
        data = load_dataset("stanfordnlp/imdb")
        update_path(f"saved_attn_scores/{model_name}")
        accuracies = []
        ppl_compute = args.ppl_compute
        for i in range(args.round):
            accuracy = run_imdb(model_path, tokenizer, model, data, ppl_compute)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies) * 100
        std_deviation = np.std(accuracies) * 100
        
        fluctuation = f"({std_deviation:+.2f}%)"  # Display fluctuation with + or - sign
        print(accuracies)
        print(f"Accuracy: {mean_accuracy:.1f}% {fluctuation}")
        
        
    
        
    elif args.pattern == "sports":
        file = 'datasets/sports_qa.json'
        with open(file, "r") as file:
            data = json.load(file)
        questions_data = data["data"]  
        update_path(f"saved_attn_scores/{model_name}")
        accuracies = []
        for i in range(3):
            accuracy = run_sports(model_path, tokenizer, model, questions_data)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies) * 100
        std_deviation = np.std(accuracies) * 100
        
        fluctuation = f"({std_deviation:+.2f}%)"  
        print(f"Accuracy: {mean_accuracy:.1f}% {fluctuation}")
        

        
        
        
    elif args.pattern == "art":
        file = 'datasets/art_qa.json'
        with open(file, "r") as file:
            data = json.load(file)
        questions_data = data["data"]  
        update_path(f"saved_attn_scores/{model_name}")
        accuracies = []
        for i in range(3):
            accuracy = run_art(model_path, tokenizer, model, questions_data)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies) * 100
        std_deviation = np.std(accuracies) * 100
        
        fluctuation = f"({std_deviation:+.2f}%)"  
        print(f"Accuracy: {mean_accuracy:.1f}% {fluctuation}")
       
        
       
        
    elif args.pattern == "tech":
        file = 'datasets/technology_qa.json'
        with open(file, "r") as file:
            data = json.load(file)
        questions_data = data["data"]  

        update_path(f"saved_attn_scores/{model_name}")
        accuracies = []
        for i in range(3):
            accuracy = run_tech(model_path, tokenizer, model, questions_data)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies) * 100
        std_deviation = np.std(accuracies) * 100
        
        fluctuation = f"({std_deviation:+.2f}%)"
        print(f"Accuracy: {mean_accuracy:.1f}% {fluctuation}")
        
    elif args.pattern == "cele":
        file = 'datasets/celebrity_qa.json'
        with open(file, "r") as file:
            data = json.load(file)
        questions_data = data["data"]  
        update_path(f"saved_attn_scores/{model_name}")
        accuracies = []
        for i in range(3):
            accuracy = run_cele(model_path, tokenizer, model, questions_data)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies) * 100
        std_deviation = np.std(accuracies) * 100
        
        fluctuation = f"({std_deviation:+.2f}%)"  
        print(f"Accuracy: {mean_accuracy:.1f}% {fluctuation}")
        
   
           
        
    elif args.pattern == "long": 
        file = 'synthetic_tasks/passkey_retrieval/LlamaTokenizer_256_100_500_len_12/test.jsonl'
        data = pd.read_json(file,lines=True)
        length = len(data)
        
        update_path(f"saved_attn_scores/{model_name}")
        num_correct = 0
        
        list = data['input']
        k = 0
        for statement in list:
            question = statement
            
            if 'Qwen' in model_path:
                prompt = question + ' What is the passkey?'
                system_prompt = "You are a helpful expert who can help me. Answer the question and put the final answer at the end of the sentence."
                max_new_tokens = 2000
                answer = run_qwen(system_prompt, prompt, tokenizer,model,max_new_tokens)
                
            elif 'gemma' in model_path:
                prompt = question
                max_new_tokens = 2000
                answer = run_gemma(prompt,tokenizer,model,max_new_tokens)
                
            elif 'llama' in model_path:
                system_prompt = "You are a helpful expert who can help me. Answer the question and put the final answer at the end of the sentence."
                max_new_tokens = 2000
                answer = run_llama(system_prompt, question, tokenizer,model,max_new_tokens)
                
                
            print('---------------------------')
            print('answer',answer) 
            
            numbers = re.findall(r'\d+', question)
           
            label = str(numbers[0])
            
            print('groudtruth', label)
            
           
            if label in answer:
                print("Correct")
                num_correct += 1
            else:
                print("Incorrect")
        
                
        print(f"Accuracy: {num_correct}/{length}")
       
        
        
            
            
        
       
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--ppl_compute", type=bool, required=False, default=True)
    args = parser.parse_args()
    main(args)
