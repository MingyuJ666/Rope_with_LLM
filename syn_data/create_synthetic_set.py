# Do nothing, but create dataset for synthetic tasks
import argparse
import datasets
import gc
import sys
import os
import difflib
import torch
import json
import warnings
import transformers
import pickle
import re
from pathlib import Path
from transformers import AutoTokenizer, LlamaTokenizer
from tqdm import tqdm
import generate_syn_data as GSDATA



def passkey_retrieval_accuracy(predictions, references):
    # This function is depreceated, please refer the main task file
    def extract_answer(text):
        answer = re.findall(r'\d+', text)
        if len(answer) == 0:
            return 0
        else:
            return int(answer[0])

    predictions = [extract_answer(pred) for pred in predictions]
    references = [extract_answer(ref) for ref in references]
    print(predictions)
    print(references)

    accuracy = sum(pred == ref for pred, ref in zip(predictions, references)) / len(predictions)
    return {
        "accuracy": accuracy
    }

def duplicate_string_accuracy(predictions, references):
    def longest_substring_matching(s1, s2):
        # Initialize SequenceMatcher with the two strings
        seq_matcher = difflib.SequenceMatcher(None, s1, s2)
        
        # Find the longest matching block
        match = seq_matcher.find_longest_match(0, len(s1), 0, len(s2))
        
        # Extract the longest common substring using the match's start and end indices
        return s1[match.a: match.a + match.size]

    
    cover_ratio = []
    for p, r in zip(predictions, references):
        sub_string = longest_substring_matching(p, r)
        print(sub_string)
        cover_ratio.append(len(sub_string)/len(r))

    avg_ratio = sum(cover_ratio)/len(cover_ratio)
    return {
        "Repeat Ratio": avg_ratio
    }
    

def get_passkey_retrieval(dataset_dir, dataset_group, structured_prompt,
                          max_data_num, start_data_from, ):
    
    system_prompt = "<s>[INST] <<SYS>> \n " \
        "You are a helpful, concise and honest assistant. <</SYS>> \n "
    concise_prompt = "Do not provide any explanation. [/INST] \n"


    filename = dataset_dir / dataset_group / "test.jsonl"

    with open(filename, "r") as f:
        dataset = list(map(json.loads, tqdm(f.readlines())))

    if start_data_from is not None:
        dataset = dataset[start_data_from:]
    if max_data_num is not None:
        dataset = dataset[:max_data_num]

    for _id, _datum in enumerate(dataset):

        _datum["prompt"] = _datum["input"] + \
            " What is the pass key? The pass key is "
        if structured_prompt:
            _datum["prompt"] = system_prompt + _datum["input"] + concise_prompt

        _datum["output"] = _datum["target"]
        _datum["id"] = _id
    return {
        "data": dataset,
        "metadata": {
            "metric_func": passkey_retrieval_accuracy,
            "recommended_length": None,
        }
    }

def get_duplicate_string(dataset_dir, dataset_group, structured_prompt, max_data_num, start_data_from, ):
    system_prompt = "<s>[INST] <<SYS>> \n " \
    "You are a helpful, concise and honest assistant. <</SYS>> \n "
    concise_prompt = "Do not provide any explanation. [/INST] \n"

    filename = dataset_dir / dataset_group / "test.jsonl"

    with open(filename, "r") as f:
        dataset = list(map(json.loads, tqdm(f.readlines())))

    if start_data_from is not None:
        dataset = dataset[start_data_from:]
    if max_data_num is not None:
        dataset = dataset[:max_data_num]

    for _id, _datum in enumerate(dataset):

        _datum["prompt"] = _datum["input"] + \
            "\n Begin to repeat the string exactly as you see it before, without any errors or omissions."
        if structured_prompt:
            _datum["prompt"] = system_prompt + _datum["input"] + concise_prompt

        _datum["output"] = _datum["target"]
        _datum["id"] = _id
    return {
        "data": dataset,
        "metadata": {
            "metric_func": duplicate_string_accuracy,
            "recommended_length": None,
        }
    }




def load_data(dataset_name, dataset_path, dataset_group, split, structured_prompt,max_data_num=None, start_data_from=None, tokenizer=None, args=None):
    dataset_folder = Path(dataset_path)
    dataset_path = dataset_folder / dataset_name 


    if dataset_name == "passkey_retrieval":
        if args.insert_position > 100:
            exit('skip for now')
        dataset_group = dataset_group + f"_{args.seq_length}_{args.insert_position}_{args.interval}_len_{args.passkey_length}"
        if (dataset_path/dataset_group).exists():
            dataset = get_passkey_retrieval(dataset_path, dataset_group, structured_prompt, max_data_num, start_data_from)
        else:
            print(f"Dataset {dataset_path/dataset_group} not found, create it now.")
            GSDATA.create_passkey(tokenizer, dataset_path, args.seq_length, args.num_gen_example, insert_position=args.insert_position, pos_interval=args.interval, passkey_length=args.passkey_length)
            dataset = get_passkey_retrieval(dataset_path, dataset_group, structured_prompt, max_data_num, start_data_from)

    elif dataset_name == "duplicate_string":
        pass
    elif dataset_name == "missing_dupilication":
        pass
    elif dataset_name == "parity_check":
        pass
    else:
        raise NotImplementedError()

    return dataset



def main(args):
    

    model_list = [x[0] for x in args.model_list]
    print(model_list)
    
    

    for model_path in model_list:
        # We'd better generate all datasets first in another run....
        if 'llama' in model_path:
            tokenizer_for_dataset = LlamaTokenizer.from_pretrained(model_path, token="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")
        else:
            tokenizer_for_dataset = AutoTokenizer.from_pretrained(model_path, token="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")

        

        if tokenizer_for_dataset.pad_token is None:
            tokenizer_for_dataset.pad_token = tokenizer_for_dataset.eos_token

        args.dataset_group = f"{tokenizer_for_dataset.__class__.__name__}"
        print(f"Dataset group: {args.dataset_group}")
        
       
            
        loaded_dataset = load_data(args.dataset, 
                                args.dataset_folder, 
                                args.dataset_group,
                                args.split, 
                                args.structured_prompt,
                                args.max_data_num, 
                                args.start_data_from, 
                                tokenizer=tokenizer_for_dataset, 
                                args=args)



if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument("--dynamic-linear", action="store_true")
    parser.add_argument("--dynamic-ntk", type=float)
    parser.add_argument("--dynamic-part-ntk", action="store_true")
    parser.add_argument("--dynamic-yarn", action="store_true")
    parser.add_argument("--ntk", type=float)
    parser.add_argument("--part-ntk", type=float)
    parser.add_argument("--linear", type=float)
    parser.add_argument("--yarn", type=float)
    parser.add_argument("--rerope", type=float)
    parser.add_argument("--factor", type=float)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--gpt-neox-max-length", type=int)
    parser.add_argument("--adapter", type=str)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--original-max-position-embeddings", type=int)
    parser.add_argument("--custom-model", action="store_true")
    parser.add_argument("--custom-model-together", action="store_true")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--no-use-cache", action="store_true")


    # Data/Task settings
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--dataset_group", type=str)
    parser.add_argument("--dataset_folder", type=str, default="/data/HyJ/synthetic_tasks")
    parser.add_argument("--max_data_num", type=int)
    parser.add_argument("--start_data_from", type=int, default=0)


    # Experiment settings
    parser.add_argument("--structured_prompt", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./synthetic_task_logs")
    parser.add_argument("-m", "--model_list", action="append", nargs="+")
    parser.add_argument("--max_generation_length", type=int, default=128)
    parser.add_argument("--min_generation_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("--group_size_1", type=int)
    parser.add_argument("--group_size_2", type=int)
    parser.add_argument("--group_weight", type=float)
    
    parser.add_argument("--suppress_tokens", type=int, default=[], nargs="*", help="tokens to suppress when generating(logit is set to 0)")
    parser.add_argument("--do_sample", action="store_true", help='if true, use multinomial sampling instead of greedy decoding')
    
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0) # for deepspeed excution
    parser.add_argument("--ctx_step", type=int)
    parser.add_argument("--sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--output_attentions", action="store_true")
    parser.add_argument("--log_scale", action="store_true")
    parser.add_argument("--passkey_length", type=int, default=5)

    args = GSDATA.add_dataset_creation_args(parser).parse_args()
    main(args)