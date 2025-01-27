import os, re
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import nltk
from nltk.util import ngrams


def convert_to_question(sentence):
    # Lowercase the first character if it's uppercase
    if sentence[0].isupper():
        sentence = sentence[0].lower() + sentence[1:]
    
    # Locate 'is' and build the question
    parts = sentence.split()
    is_index = parts.index('is')  # Find the index of 'is'
    
    if is_index != -1:
        # Move 'is' to the beginning and rearrange the parts
        parts.pop(is_index)
        question = 'Is ' + ' '.join(parts)
        
        # Replace the final period with a question mark
        if question.endswith('.'):
            question = question[:-1] + '?'
        else:
            question += '?'
        return question
    
    return sentence  # Return the original if 'is' is not found


def extract_last_answer(text):
    try:
        
        # matches = re.findall(r'\d+(?:,\d+)*', text)  
        matches = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        matches = [float(match.replace(",", "")) for match in matches]

        # matches = [int(match.replace(",", "")) for match in matches]

        last_answer = matches[-1] if matches else None
    except ValueError as e:
        print(f"Error: {e}")
        last_answer = None
    return last_answer


def run_gemma(prompt,tokenizer,model,max_new_tokens):
    chat = [
    {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_text = re.sub(r"user[\s\S]*?model", "", answer)

    
    return cleaned_text


def run_opt(prompt,tokenizer,model): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
    tokenizer.batch_decode(generated_ids)[0]
    return tokenizer.batch_decode(generated_ids)[0]
  
def run_qwen(system_prompt, prompt,tokenizer,model,max_new_tokens):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer

def run_mistral(system_prompt, prompt,tokenizer,model,max):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer

    
    
def run_llama(system_prompt,prompt,tokenizer,model,max_token):   
    messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    tokens = model.generate(input_ids ,max_length=max_token, eos_token_id=terminators)
    answer= tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True)
   
    return answer


def calculate_perplexity(answer, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        answer_tokens = tokenizer(answer, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    
        if answer_tokens.size(1) == 0:
            print("Generated answer is empty. Perplexity is undefined (infinite).")
            return float('inf')
        
        
        lm_logits = model(answer_tokens).logits  # shape: [batch_size, seq_len, vocab_size]
        
        shift_logits = lm_logits[:, :-1, :].contiguous()  # shape: [batch_size, seq_len-1, vocab_size]
        shift_labels = answer_tokens[:, 1:].contiguous()  # shape: [batch_size, seq_len-1]
        
        loss_fct = nn.CrossEntropyLoss(reduction='sum')
        
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # shape: [batch_size * (seq_len-1), vocab_size]
            shift_labels.view(-1)  # shape: [batch_size * (seq_len-1)]
        )
        
        num_tokens = shift_labels.numel()
        perplexity = torch.exp(loss / num_tokens).item()
        
    return perplexity


def calculate_ngram_diversity(texts, n=2):
   
    all_ngrams = []
    
    for text in texts:
        words = nltk.word_tokenize(text.lower())  # 将文本分割为单词并转换为小写
        ngram_list = list(ngrams(words, n))  # 生成 n-grams
        all_ngrams.extend(ngram_list)
    
    total_ngrams = len(all_ngrams)
    unique_ngrams = len(set(all_ngrams))

    diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    return diversity

def check_sentence(sentence):
    sentence_lower = sentence.lower()
    
    positive_keywords = ['yes', 'true', 'correct', 'right','ok']
    negative_keywords = ['no', 'false', 'incorrect', 'wrong']
    
    if any(keyword in sentence_lower for keyword in positive_keywords):
        return 'True'
    elif any(keyword in sentence_lower for keyword in negative_keywords):
        return 'False'
    else:
        return sentence




def run_cities(model_path, tokenizer, model, data):
    num_correct = 0
    list = zip(data['statement'], data['label'])
    k = 0
        
    for statement, label in list:
        question = convert_to_question(statement)
        print(question)
        
        if 'Qwen' in model_path:
            system_prompt = "You are a helpful geography expert who can help me. Answer ''Yes'' or ''No''."
            answer = run_qwen(system_prompt, question, tokenizer, model, 128)
            
        elif 'gemma' in model_path:
            max_new_tokens = 128
            prompt = question
            answer = run_gemma(prompt, tokenizer, model, max_new_tokens)
            print(answer)
            
        elif 'llama' in model_path:
            max_new_tokens = 300
            system_prompt = "You are a helpful geography expert who can help me. Answer ''Yes'' or ''No''."
            answer = run_llama(system_prompt, question, tokenizer, model, max_new_tokens)
            print(answer)
            
        elif 'mistral' in model_path:
            system_prompt = "You are a helpful geography expert who can help me. Answer ''Yes'' or ''No''."
            answer = run_mistral(system_prompt, question, tokenizer, model)
            
        answer = check_sentence(answer)
        
        
        gt = "True" if label == 1 else "False"
        
        if gt == answer:
            print("Correct")
            num_correct += 1
        else:
            print("Incorrect")
        
        k += 1
        if k == 1000:
            break
    
    accuracy = num_correct / 1000
    print('Accuracy:', accuracy)
    return accuracy   




def run_knowledge(model_path, tokenizer, model, data):
    num_correct = 0
    list = zip(data['statement'], data['label'])
    k = 0
        
    for statement, label in list:
        question = convert_to_question(statement)
        index_of = statement.index('of')
        index_is = statement.index('is')

        # Extract the word between 'of' and 'is'
        word_between = statement[index_of + 2:index_is]
        previus_prompt = f'You should Know the new geography knowledge: {word_between} is a new city in The USA.'
        previus_prompt1 = 'You should Know the new geography knowledge: ShiJiaZhuang is a new city in The USA.'
        question = previus_prompt1 + ' ' + question
        print("__________________________")
        print(question)
        
        if 'Qwen' in model_path:
            system_prompt = "You are a helpful geography expert who can help me. Answer ''Yes'' or ''No''."
            answer = run_qwen(system_prompt, question, tokenizer, model, 128)
            
            
        elif 'llama' in model_path:
            max_new_tokens = 300
            system_prompt = "You are a helpful geography expert who can help me. Answer ''Yes'' or ''No''."
            answer = run_llama(system_prompt, question, tokenizer, model, max_new_tokens)
            
            
        answer = check_sentence(answer)
        print("answer:", answer)
        
        
        gt = "True" if label == 1 else "False"
        
        if gt == answer:
            print("Correct")
            num_correct += 1
        else:
            print("Incorrect")
        print("__________________________")
        
        k += 1
        if k == 100:
            break
    
    accuracy = num_correct / 100
    print('Accuracy:', accuracy)
    return accuracy   





def run_aqua(model_path, tokenizer, model, data):
    # data = data.select(range(3))
    num_correct = 0
    num_samples = len(data)
    k = 0
    sum_ppl = []
    diversity = []
        
    for i in data:
        
        question = i['question']
        gt = i['correct']
        options = i['options']
        correct_answer = next(option for option in options if option.startswith(gt))
        correct_answer = correct_answer[2:]
        
        print(question)
        if 'gemma' in model_path:
            max_new_tokens = 1000
            prompt = question + ' Choose the correct answer from the options: ' + ' '.join(options)
            answer = run_gemma(prompt, tokenizer, model, max_new_tokens)
            
        elif 'llama' in model_path:
            max_new_tokens = 1200
            prompt = question + ' Choose the correct answer from the options: ' + ' '.join(options)
            system_prompt = "You are a helpful math expert who can help me. Put the final option and answer at the end of the sentence. Do not show other incorrect options."
            answer = run_llama(system_prompt, prompt, tokenizer, model, max_new_tokens)
           
            
        elif 'Qwen' in model_path:
            max_new_tokens = 1500
            prompt = question + ' Choose the correct answer from the options: ' + ' '.join(options)
            system_prompt = "You are a helpful math expert who can help me. Answer the question and put the final answer at the end of the sentence."
            answer = run_qwen(system_prompt, question, tokenizer, model, max_new_tokens)
            
        ppl = calculate_perplexity(answer, model, tokenizer)
        print('ppl:', ppl)
        
        sum_ppl.append(ppl)
        diversity.append(calculate_ngram_diversity([answer], n=2))
        
        print('Answer:', answer)
        print('Ground Truth:', correct_answer)
        
        if correct_answer in answer:
            print("Correct")
            num_correct += 1
        else:
            print("Incorrect")
        
        print('---------------------------')
        
        
    accuracy = num_correct / num_samples
    nums_cleaned = [x for x in sum_ppl if not np.isnan(x)]
    ppl_avg = sum(nums_cleaned) / len(nums_cleaned)
    print('Accuracy:', accuracy)
    print('Perplexity:', ppl_avg)
    print('Diversity:', sum(diversity) / len(diversity))
    return accuracy, ppl_avg


def run_math(model_path, tokenizer, model, data):
    num_correct = 0
    num = 0
    sum_ppl = []
    diversity = []
    
    for i in data:
        question = i['question']
        answers = i['answer']
        
        if 'Qwen' in model_path:
            max_new_tokens = 500
            system_prompt = "You are a helpful math expert who can help me. Answer the question and put the final answer at the end of the sentence."
            answer = run_qwen(system_prompt, question, tokenizer, model, max_new_tokens)
            
        elif 'gemma' in model_path:
            max_new_tokens = 500
            prompt = question
            answer = run_gemma(prompt, tokenizer, model, max_new_tokens)
            
        elif 'llama' in model_path:
            max_new_tokens = 500
            system_prompt = "You are a helpful math expert who can help me. Answer the question and put the final answer at the end of the sentence."
            answer = run_llama(system_prompt, question, tokenizer, model, max_new_tokens)
            
            
        ppl = calculate_perplexity(answer, model, tokenizer)
        diversity.append(calculate_ngram_diversity([answer], n=2))
        
        print('Perplexity:', ppl)
        sum_ppl.append(ppl)
            
        print(question)
        print('---------------------------')
        print('Answer:', answer)
        
        
        # Extract the final answer
        answer = extract_last_answer(answer)
        print('Extracted Answer:', answer)
        
        answers = extract_last_answer(answers)
        print('Ground Truth:', answers)
        
        # Check correctness
        if answer is not None and answers is not None:
            if float(answer) == float(answers):
                print("Correct")
                num_correct += 1
            else:
                print("Incorrect")
        else:
            print("Error: One of the values is None")
        
        num += 1
        if num == 1000:
            break
    
    accuracy = num_correct / 1000  # Accuracy based on 1000 samples
    nums_cleaned =  [x for x in sum_ppl if not np.isnan(x)]
    ppl_avg = sum(nums_cleaned) / len(nums_cleaned)
    print('Accuracy:', accuracy)
    print('Perplexity:', ppl_avg)
    print('Diversity:', sum(diversity) / len(diversity))
    
    return accuracy, ppl_avg


def run_imdb(model_path, tokenizer, model, data, judge_ppl):
    num_correct = 0
    num = 0
    ppl_sum = []
    diversity = []
    #if you want to calculate ppl and diversity in imdb dataset, you should use this system prompt. If you just want to compute accuracy, you can use the system prompt in run_gemma function
    # if you did not use this system prompt,ppl and diversity is meaningless
    if judge_ppl:
       ppl_prompt = 'You should explain the reason for your answer.'
    
    for i in data['train']:
        
        question = i['text']
        answers = i['label']
        
        # Model-specific processing
        if 'gemma' in model_path:
            prompt = question + ' Judge the statement Negative or Positive'
            if judge_ppl:
                prompt = prompt + ppl_prompt
            max_new_tokens = 300
            answer = run_gemma(prompt, tokenizer, model, max_new_tokens)
        elif 'llama' in model_path:
            max_new_tokens = 2000
            system_prompt = "You are a helpful movie critic who can help me. Answer ''Negative'' or ''Positive''"
            if judge_ppl:
                system_prompt = system_prompt + ppl_prompt
            answer = run_llama(system_prompt, question, tokenizer, model, max_new_tokens)
        
        elif 'Qwen' in model_path:
            max_new_tokens = 2000
            system_prompt = "You are a helpful movie critic who can help me. Answer ''Negative'' or ''Positive''"
            if judge_ppl:
                system_prompt = system_prompt + ppl_prompt
            answer = run_qwen(system_prompt, question, tokenizer, model, max_new_tokens)
        
        print('---------------------------')
        if judge_ppl:
            ppl = calculate_perplexity(answer, model, tokenizer)
            print('Perplexity:', ppl)    
            diversity.append(calculate_ngram_diversity([answer], n=2))
            ppl_sum.append(ppl)
        
        # Print the question and answer
        
        print('Answer:', answer)
       

        
        # Determine the ground truth label
        if answers == 0:
            gt = "Negative"
        else:
            gt = "Positive"
        
        print('Ground Truth:', gt)
        
        # Check correctness
        if "positive" in answer.lower() and "negative" in answer.lower():
            print("Incorrect (both options present)")
        else:
            if gt.lower() in answer.lower():
                print("Correct")
                num_correct += 1
            else:
                print("Incorrect")
        
        if num == 1000:
            break
        num += 1
    
    accuracy = num_correct / 1000  # Accuracy based on 1000 samples
    
    if judge_ppl:
        nums_cleaned =  [x for x in ppl_sum if not np.isnan(x)]
        diversity_avg = sum(diversity) / len(diversity)
        ppl_avg = sum(nums_cleaned) / len(nums_cleaned)
        print('Diversity:', diversity_avg)
        print('Perplexity:', ppl_avg)
        
    print('Accuracy:', accuracy)
   
    
    
    return accuracy


def run_sports(model_path, tokenizer, model, data):
    
    questions_data = data
    num_correct = 0
    k = 0
   

    for i in questions_data:
        question = i['question']
        label = i['answer']
        print(question)

        # Construct the prompt based on the domain
        system_prompt = f"You are a helpful sports expert who can help me. Answer ''Yes'' or ''No''."
        
        if 'gemma' in model_path:
            max_new_tokens = 128
            answer = run_gemma(question, tokenizer, model, max_new_tokens)
        elif 'llama' in model_path:
            max_new_tokens = 300
            answer = run_llama(system_prompt, question, tokenizer, model, max_new_tokens)
        elif 'Qwen' in model_path:
            max_new_tokens = 300
            answer = run_qwen(system_prompt, question, tokenizer, model, max_new_tokens)
        else:
            answer = None  # You can add other models here if needed

        print(answer)
        final_answer = check_sentence(answer)  # Ensure the answer is processed (e.g., removed extra text)
        print('---------------------------')

        # Convert label to ground truth
        gt = "True" if label else "False"
        print('Ground Truth:', gt)
        
        # Check if the answer matches the ground truth
        if gt in final_answer:
            print("Correct")
            num_correct += 1
        else:
            print("Incorrect")
        print('---------------------------')

        k += 1

    print(f"Accuracy: {num_correct}/200")
    return num_correct / 200

def run_art(model_path, tokenizer, model, data):
    
    questions_data = data
    num_correct = 0
    k = 0
   

    for i in questions_data:
        question = i['question']
        label = i['answer']
        print(question)

        # Construct the prompt based on the domain
        system_prompt = f"You are a helpful art expert who can help me. Answer ''Yes'' or ''No''."
        
        if 'gemma' in model_path:
            max_new_tokens = 128
            answer = run_gemma(question, tokenizer, model, max_new_tokens)
        elif 'llama' in model_path:
            max_new_tokens = 300
            answer = run_llama(system_prompt, question, tokenizer, model, max_new_tokens)
        elif 'Qwen' in model_path:
            max_new_tokens = 300
            answer = run_qwen(system_prompt, question, tokenizer, model, max_new_tokens)
        else:
            answer = None  # You can add other models here if needed

        print(answer)
        final_answer = check_sentence(answer)  # Ensure the answer is processed (e.g., removed extra text)
        print('---------------------------')

        # Convert label to ground truth
        gt = "True" if label else "False"
        print('Ground Truth:', gt)
        
        # Check if the answer matches the ground truth
        if gt in final_answer:
            print("Correct")
            num_correct += 1
        else:
            print("Incorrect")
        print('---------------------------')

        k += 1

    print(f"Accuracy: {num_correct}/200")
    return num_correct / 200


def run_cele(model_path, tokenizer, model, data):
    
    questions_data = data
    num_correct = 0
    k = 0
   

    for i in questions_data:
        question = i['question']
        label = i['answer']
        print(question)

        # Construct the prompt based on the domain
        system_prompt = f"You are a helpful expert who can help me. Answer ''Yes'' or ''No''."
        
        if 'gemma' in model_path:
            max_new_tokens = 128
            answer = run_gemma(question, tokenizer, model, max_new_tokens)
        elif 'llama' in model_path:
            max_new_tokens = 300
            answer = run_llama(system_prompt, question, tokenizer, model, max_new_tokens)
        elif 'Qwen' in model_path:
            max_new_tokens = 300
            answer = run_qwen(system_prompt, question, tokenizer, model, max_new_tokens)
        else:
            answer = None  # You can add other models here if needed

        print(answer)
        final_answer = check_sentence(answer)  # Ensure the answer is processed (e.g., removed extra text)
        print('---------------------------')

        # Convert label to ground truth
        gt = "True" if label else "False"
        print('Ground Truth:', gt)
        
        # Check if the answer matches the ground truth
        if gt in final_answer:
            print("Correct")
            num_correct += 1
        else:
            print("Incorrect")
        print('---------------------------')

        k += 1

    print(f"Accuracy: {num_correct}/200")
    return num_correct / 200


def run_tech(model_path, tokenizer, model, data):
    
    questions_data = data
    num_correct = 0
    k = 0
   

    for i in questions_data:
        question = i['question']
        label = i['answer']
        print(question)

        # Construct the prompt based on the domain
        system_prompt = f"You are a helpful tech expert who can help me. Answer ''Yes'' or ''No''."
        
        if 'gemma' in model_path:
            max_new_tokens = 128
            answer = run_gemma(question, tokenizer, model, max_new_tokens)
        elif 'llama' in model_path:
            max_new_tokens = 300
            answer = run_llama(system_prompt, question, tokenizer, model, max_new_tokens)
        elif 'Qwen' in model_path:
            max_new_tokens = 300
            answer = run_qwen(system_prompt, question, tokenizer, model, max_new_tokens)
        else:
            answer = None  # You can add other models here if needed

        print(answer)
        final_answer = check_sentence(answer)  # Ensure the answer is processed (e.g., removed extra text)
        print('---------------------------')

        # Convert label to ground truth
        gt = "True" if label else "False"
        print('Ground Truth:', gt)
        
        # Check if the answer matches the ground truth
        if gt in final_answer:
            print("Correct")
            num_correct += 1
        else:
            print("Incorrect")
        print('---------------------------')

        k += 1

    print(f"Accuracy: {num_correct}/200")
    return num_correct / 200