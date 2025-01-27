import json
from tqdm import tqdm
import nltk
nltk.download('words')  # Download the words corpus if you haven't already
from nltk.corpus import words
import re
import argparse
import os
import numpy as np
import sys
from pathlib import Path
import argparse
import string
import random

from transformers import AutoTokenizer
sys.path.append(".")
from transformers import LlamaTokenizer


class DuplicateStringCreator:
    """A task with the goal of duplicating a string.

    The input is a string s_1 ... s_n composed of symbols from a finite set S. The
    output is the same string outputted twice without any separator, ie:
    s_1 ... s_n s_1 ... s_n

    Examples:
        101 -> 101 101
        111111 -> 111111 111111

    In the paper, we use only binary strings (ie S = {0, 1}).
    Note that the sampling is jittable so this task is fast.
    """
    TASK_PREFIX = "There is a long string composed of many tokens. It's localed between '>>' and '<<'. Memorize it. I will require you to repeat it. The target string is: \n"
    
    def __init__(self, tokenizer):
        """Initializes the remember_string task.

        Args:
        tokenizer: The tokenizer to use.
        vocab_size: The size of the alphabet.
        """
        self.tokenizer = tokenizer
        self.descriptor_len = len(self.tokenizer(self.TASK_PREFIX))

    def sample_string(self, length):
        choices = np.array(list(string.ascii_letters))
        return ''.join(np.random.choice(choices, size=length))

    def sample_words(self, length):
        word_list = words.words()
        return ' '.join(random.choices(word_list, k=int(length * 0.6)))

    def create_task_duplicate(self, max_token_length, num_examples=100, word_seq=False):

        assert max_token_length > self.descriptor_len

        num_left_tokens = max_token_length - self.descriptor_len

        repeated_len = num_left_tokens // 2

        assert repeated_len >= 2

        samples = []
        for _ in range(num_examples):
            if word_seq:
                random_sequence = self.sample_words(repeated_len)
            else:
                random_sequence = self.sample_string(repeated_len)

            input_ = [self.TASK_PREFIX] + [f">>\n{random_sequence}\n<<",] 
            # maybe we need add more beyongd this part to instruc the model to answer
            samples.append({
                "input": " ".join(input_),
                "target": str(random_sequence)

            })
        return samples 
    
def create_duplicate_string(tokenizer, data_path:Path, seq_length, num_example, word_seq=False):
    
    task_creator = DuplicateStringCreator(tokenizer)
    samples = task_creator.create_task_duplicate(seq_length, num_example, word_seq)

    print(f"Created Duplicate String Task Data: {len(samples)} instances")

    random_length = len(tokenizer.encode(random.choice(samples)['input']))
    print(f"Length of random sample: {random_length}")

    output_dir = Path(data_path) / f"{tokenizer.__class__.__name__}_{seq_length}_{word_seq}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir/"test.jsonl", "w") as f:
        for entry in samples:
            jout = json.dumps(entry) + "\n"
            f.write(jout)
    
    print("interval ")
    print(output_dir / f"{tokenizer.__class__.__name__}_{seq_length}_{word_seq}"/"test.jsonl")


class PassKeyTaskCreator:
    TASK_PREFIX = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information there."
    DEFAULT_CONTENT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    KEY_CONTENT = "The pass key is {KEY}. Remember it. {KEY} is the pass key."
    def __init__(self, tokenizer, passkey_length, data=None):
        """
        data: List containing inputs
        """

        self.data = data # if not None, will use random sentences instead of "DEFAULT_CONTENT" to fill the prompt
        self.tokenizer = tokenizer
        self.random_char_list = list(string.ascii_letters)
        self.passkey_length = passkey_length

        if self.data is None:
            self.distractor = self.DEFAULT_CONTENT
            self.descriptor_len = len(self.tokenizer.encode(self.TASK_PREFIX))
            self.distractor_len = len(self.tokenizer.encode(self.distractor))
            self.key_info_len = len(self.tokenizer.encode(self.KEY_CONTENT))

    def create_task_retrieve(self, max_token_length, num_examples=100, insert_position=None, pos_interval=None):
        assert max_token_length > (self.key_info_len + self.descriptor_len)

        num_distractors = (max_token_length - self.key_info_len - self.descriptor_len) // self.distractor_len

        assert num_distractors >= 2

        rng = np.random.RandomState(seed=(max_token_length+insert_position))
    
        def recursive_passkey(L):
            if L > 18:
                rand_s_1, rand_e_1 = 10**(18-1), 10**18-1
                random_answer_1 = rng.randint(rand_s_1, rand_e_1)
                new_L = L-18
                return random_answer_1*10**new_L + recursive_passkey(new_L)
            else:
                rand_s_1, rand_e_1 = 10**(L-1), 10**L-1
                random_answer_1 = rng.randint(rand_s_1, rand_e_1)
                return  random_answer_1            

        samples = []
        for _ in range(num_examples):
            #if self.passkey_length <= 19:
            #    rand_s, rand_e = 10**(self.passkey_length-1), 10**self.passkey_length-1
            #    #random_answer = rng.randint(1000000000000000,9999999999999999)
            #    random_answer = rng.randint(rand_s, rand_e)
            #elif self.passkey_length <= 37:
            #    part1_len = 18
            #    part2_len = self.passkey_length - part1_len
            #    rand_s_1, rand_e_1 = 10**(part1_len-1), 10**part1_len-1
            #    rand_s_2, rand_e_2 = 10**(part2_len-1), 10**part2_len-1
            #    random_answer_1 = rng.randint(rand_s_1, rand_e_1)
            #    random_answer_2 = rng.randint(rand_s_2, rand_e_2)
            #    random_answer = random_answer_1 * (10**part2_len) + random_answer_2
            #elif self.passkey_length <= 37:
            #    part1_len = 18
            #    part2_len = 37-18
            #    part3_len = 
            #    part2_len = self.passkey_length - part1_len
            #    rand_s_1, rand_e_1 = 10**(part1_len-1), 10**part1_len-1
            #    rand_s_2, rand_e_2 = 10**(part2_len-1), 10**part2_len-1
            #    random_answer_1 = rng.randint(rand_s_1, rand_e_1)
            #    random_answer_2 = rng.randint(rand_s_2, rand_e_2)
            #    random_answer = random_answer_1 * (10**part2_len) + random_answer_2
            #random_answer = ''.join(rng.choice(self.random_char_list, size=10)) # it's random sampling with replacement.
            random_answer = recursive_passkey(self.passkey_length)
            answer_sentence = self.KEY_CONTENT.format(KEY=random_answer)

            if insert_position:
                if pos_interval:
                # insert_position is counted by number of tokens.
                    insert_location_s = (insert_position - self.descriptor_len) // self.distractor_len
                    insert_location_e = (insert_position + pos_interval - self.descriptor_len) // self.distractor_len
                    insert_location = rng.randint(insert_location_s, insert_location_e+1)
                else:
                    insert_location = (insert_position - self.descriptor_len) // self.distractor_len
            else:
                insert_location = rng.randint(0, num_distractors)
            input_ = [self.TASK_PREFIX] + [self.distractor] * insert_location + [answer_sentence] + [self.distractor] * (num_distractors - insert_location)
            input_seq = "\n".join(input_)
            pre_fix_seq = "\n".join([self.TASK_PREFIX,] + [self.distractor,] * insert_location)
            insert_pos = len(self.tokenizer.encode(pre_fix_seq))
            # maybe we need add more beyongd this part to instruc the model to answer
            samples.append({
                "input": "\n".join(input_),
                "target": str(random_answer),
                "passkey_position": insert_pos
            })
        return samples 

def create_passkey(tokenizer, data_path:Path, seq_length, num_example, insert_position=None, pos_interval=None, passkey_length=5):
    task_creator = PassKeyTaskCreator(tokenizer, passkey_length)
    if pos_interval:
        samples = task_creator.create_task_retrieve(seq_length, num_example, insert_position, pos_interval=pos_interval)

        print(f"Created Pass Task Data: {len(samples)} instances")

        random_length = len(tokenizer.encode(random.choice(samples)['input']))
        print(f"Length of random sample: {random_length}")

        output_dir = Path(data_path) / f"{tokenizer.__class__.__name__}_{seq_length}_{insert_position}_{pos_interval}_len_{passkey_length}"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir/"test.jsonl", "w") as f:
            for entry in samples:
                jout = json.dumps(entry) + "\n"
                f.write(jout)
        
        print("interval ")
        print(output_dir/"test.jsonl")
    else:
        raise ValueError("Either pos_interval or pos_ratio should be specified")


def add_dataset_creation_args(parser: argparse.ArgumentParser):
    parser.add_argument("--seq_length", type=int, default=4096, help="length of the sequence")
    parser.add_argument("--insert_position", type=int, default=None, help="position (by token) of the key, for passkey task")
    parser.add_argument("--num_gen_example", type=int, default=1000, help="number of generated examples")
    parser.add_argument("--word_seq", action="store_true", help="whether to use word sequence instead of random string")
    parser.add_argument("--interval", type=int, default=2000, help="interval of the insert position")
    parser.add_argument("--begin_pos", type=int, default=500, help="begin position of the insert position")

    return parser

    
