import json
import time
from typing import List, Dict, Set
from openai import OpenAI
import random
import hashlib

from dotenv import load_dotenv

load_dotenv()

import argparse

class TrueFalseGenerator:
    def __init__(self, category):
        self.client = OpenAI()
        self.seen_questions = set()  # Store hash of questions to check duplicates
        self.true_count = 0  # Track number of true questions
        self.false_count = 0  # Track number of false questions
        
        self.categories = [
            category
        ]

    def hash_question(self, question: str) -> str:
        """Create a hash of the question for deduplication."""
        # Normalize the question by converting to lowercase and removing extra spaces
        normalized = " ".join(question.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def is_duplicate(self, question: str) -> bool:
        """Check if a question is duplicate."""
        question_hash = self.hash_question(question)
        if question_hash in self.seen_questions:
            return True
        self.seen_questions.add(question_hash)
        return False
    
    def build_system_prompt(self, category: str, batch_size: int = 20, target_true_ratio: float = 0.5) -> Dict:
        current_total = self.true_count + self.false_count
        current_ratio = self.true_count / current_total if current_total > 0 else 0.5
        
        # Adjust desired true/false ratio for this batch to balance overall ratio
        if current_ratio < target_true_ratio:
            desired_true_count = batch_size  # Generate all true
        elif current_ratio > target_true_ratio:
            desired_true_count = 0  # Generate all false
        else:
            desired_true_count = int(batch_size * target_true_ratio)
        prompt = f"""Generate {batch_size} true/false knowledge questions about {category}. 
        Requirements:
        1. Questions should be in the format "Is [statement]?" or "Was [statement]?"
        2. Generate exactly {desired_true_count} TRUE statements and {batch_size - desired_true_count} FALSE statements
        3. For false statements, make subtle but clear changes to true facts
        4. Ensure high accuracy of facts
        5. Each question should be unique and non-redundant
        6. Keep questions simple, easy and factual

        Format your response as a JSON array with this structure:
        {{
            "qa_pairs": [
                {{
                    "question": "Is Beijing the capital of China?",
                    "answer": true,
                    "category": "{category}",
                    "correct_fact": "Beijing is the capital of China"
                }},
                {{
                    "question": "Is Tokyo the capital of China?",
                    "answer": false,
                    "category": "{category}",
                    "correct_fact": "Tokyo is the capital of Japan"
                }}
            ]
        }}
        """
        # self.prompt = prompt
        return prompt

    def generate_qa_batch(self, category: str, batch_size: int = 5, target_true_ratio: float = 0.5) -> List[Dict]:
        # Calculate how many true/false questions we want based on current ratio

        prompt = self.build_system_prompt(category, batch_size, target_true_ratio)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates true/false knowledge questions."},
                    {"role": "user", "content": prompt}
                ],
                # temperature=0.2,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Filter out duplicates
            non_duplicate_pairs = []
            for pair in result.get('qa_pairs', []):
                if not self.is_duplicate(pair['question']):
                    non_duplicate_pairs.append(pair)
            
            return non_duplicate_pairs
            
        except Exception as e:
            print(f"Error generating T/F pairs for {category}: {str(e)}")
            return []

    def verify_qa_pair(self, qa_pair: Dict) -> Dict:
        """Verify and potentially improve a single true/false question."""
        prompt = f"""Verify this true/false question for accuracy and improve if needed:
        Question: {qa_pair['question']}
        Answer: {qa_pair['answer']}
        Correct Fact: {qa_pair['correct_fact']}
        
        Requirements:
        1. Verify the factual accuracy
        2. Make sure the question is clear and unambiguous
        3. Ensure the true/false nature is definitive
        
        Respond in JSON format:
        {{
            "is_accurate": true/false,
            "improved_question": "...",
            "answer": true/false,
            "correct_fact": "...",
            "confidence": 0-1
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that verifies true/false questions."},
                    {"role": "user", "content": prompt}
                ],
                # temperature=0,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result['is_accurate'] and result['confidence'] > 0.8:
                # Check if improved question is duplicate
                if not self.is_duplicate(result['improved_question']):
                    return {
                        'question': result['improved_question'],
                        'answer': result['answer'],
                        'category': qa_pair['category'],
                        'correct_fact': result['correct_fact'],
                        'confidence': result['confidence']
                    }
            return None
            
        except Exception as e:
            print(f"Error verifying T/F pair: {str(e)}")
            return None

    def generate_dataset(self, total_pairs: int, output_filepath: str, target_true_ratio: float = 0.5):
        """Generate and save a true/false knowledge dataset with specified true/false ratio."""
        all_qa_pairs = []
        pairs_per_category = total_pairs // len(self.categories)
        
        for category in self.categories:
            pairs_generated = 0
            attempts = 0
            max_attempts = pairs_per_category * 2  # Allow for some failed attempts
            
            while pairs_generated < pairs_per_category and attempts < max_attempts:
                # Generate a batch of questions
                batch_size = min(20, pairs_per_category - pairs_generated)
                qa_batch = self.generate_qa_batch(category, batch_size)
                
                print("-------")
                print(qa_batch)
                print("-------\n")
                
                # Verify each QA pair
                for qa_pair in qa_batch:
                    # verified_pair = self.verify_qa_pair(qa_pair)
                    verified_pair = qa_pair
                    if verified_pair:
                        # Update true/false counts
                        if verified_pair['answer']:
                            self.true_count += 1
                        else:
                            self.false_count += 1
                        all_qa_pairs.append(verified_pair)
                        pairs_generated += 1
                
                # Print current true/false ratio
                total = self.true_count + self.false_count
                if total > 0:
                    true_ratio = self.true_count / total
                    print(f"Current true/false ratio: {true_ratio:.2f} ({self.true_count}/{total})")
                
                attempts += 1
                # Rate limiting
                time.sleep(1)
                
                print(f"Generated {pairs_generated}/{pairs_per_category} pairs for {category}")

        # Save the dataset
        dataset = {
            'version': '1.0',
            'total_pairs': len(all_qa_pairs),
            'categories': self.categories,
            'data': all_qa_pairs
        }
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Generated {len(all_qa_pairs)} total T/F pairs")
        return all_qa_pairs

    def sample_dataset(self, qa_pairs: List[Dict], num_samples: int = 5) -> List[Dict]:
        """Sample random T/F pairs from the dataset for quality inspection."""
        return random.sample(qa_pairs, min(num_samples, len(qa_pairs)))

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num_pairs", type=int, required=True)
    args = parser.parse_args()
    generator = TrueFalseGenerator(args.category)
    
    qa_pairs = generator.generate_dataset(args.num_pairs, args.category + "_qa.json")
