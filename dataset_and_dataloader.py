import pandas as pd
import torch
import numpy as np

def read_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

class GSM8K_Base_Dataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file_path, tokenizer, tiny=False):
        self.tokenizer = tokenizer
        self.data = pd.read_json(jsonl_file_path, lines=True)
        if tiny:
          self.data = self.data[:10]
        self.train_data = pd.read_json('GSM8K/train.jsonl', lines=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        shots1_8_q = self.train_data['question'][:8].to_list()
        shots1_8_a = self.train_data['answer'][:8].to_list()
        prompt_prefix = """// preamble or system instruction \
                        As an expert problem solver, solve step by step the following mathematical questions.\n\n"""
        questions = ["\n\n// shot-"+str(i)+"\nQ: " + shots1_8_q[i] + "\nA: Let's think step by step. " + shots1_8_a[i] + "\n" for i in range(8)]
        prompt_prefix += "".join(questions)
        prompt_prefix += "\n\n// target question\nQ: "
        prompt_suffix = "\nA: Let's think step by step. "
        question = [prompt_prefix + self.data['question'][idx] + prompt_suffix]
        tok_question = self.tokenizer(question, padding=True, padding_side='left')
        q = tok_question['input_ids'][0]
        q_mask = tok_question['attention_mask'][0]
        
        return q, q_mask, self.data['answer'][idx]

class GSM8K_Val_Dataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file_path, tokenizer, tiny=False):
        self.tokenizer = tokenizer
        self.data = pd.read_json(jsonl_file_path, lines=True)
        if tiny:
          self.data = self.data[:10]
        self.train_data = pd.read_json('GSM8K/train.jsonl', lines=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        shots1_8_q = self.train_data['question'][:8].to_list()
        shots1_8_a = self.train_data['answer'][:8].to_list()
        prompt_prefix = """// preamble or system instruction \
                        As an expert problem solver, solve step by step the following mathematical questions.\n\n"""
        questions = ["\n\n// shot-"+str(i)+"\nQ: " + shots1_8_q[i] + "\nA: Let's think step by step. " + shots1_8_a[i] + "\n" for i in range(8)]
        prompt_prefix += "".join(questions)
        prompt_prefix += "\n\n// target question\nQ: "
        prompt_suffix = "\nA: Let's think step by step. "
        question = [prompt_prefix + self.data['question'][idx] + prompt_suffix]
        tok_question = self.tokenizer(question, padding=True, padding_side='left')
        q = tok_question['input_ids'][0]
        q_mask = tok_question['attention_mask'][0]
        
        return self.data['id'][idx], self.data['instance'][idx], q, q_mask, self.data['answer'][idx]