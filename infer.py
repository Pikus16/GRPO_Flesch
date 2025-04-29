
#import unsloth
from transformers import AutoModelForCausalLM, AutoTokenizer
#from unsloth import FastLanguageModel
import argparse
import torch
import re
from datasets import load_dataset, Dataset
import textstat
import numpy as np
import json
import os
from tqdm import tqdm
from glob import glob
import pandas as pd
from gsm8k import SYSTEM_PROMPT, XML_COT_FORMAT, extract_hash_answer, get_gsm8k_questions
from vllm import LLM, SamplingParams
import gc
import json

params = {
    "load_in_4bit": True,
    "device_map": "auto",  # Automatically handle model placement
    "torch_dtype": torch.float16,  # Use half precision
    "use_cache": True,
    #"attn_implementation": "flash_attention_2",  # Use FlashAttention if available
}

dataset = get_gsm8k_questions(split = "test")

generation_config = {
    "max_new_tokens": 2048,
    "do_sample": False,
    "use_cache": True
}

def run_generation(model_, tokenizer_, batch_size=8, num_ex=100):
    outputs = []
    if num_ex == -1:
        num_ex = len(dataset)
    for i in tqdm(range(0, num_ex, batch_size)):
        batch_end = min(i + batch_size, num_ex)
        batch_prompts = []
        
        # Prepare batch of prompts
        for j in range(i, batch_end):
            prompt = dataset[j]['question']
            answer = dataset[j]['answer']
            batch_prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
        text = [tokenizer_.apply_chat_template(prompt, tokenize = False, add_generation_prompt = True) for prompt in batch_prompts]

        inputs = tokenizer_(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        input_length = inputs.input_ids.shape[-1]
        output = model_.generate(**inputs, **generation_config )
        output = output[:, input_length: ]
        batch_outputs = tokenizer_.batch_decode(output, skip_special_tokens=True)
        outputs.extend(batch_outputs)
    return outputs

def load_model(model_path: str, max_seq_length: int = 1024):
    model = AutoModelForCausalLM.from_pretrained(model_path, **params).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model, tokenizer = FastLanguageModel.from_pretrained(model_path,
    #                                                      max_seq_length=max_seq_length,
    #     load_in_4bit=True,
    #     fast_inference=True,
    #     max_lora_rank=32,
    #     gpu_memory_utilization=0.9,
    #     use_vllm=True)
    return model, tokenizer


def run(model_path: str, num_ex=100):
    model, tokenizer = load_model(model_path)
    outputs = run_generation(model, tokenizer, num_ex=num_ex)
    model.to("cpu")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return outputs

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Checkpoint output directory (e.g., 'output')")
    parser.add_argument("--num_ex", type=int, default=100, help="Number of examples to run")
    args = parser.parse_args()

    DIR = args.dir

    checkpoints = range(500, 3001, 500)
    BASE_PATH = f'lora_checkpoints/{DIR}/checkpoint-'
    model_paths = [
        #'unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit',
        *[f'{BASE_PATH}{c}' for c in checkpoints]
    ]

    filename = f'lora_checkpoints/{DIR}/test_examples_fixed.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_outputs = json.load(f)
    else:
        all_outputs = []
        
    model_path = model_paths[len(all_outputs)]
    if not os.path.exists(model_path):
        print('DOESNT EXIST')
    else:
        outputs = run(model_path)
        all_outputs.append(outputs)
        with open(filename, 'w') as f:
            json.dump(all_outputs, f)