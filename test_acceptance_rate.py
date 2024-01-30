import transformers
import torch
import datasets
import json
import os
from datasets import Dataset, load_dataset
from tqdm import trange, tqdm
from lssp.ssp import ssp
from awq import AutoAWQForCausalLM
import torch.nn.functional as F
import torch.nn as nn
from random import shuffle
import numpy as np

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM
)

#model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
#model_name = 'JackFram/llama-160m'
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v0.6'
#model_name = "TheBloke/TinyLlama-1.1B-1T-OpenOrca-AWQ"
#model_name = "jeff31415/TinyLlama-1.1B-1T-OpenOrca"
#teacher_model_name = 'meta-llama/Llama-2-7b-chat-hf'
teacher_model_name = 'meta-llama/Llama-2-70b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)


def test_acceptance_rate(test_data, teacher_model, model, t_temp, s_temp):

    teacher_model.eval()
    model.eval()
    avg_acceptance_rate = 0
    for sample_id in trange(len(test_data)):
        input = test_data[sample_id]
        prompt_len = input['prompt_len']
        input_ids = torch.tensor(input['token_ids'][:prompt_len]).unsqueeze(0).cuda()
        _, total_count_accepted, total_count_generated = ssp(teacher_model, model, 512,
            input_ids, t_temp, s_temp, K=5, display=False)
        
        acceptance_rate = total_count_accepted / total_count_generated
        avg_acceptance_rate += acceptance_rate / len(test_data)
        print(acceptance_rate)

    return avg_acceptance_rate

def main():
    init_lr = 5e-7
    train_val_split = 0.8
    max_num_token = 512
    ckpt = 12

    with open("/home/ubuntu/llama-ssp/results/llama70b_llama1.1b_analysis_mbpp.json", 'r') as json_file:
        json_list = list(json_file)

    dataset = json.loads(json_list[0])

    test_data = dataset[int(train_val_split * len(dataset)):]
    saved_model_ckpt = f"/home/ubuntu/llama-ssp/KD_draft_models/{model_name}/70b_lr_{init_lr}_512_nc_token/ckpt_2"
    teacher_model = LlamaForCausalLM.from_pretrained(teacher_model_name, device_map='balanced')
    #model = LlamaForCausalLM.from_pretrained(model_name, device_map='balanced')
    model = LlamaForCausalLM.from_pretrained(saved_model_ckpt, device_map='balanced')

    acceptance_rate = test_acceptance_rate(test_data[:50], teacher_model, model, 0.1, 0.1)
    print(f"\n acceptance_rate_human_eval:{acceptance_rate}")

if __name__ == '__main__':
    main()