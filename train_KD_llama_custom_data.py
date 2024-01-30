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

mbpp_name = 'mbpp'
mbpp_path = '/home/ubuntu/vllm-spec-dec/examples/34B_1.1B_SD_1.json'
human_eval_path = '/home/ubuntu/vllm-spec-dec/examples/70b_human_eval_analysis.json'
chatbot_arena_name = 'lmsys/chatbot_arena_conversations'
shareGPT_name = 'anon8231489123/ShareGPT_Vicuna_unfiltered'
chatbot_path = '/home/ubuntu/vllm-spec-dec/examples/70b_1.1b_10k_chatbot_analysis.json'

#model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
#model_name = 'JackFram/llama-160m'
#model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
#model_name = '/home/ubuntu/pretrained_draft_models/TinyLlama_73M_checkpoints'
model_name = '/home/ubuntu/pretrained_draft_models/Tinyllama_120M_checkpoints'
#teacher_model_name = 'meta-llama/Llama-2-70b-chat-hf'
teacher_model_name = 'codellama/CodeLlama-34b-Python-hf'
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

def train(model, teacher_model, dataset, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    teacher_model.eval()
    model.train()

    shuffle(dataset)

    losses = []

    for sample_id in trange(len(dataset)):
        input = dataset[sample_id]
        prompt_len = input['prompt_len']
        response_len = input['response_len']
        len_to_train = prompt_len + min(response_len, 256)
        input_ids = torch.tensor(input['token_ids'][:len_to_train]).unsqueeze(0).cuda()
        
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids)
            labels = teacher_outputs.get('logits')[:, prompt_len:, :].squeeze()

        outputs = model(input_ids)
        logits = outputs.get('logits')[:, prompt_len:, :].squeeze()
        
        error_tokens = (torch.argmax(labels, dim=-1) != torch.argmax(logits, dim=-1)).to(torch.int)
        nc_tokens = (find_nucleus_size(logits) > 2 * find_nucleus_size(labels)).to(torch.int)

        #error_tokens_indices = torch.nonzero(error_tokens).squeeze(-1)
        error_nc_tokens_indices = torch.nonzero(error_tokens + nc_tokens).squeeze(-1)

        error_logits = torch.index_select(logits, 0, error_nc_tokens_indices)
        error_labels = torch.index_select(labels, 0, error_nc_tokens_indices)

        loss = kl_loss(F.log_softmax(error_logits, dim=-1), F.softmax(error_labels, dim=-1))
        

        #loss = kl_loss(F.log_softmax(logits, dim=-1), F.softmax(labels, dim=-1))

        losses.append(loss.detach().cpu())

        loss.backward()
        optimizer.step()
    
    return losses

def test_acceptance_rate(test_data, teacher_model, model, t_temp, s_temp):

    teacher_model.eval()
    model.eval()
    avg_acceptance_rate = 0
    for sample_id in trange(len(test_data)):
        input = test_data[sample_id]
        prompt_len = input['prompt_len']
        input_ids = torch.tensor(input['token_ids'][:prompt_len]).unsqueeze(0).cuda()
        _, total_count_accepted, total_count_generated = ssp(teacher_model, model, 256,
            input_ids, t_temp, s_temp, K=5, display=False)
        
        acceptance_rate = total_count_accepted / total_count_generated
        avg_acceptance_rate += acceptance_rate / len(test_data)
        print(acceptance_rate)

    return avg_acceptance_rate

def moving_avg(losses, block_size=50):
    assert len(losses) > block_size

    moving_avg_ls = np.zeros(len(losses) - block_size)
    for i in range(len(losses) - block_size):
        moving_avg_ls[i] = np.mean(losses[i:i+block_size])

    return moving_avg_ls

def find_nucleus_size(logits, tempurature=1.0, p_threshold=0.9):
    probs = torch.softmax(logits / tempurature, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = (cum_sum_probs < p_threshold).to(torch.int)
    nucleus_size = torch.sum(nucleus, dim=-1) + 1

    return nucleus_size

def filter_dataset(dataset, prompt_threshold=128, response_threshold=10):

    new_dataset = [item for item in dataset if (item['prompt_len'] < prompt_threshold and item['response_len'] > response_threshold)]

    print(len(new_dataset))

    return new_dataset


def main():
    # training hyperparameters
    init_lr = 5e-7
    n_epochs = 5
    train_val_split = 0.2

    path_to_saved_model = f"/home/ubuntu/llama-ssp/KD_draft_models/120M/34b_120m_nc_token_20_train_mbpp"
    os.makedirs(path_to_saved_model, exist_ok=True)
    #saved_model_ckpt = f"/home/ubuntu/llama-ssp/KD_draft_models/{model_name}/70b_nc_token_20_train/ckpt_0"

    with open(mbpp_path, 'r') as json_file:
        dataset = json.load(json_file)

    #dataset = filter_dataset(ds)

    num_train_data = int(train_val_split * len(dataset))

    train_data = dataset[:num_train_data]
    test_data = dataset[num_train_data:]

    json_dict = dict()
    json_dict[0] = test_data

    with open("/home/ubuntu/llama-ssp/results/test_data.json", "w") as outfile:
        json.dump(json_dict, outfile)

    #load model 
    teacher_model = LlamaForCausalLM.from_pretrained(teacher_model_name, device_map='balanced')
    model = LlamaForCausalLM.from_pretrained(model_name, device_map={'':0})
    #model = LlamaForCausalLM.from_pretrained(saved_model_ckpt, device_map='balanced')

    #with open("/home/ubuntu/llama-ssp/results/llama70b_llama1.1b_analysis_mbpp.json", 'r') as json_file:
    #    dataset = json.loads(json_file)

    #train and test acceptance rate every 100 samples

    total_losses = []
    acceptance_rate_ls = []

    for epoch in trange(n_epochs):

        losses = train(model, teacher_model, train_data, lr=init_lr)
        print(losses)

        total_losses.extend(losses)

        path = path_to_saved_model + f"/ckpt_{epoch}"
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)

    return total_losses, acceptance_rate_ls


if __name__ == '__main__':
    total_losses, acceptance_rate_ls = main()
    moving_avg_ls = moving_avg(total_losses)
    np.savetxt('/home/ubuntu/llama-ssp/results/34b_1.1b_v0.6_20_train_nc_mbpp_moving_avg.txt', moving_avg_ls)
    np.savetxt('/home/ubuntu/llama-ssp/results/34b_1.1b_v0.6_20_train_nc_mbpp_alpha.txt', np.asarray(acceptance_rate_ls))
