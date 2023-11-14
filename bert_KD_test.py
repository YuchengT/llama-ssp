import os
import argparse
import json
import numpy as np
import logging
import scipy
from lssp.base import create_model
from lssp.base import sample_model, stream_token_if_required
from lssp import evals
from lssp.ssp import ssp, ssp_beam_greedy
import sys
import time
import torch
#import code_bert_score
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, GPT2Tokenizer, GPT2LMHeadModel

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

texts = [
    'In which country is Hamburg?\n',
    'How are you doing today?\n',
    'It was a dark and stormy night.',
    'The sun rose slowly over the horizon, casting a warm glow on the world below.',
    'I never believed in ghosts until the day I met one.',
    'The sound of the train whistle echoed through the valley as I stood at the station, waiting.',
    'She walked into the room and everything changed.',
    'The smell of freshly baked bread filled the air as I entered the bakery.',
    'The first time I saw her, I knew she was trouble.'
    'The world was ending, and I was the only one who knew.',
    'It was the best of times, it was the worst of times.',
    'The forest was alive with the sound of animals as I walked deeper into the woods.',
    'As I looked out over the city, I knew that anything was possible.',
    'The sound of gunfire echoed through the streets as I ran for cover.',
    'The waves crashed against the shore, a never-ending cycle of destruction and creation.',
    'I woke up to find myself in a strange place, with no memory of how I got there.',
    'The clock struck midnight, and I knew that my life would never be the same.',]

def generate_logits_for_incremental_seq(model, draft_model, text, target_tempurature):
    target_logits_ls = []
    draft_logits_ls = []
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    current_input_ids = input_ids
    print(target_tempurature)
    while current_input_ids.size(1) < input_ids.size(1) + 32:
        new_input_ids = current_input_ids
        target_outputs = model(new_input_ids)
        target_logits_ls.append(target_outputs.logits[:, -1, :])
        draft_outputs = draft_model(new_input_ids)
        draft_logits_ls.append(draft_outputs.logits[:, -1, :])

        #use greedy algorithm
        #new_token_id = torch.max(target_logits_ls[-1], dim=-1)[1]
        new_token_id = torch.multinomial(torch.softmax(target_outputs.logits[:, -1, :] / target_tempurature, dim=-1), num_samples=1).squeeze(-1)
        current_input_ids = torch.cat([new_input_ids, new_token_id.unsqueeze(-1)], dim=-1)

        stream_token_if_required(current_input_ids, new_input_ids, stream=True)

    target_logits = torch.cat(target_logits_ls, dim=0).detach().numpy()
    draft_logits = torch.cat(draft_logits_ls, dim=0).detach().numpy()

    np.savetxt(f"/home/ubuntu/llama-ssp/results/target_logits.txt", target_logits)
    np.savetxt(f"/home/ubuntu/llama-ssp/results/draft_logits.txt", draft_logits)

    #return target_logits_ls, draft_logits_ls

TEMPURATURE_LOW = 0.2
TEMPURATURE_HIGH = 1.1

def analyze_tempurature_relation(target_logits, draft_logits, target_tempurature):
    target_prob = scipy.special.softmax(target_logits / target_tempurature)
    def distribution_norm(x):
        return np.sum(abs(target_prob - scipy.special.softmax(draft_logits * x)))
    res = scipy.optimize.minimize_scalar(distribution_norm, method='brent', options={'xtol': 1.48e-04})
    wo_opt = np.sum(abs(target_prob - scipy.special.softmax(draft_logits / target_tempurature)))
    if wo_opt < res.fun:
        np.savetxt("/home/ubuntu/llama-ssp/results/exception_target_logits.txt", target_logits)
        np.savetxt("/home/ubuntu/llama-ssp/results/exception_target_prob.txt", target_prob)
        np.savetxt("/home/ubuntu/llama-ssp/results/exception_draft_logits.txt", draft_logits)
        np.savetxt("/home/ubuntu/llama-ssp/results/exception_minimizer.txt", np.asarray([res.x, res.fun, res.nit, wo_opt]))
        raise AssertionError("wrong minimization")
    return 1 / res.x, res.fun, wo_opt


def find_optimal_draft_tempurature(model, draft_model, text, n_partition):
    target_temps = np.linspace(TEMPURATURE_LOW, TEMPURATURE_HIGH, n_partition)
    tempuarture_matrix = np.zeros((n_partition, 32))
    norm_matrix = np.zeros((n_partition, 32))
    wo_opt_matrix = np.zeros((n_partition, 32))
    for i, target_temp in enumerate(target_temps):
        generate_logits_for_incremental_seq(model, draft_model, text, target_temp)
        target_logits = np.loadtxt(f"/home/ubuntu/llama-ssp/results/target_logits.txt")
        draft_logits = np.loadtxt(f"/home/ubuntu/llama-ssp/results/draft_logits.txt")
        for j, (target_logit, draft_logit) in enumerate(zip(target_logits, draft_logits)):
            tempuarture_matrix[i,j], norm_matrix[i,j], wo_opt_matrix[i,j] = analyze_tempurature_relation(target_logit, draft_logit, target_temp)
        
    return tempuarture_matrix, norm_matrix, wo_opt_matrix


if __name__ == "__main__":
    target_model_name = 'gpt2'
    draft_mode_name = 'distilgpt2'
    target_model = GPT2LMHeadModel.from_pretrained(target_model_name)
    #draft_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    draft_model = GPT2LMHeadModel.from_pretrained(draft_mode_name, load_in_4bit=True)    
    print(
        f"Find best draft tempurature for target model {target_model_name} and draft model {draft_mode_name}\n====\n")
    
    n_examples = len(texts)
    n_temps = 10
    t_mts = np.zeros((n_examples * n_temps, 32))
    n_mts = np.zeros((n_examples * n_temps, 32))
    wo_mts = np.zeros((n_examples * n_temps, 32))
    for i, text in enumerate(texts):
        print(text)
        t_mts[i * n_temps : (i+1) * n_temps], n_mts[i * n_temps : (i+1) * n_temps], wo_mts[i * n_temps : (i+1) * n_temps] = find_optimal_draft_tempurature(target_model, draft_model, text, n_temps)
        
        np.savetxt("/home/ubuntu/llama-ssp/results/gpt2_4bit_opt_temp.txt", t_mts)
        np.savetxt("/home/ubuntu/llama-ssp/results/gpt2_4bit_opt_diff.txt", n_mts)
        np.savetxt("/home/ubuntu/llama-ssp/results/gpt2_4bit_diff.txt", wo_mts)
