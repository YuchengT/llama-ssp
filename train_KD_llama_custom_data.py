import transformers
import torch
import datasets
from datasets import Dataset, load_dataset
from tqdm import trange, tqdm
from lssp.ssp import ssp

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM
)

test_texts = [
    'The future of Amazon is',
    'The future of Google is',
    'The future of Apple is',
    'The future of Microsoft is',
    'The future of Tesla is',
    'The future of Nvidia is',
    'The future of Meta is',
    'The future of Netflix is',
    'The capital of United States is',]

model_name = 'JackFram/llama-160m'
teacher_model_name = 'meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def train(model, teacher_model, train_text_ls, n_iters):
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    teacher_model.eval()
    model.train()

    for sample_id in trange(len(train_text_ls)):
        input = train_text_ls[sample_id]
        input_ids = tokenizer(input, return_tensors="pt").input_ids[:, :-1]

        for iter in range(n_iters):
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids)
                labels = teacher_outputs.get('logits')[:, -1, :]
            outputs = model(input_ids)
            logits = outputs.get('logits')[:, -1, :]
            loss = torch.norm(torch.softmax(logits.squeeze(), dim=-1) - torch.softmax(labels.squeeze(), dim=-1), p=1)

            loss.backward()
            optimizer.step()

            next_id = torch.max(labels, dim=-1)[1]
            input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=-1)

            if next_id == 2:
                break
    
    return None

def test_acceptance_rate(texts, teacher_model, model, t_temp, s_temp):
    avg_acceptance_rate = 0
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        _, total_count_accepted, total_count_generated = ssp(teacher_model, model, 64,
            input_ids, t_temp, s_temp, K=6, display=True)
        
        acceptance_rate = total_count_accepted / total_count_generated
        avg_acceptance_rate += acceptance_rate / len(test_texts)

    return avg_acceptance_rate



def main():
    # training hyperparameters
    n_epochs = 100
    n_iters = 128
    t_temp = 0.1
    s_temp = 0.1

    #load model 
    teacher_model = LlamaForCausalLM.from_pretrained(teacher_model_name, device_map='balanced')
    model = LlamaForCausalLM.from_pretrained(model_name, device_map='balanced')

    teacher_model.to(device='cuda')
    model.to(device='cuda')

    print(teacher_model.hf_device_map)

    dataset_name = "togethercomputer/RedPajama-Data-1T-Sample"

    #load dataset
    raw_datasets = load_dataset(dataset_name)
    text_ls = []
    for i in range(len(raw_datasets['train'])):
        text_ls.append(raw_datasets['train'][i]['text'][:512])

    #train and test acceptance rate every 1000 samples

    for epoch in trange(n_epochs):
        train(model, teacher_model, text_ls[1000 * epoch: 1000 * (epoch + 1)], n_iters)
        acceptance_rate = test_acceptance_rate(test_texts, teacher_model, model, t_temp, s_temp)
        print(acceptance_rate)
    
    return None


if __name__ == '__main__':

    main()
