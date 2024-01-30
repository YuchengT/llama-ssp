# Base functions for sampling and streaming

import sys
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, GPT2Tokenizer
import time
from awq import AutoAWQForCausalLM

llama7b_name = 'decapoda-research/llama-7b-hf'
codellama7b_name = 'decapoda-research/llama-7b-hf'
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
#tokenizer = AutoTokenizer.from_pretrained(codellama7b_name)
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def create_model(model_name, device_map, max_memory, load_in_8bit=True, load_in_4bit=False):
    model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=True, device_map=device_map,
                                          trust_remote_code=False, safetensors=True)

    return model
    #model = LlamaForCausalLM.from_pretrained(
    #        model_name,
    #        device_map=device_map,
    #        load_in_8bit=load_in_8bit,
    #        load_in_4bit=load_in_4bit,
    #        max_memory=max_memory
    #    )
    #return model, model.hf_device_map


def stream_token_if_required(output_ids, input_ids, stream=False):
    if stream is True:
        output_string = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True)
        previous_output_string = tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True)
        sys.stdout.write(output_string[len(previous_output_string):])
        sys.stdout.flush()


TEMPERATURE = 0.1


def get_temperature_distribution(logits, temperature=TEMPERATURE):
    if temperature == 0.0:
        return torch.softmax(logits, dim=-1)
    return torch.softmax(logits / temperature, dim=-1)


def sample_fn(logits, temperature=TEMPERATURE):
    if temperature > 0:
        probs = get_temperature_distribution(logits, temperature)
        if probs.dim() == 3:
            bs, nc, _ = probs.size()
            probs = probs.view(bs * nc, -1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1).reshape(bs, nc)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        return torch.max(logits, dim=-1)[1]


def sample_model(model,
                 input_ids,
                 nb_tokens,
                 display=False,
                 temperature=TEMPERATURE):

    for _ in range(nb_tokens):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = sample_fn(next_token_logits, temperature)
        output_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        stream_token_if_required(output_ids, input_ids, stream=display)
        input_ids = output_ids
        if 2 in input_ids:
            break
    return input_ids


def beam_search_model(model,
                      input_ids,
                      nb_tokens,
                      num_beams,
                      display=False):
    
    original_input_ids = input_ids.unsqueeze(1).repeat(1,2 * num_beams, 1)
    
    outputs = model(input_ids)
    logprobs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
    cummulative_logprobs, cummulative_token_indices = torch.topk(logprobs, 2 * num_beams)
    cummulative_token_indices = cummulative_token_indices.unsqueeze(-1)
    input_ids = torch.cat([original_input_ids, cummulative_token_indices], dim=-1)

    for _ in range(1, nb_tokens):
        batch_size, num_candidate_seq, seq_length = input_ids.size()
        input_ids = input_ids.view(batch_size * num_candidate_seq, seq_length)
        outputs = model(input_ids)
        logprobs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
        candidate_logprobs = logprobs.view(batch_size, num_candidate_seq, -1) + cummulative_logprobs.unsqueeze(-1)
        vocab_size = candidate_logprobs.size(-1)

        # regroup all num_beams * vocab_size threads for next-step selection.
        candidate_logprobs = candidate_logprobs.flatten(start_dim=-2)
        cummulative_logprobs, topk_indices = torch.topk(candidate_logprobs, 2 * num_beams)

        parent_indices = topk_indices // vocab_size
        next_token_indices = topk_indices % vocab_size

        cummulative_token_indices = cummulative_token_indices[torch.arange(batch_size).unsqueeze(-1).expand_as(parent_indices), parent_indices]
        cummulative_token_indices = torch.cat([cummulative_token_indices, next_token_indices.unsqueeze(-1)], dim=-1)

        input_ids = torch.cat([original_input_ids, cummulative_token_indices], dim=-1)
        
    return input_ids, cummulative_logprobs













