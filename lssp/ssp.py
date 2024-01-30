# Implementation of speculative sampling as per
# https://arxiv.org/abs/2302.01318
from collections import namedtuple
import torch

from torch import nn
from logging import info, debug, warning, error, critical

from lssp.base import get_temperature_distribution, sample_fn, stream_token_if_required, tokenizer, beam_search_model

torch.manual_seed(1339)


def _draft_sample_k(model, input_ids, K, draft_tempurature):
    """sample K tokens from the draft model autoregressively
    draft_logits are a (B, K, V) tensor
    inputs_plus_k are a (B, T+K) tensor
    """
    inputs_plus_k = input_ids
    draft_logits = []
    for t in range(K):
        outputs = model(inputs_plus_k)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = sample_fn(next_token_logits, temperature=draft_tempurature)
        inputs_plus_k = torch.cat(
            [inputs_plus_k, next_token_id.unsqueeze(1)],
            dim=1)
        draft_logits.append(next_token_logits)
    draft_logits = torch.stack(draft_logits, dim=1)
    return inputs_plus_k, draft_logits

def _target_sample_from_distribution(target_distribution, draft_distribution):
    distribution = (target_distribution - draft_distribution)
    distribution = torch.max(distribution,
                             torch.zeros_like(distribution))
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    return torch.multinomial(distribution, num_samples=1).squeeze(-1)

def _beam_greedy_ssp_iteration(target_model, draft_model, input_ids, num_beams, K=4, display=False):

    _, T = input_ids.shape
    input_plus_k, cumulative_logprobs = beam_search_model(draft_model, input_ids, K, num_beams)
    batch_size, num_candidates, _ = input_plus_k.size()

    target_input = input_plus_k.view(-1, T+K)
    target_logits = target_model(target_input).logits[:, -K-1:, :]
    target_output = sample_fn(target_logits).reshape(batch_size, num_candidates, K+1)

    accepted_length = torch.zeros(batch_size, num_candidates)
    accepted = torch.ones(batch_size, num_candidates)
    last_accepted = torch.ones(batch_size, num_candidates)

    for t in range(K):
        accepted = (input_plus_k[:,:,T + t] == target_output[:,:,t]) * last_accepted
        accepted_length += accepted
        last_accepted = accepted
    
    draft_score = accepted_length + torch.exp(cumulative_logprobs)
    draft_score = draft_score.reshape(batch_size, num_candidates)

    final_accepted_length = accepted_length[torch.arange(batch_size), torch.argmax(draft_score, dim=1)]
    final_accepted_tokens = target_output[torch.arange(batch_size), torch.argmax(draft_score, dim=1), :(final_accepted_length.int() + 1)]
    final_output = torch.cat([input_ids, final_accepted_tokens], dim=-1)

    stream_token_if_required(final_output, input_ids, stream=display)

    return final_output, final_accepted_length


def _ssp_iteration(target_model, draft_model, input_ids, target_tempurature, draft_tempurature, K=4, display=False):

    _, T = input_ids.shape
    # sample K tokens from the draft model autoregressively
    # draft_logits are a (B, K, V) tensor
    # inputs_plus_k are a (B, T+K) tensor
    inputs_plus_k, draft_logits = _draft_sample_k(
        draft_model, input_ids, K, draft_tempurature=draft_tempurature
    )
    debug(
        f"Possible continuations: {tokenizer.decode(inputs_plus_k[0,T:], skip_special_tokens=True)}")
    # get the logits for the same tokens from the target model
    # target_logits are a (B, K+1, V) tensor
    # TODO avoid using .logits since it is HF-specific
    target_logits = target_model(inputs_plus_k).logits[:, -K-1:, :]
    target_distribution = get_temperature_distribution(target_logits, target_tempurature)
    draft_distribution = get_temperature_distribution(draft_logits, draft_tempurature)
    # Accept-reject token loop
    all_accepted = True
    count_accepted = 0
    for t in range(1, K+1):
        sampled_ratios = (
            target_distribution[:1, t-1, inputs_plus_k[0, T+t-1]]
            / draft_distribution[:1, t-1, inputs_plus_k[0, T+t-1]]
        )
        # print(f"Sampled ratios: {sampled_ratios}")
        sampled_ratios = torch.min(sampled_ratios,
                                   torch.ones_like(sampled_ratios))
        rs = torch.rand_like(sampled_ratios)

        if (rs < sampled_ratios).any():  # TODO for B > 1, change this
            output_ids = torch.cat(
                [input_ids, inputs_plus_k[:, T + t-1].unsqueeze(1)],
                dim=1)
            stream_token_if_required(output_ids, input_ids, stream=display)
            input_ids = output_ids
            count_accepted += 1
        else:
            all_accepted = False
            next_token_id = _target_sample_from_distribution(
                target_distribution[:1, t-1, :],
                draft_distribution[:1, t-1, :])
            output_ids = torch.cat(
                [input_ids, next_token_id.unsqueeze(1)],
                dim=1)
            stream_token_if_required(output_ids, input_ids, stream=display)
            input_ids = output_ids
            break
        
    # if all tokens were accepted, sample a last one
    if all_accepted:
        next_token_id = sample_fn(target_logits[:1, -1, :], target_tempurature)
        output_ids = torch.cat(
            [input_ids, next_token_id.unsqueeze(1)],
            dim=1)
        stream_token_if_required(output_ids, input_ids, stream=display)
        input_ids = output_ids
    debug(
        f"Accepted continuations: {tokenizer.decode(input_ids[0,T:], skip_special_tokens=True)}")

    torch.cuda.empty_cache()

    return input_ids, count_accepted


def ssp(target_model, draft_model, min_nb_tokens, input_ids, target_tempurature, draft_tempurature, K=4, display=False):
    B, T = input_ids.shape
    assert B == 1, "Batch size must be 1, implement the fixes for B > 1"
    total_count_accepted = 0
    total_count_generated = 0
    while input_ids.shape[1] < T + min_nb_tokens:
        debug(f"Current length: {input_ids.shape[1]}")
        with torch.no_grad():
            input_ids, count_accepted = _ssp_iteration(target_model, draft_model, input_ids, target_tempurature, draft_tempurature, K, display)
        total_count_accepted += count_accepted
        total_count_generated += K
        if 2 in input_ids:
            break

    return input_ids, total_count_accepted, total_count_generated

def ssp_beam_greedy(target_model, draft_model, min_nb_tokens, input_ids, num_beams, K=4, display=False):
    B, T = input_ids.shape
    assert B == 1, "Batch size must be 1, implement the fixes for B > 1"
    total_count_accepted = 0
    total_count_generated = 0
    while input_ids.shape[1] < T + min_nb_tokens:
        debug(f"Current length: {input_ids.shape[1]}")
        input_ids, count_accepted = _beam_greedy_ssp_iteration(target_model, draft_model, input_ids, num_beams, K, display)
        total_count_accepted += count_accepted
        total_count_generated += K
        if 2 in input_ids:
            break

    return input_ids, total_count_accepted.item(), total_count_generated


class FakeModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        # Create fake logits randomly in the range [-1, 1]
        B, T = input_ids.shape
        logits = torch.rand(B, T, self.vocab_size) * 2 - 1
        return namedtuple('Output', ['logits'])(logits)


if __name__ == '__main__':
    # Test the SSP implementation
    vocab_size = 10
    target_model = FakeModel(vocab_size)
    draft_model = FakeModel(vocab_size)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    print(ssp(target_model, draft_model, 10, input_ids))
