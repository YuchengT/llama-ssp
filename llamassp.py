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
from transformers import LlamaTokenizer, AutoTokenizer
from termcolor import colored
torch.manual_seed(1339)

MAX_NEW_TOKENS = 96
codellama7b_name = 'codellama/CodeLlama-7b-Python-hf'
codellama13b_name = 'codellama/CodeLlama-13b-Python-hf'
codellama34b_name = 'codellama/CodeLlama-34b-Python-hf'
llama7b_name = 'decapoda-research/llama-7b-hf'
llama13b_name = 'decapoda-research/llama-13b-hf'
llama30b_name = 'decapoda-research/llama-30b-hf'
llama65b_name = 'decapoda-research/llama-65b-hf'
tinyllama_name = 'TinyLlama/TinyLlama-1.1B-Chat-v0.6'
#tinyllama_awq_name = "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ"
tinyllama_awq_name = "TheBloke/TinyLlama-1.1B-1T-OpenOrca-AWQ"
batch_size = 1

texts_1 = [
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

completion_texts = ['Once upon a time, in the vibrant, mysterious world beneath the waves, there lived an extraordinary shark named Sharky. However, Sharky was no ordinary shark; he was a tooth-brushing superhero with a heart as big as the ocean itself. Sharky\'s'\
                    'story began in a hidden corner of the Great Barrier Reef, where he was born to a family of toothless sharks. Unlike his siblings, who relied on their razor-sharp teeth for survival, Sharky was born with a unique gift – a giant, neon-blue toothbrush that appeared out of thin air whenever he needed it.']
                    #'As Sharky grew, he realized that his purpose was to ensure the dental health of all sea creatures. With his trusty toothbrush by his side, he began patrolling the reef, searching for aquatic friends in need of dental care. His mission was clear: to spread awareness about proper oral hygiene in the underwater world.'\
                    #'One sunny morning, Sharky swam to the heart of the reef, where he discovered a group of timid clownfish hiding in the anemones. Their mouths were filled with food particles and algae, a clear sign of neglecting their dental hygiene. Sharky approached them with a warm smile, his neon-blue toothbrush in hand.']
                    #'"Hello there, little friends!\" Sharky greeted them, his voice as soothing as the gentle ocean currents. "I\'m Sharky, the tooth-brushing superhero. How about I show you how to keep your teeth sparkly clean?"'\
                    #'The clownfish, initially startled by the sight of a shark, soon realized that Sharky was there to help. They gathered around, eager to learn. Sharky demonstrated the proper way to brush their tiny teeth, teaching them to use gentle, circular motions. The clownfish, mesmerized by Sharky\'s expertise, eagerly followed his lead.']
                    #'As the days went by, Sharky\'s reputation as the tooth-brushing superhero of the sea spread far and wide. Creatures from all corners of the ocean sought his guidance. Sea turtles, seahorses, and even the elusive giant squids became his devoted students. His undersea seminars were a sensation, attracting a diverse crowd of marine life eager to master the art of dental care.'\
                    #'Sharky\'s schedule grew busier with each passing day, but he embraced the challenge with a heart full of compassion. He also established a toothpaste factory, producing a special toothpaste with flavors like seaweed and shrimp that sea creatures adored. This initiative ensured that even those without a toothbrush could maintain healthy oral hygiene.']

#tokenizer = LlamaTokenizer.from_pretrained(llama7b_name)
tokenizer = AutoTokenizer.from_pretrained(codellama7b_name)

free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_mem = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

n_gpus = torch.cuda.device_count()


def max_memory(gpus, starting_gpu=0):
    return {i: max_mem for i in range(starting_gpu, gpus)}

def load_prompts():
    prompts = []
    with open("/home/ubuntu/human-eval/data/HumanEval.jsonl", 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        prompts.append(json.loads(json_str)['prompt'])

    return prompts


def time_model(model):
    model.eval()
    with torch.no_grad():
        # time the first run
        texts = test_texts
        input_ids = tokenizer(texts[0], return_tensors="pt").input_ids
        input_ids = torch.stack([input_ids[0]] * batch_size).to('cuda:0')
        generated_ids = sample_model(model, input_ids, MAX_NEW_TOKENS)

        start_time = time.time()
        nb_tokens = 0
        for text in texts[1:]:
            #print("Completing text:", text)
            intermediate_time = time.time()
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            input_ids = torch.stack([input_ids[0]] * batch_size).to('cuda:0')
            generated_ids = sample_model(model, input_ids, MAX_NEW_TOKENS)
            nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
            #print("Completion: ", tokenizer.decode(
            #    generated_ids[0], skip_special_tokens=True))
            print("Time: {:.2f}s".format(time.time() - intermediate_time))
            print("========\n")
        ms_per_token = (time.time() - start_time)*1000 / nb_tokens
    return generated_ids, ms_per_token


def print_results(tokens_s, outputs, name='Noname'):
    print("Results for ", name)
    print(f"Ms per token: {tokens_s:.2f}ms")
    print("========\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("========\n")


models_params = {
    'tiny_awq':{'model_name': tinyllama_awq_name, 'device_map':{'':0},
                'max_memory': max_memory(8),
                'load_in_8bit': False, 'load_in_4bit': False},
    'tiny_4bit':{'model_name': tinyllama_name, 'device_map':{'':0},
                'max_memory': max_memory(8),
                'load_in_8bit': False, 'load_in_4bit': True},
    'tiny_8bit':{'model_name': tinyllama_name, 'device_map':{'':0},
                'max_memory': max_memory(8),
                'load_in_8bit': True, 'load_in_4bit': False},
    'tiny':{'model_name': tinyllama_name, 'device_map':{'':0},
                'max_memory': max_memory(8),
                'load_in_8bit': False, 'load_in_4bit': False},
    '7B_4bit': {'model_name': llama7b_name, 'device_map':{'': 0},
                'max_memory': max_memory(1),
                'load_in_8bit': False, 'load_in_4bit': True},
    '7B_8bit': {'model_name': llama7b_name, 'device_map': 'balanced',
                'max_memory': max_memory(2),
                'load_in_8bit': True},
    '7B_8bit_4': {'model_name': llama7b_name,
                  'max_memory': max_memory(4),
                  'load_in_8bit': True},
    '7B': {'model_name': llama7b_name,
           'max_memory': max_memory(1),
           'load_in_8bit': False},
    '7B_8': {'model_name': llama7b_name,
             'max_memory': max_memory(8),
             'load_in_8bit': False},
    '13B_8bit': {'model_name': llama13b_name, 'device_map': 'balanced',
                 'max_memory': max_memory(2),
                 'load_in_8bit': True},
    '13B_code_8bit': {'model_name': codellama13b_name, 'device_map': 'balanced',
                 'max_memory': max_memory(8),
                 'load_in_8bit': True},
    '13B_code': {'model_name': codellama13b_name, 'device_map': 'balanced',
                 'max_memory': max_memory(8),
                 'load_in_8bit': False},
    '13B': {'model_name': llama13b_name, 'device_map': 'balanced',
            'max_memory': max_memory(2),
            'load_in_8bit': False, },
    '30B_8bit': {'model_name': llama30b_name, 'device_map': 'balanced',
                 'max_memory': max_memory(2),
                 'load_in_8bit': True, },
    '30B': {'model_name': llama30b_name,
            'max_memory': max_memory(4),
            'load_in_8bit': False},
    '34B_code_8bit':{'model_name': codellama34b_name, 'device_map': 'balanced',
                'max_memory': max_memory(8),
                'load_in_8bit': True},
    '65B_8bit': {'model_name': llama65b_name,
                 'max_memory': max_memory(4),
                 'load_in_8bit': True},
    '65B': {'model_name': llama65b_name,
            'max_memory': max_memory(8),
            'load_in_8bit': False},
    '65B_v2': {'model_name': f"{os.getenv('HOME')}/data/hf-weights/65B",
               'max_memory': max_memory(8),
               'load_in_8bit': False},
}


def time_ssp(target_name, draft_name, K=4):
    draft_model = create_model(**models_params[draft_name])
    target_model = create_model(**models_params[target_name])
    nb_tokens = 0
    # Warmup
    texts = load_prompts()[:10]
    input_ids = tokenizer(texts[0], return_tensors="pt").input_ids
    input_ids = torch.stack(
        [input_ids[0]] * batch_size).to(draft_model.device)
    generated_ids = ssp(target_model,
                        draft_model,
                        MAX_NEW_TOKENS,
                        input_ids, K=K)

    start_time = time.time()
    for text in texts[1:]:
        print("Completing text:", text)
        intermediate_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack(
            [input_ids[0]] * batch_size).to(draft_model.device)
        generated_ids = ssp(target_model,
                            draft_model,
                            MAX_NEW_TOKENS,
                            input_ids, K=K)
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
        print("Completion: ", tokenizer.decode(
            generated_ids[0], skip_special_tokens=True))
        print("Time: {:.2f}s".format(time.time() - intermediate_time))
        print("========\n")
    ms_per_token = (time.time() - start_time)*1000 / nb_tokens
    return generated_ids, ms_per_token


def print_speeds(speeds):
    print("Speeds:")
    for model_name, tokens_s in speeds.items():
        print('-'*20)
        print(f"{model_name} |  {tokens_s:.2f}ms")
    print('-'*20)


def models_raw_speed():
    speeds = {}
    del models_params['7B'], models_params['13B'], models_params['30B']
    for model_name, params in sorted(models_params.items()):
        print(f"Testing {model_name}")
        print('-'*20)
        model = create_model(**params)
        outputs, tokens_s = time_model(model)
        speeds[model_name] = tokens_s
        print_results(tokens_s, outputs, model_name)
        del model
        torch.cuda.empty_cache()
        print_speeds(speeds)
    draft_name = '7B_8bit'
    target_name = '65B_8bit'
    print(f"Testing SSP {draft_name} / {target_name}")
    tokens_s, outputs = time_ssp(draft_name, target_name)
    speeds[f"{draft_name} / {target_name}"] = tokens_s
    print(speeds)


def show_comparative_speeds(text, model, draft_model, target_tempurature, draft_tempurature, eval=False):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(colored("=> Regular sampling with target model",
                  attrs=['bold']))
    sys.stdout.write(text)
    start_time = time.time()
    target_output_ids = sample_model(model, input_ids, MAX_NEW_TOKENS, display=True, temperature=target_tempurature)
    sample_model_time = time.time() - start_time
    print("\nTime: "
          + colored(f"{sample_model_time:.2f}s", 'red', attrs=['bold']))
    print(colored(
        "=> Speculative sampling with target model helped by draft model",
        attrs=['bold']))
    sys.stdout.write(text)
    start_time = time.time()
    ssp_output_ids, total_count_accepted, total_count_generated = ssp(model, draft_model, MAX_NEW_TOKENS,
        input_ids, target_tempurature, draft_tempurature, K=8, display=True)
    ssp_time = time.time() - start_time
    #pred_results = code_bert_score.score(cands=[tokenizer.decode(ssp_output_ids[0], skip_special_tokens=True)], refs=[tokenizer.decode(target_output_ids[0], skip_special_tokens=True)], lang='python')
    print("\nTime: "
          + colored(f"{ssp_time:.2f}s", 'green', attrs=['bold']))
    print("Acceptance Rate: "
          + colored(f"{total_count_accepted / total_count_generated}", 'blue', attrs=['bold']))
    return total_count_accepted / total_count_generated, sample_model_time, ssp_time

def generate_logits_for_incremental_seq(model, draft_model, text):
    target_logits_ls = []
    draft_logits_ls = []
    all_input_ids = tokenizer(text, return_tensors="pt").input_ids
    all_input_tokens = tokenizer.tokenize(text)
    print(all_input_tokens[:20])
    current_input_ids = all_input_ids[:, :20]
    while current_input_ids.size(1) < 50:
        target_outputs = model(current_input_ids)
        target_logits_ls.append(target_outputs.logits[:, -1, :])
        draft_outputs = draft_model(current_input_ids)
        draft_logits_ls.append(draft_outputs.logits[:, -1, :])
        print(f'{current_input_ids.size(1)}, {all_input_tokens[current_input_ids.size(1)]}')
        current_input_ids = torch.cat([current_input_ids, all_input_ids[:, current_input_ids.size(1)].unsqueeze(-1)], dim=-1)
        
    return target_logits_ls, draft_logits_ls

def generate_logits_for_code_completion(model, draft_model, text, target_tempurature):
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
        
    return target_logits_ls, draft_logits_ls

TEMPURATURE_LOW = 0.1
TEMPURATURE_HIGH = 1
def analyze_tempurature_relation(target_logits, draft_logits, target_tempurature):
    target_logits = target_logits.detach().numpy()
    draft_logits = draft_logits.detach().numpy()
    def distribution_norm(draft_tempurature):
        return scipy.linalg.norm(scipy.special.softmax(target_logits / target_tempurature) - \
                                 scipy.special.softmax(draft_logits * draft_tempurature), ord=1)
    res = scipy.optimize.minimize_scalar(distribution_norm)
    wo_opt = scipy.linalg.norm(scipy.special.softmax(target_logits / target_tempurature) - \
                                 scipy.special.softmax(draft_logits / target_tempurature), ord=1)
    return 1 / res.x, res.fun, wo_opt


def find_optimal_draft_tempurature(model, draft_model, text, n_partition):
    target_temps = np.linspace(TEMPURATURE_LOW, TEMPURATURE_HIGH, n_partition)
    tempuarture_matrix = np.zeros((n_partition, 32))
    norm_matrix = np.zeros((n_partition, 32))
    wo_opt_matrix = np.zeros((n_partition, 32))
    for i, target_temp in enumerate(target_temps):
        target_logits_ls, draft_logits_ls = generate_logits_for_code_completion(model, draft_model, text, target_temp)
        for j, (target_logits, draft_logits) in enumerate(zip(target_logits_ls, draft_logits_ls)):
            tempuarture_matrix[i,j], norm_matrix[i,j], wo_opt_matrix[i,j] = analyze_tempurature_relation(target_logits, draft_logits, target_temp)
        
    return tempuarture_matrix, norm_matrix, wo_opt_matrix


def create_argument_parser():
    """
    Create a parser for the command-line arguments, with 'compare', 'latency'
    and 'eval' subcommands
    """
    parser = argparse.ArgumentParser(
        description='Test speeds of Llama models with regular sampling and speculative sampling: measure their latency, compare their speed, and evaluate their performance on a simple task.')
    # add argument to set log level
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='verbose output')

    subparsers = parser.add_subparsers(dest='subcommand')
    compare_parser = subparsers.add_parser(
        'compare', help='Compare the speed of a given model (target model) alone, and with speculative sampling with another model (draft model)')
    compare_parser.add_argument('model', help='Name of target model')
    compare_parser.add_argument('draft', help='Draft model')

    temp_relation_parser = subparsers.add_parser(
        'temp_relation', help='Compare the speed of a given model (target model) alone, and with speculative sampling with another model (draft model)')
    temp_relation_parser.add_argument('model', help='Name of target model')
    temp_relation_parser.add_argument('draft', help='Draft model')

    latency_parser = subparsers.add_parser(
        'latency', help='Measure model latency in ms per token')
    latency_parser.add_argument('model', help='Name of model')
    latency_parser.add_argument(
        '--draft', help='Draft model; if specified, will measure the latency of speculative sampling with the draft model rather than the regular latency')

    eval_parser = subparsers.add_parser(
        'eval', help='evaluate a model')
    eval_parser.add_argument('model', help='model to use')
    eval_parser.add_argument(
        '--draft', help='Draft model; if specified, will evaluate the model with speculative sampling with the draft model rather than the regular model')
    eval_parser.add_argument('--seed', type=int, default=1338,
                             help='Seed for randomly creating the eval prompts')
    eval_parser.add_argument('--nb-prompts', type=int, default=1000,
                             help='Number of eval prompts to create')
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        # set log level to debug
        logging.basicConfig(level=logging.DEBUG)

    if args.subcommand == 'compare':
        model, model_device_map = create_model(**models_params[args.model])
        draft_model, draft_model_device_map = create_model(**models_params[args.draft])
        print(model_device_map)
        print(draft_model_device_map)
        print("Warming up")
        #ssp(model, draft_model, MAX_NEW_TOKENS,
        #    tokenizer(texts_1[0], return_tensors="pt").input_ids, K=4)
        print(
            f"Comparing {args.model} model regular sampling and {args.model} SSp with {args.draft} draft model\n====\n")
        # Read from stdin until EOF
        #while True:
        #    try:
        #        sys.stdout.write("> ")
        #        sys.stdout.flush()
        #        text = input()
        #    except EOFError:
        #        break
        results = []
        text_prompts = load_prompts()
        target_tempurature = 0.5
        draft_tempurature = 0.5
        for text in text_prompts[125:]:
            acceptance_rate, sample_model_time, ssp_time = show_comparative_speeds(text, model, draft_model, target_tempurature, draft_tempurature, eval=True)
            #print(pred_results)
            result_list = [acceptance_rate, sample_model_time, ssp_time]
            results.append(result_list)
            np.savetxt("/home/ubuntu/llama-ssp/results/human_eval_results_K=8_readjusted_acceptance.txt", np.asarray(results))

    elif args.subcommand == 'temp_relation':
        model, model_device_map = create_model(**models_params[args.model])
        draft_model, draft_model_device_map = create_model(**models_params[args.draft])
        print(model_device_map)
        print(draft_model_device_map)
        print("Warming up")
        #ssp(model, draft_model, MAX_NEW_TOKENS,
        #    tokenizer(texts_1[0], return_tensors="pt").input_ids, K=4)
        print(
            f"Find best draft tempurature for target model {args.model} and draft model {args.draft}\n====\n")
        # Read from stdin until EOF
        #while True:
        #    try:
        #        sys.stdout.write("> ")
        #        sys.stdout.flush()
        #        text = input()
        #    except EOFError:
        #        break
        text_prompts = load_prompts()
        n_examples = 20
        n_temps = 10
        t_mts = np.zeros((n_examples * n_temps,32))
        n_mts = np.zeros((n_examples * n_temps,32))
        wo_mts = np.zeros((n_examples * n_temps,32))
        for i, text in enumerate(text_prompts[:n_examples]):
            print(text)
            t_mts[i * n_temps : (i+1) * n_temps], n_mts[i * n_temps : (i+1) * n_temps], wo_mts[i * n_temps : (i+1) * n_temps] = find_optimal_draft_tempurature(model, draft_model, text, 10)
            
            np.savetxt("/home/ubuntu/llama-ssp/results/optimal_draft_tempurature.txt", t_mts)
            np.savetxt("/home/ubuntu/llama-ssp/results/optimal_draft_difference.txt", n_mts)
            np.savetxt("/home/ubuntu/llama-ssp/results/optimal_draft_difference_wo_optimization.txt", wo_mts)

            
    elif (args.subcommand == 'latency' and args.draft):
        print(f"Testing {args.model} with draft {args.draft}")
        print('-'*20)
        gen_ids, ms_per_token = time_ssp(args.model, args.draft)
        print_results(ms_per_token, gen_ids, args.model)

    elif (args.subcommand == 'latency'):
        print(f"Testing {args.model}")
        print('-'*20)
        model = create_model(**models_params[args.model])
        gen_ids, ms_per_token = time_model(model)
        print_results(ms_per_token, gen_ids, args.model)

    elif (args.subcommand == 'eval'):
        print(f"Eval of {args.model} on multiplication task (seed {args.seed})"
              + (f" with draft {args.draft}" if args.draft else ""))
        print('-'*20)
        model = create_model(**models_params[args.model])
        if args.draft:
            draft_model = create_model(**models_params[args.draft])
        else:
            draft_model = None
        results = evals.measure_model_score(
            model, tokenizer, args.nb_prompts, args.seed, draft_model)
        evals.print_results(results, args.model, args.draft)

    else:
        # show usage
        parser.print_help()
