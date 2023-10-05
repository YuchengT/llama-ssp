from transformers import AutoTokenizer
import transformers
import torch
import time
import json

def load_prompts():
    prompts = []
    with open("/home/ubuntu/human-eval/data/HumanEval.jsonl", 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        prompts.append(json.loads(json_str)['prompt'])

    return prompts

model = "codellama/CodeLlama-7b-Python-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map={'':0},
)

texts = load_prompts()

inference_time = 0

for text in texts[:32]:
    stime = time.time()
    sequences = pipeline(
        text,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
    )
    inference_time += time.time() - stime
    print(inference_time)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")



