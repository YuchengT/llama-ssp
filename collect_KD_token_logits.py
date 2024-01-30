import transformers
import torch
import datasets
import json
from tqdm import trange, tqdm
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM
)
from tensordict import TensorDict
import torchsnapshot

teacher_model_name = 'meta-llama/Llama-2-7b-chat-hf'
student_model_name = 'JackFram/llama-160m'
saved_model_path = f"/home/ubuntu/llama-ssp/KD_draft_models/{student_model_name}/lr_5e-07_512/ckpt_24"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

teacher_model = LlamaForCausalLM.from_pretrained(teacher_model_name, device_map='balanced')
#student_model = LlamaForCausalLM.from_pretrained(student_model_name, device_map='balanced')
student_model = LlamaForCausalLM.from_pretrained(saved_model_path, device_map='balanced')

with open("/home/ubuntu/llama-ssp/results/llama7b_llama1.1b_analysis.json", 'r') as json_file:
    json_list = list(json_file)

dataset = json.loads(json_list[0])

logits_dict = {}
dict_path = '/home/ubuntu/llama-ssp/results/mbpp_llama7b_160m_ckpt_24_logits'

def main():
    for i in trange(100):
        data = dataset[i]
        prompt_len = data['prompt_len']
        input_ids = torch.tensor(data['token_ids'], device='cuda').view(1,-1)
        teacher_logits = teacher_model(input_ids).get('logits')[:, prompt_len:, :].half()
        student_logits = student_model(input_ids).get('logits')[:, prompt_len:, :].half()
        logits_dict[str(i)] = torch.cat([teacher_logits, student_logits], dim=0).detach().cpu()


    logits_tensor_dict = TensorDict(logits_dict, [])
    state = {"state": logits_tensor_dict}
    snapshot = torchsnapshot.Snapshot.take(app_state=state, path=dict_path)


if __name__ == '__main__':
    main()