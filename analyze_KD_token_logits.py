import torch
import json
from tqdm import trange, tqdm
from tensordict import TensorDict
import torchsnapshot

def find_error_tokens(t_logits, s_logits):
    # find the index of teacher token 
    argmax_t = torch.argmax(t_logits, dim=-1).squeeze()
    argmax_s = torch.argmax(s_logits, dim=-1).squeeze()

    return (argmax_t != argmax_s)

def find_nonconcentrated_tokens(t_logits, s_logits):
    t_nucleus_size = find_nucleus_size(t_logits)
    s_nucleus_size = find_nucleus_size(s_logits)

    return (s_nucleus_size >= 5 * t_nucleus_size)


def find_nucleus_size(logits, tempurature=1.0, p_threshold=0.9):
    probs = torch.softmax(logits / tempurature, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = (cum_sum_probs < p_threshold).to(torch.int)
    nucleus_size = torch.sum(nucleus, dim=-1) + 1
    #top_p_probs = sorted_probs[:nucleus_size]
    #top_p_inds = indices[:nucleus_size]

    return nucleus_size

def load_logits():
    tensordict = TensorDict({}, [])
    target_state = {'state': tensordict}
    snapshot = torchsnapshot.Snapshot(path='/home/ubuntu/llama-ssp/results/mbpp_llama7b_160m_ckpt_24_logits')
    snapshot.restore(app_state=target_state)

    return tensordict

def compare_error_vs_nc_tokens(t_logits, s_logits):
    error_tokens = find_error_tokens(t_logits, s_logits).to(torch.int)
    nc_tokens = find_nonconcentrated_tokens(t_logits, s_logits).to(torch.int)

    nc_in_error = torch.sum(error_tokens * nc_tokens) / torch.sum(error_tokens)
    error_in_nc = torch.sum(error_tokens * nc_tokens) / torch.sum(nc_tokens)

    return nc_in_error, error_in_nc

def compare_ts_nucleus():
    num_abnormal_tokens = 0
    ts_nucleus_size = []
    logits_tensordict = load_logits()
    for i in trange(100):
        teacher_logits = logits_tensordict[str(i)][0].to(torch.float)
        student_logits = logits_tensordict[str(i)][1].to(torch.float)
        t_nucleus_size = find_nucleus_size(teacher_logits)
        s_nucleus_size = find_nucleus_size(student_logits)

        ts_nucleus_size.append(torch.cat([t_nucleus_size.unsqueeze(0), s_nucleus_size.unsqueeze(0)], dim=-1))

        num_abnormal_tokens += torch.sum((s_nucleus_size < t_nucleus_size).to(torch.int)) / t_nucleus_size.size(0)

    return num_abnormal_tokens, ts_nucleus_size

def compare_ts_error_vs_nc():
    nc_in_error_list = torch.zeros(100)
    error_in_nc_list = torch.zeros(100)
    logits_tensordict = load_logits()
    for i in trange(100):
        teacher_logits = logits_tensordict[str(i)][0].to(torch.float)
        student_logits = logits_tensordict[str(i)][1].to(torch.float)

        nc_in_error, error_in_nc = compare_error_vs_nc_tokens(teacher_logits, student_logits)
        nc_in_error_list[i] = nc_in_error
        error_in_nc_list[i] = error_in_nc

    return torch.concat([nc_in_error_list.unsqueeze(-1), error_in_nc_list.unsqueeze(-1)], dim=0)

if __name__ == '__main__':

    #error_vs_nc = compare_ts_error_vs_nc()
    num_abnormal_tokens, ts_nucleus_size = compare_ts_nucleus()
    print(num_abnormal_tokens)
    #print(torch.mean(error_vs_nc, dim=0))
    print(ts_nucleus_size[2])
    print(ts_nucleus_size[16])


