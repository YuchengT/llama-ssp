import numpy as np

def ssp_lower_bound(target_throughput, draft_throughput, draft_length, acceptance_rate, total_num_tokens):
    # calculate the total number of drafts when acceptance occurs in the most uneven way, where some drafts
    # are completely accepted while others are completely rejected.
    number_drafts = total_num_tokens * (1 - acceptance_rate) + total_num_tokens * acceptance_rate / draft_length

    #calculate time spent by target model and draft model individually in ssp.
    target_time = target_throughput * number_drafts
    draft_time = draft_throughput * number_drafts * draft_length

    # this is the baseline time if only target model is used to generate results.
    original_target_time = target_throughput * total_num_tokens

    # calculate the theoretic speed up when using ssp, here we assume sampling time = 0.
    speed_up = original_target_time / (target_time + draft_time)

    return speed_up

def ssp_upper_bound(target_throughput, draft_throughput, draft_length, acceptance_rate, total_num_tokens):
    # calculate the total number of drafts when acceptance occurs evenly in each draft
    number_drafts = total_num_tokens // (acceptance_rate * draft_length + 1)

    #calculate time spent by target model and draft model individually in ssp.
    target_time = target_throughput * number_drafts
    draft_time = draft_throughput * number_drafts * draft_length

    # this is the baseline time if only target model is used to generate results.
    original_target_time = target_throughput * total_num_tokens

    # calculate the theoretic speed up when using ssp, here we assume sampling time = 0.
    speed_up = original_target_time / (target_time + draft_time)

    return speed_up

def ssp_theoretical_range(target_throughput, draft_throughput, draft_length, acceptance_rate, total_num_tokens):
    lower_bound = ssp_lower_bound(target_throughput, draft_throughput, draft_length, acceptance_rate, total_num_tokens)
    upper_bound = ssp_upper_bound(target_throughput, draft_throughput, draft_length, acceptance_rate, total_num_tokens)

    return lower_bound, upper_bound

if __name__ == "__main__":
    #target_throughput = 386.10
    #draft_throughput = 213.66
    draft_length = 3
    total_num_tokens = 96

    # load the ssp experiment result.
    results = np.loadtxt("/home/ubuntu/llama-ssp/results/human_eval_results_2gpu_K=3.txt")

    # find the interpolation number mu such that log (observed) = mu * log (lower_bound) + (1 - mu) * log (upper_bound)
    interpolation_mu = []
    interpolation_float_mu = []

    for acceptance_rate, target_throughput, draft_throughput in results:
       lower_bound, upper_bound = ssp_theoretical_range(target_throughput * 1000 / 96, draft_throughput * 1000 / 96, draft_length, acceptance_rate, total_num_tokens)
       assert lower_bound < upper_bound
       mu = (np.log(target_throughput / draft_throughput) - np.log(upper_bound)) / (np.log(lower_bound) - np.log(upper_bound))
       interpolation_float_mu.append(mu)
       interpolation_mu.append(f"{mu:.2f}")

    #np.savetxt('/home/ubuntu/llama-ssp/results/interpolation_human_eval_results_2gpu_unbalanced_loading',interpolation_mu)
    print(interpolation_mu)
    print(np.mean(interpolation_float_mu))
