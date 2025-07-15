import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import random
import itertools
import json

from transformer_lens import HookedTransformer

from data_utils import process_dataset, read_jsonl
from circuit_eval_utils import get_completeness_score, identified_circuit, complete_circuit

def one_greedy_search(model, data, C, M, batch_size=4):
    all_heads = list(itertools.chain.from_iterable(C["head_circuit"].values()))

    best_K, best_diff = [], 0.0
    K, diff = [], 0.0
    
    for i in range(5):
        candidates = list(set(all_heads).difference(K))

        V = random.sample(candidates, 5)
        flag = False
        for v in V:
            score_A, score_B = get_completeness_score(model, data, C=C, K=K+[v], M=M, batch_size=batch_size)
            print(K+[v], "a: {}".format(score_A), "b: {}".format(score_B), "abs(a-b): {}".format(abs(score_A-score_B)))
            if abs(score_A-score_B) > best_diff:
                best_diff = abs(score_A-score_B)
                best_K = K + [v]
                flag = True

        if flag == False:
            break

        K = best_K
        diff = best_diff

    return best_K, best_diff

def greedy_search():
    model_name = "google/gemma-2-9b"
    model = HookedTransformer.from_pretrained(model_name, device="cuda:0")
    model.set_ungroup_grouped_query_attention(True)

    setting, nmax, offset, n_icl_examples = "setting1", 9, 1, 16
    filename = f"../data/addition/{setting}/addition_nmax{nmax}_offset{offset}.jsonl"
    data = read_jsonl(filename)
    processed_data = process_dataset(data, n_icl_examples=n_icl_examples, offset=offset)
    data = processed_data

    C = identified_circuit(n_shot=n_icl_examples)
    M = complete_circuit(n_shot=n_icl_examples)

    random.seed(202505)

    result_dict = {}
    for i in range(10):
        K, diff = one_greedy_search(model, data, C, M, batch_size=1)
        print("rollout {}:".format(i), K, diff)
        result_dict[i] = K

    print(result_dict)

    print("====")
    for i in range(10):
        K = result_dict[i]
        score_A, score_B = get_completeness_score(model, data, C=C, K=K, M=M, batch_size=1)
        print("rollout {}:".format(i), K, score_A, score_B, abs(score_A-score_B))
        result_dict[i] = [K, score_A, score_B, abs(score_A-score_B)]

    print(result_dict)

    with open("results/gemma_2_9b/completeness_greedy_search_16shot.json", "w") as f:
        json.dump(result_dict, f)

if __name__ == "__main__":
    greedy_search()