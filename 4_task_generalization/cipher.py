import os
import tqdm
import pandas as pd
import itertools
import random
import json

import torch as t
t.set_grad_enabled(False)

from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from torch import Tensor
from functools import partial

data_dir = "../data/cipher/"
max_len = 1

def read_jsonl(filename):
    with open(filename) as fin:
        lines = fin.readlines()
    json_lines = [json.loads(line) for line in lines]
    return json_lines

def process_dataset(datapoints, n_icl_examples, offset):
    new_datapoints = []
    
    for dp in datapoints:
        normal_input = []
        contrast_input = []

        for j in range(n_icl_examples):
            a, b = dp["demonstration"][j]
            a = " " + " ".join(a)
            b = " " + " ".join(b)
            normal_input.append("{} ->{}".format(a,a))
            contrast_input.append("{} ->{}".format(a,b))
            
        normal_input = "\n".join(normal_input)
        contrast_input = "\n".join(contrast_input)

        if n_icl_examples > 0:
            normal_input += "\n"
            contrast_input += "\n"

        a, b = dp["inference"]
        a = " " + " ".join(a)
        b = " " + " ".join(b)
        normal_input += "{} ->".format(a)
        normal_output = a
        contrast_input += "{} ->".format(a)
        contrast_output = b
        
        new_datapoints.append({
            "normal_input": normal_input, "normal_output": normal_output,
            "contrast_input": contrast_input, "contrast_output": contrast_output
        })

    return new_datapoints


def head_patch_hook(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_id: int,
    cache: ActivationCache,
):
   orig_head_vector[:, :, head_id, :] = cache[hook.name][:, :, head_id, :]


def is_correct(logits, label):
    decoding_len = max_len
    pos_token_logits = logits[:, -decoding_len-1:-1, :]
    pred = pos_token_logits.argmax(dim=-1)
    return int(t.equal(pred, label))

def run_one_offset(model, datapoints, heads_to_ablate=[]):
    n_correct_0_0, n_correct_0_offset = 0, 0
    n_correct_offset_0, n_correct_offset_offset = 0, 0

    for dp in tqdm.tqdm(datapoints):
        assert len(dp["contrast_output"]) == len(dp["normal_output"])
        
        contrast_output_ids = model.to_tokens(dp["contrast_output"], prepend_bos=False)
        normal_output_ids = model.to_tokens(dp["normal_output"], prepend_bos=False)

        logits, clean_cache = model.run_with_cache(
            input=dp["normal_input"]+dp["normal_output"],
            names_filter = lambda name: name.endswith("z")
        )
        n_correct_0_0 += is_correct(logits, normal_output_ids)

        # normal prompt, contrast acc
        logits = model(
            input=dp["normal_input"]+dp["contrast_output"],
            return_type="logits"
        )
        n_correct_0_offset += is_correct(logits, contrast_output_ids)

        # prepare hooks
        fwd_hooks = [(
            utils.get_act_name("z", head_layer),
            partial(head_patch_hook, head_id=head_idx, cache=clean_cache)
        ) for head_layer, head_idx in heads_to_ablate]

        # contrast prompt, normal acc
        logits = model.run_with_hooks(
            input=dp["contrast_input"]+dp["normal_output"],
            fwd_hooks=fwd_hooks
        )
        n_correct_offset_0 += is_correct(logits, normal_output_ids)

        # contrast prompt, contrast acc
        logits = model.run_with_hooks(
            input=dp["contrast_input"]+dp["contrast_output"],
            fwd_hooks=fwd_hooks
        )
        n_correct_offset_offset += is_correct(logits, contrast_output_ids)

    acc_0_0 = n_correct_0_0 / len(datapoints)
    acc_0_offset = n_correct_0_offset / len(datapoints)
    print(f"base prompt - base acc: {acc_0_0:.4f}, contrast acc: {acc_0_offset:.4f}")
    acc_offset_0 = n_correct_offset_0 / len(datapoints)
    acc_offset_offset = n_correct_offset_offset / len(datapoints)
    if len(heads_to_ablate) > 0:
        print(f"Disabling {heads_to_ablate}")
    print(f"contrast prompt - base acc: {acc_offset_0:.4f}, contrast acc: {acc_offset_offset:.4f}")
    
    return acc_0_0, acc_0_offset, acc_offset_0, acc_offset_offset


def main():
    device = "cuda:0"
    model_name = "google/gemma-2-9b"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.set_ungroup_grouped_query_attention(True)

    FIHeads = [(39, 12), (39, 7), (36, 7), (32, 6), (32, 1), (25, 13)] # + [(24, 9), (32, 4), (28, 6)]

    random.seed(42)
    AllHeads = list(itertools.product(range(0, model.cfg.n_layers), range(0, model.cfg.n_heads)))
    RandomHeads = [random.sample(AllHeads, len(FIHeads)) for _ in range(5)]
    print(RandomHeads)

    results = {}
    n_icl_examples = 16
    
    for offset in range(-12, 14, 1):

        results[(offset, n_icl_examples)] = {}
        filename = os.path.join(data_dir, "normal", f"cipher_maxlen{max_len}_offset{offset}.jsonl")
        datapoints = read_jsonl(filename)
        datapoints = process_dataset(datapoints, n_icl_examples=n_icl_examples, offset=offset)
        print(datapoints[0])
        acc_0_0, acc_0_offset, acc_offset_0, acc_offset_offset = run_one_offset(model, datapoints)
        results[(offset, n_icl_examples)]["base"] = (acc_0_0, acc_0_offset)
        results[(offset, n_icl_examples)]["contrast"] = (acc_offset_0, acc_offset_offset)

        acc_0_0, acc_0_offset, acc_offset_0, acc_offset_offset = run_one_offset(model, datapoints, heads_to_ablate=FIHeads)
        results[(offset, n_icl_examples)]["contrast_fih"] = (acc_offset_0, acc_offset_offset)

        for i, heads in enumerate(RandomHeads):
            _, _, acc_offset_0, acc_offset_offset = run_one_offset(model, datapoints, heads_to_ablate=heads)
            results[(offset, n_icl_examples)]["contrast_rh{}".format(i)] = (acc_offset_0, acc_offset_offset)

    print(results)
    print("{} FI heads".format(len(FIHeads)))

        
if __name__ == "__main__":
    main()
