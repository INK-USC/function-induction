import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import tqdm
import pandas as pd

from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint

from data_utils import read_jsonl, save_jsonl, process_dataset

from jaxtyping import Float, Int, Bool, Union
from torch import Tensor
from typing import Literal, Callable
from functools import partial
import random
import itertools
import json

torch.set_grad_enabled(False)

DIGIT_IDS_DICT = {
    "meta-llama/Llama-2-7b-hf": [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929],
    "mistralai/Mistral-7B-v0.1": [28734, 28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774],
    "google/gemma-2-9b": [235276, 235274, 235284, 235304, 235310, 235308, 235318, 235324, 235321, 235315],
}
NMAX_TO_LEN = {9: 1, 99: 2, 999: 3}

HEADS_TO_PATCH = [(39, 12), (39, 7), (36, 7), (32, 6), (32, 1), (28, 6), (25, 13), (24, 9)]

def head_patch_hook(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_id: int,
    cache: ActivationCache,
):
   orig_head_vector[:, :, head_id, :] = cache[hook.name][:, :, head_id, :]

def is_correct(logits, correct_answer, digit_ids):
    decoding_len = len(correct_answer)

    _output = ""
    for pos in range(-decoding_len-1, -1, 1):
        pos_token_logits = logits[:, pos, :].squeeze(0)
        digit_logits = pos_token_logits[digit_ids]
        _output += str(digit_logits.argmax().item())

    return int(correct_answer == _output)

def run_inference(model, datapoints, nmax, model_name, heads_to_ablate=[]):

    n_correct_0_0, n_correct_0_offset = 0, 0
    n_correct_offset_0, n_correct_offset_offset = 0, 0

    for dp in tqdm.tqdm(datapoints):
        assert len(dp["contrast_output"]) == len(dp["normal_output"])
        digit_ids = DIGIT_IDS_DICT[model_name]

        # normal prompt, normal acc
        logits, clean_cache = model.run_with_cache(
            input=dp["normal_input"]+dp["normal_output"],
            names_filter = lambda name: name.endswith("z")
        )
        n_correct_0_0 += is_correct(logits, dp["normal_output"], digit_ids)

        # normal prompt, contrast acc
        logits = model(
            input=dp["normal_input"]+dp["contrast_output"],
            return_type="logits"
        )
        n_correct_0_offset += is_correct(logits, dp["contrast_output"], digit_ids)

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
        n_correct_offset_0 += is_correct(logits, dp["normal_output"], digit_ids)

        # contrast prompt, contrast acc
        logits = model.run_with_hooks(
            input=dp["contrast_input"]+dp["contrast_output"],
            fwd_hooks=fwd_hooks
        )
        n_correct_offset_offset += is_correct(logits, dp["contrast_output"], digit_ids)

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
    model_name = "google/gemma-2-9b"
    model = HookedTransformer.from_pretrained(model_name, device="cuda:0")
    model.set_ungroup_grouped_query_attention(True)

    FIHeads = [(39, 12), (39, 7), (36, 7), (32, 6), (32, 1), (25, 13)] # + [(24, 9), (32, 4), (28, 6)]

    random.seed(42)
    AllHeads = list(itertools.product(range(0, model.cfg.n_layers), range(0, model.cfg.n_heads)))
    RandomHeads = [random.sample(AllHeads, len(FIHeads)) for _ in range(5)]
    print(RandomHeads)

    results = {}
    for setting in ["setting1"]:
        for nmax in [9]:
            for offset in range(-2, 3, 1):
                for n_icl_examples in [16]:
                    if offset == 0: continue

                    print(f"model_name: {model_name}, setting: {setting}, nmax: {nmax}, offset: {offset}, n_icl_examples: {n_icl_examples}")
                    
                    results[(offset, n_icl_examples)] = {}

                    filename = f"../data/addition/{setting}/addition_nmax{nmax}_offset{offset}.jsonl"
                    data = read_jsonl(filename)
                    processed_data = process_dataset(data, n_icl_examples=n_icl_examples, offset=offset)

                    acc_0_0, acc_0_offset, acc_offset_0, acc_offset_offset = run_inference(model, processed_data, nmax, model_name)
                    results[(offset, n_icl_examples)]["base"] = (acc_0_0, acc_0_offset)
                    results[(offset, n_icl_examples)]["contrast"] = (acc_offset_0, acc_offset_offset)

                    _, _, acc_offset_0, acc_offset_offset = run_inference(model, processed_data, nmax, model_name, heads_to_ablate=FIHeads)
                    results[(offset, n_icl_examples)]["contrast_fih"] = (acc_offset_0, acc_offset_offset)

                    for i, heads in enumerate(RandomHeads):
                        _, _, acc_offset_0, acc_offset_offset = run_inference(model, processed_data, nmax, model_name, heads_to_ablate=heads)
                        results[(offset, n_icl_examples)]["contrast_rh{}".format(i)] = (acc_offset_0, acc_offset_offset)

    print(results)
    print("{} FI heads".format(len(FIHeads)))


if __name__ == "__main__":
    main()