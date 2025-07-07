import os
import torch
import tqdm
import pandas as pd
import itertools
import random

import torch as t
t.set_grad_enabled(False)

from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from torch import Tensor
from functools import partial


subcategories = ['high_school_government_and_politics', 'us_foreign_policy', 'high_school_psychology', 'sociology', 'high_school_geography', 'marketing', 'high_school_us_history']

data_dir = "../data/mmlu/"
choices = ["A", "B", "C", "D", "E"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()

def shift(s: str, offset: int):
    if offset == 0:
        return s
    t = ""
    for c in s:
        if c.isupper():
            t += chr(ord("A") + (ord(c) - ord("A") + offset) % 26)
        elif c.islower():
            t += chr(ord("a") + (ord(c) - ord("a") + offset) % 26)
        else:
            raise Exception(f"{c} in {s} is not A-Z a-z")
    return t

def format_example(df, idx, include_answer=True, offset=0):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n({}) {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " ({})\n\n".format(shift(df.iloc[idx, k + 1], offset))
    return prompt

def gen_prompt(train_df, subject, k=-1, offset=0):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i, offset=offset)
    return prompt

def head_patch_hook(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_id: int,
    cache: ActivationCache,
):
   orig_head_vector[:, :, head_id, :] = cache[hook.name][:, :, head_id, :]

alphabet_ids = [235280, 235305, 235288, 235299, 235291]

def run_one_subcategory(subject, model, offset=0, disable_heads=[]):
    dev_file = os.path.join(data_dir, "dev", f"{subject}_dev.csv")
    test_file = os.path.join(data_dir, "test", f"{subject}_test.csv")
    dev_df = pd.read_csv(dev_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    val_file = os.path.join(data_dir, "val", f"{subject}_val.csv")
    val_df = pd.read_csv(val_file, header=None)
    if len(val_df) > 11 and subject != "high_school_us_history":
        val_df = val_df.head(11) # use 5 shots in dev and 11 shots in val.
    elif subject == "high_school_us_history":
        val_df = val_df.head(3)
    dev_df = pd.concat([val_df, dev_df], ignore_index=True, sort=False)

    print(f"{subject} - dev: {len(dev_df)}, test: {len(test_df)}")

    n_correct_0_0, n_correct_0_offset = 0, 0
    n_correct_offset_0, n_correct_offset_offset = 0, 0
    train_prompt_0 = gen_prompt(dev_df, subject, k=-1, offset=0)
    train_prompt_offset = gen_prompt(dev_df, subject, k=-1, offset=offset)

    for i in tqdm.trange(len(test_df)):
        prompt_end = format_example(test_df, i, include_answer=False)

        # base
        prompt0 = train_prompt_0 + prompt_end + " ("
        logits, clean_cache = model.run_with_cache(
            input=prompt0,
            names_filter = lambda name: name.endswith("z")
        )
        logits = logits[:, -1, :].squeeze(0)
        alphabet_logits = logits[alphabet_ids]
        pred = choices[alphabet_logits.argmax().item()]
        n_correct_0_0 += (pred == shift(test_df.iloc[i, -1], 0))
        n_correct_0_offset += (pred == shift(test_df.iloc[i, -1], offset))

        fwd_hooks = [(
                utils.get_act_name("z", head_layer),
                partial(head_patch_hook, head_id=head_idx, cache=clean_cache)
            ) for head_layer, head_idx in disable_heads]

        # contrast
        prompt_offset = train_prompt_offset + prompt_end + " ("
        logits = model.run_with_hooks(
                input=prompt_offset,
                fwd_hooks=fwd_hooks,
        )
        logits = logits[:, -1, :].squeeze(0)
        alphabet_logits = logits[alphabet_ids]
        pred = choices[alphabet_logits.argmax().item()]
        n_correct_offset_0 += (pred == shift(test_df.iloc[i, -1], 0))
        n_correct_offset_offset += (pred == shift(test_df.iloc[i, -1], offset))
    
    acc_0_0 = n_correct_0_0 / len(test_df)
    acc_0_offset = n_correct_0_offset / len(test_df)
    print(f"{subject} - base prompt - base acc: {acc_0_0:.4f}, contrast acc: {acc_0_offset:.4f}")
    acc_offset_0 = n_correct_offset_0 / len(test_df)
    acc_offset_offset = n_correct_offset_offset / len(test_df)
    if len(disable_heads) > 0:
        print(f"Disabling {disable_heads}")
    print(f"{subject} - contrast prompt - base acc: {acc_offset_0:.4f}, contrast acc: {acc_offset_offset:.4f}")
    
    return acc_0_0, acc_0_offset, acc_offset_0, acc_offset_offset

def main():
    device = "cuda:0"
    model_name = "google/gemma-2-9b"
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype="float16")
    model.set_ungroup_grouped_query_attention(True)

    FIHeads = [(39, 12), (39, 7), (36, 7), (32, 6), (32, 1), (25, 13)] # + [(24, 9), (32, 4), (28, 6)]

    random.seed(42)
    AllHeads = list(itertools.product(range(0, model.cfg.n_layers), range(0, model.cfg.n_heads)))
    RandomHeads = [random.sample(AllHeads, len(FIHeads)) for _ in range(5)]
    print(RandomHeads)

    results = {}
    for subcategory in subcategories:
        results[subcategory] = {}
        acc_0_0, acc_0_offset, acc_offset_0, acc_offset_offset = run_one_subcategory(subcategory, model, offset=1)
        results[subcategory]["base"] = (acc_0_0, acc_0_offset)
        results[subcategory]["contrast"] = (acc_offset_0, acc_offset_offset)

        _, _, acc_offset_0, acc_offset_offset = run_one_subcategory(subcategory, model, offset=1, disable_heads=FIHeads)
        results[subcategory]["contrast_fih"] = (acc_offset_0, acc_offset_offset)

        for i, heads in enumerate(RandomHeads):
            _, _, acc_offset_0, acc_offset_offset = run_one_subcategory(subcategory, model, offset=1, disable_heads=heads)
            results[subcategory]["contrast_rh{}".format(i)] = (acc_offset_0, acc_offset_offset)

    
    print(results)
    print("{} FI heads".format(len(FIHeads)))

if __name__ == "__main__":
    main()
