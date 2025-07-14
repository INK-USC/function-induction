import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch as t
from torch import Tensor
import torch.nn.functional as F

from transformer_lens import HookedTransformer, patching, ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

t.set_grad_enabled(False)

from data_utils import process_dataset, read_jsonl
from patching_utils import batched_get_path_patch_to_repr, batched_get_path_patch_to_head

from tqdm import trange
import os

def save_tensor(tensor, filename):
    t.save(tensor.cpu(), os.path.join("results", "mistral_7b", filename))

def main():
    os.makedirs(os.path.join("results", "mistral_7b"), exist_ok=True)
    
    model_name = "mistralai/Mistral-7B-v0.1"
    model = HookedTransformer.from_pretrained(model_name, device="cuda:0")
    model.set_ungroup_grouped_query_attention(True)

    setting, nmax, offset, n_icl_examples = "setting1", 9, 1, 4
    filename = f"../data/addition/{setting}/addition_nmax{nmax}_offset{offset}.jsonl"
    data = read_jsonl(filename)
    processed_data = process_dataset(data, n_icl_examples=n_icl_examples, offset=offset)

    # resid_post.31
    patched_logit_diff, normal_logit_diff, contrast_logit_diff = batched_get_path_patch_to_repr(
        [31], "resid_post", model, processed_data, begin_layer=0, batch_size=4)
    relative_patched_logit_diff = (patched_logit_diff - contrast_logit_diff) / (contrast_logit_diff  - normal_logit_diff)
    save_tensor(relative_patched_logit_diff, "resid_post_31.pt")

    # consolidation_heads
    consolidation_heads = [(31, 1), (31, 10)]
    for head in consolidation_heads:
        patched_logit_diff, normal_logit_diff, contrast_logit_diff = batched_get_path_patch_to_head(
            [head], "v", model, processed_data, begin_layer=0, batch_size=4)
        relative_patched_logit_diff = (patched_logit_diff - contrast_logit_diff) / (contrast_logit_diff  - normal_logit_diff)
        relative_patched_logit_diff[head[0]:, :] = 0.0
        save_tensor(relative_patched_logit_diff, "cons_h{}_{}_{}.pt".format(head[0], head[1], "v"))

    # induction_heads
    induction_heads = [(30, 3), (30, 4), (30, 8), (30, 10), (30, 2), (31, 2), (30, 18)]
    for head in induction_heads:
        patched_logit_diff, normal_logit_diff, contrast_logit_diff = batched_get_path_patch_to_head(
            [head], "v", model, processed_data, begin_layer=0, batch_size=4)
        relative_patched_logit_diff = (patched_logit_diff - contrast_logit_diff) / (contrast_logit_diff  - normal_logit_diff)
        relative_patched_logit_diff[head[0]:, :] = 0.0
        save_tensor(relative_patched_logit_diff, "ind_h{}_{}_{}.pt".format(head[0], head[1], "v"))

if __name__ == "__main__":
    main()