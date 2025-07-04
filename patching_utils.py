import torch as t
from torch import Tensor
import torch.nn.functional as F

from transformer_lens import HookedTransformer, patching, ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

t.set_grad_enabled(False)

from jaxtyping import Float, Int, Bool, Union
from typing import Literal, Callable
from functools import partial

from data_utils import process_dataset, read_jsonl

from tqdm import trange

### compute diffs ###
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    normal_answer: Float[Tensor, "batch"],
    contrast_answer: Float[Tensor, "batch"],
    per_prompt: bool = False
) -> Float[Tensor, "*batch"]:
    bbatch = logits.shape[0]
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    normal_logits: Float[Tensor, "batch"] = final_logits[t.arange(bbatch), normal_answer]
    contrast_logits: Float[Tensor, "batch"] = final_logits[t.arange(bbatch), contrast_answer]
    answer_logit_diff = normal_logits - contrast_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def path_patching_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    normal_answer: Float[Tensor, "batch"],
    contrast_answer: Float[Tensor, "batch"],
    contrast_logit_diff: float,
    normal_logit_diff: float,
) -> Float[Tensor, ""]:
    patched_logit_diff = logits_to_ave_logit_diff(logits, normal_answer, contrast_answer)
    return (patched_logit_diff - contrast_logit_diff) / (contrast_logit_diff  - normal_logit_diff)

### path patching helpers ###
def sender_head_patch_hook(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    layer_id: int,
    head_id: int,
    normal_cache: ActivationCache,
    contrast_cache: ActivationCache,
):
    orig_head_vector[...] = contrast_cache[hook.name][...]
    if hook.layer() == layer_id:
      orig_head_vector[:, :, head_id, :] = normal_cache[hook.name][:, :, head_id, :]

def sender_repr_patch_hook(
    orig_repr_vector: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    layer_id: int,
    normal_cache: ActivationCache,
    contrast_cache: ActivationCache,
):
    if hook.layer() == layer_id:
      orig_repr_vector[...] = normal_cache[hook.name][...]
    else:
      orig_repr_vector[...] = contrast_cache[hook.name][...]

def receiver_head_patch_hook(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_id: int,
    patched_cache: ActivationCache,
):
   orig_head_vector[:, :, head_id, :] = patched_cache[hook.name][:, :, head_id, :]

def receiver_repr_patch_hook(
    vector: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    patched_cache: ActivationCache,
):
    vector[...] = patched_cache[hook.name][...]

### path patching ###
    
def prepare_data_for_fwd(model: HookedTransformer, data: list[dict]):
    normal_input = [item["normal_input"] for item in data]
    contrast_input = [item["contrast_input"] for item in data]

    normal_answer = Tensor([model.to_tokens(item["normal_output"])[0][-1] for item in data]).long().to(model.cfg.device)
    contrast_answer = Tensor([model.to_tokens(item["contrast_output"])[0][-1] for item in data]).long().to(model.cfg.device)
    
    # Forward run with cache
    normal_logits, normal_cache = model.run_with_cache(normal_input)
    contrast_logits, contrast_cache = model.run_with_cache(contrast_input)

    return normal_input, contrast_input, normal_answer, contrast_answer, normal_cache, contrast_cache, normal_logits, contrast_logits

def get_path_patch_to_head(
    receiver_heads: list[(int, int)],
    receiver_input: Union[list[str], str],
    model: HookedTransformer,
    data: list[dict],
    begin_layer: int,
):
    device = model.cfg.device
    normal_input, contrast_input, normal_answer, contrast_answer, \
        normal_cache, contrast_cache, normal_logits, contrast_logits = prepare_data_for_fwd(model, data)

    patched_logits_diff = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device)
    normal_logits_diff = logits_to_ave_logit_diff(normal_logits, normal_answer, contrast_answer)
    contrast_logits_diff = logits_to_ave_logit_diff(contrast_logits, normal_answer, contrast_answer)

    latest_receiver_layer = max([item[0] for item in receiver_heads])
    layer_iter_end = min(latest_receiver_layer+1, model.cfg.n_layers)

    if isinstance(receiver_input,int):
        receiver_input = [receiver_input] * len(receiver_heads)
    
    for layer in trange(begin_layer, layer_iter_end, desc="layer"):
        for head in range(model.cfg.n_heads):
            tmp_hook = partial(sender_head_patch_hook, head_id=head, layer_id=layer, normal_cache=normal_cache, contrast_cache=contrast_cache)
            model.add_hook(lambda name: name.endswith("z"), tmp_hook, level=1)
            logits, patched_cache = model.run_with_cache(contrast_input, 
                                                         names_filter = lambda name: any([name.endswith(ri) for ri in set(receiver_input)]))
            model.reset_hooks()

            fwd_hooks = [(
                utils.get_act_name(receiver_input_type, head_layer),
                partial(receiver_head_patch_hook, head_id=head_idx, patched_cache=patched_cache)
            ) for (head_layer, head_idx), receiver_input_type in zip(receiver_heads, receiver_input)]
            
            logits = model.run_with_hooks(contrast_input, fwd_hooks=fwd_hooks)
            model.reset_hooks()

            patched_logits_diff[layer, head] = logits_to_ave_logit_diff(logits, normal_answer, contrast_answer)
     
    return patched_logits_diff, normal_logits_diff, contrast_logits_diff


def get_path_patch_to_repr(
    receiver_layers: list[int],
    receiver_input: str,
    model: HookedTransformer,
    data: list[dict],
    begin_layer: int,
):
    # preprocessing
    device = model.cfg.device
    normal_input, contrast_input, normal_answer, contrast_answer, \
        normal_cache, contrast_cache, normal_logits, contrast_logits = prepare_data_for_fwd(model, data)
    
    patched_logits_diff = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device)
    normal_logits_diff = logits_to_ave_logit_diff(normal_logits, normal_answer, contrast_answer)
    contrast_logits_diff = logits_to_ave_logit_diff(contrast_logits, normal_answer, contrast_answer)

    latest_receiver_layer = max(receiver_layers)
    layer_iter_end = min(latest_receiver_layer+1, model.cfg.n_layers)

    for layer in range(begin_layer, layer_iter_end):
        for head in range(model.cfg.n_heads):
            tmp_hook = partial(sender_head_patch_hook, head_id=head, layer_id=layer, normal_cache=normal_cache, contrast_cache=contrast_cache)
            model.add_hook(lambda name: name.endswith("z"), tmp_hook, level=1)

            logits, patched_cache = model.run_with_cache(contrast_input, names_filter = lambda name: name.endswith(receiver_input))
            model.reset_hooks()

            fwd_hooks = [(
                utils.get_act_name(receiver_input, l),
                partial(receiver_repr_patch_hook, patched_cache=patched_cache)
            ) for l in receiver_layers]

            logits = model.run_with_hooks(contrast_input, fwd_hooks=fwd_hooks)
            model.reset_hooks()

            patched_logits_diff[layer, head] = logits_to_ave_logit_diff(logits, normal_answer, contrast_answer)
    
    return patched_logits_diff, normal_logits_diff, contrast_logits_diff


def batched_get_path_patch_to_repr(
    receiver_layers: list[int],
    receiver_input: str,
    model: HookedTransformer,
    data: list[dict],
    begin_layer: int,
    batch_size: int
):
    agg_patched_logits_diff = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=model.cfg.device)
    agg_normal_logits_diff = 0.0
    agg_contrast_logits_diff = 0.0

    for st in trange(0, len(data), batch_size, desc="batch"):
        ed = st + batch_size
        batch = data[st: ed]
        patched_logits_diff, normal_logits_diff, contrast_logits_diff = get_path_patch_to_repr(
            receiver_layers, receiver_input, model, batch, begin_layer)

        agg_patched_logits_diff = agg_patched_logits_diff + patched_logits_diff * len(batch)
        agg_normal_logits_diff = agg_normal_logits_diff + normal_logits_diff  * len(batch)
        agg_contrast_logits_diff = agg_contrast_logits_diff + contrast_logits_diff * len(batch)

    return agg_patched_logits_diff, agg_normal_logits_diff, agg_contrast_logits_diff

def batched_get_path_patch_to_head(
    receiver_heads: list[(int, int)],
    receiver_input: Union[list[str], str],
    model: HookedTransformer,
    data: list[dict],
    begin_layer: int,
    batch_size: int,
):
    agg_patched_logits_diff = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=model.cfg.device)
    agg_normal_logits_diff = 0.0
    agg_contrast_logits_diff = 0.0
    
    for st in trange(0, len(data), batch_size, desc="batch"):
        ed = st + batch_size
        batch = data[st: ed]
        patched_logits_diff, normal_logits_diff, contrast_logits_diff = get_path_patch_to_head(
            receiver_heads, receiver_input, model, batch, begin_layer)
        
        agg_patched_logits_diff = agg_patched_logits_diff + patched_logits_diff * len(batch)
        agg_normal_logits_diff = agg_normal_logits_diff + normal_logits_diff  * len(batch)
        agg_contrast_logits_diff = agg_contrast_logits_diff + contrast_logits_diff * len(batch)

    return agg_patched_logits_diff, agg_normal_logits_diff, agg_contrast_logits_diff

def minitest():
    model_name = "google/gemma-2-9b"
    model = HookedTransformer.from_pretrained(model_name, device="cuda:0")
    model.set_ungroup_grouped_query_attention(True)

    setting, nmax, offset, n_icl_examples = "setting1", 9, 1, 4
    filename = f"data/addition/{setting}/addition_nmax{nmax}_offset{offset}.jsonl"
    data = read_jsonl(filename)
    processed_data = process_dataset(data, n_icl_examples=n_icl_examples, offset=offset)

    # load 4 example for preliminiary study
    data = processed_data[:4]

    patched_logit_diff, normal_logit_diff, contrast_logit_diff = get_path_patch_to_repr([41], "resid_post", model, data, 30)
    print(contrast_logit_diff)
    print(normal_logit_diff)
    relative_patched_logit_diff = (patched_logit_diff - contrast_logit_diff) / (contrast_logit_diff  - normal_logit_diff)
    print(relative_patched_logit_diff[20:, :])

    # in batches
    data = processed_data[:16]
    begin_layer = 30
    patched_logit_diff, normal_logit_diff, contrast_logit_diff = batched_get_path_patch_to_repr(
        [41], "resid_post", model, data, begin_layer, batch_size=4)
    relative_patched_logit_diff = (patched_logit_diff - contrast_logit_diff) / (contrast_logit_diff  - normal_logit_diff)
    relative_patched_logit_diff[:begin_layer, :] = 0.0
    print(relative_patched_logit_diff[20:, :])

    head_list_1 = [tuple(idx) for idx in t.nonzero(t.abs(relative_patched_logit_diff) > 0.05, as_tuple=False).tolist()]
    print(f"rel_diff > 0.05:\t {sorted(head_list_1, reverse=True)}")

    head_list_2 = [tuple(idx) for idx in t.nonzero(t.abs(relative_patched_logit_diff) > 0.02, as_tuple=False).tolist() if tuple(idx) not in head_list_1]
    print(f"0.05 > rel_diff > 0.02:\t {sorted(head_list_2, reverse=True)}")

if __name__ == "__main__":
    minitest()