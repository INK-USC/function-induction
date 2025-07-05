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

from jaxtyping import Float, Int, Bool
from typing import Literal, Callable
from functools import partial

from tqdm import trange
import copy
import itertools

from patching_utils import logits_to_ave_logit_diff, prepare_data_for_fwd
from data_utils import process_dataset, read_jsonl

def get_heads_posns_to_keep(
    model: HookedTransformer,
    input_tokens: Float[Tensor, "batch seq"],
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, str],
    positions: dict[str, list[int]]
) -> dict[int, Bool[Tensor, "batch seq head"]]:
    results = {}
    batch, seq = input_tokens.shape
    for layer in range(model.cfg.n_layers):
        layer_result = t.zeros(batch, seq, model.cfg.n_heads, dtype=t.bool, device=model.cfg.device)
        for circuit_name, circuit_heads in circuit.items():
            for head in circuit_heads:
                if head[0] == layer:
                    for pos in positions[seq_pos_to_keep[circuit_name]]:
                        layer_result[:, pos, head[1]] = True # head[1] = head_index
        results[layer] = layer_result.bool()
    return results

def hook_fn_mask_z(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    posns_to_keep: dict[int, Bool[Tensor, "batch seq head"]],
    normal_cache: ActivationCache,
) -> Float[Tensor, "batch seq head d_head"]:
    layer = hook.layer()
    mask = posns_to_keep.unsqueeze(-1)
    z = t.where(mask, z, normal_cache[hook.name])
    return z

def add_ablation_hook(
    model: HookedTransformer,
    input_tokens: list[Tensor],
    head_circuit: dict[str, list[tuple[int, int]]],
    head_seq_pos_to_keep: dict[str, str],
    positions: dict[str, list[int]],
    is_permanent: bool = True,
    normal_cache: ActivationCache = None,
) -> HookedTransformer:

    model.reset_hooks(including_permanent=True)
    z_posns_to_keep = get_heads_posns_to_keep(model, input_tokens, head_circuit, head_seq_pos_to_keep, positions)
    for layer in range(model.cfg.n_layers):
      tmp_hook = partial(hook_fn_mask_z, posns_to_keep=z_posns_to_keep[layer], normal_cache=normal_cache)
      model.add_hook(utils.get_act_name("z", layer), tmp_hook, is_permanent=is_permanent)

    return model

def eval_circuit_one_batch(model, data, circuit):
    normal_input, contrast_input, normal_answer, contrast_answer, \
        normal_cache, contrast_cache, normal_logits, contrast_logits = prepare_data_for_fwd(model, data)
    
    del contrast_cache
    contrast_logits_no_circuit_diff = logits_to_ave_logit_diff(contrast_logits, normal_answer, contrast_answer)
    normal_logits_no_circuit_diff = logits_to_ave_logit_diff(normal_logits, normal_answer, contrast_answer)

    normal_input_tokens = model.to_tokens(normal_input)
    contrast_input_tokens = model.to_tokens(contrast_input)
    
    model = add_ablation_hook(model, input_tokens=normal_input_tokens, 
                                    head_circuit=circuit["head_circuit"],
                                    head_seq_pos_to_keep=circuit["head_seq_pos_to_keep"],
                                    positions=circuit["positions"],
                                    normal_cache=normal_cache)
    
    contrast_logits_with_circuit = model(contrast_input_tokens)
    contrast_logits_with_circuit_diff = logits_to_ave_logit_diff(contrast_logits_with_circuit, normal_answer, contrast_answer)

    model.reset_hooks(including_permanent=True)
    
    del normal_cache
    t.cuda.empty_cache()

    return contrast_logits_with_circuit_diff, normal_logits_no_circuit_diff, contrast_logits_no_circuit_diff

def eval_circuit_batched(model, data, circuit, batch_size):
    agg_patched_logits_diff = 0.0
    agg_normal_logits_diff = 0.0
    agg_contrast_logits_diff = 0.0
    
    for st in trange(0, len(data), batch_size, desc="batch"):
        ed = st + batch_size
        batch = data[st: ed]
        patched_logits_diff, normal_logits_diff, contrast_logits_diff = eval_circuit_one_batch(model, batch, circuit)
        
        agg_patched_logits_diff = agg_patched_logits_diff + patched_logits_diff * len(batch)
        agg_normal_logits_diff = agg_normal_logits_diff + normal_logits_diff  * len(batch)
        agg_contrast_logits_diff = agg_contrast_logits_diff + contrast_logits_diff * len(batch)

    return agg_patched_logits_diff, agg_normal_logits_diff, agg_contrast_logits_diff

def complete_circuit(n_shot=4):
    all_heads = list(itertools.product(range(0,42), range(0,16)))
    CIRCUIT_HEADS = {
        "default": all_heads
    }
    SEQ_POS_TO_KEEP_HEADS = {"default": "all"}
    POSITIONS = {"all": list(range(0, n_shot*6+5))}

    circuit = {"head_circuit": CIRCUIT_HEADS, "head_seq_pos_to_keep": SEQ_POS_TO_KEEP_HEADS, "positions": POSITIONS}
    return circuit

def identified_circuit(n_shot=4):

    CIRCUIT_HEADS = {
        "function_induction": [(39, 12), (39, 7), (36, 7), (32, 6), (32, 1), (25, 13)],
        "prev_token": [(38, 9), (38, 6), (38, 7), (35, 9), (35, 14), (31, 5), (31, 4), (29, 5)] ,
        "consolidate": [(41, 5), (41, 4), (40, 12), (40, 11)],
    }

    SEQ_POS_TO_KEEP_HEADS = {
        "function_induction": "all=",
        "prev_token": "c",
        "consolidate": "all=",
    }

    POSITIONS = {
        "test=": [n_shot*6 + 4],
        "c": [i*6+5 for i in range(n_shot)],
        "=": [i*6+4 for i in range(n_shot)],
        "all=": [i*6+4 for i in range(n_shot+1)],
    }
    circuit = {"head_circuit": CIRCUIT_HEADS, "head_seq_pos_to_keep": SEQ_POS_TO_KEEP_HEADS, "positions": POSITIONS}
    return circuit

def get_circuit_set_completeness(C, K, M):
    """return A: C\\K and B: M\\K as defined in sec 4.1. C: current circuit, K: heads to remove, M: full model circuit"""
    circuit_A, circuit_B = copy.deepcopy(C), copy.deepcopy(M)
    for key in C["head_circuit"].keys():
        circuit_A["head_circuit"][key] = list(set(C["head_circuit"][key]).difference(K))
    for key in M["head_circuit"].keys():
        circuit_B["head_circuit"][key] = list(set(M["head_circuit"][key]).difference(K))
    return circuit_A, circuit_B

def get_completeness_score(model, data, C, K, M, batch_size=4):
    circuit_A, circuit_B = get_circuit_set_completeness(C, K, M)
    score_A, _, _ = eval_circuit_batched(model, data, circuit_A, batch_size)
    score_B, _, _ = eval_circuit_batched(model, data, circuit_B, batch_size)
    N = len(data)
    return score_A.item() / N, score_B.item() / N

def minitest():
    model_name = "google/gemma-2-9b"
    model = HookedTransformer.from_pretrained(model_name, device="cuda:0")
    model.set_ungroup_grouped_query_attention(True)

    setting, nmax, offset, n_icl_examples = "setting1", 9, 1, 16
    filename = f"../data/addition/{setting}/addition_nmax{nmax}_offset{offset}.jsonl"
    data = read_jsonl(filename)
    processed_data = process_dataset(data, n_icl_examples=n_icl_examples, offset=offset)

    # load 4 example for testing
    data = processed_data[:1]
    circuit = identified_circuit(n_shot=16)
    a, b, c = eval_circuit_one_batch(model, data, circuit)
    print(a, b, c)

    data = processed_data
    a, b, c = eval_circuit_batched(model, data, circuit, batch_size=1)
    
    print(a, b, c)
    r = (a - b) / (c - b)
    print(r.item())


if __name__ == "__main__":
    minitest()