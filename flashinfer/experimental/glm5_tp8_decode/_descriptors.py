# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Private immutable device descriptors built from named tensors outside capture."""

from dataclasses import dataclass


def _ptr(value):
    return int(value.data_ptr()) if hasattr(value, "data_ptr") else int(value)


@dataclass
class PreparedContext:
    tensor: object
    operands: dict

    @property
    def ptr(self):
        return int(self.tensor.data_ptr())


def _upload(device, rows, operands, scalars=()):
    import torch

    words = [0] * (len(rows) * 16)
    for row, fields in enumerate(rows):
        for col, name in fields.items():
            words[row * 16 + col] = _ptr(operands[name])
    for row, col, value in scalars:
        words[row * 16 + col] = value
    return PreparedContext(
        torch.tensor(words, dtype=torch.int64, device=device), operands
    )


def prepare_q1_attention(
    device,
    *,
    x,
    x_gamma,
    q_gamma,
    kv_gamma,
    qkva,
    wqb,
    projq,
    projo,
    unproj,
    cur_pos,
    rope,
    kv_cache,
    pe_cache,
    selected_ids,
    ll_recv,
    post_q,
    post_kv,
    post_qnd,
    post_qpe,
    post_qnope,
    post_o,
    post_projo,
    o_acc,
    lse_acc,
    hidden_out,
    nan_fill_f32=0,
    nan_fill_bf16=0,
    ll_tag=1,
):
    operands = dict(locals())
    rows = [
        {
            0: "x",
            1: "x_gamma",
            2: "qkva",
            3: "cur_pos",
            4: "post_q",
            5: "post_kv",
            6: "pe_cache",
        },
        {0: "post_q", 1: "wqb", 3: "q_gamma", 4: "post_qnd", 5: "post_qpe"},
        {0: "post_qpe", 1: "pe_cache", 2: "rope", 3: "cur_pos"},
        {0: "post_kv", 1: "kv_gamma", 2: "kv_cache", 3: "cur_pos"},
        {0: "post_qnd", 1: "projq", 3: "post_qnope"},
        {
            0: "post_qnope",
            1: "post_qpe",
            2: "kv_cache",
            3: "pe_cache",
            4: "selected_ids",
            5: "post_o",
            6: "o_acc",
            8: "lse_acc",
            9: "cur_pos",
            12: "ll_recv",
        },
        {0: "post_o", 1: "projo", 3: "post_projo"},
        {
            0: "post_projo",
            1: "unproj",
            3: "hidden_out",
            4: "x",
            6: "nan_fill_f32",
            7: "nan_fill_bf16",
        },
    ]
    return _upload(device, rows, operands, [(2, 4, 10), (5, 10, 10), (5, 13, ll_tag)])


def prepare_q1_indexshare(
    device,
    *,
    x,
    cur_pos,
    selected_ids,
    id_peer_table,
    own_slice,
    hidden_out,
    nan_fill_f32=0,
    nan_fill_bf16=0,
    ll_tag=1,
):
    operands = dict(locals())
    rows = [
        {0: "x"},
        {},
        {1: "cur_pos"},
        {},
        {4: "cur_pos", 6: "selected_ids"},
        {1: "selected_ids", 2: "cur_pos"},
        {0: "selected_ids", 1: "id_peer_table"},
        {
            0: "own_slice",
            1: "x",
            2: "hidden_out",
            4: "nan_fill_f32",
            5: "nan_fill_bf16",
        },
    ]
    # Only the accepted query row is published. No indexer weights are needed.
    return _upload(device, rows, operands, [(6, 2, (2048 << 32) | ll_tag), (6, 3, 7)])


def prepare_q1_moe(
    device,
    *,
    x,
    gamma,
    router_w,
    upgate_w,
    bias,
    down_w,
    down_scale,
    scores,
    hidden_mid,
    x_norm,
    hidden_out,
    sel_probs,
    sel_indices,
):
    operands = dict(locals())
    rows = [
        {0: "x", 1: "gamma"},
        {0: "router_w", 1: "scores"},
        {
            0: "upgate_w",
            1: "bias",
            2: "scores",
            3: "sel_probs",
            4: "sel_indices",
            5: "hidden_mid",
        },
        {0: "upgate_w", 1: "hidden_mid", 2: "x_norm", 3: "x"},
        {0: "down_w", 1: "hidden_mid", 2: "hidden_out", 3: "x", 5: "down_scale"},
    ]
    return _upload(device, rows, operands)


def prepare_indexer(
    device,
    *,
    x,
    x_gamma,
    q_gamma,
    qakis,
    wis,
    wqi,
    cur_pos,
    rope,
    ki_cache,
    ln_gamma,
    ln_beta,
    q_a,
    ki,
    head_weights,
    iq,
    iq_rope,
    logits,
    histogram,
    candidates,
    selected_ids,
    id_peer_table,
    k_tmap,
    q_tmap,
    hidden_out,
    nan_fill_f32=0,
    nan_fill_bf16=0,
    ll_tag=1,
):
    operands = dict(locals())
    rows = [
        {
            0: "x",
            1: "x_gamma",
            2: "qakis",
            3: "q_a",
            4: "ki",
            5: "wis",
            6: "head_weights",
        },
        {0: "q_a", 1: "wqi", 3: "q_gamma", 4: "iq"},
        {0: "ki", 1: "cur_pos", 2: "ki_cache", 3: "ln_gamma", 4: "ln_beta", 5: "rope"},
        {0: "iq", 1: "iq_rope", 2: "rope"},
        {
            1: "ki_cache",
            2: "head_weights",
            3: "logits",
            4: "cur_pos",
            5: "histogram",
            6: "selected_ids",
            7: "candidates",
            8: "k_tmap",
            9: "q_tmap",
        },
        {0: "logits", 1: "selected_ids", 2: "cur_pos", 4: "histogram", 5: "candidates"},
        {0: "selected_ids", 1: "id_peer_table"},
        {1: "x", 2: "hidden_out", 4: "nan_fill_f32", 5: "nan_fill_bf16"},
    ]
    return _upload(
        device, rows, operands, [(3, 3, 32), (6, 2, (8192 << 32) | ll_tag), (6, 3, 7)]
    )
