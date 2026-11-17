# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The only test-side adaptation layer for the public operator API."""

from flashinfer import glm5_tp8_decode as api


def workspace(control, *, capacity=4096, operators=("moe", "attention")):
    value = api.create_glm5_tp8_decode_workspace(
        rank=control.rank,
        group=control.group,
        device=control.device,
        max_seq_len=capacity,
        operators=operators,
    )
    value.warmup()
    return value


def pack_moe(raw):
    return api.pack_glm5_tp8_moe_weights(
        raw["gamma"],
        raw["router"],
        raw["bias"],
        raw["gate"],
        raw["gs"],
        raw["up"],
        raw["us"],
        raw["down"],
        raw["ds"],
    )


def prepare_moe(x, packed, out, ws):
    return api.prepare_glm5_tp8_moe(x, packed, out=out, workspace=ws)


def pack_attention(raw, rank):
    return api.pack_glm5_tp8_attention_weights(raw, rank=rank)


def prepare_attention(
    x, packed, caches, rope, position, out, ws, *, indexer, selection=None
):
    return api.prepare_glm5_tp8_sparse_attention(
        x,
        packed,
        caches=caches,
        rope=rope,
        cur_pos=position,
        out=out,
        workspace=ws,
        indexer=indexer,
        selection=selection,
    )
