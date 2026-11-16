# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic test data in the real GLM geometry.

Full weight/reference construction is intentionally confined to the eight-GPU
worker tests. The shared router/normalization tensors are broadcast separately.
"""

from __future__ import annotations

import math
import torch

HIDDEN, QLORA, KVLORA = 6144, 2048, 512
NOPE, ROPE, VDIM, HEADS = 192, 64, 256, 10
IDX_HEADS, IDX_DIM, WORLD = 32, 128, 8
PAD_RANK, PAD_HEAD0 = 7, 4
KI_CONC_SIGMA = {"L1": 0.003, "T": 0.05}


def make_input(q_len: int, seed: int, device, *, cancellation=False):
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(q_len, HIDDEN, generator=generator).bfloat16()
    if cancellation:
        value.zero_()
        value[:, 0] = 0 if seed == 23 else 2
    return value.to(device)


def make_rope_table(capacity: int, device, theta=8000000.0):
    # Generate on CPU in FP32, as for checkpoint loading. Both implementations
    # consume these exact values; testing does not substitute new sin/cos math.
    inv = 1.0 / (theta ** (torch.arange(0, ROPE, 2, dtype=torch.float32) / ROPE))
    angles = torch.outer(torch.arange(capacity + 8), inv)
    table = torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
    return table.reshape(capacity + 8, ROPE).contiguous().to(device)


def make_attention_input(q_len, seed, device):
    # Draw all four rows even for q1, preserving the accepted token-zero fixture.
    generator = torch.Generator(device=device).manual_seed(seed * 1000 + 900)
    return (
        torch.randn(4, HIDDEN, generator=generator, device=device)
        .bfloat16()[:q_len]
        .contiguous()
    )


def make_selection(q_len, past, seed):
    """Explicit accepted rows, with changed membership across replay seeds."""
    ids = torch.full((q_len, 2048), -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(1000003 * seed + past)
    for query in range(q_len):
        visible = past + query + 1
        if visible <= 2048:
            ids[query, :visible] = torch.arange(visible, dtype=torch.int32)
        else:
            scores = torch.randn(visible, generator=generator, dtype=torch.float64)
            ids[query] = torch.topk(scores, 2048).indices.sort().values.int()
    return ids


def make_moe_raw(rank: int, seed: int, device):
    """One rank's FP8 codes/scales; rank zero also creates replicated tensors."""
    generator = torch.Generator(device=device).manual_seed(seed + rank)

    def codes(shape):
        values = torch.randint(
            0, 32, shape, device=device, dtype=torch.uint8, generator=generator
        ).add_(0x28)
        signs = torch.randint(
            0, 2, shape, device=device, dtype=torch.uint8, generator=generator
        )
        values.bitwise_or_(signs.mul_(128))
        return values.view(torch.float8_e4m3fn)

    gate = codes((257, 256, 6144))
    up = codes((257, 256, 6144))
    gs = 0.001 + 0.002 * torch.rand(257, 2, 48, device=device, generator=generator)
    us = 0.001 + 0.002 * torch.rand(257, 2, 48, device=device, generator=generator)
    down = codes((257, 6144, 256))
    ds = 0.001 + 0.002 * torch.rand(257, 48, 2, device=device, generator=generator)
    raw = dict(gate=gate, up=up, gs=gs, us=us, down=down, ds=ds)
    if rank == 0:
        raw["gamma"] = (
            (0.8 + 0.4 * torch.rand(6144, device=device, generator=generator))
            .bfloat16()
            .float()
        )
        raw["router"] = (
            0.02 * torch.randn(256, 6144, device=device, generator=generator)
        ).bfloat16()
        raw["bias"] = 0.01 * torch.randn(256, device=device, generator=generator)
    return raw


def make_moe_cancellation(raw, rank, device):
    """Force a TP sum where an extra BF16 rounding loses the 1/256 result."""
    from . import moe_reference as ref

    raw["gamma"].fill_(1)
    for key in ("gate", "up", "down"):
        raw[key].view(torch.uint8).zero_()
    for key in ("gate", "up"):
        raw[key].view(torch.uint8)[0, :, 0].fill_(0x38)
    for key in ("gs", "us", "ds"):
        raw[key].fill_(1)
    x = make_input(1, 1, device, cancellation=True)
    norm = ref.rmsnorm(x, raw["gamma"]).to(device)
    quantized, scales = ref.q128(ref.shared_mid(norm, raw))
    denominator = quantized[0, :128].float().sum() * scales[0, 0]
    assert denominator > 0
    if rank < 2:
        raw["down"].view(torch.uint8)[0, 0, :128].fill_(0xB8 if rank == 0 else 0x38)
        raw["ds"][0, 0, 0] = (2 if rank == 0 else 1 / 256) / denominator
    return raw


def _gen(seed: int, device) -> torch.Generator:
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


def _fp8(g: torch.Generator, *shape, device) -> torch.Tensor:
    return (
        torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * 0.5
    ).to(torch.float8_e4m3fn)


def _scale(g: torch.Generator, *shape, K: int, device) -> torch.Tensor:
    return (
        torch.rand(*shape, generator=g, device=device, dtype=torch.float32) + 0.5
    ) / math.sqrt(K)


def _gamma(g: torch.Generator, n: int, device) -> torch.Tensor:
    return 1.0 + 0.1 * torch.randn(n, generator=g, device=device, dtype=torch.float32)


def make_attention_weights(rank: int, seed: int, device) -> dict:
    if not 1 <= rank < WORLD:
        raise ValueError("attention rank must be 1..7")
    dev = torch.device(device)
    g = _gen(seed * 1000 + rank, dev)
    W = {}
    W["x_rmsnorm_gamma"] = _gamma(g, HIDDEN, dev)
    W["q_a_weights"] = _fp8(g, QLORA, HIDDEN, device=dev)
    W["q_a_scale"] = _scale(g, QLORA // 128, HIDDEN // 128, K=HIDDEN, device=dev)
    W["kv_a_weights"] = _fp8(g, KVLORA, HIDDEN, device=dev)
    W["kv_a_scale"] = _scale(g, KVLORA // 128, HIDDEN // 128, K=HIDDEN, device=dev)
    W["w_pe_weights"] = _fp8(g, ROPE, HIDDEN, device=dev)
    W["w_pe_scale"] = _scale(g, 1, HIDDEN // 128, K=HIDDEN, device=dev)
    W["q_rmsnorm_gamma"] = _gamma(g, QLORA, dev)
    wqb = (
        torch.randn(
            HEADS * (NOPE + ROPE), QLORA, generator=g, device=dev, dtype=torch.float32
        )
        * 0.5
    )
    W["wqb_scale"] = _scale(
        g, HEADS * (NOPE + ROPE) // 128, QLORA // 128, K=QLORA, device=dev
    )
    W["kv_rmsnorm_gamma"] = _gamma(g, KVLORA, dev)
    wkv_b1 = (
        torch.randn(HEADS, NOPE, KVLORA, generator=g, device=dev, dtype=torch.float32)
        * 0.5
    )
    wkv_b1_scale = _scale(g, HEADS, NOPE // 64, KVLORA // 128, K=KVLORA, device=dev)
    wkv_b2 = (
        torch.randn(HEADS, VDIM, KVLORA, generator=g, device=dev, dtype=torch.float32)
        * 0.5
    )
    W["wkv_b2_scale"] = _scale(
        g, HEADS, VDIM // 64, KVLORA // 128, K=KVLORA, device=dev
    )
    unproj = (
        torch.randn(HIDDEN, HEADS * VDIM, generator=g, device=dev, dtype=torch.float32)
        * 0.5
    )
    W["unproj_scale"] = _scale(
        g, HIDDEN // 128, HEADS * VDIM // 128, K=HEADS * VDIM, device=dev
    )
    if rank == PAD_RANK:
        wqb[PAD_HEAD0 * (NOPE + ROPE) :] = 0.0
        wkv_b1[PAD_HEAD0:] = 0.0
        wkv_b2[PAD_HEAD0:] = 0.0
        unproj[:, PAD_HEAD0 * VDIM :] = 0.0
    W["wqb_weights"] = wqb.to(torch.float8_e4m3fn)
    W["wkv_b1_weights"] = wkv_b1.to(torch.float8_e4m3fn).transpose(1, 2).contiguous()
    W["wkv_b1_scale"] = wkv_b1_scale.transpose(1, 2).contiguous()
    W["wkv_b2_weights"] = wkv_b2.to(torch.float8_e4m3fn)
    W["unproj_weights"] = unproj.to(torch.float8_e4m3fn)
    return W


def make_indexer_weights(seed: int, device) -> dict:
    dev = torch.device(device)
    g = _gen(seed * 1000, dev)
    W = {}
    W["x_rmsnorm_gamma"] = _gamma(g, HIDDEN, dev)
    W["q_a_weights"] = _fp8(g, QLORA, HIDDEN, device=dev)
    W["q_a_scale"] = _scale(g, QLORA // 128, HIDDEN // 128, K=HIDDEN, device=dev)
    W["wk_weights"] = _fp8(g, IDX_DIM, HIDDEN, device=dev)
    W["wk_scale"] = _scale(g, 1, HIDDEN // 128, K=HIDDEN, device=dev)
    W["id_score_weights"] = (
        torch.randn(IDX_HEADS, HIDDEN, generator=g, device=dev, dtype=torch.float32)
        * 0.5
        / math.sqrt(HIDDEN)
    ).to(torch.bfloat16)
    W["q_rmsnorm_gamma_qi"] = _gamma(g, QLORA, dev)
    W["wqi_weights"] = _fp8(g, IDX_HEADS * IDX_DIM, QLORA, device=dev)
    W["wqi_scale"] = _scale(
        g, IDX_HEADS * IDX_DIM // 128, QLORA // 128, K=QLORA, device=dev
    )
    W["k_weights"] = _gamma(g, IDX_DIM, dev)
    W["k_bias"] = 0.1 * torch.randn(
        IDX_DIM, generator=g, device=dev, dtype=torch.float32
    )
    return W


def ki_sigma(seed: int):
    if 9000 <= seed < 9100:
        return KI_CONC_SIGMA["L1"]
    if 9100 <= seed < 10000:
        return KI_CONC_SIGMA["T"]
    return None


def make_state(rank: int, cache_rows: int, seed: int, device) -> dict:
    dev = torch.device(device)
    if rank == 0:
        g = _gen(seed * 1000 + 500, dev)
        sigma = ki_sigma(seed)
        if sigma is None:
            return {
                "ki_cache": torch.randn(
                    cache_rows, IDX_DIM, generator=g, device=dev, dtype=torch.float32
                ).to(torch.bfloat16)
            }
        u = torch.randn(1, IDX_DIM, generator=g, device=dev, dtype=torch.float32)
        eps = torch.randn(
            cache_rows, IDX_DIM, generator=g, device=dev, dtype=torch.float32
        )
        return {"ki_cache": (u + sigma * eps).to(torch.bfloat16)}
    g = _gen(seed * 1000 + 501, dev)
    kv = torch.randn(
        cache_rows, KVLORA, generator=g, device=dev, dtype=torch.float32
    ).to(torch.bfloat16)
    pe = torch.randn(cache_rows, ROPE, generator=g, device=dev, dtype=torch.float32).to(
        torch.bfloat16
    )
    return {"kv_cache": kv, "pe_cache": pe}
