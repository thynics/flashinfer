# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Load-time packing of rank-local FP8 attention and indexer weights."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionWeights:
    """Owned weights for one logical TP rank, shared by q1 and q4 plans.

    ``tensors`` contains contiguous, independently owned output storage. It does
    not retain the raw input dictionary or alias its tensors. Treat this mapping
    and its tensors as read-only after constructing a plan. Projection pages
    contain FP8 codes and FP32 scale bytes in uint8 storage; ``wis`` is BF16 and
    gamma/bias vectors are FP32.
    """

    rank: int
    tensors: dict[str, torch.Tensor]


_FP8 = torch.float8_e4m3fn
_F32 = torch.float32
_COMMON_INPUTS = {
    "x_rmsnorm_gamma": ((6144,), _F32),
    "q_a_weights": ((2048, 6144), _FP8),
    "q_a_scale": ((16, 48), _F32),
}
_INDEXER_INPUTS = {
    "wk_weights": ((128, 6144), _FP8),
    "wk_scale": ((1, 48), _F32),
    "id_score_weights": ((32, 6144), torch.bfloat16),
    "q_rmsnorm_gamma_qi": ((2048,), _F32),
    "wqi_weights": ((4096, 2048), _FP8),
    "wqi_scale": ((32, 16), _F32),
    "k_weights": ((128,), _F32),
    "k_bias": ((128,), _F32),
}
_MLA_INPUTS = {
    "kv_a_weights": ((512, 6144), _FP8),
    "kv_a_scale": ((4, 48), _F32),
    "w_pe_weights": ((64, 6144), _FP8),
    "w_pe_scale": ((1, 48), _F32),
    "q_rmsnorm_gamma": ((2048,), _F32),
    "wqb_weights": ((2560, 2048), _FP8),
    "wqb_scale": ((20, 16), _F32),
    "kv_rmsnorm_gamma": ((512,), _F32),
    "wkv_b1_weights": ((10, 512, 192), _FP8),
    "wkv_b1_scale": ((10, 4, 3), _F32),
    "wkv_b2_weights": ((10, 256, 512), _FP8),
    "wkv_b2_scale": ((10, 4, 4), _F32),
    "unproj_weights": ((6144, 2560), _FP8),
    "unproj_scale": ((48, 20), _F32),
}


def _validate(raw: dict, rank: int) -> torch.device:
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < 8:
        raise ValueError("rank must be an integer in [0, 7]")
    if not isinstance(raw, dict):
        raise TypeError("raw must be a dictionary of rank-local weight tensors")
    contract = _COMMON_INPUTS | (_INDEXER_INPUTS if rank == 0 else _MLA_INPUTS)
    missing = contract.keys() - raw.keys()
    extra = raw.keys() - contract.keys()
    if missing or extra:
        raise ValueError(
            f"rank {rank} weight keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(map(str, extra))}"
        )
    device = None
    for name, (shape, dtype) in contract.items():
        tensor = raw[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must use the torch.strided layout")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name}: weight packing supports CPU or CUDA tensors")
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    return device


def _owned(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.contiguous_format)


def pack_attention_weights(raw: dict, *, rank: int) -> AttentionWeights:
    """Pack a complete rank-local dictionary once, on CPU or CUDA.

    Rank 0 accepts the eleven indexer keys in ``_COMMON_INPUTS`` and
    ``_INDEXER_INPUTS``; ranks 1..7 accept the seventeen MLA keys in
    ``_COMMON_INPUTS`` and ``_MLA_INPUTS``. Those tables specify exact shapes and
    dtypes. All tensors must be on the same device; noncontiguous inputs are
    supported. No quantization or scale conversion is performed.

    MLA shards have ten head slots. Ranks 1..6 each contain ten real heads;
    rank 7 contains four real heads and six zero-padded slots. The caller must
    supply zeros for those slots in WqB, both WkvB tensors, and the corresponding
    UnProj columns. Packing preserves the supplied values and does not create
    head padding. WkvB1 input is already transposed to [10, 512, 192], with
    scales [10, 4, 3] for 128-row by 64-column blocks.

    Returned rank-0 keys are x_gamma, q_gamma, qakis, wis, wqi, ln_gamma,
    ln_beta. Other ranks return x_gamma, q_gamma, kv_gamma, qkva, wqb, projq,
    projo, unproj. The returned tensors own their storage, so callers may
    release the raw dictionary immediately. Call outside CUDA graph capture.
    """
    from . import _attention_packing as packing

    device = _validate(raw, rank)
    device_guard = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with device_guard, torch.no_grad():
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "pack_attention_weights must run outside CUDA graph capture"
            )
        tensors = {"x_gamma": _owned(raw["x_rmsnorm_gamma"])}
        if rank == 0:
            tensors.update(
                q_gamma=_owned(raw["q_rmsnorm_gamma_qi"]),
                qakis=packing.pack_qakis(raw),
                wis=packing.pack_wis(raw["id_score_weights"]),
                wqi=packing.pack_wqi(raw["wqi_weights"], raw["wqi_scale"]),
                ln_gamma=_owned(raw["k_weights"]),
                ln_beta=_owned(raw["k_bias"]),
            )
        else:
            tensors.update(
                q_gamma=_owned(raw["q_rmsnorm_gamma"]),
                kv_gamma=_owned(raw["kv_rmsnorm_gamma"]),
                qkva=packing.pack_qkva(raw),
                wqb=packing.pack_wqb(raw["wqb_weights"], raw["wqb_scale"]),
                projq=packing.pack_projq(raw["wkv_b1_weights"], raw["wkv_b1_scale"]),
                projo=packing.pack_projo(raw["wkv_b2_weights"], raw["wkv_b2_scale"]),
                unproj=packing.pack_unproj(raw["unproj_weights"], raw["unproj_scale"]),
            )
    return AttentionWeights(rank=rank, tensors=tensors)


_PACKED_INDEXER = {
    "x_gamma": ((6144,), _F32),
    "q_gamma": ((2048,), _F32),
    "qakis": ((68 * 6 * 32896,), torch.uint8),
    "wis": ((2, 98304), torch.bfloat16),
    "wqi": ((128, 2, 32896), torch.uint8),
    "ln_gamma": ((128,), _F32),
    "ln_beta": ((128,), _F32),
}
_PACKED_MLA = {
    "x_gamma": ((6144,), _F32),
    "q_gamma": ((2048,), _F32),
    "kv_gamma": ((512,), _F32),
    "qkva": ((82 * 6 * 32896,), torch.uint8),
    "wqb": ((80, 2, 32896), torch.uint8),
    "projq": ((80, 12416), torch.uint8),
    "projo": ((80, 16512), torch.uint8),
    "unproj": ((128, 5, 24704), torch.uint8),
}


def validate_packed(
    weights: AttentionWeights, rank: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Validate the packed ABI before uploading any pointer-only descriptors.

    Return the checked tensor mapping. No device data is read or synchronized;
    preparation remains responsible for collectively propagating any failure.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < 8:
        raise ValueError("rank must be an integer in [0, 7]")
    if (
        not isinstance(weights, AttentionWeights)
        or isinstance(weights.rank, bool)
        or not isinstance(weights.rank, int)
        or weights.rank != rank
    ):
        raise ValueError("weights must be packed for this logical rank")
    if not isinstance(weights.tensors, dict):
        raise TypeError("packed weights.tensors must be a dictionary")
    tensors = weights.tensors
    contract = _PACKED_INDEXER if rank == 0 else _PACKED_MLA
    missing = contract.keys() - tensors.keys()
    extra = tensors.keys() - contract.keys()
    if missing or extra:
        raise ValueError(
            f"rank {rank} packed weight keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(map(str, extra))}"
        )
    device = torch.device(device)
    for name, (shape, dtype) in contract.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"packed {name} must be a torch.Tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"packed {name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise TypeError(
                f"packed {name} must have dtype {dtype}, got {tensor.dtype}"
            )
        if tensor.device != device:
            raise ValueError(f"packed {name} must be on {device}, got {tensor.device}")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"packed {name} must be contiguous with strided layout")
        if tensor.data_ptr() % 16:
            raise ValueError(f"packed {name} must be 16-byte aligned")
        if tensor.requires_grad:
            raise ValueError(f"packed {name} does not support autograd")
    return tensors
