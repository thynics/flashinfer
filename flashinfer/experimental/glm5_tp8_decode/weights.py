# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Load-time weight packing; checkpoint FP8 codes and FP32 scales are preserved."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import EXPERT_BANKS, EXPERTS, HIDDEN_SIZE, LOCAL_INTERMEDIATE


@dataclass(frozen=True)
class MoEWeights:
    """One rank's packed weights, shared by q1 and q4 plans.

    Bank zero is the shared expert; bank e+1 is routed expert e. Intermediate
    dimensions are tensor-parallel shards. Router, bias and gamma are replicated.
    Packed byte tensors contain layout-specific FP8 codes and scale metadata.
    """

    gamma: torch.Tensor
    router: torch.Tensor
    bias: torch.Tensor
    upgate: torch.Tensor
    down: torch.Tensor
    down_scale: torch.Tensor


def _check(name, tensor, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def pack_moe_weights(
    gamma: torch.Tensor,
    router: torch.Tensor,
    bias: torch.Tensor,
    gate: torch.Tensor,
    gate_scale: torch.Tensor,
    up: torch.Tensor,
    up_scale: torch.Tensor,
    down: torch.Tensor,
    down_scale: torch.Tensor,
) -> MoEWeights:
    """Pack already tensor-parallel-sharded weights once, on CPU or CUDA.

    Gate/Up: FP8 [257, 256, 6144], scales FP32 [257, 2, 48].
    Down: FP8 [257, 6144, 256], scales FP32 [257, 48, 2].
    Scales multiply dequantized FP8 codes; this function performs no quantization.
    """
    from ._moe_packing import (
        expert_down_convert_to_bf16mma,
        expert_select_convert_to_mma,
    )

    if not isinstance(gamma, torch.Tensor):
        raise TypeError("gamma must be a torch.Tensor")
    device = gamma.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("weight packing supports CPU or CUDA tensors")
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("pack weights outside CUDA graph capture")
    _check("gamma", gamma, (HIDDEN_SIZE,), torch.float32, device)
    _check("router", router, (EXPERTS, HIDDEN_SIZE), torch.bfloat16, device)
    _check("bias", bias, (EXPERTS,), torch.float32, device)
    for name, value in (("gate", gate), ("up", up)):
        _check(
            name,
            value,
            (EXPERT_BANKS, LOCAL_INTERMEDIATE, HIDDEN_SIZE),
            torch.float8_e4m3fn,
            device,
        )
    for name, value in (("gate_scale", gate_scale), ("up_scale", up_scale)):
        _check(name, value, (EXPERT_BANKS, 2, 48), torch.float32, device)
    _check(
        "down",
        down,
        (EXPERT_BANKS, HIDDEN_SIZE, LOCAL_INTERMEDIATE),
        torch.float8_e4m3fn,
        device,
    )
    _check("down_scale", down_scale, (EXPERT_BANKS, 48, 2), torch.float32, device)
    with torch.no_grad():
        packed_bias, packed_upgate = expert_select_convert_to_mma(
            bias,
            gate.contiguous(),
            gate_scale.contiguous(),
            up.contiguous(),
            up_scale.contiguous(),
        )
        packed_down, packed_down_scale = expert_down_convert_to_bf16mma(
            down.contiguous(), down_scale.contiguous()
        )
    return MoEWeights(
        gamma.detach().clone(memory_format=torch.contiguous_format),
        router.detach().clone(memory_format=torch.contiguous_format),
        packed_bias.detach().clone(),
        packed_upgate.contiguous(),
        packed_down.contiguous(),
        packed_down_scale.contiguous(),
    )
