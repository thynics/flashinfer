# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving layouts for the fixed TP8 attention projection kernels.

FP8 weights are reinterpreted as bytes before copying or concatenating. This
also supports CPU PyTorch builds without FP8 concatenation kernels. FP32 scale
bytes occupy the metadata tail of each page; unused metadata bytes are zero.
"""

from __future__ import annotations

import torch


def _swizzle_mma_16x16(matrix: torch.Tensor) -> torch.Tensor:
    prefix = matrix.shape[:-2]
    matrix = matrix.reshape(*prefix, 2, 8, 2, 4, 2)
    matrix = matrix.transpose(-4, -3).transpose(-5, -4)
    return matrix.reshape(*prefix, 4, 32, 2).transpose(-3, -2)


def _swizzle_pages(matrix: torch.Tensor, k_dim: int, pages: int) -> torch.Tensor:
    k_per_page = k_dim // pages
    prefix = matrix.shape[:-2]
    matrix = matrix.reshape(*prefix, 16, pages, k_per_page).transpose(-3, -2)
    matrix = matrix.reshape(*prefix, pages, 16, k_per_page // 16, 16)
    return _swizzle_mma_16x16(matrix.transpose(-3, -2)).contiguous()


def _permute_lane_fragment(tile: torch.Tensor) -> torch.Tensor:
    prefix = tile.shape[:-2]
    n = len(prefix)
    return (
        tile.reshape(*prefix, 2, 8, 2, 4, 2)
        .permute(*range(n), n + 1, n + 3, n + 2, n, n + 4)
        .contiguous()
        .reshape(*prefix, 32, 8)
    )


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().view(torch.uint8).contiguous()


def pack_lane_major(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack 32-row CTAs with six 1024-column pages and eight FP32 scales."""
    num_ctas = weights.shape[0] // 32
    num_pages = 6
    matrix = weights.reshape(num_ctas, 2, 16, num_pages, 1024)
    matrix = matrix.permute(0, 3, 1, 2, 4).contiguous()
    matrix = matrix.reshape(num_ctas, num_pages, 2, 16, 8, 8, 16)
    matrix = matrix.permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    matrix = _permute_lane_fragment(matrix).reshape(num_ctas, num_pages, 32768)

    scale_rows = torch.arange(num_ctas, device=weights.device) // 4
    scale_bytes = scales[scale_rows].reshape(num_ctas, num_pages, 8)
    scale_bytes = scale_bytes.contiguous().view(torch.uint8)
    output = torch.zeros(
        num_ctas, num_pages, 32896, dtype=torch.uint8, device=weights.device
    )
    output[:, :, :32768] = matrix
    output[:, :, 32768:32800] = scale_bytes
    return output.reshape(-1)


def pack_qkva(raw: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = torch.cat(
        [_bytes(raw[name]) for name in ("q_a_weights", "kv_a_weights", "w_pe_weights")],
        dim=0,
    )
    scales = torch.cat(
        [raw[name] for name in ("q_a_scale", "kv_a_scale", "w_pe_scale")], dim=0
    )
    return pack_lane_major(weights, scales)


def pack_qakis(raw: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = torch.cat([_bytes(raw["q_a_weights"]), _bytes(raw["wk_weights"])])
    scales = torch.cat([raw["q_a_scale"], raw["wk_scale"]])
    return pack_lane_major(weights, scales)


def _append_scale_tail(
    matrix: torch.Tensor, scales: torch.Tensor, page_bytes: int
) -> torch.Tensor:
    scale_bytes = scales.contiguous().view(torch.uint8)
    padding_bytes = page_bytes - matrix.shape[-1] - scale_bytes.shape[-1]
    padding = torch.zeros(
        *matrix.shape[:-1], padding_bytes, dtype=torch.uint8, device=matrix.device
    )
    return torch.cat([matrix, scale_bytes, padding], dim=-1).contiguous()


def pack_wqb(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    weights = _bytes(weights).reshape(10, 256, 2048)
    scales = scales.reshape(20, 1, 16).repeat(1, 128, 1).reshape(10, 256, 16)
    matrices, scale_parts = [], []
    for start, stop, num_ctas in ((0, 192, 60), (192, 256, 20)):
        matrix = weights[:, start:stop, :].reshape(num_ctas, 2, 16, 2048)
        matrix = _swizzle_pages(matrix, 2048, 2)
        matrix = matrix.reshape(num_ctas, 2, 2, 16, -1).transpose(1, 2)
        matrices.append(matrix.reshape(num_ctas, 2, 32768))
        scale = scales[:, start:stop, :].reshape(num_ctas, 32, 2, 8)
        scale_parts.append(scale.transpose(1, 2)[:, :, 0, :])
    return _append_scale_tail(
        torch.cat(matrices, dim=0), torch.cat(scale_parts, dim=0), 32896
    )


def pack_projq(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    # Input is already the load-time transpose [head, kv, nope]. Avoid building
    # the combined checkpoint tensor just to select and transpose it back.
    matrix = _bytes(weights).reshape(80, 4, 16, 192)
    matrix = _swizzle_pages(matrix, 192, 1).reshape(80, 12288)
    scale = scales.repeat_interleave(2, dim=1).reshape(80, 3)
    return _append_scale_tail(matrix, scale, 12416)


def pack_projo(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    matrix = _bytes(weights).reshape(80, 2, 16, 512)
    matrix = _swizzle_pages(matrix, 512, 1).reshape(80, 16384)
    scale = scales.repeat_interleave(2, dim=1).reshape(80, 4)
    return _append_scale_tail(matrix, scale, 16512)


def pack_unproj(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = scales.reshape(48, 1, 20).repeat(1, 8, 1)
    scale = scale.reshape(128, 3, 5, 4).transpose(1, 2).reshape(128, 5, 12)
    matrix = _bytes(weights).reshape(128, 3, 16, 5, 512)
    matrix = matrix.transpose(2, 3).reshape(128, 3, 5, 16, 32, 16)
    matrix = matrix.transpose(3, 4).reshape(128, 3, 5, 32, 16, 16)
    matrix = _swizzle_mma_16x16(matrix)
    matrix = matrix.transpose(1, 2).reshape(128, 5, 24576)
    return _append_scale_tail(matrix, scale, 24704)


def pack_wqi(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = scales.reshape(32, 1, 16).repeat(1, 128, 1).reshape(4096, 16)
    scale = scale.reshape(128, 32, 2, 8).transpose(1, 2)[:, :, 0, :]
    matrix = _bytes(weights).reshape(128, 2, 16, 2048)
    matrix = _swizzle_pages(matrix, 2048, 2)
    matrix = matrix.reshape(128, 2, 2, 16, -1).transpose(1, 2)
    matrix = matrix.reshape(128, 2, 32768)
    return _append_scale_tail(matrix, scale, 32896)


def pack_wis(weights: torch.Tensor) -> torch.Tensor:
    matrix = weights.detach().clone().reshape(2, 16, 6, 1024).transpose(1, 2)
    matrix = matrix.reshape(2, 6, 1, 16, 64, 16).transpose(-3, -2)
    return _swizzle_mma_16x16(matrix).reshape(2, -1).contiguous()
