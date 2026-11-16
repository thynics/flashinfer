"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import torch

DIM = 6144
MOE_INTER_DIM = 2048
BLOCK_SIZE = 128
FP8 = torch.float8_e4m3fn


def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
    assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
    pre_shape = mat_in.shape[:-2]
    mat_in = (
        mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
    )
    return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)


def _swizzle_bf16mma_full_16x32(mat_in: torch.Tensor) -> torch.Tensor:
    assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
    assert mat_in.dtype == torch.uint8
    pre = mat_in.shape[:-2]
    mat = mat_in.reshape(*pre, 2, 8, 2, 2, 4, 2)
    n = len(pre)
    mat = mat.permute(*range(n), 1 + n, 4 + n, 2 + n, 3 + n, 0 + n, 5 + n)
    return mat.reshape(*pre, 32, 16).contiguous()


def _swizzle_bf16mma_partial_8x32(mat_in: torch.Tensor) -> torch.Tensor:
    assert mat_in.shape[-2] == 8 and mat_in.shape[-1] == 32
    assert mat_in.dtype == torch.uint8
    pre = mat_in.shape[:-2]
    mat = mat_in.reshape(*pre, 8, 2, 2, 4, 2)
    n = len(pre)
    mat = mat.permute(*range(n), 0 + n, 3 + n, 1 + n, 2 + n, 4 + n)
    return mat.reshape(*pre, 32, 8).contiguous()


def expert_down_convert_to_bf16mma(mat_in: torch.Tensor, scale_in: torch.Tensor):
    mat_in = mat_in.view(torch.uint8)
    dim = DIM
    num_sms = 128
    dim_per_sm = dim // num_sms
    expert_dim = MOE_INTER_DIM // 8
    k_chunks = expert_dim // 32
    scale_cols = expert_dim // BLOCK_SIZE
    n_full_tiles = dim_per_sm // 16
    remainder_rows = dim_per_sm % 16
    full_rows = n_full_tiles * 16

    with torch.inference_mode():
        exp_num = mat_in.shape[0]
        mat_per_cta = mat_in.reshape(exp_num, num_sms, dim_per_sm, expert_dim)

        full_part = mat_per_cta[:, :, :full_rows, :]
        full_tiles = full_part.reshape(
            exp_num, num_sms, n_full_tiles, 16, k_chunks, 32
        ).transpose(3, 4)
        full_swizzled = _swizzle_bf16mma_full_16x32(full_tiles)
        full_swizzled = full_swizzled.reshape(
            exp_num, num_sms, n_full_tiles * k_chunks * 32 * 16
        )

        mats = [full_swizzled]
        if remainder_rows > 0:
            partial_part = mat_per_cta[:, :, full_rows:, :]
            partial_tiles = partial_part.reshape(
                exp_num, num_sms, 1, remainder_rows, k_chunks, 32
            ).transpose(3, 4)
            partial_swizzled = _swizzle_bf16mma_partial_8x32(partial_tiles)
            partial_swizzled = partial_swizzled.reshape(
                exp_num, num_sms, k_chunks * 32 * remainder_rows
            )
            mats.append(partial_swizzled)

        mat_swizzled = torch.cat(mats, dim=2) if len(mats) > 1 else mats[0]
        mat_swizzled = mat_swizzled.reshape(exp_num, dim, expert_dim)

        mat_scale_packed = (
            scale_in.reshape(exp_num, dim // BLOCK_SIZE, 1, scale_cols)
            .repeat(1, 1, 16, 1)
            .reshape(exp_num, num_sms, -1)
        )
        target_cols_per_sm = 1024 * scale_cols // num_sms
        pad_amount = target_cols_per_sm - mat_scale_packed.shape[-1]
        if pad_amount > 0:
            padding_zeros = torch.zeros(
                (exp_num, num_sms, pad_amount),
                dtype=scale_in.dtype,
                device=scale_in.device,
            )
            mat_scale_packed = torch.cat([mat_scale_packed, padding_zeros], dim=2)
        mat_scale_packed = mat_scale_packed.reshape(exp_num, 1024, scale_cols)
        mat_scale_packed = mat_scale_packed.to(torch.float32)

        return mat_swizzled.contiguous(), mat_scale_packed.contiguous()


def expert_select_convert_to_mma(
    bias_or_gamma,
    weights_w1,
    scales_w1,
    weights_w3,
    scales_w3,
    algorithm: str = "fp16mma",
):
    weights_w1 = weights_w1.view(torch.uint8)
    weights_w3 = weights_w3.view(torch.uint8)
    dim = DIM
    pages = dim // 1024
    dim_scale_dim = dim // BLOCK_SIZE
    with torch.inference_mode():
        exp_num = weights_w1.shape[0]
        moe_rows = weights_w1.shape[1]
        n_row_groups = moe_rows // 16
        scale_m_dim = moe_rows // BLOCK_SIZE
        weights_w1 = weights_w1.reshape(
            exp_num, n_row_groups, 16, pages, 1024
        ).transpose(2, 3)
        weights_w3 = weights_w3.reshape(
            exp_num, n_row_groups, 16, pages, 1024
        ).transpose(2, 3)
        if algorithm == "fp8mma":
            raise NotImplementedError("fp8mma path is not supported by this layout")
        elif algorithm == "fp16mma":
            weights_w1 = weights_w1.reshape(
                exp_num, n_row_groups, pages, 16, 64, 16
            ).transpose(3, 4)
            weights_w1 = _swizzle_mma_16x16(weights_w1)
            weights_w1 = weights_w1.reshape(exp_num, n_row_groups, pages, 16, 1024)
            weights_w3 = weights_w3.reshape(
                exp_num, n_row_groups, pages, 16, 64, 16
            ).transpose(3, 4)
            weights_w3 = _swizzle_mma_16x16(weights_w3)
            weights_w3 = weights_w3.reshape(exp_num, n_row_groups, pages, 16, 1024)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        weights = torch.cat([weights_w1, weights_w3], dim=3)
        assert weights.shape == (exp_num, n_row_groups, pages, 32, 1024)
        weights = weights.reshape(exp_num, n_row_groups, pages, 32 * 1024)

        scales_per_page = 1024 // BLOCK_SIZE
        repeat_factor = n_row_groups // scale_m_dim
        scales_w1 = (
            scales_w1.reshape(exp_num, scale_m_dim, 1, dim_scale_dim)
            .repeat(1, 1, repeat_factor, 1)
            .reshape(exp_num, n_row_groups, 1, pages, scales_per_page)
        )
        scales_w1 = scales_w1.transpose(2, 3)
        scales_w3 = (
            scales_w3.reshape(exp_num, scale_m_dim, 1, dim_scale_dim)
            .repeat(1, 1, repeat_factor, 1)
            .reshape(exp_num, n_row_groups, 1, pages, scales_per_page)
        )
        scales_w3 = scales_w3.transpose(2, 3)
        scales = torch.cat([scales_w1, scales_w3], dim=3)
        assert scales.shape == (exp_num, n_row_groups, pages, 2, scales_per_page)

        scales = scales.to(torch.float32)

        scales = scales.reshape(exp_num, n_row_groups, pages, 2 * scales_per_page).view(
            dtype=torch.uint8
        )

        weights_and_scales = torch.zeros(
            exp_num,
            n_row_groups,
            pages,
            32 * 1024 + 128,
            dtype=torch.uint8,
            device=weights_w1.device,
        )
        weights_and_scales[:, :, :, : 32 * 1024].copy_(weights)
        weights_and_scales[:, :, :, 32 * 1024 : 32 * 1024 + scales.shape[-1]].copy_(
            scales
        )

        return bias_or_gamma.float(), weights_and_scales.contiguous()
