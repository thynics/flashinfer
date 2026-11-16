# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Object-collective payloads that avoid pickling FP8 tensor storage."""

import torch


def tensor_bytes(tensor):
    value = tensor.detach().contiguous()
    if value.ndim == 0:
        value = value.reshape(1)
    if value.stride(-1) != 1:
        value = value.clone(memory_format=torch.contiguous_format)
    return value.view(torch.uint8).cpu()


def tensor_from_bytes(payload, template):
    assert payload.dtype == torch.uint8 and payload.device.type == "cpu"
    return payload.view(template.dtype).reshape(template.shape).to(template.device)
