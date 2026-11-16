# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Minimal GLM-5 TP8 MoE decode example on one node with eight B300 GPUs.

Run from an installed FlashInfer environment:

    torchrun --standalone --nproc-per-node=8 examples/glm5_tp8_decode.py

This specialized operator supports TP=8, hidden size 6144, 256 routed experts
plus one shared expert, local intermediate size 256, and q_len=1 or q_len=4.
Its qualified target is B300 with 148 SMs; it is not a general-purpose MoE API.
The example uses zero expert weights, so the correct output equals the residual
input. Real callers supply their own tensor-parallel FP8 checkpoint shards and
FP32 block scales. Weight packing happens once, outside CUDA graph capture.
"""

from datetime import timedelta
import os

import torch
import torch.distributed as dist

from flashinfer.glm5_tp8_decode import (
    create_glm5_tp8_decode_workspace,
    pack_glm5_tp8_moe_weights,
    prepare_glm5_tp8_moe,
)


def make_zero_expert_weights(device):
    gamma = torch.ones(6144, dtype=torch.float32, device=device)
    router = torch.zeros((256, 6144), dtype=torch.bfloat16, device=device)
    bias = torch.zeros(256, dtype=torch.float32, device=device)
    # Use a byte view so CPU/torch builds need no FP8 fill kernel. Gate and Up
    # may share this immutable raw tensor during load-time packing.
    gate_up = torch.zeros((257, 256, 6144), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    gate_up_scale = torch.ones((257, 2, 48), dtype=torch.float32, device=device)
    down = torch.zeros((257, 6144, 256), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    down_scale = torch.ones((257, 48, 2), dtype=torch.float32, device=device)
    return pack_glm5_tp8_moe_weights(
        gamma,
        router,
        bias,
        gate_up,
        gate_up_scale,
        gate_up,
        gate_up_scale,
        down,
        down_scale,
    )


def check_residual(actual, expected, group):
    # A collective check ensures all workers take the same cleanup path.
    torch.cuda.synchronize(actual.device)
    local_ok = torch.equal(actual, expected)
    accepted = [None] * dist.get_world_size(group)
    dist.all_gather_object(accepted, local_ok, group=group)
    if not all(accepted):
        raise AssertionError(f"zero-expert residual check failed on ranks {accepted}")


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("launch this example with torchrun --nproc-per-node=8")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_index = 0 if torch.cuda.device_count() == 1 else local_rank
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    workspace = None
    graphs = {}
    try:
        # Collective creation/warmup initializes all kernel entry points before
        # the first peer exchange, including under CUDA_MODULE_LOADING=LAZY.
        workspace = create_glm5_tp8_decode_workspace(
            rank=rank,
            group=group,
            device=device,
            max_seq_len=4096,
            operators=("moe",),
        )
        workspace.warmup()
        weights = make_zero_expert_weights(device)
        stream = torch.cuda.Stream(device=device)
        inputs, outputs, plans = {}, {}, {}
        for q_len in (1, 4):
            inputs[q_len] = torch.ones(
                (q_len, 6144), dtype=torch.bfloat16, device=device
            )
            outputs[q_len] = torch.empty_like(inputs[q_len])
            plans[q_len] = prepare_glm5_tp8_moe(
                inputs[q_len],
                weights,
                out=outputs[q_len],
                workspace=workspace,
            )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

        # Every rank issues the same sequence. Keep calls on one stream so the
        # persistent collective grids and their shared scratch are serialized.
        for q_len in (1, 4):
            with torch.cuda.stream(stream):
                plans[q_len]()
            check_residual(outputs[q_len], inputs[q_len], group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plans[q_len]()
            graphs[q_len] = graph
            del graph

        # Addresses stay fixed while values change in place; no recapture or
        # host-side counter updates are required when alternating q1 and q4.
        for q_len, value in ((4, -0.75), (1, 0.5), (4, 1.25), (1, -1.0)):
            with torch.cuda.stream(stream):
                inputs[q_len].fill_(value)
                graphs[q_len].replay()
            check_residual(outputs[q_len], inputs[q_len], group)
        if rank == 0:
            print("PASS: TP8 q1/q4 eager calls and graph replay with changed inputs")
    finally:
        if workspace is not None:
            # Retain workspace, plans, weights and bound tensors until every
            # replay is complete. Captured graphs cannot outlive their workspace.
            torch.cuda.synchronize(device)
            graphs.clear()
            workspace.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
