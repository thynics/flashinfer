# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
from datetime import timedelta
import faulthandler
import os
import traceback

import pytest
import torch
import torch.distributed as dist

from flashinfer.utils import get_compute_capability


class Collective:
    def __init__(self, rank, device, group):
        self.rank, self.device, self.group = rank, device, group

    def gather(self, value):
        result = [None] * 8
        dist.all_gather_object(result, value, group=self.group)
        return result

    def broadcast(self, value, src=0):
        result = [value]
        dist.broadcast_object_list(result, src=src, group=self.group)
        return result[0]

    def sync(self):
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    def checked(self, function):
        """Propagate local reference/validation failures to every rank together."""
        error, result = None, None
        try:
            result = function()
        except Exception:
            error = traceback.format_exc()
        errors = self.gather(error)
        assert not any(errors), "\n".join(
            f"rank {rank}:\n{error}" for rank, error in enumerate(errors) if error
        )
        return result


@pytest.fixture(scope="session")
def collective():
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("run with torchrun --nproc-per-node=8 on eight B300 GPUs")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    count = torch.cuda.device_count()
    if count not in (1, 8):
        pytest.skip(
            "requires one assigned device per process or eight visible assigned GPUs"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # A nonidentity mapping catches code that confuses logical TP rank and CUDA ordinal.
    index = 0 if count == 1 else (local_rank + 3) % count
    device = torch.device("cuda", index)
    if get_compute_capability(device) != (10, 3):
        pytest.skip("GLM TP8 decode is qualified on B300 (SM103)")
    props = torch.cuda.get_device_properties(device)
    if (
        "B300" not in props.name
        or "GB300" in props.name
        or props.multi_processor_count != 148
    ):
        pytest.skip("requires the eight-B300 / 148-SM platform")
    torch.cuda.set_device(device)
    created_world = not dist.is_initialized()
    if created_world:
        dist.init_process_group("gloo", timeout=timedelta(seconds=240))
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=240))
    rank = dist.get_rank(group)
    control = Collective(rank, device, group)
    identities = control.gather(str(props.uuid))
    assert len(set(identities)) == 8, "each logical rank must own a different GPU"
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cudnn.allow_tf32,
        torch.get_num_threads(),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        yield control
    finally:
        control.sync()
        dist.destroy_process_group(group)
        if created_world:
            dist.destroy_process_group()
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cudnn.allow_tf32,
        ) = previous[:3]
        torch.set_num_threads(previous[3])


@pytest.fixture
def collective_watchdog():
    # Report blocked collective stacks without destroying the failure scene.
    # Process termination and allocation cleanup belong to the controller.
    faulthandler.dump_traceback_later(240, repeat=False, exit=False)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
