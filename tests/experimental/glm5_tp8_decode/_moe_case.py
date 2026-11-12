# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
import torch

from . import _api
from . import data
from . import moe_reference as ref
from ._moe_checks import check_moe


class MoECase:
    def __init__(
        self, control, weight_seed=531017, *, cancellation=False, raw=None, packed=None
    ):
        self.control = control
        self.cancellation = cancellation
        self.ws = None
        self.plans = {}
        self.graphs = {}
        self.counts = {1: 0, 4: 0}
        self.stream = torch.cuda.Stream(device=control.device)
        if raw is None:
            raw = data.make_moe_raw(control.rank, weight_seed, control.device)
            replicated = (
                {key: raw[key].cpu() for key in ("gamma", "router", "bias")}
                if control.rank == 0
                else None
            )
            replicated = control.broadcast(replicated)
            raw.update(
                {key: value.to(control.device) for key, value in replicated.items()}
            )
            if cancellation:
                raw = data.make_moe_cancellation(raw, control.rank, control.device)
        self.raw = raw
        self.packed = (
            control.checked(lambda: _api.pack_moe(raw)) if packed is None else packed
        )
        self.weight_snapshots = {
            name: getattr(self.packed, name).clone()
            for name in ("gamma", "router", "bias", "upgate", "down", "down_scale")
        }
        self.x = {}
        self.out_storage = {}
        self.out = {}
        try:
            self.ws = _api.workspace(control, operators=("moe",))
            for q_len in (1, 4):
                self.x[q_len] = torch.empty(
                    (q_len, 6144), dtype=torch.bfloat16, device=control.device
                )
                self.out_storage[q_len] = torch.full(
                    (q_len + 1, 6144), 17, dtype=torch.bfloat16, device=control.device
                )
                self.out[q_len] = self.out_storage[q_len][:q_len]
                self.plans[q_len] = _api.prepare_moe(
                    self.x[q_len], self.packed, self.out[q_len], self.ws
                )
            # Finite old contents must be invalidated by the operator itself.
            # Do this only once: no test-side sentinel refill accompanies calls.
            self.plans[4]._diagnostics["scores"].fill_(-13.125)
            self.plans[4]._diagnostics["mid"].fill_(17)
            control.sync()
        except Exception:
            if self.ws is not None:
                self.ws.destroy()
            raise

    def capture(self):
        self.control.sync()
        before = {
            q: int(plan._diagnostics["generation"]()) for q, plan in self.plans.items()
        }
        for q_len, plan in self.plans.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                plan()
            self.graphs[q_len] = graph
        self.control.sync()
        assert {
            q: int(plan._diagnostics["generation"]()) for q, plan in self.plans.items()
        } == before

    def run(self, q_len, seed, *, graph=False, validate=True):
        expected_input = data.make_input(
            q_len, seed, "cpu", cancellation=self.cancellation
        )
        with torch.cuda.stream(self.stream):
            self.x[q_len].copy_(expected_input)
            self.out_storage[q_len].fill_(17)
            if graph:
                self.graphs[q_len].replay()
            else:
                result = self.plans[q_len]()
                assert result.data_ptr() == self.out[q_len].data_ptr()
        self.counts[q_len] += 1
        self.control.sync()

        def guards():
            assert ref.bitwise(self.x[q_len].cpu(), expected_input), (
                "input was modified"
            )
            assert bool((self.out_storage[q_len][q_len:] == 17).all()), (
                "output guard was modified"
            )
            generation = int(self.plans[q_len]._diagnostics["generation"]())
            assert generation == self.counts[q_len], (
                q_len,
                generation,
                self.counts[q_len],
            )
            return generation

        generation = self.control.checked(guards)
        assert self.control.gather(generation) == [generation] * 8
        if validate:
            return check_moe(
                self.control,
                self.plans[q_len],
                self.raw,
                self.x[q_len],
                self.out[q_len],
                cancellation=self.cancellation and seed != 23,
            )
        return self.out[q_len].cpu().clone()

    def close(self):
        if self.ws is None:
            return
        self.control.sync()

        def immutable_weights():
            for name, snapshot in self.weight_snapshots.items():
                assert ref.bitwise(getattr(self.packed, name), snapshot), (
                    f"packed weight {name} was modified"
                )

        self.control.checked(immutable_weights)
        self.graphs.clear()
        self.ws.destroy()
        self.ws.destroy()  # idempotent
