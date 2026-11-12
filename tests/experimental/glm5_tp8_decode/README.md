# GLM-5 TP8 decode acceptance

These tests exercise the public `flashinfer.glm5_tp8_decode` API. References and
deterministic data are self-contained: no serving runtime, model graph, checkpoint,
private source tree, or old extension is imported.

Run the CPU contract/packing checks in an environment with PyTorch and pytest:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/experimental/glm5_tp8_decode/test_cpu.py
```

Run GPU tests serially on one node with eight assigned B300 GPUs. The fixture uses
eight real processes, Gloo metadata exchange and FlashInfer's public workspace
(TorchDistBackend and symmetric device memory). With eight visible devices, it
maps logical rank `r` to device `(r + 3) % 8` to catch logical/physical rank bugs.
If each process sees only its own assigned GPU, each uses local ordinal zero.
This test package neither acquires GPUs nor submits/retries jobs.

Start with a bounded smoke test:

```bash
timeout --signal=TERM --kill-after=15s 900s \
  torchrun --standalone --nproc-per-node=8 -m pytest -q -x -p no:cacheprovider \
  tests/experimental/glm5_tp8_decode/test_moe.py -k 847223
```

Run the lazy-loading checks in a **fresh** process group before the main suite:

```bash
CUDA_MODULE_LOADING=LAZY timeout --signal=TERM --kill-after=15s 900s \
  torchrun --standalone --nproc-per-node=8 -m pytest -q -x -p no:cacheprovider \
  tests/experimental/glm5_tp8_decode/test_lazy_first_exchange.py
```

The first collective exchange after workspace warmup is graph replay. Compilation
and kernel-function initialization happen before capture. The per-test watchdog
prints worker stacks and terminates after 240 seconds; the outer command also
bounds teardown if a CUDA peer-wait stalls. There is no rescue write to protocol
state and no automatic resubmission. For slow first-time compilation, warm the
workspace in a separate bounded process before running acceptance.

Full acceptance (run each command separately, with a suitable outer timeout):

```bash
torchrun --standalone --nproc-per-node=8 -m pytest -q -x -p no:cacheprovider \
  tests/experimental/glm5_tp8_decode/test_moe.py
torchrun --standalone --nproc-per-node=8 -m pytest -q -x -p no:cacheprovider \
  tests/experimental/glm5_tp8_decode/test_attention.py
```

The MoE matrix retains three q1 weight seeds, the q4 weight seed and the explicit
cancellation fixture. q1 checks normalized input, legal approximate-sigmoid
routing, BF16 intermediate values and Down from actual validated intermediates.
q4 checks router-dot rounding bounds, enumerates legal ambiguous top-eight sets
and carries the analytical sigmoid error through expert contributions. Both use
per-128 FP8 activation quantization and the established two-BF16-ULP error metric
with an absolute magnitude floor of `1/16`. The cancellation fixture distinguishes
one final BF16 round from a second round before residual addition.

RMSNorm evaluates `x * (gamma * rstd)` before BF16 rounding. The q4 routed
normalization and shared-expert normalization assign input pairs to lanes
differently, so shared scratch cannot establish the router's exact normalized
input. The router oracle independently reconstructs its FP32 square-sum order
and enumerates the BF16 rows produced by the documented `rsqrtf` accuracy range.
CUDA 13.2 lists two FP32 ULP from the correctly rounded reciprocal square root;
this is a [tested accuracy bound, not an exhaustive guarantee](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html).
One scalar reciprocal square root must explain **all 256 logits in a token
row**. Each candidate retains the original dot-product gate
`gamma_64 * sum(abs(normalized_x * router))`, with
`gamma_64 = 64 * 2^-24 / (1 - 64 * 2^-24)`; the dot tolerance is not enlarged.
The independent RMSNorm check retains its two floored BF16 ULP gate. CPU
negative controls reject BF16-rounded logits, mixed per-expert norm candidates,
and an incorrect normalized input, and cover FP32 spacing at exponent edges.

Attention covers q4 full/shared indexers and q1 accepted-row sharing, dense/sparse
boundaries at 0, 31, 63 and 2043 through 2049, and contexts up to 196608 tokens.
The long-context matrix retains ordinary and concentrated indexer-cache families.
Actual-input stage checks preserve BF16 boundaries, the two-ULP projection gates,
one-row-max-ULP attention gate, exact zero rows and the rotation-invariant RoPE
pair metric. Causal logits have the established `1e-5` relative-max-error bound.
Sparse selection must contain the **exact** top-2048 set of actual logits;
threshold ties need no particular order. Dense queries use their causal prefix.
Both repeated full-indexer replays run every numerical and communication gate;
their sorted selected-ID sets must match exactly. Publication slot order is free,
so full-indexer outputs are not required to match bit for bit across calls.
Shared replays with the same fixed selection order retain bitwise output checks.
The q4 source-FP32 UnProj endpoint retains its two-ULP relative-floor `1/16` gate;
a compound ideal FP64 endpoint is not used as a replacement acceptance rule.

For both families, independently decoded current-generation wire records are
summed in logical-rank order in FP32, followed by residual addition and one BF16
round. Every rank must match this result bit for bit. The tests also cover input,
weight, cache and output guards; q1/q4 alternation; changed inputs, positions and
selections without recapture; interleaved independent workspaces; destroy/recreate;
collective host-validation failures; and preservation of generations across request
reset. A side-stream lifetime test drops a temporary MoE plan and its input/weight
owners, churns their allocation sizes on the original stream, then checks the
completed output. MoE scratch is finite-poisoned only once, and tests never refill its NaN
sentinels or reset state/inboxes between calls. Attention-only `FFFF` and `FSSS`
chains delay rank zero before the second layer to exercise exit credits.
These four launches remain uninterrupted. Afterward, all observed outputs must
agree across ranks and the final call passes the full stage and exact-wire gates.
The full indexer's actual ID order is copied and replayed through independently
validated shared mode; every observed full output must match that reconstruction
bit for bit. Shared outputs must match the validated fixed-selection baseline,
and full-indexer sorted sets must match their validated baseline.

These tests establish standalone operator acceptance. They do not certify a
complete serving model or substitute for a separate same-protocol performance
comparison.
