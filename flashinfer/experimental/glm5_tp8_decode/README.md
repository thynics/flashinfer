# Experimental GLM-5 TP8 decode

Owner: Long Cheng (@thynics).

This backend exposes six specialized decode kernels through two collective
operator families. It is intended for one node with eight NVLink-connected
B300 GPUs (SM103, 148 SMs each), batch one, query length one or four, hidden
size 6144, and contiguous BF16 caches. It has no serving-framework dependency.
Calling an API in `flashinfer.glm5_tp8_decode` is the explicit experimental opt-in.
The kernels compile on demand and are not registered in the AOT package.
Use a source checkout or a nightly wheel that includes experimental sources;
release wheels built with `FLASHINFER_WHEEL_EXCLUDE_EXPERIMENTAL_SOURCES=1`
omit these CUDA files.

The fused stages share a collective workspace and fixed model geometry, which
form one explicit experimental API contract. The proposed graduation review
checkpoint is December 14, 2026, targeting a decision for FlashInfer 0.8.0,
subject to maintainer agreement. That review covers deployment-independent
ownership, graph replay, numerical bounds, reproducible operator timings, and
whether the fixed geometry should remain a model-specific API or move behind
broader operations. The experimental tracking issue accompanies the PR.

## Operators and ownership

- **MoE:** RMSNorm, replicated router, top-eight routed experts, one shared
  expert, tensor-parallel reduction and residual addition. All eight ranks
  participate; the intermediate dimension per rank is 256. There are 256 routed
  experts and 257 weight banks including the shared expert.
- **Sparse attention:** logical rank zero computes the q4 indexer or publishes a
  supplied selection; ranks 1–7 compute MLA and update their latent/positional
  caches. Each has ten head slots; rank 7 contains four real heads and six
  zero-padded slots. The operator includes output projection, tensor-parallel
  reduction and residual addition. q1 reuses an accepted selection; q4 can use
  full indexing or a supplied selection. Full indexing selects the exact top
  2048 of actual computed logits. For unchanged inputs, the selected set is stable;
  its slot order is unspecified.

Create a workspace collectively with `create_glm5_tp8_decode_workspace` and a
logical rank in an eight-rank communication group. Physical CUDA ordinals may
be permuted or masked. The workspace checks distinct GPU identities, device
geometry and native peer atomic support. A Gloo process group is suitable for
metadata and IPC-handle exchange. `operators=("moe",)` or `("attention",)` avoids
allocating and compiling the other family.

Pack weights once outside capture, then prepare plans with fixed tensor
addresses. Preparation validates rank-local arguments collectively before any
peer-waiting kernel launches. Packed weights own their storage. Treat their
objects, dictionaries and contents as immutable while plans use them. Plans
retain their operands and private device descriptors. The caller owns inputs,
outputs, caches and RoPE tables; a workspace owns communication records,
generation counters and operator scratch.

A plan call launches on the current stream and returns its output without a
host synchronization. Capture that call and update input contents, positions
and selections in place between replays. Keep the plan and workspace alive for
all replays; CUDA graphs do not retain Python pointer owners. Never resize or
replace bound tensors during replay. Destroy the workspace collectively after
all work finishes and before destroying its process group. Destroying a workspace
invalidates its plans. `memory_usage()` reports symmetric physical allocation
bytes and local scratch/descriptor bytes; caller-owned weights and caches are
excluded.

All ranks must execute the same operator sequence. Do not overlap these
persistent peer-waiting grids, including grids from separate workspaces, on the
same GPUs. Eager calls on one workspace order successive streams. The caller
must order graph replays across streams, order input producers before execution,
and order output consumers after execution. Independent workspaces may be
interleaved when their launches are serialized. `reset_request()` clears the
selection-consumption channel while preserving collective generations; it is a
collective operation outside capture. There is no per-call external NaN fill.

## Tensor contracts

Inputs and outputs are contiguous, 16-byte-aligned BF16 `[q_len,6144]` tensors.
All ranks provide the same residual input. Output may not alias inputs, weights,
cache or workspace scratch. Autograd is unsupported. Native CUDA cooperative
launch and programmatic dependent launch are enabled by default. Initialization
checks the kernel's resource occupancy before execution.

MoE packing takes gamma FP32 `[6144]`, router BF16 `[256,6144]`, bias FP32
`[256]`, Gate and Up E4M3FN `[257,256,6144]` with FP32 `[257,2,48]` scales,
and Down E4M3FN `[257,6144,256]` with FP32 `[257,48,2]` scales. The scale values
multiply decoded FP8 codes. Packing rearranges bytes without requantization.
Bank zero is shared; routed expert e is bank e+1.

Attention packing accepts exactly the following raw dictionary keys. Common
keys on every rank:

| Key | Dtype | Shape |
|---|---|---|
| x_rmsnorm_gamma | FP32 | [6144] |
| q_a_weights | E4M3FN | [2048,6144] |
| q_a_scale | FP32 | [16,48] |

Additional keys on rank zero:

| Key | Dtype | Shape |
|---|---|---|
| wk_weights / wk_scale | E4M3FN / FP32 | [128,6144] / [1,48] |
| id_score_weights | BF16 | [32,6144] |
| q_rmsnorm_gamma_qi | FP32 | [2048] |
| wqi_weights / wqi_scale | E4M3FN / FP32 | [4096,2048] / [32,16] |
| k_weights / k_bias | FP32 | [128] / [128] |

Additional keys on ranks 1–7:

| Key | Dtype | Shape |
|---|---|---|
| kv_a_weights / kv_a_scale | E4M3FN / FP32 | [512,6144] / [4,48] |
| w_pe_weights / w_pe_scale | E4M3FN / FP32 | [64,6144] / [1,48] |
| q_rmsnorm_gamma | FP32 | [2048] |
| wqb_weights / wqb_scale | E4M3FN / FP32 | [2560,2048] / [20,16] |
| kv_rmsnorm_gamma | FP32 | [512] |
| wkv_b1_weights / wkv_b1_scale | E4M3FN / FP32 | [10,512,192] / [10,4,3] |
| wkv_b2_weights / wkv_b2_scale | E4M3FN / FP32 | [10,256,512] / [10,4,4] |
| unproj_weights / unproj_scale | E4M3FN / FP32 | [6144,2560] / [48,20] |

The caller supplies the six zero-padded head slots on rank 7. For shared-index
mode, rank zero may pass `weights=None, caches=None, rope=None`.

Cache dictionaries contain `ki_cache` BF16 `[rows,128]` on rank zero when full
indexing is used, and `kv_cache` BF16 `[rows,512]` plus `pe_cache` BF16 `[rows,64]`
on other ranks. All caches have at least `workspace.max_seq_len+8` rows; cache
rows may include padding. RoPE is the complete FP32 `[rows,64]` table, with at
least as many rows as the caches. Pass its base address, not a current-query
slice. `cur_pos` is a device int32 scalar, 4-byte aligned, holding the first query
position. Its value must satisfy `0 <= cur_pos` and
`cur_pos+q_len <= workspace.max_seq_len` on **every replay**. Preparation checks
the initial value; subsequent replay values are a caller precondition.

The default capacity remains 202752 tokens. An explicit `max_seq_len` supports
up to 1052672 tokens, allowing a 1M-token prefix and 4096 decode tokens. Private
communication arenas retain their original size; public caches and the
workspace's logits/candidate scratch grow with the requested capacity. A
single layer uses approximately 576 MiB of KV/PE cache on each attention rank
at 512K tokens, or 1152 MiB at 1M tokens, excluding padding and weights.

Full q4 indexing switches to an exact long-row selector above 202760 scored
rows. P4 publishes the full 32-bit ordering key of the FP32 maximum for each
32-token causal group. Group IDs are implicit: when maxima tie, every token ID
in an earlier group is smaller than every ID in a later group. Lower group
indices therefore resolve group ties without discarding any score bits.

The selector retains a bounded superset of groups containing every global
top-2048 token. On long rows, maxima over four consecutive group keys provide
a conservative threshold for a coarse group filter. Adaptive radix refinement
bounds the retained set; if the coarse filter cannot do so, selection continues
over the complete group-key row. Equal full score keys are resolved by lower
group index when the set must be reduced to exactly 2048 groups. These filters
preserve the full score keys and every possible winning token.

P4 also publishes the exact lowest lane attaining each maximum and a conservative
upper bound on the group's second-highest score key, in an aligned metadata
region after the group maxima. If that bound proves that only the maximum can
reach the retained set's conservative score floor, the selector reconstructs its
exact score/ID key from the metadata. Other groups are gathered from their
original FP32 scores. The bound only decides when these loads can be skipped;
it does not replace the full score keys used for ranking.

Group classification and candidate staging share a scan, which also counts the
first radix histogram for that invocation. Final selection reuses that histogram
and compares complete 64-bit score/ID keys. A target radix bucket containing at most 384 keys can be ranked directly
with exact comparisons. If the candidate pool exceeds its bounded storage,
selection falls back to an exact scan of the full row; candidates are never
truncated. The existing score ordering is preserved, including signed zero,
with smaller token IDs winning equal-score ties. Slot order remains unspecified.
Group summaries and aligned metadata reuse existing candidate scratch, with no
new caller buffer or changes to the collective publication protocol.

Shared mode takes rank-zero `selection` int32 `[q_len,2048]`. For sparse queries,
IDs must be unique, in cache bounds and causal for their query. Dense queries
use their causal prefix and ignore unused selection entries. Runtime values
inside captured tensors are not host-validated on replay. Finite inputs and
scale values are required. Counter generations are 32-bit. The supported lifetime is fewer than 2^24
calls per query-length counter; recreate workspaces before reaching that limit
to avoid scaled-counter wraparound.

## Numerical and protocol boundaries

Weights retain FP8 codes and FP32 scales. Activations are quantized per 128
values where required by the original computation. BF16 intermediate boundaries,
FP32 accumulation/reduction order, approximate sigmoid routing and RoPE arithmetic
are part of the operator contract. This implementation uses precise compile
flags and does not enable `--use_fast_math`.

MoE publication reserves exact F32 and BF16 pending encodings. Computed NaNs
retain their numerical class and are published with a different canonical
encoding. Each polled component uses a matching scalar access width; consumers
wait for every component. Finite inputs can still overflow internally or
normalize zero routing mass and produce nonfinite numerical results.

Each rank emits BF16 partials. Wire records are summed in logical rank order in
FP32, the residual is added in FP32, and one final BF16 rounding produces output.
Tests independently decode current-generation records and require bitwise
agreement with that reduction. Full indexing uses atomic tickets to pack IDs,
so the same selected set can reach attention in a different order on replay.
Consequent floating-point reduction-order changes can alter output bits across
full-indexer calls. Tests require stable sorted ID sets and validate every call
against the numerical contract; fixed-selection replays retain bitwise checks. Projection and MoE stage tests retain their
bounded BF16 error metrics and explicitly handle analytically ambiguous router
ties. A compound ideal high-precision endpoint is diagnostic, not a substitute
for the established stage-level contract.

Communication uses aligned scalar 128-bit system-scope record loads/stores and
native peer atomics. Lifetime credits prevent reuse while peers still read a
previous generation. Initialization and NaN sentinels are internal to the
workspace/kernel; callers must not reset protocol storage between layers.

## Running

From a source checkout on eight assigned B300 GPUs:

```bash
torchrun --standalone --nproc-per-node=8 examples/glm5_tp8_decode.py
```

The example uses zero expert weights to demonstrate ownership and graph replay.
Acceptance commands and numerical coverage are documented in
`tests/experimental/glm5_tp8_decode/README.md`. Run that package with eight
processes; single-process invocation skips the GPU tests. Tests include changed
inputs/positions, q1/q4 alternation, interleaved workspaces, lifecycle failures,
first graph exchange under lazy module loading, and physical-device permutation.
Standalone operator validation does not certify a complete serving model's
startup time, memory headroom or end-to-end determinism.
