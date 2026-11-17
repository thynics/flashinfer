# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental GLM-5 TP8 decode on one node with eight B300 GPUs.

These collective operators specialize batch-one decoding to query lengths one
and four, hidden size 6144 and contiguous caches. Rank zero runs the indexer;
ranks one through seven run attention. All eight ranks run MoE.
"""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def create_glm5_tp8_decode_workspace(
    *,
    rank,
    max_seq_len=202752,
    group=None,
    device=None,
    comm_backend=None,
    operators=("attention", "moe"),
):
    """Collectively create and warm an eight-rank decode workspace.

    ``rank`` is a logical rank in the supplied process group, independent of
    ``device``. The default communication backend uses torch.distributed; a
    Gloo group avoids creating an additional NCCL device communicator. All ranks
    must create, prepare, execute, reset and destroy workspaces in matching order.
    ``operators`` may restrict allocation and compilation to one family.
    ``max_seq_len`` defaults to 202752 and supports up to 1052672 tokens
    (a 1M-token prefix plus 4096 decode tokens). Cache and scratch allocation
    grows with this capacity; the private communication arenas stay fixed.

    Call ``destroy()`` collectively after all graph replays finish and before
    destroying the process group. Plans and captured graphs must not outlive
    the workspace. Serialize persistent grids on these GPUs, including grids
    from different workspaces. Order graph replays across CUDA streams explicitly.
    """
    from .experimental.glm5_tp8_decode.workspace import DecodeWorkspace

    return DecodeWorkspace(
        rank=rank,
        max_seq_len=max_seq_len,
        group=group,
        device=device,
        comm_backend=comm_backend,
        operators=operators,
    )


@flashinfer_experimental_api
def pack_glm5_tp8_moe_weights(
    gamma, router, bias, gate, gate_scale, up, up_scale, down, down_scale
):
    """Pack one rank's FP8 expert weights on CPU or CUDA, outside graph capture.

    Gamma is FP32[6144], router BF16[256,6144], bias FP32[256]. Gate and Up
    are FP8 E4M3FN[257,256,6144] with FP32[257,2,48] scales. Down is
    FP8 E4M3FN[257,6144,256] with FP32[257,48,2] scales. Bank zero is the
    shared expert; bank e+1 is routed expert e. Scales multiply FP8 values.
    Expert intermediate dimensions are already sharded across eight ranks.
    The result owns its storage; preserve it unchanged while plans use it.
    """
    from .experimental.glm5_tp8_decode.weights import pack_moe_weights

    return pack_moe_weights(
        gamma, router, bias, gate, gate_scale, up, up_scale, down, down_scale
    )


@flashinfer_experimental_api
def prepare_glm5_tp8_moe(
    x, weights, *, out=None, workspace, enable_pdl=True, cooperative=True
):
    """Prepare a callable fused RMSNorm/router/experts/TP-reduction plan.

    ``x`` and ``out`` are contiguous BF16[q_len,6144], q_len in {1,4}.
    All ranks use the same residual x. The output includes the residual once.
    Preparation is collective and must run outside graph capture. Calling the
    returned plan launches asynchronously on the current stream and returns out.
    Capture the plan call, retain the plan during replay, and update tensor
    contents in place; changing tensor addresses requires a new plan.
    """
    from .experimental.glm5_tp8_decode.backend import prepare_moe

    return prepare_moe(
        x,
        weights,
        out=out,
        workspace=workspace,
        enable_pdl=enable_pdl,
        cooperative=cooperative,
    )


@flashinfer_experimental_api
def pack_glm5_tp8_attention_weights(raw, *, rank):
    """Pack a dictionary of rank-local attention or indexer weights.

    Rank zero consumes indexer weights, ranks one through seven consume MLA
    weights. See the backend README for exact dictionary keys and shapes.
    Values may reside on CPU or CUDA; the result owns its packed storage.
    Rank seven has four real heads and six zero-padded head slots.
    """
    from .experimental.glm5_tp8_decode.attention_weights import pack_attention_weights

    return pack_attention_weights(raw, rank=rank)


@flashinfer_experimental_api
def prepare_glm5_tp8_sparse_attention(
    x,
    weights,
    *,
    caches,
    rope,
    cur_pos,
    out=None,
    workspace,
    indexer=None,
    selection=None,
    enable_pdl=True,
    cooperative=True,
):
    """Prepare fused sparse MLA/indexing, cache writes and TP residual reduction.

    x/out are BF16[q_len,6144]. cur_pos is a device int32 scalar giving the first
    query's cache row, updated in place between calls. The caller guarantees
    0 <= cur_pos and cur_pos+q_len <= workspace.max_seq_len. Cache tensors have
    at least max_seq_len+8 rows: rank zero uses ki_cache BF16[rows,128]; other
    ranks use kv_cache BF16[rows,512] and pe_cache BF16[rows,64]. rope is the
    complete FP32[rows,64] table, not a sliced query window.

    q4 supports indexer="full" (default) or "shared"; q1 supports only "shared".
    Shared mode requires rank-zero selection int32[q_len,2048], containing
    valid, unique, causal IDs for sparse queries. Dense queries use the causal
    prefix. Rank zero then needs no weights, caches or rope. Other ranks receive
    selections from rank zero. Preparation is collective and outside capture;
    retain the plan and workspace throughout graph replay. See the backend README
    for numerical boundaries, lifecycle and concurrency requirements.
    """
    from .experimental.glm5_tp8_decode.backend import prepare_attention

    return prepare_attention(
        x,
        weights,
        caches=caches,
        rope=rope,
        cur_pos=cur_pos,
        out=out,
        workspace=workspace,
        indexer=indexer,
        selection=selection,
        enable_pdl=enable_pdl,
        cooperative=cooperative,
    )
