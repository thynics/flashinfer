# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Stage gates, exact selection membership, and exact TP8 communication math."""

import torch

from . import attention_reference as ref
from .moe_reference import bitwise
from ._records import attention_rank0_partials


def check_indexer(control, plan, raw, caches, x, rope, past, *, indexer, selection):
    q_len, diag = x.shape[0], plan._diagnostics
    freqs = rope[past : past + q_len]

    def indexer_checks():
        if control.rank != 0:
            return None
        if indexer != "full":
            return selection.cpu().clone()
        assert q_len == 4
        xn = ref.source_indexer_rms(x, raw["x_rmsnorm_gamma"])
        ref.assert_close(
            diag["qa"],
            ref.ref_gemv_w8a16(xn, raw["q_a_weights"], raw["q_a_scale"]),
            2,
            2**-14,
        )
        ref.assert_close(
            diag["ki"],
            ref.ref_gemv_w8a16(xn, raw["wk_weights"], raw["wk_scale"]),
            2,
            2**-14,
        )
        ref.assert_close(
            diag["headw"], ref.ref_gemv_bf16(xn, raw["id_score_weights"]), 2, 2**-14
        )
        qn = ref.source_indexer_rms(diag["qa"], raw["q_rmsnorm_gamma_qi"])
        iq = ref.ref_gemv_w8a16(qn, raw["wqi_weights"], raw["wqi_scale"]).reshape(
            4, 32, 128
        )
        ref.assert_close(diag["iq"], iq, 2, 2**-14)
        ref.assert_close(diag["iqrt"], ref.ref_rotate_q(diag["iq"], freqs), 1, 2**-14)
        new_k = ref.ref_k_ln_rope_had(
            diag["ki"], raw["k_weights"], raw["k_bias"], ref.LN_EPS, freqs
        )
        ref.assert_close(caches["ki_cache"][past : past + 4], new_k, 1, 2**-14)
        # The source only defines causal logits, and dense rows do not publish IDs.
        logits = diag["logits"].reshape(-1)[: 4 * (past + 4)].reshape(4, past + 4)
        if past >= 2044:
            expected = ref.ref_index_score(
                caches["ki_cache"], past + 4, diag["iqrt"], diag["headw"]
            )
            causal = (
                torch.arange(past + 4, device=x.device)[None, :]
                <= torch.arange(past, past + 4, device=x.device)[:, None]
            )
            actual, expected = logits[causal], expected[causal]
            assert bool(torch.isfinite(actual).all())
            relative = float((actual - expected).abs().max()) / max(
                float(expected.abs().max()), 1e-30
            )
            assert relative <= 1e-5, {
                "causal_logit_relative_error": relative,
                "budget": 1e-5,
            }
        return ref.check_exact_topk(logits, diag["ids"], past, q_len).cpu()

    ids = control.checked(indexer_checks)
    return control.broadcast(ids)


def check_attention(
    control, plan, raw, caches, initial, x, out, rope, past, *, indexer, selection
):
    q_len, diag = x.shape[0], plan._diagnostics
    freqs = rope[past : past + q_len]
    ids = check_indexer(
        control, plan, raw, caches, x, rope, past, indexer=indexer, selection=selection
    ).to(control.device)

    def records():
        if control.rank == 0:
            return attention_rank0_partials(
                diag["records"], q_len, int(diag["generation"]())
            )
        return None

    partials = control.broadcast(control.checked(records))
    outputs = control.gather(out.cpu())
    expected_exit = ref.ref_tp_exit_sum(list(partials.unbind(0)), x.cpu())
    assert all(bitwise(value, expected_exit) for value in outputs), (
        "TP8 wire sum + residual must be bit exact on all ranks"
    )

    def local_checks():
        for name, cache in caches.items():
            before = initial[name]
            if control.rank == 0 and indexer != "full":
                assert bitwise(cache, before), (
                    "shared indexer changed its caller-owned KI cache"
                )
            else:
                assert bitwise(cache[:past], before[:past]), (
                    f"{name} wrote before cur_pos"
                )
                assert bitwise(cache[past + q_len :], before[past + q_len :]), (
                    f"{name} wrote beyond the new query rows"
                )
        if control.rank == 0:
            return None
        pipeline = ref.attention_pipeline(raw, initial, x, past, freqs, ids)
        ref.assert_close(diag["post_kv"][:q_len], pipeline["kv"], 2, 2**-14)
        kv = ref.ref_rmsnorm(
            diag["post_kv"][:q_len], raw["kv_rmsnorm_gamma"], ref.KV_EPS
        )
        ref.assert_close(caches["kv_cache"][past : past + q_len], kv, 1, 2**-14)
        ref.assert_rope_close(
            caches["pe_cache"][past : past + q_len], pipeline["pe_row"]
        )
        actual_latent = diag["post_o"][:q_len]
        ref.assert_close(actual_latent, pipeline["latent"], 2, 1, per_row=True)
        expected_latent = ref.ref_sparse_attention_glm53(
            diag["post_qnope"][:q_len],
            diag["post_qpe"][:q_len],
            caches["kv_cache"],
            caches["pe_cache"],
            ids,
            past,
        ).reshape_as(actual_latent)
        ref.assert_close(actual_latent, expected_latent, 1, 1, per_row=True)
        zero = expected_latent.abs().amax(-1) == 0
        assert bitwise(actual_latent[zero], expected_latent[zero]), (
            "zero latent rows must be exact"
        )
        projo = (
            ref.ref_gemv_w8a16(
                actual_latent.permute(1, 0, 2).contiguous(),
                raw["wkv_b2_weights"],
                raw["wkv_b2_scale"],
                row_block=64,
                col_block=128,
            )
            .permute(1, 0, 2)
            .contiguous()
        )
        ref.assert_close(diag["post_projo"][:q_len], projo, 2, 2**-14)
        actual_projo = diag["post_projo"][:q_len].reshape(q_len, 2560)
        unproj = ref.ref_gemv_w8a16(
            actual_projo, raw["unproj_weights"], raw["unproj_scale"]
        )
        ref.assert_close(partials[control.rank - 1].to(x.device), unproj, 2, 2**-14)
        return ref.source_unproj(
            actual_projo, raw["unproj_weights"], raw["unproj_scale"]
        ).cpu()

    predicted = control.gather(control.checked(local_checks))
    if q_len == 4:
        # This preserves the source-FP32 endpoint gate. q1's q4-source-order
        # comparison was diagnostic only; its actual-input stages remain gates.
        source_exit = ref.ref_tp_exit_sum(predicted[1:], x.cpu())
        ref.assert_close(outputs[0], source_exit, 2, 1 / 16)
    return outputs[0].clone()
