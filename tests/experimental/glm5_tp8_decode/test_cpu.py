# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only numerical-contract and packed-storage checks."""

import pickle

import pytest
import torch

from flashinfer.experimental.glm5_tp8_decode import _moe_packing
from flashinfer.experimental.glm5_tp8_decode import _attention_packing
from flashinfer.experimental.glm5_tp8_decode import geometry

from . import attention_reference as attention_ref
from . import moe_reference as moe_ref
from ._tensor_wire import tensor_bytes, tensor_from_bytes


@pytest.mark.parametrize(
    "capacity", [1, 202752, 202753, 202760, 524288, 1048576, 1052672]
)
def test_long_context_capacity_accepts_the_extended_logical_range(capacity):
    assert geometry.check_capacity(capacity) == capacity


@pytest.mark.parametrize("capacity", [0, -1, 1052673, True, False, 1.0, None])
def test_long_context_capacity_rejects_invalid_values(capacity):
    with pytest.raises(ValueError):
        geometry.check_capacity(capacity)


def test_long_context_extension_preserves_the_default_allocation_capacity():
    import inspect

    from flashinfer.glm5_tp8_decode import create_glm5_tp8_decode_workspace
    from flashinfer.experimental.glm5_tp8_decode.workspace import DecodeWorkspace

    assert geometry.DEFAULT_MAX_SEQ_LEN == 202752
    assert geometry.MAX_SEQ_LEN == 1052672
    for factory in (create_glm5_tp8_decode_workspace, DecodeWorkspace):
        assert inspect.signature(factory).parameters["max_seq_len"].default == 202752


def test_long_context_reference_uses_unsigned_key32_order():
    # Do not pack key32<<32 into signed torch.int64: positive-score composites
    # would acquire bit63. The independent reference ranks key32 and ID separately.
    scores = torch.tensor([[-1.0, -0.0, 0.0, 1.0]], dtype=torch.float32)
    expected_keys = torch.tensor(
        [[0x407FFFFF, 0x7FFFFFFF, 0x80000000, 0xBF800000]], dtype=torch.int64
    )
    assert torch.equal(attention_ref.topk_key(scores), expected_keys)
    selected, lengths = attention_ref.ref_topk_select(scores, 3, topk=2)
    assert torch.equal(selected, torch.tensor([[2, 3]], dtype=torch.int32))
    assert lengths.tolist() == [2]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16, torch.float32])
def test_weight_object_collective_serialization_preserves_bits(dtype):
    source = torch.arange(32, dtype=torch.float32).to(dtype).reshape(4, 8).T
    payload = pickle.loads(pickle.dumps(tensor_bytes(source)))
    restored = tensor_from_bytes(payload, source)
    assert moe_ref.bitwise(restored, source)


def test_bitwise_singleton_strided_slice_preserves_exact_bits():
    storage = torch.zeros((1, 6144), dtype=torch.bfloat16)
    value = storage[:, 0]
    assert value.stride() == (6144,) and value.is_contiguous()
    expected = torch.tensor([1 / 256], dtype=torch.bfloat16)
    value.copy_(expected)
    assert moe_ref.bitwise(value, expected)
    changed = torch.tensor([1 / 256 + 3 * 2**-15], dtype=torch.bfloat16)
    assert not moe_ref.bitwise(value, changed)
    payload = pickle.loads(pickle.dumps(tensor_bytes(value)))
    assert moe_ref.bitwise(tensor_from_bytes(payload, value), expected)
    value.fill_(-0.0)
    expected.zero_()
    assert torch.equal(value, expected)
    assert not moe_ref.bitwise(value, expected), "signed zero must compare by bits"


def test_tp_exit_has_one_final_bf16_rounding():
    residual = torch.tensor([[2.0]], dtype=torch.bfloat16)
    partials = [
        torch.tensor([[-2.0]], dtype=torch.bfloat16),
        torch.tensor([[1 / 256]], dtype=torch.bfloat16),
    ]
    partials += [torch.zeros_like(residual) for _ in range(6)]
    expected = torch.tensor([[1 / 256]], dtype=torch.bfloat16)
    assert torch.equal(moe_ref.reduce_ranks(partials, residual), expected)
    assert torch.equal(attention_ref.ref_tp_exit_sum(partials, residual), expected)
    # Negative control: rounding the partial sum before adding the residual is wrong.
    rounded_sum = sum(p.float() for p in partials).bfloat16()
    assert not torch.equal(
        (rounded_sum.float() + residual.float()).bfloat16(), expected
    )


def test_q128_zero_floor_and_batch_independence():
    x = torch.zeros(1, 256, dtype=torch.bfloat16)
    codes, scale = moe_ref.q128(x)
    assert torch.equal(codes.float(), torch.zeros_like(x, dtype=torch.float32))
    assert bool((scale > 0).all())
    expected_scale = torch.tensor(1e-4, dtype=torch.float32) * torch.tensor(
        1 / 448, dtype=torch.float32
    )
    assert torch.equal(scale, expected_scale.expand_as(scale))
    x[0, 0] = 3.5
    one, one_scale = moe_ref.q128(x)
    four, four_scale = moe_ref.q128(x.repeat(4, 1))
    assert torch.equal(four.view(torch.uint8), one.view(torch.uint8).repeat(4, 1))
    assert torch.equal(four_scale, one_scale.repeat(4, 1))


def test_exact_topk_accepts_unordered_set_but_rejects_wrong_membership():
    logits = torch.arange(2050, dtype=torch.float32).reshape(1, -1)
    ids = torch.arange(2, 2050, dtype=torch.int32).flip(0).reshape(1, -1)
    assert torch.equal(attention_ref.check_exact_topk(logits, ids, 2049, 1), ids)
    below = ids.clone()
    below[0, -1] = 0
    with pytest.raises(AssertionError):
        attention_ref.check_exact_topk(logits, below, 2049, 1)
    duplicate = ids.clone()
    duplicate[0, -1] = duplicate[0, 0]
    with pytest.raises(AssertionError):
        attention_ref.check_exact_topk(logits, duplicate, 2049, 1)


def test_dense_selection_is_a_causal_prefix_not_stale_storage():
    old_ids = torch.full((1, 2048), 123456, dtype=torch.int32)
    selected = attention_ref.check_exact_topk(torch.empty(1, 32), old_ids, 31, 1)
    assert torch.equal(selected[0, :32], torch.arange(32, dtype=torch.int32))
    assert bool((selected[0, 32:] == -1).all())
    assert bool((old_ids == 123456).all())


def test_numerical_gate_rejects_three_bf16_ulps():
    expected = torch.ones(1, 4, dtype=torch.bfloat16)
    wrong = torch.full_like(expected, 1 + 3 / 128)
    with pytest.raises(AssertionError):
        moe_ref.floored_ulp(wrong, expected)
    with pytest.raises(AssertionError):
        attention_ref.assert_close(wrong, expected, 2, 1 / 16)


@pytest.mark.parametrize("argument,center", [(1.0, 1.0), (4.0, 0.5)])
def test_rsqrt_candidates_cover_both_spacings_at_exponent_boundary(argument, center):
    candidates = moe_ref._rsqrt_candidates(argument)
    ulp = center * 2.0**-23
    assert candidates == [
        center - 2 * ulp,
        center - 1.5 * ulp,
        center - ulp,
        center - 0.5 * ulp,
        center,
        center + ulp,
        center + 2 * ulp,
    ]


def test_router_gate_rejects_bf16_rounded_logits_and_wrong_normalization():
    x = torch.ones(1, 6144, dtype=torch.bfloat16)
    gamma = torch.ones(6144, dtype=torch.float32)
    router = torch.zeros(256, 6144, dtype=torch.bfloat16)
    router[:, 0] = 1
    router[:, 1] = 1 / 512
    # Every permitted rstd rounds this row to BF16 ones. Its dot is exactly
    # 1+1/512 in FP32, but rounds to 1 in BF16: the gamma64 gate must catch it.
    logits = torch.full((1, 256), 1 + 1 / 512, dtype=torch.float32)
    moe_ref.check_router_logits(logits, x, gamma, router)
    with pytest.raises(AssertionError):
        moe_ref.check_router_logits(logits.bfloat16().float(), x, gamma, router)
    wrong_norm = torch.full_like(x, 1 + 3 / 128)
    with pytest.raises(AssertionError):
        moe_ref.floored_ulp(wrong_norm, moe_ref.rmsnorm(x, gamma))
    with pytest.raises(AssertionError):
        moe_ref.check_router_logits(
            wrong_norm.double() @ router.double().T, x, gamma, router
        )


def test_router_gate_requires_one_candidate_for_the_whole_expert_row():
    # Test candidate selection separately from the norm generator: accepting
    # each expert from a different allowed candidate would incorrectly pass all
    # ones here, even though neither complete candidate produces that row.
    norms = torch.zeros(2, 6144, dtype=torch.bfloat16)
    norms[0, 0] = 1
    norms[1, 1] = 1
    router = torch.zeros(256, 6144, dtype=torch.bfloat16)
    router[::2, 0] = 1
    router[1::2, 1] = 1
    for candidate in norms:
        logits = candidate.double()[None, :] @ router.double().T
        moe_ref.check_router_candidates(logits, [norms], router)
    with pytest.raises(AssertionError):
        moe_ref.check_router_candidates(torch.ones(1, 256), [norms], router)


def test_approximate_sigmoid_accepts_only_analytically_ambiguous_routes():
    sigma = torch.full((1, 256), 0.5, dtype=torch.float64)
    choice = torch.full_like(sigma, 0.25)
    choice[0, :7] = 0.75
    choice[0, 7] = 0.5 + moe_ref.SIGMOID_ABS_ERR / 2
    choice[0, 8] = 0.5
    swapped = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 8]], dtype=torch.int32)
    probabilities = torch.full((1, 8), 2.5 / 8)
    assert (
        moe_ref.check_routing(swapped, probabilities, sigma, choice)[
            "swapped_at_boundary"
        ]
        == 1
    )
    legal = moe_ref.candidate_selections(choice[0], moe_ref.SIGMOID_ABS_ERR + 1e-6)
    assert set(swapped[0].tolist()) in [set(candidate) for candidate in legal]
    wrong_ids = swapped.clone()
    wrong_ids[0, -1] = 9
    with pytest.raises(AssertionError):
        moe_ref.check_routing(wrong_ids, probabilities, sigma, choice)
    wrong_probs = probabilities.clone()
    wrong_probs[0, 0] += 0.01
    wrong_probs[0, 1] -= 0.01
    with pytest.raises(AssertionError):
        moe_ref.check_routing(swapped, wrong_probs, sigma, choice)


def test_numerical_gates_reject_nonfinite_values():
    expected = torch.ones(1, 4, dtype=torch.bfloat16)
    for value in (float("nan"), float("inf")):
        wrong = torch.full_like(expected, value)
        with pytest.raises(AssertionError):
            moe_ref.floored_ulp(wrong, expected)
        with pytest.raises(AssertionError):
            attention_ref.assert_close(wrong, expected, 2, 1 / 16)


def test_exact_topk_allows_equal_threshold_membership():
    logits = torch.zeros(1, 2050, dtype=torch.float32)
    logits[0, 0] = 1
    ids = torch.arange(2, 2050, dtype=torch.int32).reshape(1, -1)
    ids[0, -1] = 0
    assert torch.equal(attention_ref.check_exact_topk(logits, ids, 2049, 1), ids)
    ids[0, -1] = 1
    with pytest.raises(AssertionError):
        attention_ref.check_exact_topk(logits, ids, 2049, 1)


def test_moe_gate_up_packing_preserves_codes_and_scale_headers():
    # One bank exercises the same private layout without allocating the public
    # 257-bank fixture. Selected coordinates cross row-group, K-page and MMA tiles.
    gate = torch.zeros(1, 256, 6144, dtype=torch.uint8)
    up = torch.zeros_like(gate)
    coordinates = [
        (0, 0),
        (7, 7),
        (8, 8),
        (15, 15),
        (16, 1023),
        (127, 1024),
        (128, 2048),
        (255, 6143),
    ]
    for i, (row, col) in enumerate(coordinates):
        gate[0, row, col] = 0x30 + i
        up[0, row, col] = 0xB0 + i
    gs = torch.arange(96, dtype=torch.float32).reshape(1, 2, 48) / 128 + 0.25
    us = gs + 1
    bias = torch.zeros(256, dtype=torch.float32)
    _, packed = _moe_packing.expert_select_convert_to_mma(
        bias,
        gate.view(torch.float8_e4m3fn),
        gs,
        up.view(torch.float8_e4m3fn),
        us,
    )
    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (1, 16, 6, 32896)
    for i, (row, col) in enumerate(coordinates):
        group, within_row = divmod(row, 16)
        page, within_page = divmod(col, 1024)
        tile, within_col = divmod(within_page, 16)
        lane = (within_row % 8) * 4 + (within_col % 8) // 2
        fragment = (within_col // 8) * 2 + within_row // 8
        offset = tile * 256 + lane * 8 + fragment * 2 + within_col % 2
        assert packed[0, group, page, offset].item() == 0x30 + i
        assert packed[0, group, page, 16384 + offset].item() == 0xB0 + i
    assert int(torch.count_nonzero(packed[..., :32768])) == 2 * len(coordinates)
    for group in (0, 7, 8, 15):
        for page in (0, 2, 5):
            header = (
                packed[0, group, page, 32768:32832].contiguous().view(torch.float32)
            )
            assert torch.equal(header[:8], gs[0, group // 8, page * 8 : (page + 1) * 8])
            assert torch.equal(header[8:], us[0, group // 8, page * 8 : (page + 1) * 8])
    assert bool((packed[..., 32832:] == 0).all())


def test_moe_down_scale_replication_and_padding():
    data = torch.full((1, 6144, 256), 0x38, dtype=torch.uint8)
    scales = torch.arange(96, dtype=torch.float32).reshape(1, 48, 2) + 1
    packed, packed_scales = _moe_packing.expert_down_convert_to_bf16mma(
        data.view(torch.float8_e4m3fn), scales
    )
    assert packed.dtype == torch.uint8 and packed.shape == data.shape
    assert bool((packed == 0x38).all())
    scale_rows = packed_scales.reshape(1, 128, 8, 2)
    for cta in (0, 1, 2, 63, 127):
        for pair in range(6):
            assert torch.equal(
                scale_rows[0, cta, pair], scales[0, (cta * 6 + pair) // 16]
            )
    assert bool((scale_rows[:, :, 6:] == 0).all())


@pytest.mark.parametrize("value", [0, 2, 3, 8, True, 1.0])
def test_supported_query_lengths_reject_other_shapes(value):
    with pytest.raises(ValueError):
        geometry.check_q_len(value)


def test_workspace_regions_are_disjoint_for_all_operator_sets():
    sizes = {
        "attention": 16 << 20,
        "attention_q1": 0x38000,
        "ll": 128 << 10,
        "moe_q4": 0xE0000,
        "moe_q1": 0x38000,
    }
    for operators in (("moe",), ("attention",), ("attention", "moe")):
        total, offsets = geometry.communication_layout(operators)
        spans = sorted(
            (offset, offset + sizes[name]) for name, offset in offsets.items()
        )
        assert spans[0][0] == 0 and spans[-1][1] == total
        assert all(
            left[1] == right[0] for left, right in zip(spans, spans[1:], strict=False)
        )
        assert all(begin % 16 == 0 for begin, _ in spans)


def _mma_element_offset(row, column):
    # Scalar inverse of the documented m16n8 register convention.
    lane = (row % 8) * 4 + (column % 8) // 2
    fragment = (column // 8) * 2 + row // 8
    return lane * 8 + fragment * 2 + column % 2


def test_attention_lane_major_codes_headers_and_padding():
    weights = torch.zeros((256, 6144), dtype=torch.uint8)
    points = [
        (0, 0),
        (15, 15),
        (16, 16),
        (31, 127),
        (32, 128),
        (127, 1023),
        (128, 1024),
        (255, 6143),
    ]
    for index, (row, col) in enumerate(points):
        weights[row, col] = 0x31 + index
    scales = torch.arange(96, dtype=torch.float32).reshape(2, 48) + 0.25
    packed = _attention_packing.pack_lane_major(weights, scales).reshape(8, 6, 32896)
    for index, (row, col) in enumerate(points):
        cta, row32 = divmod(row, 32)
        page, column1024 = divmod(col, 1024)
        row_group, row16 = divmod(row32, 16)
        tile, col16 = divmod(column1024, 16)
        # K tiles are lane-major: eight interleaved tiles for each K warp.
        tile = (tile % 8) * 8 + tile // 8
        offset = row_group * 16384 + tile * 256 + _mma_element_offset(row16, col16)
        assert packed[cta, page, offset].item() == 0x31 + index
    assert int(torch.count_nonzero(packed[..., :32768])) == len(points)
    for cta in range(8):
        header = packed[cta, :, 32768:32800].contiguous().view(torch.float32)
        assert torch.equal(header, scales[cta // 4].reshape(6, 8))
    assert bool((packed[..., 32800:] == 0).all())


@pytest.mark.parametrize("kind", ["projq", "projo"])
def test_attention_per_head_projection_codes_and_scale_bytes(kind):
    rows, columns, row_group, code_bytes, page_bytes = (
        (512, 192, 64, 12288, 12416)
        if kind == "projq"
        else (256, 512, 32, 16384, 16512)
    )
    weights = torch.zeros((10, rows, columns), dtype=torch.uint8)
    points = [
        (0, 0, 0),
        (0, row_group - 1, columns - 1),
        (3, row_group, 16),
        (9, rows - 1, columns - 1),
    ]
    for index, (head, row, col) in enumerate(points):
        weights[head, row, col] = 0x39 + index
    scale_columns = columns // (64 if kind == "projq" else 128)
    scales = (
        torch.arange(10 * 4 * scale_columns, dtype=torch.float32).reshape(
            10, 4, scale_columns
        )
        / 128
        + 0.5
    )
    packed = getattr(_attention_packing, "pack_" + kind)(
        weights.view(torch.float8_e4m3fn), scales
    )
    assert packed.shape == (80, page_bytes)
    for index, (head, row, col) in enumerate(points):
        cta = head * 8 + row // row_group
        r = row % row_group
        offset = (
            (r // 16) * (16 * columns)
            + (col // 16) * 256
            + _mma_element_offset(r % 16, col % 16)
        )
        assert packed[cta, offset].item() == 0x39 + index
    assert int(torch.count_nonzero(packed[:, :code_bytes])) == len(points)
    header = (
        packed[:, code_bytes : code_bytes + 4 * scale_columns]
        .contiguous()
        .view(torch.float32)
    )
    assert torch.equal(
        header, scales.repeat_interleave(2, 1).reshape(80, scale_columns)
    )
    assert bool((packed[:, code_bytes + 4 * scale_columns :] == 0).all())


def test_attention_head_weight_packing_preserves_bf16_bits():
    bits = torch.zeros((32, 6144), dtype=torch.int16)
    points = [(0, 0), (7, 7), (8, 8), (15, 1023), (16, 1024), (31, 6143)]
    for index, (row, col) in enumerate(points):
        bits[row, col] = 0x3F80 + index
    packed = _attention_packing.pack_wis(bits.view(torch.bfloat16)).view(torch.int16)
    assert packed.shape == (2, 98304)
    for index, (row, col) in enumerate(points):
        group, row16 = divmod(row, 16)
        page, col1024 = divmod(col, 1024)
        tile, col16 = divmod(col1024, 16)
        offset = page * 16384 + tile * 256 + _mma_element_offset(row16, col16)
        assert packed[group, offset].item() == 0x3F80 + index
    assert int(torch.count_nonzero(packed)) == len(points)


def test_public_moe_packing_rejects_wrong_gamma_without_large_allocation():
    from flashinfer.glm5_tp8_decode import pack_glm5_tp8_moe_weights

    unused = [torch.empty(0)] * 8
    with pytest.raises(ValueError, match="gamma.*6144|shape"):
        pack_glm5_tp8_moe_weights(torch.empty(6143), *unused)
    with pytest.raises(TypeError, match="gamma.*dtype|float32"):
        pack_glm5_tp8_moe_weights(torch.empty(6144, dtype=torch.bfloat16), *unused)


def test_public_attention_packing_rejects_incomplete_or_wrong_rank_inputs():
    from flashinfer.glm5_tp8_decode import pack_glm5_tp8_attention_weights

    with pytest.raises(ValueError, match="missing|keys"):
        pack_glm5_tp8_attention_weights({}, rank=0)
    with pytest.raises(ValueError, match="rank"):
        pack_glm5_tp8_attention_weights({}, rank=8)


def test_packed_attention_validation_rejects_short_gamma_before_large_weights():
    from flashinfer.experimental.glm5_tp8_decode.attention_weights import (
        AttentionWeights,
        validate_packed,
    )

    # Supply the complete key set but only the two small vectors preceding
    # projection blobs. The short gamma must fail before blobs are inspected.
    weights = AttentionWeights(
        rank=1,
        tensors={
            "x_gamma": torch.empty(6144, dtype=torch.float32),
            "q_gamma": torch.empty(2047, dtype=torch.float32),
            "kv_gamma": None,
            "qkva": None,
            "wqb": None,
            "projq": None,
            "projo": None,
            "unproj": None,
        },
    )
    with pytest.raises(ValueError, match="q_gamma.*shape"):
        validate_packed(weights, rank=1, device=torch.device("cpu"))
