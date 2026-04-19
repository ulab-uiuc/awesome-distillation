import math
import pathlib
import sys

import torch
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from slime.rollout.on_policy_distillation import (  # noqa: E402
    _combine_requested_token_ids,
    _extract_optional_requested_token_log_probs,
    _extract_requested_token_log_probs,
    _extract_response_aux_id_logprob_maps,
    _extract_response_top_logprob_maps,
    _extract_teacher_topk_token_ids,
)
from slime.utils.opd_token_stats import (  # noqa: E402
    build_token_repetition_mask,
    compute_teacher_rank_at_k,
    compute_token_stats_metrics,
    compute_topk_overlap_ratio,
    extract_global_topk_token_ids,
)


def test_build_token_repetition_mask_token_3gram():
    token_ids = [1, 2, 3, 4, 2, 3, 4, 5]
    assert build_token_repetition_mask(token_ids, ngram=3) == [0, 0, 0, 0, 0, 0, 1, 0]


def test_rank_and_overlap_helpers():
    teacher_topk = [10, 20, 30, 40]
    student_topk = [30, 99, 10, 77]
    assert compute_teacher_rank_at_k(30, teacher_topk, k=4) == 3.0
    assert compute_teacher_rank_at_k(88, teacher_topk, k=4) == 5.0
    assert compute_topk_overlap_ratio(student_topk, teacher_topk, k=4) == 0.5


def test_extract_global_topk_token_ids_single_rank():
    logits = torch.tensor(
        [
            [0.1, 1.2, -0.3, 0.7],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    topk_ids = extract_global_topk_token_ids(logits, k=2, process_group=None, tp_rank=0)
    assert topk_ids.tolist() == [[1, 3], [0, 1]]


def test_compute_token_stats_metrics_grouped_means():
    metrics = compute_token_stats_metrics(
        repeat_mask=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        effective_mask=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        valid_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        student_log_probs=torch.tensor([-2.0, -4.0, -1.0, -3.0]),
        teacher_log_probs=torch.tensor([-1.0, -2.0, -1.5, -2.5]),
        teacher_rank_at_k=torch.tensor([51.0, 2.0, 10.0, 1.0]),
        topk_overlap=torch.tensor([0.2, 0.4, 0.6, 0.8]),
    )

    assert torch.isclose(metrics["repeat_ratio"], torch.tensor(0.5))
    assert torch.isclose(metrics["repeat_teacher_minus_student_logprob"], torch.tensor(2.0))
    assert torch.isclose(metrics["repeat_teacher_logprob"], torch.tensor(-2.0))
    assert torch.isclose(metrics["repeat_student_logprob"], torch.tensor(-4.0))
    assert torch.isclose(metrics["repeat_teacher_rank_at_k"], torch.tensor(2.0))
    assert torch.isclose(metrics["other_teacher_minus_student_logprob"], torch.tensor(0.25))
    assert torch.isclose(metrics["other_teacher_logprob"], torch.tensor(-1.25))
    assert torch.isclose(metrics["other_student_logprob"], torch.tensor(-1.5))
    assert torch.isclose(metrics["other_teacher_rank_at_k"], torch.tensor(30.5))
    assert torch.isclose(metrics["topk_overlap_ratio"], torch.tensor(0.4))
    assert "teacher_eos_prob" not in metrics
    assert "student_eos_prob" not in metrics


def test_extract_teacher_topk_and_requested_logprobs():
    teacher_output = {
        "meta_info": {
            "token_ids_logprob": [
                {"151645": -2.0},
                {"151645": -3.0},
            ],
            "input_token_logprobs": [
                [-0.1, 100, {"10": -0.1, "20": -0.4, "30": -0.8, "151645": -2.0}],
                [-0.2, 101, {"11": -0.2, "31": -0.5, "41": -0.9, "151645": -3.0}],
            ]
        }
    }
    response_items = teacher_output["meta_info"]["input_token_logprobs"]
    aux_maps = _extract_response_aux_id_logprob_maps(
        teacher_output,
        response_items,
        full_input_len=2,
        response_start=0,
    )
    topk_maps = _extract_response_top_logprob_maps(
        teacher_output,
        response_items,
        full_input_len=2,
        response_start=0,
    )

    teacher_topk_ids = _extract_teacher_topk_token_ids(topk_maps, topk=2)
    eos_logprobs = _extract_requested_token_log_probs(
        aux_maps,
        [[151645], [151645]],
        missing_error_prefix="teacher eos",
    )

    assert teacher_topk_ids == [[10, 20], [11, 31]]
    assert eos_logprobs == [[-2.0], [-3.0]]


def test_extract_optional_requested_token_log_probs_falls_back_and_allows_missing():
    aux_maps = [{}, {}]
    fallback_aux_maps = [
        {151645: -2.5, 10: -0.1},
        {11: -0.2, 31: -0.5},
    ]

    eos_logprobs = _extract_optional_requested_token_log_probs(
        aux_maps,
        [[151645], [151645]],
        fallback_aux_maps=fallback_aux_maps,
    )

    assert eos_logprobs[0] == [-2.5]
    assert math.isnan(eos_logprobs[1][0])



@pytest.mark.parametrize(
    ("sidecar_len", "expected_token_ids"),
    [
        (3, [101, 102]),
        (4, [102, 103]),
        (5, [103, 104]),
    ],
)
def test_extract_response_aux_id_logprob_maps_aligns_token_ids_sidecar_with_response_span(
    sidecar_len: int,
    expected_token_ids: list[int],
):
    teacher_output = {
        "meta_info": {
            "token_ids_logprob": [
                {str(100 + idx): -(idx + 0.1)}
                for idx in range(sidecar_len)
            ]
        }
    }

    aux_maps = _extract_response_aux_id_logprob_maps(
        teacher_output,
        [None, None],
        full_input_len=4,
        response_start=2,
    )

    assert [next(iter(row.keys())) for row in aux_maps] == expected_token_ids


@pytest.mark.parametrize(
    ("sidecar_len", "expected_token_ids"),
    [
        (3, [201, 202]),
        (4, [202, 203]),
        (5, [203, 204]),
    ],
)
def test_extract_response_top_logprob_maps_aligns_sidecar_with_response_span(
    sidecar_len: int,
    expected_token_ids: list[int],
):
    teacher_output = {
        "meta_info": {
            "input_top_logprobs": [
                {str(200 + idx): -(idx + 0.1)}
                for idx in range(sidecar_len)
            ]
        }
    }

    topk_maps = _extract_response_top_logprob_maps(
        teacher_output,
        [None, None],
        full_input_len=4,
        response_start=2,
    )

    assert [next(iter(row.keys())) for row in topk_maps] == expected_token_ids


def test_combined_topk_and_eos_requests_stay_aligned_when_sidecar_has_boundary_shift():
    requested_topk = [[10, 20], [11, 21]]
    requested_eos = [[151645], [151645]]
    teacher_output = {
        "meta_info": {
            "token_ids_logprob": [
                {"900": -9.0},
                {"901": -8.0},
                {"902": -7.0},
                {"10": -0.1, "20": -0.4, "151645": -2.0},
                {"11": -0.2, "21": -0.5, "151645": -3.0},
            ],
            "input_top_logprobs": [
                {"700": -9.0, "701": -9.5},
                {"710": -8.0, "711": -8.5},
                {"720": -7.0, "721": -7.5},
                {"10": -0.1, "20": -0.4, "30": -0.8},
                {"11": -0.2, "21": -0.5, "31": -0.9},
            ],
        }
    }

    combined = _combine_requested_token_ids(requested_topk, requested_eos)
    assert combined == [[10, 20, 151645], [11, 21, 151645]]

    aux_maps = _extract_response_aux_id_logprob_maps(
        teacher_output,
        [None, None],
        full_input_len=4,
        response_start=2,
    )
    topk_maps = _extract_response_top_logprob_maps(
        teacher_output,
        [None, None],
        full_input_len=4,
        response_start=2,
    )

    teacher_topk_ids = _extract_teacher_topk_token_ids(topk_maps, topk=2)
    teacher_topk_logprobs = _extract_requested_token_log_probs(
        aux_maps,
        requested_topk,
        missing_error_prefix="teacher topk",
    )
    teacher_eos_logprobs = _extract_optional_requested_token_log_probs(
        aux_maps,
        requested_eos,
        fallback_aux_maps=topk_maps,
    )

    assert teacher_topk_ids == [[10, 20], [11, 21]]
    assert teacher_topk_logprobs == [[-0.1, -0.4], [-0.2, -0.5]]
    assert [row[0] for row in teacher_eos_logprobs] == [-2.0, -3.0]
