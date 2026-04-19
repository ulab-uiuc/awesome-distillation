from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist


def build_token_repetition_mask(response_token_ids: Sequence[int] | torch.Tensor, ngram: int = 3) -> list[int]:
    if ngram < 2:
        raise ValueError(f"ngram must be >= 2, got {ngram}.")

    if isinstance(response_token_ids, torch.Tensor):
        token_ids = [int(x) for x in response_token_ids.detach().cpu().tolist()]
    else:
        token_ids = [int(x) for x in response_token_ids]

    repeat_mask = [0] * len(token_ids)
    seen_ngrams: set[tuple[int, ...]] = set()
    for end in range(ngram - 1, len(token_ids)):
        current_ngram = tuple(token_ids[end - ngram + 1 : end + 1])
        if current_ngram in seen_ngrams:
            repeat_mask[end] = 1
        else:
            seen_ngrams.add(current_ngram)
    return repeat_mask


def compute_teacher_rank_at_k(sampled_token_id: int, teacher_topk_ids: Sequence[int], k: int) -> float:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")
    sampled_token_id = int(sampled_token_id)
    for rank, token_id in enumerate(teacher_topk_ids[:k], start=1):
        if int(token_id) == sampled_token_id:
            return float(rank)
    return float(k + 1)


def compute_topk_overlap_ratio(student_topk_ids: Sequence[int], teacher_topk_ids: Sequence[int], k: int) -> float:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")
    student = {int(token_id) for token_id in student_topk_ids[:k]}
    teacher = {int(token_id) for token_id in teacher_topk_ids[:k]}
    return float(len(student & teacher) / float(k))


def compute_teacher_rank_at_k_tensor(
    sampled_token_ids: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")
    if sampled_token_ids.dim() != 1:
        raise ValueError(f"sampled_token_ids must be rank-1, got shape={tuple(sampled_token_ids.shape)}.")
    if teacher_topk_ids.dim() != 2:
        raise ValueError(f"teacher_topk_ids must be rank-2, got shape={tuple(teacher_topk_ids.shape)}.")
    if sampled_token_ids.size(0) != teacher_topk_ids.size(0):
        raise ValueError(
            "sampled_token_ids / teacher_topk_ids length mismatch: "
            f"{sampled_token_ids.size(0)} vs {teacher_topk_ids.size(0)}."
        )

    matches = teacher_topk_ids.eq(sampled_token_ids.unsqueeze(-1))
    has_match = matches.any(dim=-1)
    first_match = matches.float().argmax(dim=-1).to(dtype=torch.float32) + 1.0
    missing_rank = torch.full_like(first_match, float(k + 1))
    return torch.where(has_match, first_match, missing_rank)


def compute_topk_overlap_ratio_tensor(
    student_topk_ids: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")
    if student_topk_ids.dim() != 2 or teacher_topk_ids.dim() != 2:
        raise ValueError(
            "student_topk_ids and teacher_topk_ids must both be rank-2. "
            f"Got {tuple(student_topk_ids.shape)} and {tuple(teacher_topk_ids.shape)}."
        )
    if student_topk_ids.shape[0] != teacher_topk_ids.shape[0]:
        raise ValueError(
            "student_topk_ids / teacher_topk_ids length mismatch: "
            f"{student_topk_ids.shape[0]} vs {teacher_topk_ids.shape[0]}."
        )

    student_expanded = student_topk_ids.unsqueeze(-1)
    teacher_expanded = teacher_topk_ids.unsqueeze(-2)
    overlap = student_expanded.eq(teacher_expanded).any(dim=-1).float().sum(dim=-1)
    return overlap / float(k)


def extract_global_topk_token_ids(
    logits: torch.Tensor,
    k: int,
    *,
    process_group: dist.ProcessGroup | None = None,
    tp_rank: int = 0,
) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")
    if logits.dim() != 2:
        raise ValueError(f"logits must be rank-2 [seq_len, vocab_local], got shape={tuple(logits.shape)}.")
    if logits.size(0) == 0:
        return torch.empty((0, 0), dtype=torch.long, device=logits.device)

    local_vocab_size = logits.size(-1)
    local_k = max(1, min(k, local_vocab_size))
    local_values, local_indices = logits.topk(local_k, dim=-1, sorted=True)
    global_local_indices = local_indices + int(tp_rank) * local_vocab_size

    if process_group is not None:
        world_size = dist.get_world_size(process_group)
        gathered_values = [torch.zeros_like(local_values) for _ in range(world_size)]
        gathered_indices = [torch.zeros_like(global_local_indices) for _ in range(world_size)]
        dist.all_gather(gathered_values, local_values, group=process_group)
        dist.all_gather(gathered_indices, global_local_indices, group=process_group)
        all_values = torch.cat(gathered_values, dim=-1)
        all_indices = torch.cat(gathered_indices, dim=-1)
    else:
        all_values = local_values
        all_indices = global_local_indices

    actual_k = min(k, all_values.size(-1))
    top_positions = all_values.topk(actual_k, dim=-1, sorted=True).indices
    return all_indices.gather(-1, top_positions).long()


def _masked_mean_or_nan(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"values/mask shape mismatch: {tuple(values.shape)} vs {tuple(mask.shape)}.")
    mask = mask.to(device=values.device, dtype=values.dtype)
    finite_mask = torch.isfinite(values).to(dtype=values.dtype, device=values.device)
    safe_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    mask = mask * finite_mask
    denom = mask.sum()
    if denom.item() <= 0:
        return torch.full((), float("nan"), dtype=values.dtype, device=values.device)
    return (safe_values * mask).sum() / denom


def compute_token_stats_metrics(
    *,
    repeat_mask: torch.Tensor,
    effective_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_rank_at_k: torch.Tensor,
    topk_overlap: torch.Tensor,
) -> dict[str, torch.Tensor]:
    tensors = [
        repeat_mask,
        effective_mask,
        valid_mask,
        student_log_probs,
        teacher_log_probs,
        teacher_rank_at_k,
        topk_overlap,
    ]
    flat_tensors = [tensor.reshape(-1).to(dtype=torch.float32) for tensor in tensors]
    (
        repeat_mask,
        effective_mask,
        valid_mask,
        student_log_probs,
        teacher_log_probs,
        teacher_rank_at_k,
        topk_overlap,
    ) = flat_tensors

    repeat_ratio = _masked_mean_or_nan(repeat_mask, effective_mask)
    repeat_group_mask = repeat_mask * valid_mask
    other_group_mask = (1.0 - repeat_mask) * valid_mask
    teacher_minus_student = teacher_log_probs - student_log_probs

    return {
        "repeat_ratio": repeat_ratio,
        "repeat_teacher_minus_student_logprob": _masked_mean_or_nan(teacher_minus_student, repeat_group_mask),
        "repeat_teacher_logprob": _masked_mean_or_nan(teacher_log_probs, repeat_group_mask),
        "repeat_student_logprob": _masked_mean_or_nan(student_log_probs, repeat_group_mask),
        "repeat_teacher_rank_at_k": _masked_mean_or_nan(teacher_rank_at_k, repeat_group_mask),
        "other_teacher_minus_student_logprob": _masked_mean_or_nan(teacher_minus_student, other_group_mask),
        "other_teacher_logprob": _masked_mean_or_nan(teacher_log_probs, other_group_mask),
        "other_student_logprob": _masked_mean_or_nan(student_log_probs, other_group_mask),
        "other_teacher_rank_at_k": _masked_mean_or_nan(teacher_rank_at_k, other_group_mask),
        "topk_overlap_ratio": _masked_mean_or_nan(topk_overlap, valid_mask),
    }
