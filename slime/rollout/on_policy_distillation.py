import json
import logging
import asyncio
from functools import lru_cache

import aiohttp
import torch

from slime.rollout.rm_hub import grade_answer_verl
from slime.utils.opd_token_stats import compute_teacher_rank_at_k, compute_topk_overlap_ratio
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
_RM_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}
_OPTIONAL_LOGPROB_WARNING_COUNTS: dict[str, int] = {}


def _get_rm_semaphore(args) -> asyncio.Semaphore:
    limit = int(getattr(args, "rm_max_concurrency", 8) or 8)
    if limit <= 0:
        limit = 1
    loop = asyncio.get_running_loop()
    key = (id(loop), limit)
    semaphore = _RM_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        _RM_SEMAPHORES[key] = semaphore
    return semaphore

_ANSWER_ONLY_PROMPT = (
    "The correct final answer to this problem is: {answer}\n"
    "Now solve the problem yourself step by step and arrive at the same answer:"
)


def _first_nonempty_text(*values) -> str:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            return s
    return ""


def _extract_user_content_from_prompt(prompt) -> str:
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return str(prompt or "")


def _infer_answer_format(metadata: dict) -> str:
    fmt = str((metadata or {}).get("format_instruction", "") or "")
    if "boxed" in fmt:
        return "boxed"
    if "Answer" in fmt:
        return "answer"
    return "auto"


def _resolve_chat_template_kwargs(args) -> dict:
    raw_kwargs = getattr(args, "apply_chat_template_kwargs", None)
    if not raw_kwargs:
        return {}
    if isinstance(raw_kwargs, dict):
        return raw_kwargs
    try:
        parsed = json.loads(raw_kwargs)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


@lru_cache(maxsize=4)
def _load_cached_tokenizer(name_or_path: str):
    return load_tokenizer(name_or_path, trust_remote_code=True)


def _get_teacher_tokenizer(args):
    tokenizer_source = getattr(args, "opd_teacher_tokenizer", None) or getattr(args, "hf_checkpoint", None)
    if not tokenizer_source:
        raise ValueError(
            "Cannot build privileged teacher prompt for OPD-SGLang because tokenizer source is empty. "
            "Set --opd-teacher-tokenizer or --hf-checkpoint."
        )
    return _load_cached_tokenizer(tokenizer_source)


def _build_teacher_input_ids(args, sample: Sample) -> list[int]:
    mode = getattr(args, "opd_teacher_info_mode", "same_as_student")
    if mode == "same_as_student":
        return list(sample.tokens)

    if mode != "answer_only":
        raise ValueError(f"Unsupported opd_teacher_info_mode: {mode!r}")

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw_content = _first_nonempty_text(metadata.get("raw_content"))
    student_user_content = _first_nonempty_text(
        metadata.get("student_user_content"),
        raw_content,
        _extract_user_content_from_prompt(sample.prompt),
    )
    answer_hint = _first_nonempty_text(
        sample.label,
        metadata.get("solution"),
        metadata.get("reference_solution"),
    )
    teacher_user_content = (
        f"{student_user_content}\n\n" + _ANSWER_ONLY_PROMPT.format(answer=answer_hint)
        if answer_hint
        else student_user_content
    )

    tokenizer = _get_teacher_tokenizer(args)
    template_kwargs = _resolve_chat_template_kwargs(args)
    teacher_enable_thinking = template_kwargs.get("enable_thinking", True)
    teacher_messages = [{"role": "user", "content": teacher_user_content}]
    try:
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=teacher_enable_thinking,
        )
    except TypeError as e:
        if "enable_thinking" not in str(e):
            raise
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    teacher_prompt_tokens = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
    response_tokens = list(sample.tokens[-sample.response_length:]) if sample.response_length > 0 else []
    return teacher_prompt_tokens + response_tokens


def _extract_scalar_logprob(item) -> float:
    if item is None:
        raise ValueError("Logprob item is None.")
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, str):
        return float(item)
    if isinstance(item, (list, tuple)):
        if not item:
            raise ValueError("Empty logprob item.")
        return _extract_scalar_logprob(item[0])
    if isinstance(item, dict):
        if "logprob" in item:
            return _extract_scalar_logprob(item["logprob"])
        if "value" in item:
            return _extract_scalar_logprob(item["value"])
    raise ValueError(f"Unsupported logprob item format: {type(item)}")


def _is_missing_logprob_item(item) -> bool:
    if item is None:
        return True
    if isinstance(item, (list, tuple)):
        if not item:
            return False
        return _is_missing_logprob_item(item[0])
    if isinstance(item, dict):
        if "logprob" in item:
            return _is_missing_logprob_item(item.get("logprob"))
        if "value" in item:
            return _is_missing_logprob_item(item.get("value"))
    return False


def _slice_response_items(
    items: list,
    full_input_len: int,
    response_start: int,
    response_len: int,
) -> list:
    if response_len <= 0:
        return []
    if len(items) == response_len:
        return list(items)
    if len(items) == full_input_len:
        begin = response_start
        end = response_start + response_len
        if end <= len(items):
            return list(items[begin:end])
    elif len(items) == full_input_len - 1:
        begin = max(response_start - 1, 0)
        end = begin + response_len
        if end <= len(items):
            return list(items[begin:end])
    elif len(items) == full_input_len + 1:
        begin = response_start + 1
        end = begin + response_len
        if end <= len(items):
            return list(items[begin:end])
    # Conservative fallback: keep old behavior (tail slice) for robustness.
    return list(items[-response_len:])


def _candidate_response_slices(
    items: list,
    full_input_len: int,
    response_start: int,
    response_len: int,
) -> list[list]:
    if response_len <= 0:
        return [[]]

    candidates: list[list] = []

    def _add(candidate: list | None) -> None:
        if candidate is None:
            return
        candidate = list(candidate)
        if len(candidate) != response_len:
            return
        if candidate not in candidates:
            candidates.append(candidate)

    _add(_slice_response_items(items, full_input_len, response_start, response_len))

    if len(items) >= response_len:
        for shift in (-1, 1):
            begin = response_start + shift
            end = begin + response_len
            if begin >= 0 and end <= len(items):
                _add(items[begin:end])
        _add(items[-response_len:])

    return candidates


def _extract_response_log_probs_with_mask(
    teacher_output: dict,
    *,
    full_input_len: int,
    response_start: int,
    response_len: int,
) -> tuple[list[float], list[int]]:
    if response_len <= 0:
        return [], []

    meta_info = teacher_output.get("meta_info", {}) if isinstance(teacher_output, dict) else {}
    candidate_specs: list[tuple[str, list[list]]] = []

    input_items = meta_info.get("input_token_logprobs")
    if isinstance(input_items, list):
        candidate_specs.append(
            (
                "meta_info.input_token_logprobs",
                _candidate_response_slices(input_items, full_input_len, response_start, response_len),
            )
        )

    output_items = meta_info.get("output_token_logprobs")
    if isinstance(output_items, list):
        candidate_specs.append(
            (
                "meta_info.output_token_logprobs",
                _candidate_response_slices(
                    output_items,
                    full_input_len=len(output_items),
                    response_start=max(len(output_items) - response_len, 0),
                    response_len=response_len,
                ),
            )
        )

    errors: list[str] = []
    for source_name, candidates in candidate_specs:
        for idx, candidate in enumerate(candidates):
            values: list[float] = []
            mask: list[int] = []
            candidate_errors: list[str] = []
            missing_positions: list[int] = []
            for pos, item in enumerate(candidate):
                try:
                    values.append(_extract_scalar_logprob(item))
                    mask.append(1)
                except Exception as exc:
                    if _is_missing_logprob_item(item):
                        values.append(0.0)
                        mask.append(0)
                        missing_positions.append(pos)
                        candidate_errors.append(f"pos={pos}: {exc}")
                    else:
                        candidate_errors.append(f"pos={pos}: {exc}")
                        values = []
                        mask = []
                        break
            valid_positions = sum(mask)
            if values and valid_positions > 0:
                missing = len(missing_positions)
                if missing > 0:
                    preview = missing_positions[:8]
                    logger.warning(
                        "Teacher response logprobs contain %s missing positions; they will be masked out. "
                        "source=%s candidate=%s missing_positions=%s response_len=%s response_start=%s "
                        "full_input_len=%s",
                        missing,
                        source_name,
                        idx,
                        preview,
                        response_len,
                        response_start,
                        full_input_len,
                    )
                return values, mask
            if candidate_errors:
                if values and valid_positions == 0:
                    candidate_errors.append("all positions are missing")
                errors.append(f"{source_name}[candidate={idx}]: {'; '.join(candidate_errors[:4])}")

    available_keys = sorted(meta_info.keys()) if isinstance(meta_info, dict) else []
    raise ValueError(
        "Failed to extract teacher response logprobs. "
        f"response_len={response_len}, response_start={response_start}, "
        f"full_input_len={full_input_len}, available_meta_info_keys={available_keys}, "
        f"errors={errors[:4]}"
    )


def _extract_id_logprob_map(raw) -> dict[int, float]:
    mapping: dict[int, float] = {}
    if raw is None:
        return mapping
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                tid = int(k)
            except Exception:
                continue
            try:
                mapping[tid] = _extract_scalar_logprob(v)
            except Exception:
                continue
        return mapping
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                token_id = item.get("token_id", item.get("id", item.get("token")))
                if token_id is None:
                    continue
                try:
                    mapping[int(token_id)] = _extract_scalar_logprob(item)
                except Exception:
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                a, b = item[0], item[1]
                try:
                    tid = int(b)
                    lp = _extract_scalar_logprob(a)
                except Exception:
                    try:
                        tid = int(a)
                        lp = _extract_scalar_logprob(b)
                    except Exception:
                        continue
                mapping[tid] = lp
    return mapping


def _extract_response_aux_id_logprob_maps(
    teacher_output: dict,
    response_items: list,
    *,
    full_input_len: int,
    response_start: int,
) -> list[dict[int, float]]:
    meta_info = teacher_output.get("meta_info", {}) if isinstance(teacher_output, dict) else {}
    sidecar = None
    for key in [
        "input_token_ids_logprobs",
        "token_ids_logprob",
        "input_top_logprobs",
        "input_token_top_logprobs",
    ]:
        if key in meta_info:
            sidecar = meta_info[key]
            break

    if sidecar is not None:
        # Reuse the same response alignment as input_token_logprobs rather than
        # inferring offsets from the sidecar length alone.
        response_sidecar = _slice_response_items(
            sidecar,
            full_input_len=full_input_len,
            response_start=response_start,
            response_len=len(response_items),
        )
    else:
        response_sidecar = [None] * len(response_items)

    aux_maps: list[dict[int, float]] = []
    for pos, item in enumerate(response_items):
        aux_map = {}
        if pos < len(response_sidecar):
            aux_map = _extract_id_logprob_map(response_sidecar[pos])
        if not aux_map and isinstance(item, dict):
            for key in ["token_ids_logprob", "top_logprobs", "top_logprob", "logprobs"]:
                if key in item:
                    aux_map = _extract_id_logprob_map(item[key])
                    if aux_map:
                        break
        if not aux_map and isinstance(item, (list, tuple)) and len(item) >= 3:
            aux_map = _extract_id_logprob_map(item[2])
        aux_maps.append(aux_map)
    return aux_maps


def _extract_response_top_logprob_maps(
    teacher_output: dict,
    response_items: list,
    *,
    full_input_len: int,
    response_start: int,
) -> list[dict[int, float]]:
    meta_info = teacher_output.get("meta_info", {}) if isinstance(teacher_output, dict) else {}
    sidecar = None
    for key in ["input_top_logprobs", "input_token_top_logprobs"]:
        if key in meta_info:
            sidecar = meta_info[key]
            break

    if sidecar is not None:
        response_sidecar = _slice_response_items(
            sidecar,
            full_input_len=full_input_len,
            response_start=response_start,
            response_len=len(response_items),
        )
    else:
        response_sidecar = [None] * len(response_items)

    aux_maps: list[dict[int, float]] = []
    for pos, item in enumerate(response_items):
        aux_map = {}
        if pos < len(response_sidecar):
            aux_map = _extract_id_logprob_map(response_sidecar[pos])
        if not aux_map and isinstance(item, dict):
            for key in ["top_logprobs", "top_logprob", "logprobs"]:
                if key in item:
                    aux_map = _extract_id_logprob_map(item[key])
                    if aux_map:
                        break
        if not aux_map and isinstance(item, (list, tuple)) and len(item) >= 3:
            aux_map = _extract_id_logprob_map(item[2])
        aux_maps.append(aux_map)
    return aux_maps


def _extract_requested_token_log_probs(
    aux_maps: list[dict[int, float]],
    requested_token_ids: list[list[int]],
    *,
    missing_error_prefix: str,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for pos, tok_ids in enumerate(requested_token_ids):
        aux_map = aux_maps[pos] if pos < len(aux_maps) else {}
        if not aux_map:
            raise ValueError(f"{missing_error_prefix} at position {pos}: token-level auxiliary logprobs are missing.")
        row = []
        for tid in tok_ids:
            if int(tid) not in aux_map:
                raise ValueError(f"{missing_error_prefix} at position {pos}: token id {tid} is missing.")
            row.append(float(aux_map[int(tid)]))
        rows.append(row)
    return rows


def _extract_optional_requested_token_log_probs(
    aux_maps: list[dict[int, float]],
    requested_token_ids: list[list[int]],
    *,
    fallback_aux_maps: list[dict[int, float]] | None = None,
    missing_value: float = float("nan"),
    warning_prefix: str | None = None,
) -> list[list[float]]:
    rows: list[list[float]] = []
    fallback_positions: list[int] = []
    missing_positions: list[int] = []
    for pos, tok_ids in enumerate(requested_token_ids):
        aux_map = aux_maps[pos] if pos < len(aux_maps) else {}
        fallback_map = fallback_aux_maps[pos] if fallback_aux_maps is not None and pos < len(fallback_aux_maps) else {}
        row = []
        used_fallback = False
        has_missing = False
        for tid in tok_ids:
            tid = int(tid)
            if tid in aux_map:
                row.append(float(aux_map[tid]))
            elif tid in fallback_map:
                row.append(float(fallback_map[tid]))
                used_fallback = True
            else:
                row.append(float(missing_value))
                has_missing = True
        if used_fallback:
            fallback_positions.append(pos)
        if has_missing:
            missing_positions.append(pos)
        rows.append(row)

    if warning_prefix is not None and (fallback_positions or missing_positions):
        count = _OPTIONAL_LOGPROB_WARNING_COUNTS.get(warning_prefix, 0) + 1
        _OPTIONAL_LOGPROB_WARNING_COUNTS[warning_prefix] = count
        if count <= 3 or count % 50 == 0:
            suffix = ""
            if count > 3:
                suffix = f" (occurrence={count}; intermediate warnings suppressed)"
            logger.warning(
                "%s: exact_positions=%d/%d, fallback_positions=%s, missing_positions=%s%s",
                warning_prefix,
                len(requested_token_ids) - len(fallback_positions) - len(missing_positions),
                len(requested_token_ids),
                fallback_positions[:20],
                missing_positions[:20],
                suffix,
            )
    return rows


def _extract_teacher_topk_token_ids(
    aux_maps: list[dict[int, float]],
    topk: int,
) -> list[list[int]]:
    teacher_topk_ids: list[list[int]] = []
    for pos, aux_map in enumerate(aux_maps):
        if not aux_map:
            raise ValueError(f"Missing teacher top-k logprobs at position {pos}.")
        sorted_pairs = sorted(aux_map.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_pairs) < topk:
            raise ValueError(
                f"Teacher top-k at position {pos} has only {len(sorted_pairs)} entries, expected at least {topk}."
            )
        teacher_topk_ids.append([int(token_id) for token_id, _ in sorted_pairs[:topk]])
    return teacher_topk_ids


def _extract_teacher_topk_log_probs(
    aux_maps: list[dict[int, float]],
    requested_token_ids: list[list[int]],
) -> list[list[float]]:
    return _extract_requested_token_log_probs(
        aux_maps,
        requested_token_ids,
        missing_error_prefix="Missing teacher token-level top-k logprobs",
    )


def _combine_requested_token_ids(
    primary: list[list[int]] | None,
    extra: list[list[int]] | None,
) -> list[list[int]] | None:
    if primary is None and extra is None:
        return None
    if primary is None:
        return [list(row) for row in extra]
    if extra is None:
        return [list(row) for row in primary]
    if len(primary) != len(extra):
        raise ValueError(f"Requested token-id length mismatch: {len(primary)} vs {len(extra)}.")
    combined: list[list[int]] = []
    for row_primary, row_extra in zip(primary, extra, strict=False):
        merged = list(dict.fromkeys([int(x) for x in row_primary] + [int(x) for x in row_extra]))
        combined.append(merged)
    return combined


def _get_sample_student_topk_ids(sample: Sample) -> list[list[int]] | None:
    if getattr(sample, "opd_diag_student_topk_token_ids", None) is not None:
        return getattr(sample, "opd_diag_student_topk_token_ids")
    if getattr(sample, "opd_topk_token_ids", None) is not None:
        return getattr(sample, "opd_topk_token_ids")
    return None


async def reward_func(args, sample, **kwargs):
    teacher_input_ids = _build_teacher_input_ids(args, sample)
    response_len = int(sample.response_length or 0)
    response_start = max(len(teacher_input_ids) - response_len, 0)
    use_opd_sglang = getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"
    use_topk_kl = use_opd_sglang and getattr(args, "opd_kl_mode", "token_reverse_kl") == "full_vocab_topk_reverse_kl"
    diag_enabled = use_opd_sglang and bool(getattr(args, "opd_token_stats", False))
    diag_topk = int(getattr(args, "opd_token_stats_topk", 50))
    payload = {
        "input_ids": teacher_input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # For OPD we only need teacher logprobs on the generated response tokens.
        # We request from (response_start - 1) rather than response_start because SGLang
        # always prepends a None to input_token_logprobs_val for the boundary token at
        # logprob_start_len (it needs the hidden state one position *before* to compute
        # that token's logprob). By including one extra prompt token, the first response
        # token's logprob is available and the leading None lands on the last prompt token
        # instead, which we discard via the tail-slice in _slice_response_items.
        "logprob_start_len": max(response_start - 1, 0),
    }
    topk_request_size = 0
    requested_topk_token_ids = None
    if use_topk_kl:
        topk_token_ids = getattr(sample, "opd_topk_token_ids", None)
        topk_student_lp = getattr(sample, "opd_topk_student_log_probs", None)
        if topk_token_ids is None or topk_student_lp is None:
            raise ValueError(
                "full_vocab_topk_reverse_kl requires student top-k logprobs from rollout, but they are missing."
            )
        if len(topk_token_ids) != response_len or len(topk_student_lp) != response_len:
            raise ValueError(
                f"Student top-k data length mismatch: token_ids={len(topk_token_ids)}, "
                f"student_logprobs={len(topk_student_lp)}, response_len={response_len}."
            )
        requested_topk_token_ids = topk_token_ids
        topk_request_size = max(topk_request_size, int(getattr(args, "opd_topk", 50)))
    if diag_enabled:
        topk_request_size = max(topk_request_size, diag_topk)


    requested_token_ids = _combine_requested_token_ids(requested_topk_token_ids, None)
    if requested_token_ids is not None:
        payload["token_ids_logprob"] = requested_token_ids
    if topk_request_size > 0:
        payload["top_logprobs_num"] = topk_request_size

    session_kwargs = {}
    rm_semaphore = _get_rm_semaphore(args)
    async with rm_semaphore:
        async with aiohttp.ClientSession(**session_kwargs) as session:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                teacher_output = await resp.json()

    teacher_topk_log_probs = None
    teacher_topk_token_ids = None
    if use_topk_kl or diag_enabled:
        input_items = teacher_output.get("meta_info", {}).get("input_token_logprobs")
        if input_items is None:
            raise ValueError(
                "OPD token diagnostics require `meta_info.input_token_logprobs` from teacher response."
            )
        response_items = _slice_response_items(
            input_items,
            full_input_len=len(teacher_input_ids),
            response_start=response_start,
            response_len=response_len,
        )
        aux_maps = _extract_response_aux_id_logprob_maps(
            teacher_output=teacher_output,
            response_items=response_items,
            full_input_len=len(teacher_input_ids),
            response_start=response_start,
        )
        teacher_top_logprob_maps = None
        if use_topk_kl:
            teacher_topk_log_probs = _extract_teacher_topk_log_probs(aux_maps, requested_topk_token_ids)
        if diag_enabled:
            teacher_top_logprob_maps = _extract_response_top_logprob_maps(
                teacher_output=teacher_output,
                response_items=response_items,
                full_input_len=len(teacher_input_ids),
                response_start=response_start,
            )
            teacher_topk_token_ids = _extract_teacher_topk_token_ids(
                teacher_top_logprob_maps,
                diag_topk,
            )


    # Dual-metric grading for diagnosis:
    # - strict follows dataset format (boxed/answer) inferred from metadata
    # - relaxed uses auto extraction to reduce format-mismatch false negatives
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    strict_mode = _infer_answer_format(metadata)
    response = sample.response or ""
    label = sample.label or ""
    accuracy_strict = 1.0 if grade_answer_verl(response, label, mode=strict_mode) else 0.0
    # True relaxed metric: accept if any extraction mode gets the correct final answer.
    # This avoids `auto` under-counting when an incorrect "Answer:" line appears before
    # a correct final boxed answer.
    relaxed_hit = (
        grade_answer_verl(response, label, mode="boxed")
        or grade_answer_verl(response, label, mode="answer")
        or grade_answer_verl(response, label, mode="auto")
    )
    accuracy_relaxed = 1.0 if relaxed_hit else 0.0
    return {
        "teacher_output": teacher_output,
        "teacher_input_len": len(teacher_input_ids),
        "teacher_response_start": response_start,
        "teacher_topk_log_probs": teacher_topk_log_probs,
        "teacher_topk_token_ids": teacher_topk_token_ids,
        "accuracy_strict": accuracy_strict,
        "accuracy_relaxed": accuracy_relaxed,
        # Backward-compatible alias for existing scripts/checkpoints.
        "accuracy": accuracy_strict,
    }


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    reward_key = getattr(args, "reward_key", None) or "accuracy"
    raw_rewards = []
    response_lengths = [sample.response_length for sample in samples]
    use_opd_sglang = getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"
    opd_distill_max_response_len = int(getattr(args, "opd_distill_max_response_len", 2048))
    diag_enabled = use_opd_sglang and bool(getattr(args, "opd_token_stats", False))
    diag_topk = int(getattr(args, "opd_token_stats_topk", 50))

    if use_opd_sglang:
        for sample, response_length in zip(samples, response_lengths, strict=False):
            if opd_distill_max_response_len == -1:
                sample.opd_distill_sample_mask = 1
            else:
                sample.opd_distill_sample_mask = 1 if response_length <= opd_distill_max_response_len else 0

    teacher_outputs = []
    teacher_input_lens = []
    teacher_response_starts = []
    teacher_topk_logprobs_list = []
    teacher_topk_token_ids_list = []
    for sample in samples:
        reward = sample.reward
        if isinstance(reward, dict) and "teacher_output" in reward:
            teacher_output = reward["teacher_output"]
            raw_rewards.append(float(reward.get(reward_key, reward.get("accuracy", 0.0))))
            teacher_input_lens.append(int(reward.get("teacher_input_len", len(sample.tokens))))
            teacher_response_starts.append(
                int(reward.get("teacher_response_start", max(len(sample.tokens) - sample.response_length, 0)))
            )
            teacher_topk_logprobs_list.append(reward.get("teacher_topk_log_probs"))
            teacher_topk_token_ids_list.append(reward.get("teacher_topk_token_ids"))
        else:
            # Backward-compatible path for historical checkpoints/scripts.
            teacher_output = reward
            raw_rewards.append(0.0)
            teacher_input_lens.append(len(sample.tokens))
            teacher_response_starts.append(max(len(sample.tokens) - sample.response_length, 0))
            teacher_topk_logprobs_list.append(None)
            teacher_topk_token_ids_list.append(None)
        teacher_outputs.append(teacher_output)

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = []
    for sample, reward, response_length, input_len, response_start in zip(
        samples, teacher_outputs, response_lengths, teacher_input_lens, teacher_response_starts, strict=False
    ):
        extracted, extracted_mask = _extract_response_log_probs_with_mask(
            reward,
            full_input_len=input_len,
            response_start=response_start,
            response_len=response_length,
        )
        teacher_log_probs.append(torch.tensor(extracted, dtype=torch.float32))
        if any(x == 0 for x in extracted_mask):
            sample.teacher_logprob_mask = torch.tensor(extracted_mask, dtype=torch.int)
        else:
            sample.teacher_logprob_mask = torch.ones(response_length, dtype=torch.int)

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    use_topk_kl = (
        getattr(args, "use_opd", False)
        and getattr(args, "opd_type", None) == "sglang"
        and getattr(args, "opd_kl_mode", "token_reverse_kl") == "full_vocab_topk_reverse_kl"
    )
    if use_topk_kl:
        for idx, (sample, teacher_topk) in enumerate(
            zip(samples, teacher_topk_logprobs_list, strict=False)
        ):
            student_topk = getattr(sample, "opd_topk_student_log_probs", None)
            if student_topk is None or teacher_topk is None:
                raise ValueError(
                    f"Sample {idx}: missing top-k logprob data for full_vocab_topk_reverse_kl."
                )
            if len(student_topk) != sample.response_length or len(teacher_topk) != sample.response_length:
                raise ValueError(
                    f"Sample {idx}: top-k length mismatch. "
                    f"student={len(student_topk)}, teacher={len(teacher_topk)}, response={sample.response_length}."
                )
            sample.opd_topk_student_log_probs = torch.tensor(student_topk, dtype=torch.float32)
            sample.opd_topk_teacher_log_probs = torch.tensor(teacher_topk, dtype=torch.float32)

    if diag_enabled:
        for idx, (sample, teacher_topk_ids) in enumerate(
            zip(samples, teacher_topk_token_ids_list, strict=False)
        ):
            student_topk_ids = _get_sample_student_topk_ids(sample)
            if student_topk_ids is None or teacher_topk_ids is None:
                raise ValueError(f"Sample {idx}: missing student/teacher top-k ids for OPD token diagnostics.")
            if len(student_topk_ids) != sample.response_length or len(teacher_topk_ids) != sample.response_length:
                raise ValueError(
                    f"Sample {idx}: top-k length mismatch for token diagnostics. "
                    f"student={len(student_topk_ids)}, teacher={len(teacher_topk_ids)}, response={sample.response_length}."
                )
            response_tokens = list(sample.tokens[-sample.response_length:]) if sample.response_length > 0 else []
            sample.opd_diag_topk_overlap = torch.tensor(
                [
                    compute_topk_overlap_ratio(student_topk_ids[pos], teacher_topk_ids[pos], diag_topk)
                    for pos in range(sample.response_length)
                ],
                dtype=torch.float32,
            )
            sample.opd_diag_teacher_rank_at_k = torch.tensor(
                [
                    compute_teacher_rank_at_k(response_tokens[pos], teacher_topk_ids[pos], diag_topk)
                    for pos in range(sample.response_length)
                ],
                dtype=torch.float32,
            )

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # When --opd-zero-task-reward is enabled, task rewards are zeroed so training
    # is driven by OPD KL only. Otherwise use raw task rewards.
    if getattr(args, "opd_zero_task_reward", False):
        scalar_rewards = [0.0] * len(samples)
    else:
        scalar_rewards = list(raw_rewards)

    return raw_rewards, scalar_rewards
