"""Collect LLM-generated SFT conversations for OpenThoughts math prompts.

Supports Qwen2.5-Math-7B-Instruct (default) and other OpenAI-compatible models.

Prompt pre-filtering mirrors ``filter_openthoughts_math_sft.py``:

  1. keep only rows with ``domain == "math"``;
  2. read the reference solution from ``ground_truth_solution`` with
     ``deepseek_solution`` as fallback;
  3. keep only rows with exactly one extractable reference ``\\boxed{...}``
     answer;
  4. skip rows with an empty problem.

Generated responses are validated with ``is_valid_output``: they must contain
``\\boxed{}`` and must not exhibit repeated-line, n-gram, or consecutive-block
repetition patterns.

Supported output suffixes are ``.parquet`` and ``.jsonl``.
"""



"""python3 examples/on_policy_distillation/collect_qwen3_llm_sft_openthoughts_math.py \
  --output /root/math/data/sft_qwen25math_7b_generated_openthoughts_math.parquet \
  --api-base http://localhost:30006/v1 \
  --api-key EMPTY \
  --model Qwen2.5-Math-7B-Instruct \
  --max-tokens 8192 \
  --temperature 0.7 \
  --top-p 0.95 \
  --concurrency 16 \
  --max-samples 10

  
标准模式（行为与 Qwen2.5 一致）：                                                                              
python3 examples/on_policy_distillation/collect_qwen3_llm_sft_openthoughts_math.py \
    --output /root/math/data/sft_qwen3_4b_math.parquet \
    --api-base http://localhost:30006/v1 \
    --api-key EMPTY \
    --model Qwen3-4B-Thinking-2507 \
    --concurrency 512 \
    --max-samples 1 \
    --enable-thinking False
                                                                                                                 
  开启 thinking，保留 <think> 标签（SFT 训练推理能力）：                                                         
    --enable-thinking True                                                                                       
                                                                                                                 
  开启 thinking，但去掉 <think> 标签（只保留最终答案）：                                                         
    --enable-thinking True --strip-thinking True
"""
import argparse
import asyncio
import json
import logging
import os
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx

try:
    from examples.on_policy_distillation.filter_openthoughts_math import (
        _ANSWER_FORMAT_INSTRUCTION,
        _BOXED_FORMAT_INSTRUCTION,
        _FORMAT_SUFFIX,
        _extract_boxed_answers,
        _normalize,
    )
except ImportError:
    # Allows direct execution as:
    #   python examples/on_policy_distillation/collect_qwen3_llm_sft_openthoughts_math.py
    from filter_openthoughts_math import (  # type: ignore
        _ANSWER_FORMAT_INSTRUCTION,
        _BOXED_FORMAT_INSTRUCTION,
        _FORMAT_SUFFIX,
        _extract_boxed_answers,
        _normalize,
    )


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_QWEN25MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Qwen3 does not require a special math system prompt; use empty by default.
_QWEN3_SYSTEM_PROMPT = ""


# ---------------------------------------------------------------------------
# Output quality filters
# ---------------------------------------------------------------------------

def has_boxed(text: str) -> bool:
    """Check if output contains \\boxed{}."""
    return "\\boxed" in text


def detect_repeated_lines(text: str, min_len: int = 20, threshold: int = 5) -> bool:
    """Detect if any line (len >= min_len) appears >= threshold times."""
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) >= min_len]
    if not lines:
        return False
    counter = Counter(lines)
    return counter.most_common(1)[0][1] >= threshold


def detect_ngram_repetition(text: str, n: int = 100, threshold: int = 3) -> bool:
    """Detect if any n-char substring appears >= threshold times (sliding window)."""
    if len(text) < n * threshold:
        return False
    seen: dict[str, int] = {}
    for i in range(0, len(text) - n + 1, 10):
        chunk = text[i : i + n]
        seen[chunk] = seen.get(chunk, 0) + 1
        if seen[chunk] >= threshold:
            return True
    return False


def detect_consecutive_repeat(text: str, block_size: int = 50, threshold: int = 3) -> bool:
    """Detect if a consecutive block (>= block_size chars) repeats >= threshold times in a row."""
    if len(text) < block_size * threshold:
        return False
    for i in range(len(text) - block_size * threshold + 1):
        block = text[i : i + block_size]
        count = 1
        pos = i + block_size
        while pos + block_size <= len(text) and text[pos : pos + block_size] == block:
            count += 1
            pos += block_size
            if count >= threshold:
                return True
    return False


def strip_thinking(text: str) -> str:
    """Remove leading <think>...</think> block produced by Qwen3 thinking mode."""
    import re
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def is_valid_output(text: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Reject outputs missing \\boxed{} or with repetitive text."""
    if not has_boxed(text):
        return False, "no_boxed"
    if detect_repeated_lines(text):
        return False, "repeated_lines"
    if detect_ngram_repetition(text):
        return False, "ngram_repetition"
    if len(text) > 5000 and detect_consecutive_repeat(text):
        return False, "consecutive_repeat"
    return True, "ok"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    source_index: int
    problem_raw: str
    student_user_content: str
    label: str
    reference_solution: str
    source: str
    format_instruction: str


@dataclass(frozen=True)
class GenerationResult:
    text: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _fetch_server_info(api_base: str, api_key: str) -> tuple[int | None, str | None]:
    """Query /v1/models for (max_model_len, model_root_path). Returns (None, None) on failure."""
    import urllib.request
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    url = f"{base}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        for entry in data.get("data", []):
            max_len = entry.get("max_model_len")
            root = entry.get("root") or entry.get("id")
            if max_len:
                return int(max_len), root
    except Exception:
        pass
    return None, None


def _filter_by_length(
    candidates: list[Candidate],
    *,
    model_path: str,
    system_prompt: str,
    max_tokens: int,
    max_model_len: int,
) -> tuple[list[Candidate], int]:
    """Remove candidates whose tokenized input + max_tokens would exceed max_model_len."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("transformers not installed; skipping length pre-filter.")
        return candidates, 0

    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from {model_path!r}: {e}; skipping length pre-filter.")
        return candidates, 0

    budget = max_model_len - max_tokens
    if budget <= 0:
        raise ValueError(
            f"max_tokens={max_tokens} >= max_model_len={max_model_len}; "
            "lower --max-tokens so there is room for the prompt."
        )

    kept: list[Candidate] = []
    skipped = 0
    for c in candidates:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": c.student_user_content})
        try:
            ids = tok.apply_chat_template(messages, add_generation_prompt=True)
            n_tokens = len(ids)
        except Exception:
            # Fallback: rough char-based estimate (4 chars ≈ 1 token)
            n_tokens = sum(len(m["content"]) for m in messages) // 4
        if n_tokens <= budget:
            kept.append(c)
        else:
            skipped += 1
    return kept, skipped


def _write_records(output: str, records: list[dict]) -> None:
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as fout:
            for record in records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    if output_path.suffix == ".parquet":
        try:
            from datasets import Dataset
        except ImportError:
            logger.error("datasets package is required to write parquet: pip install datasets")
            sys.exit(1)
        Dataset.from_list(records).to_parquet(str(output_path))
        return

    raise ValueError(f"Unsupported output suffix {output_path.suffix!r}; use .parquet or .jsonl.")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_candidates(answer_format: str, max_samples: int | None) -> tuple[list[Candidate], Counter, dict[str, int]]:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package is required: pip install datasets")
        sys.exit(1)

    if answer_format not in _FORMAT_SUFFIX:
        raise ValueError(f"--answer-format must be 'answer' or 'boxed', got: {answer_format!r}")

    fmt_template = _ANSWER_FORMAT_INSTRUCTION if answer_format == "answer" else _BOXED_FORMAT_INSTRUCTION
    fmt_suffix = _FORMAT_SUFFIX[answer_format]

    logger.info("Loading open-thoughts/OpenThoughts-114k (config=metadata, split=train)...")
    ds = load_dataset("open-thoughts/OpenThoughts-114k", name="metadata", split="train")
    logger.info(f"Total samples: {len(ds)}")

    domain_counts: Counter = Counter()
    stats = {
        "scanned": 0,
        "skipped_domain": 0,
        "skipped_no_reference_boxed": 0,
        "skipped_multi_reference_boxed": 0,
        "skipped_empty_problem": 0,
    }
    candidates: list[Candidate] = []

    for source_index, row in enumerate(ds):
        if max_samples is not None and len(candidates) >= max_samples:
            break

        stats["scanned"] += 1
        domain = _normalize(row.get("domain", ""))
        domain_counts[domain] += 1

        if domain != "math":
            stats["skipped_domain"] += 1
            continue

        reference_solution = _normalize(row.get("ground_truth_solution") or row.get("deepseek_solution") or "")
        reference_boxed_answers = _extract_boxed_answers(reference_solution)
        if len(reference_boxed_answers) == 0:
            stats["skipped_no_reference_boxed"] += 1
            continue
        if len(reference_boxed_answers) > 1:
            stats["skipped_multi_reference_boxed"] += 1
            continue

        problem_raw = _normalize(row.get("problem", ""))
        if not problem_raw:
            stats["skipped_empty_problem"] += 1
            continue

        student_user_content = fmt_template.format(problem=problem_raw)
        source = _normalize(row.get("source") or row.get("domain") or "openthoughts")
        candidates.append(
            Candidate(
                source_index=source_index,
                problem_raw=problem_raw,
                student_user_content=student_user_content,
                label=reference_boxed_answers[0],
                reference_solution=reference_solution,
                source=source,
                format_instruction=fmt_suffix,
            )
        )

    stats["candidates"] = len(candidates)
    return candidates, domain_counts, stats


# ---------------------------------------------------------------------------
# API generation
# ---------------------------------------------------------------------------

def _build_payload(
    candidate: Candidate,
    *,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_tokens: int,
    seed: int | None,
    system_prompt: str,
    enable_thinking: bool,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": candidate.student_user_content})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        # Qwen3: control thinking mode via chat_template_kwargs.
        # When enable_thinking=False this is a no-op for Qwen2.5-style models.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if seed is not None:
        payload["seed"] = seed
    return payload


async def _generate_one(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    candidate: Candidate,
    *,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_tokens: int,
    seed: int | None,
    system_prompt: str,
    enable_thinking: bool,
    retries: int,
) -> GenerationResult:
    payload = _build_payload(
        candidate,
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.post(url, json=payload, headers=headers)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = response.text[:1000]
                raise RuntimeError(f"HTTP {response.status_code}: {body}") from exc

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Response does not contain choices: {data!r}")
            message = choices[0].get("message") or {}
            return GenerationResult(text=message.get("content") or "")
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(min(2**attempt, 8))

    return GenerationResult(error=str(last_error))


async def _generate_all(
    candidates: list[Candidate],
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_tokens: int,
    seed: int | None,
    system_prompt: str,
    enable_thinking: bool,
    concurrency: int,
    retries: int,
    timeout: float,
) -> list[GenerationResult]:
    if not candidates:
        return []

    url = _chat_completions_url(api_base)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    queue: asyncio.Queue[tuple[int, Candidate]] = asyncio.Queue()
    for result_index, candidate in enumerate(candidates):
        queue.put_nowait((result_index, candidate))

    results = [GenerationResult(error="generation did not run")] * len(candidates)
    completed = 0
    total = len(candidates)
    progress_interval = max(1, total // 20)

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=120.0)) as client:
        async def _worker() -> None:
            nonlocal completed
            while True:
                try:
                    result_index, candidate = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                per_sample_seed = seed + result_index if seed is not None else None
                results[result_index] = await _generate_one(
                    client,
                    url,
                    headers,
                    candidate,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    seed=per_sample_seed,
                    system_prompt=system_prompt,
                    enable_thinking=enable_thinking,
                    retries=retries,
                )
                completed += 1
                if completed == total or completed % progress_interval == 0:
                    logger.info(f"Generated {completed}/{total} responses...")
                queue.task_done()

        workers = [asyncio.create_task(_worker()) for _ in range(max(1, min(concurrency, len(candidates))))]
        await queue.join()
        await asyncio.gather(*workers)

    return results


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def _build_sft_records(
    candidates: list[Candidate],
    results: list[GenerationResult],
    *,
    api_model: str,
    do_strip_thinking: bool = False,
) -> tuple[list[dict], dict[str, int], list[str]]:
    stats: dict[str, int] = {
        "generation_failed": 0,
        "generated_empty": 0,
        "filtered_no_boxed": 0,
        "filtered_repeated_lines": 0,
        "filtered_ngram_repetition": 0,
        "filtered_consecutive_repeat": 0,
        "written": 0,
    }
    first_errors: list[str] = []
    records: list[dict] = []

    for candidate, result in zip(candidates, results):
        if result.error is not None:
            stats["generation_failed"] += 1
            if len(first_errors) < 5:
                first_errors.append(f"source_index={candidate.source_index}: {result.error}")
            continue

        generated_response = _normalize(result.text)
        if not generated_response:
            stats["generated_empty"] += 1
            continue

        # For Qwen3 thinking mode: validate against the full output (thinking + answer),
        # then optionally strip the <think> block from what we store.
        sft_response = strip_thinking(generated_response) if do_strip_thinking else generated_response
        valid, reason = is_valid_output(sft_response)
        if not valid:
            stats[f"filtered_{reason}"] += 1
            continue

        generated_boxed_answers = _extract_boxed_answers(sft_response)

        records.append(
            {
                "messages": [
                    {"role": "user", "content": candidate.student_user_content},
                    {"role": "assistant", "content": sft_response},
                ],
                "label": candidate.label,
                "metadata": {
                    "raw_content": candidate.problem_raw,
                    "raw_problem": candidate.problem_raw,
                    "student_user_content": candidate.student_user_content,
                    "source": candidate.source,
                    "reference_solution": candidate.reference_solution,
                    "solution": candidate.reference_solution,
                    "format_instruction": candidate.format_instruction,
                    "generated_response": sft_response,
                    "raw_generated_response": generated_response,  # full output before stripping
                    "generated_boxed_answer": generated_boxed_answers[0] if generated_boxed_answers else "",
                    "api_model": api_model,
                },
            }
        )
        stats["written"] += 1

    return records, stats, first_errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_qwen3_llm_sft(
    *,
    output: str,
    api_base: str,
    api_key: str,
    model: str,
    answer_format: str,
    max_samples: int | None,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_tokens: int,
    seed: int | None,
    system_prompt: str,
    enable_thinking: bool,
    strip_thinking_output: bool,
    concurrency: int,
    retries: int,
    timeout: float,
    stats_only: bool,
) -> None:
    candidates, domain_counts, prefilter_stats = _load_candidates(
        answer_format=answer_format,
        max_samples=max_samples,
    )

    logger.info("")
    logger.info("=== Prefilter results ===")
    logger.info(f"  Scanned rows:                    {prefilter_stats['scanned']}")
    logger.info(f"  Skipped (non-math domain):       {prefilter_stats['skipped_domain']}")
    logger.info(f"  Skipped (no reference boxed):    {prefilter_stats['skipped_no_reference_boxed']}")
    logger.info(f"  Skipped (multiple reference):    {prefilter_stats['skipped_multi_reference_boxed']}")
    logger.info(f"  Skipped (empty problem):         {prefilter_stats['skipped_empty_problem']}")
    logger.info(f"  Candidates for generation:       {prefilter_stats['candidates']}")

    logger.info("")
    logger.info("=== Domain breakdown for scanned rows ===")
    for domain, count in domain_counts.most_common():
        logger.info(f"  {domain or '(empty)'}: {count}")

    if stats_only:
        logger.info("")
        logger.info("Stats-only mode; no API calls made and no output written.")
        return

    if not candidates:
        raise RuntimeError("No candidate prompts found; no output written.")

    # Query the server for max_model_len and pre-filter prompts that are too long.
    max_model_len, model_root = _fetch_server_info(api_base, api_key)
    if max_model_len is not None:
        logger.info(f"\nServer max_model_len={max_model_len}; filtering prompts where input + {max_tokens} > {max_model_len}.")
        candidates, n_skipped_len = _filter_by_length(
            candidates,
            model_path=model_root or model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
        )
        logger.info(f"  Skipped (too long for context): {n_skipped_len}")
        logger.info(f"  Remaining after length filter:  {len(candidates)}")
        if not candidates:
            raise RuntimeError("All candidates were filtered out as too long; lower --max-tokens or use a larger context server.")
    else:
        logger.info("\nCould not fetch max_model_len from server; skipping length pre-filter.")

    logger.info("")
    logger.info("=== Generation config ===")
    logger.info(f"  API base:        {api_base}")
    logger.info(f"  Model:           {model}")
    logger.info(f"  system_prompt:   {system_prompt!r}")
    logger.info(f"  max_tokens:      {max_tokens}")
    logger.info(f"  temperature:     {temperature}")
    logger.info(f"  top_p:           {top_p}")
    logger.info(f"  top_k:           {top_k}")
    logger.info(f"  enable_thinking: {enable_thinking}")
    logger.info(f"  strip_thinking:  {strip_thinking_output}")
    logger.info(f"  concurrency:     {concurrency}")
    logger.info(f"  retries:         {retries}")

    results = asyncio.run(
        _generate_all(
            candidates,
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            concurrency=concurrency,
            retries=retries,
            timeout=timeout,
        )
    )
    records, generation_stats, first_errors = _build_sft_records(
        candidates, results, api_model=model, do_strip_thinking=strip_thinking_output
    )

    if not records:
        raise RuntimeError("Generation produced no usable SFT records after filtering; no output written.")

    _write_records(output, records)

    logger.info("")
    logger.info("=== Generation filter results ===")
    logger.info(f"  API failures:                    {generation_stats['generation_failed']}")
    logger.info(f"  Skipped (empty response):        {generation_stats['generated_empty']}")
    logger.info(f"  Filtered (no boxed):             {generation_stats['filtered_no_boxed']}")
    logger.info(f"  Filtered (repeated lines):       {generation_stats['filtered_repeated_lines']}")
    logger.info(f"  Filtered (ngram repetition):     {generation_stats['filtered_ngram_repetition']}")
    logger.info(f"  Filtered (consecutive repeat):   {generation_stats['filtered_consecutive_repeat']}")
    logger.info(f"  Written SFT conversations:       {generation_stats['written']}")
    if first_errors:
        logger.info("")
        logger.info("First generation errors:")
        for error in first_errors:
            logger.info(f"  {error}")
    logger.info(f"  Output written to:               {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="/root/math/data/sft_qwen25math_7b_generated_openthoughts_math.parquet",
        help="Output path ending in .parquet or .jsonl.",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:30000/v1"),
        help="OpenAI-compatible API base URL. Defaults to OPENAI_API_BASE or http://127.0.0.1:30000/v1.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key for the OpenAI-compatible server. Defaults to OPENAI_API_KEY or EMPTY.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "Qwen3-4B-Instruct-2507"),
        help="Model name sent to /chat/completions. Defaults to OPENAI_MODEL or Qwen3-4B-Instruct-2507.",
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt",
        default=_QWEN3_SYSTEM_PROMPT,
        help=(
            "System prompt prepended to each conversation. "
            "Defaults to empty (Qwen3 style). "
            f"For Qwen2.5-Math pass: {_QWEN25MATH_SYSTEM_PROMPT!r}. "
            "Pass an empty string to omit the system message."
        ),
    )
    parser.add_argument(
        "--answer-format",
        dest="answer_format",
        choices=["answer", "boxed"],
        default="boxed",
        help=(
            "Format instruction style appended to each problem. "
            "'boxed' (default): model wraps answer in \\boxed{}. "
            "'answer': DAPO style - model outputs 'Answer: <value>' on last line."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after preparing N reference-filtered prompts for API generation.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling probability (Qwen3 default: 0.8).")
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=20,
        help="Top-k sampling (Qwen3 default: 20). Pass -1 to omit.",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        type=_parse_bool,
        default=False,
        help=(
            "Enable Qwen3 thinking mode (<think>...</think> blocks). "
            "Default False keeps behaviour identical to Qwen2.5. "
            "When True the full reasoning trace is stored in the SFT data "
            "unless --strip-thinking is also set."
        ),
    )
    parser.add_argument(
        "--strip-thinking",
        dest="strip_thinking",
        type=_parse_bool,
        default=False,
        help=(
            "Strip <think>...</think> blocks from stored assistant content. "
            "Only meaningful when --enable-thinking=True. "
            "When False (default) the thinking trace is kept in the SFT record."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum generated tokens per response.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed. Per-sample seed is base seed plus candidate index. Use --seed -1 to omit seed.",
    )
    parser.add_argument("--concurrency", type=int, default=16, help="Max in-flight API requests.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per request after the first attempt.")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Total request timeout in seconds.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print OpenThoughts prefilter statistics without API calls or output writing.",
    )

    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive.")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")

    seed = None if args.seed == -1 else args.seed
    top_k = None if args.top_k == -1 else args.top_k
    collect_qwen3_llm_sft(
        output=args.output,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        answer_format=args.answer_format,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
        max_tokens=args.max_tokens,
        seed=seed,
        system_prompt=args.system_prompt,
        enable_thinking=args.enable_thinking,
        strip_thinking_output=args.strip_thinking,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout=args.timeout,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    main()
