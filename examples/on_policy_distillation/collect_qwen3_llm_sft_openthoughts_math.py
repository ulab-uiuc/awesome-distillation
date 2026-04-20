"""Collect Qwen3-generated SFT conversations for OpenThoughts math prompts.

This script mirrors ``filter_openthoughts_math_sft.py`` for prompt selection:

  1. keep only rows with ``domain == "math"``;
  2. read the reference solution from ``ground_truth_solution`` with
     ``deepseek_solution`` as fallback;
  3. keep only rows with exactly one extractable reference ``\\boxed{...}``
     answer;
  4. skip rows with an empty problem.

Unlike ``filter_openthoughts_math_sft.py``, the assistant message is generated
through an OpenAI-compatible ``/chat/completions`` API. The generated response
must contain exactly one extractable ``\\boxed{...}`` answer to be written.

Supported output suffixes are ``.parquet`` and ``.jsonl``.
"""



"""python3 examples/on_policy_distillation/collect_qwen3_llm_sft_openthoughts_math.py \
  --output /root/math/data/sft_qwen3_1p7b_generated_openthoughts_math.parquet \
  --api-base http://localhost:30006/v1 \
  --api-key EMPTY \
  --model your-model-name \
  --enable-thinking false \
  --max-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.95 \
  --concurrency 16 \
  --max-samples 10
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


def _build_payload(
    candidate: Candidate,
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    enable_thinking: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": candidate.student_user_content}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        # This is the raw request-body form produced by OpenAI client's
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}}.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
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
    max_tokens: int,
    seed: int | None,
    enable_thinking: bool,
    retries: int,
) -> GenerationResult:
    payload = _build_payload(
        candidate,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
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
    max_tokens: int,
    seed: int | None,
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
                    max_tokens=max_tokens,
                    seed=per_sample_seed,
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


def _build_sft_records(
    candidates: list[Candidate],
    results: list[GenerationResult],
    *,
    api_model: str,
) -> tuple[list[dict], dict[str, int], list[str]]:
    stats = {
        "generation_failed": 0,
        "generated_empty": 0,
        "generated_no_boxed": 0,
        "generated_multi_boxed": 0,
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

        generated_boxed_answers = _extract_boxed_answers(generated_response)
        if len(generated_boxed_answers) == 0:
            stats["generated_no_boxed"] += 1
            continue
        if len(generated_boxed_answers) > 1:
            stats["generated_multi_boxed"] += 1
            continue

        records.append(
            {
                "messages": [
                    {"role": "user", "content": candidate.student_user_content},
                    {"role": "assistant", "content": generated_response},
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
                    "generated_response": generated_response,
                    "generated_boxed_answer": generated_boxed_answers[0],
                    "api_model": api_model,
                },
            }
        )
        stats["written"] += 1

    return records, stats, first_errors


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
    max_tokens: int,
    seed: int | None,
    enable_thinking: bool,
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

    logger.info("")
    logger.info("=== Generation config ===")
    logger.info(f"  API base:        {api_base}")
    logger.info(f"  Model:           {model}")
    logger.info(f"  enable_thinking: {enable_thinking}")
    logger.info(f"  max_tokens:      {max_tokens}")
    logger.info(f"  temperature:     {temperature}")
    logger.info(f"  top_p:           {top_p}")
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
            max_tokens=max_tokens,
            seed=seed,
            enable_thinking=enable_thinking,
            concurrency=concurrency,
            retries=retries,
            timeout=timeout,
        )
    )
    records, generation_stats, first_errors = _build_sft_records(candidates, results, api_model=model)

    if not records:
        raise RuntimeError("Generation produced no usable SFT records after boxed-answer filtering; no output written.")

    _write_records(output, records)

    logger.info("")
    logger.info("=== Generation filter results ===")
    logger.info(f"  API failures:                  {generation_stats['generation_failed']}")
    logger.info(f"  Skipped (empty response):      {generation_stats['generated_empty']}")
    logger.info(f"  Skipped (no generated boxed):  {generation_stats['generated_no_boxed']}")
    logger.info(f"  Skipped (multi generated):     {generation_stats['generated_multi_boxed']}")
    logger.info(f"  Written SFT conversations:     {generation_stats['written']}")
    if first_errors:
        logger.info("")
        logger.info("First generation errors:")
        for error in first_errors:
            logger.info(f"  {error}")
    logger.info(f"  Output written to:             {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="/root/math/data/sft_qwen3_1p7b_generated_openthoughts_math.parquet",
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
        default=os.environ.get("OPENAI_MODEL", "qwen3-1.7b"),
        help="Model name sent to /chat/completions. Defaults to OPENAI_MODEL or qwen3-1.7b.",
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
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum generated tokens per response.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed. Per-sample seed is base seed plus candidate index. Use --seed -1 to omit seed.",
    )
    parser.add_argument(
        "--enable-thinking",
        type=_parse_bool,
        default=False,
        help="Value for chat_template_kwargs.enable_thinking. Defaults to false.",
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
    collect_qwen3_llm_sft(
        output=args.output,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        answer_format=args.answer_format,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=seed,
        enable_thinking=args.enable_thinking,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout=args.timeout,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    main()
