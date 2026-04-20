"""Filter OpenThoughts math rows into SFT conversation data.

This script intentionally mirrors the filtering criteria in
``filter_openthoughts_math.py``:

  1. keep only rows with ``domain == "math"``;
  2. read the reference solution from ``ground_truth_solution`` with
     ``deepseek_solution`` as fallback;
  3. keep only rows with exactly one extractable ``\\boxed{...}`` answer;
  4. skip rows with an empty problem or empty reference solution.

Unlike ``filter_openthoughts_math.py``, the output is SFT-ready data with a
``messages`` field:

    {
        "messages": [
            {"role": "user", "content": "<formatted problem>"},
            {"role": "assistant", "content": "<reference solution>"}
        ],
        "label": "<boxed answer>",
        "metadata": {...}
    }

Supported output suffixes are ``.parquet`` and ``.jsonl``.
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import Counter

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
    #   python examples/on_policy_distillation/filter_openthoughts_math_sft.py
    from filter_openthoughts_math import (  # type: ignore
        _ANSWER_FORMAT_INSTRUCTION,
        _BOXED_FORMAT_INSTRUCTION,
        _FORMAT_SUFFIX,
        _extract_boxed_answers,
        _normalize,
    )


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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


def filter_math_sft(
    output: str,
    answer_format: str = "boxed",
    max_samples: int | None = None,
    stats_only: bool = False,
) -> None:
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
    skipped_domain_breakdown: Counter = Counter()
    kept = 0
    skipped_domain = 0
    skipped_no_answer = 0
    skipped_multi_answer = 0
    skipped_empty_problem = 0
    records: list[dict] = []

    for row in ds:
        if max_samples is not None and kept >= max_samples:
            break

        domain = _normalize(row.get("domain", ""))
        domain_counts[domain] += 1

        if domain != "math":
            skipped_domain += 1
            skipped_domain_breakdown[domain] += 1
            continue

        reference_solution = _normalize(row.get("ground_truth_solution") or row.get("deepseek_solution") or "")
        boxed_answers = _extract_boxed_answers(reference_solution)
        if len(boxed_answers) == 0:
            skipped_no_answer += 1
            continue
        if len(boxed_answers) > 1:
            skipped_multi_answer += 1
            continue

        problem_raw = _normalize(row.get("problem", ""))
        if not problem_raw:
            skipped_empty_problem += 1
            continue

        if stats_only:
            kept += 1
            continue

        label = boxed_answers[0]
        student_user_content = fmt_template.format(problem=problem_raw)
        source = _normalize(row.get("source") or row.get("domain") or "openthoughts")
        records.append(
            {
                "messages": [
                    {"role": "user", "content": student_user_content},
                    {"role": "assistant", "content": reference_solution},
                ],
                "label": label,
                "metadata": {
                    "raw_content": problem_raw,
                    "raw_problem": problem_raw,
                    "student_user_content": student_user_content,
                    "source": source,
                    "reference_solution": reference_solution,
                    "solution": reference_solution,
                    "format_instruction": fmt_suffix,
                },
            }
        )
        kept += 1

    if not stats_only:
        _write_records(output, records)

    logger.info("")
    logger.info("=== Domain breakdown ===")
    for domain, count in domain_counts.most_common():
        logger.info(f"  {domain or '(empty)'}: {count}")

    logger.info("")
    logger.info("=== Filter results ===")
    logger.info(f"  Total samples:                {len(ds)}")
    logger.info(f"  Skipped (non-math domain):    {skipped_domain}")
    for domain, count in skipped_domain_breakdown.most_common():
        logger.info(f"    - {domain or '(empty)'}: {count}")
    logger.info(f"  Skipped (no boxed answer):    {skipped_no_answer}")
    logger.info(f"  Skipped (multiple boxed):     {skipped_multi_answer}")
    logger.info(f"  Skipped (empty problem):      {skipped_empty_problem}")
    logger.info(f"  Kept (SFT math conversations): {kept}")
    if not stats_only:
        logger.info(f"  Output written to:            {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="/root/math/data/sft_openthoughts_math.parquet",
        help="Output path ending in .parquet or .jsonl.",
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
        help="Stop after writing N kept samples, useful for smoke tests.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print filter statistics without writing an output file.",
    )
    args = parser.parse_args()
    filter_math_sft(
        output=args.output,
        answer_format=args.answer_format,
        max_samples=args.max_samples,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    main()
