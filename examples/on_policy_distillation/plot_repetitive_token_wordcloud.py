#!/usr/bin/env python3
"""
Plot repetitive-token word clouds from eval_student_teacher_inference JSONL files
or debug rollout PT dumps.

Examples:
python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
    --input eval_math500_student_teacher_inference_s1.7t8b_noanswer.jsonl \
    --output-dir ./repetitive_token_wordcloud

python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
    --input-pt output/debug_rollout/eval_0.pt \
    --output-dir ./repetitive_token_wordcloud

python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
  --input-pt output/debug_rollout/eval_0.pt \
  --output-dir repetitive_token_wordcloud/eval_0 \
  --response-field student_response \
  --tokenizer Qwen/Qwen3-1.7B \
  --repeat-detector compressibility \
  --compressibility-algorithm zlib \
  --compressibility-span-tokens 2 \
  --compressibility-context-tokens 128 \
  --compressibility-min-savings-pct 50 \
  --max-words 200

python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
  --input-pt output/debug_rollout/eval_149.pt \
  --output-dir repetitive_token_wordcloud/eval_149 \
  --response-field student_response \
  --tokenizer Qwen/Qwen3-1.7B \
  --repeat-detector ngram


python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
  --input-pt output/debug_rollout/eval_119.pt \
  --output-dir repetitive_token_wordcloud/eval_119 \
  --response-field student_response \
  --tokenizer Qwen/Qwen3-1.7B \
  --repeat-detector ngram


python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
  --input-pt output/debug_rollout/eval_89.pt \
  --output-dir repetitive_token_wordcloud/eval_89 \
  --response-field student_response \
  --tokenizer Qwen/Qwen3-1.7B \
  --repeat-detector ngram

python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
  --input-pt output/debug_rollout/eval_59.pt \
  --output-dir repetitive_token_wordcloud/eval_59 \
  --response-field student_response \
  --tokenizer Qwen/Qwen3-1.7B \
  --repeat-detector ngram

python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
  --input-pt output/debug_rollout/eval_29.pt \
  --output-dir repetitive_token_wordcloud/eval_29 \
  --response-field student_response \
  --tokenizer Qwen/Qwen3-1.7B \
  --repeat-detector ngram

  
python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
  --input-pt output/debug_rollout/eval_0.pt \
  --output-dir repetitive_token_wordcloud/eval_0 \
  --response-field student_response \
  --tokenizer Qwen/Qwen3-1.7B \
  --repeat-detector ngram

python examples/on_policy_distillation/plot_repetitive_token_wordcloud.py \
  --input-pt output/debug_rollout/eval_149.pt \
  --compare-input-pt output/debug_rollout/eval_0.pt \
  --compare-terms step first/second/third so but wait let \
  --output-dir repetitive_token_wordcloud/eval_149_vs_eval_0

"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import re
import sys
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with: pip install matplotlib"
    ) from e

from slime.utils.opd_token_stats import build_token_repetition_mask


def _int_ge(min_value: int):
    def _parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Expected an integer, got {value!r}.") from exc
        if parsed < min_value:
            raise argparse.ArgumentTypeError(f"Expected an integer >= {min_value}, got {parsed}.")
        return parsed

    return _parse


def _float_between(min_value: float, max_value: float):
    def _parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Expected a float, got {value!r}.") from exc
        if parsed < min_value or parsed > max_value:
            raise argparse.ArgumentTypeError(
                f"Expected a float in [{min_value}, {max_value}], got {parsed}."
            )
        return parsed

    return _parse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to eval_student_teacher_inference JSONL.")
    input_group.add_argument(
        "--input-pt",
        help="Path to a debug rollout PT dump (for example eval_0.pt).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the PDF word cloud and JSON summary.")
    parser.add_argument(
        "--response-field",
        default="student_response",
        choices=("student_response", "teacher_response"),
        help="Record field to analyze. Default: student_response.",
    )
    parser.add_argument(
        "--pt-sample-response-key",
        default="response",
        help="Sample field to read from each PT dump sample. Default: response.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-1.7B",
        help="Tokenizer source for response tokenization. Default: Qwen/Qwen3-1.7B.",
    )
    parser.add_argument(
        "--repeat-detector",
        default="compressibility",
        choices=("ngram", "compressibility"),
        help="Repetition detector to use. Default: ngram.",
    )
    parser.add_argument(
        "--repeat-ngram",
        type=_int_ge(2),
        default=3,
        help="Repeat n-gram size used when --repeat-detector=ngram. Default: 3.",
    )
    parser.add_argument(
        "--compressibility-algorithm",
        default="zlib",
        choices=("zlib", "gzip", "bz2", "lzma"),
        help="Compression backend used when --repeat-detector=compressibility. Default: zlib.",
    )
    parser.add_argument(
        "--compressibility-level",
        type=_int_ge(0),
        default=9,
        help="Compression level/preset used when --repeat-detector=compressibility. Default: 9.",
    )
    parser.add_argument(
        "--compressibility-span-tokens",
        type=_int_ge(1),
        default=2,
        help="Number of consecutive tokens grouped into one compressibility check. Default: 2.",
    )
    parser.add_argument(
        "--compressibility-context-tokens",
        type=_int_ge(1),
        default=128,
        help="How many previous tokens to expose as compression context. Default: 128.",
    )
    parser.add_argument(
        "--compressibility-min-savings-pct",
        type=_float_between(0.0, 100.0),
        default=50.0,
        help="Minimum chunk-level compression savings percentage to mark tokens as repetitive. Default: 50.0.",
    )
    parser.add_argument(
        "--max-words",
        type=_int_ge(1),
        default=200,
        help="Maximum number of tokens to display in the word cloud. Default: 200.",
    )
    parser.add_argument(
        "--font-path",
        default=None,
        help="Optional font path passed to wordcloud.WordCloud.",
    )
    parser.add_argument(
        "--compare-input-pt",
        default=None,
        help=(
            "Optional second debug rollout PT dump used to compare whole-response term "
            "frequencies against the primary input."
        ),
    )
    parser.add_argument(
        "--compare-terms",
        nargs="+",
        default=None,
        help=(
            "Optional list of term queries used with --compare-input-pt. Use slash-delimited "
            "variants such as first/second/third to aggregate several words into one bar."
        ),
    )
    return parser.parse_args(argv)


def _load_tokenizer(source: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source, trust_remote_code=True)


def _load_wordcloud_class():
    try:
        from wordcloud import WordCloud
    except ImportError as e:
        raise SystemExit(
            "wordcloud is required for plotting. Install it with: pip install wordcloud"
        ) from e
    return WordCloud


def _make_unicode_to_bytes() -> dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = list(bs)
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {chr(c): byte for byte, c in zip(bs, cs)}


_UNICODE_TO_BYTES = _make_unicode_to_bytes()
_WORD_RE = re.compile(r"[A-Za-z]+")
_DEFAULT_COMPARE_TERMS = ("step", "first/second/third", "but", "wait", "let")
_COMPARE_RATE_SCALE = 1000.0


def _decode_ids_to_text_tokens(tokenizer, token_ids: list[int], n: int) -> list[str]:
    out: list[str] = []
    for tid in token_ids[:n]:
        try:
            piece = tokenizer.decode(
                [int(tid)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if "\ufffd" in piece:
                raw_list = tokenizer.convert_ids_to_tokens([int(tid)])
                raw = raw_list[0] if raw_list else ""
                try:
                    byte_seq = bytes([_UNICODE_TO_BYTES[c] for c in raw])
                    piece = byte_seq.decode("utf-8", errors="replace")
                except Exception:
                    piece = f"<0x{int(tid):04x}>"
        except Exception:
            piece = str(tid)
        out.append(piece)
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out


def _make_token_text_visible(text: str) -> str:
    if text == "":
        return "<empty>"

    pieces: list[str] = []
    for ch in text:
        if ch == " ":
            pieces.append("<space>")
        elif ch == "\n":
            pieces.append("<newline>")
        elif ch == "\t":
            pieces.append("<tab>")
        elif ch == "\r":
            pieces.append("<cr>")
        elif ch.isprintable():
            pieces.append(ch)
        else:
            pieces.append(f"<0x{ord(ch):02x}>")
    visible = "".join(pieces)
    return visible or "<empty>"


def _normalize_token_text_for_display(text: str) -> str:
    if text == "":
        return ""
    # Wordcloud labels should emphasize the lexical content of the token rather
    # than tokenizer-added whitespace prefixes/suffixes.
    return "".join(ch for ch in text if not ch.isspace())


def _is_display_filtered_token_text(text: str) -> bool:
    return text == "" or text.isspace()


def _is_display_filtered_numeric_token_text(text: str) -> bool:
    normalized_text = _normalize_token_text_for_display(text)
    return normalized_text != "" and normalized_text.isdigit()


def _is_display_filtered_punctuation_token_text(text: str) -> bool:
    normalized_text = _normalize_token_text_for_display(text)
    if normalized_text == "":
        return False
    return all(unicodedata.category(ch).startswith(("P", "S")) for ch in normalized_text)


def _extract_input_ids(encoded: Any) -> list[int]:
    input_ids = encoded
    if isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids

    if input_ids is None:
        return []
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


@dataclass(frozen=True)
class RepeatDetectionConfig:
    detector: str
    repeat_ngram: int = 3
    compressibility_algorithm: str = "zlib"
    compressibility_level: int = 9
    compressibility_span_tokens: int = 2
    compressibility_context_tokens: int = 128
    compressibility_min_savings_pct: float = 50.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RepeatDetectionConfig":
        return cls(
            detector=str(args.repeat_detector),
            repeat_ngram=int(args.repeat_ngram),
            compressibility_algorithm=str(args.compressibility_algorithm),
            compressibility_level=int(args.compressibility_level),
            compressibility_span_tokens=int(args.compressibility_span_tokens),
            compressibility_context_tokens=int(args.compressibility_context_tokens),
            compressibility_min_savings_pct=float(args.compressibility_min_savings_pct),
        )

    def output_suffix(self) -> str:
        if self.detector == "ngram":
            return f"repeat_ngram{self.repeat_ngram}"
        savings_tag = f"{self.compressibility_min_savings_pct:.1f}".replace(".", "p")
        return (
            "repeat_compressibility"
            f"_{self.compressibility_algorithm}"
            f"_span{self.compressibility_span_tokens}"
            f"_ctx{self.compressibility_context_tokens}"
            f"_s{savings_tag}"
        )

    def title_fragment(self) -> str:
        if self.detector == "ngram":
            return f"repeat_ngram={self.repeat_ngram}"
        return (
            "repeat_detector=compressibility"
            f", algorithm={self.compressibility_algorithm}"
            f", span_tokens={self.compressibility_span_tokens}"
            f", context_tokens={self.compressibility_context_tokens}"
            f", min_savings_pct={self.compressibility_min_savings_pct:.1f}"
        )

    def as_summary_fields(self) -> dict[str, Any]:
        return {
            "repeat_detector": self.detector,
            "repeat_ngram": int(self.repeat_ngram) if self.detector == "ngram" else None,
            "compressibility_algorithm": (
                self.compressibility_algorithm if self.detector == "compressibility" else None
            ),
            "compressibility_level": int(self.compressibility_level) if self.detector == "compressibility" else None,
            "compressibility_span_tokens": (
                int(self.compressibility_span_tokens) if self.detector == "compressibility" else None
            ),
            "compressibility_context_tokens": (
                int(self.compressibility_context_tokens) if self.detector == "compressibility" else None
            ),
            "compressibility_min_savings_pct": (
                float(self.compressibility_min_savings_pct) if self.detector == "compressibility" else None
            ),
        }


@dataclass
class AnalysisResult:
    num_records: int = 0
    num_skipped_records: int = 0
    num_analyzed_records: int = 0
    total_response_tokens: int = 0
    total_repetitive_token_positions: int = 0
    repetitive_token_counts: Counter[int] = field(default_factory=Counter)

    @property
    def repeat_ratio(self) -> float:
        if self.total_response_tokens <= 0:
            return 0.0
        return float(self.total_repetitive_token_positions) / float(self.total_response_tokens)


@dataclass(frozen=True)
class CompareTerm:
    label: str
    variants: tuple[str, ...]


@dataclass
class WordFrequencyStats:
    num_records: int = 0
    num_analyzed_records: int = 0
    total_words: int = 0
    word_counts: Counter[str] = field(default_factory=Counter)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} of {path!s}: {exc}") from exc


def _extract_pt_sample_response(sample: Any, response_key: str) -> str | None:
    if isinstance(sample, dict):
        response = sample.get(response_key)
    else:
        response = getattr(sample, response_key, None)
    return response if isinstance(response, str) else None


def iter_pt_samples(
    path: str | Path,
    *,
    response_field: str,
    pt_sample_response_key: str,
) -> Iterable[dict[str, Any]]:
    import torch

    pack = torch.load(path, map_location="cpu")
    if not isinstance(pack, dict):
        raise ValueError(f"Expected PT dump {path!s} to contain a dict, got {type(pack).__name__}.")

    samples = pack.get("samples")
    if not isinstance(samples, list):
        raise ValueError(
            f"Expected PT dump {path!s} to contain a list under 'samples', got {type(samples).__name__}."
        )

    for idx, sample in enumerate(samples):
        response = _extract_pt_sample_response(sample, pt_sample_response_key)
        if response is None:
            yield {}
            continue
        yield {
            response_field: response,
            "source_index": idx,
        }


def iter_input_records(
    *,
    input_path: Path,
    input_pt: bool,
    response_field: str,
    pt_sample_response_key: str,
) -> Iterable[dict[str, Any]]:
    if input_pt:
        return iter_pt_samples(
            input_path,
            response_field=response_field,
            pt_sample_response_key=pt_sample_response_key,
        )
    return iter_jsonl(input_path)


def parse_compare_terms(raw_terms: list[str] | None) -> list[CompareTerm]:
    term_specs = raw_terms or list(_DEFAULT_COMPARE_TERMS)
    parsed_terms: list[CompareTerm] = []
    for term_spec in term_specs:
        parts = [part.strip().lower() for part in str(term_spec).split("/") if part.strip()]
        if not parts:
            continue
        parsed_terms.append(
            CompareTerm(
                label="/".join(parts),
                variants=tuple(dict.fromkeys(parts)),
            )
        )
    if not parsed_terms:
        raise ValueError("No valid compare terms were provided.")
    return parsed_terms


def collect_response_word_frequencies(
    records: Iterable[dict[str, Any]],
    *,
    response_field: str,
) -> WordFrequencyStats:
    stats = WordFrequencyStats()
    for record in records:
        stats.num_records += 1
        response = record.get(response_field)
        if not isinstance(response, str) or response == "":
            continue

        # Split contractions such as "let's" into "let" and "s" so common
        # reasoning markers can be queried with their base form.
        words = [match.group(0).lower() for match in _WORD_RE.finditer(response)]
        if not words:
            continue

        stats.num_analyzed_records += 1
        stats.total_words += len(words)
        stats.word_counts.update(words)
    return stats


def build_term_frequency_comparison_summary(
    *,
    primary_input_path: Path,
    primary_stats: WordFrequencyStats,
    compare_input_path: Path,
    compare_stats: WordFrequencyStats,
    response_field: str,
    compare_terms: list[CompareTerm],
) -> dict[str, Any]:
    rows = []
    for compare_term in compare_terms:
        primary_count = sum(primary_stats.word_counts[variant] for variant in compare_term.variants)
        compare_count = sum(compare_stats.word_counts[variant] for variant in compare_term.variants)
        primary_rate = (
            _COMPARE_RATE_SCALE * float(primary_count) / float(primary_stats.total_words)
            if primary_stats.total_words > 0
            else 0.0
        )
        compare_rate = (
            _COMPARE_RATE_SCALE * float(compare_count) / float(compare_stats.total_words)
            if compare_stats.total_words > 0
            else 0.0
        )
        rows.append(
            {
                "label": compare_term.label,
                "variants": list(compare_term.variants),
                "primary_count": int(primary_count),
                "compare_count": int(compare_count),
                "primary_rate_per_1k_words": primary_rate,
                "compare_rate_per_1k_words": compare_rate,
                "delta_rate_per_1k_words": primary_rate - compare_rate,
            }
        )

    return {
        "response_field": response_field,
        "rate_unit": "per_1k_words",
        "primary_input": {
            "path": str(primary_input_path),
            "num_records": int(primary_stats.num_records),
            "num_analyzed_records": int(primary_stats.num_analyzed_records),
            "total_words": int(primary_stats.total_words),
        },
        "compare_input": {
            "path": str(compare_input_path),
            "num_records": int(compare_stats.num_records),
            "num_analyzed_records": int(compare_stats.num_analyzed_records),
            "total_words": int(compare_stats.total_words),
        },
        "terms": rows,
    }


def _compressed_length(data: bytes, *, algorithm: str, level: int) -> int:
    if algorithm == "zlib":
        return len(zlib.compress(data, level))
    if algorithm == "gzip":
        return len(gzip.compress(data, compresslevel=level))
    if algorithm == "bz2":
        return len(bz2.compress(data, compresslevel=max(1, min(level, 9))))
    if algorithm == "lzma":
        return len(lzma.compress(data, preset=max(0, min(level, 9))))
    raise ValueError(f"Unsupported compressibility algorithm: {algorithm!r}")


def build_token_compressibility_mask(
    token_texts: list[str],
    *,
    algorithm: str,
    level: int,
    span_tokens: int,
    context_tokens: int,
    min_savings_pct: float,
) -> list[int]:
    if span_tokens < 1:
        raise ValueError(f"span_tokens must be >= 1, got {span_tokens}.")
    if context_tokens < 1:
        raise ValueError(f"context_tokens must be >= 1, got {context_tokens}.")

    token_bytes = [token_text.encode("utf-8", errors="replace") for token_text in token_texts]
    repeat_mask = [0] * len(token_bytes)
    if len(token_bytes) < span_tokens:
        return repeat_mask

    for start in range(len(token_bytes) - span_tokens + 1):
        chunk = b"".join(token_bytes[start : start + span_tokens])
        if not chunk:
            continue
        context_start = max(0, start - context_tokens)
        context = b"".join(token_bytes[context_start:start])
        if not context:
            continue
        marginal_compressed_bytes = _compressed_length(
            context + chunk,
            algorithm=algorithm,
            level=level,
        ) - _compressed_length(
            context,
            algorithm=algorithm,
            level=level,
        )
        marginal_compressed_bytes = max(0, marginal_compressed_bytes)
        savings_pct = 100.0 * (1.0 - float(marginal_compressed_bytes) / float(len(chunk)))
        if savings_pct >= min_savings_pct:
            for pos in range(start, start + span_tokens):
                if token_bytes[pos]:
                    repeat_mask[pos] = 1
    return repeat_mask


def analyze_records(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer,
    response_field: str,
    detection_config: RepeatDetectionConfig,
) -> AnalysisResult:
    result = AnalysisResult()
    for record in records:
        result.num_records += 1
        response = record.get(response_field)
        if not isinstance(response, str) or response == "":
            result.num_skipped_records += 1
            continue

        token_ids = _extract_input_ids(tokenizer(response, add_special_tokens=False))
        if not token_ids:
            result.num_skipped_records += 1
            continue

        if detection_config.detector == "ngram":
            repeat_mask = build_token_repetition_mask(token_ids, ngram=detection_config.repeat_ngram)
        elif detection_config.detector == "compressibility":
            token_texts = _decode_ids_to_text_tokens(tokenizer, token_ids, len(token_ids))
            repeat_mask = build_token_compressibility_mask(
                token_texts,
                algorithm=detection_config.compressibility_algorithm,
                level=detection_config.compressibility_level,
                span_tokens=detection_config.compressibility_span_tokens,
                context_tokens=detection_config.compressibility_context_tokens,
                min_savings_pct=detection_config.compressibility_min_savings_pct,
            )
        else:
            raise ValueError(f"Unsupported repeat detector: {detection_config.detector!r}")
        result.num_analyzed_records += 1
        result.total_response_tokens += len(token_ids)
        result.total_repetitive_token_positions += int(sum(repeat_mask))

        for token_id, is_repeat in zip(token_ids, repeat_mask, strict=False):
            if is_repeat:
                result.repetitive_token_counts[int(token_id)] += 1
    return result


def build_token_display_map(tokenizer, token_ids: Iterable[int]) -> tuple[dict[int, str], dict[int, str]]:
    ordered_token_ids = [int(token_id) for token_id in token_ids]
    decoded = _decode_ids_to_text_tokens(tokenizer, ordered_token_ids, len(ordered_token_ids))
    raw_text_by_id: dict[int, str] = {}
    visible_by_id: dict[int, str] = {}
    for token_id, raw_text in zip(ordered_token_ids, decoded, strict=False):
        raw_text_by_id[token_id] = raw_text
        normalized_text = _normalize_token_text_for_display(raw_text)
        visible_by_id[token_id] = _make_token_text_visible(normalized_text)
    return visible_by_id, raw_text_by_id


def filter_display_token_counts(
    repetitive_token_counts: Counter[int],
    raw_text_by_id: dict[int, str],
) -> tuple[Counter[int], dict[str, int]]:
    filtered_counts: Counter[int] = Counter()
    filtered_stats = {
        "whitespace": 0,
        "numeric": 0,
        "punctuation": 0,
    }
    for token_id, count in repetitive_token_counts.items():
        raw_text = raw_text_by_id.get(int(token_id), "")
        if _is_display_filtered_token_text(raw_text):
            filtered_stats["whitespace"] += int(count)
            continue
        if _is_display_filtered_numeric_token_text(raw_text):
            filtered_stats["numeric"] += int(count)
            continue
        if _is_display_filtered_punctuation_token_text(raw_text):
            filtered_stats["punctuation"] += int(count)
            continue
        filtered_counts[int(token_id)] = int(count)
    return filtered_counts, filtered_stats


def aggregate_display_token_counts(
    display_token_counts: Counter[int],
    display_by_id: dict[int, str],
    raw_text_by_id: dict[int, str],
) -> tuple[Counter[str], dict[str, dict[str, Any]]]:
    aggregated_counts: Counter[str] = Counter()
    aggregated_meta: dict[str, dict[str, Any]] = {}

    for token_id, count in display_token_counts.items():
        display_token = display_by_id[int(token_id)]
        aggregated_counts[display_token] += int(count)

        meta = aggregated_meta.setdefault(
            display_token,
            {
                "token_ids": [],
                "decoded_tokens": [],
            },
        )
        meta["token_ids"].append(int(token_id))
        raw_text = raw_text_by_id.get(int(token_id), "")
        if raw_text not in meta["decoded_tokens"]:
            meta["decoded_tokens"].append(raw_text)

    for meta in aggregated_meta.values():
        meta["token_ids"].sort()
        meta["decoded_tokens"].sort()

    return aggregated_counts, aggregated_meta


def build_summary(
    *,
    input_path: Path,
    response_field: str,
    tokenizer_source: str,
    detection_config: RepeatDetectionConfig,
    result: AnalysisResult,
    display_token_counts: Counter[str],
    filtered_display_token_positions: dict[str, int],
    aggregated_display_meta: dict[str, dict[str, Any]],
    topk: int = 50,
) -> dict[str, Any]:
    top_tokens = []
    for display_token, count in display_token_counts.most_common(topk):
        meta = aggregated_display_meta.get(display_token, {})
        top_tokens.append(
            {
                "display_token": display_token,
                "token_ids": meta.get("token_ids", []),
                "decoded_tokens": meta.get("decoded_tokens", []),
                "count": int(count),
            }
        )

    return {
        "input_path": str(input_path),
        "response_field": response_field,
        "tokenizer": tokenizer_source,
        **detection_config.as_summary_fields(),
        "num_records": int(result.num_records),
        "num_skipped_records": int(result.num_skipped_records),
        "num_analyzed_records": int(result.num_analyzed_records),
        "total_response_tokens": int(result.total_response_tokens),
        "total_repetitive_token_positions": int(result.total_repetitive_token_positions),
        "displayed_repetitive_token_positions": int(sum(display_token_counts.values())),
        "filtered_whitespace_repetitive_token_positions": int(filtered_display_token_positions["whitespace"]),
        "filtered_numeric_repetitive_token_positions": int(filtered_display_token_positions["numeric"]),
        "filtered_punctuation_repetitive_token_positions": int(filtered_display_token_positions["punctuation"]),
        "filtered_noncontent_repetitive_token_positions": int(sum(filtered_display_token_positions.values())),
        "repeat_ratio": result.repeat_ratio,
        "top_repetitive_tokens": top_tokens,
    }


def _render_empty_wordcloud(out_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.text(
        0.5,
        0.56,
        message,
        ha="center",
        va="center",
        fontsize=24,
        color="#333333",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.44,
        title,
        ha="center",
        va="center",
        fontsize=12,
        color="#666666",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_wordcloud_figure(wordcloud, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_term_frequency_comparison_figure(summary: dict[str, Any], out_path: Path) -> None:
    terms = summary["terms"]
    primary_name = Path(summary["primary_input"]["path"]).stem
    compare_name = Path(summary["compare_input"]["path"]).stem
    if not terms:
        _render_empty_wordcloud(
            out_path,
            f"Term frequency comparison ({primary_name} vs {compare_name})",
            "No compare terms requested",
        )
        return

    labels = [term["label"] for term in terms]
    primary_rates = [term["primary_rate_per_1k_words"] for term in terms]
    compare_rates = [term["compare_rate_per_1k_words"] for term in terms]
    delta_rates = [term["delta_rate_per_1k_words"] for term in terms]
    positions = list(range(len(labels)))
    bar_height = 0.34

    fig_height = max(4.8, 1.2 + 0.8 * len(labels))
    fig, (ax_rates, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(16, fig_height),
        gridspec_kw={"width_ratios": [2.4, 1.4]},
    )

    ax_rates.barh(
        [pos - bar_height / 2 for pos in positions],
        primary_rates,
        height=bar_height,
        color="#4C72B0",
        label=primary_name,
    )
    ax_rates.barh(
        [pos + bar_height / 2 for pos in positions],
        compare_rates,
        height=bar_height,
        color="#DD8452",
        label=compare_name,
    )
    ax_rates.set_yticks(positions)
    ax_rates.set_yticklabels(labels)
    ax_rates.invert_yaxis()
    ax_rates.set_xlabel("Occurrences per 1k words")
    ax_rates.set_title("Absolute frequency")
    ax_rates.legend(loc="best")

    delta_colors = ["#55A868" if delta >= 0 else "#C44E52" for delta in delta_rates]
    ax_delta.barh(positions, delta_rates, height=0.5, color=delta_colors)
    ax_delta.axvline(0.0, color="#444444", linewidth=1.0)
    ax_delta.set_yticks(positions)
    ax_delta.set_yticklabels([])
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("Delta per 1k words")
    ax_delta.set_title(f"{primary_name} - {compare_name}")

    fig.suptitle(
        f"Whole-response term frequency comparison ({summary['response_field']})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_outputs(
    *,
    output_dir: Path,
    input_path: Path,
    response_field: str,
    detection_config: RepeatDetectionConfig,
    max_words: int,
    font_path: str | None,
    summary: dict[str, Any],
    display_token_counts: Counter[str],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{input_path.stem}_{response_field}_{detection_config.output_suffix()}"
    summary_path = output_dir / f"{prefix}_summary.json"
    wordcloud_path = output_dir / f"{prefix}_wordcloud.pdf"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    title = (
        f"Repetitive token word cloud ({response_field}, {detection_config.title_fragment()}, "
        f"repeat_ratio={summary['repeat_ratio']:.4f})"
    )

    if not display_token_counts:
        message = "No content-like repetitive tokens found"
        if summary["total_repetitive_token_positions"] <= 0:
            message = "No repetitive tokens found"
        _render_empty_wordcloud(wordcloud_path, title, message)
        return summary_path, wordcloud_path

    WordCloud = _load_wordcloud_class()
    frequencies = {
        display_token: int(count)
        for display_token, count in display_token_counts.most_common(max_words)
    }
    wordcloud = WordCloud(
        width=1800,
        height=1000,
        background_color="white",
        prefer_horizontal=1.0,
        collocations=False,
        max_words=max_words,
        font_path=font_path,
    ).generate_from_frequencies(frequencies)
    _save_wordcloud_figure(wordcloud, wordcloud_path, title)
    return summary_path, wordcloud_path


def write_term_frequency_comparison_outputs(
    *,
    output_dir: Path,
    primary_input_path: Path,
    compare_input_path: Path,
    response_field: str,
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{primary_input_path.stem}_vs_{compare_input_path.stem}_{response_field}_term_frequency_comparison"
    summary_path = output_dir / f"{prefix}_summary.json"
    figure_path = output_dir / f"{prefix}.pdf"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    _save_term_frequency_comparison_figure(summary, figure_path)
    return summary_path, figure_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare_terms is not None and args.compare_input_pt is None:
        raise SystemExit("--compare-terms requires --compare-input-pt.")

    input_path = Path(args.input_pt or args.input)
    output_dir = Path(args.output_dir)
    detection_config = RepeatDetectionConfig.from_args(args)
    records = iter_input_records(
        input_path=input_path,
        input_pt=args.input_pt is not None,
        response_field=args.response_field,
        pt_sample_response_key=str(args.pt_sample_response_key),
    )

    tokenizer = _load_tokenizer(args.tokenizer)
    result = analyze_records(
        records,
        tokenizer=tokenizer,
        response_field=args.response_field,
        detection_config=detection_config,
    )
    display_by_id, raw_text_by_id = build_token_display_map(
        tokenizer,
        result.repetitive_token_counts.keys(),
    )
    display_token_counts, filtered_display_token_positions = filter_display_token_counts(
        result.repetitive_token_counts,
        raw_text_by_id,
    )
    aggregated_display_token_counts, aggregated_display_meta = aggregate_display_token_counts(
        display_token_counts,
        display_by_id,
        raw_text_by_id,
    )
    summary = build_summary(
        input_path=input_path,
        response_field=args.response_field,
        tokenizer_source=args.tokenizer,
        detection_config=detection_config,
        result=result,
        display_token_counts=aggregated_display_token_counts,
        filtered_display_token_positions=filtered_display_token_positions,
        aggregated_display_meta=aggregated_display_meta,
    )
    summary_path, wordcloud_path = write_outputs(
        output_dir=output_dir,
        input_path=input_path,
        response_field=args.response_field,
        detection_config=detection_config,
        max_words=args.max_words,
        font_path=args.font_path,
        summary=summary,
        display_token_counts=aggregated_display_token_counts,
    )

    print(f"Analyzed {result.num_analyzed_records}/{result.num_records} records from {input_path}")
    print(
        "Repeat positions: "
        f"{result.total_repetitive_token_positions}/{result.total_response_tokens} "
        f"({result.repeat_ratio:.4%})"
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved word cloud: {wordcloud_path}")

    if args.compare_input_pt is not None:
        compare_input_path = Path(args.compare_input_pt)
        compare_terms = parse_compare_terms(args.compare_terms)
        primary_word_stats = collect_response_word_frequencies(
            iter_input_records(
                input_path=input_path,
                input_pt=args.input_pt is not None,
                response_field=args.response_field,
                pt_sample_response_key=str(args.pt_sample_response_key),
            ),
            response_field=args.response_field,
        )
        compare_word_stats = collect_response_word_frequencies(
            iter_input_records(
                input_path=compare_input_path,
                input_pt=True,
                response_field=args.response_field,
                pt_sample_response_key=str(args.pt_sample_response_key),
            ),
            response_field=args.response_field,
        )
        comparison_summary = build_term_frequency_comparison_summary(
            primary_input_path=input_path,
            primary_stats=primary_word_stats,
            compare_input_path=compare_input_path,
            compare_stats=compare_word_stats,
            response_field=args.response_field,
            compare_terms=compare_terms,
        )
        comparison_summary_path, comparison_figure_path = write_term_frequency_comparison_outputs(
            output_dir=output_dir,
            primary_input_path=input_path,
            compare_input_path=compare_input_path,
            response_field=args.response_field,
            summary=comparison_summary,
        )
        print(f"Saved term comparison summary: {comparison_summary_path}")
        print(f"Saved term comparison figure: {comparison_figure_path}")
        for row in comparison_summary["terms"]:
            print(
                f"Term {row['label']}: "
                f"{input_path.stem}={row['primary_count']} ({row['primary_rate_per_1k_words']:.3f}/1k), "
                f"{compare_input_path.stem}={row['compare_count']} ({row['compare_rate_per_1k_words']:.3f}/1k), "
                f"delta={row['delta_rate_per_1k_words']:.3f}/1k"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
