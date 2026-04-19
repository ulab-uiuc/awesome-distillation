import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples.on_policy_distillation import plot_repetitive_token_wordcloud as mod


class FakeWhitespaceTokenizer:
    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def _get_id(self, token: str) -> int:
        token_id = self.token_to_id.get(token)
        if token_id is None:
            token_id = len(self.token_to_id) + 1
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return token_id

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        tokens = text.split()
        return {"input_ids": [self._get_id(token) for token in tokens]}

    def decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return self.id_to_token[int(token_ids[0])]

    def convert_ids_to_tokens(self, token_ids):
        return [self.id_to_token[int(token_ids[0])]]


class FakeDisplayTokenizer:
    def __init__(self, decoded_by_id, raw_by_id=None):
        self.decoded_by_id = decoded_by_id
        self.raw_by_id = raw_by_id or decoded_by_id

    def decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return self.decoded_by_id[int(token_ids[0])]

    def convert_ids_to_tokens(self, token_ids):
        return [self.raw_by_id[int(token_ids[0])]]


class FakeWordCloud:
    last_frequencies = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate_from_frequencies(self, frequencies):
        FakeWordCloud.last_frequencies = dict(frequencies)
        return self

    def __array__(self, dtype=None):
        arr = np.full((16, 16, 3), 255, dtype=np.uint8)
        if dtype is not None:
            return arr.astype(dtype)
        return arr


def _ngram_config(repeat_ngram: int = 3) -> mod.RepeatDetectionConfig:
    return mod.RepeatDetectionConfig(detector="ngram", repeat_ngram=repeat_ngram)


def _compressibility_config(
    *,
    algorithm: str = "zlib",
    level: int = 9,
    span_tokens: int = 2,
    context_tokens: int = 128,
    min_savings_pct: float = 50.0,
) -> mod.RepeatDetectionConfig:
    return mod.RepeatDetectionConfig(
        detector="compressibility",
        compressibility_algorithm=algorithm,
        compressibility_level=level,
        compressibility_span_tokens=span_tokens,
        compressibility_context_tokens=context_tokens,
        compressibility_min_savings_pct=min_savings_pct,
    )


def test_analyze_records_matches_trigram_repeat_end_position_only():
    tokenizer = FakeWhitespaceTokenizer()
    result = mod.analyze_records(
        [{"student_response": "a b c a b c"}],
        tokenizer=tokenizer,
        response_field="student_response",
        detection_config=_ngram_config(repeat_ngram=3),
    )

    c_id = tokenizer.token_to_id["c"]
    assert result.num_records == 1
    assert result.num_skipped_records == 0
    assert result.total_response_tokens == 6
    assert result.total_repetitive_token_positions == 1
    assert result.repetitive_token_counts == {c_id: 1}


def test_analyze_records_matches_bigram_repeat_and_skips_empty_records():
    tokenizer = FakeWhitespaceTokenizer()
    result = mod.analyze_records(
        [
            {"student_response": "a b a b"},
            {"student_response": ""},
            {"student_response": None},
            {},
        ],
        tokenizer=tokenizer,
        response_field="student_response",
        detection_config=_ngram_config(repeat_ngram=2),
    )

    b_id = tokenizer.token_to_id["b"]
    assert result.num_records == 4
    assert result.num_skipped_records == 3
    assert result.num_analyzed_records == 1
    assert result.total_response_tokens == 4
    assert result.total_repetitive_token_positions == 1
    assert result.repetitive_token_counts == {b_id: 1}


def test_build_token_compressibility_mask_marks_repeated_chunk_tokens():
    repeat_mask = mod.build_token_compressibility_mask(
        ["hello", "world", "hello", "world"],
        algorithm="zlib",
        level=9,
        span_tokens=2,
        context_tokens=8,
        min_savings_pct=50.0,
    )

    assert repeat_mask == [0, 0, 1, 1]


def test_analyze_records_supports_compressibility_detector():
    tokenizer = FakeWhitespaceTokenizer()
    result = mod.analyze_records(
        [{"student_response": "hello world hello world"}],
        tokenizer=tokenizer,
        response_field="student_response",
        detection_config=_compressibility_config(
            span_tokens=2,
            context_tokens=8,
            min_savings_pct=50.0,
        ),
    )

    hello_id = tokenizer.token_to_id["hello"]
    world_id = tokenizer.token_to_id["world"]
    assert result.total_response_tokens == 4
    assert result.total_repetitive_token_positions == 2
    assert result.repetitive_token_counts == {hello_id: 1, world_id: 1}


def test_build_token_display_map_strips_whitespace_markers_and_normalizes_collisions():
    tokenizer = FakeDisplayTokenizer(
        {
            1: " the",
            2: "\n",
            3: "dup",
            4: " dup",
        }
    )
    display_by_id, raw_text_by_id = mod.build_token_display_map(tokenizer, [1, 2, 3, 4])

    assert display_by_id[1] == "the"
    assert display_by_id[2] == "<empty>"
    assert display_by_id[3] == "dup"
    assert display_by_id[4] == "dup"
    assert raw_text_by_id[2] == "\n"


def test_filter_display_token_counts_excludes_noncontent_tokens():
    filtered_counts, filtered_stats = mod.filter_display_token_counts(
        {1: 7, 2: 5, 3: 11, 4: 13, 5: 17, 6: 2, 7: 3},
        {
            1: " ",
            2: "\n",
            3: "123",
            4: ",",
            5: "=",
            6: "word",
            7: "x1",
        },
    )

    assert filtered_counts == {6: 2, 7: 3}
    assert filtered_stats == {
        "whitespace": 12,
        "numeric": 11,
        "punctuation": 30,
    }


def test_aggregate_display_token_counts_merges_same_display_label():
    aggregated_counts, aggregated_meta = mod.aggregate_display_token_counts(
        {10: 7, 11: 5, 12: 3},
        {
            10: "the",
            11: "the",
            12: "word",
        },
        {
            10: " the",
            11: "the",
            12: "word",
        },
    )

    assert aggregated_counts == {"the": 12, "word": 3}
    assert aggregated_meta["the"]["token_ids"] == [10, 11]
    assert aggregated_meta["the"]["decoded_tokens"] == [" the", "the"]


def test_decode_ids_to_text_tokens_falls_back_when_decode_returns_replacement_char():
    tokenizer = FakeDisplayTokenizer({7: "\ufffd"}, {7: "A"})
    assert mod._decode_ids_to_text_tokens(tokenizer, [7], 1) == ["A"]


def test_main_smoke_writes_pdf_and_json(tmp_path, monkeypatch):
    input_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    input_path.write_text(
        json.dumps({"student_response": "a b c a b c"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_load_tokenizer", lambda source: FakeWhitespaceTokenizer())
    monkeypatch.setattr(mod, "_load_wordcloud_class", lambda: FakeWordCloud)

    exit_code = mod.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--repeat-detector",
            "ngram",
        ]
    )

    summary_path = output_dir / "eval_student_response_repeat_ngram3_summary.json"
    wordcloud_path = output_dir / "eval_student_response_repeat_ngram3_wordcloud.pdf"

    assert exit_code == 0
    assert summary_path.exists()
    assert wordcloud_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["response_field"] == "student_response"
    assert summary["repeat_detector"] == "ngram"
    assert summary["repeat_ngram"] == 3
    assert summary["num_records"] == 1
    assert summary["num_skipped_records"] == 0
    assert summary["total_response_tokens"] == 6
    assert summary["total_repetitive_token_positions"] == 1
    assert summary["displayed_repetitive_token_positions"] == 1
    assert summary["filtered_whitespace_repetitive_token_positions"] == 0
    assert summary["filtered_numeric_repetitive_token_positions"] == 0
    assert summary["filtered_punctuation_repetitive_token_positions"] == 0
    assert summary["filtered_noncontent_repetitive_token_positions"] == 0
    assert summary["top_repetitive_tokens"][0]["display_token"] == "c"
    assert summary["top_repetitive_tokens"][0]["token_ids"] == [3]
    assert summary["top_repetitive_tokens"][0]["decoded_tokens"] == ["c"]
    assert FakeWordCloud.last_frequencies == {"c": 1}


def test_main_supports_compressibility_outputs(tmp_path, monkeypatch):
    input_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    input_path.write_text(
        json.dumps({"student_response": "hello world hello world"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_load_tokenizer", lambda source: FakeWhitespaceTokenizer())
    monkeypatch.setattr(mod, "_load_wordcloud_class", lambda: FakeWordCloud)

    exit_code = mod.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--repeat-detector",
            "compressibility",
            "--compressibility-span-tokens",
            "2",
            "--compressibility-context-tokens",
            "8",
            "--compressibility-min-savings-pct",
            "50",
        ]
    )

    summary_path = output_dir / "eval_student_response_repeat_compressibility_zlib_span2_ctx8_s50p0_summary.json"
    wordcloud_path = output_dir / "eval_student_response_repeat_compressibility_zlib_span2_ctx8_s50p0_wordcloud.pdf"

    assert exit_code == 0
    assert summary_path.exists()
    assert wordcloud_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["repeat_detector"] == "compressibility"
    assert summary["repeat_ngram"] is None
    assert summary["compressibility_algorithm"] == "zlib"
    assert summary["compressibility_span_tokens"] == 2
    assert summary["compressibility_context_tokens"] == 8
    assert summary["compressibility_min_savings_pct"] == 50.0
    assert summary["total_repetitive_token_positions"] == 2
    assert FakeWordCloud.last_frequencies == {"hello": 1, "world": 1}


def test_main_supports_pt_input(tmp_path, monkeypatch):
    input_pt = tmp_path / "eval_0.pt"
    output_dir = tmp_path / "out"
    torch.save(
        {
            "samples": [
                {"response": "a b c a b c"},
                {"response": ""},
                {"response": None},
            ]
        },
        input_pt,
    )

    monkeypatch.setattr(mod, "_load_tokenizer", lambda source: FakeWhitespaceTokenizer())
    monkeypatch.setattr(mod, "_load_wordcloud_class", lambda: FakeWordCloud)

    exit_code = mod.main(
        [
            "--input-pt",
            str(input_pt),
            "--output-dir",
            str(output_dir),
            "--repeat-detector",
            "ngram",
        ]
    )

    summary_path = output_dir / "eval_0_student_response_repeat_ngram3_summary.json"
    wordcloud_path = output_dir / "eval_0_student_response_repeat_ngram3_wordcloud.pdf"

    assert exit_code == 0
    assert summary_path.exists()
    assert wordcloud_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_path"] == str(input_pt)
    assert summary["response_field"] == "student_response"
    assert summary["num_records"] == 3
    assert summary["num_analyzed_records"] == 1
    assert summary["num_skipped_records"] == 2
    assert summary["total_response_tokens"] == 6
    assert summary["total_repetitive_token_positions"] == 1
    assert FakeWordCloud.last_frequencies == {"c": 1}


def test_main_supports_pt_term_frequency_comparison(tmp_path, monkeypatch):
    primary_input_pt = tmp_path / "eval_59.pt"
    compare_input_pt = tmp_path / "eval_0.pt"
    output_dir = tmp_path / "out"
    torch.save(
        {
            "samples": [
                {"response": "Step one. But wait, let's do the first step."},
                {"response": "Second step, third step, but let us wait."},
            ]
        },
        primary_input_pt,
    )
    torch.save(
        {
            "samples": [
                {"response": "First try. Wait, but no extra step."},
                {"response": "Let them rest."},
            ]
        },
        compare_input_pt,
    )

    monkeypatch.setattr(mod, "_load_tokenizer", lambda source: FakeWhitespaceTokenizer())
    monkeypatch.setattr(mod, "_load_wordcloud_class", lambda: FakeWordCloud)

    exit_code = mod.main(
        [
            "--input-pt",
            str(primary_input_pt),
            "--compare-input-pt",
            str(compare_input_pt),
            "--compare-terms",
            "step",
            "first/second/third",
            "but",
            "wait",
            "let",
            "--output-dir",
            str(output_dir),
            "--repeat-detector",
            "ngram",
        ]
    )

    comparison_summary_path = (
        output_dir / "eval_59_vs_eval_0_student_response_term_frequency_comparison_summary.json"
    )
    comparison_pdf_path = output_dir / "eval_59_vs_eval_0_student_response_term_frequency_comparison.pdf"

    assert exit_code == 0
    assert comparison_summary_path.exists()
    assert comparison_pdf_path.exists()

    summary = json.loads(comparison_summary_path.read_text(encoding="utf-8"))
    assert summary["primary_input"]["path"] == str(primary_input_pt)
    assert summary["compare_input"]["path"] == str(compare_input_pt)
    assert summary["rate_unit"] == "per_1k_words"

    term_rows = {row["label"]: row for row in summary["terms"]}
    assert term_rows["step"]["primary_count"] == 4
    assert term_rows["step"]["compare_count"] == 1
    assert term_rows["first/second/third"]["primary_count"] == 3
    assert term_rows["first/second/third"]["compare_count"] == 1
    assert term_rows["but"]["primary_count"] == 2
    assert term_rows["but"]["compare_count"] == 1
    assert term_rows["wait"]["primary_count"] == 2
    assert term_rows["wait"]["compare_count"] == 1
    assert term_rows["let"]["primary_count"] == 2
    assert term_rows["let"]["compare_count"] == 1
