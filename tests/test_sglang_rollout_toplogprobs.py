import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.modules.setdefault("sglang_router", types.ModuleType("sglang_router"))

from slime.rollout.sglang_rollout import (  # noqa: E402
    _get_output_top_logprobs_entry,
    _parse_top_logprobs_entry,
)


def test_get_output_top_logprobs_entry_falls_back_to_output_top_logprobs_sidecar():
    meta_info = {
        "output_top_logprobs": [
            {"10": -0.1, "20": -0.5, "30": -0.9},
            {"11": -0.2, "21": -0.4, "31": -0.8},
        ]
    }
    output_item = [-0.1, 10, None]

    entry = _get_output_top_logprobs_entry(meta_info, output_item, 0)
    token_ids, logprobs = _parse_top_logprobs_entry(entry, 2)

    assert token_ids == [10, 20]
    assert logprobs == [-0.1, -0.5]


def test_get_output_top_logprobs_entry_falls_back_to_output_token_top_logprobs_sidecar():
    meta_info = {
        "output_token_top_logprobs": [
            [[-0.1, 10], [-0.3, 20], [-0.7, 30]],
        ]
    }
    output_item = [-0.1, 10, None]

    entry = _get_output_top_logprobs_entry(meta_info, output_item, 0)
    token_ids, logprobs = _parse_top_logprobs_entry(entry, 2)

    assert token_ids == [10, 20]
    assert logprobs == [-0.1, -0.3]
