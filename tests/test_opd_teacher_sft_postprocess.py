from argparse import Namespace
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import slime.rollout.on_policy_distillation as opd
from slime.utils.types import Sample


class FakeMaskGenerator:
    def get_loss_mask(self, messages, tools=None):
        del messages, tools
        return [10, 11, 31, 32], [0, 0, 1, 0]

    def get_response_lengths(self, loss_masks):
        return [2]


def test_opd_teacher_sft_postprocess_appends_masked_sft_sample(monkeypatch):
    monkeypatch.setattr(opd, "_get_mask_generator", lambda args: FakeMaskGenerator())

    sample = Sample(
        prompt="prompt",
        tokens=[10, 11, 21, 22],
        response="student",
        response_length=2,
        label="42",
        metadata={"source": "unit"},
    )
    sample.reward = {
        "teacher_output": {
            "meta_info": {
                "input_token_logprobs": [
                    [0.0, 10],
                    [-0.1, 11],
                    [-0.3, 21],
                    [-0.4, 22],
                ],
            },
        },
        "teacher_input_len": 4,
        "teacher_response_start": 2,
        "teacher_sft_prompt_input_ids": [10, 11],
        "teacher_sft_output": {
            "text": "teacher",
            "meta_info": {
                "output_token_logprobs": [
                    [-0.05, 31],
                    [-0.06, 32],
                ],
            },
        },
        "accuracy_strict": 1.0,
        "accuracy": 1.0,
    }

    args = Namespace(
        use_opd=True,
        opd_type="sglang",
        opd_teacher_sft=True,
        opd_distill_max_response_len=-1,
        opd_kl_mode="token_reverse_kl",
        opd_token_stats=False,
        opd_token_stats_topk=20,
        reward_key="accuracy_strict",
        opd_zero_task_reward=True,
    )

    samples = [sample]
    raw_rewards, scalar_rewards = opd.post_process_rewards(args, samples)

    assert len(samples) == 2
    assert raw_rewards == [1.0, 0.0]
    assert scalar_rewards == [0.0, 0.0]
    assert [s.opd_distill_sample_mask for s in samples] == [1, 0]
    assert [s.opd_sft_sample_mask for s in samples] == [0, 1]

    sft_sample = samples[1]
    assert sft_sample.tokens == [10, 11, 31, 32]
    assert sft_sample.response == "teacher"
    assert sft_sample.response_length == 2
    assert sft_sample.loss_mask == [1, 0]
    assert torch.equal(sft_sample.teacher_log_probs, torch.zeros(2))
    assert torch.equal(sft_sample.teacher_logprob_mask, torch.zeros(2, dtype=torch.int))
