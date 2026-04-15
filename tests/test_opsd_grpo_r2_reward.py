import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from slime.utils.opsd_grpo_reward import (
    compute_opsd_grpo_r2_from_gap_means,
    recompute_local_opsd_grpo_r2_rewards,
)


def test_compute_opsd_grpo_r2_from_gap_means_minmax_and_ties():
    assert compute_opsd_grpo_r2_from_gap_means([0.2, 0.6, -0.5, 0.5], 2) == [0.0, 1.0, 0.0, 1.0]
    assert compute_opsd_grpo_r2_from_gap_means([0.3, 0.3, 1.2, 1.2], 2) == [0.0, 0.0, 0.0, 0.0]


def test_recompute_local_opsd_grpo_r2_rewards_default_flips_sign_on_reward0():
    result = recompute_local_opsd_grpo_r2_rewards(
        raw_rewards_full=[1.0, 0.0, 1.0, 0.0],
        local_batch_indices=[0, 3],
        gathered_batch_indices=[[0, 3], [1, 2]],
        gathered_gap_means=[[0.2, 0.5], [0.4, 0.1]],
        alpha=0.5,
        n_samples_per_prompt=2,
    )

    assert result["opsd_reward_r2"] == [0.0, 1.0]
    assert result["opsd_reward_alpha_term"] == [0.0, -0.5]
    assert result["opsd_reward_combined_raw"] == [1.0, -0.5]
    assert result["rewards"] == pytest.approx([0.70710617, -0.70710617], abs=1e-6)


def test_recompute_local_opsd_grpo_r2_rewards_sign_mode_none_keeps_legacy_bonus():
    result = recompute_local_opsd_grpo_r2_rewards(
        raw_rewards_full=[1.0, 0.0, 1.0, 0.0],
        local_batch_indices=[0, 3],
        gathered_batch_indices=[[0, 3], [1, 2]],
        gathered_gap_means=[[0.2, 0.5], [0.4, 0.1]],
        alpha=0.5,
        sign_mode="none",
        n_samples_per_prompt=2,
    )

    assert result["opsd_reward_r2"] == [0.0, 1.0]
    assert result["opsd_reward_alpha_term"] == [0.0, 0.5]
    assert result["opsd_reward_combined_raw"] == [1.0, 0.5]
    assert result["rewards"] == pytest.approx([0.70710474, -0.70710474], abs=1e-6)


def test_recompute_local_opsd_grpo_r2_rewards_alpha_zero_matches_plain_grpo_reward():
    result = recompute_local_opsd_grpo_r2_rewards(
        raw_rewards_full=[1.0, 0.0],
        local_batch_indices=[0, 1],
        gathered_batch_indices=[[0, 1]],
        gathered_gap_means=[[0.1, 0.9]],
        alpha=0.0,
        n_samples_per_prompt=2,
    )

    assert result["opsd_reward_alpha_term"] == [0.0, 0.0]
    assert result["opsd_reward_combined_raw"] == [1.0, 0.0]
    assert result["rewards"] == pytest.approx([0.707105, -0.707105], abs=1e-6)


def test_recompute_local_opsd_grpo_r2_rewards_mixed_group_changes_final_normalized_rewards():
    result = recompute_local_opsd_grpo_r2_rewards(
        raw_rewards_full=[1.0, 0.0, 1.0, 0.0],
        local_batch_indices=[0, 1, 2, 3],
        gathered_batch_indices=[[0, 1, 2, 3]],
        gathered_gap_means=[[0.2, 0.4, 0.6, 0.8]],
        alpha=0.5,
        n_samples_per_prompt=4,
    )

    assert result["opsd_reward_r2"] == pytest.approx([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], abs=1e-6)
    assert result["opsd_reward_alpha_term"] == pytest.approx([0.0, -1.0 / 6.0, 1.0 / 3.0, -0.5], abs=1e-6)
    assert result["opsd_reward_combined_raw"] == pytest.approx([1.0, -1.0 / 6.0, 4.0 / 3.0, -0.5], abs=1e-6)
    assert result["rewards"] == pytest.approx(
        [0.65753472, -0.65753478, 1.03326893, -1.03326893],
        abs=1e-6,
    )


def test_recompute_local_opsd_grpo_r2_rewards_handles_later_rollout_step_global_ids():
    global_sample_indices = list(range(128, 136))
    assert min(global_sample_indices) >= 128

    result = recompute_local_opsd_grpo_r2_rewards(
        raw_rewards_full=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        local_batch_indices=[0, 3, 5, 6],
        gathered_batch_indices=[[0, 3, 5, 6], [1, 2, 4, 7]],
        gathered_gap_means=[[0.2, 0.9, 0.6, 0.3], [0.5, 0.1, 0.8, 0.4]],
        alpha=0.5,
        n_samples_per_prompt=2,
    )

    assert result["opsd_reward_r2"] == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-6)
    assert result["opsd_reward_alpha_term"] == pytest.approx([0.0, -0.5, 0.0, 0.0], abs=1e-6)
    assert result["opsd_reward_combined_raw"] == pytest.approx([1.0, -0.5, 0.0, 1.0], abs=1e-6)
    assert result["rewards"] == pytest.approx(
        [0.70710617, -0.70710617, -0.70710617, 0.70710617],
        abs=1e-6,
    )


def test_recompute_local_opsd_grpo_r2_rewards_rebuilds_from_non_contiguous_dp_shards():
    result = recompute_local_opsd_grpo_r2_rewards(
        raw_rewards_full=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        local_batch_indices=[5, 0, 7, 3],
        gathered_batch_indices=[[4, 2, 5, 7], [1, 6, 0, 3]],
        gathered_gap_means=[[0.8, 0.1, 0.6, 0.4], [0.5, 0.3, 0.2, 0.9]],
        alpha=0.5,
        n_samples_per_prompt=2,
    )

    assert result["opsd_reward_r2"] == pytest.approx([0.0, 0.0, 1.0, 1.0], abs=1e-6)
    assert result["opsd_reward_alpha_term"] == pytest.approx([0.0, 0.0, -0.5, -0.5], abs=1e-6)
    assert result["opsd_reward_combined_raw"] == pytest.approx([0.0, 1.0, -0.5, -0.5], abs=1e-6)
    assert result["rewards"] == pytest.approx(
        [-0.70710617, 0.70710617, -0.70710617, -0.70710617],
        abs=1e-6,
    )
