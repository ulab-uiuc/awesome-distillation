import torch


def normalize_grpo_group_rewards(raw_rewards: list[float], n_samples_per_prompt: int) -> list[float]:
    if n_samples_per_prompt <= 1 or not raw_rewards:
        return [float(x) for x in raw_rewards]
    if len(raw_rewards) % n_samples_per_prompt != 0:
        raise ValueError(
            "GRPO reward normalization requires the reward count to be divisible by "
            f"n_samples_per_prompt. Got {len(raw_rewards)} and {n_samples_per_prompt}."
        )

    rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    rewards = rewards.view(-1, n_samples_per_prompt)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    normalized = (rewards - mean) / (std + 1e-6)
    normalized[std.squeeze(-1) < 1e-8] = 0.0
    return normalized.flatten().tolist()


def compute_opsd_grpo_r2_from_gap_means(gap_means: list[float], n_samples_per_prompt: int) -> list[float]:
    if not gap_means:
        return []
    if n_samples_per_prompt <= 1:
        return [0.0] * len(gap_means)
    if len(gap_means) % n_samples_per_prompt != 0:
        raise ValueError(
            "OPSD-GRPO R2 requires the gap count to be divisible by n_samples_per_prompt. "
            f"Got {len(gap_means)} and {n_samples_per_prompt}."
        )

    gaps = torch.tensor(gap_means, dtype=torch.float32).view(-1, n_samples_per_prompt)
    gap_min = gaps.min(dim=-1, keepdim=True).values
    gap_max = gaps.max(dim=-1, keepdim=True).values
    denom = gap_max - gap_min

    r2 = torch.zeros_like(gaps)
    non_degenerate = denom.squeeze(-1) > 1e-12
    if non_degenerate.any():
        r2[non_degenerate] = (gaps[non_degenerate] - gap_min[non_degenerate]) / denom[non_degenerate]

    return r2.flatten().tolist()


def compute_opsd_grpo_r2_alpha_terms(
    raw_rewards: list[float],
    r2_values: list[float],
    *,
    alpha: float,
    sign_mode: str,
) -> list[float]:
    if len(raw_rewards) != len(r2_values):
        raise ValueError(
            "OPSD-GRPO R2 alpha term requires raw_rewards and r2_values to have the same length. "
            f"Got {len(raw_rewards)} and {len(r2_values)}."
        )

    sign_mode = str(sign_mode).lower()
    alpha_terms: list[float] = []
    for raw_reward, r2 in zip(raw_rewards, r2_values, strict=False):
        alpha_term = float(alpha) * float(r2)
        if sign_mode == "none":
            pass
        elif sign_mode == "flip_on_reward0":
            if float(raw_reward) == 0.0:
                alpha_term = -alpha_term
        else:
            raise ValueError(
                "Unsupported OPSD-GRPO R2 sign mode: "
                f"{sign_mode!r}. Expected one of ['none', 'flip_on_reward0']."
            )
        alpha_terms.append(alpha_term)

    return alpha_terms


def rebuild_full_metric_from_shards(
    total_samples: int,
    gathered_batch_indices: list[list[int]],
    gathered_metric_values: list[list[float]],
    *,
    metric_name: str,
) -> list[float]:
    if len(gathered_batch_indices) != len(gathered_metric_values):
        raise ValueError(
            f"{metric_name}: shard count mismatch {len(gathered_batch_indices)} vs {len(gathered_metric_values)}."
        )

    full_values: list[float | None] = [None] * total_samples
    for shard_id, (batch_indices, metric_values) in enumerate(
        zip(gathered_batch_indices, gathered_metric_values, strict=False)
    ):
        if len(batch_indices) != len(metric_values):
            raise ValueError(
                f"{metric_name}: batch-index/value count mismatch on shard {shard_id}: "
                f"{len(batch_indices)} vs {len(metric_values)}."
            )
        for batch_idx, value in zip(batch_indices, metric_values, strict=False):
            batch_idx = int(batch_idx)
            value = float(value)
            if batch_idx < 0 or batch_idx >= total_samples:
                raise ValueError(
                    f"{metric_name}: batch index {batch_idx} is out of range for total_samples={total_samples}."
                )
            if full_values[batch_idx] is not None:
                raise ValueError(f"{metric_name}: duplicate batch index {batch_idx} encountered while rebuilding.")
            full_values[batch_idx] = value

    missing = [idx for idx, value in enumerate(full_values) if value is None]
    if missing:
        raise ValueError(f"{metric_name}: missing batch indices while rebuilding full order: {missing[:8]}.")

    return [float(value) for value in full_values]


def recompute_local_opsd_grpo_r2_rewards(
    *,
    raw_rewards_full: list[float],
    local_batch_indices: list[int],
    gathered_batch_indices: list[list[int]],
    gathered_gap_means: list[list[float]],
    alpha: float,
    sign_mode: str = "flip_on_reward0",
    n_samples_per_prompt: int,
) -> dict[str, list[float]]:
    total_samples = len(raw_rewards_full)
    full_gap_means = rebuild_full_metric_from_shards(
        total_samples,
        gathered_batch_indices,
        gathered_gap_means,
        metric_name="opsd_grpo_r2_gap_mean",
    )
    full_r2 = compute_opsd_grpo_r2_from_gap_means(full_gap_means, n_samples_per_prompt)
    full_alpha_term = compute_opsd_grpo_r2_alpha_terms(
        raw_rewards_full,
        full_r2,
        alpha=alpha,
        sign_mode=sign_mode,
    )
    full_combined_raw = [
        float(raw_reward) + alpha_term
        for raw_reward, alpha_term in zip(raw_rewards_full, full_alpha_term, strict=False)
    ]
    full_normalized_rewards = normalize_grpo_group_rewards(full_combined_raw, n_samples_per_prompt)

    local_rewards = [full_normalized_rewards[int(batch_idx)] for batch_idx in local_batch_indices]
    local_r2 = [full_r2[int(batch_idx)] for batch_idx in local_batch_indices]
    local_alpha_term = [full_alpha_term[int(batch_idx)] for batch_idx in local_batch_indices]
    local_combined_raw = [full_combined_raw[int(batch_idx)] for batch_idx in local_batch_indices]

    return {
        "rewards": local_rewards,
        "opsd_reward_r2": local_r2,
        "opsd_reward_alpha_term": local_alpha_term,
        "opsd_reward_combined_raw": local_combined_raw,
    }
