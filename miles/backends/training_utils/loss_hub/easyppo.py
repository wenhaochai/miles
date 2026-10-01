"""EasyPPO (arXiv 2609.36802, github.com/EasyPPO/EasyPPO) critic and actor changes for miles PPO.

1. Actor-only overlong filtering: responses that hit the rollout length limit keep full weight for GAE, advantage
   whitening and the critic update, and are masked out of the actor update only (EasyPPO
   trainer/ppo/v1/trainer_base.py: mask_overlong_response_keys around update_actor). `apply_actor_overlong_mask`.
2. Noise-normalized critic regression: each response's clipped value loss is multiplied by an inverse return-std
   weight of its prompt group (`compute_prompt_variance_loss_weights`, verbatim from EasyPPO
   verl/trainer/ppo/core_algos.py), computed on the rollout side over the whole batch.
3. Critic mini-batches: --critic-global-batch-size (the critic trainer's own global batch size).
"""
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch


# verbatim: EasyPPO verl/trainer/ppo/core_algos.py (EasyPPO 85bde92)
def compute_prompt_variance_loss_weights(
    sequence_rewards: torch.Tensor,
    sample_uids: np.ndarray | list[Any],
    beta: float = 1.0,
    w_min: float = 0.02,
    valid_mask: Optional[torch.Tensor] = None,
    w_max: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sequence inverse-variance weights from same-prompt rollout rewards.

    The empirical variance uses the population convention (correction=0). Every rollout from the
    same prompt first receives ``1 / max(variance**beta, w_min)``. The weights are then divided by
    their mean across unique prompts so they preserve the unweighted critic loss scale. If ``w_max``
    is set, the normalized weights are finally clipped to that maximum without renormalizing.
    Rows excluded by ``valid_mask`` receive zero weight and do not affect prompt statistics.
    """
    if sequence_rewards.ndim != 1:
        raise ValueError(f"sequence_rewards must be one-dimensional, got {tuple(sequence_rewards.shape)}")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if w_min <= 0:
        raise ValueError(f"w_min must be positive, got {w_min}")
    if w_max is not None and w_max <= 0:
        raise ValueError(f"w_max must be positive or None, got {w_max}")

    sample_uids = np.asarray(sample_uids, dtype=object)
    if sample_uids.ndim != 1 or sample_uids.shape[0] != sequence_rewards.numel():
        raise ValueError(
            "sample_uids must be one-dimensional and match sequence_rewards: "
            f"got shape {sample_uids.shape} for {sequence_rewards.numel()} rewards"
        )

    if valid_mask is None:
        valid_mask = torch.ones_like(sequence_rewards, dtype=torch.bool)
    else:
        if valid_mask.ndim != 1 or valid_mask.numel() != sequence_rewards.numel():
            raise ValueError(
                "valid_mask must be one-dimensional and match sequence_rewards: "
                f"got shape {tuple(valid_mask.shape)} for {sequence_rewards.numel()} rewards"
            )
        valid_mask = valid_mask.to(device=sequence_rewards.device, dtype=torch.bool)

    weights = torch.zeros_like(sequence_rewards, dtype=torch.float32)
    variances = torch.zeros_like(sequence_rewards, dtype=torch.float32)
    uid_to_indices: dict[Any, list[int]] = defaultdict(list)
    for index, sample_uid in enumerate(sample_uids.tolist()):
        if valid_mask[index]:
            uid_to_indices[sample_uid].append(index)

    if not uid_to_indices:
        raise ValueError("sequence_rewards contains no valid rows")

    detached_rewards = sequence_rewards.detach().float()
    prompt_weights = []
    for indices in uid_to_indices.values():
        index_tensor = torch.tensor(indices, device=sequence_rewards.device, dtype=torch.long)
        prompt_rewards = detached_rewards[index_tensor]
        prompt_variance = torch.var(prompt_rewards, correction=0)
        variance_scale = torch.clamp(prompt_variance.pow(beta), min=w_min)
        prompt_weight = variance_scale.reciprocal()
        prompt_weights.append(prompt_weight)
        weights[index_tensor] = prompt_weight
        variances[index_tensor] = prompt_variance

    weights /= torch.stack(prompt_weights).mean()
    if w_max is not None:
        weights.clamp_(max=w_max)
    return weights, variances


def overlong_flags(args, rollout_data) -> list[bool]:
    """EasyPPO's criterion (response length reaches the rollout limit) or SGLang's finish_reason=length."""
    limit = args.rollout_max_response_len
    truncated = rollout_data.get("truncated")
    flags = []
    for i, response_length in enumerate(rollout_data["response_lengths"]):
        hit = limit is not None and int(response_length) >= limit
        if truncated is not None:
            hit = hit or bool(int(truncated[i]))
        flags.append(hit)
    return flags


def apply_actor_overlong_mask(args, rollout_data) -> int:
    """Zero the actor loss masks of overlong responses in place (after advantages are computed); returns the count."""
    flags = overlong_flags(args, rollout_data)
    loss_masks = rollout_data["loss_masks"]
    for i, hit in enumerate(flags):
        if hit:
            loss_masks[i] = torch.zeros_like(loss_masks[i])
    return sum(flags)
