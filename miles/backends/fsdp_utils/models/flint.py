"""Flint (Recursive Flint, model_type ``flint``) on the FSDP backend. The model definition, including its training
kernels, is ``flint_hf`` from the mas-rl repository (on PYTHONPATH; hf_utils.config registers it with transformers'
Auto classes); this module adds the rollout-routing-replay hook on its expert selection."""

import torch

from miles.utils.replay_base import routing_replay_manager


def install_flint_router_replay(router) -> None:
    """Rebind ``FlintTopKRouter.select`` (top-k over the sigmoid scores plus expert_bias) to the replay stream."""
    router.select = routing_replay_manager.get_topk_fn(
        lambda scores, k: torch.topk(scores, k, dim=-1, sorted=False)[1], return_probs=False
    )
