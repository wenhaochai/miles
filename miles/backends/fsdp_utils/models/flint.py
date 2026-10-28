"""Flint (Recursive Flint, model_type ``flint``) on the FSDP backend. The model definition is ``flint_hf`` from the
mas-rl repository (on PYTHONPATH; hf_utils.config registers it with transformers' Auto classes); this module supplies
the training kernels it leaves pluggable: fused MoE experts and the rollout-routing-replay hook on its expert
selection."""

import torch

from miles.kernels.moe.fused_experts import fused_experts
from miles.utils.replay_base import routing_replay_manager


def fused_flint_experts(experts, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``FlintSparseMoeBlock.experts_fn`` on miles' fused Triton experts: the checkpoint's per-expert weights stacked
    per call (as the qwen3_moe true-on-policy block does), Flint's renormalized sigmoid weights applied inside."""
    w13 = torch.stack([torch.cat([e.gate_proj.weight, e.up_proj.weight], dim=0) for e in experts])
    w2 = torch.stack([e.down_proj.weight for e in experts])
    return fused_experts(x.to(torch.bfloat16), w13, w2, weights, ids).to(x.dtype)


def install_flint_router_replay(router) -> None:
    """Rebind ``FlintTopKRouter.select`` (top-k over the sigmoid scores plus expert_bias) to the replay stream."""
    router.select = routing_replay_manager.get_topk_fn(
        lambda scores, k: torch.topk(scores, k, dim=-1, sorted=False)[1], return_probs=False
    )
