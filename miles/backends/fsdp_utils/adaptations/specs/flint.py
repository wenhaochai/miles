"""flint: the Recursive Flint MoE (sinks, a 513-token window, sigmoid routing, an fp32 LM head), modeled by
``flint_hf`` (registered in hf_utils.config). Trains with Transformer Engine's sink attention and grouped-GEMM experts
(miles' fused Triton experts import an SGLang module that 0.5.23 no longer has), scores responses through the model's
chunked head (``fused_head``: no [tokens, vocab] logits), and replays rollout routing at its top-k."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.routing_replay import RoutingReplayAdapter
from miles.backends.fsdp_utils.adaptations.weight_bridge import is_batched_experts_param, unfuse_batched_experts


def _install_replay(router):
    from miles.backends.fsdp_utils.models.flint import install_flint_router_replay

    install_flint_router_replay(router)


class FlintAdapter(ArchAdapter):
    model_types = frozenset({"flint"})
    fused_head = True
    routing_replay = RoutingReplayAdapter(name="flint", module_cls_name="FlintTopKRouter", install=_install_replay)

    def patch_model(self, model, args):
        from flint_hf.modeling_flint import FlintAttention, FlintSparseMoeBlock, grouped_experts

        for module in model.modules():
            if isinstance(module, FlintAttention):
                module.attn_impl = "te"
            elif isinstance(module, FlintSparseMoeBlock):
                module.experts_fn = grouped_experts

    def packing_kwargs(self, *, cu_seqlens, cu_seqlens_host, max_seqlen):
        return {"cu_seq_lens_q": cu_seqlens}

    def param_transform(self, name, param):
        # flint_hf stacks each layer's experts; SGLang loads the checkpoint's per-expert names
        return unfuse_batched_experts if is_batched_experts_param(name, param) else None
