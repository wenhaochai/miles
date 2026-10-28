"""Flint's fused head (``ArchAdapter.fused_head``) against miles' logits path, on a tiny random flint_hf model: the
score-centering loss and its parameter gradients, and the no-grad log-probs and entropy, computed once from full
logits and once from the rows and ids the FSDP actor's ``_head_kwargs`` asks the model to score. Needs flint_hf (the
mas-rl repository) on PYTHONPATH; skipped without it."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

flint_hf = pytest.importorskip("flint_hf")

from flint_hf import FlintConfig, FlintForCausalLM  # noqa: E402

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor  # noqa: E402
from miles.backends.training_utils.data import context_parallel  # noqa: E402
from miles.backends.training_utils.loss.hub import logit_processors, score_centering_loss  # noqa: E402

K = 4


def _model() -> FlintForCausalLM:
    cfg = FlintConfig(vocab_size=31, hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=4, moe_intermediate_size=8, shared_expert_intermediate_size=12,
                      num_experts=4, num_experts_per_tok=2, sliding_window=3,
                      layer_types=["sliding_attention", "full_attention"], no_rope_layer_indices=[1],
                      mlp_only_layers=[0])
    torch.manual_seed(0)
    model = FlintForCausalLM(cfg).float()   # miles' loss path takes fp32 logits
    for module in model.modules():
        if hasattr(module, "softmax_offset"):
            module.softmax_offset.data.normal_()
    return model


def _batch() -> dict:
    """Two packed samples (prompt 3 + response 4, prompt 2 + response 5) and 2 padding tokens."""
    g = torch.Generator().manual_seed(1)
    samples = [torch.randint(0, 31, (7,), generator=g), torch.randint(0, 31, (7,), generator=g)]
    responses = [4, 5]
    topk = [torch.randint(0, 31, (r, K), generator=g) for r in responses]
    topk[0][1, 2:] = -1
    return {
        "tokens": torch.cat(samples + [torch.zeros(2, dtype=torch.long)])[None],
        "position_ids": torch.tensor([list(range(7)) + list(range(7)) + [0, 1]]),
        "unconcat_tokens": samples,
        "total_lengths": [7, 7],
        "response_lengths": responses,
        "loss_masks": [torch.tensor([1, 1, 0, 1]), torch.tensor([1, 0, 1, 1, 1])],
        "rollout_log_probs": [-torch.rand(r, generator=g) for r in responses],
        "advantages": [torch.full((4,), 0.7), torch.full((5,), -0.4)],
        "rollout_topk_token_ids": [t.int() for t in topk],
        "rollout_topk_log_probs": [torch.where(t >= 0, -torch.rand(t.shape, generator=g), -torch.inf).float() for t in topk],
    }


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    state = SimpleNamespace(tp=SimpleNamespace(size=1, group=None, rank=0), cp=SimpleNamespace(size=1, rank=0))
    for module in (context_parallel, logit_processors, score_centering_loss):
        monkeypatch.setattr(module, "get_parallel_state", lambda: state, raising=False)


def _args(fused: bool) -> Namespace:
    return Namespace(fused_head_log_probs=fused, qkv_format="thd", allgather_cp=False, true_on_policy_mode=False,
                     rollout_temperature=0.9, log_probs_chunk_size=-1, vocab_size=31, use_kl_loss=False,
                     entropy_coef=0, observe_training_entropy=False, use_sampling_support_replay=False,
                     score_centering_is="none", score_centering_tis_clip=2.0, score_centering_mis_low=0.0,
                     score_centering_mis_high=2.0, log_probs_backend="torch", loss_type="score_centering")


def _forward(model, batch, fused: bool):
    kwargs = {}
    if fused:
        kwargs = FSDPTrainRayActor._head_kwargs(SimpleNamespace(args=_args(True)), batch, batch["tokens"].size(-1))
    return model(batch["tokens"], batch["position_ids"], None, **kwargs).logits


def test_score_centering_loss_and_gradients_match_full_logits():
    batch = _batch()
    results = []
    for fused in (False, True):
        model = _model()
        logits = _forward(model, batch, fused)
        loss, log = score_centering_loss.score_centering_loss_function(_args(fused), batch, logits, torch.sum)
        loss.backward()
        results.append((loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (loss_ref, grads_ref), (loss_fused, grads_fused) = results
    torch.testing.assert_close(loss_fused, loss_ref, rtol=1e-5, atol=1e-6)
    assert grads_ref.keys() == grads_fused.keys()
    for name in grads_ref:
        torch.testing.assert_close(grads_fused[name], grads_ref[name], rtol=1e-4, atol=1e-6, msg=name)


def test_no_grad_log_probs_and_entropy_match_full_logits():
    batch = _batch()
    model = _model()
    out = []
    with torch.no_grad():
        for fused in (False, True):
            logits = _forward(model, batch, fused)
            out.append(logit_processors.get_log_probs_and_entropy(
                logits, args=_args(fused), unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"], with_entropy=True))
    for key in ("log_probs", "entropy"):
        for ref, fused in zip(out[0][key], out[1][key], strict=True):
            torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-6)
