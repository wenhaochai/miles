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


# Sampling-support replay (rollouts at top_k <= K): the trainer's policy is the sampler's support, renormalized. The
# fused head scores the support ids and renormalizes over them; the reference is miles' full-logits path with the mask.


def _support_batch() -> dict:
    """_batch() with a sampling mask: each response row's support (1 to K ids, its sampled token among them), which
    the rollout's top-K candidates equal (as after train_data_conversion's compaction, with dense ids kept)."""
    batch = _batch()
    g = torch.Generator().manual_seed(2)
    ids, offsets, topk, topk_lp = [], [], [], []
    layout = context_parallel.iter_local_response_rows(
        batch["tokens"].size(-1), unconcat_tokens=batch["unconcat_tokens"], total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"], qkv_format="thd", allgather_cp=False, max_seq_lens=None,
        include_response_indices=False)
    for (_ranges, sampled, _indices), r in zip(layout, batch["response_lengths"], strict=True):
        rows = []
        for t in sampled.tolist():   # each row's sampled token, as the loss reads it
            others = [x for x in torch.randperm(31, generator=g).tolist() if x != t]
            rows.append([t] + others[: int(torch.randint(0, K, (1,), generator=g))])
        flat = [x for row in rows for x in row]
        ids.append(torch.tensor(flat, dtype=torch.int32))
        offsets.append(torch.tensor([0] + torch.tensor([len(row) for row in rows]).cumsum(0).tolist()))
        dense = torch.full((r, K), -1, dtype=torch.int32)
        for j, row in enumerate(rows):
            dense[j, : len(row)] = torch.tensor(row)
        topk.append(dense)
        topk_lp.append(torch.where(dense >= 0, -torch.rand(dense.shape, generator=g), -torch.inf).float())
    batch.update(rollout_sampling_mask_ids=ids, rollout_sampling_mask_offsets=offsets, rollout_topk_token_ids=topk,
                 rollout_topk_log_probs=topk_lp)
    return batch


def _support_args(fused: bool) -> Namespace:
    return Namespace(**{**vars(_args(fused)), "use_sampling_support_replay": True, "rollout_top_k": K,
                        "debug_unified_grad_fused_logprob": False})


def test_support_score_centering_loss_and_gradients_match_full_logits():
    batch = _support_batch()
    results = []
    for fused in (False, True):
        model = _model()
        kwargs = {}
        if fused:
            kwargs = FSDPTrainRayActor._head_kwargs(SimpleNamespace(args=_support_args(True)), batch,
                                                    batch["tokens"].size(-1))
        logits = model(batch["tokens"], batch["position_ids"], None, **kwargs).logits
        loss, log = score_centering_loss.score_centering_loss_function(_support_args(fused), batch, logits, torch.sum)
        loss.backward()
        results.append((loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (loss_ref, grads_ref), (loss_fused, grads_fused) = results
    torch.testing.assert_close(loss_fused, loss_ref, rtol=1e-5, atol=1e-6)
    assert grads_ref.keys() == grads_fused.keys()
    for name in grads_ref:
        torch.testing.assert_close(grads_fused[name], grads_ref[name], rtol=1e-4, atol=1e-6, msg=name)


def _masked_reference(logits, batch, args):
    """miles' full-logits masked path (logit_processors._torch_log_probs_and_entropy -> compute_log_probs with the
    sampling mask): the logits outside the support masked out, then the log-softmax at the sampled token and the
    masked distribution's entropy. Written out here, since that path's fused cross entropy needs CUDA."""
    from miles.backends.training_utils.data.sampling_mask import build_local_sampling_mask, get_rollout_sampling_masks

    masks = get_rollout_sampling_masks(batch)
    out = {"log_probs": [], "entropy": []}
    for i, (chunk, tokens, indices) in enumerate(logit_processors._iter_response_chunks(
            logits, args=args, unconcat_tokens=batch["unconcat_tokens"], total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"], include_response_indices=True)):
        mask = build_local_sampling_mask(chunk, masks[i], indices, tp_rank=0)
        lp = torch.log_softmax(chunk.masked_fill(~mask, -torch.inf) / args.rollout_temperature, dim=-1)
        out["log_probs"].append(lp.gather(-1, tokens[:, None]).squeeze(-1))
        out["entropy"].append(-(lp.exp() * torch.where(mask, lp, 0.0)).sum(-1))
    return out


def test_support_no_grad_log_probs_and_entropy_match_full_logits():
    """The log-prob pass fetches no candidates (FORWARD_ONLY_KEYS + the mask): the fused head scores each row's
    support from its sampling mask."""
    from miles.backends.training_utils.data.sampling_mask import get_rollout_sampling_masks

    batch = {k: v for k, v in _support_batch().items() if not k.startswith("rollout_topk")}
    model = _model()
    with torch.no_grad():
        ref = _masked_reference(_forward(model, batch, False), batch, _support_args(False))
        kwargs = FSDPTrainRayActor._head_kwargs(SimpleNamespace(args=_support_args(True)), batch, batch["tokens"].size(-1))
        assert kwargs["head_ids"].size(1) == 1 + K
        logits = model(batch["tokens"], batch["position_ids"], None, **kwargs).logits
        fused = logit_processors.get_log_probs_and_entropy(
            logits, args=_support_args(True), unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"], with_entropy=True,
            rollout_sampling_mask=get_rollout_sampling_masks(batch))
        plain = logit_processors.get_log_probs_and_entropy(
            _forward(model, batch, False), args=_args(False), unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"], with_entropy=True)
    for key in ("log_probs", "entropy"):
        for r, f in zip(ref[key], fused[key], strict=True):
            torch.testing.assert_close(f, r, rtol=1e-5, atol=1e-6)
    # the support's log-probs differ from the full vocabulary's: the test sees the renormalization
    assert not torch.allclose(plain["log_probs"][0], ref["log_probs"][0])
