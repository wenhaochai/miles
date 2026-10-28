"""Cross-check the score-centering estimator against the labs-molt reference.

The reference is ``score_centering_correction`` from labs-molt branch feat/score-centering (74cb05d,
molt/models/loss.py), copied below with its logging metrics dropped. labs-molt applies it as
``coef * (-A * ratio) + A * correction`` with ``ratio == 1`` on-policy and ``coef`` the sampled token's
importance weight, so the two implementations must give the same gradient.
"""

import math

import pytest
import torch

from miles.backends.training_utils.loss.hub.score_centering import ScoreCenteringInputs, score_centering_loss


def molt_score_centering_correction(head_log_probs, rollout_topk_logprobs, head_mask=None, weight_fn=None):
    train_lp = head_log_probs.float()
    samp_lp = rollout_topk_logprobs.float()
    hm = torch.ones_like(samp_lp) if head_mask is None else head_mask.float()
    q = samp_lp.exp() * hm
    p = train_lp.detach().exp() * hm
    tail_q = (1.0 - q.sum(dim=-1)).clamp(min=1e-6)
    tail_p = (1.0 - p.sum(dim=-1)).clamp(min=1e-6)
    rho = (tail_q / tail_p).unsqueeze(-1)
    if weight_fn is None:
        qw = q
        alpha = rho
    else:
        r = (train_lp.detach() - samp_lp).exp()  # r_v = p_v / q_v on the head
        qw = q * weight_fn(r) * hm
        alpha = rho * weight_fn(1.0 / rho)
    residual = ((qw - alpha * p) * hm).detach()
    return (residual * train_lp).sum(dim=-1)


WEIGHT_FN = {
    "none": None,
    "tis": lambda r: r.clamp(max=2.0),
    "mis": lambda r: torch.where((r >= 0.5) & (r <= 5.0), r, torch.zeros_like(r)),
}


def _drifted_case(seed: int, tokens: int = 6, vocab: int = 48, k: int = 8, drift: float = 0.4):
    """Trainer logits with grad, a drifted sampler, its top-k head, sampled tokens and advantages."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, vocab, generator=g).requires_grad_(True)
    sampler = torch.log_softmax(logits.detach() + drift * torch.randn(tokens, vocab, generator=g), -1)
    head_ids = sampler.topk(k, dim=-1).indices
    sampled = torch.multinomial(sampler.exp(), 1, generator=g).squeeze(-1)
    advantages = torch.randn(tokens, generator=g)
    advantages[0] = 0.0  # an inactive token
    head_mask = torch.ones(tokens, k, dtype=torch.bool)
    head_mask[1, k // 2 :] = False  # a short head (fewer candidates returned)
    return logits, sampler, head_ids, head_mask, sampled, advantages


def _miles_gradient(logits, sampler, head_ids, head_mask, sampled, advantages, mode):
    train = torch.log_softmax(logits, -1)
    rows = torch.arange(len(sampled))
    loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=train[rows, sampled],
            train_head_log_probs=train.gather(-1, head_ids),
            rollout_log_probs=sampler[rows, sampled],
            rollout_head_log_probs=sampler.gather(-1, head_ids),
            head_mask=head_mask,
            advantages=advantages,
            mode=mode,
        )
    )
    (grad,) = torch.autograd.grad(loss.sum(), logits)
    return grad, metrics


def _molt_gradient(logits, sampler, head_ids, head_mask, sampled, advantages, mode):
    train = torch.log_softmax(logits, -1)
    rows = torch.arange(len(sampled))
    weight_fn = WEIGHT_FN[mode]
    log_p = train[rows, sampled]
    ratio = (log_p - log_p.detach()).exp()  # == 1 on-policy, gradient d log p
    coef = torch.ones_like(log_p) if weight_fn is None else weight_fn((log_p - sampler[rows, sampled]).exp()).detach()
    correction = molt_score_centering_correction(
        train.gather(-1, head_ids), sampler.gather(-1, head_ids), head_mask=head_mask, weight_fn=weight_fn
    )
    loss = coef * (-advantages * ratio) + advantages * correction
    (grad,) = torch.autograd.grad(loss.sum(), logits)
    return grad, correction.detach()


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_and_correction_match_molt(mode: str, seed: int) -> None:
    case = _drifted_case(seed)
    ours, metrics = _miles_gradient(*case, mode)
    reference, correction = _molt_gradient(*case, mode)
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-6)
    active = case[-1] != 0
    torch.testing.assert_close(metrics.correction[active], correction[active], rtol=1e-5, atol=1e-6)


def test_short_and_empty_heads_match_molt() -> None:
    logits, sampler, head_ids, head_mask, sampled, advantages = _drifted_case(3)
    head_mask[2] = False  # no candidate at all: no correction on this token
    ours, metrics = _miles_gradient(logits, sampler, head_ids, head_mask, sampled, advantages, "none")
    reference, correction = _molt_gradient(logits, sampler, head_ids, head_mask, sampled, advantages, "none")
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-6)
    assert metrics.correction[2] == 0 and correction[2] == 0


@pytest.mark.parametrize("head_mass", [1e-3, 0.5, 1 - 1e-7])
def test_extreme_tail_masses_stay_finite_and_match_molt(head_mass: float) -> None:
    """Tails near 1 (a flat head) and near 0 (a peaked head, the 1e-6 floor) behave like labs-molt."""
    tokens, vocab, k = 4, 64, 4
    g = torch.Generator().manual_seed(7)
    head = torch.log(torch.full((tokens, k), head_mass / k))
    tail = torch.log(torch.full((tokens, vocab - k), (1 - head_mass) / (vocab - k)))
    base = torch.cat((head, tail), -1)
    logits = (base + 0.05 * torch.randn(tokens, vocab, generator=g)).requires_grad_(True)
    sampler = torch.log_softmax(base + 0.05 * torch.randn(tokens, vocab, generator=g), -1)
    head_ids = torch.arange(k).expand(tokens, k)
    head_mask = torch.ones(tokens, k, dtype=torch.bool)
    sampled = torch.zeros(tokens, dtype=torch.long)
    advantages = torch.ones(tokens)
    ours, _ = _miles_gradient(logits, sampler, head_ids, head_mask, sampled, advantages, "none")
    reference, _ = _molt_gradient(logits, sampler, head_ids, head_mask, sampled, advantages, "none")
    assert torch.isfinite(ours).all()
    torch.testing.assert_close(ours, reference, rtol=1e-4, atol=1e-6)


def _expected_gradient(logits, sampler, head_ids, head_mask, centered: bool) -> torch.Tensor:
    """d/dlogits of E_{y~q}[-(log p_y - correction)] with advantage 1, by enumerating every y."""
    tokens, vocab = logits.shape
    train = torch.log_softmax(logits, -1)
    q = sampler.exp()
    total = logits.new_zeros(())
    for y in range(vocab):
        sampled = torch.full((tokens,), y, dtype=torch.long)
        rows = torch.arange(tokens)
        if centered:
            loss, _ = score_centering_loss(
                ScoreCenteringInputs(
                    train_log_probs=train[rows, sampled],
                    train_head_log_probs=train.gather(-1, head_ids),
                    rollout_log_probs=sampler[rows, sampled],
                    rollout_head_log_probs=sampler.gather(-1, head_ids),
                    head_mask=head_mask,
                    advantages=torch.ones(tokens),
                )
            )
        else:
            loss = -train[rows, sampled]
        total = total + (q[:, y] * loss).sum()
    (grad,) = torch.autograd.grad(total, logits)
    return grad


@pytest.mark.parametrize("k", [16, 4])
def test_centering_removes_the_expected_score_drift(k: int) -> None:
    """Under a sampler q != p the plain score has E_q[grad log p] != 0, a drift with no reward signal.

    With the full vocabulary as the head (k = vocab), or a sampler tail proportional to the trainer
    tail (the estimator's tail model), the centered score has expectation zero exactly.
    """
    tokens, vocab = 3, 16
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(tokens, vocab, generator=g).requires_grad_(True)
    p = torch.softmax(logits.detach(), -1)
    head_ids = p.topk(k, dim=-1).indices
    in_head = torch.zeros(tokens, vocab, dtype=torch.bool).scatter_(-1, head_ids, True)
    # q: a different head; on the tail, p rescaled to the remaining mass (exactly the modeled tail)
    q_head = torch.softmax(torch.randn(tokens, k, generator=g), -1) * (0.9 if k < vocab else 1.0)
    q = torch.zeros(tokens, vocab).scatter_(-1, head_ids, q_head)
    if k < vocab:
        p_tail = torch.where(in_head, 0.0, p)
        q = q + 0.1 * p_tail / p_tail.sum(-1, keepdim=True)
    sampler = q.log()
    head_mask = torch.ones(tokens, k, dtype=torch.bool)

    drift = _expected_gradient(logits, sampler, head_ids, head_mask, centered=False)
    assert drift.abs().max() > 1e-2  # the plain score drifts: the check below is not vacuous
    centered = _expected_gradient(logits, sampler, head_ids, head_mask, centered=True)
    assert centered.abs().max() < 1e-6, centered.abs().max()
    assert math.isfinite(float(centered.sum()))
