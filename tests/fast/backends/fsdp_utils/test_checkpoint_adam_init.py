"""Resuming an (Adam)W optimizer from a dcp checkpoint initializes its state as zeros, not with torch's step on zero
gradients (no gradient is allocated), and loads every state value back exactly."""

import torch
import torch.distributed.checkpoint as dcp

from miles.backends.fsdp_utils.checkpoint import OptimizerState


def _trained(seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.98), weight_decay=0.0)
    for _ in range(2):
        model(torch.randn(5, 4)).square().sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return model, opt


def test_adam_state_round_trips_without_gradients(tmp_path):
    model, opt = _trained(0)
    dcp.save({"optim_state": OptimizerState(model, opt)}, checkpoint_id=str(tmp_path), no_dist=True)

    fresh = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    fresh.load_state_dict(model.state_dict())
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-2, betas=(0.9, 0.98), weight_decay=0.0)
    dcp.load({"optim_state": OptimizerState(fresh, fresh_opt)}, checkpoint_id=str(tmp_path), no_dist=True)

    assert all(p.grad is None for p in fresh.parameters())
    for p, q in zip(model.parameters(), fresh.parameters(), strict=True):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(fresh_opt.state[q][key], opt.state[p][key], rtol=0, atol=0)
