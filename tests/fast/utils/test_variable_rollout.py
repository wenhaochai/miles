"""miles/utils/variable_rollout.py: DP partitions and step plans for rollouts whose sample count varies."""

import random

import pytest

from miles.utils.variable_rollout import balanced_dp_partitions, num_steps, plan_local_steps


@pytest.mark.parametrize("n,dp", [(2, 2), (7, 2), (9, 2), (1437, 2), (10, 4), (13, 8)])
def test_partitions_cover_every_sample_once_with_sizes_within_one(n, dp):
    rng = random.Random(n * 31 + dp)
    lengths = [rng.randint(50, 60000) for _ in range(n)]
    parts = balanced_dp_partitions(lengths, dp)
    assert sorted(i for p in parts for i in p) == list(range(n))
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1 and min(sizes) >= 1
    assert all(p == sorted(p) for p in parts)  # rollout order kept inside a rank


def test_partitions_balance_tokens():
    rng = random.Random(0)
    lengths = [rng.randint(500, 60000) for _ in range(1001)]
    loads = [sum(lengths[i] for i in p) for p in balanced_dp_partitions(lengths, 2)]
    assert abs(loads[0] - loads[1]) <= max(lengths)  # longest-first greedy: within one sample


def test_partitions_need_one_sample_per_rank():
    with pytest.raises(AssertionError):
        balanced_dp_partitions([5], 2)


def test_num_steps():
    assert num_steps(1440, 1440) == 1 and num_steps(1440, 360) == 4
    with pytest.raises(AssertionError):
        num_steps(1440, 500)


@pytest.mark.parametrize("n,steps,mbs", [(4, 4, 1), (5, 4, 1), (719, 4, 1), (720, 1, 1), (11, 3, 2)])
def test_plan_covers_each_local_sample_once(n, steps, mbs):
    mb_indices, num_mbs = plan_local_steps(n, steps, mbs)
    assert len(num_mbs) == steps and sum(num_mbs) == len(mb_indices)
    flat = [i for mb in mb_indices for i in mb]
    assert flat == list(range(n))  # in order, every sample once
    assert all(1 <= len(mb) <= mbs for mb in mb_indices)
    chunk = [sum(len(mb) for mb in mb_indices[sum(num_mbs[:k]):sum(num_mbs[:k + 1])]) for k in range(steps)]
    assert max(chunk) - min(chunk) <= 1 and min(chunk) >= 1


def test_plan_needs_a_sample_per_step():
    with pytest.raises(AssertionError):
        plan_local_steps(3, 4, 1)


def _args(**kw):
    from argparse import Namespace

    base = dict(variable_rollout_samples=1440, global_batch_size=1440, critic_global_batch_size=360,
                calculate_per_token_loss=True, use_dynamic_batch_size=False, use_dynamic_global_batch_size=False,
                train_backend="megatron", pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None,
                num_experts=None, expert_model_parallel_size=1, moe_router_load_balancing_type="aux_loss", fp8=None,
                qkv_format="bshd", micro_batch_size=1, bshd_pad_per_sample=True)
    base.update(kw)
    return Namespace(**base)


@pytest.mark.parametrize("kw", [
    dict(variable_rollout_samples=2880),  # not the actor's global batch: the actor would take 2 steps silently
    dict(variable_rollout_samples=-1440),
    dict(critic_global_batch_size=500),
    dict(calculate_per_token_loss=False),
    dict(train_backend="fsdp"),
    dict(pipeline_model_parallel_size=2),
    dict(num_experts=8, expert_model_parallel_size=2),
    dict(num_experts=8, moe_router_load_balancing_type="global_aux_loss"),
    dict(fp8="hybrid", fp8_recipe="delayed"),
    dict(micro_batch_size=2),  # --bshd-pad-per-sample needs micro-batch size 1
])
def test_validation_rejects(kw):
    from miles.utils.arguments import _validate_variable_rollout

    with pytest.raises(AssertionError):
        _validate_variable_rollout(_args(**kw))


def test_validation_accepts_the_easyppo_team_config():
    from miles.utils.arguments import _validate_variable_rollout

    _validate_variable_rollout(_args())
    _validate_variable_rollout(_args(num_experts=8, expert_model_parallel_size=1))  # EP = 1, local aux loss
