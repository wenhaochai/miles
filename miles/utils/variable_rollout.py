"""Training on rollouts whose sample count varies (--variable-rollout-samples).

A multi-turn or multi-agent rollout returns one training sample per agent turn, so the number of samples per
rollout changes from rollout to rollout. The sample-count DP path (taken whenever the actor and the critic step at
different batch sizes, e.g. EasyPPO's critic mini-batches) assumes exactly ``global_batch_size`` samples per step:
with fewer it plans zero steps and trains nothing, so callers used to pad every rollout with masked dummy samples.
Here a rollout's samples are instead split into a fixed number of optimizer steps,
``variable_rollout_samples // global_batch_size`` per trainer (the actor's one step and the critic's four for
EasyPPO), whatever their count. Pure functions, no torch: tests/fast/utils/test_variable_rollout.py.
"""

from __future__ import annotations

import heapq


def balanced_dp_partitions(lengths: list[int], dp_size: int) -> list[list[int]]:
    """Sample indices per DP rank: rank sizes differ by at most one, token totals balanced (longest first onto the
    lightest rank that still has room). Every rank gets at least one sample."""
    n = len(lengths)
    assert n >= dp_size, f"{n} samples < dp_size {dp_size}: every rank needs at least one sample"
    caps = [n // dp_size + (1 if r < n % dp_size else 0) for r in range(dp_size)]
    heap = [(0, r) for r in range(dp_size)]
    heapq.heapify(heap)
    parts: list[list[int]] = [[] for _ in range(dp_size)]
    for i in sorted(range(n), key=lambda i: -lengths[i]):
        full = []
        while True:
            load, r = heapq.heappop(heap)
            if len(parts[r]) < caps[r]:
                break
            full.append((load, r))
        parts[r].append(i)
        heapq.heappush(heap, (load + lengths[i], r))
        for item in full:
            heapq.heappush(heap, item)
    for p in parts:
        p.sort()  # keep the rollout order inside a rank
    return parts


def num_steps(variable_rollout_samples: int, global_batch_size: int) -> int:
    """Optimizer steps per rollout for a trainer: the nominal rollout size over its global batch size."""
    k = variable_rollout_samples // global_batch_size
    assert k >= 1 and variable_rollout_samples % global_batch_size == 0, (
        f"--variable-rollout-samples {variable_rollout_samples} must be a positive multiple of the trainer's "
        f"global batch size {global_batch_size}"
    )
    return k


def plan_local_steps(num_local_samples: int, steps: int, micro_batch_size: int) -> tuple[list[list[int]], list[int]]:
    """Split a rank's samples, in order, into `steps` contiguous chunks (sizes differ by at most one) and each chunk
    into micro-batches of up to `micro_batch_size`. Returns (micro_batch_indices, micro-batches per step)."""
    assert num_local_samples >= steps, (
        f"a rank holds {num_local_samples} samples for {steps} optimizer steps; every step needs at least one "
        f"sample on every rank (a rank with no micro-batch would skip the gradient reduction and hang the others)"
    )
    micro_batch_indices: list[list[int]] = []
    num_microbatches: list[int] = []
    for k in range(steps):
        start = k * num_local_samples // steps
        end = (k + 1) * num_local_samples // steps
        mbs = [list(range(i, min(i + micro_batch_size, end))) for i in range(start, end, micro_batch_size)]
        micro_batch_indices.extend(mbs)
        num_microbatches.append(len(mbs))
    return micro_batch_indices, num_microbatches
