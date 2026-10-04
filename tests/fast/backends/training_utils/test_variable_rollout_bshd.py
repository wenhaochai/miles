"""--variable-rollout-samples with --bshd-pad-per-sample through the real trainer data path, on CPU: get_data_iterator
plans the trainer's steps, get_batch pads each sample to its own length, and the response slice of the logits lines up
with the response tokens. Adapted from the 2026-10-04 review simulation."""

import random
import types
from argparse import Namespace

import torch

import miles.backends.training_utils.cp_utils as cp_utils
import miles.backends.training_utils.data as data
import miles.backends.training_utils.loss_hub.logit_processors as logit_processors


class _Group:
    def __init__(self, size, rank=0):
        self.size, self.rank, self.group = size, rank, None


_PS = types.SimpleNamespace(tp=_Group(4), cp=_Group(1), effective_dp=_Group(2), vpp_size=1,
                            microbatch_group_size_per_vp_stage=None, intra_dp=_Group(2), intra_dp_cp=_Group(2),
                            is_pp_last_stage=True)
PAD = 4 * 128  # TP 4 x the default data pad multiplier


def test_variable_steps_and_per_sample_padding(monkeypatch):
    for module in (data, cp_utils, logit_processors):
        monkeypatch.setattr(module, "get_parallel_state", lambda: _PS)
    monkeypatch.setattr(data, "expand_multimodal_rollout_data_in_place", lambda *a, **k: None)
    rng = random.Random(1)
    for _ in range(60):
        n = rng.randint(4, 40)
        prompts = [rng.randint(1, 3000) for _ in range(n)]
        resps = [rng.randint(1, 5000) for _ in range(n)]
        tokens = [torch.randint(1, 1000, (p + r,)) for p, r in zip(prompts, resps)]
        total = [len(t) for t in tokens]
        rollout = dict(tokens=tokens, total_lengths=total, response_lengths=resps,
                       loss_masks=[torch.ones(r, dtype=torch.int) for r in resps],
                       max_seq_lens=[(x + PAD - 1) // PAD * PAD for x in total])
        for gbs, steps in ((1440, 1), (360, 4)):  # the actor's one step, the critic's four
            args = Namespace(qkv_format="bshd", variable_rollout_samples=1440, global_batch_size=gbs,
                             micro_batch_size=1, use_dynamic_global_batch_size=False, use_dynamic_batch_size=False)
            iterators, num_mbs = data.get_data_iterator(args, None, rollout)
            assert len(num_mbs) == steps and sum(num_mbs) == n and min(num_mbs) >= 1
            iterators[0].reset()
            seen = []
            for _ in range(sum(num_mbs)):
                b = data.get_batch(iterators[0], ["tokens", "loss_masks", "total_lengths", "response_lengths",
                                                  "max_seq_lens"], 128, "bshd")
                S = b["tokens"].shape[1]
                assert b["tokens"].shape == (1, S) and S % PAD == 0 and 0 <= S - b["total_lengths"][0] < PAD
                assert S == b["max_seq_lens"][0] and b["input_loss_masks"].shape == b["tokens"].shape
                # logits equal to their position: the response chunk must be positions [prompt - 1, total - 1)
                logits = torch.arange(S, dtype=torch.float32).view(1, S, 1).expand(1, S, 2).contiguous()
                a = Namespace(qkv_format="bshd", true_on_policy_mode=False, allgather_cp=False)
                ((chunk, toks, _),) = list(logit_processors._iter_response_chunks(
                    logits, args=a, unconcat_tokens=b["unconcat_tokens"], total_lengths=b["total_lengths"],
                    response_lengths=b["response_lengths"], max_seq_lens=b["max_seq_lens"],
                    include_response_indices=False))
                p = b["total_lengths"][0] - b["response_lengths"][0]
                assert torch.equal(chunk[:, 0], torch.arange(p - 1, b["total_lengths"][0] - 1, dtype=torch.float32))
                assert torch.equal(toks, b["unconcat_tokens"][0][p:])
                seen.append(id(b["unconcat_tokens"][0]))
            assert seen == [id(t) for t in tokens]  # every sample once, in order
