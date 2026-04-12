from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # TODO(student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.
    N = int(batch.input_ids.shape[0])
    if N == 0: return

    if shuffle:
        perm = torch.randperm(N, generator=generator, device=batch.input_ids.device)
    else:
        perm = torch.arange(N, device=batch.input_ids.device)

    for start in range(0, N, minibatch_size):
        idx = perm[start : start + minibatch_size]

        mb = RolloutBatch(
            input_ids=batch.input_ids.index_select(0, idx),
            attention_mask=batch.attention_mask.index_select(0, idx),
            completion_mask=batch.completion_mask.index_select(0, idx),
            old_logprobs=batch.old_logprobs.index_select(0, idx),
            ref_logprobs=batch.ref_logprobs.index_select(0, idx),
            rewards=batch.rewards.index_select(0, idx),
            advantages=batch.advantages.index_select(0, idx),
            task_names=None,
            completion_texts=None,
        )

        if batch.task_names is not None:
            idx_list = idx.tolist()
            mb.task_names = [batch.task_names[i] for i in idx_list]
        if batch.completion_texts is not None:
            idx_list = idx.tolist()
            mb.completion_texts = [batch.completion_texts[i] for i in idx_list]

        if device is not None:
            mb = mb.to(device)

        yield mb
