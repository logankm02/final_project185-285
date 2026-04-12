from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch, iter_minibatches
from llm_rl_final_proj.models.logprobs import (approx_kl_from_logprobs,compute_per_token_logprobs, masked_mean_per_row,)


class GRPO(RLAlgorithm):
    """GRPO update with a PPO-style clipped surrogate over completion tokens."""

    name = "grpo"

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        # TODO(student): implement one GRPO training iteration.

        accum = 0
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        n_mb = 0
        skipped_empty = 0
        skipped_nonfinite = 0
        total_grad_norm = 0.0
        opt_steps = 0

        # The intended structure is:
        #   1. loop over PPO epochs,

        for epoch in range(self.cfg.ppo_epochs):
            
        #   2. iterate over rollout minibatches,
            for mb in iter_minibatches(
                rollout,
                minibatch_size=self.cfg.minibatch_size,
                shuffle=True,
            ):
        #   3. recompute token log-probabilities under the current policy,
                new_logprobs = compute_per_token_logprobs(
                    model=model,
                    input_ids=mb.input_ids,
                    attention_mask=mb.attention_mask,
                )
        #   4. form PPO ratios against mb.old_logprobs,
                ratios = torch.exp(new_logprobs - mb.old_logprobs)
        #   5. apply token-level clipping with the sequence-level GRPO averaging used in this codebase,
                clipped_ratios = torch.clamp(ratios, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
        #   6. add KL regularization against mb.ref_logprobs,
                kl_divergence = approx_kl_from_logprobs(
                    new_logprobs=new_logprobs,
                    ref_logprobs=mb.ref_logprobs,
                    mask=mb.completion_mask,
                )
        #   7. handle gradient accumulation / clipping / optimizer steps,
                surr1 = ratios * mb.advantages.unsqueeze(-1)
                surr2 = clipped_ratios * mb.advantages.unsqueeze(-1)
                surr = torch.min(surr1, surr2)
                pg_loss = -masked_mean_per_row(surr, mb.completion_mask).mean()
                loss = (pg_loss + self.cfg.kl_coef * kl_divergence) / grad_accum_steps 

                if float(mb.completion_mask.sum().item()) == 0:
                    skipped_empty += 1
                    continue
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    continue

                loss.backward()
                accum += 1
                total_loss += loss.item()
                total_kl += kl_divergence.mean().item()
                total_entropy += (-masked_mean_per_row(new_logprobs, mb.completion_mask)).mean().item()
                n_mb += 1

                if accum % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                    total_grad_norm += grad_norm.item()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_steps += 1
                    accum = 0
        #   8. return the logged metrics expected by the training script.
        if accum > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
            total_grad_norm += float(grad_norm.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1

        denom = max(1, n_mb)
        return {
            "train/policy_loss_with_kl_penalty_mean_over_minibatches": total_loss / denom,
            "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches": total_kl / denom,
            "train/policy_token_entropy_mean_over_minibatches": total_entropy / denom,
            "train/count_minibatches_skipped_because_completion_mask_had_no_tokens": float(skipped_empty),
            "train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients": float(skipped_nonfinite),
            "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps": total_grad_norm / max(1, opt_steps),
            "train/count_optimizer_steps_per_training_iteration": float(opt_steps),
        }

        
        # raise NotImplementedError("Implement GRPO.update in the student starter.")
