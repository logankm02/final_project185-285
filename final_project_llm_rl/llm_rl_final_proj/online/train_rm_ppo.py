from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from llm_rl_final_proj.data.ultrafeedback import GenerationExample, build_generation_examples, dataset_overview
from llm_rl_final_proj.models.load import load_lora_policy_model_and_tokenizer, load_reward_model_and_tokenizer
from llm_rl_final_proj.models.logprobs import approx_kl_from_logprobs, compute_per_token_logprobs, masked_mean
from llm_rl_final_proj.offline.evaluation import generate_samples, summarize_generation_rows
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs
from llm_rl_final_proj.rollout.hf_sampler import HFSampler, SamplingConfig
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch
from llm_rl_final_proj.utils.hardware import (
    get_cuda_memory_metrics,
    get_hardware_metrics,
    get_model_device_metrics,
    require_cuda_if_requested,
    resolve_device_and_dtype,
)
from llm_rl_final_proj.utils.seed import set_seed
from llm_rl_final_proj.utils.wandb_utils import WandBLogger


@dataclass
class OnlineRMPPOConfig:
    """On-policy PPO with a learned reward model and a learned value baseline."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_adapter_path: str = ""
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_gen"
    eval_split: str = "test_gen"
    output_dir: str = "runs/rm_ppo_default"

    seed: int = 0

    steps: int = 101
    batch_size: int = 8
    group_size: int = 1

    min_new_tokens: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    lr: float = 3e-5
    value_lr: float = 3e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 20
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.5

    ppo_epochs: int = 2
    minibatch_size: int = 8
    clip_eps: float = 0.2
    vf_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    kl_coef: float = 0.01
    adv_clip: float = 5.0
    normalize_advantages: bool = True

    max_prompt_tokens: int = 700
    max_response_tokens: int = 512
    train_limit: int = 0
    eval_limit: int = 64
    reward_batch_size: int = 16

    eval_interval: int = 25
    save_interval: int = 50
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_batch_size: int = 8

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    grad_checkpointing: bool = True

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "rm_ppo"
    wandb_enabled: bool = True
    sample_log_n: int = 8
    sample_log_max_chars: int = 2500


def parse_args() -> OnlineRMPPOConfig:
    ap = argparse.ArgumentParser(description="Train a policy online with PPO using a learned reward model.")
    ap.add_argument("--model_name", type=str, default=OnlineRMPPOConfig.model_name)
    ap.add_argument("--reward_model_name", type=str, default=OnlineRMPPOConfig.reward_model_name)
    ap.add_argument("--reward_adapter_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default=OnlineRMPPOConfig.dataset_name)
    ap.add_argument("--train_split", type=str, default=OnlineRMPPOConfig.train_split)
    ap.add_argument("--eval_split", type=str, default=OnlineRMPPOConfig.eval_split)
    ap.add_argument("--output_dir", type=str, default=OnlineRMPPOConfig.output_dir)

    ap.add_argument("--seed", type=int, default=OnlineRMPPOConfig.seed)
    ap.add_argument("--steps", type=int, default=OnlineRMPPOConfig.steps)
    ap.add_argument("--batch_size", type=int, default=OnlineRMPPOConfig.batch_size)
    ap.add_argument("--group_size", type=int, default=OnlineRMPPOConfig.group_size)

    ap.add_argument("--min_new_tokens", type=int, default=OnlineRMPPOConfig.min_new_tokens)
    ap.add_argument("--max_new_tokens", type=int, default=OnlineRMPPOConfig.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=OnlineRMPPOConfig.temperature)
    ap.add_argument("--top_p", type=float, default=OnlineRMPPOConfig.top_p)
    ap.add_argument("--top_k", type=int, default=OnlineRMPPOConfig.top_k)
    ap.add_argument("--repetition_penalty", type=float, default=OnlineRMPPOConfig.repetition_penalty)

    ap.add_argument("--lr", type=float, default=OnlineRMPPOConfig.lr)
    ap.add_argument("--value_lr", type=float, default=OnlineRMPPOConfig.value_lr)
    ap.add_argument("--weight_decay", type=float, default=OnlineRMPPOConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=OnlineRMPPOConfig.betas1)
    ap.add_argument("--betas2", type=float, default=OnlineRMPPOConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=OnlineRMPPOConfig.warmup_steps)
    ap.add_argument("--grad_accum_steps", type=int, default=OnlineRMPPOConfig.grad_accum_steps)
    ap.add_argument("--max_grad_norm", type=float, default=OnlineRMPPOConfig.max_grad_norm)

    ap.add_argument("--ppo_epochs", type=int, default=OnlineRMPPOConfig.ppo_epochs)
    ap.add_argument("--minibatch_size", type=int, default=OnlineRMPPOConfig.minibatch_size)
    ap.add_argument("--clip_eps", type=float, default=OnlineRMPPOConfig.clip_eps)
    ap.add_argument("--vf_clip", type=float, default=OnlineRMPPOConfig.vf_clip)
    ap.add_argument("--vf_coef", type=float, default=OnlineRMPPOConfig.vf_coef)
    ap.add_argument("--ent_coef", type=float, default=OnlineRMPPOConfig.ent_coef)
    ap.add_argument("--kl_coef", type=float, default=OnlineRMPPOConfig.kl_coef)
    ap.add_argument("--adv_clip", type=float, default=OnlineRMPPOConfig.adv_clip)
    ap.add_argument(
        "--normalize_advantages",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMPPOConfig.normalize_advantages,
    )

    ap.add_argument("--max_prompt_tokens", type=int, default=OnlineRMPPOConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=OnlineRMPPOConfig.max_response_tokens)
    ap.add_argument("--train_limit", type=int, default=OnlineRMPPOConfig.train_limit)
    ap.add_argument("--eval_limit", type=int, default=OnlineRMPPOConfig.eval_limit)
    ap.add_argument("--reward_batch_size", type=int, default=OnlineRMPPOConfig.reward_batch_size)

    ap.add_argument("--eval_interval", type=int, default=OnlineRMPPOConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=OnlineRMPPOConfig.save_interval)
    ap.add_argument("--eval_max_new_tokens", type=int, default=OnlineRMPPOConfig.eval_max_new_tokens)
    ap.add_argument("--eval_temperature", type=float, default=OnlineRMPPOConfig.eval_temperature)
    ap.add_argument("--eval_top_p", type=float, default=OnlineRMPPOConfig.eval_top_p)
    ap.add_argument("--eval_batch_size", type=int, default=OnlineRMPPOConfig.eval_batch_size)

    ap.add_argument("--lora_r", type=int, default=OnlineRMPPOConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=OnlineRMPPOConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=OnlineRMPPOConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=OnlineRMPPOConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=OnlineRMPPOConfig.lora_bias)
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMPPOConfig.grad_checkpointing,
    )

    ap.add_argument("--wandb_project", type=str, default=OnlineRMPPOConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=OnlineRMPPOConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMPPOConfig.wandb_enabled,
    )
    ap.add_argument("--sample_log_n", type=int, default=OnlineRMPPOConfig.sample_log_n)
    ap.add_argument("--sample_log_max_chars", type=int, default=OnlineRMPPOConfig.sample_log_max_chars)
    args = ap.parse_args()
    return OnlineRMPPOConfig(**vars(args))


def maybe_update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def maybe_update_warmup_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs: Sequence[float],
    step: int,
    warmup_steps: int,
) -> None:
    """Warmup helper for multi-param-group optimizers."""
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    if len(base_lrs) != len(optimizer.param_groups):
        raise ValueError(
            f"base_lrs length ({len(base_lrs)}) must match optimizer.param_groups ({len(optimizer.param_groups)})"
        )
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = float(base_lr) * float(scale)


def _normalize_lora_target_modules(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _sample_prompt_batch(examples: Sequence[GenerationExample], batch_size: int, rng: random.Random) -> List[GenerationExample]:
    if not examples:
        raise RuntimeError("Cannot sample prompts from an empty generation split.")
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


def _normalize_completion_for_reward_scoring(text: str) -> str:
    if text.strip():
        return text
    return "[no response]"


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def _sample_rows_for_logging(
    examples: Sequence[GenerationExample],
    rows: Sequence[Dict[str, Any]],
    rm_scores: Sequence[float],
    *,
    sample_log_n: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex, row, score in list(zip(examples, rows, rm_scores))[: max(0, sample_log_n)]:
        out.append(
            {
                "row_id": ex.row_id,
                "prompt": _truncate(ex.prompt_text, max_chars),
                "reference_response": _truncate(ex.reference_response_text, max_chars),
                "model_response": _truncate(str(row.get("model_response", "")), max_chars),
                "reward_model_score": float(score),
            }
        )
    return out


class ValueHead(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value = torch.nn.Linear(hidden_size, 1)

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return self.value(last_hidden).squeeze(-1)


def _compute_sequence_values(
    *,
    model: torch.nn.Module,
    value_head: ValueHead,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = out.hidden_states[-1]
    last_idx = attention_mask.sum(dim=1).clamp_min(1) - 1 
    last_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx] 
    last_hidden = last_hidden.to(dtype=value_head.value.weight.dtype)
    return value_head(last_hidden) 


def save_checkpoint(model: torch.nn.Module, value_head: ValueHead, cfg: OnlineRMPPOConfig, step: int) -> None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    torch.save(value_head.state_dict(), ckpt_dir / "value_head.pt")
    meta = {
        "step": step,
        "model_type": "online_policy_rm_ppo",
        "algo": "ppo",
        "model_name": cfg.model_name,
        "reward_model_name": cfg.reward_model_name,
        "reward_adapter_path": cfg.reward_adapter_path,
        "dataset_name": cfg.dataset_name,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def evaluate_policy_with_reward_model(
    *,
    policy_model: torch.nn.Module,
    policy_tokenizer,
    reward_model: torch.nn.Module,
    reward_tokenizer,
    examples: Sequence[GenerationExample],
    device: torch.device,
    max_prompt_tokens: int,
    max_response_tokens: int,
    generation_max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_batch_size: int,
) -> tuple[Dict[str, float], List[Dict[str, Any]], List[float]]:
    rows = generate_samples(
        policy_model,
        policy_tokenizer,
        examples,
        device=device,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=generation_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=generation_batch_size,
    )
    metrics = summarize_generation_rows(rows)
    scoring_rows = []
    reference_rows = []
    has_reference = True
    for ex, row in zip(examples, rows):
        scoring_rows.append(
            {
                "row_id": ex.row_id,
                "prompt_messages": ex.prompt_messages,
                "prompt_text": ex.prompt_text,
                "response_text": _normalize_completion_for_reward_scoring(str(row.get("model_response", ""))),
            }
        )
        if ex.reference_response_text:
            reference_rows.append(
                {
                    "row_id": ex.row_id,
                    "prompt_messages": ex.prompt_messages,
                    "prompt_text": ex.prompt_text,
                    "response_text": ex.reference_response_text,
                }
            )
        else:
            has_reference = False

    rm_scores = score_prompt_response_pairs(
        reward_model,
        reward_tokenizer,
        scoring_rows,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
        per_device_batch_size=generation_batch_size,
        device=device,
    )
    score_tensor = torch.tensor(rm_scores, dtype=torch.float32)
    metrics["eval/rm_score_mean_on_policy_generations"] = float(score_tensor.mean().item())
    metrics["eval/rm_score_std_on_policy_generations"] = float(score_tensor.std(unbiased=False).item())
    if has_reference and reference_rows:
        ref_scores = score_prompt_response_pairs(
            reward_model,
            reward_tokenizer,
            reference_rows,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            per_device_batch_size=generation_batch_size,
            device=device,
        )
        ref_tensor = torch.tensor(ref_scores, dtype=torch.float32)
        margin = score_tensor - ref_tensor
        metrics["eval/rm_reference_score_mean_on_dataset_reference_responses"] = float(ref_tensor.mean().item())
        metrics["eval/rm_fraction_policy_scores_above_reference"] = float((margin > 0).float().mean().item())
        metrics["eval/rm_margin_policy_minus_reference_mean"] = float(margin.mean().item())
    return metrics, rows, rm_scores


def _build_advantages_and_returns(
    rewards: torch.Tensor, values: torch.Tensor, *, normalize: bool, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = rewards
    adv = (returns - values).detach()
    if normalize:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + eps)
    return adv, returns


def _masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def ppo_update(
    *,
    cfg: OnlineRMPPOConfig,
    policy_model: torch.nn.Module,
    value_head: ValueHead,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    old_values: torch.Tensor,
    update_seed: int,
) -> Dict[str, float]:
    policy_model.train()
    policy_model.config.use_cache = False

    trainable_params = [p for p in list(policy_model.parameters()) + list(value_head.parameters()) if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_kl = 0.0
    total_entropy = 0.0
    total_ratio_mean = 0.0
    n_mb = 0
    accum = 0
    skipped_empty = 0
    skipped_nonfinite = 0
    total_grad_norm = 0.0
    opt_steps = 0

    rng = torch.Generator(device=rollout.input_ids.device)
    rng.manual_seed(int(cfg.seed + update_seed))

    advantages, returns = _build_advantages_and_returns(
        rollout.rewards, old_values, normalize=cfg.normalize_advantages
    )
    advantages = advantages.clamp(-cfg.adv_clip, cfg.adv_clip)

    for _epoch in range(cfg.ppo_epochs):
        for mb, mb_old_values, mb_adv, mb_returns in _iter_minibatches_with_extras(
            rollout,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            minibatch_size=cfg.minibatch_size,
            shuffle=True,
            generator=rng,
            device=next(policy_model.parameters()).device,
        ):
            mask = mb.completion_mask
            if float(mask.sum().item()) <= 0.0:
                skipped_empty += 1
                continue

            new_logp = compute_per_token_logprobs(policy_model, mb.input_ids, mb.attention_mask, enable_grad=True)
            log_ratio_tok = new_logp - mb.old_logprobs
            log_ratio_seq = _masked_mean_per_row(log_ratio_tok, mask)
            ratio = torch.exp(log_ratio_seq)
            ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

            surr1 = ratio * mb_adv
            surr2 = ratio_clipped * mb_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            values = _compute_sequence_values(
                model=policy_model,
                value_head=value_head,
                input_ids=mb.input_ids,
                attention_mask=mb.attention_mask,
            )
            vpred = values
            v_old = mb_old_values
            v_target = mb_returns
            if cfg.vf_clip > 0:
                vpred_clipped = v_old + (vpred - v_old).clamp(-cfg.vf_clip, cfg.vf_clip)
                vf_loss1 = (vpred - v_target).pow(2)
                vf_loss2 = (vpred_clipped - v_target).pow(2)
                vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
            else:
                vf_loss = 0.5 * (vpred - v_target).pow(2).mean()

            kl = approx_kl_from_logprobs(new_logp, mb.ref_logprobs, mask)
            entropy = -masked_mean(new_logp, mask)

            loss = pg_loss + cfg.vf_coef * vf_loss + cfg.kl_coef * kl - cfg.ent_coef * entropy
            loss = loss / max(1, cfg.grad_accum_steps)
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                continue

            loss.backward()
            accum += 1

            if (accum % max(1, cfg.grad_accum_steps)) == 0:
                gnorm = float(torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm).item())
                if not math.isfinite(gnorm):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_grad_norm += gnorm
                opt_steps += 1

            total_loss += float((loss.detach() * max(1, cfg.grad_accum_steps)).item())
            total_pg_loss += float(pg_loss.detach().item())
            total_vf_loss += float(vf_loss.detach().item())
            total_kl += float(kl.detach().item())
            total_entropy += float(entropy.detach().item())
            total_ratio_mean += float(ratio.detach().mean().item())
            n_mb += 1

    if accum > 0 and (accum % max(1, cfg.grad_accum_steps)) != 0:
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm).item())
        if math.isfinite(gnorm):
            optimizer.step()
            total_grad_norm += gnorm
            opt_steps += 1
        else:
            skipped_nonfinite += 1
        optimizer.zero_grad(set_to_none=True)

    denom = max(1, n_mb)
    return {
        "train/total_loss_mean_over_minibatches": total_loss / denom,
        "train/policy_loss_mean_over_minibatches": total_pg_loss / denom,
        "train/value_loss_mean_over_minibatches": total_vf_loss / denom,
        "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches": total_kl / denom,
        "train/policy_token_entropy_mean_over_minibatches": total_entropy / denom,
        "train/ppo_ratio_mean_over_minibatches": total_ratio_mean / denom,
        "train/count_minibatches_skipped_because_completion_mask_had_no_tokens": float(skipped_empty),
        "train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients": float(skipped_nonfinite),
        "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps": total_grad_norm / max(1, opt_steps),
        "train/count_optimizer_steps_per_training_iteration": float(opt_steps),
    }


def _iter_minibatches_with_extras(
    rollout: RolloutBatch,
    *,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    minibatch_size: int,
    shuffle: bool,
    generator: torch.Generator,
    device: torch.device,
):

    N = int(rollout.input_ids.shape[0])
    if N <= 0:
        return
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=rollout.input_ids.device)
    else:
        perm = torch.arange(N, device=rollout.input_ids.device)
    for start in range(0, N, minibatch_size):
        idx = perm[start : start + minibatch_size]
        mb = RolloutBatch(
            input_ids=rollout.input_ids.index_select(0, idx),
            attention_mask=rollout.attention_mask.index_select(0, idx),
            completion_mask=rollout.completion_mask.index_select(0, idx),
            old_logprobs=rollout.old_logprobs.index_select(0, idx),
            ref_logprobs=rollout.ref_logprobs.index_select(0, idx),
            rewards=rollout.rewards.index_select(0, idx),
            advantages=rollout.advantages.index_select(0, idx),
            task_names=None,
            completion_texts=None,
        )
        if device is not None:
            mb = mb.to(device)
        yield mb, old_values.index_select(0, idx).to(device), advantages.index_select(0, idx).to(device), returns.index_select(
            0, idx
        ).to(device)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()
    if cfg.steps <= 0:
        raise ValueError(f"--steps must be >= 1, got {cfg.steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"--batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.group_size <= 0:
        raise ValueError(f"--group_size must be >= 1, got {cfg.group_size}")
    if not cfg.reward_adapter_path:
        raise ValueError("--reward_adapter_path is required")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_online_rm_ppo_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rng = random.Random(cfg.seed)
    device, dtype = resolve_device_and_dtype()
    print(
        f"[setup] device={device} dtype={dtype} algo=ppo policy={cfg.model_name} reward_model={cfg.reward_model_name}"
    )
    print("[setup][hardware]", json.dumps(get_hardware_metrics(device), indent=2, sort_keys=True))

    dataset_info = dataset_overview(cfg.dataset_name)
    train_examples = build_generation_examples(cfg.dataset_name, cfg.train_split, limit=cfg.train_limit)
    eval_examples = build_generation_examples(cfg.dataset_name, cfg.eval_split, limit=cfg.eval_limit)
    if not train_examples:
        raise RuntimeError("Training generation split produced zero examples.")
    if not eval_examples:
        raise RuntimeError("Evaluation generation split produced zero examples.")

    loaded_policy = load_lora_policy_model_and_tokenizer(
        cfg.model_name,
        device=device,
        dtype=dtype,
        grad_checkpointing=cfg.grad_checkpointing,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=_normalize_lora_target_modules(cfg.lora_target_modules),
        lora_bias=cfg.lora_bias,
    )
    policy_model = loaded_policy.model
    policy_tokenizer = loaded_policy.tokenizer

    hidden_size = int(getattr(policy_model.config, "hidden_size", 0) or getattr(policy_model.config, "n_embd", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError("Could not infer hidden size for value head from model.config.")
    value_head = ValueHead(hidden_size).to(device)

    loaded_reward = load_reward_model_and_tokenizer(
        cfg.reward_model_name,
        device=device,
        dtype=dtype,
        adapter_path=cfg.reward_adapter_path,
    )
    reward_model = loaded_reward.model
    reward_tokenizer = loaded_reward.tokenizer
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in policy_model.parameters() if p.requires_grad], "lr": cfg.lr},
            {"params": list(value_head.parameters()), "lr": cfg.value_lr},
        ],
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )

    sampler = HFSampler(policy_tokenizer, device=device)
    sampling_cfg = SamplingConfig(
        min_new_tokens=cfg.min_new_tokens,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=cfg.temperature > 0.0,
    )

    logger = WandBLogger(
        project=cfg.wandb_project,
        run_name=cfg.wandb_name,
        config=vars(cfg),
        enabled=cfg.wandb_enabled,
        local_dir=output_dir,
    )
    logger.log(
        {
            "setup/trainable_params": float(loaded_policy.trainable_params + sum(p.numel() for p in value_head.parameters())),
            "setup/total_params": float(loaded_policy.total_params + sum(p.numel() for p in value_head.parameters())),
            "setup/trainable_fraction": float(
                (loaded_policy.trainable_params + sum(p.numel() for p in value_head.parameters()))
                / max(1, (loaded_policy.total_params + sum(p.numel() for p in value_head.parameters())))
            ),
            "dataset/train_examples": float(len(train_examples)),
            "dataset/eval_examples": float(len(eval_examples)),
            **{f"dataset/{k}": float(v) for k, v in dataset_info["splits"].items()},
            **get_hardware_metrics(device),
            **get_model_device_metrics(policy_model),
        },
        step=0,
    )

    def run_eval(step: int, phase: str) -> Dict[str, float]:
        metrics, rows, rm_scores = evaluate_policy_with_reward_model(
            policy_model=policy_model,
            policy_tokenizer=policy_tokenizer,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            examples=eval_examples,
            device=device,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            generation_max_new_tokens=cfg.eval_max_new_tokens,
            temperature=cfg.eval_temperature,
            top_p=cfg.eval_top_p,
            generation_batch_size=cfg.eval_batch_size,
        )
        logger.log(metrics, step=step)
        logger.log_table(
            f"samples/eval_{phase}",
            _sample_rows_for_logging(
                eval_examples,
                rows,
                rm_scores,
                sample_log_n=cfg.sample_log_n,
                max_chars=cfg.sample_log_max_chars,
            ),
            step=step,
        )
        return metrics

    print("[eval] running baseline evaluation at step=0")
    run_eval(step=0, phase="baseline")

    start_time = time.time()
    base_lrs = [float(cfg.lr), float(cfg.value_lr)]
    for step in range(1, cfg.steps + 1):
        maybe_update_warmup_lrs(optimizer, base_lrs, step - 1, cfg.warmup_steps)
        prompt_batch = _sample_prompt_batch(train_examples, cfg.batch_size, rng)
        rollout = sampler.rollout(
            policy_model=policy_model,
            prompt_messages=[ex.prompt_messages for ex in prompt_batch],
            task_names=["synthetic_instruction_following"] * len(prompt_batch),
            task_metas=[
                {
                    "row_id": ex.row_id,
                    "prompt_text": ex.prompt_text,
                    "reference_response_text": ex.reference_response_text,
                }
                for ex in prompt_batch
            ],
            group_size=cfg.group_size,
            sampling=sampling_cfg,
            max_prompt_tokens=cfg.max_prompt_tokens,
            output_to_cpu=False,
        )

        reward_rows = []
        for i, completion_text in enumerate(rollout.completion_texts):
            meta = rollout.task_metas[i]
            reward_rows.append(
                {
                    "row_id": f"{meta.get('row_id', i)}:{i}",
                    "prompt_messages": rollout.prompt_messages[i],
                    "prompt_text": str(meta.get("prompt_text", "")),
                    "response_text": _normalize_completion_for_reward_scoring(completion_text),
                }
            )
        reward_scores = score_prompt_response_pairs(
            reward_model,
            reward_tokenizer,
            reward_rows,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            per_device_batch_size=cfg.reward_batch_size,
            device=device,
        )
        rewards = torch.tensor(reward_scores, device=device, dtype=torch.float32)

        with torch.no_grad():
            old_values = _compute_sequence_values(
                model=policy_model,
                value_head=value_head,
                input_ids=rollout.input_ids,
                attention_mask=rollout.attention_mask,
            )

        batch = RolloutBatch(
            input_ids=rollout.input_ids,
            attention_mask=rollout.attention_mask,
            completion_mask=rollout.completion_mask,
            old_logprobs=rollout.old_logprobs,
            ref_logprobs=rollout.ref_logprobs,
            rewards=rewards,
            advantages=torch.zeros_like(rewards),
            task_names=rollout.task_names,
            completion_texts=rollout.completion_texts,
        )

        train_metrics = ppo_update(
            cfg=cfg,
            policy_model=policy_model,
            value_head=value_head,
            optimizer=optimizer,
            rollout=batch,
            old_values=old_values,
            update_seed=step,
        )

        completion_lengths = batch.completion_mask.sum(dim=1).float()
        adv, _ret = _build_advantages_and_returns(rewards, old_values, normalize=cfg.normalize_advantages)
        log_metrics = {
            "rollout/reward_model_score_mean": float(rewards.mean().item()),
            "rollout/reward_model_score_std": float(rewards.std(unbiased=False).item()),
            "rollout/reward_model_score_min": float(rewards.min().item()),
            "rollout/reward_model_score_max": float(rewards.max().item()),
            "rollout/value_mean": float(old_values.mean().item()),
            "rollout/value_std": float(old_values.std(unbiased=False).item()),
            "rollout/advantage_mean": float(adv.mean().item()),
            "rollout/advantage_std": float(adv.std(unbiased=False).item()),
            "rollout/completion_mean_tokens": float(completion_lengths.mean().item()),
            "rollout/completion_max_tokens": float(completion_lengths.max().item()),
            "rollout/count_completions": float(rewards.numel()),
            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
            "time/seconds_since_start": float(time.time() - start_time),
            **train_metrics,
            **get_cuda_memory_metrics(prefix="train"),
        }
        logger.log(log_metrics, step=step)

        should_eval = (step % cfg.eval_interval == 0) or (step == cfg.steps)
        should_save = (step % cfg.save_interval == 0) or (step == cfg.steps)
        if should_eval:
            print(f"[eval] running evaluation at step={step}")
            run_eval(step=step, phase=f"step_{step}")
        if should_save:
            print(f"[checkpoint] saving step={step}")
            save_checkpoint(policy_model, value_head, cfg, step=step)

    logger.finish()

if __name__ == "__main__":
    main()
