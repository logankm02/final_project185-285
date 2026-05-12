from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence

import torch  # pyright: ignore[reportMissingImports]

from llm_rl_final_proj.data.ultrafeedback import (
    GenerationExample,
    PreferenceExample,
    build_generation_examples,
    dataset_overview,
)
from llm_rl_final_proj.models.load import load_lora_policy_model_and_tokenizer, load_reward_model_and_tokenizer
from llm_rl_final_proj.offline.evaluation import generate_samples, summarize_generation_rows
from llm_rl_final_proj.offline.losses import compute_policy_and_reference_scores
from llm_rl_final_proj.offline.losses import compute_offline_preference_loss
from llm_rl_final_proj.offline.batch import PreferenceCollator
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs
from llm_rl_final_proj.rollout.hf_sampler import HFSampler, SamplingConfig
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
class OnlinePreferenceConfig:
    """Online preference optimization from a reward model with replay.

    Mechanics:
      - sample prompts from `train_gen`,
      - sample `group_size` responses per prompt,
      - score with a frozen reward model,
      - create a synthetic preference pair per prompt (best vs worst),
      - push into a replay buffer,
      - run DPO/IPO/AOT updates on replay (same loss code as Part 1).
    """

    algo: str = "dpo"
    beta: float = 0.1

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_adapter_path: str = ""

    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_gen"
    eval_split: str = "test_gen"
    output_dir: str = "runs/rm_online_pref_default"

    seed: int = 0
    steps: int = 101
    prompt_batch_size: int = 8
    group_size: int = 4

    replay_size: int = 4096
    min_replay_size_to_train: int = 128

    updates_per_rollout: int = 1
    per_device_train_batch_size: int = 8
    grad_accum_steps: int = 2

    lr: float = 5e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 20
    max_grad_norm: float = 1.0

    max_prompt_tokens: int = 700
    max_response_tokens: int = 512
    min_new_tokens: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    reward_batch_size: int = 16

    eval_interval: int = 25
    save_interval: int = 50
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_batch_size: int = 8
    eval_limit: int = 64
    train_limit: int = 0

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    grad_checkpointing: bool = True

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "rm_online_pref"
    wandb_enabled: bool = True
    log_interval: int = 1
    sample_log_n: int = 8
    sample_log_max_chars: int = 2500


def parse_args() -> OnlinePreferenceConfig:
    ap = argparse.ArgumentParser(
        description="Online preference optimization with replay (reward-model-ranked rollouts → DPO/IPO/AOT)."
    )
    ap.add_argument("--algo", type=str, default=OnlinePreferenceConfig.algo, choices=["dpo", "ipo", "aot"])
    ap.add_argument("--beta", type=float, default=OnlinePreferenceConfig.beta)

    ap.add_argument("--model_name", type=str, default=OnlinePreferenceConfig.model_name)
    ap.add_argument("--reward_model_name", type=str, default=OnlinePreferenceConfig.reward_model_name)
    ap.add_argument("--reward_adapter_path", type=str, required=True)

    ap.add_argument("--dataset_name", type=str, default=OnlinePreferenceConfig.dataset_name)
    ap.add_argument("--train_split", type=str, default=OnlinePreferenceConfig.train_split)
    ap.add_argument("--eval_split", type=str, default=OnlinePreferenceConfig.eval_split)
    ap.add_argument("--output_dir", type=str, default=OnlinePreferenceConfig.output_dir)

    ap.add_argument("--seed", type=int, default=OnlinePreferenceConfig.seed)
    ap.add_argument("--steps", type=int, default=OnlinePreferenceConfig.steps)
    ap.add_argument("--prompt_batch_size", type=int, default=OnlinePreferenceConfig.prompt_batch_size)
    ap.add_argument("--group_size", type=int, default=OnlinePreferenceConfig.group_size)

    ap.add_argument("--replay_size", type=int, default=OnlinePreferenceConfig.replay_size)
    ap.add_argument("--min_replay_size_to_train", type=int, default=OnlinePreferenceConfig.min_replay_size_to_train)
    ap.add_argument("--updates_per_rollout", type=int, default=OnlinePreferenceConfig.updates_per_rollout)
    ap.add_argument("--per_device_train_batch_size", type=int, default=OnlinePreferenceConfig.per_device_train_batch_size)
    ap.add_argument("--grad_accum_steps", type=int, default=OnlinePreferenceConfig.grad_accum_steps)

    ap.add_argument("--lr", type=float, default=OnlinePreferenceConfig.lr)
    ap.add_argument("--weight_decay", type=float, default=OnlinePreferenceConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=OnlinePreferenceConfig.betas1)
    ap.add_argument("--betas2", type=float, default=OnlinePreferenceConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=OnlinePreferenceConfig.warmup_steps)
    ap.add_argument("--max_grad_norm", type=float, default=OnlinePreferenceConfig.max_grad_norm)

    ap.add_argument("--max_prompt_tokens", type=int, default=OnlinePreferenceConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=OnlinePreferenceConfig.max_response_tokens)
    ap.add_argument("--min_new_tokens", type=int, default=OnlinePreferenceConfig.min_new_tokens)
    ap.add_argument("--max_new_tokens", type=int, default=OnlinePreferenceConfig.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=OnlinePreferenceConfig.temperature)
    ap.add_argument("--top_p", type=float, default=OnlinePreferenceConfig.top_p)
    ap.add_argument("--top_k", type=int, default=OnlinePreferenceConfig.top_k)
    ap.add_argument("--repetition_penalty", type=float, default=OnlinePreferenceConfig.repetition_penalty)
    ap.add_argument("--reward_batch_size", type=int, default=OnlinePreferenceConfig.reward_batch_size)

    ap.add_argument("--eval_interval", type=int, default=OnlinePreferenceConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=OnlinePreferenceConfig.save_interval)
    ap.add_argument("--eval_max_new_tokens", type=int, default=OnlinePreferenceConfig.eval_max_new_tokens)
    ap.add_argument("--eval_temperature", type=float, default=OnlinePreferenceConfig.eval_temperature)
    ap.add_argument("--eval_top_p", type=float, default=OnlinePreferenceConfig.eval_top_p)
    ap.add_argument("--eval_batch_size", type=int, default=OnlinePreferenceConfig.eval_batch_size)
    ap.add_argument("--eval_limit", type=int, default=OnlinePreferenceConfig.eval_limit)
    ap.add_argument("--train_limit", type=int, default=OnlinePreferenceConfig.train_limit)

    ap.add_argument("--lora_r", type=int, default=OnlinePreferenceConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=OnlinePreferenceConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=OnlinePreferenceConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=OnlinePreferenceConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=OnlinePreferenceConfig.lora_bias)
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=OnlinePreferenceConfig.grad_checkpointing,
    )

    ap.add_argument("--wandb_project", type=str, default=OnlinePreferenceConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=OnlinePreferenceConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=OnlinePreferenceConfig.wandb_enabled,
    )
    ap.add_argument(
        "--log_interval",
        type=int,
        default=OnlinePreferenceConfig.log_interval,
        help="Log scalar training metrics to W&B every N steps (1 = every step).",
    )
    ap.add_argument("--sample_log_n", type=int, default=OnlinePreferenceConfig.sample_log_n)
    ap.add_argument("--sample_log_max_chars", type=int, default=OnlinePreferenceConfig.sample_log_max_chars)
    args = ap.parse_args()
    return OnlinePreferenceConfig(**vars(args))


def maybe_update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


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


def save_checkpoint(model: torch.nn.Module, cfg: OnlinePreferenceConfig, step: int) -> None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    meta = {
        "step": step,
        "model_type": "online_preference_rm_rl",
        "algo": cfg.algo,
        "beta": cfg.beta,
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


def _build_pairs_best_vs_worst(
    prompt_batch: Sequence[GenerationExample],
    completion_texts: Sequence[str],
    rm_scores: Sequence[float],
    *,
    group_size: int,
    step: int,
) -> List[PreferenceExample]:
    out: List[PreferenceExample] = []
    if len(prompt_batch) * group_size != len(completion_texts):
        raise ValueError("Rollout shape mismatch when building preference pairs.")
    score_t = torch.tensor(list(rm_scores), dtype=torch.float32).reshape(len(prompt_batch), group_size)
    for i, ex in enumerate(prompt_batch):
        scores = score_t[i]
        chosen_j = int(torch.argmax(scores).item())
        rejected_j = int(torch.argmin(scores).item())
        chosen_text = _normalize_completion_for_reward_scoring(str(completion_texts[i * group_size + chosen_j]))
        rejected_text = _normalize_completion_for_reward_scoring(str(completion_texts[i * group_size + rejected_j]))
        chosen_score = float(scores[chosen_j].item())
        rejected_score = float(scores[rejected_j].item())
        out.append(
            PreferenceExample(
                row_id=f"{ex.row_id}:online_pref:{step}:{i}",
                prompt_messages=list(ex.prompt_messages),
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                prompt_text=str(ex.prompt_text),
                chosen_text_full=chosen_text,
                rejected_text_full=rejected_text,
                score_chosen=chosen_score,
                score_rejected=rejected_score,
                avg_confidence=None,
                avg_preference_strength=None,
                avg_training_quality=None,
            )
        )
    return out


def _main_impl() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()
    if cfg.steps <= 0:
        raise ValueError(f"--steps must be >= 1, got {cfg.steps}")
    if cfg.prompt_batch_size <= 0:
        raise ValueError(f"--prompt_batch_size must be >= 1, got {cfg.prompt_batch_size}")
    if cfg.group_size <= 0:
        raise ValueError(f"--group_size must be >= 1, got {cfg.group_size}")
    if cfg.per_device_train_batch_size <= 0:
        raise ValueError(f"--per_device_train_batch_size must be >= 1, got {cfg.per_device_train_batch_size}")
    if cfg.grad_accum_steps <= 0:
        raise ValueError(f"--grad_accum_steps must be >= 1, got {cfg.grad_accum_steps}")
    if cfg.updates_per_rollout < 0:
        raise ValueError(f"--updates_per_rollout must be >= 0, got {cfg.updates_per_rollout}")
    if not cfg.reward_adapter_path:
        raise ValueError("--reward_adapter_path is required")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_online_preference_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rng = random.Random(cfg.seed)
    device, dtype = resolve_device_and_dtype()
    print(
        f"[setup] device={device} dtype={dtype} algo={cfg.algo} beta={cfg.beta} "
        f"policy={cfg.model_name} reward_model={cfg.reward_model_name}"
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
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )
    collator = PreferenceCollator(
        policy_tokenizer,
        max_prompt_tokens=cfg.max_prompt_tokens,
        max_response_tokens=cfg.max_response_tokens,
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
            "setup/trainable_params": float(loaded_policy.trainable_params),
            "setup/total_params": float(loaded_policy.total_params),
            "setup/trainable_fraction": float(loaded_policy.trainable_params / max(1, loaded_policy.total_params)),
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

    replay: Deque[PreferenceExample] = deque(maxlen=int(cfg.replay_size))
    start_time = time.time()
    optimizer_step = 0

    policy_model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, cfg.steps + 1):
        prompt_batch = _sample_prompt_batch(train_examples, cfg.prompt_batch_size, rng)
        rollout = sampler.rollout(
            policy_model=policy_model,
            prompt_messages=[ex.prompt_messages for ex in prompt_batch],
            task_names=["synthetic_instruction_following"] * len(prompt_batch),
            task_metas=[{"row_id": ex.row_id, "prompt_text": ex.prompt_text} for ex in prompt_batch],
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
        new_pairs = _build_pairs_best_vs_worst(
            prompt_batch,
            rollout.completion_texts,
            reward_scores,
            group_size=cfg.group_size,
            step=step,
        )
        for ex in new_pairs:
            replay.append(ex)

        update_metrics: Dict[str, float] = {}
        did_opt_step = False
        if len(replay) >= int(cfg.min_replay_size_to_train) and cfg.updates_per_rollout > 0:
            for _ in range(cfg.updates_per_rollout):
                microbatch_size = int(cfg.per_device_train_batch_size)
                for micro in range(int(cfg.grad_accum_steps)):
                    mb_examples = [replay[rng.randrange(len(replay))] for _ in range(microbatch_size)]
                    batch = collator(mb_examples).to(device)
                    need_reference = cfg.algo in {"dpo", "ipo", "aot"}
                    policy_scores, reference_scores = compute_policy_and_reference_scores(
                        policy_model,
                        batch=batch,
                        need_reference=need_reference,
                    )
                    loss_out = compute_offline_preference_loss(
                        algo=cfg.algo,
                        beta=cfg.beta,
                        policy_scores=policy_scores,
                        reference_scores=reference_scores,
                        example_weights=None,
                    )
                    (loss_out.loss / cfg.grad_accum_steps).backward()
                    update_metrics = loss_out.metrics

                maybe_update_warmup_lr(optimizer, cfg.lr, optimizer_step, cfg.warmup_steps)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm).item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                did_opt_step = True

                update_metrics = {
                    **{f"train/{k}": float(v) for k, v in update_metrics.items()},
                    "train/optimizer_step": float(optimizer_step),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train/gradient_global_norm_after_clipping": float(grad_norm),
                }

        rewards_t = torch.tensor(reward_scores, dtype=torch.float32)
        gaps = torch.tensor(
            [
                float((ex.score_chosen or 0.0) - (ex.score_rejected or 0.0))
                for ex in new_pairs
            ],
            dtype=torch.float32,
        )
        log_metrics = {
            "rollout/reward_model_score_mean": float(rewards_t.mean().item()),
            "rollout/reward_model_score_std": float(rewards_t.std(unbiased=False).item()),
            "rollout/reward_model_score_min": float(rewards_t.min().item()),
            "rollout/reward_model_score_max": float(rewards_t.max().item()),
            "rollout/pair_gap_mean_best_minus_worst": float(gaps.mean().item()) if gaps.numel() else 0.0,
            "rollout/pair_gap_std_best_minus_worst": float(gaps.std(unbiased=False).item()) if gaps.numel() else 0.0,
            "replay/size": float(len(replay)),
            "replay/new_pairs_added": float(len(new_pairs)),
            "time/seconds_since_start": float(time.time() - start_time),
            **get_cuda_memory_metrics(prefix="train"),
            **update_metrics,
        }
        if (cfg.log_interval <= 1) or (step % cfg.log_interval == 0) or (step == cfg.steps):
            logger.log(log_metrics, step=step)

        should_eval = (cfg.eval_interval > 0 and step % cfg.eval_interval == 0) or (step == cfg.steps)
        should_save = (cfg.save_interval > 0 and step % cfg.save_interval == 0) or (step == cfg.steps)
        if should_eval:
            print(f"[eval] running evaluation at step={step}")
            run_eval(step=step, phase=f"step_{step}")
        if should_save:
            print(f"[checkpoint] saving step={step}")
            save_checkpoint(policy_model, cfg, step=step)

        if did_opt_step:
            policy_model.train()

    logger.finish()


def main() -> None:
    return _main_impl()


if __name__ == "__main__":
    main()
