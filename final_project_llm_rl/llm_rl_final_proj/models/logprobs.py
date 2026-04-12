from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_per_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. Shape: [B, L-1]."""
    with torch.set_grad_enabled(enable_grad):
        # TODO(student): run the causal LM, align logits with the next-token targets,
        # and return per-token log-probabilities of the observed tokens.
        # Hint: use F.cross_entropy with reduction='none' for memory efficiency.
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,)
        logits = out.logits

        logits_next = logits[:, :-1, :] 
        targets = input_ids[:, 1:]

        B, T, V = logits_next.shape
        logits_flat = logits_next.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)

        if attention_mask is not None:
            valid = attention_mask[:, 1:].to(dtype=torch.bool)  
            ignore_index = -100
            targets_flat = targets_flat.clone()
            targets_flat[~valid.reshape(B * T)] = ignore_index
        else:
            ignore_index = -100

        nll_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=ignore_index,)
        logp = (-nll_flat).reshape(B, T)
        return logp


def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    del pad_token_id
    # TODO(student): build a float mask of shape [B, L-1] that selects only completion tokens.
    # Be careful about the one-token shift between logits[:, :-1] and input_ids[:, 1:].
    mask = attention_mask[:, 1:].to(dtype=torch.float32) 
   
    start = max(0, int(prompt_input_len) - 1)
    if start > 0: mask[:, :start] = 0.0
    return mask

def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions.

    Uses estimator: exp(delta) - delta - 1 where delta = log p_ref(a) - log p_new(a).
    """
    # TODO(student): implement the sampled-token KL proxy used throughout the codebase.
    # You should mask out non-completion positions and return a scalar batch mean.
    delta = ref_logprobs - new_logprobs
    if log_ratio_clip is not None:
        delta = delta.clamp(min=-float(log_ratio_clip), max=float(log_ratio_clip))

    kl_tok = torch.exp(delta) - delta - 1.0
    return masked_mean(kl_tok, mask, eps=eps)
