import torch
from torch.nn import functional as F
from mobilevla_r1.schema import structured_parts


def token_logps(logits, ids):
    # Row-wise avoids materializing a second [B,L,vocab] float32 tensor.
    return torch.stack([F.log_softmax(row.float(), dim=-1).gather(-1, target[:, None]).squeeze(-1)
                        for row, target in zip(logits, ids)])


def rewards(velocity, behavior_logits, target_velocity, target_behavior, texts, cfg):
    scales = velocity.new_tensor([cfg.get("velocity_scale", 1.0)] * 2 + [cfg.get("yaw_scale", 1.0)])
    if (scales <= 0).any():
        raise ValueError("Reward scales must be positive")
    predicted, target = velocity.float() / scales, target_velocity.float() / scales
    movement = F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)
    # Manuscript leaves zero-vector cosine undefined. Explicit bounded convention.
    zero_p, zero_t = predicted.norm(dim=-1) < 1e-8, target.norm(dim=-1) < 1e-8
    movement = torch.where(zero_p & zero_t, torch.ones_like(movement), movement)
    behavior = (behavior_logits.argmax(-1) == target_behavior).float()
    fmt = movement.new_tensor([float(structured_parts(text) is not None) for text in texts])
    total = cfg.get("movement_weight", 1.0) * movement + cfg.get("behavior_weight", 1.0) * behavior + cfg.get("format_weight", 0.2) * fmt
    return total, {"movement": movement, "behavior": behavior, "format": fmt}


def group_advantages(reward):
    """Input [number of prompts, candidates]; never normalize across prompts."""
    if reward.ndim != 2 or reward.size(1) < 2:
        raise ValueError("Rewards must be [prompts, group_size >= 2]")
    return (reward - reward.mean(-1, keepdim=True)) / (reward.std(-1, keepdim=True, unbiased=False) + 1e-8)


def grpo_loss(logps, old_logps, ref_logps, advantages, mask, beta=0.04, clip=0.2):
    ratio = (logps - old_logps.detach()).exp()
    surrogate = torch.minimum(ratio * advantages[:, None], ratio.clamp(1 - clip, 1 + clip) * advantages[:, None])
    # Nonnegative sampled KL estimator; reference and rollout probabilities fixed.
    delta = ref_logps.detach() - logps
    kl = delta.exp() - delta - 1
    lengths = mask.sum(-1).clamp_min(1)
    loss = ((-surrogate + beta * kl) * mask).sum(-1) / lengths
    return loss.mean(), ((kl * mask).sum(-1) / lengths).mean().detach()
