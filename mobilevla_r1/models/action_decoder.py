import torch
from torch import nn
from torch.nn import functional as F


class ReasoningActionDecoder(nn.Module):
    """Equations 9–12: two learned queries, one eight-head cross attention."""

    def __init__(self, hidden_size, num_behaviors, num_heads=8):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Hidden dimension must be divisible by attention heads")
        self.queries = nn.Parameter(torch.empty(1, 2, hidden_size))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=0, batch_first=True)
        self.locomotion = nn.Sequential(nn.Linear(hidden_size, hidden_size // 4), nn.GELU(), nn.Linear(hidden_size // 4, 3))
        self.behavior = nn.Sequential(nn.Linear(hidden_size, hidden_size // 4), nn.GELU(), nn.Linear(hidden_size // 4, num_behaviors))

    def forward(self, hidden_states, context_mask):
        if not context_mask.any(dim=1).all():
            raise ValueError("Each decoder context needs observation or reasoning tokens")
        # Mask removes instruction, answer, EOS, and padding states from K/V.
        states, _ = self.attention(self.queries.expand(hidden_states.size(0), -1, -1),
                                   hidden_states, hidden_states,
                                   key_padding_mask=~context_mask.bool(), need_weights=False)
        return self.locomotion(states[:, 0]), self.behavior(states[:, 1])


def action_loss(velocity, behavior_logits, targets, behavior_ids, omega_weight=0.5):
    delta = (velocity.float() - targets.float()).abs()
    loc = (delta[:, :2].sum(-1) + omega_weight * delta[:, 2]).mean()
    beh = F.cross_entropy(behavior_logits.float(), behavior_ids)
    return loc, beh
