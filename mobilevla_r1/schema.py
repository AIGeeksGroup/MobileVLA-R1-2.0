"""Text is annotation/format supervision, never a physical action parser."""
import math
import re
from dataclasses import dataclass

FORMAT = re.compile(r"\s*<think>\s*(?P<think>.+?)\s*</think>\s*<answer>\s*(?P<answer>.+?)\s*</answer>\s*", re.DOTALL)
TAGS = ("<think>", "</think>", "<answer>", "</answer>")


def structured_parts(text):
    if not isinstance(text, str):
        return None
    match = FORMAT.fullmatch(text)
    if match is None or any(text.count(tag) != 1 for tag in TAGS):
        return None
    if not all(match.group(name).strip() for name in ("think", "answer")):
        return None
    return match.group("think"), match.group("answer")


@dataclass(frozen=True)
class TaskAction:
    vx: float
    vy: float
    omega: float
    behavior: str

    def __post_init__(self):
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (self.vx, self.vy, self.omega)):
            raise ValueError("Action contains nonfinite velocity")
        if not isinstance(self.behavior, str) or not self.behavior.strip():
            raise ValueError("Behavior is required")

    def as_dict(self):
        return {"velocity": [self.vx, self.vy, self.omega], "behavior": self.behavior}
