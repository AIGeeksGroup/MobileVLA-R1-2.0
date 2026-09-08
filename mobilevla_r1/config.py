"""Configuration parsing and early validation, before model allocation."""
import math
from pathlib import Path
import yaml


def _positive(value, name, integer=False, allow_zero=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} {'integer' if integer else 'number'}")


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Configuration must contain a model mapping")
    model = config["model"]
    names = model.get("behaviors")
    if not isinstance(names, list) or not names or not all(isinstance(x, str) and x.strip() for x in names) or len(set(names)) != len(names):
        raise ValueError("behaviors must contain unique nonempty strings")
    if not isinstance(model.get("base_model"), str) or not model["base_model"]:
        raise ValueError("model.base_model must be a local checkpoint path")
    if model.get("dtype", "bfloat16") not in {"bfloat16", "float32"}:
        raise ValueError("Supported model dtypes are bfloat16 and float32")
    if model.get("attention", "sdpa") not in {"sdpa", "eager"}:
        raise ValueError("Supported attention implementations are sdpa and eager")
    dims = model.get("feature_dims")
    if not isinstance(dims, dict) or not dims or set(dims) - {"rgb", "depth", "point"}:
        raise ValueError("feature_dims must map known modalities to channel dimensions")
    for name, dim in dims.items():
        _positive(dim, f"feature_dims.{name}", integer=True)
    required = model.get("required_modalities", ["rgb", "depth", "point"])
    if not isinstance(required, list) or not required or not all(isinstance(x, str) and x in dims for x in required) or len(set(required)) != len(required):
        raise ValueError("required_modalities must be a unique nonempty subset of feature_dims")
    for name, budget in model.get("token_budgets", {}).items():
        if name not in dims:
            raise ValueError(f"Unknown token budget modality: {name}")
        _positive(budget, f"token_budgets.{name}", integer=True)
    for name in ("max_length", "decoder_heads", "rgb_tokens_per_frame"):
        if name in model:
            _positive(model[name], f"model.{name}", integer=True)
    encoder_types = {"rgb": {"rgb", "cached"}, "depth": {"depth_anything_v2", "cached"},
                     "point": {"point_transformer_v3", "cached"}}
    for name, spec in model.get("encoders", {}).items():
        if name not in dims or not isinstance(spec, dict) or spec.get("type") not in encoder_types[name]:
            raise ValueError(f"Invalid encoder specification for {name}")
    if config.get("stage") not in {"long_sft", "action_sft", "grpo", None}:
        raise ValueError("Unknown training stage")
    train = config.get("training", {})
    for name in ("epochs", "steps", "save_every", "log_every", "gradient_accumulation_steps"):
        if name in train:
            _positive(train[name], f"training.{name}", integer=True)
    for name in ("learning_rate", "max_grad_norm"):
        if name in train:
            _positive(train[name], f"training.{name}")
    for name in ("weight_decay", "loc_weight", "beh_weight", "omega_weight", "warmup_ratio"):
        if name in train:
            _positive(train[name], f"training.{name}", allow_zero=True)
    if train.get("warmup_ratio", 0.03) > 1:
        raise ValueError("warmup_ratio must be in [0,1]")
    grpo = config.get("grpo", {})
    for name in ("group_size", "prompts_per_update", "max_new_tokens"):
        if name in grpo:
            _positive(grpo[name], f"grpo.{name}", integer=True)
    if grpo.get("group_size", 8) < 2:
        raise ValueError("GRPO requires at least two candidates per input")
    for name in ("velocity_scale", "yaw_scale"):
        if name in grpo:
            _positive(grpo[name], f"grpo.{name}")
    for name in ("beta", "movement_weight", "behavior_weight", "format_weight", "clip"):
        if name in grpo:
            _positive(grpo[name], f"grpo.{name}", allow_zero=True)
    if not 0 < grpo.get("clip", 0.2) < 1:
        raise ValueError("GRPO clip must be in (0,1)")
    return config


def save_config(config, directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
