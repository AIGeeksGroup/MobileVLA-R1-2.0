"""Local tensor checkpoint I/O with explicit formats and no unsafe fallback."""
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_tensors(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
    elif path.suffix in {".pt", ".pth", ".bin"}:
        # Also enforce the floor when the package was installed with --no-deps.
        version = tuple(int(part) for part in torch.__version__.split(".")[:2])
        if version < (2, 10):
            raise RuntimeError("PyTorch >=2.10 is required for weights-only checkpoint loading")
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("Use a .safetensors, .pt, .pth, or .bin tensor checkpoint")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    return state


def save_modules(modules, path):
    tensors = {f"{name}.{key}": value.detach().cpu().contiguous()
               for name, module in modules.items() for key, value in module.state_dict().items()}
    save_file(tensors, str(path))


def load_modules(modules, path):
    state = load_file(str(path), device="cpu")
    expected = {f"{name}.{key}" for name, module in modules.items() for key in module.state_dict()}
    if set(state) != expected:
        raise ValueError("Checkpoint module parameters do not match the configured model")
    for name, module in modules.items():
        prefix = name + "."
        module.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                if key.startswith(prefix)}, strict=True)
