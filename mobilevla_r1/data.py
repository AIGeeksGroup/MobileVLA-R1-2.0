"""JSONL dataset, strict schema checks and lazy multimodal loading."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from mobilevla_r1.config import load_config
from mobilevla_r1.schema import structured_parts


def validate_record(record, behaviors, stage=None):
    if not isinstance(record, dict):
        raise ValueError("Each record must be a JSON object")
    if not isinstance(record.get("instruction"), str) or not record["instruction"].strip():
        raise ValueError("Nonempty instruction required")
    granularity = record.get("granularity")
    if granularity not in {"episode", "nav", "step"}:
        raise ValueError("granularity must be episode, nav, or step")
    if stage == "long_sft" and granularity not in {"episode", "nav"}:
        raise ValueError("long_sft accepts only episode/nav records")
    if stage in {"action_sft", "grpo"} and granularity != "step":
        raise ValueError("action_sft/grpo require step records")
    if stage is not None and record.get("split") != "train":
        raise ValueError("Training requires explicit split=train")
    if record.get("split") not in {"train", "val", "test"}:
        raise ValueError("Explicit split=train/val/test required")
    if "output" in record and structured_parts(record["output"]) is None:
        raise ValueError("Invalid <think>...</think><answer>...</answer> output")
    if stage in {"long_sft", "action_sft"} and "output" not in record:
        raise ValueError("SFT requires output")
    if "action" in record:
        action = record["action"]
        if not isinstance(action, dict):
            raise ValueError("action must be a mapping")
        velocity = action.get("velocity", [])
        if not isinstance(velocity, list) or len(velocity) != 3 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in velocity):
            raise ValueError("action.velocity must contain finite [Vx,Vy,omega]")
        if action.get("behavior") not in behaviors:
            raise ValueError(f"Unknown behavior: {action.get('behavior')}")
    elif stage in {"action_sft", "grpo"}:
        raise ValueError("Explicit action targets required; text is not parsed into control targets")
    observation = record.get("observation", {})
    if not isinstance(observation, dict) or not observation:
        raise ValueError("At least one observation modality is required")
    for modality, item in observation.items():
        if modality not in {"rgb", "depth", "point"}:
            raise ValueError(f"Unknown modality {modality}")
        if not isinstance(item, dict) or set(item) not in ({"features"}, {"paths"}):
            raise ValueError("Each modality needs either features=.npy or paths=[raw files]")
        if "paths" in item and (not isinstance(item["paths"], list) or not item["paths"]):
            raise ValueError("Raw paths must be a nonempty list")
        paths = [item["features"]] if "features" in item else item["paths"]
        if not all(isinstance(path, str) and path.strip() for path in paths):
            raise ValueError("Observation paths must be nonempty strings")


class CoTDataset(Dataset):
    def __init__(self, path, behaviors, stage=None, root=None):
        self.path = Path(path).resolve()
        self.root = Path(root).resolve() if root else self.path.parent
        self.rows = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_record(record, behaviors, stage)
                except ValueError as error:
                    raise ValueError(f"{self.path}:{line_number}: {error}") from error
                self.rows.append(record)
        if not self.rows:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def load_observation(self, record):
        result = {}
        for modality, item in record["observation"].items():
            if "features" in item:
                path = self.root / item["features"]
                value = np.load(path, allow_pickle=False)
                if value.ndim == 3:
                    value = value.reshape(-1, value.shape[-1])
                if value.ndim != 2 or not value.shape[0] or not np.isfinite(value).all():
                    raise ValueError(f"{path}: cached features must be finite [tokens, channels]")
                result[modality] = {"features": value}
            else:
                values = []
                for name in item["paths"]:
                    path = self.root / name
                    if modality == "rgb":
                        with Image.open(path) as image:
                            values.append(image.convert("RGB").copy())
                    else:
                        values.append(np.load(path, allow_pickle=False))
                result[modality] = {"raw": values}
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--root")
    parser.add_argument("--stage", choices=["long_sft", "action_sft", "grpo"])
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = CoTDataset(args.data, cfg["model"]["behaviors"], args.stage, args.root)
    for record in dataset:
        missing = set(cfg["model"].get("required_modalities", ["rgb", "depth", "point"])) - set(record["observation"])
        if missing:
            raise ValueError(f"Missing required observation modalities: {sorted(missing)}")
    if args.check_files:
        for record in dataset:
            observation = dataset.load_observation(record)
            for modality, payload in observation.items():
                if "features" in payload and payload["features"].shape[-1] != cfg["model"]["feature_dims"][modality]:
                    raise ValueError(f"Feature dimension mismatch for {modality}")
    print(json.dumps({"valid_records": len(dataset), "files_checked": args.check_files}))


if __name__ == "__main__":
    main()
