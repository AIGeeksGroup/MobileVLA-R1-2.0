"""Cache frozen pre-projector modality features; use the same encoders at inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mobilevla_r1.config import load_config
from mobilevla_r1.data import CoTDataset
from mobilevla_r1.models.encoders import build_encoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = CoTDataset(args.input, config["model"]["behaviors"], root=args.root)
    output = Path(args.output).resolve()
    if output == Path(args.input).resolve():
        raise ValueError("Output must differ from input")
    if output.exists():
        raise FileExistsError("Feature annotation output already exists; choose a new path")
    directory = output.parent / (output.stem + "_features")
    directory.mkdir(parents=True, exist_ok=False)
    encoders = build_encoders(config["model"]["encoders"]).to(args.device).eval()
    with output.open("x", encoding="utf-8") as handle, torch.no_grad():
        for index, record in enumerate(dataset):
            observation = dataset.load_observation(record)
            cached = {}
            for modality, payload in observation.items():
                if "features" in payload:
                    value = payload["features"]
                else:
                    if modality not in encoders:
                        raise ValueError(f"Missing raw encoder for {modality}")
                    features = encoders[modality](payload["raw"])
                    value = torch.cat(list(features), dim=0).float().cpu().numpy()
                if value.shape[-1] != config["model"]["feature_dims"][modality] or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {modality} encoder feature shape/values")
                path = directory / f"{index:08d}_{modality}.npy"
                np.save(path, value, allow_pickle=False)
                cached[modality] = {"features": str(path.relative_to(output.parent))}
            handle.write(json.dumps({**record, "observation": cached}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
