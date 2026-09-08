import argparse
import json
from pathlib import Path
import torch
from mobilevla_r1.config import load_config
from mobilevla_r1.data import CoTDataset
from mobilevla_r1.models.policy import MobileVLAPolicy
from mobilevla_r1.schema import TaskAction


@torch.no_grad()
def predict(policy, record, observation, max_new_tokens=512):
    if not policy.decoder_trained:
        raise ValueError("Action inference requires a trained action_sft/grpo checkpoint")
    policy.eval()
    prefix, nobs = policy.prepare(record, observation)
    ids = policy.sample(prefix, max_new_tokens, greedy=True)
    output = policy.score(prefix, nobs, ids)
    velocity = output["velocity"][0].float().cpu().tolist()
    probabilities = output["behavior"].float().softmax(-1)[0]
    behavior_id = probabilities.argmax().item()
    behavior = policy.settings["model"]["behaviors"][behavior_id]
    action = TaskAction(*velocity, behavior)
    return {"id": record.get("id"), "output": policy.decode_text(ids), "action": action.as_dict(),
            "behavior_probability": probabilities[behavior_id].item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional relocated paths; must match checkpoint architecture")
    parser.add_argument("--input", required=True, help="JSONL observations")
    parser.add_argument("--root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Prediction output already exists; choose a new path")
    config = load_config(args.config or Path(args.checkpoint) / "config.yaml")
    policy = MobileVLAPolicy(config)
    policy.load_checkpoint(args.checkpoint)
    policy.to(args.device).eval()
    dataset = CoTDataset(args.input, config["model"]["behaviors"], root=args.root)
    path = Path(args.output)
    if path.resolve() == Path(args.input).resolve():
        raise ValueError("Output must differ from input")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in dataset:
            result = predict(policy, record, dataset.load_observation(record), args.max_new_tokens)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
