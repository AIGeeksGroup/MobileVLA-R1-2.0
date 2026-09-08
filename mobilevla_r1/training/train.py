"""Serial microbatch execution keeps complete per-prompt GRPO groups intact."""
import argparse
import copy
import json
import math
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, broadcast_object_list
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from mobilevla_r1.config import load_config
from mobilevla_r1.data import CoTDataset
from mobilevla_r1.models.policy import MobileVLAPolicy
from mobilevla_r1.models.action_decoder import action_loss
from mobilevla_r1.training.objectives import rewards, group_advantages, grpo_loss


def single_record(batch):
    return batch[0]


def save(model, directory, stage, accelerator):
    accelerator.wait_for_everyone()
    status = [None]
    if accelerator.is_main_process:
        try:
            accelerator.unwrap_model(model).save_checkpoint(directory, stage)
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    broadcast_object_list(status)
    if status[0] is not None:
        raise RuntimeError(f"Checkpoint save failed: {status[0]}")
    accelerator.wait_for_everyone()


def supervised(model, dataset, config, accelerator):
    cfg, stage = config["training"], config["stage"]
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=single_record, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.get("learning_rate", 2e-4), weight_decay=cfg.get("weight_decay", 0.01))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    # Scheduler advances once per actual optimizer update, independently of world size.
    updates = math.ceil(len(loader) / accelerator.gradient_accumulation_steps) * cfg.get("epochs", 3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * cfg.get("warmup_ratio", 0.03)), updates)
    policy = accelerator.unwrap_model(model)
    model.train()
    optimizer.zero_grad()
    step = 0
    for epoch in range(cfg.get("epochs", 3)):
        for batch_index, record in enumerate(loader):
            observation = dataset.load_observation(record)
            with accelerator.accumulate(model):
                output = model(record=record, observation=observation, decode_action=stage == "action_sft")
                loss = -output["logps"].mean()
                loc = beh = loss.detach() * 0
                if stage == "action_sft":
                    target = torch.tensor([record["action"]["velocity"]], device=accelerator.device)
                    behavior = torch.tensor([config["model"]["behaviors"].index(record["action"]["behavior"])], device=accelerator.device)
                    loc, beh = action_loss(output["velocity"], output["behavior"], target, behavior, cfg.get("omega_weight", 0.5))
                    loss = loss + cfg.get("loc_weight", 1.0) * loc + cfg.get("beh_weight", 1.0) * beh
                finite = accelerator.reduce(torch.isfinite(loss.detach()).to(torch.int32), reduction="sum")
                if finite.item() != accelerator.num_processes:
                    raise FloatingPointError("Nonfinite SFT loss on at least one worker")
                accumulation = accelerator.gradient_accumulation_steps
                window_size = min(accumulation, len(loader) - (batch_index // accumulation) * accumulation)
                accelerator.backward(loss * accumulation / window_size)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0), error_if_nonfinite=True)
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                step += 1
                if stage == "action_sft":
                    policy.decoder_trained = True
                if step % cfg.get("log_every", 10) == 0:
                    accelerator.print(json.dumps({"step": step, "loss": loss.item(), "loc": loc.item(), "beh": beh.item()}))
        save(model, Path(cfg["output_dir"]) / f"epoch-{epoch + 1}", stage, accelerator)
    save(model, Path(cfg["output_dir"]) / "final", stage, accelerator)


def reinforcement(model, dataset, config, accelerator):
    if accelerator.num_processes != 1:
        raise ValueError("GRPO follows the paper's single-GPU setup; launch with one process")
    cfg = config["grpo"]
    train_cfg = config["training"]
    model.to(accelerator.device)
    # Fixed reference is the full SFT-initialized policy, including its SFT adapter.
    reference = copy.deepcopy(model.llm).requires_grad_(False).eval()
    reference.gradient_checkpointing_disable()
    model.eval()  # No dropout mismatch between sampling and rescoring; gradients still enabled.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=train_cfg.get("learning_rate", 1e-6), weight_decay=train_cfg.get("weight_decay", 0.01))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=single_record)
    iterator = iter(loader)
    groups = cfg.get("group_size", 8)
    prompts = cfg.get("prompts_per_update", 5)
    if prompts < 1:
        raise ValueError("prompts_per_update must be positive")
    for step in range(1, train_cfg.get("steps", 1000) + 1):
        optimizer.zero_grad()
        mean_loss, mean_reward, mean_kl = 0.0, 0.0, 0.0
        for _ in range(prompts):
            try:
                record = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                record = next(iterator)
            observation = dataset.load_observation(record)
            with torch.no_grad():
                prefix, nobs = model.prepare(record, observation)
                target = torch.tensor([record["action"]["velocity"]], device=model.device)
                behavior = torch.tensor([config["model"]["behaviors"].index(record["action"]["behavior"])], device=model.device)
                samples, reward_list = [], []
                for _ in range(groups):
                    # Temperature=1 and no top-p/top-k: old logps match sampling distribution.
                    ids = model.sample(prefix, cfg.get("max_new_tokens", 512), temperature=1.0)
                    old = model.score(prefix, nobs, ids)
                    ref = model.score(prefix, nobs, ids, llm=reference, decode_action=False)
                    reward, _ = rewards(old["velocity"], old["behavior"], target, behavior, [model.decode_text(ids)], cfg)
                    samples.append((ids, old["logps"].detach(), ref["logps"].detach()))
                    reward_list.append(reward)
                reward_values = torch.cat(reward_list)
                advantages = group_advantages(reward_values[None])[0]
            for index, (ids, old_logps, ref_logps) in enumerate(samples):
                current = model(prefix=prefix, observation_length=nobs, ids=ids, decode_action=False)
                loss, kl = grpo_loss(current["logps"], old_logps, ref_logps, advantages[index:index + 1],
                                    torch.ones_like(ids), cfg.get("beta", 0.04), cfg.get("clip", 0.2))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GRPO loss; optimizer update aborted")
                (loss / (prompts * groups)).backward()
                mean_loss += loss.item() / (prompts * groups)
                mean_kl += kl.item() / (prompts * groups)
            mean_reward += reward_values.mean().item() / prompts
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], train_cfg.get("max_grad_norm", 1.0), error_if_nonfinite=True)
        optimizer.step()
        if step % train_cfg.get("log_every", 10) == 0:
            print(json.dumps({"step": step, "loss": mean_loss, "reward": mean_reward, "kl": mean_kl}), flush=True)
        if step % train_cfg.get("save_every", 100) == 0:
            model.save_checkpoint(Path(train_cfg["output_dir"]) / f"step-{step}", "grpo")
    model.save_checkpoint(Path(train_cfg["output_dir"]) / "final", "grpo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="Overrides data.path")
    parser.add_argument("--init", help="Initialize from a prior-stage checkpoint (not optimizer resume)")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data:
        config["data"]["path"] = args.data
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    set_seed(config.get("seed", 42))
    stage = config.get("stage")
    if stage not in {"long_sft", "action_sft", "grpo"}:
        raise ValueError("A training stage is required")
    output_dir = Path(config["training"]["output_dir"])
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Training output directory must be new or empty; use --output-dir")
    initial = args.init or config.get("init_checkpoint")
    if stage in {"action_sft", "grpo"} and initial is None:
        raise ValueError(f"{stage} requires initialization from a prior-stage checkpoint")
    accumulation = config["training"].get("gradient_accumulation_steps", 1) if stage != "grpo" else 1
    accelerator = Accelerator(gradient_accumulation_steps=accumulation, mixed_precision="bf16" if config["model"].get("dtype") == "bfloat16" else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    dataset = CoTDataset(config["data"]["path"], config["model"]["behaviors"], stage, config["data"].get("root"))
    model = MobileVLAPolicy(config)
    if initial:
        model.load_checkpoint(initial)
    model.configure_stage(stage)
    if stage == "grpo":
        reinforcement(model, dataset, config, accelerator)
    else:
        supervised(model, dataset, config, accelerator)


if __name__ == "__main__":
    main()
