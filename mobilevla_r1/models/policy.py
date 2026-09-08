import json
import os
import shutil
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from mobilevla_r1.models.action_decoder import ReasoningActionDecoder
from mobilevla_r1.models.encoders import build_encoders
from mobilevla_r1.models.navila_projector import MultimodalProjector
from mobilevla_r1.training.objectives import token_logps
from mobilevla_r1.checkpoints import save_modules, load_modules


class MobileVLAPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.settings = config
        cfg = config["model"]
        base = Path(cfg["base_model"])
        llm_path = base / "llm" if (base / "llm").is_dir() else base
        self.tokenizer = AutoTokenizer.from_pretrained(str(llm_path), use_fast=True, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        llm = AutoModelForCausalLM.from_pretrained(str(llm_path), torch_dtype=dtype,
                                                   attn_implementation=cfg.get("attention", "sdpa"),
                                                   local_files_only=True, trust_remote_code=False, use_safetensors=True)
        hidden = llm.config.hidden_size
        lora = cfg.get("lora", {})
        self.llm = get_peft_model(llm, LoraConfig(r=lora.get("rank", 16), lora_alpha=lora.get("alpha", 32),
            lora_dropout=0.0, target_modules=lora.get("targets", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            bias="none", task_type="CAUSAL_LM"))
        if cfg.get("gradient_checkpointing", True):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.encoders = build_encoders(cfg.get("encoders", {}))
        self.projectors = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
                                        for name, dim in cfg["feature_dims"].items()})
        projector_path = cfg.get("rgb_projector")
        if projector_path:
            from transformers import PretrainedConfig
            # Reuse the original NaVILA projector class and checkpoint key layout.
            self.projectors["rgb"] = MultimodalProjector.from_pretrained(projector_path,
                PretrainedConfig(mm_hidden_size=cfg["feature_dims"]["rgb"], hidden_size=hidden),
                local_files_only=True, use_safetensors=True)
        self.modality_types = nn.Embedding(3, hidden)
        self.decoder = ReasoningActionDecoder(hidden, len(cfg["behaviors"]), cfg.get("decoder_heads", 8))
        self.projectors.to(dtype=dtype)
        self.modality_types.to(dtype=dtype)
        self.decoder.to(dtype=dtype)
        self.encoders.requires_grad_(False)
        self.encoders.eval()
        self.decoder_trained = False
        self.loaded_stage = None
        self._set_stop_ids()

    def _set_stop_ids(self):
        self.stop_ids = {self.tokenizer.eos_token_id}
        end_turn = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if end_turn is not None and end_turn != self.tokenizer.unk_token_id:
            self.stop_ids.add(end_turn)
        self.stop_ids.discard(None)
        if not self.stop_ids or self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define EOS and padding tokens")

    @property
    def device(self):
        return next(self.llm.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.encoders.eval()
        return self

    def configure_stage(self, stage):
        if stage not in {"long_sft", "action_sft", "grpo"}:
            raise ValueError("Unknown training stage")
        if stage == "long_sft":
            self.decoder_trained = False
        if stage == "grpo" and not self.decoder_trained:
            raise ValueError("GRPO requires a checkpoint from action_sft")
        self.decoder.requires_grad_(stage == "action_sft")
        # Freeze projections in GRPO: only LoRA policy weights are optimized.
        self.projectors.requires_grad_(stage != "grpo")
        self.modality_types.requires_grad_(stage != "grpo")
        self.encoders.requires_grad_(False)

    def observation_embeddings(self, observation):
        tokens = []
        cfg = self.settings["model"]
        required = cfg.get("required_modalities", ["rgb", "depth", "point"])
        missing = set(required) - set(observation)
        if missing:
            raise ValueError(f"Missing required modalities: {sorted(missing)}")
        for index, name in enumerate(("rgb", "depth", "point")):
            if name not in observation:
                continue
            payload = observation[name]
            if "features" in payload:
                value = torch.as_tensor(payload["features"], device=self.device)
            else:
                if name not in self.encoders:
                    raise ValueError(f"Raw {name} needs an encoder configured in model.encoders")
                with torch.no_grad():
                    encoded = self.encoders[name](payload["raw"])
                value = torch.cat(list(encoded), dim=0)
            if value.ndim != 2 or value.shape[-1] != cfg["feature_dims"][name] or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name} features: {tuple(value.shape)}")
            projector = self.projectors[name]
            dtype = self.modality_types.weight.dtype
            # Projection before pooling preserves mlp_downsample spatial grouping.
            if name == "rgb" and isinstance(projector, MultimodalProjector):
                per_frame = cfg.get("rgb_tokens_per_frame")
                if not per_frame or value.shape[0] % per_frame:
                    raise ValueError("rgb_tokens_per_frame must match pre-projector frame token count")
                value = projector(value.to(dtype).reshape(-1, per_frame, value.shape[-1])).flatten(0, 1)
            else:
                value = projector(value.to(dtype))
            budget = cfg.get("token_budgets", {}).get(name)
            if budget and value.size(0) > budget:
                value = F.adaptive_avg_pool1d(value.T[None], budget)[0].T
            tokens.append(value + self.modality_types.weight[index])
        if not tokens:
            raise ValueError("No observation tokens")
        return torch.cat(tokens)[None]

    def prompt_ids(self, instruction, history=None):
        content = instruction
        if history:
            content += "\nState/action history:\n" + json.dumps(history, ensure_ascii=False)
        content += "\nRespond with <think>...</think><answer>...</answer>."
        messages = [{"role": "system", "content": "You are a mobile robot reasoning about observations and task-level actions."},
                    {"role": "user", "content": content}]
        if self.tokenizer.chat_template:
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            ids = self.tokenizer("System: " + messages[0]["content"] + "\nUser: " + content + "\nAssistant:", add_special_tokens=True).input_ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def prepare(self, record, observation):
        obs = self.observation_embeddings(observation)
        prompt = self.prompt_ids(record["instruction"], record.get("history"))
        embeddings = self.llm.get_input_embeddings()(prompt)
        return torch.cat([obs, embeddings], dim=1), obs.size(1)

    def completion_ids(self, text):
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def reasoning_mask(self, ids):
        """Map actual sampled token IDs to the interior of <think>, without retokenizing."""
        result = torch.zeros_like(ids, dtype=torch.bool)
        for row, sequence in enumerate(ids.tolist()):
            text = self.tokenizer.decode(sequence, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            start = text.find("<think>")
            if start < 0:
                continue
            start += len("<think>")
            end = text.find("</think>", start)
            # Malformed candidates remain scoreable but get format reward zero.
            if end < 0:
                end = text.find("<answer>", start)
                end = len(text) if end < 0 else end
            previous = 0
            for column in range(len(sequence)):
                boundary = len(self.tokenizer.decode(sequence[:column + 1], skip_special_tokens=False,
                                                     clean_up_tokenization_spaces=False))
                if previous >= start and boundary <= end and boundary > previous:
                    result[row, column] = True
                previous = boundary
        return result

    def score(self, prefix, observation_length, ids, completion_mask=None, llm=None, decode_action=True):
        llm = self.llm if llm is None else llm
        batch = ids.size(0)
        prefix = prefix.expand(batch, -1, -1)
        tail = llm.get_input_embeddings()(ids)
        inputs = torch.cat([prefix, tail], dim=1)
        if inputs.size(1) > self.settings["model"].get("max_length", 8192):
            raise ValueError("Context exceeds model.max_length; reduce token budgets/history/output length")
        if completion_mask is None:
            completion_mask = torch.ones_like(ids, dtype=torch.bool)
        attention = torch.cat([torch.ones(prefix.shape[:2], device=ids.device, dtype=torch.bool), completion_mask], dim=1)
        output = llm(inputs_embeds=inputs, attention_mask=attention, output_hidden_states=decode_action,
                     return_dict=True, use_cache=False)
        start = prefix.size(1) - 1
        logps = token_logps(output.logits[:, start:start + ids.size(1)], ids)
        velocity = behavior = None
        if decode_action:
            context = torch.zeros(inputs.shape[:2], dtype=torch.bool, device=ids.device)
            context[:, :observation_length] = True
            context[:, prefix.size(1):] = self.reasoning_mask(ids) & completion_mask
            velocity, behavior = self.decoder(output.hidden_states[-1], context)
        return {"logps": logps, "velocity": velocity, "behavior": behavior}

    @torch.no_grad()
    def sample(self, prefix, max_new_tokens=512, temperature=1.0, greedy=False):
        """Single candidate with KV caching; same unwarped policy used for GRPO ratios."""
        if self.training:
            raise ValueError("Call policy.eval() before sampling with KV caching")
        if max_new_tokens < 1 or temperature <= 0:
            raise ValueError("Positive generation length and temperature required")
        if prefix.size(1) + max_new_tokens > self.settings["model"].get("max_length", 8192):
            raise ValueError("Prompt plus generation exceeds context budget")
        output = self.llm(inputs_embeds=prefix, use_cache=True, return_dict=True)
        generated = []
        for step in range(max_new_tokens):
            probability = F.softmax(output.logits[:, -1].float() / temperature, dim=-1)
            token = probability.argmax(-1, keepdim=True) if greedy else torch.multinomial(probability, 1)
            generated.append(token)
            if token.item() in self.stop_ids:
                break
            length = prefix.size(1) + step + 1
            output = self.llm(input_ids=token, past_key_values=output.past_key_values,
                attention_mask=torch.ones(1, length, dtype=torch.long, device=self.device),
                use_cache=True, return_dict=True)
        return torch.cat(generated, dim=1)

    def decode_text(self, ids):
        return self.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def forward(self, record=None, observation=None, prefix=None, observation_length=None, ids=None, decode_action=True):
        if prefix is None:
            prefix, observation_length = self.prepare(record, observation)
        if ids is None:
            ids = self.completion_ids(record["output"])
        return self.score(prefix, observation_length, ids, decode_action=decode_action)

    def save_checkpoint(self, path, stage):
        from mobilevla_r1.config import save_config
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Checkpoint already exists: {path}; choose a new output directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
        try:
            self.llm.save_pretrained(temporary / "adapter", safe_serialization=True)
            self.tokenizer.save_pretrained(temporary / "tokenizer")
            save_modules({"projectors": self.projectors, "modality_types": self.modality_types,
                          "decoder": self.decoder}, temporary / "modules.safetensors")
            save_config(self.settings, temporary)
            (temporary / "metadata.json").write_text(json.dumps({"stage": stage,
                "decoder_trained": self.decoder_trained, "behaviors": self.settings["model"]["behaviors"],
                "version": "2.0.0", "checkpoint_format": 1}, indent=2), encoding="utf-8")
            os.rename(temporary, path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def load_checkpoint(self, path):
        from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
        from safetensors.torch import load_file
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("checkpoint_format") != 1:
            raise ValueError("Unsupported checkpoint format; expected format 1 with safetensors modules")
        if type(metadata.get("decoder_trained")) is not bool:
            raise ValueError("Invalid checkpoint decoder_trained flag")
        if metadata["behaviors"] != self.settings["model"]["behaviors"]:
            raise ValueError("Checkpoint behavior vocabulary/order differs from configuration")
        # Strict compatibility is checked against saved model settings before loading.
        import yaml
        saved = yaml.safe_load((path / "config.yaml").read_text())["model"]
        for key in ("feature_dims", "lora", "decoder_heads", "token_budgets", "rgb_tokens_per_frame", "required_modalities"):
            if saved.get(key) != self.settings["model"].get(key):
                raise ValueError(f"Checkpoint model setting mismatch: {key}")
        tokenizer = AutoTokenizer.from_pretrained(str(path / "tokenizer"), use_fast=True,
                                                  local_files_only=True, trust_remote_code=False)
        if tokenizer.get_vocab() != self.tokenizer.get_vocab():
            raise ValueError("Checkpoint and base-model token vocabularies differ")
        self.tokenizer = tokenizer
        self._set_stop_ids()
        state = load_file(str(path / "adapter" / "adapter_model.safetensors"), device="cpu")
        expected = get_peft_model_state_dict(self.llm)
        if set(state) != set(expected):
            raise ValueError("Adapter checkpoint has missing or unexpected parameters")
        if any(state[key].shape != expected[key].shape for key in state):
            raise ValueError("Adapter checkpoint parameter dimensions differ")
        result = set_peft_model_state_dict(self.llm, state)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")
        load_modules({"projectors": self.projectors, "modality_types": self.modality_types,
                      "decoder": self.decoder}, path / "modules.safetensors")
        self.decoder_trained = metadata["decoder_trained"]
        self.loaded_stage = metadata["stage"]
