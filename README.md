# MobileVLA-R1 2.0: RL-Enhanced Reasoning for Mobile Robot Control

Code implementation for the paper:

> **MobileVLA-R1 2.0: RL-Enhanced Reasoning for Mobile Robot Control**
>
> [Ting Huang](https://github.com/Believeht029)\*, Yue Huang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*†, [Shuicheng Yan](https://yanshuicheng.info/), and [Hao Tang](https://ha0tang.github.io/)‡
>
> \*Equal contribution. †Project lead. ‡Corresponding author.

---

## 🏃 Introduction

MobileVLA-R1 2.0 connects structured embodied reasoning with executable mobile robot control. Building on MobileVLA-R1, the framework introduces a reasoning-conditioned action decoder that maps observation and reasoning representations to continuous locomotion targets and discrete behavior primitives.

The framework follows a perception–reasoning–action pipeline:

- **Multimodal perception:** RGB, depth, and point-cloud features are projected into a shared language-model hidden space.
- **Structured reasoning:** A NaVILA-initialized backbone generates embodied reasoning in the `<think>...</think><answer>...</answer>` format.
- **Action decoding:** Two learnable queries aggregate observation and reasoning states through cross-attention to predict planar velocities, yaw velocity, and a task-level behavior.
- **Embodiment-specific execution:** Robot controllers translate task-level predictions into locomotion and manipulation routines.

Training combines supervised CoT alignment with offline Group Relative Policy Optimization (GRPO). Episode- and navigation-level data establish long-horizon reasoning, while step-level data jointly align reasoning and action prediction. GRPO subsequently improves the reasoning policy using movement, behavior, and format rewards with a frozen action decoder.

## 📦 Data Preparation

The pipeline supports three observation modalities: RGB frames, depth maps, and point clouds. Prepare JSONL annotations following the [data format specification](docs/data.md). The [examples](examples/) directory contains schema examples; observation files and pretrained weights must be supplied separately.

1. **Prepare CoT annotations.** Use episode/nav records for long-horizon alignment and step records with explicit velocity and behavior labels for action alignment and GRPO. Training records must have `split: train`.
2. **Prepare observations.** Provide ordered RGB images, depth maps as two-dimensional `.npy` arrays, and point clouds with the feature columns required by the chosen encoder checkpoint.
3. **Configure frozen encoders or cached features.** The default configurations use cached features. For raw observations, copy the mapping from `configs/encoders.example.yaml` into `model.encoders` and configure the pretrained encoder paths.
4. **Set dataset paths.** Update `data.path` and `data.root` in the training configurations. Observation paths are resolved relative to `data.root`, or to the JSONL directory when no root is specified.

Cached features must be extracted **before the trainable projectors**, with shape `[tokens, feature_dim]` or `[frames, tokens, feature_dim]`. Use identical encoder weights and preprocessing for training and inference.

```bash
# Optional: extract and cache frozen encoder features.
python scripts/cache_features.py --config configs/long_sft.yaml \
  --input data/raw_train.jsonl --root data --output data/cached_train.jsonl

# Validate step-level annotations and feature files.
mobilevla-validate --config configs/action_sft.yaml \
  --data data/step_train.jsonl --root data --stage action_sft --check-files
```

Raw depth and point-cloud encoding requires the corresponding Depth Anything V2 and Pointcept installations and pretrained weights. The PTv3 architecture, input feature columns, and CUDA extensions must match the selected checkpoint.

## ⚙️ Environment Setup

Use Python 3.10–3.12 on Linux with an NVIDIA CUDA GPU. Install PyTorch `2.10.x` and torchvision `0.25.x` for your CUDA environment, then install the project:

```bash
python -m pip install -e .
```

Dependencies are specified in `pyproject.toml`. The PyTorch version floor addresses known checkpoint-loading vulnerabilities; see [SECURITY.md](SECURITY.md). The updated dependency combination has not been runtime-validated. This implementation uses Transformers 4.44.2 and Accelerate. Use a separate environment from MobileVLA-R1; the original Transformers 4.37.2 patches are not used by these entry points.

SFT supports multiple GPUs through Accelerate. GRPO uses one GPU and scores candidates sequentially. Memory requirements depend on context length, encoder configuration, and the additional frozen reference model.

### Pretrained Checkpoints

Prepare a local NaVILA checkpoint with the following layout:

```text
checkpoints/navila/
  llm/             # Hugging Face LLaMA3 safetensors, configuration, and tokenizer
  vision_tower/    # SigLIP/CLIP safetensors and image processor for raw RGB
  mm_projector/    # NaVILA projector configuration and safetensors weights
```

Model loading uses local files only and requires safetensors for Hugging Face weights. Obtain compatible exports from the checkpoint provider; no conversion or unrestricted pickle fallback is performed automatically.

Update these settings consistently across `configs/long_sft.yaml`, `configs/action_sft.yaml`, and `configs/grpo.yaml`:

| Setting | Description |
|---|---|
| `base_model`, `rgb_projector` | Paths to the pretrained backbone and RGB projector |
| `feature_dims` | Output dimensions of the frozen modality encoders |
| `rgb_tokens_per_frame` | Number of RGB tokens per frame before projection |
| `token_budgets` | Maximum projected tokens per modality and sample |
| `behaviors` | Shared behavior vocabulary in a fixed order |
| `required_modalities` | Required observations, defaulting to RGB, depth, and point clouds |

Feature dimensions, token budgets, and the behavior vocabulary in the example configurations are configurable implementation choices. Match them to your checkpoints and annotations. Each behavior class requires suitable supervision; adding a class name does not train the corresponding skill. The new action decoder must be trained with step-level SFT before action inference or GRPO.

## 🚀 Training

### Stage 1: Supervised CoT Alignment

First align episode- and navigation-level reasoning:

```bash
accelerate launch --multi_gpu --num_processes 4 \
  -m mobilevla_r1.training.train --config configs/long_sft.yaml
```

Then jointly train step-level reasoning and the action decoder:

```bash
accelerate launch --multi_gpu --num_processes 4 \
  -m mobilevla_r1.training.train --config configs/action_sft.yaml \
  --init outputs/long_sft/final
```

Alternatively, run both supervised phases with:

```bash
NUM_PROCESSES=4 bash scripts/train_sft.sh
```

For single-GPU SFT, launch `python -m mobilevla_r1.training.train --config ...` directly.

The default SFT settings use AdamW, a learning rate of `2e-4`, weight decay of `0.01`, a warmup ratio of `0.03`, cosine scheduling, and LoRA with rank `16` and alpha `32`. Each supervised phase defaults to three epochs and can be configured independently.

### Stage 2: GRPO Reinforcement Learning

Initialize from the step-level SFT checkpoint:

```bash
python -m mobilevla_r1.training.train --config configs/grpo.yaml \
  --init outputs/action_sft/final
```

Or use the provided launcher:

```bash
bash scripts/train_grpo.sh
```

Each update uses five inputs with eight sampled candidates per input. Rewards combine movement similarity, behavior accuracy, and structured output validity with weights `1.0`, `1.0`, and `0.2`. The default configuration runs for 1,000 steps with a learning rate of `1e-6`, KL coefficient `0.04`, and clipping coefficient `0.2`.

The action decoder, projectors, and modality embeddings remain frozen during GRPO; only the backbone LoRA parameters are updated. The frozen reference includes the SFT adapter. Sampling uses temperature `1.0` without top-k or top-p filtering so rollout probabilities match policy scoring.

Checkpoints contain the LoRA adapter, projectors, modality embeddings, action decoder, tokenizer, configuration, and behavior vocabulary. Tensor weights use safetensors. Completed checkpoints are published from a temporary directory; existing checkpoint directories are never overwritten. Format 1 is required; earlier experimental `modules.pt` checkpoints are not loaded by this version. `--init` loads weights for stage initialization or warm starts; it does not restore optimizer and random-number states for exact training resumption.

## 🤖 Inference

Run action prediction from a trained SFT or GRPO checkpoint:

```bash
mobilevla-infer --checkpoint outputs/grpo/final \
  --input data/inference.jsonl --root data \
  --output outputs/predictions.jsonl
```

Inference generates structured reasoning, then uses the learned decoder to produce `[Vx, Vy, omega]` and a behavior label from observation and reasoning hidden states. The textual answer is retained for interpretation; physical actions come from the decoder predictions.

Prediction and feature-cache commands require new output paths and refuse to overwrite existing files. Training requires a new or empty output directory.

For robot integration, use `CallbackController` and `closed_loop` in `mobilevla_r1/controllers.py`. Register platform-specific velocity, stop, and behavior routines. G1 reaching, grasping, lifting, transporting, and placing require fixed controller implementations that handle trajectories, inverse kinematics, and completion detection.

## 📊 Evaluation

Evaluate action predictions against labeled records:

```bash
mobilevla-evaluate --predictions outputs/predictions.jsonl \
  --targets data/step_eval.jsonl
```

Aggregate navigation metrics from recorded environment measurements:

```bash
mobilevla-evaluate --predictions outputs/navigation_episodes.jsonl --navigation
```

The tools report offline action metrics and navigation-log metrics. Full VLN-CE or QUARD evaluation requires the corresponding environments, assets, and observation/control integration.

This implementation has undergone static checks only. Training, inference, and robot execution have not been run, and benchmark results have not been reproduced with this code.

## 😘 Acknowledgement

We thank the authors of [MobileVLA-R1](https://github.com/AIGeeksGroup/MobileVLA-R1) and [NaVILA](https://github.com/AnjieCheng/NaVILA) for their research and open-source implementations.