# Data Format

Store one JSON object per line. Convert existing MobileVLA-R1 annotations explicitly to the schema below before using the new training pipeline.

## Annotation Schema

| Field | Requirement |
|---|---|
| `id` | Unique string or number; required for evaluation |
| `split` | `train`, `val`, or `test`; training accepts only `train` |
| `granularity` | `episode`, `nav`, or `step` |
| `instruction` | Nonempty language instruction |
| `history` | Optional JSON-serializable state-action history used as context |
| `observation` | Dictionary containing RGB, depth, and/or point-cloud observations |
| `output` | Required for SFT; must follow `<think>nonempty reasoning</think><answer>nonempty answer</answer>` |
| `action.velocity` | Required for step-level SFT and GRPO: three finite values `[Vx, Vy, omega]` in m/s, m/s, and rad/s |
| `action.behavior` | Required for step-level SFT and GRPO; must belong to the configured behavior vocabulary |

The `long_sft` stage accepts episode and navigation records. The `action_sft` and `grpo` stages require step records with explicit action labels. Inference inputs do not require reference outputs or action labels.

## Observation Files

Each modality accepts either cached features or a list of raw file paths:

```json
{"rgb": {"features": "features/rgb.npy"}}
```

```json
{"rgb": {"paths": ["frames/frame1.jpg", "frames/frame2.jpg"]}}
```

Use one representation per modality. Cached and raw inputs may be mixed across modalities. File paths are resolved relative to `data.root`, or to the annotation file's parent directory when no root is supplied. Keep temporal ordering consistent across modalities.

### Cached Features

Save frozen encoder outputs before the trainable projector as `.npy` arrays with shape `[tokens, channels]` or `[frames, tokens, channels]`. Values must be finite, and channel dimensions must match `model.feature_dims`.

Frame sequences are flattened in order. The NaVILA RGB projector restores per-frame spatial grouping using `rgb_tokens_per_frame`. Do not pool RGB features in advance while retaining the original frame-token configuration. `token_budgets` limits the total projected tokens per modality and sample; longer sequences are pooled after projection.

Feature extraction writes paths relative to the output JSONL directory. Set `data.root` to that directory, or omit it. Training and inference must use the same encoder weights and preprocessing.

### Raw Depth and Point Clouds

Depth maps are two-dimensional `[H, W]` `.npy` arrays. The DAV2 adapter uses per-frame min-max normalization, three-channel replication, and ImageNet normalization before extracting pretrained encoder features. This preprocessing is an implementation choice because the manuscript does not specify the exact depth-map input transformation.

Geometric point clouds require calibrated depth scale. `scripts/depth_to_points.py` accepts camera intrinsics `fx`, `fy`, `cx`, and `cy` and unprojects metric depth into camera-frame XYZ coordinates. Its output contains XYZ only.

The PTv3 adapter expects arrays containing XYZ followed by the feature columns required by its checkpoint. If the checkpoint requires colors or normals, extract and append those features using the checkpoint's preprocessing protocol. Zero-filled substitutes do not provide equivalent features.

## Converting MobileVLA-R1 Annotations

The first three components of the original 12-dimensional Go2 action may be mapped to velocity after verifying their units. Posture and gait fields do not uniquely determine the shared behavior label. Supply explicit mappings or annotations based on the dataset's action definitions.

Episode-level summaries do not require action targets. Navigation text such as "move forward" does not specify a velocity or angular speed and should remain in `long_sft` unless explicit action labels are available. Step-level SFT and GRPO require those labels separately from the textual answer.

Training accepts only records marked `split: train`. Ensure the annotations actually originate from the training split; checking a split field cannot establish the provenance of the underlying trajectories.

## Navigation Logs

For `mobilevla-evaluate --navigation`, each JSONL record requires:

| Field | Description |
|---|---|
| `id` | Unique episode identifier |
| `final_distance` | Final geodesic distance to the goal |
| `oracle_distance` | Minimum geodesic distance to the goal during the episode |
| `shortest_path_length` | Reference shortest-path length |
| `traveled_distance` | Actual traveled distance |
| `stopped` | JSON boolean `true` or `false` indicating whether the agent issued the stop action |
| `success_distance` | Optional success threshold; defaults to 3.0 meters |
| `dtw_distance` | Optional DTW distance, required for nDTW |
| `reference_path_points` | Number of reference path points, required for nDTW |

Distances must come from the evaluation environment. Success defaults to `stopped && final_distance < 3.0`. Supply `success_distance` to use a different threshold. nDTW is reported only when every record includes both DTW fields.

SR, OS, SPL, and nDTW are reported on a 0–1 scale. These tools aggregate recorded measurements; trajectory generation and measurement must follow the relevant benchmark's evaluation protocol.

Prediction files and feature-cache destinations must be new paths. Existing outputs are not overwritten. Keep generated features and private annotations outside version control.
