#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. Configure per-stage settings in the YAML files.
processes="${NUM_PROCESSES:-4}"
launcher=(accelerate launch --num_processes "$processes")
if [ "$processes" -gt 1 ]; then
  launcher+=(--multi_gpu)
fi
"${launcher[@]}" -m mobilevla_r1.training.train --config configs/long_sft.yaml
"${launcher[@]}" -m mobilevla_r1.training.train --config configs/action_sft.yaml
