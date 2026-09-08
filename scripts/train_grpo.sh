#!/usr/bin/env bash
set -euo pipefail
python -m mobilevla_r1.training.train --config configs/grpo.yaml "$@"
