"""Calibrated pinhole unprojection from depth to camera-frame coordinates."""
import argparse
import math
from pathlib import Path
import numpy as np


def depth_to_points(depth, fx, fy, cx, cy, scale=1.0):
    if depth.ndim != 2 or not all(math.isfinite(x) for x in (fx, fy, cx, cy, scale)) or fx <= 0 or fy <= 0 or scale <= 0:
        raise ValueError("Need [H,W] metric/scaled depth and valid camera calibration")
    y, x = np.indices(depth.shape)
    z = depth.astype(np.float32) / scale
    valid = np.isfinite(z) & (z > 0)
    return np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], axis=-1)[valid].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", required=True)
    parser.add_argument("--output", required=True)
    for name in ("fx", "fy", "cx", "cy"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    points = depth_to_points(np.load(args.depth, allow_pickle=False), args.fx, args.fy, args.cx, args.cy, args.scale)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.save(handle, points, allow_pickle=False)


if __name__ == "__main__":
    main()
