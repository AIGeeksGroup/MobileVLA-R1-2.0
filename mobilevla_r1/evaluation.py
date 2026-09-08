"""Offline action fidelity and recorded navigation metrics, without a simulator."""
import argparse
import json
import math
from pathlib import Path
from mobilevla_r1.schema import structured_parts, TaskAction


def keyed_records(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Evaluation rows must be JSON objects")
        key = row.get("id")
        if type(key) not in (str, int, float) or (isinstance(key, float) and not math.isfinite(key)) or key in records:
            raise ValueError("Evaluation records need unique non-null IDs")
        records[key] = row
    if not records:
        raise ValueError("Empty evaluation data")
    return records


def action_metrics(predictions, targets):
    if not predictions:
        raise ValueError("Empty predictions")
    if predictions.keys() != targets.keys():
        raise ValueError("Prediction and target ID sets must match exactly")
    errors, correct, formatted = [0.0] * 3, 0, 0
    for key, row in predictions.items():
        predicted, target = row["action"], targets[key]["action"]
        TaskAction(*predicted["velocity"], predicted["behavior"])
        TaskAction(*target["velocity"], target["behavior"])
        for i in range(3):
            errors[i] += abs(predicted["velocity"][i] - target["velocity"][i])
        correct += predicted["behavior"] == target["behavior"]
        formatted += structured_parts(row.get("output", "")) is not None
    count = len(predictions)
    return {"count": count, "velocity_mae": [x / count for x in errors],
            "behavior_accuracy": correct / count, "format_accuracy": formatted / count}


def navigation_metrics(episodes):
    """Use simulator-measured geodesic distances; do not invent Euclidean NE.

    Each row: final_distance, shortest_path_length, traveled_distance, stopped,
    oracle_distance, and optional dtw_distance, reference_path_points.
    """
    count = len(episodes)
    if not count:
        raise ValueError("No navigation episodes")
    sums = {"NE": 0.0, "SR": 0.0, "OS": 0.0, "SPL": 0.0}
    ndtw = []
    for row in episodes.values():
        threshold = row.get("success_distance", 3.0)
        distance, shortest, traveled = (row[k] for k in ("final_distance", "shortest_path_length", "traveled_distance"))
        if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (distance, shortest, traveled, row["oracle_distance"])) or type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Invalid navigation distances")
        if type(row["stopped"]) is not bool:
            raise ValueError("stopped must be a JSON boolean")
        success = row["stopped"] and distance < threshold
        sums["NE"] += distance
        sums["SR"] += success
        sums["OS"] += row["oracle_distance"] < threshold
        sums["SPL"] += success * shortest / max(shortest, traveled, 1e-8)
        if "dtw_distance" in row:
            if type(row["reference_path_points"]) is not int or row["reference_path_points"] <= 0 or type(row["dtw_distance"]) not in (int, float) or not math.isfinite(row["dtw_distance"]) or row["dtw_distance"] < 0:
                raise ValueError("Invalid DTW inputs")
            ndtw.append(math.exp(-row["dtw_distance"] / (threshold * row["reference_path_points"])))
    result = {k: v / count for k, v in sums.items()}
    if ndtw:
        if len(ndtw) != count:
            raise ValueError("nDTW requires DTW values for every episode")
        result["nDTW"] = sum(ndtw) / count
    return {"count": count, **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--targets")
    parser.add_argument("--navigation", action="store_true")
    args = parser.parse_args()
    predictions = keyed_records(args.predictions)
    if args.navigation:
        result = navigation_metrics(predictions)
    else:
        if not args.targets:
            parser.error("--targets required for action evaluation")
        result = action_metrics(predictions, keyed_records(args.targets))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
