from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dqd_core import HAS_TORCH, HP, run_batch


ROOT = Path(__file__).resolve().parent
DEFAULT_ETAS = (0.02, 0.05, 0.1, 0.2, 0.4)
DEFAULT_BETAS = (1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_RANKS = (2, 4, 8)


@dataclass(frozen=True)
class MethodSpec:
    label: str
    mode: str
    ranks: Tuple[int, ...] = (4,)
    tune_beta: bool = True


@dataclass
class Task:
    name: str
    kind: str
    keys: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    frequency: Optional[np.ndarray] = None
    passes: int = 1


@dataclass(frozen=True)
class Condition:
    name: str
    query_mode: str
    alpha: float = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_keys(context: np.ndarray, dk: int) -> np.ndarray:
    centered = context - context.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    keys = centered @ vt[:dk].T
    return keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12)


def load_tasks(selected: Sequence[str]) -> List[Task]:
    tasks: List[Task] = []
    if "synthetic" in selected:
        tasks.append(Task("synthetic_2d", "synthetic"))
    definitions = {
        "word": ("word.npz", 128, 3),
        "module": ("module.npz", 24, 6),
    }
    for name in ("word", "module"):
        if name not in selected:
            continue
        filename, dk, passes = definitions[name]
        data = np.load(ROOT / filename)
        keys = semantic_keys(data["CK"], dk)
        values = np.asarray(data["CV"], dtype=np.float64)
        frequency = np.asarray(data["freq"], dtype=np.float64)
        frequency /= frequency.sum()
        tasks.append(Task(name, "discrete", keys, values, frequency, passes))
    return tasks


def _synthetic_parameters() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (
        rng.normal(scale=20.0, size=(2, 64)),
        rng.uniform(0.0, 2.0 * np.pi, 64),
        rng.normal(size=(8, 16)),
    )


SYN_W, SYN_B, SYN_A = _synthetic_parameters()


def synthetic_features(points: np.ndarray) -> np.ndarray:
    features = np.cos(np.atleast_2d(points) @ SYN_W + SYN_B)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def synthetic_targets(points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points)
    harmonic = np.arange(1, 5)
    basis = np.concatenate(
        [
            np.cos(2 * np.pi * np.outer(points[:, 0], harmonic)),
            np.sin(2 * np.pi * np.outer(points[:, 1], harmonic)),
            np.cos(2 * np.pi * np.outer(points[:, 0] + points[:, 1], harmonic)),
            np.sin(2 * np.pi * np.outer(points[:, 0] - points[:, 1], harmonic)),
        ],
        axis=1,
    )
    return basis @ SYN_A.T


def sample_spatial_queries(n: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "uniform":
        return rng.uniform(0.0, 1.0, size=(n, 2))
    centers_a = np.array([[0.22, 0.28], [0.70, 0.76]])
    centers_b = np.array([[0.76, 0.22], [0.30, 0.72]])
    centers = centers_b if mode == "hotspot_b" else centers_a
    choice = rng.integers(0, len(centers), n)
    return np.clip(centers[choice] + 0.06 * rng.normal(size=(n, 2)), 0.0, 1.0)


def make_synthetic_stream(
    seed: int,
    condition: Condition,
    query_every: int,
    stream_length: int,
    test_size: int,
    reference_size: int,
) -> Dict[str, np.ndarray]:
    write_rng = np.random.default_rng(10_000 + seed)
    query_rng = np.random.default_rng(20_000 + seed)
    test_rng = np.random.default_rng(30_000 + seed)
    reference_rng = np.random.default_rng(40_000 + seed)

    write_points = write_rng.uniform(0.0, 1.0, size=(stream_length, 2))
    write_keys = synthetic_features(write_points)
    write_values = synthetic_targets(write_points)
    write_values += 0.05 * write_rng.normal(size=write_values.shape)

    query_count = (stream_length - 1) // query_every + 1
    if condition.query_mode == "shift":
        first = query_count // 2
        query_points = np.concatenate(
            [
                sample_spatial_queries(first, query_rng, "hotspot_a"),
                sample_spatial_queries(query_count - first, query_rng, "hotspot_b"),
            ],
            axis=0,
        )
        test_mode = reference_mode = "hotspot_b"
    else:
        query_points = sample_spatial_queries(query_count, query_rng, condition.query_mode)
        test_mode = reference_mode = condition.query_mode

    test_points = sample_spatial_queries(test_size, test_rng, test_mode)
    reference_points = sample_spatial_queries(reference_size, reference_rng, reference_mode)
    return {
        "write_keys": write_keys,
        "write_values": write_values,
        "query_keys": synthetic_features(query_points),
        "test_keys": synthetic_features(test_points),
        "test_values": synthetic_targets(test_points),
        "reference_keys": synthetic_features(reference_points),
        "reference_values": synthetic_targets(reference_points),
        "reference_weights": None,
    }


def make_discrete_stream(
    task: Task,
    seed: int,
    condition: Condition,
    query_every: int,
    test_size: int,
) -> Dict[str, np.ndarray]:
    assert task.keys is not None and task.values is not None and task.frequency is not None
    if condition.query_mode == "uniform":
        probability = np.full(len(task.frequency), 1.0 / len(task.frequency))
    else:
        probability = np.power(task.frequency, condition.alpha)
        probability /= probability.sum()

    order_rng = np.random.default_rng(10_000 + seed)
    query_rng = np.random.default_rng(20_000 + seed)
    test_rng = np.random.default_rng(30_000 + seed)
    order = np.concatenate([order_rng.permutation(len(task.keys)) for _ in range(task.passes)])
    query_count = (len(order) - 1) // query_every + 1
    query_indices = query_rng.choice(len(task.keys), query_count, p=probability)
    test_indices = test_rng.choice(len(task.keys), test_size, p=probability)
    return {
        "write_keys": task.keys[order],
        "write_values": task.values[order],
        "query_keys": task.keys[query_indices],
        "test_keys": task.keys[test_indices],
        "test_values": task.values[test_indices],
        "reference_keys": task.keys,
        "reference_values": task.values,
        "reference_weights": probability,
    }


def make_stream(
    task: Task,
    seed: int,
    condition: Condition,
    query_every: int,
    stream_length: int,
    test_size: int,
    reference_size: int,
) -> Dict[str, np.ndarray]:
    if task.kind == "synthetic":
        return make_synthetic_stream(
            seed, condition, query_every, stream_length, test_size, reference_size
        )
    return make_discrete_stream(task, seed, condition, query_every, test_size)


def method_grid(
    spec: MethodSpec,
    etas: Sequence[float],
    betas: Sequence[float],
) -> List[HP]:
    if spec.mode == "weighted_linear_reference":
        return [(0.0, 0.0, 4)]
    if spec.mode == "plain":
        return [(eta, 0.0, 4) for eta in etas]
    chosen_betas = betas if spec.tune_beta else (0.0,)
    return [(eta, beta, rank) for eta in etas for beta in chosen_betas for rank in spec.ranks]


def suite_methods(suite: str) -> List[MethodSpec]:
    if suite == "main":
        return [
            MethodSpec("plain", "plain", tune_beta=False),
            MethodSpec("write_diag", "write_diag"),
            MethodSpec("write_trace", "write_trace", DEFAULT_RANKS),
            MethodSpec("dqd", "dqd", DEFAULT_RANKS),
            MethodSpec("full_query_trace", "full_query_trace"),
            MethodSpec("query_moment_reference", "query_moment_reference"),
            MethodSpec("weighted_linear_reference", "weighted_linear_reference", tune_beta=False),
        ]
    if suite == "ablation":
        methods = [
            MethodSpec("plain", "plain", tune_beta=False),
            MethodSpec("diag_query_trace", "diag_query_trace"),
            MethodSpec("last_query", "last_query"),
            MethodSpec("random_query_trace_r4", "random_query_trace", (4,)),
            MethodSpec("full_query_trace", "full_query_trace"),
        ]
        methods.extend(MethodSpec(f"dqd_r{rank}", "dqd", (rank,)) for rank in (1, 2, 4, 8, 16))
        return methods
    if suite == "robustness":
        return [
            MethodSpec("plain", "plain", tune_beta=False),
            MethodSpec("write_trace", "write_trace", DEFAULT_RANKS),
            MethodSpec("dqd", "dqd", DEFAULT_RANKS),
            MethodSpec("full_query_trace", "full_query_trace"),
        ]
    raise ValueError(f"unknown suite: {suite}")


def suite_conditions(suite: str, task: Task) -> List[Condition]:
    if suite == "main":
        if task.kind == "synthetic":
            return [Condition("concentrated", "hotspot_a"), Condition("uniform", "uniform", 0.0)]
        return [Condition("observed", "observed", 1.0), Condition("uniform", "uniform", 0.0)]
    if suite == "ablation":
        return [Condition("concentrated", "hotspot_a")] if task.kind == "synthetic" else [
            Condition("observed", "observed", 1.0)
        ]
    if suite == "robustness":
        if task.kind == "synthetic":
            return [
                Condition("concentrated", "hotspot_a"),
                Condition("uniform", "uniform", 0.0),
                Condition("moving_hotspot", "shift"),
            ]
        return [Condition(f"alpha_{alpha:.2f}", "observed", alpha) for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)]
    raise ValueError(f"unknown suite: {suite}")


def run_one(
    stream: Dict[str, np.ndarray],
    spec: MethodSpec,
    grid: Sequence[HP],
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    return run_batch(
        stream["write_keys"],
        stream["write_values"],
        stream["query_keys"],
        stream["test_keys"],
        stream["test_values"],
        spec.mode,
        grid,
        seed=seed,
        device=args.device,
        query_every=args.query_every,
        eta_u=args.eta_u,
        lam=args.lam,
        clip=args.clip,
        reference_keys=stream["reference_keys"],
        reference_values=stream["reference_values"],
        reference_weights=stream["reference_weights"],
    )


def critical_t_95(df: int) -> float:
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(0.975, df))
    except Exception:
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
            16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
            21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
            26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        }
        return table.get(df, 1.96)


def paired_p_value(t_stat: float, df: int) -> Optional[float]:
    try:
        from scipy.stats import t as student_t

        return float(2.0 * student_t.sf(abs(t_stat), df))
    except Exception:
        return None


def summarize(scores: np.ndarray) -> Dict[str, float]:
    mean = float(np.mean(scores))
    if len(scores) < 2:
        return {"mean": mean, "std": 0.0, "se": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(np.std(scores, ddof=1))
    se = std / math.sqrt(len(scores))
    radius = critical_t_95(len(scores) - 1) * se
    return {"mean": mean, "std": std, "se": se, "ci_low": mean - radius, "ci_high": mean + radius}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("main", "ablation", "robustness"), default="main")
    parser.add_argument("--device", default="cuda" if HAS_TORCH else "numpy")
    parser.add_argument("--datasets", default="synthetic,word,module")
    parser.add_argument(
        "--conditions",
        default="",
        help="optional comma-separated condition names, e.g. observed or moving_hotspot",
    )
    parser.add_argument("--val-seeds", type=int, default=4)
    parser.add_argument("--test-seeds", type=int, default=16)
    parser.add_argument("--val-seed-start", type=int, default=2001)
    parser.add_argument("--test-seed-start", type=int, default=10001)
    parser.add_argument("--query-every", type=int, default=3)
    parser.add_argument("--eta-u", type=float, default=0.05)
    parser.add_argument("--lam", type=float, default=0.999)
    parser.add_argument("--clip", type=float, default=1.9)
    parser.add_argument("--stream-length", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--reference-size", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="small grids and streams for pipeline testing only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    unknown = set(selected) - {"synthetic", "word", "module"}
    if unknown:
        raise SystemExit(f"unknown datasets: {sorted(unknown)}")
    if args.val_seeds < 1 or args.test_seeds < 1:
        raise SystemExit("seed counts must be positive")
    if args.quick:
        etas, betas = (0.05, 0.1), (1.0, 10.0)
        args.stream_length = min(args.stream_length, 256)
        args.test_size = min(args.test_size, 300)
        args.reference_size = min(args.reference_size, 500)
        args.val_seeds = min(args.val_seeds, 1)
        args.test_seeds = min(args.test_seeds, 2)
        args.val_seed_start = 101
        args.test_seed_start = 1
    else:
        etas, betas = DEFAULT_ETAS, DEFAULT_BETAS

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "results" / f"{args.suite}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    tasks = load_tasks(selected)
    methods = suite_methods(args.suite)
    val_seeds = tuple(args.val_seed_start + i for i in range(args.val_seeds))
    test_seeds = tuple(args.test_seed_start + i for i in range(args.test_seeds))
    all_seeds = val_seeds + test_seeds

    manifest = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "quick": args.quick,
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": None,
        "device": args.device,
        "val_seeds": val_seeds,
        "test_seeds": test_seeds,
        "etas": etas,
        "betas": betas,
        "query_every": args.query_every,
        "eta_u": args.eta_u,
        "lam": args.lam,
        "clip": args.clip,
        "stream_length": args.stream_length,
        "test_size": args.test_size,
        "reference_size": args.reference_size,
        "files": {},
        "datasets": {},
    }
    if HAS_TORCH:
        import torch

        manifest["torch"] = torch.__version__
        if args.device.startswith("cuda") and torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    for filename in ("dqd_core.py", "run_dqd_experiments.py", "word.npz", "module.npz"):
        path = ROOT / filename
        if path.exists():
            manifest["files"][filename] = sha256(path)
    for task in tasks:
        if task.kind == "discrete":
            manifest["datasets"][task.name] = {
                "keys": list(task.keys.shape),
                "values": list(task.values.shape),
                "frequency": list(task.frequency.shape),
                "passes": task.passes,
            }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    per_seed_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    paired_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    selected_hps: Dict[str, Dict[str, Dict[str, object]]] = {}
    cache: Dict[Tuple[str, str, int], Dict[str, np.ndarray]] = {}

    def stream_for(task: Task, condition: Condition, seed: int) -> Dict[str, np.ndarray]:
        key = (task.name, condition.name, seed)
        if key not in cache:
            cache[key] = make_stream(
                task, seed, condition, args.query_every, args.stream_length,
                args.test_size, args.reference_size,
            )
        return cache[key]

    requested_conditions = {x.strip() for x in args.conditions.split(",") if x.strip()}
    matched_conditions = set()
    for task in tasks:
        selected_hps[task.name] = {}
        for condition in suite_conditions(args.suite, task):
            if requested_conditions and condition.name not in requested_conditions:
                continue
            matched_conditions.add(condition.name)
            print(f"\n=== {args.suite} | {task.name} | {condition.name} ===", flush=True)
            selected_hps[task.name][condition.name] = {}
            condition_scores: Dict[str, np.ndarray] = {}
            for spec in methods:
                grid = method_grid(spec, etas, betas)
                started = time.time()
                validation = np.stack(
                    [run_one(stream_for(task, condition, seed), spec, grid, seed, args) for seed in val_seeds]
                )
                validation_mean = validation.mean(axis=0)
                validation_std = validation.std(axis=0, ddof=1) if len(val_seeds) > 1 else np.zeros(len(grid))
                for hp, mean_score, std_score in zip(grid, validation_mean, validation_std):
                    validation_rows.append(
                        {
                            "suite": args.suite,
                            "task": task.name,
                            "condition": condition.name,
                            "method": spec.label,
                            "eta": hp[0],
                            "beta": hp[1],
                            "rank": hp[2],
                            "validation_mean": float(mean_score),
                            "validation_std": float(std_score),
                            "n": len(val_seeds),
                        }
                    )
                best_index = int(np.argmin(validation_mean))
                best_hp = grid[best_index]
                test_scores = np.asarray(
                    [run_one(stream_for(task, condition, seed), spec, [best_hp], seed, args)[0] for seed in test_seeds]
                )
                elapsed = time.time() - started
                condition_scores[spec.label] = test_scores
                stats = summarize(test_scores)
                selected_hps[task.name][condition.name][spec.label] = {
                    "mode": spec.mode,
                    "eta": best_hp[0],
                    "beta": best_hp[1],
                    "rank": best_hp[2],
                    "validation_mean": float(validation_mean[best_index]),
                    "grid_size": len(grid),
                }
                for seed, score in zip(test_seeds, test_scores):
                    per_seed_rows.append(
                        {
                            "suite": args.suite,
                            "task": task.name,
                            "condition": condition.name,
                            "method": spec.label,
                            "seed": seed,
                            "score": float(score),
                            "eta": best_hp[0],
                            "beta": best_hp[1],
                            "rank": best_hp[2],
                        }
                    )
                plain_mean = float(np.mean(condition_scores["plain"])) if "plain" in condition_scores else math.nan
                gain = 100.0 * (plain_mean - stats["mean"]) / plain_mean if np.isfinite(plain_mean) else 0.0
                summary_rows.append(
                    {
                        "suite": args.suite,
                        "task": task.name,
                        "condition": condition.name,
                        "method": spec.label,
                        **stats,
                        "gain_vs_plain_percent": gain,
                        "eta": best_hp[0],
                        "beta": best_hp[1],
                        "rank": best_hp[2],
                        "grid_size": len(grid),
                        "elapsed_seconds": elapsed,
                    }
                )
                print(
                    f"  {spec.label:28s} {stats['mean']:.4f} +- {stats['std']:.4f}  "
                    f"{gain:+7.1f}%  eta={best_hp[0]:g} beta={best_hp[1]:g} r={best_hp[2]} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

            dqd_labels = [label for label in condition_scores if label == "dqd" or label.startswith("dqd_r")]
            for dqd_label in dqd_labels:
                for comparator in ("plain", "write_diag", "write_trace", "full_query_trace", "query_moment_reference"):
                    if comparator not in condition_scores or comparator == dqd_label:
                        continue
                    difference = condition_scores[dqd_label] - condition_scores[comparator]
                    diff_stats = summarize(difference)
                    if len(difference) > 1 and diff_stats["se"] > 0:
                        t_stat = diff_stats["mean"] / diff_stats["se"]
                        p_value = paired_p_value(t_stat, len(difference) - 1)
                    else:
                        t_stat, p_value = 0.0, None
                    row = {
                        "suite": args.suite,
                        "task": task.name,
                        "condition": condition.name,
                        "method": dqd_label,
                        "comparator": comparator,
                        "mean_difference": diff_stats["mean"],
                        "ci_low": diff_stats["ci_low"],
                        "ci_high": diff_stats["ci_high"],
                        "t_stat": t_stat,
                        "p_value": "" if p_value is None else p_value,
                        "wins": int(np.sum(difference < 0)),
                        "ties": int(np.sum(difference == 0)),
                        "n": len(difference),
                    }
                    paired_rows.append(row)
                    print(
                        f"    paired {dqd_label} vs {comparator}: diff={diff_stats['mean']:+.4f} "
                        f"95%CI=[{diff_stats['ci_low']:+.4f},{diff_stats['ci_high']:+.4f}] "
                        f"wins={row['wins']}/{row['n']}",
                        flush=True,
                    )

            write_csv(output / "per_seed.csv", per_seed_rows)
            write_csv(output / "summary.csv", summary_rows)
            write_csv(output / "paired.csv", paired_rows)
            write_csv(output / "validation_grid.csv", validation_rows)
            (output / "selected_hyperparameters.json").write_text(
                json.dumps(selected_hps, indent=2), encoding="utf-8"
            )

    if requested_conditions and not matched_conditions:
        raise SystemExit(f"none of the requested conditions were available: {sorted(requested_conditions)}")
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["conditions_filter"] = sorted(requested_conditions)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved complete results to {output}")


if __name__ == "__main__":
    main()
