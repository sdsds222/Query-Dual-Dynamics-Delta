"""Formal WikiText-2 evaluation for Query-Driven Dual-Dynamics Delta (QD3)."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dqd_core import HAS_TORCH, run_batch
from run_dqd_experiments import (
    MethodSpec,
    method_grid,
    paired_p_value,
    sha256,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
METHODS = (
    MethodSpec("plain", "plain", tune_beta=False),
    MethodSpec("write_trace", "write_trace", (1, 2, 4)),
    MethodSpec("dqd", "dqd", (1, 2, 4)),
    MethodSpec("full_query_trace", "full_query_trace"),
)


def format_duration(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class Progress:
    """Small dependency-free progress bar with a work-weighted ETA."""

    def __init__(self, total, enabled=True, width=28):
        self.total = max(float(total), 1.0)
        self.enabled = enabled
        self.width = width
        self.done = 0.0
        self.started = time.perf_counter()

    def advance(self, amount, label):
        self.done = min(self.total, self.done + float(amount))
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started
        fraction = self.done / self.total
        filled = min(self.width, int(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else math.inf
        status = (
            f"\r[{bar}] {100*fraction:5.1f}%  "
            f"elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}  "
            f"{label[:42]:42s}"
        )
        print(status, end="", flush=True)

    def newline(self):
        if self.enabled:
            print()


def load_data(path: Path):
    data = np.load(path, allow_pickle=False)
    required = {"keys", "values", "frequency_val", "frequency_test"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    keys = np.asarray(data["keys"], dtype=np.float64)
    values = np.asarray(data["values"], dtype=np.float64)
    frequency_val = np.asarray(data["frequency_val"], dtype=np.float64)
    frequency_test = np.asarray(data["frequency_test"], dtype=np.float64)
    if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values):
        raise ValueError("keys and values must be aligned matrices")
    if any(x.shape != (len(keys),) or np.any(x < 0) for x in (frequency_val, frequency_test)):
        raise ValueError("frequency vectors must be non-negative and aligned with keys")
    if not all(np.isfinite(x).all() for x in (keys, values, frequency_val, frequency_test)):
        raise ValueError("prepared arrays contain non-finite values")
    frequency_val /= frequency_val.sum()
    frequency_test /= frequency_test.sum()
    return keys, values, frequency_val, frequency_test


def make_stream(keys, values, probability, seed, passes, query_every, test_size):
    order_rng = np.random.default_rng(10_000 + seed)
    query_rng = np.random.default_rng(20_000 + seed)
    test_rng = np.random.default_rng(30_000 + seed)
    order = np.concatenate([order_rng.permutation(len(keys)) for _ in range(passes)])
    query_count = (len(order) - 1) // query_every + 1
    query_indices = query_rng.choice(len(keys), query_count, p=probability)
    test_indices = test_rng.choice(len(keys), test_size, p=probability)
    return {
        "write_keys": keys[order],
        "write_values": values[order],
        "query_keys": keys[query_indices],
        "test_keys": keys[test_indices],
        "test_values": values[test_indices],
    }


def evaluate(stream, spec, grid, seed, args):
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
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"wikitext2_qd3.npz")
    parser.add_argument("--device", default="cuda" if HAS_TORCH else "numpy")
    parser.add_argument("--conditions", default="observed,uniform")
    parser.add_argument("--val-seeds", type=int, default=8)
    parser.add_argument("--test-seeds", type=int, default=16)
    parser.add_argument("--val-seed-start", type=int, default=4001)
    parser.add_argument("--test-seed-start", type=int, default=30001)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--query-every", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--eta-u", type=float, default=0.05)
    parser.add_argument("--lam", type=float, default=0.999)
    parser.add_argument("--clip", type=float, default=1.9)
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"wikitext2_blind_v1")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.val_seeds, args.test_seeds, args.passes, args.query_every, args.test_size) < 1:
        raise SystemExit("seed counts, passes, query_every, and test_size must be positive")
    requested = tuple(x.strip() for x in args.conditions.split(",") if x.strip())
    unknown = set(requested) - {"observed", "uniform"}
    if unknown or not requested:
        raise SystemExit(f"invalid conditions: {sorted(unknown)}")

    etas = (0.02, 0.05, 0.1)
    betas = (1.0, 3.0, 10.0, 30.0)
    if args.quick:
        etas, betas = (0.05, 0.1), (1.0, 10.0)
        args.val_seeds = 1
        args.test_seeds = 2
        args.val_seed_start = 101
        args.test_seed_start = 1
        args.passes = 1
        args.test_size = min(args.test_size, 300)

    data_path = args.data.resolve()
    keys, values, frequency_val, frequency_test = load_data(data_path)
 
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    val_seeds = tuple(args.val_seed_start + i for i in range(args.val_seeds))
    test_seeds = tuple(args.test_seed_start + i for i in range(args.test_seeds))
    manifest = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "quick": args.quick,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "device": args.device,
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "source_hashes": {
            name: sha256(ROOT/name)
            for name in ("dqd_core.py", "run_wikitext2_experiment.py", "prepare_wikitext2.py")
        },
        "shape": {"keys": list(keys.shape), "values": list(values.shape)},
        "conditions": list(requested),
        "val_seeds": list(val_seeds),
        "test_seeds": list(test_seeds),
        "etas": list(etas),
        "betas": list(betas),
        "passes": args.passes,
        "query_every": args.query_every,
        "test_size": args.test_size,
        "eta_u": args.eta_u,
        "lam": args.lam,
        "clip": args.clip,
    }
    if HAS_TORCH:
        import torch
        manifest["torch"] = torch.__version__
        if args.device.startswith("cuda") and torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    validation_rows, per_seed_rows, summary_rows, paired_rows = [], [], [], []
    selected = {}
    work_per_condition = sum(
        args.val_seeds * len(method_grid(spec, etas, betas)) + args.test_seeds
        for spec in METHODS
    )
    progress = Progress(
        len(requested) * work_per_condition,
        enabled=not args.no_progress,
    )
    for condition in requested:
        print(f"\n=== wikitext2 | {condition} ===", flush=True)
        val_probability = frequency_val if condition == "observed" else np.full(len(keys), 1/len(keys))
        test_probability = frequency_test if condition == "observed" else np.full(len(keys), 1/len(keys))
        validation_streams = {
            seed: make_stream(keys, values, val_probability, seed, args.passes,
                              args.query_every, args.test_size)
            for seed in val_seeds
        }
        test_streams = {
            seed: make_stream(keys, values, test_probability, seed, args.passes,
                              args.query_every, args.test_size)
            for seed in test_seeds
        }
        scores_by_method = {}
        selected[condition] = {}
        for spec in METHODS:
            grid = method_grid(spec, etas, betas)
            started = time.perf_counter()
            validation_parts = []
            for index, seed in enumerate(val_seeds, start=1):
                validation_parts.append(evaluate(validation_streams[seed], spec, grid, seed, args))
                progress.advance(
                    len(grid),
                    f"{condition}/{spec.label} validation {index}/{len(val_seeds)}",
                )
            validation = np.stack(validation_parts)
            means = validation.mean(axis=0)
            best_index = int(np.argmin(means))
            best_hp = grid[best_index]
            for hp, score in zip(grid, means):
                validation_rows.append({
                    "condition": condition, "method": spec.label,
                    "eta": hp[0], "beta": hp[1], "rank": hp[2],
                    "validation_mean": float(score), "n": len(val_seeds),
                })
            test_parts = []
            for index, seed in enumerate(test_seeds, start=1):
                test_parts.append(evaluate(test_streams[seed], spec, [best_hp], seed, args)[0])
                progress.advance(
                    1,
                    f"{condition}/{spec.label} test {index}/{len(test_seeds)}",
                )
            scores = np.asarray(test_parts)
            elapsed = time.perf_counter() - started
            scores_by_method[spec.label] = scores
            stats = summarize(scores)
            selected[condition][spec.label] = {
                "eta": best_hp[0], "beta": best_hp[1], "rank": best_hp[2],
                "validation_mean": float(means[best_index]), "grid_size": len(grid),
            }
            for seed, score in zip(test_seeds, scores):
                per_seed_rows.append({
                    "condition": condition, "method": spec.label, "seed": seed,
                    "score": float(score), "eta": best_hp[0],
                    "beta": best_hp[1], "rank": best_hp[2],
                })
            plain_mean = float(scores_by_method["plain"].mean())
            gain = 100*(plain_mean-stats["mean"])/plain_mean
            summary_rows.append({
                "condition": condition, "method": spec.label, **stats,
                "gain_vs_plain_percent": gain, "eta": best_hp[0],
                "beta": best_hp[1], "rank": best_hp[2],
                "grid_size": len(grid), "elapsed_seconds": elapsed,
            })
            progress.newline()
            print(
                f"  {spec.label:22s} {stats['mean']:.4f} +- {stats['std']:.4f} "
                f"{gain:+7.1f}% eta={best_hp[0]:g} beta={best_hp[1]:g} "
                f"r={best_hp[2]} ({elapsed:.0f}s)", flush=True,
            )

        for comparator in ("plain", "write_trace", "full_query_trace"):
            difference = scores_by_method["dqd"] - scores_by_method[comparator]
            stats = summarize(difference)
            t_stat = stats["mean"]/stats["se"] if stats["se"] > 0 else 0.0
            p_value = paired_p_value(t_stat, len(difference)-1) if len(difference) > 1 else None
            row = {
                "condition": condition,
                "method": "dqd",
                "comparator": comparator,
                "mean_difference": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "t_stat": t_stat,
                "p_value": "" if p_value is None else p_value,
                "wins": int(np.sum(difference < 0)),
                "ties": int(np.sum(difference == 0)),
                "n": len(difference),
            }
            paired_rows.append(row)
            print(
                f"    paired dqd vs {comparator}: diff={stats['mean']:+.4f} "
                f"95%CI=[{stats['ci_low']:+.4f},{stats['ci_high']:+.4f}] "
                f"wins={row['wins']}/{row['n']}",
                flush=True,
            )

        write_csv(output/"validation_grid.csv", validation_rows)
        write_csv(output/"per_seed.csv", per_seed_rows)
        write_csv(output/"summary.csv", summary_rows)
        write_csv(output/"paired.csv", paired_rows)
        (output/"selected_hyperparameters.json").write_text(
            json.dumps(selected, indent=2), encoding="utf-8"
        )

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved complete results to {output}")


if __name__ == "__main__":
    main()
