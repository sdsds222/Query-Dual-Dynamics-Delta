from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from dqd_core import HAS_TORCH, HP, run_batch
from run_dqd_experiments import (
    DEFAULT_BETAS,
    DEFAULT_ETAS,
    Condition,
    Task,
    load_tasks,
    make_stream,
    paired_p_value,
    sha256,
    suite_conditions,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PER_TRACE_RANKS = (1, 2, 4)
DEFAULT_BASELINES = {
    "main": (ROOT / "results" / "main_blind_v1",),
    "robustness": (
        ROOT / "results" / "robustness_blind_v1",
        ROOT / "results" / "robustness_word_remaining_v1",
        ROOT / "results" / "robustness_module_remaining_v1",
    ),
}
COMPARATORS = ("plain", "write_trace", "dqd", "full_query_trace")


class Progress:
    def __init__(self, total: int) -> None:
        self.total = max(total, 1)
        self.done = 0
        self.started = time.time()

    def update(self, label: str) -> None:
        self.done += 1
        elapsed = time.time() - self.started
        rate = self.done / max(elapsed, 1e-9)
        remaining = (self.total - self.done) / max(rate, 1e-9)
        width = 28
        filled = min(width, int(width * self.done / self.total))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r[{bar}] {100.0 * self.done / self.total:5.1f}%  "
            f"elapsed {format_time(elapsed)}  ETA {format_time(remaining)}  {label}",
            end="",
            flush=True,
        )
        if self.done == self.total:
            print(flush=True)


def format_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def parse_numbers(text: str, cast) -> Tuple:
    values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("the list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the state-matched Dual-Trace extension and reuse prior baselines."
    )
    parser.add_argument("--suite", choices=("main", "robustness"), default="main")
    parser.add_argument("--device", default="cuda" if HAS_TORCH else "numpy")
    parser.add_argument("--datasets", default="synthetic,word,module")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--val-seeds", type=int, default=8)
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
    parser.add_argument("--etas", default=",".join(map(str, DEFAULT_ETAS)))
    parser.add_argument("--betas", default=",".join(map(str, DEFAULT_BETAS)))
    parser.add_argument("--per-trace-ranks", default="1,2,4")
    parser.add_argument("--baseline-results", type=Path, action="append", default=[])
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def load_baselines(paths: Sequence[Path]) -> Dict[Tuple[str, str, str, int], float]:
    scores: Dict[Tuple[str, str, str, int], float] = {}
    for directory in paths:
        source = directory / "per_seed.csv"
        if not source.exists():
            raise SystemExit(f"missing baseline file: {source}")
        for row in read_csv(source):
            key = (row["task"], row["condition"], row["method"], int(row["seed"]))
            score = float(row["score"])
            if key in scores and not math.isclose(scores[key], score, rel_tol=0.0, abs_tol=1e-12):
                raise SystemExit(f"conflicting baseline score for {key}")
            scores[key] = score
    return scores


def validate_baseline_manifests(
    paths: Sequence[Path], args: argparse.Namespace, val_seeds: Sequence[int], test_seeds: Sequence[int]
) -> None:
    expected = {
        "query_every": args.query_every,
        "eta_u": args.eta_u,
        "lam": args.lam,
        "clip": args.clip,
        "stream_length": args.stream_length,
        "test_size": args.test_size,
        "reference_size": args.reference_size,
    }
    available_val = set()
    available_test = set()
    for directory in paths:
        source = directory / "manifest.json"
        if not source.exists():
            raise SystemExit(f"missing baseline manifest: {source}")
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if manifest.get("suite") != args.suite:
            raise SystemExit(f"baseline suite mismatch in {source}")
        for name, value in expected.items():
            if name in manifest and manifest[name] != value:
                raise SystemExit(
                    f"baseline parameter mismatch in {source}: {name}={manifest[name]!r}, expected {value!r}"
                )
        available_val.update(int(seed) for seed in manifest.get("val_seeds", ()))
        available_test.update(int(seed) for seed in manifest.get("test_seeds", ()))
    if not set(val_seeds).issubset(available_val) or not set(test_seeds).issubset(available_test):
        raise SystemExit("baseline seed ranges do not match this run")


def dual_grid(etas: Sequence[float], betas: Sequence[float], ranks: Sequence[int]) -> List[HP]:
    return [(eta, beta, rank) for eta in etas for beta in betas for rank in ranks]


def evaluate(
    stream: Dict[str, np.ndarray], grid: Sequence[HP], seed: int, args: argparse.Namespace
) -> np.ndarray:
    return run_batch(
        stream["write_keys"],
        stream["write_values"],
        stream["query_keys"],
        stream["test_keys"],
        stream["test_values"],
        "dual_trace",
        grid,
        seed=seed,
        device=args.device,
        query_every=args.query_every,
        eta_u=args.eta_u,
        lam=args.lam,
        clip=args.clip,
    )


def paired_row(
    suite: str,
    task: str,
    condition: str,
    dual_scores: np.ndarray,
    comparator: str,
    baseline_scores: np.ndarray,
) -> Dict[str, object]:
    difference = dual_scores - baseline_scores
    stats = summarize(difference)
    if len(difference) > 1 and stats["se"] > 0:
        t_stat = stats["mean"] / stats["se"]
        p_value = paired_p_value(t_stat, len(difference) - 1)
    else:
        t_stat, p_value = 0.0, None
    return {
        "suite": suite,
        "task": task,
        "condition": condition,
        "method": "dual_trace",
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


def main() -> None:
    args = parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    unknown = set(datasets) - {"synthetic", "word", "module"}
    if unknown:
        raise SystemExit(f"unknown datasets: {sorted(unknown)}")
    if args.val_seeds < 1 or args.test_seeds < 1:
        raise SystemExit("seed counts must be positive")

    etas = parse_numbers(args.etas, float)
    betas = parse_numbers(args.betas, float)
    ranks = parse_numbers(args.per_trace_ranks, int)
    if any(value < 0 for value in etas + betas) or any(rank < 1 for rank in ranks):
        raise SystemExit("eta/beta must be non-negative and ranks must be positive")

    if args.quick:
        etas, betas, ranks = (0.05, 0.1), (1.0, 10.0), (1, 2)
        args.stream_length = min(args.stream_length, 256)
        args.test_size = min(args.test_size, 300)
        args.reference_size = min(args.reference_size, 500)
        args.val_seeds = min(args.val_seeds, 1)
        args.test_seeds = min(args.test_seeds, 2)
        args.val_seed_start = 101
        args.test_seed_start = 1
        args.no_baselines = True

    val_seeds = tuple(args.val_seed_start + index for index in range(args.val_seeds))
    test_seeds = tuple(args.test_seed_start + index for index in range(args.test_seeds))
    baseline_paths = tuple(args.baseline_results) or DEFAULT_BASELINES[args.suite]
    if args.no_baselines:
        baseline_paths = ()
        baselines = {}
    else:
        validate_baseline_manifests(baseline_paths, args, val_seeds, test_seeds)
        baselines = load_baselines(baseline_paths)

    requested_conditions = {item.strip() for item in args.conditions.split(",") if item.strip()}
    plan: List[Tuple[Task, Condition]] = []
    for task in load_tasks(datasets):
        for condition in suite_conditions(args.suite, task):
            if not requested_conditions or condition.name in requested_conditions:
                plan.append((task, condition))
    if requested_conditions and requested_conditions - {condition.name for _, condition in plan}:
        missing = sorted(requested_conditions - {condition.name for _, condition in plan})
        raise SystemExit(f"unavailable requested conditions: {missing}")
    if not plan:
        raise SystemExit("no experiment conditions selected")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "results" / f"dual_trace_{args.suite}_{stamp}"
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "suite": args.suite,
        "method": "dual_trace",
        "modulation": "hq_normalized / (hq_normalized + hw_normalized + 1e-12)",
        "causal_order": "score current write, update memory, update write trace, then update query trace",
        "state_matching": "two traces use per_trace_rank; total trace rank is twice that value",
        "quick": args.quick,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": None,
        "device": args.device,
        "datasets": datasets,
        "conditions": [condition.name for _, condition in plan],
        "val_seeds": val_seeds,
        "test_seeds": test_seeds,
        "etas": etas,
        "betas": betas,
        "per_trace_ranks": ranks,
        "total_trace_ranks": tuple(2 * rank for rank in ranks),
        "query_every": args.query_every,
        "eta_u": args.eta_u,
        "lam": args.lam,
        "clip": args.clip,
        "stream_length": args.stream_length,
        "test_size": args.test_size,
        "reference_size": args.reference_size,
        "baseline_results": [str(path.resolve()) for path in baseline_paths],
        "source_hashes": {
            "dqd_core.py": sha256(ROOT / "dqd_core.py"),
            "run_dual_trace_experiments.py": sha256(Path(__file__).resolve()),
            **{
                filename: sha256(ROOT / filename)
                for filename in ("word.npz", "module.npz")
                if (ROOT / filename).exists()
            },
        },
    }
    if HAS_TORCH:
        import torch

        manifest["torch"] = torch.__version__
        if args.device.startswith("cuda") and torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    grid = dual_grid(etas, betas, ranks)
    progress = Progress(len(plan) * (len(val_seeds) + len(test_seeds)))
    per_seed_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    paired_rows: List[Dict[str, object]] = []
    selected: Dict[str, Dict[str, object]] = {}

    for task, condition in plan:
        print(f"\n=== dual trace | {args.suite} | {task.name} | {condition.name} ===", flush=True)
        started = time.time()
        validation_parts = []
        for index, seed in enumerate(val_seeds, 1):
            stream = make_stream(
                task, seed, condition, args.query_every, args.stream_length,
                args.test_size, args.reference_size,
            )
            validation_parts.append(evaluate(stream, grid, seed, args))
            progress.update(f"{task.name}/{condition.name} validation {index}/{len(val_seeds)}")
        validation = np.stack(validation_parts)
        means = validation.mean(axis=0)
        stds = validation.std(axis=0, ddof=1) if len(val_seeds) > 1 else np.zeros(len(grid))
        for hp, mean, std in zip(grid, means, stds):
            validation_rows.append({
                "suite": args.suite,
                "task": task.name,
                "condition": condition.name,
                "method": "dual_trace",
                "eta": hp[0],
                "beta": hp[1],
                "per_trace_rank": hp[2],
                "total_trace_rank": 2 * hp[2],
                "validation_mean": float(mean),
                "validation_std": float(std),
                "n": len(val_seeds),
            })
        best_index = int(np.argmin(means))
        best_hp = grid[best_index]

        test_parts = []
        for index, seed in enumerate(test_seeds, 1):
            stream = make_stream(
                task, seed, condition, args.query_every, args.stream_length,
                args.test_size, args.reference_size,
            )
            test_parts.append(float(evaluate(stream, [best_hp], seed, args)[0]))
            progress.update(f"{task.name}/{condition.name} test {index}/{len(test_seeds)}")
        dual_scores = np.asarray(test_parts)
        stats = summarize(dual_scores)
        elapsed = time.time() - started

        baseline_arrays: Dict[str, np.ndarray] = {}
        for comparator in COMPARATORS:
            keys = [(task.name, condition.name, comparator, seed) for seed in test_seeds]
            if all(key in baselines for key in keys):
                baseline_arrays[comparator] = np.asarray([baselines[key] for key in keys])
        plain_mean = float(baseline_arrays["plain"].mean()) if "plain" in baseline_arrays else math.nan
        gain = 100.0 * (plain_mean - stats["mean"]) / plain_mean if np.isfinite(plain_mean) else math.nan

        selected.setdefault(task.name, {})[condition.name] = {
            "eta": best_hp[0],
            "beta": best_hp[1],
            "per_trace_rank": best_hp[2],
            "total_trace_rank": 2 * best_hp[2],
            "validation_mean": float(means[best_index]),
            "grid_size": len(grid),
        }
        for seed, score in zip(test_seeds, dual_scores):
            per_seed_rows.append({
                "suite": args.suite,
                "task": task.name,
                "condition": condition.name,
                "method": "dual_trace",
                "seed": seed,
                "score": float(score),
                "eta": best_hp[0],
                "beta": best_hp[1],
                "per_trace_rank": best_hp[2],
                "total_trace_rank": 2 * best_hp[2],
            })
        summary_rows.append({
            "suite": args.suite,
            "task": task.name,
            "condition": condition.name,
            "method": "dual_trace",
            **stats,
            "gain_vs_plain_percent": gain,
            "eta": best_hp[0],
            "beta": best_hp[1],
            "per_trace_rank": best_hp[2],
            "total_trace_rank": 2 * best_hp[2],
            "grid_size": len(grid),
            "elapsed_seconds": elapsed,
        })
        print(
            f"  dual_trace {stats['mean']:.4f} +- {stats['std']:.4f}  "
            f"gain={gain:+.1f}%  eta={best_hp[0]:g} beta={best_hp[1]:g} "
            f"r_each={best_hp[2]} r_total={2 * best_hp[2]} ({elapsed:.0f}s)",
            flush=True,
        )
        for comparator, scores in baseline_arrays.items():
            row = paired_row(args.suite, task.name, condition.name, dual_scores, comparator, scores)
            paired_rows.append(row)
            print(
                f"    paired dual_trace vs {comparator}: diff={row['mean_difference']:+.4f} "
                f"95%CI=[{row['ci_low']:+.4f},{row['ci_high']:+.4f}] "
                f"wins={row['wins']}/{row['n']}",
                flush=True,
            )

        write_csv(output / "validation_grid.csv", validation_rows)
        write_csv(output / "per_seed.csv", per_seed_rows)
        write_csv(output / "summary.csv", summary_rows)
        write_csv(output / "paired.csv", paired_rows)
        (output / "selected_hyperparameters.json").write_text(
            json.dumps(selected, indent=2), encoding="utf-8"
        )

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved complete Dual-Trace results to {output}")


if __name__ == "__main__":
    main()
