"""Core evaluator for Dual Query Delta (DQD).

This module contains no dataset or hyper-parameter selection logic.  It accepts
an explicit write stream, query stream, and test set so that those sources can
be generated independently and shared exactly across methods.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised on NumPy-only installations
    torch = None
    HAS_TORCH = False


HP = Tuple[float, float, int]
SUPPORTED_MODES = {
    "plain",
    "write_diag",
    "write_trace",
    "dqd",
    "full_query_trace",
    "diag_query_trace",
    "last_query",
    "random_query_trace",
    "query_moment_reference",
    "weighted_linear_reference",
}


def normalized_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    denom = float(np.sqrt(np.mean(np.sum(target * target, axis=1))))
    if denom <= 0:
        raise ValueError("target RMS norm must be positive")
    return float(np.sqrt(np.mean(np.sum((pred - target) ** 2, axis=1))) / denom)


def reference_statistics(
    keys: np.ndarray,
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return query second moment, its heat scale, and weighted linear fit.

    The second moment is deliberately *not* called a covariance: no centering is
    performed.  The heat scale matches the online trace's maximum-observed-heat
    normalization as closely as an offline stationary reference can.
    """
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(keys) != len(values):
        raise ValueError("reference keys and values must have the same length")
    if weights is None:
        w = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(keys),) or np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("reference weights must be finite and non-negative")
        total = float(w.sum())
        if total <= 0:
            raise ValueError("reference weights must have positive mass")
        w = w / total

    moment = (keys * w[:, None]).T @ keys
    heat = np.einsum("nd,df,nf->n", keys, moment, keys)
    heat_scale = max(float(np.max(heat)), 1e-12)

    cross = (keys * w[:, None]).T @ values
    ridge = 1e-6 * max(float(np.trace(moment)) / keys.shape[1], 1e-12)
    linear_map = np.linalg.solve(moment + ridge * np.eye(keys.shape[1]), cross).T
    return moment, heat_scale, linear_map


def _validate_inputs(
    write_keys: np.ndarray,
    write_values: np.ndarray,
    query_keys: np.ndarray,
    test_keys: np.ndarray,
    test_values: np.ndarray,
    hps: Sequence[HP],
    query_every: int,
) -> None:
    if query_every < 1:
        raise ValueError("query_every must be at least 1")
    if len(write_keys) != len(write_values):
        raise ValueError("write keys and values must have the same length")
    if len(test_keys) != len(test_values):
        raise ValueError("test keys and values must have the same length")
    if write_keys.ndim != 2 or query_keys.ndim != 2 or test_keys.ndim != 2:
        raise ValueError("all key arrays must be matrices")
    if write_keys.shape[1] != query_keys.shape[1] or write_keys.shape[1] != test_keys.shape[1]:
        raise ValueError("write, query, and test key dimensions must match")
    needed_queries = (len(write_keys) - 1) // query_every + 1
    if len(query_keys) < needed_queries:
        raise ValueError(f"need {needed_queries} query vectors, received {len(query_keys)}")
    if not hps:
        raise ValueError("at least one hyper-parameter tuple is required")
    if any(eta < 0 or beta < 0 or rank < 1 for eta, beta, rank in hps):
        raise ValueError("eta and beta must be non-negative and rank must be positive")


def _initial_subspace(dk: int, rank: int, seed: int, orthonormal: bool) -> np.ndarray:
    rng = np.random.default_rng(seed + 1009 * rank)
    raw = rng.normal(size=(dk, rank))
    if orthonormal:
        q, _ = np.linalg.qr(raw, mode="reduced")
        return q[:, :rank]
    return 0.1 * raw


def _run_numpy(
    write_keys: np.ndarray,
    write_values: np.ndarray,
    query_keys: np.ndarray,
    test_keys: np.ndarray,
    test_values: np.ndarray,
    mode: str,
    hps: Sequence[HP],
    seed: int,
    query_every: int,
    eta_u: float,
    lam: float,
    clip: float,
    reference_moment: Optional[np.ndarray],
    reference_heat_scale: Optional[float],
) -> np.ndarray:
    dk = write_keys.shape[1]
    dv = write_values.shape[1]
    result = np.empty(len(hps), dtype=np.float64)

    for rank in sorted({hp[2] for hp in hps}):
        selected = [i for i, hp in enumerate(hps) if hp[2] == rank]
        eta = np.asarray([hps[i][0] for i in selected], dtype=np.float64)
        beta = np.asarray([hps[i][1] for i in selected], dtype=np.float64)
        memory = np.zeros((len(selected), dv, dk), dtype=np.float64)

        learned_trace = mode in {"dqd", "write_trace"}
        random_trace = mode == "random_query_trace"
        if learned_trace or random_trace:
            subspace = _initial_subspace(dk, rank, seed, orthonormal=random_trace)
            energy = np.zeros(rank, dtype=np.float64)
            trace_scale = 1e-12
        if mode == "full_query_trace":
            full_trace = np.zeros((dk, dk), dtype=np.float64)
            trace_scale = 1e-12
        if mode == "diag_query_trace":
            diag_trace = np.zeros(dk, dtype=np.float64)
            trace_scale = 1e-12
        if mode == "write_diag":
            write_diag = np.zeros(dk, dtype=np.float64)
        last_q = None
        query_pointer = 0

        def update_low_rank(x: np.ndarray) -> None:
            nonlocal subspace, energy, trace_scale
            projection = subspace.T @ x
            subspace = subspace + eta_u * np.outer(x - subspace @ projection, projection)
            energy = lam * energy + projection * projection
            post_projection = subspace.T @ x
            trace_scale = max(trace_scale, float(np.sum(energy * post_projection * post_projection)))

        for step, (key, value) in enumerate(zip(write_keys, write_values)):
            if mode == "plain" or (mode in {"dqd", "write_trace", "full_query_trace",
                                                    "diag_query_trace", "last_query",
                                                    "random_query_trace"} and np.all(beta == 0)):
                heat = 0.0
            elif mode == "write_diag":
                scale = max(float(np.max(write_diag)), 1e-12)
                heat = float(write_diag @ (key * key)) / scale
            elif mode in {"dqd", "write_trace", "random_query_trace"}:
                projection = subspace.T @ key
                heat = float(np.sum(energy * projection * projection)) / max(trace_scale, 1e-12)
            elif mode == "full_query_trace":
                heat = float(key @ full_trace @ key) / max(trace_scale, 1e-12)
            elif mode == "diag_query_trace":
                heat = float(diag_trace @ (key * key)) / max(trace_scale, 1e-12)
            elif mode == "last_query":
                heat = 0.0 if last_q is None else float(key @ last_q) ** 2
            elif mode == "query_moment_reference":
                if reference_moment is None or reference_heat_scale is None:
                    raise ValueError("query_moment_reference requires reference statistics")
                heat = float(key @ reference_moment @ key) / reference_heat_scale
            else:
                raise ValueError(f"unsupported iterative mode: {mode}")

            gain = 1.0 + beta * heat
            rate = np.minimum(eta * gain, clip)
            prediction = np.einsum("bvd,d->bv", memory, key)
            memory += rate[:, None, None] * (value[None, :] - prediction)[:, :, None] * key[None, None, :]

            if mode == "write_diag":
                write_diag = lam * write_diag + key * key

            is_trace_step = step % query_every == 0
            if is_trace_step and mode == "write_trace":
                update_low_rank(key)

            if is_trace_step:
                query = query_keys[query_pointer]
                query_pointer += 1
                if mode == "dqd":
                    update_low_rank(query)
                elif mode == "random_query_trace":
                    projection = subspace.T @ query
                    energy = lam * energy + projection * projection
                    trace_scale = max(trace_scale, float(np.sum(energy * projection * projection)))
                elif mode == "full_query_trace":
                    full_trace = lam * full_trace + np.outer(query, query)
                    trace_scale = max(trace_scale, float(query @ full_trace @ query))
                elif mode == "diag_query_trace":
                    diag_trace = lam * diag_trace + query * query
                    trace_scale = max(trace_scale, float(diag_trace @ (query * query)))
                elif mode == "last_query":
                    last_q = query

        prediction = np.einsum("nd,bvd->bnv", test_keys, memory)
        denom = float(np.sqrt(np.mean(np.sum(test_values * test_values, axis=1))))
        errors = np.sqrt(np.mean(np.sum((prediction - test_values[None, :, :]) ** 2, axis=2), axis=1)) / denom
        result[selected] = errors
    return result


def _run_torch(
    write_keys: np.ndarray,
    write_values: np.ndarray,
    query_keys: np.ndarray,
    test_keys: np.ndarray,
    test_values: np.ndarray,
    mode: str,
    hps: Sequence[HP],
    seed: int,
    device: str,
    query_every: int,
    eta_u: float,
    lam: float,
    clip: float,
    reference_moment: Optional[np.ndarray],
    reference_heat_scale: Optional[float],
) -> np.ndarray:
    if not HAS_TORCH:
        raise RuntimeError("Torch is not installed")
    dev = torch.device(device)
    to_tensor = lambda x: torch.as_tensor(np.asarray(x), device=dev, dtype=torch.float32)
    wk, wv, qk, tk, tv = map(to_tensor, (write_keys, write_values, query_keys, test_keys, test_values))
    dk, dv = wk.shape[1], wv.shape[1]
    result = np.empty(len(hps), dtype=np.float64)

    with torch.no_grad():
        for rank in sorted({hp[2] for hp in hps}):
            selected = [i for i, hp in enumerate(hps) if hp[2] == rank]
            eta = to_tensor([hps[i][0] for i in selected])
            beta = to_tensor([hps[i][1] for i in selected])
            memory = torch.zeros((len(selected), dv, dk), device=dev)

            learned_trace = mode in {"dqd", "write_trace"}
            random_trace = mode == "random_query_trace"
            if learned_trace or random_trace:
                subspace = to_tensor(_initial_subspace(dk, rank, seed, orthonormal=random_trace))
                energy = torch.zeros(rank, device=dev)
                trace_scale = torch.tensor(1e-12, device=dev)
            if mode == "full_query_trace":
                full_trace = torch.zeros((dk, dk), device=dev)
                trace_scale = torch.tensor(1e-12, device=dev)
            if mode == "diag_query_trace":
                diag_trace = torch.zeros(dk, device=dev)
                trace_scale = torch.tensor(1e-12, device=dev)
            if mode == "write_diag":
                write_diag = torch.zeros(dk, device=dev)
            if mode == "query_moment_reference":
                if reference_moment is None or reference_heat_scale is None:
                    raise ValueError("query_moment_reference requires reference statistics")
                ref_moment_t = to_tensor(reference_moment)
                ref_scale_t = torch.tensor(reference_heat_scale, device=dev)
            last_q = None
            query_pointer = 0

            for step in range(len(wk)):
                key, value = wk[step], wv[step]
                if mode == "plain" or (mode in {"dqd", "write_trace", "full_query_trace",
                                                        "diag_query_trace", "last_query",
                                                        "random_query_trace"} and bool(torch.all(beta == 0))):
                    heat = torch.tensor(0.0, device=dev)
                elif mode == "write_diag":
                    heat = torch.dot(write_diag, key * key) / write_diag.max().clamp(min=1e-12)
                elif mode in {"dqd", "write_trace", "random_query_trace"}:
                    projection = subspace.T @ key
                    heat = torch.sum(energy * projection * projection) / trace_scale.clamp(min=1e-12)
                elif mode == "full_query_trace":
                    heat = key @ full_trace @ key / trace_scale.clamp(min=1e-12)
                elif mode == "diag_query_trace":
                    heat = torch.dot(diag_trace, key * key) / trace_scale.clamp(min=1e-12)
                elif mode == "last_query":
                    heat = torch.tensor(0.0, device=dev) if last_q is None else torch.dot(key, last_q).square()
                elif mode == "query_moment_reference":
                    heat = key @ ref_moment_t @ key / ref_scale_t
                else:
                    raise ValueError(f"unsupported iterative mode: {mode}")

                rate = torch.clamp(eta * (1.0 + beta * heat), max=clip)
                prediction = torch.einsum("bvd,d->bv", memory, key)
                memory += rate[:, None, None] * (value[None, :] - prediction)[:, :, None] * key[None, None, :]

                if mode == "write_diag":
                    write_diag = lam * write_diag + key * key

                is_trace_step = step % query_every == 0
                if is_trace_step and mode == "write_trace":
                    projection = subspace.T @ key
                    subspace = subspace + eta_u * torch.outer(key - subspace @ projection, projection)
                    energy = lam * energy + projection * projection
                    post_projection = subspace.T @ key
                    trace_scale = torch.maximum(trace_scale, torch.sum(energy * post_projection * post_projection))

                if is_trace_step:
                    query = qk[query_pointer]
                    query_pointer += 1
                    if mode == "dqd":
                        projection = subspace.T @ query
                        subspace = subspace + eta_u * torch.outer(query - subspace @ projection, projection)
                        energy = lam * energy + projection * projection
                        post_projection = subspace.T @ query
                        trace_scale = torch.maximum(trace_scale, torch.sum(energy * post_projection * post_projection))
                    elif mode == "random_query_trace":
                        projection = subspace.T @ query
                        energy = lam * energy + projection * projection
                        trace_scale = torch.maximum(trace_scale, torch.sum(energy * projection * projection))
                    elif mode == "full_query_trace":
                        full_trace = lam * full_trace + torch.outer(query, query)
                        trace_scale = torch.maximum(trace_scale, query @ full_trace @ query)
                    elif mode == "diag_query_trace":
                        diag_trace = lam * diag_trace + query * query
                        trace_scale = torch.maximum(trace_scale, torch.dot(diag_trace, query * query))
                    elif mode == "last_query":
                        last_q = query

            prediction = torch.einsum("nd,bvd->bnv", tk, memory)
            denom = torch.sqrt(torch.mean(torch.sum(tv * tv, dim=1)))
            errors = torch.sqrt(torch.mean(torch.sum((prediction - tv[None, :, :]) ** 2, dim=2), dim=1)) / denom
            result[selected] = errors.cpu().numpy()
    return result


def run_batch(
    write_keys: np.ndarray,
    write_values: np.ndarray,
    query_keys: np.ndarray,
    test_keys: np.ndarray,
    test_values: np.ndarray,
    mode: str,
    hps: Sequence[HP],
    *,
    seed: int,
    device: str = "numpy",
    query_every: int = 3,
    eta_u: float = 0.05,
    lam: float = 0.999,
    clip: float = 1.9,
    reference_keys: Optional[np.ndarray] = None,
    reference_values: Optional[np.ndarray] = None,
    reference_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate one method over a batched list of hyper-parameters."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {sorted(SUPPORTED_MODES)}")
    arrays = [write_keys, write_values, query_keys, test_keys, test_values]
    write_keys, write_values, query_keys, test_keys, test_values = [
        np.asarray(x, dtype=np.float64) for x in arrays
    ]
    _validate_inputs(write_keys, write_values, query_keys, test_keys, test_values, hps, query_every)

    ref_moment = ref_scale = linear_map = None
    if mode in {"query_moment_reference", "weighted_linear_reference"}:
        if reference_keys is None or reference_values is None:
            raise ValueError(f"{mode} requires reference keys and values")
        ref_moment, ref_scale, linear_map = reference_statistics(
            reference_keys, reference_values, reference_weights
        )
    if mode == "weighted_linear_reference":
        score = normalized_rmse(test_keys @ linear_map.T, test_values)
        return np.full(len(hps), score, dtype=np.float64)

    if device != "numpy":
        return _run_torch(
            write_keys, write_values, query_keys, test_keys, test_values, mode, hps,
            seed, device, query_every, eta_u, lam, clip, ref_moment, ref_scale,
        )
    return _run_numpy(
        write_keys, write_values, query_keys, test_keys, test_values, mode, hps,
        seed, query_every, eta_u, lam, clip, ref_moment, ref_scale,
    )
