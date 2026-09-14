"""Fast invariant and backend-parity tests for the DQD experiment code."""
from __future__ import annotations

import numpy as np

from dqd_core import HAS_TORCH, SUPPORTED_MODES, run_batch


def fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    n, nq, nt, dk, dv = 96, 40, 80, 12, 5
    wk = rng.normal(size=(n, dk)); wk /= np.linalg.norm(wk, axis=1, keepdims=True)
    wv = rng.normal(size=(n, dv))
    qk = rng.normal(size=(nq, dk)); qk /= np.linalg.norm(qk, axis=1, keepdims=True)
    tk = rng.normal(size=(nt, dk)); tk /= np.linalg.norm(tk, axis=1, keepdims=True)
    tv = rng.normal(size=(nt, dv))
    rk = rng.normal(size=(160, dk)); rk /= np.linalg.norm(rk, axis=1, keepdims=True)
    rv = rng.normal(size=(160, dv))
    rw = rng.random(160); rw /= rw.sum()
    return wk, wv, qk, tk, tv, rk, rv, rw


def call(mode, hps, device="numpy", data=None):
    wk, wv, qk, tk, tv, rk, rv, rw = fixture() if data is None else data
    return run_batch(
        wk, wv, qk, tk, tv, mode, hps, seed=11, device=device,
        query_every=3, reference_keys=rk, reference_values=rv, reference_weights=rw,
    )


def assert_close(name, left, right, tolerance=1e-10):
    difference = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    if difference > tolerance:
        raise AssertionError(f"{name}: max difference {difference:.3e} exceeds {tolerance:.3e}")
    print(f"PASS {name:44s} max|diff|={difference:.3e}")


def main():
    ordinary_hps = [(0.05, 1.0, 2), (0.1, 10.0, 4)]

    # Every declared mode executes and returns finite values.
    for mode in sorted(SUPPORTED_MODES):
        values = call(mode, ordinary_hps)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise AssertionError(f"{mode} returned invalid values: {values}")
    print("PASS every declared mode returns finite scores")

    # Batching hyper-parameters must not change an individual trajectory.
    for mode in sorted(SUPPORTED_MODES):
        batched = call(mode, ordinary_hps)
        separate = np.array([call(mode, [hp])[0] for hp in ordinary_hps])
        assert_close(f"batched equals separate: {mode}", batched, separate)

    # All trace variants reduce exactly to plain when beta is zero.
    for rank in (1, 2, 4, 8):
        hp = [(0.1, 0.0, rank)]
        plain = call("plain", [(0.1, 0.0, 4)])
        for mode in (
            "write_trace", "dqd", "dual_trace", "full_query_trace", "diag_query_trace",
            "last_query", "random_query_trace",
        ):
            assert_close(f"beta=0 {mode} r={rank} equals plain", call(mode, hp), plain)

    # The matched write trace must be independent of the query stream; DQD must not be.
    original = fixture()
    changed = list(original)
    changed[2] = np.roll(changed[2], 1, axis=0)
    changed = tuple(changed)
    assert_close(
        "write_trace ignores query ordering",
        call("write_trace", ordinary_hps, data=original),
        call("write_trace", ordinary_hps, data=changed),
    )
    dqd_difference = float(np.max(np.abs(call("dqd", ordinary_hps, data=original) - call("dqd", ordinary_hps, data=changed))))
    if dqd_difference <= 1e-8:
        raise AssertionError("DQD unexpectedly ignored a changed query history")
    print(f"PASS DQD responds to changed query history       max|diff|={dqd_difference:.3e}")

    dual_query_difference = float(np.max(np.abs(
        call("dual_trace", ordinary_hps, data=original)
        - call("dual_trace", ordinary_hps, data=changed)
    )))
    if dual_query_difference <= 1e-8:
        raise AssertionError("dual trace unexpectedly ignored a changed query history")
    changed_writes = list(original)
    changed_writes[0] = np.roll(changed_writes[0], 1, axis=0)
    changed_writes = tuple(changed_writes)
    dual_write_difference = float(np.max(np.abs(
        call("dual_trace", ordinary_hps, data=original)
        - call("dual_trace", ordinary_hps, data=changed_writes)
    )))
    if dual_write_difference <= 1e-8:
        raise AssertionError("dual trace unexpectedly ignored a changed write history")
    print(
        "PASS dual trace responds to both histories       "
        f"query={dual_query_difference:.3e} write={dual_write_difference:.3e}"
    )

    one_step = tuple(array[:1] if index < 5 else array for index, array in enumerate(fixture()))
    assert_close(
        "dual trace has no same-step leakage",
        call("dual_trace", [(0.1, 100.0, 2)], data=one_step),
        call("plain", [(0.1, 0.0, 4)], data=one_step),
    )

    # Identical calls are deterministic.
    assert_close("deterministic repeat", call("dqd", ordinary_hps), call("dqd", ordinary_hps))

    # Cross-backend parity. CPU Torch is always tested when available; CUDA too when present.
    if HAS_TORCH:
        import torch

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for mode in sorted(SUPPORTED_MODES):
                numpy_values = call(mode, ordinary_hps, "numpy")
                torch_values = call(mode, ordinary_hps, device)
                assert_close(f"{device} parity: {mode}", numpy_values, torch_values, tolerance=2e-4)
    else:
        print("SKIP Torch parity: torch is not installed")

    print("\nAll DQD invariants passed.")


if __name__ == "__main__":
    main()
