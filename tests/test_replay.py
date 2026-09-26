"""Leakage and replay-equivalence checks (run: python -m pytest tests -q)."""
import copy
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ttsce.env import ReplayEnv  # noqa: E402
from ttsce.models import AGGREGATORS, BayesParams  # noqa: E402
from ttsce.policies import (make_schedule, prefix_eval, run_adaptive_consistency, run_fixed,  # noqa: E402
                            run_stopping, stop_index, stopping_trajectory)
from ttsce.table import load_public  # noqa: E402

TABLE = os.path.join(ROOT, "data", "tables", "gpqa_l33_70b")
PARAMS = [BayesParams("IND", 1.5, 1.0, t=0.9, f=0.7, w_vote=1.0, w_ver=0.5),
          BayesParams("CORR", 1.5, 1.0, a1=3.0, b1=0.5, a0=1.0, b0=0.6, w_vote=1.0, w_ver=0.5)]


@pytest.fixture(scope="module")
def data():
    arrays, _ = load_public(TABLE)
    return arrays


def _mutate_unrevealed(arr, audit, rng):
    """Copy of `arr` where every cell NOT in `audit` is replaced by noise."""
    m = copy.deepcopy(arr)
    gen_seen = {c[1] for c in audit if c[0] == "gen"}
    ver_seen = {(c[1], c[2]) for c in audit if c[0] == "ver"}
    for j in range(len(m.answers)):
        if j not in gen_seen:
            m.answers[j] = rng.integers(-1, 4)
            m.gen_out[j] = rng.integers(1, 3000)
    for j in range(m.n_verified):
        for r in range(m.verdicts.shape[1]):
            if (j, r) not in ver_seen:
                m.verdicts[j, r] = rng.integers(0, 2)
                m.ver_out[j, r] = rng.integers(1, 3000)
    return m


def _audited_run(arr, seed, spec, stop_after):
    env = ReplayEnv(arr, seed, make_schedule(spec))
    for _ in range(stop_after):
        env.step()
    return env.audit_log()


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("tau", [0.6, 0.9, 0.99])
def test_stopping_policy_ignores_unrevealed_cells(data, params, tau):
    rng = np.random.default_rng(0)
    for p in list(data)[:8]:
        arr = data[p]
        spec = ("GV", 64, 2)
        r1 = run_stopping(arr, 3, spec, params, tau)
        steps = r1["n_gen"] + r1["n_ver"]
        audit = _audited_run(arr, 3, spec, steps)
        r2 = run_stopping(_mutate_unrevealed(arr, audit, rng), 3, spec, params, tau)
        for k in ("answer", "n_gen", "n_ver", "total_tokens"):
            assert r1[k] == r2[k]


def test_adaptive_consistency_ignores_unrevealed_cells(data):
    rng = np.random.default_rng(1)
    for p in list(data)[:8]:
        arr = data[p]
        r1 = run_adaptive_consistency(arr, 5, 64, 0.95)
        audit = _audited_run(arr, 5, ("G", 64), r1["n_gen"])
        r2 = run_adaptive_consistency(_mutate_unrevealed(arr, audit, rng), 5, 64, 0.95)
        assert r1["answer"] == r2["answer"] and r1["n_gen"] == r2["n_gen"]


def test_trajectory_matches_online_stopping(data):
    for p in list(data)[:6]:
        arr = data[p]
        for params in PARAMS:
            tr = stopping_trajectory(arr, 2, ("GV", 64, 1), params)
            for tau in (0.5, 0.8, 0.95, 0.999):
                i = stop_index(tr["stat"], tau)
                online = run_stopping(arr, 2, ("GV", 64, 1), params, tau)
                assert online["answer"] == tr["answer"][i]
                assert online["total_tokens"] == tr["total_tokens"][i]


def test_prefix_matches_fixed_runs(data):
    for p in list(data)[:4]:
        arr = data[p]
        full = ReplayEnv(arr, 7, make_schedule(("FIX", 64, 32))).run_all()
        for n, v in [(1, 1), (4, 2), (16, 8), (64, 32), (8, 0)]:
            for name, agg in AGGREGATORS.items():
                a = prefix_eval(full, n, v, agg)
                b = run_fixed(arr, 7, n, v, agg)
                for k in ("answer", "n_ver", "total_tokens", "output_tokens", "cached_tokens"):
                    assert a[k] == b[k], (p, n, v, name, k)


def test_verify_requires_generated_candidate(data):
    arr = next(iter(data.values()))
    env = ReplayEnv(arr, 0, [("ver", 0)])
    with pytest.raises(ValueError):
        env.step()
