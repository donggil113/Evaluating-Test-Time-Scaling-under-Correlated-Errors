"""Protocol N5-v1: dev fitting/selection (--phase dev), then a single test
evaluation of the frozen selection (--phase test). See docs/protocol.md."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, logsumexp

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ttsce.env import ReplayEnv  # noqa: E402
from ttsce.models import (AGGREGATORS, BayesParams, IncrementalScorer, _yes_no, agg_bayes)  # noqa: E402
from ttsce.policies import (make_schedule, prefix_eval, run_adaptive_consistency, run_fixed,  # noqa: E402
                            run_stopping, stop_index, stopping_trajectory)
from ttsce.table import ABSTAIN, load_gold, load_public  # noqa: E402

SEEDS = list(range(20))
N_GRID = [1, 2, 4, 8, 16, 32, 64]
V_GRID = [1, 2, 4, 8, 16, 32]
MV_EXTRA = [128, 256]
TAUS = [0.26, 0.28, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45,  # D1 (dev-driven, pre-test)
        0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
GV_K = [1, 2, 4, 8]
BUDGET_MULT = [2, 4, 8, 16, 32, 64, 128, 256]
FAMILIES = ["GEN", "VER", "FIXED", "STOP", "STOP-IND", "STOP-CORR"]
CHECKPOINTS = sorted({int(round(x)) for x in np.geomspace(1, 600, 40)})


# ------------------------------------------------------------------ fitting (dev only)
def fit_beta_binomial(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)

    def nll(th):
        a, b = np.exp(th)
        return -np.sum(betaln(a + k, b + n - k) - betaln(a, b))
    r = minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    a, b = np.exp(r.x)
    return float(a), float(b)


def candidate_verdict_counts(arrays, gold, problems):
    rows = []
    for p in problems:
        A = arrays[p]
        for j in range(A.n_verified):
            a = A.answers[j]
            if a == ABSTAIN:
                continue
            y, n = _yes_no(A.verdicts[j])
            rows.append((p, j, a, int(a == gold[p]), y, n))
    return pd.DataFrame(rows, columns=["problem", "cand", "answer", "correct", "yes", "no"])


def fit_params(arrays, gold, dev):
    kq, nq = [], []
    for p in dev:
        ans = arrays[p].answers
        valid = ans != ABSTAIN
        kq.append(int((ans[valid] == gold[p]).sum()))
        nq.append(int(valid.sum()))
    aq, bq = fit_beta_binomial(kq, nq)
    cv = candidate_verdict_counts(arrays, gold, dev)
    c1, c0 = cv[cv.correct == 1], cv[cv.correct == 0]
    t = c1.yes.sum() / (c1.yes.sum() + c1.no.sum())
    f = c0.yes.sum() / (c0.yes.sum() + c0.no.sum())
    a1, b1 = fit_beta_binomial(c1.yes, c1.yes + c1.no)
    a0, b0 = fit_beta_binomial(c0.yes, c0.yes + c0.no)
    return {"IND": BayesParams("IND", aq, bq, t=float(t), f=float(f)),
            "CORR": BayesParams("CORR", aq, bq, a1=a1, b1=b1, a0=a0, b0=b0)}


def collect_states(arrays, gold, problems, params, seeds):
    """Raw (vote, verdict) term vectors at checkpoints of every schedule."""
    specs = [("G", 64)] + [("GV", 64, k) for k in GV_K] + [("FIX", 64, 32)]
    V, R, G = [], [], []
    raw = BayesParams(params.kind, params.aq, params.bq, params.t, params.f,
                      params.a1, params.b1, params.a0, params.b0, 1.0, 1.0)
    for p in problems:
        for s in seeds:
            for spec in specs:
                env = ReplayEnv(arrays[p], s, make_schedule(spec))
                sc = IncrementalScorer(raw)
                step = 0
                while env.steps_left:
                    act = env.step()
                    sc.update(env.obs, act)
                    step += 1
                    if step in CHECKPOINTS or not env.steps_left:
                        V.append(sc.p.vote_term(sc.counts).copy())
                        R.append(sc.ver.copy())
                        G.append(gold[p])
    return np.array(V), np.array(R), np.array(G)


def fit_weights(V, R, G):
    def nll(th):
        wv, wr = np.exp(th)
        s = wv * V + wr * R
        return -np.mean(s[np.arange(len(G)), G] - logsumexp(s, axis=1))
    r = minimize(nll, x0=[0.0, -1.0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9})
    return [float(x) for x in np.exp(r.x)], float(r.fun)


def with_weights(p, w):
    return BayesParams(p.kind, p.aq, p.bq, p.t, p.f, p.a1, p.b1, p.a0, p.b0, w[0], w[1])


# ------------------------------------------------------------------ policy enumeration
def enumerate_rows(arrays, gold, problems, params, seeds):
    """Every policy x problem x seed on `problems`, via prefixes / trajectories."""
    aggs = dict(AGGREGATORS)
    aggs["BAYES-IND"] = agg_bayes(params["IND"])
    aggs["BAYES-CORR"] = agg_bayes(params["CORR"])
    rows = []
    for p in problems:
        A, g = arrays[p], gold[p]
        for s in seeds:
            full = ReplayEnv(A, s, make_schedule(("FIX", 64, 32))).run_all()
            gen_only = ReplayEnv(A, s, make_schedule(("G", 256))).run_all()
            for n in N_GRID + MV_EXTRA:
                r = prefix_eval(gen_only, n, 0, aggs["MV"])
                rows.append(dict(policy=f"FIX:MV:N{n}:V0", problem=p, seed=s, correct=int(r["answer"] == g), **r))
            for name, agg in aggs.items():
                if name == "MV":
                    continue
                for n in N_GRID:
                    for v in V_GRID:
                        r = prefix_eval(full, n, v, agg)
                        rows.append(dict(policy=f"FIX:{name}:N{n}:V{v}", problem=p, seed=s,
                                         correct=int(r["answer"] == g), **r))
            trajs = {("AC", "G256"): stopping_trajectory(A, s, ("G", 256), ac=True)}
            for kind in ("IND", "CORR"):
                trajs[(kind, "G256")] = stopping_trajectory(A, s, ("G", 256), params[kind])
                for k in GV_K:
                    trajs[(kind, f"GV64x{k}")] = stopping_trajectory(A, s, ("GV", 64, k), params[kind])
            for (kind, sched), tr in trajs.items():
                for tau in TAUS:
                    i = stop_index(tr["stat"], tau)
                    pol = f"AC:G256:c{tau}" if kind == "AC" else f"STOP:{kind}:{sched}:tau{tau}"
                    n_steps = i + 1
                    k = 0 if sched == "G256" else int(sched.split("x")[1])
                    rows.append(dict(policy=pol, problem=p, seed=s, correct=int(tr["answer"][i] == g),
                                     answer=int(tr["answer"][i]), n_gen=-(-n_steps // (k + 1)),
                                     n_ver=n_steps - (-(-n_steps // (k + 1))),
                                     total_tokens=int(tr["total_tokens"][i]),
                                     output_tokens=int(tr["output_tokens"][i]),
                                     cached_tokens=int(tr["cached_tokens"][i])))
    return pd.DataFrame(rows)


def families_of(policy):
    fam = []
    if policy.startswith("FIX:"):
        fam.append("FIXED")
        fam.append("GEN" if policy.startswith("FIX:MV:") else "VER")
    else:
        fam.append("STOP")
        if policy.startswith("AC:") or ":G256:" in policy:
            fam.append("GEN")
        else:
            fam.append("VER")
        if policy.startswith("STOP:IND:"):
            fam.append("STOP-IND")
        if policy.startswith("STOP:CORR:"):
            fam.append("STOP-CORR")
    return fam


def summarize(df):
    per_prob = df.groupby(["policy", "problem"]).agg(err=("correct", lambda x: 1 - x.mean()),
                                                     cost=("total_tokens", "mean"),
                                                     out=("output_tokens", "mean"),
                                                     cached=("cached_tokens", "mean")).reset_index()
    return per_prob.groupby("policy").agg(err=("err", "mean"), cost=("cost", "mean"), out=("out", "mean"),
                                          cached=("cached", "mean")).reset_index()


def select(summary, budgets):
    sel = {}
    summary = summary.assign(fams=summary.policy.map(families_of))
    for fam in FAMILIES:
        sub = summary[summary.fams.map(lambda f: fam in f)]
        for mult, B in budgets.items():
            ok = sub[sub.cost <= B]
            if ok.empty:
                sel[f"{fam}@{mult}"] = None
                continue
            best = ok.sort_values(["err", "cost"]).iloc[0]
            sel[f"{fam}@{mult}"] = {"policy": best.policy, "dev_err": float(best.err), "dev_cost": float(best.cost)}
    return sel


# ------------------------------------------------------------------ online test runs
def parse_policy(pol):
    parts = pol.split(":")
    if parts[0] == "FIX":
        return ("FIX", parts[1], int(parts[2][1:]), int(parts[3][1:]))
    if parts[0] == "AC":
        return ("AC", float(parts[2][1:]))
    k = 0 if parts[2] == "G256" else int(parts[2].split("x")[1])
    return ("STOP", parts[1], k, float(parts[3][3:]))


def run_online(pol, arrays_p, seed, params):
    kind = parse_policy(pol)
    if kind[0] == "FIX":
        agg = {"BAYES-IND": agg_bayes(params["IND"]), "BAYES-CORR": agg_bayes(params["CORR"])}.get(
            kind[1], AGGREGATORS.get(kind[1]))
        return run_fixed(arrays_p, seed, kind[2], kind[3], agg)
    if kind[0] == "AC":
        return run_adaptive_consistency(arrays_p, seed, 256, kind[1])
    spec = ("G", 256) if kind[2] == 0 else ("GV", 64, kind[2])
    return run_stopping(arrays_p, seed, spec, params[kind[1]], kind[3])


def paired_bootstrap(a, b, n_boot=10000, seed=12345):
    """a, b: per-problem values (aligned). Returns mean diff (a-b) and 95% CI, two-sided p."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "p_boot": float(min(1.0, p))}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    m = len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def calibration_metrics(V, R, G, w):
    s = w[0] * V + w[1] * R
    logp = s - logsumexp(s, axis=1, keepdims=True)
    p = np.exp(logp)
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == G).astype(float)
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = sum(abs(acc[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return {"logloss": float(-logp[np.arange(len(G)), G].mean()), "ece": float(ece),
            "mean_conf": float(conf.mean()), "acc": float(acc.mean())}


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test"], required=True)
    ap.add_argument("--table", default=os.path.join(ROOT, "data", "tables", "gpqa_l33_70b"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gpqa_l33_70b"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arrays, split = load_public(args.table)
    gold = load_gold(args.table)
    dev = sorted(p for p, s in split.items() if s == "dev")
    test = sorted(p for p, s in split.items() if s == "test")
    t_wall, t_cpu = time.time(), time.process_time()

    if args.phase == "dev":
        params = fit_params(arrays, gold, dev)
        Vs, Rs, Gs = {}, {}, {}
        for kind in ("IND", "CORR"):
            Vs[kind], Rs[kind], Gs[kind] = collect_states(arrays, gold, dev, params[kind], SEEDS[:5])
            w, nll = fit_weights(Vs[kind], Rs[kind], Gs[kind])
            params[kind] = with_weights(params[kind], w)
        fit_cpu = time.process_time() - t_cpu
        c_gen = float(np.mean([np.mean(arrays[p].gen_in + arrays[p].gen_out) for p in dev]))
        budgets = {m: m * c_gen for m in BUDGET_MULT}
        df = enumerate_rows(arrays, gold, dev, params, SEEDS)
        summary = summarize(df)
        summary.to_csv(os.path.join(args.out, "dev_policy_summary.csv"), index=False)
        sel = select(summary, budgets)
        json.dump({"params": {k: vars(v) for k, v in params.items()}, "c_gen_tokens": c_gen,
                   "budgets_tokens": budgets, "selection": sel, "fit_cpu_s": fit_cpu,
                   "n_dev_states": {k: int(len(v)) for k, v in Gs.items()},
                   "git_commit": git_rev()},
                  open(os.path.join(args.out, "dev_selection.json"), "w"), indent=2)
        print(json.dumps(sel, indent=1))
    else:
        frozen = json.load(open(os.path.join(args.out, "dev_selection.json")))
        params = {k: BayesParams(**v) for k, v in frozen["params"].items()}
        sel = frozen["selection"]
        pols = sorted({v["policy"] for v in sel.values() if v})
        rows = []
        for pol in pols:
            for p in test:
                for s in SEEDS:
                    r = run_online(pol, arrays[p], s, params)
                    rows.append(dict(policy=pol, problem=p, seed=s, correct=int(r["answer"] == gold[p]), **r))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out, "test_selected_runs.csv.gz"), index=False)
        pp = df.groupby(["policy", "problem"]).agg(err=("correct", lambda x: 1 - x.mean()),
                                                   cost=("total_tokens", "mean"),
                                                   cpu=("controller_cpu_s", "mean")).reset_index()
        pe = pp.pivot(index="problem", columns="policy", values="err")
        pc = pp.pivot(index="problem", columns="policy", values="cost")
        out = {"per_budget": {}, "endpoints": {}}
        for key, v in sel.items():
            if v:
                out["per_budget"][key] = dict(v, test_err=float(pe[v["policy"]].mean()),
                                              test_cost=float(pc[v["policy"]].mean()),
                                              test_cpu_s_per_problem=float(
                                                  pp[pp.policy == v["policy"]].cpu.mean()))
        for name, (fa, fb) in {"E1_VER_minus_GEN": ("VER", "GEN"), "E2_STOP_minus_FIXED": ("STOP", "FIXED"),
                               "E3_CORR_minus_IND": ("STOP-CORR", "STOP-IND")}.items():
            res = {}
            for m in BUDGET_MULT:
                a, b = sel.get(f"{fa}@{m}"), sel.get(f"{fb}@{m}")
                if not a or not b:
                    continue
                r = paired_bootstrap(pe[a["policy"]], pe[b["policy"]])
                r["cost_diff"] = paired_bootstrap(pc[a["policy"]], pc[b["policy"]])["diff"]
                r.update(policy_a=a["policy"], policy_b=b["policy"])
                res[str(m)] = r
            ks = list(res)
            for k, adj in zip(ks, holm([res[k]["p_boot"] for k in ks])):
                res[k]["p_holm"] = adj
            out["endpoints"][name] = res
        cal = {}
        for kind in ("IND", "CORR"):
            V, R, G = collect_states(arrays, gold, test, params[kind], SEEDS[:5])
            cal[kind] = calibration_metrics(V, R, G, [params[kind].w_vote, params[kind].w_ver])
        out["calibration_test_states"] = cal
        out["git_commit"] = git_rev()
        json.dump(out, open(os.path.join(args.out, "test_endpoints.json"), "w"), indent=2)
        print(json.dumps(out["endpoints"], indent=1))
        print(json.dumps(cal, indent=1))

    json.dump({"phase": args.phase, "wall_s": time.time() - t_wall, "cpu_s": time.process_time() - t_cpu,
               "python": platform.python_version(), "machine": platform.machine(), "n_cpu": os.cpu_count(),
               "git_commit": git_rev(), "argv": sys.argv},
              open(os.path.join(args.out, f"run_{args.phase}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
