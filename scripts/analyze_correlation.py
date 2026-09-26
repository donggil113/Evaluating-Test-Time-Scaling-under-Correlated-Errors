"""Measurement endpoints M1-M3 of docs/protocol.md (descriptive; nothing here
is used to select a policy). Unit of resampling is the problem."""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from run_experiment import fit_beta_binomial  # noqa: E402
from ttsce.table import ABSTAIN, load_gold, load_public  # noqa: E402

TABLE = os.path.join(ROOT, "data", "tables", "gpqa_l33_70b")
OUT = os.path.join(ROOT, "results", "gpqa_l33_70b")
N_BOOT = 1000
KS = [1, 2, 4, 8, 16, 24]
FIT_REPS = 8


def cand_frame(arrays, gold, split):
    rows = []
    for p, A in arrays.items():
        valid_ans = A.answers[A.answers != ABSTAIN]
        share = np.bincount(valid_ans, minlength=4) / max(len(valid_ans), 1)
        for j in range(A.n_verified):
            a = A.answers[j]
            if a == ABSTAIN:
                continue
            v = A.verdicts[j]
            ok = v >= 0
            rows.append(dict(problem=p, split=split[p], cand=j, answer=int(a), correct=int(a == gold[p]),
                             yes=int((v == 1).sum()), n=int(ok.sum()),
                             yes_fit=int((v[:FIT_REPS] == 1).sum()), n_fit=int((v[:FIT_REPS] >= 0).sum()),
                             yes_ho=int((v[FIT_REPS:] == 1).sum()), n_ho=int((v[FIT_REPS:] >= 0).sum()),
                             first=int(v[0]), vote_share=float(share[a])))
    return pd.DataFrame(rows)


def log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def empirical_moment(yes, n, k):
    """Unbiased estimate of E[alpha^k]: mean over candidates of C(yes,k)/C(n,k),
    i.e. the fraction of k-subsets of held-out verdicts that are all accepts."""
    yes, n = np.asarray(yes, float), np.asarray(n, float)
    ok = n >= k
    yes, n = yes[ok], n[ok]
    val = np.zeros(len(yes))
    hit = yes >= k
    val[hit] = np.exp(log_comb(yes[hit], k) - log_comb(n[hit], k))
    return float(val.mean())


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.empty(len(allv))
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks[order] = r
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def variance_shares(df):
    """Share of variance of per-candidate acceptance rate explained by problem
    and by (problem, answer) cells, after removing expected binomial noise."""
    r = df.yes / df.n
    tot = r.var(ddof=0)
    noise = np.mean(r * (1 - r) / (df.n - 1).clip(lower=1))
    prob_mean = df.assign(r=r).groupby("problem").r.transform("mean")
    cell_mean = df.assign(r=r).groupby(["problem", "answer"]).r.transform("mean")
    signal = tot - noise
    return {"total_var": float(tot), "binomial_noise_share": float(noise / tot),
            "problem_share_of_signal": float(prob_mean.var(ddof=0) / signal),
            "problem_answer_share_of_signal": float(cell_mean.var(ddof=0) / signal),
            "within_cell_share_of_signal": float(((r - cell_mean) ** 2).mean() - noise) / signal}


def metrics(df):
    c1, c0 = df[df.correct == 1], df[df.correct == 0]
    a1, b1 = fit_beta_binomial(c1.yes, c1.n)
    a0, b0 = fit_beta_binomial(c0.yes, c0.n)
    out = {"n_problems": int(df.problem.nunique()), "n_correct_cands": int(len(c1)), "n_wrong_cands": int(len(c0)),
           "tpr_single": float(c1.yes.sum() / c1.n.sum()), "fpr_single": float(c0.yes.sum() / c0.n.sum()),
           "rho_correct": 1 / (a1 + b1 + 1), "rho_wrong": 1 / (a0 + b0 + 1),
           "blind_spot_wrong_all_accept": float((c0.yes == c0.n).mean()),
           "wrong_accept_ge_90pct": float((c0.yes >= 0.9 * c0.n).mean()),
           "correct_all_reject": float((c1.yes == 0).mean()),
           "auroc_single_verdict": auroc(df["first"], df.correct),
           "auroc_32_verdicts": auroc(df.yes / df.n, df.correct),
           "auroc_vote_share": auroc(df.vote_share, df.correct),
           "auroc_32_verdicts_within_problem_mean": float(np.nanmean(
               [auroc(g.yes / g.n, g.correct) for _, g in df.groupby("problem")])),
           "corr_voteshare_vs_accept_wrong": float(np.corrcoef(c0.vote_share, c0.yes / c0.n)[0, 1]),
           "var_wrong": variance_shares(c0), "var_correct": variance_shares(c1)}
    # M2: moment test (fit Beta on first FIT_REPS repeats, predict held-out gates)
    m2 = {}
    for name, c in (("wrong", c0), ("correct", c1)):
        a, b = fit_beta_binomial(c.yes_fit, c.n_fit)
        abar = c.yes_fit.sum() / c.n_fit.sum()
        rows = {}
        for k in KS:
            emp = empirical_moment(c.yes_ho, c.n_ho, k)
            rows[k] = {"empirical": emp, "beta_pred": float(np.exp(betaln(a + k, b) - betaln(a, b))),
                       "indep_pred": float(abar ** k)}
        m2[name] = rows
    pi = len(c1) / (len(c1) + len(c0))
    m2["log_odds_after_k_accepts"] = {
        k: {"empirical": float(np.log(pi * m2["correct"][k]["empirical"] / ((1 - pi) * m2["wrong"][k]["empirical"]))),
            "indep": float(np.log(pi / (1 - pi)) + k * np.log(m2["correct"][1]["empirical"] / m2["wrong"][1]["empirical"]))}
        for k in KS}
    out["M2"] = m2
    return out


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def bootstrap(df, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("problem")]
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        bd = pd.concat([groups[i].assign(problem=t) for t, i in enumerate(pick)], ignore_index=True)
        samples.append(flatten(metrics(bd)))
    s = pd.DataFrame(samples)
    return {k: [float(np.nanpercentile(s[k], 2.5)), float(np.nanpercentile(s[k], 97.5))] for k in s.columns}


def main():
    arrays, split = load_public(TABLE)
    gold = load_gold(TABLE)
    df = cand_frame(arrays, gold, split)
    res = {}
    for name, sub in (("all", df), ("dev", df[df.split == "dev"]), ("test", df[df.split == "test"])):
        res[name] = {"point": metrics(sub), "ci95": bootstrap(sub, n_boot=N_BOOT if name == "all" else 500)}
    # coverage diagnostics (oracle, evaluation only)
    cov = {}
    for n in (1, 4, 16, 64, 256):
        cov[n] = float(np.mean([gold[p] in set(A.answers[:n]) for p, A in arrays.items()]))
    res["oracle_coverage_first_n_candidates"] = cov
    json.dump(res, open(os.path.join(OUT, "measurement.json"), "w"), indent=2)
    for name in ("all", "dev", "test"):
        pt, ci = flatten(res[name]["point"]), res[name]["ci95"]
        print(f"== {name}")
        for k, v in pt.items():
            c = ci.get(k)
            print(f"  {k:60s} {v:9.4f}" + (f"  [{c[0]:.4f}, {c[1]:.4f}]" if c else ""))
    print("coverage", cov)


if __name__ == "__main__":
    main()
