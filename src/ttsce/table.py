"""Load the frozen response table into per-problem arrays.

`load_public` returns everything a policy may ever see (answers, verdicts,
token counts). `load_gold` returns the sealed answer key and must only be
called by evaluation / dev-calibration code, never by a policy.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

LETTERS = "ABCD"
ABSTAIN = -1
NO_VERDICT = -1


@dataclass
class ProblemArrays:
    problem: int
    answers: np.ndarray        # (n_cand,) int in {0..3} or ABSTAIN
    gen_in: np.ndarray         # (n_cand,) prompt tokens of each generation call
    gen_out: np.ndarray        # (n_cand,) completion tokens
    n_verified: int            # candidates [0, n_verified) have verifier calls
    verdicts: np.ndarray       # (n_verified, n_rep) int in {0,1} or NO_VERDICT
    ver_in: np.ndarray         # (n_verified,) prompt tokens of a verifier call
    ver_out: np.ndarray        # (n_verified, n_rep) completion tokens


def load_public(table_dir):
    c = pd.read_csv(os.path.join(table_dir, "candidates.csv.gz"))
    v = pd.read_csv(os.path.join(table_dir, "verifications.csv.gz"))
    probs = pd.read_csv(os.path.join(table_dir, "problems.csv"))
    out = {}
    for p, cg in c.groupby("problem"):
        cg = cg.sort_values("cand")
        ans = cg["answer"].map(lambda a: LETTERS.index(a) if isinstance(a, str) else ABSTAIN).to_numpy()
        vg = v[v.problem == p].sort_values(["cand", "rep"])
        n_ver = vg["cand"].nunique()
        n_rep = vg["rep"].nunique()
        assert (np.sort(vg["cand"].unique()) == np.arange(n_ver)).all()
        verd = vg["verdict"].fillna(NO_VERDICT).astype(int).to_numpy().reshape(n_ver, n_rep)
        vin = vg.groupby("cand")["ver_in_tok"].first().to_numpy()
        vout = vg["ver_out_tok"].to_numpy().reshape(n_ver, n_rep)
        out[p] = ProblemArrays(p, ans, cg["gen_in_tok"].to_numpy(), cg["gen_out_tok"].to_numpy(),
                               n_ver, verd, vin, vout)
    split = dict(zip(probs["problem"], probs["split"]))
    return out, split


def load_gold(table_dir):
    g = pd.read_csv(os.path.join(table_dir, "gold.csv"))
    return {int(p): LETTERS.index(a) for p, a in zip(g["problem"], g["gt_answer"])}
