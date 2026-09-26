"""Descriptive (N, V) grids and cost accounting. Run only after the dev
selection was frozen; nothing here feeds back into policy selection."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from run_experiment import SEEDS, enumerate_rows, summarize  # noqa: E402
from ttsce.models import BayesParams  # noqa: E402
from ttsce.table import load_gold, load_public  # noqa: E402

TABLE = os.path.join(ROOT, "data", "tables", "gpqa_l33_70b")
OUT = os.path.join(ROOT, "results", "gpqa_l33_70b")
FLOPS_PER_TOKEN = 2 * 70e9


def main():
    arrays, split = load_public(TABLE)
    gold = load_gold(TABLE)
    frozen = json.load(open(os.path.join(OUT, "dev_selection.json")))
    params = {k: BayesParams(**v) for k, v in frozen["params"].items()}
    c_gen = frozen["c_gen_tokens"]
    report = {}
    for name in ("dev", "test"):
        probs = sorted(p for p, s in split.items() if s == name)
        df = enumerate_rows(arrays, gold, probs, params, SEEDS)
        s = summarize(df)
        s["cost_x_cgen"] = s.cost / c_gen
        s.to_csv(os.path.join(OUT, f"grid_{name}.csv"), index=False)
        fx = s[s.policy.str.startswith("FIX:")].copy()
        fx[["aggregator", "N", "V"]] = fx.policy.str.extract(r"FIX:([^:]+):N(\d+):V(\d+)")
        fx[["N", "V"]] = fx[["N", "V"]].astype(int)
        # marginal value per 1e4 tokens of doubling V (same candidates) vs doubling N (same V)
        mv = []
        for agg in ("BoN", "WMV", "BAYES-CORR"):
            g = fx[fx.aggregator == agg].set_index(["N", "V"])
            for n in (4, 8, 16):
                for v in (1, 2, 4, 8, 16):
                    base = g.loc[(n, v)]
                    rep = g.loc[(n, 2 * v)]
                    wid = g.loc[(2 * n, v)]
                    mv.append(dict(agg=agg, N=n, V=v,
                                   d_err_per_1e4tok_double_V=(rep.err - base.err) / (rep.cost - base.cost) * 1e4,
                                   d_err_per_1e4tok_double_N=(wid.err - base.err) / (wid.cost - base.cost) * 1e4))
        mvd = pd.DataFrame(mv)
        mvd.to_csv(os.path.join(OUT, f"marginal_value_{name}.csv"), index=False)
        report[name] = {
            "n_problems": len(probs),
            "share_cells_where_doubling_V_beats_doubling_N": float(
                (mvd.d_err_per_1e4tok_double_V < mvd.d_err_per_1e4tok_double_N).mean()),
            "median_d_err_per_1e4tok_double_V": float(mvd.d_err_per_1e4tok_double_V.median()),
            "median_d_err_per_1e4tok_double_N": float(mvd.d_err_per_1e4tok_double_N.median()),
            "mv": {int(n): float(fx[(fx.aggregator == "MV") & (fx.N == n)].err.iloc[0]) for n in (1, 4, 16, 64, 128, 256)},
            "best_fixed_any_cost": fx.sort_values("err").iloc[0][["policy", "err", "cost_x_cgen"]].to_dict(),
        }
    # full response-table collection cost (paid by the original authors, not by this study)
    gen = sum(int(A.gen_in.sum() + A.gen_out.sum()) for A in arrays.values())
    ver = sum(int(A.ver_in.sum() * A.verdicts.shape[1] + A.ver_out.sum()) for A in arrays.values())
    report["full_table_collection_cost"] = {
        "generation_tokens": gen, "verification_tokens": ver, "total_tokens": gen + ver,
        "approx_flops": (gen + ver) * FLOPS_PER_TOKEN,
        "note": "Llama-3.3-70B for both roles; FLOPs ~ 2*params*tokens; no prefix caching"}
    json.dump(report, open(os.path.join(OUT, "grid_report.json"), "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
