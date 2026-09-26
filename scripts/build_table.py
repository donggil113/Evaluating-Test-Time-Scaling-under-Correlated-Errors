"""Build the frozen problem x candidate x verifier x repeat response table.

Input: the released archives of Singhi et al. (2025), "When To Solve, When To
Verify" (HF org `sc-genrm-scaling`), extracted to --raw.
Output (under --out):
  problems.csv          problem id, question hash, dev/test split
  candidates.csv.gz     one row per generated candidate (answer + token counts)
  verifications.csv.gz  one row per verifier call (verdict + token counts)
  gold.csv              SEALED answer key; read only by the evaluator
  build_log.json        counts of parse failures and integrity checks

The split is a deterministic function of the question text only, so it is
fixed before any outcome is looked at.
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd
import yaml
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ttsce.parse import extract_choice, extract_verdict  # noqa: E402

SPLIT_SALT = "ttsce-split-v1|"


def ntok(tok, texts):
    return [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sol_dir", default="Llama-3.3-70B-Instruct_2K_tokens")
    ap.add_argument("--ver_dir", default="data_cleaned")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verifier", default="Llama-3.3-70B-Instruct:GenRM-Base")
    ap.add_argument("--n_dev", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)

    sol_path = os.path.join(args.raw, args.sol_dir)
    pids = sorted(int(f[:-5]) for f in os.listdir(sol_path) if f.endswith(".yaml"))
    probs, cands, vers, gold = [], [], [], []
    log = {"problems": len(pids), "question_mismatch": 0, "solution_text_mismatch": 0,
           "missing_verification_files": 0}
    for p in pids:
        s = yaml.safe_load(open(os.path.join(sol_path, f"{p}.yaml")))
        q = s["question"]
        probs.append({"problem": p, "question_sha256": hashlib.sha256(q.encode()).hexdigest()})
        gold.append({"problem": p, "gt_answer": s["gt_answer"]})
        samples = s["samples"]
        gen_in = ntok(tok, [s["prompt"]])[0]
        gen_out = ntok(tok, samples)
        vdir = os.path.join(args.raw, args.ver_dir, f"problem_{p}")
        for j, text in enumerate(samples):
            vf = os.path.join(vdir, f"solution_{j}.yaml")
            has_v = os.path.exists(vf)
            cands.append({"problem": p, "cand": j, "answer": extract_choice(text),
                          "gen_in_tok": gen_in, "gen_out_tok": gen_out[j], "has_verifications": has_v})
            if not has_v:
                continue
            v = yaml.safe_load(open(vf))
            prompt = v["prompt"]
            if q.strip()[:200] not in prompt:
                log["question_mismatch"] += 1
            if text.strip()[-300:] not in prompt:
                log["solution_text_mismatch"] += 1
            vin = ntok(tok, [prompt])[0]
            outs = [r["verification"] for r in v["verifications"]]
            vout = ntok(tok, outs)
            for r, (rec, o) in enumerate(zip(v["verifications"], vout)):
                vers.append({"problem": p, "cand": j, "verifier": args.verifier, "rep": r,
                             "verdict": extract_verdict(rec["verification"]),
                             "p_yes_raw": rec["p(yes)"], "p_no_raw": rec["p(no)"],
                             "ver_in_tok": vin, "ver_out_tok": o})
        print(f"problem {p}: {len(samples)} candidates", flush=True)

    probs = pd.DataFrame(probs)
    order = probs["question_sha256"].map(lambda h: hashlib.sha256((SPLIT_SALT + h).encode()).hexdigest())
    ranked = probs.assign(_k=order).sort_values("_k")["problem"].tolist()
    dev = set(ranked[: args.n_dev])
    probs["split"] = probs["problem"].map(lambda x: "dev" if x in dev else "test")

    cands, vers = pd.DataFrame(cands), pd.DataFrame(vers)
    log["candidates"] = len(cands)
    log["candidates_with_verifications"] = int(cands["has_verifications"].sum())
    log["candidate_answer_unparsed"] = int(cands["answer"].isna().sum())
    log["verifications"] = len(vers)
    log["verdict_unparsed"] = int(vers["verdict"].isna().sum())
    log["repeats_per_candidate"] = sorted(vers.groupby(["problem", "cand"]).size().unique().tolist())
    log["split_counts"] = probs["split"].value_counts().to_dict()

    probs.to_csv(os.path.join(args.out, "problems.csv"), index=False)
    cands.to_csv(os.path.join(args.out, "candidates.csv.gz"), index=False)
    vers.to_csv(os.path.join(args.out, "verifications.csv.gz"), index=False)
    pd.DataFrame(gold).to_csv(os.path.join(args.out, "gold.csv"), index=False)
    json.dump(log, open(os.path.join(args.out, "build_log.json"), "w"), indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
