"""Write results/manifest.json: what ran, on which inputs, with which outputs."""
import glob
import hashlib
import json
import os
import platform

ROOT = os.path.join(os.path.dirname(__file__), "..")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def files(pattern):
    return {os.path.relpath(p, ROOT): sha(p) for p in sorted(glob.glob(os.path.join(ROOT, pattern)))}


RES = "results/gpqa_l33_70b"
manifest = {
    "experiment_id": "N5-GPQA-L33-70B-selfverify-v1",
    "protocol": "docs/protocol.md (frozen at commit 7bbebce)",
    "machine": {"cpu_only": True, "n_cpu": os.cpu_count(), "python": platform.python_version(),
                "gpu": None, "llm_api_calls": 0},
    "downloads": {
        "total_bytes": 9120721 + 45989747 + 17209920,
        "files": {
            "sc-genrm-scaling/GPQA_diamond_Solutions_Llama-3.3-70B-Instruct/compressed_Llama-3.3-70B-Instruct.tar.gz":
                "af885f3972681497b5ee4f9964fce5d0ca731ca8aaa912e6166d155e2b9b191e",
            "sc-genrm-scaling/GPQA_verifications_GenRM-Base_Llama-3.3-70B-Instruct/compressed_GPQA_verifications_GenRM-Base_Llama-3.3-70B-Instruct.tar.gz":
                "fea30c7e59fcbd4cbd38efb97e85f2081c66e1d33b17a780b25a4b84e381c2f9",
            "unsloth/Llama-3.3-70B-Instruct/tokenizer.json (token counting only)":
                "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"}},
    "runs": [
        {"id": "R0-build", "cmd": "scripts/fetch_data.sh raw && python scripts/build_table.py --raw raw/gpqa "
                                  "--tokenizer raw/llama3_tokenizer.json --out data/tables/gpqa_l33_70b",
         "status": "DONE", "wall_s_approx": 194, "note": "run 3 times; parser revised twice on output text "
         "format only (answer 'is: X', verdict '(No)'), before any outcome analysis; outputs overwritten"},
        {"id": "R1-dev", "cmd": "python scripts/run_experiment.py --phase dev", "status": "DONE",
         "record": f"{RES}/run_dev.json",
         "note": "run twice; second run after deviation D1 (tau grid); working tree = 7bbebce + D1, committed as 66abf24"},
        {"id": "R2-test", "cmd": "python scripts/run_experiment.py --phase test", "status": "DONE (once)",
         "record": f"{RES}/run_test.json", "commit": "66abf24"},
        {"id": "R3-measure", "cmd": "python scripts/analyze_correlation.py", "status": "DONE", "wall_s_approx": 140},
        {"id": "R4-describe", "cmd": "python scripts/describe_grid.py", "status": "DONE", "wall_s_approx": 115,
         "note": "descriptive only, after selection was frozen"},
        {"id": "T-leak", "cmd": "python -m pytest tests -q", "status": "PASS"},
    ],
    "not_run": {
        "diversified_verifier_same_pool": "NOT_RUN: released GPQA pool has one verifier. MATH-128 Qwen-2.5-7B "
            "solutions have two fine-tuned verifiers with repeats (HF nishadsinghi/math128_qwen2p57B_VER-Llama-3.1-8B-"
            "Inst_... 185 MB, sc-genrm-scaling/MATH128_verifications_GenRM-FT_Qwen-2.5-7B-Instruct 1.04 GB) - needs "
            "approval for ~1.2 GB new download.",
        "history_dependent_query_policies": "NOT_RUN: require a live online runner; no inference access (no GPU, no API key).",
        "code_domain": "NOT_RUN: no released code traces with repeated verifier calls and independent hidden tests "
            "were identified within the download budget.",
    },
    "inputs": files("data/tables/gpqa_l33_70b/*"),
    "outputs": files(f"{RES}/*"),
}
json.dump(manifest, open(os.path.join(ROOT, "results", "manifest.json"), "w"), indent=2)
print("wrote results/manifest.json")
