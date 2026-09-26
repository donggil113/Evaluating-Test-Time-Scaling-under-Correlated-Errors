#!/usr/bin/env bash
# Download the released traces used by protocol N5-v1 and verify checksums.
# Total ~72 MB (two archives + Llama-3 tokenizer for token counting).
set -euo pipefail
RAW=${1:-raw}
mkdir -p "$RAW/gpqa"
HF=https://huggingface.co
get() { curl -sSL -o "$RAW/$2" "$HF/$1"; echo "$3  $RAW/$2" | sha256sum -c -; }
get datasets/sc-genrm-scaling/GPQA_diamond_Solutions_Llama-3.3-70B-Instruct/resolve/main/compressed_Llama-3.3-70B-Instruct.tar.gz \
    gpqa_solutions_llama33_70b.tar.gz af885f3972681497b5ee4f9964fce5d0ca731ca8aaa912e6166d155e2b9b191e
get datasets/sc-genrm-scaling/GPQA_verifications_GenRM-Base_Llama-3.3-70B-Instruct/resolve/main/compressed_GPQA_verifications_GenRM-Base_Llama-3.3-70B-Instruct.tar.gz \
    gpqa_verifs_genrm_base_llama33_70b.tar.gz fea30c7e59fcbd4cbd38efb97e85f2081c66e1d33b17a780b25a4b84e381c2f9
get unsloth/Llama-3.3-70B-Instruct/resolve/main/tokenizer.json \
    llama3_tokenizer.json 6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b
tar xzf "$RAW/gpqa_solutions_llama33_70b.tar.gz" -C "$RAW/gpqa"
tar xzf "$RAW/gpqa_verifs_genrm_base_llama33_70b.tar.gz" -C "$RAW/gpqa"
echo "then: python scripts/build_table.py --raw $RAW/gpqa --tokenizer $RAW/llama3_tokenizer.json --out data/tables/gpqa_l33_70b"
