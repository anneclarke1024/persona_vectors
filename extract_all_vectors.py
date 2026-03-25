"""Extract all 4 depression steering vectors.

Loads the model once per vector (HuggingFace, not vLLM — needs hidden states).
Handles both matched-pair (Vector 1) and cross-trait (Vectors 2-4) filtering.

Usage:
    python extract_all_vectors.py                # extract all
    python extract_all_vectors.py --resume       # skip existing vectors
"""
import os
import sys
import time
import argparse
import shutil
import torch

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from generate_vec import save_persona_vector

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = "eval_persona_extract/Llama-3.1-8B-Instruct"
VEC_DIR = "persona_vectors/Llama-3.1-8B-Instruct"

VECTORS = [
    {
        "name": "depression",
        "pos_path": f"{OUT}/depression_pos_instruct.csv",
        "neg_path": f"{OUT}/depression_neg_instruct.csv",
        "trait": "depression",
    },
    {
        "name": "depression_vs_noprompt",
        "pos_path": f"{OUT}/depression_pos_instruct.csv",
        "neg_path": f"{OUT}/depression_no_prompt.csv",
        "trait": "depression",
    },
    {
        "name": "depression_vs_eudaimonic",
        "pos_path": f"{OUT}/depression_pos_instruct.csv",
        "neg_path": f"{OUT}/eudaimonic_pos_instruct.csv",
        "trait": "depression",
    },
    {
        "name": "depression_vs_simple_healthy",
        "pos_path": f"{OUT}/depression_pos_instruct.csv",
        "neg_path": f"{OUT}/simple_healthy_pos_instruct.csv",
        "trait": "depression",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(VEC_DIR, exist_ok=True)

    for vec in VECTORS:
        name = vec["name"]
        primary_file = f"{VEC_DIR}/{name}_response_avg_diff.pt"

        if args.resume and os.path.exists(primary_file):
            print(f"\n{'='*60}")
            print(f"SKIP: {name} (exists)")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting: {name}")
        print(f"  pos: {vec['pos_path']}")
        print(f"  neg: {vec['neg_path']}")
        print(f"  trait column: {vec['trait']}")
        print(f"{'='*60}")

        t0 = time.time()
        save_persona_vector(
            MODEL,
            vec["pos_path"],
            vec["neg_path"],
            vec["trait"],
            VEC_DIR,
            threshold=50,
        )
        elapsed = time.time() - t0

        # Rename generic files to vector-specific names
        # save_persona_vector saves as {trait}_*.pt, we want {name}_*.pt
        if name != vec["trait"]:
            for suffix in ["response_avg_diff", "prompt_avg_diff", "prompt_last_diff"]:
                src = f"{VEC_DIR}/{vec['trait']}_{suffix}.pt"
                dst = f"{VEC_DIR}/{name}_{suffix}.pt"
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"  Renamed: {os.path.basename(src)} -> {os.path.basename(dst)}")

        print(f"  Done in {elapsed:.0f}s")

    # Final validation
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    import glob
    for f in sorted(glob.glob(f"{VEC_DIR}/*_response_avg_diff.pt")):
        v = torch.load(f, weights_only=False)
        norms = v.norm(dim=1)
        top = norms.argsort(descending=True)[:3]
        name = os.path.basename(f)
        print(f"  {name}: shape={list(v.shape)}, top layers: {', '.join(f'L{l.item()}={norms[l]:.2f}' for l in top)}")


if __name__ == "__main__":
    main()
