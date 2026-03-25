"""Sweep layers and alpha values for steering validation.

Loads the model ONCE via HuggingFace (not vLLM — needs forward hooks for steering),
then generates steered responses across layer × alpha combinations and scores them.

Usage:
    # Quick test at one layer/alpha
    python run_steering_sweep.py --vector_path persona_vectors/Llama-3.1-8B-Instruct/depression_response_avg_diff.pt \
        --layers 15 --alphas 1.0 --n_per_question 1

    # Full sweep
    python run_steering_sweep.py --vector_path persona_vectors/Llama-3.1-8B-Instruct/depression_response_avg_diff.pt \
        --layers 14 15 16 17 18 19 20 --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0

    # Resume (skips existing output files)
    python run_steering_sweep.py ... --resume
"""
import os
import sys
import json
import asyncio
import argparse
import time
from datetime import datetime

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
import pandas as pd
from eval.eval_persona import load_persona_questions, eval_batched, sample_steering
from eval.model_utils import load_model
from config import setup_credentials
from transformers import AutoTokenizer

config = setup_credentials()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_CONCURRENT_JUDGES = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_path", type=str, required=True, help="Path to .pt vector file")
    parser.add_argument("--layers", type=int, nargs="+", required=True, help="Layer indices to sweep")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], help="Alpha coefficients (exclude 0.0 — baseline already exists from generation)")
    parser.add_argument("--n_per_question", type=int, default=1, help="Responses per question")
    parser.add_argument("--steering_type", type=str, default="response", choices=["all", "prompt", "response"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    temperature = 0.0 if args.n_per_question == 1 else 1.0
    vector_name = os.path.basename(args.vector_path).replace("_response_avg_diff.pt", "").replace(".pt", "")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"steering_sweep/{vector_name}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    sweep_config = {
        "vector_path": args.vector_path,
        "vector_name": vector_name,
        "layers": args.layers,
        "alphas": args.alphas,
        "n_per_question": args.n_per_question,
        "steering_type": args.steering_type,
        "model": MODEL,
        "judge_model": JUDGE_MODEL,
        "timestamp": timestamp,
    }
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    print("=" * 60)
    print(f"Steering Sweep: {vector_name}")
    print(f"Layers: {args.layers}")
    print(f"Alphas: {args.alphas}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # Load model via HuggingFace (not vLLM — needs forward hooks)
    print("\nLoading model via HuggingFace...")
    t0 = time.time()
    model, tokenizer = load_model(MODEL)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load vector
    full_vector = torch.load(args.vector_path, weights_only=False)
    print(f"Vector shape: {list(full_vector.shape)}")

    # Load eval questions (the held-out set)
    questions = load_persona_questions(
        "depression",
        temperature=temperature,
        judge_model=JUDGE_MODEL,
        version="eval",
    )
    print(f"Loaded {len(questions)} eval questions\n")

    # Sweep
    results = []
    total = len(args.layers) * len(args.alphas)
    done = 0

    for layer in args.layers:
        # Extract the 1-D vector for this layer (ActivationSteerer expects 1-D)
        layer_vector = full_vector[layer]

        for alpha in args.alphas:
            done += 1
            output_path = f"{out_dir}/layer{layer}_alpha{alpha:.1f}.csv"

            print(f"[{done}/{total}] Layer {layer}, alpha {alpha:.1f}")

            if args.resume and os.path.exists(output_path):
                print(f"  SKIPPED (exists)")
                df = pd.read_csv(output_path)
            else:
                t1 = time.time()

                outputs_list = asyncio.run(
                    eval_batched(
                        questions, model, tokenizer,
                        coef=alpha, vector=layer_vector, layer=layer,
                        n_per_question=args.n_per_question,
                        max_concurrent_judges=MAX_CONCURRENT_JUDGES,
                        max_tokens=1000,
                        steering_type=args.steering_type,
                    )
                )
                df = pd.concat(outputs_list)
                df.to_csv(output_path, index=False)

                elapsed = time.time() - t1
                print(f"  {len(df)} rows, {elapsed:.0f}s")

            # Extract scores
            trait_cols = [c for c in df.columns if c not in ['question', 'prompt', 'answer', 'question_id', 'coherence']]
            trait_col = trait_cols[0] if trait_cols else "depression"
            trait_mean = df[trait_col].mean()
            trait_std = df[trait_col].std()
            coh_mean = df["coherence"].mean()
            coh_std = df["coherence"].std()

            print(f"  {trait_col}: {trait_mean:.1f}±{trait_std:.1f}, coherence: {coh_mean:.1f}±{coh_std:.1f}")

            results.append({
                "layer": layer,
                "alpha": alpha,
                "trait_mean": trait_mean,
                "trait_std": trait_std,
                "coherence_mean": coh_mean,
                "coherence_std": coh_std,
                "n_responses": len(df),
            })

    # Save summary
    summary = pd.DataFrame(results)
    summary.to_csv(f"{out_dir}/sweep_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nSaved to {out_dir}/sweep_summary.csv")

    # Find sweet spot: highest depression with coherence > 70
    viable = summary[summary["coherence_mean"] > 70]
    if len(viable) > 0:
        best = viable.loc[viable["trait_mean"].idxmax()]
        print(f"\n  → Best: layer {int(best['layer'])}, alpha {best['alpha']:.1f}")
        print(f"    depression {best['trait_mean']:.1f}, coherence {best['coherence_mean']:.1f}")


if __name__ == "__main__":
    main()
