"""Run all 5 generation + scoring steps with a single vLLM model load.

Usage:
    python run_all_generations.py                    # full run (n_per_question=10)
    python run_all_generations.py --n_per_question 1 # dry run
    python run_all_generations.py --resume           # skip existing outputs
"""
import os
import sys
import json
import asyncio
import argparse
import time
from datetime import datetime

# Ensure HF cache is on workspace
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# Use vLLM V0 engine — V1 hangs on some RunPod configurations
os.environ.setdefault("VLLM_USE_V1", "0")

from eval.eval_persona import load_persona_questions, eval_batched, sample
from eval.model_utils import load_vllm_model
from config import setup_credentials
import pandas as pd

config = setup_credentials()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
VERSION = "extract"
MAX_CONCURRENT_JUDGES = 30

RUNS = [
    {
        "name": "depression_enhance",
        "trait": "depression",
        "persona_instruction_type": "pos",
        "assistant_name": "depressed",
        "output": "depression_pos_instruct.csv",
    },
    {
        "name": "depression_suppress",
        "trait": "depression",
        "persona_instruction_type": "neg",
        "assistant_name": None,  # defaults to "helpful"
        "output": "depression_neg_instruct.csv",
    },
    {
        "name": "eudaimonic_enhance",
        "trait": "eudaimonic",
        "persona_instruction_type": "pos",
        "assistant_name": None,  # defaults to "eudaimonic"
        "output": "eudaimonic_pos_instruct.csv",
    },
    {
        "name": "simple_healthy_enhance",
        "trait": "simple_healthy",
        "persona_instruction_type": "pos",
        "assistant_name": "happy",
        "output": "simple_healthy_pos_instruct.csv",
    },
    {
        "name": "no_prompt",
        "trait": "depression",
        "persona_instruction_type": None,  # no system prompt
        "assistant_name": None,
        "output": "depression_no_prompt.csv",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_question", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    n_per_question = args.n_per_question
    temperature = 0.0 if n_per_question == 1 else 1.0

    # Create output directory
    out_dir = f"eval_persona_extract/{os.path.basename(MODEL)}"
    os.makedirs(out_dir, exist_ok=True)

    # Create run folder for provenance
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/{'dryrun' if n_per_question == 1 else 'run'}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    run_config = {
        "timestamp": timestamp,
        "model": MODEL,
        "judge_model": JUDGE_MODEL,
        "n_per_question": n_per_question,
        "temperature": temperature,
        "max_concurrent_judges": MAX_CONCURRENT_JUDGES,
        "version": VERSION,
    }
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print("=" * 60)
    print(f"Depression Pipeline — {n_per_question} per question")
    print(f"Run dir: {run_dir}")
    print("=" * 60)

    # Load vLLM ONCE
    print("\nLoading vLLM model (one-time)...")
    t0 = time.time()
    llm, tokenizer, lora_path = load_vllm_model(MODEL)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Run all generation + scoring steps
    for run in RUNS:
        output_path = f"{out_dir}/{run['output']}"
        print(f"── {run['name']} ──")
        print(f"   Output: {output_path}")

        if args.resume and os.path.exists(output_path):
            print(f"   SKIPPED (exists)")
            continue

        t1 = time.time()

        questions = load_persona_questions(
            run["trait"],
            temperature=temperature,
            persona_instructions_type=run["persona_instruction_type"],
            assistant_name=run["assistant_name"],
            judge_model=JUDGE_MODEL,
            version=VERSION,
        )

        print(f"   Generating {len(questions)} × {n_per_question} = {len(questions) * n_per_question} responses...")
        outputs_list = asyncio.run(
            eval_batched(
                questions, llm, tokenizer,
                coef=0, vector=None, layer=None,
                n_per_question=n_per_question,
                max_concurrent_judges=MAX_CONCURRENT_JUDGES,
                max_tokens=1000,
                lora_path=lora_path,
            )
        )
        outputs = pd.concat(outputs_list)
        outputs.to_csv(output_path, index=False)

        # Find trait column
        trait_cols = [c for c in outputs.columns if c not in ['question', 'prompt', 'answer', 'question_id', 'coherence']]
        trait_col = trait_cols[0] if trait_cols else None

        elapsed = time.time() - t1
        print(f"   {len(outputs)} rows, {elapsed:.0f}s")
        if trait_col:
            print(f"   {trait_col}: {outputs[trait_col].mean():.1f} ± {outputs[trait_col].std():.1f}")
        print(f"   coherence: {outputs['coherence'].mean():.1f} ± {outputs['coherence'].std():.1f}")
        print()

    print("=" * 60)
    print(f"All generation + scoring complete. Outputs in {out_dir}/")
    print(f"Run log in {run_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
