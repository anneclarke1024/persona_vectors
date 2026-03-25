"""Score non-depression CSVs on the depression judge rubric.

This adds a 'depression' column to CSVs that don't have one (eudaimonic, simple_healthy, no_prompt),
enabling cross-trait filtering in generate_vec.py.

Usage:
    python score_on_depression.py --csv_dir eval_persona_extract/Llama-3.1-8B-Instruct/
"""
import os
import sys
import json
import asyncio
import argparse
import time

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import pandas as pd
from judge import OpenAiJudge
from eval.prompts import Prompts
from config import setup_credentials
from tqdm.asyncio import tqdm_asyncio

config = setup_credentials()

JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_CONCURRENT = 10


async def score_csv_on_depression(csv_path, depression_eval_prompt, overwrite=False):
    """Add 'depression' column to a CSV by scoring each response with the depression rubric."""
    df = pd.read_csv(csv_path)

    if "depression" in df.columns and not overwrite:
        print(f"  {csv_path}: 'depression' column already exists, skipping")
        return df

    print(f"  {csv_path}: scoring {len(df)} responses on depression rubric...")

    judge = OpenAiJudge(JUDGE_MODEL, depression_eval_prompt, eval_type="0_100")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def score_one(idx, question, answer):
        async with semaphore:
            return idx, await judge(question=question, answer=answer)

    tasks = [
        score_one(idx, row["question"], row["answer"])
        for idx, row in df.iterrows()
    ]

    results = {}
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Scoring"):
        idx, score = await coro
        results[idx] = score

    df["depression"] = pd.Series(results)

    null_count = df["depression"].isna().sum()
    if null_count > 0:
        print(f"  WARNING: {null_count} null scores (judge failures)")

    df.to_csv(csv_path, index=False)
    print(f"  Saved. depression: {df['depression'].mean():.1f} ± {df['depression'].std():.1f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Load depression eval prompt from trait data
    with open("data_generation/trait_data_extract/depression.json") as f:
        depression_data = json.load(f)
    depression_eval_prompt = depression_data["eval_prompt"]

    # Score each CSV that needs it
    csvs_to_score = [
        "eudaimonic_pos_instruct.csv",
        "simple_healthy_pos_instruct.csv",
        "depression_no_prompt.csv",
    ]

    print(f"Scoring CSVs on depression rubric (judge: {JUDGE_MODEL})")
    print(f"Max concurrent: {MAX_CONCURRENT}\n")

    for csv_name in csvs_to_score:
        csv_path = os.path.join(args.csv_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"  {csv_path}: NOT FOUND, skipping")
            continue
        asyncio.run(score_csv_on_depression(csv_path, depression_eval_prompt, args.overwrite))
        print()


if __name__ == "__main__":
    main()
