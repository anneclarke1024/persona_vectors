"""Generate trait data JSON files using Chen et al.'s meta-prompt via Claude Sonnet.

Usage:
    # Dry run (prints the prompt, doesn't call API):
    python generate_trait_data.py --trait depression --trait_definition_path ../../02_prompts/trait_definition_depression_v2.txt --dry-run

    # Real run:
    python generate_trait_data.py --trait depression --trait_definition_path ../../02_prompts/trait_definition_depression_v2.txt
"""
import argparse
import json
import os
import re
import anthropic
from data_generation.prompts import PROMPTS


def generate_trait_data(trait: str, trait_definition: str, question_instruction: str = "", dry_run: bool = False):
    """Call Claude Sonnet with Chen's meta-prompt to generate trait data."""
    prompt = PROMPTS["generate_trait"].format(
        TRAIT=trait,
        trait_instruction=trait_definition,
        question_instruction=question_instruction,
    )

    if dry_run:
        print("=" * 60)
        print("DRY RUN — would send this prompt to claude-sonnet-4-20250514:")
        print("=" * 60)
        print(prompt[:2000])
        print(f"\n... ({len(prompt)} chars total)")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text

    # Extract JSON from response (may be wrapped in markdown code block)
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        raise ValueError(f"No JSON found in response:\n{response_text[:500]}")

    data = json.loads(json_match.group())

    # Validate structure
    assert "instruction" in data, "Missing 'instruction' key"
    assert "questions" in data, "Missing 'questions' key"
    assert "eval_prompt" in data, "Missing 'eval_prompt' key"
    assert len(data["instruction"]) == 5, f"Expected 5 instruction pairs, got {len(data['instruction'])}"
    assert len(data["questions"]) == 40, f"Expected 40 questions, got {len(data['questions'])}"

    for i, pair in enumerate(data["instruction"]):
        assert "pos" in pair and "neg" in pair, f"Pair {i} missing pos or neg"

    return data


def split_and_save(data: dict, trait: str, output_dir: str):
    """Split 40 questions into extract (first 20) and eval (last 20), save both."""
    extract_dir = os.path.join(output_dir, "trait_data_extract")
    eval_dir = os.path.join(output_dir, "trait_data_eval")
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    extract_data = {
        "instruction": data["instruction"],
        "questions": data["questions"][:20],
        "eval_prompt": data["eval_prompt"],
    }
    eval_data = {
        "instruction": data["instruction"],
        "questions": data["questions"][20:],
        "eval_prompt": data["eval_prompt"],
    }

    extract_path = os.path.join(extract_dir, f"{trait}.json")
    eval_path = os.path.join(eval_dir, f"{trait}.json")

    with open(extract_path, "w") as f:
        json.dump(extract_data, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"Saved: {extract_path} ({len(extract_data['questions'])} questions)")
    print(f"Saved: {eval_path} ({len(eval_data['questions'])} questions)")

    return extract_path, eval_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", type=str, required=True, help="Trait name (e.g., depression)")
    parser.add_argument("--trait_definition_path", type=str, required=True, help="Path to trait definition text file")
    parser.add_argument("--output_dir", type=str, default="data_generation", help="Base output directory")
    parser.add_argument("--question-instruction", type=str, default="", help="Additional instructions for question generation (injected into {question_instruction} slot)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling API")
    args = parser.parse_args()

    with open(args.trait_definition_path) as f:
        trait_definition = f.read().strip()

    print(f"Trait: {args.trait}")
    print(f"Definition: {trait_definition[:100]}...")
    print()

    data = generate_trait_data(args.trait, trait_definition, question_instruction=args.question_instruction, dry_run=args.dry_run)

    if data is not None:
        # Save raw response
        raw_path = os.path.join(args.output_dir, f"raw_{args.trait}.json")
        with open(raw_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Raw output saved: {raw_path}")

        split_and_save(data, args.trait, args.output_dir)
