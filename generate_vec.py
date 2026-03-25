from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    

def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer+1))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu())
        del outputs
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg

import pandas as pd
import os
import re

SELF_REF_PATTERN = re.compile(
    r'language model|an AI|artificial intelligence|I don.t have personal|'
    r'I don.t experience|I don.t have feelings|as an AI|I.m not a person|'
    r'don.t have physical|digital assistant|virtual assistant|'
    r'cannot physically|I exist as',
    re.IGNORECASE
)

def get_persona_effective(pos_path, neg_path, trait, threshold=50):
    """Filter pos and neg responses for vector extraction.

    Supports two modes:
    - Same-trait (matched pairs): pos and neg CSVs have same row count and are paired by index.
      Filter: pos[trait] >= threshold AND neg[trait] < (100-threshold) AND both coherent.
    - Cross-trait (independent): pos and neg CSVs may have different row counts or come from
      different traits. Both must have a 'depression' column (added by score_on_depression.py).
      Filter each side independently: pos[trait] >= threshold AND coherent;
      neg[trait] < (100-threshold) AND coherent AND not self-referencing.
    """
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)

    matched_pairs = (len(persona_pos) == len(persona_neg))

    if matched_pairs:
        # Same-trait mode: Chen's original matched-pair filter
        print(f"Filtering (matched-pair mode): {len(persona_pos)} pairs")

        mask = (persona_pos[trait] >= threshold) & (persona_neg[trait] < 100-threshold) & \
               (persona_pos["coherence"] >= 50) & (persona_neg["coherence"] >= 50)
        n_score_dropped = (~mask).sum()

        self_ref_mask = ~persona_neg["answer"].str.contains(SELF_REF_PATTERN, regex=True)
        mask = mask & self_ref_mask
        n_self_ref_dropped = (~self_ref_mask).sum()

        print(f"  Score/coherence filter: dropped {n_score_dropped}")
        print(f"  Self-reference filter: dropped {n_self_ref_dropped}")
        print(f"  Surviving pairs: {mask.sum()}")

        persona_pos_effective = persona_pos[mask]
        persona_neg_effective = persona_neg[mask]
    else:
        # Cross-trait mode: filter each side independently
        print(f"Filtering (independent mode): {len(persona_pos)} pos, {len(persona_neg)} neg")

        # Pos: must score high on trait and be coherent
        pos_mask = (persona_pos[trait] >= threshold) & (persona_pos["coherence"] >= 50)
        pos_self_ref = persona_pos["answer"].str.contains(SELF_REF_PATTERN, regex=True)
        pos_mask = pos_mask & ~pos_self_ref

        # Neg: must score low on trait and be coherent and not self-referencing
        neg_mask = (persona_neg[trait] < 100-threshold) & (persona_neg["coherence"] >= 50)
        neg_self_ref = persona_neg["answer"].str.contains(SELF_REF_PATTERN, regex=True)
        neg_mask = neg_mask & ~neg_self_ref

        print(f"  Pos surviving: {pos_mask.sum()}/{len(persona_pos)} (self-ref dropped: {pos_self_ref.sum()})")
        print(f"  Neg surviving: {neg_mask.sum()}/{len(persona_neg)} (self-ref dropped: {neg_self_ref.sum()})")

        persona_pos_effective = persona_pos[pos_mask]
        persona_neg_effective = persona_neg[neg_mask]

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses


def save_persona_vector(model_name, pos_path, neg_path, trait, save_dir, threshold=50):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses = get_persona_effective(pos_path, neg_path, trait, threshold)

    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}

    persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
    persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)
    


    persona_effective_prompt_avg_diff = torch.stack([persona_effective_prompt_avg["pos"][l].mean(0).float() - persona_effective_prompt_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_avg["pos"]))], dim=0)
    persona_effective_response_avg_diff = torch.stack([persona_effective_response_avg["pos"][l].mean(0).float() - persona_effective_response_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_response_avg["pos"]))], dim=0)
    persona_effective_prompt_last_diff = torch.stack([persona_effective_prompt_last["pos"][l].mean(0).float() - persona_effective_prompt_last["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_last["pos"]))], dim=0)

    os.makedirs(save_dir, exist_ok=True)

    torch.save(persona_effective_prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff.pt")
    torch.save(persona_effective_response_avg_diff, f"{save_dir}/{trait}_response_avg_diff.pt")
    torch.save(persona_effective_prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")

    print(f"Persona vectors saved to {save_dir}")    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    args = parser.parse_args()
    save_persona_vector(args.model_name, args.pos_path, args.neg_path, args.trait, args.save_dir, args.threshold)