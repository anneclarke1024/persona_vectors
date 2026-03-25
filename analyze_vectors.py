"""Analyze extracted steering vectors: norms, cosine similarity, layer selection.

Usage:
    python analyze_vectors.py --vec_dir persona_vectors/Llama-3.1-8B-Instruct/
"""
import argparse
import glob
import os
import torch
import numpy as np


def load_vectors(vec_dir):
    """Load all response_avg_diff vectors from a directory."""
    vectors = {}
    for f in sorted(glob.glob(f"{vec_dir}/*_response_avg_diff.pt")):
        name = os.path.basename(f).replace("_response_avg_diff.pt", "")
        vectors[name] = torch.load(f, weights_only=False)
    return vectors


def print_norms(vectors):
    """Print per-layer norms for each vector. Identify strongest layers."""
    print("=" * 80)
    print("PER-LAYER NORMS")
    print("=" * 80)

    for name, vec in vectors.items():
        norms = vec.norm(dim=1)
        top_layers = norms.argsort(descending=True)[:5]
        print(f"\n  {name} (shape {list(vec.shape)}):")
        print(f"    Mean norm: {norms.mean():.4f}")
        print(f"    Top 5 layers: {', '.join(f'L{l.item()}={norms[l]:.4f}' for l in top_layers)}")

    # Combined table
    print(f"\n  {'Layer':>6}", end="")
    for name in vectors:
        print(f"  {name[:20]:>20}", end="")
    print()
    print("  " + "-" * (6 + 22 * len(vectors)))

    num_layers = list(vectors.values())[0].shape[0]
    for layer in range(num_layers):
        print(f"  {layer:>6}", end="")
        for name, vec in vectors.items():
            norm = vec[layer].norm().item()
            print(f"  {norm:>20.4f}", end="")
        print()


def print_cosine_similarity(vectors):
    """Print pairwise cosine similarity between vectors at each layer."""
    print("\n" + "=" * 80)
    print("PAIRWISE COSINE SIMILARITY (per layer)")
    print("=" * 80)

    names = list(vectors.keys())
    if len(names) < 2:
        print("  Need at least 2 vectors for comparison.")
        return

    num_layers = list(vectors.values())[0].shape[0]

    # For each pair of vectors
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a, b = vectors[a_name], vectors[b_name]

            cos_sims = torch.nn.functional.cosine_similarity(a, b, dim=1)

            top_layers = cos_sims.argsort(descending=True)[:5]
            print(f"\n  {a_name} vs {b_name}:")
            print(f"    Mean cosine sim: {cos_sims.mean():.4f}")
            print(f"    Top 5 layers: {', '.join(f'L{l.item()}={cos_sims[l]:.4f}' for l in top_layers)}")

    # Summary table at top-norm layers
    print(f"\n  Cosine similarity at top-norm layers:")

    # Find top layers by average norm across all vectors
    avg_norms = torch.stack([v.norm(dim=1) for v in vectors.values()]).mean(dim=0)
    top_layers = avg_norms.argsort(descending=True)[:10]

    header = f"  {'Layer':>6} {'AvgNorm':>8}"
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = f"{names[i][:8]}×{names[j][:8]}"
            header += f"  {pair:>18}"
    print(header)
    print("  " + "-" * len(header))

    for layer in top_layers:
        l = layer.item()
        row = f"  {l:>6} {avg_norms[l]:>8.4f}"
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cos = torch.nn.functional.cosine_similarity(
                    vectors[names[i]][l].unsqueeze(0),
                    vectors[names[j]][l].unsqueeze(0)
                ).item()
                row += f"  {cos:>18.4f}"
        print(row)


def recommend_layers(vectors):
    """Recommend layers for steering based on norms and agreement."""
    print("\n" + "=" * 80)
    print("RECOMMENDED LAYERS FOR STEERING")
    print("=" * 80)

    names = list(vectors.keys())
    num_layers = list(vectors.values())[0].shape[0]

    # Score each layer: high norm + high inter-vector agreement
    avg_norms = torch.stack([v.norm(dim=1) for v in vectors.values()]).mean(dim=0)
    norm_scores = avg_norms / avg_norms.max()  # normalize to [0, 1]

    if len(names) >= 2:
        # Average pairwise cosine similarity
        cos_scores = torch.zeros(num_layers)
        n_pairs = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cos = torch.nn.functional.cosine_similarity(
                    vectors[names[i]], vectors[names[j]], dim=1
                )
                cos_scores += cos
                n_pairs += 1
        cos_scores /= n_pairs

        # Combined score: norm * cosine_agreement
        combined = norm_scores * cos_scores.clamp(min=0)
    else:
        combined = norm_scores

    top = combined.argsort(descending=True)[:5]
    print(f"\n  Top 5 layers (by norm × agreement):")
    for rank, l in enumerate(top):
        l = l.item()
        print(f"    {rank+1}. Layer {l}: combined={combined[l]:.4f}, norm={avg_norms[l]:.4f}", end="")
        if len(names) >= 2:
            print(f", cos_agreement={cos_scores[l]:.4f}")
        else:
            print()

    return [l.item() for l in top]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vec_dir", type=str, required=True)
    args = parser.parse_args()

    vectors = load_vectors(args.vec_dir)
    if not vectors:
        print(f"No *_response_avg_diff.pt files found in {args.vec_dir}")
        return

    print(f"Loaded {len(vectors)} vectors from {args.vec_dir}:")
    for name, vec in vectors.items():
        print(f"  {name}: {list(vec.shape)}")

    print_norms(vectors)
    print_cosine_similarity(vectors)
    recommended = recommend_layers(vectors)

    print(f"\n  → Use these layers for the alpha sweep: {recommended[:3]}")


if __name__ == "__main__":
    main()
