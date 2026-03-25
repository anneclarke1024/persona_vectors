#!/bin/bash
# =============================================================================
# Depression Steering Vector Pipeline
# =============================================================================
# Runs the full extraction pipeline for 4 depression vectors using Chen et al.'s
# code with our trait data.
#
# Usage:
#   ./run_depression_pipeline.sh              # full run (n_per_question=10)
#   ./run_depression_pipeline.sh --dry-run    # dry run (n_per_question=1, temperature=0)
#   ./run_depression_pipeline.sh --resume     # resume from where it left off
#
# Prerequisites:
#   - .env file with OPENAI_API_KEY and HF_TOKEN
#   - pip install -r requirements.txt
#   - GPU available for vLLM + HuggingFace
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT="Llama-3.1-8B-Instruct"
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"
VERSION="extract"
N_PER_QUESTION=10
MAX_CONCURRENT_JUDGES=100

# Parse args
DRY_RUN=false
RESUME=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true; N_PER_QUESTION=1 ;;
        --resume) RESUME=true ;;
    esac
done

# ── Run folder with provenance ─────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")

if [ "$DRY_RUN" = true ]; then
    RUN_DIR="runs/dryrun_${TIMESTAMP}"
else
    RUN_DIR="runs/run_${TIMESTAMP}"
fi

mkdir -p "$RUN_DIR"

# Save run config for reproducibility
cat > "$RUN_DIR/config.json" << CFGEOF
{
    "timestamp": "$TIMESTAMP",
    "git_hash": "$GIT_HASH",
    "model": "$MODEL",
    "judge_model": "$JUDGE_MODEL",
    "version": "$VERSION",
    "n_per_question": $N_PER_QUESTION,
    "max_concurrent_judges": $MAX_CONCURRENT_JUDGES,
    "dry_run": $DRY_RUN,
    "resume": $RESUME
}
CFGEOF

# Copy trait data for provenance
cp -r data_generation/trait_data_extract "$RUN_DIR/trait_data_extract_snapshot"
cp -r data_generation/trait_data_eval "$RUN_DIR/trait_data_eval_snapshot"

# Log file
LOG="$RUN_DIR/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "Depression Steering Vector Pipeline"
echo "=========================================="
echo "Run dir:    $RUN_DIR"
echo "Timestamp:  $TIMESTAMP"
echo "Git hash:   $GIT_HASH"
echo "Model:      $MODEL"
echo "Judge:      $JUDGE_MODEL"
echo "N/question: $N_PER_QUESTION"
echo "Dry run:    $DRY_RUN"
echo "Resume:     $RESUME"
echo "=========================================="

# ── Output paths ───────────────────────────────────────────────────────────────

OUT="eval_persona_extract/${MODEL_SHORT}"
mkdir -p "$OUT"

VEC_DIR="persona_vectors/${MODEL_SHORT}"
mkdir -p "$VEC_DIR"

# ── Helper: run a generation step with status tracking ─────────────────────────

run_step() {
    local STEP_NAME="$1"
    local OUTPUT_PATH="$2"
    shift 2
    local EXTRA_ARGS=("$@")

    echo ""
    echo "── Step: $STEP_NAME ──────────────────────────"
    echo "   Output: $OUTPUT_PATH"

    # Resume: skip if output already exists
    if [ "$RESUME" = true ] && [ -f "$OUTPUT_PATH" ]; then
        echo "   SKIPPED (output exists, --resume mode)"
        return 0
    fi

    local STEP_START=$(date +%s)

    python -m eval.eval_persona \
        --model "$MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --version "$VERSION" \
        --n_per_question "$N_PER_QUESTION" \
        --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
        --output_path "$OUTPUT_PATH" \
        "${EXTRA_ARGS[@]}"

    local STEP_END=$(date +%s)
    local DURATION=$((STEP_END - STEP_START))

    # Validate output
    if [ ! -f "$OUTPUT_PATH" ]; then
        echo "   FAILED: output file not created"
        exit 1
    fi

    local ROWS=$(wc -l < "$OUTPUT_PATH")
    echo "   DONE: $ROWS rows in ${DURATION}s"

    # Copy to run dir
    cp "$OUTPUT_PATH" "$RUN_DIR/$(basename "$OUTPUT_PATH")"
}

# ── Step 1: Depression enhance (pos) ──────────────────────────────────────────

run_step "depression_enhance" \
    "${OUT}/depression_pos_instruct.csv" \
    --trait depression \
    --persona_instruction_type pos \
    --assistant_name depressed

# ── Step 2: Depression suppress (neg) — pure Chen ────────────────────────────

run_step "depression_suppress" \
    "${OUT}/depression_neg_instruct.csv" \
    --trait depression \
    --persona_instruction_type neg

# ── Step 3: Eudaimonic enhance (pos) ──────────────────────────────────────────

run_step "eudaimonic_enhance" \
    "${OUT}/eudaimonic_pos_instruct.csv" \
    --trait eudaimonic \
    --persona_instruction_type pos

# ── Step 4: Simple healthy enhance (pos) ──────────────────────────────────────

run_step "simple_healthy_enhance" \
    "${OUT}/simple_healthy_pos_instruct.csv" \
    --trait simple_healthy \
    --persona_instruction_type pos \
    --assistant_name happy

# ── Step 5: No prompt (bare Llama) ────────────────────────────────────────────

run_step "no_prompt" \
    "${OUT}/depression_no_prompt.csv" \
    --trait depression

# ── Step 6: Vector extraction ─────────────────────────────────────────────────

echo ""
echo "── Vector Extraction ──────────────────────────"

extract_vector() {
    local VEC_NAME="$1"
    local POS_PATH="$2"
    local NEG_PATH="$3"
    local TRAIT_LABEL="$4"

    echo "   Extracting: $VEC_NAME"

    # Resume: skip if vector already exists
    if [ "$RESUME" = true ] && [ -f "${VEC_DIR}/${TRAIT_LABEL}_response_avg_diff.pt" ]; then
        echo "   SKIPPED (vector exists, --resume mode)"
        return 0
    fi

    local VEC_START=$(date +%s)

    python generate_vec.py \
        --model_name "$MODEL" \
        --pos_path "$POS_PATH" \
        --neg_path "$NEG_PATH" \
        --trait "$TRAIT_LABEL" \
        --save_dir "$VEC_DIR" \
        --threshold 50

    local VEC_END=$(date +%s)
    echo "   DONE in $((VEC_END - VEC_START))s"
}

# Vector 1: Depression enhance vs Depression suppress (pure Chen)
extract_vector "vec1_depression_vs_suppress" \
    "${OUT}/depression_pos_instruct.csv" \
    "${OUT}/depression_neg_instruct.csv" \
    "depression"

# Vector 2: Depression enhance vs No prompt
extract_vector "vec2_depression_vs_noprompt" \
    "${OUT}/depression_pos_instruct.csv" \
    "${OUT}/depression_no_prompt.csv" \
    "depression_vs_noprompt"

# Vector 3: Depression enhance vs Eudaimonic enhance
extract_vector "vec3_depression_vs_eudaimonic" \
    "${OUT}/depression_pos_instruct.csv" \
    "${OUT}/eudaimonic_pos_instruct.csv" \
    "depression_vs_eudaimonic"

# Vector 4: Depression enhance vs Simple Healthy enhance
extract_vector "vec4_depression_vs_simple_healthy" \
    "${OUT}/depression_pos_instruct.csv" \
    "${OUT}/simple_healthy_pos_instruct.csv" \
    "depression_vs_simple_healthy"

# ── Copy vectors to run dir ───────────────────────────────────────────────────

cp "$VEC_DIR"/depression*.pt "$RUN_DIR/" 2>/dev/null || true

# ── Validation ────────────────────────────────────────────────────────────────

echo ""
echo "── Validation ──────────────────────────────────"

python3 -c "
import torch, glob, os
vec_dir = '${VEC_DIR}'
for f in sorted(glob.glob(f'{vec_dir}/depression*_response_avg_diff.pt')):
    t = torch.load(f, weights_only=False)
    name = os.path.basename(f)
    print(f'  {name}: shape={list(t.shape)}, norm={t.norm():.2f}')
print()

import pandas as pd
out_dir = '${OUT}'
for f in sorted(glob.glob(f'{out_dir}/*.csv')):
    df = pd.read_csv(f)
    name = os.path.basename(f)
    # Find trait column (not question/prompt/answer/question_id/coherence)
    trait_cols = [c for c in df.columns if c not in ['question','prompt','answer','question_id','coherence']]
    trait_col = trait_cols[0] if trait_cols else None
    if trait_col:
        print(f'  {name}: {len(df)} rows, {trait_col}={df[trait_col].mean():.1f}±{df[trait_col].std():.1f}, coherence={df[\"coherence\"].mean():.1f}±{df[\"coherence\"].std():.1f}')
    else:
        print(f'  {name}: {len(df)} rows')
"

# ── Summary ───────────────────────────────────────────────────────────────────

TOTAL_END=$(date +%s)
echo ""
echo "=========================================="
echo "Pipeline complete"
echo "Run dir: $RUN_DIR"
echo "=========================================="
