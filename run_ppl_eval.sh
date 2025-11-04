#!/usr/bin/env bash
# bash run_ppl_eval.sh outputs/ddp_4_flat_bqat_bsz1_accum4/llama-3.1-8b-instruct/w4a4/exp/
set -x

CHECKPOINT_DIR="$1"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "Usage: $0 <checkpoint_dir>"
  exit 1
fi

MODEL_CONFIG=""
for dir in ./modelzoo/*/*; do
  [[ -d "$dir" ]] || continue
  name=$(basename "$dir")
  case "$CHECKPOINT_DIR" in
    */${name}/*|*/${name}|${name}/*|${name})
      MODEL_CONFIG="$dir"
      break
      ;;
  esac
done

if [[ -z "$MODEL_CONFIG" ]]; then
  echo "Failed to infer model config for checkpoint dir: $CHECKPOINT_DIR" >&2
  exit 1
fi

for dataset in wikitext2 c4; do
  python3 benchmarks/benchmark_ppl.py \
    --model-config "$MODEL_CONFIG" \
    --checkpoint "$CHECKPOINT_DIR" \
    --dataset "$dataset"
done
