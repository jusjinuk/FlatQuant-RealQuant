for dataset in wikitext2 c4; do
  python3 benchmarks/benchmark_ppl.py \
    --model-config ./modelzoo/llama-3.1-instruct/llama-3.1-8b-instruct \
    --checkpoint ./outputs/fsdp/llama-3.1-8b-instruct/w4a4/exp \
    --dataset $dataset
done
