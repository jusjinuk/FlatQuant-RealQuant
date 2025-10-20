python3 benchmarks/benchmark_lm_eval.py \
  --model-config ./modelzoo/llama-3.1-instruct/llama-3.1-8b-instruct \
  --checkpoint ./outputs/fsdp/llama-3.1-8b-instruct/w4a4/exp \
  --lm_eval_batch_size 16

