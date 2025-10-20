set -x

NUM_DDP_SIZE=$1
NUM_FSDP_SIZE=$2

if [[ -z $NUM_DDP_SIZE || -z $NUM_FSDP_SIZE ]]; then
    echo "Usage: $0 <NUM_DDP_SIZE> <NUM_FSDP_SIZE>"
    exit 1
fi

NUM_PROC=$((NUM_DDP_SIZE * NUM_FSDP_SIZE))

torchrun --nproc_per_node=$NUM_PROC main.py \
  	--model ./modelzoo/llama-3.1-instruct/llama-3.1-8b-instruct \
  	--ddp_size $NUM_DDP_SIZE \
  	--fsdp_size $NUM_FSDP_SIZE \
  	--offload \
    --learn_weight --learn_scale \
  	--w_bits 4 --a_bits 4 \
  	--k_bits 4 --k_asym --k_groupsize 128 \
  	--v_bits 4 --v_asym --v_groupsize 128 \
  	--cali_bsz 1 --cali_bsz_accumulate_step 4 --epoch 1 --flat_lr 5e-3 \
        --lwc --lac --cali_trans --add_diag \
        --output_dir ./outputs/fsdp --save_matrix \
        --quantized_save \
        --cali_dataset redpajama \
        --nsamples 128
