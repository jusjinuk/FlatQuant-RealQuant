set -x

NUM_DDP_SIZE=$1
EPOCH=$2
NSAMPLES=$3
QAT_FLAG=$4

if [[ -z $NUM_DDP_SIZE || -z $EPOCH || -z $NSAMPLES || -z $QAT_FLAG ]]; then
    echo "Usage: $0 <NUM_DDP_SIZE> <EPOCH> <NSAMPLES> <QAT_FLAG>"
    exit 1
fi

if [[ $QAT_FLAG == "t" ]]; then
    QAT_FLAG="--learn_weight --learn_scale"
    QAT_FILE="_bqat"
elif [[ $QAT_FLAG == "f" ]]; then
    QAT_FLAG=""
    QAT_FILE="_noqat"
else
    echo "Invalid QAT_FLAG: $QAT_FLAG"
    exit 1
fi

NUM_PROC=$NUM_DDP_SIZE

touch ./outputs/

torchrun --nproc_per_node=$NUM_PROC --master_port 29506 main.py \
  	--model ./modelzoo/llama-3.1-instruct/llama-3.1-8b-instruct \
  	--ddp_size $NUM_DDP_SIZE \
  	--offload \
  	--w_bits 4 --a_bits 4 \
  	--k_bits 4 --k_asym --k_groupsize 128 \
  	--v_bits 4 --v_asym --v_groupsize 128 \
  	--cali_bsz 2 --cali_bsz_accumulate_step 4 --epoch $EPOCH --flat_lr 5e-3 \
        --lwc --lac --cali_trans --add_diag $QAT_FLAG \
        --output_dir ./outputs/ddp_${NUM_DDP_SIZE}_flat${QAT_FILE}_bsz2_accum4_epoch${EPOCH}_nsamples${NSAMPLES} \
        --quantized_save \
        --cali_dataset redpajama \
        --nsamples $NSAMPLES
