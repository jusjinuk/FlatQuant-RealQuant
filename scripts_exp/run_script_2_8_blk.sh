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


# --- NEW: pick a free master port in [29501, 29599] ---
MASTER_PORT=""
for p in $(shuf -i 29501-29599); do
  python - <<PY 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", $p))
except OSError:
    sys.exit(1)
s.close()
PY
  if [[ $? -eq 0 ]]; then MASTER_PORT="$p"; break; fi
done
if [[ -z "$MASTER_PORT" ]]; then
  echo "No free port found in 29501-29599"
  exit 1
fi
# -----------------------------------------------

touch ./outputs/

torchrun --nproc_per_node=$NUM_PROC --master_port $MASTER_PORT main.py \
  	--model ./modelzoo/llama-3.1-instruct/llama-3.1-8b-instruct \
  	--ddp_size $NUM_DDP_SIZE \
  	--offload \
  	--w_bits 4 --a_bits 4 \
  	--cali_bsz 2 --cali_bsz_accumulate_step 8 --epoch $EPOCH --flat_lr 5e-3 \
        --lwc --lac --cali_trans --add_diag $QAT_FLAG \
        --output_dir ./outputs/blk_ddp_${NUM_DDP_SIZE}_flat${QAT_FILE}_bsz2_accum8_epoch${EPOCH}_nsamples${NSAMPLES} \
        --quantized_save --blockwise_save \
        --cali_dataset redpajama \
        --nsamples $NSAMPLES

bash run_ppl_eval.sh ./outputs/blk_ddp_${NUM_DDP_SIZE}_flat${QAT_FILE}_bsz2_accum8_epoch${EPOCH}_nsamples${NSAMPLES}/llama-3.1-8b-instruct/w4a4/exp/