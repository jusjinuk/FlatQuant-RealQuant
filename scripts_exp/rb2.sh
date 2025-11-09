bsr r3 1 -j "n_2_4_4_1024" -- bash run_ppl_eval.sh outputs/ddp_4_flat_noqat_bsz2_accum4_epoch4_nsamples1024/llama-3.1-8b-instruct/w4a4/exp/
bsr r3 1 -j "b_2_4_4_1024" -- bash run_ppl_eval.sh outputs/ddp_4_flat_bqat_bsz2_accum4_epoch4_nsamples1024/llama-3.1-8b-instruct/w4a4/exp/

bsr r3 1 -j "n_2_4_4_4096" -- bash run_ppl_eval.sh outputs/ddp_4_flat_noqat_bsz2_accum4_epoch4_nsamples4096/llama-3.1-8b-instruct/w4a4/exp/
bsr r3 1 -j "b_2_4_4_4096" -- bash run_ppl_eval.sh outputs/ddp_4_flat_bqat_bsz2_accum4_epoch4_nsamples4096/llama-3.1-8b-instruct/w4a4/exp/

