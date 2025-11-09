
bsr r3 4 -j "lr_n_2_1_2_1024" -x lemon,watermelon -- bash run_script_2_4_lr.sh 1 2 1024 f
bsr r3 4 -j "lr_n_2_4_2_1024" -x lemon,watermelon -- bash run_script_2_4_lr.sh 4 2 1024 f
bsr r3 8 -j "lr_n_2_8_2_1024" -x lemon,watermelon -- bash run_script_2_8_lr.sh 8 2 1024 f

bsr r3 4 -j "lr_b_2_1_2_1024" -x lemon,watermelon -- bash run_script_2_4_lr.sh 1 2 1024 t
bsr r3 4 -j "lr_b_2_4_2_1024" -x lemon,watermelon -- bash run_script_2_4_lr.sh 4 2 1024 t
bsr r3 8 -j "lr_b_2_8_2_1024" -x lemon,watermelon -- bash run_script_2_8_lr.sh 8 2 1024 t

bsr r3 4 -j "lr_n_2_1_2_4096" -x lemon,watermelon -- bash run_script_2_4_lr.sh 1 2 4096 f
bsr r3 4 -j "lr_n_2_4_2_4096" -x lemon,watermelon -- bash run_script_2_4_lr.sh 4 2 4096 f
bsr r3 8 -j "lr_n_2_8_2_4096" -x lemon,watermelon -- bash run_script_2_8_lr.sh 8 2 4096 f

bsr r3 4 -j "lr_b_2_1_2_4096"  -x lemon,watermelon -- bash run_script_2_4_lr.sh 1 2 4096 t
bsr r3 4 -j "lr_b_2_4_2_4096"  -x lemon,watermelon -- bash run_script_2_4_lr.sh 4 2 4096 t
bsr r3 8 -j "lr_b_2_8_2_4096"  -x lemon,watermelon -- bash run_script_2_8_lr.sh 8 2 4096 t

