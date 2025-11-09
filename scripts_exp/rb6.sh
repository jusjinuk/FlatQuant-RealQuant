set -x

bash scripts_exp/run_script_2_8_time.sh 8 2 1024 f
bash scripts_exp/run_script_2_8_time.sh 8 2 1024 t
bash scripts_exp/run_script_2_8_time.sh 8 4 1024 f
bash scripts_exp/run_script_2_8_time.sh 8 4 1024 t
bash scripts_exp/run_script_2_8_time.sh 8 2 4096 f
bash scripts_exp/run_script_2_8_time.sh 8 2 4096 t

bash scripts_exp/run_script_2_8_time.sh 4 2 1024 f
bash scripts_exp/run_script_2_8_time.sh 4 2 1024 t
bash scripts_exp/run_script_2_8_time.sh 4 4 1024 f
bash scripts_exp/run_script_2_8_time.sh 4 4 1024 t
bash scripts_exp/run_script_2_8_time.sh 4 2 4096 f
bash scripts_exp/run_script_2_8_time.sh 4 2 4096 t

bash scripts_exp/run_script_2_8_time.sh 1 2 1024 f
bash scripts_exp/run_script_2_8_time.sh 1 2 1024 t
bash scripts_exp/run_script_2_8_time.sh 1 4 1024 f
bash scripts_exp/run_script_2_8_time.sh 1 4 1024 t
bash scripts_exp/run_script_2_8_time.sh 1 2 4096 f
bash scripts_exp/run_script_2_8_time.sh 1 2 4096 t
