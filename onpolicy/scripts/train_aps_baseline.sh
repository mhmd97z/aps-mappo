#!/bin/sh
env="Aps"
algo="kstrongest"
seed=1
values="2" # 3 4"
for k in $values; do
    exp="updated_power_apconst05/${k}strongest" # served_ue_test_20ap_20ue
    python baseline.py --env_name ${env} --algorithm_name ${algo} \
    --n_rollout_threads 16 --seed ${seed} \
    --episode_length 100 --num_env_steps 160000 \
    --experiment_name ${exp} --K ${k} --largest \
    --log_interval 1
done
