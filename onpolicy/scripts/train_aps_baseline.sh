#!/bin/sh
env="Aps"
algo="kstrongest"
seed=1
values="2 5"
for k in $values; do
    exp="ap_off_fixed/${k}strongest"
    python baseline.py --env_name ${env} --algorithm_name ${algo} \
    --n_rollout_threads 16 --seed ${seed} \
    --episode_length 100 --num_env_steps 160000 \
    --experiment_name ${exp} --K ${k} --largest \
    --log_interval 1
done
