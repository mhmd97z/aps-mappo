#!/bin/sh
env="Aps"
algo="kstrongest"
seed=1
values="2 3 4 5 6"
for k in $values; do
    exp="comp_step5/scen1_${k}strongest"
    python baseline.py --env_name ${env} --algorithm_name ${algo} \
    --n_rollout_threads 16 --seed ${seed} \
    --episode_length 10 --num_env_steps 300000 \
    --experiment_name ${exp} --seed ${seed} \
    --K ${k} --largest # --random
done
