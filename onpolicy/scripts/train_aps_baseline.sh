#!/bin/sh
env="Aps"
algo="kstrongest"
seed=1
values="2 3 4 5 6"
for k in $values; do
    exp="comp_mob_ped/${k}strongest"
    python baseline.py --env_name ${env} --algorithm_name ${algo} \
    --n_rollout_threads 1 --seed ${seed} \
    --episode_length 2000 --num_env_steps 1000000 \
    --experiment_name ${exp} \
    --K ${k} --largest # --random
done
