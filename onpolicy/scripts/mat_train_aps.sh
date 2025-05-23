#!/bin/sh
env="Aps"
algo="mat"
exp="fullcoop_test/sinrdb1_powerdb1_sumcost0_localpowersum1_scoef20_pcoef1_ue4_ap20_sinrn10_fullcoop"
seed=1
n_rollout_threads=16
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
# do
#     echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python train/train_aps.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
 --seed ${seed} --n_training_threads 16 --n_rollout_threads ${n_rollout_threads} \
 --num_mini_batch 1 --episode_length 10 --num_env_steps 100000 --ppo_epoch 15 \
 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --entropy_coef 0.015 --gamma 0.001
# done
