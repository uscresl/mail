now=$(date +%m.%d.%H.%M)
env=ClothFold
seed=11
exp_name=ABL_${env}_SACDrQ_IR_${now}_${seed}

python experiments/run_drq.py \
--name=${exp_name} \
--env_name=${env} \
--env_kwargs_num_variations=100 \
--env_kwargs_num_picker=1 \
--action_mode=pickerpickandplace \
--action_repeat=1 \
--horizon=3 \
--log_dir=./data/${exp_name} \
--num_train_steps=510_000 \
--log_interval=4000 \
--seed=${seed} \
--rsi_file=data/Abl_ClothFold_DynModel_Particles_100eps_top_down_pickerpickandplace_2arm_11.pkl \
--enable_ir=True \
--ir_prob=0.1 \
--wandb