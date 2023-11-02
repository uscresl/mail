now=$(date +%m.%d.%H.%M)
env=DryCloth
seed=11
exp_name=ABL_${env}_SACDrQ_${now}_${seed}

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
--wandb