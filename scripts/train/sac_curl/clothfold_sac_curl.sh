now=$(date +%m.%d.%H.%M)
env=ClothFold
seed=11
exp_name=ABL_${env}_SACCURL_${now}_${seed}

python experiments/run_curl.py \
--name=${exp_name} \
--env_name=${env} \
--env_kwargs_num_variations=100 \
--env_kwargs_num_picker=1 \
--action_mode=pickerpickandplace \
--action_repeat=1 \
--horizon=3 \
--log_dir=./data/${exp_name} \
--num_train_steps=510_000 \
--seed=${seed} \
--wandb