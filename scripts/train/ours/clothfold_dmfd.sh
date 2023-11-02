# NOTE: replace rsi_file with your file generated from running ./scripts/train/ours/clothfold_gen_student_dataset.sh

env=ClothFold
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_sb3.py \
--env_name=${env} \
--name=${env}_DMfD_OneArmStudentDataset_100eps_${now}_${seed} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=100 \
--action_mode=pickerpickandplace \
--env_kwargs_num_picker=1 \
--action_repeat=1 \
--horizon=3 \
--agent=awac \
--awac_replay_size=600_000 \
--val_freq=4000 \
--rsi_file=data/cem_2armsto1/ABL_ClothFold_OneArmStudentDataset_200eps_12.10.18.50_11/one_arm_dataset_100eps.pkl \
--batch_size=256 \
--val_num_eps=10 \
--add_sac_loss=True \
--sac_loss_weight=0.1 \
--enable_img_aug=True \
--sb3_iterations=510_000 \
--save_everything_every_tsteps=300_000 \
--seed=${seed} \
--wandb