# NOTE: replace checkpoint with your file

env=ClothFold

python experiments/run_sb3.py \
--is_eval=True \
--checkpoint=data/checkpoints/ClothFold_DMfD_OneArmStudentDataset_100eps_12.11.07.57_11/model304000.pt \
--eval_videos=False \
--eval_over_five_seeds=True \
--env_name=${env} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=100 \
--action_mode=pickerpickandplace \
--env_kwargs_num_picker=1 \
--action_repeat=1 \
--horizon=3 \
--agent=awac