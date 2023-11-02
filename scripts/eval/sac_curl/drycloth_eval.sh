# NOTE: replace checkpoint with your file and specify your output logdir (log_dir)

env=DryCloth

python experiments/run_curl.py \
--env_name=${env} \
--env_kwargs_num_variations=100 \
--env_kwargs_num_picker=1 \
--action_mode=pickerpickandplace \
--action_repeat=1 \
--horizon=3 \
--is_eval=True \
--name=eval_folder \
--log_dir=data/checkpoints/ABL_DryCloth_SACCURL_11.23.12.05_11 \
--checkpoint=data/checkpoints/ABL_DryCloth_SACCURL_11.23.12.05_11/actor_450000.pt \
--eval_over_five_seeds=True