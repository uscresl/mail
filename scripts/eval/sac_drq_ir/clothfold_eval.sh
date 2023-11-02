# NOTE: replace checkpoint with your file and specify your output logdir (log_dir)

env=ClothFold

python experiments/run_drq.py \
--env_name=${env} \
--name=eval_folder \
--log_dir=./data/checkpoints/ABL_ClothFold_SACDrQ_IR_01.07.10.23_11 \
--env_kwargs_num_variations=100 \
--env_kwargs_num_picker=1 \
--action_mode=pickerpickandplace \
--action_repeat=1 \
--horizon=3 \
--is_eval=True \
--eval_over_five_seeds=True \
--checkpoint=data/checkpoints/ABL_ClothFold_SACDrQ_IR_01.07.10.23_11/actor_232000.pt