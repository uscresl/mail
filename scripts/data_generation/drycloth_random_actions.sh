env=DryCloth
cam_view=top_down
action_mode=pickerpickandplace
num_picker=1
num_eps=10000
seed=11

python experiments/generate_random_trajs.py \
--num_variations=100 \
--num_eps=${num_eps} \
--env_name=${env} \
--action_mode=${action_mode} \
--action_repeat=1 \
--num_picker=${num_picker} \
--env_horizon=6 \
--cam_view=${cam_view} \
--save_particles=True \
--seed=${seed} \
--save_every_x_timesteps=1 \
--out_filename=${env}_Random_Actions_for_DynModel_Particles_${num_eps}eps_${cam_view}_${action_mode}_${num_picker}arm_top_down_${seed}seed.pkl