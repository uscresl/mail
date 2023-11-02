num_eps=100
cam_view=top_down
action_mode=pickerpickandplace
num_picker=1
seed=11
env=DryCloth

python experiments/generate_expert_trajs.py \
--save_observation_img=True \
--num_variations=100 \
--image_mode=rgb \
--num_eps=${num_eps} \
--env_img_size=32 \
--env_name=${env} \
--action_mode=${action_mode} \
--action_repeat=1 \
--num_picker=${num_picker} \
--env_horizon=2 \
--cam_view=${cam_view} \
--save_particles=True \
--seed=${seed} \
--out_filename=Abl_${env}_DynModel_Particles_${num_eps}eps_${cam_view}_${action_mode}_${num_picker}arm_${seed}.pkl