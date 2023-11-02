# NOTE: replace cem_traj and two_arms_expert_data with your files

env_name=DryCloth
two_arms_expert_data=data/Abl_DryCloth_DynModel_Particles_100eps_top_down_pickerpickandplace_2arm_11.pkl
seed=11

python experiments/run_cem_2armsto1.py \
--env_name=${env_name} \
--two_arms_expert_data=${two_arms_expert_data} \
--env_kwargs_horizon=3 \
--is_eval=True \
--cem_traj=data/cem_2armsto1/ABL_DryCloth_OneArmStudentDataset_150eps_11.24.10.40_11/cem_traj.pkl \
--indices_for_playback=True \
--cam_view=top_down \
--seed=${seed} \
--save_dataset=True \
--only_save_num_eps=100