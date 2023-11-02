# NOTE: replace pretrained_fwd_dyn_ckpt and two_arms_expert_data with your files
# NOTE: use the best combination of cem_plan_horizon, max_iters, timestep_per_decision from the CEM hyper-parameters search

now=$(date +%m.%d.%H.%M)
teacher_data_num_eps=300
env_name=DryCloth
pretrained_fwd_dyn_ckpt=data/followup/ABL_DryCloth_FWDModel_11.22.07.58_11/checkpoints/epoch_1417.pth
two_arms_expert_data=data/Abl_DryCloth_DynModel_Particles_100eps_top_down_pickerpickandplace_2arm_11.pkl
seed=11
exp_name=ABL_${env_name}_OneArmStudentDataset_${teacher_data_num_eps}eps_${now}_${seed}

python experiments/run_cem_2armsto1.py \
--name=${exp_name} \
--env_name=${env_name} \
--env_kwargs_horizon=3 \
--two_arms_expert_data=${two_arms_expert_data} \
--teacher_data_num_eps=${teacher_data_num_eps} \
--pretrained_fwd_dyn_ckpt=${pretrained_fwd_dyn_ckpt} \
--enable_trained_fwd_dyn=True \
--fwd_dyn_mode=particles \
--particle_based_cnn_lstm_fwd_dyn_mode=1dconv \
--seed=${seed} \
--cem_plan_horizon=2 \
--max_iters=1 \
--timestep_per_decision=32000 \
--wandb