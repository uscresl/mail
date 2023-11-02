# NOTE: replace pretrained_fwd_dyn_ckpt and two_arms_data with your files

now=$(date +%m.%d.%H.%M.%S)
run_name=ABL_DryCloth
pretrained_fwd_dyn_ckpt=data/followup/ABL_DryCloth_FWDModel_11.22.07.58_11/checkpoints/epoch_1417.pth
two_arms_data=data/Abl_DryCloth_DynModel_Particles_100eps_top_down_pickerpickandplace_2arm_11.pkl

python cem_2armsto1/find_best_cem_params.py \
--pretrained_fwd_dyn_ckpt=${pretrained_fwd_dyn_ckpt} \
--particle_based_cnn_lstm_fwd_dyn_mode=1dconv \
--fwd_dyn_mode=particles \
--env_name=DryCloth \
--num_eps=1 \
--env_kwargs_horizon=3 \
--two_arms_expert_data=${two_arms_data} > data/cem_2armsto1/DEBUG/${run_name}-${now}.txt