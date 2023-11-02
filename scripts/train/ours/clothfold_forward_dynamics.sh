now=$(date +%m.%d.%H.%M)
env_name=ClothFold
exp_name=FWDModel
seed=11

python experiments/run_inv.py \
--env_name=${env_name} \
--env_kwargs_observation_mode=key_point \
--env_kwargs_num_variations=100 \
--action_mode=pickerpickandplace \
--env_kwargs_num_picker=1 \
--action_repeat=1 \
--horizon=1 \
--num_actions=1 \
--name=ABL_${env_name}_${exp_name}_${now}_${seed} \
--random_actions_data=data/ClothFold_Random_Actions_for_DynModel_Particles_10000eps_top_down_pickerpickandplace_1arm_top_down_11seed.pkl \
--train_mode=fwd \
--enable_particle_based_fwd_dyn=True \
--batch_size=128 \
--learning_rate=1e-5 \
--epoch=100_000 \
--seed=${seed} \
--wandb