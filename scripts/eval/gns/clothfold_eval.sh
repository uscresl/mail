# NOTE: replace partial_dyn_path with your file and specify your output logdir (log_dir)

seed=11

python VCD/main_plan.py \
--num_worker=2 \
--gpu_num=1 \
--partial_dyn_path=data/gns_checkpoints/Clothfold_GNS_12.07.10.59_11/vsbl_dyn_150.pth \
--log_dir=data/gns_checkpoints/Clothfold_GNS_12.07.10.59_11_planning/ \
--pick_and_place_num=3 \
--env_name=ClothFold \
--seed=${seed} \
--task=fold \
--num_variations=100 \
--cached_states_path=ours_clothfold_plan_n100.pkl