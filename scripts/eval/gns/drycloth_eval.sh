# NOTE: replace partial_dyn_path with your file and specify your output logdir (log_dir)

seed=11

python VCD/main_plan.py \
--num_worker=0 \
--gpu_num=1 \
--partial_dyn_path=data/gns_checkpoints/DryCloth_GNS_12.11.09.57_11/vsbl_dyn_150.pth \
--log_dir=data/gns_checkpoints/DryCloth_GNS_12.11.09.57_11_planning/ \
--pick_and_place_num=3 \
--env_name=DryCloth \
--task=hang \
--seed=${seed} \
--num_variations=100 \
--move_distance_range_min=1.0 \
--move_distance_range_max=1.0 \
--cached_states_path=ours_drycloth_plan_n100.pkl