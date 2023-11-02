now=$(date +%m.%d.%H.%M)
seed=11

python VCD/main.py \
--gen_data=1 \
--env_name=DryCloth \
--dataf=./data/drycloth_vcd_${now}_${seed} \
--cached_states_path=ours_drycloth_n100.pkl \
--num_variations=100 \
--collect_data_delta_move_min=1.0 \
--collect_data_delta_move_max=1.0 \
--seed=${seed}