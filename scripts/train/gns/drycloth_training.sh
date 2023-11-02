# NOTE: replace dataf with your file

now=$(date +%m.%d.%H.%M)
seed=11
exp_name=DryCloth_GNS_${now}_${seed}

python VCD/main.py \
--env_name=DryCloth \
--gen_data=0 \
--dataf=./data/drycloth_vcd_12.06.19.09_11 \
--batch_size=8 \
--seed=${seed} \
--num_variations=100 \
--cached_states_path=ours_drycloth_n100.pkl \
--collect_data_delta_move_min=1.0 \
--collect_data_delta_move_max=1.0 \
--log_dir=data/${exp_name}