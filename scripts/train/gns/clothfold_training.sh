# NOTE: replace dataf with your file

now=$(date +%m.%d.%H.%M)
seed=11
exp_name=Clothfold_GNS_${now}_${seed}

python VCD/main.py \
--env_name=ClothFold \
--gen_data=0 \
--dataf=./data/ours_clothfold_vcd_11 \
--batch_size=8 \
--seed=${seed} \
--cached_states_path=ours_clothfold_n100.pkl \
--num_variations=100 \
--log_dir=data/${exp_name}