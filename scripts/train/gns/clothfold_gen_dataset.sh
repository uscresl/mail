now=$(date +%m.%d.%H.%M)
seed=11

python VCD/main.py \
--gen_data=1 \
--env_name=ClothFold \
--dataf=./data/clothfold_vcd_${now}_${seed} \
--cached_states_path=ours_clothfold_n100.pkl \
--num_variations=100 \
--seed=${seed}
