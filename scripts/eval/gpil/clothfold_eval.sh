# NOTE: replace load-file with your file

python goal_prox_il/goal_prox/main.py \
--prefix=dpf-deep \
--env-name=ClothFold \
--alg=dpf-deep \
--num-processes=1 \
--seed=11 \
--traj-load-path=goal_prox_il/expert_datasets/clothfold_two_arm_11.pt \
--load-file=data/checkpoints/527-CF-11-JM-dpf-deep/model_12288.pt \
--eval-only \
--no-wb