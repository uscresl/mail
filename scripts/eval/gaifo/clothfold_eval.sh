# NOTE: replace load-file with your file

python goal_prox_il/goal_prox/main.py \
--prefix=gaifo-deep \
--env-name=ClothFold \
--alg=gaifo-deep \
--num-processes=1 \
--seed=11 \
--traj-load-path=goal_prox_il/expert_datasets/clothfold_two_arm_11.pt \
--load-file=data/checkpoints/510-CF-11-72-gaifo-deep/model_8192.pt \
--eval-only \
--no-wb