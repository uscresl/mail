# NOTE: replace load-file with your file

python goal_prox_il/goal_prox/main.py \
--prefix=gaifo-deep \
--env-name=DryCloth \
--alg=gaifo-deep \
--num-processes=1 \
--seed=11 \
--traj-load-path=goal_prox_il/expert_datasets/drycloth_two_arm_11.pt \
--load-file=data/checkpoints/425-DC-11-Y3-gaifo-deep/model_450560.pt \
--eval-only \
--no-wb
