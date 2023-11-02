# NOTE: replace load-file with your file

python goal_prox_il/goal_prox/main.py \
--prefix=dpf-deep \
--env-name=DryCloth \
--alg=dpf-deep \
--num-processes=1 \
--seed=11 \
--traj-load-path=goal_prox_il/expert_datasets/drycloth_two_arm_11.pt \
--load-file=data/checkpoints/527-DC-11-70-dpf-deep/model_57344.pt \
--eval-only \
--no-wb
