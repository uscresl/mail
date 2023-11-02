seed=11
dataset=goal_prox_il/expert_datasets/clothfold_two_arm_${seed}.pt

python goal_prox_il/goal_prox/main.py \
--use-proper-time-limits \
--linear-lr-decay True \
--lr 3e-4 \
--num-env-steps 510000 \
--num-eval 1 \
--eval-num-processes 1 \
--num-processes=1 \
--vid-fps 30 \
--num-render 0 \
--eval-interval 32 \
--save-interval 32 \
--pf-state-norm False \
--num-mini-batch 32 \
--num-epochs 10 \
--pf-reward-norm True \
--alg dpf-deep \
--prefix dpf-deep \
--env-name ClothFold \
--exp-succ-scale 1 \
--il-in-action-norm \
--il-out-action-norm \
--exp-sample-size 4096 \
--exp-buff-size 4096 \
--pf-uncert-scale 0.01 \
--entropy-coef 0.001 \
--pf-reward-scale 1.0 \
--cuda False \
--dmode exp \
--pf-delta 0.95 \
--traj-load-path ${dataset} \
--save-dir ./data/gpil/ \
--seed ${seed}