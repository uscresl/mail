python tests/test_cmds/bco/def.py --prefix bco --env-name HalfCheetah-v2 --eval-num-processes 32 --num-render 0 --num-env-steps 1e7 --bco-inv-lr 3e-4 --bco-alpha 50 --bco-alpha-size 200000 --bco-expl-steps 200000 --bco-inv-batch-size 128 --linear-lr-decay False --bc-state-norm --max-grad-norm -1 --normalize-env False --bc-num-epochs 10 --lr 3e-4 --bco-inv-epochs 1  --traj-load-path tests/expert_demonstrations/halfcheetah_50ep.pt --eval-interval 1 --log-smooth-len 10