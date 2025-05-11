#!/bin/bash

seeds=(0) # 1 2 3 4 5 6 7 8 9)

for seed in "${seeds[@]}"; do
    python scripts/eval.py --path output/normal/baseline_dcenet_${seed}
done