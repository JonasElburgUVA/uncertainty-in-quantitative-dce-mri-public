# paths=(
#     output/normal/mve_pinn_full_dcenet_1
#     output/normal/mve_snn_full_dcenet_0
#     output/normal/baseline_dcenet_0
#     output/normal/snn_dcenet_0
# )

# for path in "${paths[@]}"; do
#     python scripts/eval.py --path $path
# done

# python scripts/train.py --config mcd

ensemble_configs=(
    configs/ensemble_snn.yaml
    configs/ensemble_pinn.yaml
)

for config in "${ensemble_configs[@]}"; do
    python scripts/ensemble.py --config $config
done

