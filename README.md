# Uncertainty in Quantitative DCE-MRI

<p>
  <a href="https://www.sciencedirect.com/science/article/pii/S136184152500427X"><img src="https://img.shields.io/badge/Paper-Medical%20Image%20Analysis-blue"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-3776AB.svg"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg"></a>
  <a href="https://lightning.ai/docs/pytorch/stable/"><img src="https://img.shields.io/badge/PyTorch%20Lightning-2.4-792EE5.svg"></a>
</p>

This repository contains the code used for [**Uncertainty estimates in pharmacokinetic modelling of DCE-MRI**](https://www.sciencedirect.com/science/article/pii/S136184152500427X).

<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S136184152500427X-gr7_lrg.jpg" alt="Figure from the paper" width="900"/>
</p>

---


## Use

1. In a clean virtual environment, run:
   ```bash
   pip install -r requirements.txt
   ```

2. Run `example.py` in the `scripts` directory to train your own model and test if everything is working:
   ```bash
   python scripts/example.py
   ```

3. To simulate data yourself, run:
   ```bash
   python scripts/simulate.py
   python scripts/simulate_ood.py
   ```
   You should see the resulting data including some visualisations in the `data/` directory.

4. Run `train.py` using the config files in `configs/`, or create your own config files to run more model setups.

5. Notice the created output path in `output/` after training a model. This will contain results. For further evaluation, use `eval.py` with several arguments, for example:
   ```bash
   python scripts/eval.py --path output/normal/mve_pinn_fc_0 --eval True --save_preds True
   ```

6. After training N uncertainty-estimating models (`pinn_ph`, `mve_ssn`, or `pi_mve`), `ensemble.py` is used in combination with an ensemble config file to evaluate an ensemble, for example:
   ```bash
   python scripts/ensemble.py --config configs/normal/ensemble_snn.yaml --eval True --save_preds True
   ```

7. Use `nlls.py` (`scripts/nlls.py`) to generate NLLS predictions of a test dataset:
   ```bash
   python scripts/nlls.py --mode sim --simmode normal --eval True --vis True
   ```

## How to set up a logger

The current supported loggers are the default PyTorch Lightning logger and Weights & Biases. To use Weights & Biases, ensure your config file contains `logger: wandb`. To use an alternative logger such as TensorBoard, it has to be added to `scripts/train.py`.

---

## 📝 Citation

If you use this codebase, please cite the paper:

```bibtex
@article{dce_mri_uq_2025,
  title   = {Physics-Informed Uncertainty Quantification in DCE-MRI using Mean Variance Estimation},
  journal = {Medical Image Analysis},
  year    = {2025},
  url     = {https://www.sciencedirect.com/science/article/pii/S136184152500427X}
}
```
