# Uncertainty In Quantitative DCE-MRI
This repository contains the code used for "Physics-Informed Uncertainty Quantification In DCE-MRI using Mean Variance Estimation"

## Use
1. In a clean virtual environment, run 
 ``pip install -r requirements.txt
 ``

2. Run 'example.py' in the scripts directory to train your own model and test if everything is working.

 3. To simulate data yourself, run `python scripts/simulate.py` and `python scripts/simulate_ood.py`. You should see the resulting data including some visualisations in the /data directory.

 4. Run train.py using the config files in the configs/ directory, or create your own config files to run more model setups.

 5. Notice the created output path in output/ after training a model. This will contain results. For further evaluation, you can use eval.py with several arguments.

 6. After training N uncertainty-estimating models (pinn_ph, mve_ssn, or pi_mve), ensemble.py is used in combination with an ensemble config file to evaluate an ensemble.

 7. Use NLLS.py to generate nlls predictions of a test dataset.


 ## How to set up a logger
 The current supported loggers are the default pytorch-lightning logger, and weights and biases. To use weights and biases, ensure your config file contains "logger: wandb". To use an alternative logger such as tensorboard, it has to be added to "train.py" in the section at line 52.

