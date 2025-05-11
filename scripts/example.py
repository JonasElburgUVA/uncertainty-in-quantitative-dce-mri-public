import os, sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from scripts.simulate import simulate_dataset
import scripts.train as train
from PIL import Image
import pickle
import ruamel.yaml
import matplotlib.pyplot as plt

# Show paths and configurations
datapath='data'
seed=42
configpath='configs/example.yaml'


# load cov and mean
with open(f"{datapath}/learned_prior/traincov.pkl", "rb") as f:
    cov = pickle.load(f)
with open(f"{datapath}/learned_prior/trainmean.pkl", "rb") as f:
    mean = pickle.load(f)

# simulate data-set
print("Simulating data-set...")
simulate_dataset('normal', 100000, save=True, cov=cov, mean=mean, snr_range=[5,25])
print("Data-set simulated.")
plt.show()
plt.close()

# train model
with open(configpath, "r") as f:
      config = ruamel.yaml.load(f, ruamel.yaml.RoundTripLoader)
print("Training model...")
out_path = train.main(config, seed, vis=True)
print("Model trained.")

# show results
print("Showing results...")
vis_path = os.path.join(out_path, "visualisations")
acc_plot = os.path.join(vis_path, "test_accuracy.png")
uct_err = os.path.join(vis_path, "test_uct_err_smooth.png")

# show plots
Image.open(acc_plot).show()
Image.open(uct_err).show()
print("Results shown. Look into the different configuration files to carry out more experiments.")

