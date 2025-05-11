import numpy as np
import matplotlib.pyplot as plt
import torch
from .utils_torch import etofts

from scipy.ndimage import gaussian_filter1d
# param_names = ["ke", "dt", "ve", "vp"]
pk_param_names = ["$k_e$", "$v_e$", "$v_p$"]
param_names = ["$k_e$", "$dt$", "$v_e$", "$v_p$"]
param_units = ["$min^{-1}$", "$min$", "Fraction", "Fraction"]
pk_param_units = ["$min^{-1}$", "Fraction", "Fraction"]
pk_bounds = [3,1,1]

axis_font_size = 10
title_font_size = 15
ticks_font_size = 8
dpi = 500
width_in_inches = 190 / 25.4
tab20 = plt.get_cmap('tab20')
Colors = {
    "NLLS" : tab20(0),  # Blue
    "SNN" : tab20(2),  # Orange
    "PINN" : tab20(4),  # Green
    "PINN (ens)" : tab20(5),  # Light Blue-Green
    "PINN (ens-ep)" : tab20(5),  # Light Blue-Green
    "MVE" : tab20(6),  # Red
    "MVE (ens)" : tab20(7),  # Light Red-Pink
    "MVE (ens-ep)": tab20(7),  # Light Red-Pink
    "MVE (pi)" : tab20(8),  # Purple
    "MVE (pi, ens)" : tab20(9),  # Light Purple-Lavender
}
markers = {
    "NLLS": "o",
    "SNN": "s",
    "PINN": "d",
    "MVE": "x",
    "MVE (pi)": "^",
    "MVE (ens)": "P",
    "MVE (pi, ens)": "v",
    "PINN (ens)": "D",
}
def plot_predictions_regression(
        predictions, dataset, uct=False):
    fig, ax = plt.subplots(1, 3, dpi=500)
    mms = 190
    width_in_inches = mms / 25.4
    height_in_inches = width_in_inches * (1 / 3) # rows/columns
    fig.set_size_inches(width_in_inches, height_in_inches)
    preds = predictions["pred"].cpu().detach().numpy()
    N = len(preds)
    target = dataset.params.cpu().detach().numpy()[:N]

    if uct and len(preds.shape) == 3: 
        preds = preds[:, 0]

    # Store all handles and labels for the legend
    handles, labels = [], []

    for j, axi in enumerate(ax.ravel()):
        i = j
        if j >= 1:
            i = j + 1
        
        x = target[:, i]
        y = preds[:, i]
        sc = axi.scatter(x, y, alpha=0.05, s=5, color='b',label="Data Points")
        line = axi.plot([target[:, i].min(), target[:, i].max()], 
                        [target[:, i].min(), target[:, i].max()], 
                        "k--", label="x=y")[0]

        axi.set_xlabel("True", fontsize=axis_font_size)
        axi.set_ylabel("Predicted", fontsize=axis_font_size)
        axi.set_title(f"{param_names[i]}", fontsize=title_font_size)
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)

        # remove y axis label for i > 0
        if j > 0:
            axi.set_ylabel("")

        # Collect the handles and labels from the current plot
        for handle, label in zip([sc, line], ["Data Points", "x=y"]):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

        # Create a proxy artist for the scatter plot with higher alpha for the legend
        legend_scatter = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', alpha=0.8, markersize=6, label="Data Points")

        # Update handles and labels to include the proxy artist for the scatter
        handles = [legend_scatter if lbl == "Data Points" else hdl for hdl, lbl in zip(handles, labels)]

    # set sup title
    fig.suptitle("Predictions vs. True Labels", fontsize=title_font_size)
    # Place the legend with unique handles and labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.03), fontsize=axis_font_size, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust to make space for the legend

    return fig

def plot_uncertainty_binned(predictions, dataset, n_bins=10):
    fig, ax = plt.subplots(1, 3, dpi=500)
    mms = 190
    width_in_inches = mms / 25.4
    height_in_inches = width_in_inches * (1 / 3) #+1 # rows/columns
    fig.set_size_inches(width_in_inches, height_in_inches)
    pred_mean = predictions["pred"].cpu().detach().numpy()
    pred_uct = predictions["var"].cpu().detach().numpy()
    N = len(pred_mean)
    target = dataset.params.cpu().detach().numpy()[:N]
    nl = dataset.snr.cpu().detach().numpy()[:N]
    error = np.abs(pred_mean - target)
    # mean_target = target.mean(axis=0)
    rel_error = error #/ mean_target

    # Store all handles and labels for the legend
    handles, labels = [], []

    for j, axi in enumerate(ax.ravel()):
        i = j
        if j >= 1:
            i = j + 1

        x = rel_error[:, i]
        y = pred_uct[:, i]

        bins = np.arange(0,x.max(),x.max()/10)#[0, 0.05, 0.1, 0.25, 1]
        digitized = np.digitize(x, bins)
        bin_means = [y[digitized == n].mean() for n in range(1, len(bins)+1)]
        bin_stds = [y[digitized == n].std() for n in range(1, len(bins)+1)]
        
        equal_width_bins = np.arange(len(bin_means))
        bar_width = 0.9

        bar = axi.bar(equal_width_bins, bin_means, width=bar_width, align='center', alpha=0.5, label="Mean UCT")
        # err = axi.errorbar(equal_width_bins, bin_means, yerr=bin_stds, fmt='o', label="SD", capsize=5)

        axi.set_ylabel("Predicted UCT", fontsize=axis_font_size)
        axi.set_xlabel("Error", fontsize=axis_font_size)
        axi.set_title(f"{param_names[i]}", fontsize=title_font_size)
        axi.set_xticks(equal_width_bins)
        axi.set_xticklabels([f"{np.round(bins[j],2)} - {np.round(bins[j+1],2)}" for j in range(len(bins)-1)]+[f">{np.round(bins[-1],2)}"], rotation=45, fontsize=ticks_font_size)
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)

        # remove y axis label for i > 0
        if j > 0:
            axi.set_ylabel("")

        # Collect the handles and labels from the current plot
        # for handle, label in zip([bar, err], ["Mean UCT", "SD"]):
        for handle, label in zip([bar], ["Mean UCT"]):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # Place the legend with unique handles and labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.03), fontsize=axis_font_size, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust to make space for the legend

    return fig

def plot_uncertainty_error(predictions, dataset):
    fig, ax = plt.subplots(1, 3, dpi=500)
    mms = 190
    width_in_inches = mms / 25.4
    height_in_inches = width_in_inches * (1 / 3) #+1 # rows/columns
    fig.set_size_inches(width_in_inches, height_in_inches)
    pred_mean = predictions["pred"].cpu().detach().numpy()
    pred_uct = np.sqrt(predictions["var"].cpu().detach().numpy() * 2 / np.pi)
    N = len(pred_mean)
    target = dataset.params.cpu().detach().numpy()[:N]
    error = np.abs(pred_mean - target)

    # Store all handles and labels for the legend
    handles, labels = [], []

    # plot a smoothened curve showing error on the x-axis and smoothened predicted uct on the y-axis
    for j, axi in enumerate(ax.ravel()):
        i = j
        if j >= 1:
            i = j + 1

        x = error[:, i]
        y = pred_uct[:,i]

        # sort the values of x and y
        x, y = zip(*sorted(zip(x, y)))
        x = np.array(x)
        y = np.array(y)
        y = np.clip(y, 0, x.max()*2)

        # smoothen the curve
        y = gaussian_filter1d(y, sigma=10)

    # as a background plot, plot the amount of datapoints between each two tick values, so that this is also represented in the plot.
        min_error = x.min()
        max_error = x.max()
        n_bins = 10
        bins = np.linspace(min_error, max_error, n_bins)
        digitized = np.digitize(x, bins)
        bin_counts = [len(y[digitized == n]) for n in range(1, len(bins)+1)]
        bin_counts = np.array(bin_counts)
        
        equal_width_bins = np.arange(len(bin_counts))
        bar_width = bins[1] - bins[0]
        # create a secondary axis for the histogram
        axi2 = axi.twinx()
        bar = axi2.bar(bins, bin_counts, width=bar_width, align='center', alpha=0.5, label="Data Points")
        axi2.set_yscale('log')
        axi2.set_ylabel("Data Points", fontsize=axis_font_size)
        axi2.tick_params(axis='both', which='major', labelsize=ticks_font_size)

        line = axi.plot(x, y, label="Smoothed UCT", color="g", alpha=1.0)[0]

        axi.set_ylabel("Predicted UCT", fontsize=axis_font_size)
        axi.set_xlabel("Error", fontsize=axis_font_size)
        axi.set_title(f"{param_names[i]}", fontsize=title_font_size)
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)

        # remove y axis label for i > 0
        if j > 0:
            axi.set_ylabel("")
        if j < 2:
            axi2.set_ylabel("")

        # Collect the handles and labels from the current plot
        for handle, label in zip([line, bar], ["Uncertainty", "Number of Data Points"]):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # set suptitle
    fig.suptitle("Uncertainty vs. Error", fontsize=title_font_size)

    # Place the legend with unique handles and labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.03), fontsize=axis_font_size, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust to make space for the legend

    return fig


def plot_uncertainty_noise(predictions, dataset, n_bins=10):
    fig, ax = plt.subplots(1, 3, dpi=500)
    mms = 190
    width_in_inches = mms / 25.4
    height_in_inches = width_in_inches * (1 / 3) #+1 # rows/columns
    fig.set_size_inches(width_in_inches, height_in_inches)
    pred_mean = predictions["pred"].cpu().detach().numpy()
    pred_uct = predictions["var"].cpu().detach().numpy()
    N = len(pred_mean)
    target = dataset.params.cpu().detach().numpy()[:N]
    nl = dataset.snr.cpu().detach().numpy()[:N]

    # Store all handles and labels for the legend
    handles, labels = [], []

    for j, axi in enumerate(ax.ravel()):
        i = j
        if j >= 1:
            i = j + 1

        x = nl.copy()
        # round all values in x
        x = np.round(x, 2)
        y = pred_uct[:, i]

        # set bins to the unique values of x
        bins = np.unique(x)
        digitized = np.digitize(x, bins)
        bin_means = [y[digitized == n].mean() for n in range(1,len(bins)+1)]
        bin_stds = [y[digitized == n].std() for n in range(1,len(bins)+1)]
        
        equal_width_bins = np.arange(len(bin_means))
        bar_width = 0.9

        bar = axi.bar(equal_width_bins, bin_means, width=bar_width, align='center', alpha=0.5, label="Mean UCT")
        # err = axi.errorbar(equal_width_bins, bin_means, yerr=bin_stds, fmt='o', label="SD", capsize=3)

        axi.set_ylabel("Predicted UCT", fontsize=axis_font_size)
        axi.set_xlabel("Noise Level", fontsize=axis_font_size)
        axi.set_title(f"{param_names[i]}", fontsize=title_font_size)
        axi.set_xticks(equal_width_bins)
        axi.set_xticklabels([f"{bins[j]:.2f}" for j in range(len(bins))], rotation=45, fontsize=ticks_font_size)
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)

        # remove y axis label for i > 0
        if j > 0:
            axi.set_ylabel("")

        # Collect the handles and labels from the current plot
        for handle, label in zip([bar], ["Mean UCT"]):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # Place the legend with unique handles and labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.03), fontsize=axis_font_size, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust to make space for the legend

    return fig

def plot_paramdist_uct(predictions, dataset, datamodule):
    fig, ax = plt.subplots(1, 3, dpi=500)
    mms = 190
    width_in_inches = mms / 25.4
    height_in_inches = width_in_inches * (1 / 3) # rows/columns
    fig.set_size_inches(width_in_inches, height_in_inches)
    pred_mean = predictions["pred"].cpu().detach().numpy()
    pred_uct = predictions["var"].cpu().detach().numpy()
    N = len(pred_mean)
    target = dataset.params.cpu().detach().numpy()[:N]

    # Store all handles and labels for the legend
    handles, labels = [], []

    for j, axi in enumerate(ax.ravel()):
        i = j
        if j >= 1:
            i = j + 1

        x = target[:, i]
        hist1 = datamodule.train_dataset.params[:, i].cpu().detach().numpy()
        hist2 = datamodule.test_dataset.params[:, i].cpu().detach().numpy()
        hist_train = axi.hist(hist1, bins=100, alpha=0.4, label="Train", density=False)[2]
        hist_test = axi.hist(hist2, bins=100, alpha=0.4, label="Test", density=False)[2]

        error = np.abs(pred_mean - target)[:, i]
        uct = pred_uct[:, i]
        uct = np.sqrt(uct) * np.sqrt(2/np.pi) # Should be same as E[abs(error)]
        uct = np.clip(uct, 0, error.max()*2)
        x_unique = np.unique(x)
        y = [uct[x == xi].mean() for xi in x_unique]
        y2 = [error[x == xi].mean() for xi in x_unique]
        split_uct = False
        if "var_al" in predictions.keys():
            split_uct = True
            uct_al = np.sqrt(predictions["var_al"].cpu().detach().numpy()[:, i]) * np.sqrt(2/np.pi)
            var_ep = predictions["var_ep"].cpu().detach().numpy()
            var_ep = np.clip(var_ep, min=0)
            uct_ep = np.sqrt(var_ep[:, i]) * np.sqrt(2/np.pi)
            uct_al = [uct_al[x == xi].mean() for xi in x_unique]
            uct_ep = [uct_ep[x == xi].mean() for xi in x_unique]
        # x_unique, y, y2 = zip(*sorted(zip(x_unique, y, y2)))
            x_unique, y, y2, uct_al, uct_ep = zip(*sorted(zip(x_unique, y, y2, uct_al, uct_ep)))
        else:
            x_unique, y, y2 = zip(*sorted(zip(x_unique, y, y2)))
        x_unique = np.array(x_unique)
        y = np.array(y)
        y2 = np.array(y2)
        y = gaussian_filter1d(y, sigma=30)
        y2 = gaussian_filter1d(y2, sigma=30)
        if split_uct:
            uct_al = np.array(uct_al)
            uct_ep = np.array(uct_ep)
            uct_al = gaussian_filter1d(uct_al, sigma=30)
            uct_ep = gaussian_filter1d(uct_ep, sigma=30)
        axi2 = axi.twinx()
        line_uct = axi2.plot(x_unique, y, label="UCT", color="g", alpha=0.5)[0]
        line_error = axi2.plot(x_unique, y2, label="Error", color="r", alpha=0.5)[0]
        if split_uct:
            line_uct_al = axi2.plot(x_unique, uct_al, label="UCT AL", color="b", alpha=0.5)[0]
            line_uct_ep = axi2.plot(x_unique, uct_ep, label="UCT EP", color="m", alpha=0.5)[0]

        axi.set_xlabel("Parameter Value", fontsize=axis_font_size)
        axi.set_ylabel("N", fontsize=axis_font_size)
        axi2.set_ylabel("UCT / Error", fontsize=axis_font_size)
        axi.set_title(f"{param_names[i]}", fontsize=title_font_size)
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)
        axi2.tick_params(axis='both', which='major', labelsize=ticks_font_size)

        # remove y axis label for i > 0
        if j > 0:
            axi.set_ylabel("")
        if j != 2:
            axi2.set_ylabel("")

        # Collect the handles and labels from the current plot
        handles = [hist_train, hist_test, line_uct, line_error]
        labels = ["Train", "Test", "UCT", "Error"]
        if split_uct:
            handles.append(line_uct_al)
            handles.append(line_uct_ep)
            labels.append("UCT AL")
            labels.append("UCT EP")
        for handle, label in zip(handles, labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # Place the legend with unique handles and labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.03), fontsize=axis_font_size, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust to make space for the legend

    return fig

def plot_error_noise(predictions, dataset):
    fig, ax = plt.subplots(1,3,dpi=500)
    mms=190
    width_in_inches = mms / 25.4
    height_in_inches = width_in_inches * (1 / 3) # rows/columns
    fig.set_size_inches(width_in_inches, height_in_inches)
    pred_mean = predictions["pred"].cpu().detach().numpy()
    N = len(pred_mean)
    target = dataset.params.cpu().detach().numpy()[:N]
    nl = dataset.snr.cpu().detach().numpy()[:N]
    error = np.abs(pred_mean - target)

    # Store all handles and labels for the legend
    handles, labels = [], []

    for j, axi in enumerate(ax.ravel()):
        i = j
        if j >= 1:
            i = j + 1

        x = nl.copy()
        y = error[:, j]

        # set bins to the unique values of x
        bins = np.unique(x)
        digitized = np.digitize(x, bins)
        bin_means = [y[digitized == n].mean() for n in range(1,len(bins)+1)]
        # bin_stds = [y[digitized == n].std() for n in range(1,len(bins)+1)]

        equal_width_bins = np.arange(len(bin_means))
        bar_width = 0.9
        
        bar = axi.bar(equal_width_bins, bin_means, width=bar_width, align='center', alpha=0.5, label="Mean Error")
        # err = axi.errorbar(equal_width_bins, bin_means, yerr=bin_stds, fmt='o', label="SD", capsize=3)

        axi.set_ylabel("Error", fontsize=axis_font_size)
        axi.set_xlabel("Noise Level", fontsize=axis_font_size)
        axi.set_title(f"{param_names[i]}", fontsize=title_font_size)
        axi.set_xticks(equal_width_bins)
        axi.set_xticklabels([f"{bins[j]:.2f}" for j in range(len(bins))], rotation=45, fontsize=ticks_font_size)
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)

        # remove y axis label for i > 0
        if j > 0:
            axi.set_ylabel("")

        # Collect the handles and labels from the current plot
        for handle, label in zip([bar], ["Mean Error"]):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # Place the legend with unique handles and labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.03), fontsize=axis_font_size, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust to make space for the legend

    return fig

def plot_curves_vp(predictions, dataset):
    pred_mean = predictions["pred"].cpu().detach().numpy()
    pred_uct = predictions["var"].cpu().detach().numpy()
    N = len(pred_mean)
    vp_var = pred_uct[:,-1]
    target = dataset.params.cpu().detach().numpy()[:N]
    gt_curves = etofts(dataset.params[:N]).cpu().detach().numpy()
    curves = dataset.ct.cpu().detach().numpy()[:N]

    fig, ax = plt.subplots(8, 3, dpi=500)
    mms = 190
    width_in_inches = mms / 25.4
    height_in_inches = width_in_inches * 8 / 3 # rows/columns
    fig.set_size_inches(width_in_inches, height_in_inches)

    vp_error = np.abs(pred_mean - target)[:,-1]
    reconstructions = predictions["reconstruction"].cpu().detach().numpy()
    # sort reconstructions by vp_error
    sorted_indices = np.argsort(vp_error)[::-1]
    reconstructions_by_vp_error = reconstructions[sorted_indices]
    vp_error_by_vp_error = vp_error[sorted_indices]
    vp_var_by_vp_error = vp_var[sorted_indices]
    curves_by_vp_error = curves[sorted_indices]
    gt_curves_by_vp_error = gt_curves[sorted_indices]

    # sort reconstructions by vp_var
    sorted_indices = np.argsort(vp_var)[::-1]
    reconstructions_by_vp_var = reconstructions[sorted_indices]
    vp_error_by_vp_var = vp_error[sorted_indices]
    vp_var_by_vp_var = vp_var[sorted_indices]
    curves_by_vp_var = curves[sorted_indices]
    gt_curves_by_vp_var = gt_curves[sorted_indices]

    for j, axi in enumerate(ax.ravel()[:12]):
        axi.plot(curves_by_vp_error[j], label="Input", color='b', alpha=1.0, linewidth=0.5)
        axi.plot(reconstructions_by_vp_error[j], label="Reconstruction", c='lime', alpha=0.8, linestyle='--', linewidth=0.75)
        axi.plot(gt_curves_by_vp_error[j], label="Ground Truth", c='r', alpha=0.8, linestyle='--', linewidth=0.75)

        axi.set_title(f"VP Error: {100*vp_error_by_vp_error[j]:.2f}, VP UCT: {100*vp_var_by_vp_error[j]:.2f} (x100)", fontsize=8)
        # set tick font size
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)
        # remove xticks
        axi.set_xticks([])
       
    for j, axi in enumerate(ax.ravel()[12:]):
        axi.plot(curves_by_vp_var[j], label="Input", color='b', alpha=1.0, linewidth=0.5)
        axi.plot(reconstructions_by_vp_var[j], label="Reconstruction", c='lime', alpha=0.8, linestyle='--', linewidth=0.75)
        axi.plot(gt_curves_by_vp_var[j], label="Ground Truth", c='r', alpha=0.8, linestyle='--', linewidth=0.75)

        axi.set_title(f"VP Error: {100*vp_error_by_vp_var[j]:.2f}, VP UCT: {100*vp_var_by_vp_var[j]:.2f} (x100)", fontsize=8)
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)
        axi.set_xticks([])
        if j == 10:
            axi.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fontsize=7, ncol=3)


    # fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust to make space for the legend

    return fig



def plot_ood_class(predictions, datamodule):
    ood_limit = datamodule.train_dataset.params.max(dim=0).values.cpu().detach().numpy()
    N = len(predictions["pred"])
    pred = predictions["pred"].cpu().detach().numpy()
    target = datamodule.test_dataset.params.cpu().detach().numpy()[:N]
    error = np.abs(pred - target)
    uct = np.sqrt(predictions["var"].cpu().detach().numpy() * 2 / np.pi)
    var_ep = predictions["var_ep"].cpu().detach().numpy()
    var_ep = np.clip(var_ep, min=0)
    uct_ep = np.sqrt(predictions["var_ep"].cpu().detach().numpy() * 2 / np.pi)

    uct = np.clip(uct, 0, error.max()*2)

    fig, ax = plt.subplots(1, 3, dpi=500)
    mms = 190
    width_in_inches = mms / 25.4
    height_in_inches = width_in_inches * (1 / 3) # rows/columns
    fig.set_size_inches(width_in_inches, height_in_inches)

    for j, axi in enumerate(ax.ravel()):
        # plot uct at ood and at in-distribution data
        mask = target[:, j] > ood_limit[j]
        x_ind = uct[:, j][~mask]
        x_ood = uct[:, j][mask]
        x_ind_ep = uct_ep[:, j][~mask]
        x_ood_ep = uct_ep[:, j][mask]
        
        #grouped box plot of [x_ind, x_ind_ep] compared to [x_ood, x_ood_ep]
        # axi.boxplot([x_ind, x_ood], positions=[0, 1], widths=0.6, patch_artist=True, showfliers=False)
        axi.boxplot([x_ind_ep, x_ood_ep], positions=[0, 1], widths=0.6, patch_artist=True, showfliers=False, labels=["In-Distribution", "OOD"])

        # t-test
        from scipy.stats import ttest_ind
        t, p = ttest_ind(x_ind_ep, x_ood_ep)
        # set title
        axi.set_title(f"{param_names[j]}: p={p:.4f}")
        # set subtitle
        axi.set_ylabel("Uncertainty")

    plt.tight_layout()

    return fig
