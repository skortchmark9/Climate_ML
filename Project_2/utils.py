import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import matplotlib.pyplot as plt
import copy as copy
import matplotlib as mpl

# import netCDF4 as ncd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from torch import nn, optim
import matplotlib.cm as cm
import copy as copy
import multiprocessing as mp
from scipy import stats
import time as time
import matplotlib.font_manager
from tqdm import tqdm  # Import tqdm for the progress bar
from datetime import datetime
import warnings


def torch_backend():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def load_cdf_files(path):
    """Load all .cdf files in the given directory into an xarray dataset dictionary."""
    cdf_files = [f for f in os.listdir(path) if f.endswith(".cdf")]

    dss = {}
    for file in cdf_files:
        file_path = os.path.join(path, file)
        ds = xr.open_dataset(file_path)
        dss[file.split("_papa")[0]] = ds

    return dss


def filter_valid_timesteps(dss, dataset_name, quality_var, data_var):
    """Filter valid timesteps based on quality control values."""
    valid_timesteps = (
        dss[dataset_name][quality_var]
        .isin(np.float32([1, 2, 3]))
        .all(dim=("depth", "lat", "lon"))
    )
    filtered_data = dss[dataset_name].sel(time=valid_timesteps)
    return filtered_data[data_var][:]


def plot_invalid_values(dss, profile_key, var_name, valid_values):
    """
    Plot a bar chart showing the count of invalid values for a given variable
    at each depth level in the dataset.

    Parameters:
        dss (dict): Dictionary of xarray datasets.
        profile_key (str): Key in the dataset dictionary to access the desired dataset.
        var_name (str): Variable name to check for invalid values.
        valid_values (list): List of valid values for the variable.
    """
    # Create a boolean mask where the variable is NOT in valid_values
    invalid_mask = ~dss[profile_key][var_name].isin(valid_values)

    # Count occurrences along time, lat, and lon dimensions for each depth
    invalid_counts = invalid_mask.sum(dim=("time", "lat", "lon"))

    # Extract unique depth values
    depth_values = dss[
        profile_key
    ].depth.values  # This ensures all depth levels are used

    # Plot the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(depth_values, invalid_counts.values, width=3)  # Adjust width for clarity
    plt.xlabel("Depth")
    plt.ylabel("Count of Invalid Values")
    plt.title(
        f"Number of Values Where {var_name} is Not in {valid_values} at Each Depth"
    )
    plt.xticks(depth_values, rotation=45)  # Ensure all depth values are on x-axis
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()


def create_density_profile_filled(profiles, dss):
    """
    Creates a filled density profile DataFrame from the given profiles and dataset.

    This function takes a 2D array of profiles and a dataset containing time information,
    constructs a DataFrame with named columns for each profile, appends the time data,
    and removes any rows with missing values.

    Args:
        profiles (array-like): A 2D array or list of density profiles, where each row
            represents a profile and each column represents a density measurement.
        dss (xarray.Dataset): An xarray Dataset containing the "density_profile" variable
            with a "time" coordinate.

    Returns:
        pandas.DataFrame: A DataFrame containing the filled density profiles with
            columns named "DP_1", "DP_2", ..., "DP_21" and a "time" column. Rows with
            missing values are dropped.
    """
    filled_density_profile = pd.DataFrame(
        profiles, columns=[f"DP_{i + 1}" for i in range(21)]
    )
    filled_density_profile["time"] = dss["density_profile"].STH_71.time
    filled_density_profile = filled_density_profile.dropna()
    return filled_density_profile


def merge_and_prepare_datasets(
    datasets, output_datasets, merge_on, selected_columns, additional_columns=None
):
    """
    Merge multiple datasets on specified columns and prepare a final dataframe.

    Parameters:
        datasets (list of tuples): List of tuples where each tuple contains a dataset (xarray.Dataset or pd.DataFrame)
                                   and a list of columns to include from that dataset.
                                   Example: [(ws, ['time', 'WS_401']), (temp, ['time', 'T_25'])]
        merge_on (list): List of column names to merge the datasets on (e.g., ['time']).
        selected_columns (list): List of columns to keep in the final dataframe after merging.
        additional_columns (dict, optional): Dictionary of additional columns to compute, where the key is the column
                                             name and the value is a function to compute the column.
                                             Example: {'doy': lambda df: df['time'].dt.dayofyear}

    Returns:
        pd.DataFrame: A merged and prepared dataframe.
    """
    # Convert all xarray datasets to pandas dataframes and reset their indices
    dataframes = [
        ds.to_dataframe().reset_index() if hasattr(ds, "to_dataframe") else ds
        for ds, _ in datasets
    ]

    # Select only the specified columns from each dataframe
    dataframes = [df[cols] for df, (_, cols) in zip(dataframes, datasets)]

    # Merge all dataframes on the specified columns
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on=merge_on)

    # Select the desired columns for the final dataframe
    merged_df = merged_df[selected_columns]
    final_merged_df = pd.merge(merged_df, output_datasets, on=merge_on)

    # Add any additional computed columns
    if additional_columns:
        for col_name, func in additional_columns.items():
            final_merged_df[col_name] = func(final_merged_df)

    final_merged_df["doy"] = final_merged_df["time"].dt.dayofyear
    final_merged_df.drop(columns=["time"], inplace=True)
    col = final_merged_df.pop("doy")
    final_merged_df.insert(2, "doy", col)
    return final_merged_df


def plot_variable_histograms(variables):
    """
    Plots histograms for the given variables.

    Parameters:
        variables (dict): A dictionary where keys are variable names and values are data arrays.
    """
    plt.figure(figsize=(12, 8))
    for i, (name, data) in enumerate(variables.items()):
        plt.subplot(3, 3, i + 1)
        plt.hist(data, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title(name, fontsize=10)
        plt.xlabel("Value", fontsize=8)
        plt.ylabel("Frequency", fontsize=8)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_depth_levels_vs_depth(dss):
    """
    Plots depth levels against depth in meters from the given dataset.

    Parameters:
        dss (dict): Dictionary containing the dataset with 'density_profile'.

    Returns:
        None
    """
    # Extract unique depth values
    depth_values = np.unique(dss["density_profile"].depth)

    # Create depth levels (assuming increasing order)
    depth_levels = np.arange(1, len(depth_values) + 1)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(depth_values, depth_levels, marker="o", linestyle="-")
    plt.xlabel("Depth in meters")
    plt.ylabel("Depth Level")
    plt.yticks(depth_levels)
    plt.gca().invert_yaxis()  # Invert y-axis so deeper levels appear at the bottom
    plt.title("Depth Level vs Depth in Meters")
    plt.grid()
    plt.show()


def plot_dp_distributions(dataframe):
    """
    Plots histograms for the distribution of density profile (DP) values at each depth level.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing DP columns (e.g., DP_1, DP_2, ..., DP_21).
    """
    plt.figure(figsize=(12, 8))
    for i in range(21):
        plt.subplot(5, 5, i + 1)
        plt.hist(
            dataframe[f"DP_{i + 1}"].values.flatten(),
            bins=21,
            color="lightgreen",
            edgecolor="black",
            alpha=0.7,
        )
        plt.title(f"DP Distribution at Depth Level {i + 1}", fontsize=10)
        plt.xlabel("Value", fontsize=8)
        plt.ylabel("Frequency", fontsize=8)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, data):
    """
    Plots a heatmap of the correlation between SF0 components and other variables.

    Parameters:
        df (pd.DataFrame): The full dataframe containing the data.
        data (dict): A dictionary containing the data to be used for correlation.

    Returns:
        None
    """

    # Add DP columns to the data dictionary
    variables = list(data.keys())
    for i in range(21):
        data[f"DP_{21 - i}"] = df.iloc[:, 21 + len(variables) - i - 1].values.flatten()

    # Create a DataFrame from the data dictionary

    df = pd.DataFrame(data)

    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Select the relevant correlations
    sf0_correlation = correlation_matrix.loc["DP_21":"DP_1", variables]

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(sf0_correlation, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation of SF0 Components with Other Variables")
    plt.show()


def prepare_data_pipeline(full_df, variables, dp_config=None):
    """
    Flexible data preparation pipeline that allows user to specify variables

    Args:
        full_df: Input DataFrame
        variables: List of dictionaries specifying variables to include in order
            Each dict should have:
            - 'name': column name in DataFrame
            - 'type': 'static' (single column) or 'dp' (DP array)
            For DP arrays, can optionally specify:
            - 'mm1': start index (default 0)
            - 'mm2': end index (default 21)
            - 'reverse': whether to reverse DP order (default True)
        dp_config: Optional global DP configuration (if not specified in individual variables)
            - 'mm1': default start index (default 0)
            - 'mm2': default end index (default 21)
            - 'reverse': default reverse flag (default True)

    Returns:
        numpy array of combined data
    """
    if dp_config is None:
        dp_config = {"mm1": 0, "mm2": 21, "reverse": True}

    # Process each variable and collect data chunks
    data_chunks = []
    total_columns = 0

    for var in variables:
        if var["type"] == "static":
            # Single column variable
            data = full_df[var["name"]].values.reshape(-1, 1)
            data_chunks.append(data)
            total_columns += 1
        elif var["type"] == "dp":
            # DP array variable
            mm1 = var.get("mm1", dp_config["mm1"])
            mm2 = var.get("mm2", dp_config["mm2"])
            reverse = var.get("reverse", dp_config["reverse"])

            dp_arrays = []
            for _, row in full_df.iterrows():
                dp_range = (
                    range(int(var["name"].split("_")[-1]), 0, -1)
                    if reverse
                    else range(1, int(var["name"].split("_")[-1]) + 1)
                )
                result = row[
                    [f"{var['name'].split('_')[0]}_{x}" for x in dp_range]
                ].values.flatten()[mm1:mm2]
                dp_arrays.append(result)

            dp_array = np.array(dp_arrays)
            data_chunks.append(dp_array)
            total_columns += mm2 - mm1

    # Combine all data chunks
    if not data_chunks:
        return np.empty((len(full_df), 0))

    data_load_main = np.hstack(data_chunks)
    return data_load_main


def preprocess_train_valid_data(data_load, num_input_features=3, val_split=0.2):
    """
    Preprocess data with customizable number of input features.

    Args:
        data_load: Input data (n_samples x n_features)
        num_input_features: Number of columns at start to treat as input features
        val_split: Fraction of data to use for validation

    Returns same values as original function in same order:
        train_data_transformed, x, y, stats, output_means, output_stds,
        valid_x, valid_y, global_maxes
    """
    # Create and shuffle indices
    ind = np.arange(0, len(data_load), 1)
    ind_shuffle = copy.deepcopy(ind)
    # np.random.shuffle(ind_shuffle)

    # Calculate validation split
    val_size = int(len(ind_shuffle) * val_split)
    train_size = len(ind_shuffle) - val_size

    # Split indices into training and validation
    train_indices = ind_shuffle[:train_size]
    val_indices = ind_shuffle[train_size:]

    # Split data into training and validation sets
    train_data = data_load[train_indices]
    val_data = data_load[val_indices]

    # Calculate means and stds for input features ONLY on training data
    input_means = np.nanmean(train_data[:, :num_input_features], axis=0)
    input_stds = np.nanstd(train_data[:, :num_input_features], axis=0)

    # Store global maxes for output features from training data
    global_maxes = np.nanmax(train_data[:, num_input_features:], axis=0)

    def log_transform_outputs(data):
        data_transformed = data.copy()
        for j in range(len(data[:, 0])):
            j_data = data[j, num_input_features:]
            maxe = np.nanmax(data[j, num_input_features:])
            data_transformed[j, num_input_features:] = np.log((j_data / maxe) + 1e-7)
        return data_transformed

    train_data_log = log_transform_outputs(train_data)
    val_data_log = log_transform_outputs(val_data)

    # Transform training data
    train_data_transformed = train_data_log.copy()
    for i in range(num_input_features):
        train_data_transformed[:, i] = (
            train_data_log[:, i] - input_means[i]
        ) / input_stds[i]

    # Calculate means and stds for output columns ONLY on training data
    output_means = np.mean(train_data_log[:, num_input_features:], axis=0)
    output_stds = np.std(train_data_log[:, num_input_features:], axis=0)

    for k in range(train_data_transformed.shape[1] - num_input_features):
        train_data_transformed[:, k + num_input_features] = (
            train_data_transformed[:, k + num_input_features] - output_means[k]
        ) / output_stds[k]

    # Transform validation data using training statistics
    val_data_transformed = val_data_log.copy()
    for i in range(num_input_features):
        val_data_transformed[:, i] = (val_data_log[:, i] - input_means[i]) / input_stds[
            i
        ]

    for k in range(val_data_transformed.shape[1] - num_input_features):
        val_data_transformed[:, k + num_input_features] = (
            val_data_transformed[:, k + num_input_features] - output_means[k]
        ) / output_stds[k]

    # Extract features and labels
    x = train_data_transformed[:, :num_input_features]
    y = train_data_transformed[:, num_input_features:]

    valid_x = val_data_transformed[:, :num_input_features]
    valid_y = val_data_transformed[:, num_input_features:]

    # Prepare statistics for potential inverse transformation
    stats = np.array(
        [(input_means[i], input_stds[i]) for i in range(num_input_features)]
    ).flatten()

    return (
        train_data_transformed,
        x,
        y,
        stats,
        output_means,
        output_stds,
        valid_x,
        valid_y,
        global_maxes,
    )


# NN


class learnKappa_layers(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)  # First layer: Input to hidden
        self.linear2 = nn.Linear(Hid, Hid)  # Second layer: Hidden to hidden
        self.linear3 = nn.Linear(Hid, Out_nodes)  # Third layer: Hidden to output
        self.dropout = nn.Dropout(0.25)  # Dropout for regularization

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)  # ReLU activation for layer 1
        h1 = self.dropout(h1)  # Apply dropout

        h2 = self.linear2(h1)
        h3 = torch.relu(h2)  # ReLU activation for layer 2
        h3 = self.dropout(h3)  # Apply dropout

        y_pred = self.linear3(h3)  # Final output layer
        return y_pred


def modeltrain_loss(
    In_nodes,
    Hid,
    Out_nodes,
    lr,
    epochs,
    x,
    y,
    valid_x,
    valid_y,
    model,
    device,
    k_mean,
    k_std,
    patience=10,
):
    # Loss weighting option here, all weights = 1.0 in default run
    # Weight settings here
    kms1 = 1.0
    kms2 = 1.0
    # Weight per node set here. Set weights to kms1 or to kms2 to use values above
    k21 = kms1
    k20 = kms1
    k19 = kms1
    k18 = kms1
    k17 = kms1
    k16 = kms1
    k15 = kms1
    k14 = kms1
    k13 = kms1
    k12 = kms1
    k11 = kms1
    k10 = kms1
    k9 = kms1
    k8 = kms1
    k7 = kms1
    k6 = kms1
    k5 = kms1
    k4 = kms1
    k3 = kms1
    k2 = kms1
    k1 = kms1
    # Weight for each layer.
    # ARRANGED FROM TOP (node = 16) TO BOTTOM (node = 1)
    kmask = np.array(
        [
            k21,
            k20,
            k19,
            k18,
            k17,
            k16,
            k15,
            k14,
            k13,
            k12,
            k11,
            k10,
            k9,
            k8,
            k7,
            k6,
            k5,
            k4,
            k3,
            k2,
            k1,
        ]
    )
    k_std_y = torch.tensor(kmask).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)  # Adam optimizer
    loss_fn = torch.nn.L1Loss(reduction="mean")  # L1 loss for gradient computation
    loss_array = torch.zeros(
        [epochs, 3]
    )  # Array to store epoch, train, and validation losses

    best_loss = float("inf")  # Initialize the best validation loss as infinity
    no_improvement = 0  # Counter for epochs without improvement
    best_model_state = None  # Placeholder for the best model state

    # Add a progress bar
    with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
        for k in range(epochs):
            optimizer.zero_grad()  # Clear gradients from the previous step
            y_pred = model(x)  # Forward pass for training data

            valid_pred = model(valid_x)  # Forward pass for validation data

            # Loss used for gradient calculation
            loss = loss_fn(y_pred * k_std_y, y * k_std_y)

            loss_train = torch.mean(
                torch.abs(
                    torch.exp(y_pred * k_std + k_mean) - torch.exp(y * k_std + k_mean)
                )
            )
            loss_valid = torch.mean(
                torch.abs(
                    torch.exp(valid_pred * k_std + k_mean)
                    - torch.exp(valid_y * k_std + k_mean)
                )
            )

            loss.backward()  # Backpropagate the gradient
            optimizer.step()  # Update model parameters

            # Record the losses for this epoch
            loss_array[k, 0] = k
            loss_array[k, 1] = loss_train.item()
            loss_array[k, 2] = loss_valid.item()

            # Update the progress bar with the current epoch and losses
            pbar.set_postfix(
                train_loss=loss_train.item(),
                valid_loss=loss_valid.item(),
                patience_count=no_improvement,
            )
            pbar.update(1)  # Increment the progress bar

            # Early stopping: Check if validation loss improves
            if loss_valid.item() < best_loss:
                best_loss = loss_valid.item()  # Update best loss
                no_improvement = 0
                best_model_state = model.state_dict()
            else:
                no_improvement += 1  # Increment no improvement counter

            # If no improvement for 'patience' epochs, stop training
            if no_improvement >= patience:
                print(
                    f"\nEarly stopping at epoch {k + 1}. Validation loss has not improved for {patience} epochs."
                )
                break

            # Free memory by deleting intermediate variables
            del loss, y_pred

    # Restore the best model state after training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, loss_array[:k, :]


def plot_training_validation_loss(
    loss_array, title="Training and Validation Loss Over Epochs"
):
    """
    Plots the training and validation loss over epochs.

    Parameters:
        loss_array (numpy.ndarray): A 2D array where the first column represents epochs,
                                    the second column represents training loss, and the
                                    third column represents validation loss.
        title (str): Title of the plot (default is "Training and Validation Loss Over Epochs").
    """
    plt.plot(loss_array[:, 0], loss_array[:, 1], label="Training Loss")
    plt.plot(loss_array[:, 0], loss_array[:, 2], label="Validation Loss")

    # Adding labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)

    # Adding a legend
    plt.legend()

    # Displaying the plot
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def plot_density_profile(row_index, prediction, valid_y, k_std, k_mean, training_max):
    """
    Plot predicted vs actual density profile for a specific row.

    Args:
        row_index: Index of the row to plot
        prediction: Model predictions (before inverse transform)
        valid_y: Ground truth values (before inverse transform)
        k_std: Standard deviations used for normalization
        k_mean: Means used for normalization
        training_max: Global max values used for inverse log transform
    """
    depth = list(range(1, 22))  # Depth values (1 to 21)

    # Inverse transform predictions and ground truth
    predicted_dp_valus = prediction
    predicted_dp_values = np.exp(predicted_dp_valus * k_std + k_mean) * training_max
    valid_dp_valus = valid_y
    valid_dp_values = np.exp(valid_dp_valus * k_std + k_mean) * training_max

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        predicted_dp_values,
        depth[::-1],
        marker="o",
        linestyle="-",
        color="b",
        label="Predicted",
    )
    plt.plot(
        valid_dp_values,
        depth[::-1],
        marker="o",
        linestyle="-",
        color="r",
        label="Actual",
    )
    plt.yticks(depth)
    plt.xlabel("DP Value")
    plt.ylabel("Depth")
    plt.title(f"Density Profile (Row {row_index})")
    plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
    plt.grid(True)
    plt.legend()
    plt.show()


def get_hist(y, k_mean, k_std):
    """Get histogram values for normalized data."""
    vals, binss = np.histogram(np.exp(y * k_std + k_mean), range=(0, 1.2), bins=100)
    return vals, 0.5 * (binss[0:-1] + binss[1:])


def get_hist2(y):
    """Get histogram values for error data."""
    vals, binss = np.histogram(y, range=(-0.2, 0.2), bins=100)
    return vals, 0.5 * (binss[0:-1] + binss[1:])


def performance_sigma_point(model, x, valid_x, y, valid_y, k_mean, k_std, training_max):
    """Plot the performance of a neural network model.

    Parameters:
        model: Trained neural network model.
        x: Training input data.
        valid_x: Validation input data.
        y: Training output data.
        valid_y: Validation output data.
        k_mean: Mean normalization values.
        k_std: Standard deviation normalization values.
    """
    # plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams["font.size"] = 15
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = (
        "stix"  # ensures it can math compatibility with symbols in your code without erroring fix no cursive_fontsystem
    )

    y_pred_train = model(x)
    y_pred_test = model(valid_x)

    ycpu = y.cpu().detach().numpy()
    ytestcpu = valid_y.cpu().detach().numpy()
    yptraincpu = y_pred_train.cpu().detach().numpy()
    yptestcpu = y_pred_test.cpu().detach().numpy()

    ystd = np.zeros(21)
    yteststd = np.zeros(21)
    ypstd = np.zeros(21)
    ypteststd = np.zeros(21)
    yerr = np.zeros(21)
    kappa_mean = np.zeros(21)

    for i in range(21):
        ystd[i] = np.std(np.exp(ycpu[:, i] * k_std[i] + k_mean[i]))
        yteststd[i] = np.std(np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]))
        ypstd[i] = np.std(np.exp(yptraincpu[:, i] * k_std[i] + k_mean[i]))
        ypteststd[i] = np.std(np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]))
        yerr[i] = np.std(
            np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i])
            - np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i])
        )

        kappa_mean[i] = np.mean(np.exp(ycpu[:, i] * k_std[i] + k_mean[i]))

    plt.figure(figsize=(20, 10))

    ind = np.arange(0, 21)
    ind_tick = np.arange(1, 22)[::-1]

    # Subplot 1: Boxplot of network output differences
    plt.subplot(1, 4, 1)
    for i in range(21):
        plt.boxplot(
            ytestcpu[:, i] - yptestcpu[:, i],
            vert=False,
            positions=[i],
            showfliers=False,
            whis=(5, 95),
            widths=0.5,
        )
    plt.xlim([-2.0, 2.0])
    plt.yticks(ind, ind_tick)
    plt.title(r"(a) Output of network $\mathcal{N}_1$ ")
    plt.ylabel("Node")

    # Subplot 2: Boxplot of shape function differences
    plt.subplot(1, 4, 2)
    for i in range(21):
        plt.boxplot(
            kappa_mean[i]
            + (np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]))
            - np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]),
            vert=False,
            positions=[i],
            showfliers=False,
            whis=(5, 95),
            widths=0.5,
        )
    plt.yticks([])
    plt.title(r"(b) Density Profile")
    plt.xlabel(r"kg m-3")

    # Subplots 3 & 4: Histograms
    k12 = 20
    for k in range(21):
        plt.subplot(21, 4, 4 * k + 3)
        vals, binss = get_hist(ytestcpu[:, k12], k_mean[k12], k_std[k12])
        plt.plot(binss, vals, color="blue")

        vals, binss = get_hist(yptestcpu[:, k12], k_mean[k12], k_std[k12])
        plt.plot(binss, vals, color="red")
        if k < 20:
            plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title("(c) Probability density histogram")

        plt.subplot(21, 4, 4 * k + 4)
        vals, binss = get_hist2(
            np.exp(ytestcpu[:, k12] * k_std[k12] + k_mean[k12])
            - np.exp(yptestcpu[:, k12] * k_std[k12] + k_mean[k12])
        )
        plt.plot(binss, vals, color="green")
        if k < 20:
            plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title("(d) Error histogram ")

        k12 -= 1

    plt.tight_layout()


class learnKappa_layers1(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers1, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)  # Input to hidden layer
        self.linear2 = nn.Linear(Hid, Out_nodes)  # Hidden to output layer
        self.dropout = nn.Dropout(0.25)  # Dropout to reduce overfitting

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)  # ReLU activation
        h1 = self.dropout(h1)
        y_pred = self.linear2(h1)  # Output predictions
        return y_pred


class learnKappa_layers2(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers2, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        y_pred = self.linear3(h3)
        return y_pred


class learnKappa_layers3(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers3, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Hid)
        self.linear4 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        h4 = self.linear3(h3)
        h5 = torch.relu(h4)
        h5 = self.dropout(h5)
        y_pred = self.linear4(h5)
        return y_pred


class learnKappa_layers4(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers4, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Hid)
        self.linear4 = nn.Linear(Hid, Hid)
        self.linear5 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        h4 = self.linear3(h3)
        h5 = torch.relu(h4)
        h5 = self.dropout(h5)
        h6 = self.linear4(h5)
        h7 = torch.relu(h6)
        h7 = self.dropout(h7)
        y_pred = self.linear5(h7)
        return y_pred


def create_highlighted_df(hyper_parameters_and_losses):
    """
    Converts a list of hyperparameter-loss tuples into a pandas DataFrame,
    highlights the row with the lowest validation loss, and returns the styled DataFrame.

    Parameters:
        hyper_parameters_and_losses (list): List of tuples containing
            (hidden layers, hidden units, training loss tensor, validation loss tensor)

    Returns:
        pd.Styler: Styled DataFrame with the lowest validation loss row highlighted.
    """
    # Convert list to DataFrame
    df = pd.DataFrame(
        hyper_parameters_and_losses,
        columns=[
            "Hidden Layers",
            "Hidden Units",
            "Learning Rate",
            "Training Loss",
            "Validation Loss",
        ],
    )

    # Convert tensor values to floats
    df["Training Loss"] = df["Training Loss"].apply(lambda x: x.item())
    df["Validation Loss"] = df["Validation Loss"].apply(lambda x: x.item())

    # Find the index of the row with the lowest validation loss
    min_val_loss_idx = df["Validation Loss"].idxmin()

    # Function to highlight row
    def highlight_row(s):
        return [
            "background-color: yellow" if s.name == min_val_loss_idx else "" for _ in s
        ]

    # Apply highlighting
    return df.style.apply(highlight_row, axis=1)


def run_hyperparameter_sweep(
    k_mean, k_std, x, y, valid_x, valid_y, device, epochs=3000, k_points=21
):
    hid_array = np.array([32, 64])
    lrs = np.array([1e-2, 1e-3])
    lays = np.array([1, 2, 3])
    torch.manual_seed(10)

    k_mean_c = torch.tensor(k_mean).float().to(device)
    k_std_c = torch.tensor(k_std).float().to(device)

    hyper_parameters_and_losses = []

    for la in lays:
        for h in hid_array:
            for lr in lrs:
                in_nod, hid_nod, o_nod = 3, h, 21
                print("la, h, lr is >", la, h, lr)

                model_classes = {
                    1: learnKappa_layers1,
                    2: learnKappa_layers2,
                    3: learnKappa_layers3,
                    4: learnKappa_layers4,
                }

                model = model_classes.get(la, lambda *args: print("Check code"))(
                    in_nod, hid_nod, o_nod
                )
                model = model.to(device)

                model, loss_array = modeltrain_loss(
                    in_nod,
                    hid_nod,
                    o_nod,
                    lr,
                    epochs,
                    x,
                    y,
                    valid_x,
                    valid_y,
                    model,
                    device,
                    k_mean_c,
                    k_std_c,
                    patience=1000,
                )

                training_loss = loss_array[:, 1]  # Training loss
                validation_loss = loss_array[:, 2]  # Validation loss

                final_train_loss = training_loss[-1]
                final_valid_loss = validation_loss[-1]
                hyper_parameters_and_losses.append(
                    (la, h, lr, final_train_loss, final_valid_loss)
                )

                del model, loss_array
                torch.cuda.empty_cache()

    return hyper_parameters_and_losses
