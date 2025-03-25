import datetime
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
from determine_mld import (
    profile_types,
    load,
    filter_profiles,
    get_mld_from_threshold,
    stepwise_interpolate_profile_mld_safe,
)
from netCDF4 import num2date
from scipy.ndimage import gaussian_filter1d
from utils import torch_backend

import random

random.seed(42)


def load_data():
    ds = xr.open_dataset('data/processed/ows_papa.nc')
    return ds

class ProfileDataset(Dataset):
    def __init__(self, xr_dataset):
        # === Load raw variables ===
        density = xr_dataset['density'].values.astype(np.float32)      # (T, 300)
        diffusivity = xr_dataset['diffusivity'].values.astype(np.float32)  # (T, 300)

        # === Normalize density per profile (feature-wise standardization) ===
        self.X_mean = np.mean(density, axis=1, keepdims=True)
        self.X_std = np.std(density, axis=1, keepdims=True) + 1e-6
        X_norm = (density - self.X_mean) / self.X_std

        # === Normalize diffusivity per profile ===
        self.y_mean = np.mean(diffusivity, axis=1, keepdims=True)
        self.y_std = np.std(diffusivity, axis=1, keepdims=True) + 1e-6
        y_norm = (diffusivity - self.y_mean) / self.y_std

        # === Create profile and target tensors ===
        self.X_profiles = torch.tensor(X_norm, dtype=torch.float32)  # (T, 300)
        self.y = torch.tensor(y_norm, dtype=torch.float32)           # (T, 300)
        self.raw_y_mean = torch.tensor(self.y_mean, dtype=torch.float32)
        self.raw_y_std = torch.tensor(self.y_std, dtype=torch.float32)

        # === Extract auxiliary features ===
        time = xr_dataset['time']
        self.time = time
        doy = time.dt.dayofyear.values.astype(np.float32)            # (T,)
        hour = time.dt.hour.values.astype(np.float32)                # (T,)

        # Cyclical encoding for time
        doy_rad = 2 * np.pi * doy / 365.0
        hour_rad = 2 * np.pi * hour / 24.0

        doy_sin = np.sin(doy_rad)
        doy_cos = np.cos(doy_rad)
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        # === Normalize MLD by max depth ===
        mld = xr_dataset['mld_depth'].values.astype(np.float32)      # (T,)
        max_depth = xr_dataset['depth'].values.max()
        mld_norm = mld / max_depth

        # === Stack auxiliary features into one tensor: (T, 5) ===
        self.aux_features = torch.tensor(
            np.stack([doy_sin, doy_cos, hour_sin, hour_cos, mld_norm], axis=1),
            dtype=torch.float32
        )

    def __len__(self):
        return self.X_profiles.shape[0]

    def __getitem__(self, idx):
        X = torch.cat((self.X_profiles[idx], self.aux_features[idx]), dim=0)  # (305,)
        y = self.y[idx]
        y_mean = self.raw_y_mean[idx]
        y_std = self.raw_y_std[idx]
        return X, y, y_mean, y_std


class ProfileRegressor(nn.Module):
    def __init__(self, input_dim=305):  # 300 profile + 5 aux features
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 300)
        )

    def forward(self, x):
        return self.model(x)



def get_dataloaders(xr_dataset, batch_size=128):
    dataset = ProfileDataset(xr_dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)  # load all test data at once

    return train_loader, test_loader


def train(ds, epochs=1):
    train_loader, test_loader = get_dataloaders(ds)

    # Model, Loss, Optimizer
    device = torch.device(torch_backend())
    model = ProfileRegressor().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch, _, _ in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # === Save global target stats from training data ===
    # You only need the actual dataset once here
    dataset = train_loader.dataset.dataset  # unwrap the Subset -> original ProfileDataset
    global_y_mean = dataset.raw_y_mean.mean().item()
    global_y_std = dataset.raw_y_std.mean().item()

    model.global_y_mean = global_y_mean
    model.global_y_std = global_y_std
    print(f"Saved training y_mean: {global_y_mean:.2e}, y_std: {global_y_std:.2e}")

    return model, test_loader


def plot_predictions(model, test_loader, depth, num_profiles=1, show_actual=True, index=None):
    model = model.cpu()  # evaluation is light, so keep it on CPU
    model.eval()

    # Load all test data in one go
    X_batch, y_batch, y_mean_batch, y_std_batch = next(iter(test_loader))

    total_profiles = X_batch.shape[0]
    indexes = random.sample(range(total_profiles), num_profiles)
    if index:
        indexes = [index]

    global_y_mean = torch.tensor(model.global_y_mean)
    global_y_std = torch.tensor(model.global_y_std)

    with torch.no_grad():
        preds = model(X_batch)

    for i in indexes:
        X = X_batch[i]
        y = y_batch[i]
        y_mean = y_mean_batch[i]
        y_std = y_std_batch[i]
        y_pred = preds[i]

        # Check for NaNs
        if torch.isnan(y_std).any() or torch.isnan(y_mean).any():
            y_std = global_y_std
            y_mean = global_y_mean


        # Denormalize
        y_pred_denorm = (y_pred * y_std + y_mean).numpy().flatten()
        y_true_denorm = (y * y_std + y_mean).numpy().flatten()

        if np.isnan(y_pred_denorm).all():
            continue

        # Plot
        plt.figure(figsize=(6, 4))
        if show_actual:
            plt.plot(y_true_denorm, depth, label='Actual', linewidth=2)
            plt.plot(y_pred_denorm, depth, label='Predicted', linestyle='--')
        else:
            plt.plot(y_pred_denorm, depth, label='Prediction', linewidth=1, alpha=0.3)
            y_pred_smooth = gaussian_filter1d(y_pred_denorm, sigma=3)
            plt.plot(y_pred_smooth, depth, label='Prediction (smoothed)', linewidth=2)


        plt.xlabel("Diffusivity")
        plt.ylabel("Depth")

        if hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'indices'):
            original_index = test_loader.dataset.indices[i]
            raw_time = test_loader.dataset.dataset.time[original_index].values  # numpy.datetime64
        else:
            original_index = i
            raw_time = test_loader.dataset.time[i].values
        dt = raw_time.astype('M8[ms]').astype(datetime.datetime)
        friendly = dt.strftime("%b %d, %Y %I:%M %p")


        plt.title(f"Profile {i} at {friendly}")
        plt.legend()
        plt.tight_layout()
        plt.show()



def evaluate_model(model, test_loader):
    model = model.cpu()
    model.eval()

    # Load everything in one batch
    X, y, y_mean, y_std = next(iter(test_loader))

    with torch.no_grad():
        pred = model(X)

    # Denormalize in one go
    y_pred_denorm = pred * y_std + y_mean
    y_true_denorm = y * y_std + y_mean

    y_true = y_true_denorm.numpy()
    y_pred = y_pred_denorm.numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print("→ y_true mean:", y_true.mean(), "std:", y_true.std())
    print("→ y_pred mean:", y_pred.mean(), "std:", y_pred.std())
    print("→ MSE:", mean_squared_error(y_true, y_pred))
    print("→ Var(y_true):", np.var(y_true))
    r2 = 1 - mean_squared_error(y_true, y_pred) / np.var(y_true)
    print("→ R² from formula:", r2)

    print(f"Evaluation Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.4f}")


def observational_to_dataset(profile_type_name):
    profile_type = profile_types[profile_type_name]
    ds_obs = load(profile_type["path"])

    depths_obs = ds_obs["depth"][:]
    times = ds_obs["time"][:]
    time_calendar = "standard"
    times_dt = num2date(times, units=ds_obs['time'].units, calendar=time_calendar)


    profiles = filter_profiles(ds_obs, profile_type)

    depth_grid = np.linspace(0, 300, 300)

    profiles_interp = []
    for profile in profiles:
        valid_mask = ~np.isnan(profile)
        filled_profile = stepwise_interpolate_profile_mld_safe(profile, valid_mask)
        interp = np.interp(depth_grid, depths_obs, filled_profile)  # now safe to interpolate
        profiles_interp.append(interp)

    profiles_interp = np.array(profiles_interp)

    # Compute MLDs from original observed depths
    mld_depths, _ = get_mld_from_threshold(profiles, depths_obs, profile_type["threshold"])

    # Create dummy diffusivity array (same shape as density)
    dummy_diffusivity = np.full_like(profiles_interp, fill_value=np.nan)  # won't be used

    # Build xarray.Dataset
    return xr.Dataset(
        data_vars={
            "density": (("time", "depth"), profiles_interp),
            "diffusivity": (("time", "depth"), dummy_diffusivity),
            "mld_depth": (("time",), mld_depths),
        },
        coords={
            "depth": depth_grid,
            "time": ("time", np.array(times_dt, dtype="datetime64[ns]")),
        },
    )


def predict_from_observations(xr_dataset, model):
    obs_dataset = ProfileDataset(xr_dataset)
    obs_loader = DataLoader(obs_dataset, batch_size=len(obs_dataset))

    ds = load_data()
    depth = ds['depth'].values

    # Plot
    plot_predictions(model, obs_loader, depth=depth, num_profiles=5, show_actual=False)




def main():
    ds = load_data()
    depth = ds['depth'].values

    model, test_loader = train(ds, epochs=6)
    plot_predictions(model, test_loader, depth)
    y_true, y_pred = evaluate_model(model, test_loader)
    return model, test_loader, depth