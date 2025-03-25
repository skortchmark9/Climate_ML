import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random



def load_data():
    ds = xr.open_dataset('data/processed/ows_papa_2011_2024.nc')
    return ds

class ProfileDataset(Dataset):
    def __init__(self, xr_dataset):
        # Load data
        density = xr_dataset['density'].values.astype(np.float32)     # shape (T, 300)
        diffusivity = xr_dataset['diffusivity'].values.astype(np.float32)

        # Normalize density (X) per sample
        self.X_mean = np.mean(density, axis=1, keepdims=True)
        self.X_std = np.std(density, axis=1, keepdims=True) + 1e-6
        X_norm = (density - self.X_mean) / self.X_std

        # Normalize diffusivity (y) per sample
        self.y_mean = np.mean(diffusivity, axis=1, keepdims=True)
        self.y_std = np.std(diffusivity, axis=1, keepdims=True) + 1e-6
        y_norm = (diffusivity - self.y_mean) / self.y_std

        self.X_profiles = torch.tensor(X_norm, dtype=torch.float32)  # shape (T, 300)
        self.y = torch.tensor(y_norm, dtype=torch.float32)

        # Cyclical time encoding
        time = xr_dataset['time']
        dayofyear = time.dt.dayofyear.values.astype(np.float32)
        hour = time.dt.hour.values.astype(np.float32)

        doy_rad = 2 * np.pi * dayofyear / 365.0
        hour_rad = 2 * np.pi * hour / 24.0

        doy_sin = np.sin(doy_rad)
        doy_cos = np.cos(doy_rad)
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        self.temporal = torch.tensor(np.stack([doy_sin, doy_cos, hour_sin, hour_cos], axis=1), dtype=torch.float32)

        # Save raw stats for denormalization
        self.raw_y_mean = torch.tensor(self.y_mean, dtype=torch.float32)
        self.raw_y_std = torch.tensor(self.y_std, dtype=torch.float32)

    def __len__(self):
        return self.X_profiles.shape[0]

    def __getitem__(self, idx):
        X = torch.cat((self.X_profiles[idx], self.temporal[idx]), dim=0)  # shape (304,)
        y = self.y[idx]
        y_mean = self.raw_y_mean[idx]
        y_std = self.raw_y_std[idx]
        return X, y, y_mean, y_std  # return y stats for denormalization

class ProfileRegressor(nn.Module):
    def __init__(self, input_dim=304):  # 300 + 4 cyclical time features
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
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, test_loader



def train(ds, epochs=1):
    train_loader, test_loader = get_dataloaders(ds)

    # Model, Loss, Optimizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

    return model, test_loader


def plot_predictions(model, test_loader, depth, num_profiles=1):
    device = next(model.parameters()).device
    model.eval()

    test_points = list(test_loader)
    indexes = random.sample(range(len(test_points)), num_profiles)

    with torch.no_grad():
        for i in indexes:
            X, y, y_mean, y_std = test_points[i]
            X = X.to(device)
            y = y.to(device)
            y_mean = y_mean.to(device)
            y_std = y_std.to(device)

            y_pred = model(X)

            y_pred_denorm = (y_pred * y_std + y_mean).cpu().numpy().flatten()
            y_true_denorm = (y * y_std + y_mean).cpu().numpy().flatten()

            plt.figure(figsize=(6, 4))
            plt.plot(y_true_denorm, depth, label='Actual', linewidth=2)
            plt.plot(y_pred_denorm, depth, label='Predicted', linestyle='--')
            plt.xlabel("Diffusivity")
            plt.ylabel("Depth")
            plt.title(f"Profile {i}")
            plt.legend()
            plt.tight_layout()
            plt.show()


def evaluate_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y, y_mean, y_std in test_loader:
            X = X.to(device)
            y = y.to(device)
            y_mean = y_mean.to(device)
            y_std = y_std.to(device)

            pred = model(X)

            # Denormalize predictions and ground truth
            y_pred_denorm = pred * y_std + y_mean
            y_true_denorm = y * y_std + y_mean

            all_preds.append(y_pred_denorm.cpu().numpy())
            all_targets.append(y_true_denorm.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Evaluation Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.4f}")

    return y_true, y_pred


def main():
    ds = load_data()
    depth = ds['depth'].values

    model, test_loader = train(ds, epochs=4)
    plot_predictions(model, test_loader, depth)
    y_true, y_pred = evaluate_model(model, test_loader)
    return model, test_loader, depth