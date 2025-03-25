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
        self.X = torch.tensor(xr_dataset['density'].values, dtype=torch.float32)
        self.y = torch.tensor(xr_dataset['diffusivity'].values, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]  # number of time steps

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # both shape (300,)

class ProfileRegressor(nn.Module):
    def __init__(self, input_dim=300):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)  # output shape (300,)
        )

    def forward(self, x):
        return self.model(x)



def get_dataloaders(xr_dataset, batch_size=128):
    dataset = ProfileDataset(xr_dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)  # one profile at a time for plotting

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
        for X_batch, y_batch in train_loader:
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


def plot_predictions(model, test_loader, depth, num_profiles=5):
    device = next(model.parameters()).device
    model.eval()

    test_points = list(test_loader)
    test_point_indexes = random.sample(range(len(test_points)), num_profiles)

    with torch.no_grad():
        for i in test_point_indexes:
            X, y_true = test_points[i]

            X = X.to(device)
            y_true = y_true.to(device)
            y_pred = model(X)

            plt.figure(figsize=(6, 4))
            plt.plot(y_true.cpu().numpy().flatten(), depth, label='Actual', linewidth=2)
            plt.plot(y_pred.cpu().numpy().flatten(), depth, label='Predicted', linestyle='--')
            plt.xlabel("Diffusivity")
            plt.ylabel("Depth")
            plt.title(f"Profiles {i}")
            plt.legend()
            plt.tight_layout()
            plt.show()



def evaluate_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

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

    model, test_loader = train(ds, epochs=10)
    plot_predictions(model, test_loader, depth)
    y_true, y_pred = evaluate_model(model, test_loader)
    return model, test_loader, depth