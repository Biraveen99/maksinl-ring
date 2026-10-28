
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Make results reproducible
torch.manual_seed(42)
np.random.seed(42)

# 2. Load dataset
data_path = (
    Path(__file__).resolve().parent
    / "bike+sharing+dataset (1)"
    / "data"
    / "hour.csv"
)

df = pd.read_csv(data_path)

# 3. Select input features (including weather)
features = [
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

X = df[features]
y = df["cnt"]

# 4. Chronological 70/15/15 split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

# 5. Standardize input features using training data only
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Standardize target values using training data only
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
)

y_val_scaled = y_scaler.transform(
    y_val.to_numpy().reshape(-1, 1)
)

# 6. Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val = torch.tensor(y_val_scaled, dtype=torch.float32)

# 7. Define a small neural network
# 12 input features, two hidden layers, one output
model = nn.Sequential(
    nn.Linear(12, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 8. Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 9. Train with early stopping
epochs = 2000
patience = 200

best_val_loss = float("inf")
best_epoch = 0
best_weights = None

for epoch in range(epochs):
    model.train()

    predictions = model(X_train)
    loss = loss_function(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check validation loss after each epoch
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_function(val_predictions, y_val)

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_epoch = epoch + 1
        best_weights = deepcopy(model.state_dict())

    if (epoch + 1) % 200 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {loss.item():.4f} - "
            f"Validation Loss: {val_loss.item():.4f}"
        )

    if (epoch + 1) - best_epoch >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

# Restore the best model
model.load_state_dict(best_weights)

print(f"Best epoch: {best_epoch}")

# 10. Evaluate on validation data
model.eval()

with torch.no_grad():
    predictions_scaled = model(X_val).numpy()

# Convert predictions back to actual rental counts
predictions = y_scaler.inverse_transform(
    predictions_scaled
).flatten()

actual = y_scaler.inverse_transform(
    y_val.numpy()
).flatten()

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print("\nMLP - Validation Results")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 11. Final evaluation on the test set
model.eval()

with torch.no_grad():
    test_predictions_scaled = model(X_test).numpy()

# Convert predictions back to actual rental counts
test_predictions = y_scaler.inverse_transform(
    test_predictions_scaled
).flatten()

test_actual = y_test.to_numpy()

test_mae = mean_absolute_error(test_actual, test_predictions)
test_rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))

print("\nMLP - Test Results")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")
