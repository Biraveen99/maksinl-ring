
from pathlib import Path

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

# 3. Selecting the same features as Random Forest
features = [
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

X = df[features]
y = df["cnt"]

# 4. Chronological 70/15/15 split as we have done for Random Forest
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

# 5. Standardize features using training data only because we want to avoid data leakage. We will use the same scaler for validation and test data.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 6. Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

# 7. Define a small neural network this was chosen to avoid overfitting and to keep the model simple. The architecture consists of two hidden layers with ReLU activation functions. The input layer has 12 neurons corresponding to the 12 features, and the output layer has 1 neuron for the regression output.
model = nn.Sequential(
    nn.Linear(12, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 8. Loss function and optimizer for regression tasks, Mean Squared Error (MSE) is commonly used as the loss function. The Adam optimizer is chosen for its efficiency and adaptive learning rate capabilities.
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 9. Train the model on 200 epochs. The training loop consists of forward propagation, loss calculation, backpropagation, and optimizer step. Every 20 epochs, the model is evaluated on the validation set to monitor its performance and check for overfitting.
epochs = 2000

for epoch in range(epochs):
    model.train()

    predictions = model(X_train)
    loss = loss_function(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        model.eval()

        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = loss_function(val_predictions, y_val)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {loss.item():.2f} - "
            f"Validation Loss: {val_loss.item():.2f}"
        )

# 10. Evaluate on validation data for comparison with Random Forest. The model is set to evaluation mode, and predictions are made on the validation set. The Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are calculated to assess the model's performance.
model.eval()

with torch.no_grad():
    predictions = model(X_val).numpy().flatten()

actual = y_val.numpy().flatten()

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print("\nMLP - Validation Results")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
