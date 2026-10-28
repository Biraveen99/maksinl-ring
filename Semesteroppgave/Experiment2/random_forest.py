
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# 1. Load the dataset
df = pd.read_csv(r"C:\Users\birav\Desktop\git\UiA\IKT466-1_26H\Semesteroppgave\Experiment2\dataset\hour.csv")

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# 2. Select only time-related features
features = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday"
]

X = df[features]
y = df["cnt"]

# 3. Split the data in chronological order
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# 4. Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Predict on validation data
predictions = model.predict(X_val)

# 6. Calculate evaluation metrics
mae = mean_absolute_error(y_val, predictions)
rmse = sqrt(mean_squared_error(y_val, predictions))

print("\nRandom Forest - Validation Results")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 7. Final evaluation on the test set
test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = sqrt(mean_squared_error(y_test, test_predictions))

print("\nRandom Forest - Test Results")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")