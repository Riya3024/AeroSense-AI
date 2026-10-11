import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path("data/processed/splits")
OUTPUT = Path("data/processed/random_forest_results.csv")

FEATURES_TO_DROP = [
    "target_1h",
    "period_from_utc",
    "period_to_utc",
    "pollutant",
    "parameter",
    "unit",
]

def load_data(path):
    df = pd.read_csv(path)

    # Drop non-feature columns
    X = df.drop(columns=FEATURES_TO_DROP, errors="ignore")
    y = df["target_1h"]

    # Convert anything non-numeric safely
    X = X.select_dtypes(include=[np.number])

    return X, y


print("=" * 65)
print("AEROSENSE — RANDOM FOREST 1-HOUR FORECAST")
print("=" * 65)

print("\nLoading datasets...")

X_train, y_train = load_data(BASE / "train.csv")
X_val, y_val = load_data(BASE / "validation.csv")
X_test, y_test = load_data(BASE / "test.csv")

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")
print(f"Test rows:       {len(X_test):,}")
print(f"Features:        {X_train.shape[1]}")

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)

print("Training complete.")


def evaluate(name, X, y):

    pred = model.predict(X)

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    extreme = y > 500
    severe = y > 1000

    extreme_mae = (
        mean_absolute_error(y[extreme], pred[extreme])
        if extreme.any()
        else np.nan
    )

    severe_mae = (
        mean_absolute_error(y[severe], pred[severe])
        if severe.any()
        else np.nan
    )

    print()
    print(name)
    print("-" * 50)
    print(f"Rows:              {len(y):,}")
    print(f"MAE:               {mae:.4f}")
    print(f"RMSE:              {rmse:.4f}")
    print(f"R²:                {r2:.4f}")
    print(f"Extreme samples:   {extreme.sum():,}")
    print(f"Extreme MAE:       {extreme_mae:.4f}")
    print(f"Severe samples:    {severe.sum():,}")
    print(f"Severe MAE:        {severe_mae:.4f}")

    return {
        "split": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "extreme_mae": extreme_mae,
        "severe_mae": severe_mae,
    }


results = []

results.append(
    evaluate(
        "VALIDATION",
        X_val,
        y_val
    )
)

results.append(
    evaluate(
        "TEST",
        X_test,
        y_test
    )
)


# Feature importance
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nTOP 20 FEATURES")
print("=" * 50)
print(importance.head(20).to_string(index=False))


result_df = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

result_df.to_csv(
    OUTPUT,
    index=False
)

importance.to_csv(
    "data/processed/random_forest_feature_importance.csv",
    index=False
)

print("\n" + "=" * 65)
print("RANDOM FOREST COMPLETE")
print("=" * 65)

print(result_df.to_string(index=False))

print("\nSaved:")
print("data/processed/random_forest_results.csv")
print("data/processed/random_forest_feature_importance.csv")