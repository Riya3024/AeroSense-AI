import pandas as pd
import numpy as np
import json
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path("data/processed/splits")
OUTPUT = Path("data/processed/xgboost_results.csv")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgboost_pm25.json"
FEATURE_PATH = MODEL_DIR / "xgboost_features.json"

DROP_COLUMNS = [
    "target_1h",
    "period_from_utc",
    "period_to_utc",
    "pollutant",
    "parameter",
    "unit",
]


def load_data(path):
    df = pd.read_csv(path)

    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    y = df["target_1h"]

    X = X.select_dtypes(include=[np.number])

    return X, y


print("=" * 70)
print("AEROSENSE — XGBOOST 1-HOUR FORECAST")
print("=" * 70)

print("\nLoading datasets...")

X_train, y_train = load_data(BASE / "train.csv")
X_val, y_val = load_data(BASE / "validation.csv")
X_test, y_test = load_data(BASE / "test.csv")

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")
print(f"Test rows:       {len(X_test):,}")
print(f"Features:        {X_train.shape[1]}")

print("\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    objective="reg:squarederror",
    eval_metric="mae",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Training complete.")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model.save_model(MODEL_PATH)

with open(FEATURE_PATH, "w", encoding="utf-8") as f:
    json.dump(
        X_train.columns.tolist(),
        f,
        indent=2
    )

print(f"Model saved:    {MODEL_PATH}")
print(f"Features saved: {FEATURE_PATH}")


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
    print("-" * 55)
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
print("=" * 55)
print(importance.head(20).to_string(index=False))


result_df = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

result_df.to_csv(
    OUTPUT,
    index=False
)

importance.to_csv(
    "data/processed/xgboost_feature_importance.csv",
    index=False
)

print("\n" + "=" * 70)
print("XGBOOST COMPLETE")
print("=" * 70)

print(result_df.to_string(index=False))

print("\nSaved:")
print("data/processed/xgboost_results.csv")
print("data/processed/xgboost_feature_importance.csv")