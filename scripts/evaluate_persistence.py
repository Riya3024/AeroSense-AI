import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path("data/processed/splits")

def evaluate(name, path):

    df = pd.read_csv(path)

    y_true = df["target_1h"]
    y_pred = df["value"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    extreme = y_true > 500
    severe = y_true > 1000

    extreme_mae = (
        mean_absolute_error(
            y_true[extreme],
            y_pred[extreme]
        )
        if extreme.any()
        else np.nan
    )

    severe_mae = (
        mean_absolute_error(
            y_true[severe],
            y_pred[severe]
        )
        if severe.any()
        else np.nan
    )

    print()
    print(name)
    print("-" * 45)
    print(f"Rows:              {len(df):,}")
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


print("=" * 55)
print("AEROSENSE — PERSISTENCE BASELINE")
print("=" * 55)

results = []

results.append(
    evaluate(
        "VALIDATION",
        BASE / "validation.csv"
    )
)

results.append(
    evaluate(
        "TEST",
        BASE / "test.csv"
    )
)

result_df = pd.DataFrame(results)

result_df.to_csv(
    "data/processed/persistence_baseline.csv",
    index=False
)

print()
print("=" * 55)
print("BASELINE COMPLETE")
print("=" * 55)
print()
print(result_df.to_string(index=False))
print()
print("Saved:")
print("data/processed/persistence_baseline.csv")
