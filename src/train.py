"""Train the model and regenerate every artifact the interface reads.

Run once after `prepare_data.py`, and again whenever `data/m5_long.parquet`
is refreshed. Everything the UI needs (booster, per-SKU MASE, forecast files)
lands in `artifacts/`.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd

from . import model as M

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "m5_long.parquet"
META = ROOT / "data" / "meta.json"
ARTIFACTS = ROOT / "artifacts"


def main(num_boost_round: int = 300) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    test_start = pd.Timestamp(json.loads(META.read_text())["test_start"])
    print(f"Training on {len(df):,} rows, {df['id'].nunique()} series, "
          f"test starts {test_start.date()}")

    t = time.time()
    tm = M.train(df, test_start, num_boost_round=num_boost_round)
    print(f"  trained in {time.time() - t:.1f}s")

    t = time.time()
    metrics = M.evaluate(tm, df, test_start, horizon=28)
    print(f"  evaluated in {time.time() - t:.1f}s | "
          f"median MASE={metrics['mase'].median():.3f} "
          f"mean MASE={metrics['mase'].mean():.3f}")

    forecast_lgb = M.predict(tm, df, test_start, horizon=28)
    forecast_naive = M.seasonal_naive_forecast(df, test_start, horizon=28)

    actual = df[df["date"] >= test_start][["id", "date", "sales"]]
    hist = df[df["date"] < test_start][["id", "sales"]]
    rows = []
    for sid, g in forecast_naive.merge(actual, on=["id", "date"]).groupby("id", observed=True):
        h = hist.loc[hist["id"] == sid, "sales"].values
        rows.append({"id": sid,
                     "mase": M.mase(g["sales"].values, g["forecast"].values, h)})
    baseline_mase = pd.DataFrame(rows)
    print(f"  baseline median MASE={baseline_mase['mase'].median():.3f} "
          f"| win rate {(metrics.set_index('id')['mase'] < baseline_mase.set_index('id')['mase']).mean():.1%}")

    M.save(tm, ARTIFACTS / "lgb_baseline.txt")
    metrics.to_parquet(ARTIFACTS / "metrics_baseline.parquet", index=False)
    baseline_mase.to_parquet(ARTIFACTS / "baseline_mase.parquet", index=False)
    forecast_lgb.to_parquet(ARTIFACTS / "forecast_lgb.parquet", index=False)
    forecast_naive.to_parquet(ARTIFACTS / "forecast_naive.parquet", index=False)
    print(f"  wrote artifacts to {ARTIFACTS}")


if __name__ == "__main__":
    main()
