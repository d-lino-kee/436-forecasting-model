"""Slice the full M5 dataset down to a size the IDSS can train live on.

The proposal describes a system that a category buyer runs weekly, adjusting
parameters and re-running. That interaction pattern only works if a full retrain
finishes in seconds, not minutes. A pooled LightGBM on all 30,490 series is a
multi-minute job; we subset to a defensible sample so the interactivity is real.

What we keep and why
--------------------
- One store per state (CA_1, TX_1, WI_1). The proposal explicitly names
  per-state performance monitoring, so preserving state coverage matters more
  than covering more stores in one state.
- Top 200 SKUs per store by total unit volume. Low-volume SKUs are mostly zero
  in M5 and dominate the row count without adding forecasting signal; pooling
  the high-volume ones is where the LightGBM approach earns its keep. Buyers
  care more about the SKUs that move, and the SKUs that move are where a
  stockout is expensive.
- Last two years of the training window (~730 days). The proposal calls for a
  rolling two-year window, and one full cycle each of holidays and SNAP months
  is enough to fit their calendar effects.
- The last 28 days of each series are held out as a test set, matching the M5
  competition protocol so MASE numbers are comparable to published benchmarks.

Outputs one long-format parquet: (id, date, sales, sell_price, calendar_fields).
Long format is what LightGBM wants and it collapses to ~200 MB in RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("/mnt/user-data/uploads")
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

STORES = ["CA_1", "TX_1", "WI_1"]
TOP_N_PER_STORE = 200
LOOKBACK_DAYS = 730     # ~2 years, matches proposal's rolling window
TEST_DAYS = 28          # M5 evaluation horizon


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading calendar and prices...")
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(raw_dir / "sell_prices.csv")

    print("Loading sales (full file, ~120MB)...")
    sales = pd.read_csv(raw_dir / "sales_train_validation.csv")

    sales = sales[sales["store_id"].isin(STORES)].copy()

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_cols = day_cols[-LOOKBACK_DAYS:]
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # Pick the SKUs that actually move. Ranking by total units over the lookback
    # window (rather than lifetime) keeps SKUs that ramped up recently and drops
    # ones that have been discontinued.
    totals = sales[day_cols].sum(axis=1)
    sales = sales.assign(_total=totals.values)
    sales = (sales.sort_values(["store_id", "_total"], ascending=[True, False])
                  .groupby("store_id").head(TOP_N_PER_STORE).drop(columns="_total"))
    print(f"Kept {len(sales)} series across {sales['store_id'].nunique()} stores.")

    # Wide -> long. Dominates memory here (rows × days), so we downcast on the way in.
    print("Reshaping to long format...")
    long = sales.melt(id_vars=meta_cols, value_vars=day_cols,
                      var_name="d", value_name="sales")
    long["sales"] = long["sales"].astype(np.int16)

    # Attach calendar/date, then price.
    cal_slim = calendar[["d", "date", "wm_yr_wk", "wday", "month", "year",
                         "event_name_1", "event_type_1",
                         "snap_CA", "snap_TX", "snap_WI"]]
    long = long.merge(cal_slim, on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # The proposal's "SNAP feature per state" — collapse into one column relative
    # to each row's state so the model reads a single feature at prediction time.
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    long["snap"] = 0
    for st, col in snap_map.items():
        m = long["state_id"] == st
        long.loc[m, "snap"] = long.loc[m, col].astype(np.int8)
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    long["has_event"] = long["event_name_1"].notna().astype(np.int8)
    long["date"] = pd.to_datetime(long["date"])

    # NaN prices mean the SKU wasn't listed that week: known M5 quirk, real signal.
    long["price_missing"] = long["sell_price"].isna().astype(np.int8)
    long["sell_price"] = long["sell_price"].astype("float32")

    for c in ("wday", "month", "year"):
        long[c] = long[c].astype(np.int16)

    out_path = out_dir / "m5_long.parquet"
    long.to_parquet(out_path, index=False)

    # Small metadata file so the app can display coverage without re-parsing.
    meta = {
        "stores": STORES,
        "n_series": int(long["id"].nunique()),
        "n_rows": int(len(long)),
        "date_min": long["date"].min().strftime("%Y-%m-%d"),
        "date_max": long["date"].max().strftime("%Y-%m-%d"),
        "test_start": (long["date"].max() - pd.Timedelta(days=TEST_DAYS - 1)).strftime("%Y-%m-%d"),
        "test_days": TEST_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "top_n_per_store": TOP_N_PER_STORE,
    }
    import json
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote", out_path)
    print("Meta:", meta)
    return out_path


if __name__ == "__main__":
    prepare()