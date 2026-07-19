"""Turn the raw M5 competition CSVs into the single long-format table the rest
of the pipeline reads.

Run once, before `python -m src.train`. Reads the three raw M5 files from
`data/` and writes:

    data/m5_long.parquet   one row per (SKU-store series, day) with sales,
                           calendar, and price columns already joined
    data/meta.json         the test-window boundary and dataset summary the
                           interface and trainer both read

The subset matches the proposal's monitoring plan:
- 3 stores (one per state) to preserve the SNAP scheduling differences
- top-200 SKUs per store by volume over the window (the SKUs that actually move)
- the last 2 years of history (the rolling window the proposal specifies)

The last 28 days are held out as the test window; `test_start` in meta.json
marks where it begins. Everything downstream splits on that date.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

STORES = ["CA_1", "TX_1", "WI_1"]     # one per state; keeps SNAP differences
TOP_N_PER_STORE = 200                 # the SKUs that actually move
WINDOW_DAYS = 730                     # rolling 2-year window
HORIZON = 28                          # held-out test window length

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
SNAP_BY_STATE = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def main() -> None:
    if not (DATA / "sales_train_validation.csv").exists():
        raise SystemExit(
            "Raw M5 CSVs not found in data/. Download sales_train_validation.csv, "
            "calendar.csv, and sell_prices.csv from the Zenodo mirror "
            "(https://zenodo.org/records/12636070) into the data/ folder first.")

    print("Reading raw M5 files...")
    calendar = pd.read_csv(DATA / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(DATA / "sell_prices.csv")
    sales = pd.read_csv(DATA / "sales_train_validation.csv")

    # --- Subset to the three in-scope stores ------------------------------- #
    sales = sales[sales["store_id"].isin(STORES)].copy()

    # --- Restrict to the last WINDOW_DAYS of day-columns ------------------- #
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    last_d = max(int(c[2:]) for c in d_cols)
    window_d = [f"d_{i}" for i in range(last_d - WINDOW_DAYS + 1, last_d + 1)
                if f"d_{i}" in set(d_cols)]

    # --- Top-N SKUs per store by volume over the window -------------------- #
    volume = sales.set_index(ID_COLS)[window_d].sum(axis=1).reset_index(name="vol")
    top = (volume.sort_values("vol", ascending=False)
           .groupby("store_id", sort=False).head(TOP_N_PER_STORE))
    sales = sales[sales["id"].isin(set(top["id"]))]
    print(f"  kept {sales['id'].nunique()} series across {len(STORES)} stores")

    # --- Wide -> long ------------------------------------------------------ #
    long = sales.melt(id_vars=ID_COLS, value_vars=window_d,
                      var_name="d", value_name="sales")
    long["id"] = long["id"].str.replace("_validation", "", regex=False)

    # --- Join the calendar (date, week key, SNAP, events) ------------------ #
    cal_cols = ["d", "date", "wm_yr_wk", "event_name_1",
                "snap_CA", "snap_TX", "snap_WI"]
    long = long.merge(calendar[cal_cols], on="d", how="left")

    # --- Join the weekly sell price ---------------------------------------- #
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["price_missing"] = long["sell_price"].isna().astype("int8")
    # A SKU with no listed price that week isn't necessarily unsold; carry the
    # last known price forward/back within the series, then fall back to the
    # global median so the rolling price features never see a NaN.
    long["sell_price"] = (long.sort_values("date")
                          .groupby("id")["sell_price"].transform(lambda s: s.ffill().bfill()))
    long["sell_price"] = long["sell_price"].fillna(long["sell_price"].median())

    # --- Calendar-derived features (kept consistent with model's fallback) - #
    # model._recursive_forecast derives wday/month the same way for future
    # rows, so we compute them from the date here too rather than from the M5
    # `wday` column (which is Sat-indexed).
    long["wday"] = long["date"].dt.dayofweek.astype("int16") + 1
    long["month"] = long["date"].dt.month.astype("int16")

    long["snap"] = 0
    for state, col in SNAP_BY_STATE.items():
        mask = long["state_id"] == state
        long.loc[mask, "snap"] = long.loc[mask, col]
    long["snap"] = long["snap"].astype("int8")
    long["has_event"] = long["event_name_1"].notna().astype("int8")
    long["sales"] = long["sales"].astype("float32")

    long = long[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                 "date", "sales", "wday", "month", "snap", "has_event",
                 "sell_price", "price_missing"]]
    long = long.sort_values(["id", "date"]).reset_index(drop=True)

    # --- Test window and metadata ------------------------------------------ #
    max_date = long["date"].max()
    test_start = max_date - pd.Timedelta(days=HORIZON - 1)

    DATA.mkdir(exist_ok=True)
    long.to_parquet(DATA / "m5_long.parquet", index=False)

    meta = {
        "test_start": str(test_start.date()),
        "test_days": HORIZON,
        "stores": STORES,
        "n_series": int(long["id"].nunique()),
        "date_min": str(long["date"].min().date()),
        "date_max": str(max_date.date()),
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"  wrote {len(long):,} rows to {DATA / 'm5_long.parquet'}")
    print(f"  test window: {test_start.date()} .. {max_date.date()} ({HORIZON} days)")
    print(f"  meta.json: {meta}")


if __name__ == "__main__":
    main()
