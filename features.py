"""Feature engineering for the demand forecast.

Everything here is a strictly *lagged* view of the past — nothing computed from
information that wouldn't be available at 7am on the day of the forecast.
That's the discipline the proposal calls out ("all of which are available
before any prediction is made"), and it's what makes MASE on the held-out 28
days a fair measure of how the system will actually behave.

Feature families
----------------
- Lag sales: sales 7, 14, 28 days ago per series. Weekly and monthly seasonality
  arrive here — dominant in grocery, and the strongest single group in every
  M5 post-mortem.
- Rolling stats: mean, std, max over the last 7 and 28 days, again strictly
  lagged. These capture "trend and burstiness" for each SKU-store pair.
- Calendar: day-of-week, month, SNAP flag for the row's state, holiday flag.
- Price: current sell price and its change vs. a 4-week rolling mean, which is
  the proposal's markdown-detection feature.
- Promo: a runtime input the buyer supplies for the forecast horizon (default
  zero for training so the model learns baseline demand, and the buyer can turn
  it on in the interface to see the lift).

Categorical IDs (item, dept, store, state) are passed as pandas categoricals so
LightGBM can use them natively — this is the pooling mechanism, letting one
model span all series without a per-series fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (7, 14, 28)
ROLL_WINDOWS = (7, 28)

CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
FEATURE_COLS_STATIC = [
    *CAT_COLS,
    "wday", "month",
    "snap", "has_event",
    "sell_price", "price_missing",
]


def _feature_col_names() -> list[str]:
    lag_cols = [f"lag_{L}" for L in LAGS]
    roll_cols = []
    for w in ROLL_WINDOWS:
        roll_cols += [f"roll_mean_{w}", f"roll_std_{w}", f"roll_max_{w}"]
    return FEATURE_COLS_STATIC + lag_cols + roll_cols + ["price_chg_4w", "promo"]


FEATURES = _feature_col_names()


def add_features(df: pd.DataFrame, promo_flags: pd.Series | None = None) -> pd.DataFrame:
    """Return `df` with lag, rolling, price-change and promo columns added.

    `promo_flags` is aligned on (id, date) and is 0 during training. The buyer
    supplies it at prediction time to model a planned promotion.
    """
    df = df.sort_values(["id", "date"]).copy()

    # All the lag/rolling ops are per-series. groupby(id) is the expensive step;
    # we do it once and reuse the grouper.
    g = df.groupby("id", sort=False, observed=True)["sales"]
    for L in LAGS:
        df[f"lag_{L}"] = g.shift(L).astype("float32")

    # Roll on the already-shifted series so we never leak the current day into
    # its own features. .shift(1).rolling(w) is the standard way to do this.
    shifted = g.shift(1)
    for w in ROLL_WINDOWS:
        r = shifted.groupby(df["id"], observed=True).rolling(w, min_periods=max(1, w // 2))
        df[f"roll_mean_{w}"] = r.mean().reset_index(0, drop=True).astype("float32")
        df[f"roll_std_{w}"] = r.std().reset_index(0, drop=True).astype("float32")
        df[f"roll_max_{w}"] = r.max().reset_index(0, drop=True).astype("float32")

    # Price change vs. the last four weeks of prices for the same SKU-store.
    # Rolling on price (which is weekly) not on sales.
    gp = df.groupby("id", sort=False, observed=True)["sell_price"]
    price_ma = gp.transform(lambda s: s.rolling(28, min_periods=7).mean())
    df["price_chg_4w"] = ((df["sell_price"] - price_ma) / price_ma).astype("float32")
    df["price_chg_4w"] = df["price_chg_4w"].fillna(0.0)

    # The "promo" feature is a runtime input from the buyer that overrides an
    # inferred training-time flag. During training we synthesize it from the
    # data itself: a promo is a price cut of >8% versus the trailing 4-week
    # mean. That way the model learns a genuine "promo lift" from historical
    # markdowns, and the buyer flipping the flag on for a forecast day pushes
    # the prediction along the same slope.
    inferred_promo = (df["price_chg_4w"] < -0.08).astype(np.int8)
    if promo_flags is not None:
        override = df.set_index(["id", "date"]).index.map(
            promo_flags.reindex(df.set_index(["id", "date"]).index).fillna(-1))
        df["promo"] = np.where(override.values >= 0, override.values,
                               inferred_promo.values).astype(np.int8)
    else:
        df["promo"] = inferred_promo

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    return df


def split_train_test(df: pd.DataFrame, test_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: everything before `test_start` is training."""
    return df[df["date"] < test_start].copy(), df[df["date"] >= test_start].copy()