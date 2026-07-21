"""Pooled LightGBM forecaster over all SKU-store series.

Why one pooled tree ensemble instead of one model per series
------------------------------------------------------------
There are 600 series in the subset we ship, and every one of them has a distinct
seasonality, level, and burstiness. A per-series ARIMA or Prophet on each would
mean 600 fits every retrain — not viable inside an interactive session, and
badly under-powered on the sparser SKUs where you'd be fitting 40 parameters to
100 non-zero days.

Pooling with LightGBM inverts the trade. One gradient-boosted tree fits *all*
series at once, using item_id / dept_id / store_id / state_id as categorical
splits. The tree learns the shared signal (weekly cycle, SNAP months, holiday
lifts, price elasticity) once and specialises to individual series through the
categorical splits. This is the same approach the top-ranked M5 teams took —
we're not being clever, we're using what won.

What this gets us and what it costs
-----------------------------------
Wins: fits in seconds on the 600-series slice, handles the new-SKU cold start
by falling back to the category average automatically (that's what the
categorical split does at low-support leaves), and re-trains cheap enough that
the buyer can re-fit after changing horizon or feature set.

Loses: pooling means the model can under-fit a SKU with idiosyncratic dynamics
that the shared trees smooth over. The proposal names this risk and mitigates
it with SKU-level lag and rolling features, which is why those features are
carrying real weight in the feature-importance chart.

Regression target
-----------------
Direct daily unit sales, one row per (series, day). Predicting 28 days ahead
uses the recursive-forecast approach: predict day t+1, feed that prediction
into the lags for t+2, and so on. Cleaner than fitting 28 separate models and
lets the buyer's promo flag propagate into future lag values naturally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CAT_COLS, FEATURES, LAGS, ROLL_WINDOWS, add_features

TARGET = "sales"


@dataclass
class TrainedModel:
    booster: lgb.Booster
    feature_importances: pd.DataFrame
    train_mase: float
    n_rounds: int
    features: list[str]


def _lgb_params() -> dict:
    return dict(
        objective="tweedie",           # zero-inflated count data; standard for M5
        tweedie_variance_power=1.1,
        metric="rmse",
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbosity=-1,
    )


def train(long_df: pd.DataFrame, test_start: pd.Timestamp, num_boost_round: int = 400
          ) -> TrainedModel:
    """Fit one pooled model on everything before test_start."""
    df = add_features(long_df)
    df = df[df["date"] < test_start].copy()
    # Drop the first ~28 days per series where lag features are still NaN.
    df = df.dropna(subset=[f"lag_{max(LAGS)}"])

    X = df[FEATURES]
    y = df[TARGET].astype("float32")

    train_data = lgb.Dataset(X, label=y, categorical_feature=CAT_COLS, free_raw_data=False)
    booster = lgb.train(_lgb_params(), train_data, num_boost_round=num_boost_round)

    pred_train = booster.predict(X)
    train_mase = mase(y.values, pred_train, y.values)   # in-sample sanity check only

    imp = pd.DataFrame({
        "feature": FEATURES,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    return TrainedModel(booster=booster, feature_importances=imp,
                        train_mase=float(train_mase), n_rounds=num_boost_round,
                        features=FEATURES)


def _recursive_forecast(model: TrainedModel, history: pd.DataFrame,
                        horizon: int, promo_flags: pd.Series | None = None,
                        discount_pct: pd.Series | None = None,
                        ) -> pd.DataFrame:
    """One-step-at-a-time forecast that keeps lag features consistent.

    `history` is the long-format frame up to the last observed date. We append
    empty rows for each future day per series, fill them in day-by-day, and
    reuse `add_features` so lags/rolls are computed the same way at train and
    predict time.
    """
    last_date = history["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    ids = history[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates()
    future = ids.assign(key=1).merge(
        pd.DataFrame({"date": future_dates, "key": 1}), on="key").drop(columns="key")

    # Fill calendar-derived features for future rows from the calendar dataframe
    # so wday/month/snap/has_event are correct.
    cal = history[["date", "wday", "month", "snap", "has_event"]].drop_duplicates("date")
    # The known-future part of the calendar isn't in `history`, so we
    # extend it by reading the raw calendar file if available.
    cal_path = Path(__file__).resolve().parent.parent / "data" / "calendar_slim.parquet"
    if cal_path.exists():
        cal_ext = pd.read_parquet(cal_path)
        future = future.merge(cal_ext, on="date", how="left")
    else:
        # Fall back to derived values so tests can run without the parquet cache.
        future["wday"] = future["date"].dt.dayofweek.astype(np.int16) + 1
        future["month"] = future["date"].dt.month.astype(np.int16)
        future["snap"] = 0
        future["has_event"] = 0

    # Carry forward the last known price, then apply the buyer's planned
    # discount on top for the days it applies.
    last_price = history.sort_values("date").groupby("id").tail(1)[
        ["id", "sell_price", "price_missing"]]
    future = future.merge(last_price, on="id", how="left")
    if discount_pct is not None:
        d = future.set_index(["id", "date"]).index.map(
            discount_pct.reindex(future.set_index(["id", "date"]).index).fillna(0))
        future["sell_price"] = future["sell_price"] * (1 - np.asarray(d).astype(float))
    future["sales"] = np.nan

    combined = pd.concat([
        history[["id", "date", "item_id", "dept_id", "cat_id", "store_id",
                 "state_id", "wday", "month", "snap", "has_event",
                 "sell_price", "price_missing", "sales"]],
        future[["id", "date", "item_id", "dept_id", "cat_id", "store_id",
                "state_id", "wday", "month", "snap", "has_event",
                "sell_price", "price_missing", "sales"]],
    ], ignore_index=True)

    # Only the last max(LAG, ROLL) + horizon days per series affect future
    # features. Slicing the tail cuts feature engineering ~15× on this dataset,
    # which is what gets the buyer's re-run under a second.
    tail_days = max(max(LAGS), max(ROLL_WINDOWS)) + horizon + 1
    cutoff = last_date - pd.Timedelta(days=tail_days)
    tail_mask = combined["date"] > cutoff
    tail = combined.loc[tail_mask].copy()
    rest = combined.loc[~tail_mask]

    # add_features sorts internally, so writing predictions back through a
    # positional mask on the unsorted `tail` scrambled the assignment and made
    # runtime overrides look inert. Route through a MultiIndex instead.
    tail = tail.sort_values(["id", "date"]).reset_index(drop=True)
    for step_date in future_dates:
        featured = add_features(tail, promo_flags)
        step_rows = featured.loc[featured["date"] == step_date, ["id"] + FEATURES].copy()
        preds = np.clip(model.booster.predict(step_rows[FEATURES]), 0.0, None)
        pred_by_id = pd.Series(preds, index=step_rows["id"].values)
        mask = tail["date"] == step_date
        tail.loc[mask, "sales"] = tail.loc[mask, "id"].map(pred_by_id).astype("float32").values

    combined = pd.concat([rest, tail], ignore_index=True)

    out = combined[combined["date"].isin(future_dates)][["id", "date", "sales"]].copy()
    out = out.rename(columns={"sales": "forecast"})
    out["forecast"] = out["forecast"].astype("float32")
    return out


def predict(model: TrainedModel, long_df: pd.DataFrame, test_start: pd.Timestamp,
            horizon: int = 28, promo_flags: pd.Series | None = None,
            discount_pct: pd.Series | None = None) -> pd.DataFrame:
    """Forecast `horizon` days starting at `test_start`.

    `promo_flags` (0/1 per (id, date)) and `discount_pct` (0..1 per (id, date))
    let the buyer stress a planned markdown into the forecast. The proposal
    separates these because they mean different things: `promo` is the
    marketing flag ("this is on our weekly circular"), `discount_pct` is the
    actual price reduction. Both are runtime inputs from the interface.
    """
    history = long_df[long_df["date"] < test_start].copy()
    return _recursive_forecast(model, history, horizon, promo_flags, discount_pct)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_history: np.ndarray,
         seasonality: int = 7) -> float:
    """Mean Absolute Scaled Error, the M5 metric.

    Denominator is the mean absolute seasonal-naive error on the training
    history; a MASE < 1 beats the naive baseline. We use seasonality=7 for
    daily retail data.
    """
    y_history = np.asarray(y_history, dtype=float)
    if len(y_history) <= seasonality:
        return float("nan")
    scale = np.abs(y_history[seasonality:] - y_history[:-seasonality]).mean()
    if scale == 0:
        return float("nan")
    return float(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float)).mean() / scale)


def evaluate(model: TrainedModel, long_df: pd.DataFrame, test_start: pd.Timestamp,
             horizon: int = 28) -> pd.DataFrame:
    """MASE per (store, id) on the held-out window."""
    forecast = predict(model, long_df, test_start, horizon)
    actual = long_df[long_df["date"] >= test_start][["id", "date", "sales", "store_id"]]
    joined = forecast.merge(actual, on=["id", "date"], how="inner")
    joined["store_id"] = joined["store_id"].astype(str)

    hist = long_df[long_df["date"] < test_start][["id", "sales"]]
    hist = hist.sort_values("id")

    rows = []
    for sid, g in joined.groupby("id", observed=True):
        h = hist.loc[hist["id"] == sid, "sales"].values
        rows.append({
            "id": sid,
            "store_id": g["store_id"].iloc[0],
            "mase": mase(g["sales"].values, g["forecast"].values, h),
            "wape": (np.abs(g["forecast"] - g["sales"]).sum()
                     / max(g["sales"].sum(), 1e-6)),
            "bias": float((g["forecast"] - g["sales"]).mean()),
            "actual_total": float(g["sales"].sum()),
            "pred_total": float(g["forecast"].sum()),
        })
    return pd.DataFrame(rows)


def seasonal_naive_forecast(long_df: pd.DataFrame, test_start: pd.Timestamp,
                            horizon: int = 28) -> pd.DataFrame:
    """The baseline the proposal names: repeat the last 7 days for each future week.

    This is what a fixed-reorder-point policy is implicitly using: last week's
    demand is next week's demand. Our tool has to beat it or it isn't earning
    its place.
    """
    hist = long_df[long_df["date"] < test_start]
    last_dt = hist["date"].max()
    last_week = hist[hist["date"] > last_dt - pd.Timedelta(days=7)][["id", "date", "sales"]].copy()
    last_week["wday_off"] = (last_dt - last_week["date"]).dt.days
    rows = []
    for step in range(horizon):
        d = last_dt + pd.Timedelta(days=step + 1)
        off = 6 - (step % 7)
        pattern = last_week[last_week["wday_off"] == off][["id", "sales"]].copy()
        pattern["date"] = d
        rows.append(pattern.rename(columns={"sales": "forecast"}))
    return pd.concat(rows, ignore_index=True)[["id", "date", "forecast"]]


def save(model: TrainedModel, path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(path))
    model.feature_importances.to_csv(path.with_suffix(".importances.csv"), index=False)


def load(path: Path) -> TrainedModel:
    path = Path(path)
    booster = lgb.Booster(model_file=str(path))
    imp = pd.read_csv(path.with_suffix(".importances.csv"))
    return TrainedModel(booster=booster, feature_importances=imp,
                        train_mase=float("nan"),
                        n_rounds=booster.current_iteration(), features=FEATURES)