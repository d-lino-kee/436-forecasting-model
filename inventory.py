"""Turns a 28-day demand forecast into the artefact the buyer actually acts on:
a recommended reorder quantity and a cost projection for the coming month.

The math is the standard newsvendor/reorder-up-to policy, kept intentionally
transparent so the buyer can argue with each line. The interface exposes every
knob so a category buyer's judgement — "I know this promo will lift twice as
hard as your model says" — can drop straight in and change the recommendation.

Reorder quantity, in the buyer's terms
--------------------------------------
    reorder_qty = max(
        0,
        expected_demand_over_(lead_time + review_period) + safety_stock - on_hand
    )
Clamped by supplier minimums when they matter.

Safety stock scales with forecast uncertainty. We approximate it as
    z(service_level) * sigma_forecast * sqrt(lead_time)
using the standard deviation of the 28-day forecast around its mean as the
uncertainty proxy — noisier forecasts should carry more buffer. Direction of
the effect is what matters here; buyers who want a per-SKU service level get
one dial to set it.

Cost projection
---------------
The proposal names two cost components:
    holding_cost   = (avg_projected_inventory) * unit_price * holding_rate * horizon_days
    stockout_cost  = expected_units_short_over_horizon * stockout_multiplier * unit_price
Where expected_units_short comes from simulating on-hand day-by-day under the
recommendation. Both are compared to the seasonal-naive baseline so the buyer
sees whether following the model actually saves money on their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class InventoryPolicy:
    service_level: float = 0.95        # target fill rate -> safety-stock z
    holding_rate_daily: float = 0.0007  # 0.07%/day (~25%/yr)
    stockout_multiplier: float = 1.5   # stockout cost = sell_price × this
    lead_time_days: int = 7
    review_period_days: int = 7        # how often the buyer places orders
    supplier_min_qty: int = 0          # 0 = no minimum

    @property
    def z(self) -> float:
        return float(norm.ppf(min(max(self.service_level, 0.5), 0.9999)))


def recommend(forecast: pd.DataFrame, on_hand: pd.Series, sell_price: pd.Series,
              policy: InventoryPolicy) -> pd.DataFrame:
    """Compute the reorder recommendation for every SKU with a forecast.

    Parameters
    ----------
    forecast : long-format (id, date, forecast) over the horizon.
    on_hand  : current inventory in units, indexed by id. The proposal's manual
               buyer input; defaults to 0 when missing.
    sell_price : unit sell price by id, used for costing.
    """
    horizon = forecast["date"].nunique()
    lead = policy.lead_time_days
    cover = lead + policy.review_period_days

    # Aggregate forecast to the coverage window.
    agg = (forecast.sort_values(["id", "date"]).groupby("id", observed=True)["forecast"]
           .agg(demand_horizon="sum",
                sigma_horizon="std",
                demand_cover=lambda s: float(s.iloc[:cover].sum())))
    agg = agg.reset_index()
    agg["on_hand"] = agg["id"].map(on_hand).fillna(0).astype(float)
    agg["sell_price"] = agg["id"].map(sell_price).fillna(0).astype(float)

    # Safety stock scales with forecast noise and lead-time exposure.
    agg["sigma_horizon"] = agg["sigma_horizon"].fillna(0)
    agg["safety_stock"] = policy.z * agg["sigma_horizon"] * np.sqrt(lead)

    raw_qty = agg["demand_cover"] + agg["safety_stock"] - agg["on_hand"]
    agg["reorder_qty"] = np.maximum(0, raw_qty.round()).astype(int)
    if policy.supplier_min_qty > 0:
        below = (agg["reorder_qty"] > 0) & (agg["reorder_qty"] < policy.supplier_min_qty)
        agg.loc[below, "reorder_qty"] = policy.supplier_min_qty

    agg = agg.rename(columns={"demand_horizon": "expected_demand_28d",
                              "demand_cover": "expected_demand_cover"})
    return agg


def project_costs(forecast: pd.DataFrame, actual: pd.DataFrame | None,
                  reorder: pd.DataFrame, policy: InventoryPolicy,
                  horizon: int = 28) -> pd.DataFrame:
    """Simulate day-by-day inventory and cost, using actuals if available.

    When `actual` is provided (we're on a historical window), the simulation
    reflects what would have happened by following the recommendation. When
    `actual` is None (a live forecast run), the simulation uses the forecast
    itself as demand, so numbers should be read as "if the forecast is right".
    """
    horizon_forecast = forecast[["id", "date", "forecast"]].copy()
    if actual is None:
        horizon_forecast["actual"] = horizon_forecast["forecast"]
    else:
        horizon_forecast = horizon_forecast.merge(
            actual[["id", "date", "sales"]].rename(columns={"sales": "actual"}),
            on=["id", "date"], how="left")
        horizon_forecast["actual"] = horizon_forecast["actual"].fillna(0)

    rec = reorder.set_index("id")
    lead = policy.lead_time_days
    review = policy.review_period_days

    rows = []
    for sid, g in horizon_forecast.sort_values(["id", "date"]).groupby("id", observed=True):
        if sid not in rec.index:
            continue
        row = rec.loc[sid]
        price = row["sell_price"]
        # Periodic-review, order-up-to policy: every `review` days the buyer
        # re-runs the tool and places another order sized against the forecast
        # for the next lead+review window. Simulating this as if the buyer
        # never returns for 28 days would overstate stockouts by design.
        on_hand = float(row["on_hand"])
        safety = float(row["safety_stock"])
        pending = 0.0                # units on order but not yet arrived
        arrival_day = -1
        # Place the first order today (as the tool recommends).
        pending += row["reorder_qty"]; arrival_day = lead
        cum_short = 0.0
        cum_hold_units_days = 0.0
        forecast_by_day = g["forecast"].values
        actual_by_day = g["actual"].values
        for t in range(len(g)):
            if t == arrival_day:
                on_hand += pending; pending = 0.0
            # weekly re-order
            if t > 0 and t % review == 0:
                target = float(forecast_by_day[t:t + lead + review].sum()) + safety
                order_now = max(0.0, target - on_hand - pending)
                pending += order_now
                arrival_day = t + lead
            sold = min(on_hand, actual_by_day[t])
            cum_short += actual_by_day[t] - sold
            on_hand -= sold
            cum_hold_units_days += on_hand
        rows.append({
            "id": sid,
            "avg_inventory": cum_hold_units_days / max(horizon, 1),
            "units_short": cum_short,
            "holding_cost": (cum_hold_units_days * price * policy.holding_rate_daily),
            "stockout_cost": cum_short * price * policy.stockout_multiplier,
            "end_inventory": on_hand,
        })
    out = pd.DataFrame(rows)
    out["total_cost"] = out["holding_cost"] + out["stockout_cost"]
    return out


def compare_policies(forecast_model: pd.DataFrame, forecast_baseline: pd.DataFrame,
                     actual: pd.DataFrame | None, on_hand: pd.Series,
                     sell_price: pd.Series, policy: InventoryPolicy,
                     horizon: int = 28) -> dict[str, pd.DataFrame]:
    """Run recommend() + project_costs() under both forecasts. This is the
    'is the model saving me money' chart the proposal names."""
    rec_m = recommend(forecast_model, on_hand, sell_price, policy)
    rec_b = recommend(forecast_baseline, on_hand, sell_price, policy)
    cost_m = project_costs(forecast_model, actual, rec_m, policy, horizon)
    cost_b = project_costs(forecast_baseline, actual, rec_b, policy, horizon)
    return {"model_recommend": rec_m, "model_cost": cost_m,
            "baseline_recommend": rec_b, "baseline_cost": cost_b}