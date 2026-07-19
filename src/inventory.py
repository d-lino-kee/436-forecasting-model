"""Reorder policy and day-by-day cost simulation.

This is the half of the tool that turns a demand *forecast* into the decision
the buyer actually submits: how many units to order this week. The model says
what demand will be; this module says what to do about it under the buyer's
current beliefs about service level, lead time, and the cost of a stockout.

The policy is a periodic-review order-up-to (R, S) policy — the standard model
for a buyer who places orders on a fixed cadence (weekly) rather than
continuously:

    order-up-to level  S = expected demand over (lead time + review period)
                           + safety stock
    safety stock         = z(service_level) · σ_daily · √(lead time + review)
    reorder quantity     = max(0, S − on_hand)

`project_costs` then simulates that order against the *actual* held-out demand,
day by day, so the interface can show whether the model's order beats the
seasonal-naive baseline in real dollars — the "is this saving me money" check
the proposal commits to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class InventoryPolicy:
    """The buyer's current beliefs, as the sidebar controls set them."""

    service_level: float = 0.95          # target fill; sets the safety-stock z
    holding_rate_daily: float = 0.0007   # per-day holding cost as fraction of unit value
    stockout_multiplier: float = 1.5     # cost per unit short = sell_price × this
    lead_time_days: int = 7              # supplier lead time
    review_period_days: int = 7          # reorder cadence
    supplier_min_qty: int = 0            # minimum order the supplier accepts

    @property
    def z(self) -> float:
        """Safety-stock multiplier for the target service level."""
        return float(norm.ppf(min(max(self.service_level, 0.5), 0.9999)))

    @property
    def protection_days(self) -> int:
        """Interval the order must cover: lead time plus one review period."""
        return self.lead_time_days + self.review_period_days


def recommend(forecast: pd.DataFrame, on_hand: pd.Series, prices: pd.Series,
              policy: InventoryPolicy) -> pd.DataFrame:
    """Order-up-to recommendation per SKU from a demand forecast.

    `forecast` has columns id, date, forecast (one row per horizon day).
    `on_hand` and `prices` are Series indexed by id. Returns one row per id
    with the columns the order sheet reads, including `sell_price` (this
    function owns that column; `project_costs` deliberately does not, so the
    two can be merged on id without a collision).
    """
    g = forecast.groupby("id", observed=True)["forecast"]
    daily_mean = g.mean()
    horizon_demand = g.sum()
    # Poisson-style floor so smooth forecasts still carry some safety stock:
    # for count demand σ ≈ √mean is a sane lower bound on day-to-day variation.
    daily_std = g.std().fillna(0.0)
    sigma_daily = np.maximum(daily_std, np.sqrt(daily_mean.clip(lower=0)))

    ids = daily_mean.index
    oh = on_hand.reindex(ids).fillna(0).astype(float)
    price_fallback = float(prices.median()) if len(prices) else 3.0
    px = prices.reindex(ids).fillna(price_fallback)

    safety = policy.z * sigma_daily * np.sqrt(policy.protection_days)
    # Order-up-to level S: cover expected demand over the protection interval
    # plus safety stock. This week's order brings on-hand up to S.
    order_up_to = daily_mean * policy.protection_days + safety
    reorder = (order_up_to - oh).clip(lower=0).round()
    if policy.supplier_min_qty > 0:
        bump = (reorder > 0) & (reorder < policy.supplier_min_qty)
        reorder = reorder.mask(bump, policy.supplier_min_qty)

    return pd.DataFrame({
        "id": ids,
        "on_hand": oh.astype(int).to_numpy(),
        "expected_demand_28d": horizon_demand.to_numpy(),
        "safety_stock": safety.round().to_numpy(),
        "reorder_qty": reorder.astype(int).to_numpy(),
        "order_up_to": order_up_to.to_numpy(),   # target level for the cost sim
        "sell_price": px.to_numpy(),
    }).reset_index(drop=True)


def project_costs(forecast: pd.DataFrame, actual: pd.DataFrame,
                  recommend_df: pd.DataFrame, policy: InventoryPolicy,
                  horizon: int) -> pd.DataFrame:
    """Simulate a periodic-review (R, S) policy against actual demand.

    The buyer doesn't place one order and walk away — they reorder every
    `review_period_days`. So over the horizon we place the recommended order
    this week, then keep topping inventory back up to the order-up-to level S
    on each subsequent review day, with each order arriving after the lead
    time. Actual demand is met from on-hand stock; unmet demand is a lost sale
    (units short); leftover stock accrues holding cost.

    This is what makes the *accurate* forecast win: whichever forecast sizes S
    closest to real demand wastes the least on both holding and stockouts.
    Returns one row per id with units_short, holding_cost, stockout_cost, total.
    """
    rec = recommend_df.set_index("id")
    lead = policy.lead_time_days
    review = max(policy.review_period_days, 1)
    holding_rate = policy.holding_rate_daily
    stock_mult = policy.stockout_multiplier

    rows = []
    for sid, g in actual.sort_values("date").groupby("id", observed=True):
        if sid not in rec.index:
            continue
        r = rec.loc[sid]
        inv = float(r["on_hand"])
        target_S = float(r["order_up_to"])
        first_order = float(r["reorder_qty"])
        price = float(r["sell_price"])
        demands = g["sales"].to_numpy()[:horizon]

        pipeline: dict[int, float] = {}
        units_short = 0.0
        holding_cost = 0.0
        for day, demand in enumerate(demands):
            if day in pipeline:                       # replenishment arrives
                inv += pipeline.pop(day)
            if day % review == 0:                     # review day: place an order
                if day == 0:
                    qty = first_order
                else:
                    position = inv + sum(pipeline.values())
                    qty = max(0.0, round(target_S - position))
                if qty > 0:
                    pipeline[day + lead] = pipeline.get(day + lead, 0.0) + qty
            inv -= float(demand)
            if inv < 0:
                units_short += -inv
                inv = 0.0
            holding_cost += holding_rate * price * inv

        stockout_cost = units_short * price * stock_mult
        rows.append({"id": sid, "units_short": units_short,
                     "holding_cost": holding_cost, "stockout_cost": stockout_cost,
                     "total_cost": holding_cost + stockout_cost})

    return pd.DataFrame(rows, columns=["id", "units_short", "holding_cost",
                                       "stockout_cost", "total_cost"])


def compare_policies(forecast: pd.DataFrame, baseline_forecast: pd.DataFrame,
                     actual: pd.DataFrame, on_hand: pd.Series, prices: pd.Series,
                     policy: InventoryPolicy, horizon: int) -> dict:
    """Run recommend + cost simulation for both the model and the baseline.

    Returns the four frames the interface reads: model_recommend, model_cost,
    baseline_recommend, baseline_cost. Same policy and same actuals for both, so
    the only difference is the forecast driving the order.
    """
    model_recommend = recommend(forecast, on_hand, prices, policy)
    model_cost = project_costs(forecast, actual, model_recommend, policy, horizon)
    baseline_recommend = recommend(baseline_forecast, on_hand, prices, policy)
    baseline_cost = project_costs(baseline_forecast, actual, baseline_recommend,
                                  policy, horizon)
    return {
        "model_recommend": model_recommend,
        "model_cost": model_cost,
        "baseline_recommend": baseline_recommend,
        "baseline_cost": baseline_cost,
    }
