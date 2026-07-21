"""The Scenario object is the contract between the interface and the model.

Every field here is something the planner can move in the UI. Nothing else in the
model reads raw YAML at run time, so a slider change is guaranteed to reach the
optimiser rather than only the chart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from scipy.stats import norm


@dataclass(frozen=True)
class Scenario:
    # --- demand ---
    annual_units: float
    unit_value_usd: float
    growth_pct: float           # applied to annual_units
    cv_multiplier: float        # scales every region's demand CV

    # --- service policy ---
    target_fill_rate: float     # sets the safety-stock z
    max_lead_time_days: float   # hard feasibility constraint on region -> DC assignment
    review_period_days: float

    # --- cost drivers ---
    holding_rate_pct: float
    obsolescence_rate_pct: float
    inbound_usd_per_unit_km: float
    outbound_usd_per_unit_km: float
    order_cost_usd: float
    stockout_cost_usd_per_unit: float
    customs_cost_usd_per_unit: float
    facility_cost_multiplier: float

    # --- network policy ---
    max_dcs: int
    forced_open: tuple = ()
    excluded: tuple = ()

    # --- transit physics ---
    inbound_transit_km_per_day: float = 800.0
    outbound_transit_km_per_day: float = 650.0
    fixed_days_outbound: float = 1.0
    customs_days_cross_border: float = 1.5

    @property
    def demand_units(self) -> float:
        return self.annual_units * (1 + self.growth_pct / 100.0)

    @property
    def z(self) -> float:
        """Safety factor implied by the target fill rate (normal approximation)."""
        return float(norm.ppf(min(max(self.target_fill_rate, 0.5), 0.9999)))

    @property
    def carrying_rate(self) -> float:
        """Holding + obsolescence, as an annual fraction of unit value."""
        return (self.holding_rate_pct + self.obsolescence_rate_pct) / 100.0

    def with_(self, **kwargs) -> "Scenario":
        return replace(self, **kwargs)


def from_assumptions(a: dict) -> Scenario:
    return Scenario(
        annual_units=a["demand"]["annual_units"],
        unit_value_usd=a["demand"]["unit_value_usd"],
        growth_pct=a["demand"]["growth_pct"],
        cv_multiplier=a["demand"]["cv_multiplier"],
        target_fill_rate=a["inventory"]["target_fill_rate"],
        max_lead_time_days=a["service"]["max_lead_time_days"],
        review_period_days=a["inventory"]["review_period_days"],
        holding_rate_pct=a["inventory"]["holding_rate_pct"],
        obsolescence_rate_pct=a["inventory"]["obsolescence_rate_pct"],
        inbound_usd_per_unit_km=a["transport"]["inbound_usd_per_unit_km"],
        outbound_usd_per_unit_km=a["transport"]["outbound_usd_per_unit_km"],
        order_cost_usd=a["inventory"]["order_cost_usd"],
        stockout_cost_usd_per_unit=a["inventory"]["stockout_cost_usd_per_unit"],
        customs_cost_usd_per_unit=a["transport"]["customs_cost_usd_per_unit"],
        facility_cost_multiplier=1.0,
        max_dcs=a["network"]["max_dcs"],
        inbound_transit_km_per_day=a["transport"]["inbound_transit_km_per_day"],
        outbound_transit_km_per_day=a["transport"]["outbound_transit_km_per_day"],
        fixed_days_outbound=a["transport"]["fixed_days_outbound"],
        customs_days_cross_border=a["transport"]["customs_days_cross_border"],
    )