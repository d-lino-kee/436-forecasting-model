"""Chooses the network.

Why exhaustive enumeration rather than a MILP: with the hub fixed and nine
candidate regional sites, there are at most a few hundred legal configurations,
so we can evaluate every one of them exactly. That matters because the safety-stock
term is non-linear in the assignment (the square-root pooling law), which a linear
program can only approximate. Enumeration keeps the pooling effect exact, returns a
provably optimal answer for the candidate set, and runs in well under a second —
fast enough to re-solve on every slider move, which is the whole point of the tool.

If the candidate list ever grows past ~20 sites this should be swapped for a
Lagrangian or MILP formulation with a linearised inventory term.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import pandas as pd

from .data_loader import NetworkData
from .model import NetworkResult, evaluate, score
from .scenario import Scenario


def candidate_configurations(nd: NetworkData, sc: Scenario) -> list[tuple]:
    pool = [d for d in nd.dcs["dc_id"]
            if d != nd.hub_id and d not in sc.excluded]
    forced = [d for d in sc.forced_open if d != nd.hub_id and d in pool]
    optional = [d for d in pool if d not in forced]

    configs = []
    # max_dcs counts the hub, which is always open.
    for extra in range(0, max(sc.max_dcs - 1 - len(forced), 0) + 1):
        for combo in combinations(optional, extra):
            configs.append(tuple(sorted(forced + list(combo))))
    return configs


def solve(nd: NetworkData, sc: Scenario) -> tuple[NetworkResult, pd.DataFrame]:
    """Returns the best network and a frontier table of every configuration priced.

    The frontier is priced with the fast numpy path; only the winner is expanded
    into a full labelled result.
    """
    rows, best_s = [], None
    for cfg in candidate_configurations(nd, sc):
        s = score(nd, sc, cfg)
        if best_s is None or s["total_cost"] < best_s["total_cost"]:
            best_s = s
        rows.append({
            "config": " + ".join(s["open_dcs"]),
            "n_dcs": len(s["open_dcs"]),
            "total_cost_usd": s["total_cost"],
            "cost_per_unit_usd": s["total_cost"] / max(sc.demand_units, 1e-9),
            "lead_time_days": s["lead_time_days"],
            "pct_on_time": s["pct_on_time"],
            "safety_stock_units": s["safety_stock_units"],
            **s["costs"],
        })
    frontier = pd.DataFrame(rows).sort_values("total_cost_usd").reset_index(drop=True)
    return evaluate(nd, sc, best_s["open_dcs"]), frontier


def best_by_size(nd: NetworkData, sc: Scenario) -> pd.DataFrame:
    """Cheapest network for each node count — the 'how many DCs' curve."""
    _, frontier = solve(nd, sc)
    return (frontier.sort_values("total_cost_usd")
            .groupby("n_dcs", as_index=False).first()
            .sort_values("n_dcs").reset_index(drop=True))


def named_strategies(nd: NetworkData, sc: Scenario) -> dict[str, Sequence[str]]:
    """The three postures the proposal asks us to compare, plus the optimiser's answer."""
    return {
        "Centralised (hub only)": (),
        "Hybrid (hub + 2 regional)": ("CMH", "ONT"),
        "Regional (hub + 4 regional)": ("CMH", "ONT", "EWR", "YYZ"),
    }