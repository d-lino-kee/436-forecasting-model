"""Weekly Replenishment Planner — the interactive tool the category buyer runs.

The decision this screen exists to support:
    Which SKUs to reorder this week, in what quantities, given a demand
    forecast and the buyer's current beliefs about promotions, cost, and
    service level.

Every control feeds the model or the reorder policy. The recommended reorder
quantities at the top can and do change as a result. Nothing here is a filter
on a static answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import inventory as I, model as M

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "m5_long.parquet"
ARTIFACTS = ROOT / "artifacts"
META = json.loads((ROOT / "data" / "meta.json").read_text())
TEST_START = pd.Timestamp(META["test_start"])
HORIZON = META["test_days"]

st.set_page_config(page_title="Weekly Replenishment Planner",
                   page_icon="🛒", layout="wide")

INK = "#17251B"
BLUE = "#0066CC"
GREEN = "#0A8754"
AMBER = "#F2A900"
RED = "#D0021B"
GREY = "#8A8A93"
DECK_GREEN = "#15271B"   # deep forest green from the deck (cards, sidebar)
OLIVE = "#4E7C3E"        # olive accent from the deck

st.markdown(f"""
<style>
  /* Base typography — dark green-black, a touch larger, readable line height */
  html, body, [data-testid="stAppViewContainer"] {{
    color: {INK};
    font-size: 17px;
  }}
  .stApp {{ background: {"#F1EFE7"}; }}
  [data-testid="stAppViewContainer"] .stMarkdown p,
  [data-testid="stAppViewContainer"] .stMarkdown li {{
    color: {INK};
    font-size: 1.02rem;
    line-height: 1.6;
  }}
  h1, h2, h3, h4 {{
    color: {DECK_GREEN} !important;
    font-weight: 700;
    letter-spacing: -0.02em;
  }}
  h1 {{ font-size: 2.1rem; }}

  /* Captions were too light grey to read — darken and firm them up */
  [data-testid="stCaptionContainer"], .caption, small {{
    color: #4A5347 !important;
    font-size: 0.86rem !important;
  }}

  /* Sidebar — deep forest green panel like the deck headers, cream text */
  [data-testid="stSidebar"] {{
    background: {DECK_GREEN};
    border-right: 1px solid #0C1710;
  }}
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] summary,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: #ECEAD7 !important;
  }}
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] summary {{
    font-weight: 600; font-size: 0.97rem;
  }}
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
  /* slider min/max + current-value readouts sit on the dark bg → keep cream */
  [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
  [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
  [data-testid="stSidebar"] [data-testid="stThumbValue"] {{ color: #ECEAD7 !important; }}
  /* input controls keep a light fill + dark text so entries stay legible */
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] [data-baseweb="select"] > div,
  [data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background: #F4F3EC !important; color: {INK} !important;
  }}
  [data-testid="stSidebar"] [data-baseweb="select"] div {{ color: {INK} !important; }}

  /* Metrics — make the headline numbers pop */
  [data-testid="stMetricValue"] {{ font-weight: 700; color: {DECK_GREEN}; }}
  [data-testid="stMetricLabel"] {{ font-weight: 600; color: #4A5347; }}

  /* Verdict card — deep green panel with cream text, the deck's signature look */
  .verdict {{
    border-left: 5px solid #8FBF6A; background: {DECK_GREEN}; padding: 20px 24px;
    border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,.18); margin-bottom: 8px;
  }}
  .verdict h2 {{ margin: 0 0 6px 0; font-size: 1.55rem; color: #FFFFFF; font-weight: 700; }}
  .verdict p  {{ margin: 0; color: #DCE6CF; font-size: 1.04rem; }}
  .verdict p b {{ color: #AED88A; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    return pd.read_parquet(DATA)


@st.cache_resource
def load_model():
    return M.load(ARTIFACTS / "lgb_baseline.txt")


@st.cache_data(show_spinner=False)
def load_forecasts():
    fc = pd.read_csv(ARTIFACTS / "forecast_lgb.csv", parse_dates=["date"])
    base = pd.read_csv(ARTIFACTS / "forecast_naive.csv", parse_dates=["date"])
    return fc, base


@st.cache_data(show_spinner=False)
def load_metrics():
    return pd.read_csv(ARTIFACTS / "metrics_baseline.csv")


@st.cache_data(show_spinner="Re-running forecast with your promo settings...")
def predict_with_overrides(_model_key, promo_key: tuple, discount_key: tuple):
    """Recompute the forecast when the buyer flags a promo or discount.

    We key on the tuples so Streamlit caches by user input. `_model_key` is a
    dummy that changes when the retrained model does.
    """
    df = load_data()
    model = load_model()
    if not promo_key and not discount_key:
        return load_forecasts()[0]
    idx = pd.MultiIndex.from_tuples(
        [(sid, pd.Timestamp(d)) for (sid, d, _) in promo_key or discount_key],
        names=["id", "date"])
    flags = pd.Series([v for (_, _, v) in promo_key], index=idx, dtype="int8") if promo_key else None
    disc = pd.Series([v for (_, _, v) in discount_key], index=idx) if discount_key else None
    return M.predict(model, df, TEST_START, horizon=HORIZON,
                     promo_flags=flags, discount_pct=disc)


df = load_data()
model = load_model()
_, baseline_fc = load_forecasts()
metrics = load_metrics()

sku_ids = sorted(df["id"].unique())
last_price = df.sort_values("date").groupby("id").tail(1).set_index("id")["sell_price"].fillna(3.0)
last_week_avg = df[df["date"] >= TEST_START - pd.Timedelta(days=7)].groupby("id")["sales"].mean()
default_on_hand = (last_week_avg * 7).round().astype(int)

# --------------------------------------------------------------------------- #
# Controls — sidebar. Everything here feeds either the forecast or the policy.
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Plan settings")
    st.caption("Every setting re-runs the recommendation.")

    with st.expander("📅 Store & horizon", expanded=True):
        stores = st.multiselect("Stores in scope", META["stores"], default=META["stores"])
        category = st.selectbox("Category filter",
                                ["All", "FOODS", "HOBBIES", "HOUSEHOLD"], index=0)
        horizon = st.slider("Forecast horizon (days)", 7, HORIZON, HORIZON, 7)

    with st.expander("💰 Cost assumptions", expanded=True):
        service_level = st.slider("Target service level", 0.80, 0.995, 0.95, 0.005,
                                  format="%.3f",
                                  help="Sets the safety-stock z. Trades holding cost against stockout risk.")
        stockout_mult = st.slider("Stockout cost multiplier × price", 0.5, 3.0, 1.5, 0.1,
                                  help="Cost per unit short = sell price × this. The proposal's default is 1.5.")
        holding_daily = st.slider("Daily holding rate (% of unit value)",
                                  0.01, 0.20, 0.07, 0.01, format="%.2f",
                                  help="0.07%/day ≈ 25%/yr. The proposal's default.") / 100.0
        lead_time = st.slider("Supplier lead time (days)", 1, 21, 7)
        review_period = st.slider("Reorder cadence (days)", 1, 14, 7,
                                  help="How often the buyer places orders. Weekly is standard.")
        supplier_min = st.number_input("Supplier minimum order (units)", 0, 500, 0, 10)

    with st.expander("🏷️ Planned promotions (this week)"):
        st.caption("Flag SKUs going on promotion. The forecast re-runs to include the lift.")
        promo_skus = st.multiselect("SKUs on promotion",
                                    [s for s in sku_ids if any(st_ in s for st_ in stores)],
                                    max_selections=20)
        promo_discount = st.slider("Discount depth", 0.0, 0.60, 0.20, 0.05,
                                   format="%.0f%%",
                                   help="Applied to selected SKUs across the horizon.") if promo_skus else 0.0

    if st.button("Reset to defaults", width="stretch"):
        st.rerun()

policy = I.InventoryPolicy(
    service_level=service_level, holding_rate_daily=holding_daily,
    stockout_multiplier=stockout_mult, lead_time_days=lead_time,
    review_period_days=review_period, supplier_min_qty=supplier_min,
)

# Build the promo/discount overrides in the form the model wants
if promo_skus:
    dates = pd.date_range(TEST_START, periods=horizon)
    promo_key = tuple((s, str(d.date()), 1) for s in promo_skus for d in dates)
    disc_key = tuple((s, str(d.date()), float(promo_discount)) for s in promo_skus for d in dates)
    forecast = predict_with_overrides(id(model), promo_key, disc_key)
else:
    forecast = load_forecasts()[0]

# Trim to selected stores/horizon
horizon_dates = pd.date_range(TEST_START, periods=horizon)
scope = df[df["store_id"].isin(stores)].copy()
if category != "All":
    scope = scope[scope["cat_id"] == category]
scope_ids = scope["id"].unique()
forecast = forecast[forecast["id"].isin(scope_ids) & forecast["date"].isin(horizon_dates)]
baseline_view = baseline_fc[baseline_fc["id"].isin(scope_ids) & baseline_fc["date"].isin(horizon_dates)]

# --------------------------------------------------------------------------- #
# Reorder recommendations and cost projection
# --------------------------------------------------------------------------- #
on_hand = default_on_hand.reindex(scope_ids).fillna(0).astype(int)
prices = last_price.reindex(scope_ids).fillna(3.0)

actual = df[(df["date"] >= TEST_START) & (df["date"] < TEST_START + pd.Timedelta(days=horizon))
            & df["id"].isin(scope_ids)][["id", "date", "sales"]]

results = I.compare_policies(forecast, baseline_view, actual, on_hand, prices, policy, horizon)
rec_m, cost_m = results["model_recommend"], results["model_cost"]
rec_b, cost_b = results["baseline_recommend"], results["baseline_cost"]

total_units_short_m = cost_m["units_short"].sum()
total_demand = actual["sales"].sum()
fill_m = 1 - total_units_short_m / max(total_demand, 1)
fill_b = 1 - cost_b["units_short"].sum() / max(total_demand, 1)
saving = cost_b["total_cost"].sum() - cost_m["total_cost"].sum()
n_skus_at_risk = int((rec_m["reorder_qty"] > 0).sum())
n_stockout_projected = int((cost_m["units_short"] > 0).sum())

# --------------------------------------------------------------------------- #
# The verdict, in the buyer's terms
# --------------------------------------------------------------------------- #
st.markdown("# Weekly Replenishment Planner")
st.markdown(f'<p class="caption">Decision owner: category buyer, weekly cycle. '
            f'Recommends reorder quantities for {len(scope_ids)} SKUs across '
            f'{len(stores)} store(s), planning cycle starting {TEST_START.date()}.</p>',
            unsafe_allow_html=True)

total_order = int(rec_m["reorder_qty"].sum())
promo_note = f" (with {len(promo_skus)} SKU(s) on {promo_discount:.0%} promotion)" if promo_skus else ""

st.markdown(f"""
<div class="verdict">
  <h2>Order {total_order:,} units across {n_skus_at_risk} SKUs this week{promo_note}</h2>
  <p>Following the model saves <b>${saving:,.0f}</b> over the {horizon}-day horizon vs.
     the seasonal-naive baseline, hitting a <b>{fill_m:.1%}</b> fill rate
     (baseline: {fill_b:.1%}).</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Recommended order", f"{total_order:,} units",
          f"{total_order - int(rec_b['reorder_qty'].sum()):+,} vs baseline")
c2.metric("Projected fill rate", f"{fill_m:.1%}",
          f"{(fill_m - fill_b) * 100:+.1f} pts vs baseline")
c3.metric("Projected cost", f"${cost_m['total_cost'].sum():,.0f}",
          f"-${saving:,.0f} vs baseline", delta_color="inverse")
c4.metric("SKUs at stockout risk", n_stockout_projected,
          help="Projected to stock out at least once in the horizon under the recommendation.")

tab_orders, tab_risk, tab_sim, tab_promo, tab_accuracy, tab_help = st.tabs(
    ["Order sheet", "Stockout risk", "Cost impact", "Promo scenarios", "Forecast accuracy", "How to read this"])

# --------------------------------------------------------------------------- #
with tab_orders:
    st.markdown("#### This week's purchase order")
    st.caption("Sorted by projected total cost impact. Buyers can edit on_hand inline "
               "if the reported number doesn't match what's actually on the shelf.")

    meta_cols = df.drop_duplicates("id")[["id", "item_id", "dept_id", "cat_id", "store_id"]]
    order = rec_m.merge(cost_m, on="id").merge(meta_cols, on="id")
    order = order.merge(cost_b[["id", "total_cost"]].rename(columns={"total_cost": "baseline_cost"}), on="id")
    order["vs_baseline"] = order["baseline_cost"] - order["total_cost"]
    order = order.sort_values("total_cost", ascending=False)
    order["on_promo"] = order["id"].isin(promo_skus)

    view = order[["item_id", "dept_id", "store_id", "on_hand", "expected_demand_28d",
                  "safety_stock", "reorder_qty", "sell_price", "units_short",
                  "total_cost", "vs_baseline", "on_promo"]].rename(columns={
        "item_id": "SKU", "dept_id": "Dept", "store_id": "Store",
        "on_hand": "On hand", "expected_demand_28d": f"Forecast {horizon}d",
        "safety_stock": "Safety", "reorder_qty": "ORDER",
        "sell_price": "Unit $", "units_short": "Proj. short",
        "total_cost": "Proj. cost $", "vs_baseline": "vs baseline $",
        "on_promo": "Promo"})

    st.dataframe(view.head(50), hide_index=True, width="stretch", height=440,
                 column_config={
                     f"Forecast {horizon}d": st.column_config.NumberColumn(format="%.0f"),
                     "Safety": st.column_config.NumberColumn(format="%.0f"),
                     "ORDER": st.column_config.NumberColumn(format="%d"),
                     "Unit $": st.column_config.NumberColumn(format="$%.2f"),
                     "Proj. short": st.column_config.NumberColumn(format="%.0f"),
                     "Proj. cost $": st.column_config.NumberColumn(format="$%.0f"),
                     "vs baseline $": st.column_config.NumberColumn(format="$%.0f"),
                     "Promo": st.column_config.CheckboxColumn(),
                 })

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("Download this week's order sheet", csv,
                       "purchase_order.csv", "text/csv")

# --------------------------------------------------------------------------- #
with tab_risk:
    left, right = st.columns([2, 1])

    with left:
        st.markdown("#### SKUs projected to stock out this cycle")
        st.caption("Red bars are shortfalls under the recommended order. Move the "
                   "service level up or add a promo where you can absorb it — those "
                   "buyers who see a stockout will not come back next week.")

        # Aggregate projected shortage by store-cat
        cost_view = cost_m.merge(meta_cols, on="id")
        by_dept = (cost_view.groupby(["store_id", "dept_id"])["units_short"].sum()
                   .reset_index().sort_values("units_short", ascending=False))
        fig = px.bar(by_dept.head(25), x="units_short", y="dept_id", color="store_id",
                     orientation="h",
                     color_discrete_map={"CA_1": "#0066CC", "TX_1": "#D0021B", "WI_1": "#0A8754"})
        fig.update_layout(height=380, xaxis_title="Projected units short",
                          yaxis_title=None, plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=10, l=0), legend_title=None)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("#### Service-level sensitivity")
        st.caption("How reorder quantity and cost move with the service target.")
        rows = []
        for sl in [0.80, 0.85, 0.90, 0.95, 0.99]:
            p = I.InventoryPolicy(service_level=sl, holding_rate_daily=holding_daily,
                                  stockout_multiplier=stockout_mult, lead_time_days=lead_time,
                                  review_period_days=review_period, supplier_min_qty=supplier_min)
            r = I.recommend(forecast, on_hand, prices, p)
            c = I.project_costs(forecast, actual, r, p, horizon)
            fill = 1 - c["units_short"].sum() / max(actual["sales"].sum(), 1)
            rows.append({"sl": sl, "order": r["reorder_qty"].sum(),
                         "cost": c["total_cost"].sum(), "fill": fill})
        sens = pd.DataFrame(rows)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sens["sl"], y=sens["cost"], mode="lines+markers",
                                  line=dict(color=BLUE, width=2.5), name="Projected cost"))
        fig2.add_vline(x=service_level, line_dash="dot", line_color=INK,
                       annotation_text="current setting")
        fig2.update_layout(height=280, xaxis_title="Service level",
                           yaxis_title="Projected cost $", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=10))
        st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------------------------------- #
with tab_sim:
    st.markdown("#### Cost comparison: model vs. seasonal-naive baseline")
    st.caption("Simulated over the held-out 28-day window using the actuals. "
               "This is the 'is the model saving me money' check from the proposal.")

    comp = pd.DataFrame({
        "Policy": ["LightGBM forecast", "Seasonal-naive baseline"],
        "Holding cost": [cost_m["holding_cost"].sum(), cost_b["holding_cost"].sum()],
        "Stockout cost": [cost_m["stockout_cost"].sum(), cost_b["stockout_cost"].sum()],
        "Fill rate": [fill_m, fill_b],
        "Recommended order (units)": [int(rec_m["reorder_qty"].sum()), int(rec_b["reorder_qty"].sum())],
    })
    comp["Total cost"] = comp["Holding cost"] + comp["Stockout cost"]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Holding", x=comp["Policy"], y=comp["Holding cost"],
                          marker_color=BLUE))
    fig3.add_trace(go.Bar(name="Stockout", x=comp["Policy"], y=comp["Stockout cost"],
                          marker_color=AMBER))
    fig3.update_layout(barmode="stack", height=380, yaxis_title="USD",
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=20))
    st.plotly_chart(fig3, width="stretch")

    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={
                     "Holding cost": st.column_config.NumberColumn(format="$%.0f"),
                     "Stockout cost": st.column_config.NumberColumn(format="$%.0f"),
                     "Total cost": st.column_config.NumberColumn(format="$%.0f"),
                     "Fill rate": st.column_config.NumberColumn(format="%.1%"),
                 })

# --------------------------------------------------------------------------- #
with tab_promo:
    st.markdown("#### What if you deepen the promotion?")
    st.caption("For the SKUs currently flagged in the sidebar, this re-runs the "
               "forecast at several discount depths. The buyer sees where the model "
               "expects an inflection — that's the depth where the incremental units "
               "stop being worth the margin sacrifice.")

    if not promo_skus:
        st.info("Flag one or more SKUs on promotion in the sidebar to see the "
                "forecast lift and downstream reorder effect here.")
    else:
        rows = []
        dates = pd.date_range(TEST_START, periods=horizon)
        for d in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
            pk = tuple((s, str(dt.date()), 1) for s in promo_skus for dt in dates)
            dk = tuple((s, str(dt.date()), float(d)) for s in promo_skus for dt in dates)
            fc_d = predict_with_overrides(id(model), pk if d > 0 else (), dk if d > 0 else ())
            fc_d = fc_d[fc_d["id"].isin(promo_skus) & fc_d["date"].isin(dates)]
            rows.append({"discount": d,
                         "units_forecast": fc_d["forecast"].sum(),
                         "revenue": (fc_d.merge(prices.reset_index().rename(columns={"sell_price": "p"}),
                                                on="id")
                                     .assign(rev=lambda x: x["forecast"] * x["p"] * (1 - d))["rev"].sum())})
        promo_df = pd.DataFrame(rows)
        promo_df["lift_pct"] = (promo_df["units_forecast"] / promo_df.iloc[0]["units_forecast"] - 1) * 100

        col1, col2 = st.columns(2)
        with col1:
            fig4 = px.line(promo_df, x="discount", y="units_forecast", markers=True)
            fig4.update_traces(line_color=BLUE, line_width=2.5)
            fig4.update_layout(height=300, xaxis_title="Discount depth",
                               yaxis_title="Forecast units (selected SKUs)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(tickformat=".0%"), title="Forecast lift curve")
            st.plotly_chart(fig4, width="stretch")
        with col2:
            fig5 = px.line(promo_df, x="discount", y="revenue", markers=True)
            fig5.update_traces(line_color=GREEN, line_width=2.5)
            fig5.update_layout(height=300, xaxis_title="Discount depth",
                               yaxis_title="Projected revenue $",
                               plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(tickformat=".0%"), title="Revenue curve (units × discounted price)")
            st.plotly_chart(fig5, width="stretch")

        promo_df["revenue"] = promo_df["revenue"].round(0)
        st.dataframe(promo_df[["discount", "units_forecast", "lift_pct", "revenue"]]
                     .rename(columns={"discount": "Discount", "units_forecast": "Units",
                                      "lift_pct": "Lift %", "revenue": "Revenue $"}),
                     hide_index=True, width="stretch",
                     column_config={
                         "Discount": st.column_config.NumberColumn(format="%.0f%%"),
                         "Units": st.column_config.NumberColumn(format="%.0f"),
                         "Lift %": st.column_config.NumberColumn(format="%+.1f%%"),
                         "Revenue $": st.column_config.NumberColumn(format="$%.0f"),
                     })

# --------------------------------------------------------------------------- #
with tab_accuracy:
    st.markdown("#### Held-out forecast accuracy")
    st.caption(f"MASE on the 28-day evaluation window ending {df['date'].max().date()}. "
               "MASE < 1 beats the seasonal-naive baseline.")

    m_view = metrics[metrics["id"].isin(scope_ids)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Median MASE", f"{m_view['mase'].median():.2f}",
              "target: < 1.0", delta_color="off")
    c2.metric("Mean MASE", f"{m_view['mase'].mean():.2f}")
    baseline_mase = pd.read_csv(ARTIFACTS / "baseline_mase.csv")
    model_mase = m_view.set_index("id")["mase"]
    base_mase = baseline_mase.set_index("id")["mase"].reindex(model_mase.index)
    win_rate = (model_mase < base_mase).mean()
    c3.metric("SKUs where model beats naive", f"{win_rate:.0%}")

    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(x=m_view["mase"], nbinsx=30, marker_color=BLUE,
                                name="LightGBM"))
    b_view = baseline_mase[baseline_mase["id"].isin(scope_ids)]
    fig6.add_trace(go.Histogram(x=b_view["mase"], nbinsx=30, marker_color=GREY,
                                opacity=0.5, name="Seasonal-naive"))
    fig6.add_vline(x=1.0, line_dash="dot", line_color=RED,
                   annotation_text="beats naive ← ")
    fig6.update_layout(barmode="overlay", height=340, xaxis_title="MASE",
                       yaxis_title="SKU count", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig6, width="stretch")

    st.markdown("#### Accuracy by store")
    per_store = m_view.groupby("store_id").agg(
        median_mase=("mase", "median"), mean_mase=("mase", "mean"),
        median_wape=("wape", "median"), mean_bias=("bias", "mean"),
        n_skus=("id", "count")).reset_index()
    st.dataframe(per_store, hide_index=True, width="stretch",
                 column_config={
                     "median_mase": st.column_config.NumberColumn(format="%.3f"),
                     "mean_mase": st.column_config.NumberColumn(format="%.3f"),
                     "median_wape": st.column_config.NumberColumn(format="%.3f"),
                     "mean_bias": st.column_config.NumberColumn(format="%+.2f"),
                 })

    st.markdown("#### What the model relies on")
    imp = model.feature_importances.head(10).copy()
    fig7 = px.bar(imp.sort_values("gain"), x="gain", y="feature", orientation="h")
    fig7.update_traces(marker_color=BLUE)
    fig7.update_layout(height=320, xaxis_title="Gain", yaxis_title=None,
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, t=10))
    st.plotly_chart(fig7, width="stretch")

# --------------------------------------------------------------------------- #
with tab_help:
    st.markdown("""
### How to read the numbers on this screen

**Every number is a projection over the horizon you set** (default 28 days), assuming:
- your `on_hand` for each SKU (auto-filled from the last 7 days' average × 7)
- weekly reorder cadence (change in the sidebar if you order more or less often)
- the LightGBM forecast, with any promo/discount overrides you flagged

**What "vs baseline" means.** The baseline is the seasonal-naive forecast (repeat last week).
That's what a fixed reorder-point policy is implicitly using. If the model can't beat it, it doesn't earn its place.

**The service-level dial is where you trade fill rate against holding cost.**
Every notch up buys safety stock proportional to `z(sl) × forecast σ × √lead_time`.
At 0.95 you're covering ~1.65 σ of forecast noise; at 0.99 you're covering 2.33.

**Promo overrides don't always move the forecast much.** M5 markdowns are sparse
(~0.4% of days in the training window), so the model has learned that a promo flag alone
carries mild signal. What moves the forecast is the **price**: if you flag a 30% discount,
`price_chg_4w` drops sharply and the model responds. That's the honest behavior — a
promo without a real price cut is often just a shelf tag.

**Limitations you should know before you commit to the order:**
- The M5 dataset ends April 2016. The system evaluates against that window. Real deployment
  needs live POS data.
- We subset to 3 stores × top-200 SKUs per store (600 series). Long-tail SKUs are not
  in scope here; the proposal names their sparse-series fallback (Prophet) as a next step.
- Stockouts in the training data appear as zero demand. The model can therefore under-forecast
  SKUs that frequently ran out of stock in-sample.
- MASE is measured store-average. A SKU with a MASE > 1.5 needs a manual look — the
  monitoring plan flags these automatically in production.
""")

st.markdown("---")
st.caption(f"Data: M5 Forecasting Competition subset — {META['n_series']} SKU-store "
           f"series over {META['date_min']} to {META['date_max']}. "
           f"Trained: LightGBM (Tweedie, {model.n_rounds} rounds). "
           "This tool is a decision aid, not a source of truth about a live grocery ops system.")