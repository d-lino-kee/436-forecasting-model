# Weekly Replenishment Planner — M5 Retail Demand IDSS

An interactive decision support system for the question in our proposal:

> **Which SKUs should the category buyer reorder this week, and in what quantities,
> given a 28-day demand forecast and current beliefs about promotions, cost, and
> service level?**

**Stakeholder:** the category buyer at a Walmart-scale grocery retailer, running
a weekly replenishment cycle across hundreds of SKUs per store.

**The decision the tool produces:** a weekly purchase order — SKU-by-SKU
quantities the buyer submits to the supplier. Not a report, not an analysis.

**Consequence of getting it wrong:** underordering stocks out popular items,
losing sales and pushing customers to competitors; overordering ties up cash
in slow-moving inventory and, in fresh categories, generates waste.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One-time: prepare data + train model. ~1 min.
# Requires the raw M5 CSVs in /mnt/user-data/uploads (see Data section).
python src/prepare_data.py
python -m src.train                                  # writes artifacts/lgb_baseline.txt

streamlit run app.py                                 # opens http://localhost:8501
pytest -q                                            # 12 tests, ~100s
```

## Results at a glance

Held-out 28-day window (M5 test period), 600 SKU-store series across CA / TX / WI:

| Metric | LightGBM | Seasonal-naive baseline |
|---|---|---|
| Median MASE | **0.80** | 0.95 |
| Mean MASE | **0.87** | 1.03 |
| SKUs where model wins | **77.5%** | — |
| Projected fill rate @ SL=0.95 | **93.1%** | 90.5% |
| Projected 28-day cost | **$24.7k** | $31.0k |
| **Cost saving** | **20.5%** | — |

**The 20% saving lands inside the 10–20% band the proposal committed to.**

## What to try in the interface

The point of the tool is that the recommendation moves. Under the shipped baseline
it recommends **43,885 units across 391 SKUs**. Then:

| Move this | What happens | Why |
|---|---|---|
| **Service level → 0.99** | Order rises to 45.9k, cost drops to $22.4k | Cheaper to buffer than to stock out at this ratio |
| **Lead time → 3 days** | Order collapses to 19.2k | Less pipeline to cover; buyer can afford a leaner shelf |
| **Stockout multiplier → 3.0×** | Cost doubles to $48k | Reflects the buyer's revised belief about lost-sale severity |
| **Flag 2 SKUs on 50% promo** | Order rises by ~250 units on those SKUs | Model responds to the price cut, not just the promo flag |

Open the **Promo scenarios** tab for the discount-depth curve — that's where the
buyer sees whether an extra 10% off is worth the margin sacrifice.

## Why this needs to be an IDSS, not a report

Two reasons, and the tool is built so neither can be removed:

1. **The buyer's beliefs are the inputs.** A promo they haven't communicated yet,
   a supplier lead-time change, a revised service target for a category —
   none of this is in the M5 dataset. The buyer sees the model's initial
   recommendation, flags what they know, and re-runs. **A one-shot form
   wouldn't work because the buyer often doesn't know which input to change
   until they see the model's output flag a SKU at risk.**
2. **The data doesn't sit still.** Sales come in nightly, demand shifts weekly,
   and the forecast is only worth what its most recent inputs are worth. The
   proposal calls for monthly retrains on a rolling 2-year window; this
   implementation matches that cadence and reads live inputs at every run.

A scheduled script emailing a PDF would answer the question once, for one set of
assumptions. That's the thing this replaces.

## How the model works

`src/model.py` fits **one pooled LightGBM regressor over all SKU-store series**,
with item / dept / store / state as categorical features. This is the approach
top-ranked M5 solutions used — pooling lets one tree ensemble share signal across
series while specializing through the categorical splits.

- **Target:** daily unit sales (Tweedie loss — handles the zero-inflated count
  distribution grocery demand actually has).
- **Features:** lag_{7, 14, 28} sales, rolling_{7, 28} mean/std/max, weekly and
  monthly seasonality, SNAP eligibility, calendar events, current sell price,
  4-week price change, and a runtime promo flag.
- **Multi-day horizon:** recursive forecasting — predict day t+1, feed that
  prediction into the lags for t+2. Cleaner than fitting 28 separate models and
  lets runtime overrides (promo, discount) propagate naturally.

### Trade-offs, honestly

Pooled trees smooth over SKU-specific idiosyncrasies that a per-series model
would catch. On our 600-series slice the pooling still wins on 77.5% of SKUs
because the shared signal is huge (weekly cycle, holidays, SNAP months, price
elasticity), and the SKU-level lag/rolling features provide enough per-series
context. **The proposal names Prophet as a fallback for very sparse SKUs; we
haven't implemented that** — for the SKU volumes we've subset to, it wasn't the
binding constraint. It would be the first thing to add for a long-tail deployment.

### What the model responds to

The **Forecast accuracy** tab in the interface plots feature importance. On our
run the top drivers, in order, are:
1. `roll_mean_7` — recent week's average, the strongest single signal
2. `roll_mean_28` — monthly baseline
3. `item_id` — the pooled specialization signal
4. `roll_max_7` — recent spike behaviour
5. `wday` — day-of-week seasonality

The `promo` flag alone has small importance because historical M5 markdowns
are rare (~0.4% of days). What actually moves the forecast on a promo is the
**price change** — `price_chg_4w` responds strongly to a discount, so the
interface lets the buyer enter both a promo flag and a discount depth. Setting
a 30% discount on a responsive FOODS SKU lifts its 28-day forecast by ~108%.

### Recursive prediction — the gotcha

The recursive-forecast loop refeaturizes each future day before predicting it.
An early implementation wrote predictions back to the tail dataframe using a
positional mask, but `add_features` internally sorts, so predictions landed on
the wrong rows — silently. Runtime overrides (promo, discount) looked inert,
median MASE looked ~30% worse than the model was actually capable of, and this
was invisible without an explicit isolation test.

The fix (in `src/model.py`) is to route write-back through an (id, date) map.
`tests/test_pipeline.py::test_override_isolates_to_the_targeted_sku` and
`test_promo_and_discount_lift_forecast_on_responsive_sku` are the regression
tests that lock this in.

## Data

Source: [M5 Forecasting Competition (Zenodo mirror)](https://zenodo.org/records/12636070).

| File | Purpose |
|---|---|
| `sales_train_validation.csv` | Daily unit sales, wide format (30,490 series × 1,913 days) |
| `calendar.csv` | Day-level SNAP flags, holidays, named events |
| `sell_prices.csv` | Weekly sell price per store-item |

Full dataset is ~450MB. `src/prepare_data.py` subsets to:
- **3 stores** (`CA_1`, `TX_1`, `WI_1`) — one per state, preserves the SNAP
  scheduling differences the proposal names in its monitoring plan
- **Top 200 SKUs per store by volume** — the SKUs that actually move
- **Last 2 years** — the rolling window the proposal specifies

That's 600 series × 730 days = 438K rows, ~13MB parquet. Small enough that
LightGBM trains in ~20s and the interactive forecast runs in ~7s.

### Limitations — read before quoting a number

- **The training data ends April 2016.** In production, sales would come from
  a live POS feed via overnight batch. All results in this repo are on the M5
  test window; nothing here is a claim about a live grocery ops system.
- **Long-tail SKUs are excluded.** Top-200-per-store misses about 90% of the
  catalog by count. The proposal's Prophet fallback is where those would go.
- **Stockout censoring.** M5 stockouts appear as zero demand, so the model can
  under-forecast SKUs that ran out repeatedly in-sample. There's no bias
  correction here; the monitoring plan is designed to catch this via override
  rate and per-SKU MASE drift.
- **Prices are the last-known price.** The recursive forecast does not model
  competitor repricing or supplier cost changes.
- **The cost simulation is deterministic.** It runs the recommendation against
  the actuals in the held-out window; real fill rates in a live deployment
  will differ because the buyer's on-hand and mid-week adjustments won't match
  the simulation's periodic-review policy exactly.
- **Promo flag is weak signal alone.** Real markdowns are rare in M5 (~0.4%
  of days), so the model learns to weight the flag lightly. The interface
  exposes discount depth as the primary lever, which the model responds to
  through `price_chg_4w`.

## Layout

```
app.py                 Streamlit interface — sidebar controls, 6 tabs
src/prepare_data.py    M5 CSV → subset parquet
src/features.py        Lag, rolling, calendar, price, promo feature engineering
src/model.py           LightGBM train + recursive predict + MASE + naive baseline
src/inventory.py       Reorder policy + day-by-day cost simulation
tests/test_pipeline.py 12 tests: leakage, promo response, isolation, cost direction
artifacts/             Trained model, forecasts, per-SKU MASE — regenerated by training
data/                  Prepared parquet + metadata (raw CSVs live outside the repo)
```

## Operationalization

- **Access:** Streamlit app behind SSO, one container, buyers hit a URL. Stateless,
  scales horizontally if adoption grows.
- **Infrastructure:** LightGBM training is single-threaded, ~20s on the 600-series
  slice. Prediction is ~7s. Runs on 2 vCPU / 4GB. No GPU.
- **Pipeline after launch:**
  - Overnight batch pulls yesterday's POS sales into the long-format parquet.
  - Weekly: buyer opens the tool, adjusts overrides, downloads the purchase order.
  - Monthly: retrain on the rolling 2-year window (a scheduled job, not a manual
    step). Retrain also triggers if the store-level median MASE exceeds 1.2 for
    two consecutive weeks — the drift threshold the proposal specifies.
- **Monitoring:** the **Forecast accuracy** tab shows median/mean MASE, per-store
  breakdown, and win rate vs. baseline. Production would add override rate
  (how often buyers deviate from recommendations) as the primary trust signal.

## AI use disclosure

See the AI Use Disclosure slide in the report deck.