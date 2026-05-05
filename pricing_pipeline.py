"""
================================================================================
  OPTIMAL EXPORT PRICING PIPELINE UNDER FX UNCERTAINTY
  -------------------------------------------------------
  Author  : Senior Data Scientist
  Purpose : Full stochastic pricing model for confectionery export products
  Model   : GARCH(1,1) FX forecasting + Stochastic Lerner Markup Optimization
            + Monte Carlo Simulation (10,000 paths) + Price Elasticity OLS

  Output  : Optimal_Pricing_Model_Results.xlsx  (5-sheet Excel report)

  Sheets  :
    1. 📊 Executive Summary   — 7 KPI cards + 4 analytical charts + GARCH table
    2. 🎯 Optimal Prices      — Top 50 SKUs, color-coded 4-scenario optimal prices
    3. 📈 FX Scenarios        — 12-month GARCH Monte Carlo table + elasticity charts
    4. 🌍 Country Analysis    — Risk-rated pricing per destination market
    5. 📐 Methodology         — Full mathematical framework documentation

  Mathematical Core:
    - Cost Model   : C(t) = α·FX(t)·C_usd + β·W(t) + C_fixed
    - FX Dynamics  : GARCH(1,1) → σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    - Demand Model : log(D) = a − ε·log(p) + γ·log(FX)  [OLS panel]
    - Optimal Price: p*(t) = E[C(FX_t)] · |ε|/(|ε|−1) + λ·Var[C(FX_t)]
    - Pass-Through : ρ* = 1 / (1 + 1/|ε|)

  VWAP Policy (applied consistently throughout):
    - ALL "Current Avg Price" metrics use Volume-Weighted Average Price.
    - VWAP = Σ(revenue_usd_i) / Σ(weight_kg_i)  where revenue_usd_i
      = revenue_toman_i / fx_rate_i  (each transaction at its own FX rate).
    - This is the "Realized VWAP" — what was actually received in USD per kg.
    - A second flavour, "FX-normalized VWAP", restates all Toman revenue at
      last_fx for period-over-period comparability; labelled explicitly where used.
    - Simple mean of per-row price_per_kg_usd is NEVER used for reporting
      because it gives equal weight to a 10-kg and a 10,000-kg transaction.

  Requirements:
    pip install pandas numpy scipy statsmodels arch matplotlib seaborn xlsxwriter openpyxl

  Usage:
    1. Set INPUT_FILE to your Excel file path
    2. Set OUTPUT_FILE to desired output path
    3. Run:  python pricing_pipeline.py
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← change these two paths as needed
# ──────────────────────────────────────────────────────────────────────────────
INPUT_FILE  = "Export_Pricing_Model__1_.xlsx"
OUTPUT_FILE = "Optimal_Pricing_Model_Results.xlsx"
CHARTS_DIR  = "."          # folder where temporary chart PNGs are written
# ──────────────────────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns
from arch import arch_model
import xlsxwriter


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  OPTIMAL EXPORT PRICING PIPELINE")
print("=" * 70)
print("\n📦  [1/9] Loading & cleaning data ...")

df_raw = pd.read_excel(INPUT_FILE)

# Keep only forward sales invoices; remove returns and invalid rows
INVOICE_TYPE = 'فاکتور فروش'    # Sales Invoice (Persian)
df = df_raw[df_raw['transaction_type'] == INVOICE_TYPE].copy()
df = df[df['net_sales_value']       > 0].copy()
df = df[df['qty_sold']              > 0].copy()
df = df[df['net_weight_carton_gr']  > 0].copy()
df = df[df['fx_rate_to_base_Toman'] > 0].copy()

# Parse Gregorian invoice date
df['invoice_date'] = pd.to_datetime(df['G_invoice_date'])
df = df.sort_values('invoice_date').reset_index(drop=True)

print(f"    ✅ {len(df):,} clean rows | "
      f"{df['sku_code'].nunique()} SKUs | "
      f"{df['invoice_date'].min().date()} → {df['invoice_date'].max().date()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n🔧  [2/9] Engineering financial features ...")

# Currency conversion:
#   net_sales_value is in Iranian Rials → ÷10 → Toman → ÷FX_rate → USD
#   net_weight_carton_gr is grams       → ÷1000 → kg

df['total_weight_kg']        = df['qty_sold'] * df['net_weight_carton_gr'] / 1_000
df['revenue_toman']          = df['net_sales_value']  / 10
df['discount_toman']         = df['discount_value']   / 10
df['gross_revenue_toman']    = df['revenue_toman'] + df['discount_toman']

# ── Transaction-level USD revenue (each row at its own FX rate) ──────────────
# This is the foundational building block for all VWAP calculations.
# revenue_usd_tx = what this transaction actually realised in USD.
# Summing these and dividing by total kg gives true Realized VWAP.
df['revenue_usd_tx']         = df['revenue_toman'] / df['fx_rate_to_base_Toman']

df['price_per_kg_toman']     = df['revenue_toman']    / df['total_weight_kg']
df['price_per_kg_usd']       = df['revenue_usd_tx']   / df['total_weight_kg']

df['price_per_carton_toman'] = df['revenue_toman']    / df['qty_sold']
df['price_per_carton_usd']   = df['price_per_carton_toman'] / df['fx_rate_to_base_Toman']

df['discount_rate']          = (df['discount_toman']
                                 / df['gross_revenue_toman'].replace(0, np.nan))

# Time aggregation helpers
df['year_month'] = df['invoice_date'].dt.to_period('M')
df['year']       = df['invoice_date'].dt.year

# Destination country mapping (Persian → English)
DEST_MAP = {
    'آذربایجان': 'Azerbaijan', 'عراق': 'Iraq',      'پاکستان': 'Pakistan',
    'تاجیکستان': 'Tajikistan', 'ترکمنستان': 'Turkmenistan', 'عمان': 'Oman',
    'بحرین':     'Bahrain',    'افغانستان': 'Afghanistan',  'امارات عربی متحده': 'UAE',
    'استرالیا':  'Australia',  'سوریه':     'Syria',        'قزاقستان': 'Kazakhstan',
    'کویت':      'Kuwait',     'اردن':      'Jordan',       'ازبکستان': 'Uzbekistan',
    'گرجستان':   'Georgia',    'یمن':       'Yemen',        'لیبی':     'Libya',
    'بلاروس':    'Belarus',    'لبنان':     'Lebanon',      'ارمنستان': 'Armenia',
}
df['country_en'] = df['destination_country'].map(DEST_MAP).fillna('Other')
df['currency_en'] = df['currency_code'].map({'دلار': 'USD', 'ریال': 'IRR'}).fillna('IRR')

# ── Portfolio-level Realized VWAP ─────────────────────────────────────────────
# Σ(revenue_usd_tx) / Σ(weight_kg)  — used for all top-level KPI cards.
portfolio_vwap_usd_kg = df['revenue_usd_tx'].sum() / df['total_weight_kg'].sum()

# Portfolio-level revenue-weighted discount rate
portfolio_vwap_discount = np.average(
    df['discount_rate'].fillna(0),
    weights=df['revenue_toman']
)

# Portfolio-level volume-weighted price std and CV (for Price Volatility KPI)
portfolio_vwap_std = np.sqrt(
    np.average(
        (df['price_per_kg_usd'] - portfolio_vwap_usd_kg) ** 2,
        weights=df['total_weight_kg']
    )
)
portfolio_price_cv = portfolio_vwap_std / portfolio_vwap_usd_kg

print("    ✅ All financial features computed")
print(f"    📊 Portfolio Realized VWAP = ${portfolio_vwap_usd_kg:.3f} / kg")
print(f"       (vs naïve simple mean  = ${df['price_per_kg_usd'].mean():.3f} / kg)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FX TIME SERIES + GARCH(1,1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n📈  [3/9] Fitting GARCH(1,1) on FX rate ...")

# Build volume-weighted monthly FX series
fx_monthly = (
    df.groupby('year_month')
      .apply(lambda g: np.average(g['fx_rate_to_base_Toman'], weights=g['revenue_toman']))
      .reset_index(name='fx_rate')
)
fx_monthly['date'] = fx_monthly['year_month'].dt.to_timestamp()
fx_monthly = fx_monthly.sort_values('date').reset_index(drop=True)

# Log returns (in %)
fx_monthly['log_return'] = (
    np.log(fx_monthly['fx_rate'] / fx_monthly['fx_rate'].shift(1)) * 100
)
fx_monthly = fx_monthly.dropna()

# Fit GARCH(1,1) with normal innovations
garch_mdl = arch_model(fx_monthly['log_return'], vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_mdl.fit(disp='off')

omega = garch_fit.params['omega']
alpha = garch_fit.params['alpha[1]']
beta  = garch_fit.params['beta[1]']
mu    = garch_fit.params['mu']

# ── Monte Carlo simulation: 10,000 paths × 12 months ──
np.random.seed(42)
N_SIMULATIONS = 10_000
HORIZON       = 12          # months ahead

last_fx     = fx_monthly['fx_rate'].iloc[-1]
last_sigma2 = garch_fit.conditional_volatility.iloc[-1] ** 2

simulated_paths = np.zeros((N_SIMULATIONS, HORIZON))
for sim in range(N_SIMULATIONS):
    fx_t     = last_fx
    sig2_t   = last_sigma2
    for t in range(HORIZON):
        eps    = np.random.standard_normal()
        sig2_t = omega + alpha * (mu + np.sqrt(sig2_t) * eps) ** 2 + beta * sig2_t
        sig2_t = max(sig2_t, 1e-8)
        r_t    = mu + np.sqrt(sig2_t) * eps
        fx_t   = fx_t * np.exp(r_t / 100)
        simulated_paths[sim, t] = fx_t

# Percentile fan-chart bands
sim_p5  = np.percentile(simulated_paths,  5, axis=0)
sim_p25 = np.percentile(simulated_paths, 25, axis=0)
sim_p50 = np.percentile(simulated_paths, 50, axis=0)
sim_p75 = np.percentile(simulated_paths, 75, axis=0)
sim_p95 = np.percentile(simulated_paths, 95, axis=0)

fx_forecast_dates = pd.date_range(
    fx_monthly['date'].iloc[-1] + pd.DateOffset(months=1),
    periods=HORIZON, freq='MS'
)

print(f"    GARCH params → ω={omega:.5f},  α={alpha:.4f},  β={beta:.4f},  α+β={alpha+beta:.4f}")
print(f"    Current FX   : {last_fx:,.0f} Toman/USD")
print(f"    12M forecast  : P5={sim_p5[-1]:,.0f} | P50={sim_p50[-1]:,.0f} | P95={sim_p95[-1]:,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRODUCT-LEVEL AGGREGATION (VOLUME-WEIGHTED METRICS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n🍫  [4/9] Aggregating product-level metrics (volume-weighted) ...")

# Helper function for volume-weighted average
def vwap(prices, weights):
    """Volume-weighted average price"""
    return np.average(prices, weights=weights)

def vwstd(prices, weights):
    """Volume-weighted standard deviation"""
    avg = np.average(prices, weights=weights)
    variance = np.average((prices - avg)**2, weights=weights)
    return np.sqrt(variance)

# Compute volume-weighted metrics per product
product_groups = df.groupby(['sku_code', 'product_name'])
top_products_list = []

for (sku, pname), grp in product_groups:
    # Realized VWAP: sum of USD revenues at each transaction's own FX rate
    total_revenue_usd = grp['revenue_usd_tx'].sum()
    total_weight_kg   = grp['total_weight_kg'].sum()

    # Realized VWAP = total USD revenue / total kg
    vwap_price_usd_kg = total_revenue_usd / total_weight_kg if total_weight_kg > 0 else 0

    # Volume-weighted std of prices
    if len(grp) > 1 and total_weight_kg > 0:
        vwstd_price = vwstd(grp['price_per_kg_usd'].values, grp['total_weight_kg'].values)
    else:
        vwstd_price = 0

    # FX-normalized VWAP: what would all Toman revenue be worth at today's FX?
    # Useful for comparing SKU competitiveness across periods without FX drift.
    total_revenue_toman = grp['revenue_toman'].sum()
    fx_normalized_price_usd_kg = (total_revenue_toman / last_fx) / total_weight_kg if total_weight_kg > 0 else 0

    # Volume-weighted discount rate (weighted by gross revenue Toman)
    vwap_discount = vwap(grp['discount_rate'].fillna(0).values, grp['gross_revenue_toman'].values)

    top_products_list.append({
        'sku_code': sku,
        'product_name': pname,
        'total_qty_cartons': grp['qty_sold'].sum(),
        'total_weight_kg': total_weight_kg,
        'total_revenue_usd': total_revenue_usd,
        'vwap_price_usd_kg': vwap_price_usd_kg,              # Realized VWAP (at actual FX rates)
        'vwap_price_fx_normalized': fx_normalized_price_usd_kg,  # FX-normalized for comparability
        'vwstd_price_usd_kg': vwstd_price,                   # Volume-weighted price volatility
        'vwap_discount_rate': vwap_discount,                 # Volume-weighted discount
        'n_transactions': len(grp),
        'net_weight_carton_gr': grp['net_weight_carton_gr'].iloc[0],
        'avg_fx_rate': vwap(grp['fx_rate_to_base_Toman'].values, grp['revenue_toman'].values),
    })

top_products = pd.DataFrame(top_products_list)
top_products = top_products.sort_values('total_weight_kg', ascending=False).head(50)

# For backward compatibility with rest of code, create aliases
top_products['avg_price_usd_kg'] = top_products['vwap_price_usd_kg']
top_products['std_price_usd_kg'] = top_products['vwstd_price_usd_kg']
top_products['avg_discount_rate'] = top_products['vwap_discount_rate']

print(f"    ✅ Top {len(top_products)} products identified by volume")
print(f"    📊 Using Realized VWAP (Σ revenue_usd_tx / Σ weight_kg) for all price metrics")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRICE ELASTICITY ESTIMATION (OLS PANEL with VWAP)
# ══════════════════════════════════════════════════════════════════════════════
print("\n📐  [5/9] Estimating price elasticity per product group (volume-weighted) ...")

#   Model: log(qty_kg) = a + ε·log(VWAP_price_usd_kg) + γ·log(fx_rate) + error
#   Using VWAP instead of simple mean for accurate price representation

# Build monthly panel with volume-weighted prices per group
group_panel_list = []
for (ym, grp_name), grp in df.groupby(['year_month', 'Group4Name']):
    total_qty_kg      = grp['total_weight_kg'].sum()
    total_revenue_usd = grp['revenue_usd_tx'].sum()   # each tx at its own FX rate

    # Realized VWAP for this group-month
    vwap_price = total_revenue_usd / total_qty_kg if total_qty_kg > 0 else np.nan

    # Revenue-weighted FX rate (for reporting / FX-sensitivity term in OLS)
    vwap_fx = np.average(grp['fx_rate_to_base_Toman'].values,
                          weights=grp['revenue_toman'].values)

    group_panel_list.append({
        'year_month': ym,
        'Group4Name': grp_name,
        'total_qty_kg': total_qty_kg,
        'vwap_price_usd_kg': vwap_price,
        'vwap_fx': vwap_fx,
    })

group_panel = pd.DataFrame(group_panel_list)

elasticity_results = []
MIN_OBS = 8   # minimum monthly observations to estimate elasticity

for grp in group_panel['Group4Name'].unique():
    gdata = group_panel[group_panel['Group4Name'] == grp].copy()
    gdata = gdata.dropna(subset=['total_qty_kg', 'vwap_price_usd_kg', 'vwap_fx'])
    gdata = gdata[(gdata['total_qty_kg'] > 0) & (gdata['vwap_price_usd_kg'] > 0)]
    if len(gdata) < MIN_OBS:
        continue
    log_qty   = np.log(gdata['total_qty_kg'])
    log_price = np.log(gdata['vwap_price_usd_kg'])
    log_fx    = np.log(gdata['vwap_fx'])
    X = np.column_stack([np.ones(len(log_price)), log_price, log_fx])
    try:
        coef, *_ = np.linalg.lstsq(X, log_qty, rcond=None)
        elasticity_results.append({
            'Group4Name':   grp,
            'elasticity':   coef[1],   # price elasticity of demand (ε)
            'fx_sensitivity': coef[2], # FX sensitivity coefficient (γ)
            'n_obs':        len(gdata),
        })
    except Exception:
        pass

elasticity_df = pd.DataFrame(elasticity_results).sort_values('elasticity')

# Filter for chart (reasonable range, adequate observations)
elas_plot = elasticity_df[
    (elasticity_df['elasticity'] > -6) &
    (elasticity_df['n_obs'] >= 10)
].copy()

median_elas = elasticity_df['elasticity'].median()
med_abs     = abs(median_elas)
med_pt_val  = 1 / (1 + 1 / med_abs)   # optimal pass-through at median elasticity

print(f"    ✅ {len(elasticity_df)} product groups with elasticity estimates")
print(f"    📊 Median elasticity ε = {median_elas:.2f}  →  optimal pass-through = {med_pt_val*100:.1f}%")
print(f"    ⚠️  All prices are Realized VWAP (Σ rev_usd_tx / Σ weight_kg)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OPTIMAL PRICE COMPUTATION (STOCHASTIC LERNER MARKUP)
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎯  [6/9] Computing optimal prices per FX scenario ...")

RISK_AVERSION   = 0.3    # λ — moderate risk-aversion parameter
FX_COST_SHARE   = 0.60   # share of costs that are USD-linked
ASSUMED_MARGIN  = 0.25   # gross margin assumption for implied-cost estimate

# FX scenarios from GARCH 12-month Monte Carlo percentiles
FX_SCENARIOS = {
    'Optimistic (P25)':  float(sim_p25[-1]),
    'Base Case (P50)':   float(sim_p50[-1]),
    'Pessimistic (P75)': float(sim_p75[-1]),
    'Stress (P95)':      float(sim_p95[-1]),
}

# Map SKU → product group → elasticity
sku_to_group = (df[['sku_code', 'Group4Name']]
                 .drop_duplicates()
                 .set_index('sku_code')['Group4Name'])
top_products['Group4Name'] = top_products['sku_code'].map(sku_to_group)
top_products = top_products.merge(
    elasticity_df[['Group4Name', 'elasticity', 'fx_sensitivity']],
    on='Group4Name', how='left'
)

# Fill missing with portfolio median; clip to economically sensible range
top_products['elasticity'] = (top_products['elasticity']
                               .fillna(median_elas)
                               .clip(-5, -0.1))

# Lerner markup factor = |ε| / (|ε| − 1)
top_products['lerner_markup']      = (top_products['elasticity'].abs()
                                       / (top_products['elasticity'].abs() - 1))

# Optimal FX pass-through rate ρ* = 1 / (1 + 1/|ε|)
top_products['optimal_passthrough'] = 1 / (1 + 1 / top_products['elasticity'].abs())

# Implied cost (USD/kg) ≈ current VWAP × (1 − assumed margin)
# avg_price_usd_kg is already the Realized VWAP from Section 4
top_products['implied_cost_usd_kg'] = (top_products['avg_price_usd_kg']
                                        * (1 - ASSUMED_MARGIN))

# Compute p*(FX_scenario) for each scenario
for scenario_name, fx_val in FX_SCENARIOS.items():
    fx_change_factor = fx_val / last_fx
    # Cost adjusted for FX change (only FX_COST_SHARE fraction is FX-linked)
    adj_cost = top_products['implied_cost_usd_kg'] * (
        1 + (fx_change_factor - 1) * FX_COST_SHARE
    )
    # Stochastic Lerner: p* = adj_cost × markup + λ × price_std (risk premium)
    risk_premium = RISK_AVERSION * top_products['std_price_usd_kg'].fillna(0.01)
    col = f'opt_price_{scenario_name[:4]}'   # e.g.  opt_price_Opti
    top_products[col] = (
        adj_cost * top_products['lerner_markup'] + risk_premium
    ).clip(lower=top_products['avg_price_usd_kg'] * 0.70)  # floor: −30%

scenario_cols = [c for c in top_products.columns if c.startswith('opt_price_')]

print(f"    ✅ Optimal prices computed for {len(top_products)} products × 4 scenarios")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MONTHLY & COUNTRY ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊  [7/9] Building monthly and country analytics ...")

# ── Monthly aggregation ───────────────────────────────────────────────────────
# revenue_usd is the sum of per-transaction USD amounts (each at its own FX rate).
# price_per_kg_usd = total_revenue_usd / total_weight_kg  →  monthly Realized VWAP.
# avg_fx is revenue-weighted for display purposes only (not used to convert revenue).
# avg_discount is revenue-weighted for accurate reporting.

monthly = (
    df.groupby('year_month')
      .agg(
          total_revenue_toman = ('revenue_toman',   'sum'),
          total_revenue_usd   = ('revenue_usd_tx',  'sum'),   # ← Realized USD, not /avg_fx
          total_weight_kg     = ('total_weight_kg', 'sum'),
          n_transactions      = ('qty_sold',        'count'),
      )
      .reset_index()
)

# Revenue-weighted FX (for charts / GARCH input — NOT for revenue conversion)
_monthly_fx = (
    df.groupby('year_month')
      .apply(lambda g: np.average(g['fx_rate_to_base_Toman'], weights=g['revenue_toman']))
      .reset_index(name='avg_fx')
)
# Revenue-weighted discount rate per month
_monthly_disc = (
    df.groupby('year_month')
      .apply(lambda g: np.average(g['discount_rate'].fillna(0), weights=g['revenue_toman']))
      .reset_index(name='avg_discount')
)

monthly = (monthly
           .merge(_monthly_fx,   on='year_month')
           .merge(_monthly_disc, on='year_month'))

monthly['date']             = monthly['year_month'].dt.to_timestamp()
monthly['revenue_usd']      = monthly['total_revenue_usd']
monthly['price_per_kg_usd'] = monthly['total_revenue_usd'] / monthly['total_weight_kg']

country_rev = (
    df.groupby('country_en')
      .agg(revenue_usd=('revenue_usd_tx', 'sum'), weight_kg=('total_weight_kg', 'sum'))
      .reset_index()
      .sort_values('revenue_usd', ascending=False)
      .head(10)
)

print("    ✅ Monthly and country aggregations ready (revenue-weighted FX, VWAP prices)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CHART GENERATION (4 figures → PNG files)
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨  [8/9] Generating analytical charts ...")

# ── Professional colour palette ──
C = {
    'primary': '#1B2A4A', 'accent':  '#E63946', 'green':  '#2DC653',
    'gold':    '#FFB703', 'light':   '#F8F9FA', 'gray':   '#6C757D',
    'blue':    '#457B9D', 'purple':  '#7B2D8B', 'teal':   '#2A9D8F',
}

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.facecolor':     '#F8F9FA',
    'figure.facecolor':   'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.4,
    'grid.linestyle':     '--',
})

# ── FIGURE 1: Executive Dashboard (2×2 panels) ──────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(18, 12))
fig1.suptitle('Export Pricing Intelligence Dashboard',
               fontsize=22, fontweight='bold', color=C['primary'], y=0.98)
fig1.patch.set_facecolor('white')

# Panel A — Monthly Revenue USD (area) + FX rate (right axis, dashed)
ax1  = axes[0, 0]
ax1b = ax1.twinx()
mp   = monthly[monthly['date'] >= '2023-01-01']
ax1.fill_between(mp['date'], mp['revenue_usd'] / 1e6, alpha=0.3, color=C['blue'])
ax1.plot(mp['date'], mp['revenue_usd'] / 1e6, color=C['blue'], lw=2.5,
          label='Revenue (USD M)')
ax1b.plot(mp['date'], mp['avg_fx'], color=C['accent'], lw=2, ls='--', label='FX Rate')
ax1.set_ylabel('Revenue (USD Millions)', color=C['blue'],   fontsize=10)
ax1b.set_ylabel('FX Rate (Toman/USD)',   color=C['accent'], fontsize=10)
ax1.set_title('Monthly Revenue vs FX Rate', fontweight='bold', fontsize=12, pad=10)
lns = ax1.get_lines() + ax1b.get_lines()
ax1.legend(lns, [l.get_label() for l in lns], loc='upper left', fontsize=9)
ax1.tick_params(axis='x', rotation=30)

# Panel B — GARCH(1,1) FX fan-chart forecast
ax2 = axes[0, 1]
ht  = fx_monthly.tail(24)
ax2.plot(ht['date'], ht['fx_rate'], color=C['primary'], lw=2.5, label='Historical FX')
ax2.fill_between(fx_forecast_dates, sim_p5,  sim_p95, alpha=0.15,
                  color=C['accent'], label='5%–95% CI')
ax2.fill_between(fx_forecast_dates, sim_p25, sim_p75, alpha=0.25,
                  color=C['gold'],   label='25%–75% IQR')
ax2.plot(fx_forecast_dates, sim_p50, color=C['accent'], lw=2.5, ls='--',
          label='Median Forecast')
ax2.axvline(pd.Timestamp('today'), color=C['gray'], ls=':', alpha=0.7)
ax2.set_title('GARCH(1,1) FX Forecast — 12 Months', fontweight='bold', fontsize=12, pad=10)
ax2.set_ylabel('Toman / USD', fontsize=10)
ax2.legend(fontsize=9)
ax2.tick_params(axis='x', rotation=30)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Panel C — Top 10 products VWAP price (horizontal bar)
ax3   = axes[1, 0]
top10 = top_products.head(10).copy()
top10['label'] = [f"SKU-{s}" for s in top10['sku_code']]
bars  = ax3.barh(range(len(top10)), top10['avg_price_usd_kg'],
                  color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top10))),
                  edgecolor='white', height=0.7)
ax3.set_yticks(range(len(top10)))
ax3.set_yticklabels(top10['label'], fontsize=9)
ax3.set_xlabel('VWAP (USD/kg)', fontsize=10)
ax3.set_title('Top 10 Products — Realized VWAP (USD/kg)', fontweight='bold', fontsize=12, pad=10)
for bar, (_, row) in zip(bars, top10.iterrows()):
    ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
              f"${row['avg_price_usd_kg']:.2f}", va='center', fontsize=8, color=C['primary'])

# Panel D — Price distribution by top-6 countries (box plot)
ax4      = axes[1, 1]
top6c    = country_rev.head(6)['country_en'].tolist()
cdf      = df[df['country_en'].isin(top6c)].copy()
cp       = cdf.groupby('country_en')['price_per_kg_usd'].apply(list)
bp       = ax4.boxplot(list(cp.values), labels=list(cp.index), patch_artist=True,
                        medianprops={'color': C['accent'], 'linewidth': 2},
                        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
box_pal  = [C['blue'], C['teal'], C['gold'], C['purple'], C['green'], C['primary']]
for patch, color in zip(bp['boxes'], box_pal):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_title('Price Distribution by Country (USD/kg)', fontweight='bold', fontsize=12, pad=10)
ax4.set_ylabel('Price (USD/kg)', fontsize=10)
ax4.tick_params(axis='x', rotation=20)
ax4.set_ylim(0, np.percentile(cdf['price_per_kg_usd'], 95) * 1.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
FIG1 = os.path.join(CHARTS_DIR, '_fig1_dashboard.png')
fig1.savefig(FIG1, dpi=150, bbox_inches='tight')
plt.close()
print("    ✅ Figure 1: Executive Dashboard")


# ── FIGURE 2: FX Pass-Through & Elasticity (1×3 panels) ─────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))
fig2.suptitle('FX Pass-Through & Price Elasticity Analysis',
               fontsize=18, fontweight='bold', color=C['primary'])
fig2.patch.set_facecolor('white')

# 2A — FX rate vs realised VWAP price scatter + regression
ax   = axes2[0]
sdf  = monthly[monthly['price_per_kg_usd'] < monthly['price_per_kg_usd'].quantile(0.98)]
sc   = ax.scatter(sdf['avg_fx'], sdf['price_per_kg_usd'],
                   c=sdf['date'].astype(np.int64), cmap='viridis',
                   s=80, alpha=0.8, edgecolors='white', lw=0.5)
m, b, r, *_ = stats.linregress(sdf['avg_fx'], sdf['price_per_kg_usd'])
xr = np.linspace(sdf['avg_fx'].min(), sdf['avg_fx'].max(), 100)
ax.plot(xr, m * xr + b, '--', color=C['accent'], lw=2, label=f'Trend (R²={r**2:.2f})')
ax.set_xlabel('FX Rate (Toman/USD)', fontsize=11)
ax.set_ylabel('VWAP (USD/kg)',        fontsize=11)
ax.set_title('FX Rate vs Realised VWAP Price', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
plt.colorbar(sc, ax=ax, label='Time →')

# 2B — Elasticity distribution (horizontal bar, colour-coded)
ax       = axes2[1]
colors_e = [C['green'] if e > -1 else C['gold'] if e > -2 else C['accent']
            for e in elas_plot['elasticity']]
ax.barh(range(len(elas_plot)), elas_plot['elasticity'], color=colors_e, alpha=0.75)
ax.axvline(-1, color=C['primary'], ls='--', lw=1.5, label='Unit elastic (ε=−1)')
ax.axvline(elas_plot['elasticity'].median(), color=C['gold'], ls=':',
            lw=1.5, label=f'Median = {elas_plot["elasticity"].median():.2f}')
ax.set_yticks([])
ax.set_xlabel('Price Elasticity of Demand (ε)', fontsize=11)
ax.set_title('Elasticity by Product Group', fontweight='bold', fontsize=12)
ax.legend(handles=[
    Patch(facecolor=C['green'],  label='|ε| < 1 (Inelastic — raise price)'),
    Patch(facecolor=C['gold'],   label='−2 < ε < −1 (Moderate)'),
    Patch(facecolor=C['accent'], label='ε < −2 (Elastic — be careful)'),
], fontsize=8, loc='lower right')

# 2C — Optimal pass-through curve with portfolio annotation
ax         = axes2[2]
er         = np.linspace(0.1, 5, 200)
pt         = 1 / (1 + 1 / er)
ax.plot(er, pt * 100, color=C['primary'], lw=2.5)
ax.axhline(50, color=C['gray'],   ls=':', alpha=0.6, label='50% pass-through')
ax.fill_between(er, pt * 100, 100, alpha=0.1, color=C['accent'], label='Absorbed by margin')
ax.fill_between(er, 0, pt * 100,  alpha=0.1, color=C['blue'],   label='Passed to customer')
ax.axvline(med_abs, color=C['gold'], ls='--', lw=2)
ax.annotate(
    f'Your portfolio\n|ε|={med_abs:.1f} → {med_pt_val*100:.0f}% pass-through',
    xy=(med_abs, med_pt_val * 100),
    xytext=(med_abs + 0.5, med_pt_val * 100 - 15),
    fontsize=9, color=C['primary'],
    arrowprops=dict(arrowstyle='->', color=C['primary']),
)
ax.set_xlabel('|Price Elasticity| (|ε|)', fontsize=11)
ax.set_ylabel('Optimal FX Pass-Through (%)', fontsize=11)
ax.set_title('Optimal Pass-Through Rate\nρ* = 1/(1 + 1/|ε|)', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 100)

plt.tight_layout()
FIG2 = os.path.join(CHARTS_DIR, '_fig2_elasticity.png')
fig2.savefig(FIG2, dpi=150, bbox_inches='tight')
plt.close()
print("    ✅ Figure 2: FX Pass-Through & Elasticity")


# ── FIGURE 3: Scenario Pricing (heat map + grouped bar) ─────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(18, 9))
fig3.suptitle('Optimal Price Scenarios Under FX Uncertainty (Top 20 Products)',
               fontsize=16, fontweight='bold', color=C['primary'])
fig3.patch.set_facecolor('white')

top20       = top_products.head(20).copy()
top20['label'] = [f"SKU-{s}" for s in top20['sku_code']]
heatmap_df  = top20.set_index('label')[scenario_cols].copy()
heatmap_df.columns = ['Optim.', 'Base', 'Pessim.', 'Stress']
heatmap_pct = (heatmap_df
               .div(top20.set_index('label')['avg_price_usd_kg'], axis=0)
               .subtract(1)
               .multiply(100))

ax = axes3[0]
sns.heatmap(heatmap_pct, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0,
             linewidths=0.5, ax=ax, cbar_kws={'label': '% Change from Current VWAP'},
             annot_kws={'size': 9})
ax.set_title('Required Price Adjustment (%)\nvs Current Realized VWAP',
              fontweight='bold', fontsize=12)
ax.set_xlabel('FX Scenario', fontsize=10)
ax.set_ylabel('Product SKU',  fontsize=10)
ax.tick_params(axis='y', labelsize=8)

ax     = axes3[1]
top5   = top20.head(5)
x_pos  = np.arange(len(top5))
width  = 0.15
all_v  = [top5['avg_price_usd_kg'].values] + [top5[c].values for c in scenario_cols]
s_lbls = ['Current VWAP', 'Optimistic', 'Base Case', 'Pessimistic', 'Stress']
s_clrs = [C['primary'], C['green'], C['blue'], C['gold'], C['accent']]
for i, (vals, lbl, col) in enumerate(zip(all_v, s_lbls, s_clrs)):
    ax.bar(x_pos + (i - 2) * width, vals, width, label=lbl,
            color=col, alpha=0.85, edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"SKU-{s}" for s in top5['sku_code']], fontsize=10, rotation=20)
ax.set_ylabel('Optimal Price (USD/kg)', fontsize=10)
ax.set_title('Scenario Pricing — Top 5 Products\nOptimal p* per FX Scenario',
              fontweight='bold', fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
FIG3 = os.path.join(CHARTS_DIR, '_fig3_scenarios.png')
fig3.savefig(FIG3, dpi=150, bbox_inches='tight')
plt.close()
print("    ✅ Figure 3: Scenario Pricing Heat Map")


# ── FIGURE 4: Discount & Revenue Efficiency (2×2) ───────────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
fig4.suptitle('Discount & Revenue Efficiency Analysis',
               fontsize=18, fontweight='bold', color=C['primary'])
fig4.patch.set_facecolor('white')

# 4A — Revenue-weighted avg discount rate by country (horizontal bar with error bars)
ax       = axes4[0, 0]
dc = (
    df.groupby('country_en')
      .apply(lambda g: pd.Series({
          'mean':  np.average(g['discount_rate'].fillna(0), weights=g['revenue_toman']),
          'std':   g['discount_rate'].fillna(0).std(),
          'count': len(g),
      }))
      .reset_index()
)
dc       = dc[dc['count'] >= 20].sort_values('mean', ascending=True)
dc_clrs  = [C['green'] if d < 0.15 else C['gold'] if d < 0.30 else C['accent']
             for d in dc['mean']]
ax.barh(dc['country_en'], dc['mean'] * 100, xerr=dc['std'] * 100,
         color=dc_clrs, alpha=0.8, capsize=4, error_kw={'elinewidth': 1.5})
ax.axvline(20, color=C['gray'], ls='--', alpha=0.7, label='20% threshold')
ax.set_xlabel('Revenue-Weighted Avg Discount Rate (%)', fontsize=10)
ax.set_title('Discount Rate by Country\n(Revenue-Weighted)', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)

# 4B — VWAP price trend over time by top-4 countries
ax = axes4[0, 1]
for ctry in country_rev.head(4)['country_en']:
    cd = (
        df[df['country_en'] == ctry]
          .groupby('year_month')
          .agg(
              rev_usd   = ('revenue_usd_tx',  'sum'),
              weight_kg = ('total_weight_kg', 'sum'),
          )
          .reset_index()
    )
    cd['date']      = cd['year_month'].dt.to_timestamp()
    cd['avg_price'] = cd['rev_usd'] / cd['weight_kg']   # ← Realized VWAP per month
    cd = cd[cd['date'] >= '2023-01-01']
    if len(cd) > 3:
        ax.plot(cd['date'], cd['avg_price'], marker='o', ms=4,
                 lw=2, label=ctry, alpha=0.85)
ax.set_title('VWAP Trend (USD/kg) by Country', fontweight='bold', fontsize=12)
ax.set_ylabel('Realized VWAP (USD/kg)', fontsize=10)
ax.legend(fontsize=9)
ax.tick_params(axis='x', rotation=25)

# 4C — Volume vs Price bubble chart (bubble = discount, colour = elasticity)
ax  = axes4[1, 0]
vp  = top_products.head(25)
sc2 = ax.scatter(vp['avg_price_usd_kg'], vp['total_weight_kg'] / 1000,
                  s=vp['avg_discount_rate'] * 2000 + 50,
                  c=vp['elasticity'], cmap='RdYlGn',
                  alpha=0.7, edgecolors='white', lw=0.8)
plt.colorbar(sc2, ax=ax, label='Price Elasticity')
ax.set_xlabel('Realized VWAP (USD/kg)',    fontsize=10)
ax.set_ylabel("Total Volume ('000 kg)",    fontsize=10)
ax.set_title('VWAP vs Volume\n(bubble size = discount rate, colour = elasticity)',
              fontweight='bold', fontsize=11)

# 4D — Pareto / revenue concentration curve
ax       = axes4[1, 1]
sku_rev  = df.groupby('sku_code')['revenue_usd_tx'].sum().sort_values(ascending=False)
cum_pct  = (sku_rev / sku_rev.sum() * 100).cumsum()
n80      = int((cum_pct < 80).sum())
ax.plot(range(1, len(cum_pct) + 1), cum_pct, color=C['primary'], lw=2)
ax.axhline(80, color=C['accent'], ls='--', alpha=0.8, label='80% revenue')
ax.axvline(n80, color=C['gold'],   ls='--', alpha=0.8,
            label=f'Top {n80} SKUs = 80% revenue')
ax.fill_between(range(1, n80 + 2), cum_pct.iloc[:n80 + 1],
                 alpha=0.15, color=C['blue'])
ax.set_xlabel('Number of SKUs (ranked)', fontsize=10)
ax.set_ylabel('Cumulative Revenue (%)',  fontsize=10)
ax.set_title('Revenue Concentration (Pareto)', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
FIG4 = os.path.join(CHARTS_DIR, '_fig4_discount.png')
fig4.savefig(FIG4, dpi=150, bbox_inches='tight')
plt.close()
print("    ✅ Figure 4: Discount & Revenue Efficiency")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — BUILD EXCEL WORKBOOK (5 sheets)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n📄  [9/9] Writing Excel report → {OUTPUT_FILE} ...")

wb = xlsxwriter.Workbook(OUTPUT_FILE)

# ── Shared cell formats ──────────────────────────────────────────────────────
def F(**kwargs):
    """Shorthand: wb.add_format(kwargs)"""
    return wb.add_format(kwargs)

FMT = {
    'title':   F(bold=True, font_size=16, font_color='#1B2A4A',
                  align='center', valign='vcenter'),
    'hdr':     F(bold=True, font_color='white', bg_color='#1B2A4A',
                  align='center', valign='vcenter', border=1, text_wrap=True),
    'subhdr':  F(bold=True, font_color='white', bg_color='#457B9D',
                  align='center', valign='vcenter', border=1),
    'num':     F(num_format='#,##0.00',    border=1, align='right'),
    'usd3':    F(num_format='$#,##0.000',  border=1, align='right'),
    'usd0':    F(num_format='$#,##0',      border=1, align='right'),
    'pct':     F(num_format='0.0%',        border=1, align='right'),
    'int_':    F(num_format='#,##0',       border=1, align='right'),
    'txt':     F(border=1, text_wrap=True),
    'green':   F(num_format='$#,##0.000',  border=1, align='right',
                  bg_color='#D4EDDA', font_color='#155724'),
    'yellow':  F(num_format='$#,##0.000',  border=1, align='right',
                  bg_color='#FFF3CD', font_color='#856404'),
    'red':     F(num_format='$#,##0.000',  border=1, align='right',
                  bg_color='#F8D7DA', font_color='#721C24'),
    'pct_up':  F(num_format='+0.0%;-0.0%', border=1, align='center',
                  bg_color='#D4EDDA', font_color='#155724'),
    'pct_dn':  F(num_format='+0.0%;-0.0%', border=1, align='center',
                  bg_color='#F8D7DA', font_color='#721C24'),
    'kpi_lbl': F(bold=True, font_size=9, align='center', border=1,
                  valign='top', bg_color='#EEF2FF', text_wrap=True,
                  font_color='#1B2A4A'),
    'kpi_val': F(bold=True, font_size=22, align='center', border=1,
                  valign='vcenter', bg_color='#F0F4FF', font_color='#E63946'),
}


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 1 — 📊 Executive Summary
# ─────────────────────────────────────────────────────────────────────────────
ws1 = wb.add_worksheet('📊 Executive Summary')
ws1.set_tab_color('#1B2A4A')
ws1.set_zoom(85)
ws1.set_column('A:A', 3)
ws1.set_column('B:B', 22)
ws1.set_column('C:H', 16)
ws1.set_row(0, 50)
ws1.merge_range('A1:H1', '🍫 EXPORT PRICING INTELLIGENCE REPORT', FMT['title'])

# KPI cards
ws1.merge_range('B3:H3', '⚡ KEY PERFORMANCE INDICATORS', FMT['subhdr'])
ws1.set_row(3, 20)
ws1.set_row(4, 60)

# ── "Avg Price (USD/kg)" KPI uses portfolio Realized VWAP ────────────────────
# portfolio_vwap_usd_kg = Σ(revenue_usd_tx) / Σ(weight_kg) across all transactions.
# This is NOT df['price_per_kg_usd'].mean() which ignores transaction-size weights.
kpis = [
    ('Total Revenue (USD)',        f"${(df['revenue_toman'].sum() / 10 / last_fx) / 1e6:.1f}M"),
    ('Total Volume (M kg)',        f"{df['total_weight_kg'].sum() / 1e6:.2f}M"),
    ('VWAP Price (USD/kg)',        f"${portfolio_vwap_usd_kg:.3f}"),    # ← Realized VWAP
    ('Avg Discount Rate',          f"{portfolio_vwap_discount * 100:.1f}%"),  # ← revenue-weighted
    ('Unique SKUs',                f"{df['sku_code'].nunique()}"),
    ('Export Markets',             f"{df['country_en'].nunique()}"),
    ('Current FX Rate',            f"{last_fx:,.0f} TMN/$"),
]
for i, (lbl, val) in enumerate(kpis):
    col = chr(ord('B') + i)
    ws1.write(f'{col}4', lbl, FMT['kpi_lbl'])
    ws1.write(f'{col}5', val, FMT['kpi_val'])

# Embedded chart images
ws1.merge_range('B7:H7', '📈 ANALYTICAL CHARTS', FMT['subhdr'])
ws1.insert_image('B8',  FIG1, {'x_scale': 0.72, 'y_scale': 0.72})
ws1.insert_image('B38', FIG4, {'x_scale': 0.70, 'y_scale': 0.70})

# GARCH parameter table
ws1.merge_range('B70:H70', '🔬 GARCH(1,1) FX FORECAST SUMMARY', FMT['subhdr'])
garch_rows = [
    ['Parameter',        'Value',                         'Interpretation'],
    ['ω (Constant)',     f'{omega:.6f}',                  'Long-run volatility floor'],
    ['α (ARCH term)',    f'{alpha:.4f}',                  'Sensitivity to recent shocks'],
    ['β (GARCH term)',   f'{beta:.4f}',                   'Volatility persistence'],
    ['α + β',           f'{alpha + beta:.4f}',            '< 1 → stationary (good)'],
    ['Current FX',      f'{last_fx:,.0f} TMN/$',          'Starting point'],
    ['12M P50 Forecast', f'{sim_p50[-1]:,.0f} TMN/$',     'Most likely scenario'],
    ['12M P5 (Bull)',   f'{sim_p5[-1]:,.0f} TMN/$',       'FX appreciation scenario'],
    ['12M P95 (Bear)',  f'{sim_p95[-1]:,.0f} TMN/$',      'FX depreciation stress'],
]
hdr_f  = F(bold=True, bg_color='#457B9D', font_color='white', border=1, align='center')
cell_f = F(border=1, align='left',   bg_color='#F8F9FA')
val_f  = F(border=1, align='center', bg_color='#EEF2FF', bold=True, font_color='#1B2A4A')
for r, row_data in enumerate(garch_rows):
    fmt_r = hdr_f if r == 0 else cell_f
    fmt_v = hdr_f if r == 0 else val_f
    ws1.write(70 + r, 1, row_data[0], fmt_r)
    ws1.write(70 + r, 2, row_data[1], fmt_v)
    ws1.merge_range(70 + r, 3, 70 + r, 7, row_data[2], fmt_r)


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 2 — 🎯 Optimal Prices
# ─────────────────────────────────────────────────────────────────────────────
ws2 = wb.add_worksheet('🎯 Optimal Prices')
ws2.set_tab_color('#E63946')
ws2.set_zoom(90)
ws2.freeze_panes(3, 3)
ws2.set_row(0, 40)
ws2.merge_range('A1:Q1',
    '🎯 OPTIMAL PRICING TABLE — STOCHASTIC LERNER MARKUP MODEL', FMT['title'])

col_headers = [
    'SKU Code', 'Product Group', 'Wt/Carton\n(g)', 'Volume\n(kg)',
    'Transactions', 'Current VWAP\n($/kg)', 'Avg Discount\nRate (%)',
    'Price\nElasticity', 'Optimal\nPass-Through\n(%)', 'Lerner\nMarkup',
    'P* Optimistic\n($/kg)', 'P* Base Case\n($/kg)',
    'P* Pessimistic\n($/kg)', 'P* Stress\n($/kg)',
    'vs Current\nBase (%)', 'vs Current\nStress (%)', 'Pricing Signal',
]
col_widths = [12, 30, 14, 14, 13, 14, 13, 14, 16, 13, 14, 14, 14, 14, 13, 13, 20]
ws2.set_row(1, 40)
ws2.set_row(2, 40)
for c, (h, w) in enumerate(zip(col_headers, col_widths)):
    ws2.write(1, c, h, FMT['hdr'])
    ws2.set_column(c, c, w)

for r_idx, (_, row) in enumerate(top_products.iterrows()):
    rn   = r_idx + 2
    bg   = '#FFFFFF' if r_idx % 2 == 0 else '#F8F9FA'
    tf   = F(border=1, align='left',  bg_color=bg, text_wrap=True)
    nf   = F(num_format='#,##0.00',   border=1, align='right', bg_color=bg)
    pf   = F(num_format='0.0%',       border=1, align='right', bg_color=bg)
    i_f  = F(num_format='#,##0',      border=1, align='right', bg_color=bg)

    opt  = [row.get(c, row['avg_price_usd_kg']) for c in scenario_cols]
    base_p   = opt[1] if len(opt) > 1 else opt[0]
    stress_p = opt[-1]
    d_base   = (base_p   - row['avg_price_usd_kg']) / row['avg_price_usd_kg']
    d_stress = (stress_p - row['avg_price_usd_kg']) / row['avg_price_usd_kg']

    # Pricing signal classification
    if   abs(row['elasticity']) < 1.0:    signal = '🟢 RAISE PRICE'
    elif d_stress > 0.15:                 signal = '🔴 HIGH FX RISK'
    elif abs(row['elasticity']) > 2.5:    signal = '🟡 PRICE SENSITIVE'
    else:                                  signal = '🔵 HOLD / MONITOR'

    def price_cell_fmt(p):
        d = (p - row['avg_price_usd_kg']) / row['avg_price_usd_kg']
        if   d >  0.05: return FMT['green']
        elif d < -0.05: return FMT['red']
        else:           return FMT['yellow']

    ws2.write(rn, 0,  str(row['sku_code']),                          tf)
    ws2.write(rn, 1,  str(row.get('Group4Name', '')),                tf)
    ws2.write(rn, 2,  row.get('net_weight_carton_gr', 0),            i_f)
    ws2.write(rn, 3,  row['total_weight_kg'],                        i_f)
    ws2.write(rn, 4,  row['n_transactions'],                         i_f)
    ws2.write(rn, 5,  row['avg_price_usd_kg'],                       nf)
    ws2.write(rn, 6,  row['avg_discount_rate'],                      pf)
    ws2.write(rn, 7,  row['elasticity'],                             nf)
    ws2.write(rn, 8,  row['optimal_passthrough'],                    pf)
    ws2.write(rn, 9,  row['lerner_markup'],                          nf)
    for i, p in enumerate(opt[:4]):
        ws2.write(rn, 10 + i, p, price_cell_fmt(p))
    ws2.write(rn, 14, d_base,
              F(num_format='+0.0%;-0.0%', border=1, align='center',
                bg_color='#D4EDDA' if d_base   >= 0 else '#F8D7DA',
                font_color='#155724' if d_base  >= 0 else '#721C24'))
    ws2.write(rn, 15, d_stress,
              F(num_format='+0.0%;-0.0%', border=1, align='center',
                bg_color='#D4EDDA' if d_stress >= 0 else '#F8D7DA',
                font_color='#155724' if d_stress >= 0 else '#721C24'))
    ws2.write(rn, 16, signal, tf)


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 3 — 📈 FX Scenarios
# ─────────────────────────────────────────────────────────────────────────────
ws3 = wb.add_worksheet('📈 FX Scenarios')
ws3.set_tab_color('#457B9D')
ws3.set_zoom(90)
ws3.set_row(0, 40)
ws3.merge_range('A1:J1',
    '📈 GARCH(1,1) FX FORECAST — 12-MONTH MONTE CARLO (N=10,000)', FMT['title'])

ws3.insert_image('A3',  FIG2, {'x_scale': 0.78, 'y_scale': 0.78})
ws3.insert_image('A38', FIG3, {'x_scale': 0.75, 'y_scale': 0.75})

# Monthly forecast table (columns L onwards)
ws3.merge_range('L3:T3', '📅 MONTHLY FX FORECAST (12 MONTHS)', FMT['subhdr'])
fx_hdrs = ['Month', 'P5\n(Bull)', 'P25', 'P50\n(Base)', 'P75', 'P95\n(Bear)',
           'Revenue\nImpact %', 'Cost\nImpact %', 'Rec.\nPass-Through']
for c, h in enumerate(fx_hdrs):
    ws3.write(3, 11 + c, h, FMT['hdr'])
    ws3.set_column(11 + c, 11 + c, 14)

for i in range(HORIZON):
    bg       = '#FFFFFF' if i % 2 == 0 else '#F8F9FA'
    rev_imp  = (sim_p50[i] - last_fx) / last_fx
    cost_imp = rev_imp * FX_COST_SHARE
    cf       = F(border=1, align='center',  bg_color=bg)
    nf_fx    = F(num_format='#,##0',        border=1, align='right', bg_color=bg)
    nf_red   = F(num_format='#,##0',        border=1, align='right',
                  bg_color='#F8D7DA', font_color='#721C24')
    nf_grn   = F(num_format='#,##0',        border=1, align='right',
                  bg_color='#D4EDDA', font_color='#155724')
    pct_rev  = F(num_format='+0.0%;-0.0%',  border=1, align='center',
                  bg_color='#F8D7DA' if rev_imp  > 0 else '#D4EDDA',
                  font_color='#721C24' if rev_imp > 0 else '#155724')
    pct_cost = F(num_format='+0.0%;-0.0%',  border=1, align='center',
                  bg_color='#F8D7DA' if cost_imp > 0 else '#D4EDDA',
                  font_color='#721C24' if cost_imp > 0 else '#155724')
    pct_pt   = F(num_format='0.0%',         border=1, align='center',
                  bg_color='#FFF3CD', font_color='#856404')

    ws3.write(4 + i, 11, fx_forecast_dates[i].strftime('%b %Y'), cf)
    ws3.write(4 + i, 12, sim_p5[i],    nf_grn)
    ws3.write(4 + i, 13, sim_p25[i],   nf_fx)
    ws3.write(4 + i, 14, sim_p50[i],   nf_fx)
    ws3.write(4 + i, 15, sim_p75[i],   nf_fx)
    ws3.write(4 + i, 16, sim_p95[i],   nf_red)
    ws3.write(4 + i, 17, rev_imp,      pct_rev)
    ws3.write(4 + i, 18, cost_imp,     pct_cost)
    ws3.write(4 + i, 19, med_pt_val,   pct_pt)


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 4 — 🌍 Country Analysis
# ─────────────────────────────────────────────────────────────────────────────
ws4 = wb.add_worksheet('🌍 Country Analysis')
ws4.set_tab_color('#2A9D8F')
ws4.set_zoom(90)
ws4.set_row(0, 40)
ws4.merge_range('A1:L1', '🌍 PRICING ANALYSIS BY DESTINATION COUNTRY', FMT['title'])

country_agg = (
    df.groupby('country_en')
      .agg(
          total_revenue_usd   = ('revenue_usd_tx',        'sum'),   # ← realized USD
          total_revenue_toman = ('revenue_toman',          'sum'),
          total_weight_kg     = ('total_weight_kg',        'sum'),
          n_transactions      = ('qty_sold',               'count'),
          unique_skus         = ('sku_code',               'nunique'),
          total_gross_revenue = ('gross_revenue_toman',    'sum'),
          total_discount      = ('discount_toman',         'sum'),
      )
      .reset_index()
)

# Realized VWAP per country
country_agg['revenue_usd']      = country_agg['total_revenue_usd']
country_agg['revenue_share']    = country_agg['revenue_usd'] / country_agg['revenue_usd'].sum()
country_agg['avg_price_usd_kg'] = country_agg['revenue_usd'] / country_agg['total_weight_kg']

# Revenue-weighted discount rate
country_agg['avg_discount_rate'] = (
    country_agg['total_discount'] / country_agg['total_gross_revenue']
)

# For std and max discount, we need to go back to grouped data
country_stats = []
for country in country_agg['country_en']:
    cdata = df[df['country_en'] == country]
    # Volume-weighted std of price
    if len(cdata) > 1:
        vw_std = vwstd(cdata['price_per_kg_usd'].values, cdata['total_weight_kg'].values)
    else:
        vw_std = 0
    country_stats.append({
        'country_en': country,
        'std_price_usd_kg': vw_std,
        'max_discount_rate': cdata['discount_rate'].max(),
    })

country_stats_df = pd.DataFrame(country_stats)
country_agg = country_agg.merge(country_stats_df, on='country_en', how='left')
country_agg = country_agg.sort_values('revenue_usd', ascending=False)

c4_hdrs  = ['Country', 'Revenue (USD)', 'Revenue Share', 'Volume (kg)',
             'VWAP ($/kg)', 'Std Dev Price', 'CV Price (%)',
             'Avg Discount', 'Max Discount', 'Transactions', 'Unique SKUs', 'Risk Level']
c4_widths = [20, 16, 14, 14, 16, 16, 13, 14, 14, 14, 13, 14]
ws4.set_row(2, 40)
for c, (h, w) in enumerate(zip(c4_hdrs, c4_widths)):
    ws4.write(2, c, h, FMT['hdr'])
    ws4.set_column(c, c, w)

for r_idx, (_, row) in enumerate(country_agg.iterrows()):
    rn  = r_idx + 3
    bg  = '#FFFFFF' if r_idx % 2 == 0 else '#F8F9FA'
    tf  = F(border=1, bg_color=bg, align='left')
    nf  = F(num_format='$#,##0',      border=1, bg_color=bg, align='right')
    pf  = F(num_format='0.0%',        border=1, bg_color=bg, align='right')
    u3  = F(num_format='$#,##0.000',  border=1, bg_color=bg, align='right')
    i_f = F(num_format='#,##0',       border=1, bg_color=bg, align='right')
    cv  = (row['std_price_usd_kg'] / row['avg_price_usd_kg']
           if row['avg_price_usd_kg'] > 0 else 0)
    risk = ('🔴 HIGH'   if (cv > 0.4 or row['avg_discount_rate'] > 0.35) else
            '🟡 MEDIUM' if (cv > 0.2 or row['avg_discount_rate'] > 0.20) else
            '🟢 LOW')
    rf   = F(border=1, align='center', bold=True,
              bg_color='#F8D7DA' if '🔴' in risk else '#FFF3CD' if '🟡' in risk else '#D4EDDA',
              font_color='#721C24' if '🔴' in risk else '#856404' if '🟡' in risk else '#155724')

    ws4.write(rn,  0, row['country_en'],          tf)
    ws4.write(rn,  1, row['revenue_usd'],          nf)
    ws4.write(rn,  2, row['revenue_share'],        pf)
    ws4.write(rn,  3, row['total_weight_kg'],      i_f)
    ws4.write(rn,  4, row['avg_price_usd_kg'],     u3)
    ws4.write(rn,  5, row['std_price_usd_kg'],     u3)
    ws4.write(rn,  6, cv,                          pf)
    ws4.write(rn,  7, row['avg_discount_rate'],    pf)
    ws4.write(rn,  8, row['max_discount_rate'],    pf)
    ws4.write(rn,  9, row['n_transactions'],       i_f)
    ws4.write(rn, 10, row['unique_skus'],          i_f)
    ws4.write(rn, 11, risk,                        rf)


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 5 — 📐 Methodology
# ─────────────────────────────────────────────────────────────────────────────
ws5 = wb.add_worksheet('📐 Methodology')
ws5.set_tab_color('#7B2D8B')
ws5.set_zoom(95)
ws5.set_column('A:A', 4)
ws5.set_column('B:B', 28)
ws5.set_column('C:C', 30)
ws5.set_column('D:D', 75)
ws5.set_row(0, 40)
ws5.merge_range('A1:D1', '📐 MODEL METHODOLOGY & MATHEMATICAL FRAMEWORK', FMT['title'])

methodology_rows = [
    # (Section, Component, Formula / Description)
    ('SECTION', 'COMPONENT', 'FORMULA / DESCRIPTION'),
    ('0. VWAP METHOD', 'Volume-Weighted Average Price',
     'ALL price metrics use Realized VWAP = Σ(revenue_usd_i) / Σ(weight_kg_i)\n'
     'where revenue_usd_i = revenue_toman_i / fx_rate_i  (each tx at its own FX rate).\n'
     'Simple mean() treats all transactions equally — WRONG for revenue analysis.\n'
     'An FX-normalized VWAP (all Toman at last_fx) is also computed for trend comparison.'),
    ('1. COST MODEL',    'FX-Linked Cost Function',
     'C(t) = α·FX(t)·C_usd + β·W(t) + C_fixed\n'
     'α = FX-exposed cost share (~60%), β = labour share, W = wage index'),
    ('2. FX DYNAMICS',   'GARCH(1,1)',
     f'σ²_t = ω + α·ε²_(t-1) + β·σ²_(t-1)\n'
     f'Fitted: ω={omega:.5f}, α={alpha:.4f}, β={beta:.4f}\n'
     f'Persistence: α+β={alpha+beta:.4f}  (<1 ✓ stationary)'),
    ('3. SIMULATION',    'Monte Carlo (N=10,000 paths)',
     f'12-month FX: P5={sim_p5[-1]:,.0f} | P50={sim_p50[-1]:,.0f} | P95={sim_p95[-1]:,.0f} TMN/$\n'
     'Each path propagated via GARCH variance recursion'),
    ('4. DEMAND MODEL',  'Log-Linear OLS Panel (with Realized VWAP)',
     'log(D_it) = a + ε·log(VWAP_p_it) + γ·log(FX_t) + u_it\n'
     f'Portfolio median ε = {median_elas:.2f}   (estimated per product group)\n'
     'Realized VWAP used: Σ(rev_usd_tx) / Σ(weight_kg) per group-month.'),
    ('5. OPTIMAL PRICE', 'Stochastic Lerner Markup',
     'p*(t) = E[C(FX_t)] · |ε|/(|ε|−1)  +  λ·Var[C(FX_t)]\n'
     'Lerner markup = |ε|/(|ε|−1);  λ = risk aversion (default 0.3)\n'
     'Floor: −30% from current Realized VWAP'),
    ('6. PASS-THROUGH',  'Optimal FX Pass-Through ρ*',
     f'ρ* = 1 / (1 + 1/|ε|)\n'
     f'Portfolio median ρ* = {med_pt_val*100:.1f}%  — do NOT pass 100% of FX shock'),
    ('7. PRICE SIGNALS', 'Classification Rules',
     '🟢 RAISE PRICE     → |ε| < 1  (inelastic, customers price-insensitive)\n'
     '🔴 HIGH FX RISK    → stress price > 15% above current VWAP\n'
     '🟡 PRICE SENSITIVE → |ε| > 2.5 (elastic, adjust cautiously)\n'
     '🔵 HOLD / MONITOR  → all other cases; review quarterly'),
    ('8. UPDATE TRIGGER', 'Dynamic Re-Pricing Rule',
     'Update price when FX moves > 5% from last pricing date\n'
     'Formula: new_p = old_p × (1 + ρ* × Δ%FX)\n'
     'Validate vs demand signal before publishing to market'),
]

hdr_f  = F(bold=True, border=1, bg_color='#457B9D', font_color='white',
            align='center', valign='vcenter')
sec_f  = F(bold=True, border=1, bg_color='#1B2A4A', font_color='white',
            align='center', valign='vcenter', text_wrap=True, font_size=10)
comp_f = F(bold=True, border=1, bg_color='#EEF2FF', font_color='#1B2A4A',
            align='left', valign='top', text_wrap=True)
desc_f = F(border=1, bg_color='#F8F9FA', align='left', valign='top',
            text_wrap=True, font_size=9)

for r_idx, (sec, comp, desc) in enumerate(methodology_rows):
    rn = r_idx + 2
    ws5.set_row(rn, 20 if r_idx == 0 else 70)
    is_hdr = (r_idx == 0)
    ws5.write(rn, 1, sec,  hdr_f  if is_hdr else sec_f)
    ws5.write(rn, 2, comp, hdr_f  if is_hdr else comp_f)
    ws5.write(rn, 3, desc, hdr_f  if is_hdr else desc_f)


# ── Close workbook ────────────────────────────────────────────────────────────
wb.close()

# Clean up temporary chart files
for fp in [FIG1, FIG2, FIG3, FIG4]:
    try:
        os.remove(fp)
    except OSError:
        pass

print(f"\n{'='*70}")
print(f"  ✅  PIPELINE COMPLETE")
print(f"  Output : {os.path.abspath(OUTPUT_FILE)}")
print(f"{'='*70}")
print("\n  Sheets written:")
print("    📊 Executive Summary  — KPI cards + 4 charts + GARCH table")
print("    🎯 Optimal Prices     — 50 SKUs × 4 FX scenarios + pricing signal")
print("    📈 FX Scenarios       — 12-month GARCH fan chart + monthly table")
print("    🌍 Country Analysis   — Risk-rated pricing per destination market")
print("    📐 Methodology        — Full mathematical framework")
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DATAFRAME RESULTS FOR NOTEBOOK / INTERACTIVE USE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  📊  DATAFRAME RESULTS SUMMARY")
print("="*70)

# ── Dictionary to hold all results ────────────────────────────────────────────
results = {}

# ── 1. GARCH Parameters & FX Forecasts ────────────────────────────────────────
results['garch_params'] = pd.DataFrame({
    'Parameter': ['omega (ω)', 'alpha (α)', 'beta (β)', 'mu (μ)', 'Persistence (α+β)'],
    'Value': [omega, alpha, beta, mu, alpha + beta],
    'Interpretation': [
        'Long-run volatility floor',
        'Sensitivity to recent shocks (ARCH term)',
        'Volatility persistence (GARCH term)',
        'Mean log return (%)',
        'Stationarity check (must be < 1)'
    ]
})

results['fx_current'] = pd.DataFrame({
    'Metric': ['Current FX Rate', 'Last Month FX', 'FX Change (1M %)', 'Min (Historical)', 'Max (Historical)'],
    'Value': [
        f"{last_fx:,.0f} TMN/$",
        f"{fx_monthly['fx_rate'].iloc[-2]:,.0f} TMN/$" if len(fx_monthly) > 1 else "N/A",
        f"{((last_fx / fx_monthly['fx_rate'].iloc[-2] - 1) * 100):.1f}%" if len(fx_monthly) > 1 else "N/A",
        f"{fx_monthly['fx_rate'].min():,.0f} TMN/$",
        f"{fx_monthly['fx_rate'].max():,.0f} TMN/$"
    ]
})

results['fx_forecast_12m'] = pd.DataFrame({
    'Month': [d.strftime('%b %Y') for d in fx_forecast_dates],
    'P5_Bull': sim_p5,
    'P25': sim_p25,
    'P50_Base': sim_p50,
    'P75': sim_p75,
    'P95_Bear': sim_p95,
    'Rev_Impact_%': [(sim_p50[i] - last_fx) / last_fx * 100 for i in range(HORIZON)],
    'Cost_Impact_%': [(sim_p50[i] - last_fx) / last_fx * FX_COST_SHARE * 100 for i in range(HORIZON)],
    'Rec_PassThrough_%': [med_pt_val * 100] * HORIZON,
})

# ── 2. Price Elasticity Results ───────────────────────────────────────────────
results['elasticity_by_group'] = elasticity_df.copy()
results['elasticity_by_group']['abs_elasticity'] = results['elasticity_by_group']['elasticity'].abs()
results['elasticity_by_group']['classification'] = results['elasticity_by_group']['elasticity'].apply(
    lambda e: 'Inelastic (|ε|<1)' if abs(e) < 1 else
              'Moderate (1≤|ε|<2)' if abs(e) < 2 else
              'Elastic (2≤|ε|<2.5)' if abs(e) < 2.5 else
              'Highly Elastic (|ε|≥2.5)'
)
results['elasticity_by_group']['optimal_passthrough_%'] = (
    1 / (1 + 1 / results['elasticity_by_group']['abs_elasticity'])
) * 100
results['elasticity_by_group'] = results['elasticity_by_group'].sort_values('elasticity')

results['elasticity_summary'] = pd.DataFrame({
    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
    'Elasticity': [
        len(elasticity_df),
        elasticity_df['elasticity'].mean(),
        elasticity_df['elasticity'].median(),
        elasticity_df['elasticity'].std(),
        elasticity_df['elasticity'].min(),
        elasticity_df['elasticity'].max(),
        elasticity_df['elasticity'].quantile(0.25),
        elasticity_df['elasticity'].quantile(0.75),
    ]
})

# ── 3. Optimal Pricing Results ────────────────────────────────────────────────
results['optimal_prices'] = top_products.copy()

# Add pricing change % columns
for sc in scenario_cols:
    scenario_short = sc.replace('opt_price_', '')
    results['optimal_prices'][f'change_{scenario_short}_%'] = (
        (results['optimal_prices'][sc] - results['optimal_prices']['avg_price_usd_kg'])
        / results['optimal_prices']['avg_price_usd_kg'] * 100
    )

# Add pricing signal
def classify_signal(row):
    stress_p = row[scenario_cols[-1]]
    d_stress = (stress_p - row['avg_price_usd_kg']) / row['avg_price_usd_kg']
    if abs(row['elasticity']) < 1.0:
        return '🟢 RAISE PRICE'
    elif d_stress > 0.15:
        return '🔴 HIGH FX RISK'
    elif abs(row['elasticity']) > 2.5:
        return '🟡 PRICE SENSITIVE'
    else:
        return '🔵 HOLD / MONITOR'

results['optimal_prices']['pricing_signal'] = results['optimal_prices'].apply(classify_signal, axis=1)

# Key columns for display
opt_price_cols = [
    'sku_code', 'product_name', 'Group4Name', 'total_weight_kg',
    'avg_price_usd_kg', 'elasticity', 'optimal_passthrough', 'lerner_markup'
] + scenario_cols + [f'change_{sc.replace("opt_price_", "")}_%' for sc in scenario_cols] + ['pricing_signal']

results['optimal_prices_display'] = results['optimal_prices'][
    [c for c in opt_price_cols if c in results['optimal_prices'].columns]
].head(30)

# ── 4. Monthly Revenue & Price Analytics ──────────────────────────────────────
results['monthly_analytics'] = monthly.copy()
results['monthly_analytics']['revenue_usd_millions'] = results['monthly_analytics']['revenue_usd'] / 1e6
results['monthly_analytics']['yoy_growth_%'] = (
    results['monthly_analytics']['revenue_usd'].pct_change(12) * 100
)
results['monthly_analytics']['mom_growth_%'] = (
    results['monthly_analytics']['revenue_usd'].pct_change(1) * 100
)

# ── 5. Country Performance Analysis ───────────────────────────────────────────
results['country_performance'] = country_agg.copy()
results['country_performance']['cv_price'] = (
    results['country_performance']['std_price_usd_kg']
    / results['country_performance']['avg_price_usd_kg']
)
results['country_performance']['pricing_method'] = 'Realized VWAP (Σ rev_usd_tx / Σ weight_kg)'
results['country_performance']['risk_level'] = results['country_performance'].apply(
    lambda r: '🔴 HIGH' if (r['cv_price'] > 0.4 or r['avg_discount_rate'] > 0.35) else
              '🟡 MEDIUM' if (r['cv_price'] > 0.2 or r['avg_discount_rate'] > 0.20) else
              '🟢 LOW',
    axis=1
)

# ── 6. Product Performance Rankings ───────────────────────────────────────────
# Both revenue-ranked and volume-ranked tables use Realized VWAP:
#   avg_price_usd_kg = Σ(revenue_usd_tx) / Σ(weight_kg)  per SKU
_sku_agg = (
    df.groupby(['sku_code', 'product_name'])
      .agg(
          revenue_usd  = ('revenue_usd_tx',  'sum'),
          volume_kg    = ('total_weight_kg',  'sum'),
          transactions = ('qty_sold',         'count'),
      )
      .reset_index()
)
_sku_agg['avg_price_usd_kg'] = _sku_agg['revenue_usd'] / _sku_agg['volume_kg']

results['top_products_revenue'] = (
    _sku_agg
    .sort_values('revenue_usd', ascending=False)
    .head(20)
    .copy()
)
results['top_products_revenue']['revenue_share_%'] = (
    results['top_products_revenue']['revenue_usd']
    / results['top_products_revenue']['revenue_usd'].sum() * 100
)

results['top_products_volume'] = (
    _sku_agg
    .sort_values('volume_kg', ascending=False)
    .head(20)
    .copy()
)

# ── 7. Key Business Metrics Summary ──────────────────────────────────────────
total_rev_usd = df['revenue_toman'].sum() / 10 / last_fx

results['business_kpis'] = pd.DataFrame({
    'KPI': [
        'Total Revenue (USD)',
        'Total Volume (kg)',
        'VWAP Price (USD/kg)',       # ← renamed to make metric explicit
        'Avg Discount Rate',
        'Unique SKUs',
        'Export Markets',
        'Transactions',
        'Avg Transaction Size (kg)',
        'Revenue per SKU (USD)',
        'Price Volatility (CV)',
    ],
    'Value': [
        f"${total_rev_usd/1e6:.2f}M",
        f"{df['total_weight_kg'].sum()/1e3:,.0f}K kg",
        f"${portfolio_vwap_usd_kg:.3f}",          # ← Realized VWAP (not simple mean)
        f"{portfolio_vwap_discount * 100:.1f}%",   # ← revenue-weighted discount rate
        f"{df['sku_code'].nunique()}",
        f"{df['country_en'].nunique()}",
        f"{len(df):,}",
        f"{df['total_weight_kg'].mean():.1f} kg",
        f"${total_rev_usd / df['sku_code'].nunique():,.0f}",
        f"{portfolio_price_cv:.2%}",              # ← volume-weighted CV
    ]
})

# ── 7b. VWAP vs Simple Mean Comparison (Show why this matters!) ──────────────
vwap_comparison = []
for (sku, pname), grp in df.groupby(['sku_code', 'product_name']):
    simple_mean   = grp['price_per_kg_usd'].mean()
    total_rev_usd_grp = grp['revenue_usd_tx'].sum()
    total_wt      = grp['total_weight_kg'].sum()
    vwap_price    = total_rev_usd_grp / total_wt if total_wt > 0 else 0

    if simple_mean > 0 and vwap_price > 0:
        vwap_comparison.append({
            'sku_code': sku,
            'product_name': pname,
            'total_weight_kg': total_wt,
            'simple_mean_price': simple_mean,
            'vwap_price': vwap_price,
            'difference_$': vwap_price - simple_mean,
            'difference_%': (vwap_price / simple_mean - 1) * 100,
            'n_transactions': len(grp),
        })

results['vwap_vs_mean_comparison'] = (
    pd.DataFrame(vwap_comparison)
    .sort_values('total_weight_kg', ascending=False)
    .head(30)
)

# Summary statistics of the difference
vwap_comp_df = pd.DataFrame(vwap_comparison)
results['vwap_impact_summary'] = pd.DataFrame({
    'Metric': [
        'Products where VWAP > Simple Mean',
        'Products where VWAP < Simple Mean',
        'Max Overestimation by Simple Mean',
        'Max Underestimation by Simple Mean',
        'Median Absolute Difference',
        'Mean Absolute % Difference',
    ],
    'Value': [
        f"{(vwap_comp_df['difference_%'] > 0).sum()} ({(vwap_comp_df['difference_%'] > 0).mean()*100:.1f}%)",
        f"{(vwap_comp_df['difference_%'] < 0).sum()} ({(vwap_comp_df['difference_%'] < 0).mean()*100:.1f}%)",
        f"${vwap_comp_df['difference_$'].min():.3f} ({vwap_comp_df['difference_%'].min():.1f}%)",
        f"${vwap_comp_df['difference_$'].max():.3f} (+{vwap_comp_df['difference_%'].max():.1f}%)",
        f"${vwap_comp_df['difference_$'].abs().median():.3f}",
        f"{vwap_comp_df['difference_%'].abs().mean():.1f}%",
    ]
})

# ── 8. Pricing Action Recommendations ────────────────────────────────────────
signal_counts = results['optimal_prices']['pricing_signal'].value_counts()
results['pricing_actions'] = pd.DataFrame({
    'Signal': signal_counts.index,
    'Count': signal_counts.values,
    'Share_%': (signal_counts / signal_counts.sum() * 100).values,
    'Recommended_Action': [
        'Increase prices — demand is inelastic' if '🟢' in s else
        'Stress test pricing under worst-case FX' if '🔴' in s else
        'Monitor closely, adjust cautiously' if '🟡' in s else
        'Maintain current pricing, review quarterly'
        for s in signal_counts.index
    ]
})


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS (works in both script and notebook)
# ══════════════════════════════════════════════════════════════════════════════

def display_section(title, df, max_rows=10):
    """Display a dataframe section with a title"""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"\n  ... ({len(df) - max_rows} more rows)")
    else:
        print(df.to_string(index=False))

# Display key results
display_section("1️⃣  GARCH(1,1) PARAMETERS", results['garch_params'])
display_section("2️⃣  CURRENT FX STATUS", results['fx_current'])
display_section("3️⃣  12-MONTH FX FORECAST (first 6 months)", results['fx_forecast_12m'].head(6))
display_section("4️⃣  ELASTICITY SUMMARY", results['elasticity_summary'])
display_section("5️⃣  TOP 10 PRICE ELASTICITIES BY GROUP",
                results['elasticity_by_group'].head(10)[[
                    'Group4Name', 'elasticity', 'classification', 'optimal_passthrough_%', 'n_obs'
                ]])
display_section("6️⃣  BUSINESS KPIs", results['business_kpis'])
display_section("7️⃣  PRICING ACTION SIGNALS", results['pricing_actions'])
display_section("8️⃣  OPTIMAL PRICES — TOP 10 PRODUCTS",
                results['optimal_prices_display'].head(10)[[
                    'sku_code', 'avg_price_usd_kg', 'elasticity',
                    'opt_price_Base', 'change_Base_%', 'pricing_signal'
                ]])
display_section("9️⃣  TOP 10 REVENUE PRODUCTS", results['top_products_revenue'].head(10))
display_section("🔟  COUNTRY PERFORMANCE — TOP 10",
                results['country_performance'].head(10)[[
                    'country_en', 'revenue_usd', 'revenue_share',
                    'avg_price_usd_kg', 'avg_discount_rate', 'risk_level'
                ]])
display_section("1️⃣1️⃣  VWAP vs SIMPLE MEAN — TOP 15 PRODUCTS",
                results['vwap_vs_mean_comparison'].head(15)[[
                    'sku_code', 'total_weight_kg', 'simple_mean_price',
                    'vwap_price', 'difference_%', 'n_transactions'
                ]])
display_section("1️⃣2️⃣  WHY VWAP MATTERS — IMPACT SUMMARY",
                results['vwap_impact_summary'])

print(f"\n{'='*70}")
print(f"  📊  All results stored in 'results' dictionary")
print(f"{'='*70}")
print("\n  Available DataFrames:")
for key in results.keys():
    print(f"    • results['{key}']  →  shape {results[key].shape}")
print()
print("  ✅  VWAP POLICY: All price metrics use Realized VWAP consistently:")
print("     → Realized VWAP  = Σ(revenue_usd_tx) / Σ(weight_kg)  [at each tx's own FX rate]")
print("     → Simple mean of price_per_kg_usd is NEVER used for any reported metric.")
print("     → FX-normalized VWAP (all Toman at last_fx) available in top_products")
print("        as 'vwap_price_fx_normalized' for period-over-period trend analysis.")
print()

# ── For Jupyter notebook usage ───────────────────────────────────────────────
# If running in notebook, you can access any result via the results dict:
#
#   results['optimal_prices']              # Full optimal pricing table
#   results['fx_forecast_12m']             # 12-month FX forecast scenarios
#   results['elasticity_by_group']         # Price elasticities per product group
#   results['monthly_analytics']           # Monthly revenue time series
#   results['country_performance']         # Country-level performance metrics
#   results['business_kpis']               # High-level KPI summary
#   results['pricing_actions']             # Recommended pricing actions
#   ... and more
#
# Example notebook cell:
#   import pandas as pd
#   pd.options.display.max_columns = None
#   results['optimal_prices'].head(20)
