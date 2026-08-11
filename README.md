#  OPTIMAL EXPORT PRICING PIPELINE UNDER FX UNCERTAINTY
    Author  : Ali Pira

    Purpose : Full stochastic pricing model for confectionery export products

    Model   : GARCH(1,1) FX forecasting + Stochastic Lerner Markup Optimization
            + Monte Carlo Simulation (10,000 paths) + Price Elasticity OLS

    Output  : Optimal_Pricing_Model_Results.xlsx  (5-sheet Excel report)

###  Sheets  :
    1. 📊 Executive Summary   — 7 KPI cards + 4 analytical charts + GARCH table
    2. 🎯 Optimal Prices      — Top 50 SKUs, color-coded 4-scenario optimal prices
    3. 📈 FX Scenarios        — 12-month GARCH Monte Carlo table + elasticity charts
    4. 🌍 Country Analysis    — Risk-rated pricing per destination market
    5. 📐 Methodology         — Full mathematical framework documentation

###  Mathematical Core:
    - Cost Model   : C(t) = α·FX(t)·C_usd + β·W(t) + C_fixed
    - FX Dynamics  : GARCH(1,1) → σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    - Demand Model : log(D) = a − ε·log(p) + γ·log(FX)  [OLS panel]
    - Optimal Price: p*(t) = E[C(FX_t)] · |ε|/(|ε|−1) + λ·Var[C(FX_t)]
    - Pass-Through : ρ* = 1 / (1 + 1/|ε|)

###  VWAP Policy (applied consistently throughout):
    - ALL "Current Avg Price" metrics use Volume-Weighted Average Price.
    - VWAP = Σ(revenue_usd_i) / Σ(weight_kg_i)  where revenue_usd_i
      = revenue_toman_i / fx_rate_i  (each transaction at its own FX rate).
    - This is the "Realized VWAP" — what was actually received in USD per kg.
    - A second flavour, "FX-normalized VWAP", restates all Toman revenue at
      last_fx for period-over-period comparability; labelled explicitly where used.
    - Simple mean of per-row price_per_kg_usd is NEVER used for reporting
      because it gives equal weight to a 10-kg and a 10,000-kg transaction.


###  Usage:
    1. Set INPUT_FILE to your Excel file path
    2. Set OUTPUT_FILE to desired output path
    3. Run:  python pricing_pipeline.py