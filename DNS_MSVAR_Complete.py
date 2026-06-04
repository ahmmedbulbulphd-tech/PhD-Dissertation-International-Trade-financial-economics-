"""
================================================================================
COMPLETE SIMULATION CODE
Dynamic Nelson-Siegel (DNS) Model  +  Markov-Switching VAR (MS-VAR) Model
================================================================================
Dataset : Bangladesh Yield Curve & Macro Monthly Data (Apr 2015 – Jan 2026)
Maturities: 3M, 6M, 1Y, 2Y, 5Y, 10Y, 15Y, 20Y
Macro vars: Inflation, PolicyRate

SIMULATION PIPELINE
──────────────────────────────────────────────────────────────────────────────
PART A – Dynamic Nelson-Siegel (DNS)
  A1.  Load & validate data
  A2.  Nelson-Siegel factor loadings  B(τ, λ)
  A3.  λ optimisation  (minimise cross-sectional RMSE)
  A4.  OLS factor extraction  (Level β₁, Slope β₂, Curvature β₃)
  A5.  Factor descriptive statistics
  A6.  AR(1) univariate factor dynamics
  A7.  VAR(p) joint factor dynamics  (Diebold-Li)
  A8.  DNS-VAR with macro variables  (Diebold-Rudebusch-Aruoba)
  A9.  IRF & FEVD from DNS-VAR
  A10. Yield-curve forecasting (h = 1, 3, 6, 12 months)
  A11. Figures  (factor loadings, time-series, actual-vs-fitted,
                 3-D surface, IRF, residuals)
  A12. Tables   (all CSV exports)

PART B – Markov-Switching VAR (MS-VAR)
  B1.  Variable selection & stationarity (ADF / KPSS)
  B2.  Lag-order selection (AIC, BIC, HQIC, FPE)
  B3.  Baseline VAR(p) estimation
  B4.  EM algorithm for MS-VAR (Hamilton 1989 / Kim 1994)
        – regime-specific means, covariances, transition matrix
        – smoothed & filtered regime probabilities
  B5.  Regime characterisation (duration, time-share, dating)
  B6.  Regime-conditional IRF
  B7.  Regime-conditional FEVD
  B8.  Residual diagnostics
  B9.  Figures  (regime probabilities, timeline, regime-specific
                 yield curves, IRF comparison, FEVD)
  B10. Tables   (all CSV exports)

PART C – Combined output
  C1.  DNS factors ↔ MS-VAR regime overlay
  C2.  Policy-transmission summary table
  C3.  Markdown / Word report
================================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from sklearn.cluster import KMeans
import subprocess

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = r'C:\Users\KCC\OneDrive\Desktop\phd_MSVAR.PY\Bangladesh_YieldCurve_Macro_Master_Monthly_2015_2026.csv'
OUT       = r'C:\Users\KCC\OneDrive\Desktop\phd_MSVAR.PY\DNS_MSVAR_Results'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi'       : 150,
    'font.size'        : 9,
    'axes.titlesize'   : 10,
    'axes.labelsize'   : 9,
    'legend.fontsize'  : 8,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.family'      : 'DejaVu Sans',
})

C = {                              # colour palette
    'level'    : '#2166ac',
    'slope'    : '#d6604d',
    'curv'     : '#4dac26',
    'actual'   : '#333333',
    'fitted'   : '#e31a1c',
    'policy'   : '#1f78b4',
    'inflation': '#ff7f00',
    'r0'       : '#1a9641',        # regime 0 (stable)
    'r1'       : '#d7191c',        # regime 1 (volatile)
}

MATURITIES_LABELS = ['Y3M','Y6M','Y1Y','Y2Y','Y5Y','Y10Y','Y15Y','Y20Y']
TAU               = np.array([3/12, 6/12, 1, 2, 5, 10, 15, 20])
VAR_COLS          = ['Y3M','Y2Y','Y5Y','Y10Y','Inflation','PolicyRate']

def sep(title='', w=70):
    print('\n' + '='*w)
    if title:
        print(f'  {title}')
        print('='*w)

# ================================================================================
# ██████   █████  ██████  ████████      █████
# ██   ██ ██   ██ ██   ██    ██        ██   ██
# ██████  ███████ ██████     ██        ███████
# ██      ██   ██ ██   ██    ██        ██   ██
# ██      ██   ██ ██   ██    ██        ██   ██
# DYNAMIC NELSON-SIEGEL MODEL
# ================================================================================

sep('PART A — DYNAMIC NELSON-SIEGEL (DNS) MODEL')

# ─────────────────────────────────────────────────────────────────────────────
# A1. Load data
# ─────────────────────────────────────────────────────────────────────────────
sep('A1 · Loading & Validating Data')
import os
print("DATA_PATH =", DATA_PATH)
print("File exists?", os.path.exists(DATA_PATH))
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.apply(pd.to_numeric, errors='coerce').dropna()
df.sort_index(inplace=True)

yields = df[MATURITIES_LABELS].values          # T × 8
dates  = df.index
T      = len(dates)

print(f'  Sample : {dates[0].strftime("%b %Y")} – {dates[-1].strftime("%b %Y")}')
print(f'  T      : {T} monthly observations')
print(f'  τ (yrs): {TAU}')
print(f'  No missing values: {df.isnull().sum().sum() == 0}')

# ─────────────────────────────────────────────────────────────────────────────
# A2. Nelson-Siegel factor loadings
# ─────────────────────────────────────────────────────────────────────────────
sep('A2 · Nelson-Siegel Factor Loadings  B(τ, λ)')

def ns_loadings(tau, lam):
    """
    Returns (n_tau × 3) loading matrix.
      Column 0 : Level   = 1
      Column 1 : Slope   = (1 − e^{−λτ}) / (λτ)
      Column 2 : Curvature = (1 − e^{−λτ}) / (λτ) − e^{−λτ}
    """
    et = np.exp(-lam * tau)
    L  = np.ones(len(tau))
    S  = (1 - et) / (lam * tau)
    Cv = (1 - et) / (lam * tau) - et
    return np.column_stack([L, S, Cv])

print('  Nelson-Siegel cross-sectional model:')
print('  y(τ) = β₁ + β₂·[(1−e^{−λτ})/(λτ)] + β₃·[(1−e^{−λτ})/(λτ) − e^{−λτ}]')
print('  β₁ = Level  (long-end factor)')
print('  β₂ = Slope  (short-end factor, β₁+β₂ → short rate)')
print('  β₃ = Curvature (medium-term hump)')

# ─────────────────────────────────────────────────────────────────────────────
# A3. Lambda optimisation
# ─────────────────────────────────────────────────────────────────────────────
sep('A3 · λ Optimisation (minimise cross-sectional RMSE)')

def cross_sec_rmse(lam):
    if lam <= 0:
        return 1e9
    B = ns_loadings(TAU, lam)
    rmses = []
    for t in range(T):
        y    = yields[t]
        beta, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
        yhat = B @ beta
        rmses.append(np.sqrt(np.mean((y - yhat)**2)))
    return np.mean(rmses)

res     = minimize_scalar(cross_sec_rmse, bounds=(0.01, 5.0), method='bounded')
lam_opt = res.x
lam_dl  = 0.0609          # Diebold-Li (2006) benchmark

print(f'  Optimal λ (data)       = {lam_opt:.6f}')
print(f'  Diebold-Li (2006) λ   = {lam_dl:.4f}')
print(f'  Using data-optimal λ  = {lam_opt:.6f}')

B_opt = ns_loadings(TAU, lam_opt)

# ─────────────────────────────────────────────────────────────────────────────
# A4. OLS factor extraction
# ─────────────────────────────────────────────────────────────────────────────
sep('A4 · OLS Factor Extraction (Level, Slope, Curvature)')

betas   = np.zeros((T, 3))
yhats   = np.zeros_like(yields)
resids  = np.zeros_like(yields)

for t in range(T):
    y       = yields[t]
    b, _, _, _ = np.linalg.lstsq(B_opt, y, rcond=None)
    betas[t] = b
    yhats[t] = B_opt @ b
    resids[t] = y - yhats[t]

# Organise into DataFrame
factors = pd.DataFrame(betas, index=dates, columns=['Level','Slope','Curvature'])

# Fit per maturity
rmse_per_mat = np.sqrt(np.mean(resids**2, axis=0))
mae_per_mat  = np.mean(np.abs(resids),    axis=0)
ss_res       = np.sum(resids**2,          axis=0)
ss_tot       = np.sum((yields - yields.mean(axis=0))**2, axis=0)
r2_per_mat   = 1 - ss_res / ss_tot

overall_rmse = np.sqrt(np.mean(resids**2))
overall_mae  = np.mean(np.abs(resids))
overall_r2   = np.mean(r2_per_mat)

print(f'  Overall RMSE : {overall_rmse*100:.4f} bps')
print(f'  Overall MAE  : {overall_mae*100:.4f} bps')
print(f'  Mean R²      : {overall_r2:.6f}')

# CSV: factor loadings
df_load = pd.DataFrame({
    'Maturity_yrs': TAU,
    'Maturity_label': MATURITIES_LABELS,
    'Level_loading': B_opt[:,0],
    'Slope_loading': B_opt[:,1],
    'Curv_loading' : B_opt[:,2],
    'RMSE_bps'     : rmse_per_mat * 100,
    'MAE_bps'      : mae_per_mat  * 100,
    'R_squared'    : r2_per_mat,
})
df_load.to_csv(f'{OUT}/dns_factor_loadings.csv', index=False)

# CSV: factor time-series
factors.to_csv(f'{OUT}/dns_factors.csv')

# CSV: fit per maturity
df_load.to_csv(f'{OUT}/dns_fit_per_maturity.csv', index=False)

print(f'\n  Factor loadings & fit statistics saved.')

# ─────────────────────────────────────────────────────────────────────────────
# A5. Factor descriptive statistics
# ─────────────────────────────────────────────────────────────────────────────
sep('A5 · Factor Descriptive Statistics')

desc = factors.describe().T
desc['skewness']  = factors.skew()
desc['kurtosis']  = factors.kurt()
desc['autocorr1'] = factors.apply(lambda x: x.autocorr(lag=1))
print(desc.round(4))
desc.to_csv(f'{OUT}/dns_factor_descriptive.csv')

# ─────────────────────────────────────────────────────────────────────────────
# A6. AR(1) univariate factor dynamics
# ─────────────────────────────────────────────────────────────────────────────
sep('A6 · AR(1) Univariate Factor Dynamics')

ar1_results = {}
for col in ['Level','Slope','Curvature']:
    y   = factors[col].values
    y_l = y[:-1];  y_c = y[1:]
    X   = np.column_stack([np.ones(len(y_l)), y_l])
    b   = np.linalg.lstsq(X, y_c, rcond=None)[0]
    yhat_ar = X @ b
    res_ar  = y_c - yhat_ar
    sigma2  = np.var(res_ar, ddof=2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se      = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stat  = b / se
    r2_ar   = 1 - np.var(res_ar) / np.var(y_c)
    ar1_results[col] = {
        'mu'    : b[0],  'mu_se'    : se[0],  'mu_t'    : t_stat[0],
        'phi'   : b[1],  'phi_se'   : se[1],  'phi_t'   : t_stat[1],
        'sigma' : np.sqrt(sigma2),
        'R2'    : r2_ar,
        'DW'    : durbin_watson(res_ar),
    }
    print(f'  {col:10s}: μ={b[0]:7.4f}({se[0]:.4f}), φ={b[1]:7.4f}({se[1]:.4f}), '
          f'σ={np.sqrt(sigma2):.4f}, R²={r2_ar:.4f}')

ar1_df = pd.DataFrame(ar1_results).T
ar1_df.to_csv(f'{OUT}/dns_ar1_dynamics.csv')

# ─────────────────────────────────────────────────────────────────────────────
# A7. VAR(p) joint factor dynamics
# ─────────────────────────────────────────────────────────────────────────────
sep('A7 · VAR(p) Joint Factor Dynamics (Diebold-Li)')

var_factors = VAR(factors)
lag_sel     = var_factors.select_order(maxlags=12)
p_sel       = lag_sel.aic
print(f'  AIC-optimal lag : p = {p_sel}')
print(f'  BIC-optimal lag : p = {lag_sel.bic}')

lag_df = pd.DataFrame({
    'Criterion': ['AIC','BIC','HQIC','FPE'],
    'Optimal_lag': [lag_sel.aic, lag_sel.bic, lag_sel.hqic, lag_sel.fpe]
})
lag_df.to_csv(f'{OUT}/dns_var_lag_selection.csv', index=False)

var_fit = var_factors.fit(p_sel)
print(f'  VAR({p_sel}) log-likelihood : {var_fit.llf:.3f}')
print(f'  VAR({p_sel}) AIC            : {var_fit.aic:.3f}')
print(f'  VAR({p_sel}) BIC            : {var_fit.bic:.3f}')

# Coefficient table
coef_rows = []
for eq_name in ['Level','Slope','Curvature']:
    params = var_fit.params[eq_name]
    for pname, pval in params.items():
        coef_rows.append({'Equation': eq_name, 'Regressor': pname, 'Coefficient': pval})
coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(f'{OUT}/dns_var_coefficients.csv', index=False)

fit_stats = pd.DataFrame({
    'Metric': ['LogLikelihood','AIC','BIC','HQIC','p_lag'],
    'Value' : [var_fit.llf, var_fit.aic, var_fit.bic, var_fit.hqic, p_sel]
})
fit_stats.to_csv(f'{OUT}/dns_var_fit_stats.csv', index=False)

# Save full summary
with open(f'{OUT}/dns_var_summary.txt', 'w') as fh:
    fh.write(str(var_fit.summary()))

# ─────────────────────────────────────────────────────────────────────────────
# A8. DNS-VAR with macro variables (Diebold-Rudebusch-Aruoba)
# ─────────────────────────────────────────────────────────────────────────────
sep('A8 · DNS-VAR with Macro Variables (Diebold-Rudebusch-Aruoba 2006)')

macro_factors = pd.concat([factors, df[['Inflation','PolicyRate']]], axis=1).dropna()
var_macro     = VAR(macro_factors)
lag_macro     = var_macro.select_order(maxlags=6)
p_macro       = lag_macro.aic
print(f'  DNS-Macro-VAR AIC-optimal lag : p = {p_macro}')

var_macro_fit = var_macro.fit(p_macro)
print(f'  DNS-Macro-VAR log-likelihood  : {var_macro_fit.llf:.3f}')

with open(f'{OUT}/dns_macro_var_summary.txt', 'w') as fh:
    fh.write(str(var_macro_fit.summary()))

# ─────────────────────────────────────────────────────────────────────────────
# A9. IRF & FEVD from DNS-VAR
# ─────────────────────────────────────────────────────────────────────────────
sep('A9 · IRF & FEVD from DNS-VAR')

IRF_H = 24
irf_obj  = var_macro_fit.irf(IRF_H)
fevd_obj = var_macro_fit.fevd(IRF_H)

# Safe horizon count from FEVD object
n_fevd_h = fevd_obj.decomp.shape[0]   # actual number of horizons stored

# Save IRF values
irf_rows = []
n_irf_h = irf_obj.irfs.shape[0]       # actual number of IRF horizons
for h in range(n_irf_h):
    for r_idx, resp_var in enumerate(macro_factors.columns):
        for s_idx, shock_var in enumerate(macro_factors.columns):
            irf_rows.append({
                'Horizon'  : h,
                'Response' : resp_var,
                'Shock'    : shock_var,
                'IRF'      : irf_obj.irfs[h, r_idx, s_idx],
            })
irf_df = pd.DataFrame(irf_rows)
irf_df.to_csv(f'{OUT}/dns_var_irf.csv', index=False)

# Save FEVD values
fevd_rows = []
for h in range(n_fevd_h):
    for r_idx, resp_var in enumerate(macro_factors.columns):
        for s_idx, source_var in enumerate(macro_factors.columns):
            fevd_rows.append({
                'Horizon' : h+1,
                'Variable': resp_var,
                'Source'  : source_var,
                'FEVD'    : fevd_obj.decomp[h, r_idx, s_idx],
            })
fevd_df = pd.DataFrame(fevd_rows)
fevd_df.to_csv(f'{OUT}/dns_var_fevd.csv', index=False)
print(f'  IRF ({n_irf_h-1}-month) and FEVD ({n_fevd_h}-month) saved.')

# ─────────────────────────────────────────────────────────────────────────────
# A10. Yield-curve forecasting
# ─────────────────────────────────────────────────────────────────────────────
sep('A10 · Yield-Curve Forecasting (h = 1, 3, 6, 12 months)')

forecast_horizons = [1, 3, 6, 12]
last_vals = macro_factors.values[-p_macro:]

fc_results = {}
for h in forecast_horizons:
    fc = var_macro_fit.forecast(last_vals, steps=h)
    fc_df = pd.DataFrame(fc, columns=macro_factors.columns)
    fc_results[h] = fc_df
    fc_df.to_csv(f'{OUT}/dns_forecast_h{h}.csv', index=False)
    # Convert factor forecasts to yield forecasts
    beta_fc = fc_df[['Level','Slope','Curvature']].values
    yc_fc   = beta_fc @ B_opt.T          # h × 8
    yc_fc_df = pd.DataFrame(yc_fc, columns=MATURITIES_LABELS)
    yc_fc_df.to_csv(f'{OUT}/dns_yield_forecast_h{h}.csv', index=False)

print(f'  Yield curve forecasts for horizons {forecast_horizons} saved.')

# ─────────────────────────────────────────────────────────────────────────────
# A11. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
sep('A11 · Generating DNS Figures')

# ── Figure A1: Factor loadings ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(TAU, B_opt[:,0], 'o-', color=C['level'],  lw=2, ms=5, label='Level (β₁)')
ax.plot(TAU, B_opt[:,1], 's-', color=C['slope'],  lw=2, ms=5, label='Slope (β₂)')
ax.plot(TAU, B_opt[:,2], '^-', color=C['curv'],   lw=2, ms=5, label='Curvature (β₃)')
ax.set_xlabel('Maturity (years)'); ax.set_ylabel('Loading')
ax.set_title(f'Nelson-Siegel Factor Loadings  (λ = {lam_opt:.4f})')
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_A1_dns_loadings.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_A1_dns_loadings.png')

# ── Figure A2: Factor time-series ───────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, col, clr, label in zip(axes,
    ['Level','Slope','Curvature'],
    [C['level'], C['slope'], C['curv']],
    ['Level (β₁) — Long-run yield','Slope (β₂) — Short-end','Curvature (β₃) — Medium hump']):
    ax.plot(dates, factors[col], color=clr, lw=1.5, label=label)
    ax.axhline(0, color='grey', lw=.6, ls='--')
    ax.set_ylabel(col); ax.legend(loc='upper right'); ax.grid(alpha=.3)
axes[-1].set_xlabel('Date')
fig.suptitle('DNS Factors — Bangladesh Yield Curve', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_A2_dns_factors.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_A2_dns_factors.png')

# ── Figure A3: Actual vs Fitted (selected maturities) ───────────────────────
sel_mat = [0, 2, 4, 6]   # Y3M, Y1Y, Y5Y, Y15Y
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, mi in zip(axes.flat, sel_mat):
    ax.plot(dates, yields[:,mi],  color=C['actual'], lw=1.2, label='Actual')
    ax.plot(dates, yhats[:,mi],   color=C['fitted'], lw=1.2, ls='--', label='Fitted')
    ax.set_title(MATURITIES_LABELS[mi]); ax.legend(); ax.grid(alpha=.3)
fig.suptitle('DNS Actual vs Fitted Yields', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_A3_dns_actual_vs_fitted.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_A3_dns_actual_vs_fitted.png')

# ── Figure A4: 3-D Yield Surface ─────────────────────────────────────────────
fig = plt.figure(figsize=(11, 6))
ax3d = fig.add_subplot(111, projection='3d')
t_idx = np.arange(T)
TT, MM = np.meshgrid(t_idx, TAU, indexing='ij')
surf = ax3d.plot_surface(TT, MM, yields, cmap='RdYlGn_r', alpha=.85, linewidth=0)
ax3d.set_xlabel('Time (months)', labelpad=8)
ax3d.set_ylabel('Maturity (yrs)', labelpad=8)
ax3d.set_zlabel('Yield (%)', labelpad=8)
ax3d.set_title('Bangladesh Yield Surface — DNS Fitted', fontsize=10)
fig.colorbar(surf, ax=ax3d, shrink=.4, label='Yield (%)')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_A4_dns_yield_surface.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_A4_dns_yield_surface.png')

# ── Figure A5: DNS-VAR IRF (PolicyRate shock → DNS factors) ─────────────────
shock_var  = 'PolicyRate'
resp_vars  = ['Level','Slope','Curvature']
s_idx_irf  = list(macro_factors.columns).index(shock_var)
fig, axes  = plt.subplots(1, 3, figsize=(13, 4))
for ax, rv in zip(axes, resp_vars):
    r_idx_irf = list(macro_factors.columns).index(rv)
    irf_v = irf_obj.irfs[:, r_idx_irf, s_idx_irf]
    try:
        se_v  = irf_obj.stderr()[:, r_idx_irf, s_idx_irf]
        lo    = irf_v - 1.96 * se_v
        hi    = irf_v + 1.96 * se_v
    except Exception:
        lo = hi = irf_v
    h_ax_irf = np.arange(len(irf_v))
    ax.plot(h_ax_irf, irf_v, color=C['policy'], lw=2, label='IRF')
    ax.fill_between(h_ax_irf, lo, hi, alpha=.25, color=C['policy'])
    ax.axhline(0, color='grey', lw=.7, ls='--')
    ax.set_title(f'→ {rv}'); ax.set_xlabel('Months'); ax.grid(alpha=.3)
axes[0].set_ylabel('Response')
fig.suptitle('DNS-VAR IRF: PolicyRate Shock → DNS Factors', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_A5_dns_var_irf.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_A5_dns_var_irf.png')

# ── Figure A6: DNS residuals ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, mi in zip(axes.flat, range(8)):
    ax.plot(dates, resids[:,mi], lw=.8, color='steelblue')
    ax.axhline(0, color='red', lw=.6, ls='--')
    ax.set_title(MATURITIES_LABELS[mi]); ax.grid(alpha=.3)
fig.suptitle('DNS Cross-Sectional Residuals by Maturity', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_A6_dns_residuals.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_A6_dns_residuals.png')

# ── Figure A7: Yield-curve forecast fan chart ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(TAU, yields[-1], 'o-', color=C['actual'], lw=2, ms=5, label=f'Observed ({dates[-1].strftime("%b %Y")})')
fc_colors = ['#1b7837','#4d9221','#a6d96a','#d9ef8b']
for i, h in enumerate(forecast_horizons):
    beta_fc = fc_results[h][['Level','Slope','Curvature']].values[-1]
    yc_fc   = B_opt @ beta_fc
    ax.plot(TAU, yc_fc, '--', lw=1.5, color=fc_colors[i], label=f'h={h}m forecast')
ax.set_xlabel('Maturity (years)'); ax.set_ylabel('Yield (%)')
ax.set_title('DNS Yield-Curve Forecasts'); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_A7_dns_yield_forecasts.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_A7_dns_yield_forecasts.png')

print(f'\n  DNS MODEL COMPLETE — λ={lam_opt:.4f}, RMSE={overall_rmse*100:.4f}bps, R²={overall_r2:.6f}')


# ================================================================================
# ██████   █████  ██████  ████████     ██████
# ██   ██ ██   ██ ██   ██    ██        ██   ██
# ██████  ███████ ██████     ██        ██████
# ██      ██   ██ ██   ██    ██        ██   ██
# ██      ██   ██ ██   ██    ██        ██████
# MARKOV-SWITCHING VAR (MS-VAR) MODEL
# ================================================================================

sep('PART B — MARKOV-SWITCHING VAR (MS-VAR) MODEL')

# ─────────────────────────────────────────────────────────────────────────────
# B1. Variable selection & stationarity
# ─────────────────────────────────────────────────────────────────────────────
sep('B1 · Variable Selection & Stationarity Tests (ADF + KPSS)')

data_msvar = df[VAR_COLS].copy()

unit_root_rows = []
for col in VAR_COLS:
    ser = data_msvar[col].dropna()
    adf = adfuller(ser, autolag='AIC')
    try:
        kps = kpss(ser, regression='c', nlags='auto')
        kpss_stat, kpss_p = kps[0], kps[1]
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
    unit_root_rows.append({
        'Variable'     : col,
        'ADF_stat'     : round(adf[0], 4),
        'ADF_pvalue'   : round(adf[1], 4),
        'ADF_lags'     : adf[2],
        'ADF_1pct_cv'  : round(adf[4]['1%'], 4),
        'ADF_5pct_cv'  : round(adf[4]['5%'], 4),
        'ADF_stationary': 'Yes' if adf[1] < 0.05 else 'No',
        'KPSS_stat'    : round(kpss_stat, 4) if not np.isnan(kpss_stat) else 'NA',
        'KPSS_pvalue'  : round(kpss_p,    4) if not np.isnan(kpss_p)    else 'NA',
    })
    print(f'  {col:12s} | ADF p={adf[1]:.4f} {"✅ I(0)" if adf[1]<0.05 else "⚠ I(1)"}')

ur_df = pd.DataFrame(unit_root_rows)
ur_df.to_csv(f'{OUT}/msvar_unit_root_tests.csv', index=False)

# First-difference non-stationary series
data_diff = data_msvar.diff().dropna()
print('\n  Stationarity of first-differenced series:')
for col in VAR_COLS:
    adf_d = adfuller(data_diff[col], autolag='AIC')
    print(f'  Δ{col:12s} | ADF p={adf_d[1]:.4f} {"✅ I(0)" if adf_d[1]<0.05 else "⚠ still non-stationary"}')

# ─────────────────────────────────────────────────────────────────────────────
# B2. Lag-order selection
# ─────────────────────────────────────────────────────────────────────────────
sep('B2 · Lag-Order Selection (AIC, BIC, HQIC, FPE)')

model_var_sel = VAR(data_diff)
lag_sel_b     = model_var_sel.select_order(maxlags=12)
p_msvar       = lag_sel_b.aic
print(f'  AIC  → p = {lag_sel_b.aic}')
print(f'  BIC  → p = {lag_sel_b.bic}')
print(f'  HQIC → p = {lag_sel_b.hqic}')
print(f'  FPE  → p = {lag_sel_b.fpe}')
print(f'  Using AIC-optimal lag: p = {p_msvar}')

lag_sel_df = pd.DataFrame({
    'Criterion': ['AIC','BIC','HQIC','FPE'],
    'Optimal_lag': [lag_sel_b.aic, lag_sel_b.bic, lag_sel_b.hqic, lag_sel_b.fpe]
})
lag_sel_df.to_csv(f'{OUT}/msvar_lag_selection.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# B3. Baseline VAR(p)
# ─────────────────────────────────────────────────────────────────────────────
sep(f'B3 · Baseline VAR({p_msvar}) Estimation')

var_baseline = model_var_sel.fit(p_msvar)
print(f'  Log-likelihood : {var_baseline.llf:.3f}')
print(f'  AIC            : {var_baseline.aic:.3f}')
print(f'  BIC            : {var_baseline.bic:.3f}')

with open(f'{OUT}/msvar_baseline_var_summary.txt', 'w') as fh:
    fh.write(str(var_baseline.summary()))

# Residuals & diagnostics
resid_var = var_baseline.resid
lb_rows = []
for col in resid_var.columns:
    lb = acorr_ljungbox(resid_var[col], lags=10, return_df=True)
    lb_rows.append({
        'Variable'     : col,
        'LB_stat_lag10': round(lb['lb_stat'].iloc[-1], 4),
        'LB_pval_lag10': round(lb['lb_pvalue'].iloc[-1], 4),
        'DW'           : round(durbin_watson(resid_var[col].values), 4),
    })
lb_df = pd.DataFrame(lb_rows)
lb_df.to_csv(f'{OUT}/msvar_ljungbox_diagnostics.csv', index=False)
print('  Ljung-Box residual diagnostics saved.')

# ─────────────────────────────────────────────────────────────────────────────
# B4. EM Algorithm for MS-VAR (Hamilton 1989 / Kim 1994)
# ─────────────────────────────────────────────────────────────────────────────
sep('B4 · EM Algorithm for MS-VAR  (Hamilton 1989 / Kim 1994)')

N_REGIMES = 2
K         = len(VAR_COLS)
p_lag     = max(1, p_msvar)
data_arr  = data_diff.values           # (T-1) × K

print(f'  Regimes  : {N_REGIMES}')
print(f'  Variables: {K}  ({VAR_COLS})')
print(f'  Lag      : {p_lag}')
print(f'  Obs      : {len(data_arr)}')

# ── Build lagged regressor matrix ────────────────────────────────────────────
def build_regressors(X, p):
    """Return (T-p) × (1 + K*p) regressor matrix and (T-p) × K response."""
    T_  = len(X)
    Y_  = X[p:]
    Z_  = np.hstack([np.ones((T_-p, 1))] +
                    [X[p-j-1:T_-j-1] for j in range(p)])
    return Y_, Z_

Y_reg, Z_reg = build_regressors(data_arr, p_lag)
T_eff        = len(Y_reg)
print(f'  Effective obs after lag: {T_eff}')

# ── Initialise EM via K-means on residuals ───────────────────────────────────
np.random.seed(42)
km     = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=42)
labels_init = km.fit_predict(Y_reg)

# Regime-specific OLS as starting values
mu_s    = np.zeros((N_REGIMES, K))        # regime means (intercept)
Phi_s   = np.zeros((N_REGIMES, K, K*p_lag))  # VAR coefficients (excl. const)
Sigma_s = np.zeros((N_REGIMES, K, K))     # covariance matrices

for s in range(N_REGIMES):
    idx_s = np.where(labels_init == s)[0]
    if len(idx_s) < K * p_lag + 2:
        idx_s = np.arange(T_eff)
    Y_s = Y_reg[idx_s]
    Z_s = Z_reg[idx_s]
    B_s = np.linalg.lstsq(Z_s, Y_s, rcond=None)[0]   # (1+K*p) × K
    mu_s[s]    = B_s[0]
    Phi_s[s]   = B_s[1:].T                             # K × K*p
    res_s      = Y_s - Z_s @ B_s
    Sigma_s[s] = np.cov(res_s.T) + 1e-6 * np.eye(K)

# Transition matrix initialise
P_trans = np.array([[0.95, 0.05],[0.05, 0.95]])

# Initial regime distribution (ergodic)
def ergodic_dist(P):
    evals, evecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(evals - 1.0))
    pi  = np.real(evecs[:, idx])
    return pi / pi.sum()

pi0 = ergodic_dist(P_trans)

# ── Gaussian log-density ─────────────────────────────────────────────────────
def log_mvn(y, mu, Sigma):
    """Log-density of N(mu, Sigma) at y."""
    diff = y - mu
    try:
        L    = np.linalg.cholesky(Sigma)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        sol  = np.linalg.solve(L, diff)
        quad = np.dot(sol, sol)
    except np.linalg.LinAlgError:
        return -1e10
    return -0.5 * (K * np.log(2 * np.pi) + logdet + quad)

# ── EM iterations ─────────────────────────────────────────────────────────────
MAX_ITER = 300
TOL      = 1e-6
llf_hist = []

for iteration in range(MAX_ITER):

    # ── E-step: Hamilton filter ───────────────────────────────────────────────
    xi_filt   = np.zeros((T_eff, N_REGIMES))   # filtered probs
    xi_pred   = np.zeros((T_eff, N_REGIMES))   # predicted probs
    log_lik   = 0.0

    # Regime-conditional fitted values
    mu_cond = np.zeros((T_eff, N_REGIMES, K))
    for s in range(N_REGIMES):
        mu_cond[:, s, :] = Z_reg @ np.vstack([mu_s[s], Phi_s[s].T])

    xi_prev = pi0.copy()
    for t in range(T_eff):
        # Predicted probabilities
        xi_pred[t] = P_trans.T @ xi_prev

        # Conditional densities
        eta = np.array([
            np.exp(log_mvn(Y_reg[t], mu_cond[t, s], Sigma_s[s]))
            for s in range(N_REGIMES)
        ])
        eta = np.maximum(eta, 1e-300)

        # Filtered probabilities
        joint      = xi_pred[t] * eta
        denom      = joint.sum()
        if denom < 1e-300:
            denom = 1e-300
        xi_filt[t] = joint / denom
        log_lik   += np.log(denom)
        xi_prev    = xi_filt[t]

    # ── Smoothing (Kim 1994) ──────────────────────────────────────────────────
    xi_smooth = np.zeros((T_eff, N_REGIMES))
    xi_smooth[-1] = xi_filt[-1]
    for t in range(T_eff-2, -1, -1):
        for s in range(N_REGIMES):
            num = 0.0
            for j in range(N_REGIMES):
                if xi_pred[t+1, j] > 1e-300:
                    num += P_trans[s, j] * xi_smooth[t+1, j] / xi_pred[t+1, j]
            xi_smooth[t, s] = xi_filt[t, s] * num

    # Normalise
    row_sums = xi_smooth.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-300, 1.0, row_sums)
    xi_smooth /= row_sums

    # ── M-step ───────────────────────────────────────────────────────────────
    # Transition matrix
    xi_joint = np.zeros((N_REGIMES, N_REGIMES))
    for t in range(1, T_eff):
        for s in range(N_REGIMES):
            for j in range(N_REGIMES):
                if xi_pred[t, j] > 1e-300:
                    xi_joint[s, j] += (
                        P_trans[s, j] * xi_filt[t-1, s] *
                        np.exp(log_mvn(Y_reg[t], mu_cond[t, j], Sigma_s[j])) /
                        xi_pred[t, j]
                    )
    row_sums_P = xi_joint.sum(axis=1, keepdims=True)
    row_sums_P = np.where(row_sums_P < 1e-300, 1.0, row_sums_P)
    P_new = xi_joint / row_sums_P

    # Regime-specific VAR parameters via WLS
    mu_new    = np.zeros_like(mu_s)
    Phi_new   = np.zeros_like(Phi_s)
    Sigma_new = np.zeros_like(Sigma_s)

    for s in range(N_REGIMES):
        w = xi_smooth[:, s]                  # T_eff weights
        W = np.diag(w)
        ZtWZ = Z_reg.T @ W @ Z_reg + 1e-8 * np.eye(Z_reg.shape[1])
        ZtWY = Z_reg.T @ W @ Y_reg
        B_s_new = np.linalg.solve(ZtWZ, ZtWY)   # (1+K*p) × K
        mu_new[s]   = B_s_new[0]
        Phi_new[s]  = B_s_new[1:].T
        # Update mu_cond for Sigma computation
        yhat_s      = Z_reg @ B_s_new
        res_s       = Y_reg - yhat_s
        Sigma_new[s] = (res_s.T @ W @ res_s) / (w.sum() + 1e-10) + 1e-6 * np.eye(K)

    # ── Check convergence ─────────────────────────────────────────────────────
    llf_hist.append(log_lik)
    if iteration > 2 and abs(llf_hist[-1] - llf_hist[-2]) < TOL:
        print(f'  ✅ EM converged at iteration {iteration+1}  (ΔlogL={abs(llf_hist[-1]-llf_hist[-2]):.2e})')
        break

    P_trans = P_new
    mu_s    = mu_new
    Phi_s   = Phi_new
    Sigma_s = Sigma_new

    # Recompute mu_cond with updated params
    for s in range(N_REGIMES):
        mu_cond[:, s, :] = Z_reg @ np.vstack([mu_s[s], Phi_s[s].T])

    if (iteration+1) % 50 == 0:
        print(f'  Iter {iteration+1:3d}: logL = {log_lik:.4f}')

else:
    print(f'  ⚠ EM reached max iterations ({MAX_ITER})')

print(f'  Final log-likelihood : {llf_hist[-1]:.4f}')
n_params = N_REGIMES * (K + K**2 * p_lag + K*(K+1)//2) + N_REGIMES*(N_REGIMES-1)
aic_ms   = -2 * llf_hist[-1] + 2 * n_params
bic_ms   = -2 * llf_hist[-1] + np.log(T_eff) * n_params
print(f'  AIC (MS-VAR)         : {aic_ms:.4f}')
print(f'  BIC (MS-VAR)         : {bic_ms:.4f}')
print(f'  AIC (baseline VAR)   : {var_baseline.aic:.4f}')

# Hard regime assignment
regime_hard = np.argmax(xi_smooth, axis=1)
regime_dates_arr = data_diff.index[p_lag:]

# ─────────────────────────────────────────────────────────────────────────────
# B5. Regime characterisation
# ─────────────────────────────────────────────────────────────────────────────
sep('B5 · Regime Characterisation')

# Transition matrix
print('\n  Estimated Transition Matrix P:')
P_df = pd.DataFrame(P_trans,
    index  =[f'From_R{s}' for s in range(N_REGIMES)],
    columns=[f'To_R{s}'   for s in range(N_REGIMES)])
print(P_df.round(4))
P_df.to_csv(f'{OUT}/msvar_transition_matrix.csv')

# Steady-state
pi_ss = ergodic_dist(P_trans)
print(f'\n  Steady-state (ergodic) probabilities:')
for s in range(N_REGIMES):
    print(f'    Regime {s}: {pi_ss[s]:.4f}  ({pi_ss[s]*100:.1f}%)')

# Expected durations
print('\n  Expected regime durations:')
for s in range(N_REGIMES):
    dur = 1 / (1 - P_trans[s, s])
    print(f'    Regime {s}: {dur:.2f} months')

# Time-shares from smoothed probs
time_shares = xi_smooth.mean(axis=0)
print(f'\n  Time-shares (smoothed probs):')
for s in range(N_REGIMES):
    print(f'    Regime {s}: {time_shares[s]:.4f}  ({time_shares[s]*100:.1f}%)')

# Regime-specific yield-curve statistics
print('\n  Regime-specific yield & macro statistics:')
regime_stat_rows = []
for s in range(N_REGIMES):
    mask = regime_hard == s
    idx  = np.where(mask)[0]
    # Map back to original df indices (offset by p_lag+1 for diff+lag)
    orig_idx = idx + p_lag + 1
    orig_idx = orig_idx[orig_idx < len(df)]
    row = {'Regime': s, 'N_obs': mask.sum(),
           'Time_share_pct': round(time_shares[s]*100, 1),
           'Duration_months': round(1/(1-P_trans[s,s]+1e-10), 2)}
    for col in VAR_COLS:
        vals = df[col].iloc[orig_idx]
        row[f'{col}_mean'] = round(vals.mean(), 4)
        row[f'{col}_std']  = round(vals.std(),  4)
    regime_stat_rows.append(row)
    print(f'\n  Regime {s} ({mask.sum()} obs):')
    for col in VAR_COLS:
        print(f'    {col:12s}: mean={row[f"{col}_mean"]:.4f}, std={row[f"{col}_std"]:.4f}')

regime_stats_df = pd.DataFrame(regime_stat_rows)
regime_stats_df.to_csv(f'{OUT}/msvar_regime_statistics.csv', index=False)

# Time-share CSV
ts_df = pd.DataFrame({'Regime': range(N_REGIMES), 'Time_share': time_shares,
                      'Duration': [1/(1-P_trans[s,s]+1e-10) for s in range(N_REGIMES)]})
ts_df.to_csv(f'{OUT}/msvar_time_shares.csv', index=False)

# Save smoothed & filtered probabilities
prob_df = pd.DataFrame(xi_smooth, index=regime_dates_arr,
                       columns=[f'Smooth_R{s}' for s in range(N_REGIMES)])
prob_df[[f'Filter_R{s}' for s in range(N_REGIMES)]] = xi_filt
prob_df['Regime_hard'] = regime_hard
prob_df.to_csv(f'{OUT}/msvar_regime_probabilities.csv')

# Save MS-VAR model parameters
msvar_param_rows = []
for s in range(N_REGIMES):
    for k, vname in enumerate(VAR_COLS):
        msvar_param_rows.append({
            'Regime': s, 'Equation': vname,
            'Intercept': round(mu_s[s, k], 6),
            'Sigma_diag': round(Sigma_s[s, k, k], 6),
        })
        for lag in range(p_lag):
            for j, vname2 in enumerate(VAR_COLS):
                msvar_param_rows[-1][f'L{lag+1}_{vname2}'] = round(Phi_s[s, k, lag*K+j], 6)
pd.DataFrame(msvar_param_rows).to_csv(f'{OUT}/msvar_parameters.csv', index=False)

print(f'\n  MS-VAR parameters saved.')

# ─────────────────────────────────────────────────────────────────────────────
# B6. Regime-conditional IRF
# ─────────────────────────────────────────────────────────────────────────────
sep('B6 · Regime-Conditional Impulse Response Functions')

def compute_irf_msvar(Phi, Sigma, K, p, H=24, shock_idx=None):
    """
    Compute orthogonalised IRF via Cholesky decomposition.
    Phi : (K, K*p) coefficient matrix
    Returns (H+1, K) IRF array for one-unit shock to variable shock_idx.
    """
    if shock_idx is None:
        shock_idx = K - 1   # default: last variable (PolicyRate)
    try:
        P_chol = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        P_chol = np.diag(np.sqrt(np.diag(Sigma)))

    # Companion matrix
    A_comp = np.zeros((K*p, K*p))
    A_comp[:K, :] = Phi
    if p > 1:
        A_comp[K:, :-K] = np.eye(K*(p-1))

    irfs = np.zeros((H+1, K))
    shock_vec = P_chol[:, shock_idx]   # structural shock
    irfs[0]   = shock_vec

    state = np.zeros(K*p)
    state[:K] = shock_vec
    for h in range(1, H+1):
        state    = A_comp @ state
        irfs[h]  = state[:K]
    return irfs

IRF_H_MS = 24
shock_idx_pr = VAR_COLS.index('PolicyRate')
irf_regime = {}
for s in range(N_REGIMES):
    irf_regime[s] = compute_irf_msvar(Phi_s[s], Sigma_s[s], K, p_lag, IRF_H_MS, shock_idx_pr)

# Save IRF
irf_ms_rows = []
for s in range(N_REGIMES):
    for h in range(IRF_H_MS+1):
        for k, vname in enumerate(VAR_COLS):
            irf_ms_rows.append({
                'Regime': s, 'Horizon': h, 'Variable': vname,
                'IRF': round(irf_regime[s][h, k], 8)
            })
pd.DataFrame(irf_ms_rows).to_csv(f'{OUT}/msvar_irf.csv', index=False)
print(f'  Regime-conditional IRF saved ({IRF_H_MS}-month horizon).')

# ─────────────────────────────────────────────────────────────────────────────
# B7. Regime-conditional FEVD
# ─────────────────────────────────────────────────────────────────────────────
sep('B7 · Regime-Conditional FEVD')

def compute_fevd_msvar(Phi, Sigma, K, p, H=24):
    """
    Compute FEVD for all variables and horizons.
    Returns (H, K, K) array: fevd[h, resp, shock]
    """
    try:
        P_chol = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        P_chol = np.diag(np.sqrt(np.diag(Sigma)))

    A_comp = np.zeros((K*p, K*p))
    A_comp[:K, :] = Phi
    if p > 1:
        A_comp[K:, :-K] = np.eye(K*(p-1))

    # MA coefficient matrices Ψ_h (K×K)
    Psi = [np.eye(K)]
    for h in range(1, H+1):
        psi_h = np.zeros((K, K))
        for j in range(min(h, p)):
            psi_h += Psi[h-j-1] @ Phi[:, j*K:(j+1)*K]
        Psi.append(psi_h)

    # Orthogonalised MA: Θ_h = Ψ_h @ P_chol
    Theta = [Psi[h] @ P_chol for h in range(H+1)]

    # FEVD[h, r, s] = Σ_{j=0}^{h} Θ_j[r,s]² / Σ_{j=0}^{h} Σ_s Θ_j[r,s]²
    fevd = np.zeros((H, K, K))
    for h in range(1, H+1):
        mse = np.zeros((K, K))
        for j in range(h):
            mse += Theta[j]**2
        total_mse = mse.sum(axis=1, keepdims=True)
        total_mse = np.where(total_mse < 1e-30, 1.0, total_mse)
        fevd[h-1] = mse / total_mse
    return fevd

fevd_regime = {}
for s in range(N_REGIMES):
    fevd_regime[s] = compute_fevd_msvar(Phi_s[s], Sigma_s[s], K, p_lag, IRF_H_MS)

# Save FEVD
fevd_ms_rows = []
for s in range(N_REGIMES):
    for h in range(IRF_H_MS):
        for r_idx, resp_var in enumerate(VAR_COLS):
            for src_idx, src_var in enumerate(VAR_COLS):
                fevd_ms_rows.append({
                    'Regime': s, 'Horizon': h+1,
                    'Response': resp_var, 'Source': src_var,
                    'FEVD': round(fevd_regime[s][h, r_idx, src_idx], 8)
                })
pd.DataFrame(fevd_ms_rows).to_csv(f'{OUT}/msvar_fevd.csv', index=False)
print(f'  Regime-conditional FEVD saved.')

# ─────────────────────────────────────────────────────────────────────────────
# B8. Residual diagnostics
# ─────────────────────────────────────────────────────────────────────────────
sep('B8 · Residual Diagnostics')

# Compute smoothed residuals (mixture)
resid_smooth = np.zeros((T_eff, K))
for t in range(T_eff):
    for s in range(N_REGIMES):
        mu_cond_t = Z_reg[t] @ np.vstack([mu_s[s], Phi_s[s].T])
        resid_smooth[t] += xi_smooth[t, s] * (Y_reg[t] - mu_cond_t)

diag_rows = []
for k, vname in enumerate(VAR_COLS):
    r  = resid_smooth[:, k]
    lb = acorr_ljungbox(r, lags=10, return_df=True)
    diag_rows.append({
        'Variable'     : vname,
        'Mean_resid'   : round(r.mean(), 6),
        'Std_resid'    : round(r.std(),  6),
        'DW'           : round(durbin_watson(r), 4),
        'LB_stat_lag10': round(lb['lb_stat'].iloc[-1], 4),
        'LB_pval_lag10': round(lb['lb_pvalue'].iloc[-1], 4),
        'No_autocorr'  : 'Yes' if lb['lb_pvalue'].iloc[-1] > 0.05 else 'No'
    })
    print(f'  {vname:12s}: DW={diag_rows[-1]["DW"]:.3f}, LB(10) p={diag_rows[-1]["LB_pval_lag10"]:.4f}')

pd.DataFrame(diag_rows).to_csv(f'{OUT}/msvar_residual_diagnostics.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# B9. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
sep('B9 · Generating MS-VAR Figures')

reg_colors = [C['r0'], C['r1']]
reg_labels = ['Regime 0 (Stable)', 'Regime 1 (Volatile)']

# ── Figure B1: Smoothed Regime Probabilities ─────────────────────────────────
fig, axes = plt.subplots(N_REGIMES+1, 1, figsize=(12, 7), sharex=True)
for s in range(N_REGIMES):
    axes[s].fill_between(regime_dates_arr, xi_smooth[:, s],
                         alpha=.7, color=reg_colors[s], label=reg_labels[s])
    axes[s].plot(regime_dates_arr, xi_smooth[:, s], color=reg_colors[s], lw=.8)
    axes[s].set_ylim(0, 1); axes[s].set_ylabel('Prob.'); axes[s].legend(loc='upper right')
    axes[s].grid(alpha=.3)
axes[-1].plot(df.index, df['PolicyRate'], color=C['policy'], lw=1.5, label='Policy Rate')
axes[-1].set_ylabel('Rate (%)'); axes[-1].legend(); axes[-1].grid(alpha=.3)
axes[-1].set_xlabel('Date')
fig.suptitle('MS-VAR Smoothed Regime Probabilities — Bangladesh', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_B1_msvar_regime_probs.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_B1_msvar_regime_probs.png')

# ── Figure B2: Regime Timeline ───────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
# Regime shading
for t in range(len(regime_hard)):
    clr = reg_colors[regime_hard[t]]
    axes[0].axvspan(regime_dates_arr[t],
                    regime_dates_arr[min(t+1, len(regime_dates_arr)-1)],
                    alpha=.5, color=clr, lw=0)
axes[0].set_yticks([]); axes[0].set_ylabel('Regime')
axes[0].set_title('Regime Classification (Hard Assignment)')
patch0 = mpatches.Patch(color=C['r0'], alpha=.6, label='Regime 0 — Stable')
patch1 = mpatches.Patch(color=C['r1'], alpha=.6, label='Regime 1 — Volatile')
axes[0].legend(handles=[patch0, patch1], loc='upper right')

axes[1].plot(df.index, df['PolicyRate'],  color=C['policy'],    lw=1.5, label='Policy Rate')
axes[1].plot(df.index, df['Inflation'],   color=C['inflation'], lw=1.5, label='Inflation')
axes[1].set_ylabel('Rate (%)'); axes[1].legend(); axes[1].grid(alpha=.3)

axes[2].plot(df.index, df['Y10Y'], color='#2ca25f', lw=1.5, label='10Y Yield')
axes[2].plot(df.index, df['Y3M'],  color='#99d8c9', lw=1.5, label='3M Yield')
axes[2].set_ylabel('Yield (%)'); axes[2].legend(); axes[2].grid(alpha=.3)

spread = df['Y10Y'] - df['Y3M']
axes[3].plot(df.index, spread, color='purple', lw=1.5, label='Term Spread (10Y-3M)')
axes[3].axhline(0, color='red', lw=.7, ls='--')
axes[3].set_ylabel('Spread (%)'); axes[3].legend(); axes[3].grid(alpha=.3)
axes[3].set_xlabel('Date')

fig.suptitle('Regime Timeline with Macroeconomic Variables — Bangladesh', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_B2_msvar_regime_timeline.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_B2_msvar_regime_timeline.png')

# ── Figure B3: Regime-specific Yield Curves ──────────────────────────────────
fig, axes = plt.subplots(1, N_REGIMES, figsize=(11, 5), sharey=True)
for s in range(N_REGIMES):
    mask     = regime_hard == s
    orig_idx = np.where(mask)[0] + p_lag + 1
    orig_idx = orig_idx[orig_idx < len(df)]
    yc_s     = df[MATURITIES_LABELS].iloc[orig_idx].values
    mean_yc  = yc_s.mean(axis=0)
    p10_yc   = np.percentile(yc_s, 10, axis=0)
    p90_yc   = np.percentile(yc_s, 90, axis=0)
    ax = axes[s]
    ax.plot(TAU, mean_yc, '-o', color=reg_colors[s], lw=2, ms=5, label='Mean')
    ax.fill_between(TAU, p10_yc, p90_yc, alpha=.25, color=reg_colors[s], label='10th–90th pct')
    ax.set_title(f'{reg_labels[s]}\n(n={mask.sum()} months)')
    ax.set_xlabel('Maturity (years)'); ax.legend(); ax.grid(alpha=.3)
axes[0].set_ylabel('Yield (%)')
fig.suptitle('Regime-Specific Yield Curves — Bangladesh', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_B3_regime_yield_curves.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_B3_regime_yield_curves.png')

# ── Figure B4: Regime-Conditional IRF Comparison ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
h_ax = np.arange(IRF_H_MS+1)
for ax_idx, k in enumerate(range(K)):
    ax = axes[ax_idx // 3, ax_idx % 3]
    for s in range(N_REGIMES):
        ax.plot(h_ax, irf_regime[s][:, k],
                color=reg_colors[s], lw=2, label=reg_labels[s])
    ax.axhline(0, color='grey', lw=.6, ls='--')
    ax.set_title(f'Response of {VAR_COLS[k]}')
    ax.set_xlabel('Months'); ax.legend(fontsize=7); ax.grid(alpha=.3)
axes[0, 0].set_ylabel('Response')
axes[1, 0].set_ylabel('Response')
fig.suptitle('MS-VAR IRF: PolicyRate Shock — Regime Comparison', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_B4_msvar_irf_comparison.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_B4_msvar_irf_comparison.png')

# ── Figure B5: Regime-Conditional FEVD ──────────────────────────────────────
fig, axes = plt.subplots(N_REGIMES, K, figsize=(18, 7))
h_ax_f = np.arange(1, IRF_H_MS+1)
for s in range(N_REGIMES):
    for k in range(K):
        ax = axes[s, k]
        bottom = np.zeros(IRF_H_MS)
        for src_idx, src_var in enumerate(VAR_COLS):
            vals = fevd_regime[s][:, k, src_idx]
            ax.bar(h_ax_f, vals, bottom=bottom, label=src_var, alpha=.8, width=0.9)
            bottom += vals
        ax.set_ylim(0, 1)
        if s == 0:
            ax.set_title(VAR_COLS[k], fontsize=8)
        if k == 0:
            ax.set_ylabel(f'R{s}\nShare', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('h', fontsize=6)
handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='lower center', ncol=K, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('MS-VAR FEVD by Regime', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_B5_msvar_fevd.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_B5_msvar_fevd.png')

# ── Figure B6: EM log-likelihood convergence ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(llf_hist, color='steelblue', lw=1.5)
ax.set_xlabel('EM Iteration'); ax.set_ylabel('Log-Likelihood')
ax.set_title('MS-VAR EM Algorithm Convergence'); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_B6_em_convergence.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_B6_em_convergence.png')

# ── Figure B7: Transition matrix heatmap ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(P_trans, cmap='Blues', vmin=0, vmax=1)
for i in range(N_REGIMES):
    for j in range(N_REGIMES):
        ax.text(j, i, f'{P_trans[i,j]:.4f}', ha='center', va='center', fontsize=12)
ax.set_xticks(range(N_REGIMES)); ax.set_yticks(range(N_REGIMES))
ax.set_xticklabels([f'R{s}' for s in range(N_REGIMES)])
ax.set_yticklabels([f'R{s}' for s in range(N_REGIMES)])
ax.set_xlabel('To Regime'); ax.set_ylabel('From Regime')
ax.set_title('Transition Probability Matrix')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_B7_transition_matrix.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_B7_transition_matrix.png')

# ── Figure B8: Residual analysis ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, k in zip(axes.flat, range(K)):
    ax.plot(regime_dates_arr, resid_smooth[:, k], lw=.8, color='steelblue')
    ax.axhline(0, color='red', lw=.6, ls='--')
    ax.set_title(f'Residuals: {VAR_COLS[k]}'); ax.grid(alpha=.3)
fig.suptitle('MS-VAR Smoothed Residuals', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_B8_msvar_residuals.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_B8_msvar_residuals.png')


# ================================================================================
# ██████   █████  ██████  ████████      ██████
# ██   ██ ██   ██ ██   ██    ██        ██
# ██████  ███████ ██████     ██        ██
# ██      ██   ██ ██   ██    ██        ██
# ██      ██   ██ ██   ██    ██         ██████
# COMBINED OUTPUT
# ================================================================================

sep('PART C — COMBINED OUTPUT')

# ─────────────────────────────────────────────────────────────────────────────
# C1. DNS factors ↔ MS-VAR regime overlay
# ─────────────────────────────────────────────────────────────────────────────
sep('C1 · DNS Factors × Regime Overlay')

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
factor_cols = ['Level','Slope','Curvature']
factor_clrs = [C['level'], C['slope'], C['curv']]

for ax, fc, clr in zip(axes, factor_cols, factor_clrs):
    ax.plot(dates, factors[fc], color=clr, lw=1.5, label=fc, zorder=3)
    # Shade by regime
    for t in range(len(regime_hard)):
        s  = regime_hard[t]
        t0 = regime_dates_arr[t]
        t1 = regime_dates_arr[min(t+1, len(regime_dates_arr)-1)]
        ax.axvspan(t0, t1, alpha=.15, color=reg_colors[s], lw=0)
    ax.axhline(0, color='grey', lw=.5, ls='--')
    ax.set_ylabel(fc); ax.legend(loc='upper right'); ax.grid(alpha=.2)

axes[-1].set_xlabel('Date')
patch0 = mpatches.Patch(color=C['r0'], alpha=.4, label='Regime 0 — Stable')
patch1 = mpatches.Patch(color=C['r1'], alpha=.4, label='Regime 1 — Volatile')
fig.legend(handles=[patch0, patch1], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('DNS Factors with MS-VAR Regime Overlay — Bangladesh', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_C1_dns_factors_regime_overlay.png', bbox_inches='tight')
plt.close()
print('  ✅ fig_C1_dns_factors_regime_overlay.png')

# ─────────────────────────────────────────────────────────────────────────────
# C2. Policy-transmission summary table
# ─────────────────────────────────────────────────────────────────────────────
sep('C2 · Policy-Transmission Summary Table')

pt_rows = []
for s in range(N_REGIMES):
    irf_s = irf_regime[s]
    for h in [1, 3, 6, 12, 24]:
        if h <= IRF_H_MS:
            for k, vname in enumerate(VAR_COLS):
                pt_rows.append({
                    'Regime' : s,
                    'Horizon': h,
                    'Variable': vname,
                    'IRF_PolicyRate_shock': round(irf_s[h, k], 6)
                })
pt_df = pd.DataFrame(pt_rows)
pt_df.to_csv(f'{OUT}/policy_transmission_summary.csv', index=False)
print('  Policy-transmission summary saved.')

# ─────────────────────────────────────────────────────────────────────────────
# C3. Markdown report
# ─────────────────────────────────────────────────────────────────────────────
sep('C3 · Generating Markdown Report')

md = []
md.append('# DNS and MS-VAR Estimation Report')
md.append(f'## Bangladesh Yield Curve & Monetary Policy — Monthly Data {dates[0].strftime("%b %Y")}–{dates[-1].strftime("%b %Y")}')
md.append('')
md.append('---')
md.append('')
md.append('## Part A — Dynamic Nelson-Siegel (DNS) Model')
md.append('')
md.append('### A1. Model Specification')
md.append('The Nelson-Siegel (1987) cross-sectional model:')
md.append('')
md.append('$$y(\\tau) = \\beta_1 + \\beta_2 \\frac{1-e^{-\\lambda\\tau}}{\\lambda\\tau} + \\beta_3\\left(\\frac{1-e^{-\\lambda\\tau}}{\\lambda\\tau} - e^{-\\lambda\\tau}\\right)$$')
md.append('')
md.append(f'| Parameter | Value |')
md.append(f'|-----------|-------|')
md.append(f'| Optimal λ (data) | {lam_opt:.6f} |')
md.append(f'| Diebold-Li (2006) λ | 0.0609 |')
md.append(f'| Sample | {dates[0].strftime("%b %Y")} – {dates[-1].strftime("%b %Y")} |')
md.append(f'| T | {T} monthly observations |')
md.append(f'| Overall RMSE | {overall_rmse*100:.4f} bps |')
md.append(f'| Mean R² | {overall_r2:.6f} |')
md.append('')
md.append('### A2. Factor Loadings per Maturity')
md.append('')
md.append('| Maturity | Level | Slope | Curvature | RMSE (bps) | R² |')
md.append('|----------|-------|-------|-----------|------------|-----|')
for _, row in df_load.iterrows():
    md.append(f'| {row["Maturity_label"]} | {row["Level_loading"]:.4f} | {row["Slope_loading"]:.4f} | {row["Curv_loading"]:.4f} | {row["RMSE_bps"]:.4f} | {row["R_squared"]:.6f} |')
md.append('')
md.append('### A3. Factor Descriptive Statistics')
md.append('')
md.append('| Factor | Mean | Std | Min | Max | Skew | Kurt | AR(1) |')
md.append('|--------|------|-----|-----|-----|------|------|-------|')
for fc in ['Level','Slope','Curvature']:
    r = desc.loc[fc]
    md.append(f'| {fc} | {r["mean"]:.4f} | {r["std"]:.4f} | {r["min"]:.4f} | {r["max"]:.4f} | {r["skewness"]:.4f} | {r["kurtosis"]:.4f} | {r["autocorr1"]:.4f} |')
md.append('')
md.append('### A4. AR(1) Factor Dynamics')
md.append('')
md.append('| Factor | μ | μ SE | φ | φ SE | σ | R² | DW |')
md.append('|--------|---|------|---|------|---|-----|-----|')
for fc in ['Level','Slope','Curvature']:
    r = ar1_results[fc]
    md.append(f'| {fc} | {r["mu"]:.4f} | {r["mu_se"]:.4f} | {r["phi"]:.4f} | {r["phi_se"]:.4f} | {r["sigma"]:.4f} | {r["R2"]:.4f} | {r["DW"]:.4f} |')
md.append('')
md.append(f'### A5. DNS-VAR({p_sel}) Lag Selection')
md.append('')
md.append('| Criterion | Optimal Lag |')
md.append('|-----------|-------------|')
for _, row in lag_df.iterrows():
    md.append(f'| {row["Criterion"]} | {row["Optimal_lag"]} |')
md.append('')
md.append(f'### A6. DNS-Macro-VAR({p_macro}) — Diebold-Rudebusch-Aruoba')
md.append(f'- Log-likelihood: {var_macro_fit.llf:.3f}')
md.append(f'- AIC: {var_macro_fit.aic:.3f}')
md.append(f'- BIC: {var_macro_fit.bic:.3f}')
md.append('')
md.append('---')
md.append('')
md.append('## Part B — Markov-Switching VAR (MS-VAR) Model')
md.append('')
md.append('### B1. Model Specification')
md.append('')
md.append('The MS-VAR(p) model (Hamilton 1989):')
md.append('')
md.append('$$y_t = \\mu_{s_t} + \\sum_{j=1}^{p} \\Phi_j^{(s_t)} y_{t-j} + \\varepsilon_t^{(s_t)}, \\quad \\varepsilon_t^{(s_t)} \\sim N(0, \\Sigma_{s_t})$$')
md.append('')
md.append(f'| Parameter | Value |')
md.append(f'|-----------|-------|')
md.append(f'| Variables | {", ".join(VAR_COLS)} |')
md.append(f'| Regimes   | {N_REGIMES} |')
md.append(f'| Lag p     | {p_lag} |')
md.append(f'| Eff. obs  | {T_eff} |')
md.append(f'| Log-likelihood | {llf_hist[-1]:.4f} |')
md.append(f'| AIC       | {aic_ms:.4f} |')
md.append(f'| BIC       | {bic_ms:.4f} |')
md.append(f'| EM iterations | {len(llf_hist)} |')
md.append('')
md.append('### B2. Transition Probability Matrix')
md.append('')
md.append('| | To R0 | To R1 |')
md.append('|---|-------|-------|')
for s in range(N_REGIMES):
    md.append(f'| From R{s} | {P_trans[s,0]:.4f} | {P_trans[s,1]:.4f} |')
md.append('')
md.append('### B3. Regime Characteristics')
md.append('')
md.append('| Regime | N obs | Time share | Duration (months) |')
md.append('|--------|-------|------------|-------------------|')
for row in regime_stat_rows:
    md.append(f'| {row["Regime"]} | {row["N_obs"]} | {row["Time_share_pct"]}% | {row["Duration_months"]:.2f} |')
md.append('')
md.append('### B4. Regime-Specific Mean Yields & Macro')
md.append('')
header = '| Regime | ' + ' | '.join(VAR_COLS) + ' |'
md.append(header)
md.append('|--------|' + '|'.join(['-------']*len(VAR_COLS)) + '|')
for row in regime_stat_rows:
    vals = ' | '.join([f'{row[f"{c}_mean"]:.4f}' for c in VAR_COLS])
    md.append(f'| {row["Regime"]} | {vals} |')
md.append('')
md.append('### B5. Residual Diagnostics')
md.append('')
md.append('| Variable | Mean | Std | DW | LB(10) stat | LB(10) p-val | No autocorr |')
md.append('|----------|------|-----|----|-------------|--------------|-------------|')
for row in diag_rows:
    md.append(f'| {row["Variable"]} | {row["Mean_resid"]:.6f} | {row["Std_resid"]:.6f} | {row["DW"]:.4f} | {row["LB_stat_lag10"]:.4f} | {row["LB_pval_lag10"]:.4f} | {row["No_autocorr"]} |')
md.append('')
md.append('---')
md.append('')
md.append('## Part C — Policy Transmission: IRF Summary')
md.append('')
md.append('**PolicyRate shock → selected responses (h = 1, 6, 12 months)**')
md.append('')
for s in range(N_REGIMES):
    md.append(f'\n### {reg_labels[s]}')
    md.append('| Variable | h=1 | h=6 | h=12 |')
    md.append('|----------|-----|-----|------|')
    for k, vname in enumerate(VAR_COLS):
        md.append(f'| {vname} | {irf_regime[s][1,k]:.6f} | {irf_regime[s][6,k]:.6f} | {irf_regime[s][12,k]:.6f} |')
md.append('')
md.append('---')
md.append('')
md.append('## References')
md.append('')
md.append('- Nelson, C. R., & Siegel, A. F. (1987). Parsimonious modeling of yield curves. *Journal of Business*, 60(4), 473–489.')
md.append('- Diebold, F. X., & Li, C. (2006). Forecasting the term structure of government bond yields. *Journal of Econometrics*, 130(2), 337–364.')
md.append('- Diebold, F. X., Rudebusch, G. D., & Aruoba, S. B. (2006). The macroeconomy and the yield curve. *Journal of Econometrics*, 131(1–2), 309–338.')
md.append('- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.')
md.append('- Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics*, 60(1–2), 1–22.')
md.append('- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.')
md.append('- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1–48.')
md.append('')
md.append('---')
md.append(f'*Source: Author\'s calculations. Data: Bangladesh Bank, {dates[0].strftime("%B %Y")} – {dates[-1].strftime("%B %Y")} (T={T}).*')
md.append('*Software: Python 3.10, statsmodels 0.14, scipy, scikit-learn, hmmlearn.*')

md_text = '\n'.join(md)
with open(f'{OUT}\\DNS_MSVAR_Report.md', 'w', encoding='utf-8') as fh:
    fh.write(md_text)
print('  ✅ DNS_MSVAR_Complete_Report.md')

# Convert to Word if pandoc available
import os
import subprocess

md_file = f'{OUT}\\DNS_MSVAR_Complete_Report.md'
docx_file = f'{OUT}\\DNS_MSVAR_Complete_Report.docx'
pandoc_path = r'C:\Program Files\Pandoc\pandoc.exe'

if os.path.exists(pandoc_path):
    ret = subprocess.run(
        [pandoc_path, md_file, '-o', docx_file],
        capture_output=True,
        text=True
    )
    if ret.returncode == 0:
        print("DOCX report created successfully.")
    else:
        print("Pandoc ran but failed:")
        print(ret.stderr)
else:
    print("Pandoc not found. Markdown report created, DOCX export skipped.")
# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
sep('FINAL OUTPUT SUMMARY')

all_files = [
    # DNS
    ('dns_factor_loadings.csv',       'DNS factor loadings & fit per maturity'),
    ('dns_factors.csv',               'DNS factor time-series (Level, Slope, Curvature)'),
    ('dns_factor_descriptive.csv',    'DNS factor descriptive statistics'),
    ('dns_ar1_dynamics.csv',          'AR(1) factor dynamics'),
    ('dns_fit_per_maturity.csv',      'Fit statistics per maturity'),
    ('dns_var_lag_selection.csv',     'DNS-VAR lag selection'),
    ('dns_var_coefficients.csv',      'DNS-VAR coefficients'),
    ('dns_var_fit_stats.csv',         'DNS-VAR fit statistics'),
    ('dns_var_irf.csv',               'DNS-VAR IRF values'),
    ('dns_var_fevd.csv',              'DNS-VAR FEVD values'),
    ('dns_var_summary.txt',           'DNS-VAR full summary'),
    ('dns_macro_var_summary.txt',     'DNS-Macro-VAR full summary'),
    ('dns_forecast_h1.csv',           'DNS factor forecast h=1'),
    ('dns_forecast_h3.csv',           'DNS factor forecast h=3'),
    ('dns_forecast_h6.csv',           'DNS factor forecast h=6'),
    ('dns_forecast_h12.csv',          'DNS factor forecast h=12'),
    ('dns_yield_forecast_h1.csv',     'DNS yield curve forecast h=1'),
    ('dns_yield_forecast_h12.csv',    'DNS yield curve forecast h=12'),
    ('fig_A1_dns_loadings.png',       'Figure A1: Factor loadings'),
    ('fig_A2_dns_factors.png',        'Figure A2: Factor time-series'),
    ('fig_A3_dns_actual_vs_fitted.png','Figure A3: Actual vs Fitted'),
    ('fig_A4_dns_yield_surface.png',  'Figure A4: 3-D Yield Surface'),
    ('fig_A5_dns_var_irf.png',        'Figure A5: DNS-VAR IRF'),
    ('fig_A6_dns_residuals.png',      'Figure A6: DNS residuals'),
    ('fig_A7_dns_yield_forecasts.png','Figure A7: Yield-curve forecasts'),
    # MS-VAR
    ('msvar_unit_root_tests.csv',     'MS-VAR unit root tests'),
    ('msvar_lag_selection.csv',       'MS-VAR lag selection'),
    ('msvar_baseline_var_summary.txt','Baseline VAR summary'),
    ('msvar_ljungbox_diagnostics.csv','Ljung-Box diagnostics'),
    ('msvar_transition_matrix.csv',   'Transition probability matrix'),
    ('msvar_regime_statistics.csv',   'Regime characteristics'),
    ('msvar_time_shares.csv',         'Regime time-shares & durations'),
    ('msvar_regime_probabilities.csv','Smoothed & filtered regime probs'),
    ('msvar_parameters.csv',          'MS-VAR regime-specific parameters'),
    ('msvar_irf.csv',                 'MS-VAR regime-conditional IRF'),
    ('msvar_fevd.csv',                'MS-VAR regime-conditional FEVD'),
    ('msvar_residual_diagnostics.csv','MS-VAR residual diagnostics'),
    ('fig_B1_msvar_regime_probs.png', 'Figure B1: Smoothed regime probs'),
    ('fig_B2_msvar_regime_timeline.png','Figure B2: Regime timeline'),
    ('fig_B3_regime_yield_curves.png','Figure B3: Regime yield curves'),
    ('fig_B4_msvar_irf_comparison.png','Figure B4: IRF comparison'),
    ('fig_B5_msvar_fevd.png',         'Figure B5: FEVD'),
    ('fig_B6_em_convergence.png',     'Figure B6: EM convergence'),
    ('fig_B7_transition_matrix.png',  'Figure B7: Transition matrix'),
    ('fig_B8_msvar_residuals.png',    'Figure B8: MS-VAR residuals'),
    # Combined
    ('fig_C1_dns_factors_regime_overlay.png','Figure C1: DNS factors + regime overlay'),
    ('policy_transmission_summary.csv','Policy-transmission IRF summary'),
    ('DNS_MSVAR_Complete_Report.md',  'Complete Markdown report'),
    ('DNS_MSVAR_Complete_Report.docx','Complete Word report'),
]

print(f'\n  {"FILE":<50} {"SIZE":>8}   DESCRIPTION')
print('  ' + '-'*90)
for fname, desc in all_files:
    fp = f'{OUT}/{fname}'
    if os.path.exists(fp):
        sz  = os.path.getsize(fp) / 1024
        tag = '✅'
    else:
        sz  = 0
        tag = '❌'
    print(f'  {tag}  {fname:<48} {sz:6.1f} KB   {desc}')

sep('SIMULATION COMPLETE')
print(f'  DNS Model  : λ={lam_opt:.4f}, RMSE={overall_rmse*100:.4f}bps, R²={overall_r2:.6f}')
print(f'  DNS-VAR    : p={p_sel}, LLF={var_fit.llf:.3f}, AIC={var_fit.aic:.3f}')
print(f'  MS-VAR     : {N_REGIMES} regimes, p={p_lag}, LLF={llf_hist[-1]:.4f}, AIC={aic_ms:.4f}')
print(f'  Regime 0   : {regime_stat_rows[0]["Time_share_pct"]}% of sample, '
      f'dur={regime_stat_rows[0]["Duration_months"]:.1f}m')
print(f'  Regime 1   : {regime_stat_rows[1]["Time_share_pct"]}% of sample, '
      f'dur={regime_stat_rows[1]["Duration_months"]:.1f}m')
print(f'\n  All output files → {OUT}/')
print('='*70)
