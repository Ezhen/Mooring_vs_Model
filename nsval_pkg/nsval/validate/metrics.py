"""
nsval.validate.metrics
──────────────────────
Core validation metric functions, shared by daily and monthly validators.

All functions accept plain numpy arrays O (observed) and M (modelled)
of the same length with NaNs already removed.
"""

from __future__ import annotations
import numpy as np
from scipy import stats


def compute_metrics(O: np.ndarray, M: np.ndarray,
                    label: str = "") -> dict | None:
    """
    Compute a comprehensive set of model-vs-observation metrics.

    Parameters
    ----------
    O     : observed values (1-D, no NaNs)
    M     : modelled values (same shape as O)
    label : optional tag stored in the returned dict

    Returns
    -------
    dict of metrics, or None if len(O) < 3
    """
    if len(O) < 3:
        return None

    n      = len(O)
    bias   = float(np.mean(M - O))
    mae    = float(np.mean(np.abs(M - O)))
    mse    = float(np.mean((M - O)**2))
    rmse   = float(np.sqrt(mse))
    std_o  = float(np.std(O, ddof=1))
    std_m  = float(np.std(M, ddof=1))
    mean_o = float(np.mean(O))
    mean_m = float(np.mean(M))

    # Pearson
    r, p_r   = stats.pearsonr(O, M)
    r, p_r   = float(r), float(p_r)

    # Spearman
    rho, p_rho = stats.spearmanr(O, M)
    rho, p_rho = float(rho), float(p_rho)

    # R²
    ss_res = float(np.sum((O - M)**2))
    ss_tot = float(np.sum((O - mean_o)**2))
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Normalised RMSE
    nrmse  = rmse / std_o if std_o > 0 else np.nan

    # Centred RMSE
    crmse  = float(np.sqrt(np.mean(((M - mean_m) - (O - mean_o))**2)))

    # Max absolute error
    max_ae = float(np.max(np.abs(M - O)))

    # Index of Agreement (Willmott 1981)
    d_ioa  = float(np.sum((np.abs(M - mean_o) + np.abs(O - mean_o))**2))
    ioa    = 1 - ss_res / d_ioa if d_ioa > 0 else np.nan

    # Modified IoA (Willmott et al. 1985, j=1)
    d_mioa = float(np.sum(np.abs(M - mean_o) + np.abs(O - mean_o)))
    mioa   = (1 - np.sum(np.abs(M - O)) / d_mioa
              if d_mioa > 0 else np.nan)

    # Nash–Sutcliffe Efficiency
    nse    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Kling–Gupta Efficiency (Gupta et al. 2009)
    beta   = mean_m / mean_o  if mean_o  != 0 else np.nan
    gamma  = (std_m / mean_m) / (std_o / mean_o) \
             if (mean_m != 0 and mean_o != 0) else np.nan
    kge    = float(1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2))

    # Skill score vs. obs climatology as reference
    ref_mse = float(np.mean((O - mean_o)**2))
    skill   = 1 - mse / ref_mse if ref_mse > 0 else np.nan

    # Std ratio
    std_ratio = std_m / std_o if std_o > 0 else np.nan

    # Percentile errors
    p10_err = float(np.percentile(M, 10) - np.percentile(O, 10))
    p50_err = float(np.percentile(M, 50) - np.percentile(O, 50))
    p90_err = float(np.percentile(M, 90) - np.percentile(O, 90))

    # MAPE
    nz   = O != 0
    mape = float(np.mean(np.abs((O[nz] - M[nz]) / O[nz])) * 100) \
           if nz.sum() > 0 else np.nan

    # Hit rates
    hit_05 = float(np.mean(np.abs(M - O) <= 0.5) * 100)
    hit_10 = float(np.mean(np.abs(M - O) <= 1.0) * 100)

    return {
        "label"         : label,
        "n_pairs"       : int(n),
        "mean_obs"      : mean_o,
        "mean_model"    : mean_m,
        "std_obs"       : std_o,
        "std_model"     : std_m,
        "std_ratio"     : std_ratio,
        "bias"          : bias,
        "mae"           : mae,
        "mse"           : mse,
        "rmse"          : rmse,
        "crmse"         : crmse,
        "nrmse"         : nrmse,
        "max_abs_error" : max_ae,
        "mape_pct"      : mape,
        "p10_error"     : p10_err,
        "p50_error"     : p50_err,
        "p90_error"     : p90_err,
        "hit_rate_05C"  : hit_05,
        "hit_rate_10C"  : hit_10,
        "r"             : r,
        "r_pvalue"      : p_r,
        "r2"            : r2,
        "spearman_rho"  : rho,
        "spearman_p"    : p_rho,
        "ioa"           : float(ioa),
        "mioa"          : float(mioa),
        "nse"           : float(nse),
        "kge"           : kge,
        "skill_score"   : float(skill),
    }


SEASON_MAP = {
    "Winter_DJF": [12, 1, 2],
    "Spring_MAM": [3, 4, 5],
    "Summer_JJA": [6, 7, 8],
    "Autumn_SON": [9, 10, 11],
}


def seasonal_metrics(O: np.ndarray, M: np.ndarray,
                     months: np.ndarray) -> dict:
    """
    Compute metrics for each meteorological season.

    Parameters
    ----------
    O, M   : full matched arrays
    months : integer month array (1–12), same length as O and M

    Returns
    -------
    dict mapping season label → metrics dict (or None if n < 3)
    """
    result = {}
    for sname, smonths in SEASON_MAP.items():
        mask = np.isin(months, smonths)
        result[sname] = compute_metrics(O[mask], M[mask], label=sname)
    return result


def print_metrics(metrics: dict, seasonal: dict,
                  monthly_bias: dict, obs_var: str, model_var: str):
    """Pretty-print the full metrics report to stdout."""

    def fv(v, signed=True, w=10, d=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return f"{'N/A':>{w}}"
        if isinstance(v, int):
            return f"{v:>{w}}"
        fmt = f"{{:>+{w}.{d}f}}" if signed else f"{{:>{w}.{d}f}}"
        return fmt.format(v)

    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    sections = [
        ("SAMPLE", [
            ("n_pairs",       "Matched pairs",                  "",    False),
            ("mean_obs",      "Mean obs",                       "°C",  True),
            ("mean_model",    "Mean model",                     "°C",  True),
            ("std_obs",       "Std obs",                        "°C",  False),
            ("std_model",     "Std model",                      "°C",  False),
            ("std_ratio",     "Std ratio (model/obs)",          "",    False),
        ]),
        ("ERROR", [
            ("bias",          "Bias (model − obs)",             "°C",  True),
            ("mae",           "MAE",                            "°C",  False),
            ("mse",           "MSE",                            "°C²", False),
            ("rmse",          "RMSE",                           "°C",  False),
            ("crmse",         "Centred RMSE",                   "°C",  False),
            ("nrmse",         "Normalised RMSE",                "",    False),
            ("max_abs_error", "Max absolute error",             "°C",  False),
            ("mape_pct",      "MAPE",                           "%",   False),
            ("p10_error",     "P10 error",                      "°C",  True),
            ("p50_error",     "P50 error (median bias)",        "°C",  True),
            ("p90_error",     "P90 error",                      "°C",  True),
            ("hit_rate_05C",  "Hit rate |err| ≤ 0.5°C",        "%",   False),
            ("hit_rate_10C",  "Hit rate |err| ≤ 1.0°C",        "%",   False),
        ]),
        ("CORRELATION & SKILL", [
            ("r",             "Pearson r",                      "",    False),
            ("r_pvalue",      "Pearson p-value",                "",    False),
            ("r2",            "R²",                             "",    False),
            ("spearman_rho",  "Spearman ρ",                     "",    False),
            ("spearman_p",    "Spearman p-value",               "",    False),
            ("ioa",           "Index of Agreement (Willmott)",  "",    False),
            ("mioa",          "Modified IoA",                   "",    False),
            ("nse",           "Nash–Sutcliffe Efficiency",      "",    False),
            ("kge",           "Kling–Gupta Efficiency",         "",    False),
            ("skill_score",   "Skill Score (vs clim. mean)",    "",    False),
        ]),
    ]

    print(f"\n{'═'*65}")
    print(f"  VALIDATION METRICS  —  {obs_var} vs {model_var}")
    print(f"{'═'*65}")

    for section_name, items in sections:
        print(f"\n  ── {section_name} ──")
        for key, label, unit, signed in items:
            v = metrics.get(key)
            u = f"  {unit}" if unit else ""
            print(f"  {label:<42} {fv(v, signed=signed)}{u}")

    # Seasonal table
    print(f"\n  ── SEASONAL BREAKDOWN ──")
    cw = 12
    hdr = f"  {'Metric':<16}"
    for s in SEASON_MAP:
        hdr += f"  {s.split('_')[0]:>{cw}}"
    print(hdr)
    for key, label, signed in [
        ("n_pairs", "n",    False), ("bias",  "Bias", True),
        ("rmse",    "RMSE", False), ("mae",   "MAE",  False),
        ("r",       "r",    False), ("r2",    "R²",   False),
        ("nse",     "NSE",  False), ("kge",   "KGE",  False),
        ("ioa",     "IoA",  False),
    ]:
        row = f"  {label:<16}"
        for sname in SEASON_MAP:
            sm = seasonal.get(sname)
            v  = sm[key] if sm else np.nan
            row += f"  {fv(v, signed=signed, w=cw, d=3)}"
        print(row)

    # Monthly bias table
    print(f"\n  ── MONTHLY CLIMATOLOGICAL BIAS ──")
    print(f"  {'Month':<8}", end="")
    for m in range(1, 13):
        print(f"  {MONTH_NAMES[m-1]:>5}", end="")
    print()
    print(f"  {'Bias°C':<8}", end="")
    for m in range(1, 13):
        v = monthly_bias.get(m, np.nan)
        print(f"  {v:>+5.2f}" if not np.isnan(v) else f"  {'–':>5}", end="")
    print()
    print(f"\n{'═'*65}\n")
