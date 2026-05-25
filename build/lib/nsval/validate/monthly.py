"""
nsval.validate.monthly
──────────────────────
Model vs observation validation using monthly-mean matching.

Both datasets are aggregated to calendar monthly means first, then
joined on year × month. Only months present in both datasets are used.

Typical usage
─────────────
    from nsval.validate.monthly import validate

    validate(
        obs_csv   = "TEMP_scoop_54.5_4.0.csv",
        model_csv = "roms_TEMP_54.5_4.0.csv",
        obs_var   = "TEMP",
        model_var = "temp_celsius",
    )
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from nsval.utils import load_timeseries_csv
from nsval.validate.metrics import (
    compute_metrics, seasonal_metrics, print_metrics, SEASON_MAP
)
from nsval.validate.daily import _taylor_panel

warnings.filterwarnings("ignore", category=RuntimeWarning)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION & MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def _to_monthly(df: pd.DataFrame, min_count: int = 1) -> pd.DataFrame:
    df = df.copy()
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month
    grp = df.groupby(["year","month"])
    agg = grp["value"].agg(["mean","count"]).reset_index()
    agg.columns = ["year","month","mean","count"]
    agg = agg[agg["count"] >= min_count].reset_index(drop=True)
    agg["ym"] = agg["year"] * 100 + agg["month"]
    return agg


def _match_monthly(obs: pd.DataFrame, model: pd.DataFrame,
                   min_obs: int = 1) -> pd.DataFrame:
    obs_m   = _to_monthly(obs,   min_obs)
    model_m = _to_monthly(model, 1)
    merged  = pd.merge(
        obs_m[["ym","year","month","mean","count"]],
        model_m[["ym","mean"]],
        on="ym", suffixes=("_obs","_mod"),
    )
    return merged.sort_values(["year","month"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def _make_figure(O, M, merged, met, seas, obs_var, model_var, dpi):
    MO  = merged["month"].values
    YR  = merged["year"].values

    obs_clim   = {m: float(np.mean(O[MO==m])) for m in range(1,13) if (MO==m).sum()>0}
    model_clim = {m: float(np.mean(M[MO==m])) for m in range(1,13) if (MO==m).sum()>0}
    monthly_bias = {m: model_clim[m] - obs_clim[m]
                    for m in obs_clim if m in model_clim}

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Model vs Observations (monthly means) — {obs_var}  |  "
        f"n={met['n_pairs']}  bias={met['bias']:+.2f}°C  "
        f"RMSE={met['rmse']:.2f}°C  r={met['r']:.3f}  "
        f"NSE={met['nse']:.3f}  IoA={met['ioa']:.3f}",
        fontsize=11, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    # ── Scatter coloured by month ─────────────────────────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    lims = [min(O.min(),M.min())-0.5, max(O.max(),M.max())+0.5]
    ax1.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax1.plot(lims, [l+met["bias"] for l in lims],
             "r-", lw=0.8, alpha=0.6, label=f"bias={met['bias']:+.2f}°C")
    sc = ax1.scatter(O, M, c=MO, cmap="hsv", s=45, alpha=0.85,
                     vmin=1, vmax=12, zorder=3)
    cbar = plt.colorbar(sc, ax=ax1, ticks=range(1,13))
    cbar.ax.set_yticklabels(MONTH_NAMES, fontsize=7)
    ax1.set_xlim(lims); ax1.set_ylim(lims); ax1.set_aspect("equal")
    ax1.set_xlabel("Observed monthly mean (°C)")
    ax1.set_ylabel("Model monthly mean (°C)")
    ax1.set_title("Scatter (by month)", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.text(0.03, 0.97,
             f"r²={met['r2']:.3f}\nRMSE={met['rmse']:.2f}°C\n"
             f"NSE={met['nse']:.3f}\nKGE={met['kge']:.3f}",
             transform=ax1.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Taylor diagram ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    _taylor_panel(ax2, met["std_ratio"], met["r"], seas)

    # ── Monthly climatology + bias ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    mc_months = sorted(obs_clim)
    ax3.plot(mc_months, [obs_clim[m]   for m in mc_months],
             "o-", color="#2980b9", lw=2, ms=6, label="Obs")
    ax3.plot(mc_months, [model_clim[m] for m in mc_months],
             "s--", color="#c0392b", lw=2, ms=6, label="Model")
    ax3b = ax3.twinx()
    mb_v = [monthly_bias.get(m, 0) for m in mc_months]
    ax3b.bar(mc_months, mb_v,
             color=["#c0392b" if v>=0 else "#2980b9" for v in mb_v],
             alpha=0.25, width=0.6)
    ax3b.axhline(0, color="grey", lw=0.5)
    ax3b.set_ylabel("Bias (°C)", fontsize=9, color="grey")
    ax3b.tick_params(axis="y", labelcolor="grey")
    ax3.set_xticks(range(1,13)); ax3.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_title("Monthly Climatology + Bias", fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── Q-Q ───────────────────────────────────────────────────────────────────
    ax4  = fig.add_subplot(gs[1, 0])
    pcts = np.linspace(1, 99, 99)
    qo, qm = np.percentile(O, pcts), np.percentile(M, pcts)
    ax4.plot([qo.min(),qo.max()],[qo.min(),qo.max()], "k--", lw=0.8, alpha=0.5)
    sc4 = ax4.scatter(qo, qm, c=pcts, cmap="RdYlBu_r", s=30, zorder=3)
    plt.colorbar(sc4, ax=ax4, label="Percentile")
    ax4.set_xlabel("Obs quantiles (°C)"); ax4.set_ylabel("Model quantiles (°C)")
    ax4.set_title("Q–Q Plot", fontweight="bold"); ax4.grid(True, alpha=0.3)

    # ── Timeseries of monthly means ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    t_ax = pd.to_datetime({"year": merged["year"],
                            "month": merged["month"], "day": 15})
    ax5.plot(t_ax, O, "o-", color="#2980b9", lw=1.2, ms=4, alpha=0.8,
             label="Obs")
    ax5.plot(t_ax, M, "s--", color="#c0392b", lw=1.2, ms=4, alpha=0.8,
             label="Model")
    ax5.fill_between(t_ax, O, M, where=(M>=O), alpha=0.15,
                     color="#c0392b", label="Model warm")
    ax5.fill_between(t_ax, O, M, where=(M<O),  alpha=0.15,
                     color="#2980b9", label="Model cold")
    ax5.set_ylabel("Temperature (°C)")
    ax5.set_title("Monthly Means Timeseries", fontweight="bold")
    ax5.legend(fontsize=7, ncol=2); ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_locator(mdates.YearLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ── Seasonal boxplots ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    sl  = ["DJF","MAM","JJA","SON"]
    sm  = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
    xb  = np.arange(len(sl)); w = 0.30

    for i, (slabel, smonths) in enumerate(zip(sl, sm)):
        mask = np.isin(MO, smonths)
        if mask.sum() == 0: continue
        for vals, pos, fc, mc in [
            (O[mask], xb[i]-w/2, "#aec6e8", "#2980b9"),
            (M[mask], xb[i]+w/2, "#f5b7b1", "#c0392b"),
        ]:
            ax6.boxplot(vals, positions=[pos], widths=w, patch_artist=True,
                        boxprops=dict(facecolor=fc, alpha=0.8),
                        medianprops=dict(color=mc, lw=2),
                        whiskerprops=dict(color=mc),
                        capprops=dict(color=mc),
                        flierprops=dict(marker="o", ms=3, color=mc, alpha=0.5))

    ax6.legend(handles=[Patch(fc="#aec6e8", label="Obs"),
                         Patch(fc="#f5b7b1", label="Model")], fontsize=9)
    ax6.set_xticks(xb); ax6.set_xticklabels(sl, fontsize=10)
    ax6.set_ylabel("Temperature (°C)")
    ax6.set_title("Seasonal Distribution", fontweight="bold")
    ax6.grid(True, axis="y", alpha=0.3)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    obs_csv        : str | Path,
    model_csv      : str | Path,
    obs_var        : str   = "TEMP",
    model_var      : str   = "temp_celsius",
    obs_qc_col     : str | None = "TEMP_QC",
    obs_qc_good    : set   = frozenset({1, 2}),
    model_sim      : str | None = None,
    min_obs_per_month: int = 1,
    out_metrics    : str | Path | None = "validation_metrics_monthly.csv",
    out_figure     : str | Path | None = "validation_summary_monthly.png",
    figure_dpi     : int   = 130,
    show_figure    : bool  = True,
) -> dict:
    """
    Validate model against observations using monthly-mean matching.

    Returns
    -------
    dict with keys: 'O', 'M', 'merged', 'metrics', 'seasonal', 'figure'
    """
    print(f"\n{'═'*65}")
    print(f"  nsval.validate.monthly  |  {obs_var} vs {model_var}")
    print(f"{'═'*65}")

    obs   = load_timeseries_csv(obs_csv,   obs_var,   obs_qc_col,
                                set(obs_qc_good), None)
    model = load_timeseries_csv(model_csv, model_var, None, None, model_sim)

    print(f"  Obs   : {len(obs):>6} rows")
    print(f"  Model : {len(model):>6} rows")

    merged = _match_monthly(obs, model, min_obs_per_month)
    if len(merged) == 0:
        raise RuntimeError("No overlapping year/month pairs found.")

    O  = merged["mean_obs"].values.astype(float)
    M  = merged["mean_mod"].values.astype(float)
    MO = merged["month"].values

    print(f"  Monthly pairs : {len(O)}")

    met  = compute_metrics(O, M, label="ALL")
    seas = seasonal_metrics(O, M, MO)

    monthly_bias = {
        m: float(np.mean(M[MO==m]) - np.mean(O[MO==m]))
        for m in range(1,13) if (MO==m).sum() > 0
    }

    print_metrics(met, seas, monthly_bias, obs_var, model_var)

    if out_metrics:
        rows = [met] + [v for v in seas.values() if v]
        pd.DataFrame(rows).to_csv(out_metrics, index=False,
                                   float_format="%.6f")
        print(f"  Saved metrics → {out_metrics}")

    fig = _make_figure(O, M, merged, met, seas, obs_var, model_var,
                       figure_dpi)
    if out_figure:
        fig.savefig(out_figure, dpi=figure_dpi, bbox_inches="tight")
        print(f"  Saved figure  → {out_figure}")
    if show_figure:
        plt.show()

    return {"O": O, "M": M, "merged": merged, "metrics": met,
            "seasonal": seas, "figure": fig}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Monthly-mean model vs obs validation."
    )
    p.add_argument("--obs",       required=True)
    p.add_argument("--model",     required=True)
    p.add_argument("--obs-var",   default="TEMP")
    p.add_argument("--model-var", default="temp_celsius")
    p.add_argument("--no-show",   action="store_true")
    args = p.parse_args()
    validate(args.obs, args.model, args.obs_var, args.model_var,
             show_figure=not args.no_show)


if __name__ == "__main__":
    _cli()
