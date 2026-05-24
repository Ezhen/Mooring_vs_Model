"""
nsval.validate.daily
────────────────────
Model vs observation validation using nearest-timestep matching.

Each observation is paired with the model timestep closest in time,
within a configurable tolerance window.

Typical usage
─────────────
    from nsval.validate.daily import validate

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

from nsval.utils import load_timeseries_csv
from nsval.validate.metrics import (
    compute_metrics, seasonal_metrics, print_metrics, SEASON_MAP
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def _match_nearest(obs: pd.DataFrame, model: pd.DataFrame,
                   tol_days: float) -> tuple[np.ndarray, np.ndarray,
                                             pd.DatetimeIndex]:
    model_t = model["time"].values.astype("datetime64[ns]")
    obs_t   = obs["time"].values.astype("datetime64[ns]")
    tol_ns  = np.timedelta64(int(tol_days * 86400e9), "ns")

    o_vals, m_vals, times = [], [], []
    for i, ot in enumerate(obs_t):
        diff = np.abs(model_t - ot)
        idx  = np.argmin(diff)
        if diff[idx] <= tol_ns:
            o_vals.append(obs["value"].iloc[i])
            m_vals.append(model["value"].iloc[idx])
            times.append(pd.Timestamp(ot))

    return (np.array(o_vals, dtype=float),
            np.array(m_vals, dtype=float),
            pd.DatetimeIndex(times))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def _make_figure(O, M, T, metrics, monthly_bias, obs_var, model_var, dpi):
    from scipy.stats import gaussian_kde

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Model vs Observations — {obs_var}  |  n={metrics['n_pairs']}  "
        f"bias={metrics['bias']:+.2f}°C  RMSE={metrics['rmse']:.2f}°C  "
        f"r={metrics['r']:.3f}  IoA={metrics['ioa']:.3f}",
        fontsize=12, fontweight="bold", y=0.98,
    )

    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
    MO  = T.month

    # ── Scatter ──────────────────────────────────────────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    lims = [min(O.min(), M.min()) - 0.5, max(O.max(), M.max()) + 0.5]
    ax1.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax1.plot(lims, [l + metrics["bias"] for l in lims],
             "r-", lw=0.8, alpha=0.6, label=f"bias={metrics['bias']:+.2f}°C")
    try:
        z   = gaussian_kde(np.vstack([O, M]))(np.vstack([O, M]))
        idx = z.argsort()
        sc  = ax1.scatter(O[idx], M[idx], c=z[idx], s=18,
                          cmap="plasma", alpha=0.7, zorder=3)
        plt.colorbar(sc, ax=ax1, label="Density")
    except Exception:
        ax1.scatter(O, M, s=12, alpha=0.5, color="#4a90d9")
    ax1.set_xlim(lims); ax1.set_ylim(lims); ax1.set_aspect("equal")
    ax1.set_xlabel("Observed (°C)"); ax1.set_ylabel("Model (°C)")
    ax1.set_title("Scatter", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.text(0.03, 0.97,
             f"r²={metrics['r2']:.3f}\nRMSE={metrics['rmse']:.2f}°C\n"
             f"NSE={metrics['nse']:.3f}",
             transform=ax1.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Taylor diagram ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    _taylor_panel(ax2, metrics["std_ratio"], metrics["r"], {})

    # ── Monthly bias ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    mb_m = sorted(monthly_bias); mb_v = [monthly_bias[m] for m in mb_m]
    ax3.bar(mb_m, mb_v,
            color=["#c0392b" if v >= 0 else "#2980b9" for v in mb_v],
            alpha=0.8, width=0.7)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.axhline(metrics["bias"], color="grey", lw=1, ls="--",
                label=f"Overall={metrics['bias']:+.2f}°C")
    ax3.set_xticks(range(1,13)); ax3.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax3.set_ylabel("Bias (°C)"); ax3.set_title("Monthly Bias", fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(True, axis="y", alpha=0.3)

    # ── Q-Q ───────────────────────────────────────────────────────────────────
    ax4   = fig.add_subplot(gs[1, 0])
    pcts  = np.linspace(1, 99, 99)
    qo, qm = np.percentile(O, pcts), np.percentile(M, pcts)
    ax4.plot([qo.min(), qo.max()], [qo.min(), qo.max()],
             "k--", lw=0.8, alpha=0.5)
    sc4 = ax4.scatter(qo, qm, c=pcts, cmap="RdYlBu_r", s=25, zorder=3)
    plt.colorbar(sc4, ax=ax4, label="Percentile")
    ax4.set_xlabel("Obs quantiles (°C)"); ax4.set_ylabel("Model quantiles (°C)")
    ax4.set_title("Q–Q Plot", fontweight="bold"); ax4.grid(True, alpha=0.3)

    # ── Residuals timeseries ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    res = M - O
    pos = res >= 0
    ax5.bar(T[pos],  res[pos],  width=2, color="#c0392b", alpha=0.6,
            label="Model warm")
    ax5.bar(T[~pos], res[~pos], width=2, color="#2980b9", alpha=0.6,
            label="Model cold")
    ax5.axhline(0, color="k", lw=0.7)
    ax5.axhline(metrics["bias"], color="red", lw=1, ls="--",
                label=f"Mean bias={metrics['bias']:+.2f}°C")
    roll = (pd.Series(res, index=T).sort_index()
              .rolling("30D", center=True, min_periods=3).mean())
    ax5.plot(roll.index, roll.values, "k-", lw=1.2,
             label="30-day rolling bias")
    ax5.set_ylabel("Model − Obs (°C)")
    ax5.set_title("Residuals timeseries", fontweight="bold")
    ax5.legend(fontsize=8, ncol=4); ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_locator(mdates.YearLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha="right")

    return fig


def _taylor_panel(ax, std_ratio, r_val, seasonal):
    """Minimal Taylor panel (shared between daily and monthly)."""
    theta_m = np.arccos(np.clip(r_val, -1, 1))
    th_arc  = np.linspace(0, np.pi/2, 300)

    for cl, col in zip([0.5,1.0,1.5], ["#cccccc","#aaaaaa","#888888"]):
        th2  = np.linspace(0, np.pi/2, 200)
        rs2  = np.linspace(0.01, 2.5, 200)
        TH, RS = np.meshgrid(th2, rs2)
        dist = np.sqrt((RS*np.cos(TH) - 1.0)**2 + (RS*np.sin(TH))**2)
        ax.contour(TH, RS, dist, levels=[cl], colors=[col],
                   linewidths=0.7, alpha=0.8)

    for sr in [0.5, 1.0, 1.5, 2.0]:
        ax.plot(th_arc, np.full_like(th_arc, sr),
                ":", color="grey", lw=0.5, alpha=0.5)

    for r_line in [0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        th_l = np.arccos(r_line)
        ax.plot([th_l, th_l], [0, 2.2], color="#cccccc", lw=0.5)

    ax.plot(0, 1.0, "ko", ms=9, label="Obs (ref)", zorder=5)
    ax.plot(theta_m, std_ratio, "r^", ms=11, label="Model", zorder=5)

    s_col = {"Winter_DJF":"#3498db","Spring_MAM":"#2ecc71",
             "Summer_JJA":"#e74c3c","Autumn_SON":"#e67e22"}
    for sname, sm in seasonal.items():
        if sm:
            ax.plot(np.arccos(np.clip(sm["r"],-1,1)), sm["std_ratio"],
                    "D", ms=7, color=s_col[sname],
                    label=sname.split("_")[0], zorder=4)

    ax.set_thetamax(90); ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_ylim(0, max(2.2, std_ratio + 0.3))
    ax.set_xticks(np.arccos([1.0,0.99,0.95,0.9,0.8,0.6,0.4,0.0]))
    ax.set_xticklabels(["1","0.99","0.95","0.9","0.8","0.6","0.4","0"],
                       fontsize=6)
    ax.set_title("Taylor Diagram (norm.)", fontweight="bold", pad=14)
    ax.legend(fontsize=7, loc="upper right",
              bbox_to_anchor=(1.45, 1.15), ncol=1)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    obs_csv       : str | Path,
    model_csv     : str | Path,
    obs_var       : str   = "TEMP",
    model_var     : str   = "temp_celsius",
    obs_qc_col    : str | None = "TEMP_QC",
    obs_qc_good   : set   = frozenset({1, 2}),
    model_sim     : str | None = None,
    match_tol_days: float = 1.0,
    out_metrics   : str | Path | None = "validation_metrics_daily.csv",
    out_figure    : str | Path | None = "validation_summary_daily.png",
    figure_dpi    : int   = 130,
    show_figure   : bool  = True,
) -> dict:
    """
    Validate model against observations using nearest-timestep matching.

    Returns
    -------
    dict with keys: 'O', 'M', 'T', 'metrics', 'seasonal', 'figure'
    """
    print(f"\n{'═'*65}")
    print(f"  nsval.validate.daily  |  {obs_var} vs {model_var}")
    print(f"{'═'*65}")

    obs   = load_timeseries_csv(obs_csv,   obs_var,   obs_qc_col,
                                set(obs_qc_good), None)
    model = load_timeseries_csv(model_csv, model_var, None, None, model_sim)

    print(f"  Obs   : {len(obs):>6} rows  "
          f"({obs['time'].min().date()} – {obs['time'].max().date()})")
    print(f"  Model : {len(model):>6} rows  "
          f"({model['time'].min().date()} – {model['time'].max().date()})")

    O, M, T = _match_nearest(obs, model, match_tol_days)
    if len(O) == 0:
        raise RuntimeError("No matching pairs found.")
    print(f"  Matched pairs : {len(O)}  (±{match_tol_days} day)")

    met  = compute_metrics(O, M, label="ALL")
    seas = seasonal_metrics(O, M, T.month.values)

    monthly_bias = {
        m: float(np.mean(M[T.month == m] - O[T.month == m]))
        for m in range(1, 13) if (T.month == m).sum() > 0
    }

    print_metrics(met, seas, monthly_bias, obs_var, model_var)

    if out_metrics:
        rows = [met] + [v for v in seas.values() if v]
        pd.DataFrame(rows).to_csv(out_metrics, index=False,
                                   float_format="%.6f")
        print(f"  Saved metrics → {out_metrics}")

    fig = _make_figure(O, M, T, met, monthly_bias, obs_var, model_var,
                       figure_dpi)
    if out_figure:
        fig.savefig(out_figure, dpi=figure_dpi, bbox_inches="tight")
        print(f"  Saved figure  → {out_figure}")
    if show_figure:
        plt.show()

    return {"O": O, "M": M, "T": T, "metrics": met,
            "seasonal": seas, "figure": fig}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Daily nearest-match model vs obs validation."
    )
    p.add_argument("--obs",       required=True)
    p.add_argument("--model",     required=True)
    p.add_argument("--obs-var",   default="TEMP")
    p.add_argument("--model-var", default="temp_celsius")
    p.add_argument("--tol",       type=float, default=1.0)
    p.add_argument("--no-show",   action="store_true")
    args = p.parse_args()
    validate(args.obs, args.model, args.obs_var, args.model_var,
             match_tol_days=args.tol, show_figure=not args.no_show)


if __name__ == "__main__":
    _cli()
