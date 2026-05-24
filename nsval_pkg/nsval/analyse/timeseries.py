"""
nsval.analyse.timeseries
────────────────────────
Single-dataset timeseries diagnostics: statistics, climatology, and figures.

Produces
────────
  - Terminal report: overall stats, monthly table, seasonal table,
    annual trend, anomaly statistics.
  - Figure 1: full timeseries (scatter + rolling mean) + anomaly bars.
  - Figure 2: monthly climatology (mean ± std, full range, annual lines).
  - Figure 3: day-of-year climatology (smoothed mean ± std, full range).

Typical usage
─────────────
    from nsval.analyse.timeseries import analyse

    analyse(
        csv_file  = "TEMP_scoop_54.5_4.0.csv",
        variable  = "TEMP",
        qc_col    = "TEMP_QC",
        flag      = "Archive",
    )

Or run directly:
    python -m nsval.analyse.timeseries --csv TEMP_scoop_54.5_4.0.csv \\
        --variable TEMP --flag Archive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d

from nsval.utils import build_time

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def _load(csv_file, variable, qc_col, qc_good):
    df = pd.read_csv(csv_file)
    df["time"] = build_time(df)
    df = df.sort_values("time").reset_index(drop=True)

    if qc_col and qc_col in df.columns:
        before = len(df)
        df = df[df[qc_col].isin(qc_good)]
        print(f"  QC filter: {before} → {len(df)} rows")

    df = df.dropna(subset=[variable])
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["doy"]   = df["time"].dt.dayofyear
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def _print_stats(df, variable):
    temp = df[variable].values

    print(f"\n{'─'*60}")
    print(f"  OVERALL STATISTICS  ({variable})")
    print(f"{'─'*60}")
    stats = {
        "N observations" : len(temp),
        "Mean  (°C)"     : np.nanmean(temp),
        "Median (°C)"    : np.nanmedian(temp),
        "Std   (°C)"     : np.nanstd(temp, ddof=1),
        "Min   (°C)"     : np.nanmin(temp),
        "Max   (°C)"     : np.nanmax(temp),
        "P5    (°C)"     : np.nanpercentile(temp, 5),
        "P95   (°C)"     : np.nanpercentile(temp, 95),
        "Range (°C)"     : np.nanmax(temp) - np.nanmin(temp),
        "IQR   (°C)"     : np.nanpercentile(temp, 75) - np.nanpercentile(temp, 25),
    }
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:>8.3f}")
        else:
            print(f"  {k:<28} {v:>8}")

    monthly_mean = df.groupby("month")[variable].mean()
    print(f"\n{'─'*60}")
    print(f"  MONTHLY MEANS")
    print(f"{'─'*60}")
    print(f"  {'Month':<10}", end="")
    for m in range(1, 13):
        print(f"  {MONTH_NAMES[m-1]:>5}", end="")
    print()
    print(f"  {'Mean °C':<10}", end="")
    for m in range(1, 13):
        v = monthly_mean.get(m, np.nan)
        print(f"  {v:>5.2f}" if not np.isnan(v) else f"  {'–':>5}", end="")
    print()

    seasons = {
        "Winter (DJF)": [12,1,2], "Spring (MAM)": [3,4,5],
        "Summer (JJA)": [6,7,8],  "Autumn (SON)": [9,10,11],
    }
    print(f"\n{'─'*60}")
    print(f"  SEASONAL STATISTICS")
    print(f"{'─'*60}")
    for label, months in seasons.items():
        sub = df[df["month"].isin(months)][variable]
        if len(sub) == 0:
            print(f"  {label:<20} no data"); continue
        print(f"  {label:<20}  n={len(sub):>6}  "
              f"mean={sub.mean():>6.2f}  std={sub.std(ddof=1):>5.2f}  "
              f"min={sub.min():>6.2f}  max={sub.max():>6.2f}  °C")

    annual = df.groupby("year")[variable].mean().dropna()
    if len(annual) >= 3:
        yrs   = annual.index.values.astype(float)
        vals  = annual.values
        slope, intercept = np.polyfit(yrs, vals, 1)
        resid = vals - np.polyval([slope, intercept], yrs)
        r2    = 1 - np.var(resid) / np.var(vals)
        print(f"\n{'─'*60}")
        print(f"  ANNUAL TREND")
        print(f"{'─'*60}")
        print(f"  Slope  : {slope:+.4f} °C/year")
        print(f"  Total  : {slope*(yrs[-1]-yrs[0]):+.3f} °C over record")
        print(f"  R²     : {r2:.3f}")

    df["clim_mean"] = df["month"].map(monthly_mean)
    df["anomaly"]   = df[variable] - df["clim_mean"]
    print(f"\n{'─'*60}")
    print(f"  ANOMALY STATISTICS")
    print(f"{'─'*60}")
    print(f"  Mean anomaly   : {df['anomaly'].mean():+.4f} °C")
    print(f"  Std anomaly    : {df['anomaly'].std(ddof=1):.3f} °C")
    print(f"  Max warm event : "
          f"{df.loc[df['anomaly'].idxmax(),'time'].date()}  "
          f"({df['anomaly'].max():+.2f} °C)")
    print(f"  Max cold event : "
          f"{df.loc[df['anomaly'].idxmin(),'time'].date()}  "
          f"({df['anomaly'].min():+.2f} °C)")
    print(f"\n{'═'*60}\n")

    return df  # now contains 'anomaly' column


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_circular(arr, w):
    filled = np.where(np.isnan(arr), np.nanmean(arr), arr)
    tiled  = np.tile(filled, 3)
    sm     = uniform_filter1d(tiled, size=w, mode="wrap")
    return sm[365:730]


def _fig_timeseries(df, variable, flag, rolling_days, dpi, save):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"{variable} timeseries — surface — "
        f"{df['time'].min().year}–{df['time'].max().year}",
        fontsize=13, fontweight="bold",
    )

    ax1.scatter(df["time"], df[variable], s=2, alpha=0.4,
                color="#4a90d9", label="Daily observations", zorder=2)

    roll = (df.set_index("time")[variable]
              .rolling(f"{rolling_days}D", center=True, min_periods=5)
              .mean())
    ax1.plot(roll.index, roll.values, color="#c0392b", lw=1.5,
             label=f"{rolling_days}-day rolling mean", zorder=3)
    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    pos = df["anomaly"] >= 0
    ax2.bar(df["time"][pos],  df["anomaly"][pos],  width=1,
            color="#c0392b", alpha=0.7, label="Warm anomaly")
    ax2.bar(df["time"][~pos], df["anomaly"][~pos], width=1,
            color="#2980b9", alpha=0.7, label="Cold anomaly")
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_ylabel("Anomaly (°C)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    if save:
        fname = f"timeseries_{flag}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"  Saved {fname}")
    return fig


def _fig_monthly_clim(df, variable, flag, dpi, save):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"{variable} monthly climatology — surface — "
        f"{df['year'].min()}–{df['year'].max()}",
        fontsize=13, fontweight="bold",
    )

    months    = np.arange(1, 13)
    clim_mean = df.groupby("month")[variable].mean()
    clim_std  = df.groupby("month")[variable].std(ddof=1)
    clim_min  = df.groupby("month")[variable].min()
    clim_max  = df.groupby("month")[variable].max()

    for yr, grp in df.groupby("year"):
        ym = grp.groupby("month")[variable].mean()
        ax.plot(ym.index, ym.values, color="grey", alpha=0.2, lw=0.8)

    ax.fill_between(months, clim_mean - clim_std, clim_mean + clim_std,
                    alpha=0.25, color="#e67e22", label="±1 std")
    ax.fill_between(months, clim_min, clim_max,
                    alpha=0.10, color="#27ae60", label="Full range")
    ax.plot(months, clim_mean.reindex(months).values,
            "o-", color="#c0392b", lw=2, ms=6, label="Climatological mean")

    ax.set_xticks(months); ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_xlabel("Month", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        fname = f"monthly_clim_{flag}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"  Saved {fname}")
    return fig


def _fig_doy_clim(df, variable, flag, doy_smooth, dpi, save):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f"{variable} day-of-year climatology — surface — "
        f"{df['year'].min()}–{df['year'].max()}",
        fontsize=13, fontweight="bold",
    )

    doys     = np.arange(1, 366)
    doy_mean = df.groupby("doy")[variable].mean().reindex(range(1, 366))
    doy_std  = df.groupby("doy")[variable].std(ddof=1).reindex(range(1, 366))
    doy_min  = df.groupby("doy")[variable].min().reindex(range(1, 366))
    doy_max  = df.groupby("doy")[variable].max().reindex(range(1, 366))

    sm_mean = _smooth_circular(doy_mean.values, doy_smooth)
    sm_std  = _smooth_circular(doy_std.values,  doy_smooth)
    sm_min  = _smooth_circular(doy_min.values,  doy_smooth)
    sm_max  = _smooth_circular(doy_max.values,  doy_smooth)

    ax.fill_between(doys, sm_min, sm_max,
                    alpha=0.10, color="#27ae60", label="Full range (smoothed)")
    ax.fill_between(doys, sm_mean - sm_std, sm_mean + sm_std,
                    alpha=0.25, color="#e67e22", label="±1 std (smoothed)")
    ax.plot(doys, sm_mean, color="#c0392b", lw=2,
            label=f"DoY mean ({doy_smooth}-day smoothed)")

    month_starts = [1,32,60,91,121,152,182,213,244,274,305,335]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    for ms in month_starts:
        ax.axvline(ms, color="grey", lw=0.4, alpha=0.5)

    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_xlabel("Day of year", fontsize=11)
    ax.set_xlim(1, 365)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        fname = f"doy_clim_{flag}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"  Saved {fname}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    csv_file      : str | Path,
    variable      : str   = "TEMP",
    qc_col        : str | None = "TEMP_QC",
    qc_good       : set   = frozenset({1, 2}),
    flag          : str   = "dataset",
    rolling_days  : int   = 30,
    doy_smooth    : int   = 11,
    save_figures  : bool  = True,
    show_figures  : bool  = True,
    figure_dpi    : int   = 130,
) -> dict:
    """
    Run full timeseries diagnostics on a single CSV dataset.

    Parameters
    ----------
    csv_file     : path to CSV produced by nsval.intake
    variable     : column name of the variable to analyse
    qc_col       : QC flag column (None to skip)
    qc_good      : accepted QC flag values
    flag         : label used in figure filenames, e.g. 'Archive' or 'Model'
    rolling_days : window for rolling mean in Figure 1
    doy_smooth   : smoothing window for day-of-year climatology
    save_figures : write PNG files to current directory
    show_figures : call plt.show()
    figure_dpi   : PNG resolution

    Returns
    -------
    dict with keys 'data' (DataFrame) and 'figures' (list of Figure objects)
    """
    print(f"\n{'═'*60}")
    print(f"  nsval.analyse.timeseries  |  {variable}  |  {flag}")
    print(f"{'═'*60}")

    df = _load(csv_file, variable, qc_col, set(qc_good))

    print(f"  Rows  : {len(df)}")
    print(f"  Period: {df['time'].min().date()} – {df['time'].max().date()}")
    if "source_file" in df.columns:
        print(f"  Files : {df['source_file'].nunique()}")
    if "simulation" in df.columns:
        print(f"  Sims  : {list(df['simulation'].unique())}")

    df = _print_stats(df, variable)

    figs = []
    figs.append(_fig_timeseries(df, variable, flag, rolling_days,
                                figure_dpi, save_figures))
    figs.append(_fig_monthly_clim(df, variable, flag, figure_dpi,
                                  save_figures))
    figs.append(_fig_doy_clim(df, variable, flag, doy_smooth,
                              figure_dpi, save_figures))

    if show_figures:
        plt.show()

    return {"data": df, "figures": figs}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Timeseries diagnostics and climatology plots."
    )
    p.add_argument("--csv",      required=True)
    p.add_argument("--variable", default="TEMP")
    p.add_argument("--qc",       default="TEMP_QC",
                   help="QC column name (empty string to disable)")
    p.add_argument("--flag",     default="dataset",
                   help="Label for output filenames")
    p.add_argument("--no-save",  action="store_true")
    p.add_argument("--no-show",  action="store_true")
    args = p.parse_args()

    analyse(
        csv_file     = args.csv,
        variable     = args.variable,
        qc_col       = args.qc or None,
        flag         = args.flag,
        save_figures = not args.no_save,
        show_figures = not args.no_show,
    )


if __name__ == "__main__":
    _cli()
