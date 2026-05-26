"""
nsval.analyse.seasonal_dashboard
─────────────────────────────────────────────────────────────────────
Seasonal temperature dashboard for the North Sea derived from
irregular in-situ records and/or ROMS model output.

Produces three figures:

  Figure 1 — 4-panel seasonal maps
      One panel per season (DJF / MAM / JJA / SON).
      Each station plotted at its true lat/lon, coloured by its
      seasonal mean temperature. Model seasonal mean shown as
      background contour if provided.

  Figure 2 — 4-panel seasonal timeseries
      One panel per season. All observations in that season plotted
      as scatter. Model timeseries overlaid. Seasonal climatological
      mean ± 1 std shown as horizontal band.

  Figure 3 — Climatological rose (seasonal cycle)
      Both obs and model climatological seasonal cycles on the same
      axes with full spread (min/max) and ±1 std shading.
      Seasons clearly delineated.

Typical usage
─────────────
    from nsval.analyse.seasonal_dashboard import seasonal_dashboard

    seasonal_dashboard(
        obs_csv   = "TEMP_scoop_54.5_4.0.csv",
        model_csv = "roms_temp_54.5_4.0.csv",
        obs_var   = "TEMP",
        model_var = "temp_celsius",
        out_prefix= "seasonal_dashboard",
    )
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

from nsval.utils import load_timeseries_csv, build_time

# =============================================================================
# CONSTANTS
# =============================================================================

SEASONS = {
    "DJF": {"months": [12, 1, 2],  "label": "Winter (DJF)",
            "color": "#3498db", "marker": "o"},
    "MAM": {"months": [3, 4, 5],   "label": "Spring (MAM)",
            "color": "#2ecc71", "marker": "s"},
    "JJA": {"months": [6, 7, 8],   "label": "Summer (JJA)",
            "color": "#e74c3c", "marker": "^"},
    "SON": {"months": [9, 10, 11], "label": "Autumn (SON)",
            "color": "#e67e22", "marker": "D"},
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# North Sea bounding box
NS_LAT = (48.0, 62.0)
NS_LON = (-4.0, 10.0)

DEFAULT_DPI = 130


# =============================================================================
# LOAD
# =============================================================================

def _load_obs(obs_csv, obs_var, qc_col, qc_good):
    df = pd.read_csv(obs_csv)
    df["time"] = pd.to_datetime(df["time"] if "time" in df.columns
                             else df[["year","month","day"]],
                             infer_datetime_format=True,
                             errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    if qc_col and qc_col in df.columns:
        before = len(df)
        df = df[df[qc_col].isin(set(qc_good))]
        print(f"  Obs QC: {before} → {len(df)} rows")

    df = df.dropna(subset=[obs_var])
    df["month"]  = df["time"].dt.month
    df["year"]   = df["time"].dt.year
    df["season"] = df["month"].map(_month_to_season)
    return df


def _load_model(model_csv, model_var, model_sim):
    df = pd.read_csv(model_csv)
    df["time"] = pd.to_datetime(df["time"] if "time" in df.columns
                             else df[["year","month","day"]],
                             infer_datetime_format=True,
                             errors="coerce")
    if model_sim and "simulation" in df.columns:
        df = df[df["simulation"] == model_sim]
    df = df.dropna(subset=[model_var])
    df = df.sort_values("time").reset_index(drop=True)
    df["month"]  = df["time"].dt.month
    df["year"]   = df["time"].dt.year
    df["season"] = df["month"].map(_month_to_season)
    return df


def _month_to_season(m):
    for sname, sinfo in SEASONS.items():
        if m in sinfo["months"]:
            return sname
    return None


# =============================================================================
# SEASONAL STATISTICS
# =============================================================================

def _seasonal_stats(df, var):
    """Return dict of season → {mean, std, min, max, n, monthly_mean}."""
    stats = {}
    for sname, sinfo in SEASONS.items():
        sub = df[df["season"] == sname][var]
        if len(sub) == 0:
            stats[sname] = None
            continue
        # monthly means within season
        monthly = df[df["season"] == sname].groupby("month")[var].mean()
        stats[sname] = {
            "mean"   : float(sub.mean()),
            "std"    : float(sub.std(ddof=1)),
            "min"    : float(sub.min()),
            "max"    : float(sub.max()),
            "n"      : len(sub),
            "monthly": monthly,
            "values" : sub.values,
        }
    return stats


# =============================================================================
# FIGURE 1 — SEASONAL MAPS
# =============================================================================

def _fig_maps(obs_df, obs_var, model_df, model_var, title_suffix, dpi):
    """
    4-panel map: one per season.
    Stations coloured by seasonal mean temperature.
    Requires file_lat / file_lon columns in obs_df.
    """
    has_positions = ("file_lat" in obs_df.columns and
                     "file_lon" in obs_df.columns)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Seasonal mean temperature — {title_suffix}",
                 fontsize=13, fontweight="bold")

    axes_flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

    # global colour range
    vmin = obs_df[obs_var].quantile(0.02)
    vmax = obs_df[obs_var].quantile(0.98)

    for ax, (sname, sinfo) in zip(axes_flat, SEASONS.items()):
        ax.set_facecolor("#d4e8f0")
        ax.set_xlim(NS_LON)
        ax.set_ylim(NS_LAT)
        ax.set_title(sinfo["label"], fontweight="bold",
                     color=sinfo["color"])
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, alpha=0.3)

        # model seasonal mean as text annotation (no basemap needed)
        if model_df is not None:
            msub = model_df[model_df["season"] == sname][model_var]
            if len(msub) > 0:
                ax.text(0.02, 0.97,
                        f"Model mean: {msub.mean():.2f}°C",
                        transform=ax.transAxes, va="top", fontsize=8,
                        color="#c0392b",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", alpha=0.8))

        if not has_positions:
            ax.text(0.5, 0.5,
                    "No lat/lon in CSV\n(use scoop CSV, not wide format)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="grey")
            continue

        # station seasonal means
        obs_s = obs_df[obs_df["season"] == sname]
        if len(obs_s) == 0:
            continue

        station_means = (obs_s.groupby(["file_lat", "file_lon"])[obs_var]
                           .mean().reset_index())

        sc = ax.scatter(
            station_means["file_lon"],
            station_means["file_lat"],
            c    = station_means[obs_var],
            cmap = "RdYlBu_r",
            vmin = vmin, vmax = vmax,
            s    = 80, zorder = 5,
            edgecolors = "k", linewidths = 0.5,
        )

        # station count annotation
        ax.text(0.02, 0.05,
                f"n stations: {len(station_means)}  "
                f"n obs: {len(obs_s)}",
                transform=ax.transAxes, va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.3)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu_r",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Temperature (°C)")

    return fig


# =============================================================================
# FIGURE 2 — SEASONAL TIMESERIES
# =============================================================================

def _fig_timeseries(obs_df, obs_var, model_df, model_var,
                    obs_stats, model_stats, title_suffix, dpi):
    """
    4-panel timeseries: one per season.
    Observations as scatter, model as line, seasonal mean ± std as band.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=False)
    fig.suptitle(f"Seasonal timeseries — {title_suffix}",
                 fontsize=13, fontweight="bold")

    axes_flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

    for ax, (sname, sinfo) in zip(axes_flat, SEASONS.items()):
        color = sinfo["color"]

        # obs scatter
        obs_s = obs_df[obs_df["season"] == sname]
        if len(obs_s) > 0:
            ax.scatter(obs_s["time"], obs_s[obs_var],
                       s=8, alpha=0.5, color=color,
                       label="Obs", zorder=3)

        # obs seasonal mean ± std band
        st = obs_stats.get(sname)
        if st:
            ax.axhline(st["mean"], color=color, lw=1.5,
                       ls="--", alpha=0.8,
                       label=f"Obs mean={st['mean']:.1f}°C")
            ax.axhspan(st["mean"] - st["std"],
                       st["mean"] + st["std"],
                       alpha=0.10, color=color)

        # model line
        if model_df is not None:
            mod_s = model_df[model_df["season"] == sname]
            if len(mod_s) > 0:
                mod_s = mod_s.sort_values("time")
                ax.plot(mod_s["time"], mod_s[model_var],
                        color="#c0392b", lw=0.8, alpha=0.6,
                        label="Model", zorder=2)
                mst = model_stats.get(sname)
                if mst:
                    ax.axhline(mst["mean"], color="#c0392b",
                               lw=1.5, ls=":",
                               label=f"Model mean={mst['mean']:.1f}°C")

        ax.set_title(sinfo["label"], fontweight="bold", color=color)
        ax.set_ylabel("Temperature (°C)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # n label
        n_obs = len(obs_s)
        ax.text(0.99, 0.97, f"n={n_obs}",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE 3 — CLIMATOLOGICAL ROSE
# =============================================================================

def _fig_rose(obs_df, obs_var, model_df, model_var,
              title_suffix, dpi):
    """
    Climatological seasonal cycle:
    monthly means (obs + model) with ±1 std and full range shading.
    Seasons clearly delineated by background shading.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(f"Climatological seasonal cycle — {title_suffix}",
                 fontsize=13, fontweight="bold")

    months = np.arange(1, 13)

    # ── season background shading ─────────────────────────────────────────────
    season_spans = {
        "DJF": [(1, 2.5), (11.5, 12)],
        "MAM": [(2.5, 5.5)],
        "JJA": [(5.5, 8.5)],
        "SON": [(8.5, 11.5)],
    }
    for sname, spans in season_spans.items():
        col = SEASONS[sname]["color"]
        for x0, x1 in spans:
            ax.axvspan(x0, x1, alpha=0.06, color=col, zorder=0)
        ax.text(
            np.mean([s[0] + s[1] for s in spans]) / 2,
            0.97,
            SEASONS[sname]["label"],
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8,
            color=col, fontweight="bold",
        )

    # ── obs climatology ───────────────────────────────────────────────────────
    obs_monthly_mean = obs_df.groupby("month")[obs_var].mean()
    obs_monthly_std  = obs_df.groupby("month")[obs_var].std(ddof=1)
    obs_monthly_min  = obs_df.groupby("month")[obs_var].min()
    obs_monthly_max  = obs_df.groupby("month")[obs_var].max()
    obs_monthly_n    = obs_df.groupby("month")[obs_var].count()

    # full obs range
    ax.fill_between(
        obs_monthly_min.index,
        obs_monthly_min.values,
        obs_monthly_max.values,
        alpha=0.08, color="#2980b9",
        label="Obs full range",
    )
    # obs ±1 std
    ax.fill_between(
        obs_monthly_mean.index,
        (obs_monthly_mean - obs_monthly_std).values,
        (obs_monthly_mean + obs_monthly_std).values,
        alpha=0.20, color="#2980b9",
        label="Obs ±1 std",
    )
    # obs mean line
    ax.plot(obs_monthly_mean.index, obs_monthly_mean.values,
            "o-", color="#2980b9", lw=2.5, ms=7,
            label="Obs climatological mean", zorder=4)

    # ── model climatology ─────────────────────────────────────────────────────
    if model_df is not None:
        mod_monthly_mean = model_df.groupby("month")[model_var].mean()
        mod_monthly_std  = model_df.groupby("month")[model_var].std(ddof=1)
        mod_monthly_min  = model_df.groupby("month")[model_var].min()
        mod_monthly_max  = model_df.groupby("month")[model_var].max()

        ax.fill_between(
            mod_monthly_min.index,
            mod_monthly_min.values,
            mod_monthly_max.values,
            alpha=0.06, color="#c0392b",
            label="Model full range",
        )
        ax.fill_between(
            mod_monthly_mean.index,
            (mod_monthly_mean - mod_monthly_std).values,
            (mod_monthly_mean + mod_monthly_std).values,
            alpha=0.18, color="#c0392b",
            label="Model ±1 std",
        )
        ax.plot(mod_monthly_mean.index, mod_monthly_mean.values,
                "s--", color="#c0392b", lw=2.5, ms=7,
                label="Model climatological mean", zorder=4)

    # ── n per month annotation ────────────────────────────────────────────────
    y_bottom = ax.get_ylim()[0]
    for m in months:
        n = obs_monthly_n.get(m, 0)
        ax.text(m, ax.get_ylim()[0],
                f"n={n}", ha="center", va="bottom",
                fontsize=6, color="grey",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_xlabel("Month", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, zorder=1)

    fig.tight_layout()
    return fig


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def _print_summary(obs_stats, model_stats, obs_var, model_var):
    print(f"\n{'═'*65}")
    print(f"  SEASONAL STATISTICS  —  {obs_var} (obs)  vs  {model_var} (model)")
    print(f"{'═'*65}")
    print(f"  {'Season':<14} {'n_obs':>6}  {'obs_mean':>9}  "
          f"{'obs_std':>8}  {'mod_mean':>9}  {'bias':>7}")
    print(f"  {'─'*63}")

    for sname, sinfo in SEASONS.items():
        ost = obs_stats.get(sname)
        mst = model_stats.get(sname) if model_stats else None

        o_mean = f"{ost['mean']:>+9.3f}" if ost else f"{'–':>9}"
        o_std  = f"{ost['std']:>8.3f}"   if ost else f"{'–':>8}"
        o_n    = f"{ost['n']:>6}"         if ost else f"{'–':>6}"
        m_mean = f"{mst['mean']:>+9.3f}" if mst else f"{'–':>9}"

        if ost and mst:
            bias = f"{mst['mean'] - ost['mean']:>+7.3f}"
        else:
            bias = f"{'–':>7}"

        print(f"  {sinfo['label']:<14} {o_n}  {o_mean}  "
              f"{o_std}  {m_mean}  {bias}  °C")

    print(f"{'═'*65}\n")


# =============================================================================
# PUBLIC API
# =============================================================================

def seasonal_dashboard(
    obs_csv     : str | Path,
    obs_var     : str   = "TEMP",
    obs_qc_col  : str | None = "TEMP_QC",
    obs_qc_good : set   = frozenset({1, 2}),
    model_csv   : str | Path | None = None,
    model_var   : str   = "temp_celsius",
    model_sim   : str | None = None,
    out_prefix  : str   = "seasonal_dashboard",
    figure_dpi  : int   = DEFAULT_DPI,
    save_figures: bool  = True,
    show_figures: bool  = True,
) -> dict:
    """
    Build a three-figure seasonal dashboard from irregular in-situ
    records and optional ROMS model output.

    Parameters
    ----------
    obs_csv      : CSV from nsval.intake.cmems.scoop_point
    obs_var      : observed variable column name
    obs_qc_col   : QC flag column (None to skip)
    obs_qc_good  : accepted QC flag values
    model_csv    : CSV from nsval.intake.roms.extract_point (optional)
    model_var    : model variable column name
    model_sim    : filter to one simulation label (None = use all)
    out_prefix   : prefix for output PNG filenames
    figure_dpi   : PNG resolution
    save_figures : write PNG files
    show_figures : call plt.show()

    Returns
    -------
    dict with keys:
        'obs_stats'   — seasonal statistics for observations
        'model_stats' — seasonal statistics for model (or None)
        'figures'     — list of 3 Figure objects [maps, timeseries, rose]
    """
    print(f"\n{'═'*65}")
    print(f"  nsval.analyse.seasonal_dashboard")
    print(f"  Obs   : {Path(obs_csv).name}  ({obs_var})")
    if model_csv:
        print(f"  Model : {Path(model_csv).name}  ({model_var})")
    print(f"{'═'*65}\n")

    # ── load ──────────────────────────────────────────────────────────────────
    obs_df = _load_obs(obs_csv, obs_var, obs_qc_col, obs_qc_good)
    print(f"  Obs loaded   : {len(obs_df)} rows  "
          f"({obs_df['time'].min().date()} – {obs_df['time'].max().date()})")

    model_df = None
    if model_csv:
        model_df = _load_model(model_csv, model_var, model_sim)
        print(f"  Model loaded : {len(model_df)} rows  "
              f"({model_df['time'].min().date()} – "
              f"{model_df['time'].max().date()})")

    # ── statistics ────────────────────────────────────────────────────────────
    obs_stats   = _seasonal_stats(obs_df,   obs_var)
    model_stats = _seasonal_stats(model_df, model_var) if model_df is not None else None

    title_suffix = (f"{obs_var}  |  "
                    f"{obs_df['time'].min().year}–"
                    f"{obs_df['time'].max().year}")

    _print_summary(obs_stats, model_stats, obs_var, model_var)

    # ── figures ───────────────────────────────────────────────────────────────
    figs = []

    # Figure 1 — maps
    fig1 = _fig_maps(obs_df, obs_var, model_df, model_var,
                     title_suffix, figure_dpi)
    figs.append(fig1)
    if save_figures:
        fname = f"{out_prefix}_maps.png"
        fig1.savefig(fname, dpi=figure_dpi, bbox_inches="tight")
        print(f"  Saved → {fname}")

    # Figure 2 — timeseries
    fig2 = _fig_timeseries(obs_df, obs_var, model_df, model_var,
                           obs_stats, model_stats,
                           title_suffix, figure_dpi)
    figs.append(fig2)
    if save_figures:
        fname = f"{out_prefix}_timeseries.png"
        fig2.savefig(fname, dpi=figure_dpi, bbox_inches="tight")
        print(f"  Saved → {fname}")

    # Figure 3 — rose
    fig3 = _fig_rose(obs_df, obs_var, model_df, model_var,
                     title_suffix, figure_dpi)
    figs.append(fig3)
    if save_figures:
        fname = f"{out_prefix}_rose.png"
        fig3.savefig(fname, dpi=figure_dpi, bbox_inches="tight")
        print(f"  Saved → {fname}")

    if show_figures:
        plt.show()

    return {
        "obs_stats"   : obs_stats,
        "model_stats" : model_stats,
        "figures"     : figs,
    }


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(
        description="Seasonal temperature dashboard for the North Sea."
    )
    p.add_argument("--obs",       required=True, help="Obs CSV path")
    p.add_argument("--obs-var",   default="TEMP")
    p.add_argument("--obs-qc",    default="TEMP_QC",
                   help="QC column (empty string to disable)")
    p.add_argument("--model",     default=None,  help="Model CSV path")
    p.add_argument("--model-var", default="temp_celsius")
    p.add_argument("--model-sim", default=None)
    p.add_argument("--out",       default="seasonal_dashboard")
    p.add_argument("--no-save",   action="store_true")
    p.add_argument("--no-show",   action="store_true")
    args = p.parse_args()

    seasonal_dashboard(
        obs_csv      = args.obs,
        obs_var      = args.obs_var,
        obs_qc_col   = args.obs_qc or None,
        model_csv    = args.model,
        model_var    = args.model_var,
        model_sim    = args.model_sim,
        out_prefix   = args.out,
        save_figures = not args.no_save,
        show_figures = not args.no_show,
    )


if __name__ == "__main__":
    _cli()
