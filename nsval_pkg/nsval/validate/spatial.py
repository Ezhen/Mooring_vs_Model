"""
nsval.validate.spatial
──────────────────────
Spatial validation of CMEMS in-situ observations against ROMS model output.

Uses cartopy for real coastline maps.
Works with domain-wide extractions from scoop_region() rather than
single-point radius extractions.

Three figures per call:
  1. Spatial maps — bias and RMSE on real North Sea maps with coastlines
  2. Target diagram — normalised bias vs normalised RMSE, one point per station
  3. Obs vs model scatter — one point per station, coloured by latitude

Typical usage
─────────────
    from nsval.validate.spatial import validate_spatial

    validate_spatial(
        obs_csv      = "TEMP_NorthSea.csv",
        roms_folder  = "/scratch/.../CE2COAST_2006",
        roms_pattern = "Hindcast_CE2COAST_AVG_*.nc",
        roms_variable= "temp",
        obs_variable = "TEMP",
        season       = "JJA",
        out_prefix   = "examples/outputs/spatial_JJA",
    )
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from nsval.utils import haversine_km, decode_roms_time
from nsval.validate.metrics import compute_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONSTANTS
# =============================================================================

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

NS_LAT  = (48.0, 62.0)
NS_LON  = (-4.0, 10.0)
PROJ    = ccrs.PlateCarree()

DEFAULT_DPI     = 130
DEFAULT_S_LEVEL = -1


# =============================================================================
# CARTOPY MAP SETUP
# =============================================================================

def _setup_ax(ax, title=""):
    """Apply coastlines, land, sea, gridlines to a cartopy axis."""
    ax.set_extent([NS_LON[0], NS_LON[1], NS_LAT[0], NS_LAT[1]],
                  crs=PROJ)

    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "land", "50m",
        facecolor="#d4c9a8", edgecolor="grey", linewidth=0.5,
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "ocean", "50m",
        facecolor="#d4e8f0",
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m",
        facecolor="none", edgecolor="#444444", linewidth=0.8,
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "50m",
        facecolor="none", edgecolor="#7eb8d4", linewidth=0.4,
    ))
    ax.add_feature(cfeature.BORDERS, linewidth=0.4,
                   edgecolor="grey", linestyle=":")

    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color="grey", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LongitudeFormatter()
    gl.yformatter   = LatitudeFormatter()
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    if title:
        ax.set_title(title, fontweight="bold", fontsize=10)

    return ax


# =============================================================================
# ROMS SEASONAL MEAN → STATION INTERPOLATION
# =============================================================================

def _bilinear_idw(lat2d, lon2d, field2d, lat_t, lon_t):
    """Inverse-distance weighted interpolation using 3×3 neighbourhood."""
    dist     = haversine_km(lat2d, lon2d, lat_t, lon_t)
    flat_idx = np.argmin(dist)
    ei, xi   = np.unravel_index(flat_idx, dist.shape)
    n_eta, n_xi = lat2d.shape

    neighbours = []
    for de in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ni, nj = ei + de, xi + dx
            if 0 <= ni < n_eta and 0 <= nj < n_xi:
                val = field2d[ni, nj]
                if np.isfinite(val):
                    d = dist[ni, nj]
                    neighbours.append((d, val))

    if not neighbours:
        return np.nan
    dists  = np.array([d for d, _ in neighbours])
    vals   = np.array([v for _, v in neighbours])
    if np.any(dists == 0):
        return float(vals[dists == 0][0])
    weights = 1.0 / dists
    return float(np.sum(weights * vals) / np.sum(weights))


def _roms_seasonal_mean_at_stations(roms_folder, roms_pattern,
                                     roms_variable, s_level,
                                     season_months,
                                     station_lats, station_lons):
    files = sorted(roms_folder.glob(roms_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{roms_pattern}' in {roms_folder}")

    print(f"  ROMS: {len(files)} files")

    field_sum   = None
    field_count = None
    lat2d = lon2d = None

    for path in files:
        print(f"    {path.name} ...", end=" ", flush=True)
        try:
            ds = xr.open_dataset(path, decode_times=False)
        except Exception as exc:
            print(f"skip ({exc})")
            continue

        if roms_variable not in ds.data_vars:
            print("variable absent"); ds.close(); continue

        roms_times = decode_roms_time(ds["ocean_time"])
        months     = roms_times.month
        time_mask  = (np.isin(months, season_months)
                      if season_months else
                      np.ones(len(months), dtype=bool))

        if time_mask.sum() == 0:
            print("no timesteps in season"); ds.close(); continue

        if lat2d is None:
            lat2d = np.asarray(ds["lat_rho"])
            lon2d = np.asarray(ds["lon_rho"])

        da = ds[roms_variable]
        if "s_rho" in da.dims:
            da = da.isel(s_rho=s_level)

        vals = da.values[np.where(time_mask)[0]].astype(float)
        vals[vals > 1e36] = np.nan

        if field_sum is None:
            field_sum   = np.nansum(vals, axis=0)
            field_count = np.sum(np.isfinite(vals), axis=0)
        else:
            field_sum   += np.nansum(vals, axis=0)
            field_count += np.sum(np.isfinite(vals), axis=0)

        print(f"{time_mask.sum()} steps")
        ds.close()

    if field_sum is None:
        raise RuntimeError("No valid ROMS data found.")

    with np.errstate(invalid="ignore"):
        field_mean = np.where(field_count > 0,
                              field_sum / field_count, np.nan)

    n = len(station_lats)
    model_vals = np.full(n, np.nan)
    print(f"  Interpolating to {n} stations...")
    for i in range(n):
        model_vals[i] = _bilinear_idw(
            lat2d, lon2d, field_mean,
            station_lats[i], station_lons[i])

    return model_vals


# =============================================================================
# STATION METRICS
# =============================================================================

def _station_metrics(obs_df, obs_var, model_values, stations):
    rows = []
    for i, (_, st) in enumerate(stations.iterrows()):
        m_val  = model_values[i]
        obs_sub = obs_df[obs_df["source_file"] == st["source_file"]][obs_var].dropna()

        if len(obs_sub) == 0 or np.isnan(m_val):
            continue

        o_mean = float(obs_sub.mean())
        o_std  = float(obs_sub.std(ddof=1)) if len(obs_sub) > 1 else np.nan
        bias   = float(m_val - o_mean)
        rmse   = float(np.sqrt(np.mean((obs_sub.values - m_val)**2)))
        std_o  = o_std if (o_std and o_std > 0) else np.nan

        rows.append({
            "source_file" : st["source_file"],
            "file_lat"    : st["file_lat"],
            "file_lon"    : st["file_lon"],
            "n_obs"       : len(obs_sub),
            "obs_mean"    : o_mean,
            "obs_std"     : o_std,
            "model_mean"  : m_val,
            "bias"        : bias,
            "rmse"        : rmse,
            "norm_bias"   : bias   / std_o if std_o else np.nan,
            "norm_crmse"  : rmse   / std_o if std_o else np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# FIGURE 1 — SPATIAL MAPS WITH CARTOPY
# =============================================================================

def _fig_maps(met_df, season_label, dpi):
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"Spatial validation — {season_label}  |  "
        f"n stations = {len(met_df)}",
        fontsize=13, fontweight="bold",
    )

    ax1 = fig.add_subplot(1, 2, 1, projection=PROJ)
    ax2 = fig.add_subplot(1, 2, 2, projection=PROJ)

    _setup_ax(ax1, "Bias  (model − obs, °C)")
    _setup_ax(ax2, "RMSE  (°C)")

    # ── bias map — diverging ──────────────────────────────────────────────────
    finite_bias = met_df["bias"].dropna().values
    blim = max(float(np.percentile(np.abs(finite_bias), 95)), 0.1)
    bias_norm = mcolors.TwoSlopeNorm(vmin=-blim, vcenter=0, vmax=blim)

    sc1 = ax1.scatter(
        met_df["file_lon"], met_df["file_lat"],
        c=met_df["bias"], cmap="RdBu_r", norm=bias_norm,
        s=90, zorder=6, edgecolors="k", linewidths=0.5,
        transform=PROJ,
    )
    plt.colorbar(sc1, ax=ax1, label="Bias (°C)",
                 orientation="horizontal", pad=0.05, shrink=0.8)

    for _, row in met_df.iterrows():
        ax1.text(row["file_lon"] + 0.1, row["file_lat"] + 0.1,
                 f"{row['bias']:+.1f}",
                 fontsize=5, color="black",
                 transform=PROJ, zorder=7)

    # ── RMSE map — sequential ─────────────────────────────────────────────────
    rmse_max = max(float(np.nanpercentile(met_df["rmse"].values, 95)), 0.1)
    rmse_norm = mcolors.Normalize(vmin=0, vmax=rmse_max)

    sc2 = ax2.scatter(
        met_df["file_lon"], met_df["file_lat"],
        c=met_df["rmse"], cmap="YlOrRd", norm=rmse_norm,
        s=90, zorder=6, edgecolors="k", linewidths=0.5,
        transform=PROJ,
    )
    plt.colorbar(sc2, ax=ax2, label="RMSE (°C)",
                 orientation="horizontal", pad=0.05, shrink=0.8)

    for _, row in met_df.iterrows():
        ax2.text(row["file_lon"] + 0.1, row["file_lat"] + 0.1,
                 f"{row['rmse']:.1f}",
                 fontsize=5, color="black",
                 transform=PROJ, zorder=7)

    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE 2 — TARGET DIAGRAM
# =============================================================================

def _fig_target(met_df, season_label, dpi):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(
        f"Target diagram — {season_label}  |  n={len(met_df)} stations",
        fontsize=12, fontweight="bold",
    )

    for r, ls, lw in [(1.0,"-",1.2),(0.5,"--",0.7),(1.5,"--",0.7)]:
        ax.add_patch(plt.Circle((0,0), r, fill=False,
                                color="grey", ls=ls, lw=lw))
        ax.text(0, r+0.05, f"{r:.1f}σ",
                ha="center", fontsize=7, color="grey")

    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)

    valid = met_df.dropna(subset=["norm_bias","norm_crmse"])
    sc = ax.scatter(
        valid["norm_bias"], valid["norm_crmse"],
        c=valid["obs_mean"], cmap="RdYlBu_r",
        vmin=valid["obs_mean"].min(),
        vmax=valid["obs_mean"].max(),
        s=80, zorder=4, edgecolors="k", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Obs mean (°C)", shrink=0.8)

    for _, row in valid.iterrows():
        ax.annotate(
            Path(row["source_file"]).stem[:10],
            xy=(row["norm_bias"], row["norm_crmse"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=5, color="dimgrey",
        )

    lim = max(1.6,
              float(valid[["norm_bias","norm_crmse"]].abs().max().max()) + 0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.1, lim)
    ax.set_xlabel("Normalised bias  /  σ_obs", fontsize=10)
    ax.set_ylabel("Normalised RMSE  /  σ_obs", fontsize=10)
    ax.set_title(
        "Inside unit circle → model skill > climatology\n"
        "Left = cold bias  |  Right = warm bias",
        fontsize=8, color="grey",
    )
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE 3 — OBS VS MODEL SCATTER
# =============================================================================

def _fig_scatter(met_df, season_label, dpi):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(
        f"Station obs vs model — {season_label}",
        fontsize=12, fontweight="bold",
    )

    lims = [
        min(met_df["obs_mean"].min(), met_df["model_mean"].min()) - 0.5,
        max(met_df["obs_mean"].max(), met_df["model_mean"].max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="1:1")
    ob = float(met_df["bias"].mean())
    ax.plot(lims, [l + ob for l in lims], "r-", lw=0.8, alpha=0.6,
            label=f"Mean bias={ob:+.2f}°C")

    sc = ax.scatter(
        met_df["obs_mean"], met_df["model_mean"],
        c=met_df["file_lat"], cmap="plasma",
        s=70, zorder=4, edgecolors="k", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Station latitude (°N)", shrink=0.8)

    for _, row in met_df.iterrows():
        ax.annotate(
            f"n={row['n_obs']}",
            xy=(row["obs_mean"], row["model_mean"]),
            xytext=(4,-8), textcoords="offset points",
            fontsize=5, color="dimgrey",
        )

    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("Station obs mean (°C)",   fontsize=10)
    ax.set_ylabel("Interpolated model (°C)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    O = met_df["obs_mean"].values
    M = met_df["model_mean"].values
    mask = np.isfinite(O) & np.isfinite(M)
    if mask.sum() >= 3:
        met = compute_metrics(O[mask], M[mask])
        if met:
            ax.text(0.03, 0.97,
                    f"n={met['n_pairs']}\nbias={met['bias']:+.2f}°C\n"
                    f"RMSE={met['rmse']:.2f}°C\nr={met['r']:.3f}\n"
                    f"IoA={met['ioa']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", alpha=0.85))
    fig.tight_layout()
    return fig


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def _print_summary(met_df, season_label):
    print(f"\n{'═'*65}")
    print(f"  SPATIAL VALIDATION — {season_label}  |  "
          f"{len(met_df)} stations")
    print(f"{'═'*65}")
    print(f"  {'Station':<35} {'lat':>5} {'lon':>5} "
          f"{'obs':>6} {'mod':>6} {'bias':>7} {'rmse':>6}")
    print(f"  {'─'*63}")
    for _, row in met_df.sort_values("bias").iterrows():
        print(f"  {Path(row['source_file']).stem:<35} "
              f"{row['file_lat']:>5.2f} {row['file_lon']:>5.2f} "
              f"{row['obs_mean']:>6.2f} {row['model_mean']:>6.2f} "
              f"{row['bias']:>+6.2f} {row['rmse']:>6.2f}")
    print(f"  {'─'*63}")
    print(f"  {'MEAN':<35} {'':>5} {'':>5} "
          f"{met_df['obs_mean'].mean():>6.2f} "
          f"{met_df['model_mean'].mean():>6.2f} "
          f"{met_df['bias'].mean():>+6.2f} "
          f"{met_df['rmse'].mean():>6.2f}")
    print(f"{'═'*65}\n")


# =============================================================================
# PUBLIC API
# =============================================================================

def validate_spatial(
    obs_csv        : str | Path,
    roms_folder    : str | Path,
    roms_pattern   : str   = "Hindcast_CE2COAST_AVG_*.nc",
    roms_variable  : str   = "temp",
    obs_variable   : str   = "TEMP",
    obs_qc_col     : str | None = "TEMP_QC",
    obs_qc_good    : set   = frozenset({1, 2}),
    season         : str | None = None,
    s_level        : int   = DEFAULT_S_LEVEL,
    out_csv        : str | Path | None = None,
    out_prefix     : str | Path | None = "spatial_validation",
    figure_dpi     : int   = DEFAULT_DPI,
    show_figures   : bool  = True,
) -> dict:
    """
    Spatial validation of CMEMS observations against ROMS.

    Use with CSV from scoop_region() for domain-wide coverage.

    Parameters
    ----------
    obs_csv       : CSV from nsval.intake.cmems_region.scoop_region
    roms_folder   : ROMS AVG files folder
    roms_pattern  : glob pattern
    roms_variable : ROMS variable name
    obs_variable  : observed variable column name
    obs_qc_col    : QC column (None to skip)
    obs_qc_good   : accepted QC flag values
    season        : 'DJF'|'MAM'|'JJA'|'SON'|None (all)
    s_level       : ROMS vertical index (-1=surface, 0=bottom)
    out_csv       : save station metrics CSV here
    out_prefix    : base filename for output PNGs
    figure_dpi    : resolution
    show_figures  : call plt.show()

    Returns
    -------
    dict: 'station_metrics' (DataFrame), 'figures' (list of 3 figs)
    """
    obs_csv     = Path(obs_csv)
    roms_folder = Path(roms_folder)
    season_months = SEASONS.get(season) if season else None
    season_label  = season if season else "All seasons"

    print(f"\n{'═'*65}")
    print(f"  nsval.validate.spatial  (cartopy maps)")
    print(f"  Obs    : {obs_csv.name}  ({obs_variable})")
    print(f"  Model  : {roms_folder.name}  ({roms_variable})")
    print(f"  Season : {season_label}")
    print(f"{'═'*65}\n")

    # load obs
    obs_df = pd.read_csv(obs_csv)
    obs_df["time"] = pd.to_datetime(obs_df["time"],
                                     infer_datetime_format=True,
                                     errors="coerce")
    if obs_qc_col and obs_qc_col in obs_df.columns:
        obs_df = obs_df[obs_df[obs_qc_col].isin(set(obs_qc_good))]
    obs_df = obs_df.dropna(subset=[obs_variable])
    obs_df["month"] = obs_df["time"].dt.month
    if season_months:
        obs_df = obs_df[obs_df["month"].isin(season_months)]

    if "file_lat" not in obs_df.columns:
        raise ValueError(
            "obs_csv must have 'file_lat'/'file_lon' columns. "
            "Use scoop_region() output, not wide-format CSV.")

    stations = (obs_df.groupby("source_file")
                  .agg(file_lat=("file_lat","first"),
                       file_lon=("file_lon","first"),
                       n_obs   =(obs_variable,"count"))
                  .reset_index())

    print(f"  Obs: {len(obs_df)} rows  |  {len(stations)} stations")

    # interpolate ROMS
    model_vals = _roms_seasonal_mean_at_stations(
        roms_folder, roms_pattern, roms_variable, s_level,
        season_months, stations["file_lat"].values,
        stations["file_lon"].values,
    )

    # metrics
    met_df = _station_metrics(obs_df, obs_variable, model_vals, stations)

    if len(met_df) == 0:
        print("  No valid station pairs.")
        return {"station_metrics": pd.DataFrame(), "figures": []}

    _print_summary(met_df, season_label)

    if out_csv:
        met_df.to_csv(out_csv, index=False, float_format="%.4f")
        print(f"  Saved metrics → {out_csv}")

    prefix = str(out_prefix) if out_prefix else None
    figs   = []

    for fig_fn, suffix in [
        (_fig_maps,    "_maps.png"),
        (_fig_target,  "_target.png"),
        (_fig_scatter, "_scatter.png"),
    ]:
        fig = fig_fn(met_df, season_label, figure_dpi)
        figs.append(fig)
        if prefix:
            fname = prefix + suffix
            fig.savefig(fname, dpi=figure_dpi, bbox_inches="tight")
            print(f"  Saved → {fname}")

    if show_figures:
        plt.show()

    print(f"\n{'═'*65}\n")
    return {"station_metrics": met_df, "figures": figs}


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(description="Spatial model vs obs validation.")
    p.add_argument("--obs",          required=True)
    p.add_argument("--roms-folder",  required=True)
    p.add_argument("--roms-pattern", default="Hindcast_CE2COAST_AVG_*.nc")
    p.add_argument("--roms-var",     default="temp")
    p.add_argument("--obs-var",      default="TEMP")
    p.add_argument("--season",       default=None,
                   choices=["DJF","MAM","JJA","SON"])
    p.add_argument("--slevel",       type=int, default=-1)
    p.add_argument("--out-csv",      default=None)
    p.add_argument("--out-prefix",   default="spatial_validation")
    p.add_argument("--no-show",      action="store_true")
    args = p.parse_args()

    validate_spatial(
        obs_csv=args.obs, roms_folder=args.roms_folder,
        roms_pattern=args.roms_pattern, roms_variable=args.roms_var,
        obs_variable=args.obs_var, season=args.season,
        s_level=args.slevel, out_csv=args.out_csv,
        out_prefix=args.out_prefix, show_figures=not args.no_show,
    )


if __name__ == "__main__":
    _cli()
