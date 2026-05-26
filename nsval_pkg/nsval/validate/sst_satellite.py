"""
nsval.validate.sst_satellite
─────────────────────────────────────────────────────────────────────
Gridded SST validation: ROMS model vs satellite L4 SST product.

Replaces the legacy SST_bias_Satellite_ROMS.py script with a clean,
xarray-based, cartopy-mapped implementation.

Computes 8 metric fields per grid cell:
    bias          — mean model − satellite (°C)
    rmse          — root mean squared error (°C)
    mae           — mean absolute error (°C)
    correlation   — Pearson r across time
    std_ratio     — model std / satellite std (>1 = too variable)
    hit_rate      — fraction of timesteps with |bias| < 1°C
    bias_trend    — linear trend of annual bias (°C/year)
    amp_error     — seasonal amplitude error: (JJA−DJF) model vs sat

Produces:
    Figure 1 — 4-panel seasonal bias maps (cartopy)
    Figure 2 — 6-panel metric dashboard (bias, RMSE, r, std ratio,
                hit rate, bias trend)
    CSV      — per-year seasonal bias and RMSE scalars

Typical usage
─────────────
    from nsval.validate.sst_satellite import validate_sst

    validate_sst(
        roms_folder  = "/scratch/ulg/mast/eivanov/Output/CE2COAST_2006",
        roms_pattern = "Hindcast_CE2COAST_AVG_{year}_2c_era5_bcorr.nc",
        sat_folder   = "/CECI/home/ulg/mast/eivanov/Validation/Satellite_SST_replotted",
        sat_pattern  = "DMI_BAL_SST_L4_REP_OBSERVATIONS_{year}_sst.nc",
        years        = list(range(1993, 2021)),
        out_prefix   = "sst_validation",
    )
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from nsval.utils import decode_roms_time

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
SEASON_LABELS = {
    "DJF": "Winter (DJF)",
    "MAM": "Spring (MAM)",
    "JJA": "Summer (JJA)",
    "SON": "Autumn (SON)",
}

# North Sea domain (matching your original script)
DOMAIN = dict(lon_min=-3.5, lon_max=10.0, lat_min=48.5, lat_max=59.0)

PROJ    = ccrs.PlateCarree()
SAT_KELVIN_OFFSET = 273.15   # DMI L4 product stores SST in Kelvin

DEFAULT_DPI     = 150
DEFAULT_HIT_THR = 1.0        # °C threshold for hit rate


# =============================================================================
# CARTOPY MAP HELPER
# =============================================================================

def _setup_map_ax(ax, title="", cbar_label=""):
    """Apply standard North Sea cartopy formatting."""
    ax.set_extent(
        [DOMAIN["lon_min"], DOMAIN["lon_max"],
         DOMAIN["lat_min"], DOMAIN["lat_max"]],
        crs=PROJ,
    )
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "land", "50m",
        facecolor="#d4c9a8", edgecolor="#666666", linewidth=0.5))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m",
        facecolor="none", edgecolor="#333333", linewidth=0.8))
    ax.add_feature(cfeature.BORDERS,
                   linewidth=0.4, edgecolor="grey", linestyle=":")

    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="grey", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.FixedLocator(range(-4, 12, 2))
    gl.ylocator     = mticker.FixedLocator(range(48, 62, 2))
    gl.xformatter   = LongitudeFormatter()
    gl.yformatter   = LatitudeFormatter()
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    if title:
        ax.set_title(title, fontweight="bold", fontsize=10)

    return ax


# =============================================================================
# STEP 1 — LOAD ONE YEAR (ROMS + SATELLITE)
# =============================================================================

def _load_year(year: int,
               roms_folder: Path, roms_pattern: str,
               sat_folder:  Path, sat_pattern:  str,
               roms_variable: str,
               averaging_window: int) -> tuple[np.ndarray,
                                               np.ndarray,
                                               list[int],
                                               np.ndarray,
                                               np.ndarray] | None:
    """
    Load one year of ROMS and satellite SST, align temporally,
    return matched arrays.

    ROMS:      monthly (or daily) AVG files, surface level
    Satellite: daily L4 product — averaged over `averaging_window`
               days to match each ROMS timestep

    Returns
    -------
    sst_roms   : (n_times, n_eta, n_xi)
    sst_sat    : (n_times, n_eta, n_xi)  — regridded to ROMS grid
    months     : list of month integers
    lons       : (n_eta, n_xi)
    lats       : (n_eta, n_xi)
    or None if files not found
    """
    roms_path = roms_folder / roms_pattern.format(year=year)
    sat_path  = sat_folder  / sat_pattern.format(year=year)

    if not roms_path.exists():
        print(f"    ROMS not found: {roms_path.name}")
        return None
    if not sat_path.exists():
        print(f"    Satellite not found: {sat_path.name}")
        return None

    # ── ROMS ──────────────────────────────────────────────────────────────────
    ds_roms = xr.open_dataset(roms_path, decode_times=False)

    lons = np.asarray(ds_roms["lon_rho"])
    lats = np.asarray(ds_roms["lat_rho"])
    mask = np.asarray(ds_roms[roms_variable][0, -1].isnull()
                      if hasattr(ds_roms[roms_variable][0, -1], 'isnull')
                      else np.zeros_like(lons, dtype=bool))

    roms_times = decode_roms_time(ds_roms["ocean_time"])
    n_roms     = len(roms_times)

    # surface level
    sst_roms_raw = ds_roms[roms_variable].values[:, -1, :, :].astype(float)
    sst_roms_raw[sst_roms_raw > 1e36] = np.nan

    ds_roms.close()

    # ── Satellite ─────────────────────────────────────────────────────────────
    ds_sat = xr.open_dataset(sat_path, decode_times=False)

    sat_raw = ds_sat["analysed_sst"].values.astype(float) - SAT_KELVIN_OFFSET
    sat_raw[sat_raw > 100]  = np.nan
    sat_raw[sat_raw < -10]  = np.nan

    n_sat   = sat_raw.shape[0]
    sat_lat = np.asarray(ds_sat.get("lat", ds_sat.get("latitude",
              ds_sat.get("nav_lat", None))))
    sat_lon = np.asarray(ds_sat.get("lon", ds_sat.get("longitude",
              ds_sat.get("nav_lon", None))))

    ds_sat.close()

    # ── Temporal matching ─────────────────────────────────────────────────────
    # Average `averaging_window` satellite days per ROMS timestep
    # Assumes satellite is daily, ROMS is monthly (12 steps/year)
    sst_roms_out = []
    sst_sat_out  = []
    months_out   = []

    for i in range(n_roms):
        t_roms = roms_times[i]
        month  = t_roms.month

        # satellite window centred on ROMS timestep
        i_sat_centre = min(int(i * n_sat / n_roms), n_sat - 1)
        i0 = max(0, i_sat_centre - averaging_window // 2)
        i1 = min(n_sat, i0 + averaging_window)

        sat_mean = np.nanmean(sat_raw[i0:i1], axis=0)  # (n_lat_sat, n_lon_sat)

        # regrid satellite to ROMS grid using nearest-neighbour
        # (satellite L4 is typically ~4 km, ROMS ~2–5 km — nn is fine)
        sat_on_roms = _regrid_nn(sat_mean, sat_lat, sat_lon, lats, lons)

        sst_roms_out.append(sst_roms_raw[i])
        sst_sat_out.append(sat_on_roms)
        months_out.append(month)

    return (np.array(sst_roms_out),
            np.array(sst_sat_out),
            months_out,
            lons, lats)


def _regrid_nn(field_src: np.ndarray,
               lat_src: np.ndarray, lon_src: np.ndarray,
               lat_dst: np.ndarray, lon_dst: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbour regrid of field_src (lat_src × lon_src) onto
    (lat_dst, lon_dst) target grid.

    Works for both regular (1-D lat/lon) and curvilinear grids.
    """
    if lat_src.ndim == 1 and lon_src.ndim == 1:
        # regular grid → use broadcasting
        out = np.full(lat_dst.shape, np.nan)
        for ei in range(lat_dst.shape[0]):
            for xi in range(lat_dst.shape[1]):
                i_lat = int(np.argmin(np.abs(lat_src - lat_dst[ei, xi])))
                i_lon = int(np.argmin(np.abs(lon_src - lon_dst[ei, xi])))
                out[ei, xi] = field_src[i_lat, i_lon]
        return out
    else:
        # curvilinear → haversine search
        from nsval.utils import haversine_km
        out = np.full(lat_dst.shape, np.nan)
        for ei in range(lat_dst.shape[0]):
            for xi in range(lat_dst.shape[1]):
                dist = haversine_km(lat_src, lon_src,
                                    lat_dst[ei, xi], lon_dst[ei, xi])
                idx  = np.unravel_index(np.argmin(dist), dist.shape)
                out[ei, xi] = field_src[idx]
        return out


# =============================================================================
# STEP 2 — ACCUMULATE METRICS ACROSS YEARS
# =============================================================================

def _accumulate(roms_stack: np.ndarray,
                sat_stack:  np.ndarray,
                months_stack: list[int],
                hit_thr: float) -> dict:
    """
    Compute all metric fields from matched (n_total, n_eta, n_xi) arrays.

    Parameters
    ----------
    roms_stack   : (n_total, n_eta, n_xi)
    sat_stack    : (n_total, n_eta, n_xi)
    months_stack : list of month integers, length n_total
    hit_thr      : hit-rate threshold in °C

    Returns
    -------
    dict of 2-D arrays (n_eta, n_xi), one per metric
    """
    n, n_eta, n_xi = roms_stack.shape
    months = np.array(months_stack)

    bias_field = np.nanmean(roms_stack - sat_stack, axis=0)
    mae_field  = np.nanmean(np.abs(roms_stack - sat_stack), axis=0)
    rmse_field = np.sqrt(np.nanmean((roms_stack - sat_stack)**2, axis=0))

    hit_field  = np.nanmean(
        (np.abs(roms_stack - sat_stack) < hit_thr).astype(float),
        axis=0,
    )

    # Pearson r per cell
    r_field = np.full((n_eta, n_xi), np.nan)
    for ei in range(n_eta):
        for xi in range(n_xi):
            r_vals = roms_stack[:, ei, xi]
            s_vals = sat_stack[:,  ei, xi]
            valid  = np.isfinite(r_vals) & np.isfinite(s_vals)
            if valid.sum() >= 5:
                r, _ = stats.pearsonr(r_vals[valid], s_vals[valid])
                r_field[ei, xi] = r

    # std ratio per cell
    std_roms = np.nanstd(roms_stack, axis=0, ddof=1)
    std_sat  = np.nanstd(sat_stack,  axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        std_ratio = np.where(std_sat > 0, std_roms / std_sat, np.nan)

    # seasonal amplitude error
    jja_mask = np.isin(months, SEASONS["JJA"])
    djf_mask = np.isin(months, SEASONS["DJF"])
    amp_roms  = (np.nanmean(roms_stack[jja_mask], axis=0) -
                 np.nanmean(roms_stack[djf_mask], axis=0))
    amp_sat   = (np.nanmean(sat_stack[jja_mask],  axis=0) -
                 np.nanmean(sat_stack[djf_mask],  axis=0))
    amp_error = amp_roms - amp_sat

    # seasonal bias fields
    season_bias = {}
    for sname, smonths in SEASONS.items():
        smask = np.isin(months, smonths)
        if smask.sum() == 0:
            season_bias[sname] = np.full((n_eta, n_xi), np.nan)
        else:
            season_bias[sname] = np.nanmean(
                (roms_stack - sat_stack)[smask], axis=0)

    return {
        "bias"       : bias_field,
        "rmse"       : rmse_field,
        "mae"        : mae_field,
        "correlation": r_field,
        "std_ratio"  : std_ratio,
        "hit_rate"   : hit_field,
        "amp_error"  : amp_error,
        "season_bias": season_bias,
    }


def _per_year_scalars(roms_all: list, sat_all: list,
                      months_all: list, years: list) -> pd.DataFrame:
    """Compute per-year seasonal bias and RMSE scalars."""
    rows = []
    for yi, year in enumerate(years):
        r = np.array(roms_all[yi])
        s = np.array(sat_all[yi])
        m = np.array(months_all[yi])
        diff = r - s

        row = {"year": year}
        for sname, smonths in SEASONS.items():
            mask = np.isin(m, smonths)
            if mask.sum() > 0:
                row[f"bias_{sname}"] = round(
                    float(np.nanmean(diff[mask])), 3)
                row[f"rmse_{sname}"] = round(
                    float(np.sqrt(np.nanmean(diff[mask]**2))), 3)
            else:
                row[f"bias_{sname}"] = np.nan
                row[f"rmse_{sname}"] = np.nan

        row["bias_annual"] = round(float(np.nanmean(diff)), 3)
        row["rmse_annual"] = round(
            float(np.sqrt(np.nanmean(diff**2))), 3)
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# STEP 3 — FIGURES
# =============================================================================

def _fig_seasonal_bias(metrics: dict, lons: np.ndarray,
                       lats: np.ndarray, years: list, dpi: int):
    """Figure 1 — 4-panel seasonal bias maps."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"ROMS vs Satellite SST — Seasonal bias (model − sat, °C)\n"
        f"Years: {min(years)}–{max(years)}",
        fontsize=13, fontweight="bold",
    )

    bias_lim = 2.0
    cmap     = plt.cm.RdBu_r
    bounds   = np.linspace(-bias_lim, bias_lim, 17)
    norm     = mcolors.BoundaryNorm(bounds, cmap.N)

    for pi, (sname, slabel) in enumerate(SEASON_LABELS.items()):
        ax = fig.add_subplot(2, 2, pi + 1, projection=PROJ)
        _setup_map_ax(ax, title=slabel)

        sst = np.copy(metrics["season_bias"][sname])
        sst = np.clip(sst, -bias_lim * 1.05, bias_lim * 1.05)

        pcm = ax.pcolormesh(
            lons, lats, sst,
            cmap=cmap, norm=norm,
            transform=PROJ, shading="nearest",
        )
        # ±1°C contour
        ax.contour(lons, lats, sst, levels=[-1, 1],
                   colors=["#1a4fa8", "#a83232"],
                   linewidths=0.8, transform=PROJ)

        # domain mean annotation
        domain_mean = float(np.nanmean(sst))
        ax.text(0.03, 0.04,
                f"Domain mean: {domain_mean:+.2f}°C",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", alpha=0.85))

        plt.colorbar(pcm, ax=ax, label="Bias (°C)",
                     orientation="horizontal", pad=0.05,
                     shrink=0.85, ticks=[-2,-1,0,1,2])

    fig.tight_layout()
    return fig


def _fig_metric_dashboard(metrics: dict,
                           lons: np.ndarray, lats: np.ndarray,
                           years: list, dpi: int):
    """Figure 2 — 6-panel metric dashboard."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"ROMS vs Satellite SST — Metric dashboard\n"
        f"Years: {min(years)}–{max(years)}",
        fontsize=13, fontweight="bold",
    )

    panels = [
        # (field, title, cmap, symmetric, vmin, vmax, cbar_label)
        ("bias",        "Annual mean bias (°C)",
         "RdBu_r",    True,  -2.0,  2.0,  "Bias (°C)"),
        ("rmse",        "RMSE (°C)",
         "YlOrRd",    False,  0.0,  3.0,  "RMSE (°C)"),
        ("correlation", "Pearson r",
         "RdYlGn",    False, -1.0,  1.0,  "r"),
        ("std_ratio",   "Std ratio (model/sat)",
         "PuOr",      True,   0.5,  1.5,  "σ_model / σ_sat"),
        ("hit_rate",    f"Hit rate  |bias| < {DEFAULT_HIT_THR}°C",
         "RdYlGn",    False,  0.0,  1.0,  "Fraction"),
        ("amp_error",   "Seasonal amplitude error JJA−DJF (°C)",
         "RdBu_r",    True,  -2.0,  2.0,  "Amp. error (°C)"),
    ]

    for pi, (field, title, cmap_name, symmetric, vmin, vmax,
             cbar_label) in enumerate(panels):
        ax = fig.add_subplot(2, 3, pi + 1, projection=PROJ)
        _setup_map_ax(ax, title=title)

        data = np.copy(metrics[field])

        if symmetric:
            lim  = max(abs(vmin), abs(vmax))
            norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        pcm = ax.pcolormesh(
            lons, lats, data,
            cmap=cmap_name, norm=norm,
            transform=PROJ, shading="nearest",
        )

        # zero contour for symmetric fields
        if symmetric:
            ax.contour(lons, lats, data, levels=[0],
                       colors=["black"], linewidths=0.5,
                       transform=PROJ)

        domain_mean = float(np.nanmean(data))
        ax.text(0.03, 0.04,
                f"Mean: {domain_mean:+.3f}",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", alpha=0.85))

        plt.colorbar(pcm, ax=ax, label=cbar_label,
                     orientation="horizontal", pad=0.05, shrink=0.85)

    fig.tight_layout()
    return fig


def _fig_timeseries_scalars(df_scalars: pd.DataFrame, dpi: int):
    """Figure 3 — per-year seasonal bias and RMSE timeseries."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Per-year seasonal bias and RMSE — ROMS vs Satellite SST",
                 fontsize=12, fontweight="bold")

    colors = {"DJF":"#3498db","MAM":"#2ecc71",
              "JJA":"#e74c3c","SON":"#e67e22"}

    ax1, ax2 = axes

    for sname, color in colors.items():
        bias_col = f"bias_{sname}"
        rmse_col = f"rmse_{sname}"
        if bias_col in df_scalars.columns:
            ax1.plot(df_scalars["year"], df_scalars[bias_col],
                     "o-", color=color, lw=1.5, ms=5,
                     label=SEASON_LABELS[sname])
            ax2.plot(df_scalars["year"], df_scalars[rmse_col],
                     "o-", color=color, lw=1.5, ms=5,
                     label=SEASON_LABELS[sname])

    # annual mean
    ax1.plot(df_scalars["year"], df_scalars["bias_annual"],
             "k--", lw=2, ms=6, label="Annual mean")
    ax2.plot(df_scalars["year"], df_scalars["rmse_annual"],
             "k--", lw=2, ms=6, label="Annual mean")

    ax1.axhline(0, color="grey", lw=0.7)
    ax1.set_ylabel("Bias (°C)", fontsize=11)
    ax1.legend(fontsize=8, ncol=5, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Domain-mean bias", fontsize=10)

    ax2.set_ylabel("RMSE (°C)", fontsize=11)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Domain-mean RMSE", fontsize=10)
    ax2.legend(fontsize=8, ncol=5, loc="upper left")

    fig.tight_layout()
    return fig


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def _print_summary(metrics: dict, df_scalars: pd.DataFrame, years: list):
    print(f"\n{'═'*65}")
    print(f"  SST VALIDATION SUMMARY  |  {min(years)}–{max(years)}")
    print(f"{'═'*65}")

    for field, label, fmt in [
        ("bias",        "Annual bias (°C)",    "{:+.3f}"),
        ("rmse",        "RMSE (°C)",           "{:.3f}"),
        ("mae",         "MAE (°C)",            "{:.3f}"),
        ("correlation", "Pearson r",           "{:.3f}"),
        ("std_ratio",   "Std ratio",           "{:.3f}"),
        ("hit_rate",    "Hit rate",            "{:.3f}"),
        ("amp_error",   "Amplitude error (°C)","{:+.3f}"),
    ]:
        val = float(np.nanmean(metrics[field]))
        print(f"  {label:<28} domain mean: {fmt.format(val)}")

    print(f"\n  {'─'*63}")
    print(f"  SEASONAL BIAS  (domain mean, °C)")
    for sname, slabel in SEASON_LABELS.items():
        val = float(np.nanmean(metrics["season_bias"][sname]))
        print(f"  {slabel:<20}  {val:+.3f} °C")

    print(f"\n  {'─'*63}")
    print(f"  PER-YEAR ANNUAL BIAS (°C)")
    for _, row in df_scalars.iterrows():
        print(f"  {int(row['year'])}  bias={row['bias_annual']:+.3f}  "
              f"rmse={row['rmse_annual']:.3f}")

    print(f"{'═'*65}\n")


# =============================================================================
# PUBLIC API
# =============================================================================

def validate_sst(
    roms_folder     : str | Path,
    roms_pattern    : str   = "Hindcast_CE2COAST_AVG_{year}_2c_era5_bcorr.nc",
    sat_folder      : str | Path = ".",
    sat_pattern     : str   = "DMI_BAL_SST_L4_REP_OBSERVATIONS_{year}_sst.nc",
    roms_variable   : str   = "temp",
    years           : list  = None,
    averaging_window: int   = 30,
    hit_thr         : float = DEFAULT_HIT_THR,
    out_prefix      : str | Path | None = "sst_validation",
    out_csv         : str | Path | None = None,
    figure_dpi      : int   = DEFAULT_DPI,
    show_figures    : bool  = True,
) -> dict:
    """
    Validate ROMS SST against satellite L4 SST product.

    Parameters
    ----------
    roms_folder      : folder containing ROMS AVG files
    roms_pattern     : filename pattern with {year} placeholder
    sat_folder       : folder containing satellite NetCDF files
    sat_pattern      : filename pattern with {year} placeholder
    roms_variable    : ROMS temperature variable name (default 'temp')
    years            : list of years to process
    averaging_window : number of satellite days to average per ROMS step
    hit_thr          : threshold (°C) for hit rate calculation
    out_prefix       : base path for output PNG files
    out_csv          : save per-year scalars to this CSV
    figure_dpi       : PNG resolution
    show_figures     : call plt.show()

    Returns
    -------
    dict with keys:
        'metrics'    — dict of 2-D metric arrays
        'scalars'    — DataFrame of per-year seasonal scalars
        'lons'       — ROMS longitude grid
        'lats'       — ROMS latitude grid
        'figures'    — list of 3 Figure objects
    """
    if years is None:
        years = list(range(1993, 2021))

    roms_folder = Path(roms_folder)
    sat_folder  = Path(sat_folder)

    print(f"\n{'═'*65}")
    print(f"  nsval.validate.sst_satellite")
    print(f"  ROMS   : {roms_folder.name}")
    print(f"  Sat    : {sat_folder.name}")
    print(f"  Years  : {min(years)}–{max(years)}  ({len(years)} years)")
    print(f"  Window : {averaging_window} days per ROMS step")
    print(f"{'═'*65}\n")

    # ── load all years ────────────────────────────────────────────────────────
    roms_all   = []
    sat_all    = []
    months_all = []
    lons = lats = None
    valid_years = []

    for year in years:
        print(f"  Year {year} ...", end=" ", flush=True)
        result = _load_year(
            year, roms_folder, roms_pattern,
            sat_folder,  sat_pattern,
            roms_variable, averaging_window,
        )
        if result is None:
            print("skipped")
            continue
        sst_r, sst_s, months, lo, la = result
        roms_all.append(sst_r)
        sat_all.append(sst_s)
        months_all.append(months)
        if lons is None:
            lons, lats = lo, la
        valid_years.append(year)
        print(f"{len(months)} timesteps")

    if not roms_all:
        print("  No data loaded.")
        return {}

    # ── stack all years ───────────────────────────────────────────────────────
    roms_stack   = np.concatenate(roms_all,   axis=0)
    sat_stack    = np.concatenate(sat_all,    axis=0)
    months_stack = [m for ml in months_all for m in ml]

    print(f"\n  Total timesteps : {len(months_stack)}")
    print(f"  Grid shape      : {lons.shape}")

    # ── compute metrics ───────────────────────────────────────────────────────
    print("  Computing metric fields...")
    metrics = _accumulate(roms_stack, sat_stack, months_stack, hit_thr)

    # ── per-year scalars ──────────────────────────────────────────────────────
    df_scalars = _per_year_scalars(roms_all, sat_all,
                                    months_all, valid_years)

    _print_summary(metrics, df_scalars, valid_years)

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_csv or (str(out_prefix) + "_scalars.csv"
                           if out_prefix else None)
    if csv_path:
        df_scalars.to_csv(csv_path, index=False, float_format="%.3f")
        print(f"  Saved scalars → {csv_path}")

    # ── figures ───────────────────────────────────────────────────────────────
    figs    = []
    prefix  = str(out_prefix) if out_prefix else None

    for fig_fn, suffix in [
        (_fig_seasonal_bias,      "_seasonal_bias.png"),
        (_fig_metric_dashboard,   "_metrics.png"),
        (_fig_timeseries_scalars, "_timeseries.png"),
    ]:
        if fig_fn == _fig_timeseries_scalars:
            fig = fig_fn(df_scalars, figure_dpi)
        else:
            fig = fig_fn(metrics, lons, lats, valid_years, figure_dpi)

        figs.append(fig)
        if prefix:
            fname = prefix + suffix
            fig.savefig(fname, dpi=figure_dpi, bbox_inches="tight")
            print(f"  Saved → {fname}")

    if show_figures:
        plt.show()

    print(f"\n{'═'*65}\n")

    return {
        "metrics"    : metrics,
        "scalars"    : df_scalars,
        "lons"       : lons,
        "lats"       : lats,
        "figures"    : figs,
    }


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(
        description="ROMS vs satellite SST validation."
    )
    p.add_argument("--roms-folder",  required=True)
    p.add_argument("--roms-pattern",
                   default="Hindcast_CE2COAST_AVG_{year}_2c_era5_bcorr.nc")
    p.add_argument("--sat-folder",   required=True)
    p.add_argument("--sat-pattern",
                   default="DMI_BAL_SST_L4_REP_OBSERVATIONS_{year}_sst.nc")
    p.add_argument("--roms-var",     default="temp")
    p.add_argument("--years",        nargs="+", type=int, default=None)
    p.add_argument("--window",       type=int, default=30)
    p.add_argument("--out-prefix",   default="sst_validation")
    p.add_argument("--out-csv",      default=None)
    p.add_argument("--no-show",      action="store_true")
    args = p.parse_args()

    validate_sst(
        roms_folder      = args.roms_folder,
        roms_pattern     = args.roms_pattern,
        sat_folder       = args.sat_folder,
        sat_pattern      = args.sat_pattern,
        roms_variable    = args.roms_var,
        years            = args.years,
        averaging_window = args.window,
        out_prefix       = args.out_prefix,
        out_csv          = args.out_csv,
        show_figures     = not args.no_show,
    )


if __name__ == "__main__":
    _cli()
