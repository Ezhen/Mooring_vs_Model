"""
nsval.validate.vertical
────────────────────────
Find all CMEMS in-situ stations within a radius of a target point
that have a sufficient observational time span, extract their full
vertical temperature profiles, and produce a time-depth heatmap.

Uses the CSV inventory produced by nsval.inventory (or the raw NetCDF
folder directly) to locate candidate stations, then reads their
TIME × DEPTH temperature data.

Typical usage
─────────────
    from nsval.validate.vertical import analyse_vertical

    analyse_vertical(
        folder         = "/path/to/MOORING_DATA",
        variable       = "TEMP",
        lat0           = 54.5,
        lon0           = 4.0,
        radius_km      = 100,
        min_years      = 1,
        out_csv        = "vertical_TEMP_54.5_4.0.csv",
        out_figure     = "vertical_TEMP_54.5_4.0.png",
    )
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm

from nsval.utils import haversine_km, find_coord, detect_vertical_dim, add_ymd


# =============================================================================
# USER-TUNABLE DEFAULTS
# =============================================================================

DEFAULT_VARIABLE  = "TEMP"
DEFAULT_RADIUS_KM = 100.0
DEFAULT_MIN_YEARS = 1.0       # minimum record length to include a station
DEFAULT_QC_GOOD   = {1, 2}
DEFAULT_DEPTH_MAX = 200.0     # metres — clip display below this
DEFAULT_DPI       = 130


# =============================================================================
# HELPERS
# =============================================================================

def _get_position(ds) -> tuple[float | None, float | None]:
    """Return (lat, lon) of the instrument as plain floats."""
    lat_name = find_coord(ds, ["LATITUDE", "latitude", "LAT", "lat"])
    lon_name = find_coord(ds, ["LONGITUDE", "longitude", "LON", "lon"])
    if lat_name is None or lon_name is None:
        return None, None
    lat_arr = np.asarray(ds[lat_name]).ravel()
    lon_arr = np.asarray(ds[lon_name]).ravel()
    lat_arr = lat_arr[np.isfinite(lat_arr)]
    lon_arr = lon_arr[np.isfinite(lon_arr)]
    if len(lat_arr) == 0 or len(lon_arr) == 0:
        return None, None
    return float(np.median(lat_arr)), float(np.median(lon_arr))


def _find_qc(ds, variable: str) -> str | None:
    for cand in [f"{variable}_QC", f"{variable}_qc", f"QC_{variable}"]:
        if cand in ds.data_vars or cand in ds.coords:
            return cand
    return None


def _get_depth_values(ds, zdim: str) -> np.ndarray | None:
    """
    Try to return actual depth values in metres.
    CMEMS stores depth as a coordinate named DEPH, depth, z, etc.
    Falls back to integer level indices if no coordinate is found.
    """
    for cand in ["DEPH", "deph", "DEPTH", "depth", "z", "lev", "level"]:
        if cand in ds.coords or cand in ds.data_vars:
            arr = np.asarray(ds[cand]).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                return arr
    # fallback: use dimension indices
    return None


# =============================================================================
# EXTRACT ONE FILE
# =============================================================================

def _extract_file(path: Path, variable: str,
                  lat0: float, lon0: float,
                  radius_km: float, min_years: float,
                  qc_good: set) -> pd.DataFrame | None:
    """
    Extract full TIME × DEPTH profile from one CMEMS file.
    Returns a tidy DataFrame with columns:
        time, depth, {variable}, file_lat, file_lon, dist_km, source_file
    or None if the file doesn't qualify.
    """
    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception as exc:
        print(f"    Could not open {path.name}: {exc}")
        return None

    # ── variable present? ────────────────────────────────────────────────────
    if variable not in ds.data_vars:
        ds.close()
        return None

    # ── position & radius ────────────────────────────────────────────────────
    file_lat, file_lon = _get_position(ds)
    if file_lat is None:
        ds.close()
        return None

    in_north_sea = (48 <= file_lat <= 60) and (-4 <= file_lon <= 10)
    if not in_north_sea:
        ds.close()
        return None

    dist_km = haversine_km(file_lat, file_lon, lat0, lon0)
    if dist_km > radius_km:
        ds.close()
        return None

    # ── vertical dimension ───────────────────────────────────────────────────
    da   = ds[variable]
    zdim = detect_vertical_dim(da)

    if zdim is None:
        # no vertical dimension — single-depth instrument, skip
        ds.close()
        return None

    # ── depth values ─────────────────────────────────────────────────────────
    depth_vals = _get_depth_values(ds, zdim)
    n_depths   = da.sizes[zdim]

    if depth_vals is None or len(depth_vals) != n_depths:
        depth_vals = np.arange(n_depths, dtype=float)

    # ── time ─────────────────────────────────────────────────────────────────
    time_name = find_coord(ds, ["TIME", "time", "ocean_time", "datetime"])
    if time_name is None:
        ds.close()
        return None

    times = pd.to_datetime(ds[time_name].values)

    # ── minimum time span ────────────────────────────────────────────────────
    if len(times) < 2:
        ds.close()
        return None

    span_years = (times.max() - times.min()).days / 365.25
    if span_years < min_years:
        print(f"    {path.name}: span={span_years:.2f} yr < {min_years} yr, skipping")
        ds.close()
        return None

    # ── QC mask ──────────────────────────────────────────────────────────────
    qc_name = _find_qc(ds, variable)
    if qc_name:
        qc_arr = ds[qc_name].values   # (TIME, DEPTH) or same shape as da
    else:
        qc_arr = None

    # ── extract values ───────────────────────────────────────────────────────
    vals = da.values.astype(float)    # (TIME, DEPTH) — CMEMS standard order
    vals[vals > 1e36] = np.nan        # fill value

    if qc_arr is not None:
        try:
            bad = ~np.isin(qc_arr, list(qc_good))
            vals[bad] = np.nan
        except Exception:
            pass

    # ── build tidy DataFrame — vectorised, no Python loops ───────────────────
    # vals shape: (n_times, n_depths)
    # We broadcast times and depths into flat arrays then build in one shot.
    n_times  = len(times)
    n_depths = len(depth_vals)

    # ensure vals is exactly (n_times, n_depths)
    if vals.ndim == 1:
        # only one depth level — shouldn't reach here but handle gracefully
        vals = vals[:, np.newaxis]

    time_rep  = np.repeat(times,       n_depths)           # each time n_depths times
    depth_rep = np.tile(depth_vals,    n_times)            # depth array repeated n_times

    df = pd.DataFrame({
        "time"       : time_rep,
        "depth"      : depth_rep.round(2),
        variable     : vals.ravel(),
        "file_lat"   : round(file_lat, 5),
        "file_lon"   : round(file_lon, 5),
        "dist_km"    : round(dist_km, 2),
        "source_file": path.name,
    })

    df = df.dropna(subset=[variable])
    df = add_ymd(df, "time")

    # ── optional: collapse sub-daily to daily means ───────────────────────────
    # sub-hourly moorings can produce millions of rows; daily mean keeps
    # the heatmap readable and the CSV manageable
    df["_date"] = df["time"].dt.normalize()
    df = (df.groupby(["_date", "depth"], as_index=False)
            .agg({variable: "mean",
                  "file_lat": "first", "file_lon": "first",
                  "dist_km": "first",  "source_file": "first",
                  "year": "first", "month": "first", "day": "first"})
            .rename(columns={"_date": "time"}))
    df["time"] = pd.to_datetime(df["time"])

    ds.close()
    print(f"    {path.name}: {len(times)} times × {n_depths} depths  "
          f"| span={span_years:.1f} yr  | {dist_km:.1f} km  "
          f"| {len(df)} valid rows")
    return df


# =============================================================================
# FIGURE
# =============================================================================

def _make_figure(df: pd.DataFrame, variable: str,
                 lat0: float, lon0: float,
                 depth_max: float, dpi: int,
                 out_figure: str | Path | None):
    """
    One figure per station (source_file), showing time-depth heatmap
    of temperature with a mean profile panel on the right.
    """
    stations = df["source_file"].unique()
    figs     = []

    for sfile in stations:
        sub = df[df["source_file"] == sfile].copy()
        slat = sub["file_lat"].iloc[0]
        slon = sub["file_lon"].iloc[0]
        dist = sub["dist_km"].iloc[0]

        # clip depth
        sub = sub[sub["depth"] <= depth_max]
        if len(sub) == 0:
            continue

        # pivot to 2-D: rows = depth, cols = time
        pivot = (sub.pivot_table(index="depth", columns="time",
                                 values=variable, aggfunc="mean")
                    .sort_index())

        depths = pivot.index.values
        times  = pivot.columns
        Z      = pivot.values   # (n_depths, n_times)

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 5),
            gridspec_kw={"width_ratios": [5, 1], "wspace": 0.04},
        )

        stem = Path(sfile).stem
        fig.suptitle(
            f"{variable} time-depth  |  {stem}\n"
            f"{slat:.3f}°N  {slon:.3f}°E  |  {dist:.1f} km from target  "
            f"({lat0}°N  {lon0}°E)",
            fontsize=11, fontweight="bold",
        )

        # ── heatmap ───────────────────────────────────────────────────────
        vmin = np.nanpercentile(Z, 2)
        vmax = np.nanpercentile(Z, 98)

        pcm = ax1.pcolormesh(
            times, depths, Z,
            cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
            shading="nearest",
        )
        ax1.set_ylim(depth_max, 0)   # depth increases downward
        ax1.set_ylabel("Depth (m)", fontsize=11)
        ax1.set_xlabel("Time",      fontsize=11)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax1.grid(True, alpha=0.2, color="white")

        cbar = plt.colorbar(pcm, ax=ax1, pad=0.01, fraction=0.025)
        cbar.set_label(f"{variable} (°C)", fontsize=9)

        # ── mean profile ──────────────────────────────────────────────────
        mean_profile = np.nanmean(Z, axis=1)
        std_profile  = np.nanstd(Z, axis=1, ddof=1)

        ax2.fill_betweenx(depths,
                          mean_profile - std_profile,
                          mean_profile + std_profile,
                          alpha=0.25, color="#e67e22", label="±1 std")
        ax2.plot(mean_profile, depths, "o-", color="#c0392b",
                 lw=1.5, ms=4, label="Mean")
        ax2.set_ylim(depth_max, 0)
        ax2.set_xlabel("°C", fontsize=9)
        ax2.yaxis.set_ticklabels([])
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Mean\nprofile", fontsize=9)
        ax2.legend(fontsize=7)

        fig.tight_layout()

        if out_figure:
            stem_out = Path(str(out_figure)).stem
            ext      = Path(str(out_figure)).suffix or ".png"
            fname    = f"{stem_out}_{Path(sfile).stem}{ext}"
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            print(f"  Saved figure → {fname}")

        figs.append(fig)

    return figs


# =============================================================================
# PUBLIC API
# =============================================================================

def analyse_vertical(
    folder        : str | Path,
    variable      : str   = DEFAULT_VARIABLE,
    lat0          : float = 54.5,
    lon0          : float = 4.0,
    radius_km     : float = DEFAULT_RADIUS_KM,
    min_years     : float = DEFAULT_MIN_YEARS,
    qc_good       : set   = frozenset(DEFAULT_QC_GOOD),
    depth_max     : float = DEFAULT_DEPTH_MAX,
    out_csv       : str | Path | None = None,
    out_figure    : str | Path | None = None,
    figure_dpi    : int   = DEFAULT_DPI,
    show_figures  : bool  = True,
    pattern       : str   = "*.nc",
) -> dict:
    """
    Find all CMEMS stations within radius of (lat0, lon0) with sufficient
    time span, extract TIME × DEPTH profiles, plot and save.

    Parameters
    ----------
    folder      : directory containing CMEMS .nc files
    variable    : variable name, e.g. 'TEMP'
    lat0, lon0  : target coordinates (decimal degrees)
    radius_km   : search radius in kilometres
    min_years   : minimum record length to include a station
    qc_good     : accepted CMEMS QC flag values
    depth_max   : maximum depth to display in figures (metres)
    out_csv     : save extracted profiles to this CSV path
    out_figure  : base filename for figures (one per station)
    figure_dpi  : PNG resolution
    show_figures: call plt.show()
    pattern     : glob pattern for NetCDF files

    Returns
    -------
    dict with keys:
        'data'     — combined DataFrame (time, depth, variable, metadata)
        'stations' — summary DataFrame (one row per station)
        'figures'  — list of Figure objects
    """
    folder = Path(folder)
    files  = sorted(folder.glob(pattern))

    print(f"\n{'═'*65}")
    print(f"  nsval.validate.vertical  |  {variable}")
    print(f"  Target : {lat0}°N  {lon0}°E  |  r={radius_km} km")
    print(f"  Min span: {min_years} yr  |  {len(files)} files to scan")
    print(f"{'═'*65}\n")

    tables = []
    for path in files:
        df = _extract_file(path, variable, lat0, lon0,
                           radius_km, min_years, set(qc_good))
        if df is not None and len(df) > 0:
            tables.append(df)

    if not tables:
        print("  No qualifying stations found.")
        return {"data": pd.DataFrame(), "stations": pd.DataFrame(),
                "figures": []}

    data = pd.concat(tables, ignore_index=True)

    # ── station summary ───────────────────────────────────────────────────────
    stations = (
        data.groupby("source_file")
        .agg(
            file_lat  = ("file_lat",  "first"),
            file_lon  = ("file_lon",  "first"),
            dist_km   = ("dist_km",   "first"),
            time_start= ("time",      "min"),
            time_end  = ("time",      "max"),
            n_records = (variable,    "count"),
            depth_min = ("depth",     "min"),
            depth_max = ("depth",     "max"),
            mean_temp = (variable,    "mean"),
            std_temp  = (variable,    "std"),
        )
        .reset_index()
        .sort_values("dist_km")
    )
    stations["span_years"] = (
        (pd.to_datetime(stations["time_end"]) -
         pd.to_datetime(stations["time_start"])).dt.days / 365.25
    ).round(2)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*63}")
    print(f"  QUALIFYING STATIONS  ({len(stations)} found)")
    print(f"  {'─'*63}")
    for _, row in stations.iterrows():
        print(f"  {row['source_file']:<35}  "
              f"{row['dist_km']:>6.1f} km  "
              f"span={row['span_years']:.1f} yr  "
              f"depth={row['depth_min']:.0f}–{row['depth_max']:.0f} m  "
              f"mean={row['mean_temp']:.2f}°C")

    # ── save CSV ──────────────────────────────────────────────────────────────
    if out_csv:
        data.to_csv(out_csv, index=False)
        print(f"\n  Saved data     → {out_csv}  ({len(data)} rows)")
        stem     = Path(str(out_csv)).stem
        stat_csv = Path(str(out_csv)).with_name(f"{stem}_stations.csv")
        stations.to_csv(stat_csv, index=False)
        print(f"  Saved stations → {stat_csv}")

    # ── figures ───────────────────────────────────────────────────────────────
    figs = _make_figure(data, variable, lat0, lon0,
                        depth_max, figure_dpi, out_figure)

    if show_figures:
        plt.show()

    return {"data": data, "stations": stations, "figures": figs}


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(
        description="CMEMS vertical temperature profile analysis."
    )
    p.add_argument("--folder",    required=True)
    p.add_argument("--variable",  default="TEMP")
    p.add_argument("--lat",       type=float, required=True)
    p.add_argument("--lon",       type=float, required=True)
    p.add_argument("--radius",    type=float, default=100.0)
    p.add_argument("--min-years", type=float, default=1.0)
    p.add_argument("--depth-max", type=float, default=200.0)
    p.add_argument("--out-csv",   default=None)
    p.add_argument("--out-fig",   default=None)
    p.add_argument("--no-show",   action="store_true")
    args = p.parse_args()

    analyse_vertical(
        folder       = args.folder,
        variable     = args.variable,
        lat0         = args.lat,
        lon0         = args.lon,
        radius_km    = args.radius,
        min_years    = args.min_years,
        depth_max    = args.depth_max,
        out_csv      = args.out_csv,
        out_figure   = args.out_fig,
        show_figures = not args.no_show,
    )


if __name__ == "__main__":
    _cli()
