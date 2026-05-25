"""
nsval.intake.cmems
──────────────────
Extract a point timeseries from a folder of CMEMS in-situ NetCDF files.

CMEMS in-situ format specifics
──────────────────────────────
- Each file represents one instrument / platform / cruise.
- LAT / LON are scalar coordinates or TIME-dimensioned coordinates —
  they describe where the instrument is, not a grid to search through.
- Data dimensions are TIME × DEPTH.
- QC flags follow the convention  {VARIABLE}_QC.

Typical usage
─────────────
    from nsval.intake.cmems import scoop_point

    scoop_point(
        folder       = "/path/to/MOORING_DATA",
        variable     = "TEMP",
        lat0         = 54.5,
        lon0         = 4.0,
        radius_km    = 50,
        vertical_mode= "surface",
        out_csv      = "TEMP_scoop_54.5_4.0.csv",
    )

Or run directly:
    python -m nsval.intake.cmems --variable TEMP --lat 54.5 --lon 4.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nsval.utils import (
    haversine_km,
    find_coord,
    detect_vertical_dim,
    add_ymd,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_instrument_position(ds) -> tuple[float | None, float | None]:
    """
    Return (lat, lon) of the instrument as plain floats.

    For fixed moorings, LAT/LON are scalar.
    For drifting platforms (Argo floats, gliders), they vary with TIME;
    we take the median to represent the track centre.
    """
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
    """Return the QC variable name for *variable*, or None if absent."""
    for candidate in [f"{variable}_QC", f"{variable}_qc", f"QC_{variable}"]:
        if candidate in ds.data_vars or candidate in ds.coords:
            return candidate
    return None


def _to_daily(df: pd.DataFrame, time_col: str, variable: str,
              qc_col: str | None = None) -> pd.DataFrame:
    """
    Collapse sub-daily data to daily means.
    Always renames the time column to 'time' and adds year/month/day.
    """
    df[time_col] = pd.to_datetime(df[time_col])

    is_subdaily = False
    if len(df) >= 2:
        median_dt = df[time_col].sort_values().diff().median()
        is_subdaily = (not pd.isna(median_dt)) and (
            median_dt < pd.Timedelta(days=1)
        )

    if is_subdaily:
        print("    Subdaily detected: averaging to daily")
        df["_date"] = df[time_col].dt.normalize()
        agg = {variable: "mean"}
        if qc_col and qc_col in df.columns:
            agg[qc_col] = "max"
        df = df.groupby("_date", as_index=False).agg(agg)
        df.rename(columns={"_date": "time"}, inplace=True)
    else:
        df.rename(columns={time_col: "time"}, inplace=True)

    df = add_ymd(df, "time")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CORE — single file
# ─────────────────────────────────────────────────────────────────────────────

def _process_file(path: Path, variable: str, lat0: float, lon0: float,
                  radius_km: float, vertical_mode: str,
                  daily_average: bool) -> pd.DataFrame | None:

    print(f"  Reading {path.name}")

    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception as exc:
        print(f"    Could not open: {exc}")
        return None

    if variable not in ds.data_vars:
        print(f"    {variable} not found, skipping")
        ds.close()
        return None

    # ── Position check ───────────────────────────────────────────────────────
    file_lat, file_lon = _get_instrument_position(ds)

    if file_lat is None:
        print("    No lat/lon found, skipping")
        ds.close()
        return None

    dist_km = haversine_km(file_lat, file_lon, lat0, lon0)
    print(f"    {file_lat:.3f}°N  {file_lon:.3f}°E  →  {dist_km:.1f} km")

    in_north_sea = (48 <= file_lat <= 60) and (-4 <= file_lon <= 10)
    if not in_north_sea:
        print("    Outside North Sea bounds, skipping")
        ds.close()
        return None

    if dist_km > radius_km:
        print(f"    Outside radius ({radius_km} km), skipping")
        ds.close()
        return None

    # ── Vertical reduction ───────────────────────────────────────────────────
    da   = ds[variable]
    zdim = detect_vertical_dim(da)

    if zdim is not None:
        print(f"    Vertical dim: {zdim}  mode={vertical_mode}")
        if vertical_mode == "surface":
            da = da.isel({zdim: -1})
        elif vertical_mode == "bottom":
            da = da.isel({zdim: 0})
        elif vertical_mode == "mean":
            da = da.mean(zdim, skipna=True)
        elif vertical_mode == "all":
            pass
        else:
            raise ValueError(f"Unknown vertical_mode: {vertical_mode!r}")

    # ── QC ───────────────────────────────────────────────────────────────────
    qc_name = _find_qc(ds, variable)
    qc_da   = None

    if qc_name:
        qc_da = ds[qc_name]
        if zdim and zdim in qc_da.dims:
            if vertical_mode == "surface":
                qc_da = qc_da.isel({zdim: -1})
            elif vertical_mode == "bottom":
                qc_da = qc_da.isel({zdim: 0})
            elif vertical_mode in ("mean", "all"):
                qc_da = qc_da.max(zdim, skipna=True)

    # ── DataFrame ────────────────────────────────────────────────────────────
    out_vars = {variable: da}
    if qc_da is not None:
        out_vars[qc_name] = qc_da

    df = xr.Dataset(out_vars).to_dataframe().reset_index()

    time_col = next(
        (c for c in ["TIME", "time", "ocean_time", "datetime"]
         if c in df.columns),
        None,
    )

    if time_col:
        if daily_average:
            df = _to_daily(df, time_col, variable, qc_name)
        else:
            df.rename(columns={time_col: "time"}, inplace=True)
            df = add_ymd(df, "time")
    else:
        print("    Warning: no time column in output")

    df = df.dropna(subset=[variable])

    df["file_lat"]      = round(file_lat, 5)
    df["file_lon"]      = round(file_lon, 5)
    df["dist_km"]       = round(dist_km, 2)
    df["lat_center"]    = lat0
    df["lon_center"]    = lon0
    df["radius_km"]     = radius_km
    df["vertical_mode"] = vertical_mode
    df["source_file"]   = path.name

    ds.close()
    print(f"    → {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def scoop_point(
    folder        : str | Path,
    variable      : str,
    lat0          : float,
    lon0          : float,
    radius_km     : float  = 50.0,
    vertical_mode : str    = "surface",
    daily_average : bool   = True,
    out_csv       : str | Path | None = None,
    pattern       : str    = "*.nc",
) -> pd.DataFrame:
    """
    Extract a point timeseries from all CMEMS in-situ files in *folder*
    that lie within *radius_km* of (*lat0*, *lon0*).

    Parameters
    ----------
    folder        : directory containing .nc files
    variable      : CMEMS variable name, e.g. 'TEMP', 'NTRA'
    lat0, lon0    : target coordinates (decimal degrees)
    radius_km     : search radius in kilometres
    vertical_mode : 'surface' | 'bottom' | 'mean' | 'all'
    daily_average : collapse sub-daily data to daily means
    out_csv       : if given, save result to this path
    pattern       : glob pattern for NetCDF files

    Returns
    -------
    pandas.DataFrame with columns: time, {variable}, [QC], file_lat,
        file_lon, dist_km, lat_center, lon_center, radius_km,
        vertical_mode, source_file, year, month, day
    """
    folder = Path(folder)
    files  = sorted(folder.glob(pattern))
    print(f"\n[cmems.scoop_point] {variable} | {lat0}°N {lon0}°E | "
          f"r={radius_km} km | {len(files)} files")

    tables = []
    for path in files:
        df = _process_file(path, variable, lat0, lon0,
                           radius_km, vertical_mode, daily_average)
        if df is not None and len(df) > 0:
            tables.append(df)

    if not tables:
        print("[cmems.scoop_point] No data extracted.")
        return pd.DataFrame()

    result = pd.concat(tables, ignore_index=True)

    if out_csv:
        result.to_csv(out_csv, index=False)
        print(f"[cmems.scoop_point] Saved → {out_csv}  ({len(result)} rows)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Extract CMEMS in-situ point timeseries."
    )
    p.add_argument("--folder",   required=True)
    p.add_argument("--variable", default="TEMP")
    p.add_argument("--lat",      type=float, required=True)
    p.add_argument("--lon",      type=float, required=True)
    p.add_argument("--radius",   type=float, default=50.0)
    p.add_argument("--vertical", default="surface",
                   choices=["surface","bottom","mean","all"])
    p.add_argument("--out",      default=None)
    args = p.parse_args()

    out = args.out or f"{args.variable}_scoop_{args.lat}_{args.lon}.csv"
    scoop_point(args.folder, args.variable, args.lat, args.lon,
                args.radius, args.vertical, out_csv=out)


if __name__ == "__main__":
    _cli()
