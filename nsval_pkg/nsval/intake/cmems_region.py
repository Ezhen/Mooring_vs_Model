"""
nsval.intake.cmems_region
──────────────────────────
Extract timeseries from ALL CMEMS in-situ stations inside a
bounding box — no radius, no target point, just a geographic box.

Designed for domain-wide extraction (e.g. the entire North Sea)
rather than point neighbourhood extraction (use scoop_point for that).

Typical usage
─────────────
    from nsval.intake.cmems_region import scoop_region

    scoop_region(
        folder   = "/path/to/MOORING_DATA",
        variable = "TEMP",
        lat_min  = 51.0,
        lat_max  = 62.0,
        lon_min  = -4.0,
        lon_max  = 10.0,
        out_csv  = "TEMP_NorthSea.csv",
    )
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nsval.utils import find_coord, detect_vertical_dim, add_ymd


# =============================================================================
# HELPERS  (same as cmems.py — kept local to avoid circular imports)
# =============================================================================

def _get_position(ds):
    lat_name = find_coord(ds, ["LATITUDE","latitude","LAT","lat"])
    lon_name = find_coord(ds, ["LONGITUDE","longitude","LON","lon"])
    if lat_name is None or lon_name is None:
        return None, None
    lat_arr = np.asarray(ds[lat_name]).ravel()
    lon_arr = np.asarray(ds[lon_name]).ravel()
    lat_arr = lat_arr[np.isfinite(lat_arr)]
    lon_arr = lon_arr[np.isfinite(lon_arr)]
    if len(lat_arr) == 0 or len(lon_arr) == 0:
        return None, None
    return float(np.median(lat_arr)), float(np.median(lon_arr))


def _find_qc(ds, variable):
    for cand in [f"{variable}_QC", f"{variable}_qc", f"QC_{variable}"]:
        if cand in ds.data_vars or cand in ds.coords:
            return cand
    return None


def _to_daily(df, time_col, variable, qc_col=None):
    df[time_col] = pd.to_datetime(df[time_col],
                                   infer_datetime_format=True,
                                   errors="coerce")
    if len(df) >= 2:
        median_dt   = df[time_col].sort_values().diff().median()
        is_subdaily = (not pd.isna(median_dt)) and (
            median_dt < pd.Timedelta(days=1))
    else:
        is_subdaily = False

    if is_subdaily:
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


# =============================================================================
# CORE — single file
# =============================================================================

def _process_file(path, variable, lat_min, lat_max,
                  lon_min, lon_max, vertical_mode, daily_average):

    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception as exc:
        print(f"    Could not open: {exc}")
        return None

    if variable not in ds.data_vars:
        ds.close()
        return None

    file_lat, file_lon = _get_position(ds)
    if file_lat is None:
        ds.close()
        return None

    # bounding box check
    if not (lat_min <= file_lat <= lat_max and
            lon_min <= file_lon <= lon_max):
        ds.close()
        return None

    print(f"  ✓ {path.name:<40}  {file_lat:.3f}°N  {file_lon:.3f}°E")

    # vertical reduction
    da   = ds[variable]
    zdim = detect_vertical_dim(da)

    if zdim is not None:
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

    # QC
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

    # build DataFrame
    out_vars = {variable: da}
    if qc_da is not None:
        out_vars[qc_name] = qc_da

    df = xr.Dataset(out_vars).to_dataframe().reset_index()

    time_col = next(
        (c for c in ["TIME","time","ocean_time","datetime"]
         if c in df.columns), None)

    if time_col:
        if daily_average:
            df = _to_daily(df, time_col, variable, qc_name)
        else:
            df.rename(columns={time_col: "time"}, inplace=True)
            df = add_ymd(df, "time")

    df = df.dropna(subset=[variable])

    df["file_lat"]      = round(file_lat, 5)
    df["file_lon"]      = round(file_lon, 5)
    df["lat_min"]       = lat_min
    df["lat_max"]       = lat_max
    df["lon_min"]       = lon_min
    df["lon_max"]       = lon_max
    df["vertical_mode"] = vertical_mode
    df["source_file"]   = path.name

    ds.close()
    return df


# =============================================================================
# PUBLIC API
# =============================================================================

# North Sea default bounding box
NORTH_SEA_BOX = dict(lat_min=51.0, lat_max=62.0,
                     lon_min=-4.0,  lon_max=10.0)


def scoop_region(
    folder        : str | Path,
    variable      : str   = "TEMP",
    lat_min       : float = 51.0,
    lat_max       : float = 62.0,
    lon_min       : float = -4.0,
    lon_max       : float = 10.0,
    vertical_mode : str   = "surface",
    daily_average : bool  = True,
    out_csv       : str | Path | None = None,
    pattern       : str   = "*.nc",
) -> pd.DataFrame:
    """
    Extract timeseries from all CMEMS stations inside a bounding box.

    Parameters
    ----------
    folder        : directory containing CMEMS .nc files
    variable      : CMEMS variable name, e.g. 'TEMP'
    lat_min/max   : latitude bounds (decimal degrees)
    lon_min/max   : longitude bounds (decimal degrees)
    vertical_mode : 'surface' | 'bottom' | 'mean' | 'all'
    daily_average : collapse sub-daily data to daily means
    out_csv       : save result here
    pattern       : glob pattern for NetCDF files

    Returns
    -------
    pandas.DataFrame with columns:
        time, {variable}, [QC], file_lat, file_lon,
        vertical_mode, source_file, year, month, day
    """
    folder = Path(folder)
    files  = sorted(folder.glob(pattern))

    print(f"\n{'═'*65}")
    print(f"  scoop_region  |  {variable}")
    print(f"  Box : {lat_min}–{lat_max}°N  /  {lon_min}–{lon_max}°E")
    print(f"  Files to scan: {len(files)}")
    print(f"{'═'*65}\n")

    tables = []
    for path in files:
        df = _process_file(path, variable, lat_min, lat_max,
                           lon_min, lon_max, vertical_mode, daily_average)
        if df is not None and len(df) > 0:
            tables.append(df)

    if not tables:
        print("\n  No stations found inside the bounding box.")
        return pd.DataFrame()

    result = pd.concat(tables, ignore_index=True)

    n_stations = result["source_file"].nunique()
    print(f"\n  Stations extracted : {n_stations}")
    print(f"  Total rows         : {len(result)}")
    print(f"  Period             : "
          f"{result['time'].min().date()} – "
          f"{result['time'].max().date()}")

    if out_csv:
        result.to_csv(out_csv, index=False)
        print(f"  Saved → {out_csv}")

    return result


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(
        description="Extract CMEMS in-situ timeseries for a bounding box."
    )
    p.add_argument("--folder",   required=True)
    p.add_argument("--variable", default="TEMP")
    p.add_argument("--lat-min",  type=float, default=51.0)
    p.add_argument("--lat-max",  type=float, default=62.0)
    p.add_argument("--lon-min",  type=float, default=-4.0)
    p.add_argument("--lon-max",  type=float, default=10.0)
    p.add_argument("--vertical", default="surface",
                   choices=["surface","bottom","mean","all"])
    p.add_argument("--out",      default=None)
    args = p.parse_args()

    out = args.out or f"{args.variable}_NorthSea.csv"
    scoop_region(
        folder        = args.folder,
        variable      = args.variable,
        lat_min       = args.lat_min,
        lat_max       = args.lat_max,
        lon_min       = args.lon_min,
        lon_max       = args.lon_max,
        vertical_mode = args.vertical,
        out_csv       = out,
    )


if __name__ == "__main__":
    _cli()
