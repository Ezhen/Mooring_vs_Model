"""
nsval.inventory
───────────────
Scan a folder of NetCDF files and produce:
  1. A CSV inventory of every file (dimensions, variables, time range, size).
  2. A CSV catalogue of unique variables with their metadata attributes.

Typical usage
─────────────
    from nsval.inventory import build_inventory, build_variable_catalogue

    build_inventory(
        folder  = "/path/to/MOORING_DATA",
        out_csv = "netcdf_inventory.csv",
    )

    build_variable_catalogue(
        inventory_csv = "netcdf_inventory.csv",
        out_csv       = "unique_variables.csv",
    )

Or run directly:
    python -m nsval.inventory --folder /path/to/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr


# ─────────────────────────────────────────────────────────────────────────────
# INVENTORY
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_file(path: Path) -> dict:
    """Return a summary dict for one NetCDF file."""
    try:
        ds = xr.open_dataset(path, decode_times=True)

        time_name = next(
            (c for c in ["time", "ocean_time", "TIME", "t"]
             if c in ds.coords or c in ds.variables),
            None,
        )

        if time_name:
            tv         = ds[time_name].values
            time_start = str(tv[0])  if len(tv) > 0 else None
            time_end   = str(tv[-1]) if len(tv) > 0 else None
            n_times    = len(tv)
        else:
            time_start = time_end = None
            n_times    = None

        summary = {
            "file"        : path.name,
            "path"        : str(path),
            "size_MB"     : round(path.stat().st_size / 1024**2, 2),
            "n_dimensions": len(ds.dims),
            "dimensions"  : dict(ds.dims),
            "n_variables" : len(ds.data_vars),
            "variables"   : ", ".join(ds.data_vars),
            "coordinates" : ", ".join(ds.coords),
            "time_variable": time_name,
            "n_times"     : n_times,
            "time_start"  : time_start,
            "time_end"    : time_end,
        }
        ds.close()
        return summary

    except Exception as exc:
        return {"file": path.name, "path": str(path), "error": str(exc)}


def build_inventory(folder: str | Path,
                    out_csv: str | Path = "netcdf_inventory.csv",
                    pattern: str = "*.nc") -> pd.DataFrame:
    """
    Scan *folder* for NetCDF files and write a summary CSV.

    Parameters
    ----------
    folder  : directory to scan
    out_csv : output CSV path
    pattern : glob pattern (default '*.nc')

    Returns
    -------
    pandas.DataFrame with one row per file
    """
    folder = Path(folder)
    files  = sorted(folder.glob(pattern))
    print(f"[inventory] Found {len(files)} NetCDF files in {folder}")

    summaries = []
    for f in files:
        print(f"  Reading {f.name}")
        summaries.append(_summarise_file(f))

    df = pd.DataFrame(summaries)
    df.to_csv(out_csv, index=False)
    print(f"[inventory] Saved → {out_csv}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VARIABLE CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────

def build_variable_catalogue(inventory_csv: str | Path,
                              out_csv: str | Path = "unique_variables.csv"
                              ) -> pd.DataFrame:
    """
    Read an inventory CSV and produce a catalogue of unique variable names
    with their metadata attributes (units, long_name, standard_name, dims).

    Parameters
    ----------
    inventory_csv : CSV produced by build_inventory()
    out_csv       : output CSV path

    Returns
    -------
    pandas.DataFrame with one row per unique variable
    """
    inv = pd.read_csv(inventory_csv)

    unique_vars = (
        inv["variables"]
        .dropna()
        .str.split(", ")
        .explode()
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    print(f"[catalogue] {len(unique_vars)} unique variables found in inventory")

    nc_files = (
        [Path(p) for p in inv["path"].dropna().unique()]
        if "path" in inv.columns
        else sorted(Path(".").glob("*.nc"))
    )

    rows  = []
    found = set()

    for nc_file in nc_files:
        if len(found) == len(unique_vars):
            break
        try:
            ds = xr.open_dataset(nc_file, decode_times=False)
            for var in unique_vars:
                if var in found:
                    continue
                if var in ds.variables:
                    da    = ds[var]
                    attrs = da.attrs
                    rows.append({
                        "variable"     : var,
                        "dims"         : " x ".join(da.dims),
                        "shape"        : str(da.shape),
                        "dtype"        : str(da.dtype),
                        "units"        : attrs.get("units", ""),
                        "long_name"    : attrs.get("long_name", ""),
                        "standard_name": attrs.get("standard_name", ""),
                        "description"  : attrs.get("description", ""),
                        "source_file"  : nc_file.name,
                    })
                    found.add(var)
            ds.close()
        except Exception as exc:
            print(f"  Could not read {nc_file.name}: {exc}")

    # Variables listed in inventory but not found in any file
    for var in unique_vars:
        if var not in found:
            rows.append({"variable": var, "source_file": "not_found"})

    df = pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[catalogue] Saved → {out_csv}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Build NetCDF inventory and variable catalogue."
    )
    parser.add_argument("--folder",    required=True,
                        help="Folder containing .nc files")
    parser.add_argument("--inventory", default="netcdf_inventory.csv",
                        help="Output inventory CSV (default: netcdf_inventory.csv)")
    parser.add_argument("--catalogue", default="unique_variables.csv",
                        help="Output catalogue CSV (default: unique_variables.csv)")
    parser.add_argument("--pattern",   default="*.nc",
                        help="Glob pattern (default: *.nc)")
    args = parser.parse_args()

    build_inventory(args.folder, args.inventory, args.pattern)
    build_variable_catalogue(args.inventory, args.catalogue)


if __name__ == "__main__":
    _cli()
