"""
nsval.inspect
─────────────
Lightweight dataset inspection utilities — no data loading required.

Functions
─────────
    estimate_memory_load()  — estimate array size before loading
    find_variable()         — keyword search across variable catalogue
    qc_summary()            — CMEMS QC flag breakdown per variable

Typical usage
─────────────
    from nsval.inspect import estimate_memory_load, find_variable, qc_summary

    # check memory before loading
    estimate_memory_load("/path/to/MOORING_DATA", "TEMP")

    # search catalogue for oxygen-related variables
    find_variable("oxygen", catalogue_csv="unique_variables.csv")

    # QC breakdown for all files
    qc_summary("/path/to/MOORING_DATA", "TEMP")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nsval.utils import find_coord


# =============================================================================
# 1. ESTIMATE MEMORY LOAD
# =============================================================================

def estimate_memory_load(
    folder      : str | Path,
    variable    : str,
    pattern     : str = "*.nc",
    dtype_bytes : int = 4,          # float32 = 4, float64 = 8
    warn_gb     : float = 1.0,      # print warning above this threshold
    verbose     : bool = True,
) -> pd.DataFrame:
    """
    Scan a folder of NetCDF files and estimate how much memory loading
    *variable* would require — without actually loading any data.

    Parameters
    ----------
    folder       : directory containing .nc files
    variable     : variable name to estimate, e.g. 'TEMP', 'temp'
    pattern      : glob pattern (default '*.nc')
    dtype_bytes  : bytes per element (4 for float32, 8 for float64)
    warn_gb      : print a warning for files above this size in GB
    verbose      : print per-file summary

    Returns
    -------
    pandas.DataFrame with columns:
        file, shape, n_elements, size_MB, size_GB, dtype, warning
    """
    folder = Path(folder)
    files  = sorted(folder.glob(pattern))

    if not files:
        print(f"[estimate_memory_load] No files found in {folder}")
        return pd.DataFrame()

    rows = []
    total_elements = 0

    print(f"\n{'═'*65}")
    print(f"  estimate_memory_load  |  variable='{variable}'")
    print(f"  Folder : {folder}")
    print(f"  Files  : {len(files)}")
    print(f"{'═'*65}")

    for path in files:
        try:
            ds = xr.open_dataset(path, decode_times=False)
        except Exception as exc:
            rows.append({"file": path.name, "error": str(exc)})
            ds = None

        if ds is None:
            continue

        if variable not in ds.data_vars:
            ds.close()
            continue

        da      = ds[variable]
        shape   = tuple(da.shape)
        n_elem  = int(np.prod(shape))
        size_mb = round(n_elem * dtype_bytes / 1024**2, 2)
        size_gb = round(size_mb / 1024, 4)
        warn    = size_gb >= warn_gb

        total_elements += n_elem

        row = {
            "file"      : path.name,
            "shape"     : str(shape),
            "dims"      : " × ".join(da.dims),
            "n_elements": n_elem,
            "size_MB"   : size_mb,
            "size_GB"   : size_gb,
            "dtype"     : str(da.dtype),
            "warning"   : "⚠ LARGE" if warn else "",
        }
        rows.append(row)

        if verbose:
            flag = "  ⚠ LARGE" if warn else ""
            print(f"  {path.name:<40}  {str(shape):<25}  "
                  f"{size_mb:>8.1f} MB{flag}")

        ds.close()

    if not rows:
        print(f"  '{variable}' not found in any file.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    total_mb = total_elements * dtype_bytes / 1024**2
    total_gb = total_mb / 1024

    print(f"\n  {'─'*63}")
    print(f"  Files containing '{variable}' : "
          f"{len(df[df.get('error', pd.Series()).isna() if 'error' in df else df.index])}")
    print(f"  Total if loaded at once      : "
          f"{total_mb:,.1f} MB  ({total_gb:.3f} GB)")
    if total_gb >= warn_gb:
        print(f"  ⚠  Loading all at once may exhaust memory.")
        print(f"     Consider loading one file at a time or using "
              f"vertical_mode / time slicing.")
    print(f"{'═'*65}\n")

    return df


# =============================================================================
# 2. FIND VARIABLE
# =============================================================================

def find_variable(
    keyword         : str,
    catalogue_csv   : str | Path | None = None,
    folder          : str | Path | None = None,
    pattern         : str = "*.nc",
    case_sensitive  : bool = False,
) -> pd.DataFrame:
    """
    Search for variables matching *keyword* in a variable catalogue CSV
    or directly in a folder of NetCDF files.

    Searches across: variable name, long_name, standard_name, units,
    description.

    Parameters
    ----------
    keyword        : search term, e.g. 'temp', 'oxygen', 'chlorophyll'
    catalogue_csv  : CSV from nsval.inventory.build_variable_catalogue()
                     If None, *folder* must be provided.
    folder         : scan this folder directly if no catalogue CSV given
    pattern        : glob pattern when scanning folder
    case_sensitive : default False

    Returns
    -------
    pandas.DataFrame of matching rows, sorted by variable name.
    Prints a readable summary to the terminal.
    """
    # ── build or load catalogue ───────────────────────────────────────────────
    if catalogue_csv is not None:
        cat = pd.read_csv(catalogue_csv)
    elif folder is not None:
        from nsval.inventory import build_variable_catalogue, build_inventory
        folder = Path(folder)
        inv_tmp = folder / "_tmp_inventory.csv"
        cat_tmp = folder / "_tmp_catalogue.csv"
        build_inventory(folder, inv_tmp, pattern)
        cat = build_variable_catalogue(inv_tmp, cat_tmp)
        inv_tmp.unlink(missing_ok=True)
        cat_tmp.unlink(missing_ok=True)
    else:
        raise ValueError("Provide either catalogue_csv or folder.")

    # ── search ────────────────────────────────────────────────────────────────
    search_cols = ["variable", "long_name", "standard_name",
                   "units", "description"]
    search_cols = [c for c in search_cols if c in cat.columns]

    kw = keyword if case_sensitive else keyword.lower()

    def _matches(row):
        for col in search_cols:
            val = str(row.get(col, ""))
            if not case_sensitive:
                val = val.lower()
            if kw in val:
                return True
        return False

    mask    = cat.apply(_matches, axis=1)
    results = cat[mask].reset_index(drop=True)

    # ── print ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  find_variable('{keyword}')  —  {len(results)} match(es)")
    print(f"{'═'*65}")

    if len(results) == 0:
        print(f"  No variables matching '{keyword}' found.")
        print(f"  Try a broader term or check spelling.")
    else:
        for _, row in results.iterrows():
            print(f"\n  Variable     : {row.get('variable','–')}")
            print(f"  Long name    : {row.get('long_name','–')}")
            print(f"  Standard name: {row.get('standard_name','–')}")
            print(f"  Units        : {row.get('units','–')}")
            print(f"  Dims         : {row.get('dims','–')}")
            print(f"  Shape        : {row.get('shape','–')}")
            print(f"  Source file  : {row.get('source_file','–')}")

    print(f"\n{'═'*65}\n")
    return results


# =============================================================================
# 3. QC SUMMARY
# =============================================================================

# Standard CMEMS QC flag meanings
_QC_MEANINGS = {
    0: "No QC performed",
    1: "Good data",
    2: "Probably good data",
    3: "Probably bad data",
    4: "Bad data",
    7: "Nominal value",
    8: "Interpolated value",
    9: "Missing value",
}


def qc_summary(
    folder      : str | Path,
    variable    : str,
    pattern     : str = "*.nc",
    good_flags  : set = frozenset({1, 2}),
    verbose     : bool = True,
) -> pd.DataFrame:
    """
    Scan a folder of CMEMS NetCDF files and summarise QC flag
    distributions for *variable*.

    For each file that contains {variable}_QC (or similar), counts
    the number of values per flag and computes % good data.

    Parameters
    ----------
    folder     : directory containing .nc files
    variable   : variable name, e.g. 'TEMP'
    pattern    : glob pattern
    good_flags : flag values considered good (default {1, 2})
    verbose    : print per-file breakdown

    Returns
    -------
    pandas.DataFrame with one row per file:
        file, n_total, n_good, pct_good, n_bad, pct_bad,
        flag_0 … flag_9 (counts per flag value)
    """
    folder = Path(folder)
    files  = sorted(folder.glob(pattern))

    # try common QC naming conventions
    def _get_qc_name(ds, var):
        for cand in [f"{var}_QC", f"{var}_qc", f"QC_{var}", f"qc_{var}"]:
            if cand in ds.data_vars or cand in ds.coords:
                return cand
        return None

    rows = []
    all_flags = set()

    print(f"\n{'═'*65}")
    print(f"  qc_summary  |  variable='{variable}'")
    print(f"  Good flags : {sorted(good_flags)}")
    print(f"{'═'*65}")

    for path in files:
        try:
            ds = xr.open_dataset(path, decode_times=False)
        except Exception as exc:
            print(f"  Could not open {path.name}: {exc}")
            continue

        if variable not in ds.data_vars:
            ds.close()
            continue

        qc_name = _get_qc_name(ds, variable)

        if qc_name is None:
            rows.append({
                "file"    : path.name,
                "qc_var"  : "not found",
                "n_total" : int(ds[variable].size),
                "n_good"  : np.nan,
                "pct_good": np.nan,
            })
            ds.close()
            continue

        qc_vals = np.asarray(ds[qc_name]).ravel()
        qc_vals = qc_vals[~np.isnan(qc_vals.astype(float))]
        qc_int  = qc_vals.astype(int)

        n_total = len(qc_int)
        n_good  = int(np.isin(qc_int, list(good_flags)).sum())
        n_bad   = n_total - n_good
        pct_good= round(100 * n_good / n_total, 1) if n_total > 0 else np.nan
        pct_bad = round(100 * n_bad  / n_total, 1) if n_total > 0 else np.nan

        # per-flag counts
        flag_counts = {}
        for flag in range(10):
            cnt = int((qc_int == flag).sum())
            if cnt > 0:
                all_flags.add(flag)
            flag_counts[f"flag_{flag}"] = cnt

        row = {
            "file"    : path.name,
            "qc_var"  : qc_name,
            "n_total" : n_total,
            "n_good"  : n_good,
            "pct_good": pct_good,
            "n_bad"   : n_bad,
            "pct_bad" : pct_bad,
            **flag_counts,
        }
        rows.append(row)

        if verbose:
            bar_good = "█" * int(pct_good / 5) if not np.isnan(pct_good) else ""
            bar_bad  = "░" * (20 - len(bar_good))
            print(f"  {path.name:<38}  "
                  f"good={pct_good:>5.1f}%  "
                  f"|{bar_good}{bar_bad}|  "
                  f"n={n_total:>7}")

        ds.close()

    if not rows:
        print(f"  '{variable}' not found in any file.")
        print(f"{'═'*65}\n")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── aggregate summary ─────────────────────────────────────────────────────
    valid = df[df["pct_good"].notna()]

    print(f"\n  {'─'*63}")
    print(f"  FILES WITH '{variable}'     : {len(df)}")
    print(f"  Files with QC variable    : {len(valid)}")

    if len(valid) > 0:
        total_n    = int(valid["n_total"].sum())
        total_good = int(valid["n_good"].sum())
        total_bad  = int(valid["n_bad"].sum())
        pct_all    = round(100 * total_good / total_n, 1) if total_n > 0 else 0

        print(f"  Total observations        : {total_n:,}")
        print(f"  Total good (flags {sorted(good_flags)}) : "
              f"{total_good:,}  ({pct_all}%)")
        print(f"  Total bad / other         : {total_bad:,}  "
              f"({100-pct_all:.1f}%)")

        print(f"\n  {'─'*63}")
        print(f"  FLAG BREAKDOWN (all files combined)")
        print(f"  {'─'*63}")
        for flag in sorted(all_flags):
            col  = f"flag_{flag}"
            cnt  = int(valid[col].sum()) if col in valid.columns else 0
            pct  = round(100 * cnt / total_n, 1) if total_n > 0 else 0
            meaning = _QC_MEANINGS.get(flag, "Unknown")
            good_marker = " ✓" if flag in good_flags else ""
            print(f"  Flag {flag}  {meaning:<30}  "
                  f"{cnt:>8,}  ({pct:>5.1f}%){good_marker}")

    print(f"{'═'*65}\n")
    return df


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(
        description="nsval dataset inspection utilities."
    )
    sub = p.add_subparsers(dest="command")

    # memory
    pm = sub.add_parser("memory", help="Estimate memory load")
    pm.add_argument("--folder",   required=True)
    pm.add_argument("--variable", required=True)
    pm.add_argument("--warn-gb",  type=float, default=1.0)

    # find
    pf = sub.add_parser("find", help="Find variable by keyword")
    pf.add_argument("keyword")
    pf.add_argument("--catalogue", default=None)
    pf.add_argument("--folder",    default=None)

    # qc
    pq = sub.add_parser("qc", help="QC flag summary")
    pq.add_argument("--folder",   required=True)
    pq.add_argument("--variable", required=True)

    args = p.parse_args()

    if args.command == "memory":
        estimate_memory_load(args.folder, args.variable,
                             warn_gb=args.warn_gb)
    elif args.command == "find":
        find_variable(args.keyword,
                      catalogue_csv=args.catalogue,
                      folder=args.folder)
    elif args.command == "qc":
        qc_summary(args.folder, args.variable)
    else:
        p.print_help()


if __name__ == "__main__":
    _cli()
