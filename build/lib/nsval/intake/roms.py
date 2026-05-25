"""
nsval.intake.roms
─────────────────
Extract a point timeseries from ROMS AVG NetCDF files.

ROMS grid specifics
───────────────────
- Curvilinear grid: lat_rho(eta_rho, xi_rho), lon_rho(eta_rho, xi_rho).
- 3-D variables: var(ocean_time, s_rho, eta_rho, xi_rho).
- Time: ocean_time in seconds since 1858-11-17 (Modified Julian Day).
- Multiple AVG files per simulation (one per year or chunk).

Typical usage
─────────────
    from nsval.intake.roms import extract_point

    SIMULATIONS = {
        "simulation_2006": {
            "folder" : "/scratch/.../CE2COAST_2006",
            "pattern": "Hindcast_CE2COAST_AVG_*.nc",
        },
    }

    extract_point(
        simulations = SIMULATIONS,
        variable    = "temp",
        lat0        = 54.5,
        lon0        = 4.0,
        s_level     = -1,
        out_csv     = "roms_temp_54.5_4.0.csv",
    )

Or run directly:
    python -m nsval.intake.roms --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nsval.utils import haversine_km, decode_roms_time, add_ymd


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _find_nearest_cell(ds, lat0: float, lon0: float
                       ) -> tuple[int, int, float]:
    """
    Return (eta_idx, xi_idx, distance_km) for the grid cell whose centre
    is closest to (lat0, lon0) on a ROMS curvilinear grid.
    """
    lat2d = np.asarray(ds["lat_rho"])
    lon2d = np.asarray(ds["lon_rho"])
    dist  = haversine_km(lat2d, lon2d, lat0, lon0)
    eta_idx, xi_idx = np.unravel_index(np.argmin(dist), dist.shape)
    return int(eta_idx), int(xi_idx), float(dist[eta_idx, xi_idx])


# ─────────────────────────────────────────────────────────────────────────────
# CORE — single simulation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_simulation(label: str, folder: Path, pattern: str,
                        variable: str, lat0: float, lon0: float,
                        s_level: int) -> pd.DataFrame | None:

    files = sorted(folder.glob(pattern))
    if not files:
        print(f"  [{label}] No files in {folder} matching '{pattern}'")
        return None

    print(f"  [{label}] {len(files)} file(s) found")

    # Grid cell — read from first file only (grid never changes)
    with xr.open_dataset(files[0], decode_times=False) as ds_grid:
        eta_idx, xi_idx, dist_km = _find_nearest_cell(ds_grid, lat0, lon0)
        cell_lat = float(ds_grid["lat_rho"].values[eta_idx, xi_idx])
        cell_lon = float(ds_grid["lon_rho"].values[eta_idx, xi_idx])

    print(f"  [{label}] Nearest cell: eta={eta_idx}, xi={xi_idx}  "
          f"({cell_lat:.4f}°N  {cell_lon:.4f}°E  {dist_km:.2f} km)")

    times_list = []
    vals_list  = []

    for path in files:
        print(f"    {path.name} ...", end=" ", flush=True)
        with xr.open_dataset(path, decode_times=False) as ds:
            if variable not in ds.data_vars:
                print("variable absent, skipping")
                continue

            da = ds[variable]

            # Handle presence or absence of vertical dimension
            if "s_rho" in da.dims:
                da = da.isel(s_rho=s_level, eta_rho=eta_idx, xi_rho=xi_idx)
            else:
                da = da.isel(eta_rho=eta_idx, xi_rho=xi_idx)

            times = decode_roms_time(ds["ocean_time"])
            vals  = da.values.astype(float)
            vals[vals > 1e36] = np.nan   # ROMS fill value

            times_list.append(times)
            vals_list.append(vals)
            print(f"{len(times)} timesteps")

    if not vals_list:
        print(f"  [{label}] No data extracted")
        return None

    all_times = times_list[0].append(times_list[1:]) if len(times_list) > 1 \
                else times_list[0]
    all_vals  = np.concatenate(vals_list)

    df = pd.DataFrame({"time": all_times, variable: all_vals})
    df = df.sort_values("time").reset_index(drop=True)
    df = add_ymd(df, "time")

    df["cell_lat"]   = round(cell_lat, 5)
    df["cell_lon"]   = round(cell_lon, 5)
    df["dist_km"]    = round(dist_km, 3)
    df["eta_idx"]    = eta_idx
    df["xi_idx"]     = xi_idx
    df["s_level"]    = s_level
    df["simulation"] = label

    print(f"  [{label}] {len(df)} timesteps  "
          f"({df['time'].min().date()} – {df['time'].max().date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_point(
    simulations : dict,
    variable    : str   = "temp",
    lat0        : float = 54.5,
    lon0        : float = 4.0,
    s_level     : int   = -1,
    out_csv     : str | Path | None = None,
) -> pd.DataFrame:
    """
    Extract a point timeseries from one or more ROMS simulations.

    Parameters
    ----------
    simulations : dict mapping label → {'folder': Path, 'pattern': str}
    variable    : ROMS variable name (lowercase), e.g. 'temp', 'salt', 'NO3'
    lat0, lon0  : target coordinates (decimal degrees)
    s_level     : vertical s_rho index (-1 = surface, 0 = bottom)
    out_csv     : if given, save long-format result to this path
                  a wide-format CSV is also saved as out_csv.replace('.csv','_wide.csv')

    Returns
    -------
    pandas.DataFrame (long format) with columns:
        time, {variable}, year, month, day,
        cell_lat, cell_lon, dist_km, eta_idx, xi_idx, s_level, simulation
    """
    print(f"\n[roms.extract_point] {variable} | {lat0}°N {lon0}°E | "
          f"s_level={s_level}")

    frames = []
    for label, cfg in simulations.items():
        print(f"\n── {label} ──")
        df = _extract_simulation(
            label, Path(cfg["folder"]), cfg["pattern"],
            variable, lat0, lon0, s_level,
        )
        if df is not None:
            frames.append(df)

    if not frames:
        print("[roms.extract_point] No data extracted.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    if out_csv:
        out_csv = Path(out_csv)
        result.to_csv(out_csv, index=False)
        print(f"\n[roms.extract_point] Saved (long)  → {out_csv}")

        wide = (
            result.pivot_table(
                index="time", columns="simulation", values=variable
            ).reset_index()
        )
        wide.columns.name = None
        out_wide = out_csv.with_name(out_csv.stem + "_wide.csv")
        wide.to_csv(out_wide, index=False)
        print(f"[roms.extract_point] Saved (wide)  → {out_wide}")

    # Summary
    print(f"\n{'─'*55}")
    for lbl, grp in result.groupby("simulation"):
        v = grp[variable].dropna()
        print(f"  {lbl:<22}  n={len(v):>5}  "
              f"mean={v.mean():>6.2f}  std={v.std(ddof=1):>5.2f}  "
              f"min={v.min():>6.2f}  max={v.max():>6.2f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Extract ROMS model point timeseries."
    )
    p.add_argument("--folders",  nargs="+", required=True,
                   help="One or more simulation folders")
    p.add_argument("--labels",   nargs="+", required=True,
                   help="Label for each folder (same order)")
    p.add_argument("--pattern",  default="Hindcast_CE2COAST_AVG_*.nc")
    p.add_argument("--variable", default="temp")
    p.add_argument("--lat",      type=float, required=True)
    p.add_argument("--lon",      type=float, required=True)
    p.add_argument("--slevel",   type=int,   default=-1)
    p.add_argument("--out",      default=None)
    args = p.parse_args()

    sims = {
        lbl: {"folder": fld, "pattern": args.pattern}
        for lbl, fld in zip(args.labels, args.folders)
    }
    out = args.out or f"roms_{args.variable}_{args.lat}_{args.lon}.csv"
    extract_point(sims, args.variable, args.lat, args.lon,
                  args.slevel, out_csv=out)


if __name__ == "__main__":
    _cli()
