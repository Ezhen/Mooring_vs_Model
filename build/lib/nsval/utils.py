"""
nsval.utils
───────────
Shared helper functions used across intake, analyse, and validate modules.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# GEOGRAPHY
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat, lon, lat0, lon0):
    """
    Great-circle distance in kilometres between (lat, lon) and (lat0, lon0).
    All inputs in decimal degrees. Inputs may be numpy arrays.
    """
    r = 6371.0
    lat, lon   = np.deg2rad(lat),  np.deg2rad(lon)
    lat0, lon0 = np.deg2rad(lat0), np.deg2rad(lon0)
    a = (
        np.sin((lat - lat0) / 2) ** 2
        + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# TIME
# ─────────────────────────────────────────────────────────────────────────────

def decode_roms_time(time_var):
    """
    Decode a ROMS ocean_time variable to a pandas DatetimeIndex.

    ROMS stores time as seconds since 1858-11-17 (Modified Julian Day epoch).
    xarray decodes this automatically when possible; this function handles
    the cases where it doesn't.

    Parameters
    ----------
    time_var : xarray.DataArray
        The ocean_time variable from an open xarray Dataset.

    Returns
    -------
    pandas.DatetimeIndex
    """
    if np.issubdtype(time_var.dtype, np.datetime64):
        return pd.DatetimeIndex(time_var.values)

    units = time_var.attrs.get("units", "")

    if "1858-11-17" in units:
        epoch = pd.Timestamp("1858-11-17")
        return pd.DatetimeIndex(
            [epoch + pd.Timedelta(seconds=float(s)) for s in time_var.values]
        )

    try:
        import cftime
        cal   = time_var.attrs.get("calendar", "standard")
        unit  = units or "seconds since 1858-11-17 00:00:00"
        dates = cftime.num2date(time_var.values, unit, calendar=cal)
        return pd.DatetimeIndex([pd.Timestamp(str(d)) for d in dates])
    except Exception as exc:
        raise RuntimeError(f"Could not decode ocean_time: {exc}") from exc


def add_ymd(df, time_col):
    """
    Add integer year / month / day columns derived from a datetime column.
    Modifies df in place and returns it.
    """
    t = pd.to_datetime(df[time_col])
    df["year"]  = t.dt.year
    df["month"] = t.dt.month
    df["day"]   = t.dt.day
    return df


def build_time(df):
    """
    Return a pandas DatetimeIndex from a DataFrame that contains either
    a 'time' column or separate 'year', 'month', 'day' columns.
    """
    if "time" in df.columns:
        return pd.to_datetime(df["time"])
    elif {"year", "month", "day"}.issubset(df.columns):
        return pd.to_datetime(df[["year", "month", "day"]])
    else:
        raise ValueError("DataFrame has no usable time column.")


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_timeseries_csv(path, var_col, qc_col=None, qc_good=None,
                        sim_filter=None):
    """
    Load a timeseries CSV produced by nsval.intake and return a clean
    two-column DataFrame with columns ['time', 'value'].

    Parameters
    ----------
    path        : str or Path
    var_col     : str   — column holding the variable of interest
    qc_col      : str   — QC flag column name (None to skip)
    qc_good     : set   — accepted flag values, e.g. {1, 2}
    sim_filter  : str   — if a 'simulation' column exists, keep only this label
    """
    df = pd.read_csv(path)

    df["time"] = build_time(df)

    if sim_filter and "simulation" in df.columns:
        df = df[df["simulation"] == sim_filter]

    if qc_col and qc_col in df.columns and qc_good is not None:
        df = df[df[qc_col].isin(qc_good)]

    df = df.dropna(subset=[var_col])
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", var_col]].rename(columns={var_col: "value"})


# ─────────────────────────────────────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────────────────────────────────────

def find_coord(ds, candidates):
    """Return the first name in *candidates* found in ds.coords or ds.data_vars."""
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    return None


def detect_vertical_dim(da):
    """Return the name of the vertical dimension in a DataArray, or None."""
    candidates = {"depth", "deph", "z", "lev", "level", "s_rho", "s_w"}
    for dim in da.dims:
        if dim.lower() in candidates:
            return dim
    return None
