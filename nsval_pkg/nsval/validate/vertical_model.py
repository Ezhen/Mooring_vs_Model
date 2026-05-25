"""
nsval.validate.vertical_model
──────────────────────────────
Collocate CMEMS in-situ vertical profiles against ROMS model output
using a space-time window (Approach 5).

For each observed profile timestep, the module:
  1. Finds the nearest ROMS grid cell within SPACE_KM.
  2. Averages all ROMS timesteps within TIME_DAYS of the observation.
  3. Interpolates the ROMS vertical profile onto the observed depth levels.
  4. Computes depth-by-depth validation metrics.
  5. Produces side-by-side time-depth heatmaps + depth-resolved bias figure.

Typical usage
─────────────
    from nsval.validate.vertical_model import validate_vertical

    validate_vertical(
        obs_csv        = "vertical_TEMP_54.5_4.0.csv",
        roms_folder    = "/scratch/.../CE2COAST_2006",
        roms_pattern   = "Hindcast_CE2COAST_AVG_*.nc",
        roms_variable  = "temp",
        obs_variable   = "TEMP",
        space_km       = 20.0,
        time_days      = 1.0,
        out_csv        = "vertical_validation.csv",
        out_figure     = "vertical_validation.png",
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
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

from nsval.utils import haversine_km, decode_roms_time, add_ymd
from nsval.validate.metrics import compute_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# USER-TUNABLE DEFAULTS
# =============================================================================

DEFAULT_SPACE_KM  = 20.0
DEFAULT_TIME_DAYS = 1.0
DEFAULT_DEPTH_MAX = 200.0
DEFAULT_DPI       = 130


# =============================================================================
# STEP 1 — LOAD OBSERVED VERTICAL PROFILES
# =============================================================================

def _load_obs(obs_csv: Path, obs_variable: str) -> pd.DataFrame:
    """
    Load the CSV produced by nsval.validate.vertical.analyse_vertical.
    Expected columns: time, depth, {obs_variable}, file_lat, file_lon,
                      dist_km, source_file.
    """
    df = pd.read_csv(obs_csv)
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=[obs_variable])
    df = df.sort_values(["time", "depth"]).reset_index(drop=True)
    return df


# =============================================================================
# STEP 2 — LOAD ROMS MODEL INTO MEMORY (lazy xarray)
# =============================================================================

def _open_roms(roms_folder: Path, roms_pattern: str) -> xr.Dataset:
    """
    Open all ROMS AVG files as a single lazy xarray Dataset
    using open_mfdataset.
    """
    files = sorted(roms_folder.glob(roms_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{roms_pattern}' in {roms_folder}"
        )
    print(f"  ROMS: {len(files)} files found")

    try:
        ds = xr.open_mfdataset(
            files,
            combine      = "by_coords",
            decode_times = False,
            chunks       = {"ocean_time": 30},   # lazy loading
        )
    except Exception:
        # fallback for files that don't combine cleanly by coords
        ds = xr.open_mfdataset(
            files,
            combine      = "nested",
            concat_dim   = "ocean_time",
            decode_times = False,
            chunks       = {"ocean_time": 30},
        )

    return ds


def _build_roms_times(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Decode ROMS ocean_time to DatetimeIndex."""
    return decode_roms_time(ds["ocean_time"])


def _find_roms_cell(ds: xr.Dataset,
                    lat0: float, lon0: float) -> tuple[int, int, float]:
    """Find nearest ROMS grid cell to (lat0, lon0)."""
    lat2d = np.asarray(ds["lat_rho"])
    lon2d = np.asarray(ds["lon_rho"])
    dist  = haversine_km(lat2d, lon2d, lat0, lon0)
    ei, xi = np.unravel_index(np.argmin(dist), dist.shape)
    return int(ei), int(xi), float(dist[ei, xi])


def _get_roms_depths(ds: xr.Dataset,
                     eta_idx: int, xi_idx: int) -> np.ndarray:
    """
    Return approximate depth values (metres, positive down) for each
    s_rho level at the target cell.

    ROMS stores actual depths via h (bathymetry), Cs_r (stretching),
    and hc (critical depth). If those aren't available, fall back to
    level indices normalised to [0, 1] * max depth.
    """
    try:
        h  = float(ds["h"].values[eta_idx, xi_idx])
        Cs = np.asarray(ds["Cs_r"].values)
        hc = float(ds["hc"].values) if "hc" in ds else 0.0
        # ROMS terrain-following: z = S(x,y,sigma) * h
        # simplified: z_r ≈ hc*Cs + (h-hc)*Cs  for hc << h
        depths = -(hc * Cs + (h - hc) * Cs)   # positive downward
        depths = np.abs(depths)
        return depths
    except Exception:
        n = ds.sizes.get("s_rho", 20)
        return np.linspace(0, 100, n)


# =============================================================================
# STEP 3 — COLLOCATION LOOP
# =============================================================================

def _collocate(obs_df: pd.DataFrame,
               obs_variable: str,
               ds: xr.Dataset,
               roms_variable: str,
               roms_times: pd.DatetimeIndex,
               space_km: float,
               time_days: float,
               depth_max: float) -> pd.DataFrame:
    """
    For each unique (time, station) in obs_df, find matching ROMS profiles
    within the space-time window and interpolate to observed depth levels.

    Returns a DataFrame with columns:
        time, depth, obs, model, file_lat, file_lon, source_file,
        n_roms_profiles_averaged, roms_cell_dist_km
    """
    tol_td = pd.Timedelta(days=time_days)
    rows   = []

    stations = obs_df["source_file"].unique()

    for sfile in stations:
        sdf = obs_df[obs_df["source_file"] == sfile].copy()
        slat = float(sdf["file_lat"].iloc[0])
        slon = float(sdf["file_lon"].iloc[0])

        # ── find nearest ROMS cell ────────────────────────────────────────────
        eta_idx, xi_idx, cell_dist = _find_roms_cell(ds, slat, slon)

        if cell_dist > space_km:
            print(f"    {sfile}: nearest ROMS cell {cell_dist:.1f} km > "
                  f"{space_km} km, skipping")
            continue

        print(f"    {sfile}: ROMS cell dist={cell_dist:.1f} km  "
              f"eta={eta_idx} xi={xi_idx}")

        # ── ROMS depths at this cell ──────────────────────────────────────────
        roms_depths = _get_roms_depths(ds, eta_idx, xi_idx)

        # ── extract ROMS profile timeseries at this cell (lazy → load once) ──
        roms_profiles = ds[roms_variable].isel(
            eta_rho=eta_idx, xi_rho=xi_idx
        ).values.astype(float)                  # (n_times, n_s_rho)
        roms_profiles[roms_profiles > 1e36] = np.nan

        # ── loop over unique observed timesteps ───────────────────────────────
        obs_times = sdf["time"].sort_values().unique()

        for ot in obs_times:
            ot_pd = pd.Timestamp(ot)

            # time window: find ROMS indices within ±time_days
            dt = np.abs(roms_times - ot_pd)
            in_window = dt <= tol_td

            if in_window.sum() == 0:
                continue

            # average ROMS profiles in window
            roms_avg = np.nanmean(roms_profiles[in_window, :], axis=0)

            if np.all(np.isnan(roms_avg)):
                continue

            # observed depths and values at this timestep
            obs_t = sdf[sdf["time"] == ot].sort_values("depth")
            obs_d = obs_t["depth"].values
            obs_v = obs_t[obs_variable].values

            # clip to depth_max
            mask  = obs_d <= depth_max
            obs_d = obs_d[mask]
            obs_v = obs_v[mask]

            if len(obs_d) < 2:
                continue

            # interpolate ROMS profile onto observed depth levels
            valid_r = np.isfinite(roms_avg)
            if valid_r.sum() < 2:
                continue

            try:
                f_roms = interp1d(
                    roms_depths[valid_r], roms_avg[valid_r],
                    kind="linear", bounds_error=False,
                    fill_value=np.nan,
                )
                roms_interp = f_roms(obs_d)
            except Exception:
                continue

            for di in range(len(obs_d)):
                rows.append({
                    "time"                    : ot_pd,
                    "depth"                   : float(obs_d[di]),
                    "obs"                     : float(obs_v[di]),
                    "model"                   : float(roms_interp[di]),
                    "file_lat"                : slat,
                    "file_lon"                : slon,
                    "source_file"             : sfile,
                    "n_roms_in_window"        : int(in_window.sum()),
                    "roms_cell_dist_km"       : round(cell_dist, 2),
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["obs", "model"])
    return df.sort_values(["source_file", "time", "depth"]).reset_index(drop=True)


# =============================================================================
# STEP 4 — DEPTH-BY-DEPTH METRICS
# =============================================================================

def _depth_metrics(df: pd.DataFrame,
                   depth_bins: np.ndarray) -> pd.DataFrame:
    """
    Compute validation metrics in discrete depth bins.

    depth_bins defines bin edges, e.g. [0,10,25,50,100,200].
    Returns one row per bin with all metrics from compute_metrics().
    """
    labels  = [(depth_bins[i] + depth_bins[i+1]) / 2
               for i in range(len(depth_bins) - 1)]
    df      = df.copy()
    df["depth_bin"] = pd.cut(df["depth"], bins=depth_bins, labels=labels)
    df      = df.dropna(subset=["depth_bin"])

    rows = []
    for bin_centre, grp in df.groupby("depth_bin", observed=True):
        O = grp["obs"].values
        M = grp["model"].values
        valid = np.isfinite(O) & np.isfinite(M)
        if valid.sum() < 3:
            continue
        met = compute_metrics(O[valid], M[valid],
                              label=f"{float(bin_centre):.0f}m")
        if met:
            met["depth_centre"] = float(bin_centre)
            rows.append(met)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("depth_centre").reset_index(drop=True)


# =============================================================================
# STEP 5 — FIGURES
# =============================================================================

def _make_figure(paired: pd.DataFrame,
                 depth_met: pd.DataFrame,
                 obs_variable: str,
                 roms_variable: str,
                 space_km: float,
                 time_days: float,
                 depth_max: float,
                 dpi: int,
                 out_figure: str | Path | None):

    stations = paired["source_file"].unique()
    figs     = []

    for sfile in stations:
        sub  = paired[paired["source_file"] == sfile]
        slat = sub["file_lat"].iloc[0]
        slon = sub["file_lon"].iloc[0]
        cdist= sub["roms_cell_dist_km"].iloc[0]

        # ── pivot to 2-D grids ────────────────────────────────────────────────
        def _pivot(col):
            return (sub.pivot_table(index="depth", columns="time",
                                    values=col, aggfunc="mean")
                       .sort_index())

        piv_obs   = _pivot("obs")
        piv_model = _pivot("model")
        piv_bias  = piv_model - piv_obs

        depths = piv_obs.index.values
        times  = piv_obs.columns

        vmin = np.nanpercentile(piv_obs.values, 2)
        vmax = np.nanpercentile(piv_obs.values, 98)
        blim = np.nanpercentile(np.abs(piv_bias.values), 95)

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"Vertical validation  |  {obs_variable} vs ROMS {roms_variable}\n"
            f"{Path(sfile).stem}  |  {slat:.3f}°N {slon:.3f}°E  |  "
            f"ROMS cell {cdist:.1f} km  |  "
            f"window ±{space_km:.0f} km / ±{time_days:.0f} d  |  "
            f"n={len(sub)} pairs",
            fontsize=11, fontweight="bold",
        )

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               hspace=0.38, wspace=0.28)

        def _heatmap(ax, pivot, cmap, vmin, vmax, title, cbar_label):
            pcm = ax.pcolormesh(times, depths, pivot.values,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                shading="nearest")
            ax.set_ylim(depth_max, 0)
            ax.set_ylabel("Depth (m)", fontsize=10)
            ax.set_title(title, fontweight="bold")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            ax.grid(True, alpha=0.2, color="white")
            plt.colorbar(pcm, ax=ax, pad=0.01, fraction=0.025,
                         label=cbar_label)
            return pcm

        # obs heatmap
        _heatmap(fig.add_subplot(gs[0, 0]),
                 piv_obs,  "RdYlBu_r", vmin, vmax,
                 f"Observed {obs_variable}", "°C")

        # model heatmap
        _heatmap(fig.add_subplot(gs[0, 1]),
                 piv_model, "RdYlBu_r", vmin, vmax,
                 f"Model {roms_variable}", "°C")

        # bias heatmap
        _heatmap(fig.add_subplot(gs[0, 2]),
                 piv_bias,  "RdBu_r", -blim, blim,
                 "Bias (model − obs)", "°C")

        # ── depth-by-depth mean profiles ──────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        mean_obs   = sub.groupby("depth")["obs"].mean()
        mean_model = sub.groupby("depth")["model"].mean()
        std_obs    = sub.groupby("depth")["obs"].std(ddof=1)
        std_model  = sub.groupby("depth")["model"].std(ddof=1)

        ax4.fill_betweenx(mean_obs.index,
                          mean_obs - std_obs, mean_obs + std_obs,
                          alpha=0.2, color="#2980b9")
        ax4.fill_betweenx(mean_model.index,
                          mean_model - std_model, mean_model + std_model,
                          alpha=0.2, color="#c0392b")
        ax4.plot(mean_obs.values,   mean_obs.index,
                 "o-", color="#2980b9", lw=2, ms=4, label="Obs")
        ax4.plot(mean_model.values, mean_model.index,
                 "s--", color="#c0392b", lw=2, ms=4, label="Model")
        ax4.set_ylim(depth_max, 0)
        ax4.set_ylabel("Depth (m)", fontsize=10)
        ax4.set_xlabel("°C", fontsize=10)
        ax4.set_title("Mean profiles ± 1 std", fontweight="bold")
        ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

        # ── depth-by-depth bias profile ───────────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        mean_bias = sub.groupby("depth").apply(
            lambda g: float(np.mean(g["model"] - g["obs"]))
        )
        std_bias  = sub.groupby("depth").apply(
            lambda g: float(np.std(g["model"] - g["obs"], ddof=1))
        )
        colors_b = ["#c0392b" if v >= 0 else "#2980b9"
                    for v in mean_bias.values]
        ax5.barh(mean_bias.index, mean_bias.values,
                 color=colors_b, alpha=0.7, height=3)
        ax5.errorbar(mean_bias.values, mean_bias.index,
                     xerr=std_bias.values,
                     fmt="none", color="grey", lw=0.8, capsize=2)
        ax5.axvline(0, color="k", lw=0.8)
        ax5.set_ylim(depth_max, 0)
        ax5.set_xlabel("Bias (°C)", fontsize=10)
        ax5.set_title("Depth-resolved bias ± 1 std", fontweight="bold")
        ax5.grid(True, alpha=0.3)

        # ── depth-bin metrics table ───────────────────────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")

        if len(depth_met) > 0:
            cols_show = ["depth_centre", "n_pairs", "bias", "rmse", "r", "nse"]
            cols_show = [c for c in cols_show if c in depth_met.columns]
            tbl_data  = depth_met[cols_show].copy()
            tbl_data.columns = [c.replace("depth_centre","depth(m)")
                                  .replace("n_pairs","n")
                                  .upper() for c in tbl_data.columns]
            # format floats
            for c in tbl_data.columns:
                if tbl_data[c].dtype == float:
                    tbl_data[c] = tbl_data[c].map("{:.2f}".format)

            tbl = ax6.table(
                cellText  = tbl_data.values,
                colLabels = tbl_data.columns,
                loc       = "center",
                cellLoc   = "center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.3)
            ax6.set_title("Metrics by depth bin", fontweight="bold", pad=12)

        fig.tight_layout()

        if out_figure:
            stem = Path(str(out_figure)).stem
            ext  = Path(str(out_figure)).suffix or ".png"
            fname = f"{stem}_{Path(sfile).stem}{ext}"
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            print(f"  Saved figure → {fname}")

        figs.append(fig)

    return figs


# =============================================================================
# PUBLIC API
# =============================================================================

def validate_vertical(
    obs_csv        : str | Path,
    roms_folder    : str | Path,
    roms_pattern   : str   = "Hindcast_CE2COAST_AVG_*.nc",
    roms_variable  : str   = "temp",
    obs_variable   : str   = "TEMP",
    space_km       : float = DEFAULT_SPACE_KM,
    time_days      : float = DEFAULT_TIME_DAYS,
    depth_max      : float = DEFAULT_DEPTH_MAX,
    depth_bins     : list  = None,
    out_csv        : str | Path | None = None,
    out_figure     : str | Path | None = None,
    figure_dpi     : int   = DEFAULT_DPI,
    show_figures   : bool  = True,
) -> dict:
    """
    Collocate CMEMS vertical profiles against ROMS output and validate.

    Parameters
    ----------
    obs_csv       : CSV from nsval.validate.vertical.analyse_vertical
    roms_folder   : folder containing ROMS AVG NetCDF files
    roms_pattern  : glob pattern for ROMS files
    roms_variable : ROMS variable name (e.g. 'temp')
    obs_variable  : observed variable name (e.g. 'TEMP')
    space_km      : max distance (km) between obs station and ROMS cell
    time_days     : max time offset (days) between obs and model timestep
    depth_max     : maximum depth to include (metres)
    depth_bins    : depth bin edges for metrics, e.g. [0,10,25,50,100,200]
                    defaults to [0,5,10,20,30,50,75,100,150,200]
    out_csv       : save paired profiles to this path
    out_figure    : base filename for figures (one per station)
    figure_dpi    : PNG resolution
    show_figures  : call plt.show()

    Returns
    -------
    dict with keys:
        'paired'      — DataFrame of collocated (obs, model) pairs
        'depth_metrics' — DataFrame of metrics per depth bin
        'figures'     — list of Figure objects
    """
    if depth_bins is None:
        depth_bins = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]

    obs_csv    = Path(obs_csv)
    roms_folder = Path(roms_folder)

    print(f"\n{'═'*65}")
    print(f"  nsval.validate.vertical_model")
    print(f"  Obs     : {obs_csv.name}  ({obs_variable})")
    print(f"  Model   : {roms_folder.name}  ({roms_variable})")
    print(f"  Window  : ±{space_km} km  /  ±{time_days} day(s)")
    print(f"  Depth   : 0 – {depth_max} m")
    print(f"{'═'*65}\n")

    # ── load observations ─────────────────────────────────────────────────────
    obs_df = _load_obs(obs_csv, obs_variable)
    print(f"  Obs loaded: {len(obs_df)} rows  |  "
          f"{obs_df['source_file'].nunique()} station(s)")

    # ── open ROMS ─────────────────────────────────────────────────────────────
    ds          = _open_roms(roms_folder, roms_pattern)
    roms_times  = _build_roms_times(ds)
    print(f"  ROMS loaded: {len(roms_times)} timesteps  "
          f"({roms_times.min().date()} – {roms_times.max().date()})")

    # ── collocation ───────────────────────────────────────────────────────────
    print(f"\n  Collocating...")
    paired = _collocate(obs_df, obs_variable, ds, roms_variable,
                        roms_times, space_km, time_days, depth_max)

    ds.close()

    if len(paired) == 0:
        print("  No collocated pairs found. "
              "Try increasing space_km or time_days.")
        return {"paired": pd.DataFrame(),
                "depth_metrics": pd.DataFrame(), "figures": []}

    print(f"\n  Collocated pairs : {len(paired)}")
    print(f"  Depth range      : "
          f"{paired['depth'].min():.0f} – {paired['depth'].max():.0f} m")

    # ── overall metrics ───────────────────────────────────────────────────────
    O   = paired["obs"].values
    M   = paired["model"].values
    met = compute_metrics(O, M, label="ALL_DEPTHS")

    if met:
        print(f"\n  ── OVERALL (all depths) ──")
        print(f"  n={met['n_pairs']}  "
              f"bias={met['bias']:+.3f}°C  "
              f"RMSE={met['rmse']:.3f}°C  "
              f"r={met['r']:.3f}  "
              f"NSE={met['nse']:.3f}")

    # ── depth-bin metrics ─────────────────────────────────────────────────────
    depth_met = _depth_metrics(paired, np.array(depth_bins))

    if len(depth_met) > 0:
        print(f"\n  ── METRICS BY DEPTH BIN ──")
        print(f"  {'Depth(m)':<10} {'n':>6} {'Bias':>8} "
              f"{'RMSE':>8} {'r':>7} {'NSE':>7}")
        for _, row in depth_met.iterrows():
            print(f"  {row['depth_centre']:<10.0f} "
                  f"{row['n_pairs']:>6}  "
                  f"{row['bias']:>+7.3f}  "
                  f"{row['rmse']:>7.3f}  "
                  f"{row['r']:>6.3f}  "
                  f"{row['nse']:>6.3f}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    if out_csv:
        out_csv = Path(out_csv)
        paired.to_csv(out_csv, index=False)
        print(f"\n  Saved paired data  → {out_csv}")
        met_csv = out_csv.with_name(out_csv.stem + "_depth_metrics.csv")
        depth_met.to_csv(met_csv, index=False, float_format="%.6f")
        print(f"  Saved depth metrics→ {met_csv}")

    # ── figures ───────────────────────────────────────────────────────────────
    figs = _make_figure(paired, depth_met, obs_variable, roms_variable,
                        space_km, time_days, depth_max,
                        figure_dpi, out_figure)

    if show_figures:
        plt.show()

    print(f"\n{'═'*65}\n")

    return {"paired": paired, "depth_metrics": depth_met, "figures": figs}


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(
        description="Collocated vertical model vs obs validation."
    )
    p.add_argument("--obs-csv",      required=True)
    p.add_argument("--roms-folder",  required=True)
    p.add_argument("--roms-pattern", default="Hindcast_CE2COAST_AVG_*.nc")
    p.add_argument("--roms-var",     default="temp")
    p.add_argument("--obs-var",      default="TEMP")
    p.add_argument("--space-km",     type=float, default=20.0)
    p.add_argument("--time-days",    type=float, default=1.0)
    p.add_argument("--depth-max",    type=float, default=200.0)
    p.add_argument("--out-csv",      default=None)
    p.add_argument("--out-fig",      default=None)
    p.add_argument("--no-show",      action="store_true")
    args = p.parse_args()

    validate_vertical(
        obs_csv       = args.obs_csv,
        roms_folder   = args.roms_folder,
        roms_pattern  = args.roms_pattern,
        roms_variable = args.roms_var,
        obs_variable  = args.obs_var,
        space_km      = args.space_km,
        time_days     = args.time_days,
        depth_max     = args.depth_max,
        out_csv       = args.out_csv,
        out_figure    = args.out_fig,
        show_figures  = not args.no_show,
    )


if __name__ == "__main__":
    _cli()
