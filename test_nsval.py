"""
test_nsval.py
─────────────────────────────────────────────────────────────────────
Full integration test for the nsval package.

Target area: German Bight / Southern North Sea
Coordinates: 55.0°N, 8.0°E  (higher mooring density than Oyster Grounds)
Radius:       150 km         (wide enough to catch multiple stations)

Tests every public function in order:
    0.  Import and version check
    1.  scan_dataset / build_inventory
    2.  build_variable_catalogue
    3.  find_variable
    4.  estimate_memory_load
    5.  qc_summary
    6.  safe_point_timeseries (CMEMS surface extraction)
    7.  intake.roms.extract_point
    8.  analyse.timeseries.analyse
    9.  validate.daily.validate
    10. validate.monthly.validate (compare_timeseries)
    11. validate.vertical.analyse_vertical
    12. validate.vertical_model.validate_vertical

Usage:
    python test_nsval.py
    python test_nsval.py --stop-on-fail   # stop at first failure
    python test_nsval.py --skip-roms      # skip ROMS-dependent tests
    python test_nsval.py --skip-vertical  # skip vertical tests
"""

from __future__ import annotations

import argparse
import sys
import traceback
import time
from pathlib import Path

# =============================================================================
# SETTINGS — adjust to your paths
# =============================================================================

MOORING_FOLDER = Path("/home/ulg/mast/eivanov/Validation/MOORING_DATA/")

ROMS_SIMULATIONS = {
    "sim_2006": {
        "folder" : Path("/scratch/ulg/mast/eivanov/Output/CE2COAST_2006"),
        "pattern": "Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
    },
    "sim_2009": {
        "folder" : Path("/scratch/ulg/mast/eivanov/Output/CE2COAST_2009"),
        "pattern": "Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
    },
}

# Target: German Bight — denser mooring coverage than Oyster Grounds
LAT0      = 55.0
LON0      = 8.0
RADIUS_KM = 150.0

VARIABLE     = "TEMP"
ROMS_VAR     = "temp"
OUT_DIR      = Path("examples/outputs")

# =============================================================================
# TEST RUNNER
# =============================================================================

class TestRunner:
    def __init__(self, stop_on_fail=False):
        self.stop_on_fail = stop_on_fail
        self.results      = []
        self.t_start      = time.time()

    def run(self, name, fn):
        print(f"\n{'─'*65}")
        print(f"  TEST: {name}")
        print(f"{'─'*65}")
        t0 = time.time()
        try:
            result = fn()
            elapsed = time.time() - t0
            print(f"  ✓  PASSED  ({elapsed:.1f}s)")
            self.results.append((name, "PASS", elapsed, None))
            return result
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ✗  FAILED  ({elapsed:.1f}s)")
            print(f"     {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self.results.append((name, "FAIL", elapsed, str(exc)))
            if self.stop_on_fail:
                self._summary()
                sys.exit(1)
            return None

    def _summary(self):
        total   = time.time() - self.t_start
        passed  = sum(1 for _, s, _, _ in self.results if s == "PASS")
        failed  = sum(1 for _, s, _, _ in self.results if s == "FAIL")
        skipped = sum(1 for _, s, _, _ in self.results if s == "SKIP")

        print(f"\n{'═'*65}")
        print(f"  TEST SUMMARY")
        print(f"{'═'*65}")
        for name, status, elapsed, err in self.results:
            icon = "✓" if status == "PASS" else ("–" if status == "SKIP" else "✗")
            print(f"  {icon}  {name:<45}  {elapsed:>6.1f}s"
                  + (f"  → {err[:40]}" if err else ""))
        print(f"{'─'*65}")
        print(f"  Passed : {passed}  |  Failed : {failed}  |  "
              f"Skipped : {skipped}  |  Total : {total:.1f}s")
        print(f"{'═'*65}\n")

    def skip(self, name, reason=""):
        print(f"\n  –  SKIPPED: {name}"
              + (f"  ({reason})" if reason else ""))
        self.results.append((name, "SKIP", 0.0, reason))

    def summary(self):
        self._summary()
        failed = sum(1 for _, s, _, _ in self.results if s == "FAIL")
        return failed == 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="nsval integration tests")
    parser.add_argument("--stop-on-fail",   action="store_true")
    parser.add_argument("--skip-roms",      action="store_true")
    parser.add_argument("--skip-vertical",  action="store_true")
    args = parser.parse_args()

    # create output directory
    OUT_DIR.mkdir(exist_ok=True)

    runner = TestRunner(stop_on_fail=args.stop_on_fail)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 0 — import and version
    # ─────────────────────────────────────────────────────────────────────────
    def test_import():
        import nsval
        assert hasattr(nsval, "__version__"), "no __version__"
        assert hasattr(nsval, "scan_dataset"), "no scan_dataset"
        assert hasattr(nsval, "estimate_memory_load"), "no estimate_memory_load"
        assert hasattr(nsval, "find_variable"), "no find_variable"
        assert hasattr(nsval, "qc_summary"), "no qc_summary"
        assert hasattr(nsval, "safe_point_timeseries"), "no safe_point_timeseries"
        assert hasattr(nsval, "compare_timeseries"), "no compare_timeseries"
        print(f"  nsval version: {nsval.__version__}")
        return nsval

    nsval = runner.run("0. Import and version check", test_import)
    if nsval is None:
        print("  Cannot continue without nsval import.")
        runner.summary()
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1 — scan_dataset / build_inventory
    # ─────────────────────────────────────────────────────────────────────────
    inv_csv = OUT_DIR / "inventory.csv"

    def test_inventory():
        df = nsval.scan_dataset(
            folder  = MOORING_FOLDER,
            out_csv = inv_csv,
        )
        assert len(df) > 0,            "inventory is empty"
        assert inv_csv.exists(),       "inventory CSV not written"
        assert "file" in df.columns,   "no 'file' column"
        print(f"  Files found: {len(df)}")
        return df

    inv_df = runner.run("1. scan_dataset / build_inventory", test_inventory)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2 — build_variable_catalogue
    # ─────────────────────────────────────────────────────────────────────────
    cat_csv = OUT_DIR / "variables.csv"

    def test_catalogue():
        df = nsval.build_variable_catalogue(
            inventory_csv = inv_csv,
            out_csv       = cat_csv,
        )
        assert len(df) > 0,              "catalogue is empty"
        assert cat_csv.exists(),         "catalogue CSV not written"
        assert "variable" in df.columns, "no 'variable' column"
        print(f"  Unique variables: {len(df)}")
        return df

    cat_df = runner.run("2. build_variable_catalogue", test_catalogue)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3 — find_variable
    # ─────────────────────────────────────────────────────────────────────────
    def test_find_variable():
        results = nsval.find_variable("temp", catalogue_csv=cat_csv)
        print(f"  Matches for 'temp': {len(results)}")
        results2 = nsval.find_variable("salinity", catalogue_csv=cat_csv)
        print(f"  Matches for 'salinity': {len(results2)}")
        # not raising is enough — no matches is valid
        return results

    runner.run("3. find_variable", test_find_variable)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4 — estimate_memory_load
    # ─────────────────────────────────────────────────────────────────────────
    def test_memory():
        df = nsval.estimate_memory_load(
            folder   = MOORING_FOLDER,
            variable = VARIABLE,
            warn_gb  = 0.5,
        )
        assert df is not None, "returned None"
        print(f"  Files with {VARIABLE}: {len(df)}")
        return df

    runner.run("4. estimate_memory_load", test_memory)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5 — qc_summary
    # ─────────────────────────────────────────────────────────────────────────
    def test_qc():
        df = nsval.qc_summary(
            folder   = MOORING_FOLDER,
            variable = VARIABLE,
        )
        assert df is not None, "returned None"
        print(f"  Files with QC: {len(df)}")
        return df

    runner.run("5. qc_summary", test_qc)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 6 — safe_point_timeseries (CMEMS surface extraction)
    # ─────────────────────────────────────────────────────────────────────────
    obs_csv = OUT_DIR / f"TEMP_scoop_{LAT0}_{LON0}.csv"

    def test_scoop():
        df = nsval.safe_point_timeseries(
            folder        = MOORING_FOLDER,
            variable      = VARIABLE,
            lat0          = LAT0,
            lon0          = LON0,
            radius_km     = RADIUS_KM,
            vertical_mode = "surface",
            out_csv       = obs_csv,
        )
        assert obs_csv.exists(),    "output CSV not written"
        assert len(df) > 0,         "no data extracted — try wider radius"
        assert "time" in df.columns or "TIME" in df.columns, "no time column"
        print(f"  Rows extracted: {len(df)}")
        print(f"  Sources: {df['source_file'].nunique()} files")
        return df

    obs_df = runner.run("6. safe_point_timeseries", test_scoop)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 7 — intake.roms.extract_point
    # ─────────────────────────────────────────────────────────────────────────
    roms_csv = OUT_DIR / f"roms_temp_{LAT0}_{LON0}.csv"

    if args.skip_roms:
        runner.skip("7. intake.roms.extract_point", "skip-roms flag set")
        roms_df = None
    else:
        def test_roms():
            from nsval.intake.roms import extract_point
            df = extract_point(
                simulations = ROMS_SIMULATIONS,
                variable    = ROMS_VAR,
                lat0        = LAT0,
                lon0        = LON0,
                s_level     = -1,
                out_csv     = roms_csv,
            )
            assert roms_csv.exists(), "ROMS CSV not written"
            assert len(df) > 0,       "no ROMS data extracted"
            print(f"  ROMS rows: {len(df)}")
            return df

        roms_df = runner.run("7. intake.roms.extract_point", test_roms)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 8 — analyse.timeseries.analyse (obs)
    # ─────────────────────────────────────────────────────────────────────────
    def test_analyse_obs():
        if obs_df is None or len(obs_df) == 0:
            raise RuntimeError("No obs data from test 6 — cannot run analyse")
        from nsval.analyse.timeseries import analyse
        result = analyse(
            csv_file     = obs_csv,
            variable     = VARIABLE,
            qc_col       = f"{VARIABLE}_QC",
            flag         = "Test_Archive",
            save_figures = True,
            show_figures = False,
        )
        assert "data" in result,    "no data key in result"
        assert "figures" in result, "no figures key in result"
        assert len(result["figures"]) == 3, "expected 3 figures"
        print(f"  Figures produced: {len(result['figures'])}")
        return result

    runner.run("8. analyse.timeseries (obs)", test_analyse_obs)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 8b — analyse.timeseries.analyse (model)
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_roms and roms_df is not None:
        def test_analyse_model():
            from nsval.analyse.timeseries import analyse
            result = analyse(
                csv_file     = roms_csv,
                variable     = "temp_celsius",
                qc_col       = None,
                flag         = "Test_Model",
                save_figures = True,
                show_figures = False,
            )
            assert len(result["figures"]) == 3, "expected 3 figures"
            return result

        runner.run("8b. analyse.timeseries (model)", test_analyse_model)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 9 — validate.daily
    # ─────────────────────────────────────────────────────────────────────────
    if args.skip_roms or roms_df is None or obs_df is None:
        runner.skip("9. validate.daily", "requires both obs and ROMS data")
    else:
        def test_validate_daily():
            from nsval.validate.daily import validate
            result = validate(
                obs_csv        = obs_csv,
                model_csv      = roms_csv,
                obs_var        = VARIABLE,
                model_var      = "temp_celsius",
                obs_qc_col     = f"{VARIABLE}_QC",
                match_tol_days = 1.0,
                out_metrics    = OUT_DIR / "metrics_daily.csv",
                out_figure     = OUT_DIR / "validation_daily.png",
                show_figure    = False,
            )
            assert "metrics" in result, "no metrics in result"
            m = result["metrics"]
            print(f"  Pairs: {m['n_pairs']}  bias={m['bias']:+.3f}°C  "
                  f"RMSE={m['rmse']:.3f}°C  r={m['r']:.3f}")
            return result

        runner.run("9. validate.daily", test_validate_daily)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 10 — validate.monthly / compare_timeseries
    # ─────────────────────────────────────────────────────────────────────────
    if args.skip_roms or roms_df is None or obs_df is None:
        runner.skip("10. compare_timeseries (monthly)", "requires both datasets")
    else:
        def test_validate_monthly():
            result = nsval.compare_timeseries(
                obs_csv   = obs_csv,
                model_csv = roms_csv,
                obs_var   = VARIABLE,
                model_var = "temp_celsius",
                out_metrics  = OUT_DIR / "metrics_monthly.csv",
                out_figure   = OUT_DIR / "validation_monthly.png",
                show_figure  = False,
            )
            assert "metrics" in result, "no metrics in result"
            m = result["metrics"]
            print(f"  Monthly pairs: {m['n_pairs']}  "
                  f"bias={m['bias']:+.3f}°C  RMSE={m['rmse']:.3f}°C  "
                  f"r={m['r']:.3f}  NSE={m['nse']:.3f}")
            return result

        runner.run("10. compare_timeseries (monthly)", test_validate_monthly)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 11 — validate.vertical.analyse_vertical
    # ─────────────────────────────────────────────────────────────────────────
    vert_csv = OUT_DIR / f"vertical_TEMP_{LAT0}_{LON0}.csv"

    if args.skip_vertical:
        runner.skip("11. validate.vertical.analyse_vertical",
                    "skip-vertical flag set")
        vert_result = None
    else:
        def test_vertical_obs():
            from nsval.validate.vertical import analyse_vertical
            result = analyse_vertical(
                folder       = MOORING_FOLDER,
                variable     = VARIABLE,
                lat0         = LAT0,
                lon0         = LON0,
                radius_km    = RADIUS_KM,
                min_years    = 0.5,
                depth_max    = 200.0,
                out_csv      = vert_csv,
                out_figure   = OUT_DIR / "vertical_obs.png",
                show_figures = False,
            )
            assert "data" in result,     "no data key"
            assert "stations" in result, "no stations key"
            if len(result["stations"]) == 0:
                print("  Warning: no qualifying stations found — "
                      "try wider radius or lower min_years")
            else:
                print(f"  Stations found: {len(result['stations'])}")
                print(f"  Total profile rows: {len(result['data'])}")
            return result

        vert_result = runner.run("11. validate.vertical.analyse_vertical",
                                 test_vertical_obs)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 12 — validate.vertical_model.validate_vertical
    # ─────────────────────────────────────────────────────────────────────────
    if args.skip_roms or args.skip_vertical:
        runner.skip("12. validate.vertical_model",
                    "requires both ROMS and vertical obs data")
    elif vert_result is None or len(vert_result.get("data", [])) == 0:
        runner.skip("12. validate.vertical_model",
                    "no vertical obs data from test 11")
    elif not vert_csv.exists():
        runner.skip("12. validate.vertical_model",
                    "vertical obs CSV not produced in test 11")
    else:
        def test_vertical_model():
            from nsval.validate.vertical_model import validate_vertical
            result = validate_vertical(
                obs_csv       = vert_csv,
                roms_folder   = ROMS_SIMULATIONS["sim_2006"]["folder"],
                roms_pattern  = ROMS_SIMULATIONS["sim_2006"]["pattern"],
                roms_variable = ROMS_VAR,
                obs_variable  = VARIABLE,
                space_km      = 50.0,
                time_days     = 3.0,
                depth_max     = 150.0,
                depth_bins    = [0, 10, 25, 50, 100, 150],
                out_csv       = OUT_DIR / "vertical_validation.csv",
                out_figure    = OUT_DIR / "vertical_validation.png",
                show_figures  = False,
            )
            assert "paired" in result,        "no paired data"
            assert "depth_metrics" in result, "no depth metrics"
            if len(result["paired"]) == 0:
                print("  Warning: no collocated pairs — "
                      "try wider space_km or time_days")
            else:
                print(f"  Collocated pairs: {len(result['paired'])}")
                dm = result["depth_metrics"]
                if len(dm) > 0:
                    print(f"  Depth bins with metrics: {len(dm)}")
            return result

        runner.run("12. validate.vertical_model", test_vertical_model)

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    all_passed = runner.summary()

    print(f"  Output files written to: {OUT_DIR.resolve()}")
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
