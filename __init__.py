"""
nsval — North Sea Validation Toolkit
─────────────────────────────────────
Point-based extraction, climatological analysis, and model–observation
validation for CMEMS in-situ archives and ROMS ocean model output.

Modules
-------
nsval.inventory          — NetCDF archive inventory and variable catalogue
nsval.intake.cmems       — Extract point timeseries from CMEMS in-situ files
nsval.intake.roms        — Extract point timeseries from ROMS AVG files
nsval.analyse.timeseries — Timeseries diagnostics and climatology plots
nsval.validate.metrics   — Core validation metric functions (shared)
nsval.validate.daily     — Daily/nearest-match model vs obs validation
nsval.validate.monthly   — Monthly-mean model vs obs validation
nsval.utils              — Shared helpers (haversine, time decoding, I/O)
"""

__version__ = "0.1.0"
__author__  = "E. Ivanov"
