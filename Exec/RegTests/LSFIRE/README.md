# ERF + LSFIRE (ELMFIRE) fire coupling tests

Small cases for the LSFIRE level-set fire model coupled to ERF. The fire runs on its own
2D grid (`erf.fire.grid_ratio` times finer than ERF) and is coupled to **ERF level 0 only**;
ERF's AMR levels are atmospheric levels and are unrelated to the fire grid.

LSFIRE is a fire engine behind ERF's shared fire coupling (erf-model/ERF#4768): the
engine lives in `FireEngine/` at the top of this repository, registers itself as
`erf.fire.engine = lsfire`, and builds against an unmodified ERF.

## Build

LSFIRE must be built against the same AMReX version as ERF. ERF links the LSFIRE static
library and supplies the AMReX symbols from its own AMReX build.

CMake:

```
cmake -S <ERF> -B <build> ... \
      -DERF_ENABLE_FIRE=ON \
      -DERF_FIRE_ENGINE_DIRS=<this repository>/FireEngine \
      -DELMFIRE_DIR=<LSFIRE source dir>          # provides include/ELMFIRE_API.H
      -DELMFIRE_LIB=<LSFIRE build>/libelmfire_core.a
cmake --build <build> --target erf_exec
```

GNU make, from any directory:

```make
ERF_HOME    = /path/to/ERF
FIRE_HOME   = /path/to/this/repository/FireEngine
USE_FIRE    = TRUE
USE_NETCDF  = TRUE
ELMFIRE_DIR = /path/to/LSFIRE
ELMFIRE_LIB = /path/to/libelmfire_core.a
include $(ERF_HOME)/Exec/Make.ERF
```

Run from a directory containing the inputs file and `fuel_models.csv`:

```
erf_exec inputs_lsfire_constant_wind
```

## Cases

All cases: ERF level 0 is 60 x 30 x 4 cells at 100 m; `erf.fire.grid_ratio = 10` gives a
600 x 300 fire grid at 10 m; GR2 grass, point ignition at (500, 1500) m.

| Inputs | What it checks | Reference result (1 rank, CPU) |
|---|---|---|
| `inputs_lsfire_constant_wind` | `fire.mode = constant_wind`: the ERF wrapper does not change the fire model | Equal to LSFIRE `elmfire_amr` with one fire level on the same 10 m grid and the same fire inputs (phi within 1e-9, identical burned cells at 1 h and 6 h). Burned area 318,900 m^2 at 1 h, 11,092,600 m^2 at 6 h. ~5 min. |
| `inputs_lsfire_atmos_wind` | `fire.mode = atmospheric`: ERF surface wind drives the fire | Uniform 4.89 m/s at the lowest ERF cell centre (50 m above ground) reaches every fire cell; at 1 h burned area 146,700 m^2, head fire downwind. With `fire.atmos_wind_height=50` the result is bitwise identical; with `fire.atmos_wind_height=46.875` it reproduces the value before the k = 0 change (commit 036ad2e9e and earlier: k = 1 wind, fixed 46.875 m), 149,100 m^2. ~70 s. |
| `inputs_lsfire_fire_L1_mask` | Static fire L1 (ratio 5, 10 m) from `refine_mask_two_disconnected.nc` over fire L0 (`erf.fire.grid_ratio = 2`, 50 m); fire updated every 9 ERF steps (= its 4.5 s L0 step) | Equal to LSFIRE `elmfire_amr` `inputs_two_disconnected_50_10m` (same mask): at 1800 s and 3600 s phi, fuel fraction, arrival time, composite mask identical on both fire levels (L1 1319 burned cells at 1 h). Plotted fluxes are interval means and differ only by the averaging window. ~1 min. |
| `inputs_lsfire_heat_feedback` | `erf.fire.coupling_type = lagged`, `erf.fire.heat_profile = bottom_cell`, with `MoistNoCondensation` | Over the run, domain integrals of rho*theta and rho*qv increase by exactly Sum(P dt)/Cp_d and Sum(E dt) (relative difference ~1e-11), with P and E the heat power and vapor rate released by the fire (`fire.verbosity = 2` prints the heat released per fire update; `erf.fire.fire_debug = true` prints the heating ERF injects). ~16 s. |

The fire advances with its own level-0 step (`fire.cfl` x fire dx / 10 m/s), clipped to the end
of each ERF step (or of `fire.subcycle_interval` ERF steps); with an ERF step that is a multiple
of it the fire steps are unchanged, otherwise they are split, which changes results at the
round-off-to-small level (e.g. ERF step 1.0 s vs fire step 0.9 s: IoU 0.9994 after 1 h).

The constant-wind and atmospheric-wind areas differ by design: in constant-wind mode
`fire.constant_wind_u` is taken as the 20-ft wind; in atmospheric mode LSFIRE converts the
ERF wind from its actual height to the 20-ft height with a log law.

## Parameters

| Parameter | Meaning |
|---|---|
| `erf.fire.enable`, `erf.fire.engine` | `1` and `lsfire` run this engine (replaces `erf.fire_type = ELMFIRE`) |
| `fire.mode` | `constant_wind` (default; `fire.constant_wind_u/v`) or `atmospheric` (ERF wind at the lowest cell centre, k = 0). Replaces `erf.fire_met_source`. |
| `fire.atmos_wind_height` | Height above ground [m] assumed for the atmospheric wind. Default: not set, i.e. the actual height of the ERF k = 0 cell centre in each column (z_cc - surface). Set it only to impose a fixed height, e.g. in idealized tests. |
| `erf.fire.coupling_type` | `passive`: the fire heat is not fed back (replaces `erf.fire_heat_feedback = false`); `lagged`: the fire heat flux enters rho*theta, and the vapor flux rho*qv when a moisture model is active, at the next ERF step (replaces `erf.fire_heat_feedback = true`) |
| `erf.fire.heat_profile` | `bottom_cell` puts the heat in the bottom ERF cell, as the LSFIRE coupling did; `exponential` (ERF's default) spreads it with the WRF-SFIRE profile |
| `erf.fire.grid_ratio` | Fire grid refinement relative to ERF level 0 (x and y), replacing `fire.fire_ref_ratio`. The fire grid is the ERF level-0 boxes refined by this ratio, on the same MPI ranks. |
| `erf.fire.plot_file`, `erf.fire.plot_int`, `erf.fire.plot_per` | The fire-grid plotfile (replaces `fire.fire_plot_file`); by default written with the ERF 3D plotfile |
| `fire.smoke_heat_per_kg` | Heat per kg of fuel burned [J/kg] for ERF's smoke tracer (`erf.fire.smoke_enable`); default 1.86e7 |
| `fire.fuel_model_source` | `uniform` (`fire.uniform_fuel_id`) or `landfire` (`fire.nc_fuel_file`, variable `FBFM` on the fire L0 grid). LANDFIRE fuel is used with atmospheric forcing |
| `fire.nc_fuel_file_l1`, `fire.nc_ignition_file_l1` | Optional full-domain fields at fire L1 resolution (`FBFM`, `fire_mask`); without them fire L1 takes each value from its fire L0 cell |
| `fire.band_thickness`, `fire.cfl`, `fire.subcycle_interval` | Narrow-band width, fire CFL, fire update every N ERF steps |
| `fire_amr.max_level` | Fire levels above fire L0: 0 (default) or 1. Fire levels are the fire model's own (x,y-only refinement) and unrelated to ERF's `amr.*` |
| `fire_amr.refinement_mode` | Required when `fire_amr.max_level = 1`; only `static` is supported |
| `fire_amr.nc_refine_mask_file` | NetCDF with int `refine_mask(Time, south_north, west_east)` on the fire L0 grid, 1 = build fire L1 there (alternatives read by LSFIRE: `fire_amr.static_corridor_*`, `fire_amr.static_polyline_file`) |
| `fire_amr.ref_ratio`, `fire_amr.blocking_factor`, `fire_amr.max_grid_size`, `fire_amr.n_error_buf` | Fire L1 ratio (x,y) and grid generation settings (defaults 2, 2, 64, 1) |

## Notes and limitations

- With a fire L1 the fire plotfile is LSFIRE's multi-level format (all fire levels, composite
  mask, arrival time; in atmospheric mode also the wind and its height on the finest fire
  level). Every fire level takes the ERF wind, its height and the terrain directly from the
  ERF cell it lies in. Heat and vapor go back to ERF from fire L0 only.
- ERF `amr.max_level` must be 0 when the fire is on; ERF level-0 boxes must span the whole
  vertical extent (do not decompose in z).
- Restart with the fire enabled aborts: fire checkpointing is not implemented yet.
- Misspellings used by older input templates abort with a message: `fire.bandthickness`
  (use `fire.band_thickness`) and `fire.fire_dt` (not a parameter).
- With `erf.fire.coupling_type = lagged` the fluxes reach ERF through the source term of
  every Runge-Kutta stage of the next ERF step (ERF's one-step lag), not after the dycore of
  the same step as the in-tree coupling did. ERF applies the Exner factor by default
  (`erf.fire.heat_tendency_exner`); `inputs_lsfire_heat_feedback` turns it off to match
  the previous coupling. The flux returned to ERF is the fire-L0 flux averaged over the
  last fire update (all fire substeps), so the energy given to ERF equals the energy the
  fire released. With `fire.subcycle_interval = N > 1` that mean is handed to ERF for each
  of the N ERF steps after the update, so it lags the release by up to N steps.
- The latent flux is handed to ERF in W/m2 (the vapor flux times ERF's `L_v`); ERF divides
  by the same `L_v`, so the water reaching ERF is the vapor flux LSFIRE computes.
- The reference results in the table above were obtained with the in-tree coupling that
  this engine directory replaces; they have not been rerun with the engine yet.
- Atmospheric wind is copied from each ERF cell to the fire cells it covers (no sub-cell
  structure). Its height above ground is passed per column; `wind_height` can be added to
  `fire.fire_plot_vars_1` to write it on the fire grid.
- The compressible ERF step must respect its stability limit (~0.85 s here at dx = 100 m);
  a perturbed atmosphere with `erf.fixed_dt = 1.0` is unstable in this setup.
