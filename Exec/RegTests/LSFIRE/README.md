# ERF + LSFIRE (ELMFIRE) fire coupling tests

Small cases for the LSFIRE level-set fire model coupled to ERF. The fire runs on its own
2D grid (`fire.fire_ref_ratio` times finer than ERF) and is coupled to **ERF level 0 only**;
ERF's AMR levels are atmospheric levels and are unrelated to the fire grid.

## Build

LSFIRE must be built against the same AMReX version as ERF. ERF links the LSFIRE static
library and supplies the AMReX symbols from its own AMReX build.

```
cmake -S <ERF> -B <build> ... \
      -DERF_ENABLE_FIRE=ON \
      -DELMFIRE_DIR=<LSFIRE source dir>          # provides include/ELMFIRE_API.H
      -DELMFIRE_LIB=<LSFIRE build>/libelmfire_core.a
cmake --build <build> --target erf_exec
```

Run from a directory containing the inputs file and `fuel_models.csv`:

```
erf_exec inputs_lsfire_constant_wind
```

## Cases

All cases: ERF level 0 is 60 x 30 x 4 cells at 100 m; `fire.fire_ref_ratio = 10` gives a
600 x 300 fire grid at 10 m; GR2 grass, point ignition at (500, 1500) m.

| Inputs | What it checks | Reference result (1 rank, CPU) |
|---|---|---|
| `inputs_lsfire_constant_wind` | `erf.fire_met_source = Constant`: the ERF wrapper does not change the fire model | Bitwise identical to LSFIRE `elmfire_standalone` with the same fire inputs (phi, fuel fraction, heat and moisture flux, hourly to 6 h). Burned area 318,700 m^2 at 1 h, 11,092,200 m^2 at 6 h. ~280 s. |
| `inputs_lsfire_atmos_wind` | `erf.fire_met_source = Atmosphere`: ERF surface wind drives the fire | Uniform 4.89 m/s at the lowest ERF cell centre (50 m above ground) reaches every fire cell; at 1 h burned area 146,700 m^2, head fire downwind. With `fire.atmos_wind_height=50` the result is bitwise identical; with `fire.atmos_wind_height=46.875` it reproduces the value before the k = 0 change (commit 036ad2e9e and earlier: k = 1 wind, fixed 46.875 m), 149,100 m^2. ~70 s. |
| `inputs_lsfire_heat_feedback` | `erf.fire_heat_feedback = true` with `MoistNoCondensation` | Domain integrals of rho*theta and rho*qv increase by exactly Sum(P dt)/Cp_d and Sum(E dt) (relative difference ~1e-11), with P and E the applied heat power and vapor rate printed every step (`erf.v = 2`). ~16 s. |

The constant-wind and atmospheric-wind areas differ by design: in constant-wind mode
`fire.constant_wind_u` is taken as the 20-ft wind; in atmospheric mode LSFIRE converts the
ERF wind from its actual height to the 20-ft height with a log law.

## Parameters

| Parameter | Meaning |
|---|---|
| `erf.fire_type` | `None` (default) or `ELMFIRE` |
| `erf.fire_met_source` | `Constant` (use `fire.constant_wind_u/v`) or `Atmosphere` (ERF wind at the lowest cell centre, k = 0). `fire.mode` is derived from it; if `fire.mode` is given it must agree. |
| `fire.atmos_wind_height` | Height above ground [m] assumed for the atmospheric wind. Default: not set, i.e. the actual height of the ERF k = 0 cell centre in each column (z_cc - surface). Set it only to impose a fixed height, e.g. in idealized tests. |
| `erf.fire_heat_feedback` | Add the fire heat flux to rho*theta, and the vapor flux to rho*qv when a moisture model is active, in the bottom ERF cell. Default `false`. |
| `fire.fire_ref_ratio` | Fire grid refinement relative to ERF level 0 (x and y). The fire grid uses the ERF level-0 boxes refined by this ratio, on the same MPI ranks. |
| `fire.fuel_model_source` | `uniform` (`fire.uniform_fuel_id`) or `landfire` (`fire.nc_fuel_file`) |
| `fire.band_thickness`, `fire.cfl`, `fire.subcycle_interval` | Narrow-band width, fire CFL, fire update every N ERF steps |

## Notes and limitations

- ERF `amr.max_level` must be 0 when the fire is on; ERF level-0 boxes must span the whole
  vertical extent (do not decompose in z).
- Restart with the fire enabled aborts: fire checkpointing is not implemented yet.
- Misspellings used by older input templates abort with a message: `fire.bandthickness`
  (use `fire.band_thickness`) and `fire.fire_dt` (not a parameter).
- Heat feedback is applied explicitly after the dycore, omits the Exner factor
  (theta vs. T; a few percent near the surface) and uses the instantaneous fire flux.
  Off by default; these tests check the coupling, not the physics of the feedback.
- Atmospheric wind is copied from each ERF cell to the fire cells it covers (no sub-cell
  structure). Its height above ground is passed per column; `wind_height` can be added to
  `fire.fire_plot_vars_1` to write it on the fire grid.
- The compressible ERF step must respect its stability limit (~0.85 s here at dx = 100 m);
  a perturbed atmosphere with `erf.fixed_dt = 1.0` is unstable in this setup.
