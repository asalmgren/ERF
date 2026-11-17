# ------------------  INPUTS TO MAIN PROGRAM  -------------------
#
# A release of a heavy gas (erf.heavy_gas, CO2) from a localized mass source in a closed box:
# slip walls on all six faces, so nothing enters or leaves. The source releases a constant
# 1.0e4 kg/s from the cells within 100 m of the centre and less than 100 m above the ground,
# 80% of it CO2 and 20% dry air, at the temperature of the cells it enters. So, from a start
# with no gas, the gas mass ("RHO SCALAR") must grow by 8.0e3 kg/s, the dry air ("MASS") by
# 2.0e3 kg/s and "TOTAL MASS" by 1.0e4 kg/s times the elapsed time.
#
# PROBLEM SIZE & GEOMETRY
max_step = 20

amrex.fpe_trap_invalid = 1
fabarray.mfiter_tile_size = 1024 1024 1024

geometry.prob_lo     =    0.    0.    0.
geometry.prob_hi     = 1000. 1000.  500.
amr.n_cell           =   32    32    16
amr.max_grid_size    =   16    16    16
geometry.is_periodic = 0 0 0
xlo.type = "SlipWall"
xhi.type = "SlipWall"
ylo.type = "SlipWall"
yhi.type = "SlipWall"
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.compressibility_type = FullyCompressible

# THE HEAVY GAS: the first passive scalar is rho_d * q_g of CO2
erf.heavy_gas.enable = true
erf.heavy_gas.R      = 188.9      # [J/kg/K]

# MASS SOURCE
erf.mass_source.center = 500.0 500.0
erf.mass_source.radius = 100.0
erf.mass_source.height = 100.0
erf.mass_source.rate   = 1.0e4
erf.mass_source.gas_mass_fraction = 0.8

erf.init_type           = "input_sounding"
erf.input_sounding_file = "input_sounding"

# TIME STEP CONTROL
erf.fixed_dt           = 0.5
erf.fixed_mri_dt_ratio = 10

# DIAGNOSTICS & VERBOSITY
erf.sum_interval = 1
erf.v            = 1
amr.v            = 1

# REFINEMENT / REGRIDDING
amr.max_level = 0

# CHECKPOINT FILES
erf.check_int   = -1

# PLOTFILES
erf.plot_file_1 = plt
erf.plot_int_1  = 20
erf.plot_vars_1 = density rhoadv_0 theta temp pressure heavy_gas_mass_fraction total_density z_velocity

# PHYSICS
erf.use_gravity = true
erf.buoyancy_type = 1
erf.les_type        = "None"
erf.molec_diff_type = "None"
erf.dryscal_horiz_adv_type = WENOZ5
erf.dryscal_vert_adv_type  = WENOZ5
