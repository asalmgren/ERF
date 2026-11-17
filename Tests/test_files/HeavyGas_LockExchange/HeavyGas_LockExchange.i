# ------------------  INPUTS TO MAIN PROGRAM  -------------------
# Lock exchange: CO2 (mass fraction 0.5, 50 m deep) behind a lock at x = 200 m, at the ambient
# temperature, released at t = 0 into dry air at rest. Each column starts in discrete hydrostatic
# balance (see Source/Prob/ERF_InitCustomPert_HeavyGas.H); the gas then slumps and runs along
# the floor as a gravity current, at roughly Fr sqrt(g' h) with Fr ~ 0.5-0.7.
erf.prob_name = "heavy gas"
erf.init_type = Isentropic
prob.T_0      = 300.0

max_step = 40

amrex.fpe_trap_invalid = 1
fabarray.mfiter_tile_size = 1024 1024 1024

# PROBLEM SIZE & GEOMETRY
geometry.prob_lo     =    0.   0.   0.
geometry.prob_hi     = 2000.  40. 200.
amr.n_cell           =  100    2   40
amr.max_grid_size    =   50    2   40    # the columns must not be split in z
geometry.is_periodic = 0 1 0
xlo.type = "SlipWall"
xhi.type = "SlipWall"
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.compressibility_type = FullyCompressible
erf.buoyancy_type        = 1

# THE HEAVY GAS: the first passive scalar is rho_d * q_g of CO2
erf.heavy_gas.enable    = true
erf.heavy_gas.R         = 188.9      # [J/kg/K]
prob.gas_mass_fraction  = 0.5
prob.gas_depth          = 50.0       # [m]
prob.gas_x_hi           = 200.0      # [m] the lock

# TIME STEP CONTROL
erf.fixed_dt           = 0.1
erf.fixed_mri_dt_ratio = 10

# DIAGNOSTICS & VERBOSITY
erf.sum_interval = 1
erf.v            = 1
amr.v            = 1

# REFINEMENT / REGRIDDING
amr.max_level = 0

# CHECKPOINT FILES
erf.check_int = -1

# PLOTFILES
erf.plot_file_1 = plt
erf.plot_int_1  = 40
erf.plot_vars_1 = density rhoadv_0 x_velocity y_velocity z_velocity pressure temp theta total_density heavy_gas_mass_fraction

# PHYSICS
erf.use_gravity     = true
erf.les_type        = "None"
erf.molec_diff_type = "None"
erf.dryscal_horiz_adv_type = WENOZ5
erf.dryscal_vert_adv_type  = WENOZ5
