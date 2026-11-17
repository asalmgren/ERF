    //
    // A heavy gas (erf.heavy_gas) at rest in a dry ambient at rest: a layer of depth prob.gas_depth
    //    over the whole floor, or, with prob.gas_x_lo / prob.gas_x_hi, a lock of gas between them
    //    (the lock-exchange problem). The gas has mass fraction prob.gas_mass_fraction and is at the
    //    ambient temperature of its height.
    //
    // Each column is in discrete hydrostatic balance with the solver's buoyancy (buoyancy_type 1):
    //    integrating down from the top of the domain,
    //        p'(k) = p'(k+1) + g dz/2 (rho'(k) + rho'(k+1)),   rho' = rho_d (1 + q_g) - rho_0,
    //    with p = rho_d T (R_d + q_g R_g), where p' = p - p_0 and dz is the distance between the
    //    cell centres. Where there is no gas the state is the ambient one, unchanged.
    //
    // NOTE: a column is integrated within one box, so the boxes must span the domain in z
    //
    ParmParse pp_prob("prob");
    Real Y_gas = amrex::Real(0.5); pp_prob.query("gas_mass_fraction", Y_gas);
    Real h_gas = zero;             pp_prob.query("gas_depth", h_gas);
    Real x_lo_gas = geomdata.ProbLo(0); pp_prob.query("gas_x_lo", x_lo_gas);
    Real x_hi_gas = geomdata.ProbHi(0); pp_prob.query("gas_x_hi", x_hi_gas);

    if (!sc.heavy_gas) {
        Abort("prob_name = \"heavy gas\" needs erf.heavy_gas.enable = true");
    }
    if (!(Y_gas >= zero && Y_gas < one)) {
        Abort("prob.gas_mass_fraction must be in [0, 1)");
    }
    if (sc.terrain_type != TerrainType::None) {
        Abort("prob_name = \"heavy gas\" is only implemented without terrain");
    }
    const int klo = geomdata.Domain().smallEnd(2);
    const int khi = geomdata.Domain().bigEnd(2);
    if (bx.smallEnd(2) != klo || bx.bigEnd(2) != khi) {
        Abort("prob_name = \"heavy gas\" needs boxes that span the domain in z (raise amr.max_grid_size)");
    }

    const bool use_z_phys = use_physical_z();
    const Real q_gas  = Y_gas / (one - Y_gas);
    const Real R_gas  = sc.R_g;
    const Real RgoRv  = sc.RgoRv;
    const Real grav   = sc.gravity;

    Box b2d = bx; b2d.setRange(2, 0);

    ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
    {
        const auto prob_lo = geomdata.ProbLo();
        const auto dx      = geomdata.CellSize();
        const Real x = prob_lo[0] + (i + myhalf) * dx[0];
        const bool gas_column = (x > x_lo_gas && x < x_hi_gas);

        Real pp_above = zero;   // p' and rho' in the cell above
        Real rp_above = zero;
        Real z_above  = zero;

        for (int k = khi; k >= klo; --k)
        {
            const Real z = use_z_phys ? z_cc(i,j,k) : prob_lo[2] + (k + myhalf) * dx[2];
            const Real q = (gas_column && z < h_gas) ? q_gas : zero;

            if (q == zero && pp_above == zero && rp_above == zero) {
                // ambient: leave the state as it is
                state_pert(i,j,k,Rho_comp)       = zero;
                state_pert(i,j,k,RhoTheta_comp)  = zero;
                state_pert(i,j,k,RhoScalar_comp) = -state(i,j,k,RhoScalar_comp);
                z_above = z;
                continue;
            }

            const Real p0 = p_hse(i,j,k);
            const Real r0 = r_hse(i,j,k);
            const Real T  = p0 / (R_d * r0);                  // ambient temperature
            const Real a  = (one + q) / (T * (R_d + q * R_gas)); // rho_d (1 + q) = a p

            // the cell above holds no perturbation at the top of the domain
            const Real c  = (k < khi) ? myhalf * grav * (z_above - z) : zero;
            const Real pp = (pp_above + c * (rp_above + p0 * a - r0)) / (one - c * a);

            const Real rho_tot = (p0 + pp) * a;
            const Real rho_d   = rho_tot / (one + q);

            state_pert(i,j,k,Rho_comp)       = rho_d - state(i,j,k,Rho_comp);
            state_pert(i,j,k,RhoTheta_comp)  = getRhoThetagivenP(p0 + pp, getQforEOS(zero, q, RgoRv))
                                             - state(i,j,k,RhoTheta_comp);
            state_pert(i,j,k,RhoScalar_comp) = rho_d * q - state(i,j,k,RhoScalar_comp);

            pp_above = pp;
            rp_above = rho_tot - r0;
            z_above  = z;
        }
    });
