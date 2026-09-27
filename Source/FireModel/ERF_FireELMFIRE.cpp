#include "ERF_FireELMFIRE.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <fstream>
#include <iomanip>

// nvcc forbids __device__ lambdas in private/protected member functions.
// Hoist the box-fill kernel here so load_fuel_data and load_ignition_data
// can stay private.
static void LaunchFillFromNCKernel(const amrex::Box& gtbx,
                                    const amrex::Array4<int>& dst_arr,
                                    const amrex::Array4<const int>& src_arr,
                                    int i_lo, int i_hi, int j_lo, int j_hi)
{
    amrex::ParallelFor(gtbx,
        [dst_arr, src_arr, i_lo, i_hi, j_lo, j_hi] AMREX_GPU_DEVICE
        (int i, int j, int) noexcept
    {
        int li = amrex::min(amrex::max(i, i_lo), i_hi);
        int lj = amrex::min(amrex::max(j, j_lo), j_hi);
        dst_arr(i,j,0) = src_arr(li,lj,0);
    });
}

FireELMFIRE::FireELMFIRE()
{
    m_engine = std::make_unique<ELMFIRE::FireEngine>();
}

// Convert IGBP land use classification to fuel model ID
// IGBP classes: https://lpdaac.usgs.gov/products/mcd12q1v006/
int FireELMFIRE::IGBPToFuelModel(int igbp_index)
{
    switch (igbp_index) {
        case 1:  return 165; // Evergreen Needleleaf Forest → TU5
        case 2:  return 145; // Evergreen Broadleaf Forest → TL5
        case 3:  return 161; // Deciduous Needleleaf Forest → TU1
        case 4:  return 161; // Deciduous Broadleaf Forest → TU1
        case 5:  return 162; // Mixed Forest → TU2
        case 6:  return 124; // Closed Shrublands → GS4
        case 7:  return 122; // Open Shrublands → GS2
        case 8:  return 123; // Woody Savannas → GS3
        case 9:  return 123; // Savannas → GS3
        case 10: return 102; // Grasslands → GR2
        case 11: return 0;   // Permanent Wetlands → Non-burnable
        case 12: return 101; // Croplands → GR1
        case 13: return 162; // Urban and Built-Up → TU2
        case 14: return 121; // Cropland/Natural Mosaic → GS1
        case 15: return 0;   // Snow and Ice → Non-burnable
        case 16: return 0;   // Barren or Sparsely Vegetated → Non-burnable
        case 17: return 0;   // Water Bodies → Non-burnable
        case 18: return 102; // Wooded Tundra → GR2
        case 19: return 101; // Mixed Tundra → GR1
        case 20: return 0;   // Barren Tundra → Non-burnable
        case 21: return 0;   // Lake → Non-burnable
        default: return 102; // Unknown → GR2 (moderate grass)
    }
}

void FireELMFIRE::Define(SolverChoice& sc)
{
    // Load fire model parameters from ERF input file
    LoadConfigFromSolverChoice(sc);

    if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n";
        amrex::Print() << "========================================\n";
        amrex::Print() << "ELMFIRE Fire Model Configuration\n";
        amrex::Print() << "========================================\n";
        PrintConfig();
        amrex::Print() << "========================================\n";
        amrex::Print() << "\n";
    }
}

void FireELMFIRE::Init(int lev,
                      const amrex::MultiFab& cons_in,
                      const amrex::Geometry& geom,
                      const amrex::Real& dt_advance)
{
    m_lev = lev;
    m_geom = geom;
    m_current_time = 0.0;

    // Configure ELMFIRE engine
    m_config.cfl = m_cfl;
    m_config.band_thickness = m_band_thickness;
    m_config.fuel_model_source = m_fuel_model_source;
    m_config.uniform_fuel_id = m_uniform_fuel_id;
    m_config.fuel_database_path = m_fuel_database_path;

    // Set wind mode
    if (m_fire_mode == "constant_wind") {
        m_config.use_constant_wind = true;
        m_config.wind_u = m_constant_wind_u;
        m_config.wind_v = m_constant_wind_v;
        m_config.wind_height = m_wind_height;
    } else {
        m_config.use_constant_wind = false;
    }

    // Fuel moisture
    m_config.m1 = m_fuel_moisture_1hr;
    m_config.m10 = m_fuel_moisture_10hr;
    m_config.m100 = m_fuel_moisture_100hr;
    m_config.mlh = m_fuel_moisture_live_herb;
    m_config.mlw = m_fuel_moisture_live_woody;

    // Terrain slope
    m_config.slope_x = m_uniform_slope_x;
    m_config.slope_y = m_uniform_slope_y;

    // Wind data scale factor
    m_config.wind_data_scale_factor  = m_wind_data_scale_factor;
    m_config.use_grid_aware_burnout  = m_use_grid_aware_burnout;
    m_config.use_source_fuel_scaling = m_use_source_fuel_scaling;
    m_config.source_fuel_exponent    = m_source_fuel_exponent;
    m_config.terrain_scale_factor    = m_terrain_scale_factor;

    // Ignition parameters
    m_config.ignition_x = m_ignition_x;
    m_config.ignition_y = m_ignition_y;
    m_config.ignition_radius = m_ignition_radius;
    m_config.ignition_time = m_ignition_time;

    // Save ERF coarse BoxArray for heat flux averaging later
    m_ba = cons_in.boxArray();
    m_dm = cons_in.DistributionMap();
    m_geom = geom;

    // Determine whether ELMFIRE runs at a finer grid than ERF (Option A coupling)
    const amrex::Geometry* fire_geom_ptr = &geom;
    const amrex::BoxArray* fire_ba_ptr   = &m_ba;
    const amrex::DistributionMapping* fire_dm_ptr = &m_dm;

    if (m_fire_ref_ratio > 1) {
        // Build fine domain: refine ERF's coarse domain by ref_ratio in x,y only.
        // Fire model is 2D — keep z as a single layer (k=0).
        amrex::Box coarse_domain = geom.Domain();
        amrex::Box fine_domain(
            amrex::IntVect(0, 0, 0),
            amrex::IntVect(
                coarse_domain.length(0) * m_fire_ref_ratio - 1,
                coarse_domain.length(1) * m_fire_ref_ratio - 1,
                0));

        // Create fine BoxArray and matching DistributionMapping.
        // IMPORTANT: use maxSize(m_fire_ref_ratio) to guarantee every fine box is
        // exactly m_fire_ref_ratio cells in each horizontal dimension.
        // AMReX's maxSize algorithm internally divides out common powers of 2 from
        // (chunk, nlen), so maxSize(N) does NOT always produce N-cell boxes.
        // Example: maxSize(240) on 980 cells → ratio=4, bs_red=60, nlen_red=245,
        // numblk=5 → box size = 49×4 = 196 (not 240!).  196 is not divisible by 20,
        // so adjacent coarsened fine boxes overlap at cells 9, 19, 29, 39 and
        // average_down writes garbage at those cells.
        // With maxSize(20): ratio=4, bs_red=5, nlen_red=245 → numblk=49, sz=5,
        // box size = 5×4 = 20 cells exactly — no overlap after coarsening by 20.
        // Result: 49×39 = 1911 fine boxes of 20×20×1, which is fine for 64 MPI ranks.
        amrex::BoxArray ba_fine(fine_domain);
        ba_fine.maxSize(m_fire_ref_ratio);
        amrex::DistributionMapping dm_fine(ba_fine);
        m_fire_ba_fine = ba_fine;
        m_fire_dm_fine = dm_fine;

        // Build fine Geometry (same physical domain, finer cells)
        amrex::Array<int, AMREX_SPACEDIM> is_per = {0, 0, 0};
        m_geom_fine = amrex::Geometry(fine_domain, geom.ProbDomain(),
                                      geom.Coord(), is_per);

        // Allocate fine-level flux and wind MultiFabs
        m_heat_flux_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
        m_moist_flux_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
        m_heat_flux_fine.setVal(0.0);
        m_moist_flux_fine.setVal(0.0);
        m_U_fire_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
        m_V_fire_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
        m_U_fire_fine.setVal(0.0);
        m_V_fire_fine.setVal(0.0);

        // 2D coarse BA (khi=0): coarsen fire BA in x,y by fire_ref_ratio, z stays 0.
        // Used as intermediate for 2D→2D average_down (avoids OOB z access that
        // occurs when averaging from 2D fine (khi=0) directly into 3D ERF (khi=33)).
        m_ba_crse_2d = ba_fine;
        m_ba_crse_2d.coarsen(m_fire_ref_ratio);
        m_dm_crse_2d = amrex::DistributionMapping(m_ba_crse_2d);
        m_heat_flux_crse_2d.define(m_ba_crse_2d, m_dm_crse_2d, 1, 0);
        m_moist_flux_crse_2d.define(m_ba_crse_2d, m_dm_crse_2d, 1, 0);
        m_heat_flux_crse_2d.setVal(0.0);
        m_moist_flux_crse_2d.setVal(0.0);

        fire_geom_ptr = &m_geom_fine;
        fire_ba_ptr   = &m_fire_ba_fine;
        fire_dm_ptr   = &m_fire_dm_fine;

        if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "🔥 FireELMFIRE: fire_ref_ratio = " << m_fire_ref_ratio
                           << " → ELMFIRE domain: " << fine_domain << "\n"
                           << "   Cell size: " << m_geom_fine.CellSize()[0] << " m\n";
        }
    }

    // Initialize ELMFIRE engine at fire resolution (may differ from ERF resolution)
    m_engine->Initialize(*fire_geom_ptr, m_config, fire_ba_ptr, fire_dm_ptr);

#ifdef ERF_USE_NETCDF
    // LANDFIRE: Load fuel data at fire resolution
    if (m_fuel_model_source == "landfire") {
        if (m_nc_fuel_file.empty()) {
            amrex::Abort("fuel_model_source='landfire' requires fire.nc_fuel_file!");
        }
        load_fuel_data(*fire_geom_ptr, *fire_ba_ptr, *fire_dm_ptr);
    }

    // Area ignition: Load ignition mask at fire resolution
    if (m_ignition_type == "area") {
        if (m_nc_ignition_file.empty()) {
            amrex::Abort("ignition_type='area' requires fire.nc_ignition_file!");
        }
        load_ignition_data(*fire_geom_ptr, *fire_ba_ptr, *fire_dm_ptr);
    }
#else
    if (m_fuel_model_source == "landfire") {
        amrex::Abort("fuel_model_source='landfire' requires NetCDF support (ERF_USE_NETCDF)!");
    }
    if (m_ignition_type == "area") {
        amrex::Abort("ignition_type='area' requires NetCDF support (ERF_USE_NETCDF)!");
    }
#endif

    // Set ignition if ignition time is 0
    if (m_ignition_time == 0.0) {
        if (m_ignition_type == "point") {
            m_engine->SetIgnition(m_ignition_x, m_ignition_y, m_ignition_radius, 0.0);
        } else if (m_ignition_type == "area") {
            m_engine->SetAreaIgnition(m_ignition_mask_lev[0].get());
        }
    }

    m_initialized = true;

    if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "🔥 FireELMFIRE initialized at level " << lev << "\n";
        amrex::Print() << "   Mode: " << m_fire_mode << "\n";
        amrex::Print() << "   Domain: " << geom.Domain() << "\n";
        amrex::Print() << "   Cell size: " << geom.CellSize()[0] << " m\n";
    }
}

void FireELMFIRE::Advance(const amrex::Real& dt,
                         const amrex::MultiFab* U,
                         const amrex::MultiFab* V,
                         const amrex::MultiFab* terrain,
                         const amrex::iMultiFab* luindex)
{
    if (!m_initialized) {
        amrex::Abort("FireELMFIRE::Advance() called before Init()!");
    }

    // Track atmospheric time for subcycling
    m_accumulated_dt += dt;
    static int atmo_step_count = 0;
    atmo_step_count++;

    // Check if it's time to ignite (always check, even during subcycling)
    if (m_current_time <= m_ignition_time &&
        m_current_time + dt >= m_ignition_time) {
        if (m_ignition_type == "point") {
            m_engine->SetIgnition(m_ignition_x, m_ignition_y, m_ignition_radius, m_ignition_time);
        } else if (m_ignition_type == "area") {
            m_engine->SetAreaIgnition(m_ignition_mask_lev[0].get());
        }
        if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "🔥 Fire ignited at t = " << m_ignition_time << " s (type: " << m_ignition_type << ")\n";
        }
    }

    // Update current time (always advances, even if fire doesn't update)
    m_current_time += dt;

    // Check if it's time to update fire (subcycling)
    bool update_fire = (atmo_step_count % m_subcycle_interval == 0);

    if (!update_fire) {
        // Skip fire update this step (subcycling)
        if (m_verbosity > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  [Fire subcycling] Skipping fire update, accumulated_dt = "
                          << m_accumulated_dt << " s\n";
        }
        return;
    }

    // Time to update fire model with accumulated timestep
    amrex::Real fire_dt = m_accumulated_dt;
    m_accumulated_dt = 0.0;  // Reset accumulator

    if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        if (m_subcycle_interval > 1) {
            amrex::Print() << "  [Fire subcycling] Updating fire with dt = " << fire_dt
                          << " s (interval = " << m_subcycle_interval << ")\n";
        }
    }

    // Determine fuel source to pass to ELMFIRE (fallback logic)
    const amrex::iMultiFab* fuel_ptr = nullptr;

    if (m_fuel_model_source == "landfire" && !m_fuel_model_lev.empty() && m_fuel_model_lev[0]) {
        // Priority 1: Use LANDFIRE data loaded by fire model
        fuel_ptr = m_fuel_model_lev[0].get();
        if (m_verbosity > 1 && m_current_time < 1.0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "🔥 Using LANDFIRE fuel data\n";
        }
    } else if (m_fuel_model_source == "metgrid_igbp" && luindex != nullptr) {
        // Priority 2: Use luindex from ERF (IGBP conversion happens in ELMFIRE)
        fuel_ptr = luindex;
        if (m_verbosity > 1 && m_current_time < 1.0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "🔥 Using metgrid IGBP fuel data\n";
        }
    } else if (m_fuel_model_source == "uniform") {
        // Priority 3: Uniform mode - ELMFIRE uses m_uniform_fuel_id internally
        fuel_ptr = nullptr;
        if (m_verbosity > 1 && m_current_time < 1.0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "🔥 Using uniform fuel model ID: " << m_uniform_fuel_id << "\n";
        }
    }

    // Call appropriate ELMFIRE advance method based on mode
    if (m_config.use_constant_wind) {
        // Mode 1: Constant wind (offline testing)
        if (m_verbosity > 0 && m_current_time < 1.0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "FireELMFIRE: Using CONSTANT WIND mode\n";
        }
        m_engine->Advance(fire_dt);
    } else {
        // Mode 2: Atmospheric coupling (ERF-driven)
        if (m_verbosity > 0 && m_current_time < 1.0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "FireELMFIRE: Using ATMOSPHERIC COUPLING mode\n";
        }

        if (m_fire_ref_ratio > 1 && U != nullptr && V != nullptr) {
            // ELMFIRE runs at finer grid than ERF: nearest-neighbor prolongation
            // ------------------------------------------------------------------
            // Strategy: coarsen fine BoxArray → same spatial coverage as ERF coarse
            //   boxes. ParallelCopy gathers coarse ERF data to the right MPI ranks.
            //   Then prolong 1:1 within each MFIter (ic = i/rr, jc = j/rr).
            const int rr = m_fire_ref_ratio;

            amrex::BoxArray ba_c_from_f = m_fire_ba_fine;
            ba_c_from_f.coarsen(rr);
            // Use m_fire_dm_fine for the coarsened grid: ba_c_from_f has the same
            // number of boxes as m_fire_ba_fine (1:1 correspondence after coarsen),
            // so we can reuse m_fire_dm_fine to guarantee U_crse and U_fine share
            // the same box ownership.  A freshly-constructed DM would map boxes to
            // different ranks, causing U_crse.const_array(mfi) to crash on the
            // fire-box-owning ranks.
            const amrex::DistributionMapping& dm_c_from_f = m_fire_dm_fine;

            amrex::MultiFab U_crse(ba_c_from_f, dm_c_from_f, 1, 0);
            amrex::MultiFab V_crse(ba_c_from_f, dm_c_from_f, 1, 0);
            U_crse.ParallelCopy(*U, 0, 0, 1);
            V_crse.ParallelCopy(*V, 0, 0, 1);

            // Terrain: prolong from ERF coarse grid to fire fine grid (same as offline)
            const bool have_terrain = (terrain != nullptr);
            amrex::MultiFab terrain_crse, terrain_fine;
            if (have_terrain) {
                terrain_crse.define(ba_c_from_f, dm_c_from_f, 1, 0);
                terrain_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
                terrain_crse.ParallelCopy(*terrain, 0, 0, 1);
            }

            amrex::MultiFab U_fine(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
            amrex::MultiFab V_fine(m_fire_ba_fine, m_fire_dm_fine, 1, 0);

            for (amrex::MFIter mfi(U_fine, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto u_f = U_fine.array(mfi);
                auto v_f = V_fine.array(mfi);
                const auto u_c = U_crse.const_array(mfi);
                const auto v_c = V_crse.const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    int ic = i / rr, jc = j / rr;
                    u_f(i,j,k) = u_c(ic, jc, k);
                    v_f(i,j,k) = v_c(ic, jc, k);
                });
                if (have_terrain) {
                    auto t_f = terrain_fine.array(mfi);
                    const auto t_c = terrain_crse.const_array(mfi);
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        int ic = i / rr, jc = j / rr;
                        t_f(i,j,k) = t_c(ic, jc, k);
                    });
                }
            }

            m_engine->UpdateAtmosphericData(&U_fine, &V_fine,
                                            have_terrain ? &terrain_fine : nullptr,
                                            fuel_ptr);

            // Store prolongated winds for fire plotfile output (x_velocity / y_velocity)
            amrex::MultiFab::Copy(m_U_fire_fine, U_fine, 0, 0, 1, 0);
            amrex::MultiFab::Copy(m_V_fire_fine, V_fine, 0, 0, 1, 0);
        } else {
            // Same resolution: pass ERF wind directly
            m_engine->UpdateAtmosphericData(U, V, terrain, fuel_ptr);
        }

        // Advance fire with accumulated timestep
        m_engine->Advance(fire_dt);
    }

    // Diagnostics
    if (m_verbosity > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Real fire_area = m_engine->GetFireArea();
        int narrow_band_size = m_engine->GetNarrowBandSize();
        amrex::Print() << "   Fire: t = " << m_current_time << " s, "
                       << "area = " << fire_area << " m², "
                       << "active cells = " << narrow_band_size << "\n";
    }
}

void FireELMFIRE::GetHeatFlux(amrex::MultiFab& heat_flux)
{
    if (!m_initialized) {
        heat_flux.setVal(0.0);
        return;
    }

    if (m_fire_ref_ratio > 1) {
        // Get fine-level heat flux from ELMFIRE, then average down to ERF coarse grid.
        // Step 1: 2D→2D average_down with IntVect(rr,rr,1) — z stays at k=0.
        //   Using IntVect(rr) (same for all dims) would loop kk=0..rr-1 in z,
        //   accessing fire FABs at kk>0 which are out-of-bounds (fire is 2D, khi=0).
        // Step 2: ParallelCopy the 2D coarse result into k=0 of the 3D ERF MF.
        m_engine->GetHeatFlux(m_heat_flux_fine);
        m_heat_flux_crse_2d.setVal(0.0);
        amrex::average_down(m_heat_flux_fine, m_heat_flux_crse_2d,
                            0, 1, amrex::IntVect(m_fire_ref_ratio, m_fire_ref_ratio, 1));
        heat_flux.setVal(0.0);
        heat_flux.ParallelCopy(m_heat_flux_crse_2d, 0, 0, 1);
    } else {
        m_engine->GetHeatFlux(heat_flux);
    }
}

void FireELMFIRE::GetMoistureFlux(amrex::MultiFab& moisture_flux)
{
    if (!m_initialized) {
        moisture_flux.setVal(0.0);
        return;
    }

    if (m_fire_ref_ratio > 1) {
        // Same 2D→2D → ParallelCopy pattern as GetHeatFlux (see comment there).
        m_engine->GetMoistureFlux(m_moist_flux_fine);
        m_moist_flux_crse_2d.setVal(0.0);
        amrex::average_down(m_moist_flux_fine, m_moist_flux_crse_2d,
                            0, 1, amrex::IntVect(m_fire_ref_ratio, m_fire_ref_ratio, 1));
        moisture_flux.setVal(0.0);
        moisture_flux.ParallelCopy(m_moist_flux_crse_2d, 0, 0, 1);
    } else {
        m_engine->GetMoistureFlux(moisture_flux);
    }
}

amrex::Real FireELMFIRE::get_fire_area() const
{
    if (!m_initialized) {
        return 0.0;
    }

    return m_engine->GetFireArea();
}

void FireELMFIRE::GetPlotVarNames(amrex::Vector<std::string>& names) const
{
    // Always provide variable names, even before initialization
    // This is called during setup to determine which variables to plot
    auto elmfire_names = ELMFIRE::FireEngine::GetPlotVarNames();
    for (const auto& name : elmfire_names) {
        names.push_back(name);
    }
}

void FireELMFIRE::GetPlotVar(const std::string& name,
                            amrex::MultiFab& mf,
                            int lev) const
{
    if (!m_initialized) {
        mf.setVal(0.0);
        return;
    }

    // When fire runs at finer resolution than ERF (fire_ref_ratio > 1), heat/moisture
    // flux must be averaged from the 2D fire grid to the 3D ERF plotfile MultiFab.
    // We do this in two steps to avoid out-of-bounds z access:
    //   1. 2D→2D average_down with IntVect(rr,rr,1) into a temp 2D coarse MF
    //   2. ParallelCopy the 2D coarse result into k=0 of the 3D ERF plot MF
    // (Using IntVect(rr) in all 3 dims would loop kk=0..rr-1 in z and access
    //  fire FABs at kk>0 which are out-of-bounds since fire khi=0.)
    if (m_fire_ref_ratio > 1 && name == "fire_heat_flux") {
        amrex::MultiFab flux_fine(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
        m_engine->GetHeatFlux(flux_fine);
        amrex::MultiFab flux_crse_2d(m_ba_crse_2d, m_dm_crse_2d, 1, 0);
        flux_crse_2d.setVal(0.0);
        amrex::average_down(flux_fine, flux_crse_2d,
                            0, 1, amrex::IntVect(m_fire_ref_ratio, m_fire_ref_ratio, 1));
        mf.setVal(0.0);
        mf.ParallelCopy(flux_crse_2d, 0, 0, 1);
        return;
    }
    if (m_fire_ref_ratio > 1 && name == "fire_moisture_flux") {
        amrex::MultiFab flux_fine(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
        m_engine->GetMoistureFlux(flux_fine);
        amrex::MultiFab flux_crse_2d(m_ba_crse_2d, m_dm_crse_2d, 1, 0);
        flux_crse_2d.setVal(0.0);
        amrex::average_down(flux_fine, flux_crse_2d,
                            0, 1, amrex::IntVect(m_fire_ref_ratio, m_fire_ref_ratio, 1));
        mf.setVal(0.0);
        mf.ParallelCopy(flux_crse_2d, 0, 0, 1);
        return;
    }

    m_engine->FillPlotVar(name, mf);
}

// ============================================
// HELPER FUNCTIONS
// ============================================

void FireELMFIRE::LoadConfigFromSolverChoice(SolverChoice& sc)
{
    // Read fire model parameters from ERF input file using ParmParse
    amrex::ParmParse pp("fire");

    // Fire mode
    pp.query("mode", m_fire_mode);

    // Fire-level plotfile prefix (only used when fire_ref_ratio > 1)
    pp.query("fire_plot_file", m_fire_plot_file);

    // Fire plotfile variable list (e.g. fire.fire_plot_vars_1 = x_velocity y_velocity fire_phi)
    // If absent, WriteFirePlotFile falls back to the engine's default variable set.
    pp.queryarr("fire_plot_vars_1", m_fire_plot_vars);

    // Fuel model source ("uniform", "metgrid_igbp", or "landfire")
    pp.query("fuel_model_source", m_fuel_model_source);
    if (m_fuel_model_source != "uniform" &&
        m_fuel_model_source != "metgrid_igbp" &&
        m_fuel_model_source != "landfire") {
        amrex::Print() << "WARNING: Unknown fuel_model_source '" << m_fuel_model_source
                       << "', defaulting to 'uniform'\n";
        m_fuel_model_source = "uniform";
    }

    // LANDFIRE NetCDF file path (fire namespace)
    pp.query("nc_fuel_file", m_nc_fuel_file);

    // Validate: landfire mode REQUIRES nc_fuel_file
    if (m_fuel_model_source == "landfire") {
        if (m_nc_fuel_file.empty()) {
            amrex::Abort("fire.fuel_model_source='landfire' requires fire.nc_fuel_file to be specified!");
        }
    }

    // Fuel model ID (for uniform mode)
    pp.query("uniform_fuel_id", m_uniform_fuel_id);
    pp.query("fuel_database_path", m_fuel_database_path);

    // Constant wind parameters (for offline mode)
    pp.query("constant_wind_u", m_constant_wind_u);
    pp.query("constant_wind_v", m_constant_wind_v);
    pp.query("wind_height", m_wind_height);

    // Fuel moisture
    pp.query("fuel_moisture_1hr", m_fuel_moisture_1hr);
    pp.query("fuel_moisture_10hr", m_fuel_moisture_10hr);
    pp.query("fuel_moisture_100hr", m_fuel_moisture_100hr);
    pp.query("fuel_moisture_live_herb", m_fuel_moisture_live_herb);
    pp.query("fuel_moisture_live_woody", m_fuel_moisture_live_woody);

    // Terrain slope (uniform, if not using terrain MultiFab)
    pp.query("uniform_slope_x", m_uniform_slope_x);
    pp.query("uniform_slope_y", m_uniform_slope_y);

    // Ignition parameters
    pp.query("ignition_type", m_ignition_type);
    pp.query("ignition_x", m_ignition_x);
    pp.query("ignition_y", m_ignition_y);
    pp.query("ignition_radius", m_ignition_radius);
    pp.query("ignition_time", m_ignition_time);
    pp.query("nc_ignition_file", m_nc_ignition_file);
    pp.query("ignition_time_index", m_ignition_time_index);

    // Numerical parameters
    pp.query("cfl", m_cfl);
    pp.query("band_thickness", m_band_thickness);

    // Wind data scale factor (HRRR bias correction)
    pp.query("wind_data_scale_factor", m_wind_data_scale_factor);

    // Fire suppression options (default: off for backward compatibility)
    pp.query("use_grid_aware_burnout",  m_use_grid_aware_burnout);
    pp.query("use_source_fuel_scaling", m_use_source_fuel_scaling);
    pp.query("source_fuel_exponent",    m_source_fuel_exponent);
    pp.query("terrain_scale_factor",    m_terrain_scale_factor);

    // Subcycling (fire updates every N atmospheric timesteps)
    pp.query("subcycle_interval", m_subcycle_interval);
    if (m_subcycle_interval < 1) m_subcycle_interval = 1;

    // Verbosity
    pp.query("verbosity", m_verbosity);

    // AMR refinement: ELMFIRE runs at finer grid than ERF (Option A coupling)
    pp.query("fire_ref_ratio", m_fire_ref_ratio);
    if (m_fire_ref_ratio < 1) m_fire_ref_ratio = 1;
}

#ifdef ERF_USE_NETCDF
void FireELMFIRE::load_fuel_data(const amrex::Geometry& geom,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm)
{
    using namespace amrex;

    if (m_verbosity > 0 && ParallelDescriptor::IOProcessor()) {
        Print() << "🔥 Loading LANDFIRE fuel data from " << m_nc_fuel_file << "\n";
    }

    // Allocate fuel data storage for this level
    m_fuel_model_lev.resize(1);  // Single level for now
    IntVect ngv(0, 0, 0);  // No ghost cells for static data
    m_fuel_model_lev[0] = std::make_unique<iMultiFab>(ba, dm, 1, ngv);

    // Read fuel data from NetCDF on IO processor
    Box domain = geom.Domain();
    IArrayBox NC_fuel_iab;

    int success = read_static_fuel_from_netcdf(domain, m_nc_fuel_file, NC_fuel_iab);

    if (success == 0) {
        Abort("Failed to read LANDFIRE fuel data from: " + m_nc_fuel_file);
    }

    // Distribute to all processors (pattern from ERF_InitFromMetgrid.cpp:242-258)
    Box src_box = NC_fuel_iab.box();
    int i_lo = src_box.smallEnd(0);
    int i_hi = src_box.bigEnd(0);
    int j_lo = src_box.smallEnd(1);
    int j_hi = src_box.bigEnd(1);

    for (MFIter mfi(*m_fuel_model_lev[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box gtbx = mfi.growntilebox();
        IArrayBox& dst = (*m_fuel_model_lev[0])[mfi];
        const Array4<int>& dst_arr = dst.array();
        const Array4<const int>& src_arr = NC_fuel_iab.const_array();

        LaunchFillFromNCKernel(gtbx, dst_arr, src_arr, i_lo, i_hi, j_lo, j_hi);
    }

    m_fuel_model_lev[0]->FillBoundary(geom.periodicity());

    if (m_verbosity > 0 && ParallelDescriptor::IOProcessor()) {
        Print() << "🔥 LANDFIRE fuel data loaded successfully\n";
    }
}

void FireELMFIRE::load_ignition_data(const amrex::Geometry& geom,
                                     const amrex::BoxArray& ba,
                                     const amrex::DistributionMapping& dm)
{
    using namespace amrex;

    if (m_verbosity > 0 && ParallelDescriptor::IOProcessor()) {
        Print() << "🔥 Loading ignition mask from " << m_nc_ignition_file << "\n";
    }

    // Allocate ignition mask storage for this level
    m_ignition_mask_lev.resize(1);  // Single level for now
    IntVect ngv(0, 0, 0);  // No ghost cells for static data
    m_ignition_mask_lev[0] = std::make_unique<iMultiFab>(ba, dm, 1, ngv);

    // Read ignition mask from NetCDF on IO processor
    Box domain = geom.Domain();
    IArrayBox NC_ign_iab;

    int success = read_ignition_from_netcdf(domain, m_nc_ignition_file, m_ignition_time_index, NC_ign_iab);

    if (success == 0) {
        Abort("Failed to read ignition mask from: " + m_nc_ignition_file);
    }

    // Distribute to all processors (same pattern as fuel data)
    Box src_box = NC_ign_iab.box();
    int i_lo = src_box.smallEnd(0);
    int i_hi = src_box.bigEnd(0);
    int j_lo = src_box.smallEnd(1);
    int j_hi = src_box.bigEnd(1);

    int total_ignited = 0;
    for (MFIter mfi(*m_ignition_mask_lev[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box gtbx = mfi.growntilebox();
        IArrayBox& dst = (*m_ignition_mask_lev[0])[mfi];
        const Array4<int>& dst_arr = dst.array();
        const Array4<const int>& src_arr = NC_ign_iab.const_array();

        LaunchFillFromNCKernel(gtbx, dst_arr, src_arr, i_lo, i_hi, j_lo, j_hi);
        
        // Count ignited cells on this MFIter
        for (int j = gtbx.smallEnd(1); j <= gtbx.bigEnd(1); j++) {
            for (int i = gtbx.smallEnd(0); i <= gtbx.bigEnd(0); i++) {
                if (dst_arr(i,j,0) == 1) total_ignited++;
            }
        }
    }

    // Sum across all ranks
    ParallelDescriptor::ReduceIntSum(total_ignited);

    m_ignition_mask_lev[0]->FillBoundary(geom.periodicity());

    if (m_verbosity > 0 && ParallelDescriptor::IOProcessor()) {
        Print() << "🔥 Ignition mask loaded successfully (" << total_ignited << " cells to ignite)\n";
    }
}
#endif

void FireELMFIRE::PrintConfig() const
{
    amrex::Print() << "  Fire mode: " << m_fire_mode << "\n";
    amrex::Print() << "  Fuel model source: " << m_fuel_model_source << "\n";

    if (m_fuel_model_source == "uniform") {
        amrex::Print() << "    Uniform fuel ID: " << m_uniform_fuel_id << "\n";
    } else if (m_fuel_model_source == "metgrid_igbp") {
        amrex::Print() << "    Using IGBP land use from metgrid\n";
    } else if (m_fuel_model_source == "landfire") {
        amrex::Print() << "    LANDFIRE file: " << m_nc_fuel_file << "\n";
    }

    if (!m_fuel_database_path.empty()) {
        amrex::Print() << "  Fuel database: " << m_fuel_database_path << "\n";
    }

    if (m_fire_mode == "constant_wind") {
        amrex::Print() << "  Wind: (" << m_constant_wind_u << ", "
                       << m_constant_wind_v << ") m/s at "
                       << m_wind_height << " m\n";
    }

    amrex::Print() << "  Fuel moisture:\n";
    amrex::Print() << "    1hr: " << m_fuel_moisture_1hr << "\n";
    amrex::Print() << "    10hr: " << m_fuel_moisture_10hr << "\n";
    amrex::Print() << "    100hr: " << m_fuel_moisture_100hr << "\n";
    amrex::Print() << "    Live herb: " << m_fuel_moisture_live_herb << "\n";
    amrex::Print() << "    Live woody: " << m_fuel_moisture_live_woody << "\n";

    if (m_uniform_slope_x != 0.0 || m_uniform_slope_y != 0.0) {
        amrex::Print() << "  Uniform slope: (" << m_uniform_slope_x << ", "
                       << m_uniform_slope_y << ")\n";
    }

    amrex::Print() << "  Ignition:\n";
    amrex::Print() << "    Type: " << m_ignition_type << "\n";
    if (m_ignition_type == "point") {
        amrex::Print() << "    Location: (" << m_ignition_x << ", "
                       << m_ignition_y << ") m\n";
        amrex::Print() << "    Radius: " << m_ignition_radius << " m\n";
    } else if (m_ignition_type == "area") {
        amrex::Print() << "    NetCDF mask file: " << m_nc_ignition_file << "\n";
        amrex::Print() << "    Time index: " << m_ignition_time_index << "\n";
    }
    amrex::Print() << "    Time: " << m_ignition_time << " s\n";

    amrex::Print() << "  Numerics:\n";
    amrex::Print() << "    CFL: " << m_cfl << "\n";
    amrex::Print() << "    Band thickness: " << m_band_thickness << " cells\n";
    if (m_subcycle_interval > 1) {
        amrex::Print() << "    Subcycling: fire updates every " << m_subcycle_interval
                       << " atmospheric steps\n";
    }
    amrex::Print() << "  Fire suppression:\n";
    amrex::Print() << "    use_grid_aware_burnout:  " << (m_use_grid_aware_burnout  ? "true" : "false") << "\n";
    amrex::Print() << "    use_source_fuel_scaling: " << (m_use_source_fuel_scaling ? "true" : "false") << "\n";
    if (m_use_source_fuel_scaling)
        amrex::Print() << "    source_fuel_exponent:    " << m_source_fuel_exponent << "\n";
    amrex::Print() << "    terrain_scale_factor:    " << m_terrain_scale_factor << "\n";
}

void FireELMFIRE::WriteCheckpointData(const std::string& chk_dir, int lev) const
{
    if (!m_initialized || !m_engine) return;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Writing fire checkpoint data at level " << lev << "\n";
    }

    // Write fire phi (level set) - this is the essential fire state
    amrex::MultiFab phi_out(m_ba, m_dm, 1, 0);
    m_engine->GetPhiField(phi_out);
    amrex::VisMF::Write(phi_out, chk_dir + "/FirePhi");

    // Write fire time for consistency check on restart
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream ofs(chk_dir + "/FireHeader");
        ofs << std::setprecision(17) << m_current_time << "\n";
        ofs.close();
    }
}

void FireELMFIRE::ReadCheckpointData(const std::string& chk_dir, int lev)
{
    if (!m_initialized || !m_engine) return;

    // Check if fire checkpoint data exists (VisMF writes a _H header file)
    std::string phi_header = chk_dir + "/FirePhi_H";
    bool has_fire_chk = false;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ifstream test(phi_header);
        has_fire_chk = test.good();
    }
    int flag = has_fire_chk ? 1 : 0;
    amrex::ParallelDescriptor::Bcast(&flag, 1,
        amrex::ParallelDescriptor::IOProcessorNumber());
    has_fire_chk = (flag != 0);

    if (!has_fire_chk) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "No fire checkpoint data found in " << chk_dir
                          << ", fire starts from initial conditions\n";
        }
        return;
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Reading fire checkpoint data at level " << lev << "\n";
    }

    // Read phi and restore fire state
    amrex::MultiFab phi_restart(m_ba, m_dm, 1, 1);
    amrex::VisMF::Read(phi_restart, chk_dir + "/FirePhi");
    m_engine->RestorePhiFromCheckpoint(phi_restart);

    // Read and verify fire time
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::string header_file = chk_dir + "/FireHeader";
        std::ifstream ifs(header_file);
        if (ifs.good()) {
            amrex::Real fire_chk_time = 0.0;
            ifs >> fire_chk_time;
            amrex::Print() << "  Fire checkpoint time: " << fire_chk_time
                          << " s, ERF time: " << m_current_time << " s\n";
        }
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Fire state restored from checkpoint\n";
    }
}

void FireELMFIRE::WriteFirePlotFile(int step, amrex::Real time) const
{
    // Only write when ELMFIRE runs at finer resolution than ERF
    if (!m_initialized || m_fire_ref_ratio <= 1 || !m_engine) return;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "🔥 WriteFirePlotFile: step=" << step << " time=" << time << "\n";
    }

    // Determine which variables to write:
    // - If fire.fire_plot_vars_1 was specified, use that list.
    // - Otherwise fall back to the engine's default set.
    amrex::Vector<std::string> var_names;
    if (!m_fire_plot_vars.empty()) {
        var_names = m_fire_plot_vars;
    } else {
        auto std_var_names = ELMFIRE::FireEngine::GetPlotVarNames();
        var_names.assign(std_var_names.begin(), std_var_names.end());
    }
    const int ncomp = static_cast<int>(var_names.size());

    // Build multi-component MultiFab at fire resolution
    amrex::MultiFab mf(m_fire_ba_fine, m_fire_dm_fine, ncomp, 0);
    mf.setVal(0.0);

    for (int c = 0; c < ncomp; ++c) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  FillPlotVar: " << var_names[c] << "\n";
        }
        amrex::MultiFab tmp(m_fire_ba_fine, m_fire_dm_fine, 1, 0);
        tmp.setVal(0.0);

        if (var_names[c] == "x_velocity") {
            // Atmospheric U wind prolongated to fire resolution
            amrex::MultiFab::Copy(tmp, m_U_fire_fine, 0, 0, 1, 0);
        } else if (var_names[c] == "y_velocity") {
            // Atmospheric V wind prolongated to fire resolution
            amrex::MultiFab::Copy(tmp, m_V_fire_fine, 0, 0, 1, 0);
        } else {
            m_engine->FillPlotVar(var_names[c], tmp);
        }

        amrex::MultiFab::Copy(mf, tmp, 0, c, 1, 0);
        amrex::Real var_min = tmp.min(0);  // MPI collective: all ranks must call this
        amrex::Real var_max = tmp.max(0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "    done: [" << var_min << ", " << var_max << "]\n";
        }
    }

    // Plotfile name: e.g. "fire_plt0007200"
    std::string pltname = amrex::Concatenate(m_fire_plot_file, step, 7);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Writing fire plotfile: " << pltname << "\n";
    }

    // Write as single-level AMReX plotfile at fire resolution
    amrex::WriteSingleLevelPlotfile(pltname, mf, var_names, m_geom_fine, time, step);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Fire plotfile written: " << pltname
                       << "  (fire grid: " << m_geom_fine.Domain() << ")\n";
    }
}
