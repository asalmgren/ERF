#include "ERF_FireELMFIRE.H"
#include "Core/FireHierarchy.H"
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


// FireEngine reads and writes its inputs/outputs in host loops, so every MultiFab
// handed to it must be host-accessible (managed memory on GPU builds; plain memory
// on CPU builds), and device kernels writing such data must be synchronized first.
static amrex::MFInfo HostAccessible ()
{
    return amrex::MFInfo().SetArena(amrex::The_Managed_Arena());
}

// Fire cell (i,j) of a fire level takes the value of the coarse cell (i/r, j/r) it lies in;
// the coarse data were already copied onto this level's layout coarsened by (r,r,1).
static void LaunchProlongNearest (const amrex::Box& bx, int r,
                                  const amrex::Array4<const amrex::Real>& c,
                                  const amrex::Array4<amrex::Real>& f)
{
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        f(i,j,k) = c(i/r, j/r, k);
    });
}
static void LaunchProlongNearestInt (const amrex::Box& bx, int r,
                                     const amrex::Array4<const int>& c,
                                     const amrex::Array4<int>& f)
{
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        f(i,j,k) = c(i/r, j/r, k);
    });
}

/**
 * LSFIRE FireHierarchy driven by ERF. Geometry, grids and fire parameters come from
 * FireELMFIRE. Before a fire level advances, it receives the ERF surface wind, its height
 * and the terrain height taken directly from the ERF level-0 grid (each fire cell uses the
 * ERF cell it lies in), plus its fuel map. Area-ignition masks and fuel maps for fire
 * levels above 0 come from a full-resolution field when given, else from fire level 0.
 * Optionally accumulates the fire-level-0 fluxes for their time average over an interval.
 */
class ERFFireHierarchy : public FireHierarchy
{
public:
    ERFFireHierarchy (const amrex::Geometry& fire_l0_geom, const amrex::AmrInfo& info,
                      const ELMFIRE::Config& config, int fire_ref_ratio, bool atmospheric,
                      amrex::Real ign_x, amrex::Real ign_y, amrex::Real ign_radius, amrex::Real ign_time)
        : FireHierarchy(fire_l0_geom, info), m_fire_ref_ratio(fire_ref_ratio), m_atmospheric(atmospheric)
    {
        // FireELMFIRE's parsing of fire.* stays authoritative for the engine config
        m_fire_config     = config;
        m_ignition_x      = ign_x;
        m_ignition_y      = ign_y;
        m_ignition_radius = ign_radius;
        m_ignition_time   = ign_time;
        m_ignition_shape  = "point";
    }

    /** ERF 2D fields (ERF level-0 layout, bottom layer): wind, its height above ground
     *  (may be null) and terrain height. Used by every fire level until the next call. */
    void SetAtmosphere (const amrex::MultiFab* u, const amrex::MultiFab* v,
                        const amrex::MultiFab* height, const amrex::MultiFab* terrain)
    {
        m_src_u = u; m_src_v = v; m_src_h = height; m_src_terrain = terrain;
        m_atm_serial++;
    }
    void ClearAtmosphere () { m_src_u = m_src_v = m_src_h = m_src_terrain = nullptr; }

    /** Fuel map on fire level 0 and, optionally, a full-domain field at fire level-1
     *  resolution (otherwise level 1 samples level 0). */
    void SetFuel (const amrex::iMultiFab* fuel_l0, const amrex::IArrayBox* fuel_l1_full)
    { m_fuel_l0 = fuel_l0; m_fuel_l1_full = fuel_l1_full; }

    /** Area-ignition mask on fire level 0 and, optionally, at full level-1 resolution. */
    void SetAreaIgnition (const amrex::iMultiFab* mask_l0, const amrex::IArrayBox* mask_l1_full)
    { m_ign_l0 = mask_l0; m_ign_l1_full = mask_l1_full; SetIgnitionMask(0, mask_l0); }

    /** Accumulate fire-level-0 heat and vapor fluxes x dt between Begin and End. */
    void BeginFluxAverage () {
        if (!m_acc_heat.ok()) {
            m_acc_heat.define(boxArray(0), DistributionMap(0), 1, 0, HostAccessible());
            m_acc_moist.define(boxArray(0), DistributionMap(0), 1, 0, HostAccessible());
            m_tmp_flux.define(boxArray(0), DistributionMap(0), 1, 0, HostAccessible());
        }
        m_acc_heat.setVal(0.0); m_acc_moist.setVal(0.0); m_acc_time = 0.0; m_accumulating = true;
    }
    /** Time-averaged fluxes over the interval (0 if no time elapsed) into level-0 MultiFabs. */
    void EndFluxAverage (amrex::MultiFab& heat_avg, amrex::MultiFab& moist_avg) {
        m_accumulating = false;
        heat_avg.setVal(0.0); moist_avg.setVal(0.0);
        if (m_acc_time > 0.0) {
            amrex::MultiFab::Saxpy(heat_avg,  1.0 / m_acc_time, m_acc_heat,  0, 0, 1, 0);
            amrex::MultiFab::Saxpy(moist_avg, 1.0 / m_acc_time, m_acc_moist, 0, 0, 1, 0);
        }
        amrex::Gpu::streamSynchronize();
    }

    /** Level fields last passed to the fire (plotting); null if none. */
    const amrex::MultiFab* LevelWindU (int lev) const { return lev < m_u.size() ? m_u[lev].get() : nullptr; }
    const amrex::MultiFab* LevelWindV (int lev) const { return lev < m_v.size() ? m_v[lev].get() : nullptr; }
    const amrex::MultiFab* LevelWindHeight (int lev) const { return (lev < m_h.size() && m_src_h_used) ? m_h[lev].get() : nullptr; }

    void IgniteLevel (int lev, amrex::Real time, bool initial) override
    {
        if (lev > 0 && m_ign_l0 != nullptr) {
            MakeLevelIntField(lev, m_ign_l0, m_ign_l1_full, m_ign);
            SetIgnitionMask(lev, m_ign[lev].get());
        }
        FireHierarchy::IgniteLevel(lev, time, initial);
    }

protected:
    // Wind actually given to the finest fire level, in the multi-level fire plotfile
    void GetExtraPlotVarNames (amrex::Vector<std::string>& names) const override
    {
        if (m_atmospheric) { names.push_back("x_velocity"); names.push_back("y_velocity"); names.push_back("wind_height"); }
    }
    // Called for any fire level (LSFIRE may ask for every level): find the level by layout
    void FillExtraPlotVar (const std::string& name, amrex::MultiFab& mf) const override
    {
        for (int lev = 0; lev <= finest_level; ++lev) {
            if (mf.boxArray() != boxArray(lev) || mf.DistributionMap() != DistributionMap(lev)) { continue; }
            const amrex::MultiFab* src = (name == "x_velocity") ? LevelWindU(lev)
                                       : (name == "y_velocity") ? LevelWindV(lev) : LevelWindHeight(lev);
            if (src != nullptr) { amrex::MultiFab::Copy(mf, *src, 0, 0, 1, 0); }
            return;
        }
    }

    void AdvanceLevel (int lev, amrex::Real time, amrex::Real dt) override
    {
        if (m_src_u != nullptr && m_src_v != nullptr) {
            FillLevelAtmosphere(lev);
            const amrex::iMultiFab* fuel = nullptr;
            if (m_fuel_l0 != nullptr) {
                if (lev == 0) { fuel = m_fuel_l0; }
                else { MakeLevelIntField(lev, m_fuel_l0, m_fuel_l1_full, m_fuel); fuel = m_fuel[lev].get(); }
            }
            LevelEngine(lev).UpdateAtmosphericData(m_u[lev].get(), m_v[lev].get(),
                                                   m_src_terrain ? m_t[lev].get() : nullptr,
                                                   fuel, m_src_h ? m_h[lev].get() : nullptr);
        }
        FireHierarchy::AdvanceLevel(lev, time, dt);

        if (lev == 0 && m_accumulating) {
            amrex::Gpu::streamSynchronize();
            LevelEngine(0).GetHeatFlux(m_tmp_flux);
            amrex::MultiFab::Saxpy(m_acc_heat, dt, m_tmp_flux, 0, 0, 1, 0);
            amrex::Gpu::streamSynchronize();
            LevelEngine(0).GetMoistureFlux(m_tmp_flux);
            amrex::MultiFab::Saxpy(m_acc_moist, dt, m_tmp_flux, 0, 0, 1, 0);
            amrex::Gpu::streamSynchronize();
            m_acc_time += dt;
        }
    }

private:
    // Total refinement of fire level `lev` relative to ERF level 0 in x (== y)
    int RatioToERF (int lev) const {
        int r = m_fire_ref_ratio;
        for (int l = 0; l < lev; ++l) { r *= refRatio(l)[0]; }
        return r;
    }

    // Copy an ERF 2D field onto fire level `lev`: ParallelCopy onto the level's layout
    // coarsened to ERF resolution, then each fire cell takes its ERF cell.
    void ToLevel (int lev, const amrex::MultiFab& src, std::unique_ptr<amrex::MultiFab>& dst) {
        const int r = RatioToERF(lev);
        amrex::BoxArray bac = boxArray(lev);
        bac.coarsen(amrex::IntVect(r, r, 1));
        amrex::MultiFab crse(bac, DistributionMap(lev), 1, 0);
        crse.ParallelCopy(src, 0, 0, 1);
        if (!dst || dst->boxArray() != boxArray(lev)) {
            dst = std::make_unique<amrex::MultiFab>(boxArray(lev), DistributionMap(lev), 1, 0, HostAccessible());
        }
        for (amrex::MFIter mfi(*dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            LaunchProlongNearest(mfi.tilebox(), r, crse.const_array(mfi), dst->array(mfi));
        }
    }

    void FillLevelAtmosphere (int lev) {
        const int nlev = finest_level + 1;
        if (static_cast<int>(m_u.size()) < nlev) {
            m_u.resize(nlev); m_v.resize(nlev); m_h.resize(nlev); m_t.resize(nlev);
            m_filled_serial.resize(nlev, -1);
        }
        if (m_filled_serial[lev] == m_atm_serial) { return; }   // wind fixed until next SetAtmosphere
        ToLevel(lev, *m_src_u, m_u[lev]);
        ToLevel(lev, *m_src_v, m_v[lev]);
        if (m_src_h)       { ToLevel(lev, *m_src_h, m_h[lev]); }
        if (m_src_terrain) { ToLevel(lev, *m_src_terrain, m_t[lev]); }
        m_src_h_used = (m_src_h != nullptr);
        amrex::Gpu::streamSynchronize();   // device kernels -> host reads in the engine
        m_filled_serial[lev] = m_atm_serial;
    }

    // Integer field (fuel map, ignition mask) on fire level lev > 0: from a full-domain
    // level-1 field when given, else each cell takes its fire-level-0 cell.
    void MakeLevelIntField (int lev, const amrex::iMultiFab* l0, const amrex::IArrayBox* full,
                            amrex::Vector<std::unique_ptr<amrex::iMultiFab>>& out)
    {
        if (static_cast<int>(out.size()) <= lev) { out.resize(lev + 1); }
        if (out[lev] && out[lev]->boxArray() == boxArray(lev)) { return; }
        out[lev] = std::make_unique<amrex::iMultiFab>(boxArray(lev), DistributionMap(lev), 1, 0, HostAccessible());
        if (full != nullptr) {
            AMREX_ALWAYS_ASSERT(lev == 1);
            const amrex::Box& dom = Geom(lev).Domain();
            const amrex::Box& fb = full->box();
            if (fb.length(0) != dom.length(0) || fb.length(1) != dom.length(1)) {
                amrex::Abort("Fire level-1 NetCDF field dimensions do not match the fire level-1 domain");
            }
            for (amrex::MFIter mfi(*out[lev]); mfi.isValid(); ++mfi) {
                LaunchFillFromNCKernel(mfi.validbox(), out[lev]->array(mfi), full->const_array(),
                                       fb.smallEnd(0), fb.bigEnd(0), fb.smallEnd(1), fb.bigEnd(1));
            }
        } else {
            const int r = refRatio(0)[0];
            amrex::BoxArray bac = boxArray(lev);
            bac.coarsen(amrex::IntVect(r, r, 1));
            amrex::iMultiFab crse(bac, DistributionMap(lev), 1, 0);
            crse.ParallelCopy(*l0, 0, 0, 1);
            for (amrex::MFIter mfi(*out[lev]); mfi.isValid(); ++mfi) {
                LaunchProlongNearestInt(mfi.validbox(), r, crse.const_array(mfi), out[lev]->array(mfi));
            }
        }
        amrex::Gpu::streamSynchronize();
    }

    int m_fire_ref_ratio = 1;
    bool m_atmospheric = false;
    const amrex::MultiFab* m_src_u = nullptr;
    const amrex::MultiFab* m_src_v = nullptr;
    const amrex::MultiFab* m_src_h = nullptr;
    const amrex::MultiFab* m_src_terrain = nullptr;
    bool m_src_h_used = false;
    long m_atm_serial = 0;
    amrex::Vector<long> m_filled_serial;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_u, m_v, m_h, m_t;   // per fire level

    const amrex::iMultiFab* m_fuel_l0 = nullptr;
    const amrex::IArrayBox* m_fuel_l1_full = nullptr;
    const amrex::iMultiFab* m_ign_l0 = nullptr;
    const amrex::IArrayBox* m_ign_l1_full = nullptr;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_fuel, m_ign;      // per fire level (> 0)

    bool m_accumulating = false;
    amrex::Real m_acc_time = 0.0;
    amrex::MultiFab m_acc_heat, m_acc_moist, m_tmp_flux;
};

FireELMFIRE::FireELMFIRE() = default;
FireELMFIRE::~FireELMFIRE() = default;

ELMFIRE::FireEngine& FireELMFIRE::Engine0() const { return m_hier->LevelEngine(0); }

void FireELMFIRE::SetTime(const amrex::Real& current_time)
{
    m_current_time = current_time;
    if (m_hier) { Engine0().SetTime(current_time); }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Fire model time synchronized to t = " << current_time << " s (restart)\n";
    }
}

void FireELMFIRE::Define(SolverChoice& sc)
{
    m_heat_feedback = sc.fire_heat_feedback;

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
    if (m_use_atmos_wind_height_override) { m_config.atmos_wind_height = m_atmos_wind_height; }
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

    {
        // Fire level-0 domain: ERF's domain refined by fire_ref_ratio (>= 1) in x,y only.
        // The fire is 2D, so it is always a single z layer (also for fire_ref_ratio = 1).
        // Fire model is 2D — keep z as a single layer (k=0).
        amrex::Box coarse_domain = geom.Domain();
        amrex::Box fine_domain(
            amrex::IntVect(0, 0, 0),
            amrex::IntVect(
                coarse_domain.length(0) * m_fire_ref_ratio - 1,
                coarse_domain.length(1) * m_fire_ref_ratio - 1,
                0));

        // Fire BoxArray = ERF level-0 boxes (bottom layer) refined by (rr,rr,1), on the
        // ERF DistributionMapping. Coarsening it by (rr,rr,1) gives back exactly the ERF
        // 2D boxes, so the ERF<->fire transfers (ParallelCopy / average_down) are aligned
        // and rank-local, and the boxes stay as large as ERF's. (The previous
        // maxSize(fire_ref_ratio) made one fire box per ERF cell: correct but thousands
        // of tiny boxes, and the engine's per-node box lookup scales with box count.)
        const amrex::IntVect rr_vect(m_fire_ref_ratio, m_fire_ref_ratio, 1);
        amrex::BoxList bl_2d = m_ba.boxList();
        for (amrex::Box& b : bl_2d) { b.setSmall(2, 0); b.setBig(2, 0); }
        amrex::BoxArray ba_erf_2d(std::move(bl_2d));
        amrex::BoxArray ba_fine = ba_erf_2d;
        ba_fine.refine(rr_vect);
        amrex::DistributionMapping dm_fine = m_dm;
        m_fire_ba_fine = ba_fine;
        m_fire_dm_fine = dm_fine;

        // Build fine Geometry (same physical domain, finer cells)
        amrex::Array<int, AMREX_SPACEDIM> is_per = {0, 0, 0};
        m_geom_fine = amrex::Geometry(fine_domain, geom.ProbDomain(),
                                      geom.Coord(), is_per);

        // Allocate fine-level flux and wind MultiFabs
        m_heat_flux_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0, HostAccessible());
        m_moist_flux_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0, HostAccessible());
        m_heat_flux_fine.setVal(0.0);
        m_moist_flux_fine.setVal(0.0);
        m_U_fire_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0, HostAccessible());
        m_V_fire_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0, HostAccessible());
        m_H_fire_fine.define(m_fire_ba_fine, m_fire_dm_fine, 1, 0, HostAccessible());
        m_U_fire_fine.setVal(0.0);
        m_V_fire_fine.setVal(0.0);
        m_H_fire_fine.setVal(0.0);

        // 2D coarse BA (khi=0): coarsen fire BA in x,y by fire_ref_ratio, z stays 0.
        // Used as intermediate for 2D→2D average_down (avoids OOB z access that
        // occurs when averaging from 2D fine (khi=0) directly into 3D ERF (khi=33)).
        m_ba_crse_2d = ba_erf_2d;   // == ba_fine coarsened by (rr,rr,1)
        m_dm_crse_2d = m_dm;
        m_heat_flux_crse_2d.define(m_ba_crse_2d, m_dm_crse_2d, 1, 0, HostAccessible());
        m_moist_flux_crse_2d.define(m_ba_crse_2d, m_dm_crse_2d, 1, 0, HostAccessible());
        m_heat_flux_crse_2d.setVal(0.0);
        m_moist_flux_crse_2d.setVal(0.0);

        fire_geom_ptr = &m_geom_fine;
        fire_ba_ptr   = &m_fire_ba_fine;
        fire_dm_ptr   = &m_fire_dm_fine;

        if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "FireELMFIRE: fire_ref_ratio = " << m_fire_ref_ratio
                           << " → ELMFIRE domain: " << fine_domain << "\n"
                           << "   Cell size: " << m_geom_fine.CellSize()[0] << " m\n";
        }
    }

    // Fire-level hierarchy on the fire level-0 grid (fire levels are unrelated to ERF's
    // AMR levels). Fire levels refine x,y only; fire level 0 uses the ERF-aligned grids.
    {
        amrex::ParmParse ppa("fire_amr");
        m_fire_max_level = 0;
        ppa.query("max_level", m_fire_max_level);
        if (m_fire_max_level < 0 || m_fire_max_level > 1) {
            amrex::Abort("fire_amr.max_level must be 0 or 1 (static fire L1)");
        }
        int ref_ratio = 2, blocking_factor = 2, max_grid_size = 64, n_error_buf = 1;
        ppa.query("ref_ratio", ref_ratio);
        ppa.query("blocking_factor", blocking_factor);
        ppa.query("max_grid_size", max_grid_size);
        ppa.query("n_error_buf", n_error_buf);
        ppa.query("nc_refine_mask_file", m_nc_refine_mask_file);

        amrex::AmrInfo fire_info;
        fire_info.max_level = m_fire_max_level;
        fire_info.ref_ratio.assign(m_fire_max_level + 1, amrex::IntVect(ref_ratio, ref_ratio, 1));
        fire_info.blocking_factor.assign(m_fire_max_level + 1, amrex::IntVect(blocking_factor, blocking_factor, 1));
        fire_info.max_grid_size.assign(m_fire_max_level + 1, amrex::IntVect(max_grid_size, max_grid_size, 1));
        fire_info.n_error_buf.assign(m_fire_max_level + 1, amrex::IntVect(n_error_buf, n_error_buf, 0));
        m_hier = std::make_unique<ERFFireHierarchy>(*fire_geom_ptr, fire_info, m_config, m_fire_ref_ratio,
                                                    !m_config.use_constant_wind,
                                                    m_ignition_x, m_ignition_y,
                                                    m_ignition_radius, m_ignition_time);

        if (!m_nc_refine_mask_file.empty()) {
#ifdef ERF_USE_NETCDF
            std::vector<unsigned char> mask;
            int nx = 0, ny = 0;
            if (read_refine_mask_from_netcdf(m_nc_refine_mask_file, mask, nx, ny) == 0) {
                amrex::Abort("Failed to read refine_mask from fire_amr.nc_refine_mask_file = " + m_nc_refine_mask_file);
            }
            m_hier->SetStaticRefineMask(mask, nx, ny);   // checks size against fire level 0
#else
            amrex::Abort("fire_amr.nc_refine_mask_file requires NetCDF support (ERF_USE_NETCDF)");
#endif
        }
    }

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

    // Optional fire level-1 resolution fuel / ignition fields (full fire-L1 domain)
    const amrex::IArrayBox* fuel_l1 = nullptr;
    const amrex::IArrayBox* ign_l1  = nullptr;
    if (!m_nc_fuel_file_l1.empty() || !m_nc_ignition_file_l1.empty()) {
        if (m_fire_max_level < 1) {
            amrex::Abort("fire.nc_fuel_file_l1 / fire.nc_ignition_file_l1 require fire_amr.max_level = 1");
        }
#ifdef ERF_USE_NETCDF
        const amrex::Box dom_l1 = amrex::refine(fire_geom_ptr->Domain(), m_hier->refRatio(0));
        if (!m_nc_fuel_file_l1.empty() && m_fuel_model_source == "landfire") {
            if (read_static_fuel_from_netcdf(dom_l1, m_nc_fuel_file_l1, m_fuel_l1_full) == 0) {
                amrex::Abort("Failed to read FBFM from fire.nc_fuel_file_l1 = " + m_nc_fuel_file_l1);
            }
            fuel_l1 = &m_fuel_l1_full;
        }
        if (!m_nc_ignition_file_l1.empty() && m_ignition_type == "area") {
            if (read_ignition_from_netcdf(dom_l1, m_nc_ignition_file_l1, m_ignition_time_index, m_ignition_l1_full) == 0) {
                amrex::Abort("Failed to read fire_mask from fire.nc_ignition_file_l1 = " + m_nc_ignition_file_l1);
            }
            ign_l1 = &m_ignition_l1_full;
        }
#else
        amrex::Abort("fire.nc_fuel_file_l1 / fire.nc_ignition_file_l1 require NetCDF support (ERF_USE_NETCDF)");
#endif
    }

    // Fuel map (used with atmospheric forcing): fire L0 from fire.nc_fuel_file; fire L1
    // from fire.nc_fuel_file_l1 if given, else sampled from fire L0.
    if (m_fuel_model_source == "landfire") {
        m_hier->SetFuel(m_fuel_model_lev[0].get(), fuel_l1);
    }
    // Ignition: point (fire.ignition_x/y/radius) or area (mask; fire L1 from
    // fire.nc_ignition_file_l1 if given, else sampled from fire L0), at fire.ignition_time;
    // applied by the hierarchy at t = 0 (InitData) or when ignition_time is reached.
    if (m_ignition_type == "area") {
        m_hier->SetAreaIgnition(m_ignition_mask_lev[0].get(), ign_l1);
    }
    m_hier->InitData(*fire_ba_ptr, *fire_dm_ptr);

    m_initialized = true;

    if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "FireELMFIRE initialized at level " << lev << "\n";
        amrex::Print() << "   Mode: " << m_fire_mode << "\n";
        amrex::Print() << "   Domain: " << geom.Domain() << "\n";
        amrex::Print() << "   Cell size: " << geom.CellSize()[0] << " m\n";
    }
}

void FireELMFIRE::Advance(const amrex::Real& dt,
                         const amrex::MultiFab* U,
                         const amrex::MultiFab* V,
                         const amrex::MultiFab* terrain,
                         const amrex::MultiFab* wind_height_in)
{
    if (!m_initialized) {
        amrex::Abort("FireELMFIRE::Advance() called before Init()!");
    }

    // Track atmospheric time for subcycling
    m_accumulated_dt += dt;
    m_atmo_step_count++;

    // Ignition (point or area, at fire.ignition_time) is applied by the fire hierarchy.

    // Update current time (always advances, even if fire doesn't update)
    m_current_time += dt;

    // Check if it's time to update fire (subcycling)
    bool update_fire = (m_atmo_step_count % m_subcycle_interval == 0);

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

    // Fuel maps (LANDFIRE) were given to the hierarchy at Init (SetFuel).

    // Advance every fire level to the ERF time. In atmospheric mode each fire level takes
    // the ERF wind / height / terrain directly from the ERF grid, held fixed over the
    // interval. With heat feedback the fire-level-0 fluxes are time-averaged over it.
    if (m_config.use_constant_wind) {
        if (m_verbosity > 0 && m_current_time < 1.0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "FireELMFIRE: Using CONSTANT WIND mode\n";
        }
    } else {
        if (m_verbosity > 0 && m_current_time < 1.0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "FireELMFIRE: Using ATMOSPHERIC COUPLING mode\n";
        }
        if (U == nullptr || V == nullptr) {
            amrex::Abort("FireELMFIRE: atmospheric mode requires the ERF surface wind");
        }
        // fire.atmos_wind_height (fixed height) overrides ERF's per-column height
        m_hier->SetAtmosphere(U, V, m_use_atmos_wind_height_override ? nullptr : wind_height_in, terrain);
    }

    if (m_heat_feedback) { m_hier->BeginFluxAverage(); }
    m_hier->AdvanceTo(m_current_time);
    if (m_heat_feedback) {
        m_hier->EndFluxAverage(m_heat_flux_fine, m_moist_flux_fine);
        if (m_verbosity > 1) {   // sum() is an MPI reduction: all ranks
            const auto dxf = m_geom_fine.CellSizeArray();
            const amrex::Real released = m_heat_flux_fine.sum(0) * dxf[0] * dxf[1] * fire_dt;
            amrex::Print() << "Fire: heat released by fire level 0 over the last " << fire_dt
                           << " s = " << std::setprecision(12) << released << " J" << std::endl;
        }
    }

    if (!m_config.use_constant_wind) {
        // Fire-level-0 wind (and height) for the fire plotfile (x_velocity / y_velocity / wind_height)
        if (m_hier->LevelWindU(0)) {
            amrex::MultiFab::Copy(m_U_fire_fine, *m_hier->LevelWindU(0), 0, 0, 1, 0);
            amrex::MultiFab::Copy(m_V_fire_fine, *m_hier->LevelWindV(0), 0, 0, 1, 0);
        }
        if (m_hier->LevelWindHeight(0)) {
            amrex::MultiFab::Copy(m_H_fire_fine, *m_hier->LevelWindHeight(0), 0, 0, 1, 0);
        }
        m_hier->ClearAtmosphere();
    }

    // Diagnostics: GetFireArea/GetNarrowBandSize are MPI reductions, so every rank
    // must call them; amrex::Print only writes on the IO rank.
    if (m_verbosity > 1) {
        amrex::Real fire_area = m_hier->GetCompositeBurnedArea();
        int narrow_band_size = Engine0().GetNarrowBandSize();
        amrex::Print() << "   Fire: t = " << m_current_time << " s, "
                       << "area = " << fire_area << " m², "
                       << "active cells = " << narrow_band_size << "\n";
    }
}

void FireELMFIRE::GetHeatFlux(amrex::MultiFab& heat_flux)
{
    amrex::Gpu::streamSynchronize();  // pending device work on these MultiFabs before host writes
    if (!m_initialized) {
        heat_flux.setVal(0.0);
        return;
    }

    {
        // Time-averaged fire-level-0 heat flux, averaged down to the ERF grid.
        // Step 1: 2D→2D average_down with IntVect(rr,rr,1) — z stays at k=0.
        //   Using IntVect(rr) (same for all dims) would loop kk=0..rr-1 in z,
        //   accessing fire FABs at kk>0 which are out-of-bounds (fire is 2D, khi=0).
        // Step 2: ParallelCopy the 2D coarse result into k=0 of the 3D ERF MF.
        // m_heat_flux_fine = fire-level-0 heat flux averaged over the last fire update interval
        m_heat_flux_crse_2d.setVal(0.0);
        amrex::average_down(m_heat_flux_fine, m_heat_flux_crse_2d,
                            0, 1, amrex::IntVect(m_fire_ref_ratio, m_fire_ref_ratio, 1));
        heat_flux.setVal(0.0);
        heat_flux.ParallelCopy(m_heat_flux_crse_2d, 0, 0, 1);
    }
}

void FireELMFIRE::GetMoistureFlux(amrex::MultiFab& moisture_flux)
{
    amrex::Gpu::streamSynchronize();  // pending device work on these MultiFabs before host writes
    if (!m_initialized) {
        moisture_flux.setVal(0.0);
        return;
    }

    {
        // Same 2D→2D → ParallelCopy pattern as GetHeatFlux (see comment there).
        // m_moist_flux_fine = fire-level-0 vapor flux averaged over the last fire update interval
        m_moist_flux_crse_2d.setVal(0.0);
        amrex::average_down(m_moist_flux_fine, m_moist_flux_crse_2d,
                            0, 1, amrex::IntVect(m_fire_ref_ratio, m_fire_ref_ratio, 1));
        moisture_flux.setVal(0.0);
        moisture_flux.ParallelCopy(m_moist_flux_crse_2d, 0, 0, 1);
    }
}

amrex::Real FireELMFIRE::get_fire_area() const
{
    if (!m_initialized) {
        return 0.0;
    }

    return m_hier->GetCompositeBurnedArea();
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
    amrex::Gpu::streamSynchronize();  // pending device work on mf before host writes
    if (!m_initialized) {
        mf.setVal(0.0);
        return;
    }

    // The fire lives on its own 2D grid (fire_ref_ratio >= 1 times finer than ERF): every fire
    // variable lives on the 2D fire grid and must be averaged to the ERF grid:
    //   1. fill on the fire grid (fluxes via the flux getters, others via FillPlotVar)
    //   2. 2D->2D average_down with IntVect(rr,rr,1) into a temp 2D coarse MF
    //   3. ParallelCopy the 2D coarse result into k=0 of the 3D ERF plot MF
    // (IntVect(rr) in all 3 dims would read fire FABs at kk>0, out of bounds since
    //  fire khi=0; filling the ERF-resolution MF directly mismatches the grids.)
    {
        amrex::MultiFab var_fine(m_fire_ba_fine, m_fire_dm_fine, 1, 0, HostAccessible());
        if (name == "fire_heat_flux") {
            Engine0().GetHeatFlux(var_fine);
        } else if (name == "fire_moisture_flux") {
            Engine0().GetMoistureFlux(var_fine);
        } else {
            Engine0().FillPlotVar(name, var_fine);
        }
        amrex::MultiFab var_crse_2d(m_ba_crse_2d, m_dm_crse_2d, 1, 0);
        var_crse_2d.setVal(0.0);
        amrex::average_down(var_fine, var_crse_2d,
                            0, 1, amrex::IntVect(m_fire_ref_ratio, m_fire_ref_ratio, 1));
        mf.setVal(0.0);
        mf.ParallelCopy(var_crse_2d, 0, 0, 1);
    }
}

// ============================================
// HELPER FUNCTIONS
// ============================================

void FireELMFIRE::LoadConfigFromSolverChoice(SolverChoice& sc)
{
    // Read fire model parameters from ERF input file using ParmParse
    amrex::ParmParse pp("fire");

    // Fire mode: erf.fire_met_source is authoritative; fire.mode is derived from it
    // and, if given explicitly, must agree (a mismatch would extract ERF winds that
    // the engine then ignores, or vice versa).
    const std::string derived_mode =
        (sc.fire_met_source == FireMetSource::Atmosphere) ? "atmospheric" : "constant_wind";
    m_fire_mode = derived_mode;
    std::string user_mode;
    if (pp.query("mode", user_mode) && user_mode != derived_mode) {
        amrex::Abort("fire.mode = '" + user_mode + "' conflicts with erf.fire_met_source = '" +
                     amrex::getEnumNameString(sc.fire_met_source) + "' (expects fire.mode = '" +
                     derived_mode + "' or omit fire.mode)");
    }

    // Misspelled parameters that older input templates used; they were silently ignored
    if (pp.contains("bandthickness")) {
        amrex::Abort("fire.bandthickness is not a parameter; use fire.band_thickness");
    }
    if (pp.contains("fire_dt")) {
        amrex::Abort("fire.fire_dt is not a parameter; the fire advances with the ERF step "
                     "(see fire.subcycle_interval and fire.cfl)");
    }

    // Fire-level plotfile prefix (only used when fire_ref_ratio > 1)
    pp.query("fire_plot_file", m_fire_plot_file);

    // Fire plotfile variable list (e.g. fire.fire_plot_vars_1 = x_velocity y_velocity fire_phi)
    // If absent, WriteFirePlotFile falls back to the engine's default variable set.
    pp.queryarr("fire_plot_vars_1", m_fire_plot_vars);

    // Fuel model source ("uniform" or "landfire")
    pp.query("fuel_model_source", m_fuel_model_source);
    if (m_fuel_model_source != "uniform" &&
        m_fuel_model_source != "landfire") {
        amrex::Abort("fire.fuel_model_source must be 'uniform' or 'landfire', got: " + m_fuel_model_source);
    }

    // LANDFIRE NetCDF file path (fire namespace)
    pp.query("nc_fuel_file", m_nc_fuel_file);
    pp.query("nc_fuel_file_l1", m_nc_fuel_file_l1);           // optional, fire level-1 resolution
    pp.query("nc_ignition_file_l1", m_nc_ignition_file_l1);   // optional, fire level-1 resolution

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
    // Height above ground of the atmospheric wind. Default: the actual height of the ERF
    // level used (per column); set this only to impose a fixed height (e.g. idealized tests).
    m_use_atmos_wind_height_override = pp.query("atmos_wind_height", m_atmos_wind_height);

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
        Print() << "Fire: Loading LANDFIRE fuel data from " << m_nc_fuel_file << "\n";
    }

    // Allocate fuel data storage for this level
    m_fuel_model_lev.resize(1);  // Single level for now
    IntVect ngv(0, 0, 0);  // No ghost cells for static data
    m_fuel_model_lev[0] = std::make_unique<iMultiFab>(ba, dm, 1, ngv, HostAccessible());

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
    amrex::Gpu::streamSynchronize();  // fill kernels -> host reads in the engine

    if (m_verbosity > 0 && ParallelDescriptor::IOProcessor()) {
        Print() << "Fire: LANDFIRE fuel data loaded successfully\n";
    }
}

void FireELMFIRE::load_ignition_data(const amrex::Geometry& geom,
                                     const amrex::BoxArray& ba,
                                     const amrex::DistributionMapping& dm)
{
    using namespace amrex;

    if (m_verbosity > 0 && ParallelDescriptor::IOProcessor()) {
        Print() << "Fire: Loading ignition mask from " << m_nc_ignition_file << "\n";
    }

    // Allocate ignition mask storage for this level
    m_ignition_mask_lev.resize(1);  // Single level for now
    IntVect ngv(0, 0, 0);  // No ghost cells for static data
    m_ignition_mask_lev[0] = std::make_unique<iMultiFab>(ba, dm, 1, ngv, HostAccessible());

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
    amrex::Gpu::streamSynchronize();  // fill kernels -> host reads in the engine

    if (m_verbosity > 0 && ParallelDescriptor::IOProcessor()) {
        Print() << "Fire: Ignition mask loaded successfully (" << total_ignited << " cells to ignite)\n";
    }
}
#endif

void FireELMFIRE::PrintConfig() const
{
    amrex::Print() << "  Fire mode: " << m_fire_mode << "\n";
    amrex::Print() << "  Fuel model source: " << m_fuel_model_source << "\n";

    if (m_fuel_model_source == "uniform") {
        amrex::Print() << "    Uniform fuel ID: " << m_uniform_fuel_id << "\n";
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
    if (!m_initialized || !m_hier) return;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Writing fire checkpoint data at level " << lev << "\n";
    }

    // Write fire phi (level set) - this is the essential fire state
    amrex::MultiFab phi_out(m_ba, m_dm, 1, 0, HostAccessible());
    Engine0().GetPhiField(phi_out);
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
    if (!m_initialized || !m_hier) return;

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
    amrex::MultiFab phi_restart(m_ba, m_dm, 1, 1, HostAccessible());
    amrex::VisMF::Read(phi_restart, chk_dir + "/FirePhi");
    Engine0().RestorePhiFromCheckpoint(phi_restart);

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
    if (!m_initialized || !m_hier) return;

    // With a fire L1: the hierarchy's multi-level plotfile (all fire levels, composite
    // mask, arrival time), same format as LSFIRE elmfire_amr output.
    if (m_fire_max_level > 0) {
        const std::string pltname = amrex::Concatenate(m_fire_plot_file, step, 7);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Fire: writing multi-level fire plotfile " << pltname << " (t = " << time << " s)\n";
        }
        m_hier->WritePlotFile(pltname);
        return;
    }

    // Single fire level: only when the fire runs at finer resolution than ERF
    if (m_fire_ref_ratio <= 1) return;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Fire: WriteFirePlotFile: step=" << step << " time=" << time << "\n";
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
    amrex::MultiFab mf(m_fire_ba_fine, m_fire_dm_fine, ncomp, 0, HostAccessible());
    mf.setVal(0.0);

    for (int c = 0; c < ncomp; ++c) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  FillPlotVar: " << var_names[c] << "\n";
        }
        amrex::MultiFab tmp(m_fire_ba_fine, m_fire_dm_fine, 1, 0, HostAccessible());
        tmp.setVal(0.0);
        amrex::Gpu::streamSynchronize();  // setVal kernel before host writes in FillPlotVar

        if (var_names[c] == "x_velocity") {
            // Atmospheric U wind prolongated to fire resolution
            amrex::MultiFab::Copy(tmp, m_U_fire_fine, 0, 0, 1, 0);
        } else if (var_names[c] == "wind_height") {
            amrex::MultiFab::Copy(tmp, m_H_fire_fine, 0, 0, 1, 0);
        } else if (var_names[c] == "y_velocity") {
            // Atmospheric V wind prolongated to fire resolution
            amrex::MultiFab::Copy(tmp, m_V_fire_fine, 0, 0, 1, 0);
        } else {
            Engine0().FillPlotVar(var_names[c], tmp);
            amrex::Gpu::streamSynchronize();
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
