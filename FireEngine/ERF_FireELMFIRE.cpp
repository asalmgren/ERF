#include "ERF_FireELMFIRE.H"
#include "Core/FireHierarchy.H"
#include <ERF.H>
#include <ERF_Constants.H>
#include <ERF_FireEngineRegistry.H>
#include <ERF_FireUtils.H>
#include <ERF_FireWindExtract.H>
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

// ============================================
// FireELMFIRE: the FireEngine behind ERF's fire coupling
// ============================================

FireELMFIRE::FireELMFIRE (const FireCouplingParams& coupling)
    : m_coupling(coupling)
{
    read_inputs();
    if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n========================================\n"
                       << "LSFIRE (ELMFIRE) fire engine configuration\n"
                       << "========================================\n";
        print_config();
        amrex::Print() << "========================================\n\n";
    }
}

FireELMFIRE::~FireELMFIRE () = default;

ELMFIRE::FireEngine& FireELMFIRE::Engine0 () const { return m_hier->LevelEngine(0); }

void
FireELMFIRE::verify_prerequisites (const ERF& erf, int lev)
{
    // LSFIRE couples to ERF level 0 only; its own fire levels (fire_amr.*) refine the fire grid
    if (lev != 0 || erf.maxLevel() != 0) {
        amrex::Abort("[FIRE] erf.fire.engine = lsfire couples the fire to ERF level 0 only: set amr.max_level = 0"
                     " (and leave erf.fire.anchor_level unset); refine the fire grid with"
                     " erf.fire.grid_ratio and fire_amr.* instead");
    }
    if (m_fire_mode == "atmospheric" && m_coupling.is_synchronous() && m_subcycle_interval > 1) {
        amrex::Print() << "[FIRE] WARNING: lsfire with fire.subcycle_interval > 1 holds the wind of the"
                       << " update step for the whole interval\n";
    }
}

void
FireELMFIRE::initialize (const ERF& erf, int lev, const FireGrid& fg,
                         const SurfaceLayer* /*surface_layer*/, const amrex::MultiFab* z_phys_nd_atm)
{
    // ---- ERF level-0 k = 0 slabs, for the wind and terrain handed to the fire ----
    m_geom_atm  = erf.Geom(lev);
    m_dm_atm    = erf.DistributionMap(lev);
    m_z_phys_nd = z_phys_nd_atm;
    {
        const int klo = m_geom_atm.Domain().smallEnd(2);
        amrex::BoxList bl = erf.boxArray(lev).boxList();
        for (amrex::Box& b : bl) { b.setSmall(2, klo); b.setBig(2, klo); }
        m_ba_atm_2d = amrex::BoxArray(std::move(bl));
    }
    m_u_k0.define(m_ba_atm_2d, m_dm_atm, 1, 0, HostAccessible());
    m_v_k0.define(m_ba_atm_2d, m_dm_atm, 1, 0, HostAccessible());
    m_h_k0.define(m_ba_atm_2d, m_dm_atm, 1, 0, HostAccessible());
    m_terrain_k0.define(m_ba_atm_2d, m_dm_atm, 1, 0, HostAccessible());

    // ---- The fire grid: fire level 0 of the LSFIRE hierarchy ----
    // The coupler's FireGrid is the level-0 boxes at k = 0 refined by
    // erf.fire.grid_ratio on the same ranks, as LSFIRE's fire_ref_ratio grid was.
    // LSFIRE's fire grid has no periodic wrap.
    m_ba_fire = fg.ba;
    m_dm_fire = fg.dm;
    {
        amrex::Array<int, AMREX_SPACEDIM> is_per = {0, 0, 0};
        m_geom_fire = amrex::Geometry(fg.geom.Domain(), fg.geom.ProbDomain(), fg.geom.Coord(), is_per);
    }
    for (amrex::MultiFab* mf : {&m_heat_flux_fire, &m_moist_flux_fire, &m_latent_flux_fire,
                                &m_U_fire, &m_V_fire, &m_H_fire}) {
        mf->define(m_ba_fire, m_dm_fire, 1, 0, HostAccessible());
        mf->setVal(0.0);
    }

    // ---- ELMFIRE configuration ----
    m_config.cfl                = m_cfl;
    m_config.band_thickness     = m_band_thickness;
    m_config.fuel_model_source  = m_fuel_model_source;
    m_config.uniform_fuel_id    = m_uniform_fuel_id;
    m_config.fuel_database_path = m_fuel_database_path;
    if (m_fire_mode == "constant_wind") {
        m_config.use_constant_wind = true;
        m_config.wind_u      = m_constant_wind_u;
        m_config.wind_v      = m_constant_wind_v;
        m_config.wind_height = m_wind_height;
    } else {
        m_config.use_constant_wind = false;
    }
    m_config.m1   = m_fuel_moisture_1hr;
    m_config.m10  = m_fuel_moisture_10hr;
    m_config.m100 = m_fuel_moisture_100hr;
    m_config.mlh  = m_fuel_moisture_live_herb;
    m_config.mlw  = m_fuel_moisture_live_woody;
    m_config.slope_x = m_uniform_slope_x;
    m_config.slope_y = m_uniform_slope_y;
    m_config.wind_data_scale_factor = m_wind_data_scale_factor;
    if (m_use_atmos_wind_height_override) { m_config.atmos_wind_height = m_atmos_wind_height; }
    m_config.use_grid_aware_burnout  = m_use_grid_aware_burnout;
    m_config.use_source_fuel_scaling = m_use_source_fuel_scaling;
    m_config.source_fuel_exponent    = m_source_fuel_exponent;
    m_config.terrain_scale_factor    = m_terrain_scale_factor;
    m_config.ignition_x      = m_ignition_x;
    m_config.ignition_y      = m_ignition_y;
    m_config.ignition_radius = m_ignition_radius;
    m_config.ignition_time   = m_ignition_time;

    // ---- Fire-level hierarchy (fire levels are unrelated to ERF's AMR levels) ----
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
        m_hier = std::make_unique<ERFFireHierarchy>(m_geom_fire, fire_info, m_config, fg.C,
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
    if (m_fuel_model_source == "landfire") { load_fuel_data(m_geom_fire, m_ba_fire, m_dm_fire); }
    if (m_ignition_type == "area") {
        if (m_nc_ignition_file.empty()) { amrex::Abort("fire.ignition_type = area requires fire.nc_ignition_file"); }
        load_ignition_data(m_geom_fire, m_ba_fire, m_dm_fire);
    }
#else
    if (m_fuel_model_source == "landfire") {
        amrex::Abort("fire.fuel_model_source = landfire requires NetCDF support (ERF_USE_NETCDF)");
    }
    if (m_ignition_type == "area") {
        amrex::Abort("fire.ignition_type = area requires NetCDF support (ERF_USE_NETCDF)");
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
        const amrex::Box dom_l1 = amrex::refine(m_geom_fire.Domain(), m_hier->refRatio(0));
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
    if (m_fuel_model_source == "landfire") { m_hier->SetFuel(m_fuel_model_lev[0].get(), fuel_l1); }
    if (m_ignition_type == "area")         { m_hier->SetAreaIgnition(m_ignition_mask_lev[0].get(), ign_l1); }
    m_hier->InitData(m_ba_fire, m_dm_fire);

    m_initialized = true;
    if (m_verbosity > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[FIRE] lsfire initialized: mode " << m_fire_mode
                       << ", fire grid " << m_geom_fire.Domain()
                       << ", cell " << m_geom_fire.CellSize()[0] << " m\n";
    }
}

void
FireELMFIRE::advance (const FireForcing& forcing)
{
    if (!m_initialized) { amrex::Abort("[FIRE] lsfire: advance() called before initialize()"); }

    m_accumulated_dt += forcing.dt;
    m_atmo_step_count++;
    m_current_time = forcing.time + forcing.dt;

    // fire.subcycle_interval: the fire updates every N ERF steps over the time
    // accumulated since its last update. On the steps between, the fluxes of the
    // last update stay in place (FireFluxes asks for a mean over the ERF step;
    // with N > 1 the update's mean is handed over for the N steps after it).
    if (m_atmo_step_count % m_subcycle_interval != 0) {
        if (m_verbosity > 1) {
            amrex::Print() << "  [Fire subcycling] no fire update this step, accumulated dt = "
                           << m_accumulated_dt << " s\n";
        }
        return;
    }
    const amrex::Real fire_dt = m_accumulated_dt;
    m_accumulated_dt = 0.0;

    if (!m_config.use_constant_wind) {
        // The wind of the lowest ERF cell and its height above that column's
        // ground, and the cell-centre height of that cell as the terrain
        extract_fire_wind_k0(m_u_k0, m_v_k0, m_h_k0, forcing.xvel, forcing.yvel,
                             forcing.z_phys_cc, m_z_phys_nd, m_geom_atm);
        fire_copy_k0_plane(m_terrain_k0, forcing.z_phys_cc);
        amrex::Gpu::streamSynchronize();   // device kernels -> host reads in the engine
        // fire.atmos_wind_height (a fixed height) overrides the per-column height
        m_hier->SetAtmosphere(&m_u_k0, &m_v_k0, m_use_atmos_wind_height_override ? nullptr : &m_h_k0,
                              &m_terrain_k0);
    }

    // The fire-level-0 fluxes, averaged over every fire step of the update
    m_hier->BeginFluxAverage();
    m_hier->AdvanceTo(m_current_time);
    m_hier->EndFluxAverage(m_heat_flux_fire, m_moist_flux_fire);

    // The coupling takes the latent heat flux [W/m2]: the vapour mass flux
    // times ERF's L_v, which the coupling divides by again (exact round trip)
    amrex::MultiFab::Copy(m_latent_flux_fire, m_moist_flux_fire, 0, 0, 1, 0);
    m_latent_flux_fire.mult(L_v);
    amrex::Gpu::streamSynchronize();

    if (m_verbosity > 1) {   // sum() is an MPI reduction: all ranks
        const auto dxf = m_geom_fire.CellSizeArray();
        const amrex::Real released = m_heat_flux_fire.sum(0) * dxf[0] * dxf[1] * fire_dt;
        amrex::Print() << "Fire: heat released by fire level 0 over the last " << fire_dt
                       << " s = " << std::setprecision(12) << released << " J" << std::endl;
    }

    if (!m_config.use_constant_wind) {
        // Fire-level-0 wind (and height) for the fire plotfile
        if (m_hier->LevelWindU(0)) {
            amrex::MultiFab::Copy(m_U_fire, *m_hier->LevelWindU(0), 0, 0, 1, 0);
            amrex::MultiFab::Copy(m_V_fire, *m_hier->LevelWindV(0), 0, 0, 1, 0);
        }
        if (m_hier->LevelWindHeight(0)) {
            amrex::MultiFab::Copy(m_H_fire, *m_hier->LevelWindHeight(0), 0, 0, 1, 0);
        }
        m_hier->ClearAtmosphere();
    }

    if (m_verbosity > 1) {   // MPI reductions: every rank calls them
        const amrex::Real fire_area = m_hier->GetCompositeBurnedArea();
        const int narrow_band_size  = Engine0().GetNarrowBandSize();
        amrex::Print() << "   Fire: t = " << m_current_time << " s, area = " << fire_area
                       << " m2, active cells = " << narrow_band_size << "\n";
    }
}

void
FireELMFIRE::fill_fluxes (FireFluxes& fluxes)
{
    fluxes.sensible = &m_heat_flux_fire;
    fluxes.latent   = &m_latent_flux_fire;
}

amrex::Real
FireELMFIRE::burned_area () const
{
    return m_initialized ? m_hier->GetCompositeBurnedArea() : amrex::Real(0.0);
}

std::vector<std::string>
FireELMFIRE::plot_var_names () const
{
    // Known before initialize(): ERF asks while it reads erf.plot_vars_*
    std::vector<std::string> names;
    for (const auto& n : ELMFIRE::FireEngine::GetPlotVarNames()) { names.push_back(n); }
    return names;
}

const amrex::MultiFab*
FireELMFIRE::plot_var (const std::string& name) const
{
    if (!m_initialized) { return nullptr; }
    auto& mf = m_plot_cache[name];
    if (!mf) { mf = std::make_unique<amrex::MultiFab>(m_ba_fire, m_dm_fire, 1, 0, HostAccessible()); }
    mf->setVal(0.0);
    amrex::Gpu::streamSynchronize();   // setVal kernel before host writes in the engine
    if (name == "fire_heat_flux") {
        amrex::MultiFab::Copy(*mf, m_heat_flux_fire, 0, 0, 1, 0);       // the flux handed to ERF
    } else if (name == "fire_moisture_flux") {
        amrex::MultiFab::Copy(*mf, m_moist_flux_fire, 0, 0, 1, 0);
    } else {
        Engine0().FillPlotVar(name, *mf);
    }
    amrex::Gpu::streamSynchronize();
    return mf.get();
}

void
FireELMFIRE::write_plotfile (const std::string& plotfile_prefix, amrex::Real time, int step) const
{
    if (!m_initialized || !m_hier) { return; }

    // With a fire L1: the hierarchy's multi-level plotfile (all fire levels,
    // composite mask, arrival time), the LSFIRE elmfire_amr format
    if (m_fire_max_level > 0) {
        const std::string pltname = amrex::Concatenate(plotfile_prefix, step, 7);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Fire: writing multi-level fire plotfile " << pltname << " (t = " << time << " s)\n";
        }
        m_hier->WritePlotFile(pltname);
        return;
    }

    // Single fire level: fire.fire_plot_vars_1, or the engine's default set
    amrex::Vector<std::string> var_names;
    if (!m_fire_plot_vars.empty()) {
        var_names = m_fire_plot_vars;
    } else {
        for (const auto& n : ELMFIRE::FireEngine::GetPlotVarNames()) { var_names.push_back(n); }
    }
    const int ncomp = static_cast<int>(var_names.size());
    amrex::MultiFab mf(m_ba_fire, m_dm_fire, ncomp, 0, HostAccessible());
    mf.setVal(0.0);
    for (int c = 0; c < ncomp; ++c) {
        amrex::MultiFab tmp(m_ba_fire, m_dm_fire, 1, 0, HostAccessible());
        tmp.setVal(0.0);
        amrex::Gpu::streamSynchronize();   // setVal kernel before host writes in FillPlotVar
        if (var_names[c] == "x_velocity") {
            amrex::MultiFab::Copy(tmp, m_U_fire, 0, 0, 1, 0);
        } else if (var_names[c] == "y_velocity") {
            amrex::MultiFab::Copy(tmp, m_V_fire, 0, 0, 1, 0);
        } else if (var_names[c] == "wind_height") {
            amrex::MultiFab::Copy(tmp, m_H_fire, 0, 0, 1, 0);
        } else if (var_names[c] == "fire_heat_flux") {
            amrex::MultiFab::Copy(tmp, m_heat_flux_fire, 0, 0, 1, 0);
        } else if (var_names[c] == "fire_moisture_flux") {
            amrex::MultiFab::Copy(tmp, m_moist_flux_fire, 0, 0, 1, 0);
        } else {
            Engine0().FillPlotVar(var_names[c], tmp);
            amrex::Gpu::streamSynchronize();
        }
        amrex::MultiFab::Copy(mf, tmp, 0, c, 1, 0);
    }
    const std::string pltname = amrex::Concatenate(plotfile_prefix, step, 7);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Fire: writing fire plotfile " << pltname << " (fire grid " << m_geom_fire.Domain() << ")\n";
    }
    amrex::WriteSingleLevelPlotfile(pltname, mf, var_names, m_geom_fire, time, step);
}

// ============================================
// INPUTS
// ============================================

void
FireELMFIRE::read_inputs ()
{
    // Keys of the in-tree LSFIRE coupling that the shared ERF coupling replaced:
    // stop with the new key instead of running without them
    {
        amrex::ParmParse pp_erf("erf");
        const std::vector<std::pair<std::string, std::string>> moved_erf = {
            {"fire_type",          "erf.fire.enable = 1 and erf.fire.engine = lsfire"},
            {"fire_met_source",    "fire.mode = atmospheric (Atmosphere) or constant_wind (Constant)"},
            {"fire_heat_feedback", "erf.fire.coupling_type = lagged (true) or passive (false)"},
        };
        for (const auto& [key, repl] : moved_erf) {
            if (pp_erf.contains(key.c_str())) {
                amrex::Abort("[FIRE] erf." + key + " is replaced by " + repl);
            }
        }
    }
    amrex::ParmParse pp("fire");
    if (pp.contains("fire_ref_ratio")) {
        amrex::Abort("[FIRE] fire.fire_ref_ratio is replaced by erf.fire.grid_ratio");
    }
    if (pp.contains("fire_plot_file")) {
        amrex::Abort("[FIRE] fire.fire_plot_file is replaced by erf.fire.plot_file");
    }
    // Misspelled parameters that older input templates used; they were silently ignored
    if (pp.contains("bandthickness")) {
        amrex::Abort("fire.bandthickness is not a parameter; use fire.band_thickness");
    }
    if (pp.contains("fire_dt")) {
        amrex::Abort("fire.fire_dt is not a parameter; the fire advances with the ERF step "
                     "(see fire.subcycle_interval and fire.cfl)");
    }

    // Wind source: constant_wind (fire.constant_wind_u/v) or atmospheric (the ERF k = 0 wind)
    pp.query("mode", m_fire_mode);
    if (m_fire_mode != "constant_wind" && m_fire_mode != "atmospheric") {
        amrex::Abort("fire.mode must be 'constant_wind' or 'atmospheric', got: " + m_fire_mode);
    }

    pp.queryarr("fire_plot_vars_1", m_fire_plot_vars);

    pp.query("fuel_model_source", m_fuel_model_source);
    if (m_fuel_model_source != "uniform" && m_fuel_model_source != "landfire") {
        amrex::Abort("fire.fuel_model_source must be 'uniform' or 'landfire', got: " + m_fuel_model_source);
    }
    pp.query("nc_fuel_file", m_nc_fuel_file);
    pp.query("nc_fuel_file_l1", m_nc_fuel_file_l1);
    pp.query("nc_ignition_file_l1", m_nc_ignition_file_l1);
    if (m_fuel_model_source == "landfire" && m_nc_fuel_file.empty()) {
        amrex::Abort("fire.fuel_model_source='landfire' requires fire.nc_fuel_file to be specified!");
    }
    pp.query("uniform_fuel_id", m_uniform_fuel_id);
    pp.query("fuel_database_path", m_fuel_database_path);

    pp.query("constant_wind_u", m_constant_wind_u);
    pp.query("constant_wind_v", m_constant_wind_v);
    pp.query("wind_height", m_wind_height);
    // Height above ground of the atmospheric wind. Default: the actual height of
    // the ERF k = 0 cell centre (per column); set only to impose a fixed height.
    m_use_atmos_wind_height_override = pp.query("atmos_wind_height", m_atmos_wind_height);

    pp.query("fuel_moisture_1hr", m_fuel_moisture_1hr);
    pp.query("fuel_moisture_10hr", m_fuel_moisture_10hr);
    pp.query("fuel_moisture_100hr", m_fuel_moisture_100hr);
    pp.query("fuel_moisture_live_herb", m_fuel_moisture_live_herb);
    pp.query("fuel_moisture_live_woody", m_fuel_moisture_live_woody);

    pp.query("uniform_slope_x", m_uniform_slope_x);
    pp.query("uniform_slope_y", m_uniform_slope_y);

    pp.query("ignition_type", m_ignition_type);
    pp.query("ignition_x", m_ignition_x);
    pp.query("ignition_y", m_ignition_y);
    pp.query("ignition_radius", m_ignition_radius);
    pp.query("ignition_time", m_ignition_time);
    pp.query("nc_ignition_file", m_nc_ignition_file);
    pp.query("ignition_time_index", m_ignition_time_index);

    pp.query("cfl", m_cfl);
    pp.query("band_thickness", m_band_thickness);
    pp.query("wind_data_scale_factor", m_wind_data_scale_factor);

    pp.query("use_grid_aware_burnout",  m_use_grid_aware_burnout);
    pp.query("use_source_fuel_scaling", m_use_source_fuel_scaling);
    pp.query("source_fuel_exponent",    m_source_fuel_exponent);
    pp.query("terrain_scale_factor",    m_terrain_scale_factor);

    pp.query("subcycle_interval", m_subcycle_interval);
    if (m_subcycle_interval < 1) {
        amrex::Abort("fire.subcycle_interval must be >= 1, got " + std::to_string(m_subcycle_interval));
    }
    pp.query("smoke_heat_per_kg", m_smoke_heat_per_kg);
    if (!(m_smoke_heat_per_kg > 0.0)) {
        amrex::Abort("fire.smoke_heat_per_kg must be > 0");
    }
    pp.query("verbosity", m_verbosity);
}

void
FireELMFIRE::print_config () const
{
    amrex::Print() << "  Fire mode: " << m_fire_mode << "\n";
    amrex::Print() << "  Fire grid ratio (erf.fire.grid_ratio): " << m_coupling.grid_ratio << "\n";
    amrex::Print() << "  Coupling type (erf.fire.coupling_type): " << m_coupling.coupling_type << "\n";
    amrex::Print() << "  Fuel model source: " << m_fuel_model_source << "\n";
    if (m_fuel_model_source == "uniform") {
        amrex::Print() << "    Uniform fuel ID: " << m_uniform_fuel_id << "\n";
    } else {
        amrex::Print() << "    LANDFIRE file: " << m_nc_fuel_file << "\n";
    }
    if (!m_fuel_database_path.empty()) {
        amrex::Print() << "  Fuel database: " << m_fuel_database_path << "\n";
    }
    if (m_fire_mode == "constant_wind") {
        amrex::Print() << "  Wind: (" << m_constant_wind_u << ", " << m_constant_wind_v << ") m/s at "
                       << m_wind_height << " m\n";
    }
    amrex::Print() << "  Fuel moisture: 1hr " << m_fuel_moisture_1hr << ", 10hr " << m_fuel_moisture_10hr
                   << ", 100hr " << m_fuel_moisture_100hr << ", live herb " << m_fuel_moisture_live_herb
                   << ", live woody " << m_fuel_moisture_live_woody << "\n";
    if (m_uniform_slope_x != 0.0 || m_uniform_slope_y != 0.0) {
        amrex::Print() << "  Uniform slope: (" << m_uniform_slope_x << ", " << m_uniform_slope_y << ")\n";
    }
    amrex::Print() << "  Ignition: " << m_ignition_type;
    if (m_ignition_type == "point") {
        amrex::Print() << " at (" << m_ignition_x << ", " << m_ignition_y << ") m, radius "
                       << m_ignition_radius << " m";
    } else {
        amrex::Print() << " from " << m_nc_ignition_file << " (time index " << m_ignition_time_index << ")";
    }
    amrex::Print() << ", t = " << m_ignition_time << " s\n";
    amrex::Print() << "  CFL " << m_cfl << ", band thickness " << m_band_thickness << " cells\n";
    if (m_subcycle_interval > 1) {
        amrex::Print() << "  Subcycling: fire updates every " << m_subcycle_interval << " ERF steps\n";
    }
    amrex::Print() << "  use_grid_aware_burnout " << (m_use_grid_aware_burnout ? "true" : "false")
                   << ", use_source_fuel_scaling " << (m_use_source_fuel_scaling ? "true" : "false")
                   << ", terrain_scale_factor " << m_terrain_scale_factor << "\n";
}

ERF_REGISTER_FIRE_ENGINE(FireELMFIRE, "lsfire")

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
