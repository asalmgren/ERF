#ifndef ERF_FIRE_ELMFIRE_H
#define ERF_FIRE_ELMFIRE_H

#include <ERF_FireEngine.H>
#include <ERF_FireCouplingParams.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include <ELMFIRE_API.H>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ERFFireHierarchy;   // LSFIRE FireHierarchy adapted to ERF (ERF_FireELMFIRE.cpp)

/**
 * @brief The LSFIRE (ELMFIRE-CPP) fire engine behind ERF's fire coupling
 *
 * erf.fire.engine = lsfire selects it (ERF_REGISTER_FIRE_ENGINE in
 * ERF_FireELMFIRE.cpp). ERF's coupling layer (Source/FireCoupling) builds the
 * fire grid, hands the engine the atmosphere each ERF step (FireForcing) and
 * injects the fluxes the engine returns (FireFluxes); this class wraps the
 * LSFIRE FireHierarchy and translates between the two.
 *
 * - The fire level 0 of the LSFIRE hierarchy is ERF's FireGrid: the ERF
 *   level-0 boxes (k = 0) refined by erf.fire.grid_ratio, on the same ranks.
 *   An optional static fire level 1 (fire_amr.*) refines it further; fluxes
 *   go back to ERF from fire level 0.
 * - In atmospheric mode (fire.mode = atmospheric) the fire is driven by the
 *   wind of the lowest ERF cell and its height above ground
 *   (extract_fire_wind_k0), held fixed over the fire update.
 * - The engine advances once every fire.subcycle_interval ERF steps, with as
 *   many internal CFL steps as it needs, and averages its fire-level-0 fluxes
 *   over the update. With fire.subcycle_interval = 1 (default) the fluxes are
 *   the mean over exactly the ERF step, as FireFluxes asks. With N > 1 the
 *   mean of an update, over N ERF steps, is returned for each of the N steps
 *   that follow it: the energy reaching ERF equals the energy released, but
 *   lags it by up to N steps.
 * - The fire works in host loops: the fields it reads and writes are in
 *   managed memory, and are synchronised before the coupler reads them.
 * - Restart is not supported yet: FireEngine's default read_checkpoint stops
 *   a restarted run.
 *
 * The engine's own keys stay under fire.* and fire_amr.* (some are read by
 * the LSFIRE library itself); the coupling keys are erf.fire.*.
 */
class FireELMFIRE : public FireEngine
{
public:
    explicit FireELMFIRE (const FireCouplingParams& coupling);
    ~FireELMFIRE () override;

    const char* name () const override { return "lsfire"; }

    void verify_prerequisites (const ERF& erf, int lev) override;

    void initialize (const ERF& erf,
                     int lev,
                     const FireGrid& fg,
                     const SurfaceLayer* surface_layer,
                     const amrex::MultiFab* z_phys_nd_atm) override;

    void advance (const FireForcing& forcing) override;

    void fill_fluxes (FireFluxes& fluxes) override;

    amrex::Real smoke_heat_per_kg () const override { return m_smoke_heat_per_kg; }

    /// The wind and heights come from FireForcing: any bottom boundary will do
    bool needs_surface_layer () const override { return false; }

    void write_plotfile (const std::string& plotfile_prefix, amrex::Real time, int step) const override;

    std::vector<std::string> plot_var_names () const override;
    const amrex::MultiFab* plot_var (const std::string& name) const override;

    amrex::Real burned_area () const override;

private:
    /// LSFIRE fire-level hierarchy; its level 0 is ERF's FireGrid
    std::unique_ptr<ERFFireHierarchy> m_hier;

    /// Fire engine of fire level 0 (fluxes, plot variables)
    ELMFIRE::FireEngine& Engine0 () const;

    /// Read the fire.* keys (constructor)
    void read_inputs ();

    /// Print the configuration (IO rank)
    void print_config () const;

    void load_fuel_data (const amrex::Geometry& geom,
                         const amrex::BoxArray& ba,
                         const amrex::DistributionMapping& dm);

    void load_ignition_data (const amrex::Geometry& geom,
                             const amrex::BoxArray& ba,
                             const amrex::DistributionMapping& dm);

#ifdef ERF_USE_NETCDF
    static int read_static_fuel_from_netcdf (const amrex::Box& domain,
                                             const std::string& fname,
                                             amrex::IArrayBox& NC_fuel_iab);

    static int read_ignition_from_netcdf (const amrex::Box& domain,
                                          const std::string& fname,
                                          int time_index,
                                          amrex::IArrayBox& NC_ign_iab);

    static int read_refine_mask_from_netcdf (const std::string& fname,
                                             std::vector<unsigned char>& mask,
                                             int& nx, int& ny);
#endif

    /// The coupling inputs (grid_ratio, coupling_type, fire_debug)
    FireCouplingParams m_coupling;

    /// ELMFIRE configuration
    ELMFIRE::Config m_config;

    bool m_initialized = false;
    amrex::Real m_current_time = 0.0;

    // The ERF level the fire grid refines (level 0) and its k = 0 slabs
    amrex::Geometry                m_geom_atm;
    amrex::BoxArray                m_ba_atm_2d;   ///< the level's boxes at k = 0
    amrex::DistributionMapping     m_dm_atm;
    const amrex::MultiFab*         m_z_phys_nd = nullptr;   ///< nodal heights of the level (static terrain)

    // The fire grid (FireGrid of the coupler)
    amrex::Geometry                m_geom_fire;
    amrex::BoxArray                m_ba_fire;
    amrex::DistributionMapping     m_dm_fire;

    // ERF -> fire: the k = 0 wind, its height above ground and the k = 0
    // cell-centre height, on the level's k = 0 slabs (host-accessible)
    amrex::MultiFab m_u_k0, m_v_k0, m_h_k0, m_terrain_k0;

    // Fire -> ERF: fire-level-0 fluxes, mean of the last fire update [W/m2],
    // and the vapour flux [kg/m2/s] they come from
    amrex::MultiFab m_heat_flux_fire;
    amrex::MultiFab m_moist_flux_fire;
    amrex::MultiFab m_latent_flux_fire;

    // Fire-level-0 wind of the last update, for the fire plotfile
    amrex::MultiFab m_U_fire, m_V_fire, m_H_fire;

    /// plot_var() fields on the fire grid, filled on request
    mutable std::map<std::string, std::unique_ptr<amrex::MultiFab>> m_plot_cache;

    // ---- fire.* inputs ----
    std::string m_fire_mode = "constant_wind";         ///< fire.mode: constant_wind or atmospheric
    amrex::Vector<std::string> m_fire_plot_vars;       ///< fire.fire_plot_vars_1 (empty: engine defaults)
    std::string m_fuel_model_source = "uniform";       ///< fire.fuel_model_source: uniform or landfire
    int m_uniform_fuel_id = 102;                       ///< fire.uniform_fuel_id (GR2)
    std::string m_fuel_database_path;
    amrex::Real m_constant_wind_u = 5.0;
    amrex::Real m_constant_wind_v = 0.0;
    amrex::Real m_wind_height = 10.0;
    amrex::Real m_fuel_moisture_1hr = 0.03;
    amrex::Real m_fuel_moisture_10hr = 0.03;
    amrex::Real m_fuel_moisture_100hr = 0.03;
    amrex::Real m_fuel_moisture_live_herb = 0.30;
    amrex::Real m_fuel_moisture_live_woody = 0.60;
    amrex::Real m_uniform_slope_x = 0.0;
    amrex::Real m_uniform_slope_y = 0.0;
    std::string m_ignition_type = "point";
    std::string m_nc_ignition_file;
    int m_ignition_time_index = 0;
    amrex::Real m_ignition_x = 0.0;
    amrex::Real m_ignition_y = 0.0;
    amrex::Real m_ignition_radius = 50.0;
    amrex::Real m_ignition_time = 0.0;
    amrex::Real m_cfl = 0.9;
    int m_band_thickness = 5;
    amrex::Real m_wind_data_scale_factor = 0.7;
    bool m_use_grid_aware_burnout  = false;
    bool m_use_source_fuel_scaling = false;
    amrex::Real m_source_fuel_exponent = 1.0;
    amrex::Real m_terrain_scale_factor = 1.0;
    amrex::Real m_atmos_wind_height = 0.0;
    bool m_use_atmos_wind_height_override = false;
    int m_subcycle_interval = 1;                       ///< fire.subcycle_interval
    int m_verbosity = 1;
    /// fire.smoke_heat_per_kg: heat released per kg of fuel burned [J/kg], for ERF's smoke tracer
    amrex::Real m_smoke_heat_per_kg = 1.86e7;

    // Subcycling state
    int m_atmo_step_count = 0;
    amrex::Real m_accumulated_dt = 0.0;

    // Fire-level hierarchy settings (fire_amr.*) and level-1 inputs
    int m_fire_max_level = 0;
    std::string m_nc_refine_mask_file;
    std::string m_nc_fuel_file;
    std::string m_nc_fuel_file_l1;
    std::string m_nc_ignition_file_l1;
    amrex::IArrayBox m_fuel_l1_full;
    amrex::IArrayBox m_ignition_l1_full;

    // LANDFIRE fuel map and area-ignition mask on fire level 0
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_fuel_model_lev;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_ignition_mask_lev;
};

#endif // ERF_FIRE_ELMFIRE_H
