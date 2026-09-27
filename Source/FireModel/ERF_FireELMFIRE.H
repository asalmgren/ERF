#ifndef ERF_FIRE_ELMFIRE_H
#define ERF_FIRE_ELMFIRE_H

#include "ERF_NullFire.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <ELMFIRE_API.H>
#include <memory>

/**
 * @brief ERF wrapper for ELMFIRE-CPP fire spread model
 *
 * This class integrates the ELMFIRE-CPP library into ERF's fire modeling framework.
 * It wraps ELMFIRE::FireEngine and translates between ERF's interface and ELMFIRE's API.
 *
 * Features:
 * - Level set fire spread using Rothermel physics
 * - Narrow band tracking for computational efficiency
 * - Two modes: constant wind (offline testing) and atmospheric coupling (ERF-driven)
 * - MPI parallel with AMReX MultiFabs
 */
class FireELMFIRE : public NullFire {
public:
    FireELMFIRE();
    ~FireELMFIRE() override = default;

    // ============================================
    // PUBLIC INTERFACE (ERF integration)
    // ============================================

    /**
     * @brief Define fire model parameters from ERF SolverChoice
     * @param sc SolverChoice containing fire model configuration
     */
    void Define(SolverChoice& sc) override;

    /**
     * @brief Initialize fire model with domain geometry
     * @param lev AMR level
     * @param cons_in Conservative variables (for diagnostics)
     * @param geom Domain geometry
     * @param dt_advance ERF timestep (for reference)
     */
    void Init(int lev,
             const amrex::MultiFab& cons_in,
             const amrex::Geometry& geom,
             const amrex::Real& dt_advance) override;

    /**
     * @brief Set current simulation time (for restart)
     * @param current_time Current simulation time in seconds
     */
    void SetTime(const amrex::Real& current_time) override { 
        m_current_time = current_time;
        // Also synchronize ELMFIRE engine's internal time
        if (m_engine) {
            m_engine->SetTime(current_time);
        }
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "🔥 Fire model time synchronized to t = " << current_time << " s (restart)\n";
        }
    }

    /**
     * @brief Advance fire spread by dt
     * @param dt Timestep
     * @param U Wind velocity component (u) from ERF
     * @param V Wind velocity component (v) from ERF
     * @param terrain Terrain elevation
     * @param luindex Land use index (fuel map)
     */
    void Advance(const amrex::Real& dt,
                const amrex::MultiFab* U,
                const amrex::MultiFab* V,
                const amrex::MultiFab* terrain,
                const amrex::iMultiFab* luindex) override;

    /**
     * @brief Get sensible heat flux from fire (W/m²)
     * @param heat_flux Output MultiFab for heat flux
     */
    void GetHeatFlux(amrex::MultiFab& heat_flux) override;

    /**
     * @brief Get moisture flux from fire (kg/m²/s)
     * @param moisture_flux Output MultiFab for moisture flux
     */
    void GetMoistureFlux(amrex::MultiFab& moisture_flux) override;

    /**
     * @brief Get total burned area (m²)
     * @return Total area where phi < 0
     */
    amrex::Real get_fire_area() const override;

    /**
     * @brief Get names of plot variables for visualization
     * @param names Output vector of variable names
     */
    void GetPlotVarNames(amrex::Vector<std::string>& names) const override;

    /**
     * @brief Fill MultiFab with plot variable data
     * @param name Variable name
     * @param mf Output MultiFab
     * @param lev AMR level
     */
    void GetPlotVar(const std::string& name,
                   amrex::MultiFab& mf,
                   int lev) const override;

    /**
     * @brief Write fire state to checkpoint directory
     * @param chk_dir Checkpoint directory path
     * @param lev AMR level
     */
    void WriteCheckpointData(const std::string& chk_dir, int lev) const override;

    /**
     * @brief Read fire state from checkpoint directory and restore
     * @param chk_dir Checkpoint directory path
     * @param lev AMR level
     */
    void ReadCheckpointData(const std::string& chk_dir, int lev) override;

    /**
     * @brief Write fire-level (100m) data as a separate AMReX plotfile.
     *        Only active when m_fire_ref_ratio > 1.
     * @param step  Current step number (e.g. fire_plt0007200)
     * @param time  Current simulation time in seconds
     */
    void WriteFirePlotFile(int step, amrex::Real time) const override;

private:
    // ============================================
    // DATA MEMBERS
    // ============================================

    /// ELMFIRE engine (from library)
    std::unique_ptr<ELMFIRE::FireEngine> m_engine;

    /// ELMFIRE configuration
    ELMFIRE::Config m_config;

    /// Current AMR level
    int m_lev = 0;

    /// Simulation time
    amrex::Real m_current_time = 0.0;

    /// Domain geometry
    amrex::Geometry m_geom;

    /// Initialization flag
    bool m_initialized = false;

    /// BoxArray and DistributionMapping (saved for checkpoint I/O)
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    // ============================================
    // CONFIGURATION PARAMETERS (from ERF input)
    // ============================================

    /// Fire model type ("constant_wind" or "atmospheric")
    std::string m_fire_mode = "constant_wind";

    /// Prefix for fire-level plotfile directories (default "fire_plt")
    std::string m_fire_plot_file = "fire_plt";

    /// Variables to include in fire-level plotfile (from fire.fire_plot_vars_1).
    /// Empty = write all engine default variables.
    amrex::Vector<std::string> m_fire_plot_vars;

    /// Fuel model source ("uniform", "metgrid_igbp", or "landfire")
    /// - uniform: Use single fuel model ID for entire domain
    /// - metgrid_igbp: Convert IGBP land use index from metgrid to fuel models
    /// - landfire: Use LANDFIRE fuel data from ERF (placeholder)
    std::string m_fuel_model_source = "uniform";

    /// Fuel model ID for uniform mode (default: 102 = GR2 moderate grass)
    int m_uniform_fuel_id = 102;

    /// Path to fuel database CSV
    std::string m_fuel_database_path = "";

    /// Constant wind components (m/s) for offline mode
    amrex::Real m_constant_wind_u = 5.0;
    amrex::Real m_constant_wind_v = 0.0;
    amrex::Real m_wind_height = 10.0;  // Reference height (m)

    /// Fuel moisture content (fraction)
    amrex::Real m_fuel_moisture_1hr = 0.03;
    amrex::Real m_fuel_moisture_10hr = 0.03;
    amrex::Real m_fuel_moisture_100hr = 0.03;
    amrex::Real m_fuel_moisture_live_herb = 0.30;
    amrex::Real m_fuel_moisture_live_woody = 0.60;

    /// Terrain slope (dimensionless)
    amrex::Real m_uniform_slope_x = 0.0;
    amrex::Real m_uniform_slope_y = 0.0;

    /// Ignition parameters
    std::string m_ignition_type = "point";  // "point" or "area"
    std::string m_nc_ignition_file = "";    // NetCDF file with ignition mask (0/1)
    int m_ignition_time_index = 0;          // Time index to use from NetCDF file (for area ignition)
    amrex::Real m_ignition_x = 0.0;         // meters (for point ignition)
    amrex::Real m_ignition_y = 0.0;         // meters (for point ignition)
    amrex::Real m_ignition_radius = 50.0;   // meters (for point ignition)
    amrex::Real m_ignition_time = 0.0;      // seconds

    /// Numerical parameters
    amrex::Real m_cfl = 0.9;
    int m_band_thickness = 5;

    /// Wind data scale factor (HRRR bias correction, default 0.7)
    amrex::Real m_wind_data_scale_factor = 0.7;

    /// Fire suppression options (default: off)
    bool m_use_grid_aware_burnout  = false;
    bool m_use_source_fuel_scaling = false;
    amrex::Real m_source_fuel_exponent   = 1.0;
    amrex::Real m_terrain_scale_factor   = 1.0;  ///< Slope amplification multiplier (1.0 = normal)

    /// Subcycling parameters (fire runs at larger timestep than atmosphere)
    int m_subcycle_interval = 1;         // Fire updates every N atmospheric steps (1 = every step)
    amrex::Real m_accumulated_dt = 0.0;  // Accumulated time since last fire update

    /// Verbosity level
    int m_verbosity = 1;

    // ============================================
    // AMR FIRE REFINEMENT (Option A coupling)
    // fire.fire_ref_ratio > 1: ELMFIRE runs at a finer grid than ERF
    // ERF wind is nearest-neighbor prolongated to fire level;
    // heat flux is averaged back to ERF coarse level.
    // ============================================

    /// Refinement ratio between ERF L0 and ELMFIRE fire level (1 = same as ERF)
    int m_fire_ref_ratio = 1;

    /// Fine BoxArray and DistributionMapping (ELMFIRE level)
    amrex::BoxArray m_fire_ba_fine;
    amrex::DistributionMapping m_fire_dm_fine;

    /// Fine geometry (same ProbDomain as ERF, finer cells)
    amrex::Geometry m_geom_fine;

    /// Heat/moisture flux at fire level (fine resolution) — averaged to ERF grid in GetHeatFlux()
    amrex::MultiFab m_heat_flux_fine;
    amrex::MultiFab m_moist_flux_fine;

    /// Prolongated atmospheric wind at fire resolution — stored for fire plotfile output
    amrex::MultiFab m_U_fire_fine;
    amrex::MultiFab m_V_fire_fine;

    /// 2D coarse BoxArray/DM (khi=0) for proper 2D→2D average_down before expanding to 3D ERF MF
    amrex::BoxArray m_ba_crse_2d;
    amrex::DistributionMapping m_dm_crse_2d;
    amrex::MultiFab m_heat_flux_crse_2d;
    amrex::MultiFab m_moist_flux_crse_2d;

    // ============================================
    // LANDFIRE FUEL DATA
    // ============================================

    /// LANDFIRE fuel data (one iMultiFab per AMR level)
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_fuel_model_lev;

    /// Path to LANDFIRE NetCDF file
    std::string m_nc_fuel_file = "";

    // ============================================
    // AREA IGNITION DATA
    // ============================================

    /// Ignition mask data (one iMultiFab per AMR level, 0=unignited, 1=ignited)
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_ignition_mask_lev;

    // ============================================
    // HELPER FUNCTIONS
    // ============================================

    /**
     * @brief Load fuel data from NetCDF file
     * @param geom Domain geometry
     * @param ba BoxArray for decomposition
     * @param dm DistributionMapping
     */
    void load_fuel_data(const amrex::Geometry& geom,
                        const amrex::BoxArray& ba,
                        const amrex::DistributionMapping& dm);

    /**
     * @brief Load ignition mask from NetCDF file
     * @param geom Domain geometry
     * @param ba BoxArray for decomposition
     * @param dm DistributionMapping
     */
    void load_ignition_data(const amrex::Geometry& geom,
                            const amrex::BoxArray& ba,
                            const amrex::DistributionMapping& dm);

#ifdef ERF_USE_NETCDF
    /**
     * @brief Read static LANDFIRE fuel data from NetCDF (static helper)
     * @param domain Simulation domain box
     * @param fname NetCDF file path
     * @param NC_fuel_iab Output IArrayBox for fuel data
     * @return 1 if successful, 0 if failed
     */
    static int read_static_fuel_from_netcdf(const amrex::Box& domain,
                                            const std::string& fname,
                                            amrex::IArrayBox& NC_fuel_iab);

    /**
     * @brief Read ignition mask from NetCDF (static helper)
     * @param domain Simulation domain box
     * @param fname NetCDF file path
     * @param time_index Time index to read from NetCDF file (0-based)
     * @param NC_ign_iab Output IArrayBox for ignition mask
     * @return 1 if successful, 0 if failed
     */
    static int read_ignition_from_netcdf(const amrex::Box& domain,
                                         const std::string& fname,
                                         int time_index,
                                         amrex::IArrayBox& NC_ign_iab);
#endif

    /**
     * @brief Convert IGBP land use index to fuel model ID
     *
     * @param igbp_index IGBP land use classification (1-21)
     * @return Fuel model ID (0 = non-burnable)
     */
    static int IGBPToFuelModel(int igbp_index);

    /**
     * @brief Load configuration from ERF SolverChoice
     * @param sc SolverChoice
     */
    void LoadConfigFromSolverChoice(SolverChoice& sc);

    /**
     * @brief Print configuration summary (IO processor only)
     */
    void PrintConfig() const;
};

#endif // ERF_FIRE_ELMFIRE_H
