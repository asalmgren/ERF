#ifndef ERF_NULL_FIRE_H
#define ERF_NULL_FIRE_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include "ERF_DataStruct.H"

/**
 * Base class for fire spread models
 */
class NullFire {
public:
    NullFire() = default;
    virtual ~NullFire() = default;

    virtual void Define(SolverChoice& sc) {}
    
    virtual void Init(int lev,
                     const amrex::MultiFab& cons_in,
                     const amrex::Geometry& geom,
                     const amrex::Real& dt_advance) {}
    
    virtual void Advance(const amrex::Real& dt,
                        const amrex::MultiFab* U,
                        const amrex::MultiFab* V,
                        const amrex::MultiFab* terrain,
                        const amrex::iMultiFab* luindex
                    ) {}
                            
    virtual void GetHeatFlux(amrex::MultiFab& heat_flux) {}
    virtual void GetMoistureFlux(amrex::MultiFab& moisture_flux) {}
    
    virtual amrex::Real get_fire_area() const { return 0.0; }
    
    virtual void SetTime(const amrex::Real& current_time) {}
    
    // ========== Plot variable registration ==========
    virtual void GetPlotVarNames(amrex::Vector<std::string>& names) const {}

    virtual void GetPlotVar(const std::string& name,
                           amrex::MultiFab& mf,
                           int lev) const {}

    // ========== Checkpoint/Restart ==========
    virtual void WriteCheckpointData(const std::string& chk_dir, int lev) const {}
    virtual void ReadCheckpointData(const std::string& chk_dir, int lev) {}

    /**
     * @brief Write fire-level plot data as a separate AMReX plotfile.
     *        Only implemented when fire runs at finer resolution than ERF (fire_ref_ratio > 1).
     * @param step  Current step number (used for file naming)
     * @param time  Current simulation time
     */
    virtual void WriteFirePlotFile(int step, amrex::Real time) const {}
};

#endif