#ifndef ERF_FIRE_MODEL_H
#define ERF_FIRE_MODEL_H

#include "ERF_NullFire.H"
#include "ERF_DataStruct.H"
#include "ERF_FireELMFIRE.H"

#include <memory>
#include <string>

// ============================================
// FIRE MODEL CONTAINER CLASS
// ============================================
/**
 * Container class for fire model (similar to LandSurface)
 * Allows runtime selection of fire model type
 */
class FireModel {
public:
    FireModel() = default;
    ~FireModel() = default;

    void ReSize(const int& nlev) {
        m_fire.resize(nlev);
        for (int lev = 0; lev < nlev; ++lev) {
            m_fire[lev] = std::make_unique<NullFire>();
        }
    }

    template<class FireModelType>
    void SetModel() {
        for (auto& f : m_fire) {
            f = std::make_unique<FireModelType>();
        }
    }

    void Define(const int& lev, SolverChoice& sc) {
        m_fire[lev]->Define(sc);
    }

    void Init(int lev,
             const amrex::MultiFab& cons_in,
             const amrex::Geometry& geom,
             const amrex::Real& dt_advance) {
        m_fire[lev]->Init(lev, cons_in, geom, dt_advance);
    }

    void Advance(const int& lev, const amrex::Real& dt,
                const amrex::MultiFab* U, 
                const amrex::MultiFab* V,
                const amrex::MultiFab* terrain,
                const amrex::MultiFab* wind_height) {
        m_fire[lev]->Advance(dt, U, V, terrain, wind_height);
    }

    void GetHeatFlux(const int& lev, amrex::MultiFab& heat_flux) {
        m_fire[lev]->GetHeatFlux(heat_flux);
    }

    void GetMoistureFlux(const int& lev, amrex::MultiFab& moisture_flux) {
        m_fire[lev]->GetMoistureFlux(moisture_flux);
    }

    amrex::Real get_fire_area(const int& lev) const {
        return m_fire[lev]->get_fire_area();
    }

    void SetTime(const int& lev, const amrex::Real& current_time) {
        m_fire[lev]->SetTime(current_time);
    }

    void GetPlotVarNames(amrex::Vector<std::string>& names) const {
        if (m_fire.size() > 0) {
            m_fire[0]->GetPlotVarNames(names);
        }
    }

    void GetPlotVar(const std::string& name,
                    amrex::MultiFab& mf,
                    const int& lev) const {
        m_fire[lev]->GetPlotVar(name, mf, lev);
    }

    void WriteCheckpointData(const std::string& chk_dir, const int& lev) const {
        m_fire[lev]->WriteCheckpointData(chk_dir, lev);
    }

    void ReadCheckpointData(const std::string& chk_dir, const int& lev) {
        m_fire[lev]->ReadCheckpointData(chk_dir, lev);
    }

    /** Write fire-level plot data (100m) as separate AMReX plotfiles.
     *  No-op when fire_ref_ratio=1 (fire and ERF on same grid). */
    void WriteFirePlotFile(int step, amrex::Real time) const {
        if (!m_fire.empty()) {
            m_fire[0]->WriteFirePlotFile(step, time);
        }
    }

private:
    amrex::Vector<std::unique_ptr<NullFire>> m_fire;
};

#endif // ERF_FIRE_MODEL_H