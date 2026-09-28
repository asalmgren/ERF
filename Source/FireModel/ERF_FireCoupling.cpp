/**
 * \file ERF_FireCoupling.cpp
 * \brief ERF <-> LSFIRE (ELMFIRE) coupling on ERF level 0.
 *
 * The fire model owns its own 2D grid (optionally refined by fire.fire_ref_ratio);
 * that grid is unrelated to ERF's atmospheric AMR levels. Only ERF level 0 is coupled.
 */
#include "ERF.H"
#include "ERF_Constants.H"

using namespace amrex;

namespace {

// Device kernels live in free functions: nvcc does not allow extended lambdas
// whose enclosing function is a private/protected member.

// Wind at the lowest cell centre (k = k0), averaged from x/y faces to the cell centre,
// and its height above ground: z_cc(k0) minus the surface height (mean of the four
// bottom nodes when nodal heights exist, else 0 for a flat ConstantDz mesh).
void
extract_surface_wind (const Box& bx, int k0,
                      const Array4<Real const>& u, const Array4<Real const>& v,
                      const Array4<Real const>& zcc,
                      bool use_znd, const Array4<Real const>& znd,
                      const Array4<Real>& us, const Array4<Real>& vs, const Array4<Real>& hs)
{
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
    {
        us(i,j,0) = Real(0.5) * (u(i,j,k0) + u(i+1,j,k0));
        vs(i,j,0) = Real(0.5) * (v(i,j,k0) + v(i,j+1,k0));
        const Real zsfc = use_znd ? Real(0.25) * (znd(i,j,k0) + znd(i+1,j,k0) + znd(i,j+1,k0) + znd(i+1,j+1,k0))
                                  : Real(0.0);
        hs(i,j,0) = zcc(i,j,k0) - zsfc;
    });
}

// cons(i,j,k0,comp) += flux * dt * scale / dz. With nodal heights (terrain-following
// or variable-dz meshes) dz is the actual bottom-cell thickness of each column;
// otherwise (no z_phys_nd) a uniform bottom-cell thickness dz0 is used.
void
apply_bottom_flux (const Box& bx2d, int k0,
                   const Array4<Real>& cons, const Array4<Real const>& flux,
                   bool use_znd, const Array4<Real const>& znd, Real dz0,
                   Real dt, Real scale, int comp)
{
    ParallelFor(bx2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
    {
        Real dz = dz0;
        if (use_znd) {
            const Real z_top = Real(0.25) * (znd(i,j,k0+1) + znd(i+1,j,k0+1) + znd(i,j+1,k0+1) + znd(i+1,j+1,k0+1));
            const Real z_bot = Real(0.25) * (znd(i,j,k0  ) + znd(i+1,j,k0  ) + znd(i,j+1,k0  ) + znd(i+1,j+1,k0  ));
            dz = z_top - z_bot;
        }
        cons(i,j,k0,comp) += flux(i,j,k0) * dt * scale / dz;
    });
}

// The fire engine reads/writes these fields in host loops: host-accessible memory
// (managed on GPU builds, plain on CPU builds).
MFInfo HostAccessible () { return MFInfo().SetArena(The_Managed_Arena()); }

} // namespace

void
ERF::init_fire_model (int lev)
{
    if (solverChoice.fire_type == FireType::None) { return; }
    AMREX_ALWAYS_ASSERT(lev == 0);

    const MultiFab& S = vars_new[lev][Vars::cons];

    // Heat/moisture flux and surface wind use one 2D slab per ERF box; that requires
    // every level-0 box to start at the bottom of the domain.
    const int klo = geom[lev].Domain().smallEnd(2);
    for (int ib = 0; ib < S.boxArray().size(); ++ib) {
        if (S.boxArray()[ib].smallEnd(2) != klo) {
            Abort("Fire coupling requires ERF level-0 boxes to span the full vertical extent "
                  "(do not decompose in z, e.g. set amr.max_grid_size_z >= n_cell_z)");
        }
    }

    fire_heat_flux     = std::make_unique<MultiFab>(S.boxArray(), S.DistributionMap(), 1, 0, HostAccessible());
    fire_moisture_flux = std::make_unique<MultiFab>(S.boxArray(), S.DistributionMap(), 1, 0, HostAccessible());
    fire_heat_flux->setVal(zero);
    fire_moisture_flux->setVal(zero);

    fire_model.Define(lev, solverChoice);
    fire_model.Init(lev, S, geom[lev], dt[lev]);
}

void
ERF::advance_fire_model (int lev, double time, Real dt_lev,
                         MultiFab& S_new, const MultiFab& U_new, const MultiFab& V_new)
{
    if (solverChoice.fire_type == FireType::None || lev != 0) { return; }

    // One fire advance per ERF level-0 step
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(time > fire_last_advance_time,
                                     "Fire model advanced twice for the same ERF step");
    fire_last_advance_time = time;

    const int klo = geom[lev].Domain().smallEnd(2);

    // ---- Atmosphere -> fire: surface wind and terrain height ---------------------
    BoxList bl = S_new.boxArray().boxList();
    for (Box& b : bl) { b.setSmall(2, klo); b.setBig(2, klo); }
    const BoxArray ba2d(std::move(bl));

    // Terrain height of the bottom cell centre (k = klo) with one horizontal ghost cell,
    // as a host-accessible 2D slab (z_phys_cc itself may live in device memory)
    if (!fire_terrain_sfc || fire_terrain_sfc->boxArray() != ba2d) {
        fire_terrain_sfc = std::make_unique<MultiFab>(ba2d, S_new.DistributionMap(), 1, IntVect(1,1,0), HostAccessible());
    }
    fire_terrain_sfc->ParallelCopy(*z_phys_cc[lev], 0, 0, 1, IntVect(1,1,0), IntVect(1,1,0));

    const MultiFab* U_ptr = nullptr;
    const MultiFab* V_ptr = nullptr;
    const MultiFab* H_ptr = nullptr;   // height above ground of U/V [m]
    if (solverChoice.fire_met_source == FireMetSource::Atmosphere) {
        if (!fire_u_sfc || fire_u_sfc->boxArray() != ba2d) {
            fire_u_sfc = std::make_unique<MultiFab>(ba2d, S_new.DistributionMap(), 1, 1, HostAccessible());
            fire_v_sfc = std::make_unique<MultiFab>(ba2d, S_new.DistributionMap(), 1, 1, HostAccessible());
            fire_h_sfc = std::make_unique<MultiFab>(ba2d, S_new.DistributionMap(), 1, 1, HostAccessible());
        }
        const bool use_znd = (z_phys_nd[lev] != nullptr);
        for (MFIter mfi(*fire_u_sfc); mfi.isValid(); ++mfi) {
            const Array4<Real const> znd = use_znd ? z_phys_nd[lev]->const_array(mfi) : Array4<Real const>{};
            extract_surface_wind(mfi.validbox(), klo, U_new.const_array(mfi), V_new.const_array(mfi),
                                 z_phys_cc[lev]->const_array(mfi), use_znd, znd,
                                 fire_u_sfc->array(mfi), fire_v_sfc->array(mfi), fire_h_sfc->array(mfi));
        }
        fire_u_sfc->FillBoundary(geom[lev].periodicity());
        fire_v_sfc->FillBoundary(geom[lev].periodicity());
        fire_h_sfc->FillBoundary(geom[lev].periodicity());
        U_ptr = fire_u_sfc.get();
        V_ptr = fire_v_sfc.get();
        H_ptr = fire_h_sfc.get();
    }

    Gpu::streamSynchronize();  // device kernels/copies above -> host reads in the fire engine
    fire_model.Advance(lev, dt_lev, U_ptr, V_ptr, fire_terrain_sfc.get(), H_ptr);

    // ---- Fire -> atmosphere: explicit source in the bottom cell (optional) ----------
    // Kept as in the original coupling for stage 2 (off by default):
    //   RhoTheta += Q * dt / (Cp_d * dz)   -- note: omits the Exner factor (T -> theta);
    //                                         near the surface Pi ~ 1 (a few % error)
    //   RhoQ1    += E * dt / dz
    // Q is the fire model's instantaneous flux at the end of this fire update.
    if (solverChoice.fire_heat_feedback) {
        fire_model.GetHeatFlux(lev, *fire_heat_flux);
        const bool have_qv = (solverChoice.moisture_type != MoistureType::None) &&
                             (RhoQ1_comp < S_new.nComp());
        if (have_qv) { fire_model.GetMoistureFlux(lev, *fire_moisture_flux); }

        // Bottom-cell thickness: per column from z_phys_nd when it exists (it is null
        // for ConstantDz meshes), else the stretched or uniform first-level dz.
        const bool use_znd = (z_phys_nd[lev] != nullptr);
        Real dz0 = geom[lev].CellSize(2);
        if (!use_znd && solverChoice.mesh_type == MeshType::StretchedDz && !stretched_dz_h[lev].empty()) {
            dz0 = stretched_dz_h[lev][klo];
        }

        for (MFIter mfi(S_new); mfi.isValid(); ++mfi) {
            Box bx2d = mfi.validbox();
            bx2d.setSmall(2, klo); bx2d.setBig(2, klo);
            const auto& cons = S_new.array(mfi);
            const Array4<Real const> znd = use_znd ? z_phys_nd[lev]->const_array(mfi) : Array4<Real const>{};
            apply_bottom_flux(bx2d, klo, cons, fire_heat_flux->const_array(mfi),
                              use_znd, znd, dz0, dt_lev, one / Cp_d, RhoTheta_comp);
            if (have_qv) {
                apply_bottom_flux(bx2d, klo, cons, fire_moisture_flux->const_array(mfi),
                                  use_znd, znd, dz0, dt_lev, one, RhoQ1_comp);
            }
        }

        static bool warned_no_qv = false;
        if (!have_qv && !warned_no_qv) {
            Print() << "Fire: no moisture model active; fire moisture flux is not applied" << std::endl;
            warned_no_qv = true;
        }
    }

    if (verbose > 1) {
        // min/max/area are MPI reductions: every rank calls them, Print() writes on IO rank
        const Real area = fire_model.get_fire_area(lev);
        Real qmax = zero, power = zero, vapor = zero;
        if (solverChoice.fire_heat_feedback) {
            const auto dx = geom[lev].CellSizeArray();
            const bool have_qv = (solverChoice.moisture_type != MoistureType::None) &&
                                 (RhoQ1_comp < S_new.nComp());
            qmax  = fire_heat_flux->max(0);
            power = fire_heat_flux->sum(0) * dx[0] * dx[1];  // fluxes are nonzero only at k=0
            if (have_qv) { vapor = fire_moisture_flux->sum(0) * dx[0] * dx[1]; }
        }
        Print() << "Fire at t = " << time + dt_lev << " s: area = " << area << " m^2"
                << ", max heat flux = " << qmax << " W/m^2"
                << ", applied heat power = " << power << " W"
                << ", applied vapor = " << vapor << " kg/s" << std::endl;
    }
}
