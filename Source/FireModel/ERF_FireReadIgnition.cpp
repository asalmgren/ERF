#include "ERF_FireELMFIRE.H"
#include "ERF_NCWpsFile.H"  // For ReadTimeSliceFromNetCDFFile
#include <AMReX_Print.H>

#ifdef ERF_USE_NETCDF

/**
 * @brief Read ignition mask from NetCDF file
 * 
 * This function reads a 2D integer mask (0=unignited, 1=ignited) from a NetCDF file
 * to support area-based ignition patterns. The mask defines which cells should be
 * ignited at the ignition time.
 * 
 * Expected NetCDF variable: ignition
 * Data type: integer (0 or 1)
 * Dimensions: Time x South-North x West-East (2D static field)
 * Values: 0 = unignited, 1 = ignited
 * 
 * @param domain Simulation domain box
 * @param fname NetCDF file path
 * @param time_index Time index to read from NetCDF file (0-based)
 * @param NC_ign_iab Output IArrayBox for ignition mask
 * @return 1 if successful, 0 if failed
 */
int FireELMFIRE::read_ignition_from_netcdf(const amrex::Box& domain,
                                           const std::string& fname,
                                           int time_index,
                                           amrex::IArrayBox& NC_ign_iab)
{
    using namespace amrex;
    
    // Vector to hold the ignition data read from NetCDF
    Vector<NDArray<int>> nc_arrays;
    Vector<std::string> nc_var_names;
    Vector<int> success;
    
    // Ignition mask variable name in NetCDF
    nc_var_names.push_back("fire_mask");
    
    // Allocate arrays
    nc_arrays.resize(nc_var_names.size());
    success.resize(nc_var_names.size());
    success[0] = 0;  // Initialize to failure
    
    // Read specified time slice from NetCDF file (only on IO processor)
    ReadTimeSliceFromNetCDFFile<int>(fname, time_index, nc_var_names, nc_arrays, success);
    
    // Broadcast success status to all ranks
    ParallelDescriptor::Bcast(&success[0], 1, ParallelDescriptor::IOProcessorNumber());
    
    if (success[0] == 0) {
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "🔥 WARNING: Failed to read ignition mask from " << fname << std::endl;
            Print() << "   Check that the file exists and contains variable 'fire_mask'\n";
        }
        return 0;
    }
    
    // Get dimensions (broadcast from IO processor)
    int ns2 = 0, ns3 = 0;
    if (ParallelDescriptor::IOProcessor()) {
        std::vector<size_t> shape = nc_arrays[0].get_vshape();
        AMREX_ALWAYS_ASSERT(shape.size() == 3);  // Time (1), South-North, West-East
        AMREX_ALWAYS_ASSERT(shape[0] == 1);      // Should be single time slice
        ns2 = shape[1];  // south_north
        ns3 = shape[2];  // west_east
    }
    ParallelDescriptor::Bcast(&ns2, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&ns3, 1, ParallelDescriptor::IOProcessorNumber());
    
    // Create box for the data (all ranks)
    Box in_box(IntVect(0,0,0), IntVect(ns3-1, ns2-1, 0));  // 2D field (k=0)
    
    // Allocate IArrayBox on CPU (all ranks)
    Arena* Arena_Used = amrex::The_Arena();
#ifdef AMREX_USE_GPU
    Arena_Used = amrex::The_Pinned_Arena();
#endif
    NC_ign_iab.resize(in_box, 1, Arena_Used);
    
    // Only IO processor copies data from NetCDF arrays
    if (ParallelDescriptor::IOProcessor()) {
        Array4<int> iab_arr = NC_ign_iab.array();
        int* dataPtr = nc_arrays[0].get_data();
        for (int j = 0; j < ns2; ++j) {
            for (int i = 0; i < ns3; ++i) {
                int n = j * ns3 + i;
                iab_arr(i, j, 0, 0) = dataPtr[n];
            }
        }
        
        Print() << "🔥 Successfully read ignition mask (time index " << time_index 
                << ") from " << fname << std::endl;
    }
    
    // Broadcast the ignition data to all ranks
    int total_size = ns2 * ns3;
    ParallelDescriptor::Bcast(NC_ign_iab.dataPtr(), total_size, ParallelDescriptor::IOProcessorNumber());
    
    return success[0];
}

#endif // ERF_USE_NETCDF
