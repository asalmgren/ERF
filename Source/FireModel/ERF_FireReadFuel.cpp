#include "ERF_FireELMFIRE.H"
#include "ERF_NCWpsFile.H"  // For BuildFABsFromNetCDFFile template
#include <AMReX_Print.H>

#ifdef ERF_USE_NETCDF

/**
 * @brief Read static LANDFIRE fuel data from NetCDF file
 * 
 * This is a fire model utility that uses ERF's existing NetCDF infrastructure
 * (BuildFABsFromNetCDFFile template) to read static 2D fuel data.
 * 
 * Expected NetCDF variable: FBFM40 (LANDFIRE Fire Behavior Fuel Model 40)
 * Data type: integer
 * Dimensions: Time x South-North x West-East (2D static field)
 * 
 * @param domain Simulation domain box
 * @param fname NetCDF file path
 * @param NC_fuel_iab Output IArrayBox for fuel data
 * @return 1 if successful, 0 if failed
 */
int FireELMFIRE::read_static_fuel_from_netcdf(const amrex::Box& domain,
                                               const std::string& fname,
                                               amrex::IArrayBox& NC_fuel_iab)
{
    using namespace amrex;
    
    Vector<IArrayBox*> NC_iabs;
    Vector<std::string> NC_inames;
    Vector<enum NC_Data_Dims_Type> NC_idim_types;
    
    // LANDFIRE FBFM40 variable name in NetCDF
    NC_iabs.push_back(&NC_fuel_iab);
    NC_inames.push_back("FBFM");
    NC_idim_types.push_back(NC_Data_Dims_Type::Time_SN_WE);  // 2D static field
    
    // Use ERF's existing infrastructure (BuildFABsFromNetCDFFile is in ERF_NCWpsFile.H)
    Vector<int> success;
    success.resize(NC_iabs.size());
    BuildFABsFromNetCDFFile<IArrayBox,int>(domain, fname, NC_inames, 
                                           NC_idim_types, NC_iabs, success);
    
    if (success[0] == 1) {
        Print() << "🔥 Successfully read FBFM40 fuel data from " << fname << std::endl;
    } else {
        Print() << "🔥 WARNING: Failed to read FBFM40 from " << fname << std::endl;
        Print() << "   Check that the file exists and contains variable 'FBFM40'\n";
    }
    
    return success[0];
}

#endif // ERF_USE_NETCDF
