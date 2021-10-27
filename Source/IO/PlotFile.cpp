#include <iomanip>
#include <iostream>
#include <string>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <AMReX_Utility.H>
#include <AMReX_buildInfo.H>
#include <AMReX_ParmParse.H>

#include "ERF.H"
#include "PlotFile.H"
#include "IndexDefines.H"

// ERF maintains an internal checkpoint version numbering system.
// This allows us to maintain backwards compatibility with checkpoints
// generated by old versions of the code, so that new versions can
// restart from them. The version number is stored in the ERFHeader
// file inside a checkpoint. The following were the changes that were made
// in updating version numbers:
// 0: all checkpoints as of 11/21/16
// 1: add body state

namespace {
int input_version = -1;
int current_version = 1;
std::string body_state_filename = "body_state.fab";
amrex::Real vfraceps = 0.000001;
} // namespace

// I/O routines for ERF

void
IOManager::restart(amrex::Amr& papa, istream& is, bool bReadSpecial)
{
  // Let's check ERF checkpoint version first;
  // trying to read from checkpoint; if nonexisting, set it to 0.
  if (input_version == -1) {
    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ifstream ERFHeaderFile;
      std::string FullPathERFHeaderFile = papa.theRestartFile();
      FullPathERFHeaderFile += "/ERFHeader";
      ERFHeaderFile.open(FullPathERFHeaderFile.c_str(), std::ios::in);
      if (ERFHeaderFile.good()) {
        char foo[256];
        // first line: Checkpoint version: ?
        ERFHeaderFile.getline(foo, 256, ':');
        ERFHeaderFile >> input_version;
        ERFHeaderFile.close();
      } else {
        input_version = 0;
      }
    }
    amrex::ParallelDescriptor::Bcast(
      &input_version, 1, amrex::ParallelDescriptor::IOProcessorNumber());
  }

  AMREX_ASSERT(input_version >= 0);

  /*
    Deal here with new state descriptor types added, with corresponding
    input_version > 0, if applicable
   */
  amrex::Vector<int> state_in_checkpoint(erf.desc_lst.size(), 1);
  set_state_in_checkpoint(state_in_checkpoint);
  for (int i = 0; i < erf.desc_lst.size(); ++i) {
    if (state_in_checkpoint[i] == 0) {
      const amrex::Real ctime = erf.state[i - 1].curTime();
      erf.state[i].define(
        erf.geom.Domain(), erf.grids, erf.dmap, erf.desc_lst[i], ctime, erf.parent->dtLevel(erf.level),
        *(erf.m_factory));
      erf.state[i] = erf.state[i - 1];
    }
  }
  erf.buildMetrics();

  amrex::MultiFab& S_new = erf.get_new_data(State_Type);

  for (int n = 0; n < erf.src_list.size(); ++n) {
    int oldGrow = NUM_GROW;
    int newGrow = S_new.nGrow();
    erf.old_sources[erf.src_list[n]] =
      std::unique_ptr<amrex::MultiFab>(new amrex::MultiFab(
        erf.grids, erf.dmap, NVAR, oldGrow, amrex::MFInfo(), erf.Factory()));
    erf.new_sources[erf.src_list[n]] =
      std::unique_ptr<amrex::MultiFab>(new amrex::MultiFab(
        erf.grids, erf.dmap, NVAR, newGrow, amrex::MFInfo(), erf.Factory()));
  }

  erf.Sborder.define(erf.grids, erf.dmap, NVAR, NUM_GROW, amrex::MFInfo(), erf.Factory());

  // get the elapsed CPU time to now;
  if (erf.level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    // get elapsed CPU time
    std::ifstream CPUFile;
    std::string FullPathCPUFile = erf.parent->theRestartFile();
    FullPathCPUFile += "/CPUtime";
    CPUFile.open(FullPathCPUFile.c_str(), std::ios::in);

    CPUFile >> erf.previousCPUTimeUsed;
    CPUFile.close();

    amrex::Print() << "read CPU time: " << erf.previousCPUTimeUsed << "\n";
  }

  /*Not implemented for CUDA
      if (level == 0)
      {
    // get problem-specific stuff -- note all processors do this,
    // eliminating the need for a broadcast
    std::string dir = parent->theRestartFile();

    char * dir_for_pass = new char[dir.size() + 1];
    std::copy(dir.begin(), dir.end(), dir_for_pass);
    dir_for_pass[dir.size()] = '\0';

    int len = dir.size();

    Vector<int> int_dir_name(len);
    for (int j = 0; j < len; j++)
        int_dir_name[j] = (int) dir_for_pass[j];

    AMREX_FORT_PROC_CALL(PROBLEM_RESTART,problem_restart)(int_dir_name.dataPtr(),
    &len);

    delete [] dir_for_pass;

      }*/

  if (erf.level > 0 && erf.do_reflux) {
    erf.flux_reg.define(
      erf.grids, papa.boxArray(erf.level - 1), erf.dmap, papa.DistributionMap(erf.level - 1),
      erf.geom, papa.Geom(erf.level - 1), papa.refRatio(erf.level - 1), erf.level, NVAR);
  }
}

void
IOManager::set_state_in_checkpoint(amrex::Vector<int>& state_in_checkpoint)
{
  for (int i = 0; i < erf.num_state_type; ++i) {
    state_in_checkpoint[i] = 1;

    if (i == Work_Estimate_Type) {
      state_in_checkpoint[i] = 0;
    }
  }
}

void
IOManager::checkPoint(
  const std::string& dir,
  std::ostream& os,
  amrex::VisMF::How how,
  bool dump_old_default)
{
  if (erf.level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    {
      std::ofstream ERFHeaderFile;
      std::string FullPathERFHeaderFile = dir;
      FullPathERFHeaderFile += "/ERFHeader";
      ERFHeaderFile.open(FullPathERFHeaderFile.c_str(), std::ios::out);

      ERFHeaderFile << "Checkpoint version: " << current_version << std::endl;
      ERFHeaderFile.close();
    }

    {
      // store elapsed CPU time
      std::ofstream CPUFile;
      std::string FullPathCPUFile = dir;
      FullPathCPUFile += "/CPUtime";
      CPUFile.open(FullPathCPUFile.c_str(), std::ios::out);

      CPUFile << std::setprecision(15) << erf.getCPUTime();
      CPUFile.close();
    }

    /*Not implemented for CUDA{
        // store any problem-specific stuff
        char * dir_for_pass = new char[dir.size() + 1];
        std::copy(dir.begin(), dir.end(), dir_for_pass);
        dir_for_pass[dir.size()] = '\0';

        int len = dir.size();

        Vector<int> int_dir_name(len);
        for (int j = 0; j < len; j++)
      int_dir_name[j] = (int) dir_for_pass[j];

        AMREX_FORT_PROC_CALL(PROBLEM_CHECKPOINT,problem_checkpoint)(int_dir_name.dataPtr(),
    &len);

        delete [] dir_for_pass;
    }*/
  }
}

void
IOManager::setPlotVariables()
{
  amrex::ParmParse pp("erf");

  bool plot_cost = true;
  pp.query("plot_cost", plot_cost);
  if (plot_cost) {
    erf.parent->addDerivePlotVar("WorkEstimate");
  }
}

void
IOManager::writeJobInfo(const std::string& dir)
{
  // job_info file with details about the run
  std::ofstream jobInfoFile;
  std::string FullPathJobInfoFile = dir;
  FullPathJobInfoFile += "/job_info";
  jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

  std::string PrettyLine = "==================================================="
                           "============================\n";
  std::string OtherLine = "----------------------------------------------------"
                          "----------------------------\n";
  std::string SkipSpace = "        ";

  // job information
  jobInfoFile << PrettyLine;
  jobInfoFile << " ERF Job Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << "job name: " << erf.job_name << "\n\n";
  jobInfoFile << "inputs file: " << inputs_name << "\n\n";

  jobInfoFile << "number of MPI processes: "
              << amrex::ParallelDescriptor::NProcs() << "\n";
#ifdef _OPENMP
  jobInfoFile << "number of threads:       " << omp_get_max_threads() << "\n";
#endif

  jobInfoFile << "\n";
  jobInfoFile << "CPU time used since start of simulation (CPU-hours): "
              << erf.getCPUTime() / 3600.0;

  jobInfoFile << "\n\n";

  // plotfile information
  jobInfoFile << PrettyLine;
  jobInfoFile << " Plotfile Information\n";
  jobInfoFile << PrettyLine;

  time_t now = time(0);

  // Convert now to tm struct for local timezone
  tm* localtm = localtime(&now);
  jobInfoFile << "output data / time: " << asctime(localtm);

  std::string currentDir = amrex::FileSystem::CurrentPath();
  jobInfoFile << "output dir:         " << currentDir << "\n";

  jobInfoFile << "\n\n";

  // build information
  jobInfoFile << PrettyLine;
  jobInfoFile << " Build Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << "build date:    " << amrex::buildInfoGetBuildDate() << "\n";
  jobInfoFile << "build machine: " << amrex::buildInfoGetBuildMachine() << "\n";
  jobInfoFile << "build dir:     " << amrex::buildInfoGetBuildDir() << "\n";
  jobInfoFile << "AMReX dir:     " << amrex::buildInfoGetAMReXDir() << "\n";

  jobInfoFile << "\n";

  jobInfoFile << "COMP:          " << amrex::buildInfoGetComp() << "\n";
  jobInfoFile << "COMP version:  " << amrex::buildInfoGetCompVersion() << "\n";
  jobInfoFile << "FCOMP:         " << amrex::buildInfoGetFcomp() << "\n";
  jobInfoFile << "FCOMP version: " << amrex::buildInfoGetFcompVersion() << "\n";

  jobInfoFile << "\n";

  for (int n = 1; n <= amrex::buildInfoGetNumModules(); n++) {
    jobInfoFile << amrex::buildInfoGetModuleName(n) << ": "
                << amrex::buildInfoGetModuleVal(n) << "\n";
  }

  jobInfoFile << "\n";

  const char* githash1 = amrex::buildInfoGetGitHash(1);
  const char* githash2 = amrex::buildInfoGetGitHash(2);
  if (strlen(githash1) > 0) {
    jobInfoFile << "ERF       git hash: " << githash1 << "\n";
  }
  if (strlen(githash2) > 0) {
    jobInfoFile << "AMReX       git hash: " << githash2 << "\n";
  }

  const char* buildgithash = amrex::buildInfoGetBuildGitHash();
  const char* buildgitname = amrex::buildInfoGetBuildGitName();
  if (strlen(buildgithash) > 0) {
    jobInfoFile << buildgitname << " git hash: " << buildgithash << "\n";
  }

  jobInfoFile << "\n\n";

  // grid information
  jobInfoFile << PrettyLine;
  jobInfoFile << " Grid Information\n";
  jobInfoFile << PrettyLine;

  int f_lev = erf.parent->finestLevel();

  for (int i = 0; i <= f_lev; i++) {
    jobInfoFile << " level: " << i << "\n";
    jobInfoFile << "   number of boxes = " << erf.parent->numGrids(i) << "\n";
    jobInfoFile << "   maximum zones   = ";
    for (int n = 0; n < AMREX_SPACEDIM; n++) {
      jobInfoFile << erf.parent->Geom(i).Domain().length(n) << " ";
      // jobInfoFile << parent->Geom(i).ProbHi(n) << " ";
    }
    jobInfoFile << "\n\n";
  }

  jobInfoFile << " Boundary conditions\n";
  amrex::Vector<std::string> lo_bc_out(AMREX_SPACEDIM);
  amrex::Vector<std::string> hi_bc_out(AMREX_SPACEDIM);
  amrex::ParmParse pp("erf");
  pp.getarr("lo_bc", lo_bc_out, 0, AMREX_SPACEDIM);
  pp.getarr("hi_bc", hi_bc_out, 0, AMREX_SPACEDIM);

  // these names correspond to the integer flags setup in the
  // Setup.cpp

  jobInfoFile << "   -x: " << lo_bc_out[0] << "\n";
  jobInfoFile << "   +x: " << hi_bc_out[0] << "\n";
  jobInfoFile << "   -y: " << lo_bc_out[1] << "\n";
  jobInfoFile << "   +y: " << hi_bc_out[1] << "\n";
  jobInfoFile << "   -z: " << lo_bc_out[2] << "\n";
  jobInfoFile << "   +z: " << hi_bc_out[2] << "\n";

  jobInfoFile << "\n\n";

  int mlen = 20;

  jobInfoFile << PrettyLine;
  jobInfoFile << " Species Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << std::setw(6) << "index" << SkipSpace << std::setw(mlen + 1)
              << "name" << SkipSpace << std::setw(7) << "A" << SkipSpace
              << std::setw(7) << "Z"
              << "\n";
  jobInfoFile << OtherLine;
  jobInfoFile << "\n\n";

  // runtime parameters
  jobInfoFile << PrettyLine;
  jobInfoFile << " Inputs File Parameters\n";
  jobInfoFile << PrettyLine;

  amrex::ParmParse::dumpTable(jobInfoFile, true);
  jobInfoFile.close();
}

/*
 * ERF::writeBuildInfo
 * Similar to writeJobInfo, but the subset of information that makes sense
 * without an input file to enable --describe in format similar to CASTRO
 *
 */

void
IOManager::writeBuildInfo(std::ostream& os)
{
  std::string PrettyLine = std::string(78, '=') + "\n";
  std::string OtherLine = std::string(78, '-') + "\n";
  std::string SkipSpace = std::string(8, ' ');

  // build information
  os << PrettyLine;
  os << " ERF Build Information\n";
  os << PrettyLine;

  os << "build date:    " << amrex::buildInfoGetBuildDate() << "\n";
  os << "build machine: " << amrex::buildInfoGetBuildMachine() << "\n";
  os << "build dir:     " << amrex::buildInfoGetBuildDir() << "\n";
  os << "AMReX dir:     " << amrex::buildInfoGetAMReXDir() << "\n";

  os << "\n";

  os << "COMP:          " << amrex::buildInfoGetComp() << "\n";
  os << "COMP version:  " << amrex::buildInfoGetCompVersion() << "\n";

  os << "C++ compiler:  " << amrex::buildInfoGetCXXName() << "\n";
  os << "C++ flags:     " << amrex::buildInfoGetCXXFlags() << "\n";

  os << "\n";

  os << "Link flags:    " << amrex::buildInfoGetLinkFlags() << "\n";
  os << "Libraries:     " << amrex::buildInfoGetLibraries() << "\n";

  os << "\n";

  for (int n = 1; n <= amrex::buildInfoGetNumModules(); n++) {
    os << amrex::buildInfoGetModuleName(n) << ": "
       << amrex::buildInfoGetModuleVal(n) << "\n";
  }

  os << "\n";
  const char* githash1 = amrex::buildInfoGetGitHash(1);
  const char* githash2 = amrex::buildInfoGetGitHash(2);
  if (strlen(githash1) > 0) {
    os << "ERF       git hash: " << githash1 << "\n";
  }
  if (strlen(githash2) > 0) {
    os << "AMReX       git hash: " << githash2 << "\n";
  }

  const char* buildgithash = amrex::buildInfoGetBuildGitHash();
  const char* buildgitname = amrex::buildInfoGetBuildGitName();
  if (strlen(buildgithash) > 0) {
    os << buildgitname << " git hash: " << buildgithash << "\n";
  }

  os << "\n";
  os << " ERF Compile time variables: \n";

  os << "\n";
  os << " ERF Defines: \n";
#ifdef _OPENMP
  os << std::setw(35) << std::left << "_OPENMP " << std::setw(6) << "ON"
     << std::endl;
#else
  os << std::setw(35) << std::left << "_OPENMP " << std::setw(6) << "OFF"
     << std::endl;
#endif

#ifdef MPI_VERSION
  os << std::setw(35) << std::left << "MPI_VERSION " << std::setw(6)
     << MPI_VERSION << std::endl;
#else
  os << std::setw(35) << std::left << "MPI_VERSION " << std::setw(6)
     << "UNDEFINED" << std::endl;
#endif

#ifdef MPI_SUBVERSION
  os << std::setw(35) << std::left << "MPI_SUBVERSION " << std::setw(6)
     << MPI_SUBVERSION << std::endl;
#else
  os << std::setw(35) << std::left << "MPI_SUBVERSION " << std::setw(6)
     << "UNDEFINED" << std::endl;
#endif

#ifdef NUM_ADV
  os << std::setw(35) << std::left << "NUM_ADV=" << NUM_ADV << std::endl;
#else
  os << std::setw(35) << std::left << "NUM_ADV"
     << "is undefined (0)" << std::endl;
#endif

  os << "\n\n";
}

void
IOManager::writePlotFile(const std::string& dir, std::ostream& os, amrex::VisMF::How how)
{
std::cout << "writePlotFile_00" << std::endl;
  int i, n;
  //
  // The list of indices of State to write to plotfile.
  // first component of pair is state_type,
  // second component of pair is component # within the state_type
  //
  amrex::Vector<std::pair<int, int>> plot_var_map;
  for (int typ = 0; typ < erf.desc_lst.size(); typ++)
    for (int comp = 0; comp < erf.desc_lst[typ].nComp(); comp++)
      if (
        erf.parent->isStatePlotVar(erf.desc_lst[typ].name(comp)) &&
        erf.desc_lst[typ].getType() == amrex::IndexType::TheCellType())
        plot_var_map.push_back(std::pair<int, int>(typ, comp));

  int num_derive = 0;
  std::list<std::string> derive_names;
  const std::list<amrex::DeriveRec>& dlist = erf.derive_lst.dlist();

  for (std::list<amrex::DeriveRec>::const_iterator it = dlist.begin(),
                                                   end = dlist.end();
       it != end; ++it)
  {
    if (erf.parent->isDerivePlotVar(it->name())) {
      {
        derive_names.push_back(it->name());
        num_derive += it->numDerive();
      }
    }
  }

  int n_data_items = plot_var_map.size() + num_derive;

  amrex::Real cur_time = erf.state[State_Type].curTime();

  if (erf.level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    //
    // The first thing we write out is the plotfile type.
    //
    os << erf.thePlotFileType() << '\n';

    if (n_data_items == 0)
      amrex::Error("Must specify at least one valid data item to plot");

    os << n_data_items << '\n';

    //
    // Names of variables -- first state, then derived
    //
    for (i = 0; i < plot_var_map.size(); i++) {
      int typ = plot_var_map[i].first;
      int comp = plot_var_map[i].second;
      os << erf.desc_lst[typ].name(comp) << '\n';
    }

    for (std::list<std::string>::const_iterator it = derive_names.begin(),
                                                end = derive_names.end();
         it != end; ++it) {
      const amrex::DeriveRec* rec = erf.derive_lst.get(*it);
      for (i = 0; i < rec->numDerive(); i++)
        os << rec->variableName(i) << '\n';
    }

    os << AMREX_SPACEDIM << '\n';
    os << erf.parent->cumTime() << '\n';
    int f_lev = erf.parent->finestLevel();
    os << f_lev << '\n';
    for (i = 0; i < AMREX_SPACEDIM; i++)
      os << amrex::DefaultGeometry().ProbLo(i) << ' ';
    os << '\n';
    for (i = 0; i < AMREX_SPACEDIM; i++)
      os << amrex::DefaultGeometry().ProbHi(i) << ' ';
    os << '\n';
    for (i = 0; i < f_lev; i++)
      os << erf.parent->refRatio(i)[0] << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++)
      os << erf.parent->Geom(i).Domain() << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++)
      os << erf.parent->levelSteps(i) << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++) {
      for (int k = 0; k < AMREX_SPACEDIM; k++)
        os << erf.parent->Geom(i).CellSize()[k] << ' ';
      os << '\n';
    }
    os << (int)amrex::DefaultGeometry().Coord() << '\n';
    os << "0\n"; // Write bndry data.

    writeJobInfo(dir);
  }
  // Build the directory to hold the MultiFab at this level.
  // The name is relative to the directory containing the Header file.
  //
  static const std::string BaseName = "/Cell";
  char buf[64];
  sprintf(buf, "Level_%d", erf.level);
  std::string LevelStr = buf;
  //
  // Now for the full pathname of that directory.
  //
  std::string FullPath = dir;
  if (!FullPath.empty() && FullPath[FullPath.size() - 1] != '/')
    FullPath += '/';
  FullPath += LevelStr;
  //
  // Only the I/O processor makes the directory if it doesn't already exist.
  //
  if (amrex::ParallelDescriptor::IOProcessor())
    if (!amrex::UtilCreateDirectory(FullPath, 0755))
      amrex::CreateDirectoryFailed(FullPath);
  //
  // Force other processors to wait till directory is built.
  //
  amrex::ParallelDescriptor::Barrier();

  if (amrex::ParallelDescriptor::IOProcessor()) {
    os << erf.level << ' ' << erf.grids.size() << ' ' << cur_time << '\n';
    os << erf.parent->levelSteps(erf.level) << '\n';

    for (i = 0; i < erf.grids.size(); ++i) {
      amrex::RealBox gridloc =
        amrex::RealBox(erf.grids[i], erf.geom.CellSize(), erf.geom.ProbLo());
      for (n = 0; n < AMREX_SPACEDIM; n++)
        os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
    }
    //
    // The full relative pathname of the MultiFabs at this level.
    // The name is relative to the Header file containing this name.
    // It's the name that gets written into the Header.
    //
    if (n_data_items > 0) {
      std::string PathNameInHeader = LevelStr;
      PathNameInHeader += BaseName;
      os << PathNameInHeader << '\n';
    }
  }
  //
  // We combine all of the multifabs -- state, derived, etc -- into one
  // multifab -- plotMF.
  // NOTE: we are assuming that each state variable has one component,
  // but a derived variable is allowed to have multiple components.

  int cnt = 0;
  int ncomp = 1;
  const int nGrow = 0;
  amrex::MultiFab plotMF(
    erf.grids, erf.dmap, n_data_items, nGrow, amrex::MFInfo(), erf.Factory());
  amrex::MultiFab* this_dat = 0;
  //
  // Cull data from state variables -- use no ghost cells.
  //
  for (i = 0; i < plot_var_map.size(); i++) {
    int typ = plot_var_map[i].first;
    int comp = plot_var_map[i].second;
    this_dat = &(erf.state[typ].newData());
    amrex::MultiFab::Copy(plotMF, *this_dat, comp, cnt, 1, nGrow);
    cnt++;
  }
  //
  // Cull data from derived variables.
  //
  if (derive_names.size() > 0) {

    for (std::list<std::string>::const_iterator it = derive_names.begin(),
                                                end = derive_names.end();
         it != end; ++it) {
      const amrex::DeriveRec* rec = erf.derive_lst.get(*it);
      ncomp = rec->numDerive();

      auto derive_dat = erf.derive(*it, cur_time, nGrow);
      amrex::MultiFab::Copy(plotMF, *derive_dat, 0, cnt, ncomp, nGrow);
      cnt += ncomp;
    }
  }

  //
  // Use the Full pathname when naming the MultiFab.
  //
  std::string TheFullPath = FullPath;
  TheFullPath += BaseName;
  amrex::VisMF::Write(plotMF, TheFullPath, how, true);
}

void
IOManager::writeSmallPlotFile(
  const std::string& dir, ostream& os, amrex::VisMF::How how)
{
  int i, n;
  //
  // The list of indices of State to write to plotfile.
  // first component of pair is state_type,
  // second component of pair is component # within the state_type
  //
  amrex::Vector<std::pair<int, int>> plot_var_map;
  for (int typ = 0; typ < erf.desc_lst.size(); typ++)
    for (int comp = 0; comp < erf.desc_lst[typ].nComp(); comp++)
      if (
        erf.parent->isStateSmallPlotVar(erf.desc_lst[typ].name(comp)) &&
        erf.desc_lst[typ].getType() == amrex::IndexType::TheCellType())
        plot_var_map.push_back(std::pair<int, int>(typ, comp));

  int n_data_items = plot_var_map.size();

  amrex::Real cur_time = erf.state[State_Type].curTime();

  if (erf.level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    //
    // The first thing we write out is the plotfile type.
    //
    os << erf.thePlotFileType() << '\n';

    if (n_data_items == 0)
      amrex::Error("Must specify at least one valid data item to plot");

    os << n_data_items << '\n';

    //
    // Names of variables -- first state, then derived
    //
    for (i = 0; i < plot_var_map.size(); i++) {
      int typ = plot_var_map[i].first;
      int comp = plot_var_map[i].second;
      os << erf.desc_lst[typ].name(comp) << '\n';
    }

    os << AMREX_SPACEDIM << '\n';
    os << erf.parent->cumTime() << '\n';
    int f_lev = erf.parent->finestLevel();
    os << f_lev << '\n';
    for (i = 0; i < AMREX_SPACEDIM; i++)
      os << amrex::DefaultGeometry().ProbLo(i) << ' ';
    os << '\n';
    for (i = 0; i < AMREX_SPACEDIM; i++)
      os << amrex::DefaultGeometry().ProbHi(i) << ' ';
    os << '\n';
    for (i = 0; i < f_lev; i++)
      os << erf.parent->refRatio(i)[0] << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++)
      os << erf.parent->Geom(i).Domain() << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++)
      os << erf.parent->levelSteps(i) << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++) {
      for (int k = 0; k < AMREX_SPACEDIM; k++)
        os << erf.parent->Geom(i).CellSize()[k] << ' ';
      os << '\n';
    }
    os << (int)amrex::DefaultGeometry().Coord() << '\n';
    os << "0\n"; // Write bndry data.

    // job_info file with details about the run
    writeJobInfo(dir);
  }
  // Build the directory to hold the MultiFab at this level.
  // The name is relative to the directory containing the Header file.
  //
  static const std::string BaseName = "/Cell";
  char buf[64];
  sprintf(buf, "Level_%d", erf.level);
  std::string LevelStr = buf;
  //
  // Now for the full pathname of that directory.
  //
  std::string FullPath = dir;
  if (!FullPath.empty() && FullPath[FullPath.size() - 1] != '/')
    FullPath += '/';
  FullPath += LevelStr;
  //
  // Only the I/O processor makes the directory if it doesn't already exist.
  //
  if (amrex::ParallelDescriptor::IOProcessor())
    if (!amrex::UtilCreateDirectory(FullPath, 0755))
      amrex::CreateDirectoryFailed(FullPath);
  //
  // Force other processors to wait till directory is built.
  //
  amrex::ParallelDescriptor::Barrier();

  if (amrex::ParallelDescriptor::IOProcessor()) {
    os << erf.level << ' ' << erf.grids.size() << ' ' << cur_time << '\n';
    os << erf.parent->levelSteps(erf.level) << '\n';

    for (i = 0; i < erf.grids.size(); ++i) {
      amrex::RealBox gridloc =
        amrex::RealBox(erf.grids[i], erf.geom.CellSize(), erf.geom.ProbLo());
      for (n = 0; n < AMREX_SPACEDIM; n++)
        os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
    }
    //
    // The full relative pathname of the MultiFabs at this level.
    // The name is relative to the Header file containing this name.
    // It's the name that gets written into the Header.
    //
    if (n_data_items > 0) {
      std::string PathNameInHeader = LevelStr;
      PathNameInHeader += BaseName;
      os << PathNameInHeader << '\n';
    }
    os << vfraceps << '\n';
  }
  //
  // We combine all of the multifabs -- state, derived, etc -- into one
  // multifab -- plotMF.
  // NOTE: we are assuming that each state variable has one component,
  // but a derived variable is allowed to have multiple components.
  int cnt = 0;
  const int nGrow = 0;
  amrex::MultiFab plotMF(
    erf.grids, erf.dmap, n_data_items, nGrow, amrex::MFInfo(), erf.Factory());
  amrex::MultiFab* this_dat = 0;
  //
  // Cull data from state variables -- use no ghost cells.
  //
  for (i = 0; i < plot_var_map.size(); i++) {
    int typ = plot_var_map[i].first;
    int comp = plot_var_map[i].second;
    this_dat = &(erf.state[typ].newData());
    amrex::MultiFab::Copy(plotMF, *this_dat, comp, cnt, 1, nGrow);
    cnt++;
  }

  //
  // Use the Full pathname when naming the MultiFab.
  //
  std::string TheFullPath = FullPath;
  TheFullPath += BaseName;
  amrex::VisMF::Write(plotMF, TheFullPath, how, true);
}
