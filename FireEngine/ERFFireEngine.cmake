# The LSFIRE (ELMFIRE-CPP) fire engine directory, for CMake. ERF includes this
# file when ERF_FIRE_ENGINE_DIRS lists this directory and calls
# erf_fire_engine_add with its object library (ERF_FIRE_ENGINE_DIR is this
# directory during the call).
#
#   -DELMFIRE_DIR=<LSFIRE source dir>   provides include/ELMFIRE_API.H and src/Core/FireHierarchy.H
#   -DELMFIRE_LIB=<libelmfire_core.a>   built against the same AMReX version as ERF
set(ELMFIRE_DIR "" CACHE PATH "LSFIRE (ELMFIRE-CPP_MPI) source directory (provides include/ELMFIRE_API.H)")
set(ELMFIRE_LIB "" CACHE FILEPATH "libelmfire_core.a built against the same AMReX version as ERF")
if(NOT EXISTS "${ELMFIRE_DIR}/include/ELMFIRE_API.H")
  message(FATAL_ERROR "The lsfire fire engine needs -DELMFIRE_DIR=<LSFIRE source dir> (got '${ELMFIRE_DIR}')")
endif()
if(NOT EXISTS "${ELMFIRE_LIB}")
  message(FATAL_ERROR "The lsfire fire engine needs -DELMFIRE_LIB=<path to libelmfire_core.a> (got '${ELMFIRE_LIB}')")
endif()

function(erf_fire_engine_add target)
  target_sources(${target} PRIVATE
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireELMFIRE.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireReadFuel.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireReadIgnition.cpp)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${ERF_FIRE_ENGINE_DIR}>
    $<BUILD_INTERFACE:${ELMFIRE_DIR}/include>
    $<BUILD_INTERFACE:${ELMFIRE_DIR}/src>)   # Core/FireHierarchy.H
  # LSFIRE's AMReX symbols are provided by ERF's own AMReX (same version)
  target_link_libraries(${target} PUBLIC ${ELMFIRE_LIB})
  message(STATUS "LSFIRE fire engine: ${ELMFIRE_LIB}")
endfunction()
