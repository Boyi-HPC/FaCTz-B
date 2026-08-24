set(
  CUCPSZ_NVCOMP_ROOT ""
  CACHE PATH "Root of an nvCOMP installation (contains include/ and lib64/)"
)

set(_CUCPSZ_NVCOMP_HINTS)
if(CUCPSZ_NVCOMP_ROOT)
  list(APPEND _CUCPSZ_NVCOMP_HINTS "${CUCPSZ_NVCOMP_ROOT}")
endif()
if(DEFINED ENV{NVCOMP_ROOT})
  list(APPEND _CUCPSZ_NVCOMP_HINTS "$ENV{NVCOMP_ROOT}")
endif()
if(DEFINED ENV{HOME})
  list(
    APPEND _CUCPSZ_NVCOMP_HINTS
    "$ENV{HOME}/.local/nvcomp-cu12/nvidia/libnvcomp"
  )
endif()

find_path(
  nvcomp_INCLUDE_DIR
  NAMES nvcomp/ans.hpp
  HINTS ${_CUCPSZ_NVCOMP_HINTS}
  PATH_SUFFIXES include
)

find_library(
  nvcomp_LIBRARY
  NAMES nvcomp libnvcomp.so.5
  HINTS ${_CUCPSZ_NVCOMP_HINTS}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  nvcomp
  REQUIRED_VARS nvcomp_INCLUDE_DIR nvcomp_LIBRARY
)

if(nvcomp_FOUND AND NOT TARGET nvcomp::nvcomp)
  add_library(nvcomp::nvcomp SHARED IMPORTED)
  set_target_properties(
    nvcomp::nvcomp
    PROPERTIES
      IMPORTED_LOCATION "${nvcomp_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${nvcomp_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(nvcomp_INCLUDE_DIR nvcomp_LIBRARY)
