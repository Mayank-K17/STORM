# FindCGNS.cmake
find_path(/cosma/home/do009/dc-kuma4/STORM/Version1/CGNS/build/src/ cgnslib.h
  HINTS ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES include
)

find_library(CGNS_LIBRARY NAMES cgns
  HINTS ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(CGNS DEFAULT_MSG
  CGNS_LIBRARY CGNS_INCLUDE_DIR)

mark_as_advanced(CGNS_INCLUDE_DIR CGNS_LIBRARY)

set(CGNS_LIBRARIES ${CGNS_LIBRARY})
set(CGNS_INCLUDE_DIRS ${CGNS_INCLUDE_DIR})

