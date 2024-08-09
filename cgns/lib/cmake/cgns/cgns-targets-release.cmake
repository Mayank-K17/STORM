#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CGNS::cgns_static" for configuration "Release"
set_property(TARGET CGNS::cgns_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(CGNS::cgns_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcgns.a"
  )

list(APPEND _cmake_import_check_targets CGNS::cgns_static )
list(APPEND _cmake_import_check_files_for_CGNS::cgns_static "${_IMPORT_PREFIX}/lib/libcgns.a" )

# Import target "CGNS::cgns_shared" for configuration "Release"
set_property(TARGET CGNS::cgns_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(CGNS::cgns_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcgns.so.4.4"
  IMPORTED_SONAME_RELEASE "libcgns.so.4.4"
  )

list(APPEND _cmake_import_check_targets CGNS::cgns_shared )
list(APPEND _cmake_import_check_files_for_CGNS::cgns_shared "${_IMPORT_PREFIX}/lib/libcgns.so.4.4" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
