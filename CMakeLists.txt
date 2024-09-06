cmake_minimum_required (VERSION 3.22)

project (cuda-fortran Fortran)

#-------------------------------------------------------------------------------
#  Set the cmake module path.
#-------------------------------------------------------------------------------
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#-------------------------------------------------------------------------------
#  Setup build types
#-------------------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif ()
set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              Debug
              Release
              MinSizeRel
              RelWithDebInfo
)

#-------------------------------------------------------------------------------
#  Find dependencies
#-------------------------------------------------------------------------------
find_package (CudaInterface REQUIRED)

#-------------------------------------------------------------------------------
#  Setup targets
#-------------------------------------------------------------------------------
add_subdirectory (source)
