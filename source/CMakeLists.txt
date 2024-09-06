add_library(cuda_fortran)

target_sources (cuda_fortran

                PRIVATE

                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/assert.f90>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda.f90>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cusolver.f90>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cublas.f90>
)

target_link_libraries (cuda_fortran

                       PUBLIC

                       CudaInterface::CudaInterface
)

target_include_directories (cuda_fortran

                            PUBLIC

                            $<TARGET_PROPERTY:cuda_fortran,BINARY_DIR>
)
