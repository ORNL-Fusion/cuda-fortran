#[==[
Provides the following variables:
  * `CudaInterface::CudaInterface`: A target to use with `target_link_libraries`.
#]==]

find_package (CUDAToolkit REQUIRED)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (CudaInterface
                                   REQUIRED_VARS CUDAToolkit_FOUND)

if (NOT TARGET CudaInterface::CudaInterface)
    message (STATUS "Setup CudaInterface")

    add_library (CudaInterface::CudaInterface INTERFACE IMPORTED)

    target_link_libraries (CudaInterface::CudaInterface

                           INTERFACE

                           CUDA::cusolver
                           CUDA::cublasLt
    )

    target_compile_options (CudaInterface::CudaInterface

                            INTERFACE

                            $<$<COMPILE_LANGUAGE:Fortran>:-cpp>
    )

    target_compile_definitions (CudaInterface::CudaInterface

                                INTERFACE

                                cusolverStatus_t=INTEGER\(C_SIZE_T\)
                                cusolverDnHandle_t=TYPE\(C_PTR\)
                                cusolverDnParams_t=TYPE\(C_PTR\)
                                cudaDataType=INTEGER\(C_SIZE_T\)
                                cublasStatus_t=INTEGER\(C_SIZE_T\)
                                cublasLtHandle_t=TYPE\(C_PTR\)
     )
endif ()

