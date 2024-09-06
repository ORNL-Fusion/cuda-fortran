!*******************************************************************************
!>  @file cublas.f90
!>  @brief Contains module @ref cuBLAS.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  Defines the interface for cublas library.
!*******************************************************************************

      MODULE cuBLAS

      USE cudaASSERT
      USE, INTRINSIC :: iso_c_binding

      IMPLICIT NONE

!*******************************************************************************
!  Interface binding for the cusblas functions
!*******************************************************************************
      INTERFACE
!-------------------------------------------------------------------------------
!>  @brief Initialize the cublaslt library.
!>
!>  @params[out] handle Handle to the cublas library.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         cublasStatus_t FUNCTION cublasLtCreate_f(handle)                      &
         BIND(C, NAME='cublasLtCreate')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cublasLtHandle_t, INTENT(OUT) :: handle

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Destroy the cublaslt library.
!>
!>  @params[in] handle Handle to the cublas library.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         cublasStatus_t FUNCTION cublasLtDestroy_f(handle) &
         BIND(C, NAME='cublasLtDestroy')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cublasLtHandle_t, VALUE :: handle

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Compute the matrix multiplication of two matricies.
!>
!>  D = alpha*(A.B) + beta*C
!>
!>  @params[in]  handle        Handle to the cublas library.
!>  @params[in]  computeDesc   Compute discriptor for the maxtrix multiply.
!>  @params[in]  alpha         Alpha scale factor.
!>  @params[in]  A             Matrix A.
!>  @params[in]  Adesc         Discriptor for matrix A.
!>  @params[in]  B             Matrix B.
!>  @params[in]  Bdesc         Discriptor for matrix B
!>  @params[in]  beta          Beta scale factor.
!>  @params[in]  C             Matrix C.
!>  @params[in]  Cdesc         Discriptor for matrix C.
!>  @params[in]  D             Matrix D.
!>  @params[out] Ddesc         Discriptor for matrix D.
!>  @params[in]  algo          Agorthium to use.
!>  @params[in]  workSpace     Workspace buffer.
!>  @params[in]  workSpaceSize Size of workspace buffer in bytes.
!>  @params[in]  stream        Cuda stream to run the computation on.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         cublasStatus_t FUNCTION cublasLtMatmul_f(handle,                      &
                                                  computeDesc,                 &
                                                  alpha,                       &
                                                  A,                           &
                                                  Adesc,                       &
                                                  B,                           &
                                                  Bdesc,                       &
                                                  beta,                        &
                                                  C,                           &
                                                  Cdesc,                       &
                                                  D,                           &
                                                  Ddesc,                       &
                                                  algo,                        &
                                                  workSpace,                   &
                                                  workSpaceSize,               &
                                                  stream)                      &
         BIND(C, NAME='cublasLtMatmul')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cublasLtHandle_t, VALUE           :: handle
         cublasLtMatmulDesc_t, VALUE       :: computeDesc
         CUdeviceptr, VALUE                :: alpha
         CUdeviceptr, VALUE                :: A
         cublasLtMatrixLayout_t, VALUE     :: Adesc
         CUdeviceptr, VALUE                :: B
         cublasLtMatrixLayout_t, VALUE     :: Bdesc
         CUdeviceptr, VALUE                :: beta
         CUdeviceptr, VALUE                :: C
         cublasLtMatrixLayout_t, VALUE     :: Cdesc
         CUdeviceptr, INTENT(OUT)          :: D
         cublasLtMatrixLayout_t, VALUE     :: Ddesc
         cublasLtMatmulAlgo_t, INTENT(OUT) :: algo
         CUdeviceptr, VALUE                :: workSpace
         INTEGER(C_SIZE_T), INTENT(IN)     :: size
         CUstream                          :: stream

         END FUNCTION

      END INTERFACE

!*******************************************************************************
!  Enumeration Types
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Return status states for cusolver functions.
!-------------------------------------------------------------------------------
      ENUM, BIND(C)
         ENUMERATOR :: CUBLAS_STATUS_SUCCESS          = 0
         ENUMERATOR :: CUBLAS_STATUS_NOT_INITIALIZED  = 1
         ENUMERATOR :: CUBLAS_STATUS_ALLOC_FAILED     = 3
         ENUMERATOR :: CUBLAS_STATUS_INVALID_VALUE    = 7
         ENUMERATOR :: CUBLAS_STATUS_ARCH_MISMATCH    = 8
         ENUMERATOR :: CUBLAS_STATUS_MAPPING_ERROR    = 11
         ENUMERATOR :: CUBLAS_STATUS_EXECUTION_FAILED = 13
         ENUMERATOR :: CUBLAS_STATUS_INTERNAL_ERROR   = 14
         ENUMERATOR :: CUBLAS_STATUS_NOT_SUPPORTED    = 15
         ENUMERATOR :: CUBLAS_STATUS_LICENSE_ERROR    = 16
      END ENUM

      CONTAINS

!-------------------------------------------------------------------------------
!>  @brief Report status if there was an error and exit.
!>
!>  @params[in] status Status state for a cublas function.
!-------------------------------------------------------------------------------
      SUBROUTINE cublas_checkerror(status)

      IMPLICIT NONE

      cublasStatus_t :: status

      SELECT CASE (status)
         CASE (CUBLAS_STATUS_SUCCESS)
            RETURN

         CASE (CUBLAS_STATUS_NOT_INITIALIZED)
            CALL cuda_assert(status, 'CUBLAS_STATUS_NOT_INITIALIZED')

         CASE (CUBLAS_STATUS_ALLOC_FAILED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_ALLOC_FAILED')

         CASE (CUBLAS_STATUS_INVALID_VALUE)
            CALL cuda_assert(status, 'CUBLAS_STATUS_INVALID_VALUE')

         CASE (CUBLAS_STATUS_ARCH_MISMATCH)
            CALL cuda_assert(status, 'CCUBLAS_STATUS_ARCH_MISMATCH')

         CASE (CUBLAS_STATUS_MAPPING_ERROR)
            CALL cuda_assert(status, 'CUBLAS_STATUS_MAPPING_ERROR')

         CASE (CUBLAS_STATUS_EXECUTION_FAILED)
            CALL cuda_assert(status, 'CUBLAS_STATUS_EXECUTION_FAILED')

         CASE (CUBLAS_STATUS_INTERNAL_ERROR)
            CALL cuda_assert(status, 'CUBLAS_STATUS_INTERNAL_ERROR')

         CASE (CUBLAS_STATUS_NOT_SUPPORTED)
            CALL cuda_assert(status, 'CUBLAS_STATUS_NOT_SUPPORTED')

         CASE (CUBLAS_STATUS_LICENSE_ERROR)
            CALL cuda_assert(status, 'CUBLAS_STATUS_LICENSE_ERROR')

         CASE DEFAULT
            CALL cuda_assert(status, 'Undefined error')

      END SELECT

      END SUBROUTINE

      END MODULE
