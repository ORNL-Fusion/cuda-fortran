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

         cublasLtHandle_t, INTENT(out) :: handle

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

         cublasLtHandle_t, INTENT(in) :: handle

         END FUNCTION

      END INTERFACE

!*******************************************************************************
!  Enumeration Types
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Return status states for cusolver functions.
!-------------------------------------------------------------------------------
      ENUM, BIND(C)
         CUBLAS_STATUS_SUCCESS          = 0
         CUBLAS_STATUS_NOT_INITIALIZED  = 1
         CUBLAS_STATUS_ALLOC_FAILED     = 3
         CUBLAS_STATUS_INVALID_VALUE    = 7
         CUBLAS_STATUS_ARCH_MISMATCH    = 8
         CUBLAS_STATUS_MAPPING_ERROR    = 11
         CUBLAS_STATUS_EXECUTION_FAILED = 13
         CUBLAS_STATUS_INTERNAL_ERROR   = 14
         CUBLAS_STATUS_NOT_SUPPORTED    = 15
         CUBLAS_STATUS_LICENSE_ERROR    = 16
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

         CASE (CUSOLVER_STATUS_ALLOC_FAILED)
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
