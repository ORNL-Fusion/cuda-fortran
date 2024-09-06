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
         FUNCTION cublasLtCreate_f(handle) RESULT(cublasStatus_t)
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
         FUNCTION cublasLtDestroy_f(handle) RESULT(cublasStatus_t)
         BIND(C, NAME='cublasLtDestroy')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cublasLtHandle_t, INTENT(in) :: handle

         END FUNCTION

      END INTERFACE

      END MODULE
