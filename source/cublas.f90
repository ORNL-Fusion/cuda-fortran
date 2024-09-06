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

      END MODULE
