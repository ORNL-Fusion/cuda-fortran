!*******************************************************************************
!>  @file cublas_test.f90
!>  @brief Contains test for cublas interface.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  Defines the cublas interface tests.
!*******************************************************************************
      PROGRAM cublas_test
      USE cuBLAS

      IMPLICIT NONE

!  Local variables.
      cublasLtHandle_t :: handle

!  Start of executable code.
      CALL cublas_checkerror(cublasLtCreate_f(handle))
      CALL cublas_checkerror(cublasLtDestroy_f(handle))

      END PROGRAM
