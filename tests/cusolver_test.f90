!*******************************************************************************
!>  @file cusolver_test.f90
!>  @brief Contains test for cusolver interface.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  Defines the cusolver interface tests.
!*******************************************************************************
      PROGRAM cusolver_test
      USE cuSOLVER

      IMPLICIT NONE

!  Local variables.
      cusolverDnHandle_t :: handle

!  Start of executable code.
      CALL cusolver_checkerror(cusolverDnCreate_f(handle))
      CALL cusolver_checkerror(cusolverDnDestroy_f(handle))

      END PROGRAM
