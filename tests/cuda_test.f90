!*******************************************************************************
!>  @file cuda_test.f90
!>  @brief Contains test for cublas interface.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  Defines the cuba interface tests.
!*******************************************************************************
      PROGRAM cuda_test
      USE cuda

      IMPLICIT NONE

!  Local variables.
      INTEGER :: count

!  Start of executable code.
      CALL cuda_checkerror(cuInit_f(0))
      CALL cuda_checkerror(cuDeviceGetCount_f(count))

      END PROGRAM
