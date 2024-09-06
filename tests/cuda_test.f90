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
      INTEGER   :: count
      CUdevice  :: device
      CUcontext :: context

!  Start of executable code.
      CALL cuda_checkerror(cuInit_f(0))
      CALL cuda_checkerror(cuDeviceGetCount_f(count))
      CALL cuda_checkerror(cuGetDevice_f(device, 0))
      CALL cuda_checkerror(cuCtxCreate_f(context, 0, device))
      CALL cuda_checkerror(cuCtxDestroy_f(context))

      END PROGRAM
