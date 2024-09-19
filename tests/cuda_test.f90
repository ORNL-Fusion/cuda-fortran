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
      USE cuda_types
      USE cuda
      USE, INTRINSIC :: iso_fortran_env

      IMPLICIT NONE

!  Local variables.
      INTEGER     :: count
      CUdevice    :: device
      CUdeviceptr :: buffer
      CUcontext   :: context

!  Start of executable code.
      CALL cuda_checkerror(cuInit_f(0))
      CALL cuda_checkerror(cuDeviceGetCount_f(count))
      CALL cuda_checkerror(cuDeviceGet_f(device, 0))
      CALL cuda_checkerror(cuDevicePrimaryCtxRetain_f(context, device))
      CALL cuda_checkerror(cuCtxSetCurrent_f(context))

      CALL cuda_checkerror(cuMemAllocManaged_f(buffer, 800_int64, CU_MEM_ATTACH_GLOBAL))
      CALL cuda_checkerror(cuMemFree_f(buffer))

      CALL cuda_checkerror(cuDevicePrimaryCtxRelease_f(context))

      END PROGRAM
