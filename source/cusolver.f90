!*******************************************************************************
!>  @file cusolver.f90
!>  @brief Contains module @ref cuSOLVER.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  Defines the interface for cusolver library.
!*******************************************************************************

      MODULE cuSOLVER

      USE cudaASSERT
      USE cuda
      USE, INTRINSIC :: iso_c_binding

      IMPLICIT NONE

!*******************************************************************************
!  Interface binding for the cusolver functions
!*******************************************************************************
      INTERFACE
!-------------------------------------------------------------------------------
!>  @brief Create a Dense cusolver handle.
!>
!>  @params[out] handle Handle to the dense cusolver.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         cusolverStatus_t FUNCTION cusolverDnCreate_f(handle)                  &
         BIND(C, NAME='cusolverDnCreate')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cusolverDnHandle_t, INTENT(out) :: handle

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Destroy a Dense cusolver handle.
!>
!>  @params[in,out] handle Handle to the dense cusolver.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         cusolverStatus_t FUNCTION cusolverDnDestroy_f(handle)                 &
         BIND(C, NAME='cusolverDnDestroy')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cusolverDnHandle_t, VALUE :: handle

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create a Dense cusolver params object.
!>
!>  @params[out] params Dense cusolver parameter object.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         cusolverStatus_t FUNCTION cusolverDnCreateParams_f(params)            &
         BIND(C, NAME='cusolverDnCreateParams')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cusolverDnParams_t, INTENT(out) :: params

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Destroy a Dense cusolver params object.
!>
!>  @params[in,out] params Dense cusolver parameter object.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         cusolverStatus_t FUNCTION cusolverDnDestroyParams_f(params)           &
         BIND(C, NAME='cusolverDnDestroyParams')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cusolverDnParams_t, VALUE :: params

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief LU factorization of a mxn matrix
!>
!>  @params[in]     handle                   Handle to the dense cusolver.
!>  @params[in]     params                   Dense cusolver parameter object.
!>  @params[in]     m                        Num rows.
!>  @params[in]     n                        Num columns.
!>  @params[in]     dataTypeA                Datatype for matrix A.
!>  @params[in,out] A                        Matrix for A.
!>  @params[in]     lda                      Leading order of A.
!>  @params[out]    ipiv                     Pivot indices.
!>  @params[in]     computeType              Datatype for the compute.
!>  @params[in,out] bufferOnDevice           Device workspace.
!>  @params[in]     workspaceInBytesOnDevice Size of buffer on device.
!>  @params[in,out] bufferOnHost             Host workspace.
!>  @params[in]     workspaceInBytesOnHost   Size of buffer on host.
!>  @params[out]    info                     Error status of the solve.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         cusolverStatus_t FUNCTION cusolverDnXgetrf_f(handle,                   &
                                                      params,                   &
                                                      m,                        &
                                                      n,                        &
                                                      dataTypeA,                &
                                                      A,                        &
                                                      lda,                      &
                                                      ipiv,                     &
                                                      computeType,              &
                                                      bufferOnDevice,           &
                                                      workspaceInBytesOnDevice, &
                                                      bufferOnHost,             &
                                                      workspaceInBytesOnHost,   &
                                                      info)                     &
         BIND(C, NAME='cusolverDnXgetrf')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         cusolverDnHandle_t, VALUE, INTENT(in)  :: handle
         cusolverDnParams_t, VALUE, INTENT(in)  :: params
         INTEGER (C_INT64_T), VALUE, INTENT(in) :: m
         INTEGER (C_INT64_T), VALUE, INTENT(in) :: n
         cudaDataType, VALUE, INTENT(in)        :: dataTypeA
         TYPE (C_PTR), VALUE                    :: A
         INTEGER (C_INT64_T), VALUE, INTENT(in) :: lda
         INTEGER (C_INT64_T), INTENT(out)       :: ipiv
         cudaDataType, VALUE, INTENT(in)        :: computeType
         TYPE (C_PTR), VALUE                    :: bufferOnDevice
         INTEGER (C_SIZE_T), VALUE, INTENT(in)  :: workspaceInBytesOnDevice
         TYPE (C_PTR), INTENT(inout)            :: bufferOnHost
         INTEGER (C_SIZE_T), VALUE, INTENT(in)  :: workspaceInBytesOnHost
         INTEGER (C_INT), INTENT(out)           :: info

         END FUNCTION
      END INTERFACE

!*******************************************************************************
!  Enumeration Types
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Return status states for cusolver functions.
!-------------------------------------------------------------------------------
      ENUM, BIND(C)
         ENUMERATOR :: CUSOLVER_STATUS_SUCCESS                                 = 0
         ENUMERATOR :: CUSOLVER_STATUS_NOT_INITIALIZED                         = 1
         ENUMERATOR :: CUSOLVER_STATUS_ALLOC_FAILED                            = 2
         ENUMERATOR :: CUSOLVER_STATUS_INVALID_VALUE                           = 3
         ENUMERATOR :: CUSOLVER_STATUS_ARCH_MISMATCH                           = 4
         ENUMERATOR :: CUSOLVER_STATUS_MAPPING_ERROR                           = 5
         ENUMERATOR :: CUSOLVER_STATUS_EXECUTION_FAILED                        = 6
         ENUMERATOR :: CUSOLVER_STATUS_INTERNAL_ERROR                          = 7
         ENUMERATOR :: CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED               = 8
         ENUMERATOR :: CUSOLVER_STATUS_NOT_SUPPORTED                           = 9
         ENUMERATOR :: CUSOLVER_STATUS_ZERO_PIVOT                              = 10
         ENUMERATOR :: CUSOLVER_STATUS_INVALID_LICENSE                         = 11
         ENUMERATOR :: CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED              = 12
         ENUMERATOR :: CUSOLVER_STATUS_IRS_PARAMS_INVALID                      = 13
         ENUMERATOR :: CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC                 = 14
         ENUMERATOR :: CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE               = 15
         ENUMERATOR :: CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER              = 16
         ENUMERATOR :: CUSOLVER_STATUS_IRS_INTERNAL_ERROR                      = 20
         ENUMERATOR :: CUSOLVER_STATUS_IRS_NOT_SUPPORTED                       = 21
         ENUMERATOR :: CUSOLVER_STATUS_IRS_OUT_OF_RANGE                        = 22
         ENUMERATOR :: CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23
         ENUMERATOR :: CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED               = 25
         ENUMERATOR :: CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED                 = 26
         ENUMERATOR :: CUSOLVER_STATUS_IRS_MATRIX_SINGULAR                     = 30
         ENUMERATOR :: CUSOLVER_STATUS_INVALID_WORKSPACE                       = 31
      END ENUM

      CONTAINS

!-------------------------------------------------------------------------------
!>  @brief Report status if there was an error and exit.
!>
!>  @params[in] status Status state for a cusolver function.
!-------------------------------------------------------------------------------
      SUBROUTINE cusolver_checkerror(status)

      IMPLICIT NONE

      cusolverStatus_t :: status

      SELECT CASE (status)
         CASE (CUSOLVER_STATUS_SUCCESS)
            RETURN

         CASE (CUSOLVER_STATUS_NOT_INITIALIZED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_NOT_INITIALIZED')

         CASE (CUSOLVER_STATUS_ALLOC_FAILED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_ALLOC_FAILED')

         CASE (CUSOLVER_STATUS_INVALID_VALUE)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_INVALID_VALUE')

         CASE (CUSOLVER_STATUS_ARCH_MISMATCH)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_ARCH_MISMATCH')

         CASE (CUSOLVER_STATUS_MAPPING_ERROR)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_MAPPING_ERROR')

         CASE (CUSOLVER_STATUS_EXECUTION_FAILED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_EXECUTION_FAILED')

         CASE (CUSOLVER_STATUS_INTERNAL_ERROR)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_INTERNAL_ERROR')

         CASE (CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED')

         CASE (CUSOLVER_STATUS_NOT_SUPPORTED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_NOT_SUPPORTED')

         CASE (CUSOLVER_STATUS_ZERO_PIVOT)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_ZERO_PIVOT')

         CASE (CUSOLVER_STATUS_INVALID_LICENSE)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_INVALID_LICENSE')

         CASE (CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED')

         CASE (CUSOLVER_STATUS_IRS_PARAMS_INVALID)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_PARAMS_INVALID')

         CASE (CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC')

         CASE (CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE')

         CASE (CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER')

         CASE (CUSOLVER_STATUS_IRS_INTERNAL_ERROR)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_INTERNAL_ERROR')

         CASE (CUSOLVER_STATUS_IRS_NOT_SUPPORTED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_NOT_SUPPORTED')

         CASE (CUSOLVER_STATUS_IRS_OUT_OF_RANGE)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_OUT_OF_RANGE')

         CASE (CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES')

         CASE (CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED')

         CASE (CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED')

         CASE (CUSOLVER_STATUS_IRS_MATRIX_SINGULAR)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_IRS_MATRIX_SINGULAR')

         CASE (CUSOLVER_STATUS_INVALID_WORKSPACE)
            CALL cuda_assert(status, 'CUSOLVER_STATUS_INVALID_WORKSPACE')

         CASE DEFAULT
            CALL cuda_assert(status, 'Undefined error')

      END SELECT

      END SUBROUTINE

      END MODULE
