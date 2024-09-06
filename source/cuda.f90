!*******************************************************************************
!>  @file cuda.f90
!>  @brief Contains module @ref cuda.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  Defines the interface for cuda library.
!*******************************************************************************

      MODULE cuda
      
      USE cudaASSERT
      USE, INTRINSIC :: iso_c_binding

      IMPLICIT NONE

!*******************************************************************************
!  Interface binding for the cusolver functions
!*******************************************************************************
      INTERFACE
!-------------------------------------------------------------------------------
!>  @brief Get the number of cuda devices.
!>
!>  @params[in] flags Number of cuda devices.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         CUresult FUNCTION cuInit_f(flags)                                     &
         BIND(C, NAME='cuInit')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         INTEGER(C_INT), VALUE, INTENT(IN) :: flags

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Get the number of cuda devices.
!>
!>  @params[out] count Number of cuda devices.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         CUresult FUNCTION cuDeviceGetCount_f(count)                           &
         BIND(C, NAME='cuDeviceGetCount')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         INTEGER(C_INT), INTENT(OUT) :: count

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Get cuda device.
!>
!>  @params[out] device Handle to the cuda device.
!>  @params[in]  index  Index of the cuda device.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         CUresult FUNCTION cuDeviceGet_f(device, index)                        &
         BIND(C, NAME='cuDeviceGet')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         CUdevice, INTENT(OUT) :: device
         INTEGER(C_INT)        :: index

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create a cuda context.
!>
!>  @params[out] context Handle to the cuda context.
!>  @params[in]  flags   Flags to construct the context with.
!>  @params[in]  device  THe cuda device.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         CUresult FUNCTION cuCtxCreate_f(context, flags, device)               &
         BIND(C, NAME='cuCtxCreate')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         CUcontext, INTENT(OUT) :: context
         INTEGER(C_INT)         :: flags
         CUdevice, VALUE        :: device

         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Destroy a cuda context.
!>
!>  @params[in] context Handle to the cuda context.
!>  @returns Error status.
!-------------------------------------------------------------------------------
         CUresult FUNCTION cuCtxDestroy_f(context)                             &
         BIND(C, NAME='cuCtxDestroy')
         USE, INTRINSIC :: iso_c_binding

         IMPLICIT NONE

         CUcontext, VALUE :: context

         END FUNCTION

      END INTERFACE

!*******************************************************************************
!  Enumeration Types
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Cuda data types.
!-------------------------------------------------------------------------------
      ENUM, BIND(C)
         ENUMERATOR :: CUDA_R_16F     =  2
         ENUMERATOR :: CUDA_C_16F     =  6
         ENUMERATOR :: CUDA_R_16BF    = 14
         ENUMERATOR :: CUDA_C_16BF    = 15
         ENUMERATOR :: CUDA_R_32F     =  0
         ENUMERATOR :: CUDA_C_32F     =  4
         ENUMERATOR :: CUDA_R_64F     =  1
         ENUMERATOR :: CUDA_C_64F     =  5
         ENUMERATOR :: CUDA_R_4I      = 16
         ENUMERATOR :: CUDA_C_4I      = 17
         ENUMERATOR :: CUDA_R_4U      = 18
         ENUMERATOR :: CUDA_C_4U      = 19
         ENUMERATOR :: CUDA_R_8I      =  3
         ENUMERATOR :: CUDA_C_8I      =  7
         ENUMERATOR :: CUDA_R_8U      =  8
         ENUMERATOR :: CUDA_C_8U      =  9
         ENUMERATOR :: CUDA_R_16I     = 20
         ENUMERATOR :: CUDA_C_16I     = 21
         ENUMERATOR :: CUDA_R_16U     = 22
         ENUMERATOR :: CUDA_C_16U     = 23
         ENUMERATOR :: CUDA_R_32I     = 10
         ENUMERATOR :: CUDA_C_32I     = 11
         ENUMERATOR :: CUDA_R_32U     = 12
         ENUMERATOR :: CUDA_C_32U     = 13
         ENUMERATOR :: CUDA_R_64I     = 24
         ENUMERATOR :: CUDA_C_64I     = 25
         ENUMERATOR :: CUDA_R_64U     = 26
         ENUMERATOR :: CUDA_C_64U     = 27
         ENUMERATOR :: CUDA_R_8F_E4M3 = 28
         ENUMERATOR :: CUDA_R_8F_E5M2 = 29
      END ENUM

!-------------------------------------------------------------------------------
!>  @brief Cuda error types.
!-------------------------------------------------------------------------------
      ENUM, BIND(C)
         ENUMERATOR :: CUDA_SUCCESS                              = 0
         ENUMERATOR :: CUDA_ERROR_INVALID_VALUE                  = 1
         ENUMERATOR :: CUDA_ERROR_OUT_OF_MEMORY                  = 2
         ENUMERATOR :: CUDA_ERROR_NOT_INITIALIZED                = 3
         ENUMERATOR :: CUDA_ERROR_DEINITIALIZED                  = 4
         ENUMERATOR :: CUDA_ERROR_PROFILER_DISABLED              = 5
         ENUMERATOR :: CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6
         ENUMERATOR :: CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7
         ENUMERATOR :: CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8
         ENUMERATOR :: CUDA_ERROR_STUB_LIBRARY                   = 34
         ENUMERATOR :: CUDA_ERROR_DEVICE_UNAVAILABLE             = 46
         ENUMERATOR :: CUDA_ERROR_NO_DEVICE                      = 100
         ENUMERATOR :: CUDA_ERROR_INVALID_DEVICE                 = 101
         ENUMERATOR :: CUDA_ERROR_DEVICE_NOT_LICENSED            = 102
         ENUMERATOR :: CUDA_ERROR_INVALID_IMAGE                  = 200
         ENUMERATOR :: CUDA_ERROR_INVALID_CONTEXT                = 201
         ENUMERATOR :: CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202
         ENUMERATOR :: CUDA_ERROR_MAP_FAILED                     = 205
         ENUMERATOR :: CUDA_ERROR_UNMAP_FAILED                   = 206
         ENUMERATOR :: CUDA_ERROR_ARRAY_IS_MAPPED                = 207
         ENUMERATOR :: CUDA_ERROR_ALREADY_MAPPED                 = 208
         ENUMERATOR :: CUDA_ERROR_NO_BINARY_FOR_GPU              = 209
         ENUMERATOR :: CUDA_ERROR_ALREADY_ACQUIRED               = 210
         ENUMERATOR :: CUDA_ERROR_NOT_MAPPED                     = 211
         ENUMERATOR :: CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212
         ENUMERATOR :: CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213
         ENUMERATOR :: CUDA_ERROR_ECC_UNCORRECTABLE              = 214
         ENUMERATOR :: CUDA_ERROR_UNSUPPORTED_LIMIT              = 215
         ENUMERATOR :: CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216
         ENUMERATOR :: CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217
         ENUMERATOR :: CUDA_ERROR_INVALID_PTX                    = 218
         ENUMERATOR :: CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = 219
         ENUMERATOR :: CUDA_ERROR_NVLINK_UNCORRECTABLE           = 220
         ENUMERATOR :: CUDA_ERROR_JIT_COMPILER_NOT_FOUND         = 221
         ENUMERATOR :: CUDA_ERROR_UNSUPPORTED_PTX_VERSION        = 222
         ENUMERATOR :: CUDA_ERROR_JIT_COMPILATION_DISABLED       = 223
         ENUMERATOR :: CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY      = 224
         ENUMERATOR :: CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC       = 225
         ENUMERATOR :: CUDA_ERROR_INVALID_SOURCE                 = 300
         ENUMERATOR :: CUDA_ERROR_FILE_NOT_FOUND                 = 301
         ENUMERATOR :: CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
         ENUMERATOR :: CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303
         ENUMERATOR :: CUDA_ERROR_OPERATING_SYSTEM               = 304
         ENUMERATOR :: CUDA_ERROR_INVALID_HANDLE                 = 400
         ENUMERATOR :: CUDA_ERROR_ILLEGAL_STATE                  = 401
         ENUMERATOR :: CUDA_ERROR_NOT_FOUND                      = 500
         ENUMERATOR :: CUDA_ERROR_NOT_READY                      = 600
         ENUMERATOR :: CUDA_ERROR_ILLEGAL_ADDRESS                = 700
         ENUMERATOR :: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701
         ENUMERATOR :: CUDA_ERROR_LAUNCH_TIMEOUT                 = 702
         ENUMERATOR :: CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703
         ENUMERATOR :: CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704
         ENUMERATOR :: CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705
         ENUMERATOR :: CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708
         ENUMERATOR :: CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709
         ENUMERATOR :: CUDA_ERROR_ASSERT                         = 710
         ENUMERATOR :: CUDA_ERROR_TOO_MANY_PEERS                 = 711
         ENUMERATOR :: CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
         ENUMERATOR :: CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713
         ENUMERATOR :: CUDA_ERROR_HARDWARE_STACK_ERROR           = 714
         ENUMERATOR :: CUDA_ERROR_ILLEGAL_INSTRUCTION            = 715
         ENUMERATOR :: CUDA_ERROR_MISALIGNED_ADDRESS             = 716
         ENUMERATOR :: CUDA_ERROR_INVALID_ADDRESS_SPACE          = 717
         ENUMERATOR :: CUDA_ERROR_INVALID_PC                     = 718
         ENUMERATOR :: CUDA_ERROR_LAUNCH_FAILED                  = 719
         ENUMERATOR :: CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720
         ENUMERATOR :: CUDA_ERROR_NOT_PERMITTED                  = 800
         ENUMERATOR :: CUDA_ERROR_NOT_SUPPORTED                  = 801
         ENUMERATOR :: CUDA_ERROR_SYSTEM_NOT_READY               = 802
         ENUMERATOR :: CUDA_ERROR_SYSTEM_DRIVER_MISMATCH         = 803
         ENUMERATOR :: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
         ENUMERATOR :: CUDA_ERROR_MPS_CONNECTION_FAILED          = 805
         ENUMERATOR :: CUDA_ERROR_MPS_RPC_FAILURE                = 806
         ENUMERATOR :: CUDA_ERROR_MPS_SERVER_NOT_READY           = 807
         ENUMERATOR :: CUDA_ERROR_MPS_MAX_CLIENTS_REACHED        = 808
         ENUMERATOR :: CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED    = 809
         ENUMERATOR :: CUDA_ERROR_MPS_CLIENT_TERMINATED          = 810
         ENUMERATOR :: CUDA_ERROR_CDP_NOT_SUPPORTED              = 811
         ENUMERATOR :: CUDA_ERROR_CDP_VERSION_MISMATCH           = 812
         ENUMERATOR :: CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED     = 900
         ENUMERATOR :: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED     = 901
         ENUMERATOR :: CUDA_ERROR_STREAM_CAPTURE_MERGE           = 902
         ENUMERATOR :: CUDA_ERROR_STREAM_CAPTURE_UNMATCHED       = 903
         ENUMERATOR :: CUDA_ERROR_STREAM_CAPTURE_UNJOINED        = 904
         ENUMERATOR :: CUDA_ERROR_STREAM_CAPTURE_ISOLATION       = 905
         ENUMERATOR :: CUDA_ERROR_STREAM_CAPTURE_IMPLICIT        = 906
         ENUMERATOR :: CUDA_ERROR_CAPTURED_EVENT                 = 907
         ENUMERATOR :: CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD    = 908
         ENUMERATOR :: CUDA_ERROR_TIMEOUT                        = 909
         ENUMERATOR :: CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE      = 910
         ENUMERATOR :: CUDA_ERROR_EXTERNAL_DEVICE                = 911
         ENUMERATOR :: CUDA_ERROR_INVALID_CLUSTER_SIZE           = 912
         ENUMERATOR :: CUDA_ERROR_UNKNOWN                        = 999
      END ENUM

      CONTAINS

!-------------------------------------------------------------------------------
!>  @brief Report status if there was an error and exit.
!>
!>  @params[in] status Status state for a cusolver function.
!-------------------------------------------------------------------------------
      SUBROUTINE cuda_checkerror(status)

      IMPLICIT NONE

      CUresult :: status

      SELECT CASE (status)
         CASE (CUDA_SUCCESS)
            RETURN

         CASE (CUDA_ERROR_INVALID_VALUE)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_VALUE')

         CASE (CUDA_ERROR_OUT_OF_MEMORY)
            CALL cuda_assert(status, 'CUDA_ERROR_OUT_OF_MEMORY')

         CASE (CUDA_ERROR_NOT_INITIALIZED)
            CALL cuda_assert(status, 'CUDA_ERROR_NOT_INITIALIZED')

         CASE (CUDA_ERROR_DEINITIALIZED)
            CALL cuda_assert(status, 'CUDA_ERROR_DEINITIALIZED')

         CASE (CUDA_ERROR_PROFILER_DISABLED)
            CALL cuda_assert(status, 'CUDA_ERROR_PROFILER_DISABLED')

         CASE (CUDA_ERROR_PROFILER_NOT_INITIALIZED)
            CALL cuda_assert(status, 'CUDA_ERROR_PROFILER_NOT_INITIALIZED')

         CASE (CUDA_ERROR_PROFILER_ALREADY_STARTED)
            CALL cuda_assert(status, 'CUDA_ERROR_PROFILER_ALREADY_STARTED')

         CASE (CUDA_ERROR_PROFILER_ALREADY_STOPPED)
            CALL cuda_assert(status, 'CUDA_ERROR_PROFILER_ALREADY_STOPPED')

         CASE (CUDA_ERROR_STUB_LIBRARY)
            CALL cuda_assert(status, 'CUDA_ERROR_STUB_LIBRARY')

         CASE (CUDA_ERROR_DEVICE_UNAVAILABLE)
            CALL cuda_assert(status, 'CUDA_ERROR_DEVICE_UNAVAILABLE')

         CASE (CUDA_ERROR_NO_DEVICE)
            CALL cuda_assert(status, 'CUDA_ERROR_NO_DEVICE')

         CASE (CUDA_ERROR_INVALID_DEVICE)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_DEVICE')

         CASE (CUDA_ERROR_DEVICE_NOT_LICENSED)
            CALL cuda_assert(status, 'CUDA_ERROR_DEVICE_NOT_LICENSED')

         CASE (CUDA_ERROR_INVALID_IMAGE)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_IMAGE')

         CASE (CUDA_ERROR_INVALID_CONTEXT)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_CONTEXT')

         CASE (CUDA_ERROR_CONTEXT_ALREADY_CURRENT)
            CALL cuda_assert(status, 'CUDA_ERROR_CONTEXT_ALREADY_CURRENT')

         CASE (CUDA_ERROR_MAP_FAILED)
            CALL cuda_assert(status, 'CUDA_ERROR_MAP_FAILED')

         CASE (CUDA_ERROR_UNMAP_FAILED)
            CALL cuda_assert(status, 'CUDA_ERROR_UNMAP_FAILED')

         CASE (CUDA_ERROR_ARRAY_IS_MAPPED)
            CALL cuda_assert(status, 'CUDA_ERROR_ARRAY_IS_MAPPED')

         CASE (CUDA_ERROR_ALREADY_MAPPED)
            CALL cuda_assert(status, 'CUDA_ERROR_ALREADY_MAPPED')

         CASE (CUDA_ERROR_NO_BINARY_FOR_GPU)
            CALL cuda_assert(status, 'CUDA_ERROR_NO_BINARY_FOR_GPU')

         CASE (CUDA_ERROR_ALREADY_ACQUIRED)
            CALL cuda_assert(status, 'CUDA_ERROR_ALREADY_ACQUIRED')

         CASE (CUDA_ERROR_NOT_MAPPED)
            CALL cuda_assert(status, 'CUDA_ERROR_NOT_MAPPED')

         CASE (CUDA_ERROR_NOT_MAPPED_AS_ARRAY)
            CALL cuda_assert(status, 'CUDA_ERROR_NOT_MAPPED_AS_ARRAY')

         CASE (CUDA_ERROR_NOT_MAPPED_AS_POINTER)
            CALL cuda_assert(status, 'CUDA_ERROR_NOT_MAPPED_AS_POINTER')

         CASE (CUDA_ERROR_ECC_UNCORRECTABLE)
            CALL cuda_assert(status, 'CUDA_ERROR_ECC_UNCORRECTABLE')

         CASE (CUDA_ERROR_UNSUPPORTED_LIMIT)
            CALL cuda_assert(status, 'CUDA_ERROR_UNSUPPORTED_LIMIT')

         CASE (CUDA_ERROR_CONTEXT_ALREADY_IN_USE)
            CALL cuda_assert(status, 'CUDA_ERROR_CONTEXT_ALREADY_IN_USE')

         CASE (CUDA_ERROR_PEER_ACCESS_UNSUPPORTED)
            CALL cuda_assert(status, 'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED')

         CASE (CUDA_ERROR_INVALID_PTX)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_PTX')

         CASE (CUDA_ERROR_INVALID_GRAPHICS_CONTEXT)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_GRAPHICS_CONTEXT')

         CASE (CUDA_ERROR_NVLINK_UNCORRECTABLE)
            CALL cuda_assert(status, 'CUDA_ERROR_NVLINK_UNCORRECTABLE')

         CASE (CUDA_ERROR_JIT_COMPILER_NOT_FOUND)
            CALL cuda_assert(status, 'CUDA_ERROR_JIT_COMPILER_NOT_FOUND')

         CASE (CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
            CALL cuda_assert(status, 'CUDA_ERROR_UNSUPPORTED_PTX_VERSION')

         CASE (CUDA_ERROR_JIT_COMPILATION_DISABLED)
            CALL cuda_assert(status, 'CUDA_ERROR_JIT_COMPILATION_DISABLED')

         CASE (CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY)
            CALL cuda_assert(status, 'CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY')

         CASE (CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC)
            CALL cuda_assert(status, 'CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC')

         CASE (CUDA_ERROR_INVALID_SOURCE)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_SOURCE')

         CASE (CUDA_ERROR_FILE_NOT_FOUND)
            CALL cuda_assert(status, 'CUDA_ERROR_FILE_NOT_FOUND')

         CASE (CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND)
            CALL cuda_assert(status, 'CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND')

         CASE (CUDA_ERROR_SHARED_OBJECT_INIT_FAILED)
            CALL cuda_assert(status, 'CUDA_ERROR_SHARED_OBJECT_INIT_FAILED')

         CASE (CUDA_ERROR_OPERATING_SYSTEM)
            CALL cuda_assert(status, 'CUDA_ERROR_OPERATING_SYSTEM')

         CASE (CUDA_ERROR_INVALID_HANDLE)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_HANDLE')

         CASE (CUDA_ERROR_ILLEGAL_STATE)
            CALL cuda_assert(status, 'CUDA_ERROR_ILLEGAL_STATE')

         CASE (CUDA_ERROR_NOT_FOUND)
            CALL cuda_assert(status, 'CUDA_ERROR_NOT_FOUND')

         CASE (CUDA_ERROR_NOT_READY)
            CALL cuda_assert(status, 'CUDA_ERROR_NOT_READY')

         CASE (CUDA_ERROR_ILLEGAL_ADDRESS)
            CALL cuda_assert(status, 'CUDA_ERROR_ILLEGAL_ADDRESS')

         CASE (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
            CALL cuda_assert(status, 'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES')

         CASE (CUDA_ERROR_LAUNCH_TIMEOUT )
            CALL cuda_assert(status, 'CUDA_ERROR_LAUNCH_TIMEOUT')

         CASE (CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING)
            CALL cuda_assert(status, 'CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING')

         CASE (CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CALL cuda_assert(status, 'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED')

         CASE (CUDA_ERROR_PEER_ACCESS_NOT_ENABLED)
            CALL cuda_assert(status, 'CUDA_ERROR_PEER_ACCESS_NOT_ENABLED')

         CASE (CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE)
            CALL cuda_assert(status, 'CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE')

         CASE (CUDA_ERROR_CONTEXT_IS_DESTROYED)
            CALL cuda_assert(status, 'CUDA_ERROR_CONTEXT_IS_DESTROYED')

         CASE (CUDA_ERROR_ASSERT)
            CALL cuda_assert(status, 'CUDA_ERROR_ASSERT')

         CASE (CUDA_ERROR_TOO_MANY_PEERS)
            CALL cuda_assert(status, 'CUDA_ERROR_TOO_MANY_PEERS')

         CASE (CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
            CALL cuda_assert(status, 'CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED')

         CASE (CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
            CALL cuda_assert(status, 'CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED')

         CASE (CUDA_ERROR_HARDWARE_STACK_ERROR)
            CALL cuda_assert(status, 'CUDA_ERROR_HARDWARE_STACK_ERROR')

         CASE (CUDA_ERROR_ILLEGAL_INSTRUCTION)
            CALL cuda_assert(status, 'CUDA_ERROR_ILLEGAL_INSTRUCTION')

         CASE (CUDA_ERROR_MISALIGNED_ADDRESS)
            CALL cuda_assert(status, 'CUDA_ERROR_MISALIGNED_ADDRESS')

         CASE (CUDA_ERROR_INVALID_ADDRESS_SPACE)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_ADDRESS_SPACE')

         CASE (CUDA_ERROR_INVALID_PC)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_PC')

         CASE (CUDA_ERROR_LAUNCH_FAILED)
            CALL cuda_assert(status, 'CUDA_ERROR_LAUNCH_FAILED')

         CASE (CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE)
            CALL cuda_assert(status, 'CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE')

         CASE (CUDA_ERROR_NOT_PERMITTED)
            CALL cuda_assert(status, 'CUDA_ERROR_NOT_PERMITTED')

         CASE (CUDA_ERROR_NOT_SUPPORTED)
            CALL cuda_assert(status, 'CUDA_ERROR_NOT_SUPPORTED')

         CASE (CUDA_ERROR_SYSTEM_NOT_READY)
            CALL cuda_assert(status, 'CUDA_ERROR_SYSTEM_NOT_READY')

         CASE (CUDA_ERROR_SYSTEM_DRIVER_MISMATCH)
            CALL cuda_assert(status, 'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH')

         CASE (CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)
            CALL cuda_assert(status, 'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE')

         CASE (CUDA_ERROR_MPS_CONNECTION_FAILED)
            CALL cuda_assert(status, 'CUDA_ERROR_MPS_CONNECTION_FAILED')

         CASE (CUDA_ERROR_MPS_RPC_FAILURE)
            CALL cuda_assert(status, 'CUDA_ERROR_MPS_RPC_FAILURE')

         CASE (CUDA_ERROR_MPS_SERVER_NOT_READY)
            CALL cuda_assert(status, 'CUDA_ERROR_MPS_SERVER_NOT_READY')

         CASE (CUDA_ERROR_MPS_MAX_CLIENTS_REACHED)
            CALL cuda_assert(status, 'CUDA_ERROR_MPS_MAX_CLIENTS_REACHED')

         CASE (CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED)
            CALL cuda_assert(status, 'CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED')

         CASE (CUDA_ERROR_MPS_CLIENT_TERMINATED)
            CALL cuda_assert(status, 'CUDA_ERROR_MPS_CLIENT_TERMINATED')

         CASE (CUDA_ERROR_CDP_NOT_SUPPORTED)
            CALL cuda_assert(status, 'CUDA_ERROR_CDP_NOT_SUPPORTED')

         CASE (CUDA_ERROR_CDP_VERSION_MISMATCH)
            CALL cuda_assert(status, 'CUDA_ERROR_CDP_VERSION_MISMATCH')

         CASE (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED)
            CALL cuda_assert(status, 'CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED')

         CASE (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED)
            CALL cuda_assert(status, 'CUDA_ERROR_STREAM_CAPTURE_INVALIDATED')

         CASE (CUDA_ERROR_STREAM_CAPTURE_MERGE)
            CALL cuda_assert(status, 'CUDA_ERROR_STREAM_CAPTURE_MERGE')

         CASE (CUDA_ERROR_STREAM_CAPTURE_UNMATCHED)
            CALL cuda_assert(status, 'CUDA_ERROR_STREAM_CAPTURE_UNMATCHED')

         CASE (CUDA_ERROR_STREAM_CAPTURE_UNJOINED)
            CALL cuda_assert(status, 'CUDA_ERROR_STREAM_CAPTURE_UNJOINED')

         CASE (CUDA_ERROR_STREAM_CAPTURE_ISOLATION)
            CALL cuda_assert(status, 'CUDA_ERROR_STREAM_CAPTURE_ISOLATION')

         CASE (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT)
            CALL cuda_assert(status, 'CUDA_ERROR_STREAM_CAPTURE_IMPLICIT')

         CASE (CUDA_ERROR_CAPTURED_EVENT)
            CALL cuda_assert(status, 'CUDA_ERROR_CAPTURED_EVENT')

         CASE (CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD)
            CALL cuda_assert(status, 'CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD')

         CASE (CUDA_ERROR_TIMEOUT)
            CALL cuda_assert(status, 'CUDA_ERROR_TIMEOUT')

         CASE (CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE)
            CALL cuda_assert(status, 'CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE')

         CASE (CUDA_ERROR_EXTERNAL_DEVICE)
            CALL cuda_assert(status, 'CUDA_ERROR_EXTERNAL_DEVICE')

         CASE (CUDA_ERROR_INVALID_CLUSTER_SIZE)
            CALL cuda_assert(status, 'CUDA_ERROR_INVALID_CLUSTER_SIZE')

         CASE DEFAULT
            CALL cuda_assert(status, 'CUDA_ERROR_UNKNOWN')

      END SELECT

      END SUBROUTINE

      END MODULE
