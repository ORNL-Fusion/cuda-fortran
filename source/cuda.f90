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

      IMPLICIT NONE

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

      END MODULE
