!*******************************************************************************
!>  @file cuda_types.f90
!>  @brief Contains module @ref cuda_types.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  Defines the cuda types.
!*******************************************************************************
      MODULE cuda_types

      USE, INTRINSIC :: iso_c_binding

      IMPLICIT NONE

!*******************************************************************************
!  Derived types
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Cuda Context type.
!-------------------------------------------------------------------------------
      TYPE, BIND(C) :: context_t
         TYPE (C_PTR) :: context
      END TYPE

      END MODULE
