!*******************************************************************************
!>  @file cublas.f90
!>  @brief Contains module @ref cudaASSERT.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  Defines utilities for assertions.
!*******************************************************************************

      MODULE cudaASSERT
      USE, INTRINSIC :: iso_fortran_env, only : error_unit
      USE, INTRINSIC :: iso_c_binding

      IMPLICIT NONE

      CONTAINS

!-------------------------------------------------------------------------------
!>  @brief Report status if there was an error and exit.
!>
!>  @params[in] status  Error number.
!>  @params[in] message Error message for status.
!>  @params[in] iounit  Io unit number.
!-------------------------------------------------------------------------------
      SUBROUTINE cuda_assert(status, message, iounit)

      IMPLICIT NONE

!  Declare Arguments
      INTEGER(C_SIZE_T), INTENT(IN)           :: status
      CHARACTER (len=*), INTENT(IN) :: message
      INTEGER, OPTIONAL, INTENT(IN) :: iounit

!  Start of executable code
      IF (PRESENT(iounit)) THEN
         WRITE (iounit, 1000) status, message
      ELSE
         WRITE (error_unit, 1000) status, message
      END IF
      CALL EXIT(status)

1000  FORMAT('Error status ',i3,' : ',a)

      END SUBROUTINE

      END MODULE
