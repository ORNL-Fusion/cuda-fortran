      MODULE cudaASSERT
      USE, INTRINSIC :: iso_fortran_env, only : error_unit

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
      INTEGER, INTENT(IN)           :: status
      CHARACTER (len=*), INTENT(IN) :: message
      INTEGER, OPTIONAL, INTENT(IN) :: iounit

!  Start of executable code
      IF (PRESENT(iounit)) THEN
         WRITE (iounit, 1000) status, message
      ELSE
         WRITE (error_unit, 1000) status, message
      END IF
      CALL EXIT(status)

1000  FORMAT('Error stats',i2,' : ',a)

      END SUBROUTINE

      END MODULE
