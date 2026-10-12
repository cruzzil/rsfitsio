//! Routines that define the starting heap address, the value scaling and the
//! null values.
//!
//! The scaling set here affects only the automatic conversion performed as data
//! elements are read and written; it does not change the `BSCALE`, `BZERO` or
//! `TNULLn` keywords. When reading, the value handed back is
//! `(value in the FITS array) * scale + zero`; the inverse is applied when
//! writing.
//!
//! Ported from CFITSIO's `scalnull.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center. The routine descriptions draw on the "Define Data
//! Scaling and Undefined Pixel Parameters" section of the CFITSIO User's
//! Reference Guide.
#![warn(missing_docs)]

use core::ffi::CStr;

use crate::c_types::*;

use bytemuck::cast_slice;

use crate::cs;
use crate::fitscore::{ffghdt_safe, ffmahd_safe, fits_is_compressed_image_safe};
use crate::fitsio::*;
use crate::modkey::ffukyj_safe;
use crate::wrappers::*;

/// Define the starting address for the heap for a binary table.
///
/// The default address is NAXIS1 * NAXIS2.  It is in units of
/// bytes relative to the beginning of the regular binary table data.
/// This routine also writes the appropriate THEAP keyword to the
/// FITS header.
///
/// The offset is zero-indexed and measured from the start of the binary table
/// data. This is only relevant for binary tables that contain variable length
/// array columns (`TFORMn = 'Pt'`), and must be called after the required
/// keywords have been written but before any data is written to the table.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `theap`  — (I) starting address for the heap
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffpthp(fptr: *mut fitsfile, theap: c_long, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        ffpthp_safe(fptr, theap, status)
    }
}

/// Define the starting address for the heap for a binary table.
///
/// The default address is NAXIS1 * NAXIS2.  It is in units of
/// bytes relative to the beginning of the regular binary table data.
/// This routine also writes the appropriate THEAP keyword to the
/// FITS header.
///
/// The offset is zero-indexed and measured from the start of the binary table
/// data. This is only relevant for binary tables that contain variable length
/// array columns (`TFORMn = 'Pt'`), and must be called after the required
/// keywords have been written but before any data is written to the table.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `theap`  — (I) starting address for the heap
/// * `status` — (IO) error status
pub fn ffpthp_safe(fptr: &mut fitsfile, theap: c_long, status: &mut c_int) -> c_int {
    if *status > 0 || theap < 1 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }

    fptr.Fptr.heapstart = theap as LONGLONG;

    ffukyj_safe(
        fptr,
        cs!(c"THEAP"),
        theap as LONGLONG,
        Some(cs!(c"byte offset to heap area")),
        status,
    );
    *status
}

/// Define the linear scaling factor for the primary array or image extension
/// pixel values. This routine overrides the scaling values given by the
/// BSCALE and BZERO keywords if present.  Note that this routine does not
/// write or modify the BSCALE and BZERO keywords, but instead only modifies
/// the values temporarily in the internal buffer.  Thus, a subsequent call to
/// the ffrdef routine will reset the scaling back to the BSCALE and BZERO
/// keyword values (or 1. and 0. respectively if the keywords are not present).
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `scale`  — (I) scaling factor: value of BSCALE
/// * `zero`   — (I) zero point: value of BZERO
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffpscl(
    fptr: *mut fitsfile,
    scale: f64,
    zero: f64,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        ffpscl_safe(fptr, scale, zero, status)
    }
}

/// Define the linear scaling factor for the primary array or image extension
/// pixel values. This routine overrides the scaling values given by the
/// BSCALE and BZERO keywords if present.  Note that this routine does not
/// write or modify the BSCALE and BZERO keywords, but instead only modifies
/// the values temporarily in the internal buffer.  Thus, a subsequent call to
/// the ffrdef routine will reset the scaling back to the BSCALE and BZERO
/// keyword values (or 1. and 0. respectively if the keywords are not present).
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `scale`  — (I) scaling factor: value of BSCALE
/// * `zero`   — (I) zero point: value of BZERO
/// * `status` — (IO) error status
pub fn ffpscl_safe(fptr: &mut fitsfile, scale: f64, zero: f64, status: &mut c_int) -> c_int {
    let mut hdutype = 0;
    if *status > 0 {
        return *status;
    }

    if scale == 0.0 {
        /* zero scale value is illegal */
        *status = ZERO_SCALE;
        return *status;
    }

    /* get HDU type */
    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    if hdutype != IMAGE_HDU {
        /* not proper HDU type */
        *status = NOT_IMAGE;
        return *status;
    }

    if fits_is_compressed_image_safe(fptr, status) != 0 {
        /* compressed images */
        fptr.Fptr.cn_bscale = scale;
        fptr.Fptr.cn_bzero = zero;
        return *status;
    }

    /* set pointer to the first 'column' (contains group parameters if any) */
    let c = fptr.Fptr.get_tableptr_as_mut_slice();
    let colptr = &mut c[1]; /* increment to the 2nd 'column' pointer  (the image itself) */

    (colptr).tscale = scale;
    (colptr).tzero = zero;
    *status
}

/// Define the value used to represent undefined pixels in the primary array or
/// image extension. This only applies to integer image pixel (i.e. BITPIX > 0).
/// This routine overrides the null pixel value given by the BLANK keyword
/// if present.  Note that this routine does not write or modify the BLANK
/// keyword, but instead only modifies the value temporarily in the internal
/// buffer. Thus, a subsequent call to the ffrdef routine will reset the null
/// value back to the BLANK  keyword value (or not defined if the keyword is not
/// present).
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `nulvalue` — (I) null pixel value: value of BLANK
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffpnul(
    fptr: *mut fitsfile,
    nulvalue: LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        ffpnul_safe(fptr, nulvalue, status)
    }
}

/// Define the value used to represent undefined pixels in the primary array or
/// image extension. This only applies to integer image pixel (i.e. BITPIX > 0).
/// This routine overrides the null pixel value given by the BLANK keyword
/// if present.  Note that this routine does not write or modify the BLANK
/// keyword, but instead only modifies the value temporarily in the internal
/// buffer. Thus, a subsequent call to the ffrdef routine will reset the null
/// value back to the BLANK  keyword value (or not defined if the keyword is not
/// present).
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `nulvalue` — (I) null pixel value: value of BLANK
/// * `status`   — (IO) error status
pub fn ffpnul_safe(fptr: &mut fitsfile, nulvalue: LONGLONG, status: &mut c_int) -> c_int {
    let mut hdutype = 0;

    if *status > 0 {
        return *status;
    }

    /* get HDU type */
    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    if hdutype != IMAGE_HDU {
        /* not proper HDU type */
        *status = NOT_IMAGE;
        return *status;
    }

    if fits_is_compressed_image_safe(fptr, status) != 0 {
        /* ignore compressed images */
        return *status;
    }

    /* set pointer to the first 'column' (contains group parameters if any) */
    let c = fptr.Fptr.get_tableptr_as_mut_slice();
    let colptr = &mut c[1]; /* increment to the 2nd 'column' pointer  (the image itself) */
    colptr.tnull = nulvalue;
    *status
}

/// Define the linear scaling factor for the TABLE or BINTABLE extension
/// column values. This routine overrides the scaling values given by the
/// TSCALn and TZEROn keywords if present.  Note that this routine does not
/// write or modify the TSCALn and TZEROn keywords, but instead only modifies
/// the values temporarily in the internal buffer.  Thus, a subsequent call to
/// the ffrdef routine will reset the scaling back to the TSCALn and TZEROn
/// keyword values (or 1. and 0. respectively if the keywords are not present).
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) column number to apply scaling to
/// * `scale`  — (I) scaling factor: value of TSCALn
/// * `zero`   — (I) zero point: value of TZEROn
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fftscl(
    fptr: *mut fitsfile,
    colnum: c_int,
    scale: f64,
    zero: f64,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fftscl_safe(fptr, colnum, scale, zero, status)
    }
}

/// Define the linear scaling factor for the TABLE or BINTABLE extension
/// column values. This routine overrides the scaling values given by the
/// TSCALn and TZEROn keywords if present.  Note that this routine does not
/// write or modify the TSCALn and TZEROn keywords, but instead only modifies
/// the values temporarily in the internal buffer.  Thus, a subsequent call to
/// the ffrdef routine will reset the scaling back to the TSCALn and TZEROn
/// keyword values (or 1. and 0. respectively if the keywords are not present).
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) column number to apply scaling to
/// * `scale`  — (I) scaling factor: value of TSCALn
/// * `zero`   — (I) zero point: value of TZEROn
/// * `status` — (IO) error status
pub fn fftscl_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    scale: f64,
    zero: f64,
    status: &mut c_int,
) -> c_int {
    let mut hdutype = 0;

    if *status > 0 {
        return *status;
    }

    if scale == 0.0 {
        /* zero scale value is illegal */
        *status = ZERO_SCALE;
        return *status;
    }

    /* get HDU type */
    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    if hdutype == IMAGE_HDU {
        /* not proper HDU type */
        *status = NOT_TABLE;
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_mut_slice();
    let mut colptr = 0; /* set pointer to the first column */
    colptr += colnum as usize - 1; /* increment to the correct column */
    (c[colptr]).tscale = scale;
    (c[colptr]).tzero = zero;
    *status
}

/// Define the value used to represent undefined pixels in the BINTABLE column.
///
/// This only applies to integer datatype columns (TFORM = B, I, or J).
/// This routine overrides the null pixel value given by the TNULLn keyword
/// if present.  Note that this routine does not write or modify the TNULLn
/// keyword, but instead only modifies the value temporarily in the internal
/// buffer. Thus, a subsequent call to the ffrdef routine will reset the null
/// value back to the TNULLn  keyword value (or not defined if the keyword is not
/// present).
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number to apply nulvalue to
/// * `nulvalue` — (I) null pixel value: value of TNULLn
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fftnul(
    fptr: *mut fitsfile,
    colnum: c_int,
    nulvalue: LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fftnul_safe(fptr, colnum, nulvalue, status)
    }
}

/// Define the value used to represent undefined pixels in the BINTABLE column.
///
/// This only applies to integer datatype columns (TFORM = B, I, or J).
/// This routine overrides the null pixel value given by the TNULLn keyword
/// if present.  Note that this routine does not write or modify the TNULLn
/// keyword, but instead only modifies the value temporarily in the internal
/// buffer. Thus, a subsequent call to the ffrdef routine will reset the null
/// value back to the TNULLn  keyword value (or not defined if the keyword is not
/// present).
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number to apply nulvalue to
/// * `nulvalue` — (I) null pixel value: value of TNULLn
/// * `status`   — (IO) error status
pub fn fftnul_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    nulvalue: LONGLONG,
    status: &mut c_int,
) -> c_int {
    let mut hdutype = 0;
    if *status > 0 {
        return *status;
    }

    /* get HDU type */
    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    if hdutype != BINARY_TBL {
        /* not proper HDU type */
        *status = NOT_BTABLE;
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_mut_slice(); /* set pointer to the first column */
    let mut colptr = 0; /* set pointer to the first column */
    colptr += colnum as usize - 1; /* increment to the correct column */
    c[colptr].tnull = nulvalue;
    *status
}

/// Define the string used to represent undefined pixels in the ASCII TABLE
/// column. This routine overrides the null  value given by the TNULLn keyword
/// if present.  Note that this routine does not write or modify the TNULLn
/// keyword, but instead only modifies the value temporarily in the internal
/// buffer. Thus, a subsequent call to the ffrdef routine will reset the null
/// value back to the TNULLn keyword value (or not defined if the keyword is not
/// present).
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) column number to apply nulvalue to
/// * `nulstring` — (I) null pixel value: value of TNULLn
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffsnul(
    fptr: *mut fitsfile,
    colnum: c_int,
    nulstring: *const c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        let nulstring = CStr::from_ptr(nulstring).to_bytes_with_nul();

        ffsnul_safe(fptr, colnum, cast_slice(nulstring), status)
    }
}

/// Define the string used to represent undefined pixels in the ASCII TABLE
/// column. This routine overrides the null  value given by the TNULLn keyword
/// if present.  Note that this routine does not write or modify the TNULLn
/// keyword, but instead only modifies the value temporarily in the internal
/// buffer. Thus, a subsequent call to the ffrdef routine will reset the null
/// value back to the TNULLn keyword value (or not defined if the keyword is not
/// present).
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) column number to apply nulvalue to
/// * `nulstring` — (I) null pixel value: value of TNULLn
/// * `status`    — (IO) error status
pub fn ffsnul_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    nulstring: &[c_char],
    status: &mut c_int,
) -> c_int {
    let mut hdutype = 0;
    if *status > 0 {
        return *status;
    }

    /* get HDU type */
    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    if hdutype != ASCII_TBL {
        /* not proper HDU type */
        *status = NOT_ATABLE;
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_mut_slice(); /* set pointer to the first column */
    let colptr = colnum as usize - 1; /* increment to the correct column */
    c[colptr].strnull[0] = 0;
    strncat_safe(&mut c[colptr].strnull, nulstring, 19); /* limit string to 19 chars */
    *status
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfileio::{ffclos_safe, ffinit_safe};
    use crate::putkey::ffcrtb_safe as ffcrtb;
    use crate::putkey::ffphps_safe;

    /* Ported from test_scalnull.c - scaling and null value functions.
    Each test uses an in-memory file (no reopen by name). */

    /// Create a fresh mem:// file with no primary HDU written yet.
    fn new_file() -> (Option<Box<fitsfile>>, c_int) {
        let mut status = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        ffinit_safe(&mut fptr, cs!(c"mem://"), &mut status);
        assert_eq!(status, 0, "ffinit failed");
        (fptr, status)
    }

    fn make_btable(f: &mut fitsfile, tform0: &[c_char], status: &mut c_int) {
        ffphps_safe(f, BYTE_IMG, 0, &[], status);
        let ttype = [Some(cs!(c"COL1"))];
        let tform = [tform0];
        ffcrtb(f, BINARY_TBL, 1, 1, &ttype, &tform, None, None, status);
        assert_eq!(*status, 0, "ffcrtb failed");
    }

    #[test]
    fn test_ffpthp_basic() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1PA"), &mut status);
        ffpthp_safe(f, 100, &mut status);
        assert_eq!(status, 0, "ffpthp failed");
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffpthp_bad_theap() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        /* theap < 1 should return early */
        ffpthp_safe(f, 0, &mut status);
        assert_eq!(status, 0);
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffpscl_basic() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        let naxes: [c_long; 2] = [10, 10];
        ffphps_safe(f, SHORT_IMG, 2, &naxes, &mut status);
        ffpscl_safe(f, 2.0, 100.0, &mut status);
        assert_eq!(status, 0, "ffpscl failed");
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffpscl_zero_scale() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        let naxes: [c_long; 1] = [10];
        ffphps_safe(f, SHORT_IMG, 1, &naxes, &mut status);
        ffpscl_safe(f, 0.0, 0.0, &mut status);
        assert_eq!(status, ZERO_SCALE);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffpscl_not_image() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        ffpscl_safe(f, 1.0, 0.0, &mut status);
        assert_eq!(status, NOT_IMAGE);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffpscl_error_status() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        let naxes: [c_long; 1] = [10];
        ffphps_safe(f, SHORT_IMG, 1, &naxes, &mut status);
        status = 1;
        ffpscl_safe(f, 1.0, 0.0, &mut status);
        assert_eq!(status, 1);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffpnul_basic() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        let naxes: [c_long; 2] = [10, 10];
        ffphps_safe(f, SHORT_IMG, 2, &naxes, &mut status);
        ffpnul_safe(f, -999, &mut status);
        assert_eq!(status, 0, "ffpnul failed");
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffpnul_not_image() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        ffpnul_safe(f, -999, &mut status);
        assert_eq!(status, NOT_IMAGE);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffpnul_error_status() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        let naxes: [c_long; 1] = [10];
        ffphps_safe(f, SHORT_IMG, 1, &naxes, &mut status);
        status = 1;
        ffpnul_safe(f, -999, &mut status);
        assert_eq!(status, 1);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_fftscl_basic() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        fftscl_safe(f, 1, 2.0, 100.0, &mut status);
        assert_eq!(status, 0, "fftscl failed");
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_fftscl_zero_scale() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        fftscl_safe(f, 1, 0.0, 0.0, &mut status);
        assert_eq!(status, ZERO_SCALE);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_fftscl_not_table() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        let naxes: [c_long; 1] = [10];
        ffphps_safe(f, SHORT_IMG, 1, &naxes, &mut status);
        fftscl_safe(f, 1, 1.0, 0.0, &mut status);
        assert_eq!(status, NOT_TABLE);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_fftscl_error_status() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        status = 1;
        fftscl_safe(f, 1, 1.0, 0.0, &mut status);
        assert_eq!(status, 1);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_fftnul_basic() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        fftnul_safe(f, 1, -999, &mut status);
        assert_eq!(status, 0, "fftnul failed");
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_fftnul_not_btable() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        ffphps_safe(f, BYTE_IMG, 0, &[], &mut status);
        let ttype = [Some(cs!(c"COL1"))];
        let tform = [cs!(c"A10")];
        ffcrtb(f, ASCII_TBL, 1, 1, &ttype, &tform, None, None, &mut status);
        assert_eq!(status, 0, "ffcrtb failed");
        fftnul_safe(f, 1, -999, &mut status);
        assert_eq!(status, NOT_BTABLE);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_fftnul_error_status() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        status = 1;
        fftnul_safe(f, 1, -999, &mut status);
        assert_eq!(status, 1);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffsnul_basic() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        ffphps_safe(f, BYTE_IMG, 0, &[], &mut status);
        let ttype = [Some(cs!(c"COL1"))];
        let tform = [cs!(c"A10")];
        ffcrtb(f, ASCII_TBL, 1, 1, &ttype, &tform, None, None, &mut status);
        assert_eq!(status, 0, "ffcrtb failed");
        ffsnul_safe(f, 1, cs!(c"N/A"), &mut status);
        assert_eq!(status, 0, "ffsnul failed");
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffsnul_not_atable() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        make_btable(f, cs!(c"1J"), &mut status);
        ffsnul_safe(f, 1, cs!(c"N/A"), &mut status);
        assert_eq!(status, NOT_ATABLE);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffsnul_error_status() {
        let (mut fptr, mut status) = new_file();
        let f = fptr.as_deref_mut().unwrap();
        ffphps_safe(f, BYTE_IMG, 0, &[], &mut status);
        let ttype = [Some(cs!(c"COL1"))];
        let tform = [cs!(c"A10")];
        ffcrtb(f, ASCII_TBL, 1, 1, &ttype, &tform, None, None, &mut status);
        assert_eq!(status, 0, "ffcrtb failed");
        status = 1;
        ffsnul_safe(f, 1, cs!(c"N/A"), &mut status);
        assert_eq!(status, 1);
        status = 0;
        ffclos_safe(fptr.take().unwrap(), &mut status);
    }
}
