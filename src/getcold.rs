//! Routines that read data elements from a FITS image or table, with double
//! datatype.
//!
//! The [`TDOUBLE`] arm of the typed column I/O family. [`crate::getcol`]
//! dispatches here when a caller asks for this datatype at run time; the
//! write-side counterpart is [`crate::putcold`].
//!
//! Ported from CFITSIO's `getcold.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center.
#![warn(missing_docs)]

use crate::helpers::nularray::nul_tail;
use core::slice;
use core::{cmp, mem};

use crate::c_types::{c_char, c_int, c_long, c_short};
use crate::imcompress::{fits_read_compressed_img, fits_read_compressed_pixels};

use bytemuck::{cast_slice, cast_slice_mut};

use crate::fitscore::{
    ffasfm_safe, ffgcprll, ffghdt_safe, ffpmsg_slice, ffpmsg_str, fits_is_compressed_image_safe,
};
use crate::fitsio2::*;
use crate::wrappers::*;
use crate::{NullCheckType, fitsio::*};
use crate::{NullValue, bb};
use crate::{buffers::*, calculate_subsection_length};
use crate::{int_snprintf, slice_to_str};

/// Read an array of values from the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements will be set equal to NULVAL, unless NULVAL=0
/// in which case no checking for undefined values will be performed.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// The primary array is represented as a binary table:
/// each group of the primary array is a row in the table,
/// where the first column contains the group parameters
/// and the second column contains the image itself.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `group`     — (I) group to read (1 = 1st group)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for undefined pixels
/// * `array`     — (O) array of values that are returned
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgpvd(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: f64,
    array: *mut f64,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, nelem as usize);

        ffgpvd_safe(fptr, group, firstelem, nelem, nulval, array, anynul, status)
    }
}

/// Read an array of values from the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements will be set equal to NULVAL, unless NULVAL=0
/// in which case no checking for undefined values will be performed.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// The primary array is represented as a binary table:
/// each group of the primary array is a row in the table,
/// where the first column contains the group parameters
/// and the second column contains the image itself.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `group`     — (I) group to read (1 = 1st group)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for undefined pixels
/// * `array`     — (O) array of values that are returned
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgpvd_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: f64,
    array: &mut [f64],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let nullcheck = NullCheckType::SetPixel;
    let mut nullvalue: f64 = 0.0;

    let mut dummy_nularray = vec![0; (nelem) as usize];

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */
        nullvalue = nulval; /* set local variable */

        fits_read_compressed_pixels(
            fptr,
            TDOUBLE,
            firstelem,
            nelem,
            nullcheck,
            &Some(NullValue::Double(nullvalue)),
            cast_slice_mut(array),
            None,
            anynul,
            status,
        );
        return *status;
    }

    let row = cmp::max(1, group);

    ffgcld(
        fptr,
        2,
        row as LONGLONG,
        firstelem,
        nelem,
        1,
        NullCheckType::SetPixel,
        nulval,
        array,
        &mut dummy_nularray,
        anynul,
        status,
    );
    *status
}

/// Read an array of values from the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Any undefined pixels in the returned array will be set = 0 and the
/// corresponding nularray value will be set = 1.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// The primary array is represented as a binary table:
/// each group of the primary array is a row in the table,
/// where the first column contains the group parameters
/// and the second column contains the image itself.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `group`     — (I) group to read (1 = 1st group)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nularray`  — (O) array of null pixel flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgpfd(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut f64,
    nularray: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, nelem as usize);
        let nularray = slice::from_raw_parts_mut(nularray, nelem as usize);

        ffgpfd_safe(
            fptr, group, firstelem, nelem, array, nularray, anynul, status,
        )
    }
}

/// Read an array of values from the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Any undefined pixels in the returned array will be set = 0 and the
/// corresponding nularray value will be set = 1.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// The primary array is represented as a binary table:
/// each group of the primary array is a row in the table,
/// where the first column contains the group parameters
/// and the second column contains the image itself.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `group`     — (I) group to read (1 = 1st group)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nularray`  — (O) array of null pixel flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgpfd_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [f64],
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let nullcheck = NullCheckType::SetNullArray;

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        fits_read_compressed_pixels(
            fptr,
            TDOUBLE,
            firstelem,
            nelem,
            nullcheck,
            &None,
            cast_slice_mut(array),
            Some(nularray),
            anynul,
            status,
        );
        return *status;
    }

    let row = cmp::max(1, group);

    ffgcld(
        fptr,
        2,
        row as LONGLONG,
        firstelem,
        nelem,
        1,
        NullCheckType::SetNullArray,
        0.0,
        array,
        cast_slice_mut(nularray),
        anynul,
        status,
    );
    *status
}

/// Read an entire 2-D array of values to the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of the
/// FITS array is not the same as the array being read).  Any null
/// values in the array will be set equal to the value of nulval, unless
/// nulval = 0 in which case no null checking will be performed.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `group`  — (I) group to read (1 = 1st group)
/// * `nulval` — set undefined pixels equal to this
/// * `ncols`  — (I) number of pixels in each row of array
/// * `naxis1` — (I) FITS image NAXIS1 value
/// * `naxis2` — (I) FITS image NAXIS2 value
/// * `array`  — (O) array to be filled and returned
/// * `anynul` — (O) set to 1 if any values are null; else 0
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffg2dd(
    fptr: *mut fitsfile,
    group: c_long,
    nulval: f64,
    ncols: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    array: *mut f64,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, (ncols * naxis2 * naxis2) as usize);

        ffg2dd_safe(
            fptr, group, nulval, ncols, naxis1, naxis2, array, anynul, status,
        )
    }
}

/// Read an entire 2-D array of values to the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of the
/// FITS array is not the same as the array being read).  Any null
/// values in the array will be set equal to the value of nulval, unless
/// nulval = 0 in which case no null checking will be performed.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `group`  — (I) group to read (1 = 1st group)
/// * `nulval` — set undefined pixels equal to this
/// * `ncols`  — (I) number of pixels in each row of array
/// * `naxis1` — (I) FITS image NAXIS1 value
/// * `naxis2` — (I) FITS image NAXIS2 value
/// * `array`  — (O) array to be filled and returned
/// * `anynul` — (O) set to 1 if any values are null; else 0
/// * `status` — (IO) error status
pub fn ffg2dd_safe(
    fptr: &mut fitsfile,
    group: c_long,
    nulval: f64,
    ncols: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    array: &mut [f64],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    /* call the 3D reading routine, with the 3rd dimension = 1 */
    ffg3dd_safe(
        fptr, group, nulval, ncols, naxis2, naxis1, naxis2, 1, array, anynul, status,
    );

    *status
}

/// Read an entire 3-D array of values to the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of the
/// FITS array is not the same as the array being read).  Any null
/// values in the array will be set equal to the value of nulval, unless
/// nulval = 0 in which case no null checking will be performed.
///
/// The primary array is represented as a binary table:
/// each group of the primary array is a row in the table,
/// where the first column contains the group parameters
/// and the second column contains the image itself.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `group`  — (I) group to read (1 = 1st group)
/// * `nulval` — set undefined pixels equal to this
/// * `ncols`  — (I) number of pixels in each row of array
/// * `nrows`  — (I) number of rows in each plane of array
/// * `naxis1` — (I) FITS image NAXIS1 value
/// * `naxis2` — (I) FITS image NAXIS2 value
/// * `naxis3` — (I) FITS image NAXIS3 value
/// * `array`  — (O) array to be filled and returned
/// * `anynul` — (O) set to 1 if any values are null; else 0
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffg3dd(
    fptr: *mut fitsfile,
    group: c_long,
    nulval: f64,
    ncols: LONGLONG,
    nrows: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    naxis3: LONGLONG,
    array: *mut f64,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, (ncols * naxis2 * naxis3) as usize);

        ffg3dd_safe(
            fptr, group, nulval, ncols, nrows, naxis1, naxis2, naxis3, array, anynul, status,
        )
    }
}

/// Read an entire 3-D array of values to the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of the
/// FITS array is not the same as the array being read).  Any null
/// values in the array will be set equal to the value of nulval, unless
/// nulval = 0 in which case no null checking will be performed.
///
/// The primary array is represented as a binary table:
/// each group of the primary array is a row in the table,
/// where the first column contains the group parameters
/// and the second column contains the image itself.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `group`  — (I) group to read (1 = 1st group)
/// * `nulval` — set undefined pixels equal to this
/// * `ncols`  — (I) number of pixels in each row of array
/// * `nrows`  — (I) number of rows in each plane of array
/// * `naxis1` — (I) FITS image NAXIS1 value
/// * `naxis2` — (I) FITS image NAXIS2 value
/// * `naxis3` — (I) FITS image NAXIS3 value
/// * `array`  — (O) array to be filled and returned
/// * `anynul` — (O) set to 1 if any values are null; else 0
/// * `status` — (IO) error status
pub fn ffg3dd_safe(
    fptr: &mut fitsfile,
    group: c_long,
    nulval: f64,
    ncols: LONGLONG,
    nrows: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    naxis3: LONGLONG,
    array: &mut [f64],
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut narray = 0;
    let mut nfits: LONGLONG = 0;
    let nullcheck = NullCheckType::SetPixel;
    let inc: [c_long; 3] = [1; 3];
    let fpixel: [c_long; 3] = [1; 3];
    let mut lpixel: [c_long; 3] = [0; 3];
    let mut nullvalue: f64 = 0.0;

    let mut dummy_nularray = vec![0; (ncols * naxis2 * naxis3) as usize];

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        lpixel[0] = ncols as c_long;
        lpixel[1] = nrows as c_long;
        lpixel[2] = naxis3 as c_long;
        nullvalue = nulval; /* set local variable */

        fits_read_compressed_img(
            fptr,
            TDOUBLE,
            cast_slice(&fpixel),
            cast_slice(&lpixel),
            &inc,
            nullcheck,
            &Some(NullValue::Double(nullvalue)),
            cast_slice_mut(array),
            None,
            anynul,
            status,
        );
        return *status;
    }

    let tablerow = cmp::max(1, group);

    if ncols == naxis1 && nrows == naxis2 {
        /* arrays have same size? */

        /* all the image pixels are contiguous, so read all at once */
        ffgcld(
            fptr,
            2,
            tablerow as LONGLONG,
            1,
            naxis1 * naxis2 * naxis3,
            1,
            NullCheckType::SetPixel,
            nulval,
            array,
            &mut dummy_nularray,
            anynul.as_deref_mut(),
            status,
        );
        return *status;
    }

    if ncols < naxis1 || nrows < naxis2 {
        *status = BAD_DIMEN;
        return *status;
    }

    nfits = 1; /* next pixel in FITS image to read */
    narray = 0; /* next pixel in output array to be filled */

    /* loop over naxis3 planes in the data cube */
    for _jj in 0..(naxis3 as usize) {
        /* loop over the naxis2 rows in the FITS image, */
        /* reading naxis1 pixels to each row            */

        for _ii in 0..(naxis2 as usize) {
            if ffgcld(
                fptr,
                2,
                tablerow as LONGLONG,
                nfits,
                naxis1,
                1,
                NullCheckType::SetPixel,
                nulval,
                &mut array[narray..],
                &mut dummy_nularray,
                anynul.as_deref_mut(),
                status,
            ) > 0
            {
                return *status;
            }

            nfits += naxis1;
            narray += ncols as usize;
        }
        narray += ((nrows - naxis2) * ncols) as usize;
    }

    *status
}

/// Read a subsection of data values from an image or a table column.
/// This routine is set up to handle a maximum of nine dimensions.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) number of the column to read (1 = 1st)
/// * `naxis`  — (I) number of dimensions in the FITS array
/// * `naxes`  — (I) size of each dimension
/// * `blc`    — (I) 'bottom left corner' of the subsection
/// * `trc`    — (I) 'top right corner' of the subsection
/// * `inc`    — (I) increment to be applied in each dimension
/// * `nulval` — (I) value to set undefined pixels
/// * `array`  — (O) array to be filled and returned
/// * `anynul` — (O) set to 1 if any values are null; else 0
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgsvd(
    fptr: *mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: *const c_long,
    blc: *const c_long,
    trc: *const c_long,
    inc: *const c_long,
    nulval: f64,
    array: *mut f64,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let naxes = slice::from_raw_parts(naxes, naxis as usize);
        let blc = slice::from_raw_parts(blc, naxis as usize);
        let trc = slice::from_raw_parts(trc, naxis as usize);
        let inc = slice::from_raw_parts(inc, naxis as usize);

        let total_nelem = calculate_subsection_length(blc, trc, inc);

        let array = slice::from_raw_parts_mut(array, total_nelem);

        ffgsvd_safe(
            fptr, colnum, naxis, naxes, blc, trc, inc, nulval, array, anynul, status,
        )
    }
}

/// Read a subsection of data values from an image or a table column.
/// This routine is set up to handle a maximum of nine dimensions.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) number of the column to read (1 = 1st)
/// * `naxis`  — (I) number of dimensions in the FITS array
/// * `naxes`  — (I) size of each dimension
/// * `blc`    — (I) 'bottom left corner' of the subsection
/// * `trc`    — (I) 'top right corner' of the subsection
/// * `inc`    — (I) increment to be applied in each dimension
/// * `nulval` — (I) value to set undefined pixels
/// * `array`  — (O) array to be filled and returned
/// * `anynul` — (O) set to 1 if any values are null; else 0
/// * `status` — (IO) error status
pub fn ffgsvd_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: &[c_long],
    blc: &[c_long],
    trc: &[c_long],
    inc: &[c_long],
    nulval: f64,
    array: &mut [f64],
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut rstr: c_long = 0;
    let mut rstp: c_long = 0;
    let mut rinc: c_long = 0;
    let mut str: [c_long; 9] = [0; 9];
    let mut stp: [c_long; 9] = [0; 9];
    let mut incr: [c_long; 9] = [0; 9];
    let mut dir: [c_long; 9] = [0; 9];
    let mut nelem: c_long = 0;
    let mut nultyp = NullCheckType::None;
    let mut ninc: c_long = 0;
    let mut numcol: c_long = 0;
    let mut felem: LONGLONG = 0;
    let mut dsize: [LONGLONG; 10] = [0; 10];
    let mut blcll: [LONGLONG; 9] = [0; 9];
    let mut trcll: [LONGLONG; 9] = [0; 9];
    let mut hdutype: c_int = 0;
    let mut anyf: c_int = 0;
    let ldummy = 0;
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let nullcheck = NullCheckType::SetPixel;
    let mut nullvalue: f64 = 0.0;

    let naxis = naxis as usize;

    if naxis < 1 || naxis > 9 {
        int_snprintf!(
            &mut msg,
            FLEN_ERRMSG,
            "NAXIS = {} in call to ffgsvd is out of range",
            naxis,
        );
        ffpmsg_slice(&msg);
        *status = BAD_DIMEN;
        return *status;
    }

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        for ii in 0..naxis {
            blcll[ii] = blc[ii] as LONGLONG;
            trcll[ii] = trc[ii] as LONGLONG;
        }

        nullvalue = nulval; /* set local variable */

        fits_read_compressed_img(
            fptr,
            TDOUBLE,
            &blcll,
            &trcll,
            inc,
            nullcheck,
            &Some(NullValue::Double(nullvalue)),
            cast_slice_mut(array),
            None,
            anynul,
            status,
        );
        return *status;
    }

    /*
    if this is a primary array, then the input COLNUM parameter should
    be interpreted as the row number, and we will alway read the image
    data from column 2 (any group parameters are in column 1).
    */
    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    if hdutype == IMAGE_HDU {
        /* this is a primary array, or image extension */
        if colnum == 0 {
            rstr = 1;
            rstp = 1;
        } else {
            rstr = c_long::from(colnum);
            rstp = c_long::from(colnum);
        }
        rinc = 1;
        numcol = 2;
    } else {
        /* this is a table, so the row info is in the (naxis+1) elements */
        rstr = blc[naxis];
        rstp = trc[naxis];
        rinc = inc[naxis];
        numcol = c_long::from(colnum);
    }

    nultyp = NullCheckType::SetPixel;

    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = FALSE as c_int;
    }

    let mut i0 = 0;
    for ii in 0..9 {
        str[ii] = 1;
        stp[ii] = 1;
        incr[ii] = 1;
        dsize[ii] = 1;
        dir[ii] = 1;
    }

    for ii in 0..(naxis) {
        if trc[ii] < blc[ii] {
            if hdutype == IMAGE_HDU {
                dir[ii] = -1;
            } else {
                int_snprintf!(
                    &mut msg,
                    FLEN_ERRMSG,
                    "ffgsvd: illegal range specified for axis {}",
                    ii + 1,
                );
                ffpmsg_slice(&msg);
                *status = BAD_PIX_NUM;
                return *status;
            }
        }

        str[ii] = blc[ii];
        stp[ii] = trc[ii];
        incr[ii] = inc[ii];
        dsize[ii + 1] = dsize[ii] * naxes[ii] as LONGLONG;
        dsize[ii] *= dir[ii] as LONGLONG;
    }
    dsize[naxis] *= dir[naxis] as LONGLONG;

    if naxis == 1 && naxes[0] == 1 {
        /* This is not a vector column, so read all the rows at once */
        nelem = (rstp - rstr) / rinc + 1;
        ninc = rinc;
        rstp = rstr;
    } else {
        /* have to read each row individually, in all dimensions */
        nelem = (stp[0] * dir[0] - str[0] * dir[0]) / inc[0] + 1;
        ninc = incr[0] * dir[0];
    }

    for row in (rstr..=rstp).step_by(rinc as usize) {
        for i8 in ((str[8] * dir[8])..=(stp[8] * dir[8])).step_by(incr[8] as usize) {
            for i7 in ((str[7] * dir[7])..=(stp[7] * dir[7])).step_by(incr[7] as usize) {
                for i6 in ((str[6] * dir[6])..=(stp[6] * dir[6])).step_by(incr[6] as usize) {
                    for i5 in ((str[5] * dir[5])..=(stp[5] * dir[5])).step_by(incr[5] as usize) {
                        for i4 in ((str[4] * dir[4])..=(stp[4] * dir[4])).step_by(incr[4] as usize)
                        {
                            for i3 in
                                ((str[3] * dir[3])..=(stp[3] * dir[3])).step_by(incr[3] as usize)
                            {
                                for i2 in ((str[2] * dir[2])..=(stp[2] * dir[2]))
                                    .step_by(incr[2] as usize)
                                {
                                    for i1 in ((str[1] * dir[1])..=(stp[1] * dir[1]))
                                        .step_by(incr[1] as usize)
                                    {
                                        felem = (str[0] as LONGLONG)
                                            + (i1 - dir[1]) as LONGLONG * dsize[1]
                                            + (i2 - dir[2]) as LONGLONG * dsize[2]
                                            + (i3 - dir[3]) as LONGLONG * dsize[3]
                                            + (i4 - dir[4]) as LONGLONG * dsize[4]
                                            + (i5 - dir[5]) as LONGLONG * dsize[5]
                                            + (i6 - dir[6]) as LONGLONG * dsize[6]
                                            + (i7 - dir[7]) as LONGLONG * dsize[7]
                                            + (i8 - dir[8]) as LONGLONG * dsize[8];

                                        if ffgcld(
                                            fptr,
                                            numcol as c_int,
                                            row as LONGLONG,
                                            felem,
                                            nelem as LONGLONG,
                                            ninc,
                                            nultyp,
                                            nulval,
                                            &mut array[i0..],
                                            &mut [ldummy],
                                            Some(&mut anyf),
                                            status,
                                        ) > 0
                                        {
                                            return *status;
                                        }

                                        if anyf > 0
                                            && let Some(anynul) = anynul.as_deref_mut()
                                        {
                                            *anynul = TRUE as c_int;
                                        }
                                        i0 += nelem as usize;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    *status
}

/// Read a subsection of data values from an image or a table column.
/// This routine is set up to handle a maximum of nine dimensions.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `colnum`  — (I) number of the column to read (1 = 1st)
/// * `naxis`   — (I) number of dimensions in the FITS array
/// * `naxes`   — (I) size of each dimension
/// * `blc`     — (I) 'bottom left corner' of the subsection
/// * `trc`     — (I) 'top right corner' of the subsection
/// * `inc`     — (I) increment to be applied in each dimension
/// * `array`   — (O) array to be filled and returned
/// * `flagval` — (O) set to 1 if corresponding value is null
/// * `anynul`  — (O) set to 1 if any values are null; else 0
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgsfd(
    fptr: *mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: *const c_long,
    blc: *const c_long,
    trc: *const c_long,
    inc: *const c_long,
    array: *mut f64,
    flagval: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let naxes = slice::from_raw_parts(naxes, naxis as usize);
        let blc = slice::from_raw_parts(blc, naxis as usize);
        let trc = slice::from_raw_parts(trc, naxis as usize);
        let inc = slice::from_raw_parts(inc, naxis as usize);

        let total_nelem = calculate_subsection_length(blc, trc, inc);

        let array = slice::from_raw_parts_mut(array, total_nelem);
        let flagval = slice::from_raw_parts_mut(flagval, total_nelem);

        let anynul = anynul.as_mut();

        ffgsfd_safe(
            fptr, colnum, naxis, naxes, blc, trc, inc, array, flagval, anynul, status,
        )
    }
}

/// Read a subsection of data values from an image or a table column.
/// This routine is set up to handle a maximum of nine dimensions.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `colnum`  — (I) number of the column to read (1 = 1st)
/// * `naxis`   — (I) number of dimensions in the FITS array
/// * `naxes`   — (I) size of each dimension
/// * `blc`     — (I) 'bottom left corner' of the subsection
/// * `trc`     — (I) 'top right corner' of the subsection
/// * `inc`     — (I) increment to be applied in each dimension
/// * `array`   — (O) array to be filled and returned
/// * `flagval` — (O) set to 1 if corresponding value is null
/// * `anynul`  — (O) set to 1 if any values are null; else 0
/// * `status`  — (IO) error status
pub fn ffgsfd_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: &[c_long],
    blc: &[c_long],
    trc: &[c_long],
    inc: &[c_long],
    array: &mut [f64],
    flagval: &mut [c_char],
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut rstr: c_long = 0;
    let mut rstp: c_long = 0;
    let mut rinc: c_long = 0;
    let mut str: [c_long; 9] = [0; 9];
    let mut stp: [c_long; 9] = [0; 9];
    let mut incr: [c_long; 9] = [0; 9];
    let mut nelem: c_long = 0;
    let mut nultyp = NullCheckType::None;
    let mut ninc: c_long = 0;
    let mut numcol: c_long = 0;
    let mut felem: LONGLONG = 0;
    let mut dsize: [LONGLONG; 10] = [0; 10];
    let mut blcll: [LONGLONG; 9] = [0; 9];
    let mut trcll: [LONGLONG; 9] = [0; 9];
    let mut hdutype: c_int = 0;
    let mut anyf: c_int = 0;
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let nullcheck = NullCheckType::SetNullArray;
    let nullval: f64 = 0.0;

    let naxis = naxis as usize;

    if naxis < 1 || naxis > 9 {
        int_snprintf!(
            &mut msg,
            FLEN_ERRMSG,
            "NAXIS = {} in call to ffgsvd is out of range",
            naxis,
        );
        ffpmsg_slice(&msg);
        *status = BAD_DIMEN;
        return *status;
    }

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        for ii in 0..naxis {
            blcll[ii] = blc[ii] as LONGLONG;
            trcll[ii] = trc[ii] as LONGLONG;
        }

        fits_read_compressed_img(
            fptr,
            TDOUBLE,
            &blcll,
            &trcll,
            inc,
            nullcheck,
            &None,
            cast_slice_mut(array),
            Some(flagval),
            anynul,
            status,
        );
        return *status;
    }

    /*
    if this is a primary array, then the input COLNUM parameter should
    be interpreted as the row number, and we will alway read the image
    data from column 2 (any group parameters are in column 1).
    */
    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    if hdutype == IMAGE_HDU {
        /* this is a primary array, or image extension */
        if colnum == 0 {
            rstr = 1;
            rstp = 1;
        } else {
            rstr = c_long::from(colnum);
            rstp = c_long::from(colnum);
        }
        rinc = 1;
        numcol = 2;
    } else {
        /* this is a table, so the row info is in the (naxis+1) elements */
        rstr = blc[naxis];
        rstp = trc[naxis];
        rinc = inc[naxis];
        numcol = c_long::from(colnum);
    }

    nultyp = NullCheckType::SetNullArray;

    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = FALSE as c_int;
    }

    let mut i0 = 0;
    for ii in 0..9 {
        str[ii] = 1;
        stp[ii] = 1;
        incr[ii] = 1;
        dsize[ii] = 1;
    }

    for ii in 0..naxis {
        if trc[ii] < blc[ii] {
            int_snprintf!(
                &mut msg,
                FLEN_ERRMSG,
                "ffgsvd: illegal range specified for axis {}",
                ii + 1,
            );
            ffpmsg_slice(&msg);
            *status = BAD_PIX_NUM;
            return *status;
        }

        str[ii] = blc[ii];
        stp[ii] = trc[ii];
        incr[ii] = inc[ii];
        dsize[ii + 1] = dsize[ii] * naxes[ii] as LONGLONG;
    }

    if naxis == 1 && naxes[0] == 1 {
        /* This is not a vector column, so read all the rows at once */
        nelem = (rstp - rstr) / rinc + 1;
        ninc = rinc;
        rstp = rstr;
    } else {
        /* have to read each row individually, in all dimensions */
        nelem = (stp[0] - str[0]) / inc[0] + 1;
        ninc = incr[0];
    }

    for row in (rstr..=rstp).step_by(rinc as usize) {
        for i8 in (str[8]..=stp[8]).step_by(incr[8] as usize) {
            for i7 in (str[7]..=stp[7]).step_by(incr[7] as usize) {
                for i6 in (str[6]..=stp[6]).step_by(incr[6] as usize) {
                    for i5 in (str[5]..=stp[5]).step_by(incr[5] as usize) {
                        for i4 in (str[4]..=stp[4]).step_by(incr[4] as usize) {
                            for i3 in (str[3]..=stp[3]).step_by(incr[3] as usize) {
                                for i2 in (str[2]..=stp[2]).step_by(incr[2] as usize) {
                                    for i1 in (str[1]..=stp[1]).step_by(incr[1] as usize) {
                                        felem = (str[0] as LONGLONG)
                                            + (i1 as LONGLONG - 1) * dsize[1]
                                            + (i2 as LONGLONG - 1) * dsize[2]
                                            + (i3 as LONGLONG - 1) * dsize[3]
                                            + (i4 as LONGLONG - 1) * dsize[4]
                                            + (i5 as LONGLONG - 1) * dsize[5]
                                            + (i6 as LONGLONG - 1) * dsize[6]
                                            + (i7 as LONGLONG - 1) * dsize[7]
                                            + (i8 as LONGLONG - 1) * dsize[8];

                                        if ffgcld(
                                            fptr,
                                            numcol as c_int,
                                            row as LONGLONG,
                                            felem,
                                            nelem as LONGLONG,
                                            ninc,
                                            nultyp,
                                            nullval,
                                            &mut array[i0..],
                                            &mut flagval[i0..],
                                            Some(&mut anyf),
                                            status,
                                        ) > 0
                                        {
                                            return *status;
                                        }

                                        if anyf > 0
                                            && let Some(anynul) = anynul.as_deref_mut()
                                        {
                                            *anynul = TRUE as c_int;
                                        }

                                        i0 += nelem as usize;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    *status
}

///Read an array of group parameters from the primary array. Data conversion
///and scaling will be performed if necessary (e.g, if the datatype of
///the FITS array is not the same as the array being read).
///
///The primary array is represented as a binary table:
///each group of the primary array is a row in the table,
///where the first column contains the group parameters
///and the second column contains the image itself.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `group`     — (I) group to read (1 = 1st group)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffggpd(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: c_long,
    nelem: c_long,
    array: *mut f64,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let array = slice::from_raw_parts_mut(array, nelem as usize);

        ffggpd_safe(fptr, group, firstelem, nelem, array, status)
    }
}

///Read an array of group parameters from the primary array. Data conversion
///and scaling will be performed if necessary (e.g, if the datatype of
///the FITS array is not the same as the array being read).
///
///The primary array is represented as a binary table:
///each group of the primary array is a row in the table,
///where the first column contains the group parameters
///and the second column contains the image itself.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `group`     — (I) group to read (1 = 1st group)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `status`    — (IO) error status
pub fn ffggpd_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: c_long,
    nelem: c_long,
    array: &mut [f64],
    status: &mut c_int,
) -> c_int {
    let mut anynul = 0;
    let mut dummy_nularray = vec![0; nelem as usize];

    let row = cmp::max(1, group);

    ffgcld(
        fptr,
        1,
        row as LONGLONG,
        firstelem as LONGLONG,
        nelem as LONGLONG,
        1,
        NullCheckType::SetPixel,
        0.0,
        array,
        &mut dummy_nularray,
        Some(&mut anynul),
        status,
    );
    *status
}

/// Read an array of values from a column in the current FITS HDU. Automatic
/// datatype conversion will be performed if the datatype of the column does not
/// match the datatype of the array parameter. The output values will be scaled
/// by the FITS TSCALn and TZEROn values if these values have been defined.
/// Any undefined pixels will be set equal to the value of 'nulval' unless
/// nulval = 0 in which case no checks for undefined pixels will be made.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for null pixels
/// * `array`     — (O) array of values that are read
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcvd(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: f64,
    array: *mut f64,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynull = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, nelem as usize);

        ffgcvd_safe(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            nulval,
            array,
            anynull,
            status,
        )
    }
}

/// Read an array of values from a column in the current FITS HDU. Automatic
/// datatype conversion will be performed if the datatype of the column does not
/// match the datatype of the array parameter. The output values will be scaled
/// by the FITS TSCALn and TZEROn values if these values have been defined.
/// Any undefined pixels will be set equal to the value of 'nulval' unless
/// nulval = 0 in which case no checks for undefined pixels will be made.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for null pixels
/// * `array`     — (O) array of values that are read
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgcvd_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: f64,
    array: &mut [f64],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut dummy_nularray = vec![0; nelem as usize];

    ffgcld(
        fptr,
        colnum,
        firstrow,
        firstelem,
        nelem,
        1,
        NullCheckType::SetPixel,
        nulval,
        array,
        dummy_nularray.as_mut_slice(),
        anynul,
        status,
    );
    *status
}

/// Read an array of values from a column in the current FITS HDU. Automatic
/// datatype conversion will be performed if the datatype of the column does not
/// match the datatype of the array parameter. The output values will be scaled
/// by the FITS TSCALn and TZEROn values if these values have been defined.
/// Any undefined pixels will be set equal to the value of 'nulval' unless
/// nulval = 0 in which case no checks for undefined pixels will be made.
///
/// TSCAL and ZERO should not be used with complex values.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for null pixels
/// * `array`     — (O) array of values that are read
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcvm(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: f64,
    array: *mut f64,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, 2 * nelem as usize);

        ffgcvm_safe(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            nulval,
            array,
            anynul,
            status,
        )
    }
}

/// Read an array of values from a column in the current FITS HDU. Automatic
/// datatype conversion will be performed if the datatype of the column does not
/// match the datatype of the array parameter. The output values will be scaled
/// by the FITS TSCALn and TZEROn values if these values have been defined.
/// Any undefined pixels will be set equal to the value of 'nulval' unless
/// nulval = 0 in which case no checks for undefined pixels will be made.
///
/// TSCAL and ZERO should not be used with complex values.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for null pixels
/// * `array`     — (O) array of values that are read
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgcvm_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: f64,
    array: &mut [f64],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut dummy_nularray = vec![0; 2 * nelem as usize];

    /* a complex double value is interpreted as a pair of double values,   */
    /* thus need to multiply the first element and number of elements by 2 */

    ffgcld(
        fptr,
        colnum,
        firstrow,
        (firstelem - 1) * 2 + 1,
        nelem * 2,
        1,
        NullCheckType::SetPixel,
        nulval,
        array,
        &mut dummy_nularray,
        anynul,
        status,
    );
    *status
}

/// Read an array of values from a column in the current FITS HDU. Automatic
/// datatype conversion will be performed if the datatype of the column does not
/// match the datatype of the array parameter. The output values will be scaled
/// by the FITS TSCALn and TZEROn values if these values have been defined.
/// Nularray will be set = 1 if the corresponding array pixel is undefined,
/// otherwise nularray will = 0.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags: 1 if null pixel; else 0
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcfd(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut f64,
    nularray: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, nelem as usize);
        let nularray = slice::from_raw_parts_mut(nularray, nelem as usize);

        ffgcfd_safe(
            fptr, colnum, firstrow, firstelem, nelem, array, nularray, anynul, status,
        )
    }
}

/// Read an array of values from a column in the current FITS HDU. Automatic
/// datatype conversion will be performed if the datatype of the column does not
/// match the datatype of the array parameter. The output values will be scaled
/// by the FITS TSCALn and TZEROn values if these values have been defined.
/// Nularray will be set = 1 if the corresponding array pixel is undefined,
/// otherwise nularray will = 0.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags: 1 if null pixel; else 0
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgcfd_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [f64],
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let dummy: f64 = 0.0;

    ffgcld(
        fptr,
        colnum,
        firstrow,
        firstelem,
        nelem,
        1,
        NullCheckType::SetNullArray,
        dummy,
        array,
        cast_slice_mut(nularray),
        anynul,
        status,
    );
    *status
}

/// Read an array of values from a column in the current FITS HDU. Automatic
/// datatype conversion will be performed if the datatype of the column does not
/// match the datatype of the array parameter. The output values will be scaled
/// by the FITS TSCALn and TZEROn values if these values have been defined.
/// Nularray will be set = 1 if the corresponding array pixel is undefined,
/// otherwise nularray will = 0.
///
/// TSCAL and ZERO should not be used with complex values.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags: 1 if null pixel; else 0
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcfm(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut f64,
    nularray: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let nularray = slice::from_raw_parts_mut(nularray, 2 * nelem as usize);
        let array = slice::from_raw_parts_mut(array, 2 * nelem as usize);

        ffgcfm_safe(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            array,
            nularray,
            anynul,
            status,
        )
    }
}

/// Read an array of values from a column in the current FITS HDU. Automatic
/// datatype conversion will be performed if the datatype of the column does not
/// match the datatype of the array parameter. The output values will be scaled
/// by the FITS TSCALn and TZEROn values if these values have been defined.
/// Nularray will be set = 1 if the corresponding array pixel is undefined,
/// otherwise nularray will = 0.
///
/// TSCAL and ZERO should not be used with complex values.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags: 1 if null pixel; else 0
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgcfm_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [f64],
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let dummy: f64 = 0.0;

    /* a complex double value is interpreted as a pair of double values,   */
    /* thus need to multiply the first element and number of elements by 2 */

    /* allocate temporary array */
    let mut carray: Vec<c_char> = vec![0; (nelem as usize) * 2];

    ffgcld(
        fptr,
        colnum,
        firstrow,
        (firstelem - 1) * 2 + 1,
        nelem * 2,
        1,
        NullCheckType::SetNullArray,
        dummy,
        array,
        cast_slice_mut(carray.as_mut_slice()),
        anynul,
        status,
    );
    let mut ii = 0;
    for jj in 0..(nelem as usize) {
        if carray[ii] != 0 || carray[ii + 1] != 0 {
            nularray[jj] = 1;
        } else {
            nularray[jj] = 0;
        }
        ii += 2;
    }

    *status
}

/// Read an array of values from a column in the current FITS HDU.
/// The column number may refer to a real column in an ASCII or binary table,
/// or it may refer be a virtual column in a 1 or more grouped FITS primary
/// array or image extension.  FITSIO treats a primary array as a binary table
/// with 2 vector columns: the first column contains the group parameters (often
/// with length = 0) and the second column contains the array of image pixels.
/// Each row of the table represents a group in the case of multigroup FITS
/// images.
///
/// The output array of values will be converted from the datatype of the column
/// and will be scaled by the FITS TSCALn and TZEROn values if necessary.
#[allow(clippy::if_same_then_else)]
// C dispatch chain: distinct conditions deliberately share an action.
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `elemincre` — (I) pixel increment; e.g., 2 = every other
/// * `nultyp`    — (I) null value handling code:
/// * `nulval`    — (I) value for null pixels if nultyp = 1
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags = 1 if nultyp = 2
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub(crate) fn ffgcld(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    elemincre: c_long,
    nultyp: NullCheckType,
    /*     1: set undefined pixels = nulval        */
    /*     2: set nularray=1 for undefined pixels  */
    nulval: f64,
    array: &mut [f64],
    nularray: &mut [c_char],
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut scale: f64 = 0.0;
    let mut zero: f64 = 0.0;
    let mut power: f64 = 1.0;
    let mut dtemp: f64 = 0.0;
    let mut tcode: c_int = 0;
    let mut maxelem2: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut xcode: c_int = 0;
    let mut decimals: c_int = 0;
    let mut twidth: c_long = 0;
    let mut incre: c_long = 0;
    let mut ntodo: c_long = 0;
    let mut xwidth: c_long = 0;
    let mut convert: bool = false;
    let mut nulcheck = NullCheckType::None;
    let mut readcheck: c_int = 0;
    let mut repeat: LONGLONG = 0;
    let mut startpos: LONGLONG = 0;
    let mut elemnum: LONGLONG = 0;
    let mut readptr: LONGLONG = 0;
    let mut tnull: LONGLONG = 0;
    let mut rowlen: LONGLONG = 0;
    let mut rownum: LONGLONG = 0;
    let mut remain: LONGLONG = 0;
    let mut next: usize = 0;
    let mut rowincre: LONGLONG = 0;
    let mut maxelem: LONGLONG = 0;
    let mut tform: [c_char; 20] = [0; 20];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut snull: [c_char; 20] = [0; 20]; /*  the FITS null value if reading from ASCII table  */
    let mut buffer: [f64; DBUFFSIZE as usize / mem::size_of::<f64>()] =
        [0.0; DBUFFSIZE as usize / mem::size_of::<f64>()]; /* align cbuff on word boundary */

    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = 0;
    }

    if nultyp == NullCheckType::SetNullArray {
        nularray.fill(0); /* initialize nullarray */
    }

    /*  Check input and get parameters about the column: */
    if elemincre < 0 {
        readcheck -= 1; /* don't do range checking in this case */
    }

    if ffgcprll(
        fptr,
        colnum,
        firstrow,
        firstelem,
        nelem,
        readcheck,
        &mut scale,
        &mut zero,
        &mut tform,
        &mut twidth,
        &mut tcode,
        &mut maxelem2,
        &mut startpos,
        &mut elemnum,
        &mut incre,
        &mut repeat,
        &mut rowlen,
        &mut hdutype,
        &mut tnull,
        &mut snull,
        status,
    ) > 0
    {
        return *status;
    }

    maxelem = LONGLONG::from(maxelem2);

    incre *= elemincre; /* multiply incre to just get every nth pixel */

    if tcode == TSTRING && hdutype == ASCII_TBL {
        /* setup for ASCII tables */

        /* get the number of implied decimal places if no explicit decmal point */
        ffasfm_safe(
            &tform,
            Some(&mut xcode),
            Some(&mut xwidth),
            Some(&mut decimals),
            status,
        );
        for _ii in 0..(decimals as usize) {
            power *= 10.0;
        }
    }
    /*  Decide whether to check for null values in the input FITS file: */
    nulcheck = nultyp; /* by default, check for null values in the FITS file */

    if nultyp == NullCheckType::SetPixel && nulval == 0.0 {
        nulcheck = NullCheckType::None; /* calling routine does not want to check for nulls */
    } else if tcode % 10 == 1 && tnull == LONGLONG::from(NULL_UNDEFINED) {
        /* if reading an integer column, and  */
        /* if a null value is not defined,    */

        nulcheck = NullCheckType::None; /* then do not check for null values. */
    } else if tcode == TSHORT
        && (tnull > LONGLONG::from(c_short::MAX) || tnull < LONGLONG::from(c_short::MIN))
    {
        nulcheck = NullCheckType::None; /* Impossible null value */
    } else if tcode == TBYTE && (tnull > 255 || tnull < 0) {
        nulcheck = NullCheckType::None; /* Impossible null value */
    } else if tcode == TSTRING && snull[0] == ASCII_NULL_UNDEFINED {
        nulcheck = NullCheckType::None;
    }

    /*  If FITS column and output data array have same datatype, then we do */
    /*  not need to use a temporary buffer to store intermediate datatype.  */
    convert = true;
    if tcode == TDOUBLE {
        /* Special Case:                        */
        /* no type convertion required, so read */
        /* data directly into output buffer.    */

        if nelem < LONGLONG::from(INT32_MAX / 8) {
            maxelem = nelem;
        } else {
            maxelem = LONGLONG::from(INT32_MAX / 8);
        }

        if nulcheck == NullCheckType::None && scale == 1.0 && zero == 0.0 {
            convert = false; /* no need to scale data or find nulls */
        } else {
            /* Unlike the C, the conversion goes through the scratch buffer
            rather than in place in the output array, so each chunk must fit
            in the buffer. */
            maxelem = LONGLONG::from(maxelem2);
        }
    }

    /*  Now read the pixels from the FITS column. If the column does not   */
    /*  have the same datatype as the output array, then we have to read   */
    /*  the raw values into a temporary buffer (of limited size).  In      */
    /*  the case of a vector colum read only 1 vector of values at a time  */
    /*  then skip to the next row if more values need to be read.          */
    /*  After reading the raw values, then call the fffXXYY routine to (1) */
    /*  test for undefined values, (2) convert the datatype if necessary,  */
    /*  and (3) scale the values by the FITS TSCALn and TZEROn linear      */
    /*  scaling parameters.                                                */

    remain = nelem; /* remaining number of values to read */
    next = 0; /* next element in array to be read   */
    rownum = 0; /* row number, relative to firstrow   */

    while remain > 0 {
        /* limit the number of pixels to read at one time to the number that
           will fit in the buffer or to the number of pixels that remain in
           the current vector, which ever is smaller.
        */
        ntodo = cmp::min(remain, maxelem) as c_long;
        if elemincre >= 0 {
            ntodo = cmp::min(
                ntodo as LONGLONG,
                (repeat - elemnum - 1) / (elemincre as LONGLONG) + 1,
            ) as c_long;
        } else {
            ntodo = cmp::min(ntodo as LONGLONG, elemnum / (-elemincre as LONGLONG) + 1) as c_long;
        }

        readptr = startpos
            + ((rownum as LONGLONG) * rowlen)
            + (elemnum * (incre / elemincre) as LONGLONG);

        match tcode {
            TDOUBLE => {
                if convert {
                    ffgr8b(
                        fptr,
                        readptr,
                        ntodo,
                        incre,
                        cast_slice_mut(&mut buffer),
                        status,
                    );

                    fffr8r8(
                        cast_slice(&buffer),
                        ntodo,
                        scale,
                        zero,
                        nulcheck,
                        nulval,
                        nul_tail(nularray, next),
                        anynul.as_deref_mut(),
                        &mut array[next..],
                        status,
                    );
                } else {
                    ffgr8b(fptr, readptr, ntodo, incre, &mut array[next..], status);
                }
            }
            TBYTE => {
                ffgi1b(
                    fptr,
                    readptr,
                    ntodo,
                    incre,
                    cast_slice_mut(&mut buffer),
                    status,
                );

                fffi1r8(
                    cast_slice(&buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull as u8,
                    nulval,
                    nul_tail(nularray, next),
                    anynul.as_deref_mut(),
                    &mut array[next..],
                    status,
                );
            }
            TSHORT => {
                ffgi2b(
                    fptr,
                    readptr,
                    ntodo,
                    incre,
                    cast_slice_mut(&mut buffer),
                    status,
                );
                fffi2r8(
                    cast_slice(&buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull as _,
                    nulval,
                    nul_tail(nularray, next),
                    anynul.as_deref_mut(),
                    &mut array[next..],
                    status,
                );
            }
            TLONG => {
                ffgi4b(
                    fptr,
                    readptr,
                    ntodo,
                    incre,
                    cast_slice_mut(&mut buffer),
                    status,
                );
                fffi4r8(
                    cast_slice(&buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull as INT32BIT,
                    nulval,
                    nul_tail(nularray, next),
                    anynul.as_deref_mut(),
                    &mut array[next..],
                    status,
                );
            }
            TLONGLONG => {
                ffgi8b(
                    fptr,
                    readptr,
                    ntodo,
                    incre,
                    cast_slice_mut(&mut buffer),
                    status,
                );
                fffi8r8(
                    cast_slice(&buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull,
                    nulval,
                    nul_tail(nularray, next),
                    anynul.as_deref_mut(),
                    &mut array[next..],
                    status,
                );
            }
            TFLOAT => {
                ffgr4b(
                    fptr,
                    readptr,
                    ntodo,
                    incre,
                    cast_slice_mut(&mut buffer),
                    status,
                );
                fffr4r8(
                    cast_slice(&buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    nulval,
                    nul_tail(nularray, next),
                    anynul.as_deref_mut(),
                    &mut array[next..],
                    status,
                );
            }

            TSTRING => {
                ffmbyt_safe(fptr, readptr, REPORT_EOF, status);

                if incre == twidth {
                    /* contiguous bytes */
                    ffgbyt(
                        fptr,
                        (ntodo * twidth) as LONGLONG,
                        cast_slice_mut(&mut buffer),
                        status,
                    );
                } else {
                    ffgbytoff(
                        fptr,
                        twidth,
                        ntodo,
                        incre - twidth,
                        cast_slice_mut(&mut buffer),
                        status,
                    );
                }

                /* interpret the string as an ASCII formated number */
                fffstrr8(
                    cast_slice_mut(&mut buffer),
                    ntodo,
                    scale,
                    zero,
                    twidth,
                    power,
                    nulcheck,
                    &snull,
                    nulval,
                    nul_tail(nularray, next),
                    anynul.as_deref_mut(),
                    &mut array[next..],
                    status,
                );
            }
            _ => {
                /*  error trap for invalid column format */
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Cannot read numbers from column {} which has format {}",
                    colnum,
                    slice_to_str!(&tform),
                );
                ffpmsg_slice(&message);
                if hdutype == ASCII_TBL {
                    *status = BAD_ATABLE_FORMAT;
                    return *status;
                } else {
                    *status = BAD_BTABLE_FORMAT;
                    return *status;
                }
            }
        } /* End of switch block */

        /*  Check for fatal error  */
        if *status > 0 {
            /* test for error during previous read operation */

            dtemp = next as f64;
            if hdutype > 0 {
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Error reading elements {:.0} thru {:.0} from column {} (ffgcld).",
                    dtemp + 1.0,
                    dtemp + ntodo as f64,
                    colnum,
                );
            } else {
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Error reading elements {:.0} thru {:.0} from image (ffgcld).",
                    dtemp + 1.0,
                    dtemp + ntodo as f64,
                );
            }

            ffpmsg_slice(&message);
            return *status;
        }

        /*  increment the counters for the next loop  */
        remain -= ntodo as LONGLONG;
        if remain != 0 {
            next += ntodo as usize;
            elemnum += (ntodo * elemincre) as LONGLONG;

            if elemnum >= repeat
            /* completed a row; start on later row */
            {
                rowincre = elemnum / repeat;
                rownum += rowincre;
                elemnum -= rowincre * repeat;
            } else if elemnum < 0
            /* completed a row; start on a previous row */
            {
                rowincre = (-elemnum - 1) / repeat + 1;
                rownum -= rowincre;
                elemnum += rowincre * repeat;
            }
        }
    } /*  End of main while Loop  */

    /*  check for numerical overflow  */
    if *status == OVERFLOW_ERR {
        ffpmsg_str("Numerical overflow during type conversion while reading FITS data.");
        *status = NUM_OVERFLOW;
    }

    *status
}

/// Copy input to output following reading of the input from a FITS file.
/// Check for null values and do datatype conversion and scaling if required.
/// The nullcheck code value determines how any null values in the input array
/// are treated.  A null value is an input pixel that is equal to tnull.  If
/// nullcheck = 0, then no checking for nulls is performed and any null values
/// will be transformed just like any other pixel.  If nullcheck = 1, then the
/// output pixel will be set = nullval if the corresponding input pixel is null.
/// If nullcheck = 2, then if the pixel is null then the corresponding value of
/// nullarray will be set to 1; the value of nullarray for non-null pixels
/// will = 0.  The anynull parameter will be set = 1 if any of the returned
/// pixels are null, otherwise anynull will be returned with a value = 0;
///
/// # Parameters
///
/// * `input`     — (I) array of values to be converted
/// * `ntodo`     — (I) number of elements in the array
/// * `scale`     — (I) FITS TSCALn or BSCALE value
/// * `zero`      — (I) FITS TZEROn or BZERO value
/// * `nullcheck` — (I) null checking code; 0 = don't check
/// * `tnull`     — (I) value of FITS TNULLn keyword if any
/// * `nullval`   — (I) set null pixels, if nullcheck = 1
/// * `nullarray` — (O) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi1r8(
    input: &[u8],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: u8,
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
) -> c_int {
    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                output[ii] = input[ii].into(); /* copy input */
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                output[ii] = f64::from(input[ii]) * scale + zero;
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = input[ii].into();
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = f64::from(input[ii]) * scale + zero;
                }
            }
        }
    }
    *status
}

/// Copy input to output following reading of the input from a FITS file.
/// Check for null values and do datatype conversion and scaling if required.
/// The nullcheck code value determines how any null values in the input array
/// are treated.  A null value is an input pixel that is equal to tnull.  If
/// nullcheck = 0, then no checking for nulls is performed and any null values
/// will be transformed just like any other pixel.  If nullcheck = 1, then the
/// output pixel will be set = nullval if the corresponding input pixel is null.
/// If nullcheck = 2, then if the pixel is null then the corresponding value of
/// nullarray will be set to 1; the value of nullarray for non-null pixels
/// will = 0.  The anynull parameter will be set = 1 if any of the returned
/// pixels are null, otherwise anynull will be returned with a value = 0;
///
/// # Parameters
///
/// * `input`     — (I) array of values to be converted
/// * `ntodo`     — (I) number of elements in the array
/// * `scale`     — (I) FITS TSCALn or BSCALE value
/// * `zero`      — (I) FITS TZEROn or BZERO value
/// * `nullcheck` — (I) null checking code; 0 = don't check
/// * `tnull`     — (I) value of FITS TNULLn keyword if any
/// * `nullval`   — (I) set null pixels, if nullcheck = 1
/// * `nullarray` — (O) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi2r8(
    input: &[c_short],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: c_short,
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
) -> c_int {
    if nullcheck == NullCheckType::None {
        /* no null checking required */

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                output[ii] = input[ii].into();
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                output[ii] = f64::from(input[ii]) * scale + zero;
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = input[ii].into();
                }
            }
        } else {
            /* must scale the data */

            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = f64::from(input[ii]) * scale + zero;
                }
            }
        }
    }
    *status
}

/// Copy input to output following reading of the input from a FITS file.
/// Check for null values and do datatype conversion and scaling if required.
/// The nullcheck code value determines how any null values in the input array
/// are treated.  A null value is an input pixel that is equal to tnull.  If
/// nullcheck = 0, then no checking for nulls is performed and any null values
/// will be transformed just like any other pixel.  If nullcheck = 1, then the
/// output pixel will be set = nullval if the corresponding input pixel is null.
/// If nullcheck = 2, then if the pixel is null then the corresponding value of
/// nullarray will be set to 1; the value of nullarray for non-null pixels
/// will = 0.  The anynull parameter will be set = 1 if any of the returned
/// pixels are null, otherwise anynull will be returned with a value = 0;
///
/// # Parameters
///
/// * `input`     — (I) array of values to be converted
/// * `ntodo`     — (I) number of elements in the array
/// * `scale`     — (I) FITS TSCALn or BSCALE value
/// * `zero`      — (I) FITS TZEROn or BZERO value
/// * `nullcheck` — (I) null checking code; 0 = don't check
/// * `tnull`     — (I) value of FITS TNULLn keyword if any
/// * `nullval`   — (I) set null pixels, if nullcheck = 1
/// * `nullarray` — (O) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi4r8(
    input: &[INT32BIT],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: INT32BIT,
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
) -> c_int {
    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                output[ii] = input[ii].into();
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                output[ii] = f64::from(input[ii]) * scale + zero;
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = input[ii].into();
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = f64::from(input[ii]) * scale + zero;
                }
            }
        }
    }
    *status
}

/// Copy input to output following reading of the input from a FITS file.
/// Check for null values and do datatype conversion and scaling if required.
/// The nullcheck code value determines how any null values in the input array
/// are treated.  A null value is an input pixel that is equal to tnull.  If
/// nullcheck = 0, then no checking for nulls is performed and any null values
/// will be transformed just like any other pixel.  If nullcheck = 1, then the
/// output pixel will be set = nullval if the corresponding input pixel is null.
/// If nullcheck = 2, then if the pixel is null then the corresponding value of
/// nullarray will be set to 1; the value of nullarray for non-null pixels
/// will = 0.  The anynull parameter will be set = 1 if any of the returned
/// pixels are null, otherwise anynull will be returned with a value = 0;
///
/// # Parameters
///
/// * `input`     — (I) array of values to be converted
/// * `ntodo`     — (I) number of elements in the array
/// * `scale`     — (I) FITS TSCALn or BSCALE value
/// * `zero`      — (I) FITS TZEROn or BZERO value
/// * `nullcheck` — (I) null checking code; 0 = don't check
/// * `tnull`     — (I) value of FITS TNULLn keyword if any
/// * `nullval`   — (I) set null pixels, if nullcheck = 1
/// * `nullarray` — (O) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi8r8(
    input: &[LONGLONG],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: LONGLONG,
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
) -> c_int {
    let mut ulltemp: ULONGLONG = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */

        if scale == 1.0 && zero == 9223372036854775808. {
            /* The column we read contains unsigned long long values. */
            /* Instead of adding 9223372036854775808, it is more efficient */
            /* and more precise to just flip the sign bit with the XOR operator */

            for ii in 0..(ntodo as usize) {
                ulltemp = (input[ii] as ULONGLONG) ^ 0x8000000000000000;
                output[ii] = ulltemp as _;
            }
        } else if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                output[ii] = input[ii] as _; /* copy input to output */
            }
        } else {
            /* must scale the data */

            for ii in 0..(ntodo as usize) {
                output[ii] = (input[ii] as f64) * scale + zero;
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        if scale == 1.0 && zero == 9223372036854775808. {
            /* The column we read contains unsigned long long values. */
            /* Instead of adding 9223372036854775808, it is more efficient */
            /* and more precise to just flip the sign bit with the XOR operator */

            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    ulltemp = (input[ii] as ULONGLONG) ^ 0x8000000000000000;
                    output[ii] = ulltemp as _;
                }
            }
        } else if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = input[ii] as _;
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = (input[ii] as f64) * scale + zero;
                }
            }
        }
    }
    *status
}

/// Copy input to output following reading of the input from a FITS file.
/// Check for null values and do datatype conversion and scaling if required.
/// The nullcheck code value determines how any null values in the input array
/// are treated.  A null value is an input pixel that is equal to NaN.  If
/// nullcheck = 0, then no checking for nulls is performed and any null values
/// will be transformed just like any other pixel.  If nullcheck = 1, then the
/// output pixel will be set = nullval if the corresponding input pixel is null.
/// If nullcheck = 2, then if the pixel is null then the corresponding value of
/// nullarray will be set to 1; the value of nullarray for non-null pixels
/// will = 0.  The anynull parameter will be set = 1 if any of the returned
/// pixels are null, otherwise anynull will be returned with a value = 0;
///
/// # Parameters
///
/// * `input`     — (I) array of values to be converted
/// * `ntodo`     — (I) number of elements in the array
/// * `scale`     — (I) FITS TSCALn or BSCALE value
/// * `zero`      — (I) FITS TZEROn or BZERO value
/// * `nullcheck` — (I) null checking code; 0 = don't check
/// * `nullval`   — (I) set null pixels, if nullcheck = 1
/// * `nullarray` — (O) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffr4r8(
    input: &[f32],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
) -> c_int {
    let mut sptr = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                output[ii] = input[ii].into();
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                output[ii] = f64::from(input[ii]) * scale + zero;
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        if BYTESWAPPED {
            sptr += 1; /* point to MSBs */
        }

        let shortBuffer: &[c_short] = cast_slice(input);

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                let iret = fnan(shortBuffer[sptr]);
                if 0 != iret {
                    /* test for NaN or underflow */
                    if iret == 1 {
                        /* is it a NaN? */
                        *anynull = 1;
                        if nullcheck == NullCheckType::SetPixel {
                            output[ii] = nullval;
                        } else {
                            nullarray[ii] = 1;
                        }
                    } else {
                        /* it's an underflow */
                        output[ii] = 0.0;
                    }
                } else {
                    output[ii] = input[ii].into();
                }

                sptr += 2;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                let iret = fnan(shortBuffer[sptr]);
                if 0 != iret {
                    /* test for NaN or underflow */
                    if iret == 1 {
                        /* is it a NaN? */
                        *anynull = 1;
                        if nullcheck == NullCheckType::SetPixel {
                            output[ii] = nullval;
                        } else {
                            nullarray[ii] = 1;
                        }
                    } else {
                        /* it's an underflow */
                        output[ii] = zero as _;
                    }
                } else {
                    output[ii] = f64::from(input[ii]) * scale + zero;
                }

                sptr += 2;
            }
        }
    }
    *status
}

/// Copy input to output following reading of the input from a FITS file.
/// Check for null values and do datatype conversion and scaling if required.
/// The nullcheck code value determines how any null values in the input array
/// are treated.  A null value is an input pixel that is equal to NaN.  If
/// nullcheck = 0, then no checking for nulls is performed and any null values
/// will be transformed just like any other pixel.  If nullcheck = 1, then the
/// output pixel will be set = nullval if the corresponding input pixel is null.
/// If nullcheck = 2, then if the pixel is null then the corresponding value of
/// nullarray will be set to 1; the value of nullarray for non-null pixels
/// will = 0.  The anynull parameter will be set = 1 if any of the returned
/// pixels are null, otherwise anynull will be returned with a value = 0;
///
/// # Parameters
///
/// * `input`     — (I) array of values to be converted
/// * `ntodo`     — (I) number of elements in the array
/// * `scale`     — (I) FITS TSCALn or BSCALE value
/// * `zero`      — (I) FITS TZEROn or BZERO value
/// * `nullcheck` — (I) null checking code; 0 = don't check
/// * `nullval`   — (I) set null pixels, if nullcheck = 1
/// * `nullarray` — (O) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffr8r8(
    input: &[f64],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
) -> c_int {
    let mut sptr = 0;
    let mut iret = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                output[ii] = input[ii] as _;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                output[ii] = input[ii] * scale + zero;
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        let shortBuffer: &[c_short] = cast_slice(input);

        if BYTESWAPPED {
            sptr += 3; /* point to MSBs */
        }

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                iret = dnan(shortBuffer[sptr]);
                if 0 != iret {
                    /* test for NaN or underflow */
                    if iret == 1 {
                        /* is it a NaN? */
                        *anynull = 1;
                        if nullcheck == NullCheckType::SetPixel {
                            output[ii] = nullval;
                        } else {
                            nullarray[ii] = 1;
                            /* explicitly set value in case output contains a NaN */
                            output[ii] = DOUBLENULLVALUE;
                        }
                    } else {
                        /* it's an underflow */
                        output[ii] = 0.0;
                    }
                } else {
                    output[ii] = input[ii] as _;
                }
                sptr += 4;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                iret = dnan(shortBuffer[sptr]);
                if 0 != iret {
                    /* test for NaN or underflow */
                    if iret == 1 {
                        /* is it a NaN? */
                        *anynull = 1;
                        if nullcheck == NullCheckType::SetPixel {
                            output[ii] = nullval;
                        } else {
                            nullarray[ii] = 1;
                            /* explicitly set value in case output contains a NaN */
                            output[ii] = DOUBLENULLVALUE;
                        }
                    } else {
                        /* it's an underflow */
                        output[ii] = zero as _;
                    }
                } else {
                    output[ii] = input[ii] * scale + zero;
                }
                sptr += 4;
            }
        }
    }
    *status
}

/// Copy input to output following reading of the input from a FITS file.
/// Check for null values and do datatype conversion and scaling if required.
/// The nullcheck code value determines how any null values in the input array
/// are treated.  A null value is an input pixel that is equal to NaN.  If
/// nullcheck = 0, then no checking for nulls is performed and any null values
/// will be transformed just like any other pixel.  If nullcheck = 1, then the
/// output pixel will be set = nullval if the corresponding input pixel is null.
/// If nullcheck = 2, then if the pixel is null then the corresponding value of
/// nullarray will be set to 1; the value of nullarray for non-null pixels
/// will = 0.  The anynull parameter will be set = 1 if any of the returned
/// pixels are null, otherwise anynull will be returned with a value = 0;
///
/// # Parameters
///
/// * `input`     — (I) array of values to be converted
/// * `ntodo`     — (I) number of elements in the array
/// * `scale`     — (I) FITS TSCALn or BSCALE value
/// * `zero`      — (I) FITS TZEROn or BZERO value
/// * `nullcheck` — (I) null checking code; 0 = don't check
/// * `nullval`   — (I) set null pixels, if nullcheck = 1
/// * `nullarray` — (O) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `status`    — (IO) error status
pub(crate) fn fffr8r8_inplace(
    input: &mut [f64],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut sptr = 0;
    let mut iret = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                input[ii] = input[ii] as _;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                input[ii] = input[ii] * scale + zero;
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        if BYTESWAPPED {
            sptr += 3; /* point to MSBs */
        }

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                let shortBuffer: &[c_short] = cast_slice(input);
                iret = dnan(shortBuffer[sptr]);
                if 0 != iret {
                    /* test for NaN or underflow */
                    if iret == 1 {
                        /* is it a NaN? */
                        *anynull = 1;
                        if nullcheck == NullCheckType::SetPixel {
                            input[ii] = nullval;
                        } else {
                            nullarray[ii] = 1;
                            /* explicitly set value in case output contains a NaN */
                            input[ii] = DOUBLENULLVALUE;
                        }
                    } else {
                        /* it's an underflow */
                        input[ii] = 0.0;
                    }
                } else {
                    input[ii] = input[ii] as _;
                }
                sptr += 4;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                let shortBuffer: &[c_short] = cast_slice(input);
                iret = dnan(shortBuffer[sptr]);
                if 0 != iret {
                    /* test for NaN or underflow */
                    if iret == 1 {
                        /* is it a NaN? */
                        *anynull = 1;
                        if nullcheck == NullCheckType::SetPixel {
                            input[ii] = nullval;
                        } else {
                            nullarray[ii] = 1;
                            /* explicitly set value in case output contains a NaN */
                            input[ii] = DOUBLENULLVALUE;
                        }
                    } else {
                        /* it's an underflow */
                        input[ii] = zero as _;
                    }
                } else {
                    input[ii] = input[ii] * scale + zero;
                }
                sptr += 4;
            }
        }
    }
    *status
}

/// Copy input to output following reading of the input from a FITS file. Check
/// for null values and do scaling if required. The nullcheck code value
/// determines how any null values in the input array are treated. A null
/// value is an input pixel that is equal to snull.  If nullcheck= 0, then
/// no special checking for nulls is performed.  If nullcheck = 1, then the
/// output pixel will be set = nullval if the corresponding input pixel is null.
/// If nullcheck = 2, then if the pixel is null then the corresponding value of
/// nullarray will be set to 1; the value of nullarray for non-null pixels
/// will = 0.  The anynull parameter will be set = 1 if any of the returned
/// pixels are null, otherwise anynull will be returned with a value = 0;
///
/// # Parameters
///
/// * `input`      — (I) array of values to be converted
/// * `ntodo`      — (I) number of elements in the array
/// * `scale`      — (I) FITS TSCALn or BSCALE value
/// * `zero`       — (I) FITS TZEROn or BZERO value
/// * `twidth`     — (I) width of each substring of chars
/// * `implipower` — (I) power of 10 of implied decimal
/// * `nullcheck`  — (I) null checking code; 0 = don't check
/// * `snull`      — (I) value of FITS null string, if any
/// * `nullval`    — (I) set null pixels, if nullcheck = 1
/// * `nullarray`  — (O) bad pixel array, if nullcheck = 2
/// * `anynull`    — (O) set to 1 if any pixels are null
/// * `output`     — (O) array of converted pixels
/// * `status`     — (IO) error status
pub(crate) fn fffstrr8(
    input: &mut [c_char],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    twidth: c_long,
    implipower: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    snull: &[c_char],
    nullval: f64,
    nullarray: &mut [c_char],
    mut anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let chrzero: c_char = bb(b'0'); // 49
    let mut val: f64 = 0.0;
    let mut power: f64 = 0.0;
    let mut exponent: c_int = 0;
    let mut sign: c_int = 0;
    let mut esign: c_int = 0;
    let mut decpt: c_int = 0;

    let nullen = strlen_safe(snull);
    let mut cptr: usize = 0; /* pointer to start of input string */

    let twidth = twidth as usize;

    for ii in 0..(ntodo as usize) {
        let originalCptr = cptr;

        /* temporarily insert a null terminator at end of the string */
        let tpos = cptr + twidth;
        let tempstore = input[tpos];
        input[tpos] = 0;

        /* check if null value is defined, and if the    */
        /* column string is identical to the null string */
        if snull[0] != ASCII_NULL_UNDEFINED && strncmp_safe(snull, &input[cptr..], nullen) == 0 {
            if nullcheck != NullCheckType::None {
                let anynull = anynull.as_deref_mut().unwrap(); //Original code doesn't check if null pointer
                *anynull = 1;
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            }
            cptr += twidth;
        } else {
            /* value is not the null value, so decode it */
            /* remove any embedded blank characters from the string */

            let field_start = cptr; /* start of this fixed-width field */

            decpt = 0;
            sign = 1;
            val = 0.0;
            power = 1.;
            exponent = 0;
            esign = 1;

            while input[cptr] == bb(b' ') {
                /* skip leading blanks */
                cptr += 1;
            }

            if input[cptr] == bb(b'-') || input[cptr] == bb(b'+') {
                /* check for leading sign */

                if input[cptr] == bb(b'-') {
                    sign = -1;
                }

                cptr += 1;

                while input[cptr] == bb(b' ') {
                    /* skip blanks between sign and value */
                    cptr += 1;
                }
            }

            while input[cptr] >= bb(b'0') && input[cptr] <= bb(b'9') {
                val = val * 10.0 + f64::from(input[cptr] - chrzero); /* accumulate the value */
                cptr += 1;

                while input[cptr] == bb(b' ') {
                    /* skip embedded blanks in the value */
                    cptr += 1;
                }
            }

            if input[cptr] == bb(b'.') || input[cptr] == bb(b',') {
                /* check for decimal point */

                decpt = 1; /* set flag to show there was a decimal point */
                cptr += 1;
                while input[cptr] == bb(b' ') {
                    /* skip any blanks */
                    cptr += 1;
                }

                while input[cptr] >= bb(b'0') && input[cptr] <= bb(b'9') {
                    val = val * 10.0 + f64::from(input[cptr] - chrzero); /* accumulate the value */
                    power *= 10.;
                    cptr += 1;

                    while input[cptr] == bb(b' ') {
                        /* skip embedded blanks in the value */
                        cptr += 1;
                    }
                }
            }

            if input[cptr] == bb(b'E') || input[cptr] == bb(b'D') {
                /* check for exponent */

                cptr += 1;
                while input[cptr] == bb(b' ') {
                    /* skip blanks */
                    cptr += 1;
                }

                if input[cptr] == bb(b'-') || input[cptr] == bb(b'+') {
                    /* check for exponent sign */

                    if input[cptr] == bb(b'-') {
                        esign = -1;
                    }

                    cptr += 1;

                    while input[cptr] == bb(b' ') {
                        /* skip blanks between sign and exp */
                        cptr += 1;
                    }
                }

                while input[cptr] >= bb(b'0') && input[cptr] <= bb(b'9') {
                    exponent = exponent * 10 + c_int::from(input[cptr] - chrzero); /* accumulate exp */
                    cptr += 1;

                    while input[cptr] == bb(b' ') {
                        /* skip embedded blanks */
                        cptr += 1;
                    }
                }
            }

            if input[cptr] != 0 {
                /* should end up at the null terminator */

                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Cannot read number from ASCII table",
                );
                ffpmsg_slice(&message);
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Column field = {}.",
                    slice_to_str!(&input[originalCptr..]),
                );
                ffpmsg_slice(&message);
                /* restore the char that was overwritten by the null */
                input[tpos] = tempstore;
                *status = BAD_C2D;
                return *status;
            }

            if decpt == 0 {
                /* if no explicit decimal, use implied */
                power = implipower;
            }

            /* The manual walk above validates the field format and advances
            cptr, but computing the value as `val / power * 10^exp` is not
            correctly rounded (val itself loses precision past ~15 digits). Build
            a clean numeric string from the field -- dropping embedded blanks and
            mapping FITS 'D' exponents and ',' decimal separators -- and convert
            it with Rust's correctly-rounded parser so an ASCII round-trip is
            exact. Fall back to the manual result if parsing somehow fails. */
            let mut numstr = String::with_capacity(twidth as usize);
            for &c in &input[field_start..tpos] {
                match c as u8 {
                    b' ' => {}
                    b',' => numstr.push('.'),
                    b'D' | b'd' => numstr.push('E'),
                    other => numstr.push(other as char),
                }
            }
            dvalue = numstr
                .parse::<f64>()
                .unwrap_or((f64::from(sign) * val / power) * 10.0_f64.powi(esign * exponent));

            output[ii] = dvalue * scale + zero; /* apply the scaling */
        }
        /* restore the char that was overwritten by the null */
        input[tpos] = tempstore;
    }
    *status
}

// Ported from test_getcold.c - double precision read functions.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aliases::rust_api::*;
    use crate::fitsio::{
        ASCII_TBL, BAD_COL_NUM, BAD_DIMEN, BAD_PIX_NUM, BINARY_TBL, BYTE_IMG, DOUBLE_IMG,
        FLOAT_IMG, LONG_IMG, LONGLONG, LONGLONG_IMG, READONLY, SHORT_IMG, fitsfile,
    };
    use crate::helpers::testhelpers::{to_buf, with_temp_file};
    use libc::{c_char, c_int, c_long};

    /// Make a NUL-terminated `Vec<c_char>` from a `&str`.
    fn cc(s: &str) -> Vec<c_char> {
        let mut v: Vec<c_char> = s.bytes().map(|b| b as c_char).collect();
        v.push(0);
        v
    }

    /// Create a single-column table (binary or ASCII) and return the open file.
    fn make_table(
        name: &[c_char],
        tbltype: c_int,
        ttype: &str,
        tform: &str,
        nrows: LONGLONG,
        status: &mut c_int,
    ) -> Option<Box<fitsfile>> {
        let mut f: Option<Box<fitsfile>> = None;
        fits_create_file(&mut f, name, status);
        fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], status);
        let ttype_v = [Some(cc(ttype))];
        let ttype_ref: Vec<Option<&[c_char]>> = ttype_v.iter().map(|o| o.as_deref()).collect();
        let tform_v = [cc(tform)];
        let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
        fits_create_tbl(
            f.as_deref_mut().unwrap(),
            tbltype,
            nrows,
            1,
            &ttype_ref,
            &tform_ref,
            None,
            None,
            status,
        );
        f
    }

    #[test]
    fn test_read_primary_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [f64; 5] = [-1.0e100, -1.0, 0.0, 1.0, 1.0e100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], data[0]);
            assert_eq!(result[2], 0.0);
            assert_eq!(result[4], data[4]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_primary_with_null_value() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [f64; 5] = [1.0, 2.0, -999.0, 4.0, 5.0];

            // For floating point images, IEEE NaN is the null value.
            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            // Read with null substitution value.
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                -123.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1.0);
            assert_eq!(result[2], -999.0); // No BLANK in floating point.
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_primary_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [f64; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 5];
            let mut nularray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_imgnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 10.0);
            assert_eq!(result[4], 50.0);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_2d_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 3]; // 4 cols x 3 rows.
            let data: [f64; 12] = core::array::from_fn(|i| (i as f64) * 100.5);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 12, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 12];
            let mut anynull = -1;
            fits_read_2d_dbl(
                f.as_deref_mut().unwrap(),
                1,
                0.0,
                4,
                4,
                3,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 0.0);
            assert_eq!(result[5], 5.0 * 100.5);
            assert_eq!(result[11], 11.0 * 100.5);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_3d_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 3] = [2, 2, 2]; // 2x2x2 = 8 elements.
            let data: [f64; 8] = core::array::from_fn(|i| ((i + 1) as f64) * 0.5);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                3,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 8, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 8];
            let mut anynull = -1;
            fits_read_3d_dbl(
                f.as_deref_mut().unwrap(),
                1,
                0.0,
                2,
                2,
                2,
                2,
                2,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 0.5);
            assert_eq!(result[7], 4.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_subsection() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let data: [f64; 16] = core::array::from_fn(|i| i as f64);
            let fpixel: [c_long; 2] = [2, 2];
            let lpixel: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 16, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 4];
            let mut anynull = -1;
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                2,
                &naxes,
                &fpixel,
                &lpixel,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 5.0);
            assert_eq!(result[1], 6.0);
            assert_eq!(result[2], 9.0);
            assert_eq!(result[3], 10.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_subsection_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let data: [f64; 16] = core::array::from_fn(|i| i as f64);
            let fpixel: [c_long; 2] = [1, 1];
            let lpixel: [c_long; 2] = [2, 2];
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 16, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 4];
            let mut nularray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_subsetnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                2,
                &naxes,
                &fpixel,
                &lpixel,
                &inc,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 0.0);
            assert_eq!(result[1], 1.0);
            assert_eq!(result[2], 4.0);
            assert_eq!(result[3], 5.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_subsection_with_increment() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [6, 6];
            let data: [f64; 36] = core::array::from_fn(|i| i as f64);
            let fpixel: [c_long; 2] = [1, 1];
            let lpixel: [c_long; 2] = [5, 5];
            let inc: [c_long; 2] = [2, 2]; // Skip every other pixel.

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 36, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            // inc=[2,2] over [1..5] reads 3x3=9 elements; C relies on stack
            // overrun of `result[4]`, so size the Rust buffer to fit.
            let mut result = [0.0f64; 9];
            let mut anynull = -1;
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                2,
                &naxes,
                &fpixel,
                &lpixel,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // With increment 2, reading [1,3,5] x [1,3,5].
            assert_eq!(result[0], 0.0); // (1,1) -> index 0.
            assert_eq!(result[1], 2.0); // (3,1) -> index 2.
            assert_eq!(result[2], 4.0); // (5,1) -> index 4.
            assert_eq!(result[3], 12.0); // (1,3) -> index 12.
            assert_eq!(result[4], 14.0); // (3,3) -> index 14.
            assert_eq!(result[5], 16.0); // (5,3) -> index 16.
            assert_eq!(result[6], 24.0); // (1,5) -> index 24.
            assert_eq!(result[7], 26.0); // (3,5) -> index 26.
            assert_eq!(result[8], 28.0); // (5,5) -> index 28.
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 3] = [-1.0e100, 0.0, 1.0e100];

            let mut f = make_table(&name, BINARY_TBL, "DBLCOL", "1D", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], -1.0e100);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], 1.0e100);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_column_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 3] = [10.0, 20.0, 30.0];

            let mut f = make_table(&name, BINARY_TBL, "DBLCOL", "1D", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut nularray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 10.0);
            assert_eq!(result[1], 20.0);
            assert_eq!(result[2], 30.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_vector_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 5] = [1.1, 2.2, 3.3, 4.4, 5.5]; // 5 doubles per row.

            let mut f = make_table(&name, BINARY_TBL, "DBLVEC", "5D", 1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                5,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1.1);
            assert_eq!(result[4], 5.5);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_byte_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [u8; 3] = [0, 127, 255];

            let mut f = make_table(&name, BINARY_TBL, "BYTECOL", "1B", 3, &mut status);
            fits_write_col_byt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 0.0);
            assert_eq!(result[1], 127.0);
            assert_eq!(result[2], 255.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_short_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [i16; 3] = [-32768, 0, 32767];

            let mut f = make_table(&name, BINARY_TBL, "SHORTCOL", "1I", 3, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], -32768.0);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], 32767.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_long_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 3] = [-1000000, 0, 1000000];

            let mut f = make_table(&name, BINARY_TBL, "LONGCOL", "1J", 3, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], -1000000.0);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], 1000000.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_float_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f32; 3] = [-1.23e20f32, 0.0f32, 1.23e20f32];

            let mut f = make_table(&name, BINARY_TBL, "FLOATCOL", "1E", 3, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Float to double conversion preserves the float value.
            assert!((result[0] - data[0] as f64).abs() <= 1.0e10);
            assert_eq!(result[1], 0.0);
            assert!((result[2] - data[2] as f64).abs() <= 1.0e10);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_longlong_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 3] = [-1000000000, 0, 1000000000];

            let mut f = make_table(&name, BINARY_TBL, "LLCOL", "1K", 3, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], -1000000000.0);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], 1000000000.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_with_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 1] = [50.0];

            let mut f = make_table(&name, BINARY_TBL, "SCALED", "1D", 1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            // Set scaling: result = data * 2.0 + 10.0 = 110.0.
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TSCAL1"),
                2.0,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                10.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 1];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 110.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_bad_col_num() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut f = make_table(&name, BINARY_TBL, "DBLCOL", "1D", 1, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 1];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                0,
                1,
                1,
                1,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_COL_NUM);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_ascii_table() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 3] = [-1.23456789012345e10, 0.0, 1.23456789012345e10];

            let mut f = make_table(&name, ASCII_TBL, "DBLCOL", "D25.17", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((result[0] - data[0]).abs() <= 1.0);
            assert_eq!(result[1], 0.0);
            assert!((result[2] - data[2]).abs() <= 1.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_complex_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 complex nums.

            let mut f = make_table(&name, BINARY_TBL, "CMPLXCOL", "1M", 3, &mut status);
            fits_write_col_dblcmp(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 6];
            let mut anynull = -1;
            fits_read_col_dblcmp(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1.0);
            assert_eq!(result[1], 2.0);
            assert_eq!(result[4], 5.0);
            assert_eq!(result[5], 6.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_complex_column_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 4] = [1.0, 2.0, 3.0, 4.0]; // 2 complex nums.

            let mut f = make_table(&name, BINARY_TBL, "CMPLXCOL", "1M", 2, &mut status);
            fits_write_col_dblcmp(f.as_deref_mut().unwrap(), 1, 1, 1, 2, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 4];
            let mut nularray = [0 as c_char; 2];
            let mut anynull = -1;
            // ffgcfm has no `rust_api` alias; call the safe fn directly.
            ffgcfm_safe(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                2,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1.0);
            assert_eq!(result[1], 2.0);
            assert_eq!(result[2], 3.0);
            assert_eq!(result[3], 4.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_integer_image() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [c_long; 5] = [-1000, -1, 0, 1, 1000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), LONG_IMG, 1, &naxes, &mut status);
            fits_write_img_lng(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], -1000.0);
            assert_eq!(result[2], 0.0);
            assert_eq!(result[4], 1000.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_short_image() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [i16; 5] = [-32768, -100, 0, 100, 32767];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], -32768.0);
            assert_eq!(result[2], 0.0);
            assert_eq!(result[4], 32767.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_float_image() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [f32; 5] = [-1.5e30f32, -1.0f32, 0.0f32, 1.0f32, 1.5e30f32];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), FLOAT_IMG, 1, &naxes, &mut status);
            fits_write_img_flt(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((result[0] - data[0] as f64).abs() <= 1.0e20);
            assert_eq!(result[2], 0.0);
            assert!((result[4] - data[4] as f64).abs() <= 1.0e20);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_with_image_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [3];
            let data: [c_long; 3] = [10, 20, 30];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), LONG_IMG, 1, &naxes, &mut status);
            fits_write_img_lng(f.as_deref_mut().unwrap(), 1, 1, 3, &data, &mut status);
            // Set scaling: result = data * 0.5 + 100.0.
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("BSCALE"),
                0.5,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("BZERO"),
                100.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 105.0); // 10 * 0.5 + 100.
            assert_eq!(result[1], 110.0); // 20 * 0.5 + 100.
            assert_eq!(result[2], 115.0); // 30 * 0.5 + 100.
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_group_parameter() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [3];
            let pdata: [f64; 2] = [1.1, 2.2];

            // Create random groups with 2 parameters.
            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_grphdr(
                f.as_deref_mut().unwrap(),
                1,
                DOUBLE_IMG,
                1,
                &naxes,
                2,
                1,
                1,
                &mut status,
            );
            fits_write_grppar_dbl(f.as_deref_mut().unwrap(), 1, 1, 2, &pdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 2];
            fits_read_grppar_dbl(f.as_deref_mut().unwrap(), 1, 1, 2, &mut result, &mut status);
            assert_eq!(status, 0);
            assert_eq!(result[0], 1.1);
            assert_eq!(result[1], 2.2);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_multiple_rows() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

            let mut f = make_table(&name, BINARY_TBL, "DBLCOL", "1D", 5, &mut status);
            for i in 0..5 {
                fits_write_col_dbl(
                    f.as_deref_mut().unwrap(),
                    1,
                    i + 1,
                    1,
                    1,
                    &data[i as usize..],
                    &mut status,
                );
            }
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                5,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1.0);
            assert_eq!(result[2], 3.0);
            assert_eq!(result[4], 5.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_large_values() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [3];
            let data: [f64; 3] = [f64::MIN_POSITIVE, 0.0, f64::MAX];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], f64::MIN_POSITIVE);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], f64::MAX);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_special_values() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [2];
            let data: [f64; 2] = [f64::INFINITY, f64::NEG_INFINITY];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 2, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 2];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                2,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // HUGE_VAL typically stored as infinity.
            assert_eq!(result[0], f64::INFINITY);
            assert_eq!(result[1], f64::NEG_INFINITY);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_byte_image_as_double() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [3];
            let data: [u8; 3] = [0, 128, 255];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            fits_write_img_byt(f.as_deref_mut().unwrap(), 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 0.0);
            assert_eq!(result[1], 128.0);
            assert_eq!(result[2], 255.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_longlong_image_as_double() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [3];
            let data: [LONGLONG; 3] = [-1000000000000, 0, 1000000000000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                LONGLONG_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_lnglng(f.as_deref_mut().unwrap(), 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], -1000000000000.0);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], 1000000000000.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_bad_dimension() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [10, 10];
            let data: [f64; 100] = core::array::from_fn(|i| i as f64);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 100, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            // Try to read with ncols < naxis1 - should fail with BAD_DIMEN.
            let mut result = [0.0f64; 4];
            let mut anynull = -1;
            fits_read_3d_dbl(
                f.as_deref_mut().unwrap(),
                1,
                0.0,
                5,
                5,
                10,
                10,
                1,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_DIMEN);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_reverse_order() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [5, 5];
            let data: [f64; 25] = core::array::from_fn(|i| i as f64);
            let fpixel: [c_long; 2] = [4, 4]; // Start at (4,4).
            let lpixel: [c_long; 2] = [2, 2]; // End at (2,2) - reverse!
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 25, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            // Reading reverse order on image - exercises dir=-1 code path.
            // The result is more complex - just verify it works without error.
            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 9];
            let mut anynull = -1;
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                2,
                &naxes,
                &fpixel,
                &lpixel,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Just verify no error and we got some data.
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_table_subsection() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data1: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let data2: [f64; 6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
            let data3: [f64; 6] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
            let naxes: [c_long; 1] = [6];
            let blc: [c_long; 2] = [2, 1]; // Start at element 2, row 1.
            let trc: [c_long; 2] = [3, 3]; // End at element 3, row 3.
            let inc: [c_long; 2] = [1, 1];

            let mut f = make_table(&name, BINARY_TBL, "VECTOR", "6D", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 6, &data1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 2, 1, 6, &data2, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 3, 1, 6, &data3, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            // Read subsection from table: elements 2-3 from rows 1-3.
            // Reads 2 elems x 3 rows = 6 values; C relies on stack overrun
            // of `result[4]`, so size the Rust buffer to fit.
            let mut result = [0.0f64; 6];
            let mut anynull = -1;
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                &naxes,
                &blc,
                &trc,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Row 1: elem 2,3 = 2.0, 3.0. Row 2: 20.0, 30.0. Row 3: 200.0, 300.0.
            assert_eq!(result[0], 2.0);
            assert_eq!(result[1], 3.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_unsigned_longlong() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 3] = [0, 1, 9223372036854775807];

            let mut f = make_table(&name, BINARY_TBL, "ULONGCOL", "1K", 3, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            // Set TZERO to 9223372036854775808 to treat as unsigned.
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                9223372036854775808.,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // With TZERO=9223372036854775808 values become large unsigned.
            assert!(result[0] >= 9.0e18);
            assert!(result[1] >= 9.0e18);
            assert!(result[2] >= 1.8e19);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_byte_with_null_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [u8; 5] = [0, 50, 100, 200, 255];

            let mut f = make_table(&name, BINARY_TBL, "BYTECOL", "1B", 5, &mut status);
            fits_write_col_byt(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &data, &mut status);
            // Set TNULL, TSCAL, TZERO.
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                255,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TSCAL1"),
                2.0,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                100.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                5,
                -999.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // 0*2+100=100, 50*2+100=200, 100*2+100=300, 200*2+100=500, 255=null=-999.
            assert_eq!(result[0], 100.0);
            assert_eq!(result[1], 200.0);
            assert_eq!(result[2], 300.0);
            assert_eq!(result[3], 500.0);
            assert_eq!(result[4], -999.0);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_short_with_null_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [i16; 4] = [-100, 0, 100, -32768];

            let mut f = make_table(&name, BINARY_TBL, "SHORTCOL", "1I", 4, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 4, &data, &mut status);
            // Set TNULL, TSCAL, TZERO.
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                -32768,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TSCAL1"),
                0.5,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                50.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 4];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                4,
                -999.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // -100*0.5+50=0, 0*0.5+50=50, 100*0.5+50=100, -32768=null=-999.
            assert_eq!(result[0], 0.0);
            assert_eq!(result[1], 50.0);
            assert_eq!(result[2], 100.0);
            assert_eq!(result[3], -999.0);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_int_with_null_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 4] = [100, 200, 300, -2147483647];

            let mut f = make_table(&name, BINARY_TBL, "INTCOL", "1J", 4, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 4, &data, &mut status);
            // Set TNULL, TSCAL, TZERO.
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                -2147483647,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TSCAL1"),
                3.0,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                1000.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 4];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                4,
                -999.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // 100*3+1000=1300, 200*3+1000=1600, 300*3+1000=1900, null=-999.
            assert_eq!(result[0], 1300.0);
            assert_eq!(result[1], 1600.0);
            assert_eq!(result[2], 1900.0);
            assert_eq!(result[3], -999.0);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_longlong_with_null_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 4] = [1000, 2000, 3000, -9223372036854775807];

            let mut f = make_table(&name, BINARY_TBL, "LLCOL", "1K", 4, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 4, &data, &mut status);
            // Set TNULL, TSCAL, TZERO.
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                -9223372036854775807,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TSCAL1"),
                2.0,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                5000.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 4];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                4,
                -999.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // 1000*2+5000=7000, 2000*2+5000=9000, 3000*2+5000=11000, null=-999.
            assert_eq!(result[0], 7000.0);
            assert_eq!(result[1], 9000.0);
            assert_eq!(result[2], 11000.0);
            assert_eq!(result[3], -999.0);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_float_with_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f32; 3] = [1.0f32, 2.0f32, 3.0f32];

            let mut f = make_table(&name, BINARY_TBL, "FLOATCOL", "1E", 3, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            // Set TSCAL, TZERO.
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TSCAL1"),
                10.0,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                100.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // 1*10+100=110, 2*10+100=120, 3*10+100=130.
            assert!((result[0] - 110.0).abs() <= 0.001);
            assert!((result[1] - 120.0).abs() <= 0.001);
            assert!((result[2] - 130.0).abs() <= 0.001);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_double_with_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [3];
            let data: [f64; 3] = [10.0, 20.0, 30.0];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 3, &data, &mut status);
            // Set BSCALE, BZERO.
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("BSCALE"),
                0.1,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("BZERO"),
                1000.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_img_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // 10*0.1+1000=1001, 20*0.1+1000=1002, 30*0.1+1000=1003.
            assert!((result[0] - 1001.0).abs() <= 0.001);
            assert!((result[1] - 1002.0).abs() <= 0.001);
            assert!((result[2] - 1003.0).abs() <= 0.001);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_byte_with_null_noscaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [u8; 4] = [10, 20, 30, 255];

            let mut f = make_table(&name, BINARY_TBL, "BYTECOL", "1B", 4, &mut status);
            fits_write_col_byt(f.as_deref_mut().unwrap(), 1, 1, 1, 4, &data, &mut status);
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                255,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 4];
            let mut nularray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_colnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                4,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 10.0);
            assert_eq!(result[1], 20.0);
            assert_eq!(result[2], 30.0);
            assert_eq!(nularray[3], 1);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_short_with_null_noscaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [i16; 4] = [100, 200, 300, -32768];

            let mut f = make_table(&name, BINARY_TBL, "SHORTCOL", "1I", 4, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 4, &data, &mut status);
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                -32768,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 4];
            let mut nularray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_colnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                4,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 100.0);
            assert_eq!(result[1], 200.0);
            assert_eq!(result[2], 300.0);
            assert_eq!(nularray[3], 1);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_int_with_null_noscaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 4] = [1000, 2000, 3000, -2147483647];

            let mut f = make_table(&name, BINARY_TBL, "INTCOL", "1J", 4, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 4, &data, &mut status);
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                -2147483647,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 4];
            let mut nularray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_colnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                4,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1000.0);
            assert_eq!(result[1], 2000.0);
            assert_eq!(result[2], 3000.0);
            assert_eq!(nularray[3], 1);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_longlong_with_null_noscaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 4] = [10000, 20000, 30000, -9223372036854775807];

            let mut f = make_table(&name, BINARY_TBL, "LLCOL", "1K", 4, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 4, &data, &mut status);
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                -9223372036854775807,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 4];
            let mut nularray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_colnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                4,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 10000.0);
            assert_eq!(result[1], 20000.0);
            assert_eq!(result[2], 30000.0);
            assert_eq!(nularray[3], 1);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_unsigned_longlong_with_null() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 3] = [0, 1, -1]; // -1 will be null.

            let mut f = make_table(&name, BINARY_TBL, "ULONGCOL", "1K", 3, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            // Set TZERO for unsigned and TNULL.
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                9223372036854775808.,
                15,
                None,
                &mut status,
            );
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                -1,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut nularray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(nularray[2], 1);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_naxis_boundary() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [1];
            let blc: [c_long; 1] = [1];
            let trc: [c_long; 1] = [1];
            let inc: [c_long; 1] = [1];

            // Test with naxis out of range.
            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 1];
            let mut anynull = -1;
            // naxis = 0 should fail.
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                0,
                &naxes,
                &blc,
                &trc,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_DIMEN);
            status = 0;
            // naxis = 10 should fail.
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                10,
                &naxes,
                &blc,
                &trc,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_DIMEN);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_ascii_with_exponent() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 2] = [1.2345678e10, -9.87654e-5];

            let mut f = make_table(&name, ASCII_TBL, "DBLCOL", "E15.7", 2, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 2, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 2];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                2,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((result[0] - data[0]).abs() / data[0] <= 1e-6);
            assert!((result[1] - data[1]).abs() / data[1].abs() <= 1e-4);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_ascii_with_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 3] = [100.0, 200.0, 300.0];

            let mut f = make_table(&name, ASCII_TBL, "DBLCOL", "F10.2", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            // Set TSCAL, TZERO.
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TSCAL1"),
                2.0,
                15,
                None,
                &mut status,
            );
            fits_write_key_dbl(
                f.as_deref_mut().unwrap(),
                &cc("TZERO1"),
                50.0,
                15,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0.0f64; 3];
            let mut anynull = -1;
            fits_read_col_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // result = data * TSCAL + TZERO.
            assert!((result[0] - 250.0).abs() <= 0.01); // 100 * 2 + 50 = 250.
            assert!((result[1] - 450.0).abs() <= 0.01); // 200 * 2 + 50 = 450.
            assert!((result[2] - 650.0).abs() <= 0.01); // 300 * 2 + 50 = 650.
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_single_element_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0]; // Single element per row.
            let naxes: [c_long; 1] = [1];
            let blc: [c_long; 2] = [1, 1];
            let trc: [c_long; 2] = [1, 5];
            let inc: [c_long; 2] = [1, 1];

            let mut f = make_table(&name, BINARY_TBL, "SCALAR", "1D", 5, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            // Read all 5 rows of single-element column.
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                &naxes,
                &blc,
                &trc,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1.0);
            assert_eq!(result[2], 3.0);
            assert_eq!(result[4], 5.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_negative_elemincre() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];
            let data: [f64; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let fpixel: [c_long; 1] = [10];
            let lpixel: [c_long; 1] = [1];
            let inc: [c_long; 1] = [2]; // Skip every other.

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 10, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            // Read from 10 to 1 with step 2 - should read 9,7,5,3,1.
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                &naxes,
                &fpixel,
                &lpixel,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 9.0);
            assert_eq!(result[1], 7.0);
            assert_eq!(result[2], 5.0);
            assert_eq!(result[3], 3.0);
            assert_eq!(result[4], 1.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_colnum_zero() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
            let blc: [c_long; 1] = [1];
            let trc: [c_long; 1] = [5];
            let inc: [c_long; 1] = [1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            // colnum=0 on IMAGE_HDU means use default row 1.
            let mut result = [0.0f64; 5];
            let mut anynull = -1;
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                0,
                1,
                &naxes,
                &blc,
                &trc,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1.0);
            assert_eq!(result[4], 5.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_3d_noncontiguous() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 3] = [3, 3, 2]; // 3x3x2 = 18 elements.
            let data: [f64; 18] = core::array::from_fn(|i| (i + 1) as f64);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                3,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 18, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            // Read into array larger than FITS image: ncols=4 > naxis1=3.
            let mut result = [0.0f64; 32]; // 4x4x2 = 32, larger than FITS image.
            let mut anynull = -1;
            fits_read_3d_dbl(
                f.as_deref_mut().unwrap(),
                1,
                0.0,
                4,
                4,
                3,
                3,
                2,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Data should be at start of each row.
            assert_eq!(result[0], 1.0); // First element.
            assert_eq!(result[1], 2.0);
            assert_eq!(result[2], 3.0);
            // result[3] is padding (beyond naxis1=3).
            assert_eq!(result[4], 4.0); // Second row of first plane.
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_table_bad_range() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
            let naxes: [c_long; 1] = [5];
            let blc: [c_long; 2] = [3, 1]; // Start at elem 3.
            let trc: [c_long; 2] = [1, 1]; // End at elem 1 - reverse!
            let inc: [c_long; 2] = [1, 1];

            let mut f = make_table(&name, BINARY_TBL, "VECTOR", "5D", 1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            // On tables, reverse range should fail with BAD_PIX_NUM.
            let mut result = [0.0f64; 1];
            let mut anynull = -1;
            fits_read_subset_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                &naxes,
                &blc,
                &trc,
                &inc,
                0.0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_PIX_NUM);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_table_bad_range_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
            let naxes: [c_long; 1] = [5];
            let blc: [c_long; 2] = [3, 1]; // Start at elem 3.
            let trc: [c_long; 2] = [1, 1]; // End at elem 1 - reverse!
            let inc: [c_long; 2] = [1, 1];

            let mut f = make_table(&name, BINARY_TBL, "VECTOR", "5D", 1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            // On tables, reverse range should fail with BAD_PIX_NUM.
            let mut result = [0.0f64; 1];
            let mut nularray = [0 as c_char; 1];
            let mut anynull = -1;
            fits_read_subsetnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                1,
                &naxes,
                &blc,
                &trc,
                &inc,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_PIX_NUM);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_subsection_flags_naxis_boundary() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [1];
            let blc: [c_long; 1] = [1];
            let trc: [c_long; 1] = [1];
            let inc: [c_long; 1] = [1];

            // Test with naxis out of range for ffgsfd.
            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0.0f64; 1];
            let mut nularray = [0 as c_char; 1];
            let mut anynull = -1;
            // naxis = 0 should fail.
            fits_read_subsetnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                0,
                &naxes,
                &blc,
                &trc,
                &inc,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_DIMEN);
            status = 0;
            // naxis = 10 should fail.
            fits_read_subsetnull_dbl(
                f.as_deref_mut().unwrap(),
                1,
                10,
                &naxes,
                &blc,
                &trc,
                &inc,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_DIMEN);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }
}
