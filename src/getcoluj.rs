//! Routines that read data elements from a FITS image or table, with
//! unsigned long datatype.
//!
//! The [`TULONG`] arm of the typed column I/O family. [`crate::getcol`]
//! dispatches here when a caller asks for this datatype at run time; the
//! write-side counterpart is [`crate::putcoluj`].
//!
//! Ported from CFITSIO's `getcoluj.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center.
#![warn(missing_docs)]

use crate::helpers::nularray::nul_tail;
use core::slice;
use core::{cmp, mem};

use crate::c_types::{c_char, c_int, c_long, c_short, c_uint, c_ulong};

use bytemuck::{cast_slice, cast_slice_mut};

use crate::fitscore::{
    ffasfm_safe, ffgcprll, ffghdt_safe, ffpmsg_slice, ffpmsg_str, fits_is_compressed_image_safe,
};
use crate::fitsio2::*;
use crate::imcompress::{fits_read_compressed_img, fits_read_compressed_pixels};
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
pub unsafe extern "C" fn ffgpvuj(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: c_ulong,
    array: *mut c_ulong,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, nelem as usize);

        ffgpvuj_safe(fptr, group, firstelem, nelem, nulval, array, anynul, status)
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
pub fn ffgpvuj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: c_ulong,
    array: &mut [c_ulong],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut row: c_long = 0;
    let nullcheck = NullCheckType::SetPixel;
    let mut nullvalue: c_ulong = 0;

    let mut dummy_nularray = vec![0; nelem as usize];

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */
        nullvalue = nulval; /* set local variable */

        fits_read_compressed_pixels(
            fptr,
            TULONG,
            firstelem,
            nelem,
            nullcheck,
            &Some(NullValue::ULong(nullvalue)),
            cast_slice_mut(array),
            None,
            anynul,
            status,
        );

        return *status;
    }

    row = cmp::max(1, group);

    ffgcluj(
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
pub unsafe extern "C" fn ffgpfuj(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut c_ulong,
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

        ffgpfuj_safe(
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
pub fn ffgpfuj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [c_ulong],
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut row = 0;
    let nullcheck = NullCheckType::SetNullArray;

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        fits_read_compressed_pixels(
            fptr,
            TULONG,
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

    row = cmp::max(1, group);

    ffgcluj(
        fptr,
        2,
        row as LONGLONG,
        firstelem,
        nelem,
        1,
        NullCheckType::SetNullArray,
        0,
        array,
        nularray,
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
pub unsafe extern "C" fn ffg2duj(
    fptr: *mut fitsfile,
    group: c_long,
    nulval: c_ulong,
    ncols: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    array: *mut c_ulong,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        /* call the 3D reading routine, with the 3rd dimension = 1 */

        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, (ncols * naxis2) as usize);

        ffg2duj_safe(
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
pub fn ffg2duj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    nulval: c_ulong,
    ncols: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    array: &mut [c_ulong],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    /* call the 3D reading routine, with the 3rd dimension = 1 */
    ffg3duj_safe(
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
pub unsafe extern "C" fn ffg3duj(
    fptr: *mut fitsfile,
    group: c_long,
    nulval: c_ulong,
    ncols: LONGLONG,
    nrows: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    naxis3: LONGLONG,
    array: *mut c_ulong,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, (ncols * nrows) as usize);

        ffg3duj_safe(
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
pub fn ffg3duj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    nulval: c_ulong,
    ncols: LONGLONG,
    nrows: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    naxis3: LONGLONG,
    array: &mut [c_ulong],
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut tablerow = 0;
    let nullcheck = NullCheckType::SetPixel;
    let inc: [c_long; 3] = [1, 1, 1];
    let fpixel: [LONGLONG; 3] = [1, 1, 1];
    let mut nfits = 0;
    let mut narray = 0;
    let mut lpixel: [LONGLONG; 3] = [0; 3];
    let mut nullvalue: c_ulong = 0;

    let mut dummy_nularray = vec![0; (naxis1 * naxis2 * naxis3) as usize];

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        lpixel[0] = ncols as LONGLONG;
        lpixel[1] = nrows as LONGLONG;
        lpixel[2] = naxis3 as LONGLONG;
        nullvalue = nulval; /* set local variable */

        fits_read_compressed_img(
            fptr,
            TULONG,
            cast_slice(&fpixel),
            cast_slice(&lpixel),
            &inc,
            nullcheck,
            &Some(crate::NullValue::ULong(nullvalue)),
            cast_slice_mut(array),
            None,
            anynul.as_deref_mut(),
            status,
        );
        return *status;
    }

    tablerow = cmp::max(1, group);

    if ncols == naxis1 && nrows == naxis2 {
        /* arrays have same size? */

        /* all the image pixels are contiguous, so read all at once */
        ffgcluj(
            fptr,
            2,
            tablerow as LONGLONG,
            1,
            naxis1 * naxis2 * naxis3,
            1,
            NullCheckType::SetPixel,
            nulval,
            cast_slice_mut(array),
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
            if ffgcluj(
                fptr,
                2,
                tablerow as LONGLONG,
                nfits,
                naxis1,
                1,
                NullCheckType::SetPixel,
                nulval,
                cast_slice_mut(&mut array[(narray as usize)..]),
                &mut dummy_nularray,
                anynul.as_deref_mut(),
                status,
            ) > 0
            {
                return *status;
            }

            nfits += naxis1;
            narray += ncols;
        }
        narray += (nrows - naxis2) * ncols;
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
pub unsafe extern "C" fn ffgsvuj(
    fptr: *mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: *const c_long,
    blc: *const c_long,
    trc: *const c_long,
    inc: *const c_long,
    nulval: c_ulong,
    array: *mut c_ulong,
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

        ffgsvuj_safe(
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
pub fn ffgsvuj_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: &[c_long],
    blc: &[c_long],
    trc: &[c_long],
    inc: &[c_long],
    nulval: c_ulong,
    array: &mut [c_ulong],
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
    let nullcheck = NullCheckType::SetPixel;
    let mut nullvalue: c_ulong = 0;

    let naxis: usize = naxis as usize;

    if naxis < 1 || naxis > 9 {
        int_snprintf!(
            &mut msg,
            FLEN_ERRMSG,
            "NAXIS = {} in call to ffgsvuj is out of range",
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
            TULONG,
            &blcll,
            &trcll,
            inc,
            nullcheck,
            &Some(crate::NullValue::ULong(nullvalue)),
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
    }

    for ii in 0..(naxis) {
        if trc[ii] < blc[ii] {
            int_snprintf!(
                &mut msg,
                FLEN_ERRMSG,
                "ffgsvuj: illegal range specified for axis {}",
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
        for i8 in ((str[8])..=(stp[8])).step_by(incr[8] as usize) {
            for i7 in ((str[7])..=(stp[7])).step_by(incr[7] as usize) {
                for i6 in ((str[6])..=(stp[6])).step_by(incr[6] as usize) {
                    for i5 in ((str[5])..=(stp[5])).step_by(incr[5] as usize) {
                        for i4 in ((str[4])..=(stp[4])).step_by(incr[4] as usize) {
                            for i3 in ((str[3])..=(stp[3])).step_by(incr[3] as usize) {
                                for i2 in ((str[2])..=(stp[2])).step_by(incr[2] as usize) {
                                    for i1 in ((str[1])..=(stp[1])).step_by(incr[1] as usize) {
                                        felem = (str[0] as LONGLONG)
                                            + (i1 as LONGLONG - 1) * dsize[1]
                                            + (i2 as LONGLONG - 1) * dsize[2]
                                            + (i3 as LONGLONG - 1) * dsize[3]
                                            + (i4 as LONGLONG - 1) * dsize[4]
                                            + (i5 as LONGLONG - 1) * dsize[5]
                                            + (i6 as LONGLONG - 1) * dsize[6]
                                            + (i7 as LONGLONG - 1) * dsize[7]
                                            + (i8 as LONGLONG - 1) * dsize[8];

                                        if ffgcluj(
                                            fptr,
                                            numcol as c_int,
                                            row as LONGLONG,
                                            felem,
                                            nelem as LONGLONG,
                                            ninc,
                                            nultyp,
                                            nulval,
                                            &mut array[(i0 as usize)..],
                                            &mut [0],
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
                                        i0 += nelem;
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
pub unsafe extern "C" fn ffgsfuj(
    fptr: *mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: *const c_long,
    blc: *const c_long,
    trc: *const c_long,
    inc: *const c_long,
    array: *mut c_ulong,
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
        ffgsfuj_safe(
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
pub fn ffgsfuj_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: &[c_long],
    blc: &[c_long],
    trc: &[c_long],
    inc: &[c_long],
    array: &mut [c_ulong],
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
    let mut dsize: [c_long; 10] = [0; 10];
    let mut blcll: [LONGLONG; 9] = [0; 9];
    let mut trcll: [LONGLONG; 9] = [0; 9];
    let mut felem: c_long = 0;
    let mut nelem: c_long = 0;
    let mut nultyp: NullCheckType = NullCheckType::None;
    let mut ninc: c_long = 0;
    let mut numcol: c_long = 0;
    let nulval: c_ulong = 0;
    let mut hdutype: c_int = 0;
    let mut anyf: c_int = 0;
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let nullcheck = NullCheckType::SetNullArray;

    let naxis = naxis as usize;

    if naxis < 1 || naxis > 9 {
        int_snprintf!(
            &mut msg,
            FLEN_ERRMSG,
            "NAXIS = {} in call to ffgsvuj is out of range",
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
            TULONG,
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

    for ii in 0..(naxis) {
        if trc[ii] < blc[ii] {
            int_snprintf!(
                &mut msg,
                FLEN_ERRMSG,
                "ffgsvuj: illegal range specified for axis {}",
                ii + 1,
            );
            ffpmsg_slice(&msg);
            *status = BAD_PIX_NUM;
            return *status;
        }

        str[ii] = blc[ii];
        stp[ii] = trc[ii];
        incr[ii] = inc[ii];
        dsize[ii + 1] = dsize[ii] * naxes[ii] as c_long;
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
        for i8 in ((str[8])..=(stp[8])).step_by(incr[8] as usize) {
            for i7 in ((str[7])..=(stp[7])).step_by(incr[7] as usize) {
                for i6 in ((str[6])..=(stp[6])).step_by(incr[6] as usize) {
                    for i5 in ((str[5])..=(stp[5])).step_by(incr[5] as usize) {
                        for i4 in ((str[4])..=(stp[4])).step_by(incr[4] as usize) {
                            for i3 in ((str[3])..=(stp[3])).step_by(incr[3] as usize) {
                                for i2 in ((str[2])..=(stp[2])).step_by(incr[2] as usize) {
                                    for i1 in ((str[1])..=(stp[1])).step_by(incr[1] as usize) {
                                        felem = (str[0] as c_long)
                                            + (i1 as c_long - 1) * dsize[1]
                                            + (i2 as c_long - 1) * dsize[2]
                                            + (i3 as c_long - 1) * dsize[3]
                                            + (i4 as c_long - 1) * dsize[4]
                                            + (i5 as c_long - 1) * dsize[5]
                                            + (i6 as c_long - 1) * dsize[6]
                                            + (i7 as c_long - 1) * dsize[7]
                                            + (i8 as c_long - 1) * dsize[8];

                                        if ffgcluj(
                                            fptr,
                                            numcol as c_int,
                                            row as LONGLONG,
                                            felem as LONGLONG,
                                            nelem as LONGLONG,
                                            ninc,
                                            nultyp,
                                            nulval,
                                            &mut array[(i0 as usize)..],
                                            &mut flagval[(i0 as usize)..],
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
                                        i0 += nelem;
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

/// Read an array of group parameters from the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
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
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffggpuj(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: c_long,
    nelem: c_long,
    array: *mut c_ulong,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let array = slice::from_raw_parts_mut(array, nelem as usize);

        ffggpuj_safe(fptr, group, firstelem, nelem, array, status)
    }
}

/// Read an array of group parameters from the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
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
/// * `status`    — (IO) error status
pub fn ffggpuj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: c_long,
    nelem: c_long,
    array: &mut [c_ulong],
    status: &mut c_int,
) -> c_int {
    let mut anynul = 0;

    let mut dummy_nularray = vec![0; nelem as usize];

    let row = cmp::max(1, group);

    ffgcluj(
        fptr,
        1,
        row as LONGLONG,
        firstelem as LONGLONG,
        nelem as LONGLONG,
        1,
        NullCheckType::SetPixel,
        0,
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
pub unsafe extern "C" fn ffgcvuj(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: c_ulong,
    array: *mut c_ulong,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let array = slice::from_raw_parts_mut(array, nelem as usize);

        let anynul = anynul.as_mut();

        ffgcvuj_safe(
            fptr, colnum, firstrow, firstelem, nelem, nulval, array, anynul, status,
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
pub fn ffgcvuj_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: c_ulong,
    array: &mut [c_ulong],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut dummy_nularray = vec![0; nelem as usize];

    ffgcluj(
        fptr,
        colnum,
        firstrow as LONGLONG,
        firstelem as LONGLONG,
        nelem as LONGLONG,
        1,
        NullCheckType::SetPixel,
        nulval,
        array,
        &mut dummy_nularray,
        anynul,
        status,
    )
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
pub unsafe extern "C" fn ffgcfuj(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut c_ulong,
    nularray: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let array = slice::from_raw_parts_mut(array, nelem as usize);
        let nularray = slice::from_raw_parts_mut(nularray, nelem as usize);

        let anynul = anynul.as_mut();

        ffgcfuj_safe(
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
pub fn ffgcfuj_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [c_ulong],
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let dummy = 0;

    ffgcluj(
        fptr,
        colnum,
        firstrow as LONGLONG,
        firstelem as LONGLONG,
        nelem as LONGLONG,
        1,
        NullCheckType::SetNullArray,
        dummy,
        array,
        nularray,
        anynul,
        status,
    )
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
pub(crate) fn ffgcluj(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    elemincre: c_long,
    nultyp: NullCheckType,
    /*     1: set undefined pixels = nulval        */
    /*     2: set nularray=1 for undefined pixels  */
    nulval: c_ulong,
    array: &mut [c_ulong],
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
    let mut xwidth: c_long = 0;
    let mut ntodo: c_long = 0;
    let mut nulcheck = NullCheckType::None;
    let readcheck: c_int = 0;
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
    let mut cbuff: [f64; DBUFFSIZE as usize / mem::size_of::<f64>()] =
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

    if tcode == TSTRING {
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
    nulcheck = nultyp; /* by default check for null values in the FITS file */

    if nultyp == NullCheckType::SetPixel && nulval == 0 {
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

    if (tcode == TLONG) && (LONGSIZE == 32) {
        /* Special Case:                        */
        /* no type convertion required, so read */
        /* data directly into output buffer.    */

        if nelem < LONGLONG::from(INT32_MAX) / 4 {
            maxelem = nelem;
        } else {
            maxelem = LONGLONG::from(INT32_MAX / 4);
        }
    }
    /* Unlike the C, the TLONG case is read into the scratch buffer and
    converted from there rather than in place in the output array, so
    each chunk must fit in the buffer. */
    maxelem = LONGLONG::from(maxelem2);

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

    let buffer = &mut cbuff;

    while remain > 0 {
        /* limit the number of pixels to read at one time to the number that
           will fit in the buffer or to the number of pixels that remain in
           the current vector, which ever is smaller.
        */
        ntodo = cmp::min(remain, maxelem) as c_long;
        ntodo = cmp::min(
            ntodo as LONGLONG,
            (repeat - elemnum - 1) / (elemincre as LONGLONG) + 1,
        ) as c_long;

        readptr =
            startpos + (rownum as LONGLONG * rowlen) + (elemnum * (incre / elemincre) as LONGLONG);

        match tcode {
            TLONG => {
                if LONGSIZE == 32 {
                    ffgi4b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                    fffi4u4(
                        cast_slice_mut(buffer),
                        ntodo,
                        scale,
                        zero,
                        nulcheck,
                        tnull as c_int,
                        nulval,
                        cast_slice_mut(nul_tail(nularray, next)),
                        anynul.as_deref_mut(),
                        cast_slice_mut(&mut array[next..]),
                        status,
                    );
                } else {
                    /* case where sizeof(long) = 8 */
                    ffgi4b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                    fffi4u4(
                        cast_slice_mut(buffer),
                        ntodo,
                        scale,
                        zero,
                        nulcheck,
                        tnull as c_int,
                        nulval,
                        cast_slice_mut(nul_tail(nularray, next)),
                        anynul.as_deref_mut(),
                        cast_slice_mut(&mut array[next..]),
                        status,
                    );
                }
            }
            TLONGLONG => {
                ffgi8b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffi8u4(
                    cast_slice_mut(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[next..]),
                    status,
                );
            }
            TBYTE => {
                ffgi1b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffi1u4(
                    cast_slice_mut(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull as u8,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[next..]),
                    status,
                );
            }
            TSHORT => {
                ffgi2b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffi2u4(
                    cast_slice_mut(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull as c_short,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[next..]),
                    status,
                );
            }
            TFLOAT => {
                ffgr4b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffr4u4(
                    cast_slice_mut(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[next..]),
                    status,
                );
            }
            TDOUBLE => {
                ffgr8b(fptr, readptr, ntodo, incre, buffer, status);

                fffr8u4(
                    cast_slice_mut(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[next..]),
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
                        cast_slice_mut(buffer),
                        status,
                    );
                } else {
                    ffgbytoff(
                        fptr,
                        twidth,
                        ntodo,
                        incre - twidth,
                        cast_slice_mut(buffer),
                        status,
                    );
                }

                fffstru4(
                    cast_slice_mut(buffer),
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
                    cast_slice_mut(&mut array[next..]),
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
                    "Error reading elements {:.0} thru {:.0} from column {} (ffgcluj).",
                    dtemp + 1.0,
                    dtemp + ntodo as f64,
                    colnum,
                );
            } else {
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Error reading elements {:.0} thru {:.0} from image (ffgcluj).",
                    dtemp + 1.0,
                    dtemp + ntodo as f64,
                );
            }
            ffpmsg_slice(&message);
            return *status;
        }

        /*  increment the counters for the next loop  */
        remain -= ntodo as LONGLONG;
        if remain > 0 {
            next += ntodo as usize;
            elemnum += (ntodo * elemincre) as LONGLONG;

            if elemnum >= repeat {
                /* completed a row; start on later row */
                rowincre = elemnum / repeat;
                rownum += rowincre;
                elemnum -= rowincre * repeat;
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi1u4(
    input: &[u8],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: u8,
    nullval: c_ulong,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [c_ulong],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                output[ii] = c_ulong::from(input[ii]); /* copy input to output */
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = f64::from(input[ii]) * scale + zero;

                if dvalue < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if dvalue > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = dvalue as c_ulong;
                }
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
                    output[ii] = c_ulong::from(input[ii]);
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
                    dvalue = f64::from(input[ii]) * scale + zero;

                    if dvalue < DULONG_MIN {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MIN;
                    } else if dvalue > DULONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MAX;
                    } else {
                        output[ii] = dvalue as c_ulong;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi2u4(
    input: &[c_short],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: c_short,
    nullval: c_ulong,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [c_ulong],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as c_ulong; /* copy input to output */
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = f64::from(input[ii]) * scale + zero;

                if dvalue < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if dvalue > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = dvalue as c_ulong;
                }
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
                } else if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as c_ulong;
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
                    dvalue = f64::from(input[ii]) * scale + zero;

                    if dvalue < DULONG_MIN {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MIN;
                    } else if dvalue > DULONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MAX;
                    } else {
                        output[ii] = dvalue as c_ulong;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi4u4(
    input: &[INT32BIT],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: INT32BIT,
    nullval: c_ulong,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [c_ulong],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 2147483648. {
            /* Instead of adding 2147483648, it is more efficient */
            /* to just flip the sign bit with the XOR operator */

            for ii in 0..(ntodo as usize) {
                output[ii] = c_ulong::from((input[ii] as c_uint) ^ 0x80000000);
            }
        } else if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as c_ulong; /* copy input to output */
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = f64::from(input[ii]) * scale + zero;

                if dvalue < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if dvalue > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = dvalue as c_ulong;
                }
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        if scale == 1.0 && zero == 2147483648. {
            /* Instead of adding 2147483648, it is more efficient */
            /* to just flip the sign bit with the XOR operator */

            for ii in 0..(ntodo as usize) {
                if input[ii] == tnull {
                    *anynull = 1;
                    if nullcheck == NullCheckType::SetPixel {
                        output[ii] = nullval;
                    } else {
                        nullarray[ii] = 1;
                    }
                } else {
                    output[ii] = c_ulong::from((input[ii] as c_uint) ^ 0x80000000);
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
                } else if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as c_ulong;
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
                    dvalue = f64::from(input[ii]) * scale + zero;

                    if dvalue < DULONG_MIN {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MIN;
                    } else if dvalue > DULONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MAX;
                    } else {
                        output[ii] = dvalue as c_ulong;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi8u4(
    input: &[LONGLONG],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: LONGLONG,
    nullval: c_ulong,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [c_ulong],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;
    let mut ulltemp: ULONGLONG = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 9223372036854775808. {
            /* The column we read contains unsigned long long values. */
            /* Instead of adding 9223372036854775808, it is more efficient */
            /* and more precise to just flip the sign bit with the XOR operator */

            for ii in 0..(ntodo as usize) {
                ulltemp = ((input[ii] as ULONGLONG) ^ 0x8000000000000000) as ULONGLONG;

                if ulltemp > c_ulong::MAX as ULONGLONG {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = ulltemp as c_ulong;
                }
            }
        } else if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else if (input[ii] as ULONGLONG) > c_ulong::MAX as ULONGLONG {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = input[ii] as c_ulong;
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = input[ii] as f64 * scale + zero;

                if dvalue < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if dvalue > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = dvalue as c_ulong;
                }
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
                    ulltemp = ((input[ii] as ULONGLONG) ^ 0x8000000000000000) as ULONGLONG;

                    if ulltemp > c_ulong::MAX as ULONGLONG {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MAX;
                    } else {
                        output[ii] = ulltemp as c_ulong;
                    }
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
                } else if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else if (input[ii] as ULONGLONG) > c_ulong::MAX as ULONGLONG {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = input[ii] as c_ulong;
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
                    dvalue = input[ii] as f64 * scale + zero;

                    if dvalue < DULONG_MIN {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MIN;
                    } else if dvalue > DULONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MAX;
                    } else {
                        output[ii] = dvalue as c_ulong;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffr4u4(
    input: &[f32],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    nullval: c_ulong,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [c_ulong],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;
    let mut sptr = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if f64::from(input[ii]) < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if f64::from(input[ii]) > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = input[ii] as c_ulong;
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = f64::from(input[ii]) * scale + zero;

                if dvalue < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if dvalue > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = dvalue as c_ulong;
                }
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        //sptr = (short *) input;

        if BYTESWAPPED {
            sptr += 1; /* point to MSBs */
        }

        let shortBuff: &[c_short] = cast_slice(input);

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                let iret = fnan(shortBuff[sptr]);
                if iret != 0 {
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
                        output[ii] = 0;
                    }
                } else if f64::from(input[ii]) < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if f64::from(input[ii]) > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = input[ii] as c_ulong;
                }
                sptr += 2;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                let iret = fnan(shortBuff[sptr]);
                if iret != 0 {
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
                        if zero < DULONG_MIN {
                            *status = OVERFLOW_ERR;
                            output[ii] = c_ulong::MIN;
                        } else if zero > DULONG_MAX {
                            *status = OVERFLOW_ERR;
                            output[ii] = c_ulong::MAX;
                        } else {
                            output[ii] = zero as c_ulong;
                        }
                    }
                } else {
                    dvalue = f64::from(input[ii]) * scale + zero;

                    if dvalue < DULONG_MIN {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MIN;
                    } else if dvalue > DULONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MAX;
                    } else {
                        output[ii] = dvalue as c_ulong;
                    }
                }
                sptr += 2;
            }
        }
    }
    *status
}

/// # Parameters
///
/// * `input`     — (I) array of values to be converted
/// * `ntodo`     — (I) number of elements in the array
/// * `scale`     — (I) FITS TSCALn or BSCALE value
/// * `zero`      — (I) FITS TZEROn or BZERO value
/// * `nullcheck` — (I) null checking code; 0 = don't check
/// * `nullval`   — (I) set null pixels, if nullcheck = 1
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffr8u4(
    input: &[f64],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    nullval: c_ulong,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [c_ulong],
    status: &mut c_int,
) -> c_int
/*
Copy input to output following reading of the input from a FITS file.
Check for null values and do datatype conversion and scaling if required.
The nullcheck code value determines how any null values in the input array
are treated.  A null value is an input pixel that is equal to NaN.  If 
nullcheck = 0, then no checking for nulls is performed and any null values
will be transformed just like any other pixel.  If nullcheck = 1, then the
output pixel will be set = nullval if the corresponding input pixel is null.
If nullcheck = 2, then if the pixel is null then the corresponding value of
nullarray will be set to 1; the value of nullarray for non-null pixels 
will = 0.  The anynull parameter will be set = 1 if any of the returned
pixels are null, otherwise anynull will be returned with a value = 0;
*/
{
    let mut dvalue: f64 = 0.0;
    let mut sptr = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if input[ii] > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = input[ii] as c_ulong;
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = input[ii] * scale + zero;

                if dvalue < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if dvalue > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = dvalue as c_ulong;
                }
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        //sptr = (short *) input;

        if BYTESWAPPED {
            sptr += 3; /* point to MSBs */
        }

        let shortBuff: &[c_short] = cast_slice(input);

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                let iret = dnan(shortBuff[sptr]);
                if iret != 0 {
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
                        output[ii] = 0;
                    }
                } else if input[ii] < DULONG_MIN {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MIN;
                } else if input[ii] > DULONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = c_ulong::MAX;
                } else {
                    output[ii] = input[ii] as c_ulong;
                }
                sptr += 4;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                let iret = dnan(shortBuff[sptr]);
                if iret != 0 {
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

                        if zero < DULONG_MIN {
                            *status = OVERFLOW_ERR;
                            output[ii] = c_ulong::MIN;
                        } else if zero > DULONG_MAX {
                            *status = OVERFLOW_ERR;
                            output[ii] = c_ulong::MAX;
                        } else {
                            output[ii] = zero as c_ulong;
                        }
                    }
                } else {
                    dvalue = input[ii] * scale + zero;

                    if dvalue < DULONG_MIN {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MIN;
                    } else if dvalue > DULONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = c_ulong::MAX;
                    } else {
                        output[ii] = dvalue as c_ulong;
                    }
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
/// * `nullarray`  — (I) bad pixel array, if nullcheck = 2
/// * `anynull`    — (O) set to 1 if any pixels are null
/// * `output`     — (O) array of converted pixels
/// * `status`     — (IO) error status
pub(crate) fn fffstru4(
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
    nullval: c_ulong,
    nullarray: &mut [c_char],
    mut anynull: Option<&mut c_int>,
    output: &mut [c_ulong],
    status: &mut c_int,
) -> c_int {
    let mut nullen = 0;

    let mut dvalue: f64 = 0.0;

    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut tempstore: c_char = 0;
    let chrzero: c_char = bb(b'0'); // 49
    let mut val: f64 = 0.0;
    let mut power: f64 = 0.0;
    let mut exponent: c_int = 0;
    let mut sign: c_int = 0;
    let mut esign: c_int = 0;
    let mut decpt: c_int = 0;

    nullen = strlen_safe(snull);
    let mut cptr = 0; /* pointer to start of input string */
    let mut tpos = 0;
    for ii in 0..(ntodo as usize) {
        let originalCptr = cptr;
        /* temporarily insert a null terminator at end of the string */
        tpos = cptr + twidth as usize;
        tempstore = input[tpos];
        input[tpos] = 0;

        /* check if null value is defined, and if the    */
        /* column string is identical to the null string */
        if snull[0] != ASCII_NULL_UNDEFINED && strncmp_safe(snull, &input[cptr..], nullen) == 0 {
            if nullcheck != NullCheckType::None {
                if let Some(anynull) = anynull.as_deref_mut() {
                    *anynull = 1;
                }
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            }
            cptr += twidth as usize;
        } else {
            /* value is not the null value, so decode it */
            /* remove any embedded blank characters from the string */

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
                val = val * 10.0 + f64::from(input[cptr]) - f64::from(chrzero); /* accumulate the value */
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
                    val = val * 10.0 + f64::from(input[cptr]) - f64::from(chrzero); /* accumulate the value */
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
                    exponent = exponent * 10 + c_int::from(input[cptr]) - c_int::from(chrzero); /* accumulate exp */
                    cptr += 1;

                    while input[cptr] == bb(b' ') {
                        /* skip embedded blanks */
                        cptr += 1;
                    }
                }
            }

            if input[cptr] != 0 {
                /* should end up at the null terminator */
                let cstring = &input[originalCptr..];
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
                    slice_to_str!(cstring),
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
            dvalue = (f64::from(sign) * val / power) * 10.0_f64.powi(esign * exponent);

            dvalue = dvalue * scale + zero; /* apply the scaling */

            if dvalue < DULONG_MIN {
                *status = OVERFLOW_ERR;
                output[ii] = c_ulong::MIN;
            } else if dvalue > DULONG_MAX {
                *status = OVERFLOW_ERR;
                output[ii] = c_ulong::MAX;
            } else {
                output[ii] = dvalue as c_ulong;
            }
        }
        /* restore the char that was overwritten by the null */
        input[tpos] = tempstore;
    }
    *status
}

/* ======================================================================== */
/*      the following routines support the 'long long' data type            */
/* ======================================================================== */

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
pub unsafe extern "C" fn ffgpvujj(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: ULONGLONG,
    array: *mut ULONGLONG,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, nelem as usize);

        ffgpvujj_safe(fptr, group, firstelem, nelem, nulval, array, anynul, status)
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
pub fn ffgpvujj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: ULONGLONG,
    array: &mut [ULONGLONG],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut row = 0;
    let nullcheck = NullCheckType::SetPixel;
    let mut nullvalue: ULONGLONG = 0;

    let mut dummy_nularray = vec![0; nelem as usize];

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */
        nullvalue = nulval; /* set local variable */

        fits_read_compressed_pixels(
            fptr,
            TULONGLONG,
            firstelem,
            nelem,
            nullcheck,
            &Some(NullValue::ULONGLONG(nullvalue)),
            cast_slice_mut(array),
            None,
            anynul,
            status,
        );

        return *status;
    }

    row = cmp::max(1, group);

    ffgclujj(
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
pub unsafe extern "C" fn ffgpfujj(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut ULONGLONG,
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

        ffgpfujj_safe(
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
pub fn ffgpfujj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [ULONGLONG],
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut row = 0;
    let nullcheck = NullCheckType::SetNullArray;
    let dummy: ULONGLONG = 0;

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        fits_read_compressed_pixels(
            fptr,
            TULONGLONG,
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

    row = cmp::max(1, group);

    ffgclujj(
        fptr,
        2,
        row as LONGLONG,
        firstelem,
        nelem,
        1,
        NullCheckType::SetNullArray,
        dummy,
        array,
        nularray,
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
pub unsafe extern "C" fn ffg2dujj(
    fptr: *mut fitsfile,
    group: c_long,
    nulval: ULONGLONG,
    ncols: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    array: *mut ULONGLONG,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, (naxis1 * naxis2) as usize);

        ffg2dujj_safe(
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
pub fn ffg2dujj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    nulval: ULONGLONG,
    ncols: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    array: &mut [ULONGLONG],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    /* call the 3D reading routine, with the 3rd dimension = 1 */
    ffg3dujj_safe(
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
pub unsafe extern "C" fn ffg3dujj(
    fptr: *mut fitsfile,
    group: c_long,
    nulval: ULONGLONG,
    ncols: LONGLONG,
    nrows: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    naxis3: LONGLONG,
    array: *mut ULONGLONG,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, (ncols * nrows) as usize);

        ffg3dujj_safe(
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
pub fn ffg3dujj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    nulval: ULONGLONG,
    ncols: LONGLONG,
    nrows: LONGLONG,
    naxis1: LONGLONG,
    naxis2: LONGLONG,
    naxis3: LONGLONG,
    array: &mut [ULONGLONG],
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut tablerow: c_long = 0;
    let nullcheck = NullCheckType::SetPixel;
    let inc: [c_long; 3] = [1, 1, 1];

    let fpixel: [LONGLONG; 3] = [1, 1, 1];
    let mut nfits: LONGLONG = 0;
    let mut narray: LONGLONG = 0;
    let mut lpixel: [LONGLONG; 3] = [0; 3];
    let mut nullvalue: ULONGLONG = 0;

    let mut dummy_nularray = vec![0; (naxis1 * naxis2 * naxis3) as usize];

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        lpixel[0] = ncols as LONGLONG;
        lpixel[1] = nrows as LONGLONG;
        lpixel[2] = naxis3 as LONGLONG;
        nullvalue = nulval; /* set local variable */

        fits_read_compressed_img(
            fptr,
            TULONGLONG,
            cast_slice(&fpixel),
            cast_slice(&lpixel),
            &inc,
            nullcheck,
            &Some(NullValue::ULONGLONG(nullvalue)),
            cast_slice_mut(array),
            None,
            anynul,
            status,
        );

        return *status;
    }

    tablerow = cmp::max(1, group);

    if ncols == naxis1 && nrows == naxis2 {
        /* arrays have same size? */
        /* all the image pixels are contiguous, so read all at once */
        ffgclujj(
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
            if ffgclujj(
                fptr,
                2,
                tablerow as LONGLONG,
                nfits,
                naxis1,
                1,
                NullCheckType::SetPixel,
                nulval,
                &mut array[(narray as usize)..],
                &mut dummy_nularray,
                anynul.as_deref_mut(),
                status,
            ) > 0
            {
                return *status;
            }

            nfits += naxis1;
            narray += ncols;
        }
        narray += (nrows - naxis2) * ncols;
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
pub unsafe extern "C" fn ffgsvujj(
    fptr: *mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: *const c_long,
    blc: *const c_long,
    trc: *const c_long,
    inc: *const c_long,
    nulval: ULONGLONG,
    array: *mut ULONGLONG,
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

        ffgsvujj_safe(
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
pub fn ffgsvujj_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: &[c_long],
    blc: &[c_long],
    trc: &[c_long],
    inc: &[c_long],
    nulval: ULONGLONG,
    array: &mut [ULONGLONG],
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
    let ldummy: c_char = 0;
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let nullcheck = NullCheckType::SetPixel;
    let mut nullvalue: ULONGLONG = 0;

    let naxis: usize = naxis as usize;

    if naxis < 1 || naxis > 9 {
        int_snprintf!(
            &mut msg,
            FLEN_ERRMSG,
            "NAXIS = {} in call to ffgsvuj is out of range",
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
            TULONGLONG,
            &blcll,
            &trcll,
            inc,
            nullcheck,
            &Some(NullValue::ULONGLONG(nullvalue)),
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
                    "ffgsvuj: illegal range specified for axis {}",
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

                                        if ffgclujj(
                                            fptr,
                                            numcol as c_int,
                                            row as LONGLONG,
                                            felem,
                                            nelem as LONGLONG,
                                            ninc,
                                            nultyp,
                                            nulval,
                                            cast_slice_mut(&mut array[(i0 as usize)..]),
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
                                        i0 += nelem;
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
pub unsafe extern "C" fn ffgsfujj(
    fptr: *mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: *const c_long,
    blc: *const c_long,
    trc: *const c_long,
    inc: *const c_long,
    array: *mut ULONGLONG,
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

        ffgsfujj_safe(
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
pub fn ffgsfujj_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    naxis: c_int,
    naxes: &[c_long],
    blc: &[c_long],
    trc: &[c_long],
    inc: &[c_long],
    array: &mut [ULONGLONG],
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
    let mut dsize: [c_long; 10] = [0; 10];
    let mut blcll: [LONGLONG; 9] = [0; 9];
    let mut trcll: [LONGLONG; 9] = [0; 9];
    let mut felem: c_long = 0;
    let mut nelem: c_long = 0;
    let mut nultyp = NullCheckType::None;
    let mut ninc: c_long = 0;
    let mut numcol: c_long = 0;
    let nulval: ULONGLONG = 0;
    let mut hdutype: c_int = 0;
    let mut anyf: c_int = 0;
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let nullcheck = NullCheckType::SetNullArray;

    let naxis: usize = naxis as usize;

    if naxis < 1 || naxis > 9 {
        int_snprintf!(
            &mut msg,
            FLEN_ERRMSG,
            "NAXIS = {} in call to ffgsvujj is out of range",
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
            TULONGLONG,
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

    for ii in 0..(naxis) {
        if trc[ii] < blc[ii] {
            int_snprintf!(
                &mut msg,
                FLEN_ERRMSG,
                "ffgsvujj: illegal range specified for axis {}",
                ii + 1,
            );
            ffpmsg_slice(&msg);
            *status = BAD_PIX_NUM;
            return *status;
        }

        str[ii] = blc[ii];
        stp[ii] = trc[ii];
        incr[ii] = inc[ii];
        dsize[ii + 1] = dsize[ii] * naxes[ii] as c_long;
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
        for i8 in ((str[8])..=(stp[8])).step_by(incr[8] as usize) {
            for i7 in ((str[7])..=(stp[7])).step_by(incr[7] as usize) {
                for i6 in ((str[6])..=(stp[6])).step_by(incr[6] as usize) {
                    for i5 in ((str[5])..=(stp[5])).step_by(incr[5] as usize) {
                        for i4 in ((str[4])..=(stp[4])).step_by(incr[4] as usize) {
                            for i3 in ((str[3])..=(stp[3])).step_by(incr[3] as usize) {
                                for i2 in ((str[2])..=(stp[2])).step_by(incr[2] as usize) {
                                    for i1 in ((str[1])..=(stp[1])).step_by(incr[1] as usize) {
                                        felem = (str[0] as c_long)
                                            + (i1 as c_long - 1) * dsize[1]
                                            + (i2 as c_long - 1) * dsize[2]
                                            + (i3 as c_long - 1) * dsize[3]
                                            + (i4 as c_long - 1) * dsize[4]
                                            + (i5 as c_long - 1) * dsize[5]
                                            + (i6 as c_long - 1) * dsize[6]
                                            + (i7 as c_long - 1) * dsize[7]
                                            + (i8 as c_long - 1) * dsize[8];

                                        if ffgclujj(
                                            fptr,
                                            numcol as c_int,
                                            row as LONGLONG,
                                            felem as LONGLONG,
                                            nelem as LONGLONG,
                                            ninc,
                                            nultyp,
                                            nulval,
                                            &mut array[(i0 as usize)..],
                                            &mut flagval[(i0 as usize)..],
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
                                        i0 += nelem;
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

/// Read an array of group parameters from the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
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
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffggpujj(
    fptr: *mut fitsfile,
    group: c_long,
    firstelem: c_long,
    nelem: c_long,
    array: *mut ULONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let array = slice::from_raw_parts_mut(array, nelem as usize);

        ffggpujj_safe(fptr, group, firstelem, nelem, array, status)
    }
}

/// Read an array of group parameters from the primary array. Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
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
/// * `status`    — (IO) error status
pub fn ffggpujj_safe(
    fptr: &mut fitsfile,
    group: c_long,
    firstelem: c_long,
    nelem: c_long,
    array: &mut [ULONGLONG],
    status: &mut c_int,
) -> c_int {
    let mut idummy = 0;
    let dummy: ULONGLONG = 0;

    let mut dummy_nularray = vec![0; nelem as usize];

    let row = cmp::max(1, group);

    ffgclujj(
        fptr,
        1,
        row as LONGLONG,
        firstelem as LONGLONG,
        nelem as LONGLONG,
        1,
        NullCheckType::SetPixel,
        dummy,
        array,
        &mut dummy_nularray,
        Some(&mut idummy),
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
pub unsafe extern "C" fn ffgcvujj(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: ULONGLONG,
    array: *mut ULONGLONG,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, nelem as usize);

        ffgcvujj_safe(
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
pub fn ffgcvujj_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: ULONGLONG,
    array: &mut [ULONGLONG],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut dummy_nularray = vec![0; nelem as usize];

    ffgclujj(
        fptr,
        colnum,
        firstrow,
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
pub unsafe extern "C" fn ffgcfujj(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut ULONGLONG,
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

        ffgcfujj_safe(
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
pub fn ffgcfujj_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [ULONGLONG],
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let dummy: ULONGLONG = 0;

    ffgclujj(
        fptr,
        colnum,
        firstrow,
        firstelem,
        nelem,
        1,
        NullCheckType::SetNullArray,
        dummy,
        array,
        nularray,
        anynul,
        status,
    );

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
pub(crate) fn ffgclujj(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    elemincre: c_long,
    nultyp: NullCheckType,
    /*     1: set undefined pixels = nulval        */
    /*     2: set nularray=1 for undefined pixels  */
    nulval: ULONGLONG,
    array: &mut [ULONGLONG],
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
    let mut xwidth: c_long = 0;
    let mut ntodo: c_long = 0;
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
    let mut next: LONGLONG = 0;
    let mut rowincre: LONGLONG = 0;
    let mut maxelem: LONGLONG = 0;
    let mut tform: [c_char; 20] = [0; 20];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut snull: [c_char; 20] = [0; 20]; /*  the FITS null value if reading from ASCII table  */
    let mut cbuff: [f64; DBUFFSIZE as usize / mem::size_of::<f64>()] =
        [0.0; DBUFFSIZE as usize / mem::size_of::<f64>()]; /* align cbuff on word boundary */

    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    let buffer: &mut [u8] = cast_slice_mut(&mut cbuff);

    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = 0;
    }

    if nultyp == NullCheckType::SetNullArray {
        nularray.fill(0); /* initialize nullarray */
    }

    /*  Check input and get parameters about the column: */
    if elemincre < 0 {
        readcheck = -1; /* don't do range checking in this case */
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

    if tcode == TSTRING {
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
            power *= 10.;
        }
    }

    /*  Decide whether to check for null values in the input FITS file: */
    nulcheck = nultyp; /* by default check for null values in the FITS file */

    if nultyp == NullCheckType::SetPixel && nulval == 0 {
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

        readptr =
            startpos + (rownum as LONGLONG * rowlen) + (elemnum * (incre / elemincre) as LONGLONG);

        match tcode {
            TLONGLONG => {
                ffgi8b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffi8u8(
                    cast_slice_mut(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull,
                    nulval,
                    if nulcheck == NullCheckType::SetNullArray {
                        nul_tail(nularray, next as usize)
                    } else {
                        nularray
                    }, // WARNING: THIS IS A HACK. Should use Options instead
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[(next as usize)..]),
                    status,
                );
            }
            TLONG => {
                ffgi4b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffi4u8(
                    cast_slice(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull as INT32BIT,
                    nulval,
                    nul_tail(nularray, next as usize),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[(next as usize)..]),
                    status,
                );
            }
            TBYTE => {
                ffgi1b(fptr, readptr, ntodo, incre, buffer, status);

                fffi1u8(
                    cast_slice(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull as u8,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next as usize)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[(next as usize)..]),
                    status,
                );
            }
            TSHORT => {
                ffgi2b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffi2u8(
                    cast_slice(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    tnull as c_short,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next as usize)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[(next as usize)..]),
                    status,
                );
            }
            TFLOAT => {
                ffgr4b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffr4u8(
                    cast_slice(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next as usize)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[(next as usize)..]),
                    status,
                );
            }
            TDOUBLE => {
                ffgr8b(fptr, readptr, ntodo, incre, cast_slice_mut(buffer), status);

                fffr8u8(
                    cast_slice(buffer),
                    ntodo,
                    scale,
                    zero,
                    nulcheck,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next as usize)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[(next as usize)..]),
                    status,
                );
            }
            TSTRING => {
                ffmbyt_safe(fptr, readptr, REPORT_EOF, status);

                if incre == twidth {
                    /* contiguous bytes */
                    ffgbyt(fptr, (ntodo * twidth) as LONGLONG, buffer, status);
                } else {
                    ffgbytoff(fptr, twidth, ntodo, incre - twidth, buffer, status);
                }

                fffstru8(
                    cast_slice_mut(buffer),
                    ntodo,
                    scale,
                    zero,
                    twidth,
                    power,
                    nulcheck,
                    &snull,
                    nulval,
                    cast_slice_mut(nul_tail(nularray, next as usize)),
                    anynul.as_deref_mut(),
                    cast_slice_mut(&mut array[(next as usize)..]),
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
                    "Error reading elements {:.0} thru {:.0} from column {} (ffgclujj).",
                    dtemp + 1.0,
                    dtemp + ntodo as f64,
                    colnum,
                );
            } else {
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Error reading elements {:.0} thru {:.0} from image (ffgclujj).",
                    dtemp + 1.0,
                    dtemp + ntodo as f64,
                );
            }
            ffpmsg_slice(&message);
            return *status;
        }

        /*  increment the counters for the next loop  */
        remain -= ntodo as LONGLONG;
        if remain > 0 {
            next += ntodo as LONGLONG;
            elemnum += (ntodo * elemincre) as LONGLONG;

            if elemnum >= repeat {
                /* completed a row; start on later row */
                rowincre = elemnum / repeat;
                rownum += rowincre;
                elemnum -= rowincre * repeat;
            } else if elemnum < 0 {
                /* completed a row; start on a previous row */
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi1u8(
    input: &[u8],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: u8,
    nullval: ULONGLONG,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [ULONGLONG],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                output[ii] = ULONGLONG::from(input[ii]); /* copy input to output */
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = f64::from(input[ii]) * scale + zero;

                if dvalue < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if dvalue > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = dvalue as ULONGLONG;
                }
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
                    output[ii] = ULONGLONG::from(input[ii]);
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
                    dvalue = f64::from(input[ii]) * scale + zero;

                    if dvalue < 0.0 {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MIN;
                    } else if dvalue > DULONGLONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MAX;
                    } else {
                        output[ii] = dvalue as ULONGLONG;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi2u8(
    input: &[c_short],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: c_short,
    nullval: ULONGLONG,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [ULONGLONG],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as ULONGLONG; /* copy input to output */
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = f64::from(input[ii]) * scale + zero;

                if dvalue < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if dvalue > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = dvalue as ULONGLONG;
                }
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
                    output[ii] = input[ii] as ULONGLONG;
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
                    dvalue = f64::from(input[ii]) * scale + zero;

                    if dvalue < 0.0 {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MIN;
                    } else if dvalue > DULONGLONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MAX;
                    } else {
                        output[ii] = dvalue as ULONGLONG;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi4u8(
    input: &[INT32BIT],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: INT32BIT,
    nullval: ULONGLONG,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [ULONGLONG],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as ULONGLONG; /* copy input to output */
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = f64::from(input[ii]) * scale + zero;

                if dvalue < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if dvalue > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = dvalue as ULONGLONG;
                }
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
                } else if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as ULONGLONG;
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
                    dvalue = f64::from(input[ii]) * scale + zero;

                    if dvalue < 0.0 {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MIN;
                    } else if dvalue > DULONGLONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MAX;
                    } else {
                        output[ii] = dvalue as ULONGLONG;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffi8u8(
    input: &[LONGLONG],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: LONGLONG,
    nullval: ULONGLONG,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [ULONGLONG],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 9223372036854775808. {
            /* The column we read contains unsigned long long values. */
            /* Instead of adding 9223372036854775808, it is more efficient */
            /* and more precise to just flip the sign bit with the XOR operator */

            for ii in 0..(ntodo as usize) {
                output[ii] = ((input[ii] as ULONGLONG) ^ 0x8000000000000000) as ULONGLONG;
            }
        } else if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as ULONGLONG; /* copy input to output */
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = input[ii] as f64 * scale + zero;

                if dvalue < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if dvalue > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = dvalue as ULONGLONG;
                }
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        if scale == 1.0 && zero == 9223372036854775808.0 {
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
                    output[ii] = ((input[ii] as ULONGLONG) ^ 0x8000000000000000) as ULONGLONG;
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
                } else if input[ii] < 0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = 0;
                } else {
                    output[ii] = input[ii] as ULONGLONG;
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
                    dvalue = input[ii] as f64 * scale + zero;

                    if dvalue < 0.0 {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MIN;
                    } else if dvalue > DULONGLONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MAX;
                    } else {
                        output[ii] = dvalue as ULONGLONG;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffr4u8(
    input: &[f32],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    nullval: ULONGLONG,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [ULONGLONG],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;
    let mut sptr = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if f64::from(input[ii]) > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = input[ii] as ULONGLONG;
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = f64::from(input[ii]) * scale + zero;

                if dvalue < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if dvalue > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = dvalue as ULONGLONG;
                }
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        //sptr = (short *) input;

        if BYTESWAPPED {
            sptr += 1; /* point to MSBs */
        }

        let shortBuff: &[c_short] = cast_slice(input);

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                let iret = fnan(shortBuff[sptr]);
                if iret != 0 {
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
                        output[ii] = 0;
                    }
                } else if (input[ii]) < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if f64::from(input[ii]) > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = input[ii] as ULONGLONG;
                }
                sptr += 2;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                let iret = fnan(shortBuff[sptr]);
                if iret != 0 {
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

                        if zero < 0.0 {
                            *status = OVERFLOW_ERR;
                            output[ii] = ULONGLONG::MIN;
                        } else if zero > DULONGLONG_MAX {
                            *status = OVERFLOW_ERR;
                            output[ii] = ULONGLONG::MAX;
                        } else {
                            output[ii] = zero as ULONGLONG;
                        }
                    }
                } else {
                    dvalue = f64::from(input[ii]) * scale + zero;

                    if dvalue < 0.0 {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MIN;
                    } else if dvalue > DULONGLONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MAX;
                    } else {
                        output[ii] = dvalue as ULONGLONG;
                    }
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
/// * `nullarray` — (I) bad pixel array, if nullcheck = 2
/// * `anynull`   — (O) set to 1 if any pixels are null
/// * `output`    — (O) array of converted pixels
/// * `status`    — (IO) error status
pub(crate) fn fffr8u8(
    input: &[f64],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    nullval: ULONGLONG,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [ULONGLONG],
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;
    let mut sptr = 0;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                if input[ii] < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if input[ii] > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = input[ii] as ULONGLONG;
                }
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                dvalue = input[ii] * scale + zero;

                if dvalue < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if dvalue > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = dvalue as ULONGLONG;
                }
            }
        }
    } else {
        /* must check for null values */
        let anynull = anynull.unwrap();

        //sptr = (short *) input;

        if BYTESWAPPED {
            sptr += 3; /* point to MSBs */
        }

        let shortBuff: &[c_short] = cast_slice(input);

        if scale == 1.0 && zero == 0.0 {
            /* no scaling */
            for ii in 0..(ntodo as usize) {
                let iret = dnan(shortBuff[sptr]);
                if iret != 0 {
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
                        output[ii] = 0;
                    }
                } else if input[ii] < 0.0 {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MIN;
                } else if input[ii] > DULONGLONG_MAX {
                    *status = OVERFLOW_ERR;
                    output[ii] = ULONGLONG::MAX;
                } else {
                    output[ii] = input[ii] as ULONGLONG;
                }
                sptr += 4;
            }
        } else {
            /* must scale the data */
            for ii in 0..(ntodo as usize) {
                let iret = dnan(shortBuff[sptr]);
                if iret != 0 {
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

                        if zero < 0.0 {
                            *status = OVERFLOW_ERR;
                            output[ii] = ULONGLONG::MIN;
                        } else if zero > DULONGLONG_MAX {
                            *status = OVERFLOW_ERR;
                            output[ii] = ULONGLONG::MAX;
                        } else {
                            output[ii] = zero as ULONGLONG;
                        }
                    }
                } else {
                    dvalue = input[ii] * scale + zero;

                    if dvalue < 0.0 {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MIN;
                    } else if dvalue > DULONGLONG_MAX {
                        *status = OVERFLOW_ERR;
                        output[ii] = ULONGLONG::MAX;
                    } else {
                        output[ii] = dvalue as ULONGLONG;
                    }
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
/// * `nullarray`  — (I) bad pixel array, if nullcheck = 2
/// * `anynull`    — (O) set to 1 if any pixels are null
/// * `output`     — (O) array of converted pixels
/// * `status`     — (IO) error status
pub(crate) fn fffstru8(
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
    nullval: ULONGLONG,
    nullarray: &mut [c_char],
    mut anynull: Option<&mut c_int>,
    output: &mut [ULONGLONG],
    status: &mut c_int,
) -> c_int {
    let mut nullen: c_int = 0;
    let mut dvalue: f64 = 0.0;

    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut tempstore: c_char = 0;
    let chrzero: c_char = bb(b'0'); // 49
    let mut val: f64 = 0.0;
    let mut power: f64 = 0.0;
    let mut exponent: c_int = 0;
    let mut sign: c_int = 0;
    let mut esign: c_int = 0;
    let mut decpt: c_int = 0;

    nullen = strlen_safe(snull) as c_int;
    let mut cptr = 0; /* pointer to start of input string */
    let mut tpos = 0;
    for ii in 0..(ntodo as usize) {
        let originCptr = cptr;
        /* temporarily insert a null terminator at end of the string */
        tpos = cptr + twidth as usize;
        tempstore = input[tpos];
        input[tpos] = 0;

        /* check if null value is defined, and if the    */
        /* column string is identical to the null string */
        if snull[0] != ASCII_NULL_UNDEFINED
            && strncmp_safe(snull, &input[cptr..], nullen as usize) == 0
        {
            if nullcheck != NullCheckType::None {
                if let Some(anynull) = anynull.as_deref_mut() {
                    *anynull = 1;
                }
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            }
            cptr += twidth as usize;
        } else {
            /* value is not the null value, so decode it */
            /* remove any embedded blank characters from the string */

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
                val = val * 10.0 + f64::from(input[cptr]) - f64::from(chrzero); /* accumulate the value */
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
                    val = val * 10.0 + f64::from(input[cptr]) - f64::from(chrzero); /* accumulate the value */
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
                    exponent = exponent * 10 + c_int::from(input[cptr]) - c_int::from(chrzero); /* accumulate exp */
                    cptr += 1;

                    while input[cptr] == bb(b' ') {
                        /* skip embedded blanks */
                        cptr += 1;
                    }
                }
            }

            if input[cptr] != 0 {
                /* should end up at the null terminator */
                let cstring = &input[originCptr..];

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
                    slice_to_str!(cstring),
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
            dvalue = (f64::from(sign) * val / power) * (10.0_f64).powf(f64::from(esign * exponent));

            dvalue = dvalue * scale + zero; /* apply the scaling */

            if dvalue < 0.0 {
                *status = OVERFLOW_ERR;
                output[ii] = ULONGLONG::MIN;
            } else if dvalue > DULONGLONG_MAX {
                *status = OVERFLOW_ERR;
                output[ii] = ULONGLONG::MAX;
            } else {
                output[ii] = dvalue as ULONGLONG;
            }
        }
        /* restore the char that was overwritten by the null */
        input[tpos] = tempstore;
    }
    *status
}

#[cfg(test)]
mod tests {
    use crate::aliases::rust_api::*;
    use crate::fitsio::{
        ASCII_TBL, BAD_DIMEN, BINARY_TBL, BYTE_IMG, LONGLONG, NUM_OVERFLOW, READONLY, TRUE,
        ULONG_IMG, ULONGLONG, ULONGLONG_IMG, fitsfile,
    };
    use crate::helpers::testhelpers::{to_buf, with_temp_file};
    use libc::{c_char, c_int, c_long, c_ulong};

    const UINT_MAX: c_ulong = u32::MAX as c_ulong;
    const INT_MAX: c_long = i32::MAX as c_long;

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

    // ----------------------------------------------------------------------
    // UJ (unsigned long) tests
    // ----------------------------------------------------------------------

    #[test]
    fn test_read_primary_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [c_ulong; 5] = [0, 100, 1000, 10000, UINT_MAX];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 1, &naxes, &mut status);
            fits_write_img_ulng(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 5];
            let mut anynull = -1;
            fits_read_img_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 100, 1000, 10000, UINT_MAX]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_primary_no_nulls() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [c_ulong; 5] = [10, 20, 30, 40, 50];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 1, &naxes, &mut status);
            fits_write_img_ulng(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 5];
            let mut anynull = -1;
            fits_read_img_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [10, 20, 30, 40, 50]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_primary_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [c_ulong; 5] = [10, 20, 30, 40, 50];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 1, &naxes, &mut status);
            fits_write_img_ulng(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 5];
            let mut nularray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_imgnull_ulng(
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
            assert_eq!(result, [10, 20, 30, 40, 50]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_2d_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [3, 2];
            let data: [c_ulong; 6] = [1, 2, 3, 4, 5, 6];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 2, &naxes, &mut status);
            fits_write_2d_ulng(f.as_deref_mut().unwrap(), 1, 3, 3, 2, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 6];
            let mut anynull = -1;
            fits_read_2d_ulng(
                f.as_deref_mut().unwrap(),
                1,
                0,
                3,
                3,
                2,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [1, 2, 3, 4, 5, 6]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_3d_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 3] = [2, 2, 2];
            let data: [c_ulong; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 3, &naxes, &mut status);
            fits_write_3d_ulng(
                f.as_deref_mut().unwrap(),
                1,
                2,
                2,
                2,
                2,
                2,
                &data,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 8];
            let mut anynull = -1;
            fits_read_3d_ulng(
                f.as_deref_mut().unwrap(),
                1,
                0,
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
            assert_eq!(result[0], 1);
            assert_eq!(result[7], 8);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_subsection() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let fpixel: [c_long; 2] = [2, 2];
            let lpixel: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];
            let data: [c_ulong; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 2, &naxes, &mut status);
            fits_write_img_ulng(f.as_deref_mut().unwrap(), 1, 1, 16, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 4];
            let mut anynull = -1;
            fits_read_subset_ulng(
                f.as_deref_mut().unwrap(),
                1,
                2,
                &naxes,
                &fpixel,
                &lpixel,
                &inc,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [6, 7, 10, 11]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_binary_table_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_ulong; 3] = [0, 1000000, UINT_MAX];

            let mut f = make_table(&name, BINARY_TBL, "ULCOL", "1V", 3, &mut status);
            fits_write_col_ulng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 3];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000000, UINT_MAX]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_column_with_nulls() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_ulong; 5] = [10, 20, 30, 40, 50];

            let mut f = make_table(&name, BINARY_TBL, "ULCOL", "5V", 1, &mut status);
            fits_write_col_ulng(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 5];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                5,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [10, 20, 30, 40, 50]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_column_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_ulong; 5] = [10, 20, 30, 40, 50];

            let mut f = make_table(&name, BINARY_TBL, "ULCOL", "5V", 1, &mut status);
            fits_write_col_ulng(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 5];
            let mut nularray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_colnull_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                5,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [10, 20, 30, 40, 50]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_byte_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [u8; 3] = [0, 128, 255];

            let mut f = make_table(&name, BINARY_TBL, "BCOL", "1B", 3, &mut status);
            fits_write_col_byt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 3];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 128, 255]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_short_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [i16; 3] = [0, 1000, 32767];

            let mut f = make_table(&name, BINARY_TBL, "ICOL", "1I", 3, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 3];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000, 32767]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_long_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 3] = [0, 100000, INT_MAX];

            let mut f = make_table(&name, BINARY_TBL, "JCOL", "1J", 3, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 3];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 100000, INT_MAX as c_ulong]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_longlong_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 3] = [0, 100000, INT_MAX as LONGLONG];

            let mut f = make_table(&name, BINARY_TBL, "KCOL", "1K", 3, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 3];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 100000, INT_MAX as c_ulong]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_float_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f32; 3] = [0.0, 1000.0, 100000.0];

            let mut f = make_table(&name, BINARY_TBL, "ECOL", "1E", 3, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 3];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000, 100000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_double_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 3] = [0.0, 1000.0, 100000.0];

            let mut f = make_table(&name, BINARY_TBL, "DCOL", "1D", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 3];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000, 100000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_from_ascii_table() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 3] = [0.0, 1000.0, 100000.0];

            let mut f = make_table(&name, ASCII_TBL, "NUMCOL", "F12.1", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 3];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000, 100000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_multirow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_ulong; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

            let mut f = make_table(&name, BINARY_TBL, "ULCOL", "3V", 3, &mut status);
            fits_write_col_ulng(f.as_deref_mut().unwrap(), 1, 1, 1, 9, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 9];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                9,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1);
            assert_eq!(result[4], 5);
            assert_eq!(result[8], 9);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_variable_length_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data1: [c_ulong; 3] = [10, 20, 30];
            let data2: [c_ulong; 2] = [40, 50];

            let mut f = make_table(&name, BINARY_TBL, "VARUL", "1PV", 2, &mut status);
            fits_write_col_ulng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data1, &mut status);
            fits_write_col_ulng(f.as_deref_mut().unwrap(), 1, 2, 1, 2, &data2, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 5];
            let mut anynull = -1;
            let mut nelem: c_long = 0;
            fits_read_descript(
                f.as_deref_mut().unwrap(),
                1,
                1,
                Some(&mut nelem),
                None,
                &mut status,
            );
            assert_eq!(nelem, 3);
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result[0..3],
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(result[0], 10);
            assert_eq!(result[1], 20);
            assert_eq!(result[2], 30);
            fits_read_descript(
                f.as_deref_mut().unwrap(),
                1,
                2,
                Some(&mut nelem),
                None,
                &mut status,
            );
            assert_eq!(nelem, 2);
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                2,
                1,
                2,
                0,
                &mut result[3..5],
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(result[3], 40);
            assert_eq!(result[4], 50);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_short_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [i16; 1] = [-1];

            let mut f = make_table(&name, BINARY_TBL, "ICOL", "1I", 1, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 1];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_long_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 1] = [-1];

            let mut f = make_table(&name, BINARY_TBL, "JCOL", "1J", 1, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 1];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_longlong_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 1] = [-1];

            let mut f = make_table(&name, BINARY_TBL, "KCOL", "1K", 1, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 1];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_float_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f32; 1] = [-1.0];

            let mut f = make_table(&name, BINARY_TBL, "ECOL", "1E", 1, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 1];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_double_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 1] = [-1.0];

            let mut f = make_table(&name, BINARY_TBL, "DCOL", "1D", 1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as c_ulong; 1];
            let mut anynull = -1;
            fits_read_col_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_noncontiguous_3d() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 3] = [2, 2, 2];
            let data: [c_ulong; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 3, &naxes, &mut status);
            fits_write_3d_ulng(
                f.as_deref_mut().unwrap(),
                1,
                2,
                2,
                2,
                2,
                2,
                &data,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 18];
            let mut anynull = -1;
            fits_read_3d_ulng(
                f.as_deref_mut().unwrap(),
                1,
                0,
                3,
                3,
                2,
                2,
                2,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result[0], 1);
            assert_eq!(result[1], 2);
            assert_eq!(result[3], 3);
            assert_eq!(result[4], 4);
            assert_eq!(result[9], 5);
            assert_eq!(result[10], 6);
            assert_eq!(result[12], 7);
            assert_eq!(result[13], 8);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_bad_dimen_3d() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 3] = [4, 4, 2];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 3, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 32];
            let mut anynull = -1;
            fits_read_3d_ulng(
                f.as_deref_mut().unwrap(),
                1,
                0,
                2,
                2,
                4,
                4,
                2,
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
    fn test_read_group_parameters() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let pdata: [c_ulong; 2] = [10, 20];
            let idata: [c_ulong; 4] = [1, 2, 3, 4];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_grphdr(
                f.as_deref_mut().unwrap(),
                TRUE as c_int,
                ULONG_IMG,
                1,
                &naxes,
                2,
                1,
                TRUE as c_int,
                &mut status,
            );
            fits_write_grppar_ulng(f.as_deref_mut().unwrap(), 1, 1, 2, &pdata, &mut status);
            fits_write_img_ulng(f.as_deref_mut().unwrap(), 1, 1, 4, &idata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut presult = [0 as c_ulong; 2];
            fits_read_grppar_ulng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                2,
                &mut presult,
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(presult, [10, 20]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_subsection_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let fpixel: [c_long; 2] = [2, 2];
            let lpixel: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];
            let data: [c_ulong; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 2, &naxes, &mut status);
            fits_write_img_ulng(f.as_deref_mut().unwrap(), 1, 1, 16, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as c_ulong; 4];
            let mut nularray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_subsetnull_ulng(
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
            assert_eq!(result, [6, 7, 10, 11]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ----------------------------------------------------------------------
    // UJJ (unsigned LONGLONG) tests
    // ----------------------------------------------------------------------

    #[test]
    fn test_ujj_read_primary_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [ULONGLONG; 5] = [0, 100, 1000, 10000, 4294967295];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                ULONGLONG_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_ulnglng(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as ULONGLONG; 5];
            let mut anynull = -1;
            fits_read_img_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                5,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 100, 1000, 10000, 4294967295]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_primary_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let data: [ULONGLONG; 5] = [10, 20, 30, 40, 50];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                ULONGLONG_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_ulnglng(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as ULONGLONG; 5];
            let mut nularray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_imgnull_ulnglng(
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
            assert_eq!(result, [10, 20, 30, 40, 50]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_2d_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [3, 2];
            let data: [ULONGLONG; 6] = [1, 2, 3, 4, 5, 6];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                ULONGLONG_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_2d_ulnglng(f.as_deref_mut().unwrap(), 1, 3, 3, 2, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as ULONGLONG; 6];
            let mut anynull = -1;
            fits_read_2d_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                0,
                3,
                3,
                2,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [1, 2, 3, 4, 5, 6]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_3d_array() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 3] = [2, 2, 2];
            let data: [ULONGLONG; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                ULONGLONG_IMG,
                3,
                &naxes,
                &mut status,
            );
            fits_write_3d_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                2,
                2,
                2,
                2,
                2,
                &data,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as ULONGLONG; 8];
            let mut anynull = -1;
            fits_read_3d_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                0,
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
            assert_eq!(result[0], 1);
            assert_eq!(result[7], 8);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_subsection() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let fpixel: [c_long; 2] = [2, 2];
            let lpixel: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];
            let data: [ULONGLONG; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                ULONGLONG_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_img_ulnglng(f.as_deref_mut().unwrap(), 1, 1, 16, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as ULONGLONG; 4];
            let mut anynull = -1;
            fits_read_subset_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                2,
                &naxes,
                &fpixel,
                &lpixel,
                &inc,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [6, 7, 10, 11]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_subsection_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let fpixel: [c_long; 2] = [2, 2];
            let lpixel: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];
            let data: [ULONGLONG; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                ULONGLONG_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_img_ulnglng(f.as_deref_mut().unwrap(), 1, 1, 16, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut result = [0 as ULONGLONG; 4];
            let mut nularray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_subsetnull_ulnglng(
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
            assert_eq!(result, [6, 7, 10, 11]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_group_parameters() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let pdata: [ULONGLONG; 2] = [10, 20];
            let idata: [ULONGLONG; 4] = [1, 2, 3, 4];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_grphdr(
                f.as_deref_mut().unwrap(),
                TRUE as c_int,
                ULONGLONG_IMG,
                1,
                &naxes,
                2,
                1,
                TRUE as c_int,
                &mut status,
            );
            fits_write_grppar_ulnglng(f.as_deref_mut().unwrap(), 1, 1, 2, &pdata, &mut status);
            fits_write_img_ulnglng(f.as_deref_mut().unwrap(), 1, 1, 4, &idata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut presult = [0 as ULONGLONG; 2];
            fits_read_grppar_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                2,
                &mut presult,
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(presult, [10, 20]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_binary_table_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [ULONGLONG; 3] = [0, 1000000, 4294967295];

            let mut f = make_table(&name, BINARY_TBL, "ULLCOL", "1W", 3, &mut status);
            fits_write_col_ulnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 3];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000000, 4294967295]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_column_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [ULONGLONG; 5] = [10, 20, 30, 40, 50];

            let mut f = make_table(&name, BINARY_TBL, "ULLCOL", "5W", 1, &mut status);
            fits_write_col_ulnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 5];
            let mut nularray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_colnull_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                5,
                &mut result,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [10, 20, 30, 40, 50]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_from_byte_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [u8; 3] = [0, 128, 255];

            let mut f = make_table(&name, BINARY_TBL, "BCOL", "1B", 3, &mut status);
            fits_write_col_byt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 3];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 128, 255]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_from_short_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [i16; 3] = [0, 1000, 32767];

            let mut f = make_table(&name, BINARY_TBL, "ICOL", "1I", 3, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 3];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000, 32767]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_from_long_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 3] = [0, 100000, INT_MAX];

            let mut f = make_table(&name, BINARY_TBL, "JCOL", "1J", 3, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 3];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 100000, INT_MAX as ULONGLONG]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_from_longlong_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 3] = [0, 100000, LONGLONG::MAX];

            let mut f = make_table(&name, BINARY_TBL, "KCOL", "1K", 3, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 3];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 100000, LONGLONG::MAX as ULONGLONG]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_from_float_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f32; 3] = [0.0, 1000.0, 100000.0];

            let mut f = make_table(&name, BINARY_TBL, "ECOL", "1E", 3, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 3];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000, 100000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_from_double_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 3] = [0.0, 1000.0, 100000.0];

            let mut f = make_table(&name, BINARY_TBL, "DCOL", "1D", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 3];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000, 100000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_read_from_ascii_table() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 3] = [0.0, 1000.0, 100000.0];

            let mut f = make_table(&name, ASCII_TBL, "NUMCOL", "F20.1", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 3];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(result, [0, 1000, 100000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_short_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [i16; 1] = [-1];

            let mut f = make_table(&name, BINARY_TBL, "ICOL", "1I", 1, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 1];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_long_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 1] = [-1];

            let mut f = make_table(&name, BINARY_TBL, "JCOL", "1J", 1, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 1];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_longlong_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 1] = [-1];

            let mut f = make_table(&name, BINARY_TBL, "KCOL", "1K", 1, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 1];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_float_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f32; 1] = [-1.0];

            let mut f = make_table(&name, BINARY_TBL, "ECOL", "1E", 1, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 1];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ujj_double_overflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 1] = [-1.0];

            let mut f = make_table(&name, BINARY_TBL, "DCOL", "1D", 1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut result = [0 as ULONGLONG; 1];
            let mut anynull = -1;
            fits_read_col_ulnglng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                0,
                &mut result,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, NUM_OVERFLOW);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }
}
