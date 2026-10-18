//! Generic-datatype routines that read data elements from a FITS image or
//! table.
//!
//! These take a `datatype` code at run time and dispatch to the corresponding
//! typed module -- `getcolb` for [`TBYTE`], `getcold` for [`TDOUBLE`], and so
//! on. The typed routines do the real work; this is the layer the `fits_read_*`
//! entry points sit on.
//!
//! Reading converts: the value handed back is
//! `(the value in the file) * TSCALn + TZEROn`, cast to the requested type, so
//! a column stored as 16-bit integers can be read as doubles. A value equal to
//! the column's null value is reported through `anynul` and either left alone
//! or replaced by the caller's `nulval`, depending on which form is called.
//!
//! Ported from CFITSIO's `getcol.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center.
#![warn(missing_docs)]

use core::mem;
use core::slice;

use crate::c_types::{c_char, c_int, c_long, c_short, c_uint, c_ulong, c_ushort, c_void};
use crate::imcompress::{fits_read_compressed_img, fits_read_compressed_pixels};

use bytemuck::{cast_slice, cast_slice_mut};

use crate::NullValue;
use crate::bytes_per_datatype;
use crate::fitscore::{
    ffgidm_safe, ffgisz_safe, ffgiszll_safe, ffgnrwll_safe, ffgtclll_safe,
    fits_is_compressed_image_safe,
};
use crate::fitscore::{ffpmsg_slice, ffpmsg_str};
use crate::getcolb::{ffgclb, ffgpfb_safe, ffgpvb_safe, ffgsvb_safe};
use crate::getcold::{ffgcfm_safe, ffgcld, ffgpfd_safe, ffgpvd_safe, ffgsvd_safe};
use crate::getcole::{ffgcfc_safe, ffgcle, ffgpfe_safe, ffgpve_safe, ffgsve_safe};
use crate::getcoli::{ffgcli, ffgpfi_safe, ffgpvi_safe, ffgsvi_safe};
use crate::getcolj::{
    ffgclj, ffgcljj, ffgpfj_safe, ffgpfjj_safe, ffgpvj_safe, ffgpvjj_safe, ffgsvj_safe,
    ffgsvjj_safe,
};
use crate::getcolk::{ffgclk, ffgpfk_safe, ffgpvk_safe, ffgsvk_safe};
use crate::getcoll::{ffgcll, ffgcx_safe};
use crate::getcols::{ffgcls, tstring_item_len};
use crate::getcolsb::{ffgclsb, ffgpfsb_safe, ffgpvsb_safe, ffgsvsb_safe};
use crate::getcolui::{ffgclui, ffgpfui_safe, ffgpvui_safe, ffgsvui_safe};
use crate::getcoluj::{
    ffgcluj, ffgclujj, ffgpfuj_safe, ffgpfujj_safe, ffgpvuj_safe, ffgpvujj_safe, ffgsvuj_safe,
    ffgsvujj_safe,
};
use crate::getcoluk::{ffgcluk, ffgpfuk_safe, ffgpvuk_safe, ffgsvuk_safe};
use crate::int_snprintf;
use crate::{NullCheckType, fitsio::*};
use crate::{buffers::*, calculate_subsection_length};

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements will be set equal to NULVAL, unless NULVAL=0
/// in which case no checking for undefined values will be performed.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `datatype` — (I) datatype of the value
/// * `firstpix` — (I) coord of first pixel to read (1s based)
/// * `nelem`    — (I) number of values to read
/// * `nulval`   — (I) value for undefined pixels
/// * `array`    — (O) array of values that are returned
/// * `anynul`   — (O) set to 1 if any values are null; else 0
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgpxv(
    fptr: *mut fitsfile,
    datatype: c_int,
    firstpix: *const c_long,
    nelem: LONGLONG,
    nulval: *const c_void,
    array: *mut c_void,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let bytes = bytes_per_datatype(datatype).unwrap();
        let array = slice::from_raw_parts_mut(array.cast(), bytes * nelem as usize);
        let nulval = NullValue::from_raw_ptr(datatype, nulval);

        /* get the size of the image */
        let mut naxis = 0;
        ffgidm_safe(fptr, &mut naxis, status);

        let firstpix = slice::from_raw_parts(firstpix, naxis as usize);

        ffgpxv_safe(
            fptr, datatype, firstpix, nelem, nulval, array, anynul, status,
        )
    }
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements will be set equal to NULVAL, unless NULVAL=0
/// in which case no checking for undefined values will be performed.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `datatype` — (I) datatype of the value
/// * `firstpix` — (I) coord of first pixel to read (1s based)
/// * `nelem`    — (I) number of values to read
/// * `nulval`   — (I) value for undefined pixels
/// * `array`    — (O) array of values that are returned
/// * `anynul`   — (O) set to 1 if any values are null; else 0
/// * `status`   — (IO) error status
pub fn ffgpxv_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    firstpix: &[c_long],
    nelem: LONGLONG,
    nulval: Option<NullValue>,
    array: &mut [u8],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut tfirstpix: [LONGLONG; 99] = [0; 99];
    let mut naxis = 0;

    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    /* get the size of the image */
    ffgidm_safe(fptr, &mut naxis, status);

    for i in 0..(naxis as usize) {
        tfirstpix[i] = firstpix[i] as LONGLONG;
    }

    ffgpxvll_safe(
        fptr, datatype, &tfirstpix, nelem, nulval, array, anynul, status,
    );

    *status
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
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
/// * `fptr`     — (I) FITS file pointer
/// * `datatype` — (I) datatype of the value
/// * `firstpix` — (I) coord of first pixel to read (1s based)
/// * `nelem`    — (I) number of values to read
/// * `nulval`   — (I) value for undefined pixels
/// * `array`    — (O) array of values that are returned
/// * `anynul`   — (O) set to 1 if any values are null; else 0
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgpxvll(
    fptr: *mut fitsfile,
    datatype: c_int,
    firstpix: *const LONGLONG,
    nelem: LONGLONG,
    nulval: *const c_void,
    array: *mut c_void,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let bytes = bytes_per_datatype(datatype).unwrap();
        let array = slice::from_raw_parts_mut(array.cast(), bytes * nelem as usize);
        let nulval = NullValue::from_raw_ptr(datatype, nulval);

        if *status > 0 || nelem == 0 {
            /* inherit input status value if > 0 */
            return *status;
        }

        /* get the size of the image */
        let mut naxis = 0;
        ffgidm_safe(fptr, &mut naxis, status);

        if naxis == 0 {
            *status = BAD_DIMEN;
            return *status;
        }

        let firstpix = slice::from_raw_parts(firstpix, naxis as usize);

        ffgpxvll_safe(
            fptr, datatype, firstpix, nelem, nulval, array, anynul, status,
        )
    }
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
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
/// * `fptr`     — (I) FITS file pointer
/// * `datatype` — (I) datatype of the value
/// * `firstpix` — (I) coord of first pixel to read (1s based)
/// * `nelem`    — (I) number of values to read
/// * `nulval`   — (I) value for undefined pixels
/// * `array`    — (O) array of values that are returned
/// * `anynul`   — (O) set to 1 if any values are null; else 0
/// * `status`   — (IO) error status
pub fn ffgpxvll_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    firstpix: &[LONGLONG],
    nelem: LONGLONG,
    nulval: Option<NullValue>,
    array: &mut [u8],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut naxis = 0;

    let nullcheck = NullCheckType::SetPixel;
    let mut naxes: [LONGLONG; 9] = [0; 9];
    let mut trc: [LONGLONG; 9] = [1; 9];
    let inc: [c_long; 9] = [1; 9];
    let mut dimsize: LONGLONG = 1;
    let mut firstelem: LONGLONG = 0;

    let mut dummy_nularray = vec![0; nelem as usize];

    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    /* get the size of the image */
    ffgidm_safe(fptr, &mut naxis, status);

    ffgiszll_safe(fptr, 9, &mut naxes, status);

    if naxis == 0 || naxes[0] == 0 {
        *status = BAD_DIMEN;
        return *status;
    }

    /* calculate the position of the first element in the array */
    firstelem = 0;
    for ii in 0..(naxis as usize) {
        firstelem += (firstpix[ii] - 1) * dimsize;
        dimsize *= naxes[ii];
        trc[ii] = firstpix[ii];
    }
    firstelem += 1;

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */

        /* test for special case of reading an integral number of */
        /* rows in a 2D or 3D image (which includes reading the whole image */

        if naxis > 1 && naxis < 4 && firstpix[0] == 1 && (nelem / naxes[0]) * naxes[0] == nelem {
            /* calculate coordinate of last pixel */
            trc[0] = naxes[0]; /* reading whole rows */
            trc[1] = firstpix[1] + (nelem / naxes[0] - 1);
            while trc[1] > naxes[1] {
                trc[1] -= naxes[1];
                trc[2] += 1; /* increment to next plane of cube */
            }

            fits_read_compressed_img(
                fptr,
                datatype,
                firstpix,
                &trc,
                &inc,
                NullCheckType::SetPixel,
                &nulval,
                array,
                None,
                anynul,
                status,
            );
        } else {
            fits_read_compressed_pixels(
                fptr,
                datatype,
                firstelem,
                nelem,
                nullcheck,
                &nulval,
                cast_slice_mut(array),
                None,
                anynul,
                status,
            );
        }

        return *status;
    }

    if datatype == TBYTE {
        match nulval {
            None => {
                ffgclb(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclb(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::UByte(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSBYTE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclsb(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclsb(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Byte(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TUSHORT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclui(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclui(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::UShort(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSHORT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcli(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcli(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Short(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TUINT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcluk(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcluk(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::UInt(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TINT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclk(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclk(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Int(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TULONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcluj(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcluj(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::ULong(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclj(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclj(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Long(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TULONGLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclujj(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclujj(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::ULONGLONG(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLONGLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcljj(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcljj(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::LONGLONG(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TFLOAT {
        let array = cast_slice_mut(array);
        match nulval {
            None => {
                ffgcle(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0.0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcle(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Float(val) => val,
                        _ => 0.0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TDOUBLE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcld(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0.0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcld(
                    fptr,
                    2,
                    1,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Double(val) => val,
                        _ => 0.0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else {
        *status = BAD_DATATYPE;
    }

    *status
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// The nullarray values will = 1 if the corresponding array value is null.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the value
/// * `firstpix`  — (I) coord of first pixel to read (1s based)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) returned array of null value flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgpxf(
    fptr: *mut fitsfile,
    datatype: c_int,
    firstpix: *const c_long,
    nelem: LONGLONG,
    array: *mut c_void,
    nullarray: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let nullarray = slice::from_raw_parts_mut(nullarray, nelem as usize);

        let bytes = bytes_per_datatype(datatype).unwrap();
        let array = slice::from_raw_parts_mut(array.cast(), bytes * nelem as usize);
        let anynul = anynul.as_mut();

        let mut naxis = 0;

        if *status > 0 || nelem == 0 {
            /* inherit input status value if > 0 */
            return *status;
        }

        /* get the size of the image */
        ffgidm_safe(fptr, &mut naxis, status);

        let firstpix = slice::from_raw_parts(firstpix, naxis as usize);

        ffgpxf_safe(
            fptr, datatype, firstpix, nelem, array, nullarray, anynul, status,
        )
    }
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// The nullarray values will = 1 if the corresponding array value is null.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the value
/// * `firstpix`  — (I) coord of first pixel to read (1s based)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) returned array of null value flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgpxf_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    firstpix: &[c_long],
    nelem: LONGLONG,
    array: &mut [u8],
    nullarray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut tfirstpix: [LONGLONG; 99] = [0; 99];
    let mut naxis = 0;

    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    /* get the size of the image */
    ffgidm_safe(fptr, &mut naxis, status);

    for i in 0..(naxis as usize) {
        tfirstpix[i] = firstpix[i] as LONGLONG;
    }

    ffgpxfll_safe(
        fptr, datatype, &tfirstpix, nelem, array, nullarray, anynul, status,
    );

    *status
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// The nullarray values will = 1 if the corresponding array value is null.
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
/// * `datatype`  — (I) datatype of the value
/// * `firstpix`  — (I) coord of first pixel to read (1s based)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) returned array of null value flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgpxfll(
    fptr: *mut fitsfile,
    datatype: c_int,
    firstpix: *const LONGLONG,
    nelem: LONGLONG,
    array: *mut c_void,
    nullarray: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let nullarray = slice::from_raw_parts_mut(nullarray, nelem as usize);
        let bytes = bytes_per_datatype(datatype).unwrap();
        let array = slice::from_raw_parts_mut(array.cast(), bytes * nelem as usize);

        /* get the size of the image */
        let mut naxis = 0;
        ffgidm_safe(fptr, &mut naxis, status);

        let firstpix = slice::from_raw_parts(firstpix, naxis as usize);

        ffgpxfll_safe(
            fptr,
            datatype,
            firstpix,
            nelem,
            array,
            nullarray,
            anynul.as_mut(),
            status,
        )
    }
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// The nullarray values will = 1 if the corresponding array value is null.
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
/// * `datatype`  — (I) datatype of the value
/// * `firstpix`  — (I) coord of first pixel to read (1s based)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) returned array of null value flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgpxfll_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    firstpix: &[LONGLONG],
    nelem: LONGLONG,
    array: &mut [u8],
    nullarray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut naxis = 0;
    let nullcheck = NullCheckType::SetNullArray;
    let mut naxes: [LONGLONG; 9] = [0; 9];
    let mut dimsize: LONGLONG = 1;
    let mut firstelem: LONGLONG = 0;

    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    /* get the size of the image */
    ffgidm_safe(fptr, &mut naxis, status);
    ffgiszll_safe(fptr, 9, &mut naxes, status);

    /* calculate the position of the first element in the array */
    firstelem = 0;
    for ii in 0..(naxis as usize) {
        firstelem += (firstpix[ii] - 1) * dimsize;
        dimsize *= naxes[ii];
    }
    firstelem += 1;

    if fits_is_compressed_image_safe(fptr, status) > 0 {
        /* this is a compressed image in a binary table */
        fits_read_compressed_pixels(
            fptr,
            datatype,
            firstelem,
            nelem,
            nullcheck,
            &None,
            array,
            Some(nullarray),
            anynul,
            status,
        );
        return *status;
    }

    if datatype == TBYTE {
        let array = cast_slice_mut(array);

        ffgclb(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TSBYTE {
        let array = cast_slice_mut(array);

        ffgclsb(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TUSHORT {
        let array = cast_slice_mut(array);

        ffgclui(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TSHORT {
        let array = cast_slice_mut(array);

        ffgcli(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TUINT {
        let array = cast_slice_mut(array);

        ffgcluk(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TINT {
        let array = cast_slice_mut(array);

        ffgclk(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TULONG {
        let array = cast_slice_mut(array);

        ffgcluj(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TLONG {
        let array = cast_slice_mut(array);

        ffgclj(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TULONGLONG {
        let array = cast_slice_mut(array);

        ffgclujj(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TLONGLONG {
        let array = cast_slice_mut(array);

        ffgcljj(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TFLOAT {
        let array = cast_slice_mut(array);

        ffgcle(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0.0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TDOUBLE {
        let array = cast_slice_mut(array);

        ffgcld(
            fptr,
            2,
            1,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            0.0,
            array,
            nullarray,
            anynul,
            status,
        );
    } else {
        *status = BAD_DATATYPE;
    }

    *status
}

/// Read an section of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements will be set equal to NULVAL, unless NULVAL=0
/// in which case no checking for undefined values will be performed.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `datatype` — (I) datatype of the value
/// * `blc`      — (I) 'bottom left corner' of the subsection
/// * `trc`      — (I) 'top right corner' of the subsection
/// * `inc`      — (I) increment to be applied in each dim.
/// * `nulval`   — (I) value for undefined pixels
/// * `array`    — (O) array of values that are returned
/// * `anynul`   — (O) set to 1 if any values are null; else 0
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgsv(
    fptr: *mut fitsfile,
    datatype: c_int,
    blc: *const c_long,
    trc: *const c_long,
    inc: *const c_long,
    nulval: *const c_void,
    array: *mut c_void,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let nulval = NullValue::from_raw_ptr(datatype, nulval);

        let mut naxis = 0;
        /* get the size of the image */
        ffgidm_safe(fptr, &mut naxis, status);

        let blc = slice::from_raw_parts(blc, naxis as usize);
        let trc = slice::from_raw_parts(trc, naxis as usize);
        let inc = slice::from_raw_parts(inc, naxis as usize);

        let nelem = calculate_subsection_length(blc, trc, inc);
        let bytes = bytes_per_datatype(datatype).unwrap();
        let array = slice::from_raw_parts_mut(array.cast(), bytes * nelem as usize);

        ffgsv_safe(fptr, datatype, blc, trc, inc, nulval, array, anynul, status)
    }
}

/// Read an section of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements will be set equal to NULVAL, unless NULVAL=0
/// in which case no checking for undefined values will be performed.
/// ANYNUL is returned with a value of .true. if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `datatype` — (I) datatype of the value
/// * `blc`      — (I) 'bottom left corner' of the subsection
/// * `trc`      — (I) 'top right corner' of the subsection
/// * `inc`      — (I) increment to be applied in each dim.
/// * `nulval`   — (I) value for undefined pixels
/// * `array`    — (O) array of values that are returned
/// * `anynul`   — (O) set to 1 if any values are null; else 0
/// * `status`   — (IO) error status
pub fn ffgsv_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    blc: &[c_long],
    trc: &[c_long],
    inc: &[c_long],
    nulval: Option<NullValue>,
    array: &mut [u8],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut naxis = 0;
    let mut naxes: [c_long; 9] = [0; 9];
    let mut nelem: LONGLONG = 1;
    let mut ii: usize = 0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    /* get the size of the image */
    ffgidm_safe(fptr, &mut naxis, status);
    ffgisz_safe(fptr, 9, &mut naxes, status);

    /* test for the important special case where we are reading the whole image */
    /* this is only useful for images that are not tile-compressed */
    if fits_is_compressed_image_safe(fptr, status) == 0 {
        while ii < naxis as usize {
            if inc[ii] != 1 || blc[ii] != 1 || trc[ii] != naxes[ii] {
                break;
            }

            nelem *= naxes[ii] as LONGLONG;
            ii += 1;
        }

        if ii == naxis as usize {
            /* read the whole image more efficiently */
            ffgpxv_safe(fptr, datatype, blc, nelem, nulval, array, anynul, status);
            return *status;
        }
    }

    if datatype == TBYTE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvb_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvb_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::UByte(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSBYTE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvsb_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvsb_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::Byte(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TUSHORT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvui_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvui_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::UShort(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSHORT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvi_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvi_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::Short(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TUINT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvuk_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvuk_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::UInt(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TINT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvk_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvk_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::Int(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TULONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvuj_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvuj_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::ULong(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvj_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvj_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::Long(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TULONGLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvujj_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvujj_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::ULONGLONG(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLONGLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvjj_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvjj_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::LONGLONG(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TFLOAT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsve_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0.0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsve_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::Float(val) => val,
                        _ => 0.0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TDOUBLE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgsvd_safe(
                    fptr, 1, naxis, &naxes, blc, trc, inc, 0.0, array, anynul, status,
                );
            }
            Some(val) => {
                ffgsvd_safe(
                    fptr,
                    1,
                    naxis,
                    &naxes,
                    blc,
                    trc,
                    inc,
                    match val {
                        NullValue::Double(val) => val,
                        _ => 0.0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else {
        *status = BAD_DATATYPE;
    }

    *status
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
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
/// anynul can be a null pointer
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the value
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for undefined pixels
/// * `array`     — (O) array of values that are returned
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgpv(
    fptr: *mut fitsfile,
    datatype: c_int,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: *const c_void,
    array: *mut c_void,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let bytes = bytes_per_datatype(datatype).unwrap();
        let array = slice::from_raw_parts_mut(array.cast(), bytes * nelem as usize);

        let anynul = anynul.as_mut();
        let nulval = NullValue::from_raw_ptr(datatype, nulval);

        ffgpv_safe(
            fptr, datatype, firstelem, nelem, nulval, array, anynul, status,
        )
    }
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
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
/// anynul can be a null pointer
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the value
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for undefined pixels
/// * `array`     — (O) array of values that are returned
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgpv_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: Option<NullValue>,
    array: &mut [u8],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if datatype == TBYTE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvb_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvb_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::UByte(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSBYTE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvsb_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvsb_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::Byte(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TUSHORT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvui_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvui_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::UShort(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSHORT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvi_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvi_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::Short(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TUINT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvuk_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvuk_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::UInt(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TINT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvk_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvk_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::Int(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TULONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvuj_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvuj_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::ULong(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvj_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvj_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::Long(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TULONGLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvujj_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvujj_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::ULONGLONG(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLONGLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvjj_safe(fptr, 1, firstelem, nelem, 0, array, anynul, status);
            }
            Some(val) => {
                ffgpvjj_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::LONGLONG(val) => val,
                        _ => 0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TFLOAT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpve_safe(fptr, 1, firstelem, nelem, 0.0, array, anynul, status);
            }
            Some(val) => {
                ffgpve_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::Float(val) => val,
                        _ => 0.0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TDOUBLE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgpvd_safe(fptr, 1, firstelem, nelem, 0.0, array, anynul, status);
            }
            Some(val) => {
                ffgpvd_safe(
                    fptr,
                    1,
                    firstelem,
                    nelem,
                    match val {
                        NullValue::Double(val) => val,
                        _ => 0.0,
                    },
                    array,
                    anynul,
                    status,
                );
            }
        }
    } else {
        *status = BAD_DATATYPE;
    }

    *status
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// The nullarray values will = 1 if the corresponding array value is null.
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
/// * `datatype`  — (I) datatype of the value
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) array of null value flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgpf(
    fptr: *mut fitsfile,
    datatype: c_int,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut c_void,
    nullarray: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let nullarray = slice::from_raw_parts_mut(nullarray, nelem as usize);
        let bytes = bytes_per_datatype(datatype).unwrap();
        let array = slice::from_raw_parts_mut(array.cast(), bytes * nelem as usize);

        let anynul = anynul.as_mut();

        ffgpf_safe(
            fptr, datatype, firstelem, nelem, array, nullarray, anynul, status,
        )
    }
}

/// Read an array of values from the primary array. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// The nullarray values will = 1 if the corresponding array value is null.
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
/// * `datatype`  — (I) datatype of the value
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) array of null value flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgpf_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [u8],
    nullarray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if datatype == TBYTE {
        let array = cast_slice_mut(array);

        ffgpfb_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TSBYTE {
        let array = cast_slice_mut(array);

        ffgpfsb_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TUSHORT {
        let array = cast_slice_mut(array);

        ffgpfui_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TSHORT {
        let array = cast_slice_mut(array);

        ffgpfi_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TUINT {
        let array = cast_slice_mut(array);

        ffgpfuk_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TINT {
        let array = cast_slice_mut(array);

        ffgpfk_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TULONG {
        let array = cast_slice_mut(array);

        ffgpfuj_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TLONG {
        let array = cast_slice_mut(array);

        ffgpfj_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TULONGLONG {
        let array = cast_slice_mut(array);

        ffgpfujj_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TLONGLONG {
        let array = cast_slice_mut(array);

        ffgpfjj_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TFLOAT {
        let array = cast_slice_mut(array);

        ffgpfe_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else if datatype == TDOUBLE {
        let array = cast_slice_mut(array);

        ffgpfd_safe(fptr, 1, firstelem, nelem, array, nullarray, anynul, status);
    } else {
        *status = BAD_DATATYPE;
    }

    *status
}

/// Read an array of values from a table column. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements will be set equal to NULVAL, unless NULVAL=0
/// in which case no checking for undefined values will be performed.
/// ANYNUL is returned with a value of true if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the value
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for undefined pixels
/// * `array`     — (O) array of values that are returned
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcv(
    fptr: *mut fitsfile,
    datatype: c_int,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: *const c_void,
    array: *mut c_void,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let anynul = anynul.as_mut();

        let nulval = NullValue::from_raw_ptr(datatype, nulval);
        let bytes = bytes_per_datatype(datatype).unwrap();
        let array = slice::from_raw_parts_mut(array.cast(), bytes * nelem as usize);

        ffgcv_safe(
            fptr,
            datatype,
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

/// Read an array of values from a table column. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements will be set equal to NULVAL, unless NULVAL=0
/// in which case no checking for undefined values will be performed.
/// ANYNUL is returned with a value of true if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the value
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `nulval`    — (I) value for undefined pixels
/// * `array`     — (O) array of values that are returned
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgcv_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: Option<NullValue>,
    array: &mut [u8],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut cdummy: [c_char; 2] = [0; 2];

    /* The complex datatypes (TCOMPLEX, TDBLCOMPLEX) are read as pairs of    */
    /* numbers, so the underlying ffgcle/ffgcld call processes nelem * 2     */
    /* elements and would index this dummy null array up to that count.      */
    let mut dummy_nularray = vec![0; nelem as usize * 2];

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if datatype == TBIT {
        let array = cast_slice_mut(array);

        ffgcx_safe(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            array,
            status,
        );
    } else if datatype == TBYTE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclb(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclb(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::UByte(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSBYTE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclsb(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclsb(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Byte(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TUSHORT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclui(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclui(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::UShort(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSHORT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcli(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcli(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Short(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TUINT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcluk(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcluk(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::UInt(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TINT {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclk(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclk(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Int(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TULONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcluj(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcluj(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::ULong(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclj(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclj(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Long(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TULONGLONG {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgclujj(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgclujj(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::ULONGLONG(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLONGLONG {
        let array = cast_slice_mut(array);
        match nulval {
            None => {
                ffgcljj(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcljj(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::LONGLONG(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TFLOAT {
        let array = cast_slice_mut(array);
        match nulval {
            None => {
                ffgcle(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0.,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcle(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Float(val) => val,
                        _ => 0.,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TDOUBLE {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcld(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    0.,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcld(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Double(val) => val,
                        _ => 0.,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TCOMPLEX {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcle(
                    fptr,
                    colnum,
                    firstrow,
                    (firstelem - 1) * 2 + 1,
                    nelem * 2,
                    1,
                    NullCheckType::SetPixel,
                    0.,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcle(
                    fptr,
                    colnum,
                    firstrow,
                    (firstelem - 1) * 2 + 1,
                    nelem * 2,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Float(val) => val,
                        _ => 0.,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TDBLCOMPLEX {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcld(
                    fptr,
                    colnum,
                    firstrow,
                    (firstelem - 1) * 2 + 1,
                    nelem * 2,
                    1,
                    NullCheckType::SetPixel,
                    0.,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcld(
                    fptr,
                    colnum,
                    firstrow,
                    (firstelem - 1) * 2 + 1,
                    nelem * 2,
                    1,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Double(val) => val,
                        _ => 0.,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TLOGICAL {
        let array = cast_slice_mut(array);

        match nulval {
            None => {
                ffgcll(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    NullCheckType::SetPixel,
                    0,
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
            Some(val) => {
                ffgcll(
                    fptr,
                    colnum,
                    firstrow,
                    firstelem,
                    nelem,
                    NullCheckType::SetPixel,
                    match val {
                        NullValue::Logical(val) => val,
                        _ => 0,
                    },
                    array,
                    &mut dummy_nularray,
                    anynul,
                    status,
                );
            }
        }
    } else if datatype == TSTRING {
        let (nstrings, item_len) = match tstring_item_len(fptr, colnum, firstrow, nelem, status) {
            Some(v) => v,
            None => return *status,
        };

        // SAFETY: for TSTRING the caller's buffer holds an array of `char *`,
        // as in the C. Derive the pointer with as_mut_ptr() rather than
        // as_ptr(): the latter reborrows `array` shared and casting that to
        // *mut to build a &mut slice is UB (miri reports a write through a
        // SharedReadOnly tag).
        unsafe {
            // The signature types that buffer as bytes, so the pointer
            // alignment it really has is invisible here (clippy's
            // cast_ptr_alignment); assert it instead of assuming it.
            #[allow(clippy::cast_ptr_alignment)]
            let p = array.as_mut_ptr().cast::<*mut c_char>();
            debug_assert!(p.is_aligned());
            let array = slice::from_raw_parts_mut(p, nstrings);
            let mut v_array = Vec::new();
            for item in array {
                let array_item = slice::from_raw_parts_mut(*item, item_len);
                v_array.push(array_item);
            }

            match nulval {
                None => {
                    cdummy[0] = 0;
                    ffgcls(
                        fptr,
                        colnum,
                        firstrow,
                        firstelem,
                        nelem,
                        NullCheckType::SetPixel,
                        Some(&cdummy),
                        &mut v_array,
                        &mut dummy_nularray,
                        anynul,
                        status,
                    );
                }
                Some(val) => match val {
                    NullValue::String(x) => {
                        ffgcls(
                            fptr,
                            colnum,
                            firstrow,
                            firstelem,
                            nelem,
                            NullCheckType::SetPixel,
                            Some(cast_slice(x.as_bytes_with_nul())),
                            &mut v_array,
                            &mut dummy_nularray,
                            anynul,
                            status,
                        );
                    }
                    _ => {
                        ffgcls(
                            fptr,
                            colnum,
                            firstrow,
                            firstelem,
                            nelem,
                            NullCheckType::SetPixel,
                            Some(&[0]),
                            &mut v_array,
                            &mut dummy_nularray,
                            anynul,
                            status,
                        );
                    }
                },
            }
        }
    } else {
        *status = BAD_DATATYPE;
    }

    *status
}

/// Read arrays of values from NCOLS table columns. This is an optimization
/// to read all columns in one pass through the table.  The datatypes of the
/// input arrays are defined by the 3rd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements for column `i` will be set equal to `*(nulval[i])`, unless `nulval[i]=0`
/// in which case no checking for undefined values will be performed.
/// `anynul[i]` is returned with a value of true if any pixels in column `i` are undefined.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `ncols`    — (I) number of columns to read
/// * `datatype` — (I) datatypes of the values
/// * `colnum`   — (I) columns numbers to read (1 = 1st col)
/// * `firstrow` — (I) first row to read (1 = 1st row)
/// * `nrows`    — (I) number of rows to read
/// * `nulval`   — (I) array of pointers to values for undefined pixels
/// * `array`    — (O) array of pointers to values that are returned
/// * `anynul`   — (O) `anynul[i]` set to 1 if any values in column i are null; else 0
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcvn(
    fptr: *mut fitsfile,
    ncols: c_int,
    datatype: *const c_int,
    colnum: *const c_int,
    firstrow: LONGLONG,
    nrows: LONGLONG,
    nulval: *const *const c_void,
    array: *mut *mut c_void,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let array = slice::from_raw_parts_mut(array, nrows as usize);

        let datatype = slice::from_raw_parts(datatype, nrows as usize);
        let colnum = slice::from_raw_parts(colnum, nrows as usize);

        let nulval = slice::from_raw_parts(nulval, nrows as usize);
        let nulval = nulval
            .iter()
            .zip(datatype.iter())
            .map(|(&val, &dtype)| NullValue::from_raw_ptr(dtype, val))
            .collect::<Vec<_>>();

        let anynul = match anynul.is_null() {
            false => Some(slice::from_raw_parts_mut(anynul, ncols as usize)),
            true => None,
        };

        ffgcvn_safe(
            fptr, ncols, datatype, colnum, firstrow, nrows, &nulval, array, anynul, status,
        )
    }
}

/// Read arrays of values from NCOLS table columns. This is an optimization
/// to read all columns in one pass through the table.  The datatypes of the
/// input arrays are defined by the 3rd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// Undefined elements for column `i` will be set equal to `*(nulval[i])`, unless `nulval[i]=0`
/// in which case no checking for undefined values will be performed.
/// `anynul[i]` is returned with a value of true if any pixels in column `i` are undefined.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `ncols`    — (I) number of columns to read
/// * `datatype` — (I) datatypes of the values
/// * `colnum`   — (I) columns numbers to read (1 = 1st col)
/// * `firstrow` — (I) first row to read (1 = 1st row)
/// * `nrows`    — (I) number of rows to read
/// * `nulval`   — (I) array of pointers to values for undefined pixels
/// * `array`    — (O) array of pointers to values that are returned
/// * `anynul`   — (O) `anynul[i]` set to 1 if any values in column i are null; else 0
/// * `status`   — (IO) error status
pub fn ffgcvn_safe(
    fptr: &mut fitsfile,
    ncols: c_int,
    datatype: &[c_int],
    colnum: &[c_int],
    firstrow: LONGLONG,
    nrows: LONGLONG,
    nulval: &[Option<NullValue>],
    array: &mut [*mut c_void],
    mut anynul: Option<&mut [c_int]>,
    status: &mut c_int,
) -> c_int {
    let mut ntotrows: LONGLONG = 0;
    let mut ndone: LONGLONG = 0;
    let mut nread: LONGLONG = 0;
    let mut currow: LONGLONG = 0;
    let mut nrowbuf: c_long = 0;

    let mut sizes: [usize; 255] = [0; 255];

    sizes[TBYTE as usize] = mem::size_of::<c_char>();
    sizes[TSBYTE as usize] = mem::size_of::<c_char>();
    sizes[TLOGICAL as usize] = mem::size_of::<c_char>();
    sizes[TUSHORT as usize] = mem::size_of::<c_short>();
    sizes[TSHORT as usize] = mem::size_of::<c_short>();
    sizes[TINT as usize] = mem::size_of::<c_int>();
    sizes[TUINT as usize] = mem::size_of::<c_int>();
    sizes[TLONG as usize] = mem::size_of::<c_long>();
    sizes[TULONG as usize] = mem::size_of::<c_long>();
    sizes[TLONGLONG as usize] = mem::size_of::<LONGLONG>();
    sizes[TULONGLONG as usize] = mem::size_of::<LONGLONG>();
    sizes[TFLOAT as usize] = mem::size_of::<f32>();
    sizes[TDOUBLE as usize] = mem::size_of::<f64>();
    sizes[TDBLCOMPLEX as usize] = 2 * mem::size_of::<f64>();

    if *status > 0 {
        return *status;
    }

    if ncols <= 0 {
        *status = 0;
        return *status;
    }

    let mut repeats: Vec<LONGLONG> = vec![0; ncols as usize];

    ffgnrwll_safe(fptr, &mut ntotrows, status);
    ffgrsz_safe(fptr, &mut nrowbuf, status);

    /* Retrieve column repeats */
    let mut icol: usize = 0;
    while (icol < ncols as usize) && (icol < 1000) {
        let mut typecode = 0;
        let mut repeat: LONGLONG = 0;
        let mut width: LONGLONG = 0;
        ffgtclll_safe(
            fptr,
            colnum[icol],
            Some(&mut typecode),
            Some(&mut repeat),
            Some(&mut width),
            status,
        );
        repeats[icol] = repeat;

        if datatype[icol] == TBIT
            || datatype[icol] == TSTRING
            || sizes[datatype[icol] as usize] == 0
        {
            ffpmsg_str("Cannot read from TBIT or TSTRING datatypes (ffgcvn)");
            *status = BAD_DATATYPE;
        }
        if typecode < 0 {
            ffpmsg_str("Cannot read from variable-length data (ffgcvn)");
            *status = BAD_DIMEN;
        }

        if *status > 0 {
            break;
        }
        icol += 1;
    }

    if *status > 0 {
        return *status;
    }

    /* Optimize for 1 column */
    if ncols == 1 {
        let bytes = match bytes_per_datatype(datatype[0]) {
            Some(x) => x * nrows as usize * repeats[0] as usize,
            None => {
                *status = BAD_DATATYPE;
                return *status;
            }
        };

        let arr = unsafe {
            slice::from_raw_parts_mut(array[0].cast::<u8>(), bytes * (nrows * repeats[0]) as usize)
        };

        ffgcv_safe(
            fptr,
            datatype[0],
            colnum[0],
            firstrow,
            1,
            nrows * repeats[0],
            nulval[0].clone(),
            arr,
            anynul.map(|x| &mut x[0]),
            status,
        );

        return *status;
    }

    /* Scan through file, in chunks of nrowbuf */
    currow = firstrow;
    ndone = 0;
    while ndone < nrows {
        nread = nrows - ndone; /* Number of rows to read (not elements) */
        if nread > nrowbuf as LONGLONG {
            nread = nrowbuf as LONGLONG;
        }

        for icol in 0..(ncols as usize) {
            let nelem1: LONGLONG = nread * repeats[icol];

            /* bytes_per_datatype gives the size of one element, so this is
            already the whole of the caller's buffer for this column: one
            element for each of the nrows rows times the column's repeat.
            Scaling it by nrows * repeats a second time claimed a buffer
            (nrows * repeats) times larger than the caller allocated. */
            let bytes = match bytes_per_datatype(datatype[icol]) {
                Some(x) => x * nrows as usize * repeats[icol] as usize,
                None => {
                    *status = BAD_DATATYPE;
                    return *status;
                }
            };

            let arr = unsafe { slice::from_raw_parts_mut(array[icol].cast::<u8>(), bytes) };
            let array1 =
                &mut arr[(repeats[icol] * ndone) as usize * (sizes[datatype[icol] as usize])..];

            ffgcv_safe(
                fptr,
                datatype[icol],
                colnum[icol],
                currow,
                1,
                nelem1,
                nulval[icol].clone(),
                array1,
                anynul.as_deref_mut().map(|x| &mut x[icol]),
                status,
            );

            if *status > 0 {
                let mut errmsg: [c_char; 100] = [0; 100];
                int_snprintf!(
                    &mut errmsg,
                    errmsg.len(),
                    "Failed to read column {} data rows {}-{} (ffgcvn)",
                    colnum[icol],
                    currow,
                    currow + nread - 1,
                );
                ffpmsg_slice(&errmsg);
                break;
            }
        }

        if *status > 0 {
            break;
        }
        currow += nread;
        ndone += nread;
    }

    *status
}

/// Read an array of values from a table column. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// ANYNUL is returned with a value of true if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the value
/// * `colnum`    — (I) number of column to write (1 = 1st col)
/// * `firstrow`  — (I) first row to write (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) array of null value flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcf(
    fptr: *mut fitsfile,
    datatype: c_int,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut c_void,
    nullarray: *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let anynul = anynul.as_mut();

        let nullarray = slice::from_raw_parts_mut(nullarray, nelem as usize);

        let bytes = match bytes_per_datatype(datatype) {
            Some(x) => x * nelem as usize,
            None => {
                *status = BAD_DATATYPE;
                return *status;
            }
        };

        let array: &mut [u8] = slice::from_raw_parts_mut(array.cast::<u8>(), bytes);

        ffgcf_safe(
            fptr,
            datatype,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            array,
            nullarray,
            anynul,
            status,
        )
    }
}

/// Read an array of values from a table column. The datatype of the
/// input array is defined by the 2nd argument.  Data conversion
/// and scaling will be performed if necessary (e.g, if the datatype of
/// the FITS array is not the same as the array being read).
/// ANYNUL is returned with a value of true if any pixels are undefined.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the value
/// * `colnum`    — (I) number of column to write (1 = 1st col)
/// * `firstrow`  — (I) first row to write (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of values to read
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) array of null value flags
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgcf_safe(
    fptr: &mut fitsfile,
    datatype: c_int,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut [u8],
    nullarray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let nulval: f64 = 0.0;
    let cnulval: [c_char; 2] = [0; 2];

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if datatype == TBIT {
        let array = cast_slice_mut(array);

        ffgcx_safe(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            array,
            status,
        );
    } else if datatype == TBYTE {
        let array = cast_slice_mut(array);

        ffgclb(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as u8,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TSBYTE {
        let array = cast_slice_mut(array);

        ffgclsb(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as i8,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TUSHORT {
        let array = cast_slice_mut(array);

        ffgclui(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as c_ushort,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TSHORT {
        let array = cast_slice_mut(array);

        ffgcli(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as c_short,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TUINT {
        let array = cast_slice_mut(array);

        ffgcluk(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as c_uint,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TINT {
        let array = cast_slice_mut(array);

        ffgclk(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as c_int,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TULONG {
        let array = cast_slice_mut(array);

        ffgcluj(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as c_ulong,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TLONG {
        let array = cast_slice_mut(array);
        ffgclj(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as c_long,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TULONGLONG {
        let array = cast_slice_mut(array);

        ffgclujj(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as ULONGLONG,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TLONGLONG {
        let array = cast_slice_mut(array);
        ffgcljj(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as LONGLONG,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TFLOAT {
        let array = cast_slice_mut(array);
        ffgcle(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval as f32,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TDOUBLE {
        let array = cast_slice_mut(array);

        ffgcld(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            NullCheckType::SetNullArray,
            nulval,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TCOMPLEX {
        let array = cast_slice_mut(array);

        ffgcfc_safe(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TDBLCOMPLEX {
        let array = cast_slice_mut(array);

        ffgcfm_safe(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TLOGICAL {
        let array = cast_slice_mut(array);

        ffgcll(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            NullCheckType::SetNullArray,
            nulval as c_char,
            array,
            nullarray,
            anynul,
            status,
        );
    } else if datatype == TSTRING {
        let (nstrings, item_len) = match tstring_item_len(fptr, colnum, firstrow, nelem, status) {
            Some(v) => v,
            None => return *status,
        };

        // SAFETY: for TSTRING the caller's buffer holds an array of `char *`,
        // as in the C, so it carries pointer alignment even though the
        // signature types it as bytes (clippy's cast_ptr_alignment); assert
        // that rather than assuming it.
        unsafe {
            #[allow(clippy::cast_ptr_alignment)]
            let p = array.as_mut_ptr().cast::<*mut c_char>();
            debug_assert!(p.is_aligned());
            let array = slice::from_raw_parts_mut(p, nstrings);
            let mut v_array = Vec::new();
            for item in array {
                let array_item = slice::from_raw_parts_mut(*item, item_len);
                v_array.push(array_item);
            }

            ffgcls(
                fptr,
                colnum,
                firstrow,
                firstelem,
                nelem,
                NullCheckType::SetNullArray,
                Some(&cnulval),
                &mut v_array,
                nullarray,
                anynul,
                status,
            );
        }
    } else {
        *status = BAD_DATATYPE;
    }

    *status
}

#[cfg(test)]
mod tests {
    use crate::NullValue;
    use crate::aliases::rust_api::*;
    use crate::fitsio::{
        BAD_DATATYPE, BINARY_TBL, BYTE_IMG, DOUBLE_IMG, FLOAT_IMG, LONG_IMG, LONGLONG,
        LONGLONG_IMG, READONLY, SBYTE_IMG, SHORT_IMG, TBIT, TBYTE, TCOMPLEX, TDBLCOMPLEX, TDOUBLE,
        TFLOAT, TINT, TLOGICAL, TLONG, TLONGLONG, TSBYTE, TSHORT, TSTRING, TUINT, TULONG, TUSHORT,
        ULONG_IMG, USHORT_IMG, fitsfile,
    };
    use crate::helpers::testhelpers::{to_buf, with_temp_file};
    use bytemuck::cast_slice_mut;
    use libc::{c_char, c_int, c_long, c_void};

    /// Make a NUL-terminated `Vec<c_char>` from a `&str`.
    fn cc(s: &str) -> Vec<c_char> {
        let mut v: Vec<c_char> = s.bytes().map(|b| b as c_char).collect();
        v.push(0);
        v
    }

    /// Create a single-column binary table and return the open file.
    fn make_table(
        name: &[c_char],
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
            BINARY_TBL,
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
    // ffgpv - read from primary array with various datatypes
    // ----------------------------------------------------------------------

    #[test]
    fn test_ffgpv_short() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [6];
            let wdata: [i16; 6] = [-32768, -100, 0, 100, 1000, 32767];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 6, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 6];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TSHORT,
                1,
                6,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-32768, -100, 0, 100, 1000, 32767]);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_long() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [c_long; 4] = [-2000000000, -100, 100, 2000000000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), LONG_IMG, 1, &naxes, &mut status);
            fits_write_img_lng(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0 as c_long; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TLONG,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-2000000000, -100, 100, 2000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_float() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [f32; 4] = [-1.5, 0.0, 1.5, 3.14159];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), FLOAT_IMG, 1, &naxes, &mut status);
            fits_write_img_flt(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0f32; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TFLOAT,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - (-1.5)).abs() < 1e-6);
            assert!((rdata[1] - 0.0).abs() < 1e-6);
            assert!((rdata[2] - 1.5).abs() < 1e-6);
            assert!((rdata[3] - 3.14159).abs() < 1e-5);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_double() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [f64; 4] = [-1.5, 0.0, 1.5, 3.14159265358979];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0f64; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TDOUBLE,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - (-1.5)).abs() < 1e-10);
            assert!((rdata[1] - 0.0).abs() < 1e-10);
            assert!((rdata[2] - 1.5).abs() < 1e-10);
            assert!((rdata[3] - 3.14159265358979).abs() < 1e-10);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_byte() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [u8; 4] = [0, 100, 200, 255];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            fits_write_img_byt(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0u8; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TBYTE,
                1,
                4,
                None,
                &mut rdata,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 100, 200, 255]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_sbyte() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [i8; 4] = [-128, -1, 0, 127];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SBYTE_IMG, 1, &naxes, &mut status);
            fits_write_img_sbyt(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i8; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TSBYTE,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-128, -1, 0, 127]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_ushort() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [u16; 4] = [0, 100, 30000, 65535];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                USHORT_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_usht(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0u16; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TUSHORT,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 100, 30000, 65535]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_uint() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [u32; 4] = [0, 100, 3000000000, 4000000000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 1, &naxes, &mut status);
            fits_write_img_uint(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0u32; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TUINT,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 100, 3000000000, 4000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_int() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [c_int; 4] = [-2000000000, -100, 100, 2000000000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), LONG_IMG, 1, &naxes, &mut status);
            fits_write_img_int(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0 as c_int; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TINT,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-2000000000, -100, 100, 2000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_ulong() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [libc::c_ulong; 4] = [0, 100, 3000000000, 4000000000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), ULONG_IMG, 1, &naxes, &mut status);
            fits_write_img_ulng(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0 as libc::c_ulong; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TULONG,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 100, 3000000000, 4000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_longlong() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [LONGLONG; 4] = [-9000000000000, -100, 100, 9000000000000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                LONGLONG_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_lnglng(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0 as LONGLONG; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TLONGLONG,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-9000000000000, -100, 100, 9000000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpv_with_nulval() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let wdata: [i16; 5] = [100, -32768, 200, -32768, 300];
            let nulval: LONGLONG = -32768;

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            // For images, write BLANK keyword to define null value
            fits_update_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("BLANK"),
                nulval,
                Some(&cc("null value")),
                &mut status,
            );
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 5, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 5];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TSHORT,
                1,
                5,
                Some(NullValue::Short(-999)),
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [100, -999, 200, -999, 300]);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ----------------------------------------------------------------------
    // ffgpf - read with null flagging
    // ----------------------------------------------------------------------

    #[test]
    fn test_ffgpf_short() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let wdata: [i16; 5] = [100, -32768, 200, -32768, 300];
            let nulval: LONGLONG = -32768;

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_update_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("BLANK"),
                nulval,
                Some(&cc("null value")),
                &mut status,
            );
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 5, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 5];
            let mut nullarray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_imgnull(
                f.as_deref_mut().unwrap(),
                TSHORT,
                1,
                5,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 100);
            assert_eq!(rdata[2], 200);
            assert_eq!(rdata[4], 300);
            assert_eq!(nullarray, [0, 1, 0, 1, 0]);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpf_float() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), FLOAT_IMG, 1, &naxes, &mut status);
            fits_write_img_flt(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0f32; 4];
            let mut nullarray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_imgnull(
                f.as_deref_mut().unwrap(),
                TFLOAT,
                1,
                4,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 1.0).abs() < 1e-6);
            assert!((rdata[1] - 2.0).abs() < 1e-6);
            assert_eq!(nullarray[0], 0);
            assert_eq!(nullarray[1], 0);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpf_double() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                DOUBLE_IMG,
                1,
                &naxes,
                &mut status,
            );
            fits_write_img_dbl(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0f64; 4];
            let mut nullarray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_imgnull(
                f.as_deref_mut().unwrap(),
                TDOUBLE,
                1,
                4,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 1.0).abs() < 1e-10);
            assert!((rdata[3] - 4.0).abs() < 1e-10);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpf_byte() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [u8; 4] = [0, 100, 200, 255];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            fits_write_img_byt(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0u8; 4];
            let mut nullarray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_imgnull(
                f.as_deref_mut().unwrap(),
                TBYTE,
                1,
                4,
                &mut rdata,
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 0);
            assert_eq!(rdata[3], 255);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpf_long() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [c_long; 4] = [-1000000, 0, 1000000, 2000000000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), LONG_IMG, 1, &naxes, &mut status);
            fits_write_img_lng(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0 as c_long; 4];
            let mut nullarray = [0 as c_char; 4];
            let mut anynull = -1;
            fits_read_imgnull(
                f.as_deref_mut().unwrap(),
                TLONG,
                1,
                4,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], -1000000);
            assert_eq!(rdata[3], 2000000000);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ----------------------------------------------------------------------
    // ffgpxv - read pixels by coordinates
    // ----------------------------------------------------------------------

    #[test]
    fn test_ffgpxv_1d() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];
            let wdata: [i16; 10] = core::array::from_fn(|i| (i * 10) as i16);
            let firstpix: [c_long; 1] = [3];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 10, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 5];
            let mut anynull = -1;
            fits_read_pix(
                f.as_deref_mut().unwrap(),
                TSHORT,
                &firstpix,
                5,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [20, 30, 40, 50, 60]); // pixel 3 = index 2
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpxv_2d() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let wdata: [i16; 16] = core::array::from_fn(|i| (i + 1) as i16);
            let firstpix: [c_long; 2] = [2, 2];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_write_2d_sht(f.as_deref_mut().unwrap(), 1, 4, 4, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 4];
            let mut anynull = -1;
            fits_read_pix(
                f.as_deref_mut().unwrap(),
                TSHORT,
                &firstpix,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Starting at (2,2), reading 4 elements: (2,2),(3,2),(4,2),(1,3)
            assert_eq!(rdata, [6, 7, 8, 9]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpxvll_1d() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];
            let wdata: [i16; 10] = core::array::from_fn(|i| (i * 100) as i16);
            let firstpix: [LONGLONG; 1] = [5];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 10, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 3];
            let mut anynull = -1;
            fits_read_pixll(
                f.as_deref_mut().unwrap(),
                TSHORT,
                &firstpix,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [400, 500, 600]); // pixel 5 = index 4
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpxf_1d() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let wdata: [i16; 5] = [100, -32768, 200, -32768, 300];
            let nulval: LONGLONG = -32768;
            let firstpix: [c_long; 1] = [1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_update_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("BLANK"),
                nulval,
                Some(&cc("null value")),
                &mut status,
            );
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 5, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 5];
            let mut nullarray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_pixnull(
                f.as_deref_mut().unwrap(),
                TSHORT,
                &firstpix,
                5,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 100);
            assert_eq!(rdata[2], 200);
            assert_eq!(rdata[4], 300);
            assert_eq!(nullarray[1], 1);
            assert_eq!(nullarray[3], 1);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgpxfll_1d() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];
            let wdata: [i16; 5] = [100, -32768, 200, -32768, 300];
            let nulval: LONGLONG = -32768;
            let firstpix: [LONGLONG; 1] = [1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_update_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("BLANK"),
                nulval,
                Some(&cc("null value")),
                &mut status,
            );
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 5, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 5];
            let mut nullarray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_pixnullll(
                f.as_deref_mut().unwrap(),
                TSHORT,
                &firstpix,
                5,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 100);
            assert_eq!(nullarray[1], 1);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ----------------------------------------------------------------------
    // ffgsv - read subsection with strides
    // ----------------------------------------------------------------------

    #[test]
    fn test_ffgsv_whole_image() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let wdata: [i16; 16] = core::array::from_fn(|i| (i + 1) as i16);
            let blc: [c_long; 2] = [1, 1];
            let trc: [c_long; 2] = [4, 4];
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_write_2d_sht(f.as_deref_mut().unwrap(), 1, 4, 4, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 16];
            let mut anynull = -1;
            fits_read_subset(
                f.as_deref_mut().unwrap(),
                TSHORT,
                &blc,
                &trc,
                &inc,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 1);
            assert_eq!(rdata[15], 16);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgsv_subsection() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let wdata: [i16; 16] = core::array::from_fn(|i| (i + 1) as i16);
            let blc: [c_long; 2] = [2, 2];
            let trc: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_write_2d_sht(f.as_deref_mut().unwrap(), 1, 4, 4, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 4];
            let mut anynull = -1;
            fits_read_subset(
                f.as_deref_mut().unwrap(),
                TSHORT,
                &blc,
                &trc,
                &inc,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Reading 2x2 subsection starting at (2,2)
            assert_eq!(rdata, [6, 7, 10, 11]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgsv_with_stride() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [6, 6];
            let wdata: [i16; 36] = core::array::from_fn(|i| (i + 1) as i16);
            let blc: [c_long; 2] = [1, 1];
            let trc: [c_long; 2] = [6, 6];
            let inc: [c_long; 2] = [2, 2];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_write_2d_sht(f.as_deref_mut().unwrap(), 1, 6, 6, 6, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 9];
            let mut anynull = -1;
            fits_read_subset(
                f.as_deref_mut().unwrap(),
                TSHORT,
                &blc,
                &trc,
                &inc,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Reading every other pixel: (1,1),(3,1),(5,1),(1,3),...
            assert_eq!(rdata[0], 1);
            assert_eq!(rdata[1], 3);
            assert_eq!(rdata[2], 5);
            assert_eq!(rdata[3], 13);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgsv_float() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let wdata: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32 * 0.5);
            let blc: [c_long; 2] = [2, 2];
            let trc: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), FLOAT_IMG, 2, &naxes, &mut status);
            fits_write_2d_flt(f.as_deref_mut().unwrap(), 1, 4, 4, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0f32; 4];
            let mut anynull = -1;
            fits_read_subset(
                f.as_deref_mut().unwrap(),
                TFLOAT,
                &blc,
                &trc,
                &inc,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 3.0).abs() < 1e-6); // (2,2) = index 5 -> 6*0.5
            assert!((rdata[1] - 3.5).abs() < 1e-6);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgsv_double() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let wdata: [f64; 16] = core::array::from_fn(|i| (i + 1) as f64 * 0.1);
            let blc: [c_long; 2] = [2, 2];
            let trc: [c_long; 2] = [3, 3];
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
            fits_write_2d_dbl(f.as_deref_mut().unwrap(), 1, 4, 4, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0f64; 4];
            let mut anynull = -1;
            fits_read_subset(
                f.as_deref_mut().unwrap(),
                TDOUBLE,
                &blc,
                &trc,
                &inc,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 0.6).abs() < 1e-10); // (2,2) = index 5 -> 6*0.1
            assert!((rdata[1] - 0.7).abs() < 1e-10);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgsv_byte() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let wdata: [u8; 16] = core::array::from_fn(|i| ((i + 1) * 10) as u8);
            let blc: [c_long; 2] = [2, 2];
            let trc: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 2, &naxes, &mut status);
            fits_write_2d_byt(f.as_deref_mut().unwrap(), 1, 4, 4, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0u8; 4];
            let mut anynull = -1;
            fits_read_subset(
                f.as_deref_mut().unwrap(),
                TBYTE,
                &blc,
                &trc,
                &inc,
                None,
                &mut rdata,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 60); // (2,2) = index 5 -> 6*10
            assert_eq!(rdata[1], 70);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgsv_long() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let wdata: [c_long; 16] = core::array::from_fn(|i| ((i + 1) * 1000) as c_long);
            let blc: [c_long; 2] = [2, 2];
            let trc: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), LONG_IMG, 2, &naxes, &mut status);
            fits_write_2d_lng(f.as_deref_mut().unwrap(), 1, 4, 4, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0 as c_long; 4];
            let mut anynull = -1;
            fits_read_subset(
                f.as_deref_mut().unwrap(),
                TLONG,
                &blc,
                &trc,
                &inc,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 6000); // (2,2) = index 5 -> 6*1000
            assert_eq!(rdata[1], 7000);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgsv_longlong() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 4];
            let wdata: [LONGLONG; 16] =
                core::array::from_fn(|i| ((i + 1) as LONGLONG) * 1000000000);
            let blc: [c_long; 2] = [2, 2];
            let trc: [c_long; 2] = [3, 3];
            let inc: [c_long; 2] = [1, 1];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(
                f.as_deref_mut().unwrap(),
                LONGLONG_IMG,
                2,
                &naxes,
                &mut status,
            );
            fits_write_2d_lnglng(f.as_deref_mut().unwrap(), 1, 4, 4, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0 as LONGLONG; 4];
            let mut anynull = -1;
            fits_read_subset(
                f.as_deref_mut().unwrap(),
                TLONGLONG,
                &blc,
                &trc,
                &inc,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 6000000000);
            assert_eq!(rdata[1], 7000000000);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ----------------------------------------------------------------------
    // ffgcv - read table column
    // ----------------------------------------------------------------------

    #[test]
    fn test_ffgcv_short() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [i16; 3] = [-32768, 0, 32767];

            let mut f = make_table(&name, "ICOL", "1I", 3, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0i16; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TSHORT,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-32768, 0, 32767]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_long() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [c_long; 3] = [-2000000000, 0, 2000000000];

            let mut f = make_table(&name, "JCOL", "1J", 3, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as c_long; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TLONG,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-2000000000, 0, 2000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_float() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [f32; 3] = [-1.5, 0.0, 3.14159];

            let mut f = make_table(&name, "ECOL", "1E", 3, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0f32; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TFLOAT,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - (-1.5)).abs() < 1e-6);
            assert!((rdata[1] - 0.0).abs() < 1e-6);
            assert!((rdata[2] - 3.14159).abs() < 1e-5);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_double() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [f64; 3] = [-1.5, 0.0, 3.14159265358979];

            let mut f = make_table(&name, "DCOL", "1D", 3, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0f64; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TDOUBLE,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - (-1.5)).abs() < 1e-10);
            assert!((rdata[1] - 0.0).abs() < 1e-10);
            assert!((rdata[2] - 3.14159265358979).abs() < 1e-10);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_byte() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [u8; 3] = [0, 127, 255];

            let mut f = make_table(&name, "BCOL", "1B", 3, &mut status);
            fits_write_col_byt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0u8; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TBYTE,
                1,
                1,
                1,
                3,
                None,
                &mut rdata,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 127, 255]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_sbyte() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [i8; 3] = [-128, 0, 127];

            let mut f = make_table(&name, "SBCOL", "1S", 3, &mut status);
            fits_write_col_sbyt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0i8; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TSBYTE,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-128, 0, 127]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_ushort() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [u16; 3] = [0, 30000, 65535];

            let mut f = make_table(&name, "UICOL", "1U", 3, &mut status);
            fits_write_col_usht(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0u16; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TUSHORT,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 30000, 65535]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_uint() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [u32; 3] = [0, 2000000000, 4000000000];

            let mut f = make_table(&name, "UKCOL", "1V", 3, &mut status);
            fits_write_col_uint(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0u32; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TUINT,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 2000000000, 4000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_int() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [c_int; 3] = [-2000000000, 0, 2000000000];

            let mut f = make_table(&name, "KCOL", "1J", 3, &mut status);
            fits_write_col_int(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as c_int; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TINT,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-2000000000, 0, 2000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_ulong() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [libc::c_ulong; 3] = [0, 2000000000, 4000000000];

            let mut f = make_table(&name, "UJCOL", "1V", 3, &mut status);
            fits_write_col_ulng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as libc::c_ulong; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TULONG,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 2000000000, 4000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_longlong() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [LONGLONG; 3] = [-9000000000000, 0, 9000000000000];

            let mut f = make_table(&name, "KCOL", "1K", 3, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as LONGLONG; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TLONGLONG,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [-9000000000000, 0, 9000000000000]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_logical() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [c_char; 3] = [0, 1, 1];

            let mut f = make_table(&name, "LCOL", "1L", 3, &mut status);
            fits_write_col_log(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TLOGICAL,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 1, 1]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata = ["hello", "world", "test"];

            let mut f = make_table(&name, "ACOL", "10A", 3, &mut status);
            let w_v: Vec<Vec<c_char>> = wdata.iter().map(|s| cc(s)).collect();
            let w_ref: Vec<&[c_char]> = w_v.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &w_ref, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            // Generic ffgcv with TSTRING: the byte buffer holds an array of
            // `*mut c_char` pointers to per-element string buffers.
            let mut buf: [[c_char; 11]; 3] = [[0; 11]; 3];
            let mut ptrs: [*mut c_char; 3] = core::array::from_fn(|i| buf[i].as_mut_ptr());
            let ptr_bytes: &mut [u8] = unsafe {
                core::slice::from_raw_parts_mut(
                    ptrs.as_mut_ptr().cast::<u8>(),
                    core::mem::size_of_val(&ptrs),
                )
            };
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TSTRING,
                1,
                1,
                1,
                3,
                None,
                ptr_bytes,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            let s0: Vec<u8> = buf[0]
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            let s1: Vec<u8> = buf[1]
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            let s2: Vec<u8> = buf[2]
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            assert_eq!(s0, b"hello");
            assert_eq!(s1, b"world");
            assert_eq!(s2, b"test");
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    /// Write `content` to row 1 of a `1PA` column and read it back through the
    /// generic TSTRING readers into a buffer sized from the row's descriptor,
    /// as CFITSIO callers do. ffgcdw reports a width of 1 for `1PA`, so the
    /// slice must be sized from the descriptor, not the display width. Callers
    /// such as python-fitsio pass `nelem` = the string length with a single
    /// `char *`, which CFITSIO allows since it reads one string regardless.
    fn read_var_string_generic(use_ffgcf: bool, nelem: LONGLONG) {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let content = "hello world";

            let mut f = make_table(&name, "VARSTR", "1PA", 1, &mut status);
            let w_v = [cc(content)];
            let w_ref: Vec<&[c_char]> = w_v.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &w_ref, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut buf: Vec<c_char> = vec![0; content.len() + 1];
            let mut ptrs: [*mut c_char; 1] = [buf.as_mut_ptr()];
            let ptr_bytes: &mut [u8] = unsafe {
                core::slice::from_raw_parts_mut(
                    ptrs.as_mut_ptr().cast::<u8>(),
                    core::mem::size_of_val(&ptrs),
                )
            };
            let mut anynull = -1;
            if use_ffgcf {
                let mut nullarray: [c_char; 1] = [0];
                fits_read_colnull(
                    f.as_deref_mut().unwrap(),
                    TSTRING,
                    1,
                    1,
                    1,
                    nelem,
                    ptr_bytes,
                    &mut nullarray,
                    Some(&mut anynull),
                    &mut status,
                );
            } else {
                fits_read_col(
                    f.as_deref_mut().unwrap(),
                    TSTRING,
                    1,
                    1,
                    1,
                    nelem,
                    None,
                    ptr_bytes,
                    Some(&mut anynull),
                    &mut status,
                );
            }
            assert_eq!(status, 0);
            let s: Vec<u8> = buf
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            assert_eq!(s, content.as_bytes());
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_variable_length_string() {
        read_var_string_generic(false, 1);
        read_var_string_generic(false, 11);
    }

    #[test]
    fn test_ffgcf_variable_length_string() {
        read_var_string_generic(true, 1);
        read_var_string_generic(true, 11);
    }

    /// A `10A` column with a narrower `TDISP1 = 'A5'`: ffgcls2 writes the
    /// TFORM width, not the display width, so the caller's buffer (sized from
    /// TFORM, as CFITSIO requires) must be claimed in full.
    #[test]
    fn test_ffgcv_string_narrow_tdisp() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let content = "abcdefghij";

            let mut f = make_table(&name, "ACOL", "10A", 1, &mut status);
            fits_write_key_str(
                f.as_deref_mut().unwrap(),
                &cc("TDISP1"),
                &cc("A5"),
                None,
                &mut status,
            );
            let w_v = [cc(content)];
            let w_ref: Vec<&[c_char]> = w_v.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &w_ref, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut buf: Vec<c_char> = vec![0; content.len() + 1];
            let mut ptrs: [*mut c_char; 1] = [buf.as_mut_ptr()];
            let ptr_bytes: &mut [u8] = unsafe {
                core::slice::from_raw_parts_mut(
                    ptrs.as_mut_ptr().cast::<u8>(),
                    core::mem::size_of_val(&ptrs),
                )
            };
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TSTRING,
                1,
                1,
                1,
                1,
                None,
                ptr_bytes,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            let s: Vec<u8> = buf
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            assert_eq!(s, content.as_bytes());
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_complex() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 complex

            let mut f = make_table(&name, "CCOL", "1C", 3, &mut status);
            fits_write_col_cmp(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0f32; 6];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TCOMPLEX,
                1,
                1,
                1,
                3,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 1.0).abs() < 1e-6);
            assert!((rdata[1] - 2.0).abs() < 1e-6);
            assert!((rdata[2] - 3.0).abs() < 1e-6);
            assert!((rdata[3] - 4.0).abs() < 1e-6);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_dblcomplex() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [f64; 4] = [1.0, 2.0, 3.0, 4.0]; // 2 complex

            let mut f = make_table(&name, "MCOL", "1M", 2, &mut status);
            fits_write_col_dblcmp(f.as_deref_mut().unwrap(), 1, 1, 1, 2, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0f64; 4];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TDBLCOMPLEX,
                1,
                1,
                1,
                2,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 1.0).abs() < 1e-10);
            assert!((rdata[1] - 2.0).abs() < 1e-10);
            assert!((rdata[2] - 3.0).abs() < 1e-10);
            assert!((rdata[3] - 4.0).abs() < 1e-10);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_bit() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            // ffpclx expects each byte to be 0/non-0 representing one bit
            let wdata: [c_char; 8] = [1, 0, 1, 0, 1, 0, 1, 0];

            let mut f = make_table(&name, "XCOL", "8X", 1, &mut status);
            fits_write_col_bit(f.as_deref_mut().unwrap(), 1, 1, 1, 8, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as c_char; 8];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TBIT,
                1,
                1,
                1,
                8,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [1, 0, 1, 0, 1, 0, 1, 0]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcv_with_nulval() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [i16; 5] = [100, -32768, 200, -32768, 300];
            let nulval: LONGLONG = -32768;

            let mut f = make_table(&name, "ICOL", "5I", 1, &mut status);
            // Write TNULL keyword to define null value for column
            fits_update_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                nulval,
                Some(&cc("null value")),
                &mut status,
            );
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0i16; 5];
            let mut anynull = -1;
            fits_read_col(
                f.as_deref_mut().unwrap(),
                TSHORT,
                1,
                1,
                1,
                5,
                Some(NullValue::Short(-999)),
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [100, -999, 200, -999, 300]);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ----------------------------------------------------------------------
    // ffgcf - read column with null flagging
    // ----------------------------------------------------------------------

    #[test]
    fn test_ffgcf_short() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [i16; 5] = [100, -32768, 200, -32768, 300];
            let nulval: LONGLONG = -32768;

            let mut f = make_table(&name, "ICOL", "5I", 1, &mut status);
            fits_update_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("TNULL1"),
                nulval,
                Some(&cc("null value")),
                &mut status,
            );
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 5, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0i16; 5];
            let mut nullarray = [0 as c_char; 5];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TSHORT,
                1,
                1,
                1,
                5,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 100);
            assert_eq!(rdata[2], 200);
            assert_eq!(rdata[4], 300);
            assert_eq!(nullarray, [0, 1, 0, 1, 0]);
            assert_eq!(anynull, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_float() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [f32; 3] = [1.0, 2.0, 3.0];

            let mut f = make_table(&name, "ECOL", "3E", 1, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0f32; 3];
            let mut nullarray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TFLOAT,
                1,
                1,
                1,
                3,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 1.0).abs() < 1e-6);
            assert!((rdata[1] - 2.0).abs() < 1e-6);
            assert!((rdata[2] - 3.0).abs() < 1e-6);
            assert_eq!(nullarray[0], 0);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_double() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [f64; 3] = [1.0, 2.0, 3.0];

            let mut f = make_table(&name, "DCOL", "3D", 1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0f64; 3];
            let mut nullarray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TDOUBLE,
                1,
                1,
                1,
                3,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 1.0).abs() < 1e-10);
            assert!((rdata[2] - 3.0).abs() < 1e-10);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_byte() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [u8; 3] = [0, 127, 255];

            let mut f = make_table(&name, "BCOL", "3B", 1, &mut status);
            fits_write_col_byt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0u8; 3];
            let mut nullarray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TBYTE,
                1,
                1,
                1,
                3,
                &mut rdata,
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], 0);
            assert_eq!(rdata[2], 255);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_long() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [c_long; 3] = [-1000000, 0, 1000000];

            let mut f = make_table(&name, "JCOL", "3J", 1, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as c_long; 3];
            let mut nullarray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TLONG,
                1,
                1,
                1,
                3,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], -1000000);
            assert_eq!(rdata[2], 1000000);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_longlong() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [LONGLONG; 3] = [-9000000000000, 0, 9000000000000];

            let mut f = make_table(&name, "KCOL", "3K", 1, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as LONGLONG; 3];
            let mut nullarray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TLONGLONG,
                1,
                1,
                1,
                3,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata[0], -9000000000000);
            assert_eq!(rdata[2], 9000000000000);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_logical() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [c_char; 3] = [0, 1, 1];

            let mut f = make_table(&name, "LCOL", "3L", 1, &mut status);
            fits_write_col_log(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0 as c_char; 3];
            let mut nullarray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TLOGICAL,
                1,
                1,
                1,
                3,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(rdata, [0, 1, 1]);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata = ["hello", "world", "test"];

            let mut f = make_table(&name, "ACOL", "10A", 3, &mut status);
            let w_v: Vec<Vec<c_char>> = wdata.iter().map(|s| cc(s)).collect();
            let w_ref: Vec<&[c_char]> = w_v.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &w_ref, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut buf: [[c_char; 11]; 3] = [[0; 11]; 3];
            let mut ptrs: [*mut c_char; 3] = core::array::from_fn(|i| buf[i].as_mut_ptr());
            let ptr_bytes: &mut [u8] = unsafe {
                core::slice::from_raw_parts_mut(
                    ptrs.as_mut_ptr().cast::<u8>(),
                    core::mem::size_of_val(&ptrs),
                )
            };
            let mut nullarray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TSTRING,
                1,
                1,
                1,
                3,
                ptr_bytes,
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            let s0: Vec<u8> = buf[0]
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            let s1: Vec<u8> = buf[1]
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            let s2: Vec<u8> = buf[2]
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            assert_eq!(s0, b"hello");
            assert_eq!(s1, b"world");
            assert_eq!(s2, b"test");
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_complex() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; // 2 complex

            let mut f = make_table(&name, "CCOL", "2C", 1, &mut status);
            fits_write_col_cmp(f.as_deref_mut().unwrap(), 1, 1, 1, 2, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0f32; 4];
            let mut nullarray = [0 as c_char; 2];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TCOMPLEX,
                1,
                1,
                1,
                2,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 1.0).abs() < 1e-6);
            assert!((rdata[1] - 2.0).abs() < 1e-6);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcf_dblcomplex() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let wdata: [f64; 4] = [1.0, 2.0, 3.0, 4.0]; // 2 complex

            let mut f = make_table(&name, "MCOL", "2M", 1, &mut status);
            fits_write_col_dblcmp(f.as_deref_mut().unwrap(), 1, 1, 1, 2, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut rdata = [0f64; 4];
            let mut nullarray = [0 as c_char; 2];
            let mut anynull = -1;
            fits_read_colnull(
                f.as_deref_mut().unwrap(),
                TDBLCOMPLEX,
                1,
                1,
                1,
                2,
                cast_slice_mut(&mut rdata),
                &mut nullarray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - 1.0).abs() < 1e-10);
            assert!((rdata[1] - 2.0).abs() < 1e-10);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ----------------------------------------------------------------------
    // Error / conversion / multi-column
    // ----------------------------------------------------------------------

    #[test]
    fn test_bad_datatype() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [i16; 4] = [1, 2, 3, 4];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut rdata = [0i16; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                999,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, BAD_DATATYPE);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_data_conversion() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [4];
            let wdata: [i16; 4] = [-100, 0, 100, 1000];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 1, &naxes, &mut status);
            fits_write_img_sht(f.as_deref_mut().unwrap(), 1, 1, 4, &wdata, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            // Read short data as float
            let mut rdata = [0f32; 4];
            let mut anynull = -1;
            fits_read_img(
                f.as_deref_mut().unwrap(),
                TFLOAT,
                1,
                4,
                None,
                cast_slice_mut(&mut rdata),
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((rdata[0] - (-100.0)).abs() < 1e-6);
            assert!((rdata[1] - 0.0).abs() < 1e-6);
            assert!((rdata[2] - 100.0).abs() < 1e-6);
            assert!((rdata[3] - 1000.0).abs() < 1e-6);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcvn() {
        // Test ffgcvn - read from multiple columns at once.
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let col1_data: [c_long; 3] = [100, 200, 300];
            let col2_data: [f32; 3] = [1.5, 2.5, 3.5];
            let col3_data: [f64; 3] = [10.1, 20.2, 30.3];

            // Create table with data.
            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            let ttype_v = [Some(cc("COL1")), Some(cc("COL2")), Some(cc("COL3"))];
            let ttype_ref: Vec<Option<&[c_char]>> = ttype_v.iter().map(|o| o.as_deref()).collect();
            let tform_v = [cc("1J"), cc("1E"), cc("1D")];
            let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
            fits_create_tbl(
                f.as_deref_mut().unwrap(),
                BINARY_TBL,
                3,
                3,
                &ttype_ref,
                &tform_ref,
                None,
                Some(&cc("DATA")),
                &mut status,
            );
            fits_write_col_lng(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                &col1_data,
                &mut status,
            );
            fits_write_col_flt(
                f.as_deref_mut().unwrap(),
                2,
                1,
                1,
                3,
                &col2_data,
                &mut status,
            );
            fits_write_col_dbl(
                f.as_deref_mut().unwrap(),
                3,
                1,
                1,
                3,
                &col3_data,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            // Read all 3 columns at once.
            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut col1_result = [0 as c_long; 3];
            let mut col2_result = [0f32; 3];
            let mut col3_result = [0f64; 3];

            let datatypes: [c_int; 3] = [TLONG, TFLOAT, TDOUBLE];
            let colnums: [c_int; 3] = [1, 2, 3];
            let nulvals: [Option<NullValue>; 3] = [
                Some(NullValue::Long(0)),
                Some(NullValue::Float(0.0)),
                Some(NullValue::Double(0.0)),
            ];
            let mut arrays: [*mut c_void; 3] = [
                col1_result.as_mut_ptr().cast::<c_void>(),
                col2_result.as_mut_ptr().cast::<c_void>(),
                col3_result.as_mut_ptr().cast::<c_void>(),
            ];
            let mut anynul = [0 as c_int; 3];

            fits_read_cols(
                f.as_deref_mut().unwrap(),
                3,
                &datatypes,
                &colnums,
                1,
                3,
                &nulvals,
                &mut arrays,
                Some(&mut anynul),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(col1_result[0], 100);
            assert_eq!(col1_result[2], 300);
            assert!(col2_result[1] >= 2.4 && col2_result[1] <= 2.6);
            assert!(col3_result[2] >= 30.2 && col3_result[2] <= 30.4);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }
}
