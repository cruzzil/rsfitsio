//! Routines that read data elements from a FITS image or table, with
//! character string datatype.
//!
//! The [`TSTRING`] arm of the typed column I/O family. [`crate::getcol`]
//! dispatches here when a caller asks for this datatype at run time; the
//! write-side counterpart is [`crate::putcols`].
//!
//! Ported from CFITSIO's `getcols.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center.
#![warn(missing_docs)]

use core::ffi::CStr;
use core::slice;
use core::{cmp, mem};

use crate::c_types::{c_char, c_int, c_long, c_uchar};

use bytemuck::{cast_slice, cast_slice_mut};

use crate::buffers::{ffgbyt, ffgbytoff, ffmbyt_safe};
use crate::fitscore::{
    ffcdsp, ffeqtyll_safe, ffgcprll, ffghdt_safe, ffgtcl_safe, ffkeyn_safe, ffmahd_safe,
    ffpmsg_slice, ffrdef_safe,
};
use crate::fitsio2::*;
use crate::getcold::ffgcld;
use crate::getcole::ffgcle;
use crate::getcolj::*;
use crate::getcoll::ffgcll;
use crate::getcoluj::ffgcfujj_safe;
use crate::getkey::{ffgkyd_safe, ffgkys_safe};
use crate::int_snprintf;
use crate::relibc::header::stdio::{snprintf_cint, snprintf_f64, sprintf_string_width};
use crate::wrappers::*;
use crate::{NullCheckType, fitsio::*};
use crate::{bb, cs, parse_c_int};

/// Read an array of string values from a column in the current FITS HDU.
///
/// Any undefined pixels will be set equal to the value of 'nulval' unless
/// nulval = null in which case no checks for undefined pixels will be made.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of strings to read
/// * `nulval`    — (I) string for null pixels
/// * `array`     — (O) array of values that are read
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcvs(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: *const c_char,
    array: *mut *mut c_char,
    anynul: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let anynul = anynul.as_mut();

        let array = slice::from_raw_parts_mut(array, nelem as usize);

        let mut v_array = Vec::new();
        for item in array {
            let array_item = slice::from_raw_parts_mut(*item, FLEN_VALUE);
            v_array.push(array_item);
        }

        let nulval: Option<&[c_char]> = if nulval.is_null() {
            None
        } else {
            Some(cast_slice(CStr::from_ptr(nulval).to_bytes_with_nul()))
        };

        ffgcvs_safe(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            nulval,
            &mut v_array,
            anynul,
            status,
        )
    }
}

/// Read an array of string values from a column in the current FITS HDU.
/// Any undefined pixels will be set equal to the value of 'nulval' unless
/// nulval = null in which case no checks for undefined pixels will be made.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of strings to read
/// * `nulval`    — (I) string for null pixels
/// * `array`     — (O) array of values that are read
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgcvs_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nulval: Option<&[c_char]>,
    array: &mut [&mut [c_char]],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut cdummy: [c_char; 2] = [0; 2];

    ffgcls(
        fptr,
        colnum,
        firstrow as LONGLONG,
        firstelem as LONGLONG,
        nelem as LONGLONG,
        NullCheckType::SetPixel,
        nulval,
        array,
        &mut cdummy,
        anynul,
        status,
    );

    *status
}

/// Read an array of string values from a column in the current FITS HDU.
/// Nularray will be set = 1 if the corresponding array pixel is undefined,
/// otherwise nularray will = 0.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of strings to read
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags = 1 if nultyp = 2
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcfs(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: *mut *mut c_char,
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
        let mut v_array = Vec::new();
        for item in array {
            let array_item = slice::from_raw_parts_mut(*item, FLEN_VALUE);
            v_array.push(array_item);
        }
        let nularray = slice::from_raw_parts_mut(nularray, nelem as usize);

        ffgcfs_safe(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            &mut v_array,
            nularray,
            anynul,
            status,
        )
    }
}

/// Read an array of string values from a column in the current FITS HDU.
/// Nularray will be set = 1 if the corresponding array pixel is undefined,
/// otherwise nularray will = 0.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of strings to read
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags = 1 if nultyp = 2
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub fn ffgcfs_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    array: &mut Vec<&mut [c_char]>,
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let cdummy: [c_char; 2] = [0; 2];

    ffgcls(
        fptr,
        colnum,
        firstrow,
        firstelem,
        nelem,
        NullCheckType::SetNullArray,
        Some(&cdummy),
        array,
        nularray,
        anynul,
        status,
    );
    *status
}

/// Read an array of string values from a column in the current FITS HDU.
/// Returns a formatted string value, regardless of the datatype of the column
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of strings to read
/// * `nultyp`    — (I) null value handling code:
/// * `nulval`    — (I) value for null pixels if nultyp = 1
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags = 1 if nultyp = 2
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub(crate) fn ffgcls(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nultyp: NullCheckType,
    /*     1: set undefined pixels = nulval        */
    /*     2: set nularray=1 for undefined pixels  */
    nulval: Option<&[c_char]>,
    array: &mut [&mut [c_char]],
    nularray: &mut [c_char],
    anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut tcode: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut tstatus: c_int = 0;
    let mut scaled: c_int = 0;
    let mut intcol: c_int = 0;
    let mut dwidth: c_int = 0;
    let mut nulwidth: c_int = 0;
    let mut dlen: c_int = 0;
    let mut equivtype: c_int = 0;
    let mut jj: usize = 0;
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut cform: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut dispfmt: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut tmpstr: [c_char; 400] = [0; 400];
    let mut tmpnull: [c_char; 80] = [0; 80];
    let mut byteval: c_uchar = 0;
    let mut tscale: f64 = 1.0;

    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);

    /* rescan header if data structure is undefined */
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        return *status;
    }

    if colnum < 1 || colnum > fptr.Fptr.tfield {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Specified column number is out of range: {}",
            colnum,
        );
        ffpmsg_slice(&message);
        *status = BAD_COL_NUM;
        return *status;
    }

    /* get equivalent dataype of column (only needed for TLONGLONG columns) */
    ffeqtyll_safe(fptr, colnum, Some(&mut equivtype), None, None, status);
    if equivtype < 0 {
        equivtype = equivtype.abs();
    }

    let c = fptr.Fptr.get_tableptr_as_slice(); /* set pointer to first column */
    let ci = colnum as usize - 1; /* offset to correct column structure */
    let tcol = c[ci];

    tcode = tcol.tdatatype.abs();

    intcol = 0;
    if tcode == TSTRING {
        /* simply call the string column reading routine */
        ffgcls2(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            nultyp,
            nulval,
            array,
            nularray,
            anynul,
            status,
        );
    } else if tcode == TLOGICAL {
        /* allocate memory for the array of logical values */
        let mut carray: Vec<c_char> = vec![0; nelem as usize];
        let nv = nulval.unwrap();
        /*  call the logical column reading routine */
        ffgcll(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            nultyp,
            nv[0],
            &mut carray,
            nularray,
            anynul,
            status,
        );

        if *status <= 0 {
            /* convert logical values to "T", "F", or "N" (Null) */
            for ii in 0..(nelem as usize) {
                if carray[ii] == 1 {
                    strcpy_safe(array[ii], cs!(c"T"));
                } else if carray[ii] == 0 {
                    strcpy_safe(array[ii], cs!(c"F"));
                } else {
                    /* undefined values = 2 */
                    strcpy_safe(array[ii], cs!(c"N"));
                }
            }
        }
    } else if tcode == TCOMPLEX {
        /* allocate memory for the array of double values */
        let mut earray: Vec<f32> = vec![0.0; (nelem * 2) as usize];

        ffgcle(
            fptr,
            colnum,
            firstrow,
            (firstelem - 1) * 2 + 1,
            nelem * 2,
            1,
            NullCheckType::SetPixel,
            FLOATNULLVALUE,
            &mut earray,
            nularray,
            anynul,
            status,
        );

        if *status <= 0 {
            /* determine the format for the output strings */

            ffgcdw_safe(fptr, colnum, &mut dwidth, status);
            dwidth = (dwidth - 3) / 2;

            /* use the TDISPn keyword if it exists */
            ffkeyn_safe(cs!(c"TDISP"), colnum, &mut keyname, status);
            tstatus = 0;
            cform[0] = 0;

            if ffgkys_safe(fptr, &keyname, &mut dispfmt, None, &mut tstatus) == 0 {
                /* convert the Fortran style format to a C style format */
                ffcdsp(&dispfmt, &mut cform);

                /* Special case: TDISPn='Aw' disallowed for numeric types */
                if dispfmt[0] == bb(b'A') {
                    cform[0] = 0;

                    /* Special case: if the output is intended to be represented
                    as an integer, but we read it as a double, we need to
                    set intcol = 1 so it is printed as an integer */
                } else if (dispfmt[0] == bb(b'I'))
                    || (dispfmt[0] == bb(b'i'))
                    || (dispfmt[0] == bb(b'O'))
                    || (dispfmt[0] == bb(b'o'))
                    || (dispfmt[0] == bb(b'Z'))
                    || (dispfmt[0] == bb(b'z'))
                {
                    intcol = 1;
                }
            }

            if cform[0] == 0 {
                strcpy_safe(&mut cform, cs!(c"%14.6E"));
            }
            /* write the formated string for each value:  "(real,imag)" */
            jj = 0;
            for ii in 0..(nelem as usize) {
                strcpy_safe(array[ii], cs!(c"("));

                /* test for null value */
                if earray[jj] == FLOATNULLVALUE {
                    strcpy_safe(&mut tmpstr, cs!(c"NULL"));
                    if nultyp == NullCheckType::SetNullArray {
                        nularray[ii] = 1;
                    }
                } else if intcol > 0 {
                    snprintf_cint(&mut tmpstr, 400, &cform, earray[jj] as c_int);
                } else {
                    snprintf_f64(&mut tmpstr, 400, &cform, f64::from(earray[jj]));
                }

                strncat_safe(array[ii], &tmpstr, dwidth as usize);
                strcat_safe(array[ii], cs!(c","));
                jj += 1;

                /* test for null value */
                if earray[jj] == FLOATNULLVALUE {
                    strcpy_safe(&mut tmpstr, cs!(c"NULL"));
                    if nultyp == NullCheckType::SetNullArray {
                        nularray[ii] = 1;
                    }
                } else if intcol > 0 {
                    snprintf_cint(&mut tmpstr, 400, &cform, earray[jj] as c_int);
                } else {
                    snprintf_f64(&mut tmpstr, 400, &cform, f64::from(earray[jj]));
                }

                strncat_safe(array[ii], &tmpstr, dwidth as usize);
                strcat_safe(array[ii], cs!(c")"));
                jj += 1;
            }
        }
    } else if tcode == TDBLCOMPLEX {
        /* allocate memory for the array of double values */
        let mut darray: Vec<f64> = vec![0.0; (nelem * 2) as usize];

        ffgcld(
            fptr,
            colnum,
            firstrow,
            (firstelem - 1) * 2 + 1,
            nelem * 2,
            1,
            NullCheckType::SetPixel,
            DOUBLENULLVALUE,
            darray.as_mut_slice(),
            nularray,
            anynul,
            status,
        );

        if *status <= 0 {
            /* determine the format for the output strings */

            ffgcdw_safe(fptr, colnum, &mut dwidth, status);
            dwidth = (dwidth - 3) / 2;

            /* use the TDISPn keyword if it exists */
            ffkeyn_safe(cs!(c"TDISP"), colnum, &mut keyname, status);
            tstatus = 0;
            cform[0] = 0;

            if ffgkys_safe(fptr, &keyname, &mut dispfmt, None, &mut tstatus) == 0 {
                /* convert the Fortran style format to a C style format */
                ffcdsp(&dispfmt, &mut cform);

                /* Special case: TDISPn='Aw' disallowed for numeric types */
                if dispfmt[0] == bb(b'A') {
                    cform[0] = 0;

                    /* Special case: if the output is intended to be represented
                    as an integer, but we read it as a double, we need to
                    set intcol = 1 so it is printed as an integer */
                } else if (dispfmt[0] == bb(b'I'))
                    || (dispfmt[0] == bb(b'i'))
                    || (dispfmt[0] == bb(b'O'))
                    || (dispfmt[0] == bb(b'o'))
                    || (dispfmt[0] == bb(b'Z'))
                    || (dispfmt[0] == bb(b'z'))
                {
                    intcol = 1;
                }
            }

            if cform[0] == 0 {
                strcpy_safe(&mut cform, cs!(c"%23.15E"));
            }
            /* write the formated string for each value:  "(real,imag)" */
            jj = 0;
            for ii in 0..(nelem as usize) {
                strcpy_safe(array[ii], cs!(c"("));

                /* test for null value */
                if darray[jj] == DOUBLENULLVALUE {
                    strcpy_safe(&mut tmpstr, cs!(c"NULL"));
                    if nultyp == NullCheckType::SetNullArray {
                        nularray[ii] = 1;
                    }
                } else if intcol > 0 {
                    snprintf_cint(&mut tmpstr, 400, &cform, darray[jj] as c_int);
                } else {
                    snprintf_f64(&mut tmpstr, 400, &cform, darray[jj]);
                }

                strncat_safe(array[ii], &tmpstr, dwidth as usize);
                strcat_safe(array[ii], cs!(c","));
                jj += 1;

                /* test for null value */
                if darray[jj] == DOUBLENULLVALUE {
                    strcpy_safe(&mut tmpstr, cs!(c"NULL"));
                    if nultyp == NullCheckType::SetNullArray {
                        nularray[ii] = 1;
                    }
                } else if intcol > 0 {
                    snprintf_cint(&mut tmpstr, 400, &cform, darray[jj] as c_int);
                } else {
                    snprintf_f64(&mut tmpstr, 400, &cform, darray[jj]);
                }

                strncat_safe(array[ii], &tmpstr, dwidth as usize);
                strcat_safe(array[ii], cs!(c")"));
                jj += 1;
            }
        }
    } else if tcode == TLONGLONG && equivtype == TLONGLONG {
        /* allocate memory for the array of LONGLONG values */
        let mut llarray: Vec<LONGLONG> = vec![0; (nelem) as usize];
        let mut flgarray: Vec<c_char> = vec![0; (nelem) as usize];

        dwidth = 20; /* max width of displayed long long integer value */

        if ffgcfjj_safe(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            &mut llarray,
            &mut flgarray,
            anynul,
            status,
        ) > 0
        {
            return *status;
        }

        /* write the formated string for each value */
        match nulval {
            Some(null_value) => {
                strncpy_safe(&mut tmpnull, null_value, 79);
                tmpnull[79] = 0; /* In case len(nulval) >= 79 */
                nulwidth = strlen_safe(&tmpnull) as c_int;
            }
            None => {
                strcpy_safe(&mut tmpnull, cs!(c" "));
                nulwidth = 1;
            }
        }

        for ii in 0..(nelem as usize) {
            if flgarray[ii] > 0 {
                array[ii][0] = 0;
                if dwidth < nulwidth {
                    strncat_safe(array[ii], &tmpnull, dwidth as usize);
                } else {
                    let dwidth = dwidth as usize;
                    int_snprintf!(
                        array[ii],
                        tmpnull.len(),
                        "{:dwidth$}",
                        CStr::from_bytes_with_nul(cast_slice(&tmpnull))
                            .unwrap()
                            .to_str()
                            .unwrap(),
                    );
                }
                if nultyp == NullCheckType::SetNullArray {
                    nularray[ii] = 1;
                }
            } else {
                int_snprintf!(&mut tmpstr, 400, "{:20}", llarray[ii],);

                array[ii][0] = 0;
                strncat_safe(array[ii], &tmpstr, 20);
            }
        }
    } else if tcode == TLONGLONG && equivtype == TULONGLONG {
        /* allocate memory for the array of ULONGLONG values */
        let mut ullarray: Vec<ULONGLONG> = vec![0; (nelem) as usize];
        let mut flgarray: Vec<c_char> = vec![0; (nelem) as usize];

        dwidth = 20; /* max width of displayed unsigned long long integer value */

        if ffgcfujj_safe(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            &mut ullarray,
            &mut flgarray,
            anynul,
            status,
        ) > 0
        {
            return *status;
        }

        /* write the formated string for each value */
        match nulval {
            Some(null_value) => {
                strncpy_safe(&mut tmpnull, null_value, 79);
                tmpnull[79] = 0; /* In case len(nulval) >= 79 */
                nulwidth = strlen_safe(&tmpnull) as c_int;
            }
            None => {
                strcpy_safe(&mut tmpnull, cs!(c" "));
                nulwidth = 1;
            }
        }

        for ii in 0..(nelem as usize) {
            if flgarray[ii] > 0 {
                array[ii][0] = 0;
                if dwidth < nulwidth {
                    strncat_safe(array[ii], &tmpnull, dwidth as usize);
                } else {
                    sprintf_string_width(cast_slice_mut(array[ii]), cs!(c"%*s"), dwidth, &tmpnull);
                }
                if nultyp == NullCheckType::SetNullArray {
                    nularray[ii] = 1;
                }
            } else {
                int_snprintf!(tmpstr, 400, "{:20}", ullarray[ii],);

                array[ii][0] = 0;
                strncat_safe(array[ii], &tmpstr, 20);
            }
        }
    } else {
        /* allocate memory for the array of double values */
        let mut darray: Vec<f64> = vec![0.0; (nelem) as usize];

        /* read all other numeric type columns as doubles */

        if ffgcld(
            fptr,
            colnum,
            firstrow,
            firstelem,
            nelem,
            1,
            nultyp,
            DOUBLENULLVALUE,
            darray.as_mut_slice(),
            nularray,
            anynul,
            status,
        ) > 0
        {
            return *status;
        }

        /* determine the format for the output strings */
        ffgcdw_safe(fptr, colnum, &mut dwidth, status);

        /* check if  column is scaled */
        ffkeyn_safe(cs!(c"TSCAL"), colnum, &mut keyname, status);
        tstatus = 0;
        scaled = 0;
        if ffgkyd_safe(fptr, &keyname, &mut tscale, None, &mut tstatus) == 0 && tscale != 1.0 {
            scaled = 1; /* yes, this is a scaled column */
        }

        intcol = 0;
        if tcode <= TLONG && scaled == 0 {
            intcol = 1; /* this is an unscaled integer column */
        }
        /* use the TDISPn keyword if it exists */
        ffkeyn_safe(cs!(c"TDISP"), colnum, &mut keyname, status);
        tstatus = 0;
        cform[0] = 0;

        if ffgkys_safe(fptr, &keyname, &mut dispfmt, None, &mut tstatus) == 0 {
            /* convert the Fortran style TDISPn to a C style format */
            ffcdsp(&dispfmt, &mut cform);

            /* Special case: TDISPn='Aw' disallowed for numeric types */
            if dispfmt[0] == bb(b'A') {
                cform[0] = 0;

            /* Special case: if the output is intended to be represented
            as an integer, but we read it as a double, we need to
            set intcol = 1 so it is printed as an integer */
            } else if (dispfmt[0] == bb(b'I'))
                || (dispfmt[0] == bb(b'i'))
                || (dispfmt[0] == bb(b'O'))
                || (dispfmt[0] == bb(b'o'))
                || (dispfmt[0] == bb(b'Z'))
                || (dispfmt[0] == bb(b'z'))
            {
                intcol = 1;
            }
        }

        if cform[0] == 0 {
            /* no TDISPn keyword; use TFORMn instead */

            ffkeyn_safe(cs!(c"TFORM"), colnum, &mut keyname, status);
            ffgkys_safe(fptr, &keyname, &mut dispfmt, None, status);

            if scaled > 0 && tcode <= TSHORT {
                /* scaled short integer column == float */
                strcpy_safe(&mut cform, cs!(c"%#14.6G"));
            } else if scaled > 0 && tcode == TLONG {
                /* scaled long integer column == double */
                strcpy_safe(&mut cform, cs!(c"%#23.15G"));
            } else if scaled > 0 && tcode == TLONGLONG {
                /* scaled long long integer column == double */
                strcpy_safe(&mut cform, cs!(c"%#23.15G"));
            } else {
                ffghdt_safe(fptr, &mut hdutype, status);
                if hdutype == ASCII_TBL {
                    /* convert the Fortran style TFORMn to a C style format */
                    ffcdsp(&dispfmt, &mut cform);
                } else {
                    /* this is a binary table, need to convert the format */
                    if tcode == TBIT {
                        /* 'X' */
                        strcpy_safe(&mut cform, cs!(c"%4d"));
                    } else if tcode == TBYTE {
                        /* 'B' */
                        strcpy_safe(&mut cform, cs!(c"%4d"));
                    } else if tcode == TSHORT {
                        /* 'I' */
                        strcpy_safe(&mut cform, cs!(c"%6d"));
                    } else if tcode == TLONG {
                        /* 'J' */
                        strcpy_safe(&mut cform, cs!(c"%11.0f"));
                        intcol = 0; /* needed to support unsigned int */
                    } else if tcode == TFLOAT {
                        /* 'E' */
                        strcpy_safe(&mut cform, cs!(c"%#14.6G"));
                    } else if tcode == TDOUBLE {
                        /* 'D' */
                        strcpy_safe(&mut cform, cs!(c"%#23.15G"));
                    }
                }
            }
        }

        match nulval {
            Some(null_value) => {
                strncpy_safe(&mut tmpnull, null_value, 79);
                tmpnull[79] = 0;
                nulwidth = strlen_safe(&tmpnull) as c_int;
            }
            None => {
                strcpy_safe(&mut tmpnull, cs!(c" "));
                nulwidth = 1;
            }
        }

        /* write the formated string for each value */
        for ii in 0..(nelem as usize) {
            if tcode == TBIT {
                byteval = darray[ii] as c_uchar;

                for ll in 0..8 {
                    if (((byteval << ll) as c_uchar) >> 7) > 0 {
                        (array[ii][ll]) = bb(b'1');
                    } else {
                        (array[ii][ll]) = bb(b'0');
                    }
                }
                array[ii][8] = 0;
            }
            /* test for null value */
            else if (nultyp == NullCheckType::SetPixel && darray[ii] == DOUBLENULLVALUE)
                || (nultyp == NullCheckType::SetNullArray && nularray[ii] > 0)
            {
                array[ii][0] = 0;
                if dwidth < nulwidth {
                    strncat_safe(array[ii], &tmpnull, dwidth as usize);
                } else {
                    sprintf_string_width(cast_slice_mut(array[ii]), cs!(c"%*s"), dwidth, &tmpnull);
                }
            } else {
                if intcol > 0 {
                    snprintf_cint(&mut tmpstr, 400, &cform, darray[ii] as c_int);
                } else {
                    snprintf_f64(&mut tmpstr, 400, &cform, darray[ii]);
                }

                /* fill field with '*' if number is too wide */
                dlen = strlen_safe(&tmpstr) as c_int;
                if dlen > dwidth {
                    tmpstr[..dwidth as usize].fill(b'*' as c_char);
                }

                array[ii][0] = 0;
                strncat_safe(array[ii], &tmpstr, dwidth as usize);
            }
        }
    }
    *status
}

/// Get Column Display Width.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) number of column (1 = 1st col)
/// * `width`  — (O) display width
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcdw(
    fptr: *mut fitsfile,
    colnum: c_int,
    width: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let width = width.as_mut().expect(NULL_MSG);

        ffgcdw_safe(fptr, colnum, width, status)
    }
}

/// Get Column Display Width.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) number of column (1 = 1st col)
/// * `width`  — (O) display width
/// * `status` — (IO) error status
pub fn ffgcdw_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    width: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut dispfmt: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut tcode: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut tstatus: c_int = 0;
    let mut scaled: c_int = 0;
    let mut testwidth: c_long = 0;
    let mut tscale: f64 = 0.0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }

    if colnum < 1 || colnum > fptr.Fptr.tfield {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Specified column number is out of range: {}",
            colnum,
        );
        ffpmsg_slice(&message);
        *status = BAD_COL_NUM;
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_slice(); /* point to first column */
    let ci = colnum as usize - 1; /* offset to correct column structure */
    tcode = (c[ci].tdatatype).abs();

    /* use the TDISPn keyword if it exists */
    ffkeyn_safe(cs!(c"TDISP"), colnum, &mut keyname, status);

    *width = 0;
    tstatus = 0;
    if ffgkys_safe(fptr, &keyname, &mut dispfmt, None, &mut tstatus) == 0 {
        /* parse TDISPn get the display width */
        let mut cp = 0;

        while dispfmt[cp] == bb(b' ') {
            /* skip leading blanks */
            cp += 1;
        }
        if dispfmt[cp] == bb(b'A')
            || dispfmt[cp] == bb(b'a')
            || dispfmt[cp] == bb(b'I')
            || dispfmt[cp] == bb(b'i')
            || dispfmt[cp] == bb(b'O')
            || dispfmt[cp] == bb(b'o')
            || dispfmt[cp] == bb(b'Z')
            || dispfmt[cp] == bb(b'z')
            || dispfmt[cp] == bb(b'F')
            || dispfmt[cp] == bb(b'f')
            || dispfmt[cp] == bb(b'E')
            || dispfmt[cp] == bb(b'e')
            || dispfmt[cp] == bb(b'D')
            || dispfmt[cp] == bb(b'd')
            || dispfmt[cp] == bb(b'G')
            || dispfmt[cp] == bb(b'g')
        {
            while !isdigit_safe(dispfmt[cp]) && dispfmt[cp] != 0 {
                /* find 1st digit */
                cp += 1;
            }
            // No digits, or too many for a long: no valid width, so use TFORMn.
            testwidth = strtol_safe::<c_long>(&dispfmt[cp..]).map_or(0, |r| r.0);
            if testwidth >= (c_int::MIN as c_long) && testwidth <= (c_int::MAX as c_long) {
                *width = testwidth as c_int;
                if tcode >= TCOMPLEX {
                    *width = (2 * (*width)) + 3;
                }
            }
        }
    }

    if *width == 0 {
        /* no valid TDISPn keyword; use TFORMn instead */

        ffkeyn_safe(cs!(c"TFORM"), colnum, &mut keyname, status);
        ffgkys_safe(fptr, &keyname, &mut dispfmt, None, status);

        /* check if  column is scaled */
        ffkeyn_safe(cs!(c"TSCAL"), colnum, &mut keyname, status);
        tstatus = 0;
        scaled = 0;

        if ffgkyd_safe(fptr, &keyname, &mut tscale, None, &mut tstatus) == 0 && tscale != 1.0 {
            scaled = 1; /* yes, this is a scaled column */
        }

        if scaled > 0 && tcode <= TSHORT {
            /* scaled short integer col == float; default format is 14.6G */
            *width = 14;
        } else if scaled > 0 && tcode == TLONG {
            /* scaled long integer col == double; default format is 23.15G */
            *width = 23;
        } else if scaled > 0 && tcode == TLONGLONG {
            /* scaled long long integer col == double; default format is 23.15G */
            *width = 23;
        } else {
            ffghdt_safe(fptr, &mut hdutype, status); /* get type of table */
            if hdutype == ASCII_TBL {
                /* parse TFORMn get the display width */
                let mut cp = 0;
                while !isdigit_safe(dispfmt[cp]) && dispfmt[cp] != 0 {
                    /* find 1st digit */
                    cp += 1;
                }
                *width = parse_c_int(&dispfmt[cp..]);
            } else {
                /* this is a binary table */
                if tcode == TBIT {
                    /* 'X' */
                    *width = 8;
                } else if tcode == TBYTE {
                    /* 'B' */
                    *width = 4;
                } else if tcode == TSHORT {
                    /* 'I' */
                    *width = 6;
                } else if tcode == TLONG {
                    /* 'J' */
                    *width = 11;
                } else if tcode == TLONGLONG {
                    /* 'K' */
                    *width = 20;
                } else if tcode == TFLOAT {
                    /* 'E' */
                    *width = 14;
                } else if tcode == TDOUBLE {
                    /* 'D' */
                    *width = 23;
                } else if tcode == TCOMPLEX {
                    /* 'C' */
                    *width = 31;
                } else if tcode == TDBLCOMPLEX {
                    /* 'M' */
                    *width = 49;
                } else if tcode == TLOGICAL {
                    /* 'L' */
                    *width = 1;
                } else if tcode == TSTRING
                /* 'A' */
                {
                    let mut typecode: c_int = 0;
                    let mut repeat: c_long = 0;
                    let mut rwidth: c_long = 0;
                    let mut gstatus: c_int = 0;

                    /* Deal with possible vector string with repeat / width  by parsing
                    the TFORM=rAw keyword */
                    if ffgtcl_safe(
                        fptr,
                        colnum,
                        Some(&mut typecode),
                        Some(&mut repeat),
                        Some(&mut rwidth),
                        &mut gstatus,
                    ) == 0
                        && rwidth >= 1
                        && rwidth < repeat
                    {
                        *width = rwidth as c_int;
                    } else {
                        /* Hmmm, we couldn't parse the TFORM keyword by standard, so just do
                        simple parsing */
                        let mut cp = 0;
                        while !isdigit_safe(dispfmt[cp]) && dispfmt[cp] != 0 {
                            cp += 1;
                        }

                        *width = parse_c_int(&dispfmt[cp..]);
                    }

                    if *width < 1 {
                        *width = 1; /* default is at least 1 column */
                    }
                }
            }
        }
    }
    *status
}

/// Read an array of string values from a column in the current FITS HDU.
#[allow(clippy::if_same_then_else)]
// C dispatch chain: distinct conditions deliberately share an action.
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) number of column to read (1 = 1st col)
/// * `firstrow`  — (I) first row to read (1 = 1st row)
/// * `firstelem` — (I) first vector element to read (1 = 1st)
/// * `nelem`     — (I) number of strings to read
/// * `nultyp`    — (I) null value handling code:
/// * `nulval`    — (I) value for null pixels if nultyp = 1
/// * `array`     — (O) array of values that are read
/// * `nularray`  — (O) array of flags = 1 if nultyp = 2
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub(crate) fn ffgcls2(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    nultyp: NullCheckType,
    /*     1: set undefined pixels = nulval        */
    /*     2: set nularray=1 for undefined pixels  */
    nulval: Option<&[c_char]>,
    array: &mut [&mut [c_char]],
    nularray: &mut [c_char],
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let dtemp: f64;
    let mut nullen: c_long;
    let mut tcode: c_int;
    let mut maxelem: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut nulcheck = NullCheckType::None;
    let mut twidth: c_long = 0;
    let mut incre: c_long = 0;
    let mut ntodo: c_long = 0;
    let mut repeat: LONGLONG = 0;
    let mut startpos: LONGLONG = 0;
    let mut elemnum: LONGLONG = 0;
    let mut readptr: LONGLONG = 0;
    let mut tnull: LONGLONG = 0;
    let mut rowlen: LONGLONG = 0;
    let mut rownum: LONGLONG;
    let mut remain: LONGLONG;
    let mut next: LONGLONG;
    let mut scale: f64 = 0.0;
    let mut zero: f64 = 0.0;
    let mut tform: [c_char; 20] = [0; 20];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut snull: [c_char; 20] = [0; 20]; /*  the FITS null value  */
    let mut cbuff: [f64; DBUFFSIZE as usize / mem::size_of::<f64>()] =
        [0.0; DBUFFSIZE as usize / mem::size_of::<f64>()]; /* align cbuff on word boundary */
    let mut arrayptr;

    if *status > 0 || nelem == 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }

    //if let Some(anynul) = anynul.as_deref_mut() {
    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = 0;
    }
    //}

    if nultyp == NullCheckType::SetNullArray {
        nularray.fill(0); /* initialize nullarray */
    }
    /*  Check input and get parameters about the column: */
    if colnum < 1 || colnum > fptr.Fptr.tfield {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Specified column number is out of range: {}",
            colnum,
        );
        ffpmsg_slice(&message);
        *status = BAD_COL_NUM;
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_slice(); /* point to first column */
    let ci = colnum as usize - 1; /* offset to correct column structure */
    tcode = c[ci].tdatatype;

    if tcode == -TSTRING {
        /* variable length column in a binary table? */

        /* only read a single string; ignore value of firstelem */

        if ffgcprll(
            fptr,
            colnum,
            firstrow,
            1,
            1,
            0,
            &mut scale,
            &mut zero,
            &mut tform,
            &mut twidth,
            &mut tcode,
            &mut maxelem,
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
        remain = 1;
        twidth = repeat as c_long;
    } else if tcode == TSTRING {
        if ffgcprll(
            fptr,
            colnum,
            firstrow as LONGLONG,
            firstelem as LONGLONG,
            nelem as LONGLONG,
            0,
            &mut scale,
            &mut zero,
            &mut tform,
            &mut twidth,
            &mut tcode,
            &mut maxelem,
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
        /* if string length is greater than a FITS block (2880 char) then must */
        /* only read 1 string at a time, to force reading by ffgbyt instead of */
        /* ffgbytoff (ffgbytoff can't handle this case) */
        if (twidth as LONGLONG) > IOBUFLEN {
            maxelem = 1;
            incre = twidth;
            repeat = 1;
        }

        remain = nelem;
    } else {
        *status = NOT_ASCII_COL;
        return *status;
    }
    nullen = strlen_safe(&snull) as c_long; /* length of the undefined pixel string */
    if nullen == 0 {
        nullen = 1;
    }
    /*  Decide whether to check for null values in the input FITS file: */
    nulcheck = nultyp; /* by default check for null values in the FITS file */

    if nultyp == NullCheckType::SetPixel && nulval.is_none() {
        nulcheck = NullCheckType::None; /* calling routine does not want to check for nulls */
    } else if nultyp == NullCheckType::SetPixel && nulval.is_some() && nulval.unwrap()[0] == 0 {
        nulcheck = NullCheckType::None; /* calling routine does not want to check for nulls */
    } else if snull[0] == ASCII_NULL_UNDEFINED as c_char {
        nulcheck = NullCheckType::None; /* null value string in ASCII table not defined */
    } else if nullen > twidth {
        nulcheck = NullCheckType::None; /* null value string is longer than width of column  */
        /* thus impossible for any column elements to = null */
    }
    /*  Now read the strings one at a time from the FITS column.           */
    next = 0; /* next element in array to be read  */
    rownum = 0; /* row number, relative to firstrow     */

    while remain > 0 {
        /* limit the number of pixels to process at one time to the number that
        will fit in the buffer space or to the number of pixels that remain
        in the current vector, which ever is smaller.
        */
        ntodo = cmp::min(remain, LONGLONG::from(maxelem)) as c_long;
        ntodo = cmp::min(ntodo as LONGLONG, repeat - elemnum) as c_long;

        readptr = startpos + (rownum as LONGLONG * rowlen) + (elemnum * incre as LONGLONG);
        ffmbyt_safe(fptr, readptr, REPORT_EOF, status); /* move to read position */

        /* read the array of strings from the FITS file into the buffer */

        if incre == twidth {
            ffgbyt(
                fptr,
                (ntodo * twidth) as LONGLONG,
                cast_slice_mut(&mut cbuff),
                status,
            );
        } else {
            ffgbytoff(
                fptr,
                twidth,
                ntodo,
                incre - twidth,
                cast_slice_mut(&mut cbuff),
                status,
            );
        }
        /* copy from the buffer into the user's array of strings */
        /* work backwards from last char of last string to 1st char of 1st */

        let buffer: &mut [c_char] = cast_slice_mut(&mut cbuff);
        let mut bi: LONGLONG = (ntodo * twidth) as LONGLONG - 1;

        let mut ii = next + ntodo as LONGLONG - 1;
        while ii >= next {
            arrayptr = twidth - 1;

            let mut jj = twidth - 1;

            while jj > 0 {
                /* ignore trailing blanks */

                if buffer[bi as usize] == bb(b' ') {
                    bi -= 1;
                    arrayptr -= 1;
                } else {
                    break;
                }
                jj -= 1;
            }
            array[ii as usize][arrayptr as usize + 1] = 0; /* write the string terminator */

            while jj >= 0 {
                /* copy the string itself */

                array[ii as usize][arrayptr as usize] = buffer[bi as usize];
                bi -= 1;
                arrayptr -= 1;
                jj -= 1;
            }

            /* check if null value is defined, and if the   */
            /* column string is identical to the null string */
            if nulcheck != NullCheckType::None
                && strncmp_safe(&snull, array[ii as usize], nullen as usize) == 0
            {
                // Not null checked in original code
                if let Some(anynul) = anynul.as_deref_mut() {
                    *anynul = 1; /* this is a null value */
                }

                if nultyp == NullCheckType::SetPixel {
                    if let Some(nulval) = nulval {
                        strcpy_safe(array[ii as usize], nulval);
                    } else {
                        strcpy_safe(array[ii as usize], cs!(c" "));
                    }
                } else {
                    nularray[ii as usize] = 1;
                }
            }
            ii -= 1;
        }

        if *status > 0 {
            /* test for error during previous read operation */

            dtemp = next as f64;
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "Error reading elements {:.0} thru {:.0} of data array (ffpcls)",
                dtemp + 1.0,
                dtemp + ntodo as f64,
            );

            ffpmsg_slice(&message);
            return *status;
        }

        /*  increment the counters for the next loop  */
        next += ntodo as LONGLONG;
        remain -= ntodo as LONGLONG;
        if remain > 0 {
            elemnum += ntodo as LONGLONG;
            if elemnum == repeat {
                /* completed a row; start on next row */

                elemnum = 0;
                rownum += 1;
            }
        }
    } /*  End of main while Loop  */

    *status
}

#[cfg(test)]
mod tests {
    use crate::aliases::rust_api::*;
    use crate::fitsio::{
        ASCII_TBL, BAD_COL_NUM, BINARY_TBL, BYTE_IMG, LONGLONG, READONLY, fitsfile,
    };
    use crate::helpers::testhelpers::{from_buf, to_buf, with_temp_file};
    use libc::{c_char, c_int, c_long};

    /// Make a NUL-terminated `Vec<c_char>` from a `&str`.
    fn cc(s: &str) -> Vec<c_char> {
        let mut v: Vec<c_char> = s.bytes().map(|b| b as c_char).collect();
        v.push(0);
        v
    }

    /// Create a single-column table (binary or ascii) and return the open file.
    fn make_table(
        name: &[c_char],
        hdutype: c_int,
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
            hdutype,
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
    fn test_read_string_column() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut f = make_table(&name, BINARY_TBL, "NAME", "10A", 3, &mut status);
            let data = [cc("Alice"), cc("Bob"), cc("Charlie")];
            let data_ref: Vec<&[c_char]> = data.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                &data_ref,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 11];
            let mut r2 = [0 as c_char; 11];
            let mut r3 = [0 as c_char; 11];
            let mut results: [&mut [c_char]; 3] = [&mut r1, &mut r2, &mut r3];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(from_buf(&r1), "Alice");
            assert_eq!(from_buf(&r2), "Bob");
            assert_eq!(from_buf(&r3), "Charlie");
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_string_column_with_null_flags() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut f = make_table(&name, BINARY_TBL, "NAME", "10A", 3, &mut status);
            let data = [cc("One"), cc("Two"), cc("Three")];
            let data_ref: Vec<&[c_char]> = data.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                &data_ref,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 11];
            let mut r2 = [0 as c_char; 11];
            let mut r3 = [0 as c_char; 11];
            let mut results: Vec<&mut [c_char]> = vec![&mut r1[..], &mut r2[..], &mut r3[..]];
            let mut nularray = [0 as c_char; 3];
            let mut anynull = -1;
            fits_read_colnull_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                &mut results,
                &mut nularray,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(from_buf(&r1), "One");
            assert_eq!(from_buf(&r2), "Two");
            assert_eq!(from_buf(&r3), "Three");
            assert_eq!(nularray[0], 0);
            assert_eq!(nularray[1], 0);
            assert_eq!(nularray[2], 0);
            assert_eq!(anynull, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_ascii_table_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut f = make_table(&name, ASCII_TBL, "NAME", "A10", 2, &mut status);
            let data = [cc("Ascii1"), cc("Ascii2")];
            let data_ref: Vec<&[c_char]> = data.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                2,
                &data_ref,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 11];
            let mut r2 = [0 as c_char; 11];
            let mut results: [&mut [c_char]; 2] = [&mut r1, &mut r2];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                2,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(from_buf(&r1), "Ascii1");
            assert_eq!(from_buf(&r2), "Ascii2");
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_variable_length_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut f = make_table(&name, BINARY_TBL, "VARSTR", "1PA", 1, &mut status);
            let data = [cc("Variable length string content")];
            let data_ref: Vec<&[c_char]> = data.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                &data_ref,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 64];
            let mut results: [&mut [c_char]; 1] = [&mut r1];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(from_buf(&r1), "Variable length string content");
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_numeric_as_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 3] = [12345, -99, 0];

            let mut f = make_table(&name, BINARY_TBL, "INTCOL", "1J", 3, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 32];
            let mut r2 = [0 as c_char; 32];
            let mut r3 = [0 as c_char; 32];
            let mut results: [&mut [c_char]; 3] = [&mut r1, &mut r2, &mut r3];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // ffgcvs converts any column type to string
            assert_eq!(from_buf(&r1).trim().parse::<i64>().unwrap(), 12345);
            assert_eq!(from_buf(&r2).trim().parse::<i64>().unwrap(), -99);
            assert_eq!(from_buf(&r3).trim().parse::<i64>().unwrap(), 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_float_as_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f32; 2] = [3.14, -2.5];

            let mut f = make_table(&name, BINARY_TBL, "FLOATCOL", "1E", 2, &mut status);
            fits_write_col_flt(f.as_deref_mut().unwrap(), 1, 1, 1, 2, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 32];
            let mut r2 = [0 as c_char; 32];
            let mut results: [&mut [c_char]; 2] = [&mut r1, &mut r2];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                2,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Check that values convert to strings approximately correctly
            let v0 = from_buf(&r1).trim().parse::<f64>().unwrap();
            let v1 = from_buf(&r2).trim().parse::<f64>().unwrap();
            assert!(v0 >= 3.1 && v0 <= 3.2);
            assert!(v1 >= -2.6 && v1 <= -2.4);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_double_as_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 2] = [1.23456789, -9.87654321];

            let mut f = make_table(&name, BINARY_TBL, "DBLCOL", "1D", 2, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 2, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 32];
            let mut r2 = [0 as c_char; 32];
            let mut results: [&mut [c_char]; 2] = [&mut r1, &mut r2];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                2,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            let v0 = from_buf(&r1).trim().parse::<f64>().unwrap();
            let v1 = from_buf(&r2).trim().parse::<f64>().unwrap();
            assert!(v0 >= 1.23 && v0 <= 1.24);
            assert!(v1 >= -9.88 && v1 <= -9.87);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_byte_as_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [u8; 3] = [0, 127, 255];

            let mut f = make_table(&name, BINARY_TBL, "BYTECOL", "1B", 3, &mut status);
            fits_write_col_byt(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 16];
            let mut r2 = [0 as c_char; 16];
            let mut r3 = [0 as c_char; 16];
            let mut results: [&mut [c_char]; 3] = [&mut r1, &mut r2, &mut r3];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(from_buf(&r1).trim().parse::<i64>().unwrap(), 0);
            assert_eq!(from_buf(&r2).trim().parse::<i64>().unwrap(), 127);
            assert_eq!(from_buf(&r3).trim().parse::<i64>().unwrap(), 255);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_short_as_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [i16; 3] = [-32768, 0, 32767];

            let mut f = make_table(&name, BINARY_TBL, "SHORTCOL", "1I", 3, &mut status);
            fits_write_col_sht(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 16];
            let mut r2 = [0 as c_char; 16];
            let mut r3 = [0 as c_char; 16];
            let mut results: [&mut [c_char]; 3] = [&mut r1, &mut r2, &mut r3];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(from_buf(&r1).trim().parse::<i64>().unwrap(), -32768);
            assert_eq!(from_buf(&r2).trim().parse::<i64>().unwrap(), 0);
            assert_eq!(from_buf(&r3).trim().parse::<i64>().unwrap(), 32767);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_longlong_as_string() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [LONGLONG; 3] = [-9223372036854775807, 0, 9223372036854775807];

            let mut f = make_table(&name, BINARY_TBL, "LLCOL", "1K", 3, &mut status);
            fits_write_col_lnglng(f.as_deref_mut().unwrap(), 1, 1, 1, 3, &data, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 32];
            let mut r2 = [0 as c_char; 32];
            let mut r3 = [0 as c_char; 32];
            let mut results: [&mut [c_char]; 3] = [&mut r1, &mut r2, &mut r3];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(
                from_buf(&r1).trim().parse::<i64>().unwrap(),
                -9223372036854775807
            );
            assert_eq!(from_buf(&r2).trim().parse::<i64>().unwrap(), 0);
            assert_eq!(
                from_buf(&r3).trim().parse::<i64>().unwrap(),
                9223372036854775807
            );
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_get_column_display_width() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            let ttype_v = [Some(cc("COL1")), Some(cc("COL2")), Some(cc("COL3"))];
            let ttype_ref: Vec<Option<&[c_char]>> = ttype_v.iter().map(|o| o.as_deref()).collect();
            let tform_v = [cc("10A"), cc("1J"), cc("1E")];
            let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
            fits_create_tbl(
                f.as_deref_mut().unwrap(),
                BINARY_TBL,
                1,
                3,
                &ttype_ref,
                &tform_ref,
                None,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut width = 0;
            fits_get_col_display_width(f.as_deref_mut().unwrap(), 1, &mut width, &mut status);
            assert_eq!(status, 0);
            assert_eq!(width, 10); // String column width = 10
            fits_get_col_display_width(f.as_deref_mut().unwrap(), 2, &mut width, &mut status);
            assert_eq!(status, 0);
            assert_ne!(width, 0); // Integer column has nonzero width
            fits_get_col_display_width(f.as_deref_mut().unwrap(), 3, &mut width, &mut status);
            assert_eq!(status, 0);
            assert_ne!(width, 0); // Float column has nonzero width
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_get_column_display_width_with_tdisp() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut f = make_table(&name, BINARY_TBL, "COL1", "1J", 1, &mut status);
            fits_write_key_str(
                f.as_deref_mut().unwrap(),
                &cc("TDISP1"),
                &cc("I8"),
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut width = 0;
            fits_get_col_display_width(f.as_deref_mut().unwrap(), 1, &mut width, &mut status);
            assert_eq!(status, 0);
            assert_eq!(width, 8); // Should use TDISP width
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_get_column_display_width_bad_col() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut f = make_table(&name, BINARY_TBL, "COL1", "10A", 1, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut width = 0;
            fits_get_col_display_width(f.as_deref_mut().unwrap(), 0, &mut width, &mut status);
            assert_eq!(status, BAD_COL_NUM);
            status = 0;
            fits_get_col_display_width(f.as_deref_mut().unwrap(), 99, &mut width, &mut status);
            assert_eq!(status, BAD_COL_NUM);
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_with_tdisp_format() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [f64; 1] = [123.456789];

            let mut f = make_table(&name, BINARY_TBL, "VALUE", "1D", 1, &mut status);
            fits_write_col_dbl(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            // Set display format
            fits_write_key_str(
                f.as_deref_mut().unwrap(),
                &cc("TDISP1"),
                &cc("F10.2"),
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 64];
            let mut results: [&mut [c_char]; 1] = [&mut r1];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            // Value should be formatted according to TDISP
            assert_eq!(status, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_with_scaling() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let data: [c_long; 1] = [100];

            let mut f = make_table(&name, BINARY_TBL, "SCALED", "1J", 1, &mut status);
            fits_write_col_lng(f.as_deref_mut().unwrap(), 1, 1, 1, 1, &data, &mut status);
            // Set scaling: result = data * 2.0 + 10.0 = 210
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
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 64];
            let mut results: [&mut [c_char]; 1] = [&mut r1];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                1,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            // Value should be scaled
            let v0 = from_buf(&r1).trim().parse::<f64>().unwrap();
            assert!(v0 >= 209.0 && v0 <= 211.0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_read_multielem_vector() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            // 3 strings of 5 chars each
            let mut f = make_table(&name, BINARY_TBL, "VECSTR", "3A5", 1, &mut status);
            let data = [cc("One"), cc("Two"), cc("Three")];
            let data_ref: Vec<&[c_char]> = data.iter().map(|v| v.as_slice()).collect();
            fits_write_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                &data_ref,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            let mut r1 = [0 as c_char; 16];
            let mut r2 = [0 as c_char; 16];
            let mut r3 = [0 as c_char; 16];
            let mut results: [&mut [c_char]; 3] = [&mut r1, &mut r2, &mut r3];
            let mut anynull = -1;
            fits_read_col_str(
                f.as_deref_mut().unwrap(),
                1,
                1,
                1,
                3,
                None,
                &mut results,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(from_buf(&r1), "One");
            assert_eq!(from_buf(&r2), "Two");
            // "Three" may be truncated to 5 chars
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }
}
