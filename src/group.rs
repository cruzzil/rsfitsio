//! Support for the FITS hierarchical grouping convention.
//!
//! A grouping table is a binary table whose rows name other HDUs -- its
//! members -- so that a set of related HDUs, possibly spread across several
//! files, can be treated as a unit. A member is identified by some combination
//! of position, reference and URI; the `GT_ID_*` constants select which.
//!
//! Because a member may live in another file, most of the work here is
//! resolving the URI that names it, which is why the routines below deal in
//! partial and absolute URLs.
//!
//! Ported from CFITSIO's `grouping.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center. See the "Hierarchical Grouping Routines" chapter of the
//! CFITSIO User's Reference Guide.
#![warn(missing_docs)]
/*                                                                         */
/*  The group.c module of CFITSIO was written by Donald G. Jennings of     */
/*  the INTEGRAL Science Data Centre (ISDC) under NASA contract task       */
/*  66002J6. The above copyright laws apply. Copyright guidelines of The   */
/*  University of Geneva might also apply.                                 */

/*  The following routines are designed to create, read, and manipulate    */
/*  FITS Grouping Tables as defined in the FITS Grouping Convention paper  */
/*  by Jennings, Pence, Folk and Schlesinger. The development of the       */
/*  grouping structure was partially funded under the NASA AISRP Program.  */

use alloc::collections::VecDeque;

use crate::aliases::rust_api::*;
use crate::c_types::{c_char, c_int, c_long, c_uchar};
use bytemuck::{cast_slice, cast_slice_mut};

use core::ffi::CStr;

use crate::{
    aliases::rust_api::{fits_read_key_lng, fits_read_keyword},
    bb, cs,
    fitscore::*,
    fitsio::*,
    int_snprintf, raw_to_slice,
    wrappers::{
        strcat_safe, strchr_safe, strcmp_safe, strcpy_safe, strlen_safe, strncmp_safe,
        strncpy_safe, strrchr_safe, strstr_safe,
    },
};

/// The character introducing a percent-escape in a group member URI.
pub const HEX_ESCAPE: u8 = b'%';

/// Maximum number of HDUs that can be tracked while copying a group.
pub const MAX_HDU_TRACKER: usize = 1000;

// The C uses `char *filename[MAXHDU]` etc., where each entry is either NULL or a
// malloc'd buffer. The safe equivalent is a fixed MAX_HDU_TRACKER-sized array whose
// entries are `Option<Box<[c_char; FLEN_FILENAME]>>` (None == the C NULL, Some == an
// owned, heap-allocated buffer). The array itself is just MAX_HDU_TRACKER pointers, so
// the struct stays small, and the per-entry buffers replace the manual malloc/free.
pub(crate) struct HDUtracker {
    nHDU: c_int,

    filename: [Option<Box<[c_char; FLEN_FILENAME]>>; MAX_HDU_TRACKER],
    position: [c_int; MAX_HDU_TRACKER],

    newFilename: [Option<Box<[c_char; FLEN_FILENAME]>>; MAX_HDU_TRACKER],
    newPosition: [c_int; MAX_HDU_TRACKER],
}

impl Default for HDUtracker {
    fn default() -> Self {
        HDUtracker {
            nHDU: 0,
            // `Option<Box<_>>` is not Copy, so build the all-None arrays with from_fn.
            filename: core::array::from_fn(|_| None),
            position: [0; MAX_HDU_TRACKER],
            newFilename: core::array::from_fn(|_| None),
            newPosition: [0; MAX_HDU_TRACKER],
        }
    }
}

/// Create a grouping table at the end of the current FITS file.
///
/// This function makes the last HDU in the file the CHDU, then calls the
/// fits_insert_group() function to actually create the new grouping table.
/// grouping table information:
///   GT_ID_ALL_URI  0 ==> defualt (all columns)
///   GT_ID_REF      1 ==> ID by reference
///   GT_ID_POS      2 ==> ID by position
///   GT_ID_ALL      3 ==> ID by ref. and position
///   GT_ID_REF_URI 11 ==> (1) + URI info
///   GT_ID_POS_URI 12 ==> (2) + URI info  
///
/// # Parameters
///
/// * `fptr`      — FITS file pointer
/// * `grpname`   — name of the grouping table
/// * `grouptype` — code specifying the type of
/// * `status`    — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtcr(
    fptr: *mut fitsfile,
    grpname: *const c_char,
    grouptype: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        raw_to_slice!(grpname);

        ffgtcr_safe(fptr, grpname, grouptype, status)
    }
}

/// Create a grouping table at the end of the current FITS file.
///
/// This function makes the last HDU in the file the CHDU, then calls the
/// fits_insert_group() function to actually create the new grouping table.
/// grouping table information:
///   GT_ID_ALL_URI  0 ==> defualt (all columns)
///   GT_ID_REF      1 ==> ID by reference
///   GT_ID_POS      2 ==> ID by position
///   GT_ID_ALL      3 ==> ID by ref. and position
///   GT_ID_REF_URI 11 ==> (1) + URI info
///   GT_ID_POS_URI 12 ==> (2) + URI info  
///
/// # Parameters
///
/// * `fptr`      — FITS file pointer
/// * `grpname`   — name of the grouping table
/// * `grouptype` — code specifying the type of
/// * `status`    — return status code
pub fn ffgtcr_safe(
    fptr: &mut fitsfile,
    grpname: &[c_char],
    grouptype: c_int,
    status: &mut c_int,
) -> c_int {
    let mut hdutype: c_int = 0;
    let mut hdunum: c_int = 0;

    if *status != 0 {
        return *status;
    }

    *status = fits_get_num_hdus(fptr, &mut hdunum, status);

    /* If hdunum is 0 then we are at the beginning of the file and
    we actually haven't closed the first header yet, so don't do
    anything more */

    if 0 != hdunum {
        *status = fits_movabs_hdu(fptr, hdunum, Some(&mut hdutype), status);
    }

    /* Now, the whole point of the above two fits_ calls was to get to
       the end of file.  Let's ignore errors at this point and keep
       going since any error is likely to mean that we are already at the
       EOF, or the file is fatally corrupted.  If we are at the EOF then
       the next fits_ call will be ok.  If it's corrupted then the
       next call will fail, but that's not big deal at this point.
    */

    if 0 != *status {
        *status = 0;
    }

    *status = fits_insert_group(fptr, grpname, grouptype, status);

    *status
}

/// Insert a grouping table just after the current HDU of the current FITS file.
///
/// This is the same as fits_create_group() only it allows the user to select
/// the place within the FITS file to add the grouping table.
/// grouping table information:
///   GT_ID_ALL_URI  0 ==> defualt (all columns)
///   GT_ID_REF      1 ==> ID by reference
///   GT_ID_POS      2 ==> ID by position
///   GT_ID_ALL      3 ==> ID by ref. and position
///   GT_ID_REF_URI 11 ==> (1) + URI info
///   GT_ID_POS_URI 12 ==> (2) + URI info  
///
/// # Parameters
///
/// * `fptr`      — FITS file pointer
/// * `grpname`   — name of the grouping table
/// * `grouptype` — code specifying the type of
/// * `status`    — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtis(
    fptr: *mut fitsfile,
    grpname: *const c_char,
    grouptype: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        raw_to_slice!(grpname);

        ffgtis_safe(fptr, grpname, grouptype, status)
    }
}

/// Insert a grouping table just after the current HDU of the current FITS file.
///
/// This is the same as fits_create_group() only it allows the user to select
/// the place within the FITS file to add the grouping table.
/// grouping table information:
///   GT_ID_ALL_URI  0 ==> defualt (all columns)
///   GT_ID_REF      1 ==> ID by reference
///   GT_ID_POS      2 ==> ID by position
///   GT_ID_ALL      3 ==> ID by ref. and position
///   GT_ID_REF_URI 11 ==> (1) + URI info
///   GT_ID_POS_URI 12 ==> (2) + URI info  
///
/// # Parameters
///
/// * `fptr`      — FITS file pointer
/// * `grpname`   — name of the grouping table
/// * `grouptype` — code specifying the type of
/// * `status`    — return status code
pub fn ffgtis_safe(
    fptr: &mut fitsfile,
    grpname: &[c_char],
    grouptype: c_int,
    status: &mut c_int,
) -> c_int {
    let mut tfields: c_int = 0;
    let mut hdunum: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut extver: c_int;

    let pcount: c_long = 0;

    /* `char *ttype[6]` / `char *tform[6]` are 6 column pointers that in C address    */
    /* the flat backing buffers `char ttypeBuff[102]` (6*17) and `char tformBuff[54]` */
    /* (6*9). A 2-D array captures the same "6 columns of fixed width" layout.        */
    let mut ttypeBuff: [[c_char; 17]; 6] = [[0; 17]; 6];
    let mut tformBuff: [[c_char; 9]; 6] = [[0; 9]; 6];

    let extname = cs!(c"GROUPING"); /* char  extname[] = "GROUPING"; */
    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    // `do { ... } while(0)` is modelled as a `loop { ...; break; }`; the C `continue`
    // statements (which fall through to the `while(0)` test) become `break`.
    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        /* set up the ttype and tform character buffers */
        // Each column is one row of the 2-D array; pass them to ffgtdc as mutable
        // slices (stack-allocated, like the C `char *ttype[6]`) so it can write each
        // column's TTYPE/TFORM string.
        {
            let mut ttype: [&mut [c_char]; 6] = ttypeBuff.each_mut().map(|c| c.as_mut_slice());
            let mut tform: [&mut [c_char]; 6] = tformBuff.each_mut().map(|c| c.as_mut_slice());

            /* define the columns required according to the grouptype parameter */

            *status = ffgtdc(
                grouptype,
                0,
                0,
                0,
                0,
                0,
                0,
                &mut ttype,
                &mut tform,
                &mut tfields,
                status,
            );
        }

        /* create the grouping table using the columns defined above */
        // Re-borrow the (now populated) columns as immutable views.
        let ttype: [Option<&[c_char]>; 6] = ttypeBuff.each_ref().map(|c| Some(c.as_slice()));
        let tform: [&[c_char]; 6] = tformBuff.each_ref().map(|c| c.as_slice());

        *status = fits_insert_btbl(
            fptr,
            0,
            tfields,
            &ttype,
            &tform,
            None,
            None,
            pcount as LONGLONG,
            status,
        );

        if *status != 0 {
            break;
        }

        /*
        retrieve the hdu position of the new grouping table for
        future use
         */

        fits_get_hdu_num(fptr, &mut hdunum);

        /*
        add the EXTNAME and EXTVER keywords to the HDU just after the
        TFIELDS keyword; for now the EXTVER value is set to 0, it will be
        set to the correct value later on
         */

        fits_read_keyword(
            fptr,
            cs!(c"TFIELDS"),
            &mut keyvalue,
            Some(&mut comment),
            status,
        );

        fits_insert_key_str(
            fptr,
            cs!(c"EXTNAME"),
            extname,
            Some(cs!(c"HDU contains a Grouping Table")),
            status,
        );
        fits_insert_key_lng(
            fptr,
            cs!(c"EXTVER"),
            0,
            Some(cs!(c"Grouping Table vers. (this file)")),
            status,
        );

        /*
        if the grpname parameter value was defined (Non NULL and non zero
        length) then add the GRPNAME keyword and value
         */

        if !grpname.is_empty() && strlen_safe(grpname) > 0 {
            fits_insert_key_str(
                fptr,
                cs!(c"GRPNAME"),
                grpname,
                Some(cs!(c"Grouping Table name")),
                status,
            );
        }

        /*
        add the TNULL keywords and values for each integer column defined;
        integer null values are zero (0) for the MEMBER_POSITION and
        MEMBER_VERSION columns.
         */

        let mut i: c_int = 0;
        while i < tfields && *status == 0 {
            if fits_strcasecmp(ttype[i as usize].unwrap(), cs!(c"MEMBER_POSITION")) == 0
                || fits_strcasecmp(ttype[i as usize].unwrap(), cs!(c"MEMBER_VERSION")) == 0
            {
                int_snprintf!(&mut keyword, FLEN_KEYWORD, "TFORM{}", i + 1);
                *status =
                    fits_read_key_str(fptr, &keyword, &mut keyvalue, Some(&mut comment), status);

                int_snprintf!(&mut keyword, FLEN_KEYWORD, "TNULL{}", i + 1);

                *status =
                    fits_insert_key_lng(fptr, &keyword, 0, Some(cs!(c"Column Null Value")), status);
            }
            i += 1;
        }

        /*
        determine the correct EXTVER value for the new grouping table
        by finding the highest numbered grouping table EXTVER value
        the currently exists
         */

        extver = 1;
        while fits_movnam_hdu(fptr, ANY_HDU, cs!(c"GROUPING"), extver, status) == 0 {
            // Use a for loop to find the highest EXTVER value for GROUPING HDUs
            extver += 1;
        }

        if *status == BAD_HDU_NUM {
            *status = 0;
        }

        /*
        move back to the new grouping table HDU and update the EXTVER
        keyword value
         */

        fits_movabs_hdu(fptr, hdunum, Some(&mut hdutype), status);

        fits_modify_key_lng(
            fptr,
            cs!(c"EXTVER"),
            extver as LONGLONG,
            Some(cs!(c"&")),
            status,
        );

        break;
    } // while(0)

    *status
}

/// Change the grouping table structure of the grouping table pointed to by gfptr.
///
/// The grouptype code specifies the new structure of the table. This
/// operation only adds or removes grouping table columns, it does not add
/// or delete group members (i.e., table rows). If the grouping table already
/// has the desired structure then no operations are performed and function   
/// simply returns with a (0) success status code. If the requested structure
/// change creates new grouping table columns, then the column values for all
/// existing members will be filled with the appropriate null values.
///   GT_ID_ALL_URI  0 ==> defualt (all columns)
///   GT_ID_REF      1 ==> ID by reference
///   GT_ID_POS      2 ==> ID by position
///   GT_ID_ALL      3 ==> ID by ref. and position
///   GT_ID_REF_URI 11 ==> (1) + URI info
///   GT_ID_POS_URI 12 ==> (2) + URI info  
///
/// # Parameters
///
/// * `gfptr`     — FITS file pointer
/// * `grouptype` — code specifying the type of
/// * `status`    — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtch(
    gfptr: *mut fitsfile,
    grouptype: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);

        ffgtch_safe(gfptr, grouptype, status)
    }
}

/// Change the grouping table structure of the grouping table pointed to by gfptr.
///
/// The grouptype code specifies the new structure of the table. This
/// operation only adds or removes grouping table columns, it does not add
/// or delete group members (i.e., table rows). If the grouping table already
/// has the desired structure then no operations are performed and function   
/// simply returns with a (0) success status code. If the requested structure
/// change creates new grouping table columns, then the column values for all
/// existing members will be filled with the appropriate null values.
///   GT_ID_ALL_URI  0 ==> defualt (all columns)
///   GT_ID_REF      1 ==> ID by reference
///   GT_ID_POS      2 ==> ID by position
///   GT_ID_ALL      3 ==> ID by ref. and position
///   GT_ID_REF_URI 11 ==> (1) + URI info
///   GT_ID_POS_URI 12 ==> (2) + URI info  
///
/// # Parameters
///
/// * `gfptr`     — FITS file pointer
/// * `grouptype` — code specifying the type of
/// * `status`    — return status code
pub fn ffgtch_safe(gfptr: &mut fitsfile, grouptype: c_int, status: &mut c_int) -> c_int {
    let mut xtensionCol: c_int = 0;
    let mut extnameCol: c_int = 0;
    let mut extverCol: c_int = 0;
    let mut positionCol: c_int = 0;
    let mut locationCol: c_int = 0;
    let mut uriCol: c_int = 0;
    let mut ncols: c_int = 0;
    let mut colnum: c_int = 0;
    let nrows: c_int = 0;
    let mut grptype: c_int = 0;
    let mut i: c_int;
    let mut j: c_int;

    let intNull: c_long = 0;
    let mut tfields: c_long = 0;

    /* `char *tform[6]` / `char *ttype[6]` are 6 column pointers that in C address    */
    /* the flat backing buffers `char ttypeBuff[102]` (6*17) and `char tformBuff[54]` */
    /* (6*9). A 2-D array captures the same "6 columns of fixed width" layout.        */
    let mut ttypeBuff: [[c_char; 17]; 6] = [[0; 17]; 6];
    let mut tformBuff: [[c_char; 9]; 6] = [[0; 9]; 6];

    let charNull: [c_uchar; 1] = *b"\0"; /* unsigned char charNull[1] = {'\0'}; */

    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    if *status != 0 {
        return *status;
    }

    // `do { ... } while(0)` is modelled as a `loop { ...; break; }`; the C `continue`
    // statements (which fall through to the `while(0)` test) become `break`.
    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        /* retrieve positions of all Grouping table reserved columns */

        *status = ffgtgc(
            gfptr,
            &mut xtensionCol,
            &mut extnameCol,
            &mut extverCol,
            &mut positionCol,
            &mut locationCol,
            &mut uriCol,
            &mut grptype,
            status,
        );

        if *status != 0 {
            break;
        }

        /* determine the total number of grouping table columns */

        *status = fits_read_key_lng(
            gfptr,
            cs!(c"TFIELDS"),
            &mut tfields,
            Some(&mut comment),
            status,
        );

        /* set up the ttype and tform character buffers */
        // Each column is one row of the 2-D buffers; pass them to ffgtdc as mutable
        // slices (stack-allocated, like the C `char *ttype[6]`) so it can write each
        // column's TTYPE/TFORM string.
        {
            let mut ttype: [&mut [c_char]; 6] = ttypeBuff.each_mut().map(|c| c.as_mut_slice());
            let mut tform: [&mut [c_char]; 6] = tformBuff.each_mut().map(|c| c.as_mut_slice());

            /* define grouping table columns to be added to the configuration */

            *status = ffgtdc(
                grouptype,
                xtensionCol,
                extnameCol,
                extverCol,
                positionCol,
                locationCol,
                uriCol,
                &mut ttype,
                &mut tform,
                &mut ncols,
                status,
            );
        }

        // Re-borrow the (now populated) columns as immutable views for the code below.
        let ttype: [&[c_char]; 6] = ttypeBuff.each_ref().map(|c| c.as_slice());
        let tform: [&[c_char]; 6] = tformBuff.each_ref().map(|c| c.as_slice());

        /*
        delete any grouping tables columns that exist but do not belong to
        new desired configuration; note that we delete before creating new
        columns for (file size) efficiency reasons
          */

        match grouptype as u64 {
            GT_ID_ALL_URI => { /* no columns to be deleted in this case */ }

            GT_ID_REF => {
                if positionCol != 0 {
                    *status = fits_delete_col(gfptr, positionCol, status);
                    tfields -= 1;
                    if uriCol > positionCol {
                        uriCol -= 1;
                    }
                    if locationCol > positionCol {
                        locationCol -= 1;
                    }
                }
                if uriCol != 0 {
                    *status = fits_delete_col(gfptr, uriCol, status);
                    tfields -= 1;
                    if locationCol > uriCol {
                        locationCol -= 1;
                    }
                }
                if locationCol != 0 {
                    *status = fits_delete_col(gfptr, locationCol, status);
                }
            }

            GT_ID_POS => {
                if xtensionCol != 0 {
                    *status = fits_delete_col(gfptr, xtensionCol, status);
                    tfields -= 1;
                    if extnameCol > xtensionCol {
                        extnameCol -= 1;
                    }
                    if extverCol > xtensionCol {
                        extverCol -= 1;
                    }
                    if uriCol > xtensionCol {
                        uriCol -= 1;
                    }
                    if locationCol > xtensionCol {
                        locationCol -= 1;
                    }
                }
                if extnameCol != 0 {
                    *status = fits_delete_col(gfptr, extnameCol, status);
                    tfields -= 1;
                    if extverCol > extnameCol {
                        extverCol -= 1;
                    }
                    if uriCol > extnameCol {
                        uriCol -= 1;
                    }
                    if locationCol > extnameCol {
                        locationCol -= 1;
                    }
                }
                if extverCol != 0 {
                    *status = fits_delete_col(gfptr, extverCol, status);
                    tfields -= 1;
                    if uriCol > extverCol {
                        uriCol -= 1;
                    }
                    if locationCol > extverCol {
                        locationCol -= 1;
                    }
                }
                if uriCol != 0 {
                    *status = fits_delete_col(gfptr, uriCol, status);
                    tfields -= 1;
                    if locationCol > uriCol {
                        locationCol -= 1;
                    }
                }
                if locationCol != 0 {
                    *status = fits_delete_col(gfptr, locationCol, status);
                    tfields -= 1;
                }
            }

            GT_ID_ALL => {
                if uriCol != 0 {
                    *status = fits_delete_col(gfptr, uriCol, status);
                    tfields -= 1;
                    if locationCol > uriCol {
                        locationCol -= 1;
                    }
                }
                if locationCol != 0 {
                    *status = fits_delete_col(gfptr, locationCol, status);
                    tfields -= 1;
                }
            }

            GT_ID_REF_URI => {
                if positionCol != 0 {
                    *status = fits_delete_col(gfptr, positionCol, status);
                    tfields -= 1;
                }
            }

            GT_ID_POS_URI => {
                if xtensionCol != 0 {
                    *status = fits_delete_col(gfptr, xtensionCol, status);
                    tfields -= 1;
                    if extnameCol > xtensionCol {
                        extnameCol -= 1;
                    }
                    if extverCol > xtensionCol {
                        extverCol -= 1;
                    }
                }
                if extnameCol != 0 {
                    *status = fits_delete_col(gfptr, extnameCol, status);
                    tfields -= 1;
                    if extverCol > extnameCol {
                        extverCol -= 1;
                    }
                }
                if extverCol != 0 {
                    *status = fits_delete_col(gfptr, extverCol, status);
                    tfields -= 1;
                }
            }

            _ => {
                *status = BAD_OPTION;
                ffpmsg_str("Invalid value for grouptype parameter specified (ffgtch)");
            }
        }

        /*
        add all the new grouping table columns that were not there
        previously but are called for by the grouptype parameter
          */

        i = 0;
        while i < ncols && *status == 0 {
            *status = fits_insert_col(
                gfptr,
                tfields as c_int + i + 1,
                ttype[i as usize],
                tform[i as usize],
                status,
            );
            i += 1;
        }

        /*
        add the TNULL keywords and values for each new integer column defined;
        integer null values are zero (0) for the MEMBER_POSITION and
        MEMBER_VERSION columns. Insert a null ("/0") into each new string
        column defined: MEMBER_XTENSION, MEMBER_NAME, MEMBER_URI_TYPE and
        MEMBER_LOCATION. Note that by convention a null string is the
        TNULL value for character fields so no TNULL is required.
         */

        i = 0;
        while i < ncols && *status == 0 {
            if fits_strcasecmp(ttype[i as usize], cs!(c"MEMBER_POSITION")) == 0
                || fits_strcasecmp(ttype[i as usize], cs!(c"MEMBER_VERSION")) == 0
            {
                /* col contains int data; set TNULL and insert 0 for each col */

                *status = fits_get_colnum(
                    gfptr,
                    CASESEN as c_int,
                    ttype[i as usize],
                    &mut colnum,
                    status,
                );

                int_snprintf!(&mut keyword, FLEN_KEYWORD, "TFORM{}", colnum);

                *status =
                    fits_read_key_str(gfptr, &keyword, &mut keyvalue, Some(&mut comment), status);

                int_snprintf!(&mut keyword, FLEN_KEYWORD, "TNULL{}", colnum);

                *status = fits_insert_key_lng(
                    gfptr,
                    &keyword,
                    0,
                    Some(cs!(c"Column Null Value")),
                    status,
                );

                j = 1;
                while j <= nrows && *status == 0 {
                    *status =
                        fits_write_col_lng(gfptr, colnum, j as LONGLONG, 1, 1, &[intNull], status);
                    j += 1;
                }
            } else if fits_strcasecmp(ttype[i as usize], cs!(c"MEMBER_XTENSION")) == 0
                || fits_strcasecmp(ttype[i as usize], cs!(c"MEMBER_NAME")) == 0
                || fits_strcasecmp(ttype[i as usize], cs!(c"MEMBER_URI_TYPE")) == 0
                || fits_strcasecmp(ttype[i as usize], cs!(c"MEMBER_LOCATION")) == 0
            {
                /* new col contains character data; insert NULLs into each col */

                *status = fits_get_colnum(
                    gfptr,
                    CASESEN as c_int,
                    ttype[i as usize],
                    &mut colnum,
                    status,
                );

                j = 1;
                while j <= nrows && *status == 0
                /* WILL THIS WORK FOR VAR LENTH CHAR COLS??????*/
                {
                    *status =
                        fits_write_col_byt(gfptr, colnum, j as LONGLONG, 1, 1, &charNull, status);
                    j += 1;
                }
            }
            i += 1;
        }

        break;
    } // while(0)

    *status
}

/// Remove a grouping table, and optionally all its members.
///
/// Any groups containing the grouping table are updated, and all members (if not
/// deleted) have their GRPIDn and GRPLCn keywords updated accordingly.
/// If the (deleted) members are members of another grouping table then those
/// tables are also updated. The CHDU of the FITS file pointed to by gfptr must
/// be positioned to the grouping table to be deleted.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to group
/// * `rmopt`  — code specifying if member elements are to be deleted: OPT_RM_GPT ==>
///             remove only group table OPT_RM_ALL ==> recursively remove members and
///             their members (if groups)
/// * `status` — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtrm(gfptr: *mut fitsfile, rmopt: c_int, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);

        ffgtrm_safe(gfptr, rmopt, status)
    }
}

/// Remove a grouping table, and optionally all its members.
///
/// Any groups containing the grouping table are updated, and all members (if not
/// deleted) have their GRPIDn and GRPLCn keywords updated accordingly.
/// If the (deleted) members are members of another grouping table then those
/// tables are also updated. The CHDU of the FITS file pointed to by gfptr must
/// be positioned to the grouping table to be deleted.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to group
/// * `rmopt`  — code specifying if member elements are to be deleted: OPT_RM_GPT ==>
///             remove only group table OPT_RM_ALL ==> recursively remove members and
///             their members (if groups)
/// * `status` — return status code
pub fn ffgtrm_safe(gfptr: &mut fitsfile, rmopt: c_int, status: &mut c_int) -> c_int {
    let mut hdutype: c_int = 0;

    let mut i: c_long;
    let mut nmembers: c_long = 0;

    let mut HDU = HDUtracker::default();

    if *status != 0 {
        return *status;
    }

    /*
     remove the grouping table depending upon the rmopt parameter
    */

    match rmopt as u64 {
        OPT_RM_GPT => {
            /*
            for this option, the grouping table is deleted, but the member
            HDUs remain; in this case we only have to remove each member from
            the grouping table by calling fits_remove_member() with the
            OPT_RM_ENTRY option
               */

            /* get the number of members contained by this table */

            *status = fits_get_num_members(gfptr, &mut nmembers, status);

            /* loop over all grouping table members and remove them */

            i = nmembers;
            while i > 0 && *status == 0 {
                *status = fits_remove_member(gfptr, i, OPT_RM_ENTRY as c_int, status);
                i -= 1;
            }
        }

        OPT_RM_ALL => {
            /*
            for this option the entire Group is deleted -- this includes all
            members and their members (if grouping tables themselves). Call
            the recursive form of this function to perform the removal.
                  */

            /* add the current grouping table to the HDUtracker struct */

            HDU.nHDU = 0;

            *status = fftsad(gfptr, &mut HDU, None, None);

            /* call the recursive group remove function */

            *status = ffgtrmr(gfptr, &mut HDU, status);

            /* free the memory allocated to the HDUtracker struct */
            // In the safe-Rust port the HDUtracker owns its storage and is dropped
            // automatically, so the C per-entry free() loop is not required.
        }

        _ => {
            *status = BAD_OPTION;
            ffpmsg_str("Invalid value for the rmopt parameter specified (ffgtrm)");
        }
    }

    /*
     if all went well then unlink and delete the grouping table HDU
    */

    *status = ffgmul(gfptr, 0, status);

    *status = fits_delete_hdu(gfptr, Some(&mut hdutype), status);

    *status
}

/// Copy a grouping table, and optionally all its members, to a new FITS file.
///
/// If the cpopt is set to OPT_GCP_GPT (copy grouping table only) then the
/// existing members have their GRPIDn and GRPLCn keywords updated to reflect
/// the existance of the new group, since they now belong to another group. If
/// cpopt is set to OPT_GCP_ALL (copy grouping table and members recursively)
/// then the original members are not updated; the new grouping table is
/// modified to include only the copied member HDUs and not the original members.
///
/// Note that the recursive version of this function, ffgtcpr(), is called
/// to perform the group table copy. In the case of cpopt == OPT_GCP_GPT
/// ffgtcpr() does not actually use recursion.
///
/// # Parameters
///
/// * `infptr`  — input FITS file pointer
/// * `outfptr` — output FITS file pointer
/// * `cpopt`   — code specifying copy options: OPT_GCP_GPT (0) ==> copy only grouping
///              table OPT_GCP_ALL (2) ==> recusrively copy members and their members
///              (if groups)
/// * `status`  — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtcp(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    cpopt: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);

        ffgtcp_safe(infptr, outfptr, cpopt, status)
    }
}

/// Copy a grouping table, and optionally all its members, to a new FITS file.
///
/// If the cpopt is set to OPT_GCP_GPT (copy grouping table only) then the
/// existing members have their GRPIDn and GRPLCn keywords updated to reflect
/// the existance of the new group, since they now belong to another group. If
/// cpopt is set to OPT_GCP_ALL (copy grouping table and members recursively)
/// then the original members are not updated; the new grouping table is
/// modified to include only the copied member HDUs and not the original members.
///
/// Note that the recursive version of this function, ffgtcpr(), is called
/// to perform the group table copy. In the case of cpopt == OPT_GCP_GPT
/// ffgtcpr() does not actually use recursion.
///
/// # Parameters
///
/// * `infptr`  — input FITS file pointer
/// * `outfptr` — output FITS file pointer
/// * `cpopt`   — code specifying copy options: OPT_GCP_GPT (0) ==> copy only grouping
///              table OPT_GCP_ALL (2) ==> recusrively copy members and their members
///              (if groups)
/// * `status`  — return status code
pub fn ffgtcp_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    cpopt: c_int,
    status: &mut c_int,
) -> c_int {
    let mut HDU = HDUtracker::default();

    if *status != 0 {
        return *status;
    }

    /* make sure infptr and outfptr are not the same pointer */
    // (In safe Rust two `&mut fitsfile` are guaranteed not to alias, so this can only
    // be true if the FFI wrapper was handed the same raw pointer twice.)

    if core::ptr::eq(infptr, outfptr) {
        *status = IDENTICAL_POINTERS;
    } else {
        /* initialize the HDUtracker struct */

        HDU.nHDU = 0;

        *status = fftsad(infptr, &mut HDU, None, None);

        /*
        call the recursive form of this function to copy the grouping table.
        If the cpopt is OPT_GCP_GPT then there is actually no recursion
        performed
         */

        *status = ffgtcpr(infptr, outfptr, cpopt, &mut HDU, status);

        /* free memory allocated for the HDUtracker struct */
        // In the safe-Rust port the HDUtracker owns its storage and is dropped
        // automatically, so the C per-entry free() loop is not required.
    }

    *status
}

/// Merge two grouping tables by combining their members into a single table.
///
/// The source grouping table must be the CHDU of the fitsfile pointed to by
/// infptr, and the target grouping table must be the CHDU of the fitsfile to by
/// outfptr. All members of the source grouping table shall be copied to the
/// target grouping table. If the mgopt parameter is OPT_MRG_COPY then the source
/// grouping table continues to exist after the merge. If the mgopt parameter
/// is OPT_MRG_MOV then the source grouping table is deleted after the merge,
/// and all member HDUs are updated accordingly.
///
/// # Parameters
///
/// * `infptr`  — FITS file ptr to source grouping table
/// * `outfptr` — FITS file ptr to target grouping table
/// * `mgopt`   — code specifying merge options: OPT_MRG_COPY (0) ==> copy members to
///              target group, leaving source group in place OPT_MRG_MOV (1) ==> move
///              members to target group, source group is deleted after merge
/// * `status`  — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtmg(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    mgopt: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);

        ffgtmg_safe(infptr, outfptr, mgopt, status)
    }
}

/// Merge two grouping tables by combining their members into a single table.
///
/// The source grouping table must be the CHDU of the fitsfile pointed to by
/// infptr, and the target grouping table must be the CHDU of the fitsfile to by
/// outfptr. All members of the source grouping table shall be copied to the
/// target grouping table. If the mgopt parameter is OPT_MRG_COPY then the source
/// grouping table continues to exist after the merge. If the mgopt parameter
/// is OPT_MRG_MOV then the source grouping table is deleted after the merge,
/// and all member HDUs are updated accordingly.
///
/// # Parameters
///
/// * `infptr`  — FITS file ptr to source grouping table
/// * `outfptr` — FITS file ptr to target grouping table
/// * `mgopt`   — code specifying merge options: OPT_MRG_COPY (0) ==> copy members to
///              target group, leaving source group in place OPT_MRG_MOV (1) ==> move
///              members to target group, source group is deleted after merge
/// * `status`  — return status code
pub fn ffgtmg_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    mgopt: c_int,
    status: &mut c_int,
) -> c_int {
    let mut i: c_long;
    let mut nmembers: c_long = 0;

    // A C `fitsfile *tmpfptr = NULL;` becomes an owned, nullable handle.
    let mut tmpfptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        *status = fits_get_num_members(infptr, &mut nmembers, status);

        i = 1;
        while i <= nmembers && *status == 0 {
            *status = fits_open_member(infptr, i, &mut tmpfptr, status);
            *status = fits_add_group_member(outfptr, tmpfptr.as_deref_mut(), 0, status);

            if *status == HDU_ALREADY_MEMBER {
                *status = 0;
            }

            if let Some(f) = tmpfptr.take() {
                fits_close_file(f, status);
            }
            i += 1;
        }

        if *status != 0 {
            break;
        }

        if mgopt as u64 == OPT_MRG_MOV {
            *status = fits_remove_group(infptr, OPT_RM_GPT as c_int, status);
        }

        break;
    } // while(0)

    if let Some(f) = tmpfptr.take() {
        fits_close_file(f, status);
    }

    *status
}

/// "Compact" a group pointed to by the FITS file pointer gfptr.
///
/// This is achieved by flattening the tree structure of a group and its
/// (grouping table) members. All members HDUs of a grouping table which is
/// itself a member of the grouping table gfptr are added to gfptr. Optionally,
/// the grouping tables which are "compacted" are deleted. If the grouping
/// table contains no members that are themselves grouping tables then this
/// function performs a NOOP.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to grouping table
/// * `cmopt`  — code specifying compact options OPT_CMT_MBR (1) ==> compact only direct
///             members (if groups) OPT_CMT_MBR_DEL (11) ==> (1) + delete all compacted
///             groups
/// * `status` — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtcm(gfptr: *mut fitsfile, cmopt: c_int, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);

        ffgtcm_safe(gfptr, cmopt, status)
    }
}

/// "Compact" a group pointed to by the FITS file pointer gfptr.
///
/// This is achieved by flattening the tree structure of a group and its
/// (grouping table) members. All members HDUs of a grouping table which is
/// itself a member of the grouping table gfptr are added to gfptr. Optionally,
/// the grouping tables which are "compacted" are deleted. If the grouping
/// table contains no members that are themselves grouping tables then this
/// function performs a NOOP.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to grouping table
/// * `cmopt`  — code specifying compact options OPT_CMT_MBR (1) ==> compact only direct
///             members (if groups) OPT_CMT_MBR_DEL (11) ==> (1) + delete all compacted
///             groups
/// * `status` — return status code
pub fn ffgtcm_safe(gfptr: &mut fitsfile, cmopt: c_int, status: &mut c_int) -> c_int {
    let mut i: c_long;
    let mut nmembers: c_long = 0;

    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    let mut mfptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        if cmopt as u64 != OPT_CMT_MBR && cmopt as u64 != OPT_CMT_MBR_DEL {
            *status = BAD_OPTION;
            ffpmsg_str("Invalid value for cmopt parameter specified (ffgtcm)");
            break; // C: continue (do-while)
        }

        /* reteive the number of grouping table members */

        *status = fits_get_num_members(gfptr, &mut nmembers, status);

        /*
        loop over all the grouping table members; if the member is a
        grouping table then merge its members with the parent grouping
        table
          */

        // The loop counter is bumped at the top so the C `continue` statements (which
        // fall through to the for-loop's ++i) map to a plain Rust `continue`.
        i = 0;
        while i < nmembers && *status == 0 {
            i += 1;

            *status = fits_open_member(gfptr, i, &mut mfptr, status);

            if *status != 0 {
                continue;
            }

            *status = fits_read_key_str(
                mfptr.as_deref_mut().expect(NULL_MSG),
                cs!(c"EXTNAME"),
                &mut keyvalue,
                Some(&mut comment),
                status,
            );

            /* if no EXTNAME keyword then cannot be a grouping table */

            if *status == KEY_NO_EXIST {
                *status = 0;
                continue;
            }
            prepare_keyvalue(&mut keyvalue);

            if *status != 0 {
                continue;
            }

            /* if EXTNAME == "GROUPING" then process member as grouping table */

            if fits_strcasecmp(&keyvalue, cs!(c"GROUPING")) == 0 {
                /* merge the member (grouping table) into the grouping table */

                *status = fits_merge_groups(
                    mfptr.as_deref_mut().expect(NULL_MSG),
                    gfptr,
                    OPT_MRG_COPY as c_int,
                    status,
                );

                if let Some(f) = mfptr.take() {
                    *status = fits_close_file(f, status);
                }

                /*
                remove the member from the grouping table now that all of
                its members have been transferred; if cmopt is set to
                OPT_CMT_MBR_DEL then remove and delete the member
                   */

                if cmopt as u64 == OPT_CMT_MBR {
                    *status = fits_remove_member(gfptr, i, OPT_RM_ENTRY as c_int, status);
                } else {
                    *status = fits_remove_member(gfptr, i, OPT_RM_MBR as c_int, status);
                }
            } else {
                /* not a grouping table; just close the opened member */

                if let Some(f) = mfptr.take() {
                    *status = fits_close_file(f, status);
                }
            }
        }

        break;
    } // while(0)

    *status
}

/// Check the integrity of a grouping table to make sure that all group members
/// are accessible and all the links to other grouping tables are valid. The
/// firstfailed parameter returns the member ID of the first member HDU to fail
/// verification if positive or the first group link to fail if negative;
/// otherwise firstfailed contains a return value of 0.
///
/// # Parameters
///
/// * `gfptr`       — FITS file pointer to group
/// * `firstfailed` — Member ID (if positive) of first failed member HDU verify check or
///                  GRPID index (if negitive) of first failed group link verify check.
/// * `status`      — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtvf(
    gfptr: *mut fitsfile,
    firstfailed: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);
        let firstfailed = firstfailed.as_mut().expect(NULL_MSG);

        ffgtvf_safe(gfptr, firstfailed, status)
    }
}

/// Check the integrity of a grouping table to make sure that all group members
/// are accessible and all the links to other grouping tables are valid. The
/// firstfailed parameter returns the member ID of the first member HDU to fail
/// verification if positive or the first group link to fail if negative;
/// otherwise firstfailed contains a return value of 0.
///
/// # Parameters
///
/// * `gfptr`       — FITS file pointer to group
/// * `firstfailed` — Member ID (if positive) of first failed member HDU verify check or
///                  GRPID index (if negitive) of first failed group link verify check.
/// * `status`      — return status code
pub fn ffgtvf_safe(gfptr: &mut fitsfile, firstfailed: &mut c_long, status: &mut c_int) -> c_int {
    let mut i: c_long;
    let mut nmembers: c_long = 0;
    let mut ngroups: c_long = 0;

    let mut errstr: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    let mut fptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    *firstfailed = 0;

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        /*
        attempt to open all the members of the grouping table. We stop
        at the first member which cannot be opened (which implies that it
        cannot be located)
          */

        *status = fits_get_num_members(gfptr, &mut nmembers, status);

        i = 1;
        while i <= nmembers && *status == 0 {
            *status = fits_open_member(gfptr, i, &mut fptr, status);
            if let Some(f) = fptr.take() {
                fits_close_file(f, status);
            }
            i += 1;
        }

        /*
        if the status is non-zero from the above loop then record the
        member index that caused the error
          */

        if *status != 0 {
            *firstfailed = i;
            int_snprintf!(
                &mut errstr,
                FLEN_VALUE,
                "Group table verify failed for member {} (ffgtvf)",
                i
            );
            ffpmsg_slice(&errstr);
            break;
        }

        /*
        attempt to open all the groups linked to this grouping table. We stop
        at the first group which cannot be opened (which implies that it
        cannot be located)
          */

        *status = fits_get_num_groups(gfptr, &mut ngroups, status);

        i = 1;
        while i <= ngroups && *status == 0 {
            *status = fits_open_group(gfptr, i as c_int, &mut fptr, status);
            if let Some(f) = fptr.take() {
                fits_close_file(f, status);
            }
            i += 1;
        }

        /*
        if the status from the above loop is non-zero, then record the
        GRPIDn index of the group that caused the failure
          */

        if *status != 0 {
            *firstfailed = -i;
            int_snprintf!(
                &mut errstr,
                FLEN_VALUE,
                "Group table verify failed for GRPID index {} (ffgtvf)",
                i
            );
            ffpmsg_slice(&errstr);
            break;
        }

        break;
    } // while(0)

    *status
}

/// Open the grouping table that contains the member HDU.
///
/// The member HDU must be the CHDU of the FITS file pointed to by mfptr, and the
/// grouping table is identified by the Nth index number of the GRPIDn keywords specified in
/// the member HDU's header. The fitsfile gfptr pointer is positioned with the
/// appropriate FITS file with the grouping table as the CHDU. If the group
/// grouping table resides in a file other than the member then an attempt
/// is first made to open the file readwrite, and failing that readonly.
///
/// Note that it is possible for the GRPIDn/GRPLCn keywords in a member
/// header to be non-continuous, e.g., GRPID1, GRPID2, GRPID5, GRPID6. In
/// such cases, the grpid index value specified in the function call shall
/// identify the (grpid)th GRPID value. In the above example, if grpid == 3,
/// then the group specified by GRPID5 would be opened.
///
/// # Parameters
///
/// * `mfptr`  — FITS file pointer to the member HDU
/// * `grpid`  — group ID (GRPIDn index) within member HDU
/// * `gfptr`  — FITS file pointer to grouping table HDU
/// * `status` — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtop(
    mfptr: *mut fitsfile,
    grpid: c_int,
    gfptr: *mut *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let mfptr = mfptr.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);

        // Bridge the C `fitsfile**` output to the safe `Option<Box<fitsfile>>` interface.
        let mut group_fptr: Option<Box<fitsfile>> = None;
        let r = ffgtop_safe(mfptr, grpid, &mut group_fptr, status);
        *gfptr = group_fptr.map_or(core::ptr::null_mut(), Box::into_raw);
        r
    }
}

/// Open the grouping table that contains the member HDU.
///
/// The member HDU must be the CHDU of the FITS file pointed to by mfptr, and the
/// grouping table is identified by the Nth index number of the GRPIDn keywords specified in
/// the member HDU's header. The fitsfile gfptr pointer is positioned with the
/// appropriate FITS file with the grouping table as the CHDU. If the group
/// grouping table resides in a file other than the member then an attempt
/// is first made to open the file readwrite, and failing that readonly.
///
/// Note that it is possible for the GRPIDn/GRPLCn keywords in a member
/// header to be non-continuous, e.g., GRPID1, GRPID2, GRPID5, GRPID6. In
/// such cases, the grpid index value specified in the function call shall
/// identify the (grpid)th GRPID value. In the above example, if grpid == 3,
/// then the group specified by GRPID5 would be opened.
///
/// # Parameters
///
/// * `mfptr`  — FITS file pointer to the member HDU
/// * `grpid`  — group ID (GRPIDn index) within member HDU
/// * `gfptr`  — FITS file pointer to grouping table HDU
/// * `status` — return status code
pub fn ffgtop_safe(
    mfptr: &mut fitsfile,
    grpid: c_int,
    gfptr: &mut Option<Box<fitsfile>>,
    status: &mut c_int,
) -> c_int {
    let mut i: c_int;
    let mut found: c_int;

    let mut ngroups: c_long = 0;
    let mut grpExtver: c_long = 0;

    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut keyvalue: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut location: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut location1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let location2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    'outer: loop {
        /* set the grouping table pointer to NULL for error checking later */

        *gfptr = None;

        /*
        make sure that the group ID requested is valid ==> cannot be
        larger than the number of GRPIDn keywords in the member HDU header
          */

        *status = fits_get_num_groups(mfptr, &mut ngroups, status);

        if grpid as c_long > ngroups {
            *status = BAD_GROUP_ID;
            int_snprintf!(
                &mut comment,
                FLEN_COMMENT,
                "GRPID index {} larger total GRPID keywords {} (ffgtop)",
                grpid,
                ngroups
            );
            ffpmsg_slice(&comment);
            break 'outer;
        }

        /*
        find the (grpid)th group that the member HDU belongs to and read
        the value of the GRPID(grpid) keyword; fits_get_num_groups()
        automatically re-enumerates the GRPIDn/GRPLCn keywords to fill in
        any gaps
          */

        int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPID{}", grpid);

        *status = fits_read_key_lng(mfptr, &keyword, &mut grpExtver, Some(&mut comment), status);

        if *status != 0 {
            break 'outer;
        }

        /*
        if the value of the GRPIDn keyword is positive then the member is
        in the same FITS file as the grouping table ... else ... another
        FITS file must be opened as specified by the corresponding GRPLCn
        keyword.

        The DO WHILE loop only executes once and is used to control the
        file opening logic.
          */

        #[allow(clippy::never_loop)] // transpiled do{}while(0)
        'inner: loop {
            if grpExtver > 0 {
                /*
                the member resides in the same file as the grouping
                 table, so just reopen the grouping table file
                  */

                // SAFETY: ffreopen_safer outputs a raw *mut fitsfile (it is not yet
                // ported to the Box interface); take ownership of it here.
                let mut raw: *mut fitsfile = core::ptr::null_mut();
                *status = fits_reopen_file(mfptr, &mut raw, status);
                *gfptr = if raw.is_null() {
                    None
                } else {
                    Some(unsafe { Box::from_raw(raw) })
                };
                break 'inner;
            } else if grpExtver == 0 {
                /* a GRPIDn value of zero (0) is undefined */

                *status = BAD_GROUP_ID;
                int_snprintf!(
                    &mut comment,
                    FLEN_COMMENT,
                    "Invalid value of {} for GRPID{} (ffgtop)",
                    grpExtver,
                    grpid
                );
                ffpmsg_slice(&comment);
                break 'inner;
            }

            /*
            The GRPLCn keyword value is negative, which implies that
            the grouping table must reside in another FITS file;
            search for the corresponding GRPLCn keyword
             */

            /* set the grpExtver value positive */

            grpExtver = -grpExtver;

            /* read the GRPLCn keyword value */

            int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPLC{}", grpid);
            /* SPR 1738 */
            // SAFETY: ffgkls_safe sets tkeyvalue to a heap-allocated C string tracked in
            // the ALLOCATIONS table (it is not yet ported to an owned String); copy it out
            // and release it through the same mechanism.
            let mut tkeyvalue: *mut c_char = core::ptr::null_mut();
            *status =
                fits_read_key_longstr(mfptr, &keyword, &mut tkeyvalue, Some(&mut comment), status);
            if 0 == *status {
                unsafe {
                    let bytes =
                        cast_slice::<u8, c_char>(CStr::from_ptr(tkeyvalue).to_bytes_with_nul());
                    let n = bytes.len().min(FLEN_FILENAME);
                    keyvalue[..n].copy_from_slice(&bytes[..n]);
                    if let Some((l, c)) =
                        crate::lock_global(&ALLOCATIONS).remove(&(tkeyvalue as usize))
                    {
                        let _ = Vec::from_raw_parts(tkeyvalue, l, c);
                    }
                }
            }

            /* if the GRPLCn keyword was not found then there is a problem */

            if *status == KEY_NO_EXIST {
                *status = BAD_GROUP_ID;
                int_snprintf!(
                    &mut comment,
                    FLEN_COMMENT,
                    "Cannot find GRPLC{} keyword (ffgtop)",
                    grpid
                );
                ffpmsg_slice(&comment);
                break 'inner;
            }

            prepare_keyvalue(&mut keyvalue);

            /*
            if the GRPLCn keyword value specifies an absolute URL then
            try to open the file ...
            */

            if fits_is_url_absolute(&keyvalue) != 0 {
                ffpmsg_str("Try to open group table file as absolute URL (ffgtop)");

                *status = fits_open_file(gfptr, &keyvalue, READWRITE, status);

                /* if the open was successful then continue */

                if *status == 0 {
                    break 'inner;
                }

                /* if READWRITE failed then try opening it READONLY */

                ffpmsg_str("OK, try open group table file as READONLY (ffgtop)");

                *status = 0;
                *status = fits_open_file(gfptr, &keyvalue, READONLY, status);

                /* continue regardless of the outcome */

                break 'inner;
            }

            /*
            see if the URL gives a file path that is absolute on the
            host machine
            */

            *status = fits_url2path(&keyvalue, &mut location1, status);

            *status = fits_open_file(gfptr, &location1, READWRITE, status);

            /* if the file opened then continue */

            if *status == 0 {
                break 'inner;
            }

            /* if READWRITE failed then try opening it READONLY */

            ffpmsg_str("OK, try open group table file as READONLY (ffgtop)");

            *status = 0;
            *status = fits_open_file(gfptr, &location1, READONLY, status);

            /* if the file opened then continue */

            if *status == 0 {
                break 'inner;
            }

            /*
            the grouping table location given by GRPLCn must specify a
            relative URL ...
            */

            *status = 0;

            /* retrieve the URL information for the member HDU's file */

            // C aliases url[0]=location1, url[1]=location2; model the two-element
            // `char *url[2]` as a 2-D array of the two location buffers.
            let mut url: [[c_char; FLEN_FILENAME]; 2] = [location1, location2];

            {
                // The trailing access/iostate outputs are unused here (C passed NULL).
                let mut realaccess: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
                let mut startaccess: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
                let mut iostate: c_int = 0;
                let [u0, u1] = url.each_mut();
                *status = fits_get_url(
                    mfptr,
                    u0,
                    u1,
                    &mut realaccess,
                    &mut startaccess,
                    &mut iostate,
                    status,
                );
            }

            /*
            For each possible URL try to construct a full grouping-table URL
            and open it.
            */

            found = 0;
            *gfptr = None;
            // The loop counter is bumped at the top so the C `continue` statements map
            // to a plain Rust `continue`.
            i = 0;
            while i < 2 && found == 0 {
                let idx = i as usize;
                i += 1;

                /* the url string could be empty */

                if url[idx][0] == 0 {
                    continue;
                }

                /*
                create a full URL from the partial and the member
                HDU file URL
                 */

                *status = fits_relurl2url(&url[idx], &keyvalue, &mut location, status);

                /* if an error occured then contniue */

                if *status != 0 {
                    *status = 0;
                    continue;
                }

                /*
                if the location does not specify an access method
                then turn it into a host dependent path
                  */

                if fits_is_url_absolute(&location) == 0 {
                    *status = fits_url2path(&location, &mut url[idx], status);
                    strcpy_safe(&mut location, &url[idx]);
                }

                /* try to open the grouping table file READWRITE */

                *status = fits_open_file(gfptr, &location, READWRITE, status);

                if *status != 0 {
                    /* try to open the grouping table file READONLY */

                    ffpmsg_str("opening file as READWRITE failed (ffgtop)");
                    ffpmsg_str("OK, try to open file as READONLY (ffgtop)");
                    *status = 0;
                    *status = fits_open_file(gfptr, &location, READONLY, status);
                }

                /* either set the found flag or reset the status flag */

                if *status == 0 {
                    found = 1;
                } else {
                    *status = 0;
                }
            }

            break 'inner;
        } /* end of file opening loop */

        /* if an error occured with the file opening then exit */

        if *status != 0 {
            break 'outer;
        }

        if gfptr.is_none() {
            ffpmsg_str("Cannot open or find grouping table FITS file (ffgtop)");
            *status = GROUP_NOT_FOUND;
            break 'outer;
        }

        /* search for the grouping table in its FITS file */

        *status = fits_movnam_hdu(
            gfptr.as_deref_mut().expect(NULL_MSG),
            ANY_HDU,
            cs!(c"GROUPING"),
            grpExtver as c_int,
            status,
        );

        if *status != 0 {
            *status = GROUP_NOT_FOUND;
        }

        break 'outer;
    } // while(0)

    if *status != 0 && gfptr.is_some() {
        if let Some(f) = gfptr.take() {
            fits_close_file(f, status);
        }
        *gfptr = None;
    }

    *status
}

/// Add a member HDU to an existing grouping table.
///
/// The fitsfile pointer gfptr
/// must be positioned with the grouping table as the CHDU. The member HDU
/// may either be identifed with the fitsfile *mfptr (which must be positioned
/// to the member HDU) or the hdupos parameter (the HDU number of the member
/// HDU) if both reside in the same FITS file. The hdupos value is only used
/// if the mfptr parameter has a value of NULL (0). The new member HDU shall
/// have the appropriate GRPIDn and GRPLCn keywords created in its header.
///
/// Note that if the member HDU to be added to the grouping table is already
/// a member of the group then it will not be added a sceond time.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to grouping table HDU
/// * `mfptr`  — FITS file pointer to member HDU
/// * `hdupos` — member HDU position IF in the same file as the grouping table AND mfptr
///             == NULL
/// * `status` — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtam(
    gfptr: *mut fitsfile,
    mfptr: *mut fitsfile,
    hdupos: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);

        ffgtam_safe(gfptr, mfptr.as_mut(), hdupos, status)
    }
}

/// Add a member HDU to an existing grouping table.
///
/// The fitsfile pointer gfptr
/// must be positioned with the grouping table as the CHDU. The member HDU
/// may either be identifed with the fitsfile *mfptr (which must be positioned
/// to the member HDU) or the hdupos parameter (the HDU number of the member
/// HDU) if both reside in the same FITS file. The hdupos value is only used
/// if the mfptr parameter has a value of NULL (0). The new member HDU shall
/// have the appropriate GRPIDn and GRPLCn keywords created in its header.
///
/// Note that if the member HDU to be added to the grouping table is already
/// a member of the group then it will not be added a sceond time.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to grouping table HDU
/// * `mfptr`  — FITS file pointer to member HDU; None if the member is in the same file
///             as the grouping table
/// * `hdupos` — member HDU position IF in the same file as the grouping table AND mfptr
///             == NULL
/// * `status` — return status code
pub fn ffgtam_safe(
    gfptr: &mut fitsfile,
    mut mfptr: Option<&mut fitsfile>,
    hdupos: c_int,
    status: &mut c_int,
) -> c_int {
    /* SPR 3463: consolidates the repeated "are the member and grouping-table
    files the same?" test, matching the C's combined
    `(tmpfptr->Fptr != gfptr->Fptr) && strncmp(rootnames)`. Handles that share a
    FITSfile now compare equal by address, so the first half of the test is
    effective rather than always true. */
    fn files_differ(tmpfptr: &fitsfile, gfptr: &fitsfile, status: &mut c_int) -> bool {
        let mut tmprootname: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
        let mut grootname: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

        let tfn = cast_slice::<u8, c_char>(tmpfptr.Fptr.get_filename_as_cstr().to_bytes_with_nul());
        fits_parse_rootname(tfn, &mut tmprootname, status);
        let gfn = cast_slice::<u8, c_char>(gfptr.Fptr.get_filename_as_cstr().to_bytes_with_nul());
        fits_parse_rootname(gfn, &mut grootname, status);

        let same_fptr = core::ptr::eq(tmpfptr.Fptr.as_ptr(), gfptr.Fptr.as_ptr());
        !same_fptr && strncmp_safe(&tmprootname, &grootname, FLEN_FILENAME) != 0
    }

    let mut xtensionCol: c_int = 0;
    let mut extnameCol: c_int = 0;
    let mut extverCol: c_int = 0;
    let mut positionCol: c_int = 0;
    let mut locationCol: c_int = 0;
    let mut uriCol: c_int = 0;
    let mut memberPosition: c_int = 0;
    let mut grptype: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut useLocation: c_int = 0;
    let mut nkeys: c_int = 6;
    let mut found: c_int;
    let mut i: c_int;

    let mut memberIOstate: c_int = 0;
    let mut groupIOstate: c_int = 0;
    let mut iomode: c_int = 0;

    let mut memberExtver: c_long = 0;
    let mut groupExtver: c_long = 0;
    let mut memberID: c_long = 0;
    let mut nmembers: c_long = 0;
    let mut ngroups: c_long = 0;
    let mut grpid: c_long = 0;

    let mut memberAccess1: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut memberAccess2: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut memberFileName: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut memberLocation: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut grplc: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut memberHDUtype: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut memberExtname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let memberURI = cs!(c"URL");

    let mut groupAccess1: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut groupAccess2: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut groupFileName: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut groupLocation: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut cwd: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    let keys: [&[c_char]; 6] = [
        cs!(c"GRPNAME"),
        cs!(c"EXTVER"),
        cs!(c"EXTNAME"),
        cs!(c"TFIELDS"),
        cs!(c"GCOUNT"),
        cs!(c"EXTEND"),
    ];

    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    /* the C reuses `card` for the (discarded) comment output of the read_key
    calls; the safe wrappers want a FLEN_COMMENT buffer, so use a separate one */
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    let charNull: [c_uchar; 1] = *b"\0";

    /* the member HDU pointer used internally: either the caller's mfptr or, when
    only an HDU position was supplied, a re-opened copy of the grouping table */
    let mfptr_is_null = mfptr.is_none();
    let mut reopened: Option<Box<fitsfile>> = None;

    let mut parentStatus: c_int = 0;

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    'outer: loop {
        /*
        make sure the grouping table can be modified before proceeding
        */

        fits_file_mode(gfptr, &mut iomode, status);

        if iomode != READWRITE {
            ffpmsg_str("cannot modify grouping table (ffgtam)");
            *status = BAD_GROUP_ATTACH;
            break 'outer;
        }

        /*
        if the calling function supplied the HDU position of the member
        HDU instead of fitsfile pointer then get a fitsfile pointer
        */

        if mfptr_is_null {
            let mut raw: *mut fitsfile = core::ptr::null_mut();
            *status = fits_reopen_file(gfptr, &mut raw, status);
            // SAFETY: ffreopen_safer outputs an owned, heap-allocated fitsfile.
            reopened = if raw.is_null() {
                None
            } else {
                Some(unsafe { Box::from_raw(raw) })
            };
            if let Some(t) = reopened.as_deref_mut() {
                *status = fits_movabs_hdu(t, hdupos, Some(&mut hdutype), status);
            }

            if *status != 0 {
                break 'outer;
            }
        }

        /* tmpfptr aliases the member HDU file (reopened copy or caller's mfptr) */
        let tmpfptr: &mut fitsfile = if mfptr_is_null {
            reopened.as_deref_mut().unwrap()
        } else {
            mfptr.as_deref_mut().unwrap()
        };

        /*
        determine all the information about the member HDU that will
        be needed later; note that we establish the default values for
        all information values that are not explicitly found
        */

        *status = fits_read_key_str(
            tmpfptr,
            cs!(c"XTENSION"),
            &mut memberHDUtype,
            Some(&mut comment),
            status,
        );

        if *status == KEY_NO_EXIST {
            strcpy_safe(&mut memberHDUtype, cs!(c"PRIMARY"));
            *status = 0;
        }
        prepare_keyvalue(&mut memberHDUtype);

        *status = fits_read_key_lng(
            tmpfptr,
            cs!(c"EXTVER"),
            &mut memberExtver,
            Some(&mut comment),
            status,
        );

        if *status == KEY_NO_EXIST {
            memberExtver = 1;
            *status = 0;
        }

        *status = fits_read_key_str(
            tmpfptr,
            cs!(c"EXTNAME"),
            &mut memberExtname,
            Some(&mut comment),
            status,
        );

        if *status == KEY_NO_EXIST {
            memberExtname[0] = 0;
            *status = 0;
        }
        prepare_keyvalue(&mut memberExtname);

        fits_get_hdu_num(tmpfptr, &mut memberPosition);

        /*
        Determine if the member HDU's FITS file location needs to be
        taken into account when building its grouping table reference

        If the member location needs to be used (==> grouping table and member
        HDU reside in different files) then create an appropriate URL for
        the member HDU's file and grouping table's file. Note that the logic
        for this is rather complicated
        */

        /* SPR 3463, don't do this
        if(tmpfptr->Fptr == gfptr->Fptr)
        {  */
        /*
        member HDU and grouping table reside in the same file, no need
        to use the location information */

        /* printf ("same file\n");

        useLocation     = 0;
        memberIOstate   = 1;
        *memberFileName = 0;
        }
        else
        { */
        /*
        the member HDU and grouping table FITS file location information
        must be used.

        First determine the correct driver and file name for the group
        table and member HDU files. If either are disk files then
        construct an absolute file path for them. Finally, if both are
        disk files construct relative file paths from the group(member)
        file to the member(group) file.

        */

        /* set the USELOCATION flag to true */

        useLocation = 1;

        /*
        get the location, access type and iostate (RO, RW) of the
        member HDU file
        */

        *status = fits_get_url(
            tmpfptr,
            &mut memberFileName,
            &mut memberLocation,
            &mut memberAccess1,
            &mut memberAccess2,
            &mut memberIOstate,
            status,
        );

        /*
        if the memberFileName string is empty then use the values of
        the memberLocation string. This corresponds to a file where
        the "real" file is a temporary memory file, and we must assume
        the the application really wants the original file to be the
        group member
        */

        if strlen_safe(&memberFileName) == 0 {
            strcpy_safe(&mut memberFileName, &memberLocation);
            strcpy_safe(&mut memberAccess1, &memberAccess2);
        }

        /*
        get the location, access type and iostate (RO, RW) of the
        grouping table file
        */

        *status = fits_get_url(
            gfptr,
            &mut groupFileName,
            &mut groupLocation,
            &mut groupAccess1,
            &mut groupAccess2,
            &mut groupIOstate,
            status,
        );

        if *status != 0 {
            break 'outer;
        }

        /*
        the grouping table file must be writable to continue
        */

        if groupIOstate == 0 {
            ffpmsg_str("cannot modify grouping table (ffgtam)");
            *status = BAD_GROUP_ATTACH;
            break 'outer;
        }

        /*
        determine how to construct the resulting URLs for the member and
        group files
        */

        if fits_strcasecmp(&groupAccess1, cs!(c"file://")) != 0
            && fits_strcasecmp(&memberAccess1, cs!(c"file://")) != 0
        {
            cwd[0] = 0;
            /*
            nothing to do in this case; both the member and group files
            must be of an access type that already gives valid URLs;
            i.e., URLs that we can pass directly to the file drivers
            */
        } else {
            /*
            retrieve the Current Working Directory as a Unix-like
            URL standard string
            */

            *status = fits_get_cwd(&mut cwd, status);

            /*
            create full file path for the member HDU FITS file URL
            if it is of access type file://
            */

            if fits_strcasecmp(&memberAccess1, cs!(c"file://")) == 0 {
                if memberFileName[0] == bb(b'/') {
                    strcpy_safe(&mut memberLocation, &memberFileName);
                } else {
                    strcpy_safe(&mut memberLocation, &cwd);
                    if strlen_safe(&memberLocation) + strlen_safe(&memberFileName) + 1
                        > FLEN_FILENAME - 1
                    {
                        ffpmsg_str("member path and filename is too long (ffgtam)");
                        *status = URL_PARSE_ERROR;
                        break 'outer;
                    }
                    strcat_safe(&mut memberLocation, cs!(c"/"));
                    strcat_safe(&mut memberLocation, &memberFileName);
                }

                *status = fits_clean_url(&memberLocation, &mut memberFileName, status);
            }

            /*
            create full file path for the grouping table HDU FITS file URL
            if it is of access type file://
            */

            if fits_strcasecmp(&groupAccess1, cs!(c"file://")) == 0 {
                if groupFileName[0] == bb(b'/') {
                    strcpy_safe(&mut groupLocation, &groupFileName);
                } else {
                    strcpy_safe(&mut groupLocation, &cwd);
                    if strlen_safe(&groupLocation) + strlen_safe(&groupFileName) + 1
                        > FLEN_FILENAME - 1
                    {
                        ffpmsg_str("group path and filename is too long (ffgtam)");
                        *status = URL_PARSE_ERROR;
                        break 'outer;
                    }

                    strcat_safe(&mut groupLocation, cs!(c"/"));
                    strcat_safe(&mut groupLocation, &groupFileName);
                }

                *status = fits_clean_url(&groupLocation, &mut groupFileName, status);
            }

            /*
            if both the member and group files are disk files then
            create a relative path (relative URL) strings with
            respect to the grouping table's file and the grouping table's
            file with respect to the member HDU's file
            */

            if fits_strcasecmp(&groupAccess1, cs!(c"file://")) == 0
                && fits_strcasecmp(&memberAccess1, cs!(c"file://")) == 0
            {
                fits_url2relurl(&memberFileName, &groupFileName, &mut groupLocation, status);
                fits_url2relurl(&groupFileName, &memberFileName, &mut memberLocation, status);

                /*
                copy the resulting partial URL strings to the
                memberFileName and groupFileName variables for latter
                use in the function
                */

                strcpy_safe(&mut memberFileName, &memberLocation);
                strcpy_safe(&mut groupFileName, &groupLocation);
            }
        }
        /* beo done */
        /* }  */

        /* retrieve the grouping table's EXTVER value */

        *status = fits_read_key_lng(
            gfptr,
            cs!(c"EXTVER"),
            &mut groupExtver,
            Some(&mut comment),
            status,
        );

        /*
        if useLocation is true then make the group EXTVER value negative
        for the subsequent GRPIDn/GRPLCn matching
        */
        /* SPR 3463 change test;  WDP added test for same filename */
        /* Now, if either the Fptr values are the same, or the root filenames
        are the same, then assume these refer to the same file.
        */
        if files_differ(tmpfptr, gfptr, status) {
            groupExtver = -groupExtver;
        }

        /* retrieve the number of group members */

        *status = fits_get_num_members(gfptr, &mut nmembers, status);

        #[allow(clippy::never_loop)] // transpiled do{}while(0)
        'inner: loop {
            /*
            make sure the member HDU is not already an entry in the
            grouping table before adding it
            */

            *status = ffgmf(
                gfptr,
                &memberHDUtype,
                &memberExtname,
                memberExtver as c_int,
                memberPosition,
                Some(&memberFileName),
                &mut memberID,
                status,
            );

            if *status == MEMBER_NOT_FOUND {
                *status = 0;
            } else if *status == 0 {
                parentStatus = HDU_ALREADY_MEMBER;
                ffpmsg_str("Specified HDU is already a member of the Grouping table (ffgtam)");
                break 'inner;
            } else {
                break 'inner;
            }

            /*
            if the member HDU is not already recorded in the grouping table
            then add it
            */

            /* add a new row to the grouping table */

            *status = fits_insert_rows(gfptr, nmembers as LONGLONG, 1, status);
            nmembers += 1;

            /* retrieve the grouping table column IDs and structure type */

            *status = ffgtgc(
                gfptr,
                &mut xtensionCol,
                &mut extnameCol,
                &mut extverCol,
                &mut positionCol,
                &mut locationCol,
                &mut uriCol,
                &mut grptype,
                status,
            );

            /* fill in the member HDU data in the new grouping table row */

            if xtensionCol != 0 {
                fits_write_col_str(
                    gfptr,
                    xtensionCol,
                    nmembers as LONGLONG,
                    1,
                    1,
                    &[&memberHDUtype[..]],
                    status,
                );
            }

            if extnameCol != 0 {
                if strlen_safe(&memberExtname) != 0 {
                    fits_write_col_str(
                        gfptr,
                        extnameCol,
                        nmembers as LONGLONG,
                        1,
                        1,
                        &[&memberExtname[..]],
                        status,
                    );
                } else {
                    /* WILL THIS WORK FOR VAR LENTH CHAR COLS??????*/
                    fits_write_col_byt(
                        gfptr,
                        extnameCol,
                        nmembers as LONGLONG,
                        1,
                        1,
                        &charNull,
                        status,
                    );
                }
            }

            if extverCol != 0 {
                fits_write_col_lng(
                    gfptr,
                    extverCol,
                    nmembers as LONGLONG,
                    1,
                    1,
                    &[memberExtver],
                    status,
                );
            }

            if positionCol != 0 {
                fits_write_col_int(
                    gfptr,
                    positionCol,
                    nmembers as LONGLONG,
                    1,
                    1,
                    &[memberPosition],
                    status,
                );
            }

            if locationCol != 0 {
                /* Change the test for SPR 3463 */
                /* Now, if either the Fptr values are the same, or the root filenames
                are the same, then assume these refer to the same file.
                */
                if files_differ(tmpfptr, gfptr, status) {
                    fits_write_col_str(
                        gfptr,
                        locationCol,
                        nmembers as LONGLONG,
                        1,
                        1,
                        &[&memberFileName[..]],
                        status,
                    );
                } else {
                    /* WILL THIS WORK FOR VAR LENTH CHAR COLS??????*/
                    fits_write_col_byt(
                        gfptr,
                        locationCol,
                        nmembers as LONGLONG,
                        1,
                        1,
                        &charNull,
                        status,
                    );
                }
            }

            if uriCol != 0 {
                /* Change the test for SPR 3463 */
                /* Now, if either the Fptr values are the same, or the root filenames
                are the same, then assume these refer to the same file.
                */
                if files_differ(tmpfptr, gfptr, status) {
                    fits_write_col_str(
                        gfptr,
                        uriCol,
                        nmembers as LONGLONG,
                        1,
                        1,
                        &[memberURI],
                        status,
                    );
                } else {
                    /* WILL THIS WORK FOR VAR LENTH CHAR COLS??????*/
                    fits_write_col_byt(
                        gfptr,
                        uriCol,
                        nmembers as LONGLONG,
                        1,
                        1,
                        &charNull,
                        status,
                    );
                }
            }

            break 'inner;
        } //while(0)

        if 0 != *status {
            break 'outer;
        }
        /*
        add GRPIDn/GRPLCn keywords to the member HDU header to link
        it to the grouing table if the they do not already exist and
        the member file is RW
        */

        fits_file_mode(tmpfptr, &mut iomode, status);

        if memberIOstate == 0 || iomode != READWRITE {
            ffpmsg_str("cannot add GRPID/LC keywords to member HDU: (ffgtam)");
            ffpmsg_slice(&memberFileName);
            break 'outer;
        }

        *status = fits_get_num_groups(tmpfptr, &mut ngroups, status);

        /*
        look for the GRPID/LC keywords in the member HDU; if the keywords
        for the back-link to the grouping table already exist then no
        need to add them again
        */

        i = 1;
        found = 0;
        while i <= ngroups as c_int && found == 0 && *status == 0 {
            /* body wrapped in a loop so the C `continue` (which advances to the
            for-loop's `++i`) becomes a `break` */
            #[allow(clippy::never_loop)] // transpiled do{}while(0)
            'forbody: loop {
                int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPID{}", ngroups as c_int);
                *status =
                    fits_read_key_lng(tmpfptr, &keyword, &mut grpid, Some(&mut comment), status);

                if grpid == groupExtver {
                    if grpid < 0 {
                        /* have to make sure the GRPLCn keyword matches too */

                        int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPLC{}", ngroups as c_int);
                        /* SPR 1738 */
                        /* fits_read_key_longstr (ffgkls_safe) outputs a raw heap C
                        string tracked in ALLOCATIONS. The C reads from `mfptr`
                        here; when supplied by pointer that is the same as
                        tmpfptr, and the hdupos path does not reach this
                        negative-GRPID branch (it implies a different file). */
                        let mut tgrplc: *mut c_char = core::ptr::null_mut();
                        *status = fits_read_key_longstr(
                            tmpfptr,
                            &keyword,
                            &mut tgrplc,
                            Some(&mut comment),
                            status,
                        );
                        if 0 == *status {
                            // SAFETY: tgrplc is the heap C string from ffgkls_safe,
                            // tracked in ALLOCATIONS; copy it out then release it.
                            unsafe {
                                let cs = CStr::from_ptr(tgrplc);
                                let too_long = cs.to_bytes().len() > FLEN_FILENAME - 1;
                                if !too_long {
                                    let val = cast_slice::<u8, c_char>(cs.to_bytes_with_nul());
                                    strcpy_safe(&mut grplc, val);
                                }
                                if let Some((l, c)) =
                                    crate::lock_global(&ALLOCATIONS).remove(&(tgrplc as usize))
                                {
                                    let _ = Vec::from_raw_parts(tgrplc, l, c);
                                }
                                if too_long {
                                    ffpmsg_str("GRPLCn string is too long (ffgtam)");
                                    *status = URL_PARSE_ERROR;
                                    break 'forbody;
                                }
                            }
                        }

                        /*
                        always compare files using absolute paths
                        the presence of a non-empty cwd indicates
                        that the file names may require conversion
                        to absolute paths
                        */

                        if 0 < strlen_safe(&cwd) {
                            /* temp buffer for use in assembling abs. path(s) */
                            let mut tmp: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

                            /* make grplc absolute if necessary */
                            if fits_is_url_absolute(&grplc) == 0 {
                                fits_path2url(&grplc, FLEN_FILENAME, &mut groupLocation, status);

                                if groupLocation[0] != bb(b'/') {
                                    strcpy_safe(&mut tmp, &cwd);
                                    if strlen_safe(&tmp) + strlen_safe(&groupLocation) + 1
                                        > FLEN_FILENAME - 1
                                    {
                                        ffpmsg_str("path and group location is too long (ffgtam)");
                                        *status = URL_PARSE_ERROR;
                                        break 'forbody;
                                    }
                                    strcat_safe(&mut tmp, cs!(c"/"));
                                    strcat_safe(&mut tmp, &groupLocation);
                                    fits_clean_url(&tmp, &mut grplc, status);
                                }
                            }

                            /* make groupFileName absolute if necessary */
                            if fits_is_url_absolute(&groupFileName) == 0 {
                                fits_path2url(
                                    &groupFileName,
                                    FLEN_FILENAME,
                                    &mut groupLocation,
                                    status,
                                );

                                if groupLocation[0] != bb(b'/') {
                                    strcpy_safe(&mut tmp, &cwd);
                                    if strlen_safe(&tmp) + strlen_safe(&groupLocation) + 1
                                        > FLEN_FILENAME - 1
                                    {
                                        ffpmsg_str("path and group location is too long (ffgtam)");
                                        *status = URL_PARSE_ERROR;
                                        break 'forbody;
                                    }
                                    strcat_safe(&mut tmp, cs!(c"/"));
                                    strcat_safe(&mut tmp, &groupLocation);
                                    /*
                                    note: use groupLocation (which is not used
                                    below this block), to store the absolute
                                    file name instead of using groupFileName.
                                    The latter may be needed unaltered if the
                                    GRPLC is written below
                                    */

                                    fits_clean_url(&tmp, &mut groupLocation, status);
                                }
                            }
                        }
                        /*
                        see if the grplc value and the group file name match
                        */

                        if strcmp_safe(&grplc, &groupLocation) == 0 {
                            found = 1;
                        }
                    } else {
                        /* the match is found with GRPIDn alone */
                        found = 1;
                    }
                }

                break 'forbody;
            }

            i += 1;
        }

        /*
        if FOUND is true then no need to continue
        */

        if found != 0 {
            ffpmsg_str("HDU already has GRPID/LC keywords for group table (ffgtam)");
            break 'outer;
        }

        /*
        add the GRPID/LC keywords to the member header for this grouping
        table

        If NGROUPS == 0 then we must position the header pointer to the
        record where we want to insert the GRPID/LC keywords (the pointer
        is already correctly positioned if the above search loop activiated)
        */

        if ngroups == 0 {
            /*
            no GRPIDn/GRPLCn keywords currently exist in header so try
            to position the header pointer to a desirable position
            */

            i = 0;
            *status = KEY_NO_EXIST;
            while i < nkeys && *status == KEY_NO_EXIST {
                *status = 0;
                *status = fits_read_card(tmpfptr, keys[i as usize], &mut card, status);
                i += 1;
            }

            /* all else fails: move write pointer to end of header */

            if *status == KEY_NO_EXIST {
                *status = 0;
                fits_get_hdrspace(tmpfptr, Some(&mut nkeys), Some(&mut i), status);
                fits_read_record(tmpfptr, nkeys, Some(&mut card), status);
            }

            /* any other error status then abort */

            if *status != 0 {
                break 'outer;
            }
        }

        /*
        now that the header pointer is positioned for the GRPID/LC
        keyword insertion increment the number of group links counter for
        the member HDU
        */

        ngroups += 1;

        /*
        if the member HDU and grouping table reside in the same FITS file
        then there is no need to add a GRPLCn keyword
        */
        /* SPR 3463 change test */
        /* Now, if either the Fptr values are the same, or the root filenames
        are the same, then assume these refer to the same file.
        */
        if !files_differ(tmpfptr, gfptr, status) {
            /* add the GRPIDn keyword only */

            int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPID{}", ngroups as c_int);
            fits_insert_key_lng(
                tmpfptr,
                &keyword,
                groupExtver as LONGLONG,
                Some(cs!(c"EXTVER of Group containing this HDU")),
                status,
            );
        } else {
            /* add the GRPIDn and GRPLCn keywords */

            int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPID{}", ngroups as c_int);
            fits_insert_key_lng(
                tmpfptr,
                &keyword,
                groupExtver as LONGLONG,
                Some(cs!(c"EXTVER of Group containing this HDU")),
                status,
            );

            int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPLC{}", ngroups as c_int);
            /* SPR 1738 */
            fits_insert_key_longstr(
                tmpfptr,
                &keyword,
                &groupFileName,
                Some(cs!(c"URL of file containing Group")),
                status,
            );
            fits_write_key_longwarn(tmpfptr, status);
        }

        break 'outer;
    } //while(0)

    /* close the tmpfptr pointer if it was opened in this function */

    if mfptr_is_null && let Some(f) = reopened.take() {
        *status = fits_close_file(f, status);
    }

    *status = if 0 == *status { parentStatus } else { *status };

    *status
}

/// Return the number of member HDUs in a grouping table.
///
/// The fitsfile pointer gfptr must be positioned with the grouping table as the CHDU.
/// The number of grouping table member HDUs is just the NAXIS2 value of the grouping
/// table.
///
/// # Parameters
///
/// * `gfptr`    — FITS file pointer to grouping table
/// * `nmembers` — member count of the grouping table
/// * `status`   — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtnm(
    gfptr: *mut fitsfile,
    nmembers: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);
        let nmembers = nmembers.as_mut().expect(NULL_MSG);

        ffgtnm_safe(gfptr, nmembers, status)
    }
}

/// Return the number of member HDUs in a grouping table.
///
/// The fitsfile pointer gfptr must be positioned with the grouping table as the CHDU.
/// The number of grouping table member HDUs is just the NAXIS2 value of the grouping
/// table.
///
/// # Parameters
///
/// * `gfptr`    — FITS file pointer to grouping table
/// * `nmembers` — member count of the grouping table
/// * `status`   — return status code
pub fn ffgtnm_safe(gfptr: &mut fitsfile, nmembers: &mut c_long, status: &mut c_int) -> c_int {
    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    if *status != 0 {
        return *status;
    }

    *status = fits_read_keyword(
        gfptr,
        cs!(c"EXTNAME"),
        &mut keyvalue,
        Some(&mut comment),
        status,
    );

    if *status == KEY_NO_EXIST {
        *status = NOT_GROUP_TABLE;
    } else {
        prepare_keyvalue(&mut keyvalue);

        if fits_strcasecmp(&keyvalue, cs!(c"GROUPING")) != 0 {
            *status = NOT_GROUP_TABLE;
            ffpmsg_str("Specified HDU is not a Grouping table (ffgtnm)");
        }

        *status = fits_read_key_lng(gfptr, cs!(c"NAXIS2"), nmembers, Some(&mut comment), status);
    }

    *status
}

/// Return the number of groups to which a HDU belongs, as defined by the number
/// of GRPIDn/GRPLCn keyword records that appear in the HDU header. The
/// fitsfile pointer mfptr must be positioned with the member HDU as the CHDU.
/// Each time this function is called, the indicies of the GRPIDn/GRPLCn
/// keywords are checked to make sure they are continuous (ie no gaps) and
/// are re-enumerated to eliminate gaps if gaps are found to be present.
///
/// # Parameters
///
/// * `mfptr`   — FITS file pointer to member HDU
/// * `ngroups` — total number of groups linked to HDU
/// * `status`  — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgmng(
    mfptr: *mut fitsfile,
    ngroups: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let mfptr = mfptr.as_mut().expect(NULL_MSG);
        let ngroups = ngroups.as_mut().expect(NULL_MSG);

        ffgmng_safe(mfptr, ngroups, status)
    }
}

/// Return the number of groups to which a HDU belongs, as defined by the number
/// of GRPIDn/GRPLCn keyword records that appear in the HDU header. The
/// fitsfile pointer mfptr must be positioned with the member HDU as the CHDU.
/// Each time this function is called, the indicies of the GRPIDn/GRPLCn
/// keywords are checked to make sure they are continuous (ie no gaps) and
/// are re-enumerated to eliminate gaps if gaps are found to be present.
///
/// # Parameters
///
/// * `mfptr`   — FITS file pointer to member HDU
/// * `ngroups` — total number of groups linked to HDU
/// * `status`  — return status code
pub fn ffgmng_safe(mfptr: &mut fitsfile, ngroups: &mut c_long, status: &mut c_int) -> c_int {
    let mut offset: c_int;
    let mut index: c_int;
    let mut newIndex: c_int;
    let mut i: c_int;

    let mut grpid: c_long = 0;

    let inclist: [&[c_char]; 1] = [cs!(c"GRPID#")];
    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut newKeyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    if *status != 0 {
        return *status;
    }

    *ngroups = 0;

    /* reset the member HDU keyword counter to the beginning */

    *status = fits_read_record(mfptr, 0, Some(&mut card), status);

    /*
      search for the number of GRPIDn keywords in the member HDU header
      and count them with the ngroups variable
    */

    while *status == 0 {
        /* read the next GRPIDn keyword in the series */

        *status = fits_find_nextkey(mfptr, &inclist, 1, &[], 0, &mut card, status);

        if *status != 0 {
            continue;
        }

        *ngroups += 1;
    }

    if *status == KEY_NO_EXIST {
        *status = 0;
    }

    /*
       read each GRPIDn/GRPLCn keyword and adjust their index values so that
       there are no gaps in the index count
    */

    index = 1;
    offset = 0;
    i = 1;
    while i as c_long <= *ngroups && *status == 0 {
        int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPID{}", index);

        /* try to read the next GRPIDn keyword in the series */

        *status = fits_read_key_lng(mfptr, &keyword, &mut grpid, Some(&mut comment), status);

        /* if not found then increment the offset counter and continue */

        if *status == KEY_NO_EXIST {
            *status = 0;
            offset += 1;
        } else {
            /*
               increment the number_keys_found counter and see if the index
               of the keyword needs to be updated
            */

            i += 1;

            if offset > 0 {
                /* compute the new index for the GRPIDn/GRPLCn keywords */
                newIndex = index - offset;

                /* update the GRPIDn keyword index */

                int_snprintf!(&mut newKeyword, FLEN_KEYWORD, "GRPID{}", newIndex);
                fits_modify_name(mfptr, &keyword, &newKeyword, status);

                /* If present, update the GRPLCn keyword index */

                int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPLC{}", index);
                int_snprintf!(&mut newKeyword, FLEN_KEYWORD, "GRPLC{}", newIndex);
                /* SPR 1738 */
                // ffgkls_safe still outputs a raw, heap-allocated C string.
                let mut tkeyvalue: *mut c_char = core::ptr::null_mut();
                *status = fits_read_key_longstr(
                    mfptr,
                    &keyword,
                    &mut tkeyvalue,
                    Some(&mut comment),
                    status,
                );
                if 0 == *status {
                    fits_delete_key(mfptr, &keyword, status);
                    // SAFETY: tkeyvalue is the heap C string allocated by ffgkls_safe and
                    // tracked in ALLOCATIONS; use it then release it via that table.
                    unsafe {
                        let val =
                            cast_slice::<u8, c_char>(CStr::from_ptr(tkeyvalue).to_bytes_with_nul());
                        fits_insert_key_longstr(mfptr, &newKeyword, val, Some(&comment), status);
                        fits_write_key_longwarn(mfptr, status);
                        if let Some((l, c)) =
                            crate::lock_global(&ALLOCATIONS).remove(&(tkeyvalue as usize))
                        {
                            let _ = Vec::from_raw_parts(tkeyvalue, l, c);
                        }
                    }
                }

                if *status == KEY_NO_EXIST {
                    *status = 0;
                }
            }
        }

        index += 1;
    }

    *status
}

/// Open a grouping table member, returning a pointer to the member's FITS file
/// with the CHDU set to the member HDU. The grouping table must be the CHDU of
/// the FITS file pointed to by gfptr. The member to open is identified by its
/// row number within the grouping table (first row/member == 1).
///
/// If the member resides in a FITS file different from the grouping
/// table the member file is first opened readwrite and if this fails then
/// it is opened readonly. For access type of FILE:// the member file is
/// searched for assuming (1) an absolute path is given, (2) a path relative
/// to the CWD is given, and (3) a path relative to the grouping table file
/// but not relative to the CWD is given. If all of these fail then the
/// error FILE_NOT_FOUND is returned.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to grouping table
/// * `member` — member ID (row num) within grouping table
/// * `mfptr`  — FITS file pointer to member HDU
/// * `status` — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgmop(
    gfptr: *mut fitsfile,
    member: c_long,
    mfptr: *mut *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);
        let mfptr = mfptr.as_mut().expect(NULL_MSG);

        // Bridge the C `fitsfile**` output to the safe `Option<Box<fitsfile>>` interface.
        let mut member_fptr: Option<Box<fitsfile>> = None;
        let r = ffgmop_safe(gfptr, member, &mut member_fptr, status);
        *mfptr = member_fptr.map_or(core::ptr::null_mut(), Box::into_raw);
        r
    }
}

/// Open a grouping table member, returning a pointer to the member's FITS file
/// with the CHDU set to the member HDU. The grouping table must be the CHDU of
/// the FITS file pointed to by gfptr. The member to open is identified by its
/// row number within the grouping table (first row/member == 1).
///
/// If the member resides in a FITS file different from the grouping
/// table the member file is first opened readwrite and if this fails then
/// it is opened readonly. For access type of FILE:// the member file is
/// searched for assuming (1) an absolute path is given, (2) a path relative
/// to the CWD is given, and (3) a path relative to the grouping table file
/// but not relative to the CWD is given. If all of these fail then the
/// error FILE_NOT_FOUND is returned.
#[allow(clippy::if_same_then_else)]
// C dispatch chain: distinct conditions deliberately share an action.
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to grouping table
/// * `member` — member ID (row num) within grouping table
/// * `mfptr`  — FITS file pointer to member HDU
/// * `status` — return status code
pub fn ffgmop_safe(
    gfptr: &mut fitsfile,
    member: c_long,
    mfptr: &mut Option<Box<fitsfile>>,
    status: &mut c_int,
) -> c_int {
    let mut xtensionCol: c_int = 0;
    let mut extnameCol: c_int = 0;
    let mut extverCol: c_int = 0;
    let mut positionCol: c_int = 0;
    let mut locationCol: c_int = 0;
    let mut uriCol: c_int = 0;
    let mut grptype: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut dummy: c_int = 0;

    let mut hdupos: c_long = 0;
    let mut extver: c_long = 0;

    let mut xtension: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut extname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut uri: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut grpLocation1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut grpLocation2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut mbrLocation1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut mbrLocation2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut mbrLocation3: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut cwd: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let nstr: [c_char; 1] = [0]; /* char nstr[] = {'\0'}; the null-value string */

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    'outer: loop {
        /*
        retrieve the Grouping Convention reserved column positions within
        the grouping table
         */

        *status = ffgtgc(
            gfptr,
            &mut xtensionCol,
            &mut extnameCol,
            &mut extverCol,
            &mut positionCol,
            &mut locationCol,
            &mut uriCol,
            &mut grptype,
            status,
        );

        if *status != 0 {
            break 'outer;
        }

        /* verify the column formats */

        *status = ffvcfm(
            gfptr,
            xtensionCol,
            extnameCol,
            extverCol,
            positionCol,
            locationCol,
            uriCol,
            status,
        );

        if *status != 0 {
            break 'outer;
        }

        /*
        extract the member information from grouping table
         */
        // C uses `char *tmpPtr[1]` pointing at the destination buffer; here each
        // fits_read_col_str gets a one-element &mut[&mut[c_char]] over that buffer.

        if xtensionCol != 0 {
            *status = fits_read_col_str(
                gfptr,
                xtensionCol,
                member as LONGLONG,
                1,
                1,
                Some(&nstr),
                &mut [&mut xtension[..]],
                Some(&mut dummy),
                status,
            );

            /* convert the xtension string to a hdutype code */

            if fits_strcasecmp(&xtension, cs!(c"PRIMARY")) == 0 {
                hdutype = IMAGE_HDU;
            } else if fits_strcasecmp(&xtension, cs!(c"IMAGE")) == 0 {
                hdutype = IMAGE_HDU;
            } else if fits_strcasecmp(&xtension, cs!(c"TABLE")) == 0 {
                hdutype = ASCII_TBL;
            } else if fits_strcasecmp(&xtension, cs!(c"BINTABLE")) == 0 {
                hdutype = BINARY_TBL;
            } else {
                hdutype = ANY_HDU;
            }
        }

        if extnameCol != 0 {
            *status = fits_read_col_str(
                gfptr,
                extnameCol,
                member as LONGLONG,
                1,
                1,
                Some(&nstr),
                &mut [&mut extname[..]],
                Some(&mut dummy),
                status,
            );
        }

        if extverCol != 0 {
            *status = fits_read_col_lng(
                gfptr,
                extverCol,
                member as LONGLONG,
                1,
                1,
                0,
                core::slice::from_mut(&mut extver),
                Some(&mut dummy),
                status,
            );
        }

        if positionCol != 0 {
            *status = fits_read_col_lng(
                gfptr,
                positionCol,
                member as LONGLONG,
                1,
                1,
                0,
                core::slice::from_mut(&mut hdupos),
                Some(&mut dummy),
                status,
            );
        }

        if locationCol != 0 {
            *status = fits_read_col_str(
                gfptr,
                locationCol,
                member as LONGLONG,
                1,
                1,
                Some(&nstr),
                &mut [&mut mbrLocation1[..]],
                Some(&mut dummy),
                status,
            );
        }

        if uriCol != 0 {
            *status = fits_read_col_str(
                gfptr,
                uriCol,
                member as LONGLONG,
                1,
                1,
                Some(&nstr),
                &mut [&mut uri[..]],
                Some(&mut dummy),
                status,
            );
        }

        if *status != 0 {
            break 'outer;
        }

        /*
        decide what FITS file the member HDU resides in and open the file
        using the fitsfile* pointer mfptr; note that this logic is rather
        complicated and is based primiarly upon if a URL specifier is given
        for the member file in the grouping table
         */

        match grptype as u64 {
            GT_ID_POS | GT_ID_REF | GT_ID_ALL => {
                /*
                no location information is given so we must assume that the
                member HDU resides in the same FITS file as the grouping table
                  */

                // SAFETY: ffreopen_safer still outputs a raw *mut fitsfile.
                let mut raw: *mut fitsfile = core::ptr::null_mut();
                *status = fits_reopen_file(gfptr, &mut raw, status);
                *mfptr = if raw.is_null() {
                    None
                } else {
                    Some(unsafe { Box::from_raw(raw) })
                };
            }

            GT_ID_REF_URI | GT_ID_POS_URI | GT_ID_ALL_URI => {
                /*
                The member location column exists. Determine if the member
                resides in the same file as the grouping table or in a
                separate file; open the member file in either case
                  */

                if strlen_safe(&mbrLocation1) == 0 {
                    /*
                    since no location information was given we must assume
                    that the member is in the same FITS file as the grouping
                    table
                     */
                    // SAFETY: ffreopen_safer still outputs a raw *mut fitsfile.
                    let mut raw: *mut fitsfile = core::ptr::null_mut();
                    *status = fits_reopen_file(gfptr, &mut raw, status);
                    *mfptr = if raw.is_null() {
                        None
                    } else {
                        Some(unsafe { Box::from_raw(raw) })
                    };
                } else {
                    #[allow(clippy::never_loop)] // transpiled do{}while(0)
                    'inner: loop {
                        /*
                        make sure the location specifiation is "URL"; we cannot
                        decode any other URI types at this time
                          */

                        if fits_strcasecmp(&uri, cs!(c"URL")) != 0 {
                            *status = FILE_NOT_OPENED;
                            let uri_str = CStr::from_bytes_until_nul(cast_slice(&uri))
                                .unwrap()
                                .to_str()
                                .unwrap();
                            int_snprintf!(
                                &mut card,
                                FLEN_CARD,
                                "Cannot open member HDU file with URI type {} (ffgmop)",
                                uri_str
                            );
                            ffpmsg_slice(&card);
                            break 'inner;
                        }

                        /*
                        The location string for the member is not NULL, so it
                        does not necessially reside in the same FITS file as the
                        grouping table.

                        Three cases are attempted for opening the member's file
                        in the following order:

                        1. The URL given for the member's file is absolute (i.e.,
                        access method supplied); try to open the member

                        2. The URL given for the member's file is not absolute but
                        is an absolute file path; try to open the member as a file
                        after the file path is converted to a host-dependent form

                        3. The URL given for the member's file is not absolute
                        and is given as a relative path to the location of the
                        grouping table's file. Create an absolute URL using the
                        grouping table's file URL and try to open the member.

                        If all three cases fail then an error is returned. In each
                        case the file is first opened in read/write mode and failing
                        that readonly mode.

                        The following loop is only used as a mechanism to break
                        out when the proper file opening method is found
                          */

                        /*
                        CASE 1:

                        See if the member URL is absolute (i.e., includes a
                        access directive) and if so open the file
                         */

                        if fits_is_url_absolute(&mbrLocation1) != 0 {
                            /*
                            the URL must specify an access method, which
                            implies that its an absolute reference

                            regardless of the access method, pass the whole
                            URL to the open function for processing
                              */

                            ffpmsg_str("member URL is absolute, try open R/W (ffgmop)");

                            *status = fits_open_file(mfptr, &mbrLocation1, READWRITE, status);

                            if *status == 0 {
                                break 'inner;
                            }

                            *status = 0;

                            /* now try to open file using full URL specs in readonly mode */

                            ffpmsg_str("OK, now try to open read-only (ffgmop)");

                            *status = fits_open_file(mfptr, &mbrLocation1, READONLY, status);

                            /* break from loop regardless of status */

                            break 'inner;
                        }

                        /*
                        CASE 2:

                        If we got this far then the member URL location
                        has no access type ==> FILE:// Try to open the member
                        file using the URL as is, i.e., assume that it is given
                        as absolute, if it starts with a '/' character
                         */

                        ffpmsg_str("Member URL is of type FILE (ffgmop)");

                        if mbrLocation1[0] == bb(b'/') {
                            ffpmsg_str("Member URL specifies abs file path (ffgmop)");

                            /* convert the URL path to a host dependent path */

                            *status = fits_url2path(&mbrLocation1, &mut mbrLocation2, status);

                            ffpmsg_str("Try to open member URL in R/W mode (ffgmop)");

                            *status = fits_open_file(mfptr, &mbrLocation2, READWRITE, status);

                            if *status == 0 {
                                break 'inner;
                            }

                            *status = 0;

                            /*
                            now try to open file using the URL as an absolute
                            path in readonly mode
                              */

                            ffpmsg_str("OK, now try to open read-only (ffgmop)");

                            *status = fits_open_file(mfptr, &mbrLocation2, READONLY, status);

                            /* break from the loop regardless of the status */

                            break 'inner;
                        }

                        /*
                        CASE 3:

                        If we got this far then the URL does not specify an
                        absoulte file path or URL with access method. Since
                        the path to the group table's file is (obviously) valid
                        for the CWD, create a full location string for the
                        member HDU using the grouping table URL as a basis

                        The only problem is that the grouping table file might
                        have two URLs, the original one used to open it and
                        the one that points to the real file being accessed
                        (i.e., a file accessed via HTTP but transferred to a
                        local disk file). Have to attempt to build a URL to
                        the member HDU file using both of these URLs if
                        defined.
                         */

                        ffpmsg_str("Try to open member file as relative URL (ffgmop)");

                        /* get the URL information for the grouping table file */
                        // (the access/iostate outputs are unused here -- C passed NULL)
                        let mut access1: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
                        let mut access2: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
                        let mut iostate: c_int = 0;
                        *status = fits_get_url(
                            gfptr,
                            &mut grpLocation1,
                            &mut grpLocation2,
                            &mut access1,
                            &mut access2,
                            &mut iostate,
                            status,
                        );

                        /*
                        if the "real" grouping table file URL is defined then
                        build a full url for the member HDU file using it
                        and try to open the member HDU file
                         */

                        if grpLocation1[0] != 0 {
                            /* make sure the group location is absolute */
                            if fits_is_url_absolute(&grpLocation1) == 0
                                && grpLocation1[0] != bb(b'/')
                            {
                                fits_get_cwd(&mut cwd, status);
                                strcat_safe(&mut cwd, cs!(c"/"));
                                if strlen_safe(&cwd) + strlen_safe(&grpLocation1) + 1
                                    > FLEN_FILENAME - 1
                                {
                                    ffpmsg_str("cwd and group location1 is too long (ffgmop)");
                                    *status = URL_PARSE_ERROR;
                                    break 'inner;
                                }
                                strcat_safe(&mut cwd, &grpLocation1);
                                strcpy_safe(&mut grpLocation1, &cwd);
                            }

                            /* create a full URL for the member HDU file */
                            *status = fits_relurl2url(
                                &grpLocation1,
                                &mbrLocation1,
                                &mut mbrLocation2,
                                status,
                            );

                            if *status != 0 {
                                break 'inner;
                            }

                            /*
                            if the URL does not have an access method given then
                            translate it into a host dependent file path
                              */
                            if fits_is_url_absolute(&mbrLocation2) == 0 {
                                *status = fits_url2path(&mbrLocation2, &mut mbrLocation3, status);
                                strcpy_safe(&mut mbrLocation2, &mbrLocation3);
                            }

                            /* try to open the member file READWRITE */
                            *status = fits_open_file(mfptr, &mbrLocation2, READWRITE, status);
                            if *status == 0 {
                                break 'inner;
                            }
                            *status = 0;

                            /* now try to open in readonly mode */
                            ffpmsg_str("now try to open file as READONLY (ffgmop)");
                            *status = fits_open_file(mfptr, &mbrLocation2, READONLY, status);
                            if *status == 0 {
                                break 'inner;
                            }
                            *status = 0;
                        }

                        /*
                           if we got this far then either the "real" grouping table
                           file URL was not defined or all attempts to open the
                           resulting member HDU file URL failed.

                           if the "original" grouping table file URL is defined then
                           build a full url for the member HDU file using it
                           and try to open the member HDU file
                        */

                        if grpLocation2[0] != 0 {
                            /* make sure the group location is absolute */
                            if fits_is_url_absolute(&grpLocation2) == 0
                                && grpLocation2[0] != bb(b'/')
                            {
                                fits_get_cwd(&mut cwd, status);
                                if strlen_safe(&cwd) + strlen_safe(&grpLocation2) + 1
                                    > FLEN_FILENAME - 1
                                {
                                    ffpmsg_str("cwd and group location2 is too long (ffgmop)");
                                    *status = URL_PARSE_ERROR;
                                    break 'inner;
                                }
                                strcat_safe(&mut cwd, cs!(c"/"));
                                strcat_safe(&mut cwd, &grpLocation2);
                                strcpy_safe(&mut grpLocation2, &cwd);
                            }

                            /* create an absolute URL for the member HDU file */
                            *status = fits_relurl2url(
                                &grpLocation2,
                                &mbrLocation1,
                                &mut mbrLocation2,
                                status,
                            );
                            if *status != 0 {
                                break 'inner;
                            }

                            /*
                            if the URL does not have an access method given then
                            translate it into a host dependent file path
                              */
                            if fits_is_url_absolute(&mbrLocation2) == 0 {
                                *status = fits_url2path(&mbrLocation2, &mut mbrLocation3, status);
                                strcpy_safe(&mut mbrLocation2, &mbrLocation3);
                            }

                            /* try to open the member file READWRITE */
                            *status = fits_open_file(mfptr, &mbrLocation2, READWRITE, status);
                            if *status == 0 {
                                break 'inner;
                            }
                            *status = 0;

                            /* now try to open in readonly mode */
                            ffpmsg_str("now try to open file as READONLY (ffgmop)");
                            *status = fits_open_file(mfptr, &mbrLocation2, READONLY, status);
                            if *status == 0 {
                                break 'inner;
                            }
                            *status = 0;
                        }

                        /*
                        if we got this far then the member HDU file could not
                        be opened using any method. Log the error.
                         */

                        ffpmsg_str("Cannot open member HDU FITS file (ffgmop)");
                        *status = MEMBER_NOT_FOUND;

                        break 'inner;
                    } // while(0)
                }
            }

            _ => { /* no default action */ }
        }

        if *status != 0 {
            break 'outer;
        }

        /*
        attempt to locate the member HDU within its FITS file as determined
        and opened above
         */

        match grptype as u64 {
            GT_ID_POS | GT_ID_POS_URI => {
                /*
                  try to find the member hdu in the the FITS file pointed to
                  by mfptr based upon its HDU posistion value. Note that is
                  impossible to verify if the HDU is actually the correct HDU due
                  to a lack of information.
                */
                *status = fits_movabs_hdu(
                    mfptr.as_deref_mut().expect(NULL_MSG),
                    hdupos as c_int,
                    Some(&mut hdutype),
                    status,
                );
            }

            GT_ID_REF | GT_ID_REF_URI => {
                /*
                      try to find the member hdu in the FITS file pointed to
                by mfptr based upon its XTENSION, EXTNAME and EXTVER keyword
                values
                       */
                *status = fits_movnam_hdu(
                    mfptr.as_deref_mut().expect(NULL_MSG),
                    hdutype,
                    &extname,
                    extver as c_int,
                    status,
                );

                if *status == BAD_HDU_NUM {
                    *status = MEMBER_NOT_FOUND;
                    ffpmsg_str("Cannot find specified member HDU (ffgmop)");
                }

                /*
                   if the above function returned without error then the
                   mfptr is pointed to the member HDU
                */
            }

            GT_ID_ALL | GT_ID_ALL_URI => {
                /*
                if the member entry has reference information then use it
                (ID by reference is safer than ID by position) else use the
                position information
                  */

                if strlen_safe(&xtension) > 0 && strlen_safe(&extname) > 0 && extver > 0 {
                    /* valid reference info exists so use it */

                    /* try to find the member hdu in the grouping table's file */

                    *status = fits_movnam_hdu(
                        mfptr.as_deref_mut().expect(NULL_MSG),
                        hdutype,
                        &extname,
                        extver as c_int,
                        status,
                    );

                    if *status == BAD_HDU_NUM {
                        *status = MEMBER_NOT_FOUND;
                        ffpmsg_str("Cannot find specified member HDU (ffgmop)");
                    }
                } else {
                    *status = fits_movabs_hdu(
                        mfptr.as_deref_mut().expect(NULL_MSG),
                        hdupos as c_int,
                        Some(&mut hdutype),
                        status,
                    );
                    if *status == END_OF_FILE {
                        *status = MEMBER_NOT_FOUND;
                    }
                }
            }

            _ => { /* no default action */ }
        }

        break 'outer;
    } // while(0)

    if *status != 0
        && let Some(f) = mfptr.take()
    {
        fits_close_file(f, status);
    }

    *status
}

/// Copy a member HDU of a grouping table to a new FITS file. The grouping table
/// must be the CHDU of the FITS file pointed to by gfptr. The copy of the
/// group member shall be appended to the end of the FITS file pointed to by
/// mfptr. If the cpopt parameter is set to OPT_MCP_ADD then the copy of the
/// member is added to the grouping table as a new member, if OPT_MCP_NADD
/// then the copied member is not added to the grouping table, and if
/// OPT_MCP_REPL then the copied member is used to replace the original member.
/// The copied member HDU also has its EXTVER value updated so that its
/// combination of XTENSION, EXTNAME and EXVTER is unique within its new
/// FITS file.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to group
/// * `mfptr`  — FITS file pointer to new member FITS file
/// * `member` — member ID (row num) within grouping table
/// * `cpopt`  — code specifying copy options: OPT_MCP_ADD (0) ==> add copied member to
///             the grouping table OPT_MCP_NADD (1) ==> do not add member copy to the
///             grouping table OPT_MCP_REPL (2) ==> replace current member entry with
///             member copy
/// * `status` — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgmcp(
    gfptr: *mut fitsfile,
    mfptr: *mut fitsfile,
    member: c_long,
    cpopt: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let mfptr = mfptr.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);

        ffgmcp_safe(gfptr, mfptr, member, cpopt, status)
    }
}

/// Copy a member HDU of a grouping table to a new FITS file. The grouping table
/// must be the CHDU of the FITS file pointed to by gfptr. The copy of the
/// group member shall be appended to the end of the FITS file pointed to by
/// mfptr. If the cpopt parameter is set to OPT_MCP_ADD then the copy of the
/// member is added to the grouping table as a new member, if OPT_MCP_NADD
/// then the copied member is not added to the grouping table, and if
/// OPT_MCP_REPL then the copied member is used to replace the original member.
/// The copied member HDU also has its EXTVER value updated so that its
/// combination of XTENSION, EXTNAME and EXVTER is unique within its new
/// FITS file.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to group
/// * `mfptr`  — FITS file pointer to new member FITS file
/// * `member` — member ID (row num) within grouping table
/// * `cpopt`  — code specifying copy options: OPT_MCP_ADD (0) ==> add copied member to
///             the grouping table OPT_MCP_NADD (1) ==> do not add member copy to the
///             grouping table OPT_MCP_REPL (2) ==> replace current member entry with
///             member copy
/// * `status` — return status code
pub fn ffgmcp_safe(
    gfptr: &mut fitsfile,
    mfptr: &mut fitsfile,
    member: c_long,
    cpopt: c_int,
    status: &mut c_int,
) -> c_int {
    let mut numkeys: c_int = 0;
    let mut keypos: c_int = 0;
    let mut hdunum: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut i: c_int;

    let incList: [&[c_char]; 2] = [cs!(c"GRPID#"), cs!(c"GRPLC#")];
    let mut extname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    /* C declares keyname/value as FLEN_CARD; the safe fits_read_keyn wants
    FLEN_KEYWORD/FLEN_VALUE buffers, which are large enough for what it writes */
    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut value: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    let mut tmpfptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    'outer: loop {
        /* open the member HDU to be copied */

        *status = fits_open_member(gfptr, member, &mut tmpfptr, status);

        if *status != 0 {
            break 'outer;
        }

        /*
        if the member is a grouping table then copy it with a call to
        fits_copy_group() using the "copy only the grouping table" option

        if it is not a grouping table then copy the hdu with fits_copy_hdu()
        remove all GRPIDn and GRPLCn keywords, and update the EXTVER keyword
        value
        */

        /* get the member HDU's EXTNAME value */

        *status = fits_read_key_str(
            tmpfptr.as_deref_mut().unwrap(),
            cs!(c"EXTNAME"),
            &mut extname,
            Some(&mut comment),
            status,
        );

        /* if no EXTNAME value was found then set the extname to a null string */

        if *status == KEY_NO_EXIST {
            extname[0] = 0;
            *status = 0;
        } else if *status != 0 {
            break 'outer;
        }

        prepare_keyvalue(&mut extname);

        /* if a grouping table then copy with fits_copy_group() */

        if fits_strcasecmp(&extname, cs!(c"GROUPING")) == 0 {
            *status = fits_copy_group(
                tmpfptr.as_deref_mut().unwrap(),
                mfptr,
                OPT_GCP_GPT as c_int,
                status,
            );
        } else {
            /* copy the non-grouping table HDU the conventional way */

            *status = fits_copy_hdu(tmpfptr.as_deref_mut().unwrap(), mfptr, 0, status);

            fits_read_record(mfptr, 0, Some(&mut card), status);

            /* delete all the GRPIDn and GRPLCn keywords in the copied HDU */

            while *status == 0 {
                *status = fits_find_nextkey(mfptr, &incList, 2, &[], 0, &mut card, status);
                *status = fits_get_hdrpos(mfptr, Some(&mut numkeys), Some(&mut keypos), status);
                /* SPR 1738 */
                *status = fits_read_keyn(
                    mfptr,
                    keypos - 1,
                    &mut keyname,
                    &mut value,
                    Some(&mut comment),
                    status,
                );
                *status = fits_read_record(mfptr, keypos - 1, Some(&mut card), status);
                *status = fits_delete_key(mfptr, &keyname, status);
            }

            if *status == KEY_NO_EXIST {
                *status = 0;
            }
            if *status != 0 {
                break 'outer;
            }
        }

        /*
        if the member HDU does not have an EXTNAME keyword then add one
        with a default value
        */

        if strlen_safe(&extname) == 0 {
            if fits_get_hdu_num(tmpfptr.as_deref_mut().unwrap(), &mut hdunum) == 1 {
                strcpy_safe(&mut extname, cs!(c"PRIMARY"));
                *status = fits_write_key_str(
                    mfptr,
                    cs!(c"EXTNAME"),
                    &extname,
                    Some(cs!(c"HDU was Formerly a Primary Array")),
                    status,
                );
            } else {
                strcpy_safe(&mut extname, cs!(c"DEFAULT"));
                *status = fits_write_key_str(
                    mfptr,
                    cs!(c"EXTNAME"),
                    &extname,
                    Some(cs!(c"default EXTNAME set by CFITSIO")),
                    status,
                );
            }
        }

        /*
        update the member HDU's EXTVER value (add it if not present)
        */

        fits_get_hdu_num(mfptr, &mut hdunum);
        fits_get_hdu_type(mfptr, &mut hdutype, status);

        /* set the EXTVER value to 0 for now */

        *status = fits_modify_key_lng(mfptr, cs!(c"EXTVER"), 0, None, status);

        /* if the EXTVER keyword was not found then add it */

        if *status == KEY_NO_EXIST {
            *status = 0;
            *status = fits_read_key_str(
                mfptr,
                cs!(c"EXTNAME"),
                &mut extname,
                Some(&mut comment),
                status,
            );
            *status = fits_insert_key_lng(
                mfptr,
                cs!(c"EXTVER"),
                0,
                Some(cs!(c"Extension version ID")),
                status,
            );
        }

        if *status != 0 {
            break 'outer;
        }

        /* find the first available EXTVER value for the copied HDU */

        i = 1;
        while fits_movnam_hdu(mfptr, hdutype, &extname, i, status) == 0 {
            i += 1;
        }

        *status = 0;

        fits_movabs_hdu(mfptr, hdunum, Some(&mut hdutype), status);

        /* reset the copied member HDUs EXTVER value */

        *status = fits_modify_key_lng(mfptr, cs!(c"EXTVER"), i as LONGLONG, None, status);

        /*
        perform member copy operations that are dependent upon the cpopt
        parameter value
        */

        match cpopt as u64 {
            OPT_MCP_ADD => {
                /*
                add the copied member to the grouping table, leaving the
                entry for the original member in place
                */

                *status = fits_add_group_member(gfptr, Some(&mut *mfptr), 0, status);
            }

            OPT_MCP_NADD => {
                /*
                nothing to do for this copy option
                */
            }

            OPT_MCP_REPL => {
                /*
                remove the original member from the grouping table and add the
                copied member in its place
                */

                *status = fits_remove_member(gfptr, member, OPT_RM_ENTRY as c_int, status);
                *status = fits_add_group_member(gfptr, Some(&mut *mfptr), 0, status);
            }

            _ => {
                *status = BAD_OPTION;
                ffpmsg_str("Invalid value specified for the cmopt parameter (ffgmcp)");
            }
        }

        break 'outer;
    } //while(0)

    if let Some(f) = tmpfptr.take() {
        fits_close_file(f, status);
    }

    *status
}

/// Transfer a group member from one grouping table to another. The source
/// grouping table must be the CHDU of the fitsfile pointed to by infptr, and
/// the destination grouping table must be the CHDU of the fitsfile to by
/// outfptr. If the tfopt parameter is OPT_MCP_ADD then the member is made a
/// member of the target group and remains a member of the source group. If
/// the tfopt parameter is OPT_MCP_MOV then the member is deleted from the
/// source group after the transfer to the destination group. The member to be
/// transfered is identified by its row number within the source grouping table.
///
/// # Parameters
///
/// * `infptr`  — FITS file pointer to source grouping table
/// * `outfptr` — FITS file pointer to target grouping table
/// * `member`  — member ID within source grouping table
/// * `tfopt`   — code specifying transfer opts: OPT_MCP_ADD (0) ==> copy member to
///              dest. OPT_MCP_MOV (3) ==> move member to dest.
/// * `status`  — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgmtf(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    member: c_long,
    tfopt: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);

        ffgmtf_safe(infptr, outfptr, member, tfopt, status)
    }
}

/// Transfer a group member from one grouping table to another. The source
/// grouping table must be the CHDU of the fitsfile pointed to by infptr, and
/// the destination grouping table must be the CHDU of the fitsfile to by
/// outfptr. If the tfopt parameter is OPT_MCP_ADD then the member is made a
/// member of the target group and remains a member of the source group. If
/// the tfopt parameter is OPT_MCP_MOV then the member is deleted from the
/// source group after the transfer to the destination group. The member to be
/// transfered is identified by its row number within the source grouping table.
///
/// # Parameters
///
/// * `infptr`  — FITS file pointer to source grouping table
/// * `outfptr` — FITS file pointer to target grouping table
/// * `member`  — member ID within source grouping table
/// * `tfopt`   — code specifying transfer opts: OPT_MCP_ADD (0) ==> copy member to
///              dest. OPT_MCP_MOV (3) ==> move member to dest.
/// * `status`  — return status code
pub fn ffgmtf_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    member: c_long,
    tfopt: c_int,
    status: &mut c_int,
) -> c_int {
    let mut mfptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    if tfopt != OPT_MCP_MOV as c_int && tfopt != OPT_MCP_ADD as c_int {
        *status = BAD_OPTION;
        ffpmsg_str("Invalid value specified for the tfopt parameter (ffgmtf)");
    } else {
        /* open the member of infptr to be transfered */

        *status = fits_open_member(infptr, member, &mut mfptr, status);

        /* add the member to the outfptr grouping table */

        *status = fits_add_group_member(outfptr, mfptr.as_deref_mut(), 0, status);

        /* close the member HDU */

        if let Some(f) = mfptr.take() {
            *status = fits_close_file(f, status);
        }

        /*
        if the tfopt is "move member" then remove it from the infptr
        grouping table
        */

        if tfopt == OPT_MCP_MOV as c_int {
            *status = fits_remove_member(infptr, member, OPT_RM_ENTRY as c_int, status);
        }
    }

    *status
}

/// Remove a member HDU from a grouping table. The fitsfile pointer gfptr must
/// be positioned with the grouping table as the CHDU, and the member to
/// delete is identified by its row number in the table (first member == 1).
/// The rmopt parameter determines if the member entry is deleted from the
/// grouping table (in which case GRPIDn and GRPLCn keywords in the member
/// HDU's header shall be updated accordingly) or if the member HDU shall
/// itself be removed from its FITS file.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to group table
/// * `member` — member ID (row num) in the group
/// * `rmopt`  — code specifying the delete option: OPT_RM_ENTRY ==> delete the member
///             entry OPT_RM_MBR ==> delete entry and member HDU
/// * `status` — return status code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgmrm(
    gfptr: *mut fitsfile,
    member: c_long,
    rmopt: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let gfptr = gfptr.as_mut().expect(NULL_MSG);

        ffgmrm_safe(gfptr, member, rmopt, status)
    }
}

/// Remove a member HDU from a grouping table. The fitsfile pointer gfptr must
/// be positioned with the grouping table as the CHDU, and the member to
/// delete is identified by its row number in the table (first member == 1).
/// The rmopt parameter determines if the member entry is deleted from the
/// grouping table (in which case GRPIDn and GRPLCn keywords in the member
/// HDU's header shall be updated accordingly) or if the member HDU shall
/// itself be removed from its FITS file.
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to group table
/// * `member` — member ID (row num) in the group
/// * `rmopt`  — code specifying the delete option: OPT_RM_ENTRY ==> delete the member
///             entry OPT_RM_MBR ==> delete entry and member HDU
/// * `status` — return status code
pub fn ffgmrm_safe(
    gfptr: &mut fitsfile,
    member: c_long,
    rmopt: c_int,
    status: &mut c_int,
) -> c_int {
    let mut found: c_int;
    let mut hdutype: c_int = 0;
    let mut index: c_int;
    let mut iomode: c_int = 0;

    let mut i: c_long;
    let mut ngroups: c_long = 0;
    let mut nmembers: c_long = 0;
    let mut groupExtver: c_long = 0;
    let mut grpid: c_long = 0;

    let mut grpLocation1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut grpLocation2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut grpLocation3: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut cwd: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    /* SPR 1738 This can now be longer */
    let mut grplc: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut mrootname: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut grootname: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    // The C passes `card` (FLEN_CARD) as the comment buffer to the fits_read_key_*
    // helpers; those want a FLEN_COMMENT buffer, so use a dedicated one. The value
    // is discarded either way.
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    // (the access/iostate outputs of fits_get_url are unused here -- C passed NULL)
    let mut access1: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut access2: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut iostate: c_int = 0;

    let mut mfptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    'outer: loop {
        /*
        make sure the grouping table can be modified before proceeding
        */

        fits_file_mode(gfptr, &mut iomode, status);

        if iomode != READWRITE {
            ffpmsg_str("cannot modify grouping table (ffgtam)");
            *status = BAD_GROUP_DETACH;
            break 'outer;
        }

        /* open the group member to be deleted and get its IOstatus*/

        *status = fits_open_member(gfptr, member, &mut mfptr, status);

        // In this port fits_open_member yields None on failure (it closes the
        // partially-opened handle). The C would then cascade no-ops on *status>0
        // down to the closing while(0); breaking out here when no member was
        // opened is equivalent (and avoids a NULL deref).
        if mfptr.is_none() {
            break 'outer;
        }

        *status = fits_file_mode(mfptr.as_deref_mut().expect(NULL_MSG), &mut iomode, status);

        /*
           if the member HDU is to be deleted then call fits_unlink_member()
           to remove it from all groups to which it belongs (including
           this one) and then delete it. Note that if the member is a
           grouping table then we have to recursively call fits_remove_member()
           for each member of the member before we delete the member itself.
        */

        if rmopt as u64 == OPT_RM_MBR {
            /* cannot delete a PHDU */
            if fits_get_hdu_num(mfptr.as_deref_mut().expect(NULL_MSG), &mut hdutype) == 1 {
                *status = BAD_HDU_NUM;
                break 'outer;
            }

            /* determine if the member HDU is itself a grouping table */

            *status = fits_read_key_str(
                mfptr.as_deref_mut().expect(NULL_MSG),
                cs!(c"EXTNAME"),
                &mut keyvalue,
                Some(&mut comment),
                status,
            );

            /* if no EXTNAME is found then the HDU cannot be a grouping table */

            if *status == KEY_NO_EXIST {
                keyvalue[0] = 0;
                *status = 0;
            }
            prepare_keyvalue(&mut keyvalue);

            /* Any other error is a reason to abort */

            if *status != 0 {
                break 'outer;
            }

            /* if the EXTNAME == GROUPING then the member is a grouping table */

            if fits_strcasecmp(&keyvalue, cs!(c"GROUPING")) == 0 {
                /* remove each of the grouping table members */

                *status = fits_get_num_members(
                    mfptr.as_deref_mut().expect(NULL_MSG),
                    &mut nmembers,
                    status,
                );

                i = nmembers;
                while i > 0 && *status == 0 {
                    *status = fits_remove_member(
                        mfptr.as_deref_mut().expect(NULL_MSG),
                        i,
                        OPT_RM_ENTRY as c_int,
                        status,
                    );
                    i -= 1;
                }

                if *status != 0 {
                    break 'outer;
                }
            }

            /* unlink the member HDU from all groups that contain it */

            *status = ffgmul(mfptr.as_deref_mut().expect(NULL_MSG), 0, status);

            if *status != 0 {
                break 'outer;
            }

            /* reset the grouping table HDU struct */

            fits_set_hdustruc(gfptr, status);

            /* delete the member HDU */

            if iomode != READONLY {
                *status = fits_delete_hdu(
                    mfptr.as_deref_mut().expect(NULL_MSG),
                    Some(&mut hdutype),
                    status,
                );
            }
        } else if rmopt as u64 == OPT_RM_ENTRY {
            /*
               The member HDU is only to be removed as an entry from this
               grouping table. Actions are (1) find the GRPIDn/GRPLCn
               keywords that link the member to the grouping table, (2)
               remove the GRPIDn/GRPLCn keyword from the member HDU header
               and (3) remove the member entry from the grouping table
            */

            /*
              there is no need to seach for and remove the GRPIDn/GRPLCn
              keywords from the member HDU if it has not been opened
              in READWRITE mode
            */

            if iomode == READWRITE {
                /*
                   determine the group EXTVER value of the grouping table; if
                   the member HDU and grouping table HDU do not reside in the
                   same file then set the groupExtver value to its negative
                */

                *status = fits_read_key_lng(
                    gfptr,
                    cs!(c"EXTVER"),
                    &mut groupExtver,
                    Some(&mut comment),
                    status,
                );
                /* Now, if either the Fptr values are the same, or the root filenames
                   are the same, then assume these refer to the same file.
                */
                let mfn = cast_slice::<u8, c_char>(
                    mfptr
                        .as_deref()
                        .expect(NULL_MSG)
                        .Fptr
                        .get_filename_as_cstr()
                        .to_bytes_with_nul(),
                );
                fits_parse_rootname(mfn, &mut mrootname, status);
                let gfn =
                    cast_slice::<u8, c_char>(gfptr.Fptr.get_filename_as_cstr().to_bytes_with_nul());
                fits_parse_rootname(gfn, &mut grootname, status);

                /* The C's combined condition: same FITSfile by address, or
                failing that the same rootname. */
                let same_fptr = core::ptr::eq(
                    mfptr.as_deref().expect(NULL_MSG).Fptr.as_ptr(),
                    gfptr.Fptr.as_ptr(),
                );
                if !same_fptr && strncmp_safe(&mrootname, &grootname, FLEN_FILENAME) != 0 {
                    groupExtver = -groupExtver;
                }

                /*
                  retrieve the URLs for the grouping table; note that it is
                  possible that the grouping table file has two URLs, the
                  one used to open it and the "real" one pointing to the
                  actual file being accessed
                */

                *status = fits_get_url(
                    gfptr,
                    &mut grpLocation1,
                    &mut grpLocation2,
                    &mut access1,
                    &mut access2,
                    &mut iostate,
                    status,
                );

                if *status != 0 {
                    break 'outer;
                }

                /*
                  if either of the group location strings specify a relative
                  file path then convert them into absolute file paths
                */

                *status = fits_get_cwd(&mut cwd, status);

                if grpLocation1[0] != 0
                    && grpLocation1[0] != bb(b'/')
                    && fits_is_url_absolute(&grpLocation1) == 0
                {
                    strcpy_safe(&mut grpLocation3, &cwd);
                    if strlen_safe(&grpLocation3) + strlen_safe(&grpLocation1) + 1
                        > FLEN_FILENAME - 1
                    {
                        ffpmsg_str("group locations are too long (ffgmrm)");
                        *status = URL_PARSE_ERROR;
                        break 'outer;
                    }
                    strcat_safe(&mut grpLocation3, cs!(c"/"));
                    strcat_safe(&mut grpLocation3, &grpLocation1);
                    fits_clean_url(&grpLocation3, &mut grpLocation1, status);
                }

                if grpLocation2[0] != 0
                    && grpLocation2[0] != bb(b'/')
                    && fits_is_url_absolute(&grpLocation2) == 0
                {
                    strcpy_safe(&mut grpLocation3, &cwd);
                    if strlen_safe(&grpLocation3) + strlen_safe(&grpLocation2) + 1
                        > FLEN_FILENAME - 1
                    {
                        ffpmsg_str("group locations are too long (ffgmrm)");
                        *status = URL_PARSE_ERROR;
                        break 'outer;
                    }
                    strcat_safe(&mut grpLocation3, cs!(c"/"));
                    strcat_safe(&mut grpLocation3, &grpLocation2);
                    fits_clean_url(&grpLocation3, &mut grpLocation2, status);
                }

                /*
                  determine the number of groups to which the member HDU
                  belongs
                */

                *status = fits_get_num_groups(
                    mfptr.as_deref_mut().expect(NULL_MSG),
                    &mut ngroups,
                    status,
                );

                /* reset the HDU keyword position counter to the beginning */

                *status = fits_read_record(
                    mfptr.as_deref_mut().expect(NULL_MSG),
                    0,
                    Some(&mut card),
                    status,
                );

                /*
                  loop over all the GRPIDn keywords in the member HDU header
                  and find the appropriate GRPIDn and GRPLCn keywords that
                  identify it as belonging to the group
                */

                // The for-loop increment (++index) is replicated before each
                // `continue` so the C `continue` statements still advance index.
                index = 1;
                found = 0;
                while (index as c_long) <= ngroups && *status == 0 && found == 0 {
                    /* read the next GRPIDn keyword in the series */

                    int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPID{}", index);

                    *status = fits_read_key_lng(
                        mfptr.as_deref_mut().expect(NULL_MSG),
                        &keyword,
                        &mut grpid,
                        Some(&mut comment),
                        status,
                    );
                    if *status != 0 {
                        index += 1;
                        continue;
                    }

                    /*
                       grpid value == group EXTVER value then we could have a
                       match
                    */

                    if grpid == groupExtver && grpid > 0 {
                        /*
                          if GRPID is positive then its a match because
                          both the member HDU and grouping table HDU reside
                          in the same FITS file
                        */

                        found = index;
                    } else if grpid == groupExtver && grpid < 0 {
                        /*
                           have to look at the GRPLCn value to determine a
                           match because the member HDU and grouping table
                           HDU reside in different FITS files
                        */

                        int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPLC{}", index);

                        /* SPR 1738 */
                        // ffgkls_safe still outputs a raw, heap-allocated C string.
                        let mut tgrplc: *mut c_char = core::ptr::null_mut();
                        *status = fits_read_key_longstr(
                            mfptr.as_deref_mut().expect(NULL_MSG),
                            &keyword,
                            &mut tgrplc,
                            None,
                            status,
                        );
                        if 0 == *status {
                            // SAFETY: tgrplc is the heap C string from ffgkls_safe,
                            // tracked in ALLOCATIONS; copy it out then release it.
                            let too_long = unsafe {
                                let cs = CStr::from_ptr(tgrplc);
                                let too_long = cs.to_bytes().len() > FLEN_FILENAME - 1;
                                if !too_long {
                                    strcpy_safe(&mut grplc, cast_slice(cs.to_bytes_with_nul()));
                                }
                                if let Some((l, c)) =
                                    crate::lock_global(&ALLOCATIONS).remove(&(tgrplc as usize))
                                {
                                    let _ = Vec::from_raw_parts(tgrplc, l, c);
                                }
                                too_long
                            };

                            if too_long {
                                ffpmsg_str("GRPLCn string is too long (ffgmrm)");
                                *status = URL_PARSE_ERROR;
                                index += 1;
                                continue;
                            }
                        }

                        if *status == KEY_NO_EXIST {
                            /*
                               no GRPLCn keyword value found ==> grouping
                               convention not followed; nothing we can do
                               about it, so just continue
                            */

                            int_snprintf!(
                                &mut card,
                                FLEN_CARD,
                                "No GRPLC{} found for GRPID{}",
                                index,
                                index
                            );
                            ffpmsg_slice(&card);
                            *status = 0;
                            index += 1;
                            continue;
                        } else if *status != 0 {
                            index += 1;
                            continue;
                        }

                        /* construct the URL for the GRPLCn value */

                        prepare_keyvalue(&mut grplc);

                        /*
                          if the grplc value specifies a relative path then
                          turn it into a absolute file path for comparison
                          purposes
                        */

                        if grplc[0] != 0
                            && fits_is_url_absolute(&grplc) == 0
                            && grplc[0] != bb(b'/')
                        {
                            /* No, wrong,
                            strcpy(grpLocation3,cwd);
                            should be */
                            *status = fits_file_name(
                                mfptr.as_deref_mut().expect(NULL_MSG),
                                &mut grpLocation3,
                                status,
                            );
                            /* Remove everything after the last / */
                            if let Some(editLocation) = strrchr_safe(&grpLocation3, bb(b'/')) {
                                grpLocation3[editLocation] = 0;
                            }

                            if strlen_safe(&grpLocation3) + strlen_safe(&grplc) + 1
                                > FLEN_FILENAME - 1
                            {
                                ffpmsg_str("group locations are too long (ffgmrm)");
                                *status = URL_PARSE_ERROR;
                                index += 1;
                                continue;
                            }
                            strcat_safe(&mut grpLocation3, cs!(c"/"));
                            strcat_safe(&mut grpLocation3, &grplc);
                            *status = fits_clean_url(&grpLocation3, &mut grplc, status);
                        }

                        /*
                          if the absolute value of GRPIDn is equal to the
                          EXTVER value of the grouping table and (one of the
                          possible two) grouping table file URL matches the
                          GRPLCn keyword value then we hava a match
                        */

                        if strcmp_safe(&grplc, &grpLocation1) == 0
                            || strcmp_safe(&grplc, &grpLocation2) == 0
                        {
                            found = index;
                        }
                    }

                    index += 1;
                }

                /*
                  if found == 0 (false) after the above search then we assume
                  that it is due to an inpromper updating of the GRPIDn and
                  GRPLCn keywords in the member header ==> nothing to delete
                  in the header. Else delete the GRPLCn and GRPIDn keywords
                  that identify the member HDU with the group HDU and
                  re-enumerate the remaining GRPIDn and GRPLCn keywords
                */

                if found != 0 {
                    int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPID{}", found);
                    *status =
                        fits_delete_key(mfptr.as_deref_mut().expect(NULL_MSG), &keyword, status);

                    int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPLC{}", found);
                    *status =
                        fits_delete_key(mfptr.as_deref_mut().expect(NULL_MSG), &keyword, status);

                    *status = 0;

                    /* call fits_get_num_groups() to re-enumerate the GRPIDn */

                    *status = fits_get_num_groups(
                        mfptr.as_deref_mut().expect(NULL_MSG),
                        &mut ngroups,
                        status,
                    );
                }
            }

            /*
               finally, remove the member entry from the current grouping table
               pointed to by gfptr
            */

            *status = fits_delete_rows(gfptr, member as LONGLONG, 1, status);
        } else {
            *status = BAD_OPTION;
            ffpmsg_str("Invalid value specified for the rmopt parameter (ffgmrm)");
        }

        break 'outer;
    } //while(0)

    if let Some(f) = mfptr.take() {
        fits_close_file(f, status);
    }

    *status
}

/*---------------------------------------------------------------------------
                 Grouping Table support functions
---------------------------------------------------------------------------*/

/// Examine the grouping table pointed to by gfptr and determine the column
/// index ID of each possible grouping column. If a column is not found then
/// an index of 0 is returned. the grptype parameter returns the structure
/// of the grouping table ==> what columns are defined.
///
/// # Parameters
///
/// * `gfptr`       — pointer to the grouping table
/// * `xtensionCol` — column ID of the MEMBER_XTENSION column
/// * `extnameCol`  — column ID of the MEMBER_NAME column
/// * `extverCol`   — column ID of the MEMBER_VERSION column
/// * `positionCol` — column ID of the MEMBER_POSITION column
/// * `locationCol` — column ID of the MEMBER_LOCATION column
/// * `uriCol`      — column ID of the MEMBER_URI_TYPE column
/// * `grptype`     — group structure type code specifying the grouping table columns
///                  that are defined: GT_ID_ALL_URI (0) ==> all columns defined
///                  GT_ID_REF (1) ==> reference cols only GT_ID_POS (2) ==> position
///                  col only GT_ID_ALL (3) ==> ref & pos cols GT_ID_REF_URI (11) ==>
///                  ref & loc cols GT_ID_POS_URI (12) ==> pos & loc cols
/// * `status`      — return status code
pub(crate) fn ffgtgc(
    gfptr: &mut fitsfile,
    xtensionCol: &mut c_int,
    extnameCol: &mut c_int,
    extverCol: &mut c_int,
    positionCol: &mut c_int,
    locationCol: &mut c_int,
    uriCol: &mut c_int,
    grptype: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        /*
        if the HDU does not have an extname of "GROUPING" then it is not
        a grouping table
        */

        *status = fits_read_key_str(
            gfptr,
            cs!(c"EXTNAME"),
            &mut keyvalue,
            Some(&mut comment),
            status,
        );

        if *status == KEY_NO_EXIST {
            *status = NOT_GROUP_TABLE;
            ffpmsg_str("Specified HDU is not a Grouping Table (ffgtgc)");
        }
        if *status != 0 {
            break;
        }

        prepare_keyvalue(&mut keyvalue);

        if fits_strcasecmp(&keyvalue, cs!(c"GROUPING")) != 0 {
            *status = NOT_GROUP_TABLE;
            break;
        }

        /*
          search for the MEMBER_XTENSION, MEMBER_NAME, MEMBER_VERSION,
          MEMBER_POSITION, MEMBER_LOCATION and MEMBER_URI_TYPE columns
          and determine their column index ID
        */

        *status = fits_get_colnum(
            gfptr,
            CASESEN as c_int,
            cs!(c"MEMBER_XTENSION"),
            xtensionCol,
            status,
        );

        if *status == COL_NOT_FOUND {
            *status = 0;
            *xtensionCol = 0;
        }

        if *status != 0 {
            break;
        }

        *status = fits_get_colnum(
            gfptr,
            CASESEN as c_int,
            cs!(c"MEMBER_NAME"),
            extnameCol,
            status,
        );

        if *status == COL_NOT_FOUND {
            *status = 0;
            *extnameCol = 0;
        }

        if *status != 0 {
            break;
        }

        *status = fits_get_colnum(
            gfptr,
            CASESEN as c_int,
            cs!(c"MEMBER_VERSION"),
            extverCol,
            status,
        );

        if *status == COL_NOT_FOUND {
            *status = 0;
            *extverCol = 0;
        }

        if *status != 0 {
            break;
        }

        *status = fits_get_colnum(
            gfptr,
            CASESEN as c_int,
            cs!(c"MEMBER_POSITION"),
            positionCol,
            status,
        );

        if *status == COL_NOT_FOUND {
            *status = 0;
            *positionCol = 0;
        }

        if *status != 0 {
            break;
        }

        *status = fits_get_colnum(
            gfptr,
            CASESEN as c_int,
            cs!(c"MEMBER_LOCATION"),
            locationCol,
            status,
        );

        if *status == COL_NOT_FOUND {
            *status = 0;
            *locationCol = 0;
        }

        if *status != 0 {
            break;
        }

        *status = fits_get_colnum(
            gfptr,
            CASESEN as c_int,
            cs!(c"MEMBER_URI_TYPE"),
            uriCol,
            status,
        );

        if *status == COL_NOT_FOUND {
            *status = 0;
            *uriCol = 0;
        }

        if *status != 0 {
            break;
        }

        /*
           determine the type of grouping table structure used by this
           grouping table and record it in the grptype parameter
        */

        if *xtensionCol != 0
            && *extnameCol != 0
            && *extverCol != 0
            && *positionCol != 0
            && *locationCol != 0
            && *uriCol != 0
        {
            *grptype = GT_ID_ALL_URI as c_int;
        } else if *xtensionCol != 0
            && *extnameCol != 0
            && *extverCol != 0
            && *locationCol != 0
            && *uriCol != 0
        {
            *grptype = GT_ID_REF_URI as c_int;
        } else if *xtensionCol != 0 && *extnameCol != 0 && *extverCol != 0 && *positionCol != 0 {
            *grptype = GT_ID_ALL as c_int;
        } else if *xtensionCol != 0 && *extnameCol != 0 && *extverCol != 0 {
            *grptype = GT_ID_REF as c_int;
        } else if *positionCol != 0 && *locationCol != 0 && *uriCol != 0 {
            *grptype = GT_ID_POS_URI as c_int;
        } else if *positionCol != 0 {
            *grptype = GT_ID_POS as c_int;
        } else {
            *status = NOT_GROUP_TABLE;
        }

        break;
    } //while(0)

    /*
      if the table contained more than one column with a reserved name then
      this cannot be considered a vailid grouping table
    */

    if *status == COL_NOT_UNIQUE {
        *status = NOT_GROUP_TABLE;
        ffpmsg_str("Specified HDU has multipule Group table cols defined (ffgtgc)");
    }

    *status
}

/// Perform validation on column formats to ensure this matches the grouping
/// format the get functions expect.  Particularly want to check widths of
/// string columns.
pub(crate) fn ffvcfm(
    gfptr: &mut fitsfile,
    xtensionCol: c_int,
    extnameCol: c_int,
    extverCol: c_int,
    positionCol: c_int,
    locationCol: c_int,
    uriCol: c_int,
    status: &mut c_int,
) -> c_int {
    let mut typecode: c_int = 0;
    let mut repeat: c_long = 0;
    let mut width: c_long = 0;

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        if xtensionCol != 0 {
            fits_get_coltype(
                gfptr,
                xtensionCol,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                status,
            );
            if *status != 0 || typecode != TSTRING || repeat != width || repeat > 8 {
                if *status == 0 {
                    *status = NOT_GROUP_TABLE;
                }
                ffpmsg_str("Wrong format for Grouping xtension col. (ffvcfm)");
                break;
            }
        }
        if extnameCol != 0 {
            fits_get_coltype(
                gfptr,
                extnameCol,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                status,
            );
            if *status != 0 || typecode != TSTRING || repeat != width || repeat > 32 {
                if *status == 0 {
                    *status = NOT_GROUP_TABLE;
                }
                ffpmsg_str("Wrong format for Grouping name col. (ffvcfm)");
                break;
            }
        }
        if extverCol != 0 {
            fits_get_coltype(
                gfptr,
                extverCol,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                status,
            );
            if *status != 0 || typecode != TINT32BIT || repeat > 1 {
                if *status == 0 {
                    *status = NOT_GROUP_TABLE;
                }
                ffpmsg_str("Wrong format for Grouping version col. (ffvcfm)");
                break;
            }
        }
        if positionCol != 0 {
            fits_get_coltype(
                gfptr,
                positionCol,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                status,
            );
            if *status != 0 || typecode != TINT32BIT || repeat > 1 {
                if *status == 0 {
                    *status = NOT_GROUP_TABLE;
                }
                ffpmsg_str("Wrong format for Grouping position col. (ffvcfm)");
                break;
            }
        }
        if locationCol != 0 {
            fits_get_coltype(
                gfptr,
                locationCol,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                status,
            );
            if *status != 0 || typecode != TSTRING || repeat != width || repeat > 256 {
                if *status == 0 {
                    *status = NOT_GROUP_TABLE;
                }
                ffpmsg_str("Wrong format for Grouping location col. (ffvcfm)");
                break;
            }
        }
        if uriCol != 0 {
            fits_get_coltype(
                gfptr,
                uriCol,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                status,
            );
            if *status != 0 || typecode != TSTRING || repeat != width || repeat > 3 {
                if *status == 0 {
                    *status = NOT_GROUP_TABLE;
                }
                ffpmsg_str("Wrong format for Grouping URI col. (ffvcfm)");
                break;
            }
        }

        break;
    } //while(0)

    *status
}

/// Create the TTYPE and TFORM values for the grouping table according to the
/// value of the grouptype parameter and the values of the *col flags. The
/// resulting TTYPE and TFORM are returned in ttype[] and tform[] respectively.
/// The number of TTYPE and TFORMs returned is given by ncols. Both the TTYPE[]
/// and TTFORM[] arrays must contain enough pre-allocated strings to hold
/// the returned information.
///
/// # Parameters
///
/// * `grouptype`   — code specifying the type of grouping table information:
///                  GT_ID_ALL_URI 0 ==> defualt (all columns) GT_ID_REF 1 ==> ID by
///                  reference GT_ID_POS 2 ==> ID by position GT_ID_ALL 3 ==> ID by
///                  ref. and position GT_ID_REF_URI 11 ==> (1) + URI info
///                  GT_ID_POS_URI 12 ==> (2) + URI info
/// * `xtensioncol` — does MEMBER_XTENSION already exist?
/// * `extnamecol`  — does MEMBER_NAME aleady exist?
/// * `extvercol`   — does MEMBER_VERSION already exist?
/// * `positioncol` — does MEMBER_POSITION already exist?
/// * `locationcol` — does MEMBER_LOCATION already exist?
/// * `uricol`      — does MEMBER_URI_TYPE aleardy exist?
/// * `ttype`       — array of grouping table column TTYPE names to define (if *col var
///                  false)
/// * `tform`       — array of grouping table column TFORM values to define (if*col
///                  variable false)
/// * `ncols`       — number of TTYPE and TFORM values returned
/// * `status`      — return status code
pub(crate) fn ffgtdc(
    grouptype: c_int,
    xtensioncol: c_int,
    extnamecol: c_int,
    extvercol: c_int,
    positioncol: c_int,
    locationcol: c_int,
    uricol: c_int,
    ttype: &mut [&mut [c_char]],
    tform: &mut [&mut [c_char]],
    ncols: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut i: usize = 0;

    let xtension = cs!(c"MEMBER_XTENSION");
    let xtenTform = cs!(c"8A");

    let name = cs!(c"MEMBER_NAME");
    let nameTform = cs!(c"32A");

    let version = cs!(c"MEMBER_VERSION");
    let verTform = cs!(c"1J");

    let position = cs!(c"MEMBER_POSITION");
    let posTform = cs!(c"1J");

    let URI = cs!(c"MEMBER_URI_TYPE");
    let URITform = cs!(c"3A");

    let location = cs!(c"MEMBER_LOCATION");
    /* SPR 01720, move from 160A to 256A */
    let locTform = cs!(c"256A");

    if *status != 0 {
        return *status;
    }

    match grouptype as u64 {
        GT_ID_ALL_URI => {
            if xtensioncol == 0 {
                strcpy_safe(ttype[i], xtension);
                strcpy_safe(tform[i], xtenTform);
                i += 1;
            }
            if extnamecol == 0 {
                strcpy_safe(ttype[i], name);
                strcpy_safe(tform[i], nameTform);
                i += 1;
            }
            if extvercol == 0 {
                strcpy_safe(ttype[i], version);
                strcpy_safe(tform[i], verTform);
                i += 1;
            }
            if positioncol == 0 {
                strcpy_safe(ttype[i], position);
                strcpy_safe(tform[i], posTform);
                i += 1;
            }
            if locationcol == 0 {
                strcpy_safe(ttype[i], location);
                strcpy_safe(tform[i], locTform);
                i += 1;
            }
            if uricol == 0 {
                strcpy_safe(ttype[i], URI);
                strcpy_safe(tform[i], URITform);
                i += 1;
            }
        }

        GT_ID_REF => {
            if xtensioncol == 0 {
                strcpy_safe(ttype[i], xtension);
                strcpy_safe(tform[i], xtenTform);
                i += 1;
            }
            if extnamecol == 0 {
                strcpy_safe(ttype[i], name);
                strcpy_safe(tform[i], nameTform);
                i += 1;
            }
            if extvercol == 0 {
                strcpy_safe(ttype[i], version);
                strcpy_safe(tform[i], verTform);
                i += 1;
            }
        }

        GT_ID_POS => {
            if positioncol == 0 {
                strcpy_safe(ttype[i], position);
                strcpy_safe(tform[i], posTform);
                i += 1;
            }
        }

        GT_ID_ALL => {
            if xtensioncol == 0 {
                strcpy_safe(ttype[i], xtension);
                strcpy_safe(tform[i], xtenTform);
                i += 1;
            }
            if extnamecol == 0 {
                strcpy_safe(ttype[i], name);
                strcpy_safe(tform[i], nameTform);
                i += 1;
            }
            if extvercol == 0 {
                strcpy_safe(ttype[i], version);
                strcpy_safe(tform[i], verTform);
                i += 1;
            }
            if positioncol == 0 {
                strcpy_safe(ttype[i], position);
                strcpy_safe(tform[i], posTform);
                i += 1;
            }
        }

        GT_ID_REF_URI => {
            if xtensioncol == 0 {
                strcpy_safe(ttype[i], xtension);
                strcpy_safe(tform[i], xtenTform);
                i += 1;
            }
            if extnamecol == 0 {
                strcpy_safe(ttype[i], name);
                strcpy_safe(tform[i], nameTform);
                i += 1;
            }
            if extvercol == 0 {
                strcpy_safe(ttype[i], version);
                strcpy_safe(tform[i], verTform);
                i += 1;
            }
            if locationcol == 0 {
                strcpy_safe(ttype[i], location);
                strcpy_safe(tform[i], locTform);
                i += 1;
            }
            if uricol == 0 {
                strcpy_safe(ttype[i], URI);
                strcpy_safe(tform[i], URITform);
                i += 1;
            }
        }

        GT_ID_POS_URI => {
            if positioncol == 0 {
                strcpy_safe(ttype[i], position);
                strcpy_safe(tform[i], posTform);
                i += 1;
            }
            if locationcol == 0 {
                strcpy_safe(ttype[i], location);
                strcpy_safe(tform[i], locTform);
                i += 1;
            }
            if uricol == 0 {
                strcpy_safe(ttype[i], URI);
                strcpy_safe(tform[i], URITform);
                i += 1;
            }
        }

        _ => {
            *status = BAD_OPTION;
            ffpmsg_str("Invalid value specified for the grouptype parameter (ffgtdc)");
        }
    }

    *ncols = i as c_int;

    *status
}

/// Examine all the GRPIDn and GRPLCn keywords in the member HDUs header
/// and remove the member from the grouping tables referenced; This
/// effectively "unlinks" the member from all of its groups. The rmopt
/// specifies if the GRPIDn/GRPLCn keywords are to be removed from the
/// member HDUs header after the unlinking.
///
/// # Parameters
///
/// * `mfptr`  — pointer to the grouping table member HDU
/// * `rmopt`  — 0 ==> leave GRPIDn/GRPLCn keywords, 1 ==> remove GRPIDn/GRPLCn keywords
/// * `status` — return status code
pub(crate) fn ffgmul(mfptr: &mut fitsfile, rmopt: c_int, status: &mut c_int) -> c_int {
    let mut memberPosition: c_int = 0;
    let mut iomode: c_int = 0;

    let mut ngroups: c_long = 0;
    let mut memberExtver: c_long = 0;
    let mut memberID: c_long;

    let mut mbrLocation1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut mbrLocation2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut memberHDUtype: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut memberExtname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    let mut gfptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    'outer: loop {
        /*
        determine location parameters of the member HDU; note that
        default values are supplied if the expected keywords are not
        found
         */

        *status = fits_read_key_str(
            mfptr,
            cs!(c"XTENSION"),
            &mut memberHDUtype,
            Some(&mut comment),
            status,
        );

        if *status == KEY_NO_EXIST {
            strcpy_safe(&mut memberHDUtype, cs!(c"PRIMARY"));
            *status = 0;
        }
        prepare_keyvalue(&mut memberHDUtype);

        *status = fits_read_key_lng(
            mfptr,
            cs!(c"EXTVER"),
            &mut memberExtver,
            Some(&mut comment),
            status,
        );

        if *status == KEY_NO_EXIST {
            memberExtver = 1;
            *status = 0;
        }

        *status = fits_read_key_str(
            mfptr,
            cs!(c"EXTNAME"),
            &mut memberExtname,
            Some(&mut comment),
            status,
        );

        if *status == KEY_NO_EXIST {
            memberExtname[0] = 0;
            *status = 0;
        }
        prepare_keyvalue(&mut memberExtname);

        fits_get_hdu_num(mfptr, &mut memberPosition);

        // (the access/iostate outputs are unused here -- C passed NULL)
        let mut realAccess: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
        let mut startAccess: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
        let mut dummyIostate: c_int = 0;
        *status = fits_get_url(
            mfptr,
            &mut mbrLocation1,
            &mut mbrLocation2,
            &mut realAccess,
            &mut startAccess,
            &mut dummyIostate,
            status,
        );

        if *status != 0 {
            break 'outer;
        }

        /*
        open each grouping table linked to this HDU and remove the member
        from the grouping tables
         */

        *status = fits_get_num_groups(mfptr, &mut ngroups, status);

        /* loop over each group linked to the member HDU */

        let mut index: c_long = 1;
        while index <= ngroups && *status == 0 {
            // C: for(index = 1; index <= ngroups && *status == 0; ++index)
            // the body's `continue`s map to Rust `continue`; the trailing ++index
            // is reproduced at the bottom and the status guard re-tested at the top.

            /* open the (index)th group linked to the member HDU */

            *status = fits_open_group(mfptr, index as c_int, &mut gfptr, status);

            /* if the group could not be opened then just skip it */

            if *status != 0 {
                *status = 0;
                int_snprintf!(
                    &mut card,
                    FLEN_CARD,
                    "Cannot open the {}th group table (ffgmul)",
                    index as c_int
                );
                ffpmsg_slice(&card);
                index += 1;
                continue;
            }

            /*
            make sure the grouping table can be modified before proceeding
              */

            fits_file_mode(gfptr.as_deref_mut().expect(NULL_MSG), &mut iomode, status);

            if iomode != READWRITE {
                int_snprintf!(
                    &mut card,
                    FLEN_CARD,
                    "The {}th group cannot be modified (ffgtam)",
                    index as c_int
                );
                ffpmsg_slice(&card);
                index += 1;
                continue;
            }

            /*
            try to find the member's row within the grouping table; first
            try using the member HDU file's "real" URL string then try
            using its originally opened URL string if either string exist
              */

            memberID = 0;

            if strlen_safe(&mbrLocation1) != 0 {
                *status = ffgmf(
                    gfptr.as_deref_mut().expect(NULL_MSG),
                    &memberHDUtype,
                    &memberExtname,
                    memberExtver as c_int,
                    memberPosition,
                    Some(&mbrLocation1),
                    &mut memberID,
                    status,
                );
            }

            if *status == MEMBER_NOT_FOUND && strlen_safe(&mbrLocation2) != 0 {
                *status = 0;
                *status = ffgmf(
                    gfptr.as_deref_mut().expect(NULL_MSG),
                    &memberHDUtype,
                    &memberExtname,
                    memberExtver as c_int,
                    memberPosition,
                    Some(&mbrLocation2),
                    &mut memberID,
                    status,
                );
            }

            /* if the member was found then delete it from the grouping table */

            if *status == 0 {
                *status = fits_delete_rows(
                    gfptr.as_deref_mut().expect(NULL_MSG),
                    memberID as LONGLONG,
                    1,
                    status,
                );
            }

            /*
            continue the loop over all member groups even if an error
            was generated
             */

            if *status == MEMBER_NOT_FOUND {
                ffpmsg_str("cannot locate member's entry in group table (ffgmul)");
            }
            *status = 0;

            /*
            close the file pointed to by gfptr if it is non NULL to
            prepare for the next loop iterration
             */

            if let Some(f) = gfptr.take() {
                fits_close_file(f, status);
            }

            index += 1;
        }

        if *status != 0 {
            break 'outer;
        }

        /*
        if rmopt is non-zero then find and delete the GRPIDn/GRPLCn
        keywords from the member HDU header
         */

        if rmopt != 0 {
            fits_file_mode(mfptr, &mut iomode, status);

            if iomode == READONLY {
                ffpmsg_str("Cannot modify member HDU, opened READONLY (ffgmul)");
                break 'outer;
            }

            /* delete all the GRPIDn/GRPLCn keywords */

            let mut index: c_long = 1;
            while index <= ngroups && *status == 0 {
                int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPID{}", index as c_int);
                fits_delete_key(mfptr, &keyword, status);

                int_snprintf!(&mut keyword, FLEN_KEYWORD, "GRPLC{}", index as c_int);
                fits_delete_key(mfptr, &keyword, status);

                if *status == KEY_NO_EXIST {
                    *status = 0;
                }

                index += 1;
            }
        }

        break 'outer;
    } // while(0)

    /* make sure the gfptr has been closed */

    if let Some(f) = gfptr.take() {
        fits_close_file(f, status);
    }

    *status
}

/// Try to find the entry for the member HDU defined by the xtension, extname,
/// extver, position, and location parameters within the grouping table
/// pointed to by gfptr. If the member HDU is found then its ID (row number)
/// within the grouping table is returned in the member variable; if not
/// found then member is returned with a value of 0 and the status return
/// code will be set to MEMBER_NOT_FOUND.
///
/// Note that the member HDU postion information is used to obtain a member
/// match only if the grouping table type is GT_ID_POS_URI or GT_ID_POS. This
/// is because the position information can become invalid much more
/// easily then the reference information for a group member.
#[allow(clippy::if_same_then_else)]
// C dispatch chain: distinct conditions deliberately share an action.
/// # Parameters
///
/// * `gfptr`    — pointer to grouping table HDU to search
/// * `xtension` — XTENSION value for member HDU
/// * `extname`  — EXTNAME value for member HDU
/// * `extver`   — EXTVER value for member HDU
/// * `position` — HDU position value for member HDU
/// * `location` — FITS file location value for member HDU
/// * `member`   — member HDU ID within group table (if found)
/// * `status`   — return status code
pub(crate) fn ffgmf(
    gfptr: &mut fitsfile,
    xtension: &[c_char],
    extname: &[c_char],
    extver: c_int,
    position: c_int,
    location: Option<&[c_char]>,
    member: &mut c_long,
    status: &mut c_int,
) -> c_int {
    let mut xtensionCol: c_int = 0;
    let mut extnameCol: c_int = 0;
    let mut extverCol: c_int = 0;
    let mut positionCol: c_int = 0;
    let mut locationCol: c_int = 0;
    let mut uriCol: c_int = 0;
    let mut mposition: c_int = 0;
    let mut grptype: c_int = 0;
    let mut dummy: c_int = 0;
    let mut i: c_long;

    let mut nmembers: c_long = 0;
    let mut mextver: c_long = 0;

    let mut charBuff1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut tmpLocation: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut mbrLocation1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut mbrLocation2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut mbrLocation3: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut grpLocation1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut grpLocation2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut cwd: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    let nstr: [c_char; 1] = [0]; /* char nstr[] = {'\0'}; the null-value string */

    // C used `char *tmpPtr[2]` pointing at charBuff1/charBuff2; only tmpPtr[0]
    // (charBuff1) is ever read since each read is for a single element (nelem=1).

    // (the access/iostate outputs of fits_get_url are unused here -- C passed NULL)
    let mut access1: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut access2: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut iostate: c_int = 0;

    if *status != 0 {
        return *status;
    }

    *member = 0;

    if *status != 0 {
        return *status;
    }

    /*
      if the passed LOCATION value is not an absolute URL then turn it
      into an absolute path
    */

    if location.is_none_or(|location| location[0] == 0) {
        tmpLocation[0] = 0;
    } else if let Some(location) = location
        && fits_is_url_absolute(location) == 0
    {
        fits_path2url(location, FLEN_FILENAME, &mut tmpLocation, status);

        if tmpLocation[0] != bb(b'/') {
            fits_get_cwd(&mut cwd, status);
            if strlen_safe(&cwd) + strlen_safe(&tmpLocation) + 1 > FLEN_FILENAME - 1 {
                ffpmsg_str("cwd and location are too long (ffgmf)");
                *status = URL_PARSE_ERROR;
                return *status;
            }
            strcat_safe(&mut cwd, cs!(c"/"));
            strcat_safe(&mut cwd, &tmpLocation);
            fits_clean_url(&cwd, &mut tmpLocation, status);
        }
    } else if let Some(location) = location {
        strcpy_safe(&mut tmpLocation, location);
    }

    /*
       retrieve the Grouping Convention reserved column positions within
       the grouping table
    */

    *status = ffgtgc(
        gfptr,
        &mut xtensionCol,
        &mut extnameCol,
        &mut extverCol,
        &mut positionCol,
        &mut locationCol,
        &mut uriCol,
        &mut grptype,
        status,
    );

    /* retrieve the number of group members */

    *status = fits_get_num_members(gfptr, &mut nmembers, status);

    /*
       loop over all grouping table rows until the member HDU is found
    */

    // The loop counter is bumped at the top so the C `continue` statements (which
    // fall through to the for-loop's ++i) map to a plain Rust `continue`.
    i = 0;
    while i < nmembers && *member == 0 && *status == 0 {
        i += 1;

        if xtensionCol != 0 {
            fits_read_col_str(
                gfptr,
                xtensionCol,
                i as LONGLONG,
                1,
                1,
                Some(&nstr),
                &mut [&mut charBuff1[..]],
                Some(&mut dummy),
                status,
            );
            if fits_strcasecmp(&charBuff1, xtension) != 0 {
                continue;
            }
        }

        if extnameCol != 0 {
            fits_read_col_str(
                gfptr,
                extnameCol,
                i as LONGLONG,
                1,
                1,
                Some(&nstr),
                &mut [&mut charBuff1[..]],
                Some(&mut dummy),
                status,
            );
            if fits_strcasecmp(&charBuff1, extname) != 0 {
                continue;
            }
        }

        if extverCol != 0 {
            fits_read_col_lng(
                gfptr,
                extverCol,
                i as LONGLONG,
                1,
                1,
                0,
                core::slice::from_mut(&mut mextver),
                Some(&mut dummy),
                status,
            );
            if extver as c_long != mextver {
                continue;
            }
        }

        /* note we only use postionCol if we have to */

        if positionCol != 0 && (grptype as u64 == GT_ID_POS || grptype as u64 == GT_ID_POS_URI) {
            fits_read_col_int(
                gfptr,
                positionCol,
                i as LONGLONG,
                1,
                1,
                0,
                core::slice::from_mut(&mut mposition),
                Some(&mut dummy),
                status,
            );
            if position != mposition {
                continue;
            }
        }

        /*
          if no location string was passed to the function then assume that
          the calling application does not wish to use it as a comparision
          critera ==> if we got this far then we have a match
        */

        if location.is_none() {
            ffpmsg_str("NULL Location string given ==> ignore location (ffgmf)");
            *member = i;
            continue;
        }

        /*
          if the grouping table MEMBER_LOCATION column exists then read the
          location URL for the member, else set the location string to
          a zero-length string for subsequent comparisions
        */

        if locationCol != 0 {
            fits_read_col_str(
                gfptr,
                locationCol,
                i as LONGLONG,
                1,
                1,
                Some(&nstr),
                &mut [&mut charBuff1[..]],
                Some(&mut dummy),
                status,
            );
            strcpy_safe(&mut mbrLocation1, &charBuff1);
            mbrLocation2[0] = 0;
        } else {
            mbrLocation1[0] = 0;
        }

        /*
          if the member location string from the grouping table is zero
          length (either implicitly or explicitly) then assume that the
          member HDU is in the same file as the grouping table HDU; retrieve
          the possible URL values of the grouping table HDU file
        */

        if mbrLocation1[0] == 0 {
            /* retrieve the possible URLs of the grouping table file */
            *status = fits_get_url(
                gfptr,
                &mut mbrLocation1,
                &mut mbrLocation2,
                &mut access1,
                &mut access2,
                &mut iostate,
                status,
            );

            /* if non-NULL, make sure the first URL is absolute or a full path */
            if mbrLocation1[0] != 0
                && fits_is_url_absolute(&mbrLocation1) == 0
                && mbrLocation1[0] != bb(b'/')
            {
                fits_get_cwd(&mut cwd, status);
                if strlen_safe(&cwd) + strlen_safe(&mbrLocation1) + 1 > FLEN_FILENAME - 1 {
                    ffpmsg_str("cwd and member locations are too long (ffgmf)");
                    *status = URL_PARSE_ERROR;
                    continue;
                }
                strcat_safe(&mut cwd, cs!(c"/"));
                strcat_safe(&mut cwd, &mbrLocation1);
                fits_clean_url(&cwd, &mut mbrLocation1, status);
            }

            /* if non-NULL, make sure the first URL is absolute or a full path */
            if mbrLocation2[0] != 0
                && fits_is_url_absolute(&mbrLocation2) == 0
                && mbrLocation2[0] != bb(b'/')
            {
                fits_get_cwd(&mut cwd, status);
                if strlen_safe(&cwd) + strlen_safe(&mbrLocation2) + 1 > FLEN_FILENAME - 1 {
                    ffpmsg_str("cwd and member locations are too long (ffgmf)");
                    *status = URL_PARSE_ERROR;
                    continue;
                }
                strcat_safe(&mut cwd, cs!(c"/"));
                strcat_safe(&mut cwd, &mbrLocation2);
                fits_clean_url(&cwd, &mut mbrLocation2, status);
            }
        }
        /*
          if the member location was specified, then make sure that it is
          either an absolute URL or specifies a full path
        */
        else if fits_is_url_absolute(&mbrLocation1) == 0 && mbrLocation1[0] != bb(b'/') {
            strcpy_safe(&mut mbrLocation2, &mbrLocation1);

            /* get the possible URLs for the grouping table file */
            *status = fits_get_url(
                gfptr,
                &mut grpLocation1,
                &mut grpLocation2,
                &mut access1,
                &mut access2,
                &mut iostate,
                status,
            );

            if grpLocation1[0] != 0 {
                /* make sure the first grouping table URL is absolute */
                if fits_is_url_absolute(&grpLocation1) == 0 && grpLocation1[0] != bb(b'/') {
                    fits_get_cwd(&mut cwd, status);
                    if strlen_safe(&cwd) + strlen_safe(&grpLocation1) + 1 > FLEN_FILENAME - 1 {
                        ffpmsg_str("cwd and group locations are too long (ffgmf)");
                        *status = URL_PARSE_ERROR;
                        continue;
                    }
                    strcat_safe(&mut cwd, cs!(c"/"));
                    strcat_safe(&mut cwd, &grpLocation1);
                    fits_clean_url(&cwd, &mut grpLocation1, status);
                }

                /* create an absoute URL for the member */

                fits_relurl2url(&grpLocation1, &mbrLocation1, &mut mbrLocation3, status);

                /*
                   if URL construction succeeded then copy it to the
                   first location string; else set the location string to
                   empty
                */

                if *status == 0 {
                    strcpy_safe(&mut mbrLocation1, &mbrLocation3);
                } else if *status == URL_PARSE_ERROR {
                    *status = 0;
                    mbrLocation1[0] = 0;
                }
            } else {
                mbrLocation1[0] = 0;
            }

            if grpLocation2[0] != 0 {
                /* make sure the second grouping table URL is absolute */
                if fits_is_url_absolute(&grpLocation2) == 0 && grpLocation2[0] != bb(b'/') {
                    fits_get_cwd(&mut cwd, status);
                    if strlen_safe(&cwd) + strlen_safe(&grpLocation2) + 1 > FLEN_FILENAME - 1 {
                        ffpmsg_str("cwd and group locations are too long (ffgmf)");
                        *status = URL_PARSE_ERROR;
                        continue;
                    }
                    strcat_safe(&mut cwd, cs!(c"/"));
                    strcat_safe(&mut cwd, &grpLocation2);
                    fits_clean_url(&cwd, &mut grpLocation2, status);
                }

                /* create an absolute URL for the member */

                fits_relurl2url(&grpLocation2, &mbrLocation2, &mut mbrLocation3, status);

                /*
                   if URL construction succeeded then copy it to the
                   second location string; else set the location string to
                   empty
                */

                if *status == 0 {
                    strcpy_safe(&mut mbrLocation2, &mbrLocation3);
                } else if *status == URL_PARSE_ERROR {
                    *status = 0;
                    mbrLocation2[0] = 0;
                }
            } else {
                mbrLocation2[0] = 0;
            }
        }

        /*
         compare the passed member HDU file location string with the
         (possibly two) member location strings to see if there is a match
        */

        if strcmp_safe(&mbrLocation1, &tmpLocation) != 0
            && strcmp_safe(&mbrLocation2, &tmpLocation) != 0
        {
            continue;
        }

        /* if we made it this far then a match to the member HDU was found */

        *member = i;
    }

    /* if a match was not found then set the return status code */

    if *member == 0 && *status == 0 {
        *status = MEMBER_NOT_FOUND;
        ffpmsg_str("Cannot find specified member HDU (ffgmf)");
    }

    *status
}

/*--------------------------------------------------------------------------
                      Recursive Group Functions
--------------------------------------------------------------------------*/

/// Recursively remove a grouping table and all its members. Each member of
/// the grouping table pointed to by gfptr it processed. If the member is itself
/// a grouping table then ffgtrmr() is recursively called to process all
/// of its members. The HDUtracker struct *HDU is used to make sure a member
/// is not processed twice, thus avoiding an infinite loop (e.g., a grouping
/// table contains itself as a member).
///
/// # Parameters
///
/// * `gfptr`  — FITS file pointer to group
/// * `HDU`    — list of processed HDUs
/// * `status` — return status code
pub(crate) fn ffgtrmr(gfptr: &mut fitsfile, HDU: &mut HDUtracker, status: &mut c_int) -> c_int {
    let mut i: c_int;
    let mut hdutype: c_int = 0;

    let mut nmembers: c_long = 0;

    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    let mut mfptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    /* get the number of members contained by this grouping table */

    *status = fits_get_num_members(gfptr, &mut nmembers, status);

    /* loop over all group members and delete them */

    i = nmembers as c_int;
    while i > 0 && *status == 0 {
        /* the body is wrapped in a loop so the C `continue` (which advances to
        the for-loop's `--i`) becomes `break`, after which `i` is decremented */
        #[allow(clippy::never_loop)] // transpiled do{}while(0)
        loop {
            /* open the member HDU */

            *status = fits_open_member(gfptr, i as c_long, &mut mfptr, status);

            /* if the member cannot be opened then just skip it and continue */

            if *status == MEMBER_NOT_FOUND {
                *status = 0;
                break;
            }

            /* Any other error is a reason to abort */

            if *status != 0 {
                break;
            }

            /* add the member HDU to the HDUtracker struct */

            *status = fftsad(mfptr.as_deref_mut().unwrap(), HDU, None, None);

            /* status == HDU_ALREADY_TRACKED ==> HDU has already been processed */

            if *status == HDU_ALREADY_TRACKED {
                *status = 0;
                if let Some(f) = mfptr.take() {
                    fits_close_file(f, status);
                }
                break;
            } else if *status != 0 {
                break;
            }

            /* determine if the member HDU is itself a grouping table */

            *status = fits_read_key_str(
                mfptr.as_deref_mut().unwrap(),
                cs!(c"EXTNAME"),
                &mut keyvalue,
                Some(&mut comment),
                status,
            );

            /* if no EXTNAME is found then the HDU cannot be a grouping table */

            if *status == KEY_NO_EXIST {
                *status = 0;
                keyvalue[0] = 0;
            }
            prepare_keyvalue(&mut keyvalue);

            /* Any other error is a reason to abort */

            if *status != 0 {
                break;
            }

            /*
            if the EXTNAME == GROUPING then the member is a grouping table
            and we must call ffgtrmr() to process its members
            */

            if fits_strcasecmp(&keyvalue, cs!(c"GROUPING")) == 0 {
                *status = ffgtrmr(mfptr.as_deref_mut().unwrap(), HDU, status);
            }

            /*
            unlink all the grouping tables that contain this HDU as a member
            and then delete the HDU (if not a PHDU)
            */

            if fits_get_hdu_num(mfptr.as_deref_mut().unwrap(), &mut hdutype) == 1 {
                *status = ffgmul(mfptr.as_deref_mut().unwrap(), 1, status);
            } else {
                *status = ffgmul(mfptr.as_deref_mut().unwrap(), 0, status);
                *status =
                    fits_delete_hdu(mfptr.as_deref_mut().unwrap(), Some(&mut hdutype), status);
            }

            /* close the fitsfile pointer */

            if let Some(f) = mfptr.take() {
                fits_close_file(f, status);
            }

            break;
        } //while(0) body

        i -= 1;
    }

    *status
}

/// Copy a Group to a new FITS file. If the cpopt parameter is set to
/// OPT_GCP_GPT (copy grouping table only) then the existing members have their
/// GRPIDn and GRPLCn keywords updated to reflect the existance of the new group,
/// since they now belong to another group. If cpopt is set to OPT_GCP_ALL
/// (copy grouping table and members recursively) then the original members are
/// not updated; the new grouping table is modified to include only the copied
/// member HDUs and not the original members.
///
/// Note that this function is recursive. When copt is OPT_GCP_ALL it will call
/// itself whenever a member HDU of the current grouping table is itself a
/// grouping table (i.e., EXTNAME = 'GROUPING').
///
/// # Parameters
///
/// * `infptr`  — input FITS file pointer
/// * `outfptr` — output FITS file pointer
/// * `cpopt`   — code specifying copy options: OPT_GCP_GPT (0) ==> cp only grouping
///              table OPT_GCP_ALL (2) ==> recusrively copy members and their members
///              (if groups)
/// * `HDU`     — list of already copied HDUs
/// * `status`  — return status code
pub(crate) fn ffgtcpr(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    cpopt: c_int,
    HDU: &mut HDUtracker,
    status: &mut c_int,
) -> c_int {
    let mut i: c_int;
    let nexclude: c_int = 8;
    let mut hdutype: c_int = 0;
    let mut groupHDUnum: c_int = 0;
    let mut numkeys: c_int = 0;
    let mut keypos: c_int = 0;
    let mut startSearch: c_int = 0;
    let mut newPosition: c_int = 0;

    let mut nmembers: c_long = 0;
    let mut tfields: c_long = 0;
    let mut newTfields: c_long = 0;

    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    /* C declares comment as FLEN_CARD; the safe read_key/longstr helpers want a
    FLEN_COMMENT buffer. It is also reused for the (discarded) comment output of
    the fits_read_key_* calls (C passed `card` there). */
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    let includeList: [&[c_char]; 1] = [cs!(c"*")];
    let excludeList: [&[c_char]; 8] = [
        cs!(c"EXTNAME"),
        cs!(c"EXTVER"),
        cs!(c"GRPNAME"),
        cs!(c"GRPID#"),
        cs!(c"GRPLC#"),
        cs!(c"THEAP"),
        cs!(c"TDIM#"),
        cs!(c"T????#"),
    ];

    let mut mfptr: Option<Box<fitsfile>> = None;

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    'outer: loop {
        /*
        create a new grouping table in the FITS file pointed to by outptr
        */

        *status = fits_get_num_members(infptr, &mut nmembers, status);

        *status = fits_read_key_str(
            infptr,
            cs!(c"GRPNAME"),
            &mut keyvalue,
            Some(&mut comment),
            status,
        );

        if *status == KEY_NO_EXIST {
            keyvalue[0] = 0;
            *status = 0;
        }
        prepare_keyvalue(&mut keyvalue);

        *status = fits_create_group(outfptr, &keyvalue, GT_ID_ALL_URI as c_int, status);

        /* save the new grouping table's HDU position for future use */

        fits_get_hdu_num(outfptr, &mut groupHDUnum);

        /* update the HDUtracker struct with the grouping table's new position */

        *status = fftsud(infptr, HDU, groupHDUnum, None);

        /*
        Now populate the copied grouping table depending upon the
        copy option parameter value
        */

        match cpopt as u64 {
            /*
            for the "copy grouping table only" option we only have to
            add the members of the original grouping table to the new
            grouping table
            */
            OPT_GCP_GPT => {
                i = 1;
                while (i as c_long) <= nmembers && *status == 0 {
                    *status = fits_open_member(infptr, i as c_long, &mut mfptr, status);
                    *status = fits_add_group_member(outfptr, mfptr.as_deref_mut(), 0, status);

                    if let Some(f) = mfptr.take() {
                        fits_close_file(f, status);
                    }
                    i += 1;
                }
            }

            OPT_GCP_ALL => {
                /*
                for the "copy the entire group" option
                */

                /* loop over all the grouping table members */

                i = 1;
                while (i as c_long) <= nmembers && *status == 0 {
                    /* body wrapped in a loop so the C `continue` (advancing the
                    for-loop's `++i`) becomes a `break` */
                    #[allow(clippy::never_loop)] // transpiled do{}while(0)
                    'body: loop {
                        /* open the ith member */

                        *status = fits_open_member(infptr, i as c_long, &mut mfptr, status);

                        if *status != 0 {
                            break 'body;
                        }

                        /* add it to the HDUtracker struct */

                        *status = fftsad(
                            mfptr.as_deref_mut().unwrap(),
                            HDU,
                            Some(&mut newPosition),
                            None,
                        );

                        /* if already copied then just add the member to the group */

                        if *status == HDU_ALREADY_TRACKED {
                            *status = 0;
                            *status = fits_add_group_member(outfptr, None, newPosition, status);
                            if let Some(f) = mfptr.take() {
                                fits_close_file(f, status);
                            }
                            break 'body;
                        } else if *status != 0 {
                            break 'body;
                        }

                        /* see if the member is a grouping table */

                        *status = fits_read_key_str(
                            mfptr.as_deref_mut().unwrap(),
                            cs!(c"EXTNAME"),
                            &mut keyvalue,
                            Some(&mut comment),
                            status,
                        );

                        if *status == KEY_NO_EXIST {
                            keyvalue[0] = 0;
                            *status = 0;
                        }
                        prepare_keyvalue(&mut keyvalue);

                        /*
                        if the member is a grouping table then copy it and all of
                        its members using ffgtcpr(), else copy it using
                        fits_copy_member(); the outptr will point to the newly
                        copied member upon return from both functions
                        */

                        if fits_strcasecmp(&keyvalue, cs!(c"GROUPING")) == 0 {
                            *status = ffgtcpr(
                                mfptr.as_deref_mut().unwrap(),
                                outfptr,
                                OPT_GCP_ALL as c_int,
                                HDU,
                                status,
                            );
                        } else {
                            *status = fits_copy_member(
                                infptr,
                                outfptr,
                                i as c_long,
                                OPT_MCP_NADD as c_int,
                                status,
                            );
                        }

                        /* retrieve the position of the newly copied member */

                        fits_get_hdu_num(outfptr, &mut newPosition);

                        /* update the HDUtracker struct with member's new position */

                        if fits_strcasecmp(&keyvalue, cs!(c"GROUPING")) != 0 {
                            *status = fftsud(mfptr.as_deref_mut().unwrap(), HDU, newPosition, None);
                        }

                        /* move the outfptr back to the copied grouping table HDU */

                        *status = fits_movabs_hdu(outfptr, groupHDUnum, Some(&mut hdutype), status);

                        /* add the copied member HDU to the copied grouping table */

                        *status = fits_add_group_member(outfptr, None, newPosition, status);

                        /* close the mfptr pointer */

                        if let Some(f) = mfptr.take() {
                            fits_close_file(f, status);
                        }

                        break 'body;
                    }

                    i += 1;
                }
            }

            _ => {
                *status = BAD_OPTION;
                ffpmsg_str("Invalid value specified for cmopt parameter (ffgtcpr)");
            }
        }

        if *status != 0 {
            break 'outer;
        }

        /*
        reposition the outfptr to the grouping table so that the grouping
        table is the CHDU upon return to the calling function
        */

        fits_movabs_hdu(outfptr, groupHDUnum, Some(&mut hdutype), status);

        /*
        copy all auxiliary keyword records from the original grouping table
        to the new grouping table; they are copied in their original order
        and inserted just before the TTYPE1 keyword record
        */

        *status = fits_read_card(outfptr, cs!(c"TTYPE1"), &mut card, status);
        *status = fits_get_hdrpos(outfptr, Some(&mut numkeys), Some(&mut keypos), status);
        keypos -= 1;

        startSearch = 8;

        while *status == 0 {
            fits_read_record(infptr, startSearch, Some(&mut card), status);

            *status = fits_find_nextkey(
                infptr,
                &includeList,
                1,
                &excludeList,
                nexclude,
                &mut card,
                status,
            );

            *status = fits_get_hdrpos(infptr, Some(&mut numkeys), Some(&mut startSearch), status);

            startSearch -= 1;
            /* SPR 1738 */
            if strncmp_safe(&card, cs!(c"GRPLC"), 5) != 0 {
                /* Not going to be a long string so we're ok */
                *status = fits_insert_record(outfptr, keypos, &card, status);
            } else {
                /* We could have a long string */
                *status = fits_read_record(infptr, startSearch, Some(&mut card), status);
                card[9] = b'\0' as c_char;
                let mut tkeyvalue: *mut c_char = core::ptr::null_mut();
                *status = fits_read_key_longstr(
                    infptr,
                    &card,
                    &mut tkeyvalue,
                    Some(&mut comment),
                    status,
                );
                if 0 == *status {
                    // SAFETY: tkeyvalue is the heap C string from ffgkls_safe,
                    // tracked in ALLOCATIONS; copy it out, use it, then release it.
                    unsafe {
                        let val =
                            cast_slice::<u8, c_char>(CStr::from_ptr(tkeyvalue).to_bytes_with_nul());
                        fits_insert_key_longstr(outfptr, &card, val, Some(&comment), status);
                        fits_write_key_longwarn(outfptr, status);
                        if let Some((l, c)) =
                            crate::lock_global(&ALLOCATIONS).remove(&(tkeyvalue as usize))
                        {
                            let _ = Vec::from_raw_parts(tkeyvalue, l, c);
                        }
                    }
                }
            }

            keypos += 1;
        }

        if *status == KEY_NO_EXIST {
            *status = 0;
        } else if *status != 0 {
            break 'outer;
        }

        /*
        search all the columns of the original grouping table and copy
        those to the new grouping table that were not part of the grouping
        convention. Note that is legal to have additional columns in a
        grouping table. Also note that the order of the columns may
        not be the same in the original and copied grouping table.
        */

        /* retrieve the number of columns in the original and new group tables */

        *status = fits_read_key_lng(
            infptr,
            cs!(c"TFIELDS"),
            &mut tfields,
            Some(&mut comment),
            status,
        );
        *status = fits_read_key_lng(
            outfptr,
            cs!(c"TFIELDS"),
            &mut newTfields,
            Some(&mut comment),
            status,
        );

        i = 1;
        while (i as c_long) <= tfields {
            int_snprintf!(&mut keyword, FLEN_KEYWORD, "TTYPE{}", i);
            *status =
                fits_read_key_str(infptr, &keyword, &mut keyvalue, Some(&mut comment), status);

            if *status == KEY_NO_EXIST {
                *status = 0;
                keyvalue[0] = 0;
            }
            prepare_keyvalue(&mut keyvalue);

            if fits_strcasecmp(&keyvalue, cs!(c"MEMBER_XTENSION")) != 0
                && fits_strcasecmp(&keyvalue, cs!(c"MEMBER_NAME")) != 0
                && fits_strcasecmp(&keyvalue, cs!(c"MEMBER_VERSION")) != 0
                && fits_strcasecmp(&keyvalue, cs!(c"MEMBER_POSITION")) != 0
                && fits_strcasecmp(&keyvalue, cs!(c"MEMBER_LOCATION")) != 0
                && fits_strcasecmp(&keyvalue, cs!(c"MEMBER_URI_TYPE")) != 0
            {
                /* SPR 3956, add at the end of the table */
                *status = fits_copy_col(infptr, outfptr, i, (newTfields + 1) as c_int, 1, status);
                newTfields += 1;
            }

            i += 1;
        }

        break 'outer;
    } //while(0)

    if let Some(f) = mfptr.take() {
        fits_close_file(f, status);
    }

    *status
}

/*--------------------------------------------------------------------------
              HDUtracker struct manipulation functions
--------------------------------------------------------------------------*/

/// Add an HDU to the HDUtracker struct pointed to by HDU. The HDU is only
/// added if it does not already reside in the HDUtracker. If it already
/// resides in the HDUtracker then the new HDU postion and file name are
/// returned in  newPosition and newFileName (if != NULL)
///
/// # Parameters
///
/// * `mfptr`       — pointer to an member HDU
/// * `HDU`         — pointer to an HDU tracker struct
/// * `newPosition` — new HDU position of the member HDU; None if not requested
/// * `newFileName` — file containing member HDU; None if not requested
pub(crate) fn fftsad(
    mfptr: &mut fitsfile,
    HDU: &mut HDUtracker,
    newPosition: Option<&mut c_int>,
    newFileName: Option<&mut [c_char]>,
) -> c_int {
    let mut i: c_int;
    let mut hdunum: c_int = 0;
    let mut status: c_int = 0;

    let mut filename1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut filename2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        /* retrieve the HDU's position within the FITS file */

        fits_get_hdu_num(mfptr, &mut hdunum);

        /* retrieve the HDU's file name */

        status = fits_file_name(mfptr, &mut filename1, &mut status);

        /* parse the file name and construct the "standard" URL for it */

        status = fits_parse_rootname(&filename1, &mut filename2, &mut status);

        /*
        examine all the existing HDUs in the HDUtracker an see if this HDU
        has already been registered
         */

        i = 0;
        while i < HDU.nHDU
            && !(HDU.position[i as usize] == hdunum
                && strcmp_safe(HDU.filename[i as usize].as_deref().unwrap(), &filename2) == 0)
        {
            i += 1;
        }

        if i != HDU.nHDU {
            status = HDU_ALREADY_TRACKED;
            if let Some(newPosition) = newPosition {
                *newPosition = HDU.newPosition[i as usize];
            }
            if let Some(newFileName) = newFileName {
                strcpy_safe(newFileName, HDU.newFilename[i as usize].as_deref().unwrap());
            }
            break;
        }

        if HDU.nHDU == MAX_HDU_TRACKER as c_int {
            status = TOO_MANY_HDUS_TRACKED;
            break;
        }

        // The C malloc()s filename[i] and newFilename[i] here, returning
        // MEMORY_ALLOCATION if either allocation fails. try_reserve_exact lets us catch
        // the failure; each Vec is then turned into the boxed buffer stored in the slot.
        let mut fbuf: Vec<c_char> = Vec::new();
        if fbuf.try_reserve_exact(FLEN_FILENAME).is_err() {
            status = MEMORY_ALLOCATION;
            break;
        }
        fbuf.resize(FLEN_FILENAME, 0);

        let mut nbuf: Vec<c_char> = Vec::new();
        if nbuf.try_reserve_exact(FLEN_FILENAME).is_err() {
            status = MEMORY_ALLOCATION;
            // fbuf is dropped here, mirroring the C free(HDU->filename[i])
            break;
        }
        nbuf.resize(FLEN_FILENAME, 0);

        HDU.position[i as usize] = hdunum;
        HDU.newPosition[i as usize] = hdunum;

        strcpy_safe(&mut fbuf, &filename2);
        strcpy_safe(&mut nbuf, &filename2);

        HDU.filename[i as usize] = Some(fbuf.into_boxed_slice().try_into().expect("Size mismatch"));
        HDU.newFilename[i as usize] =
            Some(nbuf.into_boxed_slice().try_into().expect("Size mismatch"));

        HDU.nHDU += 1;

        break;
    } // while(0)

    status
}

/// Update the HDU information in the HDUtracker struct pointed to by HDU. The
/// HDU to update is pointed to by mfptr. If non-zero, the value of newPosition
/// is used to update the HDU->newPosition[] value for the mfptr, and if
/// non-NULL the newFileName value is used to update the HDU->newFilename[]
/// value for mfptr.
///
/// # Parameters
///
/// * `mfptr`       — pointer to an member HDU
/// * `HDU`         — pointer to an HDU tracker struct
/// * `newPosition` — new HDU position of the member HDU; 0 to leave unchanged
/// * `newFileName` — file containing member HDU; None to leave unchanged
pub(crate) fn fftsud(
    mfptr: &mut fitsfile,
    HDU: &mut HDUtracker,
    newPosition: c_int,
    newFileName: Option<&[c_char]>,
) -> c_int {
    let mut i: c_int;
    let mut hdunum: c_int = 0;
    let mut status: c_int = 0;

    let mut filename1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut filename2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    /* retrieve the HDU's position within the FITS file */

    fits_get_hdu_num(mfptr, &mut hdunum);

    /* retrieve the HDU's file name */

    status = fits_file_name(mfptr, &mut filename1, &mut status);

    /* parse the file name and construct the "standard" URL for it */

    status = fits_parse_rootname(&filename1, &mut filename2, &mut status);

    /*
     examine all the existing HDUs in the HDUtracker an see if this HDU
     has already been registered
    */

    i = 0;
    while i < HDU.nHDU
        && !(HDU.position[i as usize] == hdunum
            && strcmp_safe(HDU.filename[i as usize].as_deref().unwrap(), &filename2) == 0)
    {
        i += 1;
    }

    /* if previously registered then change newPosition and newFileName */

    if i != HDU.nHDU {
        if newPosition != 0 {
            HDU.newPosition[i as usize] = newPosition;
        }
        if let Some(newFileName) = newFileName {
            // the slot is already Some for a tracked HDU; copy into its buffer
            strcpy_safe(
                HDU.newFilename[i as usize].as_deref_mut().unwrap(),
                newFileName,
            );
        }
    } else {
        status = MEMBER_NOT_FOUND;
    }

    status
}

/// Strip off all single quote characters "'" and blank spaces from a keyword
/// value retrieved via fits_read_key*() routines
///
/// this is necessary so that a standard comparision of keyword values may
/// be made
pub(crate) fn prepare_keyvalue(keyvalue: &mut [c_char]) /* string containing keyword value     */
{
    /*
      strip off all single quote characters "'" and blank spaces from a keyword
      value retrieved via fits_read_key*() routines

      this is necessary so that a standard comparision of keyword values may
      be made
    */

    let mut i: c_int;
    let mut length: c_int;

    /*
      strip off any leading or trailing single quotes (`) and (') from
      the keyword value
    */

    length = strlen_safe(keyvalue) as c_int - 1;

    if keyvalue[0] == bb(b'\'') && keyvalue[length as usize] == bb(b'\'') {
        i = 0;
        while i < length - 1 {
            keyvalue[i as usize] = keyvalue[(i + 1) as usize];
            i += 1;
        }
        keyvalue[(length - 1) as usize] = 0;
    }

    /*
      strip off any trailing blanks from the keyword value; note that if the
      keyvalue consists of nothing but blanks then no blanks are stripped
    */

    length = strlen_safe(keyvalue) as c_int - 1;

    i = 0;
    while i < length && keyvalue[i as usize] == bb(b' ') {
        i += 1;
    }

    if i != length {
        i = length;
        while i >= 0 && keyvalue[i as usize] == bb(b' ') {
            keyvalue[i as usize] = 0;
            i -= 1;
        }
    }
}

/*---------------------------------------------------------------------------
      Host dependent directory path to/from URL functions
--------------------------------------------------------------------------*/

/// Convert a file path into its Unix-style equivelent for URL
/// purposes. Note that this process is platform dependent. This
/// function supports Unix, MSDOS/WIN32, VMS and Macintosh platforms.
/// The plaform dependant code is conditionally compiled depending upon
/// the setting of the appropriate C preprocessor macros.
///
/// # Parameters
///
/// * `inpath`    — input file path string
/// * `maxlength` — I max number of chars that can be written to output, including
///                terminating NULL
/// * `outpath`   — output file path string
pub(crate) fn fits_path2url(
    inpath: &[c_char],
    maxlength: usize,
    outpath: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut buff: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    if cfg!(target_os = "windows") {
        /*
        MSDOS or Microsoft windows/NT case. The assumed form of the
        input path is:

        disk:\path\filename

        All path segments may be null, so that a single file name is the
        simplist case.

        All back-slashes '\' become slashes '/'; if the path starts with a
        string of the form "X:" then it is replaced with "/X/"
        */

        let mut i: usize = 0;
        let mut j: usize = 0;
        let mut k: isize = 0;
        let mut size = 0;

        if *status != 0 {
            return *status;
        }

        size = strlen_safe(inpath);
        buff[0] = 0;

        while i < size {
            match inpath[i] as u8 {
                b':' => {
                    /*
                    must be a disk desiginator; add a slash '/' at the start of
                    outpath to designate that the path is absolute, then change
                    the colon ':' to a slash '/'
                    */

                    k = j as isize;
                    while k >= 0 {
                        buff[(k + 1) as usize] = buff[k as usize];
                        k -= 1;
                    }

                    buff[0] = bb(b'/');
                    strcat_safe(&mut buff, cs!(c"/"));
                    i += 1;
                }

                b'\\' => {
                    /* just replace the '\' with a '/' IF its not the first character */

                    if i != 0 && buff[if j == 0 { 0 } else { j - 1 }] != bb(b'/') {
                        buff[j] = bb(b'/');
                        buff[j + 1] = 0;
                    }

                    i += 1;
                }
                _ => {
                    /* copy the character from inpath to buff as is */

                    buff[j] = inpath[i];
                    buff[j + 1] = 0;
                    i += 1;
                }
            }
            j = strlen_safe(&buff);
        }
    } else if cfg!(target_os = "macos") {
        /*
        MacOS case. The assumed form of the input path is:

        disk:path:filename

        It is assumed that all paths are absolute with disk and path specified,
        unless no colons ":" are supplied with the string ==> a single file name
        only. All colons ":" become slashes "/", and if one or more colon is
        encountered then the path is specified as absolute.
        */

        let mut i: usize = 0;
        let mut j: usize = 0;

        let mut firstColon: bool = true; // first colon encountered
        let mut size: usize = 0;

        if *status > 0 {
            return *status;
        }

        size = strlen_safe(inpath);
        buff[0] = 0;

        while i < size {
            match inpath[i] as u8 {
                b':' => {
                    /*
                    colons imply path separators. If its the first colon encountered
                    then assume that its the disk designator and add a slash to the
                    beginning of the buff string
                    */

                    if firstColon {
                        firstColon = false;

                        for k in (0..=j).rev() {
                            buff[k + 1] = buff[k];
                        }

                        buff[0] = bb(b'/');
                    }

                    /* all colons become slashes */

                    strcat_safe(&mut buff, cs!(c"/"));

                    i += 1;
                }
                _ => {
                    /* copy the character from inpath to buff as is */

                    buff[j] = inpath[i];
                    buff[j + 1] = 0;

                    i += 1;
                }
            }
            j = strlen_safe(&buff);
        }
    } else {
        /*
        Default Unix case.

        Nothing special to do here except to remove the double or more // and
        replace them with single /
        */

        let mut ii = 0;
        let mut jj = 0;

        if *status > 0 {
            return *status;
        }

        while inpath[ii] > 0 {
            if inpath[ii] == bb(b'/') && inpath[ii + 1] == bb(b'/') {
                /* do nothing */
            } else {
                buff[jj] = inpath[ii];
                jj += 1;
            }
            ii += 1;
        }
        buff[jj] = 0;
        /* printf("buff is %s\ninpath is %s\n",buff,inpath); */
        /* strcpy(buff,inpath); */
    }

    /*
    encode all "unsafe" and "reserved" URL characters
    */

    *status = fits_encode_url(&buff, maxlength, outpath, status);

    *status
}

/// Convert a Unix-style URL into a platform dependent directory path.
/// Note that this process is platform dependent. This
/// function supports Unix, MSDOS/WIN32, VMS and Macintosh platforms. Each
/// platform dependent code segment is conditionally compiled depending
/// upon the setting of the appropriate C preprocesser macros.
///
/// # Parameters
///
/// * `inpath`  — input file path string
/// * `outpath` — output file path string
pub(crate) fn fits_url2path(
    inpath: &[c_char],
    outpath: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut buff: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    if *status != 0 {
        return *status;
    }

    /*
      make a copy of the inpath so that we can manipulate it
    */

    strcpy_safe(&mut buff, inpath);

    /*
      convert any encoded characters to their unencoded values
    */

    *status = fits_unencode_url(inpath, &mut buff, status);

    /*
      see if the URL is given as absolute w.r.t. the "local" file system
    */

    let mut absolute = buff[0] == bb(b'/');

    // The C selects a platform branch with #if; mirror it with runtime cfg!() (as
    // fits_path2url does). The C `ffstrtok(buff,"/")` token loop becomes a split on
    // '/' that skips empty tokens (strtok merges consecutive/leading delimiters).
    // The VMS and Windows-NT (`//disk/...`) variants have no Rust target and are not
    // ported, matching fits_path2url.
    if cfg!(target_os = "windows") {
        /*
           MSDOS or Microsoft windows/NT case. The output path will be of the
           form

           disk:\path\filename

           All path segments but the last may be null, so that a single file name
           is the simplist case.
        */

        /*
          separate the URL into tokens at each slash '/' and process until
          all tokens have been examined
        */

        outpath[0] = 0;
        let len = strlen_safe(&buff);
        for tmpStr in buff[..len]
            .split(|&c| c == bb(b'/'))
            .filter(|t| !t.is_empty())
        {
            // strcat(outpath, tmpStr) — tmpStr is not NUL-terminated, so copy by length
            let l = strlen_safe(outpath);
            outpath[l..l + tmpStr.len()].copy_from_slice(tmpStr);
            outpath[l + tmpStr.len()] = 0;

            /*
               if the absolute flag is set then process the token as a disk
               specification; else just process it as a directory path or filename
            */

            if absolute {
                strcat_safe(outpath, cs!(c":\\"));
                absolute = false;
            } else {
                strcat_safe(outpath, cs!(c"\\"));
            }
        }

        /* remove the last "\" from the outpath, it does not belong there */

        let l = strlen_safe(outpath);
        if l > 0 {
            outpath[l - 1] = 0;
        }
    } else if cfg!(target_os = "macos") {
        /*
           MacOS case. The output path will be of the form

           disk:path:filename

           All path segments but the last may be null, so that a single file name
           is the simplist case.
        */

        /*
          separate the URL into tokens at each slash '/' and process until
          all tokens have been examined
        */

        outpath[0] = 0;
        let len = strlen_safe(&buff);
        for tmpStr in buff[..len]
            .split(|&c| c == bb(b'/'))
            .filter(|t| !t.is_empty())
        {
            let l = strlen_safe(outpath);
            outpath[l..l + tmpStr.len()].copy_from_slice(tmpStr);
            outpath[l + tmpStr.len()] = 0;
            strcat_safe(outpath, cs!(c":"));
        }

        /* remove the last ":" from the outpath, it does not belong there */

        let l = strlen_safe(outpath);
        if l > 0 {
            outpath[l - 1] = 0;
        }
    } else {
        /*
           Default Unix case.

           Nothing special to do here
        */

        strcpy_safe(outpath, &buff);
    }

    *status
}

/// Retrieve the string containing the current working directory absolute
/// path in Unix-like URL standard notation. It is assumed that the CWD
/// string has a size of at least FLEN_FILENAME.
///
/// Note that this process is platform dependent. This
/// function supports Unix, MSDOS/WIN32, VMS and Macintosh platforms. Each
/// platform dependent code segment is conditionally compiled depending
/// upon the setting of the appropriate C preprocesser macros.
///
/// # Parameters
///
/// * `cwd` — IO current working directory string
pub(crate) fn fits_get_cwd(cwd: &mut [c_char], status: &mut c_int) -> c_int {
    let mut buff: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let current_dir;

    if cwd.len() < FLEN_FILENAME {
        let msg = format!(
            "fits_get_cwd expects the buffer to be at least FLEN_FILENAME={FLEN_FILENAME} bytes long"
        );
        ffpmsg_str(&msg);
        *status = URL_PARSE_ERROR;
        return *status;
    }

    if *status != 0 {
        return *status;
    }

    if cfg!(target_os = "macos") {
        /*
        MacOS case. Currently unknown !!!!
        */
        current_dir = std::env::current_dir();

        if current_dir.is_err() {
            cwd[0] = 0;
            ffpmsg_str("Path and file name too long (fits_get_cwd)");
            *status = URL_PARSE_ERROR;
            return *status;
        }
    } else {
        /*
        Good old getcwd() seems to work with all other platforms
        */

        current_dir = std::env::current_dir();

        if current_dir.is_err() {
            cwd[0] = 0;
            ffpmsg_str("Path and file name too long (fits_get_cwd)");
            *status = URL_PARSE_ERROR;
            return *status;
        }
    }

    /*
    convert the cwd string to a URL standard path string
    */
    let current_dir = current_dir.unwrap();
    let cwd_bytes = current_dir.to_str().unwrap().as_bytes();
    for i in 0..cwd_bytes.len() {
        buff[i] = cwd_bytes[i] as c_char;
    }
    buff[cwd_bytes.len()] = 0;

    fits_path2url(&buff, FLEN_FILENAME, cwd, status);

    *status
}

/// For grouping convention purposes, determine the URL of the FITS file
/// associated with the fitsfile pointer fptr. The true access type (file://,
/// mem://, shmem://, root://), starting "official" access type, and iostate
/// (0 ==> readonly, 1 ==> readwrite) are also returned.
///
/// It is assumed that the url string has enough room to hold the resulting
/// URL, and the the accessType string has enough room to hold the access type.
#[allow(clippy::if_same_then_else)]
// C dispatch chain: distinct conditions deliberately share an action.
/// # Parameters
///
/// * `fptr`        — I ptr to FITS file to evaluate
/// * `realURL`     — O URL of real FITS file
/// * `startURL`    — O URL of starting FITS file
/// * `realAccess`  — O true access method of FITS file
/// * `startAccess` — O "official" access of FITS file
/// * `iostate`     — O can this file be modified?
pub(crate) fn fits_get_url(
    fptr: &mut fitsfile,
    realURL: &mut [c_char],
    startURL: &mut [c_char],
    realAccess: &mut [c_char],
    startAccess: &mut [c_char],
    iostate: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let i: usize;
    let mut tmpIOstate: c_int = 0;

    let mut infile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut outfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut tmpStr1: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut tmpStr2: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut tmpStr3: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut tmpStr4: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        /*
        retrieve the member HDU's file name as opened by ffopen()
        and parse it into its constitutent pieces; get the currently
        active driver token too
         */

        tmpStr1[0] = 0;
        tmpStr2[0] = 0;
        tmpStr3[0] = 0;
        tmpStr4[0] = 0;

        *status = fits_file_name(fptr, &mut tmpStr1, status);

        *status = fits_parse_input_url(
            &tmpStr1,
            None,
            Some(&mut infile),
            Some(&mut outfile),
            None,
            Some(&mut tmpStr2),
            Some(&mut tmpStr3),
            Some(&mut tmpStr4),
            status,
        );

        if tmpStr2[0] != 0 || tmpStr3[0] != 0 || tmpStr4[0] != 0 {
            tmpIOstate = -1;
        }

        *status = fits_url_type(fptr, &mut tmpStr3, status);

        strcpy_safe(&mut tmpStr4, &tmpStr3);

        *status = fits_parse_rootname(&tmpStr1, &mut tmpStr2, status);
        strcpy_safe(&mut tmpStr1, &tmpStr2);

        /*
        for grouping convention purposes (only) determine the URL of the
        actual FITS file being used for the given fptr, its true access
        type (file://, mem://, shmem://, root://) and its iostate (0 ==>
        read only, 1 ==> readwrite)
          */

        /*
        The first set of access types are "simple" in that they do not
        use any redirection to temporary memory or outfiles
          */

        /* standard disk file driver is in use */

        if fits_strcasecmp(&tmpStr3, cs!(c"file://")) == 0 {
            tmpIOstate = 1;

            if strlen_safe(&outfile) != 0 {
                strcpy_safe(&mut tmpStr1, &outfile);
            } else {
                tmpStr2[0] = 0;
            }

            /*
            make sure no FILE:// specifier is given in the tmpStr1
            or tmpStr2 strings; the convention calls for local files
            to have no access specification
            */

            if let Some(p) = strstr_safe(&tmpStr1, cs!(c"://")) {
                strcpy_safe(&mut infile, &tmpStr1[p + 3..]);
                strcpy_safe(&mut tmpStr1, &infile);
            }

            if let Some(p) = strstr_safe(&tmpStr2, cs!(c"://")) {
                strcpy_safe(&mut infile, &tmpStr2[p + 3..]);
                strcpy_safe(&mut tmpStr2, &infile);
            }
        }
        /* file stored in conventional memory */
        else if fits_strcasecmp(&tmpStr3, cs!(c"mem://")) == 0 {
            if tmpIOstate < 0 {
                /* file is a temp mem file only */
                ffpmsg_str("cannot make URL from temp MEM:// file (fits_get_url)");
                *status = URL_PARSE_ERROR;
            } else {
                /* file is a "perminate" mem file for this process */
                tmpIOstate = 1;
                tmpStr2[0] = 0;
            }
        }
        /* file stored in conventional memory */
        else if fits_strcasecmp(&tmpStr3, cs!(c"memkeep://")) == 0 {
            strcpy_safe(&mut tmpStr3, cs!(c"mem://"));
            tmpStr4[0] = 0;
            tmpStr2[0] = 0;
            tmpIOstate = 1;
        }
        /* file residing in shared memory */
        else if fits_strcasecmp(&tmpStr3, cs!(c"shmem://")) == 0 {
            tmpStr4[0] = 0;
            tmpStr2[0] = 0;
            tmpIOstate = 1;
        }
        /* file accessed via the ROOT network protocol */
        else if fits_strcasecmp(&tmpStr3, cs!(c"root://")) == 0 {
            tmpStr4[0] = 0;
            tmpStr2[0] = 0;
            tmpIOstate = 1;
        }
        /*
        the next set of access types redirect the contents of the original
        file to an special outfile because the original could not be
        directly modified (i.e., resides on the network, was compressed).
        In these cases the URL string takes on the value of the OUTFILE,
        the access type becomes file://, and the iostate is set to 1 (can
        read/write to the file).
          */
          /* compressed file uncompressed and written to disk */
        else if fits_strcasecmp(&tmpStr3, cs!(c"compressfile://")) == 0 {
            strcpy_safe(&mut tmpStr1, &outfile);
            strcpy_safe(&mut tmpStr2, &infile);
            strcpy_safe(&mut tmpStr3, cs!(c"file://"));
            strcpy_safe(&mut tmpStr4, cs!(c"file://"));
            tmpIOstate = 1;
        }
        /* HTTP accessed file written locally to disk */
        else if fits_strcasecmp(&tmpStr3, cs!(c"httpfile://")) == 0 {
            strcpy_safe(&mut tmpStr1, &outfile);
            strcpy_safe(&mut tmpStr3, cs!(c"file://"));
            strcpy_safe(&mut tmpStr4, cs!(c"http://"));
            tmpIOstate = 1;
        }
        /* FTP accessd file written locally to disk */
        else if fits_strcasecmp(&tmpStr3, cs!(c"ftpfile://")) == 0 {
            strcpy_safe(&mut tmpStr1, &outfile);
            strcpy_safe(&mut tmpStr3, cs!(c"file://"));
            strcpy_safe(&mut tmpStr4, cs!(c"ftp://"));
            tmpIOstate = 1;
        }
        /* file from STDIN written to disk */
        else if fits_strcasecmp(&tmpStr3, cs!(c"stdinfile://")) == 0 {
            strcpy_safe(&mut tmpStr1, &outfile);
            strcpy_safe(&mut tmpStr3, cs!(c"file://"));
            strcpy_safe(&mut tmpStr4, cs!(c"stdin://"));
            tmpIOstate = 1;
        }
        /*
        the following access types use memory resident files as temporary
        storage; they cannot be modified or be made group members for
        grouping conventions purposes, but their original files can be.
        Thus, their tmpStr3s are reset to mem://, their iostate
        values are set to 0 (for no-modification), and their URL string
        values remain set to their original values
          */
          /* compressed disk file uncompressed into memory */
        else if fits_strcasecmp(&tmpStr3, cs!(c"compress://")) == 0 {
            tmpStr1[0] = 0;
            strcpy_safe(&mut tmpStr2, &infile);
            strcpy_safe(&mut tmpStr3, cs!(c"mem://"));
            strcpy_safe(&mut tmpStr4, cs!(c"file://"));
            tmpIOstate = 0;
        }
        /* HTTP accessed file transferred into memory */
        else if fits_strcasecmp(&tmpStr3, cs!(c"http://")) == 0 {
            tmpStr1[0] = 0;
            strcpy_safe(&mut tmpStr3, cs!(c"mem://"));
            strcpy_safe(&mut tmpStr4, cs!(c"http://"));
            tmpIOstate = 0;
        }
        /* HTTP accessed compressed file transferred into memory */
        else if fits_strcasecmp(&tmpStr3, cs!(c"httpcompress://")) == 0 {
            tmpStr1[0] = 0;
            strcpy_safe(&mut tmpStr3, cs!(c"mem://"));
            strcpy_safe(&mut tmpStr4, cs!(c"http://"));
            tmpIOstate = 0;
        }
        /* FTP accessed file transferred into memory */
        else if fits_strcasecmp(&tmpStr3, cs!(c"ftp://")) == 0 {
            tmpStr1[0] = 0;
            strcpy_safe(&mut tmpStr3, cs!(c"mem://"));
            strcpy_safe(&mut tmpStr4, cs!(c"ftp://"));
            tmpIOstate = 0;
        }
        /* FTP accessed compressed file transferred into memory */
        else if fits_strcasecmp(&tmpStr3, cs!(c"ftpcompress://")) == 0 {
            tmpStr1[0] = 0;
            strcpy_safe(&mut tmpStr3, cs!(c"mem://"));
            strcpy_safe(&mut tmpStr4, cs!(c"ftp://"));
            tmpIOstate = 0;
        }
        /*
        The last set of access types cannot be used to make a meaningful URL
        strings from; thus an error is generated
          */
        else if fits_strcasecmp(&tmpStr3, cs!(c"stdin://")) == 0 {
            *status = URL_PARSE_ERROR;
            ffpmsg_str("cannot make valid URL from stdin:// (fits_get_url)");
            tmpStr1[0] = 0;
            tmpStr2[0] = 0;
        } else if fits_strcasecmp(&tmpStr3, cs!(c"stdout://")) == 0 {
            *status = URL_PARSE_ERROR;
            ffpmsg_str("cannot make valid URL from stdout:// (fits_get_url)");
            tmpStr1[0] = 0;
            tmpStr2[0] = 0;
        } else if fits_strcasecmp(&tmpStr3, cs!(c"irafmem://")) == 0 {
            *status = URL_PARSE_ERROR;
            ffpmsg_str("cannot make valid URL from irafmem:// (fits_get_url)");
            tmpStr1[0] = 0;
            tmpStr2[0] = 0;
        }

        if *status != 0 {
            break;
        }

        /*
        assign values to the calling parameters if they are non-NULL
         */
        // (In the safe interface the output buffers are always present.)

        if strlen_safe(&tmpStr1) == 0 {
            realURL[0] = 0;
        } else {
            // C: i is the length of the "scheme://" prefix copied verbatim (0 if none).
            i = match strstr_safe(&tmpStr1, cs!(c"://")) {
                Some(p) => {
                    let ii = p + 3;
                    strncpy_safe(realURL, &tmpStr1, ii);
                    ii
                }
                None => 0,
            };

            *status = fits_path2url(&tmpStr1[i..], FLEN_FILENAME - i, &mut realURL[i..], status);
        }

        if strlen_safe(&tmpStr2) == 0 {
            startURL[0] = 0;
        } else {
            let j = match strstr_safe(&tmpStr2, cs!(c"://")) {
                Some(p) => {
                    let jj = p + 3;
                    strncpy_safe(startURL, &tmpStr2, jj);
                    jj
                }
                None => 0,
            };

            *status = fits_path2url(&tmpStr2[j..], FLEN_FILENAME - j, &mut startURL[j..], status);
        }

        strcpy_safe(realAccess, &tmpStr3);
        strcpy_safe(startAccess, &tmpStr4);
        *iostate = tmpIOstate;

        break;
    } // while(0)

    *status
}

/*--------------------------------------------------------------------------
                         URL parse support functions
--------------------------------------------------------------------------*/

// HAVE NOT INCLUDED THIS SET OF FUNTIONS.

/// Clean the URL by eliminating any ".." or "." specifiers in the inURL
/// string, and write the output to the outURL string.
///
/// Note that this function must have a valid Unix-style URL as input; platform
/// dependent path strings are not allowed.
///
/// # Parameters
///
/// * `inURL`  — I input URL string
/// * `outURL` — O output URL string
pub(crate) fn fits_clean_url(inURL: &[c_char], outURL: &mut [c_char], status: &mut c_int) -> c_int {
    let mut mystack: VecDeque<&[c_char]> = VecDeque::new(); /* stack to hold pieces of URL */

    let mut inURL_ptr = inURL;

    if *status > 0 {
        return *status;
    }

    outURL[0] = 0;

    loop {
        /* handle URL scheme and domain if they exist */
        let tmp = strstr_safe(inURL_ptr, cs!(c"://"));

        if let Some(tmp) = tmp {
            // tmp is now the index into the current inURL_ptr

            /* there is a URL scheme, so look for the end of the domain too */
            let tmp_inner = strchr_safe(&inURL_ptr[(tmp + 3)..], b'/' as c_char);

            if let Some(tmp_inner) = tmp_inner {
                // tmp_inner is the index into tmp
                // adjust so that its the index into inURL_ptr

                let tmp_inner = tmp + tmp_inner + 3;

                /* tmp is now the end of the domain, so copy URL scheme and domain as is, and terminate by hand */
                let string_size = tmp_inner;

                strncpy_safe(outURL, &inURL_ptr[tmp_inner..], string_size);
                outURL[string_size] = 0;

                /* now advance the input pointer to just after the domain and go on */
                inURL_ptr = &inURL_ptr[tmp_inner..];
            } else {
                /* '/' was not found, which means there are no path-like
                 * portions, so copy whole inURL to outURL and we're done */
                strcpy_safe(outURL, inURL);
                continue; /* while(0) */
            }
        }

        /* explicitly copy a leading / (absolute path) */
        if bb(b'/') == inURL_ptr[0] {
            strcat_safe(outURL, cs!(c"/"));
        }

        /* now clean the remainder of the inURL. push URL segments onto stack, dealing with .. and . as we go */
        let first_null = inURL_ptr
            .iter()
            .position(|&x| x == 0)
            .unwrap_or(inURL_ptr.len());
        let tokens = inURL_ptr[..first_null].split(|f| (*f == b'/' as c_char) || (*f == 0));

        for tmp in tokens {
            if tmp.is_empty() {
                continue;
            }

            if strcmp_safe(tmp, cast_slice(b"..")) == 0 {
                // WARNING tokens are not null-terminated
                /*
                discard previous URL segment, if there was one. if not,
                add the .. to the stack if this is *not* an absolute path
                (for absolute paths, leading .. has no effect, so skip it)
                */
                if !mystack.is_empty() {
                    mystack.pop_back();
                } else if bb(b'/') != inURL_ptr[0] {
                    mystack.push_back(tmp);
                }
            } else {
                /* always just skip ., but otherwise add segment to stack */
                if strcmp_safe(tmp, cast_slice(b".")) > 0 {
                    mystack.push_back(tmp);
                }
            }
        }

        /*
        stack now has pieces of cleaned URL, so just catenate them
        onto output string until stack is empty
        */
        while !mystack.is_empty() {
            let tmp = mystack.pop_front().unwrap(); // This will be missing all the null terminators

            if strlen_safe(outURL) + tmp.len() + 1 > FLEN_FILENAME - 1 {
                outURL[0] = 0;
                ffpmsg_str("outURL is too long (fits_clean_url)");
                *status = URL_PARSE_ERROR;
                return *status;
            }

            // Note: We need to manually keep track of the len since we aren't using C null terminators here
            let len = strlen_safe(outURL) + tmp.len();

            let mut c = Vec::from(tmp);
            c.push(0); // Add a null terminator to the end of the tmp slice
            let tmp = c.as_slice(); // Convert back to a slice

            strcat_safe(outURL, tmp);

            outURL[len] = 0; // Manually add the null terminator

            strcat_safe(outURL, cs!(c"/"));
        }

        let outurl_len = strlen_safe(outURL);
        if outurl_len > 0 {
            outURL[strlen_safe(outURL) - 1] = 0; /* blank out trailing / */
        }
        break;
    }
    *status
}

/// Create a relative URL to the file referenced by absURL with respect to the
/// reference URL refURL. The relative URL is returned in relURL.
///
/// Both refURL and absURL must be absolute URL strings; i.e. either begin
/// with an access method specification "XXX://" or with a '/' character
/// signifiying that they are absolute file paths.
///
/// Note that it is possible to make a relative URL from two input URLs
/// (absURL and refURL) that are not compatable. This function does not
/// check to see if the resulting relative URL makes any sence. For instance,
/// it is impossible to make a relative URL from the following two inputs:
///
/// absURL = ftp://a.b.c.com/x/y/z/foo.fits
/// refURL = /a/b/c/ttt.fits
///
/// The resulting relURL will be:
///
/// ../../../ftp://a.b.c.com/x/y/z/foo.fits
///
/// Which is syntically correct but meaningless. The problem is that a file
/// with an access method of ftp:// cannot be expressed a a relative URL to
/// a local disk file.
///
/// # Parameters
///
/// * `refURL` — I reference URL string
/// * `absURL` — I absoulute URL string to process
/// * `relURL` — O resulting relative URL string
pub(crate) fn fits_url2relurl(
    refURL: &[c_char],
    absURL: &[c_char],
    relURL: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut i: c_int;
    let mut j: c_int;
    let mut refcount: c_int;
    let mut abscount: c_int;
    let refsize: c_int;
    let abssize: c_int;
    let mut done: c_int;

    if *status != 0 {
        return *status;
    }

    /* initialize the relative URL string */
    relURL[0] = 0;

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        /*
        refURL and absURL must be absolute to process
        */

        if !(fits_is_url_absolute(refURL) != 0 || refURL[0] == bb(b'/'))
            || !(fits_is_url_absolute(absURL) != 0 || absURL[0] == bb(b'/'))
        {
            *status = URL_PARSE_ERROR;
            ffpmsg_str("Cannot make rel. URL from non abs. URLs (fits_url2relurl)");
            break;
        }

        /* determine the size of the refURL and absURL strings */

        refsize = strlen_safe(refURL) as c_int;
        abssize = strlen_safe(absURL) as c_int;

        /* process the two URL strings and build the relative URL between them */

        done = 0;
        refcount = 0;
        abscount = 0;
        while done == 0 && refcount < refsize && abscount < abssize {
            while abscount < abssize && absURL[abscount as usize] == bb(b'/') {
                abscount += 1;
            }
            while refcount < refsize && refURL[refcount as usize] == bb(b'/') {
                refcount += 1;
            }

            /* find the next path segment in absURL */
            i = abscount;
            while absURL[i as usize] != bb(b'/') && i < abssize {
                i += 1;
            }

            /* find the next path segment in refURL */
            j = refcount;
            while refURL[j as usize] != bb(b'/') && j < refsize {
                j += 1;
            }

            /* do the two path segments match? */
            if i == j
                && strncmp_safe(
                    &absURL[abscount as usize..],
                    &refURL[refcount as usize..],
                    (i - refcount) as usize,
                ) == 0
            {
                /* they match, so ignore them and continue */
                abscount = i;
                refcount = j;
                /* the C for-loop increment (++refcount, ++abscount) runs on `continue` */
                refcount += 1;
                abscount += 1;
                continue;
            }

            /* We found a difference in the paths in refURL and absURL.
               For every path segment remaining in the refURL string, append
               a "../" path segment to the relataive URL relURL.
            */

            j = refcount;
            while j < refsize {
                if refURL[j as usize] == bb(b'/') {
                    if strlen_safe(relURL) + 3 > FLEN_FILENAME - 1 {
                        *status = URL_PARSE_ERROR;
                        ffpmsg_str("relURL too long (fits_url2relurl)");
                        return *status;
                    }
                    strcat_safe(relURL, cs!(c"../"));
                }
                j += 1;
            }

            /* copy all remaining characters of absURL to the output relURL */

            if strlen_safe(relURL) + strlen_safe(&absURL[abscount as usize..]) > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                ffpmsg_str("relURL too long (fits_url2relurl)");
                return *status;
            }
            strcat_safe(relURL, &absURL[abscount as usize..]);

            /* we are done building the relative URL */
            done = 1;
        }

        break;
    } //while(0)

    *status
}

/// Create an absolute URL from a relative url and a reference URL. The
/// reference URL is given by the FITS file pointed to by fptr.
///
/// The construction of the absolute URL from the partial and reference URl
/// is performed using the rules set forth in:
///
/// <http://www.w3.org/Addressing/URL/URL_TOC.html>
/// and
/// <http://www.w3.org/Addressing/URL/4_3_Partial.html>
///
/// Note that the relative URL string relURL must conform to the Unix-like
/// URL syntax; host dependent partial URL strings are not allowed.
///
/// # Parameters
///
/// * `refURL` — I reference URL string
/// * `relURL` — I relative URL string to process
/// * `absURL` — O absolute URL string
pub fn fits_relurl2url(
    refURL: &[c_char],
    relURL: &[c_char],
    absURL: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut i: usize;

    let mut tmpStr: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    if *status != 0 {
        return *status;
    }

    #[allow(clippy::never_loop)] // transpiled do{}while(0)
    loop {
        /*
        make a copy of the reference URL string refURL for parsing purposes
          */

        if strlen_safe(refURL) > FLEN_FILENAME - 1 {
            absURL[0] = 0;
            ffpmsg_str("ref URL is too long (fits_relurl2url)");
            *status = URL_PARSE_ERROR;
            break;
        }
        strcpy_safe(&mut tmpStr, refURL);

        /*
        if the reference file has an access method of mem:// or shmem://
        then we cannot use it as the basis of an absolute URL construction
        for a partial URL
          */

        if fits_strncasecmp(&tmpStr, cs!(c"MEM:"), 4) == 0
            || fits_strncasecmp(&tmpStr, cs!(c"SHMEM:"), 6) == 0
        {
            ffpmsg_str("ref URL has access mem:// or shmem:// (fits_relurl2url)");
            ffpmsg_str("   cannot construct full URL from a partial URL and ");
            ffpmsg_str("   MEM/SHMEM base URL");
            *status = URL_PARSE_ERROR;
            break;
        }

        if relURL[0] != bb(b'/') {
            /*
            just append the relative URL string to the reference URL
            string (minus the reference URL file name) to form the
            absolute URL string
              */

            // C: tmpStr1 = strrchr(tmpStr,'/'); if(tmpStr1) tmpStr1[1]=0; else tmpStr[0]=0;
            match strrchr_safe(&tmpStr, bb(b'/')) {
                Some(k) => tmpStr[k + 1] = 0,
                None => tmpStr[0] = 0,
            }

            if strlen_safe(&tmpStr) + strlen_safe(relURL) > FLEN_FILENAME - 1 {
                absURL[0] = 0;
                ffpmsg_str("rel + ref URL is too long (fits_relurl2url)");
                *status = URL_PARSE_ERROR;
                break;
            }
            strcat_safe(&mut tmpStr, relURL);
        } else {
            /*
            have to parse the refURL string for the first occurnace of the
            same number of '/' characters as contained in the beginning of
            location that is not followed by a greater number of consective
            '/' charaters (yes, that is a confusing statement); this is the
            location in the refURL string where the relURL string is to
            be appended to form the new absolute URL string
              */

            /*
            first, build up a slash pattern string that has one more
            slash in it than the starting slash pattern of the
            relURL string
              */

            strcpy_safe(absURL, cs!(c"/"));

            i = 0;
            while relURL[i] == bb(b'/') {
                if strlen_safe(absURL) + 1 > FLEN_FILENAME - 1 {
                    absURL[0] = 0;
                    ffpmsg_str("abs URL is too long (fits_relurl2url)");
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
                strcat_safe(absURL, cs!(c"/"));
                i += 1;
            }

            /*
            loop over the refURL string until the slash pattern stored
            in absURL is no longer found
              */

            // C: for(tmpStr1 = tmpStr, i = strlen(absURL);
            //         (tmpStr2 = strstr(tmpStr1,absURL)) != NULL; tmpStr1 = tmpStr2 + i);
            let mut tmpStr1: usize = 0;
            i = strlen_safe(absURL);
            loop {
                let start = tmpStr1.min(tmpStr.len());
                match strstr_safe(&tmpStr[start..], absURL) {
                    Some(rel) => {
                        let tmpStr2 = start + rel;
                        tmpStr1 = tmpStr2 + i;
                    }
                    None => break,
                }
            }

            /* reduce the slash pattern string by one slash */

            absURL[i - 1] = 0;

            /*
            search for the slash pattern in the remaining portion
            of the refURL string
             */

            let start = tmpStr1.min(tmpStr.len());
            match strstr_safe(&tmpStr[start..], absURL) {
                /* set a string terminator at the slash pattern match */
                Some(rel) => tmpStr[start + rel] = 0,
                None => {
                    /* just strip off the file name from the refURL  */
                    match strrchr_safe(&tmpStr[start..], bb(b'/')) {
                        Some(rel) => tmpStr[start + rel] = 0,
                        None => tmpStr[0] = 0,
                    }
                }
            }

            /*
            conatenate the relURL string to the refURL string to form
            the absURL
              */

            if strlen_safe(&tmpStr) + strlen_safe(relURL) > FLEN_FILENAME - 1 {
                absURL[0] = 0;
                ffpmsg_str("rel + ref URL is too long (fits_relurl2url)");
                *status = URL_PARSE_ERROR;
                break;
            }
            strcat_safe(&mut tmpStr, relURL);
        }

        /*
        normalize the absURL by removing any ".." or "." specifiers
        in the string
          */

        *status = fits_clean_url(&tmpStr, absURL, status);

        break;
    } // while(0)

    *status
}

/// Encode all URL "unsafe" and "reserved" characters using the "%XX"
/// convention, where XX stand for the two hexidecimal digits of the
/// encode character's ASCII code.
///
/// Note that the outpath length, as specified by the maxlength argument,
/// should be at least as large as inpath and preferably larger (to hold
/// any characters that need encoding).  If more than maxlength chars are
/// required for outpath, including the terminating NULL, outpath will
/// be set to size 0 and an error status will be returned.
///
/// This function was adopted from code in the libwww.a library available
/// via the W3 consortium, <http://www.w3.org>
///
/// # Parameters
///
/// * `inpath`    — I URL to be encoded
/// * `maxlength` — I max number of chars that may be copied to outpath, including
///                terminating NULL.
/// * `outpath`   — O output encoded URL
pub(crate) fn fits_encode_url(
    inpath: &[c_char],
    maxlength: usize,
    outpath: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut a: c_uchar;

    let mut p = 0;
    let mut q = 0;
    let hex: &[c_char] = cast_slice(b"0123456789ABCDEF");
    let mut iout = 0;

    let isAcceptable: [c_uchar; 96] = [
        /* 0x0 0x1 0x2 0x3 0x4 0x5 0x6 0x7 0x8 0x9 0xA 0xB 0xC 0xD 0xE 0xF */
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xF, 0xE, 0x0, 0xF, 0xF,
        0xC, /* 2x  !"#$%&'()*+,-./   */
        0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0x8, 0x0, 0x0, 0x0, 0x0,
        0x0, /* 3x 0123456789:;<=>?   */
        0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF,
        0xF, /* 4x @ABCDEFGHIJKLMNO   */
        0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0x0, 0x0, 0x0, 0x0,
        0xF, /* 5X PQRSTUVWXYZ[\]^_   */
        0x0, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF,
        0xF, /* 6x `abcdefghijklmno   */
        0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0x0, 0x0, 0x0, 0x0,
        0x0, /* 7X pqrstuvwxyz{\}~DEL */
    ];

    if *status != 0 {
        return *status;
    }

    /* loop over all characters in inpath until 0 is encountered */

    while inpath[p] > 0 && (iout < maxlength - 1) {
        a = inpath[p] as u8;

        /* if the charcter requires encoding then process it */

        if !(a >= 32 && a < 128 && (isAcceptable[a as usize - 32] > 0)) {
            if iout + 2 < maxlength - 1 {
                /* add a '%' character to the outpath */
                outpath[q] = HEX_ESCAPE as c_char;
                q += 1;

                /* add the most significant ASCII code hex value */
                outpath[q] = hex[(a >> 4) as usize];
                q += 1;

                /* add the least significant ASCII code hex value */
                outpath[q] = hex[(a & 15) as usize];
                q += 1;
                iout += 3;
            } else {
                ffpmsg_str("URL input is too long to encode (fits_encode_url)");
                *status = URL_PARSE_ERROR;
                outpath[0] = 0;
                return *status;
            }
        }
        /* else just copy the character as is */
        else {
            outpath[q] = inpath[p];
            q += 1;
            iout += 1;
        }
        p += 1;
    }

    /* null terminate the outpath string */

    if inpath[p] > 0 && (iout == maxlength - 1) {
        ffpmsg_str("URL input is too long to encode (fits_encode_url)");
        *status = URL_PARSE_ERROR;
        outpath[0] = 0;
        return *status;
    }

    outpath[q] = 0;
    q += 1;

    *status
}

/// Unencode all URL "unsafe" and "reserved" characters to their actual
/// ASCII representation. All tokens of the form "%XX" where XX is the
/// hexidecimal code for an ASCII character, are searched for and
/// translated into the actuall ASCII character (so three chars become
/// 1 char).
///
/// It is assumed that OUTPATH has enough room to hold the unencoded
/// URL.
///
/// This function was adopted from code in the libwww.a library available
/// via the W3 consortium, <http://www.w3.org>
///
/// # Parameters
///
/// * `inpath`  — I input URL with encoding
/// * `outpath` — O unencoded URL
pub(crate) fn fits_unencode_url(
    inpath: &[c_char],
    outpath: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    if *status != 0 {
        return *status;
    }

    // Numeric value of a hex digit (0-15); mirrors the C ternary chain. Computed in
    // u8 so the `*16` / `+` reconstruction below cannot overflow a signed c_char.
    let hexval = |c: c_char| -> u8 {
        if (bb(b'0')..=bb(b'9')).contains(&c) {
            (c - bb(b'0')) as u8
        } else if (bb(b'A')..=bb(b'F')).contains(&c) {
            (c - bb(b'A') + 10) as u8
        } else {
            (c - bb(b'a') + 10) as u8
        }
    };

    let mut p = 0; /* index into inpath  (C: char *p) */
    let mut q = 0; /* index into outpath (C: char *q) */

    /*
       loop over all characters in the inpath looking for the '%' escape
       character; if found the process the escape sequence
    */

    while inpath[p] != 0 {
        /*
           if the character is '%' then unencode the sequence, else
           just copy the character from inpath to outpath
        */

        if inpath[p] == HEX_ESCAPE as c_char {
            p += 1;
            let mut c = inpath[p]; /* c = *(++p) */
            if c != 0 {
                outpath[q] = (hexval(c) << 4) as c_char; /* *q = hexval(c) * 16 */

                p += 1;
                c = inpath[p]; /* c = *(++p) */
                if c != 0 {
                    outpath[q] = (outpath[q] as u8 + hexval(c)) as c_char; /* *q += hexval(c) */
                    p += 1;
                    q += 1;
                }
            }
        } else {
            outpath[q] = inpath[p]; /* *q++ = *p++ */
            q += 1;
            p += 1;
        }
    }

    /* terminate the outpath */
    outpath[q] = 0;

    *status
}

/// Return a True (1) or False (0) value indicating whether or not the passed
/// URL string contains an access method specifier or not. Note that this is
/// a boolean function and it neither reads nor returns the standard error
/// status parameter
/// Note that this doens't validate the URL, it just checks for the presence
pub fn fits_is_url_absolute(url: &[c_char]) -> c_int {
    let reserved: [c_char; 10] = [
        bb(b':'),
        bb(b';'),
        bb(b'/'),
        bb(b'?'),
        bb(b'@'),
        bb(b'&'),
        bb(b'='),
        bb(b'+'),
        bb(b'$'),
        bb(b','),
    ];

    /*
     The rule for determing if an URL is relative or absolute is that it (1)
     must have a colon ":" and (2) that the colon must appear before any other
     reserved URL character in the URL string. We first see if a colon exists,
     get its position in the string, and then check to see if any of the other
     reserved characters exists and if their position in the string is greater
     than that of the colons.
    */

    if let Some(colon_pos) = url.iter().position(|&c| c == reserved[0])
        && reserved[1..].iter().all(|&c| {
            url.iter()
                .position(|&x| x == c)
                .is_none_or(|pos| pos > colon_pos)
        })
    {
        return 1;
    }
    0
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fits_is_url_absolute() {
        assert_eq!(fits_is_url_absolute(cs!(c"http://example.com")), 1);
        assert_eq!(fits_is_url_absolute(cs!(c"ftp://example.com")), 1);
        assert_eq!(fits_is_url_absolute(cs!(c"file:///path/to/file")), 1);
        assert_eq!(fits_is_url_absolute(cs!(c"mem://memory")), 1);
        assert_eq!(fits_is_url_absolute(cs!(c"shmem://shared_memory")), 1);
        assert_eq!(fits_is_url_absolute(cs!(c"root://root_path")), 1);
        assert_eq!(fits_is_url_absolute(cs!(c"/absolute/path")), 0); // No colon
        assert_eq!(fits_is_url_absolute(cs!(c"relative/path")), 0);
        assert_eq!(fits_is_url_absolute(cs!(c"C:/absolute/path")), 1);
        assert_eq!(fits_is_url_absolute(cs!(c"")), 0);
        assert_eq!(fits_is_url_absolute(cs!(c"no_colon_here")), 0);
        assert_eq!(fits_is_url_absolute(cs!(c"http:/example.com")), 1);
        assert_eq!(fits_is_url_absolute(cs!(c"http:example.com")), 1);
    }

    #[test]
    fn test_fits_clean_url() {
        let mut status = 0;
        let mut out_url = [0 as c_char; FLEN_FILENAME];

        // Test cases
        let test_cases = vec![
            ("/dir/../file.fits", "/file.fits", 0),
            ("/home/user/../file.fits", "/home/file.fits", 0),
            ("/home//user///file.fits", "/home/user/file.fits", 0),
            ("", "", 0),
            ("././.", "", 0),
            ("dir/../file?.fits", "file?.fits", 0),
            ("/dir/./file.fits", "/dir/file.fits", 0),
            ("/dir/././file.fits", "/dir/file.fits", 0),
            ("/dir/../dir2/../file.fits", "/file.fits", 0),
            ("/dir/dir2/../../file.fits", "/file.fits", 0),
            ("dir/./file.fits", "dir/file.fits", 0),
            ("dir/../dir2/file.fits", "dir2/file.fits", 0),
            ("dir/dir2/../../file.fits", "file.fits", 0),
            ("dir/dir2/../../../file.fits", "../file.fits", 0),
            ("/dir/dir2/../../../file.fits", "/file.fits", 0),
            ("dir/dir2/../../dir3/./file.fits", "dir3/file.fits", 0),
            ("dir/dir2/../../dir3/../file.fits", "file.fits", 0),
            ("dir/dir2/../../dir3/.././file.fits", "file.fits", 0),
            ("/dir/dir2/../../dir3/./file.fits", "/dir3/file.fits", 0),
            ("/dir/dir2/../../dir3/../file.fits", "/file.fits", 0),
            ("/dir/dir2/../../dir3/.././file.fits", "/file.fits", 0),
        ];

        for (input, expected, expected_status) in test_cases {
            let mut input_cstr = [0 as c_char; FLEN_FILENAME];
            for (i, &byte) in input.as_bytes().iter().enumerate() {
                input_cstr[i] = byte as c_char;
            }

            fits_clean_url(&input_cstr, &mut out_url, &mut status);

            let output = CStr::from_bytes_until_nul(cast_slice(&out_url))
                .unwrap()
                .to_str()
                .unwrap();
            assert_eq!(output, expected);
            assert_eq!(status, expected_status);
        }
    }

    #[test]
    fn test_prepare_keyvalue() {
        // (input, expected) pairs, derived by hand-tracing the C:
        //  - leading+trailing single quotes are stripped only if BOTH are present
        //  - trailing blanks are stripped, UNLESS the value is entirely blanks
        //  - leading blanks are never stripped
        let cases = [
            ("'GROUPING'", "GROUPING"), // surrounding quotes removed
            ("'GROUP '", "GROUP"),      // quotes removed, then trailing blank stripped
            ("VALUE   ", "VALUE"),      // trailing blanks stripped (no quotes)
            ("ABC", "ABC"),             // nothing to strip
            ("'ABC", "'ABC"),           // unmatched leading quote: left untouched
            ("''", ""),                 // empty quoted string collapses to empty
            ("   ", "   "),             // all-blank value is preserved
            ("  AB  ", "  AB"),         // leading blanks kept, trailing stripped
        ];

        for (input, expected) in cases {
            let mut buf = to_buf(input);
            prepare_keyvalue(&mut buf);
            assert_eq!(from_buf(&buf), expected, "prepare_keyvalue({input:?})");
        }
    }

    #[test]
    fn test_fits_url2relurl() {
        // (refURL, absURL, expected relURL) triples, derived by hand-tracing the C.
        // The result is the path from refURL's location to absURL: shared leading
        // path segments are dropped, one "../" is emitted per remaining segment of
        // refURL, then the unmatched tail of absURL is appended.
        let cases = [
            ("/a/b/c.fits", "/a/b/d.fits", "d.fits"), // same dir -> bare filename
            ("/a/b/c.fits", "/a/x/d.fits", "../x/d.fits"), // up one, into sibling dir
            ("/a/b.fits", "/a/c/d.fits", "c/d.fits"), // target is in a deeper dir
            ("/a/b/c/file.fits", "/a/x.fits", "../../x.fits"), // up two dirs
            ("/a/file.fits", "/a/file.fits", ""),     // identical URLs -> empty
        ];

        for (refurl, absurl, expected) in cases {
            let mut status = 0;
            let refbuf = to_buf(refurl);
            let absbuf = to_buf(absurl);
            let mut relbuf = [0 as c_char; FLEN_FILENAME];
            fits_url2relurl(&refbuf, &absbuf, &mut relbuf, &mut status);
            assert_eq!(status, 0, "status for ({refurl:?}, {absurl:?})");
            assert_eq!(
                from_buf(&relbuf),
                expected,
                "relURL for ({refurl:?}, {absurl:?})"
            );
        }

        // Non-absolute URLs cannot be processed -> URL_PARSE_ERROR, relURL emptied.
        let mut status = 0;
        let refbuf = to_buf("a/b"); // not absolute (no access method, no leading '/')
        let absbuf = to_buf("/c/d");
        let mut relbuf = [0 as c_char; FLEN_FILENAME];
        fits_url2relurl(&refbuf, &absbuf, &mut relbuf, &mut status);
        assert_eq!(status, URL_PARSE_ERROR);
        assert_eq!(from_buf(&relbuf), "");
    }

    #[test]
    fn test_fits_unencode_url() {
        let mut status = 0;

        let cases = [
            ("abc", "abc"),       // nothing to decode
            ("%20", " "),         // space
            ("a%2Fb", "a/b"),     // '/' in the middle
            ("%41%42%43", "ABC"), // consecutive escapes
            ("100%25", "100%"),   // literal percent
            ("%2f", "/"),         // lower-case hex digits
            ("hello%20world", "hello world"),
            ("abc%", "abc"), // trailing '%' with no digits
            ("%2", ""),      // trailing '%' with one digit -> dropped
            ("", ""),        // empty input
        ];

        for (input, expected) in cases {
            let in_buf = to_buf(input);
            let mut out_buf = [0 as c_char; FLEN_FILENAME];

            let r = fits_unencode_url(&in_buf, &mut out_buf, &mut status);

            assert_eq!(r, 0, "input={input:?}");
            assert_eq!(status, 0, "input={input:?}");
            assert_eq!(from_buf(&out_buf), expected, "input={input:?}");
        }
    }

    #[test]
    fn test_fits_relurl2url() {
        let mut status = 0;
        let mut abs_url = [0 as c_char; FLEN_FILENAME];

        // (refURL, relURL, expected absURL). Note the final result is passed through
        // fits_clean_url(), which normalizes "."/".." and collapses repeated slashes.
        let cases = [
            // a plain relative name replaces the reference file name
            ("/data/file1.fits", "file2.fits", "/data/file2.fits"),
            ("dir/sub/old.fits", "new.fits", "dir/sub/new.fits"),
            // ".." in the relative URL is normalized away by fits_clean_url
            ("/a/b/c.fits", "../d.fits", "/a/d.fits"),
            // a relative URL beginning with '/' is resolved against the path root
            ("/a/b/c.fits", "/x.fits", "/x.fits"),
        ];

        for (refurl, relurl, expected) in cases {
            let refbuf = to_buf(refurl);
            let relbuf = to_buf(relurl);
            abs_url.fill(0);
            status = 0;

            let r = fits_relurl2url(&refbuf, &relbuf, &mut abs_url, &mut status);

            assert_eq!(r, 0, "ref={refurl:?} rel={relurl:?}");
            assert_eq!(status, 0, "ref={refurl:?} rel={relurl:?}");
            assert_eq!(
                from_buf(&abs_url),
                expected,
                "ref={refurl:?} rel={relurl:?}"
            );
        }

        // a mem:// reference URL cannot be used to build an absolute URL
        status = 0;
        abs_url.fill(0);
        let refbuf = to_buf("mem://block");
        let relbuf = to_buf("file.fits");
        fits_relurl2url(&refbuf, &relbuf, &mut abs_url, &mut status);
        assert_eq!(status, URL_PARSE_ERROR);
    }

    #[test]
    fn test_fits_url2path() {
        // fits_url2path decodes any %XX escapes and then converts the Unix-style URL
        // into a platform-dependent path. The expected output therefore depends on the
        // build target (Windows: disk:\path ; macOS: disk:path: ; Unix: unchanged).
        let mut status = 0;

        let cases: &[(&str, &str)] = if cfg!(target_os = "windows") {
            &[
                ("/data/file.fits", "data:\\file.fits"),
                ("a%20b", "a b"),
                ("%2Ftmp%2Ffile.fits", "tmp:\\file.fits"),
                ("relative/path.fits", "relative\\path.fits"),
            ]
        } else if cfg!(target_os = "macos") {
            &[
                ("/data/file.fits", "data:file.fits"),
                ("a%20b", "a b"),
                ("%2Ftmp%2Ffile.fits", "tmp:file.fits"),
                ("relative/path.fits", "relative:path.fits"),
            ]
        } else {
            &[
                ("/data/file.fits", "/data/file.fits"),
                ("a%20b", "a b"),
                ("%2Ftmp%2Ffile.fits", "/tmp/file.fits"),
                ("relative/path.fits", "relative/path.fits"),
            ]
        };

        for &(input, expected) in cases {
            let in_buf = to_buf(input);
            let mut out_buf = [0 as c_char; FLEN_FILENAME];
            status = 0;

            let r = fits_url2path(&in_buf, &mut out_buf, &mut status);

            assert_eq!(r, 0, "input={input:?}");
            assert_eq!(status, 0, "input={input:?}");
            assert_eq!(from_buf(&out_buf), expected, "input={input:?}");
        }
    }

    #[test]
    fn test_fftsad_and_fftsud() {
        use crate::cfileio::{ffclos_safe, ffinit_safe};

        let mut status = 0;

        // create an in-memory FITS file (with an empty primary HDU at position 1)
        let mut fptr: Option<Box<fitsfile>> = None;
        ffinit_safe(&mut fptr, cs!(c"mem://"), &mut status);
        assert_eq!(status, 0, "ffinit failed");
        let mfptr = fptr.as_deref_mut().unwrap();

        let mut hdu = HDUtracker::default();

        // first registration: succeeds and records one HDU at position 1
        let r = fftsad(mfptr, &mut hdu, None, None);
        assert_eq!(r, 0);
        assert_eq!(hdu.nHDU, 1);
        assert_eq!(hdu.position[0], 1);
        assert_eq!(hdu.newPosition[0], 1);

        // registering the same HDU again reports HDU_ALREADY_TRACKED and does not
        // add a second entry; the tracked newPosition is returned
        let mut new_position = 0;
        let mut new_filename = [0 as c_char; FLEN_FILENAME];
        let r = fftsad(
            mfptr,
            &mut hdu,
            Some(&mut new_position),
            Some(&mut new_filename),
        );
        assert_eq!(r, HDU_ALREADY_TRACKED);
        assert_eq!(hdu.nHDU, 1);
        assert_eq!(new_position, 1);

        // fftsud updates the new position of the tracked HDU
        let r = fftsud(mfptr, &mut hdu, 5, None);
        assert_eq!(r, 0);
        assert_eq!(hdu.newPosition[0], 5);

        // fftsud on an HDUtracker that doesn't contain this HDU -> MEMBER_NOT_FOUND
        let mut empty = HDUtracker::default();
        let r = fftsud(mfptr, &mut empty, 5, None);
        assert_eq!(r, MEMBER_NOT_FOUND);

        let f = fptr.take().unwrap();
        ffclos_safe(f, &mut status);
    }

    #[test]
    fn test_ffgmng_safe() {
        use crate::cfileio::{ffclos_safe, ffinit_safe};

        let mut status = 0;

        // a fresh in-memory file's HDU has no GRPIDn keywords, so it belongs to no groups
        let mut fptr: Option<Box<fitsfile>> = None;
        ffinit_safe(&mut fptr, cs!(c"mem://"), &mut status);
        assert_eq!(status, 0, "ffinit failed");
        let mfptr = fptr.as_deref_mut().unwrap();

        let mut ngroups: c_long = 42; // deliberately non-zero to check it gets reset
        let r = ffgmng_safe(mfptr, &mut ngroups, &mut status);

        assert_eq!(r, 0);
        assert_eq!(status, 0);
        assert_eq!(ngroups, 0);

        let f = fptr.take().unwrap();
        ffclos_safe(f, &mut status);
    }

    // ---------------------------------------------------------------------
    // Port of the integration tests in CFITSIO's test_group.c. Functions that
    // only create/operate/close a single handle use an in-memory file
    // (`mem://`); tests that re-open the file by name (extended `[GROUPING]`
    // syntax) use a real temp file via `with_temp_file`.
    // ---------------------------------------------------------------------

    use crate::cfileio::{ffclos_safe, ffinit_safe, ffopen_safe};
    use crate::helpers::testhelpers::{
        from_buf, new_mem_file, path_with_ext, read_str_key, to_buf, with_temp_file,
    };

    #[test]
    fn test_ffgtcr_basic() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        /* Create a grouping table at end of file */
        fits_create_group(f, cs!(c"TestGroup"), GT_ID_ALL_URI as c_int, &mut status);
        assert_eq!(status, 0, "ffgtcr failed");

        /* Should now be at the grouping table HDU */
        let mut hdutype = 0;
        fits_get_hdu_type(f, &mut hdutype, &mut status);
        assert_eq!(hdutype, BINARY_TBL);
        assert_eq!(read_str_key(f, cs!(c"EXTNAME"), &mut status), "GROUPING");

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtcr_with_name() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        fits_create_group(f, cs!(c"MyGroup"), GT_ID_REF as c_int, &mut status);
        assert_eq!(status, 0, "ffgtcr failed");
        assert_eq!(read_str_key(f, cs!(c"GRPNAME"), &mut status), "MyGroup");

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtis_insert() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        /* Insert grouping table after primary HDU */
        fits_insert_group(f, cs!(c"InsertedGroup"), GT_ID_POS as c_int, &mut status);
        assert_eq!(status, 0, "ffgtis failed");

        let mut hdunum = 0;
        fits_get_hdu_num(f, &mut hdunum);
        assert_eq!(hdunum, 2);

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtis_all_types() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        /* GT_ID_ALL_URI - all columns including URI */
        fits_insert_group(f, cs!(c"AllURI"), GT_ID_ALL_URI as c_int, &mut status);
        let mut ncols = 0;
        fits_get_num_cols(f, &mut ncols, &mut status);
        assert!(ncols >= 5, "expected >=5 cols, got {ncols}");

        /* Go back to primary and insert another type */
        fits_movabs_hdu(f, 1, None, &mut status);
        fits_insert_group(f, cs!(c"RefOnly"), GT_ID_REF as c_int, &mut status);

        fits_movabs_hdu(f, 1, None, &mut status);
        fits_insert_group(f, cs!(c"PosOnly"), GT_ID_POS as c_int, &mut status);

        fits_movabs_hdu(f, 1, None, &mut status);
        fits_insert_group(f, cs!(c"AllCols"), GT_ID_ALL as c_int, &mut status);

        fits_movabs_hdu(f, 1, None, &mut status);
        fits_insert_group(f, cs!(c"RefURI"), GT_ID_REF_URI as c_int, &mut status);

        fits_movabs_hdu(f, 1, None, &mut status);
        fits_insert_group(f, cs!(c"PosURI"), GT_ID_POS_URI as c_int, &mut status);

        assert_eq!(status, 0, "ffgtis all-types failed");

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtnm_empty_group() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        fits_create_group(f, cs!(c"EmptyGroup"), GT_ID_ALL_URI as c_int, &mut status);
        let mut nmembers: c_long = -1;
        fits_get_num_members(f, &mut nmembers, &mut status);
        assert_eq!(status, 0);
        assert_eq!(nmembers, 0);

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtch_change_type() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        /* Create with minimal columns */
        fits_insert_group(f, cs!(c"ChangeType"), GT_ID_REF as c_int, &mut status);
        let mut ncols_before = 0;
        fits_get_num_cols(f, &mut ncols_before, &mut status);

        /* Change to include position info */
        fits_change_group(f, GT_ID_ALL as c_int, &mut status);
        let mut ncols_after = 0;
        fits_get_num_cols(f, &mut ncols_after, &mut status);
        assert_eq!(status, 0, "ffgtch failed");
        assert!(
            ncols_after > ncols_before,
            "expected more cols after change: before={ncols_before} after={ncols_after}"
        );

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtvf_valid_group() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        fits_create_group(f, cs!(c"ValidGroup"), GT_ID_ALL_URI as c_int, &mut status);
        let mut firstfailed: c_long = -1;
        fits_verify_group(f, &mut firstfailed, &mut status);
        assert_eq!(status, 0, "ffgtvf failed");
        /* Empty group should verify OK, firstfailed = 0 means all OK */
        assert_eq!(firstfailed, 0);

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgmng_no_groups() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        /* Create an image extension */
        let naxes: [c_long; 2] = [10, 10];
        fits_create_img(f, BYTE_IMG, 2, &naxes, &mut status);
        /* Move to the image extension */
        fits_movabs_hdu(f, 2, None, &mut status);
        /* Check how many groups this HDU belongs to */
        let mut ngroups: c_long = -1;
        fits_get_num_groups(f, &mut ngroups, &mut status);
        assert_eq!(status, 0);
        assert_eq!(ngroups, 0);

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtcp_copy() {
        let mut status = 0;

        /* Create first file with grouping table */
        let mut fptr1 = new_mem_file(&mut status);
        fits_create_group(
            fptr1.as_deref_mut().unwrap(),
            cs!(c"OrigGroup"),
            GT_ID_ALL_URI as c_int,
            &mut status,
        );

        /* Create second file */
        let mut fptr2 = new_mem_file(&mut status);

        /* Move f1 to grouping table first, then copy it to f2 */
        fits_movabs_hdu(fptr1.as_deref_mut().unwrap(), 2, None, &mut status);
        fits_copy_group(
            fptr1.as_deref_mut().unwrap(),
            fptr2.as_deref_mut().unwrap(),
            0,
            &mut status,
        );
        assert_eq!(status, 0, "ffgtcp failed");

        /* Verify copy exists in f2 */
        let f2 = fptr2.as_deref_mut().unwrap();
        fits_movabs_hdu(f2, 2, None, &mut status);
        let mut hdutype = 0;
        fits_get_hdu_type(f2, &mut hdutype, &mut status);
        assert_eq!(hdutype, BINARY_TBL);
        assert_eq!(read_str_key(f2, cs!(c"EXTNAME"), &mut status), "GROUPING");

        ffclos_safe(fptr1.take().unwrap(), &mut status);
        ffclos_safe(fptr2.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtrm_remove() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        fits_create_group(f, cs!(c"ToRemove"), GT_ID_ALL_URI as c_int, &mut status);

        /* Get HDU count before removal */
        let mut numhdus = 0;
        fits_get_num_hdus(f, &mut numhdus, &mut status);
        assert_eq!(numhdus, 2);

        /* Remove the grouping table (OPT_RM_GPT = 0 removes just table) */
        fits_remove_group(f, OPT_RM_GPT as c_int, &mut status);
        assert_eq!(status, 0, "ffgtrm failed");

        /* Check HDU count decreased */
        fits_get_num_hdus(f, &mut numhdus, &mut status);
        assert_eq!(numhdus, 1);

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtcm_compact() {
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        fits_create_group(f, cs!(c"CompactMe"), GT_ID_ALL_URI as c_int, &mut status);
        /* Compact with OPT_CMT_MBR (1) - remove invalid member entries */
        fits_compact_group(f, OPT_CMT_MBR as c_int, &mut status);
        assert_eq!(status, 0, "ffgtcm failed");

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtcr_null_status() {
        /* This tests passing pre-existing error status */
        let mut status = 0;
        let mut fptr = new_mem_file(&mut status);
        let f = fptr.as_deref_mut().unwrap();

        /* Pass a bad status to see if it returns immediately */
        status = 1;
        fits_create_group(f, cs!(c"BadStatus"), GT_ID_ALL_URI as c_int, &mut status);
        assert_eq!(status, 1); /* Should return with same error */
        status = 0;

        ffclos_safe(fptr.take().unwrap(), &mut status);
    }

    #[test]
    fn test_ffgtam_add_member() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut fptr: Option<Box<fitsfile>> = None;
            ffinit_safe(&mut fptr, &name, &mut status);
            assert_eq!(status, 0, "ffinit failed");
            let f = fptr.as_deref_mut().unwrap();
            fits_write_imghdr(f, BYTE_IMG, 0, &[], &mut status);

            /* Create an image extension to be a member */
            let naxes: [c_long; 2] = [10, 10];
            fits_create_img(f, BYTE_IMG, 2, &naxes, &mut status);

            /* Create a grouping table */
            fits_create_group(f, cs!(c"TestGroup"), GT_ID_ALL_URI as c_int, &mut status);

            /* Position the main handle at the member HDU */
            fits_movabs_hdu(f, 2, None, &mut status);
            assert_eq!(status, 0, "setup failed");

            /* Open the grouping table as a second handle and add the member */
            let mut gfptr: Option<Box<fitsfile>> = None;
            let gname = path_with_ext(filename, "[GROUPING]");
            ffopen_safe(&mut gfptr, &gname, READWRITE, &mut status);
            assert_eq!(status, 0, "ffopen [GROUPING] failed");

            let g = gfptr.as_deref_mut().unwrap();
            fits_add_group_member(g, Some(f), 0, &mut status);
            assert_eq!(status, 0, "ffgtam failed");

            let mut nmembers: c_long = -1;
            fits_get_num_members(g, &mut nmembers, &mut status);
            assert_eq!(nmembers, 1);

            ffclos_safe(gfptr.take().unwrap(), &mut status);
            ffclos_safe(fptr.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgtmg_merge() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            let mut fptr: Option<Box<fitsfile>> = None;
            ffinit_safe(&mut fptr, &name, &mut status);
            assert_eq!(status, 0, "ffinit failed");
            let f = fptr.as_deref_mut().unwrap();
            fits_write_imghdr(f, BYTE_IMG, 0, &[], &mut status);

            /* Create two grouping tables */
            fits_create_group(f, cs!(c"Group1"), GT_ID_ALL_URI as c_int, &mut status);
            fits_movabs_hdu(f, 1, None, &mut status);
            fits_create_group(f, cs!(c"Group2"), GT_ID_ALL_URI as c_int, &mut status);

            let mut numhdus = 0;
            fits_get_num_hdus(f, &mut numhdus, &mut status);
            assert_eq!(numhdus, 3);

            /* Move to second grouping table */
            fits_movabs_hdu(f, 3, None, &mut status);

            /* Open first grouping table as a second handle and merge into it */
            let mut gfptr: Option<Box<fitsfile>> = None;
            let gname = path_with_ext(filename, "[GROUPING,1]");
            ffopen_safe(&mut gfptr, &gname, READWRITE, &mut status);
            assert_eq!(status, 0, "ffopen [GROUPING,1] failed");

            /* Merge second into first (OPT_MRG_COPY = 0) */
            fits_merge_groups(
                gfptr.as_deref_mut().unwrap(),
                f,
                OPT_MRG_COPY as c_int,
                &mut status,
            );
            assert_eq!(status, 0, "ffgtmg failed");

            ffclos_safe(gfptr.take().unwrap(), &mut status);
            ffclos_safe(fptr.take().unwrap(), &mut status);
        });
    }

    /// A member whose GRPLCn keyword is longer than a filename must be rejected
    /// with URL_PARSE_ERROR.  It used to be copied into a FLEN_FILENAME buffer
    /// unchecked, which overran the buffer in C (CFITSIO 4.7.0 added this check)
    /// and panicked here.
    #[test]
    fn test_ffgtam_overlong_grplc() {
        use crate::putkey::{ffpkls_safe, ffpkyj_safe};

        with_temp_file(|filename| {
            let mut status = 0;
            let gname = to_buf(filename);
            let mname = to_buf(&format!("{filename}.member"));

            /* The grouping table lives in its own file, so that the member's
            GRPIDn is negative and its location is taken from GRPLCn */
            let mut gfptr: Option<Box<fitsfile>> = None;
            ffinit_safe(&mut gfptr, &gname, &mut status);
            {
                let g = gfptr.as_deref_mut().unwrap();
                fits_write_imghdr(g, BYTE_IMG, 0, &[], &mut status);
                fits_create_group(g, cs!(c"TestGroup"), GT_ID_ALL_URI as c_int, &mut status);
            }
            ffclos_safe(gfptr.take().unwrap(), &mut status);
            assert_eq!(status, 0, "creating the grouping table failed");

            /* The member, carrying a GRPLC1 longer than any filename */
            let overlong: Vec<c_char> = core::iter::repeat_n(b'x' as c_char, FLEN_FILENAME)
                .chain(core::iter::once(0))
                .collect();
            let mut mfptr: Option<Box<fitsfile>> = None;
            ffinit_safe(&mut mfptr, &mname, &mut status);
            {
                let m = mfptr.as_deref_mut().unwrap();
                fits_write_imghdr(m, BYTE_IMG, 0, &[], &mut status);
                let naxes: [c_long; 2] = [10, 10];
                fits_create_img(m, BYTE_IMG, 2, &naxes, &mut status);
                ffpkyj_safe(m, cs!(c"GRPID1"), -1, Some(cs!(c"")), &mut status);
                ffpkls_safe(m, cs!(c"GRPLC1"), &overlong, Some(cs!(c"")), &mut status);
                assert_eq!(status, 0, "creating the member failed");
            }
            ffclos_safe(mfptr.take().unwrap(), &mut status);
            assert_eq!(status, 0, "closing the member failed");

            let mut gfptr: Option<Box<fitsfile>> = None;
            let gtable = path_with_ext(filename, "[GROUPING]");
            ffopen_safe(&mut gfptr, &gtable, READWRITE, &mut status);
            let mut mfptr: Option<Box<fitsfile>> = None;
            ffopen_safe(&mut mfptr, &mname, READWRITE, &mut status);
            assert_eq!(status, 0, "reopening failed");
            {
                let m = mfptr.as_deref_mut().unwrap();
                fits_movabs_hdu(m, 2, None, &mut status);
                let g = gfptr.as_deref_mut().unwrap();
                assert_eq!(status, 0, "moving to the member HDU failed");

                fits_add_group_member(g, Some(m), 0, &mut status);
                assert_eq!(
                    status, URL_PARSE_ERROR,
                    "ffgtam accepted an over-long GRPLCn"
                );
            }

            status = 0;
            ffclos_safe(gfptr.take().unwrap(), &mut status);
            ffclos_safe(mfptr.take().unwrap(), &mut status);
        });
    }

    /// The same over-long GRPLCn, seen from ffgmrm while it searches the member
    /// for the group to unlink.  CFITSIO checks this in ffgtam only, but the
    /// copy here is just as unchecked.
    #[test]
    fn test_ffgmrm_overlong_grplc() {
        use crate::modkey::ffukls_safe;

        with_temp_file(|filename| {
            let mut status = 0;
            let gname = to_buf(filename);
            let mname = to_buf(&format!("{filename}.member"));

            /* The grouping table and the member live in separate files, so the
            member's GRPIDn is negative and GRPLCn holds the table's location */
            let mut gfptr: Option<Box<fitsfile>> = None;
            ffinit_safe(&mut gfptr, &gname, &mut status);
            {
                let g = gfptr.as_deref_mut().unwrap();
                fits_write_imghdr(g, BYTE_IMG, 0, &[], &mut status);
                fits_create_group(g, cs!(c"TestGroup"), GT_ID_ALL_URI as c_int, &mut status);
            }
            ffclos_safe(gfptr.take().unwrap(), &mut status);

            let mut mfptr: Option<Box<fitsfile>> = None;
            ffinit_safe(&mut mfptr, &mname, &mut status);
            {
                let m = mfptr.as_deref_mut().unwrap();
                fits_write_imghdr(m, BYTE_IMG, 0, &[], &mut status);
                let naxes: [c_long; 2] = [10, 10];
                fits_create_img(m, BYTE_IMG, 2, &naxes, &mut status);
            }
            ffclos_safe(mfptr.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup failed");

            /* Register the member properly, so the table has a row for it */
            let gtable = path_with_ext(filename, "[GROUPING]");
            let mut gfptr: Option<Box<fitsfile>> = None;
            ffopen_safe(&mut gfptr, &gtable, READWRITE, &mut status);
            let mut mfptr: Option<Box<fitsfile>> = None;
            ffopen_safe(&mut mfptr, &mname, READWRITE, &mut status);
            {
                let m = mfptr.as_deref_mut().unwrap();
                fits_movabs_hdu(m, 2, None, &mut status);
                let g = gfptr.as_deref_mut().unwrap();
                fits_add_group_member(g, Some(m), 0, &mut status);
                assert_eq!(status, 0, "ffgtam failed");

                /* Now corrupt the back-link the member holds */
                let overlong: Vec<c_char> = core::iter::repeat_n(b'x' as c_char, FLEN_FILENAME)
                    .chain(core::iter::once(0))
                    .collect();
                ffukls_safe(m, cs!(c"GRPLC1"), &overlong, Some(cs!(c"")), &mut status);
                assert_eq!(status, 0, "rewriting GRPLC1 failed");

                fits_remove_member(g, 1, OPT_RM_ENTRY as c_int, &mut status);
                assert_eq!(
                    status, URL_PARSE_ERROR,
                    "ffgmrm accepted an over-long GRPLCn"
                );
            }

            status = 0;
            ffclos_safe(gfptr.take().unwrap(), &mut status);
            ffclos_safe(mfptr.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_group_workflow() {
        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [5, 5];

            /* Create file with primary + 2 image extensions */
            let mut fptr: Option<Box<fitsfile>> = None;
            ffinit_safe(&mut fptr, &name, &mut status);
            assert_eq!(status, 0, "ffinit failed");
            {
                let f = fptr.as_deref_mut().unwrap();
                fits_write_imghdr(f, BYTE_IMG, 0, &[], &mut status);
                fits_create_img(f, BYTE_IMG, 2, &naxes, &mut status);
                fits_write_key_str(
                    f,
                    cs!(c"EXTNAME"),
                    cs!(c"IMAGE1"),
                    Some(cs!(c"First image")),
                    &mut status,
                );
                fits_create_img(f, BYTE_IMG, 2, &naxes, &mut status);
                fits_write_key_str(
                    f,
                    cs!(c"EXTNAME"),
                    cs!(c"IMAGE2"),
                    Some(cs!(c"Second image")),
                    &mut status,
                );

                /* Create grouping table */
                fits_create_group(f, cs!(c"ImageGroup"), GT_ID_ALL_URI as c_int, &mut status);

                /* Verify we're at the grouping table */
                assert_eq!(read_str_key(f, cs!(c"EXTNAME"), &mut status), "GROUPING");
                assert_eq!(status, 0, "setup failed");
            }

            /* Close and reopen the grouping table */
            ffclos_safe(fptr.take().unwrap(), &mut status);

            let mut gfptr: Option<Box<fitsfile>> = None;
            let gname = path_with_ext(filename, "[GROUPING]");
            ffopen_safe(&mut gfptr, &gname, READWRITE, &mut status);
            assert_eq!(status, 0, "ffopen [GROUPING] failed");

            /* Get number of members (should be 0) */
            let mut nmembers: c_long = -1;
            fits_get_num_members(gfptr.as_deref_mut().unwrap(), &mut nmembers, &mut status);
            assert_eq!(nmembers, 0);

            /* Add first image as member */
            {
                let mut mf: Option<Box<fitsfile>> = None;
                let mname = path_with_ext(filename, "[IMAGE1]");
                ffopen_safe(&mut mf, &mname, READWRITE, &mut status);
                assert_eq!(status, 0, "ffopen [IMAGE1] failed");
                fits_add_group_member(
                    gfptr.as_deref_mut().unwrap(),
                    Some(mf.as_deref_mut().unwrap()),
                    0,
                    &mut status,
                );
                ffclos_safe(mf.take().unwrap(), &mut status);
            }

            /* Verify member count */
            fits_get_num_members(gfptr.as_deref_mut().unwrap(), &mut nmembers, &mut status);
            assert_eq!(nmembers, 1);

            /* Add second image as member */
            {
                let mut mf: Option<Box<fitsfile>> = None;
                let mname = path_with_ext(filename, "[IMAGE2]");
                ffopen_safe(&mut mf, &mname, READWRITE, &mut status);
                assert_eq!(status, 0, "ffopen [IMAGE2] failed");
                fits_add_group_member(
                    gfptr.as_deref_mut().unwrap(),
                    Some(mf.as_deref_mut().unwrap()),
                    0,
                    &mut status,
                );
                ffclos_safe(mf.take().unwrap(), &mut status);
            }

            /* Verify member count */
            fits_get_num_members(gfptr.as_deref_mut().unwrap(), &mut nmembers, &mut status);
            assert_eq!(nmembers, 2);

            /* Verify the group */
            let mut firstfailed: c_long = -1;
            fits_verify_group(gfptr.as_deref_mut().unwrap(), &mut firstfailed, &mut status);
            assert_eq!(firstfailed, 0);

            /* Open member 1 and check its EXTNAME */
            {
                let mut mf: Option<Box<fitsfile>> = None;
                fits_open_member(gfptr.as_deref_mut().unwrap(), 1, &mut mf, &mut status);
                assert_eq!(status, 0, "ffgmop member 1 failed");
                assert_eq!(
                    read_str_key(mf.as_deref_mut().unwrap(), cs!(c"EXTNAME"), &mut status),
                    "IMAGE1"
                );
                ffclos_safe(mf.take().unwrap(), &mut status);
            }

            /* Open member 2 and check its EXTNAME */
            {
                let mut mf: Option<Box<fitsfile>> = None;
                fits_open_member(gfptr.as_deref_mut().unwrap(), 2, &mut mf, &mut status);
                assert_eq!(status, 0, "ffgmop member 2 failed");
                assert_eq!(
                    read_str_key(mf.as_deref_mut().unwrap(), cs!(c"EXTNAME"), &mut status),
                    "IMAGE2"
                );
                ffclos_safe(mf.take().unwrap(), &mut status);
            }

            ffclos_safe(gfptr.take().unwrap(), &mut status);
        });
    }
}
