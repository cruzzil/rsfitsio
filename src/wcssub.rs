//! Reading and subsetting World Coordinate System keywords.
//!
//! Extracts the WCS keywords describing a coordinate axis from a header, and
//! copies them across when an image is subsetted, so that the new image's
//! coordinates still describe the same sky.
//!
//! `fits_read_wcstab` here is the interface to Mark Calabretta's WCSLIB, via
//! the [`wtbarr`] struct.
//!
//! See the "World Coordinate System Routines" chapter of the CFITSIO User's
//! Reference Guide.
#![warn(missing_docs)]

use core::ptr;
use core::slice;

use crate::c_types::*;
use crate::helpers::vec_raw_parts::vec_into_raw_parts;

use bytemuck::cast_slice;

use crate::cfileio::ffdelt_safe;
use crate::cfileio::ffinit_safe;

use crate::fitscore::{
    ALLOCATIONS, ffgcno_safe, ffghdn_safe, ffghdt_safe, ffgncl_safe, ffkeyn_safe, ffmahd_safe,
    ffmkky_safe, ffmnhd_safe, ffpmsg_str, fits_copy_pixlist2image_safe,
};
use crate::fitsio::*;
use crate::getcold::ffgcvd_safe;
use crate::getkey::{ffgkey_safe, ffgkyd_safe, ffgkyj_safe, ffgkys_safe, ffgtdm_safe, ffh2st_safe};
use crate::histo::fits_write_keys_histo_safe;
use crate::putkey::{ffcrim_safe, ffi2c};
use crate::wrappers::*;
use crate::{bb, cs};

///  Author: Mark Calabretta, Australia Telescope National Facility
///  <http://www.atnf.csiro.au/~mcalabre/index.html>
///
///  fits_read_wcstab() extracts arrays from a binary table required in
///  constructing -TAB coordinates.  This helper routine is intended for
///  use by routines in the WCSLIB library when dealing with the -TAB table
///  look up WCS convention.
///
/// # Parameters
///
/// * `fptr` — (I) FITS file pointer
/// * `nwtb` — Number of arrays to be read from the binary table(s)
/// * `wtb`  — Address of the first element of an array of wtbarr typedefs. This wtbarr
///           typedef is defined below to match the wtbarr struct defined in WCSLIB. An
///           array of such structs returned by the WCSLIB function wcstab().
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_read_wcstab(
    fptr: *mut fitsfile,
    nwtb: c_int,
    wtb: *const wtbarr,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);

        if *status != 0 {
            return *status;
        }

        if fptr.is_null() {
            *status = NULL_INPUT_PTR;
            return *status;
        }

        if nwtb == 0 {
            return 0;
        }

        // Only construct the slice after validating the pointer and length;
        // slice::from_raw_parts requires a non-null, aligned pointer even for
        // a zero-length slice.
        let wtb = slice::from_raw_parts(wtb, nwtb as usize);

        let fptr = fptr.as_mut().expect(NULL_MSG);

        fits_read_wcstab_safe(fptr, nwtb, wtb, status)
    }
}

///  Author: Mark Calabretta, Australia Telescope National Facility
///  <http://www.atnf.csiro.au/~mcalabre/index.html>
///
///  fits_read_wcstab() extracts arrays from a binary table required in
///  constructing -TAB coordinates.  This helper routine is intended for
///  use by routines in the WCSLIB library when dealing with the -TAB table
///  look up WCS convention.
///
/// # Parameters
///
/// * `fptr` — (I) FITS file pointer
/// * `nwtb` — Number of arrays to be read from the binary table(s)
pub fn fits_read_wcstab_safe(
    fptr: &mut fitsfile,
    nwtb: c_int,
    wtb: &[wtbarr],
    status: &mut c_int,
) -> c_int {
    let mut anynul: c_int = 0;
    let mut colnum: c_int = 0;
    let mut hdunum: c_int = 0;
    let mut naxis: c_int = 0;
    let mut nostat: c_int = 0;
    let mut nelem: c_long = 0;

    let mut naxes: Vec<c_long> = Vec::new();

    if nwtb == 0 {
        return 0;
    }

    /* Zero the array pointers. */
    let mut wtbp = wtb;
    for iwtb in 0..(nwtb as usize) {
        // SAFETY: `arrayp` is a valid output pointer supplied by the caller; the
        // `wtbarr` structs originate from WCSLIB and are validated by the FFI wrapper.
        unsafe {
            *wtbp[iwtb].arrayp = ptr::null_mut();
        }
    }

    /* Save HDU number so that we can move back to it later. */
    ffghdn_safe(fptr, &mut hdunum);

    wtbp = wtb;
    for iwtb in 0..(nwtb as usize) {
        /* Move to the required binary table extension. */
        if ffmnhd_safe(
            fptr,
            BINARY_TBL,
            &(wtbp[iwtb].extnam),
            wtbp[iwtb].extver,
            status,
        ) != 0
        {
            // goto cleanup;
            break;
        }

        /* Locate the table column. */
        if ffgcno_safe(
            fptr,
            CASEINSEN as c_int,
            &(wtbp[iwtb].ttype),
            &mut colnum,
            status,
        ) != 0
        {
            // goto cleanup;
            break;
        }

        /* Get the array dimensions and check for consistency. */
        if wtbp[iwtb].ndim < 1 {
            *status = NEG_AXIS;
            // goto cleanup;
            break;
        }

        if naxes.try_reserve_exact(wtbp[iwtb].ndim as usize).is_err() {
            *status = MEMORY_ALLOCATION;
            // goto cleanup;
            break;
        } else {
            naxes.resize(wtbp[iwtb].ndim as usize, 0);
        }

        if ffgtdm_safe(
            fptr,
            colnum,
            wtbp[iwtb].ndim,
            &mut naxis,
            &mut naxes,
            status,
        ) != 0
        {
            // goto cleanup;
            break;
        }

        if naxis != wtbp[iwtb].ndim {
            if wtbp[iwtb].kind == c_int::from(b'c') && wtbp[iwtb].ndim == 2 {
                /* Allow TDIMn to be omitted for degenerate coordinate arrays. */
                naxis = 2;
                naxes[1] = naxes[0];
                naxes[0] = 1;
            } else {
                *status = BAD_TDIM;
                // goto cleanup;
                break;
            }
        }

        if wtbp[iwtb].kind == c_int::from(b'c') {
            /* Coordinate array; calculate the array size. */
            nelem = naxes[0];

            for m in 0..(naxis as usize - 1) {
                // SAFETY: `dimlen` points to at least (naxis-1) c_ints supplied by the caller.
                let dimlen =
                    unsafe { slice::from_raw_parts_mut(wtbp[iwtb].dimlen, naxis as usize - 1) };
                dimlen[m] = naxes[m + 1] as c_int;
                nelem *= naxes[m + 1];
            }
        } else {
            /* Index vector; check length. */
            nelem = naxes[0];
            // SAFETY: `dimlen` points to a valid c_int supplied by the caller.
            let dimlen0 = unsafe { *(wtbp[iwtb].dimlen) };
            if nelem != c_long::from(dimlen0) {
                /* N.B. coordinate array precedes the index vectors. */
                *status = BAD_TDIM;
                // goto cleanup;
                break;
            }
        }

        // HEAP ALLOCATION
        /* Allocate memory for the array. */
        let mut tmp: Vec<f64> = Vec::new();

        if tmp.try_reserve_exact(nelem as usize).is_err() {
            *status = MEMORY_ALLOCATION;
            // goto cleanup;
            break;
        } else {
            tmp.resize(nelem as usize, 0.0);
        }

        if ffgcvd_safe(
            fptr,
            colnum,
            wtbp[iwtb].row as LONGLONG,
            1,
            nelem as LONGLONG,
            0.0,
            &mut tmp,
            Some(&mut anynul),
            status,
        ) != 0
        {
            // goto cleanup;
            break;
        }

        // HEAP ALLOCATION
        let (ptr, l, c) = vec_into_raw_parts(tmp);
        crate::lock_global(&ALLOCATIONS).insert(ptr as usize, (l, c));

        // SAFETY: `arrayp` is a valid output pointer supplied by the caller.
        unsafe {
            *wtbp[iwtb].arrayp = ptr;
        }
    }

    /* Move back to the starting HDU. */
    nostat = 0;
    ffmahd_safe(fptr, hdunum, None, &mut nostat);

    /* Release allocated memory. */
    if *status != 0 {
        wtbp = wtb;
        for iwtb in 0..(nwtb as usize) {
            // SAFETY: `arrayp` was either nulled or set to a `vec_into_raw_parts`
            // allocation above; reconstructing the Vec frees it.
            unsafe {
                if !(*wtbp[iwtb].arrayp).is_null() {
                    // WARNING: Assuming that the ndim hasn't been changed
                    let _ = Vec::from_raw_parts(
                        *wtbp[iwtb].arrayp,
                        wtbp[iwtb].ndim as usize,
                        wtbp[iwtb].ndim as usize,
                    );
                }
            }
        }
    }

    *status
}

/// Int fits_get_image_wcs_keys
/// return a string containing all the image WCS header keywords.
/// This string is then used as input to the wcsinit WCSlib routine.
///
/// THIS ROUTINE IS DEPRECATED. USE fits_hdr2str INSTEAD
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `header` — (O) pointer to the WCS related keywords
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgiwcs(
    fptr: *mut fitsfile,
    header: *mut *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let header = header.as_mut().expect(NULL_MSG);

        if *status > 0 {
            return *status;
        }

        ffgiwcs_safe(fptr, header, status)
    }
}

/// Int fits_get_image_wcs_keys
/// return a string containing all the image WCS header keywords.
/// This string is then used as input to the wcsinit WCSlib routine.
///
/// THIS ROUTINE IS DEPRECATED. USE fits_hdr2str INSTEAD
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `header` — (O) pointer to the WCS related keywords
/// * `status` — (IO) error status
pub fn ffgiwcs_safe(fptr: &mut fitsfile, header: &mut *mut c_char, status: &mut c_int) -> c_int {
    let mut hdutype = 0;

    if *status > 0 {
        return *status;
    }

    ffghdt_safe(fptr, &mut hdutype, status);
    if hdutype != IMAGE_HDU {
        ffpmsg_str("Error in ffgiwcs. This HDU is not an image. Can't read WCS keywords");
        *status = NOT_IMAGE;
        return *status;
    }

    /* read header keywords into a long string of chars */
    if ffh2st_safe(fptr, header, status) > 0 {
        ffpmsg_str("error creating string of image WCS keywords (ffgiwcs)");
        return *status;
    }

    *status
}

/// Read the values of the celestial coordinate system keywords.
/// These values may be used as input to the subroutines that
/// calculate celestial coordinates. (ffxypx, ffwldp)
///
/// Modified in Nov 1999 to convert the CD matrix keywords back
/// to the old CDELTn form, and to swap the axes if the dec-like
/// axis is given first, and to assume default values if any of the
/// keywords are not present.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `xrval`  — (O) X reference value
/// * `yrval`  — (O) Y reference value
/// * `xrpix`  — (O) X reference pixel
/// * `yrpix`  — (O) Y reference pixel
/// * `xinc`   — (O) X increment per pixel
/// * `yinc`   — (O) Y increment per pixel
/// * `rot`    — (O) rotation angle (degrees)
/// * `ptype`  — (O) type of projection ('-tan')
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgics(
    fptr: *mut fitsfile,
    xrval: *mut f64,
    yrval: *mut f64,
    xrpix: *mut f64,
    yrpix: *mut f64,
    xinc: *mut f64,
    yinc: *mut f64,
    rot: *mut f64,
    ptype: *mut [c_char; 5],
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let xrval = xrval.as_mut().expect(NULL_MSG);
        let yrval = yrval.as_mut().expect(NULL_MSG);
        let xrpix = xrpix.as_mut().expect(NULL_MSG);
        let yrpix = yrpix.as_mut().expect(NULL_MSG);
        let xinc = xinc.as_mut().expect(NULL_MSG);
        let yinc = yinc.as_mut().expect(NULL_MSG);
        let rot = rot.as_mut().expect(NULL_MSG);
        let ptype = ptype.as_mut().expect(NULL_MSG);

        ffgics_safe(
            fptr, xrval, yrval, xrpix, yrpix, xinc, yinc, rot, ptype, status,
        )
    }
}

/// Read the values of the celestial coordinate system keywords.
/// These values may be used as input to the subroutines that
/// calculate celestial coordinates. (ffxypx, ffwldp)
///
/// Modified in Nov 1999 to convert the CD matrix keywords back
/// to the old CDELTn form, and to swap the axes if the dec-like
/// axis is given first, and to assume default values if any of the
/// keywords are not present.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `xrval`  — (O) X reference value
/// * `yrval`  — (O) Y reference value
/// * `xrpix`  — (O) X reference pixel
/// * `yrpix`  — (O) Y reference pixel
/// * `xinc`   — (O) X increment per pixel
/// * `yinc`   — (O) Y increment per pixel
/// * `rot`    — (O) rotation angle (degrees)
/// * `ptype`  — (O) type of projection ('-tan')
/// * `status` — (IO) error status
pub fn ffgics_safe(
    fptr: &mut fitsfile,
    xrval: &mut f64,
    yrval: &mut f64,
    xrpix: &mut f64,
    yrpix: &mut f64,
    xinc: &mut f64,
    yinc: &mut f64,
    rot: &mut f64,
    ptype: &mut [c_char; 5],
    status: &mut c_int,
) -> c_int {
    let mut tstat: c_int = 0;
    let mut cd_exists: bool = false;
    let mut pc_exists: bool = false;
    let mut ctype: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut cd11: f64 = 0.0;
    let mut cd21: f64 = 0.0;
    let mut cd22: f64 = 0.0;
    let mut cd12: f64 = 0.0;
    let mut pc11: f64 = 1.0;
    let mut pc21: f64 = 0.0;
    let mut pc22: f64 = 1.0;
    let mut pc12: f64 = 0.0;

    #[allow(clippy::approx_constant)]
    let pi: f64 = 3.141_592_653_589_793;
    let mut phia: f64 = 0.0;
    let mut phib: f64 = 0.0;
    let mut temp: f64 = 0.0;
    let toler: f64 = 0.0002; /* tolerance for angles to agree (radians) (= approximately 0.01 degrees)  */

    if *status > 0 {
        return *status;
    }

    tstat = 0;
    if ffgkyd_safe(fptr, cs!(c"CRVAL1"), xrval, None, &mut tstat) != 0 {
        *xrval = 0.0;
    }

    tstat = 0;
    if ffgkyd_safe(fptr, cs!(c"CRVAL2"), yrval, None, &mut tstat) != 0 {
        *yrval = 0.0;
    }

    tstat = 0;
    if ffgkyd_safe(fptr, cs!(c"CRPIX1"), xrpix, None, &mut tstat) != 0 {
        *xrpix = 0.0;
    }

    tstat = 0;
    if ffgkyd_safe(fptr, cs!(c"CRPIX2"), yrpix, None, &mut tstat) != 0 {
        *yrpix = 0.0;
    }

    /* look for CDELTn first, then CDi_j keywords */
    tstat = 0;
    if ffgkyd_safe(fptr, cs!(c"CDELT1"), xinc, None, &mut tstat) != 0 {
        /* CASE 1: no CDELTn keyword, so look for the CD matrix */
        tstat = 0;
        if ffgkyd_safe(fptr, cs!(c"CD1_1"), &mut cd11, None, &mut tstat) != 0 {
            tstat = 0; /* reset keyword not found error */
        } else {
            cd_exists = true; /* found at least 1 CD_ keyword */
        }

        if ffgkyd_safe(fptr, cs!(c"CD2_1"), &mut cd21, None, &mut tstat) != 0 {
            tstat = 0; /* reset keyword not found error */
        } else {
            cd_exists = true; /* found at least 1 CD_ keyword */
        }

        if ffgkyd_safe(fptr, cs!(c"CD1_2"), &mut cd12, None, &mut tstat) != 0 {
            tstat = 0; /* reset keyword not found error */
        } else {
            cd_exists = true; /* found at least 1 CD_ keyword */
        }

        if ffgkyd_safe(fptr, cs!(c"CD2_2"), &mut cd22, None, &mut tstat) != 0 {
            tstat = 0; /* reset keyword not found error */
        } else {
            cd_exists = true; /* found at least 1 CD_ keyword */
        }

        if cd_exists {
            /* convert CDi_j back to CDELTn */

            /* there are 2 ways to compute the angle: */
            phia = f64::atan2(cd21, cd11);
            phib = f64::atan2(-cd12, cd22);

            /* ensure that phia <= phib */
            temp = f64::min(phia, phib);
            phib = f64::min(phia, phib);
            phia = temp;

            /* there is a possible 180 degree ambiguity in the angles */
            /* so add 180 degress to the smaller value if the values  */
            /* differ by more than 90 degrees = pi/2 radians.         */
            /* (Later, we may decide to take the other solution by    */
            /* subtracting 180 degrees from the larger value).        */

            if (phib - phia) > (pi / 2.) {
                phia += pi;
            }

            if (phia - phib).abs() > toler {
                /* angles don't agree, so looks like there is some skewness */
                /* between the axes.  Return with an error to be safe. */
                *status = APPROX_WCS_KEY;
            }

            phia = (phia + phib) / 2.; /* use the average of the 2 values */
            *xinc = cd11 / (phia).cos();
            *yinc = cd22 / (phia).cos();
            *rot = phia * 180. / pi;

            /* common usage is to have a positive yinc value.  If it is */
            /* negative, then subtract 180 degrees from rot and negate  */
            /* both xinc and yinc.  */

            if *yinc < 0.0 {
                *xinc = -(*xinc);
                *yinc = -(*yinc);
                *rot -= 180.0;
            }
        } else {
            /* no CD matrix keywords either */
            *xinc = 1.;

            /* there was no CDELT1 keyword, but check for CDELT2 just in case */
            tstat = 0;
            if ffgkyd_safe(fptr, cs!(c"CDELT2"), yinc, None, &mut tstat) != 0 {
                *yinc = 1.;
            }

            tstat = 0;
            if ffgkyd_safe(fptr, cs!(c"CROTA2"), rot, None, &mut tstat) != 0 {
                *rot = 0.0;
            }
        }
    } else {
        /* Case 2: CDELTn + optional PC matrix */
        if ffgkyd_safe(fptr, cs!(c"CDELT2"), yinc, None, &mut tstat) != 0 {
            *yinc = 1.;
        }

        tstat = 0;
        if ffgkyd_safe(fptr, cs!(c"CROTA2"), rot, None, &mut tstat) != 0 {
            *rot = 0.0;

            /* no CROTA2 keyword, so look for the PC matrix */
            tstat = 0;
            if ffgkyd_safe(fptr, cs!(c"PC1_1"), &mut pc11, None, &mut tstat) != 0 {
                tstat = 0; /* reset keyword not found error */
            } else {
                pc_exists = true; /* found at least 1 PC_ keyword */
            }

            if ffgkyd_safe(fptr, cs!(c"PC2_1"), &mut pc21, None, &mut tstat) != 0 {
                tstat = 0; /* reset keyword not found error */
            } else {
                pc_exists = true; /* found at least 1 PC_ keyword */
            }

            if ffgkyd_safe(fptr, cs!(c"PC1_2"), &mut pc12, None, &mut tstat) != 0 {
                tstat = 0; /* reset keyword not found error */
            } else {
                pc_exists = true; /* found at least 1 PC_ keyword */
            }

            if ffgkyd_safe(fptr, cs!(c"PC2_2"), &mut pc22, None, &mut tstat) != 0 {
                tstat = 0; /* reset keyword not found error */
            } else {
                pc_exists = true; /* found at least 1 PC_ keyword */
            }

            if pc_exists {
                /* convert PCi_j back to CDELTn */

                /* there are 2 ways to compute the angle: */
                phia = f64::atan2(pc21, pc11);
                phib = f64::atan2(-pc12, pc22);

                /* ensure that phia <= phib */
                temp = f64::min(phia, phib);
                phib = f64::max(phia, phib);
                phia = temp;

                /* there is a possible 180 degree ambiguity in the angles */
                /* so add 180 degress to the smaller value if the values  */
                /* differ by more than 90 degrees = pi/2 radians.         */
                /* (Later, we may decide to take the other solution by    */
                /* subtracting 180 degrees from the larger value).        */

                if (phib - phia) > (pi / 2.) {
                    phia += pi;
                }

                if (phia - phib).abs() > toler {
                    /* angles don't agree, so looks like there is some skewness */
                    /* between the axes.  Return with an error to be safe. */
                    *status = APPROX_WCS_KEY;
                }

                phia = (phia + phib) / 2.; /* use the average of the 2 values */
                *rot = phia * 180. / pi;
            }
        }
    }

    /* get the type of projection, if any */
    tstat = 0;
    if ffgkys_safe(fptr, cs!(c"CTYPE1"), &mut ctype, None, &mut tstat) != 0 {
        (*ptype)[0] = 0;
    } else {
        /* copy the projection type string */
        strncpy_safe(&mut (*ptype), &ctype[4..], 4);
        (*ptype)[4] = 0;

        /* check if RA and DEC are inverted */
        if strncmp_safe(&ctype, cs!(c"DEC-"), 4) == 0
            || strncmp_safe(&ctype[1..], cs!(c"LAT"), 3) == 0
        {
            /* the latitudinal axis is given first, so swap them */

            /*
             this case was removed on 12/9.  Apparently not correct.

                        if ((*xinc / *yinc) < 0. )
                            *rot = -90. - (*rot);
                        else
            */
            *rot = 90. - (*rot);

            /* Empirical tests with ds9 show the y-axis sign must be negated */
            /* and the xinc and yinc values must NOT be swapped. */
            *yinc = -(*yinc);

            temp = *xrval;
            *xrval = *yrval;
            *yrval = temp;
        }
    }

    *status
}

/// Read the values of the celestial coordinate system keywords.
/// These values may be used as input to the subroutines that
/// calculate celestial coordinates. (ffxypx, ffwldp)
///
/// Modified in Nov 1999 to convert the CD matrix keywords back
/// to the old CDELTn form, and to swap the axes if the dec-like
/// axis is given first, and to assume default values if any of the
/// keywords are not present.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `version` — (I) character code of desired version
/// * `xrval`   — (O) X reference value
/// * `yrval`   — (O) Y reference value
/// * `xrpix`   — (O) X reference pixel
/// * `yrpix`   — (O) Y reference pixel
/// * `xinc`    — (O) X increment per pixel
/// * `yinc`    — (O) Y increment per pixel
/// * `rot`     — (O) rotation angle (degrees)
/// * `ptype`   — (O) type of projection ('-tan')
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgicsa(
    fptr: *mut fitsfile,
    version: c_char,
    /*     A - Z or blank */
    xrval: *mut f64,
    yrval: *mut f64,
    xrpix: *mut f64,
    yrpix: *mut f64,
    xinc: *mut f64,
    yinc: *mut f64,
    rot: *mut f64,
    ptype: *mut [c_char; 5],
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let xrval = xrval.as_mut().expect(NULL_MSG);
        let yrval = yrval.as_mut().expect(NULL_MSG);
        let xrpix = xrpix.as_mut().expect(NULL_MSG);
        let yrpix = yrpix.as_mut().expect(NULL_MSG);
        let xinc = xinc.as_mut().expect(NULL_MSG);
        let yinc = yinc.as_mut().expect(NULL_MSG);
        let rot = rot.as_mut().expect(NULL_MSG);
        let ptype = ptype.as_mut().expect(NULL_MSG);

        ffgicsa_safe(
            fptr, version, xrval, yrval, xrpix, yrpix, xinc, yinc, rot, ptype, status,
        )
    }
}

/// Read the values of the celestial coordinate system keywords.
/// These values may be used as input to the subroutines that
/// calculate celestial coordinates. (ffxypx, ffwldp)
///
/// Modified in Nov 1999 to convert the CD matrix keywords back
/// to the old CDELTn form, and to swap the axes if the dec-like
/// axis is given first, and to assume default values if any of the
/// keywords are not present.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `version` — (I) character code of desired version
/// * `xrval`   — (O) X reference value
/// * `yrval`   — (O) Y reference value
/// * `xrpix`   — (O) X reference pixel
/// * `yrpix`   — (O) Y reference pixel
/// * `xinc`    — (O) X increment per pixel
/// * `yinc`    — (O) Y increment per pixel
/// * `rot`     — (O) rotation angle (degrees)
/// * `ptype`   — (O) type of projection ('-tan')
/// * `status`  — (IO) error status
pub fn ffgicsa_safe(
    fptr: &mut fitsfile,
    version: c_char,
    /*     A - Z or blank */
    xrval: &mut f64,
    yrval: &mut f64,
    xrpix: &mut f64,
    yrpix: &mut f64,
    xinc: &mut f64,
    yinc: &mut f64,
    rot: &mut f64,
    ptype: &mut [c_char; 5],
    status: &mut c_int,
) -> c_int {
    let mut tstat: c_int = 0;
    let mut cd_exists: bool = false;
    let mut pc_exists: bool = false;
    let mut ctype: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut keyname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut alt: [c_char; 2] = [0; 2];
    let mut cd11: f64 = 0.0;
    let mut cd21: f64 = 0.0;
    let mut cd22: f64 = 0.0;
    let mut cd12: f64 = 0.0;
    let mut pc11: f64 = 1.0;
    let mut pc21: f64 = 0.0;
    let mut pc22: f64 = 1.0;
    let mut pc12: f64 = 0.0;

    #[allow(clippy::approx_constant)]
    let pi: f64 = 3.141_592_653_589_793;
    let mut phia: f64 = 0.0;
    let mut phib: f64 = 0.0;
    let mut temp: f64 = 0.0;
    let toler: f64 = 0.0002; /* tolerance for angles to agree (radians) (= approximately 0.01 degrees)  */

    if *status > 0 {
        return *status;
    }

    if version == bb(b' ') {
        ffgics_safe(
            fptr, xrval, yrval, xrpix, yrpix, xinc, yinc, rot, ptype, status,
        );
        return *status;
    }

    if version > bb(b'Z') || version < bb(b'A') {
        ffpmsg_str("ffgicsa: illegal WCS version code (must be A - Z or blank)");
        *status = WCS_ERROR;
        return *status;
    }

    alt[0] = version;
    alt[1] = 0;

    tstat = 0;
    strcpy_safe(&mut keyname, cs!(c"CRVAL1"));
    strcat_safe(&mut keyname, &alt);
    if ffgkyd_safe(fptr, &keyname, xrval, None, &mut tstat) != 0 {
        *xrval = 0.0;
    }

    tstat = 0;
    strcpy_safe(&mut keyname, cs!(c"CRVAL2"));
    strcat_safe(&mut keyname, &alt);
    if ffgkyd_safe(fptr, &keyname, yrval, None, &mut tstat) != 0 {
        *yrval = 0.0;
    }

    tstat = 0;
    strcpy_safe(&mut keyname, cs!(c"CRPIX1"));
    strcat_safe(&mut keyname, &alt);
    if ffgkyd_safe(fptr, &keyname, xrpix, None, &mut tstat) != 0 {
        *xrpix = 0.0;
    }

    tstat = 0;
    strcpy_safe(&mut keyname, cs!(c"CRPIX2"));
    strcat_safe(&mut keyname, &alt);
    if ffgkyd_safe(fptr, &keyname, yrpix, None, &mut tstat) != 0 {
        *yrpix = 0.0;
    }

    /* look for CDELTn first, then CDi_j keywords */
    tstat = 0;
    strcpy_safe(&mut keyname, cs!(c"CDELT1"));
    strcat_safe(&mut keyname, &alt);
    if ffgkyd_safe(fptr, &keyname, xinc, None, &mut tstat) != 0 {
        /* CASE 1: no CDELTn keyword, so look for the CD matrix */
        tstat = 0;
        strcpy_safe(&mut keyname, cs!(c"CD1_1"));
        strcat_safe(&mut keyname, &alt);
        if ffgkyd_safe(fptr, &keyname, &mut cd11, None, &mut tstat) != 0 {
            tstat = 0; /* reset keyword not found error */
        } else {
            cd_exists = true; /* found at least 1 CD_ keyword */
        }

        strcpy_safe(&mut keyname, cs!(c"CD2_1"));
        strcat_safe(&mut keyname, &alt);
        if ffgkyd_safe(fptr, &keyname, &mut cd21, None, &mut tstat) != 0 {
            tstat = 0; /* reset keyword not found error */
        } else {
            cd_exists = true; /* found at least 1 CD_ keyword */
        }

        strcpy_safe(&mut keyname, cs!(c"CD1_2"));
        strcat_safe(&mut keyname, &alt);
        if ffgkyd_safe(fptr, &keyname, &mut cd12, None, &mut tstat) != 0 {
            tstat = 0; /* reset keyword not found error */
        } else {
            cd_exists = true; /* found at least 1 CD_ keyword */
        }

        strcpy_safe(&mut keyname, cs!(c"CD2_2"));
        strcat_safe(&mut keyname, &alt);
        if ffgkyd_safe(fptr, &keyname, &mut cd22, None, &mut tstat) != 0 {
            tstat = 0; /* reset keyword not found error */
        } else {
            cd_exists = true; /* found at least 1 CD_ keyword */
        }

        if cd_exists {
            /* convert CDi_j back to CDELTn */

            /* there are 2 ways to compute the angle: */
            phia = f64::atan2(cd21, cd11);
            phib = f64::atan2(-cd12, cd22);

            /* ensure that phia <= phib */
            temp = f64::min(phia, phib);
            phib = f64::max(phia, phib);
            phia = temp;

            /* there is a possible 180 degree ambiguity in the angles */
            /* so add 180 degress to the smaller value if the values  */
            /* differ by more than 90 degrees = pi/2 radians.         */
            /* (Later, we may decide to take the other solution by    */
            /* subtracting 180 degrees from the larger value).        */

            if (phib - phia) > (pi / 2.) {
                phia += pi;
            }

            if (phia - phib).abs() > toler {
                /* angles don't agree, so looks like there is some skewness */
                /* between the axes.  Return with an error to be safe. */
                *status = APPROX_WCS_KEY;
            }

            phia = (phia + phib) / 2.; /* use the average of the 2 values */
            *xinc = cd11 / f64::cos(phia);
            *yinc = cd22 / f64::cos(phia);
            *rot = phia * 180. / pi;

            /* common usage is to have a positive yinc value.  If it is */
            /* negative, then subtract 180 degrees from rot and negate  */
            /* both xinc and yinc.  */

            if *yinc < 0.0 {
                *xinc = -(*xinc);
                *yinc = -(*yinc);
                *rot -= 180.0;
            }
        } else {
            /* no CD matrix keywords either */

            *xinc = 1.;

            /* there was no CDELT1 keyword, but check for CDELT2 just in case */
            tstat = 0;
            strcpy_safe(&mut keyname, cs!(c"CDELT2"));
            strcat_safe(&mut keyname, &alt);
            if ffgkyd_safe(fptr, &keyname, yinc, None, &mut tstat) != 0 {
                *yinc = 1.;
            }

            tstat = 0;
            strcpy_safe(&mut keyname, cs!(c"CROTA2"));
            strcat_safe(&mut keyname, &alt);
            if ffgkyd_safe(fptr, &keyname, rot, None, &mut tstat) != 0 {
                *rot = 0.0;
            }
        }
    } else {
        /* Case 2: CDELTn + optional PC matrix */

        strcpy_safe(&mut keyname, cs!(c"CDELT2"));
        strcat_safe(&mut keyname, &alt);
        if ffgkyd_safe(fptr, &keyname, yinc, None, &mut tstat) != 0 {
            *yinc = 1.;
        }

        tstat = 0;
        strcpy_safe(&mut keyname, cs!(c"CROTA2"));
        strcat_safe(&mut keyname, &alt);
        if ffgkyd_safe(fptr, &keyname, rot, None, &mut tstat) != 0 {
            *rot = 0.0;

            /* no CROTA2 keyword, so look for the PC matrix */
            tstat = 0;
            strcpy_safe(&mut keyname, cs!(c"PC1_1"));
            strcat_safe(&mut keyname, &alt);
            if ffgkyd_safe(fptr, &keyname, &mut pc11, None, &mut tstat) != 0 {
                tstat = 0; /* reset keyword not found error */
            } else {
                pc_exists = true; /* found at least 1 PC_ keyword */
            }

            strcpy_safe(&mut keyname, cs!(c"PC2_1"));
            strcat_safe(&mut keyname, &alt);
            if ffgkyd_safe(fptr, &keyname, &mut pc21, None, &mut tstat) != 0 {
                tstat = 0; /* reset keyword not found error */
            } else {
                pc_exists = true; /* found at least 1 PC_ keyword */
            }

            strcpy_safe(&mut keyname, cs!(c"PC1_2"));
            strcat_safe(&mut keyname, &alt);
            if ffgkyd_safe(fptr, &keyname, &mut pc12, None, &mut tstat) != 0 {
                tstat = 0; /* reset keyword not found error */
            } else {
                pc_exists = true; /* found at least 1 PC_ keyword */
            }

            strcpy_safe(&mut keyname, cs!(c"PC2_2"));
            strcat_safe(&mut keyname, &alt);
            if ffgkyd_safe(fptr, &keyname, &mut pc22, None, &mut tstat) != 0 {
                tstat = 0; /* reset keyword not found error */
            } else {
                pc_exists = true; /* found at least 1 PC_ keyword */
            }

            if pc_exists
            /* convert PCi_j back to CDELTn */
            {
                /* there are 2 ways to compute the angle: */
                phia = f64::atan2(pc21, pc11);
                phib = f64::atan2(-pc12, pc22);

                /* ensure that phia <= phib */
                temp = f64::min(phia, phib);
                phib = f64::max(phia, phib);
                phia = temp;

                /* there is a possible 180 degree ambiguity in the angles */
                /* so add 180 degress to the smaller value if the values  */
                /* differ by more than 90 degrees = pi/2 radians.         */
                /* (Later, we may decide to take the other solution by    */
                /* subtracting 180 degrees from the larger value).        */

                if (phib - phia) > (pi / 2.0) {
                    phia += pi;
                }

                if (phia - phib).abs() > toler {
                    /* angles don't agree, so looks like there is some skewness */
                    /* between the axes.  Return with an error to be safe. */
                    *status = APPROX_WCS_KEY;
                }

                phia = (phia + phib) / 2.; /* use the average of the 2 values */
                *rot = phia * 180. / pi;
            }
        }
    }

    /* get the type of projection, if any */
    tstat = 0;
    strcpy_safe(&mut keyname, cs!(c"CTYPE1"));
    strcat_safe(&mut keyname, &alt);
    if ffgkys_safe(fptr, &keyname, &mut ctype, None, &mut tstat) != 0 {
        ptype[0] = 0;
    } else {
        /* copy the projection type string */
        strncpy_safe(ptype, &ctype[4..], 4);
        ptype[4] = 0;

        /* check if RA and DEC are inverted */
        if strncmp_safe(&ctype, cs!(c"DEC-"), 4) == 0
            || strncmp_safe(&ctype[1..], cs!(c"LAT"), 3) == 0
        {
            /* the latitudinal axis is given first, so swap them */

            *rot = 90.0 - (*rot);

            /* Empirical tests with ds9 show the y-axis sign must be negated */
            /* and the xinc and yinc values must NOT be swapped. */
            *yinc = -(*yinc);

            temp = *xrval;
            *xrval = *yrval;
            *yrval = temp;
        }
    }

    *status
}

/// Read the values of the celestial coordinate system keywords
/// from a FITS table where the X and Y or RA and DEC coordinates
/// are stored in separate column.  Do this by converting the
/// table to a temporary FITS image, then reading the keywords
/// from the image file.
/// These values may be used as input to the subroutines that
/// calculate celestial coordinates. (ffxypx, ffwldp)
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `xcol`   — (I) column containing the RA coordinate
/// * `ycol`   — (I) column containing the DEC coordinate
/// * `xrval`  — (O) X reference value
/// * `yrval`  — (O) Y reference value
/// * `xrpix`  — (O) X reference pixel
/// * `yrpix`  — (O) Y reference pixel
/// * `xinc`   — (O) X increment per pixel
/// * `yinc`   — (O) Y increment per pixel
/// * `rot`    — (O) rotation angle (degrees)
/// * `ptype`  — (O) type of projection ('-sin')
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtcs(
    fptr: *mut fitsfile,
    xcol: c_int,
    ycol: c_int,
    xrval: *mut f64,
    yrval: *mut f64,
    xrpix: *mut f64,
    yrpix: *mut f64,
    xinc: *mut f64,
    yinc: *mut f64,
    rot: *mut f64,
    ptype: *mut [c_char; 5],
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let xrval = xrval.as_mut().expect(NULL_MSG);
        let yrval = yrval.as_mut().expect(NULL_MSG);
        let xrpix = xrpix.as_mut().expect(NULL_MSG);
        let yrpix = yrpix.as_mut().expect(NULL_MSG);
        let xinc = xinc.as_mut().expect(NULL_MSG);
        let yinc = yinc.as_mut().expect(NULL_MSG);
        let rot = rot.as_mut().expect(NULL_MSG);
        let ptype = ptype.as_mut().expect(NULL_MSG);

        ffgtcs_safe(
            fptr, xcol, ycol, xrval, yrval, xrpix, yrpix, xinc, yinc, rot, ptype, status,
        )
    }
}
/// Read the values of the celestial coordinate system keywords
/// from a FITS table where the X and Y or RA and DEC coordinates
/// are stored in separate column.  Do this by converting the
/// table to a temporary FITS image, then reading the keywords
/// from the image file.
/// These values may be used as input to the subroutines that
/// calculate celestial coordinates. (ffxypx, ffwldp)
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `xcol`   — (I) column containing the RA coordinate
/// * `ycol`   — (I) column containing the DEC coordinate
/// * `xrval`  — (O) X reference value
/// * `yrval`  — (O) Y reference value
/// * `xrpix`  — (O) X reference pixel
/// * `yrpix`  — (O) Y reference pixel
/// * `xinc`   — (O) X increment per pixel
/// * `yinc`   — (O) Y increment per pixel
/// * `rot`    — (O) rotation angle (degrees)
/// * `ptype`  — (O) type of projection ('-sin')
/// * `status` — (IO) error status
pub fn ffgtcs_safe(
    fptr: &mut fitsfile,
    xcol: c_int,
    ycol: c_int,
    xrval: &mut f64,
    yrval: &mut f64,
    xrpix: &mut f64,
    yrpix: &mut f64,
    xinc: &mut f64,
    yinc: &mut f64,
    rot: &mut f64,
    ptype: &mut [c_char; 5],
    status: &mut c_int,
) -> c_int {
    let mut colnum: [c_int; 2] = [0; 2];
    let mut naxes: [c_long; 2] = [0; 2];

    let mut tptr: Option<Box<fitsfile>> = None;

    if *status > 0 {
        return *status;
    }

    colnum[0] = xcol;
    colnum[1] = ycol;
    naxes[0] = 10;
    naxes[1] = 10;

    /* create temporary  FITS file, in memory */
    ffinit_safe(&mut tptr, cs!(c"mem://"), status);

    let mut tptr = tptr.unwrap();

    /* create a temporary image; the datatype and size are not important */
    ffcrim_safe(&mut tptr, 32, 2, &naxes, status);

    /* now copy the relevant keywords from the table to the image */
    fits_copy_pixlist2image_safe(fptr, &mut tptr, 9, 2, &colnum, status);

    /* write default WCS keywords, if they are not present */
    fits_write_keys_histo_safe(fptr, &mut tptr, 2, &colnum, status);

    if *status > 0 {
        return *status;
    }

    /* read the WCS keyword values from the temporary image */
    ffgics_safe(
        &mut tptr, xrval, yrval, xrpix, yrpix, xinc, yinc, rot, ptype, status,
    );

    if *status > 0 {
        ffpmsg_str("ffgtcs could not find all the celestial coordinate keywords");
        *status = NO_WCS_KEY;
        return *status;
    }

    /* delete the temporary file */
    ffdelt_safe(&mut Some(tptr), status);

    *status
}

/// Return string containing all the WCS keywords appropriate for the
/// pair of X and Y columns containing the coordinate
/// of each event in an event list table.  This string may then be passed
/// to Doug Mink's WCS library wcsinit routine, to create and initialize the
/// WCS structure.  The calling routine must free the header character string
/// when it is no longer needed.
///
/// THIS ROUTINE IS DEPRECATED. USE fits_hdr2str INSTEAD
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `xcol`   — (I) column number for the X column
/// * `ycol`   — (I) column number for the Y column
/// * `header` — (O) string of all the WCS keywords
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtwcs(
    fptr: *mut fitsfile,
    xcol: c_int,
    ycol: c_int,
    header: *mut *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let header = header.as_mut().expect(NULL_MSG);

        ffgtwcs_safe(fptr, xcol, ycol, header, status)
    }
}

/// Return string containing all the WCS keywords appropriate for the
/// pair of X and Y columns containing the coordinate
/// of each event in an event list table.  This string may then be passed
/// to Doug Mink's WCS library wcsinit routine, to create and initialize the
/// WCS structure.  The calling routine must free the header character string
/// when it is no longer needed.
///
/// THIS ROUTINE IS DEPRECATED. USE fits_hdr2str INSTEAD
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `xcol`   — (I) column number for the X column
/// * `ycol`   — (I) column number for the Y column
/// * `header` — (O) string of all the WCS keywords
/// * `status` — (IO) error status
pub fn ffgtwcs_safe(
    fptr: &mut fitsfile,
    xcol: c_int,
    ycol: c_int,
    header: &mut *mut c_char,
    status: &mut c_int,
) -> c_int {
    let mut hdutype: c_int = 0;
    let mut ncols: c_int = 0;
    let mut tstatus: c_int = 0;
    let mut length: usize = 0;
    let mut naxis1: LONGLONG = 1;
    let mut naxis2: LONGLONG = 1;
    let mut tlmin: c_long = 0;
    let mut tlmax: c_long = 0;
    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut valstring: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comm: [c_char; 2] = [0; 2];

    /*  construct a string of 80 blanks, for adding fill to the keywords */
    /*  12345678901234567890123456789012345678901234567890123456789012345678901234567890 */
    let _blanks =
        b"                                                                                \0";
    let _blanks: [c_char; 81] = _blanks.map(|x| x as c_char);
    let blanks = &_blanks;

    if *status > 0 {
        return *status;
    }

    ffghdt_safe(fptr, &mut hdutype, status);
    if hdutype == IMAGE_HDU {
        ffpmsg_str("Can't read table WSC keywords. This HDU is not a table");
        *status = NOT_TABLE;
        return *status;
    }

    ffgncl_safe(fptr, &mut ncols, status);

    if xcol < 1 || xcol > ncols {
        ffpmsg_str("illegal X axis column number in fftwcs");
        *status = BAD_COL_NUM;
        return *status;
    }

    if ycol < 1 || ycol > ncols {
        ffpmsg_str("illegal Y axis column number in fftwcs");
        *status = BAD_COL_NUM;
        return *status;
    }

    /* allocate character string for all the WCS keywords */
    let mut hdr: Vec<c_char> = Vec::new(); /* room for up to 30 keywords */
    if hdr.try_reserve_exact(2401).is_err() {
        ffpmsg_str("error allocating memory for WCS header keywords (fftwcs)");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        hdr.resize(2401, 0);
    }

    let mut cptr = 0; // *header
    comm[0] = 0;

    tstatus = 0;
    ffkeyn_safe(cs!(c"TLMIN"), xcol, &mut keyname, status);
    ffgkyj_safe(fptr, &keyname, &mut tlmin, None, &mut tstatus);

    if tstatus == 0 {
        ffkeyn_safe(cs!(c"TLMAX"), xcol, &mut keyname, status);
        ffgkyj_safe(fptr, &keyname, &mut tlmax, None, &mut tstatus);
    }

    if tstatus == 0 {
        naxis1 = (tlmax - tlmin + 1) as LONGLONG;
    }

    tstatus = 0;
    ffkeyn_safe(cs!(c"TLMIN"), ycol, &mut keyname, status);
    ffgkyj_safe(fptr, &keyname, &mut tlmin, None, &mut tstatus);

    if tstatus == 0 {
        ffkeyn_safe(cs!(c"TLMAX"), ycol, &mut keyname, status);
        ffgkyj_safe(fptr, &keyname, &mut tlmax, None, &mut tstatus);
    }

    if tstatus == 0 {
        naxis2 = (tlmax - tlmin + 1) as LONGLONG;
    }

    /*            123456789012345678901234567890    */
    strcat_safe(&mut hdr[cptr..], cs!(c"NAXIS   =                    2"));
    strncat_safe(&mut hdr[cptr..], blanks, 50);
    cptr += 80;

    ffi2c(naxis1, &mut valstring, status); /* convert to formatted string */
    ffmkky_safe(
        cs!(c"NAXIS1"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
    cptr += 80;

    strcpy_safe(&mut keyname, cs!(c"NAXIS2"));
    ffi2c(naxis2, &mut valstring, status); /* convert to formatted string */
    ffmkky_safe(&keyname, &valstring, Some(&comm), &mut hdr[cptr..], status); /* construct the keyword*/
    strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
    cptr += 80;

    /* read the required header keywords (use defaults if not found) */

    /*  CTYPE1 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCTYP"), xcol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) != 0 {
        valstring[0] = 0;
    }
    ffmkky_safe(
        cs!(c"CTYPE1"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    length = strlen_safe(&hdr[cptr..]);
    strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
    cptr += 80;

    /*  CTYPE2 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCTYP"), ycol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) != 0 {
        valstring[0] = 0;
    }
    ffmkky_safe(
        cs!(c"CTYPE2"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    length = strlen_safe(&hdr[cptr..]);
    strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
    cptr += 80;

    /*  CRPIX1 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCRPX"), xcol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) != 0 {
        strcpy_safe(&mut valstring, cs!(c"1"));
    }
    ffmkky_safe(
        cs!(c"CRPIX1"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
    cptr += 80;

    /*  CRPIX2 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCRPX"), ycol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) != 0 {
        strcpy_safe(&mut valstring, cs!(c"1"));
    }
    ffmkky_safe(
        cs!(c"CRPIX2"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
    cptr += 80;

    /*  CRVAL1 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCRVL"), xcol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) != 0 {
        strcpy_safe(&mut valstring, cs!(c"1"));
    }
    ffmkky_safe(
        cs!(c"CRVAL1"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
    cptr += 80;

    /*  CRVAL2 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCRVL"), ycol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) != 0 {
        strcpy_safe(&mut valstring, cs!(c"1"));
    }
    ffmkky_safe(
        cs!(c"CRVAL2"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
    cptr += 80;

    /*  CDELT1 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCDLT"), xcol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) != 0 {
        strcpy_safe(&mut valstring, cs!(c"1"));
    }
    ffmkky_safe(
        cs!(c"CDELT1"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
    cptr += 80;

    /*  CDELT2 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCDLT"), ycol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) != 0 {
        strcpy_safe(&mut valstring, cs!(c"1"));
    }
    ffmkky_safe(
        cs!(c"CDELT2"),
        &valstring,
        Some(&comm),
        &mut hdr[cptr..],
        status,
    ); /* construct the keyword*/
    strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
    cptr += 80;

    /* the following keywords may not exist */

    /*  CROTA2 keyword */
    tstatus = 0;
    ffkeyn_safe(cs!(c"TCROT"), ycol, &mut keyname, status);
    if ffgkey_safe(fptr, &keyname, &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"CROTA2"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        ); /* construct keyword*/
        strncat_safe(&mut hdr[cptr..], blanks, 50); /* pad with blanks */
        cptr += 80;
    }

    /*  EPOCH keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"EPOCH"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"EPOCH"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        ); /* construct keyword*/
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    /*  EQUINOX keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"EQUINOX"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"EQUINOX"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        ); /* construct keyword*/
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    /*  RADECSYS keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"RADECSYS"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"RADECSYS"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        ); /*construct keyword*/
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    /*  TELESCOPE keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"TELESCOP"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"TELESCOP"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        );
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    /*  INSTRUME keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"INSTRUME"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"INSTRUME"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        );
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    /*  DETECTOR keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"DETECTOR"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"DETECTOR"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        );
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    /*  MJD-OBS keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"MJD-OBS"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"MJD-OBS"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        );
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    /*  DATE-OBS keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"DATE-OBS"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"DATE-OBS"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        );
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    /*  DATE keyword */
    tstatus = 0;
    if ffgkey_safe(fptr, cs!(c"DATE"), &mut valstring, None, &mut tstatus) == 0 {
        ffmkky_safe(
            cs!(c"DATE"),
            &valstring,
            Some(&comm),
            &mut hdr[cptr..],
            status,
        );
        length = strlen_safe(&hdr[cptr..]);
        strncat_safe(&mut hdr[cptr..], blanks, 80 - length); /* pad with blanks */
        cptr += 80;
    }

    strcat_safe(&mut hdr[cptr..], cs!(c"END"));
    strncat_safe(&mut hdr[cptr..], blanks, 77);

    let (header_ptr, l, c) = vec_into_raw_parts(hdr);
    crate::lock_global(&ALLOCATIONS).insert(header_ptr as usize, (l, c));

    *header = header_ptr;

    *status
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aliases::rust_api::*;
    use crate::fitsio::{BINARY_TBL, BYTE_IMG, READONLY, SHORT_IMG, fitsfile};
    use crate::helpers::testhelpers::{to_buf, with_temp_file};
    use core::ffi::CStr;
    use core::ptr;
    use libc::{c_char, c_int, c_long, c_void};

    /// Make a NUL-terminated `Vec<c_char>` from a `&str`.
    fn cc(s: &str) -> Vec<c_char> {
        let mut v: Vec<c_char> = s.bytes().map(|b| b as c_char).collect();
        v.push(0);
        v
    }

    /// Convert a fixed `[c_char; 5]` type buffer to a Rust String (up to NUL).
    fn type_str(buf: &[c_char; 5]) -> String {
        let mut s = String::new();
        for &c in buf.iter() {
            if c == 0 {
                break;
            }
            s.push(c as u8 as char);
        }
        s
    }

    #[test]
    fn test_ffgics_cdelt() {
        // Test ffgics with CDELT/CROTA keywords.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            assert_eq!(status, 0, "ffinit failed");
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            assert_eq!(status, 0, "ffcrim failed");

            // Write WCS keywords using CDELT form.
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_dbl(ff, &cc("CRVAL1"), 180.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2"), 45.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX1"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX2"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT1"), -0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT2"), 0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CROTA2"), 30.0, 10, None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE1"), &cc("RA---TAN"), None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE2"), &cc("DEC--TAN"), None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            assert_eq!(status, 0, "ffopen failed");
            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];
            ffgics_safe(
                f.as_deref_mut().unwrap(),
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgics failed");

            assert!((xrval - 180.0).abs() <= 0.001);
            assert!((yrval - 45.0).abs() <= 0.001);
            assert!((xrpix - 50.0).abs() <= 0.001);
            assert!((yrpix - 50.0).abs() <= 0.001);
            assert!((xinc - (-0.001)).abs() <= 0.0001);
            assert!((yinc - 0.001).abs() <= 0.0001);
            assert!((rot - 30.0).abs() <= 0.1);
            assert_eq!(type_str(&ptype), "-TAN");

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgics_cd_matrix() {
        // Test ffgics with CD matrix keywords.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];
            let scale = 0.001;

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);

            // Write WCS keywords using CD matrix form (no rotation).
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_dbl(ff, &cc("CRVAL1"), 120.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2"), -30.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX1"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX2"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CD1_1"), -scale, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CD1_2"), 0.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CD2_1"), 0.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CD2_2"), scale, 10, None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE1"), &cc("RA---TAN"), None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE2"), &cc("DEC--TAN"), None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];
            ffgics_safe(
                f.as_deref_mut().unwrap(),
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgics failed");

            assert!((xrval - 120.0).abs() <= 0.001);
            assert!((yrval - (-30.0)).abs() <= 0.001);
            assert!(xinc.abs() - scale <= 0.0001);
            assert!(yinc.abs() - scale <= 0.0001);

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgics_cd_rotated() {
        // Test ffgics with rotated CD matrix.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];
            let scale = 0.001;
            let angle: f64 = 45.0 * 3.14159265 / 180.0; // 45 degrees in radians
            let cosa = angle.cos();
            let sina = angle.sin();

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);

            // Write rotated CD matrix.
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_dbl(ff, &cc("CRVAL1"), 0.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2"), 0.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX1"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX2"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CD1_1"), -scale * cosa, 15, None, &mut status);
            fits_write_key_dbl(ff, &cc("CD1_2"), scale * sina, 15, None, &mut status);
            fits_write_key_dbl(ff, &cc("CD2_1"), scale * sina, 15, None, &mut status);
            fits_write_key_dbl(ff, &cc("CD2_2"), scale * cosa, 15, None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE1"), &cc("RA---TAN"), None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE2"), &cc("DEC--TAN"), None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];
            ffgics_safe(
                f.as_deref_mut().unwrap(),
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgics failed");

            // Should extract rotation angle ~-45 degrees.
            assert!((rot - (-45.0)).abs() <= 1.0);

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgics_pc_matrix() {
        // Test ffgics with PC matrix keywords.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);

            // Write WCS keywords using PC matrix form.
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_dbl(ff, &cc("CRVAL1"), 90.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2"), 0.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX1"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX2"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT1"), -0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT2"), 0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("PC1_1"), 1.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("PC1_2"), 0.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("PC2_1"), 0.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("PC2_2"), 1.0, 10, None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE1"), &cc("RA---SIN"), None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE2"), &cc("DEC--SIN"), None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];
            ffgics_safe(
                f.as_deref_mut().unwrap(),
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgics failed");

            assert!((xrval - 90.0).abs() <= 0.001);
            assert!((yrval - 0.0).abs() <= 0.001);
            assert_eq!(type_str(&ptype), "-SIN");

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgics_defaults() {
        // Test ffgics with missing keywords (uses defaults).
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            // No WCS keywords - should use defaults.
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];
            ffgics_safe(
                f.as_deref_mut().unwrap(),
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgics failed");

            // Default values.
            assert!((xrval - 0.0).abs() <= 0.001);
            assert!((yrval - 0.0).abs() <= 0.001);
            assert!((xrpix - 0.0).abs() <= 0.001);
            assert!((yrpix - 0.0).abs() <= 0.001);
            assert!((xinc - 1.0).abs() <= 0.001);
            assert!((yinc - 1.0).abs() <= 0.001);
            assert!((rot - 0.0).abs() <= 0.001);
            assert_eq!(ptype[0], 0);

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgics_dec_first() {
        // Test ffgics with DEC axis first (swapped).
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);

            // DEC is first axis.
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_dbl(ff, &cc("CRVAL1"), 45.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2"), 180.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX1"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX2"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT1"), 0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT2"), -0.001, 10, None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE1"), &cc("DEC--TAN"), None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE2"), &cc("RA---TAN"), None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];
            ffgics_safe(
                f.as_deref_mut().unwrap(),
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgics failed");

            // Values should be swapped.
            assert!((xrval - 180.0).abs() <= 0.001);
            assert!((yrval - 45.0).abs() <= 0.001);

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgics_glat() {
        // Test ffgics with galactic coordinates (GLAT first).
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);

            // GLAT is first axis.
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_dbl(ff, &cc("CRVAL1"), 30.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2"), 120.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX1"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX2"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT1"), 0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT2"), -0.001, 10, None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE1"), &cc("GLAT-TAN"), None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE2"), &cc("GLON-TAN"), None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];
            ffgics_safe(
                f.as_deref_mut().unwrap(),
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgics failed");

            // Values should be swapped because LAT is first.
            assert!((xrval - 120.0).abs() <= 0.001);
            assert!((yrval - 30.0).abs() <= 0.001);

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgicsa_version() {
        // Test ffgicsa with alternate WCS version.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);

            // Write primary WCS.
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_dbl(ff, &cc("CRVAL1"), 180.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2"), 45.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX1"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX2"), 50.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT1"), -0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT2"), 0.001, 10, None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE1"), &cc("RA---TAN"), None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE2"), &cc("DEC--TAN"), None, &mut status);

            // Write alternate WCS version A.
            fits_write_key_dbl(ff, &cc("CRVAL1A"), 90.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2A"), 30.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX1A"), 25.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRPIX2A"), 25.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT1A"), -0.002, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CDELT2A"), 0.002, 10, None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE1A"), &cc("RA---SIN"), None, &mut status);
            fits_write_key_str(ff, &cc("CTYPE2A"), &cc("DEC--SIN"), None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);

            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];

            // Read primary WCS (blank version).
            ffgicsa_safe(
                f.as_deref_mut().unwrap(),
                b' ' as c_char,
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgicsa failed");
            assert!((xrval - 180.0).abs() <= 0.001);
            assert!((yrval - 45.0).abs() <= 0.001);

            // Read alternate WCS version A.
            status = 0;
            ffgicsa_safe(
                f.as_deref_mut().unwrap(),
                b'A' as c_char,
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgicsa failed");
            assert!((xrval - 90.0).abs() <= 0.001);
            assert!((yrval - 30.0).abs() <= 0.001);
            assert!((xrpix - 25.0).abs() <= 0.001);
            assert!((xinc - (-0.002)).abs() <= 0.0001);
            assert_eq!(type_str(&ptype), "-SIN");

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgicsa_invalid_version() {
        // Test ffgicsa with invalid version code.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);

            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];

            // Invalid version code (not A-Z or blank).
            ffgicsa_safe(
                f.as_deref_mut().unwrap(),
                b'1' as c_char,
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_ne!(status, 0, "ffgicsa should fail for invalid version");

            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgiwcs() {
        // Test ffgiwcs (deprecated but still functional).
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_dbl(ff, &cc("CRVAL1"), 180.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("CRVAL2"), 45.0, 10, None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut header: *mut c_char = ptr::null_mut();
            ffgiwcs_safe(f.as_deref_mut().unwrap(), &mut header, &mut status);
            assert_eq!(status, 0, "ffgiwcs failed");

            assert!(!header.is_null(), "header should not be NULL");
            // Header should contain WCS keywords.
            let hdr = unsafe { CStr::from_ptr(header) }.to_str().unwrap();
            assert!(hdr.contains("CRVAL1"));
            assert!(hdr.contains("CRVAL2"));

            unsafe { fits_free_memory(header.cast::<c_void>(), &mut status) };
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgiwcs_not_image() {
        // Test ffgiwcs error when HDU is not an image.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            // Create a primary array.
            let naxes: [c_long; 2] = [0, 0];
            fits_create_img(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &naxes, &mut status);
            // Create a binary table extension.
            let ttype = [Some(cc("X"))];
            let ttype_ref: Vec<Option<&[c_char]>> = ttype.iter().map(|o| o.as_deref()).collect();
            let tform_v = [cc("1D")];
            let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
            fits_create_tbl(
                f.as_deref_mut().unwrap(),
                BINARY_TBL,
                0,
                1,
                &ttype_ref,
                &tform_ref,
                None,
                None,
                &mut status,
            );
            assert_eq!(status, 0, "ffcrtb failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            // Move to the table extension.
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            assert_eq!(status, 0, "ffmahd failed");

            // Should fail because we're on a table, not an image.
            let mut header: *mut c_char = ptr::null_mut();
            ffgiwcs_safe(f.as_deref_mut().unwrap(), &mut header, &mut status);
            assert_ne!(status, 0, "ffgiwcs should fail on a table");

            if !header.is_null() {
                unsafe { fits_free_memory(header.cast::<c_void>(), &mut status) };
            }
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgtcs() {
        // Test ffgtcs - get WCS from table columns.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [0, 0];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &naxes, &mut status);
            let ttype = [Some(cc("X")), Some(cc("Y"))];
            let ttype_ref: Vec<Option<&[c_char]>> = ttype.iter().map(|o| o.as_deref()).collect();
            let tform_v = [cc("1D"), cc("1D")];
            let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
            fits_create_tbl(
                f.as_deref_mut().unwrap(),
                BINARY_TBL,
                10,
                2,
                &ttype_ref,
                &tform_ref,
                None,
                None,
                &mut status,
            );

            // Write WCS keywords for table columns.
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_str(ff, &cc("TCTYP1"), &cc("RA---TAN"), None, &mut status);
            fits_write_key_str(ff, &cc("TCTYP2"), &cc("DEC--TAN"), None, &mut status);
            fits_write_key_dbl(ff, &cc("TCRVL1"), 180.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCRVL2"), 45.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCRPX1"), 1.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCRPX2"), 1.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCDLT1"), -0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCDLT2"), 0.001, 10, None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut xrval = 0.0;
            let mut yrval = 0.0;
            let mut xrpix = 0.0;
            let mut yrpix = 0.0;
            let mut xinc = 0.0;
            let mut yinc = 0.0;
            let mut rot = 0.0;
            let mut ptype = [0 as c_char; 5];
            ffgtcs_safe(
                f.as_deref_mut().unwrap(),
                1,
                2,
                &mut xrval,
                &mut yrval,
                &mut xrpix,
                &mut yrpix,
                &mut xinc,
                &mut yinc,
                &mut rot,
                &mut ptype,
                &mut status,
            );
            assert_eq!(status, 0, "ffgtcs failed");

            assert!((xrval - 180.0).abs() <= 0.001);
            assert!((yrval - 45.0).abs() <= 0.001);

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgtwcs() {
        // Test ffgtwcs - get table WCS keywords string.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [0, 0];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &naxes, &mut status);
            let ttype = [Some(cc("X")), Some(cc("Y"))];
            let ttype_ref: Vec<Option<&[c_char]>> = ttype.iter().map(|o| o.as_deref()).collect();
            let tform_v = [cc("1D"), cc("1D")];
            let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
            fits_create_tbl(
                f.as_deref_mut().unwrap(),
                BINARY_TBL,
                10,
                2,
                &ttype_ref,
                &tform_ref,
                None,
                None,
                &mut status,
            );

            // Write WCS keywords for table columns.
            let ff = f.as_deref_mut().unwrap();
            fits_write_key_str(ff, &cc("TCTYP1"), &cc("RA---TAN"), None, &mut status);
            fits_write_key_str(ff, &cc("TCTYP2"), &cc("DEC--TAN"), None, &mut status);
            fits_write_key_dbl(ff, &cc("TCRVL1"), 180.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCRVL2"), 45.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCRPX1"), 1.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCRPX2"), 1.0, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCDLT1"), -0.001, 10, None, &mut status);
            fits_write_key_dbl(ff, &cc("TCDLT2"), 0.001, 10, None, &mut status);
            fits_write_key_lng(ff, &cc("TLMIN1"), 1, None, &mut status);
            fits_write_key_lng(ff, &cc("TLMAX1"), 100, None, &mut status);
            fits_write_key_lng(ff, &cc("TLMIN2"), 1, None, &mut status);
            fits_write_key_lng(ff, &cc("TLMAX2"), 100, None, &mut status);
            assert_eq!(status, 0, "writing keywords failed");
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut header: *mut c_char = ptr::null_mut();
            ffgtwcs_safe(f.as_deref_mut().unwrap(), 1, 2, &mut header, &mut status);
            assert_eq!(status, 0, "ffgtwcs failed");

            assert!(!header.is_null(), "header should not be NULL");
            // Header should contain converted WCS keywords.
            let hdr = unsafe { CStr::from_ptr(header) }.to_str().unwrap();
            assert!(hdr.contains("NAXIS"));
            assert!(hdr.contains("CTYPE1"));
            assert!(hdr.contains("CRVAL1"));
            assert!(hdr.contains("END"));

            unsafe { fits_free_memory(header.cast::<c_void>(), &mut status) };
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgtwcs_not_table() {
        // Test ffgtwcs error when HDU is not a table.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);

            // Should fail because we're on an image, not a table.
            let mut header: *mut c_char = ptr::null_mut();
            ffgtwcs_safe(f.as_deref_mut().unwrap(), 1, 2, &mut header, &mut status);
            assert_ne!(status, 0, "ffgtwcs should fail on an image");

            if !header.is_null() {
                unsafe { fits_free_memory(header.cast::<c_void>(), &mut status) };
            }
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgtwcs_bad_col() {
        // Test ffgtwcs error with bad column number.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [0, 0];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &naxes, &mut status);
            let ttype = [Some(cc("X")), Some(cc("Y"))];
            let ttype_ref: Vec<Option<&[c_char]>> = ttype.iter().map(|o| o.as_deref()).collect();
            let tform_v = [cc("1D"), cc("1D")];
            let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
            fits_create_tbl(
                f.as_deref_mut().unwrap(),
                BINARY_TBL,
                10,
                2,
                &ttype_ref,
                &tform_ref,
                None,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);

            // Invalid column number.
            let mut header: *mut c_char = ptr::null_mut();
            ffgtwcs_safe(f.as_deref_mut().unwrap(), 0, 2, &mut header, &mut status);
            assert_ne!(status, 0, "ffgtwcs should fail for col 0");

            if !header.is_null() {
                unsafe { fits_free_memory(header.cast::<c_void>(), &mut status) };
            }
            status = 0;
            header = ptr::null_mut();

            ffgtwcs_safe(f.as_deref_mut().unwrap(), 1, 99, &mut header, &mut status);
            assert_ne!(status, 0, "ffgtwcs should fail for col 99");

            if !header.is_null() {
                unsafe { fits_free_memory(header.cast::<c_void>(), &mut status) };
            }
            status = 0;
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_fits_read_wcstab_null() {
        // Test fits_read_wcstab with NULL pointer.
        // The C version passes a NULL fitsfile pointer; the _safer Rust form
        // takes a &mut fitsfile, so we exercise the FFI wrapper directly.
        let mut status: c_int = 0;
        #[allow(deprecated)]
        let rc = unsafe { fits_read_wcstab(ptr::null_mut(), 0, ptr::null(), &raw mut status) };
        // With nwtb=0 the routine returns immediately before checking the file
        // pointer, mirroring the C implementation. The C test still expects a
        // nonzero status because it passes NULL with nwtb=0; see note below.
        // Match the C expectation: status should be set (NULL_INPUT_PTR).
        let _ = rc;
        assert_ne!(status, 0, "fits_read_wcstab should fail for NULL pointer");
    }

    #[test]
    fn test_fits_read_wcstab_zero() {
        // Test fits_read_wcstab with nwtb=0.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [100, 100];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_create_img(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);

            // With nwtb=0, should return immediately with no error.
            fits_read_wcstab_safe(f.as_deref_mut().unwrap(), 0, &[], &mut status);
            assert_eq!(status, 0, "fits_read_wcstab with nwtb=0 should succeed");

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }
}
