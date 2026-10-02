//! The tiled-image compression driver.
//!
//! A tile-compressed image is not stored as an image at all: it is a binary
//! table whose `COMPRESSED_DATA` column holds one variable-length byte array
//! per tile, with `ZIMAGE`, `ZCMPTYPE`, `ZBITPIX`, `ZNAXISn` and `ZTILEn`
//! keywords recording the image it represents. This module is what makes that
//! table look like an image to the rest of the library: it splits an image into
//! tiles, compresses each, and reassembles them on the way back.
//!
//! The codecs themselves are external crates -- `ricecomp`, `pliocomp` and
//! `hcompress` -- plus `libz-rs-sys` for GZIP and `libbz2-rs-sys` for BZIP2.
//! What lives here is the driver: tile geometry, the `ZSCALE`/`ZZERO` scaling,
//! null handling via `ZBLANK`, and the dithering that
//! [`crate::quantize`] applies when a floating point image is compressed
//! lossily.
//!
//! Ported from CFITSIO's `imcompress.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center. See the "Image Compression" section of the CFITSIO
//! User's Reference Guide.
#![warn(missing_docs)]

use crate::helpers::aligned::AlignedBytes;
use core::ffi::CStr;
use core::slice;
use core::{cmp, mem};
use std::collections::HashMap;
use std::os::raw::c_schar;
use std::sync::{LazyLock, Mutex, OnceLock};

use hcompress::read::HCDecoder;
use hcompress::write::HCEncoder;
use ricecomp::read::RCDecoder;
use ricecomp::write::RCEncoder;

use crate::aliases::rust_api::{
    fits_binary_tform, fits_copy_hdu, fits_copy_header, fits_create_hdu, fits_delete_key,
    fits_delete_record, fits_get_colnum, fits_get_hdrspace, fits_get_hdu_num, fits_get_hdu_type,
    fits_get_hduaddrll, fits_get_img_param, fits_get_num_cols, fits_get_num_rowsll,
    fits_insert_col, fits_make_keyn, fits_modify_key_lng, fits_modify_key_str, fits_modify_name,
    fits_movabs_hdu, fits_parse_value, fits_read_card, fits_read_col, fits_read_col_byt,
    fits_read_colnull, fits_read_descript, fits_read_descriptll, fits_read_key, fits_read_key_dbl,
    fits_read_key_flt, fits_read_key_lng, fits_read_key_log, fits_read_key_str, fits_set_hdustruc,
    fits_set_tscale, fits_update_card, fits_update_key, fits_write_col, fits_write_descript,
    fits_write_history, fits_write_img, fits_write_imgnull, fits_write_key, fits_write_key_lng,
    fits_write_record, fits_write_subset,
};
use crate::buffers::{ffgbyt, ffmbyt_safe, ffpbyt};
use crate::c_types::{
    c_char, c_int, c_long, c_short, c_uchar, c_uint, c_ulong, c_ushort, c_void, size_t,
};
use crate::fitsio::{
    BAD_DATATYPE, BAD_DIMEN, BAD_NAXIS, BAD_PIX_NUM, BINARY_TBL, BYTE_IMG, BZIP2_1, CASEINSEN,
    COL_NOT_FOUND, DATA_COMPRESSION_ERR, DATA_DECOMPRESSION_ERR, DOUBLE_IMG, DOUBLENULLVALUE,
    END_OF_FILE, FLEN_CARD, FLEN_COMMENT, FLEN_KEYWORD, FLEN_VALUE, FLOAT_IMG, FLOATNULLVALUE,
    GZIP_1, GZIP_2, HCOMPRESS_1, INT32BIT, LONG_IMG, LONG_MAX, LONG_MIN, LONGLONG,
    MAX_COMPRESS_DIM, MEMORY_ALLOCATION, NEG_AXIS, NO_COMPRESSED_TILE, NO_DITHER, NOCOMPRESS,
    NOT_BTABLE, NULL_MSG, NUM_OVERFLOW, OVERFLOW_ERR, PLIO_1, RICE_1, SBYTE_IMG, SHORT_IMG,
    SUBTRACTIVE_DITHER_1, SUBTRACTIVE_DITHER_2, TBIT, TBYTE, TCOMPLEX, TDBLCOMPLEX, TDOUBLE,
    TFLOAT, TINT, TLOGICAL, TLONG, TLONGLONG, TSBYTE, TSHORT, TSTRING, TUINT, TULONG, TUSHORT,
    TYP_CKSUM_KEY, TYP_CMPRS_KEY, ULONG_IMG, USHORT_IMG, fitsfile,
};
use crate::getcolb::{fffi1i1, fffi2i1, fffi4i1, ffgsvb_safe};
use crate::getcold::{
    fffi1r8, fffi2r8, fffi4r8, fffr4r8, fffr8r8, fffr8r8_inplace, ffgcvd_safe, ffgsvd_safe,
};
use crate::getcole::{fffi1r4, fffi2r4, fffi4r4, fffr4r4, fffr4r4_inplace, fffr8r4, ffgsve_safe};
use crate::getcoli::{fffi1i2, fffi2i2, fffi4i2, fffi8i2, ffgsvi_safe};
use crate::getcolj::{fffi1i4, fffi2i4, fffi4i4, fffi8i4};
use crate::getcolk::{fffi1int, fffi2int, fffi4int, fffi8int, ffgcvk_safe, ffgsvk_safe};
use crate::getcolsb::{fffi1s1, fffi2s1, fffi4s1};
use crate::getcolui::{fffi1u2, fffi2u2, fffi4u2, fffi8u2};
use crate::getcoluj::{fffi1u4, fffi2u4, fffi4u4, fffi8u4};
use crate::getcoluk::{fffi1uint, fffi2uint, fffi4uint, fffi8uint};
use crate::putcolb::ffpclb_safe;
use crate::putcold::ffpcld_safe;
use crate::putcoli::ffpcli_safe;
use crate::scalnull::ffpscl_safe;

use bytemuck::{cast, cast_slice, cast_slice_mut};

use crate::pliocomp::{pl_l2pi, pl_p2li, pl_p2li_max_len};

use crate::fitscore::{
    ffcmrk_safe, ffcrhd_safe, ffgcno_safe, ffgdesll_safe, ffghadll_safe, ffgidm_safe, ffgipr_safe,
    ffgisz_safe, ffgkcl_safe, ffmahd_safe, ffpmrk_safe, ffpmsg_cstr, ffpmsg_slice, ffpmsg_str,
    ffpsvc_safe, ffpxsz, ffrdef_safe, fits_is_compressed_image_safe, fits_strcasecmp,
    fits_strncasecmp, fits_translate_keywords_safe,
};
use crate::getkey::{ffdtdm_safe, ffgcrd_safe, ffghsp_safe, ffgky_safe, ffgrec_safe};
use crate::modkey::{ffikyj_safe, ffucrd_safe};
use crate::putkey::{
    ffcrim_safe, ffcrtb_safe, ffpdat_safe, ffpkye_safe, ffpkyg_safe, ffpkyj_safe, ffpkyl_safe,
    ffpkys_safe, ffprec_safe,
};
use crate::quantize::{
    DitherType, fits_img_stats_int_safe, fits_quantize_double_inplace, fits_quantize_float_inplace,
};
use crate::swapproc::{ffswap2, ffswap4, ffswap8};

use crate::zcompress::{compress2mem_from_mem, uncompress2mem_from_mem};
use crate::{FFLOCK, FFUNLOCK, KeywordDatatype, KeywordDatatypeMut};

use crate::fitsio2::*;
use crate::relibc::header::stdio::snprintf_f64;
use crate::wrappers::*;
use crate::{NullCheckType, NullValue};
use crate::{bb, cs, int_snprintf};

const NULL_VALUE: c_int = -2147483647; /* value used to represent undefined pixels */
const ZERO_VALUE: c_int = -2147483646; /* value used to represent zero-valued pixels */

/* special quantize level value indicates that floating point image pixels */
/* should not be quantized and instead losslessly compressed (with GZIP) */
const NO_QUANTIZE: f32 = 9999.0;

/* string array for storing the individual column compression stats */
//char results[999][30];

//float *fits_rand_value = 0;

pub(crate) static FITS_RAND_VALUE: OnceLock<Vec<f32>> = OnceLock::new();

pub(crate) static TILE_STRUCTS: LazyLock<Mutex<HashMap<usize, TileStruct>>> =
    LazyLock::new(Default::default);

pub(crate) struct TileStruct {
    /// Row number of the array of uncompressed tiledata.
    pub tilerow: Vec<c_int>,
    /// Length of the array of tile data in bytes.
    pub tiledatasize: Vec<c_long>,
    /// Datatype of the array of tile (TINT, TSHORT, etc).
    pub tiletype: Vec<c_int>,
    /// Array of uncompressed tile of data, for row *tilerow.
    pub tiledata: Vec<Vec<u8>>,
    /// Array of optional array of null value flags.
    pub tilenullarray: Vec<Vec<c_char>>,
    /// Anynulls in the array of tile?
    pub tileanynull: Vec<c_int>,
}

pub(crate) fn fits_init_randoms() -> c_int {
    /* initialize an array of random numbers */

    let _ii: i32;
    let a: f64 = 16807.0;
    let m: f64 = 2147483647.0;
    let mut temp: f64;

    let lock = FFLOCK();

    if FITS_RAND_VALUE.get().is_some() {
        FFUNLOCK(lock);
        return 0; /* array is already initialized */
    }

    /* allocate array for the random number sequence */
    /* WARNIGN: THIS MEMORY IS NEVER FREED */
    let mut v = Vec::new();
    if v.try_reserve_exact(N_RANDOM).is_err() {
        FFUNLOCK(lock);
        return MEMORY_ALLOCATION;
    } else {
        v.resize(N_RANDOM, 0.0);
    }

    let rand_value = v.as_mut_slice();

    /*  We need a portable algorithm that anyone can use to generate this
        exact same sequence of random number.  The C 'rand' function is not
    suitable because it is not available to Fortran or Java programmers.
    Instead, use a well known simple algorithm published here:
    "Random number generators: good ones are hard to find", Communications of the ACM,
        Volume 31 ,  Issue 10  (October 1988) Pages: 1192 - 1201
    */

    /* initialize the random numbers */
    let mut seed: f64 = 1.0;
    let mut ii = 0;
    while ii < N_RANDOM {
        temp = a * seed;
        seed = temp - m * f64::from((temp / m) as i32);
        rand_value[ii] = (seed / m) as f32;
        ii += 1;
    }

    let _ = FITS_RAND_VALUE.set(v);

    FFUNLOCK(lock);

    /*
    IMPORTANT NOTE: the 10000th seed value must have the value 1043618065 if the
       algorithm has been implemented correctly */

    if (seed as i32) != 1043618065 {
        ffpmsg_str("fits_init_randoms generated incorrect random number sequence");
        1
    } else {
        0
    }
}

/// Error callback the bzip2 library calls on an internal failure.
///
/// Declared by the bzip2 code in `bzlib_private.h`; it is never expected to be
/// reached, and pushes a message onto the error stack if it is.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub extern "C" fn bz_internal_error(_errcode: c_int) {
    // FFI WRAPPER
    /* external function declared by the bzip2 code in bzlib_private.h */
    ffpmsg_str("bzip2 returned an internal error");
    ffpmsg_str("This should never happen");
}

/// This routine specifies the image compression algorithm that should be
/// used when writing a FITS image.  The image is divided into tiles, and
/// each tile is compressed and stored in a row of at variable length binary
/// table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ctype`  — image compression type code;
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_compression_type(
    fptr: *mut fitsfile,
    ctype: c_int,
    /* allowed values: RICE_1, GZIP_1, GZIP_2, PLIO_1,     */
    /*  HCOMPRESS_1, BZIP2_1, and NOCOMPRESS               */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        fits_set_compression_type_safe(fptr, ctype, status)
    }
}

/// This routine specifies the image compression algorithm that should be
/// used when writing a FITS image.  The image is divided into tiles, and
/// each tile is compressed and stored in a row of at variable length binary
/// table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ctype`  — image compression type code;
/// * `status` — (IO) error status
pub fn fits_set_compression_type_safe(
    fptr: &mut fitsfile,
    ctype: c_int,
    /* allowed values: RICE_1, GZIP_1, GZIP_2, PLIO_1,     */
    /*  HCOMPRESS_1, BZIP2_1, and NOCOMPRESS               */
    status: &mut c_int,
) -> c_int {
    if ctype != RICE_1
        && ctype != GZIP_1
        && ctype != GZIP_2
        && ctype != PLIO_1
        && ctype != HCOMPRESS_1
        && ctype != BZIP2_1
        && ctype != NOCOMPRESS
        && ctype != 0
    {
        ffpmsg_str("unknown compression algorithm (fits_set_compression_type)");
        *status = DATA_COMPRESSION_ERR;
    } else {
        fptr.Fptr.request_compress_type = ctype;
    }

    *status
}

/// This routine specifies the size (dimension) of the image
/// compression  tiles that should be used when writing a FITS
/// image.  The image is divided into tiles, and each tile is compressed
/// and stored in a row of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ndim`   — number of dimensions in the compressed image
/// * `dims`   — size of image compression tile in each dimension
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_tile_dim(
    fptr: *mut fitsfile,
    ndim: c_int,
    dims: *const c_long,
    /* default tile size = (NAXIS1, 1, 1, ...)            */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        if ndim < 0 || ndim > MAX_COMPRESS_DIM as c_int {
            *status = BAD_DIMEN;
            ffpmsg_str("illegal number of tile dimensions (fits_set_tile_dim)");
            return *status;
        }

        let dims = slice::from_raw_parts(dims, ndim as usize);

        fits_set_tile_dim_safe(fptr, ndim, dims, status)
    }
}

/// This routine specifies the size (dimension) of the image
/// compression  tiles that should be used when writing a FITS
/// image.  The image is divided into tiles, and each tile is compressed
/// and stored in a row of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ndim`   — number of dimensions in the compressed image
/// * `dims`   — size of image compression tile in each dimension
/// * `status` — (IO) error status
pub fn fits_set_tile_dim_safe(
    fptr: &mut fitsfile,
    ndim: c_int,
    dims: &[c_long],
    /* default tile size = (NAXIS1, 1, 1, ...)            */
    status: &mut c_int,
) -> c_int {
    if ndim < 0 || ndim > MAX_COMPRESS_DIM as c_int {
        *status = BAD_DIMEN;
        ffpmsg_str("illegal number of tile dimensions (fits_set_tile_dim)");
        return *status;
    }

    fptr.Fptr.request_tilesize[..(ndim as usize)].copy_from_slice(&dims[..(ndim as usize)]);

    *status
}

/// This routine specifies the value of the quantization level, q,  that
/// should be used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `qlevel` — floating point quantization level
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_quantize_level(
    fptr: *mut fitsfile,
    qlevel: f32,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        fits_set_quantize_level_safe(fptr, qlevel, status)
    }
}

/// This routine specifies the value of the quantization level, q,  that
/// should be used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `qlevel` — floating point quantization level
/// * `status` — (IO) error status
pub fn fits_set_quantize_level_safe(fptr: &mut fitsfile, qlevel: f32, status: &mut c_int) -> c_int {
    if qlevel == 0. {
        /* this means don't quantize the floating point values. Instead, */
        /* the floating point values will be losslessly compressed */
        fptr.Fptr.request_quantize_level = NO_QUANTIZE;
    } else {
        fptr.Fptr.request_quantize_level = qlevel;
    }

    *status
}

///This routine specifies what type of dithering (randomization) should
///be performed when quantizing floating point images to integer prior to
///compression.   A value of -1 means do no dithering.  A value of 0 means
///use the default SUBTRACTIVE_DITHER_1 (which is equivalent to dither = 1).
///A value of 2 means use SUBTRACTIVE_DITHER_2.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `method` — quantization method
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_quantize_method(
    fptr: *mut fitsfile,
    method: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_set_quantize_method_safe(fptr, method, status)
    }
}

///This routine specifies what type of dithering (randomization) should
///be performed when quantizing floating point images to integer prior to
///compression.   A value of -1 means do no dithering.  A value of 0 means
///use the default SUBTRACTIVE_DITHER_1 (which is equivalent to dither = 1).
///A value of 2 means use SUBTRACTIVE_DITHER_2.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `method` — quantization method
/// * `status` — (IO) error status
pub fn fits_set_quantize_method_safe(
    fptr: &mut fitsfile,
    method: c_int,
    status: &mut c_int,
) -> c_int {
    let mut method = method;

    if method < -1 || method > 2 {
        ffpmsg_str("illegal dithering value (fits_set_quantize_method)");
        *status = DATA_COMPRESSION_ERR;
    } else {
        if method == 0 {
            method = 1;
        }
        fptr.Fptr.request_quantize_method = method;
    }

    *status
}

/// The name of this routine has changed.  This is kept here only for backwards
/// compatibility for any software that may be calling the old routine.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `dither` — dither type
/// * `status` — (IO) error status
#[deprecated]
#[cfg_attr(not(test), unsafe(no_mangle))]
pub unsafe extern "C" fn fits_set_quantize_dither(
    fptr: *mut fitsfile,
    dither: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_set_quantize_dither_safe(fptr, dither, status)
    }
}

/// The name of this routine has changed.  This is kept here only for backwards
/// compatibility for any software that may be calling the old routine.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `dither` — dither type
/// * `status` — (IO) error status
pub fn fits_set_quantize_dither_safe(
    fptr: &mut fitsfile,
    dither: c_int,
    status: &mut c_int,
) -> c_int {
    fits_set_quantize_method_safe(fptr, dither, status)
}

/// This routine specifies the value of the offset that should be applied when
/// calculating the random dithering when quantizing floating point iamges.
/// A random offset should be applied to each image to avoid quantization
/// effects when taking the difference of 2 images, or co-adding a set of
/// images.  Without this random offset, the corresponding pixel in every image
/// will have exactly the same dithering.
///
/// offset = 0 means use the default random dithering based on system time
/// offset = negative means randomly chose dithering based on 1st tile checksum
/// offset = [1 - 10000] means use that particular dithering pattern
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `seed`   — random dithering seed value (1 to 10000)
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_dither_seed(
    fptr: *mut fitsfile,
    seed: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_set_dither_seed_safe(fptr, seed, status)
    }
}

/// This routine specifies the value of the offset that should be applied when
/// calculating the random dithering when quantizing floating point iamges.
/// A random offset should be applied to each image to avoid quantization
/// effects when taking the difference of 2 images, or co-adding a set of
/// images.  Without this random offset, the corresponding pixel in every image
/// will have exactly the same dithering.
///
/// offset = 0 means use the default random dithering based on system time
/// offset = negative means randomly chose dithering based on 1st tile checksum
/// offset = [1 - 10000] means use that particular dithering pattern
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `seed`   — random dithering seed value (1 to 10000)
/// * `status` — (IO) error status
pub fn fits_set_dither_seed_safe(fptr: &mut fitsfile, seed: c_int, status: &mut c_int) -> c_int {
    /* if positive, ensure that the value is in the range 1 to 10000 */
    if seed > 10000 {
        ffpmsg_str("illegal dithering seed value (fits_set_dither_seed)");
        *status = DATA_COMPRESSION_ERR;
    } else {
        (fptr.Fptr).request_dither_seed = seed;
    }

    *status
}

/// The name of this routine has changed.  This is kept just for
/// backwards compatibility with any software that calls the old name
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `offset` — random dithering offset value (1 to 10000)
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_dither_offset(
    fptr: *mut fitsfile,
    offset: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_set_dither_offset_safe(fptr, offset, status);
        *status
    }
}

/// The name of this routine has changed.  This is kept just for
/// backwards compatibility with any software that calls the old name
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `offset` — random dithering offset value (1 to 10000)
/// * `status` — (IO) error status
pub fn fits_set_dither_offset_safe(
    fptr: &mut fitsfile,
    offset: c_int,
    status: &mut c_int,
) -> c_int {
    fits_set_dither_seed_safe(fptr, offset, status)
}

/// ********************************************************************
/// ********************************************************************
/// THIS ROUTINE IS PROVIDED ONLY FOR BACKWARDS COMPATIBILITY;
/// ALL NEW SOFTWARE SHOULD CALL fits_set_quantize_level INSTEAD
/// ********************************************************************
/// ********************************************************************
///
/// This routine specifies the value of the noice_bits parameter that
/// should be used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// Feb 2008:  the "noisebits" parameter has been replaced with the more
/// general "quantize level" parameter.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `noisebits` — noise_bits parameter value
/// * `status`    — (IO) error status
#[deprecated]
#[cfg_attr(not(test), unsafe(no_mangle))]
pub unsafe extern "C" fn fits_set_noise_bits(
    fptr: *mut fitsfile,
    noisebits: c_int,
    /* (default = 4)                    */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        #[allow(deprecated)]
        fits_set_noise_bits_safe(fptr, noisebits, status)
    }
}

/// ********************************************************************
/// ********************************************************************
/// THIS ROUTINE IS PROVIDED ONLY FOR BACKWARDS COMPATIBILITY;
/// ALL NEW SOFTWARE SHOULD CALL fits_set_quantize_level INSTEAD
/// ********************************************************************
/// ********************************************************************
///
/// This routine specifies the value of the noice_bits parameter that
/// should be used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// Feb 2008:  the "noisebits" parameter has been replaced with the more
/// general "quantize level" parameter.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `noisebits` — noise_bits parameter value
/// * `status`    — (IO) error status
#[deprecated]
pub fn fits_set_noise_bits_safe(
    fptr: &mut fitsfile,
    noisebits: c_int,
    /* (default = 4)                    */
    status: &mut c_int,
) -> c_int {
    if noisebits < 1 || noisebits > 16 {
        *status = DATA_COMPRESSION_ERR;
        ffpmsg_str("illegal number of noise bits (fits_set_noise_bits)");
        return *status;
    }

    let qlevel = 2.0_f32.powi(noisebits);
    fits_set_quantize_level_safe(fptr, qlevel, status)
}

/// This routine specifies the value of the hcompress scale parameter.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `scale`  — hcompress scale parameter value
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_hcomp_scale(
    fptr: *mut fitsfile,
    scale: f32,
    /* (default = 0.0)                    */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        fits_set_hcomp_scale_safe(fptr, scale, status)
    }
}
/// This routine specifies the value of the hcompress scale parameter.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `scale`  — hcompress scale parameter value
/// * `status` — (IO) error status
pub fn fits_set_hcomp_scale_safe(
    fptr: &mut fitsfile,
    scale: f32,
    /* (default = 0.0)                    */
    status: &mut c_int,
) -> c_int {
    fptr.Fptr.request_hcomp_scale = scale;
    *status
}

/// This routine specifies the value of the hcompress smooth parameter.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `smooth` — hcompress smooth parameter value
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_hcomp_smooth(
    fptr: *mut fitsfile,
    smooth: c_int,
    /* if scale > 1 and smooth != 0, then */
    /*  the image will be smoothed when it is */
    /* decompressed to remove some of the */
    /* 'blockiness' in the image produced */
    /* by the lossy compression    */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_set_hcomp_smooth_safe(fptr, smooth, status)
    }
}

/// This routine specifies the value of the hcompress smooth parameter.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `smooth` — hcompress smooth parameter value
/// * `status` — (IO) error status
pub fn fits_set_hcomp_smooth_safe(fptr: &mut fitsfile, smooth: c_int, status: &mut c_int) -> c_int {
    (fptr.Fptr).request_hcomp_smooth = smooth;
    *status
}

/// This routine specifies whether images with integer pixel values should
/// quantized and compressed the same way float images are compressed.
/// The default is to not do this, and instead apply a lossless compression
/// algorithm to integer images.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `lossy_int` — (I) True (!= 0) or False (0)
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_lossy_int(
    fptr: *mut fitsfile,
    lossy_int: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_set_lossy_int_safe(fptr, lossy_int, status)
    }
}

/// This routine specifies whether images with integer pixel values should
/// quantized and compressed the same way float images are compressed.
/// The default is to not do this, and instead apply a lossless compression
/// algorithm to integer images.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `lossy_int` — (I) True (!= 0) or False (0)
/// * `status`    — (IO) error status
pub fn fits_set_lossy_int_safe(fptr: &mut fitsfile, lossy_int: c_int, status: &mut c_int) -> c_int {
    (fptr.Fptr).request_lossy_int_compress = lossy_int;
    *status
}

/// This routine specifies whether the HDU that is being compressed is so large
/// (i.e., > 4 GB) that the 'Q' type variable length array columns should be used
/// rather than the normal 'P' type.  The allows the heap pointers to be stored
/// as 64-bit quantities, rather than just 32-bits.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `huge`   — (I) True (!= 0) or False (0)
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_huge_hdu(
    fptr: *mut fitsfile,
    huge: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_set_huge_hdu_safe(fptr, huge, status)
    }
}

/// This routine specifies whether the HDU that is being compressed is so large
/// (i.e., > 4 GB) that the 'Q' type variable length array columns should be used
/// rather than the normal 'P' type.  The allows the heap pointers to be stored
/// as 64-bit quantities, rather than just 32-bits.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `huge`   — (I) True (!= 0) or False (0)
/// * `status` — (IO) error status
pub fn fits_set_huge_hdu_safe(fptr: &mut fitsfile, huge: c_int, status: &mut c_int) -> c_int {
    (fptr.Fptr).request_huge_hdu = huge;
    *status
}

/// This routine returns the image compression algorithm that should be
/// used when writing a FITS image.  The image is divided into tiles, and
/// each tile is compressed and stored in a row of at variable length binary
/// table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ctype`  — image compression type code;
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_get_compression_type(
    fptr: *mut fitsfile,
    ctype: *mut c_int,
    /* allowed values:                                     */
    /* RICE_1, GZIP_1, GZIP_2, PLIO_1, HCOMPRESS_1, BZIP2_1 */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let ctype = ctype.as_mut().expect(NULL_MSG);

        fits_get_compression_type_safe(fptr, ctype, status)
    }
}

/// This routine returns the image compression algorithm that should be
/// used when writing a FITS image.  The image is divided into tiles, and
/// each tile is compressed and stored in a row of at variable length binary
/// table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ctype`  — image compression type code;
/// * `status` — (IO) error status
pub fn fits_get_compression_type_safe(
    fptr: &mut fitsfile,
    ctype: &mut c_int,
    /* allowed values:                                     */
    /* RICE_1, GZIP_1, GZIP_2, PLIO_1, HCOMPRESS_1, BZIP2_1 */
    status: &mut c_int,
) -> c_int {
    *ctype = (fptr.Fptr).request_compress_type;

    if *ctype != RICE_1
        && *ctype != GZIP_1
        && *ctype != GZIP_2
        && *ctype != PLIO_1
        && *ctype != HCOMPRESS_1
        && *ctype != BZIP2_1
        && *ctype != NOCOMPRESS
        && *ctype != 0
    {
        ffpmsg_str("unknown compression algorithm (fits_get_compression_type)");
        *status = DATA_COMPRESSION_ERR;
    }

    *status
}

/// This routine returns the size (dimension) of the image
/// compression  tiles that should be used when writing a FITS
/// image.  The image is divided into tiles, and each tile is compressed
/// and stored in a row of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ndim`   — number of dimensions in the compressed image
/// * `dims`   — size of image compression tile in each dimension
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_get_tile_dim(
    fptr: *mut fitsfile,
    ndim: c_int,
    dims: *mut c_long,
    /* default tile size = (NAXIS1, 1, 1, ...)           */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        if ndim < 0 || ndim > MAX_COMPRESS_DIM as c_int {
            *status = BAD_DIMEN;
            ffpmsg_str("illegal number of tile dimensions (fits_get_tile_dim)");
            return *status;
        }

        let dims = slice::from_raw_parts_mut(dims, ndim as usize);

        fits_get_tile_dim_safe(fptr, ndim, dims, status)
    }
}

/// This routine returns the size (dimension) of the image
/// compression  tiles that should be used when writing a FITS
/// image.  The image is divided into tiles, and each tile is compressed
/// and stored in a row of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ndim`   — number of dimensions in the compressed image
/// * `dims`   — size of image compression tile in each dimension
/// * `status` — (IO) error status
pub fn fits_get_tile_dim_safe(
    fptr: &mut fitsfile,
    ndim: c_int,
    dims: &mut [c_long],
    /* default tile size = (NAXIS1, 1, 1, ...)           */
    status: &mut c_int,
) -> c_int {
    if ndim < 0 || ndim > MAX_COMPRESS_DIM as c_int {
        *status = BAD_DIMEN;
        ffpmsg_str("illegal number of tile dimensions (fits_get_tile_dim)");
        return *status;
    }

    dims[..(ndim as usize)].copy_from_slice(&(fptr.Fptr).request_tilesize[..(ndim as usize)]);

    *status
}

/// Clear the compression parameters recorded for the current HDU.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
///
/// # Safety
///
/// `fptr` and `status` must be non-null and valid.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_unset_compression_param(
    fptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_unset_compression_param_safe(fptr, status)
    }
}

/// Clear the compression parameters recorded for the current HDU.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub fn fits_unset_compression_param_safe(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    (fptr.Fptr).compress_type = 0;
    (fptr.Fptr).quantize_level = 0.0;
    (fptr.Fptr).quantize_method = 0;
    (fptr.Fptr).dither_seed = 0;
    (fptr.Fptr).hcomp_scale = 0.0;

    for ii in 0..MAX_COMPRESS_DIM {
        (fptr.Fptr).tilesize[ii] = 0;
    }

    *status
}

/// Clear the compression settings requested for the next image written.
///
/// These are the `request_*` fields, which are what a `fits_set_*` call sets;
/// they become the HDU's actual parameters when an image is compressed.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
///
/// # Safety
///
/// `fptr` and `status` must be non-null and valid.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_unset_compression_request(
    fptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_unset_compression_request_safe(fptr, status)
    }
}

/// Clear the compression settings requested for the next image written.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub fn fits_unset_compression_request_safe(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    (fptr.Fptr).request_compress_type = 0;
    (fptr.Fptr).request_quantize_level = 0.0;
    (fptr.Fptr).request_quantize_method = 0;
    (fptr.Fptr).request_dither_seed = 0;
    (fptr.Fptr).request_hcomp_scale = 0.0;
    (fptr.Fptr).request_lossy_int_compress = 0;
    (fptr.Fptr).request_huge_hdu = 0;

    for ii in 0..MAX_COMPRESS_DIM {
        (fptr.Fptr).request_tilesize[ii] = 0;
    }

    *status
}

/// Set the preference for various compression options, based
/// on keywords in the input file that
/// provide guidance about how the HDU should be compressed when written
/// to the output file.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_set_compression_pref(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_set_compression_pref_safe(infptr, outfptr, status)
    }
}

/// Set the preference for various compression options, based
/// on keywords in the input file that
/// provide guidance about how the HDU should be compressed when written
/// to the output file.
pub fn fits_set_compression_pref_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut nkeys: c_int = 0;
    let mut naxis: c_int = 0;
    let mut ivalue: c_int = 0;
    let mut comptype: c_int = 0;
    let mut tiledim: [c_long; 6] = [1, 1, 1, 1, 1, 1];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut value: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    let mut datastart: LONGLONG = 0;
    let mut dataend: LONGLONG = 0;

    if *status > 0 {
        return *status;
    }

    /* check the size of the HDU that is to be compressed */
    ffghadll_safe(
        infptr,
        None,
        Some(&mut datastart),
        Some(&mut dataend),
        status,
    );
    if (dataend - datastart) > LONGLONG::from(UINT32_MAX) {
        /* use 64-bit '1Q' variable length columns instead of '1P' columns */
        /* for large files, in case the heap size becomes larger than 2**32 bytes*/
        fits_set_huge_hdu_safe(outfptr, 1, status);
    }

    ffghsp_safe(infptr, Some(&mut nkeys), None, status);

    /* look for a image compression directive keywords (begin with 'FZ') */
    for ii in 2..=nkeys {
        ffgrec_safe(infptr, ii, Some(&mut card), status);

        if strncmp_safe(&card, cs!(c"FZ"), 2) == 0 {
            /* get the keyword value string */
            ffpsvc_safe(&card, &mut value, None, status);

            if strncmp_safe(&card[2..], cs!(c"ALGOR"), 5) == 0 {
                /* set the desired compression algorithm */
                /* allowed values: RICE_1, GZIP_1, GZIP_2, PLIO_1,     */
                /*  HCOMPRESS_1, BZIP2_1, and NOCOMPRESS               */

                if fits_strncasecmp(&value, cs!(c"'RICE_1"), 7) == 0 {
                    comptype = RICE_1;
                } else if fits_strncasecmp(&value, cs!(c"'GZIP_1"), 7) == 0 {
                    comptype = GZIP_1;
                } else if fits_strncasecmp(&value, cs!(c"'GZIP_2"), 7) == 0 {
                    comptype = GZIP_2;
                } else if fits_strncasecmp(&value, cs!(c"'PLIO_1"), 7) == 0 {
                    comptype = PLIO_1;
                } else if fits_strncasecmp(&value, cs!(c"'HCOMPRESS_1"), 12) == 0 {
                    comptype = HCOMPRESS_1;
                } else if fits_strncasecmp(&value, cs!(c"'NONE"), 5) == 0 {
                    comptype = NOCOMPRESS;
                } else {
                    ffpmsg_str("Unknown FZALGOR keyword compression algorithm:");
                    ffpmsg_slice(&value);
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }

                fits_set_compression_type_safe(outfptr, comptype, status);
            } else if strncmp_safe(&card[2..], cs!(c"TILE  "), 6) != 0 {
                if fits_strncasecmp(&value, cs!(c"'row"), 4) == 0 {
                    tiledim[0] = -1;
                } else if fits_strncasecmp(&value, cs!(c"'whole"), 6) == 0 {
                    tiledim[0] = -1;
                    tiledim[1] = -1;
                    tiledim[2] = -1;
                } else {
                    ffdtdm_safe(infptr, &value, 0, 6, &mut naxis, &mut tiledim, status);
                }

                /* set the desired tile size */
                fits_set_tile_dim_safe(outfptr, 6, &tiledim, status);
            } else if strncmp_safe(&card[2..], cs!(c"QVALUE"), 6) == 0 {
                /* set the desired Q quantization value */
                let qvalue: f64 = CStr::from_bytes_until_nul(cast_slice(&value))
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                fits_set_quantize_level_safe(outfptr, qvalue as f32, status);
            } else if strncmp_safe(&card[2..], cs!(c"QMETHD"), 6) == 0 {
                if fits_strncasecmp(&value, cs!(c"'no_dither"), 10) == 0 {
                    ivalue = -1; /* just quantize, with no dithering */
                } else if fits_strncasecmp(&value, cs!(c"'subtractive_dither_1"), 21) == 0 {
                    ivalue = SUBTRACTIVE_DITHER_1 as c_int; /* use subtractive dithering */
                } else if fits_strncasecmp(&value, cs!(c"'subtractive_dither_2"), 21) == 0 {
                    ivalue = SUBTRACTIVE_DITHER_2 as c_int; /* dither, except preserve zero-valued pixels */
                } else {
                    ffpmsg_str("Unknown value for FZQUANT keyword: (set_compression_pref)");
                    ffpmsg_slice(&value);
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }

                fits_set_quantize_method_safe(outfptr, ivalue, status);
            } else if strncmp_safe(&card[2..], cs!(c"DTHRSD"), 6) == 0 {
                if fits_strncasecmp(&value, cs!(c"'checksum"), 9) == 0 {
                    ivalue = -1; /* use checksum of first tile */
                } else if fits_strncasecmp(&value, cs!(c"'clock"), 6) == 0 {
                    ivalue = 0; /* set dithering seed based on system clock */
                } else {
                    /* read integer value */
                    if value[0] == bb(b'\'') {
                        let lvalue: c_long = CStr::from_bytes_until_nul(cast_slice(&value[1..]))
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .parse()
                            .unwrap();
                        ivalue = lvalue as c_int; /* allow for leading quote character */
                    } else {
                        let lvalue: c_long = CStr::from_bytes_until_nul(cast_slice(&value))
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .parse()
                            .unwrap();
                        ivalue = lvalue as c_int;
                    }
                    if ivalue < 1 || ivalue > 10000 {
                        ffpmsg_str("Invalid value for FZDTHRSD keyword: (set_compression_pref)");
                        ffpmsg_slice(&value);
                        (*status = DATA_COMPRESSION_ERR);
                        return *status;
                    }
                }

                /* set the desired dithering */
                fits_set_dither_seed_safe(outfptr, ivalue, status);
            } else if strncmp_safe(&card[2..], cs!(c"I2F"), 3) == 0 {
                /* set whether to convert integers to float then use lossy compression */
                if fits_strcasecmp(&value, cs!(c"t")) == 0 {
                    fits_set_lossy_int_safe(outfptr, 1, status);
                } else if fits_strcasecmp(&value, cs!(c"f")) == 0 {
                    fits_set_lossy_int_safe(outfptr, 0, status);
                } else {
                    ffpmsg_str("Unknown value for FZI2F keyword: (set_compression_pref)");
                    ffpmsg_slice(&value);
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }
            } else if strncmp_safe(&card[2..], cs!(c"HSCALE "), 6) == 0 {
                /* set the desired Hcompress scale value */
                let dvalue: f64 = CStr::from_bytes_until_nul(cast_slice(&value))
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                let hscale: f32 = dvalue as f32;
                fits_set_hcomp_scale_safe(outfptr, hscale, status);
            }
        }
    }
    *status
}

/// ********************************************************************
/// ********************************************************************
/// THIS ROUTINE IS PROVIDED ONLY FOR BACKWARDS COMPATIBILITY;
/// ALL NEW SOFTWARE SHOULD CALL fits_set_quantize_level INSTEAD
/// ********************************************************************
/// ********************************************************************
///
///
/// This routine returns the value of the noice_bits parameter that
/// should be used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.ns
///
/// Feb 2008: code changed to use the more general "quantize level" parameter
/// rather than the "noise bits" parameter.  If quantize level is greater than
/// zero, then the previous noisebits parameter is approximately given by
///
/// noise bits = natural logarithm (quantize level) / natural log (2)
///
/// This result is rounded to the nearest integer.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `noisebits` — noise_bits parameter value
/// * `status`    — (IO) error status
#[deprecated]
#[cfg_attr(not(test), unsafe(no_mangle))]
pub unsafe extern "C" fn fits_get_noise_bits(
    fptr: *mut fitsfile,
    noisebits: *mut c_int,
    /* (default = 4)                    */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let noisebits = noisebits.as_mut().expect(NULL_MSG);

        fits_get_noise_bits_safe(fptr, noisebits, status)
    }
}

/// Get the noise_bits parameter value.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `noisebits` — noise_bits parameter value
/// * `status`    — (IO) error status
pub fn fits_get_noise_bits_safe(
    fptr: &mut fitsfile,
    noisebits: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let qlevel: f64 = f64::from((fptr.Fptr).request_quantize_level);

    if qlevel > 0. && qlevel < 65537. {
        *noisebits = (((qlevel.ln()) / (2.0_f64).ln()) + 0.5) as c_int;
    } else {
        *noisebits = 0;
    }
    *status
}

/// This routine returns the value of the noice_bits parameter that
/// should be used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `qlevel` — quantize level parameter value
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_get_quantize_level(
    fptr: *mut fitsfile,
    qlevel: *mut f32,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let qlevel = qlevel.as_mut().expect(NULL_MSG);

        fits_get_quantize_level_safe(fptr, qlevel, status)
    }
}

/// This routine returns the value of the noice_bits parameter that
/// should be used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `qlevel` — quantize level parameter value
/// * `status` — (IO) error status
pub fn fits_get_quantize_level_safe(
    fptr: &mut fitsfile,
    qlevel: &mut f32,
    status: &mut c_int,
) -> c_int {
    if (fptr.Fptr).request_quantize_level == NO_QUANTIZE {
        *qlevel = 0.0;
    } else {
        *qlevel = (fptr.Fptr).request_quantize_level;
    }

    *status
}

/// This routine returns the value of the dithering offset parameter that
/// is used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `offset` — dithering offset parameter value
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_get_dither_seed(
    fptr: *mut fitsfile,
    offset: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let offset = offset.as_mut().expect(NULL_MSG);

        fits_get_dither_seed_safe(fptr, offset, status)
    }
}

/// This routine returns the value of the dithering offset parameter that
/// is used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `offset` — dithering offset parameter value
/// * `status` — (IO) error status
pub fn fits_get_dither_seed_safe(
    fptr: &mut fitsfile,
    offset: &mut c_int,
    status: &mut c_int,
) -> c_int {
    *offset = (fptr.Fptr).request_dither_seed;
    *status
}

/// This routine returns the value of the noice_bits parameter that
/// should be used when compressing floating point images.  The image is
/// divided into tiles, and each tile is compressed and stored in a row
/// of at variable length binary table column.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `scale`  — Hcompress scale parameter value
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_get_hcomp_scale(
    fptr: *mut fitsfile,
    scale: *mut f32,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let scale = scale.as_mut().expect(NULL_MSG);

        fits_get_hcomp_scale_safe(fptr, scale, status)
    }
}

/// Get the Hcompress scale parameter value.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `scale`  — Hcompress scale parameter value
/// * `status` — (IO) error status
pub fn fits_get_hcomp_scale_safe(
    fptr: &mut fitsfile,
    scale: &mut f32,
    status: &mut c_int,
) -> c_int {
    *scale = (fptr.Fptr).request_hcomp_scale;
    *status
}

#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `smooth` — Hcompress smooth parameter value
/// * `status` — (IO) error status
pub unsafe extern "C" fn fits_get_hcomp_smooth(
    fptr: *mut fitsfile,
    smooth: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let smooth = smooth.as_mut().expect(NULL_MSG);

        fits_get_hcomp_smooth_safe(fptr, smooth, status)
    }
}

/// Get the Hcompress smooth parameter value.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `smooth` — Hcompress smooth parameter value
/// * `status` — (IO) error status
pub fn fits_get_hcomp_smooth_safe(
    fptr: &mut fitsfile,
    smooth: &mut c_int,
    status: &mut c_int,
) -> c_int {
    *smooth = (fptr.Fptr).request_hcomp_smooth;
    *status
}

#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
/// # Parameters
///
/// * `infptr`  — pointer to image to be compressed
/// * `outfptr` — empty HDU for output compressed image
/// * `status`  — (IO) error status
pub unsafe extern "C" fn fits_img_compress(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    status: *mut c_int,
    /*
    This routine initializes the output table, copies all the keywords,
    and  loops through the input image, compressing the data and
    writing the compressed tiles to the output table.

    This is a high level routine that is called by the fpack and funpack
    FITS compression utilities.
    */
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_img_compress_safe(infptr, outfptr, status)
    }
}

/// Compress an image and write to output table.
/// This routine initializes the output table, copies all the keywords,
/// and loops through the input image, compressing the data and
/// writing the compressed tiles to the output table.
///
/// # Parameters
///
/// * `infptr`  — pointer to image to be compressed
/// * `outfptr` — empty HDU for output compressed image
/// * `status`  — (IO) error status
pub fn fits_img_compress_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut bitpix: c_int = 0;
    let mut naxis: c_int = 0;
    let mut naxes: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];

    if *status > 0 {
        return *status;
    }

    /* get datatype and size of input image */
    if ffgipr_safe(
        infptr,
        MAX_COMPRESS_DIM as c_int,
        Some(&mut bitpix),
        Some(&mut naxis),
        Some(&mut naxes),
        status,
    ) > 0
    {
        return *status;
    }

    if naxis < 1 || naxis > MAX_COMPRESS_DIM as c_int {
        ffpmsg_str("Image cannot be compressed: NAXIS out of range");
        *status = BAD_NAXIS;
        return *status;
    }

    /* create a new empty HDU in the output file now, before setting the */
    /* compression preferences.  This HDU will become a binary table that */
    /* contains the compressed image.  If necessary, create a dummy primary */
    /* array, which much precede the binary table extension. */

    ffcrhd_safe(outfptr, status); /* this does nothing if the output file is empty */

    if (outfptr.Fptr).curhdu == 0 {
        /* have to create dummy primary array */

        ffcrim_safe(outfptr, 16, 0, &[], status);
        ffcrhd_safe(outfptr, status);
    } else {
        /* unset any compress parameter preferences that may have been
        set when closing the previous HDU in the output file */
        fits_unset_compression_param_safe(outfptr, status);
    }

    /* set any compress parameter preferences as given in the input file */
    fits_set_compression_pref_safe(infptr, outfptr, status);

    /* special case: the quantization level is not given by a keyword in  */
    /* the HDU header, so we have to explicitly copy the requested value */
    /* to the actual value */
    /* do this in imcomp_get_compressed_image_par, instead
    if ( (outfptr.Fptr).request_quantize_level != 0.0)
    (outfptr.Fptr).quantize_level =
    (outfptr.Fptr).request_quantize_level;
    */
    /* if requested, treat integer images same as a float image. */
    /* Then the pixels will be quantized (lossy algorithm) to achieve */
    /* higher amounts of compression than with lossless algorithms */

    if (outfptr.Fptr).request_lossy_int_compress != 0 && bitpix > 0 {
        bitpix = FLOAT_IMG; /* compress integer images as if float */
    }

    /* initialize output table */
    if imcomp_init_table(outfptr, bitpix, naxis, &naxes, false, status) > 0 {
        return *status;
    }

    /* Copy the image header keywords to the table header. */
    if imcomp_copy_img2comp(infptr, outfptr, status) > 0 {
        return *status;
    }

    /* turn off any intensity scaling (defined by BSCALE and BZERO */
    /* keywords) so that unscaled values will be read by CFITSIO */
    /* (except if quantizing an int image, same as a float image) */
    if (outfptr.Fptr).request_lossy_int_compress == 0 && bitpix > 0 {
        ffpscl_safe(infptr, 1.0, 0.0, status);
    }

    /* force a rescan of the output file keywords, so that */
    /* the compression parameters will be copied to the internal */
    /* fitsfile structure used by CFITSIO */
    ffrdef_safe(outfptr, status);

    /* turn off any intensity scaling (defined by BSCALE and BZERO */
    /* keywords) so that unscaled values will be written by CFITSIO */
    /* (except if quantizing an int image, same as a float image) */
    if (outfptr.Fptr).request_lossy_int_compress == 0 && bitpix > 0 {
        ffpscl_safe(outfptr, 1.0, 0.0, status);
    }

    /* Read each image tile, compress, and write to a table row. */
    imcomp_compress_image(infptr, outfptr, status);

    /* force another rescan of the output file keywords, to */
    /* update PCOUNT and TFORMn = '1PB(iii)' keyword values. */
    ffrdef_safe(outfptr, status);

    /* unset any previously set compress parameter preferences */
    fits_unset_compression_request_safe(outfptr, status);

    /*
    fits_get_case(&c1, &c2, &c3);
    printf("c1, c2, c3 = %d, %d, %d\n", c1, c2, c3);
    */

    *status
}

/// Create a BINTABLE extension for the output compressed image.
///
/// # Parameters
///
/// * `writebitpix` — write the ZBITPIX, ZNAXIS, and ZNAXES keyword?
pub(crate) fn imcomp_init_table(
    outfptr: &mut fitsfile,
    inbitpix: c_int,
    naxis: c_int,
    naxes: &[c_long],
    writebitpix: bool,
    status: &mut c_int,
) -> c_int {
    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut zcmptype: [c_char; 12] = [0; 12];
    let mut remain: c_long = 0;
    let mut ndiv: c_int = 0;
    let mut addToDim: c_int = 0;
    let mut ncols: c_int = 0;
    let mut bitpix: c_int = 0;
    let mut nrows: c_long = 0;
    let ttype: [&[c_char]; 3] = [cs!(c"COMPRESSED_DATA"), cs!(c"ZSCALE"), cs!(c"ZZERO")];

    let mut tform: [[c_char; 4]; 3] = [[0; 4]; 3];
    let mut tf0: [c_char; 4] = [0; 4];
    let mut tf1: [c_char; 4] = [0; 4];
    let mut tf2: [c_char; 4] = [0; 4];
    let tunit: [&[c_char]; 3] = [&[0], &[0], &[0]];

    let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut actual_tilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* Actual size to use for tiles */
    let mut is_primary: bool = false; /* Is this attempting to write to the primary? */
    let mut nQualifyDims: c_int = 0; /* For Hcompress, number of image dimensions with required pixels. */
    let mut noHigherDims = true; /* Set to true if all tile dims other than x are size 1. */
    let mut firstDim: usize = usize::MAX; /* Indices of first and second tilesdimensions with width > 1 */
    let mut secondDim: usize = usize::MAX; // Use max value to indicate not set

    if *status > 0 {
        return *status;
    }

    /* check for special case of losslessly compressing floating point */
    /* images.  Only compression algorithm that supports this is GZIP */
    if (inbitpix < 0)
        && ((outfptr.Fptr).request_quantize_level == NO_QUANTIZE)
        && ((outfptr.Fptr).request_compress_type != GZIP_1)
        && ((outfptr.Fptr).request_compress_type != GZIP_2)
    {
        ffpmsg_cstr(
            c"Lossless compression of floating point images must use GZIP (imcomp_init_table)",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    /* set default compression parameter values, if undefined */

    if (outfptr.Fptr).request_compress_type == 0 {
        /* use RICE_1 by default */
        (outfptr.Fptr).request_compress_type = RICE_1;
    }

    if inbitpix < 0 && (outfptr.Fptr).request_quantize_level != NO_QUANTIZE {
        /* set defaults for quantizing floating point images */
        if (outfptr.Fptr).request_quantize_method == 0 {
            /* set default dithering method */
            (outfptr.Fptr).request_quantize_method = SUBTRACTIVE_DITHER_1;
        }

        if (outfptr.Fptr).request_quantize_level == 0.0 {
            if (outfptr.Fptr).request_quantize_method == NO_DITHER {
                /* must use finer quantization if no dithering is done */
                (outfptr.Fptr).request_quantize_level = 16.0;
            } else {
                (outfptr.Fptr).request_quantize_level = 4.0;
            }
        }
    }

    /* special case: the quantization level is not given by a keyword in  */
    /* the HDU header, so we have to explicitly copy the requested value */
    /* to the actual value */
    /* do this in imcomp_get_compressed_image_par, instead
    if ( (outfptr.Fptr).request_quantize_level != 0.0)
    (outfptr.Fptr).quantize_level =
    (outfptr.Fptr).request_quantize_level;
    */
    /* test for the 2 special cases that represent unsigned integers */
    if inbitpix == USHORT_IMG {
        bitpix = SHORT_IMG;
    } else if inbitpix == ULONG_IMG {
        bitpix = LONG_IMG;
    } else if inbitpix == SBYTE_IMG {
        bitpix = BYTE_IMG;
    } else {
        bitpix = inbitpix;
    }

    /* reset default tile dimensions too if required */
    actual_tilesize.copy_from_slice(&outfptr.Fptr.request_tilesize);

    if (outfptr.Fptr).request_compress_type == HCOMPRESS_1 {
        /* Tiles must ultimately have 2 (and only 2) dimensions, each with
        at least 4 pixels. First catch the case where the image
        itself won't allow this. */
        if naxis < 2 {
            ffpmsg_str("Hcompress cannot be used with 1-dimensional images (imcomp_init_table)");
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }
        for ii in 0..(naxis as usize) {
            if naxes[ii] >= 4 {
                nQualifyDims += 1;
            }
        }
        if nQualifyDims < 2 {
            ffpmsg_str("Hcompress minimum image dimension is 4 pixels (imcomp_init_table)");
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }

        /* Handle 2 special cases for backwards compatibility.
        1) If both X and Y tile dims are set to full size, ignore
        any other requested dimensions and just set their sizes to 1.
        2) If X is full size and all the rest are size 1, attempt to
        find a reasonable size for Y. All other 1-D tile specifications
        will be rejected. */
        for ii in 1..(naxis as usize) {
            if actual_tilesize[ii] != 0 && actual_tilesize[ii] != 1 {
                noHigherDims = false;
                break;
            }
        }

        if (actual_tilesize[0] <= 0) && (actual_tilesize[1] == -1) {
            /* compress the whole image as a single tile */
            actual_tilesize[0] = naxes[0];
            actual_tilesize[1] = naxes[1];

            for ii in 2..(naxis as usize) {
                /* set all higher tile dimensions = 1 */
                actual_tilesize[ii] = 1;
            }
        } else if (actual_tilesize[0] <= 0) && noHigherDims {
            /*
            The Hcompress algorithm is inherently 2D in nature, so the row by
            row tiling that is used for other compression algorithms is not
            appropriate. If the image has less than 30 rows, then the entire
            image will be compressed as a single tile.  Otherwise the tiles
            will consist of 16 rows of the image. This keeps the tiles to a
            reasonable size, and it also includes enough rows to allow good
            compression efficiency.  If the last tile of the image happens to
            contain less than 4 rows, then find another tile size with between
            14 and 30 rows (preferably even), so that the last tile has at
            least 4 rows
            */

            /* 1st tile dimension is the row length of the image */
            actual_tilesize[0] = naxes[0];

            if naxes[1] <= 30 {
                /* use whole image if it is small */
                actual_tilesize[1] = naxes[1];
            } else {
                /* look for another good tile dimension */
                if naxes[1] % 16 == 0 || naxes[1] % 16 > 3 {
                    actual_tilesize[1] = 16;
                } else if naxes[1] % 24 == 0 || naxes[1] % 24 > 3 {
                    actual_tilesize[1] = 24;
                } else if naxes[1] % 20 == 0 || naxes[1] % 20 > 3 {
                    actual_tilesize[1] = 20;
                } else if naxes[1] % 30 == 0 || naxes[1] % 30 > 3 {
                    actual_tilesize[1] = 30;
                } else if naxes[1] % 28 == 0 || naxes[1] % 28 > 3 {
                    actual_tilesize[1] = 28;
                } else if naxes[1] % 26 == 0 || naxes[1] % 26 > 3 {
                    actual_tilesize[1] = 26;
                } else if naxes[1] % 22 == 0 || naxes[1] % 22 > 3 {
                    actual_tilesize[1] = 22;
                } else if naxes[1] % 18 == 0 || naxes[1] % 18 > 3 {
                    actual_tilesize[1] = 18;
                } else if naxes[1] % 14 == 0 || naxes[1] % 14 > 3 {
                    actual_tilesize[1] = 14;
                } else {
                    actual_tilesize[1] = 17;
                }
            }
        } else {
            if actual_tilesize[0] <= 0 {
                actual_tilesize[0] = naxes[0];
            }
            for ii in 1..(naxis as usize) {
                if actual_tilesize[ii] < 0 {
                    actual_tilesize[ii] = naxes[ii];
                } else if actual_tilesize[ii] == 0 {
                    actual_tilesize[ii] = 1;
                }
            }
        }

        for ii in 0..(naxis as usize) {
            if actual_tilesize[ii] > 1 {
                if firstDim == usize::MAX {
                    firstDim = ii;
                } else if secondDim == usize::MAX {
                    secondDim = ii;
                } else {
                    ffpmsg_str("Hcompress tiles can only have 2 dimensions (imcomp_init_table)");
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }
            }
        }
        if firstDim == usize::MAX || secondDim == usize::MAX {
            ffpmsg_str("Hcompress tiles must have 2 dimensions (imcomp_init_table)");
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }

        if actual_tilesize[firstDim] < 4 || actual_tilesize[secondDim] < 4 {
            ffpmsg_str("Hcompress minimum tile dimension is 4 pixels (imcomp_init_table)");
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }

        /* check if requested tile size causes the last tile to to have less than 4 pixels */
        remain = naxes[firstDim] % (actual_tilesize[firstDim]); /* 1st dimension */
        if remain > 0 && remain < 4 {
            ndiv = (naxes[firstDim] / actual_tilesize[firstDim]) as c_int; /* integer truncation is intentional */
            addToDim = ((remain as f64) / f64::from(ndiv)).ceil() as c_int;
            (actual_tilesize[firstDim]) += c_long::from(addToDim); /* increase tile size */

            remain = naxes[firstDim] % (actual_tilesize[firstDim]);
            if remain > 0 && remain < 4 {
                ffpmsg_cstr(
                    c"Last tile along 1st dimension has less than 4 pixels (imcomp_init_table)",
                );
                *status = DATA_COMPRESSION_ERR;
                return *status;
            }
        }

        remain = naxes[secondDim] % (actual_tilesize[secondDim]); /* 2nd dimension */
        if remain > 0 && remain < 4 {
            ndiv = (naxes[secondDim] / actual_tilesize[secondDim]) as c_int; /* integer truncation is intentional */
            addToDim = ((remain as f64) / f64::from(ndiv)).ceil() as c_int;
            (actual_tilesize[secondDim]) += c_long::from(addToDim); /* increase tile size */

            remain = naxes[secondDim] % (actual_tilesize[secondDim]);
            if remain > 0 && remain < 4 {
                ffpmsg_cstr(
                    c"Last tile along 2nd dimension has less than 4 pixels (imcomp_init_table)",
                );
                *status = DATA_COMPRESSION_ERR;
                return *status;
            }
        }
    } /* end, if HCOMPRESS_1 */

    for ii in 0..(naxis as usize) {
        if ii == 0 {
            /* first axis is different */
            if actual_tilesize[ii] <= 0 {
                actual_tilesize[ii] = naxes[ii];
            }
        } else if actual_tilesize[ii] < 0 {
            actual_tilesize[ii] = naxes[ii]; /* negative value maean use whole length */
        } else if actual_tilesize[ii] == 0 {
            actual_tilesize[ii] = 1; /* zero value means use default value = 1 */
        }
    }

    /* ---- set up array of TFORM strings -------------------------------*/
    if (outfptr.Fptr).request_huge_hdu != 0 {
        strcpy_safe(&mut tf0, cs!(c"1QB"));
    } else {
        strcpy_safe(&mut tf0, cs!(c"1PB"));
    }
    strcpy_safe(&mut tf1, cs!(c"1D"));
    strcpy_safe(&mut tf2, cs!(c"1D"));

    tform[0] = tf0;
    tform[1] = tf1;
    tform[2] = tf2;

    /* calculate number of rows in output table */
    nrows = 1;
    for ii in 0..(naxis as usize) {
        nrows *= (naxes[ii] - 1) / (actual_tilesize[ii]) + 1;
    }

    /* determine the default  number of columns in the output table */
    if bitpix < 0 && (outfptr.Fptr).request_quantize_level != NO_QUANTIZE {
        ncols = 3; /* quantized and scaled floating point image */
    } else {
        ncols = 1; /* default table has just one 'COMPRESSED_DATA' column */
    }

    if (outfptr.Fptr).request_compress_type == RICE_1 {
        strcpy_safe(&mut zcmptype, cs!(c"RICE_1"));
    } else if (outfptr.Fptr).request_compress_type == GZIP_1 {
        strcpy_safe(&mut zcmptype, cs!(c"GZIP_1"));
    } else if (outfptr.Fptr).request_compress_type == GZIP_2 {
        strcpy_safe(&mut zcmptype, cs!(c"GZIP_2"));
    } else if (outfptr.Fptr).request_compress_type == BZIP2_1 {
        strcpy_safe(&mut zcmptype, cs!(c"BZIP2_1"));
    } else if (outfptr.Fptr).request_compress_type == PLIO_1 {
        strcpy_safe(&mut zcmptype, cs!(c"PLIO_1"));
        /* the PLIO compression algorithm outputs short integers, not bytes */
        if (outfptr.Fptr).request_huge_hdu != 0 {
            strcpy_safe(&mut tform[0], cs!(c"1QI"));
        } else {
            strcpy_safe(&mut tform[0], cs!(c"1PI"));
        }
    } else if (outfptr.Fptr).request_compress_type == HCOMPRESS_1 {
        strcpy_safe(&mut zcmptype, cs!(c"HCOMPRESS_1"));
    } else if (outfptr.Fptr).request_compress_type == NOCOMPRESS {
        strcpy_safe(&mut zcmptype, cs!(c"NOCOMPRESS"));
    } else {
        ffpmsg_str("unknown compression type (imcomp_init_table)");
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    /* If attempting to write compressed image to primary, the
    call to ffcrtb will increment Fptr->curhdu to 1.  Therefore
    we need to test now for setting is_primary */
    is_primary = outfptr.Fptr.curhdu == 0;

    /* create the bintable extension to contain the compressed image */

    let ttype_vec = ttype.iter().map(|&f| Some(f)).collect::<Vec<_>>();
    let tform_vec = tform.iter().map(|f| f.as_slice()).collect::<Vec<_>>();
    let tunit_vec = tunit.iter().map(|&f| Some(f)).collect::<Vec<_>>();
    ffcrtb_safe(
        outfptr,
        BINARY_TBL,
        nrows as LONGLONG,
        ncols,
        &ttype_vec,
        &tform_vec,
        Some(&(tunit_vec)),
        None,
        status,
    );

    /* Add standard header keywords. */
    ffpkyl_safe(
        outfptr,
        cs!(c"ZIMAGE"),
        1,
        Some(cs!(c"extension contains compressed image")),
        status,
    );

    if writebitpix {
        /*  write the keywords defining the datatype and dimensions of */
        /*  the uncompressed image.  If not, these keywords will be */
        /*  copied later from the input uncompressed image  */

        if is_primary {
            ffpkyl_safe(
                outfptr,
                cs!(c"ZSIMPLE"),
                1,
                Some(cs!(c"file does conform to FITS standard")),
                status,
            );
        }
        ffpkyj_safe(
            outfptr,
            cs!(c"ZBITPIX"),
            LONGLONG::from(bitpix),
            Some(cs!(c"data type of original image")),
            status,
        );
        ffpkyj_safe(
            outfptr,
            cs!(c"ZNAXIS"),
            LONGLONG::from(naxis),
            Some(cs!(c"dimension of original image")),
            status,
        );

        for ii in 0..(naxis as usize) {
            int_snprintf!(&mut keyname, FLEN_KEYWORD, "ZNAXIS{}", ii + 1);
            ffpkyj_safe(
                outfptr,
                &keyname,
                naxes[ii] as LONGLONG,
                Some(cs!(c"length of original image axis")),
                status,
            );
        }
    }

    for ii in 0..(naxis as usize) {
        int_snprintf!(&mut keyname, FLEN_KEYWORD, "ZTILE{}", ii + 1);
        ffpkyj_safe(
            outfptr,
            &keyname,
            actual_tilesize[ii] as LONGLONG,
            Some(cs!(c"size of tiles to be compressed")),
            status,
        );
    }

    if bitpix < 0 {
        if (outfptr.Fptr).request_quantize_level == NO_QUANTIZE {
            ffpkys_safe(
                outfptr,
                cs!(c"ZQUANTIZ"),
                cs!(c"NONE"),
                Some(cs!(c"Lossless compression without quantization")),
                status,
            );
        } else {
            /* Unless dithering has been specifically turned off by setting */
            /* request_quantize_method = -1, use dithering by default */
            /* when quantizing floating point images. */

            if (outfptr.Fptr).request_quantize_method == 0 {
                (outfptr.Fptr).request_quantize_method = SUBTRACTIVE_DITHER_1;
            }

            /* HCompress must not use SUBTRACTIVE_DITHER_2. If user is
            requesting this, assign SUBTRACTIVE_DITHER_1 instead. */
            if (outfptr.Fptr).request_quantize_method == SUBTRACTIVE_DITHER_2
                && (strcmp_safe(&zcmptype, cs!(c"HCOMPRESS_1"))) == 0
            {
                (outfptr.Fptr).request_quantize_method = SUBTRACTIVE_DITHER_1;
                eprintln!(
                    "Warning: CFITSIO does not allow subtractive_dither_2 when using Hcompress algorithm.\nWill use subtractive_dither_1 instead."
                );
            }

            if (outfptr.Fptr).request_quantize_method == SUBTRACTIVE_DITHER_1 {
                ffpkys_safe(
                    outfptr,
                    cs!(c"ZQUANTIZ"),
                    cs!(c"SUBTRACTIVE_DITHER_1"),
                    Some(cs!(c"Pixel Quantization Algorithm")),
                    status,
                );

                /* also write the associated ZDITHER0 keyword with a default  value */
                /* which may get updated later. */
                fits_write_key_lng(
                    outfptr,
                    cs!(c"ZDITHER0"),
                    LONGLONG::from((outfptr.Fptr).request_dither_seed),
                    Some(cs!(c"dithering offset when quantizing floats")),
                    status,
                );
            } else if (outfptr.Fptr).request_quantize_method == SUBTRACTIVE_DITHER_2 {
                ffpkys_safe(
                    outfptr,
                    cs!(c"ZQUANTIZ"),
                    cs!(c"SUBTRACTIVE_DITHER_2"),
                    Some(cs!(c"Pixel Quantization Algorithm")),
                    status,
                );

                /* also write the associated ZDITHER0 keyword with a default  value */
                /* which may get updated later. */
                fits_write_key_lng(
                    outfptr,
                    cs!(c"ZDITHER0"),
                    LONGLONG::from((outfptr.Fptr).request_dither_seed),
                    Some(cs!(c"dithering offset when quantizing floats")),
                    status,
                );
            } else if (outfptr.Fptr).request_quantize_method == NO_DITHER {
                ffpkys_safe(
                    outfptr,
                    cs!(c"ZQUANTIZ"),
                    cs!(c"NO_DITHER"),
                    Some(cs!(c"No dithering during quantization")),
                    status,
                );
            }
        }
    }

    ffpkys_safe(
        outfptr,
        cs!(c"ZCMPTYPE"),
        &zcmptype,
        Some(cs!(c"compression algorithm")),
        status,
    );

    /* write any algorithm-specific keywords */
    if (outfptr.Fptr).request_compress_type == RICE_1 {
        ffpkys_safe(
            outfptr,
            cs!(c"ZNAME1"),
            cs!(c"BLOCKSIZE"),
            Some(cs!(c"compression block size")),
            status,
        );

        /* for now at least, the block size is always 32 */
        ffpkyj_safe(
            outfptr,
            cs!(c"ZVAL1"),
            32,
            Some(cs!(c"pixels per block")),
            status,
        );

        ffpkys_safe(
            outfptr,
            cs!(c"ZNAME2"),
            cs!(c"BYTEPIX"),
            Some(cs!(c"bytes per pixel (1, 2, 4, or 8)")),
            status,
        );

        if bitpix == BYTE_IMG {
            ffpkyj_safe(
                outfptr,
                cs!(c"ZVAL2"),
                1,
                Some(cs!(c"bytes per pixel (1, 2, 4, or 8)")),
                status,
            );
        } else if bitpix == SHORT_IMG {
            ffpkyj_safe(
                outfptr,
                cs!(c"ZVAL2"),
                2,
                Some(cs!(c"bytes per pixel (1, 2, 4, or 8)")),
                status,
            );
        } else {
            ffpkyj_safe(
                outfptr,
                cs!(c"ZVAL2"),
                4,
                Some(cs!(c"bytes per pixel (1, 2, 4, or 8)")),
                status,
            );
        }
    } else if (outfptr.Fptr).request_compress_type == HCOMPRESS_1 {
        ffpkys_safe(
            outfptr,
            cs!(c"ZNAME1"),
            cs!(c"SCALE"),
            Some(cs!(c"HCOMPRESS scale factor")),
            status,
        );
        ffpkye_safe(
            outfptr,
            cs!(c"ZVAL1"),
            (outfptr.Fptr).request_hcomp_scale,
            7,
            Some(cs!(c"HCOMPRESS scale factor")),
            status,
        );

        ffpkys_safe(
            outfptr,
            cs!(c"ZNAME2"),
            cs!(c"SMOOTH"),
            Some(cs!(c"HCOMPRESS smooth option")),
            status,
        );
        ffpkyj_safe(
            outfptr,
            cs!(c"ZVAL2"),
            LONGLONG::from((outfptr.Fptr).request_hcomp_smooth),
            Some(cs!(c"HCOMPRESS smooth option")),
            status,
        );
    }

    /* Write the BSCALE and BZERO keywords, if an unsigned integer image */
    if inbitpix == USHORT_IMG {
        strcpy_safe(
            &mut comm,
            cs!(c"offset data range to that of unsigned short"),
        );
        ffpkyg_safe(outfptr, cs!(c"BZERO"), 32768., 0, Some(&comm), status);
        strcpy_safe(&mut comm, cs!(c"default scaling factor"));
        ffpkyg_safe(outfptr, cs!(c"BSCALE"), 1.0, 0, Some(&comm), status);
    } else if inbitpix == SBYTE_IMG {
        strcpy_safe(&mut comm, cs!(c"offset data range to that of signed byte"));
        ffpkyg_safe(outfptr, cs!(c"BZERO"), -128., 0, Some(&comm), status);
        strcpy_safe(&mut comm, cs!(c"default scaling factor"));
        ffpkyg_safe(outfptr, cs!(c"BSCALE"), 1.0, 0, Some(&comm), status);
    } else if inbitpix == ULONG_IMG {
        strcpy_safe(
            &mut comm,
            cs!(c"offset data range to that of unsigned long"),
        );
        ffpkyg_safe(outfptr, cs!(c"BZERO"), 2147483648., 0, Some(&comm), status);
        strcpy_safe(&mut comm, cs!(c"default scaling factor"));
        ffpkyg_safe(outfptr, cs!(c"BSCALE"), 1.0, 0, Some(&comm), status);
    }

    *status
}

/// This function returns the maximum number of bytes in a compressed
/// image line.
///
///  nx = maximum number of pixels in a tile
///  blocksize is only relevant for RICE compression
fn imcomp_calc_max_elem(comptype: c_int, nx: c_int, zbitpix: c_int, blocksize: c_int) -> c_int {
    if comptype == RICE_1 {
        if zbitpix == 16 {
            (mem::size_of::<c_short>() as c_int) * nx + nx / blocksize + 2 + 4
        } else {
            (mem::size_of::<f32>() as c_int) * nx + nx / blocksize + 2 + 4
        }
    } else if (comptype == GZIP_1) || (comptype == GZIP_2) {
        /* gzip usually compressed by at least a factor of 2 for I*4 images */
        /* and somewhat less for I*2 images */
        /* If this size turns out to be too small, then the gzip */
        /* compression routine will allocate more space as required */
        /* to be on the safe size, allocate buffer same size as input */

        if zbitpix == 16 {
            nx * 2
        } else if zbitpix == 8 {
            nx
        } else {
            nx * 4
        }
    } else if comptype == BZIP2_1 {
        /* To guarantee that the compressed data will fit, allocate an output
        buffer of size 1% larger than the uncompressed data, plus 600 bytes
        */

        (f64::from(nx) * 1.01 * f64::from(zbitpix) / 8.0 + 601.0) as c_int
    } else if comptype == HCOMPRESS_1 {
        /* Imperical evidence suggests in the worst case,
        the compressed stream could be up to 10% larger than the original
        image.  Add 26 byte overhead, only significant for very small tiles

        Possible improvement: may need to allow a larger size for 32-bit images
        */

        if zbitpix == 16 || zbitpix == 8 {
            (f64::from(nx) * 2.2 + 26.0) as c_int /* will be compressing 16-bit int array */
        } else {
            (f64::from(nx) * 4.4 + 26.0) as c_int /* will be compressing 32-bit int array */
        }
    } else if comptype == PLIO_1 {
        /* DEVIATION: the C has no PLIO case and falls through to two words per
        pixel, where the worst case is three -- heasarc/cfitsio#136. */
        (pl_p2li_max_len(nx as usize) * mem::size_of::<c_short>()) as c_int
    } else {
        nx * mem::size_of::<c_int>() as c_int
    }
}

/// This routine does the following:
/// - reads an image one tile at a time
/// - if it is a float or double image, then it tries to quantize the pixels
///   into scaled integers.
/// - it then compressess the integer pixels, or if the it was not
///   possible to quantize the floating point pixels, then it losslessly
///   compresses them with gzip
/// - writes the compressed byte stream to the output FITS file
fn imcomp_compress_image(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut tiledata: Vec<f64> = Vec::new();
    let mut anynul: c_int = 0;
    let mut gotnulls = false;
    let mut datatype: c_int = 0;
    let mut row: c_long = 0;
    let mut naxis: c_int = 0;
    let dummy: f64 = 0.0;
    let dblnull: f64 = DOUBLENULLVALUE;
    let fltnull: f32 = FLOATNULLVALUE;
    let mut maxtilelen: usize = 0;
    let mut tilelen: c_long = 0;
    let incre: [c_long; 6] = [1; 6];
    let mut naxes: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut fpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut lpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tile: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];

    let mut trowsize: c_long = 0;
    let mut ntrows: c_long = 0;
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    if *status > 0 {
        return *status;
    }

    maxtilelen = (outfptr.Fptr).maxtilelen as usize;

    /*
    Allocate buffer to hold 1 tile of data; size depends on which compression
    algorithm is used:

    Rice and GZIP will compress byte, short, or int arrays without conversion.
    PLIO requires 4-byte int values, so byte and short arrays must be
    converted to int. HCompress internally converts byte or short values to
    ints, and converts int values to 8-byte longlong integers.
    */

    let mut allocation_size = 0;

    if (outfptr.Fptr).zbitpix == FLOAT_IMG {
        datatype = TFLOAT;

        if (outfptr.Fptr).compress_type == HCOMPRESS_1 {
            /* need twice as much scratch space (8 bytes per pixel) */
            allocation_size = maxtilelen * 2 * mem::size_of::<f32>();
        } else {
            allocation_size = maxtilelen * mem::size_of::<f32>();
        }
    } else if (outfptr.Fptr).zbitpix == DOUBLE_IMG {
        datatype = TDOUBLE;
        allocation_size = maxtilelen * mem::size_of::<f64>();
    } else if (outfptr.Fptr).zbitpix == SHORT_IMG {
        datatype = TSHORT;
        if (outfptr.Fptr).compress_type == RICE_1
            || (outfptr.Fptr).compress_type == GZIP_1
            || (outfptr.Fptr).compress_type == GZIP_2
            || (outfptr.Fptr).compress_type == BZIP2_1
            || (outfptr.Fptr).compress_type == NOCOMPRESS
        {
            /* only need  buffer of I*2 pixels for gzip, bzip2, and Rice */

            allocation_size = maxtilelen * mem::size_of::<c_short>();
        } else {
            /*  need  buffer of I*4 pixels for Hcompress and PLIO */
            allocation_size = maxtilelen * mem::size_of::<c_int>();
        }
    } else if (outfptr.Fptr).zbitpix == BYTE_IMG {
        datatype = TBYTE;
        if (outfptr.Fptr).compress_type == RICE_1
            || (outfptr.Fptr).compress_type == BZIP2_1
            || (outfptr.Fptr).compress_type == GZIP_1
            || (outfptr.Fptr).compress_type == GZIP_2
        {
            /* only need  buffer of I*1 pixels for gzip, bzip2, and Rice */

            allocation_size = maxtilelen;
        } else {
            /*  need  buffer of I*4 pixels for Hcompress and PLIO */
            allocation_size = maxtilelen * mem::size_of::<c_int>();
        }
    } else if (outfptr.Fptr).zbitpix == LONG_IMG {
        datatype = TINT;
        if (outfptr.Fptr).compress_type == HCOMPRESS_1 {
            /* need twice as much scratch space (8 bytes per pixel) */
            allocation_size = maxtilelen * 2 * mem::size_of::<c_int>();
        } else {
            /* only need  buffer of I*4 pixels for gzip, bzip2,  Rice, and PLIO */
            allocation_size = maxtilelen * mem::size_of::<c_int>();
        }
    } else {
        ffpmsg_str("Bad image datatype. (imcomp_compress_image)");
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    if tiledata.try_reserve_exact(allocation_size).is_err() {
        ffpmsg_str("Out of memory. (imcomp_compress_image)");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        tiledata.resize(allocation_size, 0.0);
    }

    /*  calculate size of tile in each dimension */
    naxis = (outfptr.Fptr).zndim;
    for ii in 0..MAX_COMPRESS_DIM {
        if ii < naxis as usize {
            naxes[ii] = (outfptr.Fptr).znaxis[ii];
            tilesize[ii] = (outfptr.Fptr).tilesize[ii];
        } else {
            naxes[ii] = 1;
            tilesize[ii] = 1;
        }
    }
    row = 1;

    /* set up big loop over up to 6 dimensions */
    for i5 in (1..=naxes[5]).step_by(tilesize[5] as usize) {
        fpixel[5] = i5;
        lpixel[5] = cmp::min(i5 + tilesize[5] - 1, naxes[5]);
        tile[5] = lpixel[5] - fpixel[5] + 1;
        for i4 in (1..=naxes[4]).step_by(tilesize[4] as usize) {
            fpixel[4] = i4;
            lpixel[4] = cmp::min(i4 + tilesize[4] - 1, naxes[4]);
            tile[4] = lpixel[4] - fpixel[4] + 1;
            for i3 in (1..=naxes[3]).step_by(tilesize[3] as usize) {
                fpixel[3] = i3;
                lpixel[3] = cmp::min(i3 + tilesize[3] - 1, naxes[3]);
                tile[3] = lpixel[3] - fpixel[3] + 1;
                for i2 in (1..=naxes[2]).step_by(tilesize[2] as usize) {
                    fpixel[2] = i2;
                    lpixel[2] = cmp::min(i2 + tilesize[2] - 1, naxes[2]);
                    tile[2] = lpixel[2] - fpixel[2] + 1;
                    for i1 in (1..=naxes[1]).step_by(tilesize[1] as usize) {
                        fpixel[1] = i1;
                        lpixel[1] = cmp::min(i1 + tilesize[1] - 1, naxes[1]);
                        tile[1] = lpixel[1] - fpixel[1] + 1;
                        for i0 in (1..=naxes[0]).step_by(tilesize[0] as usize) {
                            fpixel[0] = i0;
                            lpixel[0] = cmp::min(i0 + tilesize[0] - 1, naxes[0]);
                            tile[0] = lpixel[0] - fpixel[0] + 1;

                            /* number of pixels in this tile */
                            tilelen = tile[0];
                            for ii in 1..(naxis as usize) {
                                tilelen *= tile[ii];
                            }

                            /* read next tile of data from image */
                            anynul = 0;
                            if datatype == TFLOAT {
                                ffgsve_safe(
                                    infptr,
                                    1,
                                    naxis,
                                    &naxes,
                                    &fpixel,
                                    &lpixel,
                                    &incre,
                                    FLOATNULLVALUE,
                                    cast_slice_mut(&mut tiledata),
                                    Some(&mut anynul),
                                    status,
                                );
                            } else if datatype == TDOUBLE {
                                ffgsvd_safe(
                                    infptr,
                                    1,
                                    naxis,
                                    &naxes,
                                    &fpixel,
                                    &lpixel,
                                    &incre,
                                    DOUBLENULLVALUE,
                                    cast_slice_mut(&mut tiledata),
                                    Some(&mut anynul),
                                    status,
                                );
                            } else if datatype == TINT {
                                ffgsvk_safe(
                                    infptr,
                                    1,
                                    naxis,
                                    &naxes,
                                    &fpixel,
                                    &lpixel,
                                    &incre,
                                    0,
                                    cast_slice_mut(&mut tiledata),
                                    Some(&mut anynul),
                                    status,
                                );
                            } else if datatype == TSHORT {
                                ffgsvi_safe(
                                    infptr,
                                    1,
                                    naxis,
                                    &naxes,
                                    &fpixel,
                                    &lpixel,
                                    &incre,
                                    0,
                                    cast_slice_mut(&mut tiledata),
                                    Some(&mut anynul),
                                    status,
                                );
                            } else if datatype == TBYTE {
                                ffgsvb_safe(
                                    infptr,
                                    1,
                                    naxis,
                                    &naxes,
                                    &fpixel,
                                    &lpixel,
                                    &incre,
                                    0,
                                    cast_slice_mut(&mut tiledata),
                                    Some(&mut anynul),
                                    status,
                                );
                            } else {
                                ffpmsg_str("Error bad datatype of image tile to compress");
                                return *status;
                            }

                            /* now compress the tile, and write to row of binary table */
                            /*   NOTE: we don't have to worry about the presence
                               of null values in the array if it is an integer
                               array:  the null value is simply encoded in the
                               compressed array just like any other pixel value.

                                 If it is a floating point array, then we need
                               to check for null only if the anynul parameter
                               returned a true value when reading the tile
                            */

                            /* Collapse sizes of higher dimension tiles into 2
                            dimensional equivalents needed by the quantizing
                            algorithms for floating point types */
                            fits_calc_tile_rows(
                                &lpixel,
                                &fpixel,
                                naxis,
                                &mut trowsize,
                                &mut ntrows,
                                status,
                            );

                            if anynul != 0 && datatype == TFLOAT {
                                imcomp_compress_tile(
                                    outfptr,
                                    row,
                                    datatype,
                                    cast_slice_mut(&mut tiledata),
                                    tilelen,
                                    trowsize,
                                    ntrows,
                                    NullCheckType::None,
                                    &Some(NullValue::Float(fltnull)),
                                    status,
                                );
                            } else if anynul != 0 && datatype == TDOUBLE {
                                imcomp_compress_tile(
                                    outfptr,
                                    row,
                                    datatype,
                                    cast_slice_mut(&mut tiledata),
                                    tilelen,
                                    trowsize,
                                    ntrows,
                                    NullCheckType::SetPixel,
                                    &Some(NullValue::Double(dblnull)),
                                    status,
                                );
                            } else {
                                imcomp_compress_tile(
                                    outfptr,
                                    row,
                                    datatype,
                                    cast_slice_mut(&mut tiledata),
                                    tilelen,
                                    trowsize,
                                    ntrows,
                                    NullCheckType::None,
                                    &Some(NullValue::Double(dummy)),
                                    status,
                                );
                            }

                            /* set flag if we found any null values */
                            if anynul != 0 {
                                gotnulls = true;
                            }

                            /* check for any error in the previous operations */
                            if *status > 0 {
                                ffpmsg_str("Error writing compressed image to table");
                                return *status;
                            }

                            row += 1;
                        }
                    }
                }
            }
        }
    }

    /* insert ZBLANK keyword if necessary; only for TFLOAT or TDOUBLE images */
    if gotnulls {
        ffgcrd_safe(outfptr, cs!(c"ZCMPTYPE"), &mut card, status);
        ffikyj_safe(
            outfptr,
            cs!(c"ZBLANK"),
            LONGLONG::from(COMPRESS_NULL_VALUE),
            Some(cs!(c"null value in the compressed integer array")),
            status,
        );
    }

    *status
}

/// This is the main compression routine.
///
/// This routine does the following to the input tile of pixels:
/// - if it is a float or double image, then it quantizes the pixels
/// - compresses the integer pixel values
/// - writes the compressed byte stream to the FITS file.
///
/// If the tile cannot be quantized than the raw float or double values
/// are losslessly compressed with gzip and then written to the output table.
///
/// This input array may be modified by this routine.  If the array is of type
/// TINT or TFLOAT, and the compression type is HCOMPRESS, then it must have been
/// allocated to be twice as large (8 bytes per pixel) to provide scratch space.
///
/// Note that this routine does not fully support the implicit datatype conversion
/// that is supported when writing to normal FITS images.  The datatype of the
/// input array must have the same datatype (either signed or unsigned) as the
/// output (compressed) FITS image in some cases.
///
/// # Parameters
///
/// * `row` — tile number = row in the binary table that holds the compressed data
fn imcomp_compress_tile(
    outfptr: &mut fitsfile,
    row: c_long,
    datatype: c_int,
    tiledata: &mut [u8],
    tilelen: c_long,
    tilenx: c_long,
    tileny: c_long,
    mut nullcheck: NullCheckType,
    nullflagval: &Option<NullValue>,
    status: &mut c_int,
) -> c_int {
    let mut flag: c_int = 1; // true by default; only = 0 if float data couldn't be quantized
    let mut intlength: c_int = 0; // size of integers to be compressed

    let clen: usize; // size of cbuf
    let mut cbuf: Vec<c_short> = Vec::new(); // compressed data
    let mut nelem: c_int = 0; // number of bytes
    let tilecol: c_int;
    let mut gzip_nelem: usize = 0;

    let ihcompscale: c_int;
    let mut hcompscale: f32;
    let mut noise2: f64 = 0.0;
    let mut noise3: f64 = 0.0;
    let mut noise5: f64 = 0.0;
    let mut bscale: [f64; 1] = [1.0]; // scaling parameters
    let mut bzero: [f64; 1] = [0.0]; // scaling parameters
    let mut hcomp_len: c_long;

    if *status > 0 {
        return *status;
    }

    /* check for special case of losslessly compressing floating point */
    /* images.  Only compression algorithm that supports this is GZIP */
    if (outfptr.Fptr).quantize_level == NO_QUANTIZE
        && ((outfptr.Fptr).compress_type != GZIP_1)
        && ((outfptr.Fptr).compress_type != GZIP_2)
    {
        match datatype {
            TFLOAT | TDOUBLE | TCOMPLEX | TDBLCOMPLEX => {
                ffpmsg_str(
                    "Lossless compression of floating point images must use GZIP (imcomp_compress_tile)",
                );
                *status = DATA_COMPRESSION_ERR;
                return *status;
            }
            _ => {}
        }
    }

    /* free the previously saved tile if the input tile is for the same row */
    if !(outfptr.Fptr).tilerow.is_null() {
        /* has the tile cache been allocated? */

        /* calculate the column bin of the compressed tile */
        tilecol = ((row - 1)
            % (((((outfptr.Fptr).znaxis[0] - 1) / ((outfptr.Fptr).tilesize[0])) as c_long) + 1))
            as c_int;

        /* The cached tiles are owned by the TileStruct stored in the global
        TILE_STRUCTS map (see the decompress path, which reads and writes the
        cache exclusively through this map). If the cached tile is for this
        same row it is about to become stale, so invalidate it by clearing the
        owned buffers and resetting the metadata. The previous code
        reinterpreted the TileStruct's `Vec<Vec<u8>>` storage as a raw
        `*mut c_void` array and freed the entries with Vec::from_raw_parts,
        which double-freed memory still owned by the TileStruct (and, for
        columns other than the first, used the wrong element stride). */
        let key = &raw const outfptr.Fptr as usize;
        let mut tilestruct_lock = TILE_STRUCTS.lock().unwrap();
        if let Some(tilestruct) = tilestruct_lock.get_mut(&key)
            && c_long::from(tilestruct.tilerow[tilecol as usize]) == row
        {
            tilestruct.tiledata[tilecol as usize] = Vec::new();
            tilestruct.tilenullarray[tilecol as usize] = Vec::new();
            tilestruct.tilerow[tilecol as usize] = 0;
            tilestruct.tiledatasize[tilecol as usize] = 0;
            tilestruct.tiletype[tilecol as usize] = 0;
            tilestruct.tileanynull[tilecol as usize] = 0;
        }
    }

    if (outfptr.Fptr).compress_type == NOCOMPRESS {
        /* Special case when using NOCOMPRESS for diagnostic purposes in fpack */
        if imcomp_write_nocompress_tile(
            outfptr,
            row,
            datatype,
            tiledata,
            tilelen,
            nullcheck,
            nullflagval,
            status,
        ) > 0
        {
            return *status;
        }
        return *status;
    }

    /* ===========================================================================*/
    /* initialize various parameters */

    /* zbitpix is the BITPIX keyword value in the uncompressed FITS image */
    let zbitpix: c_int = (outfptr.Fptr).zbitpix;

    /* if the tile/image has an integer datatype, see if a null value has */
    /* been defined (with the BLANK keyword in a normal FITS image).  */
    /* If so, and if the input tile array also contains null pixels, */
    /* (represented by pixels that have a value = nullflagval) then  */
    /* any pixels whose value = nullflagval, must be set to the value = nullval
     */
    /* before the pixel array is compressed.  These null pixel values must */
    /* not be inverse scaled by the BSCALE/BZERO values, if present. */

    let cn_zblank: c_int = (outfptr.Fptr).cn_zblank;
    let nullval: c_int = (outfptr.Fptr).zblank;

    if zbitpix > 0 && cn_zblank != -1 {
        /* If the integer image has no defined null */
        nullcheck = NullCheckType::None; /* value, then don't bother checking input array for nulls. */
    }

    /* if the BSCALE and BZERO keywords exist, then the input values must */
    /* be inverse scaled by this factor, before the values are compressed. */
    /* (The program may have turned off scaling, which over rides the keywords)
     */

    let scale: f64 = (outfptr.Fptr).cn_bscale;
    let zero: f64 = (outfptr.Fptr).cn_bzero;
    let actual_bzero: f64 = (outfptr.Fptr).cn_actual_bzero;

    /* ===========================================================================
     */
    /* prepare the tile of pixel values for compression */
    if datatype == TSHORT {
        imcomp_convert_tile_tshort(
            outfptr,
            tiledata,
            tilelen,
            nullcheck,
            nullflagval.as_ref().map(|nv| nv.get_value_as_f64() as _),
            nullval,
            zbitpix,
            scale,
            zero,
            actual_bzero,
            &mut intlength,
            status,
        );
    } else if datatype == TUSHORT {
        imcomp_convert_tile_tushort(
            outfptr,
            tiledata,
            tilelen,
            nullcheck,
            nullflagval.as_ref().map(|nv| nv.get_value_as_f64() as _),
            nullval,
            zbitpix,
            scale,
            zero,
            &mut intlength,
            status,
        );
    } else if datatype == TBYTE {
        imcomp_convert_tile_tbyte(
            outfptr,
            tiledata,
            tilelen,
            nullcheck,
            nullflagval.as_ref().map(|nv| nv.get_value_as_f64() as _),
            nullval,
            zbitpix,
            scale,
            zero,
            &mut intlength,
            status,
        );
    } else if datatype == TSBYTE {
        imcomp_convert_tile_tsbyte(
            outfptr,
            tiledata,
            tilelen,
            nullcheck,
            nullflagval.as_ref().map(|nv| nv.get_value_as_f64() as _),
            nullval,
            zbitpix,
            scale,
            zero,
            &mut intlength,
            status,
        );
    } else if datatype == TINT {
        imcomp_convert_tile_tint(
            outfptr,
            tiledata,
            tilelen,
            nullcheck,
            nullflagval.as_ref().map(|nv| nv.get_value_as_f64() as _),
            nullval,
            zbitpix,
            scale,
            zero,
            &mut intlength,
            status,
        );
    } else if datatype == TUINT {
        imcomp_convert_tile_tuint(
            outfptr,
            tiledata,
            tilelen,
            nullcheck,
            nullflagval.as_ref().map(|nv| nv.get_value_as_f64() as _),
            nullval,
            zbitpix,
            scale,
            zero,
            &mut intlength,
            status,
        );
    } else if datatype == TLONG && mem::size_of::<c_long>() == 8 {
        ffpmsg_str("Integer*8 Long datatype is not supported when writing to compressed images");
        *status = BAD_DATATYPE;
        return *status;
    } else if datatype == TULONG && mem::size_of::<c_long>() == 8 {
        ffpmsg_cstr(
            c"Unsigned integer*8 datatype is not supported when writing to compressed images",
        );
        *status = BAD_DATATYPE;
        return *status;
    } else if datatype == TFLOAT {
        imcomp_convert_tile_tfloat(
            outfptr,
            row,
            tiledata,
            tilelen,
            tilenx,
            tileny,
            nullcheck,
            nullflagval.as_ref().map(|nv| nv.get_value_as_f64() as _),
            nullval,
            zbitpix,
            scale,
            zero,
            &mut intlength,
            &mut flag,
            &mut bscale[0],
            &mut bzero[0],
            status,
        );
    } else if datatype == TDOUBLE {
        imcomp_convert_tile_tdouble(
            outfptr,
            row,
            tiledata,
            tilelen,
            tilenx,
            tileny,
            nullcheck,
            nullflagval.as_ref().map(|nv| nv.get_value_as_f64() as _),
            nullval,
            zbitpix,
            scale,
            zero,
            &mut intlength,
            &mut flag,
            &mut bscale[0],
            &mut bzero[0],
            status,
        );
    } else {
        ffpmsg_str("unsupported image datatype (imcomp_compress_tile)");
        *status = BAD_DATATYPE;
        return *status;
    }

    if *status > 0 {
        return *status; /* return if error occurs */
    }

    /* =========================================================================== */
    if flag != 0 {
        /* now compress the integer data array */
        /* allocate buffer for the compressed tile bytes */
        clen = (outfptr.Fptr).maxelem as usize / 2; // Divide by 2 to convert char to short
        cbuf = Vec::new();

        if cbuf.try_reserve_exact(clen).is_err() {
            ffpmsg_str("Memory allocation failure. (imcomp_compress_tile)");
            *status = MEMORY_ALLOCATION;
            return *status;
        } else {
            cbuf.resize(clen, 0);
        }

        /* =========================================================================== */
        if (outfptr.Fptr).compress_type == RICE_1 {
            let idata: &mut [c_int] = cast_slice_mut(tiledata); /* may overwrite the input tiledata in place */

            let res = if intlength == 2 {
                let mut rce = RCEncoder::new(cast_slice_mut(&mut cbuf));
                rce.set_log_fn(ffpmsg_str);
                rce.encode_short(
                    cast_slice(idata),
                    tilelen as usize,
                    (outfptr.Fptr).rice_blocksize as usize,
                )
            } else if intlength == 1 {
                let mut rce = RCEncoder::new(cast_slice_mut(&mut cbuf));
                rce.set_log_fn(ffpmsg_str);
                rce.encode_byte(
                    cast_slice(idata),
                    tilelen as usize,
                    (outfptr.Fptr).rice_blocksize as usize,
                )
            } else {
                let mut rce = RCEncoder::new(cast_slice_mut(&mut cbuf));
                rce.set_log_fn(ffpmsg_str);
                rce.encode(
                    cast_slice(idata),
                    tilelen as usize,
                    (outfptr.Fptr).rice_blocksize as usize,
                )
            };

            if res.is_err() {
                /* data compression error condition */
                ffpmsg_str("error Rice compressing image tile (imcomp_compress_tile)");
                *status = DATA_COMPRESSION_ERR;
                return *status;
            }

            nelem = res.unwrap() as c_int;

            /* Write the compressed byte stream. */
            ffpclb_safe(
                outfptr,
                (outfptr.Fptr).cn_compressed,
                row as LONGLONG,
                1,
                LONGLONG::from(nelem),
                cast_slice(&cbuf),
                status,
            );
        }
        /* =========================================================================== */
        else if (outfptr.Fptr).compress_type == PLIO_1 {
            let idata: &mut [c_int] = cast_slice_mut(tiledata); /* may overwrite the input tiledata in place */

            for ii in 0..(tilelen as usize) {
                if idata[ii] < 0 || idata[ii] > 16777215 {
                    /* plio algorithn only supports positive 24 bit ints */
                    ffpmsg_str("data out of range for PLIO compression (0 - 2**24)");
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }
            }

            /* cbuf is sized for the worst case, so None is unreachable */
            nelem = match pl_p2li(idata, 0, &mut cbuf, tilelen.try_into().unwrap()) {
                Some(n) => n as c_int,
                None => {
                    ffpmsg_str("PLIO line list did not fit the tile buffer");
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }
            };

            if nelem < 0 {
                /* data compression error condition */
                ffpmsg_str("error PLIO compressing image tile (imcomp_compress_tile)");
                *status = DATA_COMPRESSION_ERR;
                return *status;
            }

            /* Write the compressed byte stream. */
            ffpcli_safe(
                outfptr,
                (outfptr.Fptr).cn_compressed,
                row as LONGLONG,
                1,
                LONGLONG::from(nelem),
                &cbuf,
                status,
            );
        }
        /* =========================================================================== */
        else if ((outfptr.Fptr).compress_type == GZIP_1)
            || ((outfptr.Fptr).compress_type == GZIP_2)
        {
            let mut gzip_buf: Vec<u8> = Vec::new(); /* compressed output; grown by compress2mem_from_mem */
            if (outfptr.Fptr).quantize_level == NO_QUANTIZE && datatype == TFLOAT {
                /* Special case of losslessly compressing floating point pixels  with GZIP */
                /* In this case we compress the input tile array directly */

                if BYTESWAPPED {
                    ffswap4(cast_slice_mut(tiledata), tilelen);
                }

                if (outfptr.Fptr).compress_type == GZIP_2 {
                    fits_shuffle_4bytes(cast_slice_mut(tiledata), tilelen as LONGLONG, status);
                }

                compress2mem_from_mem(
                    cast_slice(tiledata),
                    tilelen as usize * mem::size_of::<f32>(),
                    &mut gzip_buf,
                    Some(&mut gzip_nelem),
                    status,
                );
            } else if (outfptr.Fptr).quantize_level == NO_QUANTIZE && datatype == TDOUBLE {
                /* Special case of losslessly compressing double pixels with  GZIP */
                /* In this case we compress the input tile array directly */

                if BYTESWAPPED {
                    ffswap8(cast_slice_mut(tiledata), tilelen);
                }
                if (outfptr.Fptr).compress_type == GZIP_2 {
                    fits_shuffle_8bytes(cast_slice_mut(tiledata), tilelen as LONGLONG, status);
                }

                compress2mem_from_mem(
                    cast_slice(tiledata),
                    tilelen as usize * mem::size_of::<f64>(),
                    &mut gzip_buf,
                    Some(&mut gzip_nelem),
                    status,
                );
            } else {
                /* compress the integer idata array */

                if BYTESWAPPED {
                    let idata: &mut [c_int] = cast_slice_mut(tiledata); /* may overwrite the input tiledata in place */

                    if intlength == 2 {
                        ffswap2(cast_slice_mut(idata), tilelen);
                    } else if intlength == 4 {
                        ffswap4(idata, tilelen);
                    }
                }

                if intlength == 2 {
                    if (outfptr.Fptr).compress_type == GZIP_2 {
                        fits_shuffle_2bytes(cast_slice_mut(tiledata), tilelen as LONGLONG, status);
                    }

                    let idata: &mut [c_int] = cast_slice_mut(tiledata);

                    compress2mem_from_mem(
                        cast_slice_mut(idata),
                        tilelen as usize * mem::size_of::<c_short>(),
                        &mut gzip_buf,
                        Some(&mut gzip_nelem),
                        status,
                    );
                } else if intlength == 1 {
                    let idata: &mut [c_int] = cast_slice_mut(tiledata);
                    compress2mem_from_mem(
                        cast_slice(idata),
                        tilelen as usize * mem::size_of::<c_uchar>(),
                        &mut gzip_buf,
                        Some(&mut gzip_nelem),
                        status,
                    );
                } else {
                    if (outfptr.Fptr).compress_type == GZIP_2 {
                        fits_shuffle_4bytes(cast_slice_mut(tiledata), tilelen as LONGLONG, status);
                    }

                    let idata: &mut [c_int] = cast_slice_mut(tiledata);

                    compress2mem_from_mem(
                        cast_slice_mut(idata),
                        tilelen as usize * mem::size_of::<c_int>(),
                        &mut gzip_buf,
                        Some(&mut gzip_nelem),
                        status,
                    );
                }
            }

            /* Write the compressed byte stream. */
            ffpclb_safe(
                outfptr,
                LONGLONG::from((outfptr.Fptr).cn_compressed)
                    .try_into()
                    .unwrap(),
                row as LONGLONG,
                1,
                gzip_nelem as LONGLONG,
                &gzip_buf,
                status,
            );

        /* =========================================================================== */
        } else if (outfptr.Fptr).compress_type == BZIP2_1 {
            #[cfg(feature = "bzip2")]
            {
                use libbz2_rs_sys::BZ2_bzBuffToBuffCompress;

                if BYTESWAPPED {
                    let idata: &mut [c_int] = cast_slice_mut(tiledata);

                    if intlength == 2 {
                        ffswap2(cast_slice_mut(idata), tilelen);
                    } else if intlength == 4 {
                        ffswap4(idata, tilelen);
                    }
                }

                /* clen counts c_shorts here; the C's is in bytes */
                let mut bzlen: c_uint = (clen * 2) as c_uint;

                /* call bzip2 with blocksize = 900K, verbosity = 0, and default workfactor */

                /*  bzip2 is not supported in the public release.  This is only for
                test purposes. */
                if unsafe {
                    BZ2_bzBuffToBuffCompress(
                        cbuf.as_ptr() as *mut c_char,
                        &raw mut bzlen,
                        tiledata.as_ptr() as *mut _,
                        (tilelen * c_long::from(intlength)) as c_uint,
                        9,
                        0,
                        0,
                    )
                } != 0
                {
                    ffpmsg_str("bzip2 compression error");
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }

                /* Write the compressed byte stream. */
                ffpclb_safe(
                    outfptr,
                    (outfptr.Fptr).cn_compressed,
                    row as LONGLONG,
                    1,
                    LONGLONG::from(bzlen),
                    cast_slice_mut(&mut cbuf),
                    status,
                );
            }

        /* =========================================================================== */
        } else if (outfptr.Fptr).compress_type == HCOMPRESS_1 {
            /*
            if hcompscale is positive, then we have to multiply
            the value by the RMS background noise to get the
            absolute scale value.  If negative, then it gives the
            absolute scale value directly.
            */
            hcompscale = (outfptr.Fptr).hcomp_scale;

            if hcompscale > 0.0 {
                let idata: &mut [c_int] = cast_slice_mut(tiledata);

                fits_img_stats_int_safe(
                    idata,
                    tilenx,
                    tileny,
                    nullcheck != NullCheckType::None,
                    nullval,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(&mut noise2),
                    Some(&mut noise3),
                    Some(&mut noise5),
                    status,
                );

                /* use the minimum of the 3 noise estimates */
                if noise2 != 0. && noise2 < noise3 {
                    noise3 = noise2;
                }
                if noise5 != 0. && noise5 < noise3 {
                    noise3 = noise5;
                }

                hcompscale = (f64::from(hcompscale) * noise3) as f32;
            } else if hcompscale < 0.0 {
                hcompscale = -hcompscale;
            }

            ihcompscale = (hcompscale + 0.5) as c_int;

            /* clen counts c_shorts here; the C's is in bytes */
            hcomp_len = (clen * 2) as c_long; /* allocated size of the buffer */

            /* The C gets the compressed length back from fits_hcompress.
            HCEncoder returns none, so take it from how far it advanced the
            output slice -- `Write for &mut [u8]' consumes as it writes. */
            let mut out: &mut [u8] = cast_slice_mut(&mut cbuf);
            let capacity = out.len();

            if zbitpix == BYTE_IMG || zbitpix == SHORT_IMG {
                let idata: &mut [c_int] = cast_slice_mut(tiledata);

                let mut hc_enconder = HCEncoder::new(&mut out);

                let encode_res = hc_enconder.write(
                    idata,
                    tilenx.try_into().unwrap(),
                    tileny.try_into().unwrap(),
                    ihcompscale,
                );

                if let Err(_e) = encode_res {
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }
            } else {
                /* have to convert idata to an I*8 array, in place */
                /* idata must have been allocated large enough to do this */
                let idata: &mut [c_int] = cast_slice_mut(tiledata);

                fits_int_to_longlong_inplace(idata, tilelen, status);

                let lldata: &mut [LONGLONG] = cast_slice_mut(idata);
                let mut hc_enconder = HCEncoder::new(&mut out);

                let encode_res = hc_enconder.write64(
                    lldata,
                    tilenx.try_into().unwrap(),
                    tileny.try_into().unwrap(),
                    ihcompscale,
                );

                if let Err(_e) = encode_res {
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }
            }

            hcomp_len = (capacity - out.len()) as c_long;

            /* Write the compressed byte stream. */
            ffpclb_safe(
                outfptr,
                (outfptr.Fptr).cn_compressed,
                row as LONGLONG,
                1,
                hcomp_len as LONGLONG,
                cast_slice(&cbuf),
                status,
            );
        }

        /* =========================================================================== */
        if (outfptr.Fptr).cn_zscale > 0 {
            /* write the linear scaling parameters for this tile */
            ffpcld_safe(
                outfptr,
                (outfptr.Fptr).cn_zscale,
                row as LONGLONG,
                1,
                1,
                &bscale,
                status,
            );
            ffpcld_safe(
                outfptr,
                (outfptr.Fptr).cn_zzero,
                row as LONGLONG,
                1,
                1,
                &bzero,
                status,
            );
        }

        /* finished with this buffer */

        /* =========================================================================== */
    } else {
        /* if flag == 0., floating point data couldn't be quantized */

        /* losslessly compress the data with gzip. */

        /* if gzip2 compressed data column doesn't exist, create it */
        if (outfptr.Fptr).cn_gzip_data < 1 {
            if (outfptr.Fptr).request_huge_hdu != 0 {
                fits_insert_col(
                    outfptr,
                    999,
                    cs!(c"GZIP_COMPRESSED_DATA"),
                    cs!(c"1QB"),
                    status,
                );
            } else {
                fits_insert_col(
                    outfptr,
                    999,
                    cs!(c"GZIP_COMPRESSED_DATA"),
                    cs!(c"1PB"),
                    status,
                );
            }

            if *status <= 0 {
                /* save the number of this column */
                let mut cn_gzip_data: c_int = 0;
                ffgcno_safe(
                    outfptr,
                    CASEINSEN.try_into().unwrap(),
                    cs!(c"GZIP_COMPRESSED_DATA"),
                    &mut cn_gzip_data,
                    status,
                );
                outfptr.Fptr.cn_gzip_data = cn_gzip_data;
            }
        }

        let mut gzip_buf: Vec<u8> = Vec::new(); /* compressed output; grown by compress2mem_from_mem */
        if datatype == TFLOAT {
            /* allocate buffer for the compressed tile bytes */
            /* make it 10% larger than the original uncompressed data */

            clen = (tilelen as f64 * mem::size_of::<f32>() as f64 * 1.1 / 2.0) as usize; // Divide by 2 to convert char to short.

            cbuf = Vec::new();

            if cbuf.try_reserve_exact(clen).is_err() {
                ffpmsg_str("Memory allocation error. (imcomp_compress_tile)");
                *status = MEMORY_ALLOCATION;
                return *status;
            } else {
                cbuf.resize(clen, 0);
            }

            /* convert null values to NaNs in place, if necessary */
            if nullcheck == NullCheckType::SetPixel {
                let nv = nullflagval.clone().unwrap(); // Must be set if nullcheck is SetPixel
                imcomp_float2nan_inplace(
                    cast_slice_mut(tiledata),
                    tilelen,
                    nv.get_value_as_f64() as f32,
                    status,
                );
            }

            if BYTESWAPPED {
                ffswap4(cast_slice_mut(tiledata), tilelen);
            }

            // WARNING: Potentially unsafe memory issue here given this function
            // call can reallocate the buffer.
            compress2mem_from_mem(
                cast_slice(tiledata),
                tilelen as usize * mem::size_of::<f32>(),
                &mut gzip_buf,
                Some(&mut gzip_nelem),
                status,
            );
        } else {
            /* datatype == TDOUBLE */

            /* allocate buffer for the compressed tile bytes */
            /* make it 10% larger than the original uncompressed data */
            clen = (tilelen as f64 * mem::size_of::<f64>() as f64 * 1.1 / 2.0) as usize; // Divide by 2 to convert char to short
            if cbuf.try_reserve_exact(clen).is_err() {
                ffpmsg_str("Memory allocation error. (imcomp_compress_tile)");
                *status = MEMORY_ALLOCATION;
                return *status;
            } else {
                cbuf.resize(clen, 0);
            }

            /* convert null values to NaNs in place, if necessary */
            if nullcheck == NullCheckType::SetPixel {
                let nv = nullflagval.clone().unwrap(); // Must be set if nullcheck is SetPixel
                imcomp_double2nan_inplace(
                    cast_slice_mut(tiledata),
                    tilelen,
                    nv.get_value_as_f64(),
                    status,
                );
            }

            if BYTESWAPPED {
                ffswap8(cast_slice_mut(tiledata), tilelen);
            }

            // WARNING: Potentially unsafe memory issue here given this function
            // call can reallocate the buffer.
            compress2mem_from_mem(
                cast_slice_mut(tiledata),
                tilelen as usize * mem::size_of::<f64>(),
                &mut gzip_buf,
                Some(&mut gzip_nelem),
                status,
            );
        }

        /* Write the compressed byte stream. */
        ffpclb_safe(
            outfptr,
            (outfptr.Fptr).cn_gzip_data,
            row as LONGLONG,
            1,
            gzip_nelem.try_into().unwrap(),
            &gzip_buf,
            status,
        );

        /* we must zero out existing existing compressed data if it exists. */
        /* otherwise on read this data is read ahead of the gzipped */
        /* data and will cause a bug. */
        let mut _test_nelemll: LONGLONG = 0;
        let mut _test_offset: LONGLONG = 0;
        ffgdesll_safe(
            outfptr,
            (outfptr.Fptr).cn_compressed,
            row.into(),
            Some(&mut _test_nelemll),
            Some(&mut _test_offset),
            status,
        );
        if _test_nelemll != 0 {
            ffpclb_safe(
                outfptr,
                (outfptr.Fptr).cn_compressed,
                row.into(),
                1,
                0,
                &[],
                status,
            );
        }

        /* finished with this buffer */
    }

    *status
}

fn imcomp_write_nocompress_tile(
    outfptr: &mut fitsfile,
    row: c_long,
    datatype: c_int,
    tiledata: &[u8],
    tilelen: c_long,
    _nullcheck: NullCheckType,
    _nullflagval: &Option<NullValue>,
    status: &mut c_int,
) -> c_int {
    let mut coltype: [c_char; 4] = [0; 4];

    /* Write the uncompressed image tile pixels to the tile-compressed image file. */
    /* This is a special case when using NOCOMPRESS for diagnostic purposes in fpack. */
    /* Currently, this only supports a limited number of data types and */
    /* does not fully support null-valued pixels in the image. */

    if (outfptr.Fptr).cn_uncompressed < 1 {
        /* uncompressed data column doesn't exist, so append new column to table
         */
        if datatype == TSHORT {
            strcpy_safe(&mut coltype, cs!(c"1PI"));
        } else if datatype == TINT {
            strcpy_safe(&mut coltype, cs!(c"1PJ"));
        } else if datatype == TFLOAT {
            strcpy_safe(&mut coltype, cs!(c"1QE"));
        } else {
            ffpmsg_cstr(
                c"NOCOMPRESSION option only supported for int*2, int*4, and float*4 images",
            );
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }

        fits_insert_col(outfptr, 999, cs!(c"UNCOMPRESSED_DATA"), &coltype, status);
        /* create column */
    }

    let mut cn = (outfptr.Fptr).cn_uncompressed;
    fits_get_colnum(
        outfptr,
        CASEINSEN.try_into().unwrap(),
        cs!(c"UNCOMPRESSED_DATA"),
        &mut cn,
        status,
    ); /* save col. num. */

    (outfptr.Fptr).cn_uncompressed = cn; /* save col. num. */

    fits_write_col(
        outfptr,
        datatype,
        (outfptr.Fptr).cn_uncompressed,
        row as LONGLONG,
        1,
        tilelen as LONGLONG,
        tiledata,
        status,
    ); /* write the tile data */
    *status
}

/// Prepare the input tile array of pixels for compression.
/// Convert input integer*2 tile array in place to 4 or 8-byte ints for compression,
/// If needed, convert 4 or 8-byte ints and do null value substitution.
/// Note that the calling routine must have allocated the input array big enough
/// to be able to do this.
fn imcomp_convert_tile_tshort(
    outfptr: &mut fitsfile,
    tiledata: &mut [u8],
    tilelen: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<c_short>,
    nullval: c_int,
    zbitpix: c_int,
    scale: f64,
    zero: f64,
    actual_bzero: f64,
    intlength: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut flagval: c_short = 0;

    /* We only support writing this integer*2 tile data to a FITS image with
    BITPIX = 16 and with BZERO = 0 and BSCALE = 1.  */

    if zbitpix != SHORT_IMG || scale != 1.0 || zero != 0.0 {
        ffpmsg_cstr(
            c"Datatype conversion/scaling is not supported when writing to compressed images",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    if (outfptr.Fptr).compress_type == RICE_1
        || (outfptr.Fptr).compress_type == GZIP_1
        || (outfptr.Fptr).compress_type == GZIP_2
        || (outfptr.Fptr).compress_type == BZIP2_1
    {
        /* don't have to convert to int if using gzip, bzip2 or Rice compression */
        *intlength = 2;

        let sbuff: &mut [c_short] = cast_slice_mut(tiledata);

        if nullcheck == NullCheckType::SetPixel {
            /* reset pixels equal to flagval to the FITS null value, prior to compression */
            flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
            if i32::from(flagval) != nullval {
                for ii in (0..(tilelen as usize)).rev() {
                    if sbuff[ii] == (flagval as c_short) {
                        sbuff[ii] = nullval as c_short;
                    }
                }
            }
        }
    } else if (outfptr.Fptr).compress_type == HCOMPRESS_1 {
        /* have to convert to int if using HCOMPRESS */
        *intlength = 4;

        if nullcheck == NullCheckType::SetPixel {
            /* reset pixels equal to flagval to the FITS null value, prior to compression */
            flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
            for ii in (0..(tilelen as usize)).rev() {
                let sbuff_ii: c_short = cast_slice(tiledata)[ii];
                let idata: &mut [c_int] = cast_slice_mut(tiledata);

                if sbuff_ii == (flagval as c_short) {
                    idata[ii] = nullval;
                } else {
                    idata[ii] = c_int::from(sbuff_ii);
                }
            }
        } else {
            /* just do the data type conversion to int */
            /* have to convert sbuff to an I*4 array, in place */
            /* sbuff must have been allocated large enough to do this */
            let sbuff: &mut [c_short] = cast_slice_mut(tiledata);
            fits_short_to_int_inplace(sbuff, tilelen, 0, status);
        }
    } else {
        /* have to convert to int if using PLIO */
        *intlength = 4;
        if zero == 0. && actual_bzero == 32768. {
            /* Here we are compressing unsigned 16-bit integers that have */
            /* been offset by -32768 using the standard FITS convention. */
            /* Since PLIO cannot deal with negative values, we must apply */
            /* the shift of 32786 to the values to make them all positive. */
            /* The inverse negative shift will be applied in */
            /* imcomp_decompress_tile when reading the compressed tile. */
            if nullcheck == NullCheckType::SetPixel {
                /* reset pixels equal to flagval to the FITS null value, prior  to compression */
                flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
                for ii in (0..(tilelen as usize)).rev() {
                    let sbuff_ii: c_short = cast_slice(tiledata)[ii];
                    let idata: &mut [c_int] = cast_slice_mut(tiledata);

                    if sbuff_ii == (flagval as c_short) {
                        idata[ii] = nullval;
                    } else {
                        idata[ii] = c_int::from(sbuff_ii) + 32768;
                    }
                }
            } else {
                /* have to convert sbuff to an I*4 array, in place */
                /* sbuff must have been allocated large enough to do this */
                let sbuff: &mut [c_short] = cast_slice_mut(tiledata);
                fits_short_to_int_inplace(sbuff, tilelen, 32768, status);
            }
        } else {
            /* This is not an unsigned 16-bit integer array, so process normally */
            if nullcheck == NullCheckType::SetPixel {
                /* reset pixels equal to flagval to the FITS null value, prior  to compression */
                flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
                for ii in (0..(tilelen as usize)).rev() {
                    let sbuff_ii: c_short = cast_slice(tiledata)[ii];
                    let idata: &mut [c_int] = cast_slice_mut(tiledata);

                    if sbuff_ii == (flagval as c_short) {
                        idata[ii] = nullval;
                    } else {
                        idata[ii] = c_int::from(sbuff_ii);
                    }
                }
            } else {
                /* just do the data type conversion to int */
                /* have to convert sbuff to an I*4 array, in place */
                /* sbuff must have been allocated large enough to do this */
                let sbuff: &mut [c_short] = cast_slice_mut(tiledata);
                fits_short_to_int_inplace(sbuff, tilelen, 0, status);
            }
        }
    }
    *status
}

/// Prepare the input  tile array of pixels for compression.
/// Convert input unsigned integer*2 tile array in place to 4 or 8-byte ints
/// for compression,
/// If needed, convert 4 or 8-byte ints and do null value substitution.
/// Note that the calling routine must have allocated the input array big
/// enough
/// to be able to do this.
fn imcomp_convert_tile_tushort(
    outfptr: &mut fitsfile,
    tiledata: &mut [u8],
    tilelen: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<c_ushort>,
    nullval: c_int,
    zbitpix: c_int,
    scale: f64,
    zero: f64,
    intlength: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut flagval: c_ushort = 0;

    /* datatype of input array is unsigned short.  We only support writing this
    datatype to a FITS image with BITPIX = 16 and with BZERO = 0 and BSCALE =
    32768.  */

    if zbitpix != SHORT_IMG || scale != 1.0 || zero != 32768. {
        ffpmsg_cstr(
            c"Implicit datatype conversion is not supported when writing to compressed images",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    if (outfptr.Fptr).compress_type == RICE_1
        || (outfptr.Fptr).compress_type == GZIP_1
        || (outfptr.Fptr).compress_type == GZIP_2
        || (outfptr.Fptr).compress_type == BZIP2_1
    {
        /* don't have to convert to int if using gzip, bzip2, or Rice compression */
        *intlength = 2;

        let usbuff: &mut [c_ushort] = cast_slice_mut(tiledata);

        /* offset the unsigned value by -32768 to a signed short value. */
        /* It is more efficient to do this by just flipping the most significant of the 16 bits */

        if nullcheck == NullCheckType::SetPixel {
            /* reset pixels equal to flagval to the FITS null value, prior to compression  */
            flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
            for ii in (0..(tilelen as usize)).rev() {
                if usbuff[ii] == (flagval as c_ushort) {
                    usbuff[ii] = (nullval as c_short) as c_ushort;
                } else {
                    usbuff[ii] ^= 0x8000;
                }
            }
        } else {
            /* just offset the pixel values by 32768 (by flipping the MSB */
            for ii in (0..(tilelen as usize)).rev() {
                usbuff[ii] ^= 0x8000;
            }
        }
    } else {
        /* have to convert to int if using HCOMPRESS or PLIO */
        *intlength = 4;

        if nullcheck == NullCheckType::SetPixel {
            /* offset the pixel values by 32768, and */
            /* reset pixels equal to flagval to nullval */
            flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
            for ii in (0..(tilelen as usize)).rev() {
                let usbuff_ii: c_ushort = cast_slice(tiledata)[ii];
                let idata: &mut [c_int] = cast_slice_mut(tiledata);
                if usbuff_ii == (flagval as c_ushort) {
                    idata[ii] = nullval;
                } else {
                    idata[ii] = c_int::from(usbuff_ii) - 32768;
                }
            }
        } else {
            /* just do the data type conversion to int */
            /* for HCOMPRESS we need to simply subtract 32768 */
            /* for PLIO, have to convert usbuff to an I*4 array, in place */
            /* usbuff must have been allocated large enough to do this */

            let usbuff: &mut [c_ushort] = cast_slice_mut(tiledata);

            if (outfptr.Fptr).compress_type == HCOMPRESS_1 {
                fits_ushort_to_int_inplace(usbuff, tilelen, -32768, status);
            } else {
                fits_ushort_to_int_inplace(usbuff, tilelen, 0, status);
            }
        }
    }

    *status
}

/// Prepare the input tile array of pixels for compression.
/// Convert input integer tile array in place to 4 or 8-byte ints for compression,
/// If needed, do null value substitution.
fn imcomp_convert_tile_tint(
    _outfptr: &mut fitsfile,
    tiledata: &mut [u8],
    tilelen: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<c_int>,
    nullval: c_int,
    zbitpix: c_int,
    scale: f64,
    zero: f64,
    intlength: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut flagval: c_int = 0;

    /* datatype of input array is int.  We only support writing this datatype
    to a FITS image with BITPIX = 32 and with BZERO = 0 and BSCALE = 1.  */

    if zbitpix != LONG_IMG || scale != 1.0 || zero != 0.0 {
        ffpmsg_cstr(
            c"Implicit datatype conversion is not supported when writing to compressed images",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    let idata: &mut [c_int] = cast_slice_mut(tiledata);
    *intlength = 4;

    if nullcheck == NullCheckType::SetPixel {
        /* no datatype conversion is required for any of the compression
        algorithms, except possibly for HCOMPRESS (to I*8), which is handled
        later. Just reset pixels equal to flagval to the FITS null value */
        flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
        if flagval != nullval {
            for ii in (0..(tilelen as usize)).rev() {
                if idata[ii] == flagval {
                    idata[ii] = nullval;
                }
            }
        }
    }

    *status
}

/// Prepare the input tile array of pixels for compression.
/// Convert input unsigned integer tile array in place to 4 or 8-byte ints for compression,
/// If needed, do null value substitution.
fn imcomp_convert_tile_tuint(
    _outfptr: &mut fitsfile,
    tiledata: &mut [u8],
    tilelen: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<c_uint>,
    nullval: c_int,
    zbitpix: c_int,
    scale: f64,
    zero: f64,
    intlength: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut uintflagval: c_uint = 0;

    /* datatype of input array is unsigned int.  We only support writing this
    datatype to a FITS image with BITPIX = 32 and with BZERO = 0 and BSCALE =
    2147483648.  */

    if zbitpix != LONG_IMG || scale != 1.0 || zero != 2147483648. {
        ffpmsg_cstr(
            c"Implicit datatype conversion is not supported when writing to compressed images",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    *intlength = 4;

    /* offset the unsigned value by -2147483648 to a signed int value. */
    /* It is more efficient to do this by just flipping the most significant of the 32 bits */

    if nullcheck == NullCheckType::SetPixel {
        /* reset pixels equal to flagval to nullval and */
        /* offset the other pixel values (by flipping the MSB) */
        uintflagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
        for ii in (0..(tilelen as usize)).rev() {
            let uintbuff_ii: c_uint = cast_slice(tiledata)[ii];
            if uintbuff_ii == uintflagval {
                let idata: &mut [c_int] = cast_slice_mut(tiledata);
                idata[ii] = nullval;
            } else {
                let uintbuff: &mut [c_uint] = cast_slice_mut(tiledata);
                uintbuff[ii] ^= 0x80000000;
            }
        }
    } else {
        /* just offset the pixel values (by flipping the MSB) */
        let uintbuff: &mut [c_uint] = cast_slice_mut(tiledata);
        for ii in (0..(tilelen as usize)).rev() {
            uintbuff[ii] ^= 0x80000000;
        }
    }

    *status
}

/// Prepare the input tile array of pixels for compression.
/// Convert input unsigned integer*1 tile array in place to 4 or 8-byte ints for compression,
/// If needed, convert 4 or 8-byte ints and do null value substitution.
/// Note that the calling routine must have allocated the input array big enough
/// to be able to do this.
fn imcomp_convert_tile_tbyte(
    outfptr: &mut fitsfile,
    tiledata: &mut [u8],
    tilelen: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<c_uchar>,
    nullval: c_int,
    zbitpix: c_int,
    scale: f64,
    zero: f64,
    intlength: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut flagval: c_uchar = 0;

    /* datatype of input array is unsigned byte.  We only support writing this
    datatype to a FITS image with BITPIX = 8 and with BZERO = 0 and BSCALE
    = 1.  */

    if zbitpix != BYTE_IMG || scale != 1.0 || zero != 0.0 {
        ffpmsg_cstr(
            c"Implicit datatype conversion is not supported when writing to compressed images",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    if (outfptr.Fptr).compress_type == RICE_1
        || (outfptr.Fptr).compress_type == GZIP_1
        || (outfptr.Fptr).compress_type == GZIP_2
        || (outfptr.Fptr).compress_type == BZIP2_1
    {
        /* don't have to convert to int if using gzip, bzip2, or Rice compression */
        *intlength = 1;

        let usbbuff: &mut [c_uchar] = cast_slice_mut(tiledata);

        if nullcheck == NullCheckType::SetPixel {
            /* reset pixels equal to flagval to the FITS null value, prior to compression */
            flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
            if i32::from(flagval) != nullval {
                for ii in (0..(tilelen as usize)).rev() {
                    if usbbuff[ii] == flagval as c_uchar {
                        usbbuff[ii] = nullval as c_uchar;
                    }
                }
            }
        }
    } else {
        /* have to convert to int if using HCOMPRESS or PLIO */
        *intlength = 4;

        if nullcheck == NullCheckType::SetPixel {
            /* reset pixels equal to flagval to the FITS null value, prior to compression */
            flagval = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel

            for ii in (0..(tilelen as usize)).rev() {
                let usbuff_ii: c_uchar = cast_slice(tiledata)[ii];
                let idata: &mut [c_int] = cast_slice_mut(tiledata);

                if usbuff_ii == (flagval as c_uchar) {
                    idata[ii] = nullval;
                } else {
                    idata[ii] = c_int::from(usbuff_ii);
                }
            }
        } else {
            /* just do the data type conversion to int */
            /* have to convert usbbuff to an I*4 array, in place */
            /* usbbuff must have been allocated large enough to do this */
            let usbbuff: &mut [c_uchar] = cast_slice_mut(tiledata);
            fits_ubyte_to_int_inplace(usbbuff, tilelen, status);
        }
    }

    *status
}

/// Prepare the input tile array of pixels for compression.
/// Convert input integer*1 tile array in place to 4 or 8-byte ints for compression,
/// If needed, convert 4 or 8-byte ints and do null value substitution.
/// Note that the calling routine must have allocated the input array big enough
/// to be able to do this.
fn imcomp_convert_tile_tsbyte(
    outfptr: &mut fitsfile,
    tiledata: &mut [u8],
    tilelen: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<c_schar>,
    nullval: c_int,
    zbitpix: c_int,
    scale: f64,
    zero: f64,
    intlength: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut flagval: c_char = 0;

    /* datatype of input array is signed byte.  We only support writing this
    datatype to a FITS image with BITPIX = 8 and with BZERO = 0 and BSCALE =
    -128.  */

    if zbitpix != BYTE_IMG || scale != 1.0 || zero != -128. {
        ffpmsg_cstr(
            c"Implicit datatype conversion is not supported when writing to compressed images",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    if (outfptr.Fptr).compress_type == RICE_1
        || (outfptr.Fptr).compress_type == GZIP_1
        || (outfptr.Fptr).compress_type == GZIP_2
        || (outfptr.Fptr).compress_type == BZIP2_1
    {
        /* don't have to convert to int if using gzip, bzip2 or Rice compression */
        *intlength = 1;

        let sbbuff: &mut [c_schar] = cast_slice_mut(tiledata);

        if nullcheck == NullCheckType::SetPixel {
            /* reset pixels equal to flagval to the FITS null value, prior to compression */
            /* offset the other pixel values (by flipping the MSB) */

            flagval = nullflagval.unwrap() as c_char; // Must be set if nullcheck is SetPixel
            for ii in (0..(tilelen as usize)).rev() {
                if sbbuff[ii] == (flagval as c_schar) {
                    sbbuff[ii] = nullval as c_schar;
                } else {
                    sbbuff[ii] = ((sbbuff[ii] as u8) ^ 0x80) as c_schar;
                }
            }
        } else {
            /* just offset the pixel values (by flipping the MSB) */
            for ii in (0..(tilelen as usize)).rev() {
                sbbuff[ii] = ((sbbuff[ii] as u8) ^ 0x80) as c_schar;
            }
        }
    } else {
        /* have to convert to int if using HCOMPRESS or PLIO */
        *intlength = 4;

        if nullcheck == NullCheckType::SetPixel {
            /* reset pixels equal to flagval to the FITS null value, prior to compression */
            flagval = nullflagval.unwrap() as c_char; // Must be set if nullcheck is SetPixel
            for ii in (0..(tilelen as usize)).rev() {
                let sbbuff_ii: c_schar = cast_slice(tiledata)[ii];
                let idata: &mut [c_int] = cast_slice_mut(tiledata);

                if sbbuff_ii == (flagval as c_schar) {
                    idata[ii] = nullval;
                } else {
                    idata[ii] = c_int::from(sbbuff_ii) + 128;
                }
            }
        } else {
            /* just do the data type conversion to int */
            /* have to convert sbbuff to an I*4 array, in place */
            /* sbbuff must have been allocated large enough to do this */
            let sbbuff: &mut [c_schar] = cast_slice_mut(tiledata);
            fits_sbyte_to_int_inplace(sbbuff, tilelen, status);
        }
    }

    *status
}

/// Prepare the input tile array of pixels for compression.
/// Convert input float tile array in place to 4 or 8-byte ints for compression,
/// If needed, convert 4 or 8-byte ints and do null value substitution.
/// Note that the calling routine must have allocated the input array big enough
/// to be able to do this.
fn imcomp_convert_tile_tfloat(
    outfptr: &mut fitsfile,
    row: c_long,
    tiledata: &mut [u8],
    tilelen: c_long,
    tilenx: c_long,
    tileny: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<f32>,
    nullval: c_int,
    zbitpix: c_int,
    scale: f64,
    zero: f64,
    intlength: &mut c_int,
    flag: &mut c_int,
    bscale: &mut f64,
    bzero: &mut f64,
    status: &mut c_int,
) -> c_int {
    let mut irow: c_long = 0;
    let mut floatnull: f32 = 0.0;

    let mut dithersum: c_ulong = 0;
    let mut iminval: c_int = 0;
    let mut imaxval: c_int = 0; /* min and max quantized integers */

    /* datatype of input array is double.  We only support writing this datatype
    to a FITS image with BITPIX = -64 or -32, except we also support the
    special case where BITPIX = 32 and BZERO = 0 and BSCALE = 1.  */

    if (zbitpix != LONG_IMG && zbitpix != DOUBLE_IMG && zbitpix != FLOAT_IMG)
        || scale != 1.0
        || zero != 0.0
    {
        ffpmsg_cstr(
            c"Implicit datatype conversion is not supported when writing to compressed images",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    *intlength = 4;

    /* if the tile-compressed table contains zscale and zzero columns */
    /* then scale and quantize the input floating point data.    */

    if (outfptr.Fptr).cn_zscale > 0 {
        /* quantize the float values into integers */

        if nullcheck == NullCheckType::SetPixel {
            floatnull = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
        } else {
            floatnull = FLOATNULLVALUE; /* NaNs are represented by this, by default */
        }

        if (outfptr.Fptr).quantize_method == SUBTRACTIVE_DITHER_1
            || (outfptr.Fptr).quantize_method == SUBTRACTIVE_DITHER_2
        {
            /* see if the dithering offset value needs to be initialized */
            if (outfptr.Fptr).request_dither_seed == 0 && (outfptr.Fptr).dither_seed == 0 {
                /* This means randomly choose the dithering offset based on the  system time.
                The offset will have a value between 1 and 10000, inclusive.
                The time function returns an integer value that is  incremented each second.
                The clock function returns the elapsed CPU time, in integer  CLOCKS_PER_SEC units.
                The CPU time returned by clock is typically (on linux PC)  only good to 0.01 sec
                Summing the 2 quantities may help avoid cases where 2  executions of the program
                (perhaps in a multithreaded environoment) end up with exactly  the same dither seed
                value.  The sum is incremented by the current HDU number in  the file to provide
                further randomization.  This randomization is desireable if  multiple compressed
                images will be summed (or differenced). In such cases, the  benefits of dithering
                may be lost if all the images use exactly the same sequence  of random numbers when
                calculating the dithering offsets. */

                // The above is not valid in the rust verison, instead we just choose a random number

                (outfptr.Fptr).dither_seed = fastrand::i32(1..=10000);

                /* update the header keyword with this new value */
                let dither_seed = (outfptr.Fptr).dither_seed;
                fits_update_key(
                    outfptr,
                    crate::KeywordDatatype::TINT(&dither_seed),
                    cs!(c"ZDITHER0"),
                    None,
                    status,
                );
            } else if (outfptr.Fptr).request_dither_seed < 0 && (outfptr.Fptr).dither_seed < 0 {
                /* this means randomly choose the dithering offset based on some  hash function */
                /* of the first input tile of data to be quantized and  compressed.  This ensures that */
                /* the same offset value is used for a given image every time it  is compressed. */

                let usbbuff: &[c_uchar] = cast_slice(tiledata);
                dithersum = 0;
                for ii in 0..(4 * tilelen as usize) {
                    dithersum += c_ulong::from(usbbuff[ii]); /* doesn't matter if there is an integer overflow */
                }
                (outfptr.Fptr).dither_seed = ((dithersum % 10000) as c_int) + 1;

                /* update the header keyword with this new value */
                let dither_seed = (outfptr.Fptr).dither_seed;
                fits_update_key(
                    outfptr,
                    crate::KeywordDatatype::TINT(&dither_seed),
                    cs!(c"ZDITHER0"),
                    None,
                    status,
                );
            }

            /* subtract 1 to convert from 1-based to 0-based element number */
            irow = row + c_long::from((outfptr.Fptr).dither_seed) - 1; /* dither the quantized values */
        } else if (outfptr.Fptr).quantize_method == -1 {
            irow = 0; /* do not dither the quantized values */
        } else {
            ffpmsg_str("Unknown dithering method.");
            ffpmsg_str("May need to install a newer version of CFITSIO.");
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }

        *flag = fits_quantize_float_inplace(
            irow as usize,
            cast_slice_mut(tiledata),
            tilenx as usize,
            tileny as usize,
            nullcheck != NullCheckType::None,
            floatnull,
            (outfptr.Fptr).quantize_level,
            DitherType::from((outfptr.Fptr).quantize_method),
            bscale,
            bzero,
            &mut iminval,
            &mut imaxval,
        );

        if *flag > 1 {
            *status = *flag;
            return *status;
        }
    } else if (outfptr.Fptr).quantize_level != NO_QUANTIZE {
        /* if floating point pixels are not being losslessly compressed, then */
        /* input float data is implicitly converted (truncated) to integers */

        let idata: &mut [c_int] = cast_slice_mut(tiledata);

        if scale != 1. || zero != 0.0 {
            /* must scale the values */
            imcomp_nullscalefloats_inplace(
                idata,
                tilelen,
                scale,
                zero,
                nullcheck,
                nullflagval,
                nullval,
                status,
            );
        } else {
            imcomp_nullfloats_inplace(
                cast_slice_mut(tiledata),
                tilelen,
                nullcheck,
                nullflagval,
                nullval,
                status,
            );
        }
    } else if (outfptr.Fptr).quantize_level == NO_QUANTIZE {
        /* just convert null values to NaNs in place, if necessary, then do lossless gzip compression */
        if nullcheck == NullCheckType::SetPixel {
            imcomp_float2nan_inplace(
                cast_slice_mut(tiledata),
                tilelen,
                nullflagval.unwrap(),
                status,
            );
        }
    }

    *status
}

/// Prepare the input tile array of pixels for compression.
/// Convert input double tile array in place to 4-byte ints for compression,
/// If needed, convert 4 or 8-byte ints and do null value substitution.
/// Note that the calling routine must have allocated the input array big enough
/// to be able to do this.
fn imcomp_convert_tile_tdouble(
    outfptr: &mut fitsfile,
    row: c_long,
    tiledata: &mut [u8],
    tilelen: c_long,
    tilenx: c_long,
    tileny: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<f64>,
    nullval: c_int,
    zbitpix: c_int,
    scale: f64,
    zero: f64,
    intlength: &mut c_int,
    flag: &mut c_int,
    bscale: &mut f64,
    bzero: &mut f64,
    status: &mut c_int,
) -> c_int {
    let mut irow: c_long = 0;
    let mut doublenull: f64 = 0.0;

    let mut dithersum: c_ulong = 0;
    let mut iminval: c_int = 0;
    let mut imaxval: c_int = 0; /* min and max quantized integers */

    /* datatype of input array is double.  We only support writing this datatype
    to a FITS image with BITPIX = -64 or -32, except we also support the
    special case where BITPIX = 32 and BZERO = 0 and BSCALE = 1.  */

    if (zbitpix != LONG_IMG && zbitpix != DOUBLE_IMG && zbitpix != FLOAT_IMG)
        || scale != 1.0
        || zero != 0.0
    {
        ffpmsg_cstr(
            c"Implicit datatype conversion is not supported when writing to compressed images",
        );
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    *intlength = 4;

    /* if the tile-compressed table contains zscale and zzero columns */
    /* then scale and quantize the input floating point data.    */
    /* Otherwise, just truncate the floats to integers.          */

    if (outfptr.Fptr).cn_zscale > 0 {
        if nullcheck == NullCheckType::SetPixel {
            doublenull = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
        } else {
            doublenull = DOUBLENULLVALUE;
        }

        /* quantize the double values into integers */
        if (outfptr.Fptr).quantize_method == SUBTRACTIVE_DITHER_1
            || (outfptr.Fptr).quantize_method == SUBTRACTIVE_DITHER_2
        {
            /* see if the dithering offset value needs to be initialized (see above) */

            if (outfptr.Fptr).request_dither_seed == 0 && (outfptr.Fptr).dither_seed == 0 {
                (outfptr.Fptr).dither_seed = fastrand::i32(1..=10000);

                /* update the header keyword with this new value. Capture the
                seed AFTER it is assigned; otherwise ZDITHER0 is written as 0,
                which makes the reader compute a negative random-sequence index
                (row = nrow + 0 - 1) and panic. */
                let dither_seed = (outfptr.Fptr).dither_seed;
                fits_update_key(
                    outfptr,
                    KeywordDatatype::TINT(&dither_seed),
                    cs!(c"ZDITHER0"),
                    None,
                    status,
                );
            } else if (outfptr.Fptr).request_dither_seed < 0 && (outfptr.Fptr).dither_seed < 0 {
                let usbbuff: &[c_uchar] = cast_slice(tiledata);
                dithersum = 0;
                for ii in 0..(8 * tilelen as usize) {
                    dithersum += c_ulong::from(usbbuff[ii]);
                }
                (outfptr.Fptr).dither_seed = ((dithersum % 10000) as c_int) + 1;

                /* update the header keyword with this new value */
                let dither_seed = (outfptr.Fptr).dither_seed;
                fits_update_key(
                    outfptr,
                    KeywordDatatype::TINT(&dither_seed),
                    cs!(c"ZDITHER0"),
                    None,
                    status,
                );
            }

            irow = row + c_long::from((outfptr.Fptr).dither_seed) - 1; /* dither the quantized values */
        } else if (outfptr.Fptr).quantize_method == -1 {
            irow = 0; /* do not dither the quantized values */
        } else {
            ffpmsg_str("Unknown subtractive dithering method.");
            ffpmsg_str("May need to install a newer version of CFITSIO.");
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }

        *flag = fits_quantize_double_inplace(
            irow as usize,
            cast_slice_mut(tiledata),
            tilenx as usize,
            tileny as usize,
            nullcheck != NullCheckType::None,
            doublenull,
            (outfptr.Fptr).quantize_level,
            DitherType::from((outfptr.Fptr).quantize_method),
            bscale,
            bzero,
            &mut iminval,
            &mut imaxval,
        );

        if *flag > 1 {
            *status = *flag;
            return *status;
        }
    } else if (outfptr.Fptr).quantize_level != NO_QUANTIZE {
        /* if floating point pixels are not being losslessly compressed, then */
        /* input float data is implicitly converted (truncated) to integers */

        let idata: &mut [c_int] = cast_slice_mut(tiledata);

        if scale != 1. || zero != 0.0 {
            /* must scale the values */
            imcomp_nullscaledoubles_inplace(
                idata,
                tilelen,
                scale,
                zero,
                nullcheck,
                nullflagval.unwrap(),
                nullval,
                status,
            );
        } else {
            imcomp_nulldoubles_inplace(idata, tilelen, nullcheck, nullflagval, nullval, status);
        }
    } else if (outfptr.Fptr).quantize_level == NO_QUANTIZE {
        /* just convert null values to NaNs in place, if necessary, then do lossless gzip compression */
        if nullcheck == NullCheckType::SetPixel {
            imcomp_double2nan_inplace(
                cast_slice_mut(tiledata),
                tilelen,
                nullflagval.unwrap(),
                status,
            );
        }
    }

    *status
}

/// Do null value substitution AND scaling of the integer array.
/// If array value = nullflagval, then set the value to nullval.
/// Otherwise, inverse scale the integer value.
fn imcomp_nullscale(
    idata: &mut [c_int],
    tilelen: c_long,
    nullflagval: c_int,
    nullval: c_int,
    scale: f64,
    zero: f64,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    for ii in 0..(tilelen as usize) {
        if idata[ii] == nullflagval {
            idata[ii] = nullval;
        } else {
            dvalue = (f64::from(idata[ii]) - zero) / scale;

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Do null value substitution.
/// If array value = nullflagval, then set the value to nullval.
fn imcomp_nullvalues(
    idata: &mut [c_int],
    tilelen: c_long,
    nullflagval: c_int,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    for ii in 0..(tilelen as usize) {
        if idata[ii] == nullflagval {
            idata[ii] = nullval;
        }
    }
    *status
}

/// Do inverse scaling the integer values.
fn imcomp_scalevalues(
    idata: &mut [c_int],
    tilelen: c_long,
    scale: f64,
    zero: f64,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    for ii in 0..(tilelen as usize) {
        dvalue = (f64::from(idata[ii]) - zero) / scale;

        if dvalue < DINT_MIN {
            *status = OVERFLOW_ERR;
            idata[ii] = INT32_MIN;
        } else if dvalue > DINT_MAX {
            *status = OVERFLOW_ERR;
            idata[ii] = INT32_MAX;
        } else if dvalue >= 0.0 {
            idata[ii] = (dvalue + 0.5) as c_int;
        } else {
            idata[ii] = (dvalue - 0.5) as c_int;
        }
    }
    *status
}

/// Do null value substitution AND scaling of the integer array.
/// If array value = nullflagval, then set the value to nullval.
/// Otherwise, inverse scale the integer value.
fn imcomp_nullscalei2(
    idata: &mut [c_short],
    tilelen: c_long,
    nullflagval: c_short,
    nullval: c_short,
    scale: f64,
    zero: f64,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    for ii in 0..(tilelen as usize) {
        if idata[ii] == nullflagval {
            idata[ii] = nullval;
        } else {
            dvalue = (f64::from(idata[ii]) - zero) / scale;

            if dvalue < DSHRT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = c_short::MIN;
            } else if dvalue > DSHRT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = c_short::MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_short;
            } else {
                idata[ii] = (dvalue - 0.5) as c_short;
            }
        }
    }
    *status
}

/// Do null value substitution.
/// If array value = nullflagval, then set the value to nullval.
fn imcomp_nullvaluesi2(
    idata: &mut [c_short],
    tilelen: c_long,
    nullflagval: c_short,
    nullval: c_short,
    status: &mut c_int,
) -> c_int {
    for ii in 0..(tilelen as usize) {
        if idata[ii] == nullflagval {
            idata[ii] = nullval;
        }
    }
    *status
}

/// Do inverse scaling the integer values.
fn imcomp_scalevaluesi2(
    idata: &mut [c_short],
    tilelen: c_long,
    scale: f64,
    zero: f64,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    for ii in 0..(tilelen as usize) {
        dvalue = (f64::from(idata[ii]) - zero) / scale;

        if dvalue < DSHRT_MIN {
            *status = OVERFLOW_ERR;
            idata[ii] = c_short::MIN;
        } else if dvalue > DSHRT_MAX {
            *status = OVERFLOW_ERR;
            idata[ii] = c_short::MAX;
        } else if dvalue >= 0.0 {
            idata[ii] = (dvalue + 0.5) as c_short;
        } else {
            idata[ii] = (dvalue - 0.5) as c_short;
        }
    }
    *status
}

/// Do null value substitution  of the float array.
/// If array value = nullflagval, then set the output value to FLOATNULLVALUE.
fn imcomp_nullfloats(
    fdata: &[f32],
    tilelen: c_long,
    idata: &mut [c_int],
    nullcheck: NullCheckType,
    nullflagval: f32,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::SetPixel {
        /* must check for null values */

        for ii in 0..(tilelen as usize) {
            if fdata[ii] == nullflagval {
                idata[ii] = nullval;
            } else {
                dvalue = f64::from(fdata[ii]);

                if dvalue < DINT_MIN {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MIN;
                } else if dvalue > DINT_MAX {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MAX;
                } else if dvalue >= 0.0 {
                    idata[ii] = (dvalue + 0.5) as c_int;
                } else {
                    idata[ii] = (dvalue - 0.5) as c_int;
                }
            }
        }
    } else {
        /* don't have to worry about null values */
        for ii in 0..(tilelen as usize) {
            dvalue = f64::from(fdata[ii]);

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Do null value substitution  of the float array.
/// If array value = nullflagval, then set the output value to FLOATNULLVALUE.
fn imcomp_nullfloats_inplace(
    idata: &mut [i32],
    tilelen: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<f32>,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::SetPixel {
        /* must check for null values */

        for ii in 0..(tilelen as usize) {
            let fdata_ii: f32 = cast(idata[ii]);
            let nullflagval: f32 = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
            if fdata_ii == nullflagval {
                idata[ii] = nullval;
            } else {
                dvalue = f64::from(fdata_ii);

                if dvalue < DINT_MIN {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MIN;
                } else if dvalue > DINT_MAX {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MAX;
                } else if dvalue >= 0.0 {
                    idata[ii] = (dvalue + 0.5) as c_int;
                } else {
                    idata[ii] = (dvalue - 0.5) as c_int;
                }
            }
        }
    } else {
        /* don't have to worry about null values */
        for ii in 0..(tilelen as usize) {
            let fdata_ii: f32 = cast(idata[ii]);
            dvalue = f64::from(fdata_ii);

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Do null value substitution  of the float array.
/// If array value = nullflagval, then set the output value to FLOATNULLVALUE.
/// Otherwise, inverse scale the integer value.
fn imcomp_nullscalefloats(
    fdata: &[f32],
    tilelen: c_long,
    idata: &mut [c_int],
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    nullflagval: f32,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::SetPixel {
        /* must check for null values */
        for ii in 0..(tilelen as usize) {
            if fdata[ii] == nullflagval {
                idata[ii] = nullval;
            } else {
                dvalue = (f64::from(fdata[ii]) - zero) / scale;

                if dvalue < DINT_MIN {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MIN;
                } else if dvalue > DINT_MAX {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MAX;
                } else if dvalue >= 0.0 {
                    idata[ii] = (dvalue + 0.5) as c_int;
                } else {
                    idata[ii] = (dvalue - 0.5) as c_int;
                }
            }
        }
    } else {
        /* don't have to worry about null values */

        for ii in 0..(tilelen as usize) {
            dvalue = (f64::from(fdata[ii]) - zero) / scale;

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Do null value substitution  of the float array.
/// If array value = nullflagval, then set the output value to FLOATNULLVALUE.
/// Otherwise, inverse scale the integer value.
fn imcomp_nullscalefloats_inplace(
    idata: &mut [i32],
    tilelen: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    nullflagval: Option<f32>,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::SetPixel {
        /* must check for null values */
        for ii in 0..(tilelen as usize) {
            let fdata_ii: f32 = cast(idata[ii]);
            let nullflagval: f32 = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel
            if fdata_ii == nullflagval {
                idata[ii] = nullval;
            } else {
                dvalue = (f64::from(fdata_ii) - zero) / scale;

                if dvalue < DINT_MIN {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MIN;
                } else if dvalue > DINT_MAX {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MAX;
                } else if dvalue >= 0.0 {
                    idata[ii] = (dvalue + 0.5) as c_int;
                } else {
                    idata[ii] = (dvalue - 0.5) as c_int;
                }
            }
        }
    } else {
        /* don't have to worry about null values */

        for ii in 0..(tilelen as usize) {
            let fdata_ii: f32 = cast(idata[ii]);
            dvalue = (f64::from(fdata_ii) - zero) / scale;

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Do null value substitution  of the float array.
/// If array value = nullflagval, then set the output value to FLOATNULLVALUE.
/// Otherwise, inverse scale the integer value.
fn imcomp_nulldoubles(
    fdata: &[f64],
    tilelen: c_long,
    idata: &mut [c_int],
    nullcheck: NullCheckType,
    nullflagval: f64,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::SetPixel {
        /* must check for null values */
        for ii in 0..(tilelen as usize) {
            if fdata[ii] == nullflagval {
                idata[ii] = nullval;
            } else {
                dvalue = fdata[ii];

                if dvalue < DINT_MIN {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MIN;
                } else if dvalue > DINT_MAX {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MAX;
                } else if dvalue >= 0.0 {
                    idata[ii] = (dvalue + 0.5) as c_int;
                } else {
                    idata[ii] = (dvalue - 0.5) as c_int;
                }
            }
        }
    } else {
        /* don't have to worry about null values */
        for ii in 0..(tilelen as usize) {
            dvalue = fdata[ii];

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Do null value substitution  of the float array.
/// If array value = nullflagval, then set the output value to FLOATNULLVALUE.
/// Otherwise, inverse scale the integer value.
fn imcomp_nulldoubles_inplace(
    idata: &mut [c_int],
    tilelen: c_long,
    nullcheck: NullCheckType,
    nullflagval: Option<f64>,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::SetPixel {
        let nullflagval: f64 = nullflagval.unwrap(); // Must be set if nullcheck is SetPixel

        /* must check for null values */
        for ii in 0..(tilelen as usize) {
            let ddata_ii: f64 = cast(idata[ii]);
            if ddata_ii == nullflagval {
                idata[ii] = nullval;
            } else {
                dvalue = ddata_ii;

                if dvalue < DINT_MIN {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MIN;
                } else if dvalue > DINT_MAX {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MAX;
                } else if dvalue >= 0.0 {
                    idata[ii] = (dvalue + 0.5) as c_int;
                } else {
                    idata[ii] = (dvalue - 0.5) as c_int;
                }
            }
        }
    } else {
        /* don't have to worry about null values */
        for ii in 0..(tilelen as usize) {
            dvalue = cast(idata[ii]);

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Do null value substitution  of the float array.
/// If array value = nullflagval, then set the output value to FLOATNULLVALUE.
/// Otherwise, inverse scale the integer value.
fn imcomp_nullscaledoubles(
    fdata: &[f64],
    tilelen: c_long,
    idata: &mut [c_int],
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    nullflagval: f64,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::SetPixel {
        /* must check for null values */

        for ii in 0..(tilelen as usize) {
            if fdata[ii] == nullflagval {
                idata[ii] = nullval;
            } else {
                dvalue = (fdata[ii] - zero) / scale;

                if dvalue < DINT_MIN {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MIN;
                } else if dvalue > DINT_MAX {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MAX;
                } else if dvalue >= 0.0 {
                    idata[ii] = (dvalue + 0.5) as c_int;
                } else {
                    idata[ii] = (dvalue - 0.5) as c_int;
                }
            }
        }
    } else {
        /* don't have to worry about null values */
        for ii in 0..(tilelen as usize) {
            dvalue = (fdata[ii] - zero) / scale;

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Do null value substitution  of the float array.
/// If array value = nullflagval, then set the output value to FLOATNULLVALUE.
/// Otherwise, inverse scale the integer value.
fn imcomp_nullscaledoubles_inplace(
    idata: &mut [c_int],
    tilelen: c_long,
    scale: f64,
    zero: f64,
    nullcheck: NullCheckType,
    nullflagval: f64,
    nullval: c_int,
    status: &mut c_int,
) -> c_int {
    let mut dvalue: f64 = 0.0;

    if nullcheck == NullCheckType::SetPixel {
        /* must check for null values */

        for ii in 0..(tilelen as usize) {
            let ddata_ii: f64 = cast(idata[ii]);
            if ddata_ii == nullflagval {
                idata[ii] = nullval;
            } else {
                dvalue = (ddata_ii - zero) / scale;

                if dvalue < DINT_MIN {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MIN;
                } else if dvalue > DINT_MAX {
                    *status = OVERFLOW_ERR;
                    idata[ii] = INT32_MAX;
                } else if dvalue >= 0.0 {
                    idata[ii] = (dvalue + 0.5) as c_int;
                } else {
                    idata[ii] = (dvalue - 0.5) as c_int;
                }
            }
        }
    } else {
        /* don't have to worry about null values */
        for ii in 0..(tilelen as usize) {
            let ddata_ii: f64 = cast(idata[ii]);
            dvalue = (ddata_ii - zero) / scale;

            if dvalue < DINT_MIN {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MIN;
            } else if dvalue > DINT_MAX {
                *status = OVERFLOW_ERR;
                idata[ii] = INT32_MAX;
            } else if dvalue >= 0.0 {
                idata[ii] = (dvalue + 0.5) as c_int;
            } else {
                idata[ii] = (dvalue - 0.5) as c_int;
            }
        }
    }
    *status
}

/// Write a section of a compressed image.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the array to be written
/// * `infpixel`  — (I) 'bottom left corner' of the subsection
/// * `inlpixel`  — (I) 'top right corner' of the subsection
/// * `nullcheck` — (I) 0 for no null checking
/// * `array`     — (I) array of values to be written
/// * `nullval`   — (I) undefined pixel value
/// * `status`    — (IO) error status
pub(crate) fn fits_write_compressed_img(
    fptr: &mut fitsfile,
    datatype: c_int,
    infpixel: &[c_long],
    inlpixel: &[c_long],
    nullcheck: NullCheckType,
    /*     1: pixels that are = nullval will be     */
    /*     written with the FITS null pixel value   */
    /*     (floating point arrays only)             */
    array: &[u8],
    nullval: &Option<NullValue>,
    status: &mut c_int,
) -> c_int {
    let mut tiledim: [c_int; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut naxis: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut thistilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ftile: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ltile: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tfpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tlpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut rowdim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut offset: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ntemp = 0;
    let mut fpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut lpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];

    let mut irow: c_long = 0;
    let mut trowsize: c_long = 0;
    let mut ntrows: c_long = 0;

    let mut ndim: c_int = 0;
    let mut pixlen: usize = 0;
    let mut tilenul: c_int = 0;
    let mut tstatus: c_int = 0;
    let mut buffpixsiz: usize = 0;

    let mut bnullarray: Vec<c_char> = Vec::new();
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    if *status > 0 {
        return *status;
    }

    if fits_is_compressed_image_safe(fptr, status) == 0 {
        ffpmsg_str("CHDU is not a compressed image (fits_write_compressed_img)");
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != (fptr.Fptr).curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }
    /* rescan header if data structure is undefined */
    else if (fptr.Fptr).datastart == DATA_UNDEFINED && ffrdef_safe(fptr, status) > 0 {
        return *status;
    }

    /* ===================================================================== */

    if datatype == TSHORT || datatype == TUSHORT {
        pixlen = mem::size_of::<c_short>();
    } else if datatype == TINT || datatype == TUINT {
        pixlen = mem::size_of::<c_int>();
    } else if datatype == TBYTE || datatype == TSBYTE {
        pixlen = 1;
    } else if datatype == TLONG || datatype == TULONG {
        pixlen = mem::size_of::<c_long>();
    } else if datatype == TFLOAT {
        pixlen = mem::size_of::<f32>();
    } else if datatype == TDOUBLE {
        pixlen = mem::size_of::<f64>();
    } else {
        ffpmsg_str("unsupported datatype for compressing image");
        *status = BAD_DATATYPE;
        return *status;
    }

    /* ===================================================================== */

    /* allocate scratch space for processing one tile of the image */
    buffpixsiz = pixlen; /* this is the minimum pixel size */

    if (fptr.Fptr).compress_type == HCOMPRESS_1 {
        /* need 4 or 8 bytes per pixel */
        if (fptr.Fptr).zbitpix == BYTE_IMG || (fptr.Fptr).zbitpix == SHORT_IMG {
            buffpixsiz = cmp::max(buffpixsiz, 4);
        } else {
            buffpixsiz = 8;
        }
    } else if (fptr.Fptr).compress_type == PLIO_1 {
        /* need 4 bytes per pixel */
        buffpixsiz = cmp::max(buffpixsiz, 4);
    } else if (fptr.Fptr).compress_type == RICE_1
        || (fptr.Fptr).compress_type == GZIP_1
        || (fptr.Fptr).compress_type == GZIP_2
        || (fptr.Fptr).compress_type == BZIP2_1
    {
        /* need 1, 2, or 4 bytes per pixel */
        if (fptr.Fptr).zbitpix == BYTE_IMG {
            buffpixsiz = cmp::max(buffpixsiz, 1);
        } else if (fptr.Fptr).zbitpix == SHORT_IMG {
            buffpixsiz = cmp::max(buffpixsiz, 2);
        } else {
            buffpixsiz = cmp::max(buffpixsiz, 4);
        }
    } else {
        ffpmsg_str("unsupported image compression algorithm");
        *status = BAD_DATATYPE;
        return *status;
    }

    /* cast to double to force alignment on 8-byte addresses */
    let buffersize = ((fptr.Fptr).maxtilelen as usize) * buffpixsiz;
    let buffersize_f64 = buffersize.div_ceil(8);

    let mut f64_buffer: Vec<f64> = Vec::new();
    if f64_buffer.try_reserve_exact(buffersize_f64).is_err() {
        ffpmsg_str("Out of memory (fits_write_compress_img)");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        f64_buffer.resize(buffersize_f64, 0.0);
    }

    let buffer: &mut [u8] = cast_slice_mut(&mut f64_buffer);

    /* ===================================================================== */

    /* initialize all the arrays */
    for ii in 0..MAX_COMPRESS_DIM {
        naxis[ii] = 1;
        tiledim[ii] = 1;
        tilesize[ii] = 1;
        ftile[ii] = 1;
        ltile[ii] = 1;
        rowdim[ii] = 1;
    }

    ndim = (fptr.Fptr).zndim;
    ntemp = 1;
    for ii in 0..(ndim as usize) {
        fpixel[ii] = infpixel[ii];
        lpixel[ii] = inlpixel[ii];

        /* calc number of tiles in each dimension, and tile containing */
        /* the first and last pixel we want to read in each dimension  */
        naxis[ii] = (fptr.Fptr).znaxis[ii];
        if fpixel[ii] < 1 {
            *status = BAD_PIX_NUM;
            return *status;
        }

        tilesize[ii] = (fptr.Fptr).tilesize[ii];
        tiledim[ii] = ((naxis[ii] as c_long - 1) / tilesize[ii] + 1) as c_int;
        ftile[ii] = (fpixel[ii] - 1) / tilesize[ii] + 1;
        ltile[ii] = cmp::min(
            (lpixel[ii] - 1) / tilesize[ii] + 1,
            c_long::from(tiledim[ii]),
        );
        rowdim[ii] = ntemp; /* total tiles in each dimension */
        ntemp *= c_long::from(tiledim[ii]);
    }

    /* support up to 6 dimensions for now */
    /* tfpixel and tlpixel are the first and last image pixels */
    /* along each dimension of the compression tile */
    for i5 in ftile[5]..=(ltile[5]) {
        tfpixel[5] = (i5 - 1) * tilesize[5] + 1;
        tlpixel[5] = cmp::min(tfpixel[5] + tilesize[5] - 1, naxis[5]);
        thistilesize[5] = tlpixel[5] - tfpixel[5] + 1;
        offset[5] = (i5 - 1) * rowdim[5];
        for i4 in ftile[4]..=(ltile[4]) {
            tfpixel[4] = (i4 - 1) * tilesize[4] + 1;
            tlpixel[4] = cmp::min(tfpixel[4] + tilesize[4] - 1, naxis[4]);
            thistilesize[4] = thistilesize[5] * (tlpixel[4] - tfpixel[4] + 1);
            offset[4] = (i4 - 1) * rowdim[4] + offset[5];
            for i3 in ftile[3]..=(ltile[3]) {
                tfpixel[3] = (i3 - 1) * tilesize[3] + 1;
                tlpixel[3] = cmp::min(tfpixel[3] + tilesize[3] - 1, naxis[3]);
                thistilesize[3] = thistilesize[4] * (tlpixel[3] - tfpixel[3] + 1);
                offset[3] = (i3 - 1) * rowdim[3] + offset[4];
                for i2 in ftile[2]..=(ltile[2]) {
                    tfpixel[2] = (i2 - 1) * tilesize[2] + 1;
                    tlpixel[2] = cmp::min(tfpixel[2] + tilesize[2] - 1, naxis[2]);
                    thistilesize[2] = thistilesize[3] * (tlpixel[2] - tfpixel[2] + 1);
                    offset[2] = (i2 - 1) * rowdim[2] + offset[3];
                    for i1 in ftile[1]..=(ltile[1]) {
                        tfpixel[1] = (i1 - 1) * tilesize[1] + 1;
                        tlpixel[1] = cmp::min(tfpixel[1] + tilesize[1] - 1, naxis[1]);
                        thistilesize[1] = thistilesize[2] * (tlpixel[1] - tfpixel[1] + 1);
                        offset[1] = (i1 - 1) * rowdim[1] + offset[2];
                        for i0 in ftile[0]..=(ltile[0]) {
                            tfpixel[0] = (i0 - 1) * tilesize[0] + 1;
                            tlpixel[0] = cmp::min(tfpixel[0] + tilesize[0] - 1, naxis[0]);
                            thistilesize[0] = thistilesize[1] * (tlpixel[0] - tfpixel[0] + 1);
                            /* calculate row of table containing this tile */
                            irow = i0 + offset[1];

                            /* read and uncompress this row (tile) of the table */
                            /* also do type conversion and undefined pixel substitution */
                            /* at this point */
                            imcomp_decompress_tile(
                                fptr,
                                irow as c_int,
                                thistilesize[0] as c_int,
                                datatype,
                                nullcheck,
                                nullval,
                                cast_slice_mut(buffer),
                                &mut bnullarray,
                                Some(&mut tilenul),
                                status,
                            );

                            if *status == NO_COMPRESSED_TILE {
                                /* tile doesn't exist, so initialize to zero */
                                buffer[..pixlen * thistilesize[0] as usize].fill(0);

                                *status = 0;
                            }

                            /* copy the intersecting pixels to this tile from the input */
                            imcomp_merge_overlap(
                                cast_slice_mut(buffer),
                                pixlen.try_into().unwrap(),
                                ndim,
                                &tfpixel,
                                &tlpixel,
                                &bnullarray,
                                cast_slice(array),
                                &fpixel,
                                &lpixel,
                                nullcheck,
                                status,
                            );

                            /* Collapse sizes of higher dimension tiles into 2
                            dimensional equivalents needed by the quantizing
                            algorithms for floating point types */
                            fits_calc_tile_rows(
                                &tlpixel,
                                &tfpixel,
                                ndim,
                                &mut trowsize,
                                &mut ntrows,
                                status,
                            );

                            /* compress the tile again, and write it back to the FITS file */
                            imcomp_compress_tile(
                                fptr,
                                irow,
                                datatype,
                                cast_slice_mut(buffer),
                                thistilesize[0],
                                trowsize,
                                ntrows,
                                nullcheck,
                                nullval,
                                status,
                            );
                        }
                    }
                }
            }
        }
    }

    if (fptr.Fptr).zbitpix < 0 && nullcheck != NullCheckType::None {
        /*
        This is a floating point FITS image with possible null values.
        It is too messy to test if any null values are actually written, so
        just assume so.  We need to make sure that the
        ZBLANK keyword is present in the compressed image header.  If it is
        not there then we need to insert the keyword.
        */
        tstatus = 0;
        ffgcrd_safe(fptr, cs!(c"ZBLANK"), &mut card, &mut tstatus);

        if tstatus != 0 {
            /* have to insert the ZBLANK keyword */
            ffgcrd_safe(fptr, cs!(c"ZCMPTYPE"), &mut card, status);
            ffikyj_safe(
                fptr,
                cs!(c"ZBLANK"),
                LONGLONG::from(COMPRESS_NULL_VALUE),
                Some(cs!(c"null value in the compressed integer array")),
                status,
            );

            /* set this value into the internal structure; it is used if */
            /* the program reads back the values from the array */

            (fptr.Fptr).zblank = COMPRESS_NULL_VALUE;
            (fptr.Fptr).cn_zblank = -1; /* flag for a constant ZBLANK */
        }
    }

    *status
}

/// Write a consecutive set of pixels to a compressed image.  This routine
/// interpretes the n-dimensional image as a long one-dimensional array.
/// This is actually a rather inconvenient way to write compressed images in
/// general, and could be rather inefficient if the requested pixels to be
/// written are located in many different image compression tiles.
///
/// The general strategy used here is to write the requested pixels in blocks
/// that correspond to rectangular image sections.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the array to be written
/// * `fpixel`    — (I) 'first pixel to write
/// * `npixel`    — (I) number of pixels to write
/// * `nullcheck` — (I) 0 for no null checking
/// * `array`     — (I) array of values to write
/// * `nullval`   — (I) value used to represent undefined pixels
/// * `status`    — (IO) error status
pub(crate) fn fits_write_compressed_pixels(
    fptr: &mut fitsfile,
    datatype: c_int,
    fpixel: LONGLONG,
    npixel: LONGLONG,
    nullcheck: NullCheckType,
    /*     1: pixels that are = nullval will be     */
    /*     written with the FITS null pixel value   */
    /*     (floating point arrays only)             */
    array: &[u8],
    nullval: &Option<NullValue>,
    status: &mut c_int,
) -> c_int {
    let mut naxis: c_int = 0;
    let mut bytesperpixel: c_int = 0;
    let mut naxes: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut nread: c_long = 0;
    let mut tfirst: LONGLONG = 0;
    let mut tlast: LONGLONG = 0;
    let mut last0: LONGLONG = 0;
    let mut last1: LONGLONG = 0;
    let mut dimsize: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut nplane: c_long = 0;
    let mut firstcoord: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut lastcoord: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut arrayptr: &[u8];

    if *status > 0 {
        return *status;
    }

    arrayptr = cast_slice(array);

    /* get size of array pixels, in bytes */
    bytesperpixel = ffpxsz(datatype) as c_int;

    for ii in 0..MAX_COMPRESS_DIM {
        naxes[ii] = 1;
        firstcoord[ii] = 0;
        lastcoord[ii] = 0;
    }

    /*  determine the dimensions of the image to be written */
    ffgidm_safe(fptr, &mut naxis, status);
    ffgisz_safe(fptr, MAX_COMPRESS_DIM as c_int, &mut naxes, status);

    /* calc the cumulative number of pixels in each successive dimension */
    dimsize[0] = 1;
    for ii in 1..MAX_COMPRESS_DIM {
        dimsize[ii] = dimsize[ii - 1] * naxes[ii - 1] as LONGLONG;
    }

    /*  determine the coordinate of the first and last pixel in the image */
    /*  Use zero based indexes here */
    tfirst = fpixel - 1;
    tlast = tfirst + npixel - 1;
    for ii in (0..(naxis as usize)).rev() {
        firstcoord[ii] = (tfirst / dimsize[ii]) as c_long;
        lastcoord[ii] = (tlast / dimsize[ii]) as c_long;
        tfirst -= firstcoord[ii] as LONGLONG * dimsize[ii];
        tlast -= lastcoord[ii] as LONGLONG * dimsize[ii];
    }

    /* to simplify things, treat 1-D, 2-D, and 3-D images as separate cases */

    if naxis == 1 {
        /* Simple: just write the requested range of pixels */

        firstcoord[0] += 1;
        lastcoord[0] += 1;
        fits_write_compressed_img(
            fptr,
            datatype,
            &firstcoord,
            &lastcoord,
            nullcheck,
            array,
            nullval,
            status,
        );
        return *status;
    } else if naxis == 2 {
        nplane = 0; /* write 1st (and only) plane of the image */
        fits_write_compressed_img_plane(
            fptr,
            datatype,
            bytesperpixel,
            nplane,
            &mut firstcoord,
            &lastcoord,
            &naxes,
            nullcheck,
            array,
            nullval,
            &mut nread,
            status,
        );
    } else if naxis == 3 {
        /* test for special case: writing an integral number of planes */
        if firstcoord[0] == 0
            && firstcoord[1] == 0
            && lastcoord[0] == naxes[0] - 1
            && lastcoord[1] == naxes[1] - 1
        {
            for ii in 0..MAX_COMPRESS_DIM {
                /* convert from zero base to 1 base */
                (firstcoord[ii]) += 1;
                (lastcoord[ii]) += 1;
            }

            /* we can write the contiguous block of pixels in one go */
            fits_write_compressed_img(
                fptr,
                datatype,
                &firstcoord,
                &lastcoord,
                nullcheck,
                array,
                nullval,
                status,
            );
            return *status;
        }

        /* save last coordinate in temporary variables */
        last0 = lastcoord[0] as LONGLONG;
        last1 = lastcoord[1] as LONGLONG;

        if firstcoord[2] < lastcoord[2] {
            /* we will write up to the last pixel in all but the last plane */
            lastcoord[0] = naxes[0] - 1;
            lastcoord[1] = naxes[1] - 1;
        }

        /* write one plane of the cube at a time, for simplicity */
        for nplane in firstcoord[2]..=lastcoord[2] {
            if nplane == lastcoord[2] {
                lastcoord[0] = last0 as c_long;
                lastcoord[1] = last1 as c_long;
            }

            fits_write_compressed_img_plane(
                fptr,
                datatype,
                bytesperpixel,
                nplane,
                &mut firstcoord,
                &lastcoord,
                &naxes,
                nullcheck,
                arrayptr,
                nullval,
                &mut nread,
                status,
            );

            /* for all subsequent planes, we start with the first pixel */
            firstcoord[0] = 0;
            firstcoord[1] = 0;

            /* increment pointers to next elements to be written */
            arrayptr = &arrayptr[(nread * c_long::from(bytesperpixel)) as usize..];
        }
    } else {
        ffpmsg_str("only 1D, 2D, or 3D images are currently supported");
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    *status
}

/// In general we have to write the first partial row of the image,
/// followed by the middle complete rows, followed by the last
/// partial row of the image.  If the first or last rows are complete,
/// then write them at the same time as all the middle rows.
///
/// # Parameters
///
/// * `fptr`          — (I) FITS file
/// * `datatype`      — (I) datatype of the array to be written
/// * `bytesperpixel` — (I) number of bytes per pixel in array
/// * `nplane`        — (I) which plane of the cube to write
/// * `firstcoord`    — I coordinate of first pixel to write
/// * `lastcoord`     — I coordinate of last pixel to write
/// * `naxes`         — I size of each image dimension
/// * `nullcheck`     — (I) 0 for no null checking
/// * `array`         — (I) array of values that are written
/// * `nullval`       — (I) value for undefined pixels
/// * `nread`         — (O) total number of pixels written
/// * `status`        — (IO) error status
fn fits_write_compressed_img_plane(
    fptr: &mut fitsfile,
    datatype: c_int,
    bytesperpixel: c_int,
    nplane: c_long,
    firstcoord: &mut [c_long],
    lastcoord: &[c_long],
    naxes: &[c_long],
    nullcheck: NullCheckType,
    /*     1: pixels that are = nullval will be     */
    /*     written with the FITS null pixel value   */
    /*     (floating point arrays only)             */
    array: &[u8],
    nullval: &Option<NullValue>,
    nread: &mut c_long,
    status: &mut c_int,
) -> c_int {
    /* bottom left coord. and top right coord. */
    let mut blc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut trc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];

    *nread = 0;

    let mut arrayptr: usize = 0;

    blc[2] = nplane + 1;
    trc[2] = nplane + 1;

    if firstcoord[0] != 0 {
        /* have to read a partial first row */
        blc[0] = firstcoord[0] + 1;
        blc[1] = firstcoord[1] + 1;
        trc[1] = blc[1];
        if lastcoord[1] == firstcoord[1] {
            trc[0] = lastcoord[0] + 1; /* 1st and last pixels in same row */
        } else {
            trc[0] = naxes[0]; /* read entire rest of the row */
        }

        fits_write_compressed_img(
            fptr,
            datatype,
            &blc,
            &trc,
            nullcheck,
            &array[arrayptr..],
            nullval,
            status,
        );

        *nread = *nread + trc[0] - blc[0] + 1;

        if lastcoord[1] == firstcoord[1] {
            return *status; /* finished */
        }

        /* set starting coord to beginning of next line */
        firstcoord[0] = 0;
        firstcoord[1] += 1;
        arrayptr += ((trc[0] - blc[0] + 1) * c_long::from(bytesperpixel)) as usize;
    }

    /* write contiguous complete rows of the image, if any */
    blc[0] = 1;
    blc[1] = firstcoord[1] + 1;
    trc[0] = naxes[0];

    if lastcoord[0] + 1 == naxes[0] {
        /* can write the last complete row, too */
        trc[1] = lastcoord[1] + 1;
    } else {
        /* last row is incomplete; have to read it separately */
        trc[1] = lastcoord[1];
    }

    if trc[1] >= blc[1] {
        /* must have at least one whole line to read */
        fits_write_compressed_img(
            fptr,
            datatype,
            &blc,
            &trc,
            nullcheck,
            &array[arrayptr..],
            nullval,
            status,
        );

        *nread += (trc[1] - blc[1] + 1) * naxes[0];

        if lastcoord[1] + 1 == trc[1] {
            return *status; /* finished */
        }

        /* increment pointers for the last partial row */
        arrayptr +=
            ((trc[1] - blc[1] + 1) * naxes[0] as c_long * c_long::from(bytesperpixel)) as usize;
    }

    if trc[1] == lastcoord[1] + 1 {
        return *status; /* all done */
    }

    /* set starting and ending coord to last line */

    trc[0] = lastcoord[0] + 1;
    trc[1] = lastcoord[1] + 1;
    blc[1] = trc[1];

    fits_write_compressed_img(
        fptr,
        datatype,
        &blc,
        &trc,
        nullcheck,
        &array[arrayptr..],
        nullval,
        status,
    );

    *nread = *nread + trc[0] - blc[0] + 1;

    *status
}

/* ######################################################################## */
/* ###                 Image Decompression Routines                     ### */
/* ######################################################################## */

/// This routine decompresses the whole image and writes it to the output file.
///
/// # Parameters
///
/// * `infptr`  — image (bintable) to uncompress
/// * `outfptr` — empty HDU for output uncompressed image
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_img_decompress(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_img_decompress_safe(infptr, outfptr, status)
    }
}

/// This routine decompresses the whole image and writes it to the output file.
///
/// # Parameters
///
/// * `infptr`  — image (bintable) to uncompress
/// * `outfptr` — empty HDU for output uncompressed image
/// * `status`  — (IO) error status
pub fn fits_img_decompress_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut datatype: c_int = 0;
    let mut nullcheck: NullCheckType = NullCheckType::None;
    let mut anynul: c_int = 0;
    let mut fpixel: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut lpixel: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut inc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut fnulval: f32 = 0.0;
    let mut dnulval: f64 = 0.0;

    if fits_img_decompress_header_safe(infptr, outfptr, status) > 0 {
        return *status;
    }

    /* force a rescan of the output header keywords, then reset the scaling */
    /* in case the BSCALE and BZERO keywords are present, so that the       */
    /* decompressed values won't be scaled when written to the output image */
    ffrdef_safe(outfptr, status);
    ffpscl_safe(outfptr, 1.0, 0.0, status);
    ffpscl_safe(infptr, 1.0, 0.0, status);

    /* initialize; no null checking is needed for integer images */
    nullcheck = NullCheckType::None;
    let mut nulladdr = NullValue::Float(fnulval);

    /* determine datatype for image */
    if (infptr.Fptr).zbitpix == BYTE_IMG {
        datatype = TBYTE;
    } else if (infptr.Fptr).zbitpix == SHORT_IMG {
        datatype = TSHORT;
    } else if (infptr.Fptr).zbitpix == LONG_IMG {
        datatype = TINT;
    } else if (infptr.Fptr).zbitpix == FLOAT_IMG {
        /* In the case of float images we must check for NaNs  */
        nullcheck = NullCheckType::SetPixel;
        fnulval = FLOATNULLVALUE;
        nulladdr = NullValue::Float(fnulval);
        datatype = TFLOAT;
    } else if (infptr.Fptr).zbitpix == DOUBLE_IMG {
        /* In the case of double images we must check for NaNs  */
        nullcheck = NullCheckType::SetPixel;
        dnulval = DOUBLENULLVALUE;
        nulladdr = NullValue::Double(dnulval);
        datatype = TDOUBLE;
    }

    /* calculate size of the image (in pixels) */
    // let mut imgsize = 1;
    for ii in 0..((infptr.Fptr).zndim as usize) {
        // imgsize *= (infptr.Fptr).znaxis[ii];
        fpixel[ii] = 1; /* Set first and last pixel to */
        lpixel[ii] = (infptr.Fptr).znaxis[ii] as LONGLONG; /* include the entire image. */
        inc[ii] = 1;
    }

    /* uncompress the input image and write to output image, one tile at a time
     */

    fits_read_write_compressed_img(
        infptr,
        datatype,
        &fpixel,
        &lpixel,
        &inc,
        nullcheck,
        &Some(nulladdr),
        Some(&mut anynul),
        outfptr,
        status,
    );

    *status
}

/// THIS IS AN OBSOLETE ROUTINE.  USE fits_img_decompress instead!!!
///
/// This routine decompresses the whole image and writes it to the output file.
///
/// # Parameters
///
/// * `infptr`  — image (bintable) to uncompress
/// * `outfptr` — empty HDU for output uncompressed image
/// * `status`  — (IO) error status
#[deprecated(note = "Use fits_img_decompress instead")]
#[cfg_attr(not(test), unsafe(no_mangle))]
pub unsafe extern "C" fn fits_decompress_img(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_decompress_img_safe(infptr, outfptr, status)
    }
}

/// THIS IS AN OBSOLETE ROUTINE.  USE fits_img_decompress instead!!!
///
/// This routine decompresses the whole image and writes it to the output file.
///
/// # Parameters
///
/// * `infptr`  — image (bintable) to uncompress
/// * `outfptr` — empty HDU for output uncompressed image
/// * `status`  — (IO) error status
fn fits_decompress_img_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut datatype: c_int = 0;
    let mut byte_per_pix: usize = 0;
    let mut anynul: c_int = 0;
    let mut fpixel: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut lpixel: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut inc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut imgsize: usize = 0;
    let mut memsize: usize = 0;
    let mut fnulval: f32 = 0.0;
    let mut dnulval: f64 = 0.0;

    if *status > 0 {
        return *status;
    }

    if fits_is_compressed_image_safe(infptr, status) == 0 {
        ffpmsg_str("CHDU is not a compressed image (fits_decompress_img)");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    /* create an empty output image with the correct dimensions */
    if ffcrim_safe(
        outfptr,
        (infptr.Fptr).zbitpix,
        (infptr.Fptr).zndim,
        &(infptr.Fptr).znaxis,
        status,
    ) > 0
    {
        ffpmsg_str("error creating output decompressed image HDU");
        return *status;
    }

    /* Copy the table header to the image header. */
    if imcomp_copy_imheader(infptr, outfptr, status) > 0 {
        ffpmsg_str("error copying header of compressed image");
        return *status;
    }

    /* force a rescan of the output header keywords, then reset the scaling */
    /* in case the BSCALE and BZERO keywords are present, so that the       */
    /* decompressed values won't be scaled when written to the output image */
    ffrdef_safe(outfptr, status);
    ffpscl_safe(outfptr, 1.0, 0.0, status);
    ffpscl_safe(infptr, 1.0, 0.0, status);

    /* initialize; no null checking is needed for integer images */
    let mut nullcheck = NullCheckType::None;
    let mut nulladdr = NullValue::Float(fnulval);

    /* determine datatype for image */
    if (infptr.Fptr).zbitpix == BYTE_IMG {
        datatype = TBYTE;
        byte_per_pix = 1;
    } else if (infptr.Fptr).zbitpix == SHORT_IMG {
        datatype = TSHORT;
        byte_per_pix = mem::size_of::<c_short>();
    } else if (infptr.Fptr).zbitpix == LONG_IMG {
        datatype = TINT;
        byte_per_pix = mem::size_of::<c_int>();
    } else if (infptr.Fptr).zbitpix == FLOAT_IMG {
        /* In the case of float images we must check for NaNs  */
        nullcheck = NullCheckType::SetPixel;
        fnulval = FLOATNULLVALUE;
        nulladdr = NullValue::Float(fnulval);
        datatype = TFLOAT;
        byte_per_pix = mem::size_of::<f32>();
    } else if (infptr.Fptr).zbitpix == DOUBLE_IMG {
        /* In the case of double images we must check for NaNs  */
        nullcheck = NullCheckType::SetPixel;
        dnulval = DOUBLENULLVALUE;
        nulladdr = NullValue::Double(dnulval);
        datatype = TDOUBLE;
        byte_per_pix = mem::size_of::<f64>();
    }

    /* calculate size of the image (in pixels) */
    imgsize = 1;
    for ii in 0..((infptr.Fptr).zndim as usize) {
        imgsize *= (infptr.Fptr).znaxis[ii] as usize;
        fpixel[ii] = 1; /* Set first and last pixel to */
        lpixel[ii] = (infptr.Fptr).znaxis[ii] as LONGLONG; /* include the entire image. */
        inc[ii] = 1;
    }
    /* Calc equivalent number of double pixels same size as whole the image. */
    /* We use double datatype to force the memory to be aligned properly */
    memsize = ((imgsize * byte_per_pix) - 1) / mem::size_of::<f64>() + 1;

    /* allocate memory for the image */
    let mut data: Vec<f64> = Vec::new();
    if data.try_reserve_exact(memsize).is_err() {
        ffpmsg_str("Couldn't allocate memory for the uncompressed image");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        data.resize(memsize, 0.0);
    }

    /* uncompress the entire image into memory */
    /* This routine should be enhanced sometime to only need enough */
    /* memory to uncompress one tile at a time.  */
    fits_read_compressed_img(
        infptr,
        datatype,
        &fpixel,
        &lpixel,
        &inc,
        nullcheck,
        &Some(nulladdr.clone()),
        cast_slice_mut(&mut data),
        None,
        Some(&mut anynul),
        status,
    );

    /* write the image to the output file */
    if anynul != 0 {
        fits_write_imgnull(
            outfptr,
            datatype,
            1,
            imgsize as LONGLONG,
            cast_slice(&data),
            Some(nulladdr.clone()),
            status,
        );
    } else {
        fits_write_img(
            outfptr,
            datatype,
            1,
            imgsize as LONGLONG,
            cast_slice(&data),
            status,
        );
    }

    *status
}

/// This routine reads the header of the input tile compressed image and
/// converts it to that of a standard uncompress FITS image.
///
/// # Parameters
///
/// * `infptr`  — image (bintable) to uncompress
/// * `outfptr` — empty HDU for output uncompressed image
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_img_decompress_header(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_img_decompress_header_safe(infptr, outfptr, status)
    }
}

/// This routine reads the header of the input tile compressed image and
/// converts it to that of a standard uncompress FITS image.
///
/// # Parameters
///
/// * `infptr`  — image (bintable) to uncompress
/// * `outfptr` — empty HDU for output uncompressed image
/// * `status`  — (IO) error status
pub fn fits_img_decompress_header_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut writeprime = false;
    let mut hdupos: c_int = 0;
    let mut inhdupos: c_int = 0;
    let mut numkeys: c_int = 0;
    let mut nullprime = false;
    let mut copyprime = false;
    let mut norec = false;
    let mut tstatus: c_int = 0;
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut naxis: c_int = 0;
    let mut bitpix: c_int = 0;
    let mut naxes: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];

    if *status > 0 {
        return *status;
    } else if *status == -1 {
        *status = 0;
        writeprime = true;
    }

    if fits_is_compressed_image_safe(infptr, status) == 0 {
        ffpmsg_str("CHDU is not a compressed image (fits_img_decompress)");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    /* get information about the state of the output file; does it already */
    /* contain any keywords and HDUs?  */
    fits_get_hdu_num(infptr, &mut inhdupos); /* Get the current output HDU position */
    fits_get_hdu_num(outfptr, &mut hdupos); /* Get the current output HDU position */
    fits_get_hdrspace(outfptr, Some(&mut numkeys), None, status);

    /* Was the input compressed HDU originally the primary array image? */
    tstatus = 0;
    if fits_read_card(infptr, cs!(c"ZSIMPLE"), &mut card, &mut tstatus) == 0 {
        /* yes, input HDU was a primary array (not an IMAGE extension) */
        /* Now determine if we can uncompress it into the primary array of */
        /* the output file.  This is only possible if the output file */
        /* currently only contains a null primary array, with no addition */
        /* header keywords and with no following extension in the FITS file. */

        if hdupos == 1 {
            /* are we positioned at the primary array? */
            if numkeys == 0 {
                /* primary HDU is completely empty */
                nullprime = true;
            } else {
                fits_get_img_param(
                    outfptr,
                    MAX_COMPRESS_DIM as c_int,
                    Some(&mut bitpix),
                    Some(&mut naxis),
                    Some(&mut naxes),
                    status,
                );

                if naxis == 0 {
                    /* is this a null image? */
                    nullprime = true;

                    if inhdupos == 2 {
                        /* must be at the first extension */
                        copyprime = true;
                    }
                }
            }
        }
    }

    if nullprime {
        /* We will delete the existing keywords in the null primary array
        and uncompress the input image into the primary array of the output.
        Some of these keywords may be added back to the uncompressed image
        header later.
        */

        /* Backwards: deleting a record shifts the later ones down, so a
        forward loop walks off the end. */
        for ii in (1..=numkeys).rev() {
            fits_delete_record(outfptr, ii, status);
        }
    } else {
        /* if the ZTENSION keyword doesn't exist, then we have to
        write the required keywords manually */
        tstatus = 0;
        if fits_read_card(infptr, cs!(c"ZTENSION"), &mut card, &mut tstatus) != 0 {
            /* create an empty output image with the correct dimensions */
            if ffcrim_safe(
                outfptr,
                (infptr.Fptr).zbitpix,
                (infptr.Fptr).zndim,
                &(infptr.Fptr).znaxis,
                status,
            ) > 0
            {
                ffpmsg_str("error creating output decompressed image HDU");
                return *status;
            }

            norec = true; /* the required keywords have already been written */
        } else {
            /* the input compressed image does have ZTENSION keyword */

            if writeprime {
                /* convert the image extension to a primary array */
                /* have to write the required keywords manually */

                /* create an empty output image with the correct dimensions */
                if ffcrim_safe(
                    outfptr,
                    (infptr.Fptr).zbitpix,
                    (infptr.Fptr).zndim,
                    &(infptr.Fptr).znaxis,
                    status,
                ) > 0
                {
                    ffpmsg_str("error creating output decompressed image HDU");
                    return *status;
                }

                norec = true; /* the required keywords have already been written */
            } else {
                /* write the input compressed image to an image extension */

                if numkeys == 0 {
                    /* the output file is currently completely empty */

                    /* In this case, the input is a compressed IMAGE extension. */
                    /* Since the uncompressed output file is currently completely empty, */
                    /* we need to write a null primary array before uncompressing the */
                    /* image extension */

                    ffcrim_safe(outfptr, 8, 0, &naxes, status); /* naxes is not used */

                    /* now create the empty extension to uncompress into */
                    if fits_create_hdu(outfptr, status) > 0 {
                        ffpmsg_str("error creating output decompressed image HDU");
                        return *status;
                    }
                } else {
                    /* just create a new empty extension, then copy all the required */
                    /* keywords into it.  */
                    fits_create_hdu(outfptr, status);
                }
            }
        }
    }

    if *status > 0 {
        ffpmsg_str("error creating output decompressed image HDU");
        return *status;
    }

    /* Copy the table header to the image header. */

    if imcomp_copy_comp2img(infptr, outfptr, norec, status) > 0 {
        ffpmsg_str("error copying header keywords from compressed image");
    }

    if copyprime {
        /* append any unexpected keywords from the primary array.
        This includes any keywords except SIMPLE, BITPIX, NAXIS,
        EXTEND, COMMENT, HISTORY, CHECKSUM, and DATASUM.
        */

        fits_movabs_hdu(infptr, 1, None, status); /* move to primary array */

        /* do this so that any new keywords get written before any blank
        keywords that may have been appended by imcomp_copy_comp2img  */
        fits_set_hdustruc(outfptr, status);

        if imcomp_copy_prime2img(infptr, outfptr, status) > 0 {
            ffpmsg_str("error copying primary keywords from compressed file");
        }

        fits_movabs_hdu(infptr, 2, None, status); /* move back to where we were */
    }

    *status
}

/// Read a section of a compressed image;  Note: lpixel may be larger than the
/// size of the uncompressed image.  Only the pixels within the image will be
/// returned.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the array to be returned
/// * `infpixel`  — (I) 'bottom left corner' of the subsection
/// * `inlpixel`  — (I) 'top right corner' of the subsection
/// * `ininc`     — (I) increment to be applied in each dimension
/// * `nullcheck` — (I) 0 for no null checking
/// * `nullval`   — (I) value for undefined pixels
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) array of flags = 1 if nullcheck = 2
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub(crate) fn fits_read_compressed_img(
    fptr: &mut fitsfile,
    datatype: c_int,
    infpixel: &[LONGLONG],
    inlpixel: &[LONGLONG],
    ininc: &[c_long],
    mut nullcheck: NullCheckType,
    /*     1: set undefined pixels = nullval       */
    /*     2: set nullarray=1 for undefined pixels */
    nullval: &Option<NullValue>,
    array: &mut [u8],
    mut nullarray: Option<&mut [c_char]>,
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut naxis: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tiledim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut thistilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ftile: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ltile: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tfpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tlpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut rowdim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut offset: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ntemp: c_long = 0;
    let mut fpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut lpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut inc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];

    let mut irow: c_long = 0;
    let mut ndim: c_int = 0;
    let mut pixlen: usize = 0;
    let mut tilenul: c_int = 0;
    let mut testnullval: f64 = 0.0;

    let mut buffer = AlignedBytes::new();
    let mut buffer_len = 0;

    let mut bnullarray: Vec<c_char> = Vec::new();

    if *status > 0 {
        return *status;
    }

    if fits_is_compressed_image_safe(fptr, status) == 0 {
        ffpmsg_str("CHDU is not a compressed image (fits_read_compressed_img)");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    /* get temporary space for uncompressing one image tile */
    if datatype == TSHORT {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<c_short>();
        pixlen = mem::size_of::<c_short>();
    } else if datatype == TINT {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<c_int>();
        pixlen = mem::size_of::<c_int>();
    } else if datatype == TLONG {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<c_long>();
        pixlen = mem::size_of::<c_long>();
    } else if datatype == TFLOAT {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<f32>();
        pixlen = mem::size_of::<f32>();
    } else if datatype == TDOUBLE {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<f64>();
        pixlen = mem::size_of::<f64>();
    } else if datatype == TUSHORT {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<c_ushort>();
        pixlen = mem::size_of::<c_short>();
    } else if datatype == TUINT {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<c_uint>();
        pixlen = mem::size_of::<c_int>();
    } else if datatype == TULONG {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<c_ulong>();
        pixlen = mem::size_of::<c_long>();
    } else if datatype == TBYTE || datatype == TSBYTE {
        buffer_len = (fptr.Fptr.maxtilelen as usize) * mem::size_of::<c_char>();
        pixlen = 1;
    } else {
        ffpmsg_str("unsupported datatype for uncompressing image");
        *status = BAD_DATATYPE;
        return *status;
    }

    if let Some(nullval) = nullval {
        testnullval = nullval.get_value_as_f64();
    }

    /* If nullcheck ==1 and nullval == 0, then this means that the */
    /* calling routine does not want to check for null pixels in the array */
    if nullcheck == NullCheckType::SetPixel && testnullval == 0.0 {
        nullcheck = NullCheckType::None;
    }

    if buffer.try_resize_zeroed(buffer_len).is_err() {
        ffpmsg_str("Out of memory (fits_read_compress_img)");
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    /* allocate memory for a null flag array, if needed */
    if nullcheck == NullCheckType::SetNullArray {
        if bnullarray
            .try_reserve_exact(fptr.Fptr.maxtilelen as usize)
            .is_err()
        {
            ffpmsg_str("Out of memory (fits_read_compress_img)");
            *status = MEMORY_ALLOCATION;
            return *status;
        } else {
            bnullarray.resize(fptr.Fptr.maxtilelen as usize, 0);
        }
    }

    /* initialize all the arrays */
    for ii in 0..MAX_COMPRESS_DIM {
        naxis[ii] = 1;
        tiledim[ii] = 1;
        tilesize[ii] = 1;
        ftile[ii] = 1;
        ltile[ii] = 1;
        rowdim[ii] = 1;
    }

    ndim = fptr.Fptr.zndim;
    ntemp = 1;

    for ii in 0..(ndim as usize) {
        /* support for mirror-reversed image sections */
        if infpixel[ii] <= inlpixel[ii] {
            fpixel[ii] = infpixel[ii] as c_long;
            lpixel[ii] = inlpixel[ii] as c_long;
            inc[ii] = ininc[ii];
        } else {
            fpixel[ii] = inlpixel[ii] as c_long;
            lpixel[ii] = infpixel[ii] as c_long;
            inc[ii] = -ininc[ii];
        }

        /* calc number of tiles in each dimension, and tile containing */
        /* the first and last pixel we want to read in each dimension  */
        naxis[ii] = fptr.Fptr.znaxis[ii];
        if fpixel[ii] < 1 {
            if nullcheck == NullCheckType::SetNullArray {
                bnullarray.clear();
            }
            *status = BAD_PIX_NUM;
            return *status;
        }

        tilesize[ii] = fptr.Fptr.tilesize[ii];
        tiledim[ii] = (naxis[ii] - 1) / tilesize[ii] + 1;
        ftile[ii] = (fpixel[ii] - 1) / tilesize[ii] + 1;
        ltile[ii] = cmp::min((lpixel[ii] - 1) / tilesize[ii] + 1, tiledim[ii]);
        rowdim[ii] = ntemp; /* total tiles in each dimension */
        ntemp *= tiledim[ii];
    }

    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = 0;
    } /* initialize */

    /* support up to 6 dimensions for now */
    /* tfpixel and tlpixel are the first and last image pixels */
    /* along each dimension of the compression tile */

    for i5 in ftile[5]..=ltile[5] {
        tfpixel[5] = (i5 - 1) * tilesize[5] + 1;
        tlpixel[5] = cmp::min(tfpixel[5] + tilesize[5] - 1, naxis[5]);
        thistilesize[5] = tlpixel[5] - tfpixel[5] + 1;
        offset[5] = (i5 - 1) * rowdim[5];

        for i4 in ftile[4]..=ltile[4] {
            tfpixel[4] = (i4 - 1) * tilesize[4] + 1;
            tlpixel[4] = cmp::min(tfpixel[4] + tilesize[4] - 1, naxis[4]);
            thistilesize[4] = thistilesize[5] * (tlpixel[4] - tfpixel[4] + 1);
            offset[4] = (i4 - 1) * rowdim[4] + offset[5];

            for i3 in ftile[3]..=ltile[3] {
                tfpixel[3] = (i3 - 1) * tilesize[3] + 1;
                tlpixel[3] = cmp::min(tfpixel[3] + tilesize[3] - 1, naxis[3]);
                thistilesize[3] = thistilesize[4] * (tlpixel[3] - tfpixel[3] + 1);
                offset[3] = (i3 - 1) * rowdim[3] + offset[4];

                for i2 in ftile[2]..=ltile[2] {
                    tfpixel[2] = (i2 - 1) * tilesize[2] + 1;
                    tlpixel[2] = cmp::min(tfpixel[2] + tilesize[2] - 1, naxis[2]);
                    thistilesize[2] = thistilesize[3] * (tlpixel[2] - tfpixel[2] + 1);
                    offset[2] = (i2 - 1) * rowdim[2] + offset[3];

                    for i1 in ftile[1]..=ltile[1] {
                        tfpixel[1] = (i1 - 1) * tilesize[1] + 1;
                        tlpixel[1] = cmp::min(tfpixel[1] + tilesize[1] - 1, naxis[1]);
                        thistilesize[1] = thistilesize[2] * (tlpixel[1] - tfpixel[1] + 1);
                        offset[1] = (i1 - 1) * rowdim[1] + offset[2];

                        for i0 in ftile[0]..=ltile[0] {
                            tfpixel[0] = (i0 - 1) * tilesize[0] + 1;
                            tlpixel[0] = cmp::min(tfpixel[0] + tilesize[0] - 1, naxis[0]);
                            thistilesize[0] = thistilesize[1] * (tlpixel[0] - tfpixel[0] + 1);
                            /* calculate row of table containing this tile */
                            irow = i0 + offset[1];

                            /*
                            println!("row {}, {} {}, {} {}, {} {}; {}",irow, tfpixel[0],tlpixel[0],tfpixel[1],tlpixel[1],tfpixel[2],tlpixel[2],thistilesize[0]);
                            */
                            /* test if there are any intersecting pixels in this tile and the output image */
                            if imcomp_test_overlap(
                                ndim, &tfpixel, &tlpixel, &fpixel, &lpixel, &inc, status,
                            ) != 0
                            {
                                /* read and uncompress this row (tile) of the table */
                                /* also do type conversion and undefined pixel substitution */
                                /* at this point */

                                imcomp_decompress_tile(
                                    fptr,
                                    irow as c_int,
                                    thistilesize[0] as c_int,
                                    datatype,
                                    nullcheck,
                                    nullval,
                                    cast_slice_mut(&mut buffer),
                                    cast_slice_mut(&mut bnullarray),
                                    Some(&mut tilenul),
                                    status,
                                );

                                if tilenul != 0
                                    && let Some(anynul) = anynul.as_deref_mut()
                                {
                                    *anynul = 1; /* there are null pixels */
                                }
                                /*
                                println!(" pixlen={}, ndim={}, {} {} {}, {} {} {}, {} {} {}\n",pixlen, ndim, fpixel[0],lpixel[0],inc[0],fpixel[1],lpixel[1],inc[1],fpixel[2],lpixel[2],inc[2]);
                                */

                                /* copy the intersecting pixels from this tile to the output */
                                imcomp_copy_overlap(
                                    cast_slice_mut(&mut buffer),
                                    pixlen as c_int,
                                    ndim,
                                    &tfpixel,
                                    &tlpixel,
                                    &bnullarray,
                                    cast_slice_mut(array),
                                    &fpixel,
                                    &lpixel,
                                    &inc,
                                    nullcheck,
                                    nullarray.as_deref_mut(),
                                    status,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    *status
}

/// This is similar to fits_read_compressed_img, except that it writes
/// the pixels to the output image, on a tile by tile basis instead of returning
/// the array.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the array to be returned
/// * `infpixel`  — (I) 'bottom left corner' of the subsection
/// * `inlpixel`  — (I) 'top right corner' of the subsection
/// * `ininc`     — (I) increment to be applied in each dimension
/// * `nullcheck` — (I) 0 for no null checking, 1: set undefined pixels = nullval
/// * `nullval`   — (I) value for undefined pixels
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `outfptr`   — (I) FITS file pointer
/// * `status`    — (IO) error status
fn fits_read_write_compressed_img(
    fptr: &mut fitsfile,
    datatype: c_int,
    infpixel: &[LONGLONG],
    inlpixel: &[LONGLONG],
    ininc: &[c_long],
    mut nullcheck: NullCheckType,
    nullval: &Option<NullValue>,
    mut anynul: Option<&mut c_int>,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut naxis: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tiledim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut thistilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ftile: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ltile: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tfpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tlpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut rowdim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut offset: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut ntemp: c_long = 0;
    let mut fpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut lpixel: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut inc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];

    let mut irow: c_long = 0;
    let mut ndim: c_int = 0;
    let mut tilenul: c_int = 0;
    let mut buffer = AlignedBytes::new();
    let mut bnullarray: Vec<c_char> = Vec::new();
    let mut firstelem: LONGLONG = 0;

    if *status > 0 {
        return *status;
    }

    if fits_is_compressed_image_safe(fptr, status) == 0 {
        ffpmsg_str("CHDU is not a compressed image (fits_read_compressed_img)");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    let cnull = nullval; /* used to test if the nullval = 0 */

    /* get temporary space for uncompressing one image tile */
    /* If nullval == 0, then this means that the */
    /* calling routine does not want to check for null pixels in the array */
    let mut buffer_len = 0;
    if datatype == TSHORT {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<c_short>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else if datatype == TINT {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<c_int>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else if datatype == TLONG {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<c_long>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else if datatype == TFLOAT {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<f32>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else if datatype == TDOUBLE {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<f64>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else if datatype == TUSHORT {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<c_ushort>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else if datatype == TUINT {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<c_uint>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else if datatype == TULONG {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<c_ulong>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else if datatype == TBYTE || datatype == TSBYTE {
        buffer_len = ((fptr.Fptr).maxtilelen as usize) * mem::size_of::<c_char>();
        if let Some(cnull) = cnull
            && cnull.get_value_as_f64() == 0.0
        {
            nullcheck = NullCheckType::None;
        }
    } else {
        ffpmsg_str("unsupported datatype for uncompressing image");
        *status = BAD_DATATYPE;
        return *status;
    }

    if buffer.try_resize_zeroed(buffer_len).is_err() {
        ffpmsg_str("Out of memory (fits_read_compress_img)");
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    /* initialize all the arrays */
    for ii in 0..MAX_COMPRESS_DIM {
        naxis[ii] = 1;
        tiledim[ii] = 1;
        tilesize[ii] = 1;
        ftile[ii] = 1;
        ltile[ii] = 1;
        rowdim[ii] = 1;
    }

    ndim = (fptr.Fptr).zndim;
    ntemp = 1;
    for ii in 0..(ndim as usize) {
        /* support for mirror-reversed image sections */
        if infpixel[ii] <= inlpixel[ii] {
            fpixel[ii] = infpixel[ii] as c_long;
            lpixel[ii] = inlpixel[ii] as c_long;
            inc[ii] = ininc[ii];
        } else {
            fpixel[ii] = inlpixel[ii] as c_long;
            lpixel[ii] = infpixel[ii] as c_long;
            inc[ii] = -ininc[ii];
        }

        /* calc number of tiles in each dimension, and tile containing */
        /* the first and last pixel we want to read in each dimension  */
        naxis[ii] = (fptr.Fptr).znaxis[ii];
        if fpixel[ii] < 1 {
            *status = BAD_PIX_NUM;
            return *status;
        }

        tilesize[ii] = (fptr.Fptr).tilesize[ii];
        tiledim[ii] = (naxis[ii] - 1) / tilesize[ii] + 1;
        ftile[ii] = (fpixel[ii] - 1) / tilesize[ii] + 1;
        ltile[ii] = cmp::min((lpixel[ii] - 1) / tilesize[ii] + 1, tiledim[ii]);
        rowdim[ii] = ntemp; /* total tiles in each dimension */
        ntemp *= tiledim[ii];
    }

    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = 0;
    } /* initialize */

    firstelem = 1;

    /* support up to 6 dimensions for now */
    /* tfpixel and tlpixel are the first and last image pixels */
    /* along each dimension of the compression tile */
    for i5 in ftile[5]..=ltile[5] {
        tfpixel[5] = (i5 - 1) * tilesize[5] + 1;
        tlpixel[5] = cmp::min(tfpixel[5] + tilesize[5] - 1, naxis[5]);
        thistilesize[5] = tlpixel[5] - tfpixel[5] + 1;
        offset[5] = (i5 - 1) * rowdim[5];
        for i4 in ftile[4]..=ltile[4] {
            tfpixel[4] = (i4 - 1) * tilesize[4] + 1;
            tlpixel[4] = cmp::min(tfpixel[4] + tilesize[4] - 1, naxis[4]);
            thistilesize[4] = thistilesize[5] * (tlpixel[4] - tfpixel[4] + 1);
            offset[4] = (i4 - 1) * rowdim[4] + offset[5];
            for i3 in ftile[3]..=ltile[3] {
                tfpixel[3] = (i3 - 1) * tilesize[3] + 1;
                tlpixel[3] = cmp::min(tfpixel[3] + tilesize[3] - 1, naxis[3]);
                thistilesize[3] = thistilesize[4] * (tlpixel[3] - tfpixel[3] + 1);
                offset[3] = (i3 - 1) * rowdim[3] + offset[4];
                for i2 in ftile[2]..=ltile[2] {
                    tfpixel[2] = (i2 - 1) * tilesize[2] + 1;
                    tlpixel[2] = cmp::min(tfpixel[2] + tilesize[2] - 1, naxis[2]);
                    thistilesize[2] = thistilesize[3] * (tlpixel[2] - tfpixel[2] + 1);
                    offset[2] = (i2 - 1) * rowdim[2] + offset[3];
                    for i1 in ftile[1]..=ltile[1] {
                        tfpixel[1] = (i1 - 1) * tilesize[1] + 1;
                        tlpixel[1] = cmp::min(tfpixel[1] + tilesize[1] - 1, naxis[1]);
                        thistilesize[1] = thistilesize[2] * (tlpixel[1] - tfpixel[1] + 1);
                        offset[1] = (i1 - 1) * rowdim[1] + offset[2];
                        for i0 in ftile[0]..=ltile[0] {
                            tfpixel[0] = (i0 - 1) * tilesize[0] + 1;
                            tlpixel[0] = cmp::min(tfpixel[0] + tilesize[0] - 1, naxis[0]);
                            thistilesize[0] = thistilesize[1] * (tlpixel[0] - tfpixel[0] + 1);
                            /* calculate row of table containing this tile */
                            irow = i0 + offset[1];

                            /* read and uncompress this row (tile) of the table */
                            /* also do type conversion and undefined pixel substitution */
                            /* at this point */

                            imcomp_decompress_tile(
                                fptr,
                                irow as c_int,
                                thistilesize[0] as c_int,
                                datatype,
                                nullcheck,
                                nullval,
                                &mut buffer,
                                &mut bnullarray,
                                Some(&mut tilenul),
                                status,
                            );

                            /* write the image to the output file */

                            if tilenul != 0 && anynul.is_some() {
                                /* this assumes that the tiled pixels are in the
                                   same order as in the uncompressed FITS image.
                                   This is not necessarily the case, but it
                                   almost alway is in practice. Note that null
                                   checking is not performed for integer images,
                                   so this could only be a problem for tile
                                   compressed floating point images that use an
                                   unconventional tiling pattern.
                                */
                                fits_write_imgnull(
                                    outfptr,
                                    datatype,
                                    firstelem,
                                    thistilesize[0] as LONGLONG,
                                    &buffer,
                                    nullval.clone(),
                                    status,
                                );
                            } else {
                                fits_write_subset(
                                    outfptr, datatype, &tfpixel, &tlpixel, &buffer, status,
                                );
                            }

                            firstelem += thistilesize[0] as LONGLONG;
                        }
                    }
                }
            }
        }
    }

    *status
}

/// Read a consecutive set of pixels from a compressed image.  This routine
/// interpretes the n-dimensional image as a long one-dimensional array.
/// This is actually a rather inconvenient way to read compressed images in
/// general, and could be rather inefficient if the requested pixels to be
/// read are located in many different image compression tiles.
///
/// The general strategy used here is to read the requested pixels in blocks
/// that correspond to rectangular image sections.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `datatype`  — (I) datatype of the array to be returned
/// * `fpixel`    — (I) 'first pixel to read
/// * `npixel`    — (I) number of pixels to read
/// * `nullcheck` — (I) 0 for no null checking
/// * `nullval`   — (I) value for undefined pixels
/// * `array`     — (O) array of values that are returned
/// * `nullarray` — (O) array of flags = 1 if nullcheck = 2
/// * `anynul`    — (O) set to 1 if any values are null; else 0
/// * `status`    — (IO) error status
pub(crate) fn fits_read_compressed_pixels(
    fptr: &mut fitsfile,
    datatype: c_int,
    fpixel: LONGLONG,
    npixel: LONGLONG,
    nullcheck: NullCheckType,
    /*     1: set undefined pixels = nullval       */
    /*     2: set nullarray=1 for undefined pixels */
    nullval: &Option<NullValue>,
    array: &mut [u8],
    mut nullarray: Option<&mut [c_char]>,
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut naxis: c_int = 0;
    let mut bytesperpixel: usize = 0;
    let mut planenul: c_int = 0;
    let mut naxes: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut nread: c_long = 0;
    let mut nplane: c_long = 0;
    let mut inc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tfirst: LONGLONG = 0;
    let mut tlast: LONGLONG = 0;
    let mut last0: LONGLONG = 0;
    let mut last1: LONGLONG = 0;
    let mut dimsize: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut firstcoord: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut lastcoord: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];

    if *status > 0 {
        return *status;
    }

    let mut arrayptr = 0; // array
    let mut nullarrayptr = 0; //nullarray

    /* get size of array pixels, in bytes */
    bytesperpixel = ffpxsz(datatype);

    for ii in 0..MAX_COMPRESS_DIM {
        naxes[ii] = 1;
        firstcoord[ii] = 0;
        lastcoord[ii] = 0;
        inc[ii] = 1;
    }

    /*  determine the dimensions of the image to be read */
    ffgidm_safe(fptr, &mut naxis, status);
    ffgisz_safe(fptr, MAX_COMPRESS_DIM as c_int, &mut naxes, status);

    /* calc the cumulative number of pixels in each successive dimension */
    dimsize[0] = 1;
    for ii in 1..MAX_COMPRESS_DIM {
        dimsize[ii] = dimsize[ii - 1] * naxes[ii - 1] as LONGLONG;
    }

    /*  determine the coordinate of the first and last pixel in the image */
    /*  Use zero based indexes here */
    tfirst = fpixel - 1;
    tlast = tfirst + npixel - 1;
    for ii in (0..(naxis as usize)).rev() {
        firstcoord[ii] = tfirst / dimsize[ii];
        lastcoord[ii] = tlast / dimsize[ii];
        tfirst -= firstcoord[ii] * dimsize[ii];
        tlast -= lastcoord[ii] * dimsize[ii];
    }

    /* to simplify things, treat 1-D, 2-D, and 3-D images as separate cases */

    if naxis == 1 {
        /* Simple: just read the requested range of pixels */

        firstcoord[0] += 1;
        lastcoord[0] += 1;
        fits_read_compressed_img(
            fptr,
            datatype,
            &firstcoord,
            &lastcoord,
            &inc,
            nullcheck,
            nullval,
            array,
            nullarray.as_deref_mut(),
            anynul,
            status,
        );
        return *status;
    } else if naxis == 2 {
        nplane = 0; /* read 1st (and only) plane of the image */

        fits_read_compressed_img_plane(
            fptr,
            datatype,
            bytesperpixel as c_int,
            nplane,
            &mut firstcoord,
            &lastcoord,
            &inc,
            &naxes,
            nullcheck,
            nullval,
            array,
            nullarray.as_deref_mut(),
            anynul,
            &mut nread,
            status,
        );
    } else if naxis == 3 {
        /* test for special case: reading an integral number of planes */
        if firstcoord[0] == 0
            && firstcoord[1] == 0
            && lastcoord[0] == naxes[0] as LONGLONG - 1
            && lastcoord[1] == naxes[1] as LONGLONG - 1
        {
            for ii in 0..MAX_COMPRESS_DIM {
                /* convert from zero base to 1 base */
                (firstcoord[ii]) += 1;
                (lastcoord[ii]) += 1;
            }

            /* we can read the contiguous block of pixels in one go */
            fits_read_compressed_img(
                fptr,
                datatype,
                &firstcoord,
                &lastcoord,
                &inc,
                nullcheck,
                nullval,
                array,
                nullarray,
                anynul,
                status,
            );

            return *status;
        }

        if let Some(anynul) = anynul.as_deref_mut() {
            *anynul = 0;
        } /* initialize */

        /* save last coordinate in temporary variables */
        last0 = lastcoord[0];
        last1 = lastcoord[1];

        if firstcoord[2] < lastcoord[2] {
            /* we will read up to the last pixel in all but the last plane */
            lastcoord[0] = naxes[0] as LONGLONG - 1;
            lastcoord[1] = naxes[1] as LONGLONG - 1;
        }

        /* read one plane of the cube at a time, for simplicity */
        for nplane in firstcoord[2]..=lastcoord[2] {
            if nplane == lastcoord[2] {
                lastcoord[0] = last0;
                lastcoord[1] = last1;
            }

            fits_read_compressed_img_plane(
                fptr,
                datatype,
                bytesperpixel as c_int,
                nplane as c_long,
                &mut firstcoord,
                &lastcoord,
                &inc,
                &naxes,
                nullcheck,
                nullval,
                &mut array[arrayptr..],
                if let Some(n) = nullarray.as_deref_mut() {
                    Some(&mut n[nullarrayptr..])
                } else {
                    None
                },
                Some(&mut planenul),
                &mut nread,
                status,
            );

            if planenul != 0
                && let Some(anynul) = anynul.as_deref_mut()
            {
                *anynul = 1; /* there are null pixels */
            }

            /* for all subsequent planes, we start with the first pixel */
            firstcoord[0] = 0;
            firstcoord[1] = 0;

            /* increment pointers to next elements to be read */
            arrayptr += nread as usize * bytesperpixel;
            if nullarrayptr != 0 && (nullcheck == NullCheckType::SetNullArray) {
                nullarrayptr += nread as usize;
            }
        }
    } else {
        ffpmsg_str("only 1D, 2D, or 3D images are currently supported");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    *status
}

/// In general we have to read the first partial row of the image,
/// followed by the middle complete rows, followed by the last
/// partial row of the image.  If the first or last rows are complete,
/// then read them at the same time as all the middle rows.
///
/// # Parameters
///
/// * `fptr`          — (I) FITS file
/// * `datatype`      — (I) datatype of the array to be returned
/// * `bytesperpixel` — (I) number of bytes per pixel in array
/// * `nplane`        — (I) which plane of the cube to read
/// * `firstcoord`    — coordinate of first pixel to read
/// * `lastcoord`     — coordinate of last pixel to read
/// * `inc`           — increment of pixels to read
/// * `naxes`         — size of each image dimension
/// * `nullcheck`     — (I) 0 for no null checking
/// * `nullval`       — (I) value for undefined pixels
/// * `array`         — (O) array of values that are returned
/// * `nullarray`     — (O) array of flags = 1 if nullcheck = 2
/// * `anynul`        — (O) set to 1 if any values are null; else 0
/// * `nread`         — (O) total number of pixels read and returned
/// * `status`        — (IO) error status
fn fits_read_compressed_img_plane(
    fptr: &mut fitsfile,
    datatype: c_int,
    bytesperpixel: c_int,
    nplane: c_long,
    firstcoord: &mut [LONGLONG],
    lastcoord: &[LONGLONG],
    inc: &[c_long],
    naxes: &[c_long],
    nullcheck: NullCheckType,
    /*     1: set undefined pixels = nullval       */
    /*     2: set nullarray=1 for undefined pixels */
    nullval: &Option<NullValue>,
    array: &mut [u8],
    mut nullarray: Option<&mut [c_char]>,
    mut anynul: Option<&mut c_int>,
    nread: &mut c_long,
    status: &mut c_int,
) -> c_int {
    /* bottom left coord. and top right coord. */
    let mut blc: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut trc: [LONGLONG; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut tnull: c_int = 0;

    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = 0;
    }

    *nread = 0;

    let mut arrayptr = 0; // array
    let mut nullarrayptr = 0; // nullarray

    blc[2] = nplane as LONGLONG + 1;
    trc[2] = nplane as LONGLONG + 1;

    if firstcoord[0] != 0 {
        /* have to read a partial first row */
        blc[0] = firstcoord[0] + 1;
        blc[1] = firstcoord[1] + 1;
        trc[1] = blc[1];
        if lastcoord[1] == firstcoord[1] {
            trc[0] = lastcoord[0] + 1; /* 1st and last pixels in same row */
        } else {
            trc[0] = naxes[0] as LONGLONG; /* read entire rest of the row */
        }

        fits_read_compressed_img(
            fptr,
            datatype,
            &blc,
            &trc,
            inc,
            nullcheck,
            nullval,
            &mut array[arrayptr..],
            if let Some(n) = nullarray.as_deref_mut() {
                Some(&mut n[nullarrayptr..])
            } else {
                None
            },
            Some(&mut tnull),
            status,
        );

        *nread += (trc[0] - blc[0] + 1) as c_long;

        if tnull != 0
            && let Some(anynul) = anynul.as_deref_mut()
        {
            *anynul = 1; /* there are null pixels */
        }

        if lastcoord[1] == firstcoord[1] {
            return *status; /* finished */
        }

        /* set starting coord to beginning of next line */
        firstcoord[0] = 0;
        firstcoord[1] += 1;
        arrayptr += (trc[0] - blc[0] + 1) as usize * bytesperpixel as usize;
        if nullarray.is_some() && (nullcheck == NullCheckType::SetNullArray) {
            nullarrayptr += (trc[0] - blc[0] + 1) as usize;
        }
    }

    /* read contiguous complete rows of the image, if any */
    blc[0] = 1;
    blc[1] = firstcoord[1] + 1;
    trc[0] = naxes[0] as LONGLONG;

    if lastcoord[0] + 1 == naxes[0] as LONGLONG {
        /* can read the last complete row, too */
        trc[1] = lastcoord[1] + 1;
    } else {
        /* last row is incomplete; have to read it separately */
        trc[1] = lastcoord[1];
    }

    if trc[1] >= blc[1] {
        /* must have at least one whole line to read */
        fits_read_compressed_img(
            fptr,
            datatype,
            &blc,
            &trc,
            inc,
            nullcheck,
            nullval,
            &mut array[arrayptr..],
            if let Some(n) = nullarray.as_deref_mut() {
                Some(&mut n[nullarrayptr..])
            } else {
                None
            },
            Some(&mut tnull),
            status,
        );

        *nread += ((trc[1] - blc[1] + 1) * naxes[0] as LONGLONG) as c_long;

        if let Some(anynul) = anynul.as_deref_mut()
            && tnull != 0
        {
            *anynul = 1;
        }

        if lastcoord[1] + 1 == trc[1] {
            return *status; /* finished */
        }

        /* increment pointers for the last partial row */
        arrayptr +=
            ((trc[1] - blc[1] + 1) * naxes[0] as LONGLONG) as usize * bytesperpixel as usize;
        if nullarray.is_some() && (nullcheck == NullCheckType::SetNullArray) {
            nullarrayptr += ((trc[1] - blc[1] + 1) * naxes[0] as LONGLONG) as usize;
        }
    }

    if trc[1] == lastcoord[1] + 1 {
        return *status; /* all done */
    }

    /* set starting and ending coord to last line */

    trc[0] = lastcoord[0] + 1;
    trc[1] = lastcoord[1] + 1;
    blc[1] = trc[1];

    fits_read_compressed_img(
        fptr,
        datatype,
        &blc,
        &trc,
        inc,
        nullcheck,
        nullval,
        &mut array[arrayptr..],
        if let Some(n) = nullarray {
            Some(&mut n[nullarrayptr..])
        } else {
            None
        },
        Some(&mut tnull),
        status,
    );

    if let Some(anynul) = anynul
        && tnull != 0
    {
        *anynul = 1;
    }

    *nread += (trc[0] - blc[0] + 1) as c_long;

    *status
}

/// This routine reads keywords from a BINTABLE extension containing a
/// compressed image.
pub(crate) fn imcomp_get_compressed_image_par(infptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut value: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut tstatus: c_int = 0;
    let mut tstatus2: c_int = 0;
    let mut doffset: c_int = 0;
    let mut oldFormat = false;
    let mut colNum: c_int = 0;
    let mut expect_nrows: c_long = 0;
    let mut maxtilelen: c_long = 0;
    let mut rowFactor: c_long = 0;

    if *status > 0 {
        return *status;
    }

    /* Copy relevant header keyword values to structure */
    if fits_read_key_str(infptr, cs!(c"ZCMPTYPE"), &mut value, None, status) > 0 {
        ffpmsg_str("required ZCMPTYPE compression keyword not found in");
        ffpmsg_str(" imcomp_get_compressed_image_par");
        return *status;
    }

    (infptr.Fptr).zcmptype[0] = 0;
    strncat_safe(&mut (infptr.Fptr).zcmptype, &value, 11);

    if FSTRCMP(&value, cs!(c"RICE_1")) == 0 || FSTRCMP(&value, cs!(c"RICE_ONE")) == 0 {
        (infptr.Fptr).compress_type = RICE_1;
    } else if FSTRCMP(&value, cs!(c"HCOMPRESS_1")) == 0 {
        (infptr.Fptr).compress_type = HCOMPRESS_1;
    } else if FSTRCMP(&value, cs!(c"GZIP_1")) == 0 {
        (infptr.Fptr).compress_type = GZIP_1;
    } else if FSTRCMP(&value, cs!(c"GZIP_2")) == 0 {
        (infptr.Fptr).compress_type = GZIP_2;
    } else if FSTRCMP(&value, cs!(c"BZIP2_1")) == 0 {
        (infptr.Fptr).compress_type = BZIP2_1;
    } else if FSTRCMP(&value, cs!(c"PLIO_1")) == 0 {
        (infptr.Fptr).compress_type = PLIO_1;
    } else if FSTRCMP(&value, cs!(c"NOCOMPRESS")) == 0 {
        (infptr.Fptr).compress_type = NOCOMPRESS;
    } else {
        ffpmsg_str("Unknown image compression type:");
        ffpmsg_slice(&value);
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    let mut zbitpix = (infptr.Fptr).zbitpix;
    let r = ffgky_safe(
        infptr,
        KeywordDatatypeMut::TINT(&mut zbitpix),
        cs!(c"ZBITPIX"),
        None,
        status,
    );
    (infptr.Fptr).zbitpix = zbitpix;

    if r > 0 {
        ffpmsg_str("required ZBITPIX compression keyword not found");
        return *status;
    }

    /* If ZZERO and ZSCALE columns don't exist for floating-point types,
    assume there is NO quantization.  Treat exactly as if it had
    ZQUANTIZ='NONE'. This is true regardless of whether or not file has a
    ZQUANTIZ keyword. */
    tstatus = 0;
    tstatus2 = 0;
    if (infptr.Fptr.zbitpix < 0)
        && (fits_get_colnum(
            infptr,
            CASEINSEN as c_int,
            cs!(c"ZZERO"),
            &mut colNum,
            &mut tstatus,
        ) == COL_NOT_FOUND)
        && (fits_get_colnum(
            infptr,
            CASEINSEN as c_int,
            cs!(c"ZSCALE"),
            &mut colNum,
            &mut tstatus2,
        ) == COL_NOT_FOUND)
    {
        (infptr.Fptr).quantize_level = NO_QUANTIZE;
    } else {
        /* get the floating point to integer quantization type, if present. */
        /* FITS files produced before 2009 will not have this keyword */
        /* NO_DITHER should be treated as the default if it is not present. */
        tstatus = 0;
        if fits_read_key_str(infptr, cs!(c"ZQUANTIZ"), &mut value, None, &mut tstatus) > 0 {
            (infptr.Fptr).quantize_method = NO_DITHER;
            (infptr.Fptr).quantize_level = 0.0;

            /* Note that we need to set quantize_level to something other than */
            /* NO_QUANTIZE, since that would cause quantize_method to be ignored, */
            /* and it might already be set from a different HDU. */
        } else if FSTRCMP(&value, cs!(c"NONE")) == 0 {
            (infptr.Fptr).quantize_level = NO_QUANTIZE;
        } else if FSTRCMP(&value, cs!(c"SUBTRACTIVE_DITHER_1")) == 0 {
            (infptr.Fptr).quantize_method = SUBTRACTIVE_DITHER_1;
            (infptr.Fptr).quantize_level = 0.0;
        } else if FSTRCMP(&value, cs!(c"SUBTRACTIVE_DITHER_2")) == 0 {
            (infptr.Fptr).quantize_method = SUBTRACTIVE_DITHER_2;
            (infptr.Fptr).quantize_level = 0.0;
        } else if FSTRCMP(&value, cs!(c"NO_DITHER")) == 0 {
            (infptr.Fptr).quantize_method = NO_DITHER;
            (infptr.Fptr).quantize_level = 0.0;
        } else {
            /* This is an invalid ZQUANTIZ key or an old CFITSIO */
            /* encountering some future standard key. */
            ffpmsg_str("Unknown quantization type:");
            ffpmsg_slice(&value);
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }
    }

    /* get the floating point quantization dithering offset, if present. */
    /* FITS files produced before October 2009 will not have this keyword */
    tstatus = 0;
    if ffgky_safe(
        infptr,
        KeywordDatatypeMut::TINT(&mut doffset),
        cs!(c"ZDITHER0"),
        None,
        &mut tstatus,
    ) > 0
    {
        /* by default start with 1st element of random sequence */
        (infptr.Fptr).dither_seed = 1;
    } else {
        (infptr.Fptr).dither_seed = doffset;
    }

    let mut zndim = (infptr.Fptr).zndim;
    let r = ffgky_safe(
        infptr,
        KeywordDatatypeMut::TINT(&mut zndim),
        cs!(c"ZNAXIS"),
        None,
        status,
    );
    (infptr.Fptr).zndim = zndim;

    if r > 0 {
        ffpmsg_str("required ZNAXIS compression keyword not found");
        return *status;
    }

    if (infptr.Fptr).zndim < 1 {
        ffpmsg_str("Compressed image has no data (ZNAXIS < 1)");
        *status = BAD_NAXIS;
        return *status;
    }

    if (infptr.Fptr).zndim > MAX_COMPRESS_DIM as c_int {
        ffpmsg_str("Compressed image has too many dimensions");
        *status = BAD_NAXIS;
        return *status;
    }

    expect_nrows = 1;
    maxtilelen = 1;
    for ii in 0..((infptr.Fptr).zndim as usize) {
        /* get image size */
        int_snprintf!(keyword, FLEN_KEYWORD, "ZNAXIS{}", ii + 1);

        let mut znaxis_ii = (infptr.Fptr).znaxis[ii];
        ffgky_safe(
            infptr,
            KeywordDatatypeMut::TLONG(&mut znaxis_ii),
            &keyword,
            None,
            status,
        );
        (infptr.Fptr).znaxis[ii] = znaxis_ii;

        if *status > 0 {
            ffpmsg_str("required ZNAXISn compression keyword not found");
            return *status;
        }

        /* get compression tile size */
        int_snprintf!(keyword, FLEN_KEYWORD, "ZTILE{}", ii + 1);

        /* set default tile size in case keywords are not present */
        if ii == 0 {
            (infptr.Fptr).tilesize[0] = (infptr.Fptr).znaxis[0];
        } else {
            (infptr.Fptr).tilesize[ii] = 1;
        }

        tstatus = 0;
        let mut tilesize_ii = (infptr.Fptr).tilesize[ii];
        ffgky_safe(
            infptr,
            KeywordDatatypeMut::TLONG(&mut tilesize_ii),
            &keyword,
            None,
            &mut tstatus,
        );
        (infptr.Fptr).tilesize[ii] = tilesize_ii;

        if (infptr.Fptr).tilesize[ii] == 0 {
            ffpmsg_str("invalid ZTILE value = 0 in compressed image");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        if (infptr.Fptr).znaxis[ii] == LONG_MIN
        /* cannot subtract 1 from this */
        {
            ffpmsg_str("numerical overflow in imcomp_get_compressed_image_par");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }
        rowFactor = ((infptr.Fptr).znaxis[ii] - 1) / (infptr.Fptr).tilesize[ii] + 1;
        if expect_nrows > LONG_MAX / rowFactor {
            ffpmsg_str("numerical overflow in imcomp_get_compressed_image_par");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }
        expect_nrows *= rowFactor;

        if maxtilelen > LONG_MAX / ((infptr.Fptr).tilesize[ii]) {
            ffpmsg_str("numerical overflow in imcomp_get_compressed_image_par");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        maxtilelen *= (infptr.Fptr).tilesize[ii];
    }

    /* check number of rows */
    if expect_nrows as LONGLONG != (infptr.Fptr).numrows {
        ffpmsg_str("number of table rows != the number of tiles in compressed image");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    /* read any algorithm specific parameters */
    if (infptr.Fptr).compress_type == RICE_1 {
        let mut rice_blocksize = (infptr.Fptr).rice_blocksize;
        let r = ffgky_safe(
            infptr,
            KeywordDatatypeMut::TINT(&mut rice_blocksize),
            cs!(c"ZVAL1"),
            None,
            status,
        );
        (infptr.Fptr).rice_blocksize = rice_blocksize;

        if r > 0 {
            ffpmsg_str("required ZVAL1 compression keyword not found");
            return *status;
        }

        tstatus = 0;
        /* First check for very old files, where ZVAL2 wasn't yet designated
        for bytepix */
        if fits_read_key_str(infptr, cs!(c"ZNAME2"), &mut value, None, &mut tstatus) != 0
            && FSTRCMP(&value, cs!(c"NOISEBIT")) == 0
        {
            oldFormat = true;
        }

        tstatus = 0;
        let mut rice_bytepix = (infptr.Fptr).rice_bytepix;
        if oldFormat
            || ffgky_safe(
                infptr,
                KeywordDatatypeMut::TINT(&mut rice_bytepix),
                cs!(c"ZVAL2"),
                None,
                &mut tstatus,
            ) > 0
        {
            (infptr.Fptr).rice_bytepix = 4; /* default value */
        }
        (infptr.Fptr).rice_bytepix = rice_bytepix;

        if (infptr.Fptr).rice_blocksize < 16 && (infptr.Fptr).rice_bytepix > 8 {
            /* values are reversed */
            tstatus = (infptr.Fptr).rice_bytepix;
            (infptr.Fptr).rice_bytepix = (infptr.Fptr).rice_blocksize;
            (infptr.Fptr).rice_blocksize = tstatus;
        }
    } else if (infptr.Fptr).compress_type == HCOMPRESS_1 {
        let mut hcomp_scale = (infptr.Fptr).hcomp_scale;
        let r = fits_read_key_flt(infptr, cs!(c"ZVAL1"), &mut hcomp_scale, None, status);
        (infptr.Fptr).hcomp_scale = hcomp_scale;

        if r > 0 {
            ffpmsg_str("required ZVAL1 compression keyword not found");
            return *status;
        }

        tstatus = 0;
        let mut hcomp_smooth = (infptr.Fptr).hcomp_smooth;
        ffgky_safe(
            infptr,
            KeywordDatatypeMut::TINT(&mut hcomp_smooth),
            cs!(c"ZVAL2"),
            None,
            &mut tstatus,
        );
        (infptr.Fptr).hcomp_smooth = hcomp_smooth;
    }

    /* store number of pixels in each compression tile, */
    /* and max size of the compressed tile buffer */
    (infptr.Fptr).maxtilelen = maxtilelen;

    /* prevent possible divide by zero in imcomp_calc_max_elem */
    if (infptr.Fptr).compress_type == RICE_1 && (infptr.Fptr).rice_blocksize == 0 {
        ffpmsg_str("Invalid RICE_1 blocksize = 0  (fits_get_compressed_img_par)");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    (infptr.Fptr).maxelem = c_long::from(imcomp_calc_max_elem(
        (infptr.Fptr).compress_type,
        maxtilelen as c_int,
        (infptr.Fptr).zbitpix,
        (infptr.Fptr).rice_blocksize,
    ));

    /* Get Column numbers. */
    let mut cn_compressed = (infptr.Fptr).cn_compressed;
    let r = ffgcno_safe(
        infptr,
        CASEINSEN as c_int,
        cs!(c"COMPRESSED_DATA"),
        &mut cn_compressed,
        status,
    );
    (infptr.Fptr).cn_compressed = cn_compressed;

    if r > 0 {
        ffpmsg_str("couldn't find COMPRESSED_DATA column (fits_get_compressed_img_par)");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    ffpmrk_safe(); /* put mark on message stack; erase any messages after this */

    tstatus = 0;
    let mut cn_uncompressed = (infptr.Fptr).cn_uncompressed;
    ffgcno_safe(
        infptr,
        CASEINSEN as c_int,
        cs!(c"UNCOMPRESSED_DATA"),
        &mut cn_uncompressed,
        &mut tstatus,
    );
    (infptr.Fptr).cn_uncompressed = cn_uncompressed;

    tstatus = 0;
    let mut cn_gzip_data = (infptr.Fptr).cn_gzip_data;
    ffgcno_safe(
        infptr,
        CASEINSEN as c_int,
        cs!(c"GZIP_COMPRESSED_DATA"),
        &mut cn_gzip_data,
        &mut tstatus,
    );
    (infptr.Fptr).cn_gzip_data = cn_gzip_data;

    tstatus = 0;
    let mut cn_zscale = (infptr.Fptr).cn_zscale;
    let r = ffgcno_safe(
        infptr,
        CASEINSEN as c_int,
        cs!(c"ZSCALE"),
        &mut cn_zscale,
        &mut tstatus,
    );
    (infptr.Fptr).cn_zscale = cn_zscale;

    if r > 0 {
        /* CMPSCALE column doesn't exist; see if there is a keyword */
        tstatus = 0;
        let mut zscale = (infptr.Fptr).zscale;
        let r = fits_read_key_dbl(infptr, cs!(c"ZSCALE"), &mut zscale, None, &mut tstatus);
        (infptr.Fptr).zscale = zscale;

        if r <= 0 {
            (infptr.Fptr).cn_zscale = -1; /* flag for a constant ZSCALE */
        }
    }

    tstatus = 0;
    let mut cn_zzero = (infptr.Fptr).cn_zzero;
    let r = ffgcno_safe(
        infptr,
        CASEINSEN as c_int,
        cs!(c"ZZERO"),
        &mut cn_zzero,
        &mut tstatus,
    );
    (infptr.Fptr).cn_zzero = cn_zzero;

    if r > 0 {
        /* CMPZERO column doesn't exist; see if there is a keyword */
        tstatus = 0;
        let mut zzero = (infptr.Fptr).zzero;
        let r = fits_read_key_dbl(infptr, cs!(c"ZZERO"), &mut zzero, None, &mut tstatus);
        (infptr.Fptr).zzero = zzero;

        if r <= 0 {
            (infptr.Fptr).cn_zzero = -1; /* flag for a constant ZZERO */
        }
    }

    tstatus = 0;

    let mut cn_zblank = (infptr.Fptr).cn_zblank;
    let r = ffgcno_safe(
        infptr,
        CASEINSEN as c_int,
        cs!(c"ZBLANK"),
        &mut cn_zblank,
        &mut tstatus,
    );

    (infptr.Fptr).cn_zblank = cn_zblank;

    if r > 0 {
        /* ZBLANK column doesn't exist; see if there is a keyword */
        tstatus = 0;
        let mut zblank = (infptr.Fptr).zblank;
        let r = ffgky_safe(
            infptr,
            KeywordDatatypeMut::TINT(&mut zblank),
            cs!(c"ZBLANK"),
            None,
            &mut tstatus,
        );
        (infptr.Fptr).zblank = zblank;

        if r <= 0 {
            (infptr.Fptr).cn_zblank = -1; /* flag for a constant ZBLANK */
        } else {
            /* ZBLANK keyword doesn't exist; see if there is a BLANK keyword */
            tstatus = 0;
            let mut zblank = (infptr.Fptr).zblank;
            let r = ffgky_safe(
                infptr,
                KeywordDatatypeMut::TINT(&mut zblank),
                cs!(c"BLANK"),
                None,
                &mut tstatus,
            );
            (infptr.Fptr).zblank = zblank;

            if r <= 0 {
                (infptr.Fptr).cn_zblank = -1; /* flag for a constant ZBLANK */
            }
        }
    }

    /* read the conventional BSCALE and BZERO scaling keywords, if present */
    tstatus = 0;
    let mut cn_bscale = (infptr.Fptr).cn_bscale;
    let r = fits_read_key_dbl(infptr, cs!(c"BSCALE"), &mut cn_bscale, None, &mut tstatus);
    (infptr.Fptr).cn_bscale = cn_bscale;
    if r > 0 {
        (infptr.Fptr).cn_bscale = 1.0;
    }

    tstatus = 0;
    let mut cn_bzero = (infptr.Fptr).cn_bzero;
    let r = fits_read_key_dbl(infptr, cs!(c"BZERO"), &mut cn_bzero, None, &mut tstatus);
    (infptr.Fptr).cn_bzero = cn_bzero;

    if r > 0 {
        (infptr.Fptr).cn_bzero = 0.0;
        (infptr.Fptr).cn_actual_bzero = 0.0;
    } else {
        (infptr.Fptr).cn_actual_bzero = (infptr.Fptr).cn_bzero;
    }

    /* special case: the quantization level is not given by a keyword in  */
    /* the HDU header, so we have to explicitly copy the requested value */
    /* to the actual value */
    if (infptr.Fptr).request_quantize_level != 0.0 {
        (infptr.Fptr).quantize_level = (infptr.Fptr).request_quantize_level;
    }

    ffcmrk_safe(); /* clear any spurious error messages, back to the mark */
    *status
}

/// This routine reads the header keywords from the input image and
/// copies them to the output image;  the manditory structural keywords
/// and the checksum keywords are not copied. If the DATE keyword is copied,
/// then it is updated with the current date and time.
fn imcomp_copy_imheader(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut nkeys: c_int = 0;
    let mut keyclass: c_int = 0;
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD]; /* a header record */

    if *status > 0 {
        return *status;
    }

    ffghsp_safe(infptr, Some(&mut nkeys), None, status); /* get number of keywords in image */
    for ii in 5..=(nkeys as usize) {
        /* skip the first 4 keywords */

        ffgrec_safe(infptr, ii.try_into().unwrap(), Some(&mut card), status);

        keyclass = ffgkcl_safe(&card); /* Get the type/class of keyword */

        /* don't copy structural keywords or checksum keywords */
        if (keyclass <= TYP_CMPRS_KEY) || (keyclass == TYP_CKSUM_KEY) {
            continue;
        }

        if FSTRNCMP(&card, cs!(c"DATE "), 5) == 0 {
            /* write current date */

            ffpdat_safe(outfptr, status);
        } else if FSTRNCMP(&card, cs!(c"EXTNAME "), 8) == 0 {
            /* don't copy default EXTNAME keyword from a compressed image */
            if FSTRNCMP(&card, cs!(c"EXTNAME = 'COMPRESSED_IMAGE'"), 28) != 0 {
                /* if EXTNAME keyword already exists, overwrite it */
                /* otherwise append a new EXTNAME keyword */
                ffucrd_safe(outfptr, cs!(c"EXTNAME"), &card, status);
            }
        } else {
            /* just copy the keyword to the output header */
            ffprec_safe(outfptr, &card, status);
        }

        if *status > 0 {
            return *status;
        }
    }
    *status
}

/// This routine copies the header keywords from the uncompressed input image
/// and to the compressed image (in a binary table)
fn imcomp_copy_img2comp(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    /* a header record */
    let mut card2: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut nkeys: c_int = 0;
    let mut nmore: c_int = 0;

    let mut tstatus: c_int = 0;
    let mut bitpix: c_int = 0;

    /* tile compressed image keyword translation table  */
    /*                        INPUT      OUTPUT  */
    /*                       01234567   01234567 */
    let patterns: [[&CStr; 2]; 12] = [
        [c"SIMPLE", c"ZSIMPLE"],
        [c"XTENSION", c"ZTENSION"],
        [c"BITPIX", c"ZBITPIX"],
        [c"NAXIS", c"ZNAXIS"],
        [c"NAXISm", c"ZNAXISm"],
        [c"EXTEND", c"ZEXTEND"],
        [c"BLOCKED", c"ZBLOCKED"],
        [c"PCOUNT", c"ZPCOUNT"],
        [c"GCOUNT", c"ZGCOUNT"],
        [c"CHECKSUM", c"ZHECKSUM"], /* save original checksums */
        [c"DATASUM", c"ZDATASUM"],
        [c"*", c"+"],
    ]; /* copy all other keywords */

    let mut npat: c_int = 0;

    if *status > 0 {
        return *status;
    }

    /* write a default EXTNAME keyword if it doesn't exist in input file*/
    fits_read_card(infptr, cs!(c"EXTNAME"), &mut card, status);

    if *status != 0 {
        *status = 0;
        strcpy_safe(&mut card, cs!(c"EXTNAME = 'COMPRESSED_IMAGE'"));
        fits_write_record(outfptr, &card, status);
    }

    /* copy all the keywords from the input file to the output */
    npat = patterns.len() as c_int;
    fits_translate_keywords_safe(infptr, outfptr, 1, &patterns, npat, 0, 0, 0, status);

    if (outfptr.Fptr).request_lossy_int_compress != 0 {
        /* request was made to compress integer images as if they had float pixels. */
        /* If input image has positive bitpix value, then reset the output ZBITPIX */
        /* value to -32. */

        fits_read_key(
            infptr,
            crate::KeywordDatatypeMut::TINT(&mut bitpix),
            cs!(c"BITPIX"),
            None,
            status,
        );

        if *status <= 0 && bitpix > 0 {
            fits_modify_key_lng(outfptr, cs!(c"ZBITPIX"), -32, None, status);

            /* also delete the BSCALE, BZERO, and BLANK keywords */
            tstatus = 0;
            fits_delete_key(outfptr, cs!(c"BSCALE"), &mut tstatus);
            tstatus = 0;
            fits_delete_key(outfptr, cs!(c"BZERO"), &mut tstatus);
            tstatus = 0;
            fits_delete_key(outfptr, cs!(c"BLANK"), &mut tstatus);
        }
    }

    /*
    For compatibility with software that uses an older version of CFITSIO,
    we must make certain that the new ZQUANTIZ keyword, if it exists, must
    occur after the other peudo-required keywords (e.g., ZSIMPLE, ZBITPIX,
    etc.).  Do this by trying to delete the keyword.  If that succeeds (and
    thus the keyword did exist) then rewrite the keyword at the end of header.
    In principle this should not be necessary once all software has upgraded
    to a newer version of CFITSIO (version number greater than 3.181, newer
    than August 2009).

    Do the same for the new ZDITHER0 keyword.
    */

    tstatus = 0;
    if fits_read_card(outfptr, cs!(c"ZQUANTIZ"), &mut card, &mut tstatus) == 0 {
        fits_delete_key(outfptr, cs!(c"ZQUANTIZ"), status);

        /* rewrite the deleted keyword at the end of the header */
        fits_write_record(outfptr, &card, status);

        /* write some associated HISTORY keywords */
        fits_parse_value(&card, &mut card2, None, status);
        if fits_strncasecmp(&card2, cs!(c"'NONE"), 5) != 0 {
            /* the value is not 'NONE' */
            fits_write_history(
                outfptr,
                cs!(c"Image was compressed by CFITSIO using scaled integer quantization:"),
                status,
            );
            /* %f, not `{}': the C always prints six decimals here */
            snprintf_f64(
                &mut card2,
                FLEN_CARD,
                cs!(c"  q = %f / quantized level scaling parameter"),
                f64::from((outfptr.Fptr).request_quantize_level),
            );
            fits_write_history(outfptr, &card2, status);
            fits_write_history(outfptr, &card[10..], status);
        }
    }

    tstatus = 0;
    if fits_read_card(outfptr, cs!(c"ZDITHER0"), &mut card, &mut tstatus) == 0 {
        fits_delete_key(outfptr, cs!(c"ZDITHER0"), status);

        /* rewrite the deleted keyword at the end of the header */
        fits_write_record(outfptr, &card, status);
    }

    ffghsp_safe(infptr, Some(&mut nkeys), Some(&mut nmore), status); /* get number of keywords in image */

    nmore /= 36; /* how many completely empty header blocks are there? */

    /* preserve the same number of spare header blocks in the output header */

    for _jj in 0..(nmore as usize) {
        for _ii in 0..36 {
            fits_write_record(outfptr, cs!(c"    "), status);
        }
    }

    *status
}

/// This routine copies the header keywords from the compressed input image
/// and to the uncompressed image (in a binary table)
fn imcomp_copy_comp2img(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    norec: bool,
    status: &mut c_int,
) -> c_int {
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD]; /* a header record */
    let mut patterns: [[&CStr; 2]; 40] = [[c""; 2]; 40];
    let negative: &CStr = c"-";

    let mut npat: c_int = 0;
    let mut nreq: c_int = 0;
    let mut nsp: c_int = 0;
    let mut tstatus: c_int = 0;
    let mut nkeys: c_int = 0;
    let mut nmore: c_int = 0;

    /* tile compressed image keyword translation table  */
    /*                        INPUT      OUTPUT  */
    /*                       01234567   01234567 */

    /*  only translate these if required keywords not already written */
    let reqkeys: [[&CStr; 2]; 11] = [
        [c"ZSIMPLE", c"SIMPLE"],
        [c"ZTENSION", c"XTENSION"],
        [c"ZBITPIX", c"BITPIX"],
        [c"ZNAXIS", c"NAXIS"],
        [c"ZNAXISm", c"NAXISm"],
        [c"ZEXTEND", c"EXTEND"],
        [c"ZBLOCKED", c"BLOCKED"],
        [c"ZPCOUNT", c"PCOUNT"],
        [c"ZGCOUNT", c"GCOUNT"],
        [c"ZHECKSUM", c"CHECKSUM"], /* restore original checksums */
        [c"ZDATASUM", c"DATASUM"],
    ];

    /* other special keywords */
    let spkeys: [[&CStr; 2]; 22] = [
        [c"XTENSION", c"-"],
        [c"BITPIX", c"-"],
        [c"NAXIS", c"-"],
        [c"NAXISm", c"-"],
        [c"PCOUNT", c"-"],
        [c"GCOUNT", c"-"],
        [c"TFIELDS", c"-"],
        [c"TTYPEm", c"-"],
        [c"TFORMm", c"-"],
        [c"THEAP", c"-"],
        [c"ZIMAGE", c"-"],
        [c"ZQUANTIZ", c"-"],
        [c"ZDITHER0", c"-"],
        [c"ZTILEm", c"-"],
        [c"ZCMPTYPE", c"-"],
        [c"ZBLANK", c"-"],
        [c"ZNAMEm", c"-"],
        [c"ZVALm", c"-"],
        [c"CHECKSUM", c"-"], /* delete checksums */
        [c"DATASUM", c"-"],
        [c"EXTNAME", c"+"], /* we may change this, below */
        [c"*", c"+"],
    ];

    if *status > 0 {
        return *status;
    }

    nreq = reqkeys.len() as c_int;
    nsp = spkeys.len() as c_int;

    /* construct translation patterns */

    for ii in 0..(nreq as usize) {
        patterns[ii][0] = reqkeys[ii][0];

        if norec {
            patterns[ii][1] = negative;
        } else {
            patterns[ii][1] = reqkeys[ii][1];
        }
    }

    for ii in 0..(nsp as usize) {
        patterns[ii + nreq as usize][0] = spkeys[ii][0];
        patterns[ii + nreq as usize][1] = spkeys[ii][1];
    }

    npat = nreq + nsp;

    /* see if the EXTNAME keyword should be copied or not */
    fits_read_card(infptr, cs!(c"EXTNAME"), &mut card, &mut tstatus);

    if tstatus == 0 && strncmp_safe(&card, cs!(c"EXTNAME = 'COMPRESSED_IMAGE'"), 28) == 0 {
        patterns[npat as usize - 2][1] = negative;
    }

    /* translate and copy the keywords from the input file to the output */
    fits_translate_keywords_safe(infptr, outfptr, 1, &patterns, npat, 0, 0, 0, status);

    ffghsp_safe(infptr, Some(&mut nkeys), Some(&mut nmore), status); /* get number of keywords in image */

    nmore /= 36; /* how many completely empty header blocks are there? */

    /* preserve the same number of spare header blocks in the output header */

    for _jj in 0..(nmore as usize) {
        for _ii in 0..36 {
            fits_write_record(outfptr, cs!(c"    "), status);
        }
    }

    *status
}

/// This routine copies any unexpected keywords from the primary array
/// of the compressed input image into the header of the uncompressed image
/// (which is the primary array of the output file).
fn imcomp_copy_prime2img(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut nsp: c_int = 0;

    /* keywords that will not be copied */
    let spkeys: [[&CStr; 2]; 13] = [
        [c"SIMPLE", c"-"],
        [c"BITPIX", c"-"],
        [c"NAXIS", c"-"],
        [c"NAXISm", c"-"],
        [c"PCOUNT", c"-"],
        [c"EXTEND", c"-"],
        [c"GCOUNT", c"-"],
        [c"CHECKSUM", c"-"],
        [c"DATASUM", c"-"],
        [c"EXTNAME", c"-"],
        [c"HISTORY", c"-"],
        [c"COMMENT", c"-"],
        [c"*", c"+"],
    ];

    if *status > 0 {
        return *status;
    }

    nsp = spkeys.len() as c_int;

    /* translate and copy the keywords from the input file to the output */
    fits_translate_keywords_safe(infptr, outfptr, 1, &spkeys, nsp, 0, 0, 0, status);

    *status
}

/// Whether a variable-length array descriptor (`nelem` elements of `elsize` bytes
/// at heap `offset`) lies inside the table's heap.
///
/// Not checked in CFITSIO, which allocates for the length a descriptor states and
/// only then fails to read it. A corrupt descriptor can state any length, so the
/// tile code checks it first rather than allocate gigabytes for a short file.
fn descriptor_in_heap(
    fptr: &fitsfile,
    nelem: LONGLONG,
    offset: LONGLONG,
    elsize: LONGLONG,
) -> bool {
    let f = &fptr.Fptr;
    let data_end = f
        .rowlength
        .checked_mul(f.numrows)
        .and_then(|t| t.checked_add(f.heapsize));
    let end = nelem
        .checked_mul(elsize)
        .and_then(|n| n.checked_add(offset))
        .and_then(|n| n.checked_add(f.heapstart));
    nelem >= 0 && offset >= 0 && matches!((end, data_end), (Some(e), Some(d)) if e <= d)
}

/// This routine decompresses one tile of the image
#[allow(clippy::if_same_then_else)]
// C dispatch chain: distinct conditions deliberately share an action.
/// # Parameters
///
/// * `nrow`       — (I) row of table to read and uncompress
/// * `tilelen`    — (I) number of pixels in the tile
/// * `datatype`   — (I) datatype to be returned in 'buffer'
/// * `nullcheck`  — (I) 0 for no null checking
/// * `nulval`     — (I) value to be used for undefined pixels
/// * `buffer`     — (O) buffer for returned decompressed values
/// * `bnullarray` — (O) buffer for returned null flags
/// * `anynul`     — (O) any null values returned?
fn imcomp_decompress_tile(
    infptr: &mut fitsfile,
    nrow: c_int,
    tilelen: c_int,
    datatype: c_int,
    mut nullcheck: NullCheckType,
    nulval: &Option<NullValue>,
    buffer: &mut [u8],
    bnullarray: &mut [c_char],
    mut anynul: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut tiledatatype: c_int = 0;
    let mut pixlen: usize = 0; /* uncompressed integer data */
    let mut idatalen: size_t = 0;
    let mut tilebytesize: size_t = 0;
    let mut tnull: c_int = 0; /* value in the data which represents nulls */
    let mut cbuf = AlignedBytes::new(); /* compressed data */
    let charnull: c_uchar = 0;
    let snull: c_short = 0;
    let mut blocksize: c_int = 0;
    let mut ntilebins: c_int = 0;
    let mut tilecol: c_long = 0;
    let mut fnulval: f32 = 0.0;
    let mut dnulval: f64 = 0.0;

    let mut tempfloat: Vec<f32> = Vec::new();
    let mut tempdouble: Vec<f64> = Vec::new();

    /* scaling parameters */
    let mut bscale: f64 = 0.;
    let mut bzero: f64 = 0.;
    let mut actual_bzero: f64 = 0.;

    let mut tilesize: c_long = 0; /* number of bytes */

    /* hcompress parameters */
    let mut smooth: c_int = 0;
    // let mut nx: c_int = 0;
    // let mut ny: c_int = 0;
    // let mut scale: c_int = 0;

    let mut nelemll: LONGLONG = 0;
    let mut offset: LONGLONG = 0;

    if *status > 0 {
        return *status;
    }

    /* **************************************************************** */
    /* allocate pointers to array of cached uncompressed tiles, if not already done */
    if (infptr.Fptr).tilerow.is_null() {
        /* calculate number of column bins of compressed tile */
        ntilebins = (((infptr.Fptr).znaxis[0] - 1) / ((infptr.Fptr).tilesize[0])) as c_int + 1;

        if (infptr.Fptr).znaxis[0] != (infptr.Fptr).tilesize[0] || (infptr.Fptr).tilesize[1] != 1 {
            /* don't cache the tile if only single row of the image */

            // HEAP ALLOCATION
            let mut tile_struct = TileStruct {
                tilerow: vec![0; ntilebins as usize],
                tiledatasize: vec![0; ntilebins as usize],
                tiletype: vec![0; ntilebins as usize],
                tiledata: vec![Vec::new(); ntilebins as usize],
                tilenullarray: vec![Vec::new(); ntilebins as usize],
                tileanynull: vec![0; ntilebins as usize],
            };

            let ptr_addr = (&raw const infptr.Fptr) as usize;
            (infptr.Fptr).tilerow = tile_struct.tilerow.as_mut_ptr();
            (infptr.Fptr).tiledata = tile_struct.tiledata.as_mut_ptr().cast::<*mut c_void>();
            (infptr.Fptr).tilenullarray =
                tile_struct.tilenullarray.as_mut_ptr().cast::<*mut c_void>();
            (infptr.Fptr).tiledatasize = tile_struct.tiledatasize.as_mut_ptr();
            (infptr.Fptr).tiletype = tile_struct.tiletype.as_mut_ptr();
            (infptr.Fptr).tileanynull = tile_struct.tileanynull.as_mut_ptr();

            // Insert into the static hashmap. `HashMap::insert` returns `None`
            // for a fresh key (the normal case) and `Some(old)` only when it
            // replaces a stale entry left over from a previously-freed fptr that
            // happened to reuse this address; both are fine, so ignore the result.
            TILE_STRUCTS.lock().unwrap().insert(ptr_addr, tile_struct);
        }
    }

    /* **************************************************************** */
    /* check if this tile was cached; if so, just copy it out */
    if !(infptr.Fptr).tilerow.is_null() {
        /* calculate the column bin of the compressed tile */
        tilecol = (c_long::from(nrow) - 1)
            % (((((infptr.Fptr).znaxis[0] - 1) / ((infptr.Fptr).tilesize[0])) as c_long) + 1);

        let mut tilestruct_lock = TILE_STRUCTS.lock().unwrap();

        let tilestruct = tilestruct_lock
            .get_mut(&(&raw const infptr.Fptr as usize))
            .unwrap();

        if nrow == tilestruct.tilerow[tilecol as usize]
            && datatype == tilestruct.tiletype[tilecol as usize]
        {
            let len = tilestruct.tiledatasize[tilecol as usize] as usize;

            buffer[..len].copy_from_slice(&tilestruct.tiledata[tilecol as usize][..len]);

            if nullcheck == NullCheckType::SetNullArray {
                bnullarray[..tilelen as usize].copy_from_slice(
                    &tilestruct.tilenullarray[tilecol as usize][..tilelen as usize],
                );
            }

            if let Some(anyval) = anynul {
                *anyval = tilestruct.tileanynull[tilecol as usize];
            }

            return *status;
        }
    }

    /* **************************************************************** */
    /* get length of the compressed byte stream */
    ffgdesll_safe(
        infptr,
        (infptr.Fptr).cn_compressed,
        nrow.into(),
        Some(&mut nelemll),
        Some(&mut offset),
        status,
    );

    /* EOF error here indicates that this tile has not yet been written */
    if *status == END_OF_FILE {
        *status = NO_COMPRESSED_TILE;
        return *status;
    }
    let elsize = if (infptr.Fptr).compress_type == PLIO_1 {
        2
    } else {
        1
    };
    if *status <= 0 && !descriptor_in_heap(infptr, nelemll, offset, elsize) {
        ffpmsg_str("compressed tile descriptor points outside the heap (imcomp_decompress_tile)");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    /* **************************************************************** */
    if nelemll == 0 {
        /* special case: tile was not compressed normally */
        if (infptr.Fptr).cn_uncompressed >= 1 {
            /* This option of writing the uncompressed floating point data */
            /* to the tile compressed file was used until about May 2011. */
            /* This was replaced by the more efficient option of gzipping the */
            /* floating point data before writing it to the tile-compressed file
             */

            /* no compressed data, so simply read the uncompressed data */
            /* directly from the UNCOMPRESSED_DATA column */
            ffgdesll_safe(
                infptr,
                (infptr.Fptr).cn_uncompressed,
                nrow.into(),
                Some(&mut nelemll),
                Some(&mut offset),
                status,
            );

            if nelemll == 0 && offset == 0 {
                /* this should never happen */
                *status = NO_COMPRESSED_TILE;
                return *status;
            }

            /* the values are read straight into the tile buffer, so there can be */
            /* no more of them than the tile has pixels */
            if *status <= 0
                && (nelemll > LONGLONG::from(tilelen)
                    || !descriptor_in_heap(infptr, nelemll, offset, 1))
            {
                ffpmsg_str("uncompressed tile descriptor is inconsistent (imcomp_decompress_tile)");
                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }

            if nullcheck != NullCheckType::None && nullcheck != NullCheckType::SetPixel {
                /* set any null values in the array = nulval */
                fits_read_col(
                    infptr,
                    datatype,
                    (infptr.Fptr).cn_uncompressed,
                    LONGLONG::from(nrow),
                    1,
                    nelemll as LONGLONG,
                    nulval.clone(),
                    buffer,
                    anynul,
                    status,
                );
            } else {
                /* set the bnullarray = 1 for any null values in the array */
                fits_read_colnull(
                    infptr,
                    datatype,
                    (infptr.Fptr).cn_uncompressed,
                    LONGLONG::from(nrow),
                    1,
                    nelemll as LONGLONG,
                    buffer,
                    bnullarray,
                    anynul,
                    status,
                );
            }
        } else if (infptr.Fptr).cn_gzip_data >= 1 {
            /* This is the newer option, that was introduced in May 2011 */
            /* floating point data was not quantized,  so read the losslessly */
            /* compressed data from the GZIP_COMPRESSED_DATA column */

            ffgdesll_safe(
                infptr,
                (infptr.Fptr).cn_gzip_data,
                LONGLONG::from(nrow),
                Some(&mut nelemll),
                Some(&mut offset),
                status,
            );

            if nelemll == 0 && offset == 0 {
                /* this should never happen */
                *status = NO_COMPRESSED_TILE;
                return *status;
            }
            if *status <= 0 && !descriptor_in_heap(infptr, nelemll, offset, 1) {
                ffpmsg_str(
                    "gzipped tile descriptor points outside the heap (imcomp_decompress_tile)",
                );
                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }

            /* allocate memory for the compressed tile of data */
            let mut cbuf = Vec::new();
            if cbuf.try_reserve_exact((nelemll) as usize).is_err() {
                ffpmsg_str("error allocating memory for gzipped tile (imcomp_decompress_tile)");
                *status = MEMORY_ALLOCATION;
                return *status;
            } else {
                cbuf.resize(nelemll as usize, 0);
            }

            /* read array of compressed bytes */
            if fits_read_col(
                infptr,
                TBYTE,
                (infptr.Fptr).cn_gzip_data,
                LONGLONG::from(nrow),
                1,
                nelemll as LONGLONG,
                Some(NullValue::UByte(charnull)),
                cast_slice_mut(&mut cbuf),
                None,
                status,
            ) > 0
            {
                ffpmsg_str("error reading compressed byte stream from binary table");
                return *status;
            }

            /* size of the returned (uncompressed) data buffer, in bytes */
            if (infptr.Fptr).zbitpix == FLOAT_IMG {
                idatalen = (tilelen as usize) * mem::size_of::<f32>();
            } else if (infptr.Fptr).zbitpix == DOUBLE_IMG {
                idatalen = (tilelen as usize) * mem::size_of::<f64>();
            } else {
                /* this should never happen! */
                ffpmsg_cstr(
                    c"incompatible data type in gzipped floating-point tile-compressed image",
                );

                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }

            /* Do not allow image float/doubles into int arrays */
            if datatype != TFLOAT && datatype != TDOUBLE {
                ffpmsg_str(
                    "attempting to read compressed float or double image into incompatible data type",
                );

                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }

            if datatype == TFLOAT && (infptr.Fptr).zbitpix == DOUBLE_IMG {
                if tempdouble.try_reserve_exact(idatalen as usize).is_err() {
                    ffpmsg_cstr(
                        c"Memory allocation failure for tempdouble. (imcomp_decompress_tile)",
                    );
                    *status = MEMORY_ALLOCATION;
                    return *status;
                } else {
                    tempdouble.resize(idatalen as usize, 0.0);
                }

                /* uncompress the data into temp buffer */
                // WARNING: Potentially unsafe memory issue here given this function
                // call can reallocate the buffer.
                if unsafe {
                    uncompress2mem_from_mem(
                        &cbuf,
                        (nelemll as c_long).try_into().unwrap(),
                        &mut tempdouble.as_mut_ptr().cast::<u8>(),
                        &mut idatalen,
                        None,
                        Some(&mut tilebytesize),
                        status,
                    )
                } != 0
                {
                    ffpmsg_str("failed to gunzip the image tile");
                    return *status;
                }
            } else if datatype == TDOUBLE && (infptr.Fptr).zbitpix == FLOAT_IMG {
                /*  have to allocat a temporary buffer for the uncompressed data  in the */
                /*  case where a gzipped "float" tile is returned as a "double"  array   */
                if tempfloat.try_reserve_exact(idatalen as usize).is_err() {
                    ffpmsg_cstr(
                        c"Memory allocation failure for tempfloat. (imcomp_decompress_tile)",
                    );
                    *status = MEMORY_ALLOCATION;
                    return *status;
                } else {
                    tempfloat.resize(idatalen as usize, 0.0);
                }

                /* uncompress the data into temp buffer */
                // WARNING: Potentially unsafe memory issue here given this function
                // call can reallocate the buffer.
                if unsafe {
                    uncompress2mem_from_mem(
                        &cbuf,
                        (nelemll as c_long).try_into().unwrap(),
                        &mut tempfloat.as_mut_ptr().cast::<u8>(),
                        &mut idatalen,
                        None,
                        Some(&mut tilebytesize),
                        status,
                    )
                } != 0
                {
                    ffpmsg_str("failed to gunzip the image tile");
                    return *status;
                }
            } else {
                /* uncompress the data directly into the output buffer in all  other cases */
                /* zlib writes up to idatalen bytes through the raw pointer, so the */
                /* buffer must really be that long */
                if buffer.len() < idatalen {
                    ffpmsg_str(
                        "output buffer too small for the gzipped tile (imcomp_decompress_tile)",
                    );
                    *status = DATA_DECOMPRESSION_ERR;
                    return *status;
                }
                if unsafe {
                    uncompress2mem_from_mem(
                        &cbuf,
                        (nelemll as c_long).try_into().unwrap(),
                        &mut (buffer.as_mut_ptr()),
                        &mut idatalen,
                        None,
                        Some(&mut tilebytesize),
                        status,
                    )
                } != 0
                {
                    ffpmsg_str("failed to gunzip the image tile");
                    return *status;
                }
            }

            /* do byte swapping and null value substitution for the tile of pixels */
            if tilebytesize == 4 * tilelen as usize {
                /* float pixels */

                if BYTESWAPPED {
                    if !tempfloat.is_empty() {
                        ffswap4(cast_slice_mut(&mut tempfloat), c_long::from(tilelen));
                    } else {
                        ffswap4(cast_slice_mut(buffer), c_long::from(tilelen));
                    }
                }
                if datatype == TFLOAT {
                    if let Some(nulval) = nulval {
                        fnulval = nulval.get_value_as_f64() as f32;
                    }

                    fffr4r4_inplace(
                        cast_slice_mut(buffer),
                        c_long::from(tilelen),
                        1.0,
                        0.0,
                        nullcheck,
                        fnulval,
                        bnullarray,
                        anynul,
                        status,
                    );
                } else if datatype == TDOUBLE {
                    if let Some(nulval) = nulval {
                        dnulval = nulval.get_value_as_f64();
                    }

                    /* note that the R*4 data are in the tempfloat array in this case */
                    fffr4r8(
                        &tempfloat,
                        c_long::from(tilelen),
                        1.0,
                        0.0,
                        nullcheck,
                        dnulval,
                        bnullarray,
                        anynul,
                        cast_slice_mut(buffer),
                        status,
                    );
                } else {
                    ffpmsg_cstr(
                        c"implicit data type conversion is not supported for gzipped image tiles",
                    );
                    *status = DATA_DECOMPRESSION_ERR;
                    return *status;
                }
            } else if tilebytesize == 8 * tilelen as usize {
                /* double pixels */

                if BYTESWAPPED {
                    if !tempdouble.is_empty() {
                        ffswap8(cast_slice_mut(&mut tempdouble), c_long::from(tilelen));
                    } else {
                        ffswap8(cast_slice_mut(buffer), c_long::from(tilelen));
                    }
                }
                if datatype == TFLOAT {
                    if let Some(nulval) = nulval {
                        fnulval = nulval.get_value_as_f64() as f32;
                    }

                    fffr8r4(
                        &tempdouble,
                        c_long::from(tilelen),
                        1.0,
                        0.0,
                        nullcheck,
                        fnulval,
                        bnullarray,
                        anynul,
                        cast_slice_mut(buffer),
                        status,
                    );
                    tempdouble.clear();
                } else if datatype == TDOUBLE {
                    if let Some(nulval) = nulval {
                        dnulval = nulval.get_value_as_f64();
                    }

                    fffr8r8_inplace(
                        cast_slice_mut(buffer),
                        c_long::from(tilelen),
                        1.0,
                        0.0,
                        nullcheck,
                        dnulval,
                        bnullarray,
                        anynul,
                        status,
                    );
                } else {
                    ffpmsg_cstr(
                        c"implicit data type conversion is not supported in tile-compressed images",
                    );
                    *status = DATA_DECOMPRESSION_ERR;
                    return *status;
                }
            } else {
                ffpmsg_str("error: uncompressed tile has wrong size");
                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }

        /* end of special case of losslessly gzipping a floating-point image tile */
        } else {
            /* this should never happen */
            *status = NO_COMPRESSED_TILE;
        }

        return *status;
    }

    /* **************************************************************** */
    /* deal with the normal case of a compressed tile of pixels */
    if nullcheck == NullCheckType::SetNullArray {
        for ii in 0..(tilelen as usize) {
            /* initialize the null flage array */
            bnullarray[ii] = 0;
        }
    }

    if let Some(anynul) = anynul.as_deref_mut() {
        *anynul = 0;
    }

    /* get linear scaling and offset values, if they exist */
    actual_bzero = (infptr.Fptr).cn_actual_bzero;
    if (infptr.Fptr).cn_zscale == 0 {
        /* set default scaling, if scaling is not defined */
        bscale = 1.;
        bzero = 0.;
    } else if (infptr.Fptr).cn_zscale == -1 {
        bscale = (infptr.Fptr).zscale;
        bzero = (infptr.Fptr).zzero;
    } else {
        /* read the linear scale and offset values for this row */
        let mut dummy = [0.0; 1];

        dummy[0] = bscale;
        ffgcvd_safe(
            infptr,
            (infptr.Fptr).cn_zscale,
            nrow.into(),
            1,
            1,
            0.,
            &mut dummy,
            None,
            status,
        );
        bscale = dummy[0];

        dummy[0] = bzero;
        ffgcvd_safe(
            infptr,
            (infptr.Fptr).cn_zzero,
            nrow.into(),
            1,
            1,
            0.,
            &mut dummy,
            None,
            status,
        );
        bzero = dummy[0];

        if *status > 0 {
            ffpmsg_str("error reading scaling factor and offset for compressed tile");
            return *status;
        }

        /* test if floating-point FITS image also has non-default BSCALE and  */
        /* BZERO keywords.  If so, we have to combine the 2 linear scaling factors. */

        if ((infptr.Fptr).zbitpix == FLOAT_IMG || (infptr.Fptr).zbitpix == DOUBLE_IMG)
            && ((infptr.Fptr).cn_bscale != 1.0 || (infptr.Fptr).cn_bzero != 0.0)
        {
            bscale *= (infptr.Fptr).cn_bscale;
            bzero = bzero * (infptr.Fptr).cn_bscale + (infptr.Fptr).cn_bzero;
        }
    }

    if bscale == 1.0 && bzero == 0.0 {
        /* if no other scaling has been specified, try using the values
        given by the BSCALE and BZERO keywords, if any */

        bscale = (infptr.Fptr).cn_bscale;
        bzero = (infptr.Fptr).cn_bzero;
    }

    /* ************************************************************* */
    /* get the value used to represent nulls in the int array */
    if (infptr.Fptr).cn_zblank == 0 {
        nullcheck = NullCheckType::None; /* no null value; don't check for nulls */
    } else if (infptr.Fptr).cn_zblank == -1 {
        tnull = (infptr.Fptr).zblank; /* use the the ZBLANK keyword */
    } else {
        /* read the null value for this row */
        let mut dummy = [0; 1];
        dummy[0] = tnull;
        ffgcvk_safe(
            infptr,
            (infptr.Fptr).cn_zblank,
            LONGLONG::from(nrow),
            1,
            1,
            0,
            &mut dummy,
            None,
            status,
        );
        tnull = dummy[0];

        if *status > 0 {
            ffpmsg_str("error reading null value for compressed tile");
            return *status;
        }
    }

    /* ************************************************************* */
    /* allocate memory for the uncompressed array of tile integers */
    /* The size depends on the datatype and the compression type. */

    if (infptr.Fptr).compress_type == HCOMPRESS_1
        && ((infptr.Fptr).zbitpix != BYTE_IMG && (infptr.Fptr).zbitpix != SHORT_IMG)
    {
        idatalen = (tilelen as usize) * mem::size_of::<LONGLONG>(); /* 8 bytes per pixel */
    } else if (infptr.Fptr).compress_type == RICE_1
        && (infptr.Fptr).zbitpix == BYTE_IMG
        && (infptr.Fptr).rice_bytepix == 1
    {
        idatalen = (tilelen as usize) * mem::size_of::<c_char>(); /* 1 byte per pixel */
    } else if ((infptr.Fptr).compress_type == GZIP_1
        || (infptr.Fptr).compress_type == GZIP_2
        || (infptr.Fptr).compress_type == BZIP2_1)
        && (infptr.Fptr).zbitpix == BYTE_IMG
    {
        idatalen = (tilelen as usize) * mem::size_of::<c_char>(); /* 1 byte per pixel */
    } else if (infptr.Fptr).compress_type == RICE_1
        && (infptr.Fptr).zbitpix == SHORT_IMG
        && (infptr.Fptr).rice_bytepix == 2
    {
        idatalen = (tilelen as usize) * mem::size_of::<c_short>(); /* 2 bytes per pixel */
    } else if ((infptr.Fptr).compress_type == GZIP_1
        || (infptr.Fptr).compress_type == GZIP_2
        || (infptr.Fptr).compress_type == BZIP2_1)
        && (infptr.Fptr).zbitpix == SHORT_IMG
    {
        idatalen = (tilelen as usize) * mem::size_of::<c_short>(); /* 2 bytes per pixel */
    } else if ((infptr.Fptr).compress_type == GZIP_1
        || (infptr.Fptr).compress_type == GZIP_2
        || (infptr.Fptr).compress_type == BZIP2_1)
        && (infptr.Fptr).zbitpix == DOUBLE_IMG
    {
        idatalen = (tilelen as usize) * mem::size_of::<f64>(); /* 8 bytes per pixel  */
    } else {
        idatalen = (tilelen as usize) * mem::size_of::<c_int>(); /* all other cases have int pixels */
    }

    let mut tmp_idata = AlignedBytes::new();
    if tmp_idata.try_resize_zeroed(idatalen as usize).is_err() {
        ffpmsg_str("Memory allocation failure for idata. (imcomp_decompress_tile)");
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    let idata = &mut tmp_idata;

    /* ************************************************************* */
    /* allocate memory for the compressed bytes */

    let mut cbuf_len = 0;
    if (infptr.Fptr).compress_type == PLIO_1 {
        cbuf_len = (nelemll as usize) * mem::size_of::<c_short>();
    } else {
        cbuf_len = nelemll as usize;
    }

    if cbuf.try_resize_zeroed(cbuf_len).is_err() {
        ffpmsg_str("Out of memory for cbuf. (imcomp_decompress_tile)");
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    /* ************************************************************* */
    /* read the compressed bytes from the FITS file */

    if (infptr.Fptr).compress_type == PLIO_1 {
        fits_read_col(
            infptr,
            TSHORT,
            (infptr.Fptr).cn_compressed,
            nrow.into(),
            1,
            nelemll as LONGLONG,
            Some(NullValue::Short(snull)),
            &mut cbuf,
            None,
            status,
        );
    } else {
        fits_read_col(
            infptr,
            TBYTE,
            (infptr.Fptr).cn_compressed,
            nrow.into(),
            1,
            nelemll as LONGLONG,
            Some(NullValue::UByte(charnull)),
            &mut cbuf,
            None,
            status,
        );
    }

    if *status > 0 {
        ffpmsg_str("error reading compressed byte stream from binary table");

        return *status;
    }

    /* ************************************************************* */
    /*  call the algorithm-specific code to uncompress the tile */

    if (infptr.Fptr).compress_type == RICE_1 {
        blocksize = (infptr.Fptr).rice_blocksize;

        let rcd = RCDecoder::new();

        /* The decoders want an output of exactly tilelen pixels. idata is sized */
        /* by ZBITPIX, which a malformed header can set inconsistently with      */
        /* BYTEPIX, so take a tilelen slice of it, if it is long enough.          */
        let nx = tilelen as usize;
        let r = if (infptr.Fptr).rice_bytepix == 1 {
            tiledatatype = TBYTE;
            let out: &mut [u8] = cast_slice_mut(idata);
            out.get_mut(..nx)
                .map(|out| rcd.decode_byte(&cbuf, nx, blocksize as usize, out).is_ok())
        } else if (infptr.Fptr).rice_bytepix == 2 {
            tiledatatype = TSHORT;
            let out: &mut [c_ushort] = cast_slice_mut(idata);
            out.get_mut(..nx)
                .map(|out| rcd.decode_short(&cbuf, nx, blocksize as usize, out).is_ok())
        } else {
            tiledatatype = TINT;
            let out: &mut [c_uint] = cast_slice_mut(idata);
            out.get_mut(..nx)
                .map(|out| rcd.decode(&cbuf, nx, blocksize as usize, out).is_ok())
        };
        if r != Some(true) {
            *status = DATA_DECOMPRESSION_ERR;
        }

    /* ************************************************************* */
    } else if (infptr.Fptr).compress_type == HCOMPRESS_1 {
        smooth = (infptr.Fptr).hcomp_smooth;

        let mut hcd = HCDecoder::new();
        let res = if (infptr.Fptr).zbitpix == BYTE_IMG || (infptr.Fptr).zbitpix == SHORT_IMG {
            hcd.read(&cbuf, smooth, cast_slice_mut(idata))
        } else {
            /* zbitpix = LONG_IMG (32) */
            /* idata must have been allocated twice as large for this to work */
            hcd.read64(&cbuf, smooth, cast_slice_mut(idata))
        };

        if let Ok(_res) = res {
            // nx = res.0 as c_int;
            // ny = res.1 as c_int;
            // scale = res.2;
        } else {
            *status = DATA_DECOMPRESSION_ERR;
        }

        tiledatatype = TINT;

    /* ************************************************************* */
    } else if (infptr.Fptr).compress_type == PLIO_1 {
        /* uncompress the data */
        if pl_l2pi(
            cast_slice(&cbuf),
            0,
            cast_slice_mut(idata),
            tilelen as usize,
        )
        .is_none()
        {
            ffpmsg_str("error: out-of-bounds memory access attempt (imcomp_decompress_tile)");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }
        tiledatatype = TINT;

    /* ************************************************************* */
    } else if ((infptr.Fptr).compress_type == GZIP_1) || ((infptr.Fptr).compress_type == GZIP_2) {
        unsafe {
            uncompress2mem_from_mem(
                cast_slice(&cbuf),
                (nelemll as c_long).try_into().unwrap(),
                &mut (idata.as_mut_ptr()),
                &mut idatalen,
                /* idata is Rust-owned and cannot be grown by realloc; it is already */
                /* as large as any valid tile, so a longer stream is an error */
                None,
                Some(&mut tilebytesize),
                status,
            );
        }

        /* determine the data type of the uncompressed array, and */
        /*  do byte unshuffling and unswapping if needed */
        if tilebytesize == (tilelen * 2) as usize {
            /* this is a short I*2 array */
            tiledatatype = TSHORT;

            if (infptr.Fptr).compress_type == GZIP_2 {
                fits_unshuffle_2bytes(cast_slice_mut(idata), LONGLONG::from(tilelen), status);
            }

            if BYTESWAPPED {
                ffswap2(cast_slice_mut(idata), c_long::from(tilelen));
            }
        } else if tilebytesize == (tilelen * 4) as usize {
            /* this is a int I*4 array (or maybe R*4) */
            tiledatatype = TINT;

            if (infptr.Fptr).compress_type == GZIP_2 {
                fits_unshuffle_4bytes(cast_slice_mut(idata), LONGLONG::from(tilelen), status);
            }

            if BYTESWAPPED {
                ffswap4(cast_slice_mut(idata), c_long::from(tilelen));
            }
        } else if tilebytesize == ((tilelen * 8) as usize) {
            /* this is a R*8 double array */
            tiledatatype = TDOUBLE;

            if (infptr.Fptr).compress_type == GZIP_2 {
                fits_unshuffle_8bytes(cast_slice_mut(idata), LONGLONG::from(tilelen), status);
            }
            if BYTESWAPPED {
                ffswap8(cast_slice_mut(idata), c_long::from(tilelen));
            }
        } else if tilebytesize == (tilelen as size_t) {
            /* this is an unsigned char I*1 array */
            tiledatatype = TBYTE;
        } else {
            ffpmsg_str("error: uncompressed tile has wrong size");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

    /* ************************************************************* */
    } else if (infptr.Fptr).compress_type == BZIP2_1 {
        /*  BZIP2 is not supported in the public release; this is only for test
        purposes

        if (BZ2_bzBuffToBuffDecompress ((char *) idata, &idatalen,
              (char *)cbuf, (unsigned int) nelemll, 0, 0) )
        */
        {
            ffpmsg_str("bzip2 decompression error");

            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        if (infptr.Fptr).zbitpix == BYTE_IMG {
            tiledatatype = TBYTE;
        } else if (infptr.Fptr).zbitpix == SHORT_IMG {
            tiledatatype = TSHORT;
            if BYTESWAPPED {
                ffswap2(cast_slice_mut(idata), c_long::from(tilelen));
            }
        } else {
            tiledatatype = TINT;
            if BYTESWAPPED {
                ffswap4(cast_slice_mut(idata), c_long::from(tilelen));
            }
        }

    /* ************************************************************* */
    } else {
        ffpmsg_str("unknown compression algorithm");
        *status = DATA_DECOMPRESSION_ERR;
        return *status;
    }

    if *status != 0 {
        /* error uncompressing the tile */
        return *status;
    }

    /* ************************************************************* */
    /* copy the uncompressed tile data to the output buffer, doing */
    /* null checking, datatype conversion and linear scaling, if necessary */

    let nulval = (nulval.clone()).unwrap_or(NullValue::Double(0.0)); /* set address to dummy value */

    if datatype == TSHORT {
        pixlen = mem::size_of::<c_short>();

        if tiledatatype == TINT {
            if (infptr.Fptr).compress_type == PLIO_1 && actual_bzero == 32768. {
                /* special case where unsigned 16-bit integers have been */
                /* offset by +32768 when using PLIO */
                fffi4i2(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero - 32768.,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_short,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else {
                fffi4i2(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_short,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );

                /*
                 Hcompress is a special case:  ignore any numerical overflow
                 errors that may have occurred during the integer*4 to
                 integer*2 convertion.  Overflows can happen when a lossy
                 Hcompress algorithm is invoked (with a non-zero scale
                 factor).  The fffi4i2 routine clips the returned values to be
                 within the legal I*2 range, so all we need to is to reset the
                 error status to zero.
                */

                if (infptr.Fptr).compress_type == HCOMPRESS_1
                    && ((*status == NUM_OVERFLOW) || (*status == OVERFLOW_ERR))
                {
                    *status = 0;
                }
            }
        } else if tiledatatype == TSHORT {
            fffi2i2(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_short,
                nulval.get_value_as_f64() as c_short,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TBYTE {
            fffi1i2(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_uchar,
                nulval.get_value_as_f64() as c_short,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else {
            fffi8i2(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as LONGLONG,
                nulval.get_value_as_f64() as c_short,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        }
    } else if datatype == TINT {
        pixlen = mem::size_of::<c_int>();

        if tiledatatype == TINT {
            if (infptr.Fptr).compress_type == PLIO_1 && actual_bzero == 32768. {
                /* special case where unsigned 16-bit integers have been */
                /* offset by +32768 when using PLIO */
                fffi4int(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero - 32768.,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_int,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else {
                fffi4int(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_int,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else if tiledatatype == TSHORT {
            fffi2int(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_short,
                nulval.get_value_as_f64() as c_int,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TBYTE {
            fffi1int(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_uchar,
                nulval.get_value_as_f64() as c_int,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else {
            fffi8int(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as LONGLONG,
                nulval.get_value_as_f64() as c_int,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        }
    } else if datatype == TLONG {
        pixlen = mem::size_of::<c_long>();

        if tiledatatype == TINT {
            if (infptr.Fptr).compress_type == PLIO_1 && actual_bzero == 32768. {
                /* special case where unsigned 16-bit integers have been */
                /* offset by +32768 when using PLIO */
                fffi4i4(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero - 32768.,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_long,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else {
                fffi4i4(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_long,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else if tiledatatype == TSHORT {
            fffi2i4(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_short,
                nulval.get_value_as_f64() as c_long,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TBYTE {
            fffi1i4(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_uchar,
                nulval.get_value_as_f64() as c_long,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else {
            fffi8i4(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as LONGLONG,
                nulval.get_value_as_f64() as c_long,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        }
    } else if datatype == TFLOAT {
        pixlen = mem::size_of::<f32>();
        fnulval = nulval.get_value_as_f64() as f32;

        if (infptr.Fptr).quantize_level == NO_QUANTIZE {
            /* the floating point pixels were losselessly compressed with GZIP */
            /* Just have to copy the values to the output array */

            if tiledatatype == TINT {
                fffr4r4(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    fnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else {
                fffr8r4(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    fnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else if (infptr.Fptr).quantize_method == SUBTRACTIVE_DITHER_1
            || (infptr.Fptr).quantize_method == SUBTRACTIVE_DITHER_2
        {
            /* use the new dithering algorithm (introduced in July 2009) */

            if tiledatatype == TINT {
                unquantize_i4r4(
                    c_long::from(nrow + (infptr.Fptr).dither_seed - 1) as c_long,
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    (infptr.Fptr).quantize_method,
                    nullcheck,
                    tnull,
                    fnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else if tiledatatype == TSHORT {
                unquantize_i2r4(
                    c_long::from(nrow + (infptr.Fptr).dither_seed - 1),
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    (infptr.Fptr).quantize_method,
                    nullcheck,
                    tnull as c_short,
                    fnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else if tiledatatype == TBYTE {
                unquantize_i1r4(
                    c_long::from(nrow + (infptr.Fptr).dither_seed - 1),
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    (infptr.Fptr).quantize_method,
                    nullcheck,
                    tnull as c_uchar,
                    fnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else {
            /* use the old "round to nearest level" quantization algorithm
             */

            if tiledatatype == TINT {
                if (infptr.Fptr).compress_type == PLIO_1 && actual_bzero == 32768.0 {
                    /* special case where unsigned 16-bit integers have been */
                    /* offset by +32768 when using PLIO */
                    fffi4r4(
                        cast_slice_mut(idata),
                        c_long::from(tilelen),
                        bscale,
                        bzero - 32768.,
                        nullcheck,
                        tnull,
                        fnulval,
                        bnullarray,
                        anynul.as_deref_mut(),
                        cast_slice_mut(buffer),
                        status,
                    );
                } else {
                    fffi4r4(
                        cast_slice_mut(idata),
                        c_long::from(tilelen),
                        bscale,
                        bzero,
                        nullcheck,
                        tnull,
                        fnulval,
                        bnullarray,
                        anynul.as_deref_mut(),
                        cast_slice_mut(buffer),
                        status,
                    );
                }
            } else if tiledatatype == TSHORT {
                fffi2r4(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull as c_short,
                    fnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else if tiledatatype == TBYTE {
                fffi1r4(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull as c_uchar,
                    fnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        }
    } else if datatype == TDOUBLE {
        pixlen = mem::size_of::<f64>();
        dnulval = nulval.get_value_as_f64();

        if (infptr.Fptr).quantize_level == NO_QUANTIZE {
            /* the floating point pixels were losselessly compressed with GZIP */
            /* Just have to copy the values to the output array */

            if tiledatatype == TINT {
                fffr4r8(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    dnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else {
                fffr8r8(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    dnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else if (infptr.Fptr).quantize_method == SUBTRACTIVE_DITHER_1
            || (infptr.Fptr).quantize_method == SUBTRACTIVE_DITHER_2
        {
            /* use the new dithering algorithm (introduced in July 2009) */
            if tiledatatype == TINT {
                unquantize_i4r8(
                    c_long::from(nrow + (infptr.Fptr).dither_seed - 1),
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    (infptr.Fptr).quantize_method,
                    nullcheck,
                    tnull,
                    dnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else if tiledatatype == TSHORT {
                unquantize_i2r8(
                    c_long::from(nrow + (infptr.Fptr).dither_seed - 1),
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    (infptr.Fptr).quantize_method,
                    nullcheck,
                    tnull as c_short,
                    dnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else if tiledatatype == TBYTE {
                unquantize_i1r8(
                    c_long::from(nrow + (infptr.Fptr).dither_seed - 1),
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    (infptr.Fptr).quantize_method,
                    nullcheck,
                    tnull as c_uchar,
                    dnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else {
            /* use the old "round to nearest level" quantization algorithm */

            if tiledatatype == TINT {
                if (infptr.Fptr).compress_type == PLIO_1 && actual_bzero == 32768. {
                    /* special case where unsigned 16-bit integers have been */
                    /* offset by +32768 when using PLIO */
                    fffi4r8(
                        cast_slice_mut(idata),
                        c_long::from(tilelen),
                        bscale,
                        bzero - 32768.,
                        nullcheck,
                        tnull,
                        dnulval,
                        bnullarray,
                        anynul.as_deref_mut(),
                        cast_slice_mut(buffer),
                        status,
                    );
                } else {
                    fffi4r8(
                        cast_slice_mut(idata),
                        c_long::from(tilelen),
                        bscale,
                        bzero,
                        nullcheck,
                        tnull,
                        dnulval,
                        bnullarray,
                        anynul.as_deref_mut(),
                        cast_slice_mut(buffer),
                        status,
                    );
                }
            } else if tiledatatype == TSHORT {
                fffi2r8(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull as c_short,
                    dnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else if tiledatatype == TBYTE {
                fffi1r8(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull as c_uchar,
                    dnulval,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        }
    } else if datatype == TBYTE {
        pixlen = mem::size_of::<c_char>();
        if tiledatatype == TINT {
            fffi4i1(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull,
                nulval.get_value_as_f64() as c_uchar,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TSHORT {
            fffi2i1(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_short,
                nulval.get_value_as_f64() as c_uchar,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TBYTE {
            fffi1i1(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_uchar,
                nulval.get_value_as_f64() as c_uchar,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        }
    } else if datatype == TSBYTE {
        pixlen = mem::size_of::<c_char>();
        if tiledatatype == TINT {
            fffi4s1(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull,
                nulval.get_value_as_f64() as i8,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TSHORT {
            fffi2s1(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_short,
                nulval.get_value_as_f64() as i8,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TBYTE {
            fffi1s1(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_uchar,
                nulval.get_value_as_f64() as i8,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        }
    } else if datatype == TUSHORT {
        pixlen = mem::size_of::<c_short>();

        if tiledatatype == TINT {
            if (infptr.Fptr).compress_type == PLIO_1 && actual_bzero == 32768. {
                /* special case where unsigned 16-bit integers have been */
                /* offset by +32768 when using PLIO */
                fffi4u2(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero - 32768.,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_ushort,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else {
                fffi4u2(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_ushort,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else if tiledatatype == TSHORT {
            fffi2u2(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_short,
                nulval.get_value_as_f64() as c_ushort,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TBYTE {
            fffi1u2(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_uchar,
                nulval.get_value_as_f64() as c_ushort,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else {
            fffi8u2(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as LONGLONG,
                nulval.get_value_as_f64() as c_ushort,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        }
    } else if datatype == TUINT {
        pixlen = mem::size_of::<c_int>();

        if tiledatatype == TINT {
            if (infptr.Fptr).compress_type == PLIO_1 && actual_bzero == 32768. {
                /* special case where unsigned 16-bit integers have been */
                /* offset by +32768 when using PLIO */
                fffi4uint(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero - 32768.,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_uint,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else {
                fffi4uint(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_uint,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else if tiledatatype == TSHORT {
            fffi2uint(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_short,
                nulval.get_value_as_f64() as c_uint,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TBYTE {
            fffi1uint(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_uchar,
                nulval.get_value_as_f64() as c_uint,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else {
            fffi8uint(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as LONGLONG,
                nulval.get_value_as_f64() as c_uint,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        }
    } else if datatype == TULONG {
        pixlen = mem::size_of::<c_long>();

        if tiledatatype == TINT {
            if (infptr.Fptr).compress_type == PLIO_1 && actual_bzero == 32768. {
                /* special case where unsigned 16-bit integers have been */
                /* offset by +32768 when using PLIO */
                fffi4u4(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero - 32768.,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_ulong,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            } else {
                fffi4u4(
                    cast_slice_mut(idata),
                    c_long::from(tilelen),
                    bscale,
                    bzero,
                    nullcheck,
                    tnull,
                    nulval.get_value_as_f64() as c_ulong,
                    bnullarray,
                    anynul.as_deref_mut(),
                    cast_slice_mut(buffer),
                    status,
                );
            }
        } else if tiledatatype == TSHORT {
            fffi2u4(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_short,
                nulval.get_value_as_f64() as c_ulong,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else if tiledatatype == TBYTE {
            fffi1u4(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as c_uchar,
                nulval.get_value_as_f64() as c_ulong,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        } else {
            fffi8u4(
                cast_slice_mut(idata),
                c_long::from(tilelen),
                bscale,
                bzero,
                nullcheck,
                tnull as LONGLONG,
                nulval.get_value_as_f64() as c_ulong,
                bnullarray,
                anynul.as_deref_mut(),
                cast_slice_mut(buffer),
                status,
            );
        }
    } else {
        *status = BAD_DATATYPE;
    }

    /* **************************************************************** */
    /* cache the tile, in case the application wants it again  */

    /*   Don't cache the tile if tile is a single row of the image;
    it is less likely that the cache will be used in this cases,
    so it is not worth the time and the memory overheads.
    */

    if !(infptr.Fptr).tilerow.is_null() {
        /* make sure cache has been allocated */
        if (infptr.Fptr).znaxis[0] != (infptr.Fptr).tilesize[0] || (infptr.Fptr).tilesize[1] != 1 {
            let mut tilestruct_lock = TILE_STRUCTS.lock().unwrap();

            let tilestruct = tilestruct_lock
                .get_mut(&(&raw const infptr.Fptr as usize))
                .unwrap();

            tilesize = pixlen as c_long * c_long::from(tilelen);

            /* check that tile size/type has not changed */
            if tilesize != tilestruct.tiledatasize[tilecol as usize]
                || datatype != tilestruct.tiletype[tilecol as usize]
            {
                if !(tilestruct.tiledata)[tilecol as usize].is_empty() {
                    tilestruct.tiledata[tilecol as usize].clear();
                }

                if !(tilestruct.tilenullarray)[tilecol as usize].is_empty() {
                    tilestruct.tilenullarray[tilecol as usize].clear();
                }

                // (tilestruct.tilenullarray)[tilecol as usize] = 0;
                (tilestruct.tilerow)[tilecol as usize] = 0;
                (tilestruct.tiledatasize)[tilecol as usize] = 0;
                (tilestruct.tiletype)[tilecol as usize] = 0;

                /* allocate new array(s). Note tiledata holds `tilesize` bytes
                (pixlen * tilelen), and must be resized to that length (not just
                have capacity reserved) because the tile is copied in by slice
                below. */
                let mut v = Vec::new();
                if v.try_reserve_exact(tilesize as usize).is_err() {
                    return *status;
                } else {
                    v.resize(tilesize as usize, 0);
                    tilestruct.tiledata[tilecol as usize] = v;
                }

                if nullcheck == NullCheckType::SetNullArray {
                    /* also need array of null pixel flags (one per pixel) */

                    let mut v = Vec::new();
                    if v.try_reserve_exact(tilelen as usize).is_err() {
                        return *status;
                    } else {
                        v.resize(tilelen as usize, 0);
                        tilestruct.tilenullarray[tilecol as usize] = v;
                    }
                }

                tilestruct.tiledatasize[tilecol as usize] = tilesize;
                tilestruct.tiletype[tilecol as usize] = datatype;
            }

            /* copy the tile array(s) into cache buffer */
            tilestruct.tiledata[tilecol as usize][..tilesize as usize]
                .copy_from_slice(&buffer[..tilesize as usize]);

            if nullcheck == NullCheckType::SetNullArray {
                /*
                if (infptr.Fptr).tilenullarray == std::ptr::null_mut() {
                    (infptr.Fptr).tilenullarray[tilecol as usize] = malloc(tilelen);
                }
                */
                tilestruct.tilenullarray[tilecol as usize][..tilelen as usize]
                    .copy_from_slice(&bnullarray[..tilelen as usize]);
            }

            tilestruct.tilerow[tilecol as usize] = nrow;
            tilestruct.tileanynull[tilecol as usize] = *anynul.unwrap();
        }
    }
    *status
}

/// Test if there are any intersecting pixels between this tile and the section
/// of the image defined by fixel, lpixel, ininc.
///
/// # Parameters
///
/// * `ndim`    — (I) number of dimension in the tile and image
/// * `tfpixel` — (I) first pixel number in each dim. of the tile
/// * `tlpixel` — (I) last pixel number in each dim. of the tile
/// * `fpixel`  — (I) first pixel number in each dim. of the image
/// * `lpixel`  — (I) last pixel number in each dim. of the image
/// * `ininc`   — (I) increment to be applied in each image dimen.
fn imcomp_test_overlap(
    ndim: c_int,
    tfpixel: &[c_long],
    tlpixel: &[c_long],
    fpixel: &[c_long],
    lpixel: &[c_long],
    ininc: &[c_long],
    status: &mut c_int,
) -> c_int {
    let mut imgdim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* product of preceding dimensions in the  output image, allowing for inc factor */
    let mut tiledim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* product of preceding dimensions in the tile, array;  inc factor is not relevant */
    let mut imgfpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* 1st img pix overlapping tile: 0 base, allowing for inc factor */
    let mut imglpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* last img pix overlapping tile 0 base, allowing for inc factor */
    let mut tilefpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* 1st tile pix overlapping img 0 base, allowing for inc factor */
    let mut inc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* local copy of input ininc */

    let mut tf: c_long = 0;
    let mut tl: c_long = 0;

    if *status > 0 {
        return *status;
    }

    /* ------------------------------------------------------------ */
    /* calc amount of overlap in each dimension; if there is zero   */
    /* overlap in any dimension then just return  */
    /* ------------------------------------------------------------ */

    for ii in 0..(ndim as usize) {
        if tlpixel[ii] < fpixel[ii] || tfpixel[ii] > lpixel[ii] {
            return 0; /* there are no overlapping pixels */
        }

        inc[ii] = ininc[ii];

        /* calc dimensions of the output image section */
        imgdim[ii] = (lpixel[ii] - fpixel[ii]) / (inc[ii]).abs() + 1;
        if imgdim[ii] < 1 {
            *status = NEG_AXIS;
            return 0;
        }

        /* calc dimensions of the tile */
        tiledim[ii] = tlpixel[ii] - tfpixel[ii] + 1;
        if tiledim[ii] < 1 {
            *status = NEG_AXIS;
            return 0;
        }

        if ii > 0 {
            tiledim[ii] *= tiledim[ii - 1]; /* product of dimensions */
        }

        /* first and last pixels in image that overlap with the tile, 0 base */
        tf = tfpixel[ii] - 1;
        tl = tlpixel[ii] - 1;

        /* skip this plane if it falls in the cracks of the subsampled image */
        while ((tf - (fpixel[ii] - 1)) % (inc[ii]).abs()) != 0 {
            tf += 1;
            if tf > tl {
                return 0; /* no overlapping pixels */
            }
        }

        while ((tl - (fpixel[ii] - 1)) % (inc[ii]).abs()) != 0 {
            tl -= 1;
            if tf > tl {
                return 0; /* no overlapping pixels */
            }
        }
        imgfpix[ii] = cmp::max((tf - fpixel[ii] + 1) / (inc[ii]).abs(), 0);
        imglpix[ii] = cmp::min((tl - fpixel[ii] + 1) / (inc[ii]).abs(), imgdim[ii] - 1);

        /* first pixel in the tile that overlaps with the image (0 base) */
        tilefpix[ii] = cmp::max(fpixel[ii] - tfpixel[ii], 0);

        while ((tfpixel[ii] + tilefpix[ii] - fpixel[ii]) % (inc[ii]).abs()) != 0 {
            (tilefpix[ii]) += 1;
            if tilefpix[ii] >= tiledim[ii] {
                return 0; /* no overlapping pixels */
            }
        }

        if ii > 0 {
            imgdim[ii] *= imgdim[ii - 1]; /* product of dimensions */
        }
    }

    1 /* there appears to be  intersecting pixels */
}

/// Copy the intersecting pixels from a decompressed tile to the output image.
/// Both the tile and the image must have the same number of dimensions.
///
/// # Parameters
///
/// * `tile`       — (I) multi dimensional array of tile pixels
/// * `pixlen`     — (I) number of bytes in each tile or image pixel
/// * `ndim`       — (I) number of dimension in the tile and image
/// * `tfpixel`    — (I) first pixel number in each dim. of the tile
/// * `tlpixel`    — (I) last pixel number in each dim. of the tile
/// * `bnullarray` — (I) array of null flags; used if nullcheck = 2
/// * `image`      — (O) multi dimensional output image
/// * `fpixel`     — (I) first pixel number in each dim. of the image
/// * `lpixel`     — (I) last pixel number in each dim. of the image
/// * `ininc`      — (I) increment to be applied in each image dimen.
/// * `nullcheck`  — (I) 0, 1: do nothing; 2: set nullarray for nulls
fn imcomp_copy_overlap(
    tile: &[c_char],
    pixlen: c_int,
    ndim: c_int,
    tfpixel: &[c_long],
    tlpixel: &[c_long],
    bnullarray: &[c_char],
    image: &mut [c_char],
    fpixel: &[c_long],
    lpixel: &[c_long],
    ininc: &[c_long],
    nullcheck: NullCheckType,
    mut nullarray: Option<&mut [c_char]>,
    status: &mut c_int,
) -> c_int {
    let mut imgdim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* product of preceding dimensions in the output image, allowing for inc factor */
    let mut tiledim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* product of preceding dimensions in the tile, array;  inc factor is not relevant */
    let mut imgfpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* 1st img pix overlapping tile: 0 base, allowing for inc factor */
    let mut imglpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* last img pix overlapping tile 0 base, allowing for inc factor */
    let mut tilefpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* 1st tile pix overlapping img 0 base, allowing for inc factor */
    let mut inc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; /* local copy of input ininc */

    // offset to image pixel, allowing for inc
    let mut im1: c_long = 0;
    let mut im2: c_long = 0;
    let mut im3: c_long = 0;
    let mut im4: c_long = 0;
    let mut tf: c_long = 0;
    let mut tl: c_long = 0;

    // offset along each axis of the tile
    let mut t2: c_long = 0;
    let mut t3: c_long = 0;
    let mut t4: c_long = 0;
    let mut tilepix: c_long = 0;
    let mut imgpix: c_long = 0;
    let mut tilepixbyte: c_long = 0;
    let mut imgpixbyte: c_long = 0;
    let mut overlap_bytes: c_int = 0;
    let mut overlap_flags: c_int = 0;

    if *status > 0 {
        return *status;
    }

    for ii in 0..MAX_COMPRESS_DIM {
        /* set default values for higher dimensions */
        inc[ii] = 1;
        imgdim[ii] = 1;
        tiledim[ii] = 1;
        imgfpix[ii] = 0;
        imglpix[ii] = 0;
        tilefpix[ii] = 0;
    }

    /* ------------------------------------------------------------ */
    /* calc amount of overlap in each dimension; if there is zero   */
    /* overlap in any dimension then just return  */
    /* ------------------------------------------------------------ */

    for ii in 0..(ndim as usize) {
        if tlpixel[ii] < fpixel[ii] || tfpixel[ii] > lpixel[ii] {
            return *status; /* there are no overlapping pixels */
        }

        inc[ii] = ininc[ii];

        /* calc dimensions of the output image section */
        imgdim[ii] = (lpixel[ii] - fpixel[ii]) / (inc[ii]).abs() + 1;
        if imgdim[ii] < 1 {
            *status = NEG_AXIS;
            return *status;
        }

        /* calc dimensions of the tile */
        tiledim[ii] = tlpixel[ii] - tfpixel[ii] + 1;
        if tiledim[ii] < 1 {
            *status = NEG_AXIS;
            return *status;
        }

        if ii > 0 {
            tiledim[ii] *= tiledim[ii - 1]; /* product of dimensions */
        }

        /* first and last pixels in image that overlap with the tile, 0 base */
        tf = tfpixel[ii] - 1;
        tl = tlpixel[ii] - 1;

        /* skip this plane if it falls in the cracks of the subsampled image */
        while ((tf - (fpixel[ii] - 1)) % (inc[ii]).abs()) != 0 {
            tf += 1;
            if tf > tl {
                return *status; /* no overlapping pixels */
            }
        }

        while ((tl - (fpixel[ii] - 1)) % (inc[ii]).abs()) != 0 {
            tl -= 1;
            if tf > tl {
                return *status; /* no overlapping pixels */
            }
        }
        imgfpix[ii] = cmp::max((tf - fpixel[ii] + 1) / (inc[ii]).abs(), 0);
        imglpix[ii] = cmp::min((tl - fpixel[ii] + 1) / (inc[ii]).abs(), imgdim[ii] - 1);

        /* first pixel in the tile that overlaps with the image (0 base) */
        tilefpix[ii] = cmp::max(fpixel[ii] - tfpixel[ii], 0);

        while ((tfpixel[ii] + tilefpix[ii] - fpixel[ii]) % (inc[ii]).abs()) != 0 {
            (tilefpix[ii]) += 1;
            if tilefpix[ii] >= tiledim[ii] {
                return *status; /* no overlapping pixels */
            }
        }
        /*
        printf("ii tfpixel, tlpixel %d %d %d \n",ii, tfpixel[ii], tlpixel[ii]);
        printf("ii, tf, tl, imgfpix,imglpix, tilefpix %d %d %d %d %d %d\n",ii,
        tf,tl,imgfpix[ii], imglpix[ii],tilefpix[ii]);
        */
        if ii > 0 {
            imgdim[ii] *= imgdim[ii - 1]; /* product of dimensions */
        }
    }

    /* ---------------------------------------------------------------- */
    /* calc number of pixels in each row (first dimension) that overlap */
    /* multiply by pixlen to get number of bytes to copy in each loop   */
    /* ---------------------------------------------------------------- */

    if inc[0] != 1 {
        overlap_flags = 1; /* can only copy 1 pixel at a time */
    } else {
        overlap_flags = (imglpix[0] - imgfpix[0] + 1) as c_int; /* can copy whole row */
    }

    overlap_bytes = overlap_flags * pixlen;

    /* support up to 5 dimensions for now */
    let mut it4 = 0;
    for i4 in 0..=(imglpix[4] - imgfpix[4]) {
        /* increment plane if it falls in the cracks of the subsampled image */
        while ndim > 4 && (tfpixel[4] + tilefpix[4] - fpixel[4] + it4) % (inc[4]).abs() != 0 {
            it4 += 1;
        }

        /* offset to start of hypercube */
        if inc[4] > 0 {
            im4 = (i4 + imgfpix[4]) * imgdim[3];
        } else {
            im4 = imgdim[4] - (i4 + 1 + imgfpix[4]) * imgdim[3];
        }

        t4 = (tilefpix[4] + it4) * tiledim[3];
        let mut it3 = 0;
        for i3 in 0..=(imglpix[3] - imgfpix[3]) {
            /* increment plane if it falls in the cracks of the subsampled image
             */
            while ndim > 3 && (tfpixel[3] + tilefpix[3] - fpixel[3] + it3) % (inc[3]).abs() != 0 {
                it3 += 1;
            }

            /* offset to start of cube */
            if inc[3] > 0 {
                im3 = (i3 + imgfpix[3]) * imgdim[2] + im4;
            } else {
                im3 = imgdim[3] - (i3 + 1 + imgfpix[3]) * imgdim[2] + im4;
            }

            t3 = (tilefpix[3] + it3) * tiledim[2] + t4;

            /* loop through planes of the image */
            let mut it2 = 0;
            for i2 in 0..=(imglpix[2] - imgfpix[2]) {
                /* incre plane if it falls in the cracks of the subsampled image */
                while ndim > 2 && (tfpixel[2] + tilefpix[2] - fpixel[2] + it2) % (inc[2]).abs() != 0
                {
                    it2 += 1;
                }

                /* offset to start of plane */
                if inc[2] > 0 {
                    im2 = (i2 + imgfpix[2]) * imgdim[1] + im3;
                } else {
                    im2 = imgdim[2] - (i2 + 1 + imgfpix[2]) * imgdim[1] + im3;
                }

                t2 = (tilefpix[2] + it2) * tiledim[1] + t3;

                /* loop through rows of the image */
                let mut it1 = 0;
                for i1 in 0..=(imglpix[1] - imgfpix[1]) {
                    /* incre row if it falls in the cracks of the subsampled image */
                    while ndim > 1
                        && (tfpixel[1] + tilefpix[1] - fpixel[1] + it1) % (inc[1]).abs() != 0
                    {
                        it1 += 1;
                    }

                    /* calc position of first pixel in tile to be copied */
                    tilepix = tilefpix[0] + (tilefpix[1] + it1) * tiledim[0] + t2;

                    /* offset to start of row */
                    if inc[1] > 0 {
                        im1 = (i1 + imgfpix[1]) * imgdim[0] + im2;
                    } else {
                        im1 = imgdim[1] - (i1 + 1 + imgfpix[1]) * imgdim[0] + im2;
                    }
                    /*
                    printf("inc = %d %d %d %d\n",inc[0],inc[1],inc[2],inc[3]);
                    printf("im1,im2,im3,im4 = %d %d %d %d\n",im1,im2,im3,im4);
                    */
                    /* offset to byte within the row */
                    if inc[0] > 0 {
                        imgpix = imgfpix[0] + im1;
                    } else {
                        imgpix = imgdim[0] - 1 - imgfpix[0] + im1;
                    }
                    /*
                    printf("tilefpix0,1, imgfpix1, it1, inc1, t2= %d %d %d %d %d
                    %d\n", tilefpix[0],tilefpix[1],imgfpix[1],it1,inc[1], t2);
                    printf("i1, it1, tilepix, imgpix %d %d %d %d \n", i1, it1,
                    tilepix, imgpix);
                    */
                    /* loop over pixels along one row of the image */
                    for _ipos in
                        (imgfpix[0]..=imglpix[0]).step_by(overlap_flags.try_into().unwrap())
                    {
                        if nullcheck == NullCheckType::SetNullArray {
                            /* copy overlapping null flags from tile to image */
                            let n = nullarray.as_deref_mut().unwrap();
                            n[imgpix as usize..(imgpix + c_long::from(overlap_flags)) as usize]
                                .copy_from_slice(
                                    &bnullarray[tilepix as usize
                                        ..(tilepix + c_long::from(overlap_flags)) as usize],
                                );
                        }

                        /* convert from image pixel to byte offset */
                        tilepixbyte = tilepix * c_long::from(pixlen);
                        imgpixbyte = imgpix * c_long::from(pixlen);
                        /*
                        printf("  tilepix, tilepixbyte, imgpix, imgpixbyte= %d
                        %d %d %d\n", tilepix, tilepixbyte, imgpix, imgpixbyte);
                        */
                        /* copy overlapping row of pixels from tile to image */
                        image[imgpixbyte as usize
                            ..(imgpixbyte + c_long::from(overlap_bytes)) as usize]
                            .copy_from_slice(
                                &tile[tilepixbyte as usize
                                    ..(tilepixbyte + c_long::from(overlap_bytes)) as usize],
                            );

                        tilepix += c_long::from(overlap_flags) * (inc[0]).abs();
                        if inc[0] > 0 {
                            imgpix += c_long::from(overlap_flags);
                        } else {
                            imgpix -= c_long::from(overlap_flags);
                        }
                    }
                    it1 += 1;
                }
                it2 += 1;
            }
            it3 += 1;
        }
        it4 += 1;
    }
    *status
}

/// Similar to imcomp_copy_overlap, except it copies the overlapping pixels from
/// the 'image' to the 'tile'.
///
/// # Parameters
///
/// * `tile`        — (O) multi dimensional array of tile pixels
/// * `pixlen`      — (I) number of bytes in each tile or image pixel
/// * `ndim`        — (I) number of dimension in the tile and image
/// * `tfpixel`     — (I) first pixel number in each dim. of the tile
/// * `tlpixel`     — (I) last pixel number in each dim. of the tile
/// * `_bnullarray` — (I) array of null flags; used if nullcheck = 2
/// * `image`       — (I) multi dimensional output image
/// * `fpixel`      — (I) first pixel number in each dim. of the image
/// * `lpixel`      — (I) last pixel number in each dim. of the image
/// * `_nullcheck`  — (I) 0, 1: do nothing; 2: set nullarray for nulls
fn imcomp_merge_overlap(
    tile: &mut [c_char],
    pixlen: c_int,
    ndim: c_int,
    tfpixel: &[c_long],
    tlpixel: &[c_long],
    _bnullarray: &[c_char],
    image: &[c_char],
    fpixel: &[c_long],
    lpixel: &[c_long],
    _nullcheck: NullCheckType,
    status: &mut c_int,
) -> c_int {
    let mut imgdim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; // product of preceding dimensions in the output image, allowing for inc factor
    let mut tiledim: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; // product of preceding dimensions in the tile, array;  inc factor is not relevant
    let mut imgfpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; // 1st img pix overlapping tile: 0 base, allowing for inc factor
    let mut imglpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; // last img pix overlapping tile 0 base, allowing for inc factor
    let mut tilefpix: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; // 1st tile pix overlapping img 0 base, allowing for inc factor
    let mut inc: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM]; // local copy of input ininc

    // offset to image pixel, allowing for inc
    let mut im1: c_long = 0;
    let mut im2: c_long = 0;
    let mut im3: c_long = 0;
    let mut im4: c_long = 0;
    let mut tf: c_long = 0;
    let mut tl: c_long = 0;

    // offset along each axis of the tile
    let mut t2: c_long = 0;
    let mut t3: c_long = 0;
    let mut t4: c_long = 0;
    let mut tilepix: c_long = 0;
    let mut imgpix: c_long = 0;
    let mut tilepixbyte: c_long = 0;
    let mut imgpixbyte: c_long = 0;
    let mut overlap_bytes: c_int = 0;
    let mut overlap_flags: c_int = 0;

    if *status > 0 {
        return *status;
    }

    for ii in 0..MAX_COMPRESS_DIM {
        /* set default values for higher dimensions */
        inc[ii] = 1;
        imgdim[ii] = 1;
        tiledim[ii] = 1;
        imgfpix[ii] = 0;
        imglpix[ii] = 0;
        tilefpix[ii] = 0;
    }

    /* ------------------------------------------------------------ */
    /* calc amount of overlap in each dimension; if there is zero   */
    /* overlap in any dimension then just return  */
    /* ------------------------------------------------------------ */

    for ii in 0..(ndim as usize) {
        if tlpixel[ii] < fpixel[ii] || tfpixel[ii] > lpixel[ii] {
            return *status; /* there are no overlapping pixels */
        }

        /* calc dimensions of the output image section */
        imgdim[ii] = (lpixel[ii] - fpixel[ii]) / (inc[ii]).abs() + 1;
        if imgdim[ii] < 1 {
            *status = NEG_AXIS;
            return *status;
        }

        /* calc dimensions of the tile */
        tiledim[ii] = tlpixel[ii] - tfpixel[ii] + 1;
        if tiledim[ii] < 1 {
            *status = NEG_AXIS;
            return *status;
        }

        if ii > 0 {
            tiledim[ii] *= tiledim[ii - 1]; /* product of dimensions */
        }

        /* first and last pixels in image that overlap with the tile, 0 base */
        tf = tfpixel[ii] - 1;
        tl = tlpixel[ii] - 1;

        /* skip this plane if it falls in the cracks of the subsampled image */
        while ((tf - (fpixel[ii] - 1)) % (inc[ii]).abs()) != 0 {
            tf += 1;
            if tf > tl {
                return *status; /* no overlapping pixels */
            }
        }

        while ((tl - (fpixel[ii] - 1)) % (inc[ii]).abs()) != 0 {
            tl -= 1;
            if tf > tl {
                return *status; /* no overlapping pixels */
            }
        }
        imgfpix[ii] = cmp::max((tf - fpixel[ii] + 1) / (inc[ii]).abs(), 0);
        imglpix[ii] = cmp::min((tl - fpixel[ii] + 1) / (inc[ii]).abs(), imgdim[ii] - 1);

        /* first pixel in the tile that overlaps with the image (0 base) */
        tilefpix[ii] = cmp::max(fpixel[ii] - tfpixel[ii], 0);

        while ((tfpixel[ii] + tilefpix[ii] - fpixel[ii]) % (inc[ii]).abs()) != 0 {
            (tilefpix[ii]) += 1;
            if tilefpix[ii] >= tiledim[ii] {
                return *status; /* no overlapping pixels */
            }
        }
        /*
        printf("ii tfpixel, tlpixel %d %d %d \n",ii, tfpixel[ii], tlpixel[ii]);
        printf("ii, tf, tl, imgfpix,imglpix, tilefpix %d %d %d %d %d %d\n",ii,
        tf,tl,imgfpix[ii], imglpix[ii],tilefpix[ii]);
        */
        if ii > 0 {
            imgdim[ii] *= imgdim[ii - 1]; /* product of dimensions */
        }
    }

    /* ---------------------------------------------------------------- */
    /* calc number of pixels in each row (first dimension) that overlap */
    /* multiply by pixlen to get number of bytes to copy in each loop   */
    /* ---------------------------------------------------------------- */

    if inc[0] != 1 {
        overlap_flags = 1; /* can only copy 1 pixel at a time */
    } else {
        overlap_flags = (imglpix[0] - imgfpix[0] + 1) as c_int; /* can copy whole row */
    }

    overlap_bytes = overlap_flags * pixlen;

    /* support up to 5 dimensions for now */
    let mut it4 = 0;
    for i4 in 0..=(imglpix[4] - imgfpix[4]) {
        /* increment plane if it falls in the cracks of the subsampled image */
        while ndim > 4 && (tfpixel[4] + tilefpix[4] - fpixel[4] + it4) % (inc[4]).abs() != 0 {
            it4 += 1;
        }

        /* offset to start of hypercube */
        if inc[4] > 0 {
            im4 = (i4 + imgfpix[4]) * imgdim[3];
        } else {
            im4 = imgdim[4] - (i4 + 1 + imgfpix[4]) * imgdim[3];
        }

        t4 = (tilefpix[4] + it4) * tiledim[3];
        let mut it3 = 0;
        for i3 in 0..=(imglpix[3] - imgfpix[3]) {
            /* increment plane if it falls in the cracks of the subsampled image
             */
            while ndim > 3 && (tfpixel[3] + tilefpix[3] - fpixel[3] + it3) % (inc[3]).abs() != 0 {
                it3 += 1;
            }

            /* offset to start of cube */
            if inc[3] > 0 {
                im3 = (i3 + imgfpix[3]) * imgdim[2] + im4;
            } else {
                im3 = imgdim[3] - (i3 + 1 + imgfpix[3]) * imgdim[2] + im4;
            }

            t3 = (tilefpix[3] + it3) * tiledim[2] + t4;

            /* loop through planes of the image */
            let mut it2 = 0;
            for i2 in 0..=(imglpix[2] - imgfpix[2]) {
                /* incre plane if it falls in the cracks of the subsampled image*/
                while ndim > 2 && (tfpixel[2] + tilefpix[2] - fpixel[2] + it2) % (inc[2]).abs() != 0
                {
                    it2 += 1;
                }

                /* offset to start of plane */
                if inc[2] > 0 {
                    im2 = (i2 + imgfpix[2]) * imgdim[1] + im3;
                } else {
                    im2 = imgdim[2] - (i2 + 1 + imgfpix[2]) * imgdim[1] + im3;
                }

                t2 = (tilefpix[2] + it2) * tiledim[1] + t3;

                /* loop through rows of the image */
                let mut it1 = 0;
                for i1 in 0..=(imglpix[1] - imgfpix[1]) {
                    /* incre row if it falls in the cracks of the subsampled image */
                    while ndim > 1
                        && (tfpixel[1] + tilefpix[1] - fpixel[1] + it1) % (inc[1]).abs() != 0
                    {
                        it1 += 1;
                    }

                    /* calc position of first pixel in tile to be copied */
                    tilepix = tilefpix[0] + (tilefpix[1] + it1) * tiledim[0] + t2;

                    /* offset to start of row */
                    if inc[1] > 0 {
                        im1 = (i1 + imgfpix[1]) * imgdim[0] + im2;
                    } else {
                        im1 = imgdim[1] - (i1 + 1 + imgfpix[1]) * imgdim[0] + im2;
                    }
                    /*
                    printf("inc = %d %d %d %d\n",inc[0],inc[1],inc[2],inc[3]);
                    printf("im1,im2,im3,im4 = %d %d %d %d\n",im1,im2,im3,im4);
                    */
                    /* offset to byte within the row */
                    if inc[0] > 0 {
                        imgpix = imgfpix[0] + im1;
                    } else {
                        imgpix = imgdim[0] - 1 - imgfpix[0] + im1;
                    }
                    /*
                    printf("tilefpix0,1, imgfpix1, it1, inc1, t2= %d %d %d %d %d
                    %d\n", tilefpix[0],tilefpix[1],imgfpix[1],it1,inc[1], t2);
                    printf("i1, it1, tilepix, imgpix %d %d %d %d \n", i1, it1,
                    tilepix, imgpix);
                    */
                    /* loop over pixels along one row of the image */
                    for _ipos in (imgfpix[0]..=imglpix[0]).step_by(overlap_flags as usize) {
                        /* convert from image pixel to byte offset */
                        tilepixbyte = tilepix * c_long::from(pixlen);
                        imgpixbyte = imgpix * c_long::from(pixlen);
                        /*
                        printf("  tilepix, tilepixbyte, imgpix, imgpixbyte= %d
                        %d %d %d\n", tilepix, tilepixbyte, imgpix, imgpixbyte);
                        */
                        /* copy overlapping row of pixels from image to tile */
                        tile[tilepixbyte as usize
                            ..(tilepixbyte + c_long::from(overlap_bytes)) as usize]
                            .copy_from_slice(
                                &image[imgpixbyte as usize
                                    ..(imgpixbyte + c_long::from(overlap_bytes)) as usize],
                            );

                        tilepix += c_long::from(overlap_flags) * (inc[0]).abs();
                        if inc[0] > 0 {
                            imgpix += c_long::from(overlap_flags);
                        } else {
                            imgpix -= c_long::from(overlap_flags);
                        }
                    }
                    it1 += 1;
                }
                it2 += 1;
            }
            it3 += 1;
        }
        it4 += 1;
    }
    *status
}

/// Unquantize byte values into the scaled floating point values
///
/// # Parameters
///
/// * `row`            — tile number = row number in table
/// * `input`          — (I) array of values to be converted
/// * `ntodo`          — (I) number of elements in the array
/// * `scale`          — (I) FITS TSCALn or BSCALE value
/// * `zero`           — (I) FITS TZEROn or BZERO value
/// * `_dither_method` — (I) dithering method to use
/// * `nullcheck`      — (I) null checking code; 0 = don't check
/// * `tnull`          — (I) value of FITS TNonen keyword if any
/// * `nullval`        — (I) set null pixels, if nullcheck = 1
/// * `nullarray`      — (I) bad pixel array, if nullcheck = 2
/// * `anynull`        — (O) set to 1 if any pixels are null
/// * `output`         — (O) array of converted pixels
/// * `status`         — (IO) error status
fn unquantize_i1r4(
    row: c_long,
    input: &[c_uchar],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    _dither_method: c_int,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: c_uchar,
    nullval: f32,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f32],
    status: &mut c_int,
) -> c_int {
    let mut nextrand: c_int = 0;
    let mut iseed: c_int = 0;

    if FITS_RAND_VALUE.get().is_none() && fits_init_randoms() != 0 {
        return MEMORY_ALLOCATION;
    }

    let fits_rand_value = FITS_RAND_VALUE.get().unwrap();

    /* initialize the index to the next random number in the list */
    iseed = ((row - 1) % (N_RANDOM as c_long)) as c_int;
    nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;

    if nullcheck == NullCheckType::None {
        /* no null checking required */
        for ii in 0..(ntodo as usize) {
            /*
                      if (dither_method == SUBTRACTIVE_DITHER_2 &&
            input[ii] == ZERO_VALUE) output[ii] = 0.0; else
            */
            output[ii] = ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize])
                + 0.5)
                * scale
                + zero) as f32;

            nextrand += 1;
            if nextrand == (N_RANDOM as c_int) {
                iseed += 1;
                if iseed == (N_RANDOM as c_int) {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    } else {
        /* must check for null values */

        let anynull = anynull.unwrap(); // Original code assumes this is valid

        for ii in 0..(ntodo as usize) {
            if input[ii] == tnull {
                *anynull = 1;
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            } else {
                /*
                if (dither_method == SUBTRACTIVE_DITHER_2 &&
                 input[ii] == ZERO_VALUE) output[ii] = 0.0; else
                */
                output[ii] =
                    ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize]) + 0.5)
                        * scale
                        + zero) as f32;
            }

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    }

    *status
}

/// Unquantize short integer values into the scaled floating point values
///
/// # Parameters
///
/// * `row`            — seed for random values
/// * `input`          — (I) array of values to be converted
/// * `ntodo`          — (I) number of elements in the array
/// * `scale`          — (I) FITS TSCALn or BSCALE value
/// * `zero`           — (I) FITS TZEROn or BZERO value
/// * `_dither_method` — (I) dithering method to use
/// * `nullcheck`      — (I) null checking code; 0 = don't check
/// * `tnull`          — (I) value of FITS TNonen keyword if any
/// * `nullval`        — (I) set null pixels, if nullcheck = 1
/// * `nullarray`      — (I) bad pixel array, if nullcheck = 2
/// * `anynull`        — (O) set to 1 if any pixels are null
/// * `output`         — (O) array of converted pixels
/// * `status`         — (IO) error status
fn unquantize_i2r4(
    row: c_long,
    input: &[c_short],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    _dither_method: c_int,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: c_short,
    nullval: f32,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f32],
    status: &mut c_int,
) -> c_int {
    let mut nextrand: c_int = 0;
    let mut iseed: c_int = 0;

    if FITS_RAND_VALUE.get().is_none() && fits_init_randoms() != 0 {
        return MEMORY_ALLOCATION;
    }

    let fits_rand_value = FITS_RAND_VALUE.get().unwrap();

    /* initialize the index to the next random number in the list */
    iseed = ((row - 1) % (N_RANDOM as c_long)) as c_int;
    nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;

    if nullcheck == NullCheckType::None {
        /* no null checking required */

        for ii in 0..(ntodo as usize) {
            /*
                      if (dither_method == SUBTRACTIVE_DITHER_2 &&
            input[ii] == ZERO_VALUE) output[ii] = 0.0; else
            */
            output[ii] = ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize])
                + 0.5)
                * scale
                + zero) as f32;

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    } else {
        /* must check for null values */

        let anynull = anynull.unwrap(); // Original code assumes this is valid

        for ii in 0..(ntodo as usize) {
            if input[ii] == tnull {
                *anynull = 1;
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            } else {
                /*
                if (dither_method == SUBTRACTIVE_DITHER_2 &&
                 input[ii] == ZERO_VALUE) output[ii] = 0.0; else
                */
                output[ii] =
                    ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize]) + 0.5)
                        * scale
                        + zero) as f32;
            }

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    }

    *status
}

/// Unquantize int integer values into the scaled floating point values
///
/// # Parameters
///
/// * `row`           — tile number = row number in table
/// * `input`         — (I) array of values to be converted
/// * `ntodo`         — (I) number of elements in the array
/// * `scale`         — (I) FITS TSCALn or BSCALE value
/// * `zero`          — (I) FITS TZEROn or BZERO value
/// * `dither_method` — (I) dithering method to use
/// * `nullcheck`     — (I) null checking code; 0 = don't check
/// * `tnull`         — (I) value of FITS TNonen keyword if any
/// * `nullval`       — (I) set null pixels, if nullcheck = 1
/// * `nullarray`     — (I) bad pixel array, if nullcheck = 2
/// * `anynull`       — (O) set to 1 if any pixels are null
/// * `output`        — (O) array of converted pixels
/// * `status`        — (IO) error status
fn unquantize_i4r4(
    row: c_long,
    input: &[INT32BIT],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    dither_method: c_int,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: INT32BIT,
    nullval: f32,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f32],
    status: &mut c_int,
) -> c_int {
    let mut nextrand: c_int = 0;
    let mut iseed: c_int = 0;

    if FITS_RAND_VALUE.get().is_none() && fits_init_randoms() != 0 {
        return MEMORY_ALLOCATION;
    }

    let fits_rand_value = FITS_RAND_VALUE.get().unwrap();

    /* initialize the index to the next random number in the list */
    iseed = ((row - 1) % (N_RANDOM as c_long)) as c_int;
    nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;

    if nullcheck == NullCheckType::None {
        /* no null checking required */

        for ii in 0..(ntodo as usize) {
            if dither_method == SUBTRACTIVE_DITHER_2 && input[ii] == ZERO_VALUE {
                output[ii] = 0.0;
            } else {
                output[ii] =
                    ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize]) + 0.5)
                        * scale
                        + zero) as f32;
            }

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    } else {
        /* must check for null values */

        let anynull = anynull.unwrap(); // Original code assumes this is valid

        for ii in 0..(ntodo as usize) {
            if input[ii] == tnull {
                *anynull = 1;
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            } else if dither_method == SUBTRACTIVE_DITHER_2 && input[ii] == ZERO_VALUE {
                output[ii] = 0.0;
            } else {
                output[ii] =
                    ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize]) + 0.5)
                        * scale
                        + zero) as f32;
            }

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    }

    *status
}

/// Unquantize byte values into the scaled floating point values
///
/// # Parameters
///
/// * `row`            — tile number = row number in table
/// * `input`          — (I) array of values to be converted
/// * `ntodo`          — (I) number of elements in the array
/// * `scale`          — (I) FITS TSCALn or BSCALE value
/// * `zero`           — (I) FITS TZEROn or BZERO value
/// * `_dither_method` — (I) dithering method to use
/// * `nullcheck`      — (I) null checking code; 0 = don't check
/// * `tnull`          — (I) value of FITS TNonen keyword if any
/// * `nullval`        — (I) set null pixels, if nullcheck = 1
/// * `nullarray`      — (I) bad pixel array, if nullcheck = 2
/// * `anynull`        — (O) set to 1 if any pixels are null
/// * `output`         — (O) array of converted pixels
/// * `status`         — (IO) error status
fn unquantize_i1r8(
    row: c_long,
    input: &[c_uchar],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    _dither_method: c_int,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: c_uchar,
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
) -> c_int {
    let mut nextrand: c_int = 0;
    let mut iseed: c_int = 0;

    if FITS_RAND_VALUE.get().is_none() && fits_init_randoms() != 0 {
        return MEMORY_ALLOCATION;
    }

    let fits_rand_value = FITS_RAND_VALUE.get().unwrap();

    /* initialize the index to the next random number in the list */
    iseed = ((row - 1) % (N_RANDOM as c_long)) as c_int;
    nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;

    if nullcheck == NullCheckType::None {
        /* no null checking required */

        for ii in 0..(ntodo as usize) {
            /*
                      if (dither_method == SUBTRACTIVE_DITHER_2 &&
            input[ii] == ZERO_VALUE) output[ii] = 0.0; else
            */
            output[ii] = ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize])
                + 0.5)
                * scale
                + zero) as f64;

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    } else {
        /* must check for null values */

        let anynull = anynull.unwrap(); // Original code assumes this is valid

        for ii in 0..(ntodo as usize) {
            if input[ii] == tnull {
                *anynull = 1;
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            } else {
                /*
                                  if (dither_method == SUBTRACTIVE_DITHER_2 &&
                 input[ii] == ZERO_VALUE) output[ii] = 0.0; else
                */
                output[ii] =
                    ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize]) + 0.5)
                        * scale
                        + zero) as f64;
            }

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    }

    *status
}

/// # Parameters
///
/// * `row`            — tile number = row number in table
/// * `input`          — (I) array of values to be converted
/// * `ntodo`          — (I) number of elements in the array
/// * `scale`          — (I) FITS TSCALn or BSCALE value
/// * `zero`           — (I) FITS TZEROn or BZERO value
/// * `_dither_method` — (I) dithering method to use
/// * `nullcheck`      — (I) null checking code; 0 = don't check
/// * `tnull`          — (I) value of FITS TNonen keyword if any
/// * `nullval`        — (I) set null pixels, if nullcheck = 1
/// * `nullarray`      — (I) bad pixel array, if nullcheck = 2
/// * `anynull`        — (O) set to 1 if any pixels are null
/// * `output`         — (O) array of converted pixels
/// * `status`         — (IO) error status
fn unquantize_i2r8(
    row: c_long,
    input: &[c_short],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    _dither_method: c_int,
    nullcheck: NullCheckType,
    /*     1:set null pixels = nullval         */
    /*     2: if null pixel, set nullarray = 1 */
    tnull: c_short,
    nullval: f64,
    nullarray: &mut [c_char],
    anynull: Option<&mut c_int>,
    output: &mut [f64],
    status: &mut c_int,
    /*
    Unquantize short integer values into the scaled floating point values
    */
) -> c_int {
    let mut nextrand: c_int = 0;
    let mut iseed: c_int = 0;

    if FITS_RAND_VALUE.get().is_none() && fits_init_randoms() != 0 {
        return MEMORY_ALLOCATION;
    }

    let fits_rand_value = FITS_RAND_VALUE.get().unwrap();
    /* initialize the index to the next random number in the list */
    iseed = ((row - 1) % (N_RANDOM as c_long)) as c_int;
    nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;

    if nullcheck == NullCheckType::None {
        /* no null checking required */

        for ii in 0..(ntodo as usize) {
            /*
                      if (dither_method == SUBTRACTIVE_DITHER_2 &&
            input[ii] == ZERO_VALUE) output[ii] = 0.0; else
            */
            output[ii] = ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize])
                + 0.5)
                * scale
                + zero) as f64;

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    } else {
        /* must check for null values */

        let anynull = anynull.unwrap(); // Original code assumes this is valid

        for ii in 0..(ntodo as usize) {
            if input[ii] == tnull {
                *anynull = 1;
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            } else {
                /*                    if (dither_method == SUBTRACTIVE_DITHER_2
                 && input[ii] == ZERO_VALUE) output[ii] = 0.0; else
                */
                output[ii] =
                    ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize]) + 0.5)
                        * scale
                        + zero) as f64;
            }

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    }

    *status
}

/// Unquantize int integer values into the scaled floating point values
///
/// # Parameters
///
/// * `row`           — tile number = row number in table
/// * `input`         — (I) array of values to be converted
/// * `ntodo`         — (I) number of elements in the array
/// * `scale`         — (I) FITS TSCALn or BSCALE value
/// * `zero`          — (I) FITS TZEROn or BZERO value
/// * `dither_method` — (I) dithering method to use
/// * `nullcheck`     — (I) null checking code; 0 = don't check
/// * `tnull`         — (I) value of FITS TNonen keyword if any
/// * `nullval`       — (I) set null pixels, if nullcheck = 1
/// * `nullarray`     — (I) bad pixel array, if nullcheck = 2
/// * `anynull`       — (O) set to 1 if any pixels are null
/// * `output`        — (O) array of converted pixels
/// * `status`        — (IO) error status
fn unquantize_i4r8(
    row: c_long,
    input: &[INT32BIT],
    ntodo: c_long,
    scale: f64,
    zero: f64,
    dither_method: c_int,
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
    let mut nextrand: c_int = 0;
    let mut iseed: c_int = 0;

    if FITS_RAND_VALUE.get().is_none() && fits_init_randoms() != 0 {
        return MEMORY_ALLOCATION;
    }

    let fits_rand_value = FITS_RAND_VALUE.get().unwrap();

    /* initialize the index to the next random number in the list */
    iseed = ((row - 1) % (N_RANDOM as c_long)) as c_int;
    nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;

    if nullcheck == NullCheckType::None {
        /* no null checking required */

        for ii in 0..(ntodo as usize) {
            if dither_method == SUBTRACTIVE_DITHER_2 && input[ii] == ZERO_VALUE {
                output[ii] = 0.0;
            } else {
                output[ii] =
                    ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize]) + 0.5)
                        * scale
                        + zero) as f64;
            }

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    } else {
        /* must check for null values */

        let anynull = anynull.unwrap(); // Original code assumes this is valid

        for ii in 0..(ntodo as usize) {
            if input[ii] == tnull {
                *anynull = 1;
                if nullcheck == NullCheckType::SetPixel {
                    output[ii] = nullval;
                } else {
                    nullarray[ii] = 1;
                }
            } else if dither_method == SUBTRACTIVE_DITHER_2 && input[ii] == ZERO_VALUE {
                output[ii] = 0.0;
            } else {
                output[ii] =
                    ((f64::from(input[ii]) - f64::from(fits_rand_value[nextrand as usize]) + 0.5)
                        * scale
                        + zero) as f64;
            }

            nextrand += 1;
            if nextrand == N_RANDOM as c_int {
                iseed += 1;
                if iseed == N_RANDOM as c_int {
                    iseed = 0;
                }
                nextrand = (fits_rand_value[iseed as usize] * 500.0) as c_int;
            }
        }
    }

    *status
}

/// Convert pixels that are equal to nullflag to NaNs.
/// Note that indata and outdata point to the same location.
fn imcomp_float2nan(
    indata: &[f32],
    tilelen: c_long,
    outdata: &mut [c_int],
    nullflagval: f32,
    status: &mut c_int,
) -> c_int {
    for ii in 0..(tilelen as usize) {
        if indata[ii] == nullflagval {
            outdata[ii] = -1; /* integer -1 has the same bit pattern as a real*4 NaN */
        }
    }

    *status
}

/// Convert pixels that are equal to nullflag to NaNs.
/// Note that indata and outdata point to the same location.
fn imcomp_float2nan_inplace(
    indata: &mut [f32],
    tilelen: c_long,
    nullflagval: f32,
    status: &mut c_int,
) -> c_int {
    for ii in 0..(tilelen as usize) {
        if indata[ii] == nullflagval {
            indata[ii] = cast::<c_int, f32>(-1); /* integer -1 has the same bit pattern as a real*4 NaN */
        }
    }

    *status
}

/// Convert pixels that are equal to nullflag to NaNs.
/// Note that indata and outdata point to the same location.
fn imcomp_double2nan(
    indata: &[f64],
    tilelen: c_long,
    outdata: &mut [LONGLONG],
    nullflagval: f64,
    status: &mut c_int,
) -> c_int {
    for ii in 0..(tilelen as usize) {
        if indata[ii] == nullflagval {
            outdata[ii] = -1; /* integer -1 has the same bit pattern as a real*8 NaN */
        }
    }

    *status
}

/// Convert pixels that are equal to nullflag to NaNs.
/// Note that indata and outdata point to the same location.
fn imcomp_double2nan_inplace(
    indata: &mut [f64],
    tilelen: c_long,
    nullflagval: f64,
    status: &mut c_int,
) -> c_int {
    for ii in 0..(tilelen as usize) {
        if indata[ii] == nullflagval {
            indata[ii] = cast::<LONGLONG, f64>(-1); /* integer -1 has the same bit pattern as a real*8 NaN */
        }
    }

    *status
}

/* ======================================================================= */
/*    TABLE COMPRESSION ROUTINES                                           */
/* =-====================================================================== */

/// Compress the input FITS Binary Table.
///
/// First divide the table into equal sized chunks (analogous to image tiles)
/// where all the contain the same number of rows (except perhaps for the last
/// chunk which may contain fewer rows).   The chunks should not be too large to
/// copy into memory (currently, about 100 MB max seems a reasonable size).
///
/// Then, on a chunk by piece basis, do the following:
///
/// 1. Transpose the table from its original row-major order, into column-major
///    order. All the bytes for each column are then continuous.  In addition, the
///    bytes within each table element may be shuffled so that the most significant
///    byte of every element occurs first in the array, followed by the next most
///    significant byte, and so on to the least significant byte.  Byte shuffling
///    often improves the gzip compression of floating-point arrays.
///
/// 2. Compress the contiguous array of bytes in each column using the specified
///    compression method.  If no method is specifed, then a default method for that
///    data type is chosen.
///
/// 3. Store the compressed stream of bytes into a column that has the same name
///    as in the input table, but which has a variable-length array data type (1QB).
///    The output table will contain one row for each piece of the original table.
///
/// 4. If the input table contain variable-length arrays, then each VLA
///    is compressed individually, and written to the heap in the output table.
///    Note that the output table will contain 2 sets of pointers for each VLA
///    column. The first set contains the pointers to the uncompressed VLAs from the
///    input table and the second is the set of pointers to the compressed VLAs in
///    the output table. The latter set of pointers is used to reconstruct table when
///    it is uncompressed, so that the heap has exactly the same structure as in the
///    original file.  The 2 sets of pointers are concatinated together, compressed
///    with gzip, and written to the output table.  When reading the compressed
///    table, the only VLA that is directly visible is this compressed array of
///    descriptors.  One has to uncompress this array to be able to to read all the
///    descriptors to the individual VLAs in the column.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_compress_table(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        // Call the safer version of the function
        fits_compress_table_safe(infptr, outfptr, status)
    }
}

/// Compress a binary table, writing the result to `outfptr`.
///
/// The table is transposed from row-major to column-major in chunks, so that
/// each column's values sit together and compress well, then each column is
/// compressed with the algorithm suited to its datatype.
///
/// # Parameters
///
/// * `infptr`  — (I) FITS file pointer to the input table
/// * `outfptr` — (I) FITS file pointer to the output table
/// * `status`  — (IO) error status
pub fn fits_compress_table_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let maxchunksize: c_long = 10000000; // default value for the size of each chunk of the table

    let mut cm_buffer: Vec<c_char>; // memory buffer for the transposed, Column-Major, chunk of the table
    let mut cm_colstart: [LONGLONG; 1000] = [0; 1000]; // starting offset of each column in the cm_buffer
    let mut rm_repeat: [LONGLONG; 1000] = [0; 1000]; // repeat count of each column in the input row-major table
    let mut rm_colwidth: [LONGLONG; 999] = [0; 999]; // width in bytes of each column in the input row-major table
    let mut cm_repeat: [LONGLONG; 999] = [0; 999]; // total number of elements in each column of the transposed column-major table

    let mut coltype: [c_int; 999] = [0; 999]; // data type code for each column
    let mut compalgor: [c_int; 999] = [0; 999]; // compression algorithm to be applied to each column
    let mut default_algor: c_int = 0;
    let mut cratio: [f32; 999] = [0.0; 999]; // compression ratio for each column (for diagnostic purposes)

    let mut compressed_size: f32;
    let mut uncompressed_size: f32;
    let mut tot_compressed_size: f32;
    let mut tot_uncompressed_size: f32;
    let mut nrows: LONGLONG = 0;

    let mut headstart: LONGLONG = 0;
    let mut datastart: LONGLONG = 0;
    let mut dataend: LONGLONG = 0;
    let mut startbyte: LONGLONG = 0;
    let mut kk: LONGLONG;
    let mut naxis1: LONGLONG = 0;
    let mut vlalen: LONGLONG = 0;
    let mut vlamemlen: LONGLONG;
    let mut vlastart: LONGLONG;
    let mut bytepos: LONGLONG;
    let mut repeat: c_long = 0;
    let mut width: c_long = 0;
    let mut nchunks: c_long = 0;
    let mut rowspertile: c_long = 0;
    let mut lastrows: c_long = 0;
    let mut ncols: c_int = 0;
    let mut hdutype: c_int = 0;
    let ltrue: c_int = 1;
    let mut print_report = false;
    let mut tstatus: c_int;

    let mut keyname: [c_char; 9] = [0; 9];
    let mut tform: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut cdescript: Vec<c_char> = Vec::new();
    let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut keyvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut cvlamem: Vec<u8> = Vec::new();
    let mut tempstring: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    let mut vlamem: Vec<LONGLONG> = Vec::new();

    let mut dlen: usize = 0;
    let mut datasize: usize;
    let mut compmemlen: usize;

    let mut results: [[c_char; 30]; 999] = [[0; 30]; 999]; // string array for storing the individual column compression stats

    /* ==================================================================================
     */
    /* perform initial sanity checks */
    /* ==================================================================================
     */

    /* special input flag value that means print out diagnostics */
    if *status == -999 {
        print_report = true;
        *status = 0;
    }

    if *status > 0 {
        return *status;
    }

    fits_get_hdu_type(infptr, &mut hdutype, status);
    if hdutype != BINARY_TBL {
        *status = NOT_BTABLE;
        return *status;
    }

    if core::ptr::eq(infptr, outfptr) {
        ffpmsg_str("Cannot compress table 'in place' (fits_compress_table)");
        ffpmsg_str(" outfptr cannot be the same as infptr.");
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    /* get dimensions of the table */
    fits_get_num_rowsll(infptr, &mut nrows, status);
    fits_get_num_cols(infptr, &mut ncols, status);
    fits_read_key(
        infptr,
        crate::KeywordDatatypeMut::TLONGLONG(&mut naxis1),
        cs!(c"NAXIS1"),
        None,
        status,
    );
    /* get offset to the start of the data and total size of the table (including the heap) */
    fits_get_hduaddrll(
        infptr,
        Some(&mut headstart),
        Some(&mut datastart),
        Some(&mut dataend),
        status,
    );

    if *status > 0 {
        return *status;
    }

    tstatus = 0;
    if fits_read_key(
        infptr,
        crate::KeywordDatatypeMut::TSTRING(&mut tempstring),
        cs!(c"FZALGOR"),
        None,
        &mut tstatus,
    ) == 0
    {
        if fits_strcasecmp(&tempstring, cs!(c"NONE")) == 0 {
            default_algor = NOCOMPRESS;
        } else if fits_strcasecmp(&tempstring, cs!(c"GZIP")) == 0
            || fits_strcasecmp(&tempstring, cs!(c"GZIP_1")) == 0
        {
            default_algor = GZIP_1;
        } else if fits_strcasecmp(&tempstring, cs!(c"GZIP_2")) == 0 {
            default_algor = GZIP_2;
        } else if fits_strcasecmp(&tempstring, cs!(c"RICE_1")) == 0 {
            default_algor = RICE_1;
        } else {
            ffpmsg_str("FZALGOR specifies unsupported table compression algorithm:");
            ffpmsg_slice(&tempstring);
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }
    }

    /* just copy the HDU verbatim if the table has 0 columns or rows or if the table */
    /* is less than 5760 bytes (2 blocks) in size, or compression directive keyword = "NONE" */
    if nrows < 1 || ncols < 1 || (dataend - datastart) < 5760 || default_algor == NOCOMPRESS {
        fits_copy_hdu(infptr, outfptr, 0, status);
        return *status;
    }

    /* Check if the chunk size has been specified with the FZTILELN keyword. */
    /* If not, calculate a default number of rows per chunck, */

    tstatus = 0;
    if fits_read_key(
        infptr,
        crate::KeywordDatatypeMut::TLONG(&mut rowspertile),
        cs!(c"FZTILELN"),
        None,
        &mut tstatus,
    ) != 0
    {
        rowspertile = (maxchunksize as LONGLONG / naxis1) as c_long;
    }

    if rowspertile < 1 {
        rowspertile = 1;
    }

    if rowspertile as LONGLONG > nrows {
        rowspertile = nrows as c_long;
    }

    nchunks = ((nrows - 1) / rowspertile as LONGLONG + 1) as c_long; /* total number of chunks */
    lastrows = (nrows - ((nchunks as LONGLONG - 1) * rowspertile as LONGLONG)) as c_long; /* number of rows in last chunk */

    /* allocate space for the transposed, column-major chunk of the table */
    cm_buffer = Vec::new();
    let tmp_len = naxis1 as usize * rowspertile as usize;
    if cm_buffer.try_reserve_exact(tmp_len).is_err() {
        ffpmsg_str("Could not allocate cm_buffer for transposed table");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        cm_buffer.resize(tmp_len, 0);
    }

    /* ==================================================================================
     */
    /*  Construct the header of the output compressed table  */
    /* ==================================================================================
     */
    fits_copy_header(infptr, outfptr, status); /* start with verbatim copy of the input header */

    fits_write_key(
        outfptr,
        crate::KeywordDatatype::TLOGICAL(&ltrue),
        cs!(c"ZTABLE"),
        Some(cs!(c"this is a compressed table")),
        status,
    );
    fits_write_key(
        outfptr,
        crate::KeywordDatatype::TLONG(&rowspertile),
        cs!(c"ZTILELEN"),
        Some(cs!(c"number of rows in each tile")),
        status,
    );

    fits_read_card(outfptr, cs!(c"NAXIS1"), &mut card, status); /* copy NAXIS1 to ZNAXIS1 */
    strncpy_safe(&mut card, cs!(c"ZNAXIS1"), 7);
    fits_write_record(outfptr, &card, status);

    fits_read_card(outfptr, cs!(c"NAXIS2"), &mut card, status); /* copy NAXIS2 to ZNAXIS2 */
    strncpy_safe(&mut card, cs!(c"ZNAXIS2"), 7);
    fits_write_record(outfptr, &card, status);

    fits_read_card(outfptr, cs!(c"PCOUNT"), &mut card, status); /* copy PCOUNT to ZPCOUNT */
    strncpy_safe(&mut card, cs!(c"ZPCOUNT"), 7);
    fits_write_record(outfptr, &card, status);

    fits_modify_key_lng(
        outfptr,
        cs!(c"NAXIS2"),
        nchunks as LONGLONG,
        Some(cs!(c"&")),
        status,
    ); /* 1 row per chunk */
    fits_modify_key_lng(
        outfptr,
        cs!(c"NAXIS1"),
        LONGLONG::from(ncols * 16),
        Some(cs!(c"&")),
        status,
    ); /* 16 bytes for each 1QB column */
    fits_modify_key_lng(outfptr, cs!(c"PCOUNT"), 0, Some(cs!(c"&")), status); /* reset PCOUNT to 0 */

    /* rename the Checksum keywords, if they exist */
    tstatus = 0;
    fits_modify_name(outfptr, cs!(c"CHECKSUM"), cs!(c"ZHECKSUM"), &mut tstatus);
    tstatus = 0;
    fits_modify_name(outfptr, cs!(c"DATASUM"), cs!(c"ZDATASUM"), &mut tstatus);

    /* ==================================================================================
     */
    /*  Now loop over each column of the input table: write the column-specific keywords */
    /*  and determine which compression algorithm to use.     */
    /*  Also calculate various offsets to the start of the column data in both the */
    /*  original row-major table and in the transposed column-major form of the table.  */
    /* ==================================================================================
     */

    cm_colstart[0] = 0;
    for ii in 0..(ncols as usize) {
        /* get the structural parameters of the original uncompressed column */
        fits_make_keyn(
            cs!(c"TFORM"),
            (ii + 1).try_into().unwrap(),
            &mut keyname,
            status,
        );
        fits_read_key(
            outfptr,
            crate::KeywordDatatypeMut::TSTRING(&mut tform),
            &keyname,
            Some(&mut comm),
            status,
        );

        fits_binary_tform(
            &tform,
            Some(&mut coltype[ii]),
            Some(&mut repeat),
            Some(&mut width),
            status,
        ); /* get the repeat count and the width */

        /* preserve the original TFORM value and comment string in a ZFORMn keyword */
        fits_read_card(outfptr, &keyname, &mut card, status);
        card[0] = bb(b'Z');
        fits_write_record(outfptr, &card, status);

        /* All columns in the compressed table will have a variable-length array type. */
        fits_modify_key_str(outfptr, &keyname, cs!(c"1QB"), Some(cs!(c"&")), status); /* Use 'Q' pointers (64-bit) */

        /* deal with special cases: bit, string, and variable length array columns */
        if coltype[ii] == TBIT {
            repeat = (repeat + 7) / 8; /* convert from bits to equivalent number of bytes */
        } else if coltype[ii] == TSTRING {
            width = 1; /* ignore the optional 'w' in 'rAw' format */
        } else if coltype[ii] < 0 {
            /* pointer to variable length array */
            if strchr_safe(&tform, bb(b'Q')).is_some() {
                width = 16; /* 'Q' descriptor has 64-bit pointers */
            } else {
                width = 8; /* 'P' descriptor has 32-bit pointers */
            }
            repeat = 1;
        }

        rm_repeat[ii] = repeat as LONGLONG;
        rm_colwidth[ii] = (repeat * width) as LONGLONG; /* column width (in bytes)in the input table */

        /* starting offset of each field in the OUTPUT transposed column-major table */
        cm_colstart[ii + 1] = cm_colstart[ii] + rm_colwidth[ii] * rowspertile as LONGLONG;
        /* total number of elements in each column of the transposed column-major table */
        cm_repeat[ii] = rm_repeat[ii] * rowspertile as LONGLONG;

        compalgor[ii] = default_algor; /* initialize the column compression
        algorithm to the default */

        /*  check if a compression method has been specified for this column */
        fits_make_keyn(cs!(c"FZALG"), (ii as c_int) + 1, &mut keyname, status);
        tstatus = 0;
        if fits_read_key(
            outfptr,
            crate::KeywordDatatypeMut::TSTRING(&mut tempstring),
            &keyname,
            None,
            &mut tstatus,
        ) == 0
        {
            if fits_strcasecmp(&tempstring, cs!(c"GZIP")) == 0
                || fits_strcasecmp(&tempstring, cs!(c"GZIP_1")) == 0
            {
                compalgor[ii] = GZIP_1;
            } else if fits_strcasecmp(&tempstring, cs!(c"GZIP_2")) == 0 {
                compalgor[ii] = GZIP_2;
            } else if fits_strcasecmp(&tempstring, cs!(c"RICE_1")) == 0 {
                compalgor[ii] = RICE_1;
            } else {
                ffpmsg_str("Unsupported table compression algorithm specification.");
                ffpmsg_slice(&keyname);
                ffpmsg_slice(&tempstring);
                *status = DATA_COMPRESSION_ERR;
                return *status;
            }
        }

        /* do sanity check of the requested algorithm and override if necessary
         */
        if (coltype[ii]).abs() == TLOGICAL
            || (coltype[ii]).abs() == TBIT
            || (coltype[ii]).abs() == TSTRING
        {
            if compalgor[ii] != GZIP_1 {
                compalgor[ii] = GZIP_1;
            }
        } else if (coltype[ii]).abs() == TCOMPLEX
            || (coltype[ii]).abs() == TDBLCOMPLEX
            || (coltype[ii]).abs() == TFLOAT
            || (coltype[ii]).abs() == TDOUBLE
            || (coltype[ii]).abs() == TLONGLONG
        {
            if compalgor[ii] != GZIP_1 && compalgor[ii] != GZIP_2 {
                compalgor[ii] = GZIP_2; /* gzip_2 usually works better gzip_1 */
            }
        } else if (coltype[ii]).abs() == TSHORT {
            if compalgor[ii] != GZIP_1 && compalgor[ii] != GZIP_2 && compalgor[ii] != RICE_1 {
                compalgor[ii] = GZIP_2; /* gzip_2 usually works better rice_1 */
            }
        } else if (coltype[ii]).abs() == TLONG {
            if compalgor[ii] != GZIP_1 && compalgor[ii] != GZIP_2 && compalgor[ii] != RICE_1 {
                compalgor[ii] = RICE_1;
            }
        } else if (coltype[ii]).abs() == TBYTE && compalgor[ii] != GZIP_1 && compalgor[ii] != RICE_1
        {
            compalgor[ii] = GZIP_1;
        }
    } /* end of loop over columns */

    /* ==================================================================================
     */
    /*    now process each chunk of the table, in turn          */
    /* ==================================================================================
     */

    tot_uncompressed_size = 0.0;
    tot_compressed_size = 0.0;
    // let mut firstrow: LONGLONG = 1;
    for ll in 0..(nchunks as usize) {
        if ll as c_long == nchunks - 1 {
            /* the last chunk may have fewer rows */
            rowspertile = lastrows;
            for ii in 0..(ncols as usize) {
                cm_colstart[ii + 1] = cm_colstart[ii] + (rm_colwidth[ii] * rowspertile as LONGLONG);
                cm_repeat[ii] = rm_repeat[ii] * rowspertile as LONGLONG;
            }
        }

        /* move to the start of the chunk in the input table */
        ffmbyt_safe(infptr, datastart, 0, status);

        /* ================================================================================*/
        /*  First, transpose this chunck from row-major order to column-major order  */
        /*  At the same time, shuffle the bytes in each datum, if doing GZIP_2 compression */
        /* ================================================================================*/

        let mut cptr: usize = 0; // into cm_buffer

        for jj in 0..(rowspertile as usize) {
            /* loop over rows */
            for ii in 0..(ncols as usize) {
                /* loop over columns */

                if rm_repeat[ii] > 0 {
                    /*  skip virtual columns that have 0 elements */

                    kk = 0;

                    /* if the  GZIP_2 compression algorithm is used, shuffle the bytes */
                    if coltype[ii] == TSHORT && compalgor[ii] == GZIP_2 {
                        while kk < rm_colwidth[ii] {
                            cptr = (cm_colstart[ii]
                                + (jj as LONGLONG * rm_repeat[ii])
                                + kk as LONGLONG / 2) as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 1st byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 2nd byte */
                            kk += 2;
                        }
                    } else if (coltype[ii] == TFLOAT || coltype[ii] == TLONG)
                        && compalgor[ii] == GZIP_2
                    {
                        while kk < rm_colwidth[ii] {
                            cptr = (cm_colstart[ii]
                                + (jj as LONGLONG * rm_repeat[ii])
                                + kk as LONGLONG / 4) as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 1st byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 2nd byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 3rd byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 4th byte */
                            kk += 4;
                        }
                    } else if (coltype[ii] == TDOUBLE || coltype[ii] == TLONGLONG)
                        && compalgor[ii] == GZIP_2
                    {
                        while kk < rm_colwidth[ii] {
                            cptr = (cm_colstart[ii]
                                + (jj as LONGLONG * rm_repeat[ii])
                                + kk as LONGLONG / 8) as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 1st byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 2nd byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 3rd byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 4th byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 5th byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 6th byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 7th byte */
                            cptr += cm_repeat[ii] as usize;
                            ffgbyt(infptr, 1, cast_slice_mut(&mut cm_buffer[cptr..]), status); /* get 8th byte */
                            kk += 8;
                        }
                    } else {
                        /* all other cases: don't shuffle the bytes; simply
                        transpose the column */
                        cptr = (cm_colstart[ii] + (jj as LONGLONG * rm_colwidth[ii])) as usize; /* addr to copy to */
                        startbyte = (infptr.Fptr).bytepos; /* save the starting byte location */
                        ffgbyt(
                            infptr,
                            rm_colwidth[ii],
                            cast_slice_mut(&mut cm_buffer[cptr..]),
                            status,
                        ); /* copy all the bytes */

                        if rm_colwidth[ii] >= MINDIRECT {
                            /* have to explicitly move to next byte */
                            ffmbyt_safe(infptr, startbyte + rm_colwidth[ii], 0, status);
                        }
                    } /* end of test of coltypee */
                } /* end of not virtual column */
            } /* end of loop over columns */
        } /* end of loop over rows */

        /* ================================================================================*/
        /*  now compress each column in the transposed chunk of the table    */
        /* ================================================================================*/

        fits_set_hdustruc(outfptr, status); /* initialize structures in the output table */

        for ii in 0..(ncols as usize) {
            /* loop over columns */
            /* initialize the diagnostic compression results string */
            int_snprintf!(
                results[ii],
                30,
                "{:3} {:3} {:3} ",
                ii + 1,
                coltype[ii],
                compalgor[ii],
            );
            cratio[ii] = 0.0;

            if rm_repeat[ii] > 0 {
                /* skip virtual columns with zero width */

                if coltype[ii] < 0 {
                    /* this is a variable length array (VLA) column */

                    /* variable-length array columns are a complicated special case  */

                    /* allocate memory to hold all the VLA descriptors from the input table, plus */
                    /* room to hold the descriptors to the compressed VLAs in the output table */
                    /* In total, there will be 2 descriptors for each row in this chunk */

                    uncompressed_size = 0.;
                    compressed_size = 0.0;

                    datasize = (cm_colstart[ii + 1] - cm_colstart[ii]) as usize; /* size of input descriptors */

                    /* room for both descriptors */
                    let tmp_len = datasize + (rowspertile as usize * 16);
                    if cdescript.try_reserve_exact(tmp_len).is_err() {
                        ffpmsg_str("Could not allocate buffer for descriptors");
                        *status = MEMORY_ALLOCATION;
                        return *status;
                    } else {
                        cdescript.resize(tmp_len, 0);
                    }

                    /* copy the input descriptors to this array */
                    cdescript[..datasize].copy_from_slice(cast_slice(
                        &cm_buffer[cm_colstart[ii] as usize..(cm_colstart[ii] as usize + datasize)],
                    ));

                    if BYTESWAPPED {
                        /* byte-swap the integer values into the native machine representation */
                        if rm_colwidth[ii] == 16 {
                            ffswap8(cast_slice_mut(&mut cdescript), rowspertile * 2);
                        } else {
                            ffswap4(cast_slice_mut(&mut cdescript), rowspertile * 2);
                        }
                    }

                    /* pointer to the 2nd set of descriptors */

                    for jj in 0..(rowspertile as usize) {
                        /* loop to compress each VLA in turn */

                        if rm_colwidth[ii] == 16 {
                            /* if Q pointers */
                            let descriptors: &[LONGLONG] = cast_slice(&cdescript); /* use this for Q type descriptors */
                            vlalen = descriptors[jj * 2];
                            vlastart = descriptors[(jj * 2) + 1];
                        } else {
                            /* if P pointers */
                            let pdescriptors: &[c_int] = cast_slice(&cdescript); /* use this instead for or P type descriptors */
                            vlalen = LONGLONG::from(pdescriptors[jj * 2]);
                            vlastart = LONGLONG::from(pdescriptors[(jj * 2) + 1]);
                        }

                        if vlalen > 0 {
                            /* skip zero-length VLAs */

                            vlamemlen = vlalen * LONGLONG::from(-coltype[ii] / 10);

                            /* memory for the input uncompressed VLA */
                            if vlamem.try_reserve_exact(vlamemlen as usize).is_err() {
                                ffpmsg_str("Could not allocate buffer for VLA");
                                *status = MEMORY_ALLOCATION;
                                return *status;
                            } else {
                                vlamem.resize(vlamemlen as usize, 0);
                            }

                            compmemlen = ((vlalen * LONGLONG::from(-coltype[ii] / 10)) as f64 * 1.5)
                                as usize;
                            if compmemlen < 100 {
                                compmemlen = 100;
                            }

                            /* memory for the output compressed VLA */
                            if cvlamem.try_reserve_exact(compmemlen).is_err() {
                                ffpmsg_str("Could not allocate buffer for compressed data");
                                *status = MEMORY_ALLOCATION;
                                return *status;
                            } else {
                                cvlamem.resize(compmemlen, 0);
                            }

                            /* read the raw bytes directly from the heap, without any byte-swapping or null value detection */
                            bytepos = (infptr.Fptr).datastart + (infptr.Fptr).heapstart + vlastart;
                            ffmbyt_safe(infptr, bytepos, REPORT_EOF, status);
                            ffgbyt(infptr, vlamemlen, cast_slice_mut(&mut vlamem), status); /* read the bytes */
                            uncompressed_size += vlamemlen as f32; /* total size of the uncompressed VLAs */
                            tot_uncompressed_size += vlamemlen as f32; /* total size of the uncompressed file */

                            /* compress the VLA with the appropriate algorithm */
                            if compalgor[ii] == RICE_1 {
                                /* A slice, not the Vec: `Write for Vec'
                                appends, past the zeros it was sized to. */
                                let mut rce = RCEncoder::new(cvlamem.as_mut_slice());
                                rce.set_log_fn(ffpmsg_str);

                                if -coltype[ii] == TSHORT {
                                    if BYTESWAPPED {
                                        ffswap2(cast_slice_mut(&mut vlamem), vlalen as c_long);
                                    }

                                    let r =
                                        rce.encode_short(cast_slice(&vlamem), vlalen as usize, 32);
                                    match r {
                                        Ok(x) => dlen = x,
                                        Err(_e) => {
                                            *status = DATA_COMPRESSION_ERR;
                                            // return *status;
                                        }
                                    }
                                } else if -coltype[ii] == TLONG {
                                    if BYTESWAPPED {
                                        ffswap4(cast_slice_mut(&mut vlamem), vlalen as c_long);
                                    }

                                    let r = rce.encode(cast_slice(&vlamem), vlalen as usize, 32);
                                    match r {
                                        Ok(x) => dlen = x,
                                        Err(_e) => {
                                            *status = DATA_COMPRESSION_ERR;
                                            // return *status;
                                        }
                                    }
                                } else if -coltype[ii] == TBYTE {
                                    let r =
                                        rce.encode_byte(cast_slice(&vlamem), vlalen as usize, 32);
                                    match r {
                                        Ok(x) => dlen = x,
                                        Err(_e) => {
                                            *status = DATA_COMPRESSION_ERR;
                                            // return *status;
                                        }
                                    }
                                } else {
                                    /* this should not happen */
                                    ffpmsg_str(
                                        " Error: cannot compress this column type with the RICE algorithm",
                                    );
                                    *status = DATA_COMPRESSION_ERR;
                                    return *status;
                                }
                            } else if compalgor[ii] == GZIP_1 || compalgor[ii] == GZIP_2 {
                                if compalgor[ii] == GZIP_2 {
                                    /* shuffle the bytes before
                                    gzipping them */
                                    if ((-coltype[ii] / 10) as c_int) == 2 {
                                        fits_shuffle_2bytes(
                                            cast_slice_mut(&mut vlamem),
                                            vlalen,
                                            status,
                                        );
                                    } else if ((-coltype[ii] / 10) as c_int) == 4 {
                                        fits_shuffle_4bytes(
                                            cast_slice_mut(&mut vlamem),
                                            vlalen,
                                            status,
                                        );
                                    } else if ((-coltype[ii] / 10) as c_int) == 8 {
                                        fits_shuffle_8bytes(
                                            cast_slice_mut(&mut vlamem),
                                            vlalen,
                                            status,
                                        );
                                    }
                                }
                                /*: gzip compress the array of bytes */
                                compress2mem_from_mem(
                                    cast_slice(&vlamem),
                                    vlamemlen as usize,
                                    &mut cvlamem,
                                    Some(&mut dlen),
                                    status,
                                );
                            } else {
                                /* this should not happen */
                                ffpmsg_str(" Error: unknown compression algorithm");
                                *status = DATA_COMPRESSION_ERR;
                                return *status;
                            }

                            /* write the compressed array to the output table, but... */
                            /* We use a trick of always writing the array to the same row of the output table */
                            /* and then copy the descriptor into the array of descriptors that we allocated. */

                            /* First, reset the descriptor */
                            fits_write_descript(
                                outfptr,
                                (ii + 1) as c_int,
                                (ll + 1) as LONGLONG,
                                0,
                                0,
                                status,
                            );

                            /* write the compressed VLA if it is smaller than the original, else write */
                            /* the uncompressed array */
                            fits_set_tscale(outfptr, (ii + 1) as c_int, 1.0, 0.0, status); /* turn off any data scaling, first */
                            if dlen < vlamemlen as usize {
                                fits_write_col(
                                    outfptr,
                                    TBYTE,
                                    (ii + 1) as c_int,
                                    (ll + 1) as LONGLONG,
                                    1,
                                    dlen as LONGLONG,
                                    &cvlamem,
                                    status,
                                );
                                compressed_size += dlen as f32; /* total size of the compressed VLAs */
                                tot_compressed_size += dlen as f32; /* total size of the compressed file */
                            } else {
                                if -coltype[ii] != TBYTE && compalgor[ii] != GZIP_1 {
                                    /* it is probably faster to reread the raw bytes, rather than unshuffle or unswap them */
                                    bytepos = (infptr.Fptr).datastart
                                        + (infptr.Fptr).heapstart
                                        + vlastart;
                                    ffmbyt_safe(infptr, bytepos, REPORT_EOF, status);
                                    ffgbyt(infptr, vlamemlen, cast_slice_mut(&mut vlamem), status); /* read the bytes */
                                }
                                fits_write_col(
                                    outfptr,
                                    TBYTE,
                                    (ii + 1) as c_int,
                                    (ll + 1) as LONGLONG,
                                    1,
                                    vlamemlen,
                                    cast_slice(&vlamem),
                                    status,
                                );
                                compressed_size += vlamemlen as f32; /* total size of the compressed VLAs */
                                tot_compressed_size += vlamemlen as f32; /* total size of the compressed file */
                            }

                            /* read back the descriptor and save it in the array of descriptors */
                            let outdescript: &mut [LONGLONG] =
                                cast_slice_mut(&mut cdescript[datasize..]); // cdescript + datasize as LONGLONG
                            let mut read_len = outdescript[jj * 2];
                            let mut read_heapaddr = outdescript[jj * 2 + 1];
                            fits_read_descriptll(
                                outfptr,
                                (ii + 1) as c_int,
                                (ll + 1) as LONGLONG,
                                Some(&mut read_len),
                                Some(&mut read_heapaddr),
                                status,
                            );

                            outdescript[jj * 2] = read_len;
                            outdescript[jj * 2 + 1] = read_heapaddr;
                        } /* end of vlalen > 0 */
                    } /* end of loop over rows */

                    if compressed_size != 0.0 {
                        cratio[ii] = uncompressed_size / compressed_size;
                    }

                    int_snprintf!(tempstring, FLEN_VALUE, " r={:6.2}", cratio[ii]);
                    let len = strlen_safe(&results[ii]);
                    strncat_safe(&mut results[ii], &tempstring, 29 - len);

                    /* now we just have to compress the array of descriptors (both input and output) */
                    /* and write them to the output table. */

                    /* allocate memory for the compressed descriptors */
                    if cvlamem
                        .try_reserve_exact(datasize + (rowspertile as usize * 16))
                        .is_err()
                    {
                        ffpmsg_str("Could not allocate buffer for compressed data");
                        *status = MEMORY_ALLOCATION;
                        return *status;
                    } else {
                        cvlamem.resize(datasize + (rowspertile as usize * 16), 0);
                    }

                    if BYTESWAPPED {
                        /* byte swap the input and output descriptors */
                        if rm_colwidth[ii] == 16 {
                            ffswap8(cast_slice_mut(&mut cdescript), rowspertile * 2);
                        } else {
                            ffswap4(cast_slice_mut(&mut cdescript), rowspertile * 2);
                        }

                        let outdescript: &mut [LONGLONG] =
                            cast_slice_mut(&mut cdescript[datasize..]); // cdescript + datasize as LONGLONG
                        ffswap8(outdescript, rowspertile * 2);
                    }
                    /* compress the array contain both sets of descriptors */
                    compress2mem_from_mem(
                        &cdescript,
                        datasize + (rowspertile * 16) as usize,
                        &mut cvlamem,
                        Some(&mut dlen),
                        status,
                    );

                    /* write the compressed descriptors to the output column */
                    fits_set_tscale(outfptr, (ii + 1) as c_int, 1.0, 0.0, status); /* turn off any data scaling, first */
                    fits_write_descript(
                        outfptr,
                        (ii + 1) as c_int,
                        (ll + 1) as LONGLONG,
                        0,
                        0,
                        status,
                    ); /* First, reset the descriptor */
                    fits_write_col(
                        outfptr,
                        TBYTE,
                        (ii + 1) as c_int,
                        (ll + 1) as LONGLONG,
                        1,
                        dlen.try_into().unwrap(),
                        cast_slice(&cvlamem),
                        status,
                    );

                    if ll == 0 {
                        /* only write the ZCTYPn keyword once, while
                        processing the first column */
                        fits_make_keyn(cs!(c"ZCTYP"), (ii + 1) as c_int, &mut keyname, status);

                        if compalgor[ii] == RICE_1 {
                            strcpy_safe(&mut keyvalue, cs!(c"RICE_1"));
                        } else if compalgor[ii] == GZIP_2 {
                            strcpy_safe(&mut keyvalue, cs!(c"GZIP_2"));
                        } else {
                            strcpy_safe(&mut keyvalue, cs!(c"GZIP_1"));
                        }

                        fits_write_key(
                            outfptr,
                            crate::KeywordDatatype::TSTRING(&keyvalue),
                            &keyname,
                            Some(cs!(c"compression algorithm for column")),
                            status,
                        );
                    }

                    continue; /* jump to end of loop, to go to next column */
                } /* end of VLA case */

                /* ================================================================================*/
                /* deal with all the normal fixed-length columns here */
                /* ================================================================================*/

                /* allocate memory for the compressed data */
                datasize = (cm_colstart[ii + 1] - cm_colstart[ii]) as usize;
                if cvlamem.try_reserve_exact(datasize * 2).is_err() {
                    ffpmsg_str("Could not allocate buffer for compressed data");
                    *status = MEMORY_ALLOCATION;
                    return *status;
                } else {
                    cvlamem.resize(datasize * 2, 0);
                }

                tot_uncompressed_size += datasize as f32;

                if compalgor[ii] == RICE_1 {
                    /* a slice, not the Vec -- see the branch above */
                    let mut rce = RCEncoder::new(cvlamem.as_mut_slice());
                    rce.set_log_fn(ffpmsg_str);

                    if coltype[ii] == TSHORT {
                        if BYTESWAPPED {
                            ffswap2(
                                cast_slice_mut(&mut cm_buffer[cm_colstart[ii] as usize..]),
                                (datasize / 2).try_into().unwrap(),
                            );
                        }

                        /* datasize/2, not vlalen -- that belongs to the
                        variable-length branch above */
                        let r = rce.encode_short(
                            cast_slice(&cm_buffer[cm_colstart[ii] as usize..]),
                            datasize / 2,
                            32,
                        );
                        match r {
                            Ok(x) => dlen = x,
                            Err(_e) => {
                                *status = DATA_COMPRESSION_ERR;
                                // return *status;
                            }
                        }
                    } else if coltype[ii] == TLONG {
                        if BYTESWAPPED {
                            ffswap4(
                                cast_slice_mut(&mut cm_buffer[cm_colstart[ii] as usize..]),
                                (datasize / 4).try_into().unwrap(),
                            );
                        }

                        let r = rce.encode(
                            cast_slice(&cm_buffer[cm_colstart[ii] as usize..]),
                            datasize / 4,
                            32,
                        );
                        match r {
                            Ok(x) => dlen = x,
                            Err(_e) => {
                                *status = DATA_COMPRESSION_ERR;
                                // return *status;
                            }
                        }
                    } else if coltype[ii] == TBYTE {
                        let r = rce.encode_byte(
                            cast_slice(&cm_buffer[cm_colstart[ii] as usize..]),
                            datasize,
                            32,
                        );
                        match r {
                            Ok(x) => dlen = x,
                            Err(_e) => {
                                *status = DATA_COMPRESSION_ERR;
                                // return *status;
                            }
                        }
                    } else {
                        /* this should not happen */
                        ffpmsg_cstr(
                            c" Error: cannot compress this column type with the RICE algorthm",
                        );
                        *status = DATA_COMPRESSION_ERR;
                        return *status;
                    }
                } else {
                    /* all other cases: gzip compress the column (bytes may have been shuffled previously) */
                    compress2mem_from_mem(
                        cast_slice(&cm_buffer[cm_colstart[ii] as usize..]),
                        datasize,
                        &mut cvlamem,
                        Some(&mut dlen),
                        status,
                    );
                }

                if ll == 0 {
                    /* only write the ZCTYPn keyword once, while
                    processing the first column */
                    fits_make_keyn(cs!(c"ZCTYP"), (ii + 1) as c_int, &mut keyname, status);

                    if compalgor[ii] == RICE_1 {
                        strcpy_safe(&mut keyvalue, cs!(c"RICE_1"));
                    } else if compalgor[ii] == GZIP_2 {
                        strcpy_safe(&mut keyvalue, cs!(c"GZIP_2"));
                    } else {
                        strcpy_safe(&mut keyvalue, cs!(c"GZIP_1"));
                    }

                    fits_write_key(
                        outfptr,
                        crate::KeywordDatatype::TSTRING(&keyvalue),
                        &keyname,
                        Some(cs!(c"compression algorithm for column")),
                        status,
                    );
                }

                /* write the compressed data to the output column */
                fits_set_tscale(outfptr, (ii + 1).try_into().unwrap(), 1.0, 0.0, status); /* turn off any data scaling, first */
                fits_write_col(
                    outfptr,
                    TBYTE,
                    (ii + 1).try_into().unwrap(),
                    (ll + 1).try_into().unwrap(),
                    1,
                    dlen.try_into().unwrap(),
                    cast_slice(&cvlamem),
                    status,
                );
                tot_compressed_size += dlen as f32;

                /* create diagnostic messages */
                if dlen != 0 {
                    cratio[ii] = (datasize as f32) / (dlen as f32); /* compression ratio of the column */
                }

                int_snprintf!(tempstring, FLEN_VALUE, " r={:6.2}", cratio[ii]);
                let str_len = strlen_safe(&results[ii]);
                strncat_safe(&mut results[ii], &tempstring, 29 - str_len);
            } /* end of not a virtual column */
        } /* end of loop over columns */

        datastart += rowspertile as LONGLONG * naxis1; /* increment to start of next chunk */
        // firstrow += rowspertile as LONGLONG; /* increment first row in next chunk */
        if print_report {
            println!("\nChunk = {}", ll + 1);
            for ii in 0..(ncols as usize) {
                let results_str = CStr::from_bytes_until_nul(cast_slice(&results[ii]))
                    .unwrap()
                    .to_str()
                    .unwrap();
                println!("{results_str}\n");
            }
        }
    } /* end of loop over chunks of the table */

    /* =================================================================================*/
    /*  all done; just clean up and return  */
    /* ================================================================================*/

    fits_set_hdustruc(outfptr, status); /* reset internal structures */

    if print_report && tot_compressed_size != 0.0 {
        println!(
            "\nTotal data size (MB) {:.3} -> {:.3}, ratio = {:.3}",
            tot_uncompressed_size / 1000000.,
            tot_compressed_size / 1000000.,
            tot_uncompressed_size / tot_compressed_size,
        );
    }
    *status
}

/// Uncompress the table that was compressed with fits_compress_table
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_uncompress_table(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        fits_uncompress_table_safe(infptr, outfptr, status)
    }
}

/// Uncompress the table that was compressed with fits_compress_table
pub fn fits_uncompress_table_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    status: &mut c_int,
) -> c_int {
    let mut colcode: [c_schar; 999] = [0; 999]; // column data type code character
    let mut coltype: [c_schar; 999] = [0; 999]; // column data type numeric code value
    let mut cm_buffer: Vec<c_char> = Vec::new(); // memory buffer for the transposed, Column-Major, chunk of the table
    let mut rm_buffer: Vec<c_char> = Vec::new(); // memory buffer for the original, Row-Major, chunk of the table
    let mut nrows: i64 = 0;
    let mut rmajor_colwidth: [LONGLONG; 999] = [0; 999];
    let mut rmajor_colstart: [LONGLONG; 1000] = [0; 1000];
    let mut cmajor_colstart: [LONGLONG; 1000] = [0; 1000];
    let mut cmajor_repeat: [LONGLONG; 999] = [0; 999];
    let mut rmajor_repeat: [LONGLONG; 999] = [0; 999];
    let mut cmajor_bytespan: [LONGLONG; 999] = [0; 999];
    let mut headstart: LONGLONG = 0;
    let mut datastart: LONGLONG = 0;
    let mut dataend: LONGLONG = 0;
    let mut rowsremain: LONGLONG;
    // let mut rowstart: LONGLONG;
    let mut cvlalen: LONGLONG;
    let mut cvlastart: LONGLONG;
    let mut vlalen: LONGLONG;
    let mut vlastart: LONGLONG;
    let mut repeat: c_long = 0;
    let mut width: c_long = 0;
    let mut vla_repeat: c_long = 0;
    let mut vla_address: c_long = 0;
    let mut rowspertile: c_long = 0;
    let mut ntile: c_long;
    let mut ncols: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut inttype: c_int = 0;
    let mut anynull: c_int = 0;
    let mut tstatus: c_int = 0;
    let mut zctype: [c_int; 999] = [0; 999];
    let mut addspace: c_int = 0;
    let mut keyname: [c_char; 9] = [0; 9];
    let mut tform: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut pcount: c_long = 0;
    let mut zheapptr: c_long = 0;
    let mut naxis1: c_long = 0;
    let mut naxis2: c_long = 0;
    let mut ptr: Vec<c_char> = Vec::new();
    let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut zvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    let mut uncompressed_vla: Vec<u8> = Vec::new();
    let mut compressed_vla: Vec<u8> = Vec::new();

    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut dlen: usize = 0;
    let mut fullsize: usize;

    let mut bytepos: usize;
    let mut vlamemlen: usize;

    unsafe {
        /* ==================================================================================
         */
        /* perform initial sanity checks */
        /* ==================================================================================
         */
        if *status > 0 {
            return *status;
        }

        fits_get_hdu_type(infptr, &mut hdutype, status);
        if hdutype != BINARY_TBL {
            ffpmsg_str("This is not a binary table, so cannot uncompress it!");
            *status = NOT_BTABLE;
            return *status;
        }

        if fits_read_key_log(infptr, cs!(c"ZTABLE"), &mut tstatus, None, status) != 0 {
            /* just copy the HDU if the table is not compressed */
            if !core::ptr::eq(infptr, outfptr) {
                fits_copy_hdu(infptr, outfptr, 0, status);
            }
            return *status;
        }

        fits_get_num_rowsll(infptr, &mut nrows, status);
        fits_get_num_cols(infptr, &mut ncols, status);

        if ncols < 1 {
            /* just copy the HDU if the table does not have  more than 0 columns */
            if !core::ptr::eq(infptr, outfptr) {
                fits_copy_hdu(infptr, outfptr, 0, status);
            }
            return *status;
        }

        fits_read_key_lng(
            infptr,
            cs!(c"ZTILELEN"),
            &mut rowspertile,
            Some(&mut comm),
            status,
        );
        if *status > 0 {
            ffpmsg_str("Could not find the required ZTILELEN keyword");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        fits_read_key_lng(
            infptr,
            cs!(c"ZNAXIS1"),
            &mut naxis1,
            Some(&mut comm),
            status,
        );
        if *status > 0 {
            ffpmsg_str("Could not find the required ZNAXIS1 keyword");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        fits_read_key_lng(
            infptr,
            cs!(c"ZNAXIS2"),
            &mut naxis2,
            Some(&mut comm),
            status,
        );
        if *status > 0 {
            ffpmsg_str("Could not find the required ZNAXIS2 keyword");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        /* silently ignore illegal ZTILELEN value if too large */
        if rowspertile > naxis2 {
            rowspertile = naxis2;
        }

        fits_read_key_lng(
            infptr,
            cs!(c"ZPCOUNT"),
            &mut pcount,
            Some(&mut comm),
            status,
        );
        if *status > 0 {
            ffpmsg_str("Could not find the required ZPCOUNT keyword");
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        tstatus = 0;
        fits_read_key_lng(
            infptr,
            cs!(c"ZHEAPPTR"),
            &mut zheapptr,
            Some(&mut comm),
            &mut tstatus,
        );
        if tstatus > 0 {
            zheapptr = 0; /* uncompressed table has no heap */
        }

        /* ==================================================================================
         */
        /* copy of the input header, then recreate the uncompressed table keywords
         */
        /* ==================================================================================
         */
        fits_copy_header(infptr, outfptr, status);

        /* reset the NAXIS1, NAXIS2. and PCOUNT keywords to the original */
        /* Seven characters including the trailing space: it overwrites
        "ZNAXIS1" in place.  A shorter literal makes strncpy NUL-pad to n,
        truncating the card to a valueless keyword. */
        fits_read_card(outfptr, cs!(c"ZNAXIS1"), &mut card, status);
        strncpy_safe(&mut card, cs!(c"NAXIS1 "), 7);
        fits_update_card(outfptr, cs!(c"NAXIS1"), &card, status);

        fits_read_card(outfptr, cs!(c"ZNAXIS2"), &mut card, status);
        strncpy_safe(&mut card, cs!(c"NAXIS2 "), 7);
        fits_update_card(outfptr, cs!(c"NAXIS2"), &card, status);

        fits_read_card(outfptr, cs!(c"ZPCOUNT"), &mut card, status);
        strncpy_safe(&mut card, cs!(c"PCOUNT "), 7);
        fits_update_card(outfptr, cs!(c"PCOUNT"), &card, status);

        fits_delete_key(outfptr, cs!(c"ZTABLE"), status);
        fits_delete_key(outfptr, cs!(c"ZTILELEN"), status);
        fits_delete_key(outfptr, cs!(c"ZNAXIS1"), status);
        fits_delete_key(outfptr, cs!(c"ZNAXIS2"), status);
        fits_delete_key(outfptr, cs!(c"ZPCOUNT"), status);
        tstatus = 0;
        fits_delete_key(outfptr, cs!(c"CHECKSUM"), &mut tstatus);
        tstatus = 0;
        fits_delete_key(outfptr, cs!(c"DATASUM"), &mut tstatus);
        /* restore the Checksum keywords, if they exist */
        tstatus = 0;
        fits_modify_name(outfptr, cs!(c"ZHECKSUM"), cs!(c"CHECKSUM"), &mut tstatus);
        tstatus = 0;
        fits_modify_name(outfptr, cs!(c"ZDATASUM"), cs!(c"DATASUM"), &mut tstatus);

        /* ==================================================================================
         */
        /* determine compression paramters for each column and write column-specific keywords */
        /* ==================================================================================
         */
        for ii in 0..(ncols as usize) {
            /* get the original column type, repeat count, and unit width */
            fits_make_keyn(
                cs!(c"ZFORM"),
                (ii + 1).try_into().unwrap(),
                &mut keyname,
                status,
            );
            fits_read_key(
                infptr,
                crate::KeywordDatatypeMut::TSTRING(&mut tform),
                &keyname,
                Some(&mut comm),
                status,
            );

            /* restore the original TFORM value and comment */
            fits_read_card(outfptr, &keyname, &mut card, status);
            card[0] = bb(b'T');
            keyname[0] = bb(b'T');
            fits_update_card(outfptr, &keyname, &card, status);

            /* now delete the ZFORM keyword */
            keyname[0] = bb(b'Z');
            fits_delete_key(outfptr, &keyname, status);

            let mut cptr = 0; // tform
            while isdigit_safe(tform[cptr]) {
                cptr += 1;
            }
            colcode[ii] = tform[cptr] as c_schar; /* save the column type code */

            fits_binary_tform(
                &tform,
                Some(&mut inttype),
                Some(&mut repeat),
                Some(&mut width),
                status,
            );
            coltype[ii] = inttype as c_schar;

            /* deal with special cases */
            if i32::from((coltype[ii]).abs()) == TBIT {
                repeat = (repeat + 7) / 8; /* convert from bits to bytes */
            } else if i32::from((coltype[ii]).abs()) == TSTRING {
                width = 1;
            } else if coltype[ii] < 0 {
                /* pointer to variable length array */
                if colcode[ii] == (b'P') as c_schar {
                    width = 8; /* this is a 'P' column */
                } else {
                    width = 16; /* this is a 'Q' not a 'P' column */
                }

                addspace += 16; /* need space for a second set of Q pointers for
                this column */
            }

            rmajor_repeat[ii] = repeat as LONGLONG;

            /* width (in bytes) of each field in the row-major table */
            rmajor_colwidth[ii] = rmajor_repeat[ii] * width as LONGLONG;

            /* construct the ZCTYPn keyword name then read the keyword */
            fits_make_keyn(cs!(c"ZCTYP"), (ii + 1) as c_int, &mut keyname, status);
            tstatus = 0;
            fits_read_key(
                infptr,
                crate::KeywordDatatypeMut::TSTRING(&mut zvalue),
                &keyname,
                None,
                &mut tstatus,
            );
            if tstatus != 0 {
                zctype[ii] = GZIP_2;
            } else {
                if strcmp_safe(&zvalue, cs!(c"GZIP_2")) == 0 {
                    zctype[ii] = GZIP_2;
                } else if strcmp_safe(&zvalue, cs!(c"GZIP_1")) == 0 {
                    zctype[ii] = GZIP_1;
                } else if strcmp_safe(&zvalue, cs!(c"RICE_1")) == 0 {
                    zctype[ii] = RICE_1;
                } else {
                    ffpmsg_str("Unrecognized ZCTYPn keyword compression code:");
                    ffpmsg_slice(&zvalue);
                    *status = DATA_DECOMPRESSION_ERR;
                    return *status;
                }

                /* delete this keyword from the uncompressed header */
                fits_delete_key(outfptr, &keyname, status);
            }
        }

        /* rescan header keywords to reset internal table structure parameters */
        fits_set_hdustruc(outfptr, status);

        /* ==================================================================================
         */
        /* allocate memory for the transposed and untransposed tile of the table */
        /* ==================================================================================
         */

        fullsize = (naxis1 * rowspertile) as usize;
        let cm_size: usize = fullsize + (c_long::from(addspace) * rowspertile) as usize;

        if cm_buffer.try_reserve_exact(cm_size).is_err() {
            ffpmsg_str("Could not allocate buffer for transformed column-major table");
            *status = MEMORY_ALLOCATION;
            return *status;
        } else {
            cm_buffer.resize(cm_size, 0);
        }

        if rm_buffer.try_reserve_exact(fullsize).is_err() {
            ffpmsg_str("Could not allocate buffer for untransformed row-major table");
            *status = MEMORY_ALLOCATION;
            return *status;
        } else {
            rm_buffer.resize(fullsize, 0);
        }

        /* ==================================================================================
         */
        /* Main loop over all the tiles */
        /* ==================================================================================
         */

        rowsremain = naxis2 as LONGLONG;
        // rowstart = 1;
        ntile = 0;

        while rowsremain > 0 {
            /* ================================================================================== */
            /* loop over each column: read and uncompress the bytes */
            /* ================================================================================== */
            ntile += 1;
            rmajor_colstart[0] = 0;
            cmajor_colstart[0] = 0;
            for ii in 0..(ncols as usize) {
                cmajor_repeat[ii] = rmajor_repeat[ii] * rowspertile as LONGLONG;

                /* starting offset of each field in the column-major table */
                if coltype[ii] > 0 {
                    /* normal fixed length column */
                    cmajor_colstart[ii + 1] =
                        cmajor_colstart[ii] + rmajor_colwidth[ii] * rowspertile as LONGLONG;
                } else {
                    /* VLA column: reserve space for the 2nd set of Q pointers */
                    cmajor_colstart[ii + 1] =
                        cmajor_colstart[ii] + (rmajor_colwidth[ii] + 16) * rowspertile as LONGLONG;
                }
                /* length of each sequence of bytes, after sorting them in signicant order */
                cmajor_bytespan[ii] = rmajor_repeat[ii] * rowspertile as LONGLONG;

                /* starting offset of each field in the  row-major table */
                rmajor_colstart[ii + 1] = rmajor_colstart[ii] + rmajor_colwidth[ii];

                if rmajor_repeat[ii] > 0 {
                    /* ignore columns with 0 elements */

                    /* read compressed bytes from input table */
                    fits_read_descript(
                        infptr,
                        (ii + 1) as c_int,
                        ntile as LONGLONG,
                        Some(&mut vla_repeat),
                        Some(&mut vla_address),
                        status,
                    );

                    /* allocate memory and read in the compressed bytes */
                    if ptr.try_reserve_exact(vla_repeat as usize).is_err() {
                        ffpmsg_str("Could not allocate buffer for uncompressed bytes");
                        *status = MEMORY_ALLOCATION;
                        return *status;
                    } else {
                        ptr.resize(vla_repeat as usize, 0);
                    }

                    fits_set_tscale(infptr, (ii + 1) as c_int, 1.0, 0.0, status); /* turn off any data scaling, first */
                    fits_read_col_byt(
                        infptr,
                        (ii + 1).try_into().unwrap(),
                        ntile as LONGLONG,
                        1,
                        vla_repeat as LONGLONG,
                        0,
                        cast_slice_mut(&mut ptr),
                        Some(&mut anynull),
                        status,
                    );

                    /* size in bytes of the uncompressed column of bytes */
                    fullsize = (cmajor_colstart[ii + 1] - cmajor_colstart[ii]) as usize;

                    /* cm_buffer is sized as the C sizes it, which does not
                    cover a variable-length string payload -- the
                    under-allocation behind heasarc/cfitsio#134.  The C grows
                    past the end of it; refuse instead. */
                    if (cmajor_colstart[ii] as usize) + fullsize > cm_buffer.len() {
                        ffpmsg_str("Compressed table column does not fit the transposed buffer");
                        ffpmsg_str(
                            " (known limitation: variable-length string columns, cfitsio#134)",
                        );
                        *status = DATA_DECOMPRESSION_ERR;
                        return *status;
                    }

                    /* cm_buffer indexed by the column offset, not a slice of
                    the offset table itself */
                    let cptr = &mut cm_buffer[cmajor_colstart[ii] as usize..];

                    match colcode[ii] as u8 {
                        b'I' => {
                            if zctype[ii] == RICE_1 {
                                let mut rcd = RCDecoder::new();
                                rcd.set_log_fn(ffpmsg_str);
                                /* the decoder asserts its output slice is
                                exactly nx long */
                                match rcd.decode_short(
                                    cast_slice(&ptr),
                                    fullsize / 2_usize,
                                    32,
                                    cast_slice_mut(&mut cptr[..fullsize]),
                                ) {
                                    Ok(()) => dlen = fullsize,
                                    Err(_e) => {
                                        *status = DATA_DECOMPRESSION_ERR;
                                        // return *status;
                                    }
                                }

                                if BYTESWAPPED {
                                    ffswap2(
                                        cast_slice_mut(cptr),
                                        (fullsize / 2).try_into().unwrap(),
                                    );
                                }
                            } else {
                                /* gunzip the data into the correct location */
                                uncompress2mem_from_mem(
                                    &ptr,
                                    vla_repeat.try_into().unwrap(),
                                    &mut cptr.as_mut_ptr().cast::<u8>(),
                                    &mut fullsize,
                                    /* no grow callback: cptr is an interior
                                    pointer of cm_buffer, and the C's realloc of
                                    one is heasarc/cfitsio#134 */
                                    None,
                                    Some(&mut dlen),
                                    status,
                                );
                            }
                        }

                        b'J' => {
                            if zctype[ii] == RICE_1 {
                                let mut rcd = RCDecoder::new();
                                rcd.set_log_fn(ffpmsg_str);
                                match rcd.decode(
                                    cast_slice(&ptr),
                                    fullsize / 4_usize,
                                    32,
                                    cast_slice_mut(&mut cptr[..fullsize]),
                                ) {
                                    Ok(()) => dlen = fullsize,
                                    Err(_e) => {
                                        *status = DATA_DECOMPRESSION_ERR;
                                        // return *status;
                                    }
                                }

                                if BYTESWAPPED {
                                    ffswap4(
                                        cast_slice_mut(cptr),
                                        (fullsize / 4).try_into().unwrap(),
                                    );
                                }
                            } else {
                                /* gunzip the data into the correct location */
                                uncompress2mem_from_mem(
                                    &ptr,
                                    vla_repeat.try_into().unwrap(),
                                    &mut cptr.as_mut_ptr().cast::<u8>(),
                                    &mut fullsize,
                                    /* no grow callback: cptr is an interior
                                    pointer of cm_buffer, and the C's realloc of
                                    one is heasarc/cfitsio#134 */
                                    None,
                                    Some(&mut dlen),
                                    status,
                                );
                            }
                        }

                        b'B' => {
                            if zctype[ii] == RICE_1 {
                                let mut rcd = RCDecoder::new();
                                rcd.set_log_fn(ffpmsg_str);
                                match rcd.decode_byte(
                                    cast_slice(&ptr),
                                    fullsize,
                                    32,
                                    cast_slice_mut(&mut cptr[..fullsize]),
                                ) {
                                    Ok(()) => dlen = fullsize,
                                    Err(_e) => {
                                        *status = DATA_DECOMPRESSION_ERR;
                                        // return *status;
                                    }
                                }
                            } else {
                                /* gunzip the data into the correct location */
                                uncompress2mem_from_mem(
                                    &ptr,
                                    vla_repeat.try_into().unwrap(),
                                    &mut cptr.as_mut_ptr().cast::<u8>(),
                                    &mut fullsize,
                                    /* no grow callback: cptr is an interior
                                    pointer of cm_buffer, and the C's realloc of
                                    one is heasarc/cfitsio#134 */
                                    None,
                                    Some(&mut dlen),
                                    status,
                                );
                            }
                        }

                        _ => {
                            /* all variable length array columns are included in this case */
                            /* gunzip the data into the correct location in the full table buffer */
                            uncompress2mem_from_mem(
                                &ptr,
                                vla_repeat.try_into().unwrap(),
                                &mut cptr.as_mut_ptr().cast::<u8>(),
                                &mut fullsize,
                                /* no interior-pointer realloc; see above */
                                None,
                                Some(&mut dlen),
                                status,
                            );
                        }
                    } /* end of switch block */
                } /* end of rmajor_repeat > 0 */
            } /* end of loop over columns */

            /* now transpose the rows and columns (from cm_buffer to rm_buffer) */
            /* move each byte, in turn, from the cm_buffer to the appropriate place in the rm_buffer */
            for ii in 0..(ncols as usize) {
                /* loop over columns */
                let ptr = &cm_buffer[cmajor_colstart[ii] as usize..]; /* initialize ptr to start of the column in the cm_buffer */
                let mut ptr_idx = 0;

                if rmajor_repeat[ii] > 0 {
                    /* skip columns with zero elements */
                    if coltype[ii] > 0 {
                        /* normal fixed length array columns */
                        if zctype[ii] == GZIP_2 {
                            /*  need to unshuffle the bytes */

                            /* recombine the byte planes for the 2-byte, 4-byte, and 8-byte numeric columns */
                            match colcode[ii] as u8 {
                                b'I' => {
                                    /* get the 1st byte of each I*2 value */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize))..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 2;
                                        }
                                    }

                                    /* get the 2nd byte of each I*2 value */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 1)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 2;
                                        }
                                    }
                                }

                                b'J' | b'E' => {
                                    /* get the 1st byte of each 4-byte value */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize))..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 4;
                                        }
                                    }

                                    /* get the 2nd byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 1)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 4;
                                        }
                                    }

                                    /* get the 3rd byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 2)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 4;
                                        }
                                    }
                                    /* get the 4th byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 3)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 4;
                                        }
                                    }
                                }

                                b'D' | b'K' => {
                                    /* get the 1st byte of each 8-byte value */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize))..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 8;
                                        }
                                    }

                                    /* get the 2nd byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 1)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 8;
                                        }
                                    }

                                    /* get the 3rd byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 2)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 8;
                                        }
                                    }

                                    /* get the 4th byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 3)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 8;
                                        }
                                    }

                                    /* get the 5th byte */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 4)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 8;
                                        }
                                    }

                                    /* get the 6th byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 5)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 8;
                                        }
                                    }

                                    /* get the 7th byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 6)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 8;
                                        }
                                    }

                                    /* get the 8th byte  */
                                    for jj in 0..(rowspertile as usize) {
                                        /* loop over number of rows in the output table */
                                        let cptr = &mut rm_buffer[(rmajor_colstart[ii] as usize
                                            + (jj * rmajor_colstart[ncols as usize] as usize)
                                            + 7)..];
                                        let mut cptr_idx = 0;
                                        for _kk in 0..(rmajor_repeat[ii] as usize) {
                                            cptr[cptr_idx] = ptr[ptr_idx]; /* copy 1 byte */
                                            ptr_idx += 1;
                                            cptr_idx += 8;
                                        }
                                    }
                                }

                                _ => {
                                    /*  should never get here */
                                    ffpmsg_str(
                                        "Error: unexpected attempt to use GZIP_2 to compress a column unsuitable data type",
                                    );
                                    *status = DATA_DECOMPRESSION_ERR;
                                    return *status;
                                }
                            } /* end of switch  for shuffling the bytes*/
                        } else {
                            /* not GZIP_2, don't have to shuffle bytes, so just transpose the rows and columns */

                            for jj in 0..(rowspertile as usize) {
                                /* loop over number of rows in the output table */
                                let cptr = &mut rm_buffer[((rmajor_colstart[ii] as usize)
                                    + jj * (rmajor_colstart[ncols as usize] as usize))..]; /* addr to copy to */
                                let cptr_idx = 0;

                                cptr[cptr_idx..(cptr_idx + rmajor_colwidth[ii] as usize)]
                                    .copy_from_slice(
                                        &ptr[ptr_idx..(ptr_idx + rmajor_colwidth[ii] as usize)],
                                    ); /* copy the bytes */

                                ptr_idx += rmajor_colwidth[ii] as usize;
                            }
                        }
                    } else {
                        /* transpose the variable length array pointers */

                        for jj in 0..(rowspertile as usize) {
                            /* loop over number of rows in the output uncompressed table */
                            let cptr = &mut rm_buffer[((rmajor_colstart[ii] as usize)
                                + jj * (rmajor_colstart[ncols as usize] as usize))..]; /* addr to copy to */
                            let cptr_idx = 0;

                            cptr[cptr_idx..(cptr_idx + rmajor_colwidth[ii] as usize)]
                                .copy_from_slice(
                                    &ptr[ptr_idx..(ptr_idx + rmajor_colwidth[ii] as usize)],
                                ); /* copy the bytes */

                            ptr_idx += rmajor_colwidth[ii] as usize;
                        }

                        if rmajor_colwidth[ii] == 8 {
                            /* these are P-type descriptors */
                            let pdescript: &mut [c_int] =
                                cast_slice_mut(&mut cm_buffer[cmajor_colstart[ii] as usize..]);
                            if BYTESWAPPED {
                                ffswap4(cast_slice_mut(pdescript), rowspertile * 2); /* byte-swap the descriptor */
                            }
                        } else if rmajor_colwidth[ii] == 16 {
                            /* these are Q-type descriptors */
                            let qdescript: &mut [LONGLONG] =
                                cast_slice_mut(&mut cm_buffer[cmajor_colstart[ii] as usize..]);
                            if BYTESWAPPED {
                                ffswap8(cast_slice_mut(qdescript), rowspertile * 2); /* byte-swap the descriptor */
                            }
                        } else {
                            /* this should never happen */
                            ffpmsg_str("Error: Descriptor column is neither 8 nor 16 bytes wide");
                            *status = DATA_DECOMPRESSION_ERR;
                            return *status;
                        }

                        /* First, set pointer to the Q descriptors, and byte-swap them, if needed */
                        let descript: &mut [LONGLONG] = cast_slice_mut(
                            &mut cm_buffer[(cmajor_colstart[ii]
                                + (rmajor_colwidth[ii] * rowspertile as LONGLONG))
                                as usize..],
                        );
                        if BYTESWAPPED {
                            /* byte-swap the descriptor */
                            ffswap8(cast_slice_mut(descript), rowspertile * 2);
                        }

                        /* now uncompress all the individual VLAs, and */
                        /* write them to their original location in the uncompressed file */

                        let pdescript: &[c_int] =
                            cast_slice(&cm_buffer[cmajor_colstart[ii] as usize..]);
                        let qdescript: &[LONGLONG] =
                            cast_slice(&cm_buffer[cmajor_colstart[ii] as usize..]);
                        let descript: &[LONGLONG] = cast_slice(
                            &cm_buffer[(cmajor_colstart[ii]
                                + (rmajor_colwidth[ii] * rowspertile as LONGLONG))
                                as usize..],
                        );

                        for jj in 0..(rowspertile as usize) {
                            /* loop over rows */
                            /* get the size and location of the compressed VLA in the compressed table */
                            cvlalen = descript[jj * 2];
                            cvlastart = descript[(jj * 2) + 1];
                            if cvlalen > 0 {
                                /* get the size and location to write the uncompressed VLA in the uncompressed table */
                                if rmajor_colwidth[ii] == 8 {
                                    vlalen = LONGLONG::from(pdescript[jj * 2]);
                                    vlastart = LONGLONG::from(pdescript[(jj * 2) + 1]);
                                } else {
                                    vlalen = qdescript[jj * 2];
                                    vlastart = qdescript[(jj * 2) + 1];
                                }
                                vlamemlen = vlalen as usize * (-coltype[ii] / 10) as usize; /* size of the uncompressed VLA, in bytes */

                                /* allocate memory for the compressed vla */
                                if compressed_vla.try_reserve_exact(cvlalen as usize).is_err() {
                                    ffpmsg_str("Could not allocate buffer for compressed VLA");
                                    *status = MEMORY_ALLOCATION;
                                    return *status;
                                } else {
                                    compressed_vla.resize(cvlalen as usize, 0);
                                }

                                /* read the compressed VLA from the heap in the input compressed table */
                                bytepos = ((infptr.Fptr).datastart
                                    + (infptr.Fptr).heapstart
                                    + cvlastart) as usize;
                                ffmbyt_safe(infptr, bytepos as LONGLONG, REPORT_EOF, status);
                                ffgbyt(infptr, cvlalen, &mut compressed_vla, status); /* read the bytes */
                                /* if the VLA couldn't be compressed, just copy it directly to the output uncompressed table */
                                if cvlalen == vlamemlen.try_into().unwrap() {
                                    bytepos = ((outfptr.Fptr).datastart
                                        + (outfptr.Fptr).heapstart
                                        + vlastart)
                                        as usize;
                                    ffmbyt_safe(outfptr, bytepos as LONGLONG, IGNORE_EOF, status);
                                    ffpbyt(outfptr, cvlalen, &compressed_vla, status);
                                /* write the bytes */
                                } else {
                                    /* uncompress the VLA  */

                                    /* allocate memory for the uncompressed VLA */
                                    if uncompressed_vla.try_reserve_exact(vlamemlen).is_err() {
                                        ffpmsg_str(
                                            "Could not allocate buffer for uncompressed VLA",
                                        );
                                        *status = MEMORY_ALLOCATION;
                                        return *status;
                                    } else {
                                        uncompressed_vla.resize(vlamemlen, 0);
                                    }

                                    /* uncompress the VLA with the appropriate algorithm */
                                    if zctype[ii] == RICE_1 {
                                        let rcd = RCDecoder::new();

                                        if i32::from(-coltype[ii]) == TSHORT {
                                            let res = rcd.decode_short(
                                                cast_slice(&compressed_vla),
                                                vlalen as usize,
                                                32,
                                                cast_slice_mut(&mut uncompressed_vla),
                                            );

                                            if let Err(_e) = res {
                                                *status = DATA_DECOMPRESSION_ERR;
                                                return *status;
                                            }

                                            if BYTESWAPPED {
                                                ffswap2(
                                                    cast_slice_mut(&mut uncompressed_vla),
                                                    vlalen as c_long,
                                                );
                                            }
                                        } else if i32::from(-coltype[ii]) == TLONG {
                                            let res = rcd.decode(
                                                cast_slice(&compressed_vla),
                                                vlalen as usize,
                                                32,
                                                cast_slice_mut(&mut uncompressed_vla),
                                            );

                                            if let Err(_e) = res {
                                                *status = DATA_DECOMPRESSION_ERR;
                                                return *status;
                                            }

                                            if BYTESWAPPED {
                                                ffswap4(
                                                    cast_slice_mut(&mut uncompressed_vla),
                                                    vlalen as c_long,
                                                );
                                            }
                                        } else if i32::from(-coltype[ii]) == TBYTE {
                                            let res = rcd.decode_byte(
                                                cast_slice(&compressed_vla),
                                                vlalen as usize,
                                                32,
                                                cast_slice_mut(&mut uncompressed_vla),
                                            );
                                            if let Err(_e) = res {
                                                *status = DATA_DECOMPRESSION_ERR;
                                                return *status;
                                            }
                                        } else {
                                            /* this should not happen */
                                            ffpmsg_str(
                                                " Error: cannot uncompress this column type with the RICE algorithm",
                                            );

                                            *status = DATA_DECOMPRESSION_ERR;
                                            return *status;
                                        }
                                    } else if zctype[ii] == GZIP_1 || zctype[ii] == GZIP_2 {
                                        /*: gzip uncompress the array of bytes */
                                        let mut filesize = vlamemlen;

                                        uncompress2mem_from_mem(
                                            cast_slice(&compressed_vla),
                                            cvlalen.try_into().unwrap(),
                                            &mut uncompressed_vla.as_mut_ptr(),
                                            &mut vlamemlen,
                                            /* Rust-owned: cannot be grown by realloc */
                                            None,
                                            Some(&mut filesize),
                                            status,
                                        );

                                        // TODO: Do we need to reassign filesize to vlamemlen?

                                        if zctype[ii] == GZIP_2 {
                                            /* unshuffle the bytes after ungzipping them */
                                            if c_int::from(-coltype[ii] / 10) == 2 {
                                                fits_unshuffle_2bytes(
                                                    cast_slice_mut(&mut uncompressed_vla),
                                                    vlalen,
                                                    status,
                                                );
                                            } else if c_int::from(-coltype[ii] / 10) == 4 {
                                                fits_unshuffle_4bytes(
                                                    cast_slice_mut(&mut uncompressed_vla),
                                                    vlalen,
                                                    status,
                                                );
                                            } else if c_int::from(-coltype[ii] / 10) == 8 {
                                                fits_unshuffle_8bytes(
                                                    cast_slice_mut(&mut uncompressed_vla),
                                                    vlalen,
                                                    status,
                                                );
                                            }
                                        }
                                    } else {
                                        /* this should not happen */
                                        ffpmsg_str(" Error: unknown compression algorithm");
                                        *status = DATA_COMPRESSION_ERR;
                                        return *status;
                                    }

                                    bytepos = ((outfptr.Fptr).datastart
                                        + (outfptr.Fptr).heapstart
                                        + vlastart)
                                        as usize;
                                    ffmbyt_safe(outfptr, bytepos as LONGLONG, IGNORE_EOF, status);
                                    ffpbyt(
                                        outfptr,
                                        vlamemlen.try_into().unwrap(),
                                        cast_slice(&uncompressed_vla),
                                        status,
                                    );
                                    /* write the bytes */
                                } /* end of uncompress VLA */
                            } /* end of vlalen > 0 */
                        } /* end of loop over rowspertile */
                    } /* end of variable length array section*/
                } /* end of if column repeat > 0 */
            } /* end of ncols loop */

            /* copy the buffer of data to the output data unit */

            if datastart == 0 {
                fits_get_hduaddrll(
                    outfptr,
                    Some(&mut headstart),
                    Some(&mut datastart),
                    Some(&mut dataend),
                    status,
                );
            }

            ffmbyt_safe(outfptr, datastart, 1, status);
            ffpbyt(
                outfptr,
                (naxis1 * rowspertile) as LONGLONG,
                cast_slice(&rm_buffer),
                status,
            );

            /* increment pointers for next tile */
            // rowstart += rowspertile as LONGLONG;
            rowsremain -= rowspertile as LONGLONG;
            datastart += (naxis1 * rowspertile) as LONGLONG;
            if rowspertile as LONGLONG > rowsremain {
                rowspertile = rowsremain as c_long;
            }
        } /* end of while rows still remain */

        /* reset internal table structure parameters */
        fits_set_hdustruc(outfptr, status);
        *status
    }
}
/// Shuffle the bytes in an array of 2-byte integers in the heap
fn fits_shuffle_2bytes(heap: &mut [c_char], length: LONGLONG, status: &mut c_int) -> c_int {
    let length = length as usize;
    let heap = cast_slice_mut(heap);

    let mut p: Vec<u8> = Vec::new();
    if p.try_reserve_exact(length * 2).is_err() {
        ffpmsg_str("malloc failed\n");
        return *status;
    } else {
        p.resize(length * 2, 0);
    }

    let mut heapptr: usize = 0;

    for ii in 0..length {
        p[ii] = heap[heapptr];
        heapptr += 1;

        p[ii + length] = heap[heapptr];
        heapptr += 1;
    }

    heap[0..(length * 2)].copy_from_slice(&p[..(length * 2)]);

    *status
}

/// Shuffle the bytes in an array of 4-byte integers or floats
fn fits_shuffle_4bytes(heap: &mut [c_char], length: LONGLONG, status: &mut c_int) -> c_int {
    let length = length as usize;
    let heap = cast_slice_mut(heap);

    let mut p: Vec<u8> = Vec::new();
    if p.try_reserve_exact(length * 4).is_err() {
        ffpmsg_str("malloc failed\n");
        return *status;
    } else {
        p.resize(length * 4, 0);
    }

    let mut heapptr: usize = 0;

    for ii in 0..length {
        p[ii] = heap[heapptr];
        heapptr += 1;

        p[ii + length] = heap[heapptr];
        heapptr += 1;

        p[ii + (length * 2)] = heap[heapptr];
        heapptr += 1;

        p[ii + (length * 3)] = heap[heapptr];
        heapptr += 1;
    }

    heap[0..(length * 4)].copy_from_slice(&p[..(length * 4)]);

    *status
}

/// Shuffle the bytes in an array of 8-byte integers or doubles in the heap
fn fits_shuffle_8bytes(heap: &mut [c_char], length: LONGLONG, status: &mut c_int) -> c_int {
    let length = length as usize;
    let heap = cast_slice_mut(heap);

    let mut p: Vec<u8> = Vec::new();
    if p.try_reserve_exact(length * 8).is_err() {
        ffpmsg_str("malloc failed\n");
        return *status;
    } else {
        p.resize(length * 8, 0);
    }

    let ptr: usize = 0; // index into p
    let mut heapptr: usize = 0;
    let mut cptr: usize = 0;

    /* for some bizarre reason this loop fails to compile under OpenSolaris
    using the proprietary SunStudioExpress C compiler;  use the following
    equivalent loop instead.

    cptr = ptr;

    for ii in 0..(length as usize) {
    *cptr = *heapptr;
    heapptr+=1;
    *(cptr + length) = *heapptr;
    heapptr+=1;
    *(cptr + (length * 2)) = *heapptr;
    heapptr+=1;
    *(cptr + (length * 3)) = *heapptr;
    heapptr+=1;
    *(cptr + (length * 4)) = *heapptr;
    heapptr+=1;
    *(cptr + (length * 5)) = *heapptr;
    heapptr+=1;
    *(cptr + (length * 6)) = *heapptr;
    heapptr+=1;
    *(cptr + (length * 7)) = *heapptr;
    heapptr+=1;
    cptr+=1;
    }
    */
    for ii in 0..length {
        cptr = ptr + ii;

        p[cptr] = heap[heapptr];

        heapptr += 1;
        cptr += length;
        p[cptr] = heap[heapptr];

        heapptr += 1;
        cptr += length;
        p[cptr] = heap[heapptr];

        heapptr += 1;
        cptr += length;
        p[cptr] = heap[heapptr];

        heapptr += 1;
        cptr += length;
        p[cptr] = heap[heapptr];

        heapptr += 1;
        cptr += length;
        p[cptr] = heap[heapptr];

        heapptr += 1;
        cptr += length;
        p[cptr] = heap[heapptr];

        heapptr += 1;
        cptr += length;
        p[cptr] = heap[heapptr];

        heapptr += 1;
    }

    heap[0..(length * 8)].copy_from_slice(&p[..(length * 8)]);

    *status
}

/// Undo the byte shuffling of an array of `length` values of `NBYTES` bytes each.
///
/// The shuffled heap holds `NBYTES` planes of `length` bytes: every value's first
/// byte, then every value's second byte, and so on.  Reassembling the values is
/// simply the transpose of that layout.
fn unshuffle_bytes<const NBYTES: usize>(heap: &mut [c_char], length: usize) {
    if length == 0 {
        return;
    }

    let heap: &mut [u8] = cast_slice_mut(&mut heap[..length * NBYTES]);

    let planes = heap.to_vec();
    for (byte_index, plane) in planes.chunks_exact(length).enumerate() {
        for (value, &byte) in heap.as_chunks_mut::<NBYTES>().0.iter_mut().zip(plane) {
            value[byte_index] = byte;
        }
    }
}

/// Unshuffle the bytes in an array of 2-byte integers
fn fits_unshuffle_2bytes(heap: &mut [c_char], length: LONGLONG, status: &mut c_int) -> c_int {
    unshuffle_bytes::<2>(heap, length as usize);

    *status
}

/// Unshuffle the bytes in an array of 4-byte integers or floats
fn fits_unshuffle_4bytes(heap: &mut [c_char], length: LONGLONG, status: &mut c_int) -> c_int {
    unshuffle_bytes::<4>(heap, length as usize);

    *status
}

/// Unshuffle the bytes in an array of 8-byte integers or doubles
fn fits_unshuffle_8bytes(heap: &mut [c_char], length: LONGLONG, status: &mut c_int) -> c_int {
    unshuffle_bytes::<8>(heap, length as usize);

    *status
}

/// Convert the input array of 32-bit integers into an array of 64-bit integers,
/// in place. This will overwrite the input array with the new longer array starting
/// at the same memory location.
///
/// Note that aliasing the same memory location with pointers of different datatypes
/// is not allowed in strict ANSI C99, however it is  used here for efficency. In
/// principle, one could simply copy the input array in reverse order to the output
/// array, but this only works if the compiler performs the operation in strict
/// order.  Certain compiler optimization techniques may vioate this assumption.
/// Therefore, we first copy a section of the input array to a temporary
/// intermediate array, before copying the longer datatype values back to the
/// original array.
fn fits_int_to_longlong_inplace(
    intarray: &mut [c_int],
    length: c_long,
    status: &mut c_int,
) -> c_int {
    let mut ntodo: c_long;
    let mut firstelem: usize;
    let nmax: c_long = 10000;

    if *status > 0 {
        return *status;
    }

    ntodo = nmax;
    if length < nmax {
        ntodo = length;
    }

    firstelem = (length - ntodo).try_into().unwrap(); /* first element to be converted */

    let mut longlongarray: Vec<LONGLONG> = Vec::new();

    if longlongarray
        .try_reserve_exact(ntodo.try_into().unwrap())
        .is_err()
    {
        ffpmsg_str("Out of memory. (fits_int_to_longlong_inplace)");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        longlongarray.resize(ntodo.try_into().unwrap(), 0);
    }

    while ntodo > 0 {
        /* do datatype conversion into temp array */
        for ii in 0..(ntodo as usize) {
            longlongarray[ii] = LONGLONG::from(intarray[ii + firstelem]);
        }

        /* copy temp array back to alias */
        let aliasarray: &mut [LONGLONG] = cast_slice_mut(intarray); /* alias pointer to the input array */
        aliasarray[firstelem..(firstelem + ntodo as usize)]
            .copy_from_slice(&longlongarray[..(ntodo as usize)]);

        if firstelem == 0 {
            /* we are all done */
            ntodo = 0;
        } else {
            /* recalculate ntodo and firstelem for next loop */
            if firstelem > nmax as usize {
                firstelem -= nmax as usize;
            } else {
                ntodo = firstelem as c_long;
                firstelem = 0;
            }
        }
    }

    *status
}

/// Convert the input array of 16-bit integers into an array of 32-bit integers,
/// in place. This will overwrite the input array with the new longer array starting
/// at the same memory location.
///
/// Note that aliasing the same memory location with pointers of different datatypes
/// is not allowed in strict ANSI C99, however it is  used here for efficency. In
/// principle, one could simply copy the input array in reverse order to the output
/// array, but this only works if the compiler performs the operation in strict
/// order.  Certain compiler optimization techniques may vioate this assumption.
/// Therefore, we first copy a section of the input array to a temporary
/// intermediate array, before copying the longer datatype values back to the
/// original array.
fn fits_short_to_int_inplace(
    shortarray: &mut [c_short],
    length: c_long,
    shift: c_int,
    status: &mut c_int,
) -> c_int {
    let mut ntodo: c_long;
    let mut firstelem: usize;
    let nmax: c_long = 10000;

    if *status > 0 {
        return *status;
    }

    ntodo = nmax;
    if length < nmax {
        ntodo = length;
    }

    firstelem = (length - ntodo).try_into().unwrap(); /* first element to be converted */

    let mut intarray: Vec<c_int> = Vec::new();

    if intarray
        .try_reserve_exact(ntodo.try_into().unwrap())
        .is_err()
    {
        ffpmsg_str("Out of memory. (fits_short_to_int_inplace)");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        intarray.resize(ntodo as usize, 0);
    }

    while ntodo > 0 {
        /* do datatype conversion into temp array */
        for ii in 0..(ntodo as usize) {
            intarray[ii] = c_int::from(shortarray[ii + firstelem]) + shift;
        }

        /* copy temp array back to alias */
        let aliasarray: &mut [c_int] = cast_slice_mut(shortarray); /* alias pointer to the input array */
        aliasarray[firstelem..(firstelem + ntodo as usize)]
            .copy_from_slice(&intarray[..(ntodo as usize)]);

        if firstelem == 0 {
            /* we are all done */
            ntodo = 0;
        } else {
            /* recalculate ntodo and firstelem for next loop */
            if firstelem > nmax as usize {
                firstelem -= nmax as usize;
            } else {
                ntodo = firstelem as c_long;
                firstelem = 0;
            }
        }
    }

    *status
}

/// Convert the input array of 16-bit unsigned integers into an array of 32-bit
/// integers, in place. This will overwrite the input array with the new longer
/// array starting at the same memory location.
///
/// Note that aliasing the same memory location with pointers of different datatypes
/// is not allowed in strict ANSI C99, however it is  used here for efficency. In
/// principle, one could simply copy the input array in reverse order to the output
/// array, but this only works if the compiler performs the operation in strict
/// order.  Certain compiler optimization techniques may vioate this assumption.
/// Therefore, we first copy a section of the input array to a temporary
/// intermediate array, before copying the longer datatype values back to the
/// original array.
fn fits_ushort_to_int_inplace(
    ushortarray: &mut [c_ushort],
    length: c_long,
    shift: c_int,
    status: &mut c_int,
) -> c_int {
    let mut ntodo: c_long;
    let mut firstelem: usize;
    let nmax: c_long = 10000;

    if *status > 0 {
        return *status;
    }

    ntodo = nmax;
    if length < nmax {
        ntodo = length;
    }

    firstelem = (length - ntodo).try_into().unwrap(); /* first element to be converted */

    let mut intarray: Vec<c_int> = Vec::new();

    if intarray
        .try_reserve_exact(ntodo.try_into().unwrap())
        .is_err()
    {
        ffpmsg_str("Out of memory. (fits_ushort_to_int_inplace)");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        intarray.resize(ntodo as usize, 0);
    }

    while ntodo > 0 {
        /* do datatype conversion into temp array */
        for ii in 0..(ntodo as usize) {
            intarray[ii] = c_int::from(ushortarray[ii + firstelem]) + shift;
        }

        /* copy temp array back to alias */
        let aliasarray: &mut [c_int] = cast_slice_mut(ushortarray); /* alias pointer to the input array */
        aliasarray[firstelem..(firstelem + ntodo as usize)]
            .copy_from_slice(&intarray[..(ntodo as usize)]);

        if firstelem == 0 {
            /* we are all done */
            ntodo = 0;
        } else {
            /* recalculate ntodo and firstelem for next loop */
            if firstelem > nmax as usize {
                firstelem -= nmax as usize;
            } else {
                ntodo = firstelem as c_long;
                firstelem = 0;
            }
        }
    }

    *status
}

/// Convert the input array of 8-bit unsigned integers into an array of 32-bit
/// integers, in place. This will overwrite the input array with the new longer
/// array starting at the same memory location.
///
/// Note that aliasing the same memory location with pointers of different datatypes
/// is not allowed in strict ANSI C99, however it is  used here for efficency. In
/// principle, one could simply copy the input array in reverse order to the output
/// array, but this only works if the compiler performs the operation in strict
/// order.  Certain compiler optimization techniques may vioate this assumption.
/// Therefore, we first copy a section of the input array to a temporary
/// intermediate array, before copying the longer datatype values back to the
/// original array.
fn fits_ubyte_to_int_inplace(
    ubytearray: &mut [c_uchar],
    length: c_long,
    status: &mut c_int,
) -> c_int {
    let mut ntodo: c_long;
    let mut firstelem: usize;
    let nmax: c_long = 10000;

    if *status > 0 {
        return *status;
    }

    ntodo = nmax;
    if length < nmax {
        ntodo = length;
    }

    firstelem = (length - ntodo).try_into().unwrap(); /* first element to be converted */

    let mut intarray: Vec<c_int> = Vec::new();

    if intarray
        .try_reserve_exact(ntodo.try_into().unwrap())
        .is_err()
    {
        ffpmsg_str("Out of memory. (fits_ubyte_to_int_inplace)");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        intarray.resize(ntodo as usize, 0);
    }

    while ntodo > 0 {
        /* do datatype conversion into temp array */
        for ii in 0..(ntodo as usize) {
            intarray[ii] = c_int::from(ubytearray[ii + firstelem]);
        }

        /* copy temp array back to alias */
        let aliasarray: &mut [c_int] = cast_slice_mut(ubytearray); /* alias pointer to the input array */
        aliasarray[firstelem..(firstelem + ntodo as usize)]
            .copy_from_slice(&intarray[..(ntodo as usize)]);

        if firstelem == 0 {
            /* we are all done */
            ntodo = 0;
        } else {
            /* recalculate ntodo and firstelem for next loop */
            if firstelem > nmax as usize {
                firstelem -= nmax as usize;
            } else {
                ntodo = firstelem as c_long;
                firstelem = 0;
            }
        }
    }

    *status
}

/// Convert the input array of 8-bit signed integers into an array of 32-bit
/// integers, in place. This will overwrite the input array with the new longer
/// array starting at the same memory location.
///
/// Note that aliasing the same memory location with pointers of different datatypes
/// is not allowed in strict ANSI C99, however it is  used here for efficency. In
/// principle, one could simply copy the input array in reverse order to the output
/// array, but this only works if the compiler performs the operation in strict
/// order.  Certain compiler optimization techniques may vioate this assumption.
/// Therefore, we first copy a section of the input array to a temporary
/// intermediate array, before copying the longer datatype values back to the
/// original array.
///
/// !!!!!!!!!!!!!!!!!
/// NOTE THAT THIS IS A SPECIALIZED ROUTINE THAT ADDS AN OFFSET OF 128 TO THE ARRAY
/// VALUES
/// !!!!!!!!!!!!!!!!!
fn fits_sbyte_to_int_inplace(
    sbytearray: &mut [c_schar],
    length: c_long,
    status: &mut c_int,
) -> c_int {
    let mut ntodo: c_long;
    let mut firstelem: usize;
    let nmax: c_long = 10000;

    if *status > 0 {
        return *status;
    }

    ntodo = nmax;
    if length < nmax {
        ntodo = length;
    }

    firstelem = (length - ntodo).try_into().unwrap(); /* first element to be converted */

    let mut intarray: Vec<c_int> = Vec::new();

    if intarray
        .try_reserve_exact(ntodo.try_into().unwrap())
        .is_err()
    {
        ffpmsg_str("Out of memory. (fits_sbyte_to_int_inplace)");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        intarray.resize(ntodo as usize, 0);
    }

    while ntodo > 0 {
        /* do datatype conversion into temp array */
        for ii in 0..(ntodo as usize) {
            intarray[ii] = c_int::from(sbytearray[ii + firstelem]) + 128; /* !! Note the offset !! Convert to int before adding so the offset does not overflow the i8. */
        }

        /* copy temp array back to alias */
        let aliasarray: &mut [c_int] = cast_slice_mut(sbytearray); /* alias pointer to the input array */
        aliasarray[firstelem..(firstelem + ntodo as usize)]
            .copy_from_slice(&intarray[..(ntodo as usize)]);

        if firstelem == 0 {
            /* we are all done */
            ntodo = 0;
        } else {
            /* recalculate ntodo and firstelem for next loop */
            if firstelem > nmax.try_into().unwrap() {
                firstelem -= nmax as usize;
            } else {
                ntodo = firstelem as c_long;
                firstelem = 0;
            }
        }
    }

    *status
}

/// The quantizing algorithms treat all N-dimensional tiles as if they
/// were 2 dimensions (trowsize * ntrows).  This sets trowsize to the
/// first dimensional size encountered that's > 1 (typically the X
/// dimension). ntrows will then be the product of the remaining dimensional
/// sizes.
///
/// Examples:  Tile = (5,4,1,3):  trowsize=5, ntrows=12
///  Tile = (1,1,5):  trowsize=5, ntrows=1
fn fits_calc_tile_rows(
    tlpixel: &[c_long],
    tfpixel: &[c_long],
    ndim: c_int,
    trowsize: &mut c_long,
    ntrows: &mut c_long,
    status: &mut c_int,
) -> c_int {
    let mut np: c_long = 0;

    if *status != 0 {
        return *status;
    }

    *trowsize = 0;
    *ntrows = 1;
    for ii in 0..(ndim as usize) {
        np = tlpixel[ii] - tfpixel[ii] + 1;
        if np > 1 {
            if (*trowsize) == 0 {
                *trowsize = np;
            } else {
                *ntrows *= np;
            }
        }
    }
    if (*trowsize) == 0 {
        /* Should only get here for the unusual case of all tile dimensions having size = 1  */
        *trowsize = 1;
    }

    *status
}

/// Tests ported from cfitsio's test_ricecomp.c
///
/// rsfitsio delegates Rice compression to the external `ricecomp` crate
/// (`RCEncoder`/`RCDecoder`), which is what `imcompress` drives internally. The
/// original C test exercised `fits_rcomp`/`fits_rdecomp` directly; here we drive
/// the same encode/decode round-trips through the crate API.
#[cfg(test)]
mod ricecomp_tests {
    use ricecomp::read::RCDecoder;
    use ricecomp::write::RCEncoder;

    /// Encode `input` (int), decode it back, and check the exact round-trip.
    fn rt_int(input: &[i32], nblock: usize) {
        let nx = input.len();
        let mut compressed = Vec::new();
        let nbytes = RCEncoder::new(&mut compressed)
            .encode(input, nx, nblock)
            .expect("encode failed");
        assert!(nbytes > 0);

        let mut out = vec![0u32; nx];
        RCDecoder::new()
            .decode(&compressed, nx, nblock, &mut out)
            .expect("decode failed");

        for i in 0..nx {
            assert_eq!(out[i] as i32, input[i], "mismatch at {i}");
        }
    }

    /// Encode `input` (short), decode it back, and check the exact round-trip.
    fn rt_short(input: &[i16], nblock: usize) {
        let nx = input.len();
        let mut compressed = Vec::new();
        let nbytes = RCEncoder::new(&mut compressed)
            .encode_short(input, nx, nblock)
            .expect("encode_short failed");
        assert!(nbytes > 0);

        let mut out = vec![0u16; nx];
        RCDecoder::new()
            .decode_short(&compressed, nx, nblock, &mut out)
            .expect("decode_short failed");

        for i in 0..nx {
            assert_eq!(out[i] as i16, input[i], "mismatch at {i}");
        }
    }

    /// Encode `input` (byte), decode it back, and check the exact round-trip.
    fn rt_byte(input: &[i8], nblock: usize) {
        let nx = input.len();
        let mut compressed = Vec::new();
        let nbytes = RCEncoder::new(&mut compressed)
            .encode_byte(input, nx, nblock)
            .expect("encode_byte failed");
        assert!(nbytes > 0);

        let mut out = vec![0u8; nx];
        RCDecoder::new()
            .decode_byte(&compressed, nx, nblock, &mut out)
            .expect("decode_byte failed");

        for i in 0..nx {
            assert_eq!(out[i] as i8, input[i], "mismatch at {i}");
        }
    }

    #[test]
    fn test_rcomp_roundtrip() {
        let original: Vec<i32> = (0..64).map(|i| i * 10).collect();
        rt_int(&original, 32);
    }

    #[test]
    fn test_rcomp_short_roundtrip() {
        let original: Vec<i16> = (0..64).map(|i| (i * 100) as i16).collect();
        rt_short(&original, 32);
    }

    #[test]
    fn test_rcomp_byte_roundtrip() {
        let original: Vec<i8> = (0..64).map(|i| i as i8).collect();
        rt_byte(&original, 32);
    }

    #[test]
    fn test_rcomp_uniform() {
        let original = [42i32; 64];
        let mut compressed = Vec::new();
        let nbytes = RCEncoder::new(&mut compressed)
            .encode(&original, 64, 32)
            .unwrap();
        assert!(nbytes > 0);
        assert!(nbytes <= 50, "uniform data should compress very small");

        let mut out = vec![0u32; 64];
        RCDecoder::new()
            .decode(&compressed, 64, 32, &mut out)
            .unwrap();
        assert!(out.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_rcomp_block_sizes() {
        let original: Vec<i32> = (0..128).map(|i| i * 5).collect();
        for &nblock in &[16, 32, 64] {
            rt_int(&original, nblock);
        }
    }

    #[test]
    fn test_rcomp_large_values() {
        let original: Vec<i32> = (0..32).map(|i| 1_000_000 + i).collect();
        rt_int(&original, 16);
    }

    #[test]
    fn test_rcomp_zeros() {
        rt_int(&[0i32; 64], 32);
    }

    #[test]
    fn test_rcomp_alternating() {
        let original: Vec<i32> = (0..64)
            .map(|i| if i % 2 != 0 { 100 } else { 200 })
            .collect();
        rt_int(&original, 32);
    }

    #[test]
    fn test_rcomp_short_large_array() {
        let original: Vec<i16> = (0..256).map(|i| (i * 10) as i16).collect();
        rt_short(&original, 32);
    }

    #[test]
    fn test_rcomp_byte_range() {
        let original: Vec<i8> = (0..128).map(|i| i as i8).collect();
        rt_byte(&original, 32);
    }

    #[test]
    fn test_rcomp_negative_values() {
        let original: Vec<i32> = (0..32)
            .map(|i| if i % 2 != 0 { -i * 10 } else { i * 10 })
            .collect();
        rt_int(&original, 16);
    }

    #[test]
    fn test_rcomp_high_entropy() {
        let original: Vec<i32> = (0..64)
            .map(|i| {
                (((i as u32).wrapping_mul(1103515245).wrapping_add(12345)) >> 16 & 0x7FFF) as i32
            })
            .collect();
        rt_int(&original, 32);
    }

    #[test]
    fn test_rcomp_short_negative() {
        let original: Vec<i16> = (0..64)
            .map(|i| (if i % 2 != 0 { -i * 100 } else { i * 100 }) as i16)
            .collect();
        rt_short(&original, 32);
    }

    #[test]
    fn test_rcomp_byte_negative() {
        let original: Vec<i8> = (0..64)
            .map(|i| (if i % 2 != 0 { -(i % 128) } else { i % 128 }) as i8)
            .collect();
        rt_byte(&original, 32);
    }

    #[test]
    fn test_rcomp_small_block() {
        let original: Vec<i32> = (0..64).map(|i| i * 3).collect();
        rt_int(&original, 8);
    }

    #[test]
    fn test_rcomp_extreme_values() {
        let original: Vec<i32> = (0..16).map(|i| 2_000_000_000 + i).collect();
        rt_int(&original, 16);
    }

    #[test]
    fn test_rcomp_short_extreme() {
        let original: Vec<i16> = (0..16).map(|i| (32000 + i) as i16).collect();
        rt_short(&original, 16);
    }

    #[test]
    fn test_rcomp_high_fs() {
        let original: Vec<i32> = (0..32)
            .map(|i| if i % 2 != 0 { 1_000_000 } else { 0 })
            .collect();
        rt_int(&original, 16);
    }

    #[test]
    fn test_rcomp_single_element() {
        rt_int(&[12345], 32);
    }

    #[test]
    fn test_rcomp_decreasing() {
        let original: Vec<i32> = (0..64).map(|i| 1000 - i * 10).collect();
        rt_int(&original, 32);
    }

    #[test]
    fn test_rdecomp_buffer_too_small() {
        // Fewer than 4 bytes cannot hold the first pixel; decode must fail.
        let compressed = [0u8, 0, 0];
        let mut out = vec![0u32; 64];
        assert!(
            RCDecoder::new()
                .decode(&compressed, 64, 32, &mut out)
                .is_err()
        );
    }

    #[test]
    fn test_rdecomp_short_small_nx() {
        rt_short(&[12345], 32);
    }

    #[test]
    fn test_rdecomp_byte_small_nx() {
        rt_byte(&[42], 32);
    }

    #[test]
    fn test_rcomp_fsmax_path() {
        let original: Vec<i32> = (0..32)
            .map(|i| if i % 2 != 0 { 0x7FFFFFFF } else { 0 })
            .collect();
        rt_int(&original, 16);
    }

    #[test]
    fn test_rcomp_short_fsmax_path() {
        let original: Vec<i16> = (0..32)
            .map(|i| if i % 2 != 0 { 32767 } else { -32768 })
            .collect();
        rt_short(&original, 16);
    }

    #[test]
    fn test_rcomp_byte_fsmax_path() {
        let original: Vec<i8> = (0..32)
            .map(|i| if i % 2 != 0 { 127 } else { -128 })
            .collect();
        rt_byte(&original, 16);
    }

    #[test]
    fn test_rcomp_varying_fs() {
        let mut original = vec![0i32; 128];
        for i in 0..32 {
            original[i] = i as i32;
        }
        for i in 32..64 {
            original[i] = ((i * 7919) % 10000) as i32;
        }
        for i in 64..96 {
            original[i] = ((i * 104729) % 1000000) as i32;
        }
        for i in 96..128 {
            original[i] = 42;
        }
        rt_int(&original, 32);
    }

    #[test]
    fn test_rcomp_partial_block() {
        // 100 is not divisible by the block size of 32.
        let original: Vec<i32> = (0..100).map(|i| i * 7).collect();
        rt_int(&original, 32);
    }

    #[test]
    fn test_rcomp_multi_block_zeros() {
        let original = [0i32; 128];
        let mut compressed = Vec::new();
        let nbytes = RCEncoder::new(&mut compressed)
            .encode(&original, 128, 32)
            .unwrap();
        assert!(nbytes > 0);
        assert!(nbytes <= 40, "all-zero data should compress tiny");

        let mut out = vec![0u32; 128];
        RCDecoder::new()
            .decode(&compressed, 128, 32, &mut out)
            .unwrap();
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_rcomp_constant_multi_block() {
        rt_int(&[999999i32; 128], 32);
    }

    #[test]
    fn test_rdecomp_truncated() {
        let original: Vec<i32> = (0..64).map(|i| i * 100).collect();
        let mut compressed = Vec::new();
        let nbytes = RCEncoder::new(&mut compressed)
            .encode(&original, 64, 32)
            .unwrap();
        assert!(nbytes > 0);

        // Half the bytes is not a complete stream; decode must fail.
        let mut out = vec![0u32; 64];
        assert!(
            RCDecoder::new()
                .decode(&compressed[..nbytes / 2], 64, 32, &mut out)
                .is_err()
        );
    }

    #[test]
    fn test_rdecomp_short_truncated() {
        let original: Vec<i16> = (0..64).map(|i| (i * 50) as i16).collect();
        let mut compressed = Vec::new();
        let nbytes = RCEncoder::new(&mut compressed)
            .encode_short(&original, 64, 32)
            .unwrap();

        let mut out = vec![0u16; 64];
        assert!(
            RCDecoder::new()
                .decode_short(&compressed[..nbytes / 2], 64, 32, &mut out)
                .is_err()
        );
    }

    #[test]
    fn test_rdecomp_byte_truncated() {
        let original: Vec<i8> = (0..64).map(|i| (i % 100) as i8).collect();
        let mut compressed = Vec::new();
        let nbytes = RCEncoder::new(&mut compressed)
            .encode_byte(&original, 64, 32)
            .unwrap();

        let mut out = vec![0u8; 64];
        assert!(
            RCDecoder::new()
                .decode_byte(&compressed[..nbytes / 2], 64, 32, &mut out)
                .is_err()
        );
    }

    #[test]
    fn test_rdecomp_nx_mismatch() {
        // Compress 32 values, but ask the decoder for 64: must fail.
        let original: Vec<i32> = (0..32).map(|i| i * 10).collect();
        let mut compressed = Vec::new();
        RCEncoder::new(&mut compressed)
            .encode(&original, 32, 16)
            .unwrap();

        let mut out = vec![0u32; 64];
        assert!(
            RCDecoder::new()
                .decode(&compressed, 64, 16, &mut out)
                .is_err()
        );
    }

    #[test]
    fn test_rcomp_nblock_larger_than_nx() {
        // nblock=32 but only 16 elements.
        let original: Vec<i32> = (0..16).map(|i| i * 5).collect();
        rt_int(&original, 32);
    }

    #[test]
    fn test_rcomp_nbits_edge() {
        let original: Vec<i32> = (0..32)
            .map(|i| match i % 4 {
                0 => 0,
                1 => 0x7F,
                2 => 0x7FFF,
                _ => 0x7FFFFF,
            })
            .collect();
        rt_int(&original, 16);
    }

    #[test]
    fn test_rdecomp_extra_bytes() {
        let original: Vec<i32> = (0..32).map(|i| i * 10).collect();
        let mut compressed = Vec::new();
        RCEncoder::new(&mut compressed)
            .encode(&original, 32, 16)
            .unwrap();

        // Trailing junk bytes should be tolerated by the decoder.
        compressed.push(0xFF);
        compressed.push(0xFF);

        let mut out = vec![0u32; 32];
        RCDecoder::new()
            .decode(&compressed, 32, 16, &mut out)
            .unwrap();
        for i in 0..32 {
            assert_eq!(out[i] as i32, original[i]);
        }
    }

    #[test]
    fn test_rdecomp_short_extra_bytes() {
        let original: Vec<i16> = (0..32).map(|i| (i * 100) as i16).collect();
        let mut compressed = Vec::new();
        RCEncoder::new(&mut compressed)
            .encode_short(&original, 32, 16)
            .unwrap();
        compressed.push(0xFF);
        compressed.push(0xFF);

        let mut out = vec![0u16; 32];
        RCDecoder::new()
            .decode_short(&compressed, 32, 16, &mut out)
            .unwrap();
        for i in 0..32 {
            assert_eq!(out[i] as i16, original[i]);
        }
    }

    #[test]
    fn test_rdecomp_byte_extra_bytes() {
        let original: Vec<i8> = (0..32).map(|i| (i * 2) as i8).collect();
        let mut compressed = Vec::new();
        RCEncoder::new(&mut compressed)
            .encode_byte(&original, 32, 16)
            .unwrap();
        compressed.push(0xFF);
        compressed.push(0xFF);

        let mut out = vec![0u8; 32];
        RCDecoder::new()
            .decode_byte(&compressed, 32, 16, &mut out)
            .unwrap();
        for i in 0..32 {
            assert_eq!(out[i] as i8, original[i]);
        }
    }

    #[test]
    fn test_rcomp_fsmax_nbits_path() {
        let original: Vec<i32> = (0..64)
            .map(|i| {
                if i % 2 != 0 {
                    0x7FFFFFFFi32
                } else {
                    0x80000001u32 as i32
                }
            })
            .collect();
        rt_int(&original, 32);
    }

    #[test]
    fn test_rcomp_bit_alignment() {
        let original: Vec<i32> = (0..48)
            .map(|i| ((i * 0x1234567) ^ (i << 16)) & 0x7FFFFFFF)
            .collect();
        rt_int(&original, 16);
    }

    #[test]
    fn test_rcomp_short_nbits_path() {
        let original: Vec<i16> = (0..64)
            .map(|i| if i % 2 != 0 { 32767 } else { -32768 })
            .collect();
        rt_short(&original, 8);
    }

    #[test]
    fn test_rcomp_byte_nbits_path() {
        let original: Vec<i8> = (0..64)
            .map(|i| if i % 2 != 0 { 127 } else { -128 })
            .collect();
        rt_byte(&original, 8);
    }

    #[test]
    fn test_rcomp_various_block_sizes() {
        let original: Vec<i32> = (0..96)
            .map(|i| {
                if i % 3 != 0 {
                    ((i * 999983) % 10000000) as i32
                } else {
                    0
                }
            })
            .collect();
        for &nblock in &[8, 12, 20, 24] {
            rt_int(&original, nblock);
        }
    }

    #[test]
    fn test_rcomp_prime_sequence() {
        let primes: [i32; 32] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 1009, 1013, 1019, 1021,
            1031, 1033, 1039, 1049, 1051, 1061, 10007, 10009, 10037, 10039, 10061, 10067,
        ];
        rt_int(&primes, 8);
    }

    #[test]
    fn test_rcomp_short_mixed_blocks() {
        let mut original = vec![0i16; 128];
        for i in 0..32 {
            original[i] = 1000;
        }
        for i in 32..64 {
            original[i] = (i * 10) as i16;
        }
        for i in 64..96 {
            original[i] = if i % 2 != 0 { 32000 } else { -32000 };
        }
        for i in 96..128 {
            original[i] = ((i * 7919) % 30000) as i16;
        }
        rt_short(&original, 32);
    }

    #[test]
    fn test_rcomp_byte_boundaries() {
        let original: Vec<i8> = (0..64)
            .map(|i| {
                let mut val = (i * 5) % 256;
                if val > 127 {
                    val -= 256;
                }
                val as i8
            })
            .collect();
        rt_byte(&original, 16);
    }

    #[test]
    fn test_rcomp_byte_fsmax_extreme() {
        let original: Vec<i8> = (0..128)
            .map(|i| if i % 2 != 0 { 127 } else { -128 })
            .collect();
        for &nblock in &[4, 8, 16] {
            rt_byte(&original, nblock);
        }
    }

    #[test]
    fn test_rcomp_int_fsmax_extreme() {
        let original: Vec<i32> = (0..64)
            .map(|i| if i % 2 != 0 { 0xFFFFFFFFu32 as i32 } else { 0 })
            .collect();
        rt_int(&original, 8);
    }
}

/// Tests ported from cfitsio's test_imcompress.c
///
/// These drive the high-level tiled-image compression API end to end
/// (`fits_set_compression_type`, `fits_img_compress`, transparent decompression
/// on read, etc.) against on-disk FITS files.
#[cfg(test)]
mod tests {
    // fits_set_noise_bits is deprecated upstream; the test below covers it on purpose.
    #[allow(deprecated)]
    use super::{
        c_char, c_int, fits_get_compression_type_safe, fits_get_dither_seed_safe,
        fits_get_noise_bits_safe, fits_get_quantize_level_safe, fits_get_tile_dim_safe,
        fits_img_compress_safe, fits_set_compression_type_safe, fits_set_dither_seed_safe,
        fits_set_noise_bits_safe, fits_set_quantize_level_safe, fits_set_quantize_method_safe,
        fits_set_tile_dim_safe, fits_shuffle_2bytes, fits_shuffle_4bytes, fits_shuffle_8bytes,
        fits_unshuffle_2bytes, fits_unshuffle_4bytes, fits_unshuffle_8bytes,
    };
    use crate::c_types::c_long;
    use crate::cfileio::{ffclos_safe, ffinit_safe, ffopen_safe};
    use crate::fitscore::{ffmahd_safe, fits_is_compressed_image_safe};
    use crate::fitsio::{
        BYTE_IMG, DOUBLE_IMG, GZIP_1, GZIP_2, HCOMPRESS_1, LONG_IMG, LONGLONG, PLIO_1, READONLY,
        RICE_1, SHORT_IMG, SUBTRACTIVE_DITHER_2, TBYTE, TDOUBLE, TINT, TSHORT, fitsfile,
    };
    use crate::getcol::ffgpv_safe;
    use crate::helpers::aligned::AlignedBytes;
    use crate::helpers::testhelpers::{to_buf, with_temp_file};
    use crate::putcol::ffppr_safe;
    use crate::putkey::ffcrim_safe;
    use bytemuck::{cast_slice, cast_slice_mut};

    /// Create a fresh FITS file with a single empty image HDU of the given type.
    fn create_test_image(filename: &str, bitpix: i32, nx: c_long, ny: c_long) -> Box<fitsfile> {
        let mut status = 0;
        let name = to_buf(filename);
        let mut fptr: Option<Box<fitsfile>> = None;
        ffinit_safe(&mut fptr, &name, &mut status);
        assert_eq!(status, 0, "ffinit failed");
        {
            let f = fptr.as_deref_mut().unwrap();
            let naxes = [nx, ny];
            ffcrim_safe(f, bitpix, 2, &naxes, &mut status);
            assert_eq!(status, 0, "ffcrim failed");
        }
        fptr.take().unwrap()
    }

    /// Full compression round-trip: write `original_bytes` as an image, compress
    /// it with `ctype` into a second file, then read it back (transparently
    /// decompressed) and return the decompressed bytes.
    fn compress_roundtrip(
        filename: &str,
        bitpix: i32,
        datatype: i32,
        ctype: i32,
        nx: c_long,
        ny: c_long,
        original_bytes: &[u8],
    ) -> Vec<u8> {
        let n = (nx * ny) as LONGLONG;
        let mut status = 0;
        let in_name = to_buf(filename);
        let out_name = to_buf(&format!("{filename}.compressed"));

        // Create the uncompressed input image.
        let mut infptr: Option<Box<fitsfile>> = None;
        ffinit_safe(&mut infptr, &in_name, &mut status);
        assert_eq!(status, 0, "create input");
        {
            let f = infptr.as_deref_mut().unwrap();
            let naxes = [nx, ny];
            ffcrim_safe(f, bitpix, 2, &naxes, &mut status);
            assert_eq!(status, 0, "create img");
            ffppr_safe(f, datatype, 1, n, original_bytes, &mut status);
            assert_eq!(status, 0, "write img");
        }
        ffclos_safe(infptr.take().unwrap(), &mut status);
        assert_eq!(status, 0, "close input");

        // Reopen the input read-only and compress it into a new file.
        let mut infptr: Option<Box<fitsfile>> = None;
        ffopen_safe(&mut infptr, &in_name, READONLY, &mut status);
        assert_eq!(status, 0, "open input");
        let mut outfptr: Option<Box<fitsfile>> = None;
        ffinit_safe(&mut outfptr, &out_name, &mut status);
        assert_eq!(status, 0, "create output");
        {
            let of = outfptr.as_deref_mut().unwrap();
            fits_set_compression_type_safe(of, ctype, &mut status);
            assert_eq!(status, 0, "set compression type");
        }
        {
            let inf = infptr.as_deref_mut().unwrap();
            let of = outfptr.as_deref_mut().unwrap();
            fits_img_compress_safe(inf, of, &mut status);
            assert_eq!(status, 0, "img compress");
        }
        ffclos_safe(infptr.take().unwrap(), &mut status);
        ffclos_safe(outfptr.take().unwrap(), &mut status);
        assert_eq!(status, 0, "close after compress");

        // Reopen the compressed file; the compressed image lives in HDU 2.
        let mut outfptr: Option<Box<fitsfile>> = None;
        ffopen_safe(&mut outfptr, &out_name, READONLY, &mut status);
        assert_eq!(status, 0, "reopen output");
        let of = outfptr.as_deref_mut().unwrap();
        ffmahd_safe(of, 2, None, &mut status);
        assert_eq!(status, 0, "movabs hdu 2");

        let is_comp = fits_is_compressed_image_safe(of, &mut status);
        assert_eq!(status, 0);
        assert_eq!(is_comp, 1, "expected a compressed image in HDU 2");

        /* ffgpv_safe reinterprets this byte buffer as `datatype`, so it has to
        be aligned for the widest type the callers pass, not just for u8. */
        let mut decompressed = AlignedBytes::new();
        decompressed
            .try_resize_zeroed(original_bytes.len())
            .expect("allocating the decompressed buffer");
        ffgpv_safe(
            of,
            datatype,
            1,
            n,
            None,
            &mut decompressed,
            None,
            &mut status,
        );
        assert_eq!(status, 0, "read (decompress) img");

        ffclos_safe(outfptr.take().unwrap(), &mut status);
        assert_eq!(status, 0, "close final");

        decompressed.to_vec()
    }

    /// Regression test for https://github.com/cruzzil/rsfitsio/issues/82
    ///
    /// Writing a lossy-compressed float image with SUBTRACTIVE_DITHER_2 and an
    /// explicit dither seed must be bit-for-bit reproducible with cfitsio. The
    /// failing `fitsio` test compresses `np.random.RandomState(42).normal((100,
    /// 100))` with RICE + SUBTRACTIVE_DITHER_2 + dither_seed=42, reads it back,
    /// and compares against cfitsio's reconstruction of the same input.
    ///
    /// The fixtures are:
    ///   * `issue82_input_f64.bin` - the exact float64 input array (C order), and
    ///   * `issue82_cfitsio_recon_f64.bin` - cfitsio's compress-then-decompress
    ///     reconstruction of that input (generated with a non-FMA cfitsio via the
    ///     `fitsio` package).
    ///
    /// This exercises the direct write path (`fits_create_img` on a
    /// compression-configured file followed by `ffppr`), exactly as the `fitsio`
    /// wrapper does. rsfitsio must produce the same reconstruction as cfitsio.
    #[test]
    fn test_dither2_seed_compression_matches_cfitsio_issue_82() {
        // Load little-endian float64 fixtures (`include_bytes!` is only byte
        // aligned, so decode explicitly rather than reinterpreting the slice).
        fn load_f64_le(bytes: &[u8]) -> Vec<f64> {
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect()
        }

        // float64 input, C order (100x100).
        let input_bytes: &[u8] = include_bytes!("../tests/fixtures/issue82_input_f64.bin");
        // cfitsio's reconstruction (ground truth).
        let recon_bytes: &[u8] = include_bytes!("../tests/fixtures/issue82_cfitsio_recon_f64.bin");

        let input = load_f64_le(input_bytes);
        let expected = load_f64_le(recon_bytes);

        let nx: c_long = 100;
        let ny: c_long = 100;
        let n = (nx * ny) as usize;
        assert_eq!(input.len(), n);
        assert_eq!(expected.len(), n);
        let seed = 42;

        with_temp_file(|filename| {
            let mut status = 0;
            let name = to_buf(filename);

            // Create a fresh output file and configure compression exactly as the
            // fitsio wrapper does: set compression type, quantize method, dither
            // seed, then create the (compressed) image HDU and write the pixels.
            let mut fptr: Option<Box<fitsfile>> = None;
            ffinit_safe(&mut fptr, &name, &mut status);
            assert_eq!(status, 0, "create file");
            {
                let f = fptr.as_deref_mut().unwrap();
                fits_set_compression_type_safe(f, RICE_1, &mut status);
                assert_eq!(status, 0, "set compression type");
                fits_set_quantize_method_safe(f, SUBTRACTIVE_DITHER_2, &mut status);
                assert_eq!(status, 0, "set quantize method");
                fits_set_dither_seed_safe(f, seed, &mut status);
                assert_eq!(status, 0, "set dither seed");

                // fits_create_img on a compression-configured file creates the
                // compressed image HDU.
                let naxes = [nx, ny];
                ffcrim_safe(f, DOUBLE_IMG, 2, &naxes, &mut status);
                assert_eq!(status, 0, "create (compressed) img");

                // Write (and compress) the pixels. `input` is a Vec<f64>, so its
                // byte view is 8-byte aligned.
                let input_bytes_aligned: &[u8] = cast_slice(&input);
                ffppr_safe(
                    f,
                    TDOUBLE,
                    1,
                    n as LONGLONG,
                    input_bytes_aligned,
                    &mut status,
                );
                assert_eq!(status, 0, "write img");
            }
            ffclos_safe(fptr.take().unwrap(), &mut status);
            assert_eq!(status, 0, "close after write");

            // Reopen and read (transparently decompress).
            let mut fptr: Option<Box<fitsfile>> = None;
            ffopen_safe(&mut fptr, &name, READONLY, &mut status);
            assert_eq!(status, 0, "reopen");
            let f = fptr.as_deref_mut().unwrap();
            // The compressed image lives in HDU 2 (after the dummy primary).
            ffmahd_safe(f, 2, None, &mut status);
            assert_eq!(status, 0, "movabs hdu 2");
            assert_eq!(
                fits_is_compressed_image_safe(f, &mut status),
                1,
                "expected a compressed image in HDU 2"
            );

            // Read into an 8-byte-aligned buffer (a Vec<f64>) so the internal
            // reinterpretation to f64 is well-aligned.
            let mut got = vec![0f64; n];
            {
                let out_bytes: &mut [u8] = cast_slice_mut(&mut got);
                ffgpv_safe(
                    f,
                    TDOUBLE,
                    1,
                    n as LONGLONG,
                    None,
                    out_bytes,
                    None,
                    &mut status,
                );
            }
            assert_eq!(status, 0, "read (decompress) img");
            ffclos_safe(fptr.take().unwrap(), &mut status);
            assert_eq!(status, 0, "close final");

            // Must match cfitsio's reconstruction exactly (as np.array_equal).
            let mut nbad = 0usize;
            let mut first: Option<(usize, f64, f64)> = None;
            let mut max_abs = 0.0f64;
            for i in 0..n {
                if got[i].to_bits() != expected[i].to_bits() {
                    nbad += 1;
                    if first.is_none() {
                        first = Some((i, got[i], expected[i]));
                    }
                    let d = (got[i] - expected[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
            }

            assert!(
                nbad == 0,
                "rsfitsio lossy compression does not match cfitsio: {nbad}/{n} pixels differ \
                 (first at index {:?}, max abs diff {max_abs})",
                first
            );
        });
    }

    #[test]
    fn test_compression_type() {
        with_temp_file(|filename| {
            let mut fptr = create_test_image(filename, SHORT_IMG, 64, 64);
            let mut status = 0;
            let mut ctype = 0;

            for &want in &[RICE_1, GZIP_1, GZIP_2, PLIO_1, HCOMPRESS_1] {
                fits_set_compression_type_safe(&mut fptr, want, &mut status);
                assert_eq!(status, 0);
                fits_get_compression_type_safe(&mut fptr, &mut ctype, &mut status);
                assert_eq!(status, 0);
                assert_eq!(ctype, want);
            }

            ffclos_safe(fptr, &mut status);
            assert_eq!(status, 0);
        });
    }

    #[test]
    fn test_tile_dimensions() {
        with_temp_file(|filename| {
            let mut fptr = create_test_image(filename, SHORT_IMG, 64, 64);
            let mut status = 0;

            let dims_in: [c_long; 2] = [32, 32];
            fits_set_tile_dim_safe(&mut fptr, 2, &dims_in, &mut status);
            assert_eq!(status, 0);

            let mut dims_out: [c_long; 2] = [0, 0];
            fits_get_tile_dim_safe(&mut fptr, 2, &mut dims_out, &mut status);
            assert_eq!(status, 0);
            assert_eq!(dims_out[0], 32);
            assert_eq!(dims_out[1], 32);

            ffclos_safe(fptr, &mut status);
            assert_eq!(status, 0);
        });
    }

    #[test]
    fn test_quantize_level() {
        with_temp_file(|filename| {
            let mut fptr = create_test_image(filename, SHORT_IMG, 64, 64);
            let mut status = 0;

            fits_set_quantize_level_safe(&mut fptr, 16.0, &mut status);
            assert_eq!(status, 0);

            let mut qlevel = 0.0f32;
            fits_get_quantize_level_safe(&mut fptr, &mut qlevel, &mut status);
            assert_eq!(status, 0);
            assert!((qlevel - 16.0).abs() < 0.001);

            ffclos_safe(fptr, &mut status);
            assert_eq!(status, 0);
        });
    }

    #[test]
    fn test_noise_bits() {
        with_temp_file(|filename| {
            let mut fptr = create_test_image(filename, SHORT_IMG, 64, 64);
            let mut status = 0;

            /* fits_set_noise_bits is deprecated in CFITSIO in favour of
            fits_set_quantize_level; this test covers the deprecated path itself. */
            #[allow(deprecated)]
            let _ = fits_set_noise_bits_safe(&mut fptr, 4, &mut status);
            assert_eq!(status, 0);

            let mut noisebits = 0;
            fits_get_noise_bits_safe(&mut fptr, &mut noisebits, &mut status);
            assert_eq!(status, 0);
            assert_eq!(noisebits, 4);

            ffclos_safe(fptr, &mut status);
            assert_eq!(status, 0);
        });
    }

    #[test]
    fn test_dither_seed() {
        with_temp_file(|filename| {
            let mut fptr = create_test_image(filename, SHORT_IMG, 32, 32);
            let mut status = 0;

            fits_set_dither_seed_safe(&mut fptr, 1234, &mut status);
            assert_eq!(status, 0);

            let mut seed = 0;
            fits_get_dither_seed_safe(&mut fptr, &mut seed, &mut status);
            assert_eq!(status, 0);
            assert_eq!(seed, 1234);

            ffclos_safe(fptr, &mut status);
            assert_eq!(status, 0);
        });
    }

    #[test]
    fn test_is_compressed_uncompressed() {
        with_temp_file(|filename| {
            let mut fptr = create_test_image(filename, SHORT_IMG, 32, 32);
            let mut status = 0;

            let is_comp = fits_is_compressed_image_safe(&mut fptr, &mut status);
            assert_eq!(status, 0);
            assert_eq!(is_comp, 0, "plain image must not report as compressed");

            ffclos_safe(fptr, &mut status);
            assert_eq!(status, 0);
        });
    }

    #[test]
    fn test_rice_compress_short() {
        with_temp_file(|filename| {
            let original: Vec<i16> = (0..64 * 64).map(|i| (i % 1000) as i16).collect();
            let decompressed = compress_roundtrip(
                filename,
                SHORT_IMG,
                TSHORT,
                RICE_1,
                64,
                64,
                cast_slice(&original),
            );
            assert_eq!(decompressed, cast_slice::<i16, u8>(&original));
        });
    }

    #[test]
    fn test_gzip_compress() {
        with_temp_file(|filename| {
            let original: Vec<i32> = (0..32 * 32).map(|i| i * 100).collect();
            let decompressed = compress_roundtrip(
                filename,
                LONG_IMG,
                TINT,
                GZIP_1,
                32,
                32,
                cast_slice(&original),
            );
            assert_eq!(decompressed, cast_slice::<i32, u8>(&original));
        });
    }

    #[test]
    fn test_plio_compress() {
        with_temp_file(|filename| {
            // Mask-like data: mostly zeros with occasional ones.
            let original: Vec<i16> = (0..64 * 64)
                .map(|i| if i % 10 == 0 { 1 } else { 0 })
                .collect();
            let decompressed = compress_roundtrip(
                filename,
                SHORT_IMG,
                TSHORT,
                PLIO_1,
                64,
                64,
                cast_slice(&original),
            );
            assert_eq!(decompressed, cast_slice::<i16, u8>(&original));
        });
    }

    #[test]
    fn test_hcompress_compress() {
        with_temp_file(|filename| {
            let original: Vec<i32> = (0..64 * 64).map(|i| (i % 64) + (i / 64) * 100).collect();
            let decompressed = compress_roundtrip(
                filename,
                LONG_IMG,
                TINT,
                HCOMPRESS_1,
                64,
                64,
                cast_slice(&original),
            );
            // Lossless with default settings for integer data.
            assert_eq!(decompressed, cast_slice::<i32, u8>(&original));
        });
    }

    #[test]
    fn test_compress_byte_image() {
        with_temp_file(|filename| {
            let original: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
            let decompressed =
                compress_roundtrip(filename, BYTE_IMG, TBYTE, RICE_1, 32, 32, &original);
            assert_eq!(decompressed, original);
        });
    }

    // Byte (un)shuffling helpers used by GZIP_2.  See
    // https://github.com/cruzzil/rsfitsio/issues/89: the unshuffle functions
    // used to run their destination cursor one step past the start of the
    // buffer, which panicked on the usize underflow whenever overflow checks
    // were on.

    /// `length` values of `nbytes` bytes each, with recognisable byte values:
    /// value `ii` holds the bytes `10 * ii + 1 ..= 10 * ii + nbytes`.
    fn sample(length: usize, nbytes: usize) -> Vec<c_char> {
        (0..length)
            .flat_map(|ii| (1..=nbytes).map(move |b| (10 * ii + b) as u8))
            .map(|b| b as c_char)
            .collect()
    }

    #[test]
    fn test_shuffle_4bytes_groups_bytes_into_planes() {
        let mut heap = sample(3, 4);
        let mut status: c_int = 0;
        fits_shuffle_4bytes(&mut heap, 3, &mut status);

        // all the first bytes, then all the second bytes, and so on
        let expected: &[u8] = &[1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24];
        assert_eq!(cast_slice::<c_char, u8>(&heap), expected);
        assert_eq!(status, 0);
    }

    #[test]
    fn test_unshuffle_4bytes_restores_the_values() {
        let shuffled: Vec<c_char> = [1u8, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24]
            .iter()
            .map(|&b| b as c_char)
            .collect();
        let mut heap = shuffled;
        let mut status: c_int = 0;
        fits_unshuffle_4bytes(&mut heap, 3, &mut status);

        assert_eq!(heap, sample(3, 4));
        assert_eq!(status, 0);
    }

    /// Unshuffling is the exact inverse of shuffling, for every width and for
    /// the lengths that used to trip the cursor underflow.
    #[test]
    fn test_unshuffle_inverts_shuffle() {
        for length in 0..8_usize {
            for nbytes in [2_usize, 4, 8] {
                let original = sample(length, nbytes);
                let mut heap = original.clone();
                let mut status: c_int = 0;

                match nbytes {
                    2 => {
                        fits_shuffle_2bytes(&mut heap, length as i64, &mut status);
                        fits_unshuffle_2bytes(&mut heap, length as i64, &mut status);
                    }
                    4 => {
                        fits_shuffle_4bytes(&mut heap, length as i64, &mut status);
                        fits_unshuffle_4bytes(&mut heap, length as i64, &mut status);
                    }
                    _ => {
                        fits_shuffle_8bytes(&mut heap, length as i64, &mut status);
                        fits_unshuffle_8bytes(&mut heap, length as i64, &mut status);
                    }
                }

                assert_eq!(heap, original, "length {length}, {nbytes}-byte values");
                assert_eq!(status, 0);
            }
        }
    }

    /// The heap buffer is usually larger than the shuffled data; nothing past
    /// `length * nbytes` may be touched.
    #[test]
    fn test_unshuffle_leaves_trailing_bytes_alone() {
        let mut heap = sample(2, 8);
        heap.extend_from_slice(cast_slice_mut::<u8, c_char>(&mut [0xAA_u8; 5]));
        let mut status: c_int = 0;

        fits_shuffle_8bytes(&mut heap[..16], 2, &mut status);
        fits_unshuffle_8bytes(&mut heap, 2, &mut status);

        assert_eq!(heap[..16], sample(2, 8)[..]);
        assert!(heap[16..].iter().all(|&b| b as u8 == 0xAA));
        assert_eq!(status, 0);
    }
}
