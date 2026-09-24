//! The low-level file access routines, and the extended filename syntax.
//!
//! Opening a file here is more than opening a path. CFITSIO accepts an extended
//! filename syntax in which the name carries a driver prefix (`file://`,
//! `mem://`, `ftp://`, `shmem://`, …), an HDU or extension to move to, an image
//! section, a row filter, a column filter and a tile-compression specification.
//! This module parses that syntax, dispatches to the right driver from the
//! table below, and applies the filters -- which is why opening a file can
//! create a temporary in-memory file and copy a filtered subset into it.
//!
//! Drivers register themselves into `DRIVER_TABLE` at first use, each supplying
//! the open / close / seek / read / write hooks of [`fitsdriver`]. The
//! individual drivers live in the `drvr*` modules.
//!
//! Ported from CFITSIO's `cfileio.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center. The filename syntax is documented in full in the
//! "Extended File Name Syntax" chapter of the CFITSIO User's Reference Guide.
#![warn(missing_docs)]

use core::ffi::CStr;
use core::slice;
use core::{cmp, mem, ptr};
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::sync::{Mutex, OnceLock};

use crate::c_types::{FILE, c_char, c_int, c_long, c_short, c_void};
use crate::drvrnet::{fits_dwnld_prog_bar, fits_net_timeout, https_set_verbose};
use crate::grparser::fits_execute_template_safe;
use crate::helpers::boxed::box_try_new;
use crate::helpers::cfile::{CFile, fgets};
use crate::helpers::vec_raw_parts::vec_into_raw_parts;
use bytemuck::{cast_mut, cast_slice, cast_slice_mut};
use errno::errno;
use libc::ERANGE;

use crate::aliases::rust_api::fits_read_key_str;
use crate::drvrfile::{
    file_checkfile, file_close, file_compress_open, file_create, file_flush, file_getoptions,
    file_getversion, file_init, file_is_compressed, file_open, file_openfile, file_read,
    file_remove, file_seek, file_setoptions, file_shutdown, file_size, file_truncate, file_write,
    fits_stream_close, fits_stream_create, fits_stream_flush, fits_stream_open, fits_stream_read,
    fits_stream_seek, fits_stream_size, fits_stream_write,
};
use crate::drvrmem::{
    mem_close_comp_unsafe, mem_close_free_unsafe, mem_close_keep, mem_compress_open,
    mem_compress_openrw, mem_create, mem_create_comp_unsafe, mem_getoptions, mem_getversion,
    mem_init, mem_iraf_open, mem_openmem, mem_rawfile_open, mem_read_unsafe, mem_seek,
    mem_setoptions, mem_shutdown, mem_size, mem_truncate_unsafe, mem_write_unsafe, stdin_checkfile,
    stdin_open, stdout_close_unsafe,
};
use crate::fitscore::{ffcrhd_safe, ffgisz_safe, ffpmsg_cstr, ffxmsg_safe};

use crate::KeywordDatatypeMut;
#[cfg(all(feature = "shared_mem", not(target_os = "windows")))]
use crate::drvrsmem::{
    smem_close, smem_create, smem_flush, smem_getoptions, smem_getversion, smem_init, smem_open,
    smem_read, smem_remove, smem_seek, smem_setoptions, smem_shutdown, smem_size, smem_write,
};
use crate::editcol::{ffdcol_safe, fficol_safe};
use crate::edithdu::{ffcopy_safe, ffcphd_safe};
use crate::eval_f::{
    ffcalc_inplace_safe, ffffrw_safe, fffrow_safe, ffsrow_inplace_safe, ffsrow_safe,
    fits_pixel_filter_safe,
};
use crate::fitscore::{
    ALLOCATIONS, ffchdu, ffcmrk_safe, ffcmsg_safe, ffgcnn_safe, ffgcno_safe, ffgcprll, ffgerr_safe,
    ffghadll_safe, ffghdn_safe, ffghdt_safe, ffgidm_safe, ffgidt_safe, ffgiprll_safe, ffgkcl_safe,
    ffgmsg_safe, ffgncl_safe, ffgnrw_safe, ffgtclll_safe, ffkeyn_safe, ffmahd_safe, ffmnhd_safe,
    ffmrhd_safe, ffpmrk_safe, ffpmsg_slice, ffpmsg_str, ffrdef_safe, ffrhdu_safe, ffupch_safe,
    fits_strcasecmp, fits_strncasecmp, fits_translate_keywords_safe,
};
use crate::getcolb::ffgsvb_safe;
use crate::getcold::ffgsvd_safe;
use crate::getcole::ffgsve_safe;
use crate::getcoli::ffgsvi_safe;
use crate::getcolj::ffgsvjj_safe;
use crate::getcolk::ffgsvk_safe;
use crate::getkey::{
    ffgcrd_safe, ffghsp_safe, ffgky_safe, ffgkyl_safe, ffgrec_safe, ffgtdmll_safe, ffmaky_safe,
};
use crate::group::{fits_clean_url, fits_get_cwd, fits_path2url};
use crate::histo::{ffbinse, ffhist2e};
use crate::imcompress::{
    fits_set_compression_type_safe, fits_set_hcomp_scale_safe, fits_set_hcomp_smooth_safe,
    fits_set_quantize_level_safe, fits_set_quantize_method_safe, fits_set_tile_dim_safe,
};
use crate::modkey::{ffdkey_safe, ffmkyd_safe, ffmkyj_safe, ffmkys_safe, ffmnam_safe};
use crate::putcol::ffpcl_safe;
use crate::putcolb::ffpprb_safe;
use crate::putcold::ffppnd_safe;
use crate::putcole::ffppne_safe;
use crate::putcoli::ffppri_safe;
use crate::putcolj::ffpprjj_safe;
use crate::putcolk::ffpprk_safe;
use crate::putkey::ffcrimll_safe;
use crate::putkey::{ffcrim_safe, ffphis_safe, ffprec_safe, ffptdmll_safe};
use crate::relibc::header::stdio::{sscanf_d, sscanf_ld};
use crate::scalnull::ffpscl_safe;
use crate::wrappers::*;
use crate::zcompress::uncompress2mem_from_mem;
use crate::{FFLOCK, FFUNLOCK};
use crate::{bb, cs};
use crate::{buffers::*, raw_to_slice};
use crate::{fitsio::*, int_snprintf};
use crate::{fitsio2::*, slice_to_str};

/// Maximum length of a file type prefix, e.g. `http://`.
pub const MAX_PREFIX_LEN: usize = 20;
/// Maximum number of file I/O drivers that can be registered.
pub const MAX_DRIVERS: usize = 31;

pub(crate) trait Driver {}

/// A driver's `checkfile` hook: given the parsed URL type it may rewrite the
/// input and output file names before the file is opened.
pub type CheckFileFn = fn(
    urltype: &mut [c_char; MAX_PREFIX_LEN],
    infile: &mut [c_char; FLEN_FILENAME],
    outfile: &mut [c_char; FLEN_FILENAME],
) -> c_int;

/// A driver's `open` hook.
pub type DriverOpenFn = fn(filename: &mut [c_char], rwmode: c_int, handle: &mut c_int) -> c_int;

/// One registered I/O driver: the hooks that implement a filename prefix.
///
/// A driver is selected by matching the prefix of the file name. `close`,
/// `size`, `seek`, `read` and `write` are required; the rest are optional and
/// are skipped when absent.
pub struct fitsdriver {
    /// File name prefix this driver handles, e.g. `mem://`.
    pub prefix: [c_char; MAX_PREFIX_LEN],
    /// Called once when the driver is first registered.
    pub init: Option<fn() -> c_int>,
    /// Called when the library shuts down.
    pub shutdown: Option<fn() -> c_int>,
    /// Set a driver-specific option.
    pub setoptions: Option<fn(option: c_int) -> c_int>,
    /// Read back the driver-specific options.
    pub getoptions: Option<fn(options: &mut c_int) -> c_int>,
    /// Report the driver's version.
    pub getversion: Option<fn(version: &mut c_int) -> c_int>,
    /// Inspect and possibly rewrite the file names before opening.
    pub checkfile: Option<CheckFileFn>,
    /// Open an existing file.
    pub open: Option<DriverOpenFn>,
    /// Create a new file.
    pub create:
        Option<fn(filename: &mut [c_char; FLEN_FILENAME], drivehandle: &mut c_int) -> c_int>,
    /// Truncate the file to `size` bytes.
    pub truncate: Option<fn(drivehandle: c_int, size: usize) -> c_int>,
    /// Close an open file.
    pub close: fn(drivehandle: c_int) -> c_int,
    /// Delete a file by name.
    pub remove: Option<fn(filename: &[c_char]) -> c_int>,
    /// Report the current size of the file in bytes.
    pub size: fn(drivehandle: c_int, size: &mut usize) -> c_int,
    /// Flush any buffered writes to the underlying medium.
    pub flush: Option<fn(drivehandle: c_int) -> c_int>,
    /// Seek to a byte offset.
    pub seek: fn(drivehandle: c_int, offset: LONGLONG) -> c_int,
    /// Read `nbytes` bytes into `buffer`.
    pub read: fn(drivehandle: c_int, buffer: &mut [u8], nbytes: usize) -> c_int,
    /// Write `nbyte` bytes from `buffer`.
    pub write: fn(drivehandle: c_int, buffer: &[u8], nbyte: usize) -> c_int,
}

/// The registered I/O drivers, populated on first use.
pub static DRIVER_TABLE: OnceLock<Vec<fitsdriver>> = OnceLock::new();

/// Every currently open file, used by `fits_already_open` to notice when a file
/// is opened a second time so that both handles share one [`FITSfile`].
pub static mut FPTR_TABLE: [*mut FITSfile; NMAXFILES] =
    [core::ptr::null_mut::<FITSfile>(); NMAXFILES];

/// True until the library has been initialized and the drivers registered.
pub static NEED_TO_INITIALIZE: Mutex<bool> = Mutex::new(true);

/// Index of the `stream://` driver in [`DRIVER_TABLE`].
pub static STREAM_DRIVER: Mutex<c_int> = Mutex::new(0);

pub(crate) fn fitsio_init_lock() -> c_int {
    0
}

/// Open an existing FITS file in core memory.  This is a specialized version
/// of ffopen.
///
/// # Parameters
///
/// * `fptr`        — (O) FITS file pointer
/// * `name`        — (I) name of file to open
/// * `mode`        — (I) 0 = open readonly; 1 = read/write
/// * `buffptr`     — (I) address of memory pointer
/// * `buffsize`    — (I) size of buffer, in bytes
/// * `deltasize`   — (I) increment for future realloc's
/// * `mem_realloc` — function
/// * `status`      — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffomem(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    mode: c_int,
    buffptr: *mut *mut c_void,
    buffsize: *mut usize,
    deltasize: usize,
    mem_realloc: unsafe extern "C" fn(p: *mut c_void, newsize: usize) -> *mut c_void,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let buffsize = buffsize.as_mut().expect(NULL_MSG);

        raw_to_slice!(name);

        ffomem_safe(
            fptr,
            name,
            mode,
            buffptr,
            buffsize,
            deltasize,
            mem_realloc,
            status,
        )
    }
}

/// # Parameters
///
/// * `fptr`        — (O) FITS file pointer
/// * `name`        — (I) name of file to open
/// * `mode`        — (I) 0 = open readonly; 1 = read/write
/// * `buffptr`     — (I) address of memory pointer
/// * `buffsize`    — (I) size of buffer, in bytes
/// * `deltasize`   — (I) increment for future realloc's
/// * `mem_realloc` — function
/// * `status`      — (IO) error status
pub fn ffomem_safe(
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    mode: c_int,
    buffptr: *mut *mut c_void,
    buffsize: &mut usize,
    deltasize: usize,
    mem_realloc: unsafe extern "C" fn(p: *mut c_void, newsize: usize) -> *mut c_void,
    status: &mut c_int,
) -> c_int {
    let mut driver: c_int = 0;
    let mut handle: c_int = 0;
    let mut hdutyp: c_int = 0;

    let mut movetotype: c_int = 0;
    let mut extvers: c_int = 0;
    let mut extnum: c_int = 0;
    let mut extname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut filesize: usize = 0;
    let mut urltype: [c_char; MAX_PREFIX_LEN] = [0; MAX_PREFIX_LEN];
    let mut infile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut outfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut extspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut rowfilter: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut binspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut colspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut imagecolname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut rowexpress: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let hdtype: [&CStr; 3] = [c"IMAGE", c"TABLE", c"BINTABLE"];

    if *status > 0 {
        return *status;
    }

    /* initialize null file pointer */
    let f_tmp = fptr.take();
    if let Some(f) = f_tmp {
        // WARNING: The c version doesn't null pointers after a close, so we have a dangling pointer.
        // We need to be careful with this, as it can cause double free errors.
        // Therefore, if this function is called with a Some(), then we will leak the pointer because
        // it's probably invalid.
        let _ = Box::into_raw(f);
    }

    if *NEED_TO_INITIALIZE.lock().unwrap() {
        /* this is called only once */
        *status = fits_init_cfitsio_safe();

        if *status > 0 {
            return *status;
        }
    }

    let mut url = 0;

    while name[url] == bb(b' ') {
        /* ignore leading spaces in the file spec */
        url += 1;
    }

    /* parse the input file specification */
    ffiurl_safe(
        &name[url..],
        Some(&mut urltype),
        Some(&mut infile),
        Some(&mut outfile),
        Some(&mut extspec),
        Some(&mut rowfilter),
        Some(&mut binspec),
        Some(&mut colspec),
        status,
    );

    strcpy_safe(&mut urltype, cs!(c"memkeep://")); /* URL type for pre-existing memory file */

    *status = urltype2driver(&urltype, &mut driver);

    if *status > 0 {
        ffpmsg_str("could not find driver for pre-existing memory file: (ffomem)");
        return *status;
    }

    /* call driver routine to open the memory file */
    let lock = FFLOCK(); /* lock this while searching for vacant handle */
    *status = mem_openmem(buffptr, buffsize, deltasize, Some(mem_realloc), &mut handle);
    FFUNLOCK(lock);

    if *status > 0 {
        ffpmsg_str("failed to open pre-existing memory file: (ffomem)");
        return *status;
    }

    /* get initial file size */
    //let d = driverTable.lock().unwrap();
    let d = DRIVER_TABLE.get().unwrap();
    *status = (d[driver as usize].size)(handle, &mut filesize);

    if *status > 0 {
        (d[driver as usize].close)(handle); /* close the file */
        ffpmsg_str("failed get the size of the memory file: (ffomem)");
        return *status;
    }

    let Fptr = FITSfile::new(
        &d[driver as usize],
        handle,
        &name[url..],
        cs!(c"ffomem"),
        status,
    );
    if Fptr.is_err() {
        return *status;
    }

    let mut Fptr = Fptr.unwrap();

    /* initialize the ageindex array (relative age of the I/O buffers) */
    /* and initialize the bufrecnum array as being empty */
    for ii in 0..(NIOBUF as usize) {
        Fptr.ageindex[ii] = ii as c_int;
        Fptr.bufrecnum[ii] = -1;
    }

    /* store the parameters describing the file */
    Fptr.MAXHDU = 1000; /* initial size of headstart */
    Fptr.filehandle = handle; /* file handle */
    Fptr.driver = driver; /* driver number */
    strcpy_safe(Fptr.get_filename_as_mut_slice(), &name[url..]); /* full input filename */
    Fptr.filesize = filesize as LONGLONG; /* physical file size */
    Fptr.logfilesize = filesize as LONGLONG; /* logical file size */
    Fptr.writemode = mode; /* read-write mode    */
    Fptr.datastart = DATA_UNDEFINED as LONGLONG; /* unknown start of data */
    Fptr.curbuf = -1; /* undefined current IO buffer */
    Fptr.open_count = 1; /* structure is currently used once */
    Fptr.validcode = VALIDSTRUC; /* flag denoting valid structure */
    Fptr.noextsyntax = 0; /* extended syntax can be used in filename */

    let f_fitsfile = box_try_new(fitsfile {
        HDUposition: 0,
        Fptr: FptrRef::new(Fptr),
    });

    if f_fitsfile.is_err() {
        let d = DRIVER_TABLE.get().unwrap();
        ((d[driver as usize]).close)(handle); /* close the file */
        ffpmsg_str("failed to allocate structure for following file: (ffomem)");
        ffpmsg_slice(&name[url..]);
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    let mut f_fitsfile = f_fitsfile.unwrap();

    ffldrc(&mut f_fitsfile, 0, REPORT_EOF, status); /* load first record */

    fits_store_Fptr(f_fitsfile.Fptr.as_ptr(), status); /* store Fptr address */

    if ffrhdu_safe(&mut f_fitsfile, Some(&mut hdutyp), status) > 0 {
        /* determine HDU structure */
        ffpmsg_str("ffomem could not interpret primary array header of file: (ffomem)");
        ffpmsg_slice(&name[url..]);

        if *status == UNKNOWN_REC {
            ffpmsg_str("This does not look like a FITS file.");
        }

        ffclos_safe(f_fitsfile, status);
        *fptr = None; /* return null file pointer */
        return *status;
    }

    *fptr = Some(f_fitsfile);

    /* ---------------------------------------------------------- */
    /* move to desired extension, if specified as part of the URL */
    /* ---------------------------------------------------------- */

    imagecolname[0] = 0;
    rowexpress[0] = 0;

    if extspec[0] != 0 {
        /* parse the extension specifier into individual parameters */
        ffexts_safe(
            &extspec,
            &mut extnum,
            &mut extname,
            &mut extvers,
            &mut movetotype,
            &mut imagecolname,
            &mut rowexpress,
            status,
        );

        if *status > 0 {
            return *status;
        }

        if extnum != 0 {
            ffmahd_safe(
                (*fptr).as_mut().unwrap(),
                extnum + 1,
                Some(&mut hdutyp),
                status,
            );
        } else if extname[0] != 0 {
            /* move to named extension, if specified */
            ffmnhd_safe(
                (*fptr).as_mut().unwrap(),
                movetotype,
                &extname,
                extvers,
                status,
            );
        }

        if *status > 0 {
            ffpmsg_str("ffomem could not move to the specified extension:");
            if extnum > 0 {
                int_snprintf!(
                    &mut errmsg,
                    FLEN_ERRMSG,
                    " extension number {} doesn't exist or couldn't be opened.",
                    extnum,
                );
                ffpmsg_slice(&errmsg);
            } else {
                int_snprintf!(
                    &mut errmsg,
                    FLEN_ERRMSG,
                    " extension with EXTNAME = {},",
                    slice_to_str!(&extname),
                );
                ffpmsg_slice(&errmsg);

                if extvers != 0 {
                    int_snprintf!(
                        &mut errmsg,
                        FLEN_ERRMSG,
                        "           and with EXTVERS = {},",
                        extvers,
                    );
                    ffpmsg_slice(&errmsg);
                }

                if movetotype != ANY_HDU {
                    int_snprintf!(
                        &mut errmsg,
                        FLEN_ERRMSG,
                        "           and with XTENSION = {},",
                        hdtype[movetotype as usize].to_str().unwrap(),
                    );
                    ffpmsg_slice(&errmsg);
                }
                ffpmsg_str(" doesn't exist or couldn't be opened.");
            }
            return *status;
        }
    }

    *status
}

/// Open an existing FITS file on magnetic disk with either readonly or
/// read/write access.  The routine does not support CFITSIO's extended
/// filename syntax and simply uses the entire input 'name' string as
/// the name of the file.
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffdkopn(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    mode: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        raw_to_slice!(name);

        ffdkopn_safe(fptr, name, mode, status)
    }
}

/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
pub fn ffdkopn_safe(
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    mode: c_int,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    *status = OPEN_DISK_FILE;

    ffopen_safe(fptr, name, mode, status);

    *status
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation. and
/// move to the first HDU that contains 'interesting' data, if the primary
/// array contains a null image (i.e., NAXIS = 0).
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffdopn(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    mode: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(name);

        ffdopn_safe(fptr, name, mode, status)
    }
}

/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
pub fn ffdopn_safe(
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    mode: c_int,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    *status = SKIP_NULL_PRIMARY;

    ffopen_safe(fptr, name, mode, status);

    *status
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation. and
/// if the primary array contains a null image (i.e., NAXIS = 0) then attempt to
/// move to the first extension named in the extlist of extension names. If
/// none are found, then simply move to the 2nd extension.
///
/// # Parameters
///
/// * `fptr`    — (O) FITS file pointer
/// * `name`    — (I) full name of file to open
/// * `mode`    — (I) 0 = open readonly; 1 = read/write
/// * `extlist` — (I) list of 'good' extensions to move to
/// * `hdutype` — (O) type of extension that is moved to
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffeopn(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    mode: c_int,
    extlist: *mut c_char,
    hdutype: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let hdutype = hdutype.as_mut();

        raw_to_slice!(name);

        let extlist_slice = if !extlist.is_null() {
            raw_to_slice!(extlist);
            Some(extlist)
        } else {
            None
        };

        ffeopn_safe(fptr, name, mode, extlist_slice, hdutype, status)
    }
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation. and
/// if the primary array contains a null image (i.e., NAXIS = 0) then attempt to
/// move to the first extension named in the extlist of extension names. If
/// none are found, then simply move to the 2nd extension.
///
/// # Parameters
///
/// * `fptr`    — (O) FITS file pointer
/// * `name`    — (I) full name of file to open
/// * `mode`    — (I) 0 = open readonly; 1 = read/write
/// * `extlist` — (I) list of 'good' extensions to move to
/// * `hdutype` — (O) type of extension that is moved to
/// * `status`  — (IO) error status
pub fn ffeopn_safe(
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    mode: c_int,
    extlist: Option<&[c_char]>,
    hdutype: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut hdunum: c_int = 0;
    let mut naxis: c_int = 0;
    let mut thdutype: c_int = 0;
    let mut gotext = false;

    if *status > 0 {
        return *status;
    }

    if ffopen_safe(fptr, name, mode, status) > 0 {
        return *status;
    }

    let f = (*fptr).as_mut().expect(NULL_MSG);

    ffghdn_safe(f, &mut hdunum);
    ffghdt_safe(f, &mut thdutype, status);

    if hdunum == 1 && thdutype == IMAGE_HDU {
        ffgidm_safe(f, &mut naxis, status);
    }

    /* We are in the "default" primary extension */
    /* look through the extension list */
    if (hdunum == 1) && (naxis == 0) {
        if let Some(extlist) = extlist {
            gotext = false;

            // HEAP ALLOCATION - Temporary
            let mut textlist = Vec::new();
            if textlist.try_reserve_exact(extlist.len()).is_err() {
                *status = MEMORY_ALLOCATION;
                return *status;
            } else {
                textlist.resize(extlist.len(), 0);
            }

            strcpy_safe(&mut textlist, extlist);
            for ext_sub in textlist.split(|c| *c == bb(b' ')) {
                ffmnhd_safe(f, ANY_HDU, ext_sub, 0, status);
                if *status == 0 {
                    gotext = true;
                    break;
                } else {
                    *status = 0;
                }
            }
        }
        if !gotext {
            /* if all else fails, move to extension #2 and hope for the best */
            ffmahd_safe(f, 2, Some(&mut thdutype), status);
        }
    }

    if let Some(hdutype) = hdutype {
        ffghdt_safe(f, hdutype, status);
    }

    *status
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation. and
/// move to the first HDU that contains 'interesting' table (not an image).
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fftopn(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    mode: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        raw_to_slice!(name);

        fftopn_safe(fptr, name, mode, status)
    }
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation. and
/// move to the first HDU that contains 'interesting' table (not an image).
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
pub fn fftopn_safe(
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    mode: c_int,
    status: &mut c_int,
) -> c_int {
    let mut hdutype: c_int = 0;

    if *status > 0 {
        return *status;
    }

    *status = SKIP_IMAGE;

    if ffopen_safe(fptr, name, mode, status) > 0 {
        /* the open failed and left *fptr as None; there is no HDU to inspect. */
        return *status;
    }

    let f = (*fptr).as_mut().expect(NULL_MSG);

    if ffghdt_safe(f, &mut hdutype, status) <= 0 && hdutype == IMAGE_HDU {
        *status = NOT_TABLE;
    }

    *status
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation. and
/// move to the first HDU that contains 'interesting' image (not an table).
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffiopn(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    mode: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        raw_to_slice!(name);

        ffiopn_safe(fptr, name, mode, status)
    }
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation. and
/// move to the first HDU that contains 'interesting' image (not an table).
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
pub fn ffiopn_safe(
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    mode: c_int,
    status: &mut c_int,
) -> c_int {
    let mut hdutype: c_int = 0;

    if *status > 0 {
        return *status;
    }

    *status = SKIP_TABLE;

    if ffopen_safe(fptr, name, mode, status) > 0 {
        /* the open failed and left *fptr as None; there is no HDU to inspect. */
        return *status;
    }

    let f = (*fptr).as_mut().expect(NULL_MSG);

    if ffghdt_safe(f, &mut hdutype, status) <= 0 && hdutype != IMAGE_HDU {
        *status = NOT_IMAGE;
    }

    *status
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation.
///
/// First test that the SONAME of fitsio.h used to build the CFITSIO library
/// is the same as was used in compiling the application program that
/// links to the library.
///
/// # Parameters
///
/// * `soname` — (I) CFITSIO shared library version
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffopentest(
    soname: c_int,
    /*     application program (fitsio.h file) */
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    mode: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        raw_to_slice!(name);

        ffopentest_safe(soname, fptr, name, mode, status)
    }
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation.
///
/// First test that the SONAME of fitsio.h used to build the CFITSIO library
/// is the same as was used in compiling the application program that
/// links to the library.
///
/// # Parameters
///
/// * `soname` — (I) CFITSIO shared library version application program (fitsio.h file)
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
pub fn ffopentest_safe(
    soname: c_int,
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    mode: c_int,
    status: &mut c_int,
) -> c_int {
    if soname != CFITSIO_SONAME as c_int {
        println!("\nERROR: Mismatch in the CFITSIO_SONAME value in the fitsio.h include file");
        println!("that was used to build the CFITSIO library, and the value in the include file");
        println!("that was used when compiling the application program:");
        println!("   Version used to build the CFITSIO library   = {CFITSIO_SONAME}");
        println!("   Version included by the application program = {soname}");
        print!("\nFix this by recompiling and then relinking this application program \n");
        println!("with the CFITSIO library.");

        *status = FILE_NOT_OPENED;
        return *status;
    }

    /* now call the normal file open routine */
    ffopen_safe(fptr, name, mode, status);
    *status
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation.
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffopen(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    mode: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(name);

        ffopen_safe(fptr, name, mode, status)
    }
}

/// Open an existing FITS file with either readonly or read/write access.
///
/// The name may use the extended filename syntax: a driver prefix, an HDU or
/// extension to move to, an image section, a row or column filter, and a
/// tile-compression specification. Applying a filter produces a temporary
/// in-memory file holding the filtered subset, so the returned handle may not
/// refer to the named file on disk. See the module documentation.
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) full name of file to open
/// * `mode`   — (I) 0 = open readonly; 1 = read/write
/// * `status` — (IO) error status
pub fn ffopen_safe(
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    mode: c_int,
    status: &mut c_int,
) -> c_int {
    let mut newptr: Option<Box<fitsfile>> = None;
    let mut driver = 0;
    let mut hdutyp = 0;
    let mut hdunum = 0;
    let slen;
    let mut isopen = 0;
    let mut filesize = 0;
    let mut rownum: c_long = 0;
    let mut nrows: c_long = 0;
    let mut goodrows: c_long = 0;
    let mut extnum = 0;
    let mut extvers = 0;
    let mut handle = 0;
    let mut movetotype = 0;
    let mut tstatus;
    let mut only_one = 0;
    let mut urltype: [c_char; MAX_PREFIX_LEN] = [0; MAX_PREFIX_LEN];
    let mut infile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut outfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut origurltype: [c_char; MAX_PREFIX_LEN] = [0; MAX_PREFIX_LEN];
    let mut extspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut extname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut rowfilter: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut tblname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut imagecolname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut rowexpress: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut binspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut colspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut pixfilter: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut histfilename: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut testpath: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut filtfilename: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut compspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut wtcol: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut minname: [[c_char; FLEN_VALUE]; 4] = [[0; FLEN_VALUE]; 4];
    let mut maxname: [[c_char; FLEN_VALUE]; 4] = [[0; FLEN_VALUE]; 4];
    let mut binname: [[c_char; FLEN_VALUE]; 4] = [[0; FLEN_VALUE]; 4];
    let mut minin: [f64; 4] = [0.0; 4];
    let mut maxin: [f64; 4] = [0.0; 4];
    let mut binsizein: [f64; 4] = [0.0; 4];
    let mut weight: f64 = 0.0;
    let mut imagetype = 0;
    let mut naxis = 1;
    let mut haxis = 0;
    let mut recip = 0;
    let mut skip_null = false;
    let mut skip_image = false;
    let mut skip_table = false;
    let mut no_primary_data = false;
    let mut open_disk_file = false;
    let mut colname: [[c_char; FLEN_VALUE]; 4] = [[0; FLEN_VALUE]; 4];
    let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let hdtype: [&CStr; 3] = [c"IMAGE", c"TABLE", c"BINTABLE"];

    let rowselect = None;

    if *status > 0 {
        return *status;
    }

    if *status == SKIP_NULL_PRIMARY {
        /* this special status value is used as a flag by ffdopn to tell */
        /* ffopen to skip over a null primary array when opening the file. */

        skip_null = true;
        *status = 0;
    } else if *status == SKIP_IMAGE {
        /* this special status value is used as a flag by fftopn to tell */
        /* ffopen to move to 1st significant table when opening the file. */

        skip_image = true;
        *status = 0;
    } else if *status == SKIP_TABLE {
        /* this special status value is used as a flag by ffiopn to tell */
        /* ffopen to move to 1st significant image when opening the file. */

        skip_table = true;
        *status = 0;
    } else if *status == OPEN_DISK_FILE {
        /* this special status value is used as a flag by ffdkopn to tell */
        /* ffopen to not interpret the input filename using CFITSIO's    */
        /* extended filename syntax, and simply open the specified disk file */

        open_disk_file = true;
        *status = 0;
    }

    /* initialize null file pointer */
    let f_tmp = fptr.take();
    if let Some(f) = f_tmp {
        // WARNING: The c version doesn't null pointers after a close, so we have a dangling pointer.
        // We need to be careful with this, as it can cause double free errors.
        // Therefore, if this function is called with a Some(), then we will leak the pointer because
        // it's probably invalid.
        let _ = Box::into_raw(f);
    }

    let mut writecopy = false; /* have we made a write-able copy of the input file? */

    if *NEED_TO_INITIALIZE.lock().unwrap() {
        /* this is called only once */
        *status = fits_init_cfitsio_safe();
    }

    if *status > 0 {
        return *status;
    }

    let url = name;
    let mut ui = 0;

    while url[ui] == bb(b' ') {
        /* ignore leading spaces in the filename */
        ui += 1;
    }

    if url[ui] == 0 {
        ffpmsg_str("Name of file to open is blank. (ffopen)");
        *status = FILE_NOT_OPENED;
        return *status;
    }

    let url = &url[ui..];

    if open_disk_file {
        /* treat the input URL literally as the name of the file to open */
        /* and don't try to parse the URL using the extended filename syntax */

        if strlen_safe(url) > FLEN_FILENAME - 1 {
            ffpmsg_str("Name of file to open is too long. (ffopen)");
            *status = FILE_NOT_OPENED;
            return *status;
        }

        strcpy_safe(&mut infile, url);
        strcpy_safe(&mut urltype, cs!(c"file://"));
        outfile[0] = 0;
        extspec[0] = 0;
        binspec[0] = 0;
        colspec[0] = 0;
        rowfilter[0] = 0;
        pixfilter[0] = 0;
        compspec[0] = 0;
    } else {
        /* parse the input file specification */

        /* NOTE: This routine tests that all the strings do not */
        /* overflow the standard buffer sizes (FLEN_FILENAME, etc.) */
        /* therefore in general we do not have to worry about buffer */
        /* overflow of any of the returned strings. */

        /* call the newer version of this parsing routine that supports 'compspec' */

        ffifile2_safe(
            url,
            Some(&mut urltype[..]),
            Some(&mut infile[..]),
            Some(&mut outfile[..]),
            Some(&mut extspec[..]),
            Some(&mut rowfilter[..]),
            Some(&mut binspec[..]),
            Some(&mut colspec[..]),
            Some(&mut pixfilter[..]),
            Some(&mut compspec[..]),
            status,
        );
        let tstEnv = std::env::var("CFITSIO_DISABLE_COPY_RESTRICT").ok();
        if !tstEnv.as_deref().is_some_and(|s| s.starts_with('1')) {
            if strlen_safe(&infile) != 0 && strlen_safe(&outfile) != 0 {
                let pathstart: &[c_char] = if strncmp_safe(&urltype, cs!(c"file"), 4) != 0 {
                    skip_host_string(&infile)
                } else {
                    &infile
                };
                strcpy_safe(&mut testpath, pathstart);
                if normalize_path(&mut testpath, status) != 0 {
                    ffpmsg_str("Unable to normalize input file path (ffopen)");
                    ffpmsg_slice(&testpath);
                    return *status;
                }
                if exclude_path(&testpath) != 0 {
                    ffpmsg_str("Attempting to access an invalid directory (ffopen)");
                    ffpmsg_slice(&testpath);
                    *status = FILE_NOT_OPENED;
                    return *status;
                }
            }
            if strncmp_safe(&urltype, cs!(c"rawfile"), 7) == 0
                && strncmp_safe(&outfile, cs!(c"root:"), 5) == 0
            {
                ffpmsg_str(
                    "The copying of a raw binary file to the root driver has been disabled.",
                );
                *status = FILE_NOT_OPENED;
                return *status;
            }
        }
    }

    if *status > 0 {
        ffpmsg_str("could not parse the input filename: (ffopen)");
        ffpmsg_slice(url);
        return *status;
    }

    imagecolname[0] = 0;
    rowexpress[0] = 0;

    if extspec[0] != 0 {
        slen = strlen_safe(&extspec);
        if extspec[slen - 1] == bb(b'#') {
            /* special symbol to mean only copy this extension */
            extspec[slen - 1] = 0;
            only_one = 1;
        }

        /* parse the extension specifier into individual parameters */
        ffexts_safe(
            &extspec,
            &mut extnum,
            &mut extname,
            &mut extvers,
            &mut movetotype,
            &mut imagecolname,
            &mut rowexpress,
            status,
        );
        if *status > 0 {
            return *status;
        };
    }

    /* special cases:                                                    */

    histfilename[0] = 0;
    filtfilename[0] = 0;

    if outfile[0] != 0 && (binspec[0] != 0 || imagecolname[0] != 0 || pixfilter[0] != 0) {
        /* if binspec or imagecolumn are specified, then the  */
        /* output file name is intended for the final image,  */
        /* and not a copy of the input file.                  */

        strcpy_safe(&mut histfilename, &outfile);
        outfile[0] = 0;
    } else if outfile[0] != 0 && (rowfilter[0] != 0 || colspec[0] != 0) {
        /* if rowfilter or colspece are specified, then the    */
        /* output file name is intended for the filtered file  */
        /* and not a copy of the input file.                   */

        strcpy_safe(&mut filtfilename, &outfile);
        outfile[0] = 0;
    }

    /* check if this same file is already open, and if so, attach to it  */

    let lock = FFLOCK();

    if fits_already_open(
        fptr,
        url,
        &mut urltype,
        &mut infile,
        &mut extspec,
        &mut rowfilter,
        &mut binspec,
        &mut colspec,
        mode,
        open_disk_file,
        &mut isopen,
        status,
    ) > 0
    {
        FFUNLOCK(lock);
        return *status;
    }

    FFUNLOCK(lock);

    let mut move2hdu = false;
    if isopen != 0 {
        move2hdu = true;
    }

    if !move2hdu {
        /* get the driver number corresponding to this urltype */
        *status = urltype2driver(&urltype, &mut driver);

        if *status > 0 {
            ffpmsg_str("could not find driver for this file: (ffopen)");
            ffpmsg_slice(&urltype);
            ffpmsg_slice(url);
            return *status;
        }

        /*-------------------------------------------------------------------
          deal with all those messy special cases which may require that
          a different driver be used:
              - is disk file compressed?
              - are ftp:, gsiftp:, or http: files compressed?
              - has user requested that a local copy be made of
                the ftp or http file?
        -------------------------------------------------------------------*/

        let d = DRIVER_TABLE.get().unwrap();

        if let Some(checkfile) = (d[driver as usize]).checkfile {
            strcpy_safe(&mut origurltype, &urltype); /* Save the urltype */

            /* 'checkfile' may modify the urltype, infile and outfile strings */
            *status = checkfile(&mut urltype, &mut infile, &mut outfile);

            if *status != 0 {
                ffpmsg_str("checkfile failed for this file: (ffopen)");
                ffpmsg_slice(url);
                return *status;
            }

            if strcmp_safe(&origurltype, &urltype) != 0 {
                /* did driver changed on us? */
                *status = urltype2driver(&urltype, &mut driver);
                if *status > 0 {
                    ffpmsg_str("could not change driver for this file: (ffopen)");
                    ffpmsg_slice(url);
                    ffpmsg_slice(&urltype);
                    return *status;
                };
            };
        }

        /* call appropriate driver to open the file */
        match (d[driver as usize]).open {
            Some(open) => {
                let lock = FFLOCK(); /* lock this while searching for vacant handle */
                *status = open(&mut infile, mode, &mut handle);
                FFUNLOCK(lock);

                if *status > 0 {
                    ffpmsg_str("failed to find or open the following file: (ffopen)");
                    ffpmsg_slice(url);
                    return *status;
                };
            }
            None => {
                ffpmsg_str("cannot open an existing file of this type: (ffopen)");
                ffpmsg_slice(url);
                *status = FILE_NOT_OPENED;
                return *status;
            }
        };

        /* get initial file size */
        *status = ((d[driver as usize]).size)(handle, &mut filesize);

        if *status > 0 {
            ((d[driver as usize]).close)(handle); /* close the file */
            ffpmsg_str("failed get the size of the following file: (ffopen)");
            ffpmsg_slice(url);
            return *status;
        }

        let Fptr = FITSfile::new(&d[driver as usize], handle, url, cs!(c"ffopen"), status);
        if Fptr.is_err() {
            return *status;
        }
        let mut Fptr = Fptr.unwrap();

        /* initialize the ageindex array (relative age of the I/O buffers) */
        /* and initialize the bufrecnum array as being empty */
        for ii in 0..NIOBUF as usize {
            Fptr.ageindex[ii] = ii as c_int;
            Fptr.bufrecnum[ii] = -1;
        }

        /* store the parameters describing the file */
        Fptr.MAXHDU = 1000; /* initial size of headstart */
        Fptr.filehandle = handle; /* file handle */
        Fptr.driver = driver; /* driver number */
        strcpy_safe(Fptr.get_filename_as_mut_slice(), url); /* full input filename */
        Fptr.filesize = filesize as LONGLONG; /* physical file size */
        Fptr.logfilesize = filesize as LONGLONG; /* logical file size */
        Fptr.writemode = mode; /* read-write mode    */
        Fptr.datastart = DATA_UNDEFINED as LONGLONG; /* unknown start of data */
        Fptr.curbuf = -1; /* undefined current IO buffer */
        Fptr.open_count = 1; /* structure is currently used once */
        Fptr.validcode = VALIDSTRUC; /* flag denoting valid structure */
        Fptr.only_one = only_one; /* flag denoting only copy single extension */
        Fptr.noextsyntax = if open_disk_file { 1 } else { 0 }; /* true if extended syntax is disabled */

        // HEAP ALLOCATION
        /* allocate fitsfile structure and initialize = 0 */
        let f_fitsfile = box_try_new(fitsfile {
            HDUposition: 0,
            Fptr: FptrRef::new(Fptr),
        });

        if f_fitsfile.is_err() {
            let d = DRIVER_TABLE.get().unwrap();
            ((d[driver as usize]).close)(handle); /* close the file */
            ffpmsg_str("failed to allocate structure for following file: (ffopen)");
            ffpmsg_slice(url);
            *status = MEMORY_ALLOCATION;
            return *status;
        }

        let mut f_fitsfile = f_fitsfile.unwrap();

        //drop(d); // To break mutex guard

        ffldrc(&mut f_fitsfile, 0, REPORT_EOF, status); /* load first record */

        fits_store_Fptr(f_fitsfile.Fptr.as_ptr(), status); /* store Fptr address */

        if ffrhdu_safe(&mut f_fitsfile, Some(&mut hdutyp), status) > 0 {
            /* determine HDU structure */
            ffpmsg_str("ffopen could not interpret primary array header of file: ");
            ffpmsg_slice(url);

            if *status == UNKNOWN_REC {
                ffpmsg_str("This does not look like a FITS file.");
            }

            ffclos_safe(f_fitsfile, status);
            *fptr = None; /* return null file pointer */
            return *status;
        }

        *fptr = Some(f_fitsfile);

        /* ------------------------------------------------------------- */
        /* At this point, the input file has been opened. If outfile was */
        /* specified, then we have opened a copy of the file, not the    */
        /* original file so it is safe to modify it if necessary         */
        /* ------------------------------------------------------------- */

        if outfile[0] != 0 {
            writecopy = true;
        }
    }

    if extspec[0] != 0 {
        let f = (*fptr).as_mut().unwrap();

        if extnum != 0 {
            /* extension number was specified */

            ffmahd_safe(f, extnum + 1, Some(&mut hdutyp), status);
        } else if extname[0] != 0 {
            /* move to named extension, if specified */
            ffmnhd_safe(f, movetotype, &extname, extvers, status);
        }

        if *status > 0 {
            /* clean up after error */
            ffpmsg_str("ffopen could not move to the specified extension:");

            if extnum > 0 {
                int_snprintf!(
                    &mut errmsg,
                    FLEN_ERRMSG,
                    " extension number {} doesn't exist or couldn't be opened.",
                    extnum,
                );
                ffpmsg_slice(&errmsg);
            } else {
                int_snprintf!(
                    &mut errmsg,
                    FLEN_ERRMSG,
                    " extension with EXTNAME = {},",
                    slice_to_str!(&extname),
                );
                ffpmsg_slice(&errmsg);

                if extvers != 0 {
                    int_snprintf!(
                        &mut errmsg,
                        FLEN_ERRMSG,
                        "           and with EXTVERS = {},",
                        extvers,
                    );
                    ffpmsg_slice(&errmsg);
                }

                if movetotype != ANY_HDU {
                    int_snprintf!(
                        &mut errmsg,
                        FLEN_ERRMSG,
                        "           and with XTENSION = {},",
                        hdtype[movetotype as usize].to_str().unwrap(),
                    );
                    ffpmsg_slice(&errmsg);
                }
                ffpmsg_str(" doesn't exist or couldn't be opened.");
            }

            let f_tmp = fptr.take().unwrap(); // Nulls pointer
            ffclos_safe(f_tmp, status);
            return *status;
        };
    } else if skip_null
        || skip_image
        || skip_table
        || (imagecolname[0] != 0 || colspec[0] != 0 || rowfilter[0] != 0 || binspec[0] != 0)
    {
        /* ------------------------------------------------------------------

        If no explicit extension specifier is given as part of the file
        name, and, if a) skip_null is true (set if ffopen is called by
        ffdopn) or b) skip_image or skip_table is true (set if ffopen is
        called by fftopn or ffdopn) or c) other file filters are
        specified, then CFITSIO will attempt to move to the first
        'interesting' HDU after opening an existing FITS file (or to
        first interesting table HDU if skip_image is true);

        An 'interesting' HDU is defined to be either an image with NAXIS
        > 0 (i.e., not a null array) or a table which has an EXTNAME
        value which does not contain any of the following strings:
           'GTI'  - Good Time Interval extension
           'OBSTABLE'  - used in Beppo SAX data files

        The main purpose for this is to allow CFITSIO to skip over a null
        primary and other non-interesting HDUs when opening an existing
        file, and move directly to the first extension that contains
        significant data.
        ------------------------------------------------------------------ */

        let f = (*fptr).as_mut().unwrap();

        ffghdn_safe(f, &mut hdunum);
        if hdunum == 1 {
            ffgidm_safe(f, &mut naxis, status);

            if naxis != 0 {
                let mut naxes = vec![0; naxis as usize];
                ffgisz_safe(f, naxis, &mut naxes, status);

                for ii in 0..naxis as usize {
                    if naxes[ii] == 0 {
                        if ii == 0 {
                            /* NAXIS1=0 could be a random group indicator */
                            tstatus = 0;
                            ffmaky_safe(f, 2, status);

                            let mut group_val = 0;
                            if ffgkyl_safe(f, cs!(c"GROUPS"), &mut group_val, None, &mut tstatus)
                                != 0
                            {
                                no_primary_data = true; /* GROUPS keyword not found */
                            }
                        } else {
                            no_primary_data = true;
                        }
                    }
                }
            } else {
                no_primary_data = true;
            }

            if no_primary_data || skip_image {
                /* skip primary array */

                loop {
                    /* see if the next HDU is 'interesting' */
                    if ffmrhd_safe(f, 1, Some(&mut hdutyp), status) != 0 {
                        if *status == END_OF_FILE {
                            *status = 0; /* reset expected error */
                        }

                        /* didn't find an interesting HDU so move back to beginning */
                        ffmahd_safe(f, 1, Some(&mut hdutyp), status);
                        break;
                    }

                    if hdutyp == IMAGE_HDU && skip_image {
                        continue; /* skip images */
                    } else if hdutyp != IMAGE_HDU && skip_table {
                        continue; /* skip tables */
                    } else if hdutyp == IMAGE_HDU {
                        ffgidm_safe(f, &mut naxis, status);
                        if naxis > 0 {
                            break; /* found a non-null image */
                        };
                    } else {
                        tstatus = 0;
                        tblname[0] = 0;
                        fits_read_key_str(f, cs!(c"EXTNAME"), &mut tblname, None, &mut tstatus);

                        if (strstr_safe(&tblname, cs!(c"GTI")).is_none()
                            && strstr_safe(&tblname, cs!(c"gti")).is_none())
                            && fits_strncasecmp(&tblname, cs!(c"OBSTABLE"), 8) != 0
                        {
                            break; /* found an interesting table */
                        };
                    };
                } /* end loop */
            }
        } /* end if (hdunum==1) */
    }

    if imagecolname[0] != 0 {
        /* ----------------------------------------------------------------- */
        /* we need to open an image contained in a single table cell         */
        /* First, determine which row of the table to use.                   */
        /* ----------------------------------------------------------------- */
        let f = (*fptr).as_mut().unwrap();

        if !isdigit_safe(rowexpress[0]) {
            /* is the row specification a number? */
            sscanf_ld(&rowexpress, cs!(c"%ld"), &raw mut rownum);
            if rownum < 1 {
                ffpmsg_str("illegal rownum for image cell:");
                ffpmsg_slice(&rowexpress);
                ffpmsg_str("Could not open the following image in a table cell:");
                ffpmsg_slice(&extspec);

                let f_tmp = fptr.take().unwrap(); // Nulls fptr
                ffclos_safe(f_tmp, status);
                *status = BAD_ROW_NUM;
                return *status;
            };
        } else if ffffrw_safe(f, &rowexpress, &mut rownum, status) > 0 {
            ffpmsg_str("Failed to find row matching this expression:");
            ffpmsg_slice(&rowexpress);
            ffpmsg_str("Could not open the following image in a table cell:");
            ffpmsg_slice(&extspec);
            let f_tmp = fptr.take().unwrap(); // Nulls fptr
            ffclos_safe(f_tmp, status);
            return *status;
        }
        if rownum == 0 {
            ffpmsg_str("row satisfying this expression doesn\'t exist::");
            ffpmsg_slice(&rowexpress);
            ffpmsg_str("Could not open the following image in a table cell:");
            ffpmsg_slice(&extspec);
            let f_tmp = fptr.take().unwrap(); // Nulls fptr
            ffclos_safe(f_tmp, status);
            *status = BAD_ROW_NUM;
            return *status;
        }

        /* determine the name of the new file to contain copy of the image */
        if histfilename[0] != 0 && (pixfilter[0]) == 0 {
            strcpy_safe(&mut outfile, &histfilename); /* the original outfile name */
        } else {
            strcpy_safe(&mut outfile, cs!(c"mem://_1")); /* create image file in memory */
        }

        /* Copy the image into new primary array and open it as the current */
        /* fptr.  This will close the table that contains the original image. */

        /* create new empty file to hold copy of the image */
        if ffinit_safe(&mut newptr, &outfile, status) > 0 {
            ffpmsg_str("failed to create file for copy of image in table cell:");
            ffpmsg_slice(&outfile);
            return *status;
        }

        if fits_copy_cell2image_safe(f, newptr.as_mut().unwrap(), &imagecolname, rownum, status) > 0
        {
            ffpmsg_str("Failed to copy table cell to new primary array:");
            ffpmsg_slice(&extspec);
            let f_tmp = fptr.take().unwrap(); // Nulls fptr
            ffclos_safe(f_tmp, status);
            return *status;
        }

        /* close the original file and set fptr to the new image */
        let f_tmp = fptr.take().unwrap(); // Nulls fptr
        ffclos_safe(f_tmp, status);

        *fptr = newptr; /* reset the pointer to the new table */

        writecopy = true; /* we are now dealing with a copy of the original file */

        /*  leave it up to calling routine to write any HISTORY keywords */
    }

    /* --------------------------------------------------------------------- */
    /* edit columns (and/or keywords) in the table, if specified in the URL  */
    /* --------------------------------------------------------------------- */

    if colspec[0] != 0 {
        /* the column specifier will modify the file, so make sure */
        /* we are already dealing with a copy, or else make a new copy */

        if !writecopy {
            /* Is the current file already a copy? */
            writecopy = fits_is_this_a_copy(urltype) != 0;
        }

        if !writecopy {
            if filtfilename[0] != 0 && outfile[0] == 0 {
                strcpy_safe(&mut outfile, &filtfilename); /* the original outfile name */
            } else {
                strcpy_safe(&mut outfile, cs!(c"mem://_1")); /* will create copy in memory */
            }

            writecopy = true;
        } else {
            let f = (*fptr).as_mut().unwrap();

            f.Fptr.writemode = READWRITE; /* we have write access */
            outfile[0] = 0;
        }

        if ffedit_columns(fptr, &outfile, &mut colspec, status) > 0 {
            ffpmsg_str("editing columns in input table failed (ffopen)");
            ffpmsg_str(" while trying to perform the following operation:");
            ffpmsg_slice(&colspec);
            let f_tmp = fptr.take().unwrap(); // Nulls fptr
            ffclos_safe(f_tmp, status);
            return *status;
        };
    }

    /* ------------------------------------------------------------------- */
    /* select rows from the table, if specified in the URL                 */
    /* or select a subimage (if this is an image HDU and not a table)      */
    /* ------------------------------------------------------------------- */

    if rowfilter[0] != 0 {
        let f = (*fptr).as_mut().unwrap();

        ffghdt_safe(f, &mut hdutyp, status); /* get type of HDU */
        if hdutyp == IMAGE_HDU {
            /* this is an image so 'rowfilter' is an image section specification */

            if filtfilename[0] != 0 && outfile[0] == 0 {
                strcpy_safe(&mut outfile, &filtfilename); /* the original outfile name */
            } else if outfile[0] == 0 {
                /* output file name not already defined? */
                strcpy_safe(&mut outfile, cs!(c"mem://_2")); /* will create file in memory */
            }

            /* create new file containing the image section, plus a copy of */
            /* any other HDUs that exist in the input file.  This routine   */
            /* will close the original image file and return a pointer      */
            /* to the new file. */

            if fits_select_image_section_safe(fptr, &outfile, &rowfilter, status) > 0 {
                ffpmsg_str("on-the-fly selection of image section failed (ffopen)");
                ffpmsg_str(" while trying to use the following section filter:");
                ffpmsg_slice(&rowfilter);

                let f_tmp = fptr.take().unwrap(); // Nulls pointer
                ffclos_safe(f_tmp, status);
                return *status;
            };
        } else {
            /* this is a table HDU, so the rowfilter is really a row filter */

            if binspec[0] != 0 {
                /*  since we are going to make a histogram of the selected rows,   */
                /*  it would be a waste of time and memory to make a whole copy of */
                /*  the selected rows.  Instead, just construct an array of TRUE   */
                /*  or FALSE values that indicate which rows are to be included    */
                /*  in the histogram and pass that to the histogram generating     */
                /*  routine                                                        */

                ffgnrw_safe(f, &mut nrows, status); /* get no. of rows */

                // HEAP ALLOCATION - Temporary
                let mut rowselect = Vec::new();
                if rowselect.try_reserve_exact(nrows as usize).is_err() {
                    ffpmsg_str("failed to allocate memory for selected columns array (ffopen)");
                    ffpmsg_str(" while trying to select rows with the following filter:");
                    ffpmsg_slice(&rowfilter);

                    let f_tmp = fptr.take().unwrap(); // Nulls pointer
                    ffclos_safe(f_tmp, status);

                    *status = MEMORY_ALLOCATION;
                    return *status;
                } else {
                    rowselect.resize(nrows as usize, 0);
                }

                if fffrow_safe(
                    f,
                    &rowfilter,
                    1,
                    nrows,
                    &mut goodrows,
                    &mut rowselect,
                    status,
                ) > 0
                {
                    ffpmsg_str("selection of rows in input table failed (ffopen)");
                    ffpmsg_str(" while trying to select rows with the following filter:");
                    ffpmsg_slice(&rowfilter);

                    let f_tmp = fptr.take().unwrap(); // Nulls pointer
                    ffclos_safe(f_tmp, status);

                    return *status;
                };
            } else {
                if !writecopy {
                    /* Is the current file already a copy? */
                    writecopy = fits_is_this_a_copy(urltype) != 0;
                }

                if !writecopy {
                    if filtfilename[0] != 0 && outfile[0] == 0 {
                        strcpy_safe(&mut outfile, &filtfilename); /* the original outfile name */
                    } else if outfile[0] == 0 {
                        /* output filename not already defined? */
                        strcpy_safe(&mut outfile, cs!(c"mem://_2")); /* will create copy in memory */
                    };
                } else {
                    f.Fptr.writemode = READWRITE; /* we have write access */
                    outfile[0] = 0;
                }

                /* select rows in the table.  If a copy of the input file has */
                /* not already been made, then this routine will make a copy */
                /* and then close the input file, so that the modifications will */
                /* only be made on the copy, not the original */

                if ffselect_table(fptr, &outfile, &rowfilter, status) > 0 {
                    ffpmsg_str("on-the-fly selection of rows in input table failed (ffopen)");
                    ffpmsg_str(" while trying to select rows with the following filter:");
                    ffpmsg_slice(&rowfilter);
                    let f_tmp = fptr.take().unwrap(); // Nulls pointer
                    ffclos_safe(f_tmp, status);
                    return *status;
                }

                let f = (*fptr).as_mut().unwrap();

                /* write history records */
                ffphis_safe(
                    f,
                    cs!(c"CFITSIO used the following filtering expression to create this table:"),
                    status,
                );
                ffphis_safe(f, name, status);
            } /* end of no binspec case */
        } /* end of table HDU case */
    } /* end of rowfilter exists case */

    /* ------------------------------------------------------------------- */
    /* make an image histogram by binning columns, if specified in the URL */
    /* ------------------------------------------------------------------- */

    if binspec[0] != 0 {
        let mut exprs: [Box<[c_char]>; 5] = Default::default(); // This is a valid pointer to an empty value, i.e. is expecting a return value
        if histfilename[0] != 0 && (pixfilter[0]) == 0 {
            strcpy_safe(&mut outfile, &histfilename); /* the original outfile name */
        } else {
            strcpy_safe(&mut outfile, cs!(c"mem://_3")); /* create histogram in memory */
            /* if not already copied the file */
        }

        /* parse the binning specifier into individual parameters */
        ffbinse(
            &binspec,
            &mut imagetype,
            &mut haxis,
            &mut colname,
            &mut minin,
            &mut maxin,
            &mut binsizein,
            &mut minname,
            &mut maxname,
            &mut binname,
            &mut weight,
            &mut wtcol,
            &mut recip,
            Some(&mut exprs),
            status,
        );

        /* Create the histogram primary array and open it as the current fptr */
        /* This will close the table that was used to create the histogram. */

        // Check if any expressions exist
        let has_exprs = exprs.iter().any(|e| !e.is_empty() && e[0] != 0);

        if has_exprs {
            ffhist2e(
                fptr,
                &outfile,
                imagetype,
                haxis,
                &colname,
                Some(&[
                    Some(&exprs[0]),
                    Some(&exprs[1]),
                    Some(&exprs[2]),
                    Some(&exprs[3]),
                ]),
                &minin,
                &maxin,
                &binsizein,
                &minname,
                &maxname,
                &binname,
                weight,
                &wtcol,
                Some(&exprs[4]),
                recip,
                rowselect,
                status,
            );
        } else {
            ffhist2e(
                fptr, &outfile, imagetype, haxis, &colname, None, &minin, &maxin, &binsizein,
                &minname, &maxname, &binname, weight, &wtcol, None, recip, rowselect, status,
            );
        }

        let f = (*fptr).as_mut().unwrap();
        if *status > 0 {
            ffpmsg_str("on-the-fly histogramming of input table failed (ffopen)");
            ffpmsg_str(" while trying to execute the following histogram specification:");
            ffpmsg_slice(&binspec);

            let f_tmp = fptr.take().unwrap(); // Nulls pointer
            ffclos_safe(f_tmp, status);

            return *status;
        }

        /* write history records */
        ffphis_safe(
            f,
            cs!(c"CFITSIO used the following expression to create this histogram:"),
            status,
        );

        ffphis_safe(f, name, status);
    }

    if pixfilter[0] != 0 {
        let f = (*fptr).as_mut().unwrap();
        if histfilename[0] != 0 {
            strcpy_safe(&mut outfile, &histfilename); /* the original outfile name */
        } else {
            strcpy_safe(&mut outfile, cs!(c"mem://_4")); /* create in memory */
            /* if not already copied the file */
        }

        /* Ensure type of HDU is consistent with pixel filtering */
        ffghdt_safe(f, &mut hdutyp, status);

        if hdutyp == IMAGE_HDU {
            pixel_filter_helper(fptr, outfile, pixfilter, status);

            if *status > 0 {
                ffpmsg_str("pixel filtering of input image failed (ffopen)");
                ffpmsg_str(" while trying to execute the following:");
                ffpmsg_slice(&pixfilter);

                let f_tmp = fptr.take().unwrap(); // Nulls pointer
                ffclos_safe(f_tmp, status);

                return *status;
            }

            let f = (*fptr).as_mut().unwrap();

            /* write history records */
            ffphis_safe(
                f,
                cs!(c"CFITSIO used the following expression to create this image:"),
                status,
            );

            ffphis_safe(f, name, status);
        } else {
            let _f = (*fptr).as_mut().unwrap();
            ffpmsg_str("cannot use pixel filter on non-IMAGE HDU");
            ffpmsg_slice(&pixfilter);

            let f_tmp = fptr.take().unwrap(); // Nulls pointer
            ffclos_safe(f_tmp, status);
            *status = NOT_IMAGE;
            return *status;
        };
    }

    /* parse and save image compression specification, if given */
    if compspec[0] != 0 {
        let f = (*fptr).as_mut().unwrap();
        ffparsecompspec(f, &compspec, status);
    }
    *status
}

/// Reopen an existing FITS file with either readonly or read/write access.
///
/// The reopened file shares the same FITSfile structure but may point to a
/// different HDU within the file.
/// SAFETY: This is in no ways safe, multiple fptrs sharing the same underlying data
///
/// # Parameters
///
/// * `openfptr` — (I) FITS file pointer to open file
/// * `newfptr`  — (O) pointer to new re opened file
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffreopen(
    openfptr: *mut fitsfile,
    newfptr: *mut *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let newfptr = newfptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        /* check that the open file pointer is valid */
        if openfptr.is_null() {
            *status = NULL_INPUT_PTR;
            return *status;
        }

        let openfptr = openfptr.as_mut().expect(NULL_MSG);

        ffreopen_safe(openfptr, newfptr, status)
    }
}

/// # Parameters
///
/// * `openfptr` — (I) FITS file pointer to open file
/// * `newfptr`  — (O) pointer to new re opened file
/// * `status`   — (IO) error status
pub fn ffreopen_safe(
    openfptr: &mut fitsfile,
    newfptr: &mut *mut fitsfile,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    if openfptr.Fptr.validcode != VALIDSTRUC {
        /* check magic value */
        *status = BAD_FILEPTR;
        return *status;
    }

    /* allocate fitsfile structure and initialize = 0 */
    // HEAP ALLOCATION
    let mut n = Box::new(fitsfile {
        HDUposition: 0,              /* set initial position to primary array */
        Fptr: openfptr.Fptr.share(), /* both point to the same structure */
    });

    n.Fptr.open_count += 1; /* increment the file usage counter */

    *newfptr = Box::into_raw(n);

    *status
}

/// store the new Fptr address for future use by fits_already_open
/// Takes the address rather than a `&mut FITSfile`: the table has to hold the
/// same pointer the handles deref through, not a reference-derived child of it,
/// or the entry is invalidated by the next write through any handle.
///
/// # Parameters
///
/// * `Fptr`   — (O) FITS file pointer
/// * `status` — (IO) error status
pub(crate) fn fits_store_Fptr(Fptr: *mut FITSfile, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status;
    }

    let lock = FFLOCK();
    for ii in 0..NMAXFILES {
        unsafe {
            if FPTR_TABLE[ii].is_null() {
                FPTR_TABLE[ii] = Fptr;
                break;
            };
        }
    }
    FFUNLOCK(lock);

    *status
}

///  clear the Fptr address from the Fptr Table  
///
/// # Parameters
///
/// * `Fptr`   — (O) FITS file pointer
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_clear_Fptr(Fptr: *mut FITSfile, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let lock = FFLOCK();
        for ii in 0..NMAXFILES {
            if core::ptr::eq(FPTR_TABLE[ii], Fptr) {
                FPTR_TABLE[ii] = ptr::null_mut();
                break;
            };
        }
        FFUNLOCK(lock);
        *status
    }
}

///  clear the Fptr address from the Fptr Table  
///
/// # Parameters
///
/// * `Fptr`   — (O) FITS file pointer
/// * `status` — (IO) error status
fn fits_clear_Fptr_safer(Fptr: *mut FITSfile, status: &mut c_int) -> c_int {
    let lock = FFLOCK();
    for ii in 0..NMAXFILES {
        unsafe {
            if FPTR_TABLE[ii] == Fptr {
                FPTR_TABLE[ii] = ptr::null_mut();
                break;
            };
        }
    }
    FFUNLOCK(lock);
    *status
}

/// Check if the file to be opened is already open.  If so, then attach to it.
///
/// the input strings must not exceed the standard lengths
/// of FLEN_FILENAME, MAX_PREFIX_LEN, etc.
///
///      
/// this function was changed so that for files of access method FILE://
/// the file paths are compared using standard URL syntax and absolute
/// paths (as opposed to relative paths). This eliminates some instances
/// where a file is already opened but it is not realized because it
/// was opened with another file path. For instance, if the CWD is
/// /a/b/c and I open /a/b/c/foo.fits then open ./foo.fits the previous
/// version of this function would not have reconized that the two files
/// were the same. This version does recognize that the two files are
/// the same.
///
/// # Parameters
///
/// * `fptr`     — I/O - FITS file pointer
/// * `mode`     — (I) 0 = open readonly; 1 = read/write
/// * `noextsyn` — (I) 0 = ext syntax may be used; 1 = ext syntax disabled
/// * `isopen`   — (O) 1 = file is already open
/// * `status`   — (IO) error status
pub(crate) fn fits_already_open(
    fptr: &mut Option<Box<fitsfile>>,
    url: &[c_char],
    urltype: &mut [c_char],
    infile: &mut [c_char],
    extspec: &mut [c_char],
    rowfilter: &mut [c_char],
    binspec: &mut [c_char],
    colspec: &mut [c_char],
    mode: c_int,
    noextsyn: bool,
    isopen: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut iMatch = None;
    let mut oldurltype: [c_char; MAX_PREFIX_LEN] = [0; MAX_PREFIX_LEN];
    let mut oldinfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut oldextspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut oldoutfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut oldrowfilter: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut oldbinspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut oldcolspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut tmpinfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    *isopen = 0;

    /*  When opening a file with readonly access then we simply let
        the operating system open the file again, instead of using the CFITSIO
        trick of attaching to the previously opened file.  This is required
        if CFITSIO is running in a multi-threaded environment, because 2 different
        threads cannot share the same FITSfile pointer.

        If the file is opened/reopened with write access, then the file MUST
        only be physically opened once..
    */

    if mode == 0 {
        return *status;
    }

    strcpy_safe(&mut tmpinfile, infile);
    if fits_strcasecmp(urltype, cs!(c"FILE://")) == 0
        && standardize_path(&mut tmpinfile, status) != 0
    {
        return *status;
    };
    for ii in 0..NMAXFILES {
        /* check every buffer */

        /* SAFETY: FPTR_TABLE is a `static mut`; a non-null entry always points
        at a live FITSfile owned by an open handle.  Taking a shared reference
        rather than `&mut` keeps this from aliasing the live FptrRef handles --
        every use below is a read. */
        if let Some(oldFptr) = unsafe { FPTR_TABLE[ii].as_ref() } {
            if oldFptr.noextsyntax != 0 {
                /* old urltype must be "file://" */
                if fits_strcasecmp(urltype, cs!(c"FILE://")) == 0 {
                    /* compare tmpinfile to adjusted oldFptr->filename */

                    let ofn = cast_slice(oldFptr.get_filename_as_cstr().to_bytes_with_nul());

                    /* This shouldn't be possible, but check anyway */
                    if strlen_safe(ofn) > FLEN_FILENAME - 1 {
                        ffpmsg_str("Name of old file is too long. (fits_already_open)");
                        *status = FILE_NOT_OPENED;
                        return *status;
                    }

                    let ofn = cast_slice(oldFptr.get_filename_as_cstr().to_bytes_with_nul());
                    strcpy_safe(&mut oldinfile, ofn);
                    if standardize_path(&mut oldinfile, status) != 0 {
                        return *status;
                    }

                    if strcmp_safe(&tmpinfile, &oldinfile) == 0 {
                        /* if infile is not noextsyn, must check that it is not
                        using filters of any kind */
                        if noextsyn || (rowfilter[0] == 0 && binspec[0] == 0 && colspec[0] == 0) {
                            if mode == READWRITE as c_int && oldFptr.writemode == READONLY as c_int
                            {
                                /*
                                  cannot assume that a file previously opened with READONLY
                                  can now be written to (e.g., files on CDROM, or over the
                                  the network, or STDIN), so return with an error.
                                */

                                ffpmsg_str(
                                    "cannot reopen file READWRITE when previously opened READONLY",
                                );
                                ffpmsg_slice(url);
                                *status = FILE_NOT_OPENED;
                                return *status;
                            }
                            iMatch = Some(ii);
                        };
                    };
                };

                /* end if old file has disabled extended syntax */
            } else {
                let filename = cast_slice(oldFptr.get_filename_as_cstr().to_bytes_with_nul());

                ffiurl_safe(
                    filename,
                    Some(&mut oldurltype),
                    Some(&mut oldinfile),
                    Some(&mut oldoutfile),
                    Some(&mut oldextspec),
                    Some(&mut oldrowfilter),
                    Some(&mut oldbinspec),
                    Some(&mut oldcolspec),
                    status,
                );

                if *status > 0 {
                    ffpmsg_str("could not parse the previously opened filename: (ffopen)");
                    ffpmsg_cstr(oldFptr.get_filename_as_cstr());
                    return *status;
                }

                if fits_strcasecmp(&oldurltype, cs!(c"FILE://")) == 0
                    && standardize_path(&mut oldinfile, status) != 0
                {
                    return *status;
                };

                if strcmp_safe(urltype, &oldurltype) == 0
                    && strcmp_safe(&tmpinfile, &oldinfile) == 0
                {
                    /* identical type of file and root file name */

                    if (rowfilter[0] == 0
                    && oldrowfilter[0] == 0
                    && binspec[0] == 0
                    && oldbinspec[0] == 0
                    && colspec[0] == 0
                    && oldcolspec[0] == 0)

                 /* no filtering or binning specs for either file, so */
                 /* this is a case where the same file is being reopened. */
                 /* It doesn't matter if the extensions are different */


                    || (strcmp_safe(rowfilter, &oldrowfilter) == 0
                        && strcmp_safe(binspec, &oldbinspec) == 0
                        && strcmp_safe(colspec, &oldcolspec) == 0
                        && strcmp_safe(extspec, &oldextspec) == 0)
                    /* filtering specs are given and are identical, and */
                    /* the same extension is specified */
                    {
                        if mode == READWRITE as c_int && oldFptr.writemode == READONLY as c_int {
                            /*
                              cannot assume that a file previously opened with READONLY
                              can now be written to (e.g., files on CDROM, or over the
                              the network, or STDIN), so return with an error.
                            */

                            ffpmsg_str(
                                "cannot reopen file READWRITE when previously opened READONLY",
                            );
                            ffpmsg_slice(url);
                            *status = FILE_NOT_OPENED;
                            return *status;
                        }
                        iMatch = Some(ii);
                    }
                }
            } /* end if old file recognizes extended syntax */
        } /* end if old fptr exists */
    } /* end loop over NMAXFILES */

    if let Some(iMatch) = iMatch {
        /* SAFETY: iMatch was set from a non-null FPTR_TABLE slot above, so it
        still points at a live FITSfile.  The new handle shares it; open_count
        is incremented below, and ffclos frees it once that reaches zero. */
        let sharedFptr = unsafe { FptrRef::from_ptr(FPTR_TABLE[iMatch]) };

        // HEAP ALLOCATION
        let f = box_try_new(fitsfile {
            HDUposition: 0,   /* set initial position */
            Fptr: sharedFptr, /* point to the structure */
        });

        if f.is_err() {
            ffpmsg_str("failed to allocate structure for following file: (ffopen)");
            ffpmsg_slice(url);
            *status = MEMORY_ALLOCATION;
            return *status;
        }

        let mut f = f.unwrap();

        (f.Fptr.open_count) += 1; /* increment usage counter */

        *fptr = Some(f);

        if binspec[0] != 0 {
            /* if binning specified, don't move */
            extspec[0] = 0;
        }

        /* all the filtering has already been applied, so ignore */
        rowfilter[0] = 0;
        binspec[0] = 0;
        colspec[0] = 0;
        *isopen = 1;
    }

    *status
}

pub(crate) fn check_is_file_fits(fp: &mut File) -> c_int {
    const NBYTES: usize = 1000;
    let mut buf: [c_char; NBYTES] = [0; NBYTES];
    let mut nread: usize = 0;

    /* read up to NBYTES bytes, matching C fread semantics */
    {
        let bytes: &mut [u8] = cast_slice_mut(&mut buf);
        while nread < NBYTES {
            match fp.read(&mut bytes[nread..]) {
                Ok(0) => break,
                Ok(n) => nread += n,
                Err(_) => break,
            }
        }
    }
    let _ = fp.rewind();
    if nread == 0 {
        return 0;
    }

    check_is_mem_fits(&buf, nread)
}
pub(crate) fn check_is_mem_fits(inputmem: &[c_char], len: usize) -> c_int {
    let mut isFits = 0;

    /* Check for gzip magic number */
    if len >= 2 && (inputmem[0] as u8) == 0x1f && (inputmem[1] as u8) == 0x8b {
        /* Just need to uncompress the beginning portion of the
        file to test for FITS.  So just pass a small buffer
        that won't be reallocated by uncompress2mem_from_mem.*/
        let mut nBuff: usize = 100;
        let mut nUncomp: usize = 0;
        let mut status = 0;
        let mut tstFits: Vec<u8> = vec![0; nBuff + 1];
        let mut tstFits_ptr: *mut u8 = tstFits.as_mut_ptr();
        /* This will return a bad status if all of inputmem buffer
        can't be uncompressed into tstFits, but we don't care. */
        unsafe {
            uncompress2mem_from_mem(
                inputmem,
                len,
                &raw mut tstFits_ptr,
                &mut nBuff,
                None,
                Some(&mut nUncomp),
                &mut status,
            );
        }
        tstFits[nUncomp] = 0;
        if strlen_safe(cast_slice(&tstFits)) >= 6
            && strncmp_safe(cast_slice(&tstFits), cs!(c"SIMPLE"), 6) == 0
        {
            isFits = 1;
        }
    } else if len >= 6 && strncmp_safe(inputmem, cs!(c"SIMPLE"), 6) == 0 {
        isFits = 1;
    }

    isFits
}

/// Utility function for common operation in fits_already_open
/// fullpath:  I/O string to be standardized. Assume len = FLEN_FILENAME
fn standardize_path(fullpath: &mut [c_char], status: &mut c_int) -> c_int {
    let mut tmpPath: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut cwd: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    if fits_path2url(fullpath, FLEN_FILENAME, &mut tmpPath, status) > 0 {
        return *status;
    }

    if tmpPath[0] != bb(b'/') {
        fits_get_cwd(&mut cwd, status);
        if strlen_safe(&cwd) + strlen_safe(&tmpPath) + 1 > FLEN_FILENAME - 1 {
            ffpmsg_str("Tile name is too long. (standardize_path)");
            *status = FILE_NOT_OPENED;
            return *status;
        }
        strcat_safe(&mut cwd, cs!(c"/"));
        strcat_safe(&mut cwd, &tmpPath);
        fits_clean_url(&cwd, &mut tmpPath, status);
    }

    strcpy_safe(fullpath, &tmpPath);

    *status
}

/// This differs from the standardize_path utility function in that this is
/// intended to perform '/./' and '/../' conversion for both absolute and relative
/// input paths. standardize_path only operates on relative input paths.
fn normalize_path(fullpath: &mut [c_char], status: &mut c_int) -> c_int {
    let mut tmpPath: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut cwd: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    /* Not handling '~' in here */
    if fullpath[0] == bb(b'~') {
        return *status;
    }

    if fullpath[0] != bb(b'/') {
        fits_get_cwd(&mut cwd, status);
        if strlen_safe(&cwd) + strlen_safe(fullpath) + 1 > FLEN_FILENAME - 1 {
            ffpmsg_str("File name is too long. (normalize_path)");
            *status = FILE_NOT_OPENED;
            return *status;
        }
        strcat_safe(&mut cwd, cs!(c"/"));
        strcat_safe(&mut cwd, fullpath);
        fits_clean_url(&cwd, fullpath, status);
    } else {
        fits_clean_url(fullpath, &mut tmpPath, status);
        strcpy_safe(fullpath, &tmpPath);
    }

    *status
}

fn exclude_path(testpath: &[c_char]) -> c_int {
    const NEXCLUDE: usize = 2;
    let excludeStrs: [&[c_char]; NEXCLUDE] = [cs!(c"/etc/"), cs!(c"/var/")];
    let updir = cs!(c"..");
    let mut exclude = 0;

    if testpath[0] == bb(b'~') {
        /* For home directory '~' paths, will forbid a combination
        of '..' and an appearance of the excludeStrs anywhere in
        the testpath.  */
        if strstr_safe(testpath, updir).is_some() {
            let mut i = 0;
            while i < NEXCLUDE && exclude == 0 {
                if strstr_safe(testpath, excludeStrs[i]).is_some() {
                    exclude = 1;
                }
                i += 1;
            }
        }
    } else {
        let mut i = 0;
        while i < NEXCLUDE && exclude == 0 {
            exclude =
                (strncmp_safe(testpath, excludeStrs[i], strlen_safe(excludeStrs[i])) == 0) as c_int;
            i += 1;
        }
    }
    exclude
}
fn skip_host_string(testpath: &[c_char]) -> &[c_char] {
    if strlen_safe(testpath) < 2 || testpath[0] == bb(b'~') || testpath[0] == bb(b'/') {
        return testpath;
    }

    let pslash = match strchr_safe(testpath, bb(b'/')) {
        None => return testpath,
        Some(x) => x,
    };

    /* don't treat ./ or ../ as a host string */
    if strncmp_safe(testpath, cs!(c"./"), 2) == 0 || strncmp_safe(testpath, cs!(c"../"), 3) == 0 {
        return testpath;
    }

    /* any ':' assume is part of host */
    let p = strchr_safe(testpath, bb(b':'));
    if let Some(p) = p
        && p < pslash
    {
        return &testpath[pslash..];
    }

    let mut p = strchr_safe(testpath, bb(b'.'));
    /* If testpath starts with a '.', look for a second '.' before the '/'. */
    if p == Some(0) {
        p = strchr_safe(&testpath[1..], bb(b'.')).map(|x| x + 1);
    }

    match p {
        None => testpath,
        Some(p) if pslash < p => testpath,
        /* assume everything before the first slash is a host name.*/
        Some(_) => &testpath[pslash..],
    }
}

/// specialized routine that returns 1 if the file is known to be a temporary
/// copy of the originally opened file.  Otherwise it returns 0.
pub(crate) fn fits_is_this_a_copy(urltype: [c_char; 20] /* I - type of file */) -> c_int {
    let mut iscopy: c_int = 0;

    if strncmp_safe(&urltype, cs!(c"mem"), 3) == 0 {
        iscopy = 1; /* file copy is in memory */
    } else if strncmp_safe(&urltype, cs!(c"compress"), 8) == 0 {
        iscopy = 1; /* compressed diskfile that is uncompressed in memory */
    } else if strncmp_safe(&urltype, cs!(c"http"), 4) == 0 {
        iscopy = 1; /* copied file using http protocol */
    } else if strncmp_safe(&urltype, cs!(c"ftp"), 3) == 0 {
        iscopy = 1; /* copied file using ftp protocol */
    } else if strncmp_safe(&urltype, cs!(c"gsiftp"), 6) == 0 {
        iscopy = 1; /* copied file using gsiftp protocol */
    } else if strncmp_safe(&urltype, cs!(c"stdin"), 5) == 0 {
        iscopy = 1; /* piped stdin has been copied to memory */
    } else {
        iscopy = 0; /* file is not known to be a copy */
    }

    iscopy
}

/// Look for the closing single quote character in the input string
fn find_quote(string: &[c_char]) -> Option<usize> {
    let mut i = 0;
    let len = string.len();

    while i < len && string[i] != 0 {
        if string[i] == bb(b'\'') {
            /* found the closing quote */
            return Some(i + 1); /* set pointer to next char */
        }
        i += 1;
    }

    None /* opps, didn't find the closing character */
}

/*
  Find matching delimiter, respecting quoting and (potentially nested) parentheses

  char *string - null-terminated string to be searched for delimiter
  char delim - single delimiter to search for (one of '")]} )

  returns: pointer to character after delimiter, or 0 if not found
*/
pub(crate) fn fits_find_match_delim(string: &[c_char], delim: c_char) -> Option<usize> {
    if string.is_empty() {
        return None;
    }

    match delim as u8 {
        b'\'' => find_quote(string),
        b'"' => find_doublequote(string),
        b'}' => find_curlybracket(string),
        b']' => find_bracket(string),
        b')' => find_paren(string),
        _ => None, // Invalid delimiter, return failure
    }
}

/// Look for the closing double quote character in the input string
fn find_doublequote(string: &[c_char]) -> Option<usize> {
    let mut i = 0;
    let len = string.len();

    while i < len && string[i] != 0 {
        if string[i] == bb(b'"') {
            /* found the closing quote */
            return Some(i + 1); /* set pointer to next char */
        }
        i += 1;
    }

    None /* opps, didn't find the closing character */
}

/// look for the closing parenthesis character in the input string
fn find_paren(string: &[c_char]) -> Option<usize> {
    let mut i = 0;
    let len = string.len();

    while i < len && string[i] != 0 {
        if string[i] == bb(b')') {
            /* found the closing parens */
            return Some(i + 1); /* set pointer to next char */
        } else if string[i] == bb(b'(') {
            /* found another level of parens */
            i += 1;
            i += find_paren(&string[i..])?;
        } else if string[i] == bb(b'[') {
            /* found another level of parens */
            i += 1;
            i += find_bracket(&string[i..])?;
        } else if string[i] == bb(b'{') {
            /* found another level of parens */
            i += 1;
            i += find_curlybracket(&string[i..])?;
        } else if string[i] == bb(b'"') {
            /* found another level of parens */
            i += 1;
            i += find_doublequote(&string[i..])?;
        } else if string[i] == bb(b'\'') {
            /* found another level of parens */
            i += 1;
            i += find_quote(&string[i..])?;
        } else {
            i += 1;
        }
    }

    None /* opps, didn't find the closing character */
}

/// look for the closing bracket character in the input string
fn find_bracket(string: &[c_char]) -> Option<usize> {
    let mut i = 0;
    let len = string.len();

    while i < len && string[i] != 0 {
        if string[i] == bb(b']') {
            /* found the closing bracket */
            return Some(i + 1); /* set pointer to next char */
        } else if string[i] == bb(b'(') {
            /* found another level of parens */
            i += 1;
            i += find_paren(&string[i..])?;
        } else if string[i] == bb(b'[') {
            /* found another level of parens */
            i += 1;
            i += find_bracket(&string[i..])?;
        } else if string[i] == bb(b'{') {
            /* found another level of parens */
            i += 1;
            i += find_curlybracket(&string[i..])?;
        } else if string[i] == bb(b'"') {
            /* found another level of parens */
            i += 1;
            i += find_doublequote(&string[i..])?;
        } else if string[i] == bb(b'\'') {
            /* found another level of parens */
            i += 1;
            i += find_quote(&string[i..])?;
        } else {
            i += 1;
        }
    }

    None /* opps, didn't find the closing character */
}

/// look for the closing curly bracket character in the input string
fn find_curlybracket(string: &[c_char]) -> Option<usize> {
    let mut i = 0;
    let len = string.len();

    while i < len && string[i] != 0 {
        if string[i] == bb(b'}') {
            /* found the closing curly bracket */
            return Some(i + 1); /* set pointer to next char */
        } else if string[i] == bb(b'(') {
            /* found another level of parens */
            i += 1;
            i += find_paren(&string[i..])?;
        } else if string[i] == bb(b'[') {
            /* found another level of parens */
            i += 1;
            i += find_bracket(&string[i..])?;
        } else if string[i] == bb(b'{') {
            /* found another level of parens */
            i += 1;
            i += find_curlybracket(&string[i..])?;
        } else if string[i] == bb(b'"') {
            /* found another level of parens */
            i += 1;
            i += find_doublequote(&string[i..])?;
        } else if string[i] == bb(b'\'') {
            /* found another level of parens */
            i += 1;
            i += find_quote(&string[i..])?;
        } else {
            i += 1;
        }
    }

    None /* opps, didn't find the closing character */
}

/// replace commas with semicolons, unless the comma is within a quoted or bracketed expression
fn comma2semicolon(string: &mut [c_char]) -> c_int {
    let mut i = 0;
    let len = string.len();

    while i < len && string[i] != 0 {
        if string[i] == bb(b',') {
            /* found a comma */
            string[i] = bb(b';');
            i += 1;
        } else if string[i] == bb(b'(') {
            /* found another level of parens */
            i += 1;
            let p = find_paren(&string[i..]);

            match p {
                None => return 1,
                Some(x) => i += x,
            }
        } else if string[i] == bb(b'[') {
            i += 1;
            let p = find_bracket(&string[i..]);
            match p {
                None => return 1,
                Some(x) => i += x,
            }
        } else if string[i] == bb(b'{') {
            i += 1;
            let p = find_curlybracket(&string[i..]);
            match p {
                None => return 1,
                Some(x) => i += x,
            }
        } else if string[i] == bb(b'"') {
            i += 1;
            let p = find_doublequote(&string[i..]);
            match p {
                None => return 1,
                Some(x) => i += x,
            }
        } else if string[i] == bb(b'\'') {
            i += 1;
            let p = find_quote(&string[i..]);
            match p {
                None => return 1,
                Some(x) => i += x,
            }
        } else {
            i += 1;
        }
    }

    0 /* reached end of string */
}

/// modify columns in a table and/or header keywords in the HDU
///
/// # Parameters
///
/// * `infptr`  — (IO) pointer to input table; on output it
/// * `outfile` — (I) name for output file
/// * `expr`    — (I) column edit expression
pub(crate) fn ffedit_columns(
    infptr: &mut Option<Box<fitsfile>>,
    /*      points to the new selected rows table */
    outfile: &[c_char],
    expr: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut newptr: Option<Box<fitsfile>> = None;
    let mut hdunum: c_int = 0;
    let mut slen: c_int = 0;
    let mut colnum: c_int = -1;
    let mut testnum: c_int = 0;
    let mut deletecol: c_int = 0;
    let mut savecol: c_int = 0;
    let mut numcols: c_int = 0;

    let mut tstatus: c_int = 0;

    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut colname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut oldname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut colformat: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    let mut testname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    let mut colindex: Vec<c_int> = Vec::new();

    let fptr: &mut Box<fitsfile> = (infptr).as_mut().unwrap();

    if outfile[0] != 0 {
        /* create new empty file in to hold the selected rows */
        if ffinit_safe(&mut newptr, outfile, status) > 0 {
            ffpmsg_str("failed to create file for copy (ffedit_columns)");
            return *status;
        }

        let mut newptr = newptr.expect(NULL_MSG);

        ffghdn_safe(fptr, &mut hdunum); /* current HDU number in input file */

        /* copy all HDUs to the output copy, if the 'only_one' flag is not set */
        if fptr.Fptr.only_one == 0 {
            let mut ii = 1;
            loop {
                if ffmahd_safe(fptr, ii, None, status) > 0 {
                    break;
                }

                ffcopy_safe(fptr, &mut newptr, 0, status);
                ii += 1;
            }

            if *status == END_OF_FILE {
                *status = 0; /* got the expected EOF error; reset = 0  */
            } else if *status > 0 {
                ffclos_safe(newptr, status);
                ffpmsg_str("failed to copy all HDUs from input file (ffedit_columns)");
                return *status;
            }
        } else {
            /* only copy the primary array and the designated table extension */
            ffmahd_safe(fptr, 1, None, status);
            ffcopy_safe(fptr, &mut newptr, 0, status);
            ffmahd_safe(fptr, hdunum, None, status);
            ffcopy_safe(fptr, &mut newptr, 0, status);
            if *status > 0 {
                ffclos_safe(newptr, status);
                ffpmsg_str("failed to copy all HDUs from input file (ffedit_columns)");
                return *status;
            }
            hdunum = 2;
        }

        /* close the original file and return ptr to the new image */
        let f_tmp = infptr.take().unwrap();
        ffclos_safe(f_tmp, status);

        *infptr = Some(newptr); /* reset the pointer to the new table */

        let fptr: &mut Box<fitsfile> = (infptr).as_mut().unwrap();

        /* move back to the selected table HDU */
        if ffmahd_safe(fptr, hdunum, None, status) > 0 {
            ffpmsg_str("failed to copy the input file (ffedit_columns)");
            return *status;
        }
    }

    let fptr: &mut Box<fitsfile> = (infptr).as_mut().unwrap();

    /* remove the "col " from the beginning of the column edit expression */
    let mut cptr = &mut expr[4..]; // &expr

    while cptr[0] == bb(b' ') {
        cptr = &mut cptr[1..]; /* skip leading white space */
    }

    /* Check if need to import expression from a file */
    let mut contents = None;
    let mut c: Vec<c_char>;

    if cptr[0] == bb(b'@') {
        if ffimport_file_safe(&cptr[1..], &mut contents, status) != 0 {
            return *status;
        }

        c = contents.unwrap();
        let _len = CStr::from_bytes_until_nul(cast_slice(&c))
            .unwrap()
            .to_bytes_with_nul()
            .len();
        cptr = &mut c[.._len];

        while cptr[0] == bb(b' ') {
            cptr = &mut cptr[1..]; /* skip leading white space... again */
        }
    }

    tstatus = 0;
    ffgncl_safe(fptr, &mut numcols, &mut tstatus); /* get initial # of cols */

    /* as of July 2012, the CFITSIO column filter syntax was modified */
    /* so that commas may be used to separate clauses, as well as semi-colons. */
    /* This was done because users cannot enter the semi-colon in the HEASARC's */
    /* Hera on-line data processing system for computer security reasons.  */
    /* Therefore, we must convert those commas back to semi-colons here, but we */
    /* must not convert any columns that occur within parenthesies.  */

    if comma2semicolon(cptr) != 0 {
        ffpmsg_str("parsing error in column filter expression");
        ffpmsg_slice(cptr);

        *status = PARSE_SYNTAX_ERR;
        return *status;
    }

    /* parse expression and get first clause, if more than 1 */
    let mut tmp_clause = None;
    let mut cptr_idx = 0;

    loop {
        slen = fits_get_token2_safe(
            cptr,
            &mut cptr_idx,
            cast_slice(c";".to_bytes_with_nul()),
            &mut tmp_clause,
            None,
            status,
        );

        if slen <= 0 {
            break;
        }

        let _clause_len = tmp_clause.as_deref().unwrap().len();
        let clause = tmp_clause.as_deref_mut().unwrap();
        if cptr[0] == bb(b';') {
            cptr = &mut cptr[1..];
        }
        clause[slen as usize] = 0;

        if clause[0] == bb(b'!') || clause[0] == bb(b'-') {
            let mut clause1 = 1;
            let mut clen = if clause[clause1] != 0 {
                strlen_safe(&clause[clause1..])
            } else {
                0
            };
            /* ===================================== */
            /* Case I. delete this column or keyword */
            /* ===================================== */

            /* Case Ia. delete column names with 0-or-more wildcard
                    -COLNAME+ - delete repeated columns with exact name
                -COLNAM*+ - delete columns matching patterns
            */
            if *status == 0
                && clen > 1
                && clause[clause1] != bb(b'#')
                && clause[clause1 + clen - 1] == bb(b'+')
            {
                clause[clause1 + clen - 1] = 0;
                clen -= 1;

                /* Note that this is a delete 0 or more specification,
                which means that no matching columns is not an error. */
                loop {
                    let mut status_del = 0;

                    /* Have to set status=0 so we can reset the search at
                    start column.  Because we are deleting columns on
                    the fly here, we have to reset the search every
                    time. The only penalty here is execution time
                    because leaving *status == COL_NOT_UNIQUE is merely
                    an optimization for tables assuming the tables do
                    not change from one call to the next. (an
                    assumption broken in this loop) */
                    *status = 0;
                    ffgcno_safe(
                        fptr,
                        CASEINSEN as c_int,
                        &clause[clause1..],
                        &mut colnum,
                        status,
                    );
                    /* ffgcno returns COL_NOT_UNIQUE if there are multiple columns,
                    and COL_NOT_FOUND after the last column is found, and
                    COL_NOT_FOUND if no matches were found */
                    if *status != 0 && *status != COL_NOT_UNIQUE {
                        break;
                    }

                    if ffdcol_safe(fptr, colnum, &mut status_del) > 0 {
                        ffpmsg_str("failed to delete column in input file:");
                        ffpmsg_slice(clause);

                        *status = status_del;
                        return *status;
                    }
                    deletecol = 1; /* set flag that at least one col was deleted */
                    numcols -= 1;
                    if *status != COL_NOT_UNIQUE {
                        break;
                    }
                }

                *status = 0; /* No matches are still successful */
                colnum = -1; /* Ignore the column we found */

            /* Case Ib. delete column names with wildcard or not
                    -COLNAME  - deleted exact column
                -COLNAM*  - delete first column that matches pattern
               Note no leading '#'
            */
            } else if clause[clause1] != 0
                && clause[clause1] != bb(b'#')
                && ((ffgcno_safe(
                    fptr,
                    CASEINSEN as c_int,
                    &clause[clause1..],
                    &mut colnum,
                    status,
                ) <= 0)
                    || *status == COL_NOT_UNIQUE)
            {
                /* a column with this name exists, so try to delete it */
                *status = 0; /* Clear potential status=COL_NOT_UNIQUE */
                if ffdcol_safe(fptr, colnum, status) > 0 {
                    ffpmsg_str("failed to delete column in input file:");
                    ffpmsg_slice(clause);

                    return *status;
                }
                deletecol = 1; /* set flag that at least one col was deleted */
                numcols -= 1;
                colnum = -1;
            }
            /* Case Ic. delete keyword(s)
                    -KEYNAME,#KEYNAME  - delete exact keyword (first match)
                -KEYNAM*,#KEYNAM*  - delete first matching keyword
                -KEYNAME+,-#KEYNAME+ - delete 0-or-more exact matches of exact keyword
                -KEYNAM*+,-#KEYNAM*+ - delete 0-or-more wildcard matches
               Note the preceding # is optional if no conflicting column name exists
               and that wildcard patterns are described in "colfilter" section of
               documentation.
            */
            else {
                let mut delall = false;
                let mut haswild = false;
                ffcmsg_safe(); /* clear previous error message from ffgcno */
                /* try deleting a keyword with this name */
                *status = 0;
                /* skip past leading '#' if any */
                if clause[clause1] == bb(b'#') {
                    clause1 += 1;
                }
                clen = strlen_safe(&clause[clause1..]);

                /* Repeat deletion of keyword if requested with trailing '+' */
                if clen > 1 && clause[clause1 + clen - 1] == bb(b'+') {
                    delall = true;
                    clause[clause1 + clen - 1] = 0;
                }
                /* Determine if this pattern has wildcards */
                if strchr_safe(&clause[clause1..], bb(b'?')).is_some()
                    || strchr_safe(&clause[clause1..], bb(b'*')).is_some()
                    || strchr_safe(&clause[clause1..], bb(b'#')).is_some()
                {
                    haswild = true;
                }

                if haswild {
                    /* ffdkey() behaves differently if the pattern has a wildcard:
                    it only checks from the "current" header position to the end, and doesn't
                    check before the "current" header position.  Therefore, for the
                    case of wildcards we will have to reset to the beginning. */
                    ffmaky_safe(fptr, 1, status); /* reset pointer to beginning of header */
                }

                /* Single or repeated deletions until done */
                loop {
                    if ffdkey_safe(fptr, &clause[clause1..], status) > 0 {
                        if delall && *status == KEY_NO_EXIST {
                            /* Found last wildcard item. Stop deleting */
                            ffcmsg_safe();
                            *status = 0;
                            delall = false; /* Force end of this loop */
                        } else {
                            /* This was not a wildcard deletion, or it resulted in
                            another kind of error */
                            ffpmsg_str("column or keyword to be deleted does not exist:");
                            ffpmsg_slice(&clause[clause1..]);
                            return *status;
                        }
                    }
                    if !delall {
                        break;
                    }
                } /* end do{} */
            }
        } else {
            /* ===================================================== */
            /* Case II:
            this is either a column name, (case 1)

                or a new column name followed by double = ("==") followed
                by the old name which is to be renamed. (case 2A)

                or a column or keyword name followed by a single "=" and a
            calculation expression (case 2B) */
            /* ===================================================== */
            let mut cptr2 = 0; //clause
            let mut tstbuff = None;
            let mut ptr_index = 0;
            slen = fits_get_token2_safe(
                &clause[cptr2..],
                &mut ptr_index,
                cs!(c"( ="),
                &mut tstbuff,
                None,
                status,
            );

            cptr2 += ptr_index;

            if slen == 0 || *status != 0 {
                ffpmsg_str("error: column or keyword name is blank (ffedit_columns):");
                ffpmsg_slice(clause);
                if *status == 0 {
                    *status = URL_PARSE_ERROR;
                }
                return *status;
            }

            let tstbuff = tstbuff.unwrap(); // Since slen != 0, can assume this is valid unwrap

            if strlen_safe(&tstbuff) > FLEN_VALUE - 1 {
                ffpmsg_str("error: column or keyword name is too long (ffedit_columns):");
                ffpmsg_slice(clause);
                *status = URL_PARSE_ERROR;
                return *status;
            }

            strcpy_safe(&mut colname, &tstbuff);
            drop(tstbuff);

            /* If this is a keyword of the form
                 #KEYWORD#
               then transform to the form
                 #KEYWORDn
               where n is the previously used column number
            */
            if colname[0] == bb(b'#')
                && strstr_safe(&colname[1..], cs!(c"#")) == Some(strlen_safe(&colname) - 1)
            {
                if colnum <= 0 {
                    ffpmsg_str("The keyword name:");
                    ffpmsg_slice(&colname);
                    ffpmsg_str("is invalid unless a column has been previously");
                    ffpmsg_str("created or editted by a calculator command");
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
                colname[strlen_safe(&colname) - 1] = 0;
                /* Make keyword name and put it in oldname */
                ffkeyn_safe(&colname[1..], colnum, &mut oldname, status);
                if *status != 0 {
                    return *status;
                }
                /* Re-copy back into colname */
                strcpy_safe(&mut colname[1..], &oldname);
            } else if strstr_safe(&colname, cs!(c"#")) == Some(strlen_safe(&colname) - 1) {
                /*  colname is of the form "NAME#";  if
                      a) colnum is defined, and
                      b) a column with literal name "NAME#" does not exist, and
                      c) a keyword with name "NAMEn" (where n=colnum) exists, then
                    transfrom the colname string to "NAMEn", otherwise
                    do nothing.
                */
                if colnum > 0 {
                    /* colnum must be defined */
                    tstatus = 0;
                    ffgcno_safe(
                        fptr,
                        CASEINSEN as c_int,
                        &colname,
                        &mut testnum,
                        &mut tstatus,
                    );
                    if tstatus != 0 && tstatus != COL_NOT_UNIQUE {
                        /* OK, column doesn't exist, now see if keyword exists */
                        ffcmsg_safe(); /* clear previous error message from ffgcno */
                        strcpy_safe(&mut testname, &colname);
                        testname[strlen_safe(&testname) - 1] = 0;
                        /* Make keyword name and put it in oldname */
                        ffkeyn_safe(&testname, colnum, &mut oldname, status);
                        if *status != 0 {
                            return *status;
                        }

                        tstatus = 0;
                        if ffgcrd_safe(fptr, &oldname, &mut card, &mut tstatus) == 0 {
                            /* Keyword does exist; copy real name back into colname */
                            strcpy_safe(&mut colname, &oldname);
                        }
                    }
                }
            }

            /* if we encountered an opening parenthesis, then we need to */
            /* find the closing parenthesis, and concatinate the 2 strings */
            /* This supports expressions like:
                [col #EXTNAME(Extension name)="GTI"]
            */
            if clause[cptr2] == bb(b'(') {
                let mut tstbuff = None;
                let mut ptr_index = 0;
                if fits_get_token2_safe(
                    &clause[cptr2..],
                    &mut ptr_index,
                    cs!(c")"),
                    &mut tstbuff,
                    None,
                    status,
                ) == 0
                {
                    strcat_safe(&mut colname, cs!(c")"));
                } else {
                    cptr2 += ptr_index;
                    let tstbuff = tstbuff.unwrap();
                    if (strlen_safe(&tstbuff) + strlen_safe(&colname) + 1) > FLEN_VALUE - 1 {
                        ffpmsg_str("error: column name is too long (ffedit_columns):");
                        *status = URL_PARSE_ERROR;
                        return *status;
                    }
                    strcat_safe(&mut colname, &tstbuff);
                    strcat_safe(&mut colname, cs!(c")"));
                    drop(tstbuff);
                }
                cptr2 += 1;
            }

            while clause[cptr2] == bb(b' ') {
                cptr2 += 1; /* skip white space */
            }

            if clause[cptr2] != bb(b'=') {
                /* ------------------------------------ */
                /* case 1 - simply the name of a column */
                /* ------------------------------------ */

                /* look for matching column */
                ffgcno_safe(fptr, CASEINSEN as c_int, &colname, &mut testnum, status);

                while *status == COL_NOT_UNIQUE {
                    /* the column name contained wild cards, and it */
                    /* matches more than one column in the table. */

                    colnum = testnum;

                    /* keep this column in the output file */
                    savecol = 1;

                    if colindex.is_empty() {
                        colindex = vec![0; 999];
                    }

                    colindex[(colnum - 1) as usize] = 1; /* flag this column number */

                    /* look for other matching column names */
                    ffgcno_safe(fptr, CASEINSEN as c_int, &colname, &mut testnum, status);

                    if *status == COL_NOT_FOUND {
                        *status = 999; /* temporary status flag value */
                    }
                }

                if *status <= 0 {
                    colnum = testnum;

                    /* keep this column in the output file */
                    savecol = 1;

                    if colindex.is_empty() {
                        colindex = vec![0; 999];
                    }

                    colindex[(colnum - 1) as usize] = 1; /* flag this column number */
                } else if *status == 999 {
                    /* this special flag value does not represent an error */
                    *status = 0;
                } else {
                    ffpmsg_str("Syntax error in columns specifier in input URL:");
                    ffpmsg_slice(&clause[cptr2..]);
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
            } else {
                /* ----------------------------------------------- */
                /* case 2 where the token ends with an equals sign */
                /* ----------------------------------------------- */

                cptr2 += 1; /* skip over the first '=' */

                if clause[cptr2] == bb(b'=') {
                    /*................................................. */
                    /*  Case A:  rename a column or keyword;  syntax is
                    "new_name == old_name"  */
                    /*................................................. */

                    cptr2 += 1; /* skip the 2nd '=' */
                    while clause[cptr2] == bb(b' ') {
                        cptr2 += 1; /* skip white space */
                    }

                    let mut tstbuff = None;
                    let mut ptr_index = 0;
                    if fits_get_token2_safe(
                        &clause[cptr2..],
                        &mut ptr_index,
                        cs!(c" "),
                        &mut tstbuff,
                        None,
                        status,
                    ) == 0
                    {
                        oldname[0] = 0;
                    } else {
                        cptr2 += ptr_index;
                        let tstbuff = tstbuff.unwrap();

                        if strlen_safe(&tstbuff) > FLEN_VALUE - 1 {
                            ffpmsg_str("error: column name syntax is too long (ffedit_columns):");
                            *status = URL_PARSE_ERROR;
                            return *status;
                        }
                        strcpy_safe(&mut oldname, &tstbuff);
                    }

                    /* get column number of the existing column */
                    if ffgcno_safe(fptr, CASEINSEN as c_int, &oldname, &mut colnum, status) <= 0 {
                        /* modify the TTYPEn keyword value with the new name */
                        ffkeyn_safe(cs!(c"TTYPE"), colnum, &mut keyname, status);

                        if ffmkys_safe(fptr, &keyname, &colname, None, status) > 0 {
                            ffpmsg_str("failed to rename column in input file");
                            ffpmsg_str(" oldname =");
                            ffpmsg_slice(&oldname);
                            ffpmsg_str(" newname =");
                            ffpmsg_slice(&colname);
                            return *status;
                        }
                        /* keep this column in the output file */
                        savecol = 1;
                        if colindex.is_empty() {
                            colindex = vec![0; 999];
                        }

                        colindex[(colnum - 1) as usize] = 1; /* flag this column number */
                    } else {
                        /* try renaming a keyword */
                        ffcmsg_safe(); /* clear error message stack */
                        *status = 0;
                        if ffmnam_safe(fptr, &oldname, &colname, status) > 0 {
                            ffpmsg_str("column or keyword to be renamed does not exist:");
                            ffpmsg_slice(clause);
                            return *status;
                        }
                    }
                } else {
                    /*...................................................... */
                    /* Case B: */
                    /* this must be a general column/keyword calc expression */
                    /* "name = expression" or "colname(TFORM) = expression" */
                    /*...................................................... */

                    /* parse the name and TFORM values, if present */
                    colformat[0] = 0;
                    let mut cptr3 = &colname[..];

                    let mut tstbuff = None;
                    let mut ptr_index = 0;
                    if fits_get_token2_safe(
                        cptr3,
                        &mut ptr_index,
                        cs!(c"("),
                        &mut tstbuff,
                        None,
                        status,
                    ) == 0
                    {
                        oldname[0] = 0;
                    } else {
                        cptr3 = &cptr3[ptr_index..];
                        let tstbuff = tstbuff.unwrap();

                        if strlen_safe(&tstbuff) > FLEN_VALUE - 1 {
                            ffpmsg_str("column expression is too long (ffedit_columns)");
                            *status = URL_PARSE_ERROR;
                            return *status;
                        }
                        strcpy_safe(&mut oldname, &tstbuff);
                        drop(tstbuff);
                    }

                    if cptr3[0] == bb(b'(') {
                        cptr3 = &cptr3[1..]; /* skip the '(' */

                        let mut tstbuff = None;
                        let mut ptr_index = 0;
                        if fits_get_token2_safe(
                            cptr3,
                            &mut ptr_index,
                            cs!(c")"),
                            &mut tstbuff,
                            None,
                            status,
                        ) == 0
                        {
                            colformat[0] = 0;
                        } else {
                            cptr3 = &cptr3[ptr_index..];
                            let tstbuff = tstbuff.unwrap();
                            if strlen_safe(&tstbuff) > FLEN_VALUE - 1 {
                                ffpmsg_str("column expression is too long (ffedit_columns)");
                                *status = URL_PARSE_ERROR;
                                return *status;
                            }
                            strcpy_safe(&mut colformat, &tstbuff);
                            drop(tstbuff);
                        }
                    }

                    /* calculate values for the column or keyword */
                    /*   cptr2 = the expression to be calculated */
                    /*   oldname = name of the column or keyword */
                    /*   colformat = column format, or keyword comment string */

                    /* the C passes the same fitsfile* as input and output here,
                    which calculates into the file being edited */
                    if ffcalc_inplace_safe(fptr, &clause[cptr2..], &oldname, &colformat, status) > 0
                    {
                        ffpmsg_str("Unable to calculate expression");
                        return *status;
                    }

                    /* test if this is a column and not a keyword */
                    tstatus = 0;
                    ffgcno_safe(
                        fptr,
                        CASEINSEN as c_int,
                        &oldname,
                        &mut testnum,
                        &mut tstatus,
                    );
                    if tstatus == 0 {
                        /* keep this column in the output file */
                        colnum = testnum;
                        savecol = 1;

                        if colindex.is_empty() {
                            colindex = vec![0; 999];
                        }

                        colindex[(colnum - 1) as usize] = 1;
                        if colnum > numcols {
                            numcols += 1;
                        }
                    } else {
                        ffcmsg_safe(); /* clear the error message stack */
                    }
                }
            }
        }
        //clause = NULL;
    }

    let clause = tmp_clause.as_deref_mut().unwrap();

    if savecol != 0 && deletecol == 0 {
        /* need to delete all but the specified columns */
        let mut ii = numcols;
        while ii > 0 {
            if colindex[(ii - 1) as usize] == 0 {
                /* delete this column */

                if ffdcol_safe(fptr, ii, status) > 0 {
                    ffpmsg_str("failed to delete column in input file:");
                    ffpmsg_slice(clause);
                    return *status;
                }
            }
            ii -= 1;
        }
    }

    *status
}

/// Copy a table cell of a given row and column into an image extension.
/// The output file must already have been created.  A new image
/// extension will be created in that file.
///
/// This routine was written by Craig Markwardt, GSFC
///
/// # Parameters
///
/// * `fptr`    — (I) point to input table
/// * `newptr`  — (O) existing output file; new image HDU will be appended to it
/// * `colname` — (I) column name / number containing the image
/// * `rownum`  — (I) number of the row containing the image
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_copy_cell2image(
    fptr: *mut fitsfile,
    newptr: *mut fitsfile,
    colname: *const [c_char; FLEN_VALUE],
    rownum: c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let newptr = newptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        let colname = &*colname;

        fits_copy_cell2image_safe(fptr, newptr, colname, rownum, status)
    }
}

/// Copy a table cell of a given row and column into an image extension.
/// The output file must already have been created.  A new image
/// extension will be created in that file.
///
/// This routine was written by Craig Markwardt, GSFC
///
/// # Parameters
///
/// * `fptr`    — (I) point to input table
/// * `newptr`  — (O) existing output file; new image HDU will be appended to it
/// * `colname` — (I) column name / number containing the image
/// * `rownum`  — (I) number of the row containing the image
/// * `status`  — (IO) error status
pub fn fits_copy_cell2image_safe(
    fptr: &mut fitsfile,
    newptr: &mut fitsfile,
    colname: &[c_char],
    rownum: c_long,
    status: &mut c_int,
) -> c_int {
    let mut buffer: [u8; 30000] = [0; 30000];
    let mut hdutype: c_int = 0;
    let mut colnum: c_int = 0;
    let mut typecode: c_int = 0;
    let bitpix: c_int;
    let mut naxis: c_int = 0;
    let mut maxelem: c_int = 0;
    let mut tstatus: c_int = 0;
    let mut naxes: [LONGLONG; 9] = [0; 9];
    let mut nbytes: LONGLONG;
    let mut firstbyte: LONGLONG;
    let mut ntodo: LONGLONG;
    let mut repeat: LONGLONG = 0;
    let mut startpos: LONGLONG = 0;
    let mut elemnum: LONGLONG = 0;
    let mut rowlen: LONGLONG = 0;
    let mut tnull: LONGLONG = 0;
    let mut twidth: c_long = 0;
    let mut incre: c_long = 0;
    let mut scale: f64 = 0.0;
    let mut zero: f64 = 0.0;
    let mut tform: [c_char; 20] = [0; 20];
    let mut templt: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    /* the canonical column name; the C overwrites its `colname` argument in
    place via ffgcnn, but ours is a `&[c_char]` input, so use a local buffer */
    let mut colname_buf: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    /* Table-to-image keyword translation table  */
    /*                        INPUT      OUTPUT  */
    /*                       01234567   01234567 */
    let patterns: [[&CStr; 2]; 70] = [
        [c"TSCALn", c"BSCALE"], /* Standard FITS keywords */
        [c"TZEROn", c"BZERO"],
        [c"TUNITn", c"BUNIT"],
        [c"TNULLn", c"BLANK"],
        [c"TDMINn", c"DATAMIN"],
        [c"TDMAXn", c"DATAMAX"],
        [c"iCTYPn", c"CTYPEi"], /* Coordinate labels */
        [c"iCTYna", c"CTYPEia"],
        [c"iCUNIn", c"CUNITi"], /* Coordinate units */
        [c"iCUNna", c"CUNITia"],
        [c"iCRVLn", c"CRVALi"], /* WCS keywords */
        [c"iCRVna", c"CRVALia"],
        [c"iCDLTn", c"CDELTi"],
        [c"iCDEna", c"CDELTia"],
        [c"iCRPXn", c"CRPIXi"],
        [c"iCRPna", c"CRPIXia"],
        [c"ijPCna", c"PCi_ja"],
        [c"ijCDna", c"CDi_ja"],
        [c"iVn_ma", c"PVi_ma"],
        [c"iSn_ma", c"PSi_ma"],
        [c"iCRDna", c"CRDERia"],
        [c"iCSYna", c"CSYERia"],
        [c"iCROTn", c"CROTAi"],
        [c"WCAXna", c"WCSAXESa"],
        [c"WCSNna", c"WCSNAMEa"],
        [c"LONPna", c"LONPOLEa"],
        [c"LATPna", c"LATPOLEa"],
        [c"EQUIna", c"EQUINOXa"],
        [c"MJDOBn", c"MJD-OBS"],
        [c"MJDAn", c"MJD-AVG"],
        [c"RADEna", c"RADESYSa"],
        [c"iCNAna", c"CNAMEia"],
        [c"DAVGn", c"DATE-AVG"],
        /* Delete table keywords related to other columns */
        [c"T????#a", c"-"],
        [c"TC??#a", c"-"],
        [c"TWCS#a", c"-"],
        [c"TDIM#", c"-"],
        [c"iCTYPm", c"-"],
        [c"iCUNIm", c"-"],
        [c"iCRVLm", c"-"],
        [c"iCDLTm", c"-"],
        [c"iCRPXm", c"-"],
        [c"iCTYma", c"-"],
        [c"iCUNma", c"-"],
        [c"iCRVma", c"-"],
        [c"iCDEma", c"-"],
        [c"iCRPma", c"-"],
        [c"ijPCma", c"-"],
        [c"ijCDma", c"-"],
        [c"iVm_ma", c"-"],
        [c"iSm_ma", c"-"],
        [c"iCRDma", c"-"],
        [c"iCSYma", c"-"],
        [c"iCROTm", c"-"],
        [c"WCAXma", c"-"],
        [c"WCSNma", c"-"],
        [c"LONPma", c"-"],
        [c"LATPma", c"-"],
        [c"EQUIma", c"-"],
        [c"MJDOBm", c"-"],
        [c"MJDAm", c"-"],
        [c"RADEma", c"-"],
        [c"iCNAma", c"-"],
        [c"DAVGm", c"-"],
        [c"EXTNAME", c"-"], /* Remove structural keywords*/
        [c"EXTVER", c"-"],
        [c"EXTLEVEL", c"-"],
        [c"CHECKSUM", c"-"],
        [c"DATASUM", c"-"],
        [c"*", c"+"], /* copy all other keywords */
    ];

    if *status > 0 {
        return *status;
    }

    /* get column number */
    if ffgcno_safe(fptr, CASEINSEN as c_int, colname, &mut colnum, status) > 0 {
        ffpmsg_str("column containing image in table cell does not exist:");
        ffpmsg_slice(colname);
        return *status;
    }

    /*  Check input and get parameters about the column: */
    if ffgcprll(
        fptr,
        colnum,
        rownum as LONGLONG,
        1,
        1,
        0,
        &mut scale,
        &mut zero,
        &mut tform,
        &mut twidth,
        &mut typecode,
        &mut maxelem,
        &mut startpos,
        &mut elemnum,
        &mut incre,
        &mut repeat,
        &mut rowlen,
        &mut hdutype,
        &mut tnull,
        cast_slice_mut(&mut buffer),
        status,
    ) > 0
    {
        return *status;
    }

    /* get the actual column name, in case a column number was given */
    ffkeyn_safe(cs!(c""), colnum, &mut templt, &mut tstatus);
    ffgcnn_safe(
        fptr,
        CASEINSEN as c_int,
        &templt,
        &mut colname_buf,
        &mut colnum,
        &mut tstatus,
    );

    if hdutype != BINARY_TBL {
        ffpmsg_str("This extension is not a binary table.");
        ffpmsg_str(" Cannot open the image in a binary table cell.");
        *status = NOT_BTABLE;
        return *status;
    }

    if typecode < 0 {
        /* variable length array */
        typecode *= -1;

        /* variable length arrays are 1-dimensional by default */
        naxis = 1;
        naxes[0] = repeat;
    } else {
        /* get the dimensions of the image */
        ffgtdmll_safe(fptr, colnum, 9, &mut naxis, &mut naxes, status);
    }

    if *status > 0 {
        ffpmsg_str("Error getting the dimensions of the image");
        return *status;
    }

    /* determine BITPIX value for the image */
    if typecode == TBYTE {
        bitpix = BYTE_IMG;
        nbytes = repeat;
    } else if typecode == TSHORT {
        bitpix = SHORT_IMG;
        nbytes = repeat * 2;
    } else if typecode == TLONG {
        bitpix = LONG_IMG;
        nbytes = repeat * 4;
    } else if typecode == TFLOAT {
        bitpix = FLOAT_IMG;
        nbytes = repeat * 4;
    } else if typecode == TDOUBLE {
        bitpix = DOUBLE_IMG;
        nbytes = repeat * 8;
    } else if typecode == TLONGLONG {
        bitpix = LONGLONG_IMG;
        nbytes = repeat * 8;
    } else if typecode == TLOGICAL {
        bitpix = BYTE_IMG;
        nbytes = repeat;
    } else {
        ffpmsg_str("Error: the following image column has invalid datatype:");
        ffpmsg_slice(&colname_buf);
        ffpmsg_slice(&tform);
        ffpmsg_str("Cannot open an image in a single row of this column.");
        *status = BAD_TFORM;
        return *status;
    }

    /* create new image in output file */
    if ffcrimll_safe(newptr, bitpix, naxis, &naxes, status) > 0 {
        ffpmsg_str("failed to write required primary array keywords in the output file");
        return *status;
    }

    let npat: c_int = patterns.len() as c_int;

    /* skip over the first 8 keywords, starting just after TFIELDS */
    fits_translate_keywords_safe(fptr, newptr, 9, &patterns, npat, colnum, 0, 0, status);

    /* The C builds a HISTORY card here, but the ffprec() that would write it is
    disabled (left to the caller), so the card is never used; omitted. */

    /* the use of ffread routine, below, requires that any 'dirty' */
    /* buffers in memory be flushed back to the file first */

    ffflsh_safe(fptr, false, status);

    /* finally, copy the data, one buffer size at a time */
    ffmbyt_safe(fptr, startpos, TRUE as c_int, status);
    firstbyte = 1;

    /* the upper limit on the number of bytes must match the declaration */
    /* read up to the first 30000 bytes in the normal way with ffgbyt */

    ntodo = cmp::min(30000, nbytes);
    ffgbyt(fptr, ntodo, &mut buffer, status);
    ffptbb_safe(newptr, 1, firstbyte, ntodo, &buffer, status);

    nbytes -= ntodo;
    firstbyte += ntodo;

    /* read any additional bytes with low-level ffread routine, for speed */
    while nbytes != 0 && *status <= 0 {
        ntodo = cmp::min(30000, nbytes);
        ffread(&fptr.Fptr, ntodo as c_long, &mut buffer, status);
        ffptbb_safe(newptr, 1, firstbyte, ntodo, &buffer, status);
        nbytes -= ntodo;
        firstbyte += ntodo;
    }

    /* Re-scan the header so that CFITSIO knows about all the new keywords */
    ffrdef_safe(newptr, status);

    *status
}

/// Copy an image extension into a table cell at a given row and
/// column.  The table must have already been created.  If the "colname"
/// column exists, it will be used, otherwise a new column will be created
/// in the table.
///
/// The "copykeyflag" parameter controls which keywords to copy from the
/// input image to the output table header (with any appropriate translation).
///
/// copykeyflag = 0  -- no keywords will be copied
/// copykeyflag = 1  -- essentially all keywords will be copied
/// copykeyflag = 2  -- copy only the WCS related keywords
///
/// This routine was written by Craig Markwardt, GSFC
///
/// # Parameters
///
/// * `fptr`        — (I) pointer to input image extension
/// * `newptr`      — (I) pointer to output table
/// * `colname`     — (I) name of column containing the image
/// * `rownum`      — (I) number of the row containing the image
/// * `copykeyflag` — (I) controls which keywords to copy
/// * `status`      — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_copy_image2cell(
    fptr: *mut fitsfile,
    newptr: *mut fitsfile,
    colname: *const c_char,
    rownum: c_long,
    copykeyflag: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let newptr = newptr.as_mut().expect(NULL_MSG);
        raw_to_slice!(colname);

        fits_copy_image2cell_safe(fptr, newptr, colname, rownum, copykeyflag, status)
    }
}

/// Copy an image extension into a table cell at a given row and
/// column.  The table must have already been created.  If the "colname"
/// column exists, it will be used, otherwise a new column will be created
/// in the table.
///
/// The "copykeyflag" parameter controls which keywords to copy from the
/// input image to the output table header (with any appropriate translation).
///
/// copykeyflag = 0  -- no keywords will be copied
/// copykeyflag = 1  -- essentially all keywords will be copied
/// copykeyflag = 2  -- copy only the WCS related keywords
///
/// This routine was written by Craig Markwardt, GSFC
///
/// # Parameters
///
/// * `fptr`        — (I) pointer to input image extension
/// * `newptr`      — (I) pointer to output table
/// * `colname`     — (I) name of column containing the image
/// * `rownum`      — (I) number of the row containing the image
/// * `copykeyflag` — (I) controls which keywords to copy
/// * `status`      — (IO) error status
pub fn fits_copy_image2cell_safe(
    fptr: &mut fitsfile,
    newptr: &mut fitsfile,
    colname: &[c_char],
    rownum: c_long,
    copykeyflag: c_int,
    status: &mut c_int,
) -> c_int {
    let mut buffer: [u8; 30000] = [0; 30000];
    let mut hdutype: c_int = 0;
    let mut colnum: c_int = 0;
    let typecode: c_int;
    let mut bitpix: c_int = 0;
    let mut naxis: c_int = 0;
    let mut ncols: c_int = 0;
    let tformchar: c_char;
    let mut tform: [c_char; 20] = [0; 20];
    let imgstart: LONGLONG;
    let mut naxes: [LONGLONG; 9] = [0; 9];
    let mut nbytes: LONGLONG;
    let mut repeat: LONGLONG;
    let mut ntodo: LONGLONG;
    let mut firstbyte: LONGLONG;

    let npat: c_int;

    let mut naxis1: c_int = 0;
    let mut naxes1: [LONGLONG; 9] = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut repeat1: LONGLONG = 0;
    let mut width1: LONGLONG = 0;
    let mut typecode1: c_int = 0;
    let dummy: [u8; 1] = [0];

    let mut headstart: LONGLONG = 0;
    let mut datastart: LONGLONG = 0;
    let mut dataend: LONGLONG = 0;

    /* Image-to-table keyword translation table  */
    /*                        INPUT      OUTPUT  */
    /*                       01234567   01234567 */
    let mut patterns: [[&CStr; 2]; 43] = [
        [c"BSCALE", c"TSCALn"], /* Standard FITS keywords */
        [c"BZERO", c"TZEROn"],
        [c"BUNIT", c"TUNITn"],
        [c"BLANK", c"TNULLn"],
        [c"DATAMIN", c"TDMINn"],
        [c"DATAMAX", c"TDMAXn"],
        [c"CTYPEi", c"iCTYPn"], /* Coordinate labels */
        [c"CTYPEia", c"iCTYna"],
        [c"CUNITi", c"iCUNIn"], /* Coordinate units */
        [c"CUNITia", c"iCUNna"],
        [c"CRVALi", c"iCRVLn"], /* WCS keywords */
        [c"CRVALia", c"iCRVna"],
        [c"CDELTi", c"iCDLTn"],
        [c"CDELTia", c"iCDEna"],
        [c"CRPIXj", c"jCRPXn"],
        [c"CRPIXja", c"jCRPna"],
        [c"PCi_ja", c"ijPCna"],
        [c"CDi_ja", c"ijCDna"],
        [c"PVi_ma", c"iVn_ma"],
        [c"PSi_ma", c"iSn_ma"],
        [c"WCSAXESa", c"WCAXna"],
        [c"WCSNAMEa", c"WCSNna"],
        [c"CRDERia", c"iCRDna"],
        [c"CSYERia", c"iCSYna"],
        [c"CROTAi", c"iCROTn"],
        [c"LONPOLEa", c"LONPna"],
        [c"LATPOLEa", c"LATPna"],
        [c"EQUINOXa", c"EQUIna"],
        [c"MJD-OBS", c"MJDOBn"],
        [c"MJD-AVG", c"MJDAn"],
        [c"RADESYSa", c"RADEna"],
        [c"CNAMEia", c"iCNAna"],
        [c"DATE-AVG", c"DAVGn"],
        [c"NAXISi", c"-"], /* Remove structural keywords*/
        [c"PCOUNT", c"-"],
        [c"GCOUNT", c"-"],
        [c"EXTEND", c"-"],
        [c"EXTNAME", c"-"],
        [c"EXTVER", c"-"],
        [c"EXTLEVEL", c"-"],
        [c"CHECKSUM", c"-"],
        [c"DATASUM", c"-"],
        [c"*", c"+"], /* copy all other keywords */
    ];

    if *status > 0 {
        return *status;
    }

    /* The C returns NULL_INPUT_PTR here if fptr or newptr is NULL; a &mut
    reference can never be null, so that check is unreachable and omitted. */

    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        ffpmsg_str("could not get input HDU type");
        return *status;
    }

    if hdutype != IMAGE_HDU {
        ffpmsg_str("The input extension is not an image.");
        ffpmsg_str(" Cannot open the image.");
        *status = NOT_IMAGE;
        return *status;
    }

    if ffghdt_safe(newptr, &mut hdutype, status) > 0 {
        ffpmsg_str("could not get output HDU type");
        return *status;
    }

    if hdutype != BINARY_TBL {
        ffpmsg_str("The output extension is not a table.");
        *status = NOT_BTABLE;
        return *status;
    }

    if ffgiprll_safe(
        fptr,
        9,
        Some(&mut bitpix),
        Some(&mut naxis),
        Some(&mut naxes),
        status,
    ) > 0
    {
        ffpmsg_str("Could not read image parameters.");
        return *status;
    }

    /* Determine total number of pixels in the image */
    repeat = 1;
    for ii in 0..(naxis as usize) {
        repeat *= naxes[ii];
    }

    /* Determine the TFORM value for the table cell */
    if bitpix == BYTE_IMG {
        typecode = TBYTE;
        tformchar = bb(b'B');
        nbytes = repeat;
    } else if bitpix == SHORT_IMG {
        typecode = TSHORT;
        tformchar = bb(b'I');
        nbytes = repeat * 2;
    } else if bitpix == LONG_IMG {
        typecode = TLONG;
        tformchar = bb(b'J');
        nbytes = repeat * 4;
    } else if bitpix == FLOAT_IMG {
        typecode = TFLOAT;
        tformchar = bb(b'E');
        nbytes = repeat * 4;
    } else if bitpix == DOUBLE_IMG {
        typecode = TDOUBLE;
        tformchar = bb(b'D');
        nbytes = repeat * 8;
    } else if bitpix == LONGLONG_IMG {
        typecode = TLONGLONG;
        tformchar = bb(b'K');
        nbytes = repeat * 8;
    } else {
        ffpmsg_str("Error: the image has an invalid datatype.");
        *status = BAD_BITPIX;
        return *status;
    }

    /* get column number */
    ffpmrk_safe();
    ffgcno_safe(newptr, CASEINSEN as c_int, colname, &mut colnum, status);
    ffcmrk_safe();

    /* Column does not exist; create it */
    if *status != 0 {
        *status = 0;
        int_snprintf!(
            &mut tform,
            20,
            "{:.0}{}",
            repeat as f64,
            tformchar as u8 as char
        );
        ffgncl_safe(newptr, &mut ncols, status);
        colnum = ncols + 1;
        fficol_safe(newptr, colnum, colname, &tform, status);
        ffptdmll_safe(newptr, colnum, naxis, &naxes, status);

        if *status != 0 {
            ffpmsg_str("Could not insert new column into output table.");
            return *status;
        }
    } else {
        ffgtdmll_safe(newptr, colnum, 9, &mut naxis1, &mut naxes1, status);
        if *status > 0 || naxis != naxis1 {
            ffpmsg_str("Input image dimensions and output table cell dimensions do not match.");
            *status = BAD_DIMEN;
            return *status;
        }
        for ii in 0..(naxis as usize) {
            if naxes[ii] != naxes1[ii] {
                ffpmsg_str("Input image dimensions and output table cell dimensions do not match.");
                *status = BAD_DIMEN;
                return *status;
            }
        }

        ffgtclll_safe(
            newptr,
            colnum,
            Some(&mut typecode1),
            Some(&mut repeat1),
            Some(&mut width1),
            status,
        );
        if *status > 0 || typecode1 != typecode || repeat1 != repeat {
            ffpmsg_str("Input image data type does not match output table cell type.");
            *status = BAD_TFORM;
            return *status;
        }
    }

    /* copy keywords from input image to output table, if required */

    if copykeyflag != 0 {
        npat = patterns.len() as c_int;

        if copykeyflag == 2 {
            /* copy only the WCS-related keywords */
            patterns[(npat - 1) as usize][1] = c"-";
        }

        /* The 3rd parameter value = 5 means skip the first 4 keywords in the image */
        fits_translate_keywords_safe(fptr, newptr, 5, &patterns, npat, colnum, 0, 0, status);
    }

    /* Here is all the code to compute offsets:
     *     * byte offset from start of row to column (dest table)
     *     * byte offset from start of file to image data (source image)
     */

    /* Force the writing of the row of the table by writing the last byte of
    the array, which grows the table, and/or shifts following extensions */
    ffpcl_safe(
        newptr,
        TBYTE,
        colnum,
        rownum as LONGLONG,
        repeat,
        1,
        &dummy,
        status,
    );

    /* byte offset within the row to the start of the image column */
    {
        let colptr = newptr.Fptr.get_tableptr_as_slice(); /* point to first column */
        /* offset to correct column structure */
        firstbyte = colptr[(colnum - 1) as usize].tbcol + 1;
    }

    /* get starting address of input image to be read */
    ffghadll_safe(
        fptr,
        Some(&mut headstart),
        Some(&mut datastart),
        Some(&mut dataend),
        status,
    );
    imgstart = datastart;

    /* The C formats a "HISTORY  Table column ... copied from image" card here,
    and below it a second HISTORY line holding the input file name and HDU
    number, but both ffprec() calls that would write them are commented out in
    the C (writing HISTORY is left up to the caller).  Neither string is used
    for anything else, so both are omitted here, as in fits_copy_cell2image. */

    /* the use of ffread routine, below, requires that any 'dirty' */
    /* buffers in memory be flushed back to the file first */

    ffflsh_safe(fptr, false, status);

    /* move to the first byte of the input image */
    ffmbyt_safe(fptr, imgstart, TRUE as c_int, status);

    ntodo = cmp::min(30000, nbytes);
    ffgbyt(fptr, ntodo, &mut buffer, status); /* read input image */
    ffptbb_safe(
        newptr,
        rownum as LONGLONG,
        firstbyte,
        ntodo,
        &buffer,
        status,
    ); /* write to table */

    nbytes -= ntodo;
    firstbyte += ntodo;

    /* read any additional bytes with low-level ffread routine, for speed */
    while nbytes != 0 && *status <= 0 {
        ntodo = cmp::min(30000, nbytes);
        ffread(&fptr.Fptr, ntodo as c_long, &mut buffer, status);
        ffptbb_safe(
            newptr,
            rownum as LONGLONG,
            firstbyte,
            ntodo,
            &buffer,
            status,
        );
        nbytes -= ntodo;
        firstbyte += ntodo;
    }

    /* Re-scan the header so that CFITSIO knows about all the new keywords */
    ffrdef_safe(newptr, status);

    *status
}

/// copies an image section from the input file to a new output file.
/// Any HDUs preceding or following the image are also copied to the
/// output file.
///
/// # Parameters
///
/// * `fptr`    — (IO) pointer to input image; on output it
/// * `outfile` — (I) name for output file
/// * `expr`    — (I) Image section expression
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_select_image_section(
    fptr: *mut Option<Box<fitsfile>>,
    /*      points to the new subimage */
    outfile: *const c_char,
    expr: *const c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        raw_to_slice!(outfile);
        raw_to_slice!(expr);

        fits_select_image_section_safe(fptr, outfile, expr, status)
    }
}

/// copies an image section from the input file to a new output file.
/// Any HDUs preceding or following the image are also copied to the
/// output file.
///
/// # Parameters
///
/// * `fptr`    — (IO) pointer to input image; on output it
/// * `outfile` — (I) name for output file
/// * `expr`    — (I) Image section expression
pub fn fits_select_image_section_safe(
    fptr: &mut Option<Box<fitsfile>>,
    /*      points to the new subimage */
    outfile: &[c_char],
    expr: &[c_char],
    status: &mut c_int,
) -> c_int {
    let mut newptr: Option<Box<fitsfile>> = None;
    let mut ii: c_int;
    let mut hdunum: c_int = 0;

    /* create new empty file to hold the image section */
    if ffinit_safe(&mut newptr, outfile, status) > 0 {
        ffpmsg_str("failed to create output file for image section:");
        ffpmsg_slice(outfile);
        return *status;
    }

    ffghdn_safe(fptr.as_deref_mut().unwrap(), &mut hdunum); /* current HDU number in input file */

    /* copy all preceding extensions to the output file, if 'only_one' flag not set */
    if fptr.as_deref().unwrap().Fptr.only_one == 0 {
        for ii in 1..hdunum {
            ffmahd_safe(fptr.as_deref_mut().unwrap(), ii, None, status);
            if ffcopy_safe(
                fptr.as_deref_mut().unwrap(),
                newptr.as_deref_mut().unwrap(),
                0,
                status,
            ) > 0
            {
                ffclos_safe(newptr.take().unwrap(), status);
                return *status;
            }
        }

        /* move back to the original HDU position */
        ffmahd_safe(fptr.as_deref_mut().unwrap(), hdunum, None, status);
    }

    if fits_copy_image_section_safe(
        fptr.as_deref_mut().unwrap(),
        newptr.as_deref_mut().unwrap(),
        expr,
        status,
    ) > 0
    {
        ffclos_safe(newptr.take().unwrap(), status);
        return *status;
    }

    /* copy any remaining HDUs to the output file, if 'only_one' flag not set */

    if fptr.as_deref().unwrap().Fptr.only_one == 0 {
        /* C: for (ii = hdunum + 1; 1; ii++); `ii` is read after the loop, so
        the increment stays at the bottom to preserve the C's exit value */
        ii = hdunum + 1;
        loop {
            if ffmahd_safe(fptr.as_deref_mut().unwrap(), ii, None, status) > 0 {
                break;
            }

            ffcopy_safe(
                fptr.as_deref_mut().unwrap(),
                newptr.as_deref_mut().unwrap(),
                0,
                status,
            );
            ii += 1;
        }

        if *status == END_OF_FILE {
            *status = 0; /* got the expected EOF error; reset = 0  */
        } else if *status > 0 {
            ffclos_safe(newptr.take().unwrap(), status);
            return *status;
        }
    } else {
        ii = hdunum + 1; /* this value of ii is required below */
    }

    /* close the original file and return ptr to the new image */
    ffclos_safe(fptr.take().unwrap(), status);

    *fptr = newptr; /* reset the pointer to the new table */

    /* move back to the image subsection */
    if ii - 1 != hdunum {
        ffmahd_safe(fptr.as_deref_mut().unwrap(), hdunum, None, status);
    } else {
        /* may have to reset BSCALE and BZERO pixel scaling, */
        /* since the keywords were previously turned off */

        if ffrdef_safe(fptr.as_deref_mut().unwrap(), status) > 0 {
            /* the C closes *fptr here and returns with it left dangling;
            ffclos_safe consumes the Box, so *fptr is left None instead */
            ffclos_safe(fptr.take().unwrap(), status);
            return *status;
        }
    }

    *status
}

/// copies an image section from the input file to a new output HDU
///
/// # Parameters
///
/// * `fptr`   — (I) pointer to input image
/// * `newptr` — (I) pointer to output image
/// * `expr`   — (I) Image section expression
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_copy_image_section(
    fptr: *mut fitsfile,
    newptr: *mut fitsfile,
    expr: *const c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let newptr = newptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        raw_to_slice!(expr);

        fits_copy_image_section_safe(fptr, newptr, expr, status)
    }
}

/// copies an image section from the input file to a new output HDU
///
/// # Parameters
///
/// * `fptr`   — (I) pointer to input image
/// * `newptr` — (I) pointer to output image
/// * `expr`   — (I) Image section expression
pub fn fits_copy_image_section_safe(
    fptr: &mut fitsfile,
    newptr: &mut fitsfile,
    expr: &[c_char],
    status: &mut c_int,
) -> c_int {
    let mut bitpix: c_int = 0;
    let mut naxis: c_int = 0;
    let mut numkeys: c_int = 0;
    let mut naxes: [c_long; 9] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
    let mut smin: c_long = 0;
    let mut smax: c_long = 0;
    let mut sinc: c_long = 0;
    let mut fpixels: [c_long; 9] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
    let mut lpixels: [c_long; 9] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
    let mut incs: [c_long; 9] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut tstatus: c_int;
    let mut anynull: c_int = 0;
    let minrow: c_long;
    let maxrow: c_long;
    let minslice: c_long;
    let maxslice: c_long;
    let mincube: c_long;
    let maxcube: c_long;
    let mut firstpix: c_long;
    let ncubeiter: c_long;
    let nsliceiter: c_long;
    let nrowiter: c_long;
    let mut klen: usize;
    let mut outnaxes: [c_long; 9] = [0; 9];
    let outsize: c_long;
    let buffsize: c_long;
    let mut crpix: f64 = 0.0;
    let mut cdelt: f64 = 0.0;

    if *status > 0 {
        return *status;
    }

    /* get the size of the input image */
    ffgidt_safe(fptr, &mut bitpix, status);
    ffgidm_safe(fptr, &mut naxis, status);
    /* DEVIATION (upstream bug): the C passes `naxis` -- up to 99, whatever the
    file's NAXIS says -- as the length of the 9-element `naxes` array, so an
    image with NAXIS >= 10 overflows that stack array here, before the
    `naxis > 4` check below ever runs.  Clamp to the array length instead: for
    every naxis the C accepts (1..=4) this is identical, and the oversized
    cases now reach the BAD_NAXIS error the C already intends to return. */
    if ffgisz_safe(
        fptr,
        cmp::min(naxis, naxes.len() as c_int),
        &mut naxes,
        status,
    ) > 0
    {
        return *status;
    }

    if !(1..=4).contains(&naxis) {
        ffpmsg_str("Input image either had NAXIS = 0 (NULL image) or has > 4 dimensions");
        *status = BAD_NAXIS;
        return *status;
    }

    /* create output image with same size and type as the input image */
    /*  Will update the size later */
    ffcrim_safe(newptr, bitpix, naxis, &naxes, status);

    /* copy all other non-structural keywords from the input to output file */
    ffghsp_safe(fptr, Some(&mut numkeys), None, status);

    for nkey in 4..=numkeys {
        /* skip the first few keywords */
        ffgrec_safe(fptr, nkey, Some(&mut card), status);

        if ffgkcl_safe(&card) > TYP_CMPRS_KEY {
            /* write the record to the output file */
            ffprec_safe(newptr, &card, status);
        }
    }

    if *status > 0 {
        ffpmsg_str("error copying header from input image to output image");
        return *status;
    }

    /* parse the section specifier to get min, max, and inc for each axis */
    /* and the size of each output image axis */

    /* the C walks a `char *cptr` along `expr`; track the offset instead */
    let mut cptr: usize = 0;
    for ii in 0..(naxis as usize) {
        if fits_get_section_range_safe(expr, &mut cptr, &mut smin, &mut smax, &mut sinc, status) > 0
        {
            ffpmsg_str("error parsing the following image section specifier:");
            ffpmsg_slice(expr);
            return *status;
        }

        if smax == 0 {
            smax = naxes[ii]; /* use whole axis  by default */
        } else if smin == 0 {
            smin = naxes[ii]; /* use inverted whole axis */
        }

        if smin > naxes[ii] || smax > naxes[ii] {
            ffpmsg_str("image section exceeds dimensions of input image:");
            ffpmsg_slice(expr);
            *status = BAD_NAXIS;
            return *status;
        }

        fpixels[ii] = smin;
        lpixels[ii] = smax;
        incs[ii] = sinc;

        let lllength: LONGLONG = ((smax - smin).abs() / sinc) as LONGLONG + 1;
        if lllength > c_long::MAX as LONGLONG {
            ffpmsg_str("image range exceeds LONG_MAX limit");
            ffpmsg_slice(expr);
            *status = NUM_OVERFLOW;
            return *status;
        }
        outnaxes[ii] = lllength as c_long;

        /* modify the NAXISn keyword */
        ffkeyn_safe(cs!(c"NAXIS"), ii as c_int + 1, &mut keyname, status);
        ffmkyj_safe(newptr, &keyname, outnaxes[ii] as LONGLONG, None, status);

        /* modify the WCS keywords if necessary */

        if fpixels[ii] != 1 || incs[ii] != 1 {
            for kk in -1..26 {
                /* modify any alternate WCS keywords */

                /* read the CRPIXn keyword if it exists in the input file */
                ffkeyn_safe(cs!(c"CRPIX"), ii as c_int + 1, &mut keyname, status);

                if kk != -1 {
                    klen = strlen_safe(&keyname);
                    keyname[klen] = bb(b'A') + kk as c_char;
                    keyname[klen + 1] = 0;
                }

                tstatus = 0;
                if ffgky_safe(
                    fptr,
                    KeywordDatatypeMut::TDOUBLE(&mut crpix),
                    &keyname,
                    None,
                    &mut tstatus,
                ) == 0
                {
                    /* calculate the new CRPIXn value */
                    if fpixels[ii] <= lpixels[ii] {
                        crpix = (crpix - (fpixels[ii] as f64)) / incs[ii] as f64 + 1.0;
                        /*  crpix = (crpix - (fpixels[ii] - 1.0) - .5) / incs[ii] + 0.5; */
                    } else {
                        crpix = (fpixels[ii] as f64 - crpix) / incs[ii] as f64 + 1.0;
                        /* crpix = (fpixels[ii] - (crpix - 1.0) - .5) / incs[ii] + 0.5; */
                    }

                    /* modify the value in the output file */
                    ffmkyd_safe(newptr, &keyname, crpix, 15, None, status);

                    if incs[ii] != 1 || fpixels[ii] > lpixels[ii] {
                        /* read the CDELTn keyword if it exists in the input file */
                        ffkeyn_safe(cs!(c"CDELT"), ii as c_int + 1, &mut keyname, status);

                        if kk != -1 {
                            klen = strlen_safe(&keyname);
                            keyname[klen] = bb(b'A') + kk as c_char;
                            keyname[klen + 1] = 0;
                        }

                        tstatus = 0;
                        if ffgky_safe(
                            fptr,
                            KeywordDatatypeMut::TDOUBLE(&mut cdelt),
                            &keyname,
                            None,
                            &mut tstatus,
                        ) == 0
                        {
                            /* calculate the new CDELTn value */
                            if fpixels[ii] <= lpixels[ii] {
                                cdelt *= incs[ii] as f64;
                            } else {
                                cdelt *= -(incs[ii] as f64);
                            }

                            /* modify the value in the output file */
                            ffmkyd_safe(newptr, &keyname, cdelt, 15, None, status);
                        }

                        /* modify the CDi_j keywords if they exist in the input file */

                        ffkeyn_safe(cs!(c"CD1_"), ii as c_int + 1, &mut keyname, status);

                        if kk != -1 {
                            klen = strlen_safe(&keyname);
                            keyname[klen] = bb(b'A') + kk as c_char;
                            keyname[klen + 1] = 0;
                        }

                        for jj in 0..9 {
                            /* look for up to 9 dimensions */
                            keyname[2] = bb(b'1') + jj as c_char;

                            tstatus = 0;
                            if ffgky_safe(
                                fptr,
                                KeywordDatatypeMut::TDOUBLE(&mut cdelt),
                                &keyname,
                                None,
                                &mut tstatus,
                            ) == 0
                            {
                                /* calculate the new CDi_j value */
                                if fpixels[ii] <= lpixels[ii] {
                                    cdelt *= incs[ii] as f64;
                                } else {
                                    cdelt *= -(incs[ii] as f64);
                                }

                                /* modify the value in the output file */
                                ffmkyd_safe(newptr, &keyname, cdelt, 15, None, status);
                            }
                        }
                    } /* end of if (incs[ii]... loop */
                } /* end of fits_read_key loop */
            } /* end of for (kk  loop */
        }
    } /* end of main NAXIS loop */

    if ffrdef_safe(newptr, status) > 0 {
        /* force the header to be scanned */
        return *status;
    }

    /* turn off any scaling of the pixel values */
    ffpscl_safe(fptr, 1.0, 0.0, status);
    ffpscl_safe(newptr, 1.0, 0.0, status);

    /* to reduce memory foot print, just read/write image 1 row at a time */

    outsize = outnaxes[0];
    buffsize = (bitpix.abs() / 8) as c_long * outsize;

    /* allocate memory for the image row.  The C mallocs `buffsize` bytes as a
    double* and casts that pointer to the pixel type in use, so allocate f64
    units here (the widest pixel type, and the alignment the C pointer had)
    and reinterpret them per bitpix below. */
    let mut buffer: Vec<f64> = Vec::new();
    let nbuffelem = (buffsize as usize).div_ceil(mem::size_of::<f64>());
    if buffer.try_reserve_exact(nbuffelem).is_err() {
        ffpmsg_str("fits_copy_image_section: no memory for image section");
        *status = MEMORY_ALLOCATION;
        return *status;
    }
    buffer.resize(nbuffelem, 0.0);

    /* read the image section then write it to the output file */

    minrow = fpixels[1];
    maxrow = lpixels[1];
    nrowiter = (maxrow - minrow).abs() / incs[1] + 1;

    minslice = fpixels[2];
    maxslice = lpixels[2];
    nsliceiter = (maxslice - minslice).abs() / incs[2] + 1;

    mincube = fpixels[3];
    maxcube = lpixels[3];
    ncubeiter = (maxcube - mincube).abs() / incs[3] + 1;

    firstpix = 1;
    for kiter in 0..ncubeiter {
        if mincube > maxcube {
            fpixels[3] = mincube - (kiter * incs[3]);
        } else {
            fpixels[3] = mincube + (kiter * incs[3]);
        }

        lpixels[3] = fpixels[3];

        for jiter in 0..nsliceiter {
            if minslice > maxslice {
                fpixels[2] = minslice - (jiter * incs[2]);
            } else {
                fpixels[2] = minslice + (jiter * incs[2]);
            }

            lpixels[2] = fpixels[2];

            for iiter in 0..nrowiter {
                if minrow > maxrow {
                    fpixels[1] = minrow - (iiter * incs[1]);
                } else {
                    fpixels[1] = minrow + (iiter * incs[1]);
                }

                lpixels[1] = fpixels[1];

                if bitpix == 8 {
                    let buff: &mut [u8] = &mut cast_slice_mut(&mut buffer)[..outsize as usize];

                    ffgsvb_safe(
                        fptr,
                        1,
                        naxis,
                        &naxes,
                        &fpixels,
                        &lpixels,
                        &incs,
                        0,
                        buff,
                        Some(&mut anynull),
                        status,
                    );

                    ffpprb_safe(
                        newptr,
                        1,
                        firstpix as LONGLONG,
                        outsize as LONGLONG,
                        buff,
                        status,
                    );
                } else if bitpix == 16 {
                    let buff: &mut [c_short] = &mut cast_slice_mut(&mut buffer)[..outsize as usize];

                    ffgsvi_safe(
                        fptr,
                        1,
                        naxis,
                        &naxes,
                        &fpixels,
                        &lpixels,
                        &incs,
                        0,
                        buff,
                        Some(&mut anynull),
                        status,
                    );

                    ffppri_safe(
                        newptr,
                        1,
                        firstpix as LONGLONG,
                        outsize as LONGLONG,
                        buff,
                        status,
                    );
                } else if bitpix == 32 {
                    let buff: &mut [c_int] = &mut cast_slice_mut(&mut buffer)[..outsize as usize];

                    ffgsvk_safe(
                        fptr,
                        1,
                        naxis,
                        &naxes,
                        &fpixels,
                        &lpixels,
                        &incs,
                        0,
                        buff,
                        Some(&mut anynull),
                        status,
                    );

                    ffpprk_safe(
                        newptr,
                        1,
                        firstpix as LONGLONG,
                        outsize as LONGLONG,
                        buff,
                        status,
                    );
                } else if bitpix == -32 {
                    let buff: &mut [f32] = &mut cast_slice_mut(&mut buffer)[..outsize as usize];

                    ffgsve_safe(
                        fptr,
                        1,
                        naxis,
                        &naxes,
                        &fpixels,
                        &lpixels,
                        &incs,
                        FLOATNULLVALUE,
                        buff,
                        Some(&mut anynull),
                        status,
                    );

                    ffppne_safe(
                        newptr,
                        1,
                        firstpix as LONGLONG,
                        outsize as LONGLONG,
                        buff,
                        FLOATNULLVALUE,
                        status,
                    );
                } else if bitpix == -64 {
                    let buff: &mut [f64] = &mut buffer[..outsize as usize];

                    ffgsvd_safe(
                        fptr,
                        1,
                        naxis,
                        &naxes,
                        &fpixels,
                        &lpixels,
                        &incs,
                        DOUBLENULLVALUE,
                        buff,
                        Some(&mut anynull),
                        status,
                    );

                    ffppnd_safe(
                        newptr,
                        1,
                        firstpix as LONGLONG,
                        outsize as LONGLONG,
                        buff,
                        DOUBLENULLVALUE,
                        status,
                    );
                } else if bitpix == 64 {
                    let buff: &mut [LONGLONG] =
                        &mut cast_slice_mut(&mut buffer)[..outsize as usize];

                    ffgsvjj_safe(
                        fptr,
                        1,
                        naxis,
                        &naxes,
                        &fpixels,
                        &lpixels,
                        &incs,
                        0,
                        buff,
                        Some(&mut anynull),
                        status,
                    );

                    ffpprjj_safe(
                        newptr,
                        1,
                        firstpix as LONGLONG,
                        outsize as LONGLONG,
                        buff,
                        status,
                    );
                }

                firstpix += outsize;
            }
        }
    }

    /* the C frees `buffer` here; the Vec is dropped on return instead */

    if *status > 0 {
        ffpmsg_str("fits_copy_image_section: error copying image section");
        return *status;
    }

    *status
}

/// Parse the input image section specification string, returning
/// the  min, max and increment values.
/// Typical string =   "1:512:2"  or "1:512"
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_get_section_range(
    ptr: *mut *mut c_char,
    secmin: *mut c_long,
    secmax: *mut c_long,
    incre: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let ptr = ptr.as_mut().expect(NULL_MSG);
        let secmin = secmin.as_mut().expect(NULL_MSG);
        let secmax = secmax.as_mut().expect(NULL_MSG);
        let incre = incre.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        /* Bridge the C `char **ptr` to the safe slice + index form: parse the
        NUL-terminated remainder, then advance the caller's pointer past the
        characters that were consumed. */
        let p: &[c_char] = cast_slice(CStr::from_ptr(*ptr).to_bytes_with_nul());
        let mut ptr_index = 0usize;
        let r = fits_get_section_range_safe(p, &mut ptr_index, secmin, secmax, incre, status);
        *ptr = (*ptr).add(ptr_index);
        r
    }
}

/// Parse the input image section specification string, returning
/// the  min, max and increment values.
/// Typical string =   "1:512:2"  or "1:512"
///
/// # Parameters
///
/// * `ptr`       — (I) the image section specifier string
/// * `ptr_index` — (IO) current parse offset into ptr; advanced
pub fn fits_get_section_range_safe(
    ptr: &[c_char],
    ptr_index: &mut usize,
    secmin: &mut c_long,
    secmax: &mut c_long,
    incre: &mut c_long,
    status: &mut c_int,
) -> c_int {
    let mut isanumber: c_int = 0;
    let mut token: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut tstbuff: Option<Vec<c_char>> = None;
    let mut sublen = 0usize; /* characters consumed by the latest token */

    // C 'atol': parse a leading long, ignoring trailing junk; 0 on failure.
    let atol =
        |tok: &[c_char]| -> c_long { strtol_safe::<c_long>(tok).map(|(v, _)| v).unwrap_or(0) };

    if *status > 0 {
        return *status;
    }

    /* get 1st token. fits_get_token2 reports how many characters it consumed
    (leading blanks + token) via sublen, so advance ptr_index by that. */
    let mut slen = fits_get_token2_safe(
        &ptr[*ptr_index..],
        &mut sublen,
        cs!(c" ,:"),
        &mut tstbuff,
        Some(&mut isanumber),
        status,
    );
    *ptr_index += sublen;
    if slen == 0 {
        /* support [:2,:2] type syntax, where the leading * is implied */
        strcpy_safe(&mut token, cs!(c"*"));
    } else {
        let tstbuff = tstbuff.as_deref().unwrap_or_default();
        if strlen_safe(tstbuff) > FLEN_VALUE - 1 {
            ffpmsg_str("Error: image section string too long (fits_get_section_range)");
            *status = URL_PARSE_ERROR;
            return *status;
        }
        strcpy_safe(&mut token, tstbuff);
    }

    if token[0] == bb(b'*') {
        /* wild card means to use the whole range */
        *secmin = 1;
        *secmax = 0;
    } else if token[0] == bb(b'-') && token[1] == bb(b'*') {
        /* invert the whole range */
        *secmin = 0;
        *secmax = 1;
    } else {
        if slen == 0 || isanumber == 0 || ptr[*ptr_index] != bb(b':') {
            *status = URL_PARSE_ERROR;
            return *status;
        }

        /* the token contains the min value */
        *secmin = atol(&token);

        *ptr_index += 1; /* skip the colon between the min and max values */
        slen = fits_get_token2_safe(
            &ptr[*ptr_index..],
            &mut sublen,
            cs!(c" ,:"),
            &mut tstbuff,
            Some(&mut isanumber),
            status,
        ); /* get token */
        *ptr_index += sublen;
        if slen == 0 || isanumber == 0 {
            *status = URL_PARSE_ERROR;
            return *status;
        }
        let tstbuff = tstbuff.as_deref().unwrap_or_default();
        if strlen_safe(tstbuff) > FLEN_VALUE - 1 {
            ffpmsg_str("Error: image section string too long (fits_get_section_range)");
            *status = URL_PARSE_ERROR;
            return *status;
        }
        strcpy_safe(&mut token, tstbuff);

        /* the token contains the max value */
        *secmax = atol(&token);
    }

    if ptr[*ptr_index] == bb(b':') {
        *ptr_index += 1; /* skip the colon between the max and incre values */
        slen = fits_get_token2_safe(
            &ptr[*ptr_index..],
            &mut sublen,
            cs!(c" ,"),
            &mut tstbuff,
            Some(&mut isanumber),
            status,
        ); /* get token */
        *ptr_index += sublen;
        if slen == 0 || isanumber == 0 {
            *status = URL_PARSE_ERROR;
            return *status;
        }
        let tstbuff = tstbuff.as_deref().unwrap_or_default();
        if strlen_safe(tstbuff) > FLEN_VALUE - 1 {
            ffpmsg_str("Error: image section string too long (fits_get_section_range)");
            *status = URL_PARSE_ERROR;
            return *status;
        }
        strcpy_safe(&mut token, tstbuff);

        *incre = atol(&token);
    } else {
        *incre = 1; /* default increment if none is supplied */
    }

    if ptr[*ptr_index] == bb(b',') {
        *ptr_index += 1;
    }

    while ptr[*ptr_index] == bb(b' ') {
        /* skip any trailing blanks */
        *ptr_index += 1;
    }

    if *secmin < 0 || *secmax < 0 || *incre < 1 {
        *status = URL_PARSE_ERROR;
    }

    *status
}

/// # Parameters
///
/// * `fptr`    — (IO) pointer to input table; on output it
/// * `outfile` — (I) name for output file
/// * `expr`    — (I) Boolean expression
pub(crate) fn ffselect_table(
    fptr: &mut Option<Box<fitsfile>>,
    /*      points to the new selected rows table */
    outfile: &[c_char; FLEN_FILENAME],
    expr: &[c_char; FLEN_FILENAME],
    status: &mut c_int,
) -> c_int {
    let mut newptr: Option<Box<fitsfile>> = None;
    let mut ii: c_int;
    let mut hdunum: c_int = 0;

    if outfile[0] != 0 {
        /* create new empty file in to hold the selected rows */
        if ffinit_safe(&mut newptr, outfile, status) > 0 {
            ffpmsg_str("failed to create file for selected rows from input table");
            ffpmsg_slice(outfile);
            return *status;
        }

        ffghdn_safe(fptr.as_deref_mut().unwrap(), &mut hdunum); /* current HDU number in input file */

        /* copy all preceding extensions to the output file, if the 'only_one' flag is not set */
        if fptr.as_deref().unwrap().Fptr.only_one == 0 {
            for ii in 1..hdunum {
                ffmahd_safe(fptr.as_deref_mut().unwrap(), ii, None, status);
                if ffcopy_safe(
                    fptr.as_deref_mut().unwrap(),
                    newptr.as_deref_mut().unwrap(),
                    0,
                    status,
                ) > 0
                {
                    ffclos_safe(newptr.take().unwrap(), status);
                    return *status;
                }
            }
        } else {
            /* just copy the primary array */
            ffmahd_safe(fptr.as_deref_mut().unwrap(), 1, None, status);
            if ffcopy_safe(
                fptr.as_deref_mut().unwrap(),
                newptr.as_deref_mut().unwrap(),
                0,
                status,
            ) > 0
            {
                ffclos_safe(newptr.take().unwrap(), status);
                return *status;
            }
        }

        ffmahd_safe(fptr.as_deref_mut().unwrap(), hdunum, None, status);

        /* copy all the header keywords from the input to output file */
        if ffcphd_safe(
            fptr.as_deref_mut().unwrap(),
            newptr.as_deref_mut().unwrap(),
            status,
        ) > 0
        {
            ffclos_safe(newptr.take().unwrap(), status);
            return *status;
        }

        /* set number of rows = 0 */
        ffmkyj_safe(
            newptr.as_deref_mut().unwrap(),
            cs!(c"NAXIS2"),
            0,
            None,
            status,
        );
        {
            let n = newptr.as_deref_mut().unwrap();
            n.Fptr.numrows = 0;
            n.Fptr.origrows = 0;
        }

        if ffrdef_safe(newptr.as_deref_mut().unwrap(), status) > 0 {
            /* force the header to be scanned */
            ffclos_safe(newptr.take().unwrap(), status);
            return *status;
        }
    }
    /* else the C sets `newptr = *fptr` and deletes rows in place in the table;
    here `newptr` simply stays None and the same-file case is handled below */

    /* copy rows which satisfy the selection expression to the output table */
    /* or delete the nonqualifying rows if *fptr = newptr.   */
    let selstatus = match newptr.as_deref_mut() {
        Some(out) => ffsrow_safe(fptr.as_deref_mut().unwrap(), out, expr, status),
        None => {
            /* The C passes the same fitsfile* to fits_select_rows as both input
            and output, which deletes the non-qualifying rows in place. */
            ffsrow_inplace_safe(fptr.as_deref_mut().unwrap(), expr, status)
        }
    };

    if selstatus > 0 {
        if outfile[0] != 0 {
            ffclos_safe(newptr.take().unwrap(), status);
        }

        return *status;
    }

    if outfile[0] != 0 {
        /* copy any remaining HDUs to the output copy */

        if fptr.as_deref().unwrap().Fptr.only_one == 0 {
            /* C: for (ii = hdunum + 1; 1; ii++); the increment stays at the
            bottom, as in the C, although `ii` is not read after the loop */
            ii = hdunum + 1;
            loop {
                if ffmahd_safe(fptr.as_deref_mut().unwrap(), ii, None, status) > 0 {
                    break;
                }

                ffcopy_safe(
                    fptr.as_deref_mut().unwrap(),
                    newptr.as_deref_mut().unwrap(),
                    0,
                    status,
                );
                ii += 1;
            }

            if *status == END_OF_FILE {
                *status = 0; /* got the expected EOF error; reset = 0  */
            } else if *status > 0 {
                ffclos_safe(newptr.take().unwrap(), status);
                return *status;
            }
        } else {
            hdunum = 2;
        }

        /* close the original file and return ptr to the new image */
        ffclos_safe(fptr.take().unwrap(), status);

        *fptr = newptr; /* reset the pointer to the new table */

        /* move back to the selected table HDU */
        ffmahd_safe(fptr.as_deref_mut().unwrap(), hdunum, None, status);
    }

    *status
}

/// Parse the image compression specification that was give in square brackets
/// following the output FITS file name, as in these examples:
///
///   `myfile.fits[compress]`  - default Rice compression, row by row
///   myfile.fits[compress TYPE] -  the first letter of TYPE defines the
///                                 compression algorithm:
///                                  R = Rice
///                                  G = GZIP
///                                  H = HCOMPRESS
///                                  HS = HCOMPRESS (with smoothing)
///                  B - BZIP2
///                                  P = PLIO
///
///   myfile.fits[compress TYPE 100,100] - the numbers give the dimensions
///                                        of the compression tiles.  Default
///                                        is NAXIS1, 1, 1, ...
///
///   other optional parameters may be specified following a semi-colon
///
///   myfile.fits[compress; q 8.0]          q specifies the floating point
///   mufile.fits[compress TYPE; q -.0002]        quantization level;
///   myfile.fits[compress TYPE 100,100; q 10, s 25]  s specifies the HCOMPRESS
///                                                    integer scaling parameter
///
/// The compression parameters are saved in the fptr->Fptr structure for use
/// when writing FITS images.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `compspec` — (I) image compression specification
/// * `status`   — (IO) error status
pub(crate) fn ffparsecompspec(
    fptr: &mut fitsfile,
    compspec: &[c_char],
    status: &mut c_int,
) -> c_int {
    /* the C walks `compspec` with a `char *ptr1`; we use an index instead */
    let mut ptr1: usize;

    /* initialize with default values */
    let mut ii: usize;
    let mut compresstype: c_int = RICE_1;
    let mut smooth: c_int = 0;
    let mut quantize_method: c_int = SUBTRACTIVE_DITHER_1;
    let mut tilesize: [c_long; MAX_COMPRESS_DIM] = [0; MAX_COMPRESS_DIM];
    let mut qlevel: f32 = -99.;
    let mut scale: f32 = 0.;

    ptr1 = 0;
    while compspec[ptr1] == bb(b' ') {
        /* ignore leading blanks */
        ptr1 += 1;
    }

    if strncmp_safe(&compspec[ptr1..], cs!(c"compress"), 8) != 0
        && strncmp_safe(&compspec[ptr1..], cs!(c"COMPRESS"), 8) != 0
    {
        /* apparently this string does not specify compression parameters */
        *status = URL_PARSE_ERROR;
        return *status;
    }

    ptr1 += 8;
    while compspec[ptr1] == bb(b' ') {
        /* ignore leading blanks */
        ptr1 += 1;
    }

    /* ========================= */
    /* look for compression type */
    /* ========================= */

    if compspec[ptr1] == bb(b'r') || compspec[ptr1] == bb(b'R') {
        compresstype = RICE_1;
        while compspec[ptr1] != bb(b' ') && compspec[ptr1] != bb(b';') && compspec[ptr1] != 0 {
            ptr1 += 1;
        }
    } else if compspec[ptr1] == bb(b'g') || compspec[ptr1] == bb(b'G') {
        compresstype = GZIP_1;
        while compspec[ptr1] != bb(b' ') && compspec[ptr1] != bb(b';') && compspec[ptr1] != 0 {
            ptr1 += 1;
        }
    }
    /*
        else if (*ptr1 == 'b' || *ptr1 == 'B')
        {
            compresstype = BZIP2_1;
            while (*ptr1 != ' ' && *ptr1 != ';' && *ptr1 != '\0')
               ptr1++;
        }
    */
    else if compspec[ptr1] == bb(b'p') || compspec[ptr1] == bb(b'P') {
        compresstype = PLIO_1;
        while compspec[ptr1] != bb(b' ') && compspec[ptr1] != bb(b';') && compspec[ptr1] != 0 {
            ptr1 += 1;
        }
    } else if compspec[ptr1] == bb(b'h') || compspec[ptr1] == bb(b'H') {
        compresstype = HCOMPRESS_1;
        ptr1 += 1;
        if compspec[ptr1] == bb(b's') || compspec[ptr1] == bb(b'S') {
            smooth = 1; /* apply smoothing when uncompressing HCOMPRESSed image */
        }

        while compspec[ptr1] != bb(b' ') && compspec[ptr1] != bb(b';') && compspec[ptr1] != 0 {
            ptr1 += 1;
        }
    }

    /* ======================== */
    /* look for tile dimensions */
    /* ======================== */

    while compspec[ptr1] == bb(b' ') {
        /* ignore leading blanks */
        ptr1 += 1;
    }

    ii = 0;
    /* NOTE: the C loop bound was `ii < 9`, which would overflow the
    MAX_COMPRESS_DIM-element tilesize array; bound it to the array size */
    while isdigit_safe(compspec[ptr1]) && ii < MAX_COMPRESS_DIM {
        tilesize[ii] = strtol_safe::<c_long>(&compspec[ptr1..]).map_or(0, |(v, _)| v); /* read the integer value */
        ii += 1;

        while isdigit_safe(compspec[ptr1]) {
            /* skip over the integer */
            ptr1 += 1;
        }

        if compspec[ptr1] == bb(b',') {
            ptr1 += 1; /* skip over the comma */
        }

        while compspec[ptr1] == bb(b' ') {
            /* ignore leading blanks */
            ptr1 += 1;
        }
    }

    /* ========================================================= */
    /* look for semi-colon, followed by other optional parameters */
    /* ========================================================= */

    if compspec[ptr1] == bb(b';') {
        ptr1 += 1;
        while compspec[ptr1] == bb(b' ') {
            /* ignore leading blanks */
            ptr1 += 1;
        }

        while compspec[ptr1] != 0 {
            /* haven't reached end of string yet */

            if compspec[ptr1] == bb(b's') || compspec[ptr1] == bb(b'S') {
                /* this should be the HCOMPRESS "scale" parameter; default = 1 */

                ptr1 += 1;
                while compspec[ptr1] == bb(b' ') {
                    /* ignore leading blanks */
                    ptr1 += 1;
                }

                let mut loc: usize = 0;
                scale = strtod_safe(&compspec[ptr1..], &mut loc) as f32;
                ptr1 += loc;

                while compspec[ptr1] == bb(b' ') || compspec[ptr1] == bb(b',') {
                    /* skip over blanks or comma */
                    ptr1 += 1;
                }
            } else if compspec[ptr1] == bb(b'q') || compspec[ptr1] == bb(b'Q') {
                /* this should be the floating point quantization parameter */

                ptr1 += 1;
                if compspec[ptr1] == bb(b'z') || compspec[ptr1] == bb(b'Z') {
                    /* use the subtractive_dither_2 option */
                    quantize_method = SUBTRACTIVE_DITHER_2;
                    ptr1 += 1;
                } else if compspec[ptr1] == bb(b'0') {
                    /* do not dither */
                    quantize_method = NO_DITHER;
                    ptr1 += 1;
                }

                while compspec[ptr1] == bb(b' ') {
                    /* ignore leading blanks */
                    ptr1 += 1;
                }

                let mut loc: usize = 0;
                qlevel = strtod_safe(&compspec[ptr1..], &mut loc) as f32;
                ptr1 += loc;

                while compspec[ptr1] == bb(b' ') || compspec[ptr1] == bb(b',') {
                    /* skip over blanks or comma */
                    ptr1 += 1;
                }
            } else {
                *status = URL_PARSE_ERROR;
                return *status;
            }
        }
    }

    /* ================================= */
    /* finished parsing; save the values */
    /* ================================= */

    fits_set_compression_type_safe(fptr, compresstype, status);
    fits_set_tile_dim_safe(fptr, MAX_COMPRESS_DIM as c_int, &tilesize, status);

    if compresstype == HCOMPRESS_1 {
        fits_set_hcomp_scale_safe(fptr, scale, status);
        fits_set_hcomp_smooth_safe(fptr, smooth, status);
    }

    if qlevel != -99. {
        fits_set_quantize_level_safe(fptr, qlevel, status);
        fits_set_quantize_method_safe(fptr, quantize_method, status);
    }

    *status
}

/// Create and initialize a new FITS file on disk.  This routine differs
/// from ffinit in that the input 'name' is literally taken as the name
/// of the disk file to be created, and it does not support CFITSIO's
/// extended filename syntax.
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) name of file to create
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffdkinit(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        raw_to_slice!(name);

        ffdkinit_safe(fptr, name, status)
    }
}

/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) name of file to create
/// * `status` — (IO) error status
pub fn ffdkinit_safe(
    fptr: &mut Option<Box<fitsfile>>,
    name: &[c_char],
    status: &mut c_int,
) -> c_int {
    /* initialize null file pointer */
    let f_tmp = fptr.take();
    if let Some(f) = f_tmp {
        // WARNING: The c version doesn't null pointers after a close, so we have a dangling pointer.
        // We need to be careful with this, as it can cause double free errors.
        // Therefore, if this function is called with a Some(), then we will leak the pointer because
        // it's probably invalid.
        let _ = Box::into_raw(f);
    }

    /* regardless of the value of *status */
    if *status > 0 {
        return *status;
    }

    *status = CREATE_DISK_FILE;

    ffinit_safe(fptr, name, status);

    *status
}

/// Create and initialize a new FITS file.
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) name of file to create
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffinit(
    fptr: *mut Option<Box<fitsfile>>,
    name: *const c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        raw_to_slice!(name);

        ffinit_safe(fptr, name, status)
    }
}

/// Create and initialize a new FITS file.
///
/// # Parameters
///
/// * `fptr`   — (O) FITS file pointer
/// * `name`   — (I) name of file to create
/// * `status` — (IO) error status
pub fn ffinit_safe(fptr: &mut Option<Box<fitsfile>>, name: &[c_char], status: &mut c_int) -> c_int {
    let mut driver: c_int = 0;
    let mut clobber: c_int = 0;

    let mut urltype: [c_char; MAX_PREFIX_LEN] = [0; MAX_PREFIX_LEN];
    let mut outfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut tmplfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut compspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut handle: c_int = 0;
    let mut create_disk_file: c_int = 0;

    /* initialize null file pointer */
    let f_tmp = fptr.take();
    if let Some(f) = f_tmp {
        // WARNING: The c version doesn't null pointers after a close, so we have a dangling pointer.
        // We need to be careful with this, as it can cause double free errors.
        // Therefore, if this function is called with a Some(), then we will leak the pointer because
        // it's probably invalid.
        let _ = Box::into_raw(f);
    }

    /* regardless of the value of *status */
    if *status > 0 {
        return *status;
    }

    if *status == CREATE_DISK_FILE {
        create_disk_file = 1;
        *status = 0;
    }

    if *NEED_TO_INITIALIZE.lock().unwrap() {
        /* this is called only once */
        *status = fits_init_cfitsio_safe();
    }

    if *status > 0 {
        return *status;
    }

    let url = name;

    let mut j = 0;
    while url[j] == bb(b' ') {
        /* ignore leading spaces in the filename */
        j += 1;
    }
    if url[j] == 0 {
        ffpmsg_str("Name of file to create is blank. (ffinit)");
        *status = FILE_NOT_CREATED;
        return *status;
    }

    if create_disk_file > 0 {
        if strlen_safe(&url[j..]) > FLEN_FILENAME - 1 {
            ffpmsg_str("Filename is too long. (ffinit)");
            *status = FILE_NOT_CREATED;
            return *status;
        }

        strcpy_safe(&mut outfile, &url[j..]);
        strcpy_safe(&mut urltype, cs!(c"file://"));
        tmplfile[0] = 0;
        compspec[0] = 0;
    } else {
        /* check for clobber symbol, i.e,  overwrite existing file */
        if url[j] == bb(b'!') {
            clobber = TRUE as c_int;
            j += 1;
        } else {
            clobber = FALSE as c_int;
        }
        /* parse the output file specification */
        /* this routine checks that the strings will not overflow */

        ffourl(
            &url[j..],
            &mut urltype,
            &mut outfile,
            &mut tmplfile,
            &mut compspec,
            status,
        );

        if *status > 0 {
            ffpmsg_str("could not parse the output filename: (ffinit)");
            ffpmsg_slice(&url[j..]);
            return *status;
        }
    }

    let url = &url[j..];

    /* find which driver corresponds to the urltype */
    *status = urltype2driver(&urltype, &mut driver);

    if *status > 0 {
        ffpmsg_str("could not find driver for this file: (ffinit)");
        ffpmsg_slice(url);
        return *status;
    }

    /* delete pre-existing file, if asked to do so */
    if clobber > 0 {
        {
            //let d = driverTable.lock().unwrap();
            let d = DRIVER_TABLE.get().unwrap();
            if d[driver as usize].remove.is_some() {
                (d[driver as usize].remove.unwrap())(&outfile);
            }
        }
    }

    //let d = driverTable.lock().unwrap();
    let d = DRIVER_TABLE.get().unwrap();

    /* call appropriate driver to create the file */
    if d[driver as usize].create.is_some() {
        let lock = FFLOCK(); /* lock this while searching for vacant handle */
        *status = (d[driver as usize].create.unwrap())(&mut outfile, &mut handle);
        FFUNLOCK(lock);

        if *status > 0 {
            ffpmsg_str("failed to create new file (already exists?):");
            ffpmsg_slice(url);
            return *status;
        }
    } else {
        ffpmsg_str("cannot create a new file of this type: (ffinit)");
        ffpmsg_slice(url);
        *status = FILE_NOT_CREATED;
        return *status;
    }

    let d = DRIVER_TABLE.get().unwrap();

    /* allocate fitsfile structure and initialize = 0 */
    let Fptr = FITSfile::new(&d[driver as usize], handle, url, cs!(c"ffinit"), status);
    if Fptr.is_err() {
        return *status;
    }
    let mut Fptr = Fptr.unwrap();

    /* initialize the ageindex array (relative age of the I/O buffers) */
    /* and initialize the bufrecnum array as being empty */
    let mut ii = 0;
    while ii < NIOBUF as usize {
        Fptr.ageindex[ii] = ii as c_int;
        Fptr.bufrecnum[ii] = -1;
        ii += 1;
    }

    /* store the parameters describing the file */
    Fptr.MAXHDU = 1000; /* initial size of headstart */
    Fptr.filehandle = handle; /* store the file pointer */
    Fptr.driver = driver; /*  driver number         */

    strcpy_safe(Fptr.get_filename_as_mut_slice(), url); /* full input filename    */
    Fptr.filesize = 0; /* physical file size     */
    Fptr.logfilesize = 0; /* logical file size      */
    Fptr.writemode = 1; /* read-write mode        */
    Fptr.datastart = DATA_UNDEFINED as LONGLONG; /* unknown start of data  */
    Fptr.curbuf = -1; /* undefined current IO buffer   */
    Fptr.open_count = 1; /* structure is currently used once */
    Fptr.validcode = VALIDSTRUC; /* flag denoting valid structure */
    Fptr.noextsyntax = create_disk_file; /* true if extended syntax is disabled */

    // HEAP ALLOCATION
    /* allocate fitsfile structure and initialize = 0 */
    let f_fitsfile = box_try_new(fitsfile {
        HDUposition: 0,
        Fptr: FptrRef::new(Fptr),
    });

    if f_fitsfile.is_err() {
        let d = DRIVER_TABLE.get().unwrap();
        ((d[driver as usize]).close)(handle); /* close the file */
        ffpmsg_str("failed to allocate structure for following file: (ffopen)");
        ffpmsg_slice(url);
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    let mut f_fitsfile = f_fitsfile.unwrap();

    ffldrc(&mut f_fitsfile, 0, IGNORE_EOF, status); /* initialize first record */

    fits_store_Fptr(f_fitsfile.Fptr.as_ptr(), status); /* store Fptr address */

    /* if template file was given, use it to define structure of new file */

    if tmplfile[0] > 0 {
        ffoptplt(&mut f_fitsfile, &tmplfile, status);
    }

    /* parse and save image compression specification, if given */
    if compspec[0] > 0 {
        ffparsecompspec(&mut f_fitsfile, &compspec, status);
    }

    *fptr = Some(f_fitsfile);

    *status /* successful return */
}

/// ffimem == fits_create_memfile
/// Create and initialize a new FITS file in memory
///
/// # Parameters
///
/// * `fptr`        — (O) FITS file pointer
/// * `buffptr`     — (I) address of memory pointer
/// * `buffsize`    — (I) size of buffer, in bytes
/// * `deltasize`   — (I) increment for future realloc's
/// * `mem_realloc` — function
/// * `status`      — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffimem(
    fptr: *mut Option<Box<fitsfile>>,
    buffptr: *mut *mut c_void,
    buffsize: *mut usize,
    deltasize: usize,
    mem_realloc: Option<unsafe extern "C" fn(p: *mut c_void, newsize: usize) -> *mut c_void>,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let buffptr = buffptr.as_mut().expect(NULL_MSG);
        let buffsize = buffsize.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        ffimem_safe(fptr, buffptr, buffsize, deltasize, mem_realloc, status)
    }
}

/// Create and initialize a new FITS file in memory
///
/// # Parameters
///
/// * `fptr`        — (O) FITS file pointer
/// * `buffptr`     — (I) address of memory pointer
/// * `buffsize`    — (I) size of buffer, in bytes
/// * `deltasize`   — (I) increment for future realloc's
/// * `mem_realloc` — function
/// * `status`      — (IO) error status
pub fn ffimem_safe(
    fptr: &mut Option<Box<fitsfile>>,
    buffptr: *mut *mut c_void,
    buffsize: &mut usize,
    deltasize: usize,
    mem_realloc: Option<unsafe extern "C" fn(p: *mut c_void, newsize: usize) -> *mut c_void>,
    status: &mut c_int,
) -> c_int {
    let mut driver: c_int = 0;
    let mut handle: c_int = 0;
    let mut urltype: [c_char; MAX_PREFIX_LEN] = [0; MAX_PREFIX_LEN];

    if *status > 0 {
        return *status;
    }

    /* initialize null file pointer */
    let f_tmp = fptr.take();
    if let Some(f) = f_tmp {
        // WARNING (mirrors ffomem): the C leaves a dangling pointer after it
        // overwrites *fptr, so if we were handed an existing Some() it is
        // probably invalid; leak it rather than risk a double free.
        let _ = Box::into_raw(f);
    }

    if *NEED_TO_INITIALIZE.lock().unwrap() {
        /* this is called only once */
        *status = fits_init_cfitsio_safe();
    }

    if *status > 0 {
        return *status;
    }

    strcpy_safe(&mut urltype, cs!(c"memkeep://")); /* URL type for pre-existing memory file */

    *status = urltype2driver(&urltype, &mut driver);

    if *status > 0 {
        ffpmsg_str("could not find driver for pre-existing memory file: (ffimem)");
        return *status;
    }

    /* call driver routine to "open" the memory file */
    let lock = FFLOCK(); /* lock this while searching for vacant handle */
    *status = mem_openmem(buffptr, buffsize, deltasize, mem_realloc, &mut handle);
    FFUNLOCK(lock);

    if *status > 0 {
        ffpmsg_str("failed to open pre-existing memory file: (ffimem)");
        return *status;
    }

    /* allocate the fitsfile / FITSfile structures along with the filename,
    headstart, and iobuffer arrays (the C's calloc/malloc block) */
    let d = DRIVER_TABLE.get().unwrap();
    let Fptr = FITSfile::new(
        &d[driver as usize],
        handle,
        cs!(c"memfile"),
        cs!(c"ffimem"),
        status,
    );
    if Fptr.is_err() {
        /* FITSfile::new already reported the specific allocation failure */
        return *status;
    }

    let mut Fptr = Fptr.unwrap();

    /* initialize the ageindex array (relative age of the I/O buffers) */
    /* and initialize the bufrecnum array as being empty */
    for ii in 0..(NIOBUF as usize) {
        Fptr.ageindex[ii] = ii as c_int;
        Fptr.bufrecnum[ii] = -1;
    }

    /* store the parameters describing the file */
    Fptr.MAXHDU = 1000; /* initial size of headstart */
    Fptr.filehandle = handle; /* file handle */
    Fptr.driver = driver; /* driver number */
    strcpy_safe(Fptr.get_filename_as_mut_slice(), cs!(c"memfile")); /* dummy filename */
    Fptr.filesize = *buffsize as LONGLONG; /* physical file size */
    Fptr.logfilesize = *buffsize as LONGLONG; /* logical file size */
    Fptr.writemode = 1; /* read-write mode    */
    Fptr.datastart = DATA_UNDEFINED as LONGLONG; /* unknown start of data */
    Fptr.curbuf = -1; /* undefined current IO buffer */
    Fptr.open_count = 1; /* structure is currently used once */
    Fptr.validcode = VALIDSTRUC; /* flag denoting valid structure */
    Fptr.noextsyntax = 0; /* extended syntax can be used in filename */

    let f_fitsfile = box_try_new(fitsfile {
        HDUposition: 0,
        Fptr: FptrRef::new(Fptr),
    });

    if f_fitsfile.is_err() {
        let d = DRIVER_TABLE.get().unwrap();
        ((d[driver as usize]).close)(handle); /* close the file */
        ffpmsg_str("failed to allocate structure for memory file: (ffimem)");
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    let mut f_fitsfile = f_fitsfile.unwrap();

    ffldrc(&mut f_fitsfile, 0, IGNORE_EOF, status); /* initialize first record */
    fits_store_Fptr(f_fitsfile.Fptr.as_ptr(), status); /* store Fptr address */

    *fptr = Some(f_fitsfile);

    *status
}

/// Initialize anything that is required before using the CFITSIO routines
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_init_cfitsio() -> c_int {
    // FFI WRAPPER
    fits_init_cfitsio_safe()
}

/// Initialize anything that is required before using the CFITSIO routines
pub fn fits_init_cfitsio_safe() -> c_int {
    pub union u_tag {
        pub ival: i16,
        pub cval: [c_char; 2],
    }

    fitsio_init_lock();

    let lock = FFLOCK(); /* lockout other threads while executing this critical */
    /* section of code  */

    if !*NEED_TO_INITIALIZE.lock().unwrap() {
        /* already initialized? */
        FFUNLOCK(lock);
        return 0;
    }

    let mut u_ival: i16 = 1;
    let u_cval: &mut [c_char; 2] = cast_mut(&mut u_ival);

    // Because of the FFLOCK mutex, we can do this in two stages.
    let mut drivers = Vec::with_capacity(MAX_DRIVERS);
    let d = &mut drivers;

    /*   test for correct byteswapping.   */
    if (BYTESWAPPED && u_cval[0] != 1) || (!BYTESWAPPED && u_cval[1] != 1) {
        println!("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        println!(" Byteswapping is not being done correctly on this system.");
        println!(" Check the CFITSIO_MACHINE and BYTESWAPPED definitions in fitsio2.h");
        println!(" Please report this problem to the CFITSIO developers.");
        println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        FFUNLOCK(lock);
        return 1;
    }

    /*  test that LONGLONG is an 8 byte integer */
    if core::mem::size_of::<LONGLONG>() != 8 {
        println!("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        println!(" CFITSIO did not find an 8-byte long integer data type.");
        println!("   sizeof(LONGLONG) = {}", core::mem::size_of::<LONGLONG>());
        println!(" Please report this problem to the CFITSIO developers.");
        println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        FFUNLOCK(lock);
        return 1;
    }

    /* register the standard I/O drivers that are always available */

    /* 1--------------------disk file driver-----------------------*/
    let status = fits_register_driver(
        d,
        c"file://",
        Some(file_init),
        Some(file_shutdown),
        Some(file_setoptions),
        Some(file_getoptions),
        Some(file_getversion),
        Some(file_checkfile),
        Some(file_open),
        Some(file_create),
        Some(file_truncate),
        file_close,
        Some(file_remove),
        file_size,
        Some(file_flush),
        file_seek,
        file_read,
        file_write,
    );

    if status != 0 {
        ffpmsg_str("failed to register the file:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 2------------ output temporary memory file driver ----------------*/
    let status = fits_register_driver(
        d,
        c"mem://",
        Some(mem_init),
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None, /* checkfile not needed */
        None, /* open function not allowed */
        Some(mem_create),
        Some(mem_truncate_unsafe),
        mem_close_free_unsafe,
        None, /* remove function not required */
        mem_size,
        None, /* flush function not required */
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the mem:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 3--------------input pre-existing memory file driver----------------*/
    let status = fits_register_driver(
        d,
        c"memkeep://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None, /* checkfile not needed */
        None, /* file open driver function is not used */
        None, /* create function not allowed */
        Some(mem_truncate_unsafe),
        mem_close_keep,
        None, /* remove function not required */
        mem_size,
        None, /* flush function not required */
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the memkeep:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* ==================== SHARED MEMORY DRIVER SECTION ======================= */

    #[cfg(all(feature = "shared_mem", not(target_os = "windows")))]
    {
        /* 22--------------------shared memory driver-----------------------*/
        let status = fits_register_driver(
            d,
            c"shmem://",
            Some(smem_init),
            Some(smem_shutdown),
            Some(smem_setoptions),
            Some(smem_getoptions),
            Some(smem_getversion),
            None, /* checkfile not needed */
            Some(smem_open),
            Some(smem_create),
            None, /* truncate file not supported yet */
            smem_close,
            Some(smem_remove),
            smem_size,
            Some(smem_flush),
            smem_seek,
            smem_read,
            smem_write,
        );

        if status != 0 {
            ffpmsg_str("failed to register the shmem:// driver (init_cfitsio)");
            FFUNLOCK(lock);
            return status;
        }
    }

    /* 4-------------------stdin stream driver----------------------*/
    /*  the stdin stream is copied to memory then opened in memory */
    let status = fits_register_driver(
        d,
        c"stdin://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        Some(stdin_checkfile),
        Some(stdin_open),
        None, /* create function not allowed */
        Some(mem_truncate_unsafe),
        mem_close_free_unsafe,
        None, /* remove function not required */
        mem_size,
        None, /* flush function not required */
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the stdin:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 5-------------------stdin file stream driver----------------------*/
    /*  the stdin stream is copied to a disk file then the disk file is opened */
    let status = fits_register_driver(
        d,
        c"stdinfile://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None, /* checkfile not needed */
        Some(stdin_open),
        None, /* create function not allowed */
        Some(file_truncate),
        file_close,
        Some(file_remove),
        file_size,
        Some(file_flush),
        file_seek,
        file_read,
        file_write,
    );

    if status != 0 {
        ffpmsg_str("failed to register the stdinfile:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 6-----------------------stdout stream driver------------------*/
    let status = fits_register_driver(
        d,
        c"stdout://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None,
        None,
        Some(mem_create),
        Some(mem_truncate_unsafe),
        stdout_close_unsafe,
        None,
        mem_size,
        None, /* flush function not required */
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the stdout:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 7------------------iraf disk file to memory driver -----------*/
    let status = fits_register_driver(
        d,
        c"irafmem://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None,
        Some(mem_iraf_open),
        None,
        Some(mem_truncate_unsafe),
        mem_close_free_unsafe,
        None,
        mem_size,
        None, /* flush function not required */
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the irafmem:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 8------------------raw binary file to memory driver -----------*/
    let status = fits_register_driver(
        d,
        c"rawfile://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None,
        Some(mem_rawfile_open),
        None,
        Some(mem_truncate_unsafe),
        mem_close_free_unsafe,
        None,
        mem_size,
        None, /* flush function not required */
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the rawfile:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 9------------------compressed disk file to memory driver -----------*/
    let status = fits_register_driver(
        d,
        c"compress://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None,
        Some(mem_compress_open),
        None,
        Some(mem_truncate_unsafe),
        mem_close_free_unsafe,
        None,
        mem_size,
        None, /* flush function not required */
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the compress:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 10------------------compressed disk file to memory driver -----------*/
    let status = fits_register_driver(
        d,
        c"compressmem://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None,
        Some(mem_compress_openrw),
        None,
        Some(mem_truncate_unsafe),
        mem_close_free_unsafe,
        None, /* remove function not required */
        mem_size,
        None, /* flush function not required */
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the compressmem:// driver (init_cfitsio)"); /* checkfile not needed */
        FFUNLOCK(lock);
        return status; /* no file truncate function */
    }

    /* 11------------------compressed disk file to disk file driver -------*/
    let status = fits_register_driver(
        d,
        c"compressfile://",
        None,
        Some(file_shutdown),
        Some(file_setoptions),
        Some(file_getoptions),
        Some(file_getversion),
        None,
        Some(file_compress_open),
        Some(file_create),
        None,
        file_close,
        Some(file_remove),
        file_size,
        Some(file_flush),
        file_seek,
        file_read,
        file_write,
    );

    if status != 0 {
        ffpmsg_str("failed to register the compressfile:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    /* 12---create file in memory, then compress it to disk file on close--*/
    let status = fits_register_driver(
        d,
        c"compressoutfile://",
        None,
        Some(mem_shutdown),
        Some(mem_setoptions),
        Some(mem_getoptions),
        Some(mem_getversion),
        None,
        None,
        Some(mem_create_comp_unsafe),
        Some(mem_truncate_unsafe),
        mem_close_comp_unsafe,
        Some(file_remove),
        mem_size,
        None,
        mem_seek,
        mem_read_unsafe,
        mem_write_unsafe,
    );

    if status != 0 {
        ffpmsg_str("failed to register the compressoutfile:// driver (init_cfitsio)"); /* 13--------------------root driver-----------------------*/
        FFUNLOCK(lock);
        return status; /* checkfile not needed */
    }

    /* Register Optional drivers */

    /* 24---------------stdin and stdout stream driver-------------------*/
    let status = fits_register_driver(
        d,
        c"stream://",
        None,
        None,
        None,
        None,
        None,
        None,
        Some(fits_stream_open),
        Some(fits_stream_create),
        None, /* no stream truncate function */
        fits_stream_close,
        None, /* no stream remove */
        fits_stream_size,
        Some(fits_stream_flush),
        fits_stream_seek,
        fits_stream_read,
        fits_stream_write,
    );

    if status != 0 {
        ffpmsg_str("failed to register the stream:// driver (init_cfitsio)");
        FFUNLOCK(lock);
        return status;
    }

    let _dd = DRIVER_TABLE.set(drivers);

    let l = NEED_TO_INITIALIZE.lock();
    *l.unwrap() = false;

    FFUNLOCK(lock);
    status
}

/// register all the functions needed to support an I/O driver
pub(crate) fn fits_register_driver(
    drivers: &mut Vec<fitsdriver>,
    prefix: &CStr,
    init: Option<fn() -> c_int>,
    shutdown: Option<fn() -> c_int>,
    setoptions: Option<fn(option: c_int) -> c_int>,
    getoptions: Option<fn(options: &mut c_int) -> c_int>,
    getversion: Option<fn(version: &mut c_int) -> c_int>,
    checkfile: Option<CheckFileFn>,
    open: Option<DriverOpenFn>,
    create: Option<fn(filename: &mut [c_char; FLEN_FILENAME], drivehandle: &mut c_int) -> c_int>,
    truncate: Option<fn(drivehandle: c_int, size: usize) -> c_int>,
    close: fn(drivehandle: c_int) -> c_int,
    remove: Option<fn(filename: &[c_char]) -> c_int>,
    size: fn(drivehandle: c_int, sizex: &mut usize) -> c_int,
    flush: Option<fn(drivehandle: c_int) -> c_int>,
    seek: fn(drivehandle: c_int, offset: LONGLONG) -> c_int,
    read: fn(drivehandle: c_int, buffer: &mut [u8], nbytes: usize) -> c_int,
    write: fn(drivehandle: c_int, buffer: &[u8], nbyte: usize) -> c_int,
) -> c_int {
    let status; /* increment the number of drivers */

    if drivers.len() + 1 > MAX_DRIVERS {
        return TOO_MANY_DRIVERS;
    }

    if prefix.is_empty() {
        return BAD_URL_PREFIX;
    }

    if let Some(init) = init {
        status = init(); /* initialize the driver */
        if status != 0 {
            return status;
        };
    }

    /*  fill in data in table */
    let mut pf = [0; MAX_PREFIX_LEN];
    strncpy_safe(
        &mut pf,
        cast_slice(prefix.to_bytes_with_nul()),
        MAX_PREFIX_LEN,
    );
    pf[MAX_PREFIX_LEN - 1] = 0;

    let dd = fitsdriver {
        prefix: pf,
        init,
        shutdown,
        setoptions,
        getoptions,
        getversion,
        checkfile,
        open,
        create,
        truncate,
        close,
        remove,
        size,
        flush,
        seek,
        read,
        write,
    };

    drivers.push(dd); /* increment the number of drivers */

    0
}

/// fits_parse_input_url
/// parse the input URL into its basic components.
/// This routine does not support the pixfilter or compspec components.
///
/// # Parameters
///
/// * `url`        — input filename
/// * `urltype`    — e.g., 'file://', 'http://', 'mem://'
/// * `infilex`    — root filename (may be complete path)
/// * `outfile`    — optional output file name
/// * `extspec`    — extension spec: +n or [extname, extver]
/// * `rowfilterx` — boolean row filter expression
/// * `binspec`    — histogram binning specifier
/// * `colspec`    — column or keyword modifier expression
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffiurl(
    url: *const c_char,
    urltype: *mut c_char,
    infilex: *mut c_char,
    outfile: *mut c_char,
    extspec: *mut c_char,
    rowfilterx: *mut c_char,
    binspec: *mut c_char,
    colspec: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        raw_to_slice!(url);

        let status = status.as_mut().expect(NULL_MSG);

        let urltype = urltype
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let infilex = infilex
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let outfile = outfile
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let extspec = extspec
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let rowfilterx = rowfilterx
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let binspec = binspec
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let colspec = colspec
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));

        ffiurl_safe(
            url, urltype, infilex, outfile, extspec, rowfilterx, binspec, colspec, status,
        )
    }
}

/// fits_parse_input_url
/// parse the input URL into its basic components.
/// This routine does not support the pixfilter or compspec components.
///
/// # Parameters
///
/// * `url`        — input filename
/// * `urltype`    — e.g., 'file://', 'http://', 'mem://'
/// * `infilex`    — root filename (may be complete path)
/// * `outfile`    — optional output file name
/// * `extspec`    — extension spec: +n or [extname, extver]
/// * `rowfilterx` — boolean row filter expression
/// * `binspec`    — histogram binning specifier
/// * `colspec`    — column or keyword modifier expression
pub fn ffiurl_safe(
    url: &[c_char],
    urltype: Option<&mut [c_char]>,
    infilex: Option<&mut [c_char]>,
    outfile: Option<&mut [c_char]>,
    extspec: Option<&mut [c_char]>,
    rowfilterx: Option<&mut [c_char]>,
    binspec: Option<&mut [c_char]>,
    colspec: Option<&mut [c_char]>,
    status: &mut c_int,
) -> c_int {
    ffifile2_safe(
        url, urltype, infilex, outfile, extspec, rowfilterx, binspec, colspec, None, None, status,
    )
}

/// fits_parse_input_filename
/// parse the input URL into its basic components.
/// This routine does not support the compspec component.
///
/// # Parameters
///
/// * `url`        — input filename
/// * `urltype`    — e.g., 'file://', 'http://', 'mem://'
/// * `infilex`    — root filename (may be complete path)
/// * `outfile`    — optional output file name
/// * `extspec`    — extension spec: +n or [extname, extver]
/// * `rowfilterx` — boolean row filter expression
/// * `binspec`    — histogram binning specifier
/// * `colspec`    — column or keyword modifier expression
/// * `pixfilter`  — pixel filter expression
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffifile(
    url: *const c_char,
    urltype: *mut c_char,
    infilex: *mut c_char,
    outfile: *mut c_char,
    extspec: *mut c_char,
    rowfilterx: *mut c_char,
    binspec: *mut c_char,
    colspec: *mut c_char,
    pixfilter: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(url);

        let urltype = urltype
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let infilex = infilex
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let outfile = outfile
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let extspec = extspec
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let rowfilterx = rowfilterx
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let binspec = binspec
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let colspec = colspec
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));
        let pixfilter = pixfilter
            .as_mut()
            .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME));

        ffifile_safe(
            url, urltype, infilex, outfile, extspec, rowfilterx, binspec, colspec, pixfilter,
            status,
        )
    }
}

/// # Parameters
///
/// * `url`        — input filename
/// * `urltype`    — e.g., 'file://', 'http://', 'mem://'
/// * `infilex`    — root filename (may be complete path)
/// * `outfile`    — optional output file name
/// * `extspec`    — extension spec: +n or [extname, extver]
/// * `rowfilterx` — boolean row filter expression
/// * `binspec`    — histogram binning specifier
/// * `colspec`    — column or keyword modifier expression
/// * `pixfilter`  — pixel filter expression
pub fn ffifile_safe(
    url: &[c_char],
    urltype: Option<&mut [c_char]>,
    infilex: Option<&mut [c_char]>,
    outfile: Option<&mut [c_char]>,
    extspec: Option<&mut [c_char]>,
    rowfilterx: Option<&mut [c_char]>,
    binspec: Option<&mut [c_char]>,
    colspec: Option<&mut [c_char]>,
    pixfilter: Option<&mut [c_char]>,
    status: &mut c_int,
) -> c_int {
    ffifile2_safe(
        url, urltype, infilex, outfile, extspec, rowfilterx, binspec, colspec, pixfilter, None,
        status,
    )
}

/// fits_parse_input_filename
/// parse the input URL into its basic components.
/// This routine is big and ugly and should be redesigned someday!
///
/// # Parameters
///
/// * `url`        — input filename
/// * `urltype`    — e.g., 'file://', 'http://', 'mem://'
/// * `infilex`    — root filename (may be complete path)
/// * `outfile`    — optional output file name
/// * `extspec`    — extension spec: +n or [extname, extver]
/// * `rowfilterx` — boolean row filter expression
/// * `binspec`    — histogram binning specifier
/// * `colspec`    — column or keyword modifier expression
/// * `pixfilter`  — pixel filter expression
/// * `compspec`   — image compression specification
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffifile2(
    url: *const c_char,
    urltype: *mut c_char,
    infilex: *mut c_char,
    outfile: *mut c_char,
    extspec: *mut c_char,
    rowfilterx: *mut c_char,
    binspec: *mut c_char,
    colspec: *mut c_char,
    pixfilter: *mut c_char,
    compspec: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        raw_to_slice!(url);

        let status = status.as_mut().expect(NULL_MSG);

        ffifile2_safe(
            url,
            urltype
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            infilex
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            outfile
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            extspec
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            rowfilterx
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            binspec
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            colspec
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            pixfilter
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            compspec
                .as_mut()
                .map(|p| core::slice::from_raw_parts_mut(p, FLEN_FILENAME)),
            status,
        )
    }
}

/// fits_parse_input_filename
/// Parse the input filename or URL into its component parts, namely:
/// the file type (file://, ftp://, http://, etc),
/// the base input file name,
/// the name of the output file that the input file is to be copied to prior to opening,
/// the HDU or extension specification,
/// the filtering specifier,
/// the binning specifier,
/// the column specifier,
/// and the image pixel filtering specifier.
/// A null pointer (0) may be be specified for any of the output string arguments that are not needed. Null strings will be returned for any components that are not present in the input file name. The calling routine must allocate sufficient memory to hold the returned character strings. Allocating the string lengths equal to FLEN_FILENAME is guaranteed to be safe. These routines are mainly for internal use by other CFITSIO routines.
///
/// # Parameters
///
/// * `url`        — input filename
/// * `urltype`    — e.g., 'file://', 'http://', 'mem://'
/// * `infilex`    — root filename (may be complete path)
/// * `outfile`    — optional output file name
/// * `extspec`    — extension spec: +n or [extname, extver]
/// * `rowfilterx` — boolean row filter expression
/// * `binspec`    — histogram binning specifier
/// * `colspec`    — column or keyword modifier expression
/// * `pixfilter`  — pixel filter expression
/// * `compspec`   — image compression specification
pub fn ffifile2_safe(
    url: &[c_char],
    mut urltype: Option<&mut [c_char]>,
    mut infilex: Option<&mut [c_char]>,
    mut outfile: Option<&mut [c_char]>,
    mut extspec: Option<&mut [c_char]>,
    mut rowfilterx: Option<&mut [c_char]>,
    mut binspec: Option<&mut [c_char]>,
    mut colspec: Option<&mut [c_char]>,
    mut pixfilter: Option<&mut [c_char]>,
    mut compspec: Option<&mut [c_char]>,
    status: &mut c_int,
) -> c_int {
    let mut infilelen = 0;
    let mut plus_ext = 0;
    let ptr2: Option<usize> = None;
    let ptr3: Option<usize> = None;
    let mut ptr4: Option<usize> = None;
    let mut ptr2_index = 0;
    let mut ptr3_index = 0;
    let mut ptr4_index = 0;

    let mut hasAt;
    let mut hasDot;
    let mut hasOper;
    let mut followingOper;
    let mut spaceTerm;
    let mut rowFilter;
    let mut colStart;
    let mut binStart;
    let mut pixStart;
    let mut compStart;

    if *status > 0 {
        return *status;
    }

    /* Initialize null strings */
    if let Some(ref mut s) = infilex {
        s[0] = 0;
    }
    if let Some(ref mut s) = urltype {
        s[0] = 0;
    }
    if let Some(ref mut s) = outfile {
        s[0] = 0;
    }
    if let Some(ref mut s) = extspec {
        s[0] = 0;
    }
    if let Some(ref mut s) = binspec {
        s[0] = 0;
    }
    if let Some(ref mut s) = colspec {
        s[0] = 0;
    }
    if let Some(ref mut s) = rowfilterx {
        s[0] = 0;
    }
    if let Some(ref mut s) = pixfilter {
        s[0] = 0;
    }
    if let Some(ref mut s) = compspec {
        s[0] = 0;
    }

    let mut slen = strlen_safe(url);

    if slen == 0 {
        /* blank filename ?? */
        return *status;
    }

    // TEMP HEAP ALLOCATION
    /* allocate memory for 3 strings, each as long as the input url */
    let mut infile = Vec::new();
    if infile.try_reserve_exact(slen + 1).is_err() {
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        infile.resize(slen + 1, 0);
    }

    let mut rowfilter = Vec::new();
    if rowfilter.try_reserve_exact(slen + 1).is_err() {
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        rowfilter.resize(slen + 1, 0);
    }

    let mut tmpstr = Vec::new();
    if tmpstr.try_reserve_exact(slen + 1).is_err() {
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        tmpstr.resize(slen + 1, 0);
    }

    let mut ptr1 = url;
    let mut ptr1_index = 0;

    /* -------------------------------------------------------- */
    /*  get urltype (e.g., file://, ftp://, http://, etc.)  */
    /* --------------------------------------------------------- */

    if ptr1[0] == bb(b'-')
        && (ptr1[1] == 0 || ptr1[1] == bb(b' ') || ptr1[1] == bb(b'[') || ptr1[1] == bb(b'('))
    {
        /* "-" means read file from stdin. Also support "- ",        */
        /* "-[extname]" and '-(outfile.fits)" but exclude disk file  */
        /* names that begin with a minus sign, e.g., "-55d33m.fits"  */

        if let Some(ref mut s) = urltype {
            strcat_safe(s, cs!(c"stdin://"));
        }

        ptr1_index += 1;
    } else if fits_strncasecmp(ptr1, cs!(c"stdin"), 5) == 0 {
        if let Some(ref mut s) = urltype {
            strcat_safe(s, cs!(c"stdin://"));
        }
        ptr1_index += 5;
    } else {
        let mut ptr2 = strstr_safe(ptr1, cs!(c"://"));
        let ptr3 = strstr_safe(ptr1, cs!(c"("));
        if ptr3.is_some() && ptr2.is_some() && (ptr3.unwrap() < ptr2.unwrap()) {
            /* the urltype follows a '(' character, so it must apply */
            /* to the output file, and is not the urltype of the input file */
            ptr2 = None; /* so reset pointer to zero */
        }

        if let Some(ptr2) = ptr2 {
            /* copy the explicit urltype string */
            if (ptr2 + 3) >= MAX_PREFIX_LEN {
                ffpmsg_str("Name of urltype is too long.");
                *status = URL_PARSE_ERROR;
                return *status;
            }

            if let Some(ref mut s) = urltype {
                strncat_safe(s, ptr1, ptr2 + 3);
            }
            ptr1_index += ptr2 + 3;
        } else if strncmp_safe(ptr1, cs!(c"ftp:"), 4) == 0 {
            /* the 2 //'s are optional */
            if let Some(ref mut s) = urltype {
                strcat_safe(s, cs!(c"ftp://"));
            }
            ptr1_index += 4;
        } else if strncmp_safe(ptr1, cs!(c"gsiftp:"), 7) == 0 {
            /* the 2 //'s are optional */
            if let Some(ref mut s) = urltype {
                strcat_safe(s, cs!(c"gsiftp://"));
            }
            ptr1_index += 7;
        } else if strncmp_safe(ptr1, cs!(c"http:"), 5) == 0 {
            /* the 2 //'s are optional */
            if let Some(ref mut s) = urltype {
                strcat_safe(s, cs!(c"http://"));
            }
            ptr1_index += 5;
        } else if strncmp_safe(ptr1, cs!(c"mem:"), 4) == 0 {
            /* the 2 //'s are optional */
            if let Some(ref mut s) = urltype {
                strcat_safe(s, cs!(c"mem://"));
            }
            ptr1_index += 4;
        } else if strncmp_safe(ptr1, cs!(c"shmem:"), 6) == 0 {
            /* the 2 //'s are optional */
            if let Some(ref mut s) = urltype {
                strcat_safe(s, cs!(c"shmem://"));
            }
            ptr1_index += 6;
        } else if strncmp_safe(ptr1, cs!(c"file:"), 5) == 0 {
            /* the 2 //'s are optional */
            if let Some(ref mut s) = urltype {
                strcat_safe(s, cs!(c"file://"));
            }
            ptr1_index += 5;
        } else {
            /* assume file driver    */
            if let Some(ref mut s) = urltype {
                strcat_safe(s, cs!(c"file://"));
            };
        };
    }

    /* ----------------------------------------------------------
    If this is a http:// type file, then the cgi file name could
    include the '[' character, which should not be interpreted
    as part of CFITSIO's Extended File Name Syntax.  Test for this
    case by seeing if the last character is a ']' or ')'.  If it
    is not, then just treat the whole input string as the file name
    and do not attempt to interprete the name using the extended
    filename syntax.
    ----------------------------------------------------------- */

    // Advance slice
    ptr1 = &ptr1[ptr1_index..];
    ptr1_index = 0;

    if let Some(ref urltype_slice) = urltype
        && strncmp_safe(urltype_slice, cs!(c"http://"), 7) == 0
    {
        /* test for opening parenthesis or bracket in the file name */
        if strchr_safe(ptr1, bb(b'(')).is_some() || strchr_safe(ptr1, bb(b'[')).is_some() {
            slen = strlen_safe(ptr1);

            let mut ptr3_index = slen - 1;

            while ptr1[ptr3_index] == bb(b' ') {
                /* ignore trailing blanks */
                ptr3_index -= 1;
            }

            if ptr1[ptr3_index] != bb(b']') && ptr1[ptr3_index] != bb(b')') {
                /* name doesn't end with a ']' or ')' so don't try */
                /* to parse this unusual string (may be cgi string)  */
                if let Some(ref mut s) = infilex {
                    if strlen_safe(ptr1) > FLEN_FILENAME - 1 {
                        ffpmsg_str("Name of file is too long.");
                        *status = URL_PARSE_ERROR;
                        return *status;
                    }
                    strcpy_safe(s, ptr1);
                }

                return *status;
            };
        };
    }
    /* ----------------------------------------------------------
    Look for VMS style filenames like:
         disk:[directory.subdirectory]filename.ext, or
              [directory.subdirectory]filename.ext

    Check if the first character is a '[' and urltype != stdin
    or if there is a ':[' string in the remaining url string. If
    so, then need to move past this bracket character before
    search for the opening bracket of a filter specification.
    ----------------------------------------------------------- */
    let mut tmptr = ptr1; /* this bracket encloses a VMS directory name */
    if ptr1[0] == bb(b'[') {
        if url[0] != bb(b'-') {
            tmptr = &ptr1[1..]; /* these 2 chars are part of the VMS disk and directory */
        };
    } else {
        let tmp_index = strstr_safe(ptr1, cs!(c":["));
        if let Some(tmp_index) = tmp_index {
            tmptr = &ptr1[(tmp_index + 2)..];
        } else {
            tmptr = ptr1;
        };
    }

    /* ------------------------ */
    /*  get the input file name */
    /* ------------------------ */
    let mut ptr2 = strchr_safe(tmptr, bb(b'(')); /* search for opening parenthesis ( */
    let mut ptr3 = strchr_safe(tmptr, bb(b'[')); /* search for opening bracket [ */
    if let Some(ptr2_idx) = ptr2 {
        ptr2_index = ptr2_idx;
        ptr4 = strchr_safe(&tmptr[ptr2_idx..], bb(b')')); /* search for closing parenthesis ) */
        while let Some(ptr4_idx) = ptr4
            && let Some(ptr2_idx) = ptr2
        {
            ptr4_index = ptr4_idx + ptr2_idx; /* advance to absolute index */

            loop {
                ptr4_index += 1; /* find next non-blank char after ')' */
                if tmptr[ptr4_index] != bb(b' ') {
                    break;
                }; /* simple case: no [ or ( in the file name */
            } /* no bracket, so () enclose output file name */

            if tmptr[ptr4_index] == 0 || tmptr[ptr4_index] == bb(b'[') {
                break; /* () enclose output name before bracket */
            } /* search for closing ) */

            ptr2 = strchr_safe(&tmptr[ptr2_index + 1..], bb(b'(')); /* error, no closing ) */
            ptr4 = strchr_safe(&tmptr[ptr4_index..], bb(b')'));

            if let Some(ptr2_idx) = ptr2 {
                ptr2_index += ptr2_idx + 1; /* advance to absolute index */
            }

            if let Some(ptr4_idx) = ptr4 {
                ptr4_index += ptr4_idx; /* advance to absolute index */
            }
        }
    }

    if ptr2 == ptr3 {
        /* simple case: no [ or ( in the file name */
        strcat_safe(&mut infile, ptr1);
    } else if ptr3.is_none() || (ptr2.is_some() && ptr2 < ptr3) {
        let t = ptr2.unwrap();
        strncat_safe(&mut infile, ptr1, t);

        let ptr2_off = ptr2_index + 1;
        let p1 = strchr_safe(&tmptr[ptr2_off..], bb(b')'));
        if p1.is_none() {
            *status = URL_PARSE_ERROR;
            return *status;
        }

        if let Some(ref mut outfile_slice) = outfile {
            let t = p1.unwrap();

            if t > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return *status;
            }
            strncat_safe(outfile_slice, &tmptr[ptr2_off..], t);
        }
        /* the opening [ could have been part of output name,    */
        /*      e.g., file(out[compress])[3][#row > 5]           */
        /* so search again for opening bracket following the closing ) */
        let p1_abs = ptr2_off + p1.unwrap();
        ptr3 = strchr_safe(&tmptr[p1_abs..], bb(b'['));
        if let Some(p3_idx) = ptr3 {
            ptr3_index = p1_abs + p3_idx;
        }
        /*   bracket comes first, so there is no output name */
    } else {
        let t = ptr3.unwrap();
        ptr3_index = t;
        strncat_safe(&mut infile, ptr1, t);
    }

    /* strip off any trailing blanks in the names */
    let mut slen = strlen_safe(&infile) as isize;

    slen -= 1;
    while slen > 0 && infile[slen as usize] == bb(b' ') {
        infile[slen as usize] = 0;
        slen -= 1;
    }

    if let Some(ref mut outfile_slice) = outfile {
        slen = strlen_safe(outfile_slice) as isize;
        slen -= 1;
        while slen > 0 && outfile_slice[slen as usize] == bb(b' ') {
            outfile_slice[slen as usize] = 0;
            slen -= 1;
        }
    }
    /* --------------------------------------------- */
    /* check if this is an IRAF file (.imh extension */
    /* --------------------------------------------- */
    let ptr4 = strstr_safe(&infile, cs!(c".imh"));
    /* did the infile name end with c".imh" ? */
    if let Some(p4) = ptr4
        && infile[p4 + 4] == 0
        && let Some(ref mut s) = urltype
    {
        strcpy_safe(s, cs!(c"irafmem://"));
    }

    /* --------------------------------------------- */
    /* check if the 'filename+n' convention has been */
    /* used to specifiy which HDU number to open     */
    /* --------------------------------------------- */
    let jj = strlen_safe(&infile) as isize; /* search backwards for '+' sign */
    let mut ii: isize = jj as isize - 1;
    while ii >= 0 {
        if infile[ii as usize] == bb(b'+') {
            break; /* limit extension numbers to 5 digits */
        }; /* pointer to start of sequence */
        ii -= 1;
    }
    /* are all the chars digits? */

    if ii > 0 && (jj - ii) < 7 {
        infilelen = ii; /* yes, the '+n' convention was used.  Copy */
        ii += 1; /* the digits to the output extspec string. */
        let ptr1 = &infile[(ii as usize)..]; /* delete the extension number */
        while ii < jj {
            if !isdigit_safe(infile[ii as usize]) {
                break;
            };
            ii += 1;
        }
        if ii == jj {
            plus_ext = 1;
            if let Some(ref mut s) = extspec {
                if jj - infilelen > FLEN_FILENAME as isize - 1 {
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
                strncpy_safe(s, ptr1, (jj - infilelen) as usize);
            }
            infile[infilelen as usize] = 0;
        };
    }
    /* -------------------------------------------------------------------- */
    /* if '*' was given for the output name expand it to the root file name */
    /* -------------------------------------------------------------------- */
    if let Some(ref mut outfile_slice) = outfile
        && outfile_slice[0] == bb(b'*')
    {
        /* scan input name backwards to the first '/' character */
        let mut ii = jj - 1;
        while ii >= 0 {
            if infile[ii as usize] == bb(b'/') || ii == 0 {
                if strlen_safe(&infile[((ii + 1) as usize)..]) > FLEN_FILENAME - 1 {
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
                strcpy_safe(outfile_slice, &infile[((ii + 1) as usize)..]);
                break;
            };
            ii -= 1
        }
    }
    /* ------------------------------------------ */
    /* copy strings from local copy to the output */
    /* ------------------------------------------ */
    if let Some(ref mut s) = infilex {
        if strlen_safe(&infile) > FLEN_FILENAME - 1 {
            *status = URL_PARSE_ERROR;
            return *status;
        }
        strcpy_safe(s, &infile);
    }
    /* ---------------------------------------------------------- */
    /* if no '[' character in the input string, then we are done. */
    /* ---------------------------------------------------------- */
    if ptr3.is_none() {
        return *status;
    }
    /* ------------------------------------------- */
    /* see if [ extension specification ] is given */
    /* ------------------------------------------- */
    if plus_ext == 0 {
        /* extension no. not already specified?  Then      */
        /* first brackets must enclose extension name or # */
        /* or it encloses a image subsection specification */
        /* or a raw binary image specifier */
        /* or a image compression specifier */
        /* Or, the extension specification may have been */
        /* omitted and we have to guess what the user intended */

        let ptr1_index = ptr3_index + 1; /* pointer to first char after the [ */
        let ptr2 = strchr_safe(&tmptr[ptr1_index..], bb(b']')); /* search for closing ] */
        if ptr2.is_none() {
            ffpmsg_str("input file URL is missing closing bracket ']'"); /* error, no closing ] */
            *status = URL_PARSE_ERROR;
            return *status;
        }
        let ptr2_abs = ptr1_index + ptr2.unwrap();
        /* ---------------------------------------------- */
        /* First, test if this is a rawfile specifier     */
        /* which looks something like: '[ib512,512:2880]' */
        /* Test if first character is b,i,j,d,r,f, or u,  */
        /* and optional second character is b or l,       */
        /* followed by one or more digits,                */
        /* finally followed by a ',', ':', or ']'         */
        /* ---------------------------------------------- */
        let mut ptr1_scan = ptr1_index;
        if tmptr[ptr1_scan] == bb(b'b')
            || tmptr[ptr1_scan] == bb(b'B')
            || tmptr[ptr1_scan] == bb(b'i')
            || tmptr[ptr1_scan] == bb(b'I')
            || tmptr[ptr1_scan] == bb(b'j')
            || tmptr[ptr1_scan] == bb(b'J')
            || tmptr[ptr1_scan] == bb(b'd')
            || tmptr[ptr1_scan] == bb(b'D')
            || tmptr[ptr1_scan] == bb(b'r')
            || tmptr[ptr1_scan] == bb(b'R')
            || tmptr[ptr1_scan] == bb(b'f')
            || tmptr[ptr1_scan] == bb(b'F')
            || tmptr[ptr1_scan] == bb(b'u')
            || tmptr[ptr1_scan] == bb(b'U')
        {
            /* next optional character may be a b or l (for Big or Little) */
            ptr1_scan += 1; /* must have at least 1 digit */
            if tmptr[ptr1_scan] == bb(b'b')
                || tmptr[ptr1_scan] == bb(b'B')
                || tmptr[ptr1_scan] == bb(b'l')
                || tmptr[ptr1_scan] == bb(b'L')
            {
                ptr1_scan += 1; /* skip over digits */
            } /* OK, this looks like a rawfile specifier */

            if isdigit_safe(tmptr[ptr1_scan]) {
                while isdigit_safe(tmptr[ptr1_scan]) {
                    ptr1_scan += 1; /* append the raw array specifier to infilex */
                } /* find the closing ] char */
                if tmptr[ptr1_scan] == bb(b',')
                    || tmptr[ptr1_scan] == bb(b':')
                    || tmptr[ptr1_scan] == bb(b']')
                {
                    if let Some(ref mut s) = urltype {
                        if strstr_safe(s, cs!(c"stdin")).is_some() {
                            strcpy_safe(s, cs!(c"rawstdin://")); /* terminate string after the ] */
                        } else {
                            strcpy_safe(s, cs!(c"rawfile://")); /* the 0 ext number is implicit */
                        }; /* search for another [ char */
                    } /* copy any remaining characters into rowfilterx  */
                    if let Some(ref mut s) = infilex {
                        if strlen_safe(s) + strlen_safe(&tmptr[ptr3_index..]) > FLEN_FILENAME - 1 {
                            *status = URL_PARSE_ERROR; /* overwrite the ] with null terminator */
                            return *status; /* finished parsing, so return */
                        } /* end of rawfile specifier test */
                        strcat_safe(s, &tmptr[ptr3_index..]);
                        let ptr1_close = strchr_safe(s, bb(b']'));
                        if let Some(idx) = ptr1_close {
                            s[idx + 1] = 0;
                        };
                    }
                    if let Some(ref mut s) = extspec {
                        strcpy_safe(s, cast_slice(b"0\0"));
                    }
                    let tmptr_next = strchr_safe(&tmptr[(ptr2_abs + 1)..], bb(b'['));
                    if let Some(next_bracket) = tmptr_next
                        && let Some(ref mut s) = rowfilterx
                    {
                        let next_abs = ptr2_abs + 1 + next_bracket;
                        if strlen_safe(s) + strlen_safe(&tmptr[(next_abs + 1)..])
                            > FLEN_FILENAME - 1
                        {
                            *status = URL_PARSE_ERROR;
                            return *status;
                        }
                        strcat_safe(s, &tmptr[(next_abs + 1)..]);
                        let tmptr_close = strchr_safe(s, bb(b']'));
                        if let Some(idx) = tmptr_close {
                            s[idx] = 0;
                        };
                    }
                    return *status;
                };
            };
        }
        /* -------------------------------------------------------- */
        /* Not a rawfile, so next, test if this is an image section */
        /* i.e., an integer followed by a ':' or a '*' or '-*'      */
        /* -------------------------------------------------------- */
        let mut tmptr_scan = ptr1_index; /* reset pointer to first char after the [ */
        /* skip leading blanks */
        while tmptr[tmptr_scan] == bb(b' ') {
            tmptr_scan += 1; /* skip over leading digits */
        } /* this is an image section specifier */
        while isdigit_safe(tmptr[tmptr_scan]) {
            tmptr_scan += 1;
        }
        if tmptr[tmptr_scan] == bb(b':')
            || tmptr[tmptr_scan] == bb(b'*')
            || tmptr[tmptr_scan] == bb(b'-')
        {
            strcat_safe(&mut rowfilter, &tmptr[ptr3_index..]);
            /*
            don't want to assume 0 extension any more; may imply an image extension.
                if (extspec)
                   strcpy_safe(extspec, "0");
            */
        } else {
            /* -----------------------------------------------------------------
            Not an image section or rawfile spec so may be an extension spec.

            Examples of valid extension specifiers:
               [3]                - 3rd extension; 0 = primary array
               [events]           - events extension
               [events, 2]        - events extension, with EXTVER = 2
               [events,2]         - spaces are optional
               [events, 3, b]     - same as above, plus XTENSION = 'BINTABLE'
               [PICS; colName(12)] - an image in row 12 of the colName column
                                         in the PICS table extension
               [PICS; colName(exposure > 1000)] - as above, but find image in
                             first row with with exposure column value > 1000.
               [Rate Table] - extension name can contain spaces!
               [Rate Table;colName(exposure>1000)]

            Examples of other types of specifiers (Not extension specifiers)

               [bin]  !!! this is ambiguous, and can't be distinguished from
                          a valid extension specifier
               [bini X=1:512:16]  (also binb, binj, binr, and bind are allowed)
               [binr (X,Y) = 5]
               [bin @binfilter.txt]

               [col Time;rate]
               [col PI=PHA * 1.1]
               [col -Time; status]

               [X > 5]
               [X>5]
               [@filter.txt]
               [StatusCol]  !!! this is ambiguous, and can't be distinguished
                          from a valid extension specifier
               [StatusCol==0]
               [StatusCol || x>6]
               [gtifilter()]
               [regfilter(c"region.reg")]

               [compress Rice]

            There will always be some ambiguity between an extension name and
            a boolean row filtering expression, (as in a couple of the above
            examples).  If there is any doubt, the expression should be treated
            as an extension specification;  The user can always add an explicit
            expression specifier to override this interpretation.

            The following decision logic will be used:

            1) locate the first token, terminated with a space, comma,
               semi-colon, or closing bracket.

            2) the token is not part of an extension specifier if any of
               the following is true:

               - if the token begins with '@' and contains a '.'
               - if the token contains an operator: = > < || &&
               - if the token begins with c"gtifilter(" or "regfilter("
               - if the token is terminated by a space and is followed by
                  additional characters (not a ']')  AND any of the following:
                    - the token is 'col'
                    - the token is 3 or 4 chars long and begins with 'bin'
                    - the second token begins with an operator:
                        ! = < > | & + - * / %


            3) otherwise, the string is assumed to be an extension specifier

            ----------------------------------------------------------------- */
            let mut tmptr_idx = ptr1_index; /* test for leading @ symbol */
            while tmptr[tmptr_idx] == bb(b' ') {
                tmptr_idx += 1; /* parse the first token of the expression */
            } /* a space char? */
            hasAt = 0; /* skip spaces */
            hasDot = 0; /* is this the end? */
            hasOper = 0; /* 1st token is terminated by space */
            followingOper = 0; /* test if this is a column or binning specifier */
            spaceTerm = 0; /* check if next character is an operator */
            rowFilter = 0; /* test if this is NOT an extension specifier */
            colStart = 0; /* this is (probably) not an extension specifier */
            binStart = 0; /* so copy all chars to filter spec string */
            pixStart = 0; /* this appears to be a legit extension specifier */
            compStart = 0; /* copy the extension specification */
            if tmptr[tmptr_idx] == bb(b'@') {
                hasAt = 1; /* copy any remaining chars to filter spec string */
            } /* end of  if (!plus_ext)     */

            let _tp = &tmptr[tmptr_idx..];

            if fits_strncasecmp(_tp, cs!(c"col "), 4) == 0 {
                colStart = 1;
            }
            if fits_strncasecmp(_tp, cs!(c"bin"), 3) == 0 {
                binStart = 1;
            }
            if fits_strncasecmp(_tp, cs!(c"pix"), 3) == 0 {
                pixStart = 1;
            }
            if fits_strncasecmp(_tp, cs!(c"compress "), 9) == 0
                || fits_strncasecmp(_tp, cs!(c"compress]"), 9) == 0
            {
                compStart = 1;
            }
            if fits_strncasecmp(_tp, cs!(c"gtifilter("), 10) == 0
                || fits_strncasecmp(_tp, cs!(c"regfilter("), 10) == 0
            {
                rowFilter = 1;
            } else {
                let mut ii = 0;
                let t = ptr2.unwrap();
                let mut scan_idx = ptr1_index;
                while ii < t + 1 {
                    if tmptr[scan_idx] == bb(b'.') {
                        hasDot = 1;
                    } else if tmptr[scan_idx] == bb(b'=')
                        || tmptr[scan_idx] == bb(b'>')
                        || tmptr[scan_idx] == bb(b'<')
                        || (tmptr[scan_idx] == bb(b'|') && tmptr[scan_idx + 1] == bb(b'|'))
                        || (tmptr[scan_idx] == bb(b'&') && tmptr[scan_idx + 1] == bb(b'&'))
                    {
                        hasOper = 1;
                    } else if tmptr[scan_idx] == bb(b',')
                        || tmptr[scan_idx] == bb(b';')
                        || tmptr[scan_idx] == bb(b']')
                    {
                        break;
                    } else if tmptr[scan_idx] == bb(b' ') {
                        while tmptr[scan_idx] == bb(b' ') {
                            scan_idx += 1;
                        }
                        if tmptr[scan_idx] == bb(b']') {
                            break;
                        }
                        spaceTerm = 1;
                        if colStart != 0 || (ii <= 4 && (binStart != 0 || pixStart != 0)) {
                            rowFilter = 1;
                        } else if tmptr[scan_idx] == bb(b'=')
                            || tmptr[scan_idx] == bb(b'>')
                            || tmptr[scan_idx] == bb(b'<')
                            || tmptr[scan_idx] == bb(b'|')
                            || tmptr[scan_idx] == bb(b'&')
                            || tmptr[scan_idx] == bb(b'!')
                            || tmptr[scan_idx] == bb(b'+')
                            || tmptr[scan_idx] == bb(b'-')
                            || tmptr[scan_idx] == bb(b'*')
                            || tmptr[scan_idx] == bb(b'/')
                            || tmptr[scan_idx] == bb(b'%')
                        {
                            followingOper = 1;
                        }
                        break;
                    };
                    {
                        ii += 1;
                        scan_idx += 1
                    }
                }
            }
            if rowFilter != 0
                || (pixStart != 0 && spaceTerm != 0)
                || (hasAt != 0 && hasDot != 0)
                || hasOper != 0
                || compStart != 0
                || (spaceTerm != 0 && followingOper != 0)
            {
                strcat_safe(&mut rowfilter, &tmptr[ptr3_index..]);
            } else {
                if let Some(ref mut s) = extspec {
                    let t = ptr2.unwrap();
                    if t > FLEN_FILENAME - 1 {
                        *status = URL_PARSE_ERROR;
                        return *status;
                    }

                    strncat_safe(s, &tmptr[ptr1_index..], t);
                }
                strcat_safe(&mut rowfilter, &tmptr[(ptr2_abs + 1)..]);
            };
        };
    } else {
        /* ------------------------------------------------------------------ */
        /* already have extension, so this must be a filter spec of some sort */
        /* ------------------------------------------------------------------ */
        strcat_safe(&mut rowfilter, &tmptr[ptr3_index..]);
    }

    /* strip off any trailing blanks from filter */
    slen = strlen_safe(&rowfilter) as isize;
    slen -= 1;
    while slen >= 0 && rowfilter[slen as usize] == bb(b' ') {
        rowfilter[slen as usize] = 0;
    }

    if rowfilter[0] == 0 {
        return *status; /* nothing left to parse */
    }

    /* ------------------------------------------------ */
    /* does the filter contain a binning specification? */
    /* ------------------------------------------------ */

    let mut ptr1 = strstr_safe(&rowfilter, cs!(c"[bin")); /* search for "[bin" */

    if ptr1.is_none() {
        ptr1 = strstr_safe(&rowfilter, cs!(c"[BIN")); /* search for "[BIN" */
    }

    if ptr1.is_none() {
        ptr1 = strstr_safe(&rowfilter, cs!(c"[Bin")); /* search for "[Bin" */
    }

    if let Some(p1) = ptr1 {
        let mut p2 = p1 + 4; /* end of the '[bin' string */
        if rowfilter[p2] == bb(b'b')
            || rowfilter[p2] == bb(b'i')
            || rowfilter[p2] == bb(b'j')
            || rowfilter[p2] == bb(b'r')
            || rowfilter[p2] == bb(b'd')
        {
            p2 += 1; /* skip the datatype code letter */
        }
        if rowfilter[p2] != bb(b' ') && rowfilter[p2] != bb(b']') {
            ptr1 = None; /* bin string must be followed by space or ] */
        };
    }

    if let Some(p1) = ptr1 {
        /* found the binning string */
        if let Some(binspec_slice) = binspec {
            if strlen_safe(&rowfilter[(p1 + 1)..]) > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return *status;
            }

            strcpy_safe(binspec_slice, &rowfilter[(p1 + 1)..]);
            let binspec = binspec_slice;

            let ptr2 = fits_find_match_delim(binspec, bb(b']'));
            if let Some(mut p2) = ptr2 {
                p2 -= 1;
                binspec[p2] = 0;

                p2 -= 1;
                if binspec[p2] == bb(b' ') {
                    binspec[p2] = 0;
                };
            } else {
                ffpmsg_str("input file URL is missing closing bracket ']'");
                ffpmsg_slice(&rowfilter);
                *status = URL_PARSE_ERROR;
                return *status;
            };
        }

        /* delete the binning spec from the row filter string */
        let ptr2 = fits_find_match_delim(&rowfilter[(p1 + 1)..], bb(b']'));
        if let Some(mut p2) = ptr2 {
            p2 = p2 + p1 + 1;

            strcpy_safe(&mut tmpstr, &rowfilter[p2..]); /* copy any chars after the binspec */
            strcpy_safe(&mut rowfilter[p1..], &tmpstr); /* overwrite binspec */
        } else {
            ffpmsg_str("input file URL is missing closing bracket ']'");
            ffpmsg_slice(&rowfilter);
            *status = URL_PARSE_ERROR; /* error, no closing ] */
            return *status;
        };
    }

    /* --------------------------------------------------------- */
    /* does the filter contain a column selection specification? */
    /* --------------------------------------------------------- */

    let mut ptr1 = strstr_safe(&rowfilter, cs!(c"[col "));
    if ptr1.is_none() {
        ptr1 = strstr_safe(&rowfilter, cs!(c"[COL "));
        if ptr1.is_none() {
            ptr1 = strstr_safe(&rowfilter, cs!(c"[Col "));
        };
    }

    hasAt = 0;

    while ptr1.is_some() {
        let p1 = ptr1.unwrap();

        /* find the end of the column specifier */
        let mut p2 = p1 + 5;
        /* Scan past any whitespace and check for @filename */
        while rowfilter[p2] == bb(b' ') {
            p2 += 1; /* error, no closing ] */
        } /* start of a literal string */

        if rowfilter[p2] == bb(b'@') {
            hasAt = 1; /* find closing quote */
        } /* error, no closing ] */

        while rowfilter[p2] != bb(b']') {
            if rowfilter[p2] == 0 {
                ffpmsg_str("input file URL is missing closing bracket ']'"); /* set of nested square brackets */
                /* find closing bracket */
                *status = URL_PARSE_ERROR; /* error, no closing ] */
                return *status; /* continue search for the closing bracket character */
            } /* copy the column specifier to output string */

            if rowfilter[p2] == bb(b'\\') {
                let _ptr2 = strchr_safe(&rowfilter[(p2 + 1)..], bb(b'\\')); /* Pre-existing colspec, append with ";" */
                if let Some(_ptr2) = _ptr2 {
                    p2 += _ptr2;
                } else {
                    ffpmsg_str("literal string in input file URL is missing closing single quote");
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
            }
            if rowfilter[p2] == bb(b'[') {
                let _ptr2 = strchr_safe(&rowfilter[(p2 + 1)..], bb(b']'));
                if let Some(_ptr2) = _ptr2 {
                    p2 += _ptr2;
                } else {
                    ffpmsg_str("nested brackets in input file URL is missing closing bracket");
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
            }

            p2 += 1;
        }

        let collen = p2 - p1 - 1;

        if let Some(ref mut colspec_slice) = colspec {
            /* copy the column specifier to output string */

            if collen + strlen_safe(colspec_slice) > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return *status;
            }

            if colspec_slice[0] == 0 {
                strncpy_safe(colspec_slice, &rowfilter[(p1 + 1)..], collen);
                colspec_slice[collen] = 0;
            } else {
                strcat_safe(colspec_slice, cs!(c";"));
                strncat_safe(colspec_slice, &rowfilter[(p1 + 5)..], collen - 4);
                /* Note that strncat always null-terminates the destination string */

                /* Special error checking here.  We can't allow there to be a
                col @filename.txt includes if there are multiple col expressions */
                if hasAt != 0 {
                    ffpmsg_str("input URL multiple column filter cannot use @filename.txt");
                    *status = URL_PARSE_ERROR;
                    return *status;
                };
            }

            let mut collen = strlen_safe(colspec_slice) as isize;
            collen -= 1;
            while colspec_slice[collen as usize] == bb(b' ') {
                colspec_slice[collen as usize] = 0; /* strip trailing blanks */
            }
        }

        /* delete the column selection spec from the row filter string */
        strcpy_safe(&mut tmpstr, &rowfilter[(p2 + 1)..]); /* copy any chars after the colspec */
        strcpy_safe(&mut rowfilter[p1..], &tmpstr); /* overwrite binspec */

        /* Check for additional column specifiers */
        ptr1 = strstr_safe(&rowfilter, cs!(c"[col "));
        if ptr1.is_none() {
            ptr1 = strstr_safe(&rowfilter, cs!(c"[COL "));
        }
        if ptr1.is_none() {
            ptr1 = strstr_safe(&rowfilter, cs!(c"[Col "));
        };
    }

    /* --------------------------------------------------------- */
    /* does the filter contain a pixel filter specification?     */
    /* --------------------------------------------------------- */

    let mut ptr1 = strstr_safe(&rowfilter, cs!(c"[pix"));
    if ptr1.is_none() {
        ptr1 = strstr_safe(&rowfilter, cs!(c"[PIX"));
        if ptr1.is_none() {
            ptr1 = strstr_safe(&rowfilter, cs!(c"[Pix"));
        };
    }

    let mut p2 = 0;
    if let Some(p1) = ptr1 {
        p2 = p1 + 4; /* end of the '[pix' string */

        if rowfilter[p2] == bb(b'b')
            || rowfilter[p2] == bb(b'i')
            || rowfilter[p2] == bb(b'j')
            || rowfilter[p2] == bb(b'B')
            || rowfilter[p2] == bb(b'I')
            || rowfilter[p2] == bb(b'J')
            || rowfilter[p2] == bb(b'r')
            || rowfilter[p2] == bb(b'd')
            || rowfilter[p2] == bb(b'R')
            || rowfilter[p2] == bb(b'D')
        {
            p2 += 1 /* skip the datatype code letter */
        }
        if rowfilter[p2] == bb(b'1') {
            p2 += 1
        }
        if rowfilter[p2] != bb(b' ') {
            ptr1 = None;
        };
    }

    if let Some(p1) = ptr1 {
        while rowfilter[p2] != bb(b']') {
            if rowfilter[p2] == 0 {
                ffpmsg_str("input file URL is missing closing bracket ']'"); /* copy the column specifier to output string */

                *status = URL_PARSE_ERROR;
                return *status;
            }

            if rowfilter[p2] == bb(b'\\') {
                let _ptr2 = strchr_safe(&rowfilter[(p2 + 1)..], bb(b'\\'));
                if let Some(_ptr2) = _ptr2 {
                    p2 += _ptr2;
                } else {
                    ffpmsg_str("literal string in input file URL is missing closing single quote");
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
            }

            if rowfilter[p2] == bb(b'[') {
                let _ptr2 = strchr_safe(&rowfilter[(p2 + 1)..], bb(b']'));
                if let Some(_ptr2) = _ptr2 {
                    p2 += _ptr2;
                } else {
                    ffpmsg_str("nested brackets in input file URL is missing closing bracket");
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
            }

            p2 += 1;
        }

        let mut collen = p2 - p1 - 1;

        if let Some(ref mut pixfilter_slice) = pixfilter {
            if collen as usize > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return *status;
            }
            strncpy_safe(pixfilter_slice, &rowfilter[(p1 + 1)..], collen);
            pixfilter_slice[collen] = 0;

            collen -= 1;
            while pixfilter_slice[collen as usize] == bb(b' ') {
                pixfilter_slice[collen as usize] = 0;
            }
        }
        /* delete the pixel filter from the row filter string */
        strcpy_safe(&mut tmpstr, &rowfilter[(p2 + 1)..]); /* copy any chars after the pixel filter */
        strcpy_safe(&mut rowfilter[p1..], &tmpstr); /* overwrite binspec */
    }

    /* ------------------------------------------------------------ */
    /* does the filter contain an image compression specification?  */
    /* ------------------------------------------------------------ */

    let mut ptr1 = strstr_safe(&rowfilter, cs!(c"[compress")); /* end of the '[compress' string */

    if ptr1.is_some() {
        let p2 = ptr1.unwrap() + 9; /* compress string must be followed by space or ] */
        if rowfilter[p2] != bb(b' ') && rowfilter[p2] != bb(b']') {
            ptr1 = None;
        };
    }

    if let Some(p1) = ptr1 {
        /* found the compress string */
        if let Some(ref mut compspec_slice) = compspec {
            if strlen_safe(&rowfilter[(p1 + 1)..]) > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR; /* delete trailing spaces */
                return *status; /* error, no closing ] */
            }

            strcpy_safe(compspec_slice, &rowfilter[(p1 + 1)..]);

            let ptr2 = strchr_safe(compspec_slice, bb(b']'));
            if let Some(p2_idx) = ptr2 {
                compspec_slice[p2_idx] = 0;
                if p2_idx > 0 && compspec_slice[p2_idx - 1] == bb(b' ') {
                    compspec_slice[p2_idx - 1] = 0;
                }
            } else {
                ffpmsg_str("input file URL is missing closing bracket ']'");
                ffpmsg_slice(&rowfilter);
                *status = URL_PARSE_ERROR;
                return *status;
            };
        }

        /* delete the compression spec from the row filter string */
        let ptr2 = strchr_safe(&rowfilter[p1..], bb(b']'));
        strcpy_safe(&mut tmpstr, &rowfilter[(p1 + ptr2.unwrap() + 1)..]); /* copy any chars after the binspec */
        strcpy_safe(&mut rowfilter[p1..], &tmpstr); /* overwrite binspec */
    }

    /* copy the remaining string to the rowfilter output... should only */
    /* contain a rowfilter expression of the form c"[expr]"
     */
    if let Some(ref mut rowfilterx_slice) = rowfilterx
        && rowfilter[0] != 0
    {
        hasAt = 0;

        /* Check for multiple expressions, which would appear as c"[expr][expr]..." */
        let mut p1 = 0; // rowfilter;
        let mut p2 = strstr_safe(&rowfilter, cs!(c"][")).unwrap_or(0);

        while (rowfilter[p1] == bb(b'[')) && p2 > 2 {
            /* Advance past any white space */
            let mut p3 = p1 + 1;
            while rowfilter[p3] == bb(b' ') {
                p3 += 1;
            }

            /* Check for @filename.txt */
            if rowfilter[p3] == bb(b'@') {
                hasAt = 1;
            }

            /* Add expression of the form c"((expr))&&", note the addition of 6 characters */
            if (strlen_safe(rowfilterx_slice) + (p2 - p1) + 6) > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return *status;
            }

            /* Special error checking here.  We can't allow there to be a
            @filename.txt includes if there are multiple row expressions */
            if rowfilterx_slice[0] != 0 && hasAt != 0 {
                ffpmsg_str("input URL multiple row filter cannot use @filename.txt");
                *status = URL_PARSE_ERROR;
                return *status;
            }

            /* Append the expression */
            strcat_safe(rowfilterx_slice, cs!(c"(("));
            strncat_safe(rowfilterx_slice, &rowfilter[(p1 + 1)..], p2 - p1 - 1);
            /* Note that strncat always null-terminates the destination string */
            strcat_safe(rowfilterx_slice, cs!(c"))&&"));

            /* Advance to next expression */
            p1 = p2 + 1;

            p2 = strstr_safe(&rowfilter, cs!(c"][")).unwrap() - p1;
        }

        /* At final iteration, ptr1 points to beginning [ and ptr2 to ending ] */
        let p2 = strlen_safe(&rowfilter) - 1;
        if rowfilter[p1] == bb(b'[') && rowfilter[p2] == bb(b']') {
            /* Check for @include in final position */
            let mut p3 = p1 + 1;
            while rowfilter[p3] == bb(b' ') {
                p3 += 1;
            }

            if rowfilter[p3] == bb(b'@') {
                hasAt = 1;
            }

            /* Check for overflow; add extra 4 characters if we have pre-existing expression */
            if strlen_safe(rowfilterx_slice)
                + (p2 - p1 - 1)
                + (if rowfilterx_slice[0] != 0 { 4 } else { 0 })
                > FLEN_FILENAME - 1
            {
                *status = URL_PARSE_ERROR;
                return *status;
            }

            /* Special error checking here.  We can't allow there to be a
            @filename.txt includes if there are multiple row expressions */
            if rowfilterx_slice[0] != 0 && hasAt != 0 {
                ffpmsg_str("input URL multiple row filter cannot use @filename.txt");

                *status = URL_PARSE_ERROR;
                return *status;
            }

            if rowfilterx_slice[0] != 0 {
                /* A pre-existing row filter: we bracket by ((expr)) to be sure */
                strcat_safe(rowfilterx_slice, cs!(c"(("));
                strncat_safe(rowfilterx_slice, &rowfilter[(p1 + 1)..], p2 - p1 - 1);
                strcat_safe(rowfilterx_slice, cs!(c"))"));
            } else {
                /* We have only one filter, so just copy the expression alone.
                This will be the most typical case */

                strncat_safe(rowfilterx_slice, &rowfilter[(p1 + 1)..], p2 - p1 - 1);
            };
        } else {
            ffpmsg_str("input file URL lacks valid row filter expression");
            *status = URL_PARSE_ERROR;
        };
    }

    *status
}

/// test if the input file specifier is an existing file on disk
/// If the specified file can't be found, it then searches for a
/// compressed version of the file.
///
/// # Parameters
///
/// * `infile` — (I) input filename or URL
/// * `exists` — (O) 2 = a compressed version of file exists
/// * `status` — I/O status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffexist(
    infile: *const c_char,
    exists: *mut c_int,
    /*      1 = yes, disk file exists               */
    /*      0 = no, disk file could not be found    */
    /*     -1 = infile is not a disk file (could    */
    /*   be a http, ftp, gsiftp, smem, or stdin file) */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let exists = exists.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(infile);

        ffexist_safe(infile, exists, status)
    }
}

/// test if the input file specifier is an existing file on disk
/// If the specified file can't be found, it then searches for a
/// compressed version of the file.
///
/// # Parameters
///
/// * `infile` — (I) input filename or URL
/// * `exists` — (O) 2 = a compressed version of file exists
/// * `status` — I/O - error status
pub fn ffexist_safe(
    infile: &[c_char],
    exists: &mut c_int,
    /*      1 = yes, disk file exists               */
    /*      0 = no, disk file could not be found    */
    /*     -1 = infile is not a disk file (could    */
    /*   be a http, ftp, gsiftp, smem, or stdin file) */
    status: &mut c_int,
) -> c_int {
    let mut rootname: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    if *status > 0 {
        return *status;
    }

    /* strip off any extname or filters from the name */
    ffrtnm_safe(infile, &mut rootname, status);

    /* the C walks `rootname` with a `char *ptr1`; we use an index instead */
    let ptr1_pos = strstr_safe(&rootname, cs!(c"://"));

    let ptr1_idx: usize;
    if ptr1_pos.is_some() || rootname[0] == bb(b'-') {
        if strncmp_safe(&rootname, cs!(c"file"), 4) == 0 {
            ptr1_idx = ptr1_pos.unwrap() + 3; /* start of the disk file name (past "://") */
        } else {
            *exists = -1; /* this is not a disk file */
            return *status;
        }
    } else {
        ptr1_idx = 0; /* ptr1 = rootname */
    }

    /* file_is_compressed wants a full FLEN_FILENAME buffer, so copy the disk
    file name (the C's `ptr1` substring) into its own array */
    let mut diskname: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    strcpy_safe(&mut diskname, &rootname[ptr1_idx..]);

    /* see if the disk file exists */
    let mut diskfile: Option<File> = None;
    if file_openfile(&diskname, 0, &mut diskfile) != 0 {
        /* no, couldn't open file, so see if there is a compressed version */
        if file_is_compressed(&mut diskname) != 0 {
            *exists = 2; /* a compressed version of the file exists */
        } else {
            *exists = 0; /* neither file nor compressed version exist */
        }
    } else {
        /* yes, file exists */
        *exists = 1;
        /* C did fclose(diskfile); dropping the Option<File> closes it */
        drop(diskfile.take());
    }

    *status
}

/// parse the input URL, returning the root name (filetype://basename).
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffrtnm(
    url: *const c_char,
    rootname: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(url);

        let rootname = core::slice::from_raw_parts_mut(rootname, FLEN_FILENAME);

        ffrtnm_safe(url, rootname, status)
    }
}

/// parse the input URL, returning the root name (filetype://basename).
pub fn ffrtnm_safe(url: &[c_char], rootname: &mut [c_char], status: &mut c_int) -> c_int {
    let mut ii: isize;
    let jj: isize;
    let slen: isize;
    let infilelen: isize;

    let mut urltype: [c_char; MAX_PREFIX_LEN] = [0; MAX_PREFIX_LEN];
    let mut infile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    if *status > 0 {
        return *status;
    }

    let mut ptr1 = url; /* C: ptr1 = url */
    rootname[0] = 0;
    urltype[0] = 0;
    infile[0] = 0;

    /*  get urltype (e.g., file://, ftp://, http://, etc.)  */
    if ptr1[0] == bb(b'-')
    /* "-" means read file from stdin */
    {
        strcat_safe(&mut urltype, cs!(c"-"));
        ptr1 = &ptr1[1..];
    } else if strncmp_safe(ptr1, cs!(c"stdin"), 5) == 0 || strncmp_safe(ptr1, cs!(c"STDIN"), 5) == 0
    {
        strcat_safe(&mut urltype, cs!(c"-"));
        ptr1 = &ptr1[5..];
    } else {
        let ptr2 = strstr_safe(ptr1, cs!(c"://"));
        let ptr3 = strstr_safe(ptr1, cs!(c"("));

        /* the urltype follows a '(' character, so it must apply */
        /* to the output file, and is not the urltype of the input file */
        let ptr2 = match (ptr2, ptr3) {
            (Some(p2), Some(p3)) if p3 < p2 => None, /* so reset pointer to zero */
            _ => ptr2,
        };

        if let Some(p2) = ptr2 {
            /* copy the explicit urltype string */

            if p2 + 3 > MAX_PREFIX_LEN - 1 {
                *status = URL_PARSE_ERROR;
                return *status;
            }
            strncat_safe(&mut urltype, ptr1, p2 + 3); /* C: strncat(urltype, ptr1, ptr2-ptr1+3) */
            ptr1 = &ptr1[(p2 + 3)..]; /* C: ptr1 = ptr2 + 3 */
        } else if strncmp_safe(ptr1, cs!(c"ftp:"), 4) == 0 {
            /* the 2 //'s are optional */
            strcat_safe(&mut urltype, cs!(c"ftp://"));
            ptr1 = &ptr1[4..];
        } else if strncmp_safe(ptr1, cs!(c"gsiftp:"), 7) == 0 {
            /* the 2 //'s are optional */
            strcat_safe(&mut urltype, cs!(c"gsiftp://"));
            ptr1 = &ptr1[7..];
        } else if strncmp_safe(ptr1, cs!(c"http:"), 5) == 0 {
            /* the 2 //'s are optional */
            strcat_safe(&mut urltype, cs!(c"http://"));
            ptr1 = &ptr1[5..];
        } else if strncmp_safe(ptr1, cs!(c"mem:"), 4) == 0 {
            /* the 2 //'s are optional */
            strcat_safe(&mut urltype, cs!(c"mem://"));
            ptr1 = &ptr1[4..];
        } else if strncmp_safe(ptr1, cs!(c"shmem:"), 6) == 0 {
            /* the 2 //'s are optional */
            strcat_safe(&mut urltype, cs!(c"shmem://"));
            ptr1 = &ptr1[6..];
        } else if strncmp_safe(ptr1, cs!(c"file:"), 5) == 0 {
            /* the 2 //'s are optional */
            ptr1 = &ptr1[5..];
        }

        /* else assume file driver    */
    }

    /*  get the input file name  */
    let mut ptr2 = strchr_safe(ptr1, bb(b'(')); /* search for opening parenthesis ( */
    let ptr3 = strchr_safe(ptr1, bb(b'[')); /* search for opening bracket [ */
    if let Some(p2_start) = ptr2 {
        /* C: ptr4 = strchr(ptr2, ')') — indices below are absolute offsets into ptr1 */
        let mut ptr4 = strchr_safe(&ptr1[p2_start..], bb(b')')).map(|o| p2_start + o);
        while ptr4.is_some() && ptr2.is_some() {
            let mut p4 = ptr4.unwrap();
            loop {
                p4 += 1;
                if ptr1[p4] != bb(b' ') {
                    break;
                }
            }
            if ptr1[p4] == 0 || ptr1[p4] == bb(b'[') {
                break;
            }
            let p2 = ptr2.unwrap();
            ptr2 = strchr_safe(&ptr1[(p2 + 1)..], bb(b'(')).map(|o| p2 + 1 + o);
            ptr4 = strchr_safe(&ptr1[p4..], bb(b')')).map(|o| p4 + o);
        }
    }

    if ptr2 == ptr3
    /* simple case: no [ or ( in the file name */
    {
        if strlen_safe(ptr1) > FLEN_FILENAME - 1 {
            *status = URL_PARSE_ERROR;
            return *status;
        }

        strcat_safe(&mut infile, ptr1);
    } else if ptr3.is_none()
    /* no bracket, so () enclose output file name */
    {
        let p2 = ptr2.unwrap();
        if p2 > FLEN_FILENAME - 1 {
            *status = URL_PARSE_ERROR;
            return *status;
        }

        strncat_safe(&mut infile, ptr1, p2); /* C: strncat(infile, ptr1, ptr2-ptr1) */

        /* C: ptr2++; ptr1 = strchr(ptr2, ')'); search for closing ) */
        if strchr_safe(&ptr1[(p2 + 1)..], bb(b')')).is_none() {
            *status = URL_PARSE_ERROR; /* error, no closing ) */
            return *status;
        }
    } else if let Some(p2) = ptr2
        && ptr2 < ptr3
    /* () enclose output name before bracket */
    {
        if p2 > FLEN_FILENAME - 1 {
            *status = URL_PARSE_ERROR;
            return *status;
        }

        strncat_safe(&mut infile, ptr1, p2);

        if strchr_safe(&ptr1[(p2 + 1)..], bb(b')')).is_none() {
            *status = URL_PARSE_ERROR; /* error, no closing ) */
            return *status;
        }
    } else
    /*   bracket comes first, so there is no output name */
    {
        let p3 = ptr3.unwrap();
        if p3 > FLEN_FILENAME - 1 {
            *status = URL_PARSE_ERROR;
            return *status;
        }

        strncat_safe(&mut infile, ptr1, p3); /* C: strncat(infile, ptr1, ptr3-ptr1) */
    }

    /* strip off any trailing blanks in the names */
    slen = strlen_safe(&infile) as isize;
    ii = slen - 1;
    while ii > 0 {
        if infile[ii as usize] == bb(b' ') {
            infile[ii as usize] = 0;
        } else {
            break;
        }
        ii -= 1;
    }

    /* --------------------------------------------- */
    /* check if the 'filename+n' convention has been */
    /* used to specifiy which HDU number to open     */
    /* --------------------------------------------- */

    jj = strlen_safe(&infile) as isize;

    ii = jj - 1;
    while ii >= 0 {
        if infile[ii as usize] == bb(b'+') {
            /* search backwards for '+' sign */
            break;
        }
        ii -= 1;
    }

    if ii > 0 && (jj - ii) < 5
    /* limit extension numbers to 4 digits */
    {
        infilelen = ii;
        ii += 1;

        while ii < jj {
            if !isdigit_safe(infile[ii as usize]) {
                /* are all the chars digits? */
                break;
            }
            ii += 1;
        }

        if ii == jj {
            /* yes, the '+n' convention was used.  */

            infile[infilelen as usize] = 0; /* delete the extension number */
        }
    }

    if strlen_safe(&urltype) + strlen_safe(&infile) > FLEN_FILENAME - 1 {
        *status = URL_PARSE_ERROR;
        return *status;
    }

    strcat_safe(rootname, &urltype); /* construct the root name */
    strcat_safe(rootname, &infile);

    *status
}

/// parse the output URL into its basic components.
///
/// # Parameters
///
/// * `url`      — (I) full input URL
/// * `urltype`  — (O) url type
/// * `outfile`  — (O) base file name
/// * `tpltfile` — (O) template file name, if any
/// * `compspec` — (O) compression specification, if any
pub(crate) fn ffourl(
    url: &[c_char],
    urltype: &mut [c_char],
    outfile: &mut [c_char],
    tpltfile: &mut [c_char],
    compspec: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    urltype[0] = 0;
    outfile[0] = 0;
    tpltfile[0] = 0;
    compspec[0] = 0;

    let mut ptr1 = url; // url

    while ptr1[0] == bb(b' ') {
        /* ignore leading blanks */
        ptr1 = &ptr1[1..];
    }

    if ((ptr1[0] == bb(b'-')) && (ptr1[1] == 0 || ptr1[1] == bb(b' ')))
        || strcmp_safe(ptr1, cs!(c"stdout")) == 0
        || strcmp_safe(ptr1, cs!(c"STDOUT")) == 0
    /* "-" means write to stdout;  also support "- "            */
    /* but exclude disk file names that begin with a minus sign */
    /* e.g., "-55d33m.fits"   */
    {
        strcpy_safe(urltype, cs!(c"stdout://"));
    } else {
        /* not writing to stdout */
        /*  get urltype (e.g., file://, ftp://, http://, etc.)  */

        let ptr2 = strstr_safe(ptr1, cs!(c"://"));

        if let Some(ptr2) = ptr2 {
            /* copy the explicit urltype string */

            if ptr2 + 3 > MAX_PREFIX_LEN - 1 {
                *status = URL_PARSE_ERROR;
                return URL_PARSE_ERROR;
            }

            strncat_safe(urltype, ptr1, ptr2 + 3);

            ptr1 = &ptr1[(ptr2 + 3)..];
        } else {
            /* assume file driver    */

            strcat_safe(urltype, cs!(c"file://"));
        }

        /* look for template file name, enclosed in parenthesis */
        let ptr2 = strchr_safe(ptr1, bb(b'('));

        /* look for image compression parameters, enclosed in sq. brackets */
        let mut ptr3 = strchr_safe(ptr1, bb(b'['));

        if let Some(ptr2) = ptr2 {
            /* template file was specified  */
            if ptr2 > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return URL_PARSE_ERROR;
            }

            strncat_safe(outfile, ptr1, ptr2);
        } else if let Some(ptr3) = ptr3 {
            /* compression was specified  */
            if ptr3 > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return URL_PARSE_ERROR;
            }

            strncat_safe(outfile, ptr1, ptr3);
        } else {
            /* no template file or compression */
            if strlen_safe(ptr1) > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return URL_PARSE_ERROR;
            }

            strcpy_safe(outfile, ptr1);
        }

        if let Some(mut ptr2) = ptr2 {
            /* template file was specified  */

            ptr2 += 1;

            let tmp_ptr1 = strchr_safe(&ptr1[ptr2..], bb(b')')); /* search for closing ) */

            if tmp_ptr1.is_none() {
                *status = URL_PARSE_ERROR;
                /* error, no closing ) */
                return URL_PARSE_ERROR;
            }

            let tmp_ptr1 = tmp_ptr1.unwrap();

            if tmp_ptr1 > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return URL_PARSE_ERROR;
            }
            strncat_safe(tpltfile, &ptr1[ptr2..], tmp_ptr1);

            ptr1 = &ptr1[(ptr2 + tmp_ptr1 + 1)..];

            // After processing template, look for compression in the remaining string
            ptr3 = strchr_safe(ptr1, bb(b'['));
        }

        if let Some(mut ptr3) = ptr3 {
            /* compression was specified  */

            ptr3 += 1;

            let tmp_ptr1 = strchr_safe(&ptr1[ptr3..], bb(b']')); /* search for closing ] */

            if tmp_ptr1.is_none() {
                *status = URL_PARSE_ERROR;
                /* error, no closing ] */
                return URL_PARSE_ERROR;
            }

            let tmp_ptr1 = tmp_ptr1.unwrap();

            if tmp_ptr1 > FLEN_FILENAME - 1 {
                *status = URL_PARSE_ERROR;
                return URL_PARSE_ERROR;
            }

            strncat_safe(compspec, &ptr1[ptr3..], tmp_ptr1);

            ptr1 = &ptr1[(ptr3 + tmp_ptr1 + 1)..];
        }

        /* check if a .gz compressed output file is to be created */
        /* by seeing if the filename ends in '.gz'   */
        if strcmp_safe(urltype, cs!(c"file://")) == 0 {
            let ptr1 = strstr_safe(outfile, cs!(c".gz"));
            if let Some(mut ptr1) = ptr1 {
                /* make sure the ".gz" is at the end of the file name */
                ptr1 += 3;
                if outfile[ptr1] == 0 || outfile[ptr1] == bb(b' ') {
                    strcpy_safe(urltype, cs!(c"compressoutfile://"));
                }
            }
        }
    }
    *status
}

/// Parse the input extension specification string, returning either the
/// extension number or the values of the EXTNAME, EXTVERS, and XTENSION
/// keywords in desired extension. Also return the name of the column containing
/// an image, and an expression to be used to determine which row to use,
/// if present.
///
/// DANGER: Don't know size of extname, imagecolname, rowexpress
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffexts(
    extspec: *const c_char,
    extnum: *mut c_int,
    extname: *mut c_char,
    extvers: *mut c_int,
    hdutype: *mut c_int,
    imagecolname: *mut c_char,
    rowexpress: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let extnum = extnum.as_mut().expect(NULL_MSG);
        let extvers = extvers.as_mut().expect(NULL_MSG);
        let hdutype = hdutype.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        raw_to_slice!(extspec);

        // Using this to test for null pointer.
        let extname = extname.as_mut().expect(NULL_MSG);
        let imagecolname = imagecolname.as_mut().expect(NULL_MSG);
        let rowexpress = rowexpress.as_mut().expect(NULL_MSG);

        let extname = slice::from_raw_parts_mut(extname, FLEN_VALUE);
        let imagecolname = slice::from_raw_parts_mut(imagecolname, FLEN_VALUE);
        let rowexpress = slice::from_raw_parts_mut(rowexpress, FLEN_FILENAME);

        ffexts_safe(
            extspec,
            extnum,
            extname,
            extvers,
            hdutype,
            imagecolname,
            rowexpress,
            status,
        )
    }
}

/// Parse the input extension specification string, returning either the
/// extension number or the values of the EXTNAME, EXTVERS, and XTENSION
/// keywords in desired extension. Also return the name of the column containing
/// an image, and an expression to be used to determine which row to use,
/// if present.
///
/// DANGER: Don't know size of extname, imagecolname, rowexpress
pub fn ffexts_safe(
    extspec: &[c_char],
    extnum: &mut c_int,
    extname: &mut [c_char],
    extvers: &mut c_int,
    hdutype: &mut c_int,
    imagecolname: &mut [c_char],
    rowexpress: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut slen: usize = 0;
    let mut nvals: c_int = 0;
    let mut notint: c_int = 1;
    let mut tmpname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    *extnum = 0;
    *extvers = 0;
    *hdutype = ANY_HDU;

    extname[0] = 0;
    imagecolname[0] = 0;
    rowexpress[0] = 0;

    // Use these as intermediate variables then copy into the FFI variables
    let mut _extname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut _imagecolname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut _rowexpress: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

    if *status > 0 {
        return *status;
    }

    let mut ptr1 = 0; // ptr to extspec

    while extspec[ptr1] == bb(b' ') {
        /* skip over any leading blanks */
        ptr1 += 1;
    }

    /* is the extension specification a number? */
    if isdigit_safe(extspec[ptr1]) {
        notint = 0; /* looks like extname may actually be the ext. number */

        // Not required anymore since not using strtol from libc
        // set_errno(Errno(0)); /* reset this prior to calling strtol */
        let (r, loc): (LONGLONG, usize) = strtol_safe(&extspec[ptr1..]).unwrap(); /* read the string as an integer */

        *extnum = r as c_int;

        let mut loc = ptr1 + loc;
        while extspec[loc] == bb(b' ') {
            /* skip over trailing blanks */
            loc += 1;
        }

        /* check for read error, or junk following the integer */
        if extspec[loc] != 0 && extspec[loc] != bb(b';')
        /* || (errno().0 == ERANGE) */
        {
            *extnum = 0;
            notint = 1; /* no, extname was not a simple integer after all */

            // Not required anymore since not using strtol from libc
            // set_errno(Errno(0)); /* reset error condition flag if it was set */
        }

        if *extnum < 0 || *extnum > 99999 {
            *extnum = 0; /* this is not a reasonable extension number */
            ffpmsg_str("specified extension number is out of range:");
            ffpmsg_slice(extspec);
            *status = URL_PARSE_ERROR;
            return *status;
        }
    }

    /*  This logic was too simple, and failed on extnames like '1000TEMP'
        where it would try to move to the 1000th extension

        if (isdigit((int) extspec[ptr1]))
        {
            sscanf_d(ptr1, cs!(c"%d"), &mut extnum);
            if (*extnum < 0 || *extnum > 9999)
            {
                *extnum = 0;
                ffpmsg("specified extension number is out of range:");
                ffpmsg_slice(extspec);
                *status = URL_PARSE_ERROR;
                return *status;
            }
        }
    */

    if notint != 0 {
        /* not a number, so EXTNAME must be specified, followed by */
        /* optional EXTVERS and XTENSION  values */

        /* don't use space char as end indicator, because there */
        /* may be imbedded spaces in the EXTNAME value */
        slen = strcspn_safe(&extspec[ptr1..], cs!(c",:;")); /* length of EXTNAME */

        if slen > FLEN_VALUE - 1 {
            *status = URL_PARSE_ERROR;
            return *status;
        }

        strncat_safe(&mut _extname, &extspec[ptr1..], slen); /* EXTNAME value */

        /* now remove any trailing blanks */
        while slen > 0 && (_extname[slen - 1]) == bb(b' ') {
            (_extname[slen - 1]) = 0;
            slen -= 1;
        }

        ptr1 += slen;
        slen = strspn_safe(&extspec[ptr1..], cs!(c" ,:")); /* skip delimiter characters */
        ptr1 += slen;

        slen = strcspn_safe(&extspec[ptr1..], cs!(c" ,:;")); /* length of EXTVERS */
        if slen != 0 {
            nvals = sscanf_d(&extspec[ptr1..], cs!(c"%d"), extvers); /* EXTVERS value */
            if nvals != 1 {
                ffpmsg_str("illegal EXTVER value in input URL:");
                ffpmsg_slice(extspec);
                *status = URL_PARSE_ERROR;
                return *status;
            }

            ptr1 += slen;
            slen = strspn_safe(&extspec[ptr1..], cs!(c" ,:")); /* skip delimiter characters */
            ptr1 += slen;

            slen = strcspn_safe(&extspec[ptr1..], cs!(c";")); /* length of HDUTYPE */
            if slen != 0 {
                if extspec[ptr1] == bb(b'b') || extspec[ptr1] == bb(b'B') {
                    *hdutype = BINARY_TBL;
                } else if extspec[ptr1] == bb(b't')
                    || extspec[ptr1] == bb(b'T')
                    || extspec[ptr1] == bb(b'a')
                    || extspec[ptr1] == bb(b'A')
                {
                    *hdutype = ASCII_TBL;
                } else if extspec[ptr1] == bb(b'i') || extspec[ptr1] == bb(b'I') {
                    *hdutype = IMAGE_HDU;
                } else {
                    ffpmsg_str("unknown type of HDU in input URL:");
                    ffpmsg_slice(extspec);
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
            }
        } else {
            strcpy_safe(&mut tmpname, &_extname);
            ffupch_safe(&mut tmpname);
            if strcmp_safe(&tmpname, cs!(c"PRIMARY")) == 0 || strcmp_safe(&tmpname, cs!(c"P")) == 0
            {
                _extname[0] = 0; /* return extnum = 0 */
            }
        }
    }

    // strchr_safe returns an index relative to the start of the slice, so add
    // the current ptr1 offset to keep it an absolute index into extspec.
    let semicolon = strchr_safe(&extspec[ptr1..], bb(b';')).map(|p| p + ptr1);

    if let Some(mut ptr1) = semicolon {
        /* an image is to be opened; the image is contained in a single */
        /* cell of a binary table.  A column name and an expression to  */
        /* determine which row to use has been entered.                 */

        ptr1 += 1; /* skip over the ';' delimiter */
        while extspec[ptr1] == bb(b' ') {
            /* skip over any leading blanks */
            ptr1 += 1;
        }

        let ptr2 = strchr_safe(&extspec[ptr1..], bb(b'('));
        match ptr2 {
            None => {
                ffpmsg_str("illegal specification of image in table cell in input URL:");
                ffpmsg_str(" did not find a row expression enclosed in ( )");
                ffpmsg_slice(extspec);
                *status = URL_PARSE_ERROR;
                return *status;
            }
            Some(mut ptr2) => {
                ptr2 += ptr1;
                if ptr2 - ptr1 > FLEN_VALUE - 1 {
                    *status = URL_PARSE_ERROR;
                    return *status;
                }

                strncat_safe(&mut _imagecolname, &extspec[ptr1..], ptr2 - ptr1); /* copy column name */

                ptr2 += 1; /* skip over the '(' delimiter */
                while extspec[ptr2] == bb(b' ') {
                    /* skip over any leading blanks */
                    ptr2 += 1;
                }

                let ptr1 = strchr_safe(&extspec[ptr2..], bb(b')')).map(|p| p + ptr2);
                match ptr1 {
                    None => {
                        ffpmsg_str("illegal specification of image in table cell in input URL:");
                        ffpmsg_str(" missing closing ')' character in row expression");
                        ffpmsg_slice(extspec);
                        *status = URL_PARSE_ERROR;
                        return *status;
                    }
                    Some(ptr1) => {
                        if ptr1 - ptr2 > FLEN_FILENAME - 1 {
                            *status = URL_PARSE_ERROR;
                            return *status;
                        }

                        strncat_safe(&mut _rowexpress, &extspec[ptr2..], ptr1 - ptr2);
                        /* row expression */
                    }
                }
            }
        }
    }

    strcpy_safe(extname, &_extname);
    strcpy_safe(imagecolname, &_imagecolname);
    strcpy_safe(rowexpress, &_rowexpress);

    *status
}

/// Parse the input url string and return the number of the extension that
/// CFITSIO would automatically move to if CFITSIO were to open this input URL.
/// The extension numbers are one's based, so 1 = the primary array, 2 = the
/// first extension, etc.
///
/// The extension number that gets returned is determined by the following
/// algorithm:
///
/// 1. If the input URL includes a binning specification (e.g.
///    'myfile.fits[3][bin X,Y]') then the returned extension number
///    will always = 1, since CFITSIO would create a temporary primary
///    image on the fly in this case.  The same is true if an image
///    within a single cell of a binary table is opened.
///
/// 2.  Else if the input URL specifies an extension number (e.g.,
///     `myfile.fits[3]` or `myfile.fits+3`) then the specified extension
///     number (+ 1) is returned.  
///
/// 3.  Else if the extension name is specified in brackets
///     (e.g., this `myfile.fits[EVENTS]`) then the file will be opened and searched
///     for the extension number.  If the input URL is '-'  (reading from the stdin
///     file stream) this is not possible and an error will be returned.
///
/// 4.  Else if the URL does not specify an extension (e.g. 'myfile.fits') then
///     a special extension number = -99 will be returned to signal that no
///     extension was specified.  This feature is mainly for compatibility with
///     existing FTOOLS software.  CFITSIO would open the primary array by default
///     (extension_num = 1) in this case.
///
/// # Parameters
///
/// * `url`           — (I) input filename/URL
/// * `extension_num` — (O) returned extension number
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffextn(
    url: *const c_char,
    extension_num: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let extension_num = extension_num.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        raw_to_slice!(url);

        ffextn_safe(url, extension_num, status)
    }
}

/// Parse the input url string and return the number of the extension that
/// CFITSIO would automatically move to if CFITSIO were to open this input URL.
/// The extension numbers are one's based, so 1 = the primary array, 2 = the
/// first extension, etc.
///
/// The extension number that gets returned is determined by the following
/// algorithm:
///
/// 1. If the input URL includes a binning specification (e.g.
///    'myfile.fits[3][bin X,Y]') then the returned extension number
///    will always = 1, since CFITSIO would create a temporary primary
///    image on the fly in this case.  The same is true if an image
///    within a single cell of a binary table is opened.
///
/// 2.  Else if the input URL specifies an extension number (e.g.,
///     `myfile.fits[3]` or `myfile.fits+3`) then the specified extension
///     number (+ 1) is returned.  
///
/// 3.  Else if the extension name is specified in brackets
///     (e.g., this `myfile.fits[EVENTS]`) then the file will be opened and searched
///     for the extension number.  If the input URL is '-'  (reading from the stdin
///     file stream) this is not possible and an error will be returned.
///
/// 4.  Else if the URL does not specify an extension (e.g. 'myfile.fits') then
///     a special extension number = -99 will be returned to signal that no
///     extension was specified.  This feature is mainly for compatibility with
///     existing FTOOLS software.  CFITSIO would open the primary array by default
///     (extension_num = 1) in this case.
///
/// # Parameters
///
/// * `url`           — (I) input filename/URL
/// * `extension_num` — (O) returned extension number
pub fn ffextn_safe(url: &[c_char], extension_num: &mut c_int, status: &mut c_int) -> c_int {
    let mut fptr: Option<Box<fitsfile>> = None;
    let mut urltype: [c_char; 20] = [0; 20];
    let mut infile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut outfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut extspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut extname: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut rowfilter: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut binspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut colspec: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut imagecolname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut rowexpress: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];
    let mut extnum: c_int = 0;
    let mut extvers: c_int = 0;
    let mut hdutype: c_int = 0;
    let mut tstatus: c_int = 0;

    if *status > 0 {
        return *status;
    }

    /*  parse the input URL into its basic components  */
    ffiurl_safe(
        url,
        Some(&mut urltype),
        Some(&mut infile),
        Some(&mut outfile),
        Some(&mut extspec),
        Some(&mut rowfilter),
        Some(&mut binspec),
        Some(&mut colspec),
        status,
    );

    if *status > 0 {
        return *status;
    }

    if binspec[0] != 0
    /* is there a binning specification? */
    {
        *extension_num = 1; /* a temporary primary array image is created */
        return *status;
    }

    if extspec[0] != 0
    /* is an extension specified? */
    {
        ffexts_safe(
            &extspec,
            &mut extnum,
            &mut extname,
            &mut extvers,
            &mut hdutype,
            &mut imagecolname,
            &mut rowexpress,
            status,
        );

        if *status > 0 {
            return *status;
        }

        if imagecolname[0] != 0
        /* is an image within a table cell being opened? */
        {
            *extension_num = 1; /* a temporary primary array image is created */
            return *status;
        }

        if extname[0] != 0 {
            /* have to open the file to search for the extension name (curses!) */

            if strcmp_safe(&urltype, cs!(c"stdin://")) == 0 {
                /* opening stdin would destroying it! */
                *status = URL_PARSE_ERROR;
                return *status;
            }

            /* First, strip off any filtering specification */
            infile[0] = 0;
            strncat_safe(&mut infile, url, FLEN_FILENAME - 1);

            /* locate the closing bracket */
            match strchr_safe(&infile, bb(b']')) {
                None => {
                    *status = URL_PARSE_ERROR;
                    return *status;
                }
                Some(cptr) => {
                    infile[cptr + 1] = 0; /* terminate URL after the extension spec */
                }
            }

            if ffopen_safe(&mut fptr, &infile, READONLY, status) > 0
            /* open the file */
            {
                if let Some(f) = fptr.take() {
                    ffclos_safe(f, &mut tstatus);
                }
                return *status;
            }

            ffghdn_safe(fptr.as_deref_mut().unwrap(), &mut extnum); /* where am I in the file? */
            *extension_num = extnum;
            if let Some(f) = fptr.take() {
                ffclos_safe(f, status);
            }

            *status
        } else {
            *extension_num = extnum + 1; /* return the specified number (+ 1) */
            *status
        }
    } else {
        *extension_num = -99; /* no specific extension was specified */
        /* defaults to primary array */
        *status
    }
}

/// return the prefix string associated with the driver in use by the
/// fitsfile pointer fptr
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffurlt(
    fptr: *mut fitsfile,
    urlType: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_ref().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let urlType = slice::from_raw_parts_mut(urlType, MAX_PREFIX_LEN);

        ffurlt_safe(fptr, urlType, status)
    }
}

/// return the prefix string associated with the driver in use by the
/// fitsfile pointer fptr
pub fn ffurlt_safe(fptr: &fitsfile, urlType: &mut [c_char], status: &mut c_int) -> c_int {
    let d = DRIVER_TABLE.get().unwrap();

    strcpy_safe(urlType, &(d[fptr.Fptr.driver as usize]).prefix);

    *status
}

/// Read and concatenate all the lines from the given text file.  User
/// must free the pointer returned in contents.  Pointer is guaranteed
/// to hold 2 characters more than the length of the text... allows the
/// calling routine to append (or prepend) a newline (or quotes?) without
/// reallocating memory.
///
/// # Parameters
///
/// * `filename` — Text file to read
/// * `contents` — Pointer to pointer to hold file
/// * `status`   — CFITSIO error code
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffimport_file(
    filename: *const c_char,
    contents: *mut *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(filename);

        let mut safe_contents = None;
        let result = ffimport_file_safe(filename, &mut safe_contents, status);

        if let Some(safe_contents) = safe_contents {
            // HEAP ALLOCATION
            let (p, l, c) = vec_into_raw_parts(safe_contents);
            ALLOCATIONS.lock().unwrap().insert(p as usize, (l, c));

            *contents = p;
        }

        result
    }
}

/// Read and concatenate all the lines from the given text file.  User
/// must free the pointer returned in contents.  Pointer is guaranteed
/// to hold 2 characters more than the length of the text... allows the
/// calling routine to append (or prepend) a newline (or quotes?) without
/// reallocating memory.
///
/// # Parameters
///
/// * `filename` — Text file to read
/// * `contents` — Pointer to pointer to hold file
/// * `status`   — CFITSIO error code
pub fn ffimport_file_safe(
    filename: &[c_char],
    contents: &mut Option<Vec<c_char>>,
    status: &mut c_int,
) -> c_int {
    let mut eoline = true;
    let mut line: [c_char; 256] = [0; 256];

    if *status > 0 {
        return *status;
    }

    let mut totalLen = 0;
    let mut allocLen = 1024;

    // HEAP ALLOCATION
    let mut lines = Vec::new();
    if lines.try_reserve_exact(allocLen).is_err() {
        ffpmsg_str("Couldn't allocate memory to hold ASCII file contents.");
        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        lines.resize(allocLen, 0);
    }
    lines[0] = 0;

    let aFile = File::options()
        .read(true)
        .open(slice_to_str!(filename).as_ref() as &str);

    if aFile.is_err() {
        int_snprintf!(
            &mut line,
            256,
            "Could not open ASCII file {}.",
            slice_to_str!(&filename),
        );
        ffpmsg_slice(&line);

        *status = FILE_NOT_OPENED;
        return *status;
    }

    let mut aFile = aFile.unwrap();

    // Read the file line by line
    while fgets(cast_slice_mut(&mut line), 256, &mut aFile).is_ok() {
        let mut llen = strlen_safe(&line);
        if eoline && (llen > 1) && (line[0] == bb(b'/') && line[1] == bb(b'/')) {
            continue; /* skip comment lines begging with // */
        }

        eoline = false;

        /* replace CR and newline chars at end of line with nulls */
        if (llen > 0) && (line[llen - 1] == bb(b'\n') || line[llen - 1] == bb(b'\r')) {
            llen -= 1;
            line[llen] = 0;
            eoline = true; /* found an end of line character */

            if (llen > 0) && (line[llen - 1] == bb(b'\n') || line[llen - 1] == bb(b'\r')) {
                llen -= 1;
                line[llen] = 0;
            }
        }

        if totalLen + llen + 3 >= allocLen {
            allocLen += 256;

            if lines.try_reserve_exact(256).is_err() {
                ffpmsg_str("Couldn't allocate memory to hold ASCII file contents.");
                *status = MEMORY_ALLOCATION;
                break;
            } else {
                lines.resize(allocLen, 0);
            }
        }
        strcpy_safe(&mut lines[totalLen..], &line);
        totalLen += llen;

        if eoline {
            strcpy_safe(&mut lines[totalLen..], cs!(c" ")); /* add a space between lines */
            totalLen += 1;
        }
    }

    drop(aFile);

    *contents = Some(lines);

    *status
}

/// parse off the next token, delimited by a character in 'delimiter',
/// from the input ptr string;  increment *ptr to the end of the token.
/// Returns the length of the token, not including the delimiter char;
///
/// # Parameters
///
/// * `isanumber` — (O) is this token a number?
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_get_token(
    ptr: *mut *mut c_char,
    delimiter: *const c_char,
    token: *mut c_char,
    isanumber: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let ptr = ptr.as_mut().expect(NULL_MSG);
        let isanumber = isanumber.as_mut();

        raw_to_slice!(delimiter);

        /* The token copied out is a prefix of the remaining input, so the
        most this can write is strlen(*ptr) + 1 chars; the C contract
        requires the caller's buffer to hold at least that. */
        let token_len = CStr::from_ptr(*ptr).to_bytes().len() + 1;
        let token = slice::from_raw_parts_mut(token, token_len);

        fits_get_token_safe(ptr, delimiter, token, isanumber)
    }
}

/// parse off the next token, delimited by a character in 'delimiter',
/// from the input ptr string;  increment *ptr to the end of the token.
/// Returns the length of the token, not including the delimiter char;
///
/// Safe wrapper for fits_get_token - copies token to provided String buffer
///
/// # Parameters
///
/// * `isanumber` — (O) is this token a number?
pub fn fits_get_token_safe(
    ptr: &mut *mut c_char,
    delimiter: &[c_char],
    token: &mut [c_char],
    isanumber: Option<&mut c_int>,
) -> c_int {
    let mut loc: c_char = 0;
    let mut tval: [c_char; 73] = [0; 73];

    let mut input_str: &[c_char] = unsafe { cast_slice(CStr::from_ptr(*ptr).to_bytes_with_nul()) };
    let mut ptr_idx = 0;
    let mut slen: usize = 0;

    token[0] = 0;

    while input_str[0] == bb(b' ') {
        /* skip over leading blanks */
        ptr_idx += 1;
        input_str = &input_str[1..];
    }

    slen = strcspn_safe(input_str, delimiter); /* length of next token */
    if slen != 0 {
        strncat_safe(token, input_str, slen); /* copy token */

        ptr_idx += slen; /* skip over the token */
        input_str = &input_str[slen..];

        /* check if token is a number */
        if let Some(isanumber) = isanumber {
            *isanumber = 1;

            if strchr_safe(token, bb(b'D')).is_some() {
                strncpy_safe(&mut tval, token, 72);
                tval[72] = 0;

                /*  The C language does not support a 'D'; replace with 'E' */
                let tmp_loc = strchr_safe(&tval, bb(b'D'));
                if let Some(tmp_loc) = tmp_loc
                    && tmp_loc != 0
                {
                    tval[tmp_loc] = bb(b'E');
                }

                let mut tmp_loc = 0;
                let _ = strtod_safe(&tval, &mut tmp_loc);
                loc = tval[tmp_loc];
            } else {
                let mut tmp_loc = 0;
                let _ = strtod_safe(token, &mut tmp_loc);
                loc = token[tmp_loc];
            }

            /* check for read error, or junk following the value */
            if loc != 0 && loc != bb(b' ') {
                *isanumber = 0;
            }

            if errno().0 == ERANGE {
                *isanumber = 0;
            }
        }
    }

    /* increment *ptr to the end of the token (past leading blanks + token) */
    *ptr = unsafe { (*ptr).add(ptr_idx) };

    slen as c_int
}

/// parse off the next token, delimited by a character in 'delimiter',
/// from the input ptr string;  increment *ptr to the end of the token.
/// Returns the length of the token, not including the delimiter char;
///
/// This routine allocates the *token string;  the calling routine must free it
///
/// # Parameters
///
/// * `isanumber` — (O) is this token a number?
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_get_token2(
    ptr: *mut *mut c_char,
    delimiter: *const c_char,
    token: *mut *mut c_char,
    isanumber: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let isanumber = isanumber.as_mut();
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(delimiter);

        let mut ptr_index = 0;
        let mut token_vec = None;

        let p: &[c_char] = cast_slice(CStr::from_ptr(*ptr).to_bytes_with_nul());

        let len = fits_get_token2_safe(
            p,
            &mut ptr_index,
            delimiter,
            &mut token_vec,
            isanumber,
            status,
        );

        //HEAP ALLOCATION
        if let Some(mut token_vec) = token_vec {
            *token = token_vec.as_mut_ptr();
            mem::forget(token_vec);
        } else {
            *token = ptr::null_mut();
        }

        *ptr = *ptr.add(ptr_index);

        len
    }
}

/// A sequence of calls to fits_split_names will split the input string
/// into name tokens.  The string typically contains a list of file or
/// column names.  The names must be delimited by a comma and/or spaces.
/// This routine ignores spaces and commas that occur within parentheses,
/// brackets, or curly brackets.  It also strips any leading and trailing
/// blanks from the returned name.
///
/// This routine is similar to the ANSI C 'strtok' function:
///
/// The first call to fits_split_names has a non-null input string.
/// It finds the first name in the string and terminates it by
/// overwriting the next character of the string with a '\0' and returns
/// a pointer to the name.  Each subsequent call, indicated by a NULL
/// value of the input string, returns the next name, searching from
/// just past the end of the previous name.  It returns NULL when no
/// further names are found.
///
/// The following line illustrates how a string would be split into 3 names:
///  myfile[1][bin (x,y)=4], file2.fits  file3.fits
///  ^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^
///    1st name               2nd name    3rd name
///
///
/// NOTE:  This routine is not thread-safe.  
/// This routine is simply provided as a utility routine for other external
/// software. It is not used by any CFITSIO routine.
///
/// # Parameters
///
/// * `list` — (IO) list of names; the name returned is terminated
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_split_names(
    list: *mut c_char,
    /*       in place by overwriting its delimiter with a NUL */
) -> *mut c_char {
    // FFI WRAPPER
    unsafe { fits_split_names_safer(list) }
}

/// The C API cannot be given a safe signature: the returned pointer aliases
/// the caller's buffer, and the cursor kept between calls aliases it too, so
/// no `&mut [c_char]` borrow can span the sequence of calls.  What is done
/// here instead is to measure the buffer once, on the call that supplies it,
/// and drive the whole scan as bounds-checked indexing into a slice; the only
/// unsafe operations left are taking that measurement and rebuilding the slice
/// on each continuation call.
///
/// Callers keep the C's obligations: the buffer passed as `list` must stay
/// alive, in place and unshortened until they stop asking for further names.
pub unsafe fn fits_split_names_safer(list: *mut c_char) -> *mut c_char {
    let mut depth: c_int = 0;

    /* C: `static char *ptr;` -- file-scope state carried between calls, which
    here has to carry the buffer's extent as well so the scan can be bounds
    checked.  The routine is documented above as not thread-safe; a
    thread_local gives each thread its own cursor, which keeps the crate's
    parallel tests honest. */
    #[derive(Clone, Copy)]
    struct SplitCursor {
        base: *mut c_char, /* the caller's buffer */
        len: usize,        /* its length, including the terminating NUL */
        pos: usize,        /* offset of the next character to examine */
    }

    thread_local! {
        static CURSOR: core::cell::Cell<SplitCursor> = const {
            core::cell::Cell::new(SplitCursor {
                base: core::ptr::null_mut(),
                len: 0,
                pos: 0,
            })
        };
    }

    let mut cursor = CURSOR.get();

    if !list.is_null() {
        /* reset ptr if a string is given */
        // SAFETY: the C's contract for a non-NULL `list` is a writable,
        // NUL-terminated string.  Measure it once, here, so that every step
        // below is an index into a slice of known length.
        let len = unsafe { CStr::from_ptr(list) }.to_bytes_with_nul().len();
        cursor = SplitCursor {
            base: list,
            len,
            pos: 0,
        };
    }

    if cursor.base.is_null() {
        /* DEVIATION: the C dereferences its NULL static if the first call does
        not supply a string; report "no names" rather than crash */
        return core::ptr::null_mut();
    }

    // SAFETY: `base` and `len` describe the buffer as measured on the call
    // that supplied it, which the caller undertakes to keep alive and in place
    // for as long as it keeps asking for further names -- the same obligation
    // the C's `static char *ptr` imposes.  `pos` never leaves `0..len`: it only
    // advances while the character under it is neither NUL nor a delimiter,
    // and `buf[len - 1]` is the terminating NUL.
    let buf: &mut [c_char] = unsafe { slice::from_raw_parts_mut(cursor.base, cursor.len) };

    while buf[cursor.pos] == bb(b' ') {
        cursor.pos += 1; /* skip leading white space */
    }

    if buf[cursor.pos] == 0 {
        CURSOR.set(cursor);
        return core::ptr::null_mut(); /* no remaining file names */
    }

    let start = cursor.pos;

    while buf[cursor.pos] != 0 {
        if buf[cursor.pos] == bb(b'[') || buf[cursor.pos] == bb(b'(') || buf[cursor.pos] == bb(b'{')
        {
            depth += 1;
        } else if buf[cursor.pos] == bb(b'}')
            || buf[cursor.pos] == bb(b')')
            || buf[cursor.pos] == bb(b']')
        {
            depth -= 1;
        } else if depth == 0 && (buf[cursor.pos] == bb(b',') || buf[cursor.pos] == bb(b' ')) {
            buf[cursor.pos] = 0; /* terminate the filename here */
            cursor.pos += 1; /* save offset of start of next filename */
            break;
        }
        cursor.pos += 1;
    }

    CURSOR.set(cursor);

    /* The returned pointer is into the caller's own buffer, as in the C, and
    the caller reads the whole NUL-terminated name from it. Derive it from
    the remaining slice: `&mut buf[start]` borrows a single element, so the
    pointer would only carry provenance over that one byte. */
    buf[start..].as_mut_ptr()
}

/// compare input URL with list of known drivers, returning the
/// matching driver numberL.
pub(crate) fn urltype2driver(urltype: &[c_char], driver: &mut c_int) -> c_int {
    let d = DRIVER_TABLE.get().unwrap();
    let mut ii = (d.len() - 1) as isize;

    /* find matching driver; search most recent drivers first */
    while ii >= 0 {
        if 0 == strcmp_safe(&d[ii as usize].prefix, urltype) {
            *driver = ii as c_int;
            return 0;
        }
        ii -= 1;
    }

    NO_MATCHING_DRIVER
}

/// close the FITS file by completing the current HDU, flushing it to disk,
/// then calling the system dependent routine to physically close the FITS file
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffclos(fptr: Option<Box<fitsfile>>, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);

        match fptr {
            None => {
                *status = NULL_INPUT_PTR;
                *status
            }
            Some(fptr) => ffclos_safe(fptr, status),
        }
    }
}

/// close the FITS file by completing the current HDU, flushing it to disk,
/// then calling the system dependent routine to physically close the FITS file
pub fn ffclos_safe(mut fptr: Box<fitsfile>, status: &mut c_int) -> c_int {
    let mut tstatus = NO_CLOSE_ERROR;
    let mut zerostatus = 0;

    if fptr.Fptr.validcode != VALIDSTRUC {
        /* BAD_FILEPTR */
        *status = BAD_FILEPTR;
        return *status;
    }

    /* close and flush the current HDU */
    if *status > 0 {
        ffchdu(&mut fptr, &mut tstatus); /* turn off the error message from ffchdu */
    } else {
        ffchdu(&mut fptr, status);
    }

    (fptr.Fptr.open_count) -= 1; /* decrement usage counter */

    if fptr.Fptr.open_count == 0 {
        /* if no other files use structure */

        ffflsh_safe(&mut fptr, true, status); /* flush and disassociate IO buffers */
        //let d = driverTable.lock().unwrap();
        let d = DRIVER_TABLE.get().unwrap();

        /* call driver function to actually close the file */
        if ((d[fptr.Fptr.driver as usize]).close)(fptr.Fptr.filehandle) != 0 && *status <= 0 {
            *status = FILE_NOT_CLOSED; /* report if no previous error */
            ffpmsg_str("failed to close the following file: (ffclos)");
            ffpmsg_cstr(fptr.Fptr.get_filename_as_cstr());
        };

        fits_clear_Fptr_safer(fptr.Fptr.as_ptr(), status); /* clear Fptr address */

        /* Last handle: release the FITSfile itself. FptrRef is a non-owning
        handle, so this is the one place that frees it. */
        let fitsfile {
            HDUposition: _,
            Fptr,
        } = *fptr;
        unsafe { Fptr.free() };
    } else {
        /*
           to minimize the fallout from any previous error (e.g., trying to
           open a non-existent extension in a already opened file),
           always call ffflsh with status = 0.
        */

        /* just flush the buffers, don't disassociate them */
        if *status > 0 {
            ffflsh_safe(&mut fptr, false, &mut zerostatus);
        } else {
            ffflsh_safe(&mut fptr, false, status);
        }

        /* Other handles still refer to this FITSfile, so only the outer
        fitsfile goes away. Dropping the FptrRef is a no-op, which is why
        this no longer has to leak a Box to avoid a double free. */
        drop(fptr);
    }
    *status
}

/// close and DELETE the FITS file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffdelt(mut fptr: *mut fitsfile, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);

        if fptr.is_null() {
            *status = NULL_INPUT_PTR;
            return *status;
        }

        let mut boxed_fptr = Some(Box::from_raw(fptr));

        let result = ffdelt_safe(&mut (boxed_fptr), status);

        if result != 0 {
            fptr = ptr::null_mut(); /* set to null to  avoid dangling pointer */
        }

        *status
    }
}

/// close and DELETE the FITS file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub fn ffdelt_safe(fptr: &mut Option<Box<fitsfile>>, status: &mut c_int) -> c_int {
    let mut basename: Vec<c_char> = Vec::new();
    let mut slen: c_int = 0;
    let mut tstatus = NO_CLOSE_ERROR;
    let mut zerostatus: c_int = 0;

    let local_fptr = fptr.as_deref_mut().unwrap();

    if local_fptr.Fptr.validcode != VALIDSTRUC {
        /* check for magic value */
        *status = BAD_FILEPTR;
        return *status;
    }

    if *status > 0 {
        ffchdu(local_fptr, &mut tstatus); /* turn off the error message from ffchdu */
    } else {
        ffchdu(local_fptr, status);
    }

    ffflsh_safe(local_fptr, true, status); /* flush and disassociate IO buffers */

    /* call driver function to actually close the file */
    //let d = driverTable.lock().unwrap();
    let d = DRIVER_TABLE.get().unwrap();
    if (d[local_fptr.Fptr.driver as usize].close)(local_fptr.Fptr.filehandle) != 0 && *status <= 0 {
        *status = FILE_NOT_CLOSED; /* report error if no previous error */

        ffpmsg_str("failed to close the following file: (ffdelt)");
        ffpmsg_cstr(local_fptr.Fptr.get_filename_as_cstr());
    }

    /* call driver function to actually delete the file */
    if let Some(remove) = d[local_fptr.Fptr.driver as usize].remove {
        /* parse the input URL to get the base filename */
        slen = strlen_safe(cast_slice(
            local_fptr.Fptr.get_filename_as_cstr().to_bytes_with_nul(),
        )) as c_int;
        if basename.try_reserve_exact((slen + 1) as usize).is_err() {
            *status = MEMORY_ALLOCATION;
            return *status;
        } else {
            basename.resize((slen + 1) as usize, 0)
        }

        let url = local_fptr.Fptr.get_filename_as_cstr().to_bytes_with_nul();

        ffiurl_safe(
            cast_slice(url),
            None,
            Some(&mut basename),
            None,
            None,
            None,
            None,
            None,
            &mut zerostatus,
        );

        if remove(&basename) != 0 {
            ffpmsg_str("failed to delete the following file: (ffdelt)");
            ffpmsg_cstr(local_fptr.Fptr.get_filename_as_cstr());
            if (*status) == 0 {
                *status = FILE_NOT_CLOSED;
            }
        }
    }

    fits_clear_Fptr_safer(local_fptr.Fptr.as_ptr(), status); /* clear Fptr address */
    local_fptr.Fptr.validcode = 0; /* magic value to indicate invalid fptr */

    /* Release the FITSfile. ffdelt deletes the file outright, so unlike ffclos
    there is no use count to consult; the C frees it here unconditionally.
    FptrRef is a non-owning handle, so dropping the outer fitsfile below is
    not enough on its own. */
    if let Some(f) = fptr.take() {
        let fitsfile {
            HDUposition: _,
            Fptr,
        } = *f;
        unsafe { Fptr.free() };
    }

    *status
}

/// low level routine to truncate a file to a new smaller size.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `filesize` — (I) size to truncate the file
/// * `status`   — (O) error status
pub(crate) fn fftrun(fptr: &mut fitsfile, filesize: LONGLONG, status: &mut c_int) -> c_int {
    //let d = driverTable.lock().unwrap();
    let d = DRIVER_TABLE.get().unwrap();
    let truncate = d[fptr.Fptr.driver as usize].truncate;

    if let Some(truncate) = truncate {
        ffflsh_safe(fptr, false, status); /* flush all the buffers first */
        fptr.Fptr.filesize = filesize;
        fptr.Fptr.io_pos = filesize;
        fptr.Fptr.logfilesize = filesize;
        fptr.Fptr.bytepos = filesize;
        ffbfeof(fptr, status); /* eliminate any buffers beyond current EOF */
        *status = truncate(fptr.Fptr.filehandle, filesize as usize);
        *status
    } else {
        *status
    }
}

/// low level routine to flush internal file buffers to the file.
pub(crate) fn ffflushx(fptr: &mut FITSfile, /* I - FITS file pointer                  */) -> c_int {
    let d = DRIVER_TABLE.get().unwrap();
    if let Some(flush) = (d[fptr.driver as usize]).flush {
        flush(fptr.filehandle)
    } else {
        0 /* no flush function defined for this driver */
    }
}

/// low level routine to seek to a position in a file.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `position` — (I) byte position to seek to
pub(crate) fn ffseek(fptr: &mut FITSfile, position: LONGLONG) -> c_int {
    let d = DRIVER_TABLE.get().unwrap();
    (d[fptr.driver as usize].seek)(fptr.filehandle, position)
}

/// low level routine to write bytes to a file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nbytes` — (I) number of bytes to write
/// * `buffer` — (I) buffer to write
/// * `status` — (O) error status
pub(crate) fn ffwrite(
    fptr: &mut FITSfile,
    nbytes: c_long,
    buffer: &[u8],
    status: &mut c_int,
) -> c_int {
    {
        let d = DRIVER_TABLE.get().unwrap();
        if (d[fptr.driver as usize].write)(fptr.filehandle, buffer, nbytes as usize) > 0 {
            ffpmsg_str("Error writing data buffer to file:");
            ffpmsg_cstr(fptr.get_filename_as_cstr());

            *status = WRITE_ERROR;
        }
    }
    *status
}

/// low level routine to write bytes to a file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nbytes` — (I) number of bytes to write
/// * `nbuff`  — (I) buffer offset to write to
/// * `status` — (O) error status
pub(crate) fn ffwrite_int(
    fptr: &mut FITSfile,
    nbytes: usize,
    nbuff: usize,
    status: &mut c_int,
) -> c_int {
    {
        let d = DRIVER_TABLE.get().unwrap();
        let buffer = cast_slice(&fptr.iobuffer[(nbuff * IOBUFLEN as usize)..]);
        if (d[fptr.driver as usize].write)(fptr.filehandle, buffer, nbytes) > 0 {
            ffpmsg_str("Error writing data buffer to file:");
            ffpmsg_cstr(fptr.get_filename_as_cstr());

            *status = WRITE_ERROR;
        }
    }
    *status
}

/// low level routine to read bytes from a file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nbytes` — (I) number of bytes to read
/// * `buffer` — (O) buffer to read into
/// * `status` — (O) error status
pub(crate) fn ffread(
    fptr: &FITSfile,
    nbytes: c_long,
    buffer: &mut [u8],
    status: &mut c_int,
) -> c_int {
    let d = DRIVER_TABLE.get().unwrap();
    let readstatus = (d[fptr.driver as usize].read)(fptr.filehandle, buffer, nbytes as usize);

    if readstatus == END_OF_FILE {
        *status = END_OF_FILE;
    } else if readstatus > 0 {
        ffpmsg_str("Error reading data buffer from file:");
        ffpmsg_cstr(fptr.get_filename_as_cstr());

        *status = READ_ERROR;
    }

    *status
}

/// low level routine to read bytes from a file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nbytes` — (I) number of bytes to read
/// * `nbuff`  — (I) buffer offset to read into
/// * `status` — (O) error status
pub(crate) fn ffread_int(
    fptr: &mut FITSfile,
    nbytes: usize,
    nbuff: usize,
    status: &mut c_int,
) -> c_int {
    let d = DRIVER_TABLE.get().unwrap();
    let buffer = cast_slice_mut(&mut fptr.iobuffer[(nbuff * IOBUFLEN as usize)..]);
    let readstatus = (d[fptr.driver as usize].read)(fptr.filehandle, buffer, nbytes);

    if readstatus == END_OF_FILE {
        *status = END_OF_FILE;
    } else if readstatus > 0 {
        ffpmsg_str("Error reading data buffer from file:");
        ffpmsg_cstr(fptr.get_filename_as_cstr());

        *status = READ_ERROR;
    }

    *status
}

/// Create and initialize a new FITS file  based on a template file.
/// Uses C fopen and fgets functions.
///
/// # Parameters
///
/// * `fptr`     — (O) FITS file pointer
/// * `filename` — (I) name of file to create
/// * `tempname` — (I) name of template file
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fftplt(
    fptr: *mut Option<Box<fitsfile>>,
    filename: *const c_char,
    tempname: *const c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        raw_to_slice!(filename);
        raw_to_slice!(tempname);

        fftplt_safe(fptr, filename, tempname, status)
    }
}

/// Create and initialize a new FITS file  based on a template file.
/// Uses C fopen and fgets functions.
///
/// # Parameters
///
/// * `fptr`     — (O) FITS file pointer
/// * `filename` — (I) name of file to create
/// * `tempname` — (I) name of template file
/// * `status`   — (IO) error status
pub fn fftplt_safe(
    fptr: &mut Option<Box<fitsfile>>,
    filename: &[c_char],
    tempname: &[c_char],
    status: &mut c_int,
) -> c_int {
    /* initialize null file pointer */
    let f_tmp = fptr.take();
    if let Some(f) = f_tmp {
        // WARNING: The c version doesn't null pointers after a close, so we have a dangling pointer.
        // We need to be careful with this, as it can cause double free errors.
        // Therefore, if this function is called with a Some(), then we will leak the pointer because
        // it's probably invalid.
        let _ = Box::into_raw(f);
    }

    /* regardless of the value of *status */
    if *status > 0 {
        return *status;
    }

    if ffinit_safe(fptr, filename, status) != 0 {
        /* create empty file */
        return *status;
    }

    let f = (*fptr).as_mut().expect(NULL_MSG);

    ffoptplt(f, tempname, status); /* open and use template */

    *status
}

/// open template file and use it to create new file
///
/// # Parameters
///
/// * `fptr`     — (O) FITS file pointer
/// * `tempname` — (I) name of template file
/// * `status`   — (IO) error status
pub(crate) fn ffoptplt(fptr: &mut fitsfile, tempname: &[c_char], status: &mut c_int) -> c_int {
    let mut tptr = None;
    let mut tstatus: c_int = 0;
    let mut nkeys: c_int = 0;
    let mut nadd: c_int = 0;

    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    if *status > 0 {
        return *status;
    }

    if tempname[0] == 0 {
        /* no template file? */
        return *status;
    }

    /* try opening template */
    ffopen_safe(&mut tptr, tempname, READONLY, &mut tstatus);

    if tstatus != 0 {
        /* not a FITS file, so treat it as an ASCII template */
        ffxmsg_safe(2, Some(&mut card)); /* clear the  error message */
        fits_execute_template_safe(fptr, tempname, status);

        ffmahd_safe(fptr, 1, None, status); /* move back to the primary array */
        return *status;
    } else {
        /* template is a valid FITS file */
        let mut tptr = tptr.expect(NULL_MSG);

        ffmahd_safe(tptr.as_mut(), 1, None, status); /* make sure we are at the beginning */
        while *status <= 0 {
            ffghsp_safe(tptr.as_mut(), Some(&mut nkeys), Some(&mut nadd), status); /* get no. of keywords */

            for ii in 1..=nkeys {
                /* copy keywords */

                ffgrec_safe(tptr.as_mut(), ii, Some(&mut card), status);

                /* must reset the PCOUNT keyword to zero in the new output file */
                if strncmp_safe(&card, cs!(c"PCOUNT  "), 8) == 0 {
                    /* the PCOUNT keyword? */
                    if strncmp_safe(&card[25..], cs!(c"    0"), 5) != 0 {
                        /* non-zero value? */
                        strncpy_safe(&mut card, cs!(c"PCOUNT  =                    0"), 30);
                    }
                }

                ffprec_safe(fptr, &card, status);
            }

            ffmrhd_safe(tptr.as_mut(), 1, None, status); /* move to next HDU until error */
            ffcrhd_safe(fptr, status); /* create empty new HDU in output file */
        }

        if *status == END_OF_FILE {
            *status = 0; /* expected error condition */
        }
        ffclos_safe(tptr, status); /* close the template file */
    }

    ffmahd_safe(fptr, 1, None, status); /* move to the primary array */
    *status
}

/// Print out report of cfitsio error status and messages on the error stack.
/// Uses C FILE stream.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffrprt(stream: *mut FILE, status: c_int) {
    // FFI WRAPPER
    ffrprt_safe(stream, status)
}

/// Print out report of cfitsio error status and messages on the error stack.
/// Uses C FILE stream.
pub fn ffrprt_safe(stream: *mut FILE, status: c_int) {
    let mut status_str: [c_char; FLEN_STATUS] = [0; FLEN_STATUS];
    let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    if stream.is_null() {
        return; // If the stream is null, do nothing
    }

    let mut cfile_stream = CFile::from(stream);

    if status > 0 {
        ffgerr_safe(status, &mut status_str); /* get the error description */

        let _ = cfile_stream.write_fmt(format_args!(
            "\nFITSIO status = {}: {}\n",
            status,
            CStr::from_bytes_until_nul(cast_slice(&status_str))
                .unwrap()
                .to_str()
                .unwrap()
        ));

        while ffgmsg_safe(&mut errmsg) > 0 {
            /* get error stack messages */
            let _ = cfile_stream.write_fmt(format_args!(
                "{}\n",
                CStr::from_bytes_until_nul(cast_slice(&errmsg))
                    .unwrap()
                    .to_str()
                    .unwrap()
            ));
        }
    }
}

/// # Parameters
///
/// * `fptr`      — (IO) pointer to input image; on output it
/// * `outfile`   — (I) name for output file
/// * `pixfilter` — (I) Image filter expression
pub fn pixel_filter_helper(
    fptr: &mut Option<Box<fitsfile>>,
    /*      points to the new image */
    outfile: [c_char; FLEN_FILENAME],
    pixfilter: [c_char; FLEN_FILENAME],
    status: &mut c_int,
) -> c_int {
    let mut hdunum: c_int = 0;
    let mut singleHDU: c_int = 0;
    let mut bitpix: c_int = 0;

    /* the new output file, created below; it replaces *fptr on success */
    let mut ofptr: Option<Box<fitsfile>> = None;

    /* create new empty file for result */
    if ffinit_safe(&mut ofptr, &outfile, status) > 0 {
        ffpmsg_str("failed to create output file for pixel filter:");
        ffpmsg_slice(&outfile);
        return *status;
    }

    ffghdn_safe(fptr.as_deref_mut().unwrap(), &mut hdunum); /* current HDU number in input file */

    /* the C advanced a `char *expr`; we walk `pixfilter` with an index instead */
    let mut expr_idx: usize = 3; /* skip 'pix' */
    match pixfilter[expr_idx] as u8 {
        b'b' | b'B' => bitpix = BYTE_IMG,
        b'i' | b'I' => bitpix = SHORT_IMG,
        b'j' | b'J' => bitpix = LONG_IMG,
        b'r' | b'R' => bitpix = FLOAT_IMG,
        b'd' | b'D' => bitpix = DOUBLE_IMG,
        _ => {}
    }
    if bitpix != 0 {
        expr_idx += 1; /* skip bitpix indicator */
    }

    if pixfilter[expr_idx] == bb(b'1') {
        expr_idx += 1;
        singleHDU = 1;
    }

    if fptr.as_deref().unwrap().Fptr.only_one != 0 {
        singleHDU = 1;
    }

    if pixfilter[expr_idx] != bb(b' ') {
        ffpmsg_str("pixel filtering expression not space separated:");
        ffpmsg_slice(&pixfilter[expr_idx..]);
    }
    while pixfilter[expr_idx] == bb(b' ') {
        expr_idx += 1;
    }

    /* copy all preceding extensions to the output file */
    let mut ii: c_int = 1;
    while singleHDU == 0 && ii < hdunum {
        ffmahd_safe(fptr.as_deref_mut().unwrap(), ii, None, status);
        if ffcopy_safe(
            fptr.as_deref_mut().unwrap(),
            ofptr.as_deref_mut().unwrap(),
            0,
            status,
        ) > 0
        {
            if let Some(f) = ofptr.take() {
                ffclos_safe(f, status);
            }
            return *status;
        }
        ii += 1;
    }

    /* move back to the original HDU position */
    ffmahd_safe(fptr.as_deref_mut().unwrap(), hdunum, None, status);

    /* Run the image filter. PixelFilter is the #[repr(C)] interface to
    fits_pixel_filter, so its ifptr/ofptr/expression fields are raw pointers;
    we take pointers into the owned boxes only for this call. The callee only
    reads them (it never reassigns the pointers), so the boxes keep ownership.
    A null `tag` makes fits_pixel_filter use its own default "X" (== the C's
    DEFAULT_TAG). */
    let pf_failed = {
        let mut infptr_raw: *mut fitsfile = fptr.as_deref_mut().unwrap();
        let mut filter = PixelFilter {
            count: 1,
            path: ptr::null_mut(),
            tag: ptr::null_mut(),
            ifptr: &raw mut infptr_raw,
            expression: pixfilter[expr_idx..].as_ptr().cast_mut(),
            bitpix,
            blank: 0,
            ofptr: ofptr.as_deref_mut().unwrap(),
            keyword: [0; FLEN_KEYWORD],
            comment: [0; FLEN_COMMENT],
        };
        fits_pixel_filter_safe(&mut filter, status) != 0
    };
    if pf_failed {
        ffpmsg_str("failed to execute image filter:");
        ffpmsg_slice(&pixfilter[expr_idx..]);
        if let Some(f) = ofptr.take() {
            ffclos_safe(f, status);
        }
        return *status;
    }

    /* copy any remaining HDUs to the output file */
    ii = hdunum + 1;
    /* C: for (ii = hdunum + 1; !singleHDU; ii++) -- singleHDU is loop
    invariant there too; the loop is exited by the ffmahd break below. */
    #[allow(clippy::while_immutable_condition)]
    while singleHDU == 0 {
        if ffmahd_safe(fptr.as_deref_mut().unwrap(), ii, None, status) > 0 {
            break;
        }

        ffcopy_safe(
            fptr.as_deref_mut().unwrap(),
            ofptr.as_deref_mut().unwrap(),
            0,
            status,
        );
        ii += 1;
    }

    if *status == END_OF_FILE {
        *status = 0; /* got the expected EOF error; reset = 0  */
    } else if *status > 0 {
        if let Some(f) = ofptr.take() {
            ffclos_safe(f, status);
        }
        return *status;
    }

    /* close the original file and return ptr to the new image */
    if let Some(f) = fptr.take() {
        ffclos_safe(f, status);
    }

    *fptr = ofptr.take(); /* reset the pointer to the new image */

    /* move back to the image subsection */
    if ii - 1 != hdunum {
        ffmahd_safe(fptr.as_deref_mut().unwrap(), hdunum, None, status);
    }

    *status
}

/// Wrapper function for global initialization of curl library.
/// This is NOT THREAD-SAFE
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffihtps() {
    // FFI WRAPPER
    ffihtps_safe()
}

/// Wrapper function for global initialization of curl library.
/// This is NOT THREAD-SAFE
pub fn ffihtps_safe() {
    todo!();
}

/// Wrapper function for global cleanup of curl library.
/// This is NOT THREAD-SAFE
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffchtps() {
    // FFI WRAPPER
    ffchtps_safe()
}

/// Wrapper function for global cleanup of curl library.
/// This is NOT THREAD-SAFE
pub fn ffchtps_safe() {
    todo!();
}

/// Turn libcurl's verbose output on (1) or off (0).
/// This is NOT THREAD-SAFE
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffvhtps(flag: c_int) {
    // FFI WRAPPER
    ffvhtps_safe(flag)
}

/// Turn libcurl's verbose output on (1) or off (0).
/// This is NOT THREAD-SAFE
pub fn ffvhtps_safe(flag: c_int) {
    if cfg!(feature = "net_services") {
        https_set_verbose(flag);
    }
}

/// Display download status bar (to stderr), where applicable.
/// This is NOT THREAD-SAFE
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffshdwn(flag: c_int) {
    // FFI WRAPPER
    ffshdwn_safe(flag)
}

/// Display download status bar (to stderr), where applicable.
/// This is NOT THREAD-SAFE
pub fn ffshdwn_safe(flag: c_int) {
    if cfg!(feature = "net_services") {
        fits_dwnld_prog_bar(flag);
    }
}

/// Get the current network timeout value in seconds
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtmo() -> c_int {
    // FFI WRAPPER
    ffgtmo_safe()
}

/// Get the current network timeout value in seconds (safe wrapper)
pub fn ffgtmo_safe() -> c_int {
    let mut timeout = 0;

    if cfg!(feature = "net_services") {
        timeout = fits_net_timeout(-1);
    }

    timeout
}

/// Set the network timeout value in seconds
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffstmo(sec: c_int, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        ffstmo_safe(sec, status)
    }
}

/// Set the network timeout value in seconds (safe wrapper)
pub fn ffstmo_safe(sec: c_int, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status; // If status is already set, return immediately
    }

    #[cfg(feature = "net_services")]
    {
        if sec <= 0 {
            *status = BAD_OPTION;
            ffpmsg_str("Bad value for net timeout setting (fits_set_timeout).");
            return *status;
        }

        fits_net_timeout(sec);
    }

    *status
}

/// parse off the next token, delimited by a character in 'delimiter',
/// from the input ptr string;  increment *ptr to the end of the token.
/// Returns the length of the token, not including the delimiter char;
///
/// This routine allocates the *token string;  the calling routine must free it
///
/// # Parameters
///
/// * `isanumber` — (O) is this token a number?
pub fn fits_get_token2_safe(
    ptr: &[c_char],
    ptr_index: &mut usize,
    delimiter: &[c_char],
    token: &mut Option<Vec<c_char>>,
    isanumber: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut tval: [c_char; 73] = [0; 73];

    *ptr_index = 0; // Ensure starts at beginning of ptr

    if *status != 0 {
        return 0;
    }

    while ptr[*ptr_index] == bb(b' ') {
        /* skip over leading blanks */
        *ptr_index += 1;
    }

    let p_tmp = &ptr[*ptr_index..];
    let slen = strcspn_safe(p_tmp, delimiter); /* length of next token */
    if slen != 0 {
        // HEAP ALLOCATION
        let mut t: Vec<c_char> = Vec::new();
        if t.try_reserve_exact(slen + 1).is_err() {
            ffpmsg_str("Couldn't allocate memory to hold token string (fits_get_token2).");
            *status = MEMORY_ALLOCATION;
            return 0;
        } else {
            t.resize(slen + 1, 0);
        }

        let p_tmp = &ptr[*ptr_index..];
        strncat_safe(&mut t, p_tmp, slen); /* copy token */
        *ptr_index += slen; /* skip over the token */

        if let Some(isanumber) = isanumber {
            /* check if token is a number */

            *isanumber = 1;

            let mut loc = 0;
            let r;
            if (strchr_safe(&t, bb(b'D'))).is_some() {
                strncpy_safe(&mut tval, &t, 72);
                tval[72] = 0;

                /*  The C language does not support a 'D'; replace with 'E' */
                let tmp_loc: Option<usize> = strchr_safe(&tval, bb(b'D'));
                if let Some(tmp_loc) = tmp_loc {
                    tval[tmp_loc] = bb(b'E');
                }

                r = &tval[..];
                strtod_safe(r, &mut loc);
            } else {
                r = t.as_slice();
                strtod_safe(r, &mut loc);
            }

            /* check for read error, or junk following the value */
            if r[loc] != 0 && r[loc] != bb(b' ') {
                *isanumber = 0;
            }

            // Not needed since we aren't using libc strtod
            // if errno().0 == ERANGE {
            //     *isanumber = 0;
            // }
        }

        *token = Some(t);
    }

    slen as c_int
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KeywordDatatypeMut;
    use crate::aliases::rust_api::*;
    use crate::cfileio::MAX_PREFIX_LEN;
    use crate::fitsio::{FLEN_FILENAME, URL_PARSE_ERROR};
    use crate::helpers::testhelpers::{from_buf, path_with_ext, to_buf, with_temp_file};

    // Helper function to create and initialize C-style string buffers
    fn create_buffer(size: usize) -> Vec<c_char> {
        vec![0; size]
    }

    // Helper function to convert Rust string to C-style char array
    fn str_to_c_array(s: &str) -> Vec<c_char> {
        let mut vec: Vec<c_char> = s.bytes().map(|b| b as c_char).collect();
        vec.push(0); // Null terminate
        vec
    }

    // Helper function to convert C-style char buffer to Rust string (for verification)
    unsafe fn c_array_to_string(buf: *const c_char) -> String {
        if buf.is_null() {
            return String::new();
        }
        unsafe {
            let c_str = CStr::from_ptr(buf);
            c_str.to_string_lossy().into_owned()
        }
    }

    // ------------------------------------------------------------------
    // fits_get_section_range tests
    // ------------------------------------------------------------------

    fn section_range(spec: &str) -> (c_long, c_long, c_long) {
        let mut status: c_int = 0;
        let buf = str_to_c_array(spec); // NUL-terminated
        let mut ptr_index = 0usize;
        let mut secmin: c_long = 0;
        let mut secmax: c_long = 0;
        let mut incre: c_long = 0;
        fits_get_section_range_safe(
            &buf,
            &mut ptr_index,
            &mut secmin,
            &mut secmax,
            &mut incre,
            &mut status,
        );
        assert_eq!(status, 0);
        (secmin, secmax, incre)
    }

    #[test]
    fn test_section_range_simple() {
        assert_eq!(section_range("1:100"), (1, 100, 1));
    }

    #[test]
    fn test_section_range_increment() {
        assert_eq!(section_range("1:100:2"), (1, 100, 2));
    }

    #[test]
    fn test_section_range_wildcard() {
        assert_eq!(section_range("*"), (1, 0, 1));
    }

    #[test]
    fn test_section_range_inverted() {
        assert_eq!(section_range("-*"), (0, 1, 1));
    }

    #[test]
    fn test_section_range_implied() {
        assert_eq!(section_range(":2"), (1, 0, 2));
    }

    #[test]
    fn test_ffourl_stdout_dash() {
        unsafe {
            let url = str_to_c_array("-");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "stdout://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "");
        }
    }

    #[test]
    fn test_ffourl_stdout_dash_with_space() {
        unsafe {
            let url = str_to_c_array("- ");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut status: c_int = 0;

            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "stdout://");
        }
    }

    #[test]
    fn test_ffourl_stdout_keyword() {
        unsafe {
            let test_cases = vec!["stdout", "STDOUT"];

            for test_case in test_cases {
                let url = str_to_c_array(test_case);
                let mut urltype = create_buffer(MAX_PREFIX_LEN);
                let mut status: c_int = 0;

                let mut outfile = create_buffer(FLEN_FILENAME);
                let mut tpltfile = create_buffer(FLEN_FILENAME);
                let mut compspec = create_buffer(FLEN_FILENAME);

                let result = ffourl(
                    &url,
                    &mut urltype,
                    &mut outfile,
                    &mut tpltfile,
                    &mut compspec,
                    &mut status,
                );

                assert_eq!(result, 0);
                assert_eq!(status, 0);
                assert_eq!(c_array_to_string(urltype.as_ptr()), "stdout://");
            }
        }
    }

    #[test]
    fn test_ffourl_explicit_file_url() {
        unsafe {
            let url = str_to_c_array("file://myfile.fits");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "myfile.fits");
        }
    }

    #[test]
    fn test_ffourl_explicit_http_url() {
        unsafe {
            let url = str_to_c_array("http://example.com/data.fits");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "http://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "example.com/data.fits");
        }
    }

    #[test]
    fn test_ffourl_implicit_file_url() {
        unsafe {
            let url = str_to_c_array("myfile.fits");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "myfile.fits");
        }
    }

    #[test]
    fn test_ffourl_with_template() {
        unsafe {
            let url = str_to_c_array("output.fits(template.fits)");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "output.fits");
            assert_eq!(c_array_to_string(tpltfile.as_ptr()), "template.fits");
        }
    }

    #[test]
    fn test_ffourl_with_compression() {
        unsafe {
            let url = str_to_c_array("output.fits[compress]");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "output.fits");
            assert_eq!(c_array_to_string(compspec.as_ptr()), "compress");
        }
    }

    #[test]
    fn test_ffourl_with_template_and_compression() {
        unsafe {
            let url = str_to_c_array("output.fits(template.fits)[compress R]");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "output.fits");
            assert_eq!(c_array_to_string(tpltfile.as_ptr()), "template.fits");
            assert_eq!(c_array_to_string(compspec.as_ptr()), "compress R");
        }
    }

    #[test]
    fn test_ffourl_gz_file() {
        unsafe {
            let url = str_to_c_array("output.fits.gz");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "compressoutfile://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "output.fits.gz");
        }
    }

    #[test]
    fn test_ffourl_gz_not_at_end() {
        unsafe {
            let url = str_to_c_array("output.gz.fits");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "output.gz.fits");
        }
    }

    #[test]
    fn test_ffourl_leading_spaces() {
        unsafe {
            let url = str_to_c_array("   myfile.fits");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "myfile.fits");
        }
    }

    #[test]
    fn test_ffourl_status_propagation() {
        let url = str_to_c_array("myfile.fits");
        let mut status: c_int = 123; // Pre-existing error

        let mut urltype = create_buffer(1);
        let mut outfile = create_buffer(FLEN_FILENAME);
        let mut tpltfile = create_buffer(FLEN_FILENAME);
        let mut compspec = create_buffer(FLEN_FILENAME);

        let result = ffourl(
            &url,
            &mut urltype,
            &mut outfile,
            &mut tpltfile,
            &mut compspec,
            &mut status,
        );

        assert_eq!(result, 123);
        assert_eq!(status, 123);
    }

    #[test]
    fn test_ffourl_missing_closing_paren() {
        let url = str_to_c_array("output.fits(template.fits");
        let mut urltype = create_buffer(MAX_PREFIX_LEN);
        let mut outfile = create_buffer(FLEN_FILENAME);
        let mut tpltfile = create_buffer(FLEN_FILENAME);
        let mut status: c_int = 0;

        let mut compspec = create_buffer(FLEN_FILENAME);

        let result = ffourl(
            &url,
            &mut urltype,
            &mut outfile,
            &mut tpltfile,
            &mut compspec,
            &mut status,
        );

        assert_eq!(result, URL_PARSE_ERROR);
        assert_eq!(status, URL_PARSE_ERROR);
    }

    #[test]
    fn test_ffourl_missing_closing_bracket() {
        let url = str_to_c_array("output.fits[compress");
        let mut urltype = create_buffer(MAX_PREFIX_LEN);
        let mut outfile = create_buffer(FLEN_FILENAME);
        let mut compspec = create_buffer(FLEN_FILENAME);
        let mut status: c_int = 0;

        let mut tpltfile = create_buffer(FLEN_FILENAME);

        let result = ffourl(
            &url,
            &mut urltype,
            &mut outfile,
            &mut tpltfile,
            &mut compspec,
            &mut status,
        );

        assert_eq!(result, URL_PARSE_ERROR);
        assert_eq!(status, URL_PARSE_ERROR);
    }

    #[test]
    fn test_ffourl_not_stdout_dash_prefix() {
        unsafe {
            // Test that file names starting with - but not exactly "-" are treated as regular files
            let url = str_to_c_array("-55d33m.fits");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "-55d33m.fits");
        }
    }

    #[test]
    fn test_ffourl_empty_url() {
        unsafe {
            let url = str_to_c_array("");
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let mut tpltfile = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);

            let result = ffourl(
                &url,
                &mut urltype,
                &mut outfile,
                &mut tpltfile,
                &mut compspec,
                &mut status,
            );

            assert_eq!(result, 0);
            assert_eq!(status, 0);
            assert_eq!(c_array_to_string(urltype.as_ptr()), "file://");
            assert_eq!(c_array_to_string(outfile.as_ptr()), "");
        }
    }

    // Tests for ffifile2_safer function
    // Helper function to create test buffers and call ffifile2_safer
    fn call_ffifile2_safer(
        input: &str,
    ) -> (
        c_int,
        c_int,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
    ) {
        unsafe {
            let infile = str_to_c_array(input);
            let mut urltype = create_buffer(MAX_PREFIX_LEN);
            let mut infilex = create_buffer(FLEN_FILENAME);
            let mut outfile = create_buffer(FLEN_FILENAME);
            let mut extspec = create_buffer(FLEN_FILENAME);
            let mut rowfilterx = create_buffer(FLEN_FILENAME);
            let mut binspec = create_buffer(FLEN_FILENAME);
            let mut colspec = create_buffer(FLEN_FILENAME);
            let mut pixfilter = create_buffer(FLEN_FILENAME);
            let mut compspec = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let result = ffifile2_safe(
                &infile,
                Some(&mut urltype[..]),
                Some(&mut infilex[..]),
                Some(&mut outfile[..]),
                Some(&mut extspec[..]),
                Some(&mut rowfilterx[..]),
                Some(&mut binspec[..]),
                Some(&mut colspec[..]),
                Some(&mut pixfilter[..]),
                Some(&mut compspec[..]),
                &mut status,
            );

            (
                result,
                status,
                c_array_to_string(urltype.as_ptr()),
                c_array_to_string(infilex.as_ptr()),
                c_array_to_string(outfile.as_ptr()),
                c_array_to_string(extspec.as_ptr()),
                c_array_to_string(rowfilterx.as_ptr()),
                c_array_to_string(binspec.as_ptr()),
                c_array_to_string(colspec.as_ptr()),
                c_array_to_string(pixfilter.as_ptr()),
                c_array_to_string(compspec.as_ptr()),
            )
        }
    }

    #[test]
    fn test_ffifile2_safer_basic_file_url() {
        let (
            result,
            status,
            urltype,
            infilex,
            outfile,
            extspec,
            rowfilterx,
            binspec,
            colspec,
            pixfilter,
            compspec,
        ) = call_ffifile2_safer("file://test.fits");

        // Expected behavior based on C cfitsio output
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "");
        assert_eq!(rowfilterx, "");
        assert_eq!(binspec, "");
        assert_eq!(colspec, "");
        assert_eq!(pixfilter, "");
        assert_eq!(compspec, "");
    }

    #[test]
    fn test_ffifile2_safer_http_url() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("http://example.com/data.fits");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "http://");
        assert_eq!(infilex, "example.com/data.fits");
        assert_eq!(outfile, "");
    }

    #[test]
    fn test_ffifile2_safer_with_output_spec() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("input.fits(output.fits)");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "input.fits");
        assert_eq!(outfile, "output.fits");
    }

    #[test]
    fn test_ffifile2_safer_with_extension_name() {
        let (result, status, urltype, infilex, outfile, extspec, _, _, _, _, _) =
            call_ffifile2_safer("test.fits[EVENTS]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS");
    }

    #[test]
    fn test_ffifile2_safer_with_extension_number() {
        let (result, status, urltype, infilex, outfile, extspec, _, _, _, _, _) =
            call_ffifile2_safer("test.fits[2]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "2");
    }

    #[test]
    fn test_ffifile2_safer_with_image_section() {
        let (result, status, urltype, infilex, outfile, extspec, rowfilterx, _, _, _, _) =
            call_ffifile2_safer("test.fits[1:100,1:50]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(rowfilterx, "1:100,1:50");
    }

    #[test]
    fn test_ffifile2_safer_with_row_filter() {
        let (result, status, urltype, infilex, outfile, extspec, rowfilterx, _, _, _, _) =
            call_ffifile2_safer("test.fits[EVENTS][#row > 100 && #row < 200]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS");
        assert_eq!(rowfilterx, "#row > 100 && #row < 200");
    }

    #[test]
    fn test_ffifile2_safer_row_range_syntax() {
        let (result, status, urltype, infilex, outfile, extspec, rowfilterx, _, _, _, _) =
            call_ffifile2_safer("test.fits[1:100]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(rowfilterx, "1:100");
    }

    #[test]
    fn test_ffifile2_safer_complex_filename() {
        let (result, status, urltype, infilex, outfile, extspec, rowfilterx, _, colspec, _, _) =
            call_ffifile2_safer(
                "http://example.com/data.fits.gz[EVENTS,2](output.fits)[col TIME,ENERGY][#row>10]",
            );
        assert_eq!(result, 125);
        assert_eq!(status, 125);
        assert_eq!(urltype, "http://");
        assert_eq!(infilex, "example.com/data.fits.gz");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS,2");
        assert_eq!(colspec, "col TIME,ENERGY");
    }

    #[test]
    fn test_ffifile2_safer_gz_extension() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("data.fits.gz");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "data.fits.gz");
        assert_eq!(outfile, "");
    }

    #[test]
    fn test_ffifile2_safer_stdout() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("-");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "stdin://");
        assert_eq!(infilex, "");
        assert_eq!(outfile, "");
    }

    #[test]
    fn test_ffifile2_safer_empty_filename() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) = call_ffifile2_safer("");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "");
        assert_eq!(infilex, "");
        assert_eq!(outfile, "");
    }

    #[test]
    fn test_ffifile2_safer_invalid_bracket() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("test.fits[unclosed");
        assert_eq!(result, 125);
        assert_eq!(status, 125);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
    }

    #[test]
    fn test_ffifile2_safer_mem_protocol() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("mem://testfile");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "mem://");
        assert_eq!(infilex, "testfile");
        assert_eq!(outfile, "");
    }

    #[test]
    fn test_ffifile2_safer_column_filtering() {
        let (result, status, urltype, infilex, outfile, extspec, _, _, colspec, _, _) =
            call_ffifile2_safer("test.fits[EVENTS][col TIME,ENERGY,PHA]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS");
        assert_eq!(colspec, "col TIME,ENERGY,PHA");
    }

    // Tests for output file specification (parentheses syntax)
    #[test]
    fn test_ffifile2_safer_with_output_file_memory() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("remote.fits.gz(mem://)");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "remote.fits.gz");
        assert_eq!(outfile, "mem://");
    }

    #[test]
    fn test_ffifile2_safer_with_local_copy() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("data.fits.gz(temp.fits)");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "data.fits.gz");
        assert_eq!(outfile, "temp.fits");
    }

    // Tests for extension specification with version and type
    #[test]
    fn test_ffifile2_safer_extension_with_version() {
        let (result, status, urltype, infilex, outfile, extspec, _, _, _, _, _) =
            call_ffifile2_safer("test.fits[EVENTS,2]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS,2");
    }

    #[test]
    fn test_ffifile2_safer_extension_with_type() {
        let (result, status, urltype, infilex, outfile, extspec, _, _, _, _, _) =
            call_ffifile2_safer("test.fits[EVENTS,2,B]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS,2,B");
    }

    #[test]
    fn test_ffifile2_safer_plus_notation() {
        let (result, status, urltype, infilex, outfile, extspec, _, _, _, _, _) =
            call_ffifile2_safer("test.fits+1");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "test.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "1");
    }

    // Tests for image sectioning with different syntaxes
    #[test]
    fn test_ffifile2_safer_image_section_with_step() {
        let (result, status, urltype, infilex, outfile, _, rowfilterx, _, _, _, _) =
            call_ffifile2_safer("image.fits[1:512:2,2:512:2]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "image.fits");
        assert_eq!(outfile, "");
        assert_eq!(rowfilterx, "1:512:2,2:512:2");
    }

    #[test]
    fn test_ffifile2_safer_image_section_with_wildcard() {
        let (result, status, urltype, infilex, outfile, _, rowfilterx, _, _, _, _) =
            call_ffifile2_safer("image.fits[*,512:256]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "image.fits");
        assert_eq!(outfile, "");
        assert_eq!(rowfilterx, "*,512:256");
    }

    #[test]
    fn test_ffifile2_safer_image_section_flip_axis() {
        let (result, status, urltype, infilex, outfile, _, rowfilterx, _, _, _, _) =
            call_ffifile2_safer("image.fits[*,-*]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "image.fits");
        assert_eq!(outfile, "");
        assert_eq!(rowfilterx, "*,-*");
    }

    // Tests for compression specification
    #[test]
    fn test_ffifile2_safer_compression_default() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, compspec) =
            call_ffifile2_safer("output.fits[compress]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "output.fits");
        assert_eq!(outfile, "");
        assert_eq!(compspec, "compress");
    }

    #[test]
    fn test_ffifile2_safer_compression_with_algorithm() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, compspec) =
            call_ffifile2_safer("output.fits[compress GZIP]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "output.fits");
        assert_eq!(outfile, "");
        assert_eq!(compspec, "compress GZIP");
    }

    #[test]
    fn test_ffifile2_safer_compression_with_tile_size() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, compspec) =
            call_ffifile2_safer("output.fits[compress Rice 100,100]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "output.fits");
        assert_eq!(outfile, "");
        assert_eq!(compspec, "compress Rice 100,100");
    }

    // Tests for binning specification
    #[test]
    fn test_ffifile2_safer_binning_specification() {
        let (result, status, urltype, infilex, outfile, extspec, _, binspec, _, _, _) =
            call_ffifile2_safer("table.fits[EVENTS][bin X=1:1024:2,Y=1:1024:2]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "table.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS");
        assert_eq!(binspec, "bin X=1:1024:2,Y=1:1024:2");
    }

    #[test]
    fn test_ffifile2_safer_simple_binning() {
        let (result, status, urltype, infilex, outfile, extspec, _, binspec, _, _, _) =
            call_ffifile2_safer("table.fits[EVENTS][bin X,Y]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "table.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS");
        assert_eq!(binspec, "bin X,Y");
    }

    // Tests for column manipulation
    #[test]
    fn test_ffifile2_safer_column_deletion() {
        let (result, status, urltype, infilex, outfile, extspec, _, _, colspec, _, _) =
            call_ffifile2_safer("table.fits[EVENTS][col -TIME, Good == STATUS]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "table.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS");
        assert_eq!(colspec, "col -TIME, Good == STATUS");
    }

    #[test]
    fn test_ffifile2_safer_column_wildcards() {
        let (result, status, urltype, infilex, outfile, extspec, _, _, colspec, _, _) =
            call_ffifile2_safer("table.fits[EVENTS][col *TIME*, RATE]");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "file://");
        assert_eq!(infilex, "table.fits");
        assert_eq!(outfile, "");
        assert_eq!(extspec, "EVENTS");
        assert_eq!(colspec, "col *TIME*, RATE");
    }

    // Tests for very complex combinations
    #[test]
    fn test_ffifile2_safer_ultra_complex() {
        let (
            result,
            status,
            urltype,
            infilex,
            outfile,
            extspec,
            rowfilterx,
            binspec,
            colspec,
            pixfilter,
            compspec,
        ) = call_ffifile2_safer(
            "ftp://server.com/data.fits.gz[EVENTS,2,B](output.fits)[col TIME,PHA][#row > 100][bin X,Y=1:1024:16][pix X*2][compress GZIP 64,64]",
        );
        assert_eq!(result, 125);
        assert_eq!(status, 125);
        assert_eq!(urltype, "ftp://");
        assert_eq!(infilex, "server.com/data.fits.gz");
        assert_eq!(extspec, "EVENTS,2,B");
        // Note: cfitsio returns error 125 for output spec after brackets
    }

    // Tests for FTP and HTTPS protocols
    #[test]
    fn test_ffifile2_safer_ftp_protocol() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("ftp://archive.stsci.edu/pub/hlsp/test.fits");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "ftp://");
        assert_eq!(infilex, "archive.stsci.edu/pub/hlsp/test.fits");
        assert_eq!(outfile, "");
    }

    #[test]
    fn test_ffifile2_safer_https_protocol() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("https://fits.gsfc.nasa.gov/samples/test.fits.gz");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "https://");
        assert_eq!(infilex, "fits.gsfc.nasa.gov/samples/test.fits.gz");
        assert_eq!(outfile, "");
    }

    // Tests for shared memory protocol
    #[test]
    fn test_ffifile2_safer_shmem_protocol() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("shmem://shm_12345");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "shmem://");
        assert_eq!(infilex, "shm_12345");
        assert_eq!(outfile, "");
    }

    // Tests for gsiftp protocol
    #[test]
    fn test_ffifile2_safer_gsiftp_protocol() {
        let (result, status, urltype, infilex, outfile, _, _, _, _, _, _) =
            call_ffifile2_safer("gsiftp://gridftp.server.edu/data/test.fits");
        assert_eq!(result, 0);
        assert_eq!(status, 0);
        assert_eq!(urltype, "gsiftp://");
        assert_eq!(infilex, "gridftp.server.edu/data/test.fits");
        assert_eq!(outfile, "");
    }

    #[test]
    fn test_ffrtnm_safe() {
        // (input url, expected rootname = filetype://basename)
        let cases = [
            ("myfile.fits", "myfile.fits"), // plain file: no access prefix
            ("ftp://host/file.fits", "ftp://host/file.fits"),
            ("http://x.com/f.fits", "http://x.com/f.fits"),
            ("file://myfile.fits", "file://myfile.fits"),
            ("file:myfile.fits", "myfile.fits"), // "file:" prefix => implicit/empty type
            ("mem://blah", "mem://blah"),
            ("ftp:host/f.fits", "ftp://host/f.fits"), // the // is optional
            ("myfile.fits[1]", "myfile.fits"),        // strip the [extension] spec
            ("myfile.fits+2", "myfile.fits"),         // strip the +n HDU number
            ("-", "-"),                               // stdin
            ("stdin", "-"),
        ];

        for (url, expected) in cases {
            let url = str_to_c_array(url);
            let mut rootname = create_buffer(FLEN_FILENAME);
            let mut status: c_int = 0;

            let r = ffrtnm_safe(&url, &mut rootname, &mut status);

            assert_eq!(r, 0, "url={url:?}");
            assert_eq!(status, 0, "url={url:?}");
            assert_eq!(
                unsafe { c_array_to_string(rootname.as_ptr()) },
                expected,
                "url={url:?}"
            );
        }

        // a urltype prefix longer than MAX_PREFIX_LEN is a parse error
        let url = str_to_c_array("verylongschemename://file.fits");
        let mut rootname = create_buffer(FLEN_FILENAME);
        let mut status: c_int = 0;
        ffrtnm_safe(&url, &mut rootname, &mut status);
        assert_eq!(status, URL_PARSE_ERROR);
    }

    #[test]
    fn test_ffextn_no_extension() {
        // A plain filename with no extension specifier returns the sentinel -99
        // (CFITSIO would default to the primary array).
        let url = str_to_c_array("myfile.fits");
        let mut extnum: c_int = 0;
        let mut status: c_int = 0;
        ffextn_safe(&url, &mut extnum, &mut status);
        assert_eq!(status, 0);
        assert_eq!(extnum, -99);
    }

    #[test]
    fn test_ffextn_extension_number() {
        // An explicit extension number [n] returns n + 1 (the 1-based HDU number).
        for (spec, expected) in [("myfile.fits[0]", 1), ("myfile.fits[3]", 4)] {
            let url = str_to_c_array(spec);
            let mut extnum: c_int = 0;
            let mut status: c_int = 0;
            ffextn_safe(&url, &mut extnum, &mut status);
            assert_eq!(status, 0, "spec={spec}");
            assert_eq!(extnum, expected, "spec={spec}");
        }
    }

    #[test]
    fn test_ffextn_binning_returns_one() {
        // A binning specification always yields extension 1, since CFITSIO would
        // create a temporary primary array image on the fly.
        let url = str_to_c_array("myfile.fits[3][bin X,Y]");
        let mut extnum: c_int = 0;
        let mut status: c_int = 0;
        ffextn_safe(&url, &mut extnum, &mut status);
        assert_eq!(status, 0);
        assert_eq!(extnum, 1);
    }

    #[test]
    fn test_ffextn_extname_search() {
        // Specifying an extension by name forces the file open to search for it;
        // the returned number is the 1-based HDU position of that extension.
        use crate::helpers::testhelpers::with_temp_file;
        use crate::putkey::{ffcrim_safe, ffphps_safe, ffpkys_safe};

        with_temp_file(|filename| {
            let mut status: c_int = 0;

            /* create a file with a primary array + a named image extension */
            let mut fptr: Option<Box<fitsfile>> = None;
            let name = str_to_c_array(filename);
            ffinit_safe(&mut fptr, &name, &mut status);
            assert_eq!(status, 0, "ffinit failed");
            {
                let f = fptr.as_deref_mut().unwrap();
                ffphps_safe(f, BYTE_IMG, 0, &[], &mut status);
                let naxes: [c_long; 2] = [10, 10];
                ffcrim_safe(f, BYTE_IMG, 2, &naxes, &mut status);
                ffpkys_safe(f, cs!(c"EXTNAME"), cs!(c"EVENTS"), None, &mut status);
                assert_eq!(status, 0, "setup failed");
            }
            ffclos_safe(fptr.take().unwrap(), &mut status);
            assert_eq!(status, 0, "ffclos failed");

            /* build "<path>[EVENTS]" */
            let mut url = str_to_c_array(filename);
            url.pop(); // drop the NUL terminator
            for b in "[EVENTS]".bytes() {
                url.push(b as c_char);
            }
            url.push(0);

            let mut extnum: c_int = 0;
            let mut status2: c_int = 0;
            ffextn_safe(&url, &mut extnum, &mut status2);
            assert_eq!(status2, 0, "ffextn failed");
            assert_eq!(extnum, 2); // EVENTS is the 2nd HDU
        });
    }

    #[test]
    fn test_ffimem_create_write_read() {
        use crate::getkey::ffgkyj_safe;
        use crate::putkey::{ffphps_safe, ffpkyj_safe};

        // ffimem (fits_create_memfile) creates a new FITS file backed by a
        // caller-supplied memory buffer. Mirrors the mem_rawfile_open usage:
        // create the memfile, write to it, then read it back. The buffer is
        // generously sized so no reallocation is needed (realloc = None).
        let mut buffer: Vec<u8> = vec![0u8; 10 * 2880];
        let mut status: c_int = 0;
        let mut buffsize: usize = buffer.len();
        /* buffptr is a void** (the address of the memory pointer) so the driver
        can update it on realloc; realloc is disabled here (None) */
        let mut buf_addr: *mut c_void = buffer.as_mut_ptr().cast();

        let mut fptr: Option<Box<fitsfile>> = None;
        ffimem_safe(
            &mut fptr,
            &raw mut buf_addr,
            &mut buffsize,
            2880,
            None,
            &mut status,
        );
        assert_eq!(status, 0, "ffimem failed");
        assert!(fptr.is_some());

        {
            let f = fptr.as_deref_mut().unwrap();

            // the writable mem-file structure was set up by ffimem
            assert_eq!(f.Fptr.writemode, 1);
            assert_eq!(f.Fptr.validcode, VALIDSTRUC);
            assert_eq!(f.Fptr.filesize, buffsize as LONGLONG);

            // write a primary array header + a user keyword into the memory file
            ffphps_safe(f, BYTE_IMG, 0, &[], &mut status); // SIMPLE/BITPIX=8/NAXIS=0
            ffpkyj_safe(f, cs!(c"ANSWER"), 42, Some(cs!(c"meaning")), &mut status);
            assert_eq!(status, 0, "writing header failed");

            // read the keywords back from the in-memory file
            let mut bitpix: c_long = 0;
            ffgkyj_safe(f, cs!(c"BITPIX"), &mut bitpix, None, &mut status);
            assert_eq!(status, 0);
            assert_eq!(bitpix, 8);

            let mut answer: c_long = 0;
            ffgkyj_safe(f, cs!(c"ANSWER"), &mut answer, None, &mut status);
            assert_eq!(status, 0);
            assert_eq!(answer, 42);
        }

        // memkeep driver leaves our buffer alone; the Vec frees normally on drop
        ffclos_safe(fptr.take().unwrap(), &mut status);
        assert_eq!(status, 0, "ffclos failed");

        // keep the backing buffer alive until after the file is closed
        drop(buffer);
    }

    /* Tests for fits_find_match_delim - delimiter matching.
    Ported from test_find_match_delim.c. The C returns a `char *`; the Rust
    returns the index of the char *after* the matched delimiter, so we deref
    that index and compare the character. */
    #[test]
    fn test_find_match_delim() {
        // C: *fits_find_match_delim(s, d)  ->  s[idx]
        fn delim_char(s: &[c_char], d: c_char) -> c_char {
            let idx = fits_find_match_delim(s, d).unwrap();
            s[idx]
        }

        assert_eq!(delim_char(cs!(c"'Xaa'aaa"), bb(b'\'')), bb(b'X'));
        assert_eq!(delim_char(cs!(c"aaaa'Xaa"), bb(b'\'')), bb(b'X'));
        assert_eq!(delim_char(cs!(c"aaaaaaa'"), bb(b'\'')), 0);

        assert_eq!(delim_char(cs!(c"\"X"), bb(b'"')), bb(b'X'));
        assert_eq!(delim_char(cs!(c"x\"X"), bb(b'"')), bb(b'X'));
        assert_eq!(delim_char(cs!(c"\""), bb(b'"')), 0);

        assert_eq!(delim_char(cs!(c"}X"), bb(b'}')), bb(b'X'));
        assert_eq!(delim_char(cs!(c" {}  }X"), bb(b'}')), bb(b'X'));

        assert_eq!(delim_char(cs!(c"]X"), bb(b']')), bb(b'X'));
        assert_eq!(delim_char(cs!(c" []  ]X"), bb(b']')), bb(b'X'));

        assert_eq!(delim_char(cs!(c")X"), bb(b')')), bb(b'X'));
        assert_eq!(delim_char(cs!(c" ()  )X"), bb(b')')), bb(b'X'));
    }

    /// Make a NUL-terminated `Vec<c_char>` from a `&str`.
    fn cc(s: &str) -> Vec<c_char> {
        let mut v: Vec<c_char> = s.bytes().map(|b| b as c_char).collect();
        v.push(0);
        v
    }

    /// Helper: build the column-name/tform reference vectors for `fits_create_tbl`.
    fn make_table(
        f: &mut fitsfile,
        nrows: LONGLONG,
        names: &[&str],
        forms: &[&str],
        extname: Option<&str>,
        status: &mut c_int,
    ) {
        let ttype: Vec<Option<Vec<c_char>>> = names.iter().map(|s| Some(cc(s))).collect();
        let ttype_ref: Vec<Option<&[c_char]>> = ttype.iter().map(|o| o.as_deref()).collect();
        let tform_v: Vec<Vec<c_char>> = forms.iter().map(|s| cc(s)).collect();
        let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
        let ext = extname.map(cc);
        fits_create_tbl(
            f,
            BINARY_TBL,
            nrows,
            names.len() as c_int,
            &ttype_ref,
            &tform_ref,
            None,
            ext.as_deref(),
            status,
        );
    }

    // ------------------------------------------------------------------
    // fits_get_token tests
    // ------------------------------------------------------------------

    #[test]
    fn test_fits_get_token_basic() {
        let mut input = cc("token1,token2,token3");
        let mut ptr = input.as_mut_ptr();
        let mut token = [0 as c_char; 80];
        let mut isanumber: c_int = 0;

        let slen = fits_get_token_safe(&mut ptr, &cc(","), &mut token, Some(&mut isanumber));
        assert_eq!(slen, 6);
        assert_eq!(from_buf(&token), "token1");
        assert_eq!(isanumber, 0);

        ptr = unsafe { ptr.add(1) }; // skip comma
        let slen = fits_get_token_safe(&mut ptr, &cc(","), &mut token, None);
        assert_eq!(slen, 6);
        assert_eq!(from_buf(&token), "token2");

        ptr = unsafe { ptr.add(1) };
        let slen = fits_get_token_safe(&mut ptr, &cc(","), &mut token, None);
        assert_eq!(slen, 6);
        assert_eq!(from_buf(&token), "token3");
    }

    #[test]
    fn test_fits_get_token_numeric() {
        let mut input = cc("123 456.789 1.5D10");
        let mut ptr = input.as_mut_ptr();
        let mut token = [0 as c_char; 80];
        let mut isanumber: c_int = 0;

        let slen = fits_get_token_safe(&mut ptr, &cc(" "), &mut token, Some(&mut isanumber));
        assert_eq!(slen, 3);
        assert_eq!(from_buf(&token), "123");
        assert_eq!(isanumber, 1);

        ptr = unsafe { ptr.add(1) };
        let slen = fits_get_token_safe(&mut ptr, &cc(" "), &mut token, Some(&mut isanumber));
        assert_eq!(slen, 7);
        assert_eq!(from_buf(&token), "456.789");
        assert_eq!(isanumber, 1);

        ptr = unsafe { ptr.add(1) };
        let slen = fits_get_token_safe(&mut ptr, &cc(" "), &mut token, Some(&mut isanumber));
        assert_eq!(slen, 6);
        assert_eq!(from_buf(&token), "1.5D10");
        assert_eq!(isanumber, 1); // D notation for doubles
    }

    #[test]
    fn test_fits_get_token_blanks() {
        let mut input = cc("   spaced");
        let mut ptr = input.as_mut_ptr();
        let mut token = [0 as c_char; 80];
        let mut isanumber: c_int = 0;

        let slen = fits_get_token_safe(&mut ptr, &cc(","), &mut token, Some(&mut isanumber));
        assert_eq!(slen, 6);
        assert_eq!(from_buf(&token), "spaced");
    }

    #[test]
    fn test_fits_get_token_null_isanumber() {
        let mut input = cc("value");
        let mut ptr = input.as_mut_ptr();
        let mut token = [0 as c_char; 80];

        let slen = fits_get_token_safe(&mut ptr, &cc(","), &mut token, None);
        assert_eq!(slen, 5);
        assert_eq!(from_buf(&token), "value");
    }

    // ------------------------------------------------------------------
    // fits_get_token2 tests
    // ------------------------------------------------------------------

    #[test]
    fn test_fits_get_token2_basic() {
        let mut status: c_int = 0;
        let input = cc("first:second:third");
        let mut idx: usize = 0;
        let mut token: Option<Vec<c_char>> = None;
        let mut isanumber: c_int = 0;

        let slen = fits_get_token2_safe(
            &input,
            &mut idx,
            &cc(":"),
            &mut token,
            Some(&mut isanumber),
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(slen, 5);
        assert_eq!(from_buf(token.as_ref().unwrap()), "first");
        assert_eq!(isanumber, 0);

        // skip colon and advance the slice
        let rest = &input[idx + 1..];
        let mut idx2: usize = 0;
        let slen = fits_get_token2_safe(
            rest,
            &mut idx2,
            &cc(":"),
            &mut token,
            Some(&mut isanumber),
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(slen, 6);
        assert_eq!(from_buf(token.as_ref().unwrap()), "second");

        let rest2 = &rest[idx2 + 1..];
        let mut idx3: usize = 0;
        let slen = fits_get_token2_safe(
            rest2,
            &mut idx3,
            &cc(":"),
            &mut token,
            Some(&mut isanumber),
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(slen, 5);
        assert_eq!(from_buf(token.as_ref().unwrap()), "third");
    }

    #[test]
    fn test_fits_get_token2_numeric() {
        let mut status: c_int = 0;
        let input = cc("42,3.14159,2.5D-3");
        let mut idx: usize = 0;
        let mut token: Option<Vec<c_char>> = None;
        let mut isanumber: c_int = 0;

        let slen = fits_get_token2_safe(
            &input,
            &mut idx,
            &cc(","),
            &mut token,
            Some(&mut isanumber),
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(slen, 2);
        assert_eq!(from_buf(token.as_ref().unwrap()), "42");
        assert_eq!(isanumber, 1);

        let rest = &input[idx + 1..];
        let mut idx2: usize = 0;
        let slen = fits_get_token2_safe(
            rest,
            &mut idx2,
            &cc(","),
            &mut token,
            Some(&mut isanumber),
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(slen, 7);
        assert_eq!(from_buf(token.as_ref().unwrap()), "3.14159");
        assert_eq!(isanumber, 1);

        let rest2 = &rest[idx2 + 1..];
        let mut idx3: usize = 0;
        let slen = fits_get_token2_safe(
            rest2,
            &mut idx3,
            &cc(","),
            &mut token,
            Some(&mut isanumber),
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(slen, 6);
        assert_eq!(from_buf(token.as_ref().unwrap()), "2.5D-3");
        assert_eq!(isanumber, 1); // D notation
    }

    #[test]
    fn test_fits_get_token2_null_isanumber() {
        let mut status: c_int = 0;
        let input = cc("somevalue");
        let mut idx: usize = 0;
        let mut token: Option<Vec<c_char>> = None;

        let slen = fits_get_token2_safe(&input, &mut idx, &cc(","), &mut token, None, &mut status);
        assert_eq!(status, 0);
        assert_eq!(slen, 9);
        assert_eq!(from_buf(token.as_ref().unwrap()), "somevalue");
    }

    #[test]
    fn test_fits_get_token2_error_status() {
        let mut status: c_int = 1; // pre-existing error
        let input = cc("value");
        let mut idx: usize = 0;
        let mut token: Option<Vec<c_char>> = None;
        let mut isanumber: c_int = 0;

        let slen = fits_get_token2_safe(
            &input,
            &mut idx,
            &cc(","),
            &mut token,
            Some(&mut isanumber),
            &mut status,
        );
        assert_eq!(slen, 0); // should return 0 on error
    }

    // ------------------------------------------------------------------
    // ffexist (fits_file_exists) tests
    // ------------------------------------------------------------------

    /// Mirrors test_ffexist_exists in ~/code/cfitsio/tests/test_cfileio.c
    #[test]
    fn test_ffexist_exists() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            /* Create a test file */
            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup failed");

            /* Test that it exists */
            let mut exists: c_int = -99;
            fits_file_exists(&name, &mut exists, &mut status);
            assert_eq!(status, 0);
            assert_eq!(exists, 1);
        });
    }

    /// Mirrors test_ffexist_not_exists in ~/code/cfitsio/tests/test_cfileio.c
    #[test]
    fn test_ffexist_not_exists() {
        let mut status: c_int = 0;
        let mut exists: c_int = -99;

        fits_file_exists(
            &str_to_c_array("this_file_does_not_exist_12345.fits"),
            &mut exists,
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(exists, 0);
    }

    /// Mirrors test_ffexist_non_disk in ~/code/cfitsio/tests/test_cfileio.c
    #[test]
    fn test_ffexist_non_disk() {
        let mut status: c_int = 0;
        let mut exists: c_int = -99;

        fits_file_exists(&str_to_c_array("mem://test"), &mut exists, &mut status);
        assert_eq!(status, 0);
        assert_eq!(exists, -1);
    }

    /// Mirrors test_ffexist_stdin in ~/code/cfitsio/tests/test_cfileio.c
    #[test]
    fn test_ffexist_stdin() {
        let mut status: c_int = 0;
        let mut exists: c_int = -99;

        fits_file_exists(&str_to_c_array("-"), &mut exists, &mut status);
        assert_eq!(status, 0);
        assert_eq!(exists, -1);
    }

    // ------------------------------------------------------------------
    // ffrtnm (fits_parse_rootname) tests
    // ------------------------------------------------------------------

    fn rtnm(url: &str) -> String {
        let mut status: c_int = 0;
        let mut rootname = [0 as c_char; FLEN_FILENAME];
        ffrtnm_safe(&cc(url), &mut rootname, &mut status);
        assert_eq!(status, 0);
        from_buf(&rootname).to_string()
    }

    #[test]
    fn test_ffrtnm_simple() {
        assert_eq!(rtnm("myfile.fits"), "myfile.fits");
    }

    #[test]
    fn test_ffrtnm_extension() {
        assert_eq!(rtnm("myfile.fits[1]"), "myfile.fits");
    }

    #[test]
    fn test_ffrtnm_file_prefix() {
        // file:// is kept in the rootname - unlike file: which is stripped
        assert_eq!(rtnm("file://myfile.fits"), "file://myfile.fits");
    }

    #[test]
    fn test_ffrtnm_rowfilter() {
        assert_eq!(rtnm("myfile.fits[EVENTS][X>5]"), "myfile.fits");
    }

    #[test]
    fn test_ffrtnm_stdin() {
        assert_eq!(rtnm("-"), "-");
        assert_eq!(rtnm("stdin"), "-");
    }

    #[test]
    fn test_ffrtnm_http() {
        assert_eq!(
            rtnm("http://example.com/file.fits"),
            "http://example.com/file.fits"
        );
    }

    #[test]
    fn test_ffrtnm_ftp() {
        assert_eq!(
            rtnm("ftp://example.com/file.fits"),
            "ftp://example.com/file.fits"
        );
    }

    #[test]
    fn test_ffrtnm_fits() {
        assert!(rtnm("fitsfile://path/to/file.fits").contains("file.fits"));
    }

    #[test]
    fn test_ffrtnm_compressed() {
        assert_eq!(rtnm("file.fits.gz"), "file.fits.gz");
    }

    // ------------------------------------------------------------------
    // ffexts (fits_parse_extspec) tests
    // ------------------------------------------------------------------

    struct Exts {
        extnum: c_int,
        extname: String,
        extvers: c_int,
        hdutype: c_int,
        imagecolname: String,
        rowexpress: String,
    }

    fn exts(spec: &str) -> Exts {
        let mut status: c_int = 0;
        let mut extnum: c_int = 0;
        let mut extname = [0 as c_char; FLEN_VALUE];
        let mut extvers: c_int = 0;
        let mut hdutype: c_int = 0;
        let mut imagecolname = [0 as c_char; FLEN_VALUE];
        let mut rowexpress = [0 as c_char; FLEN_FILENAME];
        ffexts_safe(
            &cc(spec),
            &mut extnum,
            &mut extname,
            &mut extvers,
            &mut hdutype,
            &mut imagecolname,
            &mut rowexpress,
            &mut status,
        );
        assert_eq!(status, 0);
        Exts {
            extnum,
            extname: from_buf(&extname).to_string(),
            extvers,
            hdutype,
            imagecolname: from_buf(&imagecolname).to_string(),
            rowexpress: from_buf(&rowexpress).to_string(),
        }
    }

    #[test]
    fn test_ffexts_number() {
        let e = exts("3");
        assert_eq!(e.extnum, 3);
        assert_eq!(e.extname, "");
        assert_eq!(e.extvers, 0);
        assert_eq!(e.hdutype, ANY_HDU);
    }

    #[test]
    fn test_ffexts_name() {
        let e = exts("EVENTS");
        assert_eq!(e.extnum, 0);
        assert_eq!(e.extname, "EVENTS");
        assert_eq!(e.extvers, 0);
        assert_eq!(e.hdutype, ANY_HDU);
    }

    #[test]
    fn test_ffexts_name_version() {
        let e = exts("EVENTS,2");
        assert_eq!(e.extnum, 0);
        assert_eq!(e.extname, "EVENTS");
        assert_eq!(e.extvers, 2);
        assert_eq!(e.hdutype, ANY_HDU);
    }

    #[test]
    fn test_ffexts_name_version_type() {
        let e = exts("EVENTS,1,b");
        assert_eq!(e.extname, "EVENTS");
        assert_eq!(e.extvers, 1);
        assert_eq!(e.hdutype, BINARY_TBL);

        let e = exts("DATA,1,a");
        assert_eq!(e.hdutype, ASCII_TBL);

        let e = exts("IMG,1,i");
        assert_eq!(e.hdutype, IMAGE_HDU);
    }

    #[test]
    fn test_ffexts_image_column() {
        let e = exts("PICS;IMAGE(3)");
        assert_eq!(e.extname, "PICS");
        assert_eq!(e.imagecolname, "IMAGE");
        assert_eq!(e.rowexpress, "3");
    }

    #[test]
    fn test_ffexts_row_expression() {
        let e = exts("PICS;colname(exposure>1000)");
        assert_eq!(e.extname, "PICS");
        assert_eq!(e.imagecolname, "colname");
        assert_eq!(e.rowexpress, "exposure>1000");
    }

    #[test]
    fn test_ffexts_negative_name() {
        // "-5" is treated as an extension NAME, not a number
        let e = exts("-5");
        assert_eq!(e.extnum, 0);
        assert_eq!(e.extname, "-5");
    }

    #[test]
    fn test_ffexts_xtension_types() {
        let e = exts("EVENTS,1,t");
        assert_eq!(e.extname, "EVENTS");
        assert_eq!(e.extvers, 1);
        assert_eq!(e.hdutype, ASCII_TBL);
    }

    #[test]
    fn test_ffexts_table_type() {
        let e = exts("DATA,0,t");
        assert_eq!(e.hdutype, ASCII_TBL);
        let e = exts("DATA,0,T");
        assert_eq!(e.hdutype, ASCII_TBL);
    }

    // ------------------------------------------------------------------
    // ffextn (fits_parse_extnum) tests
    // ------------------------------------------------------------------

    /* The three ffextn cases the C's test_cfileio.c covers -- test_ffextn_number
    (an explicit extension number), test_ffextn_none (no extension specifier) and
    test_ffextn_binspec (a binning specification) -- are already covered, with an
    extra case each, by test_ffextn_extension_number, test_ffextn_no_extension and
    test_ffextn_binning_returns_one above. */

    // ------------------------------------------------------------------
    // Open function variants
    // ------------------------------------------------------------------

    #[test]
    fn test_ffdopn() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status); // null primary
            make_table(
                f.as_deref_mut().unwrap(),
                0,
                &["COL1"],
                &["1J"],
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            // Open with ffdopn - should skip null primary
            ffdopn_safe(&mut f, &name, READONLY, &mut status);
            assert_eq!(status, 0, "ffdopn failed");
            let mut hdunum: c_int = 0;
            fits_get_hdu_num(f.as_deref_mut().unwrap(), &mut hdunum);
            assert_eq!(hdunum, 2); // should be at extension 2
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_fftopn() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                0,
                &["COL1"],
                &["1J"],
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fftopn_safe(&mut f, &name, READONLY, &mut status);
            assert_eq!(status, 0, "fftopn failed");
            let mut hdutype: c_int = 0;
            fits_get_hdu_type(f.as_deref_mut().unwrap(), &mut hdutype, &mut status);
            assert_eq!(status, 0);
            assert_eq!(hdutype, BINARY_TBL);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffiopn() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            ffiopn_safe(&mut f, &name, READONLY, &mut status);
            assert_eq!(status, 0, "ffiopn failed");
            let mut hdutype: c_int = 0;
            fits_get_hdu_type(f.as_deref_mut().unwrap(), &mut hdutype, &mut status);
            assert_eq!(status, 0);
            assert_eq!(hdutype, IMAGE_HDU);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    /// Regression test: a failed open must report a nonzero status rather than
    /// panicking. `ffopen_safe` leaves `*fptr` as `None` when it fails, so
    /// `fftopn_safe`/`ffiopn_safe` must not go on to unwrap it.
    #[test]
    fn test_ffiopn_fftopn_failed_open_returns_status() {
        with_temp_file(|filename| {
            // `with_temp_file` hands us a path inside a fresh temp directory
            // that has not been created yet, so this name does not exist.
            let missing = to_buf(filename);

            // A file that exists but holds no FITS header at all.
            let dir = std::path::Path::new(filename).parent().unwrap();
            let junk_path = dir.join("not-a-fits-file.fits");
            std::fs::write(&junk_path, b"this is definitely not a FITS file\n").unwrap();
            let junk = to_buf(junk_path.to_str().unwrap());

            for name in [&missing, &junk] {
                let mut status: c_int = 0;
                let mut f: Option<Box<fitsfile>> = None;
                let ret = ffiopn_safe(&mut f, name, READONLY, &mut status);
                assert_ne!(status, 0, "ffiopn_safe should fail on {}", from_buf(name));
                assert_eq!(ret, status);
                assert!(f.is_none());

                let mut status: c_int = 0;
                let mut f: Option<Box<fitsfile>> = None;
                let ret = fftopn_safe(&mut f, name, READONLY, &mut status);
                assert_ne!(status, 0, "fftopn_safe should fail on {}", from_buf(name));
                assert_eq!(ret, status);
                assert!(f.is_none());

                // `ffdopn_safe` already behaved this way; keep it honest too.
                let mut status: c_int = 0;
                let mut f: Option<Box<fitsfile>> = None;
                ffdopn_safe(&mut f, name, READONLY, &mut status);
                assert_ne!(status, 0, "ffdopn_safe should fail on {}", from_buf(name));
                assert!(f.is_none());
            }
        });
    }

    #[test]
    fn test_ffreopen() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f1: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f1, &name, &mut status);
            fits_write_imghdr(f1.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            assert_eq!(status, 0, "setup");

            // Reopen the file - ffreopen outputs a raw *mut fitsfile
            let mut newptr: *mut fitsfile = core::ptr::null_mut();
            ffreopen_safe(f1.as_deref_mut().unwrap(), &mut newptr, &mut status);
            assert_eq!(status, 0, "ffreopen failed");
            assert!(!newptr.is_null());
            let mut f2: Option<Box<fitsfile>> = Some(unsafe { Box::from_raw(newptr) });

            let mut hdunum: c_int = 0;
            fits_get_hdu_num(f1.as_deref_mut().unwrap(), &mut hdunum);
            assert_eq!(hdunum, 1);
            fits_get_hdu_num(f2.as_deref_mut().unwrap(), &mut hdunum);
            assert_eq!(hdunum, 1);

            fits_close_file(f1.take().unwrap(), &mut status);
            fits_close_file(f2.take().unwrap(), &mut status);
        });
    }

    // NOTE: fits_get_section_range tests were migrated into src/cfileio.rs's
    // `mod tests` (they exercise the now-private fits_get_section_range_safe).

    // ------------------------------------------------------------------
    // fits_copy_image_section tests
    // ------------------------------------------------------------------

    /// Create a file holding a single image of the given type and shape, whose
    /// pixels are the 1-based linear index (1, 2, 3, ... in FITS order).
    fn make_img(name: &[c_char], bitpix: c_int, naxes: &[c_long], status: &mut c_int) {
        let mut f: Option<Box<fitsfile>> = None;
        fits_create_file(&mut f, name, status);
        fits_create_img(
            f.as_deref_mut().unwrap(),
            bitpix,
            naxes.len() as c_int,
            naxes,
            status,
        );
        let n: c_long = naxes.iter().product();
        let data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        fits_write_img_dbl(
            f.as_deref_mut().unwrap(),
            1,
            1,
            n as LONGLONG,
            &data,
            status,
        );
        fits_close_file(f.take().unwrap(), status);
        assert_eq!(*status, 0, "make_img setup failed");
    }

    /// Copy `expr` out of the image in `inname` into a new file `outname`,
    /// returning the (still open) output file.
    fn copy_section(
        inname: &[c_char],
        outname: &[c_char],
        expr: &str,
        status: &mut c_int,
    ) -> Option<Box<fitsfile>> {
        let mut fin: Option<Box<fitsfile>> = None;
        let mut fout: Option<Box<fitsfile>> = None;
        fits_open_file(&mut fin, inname, READONLY, status);
        fits_create_file(&mut fout, outname, status);
        assert_eq!(*status, 0, "copy_section setup failed");

        let e = str_to_c_array(expr);
        fits_copy_image_section_safe(
            fin.as_deref_mut().unwrap(),
            fout.as_deref_mut().unwrap(),
            &e,
            status,
        );
        fits_close_file(fin.take().unwrap(), status);
        fout
    }

    /// The NAXISn values of the image in `f`.
    fn img_size(f: &mut fitsfile, naxis: usize, status: &mut c_int) -> Vec<c_long> {
        let mut naxes = vec![0 as c_long; naxis];
        fits_get_img_size(f, naxis as c_int, &mut naxes, status);
        naxes
    }

    #[test]
    fn test_copy_image_section() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, BYTE_IMG, &[8, 6], &mut status);

                let mut fout = copy_section(&iname, &oname, "2:5,3:4", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 2, &mut status), vec![4, 2]);

                let mut got = [0u8; 8];
                let mut anynul: c_int = 0;
                fits_read_img_byt(f, 1, 1, 8, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [18, 19, 20, 21, 26, 27, 28, 29]);

                fits_close_file(fout.take().unwrap(), &mut status);
                assert_eq!(status, 0);
            });
        });
    }

    #[test]
    fn test_copy_image_section_stride() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONG_IMG, &[8, 6], &mut status);

                let mut fout = copy_section(&iname, &oname, "1:8:2,1:6:3", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 2, &mut status), vec![4, 2]);

                let mut got = [0 as c_int; 8];
                let mut anynul: c_int = 0;
                fits_read_img_int(f, 1, 1, 8, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [1, 3, 5, 7, 25, 27, 29, 31]);

                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_short() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, SHORT_IMG, &[6, 4], &mut status);

                let mut fout = copy_section(&iname, &oname, "2:4,2:3", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 2, &mut status), vec![3, 2]);

                let mut got = [0 as c_short; 6];
                let mut anynul: c_int = 0;
                fits_read_img_sht(f, 1, 1, 6, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [8, 9, 10, 14, 15, 16]);

                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_int() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONG_IMG, &[6, 4], &mut status);

                let mut fout = copy_section(&iname, &oname, "2:4,2:3", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 2, &mut status), vec![3, 2]);

                let mut got = [0 as c_int; 6];
                let mut anynul: c_int = 0;
                fits_read_img_int(f, 1, 1, 6, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [8, 9, 10, 14, 15, 16]);

                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_float() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, FLOAT_IMG, &[6, 4], &mut status);

                let mut fout = copy_section(&iname, &oname, "2:4,2:3", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 2, &mut status), vec![3, 2]);

                let mut got = [0.0f32; 6];
                let mut anynul: c_int = 0;
                fits_read_img_flt(f, 1, 1, 6, 0.0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [8.0, 9.0, 10.0, 14.0, 15.0, 16.0]);

                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_double() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, DOUBLE_IMG, &[6, 4], &mut status);

                let mut fout = copy_section(&iname, &oname, "2:4,2:3", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 2, &mut status), vec![3, 2]);

                let mut got = [0.0f64; 6];
                let mut anynul: c_int = 0;
                fits_read_img_dbl(f, 1, 1, 6, 0.0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [8.0, 9.0, 10.0, 14.0, 15.0, 16.0]);

                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_longlong() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONGLONG_IMG, &[6, 4], &mut status);

                let mut fout = copy_section(&iname, &oname, "2:4,2:3", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 2, &mut status), vec![3, 2]);

                let mut got = [0 as LONGLONG; 6];
                let mut anynul: c_int = 0;
                fits_read_img_lnglng(f, 1, 1, 6, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [8, 9, 10, 14, 15, 16]);

                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_3d() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONG_IMG, &[4, 3, 2], &mut status);

                /* take the second plane only */
                let mut fout = copy_section(&iname, &oname, "1:4,1:3,2:2", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 3, &mut status), vec![4, 3, 1]);

                let mut got = [0 as c_int; 12];
                let mut anynul: c_int = 0;
                fits_read_img_int(f, 1, 1, 12, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);

                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_whole_axis() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONG_IMG, &[4, 3], &mut status);

                /* '*' means the whole axis */
                let mut fout = copy_section(&iname, &oname, "*,2:2", &mut status);
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();
                assert_eq!(img_size(f, 2, &mut status), vec![4, 1]);

                let mut got = [0 as c_int; 4];
                let mut anynul: c_int = 0;
                fits_read_img_int(f, 1, 1, 4, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [5, 6, 7, 8]);

                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_too_many_axes() {
        /* NAXIS >= 10 overflows the C's 9-element `naxes` array before it gets
        to the naxis > 4 check; here it must simply report BAD_NAXIS */
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(
                    &iname,
                    BYTE_IMG,
                    &[2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    &mut status,
                );

                let mut fout = copy_section(&iname, &oname, "1:2", &mut status);
                assert_eq!(status, BAD_NAXIS);

                status = 0;
                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    #[test]
    fn test_copy_image_section_exceeds_input() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONG_IMG, &[4, 3], &mut status);

                let mut fout = copy_section(&iname, &oname, "1:9,1:3", &mut status);
                assert_eq!(status, BAD_NAXIS);

                status = 0;
                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    // ------------------------------------------------------------------
    // fits_select_image_section tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_image_section() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONG_IMG, &[8, 6], &mut status);

                let mut f: Option<Box<fitsfile>> = None;
                fits_open_file(&mut f, &iname, READONLY, &mut status);
                assert_eq!(status, 0, "setup");

                let expr = str_to_c_array("2:5,3:4");
                fits_select_image_section_safe(&mut f, &to_buf(outname), &expr, &mut status);
                assert_eq!(status, 0);

                /* on success *fptr points at the new subimage, not the input */
                let g = f.as_deref_mut().unwrap();
                assert_eq!(img_size(g, 2, &mut status), vec![4, 2]);

                let mut got = [0 as c_int; 8];
                let mut anynul: c_int = 0;
                fits_read_img_int(g, 1, 1, 8, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got, [18, 19, 20, 21, 26, 27, 28, 29]);

                fits_close_file(f.take().unwrap(), &mut status);
                assert_eq!(status, 0);
                let _ = oname;
            });
        });
    }

    // ------------------------------------------------------------------
    // fits_copy_image2cell tests
    // ------------------------------------------------------------------

    #[test]
    fn test_copy_image2cell_new_column() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONG_IMG, &[4, 3], &mut status);

                /* an output file holding a one-row table with one dummy column */
                let mut fout: Option<Box<fitsfile>> = None;
                fits_create_file(&mut fout, &oname, &mut status);
                let ttype = [Some(str_to_c_array("DUMMY"))];
                let ttype_ref: Vec<Option<&[c_char]>> =
                    ttype.iter().map(|o| o.as_deref()).collect();
                let tform = [str_to_c_array("1J")];
                let tform_ref: Vec<&[c_char]> = tform.iter().map(|v| v.as_slice()).collect();
                fits_create_tbl(
                    fout.as_deref_mut().unwrap(),
                    BINARY_TBL,
                    1,
                    1,
                    &ttype_ref,
                    &tform_ref,
                    None,
                    None,
                    &mut status,
                );
                assert_eq!(status, 0, "table setup");

                let mut fin: Option<Box<fitsfile>> = None;
                fits_open_file(&mut fin, &iname, READONLY, &mut status);
                assert_eq!(status, 0, "setup");

                let colname = str_to_c_array("IMGCOL");
                fits_copy_image2cell_safe(
                    fin.as_deref_mut().unwrap(),
                    fout.as_deref_mut().unwrap(),
                    &colname,
                    1,
                    1,
                    &mut status,
                );
                assert_eq!(status, 0);

                let f = fout.as_deref_mut().unwrap();

                /* the image landed in a newly created second column ... */
                let mut colnum: c_int = 0;
                fits_get_colnum(f, CASEINSEN as c_int, &colname, &mut colnum, &mut status);
                assert_eq!(status, 0);
                assert_eq!(colnum, 2);

                /* ... with the image's dimensions recorded in TDIM ... */
                let mut naxis: c_int = 0;
                let mut naxes = [0 as LONGLONG; 9];
                fits_read_tdimll(f, colnum, 9, &mut naxis, &mut naxes, &mut status);
                assert_eq!(status, 0);
                assert_eq!(naxis, 2);
                assert_eq!(&naxes[..2], &[4, 3]);

                /* ... and the pixels intact */
                let mut got = [0 as c_int; 12];
                let mut anynul: c_int = 0;
                fits_read_col_int(
                    f,
                    colnum,
                    1,
                    1,
                    12,
                    0,
                    &mut got,
                    Some(&mut anynul),
                    &mut status,
                );
                assert_eq!(status, 0);
                assert_eq!(got, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

                fits_close_file(fin.take().unwrap(), &mut status);
                fits_close_file(fout.take().unwrap(), &mut status);
                assert_eq!(status, 0);
            });
        });
    }

    #[test]
    fn test_copy_image2cell_rejects_non_table_output() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);
                make_img(&iname, LONG_IMG, &[4, 3], &mut status);
                make_img(&oname, LONG_IMG, &[2, 2], &mut status);

                let mut fin: Option<Box<fitsfile>> = None;
                let mut fout: Option<Box<fitsfile>> = None;
                fits_open_file(&mut fin, &iname, READONLY, &mut status);
                fits_open_file(&mut fout, &oname, READWRITE, &mut status);
                assert_eq!(status, 0, "setup");

                let colname = str_to_c_array("IMGCOL");
                fits_copy_image2cell_safe(
                    fin.as_deref_mut().unwrap(),
                    fout.as_deref_mut().unwrap(),
                    &colname,
                    1,
                    1,
                    &mut status,
                );
                assert_eq!(status, NOT_BTABLE);

                status = 0;
                fits_close_file(fin.take().unwrap(), &mut status);
                fits_close_file(fout.take().unwrap(), &mut status);
            });
        });
    }

    // ------------------------------------------------------------------
    // ffselect_table tests
    // ------------------------------------------------------------------

    /// Create a file with a one-column table holding `data` in the 2nd HDU.
    fn make_value_table(name: &[c_char], data: &[c_long], status: &mut c_int) {
        let mut f: Option<Box<fitsfile>> = None;
        fits_create_file(&mut f, name, status);
        fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], status);
        let ttype = [Some(str_to_c_array("VALUE"))];
        let ttype_ref: Vec<Option<&[c_char]>> = ttype.iter().map(|o| o.as_deref()).collect();
        let tform = [str_to_c_array("1J")];
        let tform_ref: Vec<&[c_char]> = tform.iter().map(|v| v.as_slice()).collect();
        fits_create_tbl(
            f.as_deref_mut().unwrap(),
            BINARY_TBL,
            data.len() as LONGLONG,
            1,
            &ttype_ref,
            &tform_ref,
            None,
            None,
            status,
        );
        fits_write_col_lng(
            f.as_deref_mut().unwrap(),
            1,
            1,
            1,
            data.len() as LONGLONG,
            data,
            status,
        );
        fits_close_file(f.take().unwrap(), status);
        assert_eq!(*status, 0, "make_value_table setup failed");
    }

    fn read_values(f: &mut fitsfile, n: usize, status: &mut c_int) -> Vec<c_long> {
        let mut got = vec![0 as c_long; n];
        let mut anynul: c_int = 0;
        fits_read_col_lng(
            f,
            1,
            1,
            1,
            n as LONGLONG,
            0,
            &mut got,
            Some(&mut anynul),
            status,
        );
        got
    }

    #[test]
    fn test_ffselect_table_to_new_file() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                make_value_table(&iname, &[10, 25, 30, 45, 50], &mut status);

                let mut f: Option<Box<fitsfile>> = None;
                fits_open_file(&mut f, &iname, READONLY, &mut status);
                fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
                assert_eq!(status, 0, "setup");

                let mut expr = [0 as c_char; FLEN_FILENAME];
                for (i, &b) in b"VALUE > 20".iter().enumerate() {
                    expr[i] = b as c_char;
                }

                ffselect_table(&mut f, &to_buf(outname), &expr, &mut status);
                assert_eq!(status, 0);

                /* *fptr now points at the selected-rows copy */
                let g = f.as_deref_mut().unwrap();
                let mut nrows: LONGLONG = 0;
                fits_get_num_rowsll(g, &mut nrows, &mut status);
                assert_eq!(nrows, 4);
                assert_eq!(read_values(g, 4, &mut status), vec![25, 30, 45, 50]);
                assert_eq!(status, 0);

                fits_close_file(f.take().unwrap(), &mut status);
                assert_eq!(status, 0);
            });
        });
    }

    #[test]
    fn test_ffselect_table_in_place() {
        with_temp_file(|inname| {
            let mut status: c_int = 0;
            let iname = to_buf(inname);
            make_value_table(&iname, &[10, 25, 30, 45, 50], &mut status);

            let mut f: Option<Box<fitsfile>> = None;
            fits_open_file(&mut f, &iname, READWRITE, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            assert_eq!(status, 0, "setup");

            /* an empty outfile name means "delete the non-qualifying rows in
            place", the branch where the C aliases infptr and outfptr */
            let outfile = [0 as c_char; FLEN_FILENAME];
            let mut expr = [0 as c_char; FLEN_FILENAME];
            for (i, &b) in b"VALUE > 20".iter().enumerate() {
                expr[i] = b as c_char;
            }

            ffselect_table(&mut f, &outfile, &expr, &mut status);
            assert_eq!(status, 0);

            let g = f.as_deref_mut().unwrap();
            let mut nrows: LONGLONG = 0;
            fits_get_num_rowsll(g, &mut nrows, &mut status);
            assert_eq!(nrows, 4);
            assert_eq!(read_values(g, 4, &mut status), vec![25, 30, 45, 50]);
            assert_eq!(status, 0);

            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0);
        });
    }

    // ------------------------------------------------------------------
    // fits_split_names tests
    // ------------------------------------------------------------------

    /// Drive fits_split_names to exhaustion over a writable copy of `list`.
    fn split_names(list: &str) -> Vec<String> {
        let mut buf = str_to_c_array(list);
        let mut out: Vec<String> = Vec::new();
        unsafe {
            let mut p = fits_split_names_safer(buf.as_mut_ptr());
            while !p.is_null() {
                out.push(c_array_to_string(p));
                p = fits_split_names_safer(core::ptr::null_mut());
            }
        }
        out
    }

    #[test]
    fn test_split_names_doc_example() {
        assert_eq!(
            split_names("myfile[1][bin (x,y)=4], file2.fits  file3.fits"),
            vec!["myfile[1][bin (x,y)=4]", "file2.fits", "file3.fits"]
        );
    }

    #[test]
    fn test_split_names_single() {
        assert_eq!(split_names("onefile.fits"), vec!["onefile.fits"]);
    }

    #[test]
    fn test_split_names_leading_and_repeated_delimiters() {
        /* Leading blanks are skipped, but -- unlike strtok, which the doc
        comment compares this to -- consecutive delimiters are NOT merged: each
        one yields an empty token.  Verified against the C. */
        assert_eq!(
            split_names("   a.fits ,, b.fits"),
            vec!["a.fits", "", "", "b.fits"]
        );
    }

    #[test]
    fn test_split_names_empty() {
        assert!(split_names("").is_empty());
        assert!(split_names("     ").is_empty());
    }

    #[test]
    fn test_split_names_null_first_call() {
        /* DEVIATION: the C dereferences its uninitialised static and crashes
        if the first call passes NULL; this port reports "no names".  Run it on
        a fresh thread so the thread-local cursor is guaranteed unseeded even
        under `--test-threads=1`. */
        let got = std::thread::spawn(|| unsafe {
            fits_split_names_safer(core::ptr::null_mut()).is_null()
        })
        .join()
        .unwrap();
        assert!(got);
    }

    #[test]
    fn test_split_names_nested_brackets() {
        /* commas and spaces inside [], () and {} do not split */
        assert_eq!(
            split_names("f1.fits[col a, b]{x, y},f2.fits"),
            vec!["f1.fits[col a, b]{x, y}", "f2.fits"]
        );
    }

    // ------------------------------------------------------------------
    // URL type tests
    // ------------------------------------------------------------------

    #[test]
    fn test_ffurlt() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);

            let mut urltype = [0 as c_char; 80];
            fits_url_type(f.as_deref_mut().unwrap(), &mut urltype, &mut status);
            assert_eq!(status, 0);
            assert_eq!(from_buf(&urltype), "file://");

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ------------------------------------------------------------------
    // ffiurl (fits_parse_input_url) tests
    // ------------------------------------------------------------------

    struct IUrl {
        urltype: String,
        infile: String,
        outfile: String,
        extspec: String,
        rowfilter: String,
        binspec: String,
        colspec: String,
    }

    fn iurl(url: &str) -> (c_int, IUrl) {
        let mut status: c_int = 0;
        let mut urltype = [0 as c_char; 80];
        let mut infile = [0 as c_char; FLEN_FILENAME];
        let mut outfile = [0 as c_char; FLEN_FILENAME];
        let mut extspec = [0 as c_char; FLEN_FILENAME];
        let mut rowfilter = [0 as c_char; FLEN_FILENAME];
        let mut binspec = [0 as c_char; FLEN_FILENAME];
        let mut colspec = [0 as c_char; FLEN_FILENAME];
        let rc = ffiurl_safe(
            &cc(url),
            Some(&mut urltype),
            Some(&mut infile),
            Some(&mut outfile),
            Some(&mut extspec),
            Some(&mut rowfilter),
            Some(&mut binspec),
            Some(&mut colspec),
            &mut status,
        );
        (
            rc,
            IUrl {
                urltype: from_buf(&urltype).to_string(),
                infile: from_buf(&infile).to_string(),
                outfile: from_buf(&outfile).to_string(),
                extspec: from_buf(&extspec).to_string(),
                rowfilter: from_buf(&rowfilter).to_string(),
                binspec: from_buf(&binspec).to_string(),
                colspec: from_buf(&colspec).to_string(),
            },
        )
    }

    #[test]
    fn test_ffiurl() {
        let (_, u) = iurl("file.fits[1]");
        assert_eq!(u.urltype, "file://");
        assert_eq!(u.infile, "file.fits");
        assert_eq!(u.extspec, "1");
    }

    #[test]
    fn test_ffiurl_outfile() {
        let (_, u) = iurl("file.fits(output.fits)");
        assert_eq!(u.infile, "file.fits");
        assert_eq!(u.outfile, "output.fits");
    }

    #[test]
    fn test_ffiurl_rowfilter() {
        let (_, u) = iurl("file.fits[EVENTS][X>100]");
        assert_eq!(u.infile, "file.fits");
        assert_eq!(u.extspec, "EVENTS");
        assert_eq!(u.rowfilter, "X>100");
    }

    #[test]
    fn test_ffiurl_binspec() {
        let (_, u) = iurl("file.fits[bin X,Y]");
        assert_eq!(u.infile, "file.fits");
        assert_eq!(u.binspec, "bin X,Y");
    }

    #[test]
    fn test_ffiurl_colspec() {
        let (_, u) = iurl("file.fits[col X;Y]");
        assert_eq!(u.infile, "file.fits");
        assert_eq!(u.colspec, "col X;Y");
    }

    #[test]
    fn test_ffiurl_mem() {
        let (_, u) = iurl("mem://filename");
        assert_eq!(u.urltype, "mem://");
        assert_eq!(u.infile, "filename");
    }

    #[test]
    fn test_ffiurl_compressed() {
        let (_, u) = iurl("file.fits.gz[1]");
        assert_eq!(u.urltype, "file://");
        assert_eq!(u.infile, "file.fits.gz");
        assert_eq!(u.extspec, "1");
    }

    #[test]
    fn test_ffiurl_urltype_no_slashes() {
        let (_, u) = iurl("ftp:example.com/file.fits");
        assert_eq!(u.urltype, "ftp://");
        assert_eq!(u.infile, "example.com/file.fits");

        let (_, u) = iurl("http:example.com/file.fits");
        assert_eq!(u.urltype, "http://");
        assert_eq!(u.infile, "example.com/file.fits");

        let (_, u) = iurl("file:localfile.fits");
        assert_eq!(u.urltype, "file://");
        assert_eq!(u.infile, "localfile.fits");

        let (_, u) = iurl("mem:memfile");
        assert_eq!(u.urltype, "mem://");
        assert_eq!(u.infile, "memfile");

        let (_, u) = iurl("shmem:sharedmem");
        assert_eq!(u.urltype, "shmem://");
        assert_eq!(u.infile, "sharedmem");

        let (_, u) = iurl("gsiftp:example.com/file.fits");
        assert_eq!(u.urltype, "gsiftp://");
        assert_eq!(u.infile, "example.com/file.fits");
    }

    #[test]
    fn test_ffiurl_urltype_too_long() {
        let (rc, _) = iurl("verylongurltypeprefix://file.fits");
        assert_eq!(rc, URL_PARSE_ERROR);
    }

    #[test]
    fn test_ffiurl_outfile_urltype() {
        let (_, u) = iurl("input.fits(mem://output.fits)");
        assert_eq!(u.urltype, "file://");
        assert_eq!(u.infile, "input.fits");
        assert_eq!(u.outfile, "mem://output.fits");
    }

    // ------------------------------------------------------------------
    // ffifile (fits_parse_input_filename) test
    // ------------------------------------------------------------------

    #[test]
    fn test_ffifile() {
        let mut status: c_int = 0;
        let mut urltype = [0 as c_char; 80];
        let mut infile = [0 as c_char; FLEN_FILENAME];
        let mut outfile = [0 as c_char; FLEN_FILENAME];
        let mut extspec = [0 as c_char; FLEN_FILENAME];
        let mut rowfilter = [0 as c_char; FLEN_FILENAME];
        let mut binspec = [0 as c_char; FLEN_FILENAME];
        let mut colspec = [0 as c_char; FLEN_FILENAME];
        let mut pixfilter = [0 as c_char; FLEN_FILENAME];
        ffifile_safe(
            &cc("file.fits[1]"),
            Some(&mut urltype),
            Some(&mut infile),
            Some(&mut outfile),
            Some(&mut extspec),
            Some(&mut rowfilter),
            Some(&mut binspec),
            Some(&mut colspec),
            Some(&mut pixfilter),
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(from_buf(&urltype), "file://");
        assert_eq!(from_buf(&infile), "file.fits");
        assert_eq!(from_buf(&extspec), "1");
    }

    // ------------------------------------------------------------------
    // ffourl (fits_parse_output_url) tests
    // ------------------------------------------------------------------

    struct OUrl {
        urltype: String,
        outfile: String,
        tpltfile: String,
        compspec: String,
    }

    fn ourl(url: &str) -> OUrl {
        let mut status: c_int = 0;
        let mut urltype = [0 as c_char; 80];
        let mut outfile = [0 as c_char; FLEN_FILENAME];
        let mut tpltfile = [0 as c_char; FLEN_FILENAME];
        let mut compspec = [0 as c_char; FLEN_FILENAME];
        ffourl(
            &cc(url),
            &mut urltype,
            &mut outfile,
            &mut tpltfile,
            &mut compspec,
            &mut status,
        );
        assert_eq!(status, 0);
        OUrl {
            urltype: from_buf(&urltype).to_string(),
            outfile: from_buf(&outfile).to_string(),
            tpltfile: from_buf(&tpltfile).to_string(),
            compspec: from_buf(&compspec).to_string(),
        }
    }

    #[test]
    fn test_ffourl_template() {
        let o = ourl("output.fits(template.fits)");
        assert_eq!(o.urltype, "file://");
        assert_eq!(o.outfile, "output.fits");
        assert_eq!(o.tpltfile, "template.fits");
        assert_eq!(o.compspec, "");
    }

    #[test]
    fn test_ffourl_compress() {
        let o = ourl("output.fits[compress]");
        assert_eq!(o.outfile, "output.fits");
        assert_eq!(o.compspec, "compress");
    }

    #[test]
    fn test_ffourl_gzip() {
        let o = ourl("output.fits.gz");
        assert_eq!(o.urltype, "compressoutfile://");
        assert_eq!(o.outfile, "output.fits.gz");
    }

    #[test]
    fn test_ffourl_stdout() {
        assert_eq!(ourl("-").urltype, "stdout://");
        assert_eq!(ourl("stdout").urltype, "stdout://");
        assert_eq!(ourl("STDOUT").urltype, "stdout://");
    }

    #[test]
    fn test_ffourl_explicit_urltype() {
        let o = ourl("mem://memfile.fits");
        assert_eq!(o.urltype, "mem://");
        assert_eq!(o.outfile, "memfile.fits");
    }

    // ------------------------------------------------------------------
    // Misc file operations
    // ------------------------------------------------------------------

    #[test]
    fn test_ffflus() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);

            fits_flush_file(f.as_deref_mut().unwrap(), &mut status);
            assert_eq!(status, 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffghdn() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);

            let mut hdunum: c_int = 0;
            fits_get_hdu_num(f.as_deref_mut().unwrap(), &mut hdunum);
            assert_eq!(hdunum, 1);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffdelt() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            assert_eq!(status, 0, "setup");

            // Delete it (closes and removes)
            ffdelt_safe(&mut f, &mut status);
            assert_eq!(status, 0, "ffdelt failed");

            // Verify it's gone
            let mut exists: c_int = -99;
            fits_file_exists(&name, &mut exists, &mut status);
            assert_eq!(status, 0);
            assert_eq!(exists, 0);
        });
    }

    #[test]
    fn test_ffflnm() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);

            let mut fname = [0 as c_char; FLEN_FILENAME];
            fits_file_name(f.as_deref_mut().unwrap(), &mut fname, &mut status);
            assert_eq!(status, 0);
            assert!(from_buf(&fname).contains("test.fits"));

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffflmd() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            // Test write mode
            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            let mut mode: c_int = 0;
            fits_file_mode(f.as_deref_mut().unwrap(), &mut mode, &mut status);
            assert_eq!(mode, READWRITE);
            fits_close_file(f.take().unwrap(), &mut status);

            // Test read mode
            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_file_mode(f.as_deref_mut().unwrap(), &mut mode, &mut status);
            assert_eq!(mode, READONLY);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffflmd_readwrite() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READWRITE, &mut status);
            let mut mode: c_int = 0;
            fits_file_mode(f.as_deref_mut().unwrap(), &mut mode, &mut status);
            assert_eq!(mode, READWRITE);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_memory_file() {
        let mut status: c_int = 0;
        let naxes: [c_long; 1] = [10];

        let mut f: Option<Box<fitsfile>> = None;
        fits_create_file(&mut f, &to_buf("mem://"), &mut status);
        fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);

        let data: [u8; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        fits_write_img_byt(f.as_deref_mut().unwrap(), 1, 1, 10, &data, &mut status);
        assert_eq!(status, 0);
        fits_close_file(f.take().unwrap(), &mut status);
    }

    // ------------------------------------------------------------------
    // Memory buffer operations
    // ------------------------------------------------------------------

    /// Mirrors test_ffimem in ~/code/cfitsio/tests/test_cfileio.c
    ///
    /// `buffer` and `bufsize` are handed to the mem driver by address and it
    /// keeps those addresses for the life of the file, so both locals have to
    /// stay put until the file is closed -- as in the C.
    #[test]
    fn test_ffimem() {
        let mut status: c_int = 0;
        let naxes: [c_long; 1] = [5];
        let mut buffer: *mut c_void = core::ptr::null_mut();
        let mut bufsize: usize = 0;
        let data: [u8; 5] = [10, 20, 30, 40, 50];

        /* Create file in memory with realloc capability. */
        let mut f: Option<Box<fitsfile>> = None;
        fits_create_memfile(
            &mut f,
            &raw mut buffer,
            &mut bufsize,
            2880,
            Some(libc::realloc),
            &mut status,
        );
        assert_eq!(status, 0, "ffimem failed");
        fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
        fits_write_img_byt(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
        fits_close_file(f.take().unwrap(), &mut status);
        assert_eq!(status, 0);

        /* Buffer should now contain valid FITS data. */
        assert!(!buffer.is_null());
        assert!(bufsize >= 2880, "bufsize = {bufsize}");
        // SAFETY: the driver filled `buffer` with at least `bufsize` bytes.
        let head = unsafe { core::slice::from_raw_parts(buffer.cast::<u8>(), 6) };
        assert_eq!(head, b"SIMPLE");

        // SAFETY: allocated by the driver through the libc realloc we passed.
        unsafe { libc::free(buffer) };
    }

    /// Mirrors test_ffomem in ~/code/cfitsio/tests/test_cfileio.c
    #[test]
    fn test_ffomem() {
        let mut status: c_int = 0;
        let naxes: [c_long; 1] = [5];
        let mut buffer: *mut c_void = core::ptr::null_mut();
        let mut bufsize: usize = 0;
        let data: [u8; 5] = [10, 20, 30, 40, 50];

        /* First create a FITS file in memory. */
        let mut f: Option<Box<fitsfile>> = None;
        fits_create_memfile(
            &mut f,
            &raw mut buffer,
            &mut bufsize,
            2880,
            Some(libc::realloc),
            &mut status,
        );
        fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
        fits_write_img_byt(f.as_deref_mut().unwrap(), 1, 1, 5, &data, &mut status);
        fits_close_file(f.take().unwrap(), &mut status);
        assert_eq!(status, 0, "setup failed");

        /* Now open the same buffer for reading. */
        let mut f2: Option<Box<fitsfile>> = None;
        let mut result = [0u8; 5];
        let mut anynull: c_int = 0;
        fits_open_memfile(
            &mut f2,
            &str_to_c_array("mem://"),
            READONLY,
            (&raw mut buffer).cast::<*mut c_void>(),
            &mut bufsize,
            0,
            libc::realloc,
            &mut status,
        );
        assert_eq!(status, 0, "ffomem failed");
        fits_read_img_byt(
            f2.as_deref_mut().unwrap(),
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
        fits_close_file(f2.take().unwrap(), &mut status);
        assert_eq!(status, 0);

        // SAFETY: allocated by the driver through the libc realloc we passed.
        unsafe { libc::free(buffer) };
    }

    // ------------------------------------------------------------------
    // Misc
    // ------------------------------------------------------------------

    #[test]
    fn test_fits_is_reentrant() {
        let reentrant = crate::fitscore::fits_is_reentrant_safe();
        assert!(reentrant == 0 || reentrant == 1);
    }

    #[test]
    fn test_ffdkopn() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            ffdkopn_safe(&mut f, &name, READWRITE, &mut status);
            assert_eq!(status, 0, "ffdkopn failed");
            assert!(f.is_some());

            let mut bitpix: c_int = 0;
            fits_get_img_equivtype(f.as_deref_mut().unwrap(), &mut bitpix, &mut status);
            assert_eq!(bitpix, BYTE_IMG);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffdkinit() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [10, 10];

            // with_temp_file gives a fresh dir; the file does not yet exist
            let mut f: Option<Box<fitsfile>> = None;
            ffdkinit_safe(&mut f, &name, &mut status);
            assert_eq!(status, 0, "ffdkinit failed");
            assert!(f.is_some());

            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut naxis: c_int = 0;
            fits_get_img_dim(f.as_deref_mut().unwrap(), &mut naxis, &mut status);
            assert_eq!(naxis, 2);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffrprt() {
        // Push some error messages.
        fits_write_errmsg(&cc("Test error message 1"));
        fits_write_errmsg(&cc("Test error message 2"));

        // Call ffrprt - pass null stream (function returns immediately, just
        // verify it doesn't panic). The C version writes to stdout.
        fits_report_error(core::ptr::null_mut(), 1);

        // Clear error stack.
        fits_clear_errmsg();
    }

    #[test]
    fn test_ffgtmo() {
        let timeout = fits_get_timeout();
        assert!(timeout >= 0);
    }

    #[test]
    fn test_ffstmo() {
        let mut status: c_int = 0;
        let orig_timeout = fits_get_timeout();

        fits_set_timeout(30, &mut status);
        assert_eq!(status, 0);

        let new_timeout = fits_get_timeout();
        // If net services are available, should be 30. Otherwise 0.
        assert!(new_timeout == 30 || new_timeout == 0);

        if orig_timeout > 0 {
            fits_set_timeout(orig_timeout, &mut status);
            assert_eq!(status, 0);
        }
    }

    /// Mirrors test_ffihtps in ~/code/cfitsio/tests/test_cfileio.c
    ///
    /// NOTE: the C's `fail_if(status != 0)` cannot be expressed yet -- ffihtps
    /// and ffchtps return `int` in fitsio.h, but their stubs here return `()`.
    /// Fixing that is part of implementing them.
    #[test]
    #[ignore = "ffihtps_safe / ffchtps_safe are still todo!() in src/cfileio.rs"]
    fn test_ffihtps() {
        /* Initialize HTTPS support. Returns 0 on success or if CURL not configured. */
        fits_init_https();

        /* Cleanup. */
        fits_cleanup_https();
    }

    #[test]
    fn test_ffeopn() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status); // null primary
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["DATA"],
                &["1E"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let mut hdutype: c_int = 0;
            ffeopn_safe(
                &mut f,
                &name,
                READONLY,
                Some(&cc("DATA")),
                Some(&mut hdutype),
                &mut status,
            );
            assert_eq!(status, 0, "ffeopn failed");
            assert!(f.is_some());
            assert_eq!(hdutype, BINARY_TBL);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_fits_is_this_a_copy() {
        fn isacopy(s: &str) -> c_int {
            let mut buf = [0 as c_char; 20];
            for (i, &b) in s.as_bytes().iter().enumerate() {
                buf[i] = b as c_char;
            }
            fits_is_this_a_copy(buf)
        }
        assert_eq!(isacopy("mem://"), 1);
        assert_eq!(isacopy("compress://"), 1);
        assert_eq!(isacopy("stdin://"), 1);
        assert_eq!(isacopy("http://"), 1);
        assert_eq!(isacopy("ftp://"), 1);
        assert_eq!(isacopy("file://"), 0);
    }

    #[test]
    fn test_ffvhtps_ffshdwn() {
        fits_verbose_https(1);
        fits_verbose_https(0);
        fits_show_download_progress(1);
        fits_show_download_progress(0);
    }

    #[test]
    fn test_ffimport_file() {
        // Create a simple text file in a temp dir.
        let dir = tempfile::Builder::new()
            .prefix("rsfitsio-import-")
            .tempdir()
            .unwrap();
        let path = dir.path().join("test_import.txt");
        std::fs::write(&path, "Line 1\nLine 2\nLine 3\n").unwrap();

        let mut status: c_int = 0;
        let mut contents: Option<Vec<c_char>> = None;
        ffimport_file_safe(&to_buf(path.to_str().unwrap()), &mut contents, &mut status);
        assert_eq!(status, 0);
        let s = from_buf(contents.as_ref().unwrap());
        assert!(s.contains("Line 1"));
        assert!(s.contains("Line 2"));
        assert!(s.contains("Line 3"));
    }

    // ------------------------------------------------------------------
    // fitscore functions
    // ------------------------------------------------------------------

    #[test]
    fn test_ffgnrw() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                100,
                &["DATA"],
                &["1J"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut nrows: c_long = 0;
            fits_get_num_rows(f.as_deref_mut().unwrap(), &mut nrows, &mut status);
            assert_eq!(nrows, 100);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffghof() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [10, 10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 2, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut headstart: libc::off_t = 0;
            let mut datastart: libc::off_t = 0;
            let mut dataend: libc::off_t = 0;
            fits_get_hduoff(
                f.as_deref_mut().unwrap(),
                Some(&mut headstart),
                Some(&mut datastart),
                Some(&mut dataend),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(headstart, 0); // Primary HDU starts at 0
            assert!(datastart > headstart); // Data after header
            assert!(dataend > datastart); // End after start
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffghad() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [10, 10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 2, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut headstart: c_long = 0;
            let mut datastart: c_long = 0;
            let mut dataend: c_long = 0;
            fits_get_hduaddr(
                f.as_deref_mut().unwrap(),
                Some(&mut headstart),
                Some(&mut datastart),
                Some(&mut dataend),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(headstart, 0);
            assert!(datastart > headstart);
            assert!(dataend > datastart);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgiprll() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 3] = [100, 200, 3];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), LONG_IMG, 3, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);

            fits_open_file(&mut f, &name, READONLY, &mut status);
            let mut bitpix: c_int = 0;
            let mut naxis: c_int = 0;
            let mut naxesll = [0 as LONGLONG; 3];
            fits_get_img_paramll(
                f.as_deref_mut().unwrap(),
                3,
                Some(&mut bitpix),
                Some(&mut naxis),
                Some(&mut naxesll),
                &mut status,
            );
            assert_eq!(bitpix, LONG_IMG);
            assert_eq!(naxis, 3);
            assert_eq!(naxesll[0], 100);
            assert_eq!(naxesll[1], 200);
            assert_eq!(naxesll[2], 3);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ------------------------------------------------------------------
    // Template functions
    // ------------------------------------------------------------------

    #[test]
    fn test_fftplt() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let naxes: [c_long; 2] = [20, 20];

            // template path lives in same temp dir
            let dir = std::path::Path::new(filename).parent().unwrap();
            let tpl = dir.join("template.fits");
            let out = dir.join("from_template.fits");
            let tpl_name = to_buf(tpl.to_str().unwrap());
            let out_name = to_buf(out.to_str().unwrap());

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &tpl_name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), SHORT_IMG, 2, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let mut f2: Option<Box<fitsfile>> = None;
            fftplt_safe(&mut f2, &out_name, &tpl_name, &mut status);
            assert_eq!(status, 0, "fftplt failed");
            assert!(f2.is_some());

            let mut naxis: c_int = 0;
            fits_get_img_dim(f2.as_deref_mut().unwrap(), &mut naxis, &mut status);
            assert_eq!(naxis, 2);
            let mut bitpix: c_int = 0;
            fits_get_img_equivtype(f2.as_deref_mut().unwrap(), &mut bitpix, &mut status);
            assert_eq!(bitpix, SHORT_IMG);

            fits_close_file(f2.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_fftplt_multi_hdu() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let naxes: [c_long; 1] = [10];

            let dir = std::path::Path::new(filename).parent().unwrap();
            let tpl = dir.join("template.fits");
            let out = dir.join("from_template.fits");
            let tpl_name = to_buf(tpl.to_str().unwrap());
            let out_name = to_buf(out.to_str().unwrap());

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &tpl_name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                5,
                &["DATA"],
                &["1D"],
                Some("TABLE"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let mut f2: Option<Box<fitsfile>> = None;
            fftplt_safe(&mut f2, &out_name, &tpl_name, &mut status);
            assert_eq!(status, 0, "fftplt failed");

            let mut nhdu: c_int = 0;
            fits_get_num_hdus(f2.as_deref_mut().unwrap(), &mut nhdu, &mut status);
            assert_eq!(nhdu, 2);

            fits_close_file(f2.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffoptplt_empty() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [5];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);

            // Empty template should be a no-op.
            ffoptplt(f.as_deref_mut().unwrap(), &cc(""), &mut status);
            assert_eq!(status, 0);

            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ------------------------------------------------------------------
    // Extension opening
    // ------------------------------------------------------------------

    #[test]
    fn test_ffopen_extension_number() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["DATA"],
                &["1E"],
                Some("EXT1"),
                &mut status,
            );
            make_table(
                f.as_deref_mut().unwrap(),
                20,
                &["DATA"],
                &["1E"],
                Some("EXT2"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[2]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            assert_eq!(status, 0, "ffopen[2] failed");
            let mut hdunum: c_int = 0;
            fits_get_hdu_num(f.as_deref_mut().unwrap(), &mut hdunum);
            assert_eq!(hdunum, 3);
            let mut hdutype: c_int = 0;
            fits_get_hdu_type(f.as_deref_mut().unwrap(), &mut hdutype, &mut status);
            assert_eq!(hdutype, BINARY_TBL);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffopen_extension_name() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["DATA"],
                &["1E"],
                Some("FIRST"),
                &mut status,
            );
            make_table(
                f.as_deref_mut().unwrap(),
                20,
                &["DATA"],
                &["1E"],
                Some("SECOND"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[SECOND]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            assert_eq!(status, 0, "ffopen[SECOND] failed");
            let mut hdunum: c_int = 0;
            fits_get_hdu_num(f.as_deref_mut().unwrap(), &mut hdunum);
            assert_eq!(hdunum, 3);
            let mut extname = [0 as c_char; 80];
            fits_read_key_str(
                f.as_deref_mut().unwrap(),
                &cc("EXTNAME"),
                &mut extname,
                None,
                &mut status,
            );
            assert_eq!(from_buf(&extname), "SECOND");
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffopen_extension_name_version() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["DATA"],
                &["1E"],
                Some("DATA"),
                &mut status,
            );
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("EXTVER"),
                1,
                None,
                &mut status,
            );
            make_table(
                f.as_deref_mut().unwrap(),
                20,
                &["DATA"],
                &["1E"],
                Some("DATA"),
                &mut status,
            );
            fits_write_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("EXTVER"),
                2,
                None,
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[DATA,2]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            assert_eq!(status, 0, "ffopen[DATA,2] failed");
            let mut hdunum: c_int = 0;
            fits_get_hdu_num(f.as_deref_mut().unwrap(), &mut hdunum);
            assert_eq!(hdunum, 3);
            let mut extver: c_long = 0;
            fits_read_key_lng(
                f.as_deref_mut().unwrap(),
                &cc("EXTVER"),
                &mut extver,
                None,
                &mut status,
            );
            assert_eq!(extver, 2);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffopen_plus_extension() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["DATA"],
                &["1E"],
                Some("TABLE"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "+1");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            assert_eq!(status, 0, "ffopen+1 failed");
            let mut hdunum: c_int = 0;
            fits_get_hdu_num(f.as_deref_mut().unwrap(), &mut hdunum);
            assert_eq!(hdunum, 2);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffopen_image_type() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let mut naxes: [c_long; 1] = [10];

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["DATA"],
                &["1E"],
                Some("TABLE"),
                &mut status,
            );
            naxes[0] = 20;
            fits_create_img(f.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[TABLE,1,b]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            assert_eq!(status, 0, "ffopen[TABLE,1,b] failed");
            let mut hdutype: c_int = 0;
            fits_get_hdu_type(f.as_deref_mut().unwrap(), &mut hdutype, &mut status);
            assert_eq!(hdutype, BINARY_TBL);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    /// Mirrors test_fits_copy_cell2image in ~/code/cfitsio/tests/test_cfileio.c
    #[test]
    fn test_fits_copy_cell2image() {
        with_temp_file(|inname| {
            with_temp_file(|outname| {
                let mut status: c_int = 0;
                let iname = to_buf(inname);
                let oname = to_buf(outname);

                /* Create binary table with a 5x5 image cell. */
                let mut f1: Option<Box<fitsfile>> = None;
                fits_create_file(&mut f1, &iname, &mut status);
                fits_write_imghdr(f1.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
                make_table(
                    f1.as_deref_mut().unwrap(),
                    3,
                    &["IMAGE"],
                    &["25J"],
                    Some("IMAGES"),
                    &mut status,
                );

                /* Add TDIM to define the 5x5 shape. */
                fits_write_key_str(
                    f1.as_deref_mut().unwrap(),
                    &cc("TDIM1"),
                    &cc("(5,5)"),
                    None,
                    &mut status,
                );

                /* Write test data to the first row. */
                let data: Vec<c_int> = (0..25).map(|i| i * 100).collect();
                fits_write_col_int(f1.as_deref_mut().unwrap(), 1, 1, 1, 25, &data, &mut status);
                fits_close_file(f1.take().unwrap(), &mut status);
                assert_eq!(status, 0, "setup failed");

                /* Reopen and copy the cell to a new image file. */
                let ext_name = path_with_ext(inname, "[IMAGES]");
                ffopen_safe(&mut f1, &ext_name, READONLY, &mut status);
                let mut f2: Option<Box<fitsfile>> = None;
                fits_create_file(&mut f2, &oname, &mut status);
                assert_eq!(status, 0, "reopen failed");

                fits_copy_cell2image_safe(
                    f1.as_deref_mut().unwrap(),
                    f2.as_deref_mut().unwrap(),
                    &cc("IMAGE"),
                    1,
                    &mut status,
                );
                assert_eq!(status, 0, "fits_copy_cell2image failed");

                /* Verify the output image. */
                let g = f2.as_deref_mut().unwrap();
                let mut naxis: c_int = 0;
                let mut bitpix: c_int = 0;
                let mut naxes_out = [0 as c_long; 2];
                fits_get_img_dim(g, &mut naxis, &mut status);
                assert_eq!(naxis, 2);
                fits_get_img_size(g, 2, &mut naxes_out, &mut status);
                assert_eq!(naxes_out, [5, 5]);
                fits_get_img_equivtype(g, &mut bitpix, &mut status);
                assert_eq!(bitpix, LONG_IMG);
                assert_eq!(status, 0);

                /* the C stops at the header; check the pixels came across too */
                let mut got = [0 as c_int; 25];
                let mut anynul: c_int = 0;
                fits_read_img_int(g, 1, 1, 25, 0, &mut got, Some(&mut anynul), &mut status);
                assert_eq!(status, 0);
                assert_eq!(got.to_vec(), data);

                fits_close_file(f1.take().unwrap(), &mut status);
                fits_close_file(f2.take().unwrap(), &mut status);
                assert_eq!(status, 0);
            });
        });
    }

    // ------------------------------------------------------------------
    // Table functions
    // ------------------------------------------------------------------

    #[test]
    fn test_ffgncl() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["A", "B", "C"],
                &["1J", "1E", "10A"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[DATA]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            let mut ncols: c_int = 0;
            fits_get_num_cols(f.as_deref_mut().unwrap(), &mut ncols, &mut status);
            assert_eq!(ncols, 3);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_fits_get_num_cols() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["A", "B"],
                &["1J", "1E"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[DATA]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            let mut ncols: c_int = 0;
            fits_get_num_cols(f.as_deref_mut().unwrap(), &mut ncols, &mut status);
            assert_eq!(ncols, 2);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcno() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["ALPHA", "BETA", "GAMMA"],
                &["1J", "1E", "10A"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[DATA]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            let mut colnum: c_int = 0;
            fits_get_colnum(
                f.as_deref_mut().unwrap(),
                crate::fitsio::CASEINSEN as c_int,
                &cc("BETA"),
                &mut colnum,
                &mut status,
            );
            assert_eq!(colnum, 2);
            fits_get_colnum(
                f.as_deref_mut().unwrap(),
                crate::fitsio::CASEINSEN as c_int,
                &cc("beta"),
                &mut colnum,
                &mut status,
            );
            assert_eq!(colnum, 2);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_fits_get_colname() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["ALPHA", "BETA"],
                &["1J", "1E"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[DATA]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            let mut colname = [0 as c_char; FLEN_VALUE];
            let mut colnum: c_int = 0;
            fits_get_colname(
                f.as_deref_mut().unwrap(),
                crate::fitsio::CASEINSEN as c_int,
                &cc("ALP*"),
                &mut colname,
                &mut colnum,
                &mut status,
            );
            assert_eq!(colnum, 1);
            assert_eq!(from_buf(&colname), "ALPHA");
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgnrwll() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                1000,
                &["DATA"],
                &["1J"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            fits_open_file(&mut f, &name, READONLY, &mut status);
            fits_movabs_hdu(f.as_deref_mut().unwrap(), 2, None, &mut status);
            let mut nrows: LONGLONG = 0;
            fits_get_num_rowsll(f.as_deref_mut().unwrap(), &mut nrows, &mut status);
            assert_eq!(nrows, 1000);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgtcl() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["INT", "FLOAT", "STRING"],
                &["1J", "1E", "20A"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[DATA]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            let mut typecode: c_int = 0;
            let mut repeat: c_long = 0;
            let mut width: c_long = 0;
            fits_get_coltype(
                f.as_deref_mut().unwrap(),
                1,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                &mut status,
            );
            assert_eq!(typecode, TLONG);
            assert_eq!(repeat, 1);
            fits_get_coltype(
                f.as_deref_mut().unwrap(),
                2,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                &mut status,
            );
            assert_eq!(typecode, TFLOAT);
            fits_get_coltype(
                f.as_deref_mut().unwrap(),
                3,
                Some(&mut typecode),
                Some(&mut repeat),
                Some(&mut width),
                &mut status,
            );
            assert_eq!(typecode, TSTRING);
            assert_eq!(repeat, 20);
            assert_eq!(width, 20);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgcdw() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                10,
                &["INT", "FLOAT", "STRING"],
                &["1J", "1E", "20A"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[DATA]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            let mut dispwidth: c_int = 0;
            fits_get_col_display_width(f.as_deref_mut().unwrap(), 3, &mut dispwidth, &mut status);
            assert_eq!(dispwidth, 20);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_ffgabc() {
        let mut status: c_int = 0;
        let tform_v = [cc("A20"), cc("E12.4")];
        let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
        let mut rowlen: c_long = 0;
        let mut tbcol = [0 as c_long; 2];
        fits_get_tbcol(2, &tform_ref, 1, &mut rowlen, &mut tbcol, &mut status);
        assert_eq!(status, 0);
        assert!(rowlen >= 32);
        assert!(tbcol[0] >= 1);
    }

    #[test]
    fn test_ffgrsz() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);

            let mut f: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f, &name, &mut status);
            fits_write_imghdr(f.as_deref_mut().unwrap(), BYTE_IMG, 0, &[], &mut status);
            make_table(
                f.as_deref_mut().unwrap(),
                1000,
                &["DATA"],
                &["1E"],
                Some("DATA"),
                &mut status,
            );
            fits_close_file(f.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup");

            let ext_name = path_with_ext(filename, "[DATA]");
            ffopen_safe(&mut f, &ext_name, READONLY, &mut status);
            let mut nrows: c_long = 0;
            fits_get_rowsize(f.as_deref_mut().unwrap(), &mut nrows, &mut status);
            assert!(nrows > 0);
            fits_close_file(f.take().unwrap(), &mut status);
        });
    }

    // ------------------------------------------------------------------
    // Keyword classification
    // ------------------------------------------------------------------

    #[test]
    fn test_fits_get_keyclass() {
        assert_eq!(fits_get_keyclass(&cc("SIMPLE")), TYP_STRUC_KEY);
        assert_eq!(fits_get_keyclass(&cc("BITPIX")), TYP_STRUC_KEY);
        assert_eq!(fits_get_keyclass(&cc("COMMENT")), TYP_COMM_KEY);
        assert_eq!(fits_get_keyclass(&cc("HISTORY")), TYP_COMM_KEY);
        assert_eq!(fits_get_keyclass(&cc("MYKEY")), TYP_USER_KEY);
    }

    // ------------------------------------------------------------------
    // Version and utility functions
    // ------------------------------------------------------------------

    #[test]
    fn test_fits_get_version() {
        let mut version: f32 = 0.0;
        fits_get_version(&mut version);
        assert!(version >= 4.0);
    }

    // ------------------------------------------------------------------
    // fits_already_open
    // ------------------------------------------------------------------

    #[test]
    fn test_fits_already_open() {
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 1] = [10];

            let mut f1: Option<Box<fitsfile>> = None;
            fits_create_file(&mut f1, &name, &mut status);
            fits_write_imghdr(f1.as_deref_mut().unwrap(), BYTE_IMG, 1, &naxes, &mut status);
            assert_eq!(status, 0, "setup");

            // Opening again should give a different fitsfile pointer that shares
            // the same underlying FITSfile.
            let mut f2: Option<Box<fitsfile>> = None;
            ffopen_safe(&mut f2, &name, READWRITE, &mut status);
            assert_eq!(status, 0, "ffopen again failed");
            assert!(f2.is_some());

            // Both should point to same underlying file.
            let p1: *const _ = &raw const *f1.as_deref().unwrap().Fptr;
            let p2: *const _ = &raw const *f2.as_deref().unwrap().Fptr;
            assert_eq!(p1, p2);

            fits_close_file(f1.take().unwrap(), &mut status);
            fits_close_file(f2.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_create_and_read() {
        // test_create_file + test_open_and_read
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let naxes: [c_long; 2] = [4, 3];
            let data: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

            let mut fptr: Option<Box<fitsfile>> = None;
            fits_create_file(&mut fptr, &name, &mut status);
            assert_eq!(status, 0, "ffinit failed");
            fits_write_imghdr(
                fptr.as_deref_mut().unwrap(),
                BYTE_IMG,
                2,
                &naxes,
                &mut status,
            );
            assert_eq!(status, 0, "ffphps failed");
            fits_write_img(
                fptr.as_deref_mut().unwrap(),
                TBYTE,
                1,
                data.len() as LONGLONG,
                &data,
                &mut status,
            );
            assert_eq!(status, 0, "ffppr failed");
            fits_close_file(fptr.take().unwrap(), &mut status);

            // open and read
            fits_open_file(&mut fptr, &name, READONLY, &mut status);
            assert_eq!(status, 0, "ffopen failed");
            let mut rdata = [0u8; 12];
            let mut anynull: c_int = 0;
            fits_read_img(
                fptr.as_deref_mut().unwrap(),
                TBYTE,
                1,
                12,
                None,
                &mut rdata,
                Some(&mut anynull),
                &mut status,
            );
            assert_eq!(status, 0, "ffgpv failed");
            assert_eq!(rdata[0], 1);
            assert_eq!(rdata[5], 6);
            assert_eq!(rdata[11], 12);
            fits_close_file(fptr.take().unwrap(), &mut status);
        });
    }

    #[test]
    fn test_error_handling() {
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        let name = to_buf("nonexistent_file.fits");
        let rc = fits_open_file(&mut fptr, &name, READONLY, &mut status);
        assert_ne!(rc, 0, "opening a nonexistent file should fail");
        assert_ne!(status, 0);

        let mut errmsg = [0 as c_char; FLEN_VALUE];
        fits_get_errstatus(status, &mut errmsg);
        assert!(!from_buf(&errmsg).is_empty(), "errmsg should be non-empty");
        fits_clear_errmsg();
    }

    #[test]
    fn test_hdu_and_keywords() {
        // Replicates main()'s READWRITE section operating on one handle.
        with_temp_file(|filename| {
            let mut status: c_int = 0;
            let name = to_buf(filename);
            let pnaxes: [c_long; 2] = [4, 3];
            let data: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

            // create the file (as test_create_file does)
            let mut fptr: Option<Box<fitsfile>> = None;
            fits_create_file(&mut fptr, &name, &mut status);
            fits_write_imghdr(
                fptr.as_deref_mut().unwrap(),
                BYTE_IMG,
                2,
                &pnaxes,
                &mut status,
            );
            fits_write_img(
                fptr.as_deref_mut().unwrap(),
                TBYTE,
                1,
                data.len() as LONGLONG,
                &data,
                &mut status,
            );
            fits_close_file(fptr.take().unwrap(), &mut status);
            assert_eq!(status, 0, "setup failed");

            fits_open_file(&mut fptr, &name, READWRITE, &mut status);
            assert_eq!(status, 0, "ffopen RW failed");
            let f = fptr.as_deref_mut().unwrap();

            // --- test_hdu_operations ---
            let mut hdunum: c_int = 0;
            fits_get_hdu_num(f, &mut hdunum);
            assert_eq!(hdunum, 1);

            let naxes: [c_long; 2] = [5, 5];
            fits_create_img(f, SHORT_IMG, 2, &naxes, &mut status);
            assert_eq!(status, 0, "ffcrim failed");
            fits_get_hdu_num(f, &mut hdunum);
            assert_eq!(hdunum, 2);

            let ttype = [Some(cc("COL1"))];
            let ttype_ref: Vec<Option<&[c_char]>> = ttype.iter().map(|o| o.as_deref()).collect();
            let tform_v = [cc("1J")];
            let tform_ref: Vec<&[c_char]> = tform_v.iter().map(|v| v.as_slice()).collect();
            fits_create_tbl(
                f,
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
            fits_get_hdu_num(f, &mut hdunum);
            assert_eq!(hdunum, 3);

            let mut hdutype: c_int = -1;
            fits_movabs_hdu(f, 1, Some(&mut hdutype), &mut status);
            assert_eq!(status, 0, "ffmahd failed");
            assert_eq!(hdutype, IMAGE_HDU);
            fits_get_hdu_num(f, &mut hdunum);
            assert_eq!(hdunum, 1);

            // --- test_write_keywords ---
            fits_write_key_str(
                f,
                &cc("STRKEY"),
                &cc("hello"),
                Some(&cc("string comment")),
                &mut status,
            );
            fits_write_key_lng(
                f,
                &cc("LONGKEY"),
                42,
                Some(&cc("long comment")),
                &mut status,
            );
            fits_write_key_dbl(
                f,
                &cc("DBLKEY"),
                3.14159,
                6,
                Some(&cc("double comment")),
                &mut status,
            );
            assert_eq!(status, 0, "writing keywords failed");

            // --- test_read_keywords ---
            let mut strval = [0 as c_char; FLEN_VALUE];
            let mut comment = [0 as c_char; FLEN_COMMENT];
            fits_read_key_str(
                f,
                &cc("STRKEY"),
                &mut strval,
                Some(&mut comment),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(from_buf(&strval), "hello");
            assert_eq!(from_buf(&comment), "string comment");

            let mut longval: c_long = 0;
            fits_read_key(
                f,
                KeywordDatatypeMut::TLONG(&mut longval),
                &cc("LONGKEY"),
                Some(&mut comment),
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(longval, 42);

            let mut dblval: f64 = 0.0;
            fits_read_key(
                f,
                KeywordDatatypeMut::TDOUBLE(&mut dblval),
                &cc("DBLKEY"),
                Some(&mut comment),
                &mut status,
            );
            assert_eq!(status, 0);
            assert!((3.14..=3.15).contains(&dblval));

            // --- test_keyword_types ---
            let mut value = [0 as c_char; FLEN_VALUE];
            fits_write_key_str(
                f,
                &cc("KEY1"),
                &cc("val1"),
                Some(&cc("comment")),
                &mut status,
            );
            fits_read_keyword(f, &cc("KEY1"), &mut value, Some(&mut comment), &mut status);
            assert_eq!(from_buf(&value), "'val1    '");

            fits_write_key_str(f, &cc("KEY2"), &cc(""), Some(&cc("empty")), &mut status);
            fits_read_keyword(f, &cc("KEY2"), &mut value, Some(&mut comment), &mut status);
            assert_eq!(from_buf(&value), "'        '");

            // C passes NULL value here -> an empty c_char slice maps to '' (ffs2c).
            fits_write_key_str(f, &cc("KEY3"), &[], Some(&cc("null")), &mut status);
            fits_read_keyword(f, &cc("KEY3"), &mut value, Some(&mut comment), &mut status);
            assert_eq!(from_buf(&value), "''");
            assert_eq!(status, 0, "keyword types failed");

            // --- test_raw_record ---
            let mut val = [0 as c_char; FLEN_VALUE];
            fits_write_record(f, &cc("RAWKEY  = 'rawval' / raw comment"), &mut status);
            fits_read_keyword(f, &cc("RAWKEY"), &mut val, Some(&mut comment), &mut status);
            assert_eq!(status, 0, "ffprec/ffgkey failed");
            assert_eq!(from_buf(&val), "'rawval'");

            // --- test_header_space ---
            let mut nkeys: c_int = 0;
            let mut morekeys: c_int = 0;
            fits_get_hdrspace(f, Some(&mut nkeys), Some(&mut morekeys), &mut status);
            assert_eq!(status, 0, "ffghsp failed");
            assert!(nkeys >= 1);
            for i in 1..=nkeys {
                let mut keyname = [0 as c_char; FLEN_KEYWORD];
                let mut kvalue = [0 as c_char; FLEN_VALUE];
                fits_read_keyn(
                    f,
                    i,
                    &mut keyname,
                    &mut kvalue,
                    Some(&mut comment),
                    &mut status,
                );
                assert_eq!(status, 0, "ffgkyn failed at key {i}");
            }

            fits_close_file(fptr.take().unwrap(), &mut status);
        });
    }

    // ------------------------------------------------------------------
    // check_is_mem_fits / check_is_file_fits tests
    // ------------------------------------------------------------------

    /// gzip a byte buffer using the library's own compressor (gzip format,
    /// so the output starts with the 0x1f 0x8b magic number).
    fn gzip_bytes(bytes: &[u8]) -> Vec<u8> {
        use crate::zcompress::compress2mem_from_mem;
        let mut status: c_int = 0;
        let mut out: Vec<u8> = Vec::new();
        let mut fsize: usize = 0;
        compress2mem_from_mem(
            cast_slice(bytes),
            bytes.len(),
            &mut out,
            Some(&mut fsize),
            &mut status,
        );
        assert_eq!(status, 0, "compress2mem_from_mem failed");
        out.truncate(fsize);
        out
    }

    /// Create a temporary on-disk file containing `bytes`, positioned at the
    /// start so it can be read from the beginning.
    fn temp_file_with(bytes: &[u8]) -> File {
        let mut f = tempfile::tempfile().unwrap();
        f.write_all(bytes).unwrap();
        f.rewind().unwrap();
        f
    }

    #[test]
    fn test_check_is_mem_fits_plain() {
        // A plain buffer that begins with the SIMPLE keyword is FITS.
        let simple = b"SIMPLE  =                    T / file conforms to FITS standard";
        let buf = cast_slice::<u8, c_char>(simple);
        assert_eq!(check_is_mem_fits(buf, simple.len()), 1);

        // Exactly the 6-char keyword is enough.
        let just = b"SIMPLE";
        assert_eq!(check_is_mem_fits(cast_slice(just), just.len()), 1);
    }

    #[test]
    fn test_check_is_mem_fits_not_fits() {
        // Non-FITS content.
        let other = b"RANDOM  =                    T";
        assert_eq!(check_is_mem_fits(cast_slice(other), other.len()), 0);

        // Fewer than 6 usable bytes is not FITS.
        let short = b"SIMPL";
        assert_eq!(check_is_mem_fits(cast_slice(short), short.len()), 0);

        // A prefix of SIMPLE but too short a length is rejected.
        let simple = b"SIMPLE";
        assert_eq!(check_is_mem_fits(cast_slice(simple), 5), 0);
    }

    #[test]
    fn test_check_is_mem_fits_gzip() {
        // A gzip-compressed FITS header is detected via the magic number.
        let simple = b"SIMPLE  =                    T / file conforms to FITS standard";
        let gz = gzip_bytes(simple);
        assert_eq!(gz[0] as u8, 0x1f);
        assert_eq!(gz[1] as u8, 0x8b);
        assert_eq!(check_is_mem_fits(cast_slice(&gz), gz.len()), 1);

        // A gzip-compressed non-FITS buffer is not FITS.
        let other = b"RANDOM DATA THAT IS NOT A FITS FILE AT ALL, JUST BYTES";
        let gz = gzip_bytes(other);
        assert_eq!(check_is_mem_fits(cast_slice(&gz), gz.len()), 0);
    }

    #[test]
    fn test_check_is_file_fits_plain() {
        let simple = b"SIMPLE  =                    T / file conforms to FITS standard";
        let mut f = temp_file_with(simple);
        assert_eq!(check_is_file_fits(&mut f), 1);

        // The file must be rewound by check_is_file_fits so a subsequent read
        // starts at the beginning again.
        let mut first = [0u8; 6];
        f.read_exact(&mut first).unwrap();
        assert_eq!(&first, b"SIMPLE");
    }

    #[test]
    fn test_check_is_file_fits_not_fits() {
        let other = b"RANDOM  =                    T";
        let mut f = temp_file_with(other);
        assert_eq!(check_is_file_fits(&mut f), 0);
    }

    #[test]
    fn test_check_is_file_fits_empty() {
        let mut f = temp_file_with(b"");
        assert_eq!(check_is_file_fits(&mut f), 0);
    }

    #[test]
    fn test_check_is_file_fits_gzip() {
        let simple = b"SIMPLE  =                    T / file conforms to FITS standard";
        let gz = gzip_bytes(simple);
        let mut f = temp_file_with(&gz);
        assert_eq!(check_is_file_fits(&mut f), 1);
    }

    // ------------------------------------------------------------------
    // exclude_path tests
    // ------------------------------------------------------------------

    #[test]
    fn test_exclude_path_absolute() {
        // Absolute paths into the restricted directories are excluded.
        assert_eq!(exclude_path(&str_to_c_array("/etc/passwd")), 1);
        assert_eq!(exclude_path(&str_to_c_array("/var/log/messages")), 1);

        // Other absolute paths are allowed.
        assert_eq!(exclude_path(&str_to_c_array("/home/user/data.fits")), 0);
        assert_eq!(exclude_path(&str_to_c_array("/tmp/out.fits")), 0);

        // The trailing slash matters: "/etcfoo" is not "/etc/".
        assert_eq!(exclude_path(&str_to_c_array("/etcfoo/x")), 0);
        assert_eq!(exclude_path(&str_to_c_array("/etc")), 0);
    }

    #[test]
    fn test_exclude_path_home() {
        // For '~' paths, both a '..' component and a restricted dir must appear.
        assert_eq!(exclude_path(&str_to_c_array("~/../etc/passwd")), 1);
        assert_eq!(exclude_path(&str_to_c_array("~/../var/log")), 1);

        // '..' without a restricted dir is allowed.
        assert_eq!(exclude_path(&str_to_c_array("~/../data/file.fits")), 0);

        // A restricted dir without '..' is allowed for '~' paths.
        assert_eq!(exclude_path(&str_to_c_array("~/etc/file.fits")), 0);
    }

    // ------------------------------------------------------------------
    // skip_host_string tests
    // ------------------------------------------------------------------

    fn skip_host(s: &str) -> String {
        let buf = str_to_c_array(s);
        from_buf(skip_host_string(&buf)).to_string()
    }

    #[test]
    fn test_skip_host_string_with_host() {
        // A "host:port/path" style prefix is stripped down to the path.
        assert_eq!(skip_host("host:8080/path/file.fits"), "/path/file.fits");
        // A "host.domain/path" prefix (dot before the slash) is a host too.
        assert_eq!(skip_host("host.com/path/file.fits"), "/path/file.fits");
    }

    #[test]
    fn test_skip_host_string_no_host() {
        // Absolute and home paths are returned unchanged.
        assert_eq!(skip_host("/absolute/path"), "/absolute/path");
        assert_eq!(skip_host("~/home/file"), "~/home/file");

        // Relative './' and '../' prefixes are not treated as a host.
        assert_eq!(skip_host("./rel/file"), "./rel/file");
        assert_eq!(skip_host("../rel/file"), "../rel/file");

        // No slash at all: returned unchanged.
        assert_eq!(skip_host("file.fits"), "file.fits");

        // A slash but no dot/colon before it: not a host.
        assert_eq!(skip_host("myfiles/data.fits"), "myfiles/data.fits");

        // Too short to hold a host name.
        assert_eq!(skip_host("a"), "a");
    }

    // ------------------------------------------------------------------
    // normalize_path tests
    // ------------------------------------------------------------------

    #[test]
    fn test_normalize_path_tilde_passthrough() {
        // '~' paths are left untouched.
        let mut status: c_int = 0;
        let mut buf = to_buf("~/data/file.fits");
        assert_eq!(normalize_path(&mut buf, &mut status), 0);
        assert_eq!(status, 0);
        assert_eq!(from_buf(&buf), "~/data/file.fits");
    }

    #[test]
    fn test_normalize_path_absolute() {
        // Absolute paths have '/./' and '/../' segments collapsed.
        let mut status: c_int = 0;
        let mut buf = to_buf("/usr/local/../bin/./tool");
        assert_eq!(normalize_path(&mut buf, &mut status), 0);
        assert_eq!(status, 0);
        assert_eq!(from_buf(&buf), "/usr/bin/tool");
    }
}
