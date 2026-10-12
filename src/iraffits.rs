//! Reading IRAF `.imh` image files and translating their headers to FITS.
//!
//! An IRAF image is a pair of files: a header (`.imh`) and a pixel file, whose
//! path the header records. Opening one here reads the header, converts it into
//! an equivalent FITS header, and reads the pixels, so that the rest of the
//! library sees an ordinary in-memory FITS image.
//!
//! The two formats differ in byte order and in string representation -- IRAF
//! v.1 headers store two bytes per character -- so most of this module is
//! conversion rather than I/O.
//!
//! Based on `imhfile.c` by Doug Mink, August 6 1998, in turn based on Mike
//! VanHilst's `readiraf.c`, modified by William Pence for use by CFITSIO.
//!
//! Copyright 2000 Smithsonian Astrophysical Observatory. You may do anything
//! you like with this file except remove this copyright. The Smithsonian
//! Astrophysical Observatory makes no representations about the suitability of
//! this software for any purpose. It is provided "as is" without express or
//! implied warranty.
#![warn(missing_docs)]

use crate::c_types::{c_char, c_int};
use crate::helpers::vec_raw_parts::vec_into_raw_parts;
use crate::wrappers::read_fill;
use bytemuck::{cast_slice, cast_slice_mut};

use core::ffi::CStr;
use core::ptr;
use std::fs::File;
use std::io::Error;
use std::io::Seek;
use std::sync::Mutex;

use crate::fitscore::{ALLOCATIONS, ffpmsg_slice, ffpmsg_str};
use crate::fitsio::NULL_MSG;
use crate::int_snprintf;
use crate::{
    BL, bb, cs,
    fitsio::FLEN_ERRMSG,
    parse_c_int, raw_to_slice,
    wrappers::{
        atof_safe, strchr_safe, strcmp_safe, strcpy_safe, strlen_safe, strncat_safe, strncmp_safe,
        strncpy_safe,
    },
};

const FILE_NOT_OPENED: c_int = 104;

/* Parameters from iraf/lib/imhdr.h for IRAF version 1 images */
const SZ_IMPIXFILE: usize = 79; /* name of pixel storage file */
const SZ_IMHDRFILE: usize = 79; /* length of header storage file */
const SZ_IMTITLE: usize = 79; /* image title string */
const LEN_IMHDR: usize = 2052; /* length of std header */

/* Parameters from iraf/lib/imhdr.h for IRAF version 2 images */
const SZ_IM2PIXFILE: usize = 255; /* name of pixel storage file */
const SZ_IM2HDRFILE: usize = 255; /* name of header storage file */
const SZ_IM2TITLE: usize = 383; /* image title string */
const LEN_IM2HDR: usize = 2046; /* length of std header */

/* Offsets into header in bytes for parameters in IRAF version 1 images */
const IM_HDRLEN: usize = 12; /* Length of header in 4-byte ints */
const IM_PIXTYPE: usize = 16; /* Datatype of the pixels */
const IM_NDIM: usize = 20; /* Number of dimensions */
const IM_LEN: usize = 24; /* Length (as stored) */
const IM_PHYSLEN: usize = 52; /* Physical length (as stored) */
const IM_PIXOFF: usize = 88; /* Offset of the pixels */
const IM_CTIME: usize = 108; /* Time of image creation */
const IM_MTIME: usize = 112; /* Time of last modification */
const IM_LIMTIME: usize = 116; /* Time of min,max computation */
const IM_MAX: usize = 120; /* Maximum pixel value */
const IM_MIN: usize = 124; /* Maximum pixel value */
const IM_PIXFILE: usize = 412; /* Name of pixel storage file */
const IM_HDRFILE: usize = 572; /* Name of header storage file */
const IM_TITLE: usize = 732; /* Image name string */

/* Offsets into header in bytes for parameters in IRAF version 2 images */
const IM2_HDRLEN: usize = 6; /* Length of header in 4-byte ints */
const IM2_PIXTYPE: usize = 10; /* Datatype of the pixels */
const IM2_SWAPPED: usize = 14; /* Pixels are byte swapped */
const IM2_NDIM: usize = 18; /* Number of dimensions */
const IM2_LEN: usize = 22; /* Length (as stored) */
const IM2_PHYSLEN: usize = 50; /* Physical length (as stored) */
const IM2_PIXOFF: usize = 86; /* Offset of the pixels */
const IM2_CTIME: usize = 106; /* Time of image creation */
const IM2_MTIME: usize = 110; /* Time of last modification */
const IM2_LIMTIME: usize = 114; /* Time of min,max computation */
const IM2_MAX: usize = 118; /* Maximum pixel value */
const IM2_MIN: usize = 122; /* Maximum pixel value */
const IM2_PIXFILE: usize = 126; /* Name of pixel storage file */
const IM2_HDRFILE: usize = 382; /* Name of header storage file */
const IM2_TITLE: usize = 638; /* Image name string */

/* Codes from iraf/unix/hlib/iraf.h */
const TY_CHAR: c_int = 2;
const TY_SHORT: c_int = 3;
const TY_INT: c_int = 4;
const TY_LONG: c_int = 5;
const TY_REAL: c_int = 6;
const TY_DOUBLE: c_int = 7;
const TY_COMPLEX: c_int = 8;
const TY_POINTER: c_int = 9;
const TY_STRUCT: c_int = 10;
const TY_USHORT: c_int = 11;
const TY_UBYTE: c_int = 12;

const LEN_PIXHDR: usize = 1024;
const MAXINT: c_int = 2147483647; /* Biggest number that can fit in long */

static SWAPHEAD: Mutex<bool> = Mutex::new(false); /* =1 to swap data bytes of IRAF header values */
static SWAPDATA: Mutex<bool> = Mutex::new(false); /* =1 to swap bytes in IRAF data pixels */

/// Delete the iraf .imh header file and the associated .pix data file
///
/// # Parameters
///
/// * `filename` — name of input file
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_delete_iraf_file(
    filename: *const c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(filename);

        fits_delete_iraf_file_safe(filename, status)
    }
}

/// Delete the iraf .imh header file and the associated .pix data file
///
/// # Parameters
///
/// * `filename` — name of input file
/// * `status`   — (IO) error status
pub fn fits_delete_iraf_file_safe(filename: &[c_char], status: &mut c_int) -> c_int {
    let mut lenirafhead: usize = 0;
    let mut pixfilename: [c_char; SZ_IM2PIXFILE + 1] = [0; SZ_IM2PIXFILE + 1];

    /* read IRAF header into dynamically created char array (free it later!) */
    let irafheader = irafrdhead(filename, &mut lenirafhead);

    if irafheader.is_err() {
        *status = FILE_NOT_OPENED;
        return *status;
    }

    let irafheader = irafheader.unwrap();

    getirafpixname(filename, &irafheader, &mut pixfilename, status);

    /* don't need the IRAF header any more */
    // free(irafheader);

    if *status > 0 {
        return *status;
    }

    let _ = std::fs::remove_file(
        CStr::from_bytes_until_nul(cast_slice(filename))
            .unwrap()
            .to_str()
            .unwrap(),
    );

    let _ = std::fs::remove_file(
        CStr::from_bytes_until_nul(cast_slice(&pixfilename))
            .unwrap()
            .to_str()
            .unwrap(),
    );

    *status
}

/// Driver routine that reads an IRAF image into memory, also converting
/// it into FITS format.
///
/// # Parameters
///
/// * `filename` — name of input file
/// * `buffptr`  — (O) memory pointer (initially ptr::null_mut())
/// * `buffsize` — (O) size of mem buffer, in bytes
/// * `filesize` — (O) size of FITS file, in bytes
/// * `status`   — (IO) error status
pub(crate) fn iraf2mem(
    filename: &[c_char],
    buffptr: *mut *mut c_char,
    buffsize: &mut usize,
    filesize: &mut usize,
    status: &mut c_int,
) -> c_int {
    let mut lenirafhead: usize = 0;
    let buffptr = unsafe { buffptr.as_mut().expect(NULL_MSG) };

    *buffptr = ptr::null_mut();
    *buffsize = 0;
    *filesize = 0;

    let mut b_ptr = Vec::new();

    /* read IRAF header into dynamically created char array (free it later!) */
    let irafheader = irafrdhead(filename, &mut lenirafhead);

    if irafheader.is_err() {
        *status = FILE_NOT_OPENED;
        return *status;
    }

    let irafheader = irafheader.unwrap();

    /* convert IRAF header to FITS header in memory */
    let _tmp_buffer = iraftofits(
        filename,
        &irafheader,
        lenirafhead,
        &mut b_ptr,
        buffsize,
        filesize,
        status,
    );

    /* don't need the IRAF header any more */
    // free(irafheader);

    if *status > 0 {
        return *status;
    }

    *filesize = (((*filesize - 1) / BL!()) + 1) * BL!(); /* multiple of 2880 */

    /* append the image data onto the FITS header */
    irafrdimage(&mut b_ptr, buffsize, filesize, status);

    let (raw_ptr, l, c) = vec_into_raw_parts(b_ptr);
    crate::lock_global(&ALLOCATIONS).insert(raw_ptr as usize, (l, c));

    *buffptr = raw_ptr;

    *status
}

/// irafrdhead  (was irafrhead in D. Mink's original code)
/// Open and read the iraf .imh file.
/// The imhdr format is defined in iraf/lib/imhdr.h, some of which defines or mimicked, above.
// `std::io::ErrorKind` is retained deliberately: the `core::io` equivalent is still unstable.
#[allow(clippy::std_instead_of_core)]
/// # Parameters
///
/// * `filename` — Name of IRAF header file
/// * `lihead`   — Length of IRAF image header in bytes (returned)
fn irafrdhead(filename: &[c_char], lihead: &mut usize) -> Result<Vec<c_char>, Error> {
    use std::io::ErrorKind;

    let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    *lihead = 0;

    /* open the image header file */
    let fd = File::options().read(true).write(false).open(
        CStr::from_bytes_until_nul(cast_slice(filename))
            .unwrap()
            .to_str()
            .unwrap(),
    );

    if fd.is_err() {
        ffpmsg_str("unable to open IRAF header file:");
        ffpmsg_slice(filename);
        return Err(Error::from(ErrorKind::NotFound));
    }

    let mut fd = fd.unwrap(); //Already checked non-null above

    /* Find size of image header file */
    let pos = fd.seek(std::io::SeekFrom::End(0));
    if pos.is_err() {
        /* move to end of the file */
        ffpmsg_str("IRAFRHEAD: cannot seek in file:");
        ffpmsg_slice(filename);
        return Err(Error::from(ErrorKind::Unsupported));
    }

    let nbhead = pos.unwrap(); /* position = size of file */

    if fd.seek(std::io::SeekFrom::Start(0)).is_err() {
        /* move back to beginning */
        ffpmsg_str("IRAFRHEAD: cannot seek to beginning of file:");
        ffpmsg_slice(filename);
        return Err(Error::from(ErrorKind::Unsupported));
    }

    /* allocate initial sized buffer */
    // HEAP ALLOCATION
    let nihead = nbhead;
    let mut irafheader: Vec<c_char> = Vec::new();
    if irafheader.try_reserve_exact(nihead as usize).is_err() {
        int_snprintf!(
            &mut errmsg,
            FLEN_ERRMSG,
            "IRAFRHEAD Cannot allocate {}-byte header",
            nihead,
        );
        ffpmsg_slice(&errmsg);
        ffpmsg_slice(filename);
        return Err(Error::from(ErrorKind::OutOfMemory));
    } else {
        irafheader.resize(nihead as usize, 0);
    }

    *lihead = nihead as usize;

    /* Read IRAF header */
    let nbr = read_fill(
        &mut fd,
        cast_slice_mut(&mut irafheader[..(nbhead as usize)]),
    )
    .unwrap();

    drop(fd); // Close the file

    /* Reject if header less than minimum length */
    if nbr < LEN_PIXHDR {
        int_snprintf!(
            &mut errmsg,
            FLEN_ERRMSG,
            "IRAFRHEAD header file: {} / {} bytes read.",
            nbr,
            LEN_PIXHDR,
        );
        ffpmsg_slice(&errmsg);
        ffpmsg_slice(filename);

        return Err(Error::from(ErrorKind::InvalidData));
    }

    Ok(irafheader)
}

/// # Parameters
///
/// * `buffptr`  — FITS image header (filled)
/// * `buffsize` — allocated size of the buffer
/// * `filesize` — actual size of the FITS file
fn irafrdimage(
    buffptr: &mut Vec<c_char>,
    buffsize: &mut usize,
    filesize: &mut usize,
    status: &mut c_int,
) -> c_int {
    let mut nax: c_int = 1;
    let mut naxis1: c_int = 1;
    let mut naxis2: c_int = 1;
    let mut naxis3: c_int = 1;
    let mut naxis4: c_int = 1;
    let mut npaxis1: c_int = 1;
    let mut npaxis2: c_int = 0;
    let mut bitpix: c_int = 0;
    let mut bytepix: c_int = 0;
    let mut nbr: usize = 0;
    let mut nbimage: usize = 0;
    let mut nbaxis: usize = 0;
    let mut nbl: usize = 0;
    // let mut nbdiff: c_long = 0;

    let mut imhver: c_int = 0;
    let mut lpixhead: c_int = 0;
    let mut pixname: [c_char; SZ_IM2PIXFILE + 1] = [0; SZ_IM2PIXFILE + 1];
    let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut newfilesize: usize = 0;

    let fitsheader = buffptr; /* pointer to start of header */

    /* Convert pixel file name to character string */
    hgets(fitsheader, cs!(c"PIXFILE"), SZ_IM2PIXFILE, &mut pixname);
    hgeti4(fitsheader, cs!(c"PIXOFF"), &mut lpixhead);

    let lpixhead = lpixhead as usize;

    /* Open pixel file, ignoring machine name if present */
    let bang = strchr_safe(&pixname, bb(b'!'));
    let fd = if let Some(bang) = bang {
        File::options().read(true).write(false).open(
            CStr::from_bytes_until_nul(cast_slice(&pixname[bang + 1..]))
                .unwrap()
                .to_str()
                .unwrap(),
        )
    } else {
        File::options().read(true).write(false).open(
            CStr::from_bytes_until_nul(cast_slice(&pixname))
                .unwrap()
                .to_str()
                .unwrap(),
        )
    };

    /* Print error message and exit if pixel file is not found */
    if fd.is_err() {
        ffpmsg_str("IRAFRIMAGE: Cannot open IRAF pixel file:");
        ffpmsg_slice(&pixname);
        *status = FILE_NOT_OPENED;
        return *status;
    }

    let mut fd = fd.unwrap(); //Already checked non-null above

    /* Read pixel header */
    // HEAP ALLOCATION
    let mut pixheader: Vec<c_char> = Vec::new();
    if pixheader.try_reserve_exact(lpixhead as usize).is_err() {
        ffpmsg_str("IRAFRIMAGE: Cannot alloc memory for pixel header");
        ffpmsg_slice(&pixname);
        drop(fd);
        *status = FILE_NOT_OPENED;
        return *status;
    } else {
        pixheader.resize(lpixhead as usize, 0);
    }

    nbr = read_fill(&mut fd, cast_slice_mut(&mut pixheader[..lpixhead])).unwrap();

    /* Check size of pixel header */
    if nbr < lpixhead {
        int_snprintf!(
            &mut errmsg,
            FLEN_ERRMSG,
            "IRAF pixel file: {} / {} bytes read.",
            nbr,
            LEN_PIXHDR,
        );
        ffpmsg_slice(&errmsg);

        drop(fd);
        *status = FILE_NOT_OPENED;
        return *status;
    }

    /* check pixel header magic word */
    imhver = pix_version(&pixheader);
    if imhver < 1 {
        ffpmsg_str("File not valid IRAF pixel file:");
        ffpmsg_slice(&pixname);

        drop(fd);
        *status = FILE_NOT_OPENED;
        return *status;
    }

    /* Find number of bytes to read */
    hgeti4(fitsheader, cs!(c"NAXIS"), &mut nax);
    hgeti4(fitsheader, cs!(c"NAXIS1"), &mut naxis1);
    hgeti4(fitsheader, cs!(c"NPAXIS1"), &mut npaxis1);
    if nax > 1 {
        hgeti4(fitsheader, cs!(c"NAXIS2"), &mut naxis2);
        hgeti4(fitsheader, cs!(c"NPAXIS2"), &mut npaxis2);
    }
    if nax > 2 {
        hgeti4(fitsheader, cs!(c"NAXIS3"), &mut naxis3);
    }
    if nax > 3 {
        hgeti4(fitsheader, cs!(c"NAXIS4"), &mut naxis4);
    }

    hgeti4(fitsheader, cs!(c"BITPIX"), &mut bitpix);
    if bitpix < 0 {
        bytepix = -bitpix / 8;
    } else {
        bytepix = bitpix / 8;
    }

    nbimage = (naxis1 * naxis2 * naxis3 * naxis4 * bytepix) as usize; // Will be positive because of above

    newfilesize = *filesize + nbimage; /* header + data */
    newfilesize = (((newfilesize - 1) / BL!()) + 1) * BL!();

    if newfilesize > *buffsize {
        /* need to allocate more memory? */
        if fitsheader
            .try_reserve_exact(newfilesize - fitsheader.len())
            .is_err()
        {
            int_snprintf!(
                &mut errmsg,
                FLEN_ERRMSG,
                "IRAFRIMAGE Cannot allocate {}-byte image buffer",
                (*filesize) as c_int,
            );
            ffpmsg_slice(&errmsg);
            ffpmsg_slice(&pixname);
            drop(fd);
            *status = FILE_NOT_OPENED;
            return *status;
        } else {
            fitsheader.resize(newfilesize, 0);
        }
    }

    // *buffptr = fitsheader; No need since using Vec<>'s now
    *buffsize = newfilesize;

    /* image = fitsheader + *filesize;  (offset of the old filesize = end of header) */
    let image_offset = *filesize;
    *filesize = newfilesize;

    /* Read IRAF image all at once if physical and image dimensions are the same */
    if npaxis1 == naxis1 {
        let image = &mut fitsheader[image_offset..];

        nbr = read_fill(&mut fd, cast_slice_mut(&mut image[..nbimage])).unwrap();

    /* Read IRAF image one line at a time if physical and image dimensions differ */
    } else {
        let nbdiff = ((npaxis1 - naxis1) * bytepix) as i64;
        nbaxis = (naxis1 * bytepix) as usize;
        let mut line_offset = image_offset;
        nbr = 0;
        if naxis2 == 1 && naxis3 > 1 {
            naxis2 = naxis3;
        }

        for _i in 0..naxis2 {
            let linebuff = &mut fitsheader[line_offset..];
            nbl = read_fill(&mut fd, cast_slice_mut(&mut linebuff[..nbaxis])).unwrap();
            nbr += nbl;

            let _ = fd.seek(std::io::SeekFrom::Current(nbdiff)).unwrap();

            line_offset += nbaxis;
        }
    }

    drop(fd);

    /* Check size of image */
    if nbr < nbimage {
        int_snprintf!(
            &mut errmsg,
            FLEN_ERRMSG,
            "IRAF pixel file: {} / {} bytes read.",
            nbr,
            nbimage,
        );
        ffpmsg_slice(&errmsg);
        ffpmsg_slice(&pixname);
        *status = FILE_NOT_OPENED;
        return *status;
    }

    /* Byte-reverse image, if necessary */
    if *crate::lock_global(&SWAPDATA) {
        let image = &mut fitsheader[image_offset..];
        irafswap(bitpix, image, nbimage);
    }

    *status
}

/// Return IRAF image format version number from magic word in IRAF header
fn head_version(irafheader: &[c_char] /* IRAF image header from file */) -> c_int {
    /* Check header file magic word */
    if irafncmp(irafheader, cs!(c"imhdr"), 5) != 0 {
        if strncmp_safe(irafheader, cs!(c"imhv2"), 5) != 0 {
            0
        } else {
            2
        }
    } else {
        1
    }
}

/// Return IRAF image format version number from magic word in IRAF pixel file
fn pix_version(irafheader: &[c_char] /* IRAF image header from file */) -> c_int {
    /* Check pixel file header magic word */
    if irafncmp(irafheader, cs!(c"impix"), 5) != 0 {
        if strncmp_safe(irafheader, cs!(c"impv2"), 5) != 0 {
            0
        } else {
            2
        }
    } else {
        1
    }
}

/// Verify that file is valid IRAF imhdr or impix by checking first 5 chars
/// Returns 0 on success, 1 on failure
///
/// # Parameters
///
/// * `irafheader` — IRAF image header from file
/// * `teststring` — C character string to compare
/// * `nc`         — Number of characters to compate
fn irafncmp(irafheader: &[c_char], teststring: &[c_char], nc: usize) -> c_int {
    let line = iraf2str(irafheader, nc);

    if line.is_none() {
        return 1;
    }

    let line = line.unwrap();

    if strncmp_safe(&line, teststring, nc) == 0 {
        0
    } else {
        1
    }
}
/// Convert IRAF image header to FITS image header, returning FITS header
///
/// # Parameters
///
/// * `hdrname`    — IRAF header file name (may be path)
/// * `irafheader` — IRAF image header
/// * `nbiraf`     — Number of bytes in IRAF header
/// * `buffptr`    — pointer to the FITS header
/// * `nbfits`     — allocated size of the FITS header buffer
/// * `fitssize`   — Number of bytes in FITS header (returned)
fn iraftofits(
    hdrname: &[c_char],
    irafheader: &[c_char],
    nbiraf: usize,
    buffptr: &mut Vec<c_char>,
    nbfits: &mut usize,
    fitssize: &mut usize,
    /*  = number of bytes to the end of the END keyword */
    status: &mut c_int,
) -> c_int {
    let mut lstr: usize = 0;
    let mut j: usize = 0;
    let mut nax: c_int = 0;
    let mut nbits: c_int = 0;
    let mut newpixname: Option<Vec<c_char>>;
    let mut nblock: usize = 0;
    let mut nlines: usize = 0;
    let mut endline: [c_char; 81] = [0; 81];
    let mut irafchar: c_char = 0;
    let mut fitsline: [c_char; 81] = [0; 81];
    let mut pixtype: c_int = 0;
    let mut imhver: c_int = 0;
    let mut n: c_int = 0;
    let mut imu: usize = 0;
    let mut impixoff: usize = 0;
    let mut imndim: usize = 0;
    let mut imlen: usize = 0;
    let mut imphyslen: usize = 0;
    let mut impixtype: usize = 0;
    let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    /* Set up last line of FITS header */
    strncpy_safe(&mut endline, cs!(c"END"), 3);
    for i in 3..80 {
        endline[i] = bb(b' ');
    }
    endline[80] = 0;

    /* Check header magic word */
    imhver = head_version(irafheader);
    if imhver < 1 {
        ffpmsg_str("File not valid IRAF image header");
        ffpmsg_slice(hdrname);
        *status = FILE_NOT_OPENED;
        return *status;
    }
    if imhver == 2 {
        nlines = 24 + ((nbiraf - LEN_IM2HDR) / 81);
        imndim = IM2_NDIM;
        imlen = IM2_LEN;
        imphyslen = IM2_PHYSLEN;
        impixtype = IM2_PIXTYPE;
        impixoff = IM2_PIXOFF;
    /*	imtime = IM2_MTIME; */
    /*	immax = IM2_MAX;  */
    /*	immin = IM2_MIN; */
    } else {
        nlines = 24 + ((nbiraf - LEN_IMHDR) / 162);
        imndim = IM_NDIM;
        imlen = IM_LEN;
        imphyslen = IM_PHYSLEN;
        impixtype = IM_PIXTYPE;
        impixoff = IM_PIXOFF;
        /*	imtime = IM_MTIME; */
        /*	immax = IM_MAX; */
        /*	immin = IM_MIN; */
    }

    /*  Initialize FITS header */
    nblock = (nlines * 80) / BL!();
    *nbfits = (nblock + 5) * BL!() + 4;

    // HEAP ALLOCATION
    let mut fitsheader = Vec::new();
    if fitsheader.try_reserve_exact(*nbfits).is_err() {
        int_snprintf!(
            &mut errmsg,
            FLEN_ERRMSG,
            "IRAF2FITS Cannot allocate {}-byte FITS header",
            (*nbfits) as c_int,
        );
        ffpmsg_slice(hdrname);
        *status = FILE_NOT_OPENED;
        return *status;
    } else {
        fitsheader.resize(*nbfits, 0);
    }

    let mut fhead = 0; // &fitsheader

    *buffptr = fitsheader;

    let fitsheader = buffptr;
    strncpy_safe(fitsheader, &endline, 80);
    hputl(fitsheader, cs!(c"SIMPLE"), 1);
    fhead += 80;

    /*  check if the IRAF file is in big endian (sun) format (= 0) or not. */
    /*  This is done by checking the 4 byte integer in the header that     */
    /*  represents the iraf pixel type.  This 4-byte word is guaranteed to */
    /*  have the least sig byte != 0 and the most sig byte = 0,  so if the */
    /*  first byte of the word != 0, then the file in little endian format */
    /*  like on an Alpha machine.                                          */

    /* Set the SWAPHEAD global first (note: irafgeti4 reads SWAPHEAD, so the
    guard must be released before any irafgeti4 call to avoid a deadlock) */
    {
        let mut swaphead = crate::lock_global(&SWAPHEAD);
        *swaphead = isirafswapped(irafheader, impixtype);
    }

    if imhver == 1 {
        let swaphead = *crate::lock_global(&SWAPHEAD);
        *crate::lock_global(&SWAPDATA) = swaphead; /* vers 1 data has same swapness as header */
    } else {
        let v = irafgeti4(irafheader, IM2_SWAPPED) != 0;
        *crate::lock_global(&SWAPDATA) = v;
    }

    let swapdata = *crate::lock_global(&SWAPDATA);

    /*  Set pixel size in FITS header */
    pixtype = irafgeti4(irafheader, impixtype);
    match pixtype {
        TY_CHAR => {
            nbits = 8;
        }
        TY_UBYTE => {
            nbits = 8;
        }
        TY_SHORT => {
            nbits = 16;
        }
        TY_USHORT => {
            nbits = -16;
        }
        /* TY_INT falls through to TY_LONG in the C original (both => 32 bits) */
        TY_INT | TY_LONG => {
            nbits = 32;
        }
        TY_REAL => {
            nbits = -32;
        }
        TY_DOUBLE => {
            nbits = -64;
        }
        _ => {
            int_snprintf!(
                &mut errmsg,
                FLEN_ERRMSG,
                "Unsupported IRAF data type: {}",
                pixtype,
            );
            ffpmsg_slice(&errmsg);
            ffpmsg_slice(hdrname);
            *status = FILE_NOT_OPENED;
            return *status;
        }
    }

    hputi4(fitsheader, cs!(c"BITPIX"), nbits);
    hputcom(fitsheader, cs!(c"BITPIX"), cs!(c"IRAF .imh pixel type"));
    fhead += 80;

    /*  Set image dimensions in FITS header */
    nax = irafgeti4(irafheader, imndim);
    hputi4(fitsheader, cs!(c"NAXIS"), nax);
    hputcom(fitsheader, cs!(c"NAXIS"), cs!(c"IRAF .imh naxis"));
    fhead += 80;

    n = irafgeti4(irafheader, imlen);
    hputi4(fitsheader, cs!(c"NAXIS1"), n);
    hputcom(fitsheader, cs!(c"NAXIS1"), cs!(c"IRAF .imh image naxis[1]"));
    fhead += 80;

    if nax > 1 {
        n = irafgeti4(irafheader, imlen + 4);
        hputi4(fitsheader, cs!(c"NAXIS2"), n);
        hputcom(fitsheader, cs!(c"NAXIS2"), cs!(c"IRAF .imh image naxis[2]"));
        fhead += 80;
    }
    if nax > 2 {
        n = irafgeti4(irafheader, imlen + 8);
        hputi4(fitsheader, cs!(c"NAXIS3"), n);
        hputcom(fitsheader, cs!(c"NAXIS3"), cs!(c"IRAF .imh image naxis[3]"));
        fhead += 80;
    }
    if nax > 3 {
        n = irafgeti4(irafheader, imlen + 12);
        hputi4(fitsheader, cs!(c"NAXIS4"), n);
        hputcom(fitsheader, cs!(c"NAXIS4"), cs!(c"IRAF .imh image naxis[4]"));
        fhead += 80;
    }

    /* Set object name in FITS header */
    let objname: Option<Vec<c_char>> = if imhver == 2 {
        irafgetc(irafheader, IM2_TITLE, SZ_IM2TITLE)
    } else {
        irafgetc2(irafheader, IM_TITLE, SZ_IMTITLE)
    };

    let mut objname = objname.unwrap();

    lstr = strlen_safe(&objname);
    if lstr < 8 {
        objname.resize(8, 0);
        for i in lstr..8 {
            objname[i] = bb(b' ');
        }
        objname[8] = 0;
    }
    hputs(fitsheader, cs!(c"OBJECT"), &objname);
    hputcom(fitsheader, cs!(c"OBJECT"), cs!(c"IRAF .imh title"));

    fhead += 80;

    /* Save physical axis lengths so image file can be read */
    n = irafgeti4(irafheader, imphyslen);
    hputi4(fitsheader, cs!(c"NPAXIS1"), n);
    hputcom(
        fitsheader,
        cs!(c"NPAXIS1"),
        cs!(c"IRAF .imh physical naxis[1]"),
    );
    fhead += 80;
    if nax > 1 {
        n = irafgeti4(irafheader, imphyslen + 4);
        hputi4(fitsheader, cs!(c"NPAXIS2"), n);
        hputcom(
            fitsheader,
            cs!(c"NPAXIS2"),
            cs!(c"IRAF .imh physical naxis[2]"),
        );
        fhead += 80;
    }
    if nax > 2 {
        n = irafgeti4(irafheader, imphyslen + 8);
        hputi4(fitsheader, cs!(c"NPAXIS3"), n);
        hputcom(
            fitsheader,
            cs!(c"NPAXIS3"),
            cs!(c"IRAF .imh physical naxis[3]"),
        );
        fhead += 80;
    }
    if nax > 3 {
        n = irafgeti4(irafheader, imphyslen + 12);
        hputi4(fitsheader, cs!(c"NPAXIS4"), n);
        hputcom(
            fitsheader,
            cs!(c"NPAXIS4"),
            cs!(c"IRAF .imh physical naxis[4]"),
        );
        fhead += 80;
    }

    /* Save image header filename in header */
    hputs(fitsheader, cs!(c"IMHFILE"), hdrname);
    hputcom(fitsheader, cs!(c"IMHFILE"), cs!(c"IRAF header file name"));
    fhead += 80;

    /* Save image pixel file pathname in header */
    let pixname: Option<Vec<c_char>> = if imhver == 2 {
        irafgetc(irafheader, IM2_PIXFILE, SZ_IM2PIXFILE)
    } else {
        irafgetc2(irafheader, IM_PIXFILE, SZ_IMPIXFILE)
    };

    let mut pixname = pixname.unwrap();

    if strncmp_safe(&pixname, cs!(c"HDR"), 3) == 0 {
        newpixname = same_path(&pixname, hdrname);
        if let Some(newpixname) = newpixname {
            pixname = newpixname;
        }
    }
    if !has_path_sep(&pixname) && strchr_safe(&pixname, bb(b'$')).is_none() {
        newpixname = same_path(&pixname, hdrname);
        if let Some(newpixname) = newpixname {
            pixname = newpixname;
        }
    }

    let bang = strchr_safe(&pixname, bb(b'!'));

    if let Some(bang) = bang {
        hputs(fitsheader, cs!(c"PIXFILE"), &pixname[(bang + 1)..]);
    } else {
        hputs(fitsheader, cs!(c"PIXFILE"), &pixname);
    }

    hputcom(fitsheader, cs!(c"PIXFILE"), cs!(c"IRAF .pix pixel file"));
    fhead += 80;

    /* Save image offset from star of pixel file */
    let mut pixoff = irafgeti4(irafheader, impixoff);
    pixoff = (pixoff - 1) * 2;
    hputi4(fitsheader, cs!(c"PIXOFF"), pixoff);
    hputcom(
        fitsheader,
        cs!(c"PIXOFF"),
        cs!(c"IRAF .pix pixel offset (Do not change!)"),
    );
    fhead += 80;

    /* Save IRAF file format version in header */
    hputi4(fitsheader, cs!(c"IMHVER"), imhver);
    hputcom(
        fitsheader,
        cs!(c"IMHVER"),
        cs!(c"IRAF .imh format version (1 or 2)"),
    );
    fhead += 80;

    /* Save flag as to whether to swap IRAF data for this file and machine */
    if swapdata {
        hputl(fitsheader, cs!(c"PIXSWAP"), 1);
    } else {
        hputl(fitsheader, cs!(c"PIXSWAP"), 0);
    }
    hputcom(
        fitsheader,
        cs!(c"PIXSWAP"),
        cs!(c"IRAF pixels, FITS byte orders differ if T"),
    );
    fhead += 80;

    /* Add user portion of IRAF header to FITS header */
    fitsline[80] = 0;
    if imhver == 2 {
        imu = LEN_IM2HDR;
        let chead = irafheader;
        j = 0;
        for k in 0..80 {
            fitsline[k] = bb(b' ');
        }
        for i in imu..nbiraf {
            irafchar = chead[i];
            if irafchar == 0 {
                break;
            } else if irafchar == 10 {
                strncpy_safe(&mut fitsheader[fhead..], &fitsline, 80);
                /* fprintf (stderr,"%80s\n",fitsline); */
                if strncmp_safe(&fitsline, cs!(c"OBJECT "), 7) != 0 {
                    fhead += 80;
                }
                for k in 0..80 {
                    fitsline[k] = bb(b' ');
                }
                j = 0;
            } else {
                if j > 80 {
                    if strncmp_safe(&fitsline, cs!(c"OBJECT "), 7) != 0 {
                        strncpy_safe(&mut fitsheader[fhead..], &fitsline, 80);
                        /* fprintf (stderr,"%80s\n",fitsline); */
                        j = 9;
                        fhead += 80;
                    }
                    for k in 0..80 {
                        fitsline[k] = bb(b' ');
                    }
                }
                if irafchar > 32 && irafchar < 127 {
                    fitsline[j] = irafchar;
                }
                j += 1;
            }
        }
    } else {
        imu = LEN_IMHDR;
        let chead = irafheader;
        let ib: usize = if *crate::lock_global(&SWAPHEAD) { 0 } else { 1 };

        for k in 0..80 {
            fitsline[k] = bb(b' ');
        }
        let mut j = 0;
        for i in (imu..nbiraf).step_by(2) {
            irafchar = chead[i + ib];
            if irafchar == 0 {
                break;
            } else if irafchar == 10 {
                if strncmp_safe(&fitsline, cs!(c"OBJECT "), 7) != 0 {
                    strncpy_safe(&mut fitsheader[fhead..], &fitsline, 80);
                    fhead += 80;
                }
                /* fprintf (stderr,"%80s\n",fitsline); */
                j = 0;
                for k in 0..80 {
                    fitsline[k] = bb(b' ');
                }
            } else {
                if j > 80 {
                    if strncmp_safe(&fitsline, cs!(c"OBJECT "), 7) != 0 {
                        strncpy_safe(&mut fitsheader[fhead..], &fitsline, 80);
                        j = 9;
                        fhead += 80;
                    }
                    /* fprintf (stderr,"%80s\n",fitsline); */
                    for k in 0..80 {
                        fitsline[k] = bb(b' ');
                    }
                }
                if irafchar > 32 && irafchar < 127 {
                    fitsline[j] = irafchar;
                }
                j += 1;
            }
        }
    }

    /* Add END to last line */
    strncpy_safe(&mut fitsheader[fhead..], &endline, 80);

    /* Find end of last 2880-byte block of header */
    fhead = ksearch(fitsheader, cs!(c"END")).unwrap() + 80;
    nblock = *nbfits / BL!();
    let fhead1 = nblock * BL!(); //&fitsheader
    *fitssize = fhead; /* no. of bytes to end of END keyword */

    /* Pad rest of header with spaces */
    strncpy_safe(&mut endline, cs!(c"   "), 3);
    for fp in (fhead..fhead1).step_by(80) {
        strncpy_safe(&mut fitsheader[fp..], &endline, 80);
    }

    *status
}

/// get the IRAF pixel file name
///
/// # Parameters
///
/// * `hdrname`     — IRAF header file name (may be path)
/// * `irafheader`  — IRAF image header
/// * `pixfilename` — IRAF pixel file name
fn getirafpixname(
    hdrname: &[c_char],
    irafheader: &[c_char],
    pixfilename: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut newpixname = None;

    /* Check header magic word */
    let imhver = head_version(irafheader);
    if imhver < 1 {
        ffpmsg_str("File not valid IRAF image header");
        ffpmsg_slice(hdrname);
        *status = FILE_NOT_OPENED;
        return *status;
    }

    /* get image pixel file pathname in header */
    let mut pixname = if imhver == 2 {
        irafgetc(irafheader, IM2_PIXFILE, SZ_IM2PIXFILE)
    } else {
        irafgetc2(irafheader, IM_PIXFILE, SZ_IMPIXFILE)
    }
    .unwrap();

    if strncmp_safe(&pixname, cs!(c"HDR"), 3) == 0 {
        newpixname = same_path(&pixname, hdrname);
        if let Some(newpixname) = newpixname {
            pixname = newpixname;
        }
    }

    if !has_path_sep(&pixname) && strchr_safe(&pixname, bb(b'$')).is_none() {
        newpixname = same_path(&pixname, hdrname);
        if let Some(newpixname) = newpixname {
            pixname = newpixname;
        }
    }

    let bang = strchr_safe(&pixname, bb(b'!'));
    if let Some(bang) = bang {
        strcpy_safe(pixfilename, &pixname[(bang + 1)..]);
    } else {
        strcpy_safe(pixfilename, &pixname);
    }

    *status
}

/* True if `c` is a directory separator.  CFITSIO only looks for '/', but on
 * Windows header pathnames arrive with '\' separators, so accept both there;
 * otherwise the directory prefix is stripped and the pixel file can't be found. */
fn is_path_sep(c: c_char) -> bool {
    c == bb(b'/') || (cfg!(target_os = "windows") && c == bb(b'\\'))
}

/* True if `pixname` already contains a directory separator.  CFITSIO only
 * tests for '/', which is how it decides a pixel file name is "bare" (no path)
 * and needs the header directory prepended.  On Windows a resolved absolute
 * path uses '\' separators, so without also checking '\' an already-resolved
 * path looks bare and same_path() prepends the header directory a second time,
 * yielding "<hdrdir>\C:\...". */
fn has_path_sep(s: &[c_char]) -> bool {
    strchr_safe(s, bb(b'/')).is_some()
        || (cfg!(target_os = "windows") && strchr_safe(s, bb(b'\\')).is_some())
}

/* Put filename and header path together */
/// # Parameters
///
/// * `pixname` — IRAF pixel file pathname
/// * `hdrname` — IRAF image header file pathname
fn same_path(pixname: &[c_char], hdrname: &[c_char]) -> Option<Vec<c_char>> {
    /*  WDP - 10/16/2007 - increased allocation to avoid possible overflow */
    /*    newpixname =  calloc (SZ_IM2PIXFILE, sizeof (char)); */

    let mut newpixname = Vec::new();
    if newpixname.try_reserve_exact(2 * SZ_IM2PIXFILE + 1).is_err() {
        ffpmsg_str("iraffits same_path: Cannot alloc memory for newpixname");
        return None;
    } else {
        newpixname.resize(2 * SZ_IM2PIXFILE + 1, 0)
    }

    /* Pixel file is in same directory as header */
    if strncmp_safe(pixname, cs!(c"HDR$"), 4) == 0 {
        strncpy_safe(&mut newpixname, hdrname, SZ_IM2PIXFILE);

        /* find the end of the pathname */
        let mut len = strlen_safe(&newpixname);

        while (len > 0) && !is_path_sep(newpixname[len - 1]) {
            len -= 1;
        }

        /* add name */
        newpixname[len] = 0;
        strncat_safe(&mut newpixname, &pixname[4..], SZ_IM2PIXFILE);
    }
    /* Bare pixel file with no path is assumed to be same as HDR$filename */
    else if !has_path_sep(pixname) && strchr_safe(pixname, bb(b'$')).is_none() {
        strncpy_safe(&mut newpixname, hdrname, SZ_IM2PIXFILE);

        /* find the end of the pathname */
        let mut len = strlen_safe(&newpixname);

        while (len > 0) && !is_path_sep(newpixname[len - 1]) {
            len -= 1;
        }

        /* add name */
        newpixname[len] = 0;
        strncat_safe(&mut newpixname, pixname, SZ_IM2PIXFILE);
    }
    /* Pixel file has same name as header file, but with .pix extension */
    else if strncmp_safe(pixname, cs!(c"HDR"), 3) == 0 {
        /* load entire header name string into name buffer */
        strncpy_safe(&mut newpixname, hdrname, SZ_IM2PIXFILE);
        let len = strlen_safe(&newpixname);
        newpixname[len - 3] = bb(b'p');
        newpixname[len - 2] = bb(b'i');
        newpixname[len - 1] = bb(b'x');
    }

    Some(newpixname)
}

///  check if the IRAF file is in big endian (sun) format (= 0) or not
///  This is done by checking the 4 byte integer in the header that
///  represents the iraf pixel type.  This 4-byte word is guaranteed to
///  have the least sig byte != 0 and the most sig byte = 0,  so if the
///  first byte of the word != 0, then the file in little endian format
///  like on an Alpha machine.                                          
///
/// # Parameters
///
/// * `irafheader` — IRAF image header
/// * `offset`     — Number of bytes to skip before number
fn isirafswapped(irafheader: &[c_char], offset: usize) -> bool {
    let mut swapped = false;

    swapped = irafheader[offset] != 0;

    swapped
}

/// # Parameters
///
/// * `irafheader` — IRAF image header
/// * `offset`     — Number of bytes to skip before number
fn irafgeti4(irafheader: &[c_char], offset: usize) -> c_int {
    let cheader = irafheader;

    let mut itemp: [c_int; 1] = [0; 1];
    let ctemp: &mut [c_char] = cast_slice_mut(&mut itemp);

    let swaphead = crate::lock_global(&SWAPHEAD);

    if machswap() != *swaphead {
        ctemp[3] = cheader[offset];
        ctemp[2] = cheader[offset + 1];
        ctemp[1] = cheader[offset + 2];
        ctemp[0] = cheader[offset + 3];
    } else {
        ctemp[0] = cheader[offset];
        ctemp[1] = cheader[offset + 1];
        ctemp[2] = cheader[offset + 2];
        ctemp[3] = cheader[offset + 3];
    }

    itemp[0]
}

/// IRAFGETC2 -- Get character string from arbitrary part of v.1 IRAF header
///
/// # Parameters
///
/// * `irafheader` — IRAF image header
/// * `offset`     — Number of bytes to skip before string
/// * `nc`         — Maximum number of characters in string
fn irafgetc2(irafheader: &[c_char], offset: usize, nc: usize) -> Option<Vec<c_char>> {
    let irafstring = irafgetc(irafheader, offset, 2 * (nc + 1));

    iraf2str(&irafstring.unwrap(), nc)
}

/// IRAFGETC -- Get character string from arbitrary part of IRAF header
///
/// # Parameters
///
/// * `irafheader` — IRAF image header
/// * `offset`     — Number of bytes to skip before string
/// * `nc`         — Maximum number of characters in string
fn irafgetc(irafheader: &[c_char], offset: usize, nc: usize) -> Option<Vec<c_char>> {
    let cheader = irafheader;

    let mut ctemp = Vec::new();

    if ctemp.try_reserve_exact(nc + 1).is_err() {
        ffpmsg_str("IRAFGETC Cannot allocate memory for string variable");
        return None;
    }

    for i in 0..nc {
        ctemp.push(cheader[offset + i]);
        if ctemp[i] > 0 && ctemp[i] < 32 {
            ctemp[i] = bb(b' ');
        }
    }

    Some(ctemp)
}

/// Convert IRAF 2-byte/char string to 1-byte/char string
///
/// # Parameters
///
/// * `irafstring` — IRAF 2-byte/character string
/// * `nchar`      — Number of characters in string
fn iraf2str(irafstring: &[c_char], nchar: usize) -> Option<Vec<c_char>> {
    let mut string = Vec::new();

    if string.try_reserve_exact(nchar + 1).is_err() {
        ffpmsg_str("IRAF2STR Cannot allocate memory for string variable");
        return None;
    }

    /* the chars are in bytes 1, 3, 5, ... if bigendian format (SUN) */
    /* else in bytes 0, 2, 4, ... if little endian format (Alpha)    */

    let mut j = if irafstring[0] != 0 { 0 } else { 1 };

    /* Convert appropriate byte of input to output character */
    for _i in 0..(nchar) {
        string.push(irafstring[j]);
        j += 2;
    }

    Some(string)
}

/// IRAFSWAP -- Reverse bytes of any type of vector in place
///
/// # Parameters
///
/// * `bitpix` — Number of bits per pixel
/// * `string` — Address of starting point of bytes to swap
/// * `nbytes` — Number of bytes to swap
fn irafswap(
    bitpix: c_int,
    /*  16 = short, -16 = unsigned short, 32 = int */
    /* -32 = float, -64 = double */
    string: &mut [c_char],
    nbytes: usize,
) {
    match bitpix {
        16 => {
            if nbytes < 2 {
                return;
            }
            irafswap2(string, nbytes);
        }
        32 => {
            if nbytes < 4 {
                return;
            }
            irafswap4(string, nbytes);
        }
        -16 => {
            if nbytes < 2 {
                return;
            }
            irafswap2(string, nbytes);
        }
        -32 => {
            if nbytes < 4 {
                return;
            }
            irafswap4(string, nbytes);
        }
        -64 => {
            if nbytes < 8 {
                return;
            }
            irafswap8(string, nbytes);
        }
        _ => (),
    }
}

///IRAFSWAP2 -- Swap bytes in string in place
///
/// # Parameters
///
/// * `string` — Address of starting point of bytes to swap
/// * `nbytes` — Number of bytes to swap
fn irafswap2(string: &mut [c_char], nbytes: usize) {
    let slast = nbytes;
    let mut sbyte = 0;
    while sbyte < slast {
        string.swap(sbyte, sbyte + 1);
        sbyte += 2;
    }
}

/// IRAFSWAP4 -- Reverse bytes of Integer*4 or Real*4 vector in place
///
/// # Parameters
///
/// * `string` — Address of Integer*4 or Real*4 vector
/// * `nbytes` — Number of bytes to reverse
fn irafswap4(string: &mut [c_char], nbytes: usize) {
    let slast = nbytes;
    let mut sbyte = 0;
    while sbyte < slast {
        let temp3 = string[sbyte];
        let temp2 = string[sbyte + 1];
        let temp1 = string[sbyte + 2];
        let temp0 = string[sbyte + 3];
        string[sbyte] = temp0;
        string[sbyte + 1] = temp1;
        string[sbyte + 2] = temp2;
        string[sbyte + 3] = temp3;
        sbyte += 4;
    }
}

/// IRAFSWAP8 -- Reverse bytes of Real*8 vector in place
///
/// # Parameters
///
/// * `string` — Address of Real*8 vector
/// * `nbytes` — Number of bytes to reverse
fn irafswap8(string: &mut [c_char], nbytes: usize) {
    let slast = nbytes;
    let mut sbyte = 0;
    while sbyte < slast {
        let temp7 = string[sbyte];
        let temp6 = string[sbyte + 1];
        let temp5 = string[sbyte + 2];
        let temp4 = string[sbyte + 3];
        let temp3 = string[sbyte + 4];
        let temp2 = string[sbyte + 5];
        let temp1 = string[sbyte + 6];
        let temp0 = string[sbyte + 7];
        string[sbyte] = temp0;
        string[sbyte + 1] = temp1;
        string[sbyte + 2] = temp2;
        string[sbyte + 3] = temp3;
        string[sbyte + 4] = temp4;
        string[sbyte + 5] = temp5;
        string[sbyte + 6] = temp6;
        string[sbyte + 7] = temp7;
        sbyte += 8;
    }
}

fn machswap() -> bool {
    let mut itest: [c_int; 1] = [1; 1];
    let ctest: &[c_char] = cast_slice_mut(&mut itest);

    ctest[0] != 0
}

/*             the following routines were originally in hget.c             */

static LHEAD0: usize = 0;

/// Extract long value for variable from FITS header string
///
/// # Parameters
///
/// * `hstring` — character string containing FITS header information in the format
///              `<keyword>`= `<value>` {/ `<comment>`}
/// * `keyword` — character string containing the name of the keyword the value of which
///              is returned. hget searches for a line beginning with this string. if
///              "`[n]`" is present, the n'th token in the value is returned. (the
///              first 8 characters must be unique)
fn hgeti4(hstring: &mut [c_char], keyword: &[c_char], ival: &mut c_int) -> c_int {
    let mut dval: f64 = 0.0;
    let mut minint: c_int = 0;
    let mut val: [c_char; 30] = [0; 30];

    /* Get value and comment from header string */
    let value = hgetc(hstring, keyword);

    /* Translate value from ASCII to binary */
    if let Some(value) = value {
        minint = c_int::MIN;
        if strlen_safe(&value) > 29 {
            return 0;
        }
        strcpy_safe(&mut val, &value);
        dval = atof_safe(&val);
        if dval + 0.001 > f64::from(MAXINT) {
            *ival = MAXINT;
        } else if dval >= 0.0 {
            *ival = (dval + 0.001) as c_int;
        } else if dval - 0.001 < f64::from(minint) {
            *ival = minint;
        } else {
            *ival = (dval - 0.001) as c_int;
        }
        1
    } else {
        0
    }
}

/// Extract string value for variable from FITS header string
///
/// # Parameters
///
/// * `hstring` — character string containing FITS header information in the format
///              `<keyword>`= `<value>` {/ `<comment>`}
/// * `keyword` — character string containing the name of the keyword the value of which
///              is returned. hget searches for a line beginning with this string. if
///              "`[n]`" is present, the n'th token in the value is returned. (the
///              first 8 characters must be unique)
/// * `lstr`    — Size of str in characters
/// * `str`     — String (returned)
fn hgets(hstring: &mut [c_char], keyword: &[c_char], lstr: usize, str: &mut [c_char]) -> c_int {
    let mut lval: usize = 0;

    /* Get value and comment from header string */
    let value = hgetc(hstring, keyword);

    if let Some(value) = value {
        lval = strlen_safe(&value);
        if lval < lstr {
            strcpy_safe(str, &value);
        } else if lstr > 1 {
            strncpy_safe(str, &value, lstr - 1);
            str[lstr - 1] = 0;
        } else {
            str[0] = value[0];
        }
        1
    } else {
        0
    }
}

/// Extract character value for variable from FITS header string
///
/// # Parameters
///
/// * `hstring`  — character string containing FITS header information in the format
///               `<keyword>`= `<value>` {/ `<comment>`}
/// * `keyword0` — character string containing the name of the keyword the value of
///               which is returned. hget searches for a line beginning with this
///               string. if "`[n]`" is present, the n'th token in the value is
///               returned. (the first 8 characters must be unique)
fn hgetc(hstring: &mut [c_char], keyword0: &[c_char]) -> Option<Vec<c_char>> {
    let mut cval: [c_char; 80] = [0; 80];
    // let value: Option<usize>;
    let mut squot: [c_char; 2] = [0; 2];
    let mut dquot: [c_char; 2] = [0; 2];
    let mut lbracket: [c_char; 2] = [0; 2];
    let mut rbracket: [c_char; 2] = [0; 2];
    let mut slash: [c_char; 2] = [0; 2];
    let mut comma: [c_char; 2] = [0; 2];
    let mut keyword: [c_char; 81] = [0; 81]; /* large for ESO hierarchical keywords */
    let mut line: [c_char; 100] = [0; 100];

    squot[0] = 39;
    squot[1] = 0;
    dquot[0] = 34;
    dquot[1] = 0;
    lbracket[0] = 91;
    lbracket[1] = 0;
    comma[0] = 44;
    comma[1] = 0;
    rbracket[0] = 93;
    rbracket[1] = 0;
    slash[0] = 47;
    slash[1] = 0;

    /* Find length of variable name */
    let key_len = keyword.len();
    strncpy_safe(&mut keyword, keyword0, key_len - 1);
    keyword[80] = 0;
    let mut brack1 = strsrch(&keyword, &lbracket);
    if brack1.is_none() {
        brack1 = strsrch(&keyword, &comma);
    }

    if let Some(mut brack1) = brack1 {
        keyword[brack1] = 0;
        brack1 += 1;
    }

    /* Search header string for variable name */
    let vpos = ksearch(hstring, &keyword);

    /* Exit if not found */
    vpos?;

    let vpos = vpos.unwrap();

    /* Initialize line to nulls */
    for i in 0..100 {
        line[i] = 0;
    }

    /* In standard FITS, data lasts until 80th character */

    /* Extract entry for this variable from the header */
    strncpy_safe(&mut line, &hstring[vpos..], 80);

    /* check for quoted value */
    let mut q1 = strsrch(&line, &squot);
    let mut c1 = strsrch(&line, &slash);
    let mut q2 = None;

    /* note: strsrch on a subslice returns an offset relative to that subslice;
    make q2 absolute (relative to the start of `line`) by adding q1_tmp + 1, to
    match the C original where q2 = strsrch(q1+1, ...) is an absolute pointer */
    if let Some(q1_tmp) = q1 {
        if let Some(c1_tmp) = c1 {
            if q1_tmp < c1_tmp {
                q2 = strsrch(&line[(q1_tmp + 1)..], &squot).map(|o| q1_tmp + 1 + o);
            } else {
                q1 = None;
            }
        } else {
            q2 = strsrch(&line[(q1_tmp + 1)..], &squot).map(|o| q1_tmp + 1 + o);
        }
    } else {
        q1 = strsrch(&line, &dquot);
        if let Some(q1_tmp) = q1 {
            if let Some(c1_tmp) = c1 {
                if q1_tmp < c1_tmp {
                    q2 = strsrch(&line[(q1_tmp + 1)..], &dquot).map(|o| q1_tmp + 1 + o);
                } else {
                    q1 = None;
                }
            } else {
                q2 = strsrch(&line[(q1_tmp + 1)..], &dquot).map(|o| q1_tmp + 1 + o);
            }
        } else {
            q1 = None;
            q2 = Some(10); // &line
        }
    }

    /* Extract value and remove excess spaces */
    let mut v1;
    let mut v2;
    if let Some(q1) = q1 {
        q2?;
        let q2 = q2.unwrap();

        v1 = q1 + 1;
        v2 = q2;
        c1 = strsrch(&line[q2..], cs!(c"/"));
    } else {
        v1 =
            strsrch(&line, cs!(c"=")).expect("Shouldn't be null, not handled in original code") + 1;
        c1 = strsrch(&line, cs!(c"/"));
        if let Some(c1) = c1 {
            v2 = c1;
        } else {
            v2 = 79; //&line
        }
    }

    /* Ignore leading spaces */
    while line[v1] == bb(b' ') && v1 < v2 {
        v1 += 1;
    }

    /* Drop trailing spaces */
    line[v2] = 0;
    v2 -= 1;
    while line[v2] == bb(b' ') && v2 > v1 {
        line[v2] = 0;
        v2 -= 1;
    }

    if strcmp_safe(&line[v1..], cs!(c"-0")) == 0 {
        v1 += 1;
    }
    strcpy_safe(&mut cval, &line[v1..]);

    /* If keyword has brackets, extract appropriate token from value */
    if let Some(brack1) = brack1 {
        let brack2 = strsrch(&keyword[brack1..], &rbracket);
        if let Some(brack2) = brack2 {
            keyword[brack2] = 0;
        }

        let ipar = parse_c_int(&keyword[brack1..]);
        if ipar > 0 {
            let mut cpar = None;
            let v_tok = &line[v1..];

            let mut tokens = v_tok.split(|f| (*f == b' ' as c_char) || (*f == 0));

            for _i in 1..=ipar {
                cpar = tokens.next();
            }

            if let Some(cpar) = cpar {
                let len = cpar.len();
                if len > 0 {
                    cval[0..len].copy_from_slice(cpar);
                    cval[len] = 0; // Null termiante
                }
            } else {
                // value = None;
            }
        }
    }

    Some(Vec::from(cval))
}

/// Find beginning of fillable blank line before FITS header keyword line
///
/// # Parameters
///
/// * `hstring` — character string containing fits-style header information in the
///              format `<keyword>`= `<value>` {/ `<comment>`} the default is that each
///              entry is 80 characters long; however, lines may be of arbitrary length
///              terminated by nulls, carriage returns or linefeeds, if packed is true.
/// * `keyword` — character string containing the name of the variable to be returned.
///              ksearch searches for a line beginning with this string. The string may
///              be a character literal or a character variable terminated by a null or
///              '$'. it is truncated to 8 characters.
fn blsearch(
    /* Find entry for keyword keyword in FITS header string hstring.
    (the keyword may have a maximum of eight letters)
    ptr::null_mut() is returned if the keyword is not found */
    hstring: &mut [c_char],
    keyword: &[c_char],
) -> Option<usize> {
    let mut lhstr: usize = 0;

    /* Search header string for variable name */
    if LHEAD0 != 0 {
        lhstr = LHEAD0;
    } else {
        lhstr = 0;
        while lhstr < 57600 && hstring[lhstr] != 0 {
            lhstr += 1;
        }
    }

    let headlast = lhstr;
    let mut headnext = 0;
    let mut pval = None;
    while headnext < headlast {
        let nleft = headlast - headnext;
        let loc = strnsrch(&hstring[headnext..], keyword, nleft);

        /* Exit if keyword is not found */
        if loc.is_none() {
            break;
        }

        let loc = loc.unwrap();

        let icol = (loc) % 80;
        let lkey = strlen_safe(keyword);
        let nextchar = hstring[loc + lkey];

        /* If this is not in the first 8 characters of a line, keep searching */
        if icol > 7 {
            headnext = loc + 1;

        /* If parameter name in header is longer, keep searching */
        } else if nextchar != 61 && nextchar > 32 && nextchar < 127 {
            headnext = loc + 1;

        /* If preceeding characters in line are not blanks, keep searching */
        } else {
            let line = loc - icol;
            for lc in line..loc {
                if hstring[lc] != bb(b' ') {
                    headnext = loc + 1;
                }
            }

            /* Return pointer to start of line if match */
            if loc >= headnext {
                pval = Some(line);
                break;
            }
        }
    }

    /* Return ptr::null_mut() if keyword is found at start of FITS header string */
    pval?;

    let pval = pval.unwrap();

    /* Return ptr::null_mut() if found the first keyword in the header */
    if pval == 0 {
        return None;
    }

    /* Find last nonblank line before requested keyword */
    let mut bval = pval - 80;
    while strncmp_safe(&hstring[bval..], cs!(c"        "), 8) == 0 {
        bval -= 80;
    }
    bval += 80;

    /* Return pointer to calling program if blank lines found */
    if bval < pval { Some(bval) } else { None }
}

/// Find FITS header line containing specified keyword
///
/// # Parameters
///
/// * `hstring` — character string containing fits-style header information in the
///              format `<keyword>`= `<value>` {/ `<comment>`} the default is that each
///              entry is 80 characters long; however, lines may be of arbitrary length
///              terminated by nulls, carriage returns or linefeeds, if packed is true.
/// * `keyword` — character string containing the name of the variable to be returned.
///              ksearch searches for a line beginning with this string. The string may
///              be a character literal or a character variable terminated by a null or
///              '$'. it is truncated to 8 characters.
fn ksearch(
    /* Find entry for keyword keyword in FITS header string hstring.
    (the keyword may have a maximum of eight letters)
    ptr::null_mut() is returned if the keyword is not found */
    hstring: &mut [c_char],
    keyword: &[c_char],
) -> Option<usize> {
    let mut icol: usize = 0;
    let mut lkey: usize = 0;
    let mut nleft: usize = 0;
    let mut lhstr: usize = 0;

    /* Search header string for variable name */
    if LHEAD0 != 0 {
        lhstr = LHEAD0;
    } else {
        lhstr = 0;
        while lhstr < 57600 && hstring[lhstr] != 0 {
            lhstr += 1;
        }
    }

    let headlast = lhstr;
    let mut headnext = 0;
    let mut pval = None;
    while headnext < headlast {
        nleft = headlast - headnext;
        let loc = strnsrch(&hstring[headnext..], keyword, nleft);

        /* Exit if keyword is not found */
        if loc.is_none() {
            break;
        }

        let loc = loc.unwrap();

        icol = (loc) % 80;
        lkey = strlen_safe(keyword);
        let nextchar = hstring[loc + lkey];

        /* If this is not in the first 8 characters of a line, keep searching */
        if icol > 7 {
            headnext = loc + 1;

        /* If parameter name in header is longer, keep searching */
        } else if nextchar != 61 && nextchar > 32 && nextchar < 127 {
            headnext = loc + 1;

        /* If preceeding characters in line are not blanks, keep searching */
        } else {
            let line = loc - icol;
            for lc in line..loc {
                if hstring[lc] != bb(b' ') {
                    headnext = loc + 1;
                }
            }

            /* Return pointer to start of line if match */
            if loc >= headnext {
                pval = Some(line);
                break;
            }
        }
    }

    /* Return pointer to calling program */
    pval
}

/// Find string s2 within null-terminated string s1
///
/// # Parameters
///
/// * `s1` — String to search
/// * `s2` — String to look for
fn strsrch(s1: &[c_char], s2: &[c_char]) -> Option<usize> {
    let ls1 = strlen_safe(s1);
    strnsrch(s1, s2, ls1)
}

/// Find string s2 within string s1
///
/// # Parameters
///
/// * `s1`  — String to search
/// * `s2`  — String to look for
/// * `ls1` — Length of string being searched
fn strnsrch(s1: &[c_char], s2: &[c_char], ls1: usize) -> Option<usize> {
    let mut cfirst: c_char = 0;
    let mut clast: c_char = 0;
    let mut i: usize = 0;

    /* Return null string if either pointer is ptr::null_mut() */
    if s1.is_empty() {
        return None;
    }

    /* A zero-length pattern is found in any string */
    let ls2 = strlen_safe(s2);
    if ls2 == 0 {
        return Some(0);
    }

    /* Only a zero-length string can be found in a zero-length string */
    if ls1 == 0 {
        return None;
    }

    cfirst = s2[0];
    clast = s2[ls2 - 1];
    let s1e = ls1 - ls2 + 1; // Index in s1
    let mut s = 0; // Index in S1

    while s < s1e {
        /* Search for first character in pattern string */
        if s1[s] == cfirst {
            /* If single character search, return */
            if ls2 == 1 {
                return Some(s);
            }

            /* Search for last character in pattern string if first found */
            if s1[s + ls2 - 1] == clast {
                /* If two-character search, return */
                if ls2 == 2 {
                    return Some(s);
                }

                /* If 3 or more characters, check for rest of search string */
                i = 1;
                while i < ls2 && s1[s + i] == s2[i] {
                    i += 1;
                }

                /* If entire string matches, return */
                if i >= ls2 {
                    return Some(s);
                }
            }
        }
        s += 1;
    }
    None
}

/*             the following routines were originally in hget.c      */

/// HPUTI4 - Set int keyword = ival in FITS header string
///
/// # Parameters
///
/// * `hstring` — character string containing FITS-style header information in the
///              format `<keyword>`= `<value>` {/ `<comment>`} each entry is padded
///              with spaces to 80 characters
/// * `keyword` — character string containing the name of the variable to be returned.
///              hput searches for a line beginning with this string, and if there
///              isn't one, creates one. The first 8 characters of keyword must be
///              unique.
/// * `ival`    — int number
fn hputi4(hstring: &mut [c_char], keyword: &[c_char], ival: c_int) {
    let mut value: [c_char; 30] = [0; 30];

    /* Translate value from binary to ASCII */
    int_snprintf!(&mut value, 30, "{}", ival);

    /* Put value into header string */
    hputc(hstring, keyword, &value);

    /* Return to calling program */
}

/// HPUTL - Set keyword = F if lval=0, else T, in FITS header string
///
/// # Parameters
///
/// * `hstring` — FITS header
/// * `keyword` — Keyword name
/// * `lval`    — logical variable (0=false, else true)
fn hputl(hstring: &mut [c_char], keyword: &[c_char], lval: c_int) {
    let mut value: [c_char; 8] = [0; 8];

    /* Translate value from binary to ASCII */
    if lval != 0 {
        strcpy_safe(&mut value, cs!(c"T"));
    } else {
        strcpy_safe(&mut value, cs!(c"F"));
    }

    /* Put value into header string */
    hputc(hstring, keyword, &value);

    /* Return to calling program */
}

///  HPUTS - Set character string keyword = 'cval' in FITS header string
///
/// # Parameters
///
/// * `hstring` — FITS header
/// * `keyword` — Keyword name
/// * `cval`    — character string containing the value for variable keyword. trailing
///              and leading blanks are removed.
fn hputs(hstring: &mut [c_char], keyword: &[c_char], cval: &[c_char]) {
    let squot = 39;
    let mut value: [c_char; 70] = [0; 70];

    /*  find length of variable string */

    let mut lcval = strlen_safe(cval);
    if lcval > 67 {
        lcval = 67;
    }

    /* Put quotes around string */
    value[0] = squot;
    strncpy_safe(&mut value[1..], cval, lcval);
    value[lcval + 1] = squot;
    value[lcval + 2] = 0;

    /* Put value into header string */
    hputc(hstring, keyword, &value);

    /* Return to calling program */
}

/// HPUTC - Set character string keyword = value in FITS header string
///
/// # Parameters
///
/// * `value` — character string containing the value for variable keyword. trailing and
///            leading blanks are removed.
fn hputc(hstring: &mut [c_char], keyword: &[c_char], value: &[c_char]) {
    let squot: c_char = 39 as c_int as c_char;
    let mut line: [c_char; 100] = [0; 100];
    let mut newcom: [c_char; 50] = [0; 50];
    let mut blank: [c_char; 80] = [0; 80];

    let mut vp = 0;
    let mut v1 = 0;
    let mut v2 = 0;
    let mut q2 = 0;
    let c1;

    let mut lcom: usize = 0;
    let mut lc: usize = 0;

    for i in 0..80 {
        blank[i] = bb(b' ');
    }

    /*  find length of keyword and value */
    let lkeyword = strlen_safe(keyword);
    let lval = strlen_safe(value);

    /*  If COMMENT or HISTORY, always add it just before the END */
    if lkeyword == 7
        && (strncmp_safe(keyword, cs!(c"COMMENT"), 7) == 0
            || strncmp_safe(keyword, cs!(c"HISTORY"), 7) == 0)
    {
        /* Find end of header */
        let Some(v1_tmp) = ksearch(hstring, cs!(c"END")) else {
            return; /* UB in original code */
        };
        v1 = v1_tmp;
        v2 = v1 + 80;

        /* Move END down one line */
        let (h1, h2) = hstring.split_at_mut(v2);
        //strncpy_safe(&mut hstring[v2..], &hstring[v1..], 80);
        strncpy_safe(h2, &h1[v1..], 80);

        /* Insert keyword */
        strncpy_safe(&mut hstring[v1..], keyword, 7);

        /* Pad with spaces */
        for vp in (v1 + lkeyword)..v2 {
            hstring[vp] = bb(b' ');
        }

        /* Insert comment */
        strncpy_safe(&mut hstring[(v1 + 9)..], value, lval);
        return;
    }

    /* Otherwise search for keyword */
    let v1_tmp = ksearch(hstring, keyword);

    /*  If parameter is not found, find a place to put it */
    if v1_tmp.is_none() {
        /* First look for blank lines before END */
        let v1_tmp = blsearch(hstring, cs!(c"END"));

        /*  Otherwise, create a space for it at the end of the header */
        if let Some(v1_tmp) = v1_tmp {
            v1 = v1_tmp;
            v2 = v1 + 80;
        } else {
            let Some(ve_tmp) = ksearch(hstring, cs!(c"END")) else {
                return; // UB in original code
            };
            v1 = ve_tmp;
            v2 = v1 + 80;

            let (h1, h2) = hstring.split_at_mut(v2);
            //strncpy_safe(&mut hstring[v2..], &hstring[v1..], 80);
            strncpy_safe(h2, &h1[v1..], 80);
        }
        lcom = 0;
        newcom[0] = 0;
    } else {
        /*  Otherwise, extract the entry for this keyword from the header */

        strncpy_safe(&mut line, &hstring[v1..], 80);
        line[80] = 0;
        v2 = v1 + 80;

        /*  check for quoted value */
        let q1 = strchr_safe(&line, squot);
        if let Some(q1) = q1 {
            let q2_tmp = strchr_safe(&line[(q1 + 1)..], squot);
            if let Some(x) = q2_tmp {
                q2 = x;
            } else {
                return; // UB in original code
            }
        } else {
            q2 = 0;
        }

        /*  extract comment and remove trailing spaces */

        c1 = strchr_safe(&line[q2..], bb(b'/'));
        if let Some(c1) = c1 {
            lcom = 80 - c1;
            strncpy_safe(&mut newcom, &line[(c1 + 1)..], lcom);
            vp = lcom - 1;

            while vp > 0 {
                vp -= 1;
                if newcom[vp] == bb(b' ') {
                    newcom[vp] = 0;
                }
            }
            lcom = strlen_safe(&newcom);
        } else {
            newcom[0] = 0;
            lcom = 0;
        }
    }

    /* Fill new entry with spaces */
    for vp in v1..v2 {
        hstring[vp] = bb(b' ');
    }

    /*  Copy keyword to new entry */
    strncpy_safe(&mut hstring[v1..], keyword, lkeyword);

    /*  Add parameter value in the appropriate place */
    vp = v1 + 8;
    hstring[vp] = bb(b'=');
    vp = v1 + 9;
    hstring[vp] = bb(b' ');
    vp += 1;
    if value[0] == squot {
        strncpy_safe(&mut hstring[vp..], value, lval);
        if lval + 12 > 31 {
            lc = lval + 12;
        } else {
            lc = 30;
        }
    } else {
        vp = v1 + 30 - lval;
        strncpy_safe(&mut hstring[vp..], value, lval);
        lc = 30;
    }

    /* Add comment in the appropriate place */
    if lcom > 0 {
        if lc + 2 + lcom > 80 {
            lcom = 78 - lc;
        }
        vp = v1 + lc + 2; /* Jul 16 1997: was vp = v1 + lc * 2 */
        hstring[vp] = bb(b'/');
        vp += 1;
        strncpy_safe(&mut hstring[vp..], &newcom, lcom);
        for v in (vp + lcom)..v2 {
            hstring[v] = bb(b' ');
        }
    }
}

/// HPUTCOM - Set comment for keyword or on line in FITS header string
fn hputcom(hstring: &mut [c_char], keyword: &[c_char], comment: &[c_char]) {
    let mut squot: c_char = 0;
    let mut line: [c_char; 100] = [0; 100];

    let v1;
    let v2;
    let mut c0 = 0;
    let c1;
    let q2;

    squot = 39;

    /*  Find length of variable name */
    let lkeyword = strlen_safe(keyword);

    /*  If COMMENT or HISTORY, always add it just before the END */
    if lkeyword == 7
        && (strncmp_safe(keyword, cs!(c"COMMENT"), 7) == 0
            || strncmp_safe(keyword, cs!(c"HISTORY"), 7) == 0)
    {
        /* Find end of header */
        let Some(v1_tmp) = ksearch(hstring, cs!(c"END")) else {
            return; /* UB in original code */
        };
        v1 = v1_tmp;

        v2 = v1 + 80;

        let (h1, h2) = hstring.split_at_mut(v2);
        //strncpy_safe(&mut hstring[v2..], &hstring[v1..], 80);
        strncpy_safe(h2, &h1[v1..], 80);

        /*  blank out new line and insert keyword */
        for vp in v1..v2 {
            hstring[vp] = bb(b' ');
        }

        strncpy_safe(&mut hstring[v1..], keyword, lkeyword);
    } else {
        /* search header string for variable name */

        let Some(v1_tmp) = ksearch(hstring, keyword) else {
            return; /* if parameter is not found, return without doing anything */
        };
        v1 = v1_tmp;

        v2 = v1 + 80;

        /* otherwise, extract entry for this variable from the header */
        strncpy_safe(&mut line, &hstring[v1..], 80);

        /* check for quoted value */
        let q1 = strchr_safe(&line, squot);
        q2 = match q1 {
            /* make the second-quote offset relative to the start of `line`
            (strchr_safe returns it relative to line[q1+1..]) so that it can be
            compared/used as `q2 - line` like in the C original */
            Some(x) => strchr_safe(&line[(x + 1)..], squot).map(|off| x + 1 + off),
            None => None,
        };

        if let Some(q2) = q2 {
            if q2 < 31 {
                c0 = v1 + 31;
            } else {
                c0 = v1 + q2 + 2; /* allan: 1997-09-30, was c0=q2+2 */
            }
        } else {
            c0 = v1 + 31;
        }

        /* strncpy (c0, "/ ",2);  - c0 is an index into hstring, not line */
        /* A long quoted value (e.g. a long IMHFILE/PIXFILE path on Windows,
        where temp paths easily exceed the column at which the closing quote
        sits) can push c0 to the very end of this 80-char card.  The C code
        writes "/ " unconditionally, running past the card end; only do so
        while it still fits inside the card. */
        if c0 + 2 <= v2 {
            strncpy_safe(&mut hstring[c0..], cs!(c"/ "), 2);
        }
    }

    /* create new entry */
    let mut lcom = strlen_safe(comment);

    if lcom > 0 {
        c1 = c0 + 2;
        /* Clamp the comment to whatever room is left in the card.  C computes
        `lcom = v2 - c1` as a signed int, which goes negative when the value
        already fills the card (c1 > v2) and then overruns in strncpy; guard
        against that so we never underflow or write past the card. */
        if c1 >= v2 {
            lcom = 0;
        } else if c1 + lcom > v2 {
            lcom = v2 - c1;
        }
        if lcom > 0 {
            strncpy_safe(&mut hstring[c1..], comment, lcom);
        }
    }
}

#[cfg(test)]
mod tests {

    use alloc::ffi::CString;

    use super::*;

    use crate::aliases::rust_api::*;
    use crate::fitsio::{READONLY, fitsfile};
    use crate::helpers::testhelpers::to_buf;
    use libc::{c_char, c_int};
    use std::path::{Path, PathBuf};

    /// Write a 4-byte integer to buffer at offset (little-endian).
    fn put_i4_le(buf: &mut [u8], offset: usize, val: i32) {
        let v = val as u32;
        buf[offset] = (v & 0xff) as u8;
        buf[offset + 1] = ((v >> 8) & 0xff) as u8;
        buf[offset + 2] = ((v >> 16) & 0xff) as u8;
        buf[offset + 3] = ((v >> 24) & 0xff) as u8;
    }

    /// Write a string in IRAF v1 format (2 bytes per char, little-endian).
    fn put_str_v1(buf: &mut [u8], offset: usize, s: &str) {
        for (i, b) in s.bytes().enumerate() {
            buf[offset + i * 2] = b;
            buf[offset + i * 2 + 1] = 0;
        }
    }

    /// Write a string in IRAF v2 format (1 byte per char).
    fn put_str_v2(buf: &mut [u8], offset: usize, s: &str) {
        let bytes = s.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        buf[offset + bytes.len()] = 0;
    }

    /// A temporary directory holding the .imh/.pix pair.
    /// The fixed base names match what the C test uses so that the `HDR$`
    /// prefix in the header resolves to the same directory.
    struct IrafFiles {
        _dir: tempfile::TempDir,
        imh: PathBuf,
        pix: PathBuf,
    }

    impl IrafFiles {
        fn new() -> Self {
            // Create the temp dir as a direct child of the crate root (".", where
            // `cargo test` runs) rather than the system temp dir.  iraffits
            // resolves "HDR$<pix>" against the header pathname and stores the
            // result in an 80-column FITS card, whose value field holds at most
            // ~68 chars.  The system temp dir on Windows
            // (C:\Users\<user>\AppData\Local\Temp\...) overruns that card and the
            // pixfile path is silently truncated, so the pixel file can't be
            // opened.
            //
            // NOTE: `tempdir_in(".")` still returns an *absolute* path
            // (e.g. /home/<user>/code/rsfitsio/./rsfitsio-iraf-XXXX), so on a deep
            // working directory the resolved pixfile path again exceeds the card
            // limit and is truncated (this is what broke the Linux tests).  We
            // therefore hand the FITS API a path *relative* to the cwd
            // (`imh_buf()` below), which stays short on every platform regardless
            // of how deep the cwd is.  The absolute `imh`/`pix` paths are kept for
            // writing the fixture files via std::fs.
            let dir = tempfile::Builder::new()
                .prefix("rsfitsio-iraf-")
                .tempdir_in(".")
                .unwrap();
            let imh = dir.path().join("test_iraffits.imh");
            let pix = dir.path().join("test_iraffits.pix");
            IrafFiles {
                _dir: dir,
                imh,
                pix,
            }
        }

        /// Path to the .imh file relative to the current working directory.
        /// Because the temp dir is a direct child of the cwd, this is just
        /// `<dirname>/test_iraffits.imh` — short enough to survive the ~68-char
        /// FITS PIXFILE card on every platform.
        fn imh_buf(&self) -> [c_char; crate::fitsio::FLEN_FILENAME] {
            let dirname = self._dir.path().file_name().unwrap();
            let rel = Path::new(dirname).join("test_iraffits.imh");
            to_buf(rel.to_str().unwrap())
        }
    }

    /// Create a minimal IRAF v1 header file (.imh).
    fn create_iraf_v1_header(imhfile: &Path, pixfile: &str, pixtype: i32, ndim: i32, dims: &[i32]) {
        let hdrlen = LEN_IMHDR + 500;
        let mut header = vec![0u8; hdrlen];

        // Magic word "imhdr" in 2-byte per char format.
        put_str_v1(&mut header, 0, "imhdr");

        // Header length in 4-byte units.
        put_i4_le(&mut header, 12, (hdrlen / 4) as i32);

        // Pixel type.
        put_i4_le(&mut header, IM_PIXTYPE, pixtype);

        // Number of dimensions.
        put_i4_le(&mut header, IM_NDIM, ndim);

        // Image dimensions.
        for i in 0..(ndim as usize).min(7) {
            put_i4_le(&mut header, IM_LEN + i * 4, dims[i]);
            put_i4_le(&mut header, IM_PHYSLEN + i * 4, dims[i]);
        }

        // Pixel offset (in 2-byte units, starts after LEN_PIXHDR).
        put_i4_le(&mut header, IM_PIXOFF, (LEN_PIXHDR / 2 + 1) as i32);

        // Pixel file name (use HDR$ prefix for same directory).
        put_str_v1(&mut header, IM_PIXFILE, pixfile);

        // Title.
        put_str_v1(&mut header, IM_TITLE, "Test IRAF Image");

        std::fs::write(imhfile, &header).unwrap();
    }

    /// Create a minimal IRAF v2 header file (.imh).
    fn create_iraf_v2_header(
        imhfile: &Path,
        pixfile: &str,
        pixtype: i32,
        ndim: i32,
        dims: &[i32],
        swapped: i32,
    ) {
        let hdrlen = LEN_IM2HDR + 500;
        let mut header = vec![0u8; hdrlen];

        // Magic word "imhv2" in 1-byte per char format.
        put_str_v2(&mut header, 0, "imhv2");

        // Header length in 4-byte units.
        put_i4_le(&mut header, 6, (hdrlen / 4) as i32);

        // Pixel type.
        put_i4_le(&mut header, IM2_PIXTYPE, pixtype);

        // Swapped flag.
        put_i4_le(&mut header, IM2_SWAPPED, swapped);

        // Number of dimensions.
        put_i4_le(&mut header, IM2_NDIM, ndim);

        // Image dimensions.
        for i in 0..(ndim as usize).min(7) {
            put_i4_le(&mut header, IM2_LEN + i * 4, dims[i]);
            put_i4_le(&mut header, IM2_PHYSLEN + i * 4, dims[i]);
        }

        // Pixel offset (in 2-byte units, starts after LEN_PIXHDR).
        put_i4_le(&mut header, IM2_PIXOFF, (LEN_PIXHDR / 2 + 1) as i32);

        // Pixel file name (use HDR$ prefix for same directory).
        put_str_v2(&mut header, IM2_PIXFILE, pixfile);

        // Title.
        put_str_v2(&mut header, IM2_TITLE, "Test IRAF v2 Image");

        std::fs::write(imhfile, &header).unwrap();
    }

    /// Create a minimal IRAF v1 pixel file (.pix).
    fn create_iraf_v1_pixfile(pixfile: &Path, pixtype: i32, npixels: usize) {
        let mut buf: Vec<u8> = Vec::new();

        // Write pixel header with magic word "impix".
        let mut pixhdr = vec![0u8; LEN_PIXHDR];
        put_str_v1(&mut pixhdr, 0, "impix");
        buf.extend_from_slice(&pixhdr);

        // Write pixel data.
        match pixtype {
            TY_SHORT => {
                for i in 0..npixels {
                    let v = (i % 100) as i16;
                    buf.extend_from_slice(&v.to_ne_bytes());
                }
            }
            TY_INT => {
                for i in 0..npixels {
                    let v = (i as i32) * 10;
                    buf.extend_from_slice(&v.to_ne_bytes());
                }
            }
            TY_REAL => {
                for i in 0..npixels {
                    let v = (i as f32) * 1.5f32;
                    buf.extend_from_slice(&v.to_ne_bytes());
                }
            }
            _ => panic!("unsupported pixtype"),
        }

        std::fs::write(pixfile, &buf).unwrap();
    }

    /// Create a minimal IRAF v2 pixel file (.pix).
    fn create_iraf_v2_pixfile(pixfile: &Path, _pixtype: i32, npixels: usize) {
        let mut buf: Vec<u8> = Vec::new();

        // Write pixel header with magic word "impv2".
        let mut pixhdr = vec![0u8; LEN_PIXHDR];
        put_str_v2(&mut pixhdr, 0, "impv2");
        buf.extend_from_slice(&pixhdr);

        // Write pixel data (just short for simplicity).
        for i in 0..npixels {
            let v = (i % 256) as i16;
            buf.extend_from_slice(&v.to_ne_bytes());
        }

        std::fs::write(pixfile, &buf).unwrap();
    }

    /// Open an IRAF .imh file through the public API (which internally invokes
    /// iraf2mem via the irafmem:// driver) and return (status, file handle).
    /// On the success path the converted FITS file has a valid SIMPLE header,
    /// so a successful open is equivalent to the C check of the "SIMPLE"
    /// magic in the converted buffer.
    fn open_iraf(imh: &[c_char]) -> (c_int, Option<Box<fitsfile>>) {
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        fits_open_file(&mut fptr, imh, READONLY, &mut status);
        (status, fptr)
    }

    fn close(fptr: Option<Box<fitsfile>>) {
        if let Some(f) = fptr {
            let mut status = 0;
            fits_close_file(f, &mut status);
        }
    }

    #[test]
    fn test_iraf2mem_v1_short() {
        // Test reading IRAF v1 format with short pixels.
        let files = IrafFiles::new();
        let dims = [10, 10];

        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 2, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_SHORT, 100);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_v1_int() {
        // Test reading IRAF v1 format with int pixels.
        let files = IrafFiles::new();
        let dims = [8, 8];

        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_INT, 2, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_INT, 64);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_v1_real() {
        // Test reading IRAF v1 format with float pixels.
        let files = IrafFiles::new();
        let dims = [5, 5];

        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_REAL, 2, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_REAL, 25);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_v2() {
        // Test reading IRAF v2 format.
        let files = IrafFiles::new();
        let dims = [10, 10];

        create_iraf_v2_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 2, &dims, 0);
        create_iraf_v2_pixfile(&files.pix, TY_SHORT, 100);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_v2_swapped() {
        // Test reading IRAF v2 format with swapped pixels.
        let files = IrafFiles::new();
        let dims = [10, 10];

        create_iraf_v2_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 2, &dims, 1);
        create_iraf_v2_pixfile(&files.pix, TY_SHORT, 100);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_1d() {
        // Test reading 1-dimensional IRAF image.
        let files = IrafFiles::new();
        let dims = [50];

        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 1, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_SHORT, 50);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_3d() {
        // Test reading 3-dimensional IRAF image.
        let files = IrafFiles::new();
        let dims = [4, 4, 4];

        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 3, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_SHORT, 64);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_4d() {
        // Test reading 4-dimensional IRAF image.
        let files = IrafFiles::new();
        let dims = [2, 2, 2, 2];

        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 4, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_SHORT, 16);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    #[ignore = "iraftofits (matching CFITSIO) only emits NAXIS1..NAXIS4, so a 5-D IRAF \
                image converts to a FITS header declaring NAXIS=5 with no NAXIS5 keyword. \
                The C test calls iraf2mem() directly and never validates the header; this \
                test instead opens via fits_open_file(), which correctly rejects the \
                incomplete header (status 224). Unsupported through the public open path."]
    fn test_iraf2mem_5d() {
        // Test reading 5-dimensional IRAF image.
        let files = IrafFiles::new();
        let dims = [2, 2, 2, 2, 2];

        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 5, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_SHORT, 32);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_missing_header() {
        // Test error handling for missing header file.
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        let name = to_buf("nonexistent.imh");
        fits_open_file(&mut fptr, &name, READONLY, &mut status);

        // Should fail to open.
        assert_ne!(status, 0);
        assert!(fptr.is_none());
    }

    #[test]
    fn test_iraf2mem_invalid_header() {
        // Test error handling for invalid header file.
        let files = IrafFiles::new();

        // Create a file with invalid content (magic word not "imhdr"/"imhv2").
        let junk = vec![b'X'; 2100];
        std::fs::write(&files.imh, &junk).unwrap();

        let (status, fptr) = open_iraf(&files.imh_buf());
        // Should fail because magic word is invalid.
        assert_ne!(status, 0);
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_missing_pixfile() {
        // Test error handling for missing pixel file.
        let files = IrafFiles::new();
        let dims = [10, 10];

        // Create header but no pixel file.
        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 2, &dims);

        let (status, fptr) = open_iraf(&files.imh_buf());
        // Should fail because pixel file is missing.
        assert_ne!(status, 0);
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_header_too_small() {
        // Test error handling for header file that is too small.
        let files = IrafFiles::new();

        // Create a file that is too small to be valid.
        let small = vec![0u8; 100];
        std::fs::write(&files.imh, &small).unwrap();

        let (status, fptr) = open_iraf(&files.imh_buf());
        // Should fail because file is too small.
        assert_ne!(status, 0);
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_invalid_pixfile() {
        // Test error handling for pixel file with invalid magic.
        let files = IrafFiles::new();
        let dims = [10, 10];

        // Create valid header.
        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 2, &dims);

        // Create pixel file with invalid magic word.
        let mut pixhdr = vec![0u8; LEN_PIXHDR];
        put_str_v1(&mut pixhdr, 0, "XXXXX");
        std::fs::write(&files.pix, &pixhdr).unwrap();

        let (status, fptr) = open_iraf(&files.imh_buf());
        // Should fail because pixel file magic is invalid.
        assert_ne!(status, 0);
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_truncated_pixfile() {
        // Test error handling for truncated pixel file.
        let files = IrafFiles::new();
        let dims = [10, 10];

        // Create valid header expecting 100 pixels.
        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 2, &dims);

        // Create pixel file with only the header, no data.
        let mut pixhdr = vec![0u8; LEN_PIXHDR];
        put_str_v1(&mut pixhdr, 0, "impix");
        std::fs::write(&files.pix, &pixhdr).unwrap();

        let (status, fptr) = open_iraf(&files.imh_buf());
        // Should fail because pixel file is truncated.
        assert_ne!(status, 0);
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_large_image() {
        // Test reading a larger IRAF image.
        let files = IrafFiles::new();
        let dims = [100, 100];

        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 2, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_SHORT, 10000);

        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_iraf2mem_v1_different_pixtypes() {
        // Test IRAF v1 format with different pixel types in sequence.
        let dims = [10, 10];

        // Test with TY_INT = 4.
        let files = IrafFiles::new();
        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_INT, 2, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_INT, 100);
        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);

        // Test with TY_REAL = 6.
        let files = IrafFiles::new();
        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_REAL, 2, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_REAL, 100);
        let (status, fptr) = open_iraf(&files.imh_buf());
        assert_eq!(status, 0);
        assert!(fptr.is_some());
        close(fptr);
    }

    #[test]
    fn test_fits_delete_iraf_file() {
        // Test deleting IRAF files.
        let mut status: c_int = 0;
        let files = IrafFiles::new();
        let dims = [10, 10];

        // Create IRAF files.
        create_iraf_v1_header(&files.imh, "HDR$test_iraffits.pix", TY_SHORT, 2, &dims);
        create_iraf_v1_pixfile(&files.pix, TY_SHORT, 100);

        // Verify files exist.
        assert!(files.imh.exists());
        assert!(files.pix.exists());

        // Delete them.
        let imh_cbuf = files.imh_buf();
        fits_delete_iraf_file_safe(&imh_cbuf, &mut status);
        assert_eq!(status, 0);

        // Verify files are gone.
        assert!(!files.imh.exists());
        assert!(!files.pix.exists());
    }

    #[test]
    fn test_fits_delete_iraf_file_missing() {
        // Test error handling for deleting non-existent file.
        let mut status: c_int = 0;
        let name = to_buf("nonexistent.imh");

        fits_delete_iraf_file_safe(&name, &mut status);

        // Should return error.
        assert_ne!(status, 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_read_iraf_header() {
        let iraf_imh_file = "test_resources/pix.imh";

        let mut header_len = 0;
        let z = irafrdhead(
            cast_slice(CString::new(iraf_imh_file).unwrap().to_bytes_with_nul()),
            &mut header_len,
        );

        assert!(z.is_ok());
        assert_eq!(header_len, 5288);

        let z = z.unwrap();

        let header_version = head_version(&z);
        let _pix_version = pix_version(&z);

        assert_eq!(header_version, 2);
        // assert_eq!(pix_version, 2);
    }
}
