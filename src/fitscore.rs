//! The core FITSIO routines.
//!
//! This is the largest module in the crate and the one the others lean on. It
//! holds the error message stack, the HDU structure machinery -- reading an
//! HDU's required keywords, working out the column layout of a table, moving
//! between HDUs and rewriting the header when the structure changes -- the
//! `TFORMn` / `TDISPn` format parsers, the column name matching, and the
//! string-to-number conversions the keyword routines are built from.
//!
//! Ported from CFITSIO's `fitscore.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center.
#![warn(missing_docs)]

/*

Copyright (Unpublished--all rights reserved under the copyright laws of
the United States), U.S. Government as represented by the Administrator
of the National Aeronautics and Space Administration.  No copyright is
claimed in the United States under Title 17, U.S. Code.

Permission to freely use, copy, modify, and distribute this software
and its documentation without fee is hereby granted, provided that this
copyright notice and disclaimer of warranty appears in all copies.

DISCLAIMER:

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND,
EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO,
ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE
DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE
SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NASA BE LIABLE FOR ANY
DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR
CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY
CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY,
CONTRACT, TORT , OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY
PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED
FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR
SERVICES PROVIDED HEREUNDER."

*/

use alloc::collections::VecDeque;
use core::borrow::BorrowMut;
use core::ffi::CStr;
use core::num::{ParseFloatError, ParseIntError};
use core::{cmp, ptr};
use core::{slice, str};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::c_types::{c_char, c_int, c_long, c_short, c_uint, c_ulong, c_ushort, off_t};
use crate::helpers::vec_raw_parts::vec_into_raw_parts;
use crate::wrappers::strcpy_safe;

use bytemuck::{cast_slice, cast_slice_mut};

use crate::BL;
use crate::aliases::rust_api::fits_read_key_dbl;
use crate::cfileio::{STREAM_DRIVER, ffinit_safe, fftrun, urltype2driver};
use crate::cfileio::{ffclos_safe, ffurlt_safe};
use crate::editcol::ffirow_safe;
use crate::edithdu::ffcopy_safe;
use crate::fitsio::*;
use crate::fitsio2::*;
use crate::getkey::{
    ffghsp_safe, ffgky_safe, ffgkyj_safe, ffgkyjj_safe, ffgkyl_safe, ffgkyn_safe, ffgkys_safe,
    ffgphd, ffgrec_safe, ffgttb, ffmaky_safe,
};
use crate::imcompress::{TILE_STRUCTS, imcomp_get_compressed_image_par};
use crate::modkey::{ffdkey_safe, ffmkyj_safe, ffmrec_safe};
use crate::putkey::ffprec_safe;
use crate::relibc::header::stdio::{sscanf_ld, sscanf_lf};
use crate::{atoi, bb, cs, int_snprintf};
use crate::{buffers::*, raw_to_slice};
use crate::{slice_to_str, wrappers::*};

/// Maximum number of messages the error stack holds.
pub const ERRMSGSIZ: usize = 25;
/// Escape character, used as the error stack marker.
pub const ESMARKER: c_char = 27;

/// `ffxmsg` action: delete all messages on the error stack.
pub const DEL_ALL: c_int = 1;
/// `ffxmsg` action: delete the newest messages back to and including the
/// marker.
pub const DEL_MARK: c_int = 2;
/// `ffxmsg` action: delete the newest message from the stack.
pub const DEL_NEWEST: c_int = 3;
/// `ffxmsg` action: pop and return the oldest message, ignoring marks.
pub const GET_MESG: c_int = 4;
/// `ffxmsg` action: add a new message to the stack.
pub const PUT_MESG: c_int = 5;
/// `ffxmsg` action: add a marker to the stack.
pub const PUT_MARK: c_int = 6;

// Use this to keep track of allocations so we can deallocate with the same `Layout` used for allocation.
pub(crate) static ALLOCATIONS: LazyLock<Mutex<HashMap<usize, (usize, usize)>>> =
    LazyLock::new(Default::default);

static ERROR_STACK: LazyLock<Mutex<ErrorStack>> = LazyLock::new(|| Mutex::new(ErrorStack::new()));

/// return the current version number of the FITSIO software
///
///  Note that this method of calculation limits minor/micro fields to < 100.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffvers(version: *mut f32 /* IO - version number */) -> f32 {
    // FFI WRAPPER
    unsafe {
        if version.is_null() {
            return 0.0;
        }

        let version = version.as_mut().expect(NULL_MSG);

        ffvers_safe(version)
    }
}

/// Return the current version number of the FITSIO software.
///
/// Note that this method of calculation limits the minor and micro fields to
/// less than 100.
///
/// # Parameters
///
/// * `version` — (O) version number
pub fn ffvers_safe(version: &mut f32) -> f32 {
    *version = (CFITSIO_MAJOR as f32)
        + (0.01 * (CFITSIO_MINOR as f32))
        + (0.0001 * (CFITSIO_MICRO as f32));

    /*
          *version = 4.7.0      Aug 2026

       Previous releases:
          *version = 4.6.4      Apr 2026
          *version = 4.6.3      Sep 2025
          *version = 4.6.2      Mar 2025 (autotools change only)
          *version = 4.6.1      Mar 2025 (autotools/cmake config changes only)
          *version = 4.6.0      Mar 2025
          *version = 4.5.0      Aug 2024
          *version = 4.4.1      Jun 2024 (license change)
          *version = 4.4.0      Feb 2024
          *version = 4.3.1      Nov 2023 (patch)
          *version = 4.3.0      Jul 2023
          *version = 4.2.0      Nov 2022
          *version = 4.1.0      Feb 2022
          *version = 4.0.0      May 2021
          *version = 3.49       Aug 2020
          *version = 3.48       Apr 2020
          *version = 3.47       May 2019
          *version = 3.46       Oct 2018
          *version = 3.45       May 2018
          *version = 3.44       Apr 2018
          *version = 3.43       Mar 2018
          *version = 3.42       Mar 2017
          *version = 3.41       Nov 2016
          *version = 3.40       Oct 2016
          *version = 3.39       Apr 2016
          *version = 3.38       Feb 2016
          *version = 3.37     3 Jun 2014
          *version = 3.36     6 Dec 2013
          *version = 3.35    23 May 2013
          *version = 3.34    20 Mar 2013
          *version = 3.33    14 Feb 2013
          *version = 3.32       Oct 2012
          *version = 3.31    18 Jul 2012
          *version = 3.30    11 Apr 2012
          *version = 3.29    22 Sep 2011
          *version = 3.28    12 May 2011
          *version = 3.27     3 Mar 2011
          *version = 3.26    30 Dec 2010
          *version = 3.25    9 June 2010
          *version = 3.24    26 Jan 2010
          *version = 3.23     7 Jan 2010
          *version = 3.22    28 Oct 2009
          *version = 3.21    24 Sep 2009
          *version = 3.20    31 Aug 2009
          *version = 3.18    12 May 2009 (beta version)
          *version = 3.14    18 Mar 2009
          *version = 3.13     5 Jan 2009
          *version = 3.12     8 Oct 2008
          *version = 3.11    19 Sep 2008
          *version = 3.10    20 Aug 2008
          *version = 3.09     3 Jun 2008
          *version = 3.08    15 Apr 2007  (internal release)
          *version = 3.07     5 Nov 2007  (internal release)
          *version = 3.06    27 Aug 2007
          *version = 3.05    12 Jul 2007  (internal release)
          *version = 3.03    11 Dec 2006
          *version = 3.02    18 Sep 2006
          *version = 3.01       May 2006 included in FTOOLS 6.1 release
          *version = 3.006   20 Feb 2006
          *version = 3.005   20 Dec 2005 (beta, in heasoft swift release
          *version = 3.004   16 Sep 2005 (beta, in heasoft swift release
          *version = 3.003   28 Jul 2005 (beta, in heasoft swift release
          *version = 3.002   15 Apr 2005 (beta)
          *version = 3.001   15 Mar 2005 (beta) released with heasoft 6.0
          *version = 3.000   1 Mar 2005 (internal release only)
          *version = 2.51     2 Dec 2004
          *version = 2.50    28 Jul 2004
          *version = 2.49    11 Feb 2004
          *version = 2.48    28 Jan 2004
          *version = 2.470   18 Aug 2003
          *version = 2.460   20 May 2003
          *version = 2.450   30 Apr 2003  (internal release only)
          *version = 2.440    8 Jan 2003
          *version = 2.430;   4 Nov 2002
          *version = 2.420;  19 Jul 2002
          *version = 2.410;  22 Apr 2002 used in ftools v5.2
          *version = 2.401;  28 Jan 2002
          *version = 2.400;  18 Jan 2002
          *version = 2.301;   7 Dec 2001
          *version = 2.300;  23 Oct 2001
          *version = 2.204;  26 Jul 2001
          *version = 2.203;  19 Jul 2001 used in ftools v5.1
          *version = 2.202;  22 May 2001
          *version = 2.201;  15 Mar 2001
          *version = 2.200;  26 Jan 2001
          *version = 2.100;  26 Sep 2000
          *version = 2.037;   6 Jul 2000
          *version = 2.036;   1 Feb 2000
          *version = 2.035;   7 Dec 1999 (internal release only)
          *version = 2.034;  23 Nov 1999
          *version = 2.033;  17 Sep 1999
          *version = 2.032;  25 May 1999
          *version = 2.031;  31 Mar 1999
          *version = 2.030;  24 Feb 1999
          *version = 2.029;  11 Feb 1999
          *version = 2.028;  26 Jan 1999
          *version = 2.027;  12 Jan 1999
          *version = 2.026;  23 Dec 1998
          *version = 2.025;   1 Dec 1998
          *version = 2.024;   9 Nov 1998
          *version = 2.023;   1 Nov 1998 first full release of V2.0
          *version = 1.42;   30 Apr 1998
          *version = 1.40;    6 Feb 1998
          *version = 1.33;   16 Dec 1997 (internal release only)
          *version = 1.32;   21 Nov 1997 (internal release only)
          *version = 1.31;    4 Nov 1997 (internal release only)
          *version = 1.30;   11 Sep 1997
          *version = 1.27;    3 Sep 1997 (internal release only)
          *version = 1.25;    2 Jul 1997
          *version = 1.24;    2 May 1997
          *version = 1.23;   24 Apr 1997
          *version = 1.22;   18 Apr 1997
          *version = 1.21;   26 Mar 1997
          *version = 1.2;    29 Jan 1997
          *version = 1.11;   04 Dec 1996
          *version = 1.101;  13 Nov 1996
          *version = 1.1;     6 Nov 1996
          *version = 1.04;   17 Sep 1996
          *version = 1.03;   20 Aug 1996
          *version = 1.02;   15 Aug 1996
          *version = 1.01;   12 Aug 1996
    */
    *version
}

///  return the name of the FITS file
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `filename` — (O) name of the file
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffflnm(
    fptr: *mut fitsfile,
    filename: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let filename = core::slice::from_raw_parts_mut(filename, FLEN_FILENAME);

        ffflnm_safe(&mut *fptr, filename, &mut *status)
    }
}

/// Return the name of the FITS file.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `filename` — (O) name of the file
/// * `status`   — (IO) error status
pub fn ffflnm_safe(fptr: &mut fitsfile, filename: &mut [c_char], status: &mut c_int) -> c_int {
    strcpy_safe(
        filename,
        cast_slice(fptr.Fptr.get_filename_as_cstr().to_bytes_with_nul()),
    );
    *status
}

/// return the access mode of the FITS file
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `filemode` — (O) open mode of the file
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffflmd(
    fptr: *mut fitsfile,
    filemode: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let status = status.as_mut().expect(NULL_MSG);
        ffflmd_safe(&mut *fptr, &mut *filemode, &mut *status)
    }
}

/// Return the I/O mode of the FITS file: [`READONLY`] or [`READWRITE`].
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `filemode` — (O) file mode
/// * `status`   — (IO) error status
pub fn ffflmd_safe(fptr: &mut fitsfile, filemode: &mut c_int, status: &mut c_int) -> c_int {
    *filemode = fptr.Fptr.writemode;
    *status
}

/// Return a short descriptive error message that corresponds to the input
/// error status value.  The message may be up to 30 characters long, plus
/// the terminating null character.
///
/// # Parameters
///
/// * `status`  — (I) error status value
/// * `errtext` — (O) error message (max 30 char long + null)
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgerr(status: c_int, errtext: *mut c_char) {
    // FFI WRAPPER
    unsafe {
        let errtext = slice::from_raw_parts_mut(errtext, 31);

        ffgerr_safe(status, errtext)
    }
}

/// Return a short descriptive error message that corresponds to the input
/// error status value.  The message may be up to 30 characters long, plus
/// the terminating null character.
///
/// # Parameters
///
/// * `status`  — (I) error status value
/// * `errtext` — (O) error message (max 30 char long + null)
pub fn ffgerr_safe(status: c_int, errtext: &mut [c_char]) {
    errtext[0] = 0;
    let t = if status >= 0 && status < 300 {
        match status {
            0 => c"OK - no error",
            1 => c"non-CFITSIO program error",
            101 => c"same input and output files",
            103 => c"attempt to open too many files",
            104 => c"could not open the named file",
            105 => c"couldn't create the named file",
            106 => c"error writing to FITS file",
            107 => c"tried to move past end of file",
            108 => c"error reading from FITS file",
            110 => c"could not close the file",
            111 => c"array dimensions too big",
            112 => c"cannot write to readonly file",
            113 => c"could not allocate memory",
            114 => c"invalid fitsfile pointer",
            115 => c"NULL input pointer",
            116 => c"error seeking file position",
            117 => c"bad value for file download timeout setting",
            121 => c"invalid URL prefix",
            122 => c"too many I/O drivers",
            123 => c"I/O driver init failed",
            124 => c"no I/O driver for this URLtype",
            125 => c"parse error in input file URL",
            126 => c"parse error in range list",
            151 => c"bad argument (shared mem drvr)",
            152 => c"null ptr arg (shared mem drvr)",
            153 => c"no free shared memory handles",
            154 => c"share mem drvr not initialized",
            155 => c"IPC system error (shared mem)",
            156 => c"no memory (shared mem drvr)",
            157 => c"share mem resource deadlock",
            158 => c"lock file open/create failed",
            159 => c"can't resize share mem block",
            201 => c"header already has keywords",
            202 => c"keyword not found in header",
            203 => c"keyword number out of bounds",
            204 => c"keyword value is undefined",
            205 => c"string missing closing quote",
            206 => c"error in indexed keyword name",
            207 => c"illegal character in keyword",
            208 => c"required keywords out of order",
            209 => c"keyword value not positive int",
            210 => c"END keyword not found",
            211 => c"illegal BITPIX keyword value",
            212 => c"illegal NAXIS keyword value",
            213 => c"illegal NAXISn keyword value",
            214 => c"illegal PCOUNT keyword value",
            215 => c"illegal GCOUNT keyword value",
            216 => c"illegal TFIELDS keyword value",
            217 => c"negative table row size",
            218 => c"negative number of rows",
            219 => c"named column not found",
            220 => c"illegal SIMPLE keyword value",
            221 => c"first keyword not SIMPLE",
            222 => c"second keyword not BITPIX",
            223 => c"third keyword not NAXIS",
            224 => c"missing NAXISn keywords",
            225 => c"first keyword not XTENSION",
            226 => c"CHDU not an ASCII table",
            227 => c"CHDU not a binary table",
            228 => c"PCOUNT keyword not found",
            229 => c"GCOUNT keyword not found",
            230 => c"TFIELDS keyword not found",
            231 => c"missing TBCOLn keyword",
            232 => c"missing TFORMn keyword",
            233 => c"CHDU not an IMAGE extension",
            234 => c"illegal TBCOLn keyword value",
            235 => c"CHDU not a table extension",
            236 => c"column exceeds width of table",
            237 => c"more than 1 matching col. name",
            241 => c"row width not = field widths",
            251 => c"unknown FITS extension type",
            252 => c"1st key not SIMPLE or XTENSION",
            253 => c"END keyword is not blank",
            254 => c"Header fill area not blank",
            255 => c"Data fill area invalid",
            261 => c"illegal TFORM format code",
            262 => c"unknown TFORM datatype code",
            263 => c"illegal TDIMn keyword value",
            264 => c"invalid BINTABLE heap pointer",
            _ => c"unknown error status",
        }
    } else if status < 600 {
        match status {
            301 => c"illegal HDU number",
            302 => c"column number < 1 or > tfields",
            304 => c"negative byte address",
            306 => c"negative number of elements",
            307 => c"bad first row number",
            308 => c"bad first element number",
            309 => c"not an ASCII (A) column",
            310 => c"not a logical (L) column",
            311 => c"bad ASCII table datatype",
            312 => c"bad binary table datatype",
            314 => c"null value not defined",
            317 => c"not a variable length column",
            320 => c"illegal number of dimensions",
            321 => c"1st pixel no. > last pixel no.",
            322 => c"BSCALE or TSCALn = 0.",
            323 => c"illegal axis length < 1",
            340 => c"not group table",
            341 => c"HDU already member of group",
            342 => c"group member not found",
            343 => c"group not found",
            344 => c"bad group id",
            345 => c"too many HDUs tracked",
            346 => c"HDU alread tracked",
            347 => c"bad Grouping option",
            348 => c"identical pointers (groups)",
            360 => c"malloc failed in parser",
            361 => c"file read error in parser",
            362 => c"null pointer arg (parser)",
            363 => c"empty line (parser)",
            364 => c"cannot unread > 1 line",
            365 => c"parser too deeply nested",
            366 => c"file open failed (parser)",
            367 => c"hit EOF (parser)",
            368 => c"bad argument (parser)",
            369 => c"unexpected token (parser)",
            401 => c"bad int to string conversion",
            402 => c"bad float to string conversion",
            403 => c"keyword value not integer",
            404 => c"keyword value not logical",
            405 => c"keyword value not floating pt",
            406 => c"keyword value not double",
            407 => c"bad string to int conversion",
            408 => c"bad string to float conversion",
            409 => c"bad string to double convert",
            410 => c"illegal datatype code value",
            411 => c"illegal no. of decimals",
            412 => c"datatype conversion overflow",
            413 => c"error compressing image",
            414 => c"error uncompressing image",
            420 => c"bad date or time conversion",
            431 => c"syntax error in expression",
            432 => c"expression result wrong type",
            433 => c"vector result too large",
            434 => c"missing output column",
            435 => c"bad data in parsed column",
            436 => c"output extension of wrong type",
            501 => c"WCS angle too large",
            502 => c"bad WCS coordinate",
            503 => c"error in WCS calculation",
            504 => c"bad WCS projection type",
            505 => c"WCS keywords not found",
            _ => c"unknown error status",
        }
    } else {
        c"unknown error status"
    };

    strcpy_safe(errtext, cast_slice(t.to_bytes_with_nul()));
}

/// The most a single stack entry can hold.  `push_message` splits at this
/// size and `ffgmsg` copies at most `FLEN_ERRMSG - 1`, so the two agree by
/// construction rather than by convention.
const ERRMSG_CHUNK: usize = FLEN_ERRMSG - 1;

/// One entry in the error stack.
///
/// Fixed-size, mirroring the C's `static char errbuff[errmsgsiz][81]`: the
/// whole stack is then `ERRMSGSIZ * ~82` bytes of static storage and `ffpmsg`
/// does not reach for the allocator, which matters because it is called while
/// handling failures.
///
/// The length is explicit rather than NUL-terminated.  The stack stores raw
/// bytes -- the C's ffpmsg() keeps whatever it is handed, and FITS diagnostics
/// routinely echo card bytes that are not valid UTF-8 -- and a malformed card
/// can contain an embedded NUL, which terminating would silently truncate.
#[derive(Debug, Clone, Copy)]
struct ErrorMessage {
    content: [u8; ERRMSG_CHUNK],
    len: u8,
    is_marker: bool,
}

impl ErrorMessage {
    fn new(chunk: &[u8]) -> Self {
        debug_assert!(chunk.len() <= ERRMSG_CHUNK);
        let mut content = [0u8; ERRMSG_CHUNK];
        content[..chunk.len()].copy_from_slice(chunk);
        Self {
            content,
            len: chunk.len() as u8,
            is_marker: false,
        }
    }

    fn marker() -> Self {
        Self {
            content: [0; ERRMSG_CHUNK],
            len: 0,
            is_marker: true,
        }
    }

    fn bytes(&self) -> &[u8] {
        &self.content[..self.len as usize]
    }
}

struct ErrorStack {
    messages: VecDeque<ErrorMessage>,
}

impl ErrorStack {
    fn new() -> Self {
        Self {
            messages: VecDeque::with_capacity(ERRMSGSIZ),
        }
    }

    fn push_message(&mut self, message: &[u8]) {
        let mut remaining = message;

        while !remaining.is_empty() {
            let chunk_size = core::cmp::min(ERRMSG_CHUNK, remaining.len());
            let (chunk, rest) = remaining.split_at(chunk_size);

            if self.messages.len() >= ERRMSGSIZ {
                self.messages.pop_front();
            }

            self.messages.push_back(ErrorMessage::new(chunk));
            remaining = rest;
        }
    }

    fn push_marker(&mut self) {
        if self.messages.len() >= ERRMSGSIZ {
            self.messages.pop_front();
        }

        self.messages.push_back(ErrorMessage::marker());
    }

    fn pop_message(&mut self) -> Option<ErrorMessage> {
        while let Some(msg) = self.messages.pop_front() {
            if !msg.is_marker {
                return Some(msg);
            }
        }
        None
    }

    fn clear_all(&mut self) {
        self.messages.clear();
    }

    fn clear_to_marker(&mut self) {
        while let Some(msg) = self.messages.pop_back() {
            if msg.is_marker {
                break;
            }
        }
    }

    fn clear_newest(&mut self) {
        self.messages.pop_back();
    }
}

/// put message on to error stack
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffpmsg(err_message: *const c_char) {
    // FFI WRAPPER
    ffxmsg(PUT_MESG, err_message.cast_mut());
}

/// put message on to error stack
pub fn ffpmsg_safe(err_message: &[c_char]) {
    let mut stack = ERROR_STACK.lock().unwrap();
    stack.push_message(cstr_bytes(err_message));
}

/// Push a message onto the error stack, taking it as a `CStr`.
///
/// See `ffpmsg_str` and `ffpmsg_slice` for the other spellings; those are
/// crate-private.
pub fn ffpmsg_cstr(err_message: &CStr) {
    let mut stack = ERROR_STACK.lock().unwrap();
    stack.push_message(err_message.to_bytes());
}

pub(crate) fn ffpmsg_slice(err_message: &[c_char]) {
    let mut stack = ERROR_STACK.lock().unwrap();
    stack.push_message(cstr_bytes(err_message));
}

/// The bytes of `s` up to its first NUL, or all of `s` if it has none.
fn cstr_bytes(s: &[c_char]) -> &[u8] {
    let b: &[u8] = cast_slice(s);
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    &b[..n]
}

pub(crate) fn ffpmsg_str(err_message: &str) {
    let mut stack = ERROR_STACK.lock().unwrap();
    stack.push_message(err_message.as_bytes());
}

///  write a marker to the stack.  It is then possible to pop only those
///  messages following the marker off of the stack, leaving the previous
///  messages unaffected.
///
///  The marker is ignored by the ffgmsg routine.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffpmrk() {
    // FFI WRAPPER
    ffpmrk_safe();
}

///  write a marker to the stack.  It is then possible to pop only those
///  messages following the marker off of the stack, leaving the previous
///  messages unaffected.
///
///  The marker is ignored by the ffgmsg routine.
pub fn ffpmrk_safe() {
    let mut stack = ERROR_STACK.lock().unwrap();
    stack.push_marker();
}

/// get oldest message from error stack, ignoring markers
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgmsg(err_message: *mut c_char) -> c_int {
    // FFI WRAPPER
    unsafe {
        ffxmsg(GET_MESG, err_message);
        c_int::from(*err_message)
    }
}

/// get oldest message from error stack, ignoring markers
pub fn ffgmsg_safe(err_message: &mut [c_char; FLEN_ERRMSG]) -> c_int {
    let mut stack = ERROR_STACK.lock().unwrap();

    if let Some(message) = stack.pop_message() {
        // Copy message to buffer
        let message_bytes = message.bytes();
        let copy_len = core::cmp::min(message_bytes.len(), FLEN_ERRMSG - 1);

        for (i, &byte) in message_bytes.iter().take(copy_len).enumerate() {
            err_message[i] = byte as c_char;
        }
        err_message[copy_len] = 0; // null terminate

        c_int::from(err_message[0])
    } else {
        err_message[0] = 0;
        0
    }
}

///  erase all messages in the error stack
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffcmsg() {
    // FFI WRAPPER
    let dummy = ptr::null_mut();
    ffxmsg(DEL_ALL, dummy);
}

///  erase all messages in the error stack
pub fn ffcmsg_safe() {
    let mut stack = ERROR_STACK.lock().unwrap();
    stack.clear_all();
}

/// erase newest messages in the error stack, stopping if a marker is found.
/// The marker is also erased in this case.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffcmrk() {
    // FFI WRAPPER
    ffcmrk_safe();
}

/// erase newest messages in the error stack, stopping if a marker is found.
/// The marker is also erased in this case.
pub fn ffcmrk_safe() {
    let mut stack = ERROR_STACK.lock().unwrap();
    stack.clear_to_marker();
}

/// general routine to get, put, or clear the error message stack.
/// Use a static array rather than allocating memory as needed for
/// the error messages because it is likely to be more efficient
/// and simpler to implement.
///
/// Action Code:
/// DelAll     1  delete all messages on the error stack
/// DelMark    2  delete messages back to and including the 1st marker
/// DelNewest  3  delete the newest message from the stack
/// GetMesg    4  pop and return oldest message, ignoring marks
/// PutMesg    5  add a new message to the stack
/// PutMark    6  add a marker to the stack
pub(crate) fn ffxmsg_safe(action: c_int, errmsg: Option<&mut [c_char; FLEN_ERRMSG]>) {
    let mut stack = ERROR_STACK.lock().unwrap();

    match action {
        DEL_ALL => {
            stack.clear_all();
        }
        DEL_MARK => {
            stack.clear_to_marker();
        }
        DEL_NEWEST => {
            stack.clear_newest();
        }
        GET_MESG => {
            if let Some(err_message) = errmsg {
                if let Some(message) = stack.pop_message() {
                    // Copy message to buffer
                    let message_bytes = message.bytes();
                    let copy_len = core::cmp::min(message_bytes.len(), FLEN_ERRMSG - 1);

                    for (i, &byte) in message_bytes.iter().take(copy_len).enumerate() {
                        err_message[i] = byte as c_char;
                    }
                    err_message[copy_len] = 0; // null terminate
                } else {
                    err_message[0] = 0; // empty string
                }
            }
        }
        PUT_MARK => {
            stack.push_marker();
        }
        _ => {
            // Unknown action, do nothing
        }
    }
}

/// general routine to get, put, or clear the error message stack.
/// Thread-safe implementation using Mutex-protected VecDeque.
///
/// Action Code:
/// DelAll     1  delete all messages on the error stack
/// DelMark    2  delete messages back to and including the 1st marker
/// DelNewest  3  delete the newest message from the stack
/// GetMesg    4  pop and return oldest message, ignoring marks
/// PutMesg    5  add a new message to the stack
/// PutMark    6  add a marker to the stack
pub(crate) fn ffxmsg(action: c_int, errmsg: *mut c_char) {
    let mut stack = ERROR_STACK.lock().unwrap();

    match action {
        DEL_ALL => {
            stack.clear_all();
        }
        DEL_MARK => {
            stack.clear_to_marker();
        }
        DEL_NEWEST => {
            stack.clear_newest();
        }
        GET_MESG => {
            if let Some(message) = stack.pop_message() {
                // Safely copy to the provided buffer
                if !errmsg.is_null() {
                    let message_bytes = message.bytes();
                    let copy_len = core::cmp::min(message_bytes.len(), FLEN_ERRMSG - 1);

                    unsafe {
                        ptr::copy_nonoverlapping(
                            message_bytes.as_ptr().cast::<c_char>(),
                            errmsg,
                            copy_len,
                        );
                        *errmsg.add(copy_len) = 0; // null terminate
                    }
                }
            } else if !errmsg.is_null() {
                unsafe {
                    *errmsg = 0; // empty string
                }
            }
        }
        PUT_MESG => {
            if !errmsg.is_null() {
                // Convert raw pointer to safe string
                let c_str = unsafe { CStr::from_ptr(errmsg) };
                stack.push_message(c_str.to_bytes());
            }
        }
        PUT_MARK => {
            stack.push_marker();
        }
        _ => {
            // Unknown action, do nothing
        }
    }
}

/// return the number of bytes per pixel associated with the datatype
pub(crate) fn ffpxsz(datatype: c_int) -> usize {
    if datatype == TBYTE {
        core::mem::size_of::<i8>()
    } else if datatype == TUSHORT {
        core::mem::size_of::<c_ushort>()
    } else if datatype == TSHORT {
        core::mem::size_of::<c_short>()
    } else if datatype == TULONG {
        core::mem::size_of::<c_ulong>()
    } else if datatype == TLONG {
        core::mem::size_of::<c_long>()
    } else if datatype == TINT {
        core::mem::size_of::<c_int>()
    } else if datatype == TUINT {
        core::mem::size_of::<c_uint>()
    } else if datatype == TFLOAT {
        core::mem::size_of::<f32>()
    } else if datatype == TDOUBLE {
        core::mem::size_of::<f64>()
    } else if datatype == TLOGICAL {
        core::mem::size_of::<i8>()
    } else {
        0
    }
}

/// Test that the keyword name conforms to the FITS standard.  Must contain
/// only capital letters, digits, minus or underscore chars.  Trailing spaces
/// are allowed.  If the input status value is less than zero, then the test
/// is modified so that upper or lower case letters are allowed, and no
/// error messages are printed if the keyword is not legal.
///
/// # Parameters
///
/// * `keyword` — (I) keyword name
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fftkey(keyword: *const c_char, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(keyword);

        fftkey_safe(keyword, status)
    }
}

/// Test that the keyword name conforms to the FITS standard.  Must contain
/// only capital letters, digits, minus or underscore chars.  Trailing spaces
/// are allowed.  If the input status value is less than zero, then the test
/// is modified so that upper or lower case letters are allowed, and no
/// error messages are printed if the keyword is not legal.
///
/// # Parameters
///
/// * `keyword` — (I) keyword name
/// * `status`  — (IO) error status
pub fn fftkey_safe(keyword: &[c_char], status: &mut c_int) -> c_int {
    let mut maxchr: usize;
    let mut spaces: c_int = 0;
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut testchar;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    maxchr = strlen_safe(keyword);
    if maxchr > 8 {
        maxchr = 8;
    }

    for ii in 0..maxchr {
        if *status == 0 {
            testchar = keyword[ii] as u8;
        } else {
            testchar = toupper(keyword[ii]) as u8;
        }
        if (testchar >= b'A' && testchar <= b'Z')
            || (testchar >= b'0' && testchar <= b'9')
            || testchar == b'-'
            || testchar == b'_'
        {
            if spaces > 0 {
                if *status == 0 {
                    /* don't print error message if status < 0  */

                    int_snprintf!(
                        &mut msg,
                        FLEN_ERRMSG,
                        "Keyword name contains embedded space(s): {:.8}",
                        slice_to_str!(keyword),
                    );

                    ffpmsg_slice(&msg);
                }
                *status = BAD_KEYCHAR;
                return *status;
            }
        } else if keyword[ii] == bb(b' ') {
            spaces = 1;
        } else {
            if *status == 0 {
                /* don't print error message if status < 0  */

                int_snprintf!(
                    &mut msg,
                    FLEN_ERRMSG,
                    "Character {} in this keyword is illegal: {:.8}",
                    ii + 1,
                    slice_to_str!(keyword),
                );

                ffpmsg_slice(&msg);

                /* explicitly flag the 2 most common cases */
                if keyword[ii] == 0 {
                    ffpmsg_str(" (This a NULL (0) character).");
                } else if keyword[ii] == 9 {
                    ffpmsg_str(" (This an ASCII TAB (9) character).");
                }
            }

            *status = BAD_KEYCHAR;
            return *status;
        }
    }
    *status
}

/// Test that the keyword card conforms to the FITS standard.  Must contain
/// only printable ASCII characters;
///
/// # Parameters
///
/// * `card`   — (I) keyword card to test
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fftrec(card: *mut c_char, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        raw_to_slice!(card);

        fftrec_safe(card, status)
    }
}

/// Test that the keyword card conforms to the FITS standard.  Must contain
/// only printable ASCII characters;
///
/// # Parameters
///
/// * `card`   — (I) keyword card to test
/// * `status` — (IO) error status
pub fn fftrec_safe(card: &[c_char], status: &mut c_int) -> c_int {
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    let maxchr = strlen_safe(card);

    for ii in 8..maxchr {
        if card[ii] < 32 || card[ii] > 126 {
            int_snprintf!(
                &mut msg,
                FLEN_ERRMSG,
                "Character {} in this keyword is illegal. Hex Value = {:X}",
                (ii + 1) as c_int,
                c_int::from(card[ii]),
            );

            let len = strlen_safe(&msg);
            if card[ii] == 0 {
                strncat_safe(&mut msg, cs!(c" (NULL char.)"), FLEN_ERRMSG - len - 1);
            } else if card[ii] == 9 {
                strncat_safe(&mut msg, cs!(c" (TAB char.)"), FLEN_ERRMSG - len - 1);
            } else if card[ii] == 10 {
                strncat_safe(&mut msg, cs!(c" (Line Feed char.)"), FLEN_ERRMSG - len - 1);
            } else if card[ii] == 11 {
                strncat_safe(&mut msg, cs!(c" (Vertical Tab)"), FLEN_ERRMSG - len - 1);
            } else if card[ii] == 12 {
                strncat_safe(&mut msg, cs!(c" (Form Feed char.)"), FLEN_ERRMSG - len - 1);
            } else if card[ii] == 13 {
                strncat_safe(&mut msg, cs!(c" (Carriage Return)"), FLEN_ERRMSG - len - 1);
            } else if card[ii] == 27 {
                strncat_safe(&mut msg, cs!(c" (Escape char.)"), FLEN_ERRMSG - len - 1);
            } else if card[ii] == 127 {
                strncat_safe(&mut msg, cs!(c" (Delete char.)"), FLEN_ERRMSG - len - 1);
            }
            ffpmsg_slice(&msg);
            strncpy_safe(&mut msg, card, 80);
            msg[80] = 0;
            ffpmsg_slice(&msg);

            *status = BAD_KEYCHAR;
            return *status;
        }
    }
    *status
}

/// convert string to upper case, in place.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffupch(string: *mut c_char) {
    // FFI WRAPPER
    unsafe {
        if !string.is_null() {
            // SAFETY: the C contract is a writable, NUL-terminated string.
            let len = CStr::from_ptr(string).to_bytes().len();
            let s = slice::from_raw_parts_mut(string, len + 1); //+1 for null char

            ffupch_safe(s);
        }
    }
}

/// convert string to upper case, in place.
pub fn ffupch_safe(string: &mut [c_char]) {
    let len = strlen_safe(string);
    for ii in 0..(len) {
        string[ii] = toupper(string[ii])
    }
}

/// Make a complete FITS 80-byte keyword card from the input name, value and
/// comment strings. Output card is null terminated without any trailing blanks.
///
/// # Parameters
///
/// * `keyname` — (I) keyword name
/// * `value`   — (I) keyword value
/// * `comm`    — (I) keyword comment
/// * `card`    — (O) constructed keyword card
/// * `status`  — (IO) status value
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffmkky(
    keyname: *const c_char,
    value: *const c_char,
    comm: *const c_char,
    card: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);

        if *status > 0 {
            return *status;
        }

        let card = slice::from_raw_parts_mut(card, FLEN_CARD);
        let value = cast_slice(CStr::from_ptr(value).to_bytes_with_nul());
        let keyname: &[c_char] = cast_slice(CStr::from_ptr(keyname).to_bytes_with_nul());

        let comm = match comm.is_null() {
            true => None,
            false => Some(cast_slice(CStr::from_ptr(comm).to_bytes_with_nul())),
        };

        ffmkky_safe(keyname, value, comm, card, status)
    }
}

/// Make a complete FITS 80-byte keyword card from the input name, value and
/// comment strings. Output card is null terminated without any trailing blanks.
///
/// # Parameters
///
/// * `keyname` — (I) keyword name
/// * `value`   — (I) keyword value
/// * `comm`    — (I) keyword comment
/// * `card`    — (O) constructed keyword card
/// * `status`  — (IO) status value
pub fn ffmkky_safe(
    keyname: &[c_char],
    value: &[c_char],
    comm: Option<&[c_char]>,
    card: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut namelen: usize;
    let mut len: usize;
    let mut ii: usize;
    let mut tmpname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut tmpname2: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];

    let mut tstatus: c_int = -(1);
    let mut nblank = 0;
    let mut ntoken = 0;
    let mut maxlen = 0;
    let mut specialchar: c_int = 0;

    if *status > 0 {
        return *status;
    }

    card[0] = 0;

    /* skip leading blanks in the name */
    while (keyname[nblank as usize]) == bb(b' ') {
        nblank += 1;
    }

    strncat_safe(
        &mut tmpname,
        &keyname[(nblank as usize)..],
        FLEN_KEYWORD - 1,
    );

    len = strlen_safe(value);
    namelen = strlen_safe(&tmpname);

    /* delete non-significant trailing blanks in the name */
    let mut j;
    if namelen > 0 {
        j = namelen - 1;
        while tmpname[j] == bb(b' ') {
            tmpname[j] = 0;
            j -= 1;
        }

        namelen = j + 1;
    }

    /* check that the name does not contain an '=' (equals sign) */
    if strchr_safe(&tmpname, bb(b'=')).is_some() {
        ffpmsg_str("Illegal keyword name; contains an equals sign (=)");
        ffpmsg_slice(&tmpname);
        *status = BAD_KEYCHAR;
        return *status;
    }

    if namelen <= 8 && fftkey_safe(&tmpname, &mut tstatus) <= 0 {
        /* a normal 8-char (or less) FITS keyword. */
        strcat_safe(card, &tmpname); /* copy keyword name to buffer */

        ii = namelen;
        while ii < 8 {
            card[ii] = bb(b' '); /* pad keyword name with spaces */
            ii += 1;
        }
        card[8] = bb(b'='); /* append '= ' in columns 9-10 */
        card[9] = bb(b' ');
        card[10] = 0; /* terminate the partial string */
        namelen = 10;
    } else if (FSTRNCMP(&tmpname, cs!(c"HIERARCH "), 9) == 0)
        || (FSTRNCMP(&tmpname, cs!(c"hierarch "), 9) == 0)
    {
        /* this is an explicit ESO HIERARCH keyword */

        strcat_safe(card, &tmpname); /* copy keyword name to buffer */

        if namelen + 3 + len > 80 {
            /* save 1 char by not putting a space before the equals sign */
            strcat_safe(card, cs!(c"= "));
            namelen += 2;
        } else {
            strcat_safe(card, cs!(c" = "));
            namelen += 3;
        }
    } else {
        /* scan the keyword name to determine the number and max length of the tokens */
        /* and test if any of the tokens contain nonstandard characters */

        strncat_safe(&mut tmpname2, &tmpname, FLEN_KEYWORD - 1);

        // NEW VERSION
        let tmpname3 = CStr::from_bytes_until_nul(cast_slice(&tmpname2))
            .unwrap()
            .to_str()
            .unwrap();
        tmpname3.split(' ').for_each(|token| {
            if token.len() > maxlen {
                maxlen = token.len();
            }

            /* name contains special characters? */
            tstatus = -1; /* suppress any error message */

            // Create a null-terminated version of the token
            let mut token_buf = [0u8; FLEN_KEYWORD];
            let token_bytes = token.as_bytes();
            for (i, &b) in token_bytes.iter().enumerate().take(FLEN_KEYWORD - 1) {
                token_buf[i] = b;
            }

            if fftkey_safe(cast_slice(&token_buf), &mut tstatus) > 0 {
                specialchar = 1;
            }

            ntoken += 1;
        });

        tstatus = -1; /* suppress any error message */

        /*      if (ntoken > 1) { */
        if ntoken > 0 {
            /*  temporarily change so that this case should always be true  */
            /* for now at least, treat all cases as an implicit ESO HIERARCH keyword. */
            /* This could  change if FITS is ever expanded to directly support longer keywords. */

            if namelen + 11 > FLEN_CARD - 1 {
                ffpmsg_str("The following keyword is too long to fit on a card:");
                ffpmsg_slice(keyname);
                *status = BAD_KEYCHAR;
                return *status;
            }
            strcat_safe(card, cs!(c"HIERARCH "));
            strcat_safe(card, &tmpname);
            namelen += 9;

            if namelen + 3 + len > 80 {
                /* save 1 char by not putting a space before the equals sign */
                strcat_safe(card, cs!(c"= "));
                namelen += 2;
            } else {
                strcat_safe(card, cs!(c" = "));
                namelen += 3;
            }
        } else if fftkey_safe(&tmpname, &mut tstatus) <= 0 {
            /* should never get here (at least for now) */
            /* allow keyword names longer than 8 characters */

            strncat_safe(card, &tmpname, FLEN_KEYWORD - 1);
            strcat_safe(card, cs!(c"= "));
            namelen += 2;
        } else {
            /* should never get here (at least for now) */
            ffpmsg_str("Illegal keyword name:");
            ffpmsg_slice(&tmpname);
            *status = BAD_KEYCHAR;
            return *status;
        }
    }

    if len > 0 {
        /* now process the value string */

        if value[0] == bb(b'\'') {
            /* is this a quoted string value? */
            if namelen > 77 {
                ffpmsg_str("The following keyword + value is too long to fit on a card:");
                ffpmsg_slice(keyname);
                ffpmsg_slice(value);
                *status = BAD_KEYCHAR;
                return *status;
            }

            strncat_safe(card, value, 80 - namelen); /* append the value string */
            len = cmp::min(80, namelen + len);

            /* restore the closing quote if it got truncated */
            if len == 80 {
                card[79] = bb(b'\'');
            }

            if let Some(comm) = comm
                && comm[0] != 0
                && len < 30
            {
                ii = len;
                while ii < 30 {
                    card[ii] = bb(b' '); /* fill with spaces to col 30 */
                    ii += 1;
                }
                card[30] = 0;
                len = 30;
            }
        } else {
            if namelen + len > 80 {
                ffpmsg_str("The following keyword + value is too long to fit on a card:");
                ffpmsg_slice(keyname);
                ffpmsg_slice(value);
                *status = BAD_KEYCHAR;
                return *status;
            } else if namelen + len < 30 {
                /* add spaces so field ends at least in col 30 */
                strncat_safe(card, cs!(c"                    "), 30 - (namelen + len));
            }

            strncat_safe(card, value, 80 - namelen); /* append the value string */
            len = cmp::min(80, namelen + len);
            len = cmp::max(30, len);
        }
        if let Some(comm) = comm
            && (len < 77)
            && (strlen_safe(comm) > 0)
        {
            /* room for a comment? */
            strcat_safe(card, cs!(c" / ")); /* append comment separator */
            strncat_safe(card, comm, 77 - len); /* append comment (what fits) */
        }
    } else if namelen == 10
    /* This case applies to normal keywords only */
    {
        card[8] = b' ' as c_char; /* keywords with no value have no '=' */
        if let Some(comm) = comm {
            strncat_safe(card, comm, 80 - namelen); /* append comment (what fits) */
        }
    }

    /* issue a warning if this keyword does not strictly conform to the standard
      HIERARCH convention, which requires,
        1) at least 2 tokens in the name,
    2) no tokens longer than 8 characters, and
    3) no special characters in any of the tokens */

    if ntoken == 1 || specialchar == 1 {
        ffpmsg_str("Warning: the following keyword does not conform to the HIERARCH convention");
        /*  ffpmsg(" (e.g., name is not hierarchical or contains non-standard characters)."); */
        ffpmsg_slice(card);
    }

    *status
}

/// replace the previously read card (i.e. starting 80 bytes before the
/// fptr.Fptr.nextkey position) with the contents of the input card.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `card`   — (I) card string value
/// * `status` — (IO) error status
pub(crate) fn ffmkey(fptr: &mut fitsfile, card: &[c_char], status: &mut c_int) -> c_int {
    let mut tcard: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut keylength;

    let card_len = FLEN_CARD - 1;

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }
    strncpy_safe(&mut tcard, card, card_len);
    tcard[card_len] = 0;
    let len = strlen_safe(&tcard);

    /* silently replace any illegal characters with a space */
    for ii in 0..len {
        if tcard[ii] < bb(b' ') || tcard[ii] > 126 {
            tcard[ii] = bb(b' ');
        }
    }

    let mut ii = len;
    /* fill card with spaces if necessary */
    while ii < card_len {
        tcard[ii] = bb(b' ');
        ii += 1
    }

    keylength = strcspn_safe(&tcard, cs!(c"="));
    if keylength == card_len {
        keylength = 8;
    }

    let mut ii = 0;
    /* make sure keyword name is uppercase */
    while ii < keylength {
        tcard[ii] = toupper(tcard[ii]);
        ii += 1
    }

    fftkey_safe(&tcard, status); /* test keyword name contains legal chars */

    /*  no need to do this any more, since any illegal characters have been removed
    fftrec(tcard, status);   */
    /* test rest of keyword for legal chars   */

    /* move position of keyword to be over written */

    ffmbyt_safe(
        fptr,
        (fptr.Fptr.nextkey) - card_len as LONGLONG,
        REPORT_EOF,
        status,
    );

    ffpbyt(
        fptr,
        card_len as LONGLONG,
        cast_slice_mut(&mut tcard),
        status,
    ); /* write the 80 byte card */
    *status
}

/// Construct a keyword name string by appending the index number to the root.
/// e.g., if root = "TTYPE" and value = 12 then keyname = "TTYPE12".
///
/// # Parameters
///
/// * `keyroot` — (I) root string for keyword name
/// * `value`   — (I) index number to be appended to root name
/// * `keyname` — (O) output root + index keyword name
/// * `status`  — (IO) error status
pub fn ffkeyn_safe(
    keyroot: &[c_char],
    value: c_int,
    keyname: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut suffix: [c_char; 16] = [0; 16]; /* initialize output name to null */

    keyname[0] = 0; /* initialize output name to null */
    let mut rootlen = strlen_safe(keyroot) as isize;
    if rootlen == 0 || value < 0 {
        *status = BAD_INDEX_KEY;
        return *status;
    }

    int_snprintf!(&mut suffix, 16, "{}", value); /* construct keyword suffix */

    strcpy_safe(keyname, keyroot); /* copy root string to name string */

    while rootlen > 0 && keyname[(rootlen - 1) as usize] == bb(b' ') {
        rootlen -= 1; /* remove trailing spaces in root name */
        keyname[rootlen as usize] = 0;
    }

    if strlen_safe(&suffix) + strlen_safe(keyname) > 8 {
        *status = BAD_INDEX_KEY;
        return *status;
    }

    strcat_safe(keyname, &suffix); /* append suffix to the root */
    *status
}

/// Construct a keyword name string by appending the index number to the root.
/// e.g., if root = "TTYPE" and value = 12 then keyname = "TTYPE12".
///
/// # Parameters
///
/// * `keyroot` — (I) root string for keyword name
/// * `value`   — (I) index number to be appended to root name
/// * `keyname` — (O) output root + index keyword name
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffkeyn(
    keyroot: *const c_char,
    value: c_int,
    keyname: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let keyname = slice::from_raw_parts_mut(keyname, FLEN_KEYWORD);

        raw_to_slice!(keyroot);

        ffkeyn_safe(keyroot, value, keyname, status)
    }
}

/// Construct a keyword name string by appending the root string to the index
/// number. e.g., if root = "TTYPE" and value = 12 then keyname = "12TTYPE".
///
/// # Parameters
///
/// * `value`   — (I) index number to be appended to root name
/// * `keyroot` — (I) root string for keyword name
/// * `keyname` — (O) output root + index keyword name
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffnkey(
    value: c_int,
    keyroot: *const c_char,
    keyname: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let keyname = slice::from_raw_parts_mut(keyname, 9); //Max 8 characters from bottom condition statement
        raw_to_slice!(keyroot);

        ffnkey_safe(value, keyroot, keyname, status)
    }
}

/// Construct a keyword name string by appending the root string to the index
/// number. e.g., if root = "TTYPE" and value = 12 then keyname = "12TTYPE".
///
/// # Parameters
///
/// * `value`   — (I) index number to be appended to root name
/// * `keyroot` — (I) root string for keyword name
/// * `keyname` — (O) output root + index keyword name
/// * `status`  — (IO) error status
pub fn ffnkey_safe(
    value: c_int,
    keyroot: &[c_char],
    keyname: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    keyname[0] = 0; /* initialize output name to null */
    let rootlen = strlen_safe(keyroot);

    if rootlen == 0 || rootlen > 7 || value < 0 {
        *status = 206;
        return *status;
    }

    int_snprintf!(keyname, FLEN_VALUE, "{}", value,); /* construct keyword prefix */

    if rootlen + strlen_safe(keyname) > 8 {
        *status = 206;
        return *status;
    }

    strcat_safe(keyname, keyroot); /* append root to the prefix */
    *status
}

/// ParSe the Value and Comment strings from the input header card string.
///
/// If the card contains a quoted string value, the returned value string
/// includes the enclosing quote characters.  If comm = NULL, don't return
/// the comment string.
///
/// # Parameters
///
/// * `card`   — (I) FITS header card (nominally 80 bytes long)
/// * `value`  — (O) value string parsed from the card
/// * `comm`   — (O) comment string parsed from the card
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffpsvc(
    card: *const c_char,
    value: *mut c_char,
    comm: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let card: &[c_char] = slice::from_raw_parts(card, FLEN_CARD);
        let value: &mut [c_char] = slice::from_raw_parts_mut(value, FLEN_VALUE);
        let status = status.as_mut().expect(NULL_MSG);

        let comm: Option<&mut [c_char; FLEN_COMMENT]> = match comm.is_null() {
            true => None,
            false => Some(
                slice::from_raw_parts_mut(comm, FLEN_COMMENT)
                    .try_into()
                    .unwrap(),
            ),
        };

        ffpsvc_safe(card, value, comm, status)
    }
}

/// ParSe the Value and Comment strings from the input header card string.
/// If the card contains a quoted string value, the returned value string
/// includes the enclosing quote characters.  If comm = NULL, don't return
/// the comment string.
///
/// # Parameters
///
/// * `card`   — (I) FITS header card (nominally 80 bytes long)
/// * `value`  — (O) value string parsed from the card
/// * `comm`   — (O) comment string parsed from the card
/// * `status` — (IO) error status
pub fn ffpsvc_safe(
    card: &[c_char],
    value: &mut [c_char],
    comm: Option<&mut [c_char; FLEN_COMMENT]>,
    status: &mut c_int,
) -> c_int {
    let mut valpos: usize;
    let mut strbuf: [c_char; 21] = [0; 21];

    if *status > 0 {
        return *status;
    }

    value[0] = 0;

    let mut comm = comm;
    if let Some(comm) = comm.as_deref_mut() {
        comm[0] = 0;
    }

    let cardlen = strlen_safe(card);

    if cardlen >= FLEN_CARD {
        strncpy_safe(&mut strbuf, card, 20);
        strbuf[20] = 0;
        ffpmsg_str("The card string starting with the chars below is too long:");
        ffpmsg_slice(&strbuf);

        *status = BAD_KEYCHAR;
        return *status;
    }

    /* support for ESO HIERARCH keywords; find the '=' */
    if FSTRNCMP(card, cs!(c"HIERARCH "), 9) == 0 {
        valpos = strcspn_safe(card, cs!(c"="));

        if valpos == cardlen {
            /* no value indicator ??? */
            if let Some(ref mut comm) = comm
                && cardlen > 8
            {
                //strcpy_safe(*comm, &card[8..]);

                for jj in (0..(cardlen - 8)).rev() {
                    /* replace trailing blanks with nulls */
                    if comm[jj] == bb(b' ') as c_char {
                        comm[jj] = 0;
                    } else {
                        break;
                    };
                }
            };
            return *status; /* no value indicator */
        }
        valpos += 1;
    } else if cardlen < 9
        || FSTRNCMP(card, cs!(c"COMMENT "), 8) == 0
        || FSTRNCMP(card, cs!(c"HISTORY "), 8) == 0
        || FSTRNCMP(card, cs!(c"END     "), 8) == 0
        || FSTRNCMP(card, cs!(c"CONTINUE"), 8) == 0
        || FSTRNCMP(card, cs!(c"        "), 8) == 0
    {
        /* keywords with no value */

        /*  no value, so the comment extends from cols 9 - 80  */
        if let Some(comm) = comm
            && cardlen > 8
        {
            strcpy_safe(comm, &card[8..]);

            for jj in (0..(cardlen - 8)).rev() {
                /* replace trailing blanks with nulls */
                if comm[jj] == bb(b' ') {
                    comm[jj] = 0;
                } else {
                    break;
                };
            }
        };

        return *status;
    } else if FSTRNCMP(&card[8..], cs!(c"= "), 2) == 0 {
        /* normal keyword with '= ' in cols 9-10 */

        valpos = 10; /* starting position of the value field */
    } else {
        valpos = strcspn_safe(card, cs!(c"="));
        if valpos == cardlen {
            /* no value indicator ??? */
            if let Some(comm) = comm
                && cardlen > 8
            {
                strcpy_safe(comm, &card[8..]);

                for jj in (0..(cardlen - 8)).rev() {
                    /* replace trailing blanks with nulls */
                    if comm[jj] == bb(b' ') {
                        comm[jj] = 0;
                    } else {
                        break;
                    };
                }
            };
            return *status; /* no value indicator */
        }
        valpos += 1; /* point to the position after the '=' */
    }

    let mut nblank = strspn_safe(&card[valpos..], cs!(c" ")); /* find number of leading blanks */

    if nblank + valpos == cardlen {
        /* the absence of a value string is legal, and simply indicates
           that the keyword value is undefined.  Don't write an error
           message in this case.
        */
        return *status;
    }

    let mut ii = valpos + nblank;

    if card[ii] == bb(b'/') {
        /* slash indicates start of the comment */
        ii += 1;
    } else if card[ii] == bb(b'\'') {
        /* is this a quoted string value? */

        value[0] = card[ii];
        let mut jj: usize = 1;
        ii += 1;
        while ii < cardlen && jj < FLEN_VALUE - 1 {
            if card[ii] == bb(b'\'') {
                /*  is this the closing quote?  */
                if card[ii + 1] == bb(b'\'') {
                    /* 2 successive quotes? */
                    value[jj] = card[ii];
                    ii += 1;
                    jj += 1;
                } else {
                    value[jj] = card[ii];
                    break; /* found the closing quote, so exit this loop  */
                };
            }

            if jj < FLEN_VALUE - 1 {
                value[jj] = card[ii]; /* copy the next character to the output */
            }

            ii += 1;
            jj += 1
        }

        if ii == cardlen || jj >= FLEN_VALUE - 1 {
            jj = cmp::min(jj, FLEN_VALUE - 2); /* don't exceed 70 char string length */
            value[jj] = bb(b'\''); /*  close the bad value string  */
            value[jj + 1] = 0; /*  terminate the bad value string  */
            ffpmsg_str("This keyword string value has no closing quote:");
            ffpmsg_slice(card);
            /*  May 2008 - modified to not fail on this minor error  */
            /*            return(*status = NO_QUOTE);  */
        } else {
            value[jj + 1] = 0; /*  terminate the good value string  */
            ii += 1; /*  point to the character following the value  */
        };
    } else if card[ii] == bb(b'(') {
        /* is this a complex value? */

        nblank = strcspn_safe(&card[ii..], cs!(c")")); /* find closing ) */

        if nblank == strlen_safe(&card[ii..]) || nblank >= FLEN_VALUE - 1 {
            ffpmsg_str("This complex keyword value has no closing ')' within range:");
            ffpmsg_slice(card);

            *status = NO_QUOTE;
            return *status;
        }
        nblank += 1;

        strncpy_safe(value, &card[ii..], nblank);
        value[nblank] = 0;
        ii += nblank;
    } else {
        /*  an integer, floating point, or logical FITS value string  */

        nblank = strcspn_safe(&card[ii..], cs!(c" /")); /* find the end of the token */

        if nblank >= FLEN_VALUE {
            /* This should not happen for correct input */
            nblank = FLEN_VALUE - 1;
        }
        strncpy_safe(value, &card[ii..], nblank);
        value[nblank] = 0;
        ii += nblank;
    }

    /*  now find the comment string, if any  */
    if let Some(comm) = comm {
        nblank = strspn_safe(&card[ii..], cs!(c" ")); /*  find next non-space character  */
        ii += nblank;

        if ii < 80 {
            if card[ii] == bb(b'/') {
                /*  ignore the slash separator  */
                ii += 1;
                if card[ii] == bb(b' ') {
                    /*  also ignore the following space  */
                    ii += 1;
                };
            }
            strncpy_safe(comm, &card[ii..], FLEN_COMMENT - 1); /*  copy the remaining characters  */
            comm[FLEN_COMMENT - 1] = 0;

            for jj in (0..strlen_safe(comm)).rev() {
                /* replace trailing blanks with nulls */
                if comm[jj] == bb(b' ') {
                    comm[jj] = 0;
                } else {
                    break;
                };
            }
        };
    }
    *status
}

/// 'Get Template HeaDer'
/// parse a template header line and create a formated
/// character string which is suitable for appending to a FITS header
///
/// # Parameters
///
/// * `tmplt`  — (I) input header template string
/// * `card`   — (O) returned FITS header record
/// * `hdtype` — (O) how to interpreter the returned card string
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgthd(
    tmplt: *const c_char,
    card: *mut c_char,
    hdtype: *mut c_int,
    /*
      -2 = modify the name of a keyword; the old keyword name
           is returned starting at address chars[0]; the new name
           is returned starting at address char[40] (to be consistent
           with the Fortran version).  Both names are null terminated.
      -1 = card contains the name of a keyword that is to be deleted
       0 = append this keyword if it doesn't already exist, or
           modify the value if the keyword already exists.
       1 = append this comment keyword ('HISTORY',
           'COMMENT', or blank keyword name)
       2  =  this is the END keyword; do not write it to the header
    */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let hdtype = hdtype.as_mut().expect(NULL_MSG);
        raw_to_slice!(tmplt);

        let card = slice::from_raw_parts_mut(card, FLEN_CARD)
            .try_into()
            .unwrap();

        ffgthd_safe(tmplt, card, hdtype, status)
    }
}

/// 'Get Template HeaDer'
/// parse a template header line and create a formated
/// character string which is suitable for appending to a FITS header
///
/// # Parameters
///
/// * `tmplt`  — (I) input header template string
/// * `card`   — (O) returned FITS header record
/// * `hdtype` — (O) how to interpreter the returned card string
/// * `status` — (IO) error status
pub fn ffgthd_safe(
    tmplt: &[c_char],
    card: &mut [c_char; FLEN_CARD],
    hdtype: &mut c_int,
    /*
      -2 = modify the name of a keyword; the old keyword name
           is returned starting at address chars[0]; the new name
           is returned starting at address char[40] (to be consistent
           with the Fortran version).  Both names are null terminated.
      -1 = card contains the name of a keyword that is to be deleted
       0 = append this keyword if it doesn't already exist, or
           modify the value if the keyword already exists.
       1 = append this comment keyword ('HISTORY',
           'COMMENT', or blank keyword name)
       2  =  this is the END keyword; do not write it to the header
    */
    status: &mut c_int,
) -> c_int {
    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut value: [c_char; 140] = [0; 140];
    let mut comment: [c_char; 140] = [0; 140];
    let mut tok: usize;
    let mut suffix: c_char;

    let mut tvalue: [c_char; 140] = [0; 140];
    let mut len;
    let vlen;
    let mut more: bool;
    let mut tstatus: c_int;
    let lentok1;
    let mut remainlen;
    let mut dval: f64;

    if *status > 0 {
        return *status;
    }

    card[0] = 0;
    *hdtype = 0;

    if FSTRNCMP(tmplt, cs!(c"        "), 8) == 0 {
        /* if first 8 chars of template are blank, then this is a comment */
        strncat_safe(card, tmplt, 80);
        *hdtype = 1;
        return *status;
    }

    tok = 0; /* point to start of template string 'tmplt' */

    keyname[0] = 0;
    value[0] = 0;
    comment[0] = 0;

    len = strspn_safe(&tmplt[tok..], cs!(c" ")); /* no. of spaces before keyword */
    tok += len;

    /* test for pecular case where token is a string of dashes */
    if strncmp_safe(&tmplt[tok..], cs!(c"--------------------"), 20) == 0 {
        *status = BAD_KEYCHAR;
        return *status;
    }

    if tmplt[tok] == bb(b'-') {
        /* is there a leading minus sign? */
        /* first token is name of keyword to be deleted or renamed */
        *hdtype = -1;
        tok += 1;
        len = strspn_safe(&tmplt[tok..], cs!(c" ")); /* no. of spaces before keyword */
        tok += len;

        len = strcspn_safe(&tmplt[tok..], cs!(c" =+")); /* length of name */
        if len >= FLEN_KEYWORD {
            *status = BAD_KEYCHAR;
            return *status;
        }

        lentok1 = len;
        strncat_safe(card, &tmplt[tok..], len);

        /*
          The HIERARCH convention supports non-standard characters
          in the keyword name, so don't always convert to upper case or
          abort if there are illegal characters in the name or if the
          name is greater than 8 characters long.
        */

        if len < 9
        /* this is possibly a normal FITS keyword name */
        {
            ffupch_safe(card);
            tstatus = 0;
            if fftkey_safe(card, &mut tstatus) > 0 {
                /* name contained non-standard characters, so reset */
                card[0] = 0;
                strncat_safe(card, &tmplt[tok..], len);
            }
        }

        tok += len;

        /* Check optional "+" indicator to delete multiple keywords */
        if tmplt[tok] == bb(b'+') && len < FLEN_KEYWORD {
            strcat_safe(card, cs!(c"+"));
            return *status;
        }

        /* second token, if present, is the new name for the keyword */

        len = strspn_safe(&tmplt[tok..], cs!(c" ")); /* no. of spaces before next token */
        tok += len;

        if tmplt[tok] == 0 || tmplt[tok] == bb(b'=') {
            return *status; /* no second token */
        }

        *hdtype = -2;
        len = strcspn_safe(&tmplt[tok..], cs!(c" ")); /* length of new name */
        /* this name has to fit on columns 41-80 of card,
        and first name must now fit in 1-40 */
        if lentok1 > 40 {
            card[0] = 0;
            *status = BAD_KEYCHAR;
            return *status;
        }
        if len > 40 {
            card[0] = 0;
            *status = BAD_KEYCHAR;
            return *status;
        }

        /* copy the new name to card + 40;  This is awkward, */
        /* but is consistent with the way the Fortran FITSIO works */
        strcat_safe(card, cs!(c"                                        "));
        strncpy_safe(&mut card[40..], &tmplt[tok..], len);
        card[80] = 0; /* necessary to add terminator in case len = 40 */

        /*
            The HIERARCH convention supports non-standard characters
            in the keyword name, so don't always convert to upper case or
            abort if there are illegal characters in the name or if the
            name is greater than 8 characters long.
        */

        if len < 9
        /* this is possibly a normal FITS keyword name */
        {
            ffupch_safe(&mut card[40..]);
            tstatus = 0;
            if fftkey_safe(&card[40..], &mut tstatus) > 0 {
                /* name contained non-standard characters, so reset */
                strncpy_safe(&mut card[40..], &tmplt[tok..], len);
            }
        }
    } else
    /* no negative sign at beginning of template */
    {
        /* get the keyword name token */

        len = strcspn_safe(&tmplt[tok..], cs!(c" =")); /* length of keyword name */
        if len >= FLEN_KEYWORD {
            *status = BAD_KEYCHAR;
            return *status;
        }

        strncat_safe(&mut keyname, &tmplt[tok..], len);

        /*
         The HIERARCH convention supports non-standard characters
         in the keyword name, so don't always convert to upper case or
         abort if there are illegal characters in the name or if the
         name is greater than 8 characters long.
        */

        if len < 9
        /* this is possibly a normal FITS keyword name */
        {
            ffupch_safe(&mut keyname);
            tstatus = 0;
            if fftkey_safe(&keyname, &mut tstatus) > 0 {
                /* name contained non-standard characters, so reset */
                keyname[0] = 0;
                strncat_safe(&mut keyname, &tmplt[tok..], len);
            }
        }

        if FSTRCMP(&keyname, cs!(c"END")) == 0 {
            strcpy_safe(card, cs!(c"END"));
            *hdtype = 2;
            return *status;
        }

        tok += len; /* move token pointer to end of the keyword */

        if FSTRCMP(&keyname, cs!(c"COMMENT")) == 0
            || FSTRCMP(&keyname, cs!(c"HISTORY")) == 0
            || FSTRCMP(&keyname, cs!(c"HIERARCH")) == 0
        {
            *hdtype = 1; /* simply append COMMENT and HISTORY keywords */
            strcpy_safe(card, &keyname);
            strncat_safe(card, &tmplt[tok..], FLEN_CARD - strlen_safe(&keyname) - 1);
            return *status;
        }

        /* look for the value token */
        len = strspn_safe(&tmplt[tok..], cs!(c" =")); /* spaces or = between name and value */
        tok += len;

        if tmplt[tok] == bb(b'\'') {
            /* is value enclosed in quotes? */
            more = true;
            remainlen = 139;
            while more {
                tok += 1; /* temporarily move past the quote char */
                len = strcspn_safe(&tmplt[tok..], cs!(c"'")); /* length of quoted string */
                tok -= 1;
                if len + 2 > remainlen {
                    *status = BAD_KEYCHAR;
                    return *status;
                }
                strncat_safe(&mut value, &tmplt[tok..], len + 2);
                remainlen -= len + 2;

                tok += len + 1;
                if tmplt[tok] != bb(b'\'') {
                    /* NO_QUOTE */
                    *status = NO_QUOTE;
                    return *status;
                }

                tok += 1;
                if tmplt[tok] != bb(b'\'') {
                    /* 2 quote chars = literal quote */
                    more = false;
                }
            }
        } else if tmplt[tok] == bb(b'/') || tmplt[tok] == 0
        /* There is no value */
        {
            strcat_safe(&mut value, cs!(c" "));
        } else
        /* not a quoted string value */
        {
            len = strcspn_safe(&tmplt[tok..], cs!(c" /")); /* length of value string */
            if len > 139 {
                *status = BAD_KEYCHAR;
                return *status;
            }
            strncat_safe(&mut value, &tmplt[tok..], len);
            if !((tmplt[tok] == bb(b'T') || tmplt[tok] == bb(b'F'))
                && (tmplt[tok + 1] == bb(b' ')
                    || tmplt[tok + 1] == bb(b'/')
                    || tmplt[tok + 1] == 0))
            {
                /* not a logical value */

                let mut tmp_suffix = 0;
                dval = strtod_safe(&value, &mut tmp_suffix); /* try to read value as number */
                suffix = value[tmp_suffix];

                if suffix != 0 && suffix != bb(b' ') && suffix != bb(b'/') {
                    /* value not recognized as a number; might be because it */
                    /* contains a 'd' or 'D' exponent character  */
                    strcpy_safe(&mut tvalue, &value);

                    if let Some(loc) = strchr_safe(&tvalue, bb(b'D')) {
                        tvalue[loc] = bb(b'E'); /*  replace D's with E's. */
                        dval = strtod_safe(&tvalue, &mut tmp_suffix); /* read value again */
                        suffix = value[tmp_suffix];
                    } else if let Some(loc) = strchr_safe(&tvalue, bb(b'd')) {
                        tvalue[loc] = bb(b'E'); /*  replace d's with E's. */
                        dval = strtod_safe(&tvalue, &mut tmp_suffix); /* read value again */
                        suffix = value[tmp_suffix];
                    } else if let Some(loc) = strchr_safe(&tvalue, bb(b'.')) {
                        tvalue[loc] = bb(b','); /*  replace period with a comma */
                        dval = strtod_safe(&tvalue, &mut tmp_suffix); /* read value again */
                        suffix = value[tmp_suffix];
                    }
                }

                if suffix != 0 && suffix != bb(b' ') && suffix != bb(b'/') {
                    /* value is not a number; must enclose it in quotes */
                    if len > 137 {
                        *status = BAD_KEYCHAR;
                        return *status;
                    }
                    strcpy_safe(&mut value, cs!(c"'"));
                    strncat_safe(&mut value, &tmplt[tok..], len);
                    strcat_safe(&mut value, cs!(c"'"));

                    /* the following useless statement stops the compiler warning */
                    /* that dval is not used anywhere */
                    if dval == 0.0 {
                        len += dval as usize;
                    }
                } else {
                    /* value is a number; convert any 'e' to 'E', or 'd' to 'D' */
                    let loc = strchr_safe(&value, bb(b'e'));
                    match loc {
                        Some(l) => {
                            value[l] = bb(b'E');
                        }
                        None => {
                            let loc_inner = strchr_safe(&value, bb(b'd'));
                            if let Some(loc_inner) = loc_inner {
                                value[loc_inner] = bb(b'D');
                            }
                        }
                    }
                }
            }
            tok += len;
        }

        len = strspn_safe(&tmplt[tok..], cs!(c" /")); /* no. of spaces between value and comment */
        tok += len;

        vlen = strlen_safe(&value);
        if vlen > 0 && vlen < 10 && value[0] == bb(b'\'') {
            /* pad quoted string with blanks so it is at least 8 chars long */
            value[vlen - 1] = 0;
            strncat_safe(&mut value, cs!(c"        "), 10 - vlen);
            strcat_safe(&mut value[9..], cs!(c"'"));
        }

        /* get the comment string */
        strncat_safe(&mut comment, &tmplt[tok..], 70);

        /* construct the complete FITS header card */
        ffmkky_safe(&keyname, &value, Some(&comment), card, status);
    }
    *status
}

/// Translate a keyword name to a new name, based on a set of patterns.
///
/// The user passes an array of patterns to be matched.  Input pattern
/// number `i` is `pattern[i][0]`, and output pattern number `i` is
/// `pattern[i][1]`.  Keywords are matched against the input patterns.  If a
/// match is found then the keyword is re-written according to the output
/// pattern.
///
/// Order is important.  The first match is accepted.  The fastest match
/// will be made when templates with the same first character are grouped
/// together.
///
/// Several characters have special meanings:
///
///    i,j - single digits, preserved in output template
///    n - column number of one or more digits, preserved in output template
///    m - generic number of one or more digits, preserved in output template
///    a - coordinate designator, preserved in output template
///    # - number of one or more digits
///    ? - any character
///    * - only allowed in first character position, to match all
///        keywords; only useful as last pattern in the list
///
/// i, j, n, and m are returned by the routine.
///
/// For example, the input pattern "iCTYPn" will match "1CTYP5" (if n_value
/// is 5); the output pattern "CTYPEi" will be re-written as "CTYPE1".
/// Notice that "i" is preserved.
///
/// The following output patterns are special
///
/// Special output pattern characters:
///
///   "-" - do not copy a keyword that matches the corresponding input pattern
///
///   "+" - copy the input unchanged
///
/// The inrec string could be just the 8-char keyword name, or the entire
/// 80-char header record.  Characters 9 = 80 in the input string simply get
/// appended to the translated keyword name.
///
/// If n_range = 0, then only keywords with 'n' equal to n_value will be
/// considered as a pattern match.  If n_range = +1, then all values of
/// 'n' greater than or equal to n_value will be a match, and if -1,
/// then values of 'n' less than or equal to n_value will match.
///
/// This routine was written by Craig Markwardt, GSFC
///
/// # Parameters
///
/// * `inrec`    — (I) input string
/// * `outrec`   — (O) output converted string, or
/// * `patterns` — (I) pointer to input / output string
/// * `npat`     — (I) number of templates passed
/// * `n_value`  — (I) base 'n' template value of interest
/// * `n_offset` — (I) offset to be applied to the 'n'
/// * `n_range`  — (I) controls range of 'n' template
/// * `pat_num`  — (O) matched pattern number (0 based) or -1
/// * `i`        — (O) value of i, if any, else 0
/// * `j`        — (O) value of j, if any, else 0
/// * `m`        — (O) value of m, if any, else 0
/// * `n`        — (O) value of n, if any, else 0
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_translate_keyword(
    inrec: *mut c_char,
    outrec: *mut c_char,
    /*     a null string if input does not  */
    /*     match any of the patterns */
    patterns: *const [*const c_char; 2],
    /*     templates */
    npat: c_int,
    n_value: c_int,
    n_offset: c_int,
    /*     value in the output string */
    n_range: c_int,
    /*     values of interest (-1,0, or +1) */
    pat_num: *mut c_int,
    i: *mut c_int,
    j: *mut c_int,
    m: *mut c_int,
    n: *mut c_int,

    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let pat_num = pat_num.as_mut();
        let i = i.as_mut();
        let j = j.as_mut();
        let m = m.as_mut();
        let n = n.as_mut();

        if *status > 0 {
            return *status;
        }

        if (inrec.is_null()) || (outrec.is_null()) {
            *status = NULL_INPUT_PTR;
            return *status;
        }

        let outrec = slice::from_raw_parts_mut(outrec, FLEN_CARD); // By definition must be a full record size
        let patterns = slice::from_raw_parts(patterns, npat as usize);
        let inrec = slice::from_raw_parts_mut(inrec, FLEN_CARD);

        let patterns_vec = patterns
            .iter()
            .map(|&p| [CStr::from_ptr(p[0]), CStr::from_ptr(p[1])])
            .collect::<Vec<_>>();

        fits_translate_keyword_safe(
            inrec,
            outrec,
            &patterns_vec,
            npat,
            n_value,
            n_offset,
            n_range,
            pat_num,
            i,
            j,
            m,
            n,
            status,
        )
    }
}

/// # Parameters
///
/// * `inrec`    — (I) input string
/// * `outrec`   — (O) output converted string, or
/// * `patterns` — (I) pointer to input / output string
/// * `npat`     — (I) number of templates passed
/// * `n_value`  — (I) base 'n' template value of interest
/// * `n_offset` — (I) offset to be applied to the 'n'
/// * `n_range`  — (I) controls range of 'n' template
/// * `pat_num`  — (O) matched pattern number (0 based) or -1
/// * `i`        — (O) value of i, if any, else 0
/// * `j`        — (O) value of j, if any, else 0
/// * `m`        — (O) value of m, if any, else 0
/// * `n`        — (O) value of n, if any, else 0
/// * `status`   — (IO) error status
pub fn fits_translate_keyword_safe(
    inrec: &mut [c_char],
    outrec: &mut [c_char],
    /*     a null string if input does not  */
    /*     match any of the patterns */
    patterns: &[[&CStr; 2]],
    /*     templates */
    npat: c_int,
    n_value: c_int,
    n_offset: c_int,
    /*     value in the output string */
    n_range: c_int,
    /*     values of interest (-1,0, or +1) */
    pat_num: Option<&mut c_int>,
    i: Option<&mut c_int>,
    j: Option<&mut c_int>,
    m: Option<&mut c_int>,
    n: Option<&mut c_int>,

    status: &mut c_int,
) -> c_int {
    let mut i1: c_int = 0;
    let mut j1: c_int = 0;
    let mut n1: c_int = 0;
    let mut m1: c_int = 0;
    let mut fac: c_int;
    let mut a: c_char = bb(b' ');
    let mut oldp: c_char;
    let mut c: c_char;
    let mut s: c_char;
    let mut ip: usize;
    let mut ic: usize;
    let mut pat: c_int = 0;
    let mut pass = false;
    let mut firstfail;

    if *status > 0 {
        return *status;
    }

    outrec[0] = 0;

    /*
      if (*inrec == 0) return 0;
    */

    if inrec[0] == 0 {
        /* expand to full 8 char blank keyword name */
        strcpy_safe(inrec, cs!(c"        "));
    }

    oldp = 0;
    firstfail = false;

    let mut spat;

    /* ===== Pattern match stage */
    pat = 0;
    while pat < npat {
        spat = cast_slice(patterns[pat as usize][0].to_bytes_with_nul());

        i1 = 0;
        j1 = 0;
        m1 = -1;
        n1 = -1;
        a = bb(b' '); /* Initialize the place-holders */
        pass = false;

        /* Pass the wildcard pattern */
        if spat[0] == bb(b'*') {
            pass = true;
            break;
        }

        /* Optimization: if we have seen this initial pattern character before,
        then it must have failed, and we can skip the pattern */
        if firstfail && spat[0] == oldp {
            pat += 1;
            continue;
        }
        oldp = spat[0];

        /*
        ip = index of pattern character being matched
        ic = index of keyname character being matched
        firstfail = 1 if we fail on the first characteor (0=not)
         */
        ip = 0;
        ic = 0;
        firstfail = true;
        while (spat[ip] != 0) && (ic < 8) {
            c = inrec[ic];
            s = spat[ip];

            if s == bb(b'i') {
                /* Special pattern: 'i' placeholder */
                if isdigit_safe(c) {
                    i1 = c_int::from(c - bb(b'0'));
                    pass = true;
                }
            } else if s == bb(b'j') {
                /* Special pattern: 'j' placeholder */
                if isdigit_safe(c) {
                    j1 = c_int::from(c - bb(b'0'));
                    pass = true;
                }
            } else if (s == bb(b'n')) || (s == bb(b'm')) || (s == bb(b'#')) {
                /* Special patterns: multi-digit number */
                let mut val: c_int = 0;
                pass = false;
                if isdigit_safe(c) {
                    pass = true; /* NOTE, could fail below */

                    /* Parse decimal number */
                    while ic < 8 && isdigit_safe(c) {
                        val = val * 10 + c_int::from(c - bb(b'0'));
                        ic += 1;
                        c = inrec[ic];
                    }
                    ic -= 1;
                    c = inrec[ic];

                    if s == bb(b'n') {
                        /* Is it a column number? */
                        if val >= 1 && val <= 999 &&                    /* Row range check */
         (((n_range == 0) && (val == n_value)) ||     /* Strict equality */
          ((n_range == -1) && (val <= n_value)) ||    /* n <= n_value */
          ((n_range == 1) && (val >= n_value)))
                        {
                            /* n >= n_value */
                            n1 = val;
                        } else {
                            pass = false;
                        }
                    } else if s == bb(b'm') {
                        /* Generic number */
                        m1 = val;
                    }
                }
            } else if s == bb(b'a') {
                /* Special pattern: coordinate designator */
                if isupper(c) || c == bb(b' ') {
                    a = c;
                    pass = true;
                }
            } else if s == bb(b'?') {
                /* Match any individual character */
                pass = true;
            } else if c == s {
                /* Match a specific character */
                pass = true;
            } else {
                /* FAIL */
                pass = false;
            }
            if !pass {
                break;
            }
            ip += 1;
            ic += 1;
            firstfail = false;
        }

        /* Must pass to the end of the keyword.  No partial matches allowed */
        if pass && (ic >= 8 || inrec[ic] == bb(b' ')) {
            break;
        }

        pat += 1;
    }

    /* Transfer the pattern-matched numbers to the output parameters */

    if let Some(i) = i {
        *i = i1;
    }
    if let Some(j) = j {
        *j = j1;
    }
    if let Some(n) = n {
        *n = n1;
    }
    if let Some(m) = m {
        *m = m1;
    }
    if let Some(pat_num) = pat_num {
        *pat_num = pat;
    }

    /* ===== Keyword rewriting and output stage */
    spat = cast_slice((patterns[pat as usize][1]).to_bytes_with_nul());

    /* Return case: explicit deletion, return '-' */
    if pass && strcmp_safe(spat, cs!(c"--")) == 0 {
        strcpy_safe(outrec, cs!(c"-"));
        strncat_safe(outrec, inrec, 8);
        outrec[9] = 0;

        i1 = 8;
        while i1 > 1 && outrec[i1 as usize] == bb(b' ') {
            outrec[i1 as usize] = 0;
            i1 -= 1;
        }
        return 0;
    }

    /* Return case: no match, or do-not-transfer pattern */
    if !pass || spat[0] == 0 || strcmp_safe(spat, cs!(c"-")) == 0 {
        return 0;
    }
    /* A match: we start by copying the input record to the output */
    strcpy_safe(outrec, inrec);

    /* Return case: return the input record unchanged */
    if spat[0] == bb(b'+') {
        return 0;
    }

    /* Final case: a new output pattern */
    ic = 0;
    ip = 0;
    while spat[ip] != 0 {
        s = spat[ip];
        if s == bb(b'i') {
            outrec[ic] = (i1 + c_int::from(bb(b'0'))) as c_char;
        } else if s == bb(b'j') {
            outrec[ic] = (j1 + c_int::from(bb(b'0'))) as c_char;
        } else if s == bb(b'n') {
            if n1 == -1 {
                n1 = n_value;
            }
            if n1 > 0 {
                n1 += n_offset;
                fac = 1;
                while n1 / fac > 0 {
                    fac *= 10
                }

                fac /= 10;
                while fac > 0 {
                    outrec[ic] = (((n1 / fac) % 10) + c_int::from(bb(b'0'))) as c_char;
                    fac /= 10;
                    ic += 1;
                }
                ic -= 1;
            }
        } else if s == bb(b'm') && m1 >= 0 {
            fac = 1;
            while m1 / fac > 0 {
                fac *= 10
            }
            fac /= 10;
            while fac > 0 {
                outrec[ic] = (((m1 / fac) % 10) + c_int::from(bb(b'0'))) as c_char;
                fac /= 10;
                ic += 1;
            }
            ic -= 1;
        } else if s == bb(b'a') {
            outrec[ic] = a;
        } else {
            outrec[ic] = s;
        }

        ip += 1;
        ic += 1;
    }

    /* Pad the keyword name with spaces */
    while ic < 8 {
        outrec[ic] = bb(b' ');
        ic += 1;
    }

    *status
}

/// Copy relevant keywords from the table header into the newly
/// created primary array header.  Convert names of keywords where
/// appropriate.  See fits_translate_keyword() for the definitions.
///
/// Translation begins at header record number 'firstkey', and
/// continues to the end of the header.
///
/// This routine was written by Craig Markwardt, GSFC
///
/// # Parameters
///
/// * `infptr`   — (I) pointer to input HDU
/// * `outfptr`  — (I) pointer to output HDU
/// * `firstkey` — (I) first HDU record number to start with
/// * `patterns` — (I) pointer to input / output keyword templates
/// * `npat`     — (I) number of templates passed
/// * `n_value`  — (I) base 'n' template value of interest
/// * `n_offset` — (I) offset to be applied to the 'n' value in the output string
/// * `n_range`  — (I) controls range of 'n' template values of interest (-1,0, or +1)
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_translate_keywords(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    firstkey: c_int,
    patterns: *const [*const c_char; 2],
    npat: c_int,
    n_value: c_int,
    n_offset: c_int,
    n_range: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);

        let patterns = slice::from_raw_parts(patterns, npat as usize);

        let patterns_vec = patterns
            .iter()
            .map(|&p| [CStr::from_ptr(p[0]), CStr::from_ptr(p[1])])
            .collect::<Vec<_>>();

        fits_translate_keywords_safe(
            infptr,
            outfptr,
            firstkey,
            &patterns_vec,
            npat,
            n_value,
            n_offset,
            n_range,
            status,
        )
    }
}

/// Copy relevant keywords from the table header into the newly
/// created primary array header.  Convert names of keywords where
/// appropriate.  See fits_translate_keyword() for the definitions.
///
/// Translation begins at header record number 'firstkey', and
/// continues to the end of the header.
///
/// This routine was written by Craig Markwardt, GSFC
///
/// # Parameters
///
/// * `infptr`   — (I) pointer to input HDU
/// * `outfptr`  — (I) pointer to output HDU
/// * `firstkey` — (I) first HDU record number to start with
/// * `patterns` — (I) pointer to input / output keyword templates
/// * `npat`     — (I) number of templates passed
/// * `n_value`  — (I) base 'n' template value of interest
/// * `n_offset` — (I) offset to be applied to the 'n' value in the output string
/// * `n_range`  — (I) controls range of 'n' template values of interest (-1,0, or +1)
/// * `status`   — (IO) error status
pub fn fits_translate_keywords_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    firstkey: c_int,
    patterns: &[[&CStr; 2]],
    npat: c_int,
    n_value: c_int,
    n_offset: c_int,
    n_range: c_int,
    status: &mut c_int,
) -> c_int {
    let mut nkeys: c_int = 0;
    let mut nmore: c_int = 0;
    let mut rec: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut i: c_int = 0;
    let mut j: c_int = 0;
    let mut n: c_int = 0;
    let mut m: c_int = 0;
    let mut pat_num: c_int = 0;
    let mut maxchr: usize;
    let mut outrec: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    if *status > 0 {
        return *status;
    }

    ffghsp_safe(infptr, Some(&mut nkeys), Some(&mut nmore), status); /* get number of keywords */

    let mut nrec = firstkey;
    while (*status == 0) && (nrec <= nkeys) {
        outrec[0] = 0;

        ffgrec_safe(infptr, nrec, Some(&mut rec), status);

        /* silently overlook any illegal ASCII characters in the value or */
        /* comment fields of the record. It is usually not appropriate to */
        /* abort the process because of this minor transgression of the FITS rules. */
        /* Set the offending character to a blank */

        maxchr = strlen_safe(&rec);
        for ii in 8..maxchr {
            if rec[ii] < 32 || rec[ii] > 126 {
                rec[ii] = bb(b' ');
            }
        }

        fits_translate_keyword_safe(
            &mut rec,
            &mut outrec,
            patterns,
            npat,
            n_value,
            n_offset,
            n_range,
            Some(&mut pat_num),
            Some(&mut i),
            Some(&mut j),
            Some(&mut m),
            Some(&mut n),
            status,
        );

        if *status == 0 {
            if outrec[0] == bb(b'-') {
                /* prefix -KEYNAME means delete */

                /* Preserve only the keyword portion of name */
                outrec[9] = 0;
                let mut i1 = 8;
                while i1 > 1 && outrec[i1] == bb(b' ') {
                    outrec[i1] = 0;
                    i1 -= 1;
                }

                ffpmrk_safe();
                ffdkey_safe(outfptr, &outrec[1..], status); /* delete the keyword */
                if *status == 0 {
                    let mut nkeys1 = 0;
                    /* get number of keywords again in case of change*/
                    ffghsp_safe(infptr, Some(&mut nkeys1), Some(&mut nmore), status);
                    if nkeys1 != nkeys {
                        nrec -= 1;
                        nkeys = nkeys1;
                    }
                }
                *status = 0;
                ffcmrk_safe();
            } else if outrec[0] != 0 {
                ffprec_safe(outfptr, &outrec, status); /* copy the keyword */
            }
        }
        rec[8] = 0;
        outrec[8] = 0;
        nrec += 1;
    }

    *status
}

/// Copy relevant keywords from the pixel list table header into a newly
/// created primary array header.  Convert names of keywords where
/// appropriate.  See fits_translate_pixkeyword() for the definitions.
///
/// Translation begins at header record number 'firstkey', and
/// continues to the end of the header.
///
/// # Parameters
///
/// * `infptr`   — (I) pointer to input HDU
/// * `outfptr`  — (I) pointer to output HDU
/// * `firstkey` — (I) first HDU record number to start with
/// * `naxis`    — (I) number of axes in the image
/// * `colnum`   — (I) numbers of the columns to be binned
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_copy_pixlist2image(
    infptr: *mut fitsfile,
    outfptr: *mut fitsfile,
    firstkey: c_int,
    naxis: c_int,
    colnum: *const c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let outfptr = outfptr.as_mut().expect(NULL_MSG);
        let col_num = slice::from_raw_parts(colnum, 4);

        fits_copy_pixlist2image_safe(infptr, outfptr, firstkey, naxis, col_num, status)
    }
}

/// Copy relevant keywords from the pixel list table header into a newly
/// created primary array header.  Convert names of keywords where
/// appropriate.  See fits_translate_pixkeyword() for the definitions.
///
/// Translation begins at header record number 'firstkey', and
/// continues to the end of the header.
///
/// # Parameters
///
/// * `infptr`   — (I) pointer to input HDU
/// * `outfptr`  — (I) pointer to output HDU
/// * `firstkey` — (I) first HDU record number to start with
/// * `naxis`    — (I) number of axes in the image
/// * `colnum`   — (I) numbers of the columns to be binned
/// * `status`   — (IO) error status
pub fn fits_copy_pixlist2image_safe(
    infptr: &mut fitsfile,
    outfptr: &mut fitsfile,
    firstkey: c_int,
    naxis: c_int,
    colnum: &[c_int],
    status: &mut c_int,
) -> c_int {
    let mut nkeys: c_int = 0;
    let mut nmore: c_int = 0;
    let mut rec: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut outrec: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut pat_num: c_int = 0;
    let mut iret: c_int = 0;
    let mut jret: c_int = 0;
    let mut nret: c_int = 0;
    let mut mret: c_int = 0;
    let mut lret: c_int = 0;

    let patterns: [[*const c_char; 2]; 99] = [
        [c"TCTYPn".as_ptr(), c"CTYPEn".as_ptr()],
        [c"TCTYna".as_ptr(), c"CTYPEna".as_ptr()],
        [c"TCUNIn".as_ptr(), c"CUNITn".as_ptr()],
        [c"TCUNna".as_ptr(), c"CUNITna".as_ptr()],
        [c"TCRVLn".as_ptr(), c"CRVALn".as_ptr()],
        [c"TCRVna".as_ptr(), c"CRVALna".as_ptr()],
        [c"TCDLTn".as_ptr(), c"CDELTn".as_ptr()],
        [c"TCDEna".as_ptr(), c"CDELTna".as_ptr()],
        [c"TCRPXn".as_ptr(), c"CRPIXn".as_ptr()],
        [c"TCRPna".as_ptr(), c"CRPIXna".as_ptr()],
        [c"TCROTn".as_ptr(), c"CROTAn".as_ptr()],
        [c"TPn_ma".as_ptr(), c"PCn_ma".as_ptr()],
        [c"TPCn_m".as_ptr(), c"PCn_ma".as_ptr()],
        [c"TCn_ma".as_ptr(), c"CDn_ma".as_ptr()],
        [c"TCDn_m".as_ptr(), c"CDn_ma".as_ptr()],
        [c"TVn_la".as_ptr(), c"PVn_la".as_ptr()],
        [c"TPVn_l".as_ptr(), c"PVn_la".as_ptr()],
        [c"TSn_la".as_ptr(), c"PSn_la".as_ptr()],
        [c"TPSn_l".as_ptr(), c"PSn_la".as_ptr()],
        [c"TWCSna".as_ptr(), c"WCSNAMEa".as_ptr()],
        [c"TCNAna".as_ptr(), c"CNAMEna".as_ptr()],
        [c"TCRDna".as_ptr(), c"CRDERna".as_ptr()],
        [c"TCSYna".as_ptr(), c"CSYERna".as_ptr()],
        [c"LONPna".as_ptr(), c"LONPOLEa".as_ptr()],
        [c"LATPna".as_ptr(), c"LATPOLEa".as_ptr()],
        [c"EQUIna".as_ptr(), c"EQUINOXa".as_ptr()],
        [c"MJDOBn".as_ptr(), c"MJD-OBS".as_ptr()],
        [c"MJDAn".as_ptr(), c"MJD-AVG".as_ptr()],
        [c"DAVGn".as_ptr(), c"DATE-AVG".as_ptr()],
        [c"RADEna".as_ptr(), c"RADESYSa".as_ptr()],
        [c"RFRQna".as_ptr(), c"RESTFRQa".as_ptr()],
        [c"RWAVna".as_ptr(), c"RESTWAVa".as_ptr()],
        [c"SPECna".as_ptr(), c"SPECSYSa".as_ptr()],
        [c"SOBSna".as_ptr(), c"SSYSOBSa".as_ptr()],
        [c"SSRCna".as_ptr(), c"SSYSSRCa".as_ptr()],
        /* preserve common keywords */
        [c"LONPOLEa".as_ptr(), c"+".as_ptr()],
        [c"LATPOLEa".as_ptr(), c"+".as_ptr()],
        [c"EQUINOXa".as_ptr(), c"+".as_ptr()],
        [c"EPOCH".as_ptr(), c"+".as_ptr()],
        [c"MJD-????".as_ptr(), c"+".as_ptr()],
        [c"DATE????".as_ptr(), c"+".as_ptr()],
        [c"TIME????".as_ptr(), c"+".as_ptr()],
        [c"RADESYSa".as_ptr(), c"+".as_ptr()],
        [c"RADECSYS".as_ptr(), c"+".as_ptr()],
        [c"TELESCOP".as_ptr(), c"+".as_ptr()],
        [c"INSTRUME".as_ptr(), c"+".as_ptr()],
        [c"OBSERVER".as_ptr(), c"+".as_ptr()],
        [c"OBJECT".as_ptr(), c"+".as_ptr()],
        /* Delete general table column keywords */
        [c"XTENSION".as_ptr(), c"-".as_ptr()],
        [c"BITPIX".as_ptr(), c"-".as_ptr()],
        [c"NAXIS".as_ptr(), c"-".as_ptr()],
        [c"NAXISi".as_ptr(), c"-".as_ptr()],
        [c"PCOUNT".as_ptr(), c"-".as_ptr()],
        [c"GCOUNT".as_ptr(), c"-".as_ptr()],
        [c"TFIELDS".as_ptr(), c"-".as_ptr()],
        [c"TDIM#".as_ptr(), c"-".as_ptr()],
        [c"THEAP".as_ptr(), c"-".as_ptr()],
        [c"EXTNAME".as_ptr(), c"-".as_ptr()],
        [c"EXTVER".as_ptr(), c"-".as_ptr()],
        [c"EXTLEVEL".as_ptr(), c"-".as_ptr()],
        [c"CHECKSUM".as_ptr(), c"-".as_ptr()],
        [c"DATASUM".as_ptr(), c"-".as_ptr()],
        [c"NAXLEN".as_ptr(), c"-".as_ptr()],
        [c"AXLEN#".as_ptr(), c"-".as_ptr()],
        [c"CPREF".as_ptr(), c"-".as_ptr()],
        /* Delete table keywords related to other columns */
        [c"T????#a".as_ptr(), c"-".as_ptr()],
        [c"TC??#a".as_ptr(), c"-".as_ptr()],
        [c"T??#_#".as_ptr(), c"-".as_ptr()],
        [c"TWCS#a".as_ptr(), c"-".as_ptr()],
        [c"LONP#a".as_ptr(), c"-".as_ptr()],
        [c"LATP#a".as_ptr(), c"-".as_ptr()],
        [c"EQUI#a".as_ptr(), c"-".as_ptr()],
        [c"MJDOB#".as_ptr(), c"-".as_ptr()],
        [c"MJDA#".as_ptr(), c"-".as_ptr()],
        [c"RADE#a".as_ptr(), c"-".as_ptr()],
        [c"DAVG#".as_ptr(), c"-".as_ptr()],
        [c"iCTYP#".as_ptr(), c"-".as_ptr()],
        [c"iCTY#a".as_ptr(), c"-".as_ptr()],
        [c"iCUNI#".as_ptr(), c"-".as_ptr()],
        [c"iCUN#a".as_ptr(), c"-".as_ptr()],
        [c"iCRVL#".as_ptr(), c"-".as_ptr()],
        [c"iCDLT#".as_ptr(), c"-".as_ptr()],
        [c"iCRPX#".as_ptr(), c"-".as_ptr()],
        [c"iCTY#a".as_ptr(), c"-".as_ptr()],
        [c"iCUN#a".as_ptr(), c"-".as_ptr()],
        [c"iCRV#a".as_ptr(), c"-".as_ptr()],
        [c"iCDE#a".as_ptr(), c"-".as_ptr()],
        [c"iCRP#a".as_ptr(), c"-".as_ptr()],
        [c"ijPC#a".as_ptr(), c"-".as_ptr()],
        [c"ijCD#a".as_ptr(), c"-".as_ptr()],
        [c"iV#_#a".as_ptr(), c"-".as_ptr()],
        [c"iS#_#a".as_ptr(), c"-".as_ptr()],
        [c"iCRD#a".as_ptr(), c"-".as_ptr()],
        [c"iCSY#a".as_ptr(), c"-".as_ptr()],
        [c"iCROT#".as_ptr(), c"-".as_ptr()],
        [c"WCAX#a".as_ptr(), c"-".as_ptr()],
        [c"WCSN#a".as_ptr(), c"-".as_ptr()],
        [c"iCNA#a".as_ptr(), c"-".as_ptr()],
        [c"*".as_ptr(), c"+".as_ptr()],
    ]; /* copy all other keywords */

    if *status > 0 {
        return *status;
    }

    let npat = patterns.len(); // sizeof(patterns)/sizeof(patterns[0][0])/2

    ffghsp_safe(infptr, Some(&mut nkeys), Some(&mut nmore), status); /* get number of keywords */

    for nrec in firstkey..=nkeys {
        outrec[0] = 0;

        ffgrec_safe(infptr, nrec, Some(&mut rec), status);

        fits_translate_pixkeyword(
            &rec,
            &mut outrec,
            &patterns,
            npat,
            naxis,
            colnum,
            &mut pat_num,
            &mut iret,
            &mut jret,
            &mut nret,
            &mut mret,
            &mut lret,
            status,
        );

        if outrec[0] != 0 {
            ffprec_safe(outfptr, &outrec, status); /* copy the keyword */
        }

        rec[8] = 0;
        outrec[8] = 0;
    }

    *status
}

/// Translate a keyword name to a new name, based on a set of patterns.
/// The user passes an array of patterns to be matched.  Input pattern
/// number `i` is `pattern[i][0]`, and output pattern number `i` is
/// `pattern[i][1]`.  Keywords are matched against the input patterns.  If a
/// match is found then the keyword is re-written according to the output
/// pattern.
///
/// Order is important.  The first match is accepted.  The fastest match
/// will be made when templates with the same first character are grouped
/// together.
///
/// Several characters have special meanings:
///
///    i,j - single digits, preserved in output template
///    n, m - column number of one or more digits, preserved in output template
///    k - generic number of one or more digits, preserved in output template
///    a - coordinate designator, preserved in output template
///    # - number of one or more digits
///    ? - any character
///    * - only allowed in first character position, to match all
///        keywords; only useful as last pattern in the list
///
/// i, j, n, and m are returned by the routine.
///
/// For example, the input pattern "iCTYPn" will match "1CTYP5" (if n_value
/// is 5); the output pattern "CTYPEi" will be re-written as "CTYPE1".
/// Notice that "i" is preserved.
///
/// The following output patterns are special
///
/// Special output pattern characters:
///
///   "-" - do not copy a keyword that matches the corresponding input pattern
///
///   "+" - copy the input unchanged
///
/// The inrec string could be just the 8-char keyword name, or the entire
/// 80-char header record.  Characters 9 = 80 in the input string simply get
/// appended to the translated keyword name.
///
/// If n_range = 0, then only keywords with 'n' equal to n_value will be
/// considered as a pattern match.  If n_range = +1, then all values of
/// 'n' greater than or equal to n_value will be a match, and if -1,
/// then values of 'n' less than or equal to n_value will match.
///
/// # Parameters
///
/// * `inrec`    — (I) input string
/// * `outrec`   — (O) output converted string, or
/// * `patterns` — (I) pointer to input / output string
/// * `npat`     — (I) number of templates passed
/// * `_naxis`   — (I) number of columns to be binned
/// * `colnum`   — (I) numbers of the columns to be binned
/// * `pat_num`  — (O) matched pattern number (0 based) or -1
/// * `status`   — (IO) error status
pub(crate) fn fits_translate_pixkeyword(
    inrec: &[c_char],
    outrec: &mut [c_char],
    /*     a null string if input does not  */
    /*     match any of the patterns */
    patterns: &[[*const c_char; 2]],
    /*     templates */
    npat: usize,
    _naxis: c_int,
    colnum: &[c_int],
    pat_num: &mut c_int,
    i: &mut c_int,
    j: &mut c_int,
    n: &mut c_int,
    m: &mut c_int,
    l: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut i1: c_int = 0;
    let mut j1: c_int = 0;
    let mut val: c_int;
    let mut fac: c_int;
    let mut nval: c_int = 0;
    let mut mval: c_int = 0;
    let mut lval: c_int = 0;
    let mut a: c_char = b' ' as c_char;
    let mut oldp: c_char;
    let mut c: c_char;
    let mut s: c_char;
    let mut ip: c_int;
    let mut ic: c_int;
    let mut pat: c_int = 0;
    let mut pass: bool = false;
    let mut firstfail: bool;
    let mut spat;

    if *status > 0 {
        return *status;
    }

    /*
    if ((inrec.is_null()) || (outrec.is_null())) {
        *status = NULL_INPUT_PTR;
        return *status;
    }
    */

    outrec[0] = 0;
    if inrec[0] == 0 {
        return 0;
    }

    oldp = 0;
    firstfail = false;

    /* ===== Pattern match stage */
    // NOTE: `pat` must retain its matched value after the loop so the output
    // rewriting stage below uses the correct pattern (mirrors the single C
    // `pat` variable, not a fresh per-iteration for-loop binding).
    pat = 0;
    while (pat as usize) < npat {
        spat = unsafe { cast_slice(CStr::from_ptr(patterns[pat as usize][0]).to_bytes_with_nul()) };

        i1 = 0;
        j1 = 0;
        a = bb(b' '); /* Initialize the place-holders */
        pass = false;

        /* Pass the wildcard pattern */
        if spat[0] == bb(b'*') {
            pass = true;
            break;
        }

        /* Optimization: if we have seen this initial pattern character before,
        then it must have failed, and we can skip the pattern */
        if firstfail && spat[0] == oldp {
            pat += 1;
            continue;
        }
        oldp = spat[0];

        /*
        ip = index of pattern character being matched
        ic = index of keyname character being matched
        firstfail = 1 if we fail on the first characteor (0=not)
         */
        ip = 0;
        ic = 0;
        firstfail = true;
        while (spat[ip as usize] != 0) && (ic < 8) {
            c = inrec[ic as usize];
            s = spat[ip as usize];

            if s == bb(b'i') {
                /* Special pattern: 'i' placeholder */
                if isdigit_safe(c) {
                    i1 = c_int::from(c - bb(b'0'));
                    pass = true;
                }
            } else if s == bb(b'j') {
                /* Special pattern: 'j' placeholder */
                if isdigit_safe(c) {
                    j1 = c_int::from(c - bb(b'0'));
                    pass = true;
                }
            } else if (s == bb(b'n')) || (s == bb(b'm')) || (s == bb(b'l')) || (s == bb(b'#')) {
                /* Special patterns: multi-digit number */
                val = 0;
                pass = false;
                if isdigit_safe(c) {
                    pass = true; /* NOTE, could fail below */

                    /* Parse decimal number */
                    while ic < 8 && isdigit_safe(c) {
                        val = val * 10 + c_int::from(c - bb(b'0'));
                        ic += 1;
                        c = inrec[ic as usize];
                    }
                    ic -= 1;
                    c = inrec[ic as usize];

                    if s == bb(b'n') || s == bb(b'm') {
                        /* Is it a column number? */
                        if val >= 1 && val <= 999 {
                            if val == colnum[0] {
                                val = 1;
                            } else if val == colnum[1] {
                                val = 2;
                            } else if val == colnum[2] {
                                val = 3;
                            } else if val == colnum[3] {
                                val = 4;
                            } else {
                                pass = false;
                                val = 0;
                            }

                            if s == bb(b'n') {
                                nval = val;
                            } else {
                                mval = val;
                            }
                        } else {
                            pass = false;
                        }
                    } else if s == bb(b'l') {
                        /* Generic number */
                        lval = val;
                    }
                }
            } else if s == bb(b'a') {
                /* Special pattern: coordinate designator */
                if isupper(c) || c == bb(b' ') {
                    a = c;
                    pass = true;
                }
            } else if s == bb(b'?') {
                /* Match any individual character */
                pass = true;
            } else if c == s {
                /* Match a specific character */
                pass = true;
            } else {
                /* FAIL */
                pass = false;
            }

            if !pass {
                break;
            }

            ip += 1;
            ic += 1;
            firstfail = false;
        }

        /* Must pass to the end of the keyword.  No partial matches allowed */
        if pass && (ic >= 8 || inrec[ic as usize] == bb(b' ')) {
            break;
        }

        // Advance to the next pattern. This mirrors the increment expression
        // of the C `for (pat=0; pat < npat; pat++)` loop. Without it the outer
        // loop would spin forever on a non-matching pattern.
        pat += 1;
    }

    /* Transfer the pattern-matched numbers to the output parameters */
    //if (i != 0) {
    *i = i1;
    //}
    //if (j != 0) {
    *j = j1;
    //}
    //if (n != 0) {
    *n = nval;
    //}
    //if (m != 0) {
    *m = mval;
    //}
    //if (l != 0) {
    *l = lval;
    //}
    //if (pat_num) {
    *pat_num = pat;
    //}

    /* ===== Keyword rewriting and output stage */
    spat = unsafe { cast_slice(CStr::from_ptr(patterns[pat as usize][1]).to_bytes_with_nul()) };

    /* Return case: no match, or explicit deletion pattern */
    if !pass || spat[0] == 0 || spat[0] == bb(b'-') {
        return 0;
    }

    /* A match: we start by copying the input record to the output */
    strcpy_safe(outrec, inrec);

    /* Return case: return the input record unchanged */
    if spat[0] == bb(b'+') {
        return 0;
    }

    /* Final case: a new output pattern */
    ip = 0;
    ic = 0;
    while spat[ip as usize] != 0 {
        s = spat[ip as usize];
        if s == bb(b'i') {
            outrec[ic as usize] = i1 as c_char + bb(b'0');
        } else if s == bb(b'j') {
            outrec[ic as usize] = j1 as c_char + bb(b'0');
        } else if s == bb(b'n') && nval > 0 {
            fac = 1;
            while nval / fac > 0 {
                fac *= 10
            }

            fac /= 10;
            while fac > 0 {
                outrec[ic as usize] = (((nval / fac) % 10) as c_char) + bb(b'0');
                fac /= 10;
                ic += 1;
            }
            ic -= 1;
        } else if s == bb(b'm') && mval > 0 {
            fac = 1;
            while mval / fac > 0 {
                fac *= 10
            }
            fac /= 10;
            while fac > 0 {
                outrec[ic as usize] = (((mval / fac) % 10) as c_char) + bb(b'0');
                fac /= 10;
                ic += 1;
            }
            ic -= 1;
        } else if s == bb(b'l') && lval >= 0 {
            fac = 1;
            while lval / fac > 0 {
                fac *= 10
            }
            fac /= 10;
            while fac > 0 {
                outrec[ic as usize] = (((lval / fac) % 10) as c_char) + bb(b'0');
                fac /= 10;
                ic += 1;
            }
            ic -= 1;
        } else if s == bb(b'a') {
            outrec[ic as usize] = a;
        } else {
            outrec[ic as usize] = s;
        }
        ip += 1;
        ic += 1;
    }

    /* Pad the keyword name with spaces */
    while ic < 8 {
        outrec[ic as usize] = bb(b' ');
        ic += 1;
    }

    *status
}

/// parse the ASCII table TFORM column format to determine the data
/// type, the field width, and number of decimal places (if relevant)
///
/// # Parameters
///
/// * `tform`    — (I) format code from the TFORMn keyword
/// * `dtcode`   — (O) numerical datatype code
/// * `twidth`   — (O) width of the field, in chars
/// * `decimals` — (O) number of decimal places (F, E, D format)
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffasfm(
    tform: *const c_char,
    dtcode: *mut c_int,
    twidth: *mut c_long,
    decimals: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let dtcode = dtcode.as_mut();
        let twidth = twidth.as_mut();
        let decimals = decimals.as_mut();

        raw_to_slice!(tform);

        ffasfm_safe(tform, dtcode, twidth, decimals, status)
    }
}

/// parse the ASCII table TFORM column format to determine the data
/// type, the field width, and number of decimal places (if relevant)
///
/// # Parameters
///
/// * `tform`    — (I) format code from the TFORMn keyword
/// * `dtcode`   — (O) numerical datatype code
/// * `twidth`   — (O) width of the field, in chars
/// * `decimals` — (O) number of decimal places (F, E, D format)
/// * `status`   — (IO) error status
pub fn ffasfm_safe(
    tform: &[c_char],
    dtcode: Option<&mut c_int>,
    twidth: Option<&mut c_long>,
    decimals: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut datacode;
    let mut longval: c_long = 0;
    let mut width: c_long = 0;
    let mut fwidth: f32 = 0.0;
    let mut form;
    let mut temp: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    let mut dtcode = dtcode;
    let mut twidth = twidth;
    let mut decimals = decimals;

    if *status > 0 {
        return *status;
    }

    if let Some(ref mut dtcode) = dtcode {
        **dtcode = 0;
    }

    if let Some(ref mut twidth) = twidth {
        **twidth = 0;
    }

    if let Some(ref mut decimals) = decimals {
        **decimals = 0;
    }

    let mut ii = 0;
    while tform[ii] != 0 && tform[ii] == bb(b' ') {
        /* find first non-blank char */
        ii += 1;
    }

    if strlen_safe(&tform[ii..]) > FLEN_VALUE - 1 {
        ffpmsg_str("Error: ASCII table TFORM code is too long (ffasfm)");
        *status = BAD_TFORM;
        return *status;
    }

    strcpy_safe(&mut temp, &tform[ii..]); /* copy format string */
    ffupch_safe(&mut temp); /* make sure it is in upper case */
    form = 0; /* point to start of format string */

    if temp[form] == 0 {
        ffpmsg_str("Error: ASCII table TFORM code is blank");
        *status = BAD_TFORM;
        return *status;
    }

    /*       determine default datatype code         */
    if temp[form] == bb(b'A') {
        datacode = TSTRING;
    } else if temp[form] == bb(b'I') {
        datacode = TLONG;
    } else if temp[form] == bb(b'F') || temp[form] == bb(b'E') {
        datacode = TFLOAT;
    } else if temp[form] == bb(b'D') {
        datacode = TDOUBLE;
    } else {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Illegal ASCII table TFORMn datatype: '{}'",
            slice_to_str!(&tform),
        );
        ffpmsg_slice(&message);
        *status = BAD_TFORM_DTYPE;
        return *status;
    }

    if let Some(ref mut dtcode) = dtcode {
        **dtcode = datacode;
    }

    form += 1; /* point to the start of field width */

    if datacode == TSTRING || datacode == TLONG {
        /*              A or I data formats:             */

        if ffc2ii(&temp[form..], &mut width, status) <= 0 {
            /* read the width field */
            if width <= 0 {
                width = 0;
                *status = BAD_TFORM;
            } else {
                /* set to shorter precision if I4 or less */
                if width <= 4 && datacode == TLONG {
                    datacode = TSHORT;
                }
            }
        }
    } else {
        /*              F, E or D data formats:          */

        if ffc2rr(&temp[form..], &mut fwidth, status) <= 0 {
            /* read ww.dd width field */
            if fwidth <= 0. {
                *status = BAD_TFORM;
            } else {
                width = fwidth as c_long; /* convert from float to long */

                if width > 7 && temp[0] == bb(b'F') {
                    datacode = TDOUBLE; /* type double if >7 digits */
                }
                if width < 10 {
                    form += 1; /* skip 1 digit  */
                } else {
                    form += 2; /* skip 2 digits */
                }
                if temp[form] == bb(b'.') {
                    /* should be a decimal point here */

                    form += 1; /*  point to start of decimals field */

                    if ffc2ii(&temp[form..], &mut longval, status) <= 0 {
                        /* read decimals */
                        if let Some(ref mut decimals) = decimals {
                            **decimals = longval as c_int; /* long to short convertion */
                        }

                        if longval >= width {
                            /* width < no. of decimals */
                            *status = BAD_TFORM;
                        }

                        if longval > 6 && temp[0] == bb(b'E') {
                            datacode = TDOUBLE; /* type double if >6 digits */
                        }
                    }
                }
            }
        }
    }

    if *status > 0 {
        *status = BAD_TFORM;
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Illegal ASCII table TFORMn code: '{}'",
            slice_to_str!(&tform),
        );
        ffpmsg_slice(&message);
    }

    if let Some(ref mut dtcode) = dtcode {
        **dtcode = datacode;
    }

    if let Some(ref mut twidth) = twidth {
        **twidth = width;
    }

    *status
}

/// parse the binary table TFORM column format to determine the data
/// type, repeat count, and the field width (if it is an ASCII (A) field)
///
/// # Parameters
///
/// * `tform`   — (I) format code from the TFORMn keyword
/// * `dtcode`  — (O) numerical datatype code
/// * `trepeat` — (O) repeat count of the field
/// * `twidth`  — (O) width of the field, in chars
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffbnfm(
    tform: *const c_char,
    dtcode: *mut c_int,
    trepeat: *mut c_long,
    twidth: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let dtcode = dtcode.as_mut();
        let twidth = twidth.as_mut();
        let trepeat = trepeat.as_mut();

        raw_to_slice!(tform);

        ffbnfm_safe(tform, dtcode, trepeat, twidth, status)
    }
}

/// parse the binary table TFORM column format to determine the data
/// type, repeat count, and the field width (if it is an ASCII (A) field)
///
/// # Parameters
///
/// * `tform`   — (I) format code from the TFORMn keyword
/// * `dtcode`  — (O) numerical datatype code
/// * `trepeat` — (O) repeat count of the field
/// * `twidth`  — (O) width of the field, in chars
/// * `status`  — (IO) error status
pub fn ffbnfm_safe(
    tform: &[c_char],
    mut dtcode: Option<&mut c_int>,
    mut trepeat: Option<&mut c_long>,
    mut twidth: Option<&mut c_long>,
    status: &mut c_int,
) -> c_int {
    let mut datacode; /* flag variable cols w/ neg type code */
    let variable;
    let mut iread;
    let mut width: c_long = 0;
    let repeat: c_long;
    let mut temp: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    if *status > 0 {
        return *status;
    }

    if let Some(dtcode) = dtcode.as_deref_mut() {
        *dtcode = 0;
    }

    if let Some(trepeat) = trepeat.as_deref_mut() {
        *trepeat = 0;
    }

    if let Some(twidth) = twidth.as_deref_mut() {
        *twidth = 0;
    }

    let nchar = strlen_safe(tform);

    let mut ii = 0;
    while ii < nchar {
        if tform[ii] != bb(b' ') {
            /* find first non-space char */
            break;
        }
        ii += 1;
    }

    if ii == nchar {
        ffpmsg_str("Error: binary table TFORM code is blank (ffbnfm).");
        *status = BAD_TFORM;
        return *status;
    }

    if nchar - ii > FLEN_VALUE - 1 {
        ffpmsg_str("Error: binary table TFORM code is too long (ffbnfm).");
        *status = BAD_TFORM;
        return *status;
    }
    strcpy_safe(&mut temp, &tform[ii..]); /* copy format string */
    ffupch_safe(&mut temp); /* make sure it is in upper case */

    /*       get the repeat count                    */

    let mut ii = 0;
    while isdigit_safe(temp[ii]) {
        ii += 1; /* look for leading digits in the field */
    }

    if ii == 0 {
        repeat = 1; /* no explicit repeat count */
    } else {
        let tmp: Result<c_long, ParseIntError> =
            str::from_utf8(cast_slice(&temp[..ii])).unwrap().parse();

        match tmp {
            Ok(x) => {
                repeat = x;
            }
            Err(_x) => {
                /* read repeat count */
                ffpmsg_str("Error: Bad repeat format in TFORM (ffbnfm).");
                *status = BAD_TFORM;
                return *status;
            }
        }

        /*
        unsafe {
            if sscanf_ld(&temp, cs!(c"%ld"), &mut repeat) != 1 {
                /* read repeat count */

                ffpmsg_str("Error: Bad repeat format in TFORM (ffbnfm).");
                *status = BAD_TFORM;
                return *status;
            }
        }
        */
    }

    /*             determine datatype code           */
    let mut fi = ii;
    /* skip over the repeat field */

    if temp[fi] == bb(b'P') || temp[fi] == bb(b'Q') {
        variable = 1; /* this is a variable length column */
        /*        repeat = 1;  */
        /* disregard any other repeat value */
        fi += 1; /* move to the next data type code char */
    } else {
        variable = 0;
    }

    if temp[fi] == bb(b'U') {
        /* internal code to signify unsigned short integer */
        datacode = TUSHORT;
        width = 2;
    } else if temp[fi] == bb(b'I') {
        datacode = TSHORT;
        width = 2;
    } else if temp[fi] == bb(b'V') {
        /* internal code to signify unsigned integer */
        datacode = TULONG;
        width = 4;
    } else if temp[fi] == bb(b'W') {
        /* internal code to signify unsigned long long integer */
        datacode = TULONGLONG;
        width = 8;
    } else if temp[fi] == bb(b'J') {
        datacode = TLONG;
        width = 4;
    } else if temp[fi] == bb(b'K') {
        datacode = TLONGLONG;
        width = 8;
    } else if temp[fi] == bb(b'E') {
        datacode = TFLOAT;
        width = 4;
    } else if temp[fi] == bb(b'D') {
        datacode = TDOUBLE;
        width = 8;
    } else if temp[fi] == bb(b'A') {
        datacode = TSTRING;

        /*
          the following code is used to support the non-standard
          datatype of the form rAw where r = total width of the field
          and w = width of fixed-length substrings within the field.
        */
        iread = 0;
        if temp[fi + 1] != 0 {
            if temp[fi + 1] == bb(b'(') {
                /* skip parenthesis around */
                fi += 1; /* variable length column width */
            }

            /*
            iread = unsafe { sscanf_ld(&temp[(fi + 1)..], cs!(c"%ld"), &mut width) };
            */

            /* only the string itself: bytes after its terminator are left over */
            /* from the header, and need not be text */
            let rest: &[u8] = cast_slice(&temp[(fi + 1)..]);
            let rest = &rest[..rest.iter().position(|&c| c == 0).unwrap_or(rest.len())];
            let tmp: Option<c_long> = str::from_utf8(rest).ok().and_then(|s| atoi(s).ok());

            match tmp {
                Some(x) => {
                    width = x;
                    iread = 1;
                }
                None => {
                    iread = 0;
                }
            }
        }

        if iread != 1 || (variable == 0 && (width > repeat)) {
            width = repeat;
        }
    } else if temp[fi] == bb(b'L') {
        datacode = TLOGICAL;
        width = 1;
    } else if temp[fi] == bb(b'X') {
        datacode = TBIT;
        width = 1;
    } else if temp[fi] == bb(b'B') {
        datacode = TBYTE;
        width = 1;
    } else if temp[fi] == bb(b'S')
    /* internal code to signify signed byte */
    {
        datacode = TSBYTE;
        width = 1;
    } else if temp[fi] == bb(b'C') {
        datacode = TCOMPLEX;
        width = 8;
    } else if temp[fi] == bb(b'M') {
        datacode = TDBLCOMPLEX;
        width = 16;
    } else {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Illegal binary table TFORMn datatype: '{}' ",
            slice_to_str!(&tform),
        );
        ffpmsg_slice(&message);
        *status = BAD_TFORM_DTYPE;
        return *status;
    }

    if variable > 0 {
        datacode = -datacode; /* flag variable cols w/ neg type code */
    }
    if let Some(dtcode) = dtcode {
        *dtcode = datacode;
    }
    if let Some(trepeat) = trepeat {
        *trepeat = repeat;
    }
    if let Some(twidth) = twidth {
        *twidth = width;
    }

    *status
}

/// parse the binary table TFORM column format to determine the data
/// type, repeat count, and the field width (if it is an ASCII (A) field)
///
/// # Parameters
///
/// * `tform`   — (I) format code from the TFORMn keyword
/// * `dtcode`  — (O) numerical datatype code
/// * `trepeat` — (O) repeat count of the field
/// * `twidth`  — (O) width of the field, in chars
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffbnfmll(
    tform: *const c_char,
    dtcode: *mut c_int,
    trepeat: *mut LONGLONG,
    twidth: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let dtcode = dtcode.as_mut();
        let trepeat = trepeat.as_mut();
        let twidth = twidth.as_mut();
        raw_to_slice!(tform);

        ffbnfmll_safe(tform, dtcode, trepeat, twidth, status)
    }
}

/// parse the binary table TFORM column format to determine the data
/// type, repeat count, and the field width (if it is an ASCII (A) field)
///
/// # Parameters
///
/// * `tform`   — (I) format code from the TFORMn keyword
/// * `dtcode`  — (O) numerical datatype code
/// * `trepeat` — (O) repeat count of the field
/// * `twidth`  — (O) width of the field, in chars
/// * `status`  — (IO) error status
pub fn ffbnfmll_safe(
    tform: &[c_char],
    mut dtcode: Option<&mut c_int>,
    mut trepeat: Option<&mut LONGLONG>,
    mut twidth: Option<&mut c_long>,
    status: &mut c_int,
) -> c_int {
    let mut datacode: c_int;
    let variable: c_int;
    let mut iread: c_int;
    let mut width: c_long = 0;
    let repeat: LONGLONG;
    let mut temp: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut drepeat: f64 = 0.0;

    if *status > 0 {
        return *status;
    }

    if let Some(ref mut dtcode) = dtcode {
        **dtcode = 0;
    }

    if let Some(ref mut trepeat) = trepeat {
        **trepeat = 0;
    }

    if let Some(ref mut twidth) = twidth {
        **twidth = 0;
    }

    let nchar = strlen_safe(tform);
    let mut ii = 0;
    while ii < nchar {
        if tform[ii] != bb(b' ') {
            /* find first non-space char */
            break;
        }
        ii += 1;
    }

    if ii == nchar {
        ffpmsg_str("Error: binary table TFORM code is blank (ffbnfmll).");
        *status = BAD_TFORM;
        return *status;
    }

    if strlen_safe(&tform[ii..]) > FLEN_VALUE - 1 {
        ffpmsg_str("Error: binary table TFORM code is too long (ffbnfmll).");
        *status = BAD_TFORM;
        return *status;
    }
    strcpy_safe(&mut temp, &tform[ii..]); /* copy format string */
    ffupch_safe(&mut temp); /* make sure it is in upper case */
    let mut fi: usize = 0; /* point to start of format string */

    /*       get the repeat count                    */

    let mut ii = 0;
    while isdigit_safe(temp[fi + ii]) {
        ii += 1; /* look for leading digits in the field */
    }

    if ii == 0 {
        repeat = 1; /* no explicit repeat count */
    } else {
        /* read repeat count */

        /* print as double, because the string-to-64-bit int conversion */
        /* character is platform dependent (%lld, %ld, %I64d)           */

        sscanf_lf(&temp[fi..], cs!(c"%lf"), &raw mut drepeat);
        repeat = (drepeat + 0.1) as LONGLONG;
    }
    /*             determine datatype code           */

    fi += ii; /* skip over the repeat field */

    if temp[fi] == bb(b'P') || temp[fi] == bb(b'Q') {
        variable = 1; /* this is a variable length column */
        /*        repeat = 1;  */
        /* disregard any other repeat value */
        fi += 1; /* move to the next data type code char */
    } else {
        variable = 0;
    }

    if temp[fi] == bb(b'U')
    /* internal code to signify unsigned integer */
    {
        datacode = TUSHORT;
        width = 2;
    } else if temp[fi] == bb(b'I') {
        datacode = TSHORT;
        width = 2;
    } else if temp[fi] == bb(b'V')
    /* internal code to signify unsigned integer */
    {
        datacode = TULONG;
        width = 4;
    } else if temp[fi] == bb(b'W')
    /* internal code to signify unsigned long long integer */
    {
        datacode = TULONGLONG;
        width = 8;
    } else if temp[fi] == bb(b'J') {
        datacode = TLONG;
        width = 4;
    } else if temp[fi] == bb(b'K') {
        datacode = TLONGLONG;
        width = 8;
    } else if temp[fi] == bb(b'E') {
        datacode = TFLOAT;
        width = 4;
    } else if temp[fi] == bb(b'D') {
        datacode = TDOUBLE;
        width = 8;
    } else if temp[fi] == bb(b'A') {
        datacode = TSTRING;

        /*
          the following code is used to support the non-standard
          datatype of the form rAw where r = total width of the field
          and w = width of fixed-length substrings within the field.
        */
        iread = 0;
        if temp[1 + fi] != 0 {
            if temp[1 + fi] == bb(b'(') {
                /* skip parenthesis around */
                fi += 1; /* variable length column width */
            }

            iread = sscanf_ld(&temp[(1 + fi)..], cs!(c"%ld"), &raw mut width);
        }

        if iread != 1 || (variable == 0 && (width as LONGLONG > repeat)) {
            width = repeat as c_long;
        }
    } else if temp[fi] == bb(b'L') {
        datacode = TLOGICAL;
        width = 1;
    } else if temp[fi] == bb(b'X') {
        datacode = TBIT;
        width = 1;
    } else if temp[fi] == bb(b'B') {
        datacode = TBYTE;
        width = 1;
    } else if temp[fi] == bb(b'S')
    /* internal code to signify signed byte */
    {
        datacode = TSBYTE;
        width = 1;
    } else if temp[fi] == bb(b'C') {
        datacode = TCOMPLEX;
        width = 8;
    } else if temp[fi] == bb(b'M') {
        datacode = TDBLCOMPLEX;
        width = 16;
    } else {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Illegal binary table TFORMn datatype: '{}' ",
            slice_to_str!(&tform),
        );
        ffpmsg_slice(&message);
        *status = BAD_TFORM_DTYPE;
        return *status;
    }

    if variable != 0 {
        datacode = -datacode; /* flag variable cols w/ neg type code */
    }

    if let Some(dtcode) = dtcode {
        *dtcode = datacode;
    }

    if let Some(trepeat) = trepeat {
        *trepeat = repeat;
    }

    if let Some(twidth) = twidth {
        *twidth = width;
    }

    *status
}

/// convert the FITS format string for an ASCII Table extension column into the
/// equivalent C format string that can be used in a printf statement, after
/// the values have been read as a double.
///
/// # Parameters
///
/// * `tform` — value of an ASCII table TFORMn keyword
/// * `cform` — equivalent format code in C language syntax
pub(crate) fn ffcfmt(tform: &[c_char], cform: &mut [c_char]) {
    cform[0] = 0;
    let mut ii = 0;
    while tform[ii] != 0 && tform[ii] == bb(b' ') {
        /* find first non-blank char */
        ii += 1;
    }

    if tform[ii] == 0 {
        return; /* input format string was blank */
    }

    let istart = ii;
    let mut isgood = true;
    let mut npt: usize = 0;

    if tform[ii] != bb(b'A')
        && tform[ii] != bb(b'I')
        && tform[ii] != bb(b'F')
        && tform[ii] != bb(b'E')
        && tform[ii] != bb(b'D')
    {
        isgood = false;
    }

    ii += 1;

    while isgood && tform[ii] != 0 {
        /* one period is allowed */
        if tform[ii] == bb(b'.') {
            if npt > 0 {
                isgood = false;
            } else {
                npt += 1;
            }
        } else if !isdigit_safe(tform[ii]) {
            isgood = false;
        }
        ii += 1;
    }
    if !isgood {
        return;
    }

    cform[0] = bb(b'%'); /* start the format string */

    strcpy_safe(&mut cform[1..], &tform[(istart + 1)..]); /* append the width and decimal code */

    if tform[istart] == bb(b'A') {
        strcat_safe(cform, cs!(c"s"));
    } else if tform[istart] == bb(b'I') {
        strcat_safe(cform, cs!(c".0f")); /*  0 precision to suppress decimal point */
    }
    if tform[istart] == bb(b'F') {
        strcat_safe(cform, cs!(c"f"));
    }
    if tform[istart] == bb(b'E') {
        strcat_safe(cform, cs!(c"E"));
    }
    if tform[istart] == bb(b'D') {
        strcat_safe(cform, cs!(c"E"));
    }
}

/// convert the FITS TDISPn display format into the equivalent C format
/// suitable for use in a printf statement.
///
/// # Parameters
///
/// * `tform` — value of an ASCII table TFORMn keyword
/// * `cform` — equivalent format code in C language syntax
pub(crate) fn ffcdsp(tform: &[c_char], cform: &mut [c_char]) {
    let mut ii: usize;

    cform[0] = 0;
    ii = 0;

    while tform[ii] != 0 && tform[ii] == bb(b' ') {
        /* find first non-blank char */
        ii += 1;
    }

    if tform[ii] == 0 {
        cform[0] = 0;
        return; /* input format string was blank */
    }

    if strchr_safe(&tform[ii..], bb(b'%')).is_some() {
        /* is there a % character in the string?? */
        cform[0] = 0;
        return; /* illegal TFORM string (possibly even harmful) */
    }

    cform[0] = bb(b'%'); /* start the format string */

    strcpy_safe(&mut cform[1..], &tform[(ii + 1)..]); /* append the width and decimal code */

    if tform[ii] == bb(b'A') || tform[ii] == bb(b'a') {
        strcat_safe(cform, cs!(c"s"));
    } else if tform[ii] == bb(b'I') || tform[ii] == bb(b'i') {
        strcat_safe(cform, cs!(c"d"));
    } else if tform[ii] == bb(b'O') || tform[ii] == bb(b'o') {
        strcat_safe(cform, cs!(c"o"));
    } else if tform[ii] == bb(b'Z') || tform[ii] == bb(b'z') {
        strcat_safe(cform, cs!(c"X"));
    } else if tform[ii] == bb(b'F') || tform[ii] == bb(b'f') {
        strcat_safe(cform, cs!(c"f"));
    } else if tform[ii] == bb(b'E')
        || tform[ii] == bb(b'e')
        || tform[ii] == bb(b'D')
        || tform[ii] == bb(b'd')
    {
        strcat_safe(cform, cs!(c"E"));
    } else if tform[ii] == bb(b'G') || tform[ii] == bb(b'g') {
        strcat_safe(cform, cs!(c"G"));
    } else {
        cform[0] = 0; /* unrecognized tform code */
    }
}

/// Determine the column number corresponding to an input column name.
/// The first column of the table = column 1;  
/// This supports the * and ? wild cards in the input template.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pionter
/// * `casesen` — (I) case sensitive string comparison? 0=no
/// * `templt`  — (I) input name of column (w/wildcards)
/// * `colnum`  — (O) number of the named column; 1=first col
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcno(
    fptr: *mut fitsfile,
    casesen: c_int,
    templt: *const c_char,
    colnum: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let colnum = colnum.as_mut().expect(NULL_MSG);
        raw_to_slice!(templt);

        ffgcno_safe(fptr, casesen, templt, colnum, status)
    }
}

/// Determine the column number corresponding to an input column name.
/// The first column of the table = column 1;  
/// This supports the * and ? wild cards in the input template.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pionter
/// * `casesen` — (I) case sensitive string comparison? 0=no
/// * `templt`  — (I) input name of column (w/wildcards)
/// * `colnum`  — (O) number of the named column; 1=first col
/// * `status`  — (IO) error status
pub fn ffgcno_safe(
    fptr: &mut fitsfile,
    casesen: c_int,
    templt: &[c_char],
    colnum: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut colname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE]; /*  temporary string to hold column name  */

    ffgcnn_safe(fptr, casesen, templt, &mut colname, colnum, status);

    *status
}

/// Return the full column name and column number of the next column whose
/// TTYPEn keyword value matches the input template string.
/// The template may contain the * and ? wildcards.  Status = 237 is
/// returned if the match is not unique.  If so, one may call this routine
/// again with input status=237  to get the next match.  A status value of
/// 219 is returned when there are no more matching columns.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `casesen` — (I) case sensitive string comparison? 0=no
/// * `templt`  — (I) input name of column (w/wildcards)
/// * `colname` — (O) full column name up to 68 + 1 chars long
/// * `colnum`  — (O) number of the named column; 1=first col
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgcnn(
    fptr: *mut fitsfile,
    casesen: c_int,
    templt: *const c_char,
    colname: *mut c_char,
    colnum: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let colnum = colnum.as_mut().expect(NULL_MSG);
        let colname = slice::from_raw_parts_mut(colname, 69);

        raw_to_slice!(templt);

        ffgcnn_safe(fptr, casesen, templt, colname, colnum, status)
    }
}

/// Return the full column name and column number of the next column whose
/// TTYPEn keyword value matches the input template string.
/// The template may contain the * and ? wildcards.  Status = 237 is
/// returned if the match is not unique.  If so, one may call this routine
/// again with input status=237  to get the next match.  A status value of
/// 219 is returned when there are no more matching columns.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `casesen` — (I) case sensitive string comparison? 0=no
/// * `templt`  — (I) input name of column (w/wildcards)
/// * `colname` — (O) full column name up to 68 + 1 chars long
/// * `colnum`  — (O) number of the named column; 1=first col
/// * `status`  — (IO) error status
pub fn ffgcnn_safe(
    fptr: &mut fitsfile,
    casesen: c_int,
    templt: &[c_char],
    colname: &mut [c_char],
    colnum: &mut c_int,
    status: &mut c_int,
) -> c_int {
    let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut tstatus: c_int;
    let mut founde: bool = false; /* initialize 'found exact match' flag */
    let mut foundw: bool = false; /* initialize 'found wildcard match' flag */
    let mut matchi = 0;
    let mut exact = 0;
    let mut unique: bool = false;
    let mut ivalue: c_long = 0;

    if *status <= 0 {
        fptr.Fptr.startcol = 0; /* start search with first column */
        tstatus = 0;
    } else if *status == COL_NOT_UNIQUE {
        /* start search from previous spot */
        tstatus = COL_NOT_UNIQUE;
        *status = 0;
    } else {
        return *status; /* bad input status value */
    }

    colname[0] = 0; /* initialize null return */
    *colnum = 0;

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header to get col struct */
        return *status;
    }

    let c =
        FITSfile::get_tableptr_as_slice_unlinked(&fptr.Fptr.tableptr, fptr.Fptr.tfield as usize);
    let mut ci = (fptr.Fptr.startcol) as usize; /* offset to starting column */

    let mut ii = fptr.Fptr.startcol;
    while ii < fptr.Fptr.tfield {
        ffcmps_safe(templt, &c[ci].ttype, casesen, &mut matchi, &mut exact);
        if matchi > 0 {
            if founde && exact > 0 {
                /* warning: this is the second exact match we've found     */
                /*reset pointer to first match so next search starts there */
                fptr.Fptr.startcol = *colnum;
                *status = COL_NOT_UNIQUE;
                return *status;
            } else if founde { /* a wildcard match */
                /* already found exact match so ignore this non-exact match */
            } else if exact > 0 {
                /* this is the first exact match we have found, so save it. */
                strcpy_safe(colname, &c[ci].ttype);
                *colnum = ii + 1;
                founde = true;
            } else if foundw {
                /* we have already found a wild card match, so not unique */
                /* continue searching for other matches                   */
                unique = false;
            } else {
                /* this is the first wild card match we've found. save it */
                strcpy_safe(colname, &c[ci].ttype);
                *colnum = ii + 1;
                fptr.Fptr.startcol = *colnum;
                foundw = true;
                unique = true;
            }
        }
        ii += 1;
        ci += 1;
    }

    /* OK, we've checked all the names now see if we got any matches */
    if founde {
        if tstatus == COL_NOT_UNIQUE {
            /* we did find 1 exact match but */
            *status = COL_NOT_UNIQUE; /* there was a previous match too */
        }
    } else if foundw {
        /* found one or more wildcard matches; report error if not unique */
        if !unique || tstatus == COL_NOT_UNIQUE {
            *status = COL_NOT_UNIQUE;
        }
    } else {
        /* didn't find a match; check if template is a positive integer */

        ffc2ii(templt, &mut ivalue, &mut tstatus);
        if tstatus == 0 && ivalue <= c_long::from(fptr.Fptr.tfield) && ivalue > 0 {
            *colnum = ivalue as c_int;

            let c = fptr.Fptr.get_tableptr_as_slice();
            let ci = (ivalue - 1) as usize; /* offset to correct column */

            strcpy_safe(colname, &c[ci].ttype);
        } else {
            *status = COL_NOT_FOUND;
            if tstatus != COL_NOT_UNIQUE {
                int_snprintf!(
                    &mut errmsg,
                    FLEN_ERRMSG,
                    "ffgcnn could not find column: {:.45}",
                    slice_to_str!(&templt),
                );

                ffpmsg_slice(&errmsg);
            }
        }
    }

    fptr.Fptr.startcol = *colnum; /* save pointer for next time */
    *status
}

///  compare the template to the string and test if they match.
///
///  The strings are limited to 68 characters or less (the max. length
///  of a FITS string keyword value.  This routine reports whether
///  the two strings match and whether the match is exact or
///  involves wildcards.
///
///  This algorithm is very similar to the way unix filename wildcards
///  work except that this first treats a wild card as a literal character
///  when looking for a match.  If there is no literal match, then
///  it interpretes it as a wild card.  So the template 'AB*DE'
///  is considered to be an exact rather than a wild card match to
///  the string 'AB*DE'.  The '#' wild card in the template string will
///  match any consecutive string of decimal digits in the colname.
///
/// # Parameters
///
/// * `templt`  — (I) input template (may have wildcards)
/// * `colname` — (I) full column name up to 68 + 1 chars long
/// * `casesen` — (I) case sensitive string comparison? 1=yes
/// * `matchi`  — (O) do template and colname match? 1=yes
/// * `exact`   — (O) do strings exactly match, or wildcards
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffcmps(
    templt: *const c_char,
    colname: *const c_char,
    casesen: c_int,
    matchi: *mut c_int,
    exact: *mut c_int,
) {
    // FFI WRAPPER
    unsafe {
        let matchi = matchi.as_mut().expect(NULL_MSG);
        let exact = exact.as_mut().expect(NULL_MSG);

        raw_to_slice!(templt);
        raw_to_slice!(colname);

        ffcmps_safe(templt, colname, casesen, matchi, exact);
    }
}

///  compare the template to the string and test if they match.
///
///  The strings are limited to 68 characters or less (the max. length
///  of a FITS string keyword value.  This routine reports whether
///  the two strings match and whether the match is exact or
///  involves wildcards.
///
///  This algorithm is very similar to the way unix filename wildcards
///  work except that this first treats a wild card as a literal character
///  when looking for a match.  If there is no literal match, then
///  it interpretes it as a wild card.  So the template 'AB*DE'
///  is considered to be an exact rather than a wild card match to
///  the string 'AB*DE'.  The '#' wild card in the template string will
///  match any consecutive string of decimal digits in the colname.
///
/// # Parameters
///
/// * `templt`  — (I) input template (may have wildcards)
/// * `colname` — (I) full column name up to 68 + 1 chars long
/// * `casesen` — (I) case sensitive string comparison? 1=yes
/// * `matchi`  — (O) do template and colname match? 1=yes
/// * `exact`   — (O) do strings exactly match, or wildcards
pub fn ffcmps_safe(
    templt: &[c_char],
    colname: &[c_char],
    casesen: c_int,
    matchi: &mut c_int,
    exact: &mut c_int,
) {
    let mut found;
    let mut t1;
    let mut s1;
    let mut wildsearch = false;
    let mut tsave: c_int = 0;
    let mut ssave: c_int = 0;
    let mut temp: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut col: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    *matchi = FALSE as c_int;
    *exact = TRUE as c_int;

    strncpy_safe(&mut temp, templt, FLEN_VALUE); /* copy strings to work area */
    strncpy_safe(&mut col, colname, FLEN_VALUE);
    temp[FLEN_VALUE - 1] = 0; /* make sure strings are terminated */
    col[FLEN_VALUE - 1] = 0;

    /* truncate trailing non-significant blanks */
    let mut ii = strlen_safe(&temp) as isize - 1;
    while ii >= 0 {
        if temp[ii as usize] == bb(b' ') {
            temp[ii as usize] = 0;
            ii -= 1;
        } else {
            break;
        }
    }

    let mut ii = strlen_safe(&col) as isize - 1;
    while ii >= 0 {
        if col[ii as usize] == bb(b' ') {
            col[ii as usize] = 0;
            ii -= 1;
        } else {
            break;
        }
    }
    if casesen == 0 {
        /* convert both strings to uppercase before comparison */
        ffupch_safe(&mut temp);
        ffupch_safe(&mut col);
    }

    if FSTRCMP(&temp, &col) == 0 {
        *matchi = TRUE as c_int; /* strings exactly match */
        return;
    }

    *exact = FALSE as c_int; /* strings don't exactly match */

    t1 = 0; /* start comparison with 1st char of each string */
    s1 = 0;

    loop {
        /* compare corresponding chars in each string */
        if temp[t1] == 0 && col[s1] == 0 {
            /* completely scanned both strings so they match */
            *matchi = TRUE as c_int;
            return;
        } else if temp[t1] == 0 {
            if wildsearch {
                /*
                   the previous wildcard search may have been going down
                   a blind alley.  Backtrack, and resume the wildcard
                   search with the next character in the string.
                */
                t1 = tsave as usize;
                s1 = ssave as usize + 1;
            } else {
                /* reached end of template string so they don't match */
                return;
            }
        } else if col[s1] == 0 {
            /* reached end of other string; they match if the next */
            /* character in the template string is a '*' wild card */

            if temp[t1] == bb(b'*') && temp[t1 + 1] == 0 {
                *matchi = TRUE as c_int;
            }

            return;
        }

        if temp[t1] == col[s1] || (temp[t1] == bb(b'?')) {
            s1 += 1; /* corresponding chars in the 2 strings match */
            t1 += 1; /* increment both pointers and loop back again */
        } else if temp[t1] == bb(b'#') && isdigit_safe(col[s1]) {
            s1 += 1; /* corresponding chars in the 2 strings match */
            t1 += 1; /* increment both pointers */

            /* find the end of the string of digits */
            while isdigit_safe(col[s1]) {
                s1 += 1;
            }
        } else if temp[t1] == bb(b'*') {
            /* save current string locations, in case we need to restart */
            wildsearch = true;
            tsave = t1 as c_int;
            ssave = s1 as c_int;

            /* get next char from template and look for it in the col name */
            t1 += 1;
            if temp[t1] == 0 || temp[t1] == bb(b' ') {
                /* reached end of template so strings match */
                *matchi = TRUE as c_int;
                return;
            }

            found = false;
            while col[s1] > 0 && !found {
                if temp[t1] == col[s1] {
                    t1 += 1; /* found matching characters; incre both pointers */
                    s1 += 1; /* and loop back to compare next chars */
                    found = true;
                } else {
                    s1 += 1; /* increment the column name pointer and try again */
                }
            }

            if !found {
                return; /* hit end of column name and failed to find a match */
            }
        } else if wildsearch {
            /*
               the previous wildcard search may have been going down
               a blind alley.  Backtrack, and resume the wildcard
               search with the next character in the string.
            */
            t1 = tsave as usize;
            s1 = ssave as usize + 1;
        } else {
            return; /* strings don't match */
        }
    }
}

/// Get Type of table column.
///
/// Returns the datatype code of the column, as well as the vector
/// repeat count and (if it is an ASCII character column) the
/// width of the field or a unit string within the field.  This supports the
/// TFORMn = 'rAw' syntax for specifying arrays of substrings, so
/// if TFORMn = '60A12' then repeat = 60 and width = 12.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number
/// * `typecode` — (O) datatype code (21 = short, etc)
/// * `repeat`   — (O) repeat count of field
/// * `width`    — (O) if ASCII, width of field or unit string
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtcl(
    fptr: *mut fitsfile,
    colnum: c_int,
    typecode: *mut c_int,
    repeat: *mut c_long,
    width: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let typecode = typecode.as_mut();
        let repeat = repeat.as_mut();
        let width = width.as_mut();

        ffgtcl_safe(fptr, colnum, typecode, repeat, width, status)
    }
}

/// Get Type of table column.
///
/// Returns the datatype code of the column, as well as the vector
/// repeat count and (if it is an ASCII character column) the
/// width of the field or a unit string within the field.  This supports the
/// TFORMn = 'rAw' syntax for specifying arrays of substrings, so
/// if TFORMn = '60A12' then repeat = 60 and width = 12.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number
/// * `typecode` — (O) datatype code (21 = short, etc)
/// * `repeat`   — (O) repeat count of field
/// * `width`    — (O) if ASCII, width of field or unit string
/// * `status`   — (IO) error status
pub fn ffgtcl_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    typecode: Option<&mut c_int>,
    repeat: Option<&mut c_long>,
    width: Option<&mut c_long>,
    status: &mut c_int,
) -> c_int {
    let mut trepeat: LONGLONG = 0;
    let mut twidth: LONGLONG = 0;

    ffgtclll_safe(
        fptr,
        colnum,
        typecode,
        Some(&mut trepeat),
        Some(&mut twidth),
        status,
    );

    if *status > 0 {
        return *status;
    }

    if let Some(repeat) = repeat {
        *repeat = trepeat as c_long;
    }

    if let Some(width) = width {
        *width = twidth as c_long;
    }

    *status
}

/// Get Type of table column.
///
/// Returns the datatype code of the column, as well as the vector
/// repeat count and (if it is an ASCII character column) the
/// width of the field or a unit string within the field.  This supports the
/// TFORMn = 'rAw' syntax for specifying arrays of substrings, so
/// if TFORMn = '60A12' then repeat = 60 and width = 12.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number
/// * `typecode` — (O) datatype code (21 = short, etc)
/// * `repeat`   — (O) repeat count of field
/// * `width`    — (O) if ASCII, width of field or unit string
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgtclll(
    fptr: *mut fitsfile,
    colnum: c_int,
    typecode: *mut c_int,
    repeat: *mut LONGLONG,
    width: *mut LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let typecode = typecode.as_mut();
        let repeat = repeat.as_mut();
        let width = width.as_mut();

        ffgtclll_safe(fptr, colnum, typecode, repeat, width, status)
    }
}

/// Get Type of table column.
///
/// Returns the datatype code of the column, as well as the vector
/// repeat count and (if it is an ASCII character column) the
/// width of the field or a unit string within the field.  This supports the
/// TFORMn = 'rAw' syntax for specifying arrays of substrings, so
/// if TFORMn = '60A12' then repeat = 60 and width = 12.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number
/// * `typecode` — (O) datatype code (21 = short, etc)
/// * `repeat`   — (O) repeat count of field
/// * `width`    — (O) if ASCII, width of field or unit string
/// * `status`   — (IO) error status
pub fn ffgtclll_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    typecode: Option<&mut c_int>,
    repeat: Option<&mut LONGLONG>,
    width: Option<&mut LONGLONG>,
    status: &mut c_int,
) -> c_int {
    let mut hdutype = 0;
    let mut decims = 0;
    let mut tmpwidth: c_long = 0;

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    if colnum < 1 || colnum > fptr.Fptr.tfield {
        *status = BAD_COL_NUM;
        return *status;
    }

    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_slice();
    let ci = (colnum - 1) as usize; /* offset to the correct column */

    if hdutype == ASCII_TBL {
        ffasfm_safe(
            &c[ci].tform,
            typecode,
            Some(&mut tmpwidth),
            Some(&mut decims),
            status,
        );

        // WARNING: Original code doesn't check for null ptr
        if let Some(width) = width {
            *width = tmpwidth as LONGLONG;
        }

        if let Some(repeat) = repeat {
            *repeat = 1;
        };
    } else {
        if let Some(typecode) = typecode {
            *typecode = c[ci].tdatatype;
        }
        if let Some(width) = width {
            *width = c[ci].twidth as LONGLONG;
        }
        if let Some(repeat) = repeat {
            *repeat = c[ci].trepeat;
        };
    }
    *status
}

/// Get the 'equivalent' table column type.
///
/// This routine is similar to the ffgtcl routine (which returns the physical
/// datatype of the column, as stored in the FITS file) except that if the
/// TSCALn and TZEROn keywords are defined for the column, then it returns
/// the 'equivalent' datatype.  Thus, if the column is defined as '1I'  (short
/// integer) this routine may return the type as 'TUSHORT' or as 'TFLOAT'
/// depending on the TSCALn and TZEROn values.
///
/// Returns the datatype code of the column, as well as the vector
/// repeat count and (if it is an ASCII character column) the
/// width of the field or a unit string within the field.  This supports the
/// TFORMn = 'rAw' syntax for specifying arrays of substrings, so
/// if TFORMn = '60A12' then repeat = 60 and width = 12.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number
/// * `typecode` — (O) datatype code (21 = short, etc)
/// * `repeat`   — (O) repeat count of field
/// * `width`    — (O) if ASCII, width of field or unit string
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffeqty(
    fptr: *mut fitsfile,
    colnum: c_int,
    typecode: *mut c_int,
    repeat: *mut c_long,
    width: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let typecode = typecode.as_mut();
        let repeat = repeat.as_mut();
        let width = width.as_mut();

        ffeqty_safe(fptr, colnum, typecode, repeat, width, status)
    }
}

/// Get the 'equivalent' table column type.
///
/// This routine is similar to the ffgtcl routine (which returns the physical
/// datatype of the column, as stored in the FITS file) except that if the
/// TSCALn and TZEROn keywords are defined for the column, then it returns
/// the 'equivalent' datatype.  Thus, if the column is defined as '1I'  (short
/// integer) this routine may return the type as 'TUSHORT' or as 'TFLOAT'
/// depending on the TSCALn and TZEROn values.
///
/// Returns the datatype code of the column, as well as the vector
/// repeat count and (if it is an ASCII character column) the
/// width of the field or a unit string within the field.  This supports the
/// TFORMn = 'rAw' syntax for specifying arrays of substrings, so
/// if TFORMn = '60A12' then repeat = 60 and width = 12.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number
/// * `typecode` — (O) datatype code (21 = short, etc)
/// * `repeat`   — (O) repeat count of field
/// * `width`    — (O) if ASCII, width of field or unit string
/// * `status`   — (IO) error status
pub fn ffeqty_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    typecode: Option<&mut c_int>,
    repeat: Option<&mut c_long>,
    width: Option<&mut c_long>,
    status: &mut c_int,
) -> c_int {
    let mut trepeat: LONGLONG = 0;
    let mut twidth: LONGLONG = 0;

    ffeqtyll_safe(
        fptr,
        colnum,
        typecode,
        Some(&mut trepeat),
        Some(&mut twidth),
        status,
    );

    if let Some(repeat) = repeat {
        *repeat = trepeat as c_long;
    }

    if let Some(width) = width {
        *width = twidth as c_long;
    }

    *status
}

/// Get the 'equivalent' table column type.
///
/// This routine is similar to the ffgtcl routine (which returns the physical
/// datatype of the column, as stored in the FITS file) except that if the
/// TSCALn and TZEROn keywords are defined for the column, then it returns
/// the 'equivalent' datatype.  Thus, if the column is defined as '1I'  (short
/// integer) this routine may return the type as 'TUSHORT' or as 'TFLOAT'
/// depending on the TSCALn and TZEROn values.
///
/// Returns the datatype code of the column, as well as the vector
/// repeat count and (if it is an ASCII character column) the
/// width of the field or a unit string within the field.  This supports the
/// TFORMn = 'rAw' syntax for specifying arrays of substrings, so
/// if TFORMn = '60A12' then repeat = 60 and width = 12.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number
/// * `typecode` — (O) datatype code (21 = short, etc)
/// * `repeat`   — (O) repeat count of field
/// * `width`    — (O) if ASCII, width of field or unit string
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffeqtyll(
    fptr: *mut fitsfile,
    colnum: c_int,
    typecode: *mut c_int,
    repeat: *mut LONGLONG,
    width: *mut LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let repeat = repeat.as_mut();
        let width = width.as_mut();
        let typecode = typecode.as_mut();

        ffeqtyll_safe(fptr, colnum, typecode, repeat, width, status)
    }
}

/// Get the 'equivalent' table column type.
///
/// This routine is similar to the ffgtcl routine (which returns the physical
/// datatype of the column, as stored in the FITS file) except that if the
/// TSCALn and TZEROn keywords are defined for the column, then it returns
/// the 'equivalent' datatype.  Thus, if the column is defined as '1I'  (short
/// integer) this routine may return the type as 'TUSHORT' or as 'TFLOAT'
/// depending on the TSCALn and TZEROn values.
///
/// Returns the datatype code of the column, as well as the vector
/// repeat count and (if it is an ASCII character column) the
/// width of the field or a unit string within the field.  This supports the
/// TFORMn = 'rAw' syntax for specifying arrays of substrings, so
/// if TFORMn = '60A12' then repeat = 60 and width = 12.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number
/// * `typecode` — (O) datatype code (21 = short, etc)
/// * `repeat`   — (O) repeat count of field
/// * `width`    — (O) if ASCII, width of field or unit string
/// * `status`   — (IO) error status
pub fn ffeqtyll_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    mut typecode: Option<&mut c_int>,
    repeat: Option<&mut LONGLONG>,
    width: Option<&mut LONGLONG>,
    status: &mut c_int,
) -> c_int {
    let mut hdutype: c_int = 0;
    let mut decims: c_int = 0;

    let effcode: c_int;

    let mut min_val: f64;
    let mut max_val: f64;

    let mut lngzero: c_long = 0;
    let mut tmpwidth: c_long = 0;

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    if colnum < 1 || colnum > fptr.Fptr.tfield {
        *status = BAD_COL_NUM;
        return *status;
    }

    if ffghdt_safe(fptr, &mut hdutype, status) > 0 {
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_mut_slice();
    let ci = colnum as usize - 1; /* offset to correct column structure */

    if hdutype == ASCII_TBL {
        ffasfm_safe(
            &c[ci].tform,
            typecode.as_deref_mut(),
            Some(&mut tmpwidth),
            Some(&mut decims),
            status,
        );

        if let Some(width) = width {
            *width = tmpwidth as LONGLONG;
        }
        if let Some(repeat) = repeat {
            *repeat = 1;
        }
    } else {
        if let Some(typecode) = typecode.as_deref_mut() {
            *typecode = c[ci].tdatatype;
        }
        if let Some(width) = width {
            *width = c[ci].twidth as LONGLONG;
        }
        if let Some(repeat) = repeat {
            *repeat = c[ci].trepeat;
        }
    }

    /* return if caller is not interested in the typecode value */
    if typecode.is_none() {
        return *status;
    }

    let typecode = typecode.unwrap(); // Already checked not none above.

    /* check if the tscale and tzero keywords are defined, which might
    change the effective datatype of the column  */

    let tscale: f64 = c[ci].tscale;
    let tzero: f64 = c[ci].tzero;

    if tscale == 1.0 && tzero == 0.0 {
        /* no scaling */
        return *status;
    }

    let tcode: c_int = (*typecode).abs();

    match tcode {
        TBYTE => {
            /* binary table 'rB' column */
            min_val = 0.0;
            max_val = 255.0;
        }
        TSHORT => {
            min_val = -32768.0;
            max_val = 32767.0;
        }
        TLONG => {
            min_val = -2147483648.0;
            max_val = 2147483647.0;
        }
        TLONGLONG => {
            min_val = -9.223_372_036_854_776E18;
            max_val = 9.223_372_036_854_776E18;
        }

        _ => {
            /* don't have to deal with other data types */
            return *status;
        }
    }

    if tscale >= 0. {
        min_val = tzero + tscale * min_val;
        max_val = tzero + tscale * max_val;
    } else {
        max_val = tzero + tscale * min_val;
        min_val = tzero + tscale * max_val;
    }
    if tzero < 2147483648. {
        /* don't exceed range of 32-bit integer */
        lngzero = tzero as c_long;
    }
    let lngscale: c_long = tscale as c_long;

    if (tzero != 2147483648.0) && /* special value that exceeds integer range */
(tzero != 9223372036854775808.0) &&  /* indicates unsigned long long */
(lngzero as f64 != tzero || lngscale as f64 != tscale)
    {
        /* not integers? */
        /* floating point scaled values; just decide on required precision */
        if tcode == TBYTE || tcode == TSHORT {
            effcode = TFLOAT;
        } else {
            effcode = TDOUBLE;
        }
    /*
    In all the remaining cases, TSCALn and TZEROn are integers,
    and not equal to 1 and 0, respectively.
    */
    } else if (min_val == -128.) && (max_val == 127.) {
        effcode = TSBYTE;
    } else if (min_val >= -32768.0) && (max_val <= 32767.0) {
        effcode = TSHORT;
    } else if (min_val >= 0.0) && (max_val <= 65535.0) {
        effcode = TUSHORT;
    } else if (min_val >= -2147483648.0) && (max_val <= 2147483647.0) {
        effcode = TLONG;
    } else if (min_val >= 0.0) && (max_val < 4294967296.0) {
        effcode = TULONG;
    } else if (min_val >= -9.223_372_036_854_776E18) && (max_val <= 9.223_372_036_854_776E18) {
        effcode = TLONGLONG;
    } else if (min_val >= 0.0) && (max_val <= 1.844_674_407_370_955_2E19) {
        effcode = TULONGLONG;
    } else {
        /* exceeds the range of a 64-bit integer */
        effcode = TDOUBLE;
    }

    /* return the effective datatype code (negative if variable length col.) */
    if *typecode < 0 {
        /* variable length array column */
        *typecode = -effcode;
    } else {
        *typecode = effcode;
    }
    *status
}

/// Get the number of columns in the table (= TFIELDS keyword)
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ncols`  — (O) number of columns in the table
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgncl(
    fptr: *mut fitsfile,
    ncols: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let ncols = ncols.as_mut().expect(NULL_MSG);

        ffgncl_safe(fptr, ncols, status)
    }
}

/// Get the number of columns in the table (= TFIELDS keyword)
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `ncols`  — (O) number of columns in the table
/// * `status` — (IO) error status
pub fn ffgncl_safe(fptr: &mut fitsfile, ncols: &mut c_int, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        /* rescan header */
        if ffrdef_safe(fptr, status) > 0 {
            return *status;
        }
    }

    if fptr.Fptr.hdutype == IMAGE_HDU {
        *status = NOT_TABLE;
        return *status;
    }

    *ncols = fptr.Fptr.tfield;
    *status
}

/// Get the number of rows in the table (= NAXIS2 keyword)
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nrows`  — (O) number of rows in the table
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgnrw(
    fptr: *mut fitsfile,
    nrows: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let nrows = nrows.as_mut().expect(NULL_MSG);

        ffgnrw_safe(fptr, nrows, status)
    }
}

/// Get the number of rows in the table (= NAXIS2 keyword)
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nrows`  — (O) number of rows in the table
/// * `status` — (IO) error status
pub fn ffgnrw_safe(fptr: &mut fitsfile, nrows: &mut c_long, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    if fptr.Fptr.hdutype == IMAGE_HDU {
        return {
            *status = NOT_TABLE;
            *status
        };
    }

    /* the NAXIS2 keyword may not be up to date, so use the structure value */
    *nrows = fptr.Fptr.numrows as c_long;
    *status
}

/// Get the number of rows in the table (= NAXIS2 keyword)
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nrows`  — (O) number of rows in the table
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgnrwll(
    fptr: *mut fitsfile,
    nrows: *mut LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let nrows = nrows.as_mut().expect(NULL_MSG); //WARNING: Not checked in original code

        ffgnrwll_safe(fptr, nrows, status)
    }
}

/// Get the number of rows in the table (= NAXIS2 keyword)
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nrows`  — (O) number of rows in the table
/// * `status` — (IO) error status
pub fn ffgnrwll_safe(fptr: &mut fitsfile, nrows: &mut LONGLONG, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        /* rescan header */
        if ffrdef_safe(fptr, status) > 0 {
            return *status;
        }
    }
    if fptr.Fptr.hdutype == IMAGE_HDU {
        *status = NOT_TABLE;
        return *status;
    }

    /* the NAXIS2 keyword may not be up to date, so use the structure value */
    *nrows = fptr.Fptr.numrows;
    *status
}

/// get ASCII table column keyword values
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) column number
/// * `ttype`  — (O) TTYPEn keyword value
/// * `tbcol`  — (O) TBCOLn keyword value
/// * `tunit`  — (O) TUNITn keyword value
/// * `tform`  — (O) TFORMn keyword value
/// * `tscal`  — (O) TSCALn keyword value
/// * `tzero`  — (O) TZEROn keyword value
/// * `tnull`  — (O) TNULLn keyword value
/// * `tdisp`  — (O) TDISPn keyword value
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgacl(
    fptr: *mut fitsfile,
    colnum: c_int,
    ttype: *mut c_char,
    tbcol: *mut c_long,
    tunit: *mut c_char,
    tform: *mut c_char,
    tscal: *mut f64,
    tzero: *mut f64,
    tnull: *mut c_char,
    tdisp: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        let tbcol = tbcol.as_mut();
        let tscal = tscal.as_mut();
        let tzero = tzero.as_mut();

        let ttype = match ttype.is_null() {
            true => None,
            false => Some(cast_slice_mut(slice::from_raw_parts_mut(ttype, FLEN_VALUE))),
        };

        let tunit = match tunit.is_null() {
            true => None,
            false => Some(cast_slice_mut(slice::from_raw_parts_mut(tunit, FLEN_VALUE))),
        };

        let tform = match tform.is_null() {
            true => None,
            false => Some(cast_slice_mut(slice::from_raw_parts_mut(tform, FLEN_VALUE))),
        };

        let tdisp = match tdisp.is_null() {
            true => None,
            false => Some(cast_slice_mut(slice::from_raw_parts_mut(tdisp, FLEN_VALUE))),
        };

        let tnull = match tnull.is_null() {
            true => None,
            false => Some(cast_slice_mut(slice::from_raw_parts_mut(tnull, FLEN_VALUE))),
        };

        ffgacl_safe(
            fptr, colnum, ttype, tbcol, tunit, tform, tscal, tzero, tnull, tdisp, status,
        )
    }
}

/// get ASCII table column keyword values
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) column number
/// * `ttype`  — (O) TTYPEn keyword value
/// * `tbcol`  — (O) TBCOLn keyword value
/// * `tunit`  — (O) TUNITn keyword value
/// * `tform`  — (O) TFORMn keyword value
/// * `tscal`  — (O) TSCALn keyword value
/// * `tzero`  — (O) TZEROn keyword value
/// * `tnull`  — (O) TNULLn keyword value
/// * `tdisp`  — (O) TDISPn keyword value
/// * `status` — (IO) error status
pub fn ffgacl_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    ttype: Option<&mut [c_char]>,
    tbcol: Option<&mut c_long>,
    tunit: Option<&mut [c_char]>,
    tform: Option<&mut [c_char]>,
    tscal: Option<&mut f64>,
    tzero: Option<&mut f64>,
    tnull: Option<&mut [c_char]>,
    tdisp: Option<&mut [c_char]>,
    status: &mut c_int,
) -> c_int {
    let mut name: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut tstatus;

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        /* rescan header */
        if ffrdef_safe(fptr, status) > 0 {
            return *status;
        }
    }

    if colnum < 1 || colnum > fptr.Fptr.tfield {
        *status = BAD_COL_NUM;
        return *status;
    }

    /* get what we can from the column structure */
    /* point to first column structure */
    let c = fptr.Fptr.get_tableptr_as_mut_slice();
    let ci = colnum as usize - 1; /* offset to the correct column */

    if let Some(ttype) = ttype {
        strcpy_safe(ttype, cast_slice(&c[ci].ttype));
    }
    if let Some(tbcol) = tbcol {
        /* first col is 1, not 0 */
        *tbcol = ((c[ci].tbcol) + 1) as c_long;
    }
    if let Some(tform) = tform {
        strcpy_safe(tform, cast_slice(&c[ci].tform));
    }
    if let Some(tscal) = tscal {
        *tscal = c[ci].tscale;
    }
    if let Some(tzero) = tzero {
        *tzero = c[ci].tzero;
    }

    if let Some(tnull) = tnull {
        strcpy_safe(tnull, cast_slice(&c[ci].strnull));
    }

    /* read keywords to get additional parameters */
    if let Some(tunit) = tunit {
        ffkeyn_safe(cs!(c"TUNIT"), colnum, &mut name, status);
        tstatus = 0;

        tunit[0] = 0;
        ffgkys_safe(fptr, &name, tunit, Some(&mut comm), &mut tstatus);
    }

    if let Some(tdisp) = tdisp {
        ffkeyn_safe(cs!(c"TDISP"), colnum, &mut name, status);
        tstatus = 0;

        tdisp[0] = 0;
        ffgkys_safe(fptr, &name, tdisp, Some(&mut comm), &mut tstatus);
    }
    *status
}

/// get BINTABLE column keyword values
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) column number
/// * `ttype`  — (O) TTYPEn keyword value
/// * `tunit`  — (O) TUNITn keyword value
/// * `dtype`  — (O) datatype char: I, J, E, D, etc.
/// * `repeat` — (O) vector column repeat count
/// * `tscal`  — (O) TSCALn keyword value
/// * `tzero`  — (O) TZEROn keyword value
/// * `tnull`  — (O) TNULLn keyword value integer cols only
/// * `tdisp`  — (O) TDISPn keyword value
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgbcl(
    fptr: *mut fitsfile,
    colnum: c_int,
    ttype: *mut c_char,
    tunit: *mut c_char,
    dtype: *mut [c_char; 2],
    repeat: *mut c_long,
    tscal: *mut f64,
    tzero: *mut f64,
    tnull: *mut c_long,
    tdisp: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        let dtype = dtype.as_mut();
        let repeat = repeat.as_mut();
        let tscal = tscal.as_mut();
        let tzero = tzero.as_mut();
        let tnull = tnull.as_mut();

        let ttype = if ttype.is_null() {
            None
        } else {
            Some(slice::from_raw_parts_mut(ttype, FLEN_VALUE))
        };

        let tunit = if tunit.is_null() {
            None
        } else {
            Some(slice::from_raw_parts_mut(tunit, FLEN_VALUE))
        };

        let tdisp = if tdisp.is_null() {
            None
        } else {
            Some(slice::from_raw_parts_mut(tdisp, FLEN_VALUE))
        };

        ffgbcl_safe(
            fptr, colnum, ttype, tunit, dtype, repeat, tscal, tzero, tnull, tdisp, status,
        )
    }
}

/// get BINTABLE column keyword values
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) column number
/// * `ttype`  — (O) TTYPEn keyword value
/// * `tunit`  — (O) TUNITn keyword value
/// * `dtype`  — (O) datatype char: I, J, E, D, etc.
/// * `repeat` — (O) vector column repeat count
/// * `tscal`  — (O) TSCALn keyword value
/// * `tzero`  — (O) TZEROn keyword value
/// * `tnull`  — (O) TNULLn keyword value integer cols only
/// * `tdisp`  — (O) TDISPn keyword value
/// * `status` — (IO) error status
pub fn ffgbcl_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    ttype: Option<&mut [c_char]>,
    tunit: Option<&mut [c_char]>,
    dtype: Option<&mut [c_char; 2]>,
    repeat: Option<&mut c_long>,
    tscal: Option<&mut f64>,
    tzero: Option<&mut f64>,
    tnull: Option<&mut c_long>,
    tdisp: Option<&mut [c_char]>,
    status: &mut c_int,
) -> c_int {
    let mut trepeat = 0;
    let mut ttnull = 0;

    if *status > 0 {
        return *status;
    }

    ffgbclll_safe(
        fptr,
        colnum,
        ttype,
        tunit,
        dtype,
        Some(&mut trepeat),
        tscal,
        tzero,
        Some(&mut ttnull),
        tdisp,
        status,
    );

    if let Some(repeat) = repeat {
        *repeat = trepeat as c_long;
    }

    if let Some(tnull) = tnull {
        *tnull = ttnull as c_long;
    }

    *status
}

/// get BINTABLE column keyword values
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) column number
/// * `ttype`  — (O) TTYPEn keyword value
/// * `tunit`  — (O) TUNITn keyword value
/// * `dtype`  — (O) datatype char: I, J, E, D, etc.
/// * `repeat` — (O) vector column repeat count
/// * `tscal`  — (O) TSCALn keyword value
/// * `tzero`  — (O) TZEROn keyword value
/// * `tnull`  — (O) TNULLn keyword value integer cols only
/// * `tdisp`  — (O) TDISPn keyword value
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgbclll(
    fptr: *mut fitsfile,
    colnum: c_int,
    ttype: *mut c_char,
    tunit: *mut c_char,
    dtype: *mut [c_char; 2],
    repeat: *mut LONGLONG,
    tscal: *mut f64,
    tzero: *mut f64,
    tnull: *mut LONGLONG,
    tdisp: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let dtype = dtype.as_mut();
        let repeat = repeat.as_mut();
        let tscal = tscal.as_mut();
        let tzero = tzero.as_mut();
        let tnull = tnull.as_mut();

        let ttype = if ttype.is_null() {
            None
        } else {
            Some(slice::from_raw_parts_mut(ttype, FLEN_VALUE))
        };

        let tunit = if tunit.is_null() {
            None
        } else {
            Some(slice::from_raw_parts_mut(tunit, FLEN_VALUE))
        };

        let tdisp = if tdisp.is_null() {
            None
        } else {
            Some(slice::from_raw_parts_mut(tdisp, FLEN_VALUE))
        };

        ffgbclll_safe(
            fptr, colnum, ttype, tunit, dtype, repeat, tscal, tzero, tnull, tdisp, status,
        )
    }
}

/// get BINTABLE column keyword values
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `colnum` — (I) column number
/// * `ttype`  — (O) TTYPEn keyword value
/// * `tunit`  — (O) TUNITn keyword value
/// * `dtype`  — (O) datatype char: I, J, E, D, etc.
/// * `repeat` — (O) vector column repeat count
/// * `tscal`  — (O) TSCALn keyword value
/// * `tzero`  — (O) TZEROn keyword value
/// * `tnull`  — (O) TNULLn keyword value integer cols only
/// * `tdisp`  — (O) TDISPn keyword value
/// * `status` — (IO) error status
pub fn ffgbclll_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    ttype: Option<&mut [c_char]>,
    tunit: Option<&mut [c_char]>,
    dtype: Option<&mut [c_char; 2]>,
    repeat: Option<&mut LONGLONG>,
    tscal: Option<&mut f64>,
    tzero: Option<&mut f64>,
    tnull: Option<&mut LONGLONG>,
    tdisp: Option<&mut [c_char]>,
    status: &mut c_int,
) -> c_int {
    let mut name: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

    let mut tstatus;

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    if colnum < 1 || colnum > fptr.Fptr.tfield {
        *status = BAD_COL_NUM;
        return *status;
    }

    /* get what we can from the column structure */
    /* set pointer to first column */
    let c = fptr.Fptr.get_tableptr_as_slice();
    let ci = colnum as usize - 1; /* offset to correct column structure */

    if let Some(ttype) = ttype {
        strcpy_safe(ttype, &c[ci].ttype);
    }

    if let Some(dtype) = dtype {
        if c[ci].tdatatype < 0 {
            /* add the "P" prefix for */
            strcpy_safe(dtype, cs!(c"P")); /* variable length columns */
        } else {
            dtype[0] = 0;
        }

        let abs_datatype = (c[ci].tdatatype).abs();

        if abs_datatype == TBIT {
            strcat_safe(dtype, cs!(c"X"));
        } else if abs_datatype == TBYTE {
            strcat_safe(dtype, cs!(c"B"));
        } else if abs_datatype == TLOGICAL {
            strcat_safe(dtype, cs!(c"L"));
        } else if abs_datatype == TSTRING {
            strcat_safe(dtype, cs!(c"A"));
        } else if abs_datatype == TSHORT {
            strcat_safe(dtype, cs!(c"I"));
        } else if abs_datatype == TLONG {
            strcat_safe(dtype, cs!(c"J"));
        } else if abs_datatype == TLONGLONG {
            strcat_safe(dtype, cs!(c"K"));
        } else if abs_datatype == TFLOAT {
            strcat_safe(dtype, cs!(c"E"));
        } else if abs_datatype == TDOUBLE {
            strcat_safe(dtype, cs!(c"D"));
        } else if abs_datatype == TCOMPLEX {
            strcat_safe(dtype, cs!(c"C"));
        } else if abs_datatype == TDBLCOMPLEX {
            strcat_safe(dtype, cs!(c"M"));
        }
    }

    if let Some(repeat) = repeat {
        *repeat = c[ci].trepeat;
    }

    if let Some(tscal) = tscal {
        *tscal = c[ci].tscale;
    }

    if let Some(tzero) = tzero {
        *tzero = c[ci].tzero;
    }

    if let Some(tnull) = tnull {
        *tnull = c[ci].tnull;
    }

    /* read keywords to get additional parameters */

    if let Some(tunit) = tunit {
        ffkeyn_safe(cs!(c"TUNIT"), colnum, &mut name, status);
        tstatus = 0;

        tunit[0] = 0;
        ffgkys_safe(fptr, &name, tunit, Some(&mut comm), &mut tstatus);
    }

    if let Some(tdisp) = tdisp {
        ffkeyn_safe(cs!(c"TDISP"), colnum, &mut name, status);
        tstatus = 0;

        tdisp[0] = 0;
        ffgkys_safe(fptr, &name, tdisp, Some(&mut comm), &mut tstatus);
    }

    *status
}

/// Return the number of the Current HDU in the FITS file.  The primary array
/// is HDU number 1.  Note that this is one of the few cfitsio routines that
/// does not return the error status value as the value of the function.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `chdunum` — (O) number of the CHDU; 1 = primary array
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffghdn(fptr: *mut fitsfile, chdunum: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);

        *chdunum = (fptr.HDUposition) + 1;
        *chdunum
    }
}

/// Return the number of the Current HDU in the FITS file.  The primary array
/// is HDU number 1.  Note that this is one of the few cfitsio routines that
/// does not return the error status value as the value of the function.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `chdunum` — (O) number of the CHDU; 1 = primary array
pub fn ffghdn_safe(fptr: &mut fitsfile, chdunum: &mut c_int) -> c_int {
    *chdunum = (fptr.HDUposition) + 1;
    *chdunum
}

/// Return the address (= byte offset) in the FITS file to the beginning of
/// the current HDU, the beginning of the data unit, and the end of the data unit.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `headstart` — (O) byte offset to beginning of CHDU
/// * `datastart` — (O) byte offset to beginning of next HDU
/// * `dataend`   — (O) byte offset to beginning of next HDU
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffghadll(
    fptr: *mut fitsfile,
    headstart: *mut LONGLONG,
    datastart: *mut LONGLONG,
    dataend: *mut LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let headstart = headstart.as_mut();
        let datastart = datastart.as_mut();
        let dataend = dataend.as_mut();

        ffghadll_safe(fptr, headstart, datastart, dataend, status)
    }
}

/// Return the address (= byte offset) in the FITS file to the beginning of
/// the current HDU, the beginning of the data unit, and the end of the data unit.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `headstart` — (O) byte offset to beginning of CHDU
/// * `datastart` — (O) byte offset to beginning of next HDU
/// * `dataend`   — (O) byte offset to beginning of next HDU
/// * `status`    — (IO) error status
pub fn ffghadll_safe(
    fptr: &mut fitsfile,
    headstart: Option<&mut LONGLONG>,
    datastart: Option<&mut LONGLONG>,
    dataend: Option<&mut LONGLONG>,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        if ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status) > 0 {
            return *status;
        }
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    let fptr_headstart = fptr.Fptr.get_headstart_as_slice();
    if let Some(headstart) = headstart {
        *headstart = fptr_headstart[fptr.Fptr.curhdu as usize];
    }
    if let Some(datastart) = datastart {
        *datastart = fptr.Fptr.datastart;
    }
    if let Some(dataend) = dataend {
        *dataend = fptr_headstart[(fptr.Fptr.curhdu as usize) + 1];
    }
    *status
}

/// Return the address (= byte offset) in the FITS file to the beginning of
/// the current HDU, the beginning of the data unit, and the end of the data unit.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `headstart` — (O) byte offset to beginning of CHDU
/// * `datastart` — (O) byte offset to beginning of next HDU
/// * `dataend`   — (O) byte offset to beginning of next HDU
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffghof(
    fptr: *mut fitsfile,
    headstart: *mut off_t,
    datastart: *mut off_t,
    dataend: *mut off_t,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let headstart = headstart.as_mut();
        let datastart = datastart.as_mut();
        let dataend = dataend.as_mut();

        ffghof_safe(fptr, headstart, datastart, dataend, status)
    }
}

/// Return the address (= byte offset) in the FITS file to the beginning of
/// the current HDU, the beginning of the data unit, and the end of the data unit.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `headstart` — (O) byte offset to beginning of CHDU
/// * `datastart` — (O) byte offset to beginning of next HDU
/// * `dataend`   — (O) byte offset to beginning of next HDU
/// * `status`    — (IO) error status
pub fn ffghof_safe(
    fptr: &mut fitsfile,
    headstart: Option<&mut off_t>,
    datastart: Option<&mut off_t>,
    dataend: Option<&mut off_t>,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        if ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status) > 0 {
            return *status;
        };
    } else if fptr.Fptr.datastart == DATA_UNDEFINED && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    };

    let fptr_headstart = fptr.Fptr.get_headstart_as_slice();

    if let Some(headstart) = headstart {
        *headstart = fptr_headstart[fptr.Fptr.curhdu as usize] as off_t;
    }

    if let Some(datastart) = datastart {
        *datastart = fptr.Fptr.datastart as off_t;
    }

    if let Some(dataend) = dataend {
        *dataend = fptr_headstart[(fptr.Fptr.curhdu as usize) + 1] as off_t;
    }
    *status
}

/// Return the address (= byte offset) in the FITS file to the beginning of
/// the current HDU, the beginning of the data unit, and the end of the data unit.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `headstart` — (O) byte offset to beginning of CHDU
/// * `datastart` — (O) byte offset to beginning of next HDU
/// * `dataend`   — (O) byte offset to beginning of next HDU
/// * `status`    — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffghad(
    fptr: *mut fitsfile,
    headstart: *mut c_long,
    datastart: *mut c_long,
    dataend: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let headstart = headstart.as_mut();
        let datastart = datastart.as_mut();
        let dataend = dataend.as_mut();

        ffghad_safe(fptr, headstart, datastart, dataend, status)
    }
}

/// Return the address (= byte offset) in the FITS file to the beginning of
/// the current HDU, the beginning of the data unit, and the end of the data unit.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `headstart` — (O) byte offset to beginning of CHDU
/// * `datastart` — (O) byte offset to beginning of next HDU
/// * `dataend`   — (O) byte offset to beginning of next HDU
/// * `status`    — (IO) error status
pub fn ffghad_safe(
    fptr: &mut fitsfile,
    headstart: Option<&mut c_long>,
    datastart: Option<&mut c_long>,
    dataend: Option<&mut c_long>,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    let mut shead: LONGLONG = 0;
    let mut sdata: LONGLONG = 0;
    let mut edata: LONGLONG = 0;

    ffghadll_safe(
        fptr,
        Some(&mut shead),
        Some(&mut sdata),
        Some(&mut edata),
        status,
    );

    if let Some(headstart) = headstart {
        if shead > LONG_MAX as LONGLONG {
            *status = NUM_OVERFLOW;
        } else {
            *headstart = shead as c_long;
        }
    }

    if let Some(datastart) = datastart {
        if sdata > LONG_MAX as LONGLONG {
            *status = NUM_OVERFLOW;
        } else {
            *datastart = sdata as c_long;
        }
    }

    if let Some(dataend) = dataend {
        if edata > LONG_MAX as LONGLONG {
            *status = NUM_OVERFLOW;
        } else {
            *dataend = edata as c_long;
        }
    }

    *status
}

/// read the required keywords of the CHDU and initialize the corresponding
/// structure elements that describe the format of the HDU
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `hdutype` — (O) type of HDU
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffrhdu(
    fptr: *mut fitsfile,
    hdutype: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        ffrhdu_safe(fptr, hdutype.as_mut(), status)
    }
}

/// read the required keywords of the CHDU and initialize the corresponding
/// structure elements that describe the format of the HDU
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `hdutype` — (O) type of HDU
/// * `status`  — (IO) error status
pub fn ffrhdu_safe(fptr: &mut fitsfile, hdutype: Option<&mut c_int>, status: &mut c_int) -> c_int {
    let mut tstatus;
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut name: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut value: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut xname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut urltype: [c_char; 20] = [0; 20];

    let mut hdutype = hdutype;

    if *status > 0 {
        return *status;
    }

    if ffgrec_safe(fptr, 1, Some(&mut card), status) > 0 {
        /* get the 80-byte card */
        ffpmsg_str("Cannot read first keyword in header (ffrhdu).");
        return *status;
    }

    strncpy_safe(&mut name, &card, 8); /* first 8 characters = the keyword name */
    name[8] = 0;

    for ii in (0..8).rev() {
        /* replace trailing blanks with nulls */
        if name[ii] == bb(b' ') {
            name[ii] = 0;
        } else {
            break;
        };
    }

    /* parse value and comment */
    if ffpsvc_safe(&card, &mut value, Some(&mut comm), status) > 0 {
        ffpmsg_str("Cannot read value of first keyword in header (ffrhdu):");
        ffpmsg_slice(&card);
        return *status;
    }

    if strcmp_safe(&name, cs!(c"SIMPLE")) == 0 {
        /* this is the primary array */

        ffpinit(fptr, status); /* initialize the primary array */

        if let Some(hdutype) = hdutype.as_deref_mut() {
            *hdutype = 0;
        };
    } else if strcmp_safe(&name, cs!(c"XTENSION")) == 0 {
        /* this is an XTENSION keyword */

        if ffc2s(&value, &mut xname, status) > 0 {
            /* get the value string */
            ffpmsg_str("Bad value string for XTENSION keyword:");
            ffpmsg_slice(&value);
            return *status;
        }

        let mut xtension = 0; // &xname
        while xname[xtension] == bb(b' ') {
            /* ignore any leading spaces in name */
            xtension += 1;
        }

        if strcmp_safe(&xname[xtension..], cs!(c"TABLE")) == 0 {
            ffainit(fptr, status); /* initialize the ASCII table */
            if let Some(hdutype) = hdutype.as_deref_mut() {
                *hdutype = 1;
            };
        } else if strcmp_safe(&xname[xtension..], cs!(c"BINTABLE")) == 0
            || strcmp_safe(&xname[xtension..], cs!(c"A3DTABLE")) == 0
            || strcmp_safe(&xname[xtension..], cs!(c"3DTABLE")) == 0
        {
            ffbinit(fptr, status); /* initialize the binary table */
            if let Some(hdutype) = hdutype.as_deref_mut() {
                *hdutype = 2;
            };
        } else {
            tstatus = 0;
            ffpinit(fptr, &mut tstatus); /* probably an IMAGE extension */

            if tstatus == UNKNOWN_EXT
                && let Some(hdutype) = hdutype.as_deref_mut()
            {
                /* don't recognize this extension type */
                *hdutype = -1;
            } else {
                *status = tstatus;
                if let Some(hdutype) = hdutype {
                    *hdutype = 0;
                };
            };
        };
    } else {
        /*  not the start of a new extension */

        if card[0] == 0 || card[0] == 10 {
            /* some editors append this character to EOF */
            *status = END_OF_FILE;
        } else {
            *status = UNKNOWN_REC; /* found unknown type of record */
            ffpmsg_str("Extension doesn't start with SIMPLE or XTENSION keyword. (ffrhdu)");
            ffpmsg_slice(&card);
        };
    }

    /*  compare the starting position of the next HDU (if any) with the size */
    /*  of the whole file to see if this is the last HDU in the file */

    let headstart = fptr.Fptr.get_headstart_as_slice();
    let filesize = headstart[(fptr.Fptr.curhdu + 1) as usize];

    if filesize < fptr.Fptr.logfilesize {
        fptr.Fptr.lasthdu = 0; /* no, not the last HDU */
    } else {
        fptr.Fptr.lasthdu = 1; /* yes, this is the last HDU */

        /* special code for mem:// type files (FITS file in memory) */
        /* Allocate enough memory to hold the entire HDU. */
        /* Without this code, CFITSIO would repeatedly realloc  memory */
        /* to incrementally increase the size of the file by 2880 bytes */
        /* at a time, until it reached the final size */

        ffurlt_safe(fptr, &mut urltype, status);

        if strcmp_safe(&urltype, cs!(c"mem://")) == 0
            || strcmp_safe(&urltype, cs!(c"memkeep://")) == 0
        {
            fftrun(fptr, filesize, status);
        };
    }
    *status
}

/// initialize the parameters defining the structure of the primary array
/// or an Image extension
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub(crate) fn ffpinit(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    unsafe {
        let mut simple: c_int = 0;
        let mut bitpix: c_int = 0;
        let mut naxis: c_int = 0;
        let mut extend: c_int = 0;
        let mut nspace: c_int = 0;
        let mut ttype: c_int = 0;
        let mut bytlen: LONGLONG = 0;
        let mut ii: usize;
        let mut pcount: c_long = 0;
        let mut gcount: c_long = 0;
        let mut naxes: [LONGLONG; 999] = [0; 999];
        let mut npix: LONGLONG;
        let mut blank: LONGLONG = 0;
        let mut bscale: f64 = 0.0;
        let mut bzero: f64 = 0.0;
        let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];

        if *status > 0 {
            return *status;
        }

        /* reset position to the correct HDU if necessary */
        if fptr.HDUposition != fptr.Fptr.curhdu {
            ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
        }

        fptr.Fptr.hdutype = IMAGE_HDU; /* primary array or IMAGE extension  */
        fptr.Fptr.headend = fptr.Fptr.logfilesize; /* set max size */

        let mut groups = 0;
        let mut tstatus = *status;

        /* get all the descriptive info about this HDU */
        ffgphd(
            fptr,
            999,
            &mut simple,
            &mut bitpix,
            Some(&mut naxis),
            &mut naxes,
            &mut pcount,
            &mut gcount,
            &mut extend,
            &mut bscale,
            &mut bzero,
            &mut blank,
            &mut nspace,
            status,
        );

        if *status == NOT_IMAGE {
            *status = tstatus; /* ignore 'unknown extension type' error */
        } else if *status > 0 {
            return *status;
        }

        /*
           the logical end of the header is 80 bytes before the current position,
           minus any trailing blank keywords just before the END keyword.
        */
        fptr.Fptr.headend = fptr.Fptr.nextkey - (80 * (LONGLONG::from(nspace) + 1));

        /* the data unit begins at the beginning of the next logical block */
        fptr.Fptr.datastart = ((fptr.Fptr.nextkey - 80) / BL!() + 1) * BL!();

        /* test for 'random groups' */
        if naxis > 0 && naxes[0] == 0 {
            tstatus = 0;
            ffmaky_safe(fptr, 2, status); /* reset to beginning of header */
            if ffgkyl_safe(
                fptr,
                cs!(c"GROUPS"),
                &mut groups,
                Some(&mut comm),
                &mut tstatus,
            ) != 0
            {
                groups = 0; /* GROUPS keyword not found */
            };
        }
        if bitpix == BYTE_IMG {
            /* test  bitpix and set the datatype code */
            ttype = TBYTE;
            bytlen = 1;
        } else if bitpix == SHORT_IMG {
            ttype = TSHORT;
            bytlen = 2;
        } else if bitpix == LONG_IMG {
            ttype = TLONG;
            bytlen = 4;
        } else if bitpix == LONGLONG_IMG {
            ttype = TLONGLONG;
            bytlen = 8;
        } else if bitpix == FLOAT_IMG {
            ttype = TFLOAT;
            bytlen = 4;
        } else if bitpix == DOUBLE_IMG {
            ttype = TDOUBLE;
            bytlen = 8;
        }

        /*   calculate the size of the primary array  */
        fptr.Fptr.imgdim = naxis;
        if naxis == 0 {
            npix = 0;
        } else {
            if groups != 0 {
                npix = 1; /* NAXIS1 = 0 is a special flag for 'random groups' */
            } else {
                npix = naxes[0];
            }
            fptr.Fptr.imgnaxis[0] = naxes[0];

            ii = 1;
            while ii < naxis as usize {
                npix *= naxes[ii]; /* calc number of pixels in the array */
                fptr.Fptr.imgnaxis[ii] = naxes[ii];
                ii += 1
            }
        }

        /*
           now we know everything about the array; just fill in the parameters:
           the next HDU begins in the next logical block after the data
        */
        *fptr.Fptr.headstart.offset(fptr.Fptr.curhdu as isize + 1) = fptr.Fptr.datastart
            + (((pcount as LONGLONG) + npix) * bytlen * (gcount as LONGLONG) + (BL!() - 1)) / BL!()
                * BL!();

        /*
          initialize the fictitious heap starting address (immediately following
          the array data) and a zero length heap.  This is used to find the
          end of the data when checking the fill values in the last block.
        */
        fptr.Fptr.heapstart = (npix + pcount as LONGLONG) * bytlen * (gcount as LONGLONG);
        fptr.Fptr.heapsize = 0;

        fptr.Fptr.compressimg = 0; /* this is not a compressed image */

        if naxis == 0 {
            fptr.Fptr.rowlength = 0; /* rows have zero length */
            fptr.Fptr.tfield = 0; /* table has no fields   */

            /* free the tile-compressed image cache, if it exists */
            if !fptr.Fptr.tilerow.is_null() {
                // HEAP DEALLOCATION
                let mut tilestruct_lock = TILE_STRUCTS.lock().unwrap();
                let _ = tilestruct_lock.remove_entry(&(&raw const fptr.Fptr as usize));
                drop(tilestruct_lock);

                fptr.Fptr.tileanynull = ptr::null_mut();
                fptr.Fptr.tiletype = ptr::null_mut();
                fptr.Fptr.tiledatasize = ptr::null_mut();
                fptr.Fptr.tilenullarray = ptr::null_mut();
                fptr.Fptr.tiledata = ptr::null_mut();
                fptr.Fptr.tilerow = ptr::null_mut();
            }

            if !fptr.Fptr.tableptr.is_null() {
                /* free memory for the old CHDU */

                // HEAP DEALLOCATION
                let mut alloc_lock = ALLOCATIONS.lock().unwrap();
                let alloc = alloc_lock.remove(&(fptr.Fptr.tableptr as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(fptr.Fptr.tableptr, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        fptr.Fptr.tableptr,
                        fptr.Fptr.tfield as usize,
                        fptr.Fptr.tfield as usize,
                    );
                }
            }

            fptr.Fptr.tableptr = ptr::null_mut(); /* set a null table structure pointer */
            fptr.Fptr.numrows = 0;
            fptr.Fptr.origrows = 0;
        } else {
            /*
              The primary array is actually interpreted as a binary table.  There
              are two columns: the first column contains the group parameters if any.
              The second column contains the primary array of data as a single vector
              column element. In the case of 'random grouped' format, each group
              is stored in a separate row of the table.
            */
            /* the number of rows is equal to the number of groups */
            fptr.Fptr.numrows = gcount as LONGLONG;
            fptr.Fptr.origrows = gcount as LONGLONG;

            fptr.Fptr.rowlength = (npix + pcount as LONGLONG) * bytlen as LONGLONG; /* total size */
            fptr.Fptr.tfield = 2; /* 2 fields: group params and the image */

            /* free the tile-compressed image cache, if it exists */
            if !fptr.Fptr.tilerow.is_null() {
                // HEAP DEALLOCATION
                let mut tilestruct_lock = TILE_STRUCTS.lock().unwrap();
                let _ = tilestruct_lock.remove_entry(&(&raw const fptr.Fptr as usize));
                drop(tilestruct_lock);

                fptr.Fptr.tileanynull = ptr::null_mut();
                fptr.Fptr.tiletype = ptr::null_mut();
                fptr.Fptr.tiledatasize = ptr::null_mut();
                fptr.Fptr.tilenullarray = ptr::null_mut();
                fptr.Fptr.tiledata = ptr::null_mut();
                fptr.Fptr.tilerow = ptr::null_mut();
            }

            if !fptr.Fptr.tableptr.is_null() {
                /* free memory for the old CHDU */
                // HEAP DEALLOCATION
                let mut alloc_lock = ALLOCATIONS.lock().unwrap();
                let alloc = alloc_lock.remove(&(fptr.Fptr.tableptr as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(fptr.Fptr.tableptr, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        fptr.Fptr.tableptr,
                        fptr.Fptr.tfield as usize,
                        fptr.Fptr.tfield as usize,
                    );
                }
            }

            // HEAP ALLOCATION
            let mut colptr = Vec::new();
            if colptr.try_reserve_exact(2).is_err() {
                ffpmsg_str("malloc failed to get memory for FITS array descriptors (ffpinit)");
                fptr.Fptr.tableptr = ptr::null_mut(); /* set a null table structure pointer */
                *status = ARRAY_TOO_BIG;
                return *status;
            } else {
                colptr.resize(2, tcolumn::default());
            }

            /* the first column represents the group parameters, if any */
            colptr[0].tbcol = 0;
            colptr[0].tdatatype = ttype;
            colptr[0].twidth = bytlen as c_long;
            colptr[0].trepeat = pcount as LONGLONG;
            colptr[0].tscale = 1.0;
            colptr[0].tzero = 0.0;
            colptr[0].tnull = blank;

            /* the second column represents the image array */
            colptr[1].tbcol = pcount as LONGLONG * bytlen; /* col starts after the group parms */
            colptr[1].tdatatype = ttype;
            colptr[1].twidth = bytlen as c_long;
            colptr[1].trepeat = npix;
            colptr[1].tscale = bscale;
            colptr[1].tzero = bzero;
            colptr[1].tnull = blank;

            /* copy the table structure address to the fitsfile structure */

            let (p, l, c) = vec_into_raw_parts(colptr);
            ALLOCATIONS.lock().unwrap().insert(p as usize, (l, c));
            fptr.Fptr.tableptr = p;
        }

        let headstart = fptr.Fptr.get_headstart_as_slice();

        /* reset next keyword pointer to the start of the header */
        fptr.Fptr.nextkey = headstart[fptr.Fptr.curhdu as usize];

        *status
    }
}

/// initialize the parameters defining the structure of an ASCII table
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub(crate) fn ffainit(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    unsafe {
        let mut nspace: c_int;
        let mut tfield: c_long = 0;
        let mut pcount: LONGLONG = 0;
        let mut rowlen: LONGLONG = 0;
        let mut nrows: LONGLONG = 0;
        let mut tbcoln: LONGLONG;
        let mut colptr: *mut tcolumn = ptr::null_mut();
        let mut name: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
        let mut value: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
        let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
        let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
        let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

        if *status > 0 {
            return *status;
        }

        /* reset position to the correct HDU if necessary */
        if fptr.HDUposition != fptr.Fptr.curhdu {
            ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
        }
        fptr.Fptr.hdutype = ASCII_TBL; /* set that this is an ASCII table */
        fptr.Fptr.headend = fptr.Fptr.logfilesize; /* set max size */

        /* get table parameters and test that the header is a valid: */
        if ffgttb(
            fptr,
            &mut rowlen,
            &mut nrows,
            &mut pcount,
            &mut tfield,
            status,
        ) > 0
        {
            return *status;
        }
        if pcount != 0 {
            ffpmsg_str("PCOUNT keyword not equal to 0 in ASCII table (ffainit).");
            int_snprintf!(&mut errmsg, FLEN_ERRMSG, "  PCOUNT = {}", pcount as c_long,);
            ffpmsg_slice(&errmsg);
            *status = BAD_PCOUNT;
            return *status;
        }

        fptr.Fptr.rowlength = rowlen; /* store length of a row */
        fptr.Fptr.tfield = tfield as c_int; /* store number of table fields in row */

        /* free the tile-compressed image cache, if it exists */
        if !fptr.Fptr.tilerow.is_null() {
            // HEAP DEALLOCATION
            let mut tilestruct_lock = TILE_STRUCTS.lock().unwrap();
            let _ = tilestruct_lock.remove_entry(&(&raw const fptr.Fptr as usize));
            drop(tilestruct_lock);

            fptr.Fptr.tileanynull = ptr::null_mut();
            fptr.Fptr.tiletype = ptr::null_mut();
            fptr.Fptr.tiledatasize = ptr::null_mut();
            fptr.Fptr.tilenullarray = ptr::null_mut();
            fptr.Fptr.tiledata = ptr::null_mut();
            fptr.Fptr.tilerow = ptr::null_mut();
        }

        if !fptr.Fptr.tableptr.is_null() {
            /* free memory for the old CHDU */
            // HEAP DEALLOCATION
            let mut alloc_lock = ALLOCATIONS.lock().unwrap();
            let alloc = alloc_lock.remove(&(fptr.Fptr.tableptr as usize));
            if let Some((l, c)) = alloc {
                // HEAP DEALLOCATION
                let _ = Vec::from_raw_parts(fptr.Fptr.tableptr, l, c);
            } else {
                let _ = Vec::from_raw_parts(
                    fptr.Fptr.tableptr,
                    fptr.Fptr.tfield as usize,
                    fptr.Fptr.tfield as usize,
                );
            }
        }
        /* mem for column structures ; space is initialized = 0 */
        if tfield > 0 {
            let mut tmp = Vec::new();
            if tmp.try_reserve_exact(tfield as usize).is_err() {
                ffpmsg_str("malloc failed to get memory for FITS table descriptors (ffainit)");
                fptr.Fptr.tableptr = ptr::null_mut(); /* set a null table structure pointer */
                *status = ARRAY_TOO_BIG;
                return *status;
            } else {
                tmp.resize(tfield as usize, tcolumn::default());
            }

            let (p, l, c) = vec_into_raw_parts(tmp);
            ALLOCATIONS.lock().unwrap().insert(p as usize, (l, c));
            colptr = p;
        }

        /* copy the table structure address to the fitsfile structure */
        fptr.Fptr.tableptr = colptr;

        /*  initialize the table field parameters */
        if tfield > 0 {
            let c = slice::from_raw_parts_mut(colptr, tfield as usize);
            let mut ci = 0;
            let mut ii = 0;
            while ii < tfield {
                c[ci].ttype[0] = 0; /* null column name */
                c[ci].tscale = 1.0;
                c[ci].tzero = 0.0;
                c[ci].strnull[0] = ASCII_NULL_UNDEFINED as c_char; /* null value undefined */
                c[ci].tbcol = -1; /* initialize to illegal value */
                c[ci].tdatatype = -9999; /* initialize to illegal value */
                ii += 1;
                ci += 1;
            }
        }

        /*
        Initialize the fictitious heap starting address (immediately following
        the table data) and a zero length heap.  This is used to find the
        end of the table data when checking the fill values in the last block.
        There is no special data following an ASCII table.
        */
        fptr.Fptr.numrows = nrows;
        fptr.Fptr.origrows = nrows;
        fptr.Fptr.heapstart = rowlen * nrows;
        fptr.Fptr.heapsize = 0;

        fptr.Fptr.compressimg = 0; /* this is not a compressed image */

        /* now search for the table column keywords and the END keyword */
        nspace = 0;

        let mut ii = 8;

        loop {
            /* infinite loop  */

            ffgkyn_safe(
                fptr,
                ii as c_int,
                &mut name,
                &mut value,
                Some(&mut comm),
                status,
            );

            /* try to ignore minor syntax errors */
            if *status == NO_QUOTE {
                strcat_safe(&mut value, cs!(c"'"));
                *status = 0;
            } else if *status == BAD_KEYCHAR {
                *status = 0;
            }

            if *status == END_OF_FILE {
                ffpmsg_str("END keyword not found in ASCII table header (ffainit).");
                *status = NO_END;
                return *status;
            } else if *status > 0 {
                return *status;
            } else if name[0] == bb(b'T') {
                /* keyword starts with 'T' ? */
                ffgtbp(fptr, &name, &value, status); /* test if column keyword */
            } else if FSTRCMP(&name, cs!(c"END")) == 0 {
                /* is this the END keyword? */
                break;
            }
            if name[0] == 0 && value[0] == 0 && comm[0] == 0 {
                /* a blank keyword? */
                nspace += 1;
            } else {
                nspace = 0;
            }
            ii += 1;
        }

        /* test that all required keywords were found and have legal values */
        if tfield > 0 {
            let c = slice::from_raw_parts_mut(colptr, tfield as usize);
            let mut ci = 0;

            let mut ii = 0;
            while ii < tfield {
                tbcoln = c[ci].tbcol; /* the starting column number (zero based) */

                if c[ci].tdatatype == -9999 {
                    ffkeyn_safe(cs!(c"TFORM"), (ii + 1) as c_int, &mut name, status); /* construct keyword name */
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "Required {} keyword not found (ffainit).",
                        slice_to_str!(&name),
                    );
                    ffpmsg_slice(&message);
                    *status = NO_TFORM;
                    return *status;
                } else if tbcoln == -1 {
                    ffkeyn_safe(cs!(c"TBCOL"), (ii + 1) as c_int, &mut name, status); /* construct keyword name */
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "Required {} keyword not found (ffainit).",
                        slice_to_str!(&name),
                    );
                    ffpmsg_slice(&message);
                    *status = NO_TBCOL;
                    return *status;
                } else if fptr.Fptr.rowlength != 0 && (tbcoln < 0 || tbcoln >= fptr.Fptr.rowlength)
                {
                    ffkeyn_safe(cs!(c"TBCOL"), (ii + 1) as c_int, &mut name, status); /* construct keyword name */
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "Value of {} keyword out of range: {} (ffainit).",
                        slice_to_str!(&name),
                        tbcoln as c_long,
                    );
                    ffpmsg_slice(&message);
                    *status = BAD_TBCOL;
                    return *status;
                } else if fptr.Fptr.rowlength != 0
                    && tbcoln + c[ci].twidth as LONGLONG > fptr.Fptr.rowlength
                {
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "Column {} is too wide to fit in table (ffainit)",
                        ii + 1,
                    );
                    ffpmsg_slice(&message);
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        " TFORM = {} and NAXIS1 = {}",
                        slice_to_str!(&c[ci].tform),
                        fptr.Fptr.rowlength as c_long,
                    );
                    ffpmsg_slice(&message);
                    *status = COL_TOO_WIDE;
                    return *status;
                }
                ii += 1;
                ci += 1;
            }
        }

        /*
        now we know everything about the table; just fill in the parameters:
        the 'END' record is 80 bytes before the current position, minus
        any trailing blank keywords just before the END keyword.
        */
        fptr.Fptr.headend = fptr.Fptr.nextkey - (80 * (LONGLONG::from(nspace) + 1));

        /* the data unit begins at the beginning of the next logical block */
        fptr.Fptr.datastart = ((fptr.Fptr.nextkey - 80) / 2880 + 1) * 2880;

        /* the next HDU begins in the next logical block after the data  */
        *fptr.Fptr.headstart.offset(fptr.Fptr.curhdu as isize + 1) =
            fptr.Fptr.datastart + ((rowlen * nrows + 2879) / 2880 * 2880);

        /* reset next keyword pointer to the start of the header */
        fptr.Fptr.nextkey = *fptr.Fptr.headstart.offset(fptr.Fptr.curhdu as isize);

        *status
    }
}

/// initialize the parameters defining the structure of a binary table
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub(crate) fn ffbinit(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    unsafe {
        let mut tfield: c_long = 0;
        let mut pcount: LONGLONG = 0;
        let mut rowlen: LONGLONG = 0;
        let mut nrows: LONGLONG = 0;
        let mut totalwidth: LONGLONG = 0;
        let mut colptr: *mut tcolumn = ptr::null_mut();
        let mut name: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
        let mut value: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
        let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
        let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

        if *status > 0 {
            return *status;
        }
        /* reset position to the correct HDU if necessary */
        if fptr.HDUposition != fptr.Fptr.curhdu {
            ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
        }
        fptr.Fptr.hdutype = BINARY_TBL; /* set that this is a binary table */
        fptr.Fptr.headend = fptr.Fptr.logfilesize; /* set max size */

        /* get table parameters and test that the header is valid: */
        if ffgttb(
            fptr,
            &mut rowlen,
            &mut nrows,
            &mut pcount,
            &mut tfield,
            status,
        ) > 0
        {
            return *status;
        }
        fptr.Fptr.rowlength = rowlen; /* store length of a row */
        fptr.Fptr.tfield = tfield as c_int; /* store number of table fields in row */

        /* free the tile-compressed image cache, if it exists */
        if !fptr.Fptr.tilerow.is_null() {
            // HEAP DEALLOCATION
            let mut tilestruct_lock = TILE_STRUCTS.lock().unwrap();
            let _ = tilestruct_lock.remove_entry(&(&raw const fptr.Fptr as usize));
            drop(tilestruct_lock);

            fptr.Fptr.tileanynull = ptr::null_mut();
            fptr.Fptr.tiletype = ptr::null_mut();
            fptr.Fptr.tiledatasize = ptr::null_mut();
            fptr.Fptr.tilenullarray = ptr::null_mut();
            fptr.Fptr.tiledata = ptr::null_mut();
            fptr.Fptr.tilerow = ptr::null_mut();
        }

        if !fptr.Fptr.tableptr.is_null() {
            /* free memory for the old CHDU */
            // HEAP DEALLOCATION
            let mut alloc_lock = ALLOCATIONS.lock().unwrap();
            let alloc = alloc_lock.remove(&(fptr.Fptr.tableptr as usize));
            if let Some((l, c)) = alloc {
                // HEAP DEALLOCATION
                let _ = Vec::from_raw_parts(fptr.Fptr.tableptr, l, c);
            } else {
                let _ = Vec::from_raw_parts(
                    fptr.Fptr.tableptr,
                    fptr.Fptr.tfield as usize,
                    fptr.Fptr.tfield as usize,
                );
            }
        }

        /* mem for column structures ; space is initialized = 0  */
        if tfield > 0 {
            let mut tmp = Vec::new();
            if tmp.try_reserve_exact(tfield as usize).is_err() {
                ffpmsg_str("malloc failed to get memory for FITS table descriptors (ffbinit)");
                fptr.Fptr.tableptr = ptr::null_mut(); /* set a null table structure pointer */
                *status = ARRAY_TOO_BIG;
                return *status;
            } else {
                tmp.resize(tfield as usize, tcolumn::default());
            }

            let (p, l, c) = vec_into_raw_parts(tmp);
            ALLOCATIONS.lock().unwrap().insert(p as usize, (l, c));
            colptr = p;
        }

        /* copy the table structure address to the fitsfile structure */
        fptr.Fptr.tableptr = colptr;

        /* initialize the table field parameters */
        if tfield > 0 {
            let c = slice::from_raw_parts_mut(colptr, tfield as usize);
            let mut ci = 0;
            let mut ii = 0;
            while ii < tfield {
                c[ci].ttype[0] = 0; /* null column name */
                c[ci].tscale = 1.0;
                c[ci].tzero = 0.0;
                c[ci].tnull = NULL_UNDEFINED.into(); /* (integer) null value undefined */
                c[ci].tdatatype = -9999; /* initialize to illegal value */
                c[ci].trepeat = 1;
                c[ci].strnull[0] = 0; /* for ASCII string columns (TFORM = rA) */
                ii += 1;
                ci += 1;
            }
        }

        /*
        Initialize the heap starting address (immediately following
        the table data) and the size of the heap.  This is used to find the
        end of the table data when checking the fill values in the last block.
        */
        fptr.Fptr.numrows = nrows;
        fptr.Fptr.origrows = nrows;
        fptr.Fptr.heapstart = rowlen * nrows;
        fptr.Fptr.heapsize = pcount;

        fptr.Fptr.compressimg = 0; /* initialize as not a compressed image */

        /* now search for the table column keywords and the END keyword */

        let mut ii = 8;
        let mut nspace = 0;
        loop {
            /* infinite loop  */

            ffgkyn_safe(
                fptr,
                ii as c_int,
                &mut name,
                &mut value,
                Some(&mut comm),
                status,
            );

            /* try to ignore minor syntax errors */
            if *status == NO_QUOTE {
                strcat_safe(&mut value, cs!(c"'"));
                *status = 0;
            } else if *status == BAD_KEYCHAR {
                *status = 0;
            }

            if *status == END_OF_FILE {
                ffpmsg_str("END keyword not found in binary table header (ffbinit).");
                *status = NO_END;
                return *status;
            } else if *status > 0 {
                return *status;
            } else if name[0] == bb(b'T') {
                /* keyword starts with 'T' ? */
                ffgtbp(fptr, &name, &value, status); /* test if column keyword */
            } else if FSTRCMP(&name, cs!(c"ZIMAGE")) == 0 {
                if value[0] == bb(b'T') {
                    fptr.Fptr.compressimg = 1; /* this is a compressed image */
                }
            } else if FSTRCMP(&name, cs!(c"END")) == 0 {
                /* is this the END keyword? */
                break;
            }

            if name[0] == 0 && value[0] == 0 && comm[0] == 0 {
                /* a blank keyword? */
                nspace += 1;
            } else {
                nspace = 0; /* reset number of consecutive spaces before END */
            }
            ii += 1;
        }

        /* test that all the required keywords were found and have legal values */
        if tfield > 0 {
            colptr = fptr.Fptr.tableptr; /* set pointer to first column */
            let c = slice::from_raw_parts_mut(colptr, tfield as usize);

            for ii in 0..(tfield as usize) {
                if c[ii].tdatatype == -9999 {
                    ffkeyn_safe(cs!(c"TFORM"), (ii + 1) as c_int, &mut name, status); /* construct keyword name */
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "Required {} keyword not found (ffbinit).",
                        slice_to_str!(&name),
                    );
                    ffpmsg_slice(&message);
                    *status = NO_TFORM;
                    return *status;
                }
            }
        }

        /*
        now we know everything about the table; just fill in the parameters:
        the 'END' record is 80 bytes before the current position, minus
        any trailing blank keywords just before the END keyword.
        */

        fptr.Fptr.headend = fptr.Fptr.nextkey - (80 * (nspace + 1));

        /* the data unit begins at the beginning of the next logical block */
        fptr.Fptr.datastart = ((fptr.Fptr.nextkey - 80) / 2880 + 1) * 2880;

        /* the next HDU begins in the next logical block after the data  */
        *fptr.Fptr.headstart.offset(fptr.Fptr.curhdu as isize + 1) =
            fptr.Fptr.datastart + ((fptr.Fptr.heapstart + fptr.Fptr.heapsize + 2879) / 2880 * 2880);

        /* determine the byte offset to the beginning of each column */
        ffgtbc(fptr, &mut totalwidth, status);

        if totalwidth != rowlen {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "NAXIS1 = {} is not equal to the sum of column widths: {}",
                rowlen as c_long,
                totalwidth as c_long,
            );
            ffpmsg_slice(&message);
            *status = BAD_ROW_WIDTH;
        }

        /* reset next keyword pointer to the start of the header */
        fptr.Fptr.nextkey = *fptr.Fptr.headstart.offset(fptr.Fptr.curhdu as isize);

        if fptr.Fptr.compressimg == 1 {
            /*  Is this a compressed image */
            imcomp_get_compressed_image_par(fptr, status);
        }
        *status
    }
}

/// calculate the starting byte offset of each column of an ASCII table
/// and the total length of a row, in bytes.  The input space value determines
/// how many blank spaces to leave between each column (1 is recommended).
///
/// # Parameters
///
/// * `tfields` — (I) number of columns in the table
/// * `tform`   — (I) value of TFORMn keyword for each column
/// * `space`   — (I) number of spaces to leave between cols
/// * `rowlen`  — (O) total width of a table row
/// * `tbcol`   — (O) starting byte in row for each column
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgabc(
    tfields: c_int,
    tform: *const *const c_char,
    space: c_int,
    rowlen: *mut c_long,
    tbcol: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let rowlen = rowlen.as_mut().expect(NULL_MSG);

        let tbcol = slice::from_raw_parts_mut(tbcol, tfields as usize);
        let tform = slice::from_raw_parts(tform, tfields as usize);

        let mut v_tform = Vec::new();

        for item in tform {
            let tform_item = slice::from_raw_parts(*item, FLEN_VALUE);
            v_tform.push(tform_item);
        }

        ffgabc_safe(tfields, &v_tform, space, rowlen, tbcol, status)
    }
}

/// calculate the starting byte offset of each column of an ASCII table
/// and the total length of a row, in bytes.  The input space value determines
/// how many blank spaces to leave between each column (1 is recommended).
///
/// # Parameters
///
/// * `tfields` — (I) number of columns in the table
/// * `tform`   — (I) value of TFORMn keyword for each column
/// * `space`   — (I) number of spaces to leave between cols
/// * `rowlen`  — (O) total width of a table row
/// * `tbcol`   — (O) starting byte in row for each column
/// * `status`  — (IO) error status
pub fn ffgabc_safe(
    tfields: c_int,
    tform: &[&[c_char]],
    space: c_int,
    rowlen: &mut c_long,
    tbcol: &mut [c_long],
    status: &mut c_int,
) -> c_int {
    let mut datacode = 0;
    let mut decims = 0;
    let mut width: c_long = 0;

    if *status > 0 {
        return *status;
    }

    *rowlen = 0;

    if tfields <= 0 {
        return *status;
    }

    tbcol[0] = 1;

    for ii in 0..(tfields as usize) {
        tbcol[ii] = *rowlen + 1; /* starting byte in row of column */
        ffasfm_safe(
            tform[ii],
            Some(&mut datacode),
            Some(&mut width),
            Some(&mut decims),
            status,
        );

        *rowlen += width + c_long::from(space); /* total length of row */
    }

    *rowlen -= c_long::from(space); /*  don't add space after the last field */

    *status
}

/// calculate the starting byte offset of each column of a binary table.
/// Use the values of the datatype code and repeat counts in the
/// column structure. Return the total length of a row, in bytes.
///
/// # Parameters
///
/// * `fptr`       — (I) FITS file pointer
/// * `totalwidth` — (O) total width of a table row
/// * `status`     — (IO) error status
pub(crate) fn ffgtbc(fptr: &mut fitsfile, totalwidth: &mut LONGLONG, status: &mut c_int) -> c_int {
    let mut nbytes: LONGLONG;
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }
    let tfields: c_int = fptr.Fptr.tfield;

    *totalwidth = 0;

    if tfields > 0 {
        let c = fptr.Fptr.get_tableptr_as_mut_slice();
        let mut ii = 0;
        let mut ci = 0;
        while ii < tfields {
            c[ci].tbcol = *totalwidth; /* byte offset in row to this column */

            if c[ci].tdatatype == TSTRING {
                nbytes = c[ci].trepeat; /* one byte per char */
            } else if c[ci].tdatatype == TBIT {
                nbytes = (c[ci].trepeat + 7) / 8;
            } else if c[ci].tdatatype > 0 {
                nbytes = c[ci].trepeat * (LONGLONG::from(c[ci].tdatatype) / 10);
            } else {
                let cptr = &c[ci].tform;
                let mut cpi = 0;
                while isdigit_safe(cptr[cpi]) {
                    cpi += 1;
                }

                if cptr[cpi] == bb(b'P') {
                    /* this is a 'P' variable length descriptor (neg. tdatatype) */
                    nbytes = c[ci].trepeat * 8;
                } else if cptr[cpi] == bb(b'Q') {
                    /* this is a 'Q' variable length descriptor (neg. tdatatype) */
                    nbytes = c[ci].trepeat * 16;
                } else {
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "unknown binary table column type: {}",
                        slice_to_str!(&c[ci].tform),
                    );
                    ffpmsg_slice(&message);
                    *status = BAD_TFORM;
                    return *status;
                }
            }

            *totalwidth += nbytes;
            ii += 1;
            ci += 1;
        }
    }
    *status
}

/// Get TaBle Parameter.  The input keyword name begins with the letter T.
/// Test if the keyword is one of the table column definition keywords
/// of an ASCII or binary table. If so, decode it and update the value
/// in the structure.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `name`   — (I) name of the keyword
/// * `value`  — (I) value string of the keyword
/// * `status` — (IO) error status
pub(crate) fn ffgtbp(
    fptr: &mut fitsfile,
    name: &[c_char],
    value: &[c_char],
    status: &mut c_int,
) -> c_int {
    let mut tstatus: c_int;
    let mut datacode: c_int = 0;
    let mut decimals: c_int = 0;
    let mut width: c_long = 0;
    let mut repeat: c_long = 0;
    let mut nfield: c_long = 0;
    let mut ivalue: c_long = 0;
    let mut jjvalue: LONGLONG = 0;
    let mut dvalue: f64 = 0.0;
    let mut tvalue: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    if *status > 0 {
        return *status;
    }

    tstatus = 0;

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }

    let c = if fptr.Fptr.tfield > 0 {
        // Can't do this because of borrow checker
        // fptr.Fptr.get_tableptr_as_mut_slice()
        // So instead do..
        unsafe { core::slice::from_raw_parts_mut(fptr.Fptr.tableptr, fptr.Fptr.tfield as usize) }
    /* get pointer to columns */
    } else {
        &mut []
    };

    let mut ci: usize = 0;

    if FSTRNCMP(&name[1..], cs!(c"TYPE"), 4) == 0 {
        /* get the index number */
        if ffc2ii(&name[5..], &mut nfield, &mut tstatus) > 0 {
            /* read index no. */
            return *status; /* must not be an indexed keyword */
        }
        if nfield < 1 || nfield > c_long::from(fptr.Fptr.tfield) {
            /* out of range */
            return *status;
        }

        ci = ci + nfield as usize - 1; /* point to the correct column */

        if ffc2s(value, &mut tvalue, &mut tstatus) > 0 {
            /* remove quotes */

            return *status;
        }

        strcpy_safe(&mut c[ci].ttype, &tvalue); /* copy col name to structure */
    } else if FSTRNCMP(&name[1..], cs!(c"FORM"), 4) == 0 {
        /* get the index number */
        if ffc2ii(&name[5..], &mut nfield, &mut tstatus) > 0 {
            /* read index no. */
            return *status; /* must not be an indexed keyword */
        }

        if nfield < 1 || nfield > c_long::from(fptr.Fptr.tfield) {
            /* out of range */
            return *status;
        }

        ci = ci + nfield as usize - 1; /* point to the correct column */

        if ffc2s(value, &mut tvalue, &mut tstatus) > 0 {
            /* remove quotes */
            return *status;
        }

        strncpy_safe(&mut c[ci].tform, &tvalue, 9); /* copy TFORM to structure */
        c[ci].tform[9] = 0; /* make sure it is terminated */

        if fptr.Fptr.hdutype == ASCII_TBL {
            /* ASCII table */
            if ffasfm_safe(
                &tvalue,
                Some(&mut datacode),
                Some(&mut width),
                Some(&mut decimals),
                status,
            ) > 0
            {
                return *status; /* bad format code */
            }
            c[ci].tdatatype = TSTRING; /* store datatype code */
            c[ci].trepeat = 1; /* field repeat count == 1 */
            c[ci].twidth = width; /* the width of the field, in bytes */
        } else {
            /* binary table */

            if ffbnfm_safe(
                &tvalue,
                Some(&mut datacode),
                Some(&mut repeat),
                Some(&mut width),
                status,
            ) > 0
            {
                return *status; /* bad format code */
            }
            c[ci].tdatatype = datacode; /* store datatype code */
            c[ci].trepeat = repeat as LONGLONG; /* field repeat count  */

            /* Don't overwrite the unit string width if it was previously */
            /* set by a TDIMn keyword and has a legal value */
            if datacode == TSTRING {
                if c[ci].twidth == 0 || c[ci].twidth > repeat {
                    c[ci].twidth = width; /*  width of a unit string */
                }
            } else {
                c[ci].twidth = width; /*  width of a unit value in chars */
            }
        }
    } else if FSTRNCMP(&name[1..], cs!(c"BCOL"), 4) == 0 {
        /* get the index number */
        if ffc2ii(&name[5..], &mut nfield, &mut tstatus) > 0 {
            /* read index no. */

            return *status; /* must not be an indexed keyword */
        }

        if nfield < 1 || nfield > c_long::from(fptr.Fptr.tfield) {
            /* out of range */
            return *status;
        }

        ci = ci + nfield as usize - 1; /* point to the correct column */

        if fptr.Fptr.hdutype == BINARY_TBL {
            return *status; /* binary tables don't have TBCOL keywords */
        }

        if ffc2ii(value, &mut ivalue, status) > 0 {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "Error reading value of {} as an integer: {}",
                slice_to_str!(&name),
                slice_to_str!(&value),
            );
            ffpmsg_slice(&message);
            return *status;
        }
        c[ci].tbcol = ivalue as LONGLONG - 1; /* convert to zero base */
    } else if FSTRNCMP(&name[1..], cs!(c"SCAL"), 4) == 0 {
        /* get the index number */
        if ffc2ii(&name[5..], &mut nfield, &mut tstatus) > 0 {
            /* read index no. */
            return *status; /* must not be an indexed keyword */
        }

        if nfield < 1 || nfield > c_long::from(fptr.Fptr.tfield) {
            /* out of range */
            return *status;
        }

        ci = ci + nfield as usize - 1; /* point to the correct column */

        if ffc2dd(value, &mut dvalue, &mut tstatus) > 0 {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "Error reading value of {} as a double: {}",
                slice_to_str!(&name),
                slice_to_str!(&value),
            );
            ffpmsg_slice(&message);

            /* ignore this error, so don't return error status */
            return *status;
        }
        c[ci].tscale = dvalue;
    } else if FSTRNCMP(&name[1..], cs!(c"ZERO"), 4) == 0 {
        /* get the index number */
        if ffc2ii(&name[5..], &mut nfield, &mut tstatus) > 0 {
            /* read index no. */
            return *status; /* must not be an indexed keyword */
        }
        if nfield < 1 || nfield > c_long::from(fptr.Fptr.tfield) {
            /* out of range */
            return *status;
        }

        ci = ci + nfield as usize - 1; /* point to the correct column */

        if ffc2dd(value, &mut dvalue, &mut tstatus) > 0 {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "Error reading value of {} as a double: {}",
                slice_to_str!(&name),
                slice_to_str!(&value),
            );
            ffpmsg_slice(&message);

            /* ignore this error, so don't return error status */
            return *status;
        }
        c[ci].tzero = dvalue;
    } else if FSTRNCMP(&name[1..], cs!(c"NULL"), 4) == 0 {
        /* get the index number */
        if ffc2ii(&name[5..], &mut nfield, &mut tstatus) > 0 {
            /* read index no. */
            return *status; /* must not be an indexed keyword */
        }

        if nfield < 1 || nfield > c_long::from(fptr.Fptr.tfield) {
            /* out of range */
            return *status;
        }

        ci = ci + nfield as usize - 1; /* point to the correct column */

        if fptr.Fptr.hdutype == ASCII_TBL {
            /* ASCII table */

            if ffc2s(value, &mut tvalue, &mut tstatus) > 0 {
                /* remove quotes */
                return *status;
            }
            strncpy_safe(&mut c[ci].strnull, &tvalue, 17); /* copy TNULL string */
            c[ci].strnull[17] = 0; /* terminate the strnull field */
        } else {
            /* binary table */

            if ffc2jj(value, &mut jjvalue, &mut tstatus) > 0 {
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Error reading value of {} as an integer: {}",
                    slice_to_str!(&name),
                    slice_to_str!(&value),
                );
                ffpmsg_slice(&message);

                /* ignore this error, so don't return error status */
                return *status;
            }
            c[ci].tnull = jjvalue; /* null value for integer column */
        }
    } else if FSTRNCMP(&name[1..], cs!(c"DIM"), 3) == 0 {
        if fptr.Fptr.hdutype == ASCII_TBL {
            /* ASCII table */
            return *status; /* ASCII tables don't support TDIMn keyword */
        }

        /* get the index number */
        if ffc2ii(&name[4..], &mut nfield, &mut tstatus) > 0 {
            /* read index no. */
            return *status; /* must not be an indexed keyword */
        }

        if nfield < 1 || nfield > c_long::from(fptr.Fptr.tfield) {
            /* out of range */
            return *status;
        }

        ci = ci + nfield as usize - 1; /* point to the correct column */

        /* uninitialized columns have tdatatype set = -9999 */
        if c[ci].tdatatype != -9999 && c[ci].tdatatype != TSTRING {
            return *status; /* this is not an ASCII string column */
        }

        let loc = strchr_safe(value, bb(b'(')); /* find the opening parenthesis */

        if loc.is_none() {
            return *status; /* not a proper TDIM keyword */
        }

        let loc = loc.unwrap() + 1;
        /* read size of first dimension */
        // width = strtol_safe(&value[loc..], &mut endp, 10);
        let (r, _p) = strtol_safe(&value[loc..]).unwrap();
        width = r;

        if c[ci].trepeat != 1 && c[ci].trepeat < width as LONGLONG {
            return *status; /* string length is greater than column width */
        }
        c[ci].twidth = width; /* set width of a unit string in chars */
    } else if FSTRNCMP(&name[1..], cs!(c"HEAP"), 4) == 0 {
        if fptr.Fptr.hdutype == ASCII_TBL {
            /* ASCII table */
            return *status; /* ASCII tables don't have a heap */
        }

        if ffc2jj(value, &mut jjvalue, &mut tstatus) > 0 {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "Error reading value of {} as an integer: {}",
                slice_to_str!(&name),
                slice_to_str!(&value),
            );
            ffpmsg_slice(&message);

            /* ignore this error, so don't return error status */
            return *status;
        }
        fptr.Fptr.heapstart = jjvalue; /* starting byte of the heap */
        return *status;
    }

    *status
}

/// Get Column PaRameters, and test starting row and element numbers for
/// validity.  This is a workhorse routine that is call by nearly every
/// other routine that reads or writes to FITS files.
///
/// # Parameters
///
/// * `fptr`      — (I) FITS file pointer
/// * `colnum`    — (I) column number (1 = 1st column of table)
/// * `firstrow`  — (I) first row (1 = 1st row of table)
/// * `firstelem` — (I) first element within vector (1 = 1st)
/// * `nelem`     — (I) number of elements to read or write
/// * `writemode` — (I) = 1 if writing data, = 0 if reading data
/// * `scale`     — (O) FITS scaling factor (TSCALn keyword value)
/// * `zero`      — (O) FITS scaling zero pt (TZEROn keyword value)
/// * `tform`     — (O) ASCII column format: value of TFORMn keyword
/// * `twidth`    — (O) width of ASCII column (characters)
/// * `tcode`     — (O) abs(column datatype code): I*4=41, R*4=42, etc
/// * `maxelem`   — (O) max number of elements that fit in buffer
/// * `startpos`  — (O) offset in file to starting row & column
/// * `elemnum`   — (O) starting element number ( 0 = 1st element)
/// * `incre`     — (O) byte offset between elements within a row
/// * `repeat`    — (O) number of elements in a row (vector column)
/// * `rowlen`    — (O) length of a row, in bytes
/// * `hdutype`   — (O) HDU type: 0, 1, 2 = primary, table, bintable
/// * `tnull`     — (O) null value for integer columns
/// * `snull`     — (O) null value for ASCII table columns
/// * `status`    — (IO) error status
pub(crate) fn ffgcprll(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    firstelem: LONGLONG,
    nelem: LONGLONG,
    writemode: c_int,
    /*     If = 2, then writing data, but don't modify  */
    /*     the returned values of repeat and incre.     */
    /*     If = -1, then reading data in reverse        */
    /*     direction.                                   */
    /*     If writemode has 16 added, then treat        */
    /*        TSTRING column as TBYTE vector            */
    scale: &mut f64,
    zero: &mut f64,
    tform: &mut [c_char],
    twidth: &mut c_long,
    tcode: &mut c_int,
    maxelem: &mut c_int,
    startpos: &mut LONGLONG,
    elemnum: &mut LONGLONG,
    incre: &mut c_long,
    repeat: &mut LONGLONG,
    rowlen: &mut LONGLONG,
    hdutype: &mut c_int,
    tnull: &mut LONGLONG,
    snull: &mut [c_char],
    status: &mut c_int,
) -> c_int {
    let mut writemode = writemode;

    let nulpos;
    let mut rangecheck = 1;
    let endpos: LONGLONG;
    let nblock: c_long;
    let mut heapoffset: LONGLONG = 0;
    let mut lrepeat: LONGLONG = 0;
    let endrow: LONGLONG;
    let nrows: LONGLONG;
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    if fptr.HDUposition != fptr.Fptr.curhdu {
        /* reset position to the correct HDU if necessary */
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        /* rescan header if data structure is undefined */
        if ffrdef_safe(fptr, status) > 0 {
            return *status;
        };
    } else if writemode > 0 && writemode != 15 {
        /* Only terminate the header with the END card if */
        /* writing to the stdout stream (don't have random access). */

        /* Initialize STREAM_DRIVER to be the device number for */
        /* writing FITS files directly out to the stdout stream. */
        /* This only needs to be done once and is thread safe. */

        let mut st = STREAM_DRIVER.lock().unwrap();
        if *st <= 0 || *st > 40 {
            urltype2driver(cs!(c"stream://"), st.borrow_mut());
        }

        if fptr.Fptr.driver == *st
            && fptr.Fptr.ENDpos != cmp::max(fptr.Fptr.headend, fptr.Fptr.datastart - IOBUFLEN)
        {
            ffwend(fptr, status);
        };
    }

    /* Do sanity check of input parameters */
    if firstrow < 1 {
        if fptr.Fptr.hdutype == IMAGE_HDU {
            /*  Primary Array or IMAGE */
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "Image group number is less than 1: {:.0}",
                firstrow as f64,
            );
            ffpmsg_slice(&message);

            *status = BAD_ROW_NUM;
            return *status;
        } else {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "Starting row number is less than 1: {:.0}",
                firstrow as f64,
            );
            ffpmsg_slice(&message);

            *status = BAD_ROW_NUM;
            return *status;
        };
    } else if fptr.Fptr.hdutype != ASCII_TBL && firstelem < 1 {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Starting element number less than 1: {}",
            firstelem as c_long,
        );
        ffpmsg_slice(&message);

        *status = BAD_ELEM_NUM;
        return *status;
    } else if nelem < 0 {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Tried to read or write less than 0 elements: {:.0}",
            nelem as f64,
        );
        ffpmsg_slice(&message);

        *status = NEG_BYTES;
        return *status;
    } else if colnum < 1 || colnum > fptr.Fptr.tfield {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Specified column number is out of range: {}",
            colnum,
        );
        ffpmsg_slice(&message);
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "  There are {} columns in this table.",
            fptr.Fptr.tfield,
        );
        ffpmsg_slice(&message);

        *status = BAD_COL_NUM;
        return *status;
    }

    /*  copy relevant parameters from the structure */

    *hdutype = fptr.Fptr.hdutype; /* image, ASCII table, or BINTABLE  */
    *rowlen = fptr.Fptr.rowlength; /* width of the table, in bytes     */
    let datastart = fptr.Fptr.datastart; /* offset in file to start of table */

    let ci = colnum as usize - 1; /* offset to correct column structure */
    let tbcol;
    {
        let c: &[tcolumn] = fptr.Fptr.get_tableptr_as_slice(); /* point to first column structure */

        *scale = c[ci].tscale; /* value scaling factor;    default = 1.0 */
        *zero = c[ci].tzero; /* value scaling zeropoint; default = 0.0 */
        *tnull = c[ci].tnull; /* null value for integer columns         */
        tbcol = c[ci].tbcol; /* offset to start of column within row   */
        *twidth = c[ci].twidth; /* width of a single datum, in bytes      */
        *incre = c[ci].twidth; /* increment between datums, in bytes     */

        *tcode = c[ci].tdatatype;
        *repeat = c[ci].trepeat;

        strcpy_safe(tform, &c[ci].tform); /* value of TFORMn keyword            */
        strcpy_safe(snull, &c[ci].strnull); /* null value for ASCII table columns */
    }

    if *hdutype == ASCII_TBL && snull[0] == 0 {
        /* In ASCII tables, a null value is equivalent to all spaces */

        strcpy_safe(snull, cs!(c"                 ")); /* maximum of 17 spaces */
        nulpos = cmp::min(17, *twidth); /* truncate to width of column */
        snull[nulpos as usize] = 0;
    }

    /* Special case: use writemode = 15,16,17,18 to interpret TSTRING columns
       as TBYTE vectors instead (but not for ASCII tables).
          writemode = 15 equivalent to writemode =-1
          writemode = 16 equivalent to writemode = 0
          writemode = 17 equivalent to writemode = 1
          writemode = 18 equivalent to writemode = 2
    */

    if writemode >= 15 && writemode <= 18 {
        if (*tcode).abs() == TSTRING && *hdutype != ASCII_TBL {
            *incre = 1; /* each element is 1 byte wide */

            if *tcode < 0 {
                *repeat = *twidth as LONGLONG; /* variable columns appear to put width in *twidth */
            }

            *twidth = 1; /* width of each element */
            *scale = 1.0; /* no scaling */
            *zero = 0.0;
            *tnull = LONGLONG::from(NULL_UNDEFINED); /* don't test for nulls */
            *maxelem = DBUFFSIZE as c_int;

            if *tcode < 0 {
                *tcode = -TBYTE; /* variable-length */
            } else {
                *tcode = TBYTE;
            };
        }

        /* translate to the equivalent as listed above */
        writemode -= 16;
    }

    /* Special case:  interpret writemode = -1 as reading data, but */
    /* don't do error check for exceeding the range of pixels  */
    if writemode == -1 {
        writemode = 0;
        rangecheck = 0;
    }

    /* Special case: interprete 'X' column as 'B' */
    if (*tcode).abs() == TBIT {
        *tcode = *tcode / TBIT * TBYTE;
        *repeat = (*repeat + 7) / 8;
    }

    /* Special case: support the 'rAw' format in BINTABLEs */
    if *hdutype == BINARY_TBL && *tcode == TSTRING {
        if *twidth != 0 {
            *repeat /= *twidth as LONGLONG; /* repeat = # of unit strings in field */
        } else {
            *repeat = 0;
        };
    } else if *hdutype == BINARY_TBL && *tcode == -TSTRING {
        /* variable length string */
        *incre = 1;
        *twidth = nelem as c_long;
    }

    if *hdutype == ASCII_TBL {
        *elemnum = 0; /* ASCII tables don't have vector elements */
    } else {
        *elemnum = firstelem - 1;
    }

    /* interprete complex and double complex as pairs of floats or doubles */
    if (*tcode).abs() >= TCOMPLEX {
        if *tcode > 0 {
            *tcode = (*tcode + 1) / 2;
        } else {
            *tcode = (*tcode - 1) / 2;
        }
        *repeat *= 2;
        *twidth /= 2;
        *incre /= 2;
    }

    /* calculate no. of pixels that fit in buffer */
    /* allow for case where floats are 8 bytes long */
    if (*tcode).abs() == TFLOAT {
        *maxelem = (DBUFFSIZE as usize / core::mem::size_of::<f32>()) as c_int;
    } else if (*tcode).abs() == TDOUBLE {
        *maxelem = (DBUFFSIZE as usize / core::mem::size_of::<f64>()) as c_int;
    } else if (*tcode).abs() == TSTRING {
        if *twidth != 0 {
            *maxelem = ((DBUFFSIZE - 1) / *twidth as ULONGLONG) as c_int; /* leave room for final \0 */
        } else {
            *maxelem = DBUFFSIZE as c_int - 1;
        }
        if *maxelem == 0 {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "ASCII string column is too wide: {}; max supported width is {}",
                *twidth,
                DBUFFSIZE - 1,
            );
            ffpmsg_slice(&message);
            return {
                *status = COL_TOO_WIDE;
                *status
            };
        };
    } else {
        *maxelem = (DBUFFSIZE / *twidth as ULONGLONG) as c_int;
    }

    /* calc starting byte position to 1st element of col  */
    /*  (this does not apply to variable length columns)  */
    *startpos = datastart + (((firstrow - 1) as LONGLONG) * *rowlen) + tbcol;

    if *hdutype == IMAGE_HDU && writemode != 0 {
        /*  Primary Array or IMAGE */

        /*
          For primary arrays, set the repeat count greater than the total
          number of pixels to be written.  This prevents an out-of-range
          error message in cases where the final image array size is not
          yet known or defined.
        */
        if *repeat < *elemnum + nelem {
            *repeat = *elemnum + nelem;
        };
    } else if *tcode > 0 {
        /*  Fixed length table column  */

        if *elemnum >= *repeat {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "First element to write is too large: {}; max allowed value is {}",
                ((*elemnum) + 1) as c_long,
                *repeat as c_long,
            );
            ffpmsg_slice(&message);

            *status = BAD_ELEM_NUM;
            return *status;
        }

        /* last row number to be read or written */
        endrow = ((*elemnum + nelem - 1) / *repeat) + firstrow;

        if writemode != 0 {
            /* check if we are writing beyond the current end of table */
            if (endrow > fptr.Fptr.numrows) && (nelem > 0) {
                /* if there are more HDUs following the current one, or */
                /* if there is a data heap, then we must insert space */
                /* for the new rows.  */
                if (fptr.Fptr.lasthdu) == 0 || fptr.Fptr.heapsize > 0 {
                    nrows = endrow - (fptr.Fptr.numrows);
                    if ffirow_safe(fptr, fptr.Fptr.numrows, nrows, status) > 0 {
                        int_snprintf!(
                            &mut message,
                            FLEN_ERRMSG,
                            "Failed to add space for {:.0} new rows in table.",
                            nrows as f64,
                        );
                        ffpmsg_slice(&message);
                        return *status;
                    };
                } else {
                    /* update heap starting address */
                    fptr.Fptr.heapstart +=
                        ((endrow - fptr.Fptr.numrows) as LONGLONG) * fptr.Fptr.rowlength;
                    fptr.Fptr.numrows = endrow; /* update number of rows */
                };
            };
        } else {
            /* reading from the file */

            if endrow > fptr.Fptr.numrows && rangecheck != 0 {
                if *hdutype == IMAGE_HDU {
                    /*  Primary Array or IMAGE */

                    if firstrow > fptr.Fptr.numrows {
                        int_snprintf!(
                            &mut message,
                            FLEN_ERRMSG,
                            "Attempted to read from group {} of the HDU,",
                            firstrow as c_long,
                        );
                        ffpmsg_slice(&message);
                        int_snprintf!(
                            &mut message,
                            FLEN_ERRMSG,
                            "however the HDU only contains {} group(s).",
                            fptr.Fptr.numrows as c_long,
                        );
                        ffpmsg_slice(&message);
                    } else {
                        ffpmsg_str("Attempt to read past end of array:");
                        int_snprintf!(
                            &mut message,
                            FLEN_ERRMSG,
                            "  Image has  {} elements;",
                            *repeat as c_long,
                        );
                        ffpmsg_slice(&message);
                        int_snprintf!(
                            &mut message,
                            FLEN_ERRMSG,
                            "  Tried to read {} elements starting at element {}.",
                            nelem as c_long,
                            firstelem as c_long,
                        );
                        ffpmsg_slice(&message);
                    };
                } else {
                    ffpmsg_str("Attempt to read past end of table:");
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "  Table has {:.0} rows with {:.0} elements per row;",
                        (fptr.Fptr.numrows) as f64,
                        *repeat as f64,
                    );
                    ffpmsg_slice(&message);
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "  Tried to read {:.0} elements starting at row {:.0}, element {:.0}.",
                        nelem as f64,
                        firstrow as f64,
                        ((*elemnum) + 1) as f64,
                    );
                    ffpmsg_slice(&message);
                }

                *status = BAD_ROW_NUM;
                return *status;
            };
        }

        if *repeat == 1 && nelem > 1 && writemode != 2 {
            /*
              When accessing a scalar column, fool the calling routine into
              thinking that this is a vector column with very big elements.
              This allows multiple values (up to the maxelem number of elements
              that will fit in the buffer) to be read or written with a single
              routine call, which increases the efficiency.

              If writemode == 2, then the calling program does not want to
              have this efficiency trick applied.
            */
            if *rowlen <= LONG_MAX as LONGLONG {
                *incre = *rowlen as c_long;
                *repeat = nelem;
            };
        };
    } else {
        /*  Variable length Binary Table column */

        *tcode *= -1;

        if writemode != 0 {
            /* return next empty heap address for writing */

            *repeat = nelem + *elemnum; /* total no. of elements in the field */

            /* first, check if we are overwriting an existing row, and */
            /* if so, if the existing space is big enough for the new vector */

            if firstrow <= fptr.Fptr.numrows {
                let mut tstatus = 0;
                ffgdesll_safe(
                    fptr,
                    colnum,
                    firstrow,
                    Some(&mut lrepeat),
                    Some(&mut heapoffset),
                    &mut tstatus,
                );

                let c: &[tcolumn] = fptr.Fptr.get_tableptr_as_slice(); /* point to first column structure */

                if tstatus == 0 {
                    if (c[ci]).tdatatype <= -TCOMPLEX {
                        lrepeat *= 2; /* no. of float or double values */
                    } else if (c[ci]).tdatatype == -TBIT {
                        lrepeat = (lrepeat + 7) / 8; /* convert from bits to bytes */
                    }

                    if lrepeat >= *repeat {
                        /* enough existing space? */

                        *startpos = datastart + heapoffset + fptr.Fptr.heapstart;

                        /*  write the descriptor into the fixed length part of table */
                        if (c[ci]).tdatatype <= -TCOMPLEX {
                            /* divide repeat count by 2 to get no. of complex values */
                            ffpdes_safe(fptr, colnum, firstrow, *repeat / 2, heapoffset, status);
                        } else {
                            ffpdes_safe(fptr, colnum, firstrow, *repeat, heapoffset, status);
                        }
                        return *status;
                    };
                };
            }

            /* Add more rows to the table, if writing beyond the end. */
            /* It is necessary to shift the heap down in this case */
            if firstrow > fptr.Fptr.numrows {
                nrows = firstrow - (fptr.Fptr.numrows);
                if ffirow_safe(fptr, fptr.Fptr.numrows, nrows, status) > 0 {
                    int_snprintf!(
                        &mut message,
                        FLEN_ERRMSG,
                        "Failed to add space for {:.0} new rows in table.",
                        nrows as f64,
                    );
                    ffpmsg_slice(&message);
                    return *status;
                };
            }

            /*  calculate starting position (for writing new data) in the heap */
            *startpos = datastart + fptr.Fptr.heapstart + fptr.Fptr.heapsize;

            /*  write the descriptor into the fixed length part of table */
            {
                let c: &[tcolumn] = fptr.Fptr.get_tableptr_as_slice(); /* point to first column structure */

                if (c[ci]).tdatatype <= -TCOMPLEX {
                    /* divide repeat count by 2 to get no. of complex values */
                    ffpdes_safe(
                        fptr,
                        colnum,
                        firstrow,
                        *repeat / 2,
                        fptr.Fptr.heapsize,
                        status,
                    );
                } else {
                    ffpdes_safe(fptr, colnum, firstrow, *repeat, fptr.Fptr.heapsize, status);
                }
            }

            /* If this is not the last HDU in the file, then check if */
            /* extending the heap would overwrite the following header. */
            /* If so, then have to insert more blocks. */
            if (fptr.Fptr.lasthdu) == 0 {
                endpos = datastart
                    + fptr.Fptr.heapstart
                    + fptr.Fptr.heapsize
                    + (*repeat * (*incre as LONGLONG));

                let headstart = fptr.Fptr.get_headstart_as_slice();

                if endpos > headstart[fptr.Fptr.curhdu as usize + 1] {
                    /* calc the number of blocks that need to be added */
                    nblock = (((endpos - 1 - headstart[fptr.Fptr.curhdu as usize + 1]) / BL!()) + 1)
                        as c_long;

                    if ffiblk(fptr, nblock, 1, status) > 0 {
                        /* insert blocks */

                        int_snprintf!(
                            &mut message,
                            FLEN_ERRMSG,
                            "Failed to extend the size of the variable length heap by {} blocks.",
                            nblock,
                        );
                        ffpmsg_slice(&message);
                        return *status;
                    };
                };
            }

            /* increment the address to the next empty heap position */
            fptr.Fptr.heapsize += *repeat * (*incre as LONGLONG);
        } else {
            /*  get the read start position in the heap */

            if firstrow > fptr.Fptr.numrows {
                ffpmsg_str("Attempt to read past end of table");
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "  Table has {:.0} rows and tried to read row {:.0}.",
                    (fptr.Fptr.numrows) as f64,
                    firstrow as f64,
                );
                ffpmsg_slice(&message);

                *status = BAD_ROW_NUM;
                return *status;
            }

            ffgdesll_safe(
                fptr,
                colnum,
                firstrow,
                Some(&mut lrepeat),
                Some(&mut heapoffset),
                status,
            );
            *repeat = lrepeat;

            {
                let c: &[tcolumn] = fptr.Fptr.get_tableptr_as_slice(); /* point to first column structure */
                if (c[ci]).tdatatype <= -TCOMPLEX {
                    *repeat *= 2; /* no. of float or double values */
                } else if (c[ci]).tdatatype == -TBIT {
                    *repeat = (*repeat + 7) / 8; /* convert from bits to bytes */
                }
            }

            if *elemnum >= *repeat {
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Starting element to read in variable length column is too large: {}",
                    firstelem as c_long,
                );
                ffpmsg_slice(&message);
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "  This row only contains {} elements",
                    *repeat as c_long,
                );
                ffpmsg_slice(&message);

                *status = BAD_ELEM_NUM;
                return *status;
            }
            *startpos = datastart + heapoffset + fptr.Fptr.heapstart;
        };
    }
    *status
}

/// Tests the contents of the binary table variable length array heap.
///
/// Returns the number of bytes that are currently not pointed to by any
/// of the descriptors, and also the number of bytes that are pointed to
/// by more than one descriptor.  It returns valid = FALSE if any of the
/// descriptors point to addresses that are out of the bounds of the
/// heap.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `heapsz`  — (O) current size of the heap
/// * `unused`  — (O) no. of unused bytes in the heap
/// * `overlap` — (O) no. of bytes shared by > 1 descriptors
/// * `valid`   — (O) are all the heap addresses valid?
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fftheap(
    fptr: *mut fitsfile,
    heapsz: *mut LONGLONG,
    unused: *mut LONGLONG,
    overlap: *mut LONGLONG,
    valid: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let heapsz = heapsz.as_mut();
        let unused = unused.as_mut();
        let overlap = overlap.as_mut();
        let valid = valid.as_mut();

        fftheap_safe(fptr, heapsz, unused, overlap, valid, status)
    }
}

/// Tests the contents of the binary table variable length array heap.
///
/// Returns the number of bytes that are currently not pointed to by any
/// of the descriptors, and also the number of bytes that are pointed to
/// by more than one descriptor.  It returns valid = FALSE if any of the
/// descriptors point to addresses that are out of the bounds of the
/// heap.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `heapsz`  — (O) current size of the heap
/// * `unused`  — (O) no. of unused bytes in the heap
/// * `overlap` — (O) no. of bytes shared by > 1 descriptors
/// * `valid`   — (O) are all the heap addresses valid?
/// * `status`  — (IO) error status
pub fn fftheap_safe(
    fptr: &mut fitsfile,
    mut heapsz: Option<&mut LONGLONG>,
    mut unused: Option<&mut LONGLONG>,
    mut overlap: Option<&mut LONGLONG>,
    mut valid: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut typecode = 0;
    let mut pixsize;
    let mut ii: c_long;
    let mut nbytes: c_long;
    let mut repeat: LONGLONG = 0;
    let mut offset: LONGLONG = 0;
    let mut tunused = 0;
    let mut toverlap = 0;
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if ffrdef_safe(fptr, status) > 0 {
        /* rescan header to make sure everything is up to date */
        return *status;
    }

    if let Some(valid) = valid.as_deref_mut() {
        *valid = TRUE as c_int;
    }
    if let Some(heapsz) = heapsz.as_deref_mut() {
        *heapsz = fptr.Fptr.heapsize;
    }
    if let Some(unused) = unused.as_deref_mut() {
        *unused = 0;
    }
    if let Some(overlap) = overlap.as_deref_mut() {
        *overlap = 0;
    }

    /* return if this is not a binary table HDU or if the heap is empty */
    if fptr.Fptr.hdutype != BINARY_TBL || fptr.Fptr.heapsize == 0 {
        return *status;
    }

    if fptr.Fptr.heapsize > LONG_MAX as LONGLONG {
        ffpmsg_str("Heap is too big to test ( > 2**31 bytes). (fftheap)");
        *status = MEMORY_ALLOCATION;
        return *status;
    }

    let theapsz = fptr.Fptr.heapsize as c_long;

    let mut buffer = Vec::new(); /* allocate temp space */
    if buffer.try_reserve_exact(theapsz as usize).is_err() {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Failed to allocate buffer to test the heap",
        );
        ffpmsg_slice(&message);

        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        buffer.resize(theapsz as usize, 0);
    }

    /* loop over all cols */
    let mut jj = 1;
    while jj <= fptr.Fptr.tfield && *status <= 0 {
        ffgtcl_safe(fptr, jj, Some(&mut typecode), None, None, status);
        if typecode > 0 {
            jj += 1;
            continue; /* ignore fixed length columns */
        }

        pixsize = -typecode / 10;

        ii = 1;
        while ii as LONGLONG <= fptr.Fptr.numrows {
            ffgdesll_safe(
                fptr,
                jj,
                ii as LONGLONG,
                Some(&mut repeat),
                Some(&mut offset),
                status,
            );

            if typecode == -TBIT {
                nbytes = ((repeat + 7) as c_long) / 8;
            } else {
                nbytes = (repeat as c_long) * c_long::from(pixsize);
            }
            if offset < 0 || offset + nbytes as LONGLONG > theapsz as LONGLONG {
                if let Some(valid) = valid.as_deref_mut() {
                    *valid = FALSE as c_int; /* address out of bounds */
                }
                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Descriptor in row {}, column {} has invalid heap address",
                    ii,
                    jj,
                );
                ffpmsg_slice(&message);
            } else {
                for kk in 0..(nbytes as usize) {
                    buffer[kk + offset as usize] += 1; /* increment every used byte */
                }
            };
            ii += 1
        }
        jj += 1
    }

    for kk in 0..(theapsz as usize) {
        if buffer[kk] == 0 {
            tunused += 1;
        } else if buffer[kk] > 1 {
            toverlap += 1;
        };
    }

    if let Some(heapsz) = heapsz {
        *heapsz = theapsz as LONGLONG;
    }

    if let Some(unused) = unused {
        *unused = tunused;
    }

    if let Some(overlap) = overlap {
        *overlap = toverlap;
    }

    *status
}

/// compress the binary table heap by reordering the contents heap and
/// recovering any unused space
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffcmph(fptr: *mut fitsfile, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        ffcmph_safe(fptr, status)
    }
}

/// compress the binary table heap by reordering the contents heap and
/// recovering any unused space
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub fn ffcmph_safe(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut typecode = 0;
    let mut pixsize;
    let mut valid = 0;
    let mut ii: LONGLONG;
    let mut buffsize = 10000;
    let mut nbytes: usize;
    let mut unused: LONGLONG = 0;
    let mut overlap: LONGLONG = 0;
    let mut repeat: LONGLONG = 0;
    let mut offset: LONGLONG = 0;
    let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut pcount: LONGLONG = 0;
    let mut endpos: LONGLONG;
    let t2heapsize: LONGLONG;
    let mut nblock;

    if *status > 0 {
        return *status;
    }

    /* get information about the current heap */
    fftheap_safe(
        fptr,
        None,
        Some(&mut unused),
        Some(&mut overlap),
        Some(&mut valid),
        status,
    );

    if valid == 0 {
        /* bad heap pointers */
        *status = BAD_HEAP_PTR;
        return *status;
    }

    /* return if this is not a binary table HDU or if the heap is OK as is */
    if fptr.Fptr.hdutype != BINARY_TBL
        || fptr.Fptr.heapsize == 0
        || (unused == 0 && overlap == 0)
        || *status > 0
    {
        return *status;
    }

    let mut tptr: Option<Box<fitsfile>> = None;

    /* copy the current HDU to a temporary file in memory */
    if ffinit_safe(&mut tptr, cs!(c"mem://tempheapfile"), status) != 0 {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Failed to create temporary file for the heap",
        );
        ffpmsg_slice(&message);
        return *status;
    }

    let mut tptr = tptr.expect(NULL_MSG);

    if ffcopy_safe(fptr, &mut tptr, 0, status) != 0 {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Failed to create copy of the heap",
        );
        ffpmsg_slice(&message);
        ffclos_safe(tptr, status);
        return *status;
    }

    let mut buffer: Vec<c_char> = Vec::new(); /* allocate initial buffer */

    if buffer.try_reserve_exact(buffsize).is_err() {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Failed to allocate buffer to copy the heap",
        );
        ffpmsg_slice(&message);
        ffclos_safe(tptr, status);

        *status = MEMORY_ALLOCATION;
        return *status;
    } else {
        buffer.resize(buffsize, 0)
    }

    let readheapstart = tptr.Fptr.datastart + tptr.Fptr.heapstart;
    let writeheapstart = fptr.Fptr.datastart + fptr.Fptr.heapstart;

    let t1heapsize = fptr.Fptr.heapsize; /* save original heap size */
    fptr.Fptr.heapsize = 0; /* reset heap to zero */

    /* loop over all cols */
    let mut jj = 1;
    while jj <= fptr.Fptr.tfield && *status <= 0 {
        ffgtcl_safe(&mut tptr, jj, Some(&mut typecode), None, None, status);

        if typecode > 0 {
            jj += 1;
            continue; /* ignore fixed length columns */
        }

        pixsize = -typecode / 10;

        /* copy heap data, row by row */
        ii = 1;
        while ii as LONGLONG <= fptr.Fptr.numrows {
            ffgdesll_safe(
                &mut tptr,
                jj,
                ii,
                Some(&mut repeat),
                Some(&mut offset),
                status,
            );
            if typecode == -TBIT {
                nbytes = ((repeat + 7) / 8) as usize;
            } else {
                nbytes = (repeat as usize) * pixsize as usize;
            }

            /* increase size of buffer if necessary to read whole array */
            if (nbytes) > buffsize {
                let additional = nbytes - buffer.capacity();
                if buffer.try_reserve_exact(additional).is_err() {
                    *status = MEMORY_ALLOCATION;
                } else {
                    buffsize = nbytes;
                    buffer.resize(nbytes, 0);
                };
            }

            /* If this is not the last HDU in the file, then check if */
            /* extending the heap would overwrite the following header. */
            /* If so, then have to insert more blocks. */
            if (fptr.Fptr.lasthdu) == 0 {
                endpos = writeheapstart + fptr.Fptr.heapsize + (nbytes as LONGLONG);
                let headstart = fptr.Fptr.get_headstart_as_slice();
                if endpos > headstart[(fptr.Fptr.curhdu + 1) as usize] {
                    /* calc the number of blocks that need to be added */
                    nblock = (((endpos - 1 - headstart[(fptr.Fptr.curhdu + 1) as usize]) / BL!())
                        + 1) as c_long;

                    if ffiblk(fptr, nblock, 1, status) > 0 {
                        /* insert blocks */
                        int_snprintf!(
                            &mut message,
                            FLEN_ERRMSG,
                            "Failed to extend the size of the variable length heap by {} blocks.",
                            nblock,
                        );
                        ffpmsg_slice(&message);
                    };
                };
            }

            /* read arrray of bytes from temporary copy */
            ffmbyt_safe(&mut tptr, readheapstart + offset, REPORT_EOF, status);
            ffgbyt(
                &mut tptr,
                nbytes as LONGLONG,
                cast_slice_mut(&mut buffer),
                status,
            );

            /* write arrray of bytes back to original file */
            ffmbyt_safe(
                fptr,
                writeheapstart + fptr.Fptr.heapsize,
                IGNORE_EOF,
                status,
            );
            ffpbyt(
                fptr,
                nbytes as LONGLONG,
                cast_slice_mut(&mut buffer),
                status,
            );

            /* write descriptor */
            ffpdes_safe(fptr, jj, ii as LONGLONG, repeat, fptr.Fptr.heapsize, status);
            fptr.Fptr.heapsize += nbytes as LONGLONG; /* update heapsize */

            if *status > 0 {
                ffclos_safe(tptr, status);
                return *status;
            };

            ii += 1
        }

        jj += 1
    }

    ffclos_safe(tptr, status);

    /* delete any empty blocks at the end of the HDU */
    let headstart = fptr.Fptr.get_headstart_as_slice();
    nblock = ((headstart[(fptr.Fptr.curhdu + 1) as usize] - (writeheapstart + fptr.Fptr.heapsize))
        / BL!()) as c_long;

    if nblock > 0 {
        t2heapsize = fptr.Fptr.heapsize; /* save new heap size */
        fptr.Fptr.heapsize = t1heapsize; /* restore  original heap size */
        ffdblk(fptr, nblock, status);
        fptr.Fptr.heapsize = t2heapsize; /* reset correct heap size */
    }

    /* update the PCOUNT value (size of heap) */
    ffmaky_safe(fptr, 2, status); /* reset to beginning of header */

    ffgkyjj_safe(fptr, cs!(c"PCOUNT"), &mut pcount, Some(&mut comm), status);
    if fptr.Fptr.heapsize != pcount {
        ffmkyj_safe(
            fptr,
            cs!(c"PCOUNT"),
            fptr.Fptr.heapsize,
            Some(&comm),
            status,
        );
    }

    /* rescan new HDU structure */
    ffrdef_safe(fptr, status);

    *status
}

/// get (read) the variable length vector descriptor from the table.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `rownum`   — (I) row number (1 = 1st row of table)
/// * `length`   — (O) number of elements in the row
/// * `heapaddr` — (O) heap pointer to the data
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgdes(
    fptr: *mut fitsfile,
    colnum: c_int,
    rownum: LONGLONG,
    length: *mut c_long,
    heapaddr: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let length = length.as_mut();
        let heapaddr = heapaddr.as_mut();

        ffgdes_safe(fptr, colnum, rownum, length, heapaddr, status)
    }
}

/// get (read) the variable length vector descriptor from the table.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `rownum`   — (I) row number (1 = 1st row of table)
/// * `length`   — (O) number of elements in the row
/// * `heapaddr` — (O) heap pointer to the data
/// * `status`   — (IO) error status
pub fn ffgdes_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    rownum: LONGLONG,
    length: Option<&mut c_long>,
    heapaddr: Option<&mut c_long>,
    status: &mut c_int,
) -> c_int {
    let mut lengthjj: LONGLONG = 0; /* convert the temporary 8-byte values to 4-byte values */
    let mut heapaddrjj: LONGLONG = 0; /* check for overflow */

    if ffgdesll_safe(
        fptr,
        colnum,
        rownum,
        Some(&mut lengthjj),
        Some(&mut heapaddrjj),
        status,
    ) > 0
    {
        return *status;
    }

    /* convert the temporary 8-byte values to 4-byte values */
    /* check for overflow */
    if let Some(length) = length {
        if lengthjj > LONG_MAX as LONGLONG {
            *status = NUM_OVERFLOW;
        } else {
            *length = lengthjj as c_long;
        }
    }

    if let Some(heapaddr) = heapaddr {
        if heapaddrjj > LONG_MAX as LONGLONG {
            *status = NUM_OVERFLOW;
        } else {
            *heapaddr = heapaddrjj as c_long;
        }
    }
    *status
}

/// get (read) the variable length vector descriptor from the binary table.
///
/// This is similar to ffgdes, except it supports the full 8-byte range of the
/// length and offset values in 'Q' columns, as well as 'P' columns.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `rownum`   — (I) row number (1 = 1st row of table)
/// * `length`   — (O) number of elements in the row
/// * `heapaddr` — (O) heap pointer to the data
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgdesll(
    fptr: *mut fitsfile,
    colnum: c_int,
    rownum: LONGLONG,
    length: *mut LONGLONG,
    heapaddr: *mut LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let length = length.as_mut();
        let heapaddr = heapaddr.as_mut();

        ffgdesll_safe(fptr, colnum, rownum, length, heapaddr, status)
    }
}

/// get (read) the variable length vector descriptor from the binary table.
///
/// This is similar to ffgdes, except it supports the full 8-byte range of the
/// length and offset values in 'Q' columns, as well as 'P' columns.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `rownum`   — (I) row number (1 = 1st row of table)
/// * `length`   — (O) number of elements in the row
/// * `heapaddr` — (O) heap pointer to the data
/// * `status`   — (IO) error status
pub fn ffgdesll_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    rownum: LONGLONG,
    length: Option<&mut LONGLONG>,
    heapaddr: Option<&mut LONGLONG>,
    status: &mut c_int,
) -> c_int {
    let mut descript4: [c_uint; 2] = [0, 0];
    let mut descript8: [LONGLONG; 2] = [0, 0];

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_slice(); /* point to first column structure */
    let ci = colnum as usize - 1; /* offset to the correct column */

    if c[ci].tdatatype >= 0 {
        *status = NOT_VARI_LEN;
        return *status;
    }

    let bytepos = fptr.Fptr.datastart + (fptr.Fptr.rowlength * (rownum - 1)) + c[ci].tbcol;
    if c[ci].tform[0] == bb(b'P') || c[ci].tform[1] == bb(b'P') {
        /* read 4-byte descriptor */
        if ffgi4b(fptr, bytepos, 2, 4, cast_slice_mut(&mut descript4), status) <= 0 {
            if let Some(length) = length {
                *length = LONGLONG::from(descript4[0]); /* 1st word is the length  */
            }
            if let Some(heapaddr) = heapaddr {
                *heapaddr = LONGLONG::from(descript4[1]); /* 2nd word is the address */
            };
        };
    } else {
        /* this is for 'Q' columns */

        /* read 8 byte descriptor */
        if ffgi8b(fptr, bytepos, 2, 8, cast_slice_mut(&mut descript8), status) <= 0 {
            if let Some(length) = length {
                *length = descript8[0]; /* 1st word is the length  */
            }
            if let Some(heapaddr) = heapaddr {
                *heapaddr = descript8[1]; /* 2nd word is the address */
            };
        };
    }
    *status
}

///  get (read) a range of variable length vector descriptors from the table.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `firstrow` — (I) first row (1 = 1st row of table)
/// * `nrows`    — (I) number or rows to read
/// * `length`   — (O) number of elements in the row
/// * `heapaddr` — (O) heap pointer to the data
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgdess(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    nrows: LONGLONG,
    length: *mut c_long,
    heapaddr: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        let length = match length.is_null() {
            true => Some(slice::from_raw_parts_mut(length, nrows as usize)),
            false => None,
        };

        let heapaddr = match heapaddr.is_null() {
            true => Some(slice::from_raw_parts_mut(heapaddr, nrows as usize)),
            false => None,
        };

        ffgdess_safe(fptr, colnum, firstrow, nrows, length, heapaddr, status)
    }
}

///  get (read) a range of variable length vector descriptors from the table.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `firstrow` — (I) first row (1 = 1st row of table)
/// * `nrows`    — (I) number or rows to read
/// * `length`   — (O) number of elements in the row
/// * `heapaddr` — (O) heap pointer to the data
/// * `status`   — (IO) error status
pub fn ffgdess_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    nrows: LONGLONG,
    mut length: Option<&mut [c_long]>,
    mut heapaddr: Option<&mut [c_long]>,
    status: &mut c_int,
) -> c_int {
    let mut ii: c_long;
    let mut descript4: [INT32BIT; 2] = [0, 0];
    let mut descript8: [LONGLONG; 2] = [0, 0];

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    /* point to first column structure */
    let c = fptr.Fptr.get_tableptr_as_slice();
    let ci = colnum as usize - 1; /* offset to the correct column */

    if c[ci].tdatatype >= 0 {
        *status = NOT_VARI_LEN;
        return *status;
    }

    let rowsize = fptr.Fptr.rowlength;
    let mut bytepos = fptr.Fptr.datastart + (rowsize * (firstrow - 1i64)) + c[ci].tbcol;

    if c[ci].tform[0] == bb(b'P') || c[ci].tform[1] == bb(b'P') {
        /* read 4-byte descriptors */
        ii = 0;
        let mut length_i = 0;
        let mut heapaddr_i = 0;
        while (ii as LONGLONG) < nrows {
            /* read descriptors */
            if ffgi4b(fptr, bytepos, 2, 4, &mut descript4, status) <= 0 {
                if let Some(length) = length.as_deref_mut() {
                    length[length_i] = c_long::from(descript4[0]); /* 1st word is the length  */
                    length_i += 1;
                }

                if let Some(heapaddr) = heapaddr.as_deref_mut() {
                    heapaddr[heapaddr_i] = c_long::from(descript4[1]); /* 2nd word is the address */
                    heapaddr_i += 1;
                }

                bytepos += rowsize;
            } else {
                return *status;
            };
            ii += 1
        }
    } else {
        /* this is for 'Q' columns */

        /* read 8-byte descriptors */
        ii = 0;
        let mut length_i = 0;
        let mut heapaddr_i = 0;
        while (ii as LONGLONG) < nrows {
            /* read descriptors */
            if ffgi8b(fptr, bytepos, 2, 8, cast_slice_mut(&mut descript8), status) <= 0 {
                if let Some(length) = length.as_deref_mut() {
                    #[allow(clippy::absurd_extreme_comparisons)]
                    // On some architectures, this is a valid comparison
                    if descript8[0] > LONG_MAX as LONGLONG {
                        *status = NUM_OVERFLOW;
                    }

                    length[length_i] = descript8[0] as c_long; /* 1st word is the length  */
                    length_i += 1;
                }
                if let Some(heapaddr) = heapaddr.as_deref_mut() {
                    #[allow(clippy::absurd_extreme_comparisons)]
                    // On some architectures, this is a valid comparison
                    if descript8[1] > LONG_MAX as LONGLONG {
                        *status = NUM_OVERFLOW;
                    }

                    heapaddr[heapaddr_i] = descript8[1] as c_long; /* 2nd word is the address */
                    heapaddr_i += 1;
                }
                bytepos += rowsize;
            } else {
                return *status;
            };
            ii += 1
        }
    }
    *status
}

/// get (read) a range of variable length vector descriptors from the table.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `firstrow` — (I) first row (1 = 1st row of table)
/// * `nrows`    — (I) number or rows to read
/// * `length`   — (O) number of elements in the row
/// * `heapaddr` — (O) heap pointer to the data
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgdessll(
    fptr: *mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    nrows: LONGLONG,
    length: *mut LONGLONG,
    heapaddr: *mut LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        /* the C contract is that length/heapaddr, when non-NULL, each point
        at nrows elements */
        let length = (!length.is_null()).then(|| slice::from_raw_parts_mut(length, nrows as usize));
        let heapaddr =
            (!heapaddr.is_null()).then(|| slice::from_raw_parts_mut(heapaddr, nrows as usize));

        ffgdessll_safe(
            &mut *fptr,
            colnum,
            firstrow,
            nrows,
            length,
            heapaddr,
            &mut *status,
        )
    }
}

/// Get the descriptors -- element count and heap offset -- for a range of rows
/// of a variable length array column, as 64-bit values.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number, the first column is 1
/// * `firstrow` — (I) first row, the first row is 1
/// * `nrows`    — (I) number of rows to read
/// * `length`   — (O) number of elements in each row's array
/// * `heapaddr` — (O) heap offset of each row's array
/// * `status`   — (IO) error status
pub fn ffgdessll_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    firstrow: LONGLONG,
    nrows: LONGLONG,
    mut length: Option<&mut [LONGLONG]>,
    mut heapaddr: Option<&mut [LONGLONG]>,
    status: &mut c_int,
) -> c_int {
    let mut ii: LONGLONG;
    let mut descript4: [INT32BIT; 2] = [0, 0];
    let mut descript8: [LONGLONG; 2] = [0, 0];

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    /* point to first column structure */
    let c = fptr.Fptr.get_tableptr_as_mut_slice();
    let ci = colnum as usize - 1; /* offset to the correct column */

    if c[ci].tdatatype >= 0 {
        *status = NOT_VARI_LEN;
        return *status;
    }

    let tbcol = c[ci].tbcol;

    let rowsize = fptr.Fptr.rowlength;
    let mut bytepos = fptr.Fptr.datastart + (rowsize * (firstrow - 1)) + tbcol;

    let c = fptr.Fptr.get_tableptr_as_mut_slice();
    let ci = colnum as usize - 1; /* offset to the correct column */

    if c[ci].tform[0] == bb(b'P') || c[ci].tform[1] == bb(b'P') {
        /* read 4-byte descriptors */
        ii = 0;
        let mut length_i = 0;
        let mut heapaddr_i = 0;
        while ii < nrows {
            /* read descriptors */
            if ffgi4b(fptr, bytepos, 2, 4, &mut descript4, status) <= 0 {
                if let Some(length) = length.as_deref_mut() {
                    length[length_i] = LONGLONG::from(descript4[0]); /* 1st word is the length  */
                    length_i += 1;
                }
                if let Some(heapaddr) = heapaddr.as_deref_mut() {
                    heapaddr[heapaddr_i] = LONGLONG::from(descript4[1]); /* 2nd word is the address */
                    heapaddr_i += 1;
                }
                bytepos += rowsize;
            } else {
                return *status;
            };
            ii += 1
        }
    } else {
        /* this is for 'Q' columns */

        /* read 8-byte descriptors */
        ii = 0;
        let mut length_i = 0;
        let mut heapaddr_i = 0;
        while ii < nrows {
            /* read descriptors */
            /* cast to type (long *) even though it is actually (LONGLONG *) */
            if ffgi8b(fptr, bytepos, 2, 8, cast_slice_mut(&mut descript8), status) <= 0 {
                if let Some(length) = length.as_deref_mut() {
                    length[length_i] = descript8[0] as LONGLONG; /* 1st word is the length  */
                    length_i += 1;
                }
                if let Some(heapaddr) = heapaddr.as_deref_mut() {
                    heapaddr[heapaddr_i] = descript8[1] as LONGLONG; /* 2nd word is the address */
                    heapaddr_i += 1;
                }
                bytepos += rowsize;
            } else {
                return *status;
            };
            ii += 1
        }
    }
    *status
}

///  put (write) the variable length vector descriptor to the table.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `rownum`   — (I) row number (1 = 1st row of table)
/// * `length`   — (I) number of elements in the row
/// * `heapaddr` — (I) heap pointer to the data
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffpdes(
    fptr: *mut fitsfile,
    colnum: c_int,
    rownum: LONGLONG,
    length: LONGLONG,
    heapaddr: LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        ffpdes_safe(fptr, colnum, rownum, length, heapaddr, status)
    }
}

///  put (write) the variable length vector descriptor to the table.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `colnum`   — (I) column number (1 = 1st column of table)
/// * `rownum`   — (I) row number (1 = 1st row of table)
/// * `length`   — (I) number of elements in the row
/// * `heapaddr` — (I) heap pointer to the data
/// * `status`   — (IO) error status
pub fn ffpdes_safe(
    fptr: &mut fitsfile,
    colnum: c_int,
    rownum: LONGLONG,
    length: LONGLONG,
    heapaddr: LONGLONG,
    status: &mut c_int,
) -> c_int {
    let mut descript4: [c_uint; 2] = [0; 2];
    let mut descript8: [LONGLONG; 2] = [0; 2];

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    let c = fptr.Fptr.get_tableptr_as_slice(); /* point to first column structure */
    let ci = (colnum - 1) as usize; /* offset to the correct column */

    if c[ci].tdatatype >= 0 {
        *status = NOT_VARI_LEN;
    }

    let bytepos: LONGLONG =
        fptr.Fptr.datastart + (fptr.Fptr.rowlength * (rownum - 1)) + c[ci].tbcol;

    ffmbyt_safe(fptr, bytepos, IGNORE_EOF, status); /* move to element */

    let c = fptr.Fptr.get_tableptr_as_slice(); /* point to first column structure */
    let ci = (colnum - 1) as usize; /* offset to the correct column */

    if c[ci].tform[0] == bb(b'P') || c[ci].tform[1] == bb(b'P') {
        if length > LONGLONG::from(c_uint::MAX)
            || length < 0
            || heapaddr > LONGLONG::from(c_uint::MAX)
            || heapaddr < 0
        {
            ffpmsg_str("P variable length column descriptor is out of range");
            *status = NUM_OVERFLOW;
            return *status;
        }

        descript4[0] = length as c_uint; /* 1st word is the length  */
        descript4[1] = heapaddr as c_uint; /* 2nd word is the address */

        ffpi4b(fptr, 2, 4, cast_slice(&descript4), status); /* write the descriptor */
    } else {
        /* this is a 'Q' descriptor column */

        descript8[0] = length; /* 1st word is the length  */
        descript8[1] = heapaddr; /* 2nd word is the address */

        ffpi8b(fptr, 2, 8, cast_slice(&descript8), status); /* write the descriptor */
    }

    *status
}

/// close the current HDU.  If we have write access to the file, then:
/// - write the END keyword and pad header with blanks if necessary
/// - check the data fill values, and rewrite them if not correct
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub(crate) fn ffchdu(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut stdriver = -1; // -1 so that if stream driver not found, it doesn't default

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
        /* no need to do any further updating of the HDU */
    } else if fptr.Fptr.writemode == 1 {
        urltype2driver(cs!(c"stream://"), &mut stdriver);

        /* don't rescan header in special case of writing to stdout */
        if fptr.Fptr.driver != stdriver {
            ffrdef_safe(fptr, status);
        }
        if fptr.Fptr.heapsize > 0 {
            ffuptf(fptr, status); /* update the variable length TFORM values */
        }

        ffpdfl(fptr, status); /* insure correct data fill values */
    }

    if fptr.Fptr.open_count == 1 {
        /* free memory for the CHDU structure only if no other files are using it */
        if !fptr.Fptr.tableptr.is_null() {
            // HEAP DEALLOCATION
            let mut alloc_lock = ALLOCATIONS.lock().unwrap();
            let alloc = alloc_lock.remove(&(fptr.Fptr.tableptr as usize));
            if let Some((l, c)) = alloc {
                // HEAP DEALLOCATION
                let _ = unsafe { Vec::from_raw_parts(fptr.Fptr.tableptr, l, c) };
            } else {
                let _ = unsafe {
                    Vec::from_raw_parts(
                        fptr.Fptr.tableptr,
                        fptr.Fptr.tfield as usize,
                        fptr.Fptr.tfield as usize,
                    )
                };
            }

            fptr.Fptr.tableptr = ptr::null_mut();

            /* free the tile-compressed image cache, if it exists */
            if !fptr.Fptr.tilerow.is_null() {
                // HEAP DEALLOCATION
                let mut tilestruct_lock = TILE_STRUCTS.lock().unwrap();
                let _ = tilestruct_lock.remove_entry(&(&raw const fptr.Fptr as usize));
                drop(tilestruct_lock);

                fptr.Fptr.tileanynull = ptr::null_mut();
                fptr.Fptr.tiletype = ptr::null_mut();
                fptr.Fptr.tiledatasize = ptr::null_mut();
                fptr.Fptr.tilenullarray = ptr::null_mut();
                fptr.Fptr.tiledata = ptr::null_mut();
                fptr.Fptr.tilerow = ptr::null_mut();
            };
        };
    }

    if *status > 0 && *status != NO_CLOSE_ERROR {
        int_snprintf!(
            &mut message,
            FLEN_ERRMSG,
            "Error while closing HDU number {} (ffchdu).",
            fptr.Fptr.curhdu,
        );
        ffpmsg_slice(&message);
    }
    *status
}

/// Update the value of the TFORM keywords for the variable length array
/// columns to make sure they all have the form 1Px(len) or Px(len) where
/// 'len' is the maximum length of the vector in the table (e.g., '1PE(400)')
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub(crate) fn ffuptf(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut tflds: c_long = 0;
    let mut length: LONGLONG = 0;
    let mut addr: LONGLONG = 0;
    let mut maxlen: LONGLONG;
    let mut naxis2: LONGLONG = 0;
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut keyname: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut tform: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut newform: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut lenval: [c_char; 40] = [0; 40];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    ffmaky_safe(fptr, 2, status); /* reset to beginning of header */
    ffgkyjj_safe(
        fptr,
        cs!(c"NAXIS2"),
        &mut naxis2,
        Some(&mut comment),
        status,
    );
    ffgkyj_safe(
        fptr,
        cs!(c"TFIELDS"),
        &mut tflds,
        Some(&mut comment),
        status,
    );

    for ii in 1..=(tflds as usize) {
        /* loop over all the columns */

        ffkeyn_safe(cs!(c"TFORM"), ii as c_int, &mut keyname, status); /* construct name */
        if ffgkys_safe(fptr, &keyname, &mut tform, Some(&mut comment), status) > 0 {
            int_snprintf!(
                &mut message,
                FLEN_ERRMSG,
                "Error while updating variable length vector TFORMn values (ffuptf).",
            );
            ffpmsg_slice(&message);
            return *status;
        }

        /* is this a variable array length column ? */
        if tform[0] == bb(b'P')
            || tform[1] == bb(b'P')
            || tform[0] == bb(b'Q')
            || tform[1] == bb(b'Q')
        {
            /* get the max length */
            maxlen = 0;
            for jj in 1..=(naxis2 as usize) {
                ffgdesll_safe(
                    fptr,
                    ii as c_int,
                    jj as LONGLONG,
                    Some(&mut length),
                    Some(&mut addr),
                    status,
                );

                if length > maxlen {
                    maxlen = length;
                }
            }

            /* construct the new keyword value */
            strcpy_safe(&mut newform, cs!(c"'"));
            let tmp = strchr_safe(&tform, bb(b'(')); /* truncate old length, if present */
            if let Some(tmp) = tmp {
                tform[tmp] = 0;
            }
            let lenform = strlen_safe(&tform);

            /* print as double, because the string-to-64-bit */
            /* conversion is platform dependent (%lld, %ld, %I64d) */

            int_snprintf!(&mut lenval, 40, "({:.0})", maxlen as f64,);

            if lenform + strlen_safe(&lenval) + 2 > FLEN_VALUE - 1 {
                ffpmsg_str("Error assembling TFORMn string (ffuptf).");
                *status = BAD_TFORM;
                return *status;
            }
            strcat_safe(&mut newform, &tform);

            strcat_safe(&mut newform, &lenval);

            while strlen_safe(&newform) < 9 {
                strcat_safe(&mut newform, cs!(c" ")); /* append spaces 'till length = 8 */
            }
            strcat_safe(&mut newform, cs!(c"'")); /* append closing parenthesis */
            /* would be simpler to just call ffmkyj here, but this */
            /* would force linking in all the modkey & putkey routines */
            ffmkky_safe(&keyname, &newform, Some(&comment), &mut card, status); /* make new card */
            ffmkey(fptr, &card, status); /* replace last read keyword */
        }
    }
    *status
}

/// ReDEFine the structure of a data unit.  This routine re-reads
/// the CHDU header keywords to determine the structure and length of the
/// current data unit.  This redefines the start of the next HDU.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffrdef(fptr: *mut fitsfile, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        ffrdef_safe(fptr, status)
    }
}

/// ReDEFine the structure of a data unit.  This routine re-reads
/// the CHDU header keywords to determine the structure and length of the
/// current data unit.  This redefines the start of the next HDU.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub fn ffrdef_safe(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut dummy: c_int = 0;
    let mut tstatus = 0;
    let mut naxis2: LONGLONG = 0;
    let mut pcount: LONGLONG = 0;
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut comm: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut valstring: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.writemode == 1 {
        /* write access to the file? */

        /* don't need to check NAXIS2 and PCOUNT if data hasn't been written */
        if fptr.Fptr.datastart != DATA_UNDEFINED as LONGLONG {
            /* update NAXIS2 keyword if more rows were written to the table */
            /* and if the user has not explicitly reset the NAXIS2 value */
            if fptr.Fptr.hdutype != IMAGE_HDU {
                ffmaky_safe(fptr, 2, status);
                if ffgkyjj_safe(
                    fptr,
                    cs!(c"NAXIS2"),
                    &mut naxis2,
                    Some(&mut comm),
                    &mut tstatus,
                ) > 0
                {
                    /* Couldn't read NAXIS2 (odd!);  in certain circumstances */
                    /* this may be normal, so ignore the error. */
                    naxis2 = fptr.Fptr.numrows;
                }
                if fptr.Fptr.numrows > naxis2 && fptr.Fptr.origrows == naxis2 {
                    /* if origrows is not equal to naxis2, then the user must */
                    /* have manually modified the NAXIS2 keyword value, and */
                    /* we will assume that the current value is correct. */
                    /* would be simpler to just call ffmkyj here, but this */
                    /* would force linking in all the modkey & putkey routines */

                    /* print as double because the 64-bit int conversion */
                    /* is platform dependent (%lld, %ld, %I64 )          */

                    int_snprintf!(
                        &mut valstring,
                        FLEN_VALUE,
                        "{:.0}",
                        (fptr.Fptr.numrows) as f64,
                    );
                    ffmkky_safe(cs!(c"NAXIS2"), &valstring, Some(&comm), &mut card, status);
                    ffmkey(fptr, &card, status);
                };
            }

            /* if data has been written to variable length columns in a  */
            /* binary table, then we may need to update the PCOUNT value */
            if fptr.Fptr.heapsize > 0 {
                ffmaky_safe(fptr, 2, status);
                ffgkyjj_safe(fptr, cs!(c"PCOUNT"), &mut pcount, Some(&mut comm), status);
                if fptr.Fptr.heapsize != pcount {
                    ffmkyj_safe(
                        fptr,
                        cs!(c"PCOUNT"),
                        fptr.Fptr.heapsize,
                        Some(&comm),
                        status,
                    );
                };
            };
        }

        if ffwend(fptr, status) <= 0 {
            /* rewrite END keyword and fill */
            ffrhdu_safe(fptr, Some(&mut dummy), status); /* re-scan the header keywords  */
        };
    }
    *status
}

/// based on the number of keywords which have already been written,
/// plus the number of keywords to reserve space for, we then can
/// define where the data unit should start (it must start at the
/// beginning of a 2880-byte logical block).
///
/// This routine will only have any effect if the starting location of the
/// data unit following the header is not already defined.  In any case,
/// it is always possible to add more keywords to the header even if the
/// data has already been written.  It is just more efficient to reserve
/// the space in advance.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `morekeys` — (I) reserve space for this many keywords
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffhdef(fptr: *mut fitsfile, morekeys: c_int, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        ffhdef_safe(fptr, morekeys, status)
    }
}

/// based on the number of keywords which have already been written,
/// plus the number of keywords to reserve space for, we then can
/// define where the data unit should start (it must start at the
/// beginning of a 2880-byte logical block).
///
/// This routine will only have any effect if the starting location of the
/// data unit following the header is not already defined.  In any case,
/// it is always possible to add more keywords to the header even if the
/// data has already been written.  It is just more efficient to reserve
/// the space in advance.
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `morekeys` — (I) reserve space for this many keywords
/// * `status`   — (IO) error status
pub fn ffhdef_safe(fptr: &mut fitsfile, morekeys: c_int, status: &mut c_int) -> c_int {
    let delta: LONGLONG;

    if *status > 0 || morekeys < 1 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        ffrdef_safe(fptr, status);

        /* ffrdef defines the offset to datastart and the start of */
        /* the next HDU based on the number of existing keywords. */
        /* We need to increment both of these values based on */
        /* the number of new keywords to be added.  */

        delta = ((fptr.Fptr.headend + (LONGLONG::from(morekeys) * 80)) / BL!() + 1) * BL!()
            - fptr.Fptr.datastart;

        fptr.Fptr.datastart += delta;

        let curhdu = fptr.Fptr.curhdu as usize;
        fptr.Fptr.get_headstart_as_mut_slice()[curhdu + 1] += delta;
    }
    *status
}

/// write the END card and following fill (space chars) in the current header
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub(crate) fn ffwend(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut blankkey: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut endkey: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut keyrec: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    if *status > 0 {
        return *status;
    }

    let mut endpos = fptr.Fptr.headend;

    /* we assume that the HDUposition == curhdu in all cases */

    /*  calc the data starting position if not currently defined */
    if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        fptr.Fptr.datastart = (endpos / BL!() + 1) * BL!();
    }

    /* calculate the number of blank keyword slots in the header */
    let nspace = ((fptr.Fptr.datastart - endpos) / 80i64) as c_long;
    strcpy_safe(
        &mut blankkey,
        cs!(c"                                        "),
    );
    strcat_safe(
        &mut blankkey,
        cs!(c"                                        "),
    );
    strcpy_safe(
        &mut endkey,
        cs!(c"END                                     "),
    );
    strcat_safe(
        &mut endkey,
        cs!(c"                                        "),
    );

    /* check if header is already correctly terminated with END and fill */
    let mut tstatus = 0;
    ffmbyt_safe(fptr, endpos, REPORT_EOF, &mut tstatus); /* move to header end */
    let mut ii = 0;
    while ii < nspace as usize {
        ffgbyt(fptr, 80, cast_slice_mut(&mut keyrec), &mut tstatus); /* get next keyword */
        if tstatus != 0 {
            break;
        }
        if strncmp_safe(&keyrec, &blankkey, 80) != 0 && strncmp_safe(&keyrec, &endkey, 80) != 0 {
            break;
        };
        ii += 1;
    }

    if ii == nspace as usize && tstatus == 0 {
        /* check if the END keyword exists at the correct position */
        endpos = cmp::max(endpos, fptr.Fptr.datastart - IOBUFLEN);
        ffmbyt_safe(fptr, endpos, REPORT_EOF, &mut tstatus); /* move to END position */
        ffgbyt(fptr, 80, cast_slice_mut(&mut keyrec), &mut tstatus); /* read the END keyword */
        if strncmp_safe(&keyrec, &endkey, 80) == 0 && tstatus == 0 {
            /* store this position, for later reference */
            fptr.Fptr.ENDpos = endpos;

            return *status; /* END card was already correct */
        };
    }

    /* header was not correctly terminated, so write the END and blank fill */
    endpos = fptr.Fptr.headend;
    ffmbyt_safe(fptr, endpos, IGNORE_EOF, status); /* move to header end */
    let mut ii = 0;
    while ii < nspace as usize {
        ffpbyt(fptr, 80, cast_slice_mut(&mut blankkey), status); /* write the blank keywords */
        ii += 1
    }
    /*
    The END keyword must either be placed immediately after the last
    keyword that was written (as indicated by the headend value), or
    must be in the first 80 bytes of the 2880-byte FITS record immediately
    preceeding the data unit, whichever is further in the file. The
    latter will occur if space has been reserved for more header keywords
    which have not yet been written.
    */
    endpos = cmp::max(endpos, fptr.Fptr.datastart - 2880);

    ffmbyt_safe(fptr, endpos, REPORT_EOF, status); /* move to END position */
    ffpbyt(fptr, 80, cast_slice_mut(&mut endkey), status); /*  write the END keyword to header */

    /* store this position, for later reference */
    fptr.Fptr.ENDpos = endpos;
    if *status > 0 {
        ffpmsg_str("Error while writing END card (ffwend).");
    }
    *status
}

/// Write the Data Unit Fill values if they are not already correct.
/// The fill values are used to fill out the last 2880 byte block of the HDU.
/// Fill the data unit with zeros or blanks depending on the type of HDU
/// from the end of the data to the end of the current FITS 2880 byte block
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub(crate) fn ffpdfl(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let chfill: u8;
    let mut fill: [u8; IOBUFLEN as usize] = [0; IOBUFLEN as usize];
    let mut fillstart: LONGLONG;
    let mut nfill;
    let mut tstatus;

    if *status > 0 {
        return *status;
    }

    if fptr.HDUposition != (fptr.Fptr.curhdu) {
        return *status; /* fill has already been correctly written */
    }

    if fptr.Fptr.heapstart == 0 {
        return *status; /* null data unit, so there is no fill */
    }

    fillstart = fptr.Fptr.datastart + (fptr.Fptr.heapstart + (fptr.Fptr.heapsize));

    nfill = ((fillstart + (BL!() - 1)) / BL!() * BL!() - fillstart) as c_long;

    if nfill >= BL!()
    /* can only happen if fillstart was negative */
    {
        *status = BAD_HEAP_PTR;
        return *status;
    }

    if fptr.Fptr.hdutype == ASCII_TBL {
        chfill = 32; /* ASCII tables are filled with spaces */
    } else {
        chfill = 0; /* all other extensions are filled with zeros */
    }

    tstatus = 0;

    if nfill == 0 {
        /* no fill bytes; just check that entire table exists */

        fillstart -= 1;
        nfill = 1;
        ffmbyt_safe(fptr, fillstart, REPORT_EOF, &mut tstatus); /* move to last byte */
        ffgbyt(fptr, nfill as LONGLONG, &mut fill, &mut tstatus); /* get the last byte */

        if tstatus == 0 {
            return *status; /* no EOF error, so everything is OK */
        }
    } else {
        ffmbyt_safe(fptr, fillstart, REPORT_EOF, &mut tstatus); /* move to fill area */
        ffgbyt(fptr, nfill as LONGLONG, &mut fill, &mut tstatus); /* get the fill bytes */

        if tstatus == 0 {
            let mut ii: usize = 0;
            while ii < nfill as usize {
                if fill[ii] != chfill {
                    break;
                }
                ii += 1;
            }

            if ii == nfill as usize {
                return *status; /* all the fill values were correct */
            }
        }
    }

    /* fill values are incorrect or have not been written, so write them */
    fill[..(nfill as usize)].fill(chfill); /* fill the buffer with the fill value */

    ffmbyt_safe(fptr, fillstart, IGNORE_EOF, status); /* move to fill area */
    ffpbyt(fptr, nfill as LONGLONG, &fill, status); /* write the fill bytes */

    if *status > 0 {
        ffpmsg_str("Error writing Data Unit fill bytes (ffpdfl).");
    }

    *status
}

/// Check the header fill values.
///
/// Checks that the header unit is correctly filled with blanks from the END
/// card to the end of the current FITS 2880-byte block.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (O) error status
///
/// Translated from `ftchfl` into C by Peter Wilson, Oct. 1997.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffchfl(fptr: *mut fitsfile, status: *mut c_int) -> c_int {
    if fptr.is_null() || status.is_null() {
        return NULL_INPUT_PTR;
    }
    // FFI WRAPPER
    unsafe { ffchfl_safe(&mut *fptr, &mut *status) }
}

/// Check the header fill values.
///
/// Checks that the header unit is correctly filled with blanks from the END
/// card to the end of the current FITS 2880-byte block.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (O) error status
pub fn ffchfl_safe(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut rec: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let blanks = [bb(b' '); 80]; /*  80 spaces  */

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }

    /*   calculate the number of blank keyword slots in the header  */

    let endpos = fptr.Fptr.headend;
    let nblank = ((fptr.Fptr.datastart - endpos) / 80) as c_long;

    /*   move the i/o pointer to the end of the header keywords   */

    ffmbyt_safe(fptr, endpos, TRUE as c_int, status);

    /*   find the END card (there may be blank keywords perceeding it)   */

    let mut gotend = false;
    for _i in 0..nblank {
        ffgbyt(fptr, 80, cast_slice_mut(&mut rec), status);
        if strncmp_safe(&rec, cs!(c"END     "), 8) == 0 {
            if gotend {
                /*   There is a duplicate END record   */
                *status = BAD_HEADER_FILL;
                ffpmsg_str("Warning: Header fill area contains duplicate END card:");
            }
            gotend = true;
            if strncmp_safe(&rec[8..], &blanks[8..], 72) != 0 {
                /*   END keyword has extra characters   */
                *status = END_JUNK;
                ffpmsg_str("Warning: END keyword contains extraneous non-blank characters:");
            };
        } else if gotend && strncmp_safe(&rec, &blanks, 80) != 0 {
            /*   The fill area contains extraneous characters   */
            *status = BAD_HEADER_FILL;
            ffpmsg_str("Warning: Header fill area contains extraneous non-blank characters:");
        };
        if *status > 0 {
            rec[FLEN_CARD - 1] = 0; /* make sure string is null terminated */
            ffpmsg_slice(&rec);
            return *status;
        };
    }
    *status
}

/// Check the data unit fill values.
///
/// Checks that the data unit is correctly filled with zeros or blanks from the
/// end of the data to the end of the current FITS 2880-byte block.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (O) error status
///
/// Translated from `ftcdfl` into C by Peter Wilson, Oct. 1997.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffcdfl(fptr: *mut fitsfile, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        ffcdfl_safe(&mut *fptr, &mut *status)
    }
}

/// Check the data unit fill values.
///
/// Checks that the data unit is correctly filled with zeros or blanks from the
/// end of the data to the end of the current FITS 2880-byte block.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (O) error status
pub fn ffcdfl_safe(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut chbuff: [c_char; BL!()] = [0; BL!()];

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }

    /*   check if the data unit is null   */
    if fptr.Fptr.heapstart == 0 {
        return *status;
    }

    /* calculate starting position of the fill bytes, if any */
    let filpos = fptr.Fptr.datastart + fptr.Fptr.heapstart + fptr.Fptr.heapsize;

    /*   calculate the number of fill bytes   */
    let nfill = ((filpos + (BL!() - 1)) / BL!() * BL!() - filpos) as c_long;
    if nfill == 0 {
        return *status;
    }

    /*   move to the beginning of the fill bytes   */
    ffmbyt_safe(fptr, filpos, FALSE as c_int, status);

    if ffgbyt(fptr, nfill as LONGLONG, cast_slice_mut(&mut chbuff), status) > 0 {
        ffpmsg_str("Error reading data unit fill bytes (ffcdfl).");
        return *status;
    }

    let chfill: c_char = if fptr.Fptr.hdutype == ASCII_TBL {
        32 /* ASCII tables are filled with spaces */
    } else {
        0 /* all other extensions are filled with zeros */
    };

    /*   check for all zeros or blanks   */

    for i in 0..(nfill as usize) {
        if chbuff[i] != chfill {
            *status = BAD_DATA_FILL;

            if fptr.Fptr.hdutype == ASCII_TBL {
                ffpmsg_str(
                    "Warning: remaining bytes following ASCII table data are not filled with blanks.",
                );
            } else {
                ffpmsg_str("Warning: remaining bytes following data are not filled with zeros.");
            }
            return *status;
        };
    }
    *status
}

/// Grow the headstart array by 1000 entries.
///
/// `headstart` is a raw pointer owned by `FITSfile`, so growing it means
/// briefly reconstituting the `Vec` that owns the allocation.  Ownership is
/// handed back to `fptr` on both paths: if the reservation fails the `Vec` must
/// not be dropped here, otherwise `fptr.Fptr.headstart` would be left dangling
/// and freed a second time by `FITSfile::drop`.
///
/// The array is always kept fully initialized with `MAXHDU + 1` elements
/// (length == capacity), which is the invariant `get_headstart_as_slice` and
/// the `ALLOCATIONS` bookkeeping rely on.
fn grow_headstart(fptr: &mut fitsfile) -> Result<(), c_int> {
    let old_ptr = fptr.Fptr.headstart;
    let l = fptr.Fptr.MAXHDU as usize + 1;

    // SAFETY: headstart was registered in ALLOCATIONS as a Vec<LONGLONG> of
    // exactly `l` elements, with length == capacity, and is not aliased here.
    let mut vo = unsafe { Vec::from_raw_parts(old_ptr, l, l) };

    let res = vo.try_reserve_exact(1000);
    if res.is_ok() {
        // The array is later read as a slice of MAXHDU + 1 elements, so the
        // new entries must be initialized, not merely reserved.
        vo.resize(l + 1000, 0);
    }

    // Hand the allocation back to `fptr` before reporting anything: on the
    // error path the pointer is unchanged, on the success path it may have
    // moved.  Either way `fptr` must own it again.
    let (p, len, cap) = vec_into_raw_parts(vo);
    let mut allocations = ALLOCATIONS.lock().unwrap();
    allocations.remove(&(old_ptr as usize));
    allocations.insert(p as usize, (len, cap));
    drop(allocations);
    fptr.Fptr.headstart = p;

    if res.is_err() {
        return Err(MEMORY_ALLOCATION);
    }

    fptr.Fptr.MAXHDU += 1000;
    Ok(())
}

/// Create Header Data unit:  Create, initialize, and move the i/o pointer
/// to a new extension appended to the end of the FITS file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffcrhd(fptr: *mut fitsfile, status: *mut c_int) -> c_int {
    // FFI WRAPPER
    unsafe {
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);

        ffcrhd_safe(fptr, status)
    }
}

/// Create Header Data unit:  Create, initialize, and move the i/o pointer
/// to a new extension appended to the end of the FITS file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
pub fn ffcrhd_safe(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    let mut tstatus: c_int = 0;
    let bytepos: LONGLONG;

    if *status > 0 {
        return *status;
    }

    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    }

    /* If the current header is empty, we don't have to do anything */
    let h = fptr.Fptr.get_headstart_as_slice();
    if fptr.Fptr.headend == h[fptr.Fptr.curhdu as usize] {
        return *status;
    }

    while ffmrhd_safe(fptr, 1, None, &mut tstatus) == 0 {} /* move to end of file */

    if fptr.Fptr.maxhdu == fptr.Fptr.MAXHDU {
        /* allocate more space for the headstart array */
        // HEAP ALLOCATION
        if let Err(e) = grow_headstart(fptr) {
            *status = e;
            return *status;
        }
    }

    if ffchdu(fptr, status) <= 0 {
        /* close the current HDU */

        let h = fptr.Fptr.get_headstart_as_slice();
        bytepos = h[fptr.Fptr.maxhdu as usize + 1]; /* last */
        ffmbyt_safe(fptr, bytepos, IGNORE_EOF, status); /* move file ptr to it */
        fptr.Fptr.maxhdu += 1; /* increment the known number of HDUs */
        fptr.Fptr.curhdu = fptr.Fptr.maxhdu; /* set current HDU loc */
        fptr.HDUposition = fptr.Fptr.maxhdu; /* set current HDU loc */
        fptr.Fptr.nextkey = bytepos; /* next keyword = start of header */
        fptr.Fptr.headend = bytepos; /* end of header */
        fptr.Fptr.datastart = DATA_UNDEFINED as LONGLONG; /* start data unit undefined */

        /* any other needed resets */

        /* reset the dithering offset that may have been calculated for the */
        /* previous HDU back to the requested default value */
        fptr.Fptr.dither_seed = fptr.Fptr.request_dither_seed;
    }

    *status
}

/// Delete the specified number of 2880-byte blocks from the end
/// of the CHDU by shifting all following extensions up this
/// number of blocks.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `nblocks` — (I) number of 2880-byte blocks to delete
/// * `status`  — (IO) error status
pub(crate) fn ffdblk(fptr: &mut fitsfile, nblocks: c_long, status: &mut c_int) -> c_int {
    let mut buffer: [c_char; BL!()] = [0; BL!()];

    if *status > 0 || nblocks <= 0 {
        return *status;
    }

    let nblocks = nblocks as usize;

    let mut tstatus = 0;
    /* pointers to the read and write positions */

    let mut readpos = (fptr.Fptr.datastart + fptr.Fptr.heapstart + fptr.Fptr.heapsize) as usize;
    readpos = readpos.div_ceil(BL!()) * BL!(); /* start of block */

    /*  the following formula is wrong because the current data unit
        may have been extended without updating the headstart value
        of the following HDU.

        readpos = fptr.Fptr.headstart[(fptr.Fptr.curhdu) + 1];
    */

    let mut writepos = readpos - (nblocks * BL!());
    while ffmbyt_safe(fptr, readpos as LONGLONG, REPORT_EOF, &mut tstatus) == 0
        && ffgbyt(fptr, BL!(), cast_slice_mut(&mut buffer), &mut tstatus) == 0
    {
        ffmbyt_safe(fptr, writepos as LONGLONG, REPORT_EOF, status);
        ffpbyt(fptr, BL!(), cast_slice(&buffer), status);
        if *status > 0 {
            ffpmsg_str("Error deleting FITS blocks (ffdblk)");
            return *status;
        }
        readpos += BL!(); /* increment to next block to transfer */
        writepos += BL!();
    }

    /* now fill the last nblock blocks with zeros */
    buffer.fill(0);
    ffmbyt_safe(fptr, writepos as LONGLONG, REPORT_EOF, status);

    for _ii in 0..nblocks {
        ffpbyt(fptr, BL!(), cast_slice(&buffer), status);
    }

    /* move back before the deleted blocks, since they may be deleted
    and we do not want to delete the current active buffer */
    ffmbyt_safe(fptr, (writepos - 1) as LONGLONG, REPORT_EOF, status);

    /* truncate the file to the new size, if supported on this device */
    fftrun(fptr, writepos as LONGLONG, status);

    /* recalculate the starting location of all subsequent HDUs */
    let mut ii = fptr.Fptr.curhdu as usize;
    let maxhdu = fptr.Fptr.maxhdu;
    let headstart = fptr.Fptr.get_headstart_as_mut_slice();
    while ii <= maxhdu as usize {
        headstart[ii + 1] -= (nblocks * BL!()) as LONGLONG;
        ii += 1
    }
    *status
}

/// Return the type of the CHDU. This returns the 'logical' type of the HDU,
/// not necessarily the physical type, so in the case of a compressed image
/// stored in a binary table, this will return the type as an Image, not a
/// binary table.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `exttype` — (O) type of extension, 0, 1, or 2
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffghdt(
    fptr: *mut fitsfile,
    exttype: *mut c_int,
    /*  for IMAGE_HDU, ASCII_TBL, or BINARY_TBL */
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let exttype = exttype.as_mut().expect(NULL_MSG);

        ffghdt_safe(fptr, exttype, status)
    }
}

/// Return the type of the CHDU. This returns the 'logical' type of the HDU,
/// not necessarily the physical type, so in the case of a compressed image
/// stored in a binary table, this will return the type as an Image, not a
/// binary table.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `exttype` — (O) type of extension, 0, 1, or 2
/// * `status`  — (IO) error status
pub fn ffghdt_safe(
    fptr: &mut fitsfile,
    exttype: &mut c_int,
    /*  for IMAGE_HDU, ASCII_TBL, or BINARY_TBL */
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    if fptr.HDUposition == 0 && fptr.Fptr.headend == 0 {
        /* empty primary array is alway an IMAGE_HDU */
        *exttype = IMAGE_HDU;
    } else {
        /* reset position to the correct HDU if necessary */
        if fptr.HDUposition != fptr.Fptr.curhdu {
            ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
        } else if fptr.Fptr.datastart == DATA_UNDEFINED {
            /* rescan header if data structure is undefined */
            if ffrdef_safe(fptr, status) > 0 {
                return *status;
            };
        }

        *exttype = fptr.Fptr.hdutype; /* return the type of HDU */

        /*  check if this is a compressed image */
        if fptr.Fptr.compressimg != 0 {
            *exttype = IMAGE_HDU;
        };
    }
    *status
}

/// Was CFITSIO compiled with the -D_REENTRANT flag?  1 = yes, 0 = no.
///
/// Note that specifying the -D_REENTRANT flag is required, but may not be
/// sufficient, to ensure that CFITSIO can be safely used in a multi-threaded
/// environoment.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_is_reentrant() -> c_int {
    // FFI WRAPPER
    fits_is_reentrant_safe()
}

/// Whether this build of the library is reentrant.
///
/// # Returns
///
/// Always 1: unlike the C, which compiles the thread-safe paths only under
/// `-D_REENTRANT`, this port is always built that way.
pub fn fits_is_reentrant_safe() -> c_int {
    /*
    #ifdef _REENTRANT
           return(1);
    #else
           return(0);
    #endif
    */

    1
}

/// Returns TRUE if the CHDU is a compressed image, else returns zero.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn fits_is_compressed_image(
    fptr: *mut fitsfile,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        fits_is_compressed_image_safe(fptr, status)
    }
}

/// Returns TRUE if the CHDU is a compressed image, else returns zero.
pub fn fits_is_compressed_image_safe(fptr: &mut fitsfile, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status); /*  check if this is a compressed image */
    } else if fptr.Fptr.datastart == DATA_UNDEFINED && ffrdef_safe(fptr, status) > 0 {
        return *status;
    };

    if fptr.Fptr.compressimg != 0 {
        return 1;
    }
    0
}

/// get the datatype and size of the input image
///
/// # Parameters
///
/// * `infptr`  — (I) FITS file pointer
/// * `maxaxis` — (I) max number of axes to return
/// * `bitpix`  — (O) image data type
/// * `naxis`   — (O) image dimension (NAXIS value)
/// * `naxes`   — (O) size of image dimensions
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgipr(
    infptr: *mut fitsfile,
    maxaxis: c_int,
    bitpix: *mut c_int,
    naxis: *mut c_int,
    naxes: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let bitpix = bitpix.as_mut();
        let naxis = naxis.as_mut();
        let naxes = if naxes.is_null() {
            None
        } else {
            Some(slice::from_raw_parts_mut(naxes, maxaxis as usize))
        };

        ffgipr_safe(infptr, maxaxis, bitpix, naxis, naxes, status)
    }
}

/// get the datatype and size of the input image
///
/// # Parameters
///
/// * `infptr`  — (I) FITS file pointer
/// * `maxaxis` — (I) max number of axes to return
/// * `bitpix`  — (O) image data type
/// * `naxis`   — (O) image dimension (NAXIS value)
/// * `naxes`   — (O) size of image dimensions
/// * `status`  — (IO) error status
pub fn ffgipr_safe(
    infptr: &mut fitsfile,
    maxaxis: c_int,
    bitpix: Option<&mut c_int>,
    naxis: Option<&mut c_int>,
    naxes: Option<&mut [c_long]>,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* don't return the parameter if a null pointer was given */

    if let Some(bitpix) = bitpix {
        ffgidt_safe(infptr, bitpix, status); /* get BITPIX value */
    }

    if let Some(naxis) = naxis {
        ffgidm_safe(infptr, naxis, status); /* get NAXIS value */
    }

    if let Some(naxes) = naxes {
        ffgisz_safe(infptr, maxaxis, naxes, status); /* get NAXISn values */
    }

    *status
}

/// Get the datatype and size of the input image
///
/// # Parameters
///
/// * `infptr`  — (I) FITS file pointer
/// * `maxaxis` — (I) max number of axes to return
/// * `bitpix`  — (O) image data type
/// * `naxis`   — (O) image dimension (NAXIS value)
/// * `naxes`   — (O) size of image dimensions
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgiprll(
    infptr: *mut fitsfile,
    maxaxis: c_int,
    bitpix: *mut c_int,
    naxis: *mut c_int,
    naxes: *mut LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let infptr = infptr.as_mut().expect(NULL_MSG);
        let status = status.as_mut().expect(NULL_MSG);
        let bitpix = bitpix.as_mut();
        let naxis = naxis.as_mut();
        let naxes = if naxes.is_null() {
            None
        } else {
            Some(slice::from_raw_parts_mut(naxes, maxaxis as usize))
        };

        ffgiprll_safe(infptr, maxaxis, bitpix, naxis, naxes, status)
    }
}

/// # Parameters
///
/// * `infptr`  — (I) FITS file pointer
/// * `maxaxis` — (I) max number of axes to return
/// * `bitpix`  — (O) image data type
/// * `naxis`   — (O) image dimension (NAXIS value)
/// * `naxes`   — (O) size of image dimensions
/// * `status`  — (IO) error status
pub fn ffgiprll_safe(
    infptr: &mut fitsfile,
    maxaxis: c_int,
    bitpix: Option<&mut c_int>,
    naxis: Option<&mut c_int>,
    naxes: Option<&mut [LONGLONG]>,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status; /* get NAXISn values */
    }

    /* don't return the parameter if a null pointer was given */

    if let Some(bitpix) = bitpix {
        ffgidt_safe(infptr, bitpix, status); /* get BITPIX value */
    }

    if let Some(naxis) = naxis {
        ffgidm_safe(infptr, naxis, status); /* get NAXIS value */
    }

    if let Some(naxes) = naxes {
        ffgiszll_safe(infptr, maxaxis, naxes, status); /* get NAXISn values */
    }

    *status
}

/// Get the datatype of the image (= BITPIX keyword for normal image, or
/// ZBITPIX for a compressed image)
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `imgtype` — (O) image data type
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgidt(
    fptr: *mut fitsfile,
    imgtype: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let imgtype = imgtype.as_mut().expect(NULL_MSG);

        ffgidt_safe(fptr, imgtype, status)
    }
}

/// Get the datatype of the image (= BITPIX keyword for normal image, or
/// ZBITPIX for a compressed image)
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `imgtype` — (O) image data type
/// * `status`  — (IO) error status
pub fn ffgidt_safe(fptr: &mut fitsfile, imgtype: &mut c_int, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    /* reset to beginning of header */
    ffmaky_safe(fptr, 1, status); /* simply move to beginning of header */

    if fptr.Fptr.hdutype == IMAGE_HDU {
        ffgky_safe(
            fptr,
            crate::KeywordDatatypeMut::TINT(imgtype),
            cs!(c"BITPIX"),
            None,
            status,
        );
    } else if fptr.Fptr.compressimg != 0 {
        /* this is a binary table containing a compressed image */
        ffgky_safe(
            fptr,
            crate::KeywordDatatypeMut::TINT(imgtype),
            cs!(c"ZBITPIX"),
            None,
            status,
        );
    } else {
        *status = NOT_IMAGE;
    }
    *status
}

/// Get the effective datatype of the image (= BITPIX keyword for normal image,
/// or ZBITPIX for a compressed image)
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `imgtype` — (O) image data type
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgiet(
    fptr: *mut fitsfile,
    imgtype: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let imgtype = imgtype.as_mut().expect(NULL_MSG);

        ffgiet_safe(fptr, imgtype, status)
    }
}

/// Get the effective datatype of the image (= BITPIX keyword for normal image,
/// or ZBITPIX for a compressed image)
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `imgtype` — (O) image data type
/// * `status`  — (IO) error status
pub fn ffgiet_safe(fptr: &mut fitsfile, imgtype: &mut c_int, status: &mut c_int) -> c_int {
    let mut tstatus: c_int;

    let mut lngzero: c_long = 0;
    let mut bscale: f64 = 0.0;
    let mut bzero: f64 = 0.0;
    let mut min_val: f64;
    let mut max_val: f64;

    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    /* reset to beginning of header */
    ffmaky_safe(fptr, 2, status); /* simply move to beginning of header */

    if fptr.Fptr.hdutype == IMAGE_HDU {
        ffgky_safe(
            fptr,
            crate::KeywordDatatypeMut::TINT(imgtype),
            cs!(c"BITPIX"),
            None,
            status,
        );
    } else if fptr.Fptr.compressimg != 0 {
        /* this is a binary table containing a compressed image */
        ffgky_safe(
            fptr,
            crate::KeywordDatatypeMut::TINT(imgtype),
            cs!(c"ZBITPIX"),
            None,
            status,
        );
    } else {
        *status = NOT_IMAGE;
        return *status;
    }

    /* check if the BSCALE and BZERO keywords are defined, which might
    change the effective datatype of the image  */
    tstatus = 0;
    fits_read_key_dbl(fptr, cs!(c"BSCALE"), &mut bscale, None, &mut tstatus);
    if tstatus != 0 {
        bscale = 1.0;
    }

    tstatus = 0;
    fits_read_key_dbl(fptr, cs!(c"BZERO"), &mut bzero, None, &mut tstatus);

    if tstatus != 0 {
        bzero = 0.0;
    }

    if bscale == 1.0 && bzero == 0.0 {
        /* no scaling */
        return *status;
    }

    match *imgtype {
        BYTE_IMG => {
            /* 8-bit image */
            min_val = 0.;
            max_val = 255.0;
        }
        SHORT_IMG => {
            min_val = -32768.0;
            max_val = 32767.0;
        }
        LONG_IMG => {
            min_val = -2147483648.0;
            max_val = 2147483647.0;
        }
        _ => {
            /* don't have to deal with other data types */
            return *status;
        }
    }

    if bscale >= 0. {
        min_val = bzero + bscale * min_val;
        max_val = bzero + bscale * max_val;
    } else {
        max_val = bzero + bscale * min_val;
        min_val = bzero + bscale * max_val;
    }

    if bzero < 2147483648. {
        /* don't exceed range of 32-bit integer */
        lngzero = bzero as c_long;
    }

    let lngscale: c_long = bscale as c_long;

    if (bzero != 2147483648.) && /* special value that exceeds integer range */
       ((lngzero as f64) != bzero || (lngscale as f64) != bscale)
    {
        /* not integers? */
        /* floating point scaled values; just decide on required precision */
        if *imgtype == BYTE_IMG || *imgtype == SHORT_IMG {
            *imgtype = FLOAT_IMG;
        } else {
            *imgtype = DOUBLE_IMG;
        }

    /*
       In all the remaining cases, BSCALE and BZERO are integers,
       and not equal to 1 and 0, respectively.
    */
    } else if (min_val == -128.) && (max_val == 127.) {
        *imgtype = SBYTE_IMG;
    } else if (min_val >= -32768.0) && (max_val <= 32767.0) {
        *imgtype = SHORT_IMG;
    } else if (min_val >= 0.0) && (max_val <= 65535.0) {
        *imgtype = USHORT_IMG;
    } else if (min_val >= -2147483648.0) && (max_val <= 2147483647.0) {
        *imgtype = LONG_IMG;
    } else if (min_val >= 0.0) && (max_val < 4294967296.0) {
        *imgtype = ULONG_IMG;
    } else {
        /* exceeds the range of a 32-bit integer */
        *imgtype = DOUBLE_IMG;
    }

    *status
}

/// Get the dimension of the image (= NAXIS keyword for normal image, or
/// ZNAXIS for a compressed image)
/// These values are cached for faster access.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `naxis`  — (O) image dimension (NAXIS value)
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgidm(
    fptr: *mut fitsfile,
    naxis: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let naxis = naxis.as_mut().expect(NULL_MSG);

        ffgidm_safe(fptr, naxis, status)
    }
}

/// Get the dimension of the image (= NAXIS keyword for normal image, or
/// ZNAXIS for a compressed image)
/// These values are cached for faster access.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `naxis`  — (O) image dimension (NAXIS value)
/// * `status` — (IO) error status
pub fn ffgidm_safe(fptr: &mut fitsfile, naxis: &mut c_int, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG && ffrdef_safe(fptr, status) > 0 {
        /* rescan header */
        return *status;
    }

    if fptr.Fptr.hdutype == IMAGE_HDU {
        *naxis = fptr.Fptr.imgdim;
    } else if fptr.Fptr.compressimg != 0 {
        *naxis = fptr.Fptr.zndim;
    } else {
        *status = NOT_IMAGE;
    }
    *status
}

/// Get the size of the image dimensions (= NAXISn keywords for normal image, or
/// ZNAXISn for a compressed image)
/// These values are cached for faster access.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nlen`   — (I) number of axes to return
/// * `naxes`  — (O) size of image dimensions
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgisz(
    fptr: *mut fitsfile,
    nlen: c_int,
    naxes: *mut c_long,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        if *status > 0 {
            return *status;
        }

        /* reset position to the correct HDU if necessary */
        if fptr.HDUposition != fptr.Fptr.curhdu {
            ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
        } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
            /* rescan header */
            if ffrdef_safe(fptr, status) > 0 {
                return *status;
            }
        }

        if fptr.Fptr.hdutype == IMAGE_HDU {
            let naxis = cmp::min(fptr.Fptr.imgdim, nlen);
            let naxes = slice::from_raw_parts_mut(naxes, naxis as usize);

            ffgisz_safe(fptr, nlen, naxes, status);
        } else if fptr.Fptr.compressimg != 0 {
            let naxis = cmp::min(fptr.Fptr.zndim, nlen);
            let naxes = slice::from_raw_parts_mut(naxes, naxis as usize);

            ffgisz_safe(fptr, nlen, naxes, status);
        } else {
            *status = NOT_IMAGE;
        }

        *status
    }
}

/// Get the size of the image dimensions (= NAXISn keywords for normal image, or
/// ZNAXISn for a compressed image)
/// These values are cached for faster access.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nlen`   — (I) number of axes to return
/// * `naxes`  — (O) size of image dimensions
/// * `status` — (IO) error status
pub fn ffgisz_safe(
    fptr: &mut fitsfile,
    nlen: c_int,
    naxes: &mut [c_long],
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        /* rescan header */
        if ffrdef_safe(fptr, status) > 0 {
            return *status;
        }
    }

    if fptr.Fptr.hdutype == IMAGE_HDU {
        let naxis = cmp::min(fptr.Fptr.imgdim, nlen);

        for ii in 0..(naxis as usize) {
            naxes[ii] = fptr.Fptr.imgnaxis[ii] as c_long;
        }
    } else if fptr.Fptr.compressimg != 0 {
        let naxis = cmp::min(fptr.Fptr.zndim, nlen);

        naxes[..(naxis as usize)].copy_from_slice(&fptr.Fptr.znaxis[..(naxis as usize)]);
    } else {
        *status = NOT_IMAGE;
    }
    *status
}

/// Get the size of the image dimensions (= NAXISn keywords for normal image, or
/// ZNAXISn for a compressed image)
///
/// # Parameters
///
/// * `fptr`  — (I) FITS file pointer
/// * `nlen`  — (I) number of axes to return
/// * `naxes` — (O) size of image dimensions
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgiszll(
    fptr: *mut fitsfile,
    nlen: c_int,
    naxes: *mut LONGLONG,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        let naxes = slice::from_raw_parts_mut(naxes, nlen as usize);

        ffgiszll_safe(fptr, nlen, naxes, status)
    }
}

/// Get the size of the image dimensions (= NAXISn keywords for normal image, or
/// ZNAXISn for a compressed image)
///
/// # Parameters
///
/// * `fptr`  — (I) FITS file pointer
/// * `nlen`  — (I) number of axes to return
/// * `naxes` — (O) size of image dimensions
pub fn ffgiszll_safe(
    fptr: &mut fitsfile,
    nlen: c_int,
    naxes: &mut [LONGLONG],
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    /* reset position to the correct HDU if necessary */
    if fptr.HDUposition != fptr.Fptr.curhdu {
        ffmahd_safe(fptr, (fptr.HDUposition) + 1, None, status);
    } else if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        /* rescan header */
        if ffrdef_safe(fptr, status) > 0 {
            return *status;
        }
    }

    if fptr.Fptr.hdutype == IMAGE_HDU {
        let naxis = cmp::min(fptr.Fptr.imgdim, nlen) as usize;
        let mut ii: usize = 0;
        while ii < naxis {
            naxes[ii] = fptr.Fptr.imgnaxis[ii];
            ii += 1
        }
    } else if fptr.Fptr.compressimg != 0 {
        let naxis = cmp::min(fptr.Fptr.zndim, nlen) as usize;

        let mut ii: usize = 0;
        while ii < naxis {
            naxes[ii] = fptr.Fptr.znaxis[ii] as LONGLONG;
            ii += 1
        }
    } else {
        *status = NOT_IMAGE;
    }
    *status
}

/// Move to Absolute Header Data unit.  Move to the specified HDU
/// and read the header to initialize the table structure.  Note that extnum
/// is one based, so the primary array is extnum = 1.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `hdunum`  — (I) number of the HDU to move to
/// * `exttype` — (O) type of extension, 0, 1, or 2
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffmahd(
    fptr: *mut fitsfile,
    hdunum: c_int,
    exttype: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let exttype = exttype.as_mut();

        ffmahd_safe(fptr, hdunum, exttype, status)
    }
}

/// Move to Absolute Header Data unit.  Move to the specified HDU
/// and read the header to initialize the table structure.  Note that extnum
/// is one based, so the primary array is extnum = 1.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `hdunum`  — (I) number of the HDU to move to
/// * `exttype` — (O) type of extension, 0, 1, or 2
/// * `status`  — (IO) error status
pub fn ffmahd_safe(
    fptr: &mut fitsfile,
    hdunum: c_int,
    exttype: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let mut moveto;
    let mut tstatus;
    let mut message: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    let mut exttype = exttype;

    if *status > 0 {
        return *status;
    } else if hdunum < 1 {
        *status = BAD_HDU_NUM;
        return *status;
    } else if hdunum >= fptr.Fptr.MAXHDU {
        /* allocate more space for the headstart array */
        // HEAP ALLOCATION
        if let Err(e) = grow_headstart(fptr) {
            *status = e;
            return *status;
        }
    }

    /* set logical HDU position to the actual position, in case they differ */
    fptr.HDUposition = fptr.Fptr.curhdu;

    while (fptr.Fptr.curhdu) + 1 != hdunum {
        /* at the correct HDU? */

        /* move directly to the extension if we know that it exists,
        otherwise move to the highest known extension.  */

        moveto = cmp::min(hdunum - 1, (fptr.Fptr.maxhdu) + 1);

        /* test if HDU exists */
        let headstart = fptr.Fptr.get_headstart_as_slice();
        if headstart[moveto as usize] < fptr.Fptr.logfilesize {
            if ffchdu(fptr, status) <= 0 {
                /* close out the current HDU */

                if ffgext(fptr, moveto, exttype.as_deref_mut(), status) > 0 {
                    /* failed to get the requested extension */
                    tstatus = 0;
                    ffrhdu_safe(fptr, exttype.as_deref_mut(), &mut tstatus);
                    /* restore the CHDU */
                };
            };
        } else {
            *status = END_OF_FILE;
        }
        if *status > 0 {
            if *status != END_OF_FILE {
                /* don't clutter up the message stack in the common case of */
                /* simply hitting the end of file (often an expected error) */

                int_snprintf!(
                    &mut message,
                    FLEN_ERRMSG,
                    "Failed to move to HDU number {} (ffmahd).",
                    hdunum,
                );

                ffpmsg_slice(&message);
            }
            return *status;
        };
    }

    if let Some(exttype) = exttype {
        ffghdt_safe(fptr, exttype, status);
    }

    *status
}

/// Move a Relative number of Header Data units.  Offset to the specified
/// extension and read the header to initialize the HDU structure.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `hdumov`  — (I) rel. no. of HDUs to move by (+ or -)
/// * `exttype` — (O) type of extension, 0, 1, or 2
/// * `status`  — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffmrhd(
    fptr: *mut fitsfile,
    hdumov: c_int,
    exttype: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let exttype = exttype.as_mut();

        ffmrhd_safe(fptr, hdumov, exttype, status)
    }
}

/// Move a Relative number of Header Data units.  Offset to the specified
/// extension and read the header to initialize the HDU structure.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `hdumov`  — (I) rel. no. of HDUs to move by (+ or -)
/// * `exttype` — (O) type of extension, 0, 1, or 2
/// * `status`  — (IO) error status
pub fn ffmrhd_safe(
    fptr: &mut fitsfile,
    hdumov: c_int,
    exttype: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status;
    }

    let extnum = fptr.HDUposition + 1 + hdumov; /* the absolute HDU number */
    ffmahd_safe(fptr, extnum, exttype, status); /* move to the HDU */

    *status
}

/// Move to the next HDU with a given extension type (IMAGE_HDU, ASCII_TBL,
/// BINARY_TBL, or ANY_HDU), extension name (EXTNAME or HDUNAME keyword),
/// and EXTVERS keyword values.  If hduvers = 0, then move to the first HDU
/// with the given type and name regardless of EXTVERS value.  If no matching
/// HDU is found in the file, then the current open HDU will remain unchanged.
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffmnhd(
    fptr: *mut fitsfile,
    exttype: c_int,
    hduname: *const c_char,
    hduver: c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);

        raw_to_slice!(hduname);

        ffmnhd_safe(fptr, exttype, hduname, hduver, status)
    }
}

/// Move to the next HDU with a given extension type (IMAGE_HDU, ASCII_TBL,
/// BINARY_TBL, or ANY_HDU), extension name (EXTNAME or HDUNAME keyword),
/// and EXTVERS keyword values.  If hduvers = 0, then move to the first HDU
/// with the given type and name regardless of EXTVERS value.  If no matching
/// HDU is found in the file, then the current open HDU will remain unchanged.
pub fn ffmnhd_safe(
    fptr: &mut fitsfile,
    exttype: c_int,
    hduname: &[c_char],
    hduver: c_int,
    status: &mut c_int,
) -> c_int {
    let mut extname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut hdutype: c_int = 0;
    let mut alttype: c_int;

    let mut tstatus: c_int;
    let mut matched: c_int = 0;
    let mut exact: c_int = 0;
    let mut slen;
    let mut putback: c_int = 0;
    let mut chopped: c_int = 0;
    let mut extver: c_long = 0;

    if *status > 0 {
        return *status;
    }

    let extnum: c_int = fptr.HDUposition + 1; /* save the current HDU number */

    /*
       This is a kludge to deal with a special case where the
       user specified a hduname that ended with a # character, which
       CFITSIO previously interpreted as a flag to mean "don't copy any
       other HDUs in the file into the virtual file in memory.  If the
       remaining hduname does not end with a # character (meaning that
       the user originally entered a hduname ending in 2 # characters)
       then there is the possibility that the # character should be
       treated literally, if the actual EXTNAME also ends with a #.
       Setting putback = 1 means that we need to test for this case later on.
    */

    if fptr.Fptr.only_one != 0 {
        /* if true, name orignally ended with a # */
        slen = strlen_safe(hduname);
        if hduname[slen - 1] != bb(b'#') {
            /* This will fail if real EXTNAME value */
            putback = 1; /*  ends with 2 # characters. */
        }
    }

    let mut ii = 1;
    loop
    /* loop over all HDUs until EOF */
    {
        tstatus = 0;
        if ffmahd_safe(fptr, ii, Some(&mut hdutype), &mut tstatus) != 0
        /* move to next HDU */
        {
            ffmahd_safe(fptr, extnum, None, status); /* restore original file position */
            *status = BAD_HDU_NUM; /* couldn't find desired HDU */
            return *status;
        }

        alttype = -1;
        if fits_is_compressed_image_safe(fptr, status) != 0 {
            alttype = BINARY_TBL;
        }

        /* Does this HDU have a matching type? */
        if exttype == ANY_HDU || hdutype == exttype || hdutype == alttype {
            ffmaky_safe(fptr, 2, status); /* reset to the 2nd keyword in the header */
            if ffgkys_safe(fptr, cs!(c"EXTNAME"), &mut extname, None, &mut tstatus) <= 0 {
                /* get keyword */
                if putback != 0 {
                    /* more of the kludge */
                    /* test if the EXTNAME value ends with a #;  if so, chop it  */
                    /* off before comparing the strings */
                    chopped = 0;
                    slen = strlen_safe(&extname);
                    if extname[slen - 1] == bb(b'#') {
                        extname[slen - 1] = 0;
                        chopped = 1;
                    }
                }

                /* see if the strings are an exact match */
                ffcmps_safe(
                    hduname,
                    &extname,
                    CASEINSEN as c_int,
                    &mut matched,
                    &mut exact,
                );
            }

            /* if EXTNAME keyword doesn't exist, or it does not match, then try HDUNAME */
            if tstatus != 0 || exact == 0 {
                tstatus = 0;
                if ffgkys_safe(fptr, cs!(c"HDUNAME"), &mut extname, None, &mut tstatus) <= 0 {
                    if putback != 0 {
                        /* more of the kludge */
                        chopped = 0;
                        slen = strlen_safe(&extname);
                        if extname[slen - 1] == bb(b'#') {
                            extname[slen - 1] = 0; /* chop off the # */
                            chopped = 1;
                        }
                    }

                    /* see if the strings are an exact match */
                    ffcmps_safe(
                        hduname,
                        &extname,
                        CASEINSEN as c_int,
                        &mut matched,
                        &mut exact,
                    );
                }
            }

            if tstatus == 0 && exact != 0
            /* found a matching name */
            {
                if hduver != 0
                /* need to check if version numbers match? */
                {
                    if ffgkyj_safe(fptr, cs!(c"EXTVER"), &mut extver, None, &mut tstatus) > 0 {
                        extver = 1; /* assume default EXTVER value */
                    }

                    if extver == c_long::from(hduver) {
                        if chopped != 0 {
                            /* The # was literally part of the name, not a flag */
                            fptr.Fptr.only_one = 0;
                        }
                        return *status; /* found matching name and vers */
                    }
                } else {
                    if chopped != 0 {
                        /* The # was literally part of the name, not a flag */
                        fptr.Fptr.only_one = 0;
                    }
                    return *status; /* found matching name */
                }
            } /* end of !tstatus && exact */
        } /* end of matching HDU type */
        ii += 1;
    } /* end of loop over HDUs */

    *status
}

///  Return the number of HDUs that currently exist in the file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nhdu`   — (O) number of HDUs in the file
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffthdu(
    fptr: *mut fitsfile,
    nhdu: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let fptr = fptr.as_mut().expect(NULL_MSG);
        let nhdu = nhdu.as_mut().expect(NULL_MSG);

        ffthdu_safe(fptr, nhdu, status)
    }
}

/// Count the total number of HDUs in the file.
///
/// # Parameters
///
/// * `fptr`   — (I) FITS file pointer
/// * `nhdu`   — (O) number of HDUs in the file
/// * `status` — (IO) error status
pub fn ffthdu_safe(fptr: &mut fitsfile, nhdu: &mut c_int, status: &mut c_int) -> c_int {
    if *status > 0 {
        return *status;
    }

    let extnum = fptr.HDUposition + 1; /* save the current HDU number */
    *nhdu = extnum - 1;

    /* if the CHDU is empty or not completely defined, just return */
    if fptr.Fptr.datastart == DATA_UNDEFINED as LONGLONG {
        return *status;
    }

    /* loop until EOF */
    let mut tstatus = 0;
    let mut ii = extnum;
    while ffmahd_safe(fptr, ii, None, &mut tstatus) <= 0 {
        *nhdu = ii;
        ii += 1
    }
    ffmahd_safe(fptr, extnum, None, status); /* restore orig file position */
    *status
}

/// Get Extension.  Move to the specified extension and initialize the
/// HDU structure.
///
/// # Parameters
///
/// * `fptr`    — (I) FITS file pointer
/// * `hdunum`  — (I) no. of HDU to move get (0 based)
/// * `exttype` — (O) type of extension, 0, 1, or 2
/// * `status`  — (IO) error status
pub(crate) fn ffgext(
    fptr: &mut fitsfile,
    hdunum: c_int,
    exttype: Option<&mut c_int>,
    status: &mut c_int,
) -> c_int {
    let xcurhdu: c_int;
    let xmaxhdu: c_int;
    let xheadend: LONGLONG;

    if *status > 0 {
        return *status;
    }

    let headstart = fptr.Fptr.get_headstart_as_slice();

    if ffmbyt_safe(fptr, headstart[hdunum as usize], REPORT_EOF, status) <= 0 {
        /* temporarily save current values, in case of error */
        xcurhdu = fptr.Fptr.curhdu;
        xmaxhdu = fptr.Fptr.maxhdu;
        xheadend = fptr.Fptr.headend;

        /* set new parameter values */
        fptr.Fptr.curhdu = hdunum;
        fptr.HDUposition = hdunum;
        fptr.Fptr.maxhdu = cmp::max(fptr.Fptr.maxhdu, hdunum);
        fptr.Fptr.headend = fptr.Fptr.logfilesize; /* set max size */

        if ffrhdu_safe(fptr, exttype, status) > 0 {
            /* failed to get the new HDU, so restore previous values */
            fptr.Fptr.curhdu = xcurhdu;
            fptr.HDUposition = xcurhdu;
            fptr.Fptr.maxhdu = xmaxhdu;
            fptr.Fptr.headend = xheadend;
        }
    }
    *status
}

/// Insert 2880-byte blocks at the end of the current header or data unit
///
/// # Parameters
///
/// * `fptr`     — (I) FITS file pointer
/// * `nblock`   — (I) no. of blocks to insert
/// * `headdata` — (I) insert where? 0=header, 1=data
/// * `status`   — (IO) error status
pub(crate) fn ffiblk(
    fptr: &mut fitsfile,
    nblock: c_long,
    headdata: c_int,
    /*     -1=beginning of file            */
    status: &mut c_int,
) -> c_int {
    let mut typhdu: c_int = 0;
    let mut insertpt: LONGLONG;
    let charfill: c_char;

    let buff1: [c_char; BLOCK_LEN] = [0; BLOCK_LEN];
    let buff2: [c_char; BLOCK_LEN] = [0; BLOCK_LEN];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    if *status > 0 || nblock <= 0 {
        return *status;
    }

    let mut tstatus = *status;

    if headdata == 0 || fptr.Fptr.hdutype == ASCII_TBL {
        charfill = 32; /* headers and ASCII tables have space (32) fill */
    } else {
        charfill = 0; /* images and binary tables have zero fill */
    }

    if headdata == 0 {
        insertpt = fptr.Fptr.datastart; /* insert just before data, or */
    } else if headdata == -1 {
        insertpt = 0;
        strcpy_safe(
            &mut card,
            cs!(c"XTENSION= 'IMAGE   '          / IMAGE extension"),
        );
    } else
    /* at end of data, */
    {
        insertpt = fptr.Fptr.datastart + fptr.Fptr.heapstart + fptr.Fptr.heapsize;
        insertpt = ((insertpt + (BL!() - 1)) / BL!()) * BL!(); /* start of block */

        /* the following formula is wrong because the current data unit
           may have been extended without updating the headstart value
           of the following HDU.
        */
        /* insertpt = fptr.Fptr.headstart[fptr.Fptr.curhdu + 1]; */
    }

    let mut inbuff = buff1; /* set pointers to input and output buffers */
    let mut outbuff = buff2;

    outbuff.fill(charfill); /* initialize buffer with fill */

    if nblock == 1 {
        /* insert one block */

        if headdata == -1 {
            ffmrec_safe(fptr, 1, &card, status); /* change SIMPLE -> XTENSION */
        }

        ffmbyt_safe(fptr, insertpt, REPORT_EOF, status); /* move to 1st point */
        ffgbyt(fptr, BL!(), cast_slice_mut(&mut inbuff), status); /* read first block of bytes */

        while *status <= 0 {
            ffmbyt_safe(fptr, insertpt, REPORT_EOF, status); /* insert point */
            ffpbyt(fptr, BL!(), cast_slice(&outbuff), status); /* write the output buffer */

            if *status > 0 {
                return *status;
            }

            core::mem::swap(&mut inbuff, &mut outbuff);
            insertpt += BL!(); /* increment insert point by 1 block */

            ffmbyt_safe(fptr, insertpt, REPORT_EOF, status); /* move to next block */
            ffgbyt(fptr, BL!(), cast_slice_mut(&mut inbuff), status); /* read block of bytes */
        }

        *status = tstatus; /* reset status value */
        ffmbyt_safe(fptr, insertpt, IGNORE_EOF, status); /* move back to insert pt */
        ffpbyt(fptr, BL!(), cast_slice(&outbuff), status); /* write the final block */
    } else {
        /*  inserting more than 1 block */

        let savehdu = fptr.Fptr.curhdu; /* save the current HDU number */
        tstatus = *status;
        while *status <= 0 {
            /* find the last HDU in file */
            ffmrhd_safe(fptr, 1, Some(&mut typhdu), status);
        }

        if *status == END_OF_FILE {
            *status = tstatus;
        }

        ffmahd_safe(fptr, savehdu + 1, Some(&mut typhdu), status); /* move back to CHDU */
        if headdata == -1 {
            ffmrec_safe(fptr, 1, &card, status); /* NOW change SIMPLE -> XTENSION */
        }

        /* number of 2880-byte blocks that have to be shifted down */
        let headstart = fptr.Fptr.get_headstart_as_slice();
        let nshift = ((headstart[(fptr.Fptr.maxhdu + 1) as usize] - insertpt) / BL!()) as c_long;
        /* position of last block in file to be shifted */
        let mut jpoint = headstart[(fptr.Fptr.maxhdu + 1) as usize] - BL!();

        /* move all the blocks starting at end of file working backwards */
        for _ii in 0..(nshift as usize) {
            /* move to the read start position */
            if ffmbyt_safe(fptr, jpoint, REPORT_EOF, status) > 0 {
                return *status;
            }

            ffgbyt(fptr, BL!(), cast_slice_mut(&mut inbuff), status); /* read one record */

            /* move forward to the write postion */
            ffmbyt_safe(
                fptr,
                jpoint + (nblock * BL!()) as LONGLONG,
                IGNORE_EOF,
                status,
            );

            ffpbyt(fptr, BL!(), cast_slice_mut(&mut inbuff), status); /* write the record */

            jpoint -= BL!();
        }

        /* move back to the write start postion (might be EOF) */
        ffmbyt_safe(fptr, insertpt, IGNORE_EOF, status);

        for _ii in 0..(nblock as usize) {
            /* insert correct fill value */
            ffpbyt(fptr, BL!(), cast_slice(&outbuff), status);
        }
    }

    if headdata == 0 {
        /* update data start address */
        fptr.Fptr.datastart += (nblock * BL!()) as LONGLONG;
    }

    /* update following HDU addresses */
    let range = (fptr.Fptr.curhdu as usize)..=(fptr.Fptr.maxhdu as usize);
    let headstart = fptr.Fptr.get_headstart_as_mut_slice();
    for ii in range {
        headstart[ii + 1] += (nblock * BL!()) as LONGLONG;
    }

    *status
}

/// Return the type classification of the input header record
/// ```text
/// TYP_STRUC_KEY: SIMPLE, BITPIX, NAXIS, NAXISn, EXTEND, BLOCKED,
///                GROUPS, PCOUNT, GCOUNT, END
///                XTENSION, TFIELDS, TTYPEn, TBCOLn, TFORMn, THEAP,
///                 and the first 4 COMMENT keywords in the primary array
///                 that define the FITS format.
///
/// TYP_CMPRS_KEY:
///          The keywords used in the compressed image format
///                ZIMAGE, ZCMPTYPE, ZNAMEn, ZVALn, ZTILEn,
///                ZBITPIX, ZNAXISn, ZSCALE, ZZERO, ZBLANK,
///                EXTNAME = 'COMPRESSED_IMAGE'
///        ZSIMPLE, ZTENSION, ZEXTEND, ZBLOCKED, ZPCOUNT, ZGCOUNT
///        ZQUANTIZ, ZDITHER0
///
/// TYP_SCAL_KEY:  BSCALE, BZERO, TSCALn, TZEROn
///
/// TYP_NULL_KEY:  BLANK, TNULLn
///
/// TYP_DIM_KEY:   TDIMn
///
/// TYP_RANG_KEY:  TLMINn, TLMAXn, TDMINn, TDMAXn, DATAMIN, DATAMAX
///
/// TYP_UNIT_KEY:  BUNIT, TUNITn
///
/// TYP_DISP_KEY:  TDISPn
///
/// TYP_HDUID_KEY: EXTNAME, EXTVER, EXTLEVEL, HDUNAME, HDUVER, HDULEVEL
///
/// TYP_CKSUM_KEY  CHECKSUM, DATASUM
///
/// TYP_WCS_KEY:
///         Primary array:
///                WCAXES, CTYPEn, CUNITn, CRVALn, CRPIXn, CROTAn, CDELTn
///                CDj_is, PVj_ms, LONPOLEs, LATPOLEs
///
///         Pixel list:
///                TCTYPn, TCTYns, TCUNIn, TCUNns, TCRVLn, TCRVns, TCRPXn, TCRPks,
///                TCDn_k, TCn_ks, TPVn_m, TPn_ms, TCDLTn, TCROTn
///
///         Bintable vector:
///                jCTYPn, jCTYns, jCUNIn, jCUNns, jCRVLn, jCRVns, iCRPXn, iCRPns,
///                jiCDn, jiCDns, jPVn_m, jPn_ms, jCDLTn, jCROTn
///              
/// TYP_REFSYS_KEY:
///                 EQUINOXs, EPOCH, MJD-OBSs, RADECSYS, RADESYSs
///
/// TYP_COMM_KEY:  COMMENT, HISTORY, (blank keyword)
///
/// TYP_CONT_KEY:  CONTINUE
///
/// TYP_USER_KEY:  all other keywords
/// ```
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffgkcl(tcard: *mut c_char) -> c_int {
    // FFI WRAPPER
    unsafe {
        raw_to_slice!(tcard);

        ffgkcl_safe(tcard)
    }
}

#[allow(clippy::if_same_then_else)]
// C dispatch chain: distinct conditions deliberately share an action.
/// Return the classification code of a header keyword, given its 80-character
/// card image.
///
/// # Parameters
///
/// * `tcard` — (I) the keyword card to classify
///
/// # Returns
///
/// One of the `TYP_*_KEY` codes, e.g. [`TYP_STRUC_KEY`] for `SIMPLE` or
/// [`TYP_USER_KEY`] for anything unrecognized.
pub fn ffgkcl_safe(tcard: &[c_char]) -> c_int {
    let mut card: [c_char; 20] = [0; 20];

    card[0] = 0;

    strncat_safe(&mut card, tcard, 8); /* copy the keyword name */
    strcat_safe(&mut card, cs!(c"        ")); /* append blanks to make at least 8 chars long */
    ffupch_safe(&mut card); /* make sure it is in upper case */

    let card1 = &card[1..]; /* pointer to 2nd character */
    let card5 = card[5]; /* pointer to 6th character */

    /* the strncmp function is slow, so try to be more efficient */
    if card[0] == bb(b'Z') {
        if FSTRNCMP(card1, cs!(c"IMAGE  "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"CMPTYPE"), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"NAME"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_CMPRS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"VAL"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_CMPRS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"TILE"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_CMPRS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"BITPIX "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"NAXIS"), 5) == 0 {
            if (card[6] >= bb(b'0') && card[6] <= bb(b'9')) || (card[6] == bb(b' ')) {
                return TYP_CMPRS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"SCALE  "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"ZERO   "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"BLANK  "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"SIMPLE "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"TENSION"), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"EXTEND "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"BLOCKED"), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"PCOUNT "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"GCOUNT "), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"QUANTIZ"), 7) == 0 {
            return TYP_CMPRS_KEY;
        } else if FSTRNCMP(card1, cs!(c"DITHER0"), 7) == 0 {
            return TYP_CMPRS_KEY;
        };
    } else if card[0] == bb(b' ') {
        return TYP_COMM_KEY;
    } else if card[0] == bb(b'B') {
        if FSTRNCMP(card1, cs!(c"ITPIX  "), 7) == 0 {
            return TYP_STRUC_KEY;
        }
        if FSTRNCMP(card1, cs!(c"LOCKED "), 7) == 0 {
            return TYP_STRUC_KEY;
        }
        if FSTRNCMP(card1, cs!(c"LANK   "), 7) == 0 {
            return TYP_NULL_KEY;
        }
        if FSTRNCMP(card1, cs!(c"SCALE  "), 7) == 0 {
            return TYP_SCAL_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ZERO   "), 7) == 0 {
            return TYP_SCAL_KEY;
        }
        if FSTRNCMP(card1, cs!(c"UNIT   "), 7) == 0 {
            return TYP_UNIT_KEY;
        };
    } else if card[0] == bb(b'C') {
        /* new comment string starting Oct 2001 */
        if FSTRNCMP(card1, cs!(c"OMMENT"), 6) == 0 {
            if FSTRNCMP(
                tcard,
                cs!(c"COMMENT   and Astrophysics\', volume 376, page 3"),
                47,
            ) == 0
            {
                return TYP_STRUC_KEY;
            }

            /* original COMMENT strings from 1993 - 2001 */
            if FSTRNCMP(
                tcard,
                cs!(c"COMMENT   FITS (Flexible Image Transport System"),
                47,
            ) == 0
            {
                return TYP_STRUC_KEY;
            }
            if FSTRNCMP(
                tcard,
                cs!(c"COMMENT   Astrophysics Supplement Series v44/p3"),
                47,
            ) == 0
            {
                return TYP_STRUC_KEY;
            }
            if FSTRNCMP(
                tcard,
                cs!(c"COMMENT   Contact the NASA Science Office of St"),
                47,
            ) == 0
            {
                return TYP_STRUC_KEY;
            }
            if FSTRNCMP(
                tcard,
                cs!(c"COMMENT   FITS Definition document #100 and oth"),
                47,
            ) == 0
            {
                return TYP_STRUC_KEY;
            }
            if card[7] == bb(b' ') {
                return TYP_COMM_KEY;
            } else {
                return TYP_USER_KEY;
            };
        }
        if FSTRNCMP(card1, cs!(c"HECKSUM"), 7) == 0 {
            return TYP_CKSUM_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ONTINUE"), 7) == 0 {
            return TYP_CONT_KEY;
        }
        if FSTRNCMP(card1, cs!(c"TYPE"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"UNIT"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"RVAL"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"RPIX"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"ROTA"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"RDER"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"SYER"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"DELT"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if card1[0] == bb(b'D') && card[2] >= bb(b'0') && card[2] <= bb(b'9') {
            return TYP_WCS_KEY;
        };
    } else if card[0] == bb(b'D') {
        if FSTRNCMP(card1, cs!(c"ATASUM "), 7) == 0 {
            return TYP_CKSUM_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ATAMIN "), 7) == 0 {
            return TYP_RANG_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ATAMAX "), 7) == 0 {
            return TYP_RANG_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ATE-OBS"), 7) == 0 {
            return TYP_REFSYS_KEY;
        };
    } else if card[0] == bb(b'E') {
        if FSTRNCMP(card1, cs!(c"XTEND  "), 7) == 0 {
            return TYP_STRUC_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ND     "), 7) == 0 {
            return TYP_STRUC_KEY;
        }
        if FSTRNCMP(card1, cs!(c"XTNAME "), 7) == 0 {
            /* check for special compressed image value */
            if FSTRNCMP(tcard, cs!(c"EXTNAME = \'COMPRESSED_IMAGE\'"), 28) == 0 {
                return TYP_CMPRS_KEY;
            } else {
                return TYP_HDUID_KEY;
            };
        }
        if FSTRNCMP(card1, cs!(c"XTVER  "), 7) == 0 {
            return TYP_HDUID_KEY;
        }
        if FSTRNCMP(card1, cs!(c"XTLEVEL"), 7) == 0 {
            return TYP_HDUID_KEY;
        }
        if FSTRNCMP(card1, cs!(c"QUINOX"), 6) == 0 {
            return TYP_REFSYS_KEY;
        }
        if FSTRNCMP(card1, cs!(c"QUI"), 3) == 0 && card[4] >= bb(b'0') && card[4] <= bb(b'9') {
            return TYP_REFSYS_KEY;
        };
        if FSTRNCMP(card1, cs!(c"POCH   "), 7) == 0 {
            return TYP_REFSYS_KEY;
        };
    } else if card[0] == bb(b'G') {
        if FSTRNCMP(card1, cs!(c"COUNT  "), 7) == 0 {
            return TYP_STRUC_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ROUPS  "), 7) == 0 {
            return TYP_STRUC_KEY;
        };
    } else if card[0] == bb(b'H') {
        if FSTRNCMP(card1, cs!(c"DUNAME "), 7) == 0 {
            return TYP_HDUID_KEY;
        }
        if FSTRNCMP(card1, cs!(c"DUVER  "), 7) == 0 {
            return TYP_HDUID_KEY;
        }
        if FSTRNCMP(card1, cs!(c"DULEVEL"), 7) == 0 {
            return TYP_HDUID_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ISTORY"), 6) == 0 {
            if card[7] == bb(b' ') {
                return TYP_COMM_KEY;
            } else {
                return TYP_USER_KEY;
            };
        };
    } else if card[0] == bb(b'L') {
        if FSTRNCMP(card1, cs!(c"ONPOLE"), 6) == 0 {
            return TYP_WCS_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ATPOLE"), 6) == 0 {
            return TYP_WCS_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ONP"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"ATP"), 3) == 0 && card[4] >= bb(b'0') && card[4] <= bb(b'9')
        {
            return TYP_WCS_KEY;
        };
    } else if card[0] == bb(b'M') {
        if FSTRNCMP(card1, cs!(c"JD-OBS "), 7) == 0 {
            return TYP_REFSYS_KEY;
        }
        if FSTRNCMP(card1, cs!(c"JDOB"), 4) == 0 && card[5] >= bb(b'0') && card[5] <= bb(b'9') {
            return TYP_REFSYS_KEY;
        };
    } else if card[0] == bb(b'N') {
        if FSTRNCMP(card1, cs!(c"AXIS"), 4) == 0
            && ((card5 >= bb(b'0') && card5 <= bb(b'9')) || (card5 == bb(b' ')))
        {
            return TYP_STRUC_KEY;
        };
    } else if card[0] == bb(b'P') {
        if FSTRNCMP(card1, cs!(c"COUNT  "), 7) == 0 {
            return TYP_STRUC_KEY;
        }
        if card1[0] == bb(b'C') {
            if card[2] >= bb(b'0') && card[2] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if card1[0] == bb(b'V') {
            if card[2] >= bb(b'0') && card[2] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if card1[0] == bb(b'S') && card[2] >= bb(b'0') && card[2] <= bb(b'9') {
            return TYP_WCS_KEY;
        };
    } else if card[0] == bb(b'R') {
        if FSTRNCMP(card1, cs!(c"ADECSYS"), 7) == 0 {
            return TYP_REFSYS_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ADESYS"), 6) == 0 {
            return TYP_REFSYS_KEY;
        }
        if FSTRNCMP(card1, cs!(c"ADE"), 3) == 0 && card[4] >= bb(b'0') && card[4] <= bb(b'9') {
            return TYP_REFSYS_KEY;
        };
    } else if card[0] == bb(b'S') {
        if FSTRNCMP(card1, cs!(c"IMPLE  "), 7) == 0 {
            return TYP_STRUC_KEY;
        };
    } else if card[0] == bb(b'T') {
        if FSTRNCMP(card1, cs!(c"TYPE"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_STRUC_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"FORM"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_STRUC_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"BCOL"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_STRUC_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"FIELDS "), 7) == 0 {
            return TYP_STRUC_KEY;
        } else if FSTRNCMP(card1, cs!(c"HEAP   "), 7) == 0 {
            return TYP_STRUC_KEY;
        } else if FSTRNCMP(card1, cs!(c"NULL"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_NULL_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"DIM"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_DIM_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"UNIT"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_UNIT_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"DISP"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_DISP_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"SCAL"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_SCAL_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"ZERO"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_SCAL_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"LMIN"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_RANG_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"LMAX"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_RANG_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"DMIN"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_RANG_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"DMAX"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_RANG_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CTYP"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CTY"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CUNI"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CUN"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CRVL"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CRV"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CRPX"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CRP"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CROT"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CDLT"), 4) == 0 {
            if card5 >= bb(b'0') && card5 <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CDE"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CRD"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CSY"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"WCS"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"C"), 1) == 0 {
            if card[2] >= bb(b'0') && card[2] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"P"), 1) == 0 {
            if card[2] >= bb(b'0') && card[2] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"V"), 1) == 0 {
            if card[2] >= bb(b'0') && card[2] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"S"), 1) == 0 && card[2] >= bb(b'0') && card[2] <= bb(b'9') {
            return TYP_WCS_KEY;
        };
    } else if card[0] == bb(b'X') {
        if FSTRNCMP(card1, cs!(c"TENSION"), 7) == 0 {
            return TYP_STRUC_KEY;
        };
    } else if card[0] == bb(b'W') {
        if FSTRNCMP(card1, cs!(c"CSAXES"), 6) == 0 {
            return TYP_WCS_KEY;
        }
        if FSTRNCMP(card1, cs!(c"CSNAME"), 6) == 0 {
            return TYP_WCS_KEY;
        }
        if FSTRNCMP(card1, cs!(c"CAX"), 3) == 0 {
            if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"CSN"), 3) == 0 && card[4] >= bb(b'0') && card[4] <= bb(b'9')
        {
            return TYP_WCS_KEY;
        };
    } else if card[0] >= bb(b'0') && card[0] <= bb(b'9') {
        if card1[0] == bb(b'C') {
            if FSTRNCMP(card1, cs!(c"CTYP"), 4) == 0 {
                if card5 >= bb(b'0') && card5 <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CTY"), 3) == 0 {
                if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CUNI"), 4) == 0 {
                if card5 >= bb(b'0') && card5 <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CUN"), 3) == 0 {
                if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CRVL"), 4) == 0 {
                if card5 >= bb(b'0') && card5 <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CRV"), 3) == 0 {
                if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CRPX"), 4) == 0 {
                if card5 >= bb(b'0') && card5 <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CRP"), 3) == 0 {
                if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CROT"), 4) == 0 {
                if card5 >= bb(b'0') && card5 <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CDLT"), 4) == 0 {
                if card5 >= bb(b'0') && card5 <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CDE"), 3) == 0 {
                if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CRD"), 3) == 0 {
                if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                    return TYP_WCS_KEY;
                };
            } else if FSTRNCMP(card1, cs!(c"CSY"), 3) == 0
                && card[4] >= bb(b'0')
                && card[4] <= bb(b'9')
            {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"V"), 1) == 0 {
            if card[2] >= bb(b'0') && card[2] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if FSTRNCMP(card1, cs!(c"S"), 1) == 0 {
            if card[2] >= bb(b'0') && card[2] <= bb(b'9') {
                return TYP_WCS_KEY;
            };
        } else if card1[0] >= bb(b'0') && card1[0] <= bb(b'9') {
            /* 2 digits at beginning of keyword */

            if (card[2] == bb(b'P')) && (card[3] == bb(b'C')) {
                if card[4] >= bb(b'0') && card[4] <= bb(b'9') {
                    return TYP_WCS_KEY; /*  ijPCn keyword */
                };
            } else if (card[2] == bb(b'C'))
                && (card[3] == bb(b'D'))
                && card[4] >= bb(b'0')
                && card[4] <= bb(b'9')
            {
                return TYP_WCS_KEY; /*  ijCDn keyword */
            };
        };
    }

    TYP_USER_KEY /* by default all others are user keywords */
}

/// determine implicit datatype of input string.
/// This assumes that the string conforms to the FITS standard
/// for keyword values, so may not detect all invalid formats.
///
/// # Parameters
///
/// * `cval`   — (I) formatted string representation of the value
/// * `dtype`  — (O) datatype code: C, L, F, I, or X
/// * `status` — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffdtyp(
    cval: *const c_char,
    dtype: *mut c_char,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let dtype = dtype.as_mut().expect(NULL_MSG);
        raw_to_slice!(cval);

        ffdtyp_safe(cval, dtype, status)
    }
}

/// determine implicit datatype of input string.
/// This assumes that the string conforms to the FITS standard
/// for keyword values, so may not detect all invalid formats.
///
/// # Parameters
///
/// * `cval`   — (I) formatted string representation of the value
/// * `dtype`  — (O) datatype code: C, L, F, I, or X
/// * `status` — (IO) error status
pub fn ffdtyp_safe(cval: &[c_char], dtype: &mut c_char, status: &mut c_int) -> c_int {
    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if cval[0] == 0 {
        *status = VALUE_UNDEFINED;
        return *status;
    } else if cval[0] == bb(b'\'') {
        *dtype = bb(b'C'); /* character string starts with a quote */
    } else if cval[0] == bb(b'T') || cval[0] == bb(b'F') {
        *dtype = bb(b'L'); /* logical = T or F character */
    } else if cval[0] == bb(b'(') {
        *dtype = bb(b'X'); /* complex datatype "(1.2, -3.4)" */
    } else if strchr_safe(cval, bb(b'.')).is_some() {
        *dtype = bb(b'F'); /* float usualy contains a decimal point */
    } else if strchr_safe(cval, bb(b'E')).is_some() || strchr_safe(cval, bb(b'D')).is_some() {
        *dtype = bb(b'F'); /* exponential contains a E or D */
    } else {
        *dtype = bb(b'I'); /* if none of the above assume it is integer */
    }
    *status
}

/// determine implicit datatype of input integer string.
/// This assumes that the string conforms to the FITS standard
/// for integer keyword value, so may not detect all invalid formats.
///
/// # Parameters
///
/// * `cval`     — (I) formatted string representation of the integer
/// * `dtype`    — (O) datatype code: TBYTE, TSHORT, TUSHORT, etc
/// * `negative` — (O) is cval negative?
/// * `status`   — (IO) error status
#[cfg_attr(not(test), unsafe(no_mangle), deprecated)]
pub unsafe extern "C" fn ffinttyp(
    cval: *const c_char,
    dtype: *mut c_int,
    negative: *mut c_int,
    status: *mut c_int,
) -> c_int {
    // FFI WRAPPER
    unsafe {
        let status = status.as_mut().expect(NULL_MSG);
        let dtype = dtype.as_mut().expect(NULL_MSG);
        let negative = negative.as_mut().expect(NULL_MSG);
        raw_to_slice!(cval);
        ffinttyp_safe(cval, dtype, negative, status)
    }
}

/// # Parameters
///
/// * `cval`     — (I) formatted string representation of the integer
/// * `dtype`    — (O) datatype code: TBYTE, TSHORT, TUSHORT, etc
/// * `negative` — (O) is cval negative?
/// * `status`   — (IO) error status
pub fn ffinttyp_safe(
    cval: &[c_char],
    dtype: &mut c_int,
    negative: &mut c_int,
    status: &mut c_int,
) -> c_int {
    if *status > 0 {
        return *status; /* inherit input status value if > 0 */
    }

    *dtype = 0;
    *negative = 0;

    let mut p = 0;
    if cval[0] == bb(b'+') {
        p += 1; /* ignore leading + sign */
    } else if cval[0] == bb(b'-') {
        p += 1;
        *negative = 1; /* this is a negative number */
    }

    if cval[0] == bb(b'0') {
        while cval[0] == bb(b'0') {
            p += 1; /* skip leading zeros */
        }

        if cval[0] == 0 {
            /* the value is a string of 1 or more zeros */
            *dtype = TSBYTE;
            return *status;
        };
    }

    let len = strlen_safe(&cval[p..]);
    for ii in 0..len {
        if !isdigit_safe(cval[p + ii]) {
            *status = BAD_INTKEY;
            return *status;
        };
    }

    /* check for unambiguous cases, based on length of the string */
    if len == 0 {
        *status = VALUE_UNDEFINED;
    } else if len < 3 {
        *dtype = TSBYTE;
    } else if len == 4 {
        *dtype = TSHORT;
    } else if len > 5 && len < 10 {
        *dtype = TINT;
    } else if len > 10 && len < 19 {
        *dtype = TLONGLONG;
    } else if len > 20 {
        *status = BAD_INTKEY;
    } else if (*negative) == 0 {
        if len == 3 {
            if strcmp_safe(&cval[p..], cs!(c"127")) <= 0 {
                *dtype = TSBYTE;
            } else if strcmp_safe(&cval[p..], cs!(c"255")) <= 0 {
                *dtype = TBYTE;
            } else {
                *dtype = TSHORT;
            };
        } else if len == 5 {
            if strcmp_safe(&cval[p..], cs!(c"32767")) <= 0 {
                *dtype = TSHORT;
            } else if strcmp_safe(&cval[p..], cs!(c"65535")) <= 0 {
                *dtype = TUSHORT;
            } else {
                *dtype = TINT;
            };
        } else if len == 10 {
            if strcmp_safe(&cval[p..], cs!(c"2147483647")) <= 0 {
                *dtype = TINT;
            } else if strcmp_safe(&cval[p..], cs!(c"4294967295")) <= 0 {
                *dtype = TUINT;
            } else {
                *dtype = TLONGLONG;
            };
        } else if len == 19 {
            if strcmp_safe(&cval[p..], cs!(c"9223372036854775807")) <= 0 {
                *dtype = TLONGLONG;
            } else {
                *dtype = TULONGLONG;
            };
        } else if len == 20 {
            if strcmp_safe(&cval[p..], cs!(c"18446744073709551615")) <= 0 {
                *dtype = TULONGLONG;
            } else {
                *status = BAD_INTKEY;
            };
        };
    } else {
        /* negative integers */

        if len == 3 {
            if strcmp_safe(&cval[p..], cs!(c"128")) <= 0 {
                *dtype = TSBYTE;
            } else {
                *dtype = TSHORT;
            };
        } else if len == 5 {
            if strcmp_safe(&cval[p..], cs!(c"32768")) <= 0 {
                *dtype = TSHORT;
            } else {
                *dtype = TINT;
            };
        } else if len == 10 {
            if strcmp_safe(&cval[p..], cs!(c"2147483648")) <= 0 {
                *dtype = TINT;
            } else {
                *dtype = TLONGLONG;
            };
        } else if len == 19 {
            if strcmp_safe(&cval[p..], cs!(c"9223372036854775808")) <= 0 {
                *dtype = TLONGLONG;
            } else {
                *status = BAD_INTKEY;
            };
        };
    }
    *status
}

/// high level routine to convert formatted character string to its
/// intrinsic data type
///
/// # Parameters
///
/// * `cval`   — (I) formatted string representation of the value
/// * `dtype`  — (O) datatype code: C, L, F, I or X
/// * `ival`   — (O) integer value
/// * `lval`   — (O) logical value
/// * `sval`   — (O) string value
/// * `dval`   — (O) double value
/// * `status` — (IO) error status
pub(crate) fn ffc2x(
    cval: &[c_char],
    dtype: &mut c_char,
    /* Only one of the following will be defined, depending on datatype */
    ival: &mut c_long,
    lval: &mut c_int,
    sval: &mut [c_char],
    dval: &mut f64,
    status: &mut c_int,
) -> c_int {
    ffdtyp_safe(cval, dtype, status); /* determine the datatype */

    if *dtype == bb(b'I') {
        ffc2ii(cval, ival, status);
    } else if *dtype == bb(b'F') {
        ffc2dd(cval, dval, status);
    } else if *dtype == bb(b'L') {
        ffc2ll(cval, lval, status);
    } else {
        ffc2s(cval, sval, status); /* C and X formats */
    }

    *status
}

/// high level routine to convert formatted character string to its
/// intrinsic data type
///
/// # Parameters
///
/// * `cval`   — (I) formatted string representation of the value
/// * `dtype`  — (O) datatype code: C, L, F, I or X
/// * `ival`   — (O) integer value
/// * `lval`   — (O) logical value
/// * `sval`   — (O) string value
/// * `dval`   — (O) double value
/// * `status` — (IO) error status
pub(crate) fn ffc2xx(
    cval: &[c_char],
    dtype: &mut c_char,

    /* Only one of the following will be defined, depending on datatype */
    ival: &mut LONGLONG,
    lval: &mut c_int,
    sval: &mut [c_char],
    dval: &mut f64,
    status: &mut c_int,
) -> c_int {
    ffdtyp_safe(cval, dtype, status); /* determine the datatype */

    if *dtype == bb(b'I') {
        ffc2jj(cval, ival, status);
    } else if *dtype == bb(b'F') {
        ffc2dd(cval, dval, status);
    } else if *dtype == bb(b'L') {
        ffc2ll(cval, lval, status);
    } else {
        ffc2s(cval, sval, status); /* C and X formats */
    }
    *status
}

/// High level routine to convert formatted character string to its
/// intrinsic data type
///
/// # Parameters
///
/// * `cval`   — (I) formatted string representation of the value
/// * `dtype`  — (O) datatype code: C, L, F, I or X
/// * `ival`   — (O) integer value
/// * `lval`   — (O) logical value
/// * `sval`   — (O) string value
/// * `dval`   — (O) double value
/// * `status` — (IO) error status
pub(crate) fn ffc2uxx(
    cval: &[c_char],
    dtype: &mut c_char,
    /* Only one of the following will be defined, depending on datatype */
    ival: &mut ULONGLONG,
    lval: &mut c_int,
    sval: &mut [c_char],
    dval: &mut f64,
    status: &mut c_int,
) -> c_int {
    ffdtyp_safe(cval, dtype, status); /* determine the datatype */

    if *dtype == bb(b'I') {
        ffc2ujj(cval, ival, status);
    } else if *dtype == bb(b'F') {
        ffc2dd(cval, dval, status);
    } else if *dtype == bb(b'L') {
        ffc2ll(cval, lval, status);
    } else {
        ffc2s(cval, sval, status); /* C and X formats */
    }
    *status
}

/// convert formatted string to an integer value, doing implicit
/// datatype conversion if necessary.
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `ival`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2i(cval: &[c_char], ival: &mut c_long, status: &mut c_int) -> c_int {
    let mut dtype: c_char = 0;
    let mut sval: [c_char; 81] = [0; 81];

    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut lval: c_int = 0;
    let mut dval: f64 = 0.0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if cval[0] == 0 {
        *status = VALUE_UNDEFINED; /* null value string */
        return *status;
    }

    /* convert the keyword to its native datatype */
    ffc2x(
        cval, &mut dtype, ival, &mut lval, &mut sval, &mut dval, status,
    );

    if dtype == bb(b'X') {
        *status = BAD_INTKEY;
    } else if dtype == bb(b'C') {
        /* try reading the string as a number */
        if ffc2dd(&sval, &mut dval, status) <= 0 {
            if dval > (LONG_MAX as f64) || dval < (LONG_MIN as f64) {
                *status = NUM_OVERFLOW;
            } else {
                *ival = dval as c_long;
            };
        };
    } else if dtype == bb(b'F') {
        if dval > (LONG_MAX as f64) || dval < (LONG_MIN as f64) {
            *status = NUM_OVERFLOW;
        } else {
            *ival = dval as c_long;
        };
    } else if dtype == bb(b'L') {
        *ival = c_long::from(lval);
    }

    if *status > 0 {
        *ival = 0;

        strcpy_safe(
            &mut msg,
            cs!(c"Error in ffc2i evaluating string as an integer: "),
        );
        strncat_safe(&mut msg, cval, 30);
        ffpmsg_slice(&msg);

        return *status;
    }

    *status
}

/// convert formatted string to a LONGLONG integer value, doing implicit
/// datatype conversion if necessary.
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `ival`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2j(cval: &[c_char], ival: &mut LONGLONG, status: &mut c_int) -> c_int {
    let mut dtype: c_char = 0;
    let mut sval: [c_char; 81] = [0; 81];
    let mut msg: [c_char; 81] = [0; 81];
    let mut lval: c_int = 0;
    let mut dval: f64 = 0.0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if cval[0] == 0 {
        *status = VALUE_UNDEFINED; /* null value string */
        return *status;
    }

    /* convert the keyword to its native datatype */
    ffc2xx(
        cval, &mut dtype, ival, &mut lval, &mut sval, &mut dval, status,
    );

    if dtype == bb(b'X') {
        *status = BAD_INTKEY;
    } else if dtype == bb(b'C') {
        /* try reading the string as a number */
        if ffc2dd(&sval, &mut dval, status) <= 0 {
            if dval > LONGLONG_MAX as f64 || dval < LONGLONG_MIN as f64 {
                *status = NUM_OVERFLOW;
            } else {
                *ival = dval as LONGLONG;
            }
        }
    } else if dtype == bb(b'F') {
        if dval > LONGLONG_MAX as f64 || dval < LONGLONG_MIN as f64 {
            *status = NUM_OVERFLOW;
        } else {
            *ival = dval as LONGLONG;
        }
    } else if dtype == bb(b'L') {
        *ival = LONGLONG::from(lval);
    }

    if *status > 0 {
        *ival = 0;
        strcpy_safe(
            &mut msg,
            cs!(c"Error in ffc2j evaluating string as a long integer: "),
        );
        strncat_safe(&mut msg, cval, 30);
        ffpmsg_slice(&msg);
        return *status;
    }

    *status
}

/// Convert formatted string to a ULONGLONG integer value, doing implicit
/// datatype conversion if necessary.
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `ival`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2uj(cval: &[c_char], ival: &mut ULONGLONG, status: &mut c_int) -> c_int {
    let mut dtype: c_char = 0;
    let mut sval: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut lval = 0;
    let mut dval: f64 = 0.0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if cval[0] == 0 {
        *status = VALUE_UNDEFINED;
        return *status;
        /* null value string */
    }

    /* convert the keyword to its native datatype */
    ffc2uxx(
        cval, &mut dtype, ival, &mut lval, &mut sval, &mut dval, status,
    );

    if dtype == bb(b'X') {
        *status = BAD_INTKEY;
    } else if dtype == bb(b'C') {
        /* try reading the string as a number */
        if ffc2dd(&sval, &mut dval, status) <= 0 {
            if dval > DULONGLONG_MAX || dval < -0.49 {
                *status = NUM_OVERFLOW;
            } else {
                *ival = dval as ULONGLONG;
            };
        };
    } else if dtype == bb(b'F') {
        if dval > DULONGLONG_MAX || dval < -0.49 {
            *status = NUM_OVERFLOW;
        } else {
            *ival = dval as ULONGLONG;
        };
    } else if dtype == bb(b'L') {
        *ival = lval as ULONGLONG;
    }

    if *status > 0 {
        *ival = 0;
        strcpy_safe(
            &mut msg,
            cs!(c"Error in ffc2j evaluating string as a long integer: "),
        );
        strncat_safe(&mut msg, cval, 30);
        ffpmsg_slice(&msg);
        return *status;
    }

    *status
}

/// Convert formatted string to a logical value, doing implicit
/// datatype conversion if necessary
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `lval`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2l(cval: &[c_char], lval: &mut c_int, status: &mut c_int) -> c_int {
    let mut dtype: c_char = 0;
    let mut sval: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut ival: c_long = 0;
    let mut dval: f64 = 0.0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if cval[0] == 0 {
        *status = VALUE_UNDEFINED;
        return *status;

        /* null value string */
    }

    /* convert the keyword to its native datatype */
    ffc2x(
        cval, &mut dtype, &mut ival, lval, &mut sval, &mut dval, status,
    );

    if dtype == bb(b'C') || dtype == bb(b'X') {
        *status = BAD_LOGICALKEY;
    }

    if *status > 0 {
        *lval = 0;
        strcpy_safe(
            &mut msg,
            cs!(c"Error in ffc2l evaluating string as a logical: "),
        );
        strncat_safe(&mut msg, cval, 30);
        ffpmsg_slice(&msg);
        return *status;
    }

    if dtype == bb(b'I') {
        if ival != 0 {
            *lval = 1;
        } else {
            *lval = 0;
        };
    } else if dtype == bb(b'F') {
        if dval != 0.0 {
            *lval = 1;
        } else {
            *lval = 0;
        };
    }
    *status
}

/// Convert formatted string to a real float value, doing implicit
/// datatype conversion if necessary
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `fval`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2r(cval: &[c_char], fval: &mut f32, status: &mut c_int) -> c_int {
    let mut dtype: c_char = 0;
    let mut sval: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut lval = 0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if cval[0] == 0 {
        *status = VALUE_UNDEFINED;
        return *status;
        /* null value string */
    }

    ffdtyp_safe(cval, &mut dtype, status); /* determine the datatype */

    if dtype == bb(b'I') || dtype == bb(b'F') {
        ffc2rr(cval, fval, status);
    } else if dtype == bb(b'L') {
        ffc2ll(cval, &mut lval, status);
        *fval = lval as f32;
    } else if dtype == bb(b'C') {
        /* try reading the string as a number */
        ffc2s(cval, &mut sval, status);
        ffc2rr(&sval, fval, status);
    } else {
        *status = BAD_FLOATKEY;
    }

    if *status > 0 {
        *fval = 0.0;
        strcpy_safe(
            &mut msg,
            cs!(c"Error in ffc2r evaluating string as a float: "),
        );
        strncat_safe(&mut msg, cval, 30);
        ffpmsg_slice(&msg);
        return *status;
    }
    *status
}

/// Convert formatted string to a double value, doing implicit
/// datatype conversion if necessary
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `dval`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2d(cval: &[c_char], dval: &mut f64, status: &mut c_int) -> c_int {
    let mut dtype: c_char = 0;
    let mut sval: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut lval = 0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if cval[0] == 0 {
        *status = VALUE_UNDEFINED;
        return *status;
        /* null value string */
    }

    ffdtyp_safe(cval, &mut dtype, status); /* determine the datatype */

    if dtype == bb(b'I') || dtype == bb(b'F') {
        ffc2dd(cval, dval, status);
    } else if dtype == bb(b'L') {
        ffc2ll(cval, &mut lval, status);
        *dval = f64::from(lval);
    } else if dtype == bb(b'C') {
        /* try reading the string as a number */
        ffc2s(cval, &mut sval, status);
        ffc2dd(&sval, dval, status);
    } else {
        *status = BAD_DOUBLEKEY;
    }

    if *status > 0 {
        *dval = 0.0;
        strcpy_safe(
            &mut msg,
            cs!(c"Error in ffc2d evaluating string as a double: "),
        );

        strncat_safe(&mut msg, cval, 30);
        ffpmsg_slice(&msg);
        return *status;
    }

    *status
}

/// convert null-terminated formatted string to an integer value
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `ival`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2ii(cval: &[c_char], ival: &mut c_long, status: &mut c_int) -> c_int {
    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    let cval: &[u8] = cast_slice(cval);
    // A malformed card can leave arbitrary/unterminated bytes here; treat those
    // as an unparseable value (BAD_C2I) rather than panicking.
    let tmp_str = slice_to_str!(cval);
    let tmp_str = tmp_str.trim();
    let tmp = atoi::<c_long>(tmp_str);

    match tmp {
        Ok(val) => {
            *ival = val;
        }
        Err(err) => match err.kind() {
            core::num::IntErrorKind::Empty => {
                *status = BAD_C2I;
            }
            core::num::IntErrorKind::InvalidDigit => {
                *status = BAD_C2I;
            }
            core::num::IntErrorKind::PosOverflow | core::num::IntErrorKind::NegOverflow => {
                let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
                strcpy_safe(
                    &mut msg,
                    cs!(c"Range Error in ffc2ii converting string to long int: "),
                );
                strncat_safe(&mut msg, cast_slice(cval), 25);
                ffpmsg_slice(&msg);
                *status = NUM_OVERFLOW;
            }
            _ => {
                *status = BAD_C2I;
            }
        },
    }

    *status
}

/// convert null-terminated formatted string to an long long integer value
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `ival`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2jj(cval: &[c_char], ival: &mut LONGLONG, status: &mut c_int) -> c_int {
    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    let cval: &[u8] = cast_slice(cval);
    // A malformed card can leave arbitrary/unterminated bytes here; treat those
    // as an unparseable value (BAD_C2I) rather than panicking.
    let tmp_str = slice_to_str!(cval);
    let tmp_str = tmp_str.trim();
    let tmp = atoi::<LONGLONG>(tmp_str);

    match tmp {
        Ok(val) => {
            *ival = val;
        }
        Err(err) => match err.kind() {
            core::num::IntErrorKind::Empty => {
                *status = BAD_C2I;
            }
            core::num::IntErrorKind::InvalidDigit => {
                *status = BAD_C2I;
            }
            core::num::IntErrorKind::PosOverflow | core::num::IntErrorKind::NegOverflow => {
                let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
                strcpy_safe(
                    &mut msg,
                    cs!(c"Range Error in ffc2jj converting string to longlong int: "),
                );
                strncat_safe(&mut msg, cast_slice(cval), 25);
                ffpmsg_slice(&msg);
                *status = NUM_OVERFLOW;
            }
            _ => {
                *status = BAD_C2I;
            }
        },
    }

    *status
}

/// Convert null-terminated formatted string to an unsigned long long integer value
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `ival`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2ujj(cval: &[c_char], ival: &mut ULONGLONG, status: &mut c_int) -> c_int {
    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    let cval: &[u8] = cast_slice(cval);
    /* an unterminated value, or one that is not text, is not a number either */
    let tmp_str = CStr::from_bytes_until_nul(cval)
        .ok()
        .and_then(|s| s.to_str().ok())
        .unwrap_or("")
        .trim();
    let tmp = atoi::<ULONGLONG>(tmp_str);

    match tmp {
        Ok(val) => {
            *ival = val;
        }
        Err(err) => match err.kind() {
            core::num::IntErrorKind::Empty => {
                *status = BAD_C2I;
            }
            core::num::IntErrorKind::InvalidDigit => {
                *status = BAD_C2I;
            }
            core::num::IntErrorKind::PosOverflow | core::num::IntErrorKind::NegOverflow => {
                let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
                strcpy_safe(
                    &mut msg,
                    cs!(c"Range Error in ffc2ujj converting string to unsigned longlong int: "),
                );
                strncat_safe(&mut msg, cast_slice(cval), 25);
                ffpmsg_slice(&msg);
                *status = NUM_OVERFLOW;
            }
            _ => {
                *status = BAD_C2I;
            }
        },
    }

    *status
}

/// convert null-terminated formatted string to a logical value
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value: T or F
/// * `lval`   — (O) numerical value of the input string: 1 or 0
/// * `status` — (IO) error status
pub(crate) fn ffc2ll(cval: &[c_char], lval: &mut c_int, status: &mut c_int) -> c_int {
    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if cval[0] == bb(b'T') {
        *lval = 1;
    } else {
        *lval = 0; /* any character besides T is considered false */
    }

    *status
}

/// convert an input quoted string to an unquoted string by removing
/// the leading and trailing quote character.  Also, replace any
/// pairs of single quote characters with just a single quote
/// character (FITS used a pair of single quotes to represent
/// a literal quote character within the string).
///
/// # Parameters
///
/// * `instr`  — (I) null terminated quoted input string
/// * `outstr` — (O) null terminated output string without quotes
/// * `status` — (IO) error status
pub(crate) fn ffc2s(instr: &[c_char], outstr: &mut [c_char], status: &mut c_int) -> c_int {
    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    if instr[0] != bb(b'\'') {
        if instr[0] == 0 {
            outstr[0] = 0;
            *status = VALUE_UNDEFINED; /* null value string */
            return *status;
        } else {
            strcpy_safe(outstr, instr); /* no leading quote, so return input string */
            return *status;
        }
    }

    let len = strlen_safe(instr);
    let mut ii = 1;
    let mut jj: isize = 0;
    while ii < len {
        if instr[ii] == bb(b'\'') {
            /*  is this the closing quote?  */
            if instr[ii + 1] == bb(b'\'') {
                /* 2 successive quotes? */
                ii += 1; /* copy only one of the quotes */
            } else {
                break; /*  found the closing quote, so exit this loop  */
            }
        }
        outstr[jj as usize] = instr[ii]; /* copy the next character to the output */
        ii += 1;
        jj += 1;
    }

    outstr[jj as usize] = 0; /*  terminate the output string  */

    if ii == len {
        ffpmsg_str("This string value has no closing quote (ffc2s):");
        ffpmsg_slice(instr);
        *status = NO_QUOTE;
        return *status;
    }

    jj -= 1;
    while jj >= 0 {
        /* replace trailing blanks with nulls */
        if outstr[jj as usize] == bb(b' ') {
            outstr[jj as usize] = 0;
        } else {
            break;
        }
        jj -= 1;
    }

    *status
}

/// convert null-terminated formatted string to a float value
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `fval`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2rr(cval: &[c_char], fval: &mut f32, status: &mut c_int) -> c_int {
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut tval: [c_char; 73] = [0; 73];
    let decimalpt: c_char = 0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    // let locale = SystemLocale::default().unwrap();
    // let decimalpt = locale.decimal().as_bytes()[0] as c_char;

    strcpy_safe(&mut tval, cval);

    *fval = 0.0;

    if strchr_safe(cval, bb(b'D')).is_some() || decimalpt == bb(b',') {
        /* need to modify a temporary copy of the string before parsing it */
        if strlen_safe(cval) > 72 {
            strcpy_safe(&mut msg, cs!(c"Error: Invalid string to float in ffc2rr"));
            ffpmsg_slice(&msg);
            *status = BAD_C2F;
            return *status;
        }

        /*  The C language does not support a 'D'; replace with 'E' */
        tval.iter_mut().for_each(|x| {
            if *x == bb(b'D') {
                *x = bb(b'E');
            }
        });

        /* Rust parse is not locale aware */
        /*
        if decimalpt == bb(b',') {
            /* strtod expects a comma, not a period, as the decimal point */
            (&tval).iter_mut().filter(|x| *x == bb(b'.')).for_each(|x| {
                *x = bb(b',');
            });
        }
        */
    }

    let mut i = 0;
    for c in tval.iter() {
        if *c == bb(b' ') {
            i += 1;
            break;
        }
    }

    /* an unterminated value, or one that is not text, is not a number either */
    let tmp: Result<f32, ParseFloatError> = CStr::from_bytes_until_nul(cast_slice(&tval[i..]))
        .ok()
        .and_then(|s| s.to_str().ok())
        .unwrap_or("")
        .parse();

    match tmp {
        Ok(val) => {
            *fval = val;
        }
        Err(_err) => {
            strcpy_safe(
                &mut msg,
                cs!(c"Error in ffc2rr converting string to float: "),
            );
            strncat_safe(&mut msg, cval, 30);
            ffpmsg_slice(&msg);
            *status = BAD_C2F;
            return *status;
        }
    }

    let val_bytes = &[*fval]; // To keep alignment
    let sptr: &[c_short] = cast_slice(val_bytes);
    let mut si = 0;

    if BYTESWAPPED {
        si = 1; /* point to MSBs */
    }

    let iret = fnan(sptr[si]); /* if iret == 1, then the double value is a NaN */

    if iret == 1 {
        strcpy_safe(
            &mut msg,
            cs!(c"Error in ffc2rr converting string to float: "),
        );
        strncat_safe(&mut msg, cval, 30);
        ffpmsg_slice(&msg);
        *fval = 0.0;
        *status = NUM_OVERFLOW;
    }

    *status
}

/// convert null-terminated formatted string to a double value
///
/// # Parameters
///
/// * `cval`   — (I) string representation of the value
/// * `dval`   — (O) numerical value of the input string
/// * `status` — (IO) error status
pub(crate) fn ffc2dd(cval: &[c_char], dval: &mut f64, status: &mut c_int) -> c_int {
    let mut msg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut tval: [c_char; 73] = [0; 73];
    let decimalpt: c_char = 0;

    if *status > 0 {
        /* inherit input status value if > 0 */
        return *status;
    }

    // let locale = SystemLocale::default().unwrap();
    // let decimalpt = locale.decimal().as_bytes()[0] as c_char;

    strcpy_safe(&mut tval, cval);

    *dval = 0.0;

    if strchr_safe(cval, bb(b'D')).is_some() || decimalpt == bb(b',') {
        /* need to modify a temporary copy of the string before parsing it */
        if strlen_safe(cval) > 72 {
            strcpy_safe(&mut msg, cs!(c"Error: Invalid string to double in ffc2dd"));
            ffpmsg_slice(&msg);
            *status = BAD_C2D;
            return *status;
        }

        /*  The C language does not support a 'D'; replace with 'E' */
        tval.iter_mut().for_each(|x| {
            if *x == bb(b'D') {
                *x = bb(b'E');
            }
        });

        /* Rust parse is not locale aware */
        /*
        if decimalpt == bb(b',') {
            /* strtod expects a comma, not a period, as the decimal point */
            (&tval).iter_mut().filter(|x| *x == bb(b'.')).for_each(|x| {
                *x = bb(b',');
            });
        }
        */
    }

    let mut i = 0;
    for c in tval.iter() {
        if *c == bb(b' ') {
            i += 1;
            break;
        }
    }

    /* an unterminated value, or one that is not text, is not a number either */
    let tmp: Result<f64, ParseFloatError> = CStr::from_bytes_until_nul(cast_slice(&tval[i..]))
        .ok()
        .and_then(|s| s.to_str().ok())
        .unwrap_or("")
        .parse();

    match tmp {
        Ok(val) => {
            *dval = val;
        }
        Err(_err) => {
            strcpy_safe(
                &mut msg,
                cs!(c"Error in ffc2dd converting string to double: "),
            );
            strncat_safe(&mut msg, cval, 30);
            ffpmsg_slice(&msg);
            *status = BAD_C2D;
            return *status;
        }
    }

    let val_bytes = &[*dval];
    let sptr: &[c_short] = cast_slice(val_bytes);
    let mut si = 0;

    if BYTESWAPPED {
        si = 3; /* point to MSBs */
    }

    let iret = dnan(sptr[si]); /* if iret == 1, then the double value is a NaN */

    if iret == 1 {
        strcpy_safe(
            &mut msg,
            cs!(c"Error in ffc2dd converting string to double: "),
        );
        strncat_safe(&mut msg, cval, 30);
        ffpmsg_slice(&msg);
        *dval = 0.0;
        *status = NUM_OVERFLOW;
    }
    *status
}

/* ================================================================== */
/* A hack for nonunix machines, which lack strcasecmp and strncasecmp */
/* ================================================================== */
pub(crate) fn fits_strcasecmp(s1: &[c_char], s2: &[c_char]) -> c_int {
    let mut c2: c_char;
    let mut c1: c_char;

    let mut i = 0;
    loop {
        c1 = toupper(s1[i]);
        c2 = toupper(s2[i]);
        if c1 < c2 {
            return -1;
        }
        if c1 > c2 {
            return 1;
        }
        if c1 == 0 {
            return 0;
        }
        i += 1;
    }
}

pub(crate) fn fits_strncasecmp(s1: &[c_char], s2: &[c_char], n: usize) -> c_int {
    let mut c1;
    let mut c2;

    let mut i = 0;

    while i < n {
        c1 = c_int::from(toupper(s1[i]));
        c2 = c_int::from(toupper(s2[i]));
        if c1 < c2 {
            return -1;
        }
        if c1 > c2 {
            return 1;
        }
        if c1 == 0 {
            return 0;
        }

        i += 1;
    }
    0
}

/*
 * fits_recalloc - an allocator/reallocator in the style of calloc and realloc
 *
 * Allocates or reallocates storage upon request.  Newly allocated
 * storage is zeroed in the style of calloc.
 *
 * Cases handled are:
 *    ptr == 0 or old_num == 0 - use calloc to allocate new storage
 *    new_num = 0 - frees any storage if ptr is non-NULL
 *    new_num < old_num - uses realloc() to reduce storage allocated
 *    new_num > old_num - uses realloc() and sets newly allocated
 *                        storage to zero (old portion left unchanged)
 *
 * void *ptr - "old" pointer, or NULL to allocate new storage
 * size_t old_num - old number of records allocated
 * size_t new_num  - new number of records allocated
 * size_t size - size of record in bytes
 *
 * RETURNS: newly allocated storage
 *
 * */
/// Grow, shrink or free a zero-initialised array of `T`, as the C's realloc
/// idiom in the iterator/histogram code.
///
/// Generic over the element type rather than taking a byte size: the layout has
/// to carry `align_of::<T>()`. The previous version built every layout with
/// `Layout::array::<u8>(n * size)`, so the allocation was only ever 1-byte
/// aligned and taking a `&mut T` out of it was UB for any `T` needing more
/// (Miri rejected `&mut *iterCols.add(n)` for `iteratorCol`, which wants 8).
///
/// Returns null on allocation failure, freeing the old block, and returns null
/// for `new_num == 0` after freeing.
pub(crate) unsafe fn fits_recalloc<T>(ptr: *mut T, old_num: usize, new_num: usize) -> *mut T {
    unsafe {
        use alloc::alloc::{alloc_zeroed, dealloc, realloc};
        use core::alloc::Layout;
        use core::ptr;

        let old_layout = match Layout::array::<T>(old_num) {
            Ok(l) => l,
            Err(_) => return ptr::null_mut(),
        };

        if ptr.is_null() || old_num == 0 {
            /* Starting from nothing */
            if new_num == 0 {
                return ptr::null_mut();
            }
            let layout = match Layout::array::<T>(new_num) {
                Ok(l) => l,
                Err(_) => return ptr::null_mut(),
            };
            return alloc_zeroed(layout).cast::<T>();
        }

        if new_num == old_num {
            /* Same size, do nothing */
            return ptr;
        }

        if new_num == 0 {
            /* Freeing */
            dealloc(ptr.cast::<u8>(), old_layout);
            return ptr::null_mut();
        }

        let new_layout = match Layout::array::<T>(new_num) {
            Ok(l) => l,
            Err(_) => {
                dealloc(ptr.cast::<u8>(), old_layout);
                return ptr::null_mut();
            }
        };

        /* realloc keeps the alignment of old_layout, which is align_of::<T>() */
        let newptr = realloc(ptr.cast::<u8>(), old_layout, new_layout.size());
        if newptr.is_null() {
            dealloc(ptr.cast::<u8>(), old_layout);
            return ptr::null_mut();
        }

        if new_num > old_num {
            /* Zero the new portion of the array */
            ptr::write_bytes(
                newptr.add(old_layout.size()),
                0,
                new_layout.size() - old_layout.size(),
            );
        }

        newptr.cast::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::ffi::CString;
    use std::sync::{LazyLock, Mutex};

    // Serialize tests that touch the global ERROR_STACK to avoid cross-test interference on parallel runs.
    static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn test_ffdtyp_safe_classifies_keyword_values() {
        /* A character-string value starts with a single quote.  The transpiled
        code tested for a backslash, so quoted strings were classified 'I' and
        every caller then tried an integer conversion: find_keywd reported
        BAD_C2I for '#STRKEY', and ffc2x/ffc2r did the same for any string. */
        let cases: [(&CStr, u8); 10] = [
            (c"'hello'", b'C'),
            (c"'123'", b'C'),
            (c"'T'", b'C'),
            (c"T", b'L'),
            (c"F", b'L'),
            (c"(1.2, -3.4)", b'X'),
            (c"12.5", b'F'),
            (c"1E5", b'F'),
            (c"1D5", b'F'),
            (c"42", b'I'),
        ];
        for (cval, want) in cases {
            let mut dtype: c_char = 0;
            let mut status: c_int = 0;
            ffdtyp_safe(cs!(cval), &mut dtype, &mut status);
            assert_eq!(status, 0, "cval: {cval:?}");
            assert_eq!(dtype, bb(want), "cval: {cval:?}");
        }

        /* an empty value is undefined, not a datatype */
        let mut dtype: c_char = 0;
        let mut status: c_int = 0;
        ffdtyp_safe(cs!(c""), &mut dtype, &mut status);
        assert_eq!(status, VALUE_UNDEFINED);
    }

    #[test]
    fn test_ffmkky_safe() {
        let _g = TEST_LOCK.lock().unwrap();
        let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];
        let mut status: c_int = 0;

        ffmkky_safe(
            cs!(c"LONG_KEYWORD"),
            cs!(c"'Lorem ipsum dolor sit amet'"),
            None,
            &mut card,
            &mut status,
        );

        assert_eq!(status, 0);

        let expected = c"HIERARCH LONG_KEYWORD = 'Lorem ipsum dolor sit amet'";

        let card_str = CStr::from_bytes_until_nul(cast_slice(&card)).unwrap();
        assert_eq!(card_str.to_bytes_with_nul(), expected.to_bytes_with_nul());
    }

    /* The error stack's storage invariants.  These use a local ErrorStack, so
    they do not need the global TEST_LOCK. */

    #[test]
    fn test_stored_entry_always_fits_the_reader_buffer() {
        /* ffgmsg copies at most FLEN_ERRMSG - 1 bytes, so keeping the chunk
        size below FLEN_ERRMSG means no stored entry can ever be truncated on
        the way out.  Both are consts, so check it at compile time. */
        const { assert!(ERRMSG_CHUNK < FLEN_ERRMSG) };
    }

    #[test]
    fn test_error_stack_chunks_long_messages() {
        let mut st = ErrorStack::new();
        let msg: Vec<u8> = (0..190u32).map(|i| b'a' + (i % 26) as u8).collect();
        st.push_message(&msg);

        let mut out: Vec<Vec<u8>> = Vec::new();
        while let Some(m) = st.pop_message() {
            out.push(m.bytes().to_vec());
        }

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), ERRMSG_CHUNK);
        assert_eq!(out[1].len(), ERRMSG_CHUNK);
        assert_eq!(out[2].len(), 190 - 2 * ERRMSG_CHUNK);
        /* split, not lost or reordered */
        assert_eq!(out.concat(), msg);
    }

    #[test]
    fn test_error_stack_keeps_raw_bytes() {
        /* FITS diagnostics echo card bytes, which are frequently not valid
        UTF-8; and an embedded NUL must not shorten the stored entry, which
        is why the length is explicit rather than NUL-terminated. */
        let mut st = ErrorStack::new();
        let msg = b"card \xff\xfe with a NUL \x00 in it";
        st.push_message(msg);

        let m = st.pop_message().unwrap();
        assert_eq!(m.bytes(), msg);
        assert_eq!(m.len as usize, msg.len());
    }

    #[test]
    fn test_error_stack_evicts_oldest_when_full() {
        let mut st = ErrorStack::new();
        for i in 0..(ERRMSGSIZ + 5) {
            st.push_message(&[b'a' + i as u8]);
        }

        let mut got: Vec<u8> = Vec::new();
        while let Some(m) = st.pop_message() {
            got.push(m.bytes()[0]);
        }

        assert_eq!(got.len(), ERRMSGSIZ);
        /* the first five were dropped, so the oldest survivor is the sixth */
        assert_eq!(got[0], b'a' + 5);
    }

    #[test]
    fn test_ffxmsg_basic_operations() {
        unsafe {
            let _g = TEST_LOCK.lock().unwrap();
            // Clear any existing messages first
            ffxmsg(DEL_ALL, ptr::null_mut());

            // Test putting a message
            let test_msg = CString::new("Test error message").unwrap();
            ffxmsg(PUT_MESG, test_msg.as_ptr().cast_mut());

            // Test getting the message back
            let mut buffer: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
            ffxmsg(GET_MESG, buffer.as_mut_ptr());

            let result = CStr::from_ptr(buffer.as_ptr()).to_str().unwrap();
            assert_eq!(result, "Test error message");

            // Should be empty now
            ffxmsg(GET_MESG, buffer.as_mut_ptr());
            assert_eq!(buffer[0], 0); // Should be empty
        }
    }

    #[test]
    fn test_ffxmsg_marker_functionality() {
        unsafe {
            let _g = TEST_LOCK.lock().unwrap();
            // Clear any existing messages
            ffxmsg(DEL_ALL, ptr::null_mut());

            // Add some messages
            let msg1 = CString::new("Message 1").unwrap();
            let msg2 = CString::new("Message 2").unwrap();

            ffxmsg(PUT_MESG, msg1.as_ptr().cast_mut());
            ffxmsg(PUT_MARK, ptr::null_mut()); // Add marker
            ffxmsg(PUT_MESG, msg2.as_ptr().cast_mut());

            // Should get Message 1 first (ignoring marker)
            let mut buffer: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
            ffxmsg(GET_MESG, buffer.as_mut_ptr());
            let result = CStr::from_ptr(buffer.as_ptr()).to_str().unwrap();
            assert_eq!(result, "Message 1");

            // Should get Message 2 next
            ffxmsg(GET_MESG, buffer.as_mut_ptr());
            let result = CStr::from_ptr(buffer.as_ptr()).to_str().unwrap();
            assert_eq!(result, "Message 2");
        }
    }
}
