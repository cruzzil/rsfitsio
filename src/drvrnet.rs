//! What remains of the network drivers.
//!
//! CFITSIO's `drvrnet.c` implements `http://`, `ftp://` and `root://` access.
//! This port does not: only the HTTPS configuration hooks that the public API
//! exposes are kept, so that a program calling them still links.
#![warn(missing_docs)]

use std::sync::Mutex;

use crate::c_types::{c_int, c_uint};

static SHOW_FITS_DOWNLOAD_PROGRESS: Mutex<c_int> = Mutex::new(0);
static NET_TIMEOUT: Mutex<c_uint> = Mutex::new(360); /* in seconds */
static CURL_VERBOSE: Mutex<c_int> = Mutex::new(0);

pub(crate) fn fits_net_timeout(sec: c_int) -> c_int {
    let mut net_timeout = crate::lock_global(&NET_TIMEOUT);
    /* If sec is 0 or negative, treat this as a 'get' call. */
    if sec > 0 {
        *net_timeout = sec as c_uint;
    }

    *net_timeout as c_int
}

pub(crate) fn fits_dwnld_prog_bar(flag: c_int) {
    let mut download_progress = crate::lock_global(&SHOW_FITS_DOWNLOAD_PROGRESS);
    if flag == 0 {
        *download_progress = 0;
    } else {
        *download_progress = 1;
    }
}

pub(crate) fn https_set_verbose(flag: c_int) {
    let mut curl_verbose = crate::lock_global(&CURL_VERBOSE);
    if flag == 0 {
        *curl_verbose = 0;
    } else {
        *curl_verbose = 1;
    }
}
