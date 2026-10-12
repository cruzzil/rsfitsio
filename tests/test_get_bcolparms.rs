//! `fits_get_bcolparms` reports a variable length column's datatype with its
//! `P` prefix, which with the terminating nul needs three characters.
mod common;

use crate::common::with_temp_file;
use bytemuck::cast_slice;
use libc::{c_char, c_int, c_long};
use rsfitsio::aliases::rust_api::*;
use rsfitsio::fitsio::{BINARY_TBL, fitsfile};
use std::ffi::{CStr, CString};

fn chars(s: &CStr) -> &[c_char] {
    cast_slice(s.to_bytes_with_nul())
}

/// Create a table with a fixed and a variable length column
fn create_table(filename: &str) -> Box<fitsfile> {
    let name = CString::new(filename).unwrap();
    let mut status: c_int = 0;
    let mut fptr: Option<Box<fitsfile>> = None;
    fits_create_file(&mut fptr, chars(&name), &mut status);
    let mut f = fptr.unwrap();

    let ttype = [Some(chars(c"FIXED")), Some(chars(c"VARIABLE"))];
    let tform = [chars(c"1E"), chars(c"PE(10)")];
    fits_create_tbl(
        &mut f,
        BINARY_TBL,
        0,
        2,
        &ttype,
        &tform,
        None,
        Some(chars(c"TEST")),
        &mut status,
    );
    assert_eq!(status, 0, "create table failed");
    f
}

/// The `dtype` reported for each column
fn dtypes(filename: &str) -> Vec<String> {
    let mut f = create_table(filename);
    let mut status: c_int = 0;

    let mut out = Vec::new();
    for colnum in 1..=2 {
        let mut dtype: [c_char; 3] = [0; 3];
        let mut repeat: c_long = 0;
        fits_get_bcolparms(
            &mut f,
            colnum,
            None,
            None,
            Some(&mut dtype),
            Some(&mut repeat),
            None,
            None,
            None,
            None,
            &mut status,
        );
        assert_eq!(status, 0, "get_bcolparms failed");
        let dtype = CStr::from_bytes_until_nul(cast_slice(&dtype)).unwrap();
        out.push(dtype.to_str().unwrap().to_string());
    }

    fits_close_file(f, &mut status);
    out
}

#[test]
fn test_get_bcolparms_variable_length_dtype() {
    with_temp_file(|filename| {
        assert_eq!(dtypes(filename), ["E", "PE"]);
    });
}

/// The C entry point, called with a buffer as large as a C caller would use
#[test]
#[allow(deprecated)]
fn test_ffgbcl_variable_length_dtype() {
    with_temp_file(|filename| {
        let mut f = create_table(filename);
        let mut status: c_int = 0;
        let mut dtype: [c_char; 71] = [0; 71];
        unsafe {
            rsfitsio::fitscore::ffgbcl(
                &raw mut *f,
                2,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                dtype.as_mut_ptr().cast(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                &raw mut status,
            );
        }
        assert_eq!(status, 0);
        assert_eq!(
            CStr::from_bytes_until_nul(cast_slice(&dtype)).unwrap(),
            c"PE"
        );
        fits_close_file(f, &mut status);
    });
}
