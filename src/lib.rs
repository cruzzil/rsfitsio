#![allow(
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unused_variables,
    unused_assignments,
    clippy::missing_safety_doc,
    unreachable_code,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    // C declares its locals at the top of the function and drives loops with
    // an explicit counter; TRANSPILING.md keeps that layout so the Rust can be
    // read side by side with the original.
    clippy::needless_late_init,
    clippy::explicit_counter_loop,
    clippy::manual_range_contains,
    // eval_tab.rs mirrors the bison-generated token enum from eval_tab.h;
    // BOOLEAN/BITSTR/GTIFILTER/... must keep the grammar's spelling.
    clippy::upper_case_acronyms,
    // C integer widths are target-dependent -- c_long is 64-bit on Linux/macOS
    // and 32-bit on Windows -- so a cast or Into that clippy sees as a no-op
    // here is load-bearing elsewhere. Taking these two lints has already
    // broken the Windows build once. Verify with
    //   cargo check --target i686-unknown-linux-gnu --all-targets
    // before "simplifying" any conversion in this crate.
    clippy::unnecessary_cast,
    clippy::useless_conversion,
    // The deg<->rad and pi literals are carried over verbatim from the
    // CFITSIO C; keep them as written rather than swapping in core::f64
    // constants, which would perturb the transpiled arithmetic.
    clippy::approx_constant
)]
/*
#![warn(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::unnecessary_cast,
    clippy::fn_to_numeric_cast,
    clippy::fn_to_numeric_cast_with_truncation,
    clippy::char_lit_as_u8,
    clippy::ptr_as_ptr,
    clippy::cast,
)]
*/
// NOTE: no crate-wide `#![allow(deprecated)]`. Every `extern "C"` entry point is
// `#[deprecated]` so that internal callers still going through the C ABI instead of
// the `_safe` form show up as warnings (TODO.md). Allow it locally where a wrapper
// must call its deprecated sibling, never crate-wide.
#![deny(clippy::std_instead_of_core, clippy::std_instead_of_alloc)]
// #![deny(clippy::unnecessary_cast)]

//! A Rust port of [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/), the
//! reference library for reading and writing FITS files.
//!
//! # Layout
//!
//! Every public CFITSIO routine `ffxyz` exists in three forms:
//!
//! * `ffxyz` — a `#[no_mangle] pub unsafe extern "C" fn` C ABI entry point,
//!   which null-checks its raw pointers and delegates. These are
//!   `#[deprecated]` so that internal callers reach for the safe form instead.
//! * `ffxyz_safe` — the safe implementation, taking references, slices and
//!   `Option`s. An opened file is an `Option<Box<fitsfile>>`.
//! * an alias under its `fits_*` spelling in [`aliases`].
//!
//! [`aliases::c_api`] and [`aliases::rust_api`] mirror CFITSIO's `fitsio.h` and
//! `longnam.h` exactly, and are the intended entry point; the implementation
//! modules are exposed mainly so that the port can be read against the original
//! C. [`fitsio`] holds the datatypes, symbolic constants and error status
//! codes.
//!
//! # Error handling
//!
//! Routines return a `c_int` status code and also write it through their
//! `status` parameter; `0` means success. A non-zero status on input is
//! generally passed straight back out, so a sequence of calls can be made
//! without checking in between. The symbolic names are in [`fitsio`], and
//! [`aliases::rust_api::fits_get_errstatus`] turns one into a message.
//!
//! # Compression
//!
//! The PLIO, Rice and Hcompress codecs are the separate `pliocomp`, `ricecomp`
//! and `hcompress` crates; GZIP uses `libz-rs-sys` and BZIP2 uses
//! `libbz2-rs-sys` behind the default `bzip2` feature. Quantization
//! ([`quantize`]) and the tiled-image driver ([`imcompress`]) are in this
//! crate.
#![warn(missing_docs)]

extern crate alloc;

pub mod c_types;
pub mod helpers;

pub mod aliases;
pub mod buffers;
pub mod checksum;
pub mod edithdu;
pub mod fitsio;
pub mod fitsio2;
pub mod scalnull;

//#[cfg(feature = "bzip2")]
//mod bzip2;

pub mod cfileio;
pub mod drvrfile;
pub mod drvrgsiftp;
pub mod drvrmem;
pub mod drvrnet;

#[cfg(all(feature = "shared_mem", not(target_os = "windows")))]
pub mod drvrsmem;

pub mod editcol;
pub mod eval_defs;
pub mod eval_f;
pub mod eval_l;
pub mod eval_tab;
pub mod eval_y;
pub mod fits_hcompress;
pub mod fits_hdecompress;
pub mod fitscore;
pub mod getcol;
pub mod getcolb;
pub mod getcold;
pub mod getcole;
pub mod getcoli;
pub mod getcolj;
pub mod getcolk;
pub mod getcoll;
pub mod getcols;
pub mod getcolsb;
pub mod getcolui;
pub mod getcoluj;
pub mod getcoluk;
pub mod getkey;
pub mod group;
pub mod grparser;
pub mod histo;
pub mod imcompress;
pub mod iraffits;
pub mod modkey;
pub mod pliocomp;
pub mod putcol;
pub mod putcolb;
pub mod putcold;
pub mod putcole;
pub mod putcoli;
pub mod putcolj;
pub mod putcolk;
pub mod putcoll;
pub mod putcols;
pub mod putcolsb;
pub mod putcolu;
pub mod putcolui;
pub mod putcoluj;
pub mod putcoluk;
pub mod putkey;
pub mod quantize;
pub mod region;
pub mod relibc;
pub mod simplerng;
pub mod swapproc;
pub mod wcssub;
pub mod wcsutil;
pub mod wrappers;
pub mod zcompress;
pub mod zuncompress;

use alloc::ffi::CString;
use core::{ffi::CStr, marker::PhantomData, str::FromStr};
use std::sync::{Mutex, MutexGuard};

use bytemuck::cast_slice;
use fitsio::{
    FLEN_VALUE, LONGLONG, TBIT, TBYTE, TCOMPLEX, TDBLCOMPLEX, TDOUBLE, TFLOAT, TINT, TLOGICAL,
    TLONG, TLONGLONG, TSBYTE, TSHORT, TSTRING, TUINT, TULONG, TULONGLONG, TUSHORT, ULONGLONG,
};

use crate::c_types::*;

/// Serializes the critical sections that CFITSIO guards with `FFLOCK`.
pub(crate) static MUTEX_LOCK: Mutex<bool> = Mutex::new(false);

/// Enter a critical section, as CFITSIO's `FFLOCK` macro does.
///
/// Recovers from a poisoned mutex rather than panicking; see the comment in the
/// body for why.
pub(crate) fn FFLOCK<'a>() -> MutexGuard<'a, bool> {
    // Recover from a poisoned mutex rather than panicking. The guarded value is
    // just a sentinel bool used to serialize critical sections, so it isn't
    // meaningfully corrupted if a thread (e.g. a panicking unit test) unwinds
    // while holding the lock. Without this, one panicking test poisons the
    // process-wide lock and cascades PoisonError into every later FFLOCK caller.
    MUTEX_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Lock one of the process-wide tables that stand in for CFITSIO's globals
/// (open files, memory files, the error stack, ...).
///
/// Recovers from a poisoned mutex rather than panicking, as [`FFLOCK`] does.
/// The C has no notion of poisoning: a panic while one file holds a table
/// leaves at worst that file's entry inconsistent, but propagating the
/// PoisonError would make every later operation on every file in the process
/// panic as well.
pub(crate) fn lock_global<T: ?Sized>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Leave a critical section entered with [`FFLOCK`], as CFITSIO's `FFUNLOCK`
/// macro does.
pub(crate) fn FFUNLOCK(p: MutexGuard<'_, bool>) {
    drop(p);
}

/// Hands out a raw mutable pointer to `Self`, for the `extern "C"` entry
/// points that must return one.
pub trait ToRaw {
    /// The raw mutable pointer to `self`.
    fn as_raw_mut(&mut self) -> *mut Self;
}

/// Flattens an optional buffer to the raw pointer the C API expects, mapping
/// `None` to null.
pub trait AsMutPtr<T> {
    /// The raw pointer to the first element, or null if there is no buffer.
    fn as_mut_ptr(&self) -> *mut T;
}

impl<T> AsMutPtr<T> for Option<&mut [T]> {
    fn as_mut_ptr(&self) -> *mut T {
        match self {
            Some(v) => v.as_ptr().cast_mut(), // UNSAFE
            None => core::ptr::null_mut(),
        }
    }
}

/// Reinterprets a byte literal as a `c_char`, whose signedness is
/// target-dependent.
#[inline(always)]
pub fn bb(n: u8) -> c_char {
    n as c_char
}

/// Formats into a `[c_char]` buffer, truncating at `$len - 1` and
/// nul-terminating, as C's `snprintf` does.
///
/// Evaluates to the number of characters written, excluding the terminator.
#[macro_export]
macro_rules! int_snprintf {
    ($dst:expr, $len:expr, $($arg:tt)*) => {
        {
            let s = format!($($arg)*);
            let s_bytes = s.as_bytes();
            let mut s_len = s_bytes.len();

            s_len = core::cmp::min($len-1, s_len);

            let w = cast_slice_mut::<c_char, u8>(&mut $dst[..s_len]);
            w.copy_from_slice(&s_bytes[..s_len]);
            $dst[s_len] = 0; // null-terminate

            s_len as isize
        }
    };
}

/// Borrows a `[c_char]` buffer as a string for formatting, stopping at the
/// first nul.
///
/// FITS buffers can hold arbitrary bytes and may have no terminator at all, so
/// invalid UTF-8 is replaced rather than causing a panic.
#[macro_export]
macro_rules! slice_to_str {
    ($e:expr) => {
        // FITS buffers can hold arbitrary bytes and may reach here without a
        // terminator (a malformed header is exactly what fitsverify reports
        // on), so fall back rather than panicking, as the C's printf would.
        {
            let __b: &[u8] = cast_slice($e);
            let __n = __b.iter().position(|&c| c == 0).unwrap_or(__b.len());
            ::alloc::string::String::from_utf8_lossy(&__b[..__n])
        }
    };
}

/// Borrows a `&CStr` literal as the `&[c_char]` the safe routines take,
/// including its nul terminator.
#[macro_export]
macro_rules! cs {
    ($e: expr) => {
        cast_slice::<u8, c_char>($e.to_bytes_with_nul())
    };
}

/// Rebinds a `*const c_char` parameter as an `Option<&[c_char]>`, mapping null
/// to `None`.
///
/// # Safety
///
/// The pointer, if non-null, must point at a nul-terminated string that stays
/// valid for the lifetime of the binding.
#[macro_export]
macro_rules! nullable_slice_cstr {
    ($e: ident) => {
        let $e: Option<&[c_char]> = match $e.is_null() {
            true => None,
            false => Some(cast_slice(CStr::from_ptr($e).to_bytes_with_nul())),
        };
    };
}

/// Rebinds a `*mut c_char` parameter as an `Option<&mut [c_char]>`, mapping
/// null to `None`.
///
/// # Safety
///
/// The pointer, if non-null, must point at a nul-terminated string that stays
/// valid and uniquely borrowed for the lifetime of the binding.
#[macro_export]
macro_rules! nullable_slice_cstr_mut {
    ($e: ident) => {
        let mut $e: Option<&mut [c_char]> = match $e.is_null() {
            true => None,
            false => {
                let _c = CStr::from_ptr($e).to_bytes_with_nul();
                let _l = _c.len();

                Some(slice::from_raw_parts_mut($e, _l))
            }
        };
    };
}

/// Rebinds a non-null `*const c_char` parameter as a `&[c_char]`, including
/// its nul terminator.
///
/// # Safety
///
/// The pointer must be non-null and point at a nul-terminated string that stays
/// valid for the lifetime of the binding.
#[macro_export]
macro_rules! raw_to_slice {
    ($e: ident) => {
        let $e: &[c_char] = cast_slice(CStr::from_ptr($e).to_bytes_with_nul());
    };
}

/// The borrowed (ttype, tform, tunit) column-keyword slices produced by
/// [`TKeywords::tkeywords_to_vecs`]. tunit is optional because the C API
/// allows a NULL tunit array.
pub(crate) type TKeywordVecs<'a> = (
    Vec<Option<&'a [c_char]>>,
    Vec<&'a [c_char]>,
    Option<Vec<Option<&'a [c_char]>>>,
);

/// The raw `char **` column-keyword arrays a table-creation entry point
/// receives, held together so they can be converted as a unit.
pub(crate) struct TKeywords<'a> {
    /// Number of columns in the table.
    tfields: c_int,
    /// Name of each column.
    ttype: *const *const c_char,
    /// Value of the `TFORMn` keyword for each column.
    tform: *const *const c_char,
    /// Value of the `TUNITn` keyword for each column.
    tunit: *const *const c_char,
    marker: PhantomData<&'a ()>,
}

impl<'a> TKeywords<'a> {
    /// Wraps the raw column-keyword arrays.
    ///
    /// # Parameters
    ///
    /// * `tfields` — (I) number of columns in the table
    /// * `ttype`   — (I) name of each column
    /// * `tform`   — (I) value of the `TFORMn` keyword for each column
    /// * `tunit`   — (I) value of the `TUNITn` keyword for each column, or null
    pub fn new(
        tfields: c_int,
        ttype: *const *const c_char,
        tform: *const *const c_char,
        tunit: *const *const c_char,
    ) -> Self {
        TKeywords {
            tfields,
            ttype,
            tform,
            tunit,
            marker: PhantomData,
        }
    }

    /// Converts the raw arrays into borrowed slices.
    ///
    /// Returns empty vectors if `tfields` is 0 or if `ttype` or `tform` is
    /// null, and `None` for `tunit` if that array is null.
    ///
    /// # Safety
    ///
    /// Each non-null pointer must point at `tfields` nul-terminated strings
    /// that outlive `'a`.
    pub unsafe fn tkeywords_to_vecs(&'a self) -> TKeywordVecs<'a> {
        unsafe {
            // Handle case where tfields is 0 or pointers are null
            if self.tfields == 0 || self.ttype.is_null() || self.tform.is_null() {
                return (Vec::new(), Vec::new(), None);
            }

            // Convert ttype to rust types
            let ttype = core::slice::from_raw_parts(self.ttype, self.tfields as usize);
            let mut v_ttype = Vec::new();

            for item in ttype {
                let ttype_item = if item.is_null() {
                    None
                } else {
                    Some(cast_slice(CStr::from_ptr(*item).to_bytes_with_nul()))
                };
                v_ttype.push(ttype_item);
            }

            // Convert tform to rust types
            let tform = core::slice::from_raw_parts(self.tform, self.tfields as usize);
            let mut v_tform = Vec::new();

            for item in tform {
                let tform_item = cast_slice(CStr::from_ptr(*item).to_bytes_with_nul());
                v_tform.push(tform_item);
            }

            // Convert tunit to rust types
            let mut v_tunit = Vec::new();
            let out_tunit = if self.tunit.is_null() {
                None
            } else {
                let tunit = core::slice::from_raw_parts(self.tunit, self.tfields as usize);

                for item in tunit {
                    let tunit_item = if item.is_null() {
                        None
                    } else {
                        Some(cast_slice(CStr::from_ptr(*item).to_bytes_with_nul()))
                    };
                    v_tunit.push(tunit_item);
                }
                Some(v_tunit)
            };

            (v_ttype, v_tform, out_tunit)
        }
    }
}

/// Number of pixels in an image section running from `blc` to `trc` in steps
/// of `inc`, along every axis.
pub(crate) fn calculate_subsection_length(blc: &[c_long], trc: &[c_long], inc: &[c_long]) -> usize {
    assert!(blc.len() == trc.len() && blc.len() == inc.len());

    let len = blc.len();
    let mut acc: usize = 1;
    for ii in 0..len {
        if blc[ii] < trc[ii] {
            acc *= ((trc[ii] - blc[ii]) / inc[ii] + 1) as usize; // WARNING: This also could be wrong
        } else {
            acc *= ((blc[ii] - trc[ii]) / inc[ii] + 1) as usize;
        }
    }
    acc
}

/// Number of pixels in an image section running from `blc` to `trc` with a
/// step of one along every axis.
pub(crate) fn calculate_subsection_length_unit(blc: &[c_long], trc: &[c_long]) -> usize {
    assert!(blc.len() == trc.len());

    let len = blc.len();
    let mut acc: usize = 1;
    for ii in 0..len {
        if blc[ii] < trc[ii] {
            acc *= ((trc[ii] - blc[ii]) + 1) as usize; // WARNING: This also could be wrong
        } else {
            acc *= ((blc[ii] - trc[ii]) + 1) as usize;
        }
    }
    acc
}

/// Borrows each `Vec` in a slice as a shared slice.
pub(crate) fn vecs_to_slices<T>(vecs: &[Vec<T>]) -> Vec<&[T]> {
    vecs.iter().map(Vec::as_slice).collect()
}

/// Borrows each `Vec` in a slice as a mutable slice.
pub(crate) fn vecs_to_slices_mut<T>(vecs: &mut [Vec<T>]) -> Vec<&mut [T]> {
    vecs.iter_mut().map(Vec::as_mut_slice).collect()
}

#[derive(Debug, PartialEq, Clone, Copy)]
/// How a read routine should report undefined pixels.
pub enum NullCheckType {
    /// Do no null checking.
    None = 0,
    /// Set undefined pixels to the caller's null value.
    SetPixel = 1,
    /// Set the corresponding entry of a null array to 1 for undefined pixels.
    SetNullArray = 2,
}

/// A null value of any of the datatypes the API accepts, as read from the
/// caller's `void *`.
///
/// The C passes the null value as an untyped pointer alongside a datatype code;
/// this carries the two together so the value cannot be read at the wrong type.
#[derive(Debug, PartialEq, Clone)]
pub enum NullValue {
    /// A `float` null value.
    Float(f32),
    /// A `double` null value.
    Double(f64),
    /// A `long` null value.
    Long(c_long),
    /// An `unsigned long` null value.
    ULong(c_ulong),
    /// A signed 64-bit integer null value.
    LONGLONG(LONGLONG),
    /// An unsigned 64-bit integer null value.
    ULONGLONG(ULONGLONG),
    /// An `int` null value.
    Int(c_int),
    /// An `unsigned int` null value.
    UInt(c_uint),
    /// A `short` null value.
    Short(c_short),
    /// An `unsigned short` null value.
    UShort(c_ushort),
    /// A signed byte null value.
    Byte(i8),
    /// An unsigned byte null value.
    UByte(c_uchar),
    /// A logical null value.
    Logical(c_char),
    /// A string null value.
    String(CString),
}

impl NullValue {
    /// The null value widened to an `f64`, for the comparisons the scaling
    /// code does in floating point.
    ///
    /// Returns `0.0` for [`NullValue::String`], which has no numeric value.
    pub fn get_value_as_f64(&self) -> f64 {
        match self {
            NullValue::Float(v) => f64::from(*v),
            NullValue::Double(v) => *v,
            NullValue::Long(v) => *v as f64,
            NullValue::ULong(v) => *v as f64,
            NullValue::LONGLONG(v) => *v as f64,
            NullValue::ULONGLONG(v) => *v as f64,
            NullValue::Int(v) => f64::from(*v),
            NullValue::UInt(v) => f64::from(*v),
            NullValue::Short(v) => f64::from(*v),
            NullValue::UShort(v) => f64::from(*v),
            NullValue::Byte(v) => f64::from(*v),
            NullValue::UByte(v) => f64::from(*v),
            NullValue::Logical(v) => f64::from(*v),
            _ => 0.0,
        }
    }

    /// Reads a null value of the given datatype out of a caller-supplied
    /// pointer.
    ///
    /// Returns `None` if `value` is null or `datatype` is not a recognized
    /// code; the caller decides which of those is an error.
    ///
    /// # Safety
    ///
    /// `value`, if non-null, must point at a readable value of the type
    /// `datatype` names. It is read unaligned, since the C commonly points it
    /// into a byte buffer.
    pub fn from_raw_ptr(datatype: c_int, value: *const c_void) -> Option<Self> {
        if value.is_null() {
            return None;
        }

        /* `value` is a caller-supplied void*, as in the C, and carries no
        alignment guarantee for the target type -- it commonly points into
        a byte buffer. Read unaligned rather than dereferencing. */
        match datatype {
            TFLOAT => Some(NullValue::Float(unsafe {
                value.cast::<f32>().read_unaligned()
            })),
            TDOUBLE => Some(NullValue::Double(unsafe {
                value.cast::<f64>().read_unaligned()
            })),
            TLONG => Some(NullValue::Long(unsafe {
                value.cast::<c_long>().read_unaligned()
            })),
            TULONG => Some(NullValue::ULong(unsafe {
                value.cast::<c_ulong>().read_unaligned()
            })),
            TLONGLONG => Some(NullValue::LONGLONG(unsafe {
                value.cast::<LONGLONG>().read_unaligned()
            })),
            TULONGLONG => Some(NullValue::ULONGLONG(unsafe {
                value.cast::<ULONGLONG>().read_unaligned()
            })),
            TINT => Some(NullValue::Int(unsafe {
                value.cast::<c_int>().read_unaligned()
            })),
            TUINT => Some(NullValue::UInt(unsafe {
                value.cast::<c_uint>().read_unaligned()
            })),
            TSHORT => Some(NullValue::Short(unsafe {
                value.cast::<c_short>().read_unaligned()
            })),
            TUSHORT => Some(NullValue::UShort(unsafe {
                value.cast::<c_ushort>().read_unaligned()
            })),
            TBYTE => Some(NullValue::UByte(unsafe {
                value.cast::<c_uchar>().read_unaligned()
            })),
            TSBYTE => Some(NullValue::Byte(unsafe {
                value.cast::<i8>().read_unaligned()
            })),
            TLOGICAL => Some(NullValue::Logical(unsafe {
                value.cast::<c_char>().read_unaligned()
            })),
            TSTRING => {
                let cstr = unsafe { CStr::from_ptr(value.cast::<c_char>()) };
                Some(NullValue::String(cstr.to_owned()))
            }
            _ => None, // Don't panic here, will be handled by the caller
        }
    }
}

/// A borrowed keyword value of any of the datatypes the API accepts.
///
/// The shared counterpart of [`KeywordDatatypeMut`], used by the routines that
/// write a keyword whose type is chosen at run time.
#[derive(Debug, PartialEq)]
pub enum KeywordDatatype<'a> {
    /// A keyword value that is an unsigned byte.
    TBYTE(&'a c_uchar),
    /// A keyword value that is a signed byte.
    TSBYTE(&'a c_schar),
    /// A keyword value that is a `short`.
    TSHORT(&'a c_short),
    /// A keyword value that is an `unsigned short`.
    TUSHORT(&'a c_ushort),
    /// A keyword value that is an `int`.
    TINT(&'a c_int),
    /// A keyword value that is an `unsigned int`.
    TUINT(&'a c_uint),
    /// A keyword value that is a `long`.
    TLONG(&'a c_long),
    /// A keyword value that is an `unsigned long`.
    TULONG(&'a c_ulong),
    /// A keyword value that is a `float`.
    TFLOAT(&'a f32),
    /// A keyword value that is a `double`.
    TDOUBLE(&'a f64),
    /// A keyword value that is a string.
    TSTRING(&'a [c_char]),
    /// A keyword value that is a logical value.
    TLOGICAL(&'a c_int),
    /// A keyword value that is a single-precision complex value.
    TCOMPLEX(&'a [f32; 2]),
    /// A keyword value that is a double-precision complex value.
    TDBLCOMPLEX(&'a [f64; 2]),
    /// A keyword value that is an unsigned 64-bit integer.
    TULONGLONG(&'a ULONGLONG),
    /// A keyword value that is a signed 64-bit integer.
    TLONGLONG(&'a LONGLONG),
    /// The datatype code was not recognized; it is carried through so the
    /// caller can report it.
    INVALID(c_int),
}

impl KeywordDatatype<'_> {
    /// Borrows a keyword value of the given datatype from a caller-supplied
    /// pointer, yielding [`KeywordDatatype::INVALID`] for an unrecognized code.
    ///
    /// # Safety
    ///
    /// `value` must be non-null and point at a readable, correctly aligned
    /// value of the type `datatype` names, living at least as long as `'_`.
    pub fn from_datatype(datatype: c_int, value: *const c_void) -> Self {
        match datatype {
            TBYTE => KeywordDatatype::TBYTE(unsafe { &*value.cast::<c_uchar>() }),
            TSBYTE => KeywordDatatype::TSBYTE(unsafe { &*value.cast::<c_schar>() }),
            TSHORT => KeywordDatatype::TSHORT(unsafe { &*value.cast::<c_short>() }),
            TUSHORT => KeywordDatatype::TUSHORT(unsafe { &*value.cast::<c_ushort>() }),
            TINT => KeywordDatatype::TINT(unsafe { &*value.cast::<c_int>() }),
            TUINT => KeywordDatatype::TUINT(unsafe { &*value.cast::<c_uint>() }),
            TLONG => KeywordDatatype::TLONG(unsafe { &*value.cast::<c_long>() }),
            TULONG => KeywordDatatype::TULONG(unsafe { &*value.cast::<c_ulong>() }),
            TFLOAT => KeywordDatatype::TFLOAT(unsafe { &*value.cast::<f32>() }),
            TDOUBLE => KeywordDatatype::TDOUBLE(unsafe { &*value.cast::<f64>() }),
            TSTRING => KeywordDatatype::TSTRING(unsafe {
                cast_slice(CStr::from_ptr(value.cast::<c_char>()).to_bytes_with_nul())
            }),
            TLOGICAL => KeywordDatatype::TLOGICAL(unsafe { &*value.cast::<c_int>() }),
            TCOMPLEX => KeywordDatatype::TCOMPLEX(unsafe { &*value.cast::<[f32; 2]>() }),
            TDBLCOMPLEX => KeywordDatatype::TDBLCOMPLEX(unsafe { &*value.cast::<[f64; 2]>() }),
            TULONGLONG => KeywordDatatype::TULONGLONG(unsafe { &*value.cast::<ULONGLONG>() }),
            TLONGLONG => KeywordDatatype::TLONGLONG(unsafe { &*value.cast::<LONGLONG>() }),
            _ => KeywordDatatype::INVALID(datatype),
        }
    }

    /// The datatype code this variant corresponds to, or the code that was not
    /// recognized for [`KeywordDatatype::INVALID`].
    pub fn to_datatype_code(&self) -> c_int {
        match self {
            KeywordDatatype::TBYTE(_) => TBYTE,
            KeywordDatatype::TSBYTE(_) => TSBYTE,
            KeywordDatatype::TSHORT(_) => TSHORT,
            KeywordDatatype::TUSHORT(_) => TUSHORT,
            KeywordDatatype::TINT(_) => TINT,
            KeywordDatatype::TUINT(_) => TUINT,
            KeywordDatatype::TLONG(_) => TLONG,
            KeywordDatatype::TULONG(_) => TULONG,
            KeywordDatatype::TFLOAT(_) => TFLOAT,
            KeywordDatatype::TDOUBLE(_) => TDOUBLE,
            KeywordDatatype::TSTRING(_) => TSTRING,
            KeywordDatatype::TLOGICAL(_) => TLOGICAL,
            KeywordDatatype::TCOMPLEX(_) => TCOMPLEX,
            KeywordDatatype::TDBLCOMPLEX(_) => TDBLCOMPLEX,
            KeywordDatatype::TULONGLONG(_) => TULONGLONG,
            KeywordDatatype::TLONGLONG(_) => TLONGLONG,
            KeywordDatatype::INVALID(x) => *x,
        }
    }
}

/// A mutably borrowed keyword value of any of the datatypes the API accepts.
///
/// The mutable counterpart of [`KeywordDatatype`], used by the routines that
/// read a keyword into a caller-supplied location whose type is chosen at run
/// time.
#[derive(Debug, PartialEq)]
pub enum KeywordDatatypeMut<'a> {
    /// A keyword value that is an unsigned byte.
    TBYTE(&'a mut c_uchar),
    /// A keyword value that is a signed byte.
    TSBYTE(&'a mut c_char),
    /// A keyword value that is a `short`.
    TSHORT(&'a mut c_short),
    /// A keyword value that is an `unsigned short`.
    TUSHORT(&'a mut c_ushort),
    /// A keyword value that is an `int`.
    TINT(&'a mut c_int),
    /// A keyword value that is an `unsigned int`.
    TUINT(&'a mut c_uint),
    /// A keyword value that is a `long`.
    TLONG(&'a mut c_long),
    /// A keyword value that is an `unsigned long`.
    TULONG(&'a mut c_ulong),
    /// A keyword value that is a `float`.
    TFLOAT(&'a mut f32),
    /// A keyword value that is a `double`.
    TDOUBLE(&'a mut f64),
    /// A keyword value that is a string.
    TSTRING(&'a mut [c_char; FLEN_VALUE]),
    /// A keyword value that is a logical value.
    TLOGICAL(&'a mut c_int),
    /// A keyword value that is a single-precision complex value.
    TCOMPLEX(&'a mut [f32; 2]),
    /// A keyword value that is a double-precision complex value.
    TDBLCOMPLEX(&'a mut [f64; 2]),
    /// A keyword value that is an unsigned 64-bit integer.
    TULONGLONG(&'a mut ULONGLONG),
    /// A keyword value that is a signed 64-bit integer.
    TLONGLONG(&'a mut LONGLONG),
    /// The datatype code was not recognized; it is carried through so the
    /// caller can report it.
    INVALID(c_int),
}

impl KeywordDatatypeMut<'_> {
    /// Mutably borrows a keyword value of the given datatype from a
    /// caller-supplied pointer, yielding [`KeywordDatatypeMut::INVALID`] for an
    /// unrecognized code.
    ///
    /// # Safety
    ///
    /// `value` must be non-null and point at a writable, correctly aligned,
    /// uniquely borrowed value of the type `datatype` names, living at least as
    /// long as `'_`. For [`TSTRING`] it must have room for [`FLEN_VALUE`]
    /// characters.
    pub fn from_datatype(datatype: c_int, value: *mut c_void) -> Self {
        match datatype {
            TBYTE => KeywordDatatypeMut::TBYTE(unsafe { &mut *value.cast::<c_uchar>() }),
            TSBYTE => KeywordDatatypeMut::TSBYTE(unsafe { &mut *value.cast::<c_char>() }),
            TSHORT => KeywordDatatypeMut::TSHORT(unsafe { &mut *value.cast::<c_short>() }),
            TUSHORT => KeywordDatatypeMut::TUSHORT(unsafe { &mut *value.cast::<c_ushort>() }),
            TINT => KeywordDatatypeMut::TINT(unsafe { &mut *value.cast::<c_int>() }),
            TUINT => KeywordDatatypeMut::TUINT(unsafe { &mut *value.cast::<c_uint>() }),
            TLONG => KeywordDatatypeMut::TLONG(unsafe { &mut *value.cast::<c_long>() }),
            TULONG => KeywordDatatypeMut::TULONG(unsafe { &mut *value.cast::<c_ulong>() }),
            TFLOAT => KeywordDatatypeMut::TFLOAT(unsafe { &mut *value.cast::<f32>() }),
            TDOUBLE => KeywordDatatypeMut::TDOUBLE(unsafe { &mut *value.cast::<f64>() }),
            TSTRING => {
                KeywordDatatypeMut::TSTRING(unsafe { &mut *value.cast::<[c_char; FLEN_VALUE]>() })
            }
            TLOGICAL => KeywordDatatypeMut::TLOGICAL(unsafe { &mut *value.cast::<c_int>() }),
            TCOMPLEX => KeywordDatatypeMut::TCOMPLEX(unsafe { &mut *value.cast::<[f32; 2]>() }),
            TDBLCOMPLEX => {
                KeywordDatatypeMut::TDBLCOMPLEX(unsafe { &mut *value.cast::<[f64; 2]>() })
            }
            TULONGLONG => {
                KeywordDatatypeMut::TULONGLONG(unsafe { &mut *value.cast::<ULONGLONG>() })
            }
            TLONGLONG => KeywordDatatypeMut::TLONGLONG(unsafe { &mut *value.cast::<LONGLONG>() }),
            _ => KeywordDatatypeMut::INVALID(datatype),
        }
    }

    /// The datatype code this variant corresponds to, or the code that was not
    /// recognized for [`KeywordDatatypeMut::INVALID`].
    pub fn to_datatype_code(&self) -> c_int {
        match self {
            KeywordDatatypeMut::TBYTE(_) => TBYTE,
            KeywordDatatypeMut::TSBYTE(_) => TSBYTE,
            KeywordDatatypeMut::TSHORT(_) => TSHORT,
            KeywordDatatypeMut::TUSHORT(_) => TUSHORT,
            KeywordDatatypeMut::TINT(_) => TINT,
            KeywordDatatypeMut::TUINT(_) => TUINT,
            KeywordDatatypeMut::TLONG(_) => TLONG,
            KeywordDatatypeMut::TULONG(_) => TULONG,
            KeywordDatatypeMut::TFLOAT(_) => TFLOAT,
            KeywordDatatypeMut::TDOUBLE(_) => TDOUBLE,
            KeywordDatatypeMut::TSTRING(_) => TSTRING,
            KeywordDatatypeMut::TLOGICAL(_) => TLOGICAL,
            KeywordDatatypeMut::TCOMPLEX(_) => TCOMPLEX,
            KeywordDatatypeMut::TDBLCOMPLEX(_) => TDBLCOMPLEX,
            KeywordDatatypeMut::TULONGLONG(_) => TULONGLONG,
            KeywordDatatypeMut::TLONGLONG(_) => TLONGLONG,
            KeywordDatatypeMut::INVALID(x) => *x,
        }
    }
}

/// Width in bytes of one element of a datatype, or `None` if the code is not
/// recognized. `TSTRING` reports the width of a pointer.
pub(crate) fn bytes_per_datatype(datatype: c_int) -> Option<usize> {
    match datatype {
        TBIT => Some(1),
        TBYTE => Some(1),
        TLOGICAL => Some(1),
        TSBYTE => Some(1),
        TUSHORT => Some(2),
        TSHORT => Some(2),
        TUINT => Some(4),
        TINT => Some(4),
        TULONG => Some(core::mem::size_of::<c_ulong>()),
        TLONG => Some(core::mem::size_of::<c_long>()),
        TULONGLONG => Some(8),
        TLONGLONG => Some(8),
        TFLOAT => Some(4),
        TDOUBLE => Some(8),
        TCOMPLEX => Some(8),
        TDBLCOMPLEX => Some(16),
        TSTRING => Some(core::mem::size_of::<usize>()), // pointer size
        _ => None,
    }
}

/// Parses a leading integer, ignoring any non-numeric suffix, as C's `atoi`
/// does.
///
/// See <https://stackoverflow.com/questions/65601579/parse-an-integer-ignoring-any-non-numeric-suffix>.
fn atoi<F: FromStr>(input: &str) -> Result<F, <F as FromStr>::Err> {
    let input = input.trim();

    // Handle sign at the beginning
    let (has_sign, start_idx) = match input.chars().next() {
        Some('+') | Some('-') => (true, 1),
        _ => (false, 0),
    };

    // Find the end of the numeric part (after the optional sign)
    let end_idx = if has_sign {
        input[start_idx..]
            .find(|c: char| !c.is_numeric())
            .map(|i| i + start_idx)
            .unwrap_or(input.len())
    } else {
        input.find(|c: char| !c.is_numeric()).unwrap_or(input.len())
    };

    input[..end_idx].parse::<F>()
}

/// Parses a `[c_char]` buffer as a `c_int`, matching C's `atoi`: returns 0
/// rather than an error on anything unparseable.
fn parse_c_int(cs: &[c_char]) -> c_int {
    // Convert c_char slice to UTF-8 string
    if let Ok(s) = str::from_utf8(cast_slice(cs)) {
        // Use the existing atoi function which handles numeric parsing
        atoi::<c_int>(s).unwrap_or(0)
    } else {
        0 // Return 0 on UTF-8 conversion failure, matching C's atoi behavior
    }
}

/// Formats a `double` in scientific notation with a fixed-width exponent, so
/// that a column of them lines up as C's `%E` does.
///
/// See <https://stackoverflow.com/questions/65264069/alignment-of-floating-point-numbers-printed-in-scientific-notation>.
fn fmt_f64(num: f64, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{num:.precision$E}");
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let exp = num.split_off(num.find('E').unwrap());

    let (sign, exp) = match exp.strip_prefix("E-") {
        Some(rest) => ('-', rest),
        None => ('+', &exp[1..]),
    };
    num.push_str(&format!("E{sign}{exp:0>exp_pad$}"));

    num.to_string()
}

/// `fopen` mode string for writing a binary file.
const WB_MODE: *const c_char = c"wb".as_ptr().cast::<c_char>();
/// `fopen` mode string for reading a binary file.
const RB_MODE: *const c_char = c"rb".as_ptr().cast::<c_char>();

#[cfg(not(target_os = "windows"))]
unsafe extern "C" {

    /// The C runtime's standard input stream.
    #[cfg_attr(target_os = "macos", link_name = "__stdinp")]
    pub unsafe static mut stdin: *mut FILE;

    /// The C runtime's standard output stream.
    #[cfg_attr(target_os = "macos", link_name = "__stdoutp")]
    pub unsafe static mut stdout: *mut FILE;

    /// The C runtime's standard error stream.
    #[cfg_attr(target_os = "macos", link_name = "__stderrp")]
    pub unsafe static mut stderr: *mut FILE;
}

#[cfg(windows)]
unsafe extern "C" {
    /// The Windows CRT accessor for the standard streams, which are not
    /// exported as symbols there. Index 0 is stdin, 1 stdout, 2 stderr.
    pub unsafe fn __acrt_iob_func(idx: libc::c_uint) -> *mut FILE;
}

#[macro_export]
#[cfg(not(target_os = "windows"))]
/// The C runtime's standard input stream.
///
/// A macro because Windows has no exported symbol for it: there the streams
/// are reached through `__acrt_iob_func`.
macro_rules! STDIN {
    () => {
        $crate::stdin
    };
}

#[macro_export]
#[cfg(windows)]
/// The C runtime's standard input stream.
///
/// A macro because Windows has no exported symbol for it: there the streams
/// are reached through `__acrt_iob_func`.
macro_rules! STDIN {
    () => {
        $crate::__acrt_iob_func(0)
    };
}

#[macro_export]
#[cfg(not(target_os = "windows"))]
/// The C runtime's standard output stream.
///
/// A macro because Windows has no exported symbol for it: there the streams
/// are reached through `__acrt_iob_func`.
macro_rules! STDOUT {
    () => {
        $crate::stdout
    };
}

#[macro_export]
#[cfg(windows)]
/// The C runtime's standard output stream.
///
/// A macro because Windows has no exported symbol for it: there the streams
/// are reached through `__acrt_iob_func`.
macro_rules! STDOUT {
    () => {
        $crate::__acrt_iob_func(1)
    };
}

#[macro_export]
#[cfg(not(target_os = "windows"))]
/// The C runtime's standard error stream.
///
/// A macro because Windows has no exported symbol for it: there the streams
/// are reached through `__acrt_iob_func`.
macro_rules! STDERR {
    () => {
        $crate::stderr
    };
}

#[macro_export]
#[cfg(windows)]
/// The C runtime's standard error stream.
///
/// A macro because Windows has no exported symbol for it: there the streams
/// are reached through `__acrt_iob_func`.
macro_rules! STDERR {
    () => {
        $crate::__acrt_iob_func(2)
    };
}

#[cfg(test)]
mod tests {
    use alloc::{ffi::CString, slice};

    use bytemuck::cast_slice;
    use cfileio::{ffclos_safe, ffinit_safe};

    use putkey::ffcrim_safe;
    use tempfile::Builder;

    use crate::{
        fitsio::{USHORT_IMG, fitsfile},
        helpers::testhelpers::with_temp_file,
    };

    use super::*;

    use crate::aliases::rust_api::{fits_update_key, fits_write_img};

    #[test]
    fn test_write_image() {
        unsafe {
            with_temp_file(|_filename| {
                let bitpix = USHORT_IMG;
                let naxis = 2;
                const NAXES: [c_long; 2] = [300, 200];
                let mut storage: [[u16; NAXES[0] as usize]; NAXES[1] as usize] =
                    [[0; NAXES[0] as usize]; NAXES[1] as usize];
                let mut fptr: Option<Box<fitsfile>> = None;
                let mut status = 0;

                let _tempfile = Builder::new()
                    .prefix("my-temporary-note")
                    .suffix(".fits")
                    .tempfile()
                    .unwrap();

                let tdir = Builder::new().prefix("rsfitsio-").tempdir().unwrap();
                let _abc = Builder::new().prefix("prefix").tempfile();
                let tdir_path = tdir.path();
                let filename = tdir_path.join("test.fits");

                let filename_str = filename.to_str().expect("cannot create string filename");
                let filename_cstr = CString::new(filename_str).unwrap();

                status = ffinit_safe(
                    &mut fptr,
                    cast_slice(filename_cstr.as_bytes_with_nul()),
                    &mut status,
                );
                assert_eq!(status, 0);

                let mut fptr = fptr.unwrap();

                status = ffcrim_safe(&mut fptr, bitpix, naxis, &NAXES, &mut status);
                assert_eq!(status, 0);

                for jj in 0..(NAXES[1] as usize) {
                    for ii in 0..(NAXES[0] as usize) {
                        storage[jj][ii] = (ii + jj) as u16;
                    }
                }

                let fpixel = 1; /* first pixel to write      */
                let nelements = NAXES[0] * NAXES[1]; /* number of pixels to write */

                /* write the array of unsigned integers to the FITS file */
                let s = slice::from_raw_parts(
                    storage.as_ptr() as *mut u16,
                    (NAXES[0] * NAXES[1]) as usize,
                );
                fits_write_img(
                    &mut fptr,
                    TUSHORT,
                    fpixel,
                    nelements as LONGLONG,
                    cast_slice(s),
                    &mut status,
                );
                assert_eq!(status, 0);

                /* write another optional keyword to the header */
                /* Note that the ADDRESS of the value is passed in the routine */
                let exposure = 1500;
                fits_update_key(
                    &mut fptr,
                    KeywordDatatype::TLONG(&exposure),
                    cs!(c"EXPOSURE"),
                    Some(cs!(c"Total Exposure Time")),
                    &mut status,
                );
                assert_eq!(status, 0);

                ffclos_safe(fptr, &mut status); /* close the file */
                assert_eq!(status, 0);
            });
        }
    }
}

#[cfg(test)]
mod atoi_tests {
    use super::*;

    #[test]
    fn test_atoi_basic() {
        // Basic integer parsing
        assert_eq!(atoi::<i32>("123").unwrap(), 123);
        assert_eq!(atoi::<i32>("-456").unwrap(), -456);
        assert_eq!(atoi::<i32>("+789").unwrap(), 789);
    }

    #[test]
    fn test_atoi_with_trailing_chars() {
        // Should stop at first non-numeric character
        assert_eq!(atoi::<i32>("123abc").unwrap(), 123);
        assert_eq!(atoi::<i32>("456/789").unwrap(), 456);
        assert_eq!(atoi::<i32>("99\0").unwrap(), 99); // Null terminator
        assert_eq!(atoi::<i32>("15/03/99").unwrap(), 15);
    }

    #[test]
    fn test_atoi_with_whitespace() {
        // Should trim leading/trailing whitespace
        assert_eq!(atoi::<i32>("  123  ").unwrap(), 123);
        assert_eq!(atoi::<i32>("\t456\n").unwrap(), 456);
    }

    #[test]
    fn test_atoi_empty_or_invalid() {
        // Empty string or no digits should fail
        assert!(atoi::<i32>("").is_err());
        assert!(atoi::<i32>("abc").is_err());
        assert!(atoi::<i32>("   ").is_err());
    }

    #[test]
    fn test_atoi_sign_handling() {
        // Sign at the beginning is valid
        assert_eq!(atoi::<i32>("+123").unwrap(), 123);
        assert_eq!(atoi::<i32>("-123").unwrap(), -123);

        // Sign in the middle should stop parsing before it
        // (matching C atoi behavior)
        assert_eq!(atoi::<i32>("123-456").unwrap(), 123);
        assert_eq!(atoi::<i32>("123+456").unwrap(), 123);
    }

    #[test]
    fn test_atoi_matches_c_behavior() {
        // Test cases that should match C atoi behavior
        // C atoi("99") = 99
        assert_eq!(atoi::<i32>("99").unwrap(), 99);

        // C atoi("03") = 3 (leading zeros are ok)
        assert_eq!(atoi::<i32>("03").unwrap(), 3);

        // C atoi("15") = 15
        assert_eq!(atoi::<i32>("15").unwrap(), 15);

        // C atoi("2024") = 2024
        assert_eq!(atoi::<i32>("2024").unwrap(), 2024);
    }
}
