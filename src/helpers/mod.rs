//! Small Rust-side helpers with no CFITSIO counterpart.
//!
//! These exist to express in safe Rust what the C did with raw allocation and
//! `FILE*` I/O: fallible boxing, an over-aligned byte buffer, a `Read`/`Write`
//! adapter over a C stream, and the test fixtures.
#![warn(missing_docs)]

pub mod aligned;
pub mod boxed;
pub mod cfile;
pub(crate) mod outbuf;
#[cfg(test)]
pub mod testhelpers;
pub mod vec_raw_parts;
