//! Read images larger than the internal `DBUFFSIZE` scratch buffer through
//! every output datatype, both with and without scaling.
//!
//! The `ffgclXX` readers have a special case for when the FITS column has the
//! same width as the output array: CFITSIO reads straight into the caller's
//! array and so lifts the per-chunk element limit to `nelem`. Any port of that
//! special case that reads into the fixed-size scratch buffer instead overflows
//! it once the image is larger than `DBUFFSIZE` bytes.
mod common;

use crate::common::with_temp_file;
use bytemuck::cast_slice;
use libc::{c_int, c_long};
use rsfitsio::aliases::rust_api::*;
use rsfitsio::fitsio::{
    BYTE_IMG, DOUBLE_IMG, FLOAT_IMG, LONG_IMG, LONGLONG, LONGLONG_IMG, READONLY, SBYTE_IMG,
    SHORT_IMG, TBYTE, TDOUBLE, TFLOAT, TINT, TLONG, TLONGLONG, TSBYTE, TSHORT, TUINT, TULONG,
    TULONGLONG, TUSHORT, ULONG_IMG, ULONGLONG_IMG, USHORT_IMG, fitsfile,
};
use std::ffi::CString;

/// 200 x 200 pixels: 40000 elements, larger than DBUFFSIZE for every width.
const WIDTH: c_long = 200;
const HEIGHT: c_long = 200;
const NELEM: usize = (WIDTH * HEIGHT) as usize;

const BITPIXES: [c_int; 10] = [
    BYTE_IMG,
    SBYTE_IMG,
    SHORT_IMG,
    USHORT_IMG,
    LONG_IMG,
    ULONG_IMG,
    LONGLONG_IMG,
    ULONGLONG_IMG,
    FLOAT_IMG,
    DOUBLE_IMG,
];

fn expected(i: usize) -> f64 {
    (i % 50) as f64
}

/// Decode `nelem` values of `datatype` from `bytes` as f64.
fn decode(datatype: c_int, bytes: &[u8]) -> Vec<f64> {
    macro_rules! dec {
        ($t:ty) => {
            bytes
                .chunks_exact(core::mem::size_of::<$t>())
                .map(|c| <$t>::from_ne_bytes(c.try_into().unwrap()) as f64)
                .collect()
        };
    }
    match datatype {
        TBYTE => dec!(u8),
        TSBYTE => dec!(i8),
        TSHORT => dec!(i16),
        TUSHORT => dec!(u16),
        TINT => dec!(i32),
        TUINT => dec!(u32),
        TLONG => dec!(libc::c_long),
        TULONG => dec!(libc::c_ulong),
        TLONGLONG => dec!(i64),
        TULONGLONG => dec!(u64),
        TFLOAT => dec!(f32),
        TDOUBLE => dec!(f64),
        _ => unreachable!(),
    }
}

fn width(datatype: c_int) -> usize {
    match datatype {
        TBYTE | TSBYTE => 1,
        TSHORT | TUSHORT => 2,
        TINT | TUINT | TFLOAT => 4,
        TLONG | TULONG => core::mem::size_of::<libc::c_long>(),
        TLONGLONG | TULONGLONG | TDOUBLE => 8,
        _ => unreachable!(),
    }
}

fn check(bitpix: c_int, datatype: c_int, scaled: bool) {
    with_temp_file(|filename| {
        let name = CString::new(filename).unwrap();
        let name = cast_slice(name.to_bytes_with_nul());
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;

        let data: Vec<f64> = (0..NELEM).map(expected).collect();
        fits_create_file(&mut fptr, name, &mut status);
        let f = fptr.as_mut().unwrap();
        fits_create_img(f, bitpix, 2, &[WIDTH, HEIGHT], &mut status);
        fits_write_img(
            f,
            TDOUBLE,
            1,
            NELEM as LONGLONG,
            cast_slice(&data),
            &mut status,
        );
        assert_eq!(status, 0, "write failed");
        fits_close_file(fptr.take().unwrap(), &mut status);
        assert_eq!(status, 0, "close failed");

        fits_open_file(&mut fptr, name, READONLY, &mut status);
        assert_eq!(status, 0, "open failed");
        let f = fptr.as_mut().unwrap();

        // Scaling forces the "convert" branch of each reader.
        let scale = if scaled { 2.0 } else { 1.0 };
        if scaled {
            fits_set_bscale(f, scale, 0.0, &mut status);
            assert_eq!(status, 0, "set_bscale failed");
        }

        let mut out = vec![0u8; NELEM * width(datatype)];
        let mut anynul: c_int = 0;
        fits_read_img(
            f,
            datatype,
            1,
            NELEM as LONGLONG,
            None,
            &mut out,
            Some(&mut anynul),
            &mut status,
        );
        assert_eq!(status, 0, "read failed");

        let got = decode(datatype, &out);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, expected(i) * scale, "pixel {i}");
        }

        fits_close_file(fptr.take().unwrap(), &mut status);
    });
}

/// One test per output datatype, so a failure names the reader at fault.
macro_rules! read_tests {
    ($($name:ident: $datatype:expr,)*) => {
        $(
            mod $name {
                use super::*;

                #[test]
                fn unscaled() {
                    for bitpix in BITPIXES {
                        check(bitpix, $datatype, false);
                    }
                }

                #[test]
                fn scaled() {
                    // Unsigned BITPIX values are stored with a BZERO offset
                    // that fits_set_bscale would discard, so only scale the
                    // natively signed types.
                    for bitpix in [BYTE_IMG, SHORT_IMG, LONG_IMG, LONGLONG_IMG, FLOAT_IMG, DOUBLE_IMG] {
                        check(bitpix, $datatype, true);
                    }
                }
            }
        )*
    };
}

read_tests! {
    tbyte: TBYTE,
    tsbyte: TSBYTE,
    tshort: TSHORT,
    tushort: TUSHORT,
    tint: TINT,
    tuint: TUINT,
    tlong: TLONG,
    tulong: TULONG,
    tlonglong: TLONGLONG,
    tulonglong: TULONGLONG,
    tfloat: TFLOAT,
    tdouble: TDOUBLE,
}
