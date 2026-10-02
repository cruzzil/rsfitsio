//! Malformed files must be reported through `status`, never by panicking.
//!
//! Each case was found by fuzzing arcsec's FITS reader, which reads images and
//! Astrometry.net index tables through rsfitsio.

mod common;

#[cfg(test)]
mod tests {
    use crate::common::with_temp_file;
    use bytemuck::cast_slice;
    use libc::{c_int, c_long};
    use rsfitsio::aliases::rust_api::*;
    use rsfitsio::fitsio::{
        BAD_C2D, BAD_C2F, BAD_C2I, GZIP_1, LONGLONG, READONLY, RICE_1, SHORT_IMG, ULONGLONG,
        fitsfile,
    };
    use rsfitsio::imcompress::fits_set_compression_type_safe;
    use std::ffi::CString;

    /// One 2880-byte header block from 80-column cards.
    fn header(cards: &[&[u8]]) -> Vec<u8> {
        let mut h: Vec<u8> = Vec::new();
        for c in cards.iter().chain([&b"END"[..]].iter()) {
            let mut card = c.to_vec();
            card.resize(80, b' ');
            h.extend_from_slice(&card);
        }
        h.resize(h.len().div_ceil(2880) * 2880, b' ');
        h
    }

    /// A data-less primary HDU with `cards` appended to its header.
    fn primary(cards: &[&[u8]]) -> Vec<u8> {
        let mut all: Vec<&[u8]> = vec![
            b"SIMPLE  =                    T",
            b"BITPIX  =                    8",
            b"NAXIS   =                    0",
        ];
        all.extend_from_slice(cards);
        header(&all)
    }

    fn open(name: &str, fptr: &mut Option<Box<fitsfile>>, status: &mut c_int) {
        let c = CString::new(name).unwrap();
        fits_open_diskfile(fptr, cast_slice(c.to_bytes_with_nul()), READONLY, status);
    }

    /// Numeric keywords whose values are not text (bytes that are not UTF-8)
    /// or too long to quote in the error message.
    #[test]
    fn test_bad_numeric_keyword_values() {
        with_temp_file(|name| {
            std::fs::write(
                name,
                primary(&[
                    b"DBLKEY  = 1.5\xb2",
                    b"FLTKEY  = \xff2.5",
                    b"ULLKEY  = 12\xb2",
                    // 40 characters that are not an integer: the error message
                    // quotes 30 of them after a 52-character prefix, in 81 bytes.
                    b"LLKEY   = 1234567890123456789012345678901234567X",
                ]),
            )
            .unwrap();
            let mut fptr: Option<Box<fitsfile>> = None;
            let mut status: c_int = 0;
            open(name, &mut fptr, &mut status);
            assert_eq!(status, 0);
            let f = fptr.as_mut().unwrap();

            let (mut d, mut st) = (0f64, 0);
            fits_read_key_dbl(f, cast_slice(b"DBLKEY\0"), &mut d, None, &mut st);
            assert_eq!(st, BAD_C2D);
            let (mut e, mut st) = (0f32, 0);
            fits_read_key_flt(f, cast_slice(b"FLTKEY\0"), &mut e, None, &mut st);
            assert_eq!(st, BAD_C2F);
            let (mut u, mut st): (ULONGLONG, c_int) = (0, 0);
            fits_read_key_ulnglng(f, cast_slice(b"ULLKEY\0"), &mut u, None, &mut st);
            assert_eq!(st, BAD_C2I);
            let (mut j, mut st): (LONGLONG, c_int) = (0, 0);
            fits_read_key_lnglng(f, cast_slice(b"LLKEY\0"), &mut j, None, &mut st);
            assert_ne!(st, 0);
            let (mut l, mut st): (c_long, c_int) = (0, 0);
            fits_read_key_lng(f, cast_slice(b"LLKEY\0"), &mut l, None, &mut st);
            assert_ne!(st, 0);

            fits_close_file(fptr.take().unwrap(), &mut status);
        });
    }

    /// A binary table TFORM of the `rAw` form whose width is not text.
    #[test]
    fn test_tform_with_non_text_width() {
        with_temp_file(|name| {
            let mut bytes = primary(&[b"EXTEND  =                    T"]);
            bytes.extend(header(&[
                b"XTENSION= 'BINTABLE'",
                b"BITPIX  =                    8",
                b"NAXIS   =                    2",
                b"NAXIS1  =                    8",
                b"NAXIS2  =                    1",
                b"PCOUNT  =                    0",
                b"GCOUNT  =                    1",
                b"TFIELDS =                    1",
                b"TFORM1  = '8A\xb2'",
                b"TTYPE1  = 'NAME'",
            ]));
            bytes.extend_from_slice(&[b'x'; 8]);
            bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
            std::fs::write(name, bytes).unwrap();

            let mut fptr: Option<Box<fitsfile>> = None;
            let mut status: c_int = 0;
            open(name, &mut fptr, &mut status);
            let f = fptr.as_mut().unwrap();
            fits_movabs_hdu(f, 2, None, &mut status);
            // Reaching here is the test: the width is ignored, as an unreadable
            // `w` is in CFITSIO.
            let mut st = 0;
            fits_close_file(fptr.take().unwrap(), &mut st);
        });
    }

    /// Write a 16 × 8 tile-compressed SHORT image and return the file's bytes.
    fn compressed_short_image(name: &str, comptype: c_int) -> Vec<u8> {
        let c = CString::new(name).unwrap();
        let data: Vec<i16> = (0..128).map(|i| (i * 37 % 1000) as i16).collect();
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        fits_create_diskfile(&mut fptr, cast_slice(c.to_bytes_with_nul()), &mut status);
        let f = fptr.as_mut().unwrap();
        fits_set_compression_type_safe(f, comptype, &mut status);
        fits_create_imgll(f, SHORT_IMG, 2, &[16, 8], &mut status);
        fits_write_img_sht(f, 1, 1, 128, &data, &mut status);
        fits_close_file(fptr.take().unwrap(), &mut status);
        assert_eq!(status, 0);
        std::fs::read(name).unwrap()
    }

    /// Replace the value of `key` in the header with `value` (right-justified).
    fn set_card(bytes: &mut [u8], key: &str, value: &str) {
        let k = format!("{key:<8}=");
        let at = bytes
            .chunks(80)
            .position(|c| c.starts_with(k.as_bytes()))
            .unwrap_or_else(|| panic!("{key} not found"))
            * 80;
        let card = format!("{k} {value:>20}");
        bytes[at..at + 80].fill(b' ');
        bytes[at..at + card.len()].copy_from_slice(card.as_bytes());
    }

    /// Read the compressed image back, expecting an error status.
    fn read_back_fails(name: &str, bytes: &[u8]) {
        std::fs::write(name, bytes).unwrap();
        let mut fptr: Option<Box<fitsfile>> = None;
        let mut status: c_int = 0;
        open(name, &mut fptr, &mut status);
        let f = fptr.as_mut().unwrap();
        fits_movabs_hdu(f, 2, None, &mut status);
        let mut out = vec![0f32; 128];
        fits_read_img_flt(f, 1, 1, 128, 0.0, &mut out, None, &mut status);
        assert_ne!(status, 0, "a corrupt tile must be an error");
        let mut st = 0;
        fits_close_file(fptr.take().unwrap(), &mut st);
    }

    /// A GZIP tile that inflates to more than ZBITPIX allows. The decompressor
    /// used to be handed C `realloc` for a Rust-owned buffer and panicked rather
    /// than call it.
    #[test]
    fn test_gzip_tile_larger_than_zbitpix() {
        with_temp_file(|name| {
            let mut bytes = compressed_short_image(name, GZIP_1);
            set_card(&mut bytes, "ZBITPIX", "8");
            read_back_fails(name, &bytes);
        });
    }

    /// A RICE tile whose BYTEPIX disagrees with ZBITPIX: the tile buffer is sized
    /// for one and the decoder asserted it was exactly the other.
    #[test]
    fn test_rice_bytepix_inconsistent_with_zbitpix() {
        with_temp_file(|name| {
            let mut bytes = compressed_short_image(name, RICE_1);
            set_card(&mut bytes, "ZBITPIX", "32");
            // Decoding garbage may still fail later; what matters is no panic.
            std::fs::write(name, &bytes).unwrap();
            let mut fptr: Option<Box<fitsfile>> = None;
            let mut status: c_int = 0;
            open(name, &mut fptr, &mut status);
            let f = fptr.as_mut().unwrap();
            fits_movabs_hdu(f, 2, None, &mut status);
            let mut out = vec![0f32; 128];
            fits_read_img_flt(f, 1, 1, 128, 0.0, &mut out, None, &mut status);
            let mut st = 0;
            fits_close_file(fptr.take().unwrap(), &mut st);
        });
    }

    /// Tile and image sizes that make the expected row count zero, and the
    /// overflow test after it divide by zero (or overflow the division).
    #[test]
    fn test_non_positive_ztile_or_znaxis() {
        for (key, value) in [
            // With ZTILE2 = 1: rowFactor = (0 - 1) / 1 + 1 = 0.
            ("ZNAXIS2", "0"),
            ("ZTILE1", "-1"),
            ("ZTILE2", "-3"),
            ("ZNAXIS1", "0"),
            ("ZNAXIS2", "-1"),
        ] {
            with_temp_file(|name| {
                let mut bytes = compressed_short_image(name, RICE_1);
                set_card(&mut bytes, key, value);
                std::fs::write(name, &bytes).unwrap();
                let mut fptr: Option<Box<fitsfile>> = None;
                let mut status: c_int = 0;
                open(name, &mut fptr, &mut status);
                let f = fptr.as_mut().unwrap();
                fits_movabs_hdu(f, 2, None, &mut status);
                assert_ne!(status, 0, "{key} = {value}");
                let mut st = 0;
                fits_close_file(fptr.take().unwrap(), &mut st);
            });
        }
    }

    /// A tile descriptor claiming far more bytes than the heap holds: refused
    /// before the tile buffer is allocated for it.
    #[test]
    fn test_tile_descriptor_outside_the_heap() {
        with_temp_file(|name| {
            let mut bytes = compressed_short_image(name, RICE_1);
            // The table's data follows the second header; its first row starts
            // with the COMPRESSED_DATA descriptor (1PB: i32 count, i32 offset).
            let hdr2 = bytes
                .chunks(80)
                .position(|c| c.starts_with(b"XTENSION"))
                .unwrap()
                * 80;
            let end = hdr2
                + bytes[hdr2..]
                    .chunks(80)
                    .position(|c| c.starts_with(b"END     "))
                    .unwrap()
                    * 80;
            let data = end.div_ceil(2880) * 2880;
            bytes[data..data + 4].copy_from_slice(&0x3fff_ffffi32.to_be_bytes());
            read_back_fails(name, &bytes);

            // Refused for the right reason, rather than after allocating a
            // gigabyte and failing to fill it.
            let mut messages = Vec::new();
            let mut msg = [0 as libc::c_char; rsfitsio::fitsio::FLEN_ERRMSG];
            while fits_read_errmsg(&mut msg) != 0 {
                let bytes: &[u8] = cast_slice(&msg);
                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                messages.push(String::from_utf8_lossy(&bytes[..end]).into_owned());
            }
            assert!(
                messages.iter().any(|m| m.contains("outside the heap")),
                "{messages:?}"
            );
        });
    }
}
