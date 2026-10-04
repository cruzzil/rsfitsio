//! GZIP output must be no larger than CFITSIO's.
//!
//! zlib-rs's level 1 (the C's `Z_BEST_SPEED`) emits static Huffman blocks only,
//! which on noisy pixels made GZIP tiles far larger than CFITSIO's; see
//! `GZIP_LEVEL` in `src/zcompress.rs`. The expected sizes were measured with
//! CFITSIO 4.7.0 and system zlib 1.3, writing the same image with the same calls.

mod common;

#[cfg(test)]
mod tests {
    use crate::common::with_temp_file;
    use bytemuck::cast_slice;
    use libc::c_int;
    use rsfitsio::aliases::rust_api::*;
    use rsfitsio::fitsio::{GZIP_1, GZIP_2, READONLY, RICE_1, SHORT_IMG, fitsfile};
    use rsfitsio::imcompress::fits_set_compression_type_safe;
    use std::ffi::CString;

    const NX: usize = 2900;
    const NY: usize = 64;

    /// A 2900 x 64 image of sky level 1000 with noise of +/- 30: the sum of four
    /// 4-bit values from xorshift32 seeded with 12345. The C harness that
    /// measured CFITSIO generates exactly the same pixels.
    fn noisy_image() -> Vec<i16> {
        let mut state: u32 = 12345;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        (0..NX * NY)
            .map(|_| {
                let sum: u32 = (0..4).map(|_| next() & 15).sum();
                (970 + sum) as i16
            })
            .collect()
    }

    /// Write `img` to `name` (tile-compressed with `comptype`, if given) and
    /// return the file size.
    fn write(name: &str, img: &[i16], comptype: Option<c_int>) -> u64 {
        let c = CString::new(name).unwrap();
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        fits_create_file(&mut fptr, cast_slice(c.to_bytes_with_nul()), &mut status);
        let f = fptr.as_mut().unwrap();
        if let Some(comptype) = comptype {
            fits_set_compression_type_safe(f, comptype, &mut status);
        }
        fits_create_imgll(f, SHORT_IMG, 2, &[NX as i64, NY as i64], &mut status);
        fits_write_img_sht(f, 1, 1, (NX * NY) as i64, img, &mut status);
        fits_close_file(fptr.take().unwrap(), &mut status);
        assert_eq!(status, 0);
        std::fs::metadata(name).unwrap().len()
    }

    /// Read the image back from HDU `hdu`.
    fn read(name: &str, hdu: c_int) -> Vec<i16> {
        let c = CString::new(name).unwrap();
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        fits_open_file(
            &mut fptr,
            cast_slice(c.to_bytes_with_nul()),
            READONLY,
            &mut status,
        );
        let f = fptr.as_mut().unwrap();
        fits_movabs_hdu(f, hdu, None, &mut status);
        let mut out = vec![0i16; NX * NY];
        fits_read_img_sht(f, 1, 1, (NX * NY) as i64, 0, &mut out, None, &mut status);
        fits_close_file(fptr.take().unwrap(), &mut status);
        assert_eq!(status, 0);
        out
    }

    #[test]
    fn test_tile_compressed_size_no_larger_than_cfitsio() {
        let img = noisy_image();
        let raw = (NX * NY * 2) as u64;
        // CFITSIO 4.7.0 file sizes; rsfitsio with zlib-rs level 1 wrote 282 240
        // and 221 760 bytes for the GZIP cases.
        for (comptype, cfitsio) in [(GZIP_1, 190_080), (GZIP_2, 141_120), (RICE_1, 146_880)] {
            with_temp_file(|name| {
                let size = write(name, &img, Some(comptype));
                assert!(size < raw, "comptype {comptype}: {size} bytes");
                assert!(
                    size <= cfitsio,
                    "comptype {comptype}: {size} > CFITSIO's {cfitsio}"
                );
                if comptype == RICE_1 {
                    assert_eq!(size, cfitsio, "RICE_1 does not use deflate");
                }
                assert!(read(name, 2) == img, "comptype {comptype}: round trip");
            });
        }
    }

    /// An uncompressed image written to a `.gz` name: the whole file is
    /// gzipped on close (`compress2file_from_mem`).
    #[test]
    fn test_gz_file_size_no_larger_than_cfitsio() {
        let img = noisy_image();
        with_temp_file(|name| {
            let gz = format!("{name}.gz");
            let size = write(&gz, &img, None);
            // CFITSIO 4.7.0: 165 562 bytes; zlib-rs level 1 wrote 255 087.
            assert!(size <= 165_562, "{size} bytes");
            assert!(read(&gz, 1) == img, "round trip");
        });
    }
}
