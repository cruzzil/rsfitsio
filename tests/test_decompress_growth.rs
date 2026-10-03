//! Decompression into a buffer that has to grow.
//!
//! CFITSIO's gunzip helpers grow their output buffer with `realloc` when a
//! stream inflates to more than the caller expected. Each case here was run
//! through CFITSIO 4.7 and the expected status and data are what it returned.

mod common;

#[cfg(test)]
mod tests {
    use crate::common::with_temp_file;
    use bytemuck::cast_slice;
    use libc::{c_int, c_long};
    use rsfitsio::aliases::rust_api::*;
    use rsfitsio::fitsio::{
        BYTE_IMG, DATA_DECOMPRESSION_ERR, DOUBLE_IMG, FLOAT_IMG, GZIP_1, GZIP_2, LONG_IMG,
        READONLY, SHORT_IMG, fitsfile,
    };
    use rsfitsio::imcompress::fits_set_compression_type_safe;
    use std::ffi::CString;

    fn cstr(name: &str) -> CString {
        CString::new(name).unwrap()
    }

    /// The pixel values every image here holds.
    fn pixel(i: usize) -> f64 {
        (i * 7 % 200) as f64
    }

    /// Write a 16 × 8 tile-compressed image of `bitpix` and return the file.
    fn compressed_image(name: &str, bitpix: c_int, comptype: c_int) -> Vec<u8> {
        let c = cstr(name);
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        fits_create_diskfile(&mut fptr, cast_slice(c.to_bytes_with_nul()), &mut status);
        let f = fptr.as_mut().unwrap();
        fits_set_compression_type_safe(f, comptype, &mut status);
        fits_create_imgll(f, bitpix, 2, &[16, 8], &mut status);
        match bitpix {
            BYTE_IMG => {
                let d: Vec<u8> = (0..128).map(|i| pixel(i) as u8).collect();
                fits_write_img_byt(f, 1, 1, 128, &d, &mut status);
            }
            SHORT_IMG => {
                let d: Vec<i16> = (0..128).map(|i| pixel(i) as i16).collect();
                fits_write_img_sht(f, 1, 1, 128, &d, &mut status);
            }
            LONG_IMG => {
                let d: Vec<c_int> = (0..128).map(|i| pixel(i) as c_int).collect();
                fits_write_img_int(f, 1, 1, 128, &d, &mut status);
            }
            FLOAT_IMG => {
                let d: Vec<f32> = (0..128).map(|i| pixel(i) as f32).collect();
                fits_write_img_flt(f, 1, 1, 128, &d, &mut status);
            }
            _ => {
                let d: Vec<f64> = (0..128).map(pixel).collect();
                fits_write_img_dbl(f, 1, 1, 128, &d, &mut status);
            }
        }
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

    /// Read the compressed image as doubles: (BITPIX, status, pixels).
    fn read_image(name: &str) -> (c_int, c_int, Vec<f64>) {
        let c = cstr(name);
        let mut fptr: Option<Box<fitsfile>> = None;
        let mut status: c_int = 0;
        fits_open_diskfile(
            &mut fptr,
            cast_slice(c.to_bytes_with_nul()),
            READONLY,
            &mut status,
        );
        let f = fptr.as_mut().unwrap();
        fits_movabs_hdu(f, 2, None, &mut status);
        let mut bitpix = 0;
        fits_get_img_type(f, &mut bitpix, &mut status);
        let mut out = vec![0f64; 128];
        fits_read_img_dbl(f, 1, 1, 128, 0.0, &mut out, None, &mut status);
        let mut st = 0;
        fits_close_file(fptr.take().unwrap(), &mut st);
        (bitpix, status, out)
    }

    /// A GZIP tile holding wider integers than ZBITPIX implies: the tile
    /// buffer, sized from ZBITPIX, grows to take them, and the tile decodes.
    #[test]
    fn test_gzip_tile_wider_than_zbitpix() {
        for comptype in [GZIP_1, GZIP_2] {
            for (bitpix, zbitpix) in [
                (LONG_IMG, SHORT_IMG),
                (LONG_IMG, BYTE_IMG),
                (SHORT_IMG, BYTE_IMG),
                // narrower than ZBITPIX: no growth, as before
                (SHORT_IMG, LONG_IMG),
                (BYTE_IMG, SHORT_IMG),
            ] {
                with_temp_file(|name| {
                    let mut bytes = compressed_image(name, bitpix, comptype);
                    set_card(&mut bytes, "ZBITPIX", &zbitpix.to_string());
                    std::fs::write(name, &bytes).unwrap();

                    let (got_bitpix, status, out) = read_image(name);
                    let case = format!("comptype {comptype} BITPIX {bitpix} ZBITPIX {zbitpix}");
                    assert_eq!(got_bitpix, zbitpix, "{case}");
                    assert_eq!(status, 0, "{case}");
                    for (i, v) in out.iter().enumerate() {
                        assert_eq!(*v, pixel(i), "{case}: pixel {i}");
                    }
                });
            }
        }
    }

    /// Floating-point GZIP tiles are inflated into fixed buffers (the C passes
    /// no realloc function), so a tile wider than ZBITPIX is an error.
    #[test]
    fn test_gzip_float_tile_wider_than_zbitpix() {
        for comptype in [GZIP_1, GZIP_2] {
            with_temp_file(|name| {
                let mut bytes = compressed_image(name, DOUBLE_IMG, comptype);
                set_card(&mut bytes, "ZBITPIX", &FLOAT_IMG.to_string());
                std::fs::write(name, &bytes).unwrap();

                let (got_bitpix, status, _) = read_image(name);
                assert_eq!(got_bitpix, FLOAT_IMG);
                assert_eq!(status, DATA_DECOMPRESSION_ERR);
            });
        }
    }

    /// A 200 × 200 SHORT image of `i % 7`, written to `name` gzipped.
    fn gzipped_image(name: &str) -> Vec<u8> {
        let gz = format!("{name}.gz");
        let c = cstr(&gz);
        let d: Vec<i16> = (0..40000).map(|i| (i % 7) as i16).collect();
        let mut status: c_int = 0;
        let mut fptr: Option<Box<fitsfile>> = None;
        fits_create_file(&mut fptr, cast_slice(c.to_bytes_with_nul()), &mut status);
        assert_eq!(status, 0, "create {gz}");
        let f = fptr.as_mut().unwrap();
        fits_create_imgll(f, SHORT_IMG, 2, &[200, 200], &mut status);
        fits_write_img_sht(f, 1, 1, 40000, &d, &mut status);
        fits_close_file(fptr.take().unwrap(), &mut status);
        assert_eq!(status, 0);
        let bytes = std::fs::read(&gz).unwrap();
        assert_eq!(bytes[..2], [0x1f, 0x8b], "written gzipped");
        bytes
    }

    /// Open `name` and read the primary image: (open status, NAXISn, read
    /// status, weighted sum).
    fn open_and_sum(name: &str) -> (c_int, [c_long; 2], c_int, f64) {
        let c = cstr(name);
        let mut fptr: Option<Box<fitsfile>> = None;
        let mut status: c_int = 0;
        fits_open_file(
            &mut fptr,
            cast_slice(c.to_bytes_with_nul()),
            READONLY,
            &mut status,
        );
        if status != 0 {
            return (status, [0, 0], 0, 0.0);
        }
        let f = fptr.as_mut().unwrap();
        let mut naxes: [c_long; 2] = [0; 2];
        fits_get_img_param(f, 2, None, None, Some(&mut naxes), &mut status);
        let n = (naxes[0] * naxes[1]) as usize;
        let mut out = vec![0f64; n];
        let mut rstatus = 0;
        fits_read_img_dbl(f, 1, 1, n as i64, 0.0, &mut out, None, &mut rstatus);
        let sum = out
            .iter()
            .enumerate()
            .map(|(i, v)| v * (i % 7 + 1) as f64)
            .sum();
        let mut st = 0;
        fits_close_file(fptr.take().unwrap(), &mut st);
        (status, naxes, rstatus, sum)
    }

    /// A `.gz` file whose gzip ISIZE trailer understates the uncompressed size:
    /// here a second, tiny member follows the image, and the trailer is that
    /// member's. The memory file is sized from the trailer (or, for 0, from an
    /// estimate of 3 × the compressed size) and grows as the image inflates.
    /// Only the first member is read, as in CFITSIO.
    #[test]
    fn test_open_gz_with_understated_size() {
        let members: [&[u8]; 2] = [
            /* gzip of "" (ISIZE 0: the size is estimated) */
            &[
                0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0x03, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            /* gzip of "x" (ISIZE 1) */
            &[
                0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0xab, 0x00, 0x00, 0x83,
                0x16, 0xdc, 0x8c, 0x01, 0x00, 0x00, 0x00,
            ],
        ];
        for member in members {
            with_temp_file(|name| {
                let mut bytes = gzipped_image(name);
                bytes.extend_from_slice(member);
                let gz = format!("{name}.gz");
                std::fs::write(&gz, &bytes).unwrap();

                /* CFITSIO: open status 0, 200 x 200, read status 0, sum 639970 */
                assert_eq!(open_and_sum(&gz), (0, [200, 200], 0, 639970.0));
            });
        }
    }

    /// A compressed table whose first row's VLA inflates to 42000 bytes though
    /// its descriptor says 2000. The VLA buffer grows to take the whole stream
    /// and, as in CFITSIO, all of it is written at the row's heap offset: rows
    /// 2 and 3 are rewritten after it, and the rest runs past the heap.
    ///
    /// The fixture is a 3-row `1PB(2000)` table (row 1: 300 varied bytes then
    /// 5s; row 2: 7s; row 3: 9s) compressed by CFITSIO with `FZALG1 = 'GZIP_1'`,
    /// with row 1's gzip stream replaced by one of the same length holding
    /// 42000 bytes of 5 (padded with an FCOMMENT field).
    #[test]
    fn test_uncompress_table_vla_longer_than_descriptor() {
        with_temp_file(|name| {
            let fixture = cstr(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/vla_gzip_longer_than_descriptor.fits"
            ));
            let out = cstr(name);
            let mut status: c_int = 0;
            let mut fi: Option<Box<fitsfile>> = None;
            let mut fo: Option<Box<fitsfile>> = None;
            fits_open_diskfile(
                &mut fi,
                cast_slice(fixture.to_bytes_with_nul()),
                READONLY,
                &mut status,
            );
            fits_create_diskfile(&mut fo, cast_slice(out.to_bytes_with_nul()), &mut status);
            let (i, o) = (fi.as_mut().unwrap(), fo.as_mut().unwrap());
            fits_copy_hdu(i, o, 0, &mut status);
            fits_movabs_hdu(i, 2, None, &mut status);
            fits_uncompress_table(i, o, &mut status);
            assert_eq!(status, 0);

            let mut nrows = 0;
            fits_get_num_rows(o, &mut nrows, &mut status);
            assert_eq!(nrows, 3);
            for (row, fill) in [(1, 5u8), (2, 7), (3, 9)] {
                let (mut len, mut off) = (0, 0);
                fits_read_descriptll(o, 1, row, Some(&mut len), Some(&mut off), &mut status);
                assert_eq!((len, off), (2000, (row - 1) * 2000));
                let mut v = vec![0u8; 2000];
                fits_read_col_byt(o, 1, row, 1, 2000, 0, &mut v, None, &mut status);
                assert!(v.iter().all(|&b| b == fill), "row {row}");
            }
            assert_eq!(status, 0);
            fits_close_file(fo.take().unwrap(), &mut status);
            fits_close_file(fi.take().unwrap(), &mut status);
            assert_eq!(status, 0);

            /* the 40000 bytes past the VLA's own 2000 extend the file */
            assert_eq!(std::fs::metadata(name).unwrap().len(), 48960);
        });
    }
}
