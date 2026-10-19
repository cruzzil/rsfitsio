//! Compressing and uncompressing tables whose columns don't start on a 2, 4
//! or 8 byte boundary in the column-major buffer.
//!
//! Each tile holds every row, and an odd number of rows of a `1B` column puts
//! the column after it at an odd offset. The tables are large enough (over
//! 5760 bytes) that they are compressed rather than copied.

mod common;

#[cfg(test)]
mod tests {
    use crate::common::with_temp_file;
    use bytemuck::cast_slice;
    use libc::{c_char, c_int, c_long, c_short};
    use rsfitsio::aliases::rust_api::*;
    use rsfitsio::fitsio::{BINARY_TBL, fitsfile};
    use std::ffi::CString;

    fn cstr(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    /// Write a table of a `1B` column and a column of `form` with `nrows`
    /// rows, filled by `fill`, then compress and uncompress it and hand the
    /// result to `check`.
    fn round_trip(
        form: &str,
        nrows: LONGLONG,
        keys: &[(&str, &str)],
        fill: impl Fn(&mut fitsfile, &mut c_int),
        check: impl Fn(&mut fitsfile, &mut c_int),
    ) {
        with_temp_file(|name| {
            let c = cstr(name);
            let mut status: c_int = 0;
            let mut fptr: Option<Box<fitsfile>> = None;
            fits_create_diskfile(&mut fptr, cast_slice(c.to_bytes_with_nul()), &mut status);
            let f = fptr.as_mut().unwrap();

            let (t1, t2) = (cstr("FLAG"), cstr("DATA"));
            let (f1, f2) = (cstr("1B"), cstr(form));
            let ttype: [Option<&[c_char]>; 2] = [
                Some(cast_slice(t1.to_bytes_with_nul())),
                Some(cast_slice(t2.to_bytes_with_nul())),
            ];
            let tform: [&[c_char]; 2] = [
                cast_slice(f1.to_bytes_with_nul()),
                cast_slice(f2.to_bytes_with_nul()),
            ];
            fits_create_tbl(
                f,
                BINARY_TBL,
                nrows,
                2,
                &ttype,
                &tform,
                None,
                None,
                &mut status,
            );
            for (key, value) in keys {
                let (k, v) = (cstr(key), cstr(value));
                fits_write_key_str(
                    f,
                    cast_slice(k.to_bytes_with_nul()),
                    cast_slice(v.to_bytes_with_nul()),
                    None,
                    &mut status,
                );
            }
            let flags: Vec<u8> = (0..nrows).map(|i| i as u8).collect();
            fits_write_col_byt(f, 1, 1, 1, nrows, &flags, &mut status);
            fill(f, &mut status);
            assert_eq!(status, 0, "write failed");

            let (cname, uname) = (cstr(&format!("{name}.fz")), cstr(&format!("{name}.out")));
            let mut cptr: Option<Box<fitsfile>> = None;
            let mut uptr: Option<Box<fitsfile>> = None;
            fits_create_diskfile(
                &mut cptr,
                cast_slice(cname.to_bytes_with_nul()),
                &mut status,
            );
            fits_create_diskfile(
                &mut uptr,
                cast_slice(uname.to_bytes_with_nul()),
                &mut status,
            );
            let (cf, uf) = (cptr.as_mut().unwrap(), uptr.as_mut().unwrap());
            fits_movabs_hdu(f, 1, None, &mut status);
            fits_copy_hdu(f, cf, 0, &mut status);
            fits_copy_hdu(f, uf, 0, &mut status);
            fits_movabs_hdu(f, 2, None, &mut status);
            fits_compress_table(f, cf, &mut status);
            assert_eq!(status, 0, "compress failed");
            fits_movabs_hdu(cf, 2, None, &mut status);
            fits_uncompress_table(cf, uf, &mut status);
            assert_eq!(status, 0, "uncompress failed");

            let mut read = vec![0u8; nrows as usize];
            fits_read_col_byt(uf, 1, 1, 1, nrows, 0, &mut read, None, &mut status);
            assert_eq!(read, flags);
            check(uf, &mut status);
            assert_eq!(status, 0, "read failed");

            fits_close_file(uptr.take().unwrap(), &mut status);
            fits_close_file(cptr.take().unwrap(), &mut status);
            fits_close_file(fptr.take().unwrap(), &mut status);
            assert_eq!(status, 0, "close failed");
        });
    }

    type LONGLONG = i64;

    fn vla_row(row: LONGLONG) -> Vec<c_long> {
        (0..row * 1000).map(|i| (i * row) as c_long).collect()
    }

    fn vla_round_trip(form: &str) {
        round_trip(
            form,
            3,
            &[],
            |f, status| {
                for row in 1..=3 {
                    let data = vla_row(row);
                    fits_write_col_lng(f, 2, row, 1, data.len() as _, &data, status);
                }
            },
            |f, status| {
                for row in 1..=3 {
                    let expected = vla_row(row);
                    let mut read = vec![0; expected.len()];
                    fits_read_col_lng(f, 2, row, 1, read.len() as _, 0, &mut read, None, status);
                    assert_eq!(read, expected, "row {row}");
                }
            },
        );
    }

    #[test]
    fn test_p_descriptors_at_an_odd_offset() {
        vla_round_trip("1PJ");
    }

    #[test]
    fn test_q_descriptors_at_an_odd_offset() {
        vla_round_trip("1QJ");
    }

    const NROWS: LONGLONG = 2001;

    #[test]
    fn test_rice_short_column_at_an_odd_offset() {
        let data: Vec<c_short> = (0..NROWS).map(|i| (i * 7 - 5000) as c_short).collect();
        round_trip(
            "1I",
            NROWS,
            &[("FZALG2", "RICE_1")],
            |f, status| {
                fits_write_col_sht(f, 2, 1, 1, NROWS, &data, status);
            },
            |f, status| {
                let mut read = vec![0; NROWS as usize];
                fits_read_col_sht(f, 2, 1, 1, NROWS, 0, &mut read, None, status);
                assert_eq!(read, data);
            },
        );
    }

    #[test]
    fn test_rice_int_column_at_an_odd_offset() {
        let data: Vec<c_long> = (0..NROWS).map(|i| (i * 70001 - 5000) as c_long).collect();
        round_trip(
            "1J",
            NROWS,
            &[("FZALG2", "RICE_1")],
            |f, status| {
                fits_write_col_lng(f, 2, 1, 1, NROWS, &data, status);
            },
            |f, status| {
                let mut read = vec![0; NROWS as usize];
                fits_read_col_lng(f, 2, 1, 1, NROWS, 0, &mut read, None, status);
                assert_eq!(read, data);
            },
        );
    }
}
