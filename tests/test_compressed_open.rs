//! Opening a compressed file whose stated uncompressed size is far larger than
//! the file could inflate to.
//!
//! The memory driver sizes its buffer from the gzip ISIZE trailer (plus 4 GiB
//! when the trailer is smaller than the file) or the PKZIP size field. CFITSIO
//! mallocs that size but never touches most of it; rsfitsio used to zero all of
//! it, costing gigabytes and seconds per file. The statuses and data below are
//! CFITSIO 4.7.0's for the same inputs.

mod common;

#[cfg(test)]
mod tests {
    use crate::common::with_temp_file;
    use bytemuck::cast_slice;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use libc::c_int;
    use rsfitsio::aliases::rust_api::*;
    use rsfitsio::fitsio::{DATA_DECOMPRESSION_ERR, READONLY, fitsfile};
    use std::ffi::CString;
    use std::io::Write;

    fn gzip(data: &[u8]) -> Vec<u8> {
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    /// Open `name` as an image and read it as bytes: (status, pixels).
    fn open_and_read(name: &str) -> (c_int, Vec<u8>) {
        let c = CString::new(name).unwrap();
        let mut fptr: Option<Box<fitsfile>> = None;
        let mut status: c_int = 0;
        fits_open_image(
            &mut fptr,
            cast_slice(c.to_bytes_with_nul()),
            READONLY,
            &mut status,
        );
        if status != 0 {
            return (status, Vec::new());
        }
        let f = fptr.as_mut().unwrap();
        let mut n = 0;
        fits_get_img_size(f, 1, std::slice::from_mut(&mut n), &mut status);
        let mut out = vec![0u8; n as usize];
        fits_read_img_byt(f, 1, 1, i64::from(n), 0, &mut out, None, &mut status);
        let mut st = 0;
        fits_close_file(fptr.take().unwrap(), &mut st);
        (status, out)
    }

    /// Peak resident memory of this process, in bytes.
    #[cfg(unix)]
    fn peak_rss() -> u64 {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut ru) },
            0
        );
        let maxrss = ru.ru_maxrss as u64;
        if cfg!(target_os = "macos") {
            maxrss
        } else {
            maxrss * 1024
        }
    }

    /// Files that claim gigabytes: each fails in inflate with CFITSIO's status,
    /// and opening all of them stays far below what they claim.
    #[test]
    fn test_size_claims_far_beyond_the_file() {
        // An empty gzip member whose ISIZE says 3.9 GB.
        let mut big = gzip(b"");
        let n = big.len();
        big[n - 4..].copy_from_slice(&3_897_424_373u32.to_le_bytes());
        // Over 10 000 bytes with an ISIZE of 0, so the C adds 4 GiB.
        let mut wrap = b"\x1f\x8b\x08\x00".to_vec();
        wrap.resize(10_041, 0);
        // A PKZIP header whose size field (offset 22) says 4 GB.
        let mut pkzip = b"PK\x03\x04".to_vec();
        pkzip.resize(100, 0);
        pkzip[22..26].copy_from_slice(&0xF000_0000u32.to_le_bytes());

        for (ext, bytes) in [("gz", big), ("gz", wrap), ("zip", pkzip)] {
            with_temp_file(|name| {
                let path = format!("{name}.{ext}");
                std::fs::write(&path, &bytes).unwrap();
                assert_eq!(
                    open_and_read(&path).0,
                    DATA_DECOMPRESSION_ERR,
                    "{} bytes",
                    bytes.len()
                );
            });
        }
        #[cfg(unix)]
        assert!(peak_rss() < 1 << 30, "peak RSS {} bytes", peak_rss());
    }

    /// A valid file that triggers the C's 4 GiB wrap correction: over 10 000
    /// bytes, with a small last member, so the trailer understates the size.
    /// Only the first member is read, as in CFITSIO.
    #[test]
    fn test_valid_multi_member_gzip_with_wrapped_size() {
        let n = 14_400usize;
        let mut fits = Vec::new();
        for card in [
            "SIMPLE  =                    T".to_string(),
            "BITPIX  =                    8".to_string(),
            "NAXIS   =                    1".to_string(),
            format!("NAXIS1  = {n:>20}"),
            "END".to_string(),
        ] {
            fits.extend(format!("{card:<80}").bytes());
        }
        fits.resize(2880, b' ');
        // Incompressible data, so the file is over 10 000 bytes compressed.
        let mut state: u32 = 12345;
        let data: Vec<u8> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect();
        fits.extend(&data);
        fits.resize(fits.len().div_ceil(2880) * 2880, 0);
        let mut gz = gzip(&fits);
        gz.extend(gzip(b"x"));
        assert!(gz.len() > 10_000);

        with_temp_file(|name| {
            let path = format!("{name}.gz");
            std::fs::write(&path, &gz).unwrap();
            let (status, pixels) = open_and_read(&path);
            assert_eq!(status, 0);
            assert!(pixels == data);
        });
        #[cfg(unix)]
        assert!(peak_rss() < 1 << 30, "peak RSS {} bytes", peak_rss());
    }
}
