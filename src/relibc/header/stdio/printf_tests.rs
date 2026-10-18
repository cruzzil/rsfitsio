//! Tests for the vendored `printf` implementation.
// Vendored code, documented in upstream's style.
#![allow(missing_docs)]

#[cfg(test)]
mod tests {

    use crate::{
        c_types::{c_char, c_int},
        relibc::header::stdio::{
            snprintf_cint, snprintf_f64, snprintf_f64_decim, sprintf_f64, sprintf_string_width,
        },
    };
    use bytemuck::cast_slice;
    use core::ffi::CStr;
    use libc;

    // Helper to convert string literals to c_char arrays
    macro_rules! c_str {
        ($s:literal) => {{
            const S: &str = concat!($s, "\0");
            core::mem::transmute::<*const u8, *const c_char>(S.as_ptr())
        }};
    }

    // Test the existing snprintf_cint function against libc
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_snprintf_cint() {
        unsafe {
            let value: c_int = 42;
            let mut rust_buffer: [c_char; 100] = [0; 100];

            let rust_result = snprintf_cint(&mut rust_buffer, 100, cast_slice(b"%d\0"), value);
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            assert_eq!(rust_result, 2, "Expected return value of 2 for '42'");

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%d");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result = libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, value);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(rust_str, libc_str, "Integer formatting mismatch for %d");
                assert_eq!(rust_result, libc_result, "Return value mismatch for %d");
            }

            assert_eq!(rust_str, "42");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_snprintf_f64() {
        unsafe {
            let value: f64 = 3.24159;
            let mut rust_buffer: [c_char; 100] = [0; 100];

            let rust_result = snprintf_f64(&mut rust_buffer, 100, cast_slice(b"%f\0"), value);
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            assert_eq!(
                rust_str, "3.241590",
                "Expected float formatting for 3.24159"
            );

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%f");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result = libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, value);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(rust_str, libc_str, "Float formatting mismatch for %f");
                assert_eq!(rust_result, libc_result, "Return value mismatch for %f");
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_sprintf_f64() {
        unsafe {
            let value: f64 = 2.81828;
            let mut rust_buffer: [c_char; 100] = [0; 100];

            let rust_result = sprintf_f64(&mut rust_buffer, cast_slice(b"%f\0"), value);
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            assert_eq!(
                rust_str, "2.818280",
                "Expected float formatting for 2.81828"
            );

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%f");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result = libc::sprintf(libc_buffer.as_mut_ptr(), format, value);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(
                    rust_str, libc_str,
                    "Float formatting mismatch for sprintf_f64"
                );
                assert_eq!(
                    rust_result, libc_result,
                    "Return value mismatch for sprintf_f64"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_snprintf_f64_decim() {
        unsafe {
            let value: f64 = 3.24159265;
            let decim: c_int = 2;
            let mut rust_buffer: [c_char; 100] = [0; 100];

            let rust_result =
                snprintf_f64_decim(&mut rust_buffer, 100, cast_slice(b"%.*f\0"), decim, value);
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            assert_eq!(rust_result, 4, "Expected return value of 4 for '3.24'");

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%.*f");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result =
                    libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, decim, value);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(rust_str, libc_str, "Precision float formatting mismatch");
                assert_eq!(
                    rust_result, libc_result,
                    "Return value mismatch for precision float"
                );
            }

            assert_eq!(rust_str, "3.24");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_sprintf_string_width() {
        unsafe {
            let width: c_int = 10;
            let mut rust_buffer: [c_char; 100] = [0; 100];

            let rust_result = sprintf_string_width(
                &mut rust_buffer,
                cast_slice(b"%*s\0"),
                width,
                cast_slice(b"test\0"),
            );
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            assert_eq!(
                rust_result, 10,
                "Expected return value of 10 for '      test'"
            );

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%*s");
                let value = c_str!("test");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result = libc::sprintf(libc_buffer.as_mut_ptr(), format, width, value);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(rust_str, libc_str, "String width formatting mismatch");
                assert_eq!(
                    rust_result, libc_result,
                    "Return value mismatch for string width"
                );
            }

            assert_eq!(rust_str, "      test");
        }
    }

    // Test edge cases
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_sprintf_edge_cases() {
        unsafe {
            // Test zero value
            let value: c_int = 0;
            let mut rust_buffer: [c_char; 100] = [0; 100];

            let rust_result = snprintf_cint(&mut rust_buffer, 100, cast_slice(b"%d\0"), value);
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            assert_eq!(rust_result, 1, "Expected return value of 1 for '0'");

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%d");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result = libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, value);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(rust_str, libc_str, "Zero value formatting mismatch");
                assert_eq!(rust_result, libc_result, "Return value mismatch for zero");
            }

            assert_eq!(rust_str, "0");

            // Test negative value
            let value: c_int = -42;
            rust_buffer = [0; 100];

            let rust_result = snprintf_cint(&mut rust_buffer, 100, cast_slice(b"%d\0"), value);
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            assert_eq!(rust_result, 3, "Expected return value of 3 for '-42'");

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%d");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result = libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, value);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(rust_str, libc_str, "Negative value formatting mismatch");
                assert_eq!(
                    rust_result, libc_result,
                    "Return value mismatch for negative"
                );
            }

            assert_eq!(rust_str, "-42");
        }
    }

    // Comprehensive integer tests
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_snprintf_integers_comprehensive() {
        unsafe {
            let test_values = [
                (1, "1"),
                (42, "42"),
                (123, "123"),
                (-1, "-1"),
                (-42, "-42"),
                (-123, "-123"),
                (2147483647, "2147483647"),   // INT_MAX
                (-2147483648, "-2147483648"), // INT_MIN
                (1000, "1000"),
                (-1000, "-1000"),
                (12345, "12345"),
                (-12345, "-12345"),
                (999999, "999999"),
                (-999999, "-999999"),
            ];

            for (value, expected) in test_values.iter() {
                let mut rust_buffer: [c_char; 100] = [0; 100];

                let rust_result = snprintf_cint(&mut rust_buffer, 100, cast_slice(b"%d\0"), *value);
                let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(
                    rust_result as usize,
                    expected.len(),
                    "Expected return value to match string length for {value}"
                );

                #[cfg(not(target_family = "windows"))]
                {
                    let format = c_str!("%d");
                    let mut libc_buffer: [c_char; 100] = [0; 100];
                    let libc_result = libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, *value);
                    let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                    assert_eq!(rust_str, libc_str, "Integer {value} formatting mismatch");
                    assert_eq!(
                        rust_result, libc_result,
                        "Return value mismatch for {value}"
                    );
                }

                assert_eq!(rust_str, *expected, "Expected output mismatch for {value}");
            }
        }
    }

    // Comprehensive float tests
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_snprintf_floats_comprehensive() {
        unsafe {
            let test_values = [
                (0.0, "0.000000"),
                (1.0, "1.000000"),
                (-1.0, "-1.000000"),
                (3.24159, "3.241590"),
                (-3.24159, "-3.241590"),
                (2.81828, "2.818280"),
                (123.456, "123.456000"),
                (-123.456, "-123.456000"),
                (0.000001, "0.000001"),
                (-0.000001, "-0.000001"),
                (1000000.0, "1000000.000000"),
                (-1000000.0, "-1000000.000000"),
            ];

            for (value, expected) in test_values.iter() {
                let mut rust_buffer: [c_char; 100] = [0; 100];

                let rust_result = snprintf_f64(&mut rust_buffer, 100, cast_slice(b"%f\0"), *value);
                let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(
                    rust_result as usize,
                    expected.len(),
                    "Expected return value to match string length for {value}"
                );

                #[cfg(not(target_family = "windows"))]
                {
                    let format = c_str!("%f");
                    let mut libc_buffer: [c_char; 100] = [0; 100];
                    let libc_result = libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, *value);
                    let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                    assert_eq!(rust_str, libc_str, "Float {value} formatting mismatch");
                    assert_eq!(
                        rust_result, libc_result,
                        "Return value mismatch for {value}"
                    );
                }

                assert_eq!(rust_str, *expected, "Expected output mismatch for {value}");
            }
        }
    }

    // Test precision specifiers
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_snprintf_precision_comprehensive() {
        unsafe {
            let test_cases = [
                (3.24159, 0, "3"),
                (3.24159, 1, "3.2"),
                (3.24159, 2, "3.24"),
                (3.24159, 3, "3.242"),
                (3.24159, 4, "3.2416"),
                (3.24159, 5, "3.24159"),
                (123.456, 0, "123"),
                (123.456, 1, "123.5"),
                (123.456, 2, "123.46"),
                (0.0, 0, "0"),
                (0.0, 3, "0.000"),
                (-2.5, 1, "-2.5"),
            ];

            for (value, precision, expected) in test_cases.iter() {
                let mut rust_buffer: [c_char; 100] = [0; 100];

                let rust_result = snprintf_f64_decim(
                    &mut rust_buffer,
                    100,
                    cast_slice(b"%.*f\0"),
                    *precision,
                    *value,
                );
                let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

                #[cfg(not(target_family = "windows"))]
                {
                    let format = c_str!("%.*f");
                    let mut libc_buffer: [c_char; 100] = [0; 100];
                    let libc_result =
                        libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, *precision, *value);
                    let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                    assert_eq!(
                        rust_str, libc_str,
                        "Precision {precision} for {value} formatting mismatch"
                    );
                    assert_eq!(
                        rust_result, libc_result,
                        "Return value mismatch for precision {precision} value {value}"
                    );
                }

                assert_eq!(
                    rust_result as usize,
                    expected.len(),
                    "Expected return value to match string length for precision {precision} value {value}"
                );

                assert_eq!(
                    rust_str, *expected,
                    "Expected output mismatch for precision {precision} value {value}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_many_test_cases() {
        unsafe {
            // Test case: 233 with %d should produce "233"
            let value: c_int = 233;
            let mut rust_buffer: [c_char; 100] = [0; 100];

            let rust_result = snprintf_cint(&mut rust_buffer, 100, cast_slice(b"%d\0"), value);
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%d");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result = libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, value);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(rust_str, libc_str, "test case mismatch for %d with 233");
                assert_eq!(
                    rust_result, libc_result,
                    "Return value mismatch for %d test"
                );
            }

            assert_eq!(rust_result, 3, "Expected return value of 3 for '233'");
            assert_eq!(rust_str, "233");

            // Test case: 233000L with %ld should produce "233000"
            // Note: Testing with the available integer function
            let value_long: c_int = 233000; // Using c_int for compatibility with snprintf_cint
            rust_buffer = [0; 100];

            let rust_result = snprintf_cint(&mut rust_buffer, 100, cast_slice(b"%d\0"), value_long);
            let rust_str = CStr::from_ptr(rust_buffer.as_ptr()).to_str().unwrap();

            #[cfg(not(target_family = "windows"))]
            {
                let format = c_str!("%d");
                let mut libc_buffer: [c_char; 100] = [0; 100];
                let libc_result = libc::snprintf(libc_buffer.as_mut_ptr(), 100, format, value_long);
                let libc_str = CStr::from_ptr(libc_buffer.as_ptr()).to_str().unwrap();

                assert_eq!(
                    rust_str, libc_str,
                    "Z88dk test case mismatch for large integer"
                );
                assert_eq!(
                    rust_result, libc_result,
                    "Return value mismatch for large integer"
                );
            }

            assert_eq!(rust_result, 6, "Expected return value of 6 for '233000'");
            assert_eq!(rust_str, "233000");
        }
    }
}

/// Differential sweep against the platform's C printf.
///
/// The conversions the crate uses for FITS output -- `%#14.6G' and `%#23.15G'
/// in `ffgcls', `%#8.2g'/`%#10.4g' in the fpack report -- are exactly the ones
/// where a %g implementation is easiest to get wrong, so rather than pin a
/// handful of expected strings this walks a matrix of specifications against a
/// matrix of values and demands byte-for-byte agreement with glibc, for both
/// the formatted text and the return value.
#[cfg(test)]
mod glibc_diff {
    use crate::c_types::{c_char, c_int};
    use crate::relibc::header::stdio::printf::{CustomVaList, VaArg};
    use crate::relibc::header::stdio::{snprintf_cint, snprintf_va};
    use bytemuck::cast_slice;
    use core::ffi::CStr;

    const BUF: usize = 512;

    fn ours(spec: &CStr, val: f64) -> (String, c_int) {
        let mut buf: [c_char; BUF] = [0; BUF];
        let mut args = CustomVaList::new();
        args.push(VaArg::c_double(val));
        let n = snprintf_va(&mut buf, BUF, cast_slice(spec.to_bytes_with_nul()), args);
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        (s, n)
    }

    #[cfg(not(target_family = "windows"))]
    fn theirs(spec: &CStr, val: f64) -> (String, c_int) {
        let mut buf: [c_char; BUF] = [0; BUF];
        let n = unsafe { libc::snprintf(buf.as_mut_ptr(), BUF, spec.as_ptr(), val) };
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        (s, n)
    }

    /* The values are the shapes fpack's image statistics and CFITSIO's column
    formatting actually produce: zero and signed zero, exact halves that
    exercise the rounding mode, the 9.999e-5 case where rounding to P
    significant digits moves the exponent across the fixed/scientific
    boundary, the extremes of the exponent range, and the non-finites that a
    division by a zero file size yields. */
    const VALUES: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        1.5,
        2.5,
        -2.5,
        0.1,
        1234.5,
        9.999e-5,
        1e-5,
        1e-4,
        0.000123456,
        3.14159265358979,
        100.0,
        1e5,
        123456.0,
        999999.0,
        999999.5,
        1e10,
        1e20,
        1e-300,
        1e300,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];

    /// The one place glibc, not this implementation, is the one that is wrong.
    ///
    /// C99 7.19.6.1 for `g`/`G`: "unless the # flag is used, any trailing zeros
    /// are removed from the fractional portion of the result".  With `#` they
    /// must be kept.  glibc keeps them -- except when the value has a fractional
    /// part *and* rounding to P significant digits carries into a new decade
    /// *and* the new exponent selects scientific style, where it emits a bare
    /// "1." mantissa:
    ///
    ///     printf("%#.6g", 999999.5)  glibc: 1.e+06     C99: 1.00000e+06
    ///     printf("%#.3g", 999.5)     glibc: 1.e+03     C99: 1.00e+03
    ///     printf("%#.2g", 99.5)      glibc: 1.e+02     C99: 1.0e+02
    ///
    /// It gets the same carry right when the style stays fixed (`%#.6g` of
    /// 999.9995 is "1000.00") and when the value is an exact integer (`%#.6g`
    /// of 9999995.0 is "1.00000e+07").  `test_g_alternate_carry_is_conforming`
    /// pins our behaviour; these pairs are excluded from the sweep.
    #[cfg(not(target_family = "windows"))]
    fn glibc_is_wrong(spec: &CStr, v: f64) -> bool {
        let s = spec.to_bytes();
        s.contains(&b'#') && s.iter().any(|&c| c == b'g' || c == b'G') && v == 999999.5
    }

    /* Report every divergence in one go rather than stopping at the first --
    a %g rewrite tends to break a whole family of inputs at once, and the
    shape of the family is what says which rule is wrong. */
    fn sweep(specs: &[&CStr]) {
        #[cfg(not(target_family = "windows"))]
        {
            let mut bad = Vec::new();
            for spec in specs {
                for &v in VALUES {
                    if glibc_is_wrong(spec, v) {
                        continue;
                    }
                    let (got, got_n) = ours(spec, v);
                    let (want, want_n) = theirs(spec, v);
                    let s = spec.to_str().unwrap();
                    if got != want || got_n != want_n {
                        bad.push(format!(
                            "printf(\"{s}\", {v:e}): ours [{got}] ({got_n}), glibc [{want}] ({want_n})"
                        ));
                    }
                }
            }
            assert!(
                bad.is_empty(),
                "{} divergence(s) from glibc:\n{}",
                bad.len(),
                bad.join("\n")
            );
        }
        /* keep the helper live on windows, where there is no glibc to diff
        against and the CRT's own conversions differ */
        #[cfg(target_family = "windows")]
        let _ = specs;
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_g_matches_glibc() {
        sweep(&[
            c"%g", c"%G", c"%.0g", c"%.1g", c"%.2g", c"%.3g", c"%.6g", c"%.15g", c"%.17g",
            c"%8.2g", c"%-8.2g", c"%08.2g", c"%20.15g",
        ]);
    }

    /// The `#' alternate flag: keep trailing zeros, and always keep the radix
    /// point.  `ffgcls' formats numeric columns as strings with `%#14.6G' and
    /// `%#23.15G', so this is load-bearing for the library, not just fpack.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_g_alternate_matches_glibc() {
        sweep(&[
            c"%#g",
            c"%#G",
            c"%#.0g",
            c"%#.1g",
            c"%#.2g",
            c"%#.3g",
            c"%#.6g",
            c"%#8.2g",
            c"%#7.3g",
            c"%#10.4g",
            c"%#-10.4g",
            c"%#14.6G",
            c"%#23.15G",
        ]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_e_matches_glibc() {
        sweep(&[
            c"%e", c"%E", c"%.0e", c"%#.0e", c"%.3e", c"%#.3e", c"%12.4e", c"%-12.4e", c"%012.4e",
            c"%23.15E", c"%14.6E",
        ]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_f_matches_glibc() {
        sweep(&[
            c"%f", c"%F", c"%.0f", c"%#.0f", c"%.3f", c"%10.3f", c"%-10.3f", c"%010.3f", c"%5.3f",
            c"%6.0f", c"%8.1f", c"%#5.1f", c"%#6.2f", c"%11.0f",
        ]);
    }
    /// `#` with %g must keep the trailing zeros even when rounding to P
    /// significant digits carries into a new decade.  glibc does not (see
    /// `glibc_is_wrong`), so these expectations come from C99 7.19.6.1 rather
    /// than from the platform.
    #[test]
    fn test_g_alternate_carry_is_conforming() {
        assert_eq!(ours(c"%#.6g", 999999.5).0, "1.00000e+06");
        assert_eq!(ours(c"%#.3g", 999.5).0, "1.00e+03");
        assert_eq!(ours(c"%#.2g", 99.5).0, "1.0e+02");
        assert_eq!(ours(c"%#14.6G", 999999.5).0, "   1.00000E+06");
        /* the cases glibc agrees on, kept here so a regression in either
        direction is visible in one place */
        assert_eq!(ours(c"%#.6g", 999.9995).0, "1000.00");
        assert_eq!(ours(c"%#.6g", 9999995.0).0, "1.00000e+07");
        assert_eq!(ours(c"%#.6g", 9.9999995).0, "10.0000");
    }

    /// A field width over INT_MAX is EOVERFLOW (-1), as in C, however many
    /// digits it has; one just under it pads without writing byte by byte.
    #[test]
    fn test_huge_field_width() {
        let mut buf: [c_char; 8] = [0; 8];
        for fmt in [
            &b"%3000000000d\0"[..],
            b"%99999999999999999999999d\0",
            b"%.3000000000d\0",
        ] {
            assert_eq!(snprintf_cint(&mut buf, 8, cast_slice(fmt), 42), -1);
        }
        let n = snprintf_cint(&mut buf, 8, cast_slice(b"%2000000000d\0"), 42);
        assert_eq!(n, 2_000_000_000);
        assert_eq!(cast_slice::<c_char, u8>(&buf), b"       \0");
    }
}
