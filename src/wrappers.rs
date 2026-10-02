//! Safe replacements for the C string and number-parsing functions.
//!
//! The port calls these instead of `libc`. Each takes a `&[c_char]` slice and
//! returns an index or a value rather than an interior pointer, so the whole
//! `strcpy` / `strcmp` / `strlen` family is expressible without `unsafe`. The
//! `_safe` suffix marks them as the safe counterpart of the C function of the
//! same name.
//!
//! The slices are still C strings: they are expected to hold a NUL, and the
//! routines stop there rather than at the end of the buffer, so trailing
//! padding in a fixed-size FITS buffer is ignored.
//!
//! Several of these were taken from the Redox relibc project,
//! <https://gitlab.redox-os.org/redox-os/relibc>.
#![warn(missing_docs)]

use core::{cmp, ffi::CStr, str::FromStr};

use bytemuck::cast_slice;

use crate::c_types::{c_char, c_int, c_long};

use crate::bb;

/// Copies at most `n` characters of the NUL-terminated `src` into `dst`,
/// padding the remainder of the `n` with NULs. The safe counterpart of C
/// `strncpy`.
///
/// # Panics
///
/// Panics if `n` exceeds the length of `dst`.
pub fn strncpy_safe(dst: &mut [c_char], src: &[c_char], n: usize) {
    let mut i = 0;

    assert!(n <= dst.len());

    while i < n && src[i] != 0 {
        dst[i] = src[i];
        i += 1;
    }

    for i in i..n {
        dst[i] = 0;
    }
}

/// Length of the initial run of `cs` made up of characters *not* in the set
/// `ct`. The safe counterpart of C `strcspn`.
pub fn strcspn_safe(cs: &[c_char], ct: &[c_char]) -> usize {
    strcspn_inner(cs, ct, false)
}

/// Shared implementation of [`strcspn_safe`] and [`strspn_safe`]: counts the
/// leading characters of `cs` whose membership of `ct` equals `cmp`.
fn strcspn_inner(cs: &[c_char], ct: &[c_char], cmp: bool) -> usize {
    let mut it = 0;
    while ct[it] != 0 {
        it += 1;
    }

    let mut is = 0;
    while cs[is] != 0 {
        if ct[..it].contains(&cs[is]) != cmp {
            return is;
        }
        is += 1;
    }

    is
}

/// Length of the initial run of `cs` made up entirely of characters in the set
/// `ct`. The safe counterpart of C `strspn`.
pub fn strspn_safe(cs: &[c_char], ct: &[c_char]) -> usize {
    strcspn_inner(cs, ct, true)
}

/// Copies the NUL-terminated `src` into `dst`, including the terminator. The
/// safe counterpart of C `strcpy`.
pub fn strcpy_safe(dst: &mut [c_char], src: &[c_char]) {
    let mut i = 0;
    loop {
        dst[i] = src[i];
        if src[i] == 0 {
            break;
        }
        i += 1;
    }
}

/// Compares two NUL-terminated strings. The safe counterpart of C `strcmp`.
///
/// # Returns
///
/// Negative, zero or positive as `cs` sorts before, equal to, or after `ct`.
pub fn strcmp_safe(cs: &[c_char], ct: &[c_char]) -> c_int {
    strncmp_safe(cs, ct, cmp::max(cs.len(), ct.len()))
}

/// Compares at most `n` characters of two NUL-terminated strings. The safe
/// counterpart of C `strncmp`.
///
/// # Returns
///
/// Negative, zero or positive as `cs` sorts before, equal to, or after `ct`.
pub fn strncmp_safe(cs: &[c_char], ct: &[c_char], n: usize) -> c_int {
    let min_len = cmp::min(cs.len(), ct.len());
    let min_n = cmp::min(n, min_len);
    for i in 0..min_n {
        let a = cs[i];
        let b = ct[i];

        if a != b || a == 0 {
            return c_int::from(a) - c_int::from(b);
        }
    }

    // If we reached here, all bytes so far are equal but we haven't reached
    // the end of either string. So we need to check if one of them is
    // shorter than the other.
    if min_len < n {
        // We reached the end of one string, but not the other
        return if cs.len() < ct.len() {
            -c_int::from(ct[min_len])
        } else {
            c_int::from(cs[min_len])
        };
    }

    // If we reached here, all bytes are equal and we reached the end of both

    0
}

/// Length of a `CStr`, not counting the NUL terminator. The `CStr` form of
/// [`strlen_safe`].
pub(crate) fn strlen_safe_cstr(str: &CStr) -> usize {
    str.to_bytes().len()
}

/// Length of the NUL-terminated string `cs`, not counting the terminator.
///
/// Panics if `cs` holds no NUL at all -- that is a malformed C string, and
/// every caller here goes on to index the buffer expecting one.
pub fn strlen_safe(cs: &[c_char]) -> usize {
    match strnlen_safe(cs, cs.len()) {
        n if n < cs.len() => n,
        _ => panic!("Invalid C Style String"),
    }
}

/// Length of `cs`, stopping at the first NUL or after `n` characters,
/// whichever comes first. Unlike [`strlen_safe`] this never panics: a buffer
/// with no terminator within the limit simply reports the limit.
pub fn strnlen_safe(cs: &[c_char], n: usize) -> usize {
    let limit = cmp::min(n, cs.len());
    cs[..limit].iter().position(|&c| c == 0).unwrap_or(limit)
}

/// Parses a leading integer out of `input`, skipping leading whitespace and any
/// non-numeric prefix. The safe counterpart of C `strtol`.
///
/// # Returns
///
/// The parsed value and the index one past its last digit, or the parse error.
pub(crate) fn strtol_safe<F: FromStr>(input: &[c_char]) -> Result<(F, usize), <F as FromStr>::Err> {
    let strlen = input.len().saturating_sub(1);
    let input: &[u8] = cast_slice(input);

    // Find the first non-numeric character or the end of the string
    let mut start = 0;
    let mut end = 0;

    while start < strlen && (input[start] as char).is_whitespace() {
        start += 1;
    }

    // ASCII digits only: `char::is_numeric` also accepts bytes such as 0xB2
    // (superscript two in Latin-1), which are not UTF-8 on their own.
    while start < strlen
        && !input[start].is_ascii_digit()
        && (input[start] as char) != '+'
        && (input[start] as char) != '-'
    {
        start += 1;
    }

    end = (start + 1).min(input.len());

    while end < strlen && input[end].is_ascii_digit() {
        end += 1;
    }

    // Not text at all: let `parse` report it, as it would any other non-number.
    let str = str::from_utf8(&input[start..end]).unwrap_or("");

    let res = str.parse::<F>()?;

    Ok((res, end))
}

/// Parses a leading floating point number out of `s`. The safe counterpart of C
/// `strtod`.
///
/// # Parameters
///
/// * `s`    — (I) the string to parse
/// * `endp` — (O) index one past the last character consumed
pub fn strtod_safe(s: &[c_char], endp: &mut usize) -> f64 {
    strto_float_impl(s, endp)
}

/// Index of the first occurrence of `c` in `cs`, or `None` if not found. The
/// safe counterpart of C `strchr`, returning an index rather than an interior
/// pointer.
///
/// Unlike [`strrchr_safe`], the search covers the whole slice rather than
/// stopping at the NUL, so passing `0` for `c` finds the terminator.
pub fn strchr_safe(cs: &[c_char], c: c_char) -> Option<usize> {
    cs.iter().position(|&x| x == c)
}

/// Index of the last occurrence of `c` in the NUL-terminated string `cs`, or
/// `None` if not found. Mirrors C `strrchr`: the search stops at the terminating
/// NUL so trailing buffer padding is ignored.
pub fn strrchr_safe(cs: &[c_char], c: c_char) -> Option<usize> {
    let len = strlen_safe(cs);
    cs[..len].iter().rposition(|&x| x == c)
}

/// Index of the first occurrence of the NUL-terminated needle `ct` in the
/// NUL-terminated haystack `cs`, or `None` if not present. Mirrors C `strstr`:
/// an empty needle matches at 0, and neither string is read past its NUL, so
/// trailing buffer padding is ignored.
pub fn strstr_safe(cs: &[c_char], ct: &[c_char]) -> Option<usize> {
    let hay = &cs[..strlen_safe(cs)];
    let needle = &ct[..strlen_safe(ct)];

    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > hay.len() {
        return None;
    }

    hay.windows(needle.len()).position(|w| w == needle)
}

/// Appends the NUL-terminated `ct` to the NUL-terminated `s`. The safe
/// counterpart of C `strcat`.
pub fn strcat_safe(s: &mut [c_char], ct: &[c_char]) {
    let ct_len = strlen_safe(ct);

    strncat_safe(s, ct, ct_len);
}

/// Appends at most `n` characters of the NUL-terminated `ct` to the
/// NUL-terminated `s`, adding a terminator. The safe counterpart of C
/// `strncat`.
///
/// Where C would write past the end of `s`, the result is truncated to fit
/// instead. Several of CFITSIO's error messages overflow their buffers this way
/// (`ffc2j` appends 30 characters of the bad value to a 52-character prefix in an
/// 81-byte buffer), and the value comes from the file being read.
pub fn strncat_safe(s: &mut [c_char], ct: &[c_char], n: usize) {
    let s_len = strlen_safe(s);
    let ct_len = strlen_safe(ct);

    // `s_len < s.len()`: strlen_safe has found a terminator inside `s`.
    let n = cmp::min(cmp::min(n, ct_len), s.len() - 1 - s_len);
    let mut i = 0;
    while i < n {
        let b = ct[i];
        if b == 0 {
            break;
        }

        s[s_len + i] = b;
        i += 1;
    }
    s[s_len + i] = 0;
}

/// Maps an ASCII lower-case letter to upper case, leaving anything else
/// unchanged. The safe counterpart of C `toupper`.
pub(crate) fn toupper(c: c_char) -> c_char {
    if islower(c) {
        return c & 0x5f;
    }
    c
}

/// Maps an ASCII upper-case letter to lower case, leaving anything else
/// unchanged. The safe counterpart of C `tolower`.
pub(crate) fn tolower(c: c_char) -> c_char {
    if isupper(c) {
        return c | 0x20;
    }
    c
}

/// Whether `c` is an ASCII lower-case letter. The safe counterpart of C
/// `islower`.
pub(crate) fn islower(c: c_char) -> bool {
    (c >= 97) && (c <= 122)
}

/// Whether `c` is an ASCII upper-case letter. The safe counterpart of C
/// `isupper`.
pub(crate) fn isupper(c: c_char) -> bool {
    (c >= 65) && (c <= 90)
}

/// Whether `c` is ASCII whitespace. The safe counterpart of C `isspace`.
pub(crate) fn isspace(c: c_char) -> bool {
    let c = c as c_char;
    c == bb(b' ') || c == bb(b'\t') || c == bb(b'\n') || c == bb(b'\r') || c == 0x0b || c == 0x0c
}

/// C `atol`: leading optional whitespace and sign, then decimal digits.
///
/// Parses `cs` as a `long`. The safe counterpart of C `atol`.
///
/// Matches the C on the two edge cases Rust's `parse` gets differently:
/// a value that does not fit saturates to `c_long::MIN`/`MAX` rather than
/// failing, and input with no digits at all yields 0.
pub fn atol_safe(cs: &[c_char]) -> c_long {
    let bytes: &[u8] = cast_slice(cs);
    let mut i = 0;

    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }

    let negative = match bytes.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        _ => false,
    };

    let mut acc: c_long = 0;
    let mut overflow = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        let d = c_long::from(bytes[i] - b'0');
        /* Accumulate negatively so that c_long::MIN is representable. */
        match acc.checked_mul(10).and_then(|a| a.checked_sub(d)) {
            Some(next) => acc = next,
            None => overflow = true,
        }
        i += 1;
    }

    if overflow {
        return if negative { c_long::MIN } else { c_long::MAX };
    }

    /* acc is accumulated negatively so that c_long::MIN is representable, but
    that means a positive value of exactly |MIN| lands on MIN without the
    checked arithmetic above noticing. Negating it would overflow, so
    saturate as the C does. */
    if negative {
        acc
    } else {
        acc.checked_neg().unwrap_or(c_long::MAX)
    }
}

/// Parses `cs` as a `double`, yielding 0.0 rather than an error on anything
/// unparseable. The safe counterpart of C `atof`.
pub fn atof_safe(cs: &[c_char]) -> f64 {
    let mut dummy = 0;
    strtod_safe(cs, &mut dummy)
}

/// Whether `c` is an ASCII decimal digit. The safe counterpart of C `isdigit`.
pub fn isdigit_safe(c: c_char) -> bool {
    (b'0' as c_char) <= c && c <= (b'9' as c_char)
}

/// Index of the first character of `cs` that is in the set `ct`, or `None` if
/// none is. The safe counterpart of C `strpbrk`, returning an index rather than
/// an interior pointer.
pub fn strpbrk_safe(cs: &[c_char], ct: &[c_char]) -> Option<usize> {
    let i = strcspn_safe(cs, ct);
    (cs[i] != 0).then_some(i)
}

/// Parses a leading floating point number, the shared implementation behind
/// [`strtod_safe`].
///
/// Copied and modified from Redox's relibc,
/// <https://gitlab.redox-os.org/redox-os/relibc/-/blob/master/src/macros.rs>.
fn strto_float_impl(s: &[c_char], endptr: &mut usize) -> f64 {
    let mut s = s;

    let mut si = 0;
    while isspace(s[si]) {
        si += 1;
    }

    /* Start of the literal itself, sign included: the digit-by-digit
    accumulation below is not correctly rounded (it compounds a rounding
    error per fractional digit, so e.g. 12345.6789 comes out as
    12345.678899999999), so once the extent of the literal is known the
    decimal case re-parses this span with Rust's correctly-rounded parser
    and only falls back to the accumulated value if that fails. */
    let literal_start = si;

    let mut result: f64 = 0.0;
    let mut exponent: Option<f64> = None;
    let mut radix = 10;
    let mut is_inf_or_nan = false;

    let result_sign = match s[si] as u8 {
        b'-' => {
            si += 1;
            -1.0
        }
        b'+' => {
            si += 1;
            1.0
        }
        _ => 1.0,
    };

    /* `s` is already a slice, so find the terminator inside it rather than
    walking off the end from a raw pointer.  A buffer with no NUL is read to its
    end instead of overrunning it. */
    let bytes: &[u8] = cast_slice(s);
    let rust_s = String::from_utf8_lossy(&bytes[..strnlen_safe(s, s.len())]);

    // detect NaN, Inf
    if rust_s.to_lowercase().starts_with("inf") {
        result = f64::INFINITY;
        si += 3;
        is_inf_or_nan = true;
    } else if rust_s.to_lowercase().starts_with("nan") {
        // we cannot signal negative NaN in LLVM backed languages
        // https://github.com/rust-lang/rust/issues/73328 , https://github.com/rust-lang/rust/issues/81261
        result = f64::NAN;
        si += 3;
        is_inf_or_nan = true;
    } else {
        if s[si] as u8 == b'0' && s[si + 1] as u8 == b'x' {
            si += 2;
            radix = 16;
        }

        while let Some(digit) = (s[si] as u8 as char).to_digit(radix) {
            result *= f64::from(radix);
            result += f64::from(digit);
            si += 1;
        }

        if s[si] as u8 == b'.' {
            si += 1;

            let mut i = 1.0;
            while let Some(digit) = (s[si] as u8 as char).to_digit(radix) {
                i *= f64::from(radix);
                result += f64::from(digit) / i;
                si += 1;
            }
        }

        let s_before_exponent = s;

        exponent = match (s[si] as u8, radix) {
            (b'e' | b'E', 10) | (b'p' | b'P', 16) => {
                si += 1;

                let is_exponent_positive = match s[si] as u8 {
                    b'-' => {
                        si += 1;
                        false
                    }
                    b'+' => {
                        si += 1;
                        true
                    }
                    _ => true,
                };

                // Exponent digits are always in base 10.
                if (s[si] as u8 as char).is_ascii_digit() {
                    let mut exponent_value = 0;

                    while let Some(digit) = (s[si] as u8 as char).to_digit(10) {
                        exponent_value *= 10;
                        exponent_value += digit;
                        si += 1;
                    }

                    let exponent_base = match radix {
                        10 => 10u128,
                        16 => 2u128,
                        _ => unreachable!(),
                    };

                    /* Saturate rather than panic: `1e400` overflows a u128
                    pow, and the C just returns inf/0 for out-of-range
                    exponents. The correctly-rounded re-parse below produces
                    the real value whenever the literal is well-formed. */
                    let magnitude = exponent_base
                        .checked_pow(exponent_value)
                        .map_or(f64::INFINITY, |m| m as f64);
                    if is_exponent_positive {
                        Some(magnitude)
                    } else {
                        Some(1.0 / magnitude)
                    }
                } else {
                    // Exponent had no valid digits after 'e'/'p' and '+'/'-', rollback
                    s = s_before_exponent;
                    None
                }
            }
            _ => None,
        };
    }

    *endptr = si;

    /* Correctly-rounded re-parse of the decimal case; see literal_start. */
    if !is_inf_or_nan
        && radix == 10
        && let Ok(text) = str::from_utf8(cast_slice(&s[literal_start..si]))
        && let Ok(parsed) = text.parse::<f64>()
    {
        return parsed;
    }

    if let Some(exponent) = exponent {
        result_sign * result * exponent
    } else {
        result_sign * result
    }
}

/// Read into `buf` until it is full or end-of-file is reached, returning the
/// total number of bytes read.
///
/// `Read::read` is permitted to return fewer bytes than requested; a short read
/// is not an error and happens in practice on NFS/FUSE mounts, when a signal
/// interrupts the call, and under emulation.  Callers that transliterate C's
/// `fread` semantics ("I asked for N bytes, so a result < N means truncation")
/// need this loop, otherwise a short read is misreported as a corrupt file.
// clippy points ErrorKind at core::io, but that re-export is still behind the
// unstable `core_io` feature, so the std path has to stay.
#[allow(clippy::std_instead_of_core)]
pub(crate) fn read_fill<R: std::io::Read + ?Sized>(
    reader: &mut R,
    buf: &mut [u8],
) -> std::io::Result<usize> {
    let mut nread = 0;
    while nread < buf.len() {
        match reader.read(&mut buf[nread..]) {
            Ok(0) => break, /* end of file */
            Ok(n) => nread += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(nread)
}

#[cfg(test)]
mod tests {

    use crate::c_types::*;
    use crate::cs;
    use crate::wrappers::*;

    /// A byte such as 0xB2 is numeric to `char::is_numeric` (superscript two in
    /// Latin-1) but is not UTF-8 by itself; it used to reach `from_utf8(..).unwrap()`.
    #[test]
    fn test_strtol_non_ascii_is_an_error_not_a_panic() {
        for input in [&b"\xb2\0"[..], b"  \xb2\xb3\0", b"7\xb2\0", b"\0", b""] {
            let s: &[c_char] = bytemuck::cast_slice(input);
            let _ = strtol_safe::<c_long>(s);
        }
        let s: &[c_char] = bytemuck::cast_slice(b"\xb2\0");
        assert!(strtol_safe::<c_long>(s).is_err());
        let s: &[c_char] = bytemuck::cast_slice(b"7\xb2\0");
        assert_eq!(strtol_safe::<c_long>(s).unwrap(), (7, 1));
    }

    /// C's strncat would write past the destination; this truncates instead.
    #[test]
    fn test_strncat_truncates_at_the_destination() {
        let mut dst = [0 as c_char; 8];
        strcpy_safe(&mut dst, cs!(c"abcde"));
        strncat_safe(&mut dst, cs!(c"0123456789"), 30);
        assert_eq!(&dst, bytemuck::cast_slice::<u8, c_char>(b"abcde01\0"));
        // A full buffer takes nothing more, and keeps its terminator.
        strncat_safe(&mut dst, cs!(c"x"), 1);
        assert_eq!(&dst, bytemuck::cast_slice::<u8, c_char>(b"abcde01\0"));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_strtol() {
        let s: &[c_char] = bytemuck::cast_slice(b"12345 XC");
        let mut endpu: *mut c_char = core::ptr::null_mut();

        let ru = unsafe { libc::strtol(s.as_ptr(), &raw mut endpu, 10) };
        let (rs, endps): (c_longlong, usize) = strtol_safe(s).unwrap();

        assert_eq!(ru, 12345);
        unsafe {
            assert_eq!(endpu.offset_from(s.as_ptr()), 5);
        }
        assert_eq!(rs, 12345);
        assert_eq!(endps, 5)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_strtol_safer_vs_strtol() {
        // Test with leadng whitespace
        let s: &[c_char] = bytemuck::cast_slice(b"   12345 XC\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let (ru, endp) = strtol_safe(s).unwrap();
        assert_eq!(ru, 12345);
        assert_eq!(endp, 8); // 3 spaces + 5 digits
        assert_eq!(rs, ru);
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with leading zeros
        let s: &[c_char] = bytemuck::cast_slice(b"00012345 XC\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let (ru, endp) = strtol_safe(s).unwrap();
        assert_eq!(ru, 12345);
        assert_eq!(endp, 8); // 3 zeros + 5 digits
        assert_eq!(rs, ru);
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with negative number
        let s: &[c_char] = bytemuck::cast_slice(b"-12345 XC\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let (ru, endp) = strtol_safe(s).unwrap();
        assert_eq!(ru, -12345);
        assert_eq!(endp, 6); // 1 minus + 5 digits
        assert_eq!(rs, ru);
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with invalid characters
        let s: &[c_char] = bytemuck::cast_slice(b"12345a XC\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let (ru, endp) = strtol_safe(s).unwrap();
        assert_eq!(ru, 12345);
        assert_eq!(endp, 5); // 5 digits
        assert_eq!(rs, ru);
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with empty string
        let s: &[c_char] = bytemuck::cast_slice(b"\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let _rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let r = strtol_safe::<c_longlong>(s);
        let endp = 0;
        assert!(r.is_err());
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with only whitespace
        let s: &[c_char] = bytemuck::cast_slice(b"   \0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let _rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let r = strtol_safe::<c_longlong>(s);
        let endp = 0;
        assert!(r.is_err());
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with only invalid characters
        let s: &[c_char] = bytemuck::cast_slice(b"abcde\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let _rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let r = strtol_safe::<c_longlong>(s);
        let endp = 0;
        assert!(r.is_err());
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with leading zeros and invalid characters
        let s: &[c_char] = bytemuck::cast_slice(b"00012345a XC\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let (ru, endp) = strtol_safe(s).unwrap();
        assert_eq!(ru, 12345);
        assert_eq!(endp, 8); // 3 zeros + 5 digits
        assert_eq!(rs, ru);
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with negative number and invalid characters
        let s: &[c_char] = bytemuck::cast_slice(b"-12345a XC\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let (ru, endp) = strtol_safe(s).unwrap();
        assert_eq!(ru, -12345);
        assert_eq!(endp, 6); // 1 minus + 5 digits
        assert_eq!(rs, ru);
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);

        // Test with leading whitespace and invalid characters
        let s: &[c_char] = bytemuck::cast_slice(b"   12345a XC\0");
        let mut endps: *mut c_char = core::ptr::null_mut();
        let rs = unsafe { libc::strtol(s.as_ptr(), &raw mut endps, 10) };
        let (ru, endp) = strtol_safe(s).unwrap();
        assert_eq!(ru, 12345);
        assert_eq!(endp, 8); // 3 spaces + 5 digits
        assert_eq!(rs, ru);
        assert_eq!(endp, unsafe { endps.offset_from(s.as_ptr()) } as usize);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_strlen() {
        let s: &[c_char] = bytemuck::cast_slice(b"12345 XC\0");

        let ru = unsafe { libc::strlen(s.as_ptr()) };
        let rs = strlen_safe(s);
        assert_eq!(ru, rs);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_toupper() {
        let ru = unsafe { libc::toupper(c_int::from(b'a')) as c_char };
        let rs = toupper(b'a' as c_char);

        assert_eq!(ru, rs);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_islower() {
        let ru = unsafe { libc::islower(c_int::from(b'a')) > 0 };
        let rs = islower(b'a' as c_char);
        assert_eq!(ru, rs);

        let ru = unsafe { libc::islower(c_int::from(b'A')) > 0 };
        let rs = islower(b'A' as c_char);
        assert_eq!(ru, rs);

        let ru = unsafe { libc::islower(c_int::from(b'.')) > 0 };
        let rs = islower(b'.' as c_char);
        assert_eq!(ru, rs);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_strncmp_safe() {
        let s1: &[c_char] = bytemuck::cast_slice(b"hello\0");
        let s2: &[c_char] = bytemuck::cast_slice(b"hello\0");
        let s3: &[c_char] = bytemuck::cast_slice(b"world\0");
        let s4: &[c_char] = bytemuck::cast_slice(b"worl\0");

        let rs = strncmp_safe(s1, s2, 5);
        assert_eq!(rs, 0);

        let rs = strncmp_safe(s1, s3, 5);
        assert!(rs < 0);

        let rs = strncmp_safe(s3, s1, 5);
        assert!(rs > 0);

        let rs = strncmp_safe(s1, s2, 3);
        assert_eq!(rs, 0);

        let rs = strncmp_safe(s1, s3, 3);
        assert!(rs < 0);

        let rs = strncmp_safe(s3, s4, 6);
        assert!(rs > 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_strncmp_strncmp_safe() {
        let s1: &[c_char] = bytemuck::cast_slice(b"hello\0");
        let s2: &[c_char] = bytemuck::cast_slice(b"hello\0");
        let s3: &[c_char] = bytemuck::cast_slice(b"world\0");
        let s4: &[c_char] = bytemuck::cast_slice(b"worl\0");

        let ru = unsafe { libc::strncmp(s1.as_ptr(), s2.as_ptr(), 5) };
        let rs = strncmp_safe(s1, s2, 5);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strncmp(s1.as_ptr(), s3.as_ptr(), 5) };
        let rs = strncmp_safe(s1, s3, 5);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strncmp(s3.as_ptr(), s1.as_ptr(), 5) };
        let rs = strncmp_safe(s3, s1, 5);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strncmp(s1.as_ptr(), s2.as_ptr(), 3) };
        let rs = strncmp_safe(s1, s2, 3);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strncmp(s1.as_ptr(), s3.as_ptr(), 3) };
        let rs = strncmp_safe(s1, s3, 3);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strncmp(s3.as_ptr(), s4.as_ptr(), 6) };
        let rs = strncmp_safe(s3, s4, 6);
        assert_eq!(ru.signum(), rs.signum());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_strcmp_strcmp_safe() {
        let s1: &[c_char] = bytemuck::cast_slice(b"hello\0");
        let s2: &[c_char] = bytemuck::cast_slice(b"hello\0");
        let s3: &[c_char] = bytemuck::cast_slice(b"world\0");
        let s4: &[c_char] = bytemuck::cast_slice(b"worl\0");

        let ru = unsafe { libc::strcmp(s1.as_ptr(), s2.as_ptr()) };
        let rs = strcmp_safe(s1, s2);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strcmp(s1.as_ptr(), s3.as_ptr()) };
        let rs = strcmp_safe(s1, s3);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strcmp(s3.as_ptr(), s1.as_ptr()) };
        let rs = strcmp_safe(s3, s1);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strcmp(s1.as_ptr(), s2.as_ptr()) };
        let rs = strcmp_safe(s1, s2);
        assert_eq!(ru.signum(), rs.signum());

        let ru = unsafe { libc::strcmp(s3.as_ptr(), s4.as_ptr()) };
        let rs = strcmp_safe(s3, s4);
        assert_eq!(ru.signum(), rs.signum());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_strcmp_strcmp_safe_shorter() {
        let s1: &[c_char] = bytemuck::cast_slice(b"..\0");
        let s2: &[c_char] = bytemuck::cast_slice(b".\0");

        // Without nulls
        let s3: &[c_char] = bytemuck::cast_slice(b"..");
        let s4: &[c_char] = bytemuck::cast_slice(b".");

        let ru = unsafe { libc::strcmp(s1.as_ptr(), s2.as_ptr()) };
        let rs = strcmp_safe(s3, s4);
        assert_eq!(ru.signum(), rs.signum());
    }

    #[test]
    fn test_strnlen_safe() {
        let s: &[c_char] = bytemuck::cast_slice(b"hello\0world\0");

        assert_eq!(strnlen_safe(s, 12), 5);
        assert_eq!(strnlen_safe(s, 5), 5);
        // stops at the limit before reaching the NUL
        assert_eq!(strnlen_safe(s, 3), 3);
        assert_eq!(strnlen_safe(s, 0), 0);
        // limit past the end of the slice is clamped, not a panic
        assert_eq!(strnlen_safe(s, usize::MAX), 5);

        // no terminator within the slice at all -- reports the whole length
        let unterminated: &[c_char] = bytemuck::cast_slice(b"abcd");
        assert_eq!(strnlen_safe(unterminated, usize::MAX), 4);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_strnlen_strnlen_safe() {
        let s: &[c_char] = bytemuck::cast_slice(b"hello\0");

        for n in 0..=6 {
            let ru = unsafe { libc::strnlen(s.as_ptr(), n) };
            assert_eq!(ru, strnlen_safe(s, n), "n = {n}");
        }
    }

    #[test]
    #[should_panic(expected = "Invalid C Style String")]
    fn test_strlen_safe_rejects_unterminated() {
        let s: &[c_char] = bytemuck::cast_slice(b"abcd");
        strlen_safe(s);
    }

    #[test]
    fn test_strstr_safe() {
        let hay: &[c_char] = bytemuck::cast_slice(b"the quick brown fox\0");

        assert_eq!(strstr_safe(hay, bytemuck::cast_slice(b"quick\0")), Some(4));
        assert_eq!(strstr_safe(hay, bytemuck::cast_slice(b"the\0")), Some(0));
        assert_eq!(strstr_safe(hay, bytemuck::cast_slice(b"fox\0")), Some(16));
        assert_eq!(strstr_safe(hay, bytemuck::cast_slice(b"cat\0")), None);
        /* C strstr: the empty needle matches at position 0 */
        assert_eq!(strstr_safe(hay, bytemuck::cast_slice(b"\0")), Some(0));
        /* needle longer than the haystack */
        assert_eq!(
            strstr_safe(
                bytemuck::cast_slice(b"ab\0"),
                bytemuck::cast_slice(b"abc\0")
            ),
            None
        );
        /* overlapping prefix must not stop the scan early */
        assert_eq!(
            strstr_safe(
                bytemuck::cast_slice(b"aaab\0"),
                bytemuck::cast_slice(b"aab\0")
            ),
            Some(1)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_strstr_strstr_safe() {
        /* Padding after the NUL must be ignored, exactly as C strstr does. */
        let mut hay = [0 as c_char; 32];
        hay[..20].copy_from_slice(bytemuck::cast_slice(b"the quick brown fox\0"));
        hay[20..].copy_from_slice(bytemuck::cast_slice(b"cat and dog\0"));

        for needle in [&b"quick\0"[..], b"fox\0", b"cat\0", b"the\0", b"\0"] {
            let needle: &[c_char] = bytemuck::cast_slice(needle);
            let ru = unsafe { libc::strstr(hay.as_ptr(), needle.as_ptr()) };
            let expected = if ru.is_null() {
                None
            } else {
                Some(unsafe { ru.offset_from(hay.as_ptr()) } as usize)
            };
            assert_eq!(strstr_safe(&hay, needle), expected, "needle {needle:?}");
        }
    }

    #[test]
    fn test_strpbrk_safe() {
        let s: &[c_char] = bytemuck::cast_slice(b"hello, world\0");

        assert_eq!(strpbrk_safe(s, bytemuck::cast_slice(b",;\0")), Some(5));
        assert_eq!(strpbrk_safe(s, bytemuck::cast_slice(b"ol\0")), Some(2));
        assert_eq!(strpbrk_safe(s, bytemuck::cast_slice(b"xyz\0")), None);
        /* an empty set matches nothing */
        assert_eq!(strpbrk_safe(s, bytemuck::cast_slice(b"\0")), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_strpbrk_strpbrk_safe() {
        let s: &[c_char] = bytemuck::cast_slice(b"hello, world\0");

        for set in [&b",;\0"[..], b"ol\0", b"xyz\0", b"\0", b"h\0", b"d\0"] {
            let set: &[c_char] = bytemuck::cast_slice(set);
            let ru = unsafe { libc::strpbrk(s.as_ptr(), set.as_ptr()) };
            let expected = if ru.is_null() {
                None
            } else {
                Some(unsafe { ru.offset_from(s.as_ptr()) } as usize)
            };
            assert_eq!(strpbrk_safe(s, set), expected, "set {set:?}");
        }
    }

    // Write test cases for strto_float_impl
    #[test]
    fn test_strto_float_impl() {
        let mut endptr: usize = 0;

        // Test with valid float
        let s: &[c_char] = bytemuck::cast_slice(b"123.456\0");
        let result = strto_float_impl(s, &mut endptr);
        assert_eq!(result, 123.456);
        assert_eq!(endptr, 7); // "123.456" + null terminator

        // Test with negative float
        let s: &[c_char] = bytemuck::cast_slice(b"-123.456\0");
        let result = strto_float_impl(s, &mut endptr);
        assert_eq!(result, -123.456);
        assert_eq!(endptr, 8); // "-123.456" + null terminator

        // Test with scientific notation
        let s: &[c_char] = bytemuck::cast_slice(b"1.23e4\0");
        let result = strto_float_impl(s, &mut endptr);
        assert_eq!(result, 12300.0);
        assert_eq!(endptr, 6); // "1.23e4" + null terminator

        // Test with invalid float
        let s: &[c_char] = bytemuck::cast_slice(b"abc\0");
        let result = strto_float_impl(s, &mut endptr);
        assert_eq!(result, 0.0);
        assert_eq!(endptr, 0); // No valid float parsed

        // Test with empty string
        let s: &[c_char] = bytemuck::cast_slice(b"\0");
        let result = strto_float_impl(s, &mut endptr);
        assert_eq!(result, 0.0);
        assert_eq!(endptr, 0); // No valid float parsed

        // Test with Nan
        let s: &[c_char] = bytemuck::cast_slice(b"nan\0");
        let result = strto_float_impl(s, &mut endptr);
        assert!(result.is_nan());
        assert_eq!(endptr, 3); // NaN parsed
    }

    #[test]
    fn test_strtod_safe() {
        // Write test where input is "(21,21)"
        let input: &[c_char] = bytemuck::cast_slice(b"(21,21)\0");
        let mut endp = 0;

        let result = strtod_safe(&input[1..], &mut endp);
        assert_eq!(result, 21.0);
        assert_eq!(endp, 2);
    }
}

#[cfg(test)]
mod libc_parity_tests {
    use super::*;
    use crate::c_types::c_long;

    /// NUL-terminated `[c_char]` from a `&str`, as the callers always pass.
    fn cc(s: &str) -> Vec<c_char> {
        let mut v: Vec<c_char> = s.bytes().map(|b| b as c_char).collect();
        v.push(0);
        v
    }

    /// Inputs exercised against the C: plain values, the precision cases that
    /// digit-by-digit accumulation gets wrong, signs, whitespace, exponents,
    /// trailing junk, and the no-digits case.
    const FLOAT_CASES: &[&str] = &[
        "0",
        "1",
        "1.5",
        "2.",
        ".5",
        "-1.5",
        "+1.5",
        "  3.75",
        "1e3",
        "1E3",
        "1.5e-3",
        "1.5E+3",
        "3.25",
        "0.1",
        "0.2",
        "0.3",
        "12345.6789",
        "1234567.891011",
        "0.000123456789",
        "9007199254740993",
        "1.7976931348623157e308",
        "2.2250738585072014e-308",
        "1e-400",
        "1e400",
        "123abc",
        "abc",
        "",
        "   ",
        "-0",
        "0.0000000000000001",
    ];

    const LONG_CASES: &[&str] = &[
        "0",
        "1",
        "-1",
        "+1",
        "  42",
        "42abc",
        "abc",
        "",
        "   ",
        "-0",
        "2147483647",
        "-2147483648",
        "9223372036854775807",
        "-9223372036854775808",
        "99999999999999999999",
        "-99999999999999999999",
        "007",
        "-  5",
        "1 2",
    ];

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_atof_safe_matches_libc_atof() {
        for case in FLOAT_CASES {
            let s = cc(case);
            let got = atof_safe(&s);
            let want = unsafe { libc::atof(s.as_ptr()) };
            assert!(
                got == want || (got.is_nan() && want.is_nan()),
                "atof({case:?}): atof_safe gave {got:?}, libc gave {want:?}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_atol_safe_matches_libc_atol() {
        for case in LONG_CASES {
            let s = cc(case);
            let got = atol_safe(&s);
            let want: c_long = unsafe { libc::atol(s.as_ptr()) };
            assert_eq!(got, want, "atol({case:?})");
        }
    }

    /// The specific regression: accumulating fractional digits by hand made
    /// this 12345.678899999999, so row filters comparing it with `==` failed.
    #[test]
    fn test_atof_safe_is_correctly_rounded() {
        assert_eq!(atof_safe(&cc("12345.6789")), 12345.6789);
        assert_eq!(atof_safe(&cc("0.000123456789")), 0.000123456789);
        assert_eq!(atof_safe(&cc("1234567.891011")), 1234567.891011);
    }

    /// A positive value equal to |c_long::MIN| accumulates to MIN without
    /// tripping the checked arithmetic, so negating it at the end overflows.
    /// Computed rather than hardcoded: c_long is 32-bit on Windows.
    #[test]
    fn test_atol_safe_at_the_negation_boundary() {
        let magnitude = (c_long::MIN as i128).unsigned_abs().to_string();
        assert_eq!(atol_safe(&cc(&magnitude)), c_long::MAX, "{magnitude}");
        assert_eq!(
            atol_safe(&cc(&format!("-{magnitude}"))),
            c_long::MIN,
            "-{magnitude}"
        );
    }

    /// C saturates on overflow rather than failing.
    #[test]
    fn test_atol_safe_saturates_like_c() {
        assert_eq!(atol_safe(&cc("99999999999999999999")), c_long::MAX);
        assert_eq!(atol_safe(&cc("-99999999999999999999")), c_long::MIN);
        assert_eq!(atol_safe(&cc("-9223372036854775808")), c_long::MIN);
    }
}
