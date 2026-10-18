//! `printf`-family formatting.
//!
//! Vendored from relibc; see [`crate::relibc`].
// Vendored code, documented in upstream's style.
#![allow(missing_docs)]

use crate::c_types::{
    c_char, c_double, c_int, c_long, c_longlong, c_short, c_uchar, c_uint, c_ulong, c_ulonglong,
    c_ushort, intmax_t, ptrdiff_t, size_t, ssize_t, uintmax_t,
};

use crate::relibc::io::{self, Write};
// use alloc::{
//     collections::BTreeMap,
//     string::{String, ToString},
//     vec::Vec,
// };
use alloc::collections::{BTreeMap, VecDeque};
use core::ffi::{CStr, c_void};
use core::{fmt, num::FpCategory, ops::Range, slice};

use crate::relibc::{
    header::errno::EILSEQ,
    platform::{self, types::*},
};

//  ____        _ _                 _       _
// | __ )  ___ (_) | ___ _ __ _ __ | | __ _| |_ ___ _
// |  _ \ / _ \| | |/ _ \ '__| '_ \| |/ _` | __/ _ (_)
// | |_) | (_) | | |  __/ |  | |_) | | (_| | ||  __/_
// |____/ \___/|_|_|\___|_|  | .__/|_|\__,_|\__\___(_)
//                           |_|

pub struct CustomVaList(pub VecDeque<VaArg>);

impl Default for CustomVaList {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomVaList {
    pub fn new() -> Self {
        CustomVaList(VecDeque::new())
    }

    pub fn arg(&mut self) -> VaArg {
        self.0
            .pop_front()
            .expect("No more arguments in CustomVaList")
    }

    pub fn push(&mut self, arg: VaArg) {
        self.0.push_back(arg);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IntKind {
    Byte,
    Short,
    Int,
    Long,
    LongLong,
    IntMax,
    PtrDiff,
    Size,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FmtKind {
    Percent,

    Signed,
    Unsigned,

    Scientific,
    Decimal,
    AnyNotation,

    String,
    Char,
    Pointer,
    GetWritten,
}
#[derive(Clone, Copy, Debug)]
enum Number {
    Static(usize),
    Index(usize),
    Next,
}
impl Number {
    unsafe fn resolve(self, varargs: &mut VaListCache, ap: &mut CustomVaList) -> usize {
        unsafe {
            let arg = match self {
                Number::Static(num) => return num,
                Number::Index(i) => varargs.get(i - 1, ap, None),
                Number::Next => {
                    let i = varargs.i;
                    varargs.i += 1;
                    varargs.get(i, ap, None)
                }
            };
            match arg {
                VaArg::c_char(i) => i as usize,
                VaArg::c_double(i) => i as usize,
                VaArg::c_int(i) => i as usize,
                VaArg::c_long(i) => i as usize,
                VaArg::c_longlong(i) => i as usize,
                VaArg::c_short(i) => i as usize,
                VaArg::intmax_t(i) => i as usize,
                VaArg::pointer(i) => i as usize,
                VaArg::ptrdiff_t(i) => i as usize,
                VaArg::ssize_t(i) => i as usize,
                VaArg::wint_t(i) => i as usize,
            }
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum VaArg {
    c_char(c_char),
    c_double(c_double),
    c_int(c_int),
    c_long(c_long),
    c_longlong(c_longlong),
    c_short(c_short),
    intmax_t(intmax_t),
    pointer(*const c_void),
    ptrdiff_t(ptrdiff_t),
    ssize_t(ssize_t),
    wint_t(wint_t),
}

impl VaArg {
    fn arg_from(fmtkind: FmtKind, intkind: IntKind, ap: &mut CustomVaList) -> VaArg {
        // Per the C standard using va_arg with a type with a size
        // less than that of an int for integers and double for floats
        // is invalid. As a result any arguments smaller than an int or
        // double passed to a function will be promoted to the smallest
        // possible size. The VaList::arg function will handle this
        // automagically.

        match (fmtkind, intkind) {
            (FmtKind::Percent, _) => panic!("Can't call arg_from on %"),

            (FmtKind::Char, IntKind::Long) | (FmtKind::Char, IntKind::LongLong) => {
                match ap.arg() {
                    // If the argument is a wint_t, we can safely return it
                    // as a wint_t. This is because the C standard guarantees
                    // that wint_t is an integer type.
                    VaArg::wint_t(i) => VaArg::wint_t(i),
                    _ => panic!("Expected wint_t for %c with long or long long"),
                }
            }

            (FmtKind::Char, _)
            | (FmtKind::Unsigned, IntKind::Byte)
            | (FmtKind::Signed, IntKind::Byte) => {
                match ap.arg() {
                    // If the argument is a c_char, we can safely return it
                    // as a c_char. This is because the C standard guarantees
                    // that c_char is an integer type.
                    VaArg::c_char(i) => VaArg::c_char(i),
                    _ => panic!("Expected c_char for %c with byte"),
                }
            }
            (FmtKind::Unsigned, IntKind::Short) | (FmtKind::Signed, IntKind::Short) => {
                match ap.arg() {
                    // If the argument is a c_short, we can safely return it
                    // as a c_short. This is because the C standard guarantees
                    // that c_short is an integer type.
                    VaArg::c_short(i) => VaArg::c_short(i),
                    _ => panic!("Expected c_short for %hd with short"),
                }
            }
            (FmtKind::Unsigned, IntKind::Int) | (FmtKind::Signed, IntKind::Int) => {
                match ap.arg() {
                    // If the argument is a c_int, we can safely return it
                    // as a c_int. This is because the C standard guarantees
                    // that c_int is an integer type.
                    VaArg::c_int(i) => VaArg::c_int(i),
                    _ => panic!("Expected c_int for %d with int"),
                }
            }
            (FmtKind::Unsigned, IntKind::Long) | (FmtKind::Signed, IntKind::Long) => {
                match ap.arg() {
                    // If the argument is a c_long, we can safely return it
                    // as a c_long. This is because the C standard guarantees
                    // that c_long is an integer type.
                    VaArg::c_long(i) => VaArg::c_long(i),
                    _ => panic!("Expected c_long for %ld with long"),
                }
            }
            (FmtKind::Unsigned, IntKind::LongLong) | (FmtKind::Signed, IntKind::LongLong) => {
                match ap.arg() {
                    // If the argument is a c_longlong, we can safely return it
                    // as a c_longlong. This is because the C standard guarantees
                    // that c_longlong is an integer type.
                    VaArg::c_longlong(i) => VaArg::c_longlong(i),
                    _ => panic!("Expected c_longlong for %lld with long long"),
                }
            }
            (FmtKind::Unsigned, IntKind::IntMax) | (FmtKind::Signed, IntKind::IntMax) => {
                match ap.arg() {
                    // If the argument is an intmax_t, we can safely return it
                    // as an intmax_t. This is because the C standard guarantees
                    // that intmax_t is an integer type.
                    VaArg::intmax_t(i) => VaArg::intmax_t(i),
                    _ => panic!("Expected intmax_t for %jd with intmax_t"),
                }
            }
            (FmtKind::Unsigned, IntKind::PtrDiff) | (FmtKind::Signed, IntKind::PtrDiff) => {
                match ap.arg() {
                    // If the argument is a ptrdiff_t, we can safely return it
                    // as a ptrdiff_t. This is because the C standard guarantees
                    // that ptrdiff_t is an integer type.
                    VaArg::ptrdiff_t(i) => VaArg::ptrdiff_t(i),
                    _ => panic!("Expected ptrdiff_t for %td with ptrdiff_t"),
                }
            }
            (FmtKind::Unsigned, IntKind::Size) | (FmtKind::Signed, IntKind::Size) => {
                match ap.arg() {
                    // If the argument is a ssize_t, we can safely return it
                    // as a ssize_t. This is because the C standard guarantees
                    // that ssize_t is an integer type.
                    VaArg::ssize_t(i) => VaArg::ssize_t(i),
                    _ => panic!("Expected ssize_t for %zd with size"),
                }
            }

            (FmtKind::AnyNotation, _) | (FmtKind::Decimal, _) | (FmtKind::Scientific, _) => {
                match ap.arg() {
                    // If the argument is a c_double, we can safely return it
                    // as a c_double. This is because the C standard guarantees
                    // that c_double is a floating point type.
                    VaArg::c_double(i) => VaArg::c_double(i),
                    _ => panic!("Expected c_double for %f with double"),
                }
            }

            (FmtKind::GetWritten, _) | (FmtKind::Pointer, _) | (FmtKind::String, _) => {
                match ap.arg() {
                    // If the argument is a pointer, we can safely return it
                    // as a pointer. This is because the C standard guarantees
                    // that pointers are valid for printf.
                    VaArg::pointer(i) => VaArg::pointer(i),
                    _ => panic!("Expected pointer for %p with pointer"),
                }
            }
        }
    }
    unsafe fn transmute(&self, fmtkind: FmtKind, intkind: IntKind) -> VaArg {
        unsafe {
            // At this point, there are conflicting printf arguments. An
            // example of this is:
            // ```c
            // printf("%1$d %1$lf\n", 5, 0.1);
            // ```
            // We handle it just like glibc: We read it from the VaList
            // using the *last* argument type, but we transmute it when we
            // try to access the other ones.
            union Untyped {
                c_char: c_char,
                c_double: c_double,
                c_int: c_int,
                c_long: c_long,
                c_longlong: c_longlong,
                c_short: c_short,
                intmax_t: intmax_t,
                pointer: *const c_void,
                ptrdiff_t: ptrdiff_t,
                ssize_t: ssize_t,
                wint_t: wint_t,
            }
            let untyped = match *self {
                VaArg::c_char(i) => Untyped { c_char: i },
                VaArg::c_double(i) => Untyped { c_double: i },
                VaArg::c_int(i) => Untyped { c_int: i },
                VaArg::c_long(i) => Untyped { c_long: i },
                VaArg::c_longlong(i) => Untyped { c_longlong: i },
                VaArg::c_short(i) => Untyped { c_short: i },
                VaArg::intmax_t(i) => Untyped { intmax_t: i },
                VaArg::pointer(i) => Untyped { pointer: i },
                VaArg::ptrdiff_t(i) => Untyped { ptrdiff_t: i },
                VaArg::ssize_t(i) => Untyped { ssize_t: i },
                VaArg::wint_t(i) => Untyped { wint_t: i },
            };
            match (fmtkind, intkind) {
                (FmtKind::Percent, _) => panic!("Can't call transmute on %"),

                (FmtKind::Char, IntKind::Long) | (FmtKind::Char, IntKind::LongLong) => {
                    VaArg::wint_t(untyped.wint_t)
                }

                (FmtKind::Char, _)
                | (FmtKind::Unsigned, IntKind::Byte)
                | (FmtKind::Signed, IntKind::Byte) => VaArg::c_char(untyped.c_char),
                (FmtKind::Unsigned, IntKind::Short) | (FmtKind::Signed, IntKind::Short) => {
                    VaArg::c_short(untyped.c_short)
                }
                (FmtKind::Unsigned, IntKind::Int) | (FmtKind::Signed, IntKind::Int) => {
                    VaArg::c_int(untyped.c_int)
                }
                (FmtKind::Unsigned, IntKind::Long) | (FmtKind::Signed, IntKind::Long) => {
                    VaArg::c_long(untyped.c_long)
                }
                (FmtKind::Unsigned, IntKind::LongLong) | (FmtKind::Signed, IntKind::LongLong) => {
                    VaArg::c_longlong(untyped.c_longlong)
                }
                (FmtKind::Unsigned, IntKind::IntMax) | (FmtKind::Signed, IntKind::IntMax) => {
                    VaArg::intmax_t(untyped.intmax_t)
                }
                (FmtKind::Unsigned, IntKind::PtrDiff) | (FmtKind::Signed, IntKind::PtrDiff) => {
                    VaArg::ptrdiff_t(untyped.ptrdiff_t)
                }
                (FmtKind::Unsigned, IntKind::Size) | (FmtKind::Signed, IntKind::Size) => {
                    VaArg::ssize_t(untyped.ssize_t)
                }

                (FmtKind::AnyNotation, _) | (FmtKind::Decimal, _) | (FmtKind::Scientific, _) => {
                    VaArg::c_double(untyped.c_double)
                }

                (FmtKind::GetWritten, _) | (FmtKind::Pointer, _) | (FmtKind::String, _) => {
                    VaArg::pointer(untyped.pointer)
                }
            }
        }
    }
}
#[derive(Default)]
struct VaListCache {
    args: Vec<VaArg>,
    i: usize,
}
impl VaListCache {
    unsafe fn get(
        &mut self,
        i: usize,
        ap: &mut CustomVaList,
        default: Option<(FmtKind, IntKind)>,
    ) -> VaArg {
        unsafe {
            if let Some(&arg) = self.args.get(i) {
                // This value is already cached
                let mut arg = arg;
                if let Some((fmtkind, intkind)) = default {
                    // ...but as a different type
                    arg = arg.transmute(fmtkind, intkind);
                }
                return arg;
            }

            // Get all values before this value
            while self.args.len() < i {
                // We can't POSSIBLY know the type if we reach this
                // point. Reaching here means there are unused gaps in the
                // arguments. Ultimately we'll have to settle down with
                // defaulting to c_int.
                self.args.push(ap.arg())
            }

            // Add the value to the cache
            self.args.push(match default {
                Some((_fmtkind, _intkind)) => ap.arg(),
                None => ap.arg(),
            });

            // Return the value
            self.args[i]
        }
    }
}

//  ___                 _                           _        _   _
// |_ _|_ __ ___  _ __ | | ___ _ __ ___   ___ _ __ | |_ __ _| |_(_) ___  _ __  _
//  | || '_ ` _ \| '_ \| |/ _ \ '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \(_)
//  | || | | | | | |_) | |  __/ | | | | |  __/ | | | || (_| | |_| | (_) | | | |_
// |___|_| |_| |_| .__/|_|\___|_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_(_)
//               |_|

enum FmtCase {
    Lower,
    Upper,
}

// The spelled-out "infinity"/"INFINITY" is also permitted by the standard
static INF_STR_LOWER: &str = "inf";
static INF_STR_UPPER: &str = "INF";

static NAN_STR_LOWER: &str = "nan";
static NAN_STR_UPPER: &str = "NAN";

unsafe fn pop_int_raw(format: &mut *const u8) -> Option<usize> {
    unsafe {
        let mut int: Option<usize> = None;
        while let Some(digit) = (**format as char).to_digit(10) {
            *format = format.add(1);
            // Saturate: a width or precision this large is rejected by the caller.
            int = Some(
                int.unwrap_or(0)
                    .saturating_mul(10)
                    .saturating_add(digit as usize),
            );
        }
        int
    }
}
unsafe fn pop_index(format: &mut *const u8) -> Option<usize> {
    unsafe {
        // Peek ahead for a positional argument:
        let mut format2 = *format;
        if let Some(i) = pop_int_raw(&mut format2)
            && *format2 == b'$'
        {
            *format = format2.add(1);
            return Some(i);
        }
        None
    }
}
unsafe fn pop_int(format: &mut *const u8) -> Option<Number> {
    unsafe {
        if **format == b'*' {
            *format = format.add(1);
            Some(pop_index(format).map(Number::Index).unwrap_or(Number::Next))
        } else {
            pop_int_raw(format).map(Number::Static)
        }
    }
}

fn fmt_int<I>(fmt: u8, i: I) -> String
where
    I: fmt::Display + fmt::Octal + fmt::LowerHex + fmt::UpperHex + fmt::Binary,
{
    match fmt {
        b'o' => format!("{i:o}"),
        b'u' => i.to_string(),
        b'x' => format!("{i:x}"),
        b'X' => format!("{i:X}"),
        b'b' | b'B' => format!("{i:b}"),
        _ => panic!(
            "fmt_int should never be called with the fmt {:?}",
            fmt as char
        ),
    }
}

fn pad<W: Write>(
    w: &mut W,
    current_side: bool,
    pad_char: u8,
    range: Range<usize>,
) -> io::Result<()> {
    if current_side {
        // Write in blocks: a field can be up to INT_MAX wide.
        let block = [pad_char; 4096];
        let mut left = range.len();
        while left > 0 {
            let n = left.min(block.len());
            w.write_all(&block[..n])?;
            left -= n;
        }
    }
    Ok(())
}

fn abs(float: c_double) -> c_double {
    // Don't ask me whe float.abs() seems absent...
    if float.is_sign_negative() {
        -float
    } else {
        float
    }
}

/// `trim` strips trailing zeros and a then-trailing point (%g without `#`);
/// `force_point` keeps a point that would otherwise not appear at all.
fn float_string(float: c_double, precision: usize, trim: bool, force_point: bool) -> String {
    let mut string = format!("{float:.precision$}");
    if trim && string.contains('.') {
        let truncate = {
            let slice = string.trim_end_matches('0');
            let mut truncate = slice.len();
            if slice.ends_with('.') {
                truncate -= 1;
            }
            truncate
        };
        string.truncate(truncate);
    }
    if force_point && !string.contains('.') {
        string.push('.');
    }
    string
}

fn fmt_float_exp<W: Write>(
    w: &mut W,
    exp_fmt: u8,
    trim: bool,
    force_point: bool,
    precision: usize,
    float: c_double,
    left: bool,
    pad_space: usize,
    pad_zero: usize,
) -> io::Result<()> {
    // Derive the mantissa and exponent from Rust's exponential formatting, which
    // is correctly rounded. The previous approach normalized the mantissa by
    // repeatedly dividing/multiplying by 10 (see `float_exp`); for large or small
    // magnitudes that accumulates rounding error and corrupts the last decimal
    // digits, so a round-trip through an ASCII FITS table was not exact.
    let formatted = format!("{:.*e}", precision, float);
    let epos = formatted
        .find('e')
        .expect("exponential formatting always contains 'e'");
    let exp: isize = formatted[(epos + 1)..]
        .parse()
        .expect("exponential formatting has a valid exponent");
    let mut string = formatted[..epos].to_string();
    if trim && string.contains('.') {
        let truncate = {
            let slice = string.trim_end_matches('0');
            let mut truncate = slice.len();
            if slice.ends_with('.') {
                truncate -= 1;
            }
            truncate
        };
        string.truncate(truncate);
    }
    if force_point && !string.contains('.') {
        string.push('.');
    }

    let mut exp2 = exp.abs();
    let mut exp_len = 1;
    while exp2 >= 10 {
        exp2 /= 10;
        exp_len += 1;
    }

    let len = string.len() + 2 + 2.max(exp_len);

    pad(w, !left, b' ', len..pad_space)?;
    let bytes = if string.starts_with('-') {
        w.write_all(b"-")?;
        &string.as_bytes()[1..]
    } else {
        string.as_bytes()
    };
    pad(w, !left, b'0', len..pad_zero)?;
    w.write_all(bytes)?;
    write!(w, "{}{:+03}", exp_fmt as char, exp)?;
    pad(w, left, b' ', len..pad_space)?;

    Ok(())
}

fn fmt_float_normal<W: Write>(
    w: &mut W,
    trim: bool,
    force_point: bool,
    precision: usize,
    float: c_double,
    left: bool,
    pad_space: usize,
    pad_zero: usize,
) -> io::Result<usize> {
    let string = float_string(float, precision, trim, force_point);

    pad(w, !left, b' ', string.len()..pad_space)?;
    let bytes = if string.starts_with('-') {
        w.write_all(b"-")?;
        &string.as_bytes()[1..]
    } else {
        string.as_bytes()
    };
    pad(w, true, b'0', string.len()..pad_zero)?;
    w.write_all(bytes)?;
    pad(w, left, b' ', string.len()..pad_space)?;

    Ok(string.len())
}

/// Write ±infinity or ±NaN for any floating-point style.  C pads these with
/// *spaces* even under the `0' flag (C99 7.19.6.1).
fn fmt_float_nonfinite<W: Write>(
    w: &mut W,
    float: c_double,
    case: FmtCase,
    left: bool,
    width: usize,
) -> io::Result<()> {
    let len = if float.is_sign_negative() { 4 } else { 3 };
    pad(w, !left, b' ', len..width)?;

    if float.is_sign_negative() {
        w.write_all(b"-")?;
    }

    let nonfinite_str = match float.classify() {
        FpCategory::Infinite => match case {
            FmtCase::Lower => INF_STR_LOWER,
            FmtCase::Upper => INF_STR_UPPER,
        },
        FpCategory::Nan => match case {
            FmtCase::Lower => NAN_STR_LOWER,
            FmtCase::Upper => NAN_STR_UPPER,
        },
        _ => {
            // This function should only be called with infinite or NaN value.
            panic!("this should not be possible")
        }
    };

    w.write_all(nonfinite_str.as_bytes())?;
    pad(w, left, b' ', len..width)?;

    Ok(())
}

#[derive(Clone, Copy)]
struct PrintfIter {
    format: *const u8,
}
#[derive(Clone, Copy, Debug)]
struct PrintfArg {
    index: Option<usize>,
    alternate: bool,
    zero: bool,
    left: bool,
    sign_reserve: bool,
    sign_always: bool,
    min_width: Number,
    precision: Option<Number>,
    intkind: IntKind,
    fmt: u8,
    fmtkind: FmtKind,
}
#[derive(Debug)]
enum PrintfFmt {
    Plain(&'static [u8]),
    Arg(PrintfArg),
}
impl Iterator for PrintfIter {
    type Item = Result<PrintfFmt, ()>;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            // Send PrintfFmt::Plain until the next %
            let mut len = 0;
            while *self.format.add(len) != 0 && *self.format.add(len) != b'%' {
                len += 1;
            }
            if len > 0 {
                let slice = slice::from_raw_parts(self.format, len);
                self.format = self.format.add(len);
                return Some(Ok(PrintfFmt::Plain(slice)));
            }
            self.format = self.format.add(len);
            if *self.format == 0 {
                return None;
            }

            // *self.format is guaranteed to be '%' at this point
            self.format = self.format.add(1);

            let mut peekahead = self.format;
            let index = pop_index(&mut peekahead).inspect(|_i| {
                self.format = peekahead;
            });

            // Flags:
            let mut alternate = false;
            let mut zero = false;
            let mut left = false;
            let mut sign_reserve = false;
            let mut sign_always = false;

            loop {
                match *self.format {
                    b'#' => alternate = true,
                    b'0' => zero = true,
                    b'-' => left = true,
                    b' ' => sign_reserve = true,
                    b'+' => sign_always = true,
                    _ => break,
                }
                self.format = self.format.add(1);
            }

            // Width and precision:
            let min_width = pop_int(&mut self.format).unwrap_or(Number::Static(0));
            let precision = if *self.format == b'.' {
                self.format = self.format.add(1);
                match pop_int(&mut self.format) {
                    int @ Some(_) => int,
                    None => return Some(Err(())),
                }
            } else {
                None
            };

            // Integer size:
            let mut intkind = IntKind::Int;
            loop {
                intkind = match *self.format {
                    b'h' => {
                        if intkind == IntKind::Short || intkind == IntKind::Byte {
                            IntKind::Byte
                        } else {
                            IntKind::Short
                        }
                    }
                    b'j' => IntKind::IntMax,
                    b'l' => {
                        if intkind == IntKind::Long || intkind == IntKind::LongLong {
                            IntKind::LongLong
                        } else {
                            IntKind::Long
                        }
                    }
                    b'q' | b'L' => IntKind::LongLong,
                    b't' => IntKind::PtrDiff,
                    b'z' => IntKind::Size,
                    _ => break,
                };

                self.format = self.format.add(1);
            }
            let fmt = *self.format;
            let fmtkind = match fmt {
                b'%' => FmtKind::Percent,
                b'd' | b'i' => FmtKind::Signed,
                b'o' | b'u' | b'x' | b'X' | b'b' | b'B' => FmtKind::Unsigned,
                b'e' | b'E' => FmtKind::Scientific,
                b'f' | b'F' => FmtKind::Decimal,
                b'g' | b'G' => FmtKind::AnyNotation,
                b's' => FmtKind::String,
                b'c' => FmtKind::Char,
                b'p' => FmtKind::Pointer,
                b'n' => FmtKind::GetWritten,
                _ => return Some(Err(())),
            };
            self.format = self.format.add(1);

            Some(Ok(PrintfFmt::Arg(PrintfArg {
                index,
                alternate,
                zero,
                left,
                sign_reserve,
                sign_always,
                min_width,
                precision,
                intkind,
                fmt,
                fmtkind,
            })))
        }
    }
}

unsafe fn inner_printf<W: Write>(w: W, format: &CStr, mut ap: CustomVaList) -> io::Result<c_int> {
    unsafe {
        let w = &mut platform::CountingWriter::new(w);

        let iterator = PrintfIter {
            format: format.to_bytes_with_nul().as_ptr(),
        };

        // Pre-fetch vararg types
        let mut varargs = VaListCache::default();
        let mut positional = BTreeMap::new();
        // ^ NOTE: This depends on the sorted order, do not change to HashMap or whatever

        for section in iterator {
            let arg = match section {
                Ok(PrintfFmt::Plain(_text)) => continue,
                Ok(PrintfFmt::Arg(arg)) => arg,
                Err(()) => return Ok(-1),
            };
            if arg.fmtkind == FmtKind::Percent {
                continue;
            }
            for num in &[arg.min_width, arg.precision.unwrap_or(Number::Static(0))] {
                match num {
                    Number::Next => varargs.args.push(ap.arg()),
                    Number::Index(i) => {
                        positional.insert(i - 1, (FmtKind::Signed, IntKind::Int));
                    }
                    // A field width or precision over INT_MAX is EOVERFLOW in C.
                    Number::Static(n) if *n > c_int::MAX as usize => return Ok(-1),
                    Number::Static(_) => (),
                }
            }
            match arg.index {
                Some(i) => {
                    positional.insert(i - 1, (arg.fmtkind, arg.intkind));
                }
                None => varargs
                    .args
                    .push(VaArg::arg_from(arg.fmtkind, arg.intkind, &mut ap)),
            }
        }

        // Make sure, in order, the positional arguments exist with the specified type
        for (i, arg) in positional {
            varargs.get(i, &mut ap, Some(arg));
        }

        // Main loop
        for section in iterator {
            let arg = match section {
                Ok(PrintfFmt::Plain(text)) => {
                    w.write_all(text)?;
                    continue;
                }
                Ok(PrintfFmt::Arg(arg)) => arg,
                Err(()) => return Ok(-1),
            };
            let alternate = arg.alternate;
            let zero = arg.zero;
            let mut left = arg.left;
            let sign_reserve = arg.sign_reserve;
            let sign_always = arg.sign_always;
            let min_width = arg.min_width.resolve(&mut varargs, &mut ap);
            let precision = arg.precision.map(|n| n.resolve(&mut varargs, &mut ap));
            let pad_zero = if zero { min_width } else { 0 };
            let signed_space = match pad_zero {
                0 => min_width as isize,
                _ => 0,
            };
            let pad_space = if signed_space < 0 {
                left = true;
                -signed_space as usize
            } else {
                signed_space as usize
            };
            let intkind = arg.intkind;
            let fmt = arg.fmt;
            let fmtkind = arg.fmtkind;
            let fmtcase = match fmt {
                b'x' | b'b' | b'f' | b'e' | b'g' => Some(FmtCase::Lower),
                b'X' | b'B' | b'F' | b'E' | b'G' => Some(FmtCase::Upper),
                _ => None,
            };

            let index = arg.index.map(|i| i - 1).unwrap_or_else(|| {
                if fmtkind == FmtKind::Percent {
                    0
                } else {
                    let i = varargs.i;
                    varargs.i += 1;
                    i
                }
            });

            match fmtkind {
                FmtKind::Percent => w.write_all(b"%")?,
                FmtKind::Signed => {
                    let string = match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind)))
                    {
                        VaArg::c_char(i) => i.to_string(),
                        VaArg::c_double(_i) => panic!("this should not be possible"),
                        VaArg::c_int(i) => i.to_string(),
                        VaArg::c_long(i) => i.to_string(),
                        VaArg::c_longlong(i) => i.to_string(),
                        VaArg::c_short(i) => i.to_string(),
                        VaArg::intmax_t(i) => i.to_string(),
                        VaArg::pointer(i) => (i as usize).to_string(),
                        VaArg::ptrdiff_t(i) => i.to_string(),
                        VaArg::ssize_t(i) => i.to_string(),
                        VaArg::wint_t(_) => unreachable!("this should not be possible"),
                    };
                    let positive = !string.starts_with('-');
                    let zero = precision == Some(0) && string == "0";

                    let mut len = string.len();
                    let mut final_len = string.len().max(precision.unwrap_or(0));
                    if positive && (sign_reserve || sign_always) {
                        final_len += 1;
                    }
                    if zero {
                        len = 0;
                        final_len = 0;
                    }

                    pad(w, !left, b' ', final_len..pad_space)?;

                    let bytes = if positive {
                        if sign_reserve {
                            w.write_all(b" ")?;
                        } else if sign_always {
                            w.write_all(b"+")?;
                        }
                        string.as_bytes()
                    } else {
                        w.write_all(b"-")?;
                        &string.as_bytes()[1..]
                    };
                    pad(w, true, b'0', len..precision.unwrap_or(pad_zero))?;

                    if !zero {
                        w.write_all(bytes)?;
                    }

                    pad(w, left, b' ', final_len..pad_space)?;
                }
                FmtKind::Unsigned => {
                    let string = match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind)))
                    {
                        VaArg::c_char(i) => fmt_int(fmt, i as c_uchar),
                        VaArg::c_double(_i) => panic!("this should not be possible"),
                        VaArg::c_int(i) => fmt_int(fmt, i as c_uint),
                        VaArg::c_long(i) => fmt_int(fmt, i as c_ulong),
                        VaArg::c_longlong(i) => fmt_int(fmt, i as c_ulonglong),
                        VaArg::c_short(i) => fmt_int(fmt, i as c_ushort),
                        VaArg::intmax_t(i) => fmt_int(fmt, i as uintmax_t),
                        VaArg::pointer(i) => fmt_int(fmt, i as usize),
                        VaArg::ptrdiff_t(i) => fmt_int(fmt, i as size_t),
                        VaArg::ssize_t(i) => fmt_int(fmt, i as size_t),
                        VaArg::wint_t(_) => unreachable!("this should not be possible"),
                    };
                    let zero = precision == Some(0) && string == "0";

                    // If this int is padded out to be larger than it is, don't
                    // add an extra zero if octal.
                    let no_precision = precision.map(|pad| pad < string.len()).unwrap_or(true);

                    let len;
                    let final_len = if zero {
                        len = 0;
                        0
                    } else {
                        len = string.len();
                        len.max(precision.unwrap_or(0))
                            + if alternate && string != "0" {
                                match fmt {
                                    b'o' if no_precision => 1,
                                    b'x' | b'X' | b'b' | b'B' => 2,
                                    _ => 0,
                                }
                            } else {
                                0
                            }
                    };

                    pad(w, !left, b' ', final_len..pad_space)?;

                    if alternate && string != "0" {
                        match fmt {
                            b'o' if no_precision => w.write_all(b"0")?,
                            b'x' => w.write_all(b"0x")?,
                            b'X' => w.write_all(b"0X")?,
                            b'b' => w.write_all(b"0b")?,
                            b'B' => w.write_all(b"0B")?,
                            _ => (),
                        }
                    }
                    pad(w, true, b'0', len..precision.unwrap_or(pad_zero))?;

                    if !zero {
                        w.write_all(string.as_bytes())?;
                    }

                    pad(w, left, b' ', final_len..pad_space)?;
                }
                FmtKind::Scientific => {
                    let float = match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind)))
                    {
                        VaArg::c_double(i) => i,
                        _ => panic!("this should not be possible"),
                    };
                    if float.is_finite() {
                        let precision = precision.unwrap_or(6);

                        fmt_float_exp(
                            w, fmt, false, alternate, precision, float, left, pad_space, pad_zero,
                        )?;
                    } else {
                        fmt_float_nonfinite(
                            w,
                            float,
                            fmtcase.unwrap(),
                            left,
                            pad_space.max(pad_zero),
                        )?;
                    }
                }
                FmtKind::Decimal => {
                    let float = match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind)))
                    {
                        VaArg::c_double(i) => i,
                        _ => panic!("this should not be possible"),
                    };
                    if float.is_finite() {
                        let precision = precision.unwrap_or(6);

                        fmt_float_normal(
                            w, false, alternate, precision, float, left, pad_space, pad_zero,
                        )?;
                    } else {
                        fmt_float_nonfinite(
                            w,
                            float,
                            fmtcase.unwrap(),
                            left,
                            pad_space.max(pad_zero),
                        )?;
                    }
                }
                FmtKind::AnyNotation => {
                    let float = match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind)))
                    {
                        VaArg::c_double(i) => i,
                        _ => panic!("this should not be possible"),
                    };
                    if float.is_finite() {
                        let exp_fmt = b'E' | (fmt & 32);

                        // C99 7.19.6.1: P is the precision, 6 if omitted, 1 if zero
                        let p = match precision {
                            Some(0) => 1,
                            Some(p) => p,
                            None => 6,
                        };

                        // If a style-E conversion would have exponent X, then
                        // P > X >= -4 selects style f with precision P-(X+1),
                        // else style e with precision P-1.  X is the exponent
                        // *after* rounding to P digits, hence taking it from
                        // Rust's correctly-rounded exponential formatting: at
                        // P=3, 9.999e-5 rounds to 1.00e-04 and prints fixed.
                        let rounded = format!("{:.*e}", p - 1, float);
                        let exp: isize = rounded[rounded
                            .find('e')
                            .expect("exponential formatting always contains 'e'")
                            + 1..]
                            .parse()
                            .expect("exponential formatting has a valid exponent");

                        // `#' keeps the trailing zeros and the radix point
                        let trim = !alternate;

                        if exp < -4 || exp >= p as isize {
                            fmt_float_exp(
                                w,
                                exp_fmt,
                                trim,
                                alternate,
                                p - 1,
                                float,
                                left,
                                pad_space,
                                pad_zero,
                            )?;
                        } else {
                            // P - (X + 1), which the guard above keeps >= 0
                            let precision = (p as isize - 1 - exp) as usize;
                            fmt_float_normal(
                                w, trim, alternate, precision, float, left, pad_space, pad_zero,
                            )?;
                        }
                    } else {
                        fmt_float_nonfinite(
                            w,
                            float,
                            fmtcase.unwrap(),
                            left,
                            pad_space.max(pad_zero),
                        )?;
                    }
                }
                FmtKind::String => {
                    let ptr =
                        (match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind))) {
                            VaArg::pointer(p) => p,
                            _ => panic!("this should not be possible"),
                        })
                        .cast::<c_char>();

                    if ptr.is_null() {
                        w.write_all(b"(null)")?;
                    } else {
                        let max = precision.unwrap_or(usize::MAX);

                        if intkind == IntKind::Long || intkind == IntKind::LongLong {
                            // Handle wchar_t
                            // %ls promises a wchar_t*, so the vararg is already
                            // wchar_t-aligned even though it arrives typed as
                            // bytes (clippy's cast_ptr_alignment); assert it
                            // rather than assuming it.
                            #[allow(clippy::cast_ptr_alignment)]
                            let mut ptr = ptr.cast::<wchar_t>();
                            debug_assert!(ptr.is_aligned());
                            let mut string = String::new();

                            while *ptr != 0 {
                                let c = match char::from_u32(*ptr as _) {
                                    Some(c) => c,
                                    None => {
                                        platform::ERRNO.set(EILSEQ);
                                        return Err(io::last_os_error());
                                    }
                                };
                                if string.len() + c.len_utf8() >= max {
                                    break;
                                }
                                string.push(c);
                                ptr = ptr.add(1);
                            }

                            pad(w, !left, b' ', string.len()..pad_space)?;
                            w.write_all(string.as_bytes())?;
                            pad(w, left, b' ', string.len()..pad_space)?;
                        } else {
                            let mut len = 0;
                            while *ptr.add(len) != 0 && len < max {
                                len += 1;
                            }

                            pad(w, !left, b' ', len..pad_space)?;
                            w.write_all(slice::from_raw_parts(ptr.cast::<u8>(), len))?;
                            pad(w, left, b' ', len..pad_space)?;
                        }
                    }
                }
                FmtKind::Char => {
                    match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind))) {
                        VaArg::c_char(c) => {
                            pad(w, !left, b' ', 1..pad_space)?;
                            w.write_all(&[c as u8])?;
                            pad(w, left, b' ', 1..pad_space)?;
                        }
                        VaArg::wint_t(c) => {
                            let c = match char::from_u32(c as _) {
                                Some(c) => c,
                                None => {
                                    platform::ERRNO.set(EILSEQ);
                                    return Err(io::last_os_error());
                                }
                            };
                            let mut buf = [0; 4];

                            pad(w, !left, b' ', 1..pad_space)?;
                            w.write_all(c.encode_utf8(&mut buf).as_bytes())?;
                            pad(w, left, b' ', 1..pad_space)?;
                        }
                        _ => unreachable!("this should not be possible"),
                    }
                }
                FmtKind::Pointer => {
                    let ptr = match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind))) {
                        VaArg::pointer(p) => p,
                        _ => panic!("this should not be possible"),
                    };

                    let mut len = 1;
                    if ptr.is_null() {
                        len = "(nil)".len();
                    } else {
                        let mut ptr = ptr as usize;
                        while ptr >= 10 {
                            ptr /= 10;
                            len += 1;
                        }
                    }

                    pad(w, !left, b' ', len..pad_space)?;
                    if ptr.is_null() {
                        write!(w, "(nil)")?;
                    } else {
                        write!(w, "0x{:x}", ptr as usize)?;
                    }
                    pad(w, left, b' ', len..pad_space)?;
                }
                FmtKind::GetWritten => {
                    let ptr = match varargs.get(index, &mut ap, Some((arg.fmtkind, arg.intkind))) {
                        VaArg::pointer(p) => p,
                        _ => panic!("this should not be possible"),
                    };

                    match intkind {
                        IntKind::Byte => *(ptr as *mut c_char) = w.written as c_char,
                        IntKind::Short => *(ptr as *mut c_short) = w.written as c_short,
                        IntKind::Int => *(ptr as *mut c_int) = w.written as c_int,
                        IntKind::Long => *(ptr as *mut c_long) = w.written as c_long,
                        IntKind::LongLong => *(ptr as *mut c_longlong) = w.written as c_longlong,
                        IntKind::IntMax => *(ptr as *mut intmax_t) = w.written as intmax_t,
                        IntKind::PtrDiff => *(ptr as *mut ptrdiff_t) = w.written as ptrdiff_t,
                        IntKind::Size => *(ptr as *mut size_t) = w.written as size_t,
                    }
                }
            }
        }
        Ok(w.written as c_int)
    }
}

/// Implementation of `printf` formatting function, generic over a `writer`
///
/// This implementation in currently compliant over C17 specification (lacking a few one from C23)
/// and contains extensions as well.
///
/// # The Format Specification
/// ```text
/// %[conversion-flags][field-width][precision][length-modifier]<conversion-format>
/// ```
///
/// <div class="warning">
/// ※ : This symbol means it is not implemented yet, but it is defined in the C standard
/// </div>
///
/// ## Conversion Flags
/// Conversion flags are flags that modify the behavior of the [conversion
/// format]. Each one can happen only once per format specifier. They are:
///
/// - `-`: The result of the conversion is left-justified within the field (by default it is
///   right-justified).
/// - `+`: The sign of signed conversions is always prepended to the result of the conversion (by
///   default the result is preceded by minus **only** when it is negative).
/// - ` `(space): If the result of a signed conversion does not start with a sign character, or is
///   empty, space is prepended to the result.
///   - It is ignored if `+` flag is present.
/// - `#`: Alternative form of the conversion is performed. See the documentation for each
///   [conversion format] for details.
/// - `0`: For integer and floating-point number conversions, leading zeros are used to pad the
///   field instead of space characters.
///   - For integer numbers it is ignored if the precision is explicitly specified.
///   - For other conversions using this flag results in undefined behavior.
///   - It is ignored if `-` flag is present.
///
/// ## Field Width
/// Specifies minimum field width. This makes the result to be padded (with spaces by default, with
/// zeroes if `0` conversion flag is specified) if the converted value has fewer characters than the
/// specified width. It can take three forms:
///
/// - `N` where N is a positive integer: Specifies the field width value of `N`.
/// - `*`: The width is specified by an extra argument of type [`int`], which has to appear before
///   the argument to be converted and the [precision] (if specified with `.*`).
///   - If the value of e extra argument is negative, it is interpreted as with `-` [conversion
///     flag], i.e. left-justified result.
/// - `*P$` where P is a positive integer: The width is specified by an extra argument of type
///   [`int`], which has to appear exactly at the position specified by `P`.
///   - This is a popular extension of the C and POSIX standards.
///   - If the value of e extra argument is negative, it is interpreted as with `-` [conversion
///     flag], i.e. left-justified result.
///
/// ## Precision
/// Specifies the precision of the conversion.
///
/// For integer [conversion formats], this specifies the number of digits to appear in the result.
///
/// For float point [conversion formats], this specifies the number of digits to appear after the
/// decimal-point character.
///
/// It can take three forms:
///
/// - `.N` where N is a positive integer: Specifies the precision value of `N`.
/// - `.*`: The precision is specified by an extra argument of type [`int`], which  has to appear
///   before the argument to be converted and after the the [field width] (if specified with `*`).
///   - If the value of e extra argument is negative, it is interpreted as if the precision were
///     omitted.
/// - `.*P$` where P is a positive integer: The precision is specified by an extra argument of type
///   [`int`], which has to appear exactly at the position specified by `P`.
///   - This is an popular extension of the C and POSIX standards.
///   - If the value of e extra argument is negative, it is interpreted as if the precision were
///     omitted.
///
/// ## Length Modifier
/// Specifies the size of the argument. In combination with the [conversion format], it specifies
/// the type of the corresponding argument.
///
/// - `hh`: Byte size
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - Works with written number conversion format (`n`)
/// - `h`: Short size
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - Works with written number conversion format (`n`)
/// - `l`: Long size
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - Works with character conversion format (`c`)
///   - Works with string conversion format (`s`)
///   - Works with written number conversion format (`n`)
///   - Works with float conversion formats (`f`, `F`, `e`, `E`, `a`, `A`, `g`, `G`) (C99)
/// - `ll`: Long long size
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - Works with written number conversion format (`n`)
/// - `j`: Maximum width
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - Works with written number conversion format (`n`)
/// - `z`: Pointer width size
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - Works with written number conversion format (`n`)
/// - `t`: Pointer diff width
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - Works with written number conversion format (`n`)
/// - `wN` (C23 ※): Specifies that the size should be N bits width version of the supported
///   conversion format.
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - The supported values of `N` must be the same as the widths specified in `stdint.h`
/// - `wfN` (C23 ※): Specifies that the size should be the fast N bits width version of the
///   supported conversion format.
///   - Works with integer conversion formats (`d`, `i`, `o`, `x`, `X`, `b`, `B`)
///   - The supported values of `N` must be the same as the widths specified in `stdint.h`
/// - `L`: Long double size
///   - Works with float conversion formats (`f`, `F`, `e`, `E`, `a`, `A`, `g`, `G`)
/// - `H` (C23 ※): _Decimal32 size
///   - Works with float conversion formats (`f`, `F`, `e`, `E`, `a`, `A`, `g`, `G`)
/// - `D` (C23 ※): _Decimal64 size
///   - Works with float conversion formats (`f`, `F`, `e`, `E`, `a`, `A`, `g`, `G`)
/// - `DD` (C23 ※): _Decimal128 size
///   - Works with float conversion formats (`f`, `F`, `e`, `E`, `a`, `A`, `g`, `G`)
///
/// ## Conversion Format
/// Specifies the conversion format as one of the following:
///
/// - `%`: Writes a percent symbol. The full conversion format must be `%%`.
/// - `c`: Writes as single character
///   - Without length modifier:
///     - The argument is first converted to [`unsigned char`]
///   - With `l` length modifier:
///     - The argument is first converted to a character string as if by `%ls` with a array of 2
///       [`wchar_t`] argument.
/// - `s`: Writes a character string
///   - The argument is a pointer to the first character
///   - The [precision] specifies the maximum number of bytes to be written. If not specified,
///     writes up to the first null character found.
/// - `d` and `i`: Writes a decimal representation of a signed integer
///   - The [precision] specifies the minimal number to appear (defaults to `1`).
///   - If the precision is zero and the value to be written is also zero, the result is no
///     characters written.
/// - `u`: Writes the decimal representation of a unsigned integer
///   - The [precision] specifies the minimal number to appear (defaults to `1`).
///   - If the precision is zero and the value to be written is also zero, the result is no
///     characters written.
/// - `o`: Writes the octal representation of a unsigned integer.
///   - The [precision] specifies the minimal number to appear (defaults to `1`).
///   - If the precision is zero and the value to be written is also zero, the result is no
///     characters written.
///   - The alternative representation includes a leading `0`.
///   - The types are the same as `u`
/// - `x`: Writes the hexadecimal representation of a unsigned integer with lowercase characters.
///   - The [precision] specifies the minimal number to appear (defaults to `1`).
///   - If the precision is zero and the value to be written is also zero, the result is no
///     characters written.
///   - The alternative representation includes a leading `0x`.
///   - The types are the same as `u`
/// - `X`: Writes the hexadecimal representation of a unsigned integer with uppercase characters.
///   - The [precision] specifies the minimal number to appear (defaults to `1`).
///   - If the precision is zero and the value to be written is also zero, the result is no
///     characters written.
///   - The alternative representation includes a leading `0X`.
///   - The types are the same as `u`.
/// - `b` | `B` (C23): Writes the binary representation of a unsigned integer.
///   - The [precision] specifies the minimal number to appear (defaults to `1`).
///   - If the precision is zero and the value to be written is also zero, the result is no
///     characters written.
///   - The alternative representation includes a leading `0b` and `0B`, respectively.
///   - The types are the same as `u`.
/// - `f` | `F`: Writes the decimal representation of a float point number.
///   - The [precision] specifies the exact number of digits to appear after the decimal point
///     character (defaults to `6`).
///   - The alternative representation, the decimal point character is written even if no digits
///     follow it.
/// - `e` | `E`: Writes the float point number with the decimal exponential notation (\[-\]d.ddd
///   **e**±dd | \[-\]d.ddd **E**±dd)
///   - The [precision] specifies the exact number of digits to appear after the decimal point
///     character (defaults to `6`).
///   - The exponent contains at least two digits, more digits are used only if necessary.
///   - If the value is ​zero, the exponent is also ​zero​.
///   - The alternative representation: decimal point character is written even if no digits follow
///     it.
/// - `a` | `A`: Writes the float point number with the hexadecimal exponential notation (\[-\]
///   **0x**h.hhh **p**±d | \[-\] **0X**h.hhh **P**±d)
///   - The [precision] specifies the exact number of digits to appear after the hexadecimal point
///     character (defaults to `6`).
///   - If the value is ​zero, the exponent is also ​zero​.
///   - The alternative representation: decimal point character is written even if no digits follow
///     it.
/// - `g` | `G`: Writes the float point number to decimal or decimal exponent notation depending on
///   the value and the [precision].
///   - Let `P` equal the precision if nonzero, `6` if the precision is not specified, or `1` if the
///     precision is `​0`​. Then, if a conversion with style `E` would have an exponent of `X`:
///     - If `P > X ≥ −4`, the conversion is with the format `f` and precision `P − 1 − X`.
///     - Otherwise, the conversion is with the format `e` or `E` and precision `P − 1`.
///   - Unless alternative representation is requested, the trailing zeros are removed. Also the
///     decimal point character is removed if no fractional part is left.
/// - `n`: Writes the number of characters written in the call into the argument pointer
///   - It can not contain any [conversion flag], [field width], or [precision].
/// - `p`: Writes an implementation defined character sequence defining a pointer.
///
/// ### Types
/// The types expected by the format string can change with the [length modifier].
///
/// For the `c`:
/// - Without length modifier: [`int`]
/// - With `l` length modifier: [`wint_t`]
///
/// For the `s`:
/// - Without length modifier: pointer to [`char`] (`char*`, `const char*`)
/// - With `l` length modifier: pointer to [`wchar_t`] (`wchar_t*`, `const wchar_t*`)
///
/// For the `d` and `i`:
/// - Without length modifier: [`int`]
/// - With `hh` length modifier: [`signed char`]
/// - With `h` length modifier: [`short`]
/// - With `l` length modifier: [`long`]
/// - With `ll` length modifier: [`long long`]
/// - With `j` length modifier: [`intmax_t`]
/// - With `z` length modifier: [`ssize_t`]
/// - With `t` length modifier: [`ptrdiff_t`]
///
/// For the `u`, `o`, `x`, `X`, `b`, `B`:
/// - Without length modifier: [`unsigned int`]
/// - With `hh` length modifier: [`unsigned char`]
/// - With `h` length modifier: [`unsigned short`]
/// - With `l` length modifier: [`unsigned long`]
/// - With `ll` length modifier: [`unsigned long long`]
/// - With `j` length modifier: [`uintmax_t`]
/// - With `z` length modifier: [`size_t`]
/// - With `t` length modifier: [`unsigned ptrdiff_t`]
///
/// For the `f`, `F`, `e`, `E`, `a`, `A`, `g`, `G`:
/// - Without length modifier: [`double`]
/// - With `l` length modifier: [`double`]
/// - With `L` length modifier: `long double`
/// - With `H` length modifier (C23 ※): `_Decimal32`
/// - With `D` length modifier (C23 ※): `_Decimal64`
/// - With `DD` length modifier (C23 ※): `_Decimal128`
///
/// For the `n`
/// - Without length modifier: pointer to [`int`] (`int*`)
/// - With `hh` length modifier: pointer to [`signed char`] (`signed char*`)
/// - With `h` length modifier: pointer to [`short`] (`short*`)
/// - With `l` length modifier: pointer to [`long`] (`long*`)
/// - With `ll` length modifier: pointer to [`long long`] (`long long*`)
/// - With `j` length modifier: pointer to [`intmax_t`] (`intmax_t*`)
/// - With `z` length modifier: pointer to [`ssize_t`] (`ssize_t*`)
/// - With `t` length modifier: pointer to [`ptrdiff_t`] (`ptrdiff_t*`)
///
/// For the `p`, it must always be a pointer to [`void`] (`void*` | `const void*`)
///
/// [precision]: #precision
/// [field width]: #field-width
/// [length modifier]: #length-modifier
/// [conversion format]: #conversion-format
/// [`int`]: c_int
/// [`unsigned char`]: c_uchar
/// [`unsigned short`]: c_ushort
/// [`unsigned int`]: c_uint
/// [`unsigned long`]: c_ulong
/// [`unsigned long long`]: c_ulonglong
/// [`unsigned ptrdiff_t`]: ptrdiff_t
/// [`wchar_t`]: wchar_t
/// [`char`]: c_char
/// [`signed char`]: crate::c_types::c_schar
/// [`short`]: c_short
/// [`long`]: c_long
/// [`long long`]: c_longlong
/// [`double`]: c_double
/// [`void`]: c_void
///
/// # Safety
/// Behavior is undefined if any of the following conditions are violated:
/// - `format` must point to valid null-terminated string.
/// - `ap` must follow the safety contract of variable arguments of C.
pub unsafe fn printf<W: Write>(w: W, format: &CStr, ap: CustomVaList) -> c_int {
    unsafe { inner_printf(w, format, ap).unwrap_or(-1) }
}
