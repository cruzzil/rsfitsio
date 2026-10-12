//! Offsetting into an optional null-flag array.

/// Return `&mut nularray[next..]`, or an empty slice if `next` is past its end.
///
/// The `ffgclXX` readers take a `nularray` that is only written when the
/// caller asked for null flags (`NullCheckType::SetNullArray`). Otherwise the
/// caller passes a one-element dummy, and C's `&nularray[next]` is pointer
/// arithmetic that is never dereferenced. Slicing the dummy in Rust would
/// panic as soon as a read spans more than one buffer-full of pixels.
pub(crate) fn nul_tail<T>(nularray: &mut [T], next: usize) -> &mut [T] {
    if next <= nularray.len() {
        &mut nularray[next..]
    } else {
        &mut []
    }
}
