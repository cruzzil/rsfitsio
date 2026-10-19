//! Byte-swapping utilities used by the other FITSIO routines.
//!
//! FITS stores numbers most significant byte first, so on a little-endian
//! machine ([`BYTESWAPPED`](crate::fitsio2::BYTESWAPPED)) every numeric value
//! is swapped on the way in and out.
//!
//! Ported from CFITSIO's `swapproc.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center.
#![warn(missing_docs)]

use crate::c_types::c_long;

use crate::fitsio::INT32BIT;

/// Swaps the bytes of `nvals` shorts, one at a time.
fn ffswap2_slow(svalues: &mut [i16], nvals: c_long) {
    svalues
        .iter_mut()
        .take(nvals as usize)
        .for_each(|x| *x = (*x).swap_bytes());
}

/// Swap the bytes in the input short integers: ( 0 1 -> 1 0 )
///
/// # Parameters
///
/// * `svalues` — (IO) pointer to shorts to be swapped
/// * `nvals`   — (I) number of shorts to be swapped
pub(crate) fn ffswap2(svalues: &mut [i16], nvals: c_long) {
    ffswap2_slow(svalues, nvals);
}

/// Swaps the bytes of `nvals` 4-byte integers, one at a time.
fn ffswap4_slow(svalues: &mut [INT32BIT], nvals: c_long) {
    //let mut v = std::slice::from_raw_parts_mut(svalues, nvals as usize);
    svalues
        .iter_mut()
        .take(nvals as usize)
        .for_each(|x| *x = (*x).swap_bytes());
}

/// Swap the bytes in the input 4-byte integers: ( 0 1 2 3 -> 3 2 1 0 )
///
/// # Parameters
///
/// * `ivalues` — (IO) pointer to INT*4 to be swapped
/// * `nvals`   — (I) number of floats to be swapped
pub(crate) fn ffswap4(ivalues: &mut [INT32BIT], nvals: c_long) {
    ffswap4_slow(ivalues, nvals);
}

/// Swaps the bytes of `nvals` 8-byte values, one at a time.
fn ffswap8_slow(svalues: &mut [i64], nvals: c_long) {
    svalues
        .iter_mut()
        .take(nvals as usize)
        .for_each(|x| *x = (*x).swap_bytes());
}

/// Swap the bytes in the input doubles: ( 01234567  -> 76543210 )
///
/// # Parameters
///
/// * `ivalues` — (IO) pointer to doubles to be swapped
/// * `nvals`   — (I) number of doubles to be swapped
pub(crate) fn ffswap8(ivalues: &mut [i64], nvals: c_long) {
    ffswap8_slow(ivalues, nvals);
}

/// Swaps the bytes of `nvals` 2-byte values held in a byte buffer, which,
/// unlike [`ffswap2`]'s, need not be aligned for `i16`.
pub(crate) fn ffswap2_bytes(bytes: &mut [u8], nvals: c_long) {
    bytes
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .take(nvals as usize)
        .for_each(|value| value.reverse());
}

/// Swaps the bytes of `nvals` 4-byte values held in a byte buffer, which,
/// unlike [`ffswap4`]'s, need not be aligned for `INT32BIT`.
pub(crate) fn ffswap4_bytes(bytes: &mut [u8], nvals: c_long) {
    bytes
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .take(nvals as usize)
        .for_each(|value| value.reverse());
}

/// Swaps the bytes of `nvals` 8-byte values held in a byte buffer, which,
/// unlike [`ffswap8`]'s, need not be aligned for `i64`.
pub(crate) fn ffswap8_bytes(bytes: &mut [u8], nvals: c_long) {
    bytes
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .take(nvals as usize)
        .for_each(|value| value.reverse());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_bytes_at_an_odd_offset() {
        let mut buf: Vec<u8> = (0..16).collect();
        ffswap2_bytes(&mut buf[1..], 2);
        assert_eq!(buf[..6], [0, 2, 1, 4, 3, 5]);

        let mut buf: Vec<u8> = (0..16).collect();
        ffswap4_bytes(&mut buf[1..], 1);
        assert_eq!(buf[..6], [0, 4, 3, 2, 1, 5]);

        let mut buf: Vec<u8> = (0..16).collect();
        ffswap8_bytes(&mut buf[1..], 1);
        assert_eq!(buf[..10], [0, 8, 7, 6, 5, 4, 3, 2, 1, 9]);
    }
}
