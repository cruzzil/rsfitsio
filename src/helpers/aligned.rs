//! A heap byte buffer with an alignment guarantee.
#![warn(missing_docs)]

use alloc::collections::TryReserveError;
use core::ops::{Deref, DerefMut};

use bytemuck::{cast_slice, cast_slice_mut};

/// A heap byte buffer whose data pointer is guaranteed to be 8-byte aligned.
///
/// The tiled-image code reinterprets its scratch byte buffers as `[c_short]`,
/// `[c_int]`, `[f32]` or `[f64]` with `bytemuck::cast_slice(_mut)`, which
/// requires the byte slice to be aligned for the target type.  A plain
/// `Vec<u8>` only guarantees 1-byte alignment.  The usual system allocators
/// hand back over-aligned blocks, which is why casting a `Vec<u8>` appeared to
/// work, but nothing guarantees it: the cast panics as soon as the buffer
/// lands on an odd address.  Backing the storage with `u64` makes the
/// alignment explicit rather than incidental.
///
/// Derefs to `[u8]`, so call sites that took a `Vec<u8>` keep working
/// unchanged.
pub(crate) struct AlignedBytes {
    buf: Vec<u64>,
    len: usize,
}

impl AlignedBytes {
    pub(crate) fn new() -> Self {
        AlignedBytes {
            buf: Vec::new(),
            len: 0,
        }
    }

    /// Resize to `len` zeroed bytes, reporting allocation failure rather than
    /// aborting (the callers translate this into MEMORY_ALLOCATION).
    pub(crate) fn try_resize_zeroed(&mut self, len: usize) -> Result<(), TryReserveError> {
        let words = len.div_ceil(size_of::<u64>());
        self.buf.clear();
        self.buf.try_reserve_exact(words)?;
        self.buf.resize(words, 0);
        self.len = len;
        Ok(())
    }

    /// Resize to `len` bytes keeping the existing contents; any new bytes are
    /// zero.  Reports allocation failure, leaving the buffer unchanged.
    pub(crate) fn try_resize_keep(&mut self, len: usize) -> Result<(), TryReserveError> {
        let words = len.div_ceil(size_of::<u64>());
        if words > self.buf.len() {
            self.buf.try_reserve(words - self.buf.len())?;
        }
        let oldlen = self.len;
        self.buf.resize(words, 0);
        self.len = len;
        if len > oldlen {
            self[oldlen..].fill(0);
        }
        Ok(())
    }
}

impl Deref for AlignedBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &cast_slice(&self.buf)[..self.len]
    }
}

impl DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut cast_slice_mut(&mut self.buf)[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_is_sufficient_for_widest_cast() {
        for len in [1usize, 2, 3, 7, 8, 9, 100, 1001] {
            let mut b = AlignedBytes::new();
            b.try_resize_zeroed(len).unwrap();
            assert_eq!(b.len(), len);
            assert!(b.iter().all(|&x| x == 0));
            assert_eq!(b.as_ptr().align_offset(align_of::<f64>()), 0);
        }
    }

    #[test]
    fn test_contents_survive_a_round_trip_through_a_wider_type() {
        let mut b = AlignedBytes::new();
        b.try_resize_zeroed(16).unwrap();
        cast_slice_mut::<u8, i16>(&mut b)[3] = -2;
        assert_eq!(cast_slice::<u8, i16>(&b)[3], -2);
    }

    #[test]
    fn test_resize_keep_preserves_contents_and_zeroes_growth() {
        let mut b = AlignedBytes::new();
        b.try_resize_zeroed(5).unwrap();
        b.copy_from_slice(&[1, 2, 3, 4, 5]);
        b.try_resize_keep(3).unwrap();
        b.try_resize_keep(20).unwrap();
        assert_eq!(&b[..3], &[1, 2, 3]);
        assert!(b[3..].iter().all(|&x| x == 0));
        assert_eq!(b.as_ptr().align_offset(align_of::<f64>()), 0);
    }

    #[test]
    fn test_resize_shrinks_the_visible_length() {
        let mut b = AlignedBytes::new();
        b.try_resize_zeroed(64).unwrap();
        b.try_resize_zeroed(3).unwrap();
        assert_eq!(b.len(), 3);
    }
}
