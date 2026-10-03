//! A decompression output buffer that may grow.
//!
//! The C decompressors (`uncompress2mem`, `uncompress2mem_from_mem`,
//! `zuncompress2mem`) take `char **buffptr, size_t *buffsize` and a
//! `mem_realloc` function, and grow the buffer by calling it.  A Rust-owned
//! buffer must never be passed to libc `realloc`, so [`OutBuf`] says what the
//! buffer is and grows each kind with its own allocator.
#![warn(missing_docs)]

use core::slice;

use crate::helpers::aligned::AlignedBytes;

/// A C `realloc`-like function: returns the new block, holding the old
/// contents, or null on failure leaving the old block untouched.
pub(crate) type GrowFn<'a> = &'a mut dyn FnMut(*mut u8, usize) -> *mut u8;

/// Where a decompressor writes its output: the C's `buffptr`, `buffsize` and
/// `mem_realloc` arguments.
pub(crate) enum OutBuf<'a> {
    /// A buffer that cannot grow (the C passes `mem_realloc = NULL`).
    Fixed(&'a mut [u8]),
    /// A Rust-owned buffer, grown with zeroed bytes (the C passes `realloc`).
    Vec(&'a mut Vec<u8>),
    /// A Rust-owned, 8-byte aligned buffer, grown with zeroed bytes.
    Aligned(&'a mut AlignedBytes),
    /// A buffer owned elsewhere, grown by its own function; see [`RawBuf`].
    Raw(RawBuf<'a>),
}

/// A buffer the memory driver holds by address and size, possibly allocated
/// by a library user (`ffimem`), together with the function that grows it.
pub(crate) struct RawBuf<'a> {
    addr: &'a mut *mut u8,
    size: &'a mut usize,
    realloc: Option<GrowFn<'a>>,
}

impl<'a> RawBuf<'a> {
    /// # Safety
    ///
    /// `*addr` must be valid for reads and writes of `*size` bytes (or `*size`
    /// is 0), and `realloc`, if given, must behave like C `realloc` for that
    /// block, returning a block of the requested size or null.
    pub(crate) unsafe fn new(
        addr: &'a mut *mut u8,
        size: &'a mut usize,
        realloc: Option<GrowFn<'a>>,
    ) -> Self {
        RawBuf {
            addr,
            size,
            realloc,
        }
    }
}

impl OutBuf<'_> {
    /// Size of the buffer in bytes (the C's `*buffsize`).
    pub(crate) fn len(&self) -> usize {
        match self {
            OutBuf::Fixed(b) => b.len(),
            OutBuf::Vec(v) => v.len(),
            OutBuf::Aligned(a) => a.len(),
            OutBuf::Raw(r) => *r.size,
        }
    }

    /// Whether the buffer can grow (the C's `mem_realloc != NULL`).
    pub(crate) fn can_grow(&self) -> bool {
        match self {
            OutBuf::Fixed(_) => false,
            OutBuf::Vec(_) | OutBuf::Aligned(_) => true,
            OutBuf::Raw(r) => r.realloc.is_some(),
        }
    }

    /// The buffer's start address (the C's `*buffptr`).  Valid for writes of
    /// [`len`](Self::len) bytes until the next call to
    /// [`try_resize`](Self::try_resize).
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            OutBuf::Fixed(b) => b.as_mut_ptr(),
            OutBuf::Vec(v) => v.as_mut_ptr(),
            OutBuf::Aligned(a) => a.as_mut_ptr(),
            OutBuf::Raw(r) => *r.addr,
        }
    }

    /// The whole buffer as a slice.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            OutBuf::Fixed(b) => b,
            OutBuf::Vec(v) => v,
            OutBuf::Aligned(a) => a,
            OutBuf::Raw(r) => {
                if *r.size == 0 {
                    &mut []
                } else {
                    // SAFETY: RawBuf::new's contract, kept by try_resize.
                    unsafe { slice::from_raw_parts_mut(*r.addr, *r.size) }
                }
            }
        }
    }

    /// Resize the buffer to `newsize` bytes, keeping its contents: the C's
    /// `*buffptr = mem_realloc(*buffptr, newsize)`.
    ///
    /// Returns false if the buffer cannot grow or the allocation fails; the
    /// buffer is then unchanged.  (The C stores the null pointer in
    /// `*buffptr`, losing the old block.)
    pub(crate) fn try_resize(&mut self, newsize: usize) -> bool {
        match self {
            OutBuf::Fixed(_) => false,
            OutBuf::Vec(v) => {
                if newsize > v.len() && v.try_reserve(newsize - v.len()).is_err() {
                    return false;
                }
                v.resize(newsize, 0);
                true
            }
            OutBuf::Aligned(a) => a.try_resize_keep(newsize).is_ok(),
            OutBuf::Raw(r) => {
                let Some(realloc) = r.realloc.as_mut() else {
                    return false;
                };
                let p = realloc(*r.addr, newsize);
                if p.is_null() {
                    return false;
                }
                *r.addr = p;
                *r.size = newsize;
                true
            }
        }
    }
}
