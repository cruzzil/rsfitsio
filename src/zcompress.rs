//! GZIP compression of a byte stream, over `libz-rs-sys`.
//!
//! Used when a FITS file is written to a `.gz` name, and by the `GZIP_1` and
//! `GZIP_2` tile compressors in [`crate::imcompress`].
#![warn(missing_docs)]

use core::ptr;
use std::io::{Read, Write};

use crate::c_types::{c_char, c_int, c_uint, c_ulong};

use libz_rs_sys::{
    Z_BEST_SPEED, Z_BUF_ERROR, Z_DEFAULT_STRATEGY, Z_DEFLATED, Z_FINISH, Z_NO_FLUSH, Z_OK,
    Z_STREAM_END, Z_STREAM_ERROR, deflate, deflateEnd, deflateInit2_, inflate, inflateEnd,
    inflateInit2_, uInt, uLong, voidpf, z_stream, z_streamp, zlibVersion,
};

use crate::fitsio::{DATA_COMPRESSION_ERR, DATA_DECOMPRESSION_ERR, MEMORY_ALLOCATION};
use crate::helpers::outbuf::OutBuf;

const GZBUFSIZE: usize = 115200; /* 40 FITS blocks */
const BUFFINCR: usize = 28800; /* 10 FITS blocks */

pub(crate) unsafe fn inflateInit2(strm: z_streamp, windowBits: c_int) -> c_int {
    unsafe {
        inflateInit2_(
            strm,
            windowBits,
            zlibVersion(),
            core::mem::size_of::<z_stream>() as c_int,
        )
    }
}

pub(crate) unsafe fn deflateInit2(
    strm: z_streamp,
    level: c_int,
    method: c_int,
    windowBits: c_int,
    memLevel: c_int,
    strategy: c_int,
) -> c_int {
    unsafe {
        deflateInit2_(
            strm,
            level,
            method,
            windowBits,
            memLevel,
            strategy,
            zlibVersion(),
            core::mem::size_of::<z_stream>() as c_int,
        )
    }
}

/// Uncompress the disk file into memory.  Fill whatever amount of memory has
/// already been allocated, then realloc more memory, using the supplied
/// input function, if necessary.
///
/// The body is one `unsafe` block because zlib reads and writes through the
/// `z_stream`'s raw `next_in`/`next_out` pointers.  They are only ever set
/// from `filebuff` and from `out`, whose length bounds `avail_out`.
///
/// # Parameters
///
/// * `_filename`   — name of input file
/// * `diskfile`    — (I) file pointer
/// * `out`         — (IO) memory buffer, and whether it may grow (the C's
///   `buffptr`, `buffsize` and `mem_realloc`)
/// * `filesize`    — (O) size of file, in bytes
/// * `status`      — (IO) error status
pub(crate) fn uncompress2mem<T: Read>(
    _filename: &[c_char],
    diskfile: &mut T,
    mut out: OutBuf<'_>,
    filesize: &mut usize,
    status: &mut c_int,
) -> c_int {
    unsafe {
        let mut err: c_int = 0;
        let mut d_stream: z_stream; /* decompression stream */
        /* Input args buffptr and buffsize may refer to a block of memory
        larger than the 2^32 4 byte limit.  If so, must be broken
        up into "pages" when assigned to d_stream.
        (d_stream.avail_out is a uInt type, which might be smaller
        than buffsize's size_t type.)
        */
        let nPages: libz_rs_sys::uLong = (out.len() as uLong) / uLong::from(c_uint::MAX);
        let mut iPage: uLong = 0;
        let outbuffsize: uInt = if nPages > 0 {
            c_uint::MAX
        } else {
            out.len() as uInt
        };

        if *status > 0 {
            return *status;
        }

        /* Allocate memory to hold compressed bytes read from the file. */
        let mut filebuff = Vec::new();
        if filebuff.try_reserve_exact(GZBUFSIZE).is_err() {
            *status = MEMORY_ALLOCATION;
            return *status; /* memory error */
        } else {
            filebuff.resize(GZBUFSIZE, 0);
        }

        d_stream = z_stream {
            next_in: ptr::null_mut(),
            avail_in: Default::default(),
            total_in: Default::default(),
            next_out: out.as_mut_ptr(),
            avail_out: outbuffsize,
            total_out: Default::default(),
            msg: ptr::null_mut(),
            state: ptr::null_mut(),
            /* the C sets these to NULL so zlib uses its own allocator */
            zalloc: None,
            zfree: None,
            opaque: ptr::null_mut() as voidpf,
            data_type: Default::default(),
            adler: Default::default(),
            reserved: Default::default(),
        };

        /* Initialize the decompression.  The argument (15+16) tells the
        decompressor that we are to use the gzip algorithm */

        err = inflateInit2(&raw mut d_stream, 15 + 16);
        if err != Z_OK {
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        /* loop through the file, reading a buffer and uncompressing it */
        loop {
            let len = diskfile.read(&mut filebuff[..GZBUFSIZE]);
            if len.is_err() {
                inflateEnd(&raw mut d_stream);
                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }

            let len = len.unwrap();

            if len == 0 {
                break; /* no more data */
            }

            d_stream.next_in = filebuff.as_mut_ptr();
            d_stream.avail_in = len as uInt;

            loop {
                /* uncompress as much of the input as will fit in the output */
                err = inflate(&raw mut d_stream, Z_NO_FLUSH);

                if err == Z_STREAM_END {
                    /* We reached the end of the input */
                    break;
                } else if err == Z_OK || err == Z_BUF_ERROR {
                    /* Z_BUF_ERROR means need more input data to make progress */
                    if d_stream.avail_in == 0 {
                        break; /* need more input */
                    }

                    /* need more space in output buffer */
                    /* First check if more memory is available above the 4Gb limit in the originally input buffptr array */
                    if iPage < nPages {
                        iPage += 1;
                        d_stream.next_out = out
                            .as_mut_ptr()
                            .add((iPage * uLong::from(c_uint::MAX)) as usize);
                        if iPage < nPages {
                            d_stream.avail_out = c_uint::MAX;
                        } else {
                            d_stream.avail_out =
                                ((out.len() as uLong) % uLong::from(c_uint::MAX)) as uInt;
                        }
                    } else if out.can_grow() {
                        /* *buffptr = mem_realloc(*buffptr,*buffsize + BUFFINCR); */
                        let buffsize = out.len();
                        if !out.try_resize(buffsize + BUFFINCR) {
                            inflateEnd(&raw mut d_stream);
                            *status = DATA_DECOMPRESSION_ERR;
                            return *status; /* memory allocation failed */
                        }

                        d_stream.avail_out = BUFFINCR as uInt;
                        d_stream.next_out = out.as_mut_ptr().add(buffsize);
                    } else {
                        /* error: no realloc function available */
                        inflateEnd(&raw mut d_stream);
                        *status = DATA_DECOMPRESSION_ERR;
                        return *status;
                    }
                } else {
                    /* some other error */
                    inflateEnd(&raw mut d_stream);
                    *status = DATA_DECOMPRESSION_ERR;
                    return *status;
                }
            }

            /*
            // Should not happen since break occurs higher
            if feof(diskfile) {
                break;
            }
            */

            /*
            These settings for next_out and avail_out appear to be redundant,
            as the inflate() function should already be re-setting these.
            For case where *buffsize < 4Gb this did not matter, but for
            > 4Gb it would produce the wrong value in the avail_out assignment.
            (C. Gordon Jul 2016)
            d_stream.next_out = (unsigned char*) (*buffptr + d_stream.total_out);
            d_stream.avail_out = *buffsize - d_stream.total_out;
            */
        }

        /* Set the output file size to be the total output data */
        *filesize = d_stream.total_out as usize;

        err = inflateEnd(&raw mut d_stream); /* End the decompression */
        if err != Z_OK {
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        *status
    }
}

/// Uncompress the file in memory into memory.  Fill whatever amount of memory has
/// already been allocated, then realloc more memory, using the supplied
/// input function, if necessary.
///
/// The body is one `unsafe` block because zlib reads and writes through the
/// `z_stream`'s raw `next_in`/`next_out` pointers.  They are only ever set
/// from `inmemptr` and from `out`, whose length bounds `avail_out`.
///
/// # Parameters
///
/// * `inmemptr`    — (I) memory pointer to compressed bytes
/// * `inmemsize`   — (I) size of input compressed file
/// * `out`         — (IO) memory buffer, and whether it may grow (the C's
///   `buffptr`, `buffsize` and `mem_realloc`)
/// * `filesize`    — (O) size of file, in bytes
/// * `status`      — (IO) error status
pub(crate) fn uncompress2mem_from_mem(
    inmemptr: &[c_char],
    inmemsize: usize,
    mut out: OutBuf<'_>,
    filesize: Option<&mut usize>,
    status: &mut c_int,
) -> c_int {
    unsafe {
        let mut err: c_int = 0;
        let mut d_stream: z_stream; /* decompression stream */

        if *status > 0 {
            return *status;
        }

        /* zlib must not read past the input slice */
        let inmemsize = inmemsize.min(inmemptr.len());

        d_stream = z_stream {
            next_in: ptr::null_mut(),
            avail_in: Default::default(),
            total_in: Default::default(),
            next_out: ptr::null_mut(),
            avail_out: Default::default(),
            total_out: Default::default(),
            msg: ptr::null_mut(),
            state: ptr::null_mut(),
            /* the C sets these to NULL so zlib uses its own allocator */
            zalloc: None,
            zfree: None,
            opaque: ptr::null_mut() as voidpf,
            data_type: Default::default(),
            adler: Default::default(),
            reserved: Default::default(),
        };

        /* Initialize the decompression.  The argument (15+16) tells the
        decompressor that we are to use the gzip algorithm */
        err = inflateInit2(&raw mut d_stream, 15 + 16);
        if err != Z_OK {
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        d_stream.next_in = inmemptr.as_ptr() as *mut u8; // Yes convert from const to mut
        d_stream.avail_in = inmemsize as uInt;

        d_stream.next_out = out.as_mut_ptr();
        /* the C truncates *buffsize to uInt; clamp so a buffer over 4 GB is
        not mistaken for a tiny one */
        d_stream.avail_out = out.len().min(uInt::MAX as usize) as uInt;

        loop {
            /* uncompress as much of the input as will fit in the output */
            err = inflate(&raw mut d_stream, Z_NO_FLUSH);

            if err == Z_STREAM_END {
                /* We reached the end of the input */
                break;
            } else if err == Z_OK || err == Z_BUF_ERROR {
                /* need more space in output buffer */
                /* Z_BUF_ERROR means need more input data to make progress */

                if out.can_grow() {
                    /* All the input is consumed and there is still room for
                    output, so the stream is truncated and more memory cannot
                    help.  The C reallocs here until realloc fails, then returns
                    the same status; stop now instead. */
                    if d_stream.avail_in == 0 && d_stream.avail_out > 0 {
                        inflateEnd(&raw mut d_stream);
                        *status = DATA_DECOMPRESSION_ERR;
                        return *status;
                    }

                    /* *buffptr = mem_realloc(*buffptr,*buffsize + BUFFINCR); */
                    let buffsize = out.len();
                    if !out.try_resize(buffsize + BUFFINCR) {
                        inflateEnd(&raw mut d_stream);
                        *status = DATA_DECOMPRESSION_ERR;
                        return *status; /* memory allocation failed */
                    }

                    d_stream.avail_out = BUFFINCR as uInt;
                    d_stream.next_out = out.as_mut_ptr().add(buffsize);
                } else {
                    /* error: no realloc function available */
                    inflateEnd(&raw mut d_stream);
                    if let Some(filesize) = filesize {
                        *filesize = d_stream.total_out as usize;
                    }
                    *status = DATA_DECOMPRESSION_ERR;
                    return *status;
                }
            } else {
                /* some other error */
                inflateEnd(&raw mut d_stream);
                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }
        }

        /* Set the output file size to be the total output data */
        if let Some(filesize) = filesize {
            *filesize = d_stream.total_out as usize;
        }

        /* End the decompression */
        err = inflateEnd(&raw mut d_stream);

        if err != Z_OK {
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        *status
    }
}

/// Uncompress the file into another file.
///
/// The body is one `unsafe` block because it is zlib stream manipulation throughout:
/// every call reads and writes through `z_stream`'s `next_in`/`next_out` raw
/// pointers.  The function itself is safe: it owns both buffers and is the only
/// thing that sets those pointers, so no caller can influence them.
///
/// # Parameters
///
/// * `_filename`   — name of input file
/// * `indiskfile`  — (I) input file pointer
/// * `outdiskfile` — (I) output file pointer
/// * `status`      — (IO) error status
pub(crate) fn uncompress2file<R: Read, W: Write>(
    _filename: &[c_char],
    indiskfile: &mut R,
    outdiskfile: &mut W,
    status: &mut c_int,
) -> c_int {
    unsafe {
        let mut err: c_int = 0;
        let mut bytes_out: c_ulong = 0;

        if *status > 0 {
            return *status;
        }

        /* Allocate buffers to hold compressed and uncompressed */
        let mut infilebuff: Vec<u8> = Vec::new();
        if infilebuff.try_reserve_exact(GZBUFSIZE).is_err() {
            *status = MEMORY_ALLOCATION;
            return *status; /* memory error */
        } else {
            infilebuff.resize(GZBUFSIZE, 0);
        }

        let mut outfilebuff: Vec<u8> = Vec::new();
        if outfilebuff.try_reserve_exact(GZBUFSIZE).is_err() {
            *status = MEMORY_ALLOCATION;
            return *status; /* memory error */
        } else {
            outfilebuff.resize(GZBUFSIZE, 0);
        }

        /* decompression stream */
        let mut d_stream: z_stream = z_stream {
            next_in: ptr::null_mut(),
            avail_in: Default::default(),
            total_in: Default::default(),
            next_out: outfilebuff.as_mut_ptr(),
            avail_out: GZBUFSIZE as uInt,
            total_out: Default::default(),
            msg: ptr::null_mut(),
            state: ptr::null_mut(),
            /* the C sets these to NULL so zlib uses its own allocator */
            zalloc: None,
            zfree: None,
            opaque: ptr::null_mut() as voidpf,
            data_type: Default::default(),
            adler: Default::default(),
            reserved: Default::default(),
        };

        /* Initialize the decompression.  The argument (15+16) tells the
        decompressor that we are to use the gzip algorithm */

        err = inflateInit2(&raw mut d_stream, 15 + 16);
        if err != Z_OK {
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        // This is used to keep track of the last read length
        // If last read length is 0, it means we are at the end of the file
        let mut last_read_len: usize = 0;

        /* loop through the file, reading a buffer and uncompressing it */
        loop {
            let len = indiskfile.read(&mut infilebuff[..GZBUFSIZE]);

            if len.is_err() {
                inflateEnd(&raw mut d_stream);
                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }

            last_read_len = len.unwrap();

            if last_read_len == 0 {
                break; /* no more data */
            }

            d_stream.next_in = infilebuff.as_ptr();
            d_stream.avail_in = last_read_len as uInt;

            loop {
                /* uncompress as much of the input as will fit in the output */
                err = inflate(&raw mut d_stream, Z_NO_FLUSH);

                if err == Z_STREAM_END {
                    /* We reached the end of the input */
                    break;
                } else if err == Z_OK || err == Z_BUF_ERROR {
                    /* Z_BUF_ERROR means need more input data to make progress */

                    if d_stream.avail_in == 0 {
                        break; /* need more input */
                    }

                    /* flush out the full output buffer */
                    let out_len = outdiskfile.write(&outfilebuff[..GZBUFSIZE]);
                    if out_len.is_err() {
                        inflateEnd(&raw mut d_stream);
                        *status = DATA_DECOMPRESSION_ERR;
                        return *status;
                    }

                    last_read_len = out_len.unwrap();

                    if last_read_len != GZBUFSIZE {
                        inflateEnd(&raw mut d_stream);
                        *status = DATA_DECOMPRESSION_ERR;
                        return *status;
                    }

                    bytes_out += GZBUFSIZE as c_ulong;
                    d_stream.next_out = outfilebuff.as_mut_ptr();
                    d_stream.avail_out = GZBUFSIZE as _;
                } else {
                    /* some other error */
                    inflateEnd(&raw mut d_stream);
                    *status = DATA_DECOMPRESSION_ERR;
                    return *status;
                }
            }

            /* Check for end of file */
            if last_read_len == 0 {
                // In rust, 0 indicates EOF
                break;
            }
        }

        /* write out any remaining bytes in the buffer */
        if d_stream.total_out > bytes_out {
            let out_len =
                outdiskfile.write(&outfilebuff[..(d_stream.total_out - bytes_out) as usize]);

            if out_len.is_err() {
                inflateEnd(&raw mut d_stream);
                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }

            if out_len.unwrap() != (d_stream.total_out - bytes_out) as usize {
                inflateEnd(&raw mut d_stream);
                *status = DATA_DECOMPRESSION_ERR;
                return *status;
            }
        }

        err = inflateEnd(&raw mut d_stream); /* End the decompression */
        if err != Z_OK {
            *status = DATA_DECOMPRESSION_ERR;
            return *status;
        }

        *status
    }
}

/// Compress the file into memory.  Fill whatever amount of memory has
/// already been allocated, then realloc more memory, using the supplied
/// input function, if necessary.
///
/// # Parameters
///
/// * `inmemptr`  — (I) memory pointer to uncompressed bytes
/// * `inmemsize` — (I) size of input uncompressed file
/// * `outbuf`    — (IO) buffer for compressed file; grown as needed
/// * `filesize`  — (O) size of compressed file, in bytes
/// * `status`    — (IO) error status
pub(crate) fn compress2mem_from_mem(
    inmemptr: &[c_char],
    inmemsize: usize,
    outbuf: &mut Vec<u8>,
    filesize: Option<&mut usize>,
    status: &mut c_int,
) -> c_int {
    let mut err: c_int;
    let mut c_stream: z_stream; /* compression stream */

    if *status > 0 {
        return *status;
    }

    /* make sure there is some output space to start with. If the caller
    pre-sized the buffer we honor that; otherwise start with one chunk.
    The buffer is owned by a Rust Vec and grown with the Rust allocator,
    so there is no dangling pointer or allocator mismatch (see issue #60). */
    if outbuf.is_empty() {
        outbuf.resize(BUFFINCR, 0);
    }

    c_stream = z_stream {
        next_in: ptr::null_mut(),
        avail_in: Default::default(),
        total_in: Default::default(),
        next_out: ptr::null_mut(),
        avail_out: Default::default(),
        total_out: Default::default(),
        msg: ptr::null_mut(),
        state: ptr::null_mut(),
        /* the C sets these to NULL so zlib uses its own allocator */
        zalloc: None,
        zfree: None,
        opaque: ptr::null_mut() as voidpf,
        data_type: Default::default(),
        adler: Default::default(),
        reserved: Default::default(),
    };

    /* Initialize the compression.  The argument (15+16) tells the
    compressor that we are to use the gzip algorithm.
    Also use Z_BEST_SPEED for maximum speed with very minor loss
    in compression factor. */
    /* SAFETY: c_stream is a fully initialised z_stream that outlives every zlib
    call below; the buffers it points at are the two slices above. */
    err = unsafe {
        deflateInit2(
            &raw mut c_stream,
            Z_BEST_SPEED,
            Z_DEFLATED,
            15 + 16,
            8,
            Z_DEFAULT_STRATEGY,
        )
    };

    if err != Z_OK {
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    c_stream.next_in = inmemptr.as_ptr() as *mut u8; // WARNING: Yes convert const to mut....
    c_stream.avail_in = inmemsize as uInt;

    c_stream.next_out = outbuf.as_mut_ptr();
    c_stream.avail_out = outbuf.len() as uInt;

    loop {
        /* compress as much of the input as will fit in the output */
        err = unsafe { deflate(&raw mut c_stream, Z_FINISH) };

        if err == Z_STREAM_END {
            /* We reached the end of the input */
            break;
        } else if err == Z_OK {
            /* need more space in output buffer: grow the Vec and re-point
            the zlib stream at the (possibly moved) buffer. total_out is the
            number of bytes already written from the start of the buffer. */
            let written = c_stream.total_out as usize;
            let newlen = outbuf.len() + BUFFINCR;
            outbuf.resize(newlen, 0);
            /* SAFETY: `written` <= outbuf.len(), so this stays in the Vec. */
            c_stream.next_out = unsafe { outbuf.as_mut_ptr().add(written) };
            c_stream.avail_out = (outbuf.len() - written) as uInt;
        } else {
            /* some other error */
            unsafe { deflateEnd(&raw mut c_stream) };
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }
    }

    /* Set the output file size to be the total output data */
    if let Some(filesize) = filesize {
        *filesize = c_stream.total_out as usize;
    }

    /* End the compression */
    err = unsafe { deflateEnd(&raw mut c_stream) };

    if err != Z_OK {
        *status = DATA_COMPRESSION_ERR;
        return *status;
    }

    *status
}

/// Compress the memory file into disk file.
///
/// The body is one `unsafe` block because it is zlib stream manipulation throughout:
/// every call reads and writes through `z_stream`'s `next_in`/`next_out` raw
/// pointers.  The function itself is safe: it owns both buffers and is the only
/// thing that sets those pointers, so no caller can influence them.
///
/// # Parameters
///
/// * `inmemptr`  — (I) memory pointer to uncompressed bytes
/// * `inmemsize` — (I) size of input uncompressed file
pub(crate) fn compress2file_from_mem<W: Write>(
    inmemptr: &[c_char],
    inmemsize: usize,
    outdiskfile: &mut W,
    filesize: Option<&mut usize>,
    status: &mut c_int,
) -> c_int {
    unsafe {
        let mut err: c_int = 0;
        let mut flushflag: c_int = 0;

        let mut nPages: uLong = 1;
        let mut nBytesToFile: uInt = 0;

        let mut c_stream: z_stream; /* compression stream */

        if *status > 0 {
            return *status;
        }

        /* Allocate buffer to hold compressed bytes */
        let mut outfilebuff: Vec<u8> = Vec::new();
        if outfilebuff.try_reserve_exact(GZBUFSIZE).is_err() {
            *status = MEMORY_ALLOCATION;
            return *status; /* memory error */
        } else {
            outfilebuff.resize(GZBUFSIZE, 0);
        }

        c_stream = z_stream {
            next_in: ptr::null_mut(),
            avail_in: Default::default(),
            total_in: Default::default(),
            next_out: ptr::null_mut(),
            avail_out: Default::default(),
            total_out: Default::default(),
            msg: ptr::null_mut(),
            state: ptr::null_mut(),
            /* the C sets these to NULL so zlib uses its own allocator */
            zalloc: None,
            zfree: None,
            opaque: ptr::null_mut() as voidpf,
            data_type: Default::default(),
            adler: Default::default(),
            reserved: Default::default(),
        };

        /* Initialize the compression.  The argument (15+16) tells the
        compressor that we are to use the gzip algorythm.
        Also use Z_BEST_SPEED for maximum speed with very minor loss
        in compression factor. */
        err = deflateInit2(
            &raw mut c_stream,
            Z_BEST_SPEED,
            Z_DEFLATED,
            15 + 16,
            8,
            Z_DEFAULT_STRATEGY,
        );

        if err != Z_OK {
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }

        // ####
        if inmemsize > 0 {
            nPages = 1 + (inmemsize as uLong - 1) / uLong::from(uInt::MAX);
        }

        /*
        c_stream.next_in = inmemptr.as_ptr() as *mut u8; // Yes convert const to mut
        c_stream.avail_in = inmemsize as uInt;

        c_stream.next_out = outfilebuff;
        c_stream.avail_out = GZBUFSIZE as uInt;

        loop {
            /* compress as much of the input as will fit in the output */
            err = deflate(&mut c_stream, Z_FINISH);

            if err == Z_STREAM_END {
                /* We reached the end of the input */
                break;
            } else if err == Z_OK {
                /* need more space in output buffer */

                /* flush out the full output buffer */
                if fwrite(outfilebuff as *const c_void, 1, GZBUFSIZE, outdiskfile) != GZBUFSIZE {
                    deflateEnd(&mut c_stream);
                    free(outfilebuff as *mut c_void);
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                }
                bytes_out += GZBUFSIZE as c_ulong;
                c_stream.next_out = outfilebuff;
                c_stream.avail_out = GZBUFSIZE as uInt;
            } else {
                /* some other error */
                deflateEnd(&mut c_stream);
                free(outfilebuff as *mut c_void);
                *status = DATA_COMPRESSION_ERR;
                return *status;
            }
        }
        */

        for iPage in 0..nPages {
            // SAFETY: Converted a const pointer to a mutable pointer, don't know why it needs to be.
            c_stream.next_in =
                (inmemptr.as_ptr() as *mut u8).add((iPage * uLong::from(uInt::MAX)) as usize);
            c_stream.avail_in = if iPage == nPages - 1 {
                (inmemsize as uLong - iPage * uLong::from(uInt::MAX)) as uInt
            } else {
                uInt::MAX
            };

            flushflag = if iPage < nPages - 1 {
                Z_NO_FLUSH
            } else {
                Z_FINISH
            };
            loop {
                c_stream.next_out = outfilebuff.as_mut_ptr();
                c_stream.avail_out = GZBUFSIZE as uInt;

                /* compress as much of the input as will fit in the output */
                err = deflate(&raw mut c_stream, flushflag);

                if err == Z_STREAM_ERROR {
                    deflateEnd(&raw mut c_stream);
                    *status = DATA_COMPRESSION_ERR;
                    return *status;
                } else {
                    /* c_stream.avail_out will be 0 unless we've reached the end of the avail_in
                    stream.  When that happens avail_out MAY also be 0, if by chance the output
                    buffer fills up just as the input stream ends.  That's OK though, as it will
                    execute just one more do/while where the deflate call won't actually do
                    anything.  */
                    nBytesToFile = GZBUFSIZE as uInt - c_stream.avail_out;
                    if nBytesToFile != 0 {
                        let out_len = outdiskfile.write(&outfilebuff[..nBytesToFile as usize]);
                        if out_len.is_err() {
                            deflateEnd(&raw mut c_stream);
                            *status = DATA_COMPRESSION_ERR;
                            return *status;
                        }

                        if out_len.unwrap() != nBytesToFile as usize {
                            deflateEnd(&raw mut c_stream);
                            *status = DATA_COMPRESSION_ERR;
                            return *status; /* write error */
                        }
                    }
                }
                if c_stream.avail_out != 0 {
                    break;
                }
            }
        }

        /* Set the output file size to be the total output data */
        if let Some(filesize) = filesize {
            *filesize = c_stream.total_out as usize;
        }

        /* End the compression */
        err = deflateEnd(&raw mut c_stream);

        if err != Z_OK {
            *status = DATA_COMPRESSION_ERR;
            return *status;
        }

        *status
    }
}
