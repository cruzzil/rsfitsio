//! LZW decompression, for reading files compressed by the Unix `compress`
//! utility.
//!
//! An old implementation of the algorithm, carried over from CFITSIO so that a
//! `.Z` file can still be read. The compression side is not provided; new files
//! are written with GZIP by [`crate::zcompress`].
#![warn(missing_docs)]

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    c_types::{FILE, c_char, c_int, c_long, c_uchar, c_uint, c_ulong, c_ushort},
    helpers::cfile::CFile,
};
use libc::EOF;
use std::io::{Read, Write};

use crate::{
    fitscore::{ffpmsg_slice, ffpmsg_str},
    fitsio::DATA_DECOMPRESSION_ERR,
    helpers::outbuf::OutBuf,
    wrappers::strncat_safe,
};

/* Return codes from gzip */
const OK: c_int = 0;
const ERROR: c_int = 1;
const COMPRESSED: c_int = 1;
const DEFLATED: c_int = 8;
const INBUFSIZ: usize = 0x8000; /* input buffer size */
const INBUF_EXTRA: usize = 64; /* required by unlzw() */
const OUTBUFSIZ: usize = 16384; /* output buffer size */
const OUTBUF_EXTRA: usize = 2048; /* required by unlzw() */
const DIST_BUFSIZE: usize = 0x8000; /* buffer for distances, see trees.c */
const WSIZE: usize = 0x8000; /* window size--must be a power of two, and */

const LZW_MAGIC: [u8; 2] = [0x1f, 0x9d]; /* magic header for LZW files */

const BITS: c_int = 16; /* code bits */
const INIT_BITS: c_int = 9; /* initial number of bits/code */
const BIT_MASK: c_int = 0x1f; /* mask for 'n_bits' bits */
const BLOCK_MODE: c_int = 0x80;
const LZW_RESERVED: c_int = 0x60; /* reserved bits */
const CLEAR: c_int = 256; /* table clear output code*/
const FIRST: c_int = CLEAR + 1; /* first free entry */

type char_type = c_uchar;
type code_int = c_long;
type count_int = c_ulong;
type count_short = c_ushort;
type cmp_code_int = c_ulong;

macro_rules! MAXCODE {
    ($n:expr) => {
        (1 << $n)
    };
}

/// The decompressor's working state: the buffers, the LZW code table and the
/// input and output streams.
///
/// The C keeps all of this in file-scope globals; gathering it into one struct
/// is what lets the routines below be safe methods rather than functions
/// reaching for mutable statics.
struct LZW_Compress<'a> {
    /// Input buffer.
    inbuf: [c_uchar; INBUFSIZ + INBUF_EXTRA],
    /// Output buffer.
    outbuf: [c_uchar; OUTBUFSIZ + OUTBUF_EXTRA],
    /// Distance buffer.
    d_buf: [c_ushort; DIST_BUFSIZE],
    /// Sliding window of already-decompressed output.
    window: [c_uchar; 2 * WSIZE],
    /// Prefix code for each entry of the LZW table.
    tab_prefix: [c_ushort; 1 << BITS],
    /// Maximum bits per code for LZW.
    maxbits: c_int,
    /// Compression method.
    method: c_int,
    /// Program exit code.
    exit_code: c_int,
    /// Set for `.zip` and `.Z` files.
    last_member: c_int,
    /// Number of input bytes.
    bytes_in: usize,
    /// Number of output bytes.
    bytes_out: usize,
    /// Input file name.
    ifname: [c_char; 128],
    /// Input file descriptor.
    ifd: *mut FILE,
    /// Output file descriptor.
    ofd: *mut FILE,
    /// Memory for the uncompressed file, and how to grow it (the C's
    /// `memptr`, `memsize` and `realloc_fn`); `None` writes to `ofd`.
    out: Option<OutBuf<'a>>,
    /// Valid bytes in `inbuf`.
    insize: usize,
    /// Index of the next byte to be processed in `inbuf`.
    inptr: usize,
    /// Block compress mode; `-C` compatible with 2.0.
    block_mode: c_int,
}

impl<'a> LZW_Compress<'a> {
    fn get_byte(&mut self) -> u8 {
        if self.inptr < self.insize {
            let v = self.inbuf[self.inptr];
            self.inptr += 1;
            v
        } else {
            self.fill_inbuf(0) as u8
        }
    }

    /* =========================================================================== */
    /// set if EOF acceptable as a result
    fn fill_inbuf(&mut self, eof_ok: c_int) -> c_int {
        /* Read as much as possible from file */
        self.insize = 0;
        loop {
            let mut cfile: CFile = self.ifd.into();

            let last_read_len = cfile.read(&mut self.inbuf[self.insize..INBUFSIZ]);

            match last_read_len {
                Err(_) => {
                    break; // Unexpected error while reading
                }
                Ok(0) => {
                    break; // EOF
                }
                Ok(len) => {
                    self.insize += len;
                }
            }

            if self.insize >= INBUFSIZ {
                break;
            }
        }

        if self.insize == 0 {
            if eof_ok != 0 {
                return EOF;
            }
            self.error("unexpected end of file");
            self.exit_code = ERROR;
            return ERROR;
        }

        self.bytes_in += self.insize;
        self.inptr = 1;
        c_int::from(self.inbuf[0])
    }

    /* =========================================================================== */
    ///  copy buffer into memory; allocate more memory if required
    fn write_buf(&mut self, cnt: usize) {
        let buf = &self.outbuf[..cnt];
        if let Some(out) = self.out.as_mut() {
            /* get more memory if current buffer is too small */
            if self.bytes_out + cnt > out.len() {
                /* *memptr = realloc_fn(*memptr, bytes_out + cnt); */
                if !out.try_resize(self.bytes_out + cnt) {
                    self.error("malloc failed while uncompressing (write_buf)");
                    self.exit_code = ERROR;
                    return;
                }
            }
            /* copy into memory buffer */
            out.as_mut_slice()[self.bytes_out..self.bytes_out + cnt].copy_from_slice(buf);
        } else {
            /* append buffer to file */

            let mut cfile = CFile::from(self.ofd);

            let last_write_len = cfile.write(buf);

            if last_write_len.is_err() || last_write_len.unwrap() != cnt {
                self.error("failed to write buffer to uncompressed output file (write_buf)");
                self.exit_code = ERROR;
            }
        }
    }

    /* ======================================================================== */
    /// Error handler
    fn error(&self, m: &str) {
        ffpmsg_slice(&self.ifname);
        ffpmsg_str(m);
    }
}

/// Uncompress the file into memory.  Fill whatever amount of memory has
/// already been allocated, then realloc more memory, using the supplied
/// input function, if necessary.
///
/// # Parameters
///
/// * `filename`    — name of input file
/// * `indiskfile`  — (I) file pointer
/// * `out`         — (IO) memory buffer, and whether it may grow (the C's
///   `buffptr`, `buffsize` and `mem_realloc`)
/// * `filesize`    — (O) size of file, in bytes
/// * `status`      — (IO) error status
pub(crate) fn zuncompress2mem(
    filename: &[c_char],
    indiskfile: *mut FILE,
    out: OutBuf<'_>,
    filesize: &mut usize,
    status: &mut c_int,
) -> c_int {
    let mut magic: [c_char; 2] = [0; 2]; /* magic header */

    if *status != 0 {
        return *status;
    }

    let mut fn_buffer: [c_char; 128] = [0; 128]; /* buffer for file name */
    strncat_safe(&mut fn_buffer[..128], filename, 127);

    let mut lzw: LZW_Compress = LZW_Compress {
        inbuf: [0; INBUFSIZ + INBUF_EXTRA],
        outbuf: [0; OUTBUFSIZ + OUTBUF_EXTRA],
        d_buf: [0; DIST_BUFSIZE],
        window: [0; 2 * WSIZE],
        tab_prefix: [0; 1 << BITS],
        maxbits: 0,
        method: COMPRESSED,
        exit_code: 0,
        last_member: 1,
        bytes_in: 0,
        bytes_out: 0,
        ifname: fn_buffer,
        ifd: indiskfile,
        ofd: core::ptr::null_mut(),
        out: Some(out),
        insize: 0,
        inptr: 0,
        block_mode: BLOCK_MODE,
    };

    magic[0] = lzw.get_byte() as c_char;
    magic[1] = lzw.get_byte() as c_char;

    if magic[0] != LZW_MAGIC[0] as c_char || magic[1] != LZW_MAGIC[1] as c_char {
        lzw.error("ERROR: input .Z file is in unrecognized compression format.\n");
        *status = ERROR;
        return ERROR;
    }

    if unlzw(&mut lzw, indiskfile, core::ptr::null_mut()) != OK {
        *status = DATA_DECOMPRESSION_ERR;
    }

    *filesize = lzw.bytes_out;

    *status
}

/// Decompress in to out.  This routine adapts to the codes in the
/// file building the "string" table on-the-fly; requiring no table to
/// be stored in the compressed file.
/// IN assertions: the buffer inbuf contains already the beginning of
/// the compressed data, from offsets iptr to insize-1 included.
/// The magic header has already been checked and skipped.
/// bytes_in and bytes_out have been initialized.
fn unlzw(lzw: &mut LZW_Compress, in_file: *mut FILE, out_file: *mut FILE) -> c_int {
    let mut inbits: c_long = 0;
    let mut code: code_int;
    let mut incode: code_int;
    let mut stackp: usize;

    lzw.ofd = out_file;
    lzw.maxbits = c_int::from(lzw.get_byte());
    lzw.block_mode = lzw.maxbits & BLOCK_MODE;

    if lzw.maxbits & LZW_RESERVED != 0 {
        lzw.error("warning, unknown flags in unlzw decompression");
    }

    lzw.maxbits &= BIT_MASK;

    let maxmaxcode: code_int = MAXCODE!(lzw.maxbits);

    if lzw.maxbits > BITS {
        lzw.error("compressed with too many bits; cannot handle file");
        lzw.exit_code = ERROR;
        return ERROR;
    }

    let mut rsize: c_int = lzw.insize as c_int;
    let mut n_bits: c_int = INIT_BITS;
    let mut maxcode: code_int = MAXCODE!(n_bits) - 1;
    let mut bitmask: c_uint = (1 << n_bits) - 1;
    let mut oldcode: code_int = -1;
    let mut finchar: c_int = 0;
    let mut outpos: c_int = 0;
    let mut posbits: c_long = (lzw.inptr as c_long) << 3;

    let mut free_ent: code_int = if lzw.block_mode != 0 {
        code_int::from(FIRST)
    } else {
        256
    };

    /* Initialize the first 256 entries in the table. */
    lzw.tab_prefix[..256].fill(0);

    for code in (0..256).rev() {
        lzw.window[code] = code as c_uchar;
    }

    let mut resetbuf = false;

    let mut i: c_int = 0;
    let mut e: c_int = 0;
    let mut o: c_int = 0;

    'outer_loop: loop {
        if !resetbuf {
            i = 0;
            e = 0;
            o = 0;
        }

        o = (posbits >> 3) as c_int;
        e = (lzw.insize - o as usize) as c_int;

        for i in 0..(e) {
            lzw.inbuf[i as usize] = lzw.inbuf[(o + i) as usize];
        }

        lzw.insize = e as usize;
        posbits = 0;

        if lzw.insize < INBUF_EXTRA {
            let mut cfile = CFile::from(in_file);

            let last_rsize = cfile.read(&mut lzw.inbuf[lzw.insize..(lzw.insize + INBUFSIZ)]);

            match last_rsize {
                Err(_) => {
                    lzw.error("unexpected end of file");
                    lzw.exit_code = ERROR;
                    return ERROR;
                }
                Ok(len) => {
                    rsize = len as c_int;
                    lzw.insize += rsize as usize;
                    lzw.bytes_in += rsize as usize;
                }
            }
        }

        inbits = if rsize != 0 {
            ((lzw.insize as c_long) - (lzw.insize as c_long % c_long::from(n_bits))) << 3
        } else {
            ((lzw.insize as c_long) << 3) - (c_long::from(n_bits) - 1)
        };

        while inbits > posbits {
            if free_ent > maxcode {
                posbits = (posbits - 1)
                    + ((c_long::from(n_bits) << 3)
                        - (posbits - 1 + c_long::from(n_bits << 3)) % c_long::from(n_bits << 3));
                n_bits += 1;
                if n_bits == lzw.maxbits {
                    maxcode = maxmaxcode;
                } else {
                    maxcode = MAXCODE!(n_bits) - 1;
                }
                bitmask = (1 << n_bits) - 1;

                resetbuf = true;
                continue 'outer_loop;
            }

            // 'input' macro
            let p = &(lzw.inbuf)[((posbits >> 3) as usize)..];
            (code) =
                ((c_long::from(p[0]) | (c_long::from(p[1]) << 8) | (c_long::from(p[2]) << 16))
                    >> ((posbits) & 0x7))
                    & (bitmask as c_long);
            (posbits) += c_long::from(n_bits);

            if oldcode == -1 {
                if code >= 256 {
                    lzw.error("corrupt input.");
                    lzw.exit_code = ERROR;
                    return ERROR;
                }

                oldcode = code;
                finchar = code as c_int;
                lzw.outbuf[outpos as usize] = finchar as c_uchar;
                outpos += 1;

                continue;
            }

            if code == c_long::from(CLEAR) && lzw.block_mode != 0 {
                lzw.tab_prefix[..256].fill(0);
                free_ent = c_long::from(FIRST - 1);
                posbits = (posbits - 1)
                    + ((c_long::from(n_bits) << 3)
                        - (posbits - 1 + c_long::from(n_bits << 3)) % c_long::from(n_bits << 3));
                n_bits = INIT_BITS;
                maxcode = MAXCODE!(n_bits) - 1;
                bitmask = (1 << n_bits) - 1;

                resetbuf = true;
                continue 'outer_loop;
            }

            incode = code;

            let stack: &mut [char_type] = cast_slice_mut(&mut lzw.d_buf);
            stackp = (DIST_BUFSIZE - 1) * core::mem::size_of::<c_ushort>()
                / core::mem::size_of::<char_type>();

            if code >= free_ent {
                /* Special case for KwKwK string. */
                if code > free_ent {
                    if outpos > 0 {
                        lzw.write_buf(outpos as usize);
                        lzw.bytes_out += outpos as usize;
                    }
                    lzw.error("corrupt input.");
                    lzw.exit_code = ERROR;
                    return ERROR;
                }

                stackp -= 1;
                stack[stackp] = finchar as char_type;
                code = oldcode;
            }

            while (code as cmp_code_int) >= 256 {
                /* Generate output characters in reverse order */
                stackp -= 1;
                stack[stackp] = lzw.window[code as usize] as char_type;
                code = code_int::from(lzw.tab_prefix[code as usize]);
            }

            stackp -= 1;
            finchar = c_int::from(lzw.window[code as usize]);
            stack[stackp] = finchar as char_type;

            /* And put them out in forward order */
            {
                /*	REG1 int	i;   already defined above (WDP) */

                i = ((DIST_BUFSIZE - 1) * core::mem::size_of::<c_ushort>()
                    / core::mem::size_of::<char_type>()
                    - stackp) as c_int;

                if outpos + (i) >= OUTBUFSIZ as c_int {
                    loop {
                        if i > OUTBUFSIZ as c_int - outpos {
                            i = OUTBUFSIZ as c_int - outpos;
                        }

                        if i > 0 {
                            let s: &mut [c_char] = cast_slice_mut(&mut lzw.d_buf);

                            (&mut lzw.outbuf)[(outpos as usize)..((outpos + i) as usize)]
                                .copy_from_slice(cast_slice(&s[stackp..(stackp + i as usize)]));
                            outpos += i;
                        }

                        if outpos >= OUTBUFSIZ as c_int {
                            lzw.write_buf(outpos as usize);
                            lzw.bytes_out += outpos as usize;
                            outpos = 0;
                        }
                        stackp += i as usize;

                        i = ((DIST_BUFSIZE - 1) * core::mem::size_of::<c_ushort>()
                            / core::mem::size_of::<char_type>()
                            - stackp) as c_int;

                        if i <= 0 {
                            break;
                        }
                    }
                } else {
                    (&mut lzw.outbuf)[(outpos as usize)..((outpos + i) as usize)]
                        .copy_from_slice(cast_slice(&stack[stackp..(stackp + i as usize)]));
                    outpos += i;
                }
            }

            code = free_ent;
            if code < maxmaxcode {
                /* Generate the new entry. */

                lzw.tab_prefix[code as usize] = oldcode as c_ushort;
                lzw.window[code as usize] = finchar as char_type;
                free_ent = code + 1;
            }
            oldcode = incode; /* Remember previous code.	*/
        }

        if rsize == 0 {
            break;
        }
    }

    if outpos > 0 {
        lzw.write_buf(outpos as usize);
        lzw.bytes_out += outpos as usize;
    }

    OK
}

#[cfg(test)]
mod tests {

    use alloc::ffi::CString;
    use std::{fs::File, io::Read};

    use crate::c_types::{c_char, c_int};
    use bytemuck::cast_slice;
    use libc::fdopen;

    use crate::helpers::outbuf::OutBuf;

    use crate::zuncompress::zuncompress2mem;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_zuncompress2mem() {
        // Paths to the compressed and expected decompressed files
        let compressed_file_path = "test_resources/sample.txt.Z";
        let expected_file_path = "test_resources/sample.txt";

        // Open the compressed file
        let compressed_file =
            File::open(compressed_file_path).expect("Failed to open compressed file");
        let compressed_file_ptr = unsafe {
            #[cfg(windows)]
            {
                use libc::open_osfhandle;
                use std::os::windows::io::AsRawHandle;

                let handle = compressed_file.as_raw_handle();
                let fd = open_osfhandle(handle as isize, 0);
                fdopen(fd, c"rb".as_ptr() as *const c_char)
            }

            #[cfg(unix)]
            {
                use std::os::fd::AsRawFd;

                let fd = compressed_file.as_raw_fd();

                fdopen(fd, c"rb".as_ptr().cast::<c_char>())
            }
        };

        // Start empty, so the output has to grow
        let mut decompressed_buffer: Vec<u8> = Vec::new();
        let mut decompressed_size: usize = 0;
        let mut status: c_int = 0;

        // Call the zuncompress2mem function
        let result = zuncompress2mem(
            cast_slice(
                CString::new(compressed_file_path)
                    .unwrap()
                    .as_bytes_with_nul(),
            ),
            compressed_file_ptr,
            OutBuf::Vec(&mut decompressed_buffer),
            &mut decompressed_size,
            &mut status,
        );

        // Ensure the decompression was successful
        assert_eq!(result, 0, "Decompression failed with status: {status}");

        // Read the expected decompressed content
        let mut expected_content = Vec::new();
        File::open(expected_file_path)
            .expect("Failed to open expected file")
            .read_to_end(&mut expected_content)
            .expect("Failed to read expected file");

        // Compare the decompressed content with the expected content
        assert_eq!(decompressed_buffer.len(), decompressed_size);
        assert_eq!(
            &decompressed_buffer[..decompressed_size],
            &expected_content,
            "Decompressed content does not match expected content"
        );

        // Clean up
        // unsafe {
        //libc::fclose(compressed_file_ptr);
        // }
    }
}
