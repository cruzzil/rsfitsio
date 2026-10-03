//! The driver for in-memory files, behind the `mem://` prefix.
//!
//! A memory file is a growable byte buffer that behaves like a file. It backs
//! `mem://` names, the decompression of a `.gz` or `.Z` file by
//! [`crate::drvrfile`], and the temporary file that
//! [`crate::cfileio`] creates when a filter is applied to an input file.
//!
//! Ported from CFITSIO's `drvrmem.c`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center.
#![warn(missing_docs)]

use core::ffi::CStr;
use core::slice;
use std::io::{Read, Seek, SeekFrom, Write as _, stdin};

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

#[cfg(unix)]
use std::os::fd::AsRawFd;

use core::{cmp, mem, ptr};
use std::sync::Mutex;

use crate::c_types::{FILE, c_char, c_int, c_long, c_uchar, c_uint, c_ushort, c_void};
use crate::helpers::cfile::CFile;
use crate::helpers::vec_raw_parts::vec_into_raw_parts;
use crate::zuncompress::zuncompress2mem;
use libc::{EOF, fclose, fgetc, fopen, fread, memcmp, memcpy, memset, realloc, ungetc};

use bytemuck::{cast_slice, cast_slice_mut};

use crate::cfileio::ffclos_safe;
use crate::cfileio::{MAX_PREFIX_LEN, ffimem_safe};
use crate::drvrfile::{file_close, file_create, file_open, file_openfile, file_write};
use crate::fitscore::{ALLOCATIONS, ffpmsg_slice, ffpmsg_str};
use crate::fitsio::*;
use crate::fitsio2::*;
use crate::iraffits::iraf2mem;
use crate::putkey::ffcrim_safe;
use crate::swapproc::{ffswap2, ffswap4, ffswap8};
use crate::wrappers::*;
use crate::zcompress::{compress2file_from_mem, uncompress2mem, uncompress2mem_from_mem};
use crate::{BL, STDIN, STDOUT};
use crate::{bb, cs};

/// Size of the record buffer used when reading a file from stdin.
pub const RECBUFLEN: usize = 1000;

static STDIN_OUTFILE: Mutex<[c_char; FLEN_FILENAME]> = Mutex::new([0; FLEN_FILENAME]);

/// Address/size pair for a memory file whose buffer this driver owns.
///
/// This lives in its own heap allocation rather than as two fields of
/// `memdriver`.  Storing it inline meant `memaddrptr` had to point at the
/// struct's own `memaddr` field, making `memdriver` self-referential: any
/// later access to that table slot invalidated the pointer, which miri
/// reports as a Stacked Borrows violation.  A separate allocation keeps the
/// pointer's provenance independent of the table.
#[derive(Debug, Copy, Clone)]
/// The address and size cell a memory file owns, when the caller did not
/// supply one of its own.
struct OwnedMem {
    /// Address of the allocation.
    addr: *mut c_char,
    /// Size of the allocation in bytes.
    size: usize,
}

/// One open memory file.
#[derive(Debug, Copy, Clone)]
struct memdriver {
    /// Pointer to the memory address pointer; points either into the caller's
    /// variables or into `owned_cell`.
    memaddrptr: *mut *mut c_char,
    /// Pointer to the size of the memory allocation; points either into the
    /// caller's variables or into `owned_cell`.
    memsizeptr: *mut usize,
    /// Non-null if and only if this slot allocated its own address and size
    /// cell; see `mem_createmem`.
    owned_cell: *mut OwnedMem,
    /// Suggested increment for reallocating memory.
    deltasize: usize,
    /// Reallocation function used to grow the buffer.
    mem_realloc: Option<unsafe extern "C" fn(p: *mut c_void, newsize: usize) -> *mut c_void>,
    /// Current file position, relative to the start.
    currentpos: LONGLONG,
    /// Size of the FITS file, always no greater than `*memsizeptr`.
    fitsfilesize: LONGLONG,
    /// Pointer to the compressed output disk file.
    fileptr: *mut FILE,
}

impl Default for memdriver {
    fn default() -> Self {
        Self {
            memaddrptr: ptr::null_mut(),
            memsizeptr: ptr::null_mut(),
            owned_cell: ptr::null_mut(),
            deltasize: 0,
            mem_realloc: None,
            currentpos: 0,
            fitsfilesize: 0,
            fileptr: ptr::null_mut(),
        }
    }
}

unsafe impl Send for memdriver {}

/* allocate mem file handle tables */
static MEM_TABLE: Mutex<[memdriver; NMAXFILES]> = Mutex::new(
    [memdriver {
        memaddrptr: ptr::null_mut(),
        memsizeptr: ptr::null_mut(),
        owned_cell: ptr::null_mut(),
        deltasize: 0,
        mem_realloc: None,
        currentpos: 0,
        fitsfilesize: 0,
        fileptr: ptr::null_mut(),
    }; NMAXFILES],
);

/// Mark a table slot as free, releasing the address/size cell if this slot
/// owned one.  A null `memaddrptr` is what marks the slot empty.
fn release_owned_cell(d: &mut memdriver) {
    if !d.owned_cell.is_null() {
        // SAFETY: owned_cell is non-null only when this slot allocated it in
        // mem_createmem, and it is cleared here so it cannot be freed twice.
        drop(unsafe { Box::from_raw(d.owned_cell) });
        d.owned_cell = ptr::null_mut();
    }
    d.memaddrptr = ptr::null_mut();
    d.memsizeptr = ptr::null_mut();
}

/// Resize a driver-owned buffer, returning the new address or null on failure.
///
/// mem_createmem allocates the buffer with Rust's allocator (a `Vec`), so it
/// has to be resized and freed through the same allocator; handing it to the C
/// realloc is an allocator mismatch. Buffers supplied by the caller through
/// mem_openmem keep using the caller's realloc function, which is correct
/// because the caller allocated them -- `owned_cell` is what distinguishes the
/// two.
///
/// Grows with zeroed bytes, matching what mem_truncate did by hand after
/// calling realloc.
unsafe fn owned_realloc(ptr: *mut c_char, newsize: usize) -> *mut c_char {
    unsafe {
        let mut allocations = ALLOCATIONS.lock().unwrap();

        let Some((len, capacity)) = allocations.remove(&(ptr as usize)) else {
            /* Not one of ours; refuse rather than free it with the wrong
            allocator. */
            return ptr::null_mut();
        };

        let mut v = Vec::from_raw_parts(ptr, len, capacity);

        if newsize > v.len() && v.try_reserve_exact(newsize - v.len()).is_err() {
            /* Put it back so the close path can still release it. */
            let (p, l, c) = vec_into_raw_parts(v);
            allocations.insert(p as usize, (l, c));
            return ptr::null_mut();
        }
        v.resize(newsize, 0);
        v.shrink_to(newsize);

        let (p, l, c) = vec_into_raw_parts(v);
        allocations.insert(p as usize, (l, c));
        p
    }
}

pub(crate) fn mem_init() -> c_int {
    0
}

pub(crate) fn mem_setoptions(_options: c_int) -> c_int {
    0
}
pub(crate) fn mem_getoptions(options: &mut c_int) -> c_int {
    *options = 0;
    0
}
pub(crate) fn mem_getversion(version: &mut c_int) -> c_int {
    *version = 10;
    0
}
pub(crate) fn mem_shutdown() -> c_int {
    0
}

/// Create a new empty memory file for subsequent writes.
/// The file name is ignored in this case.
pub(crate) fn mem_create(_filename: &mut [c_char; FLEN_FILENAME], handle: &mut c_int) -> c_int {
    /* initially allocate 1 FITS block = 2880 bytes */
    let status = mem_createmem(BL!(), handle);

    if status != 0 {
        ffpmsg_str("failed to create empty memory file (mem_create)");
        return status;
    }

    0
}

/// Create a new empty memory file for subsequent writes.
/// Also create an empty compressed .gz file.  The memory file
/// will be compressed and written to the disk file when the file is closed.
pub(crate) fn mem_create_comp_unsafe(
    filename: &mut [c_char; FLEN_FILENAME],
    handle: &mut c_int,
) -> c_int {
    unsafe {
        let mut diskfile = ptr::null_mut();
        let mut mode: [c_char; 4] = [0; 4];

        /* first, create disk file for the compressed output */

        if strcmp_safe(filename, cs!(c"-.gz")) == 0
            || strcmp_safe(filename, cs!(c"stdout.gz")) == 0
            || strcmp_safe(filename, cs!(c"STDOUT.gz")) == 0
        {
            /* special case: create uncompressed FITS file in memory, then
            compress it an write it out to 'stdout' when it is closed.  */

            diskfile = STDOUT!();
        } else {
            /* normal case: create disk file for the compressed output */

            strcpy_safe(&mut mode, cs!(c"w+b")); /* create file with read-write */

            let filename_str = CStr::from_bytes_until_nul(cast_slice(filename))
                .unwrap()
                .to_str()
                .unwrap();

            /* does file already exist? */
            if let Ok(true) = std::fs::exists(filename_str) {
                return FILE_NOT_CREATED;
            }

            diskfile = fopen(filename.as_ptr(), mode.as_ptr());

            if diskfile.is_null() {
                /* couldn't create file */
                return FILE_NOT_CREATED;
            }
        }

        /* now create temporary memory file */

        /* initially allocate 1 FITS block = 2880 bytes */
        let status = mem_createmem(BL!(), handle);

        if status != 0 {
            // Close the file descriptor if it isn't stdout
            if diskfile != STDOUT!() {
                fclose(diskfile);
            }
            ffpmsg_str("failed to create empty memory file (mem_create_comp)");
            return status;
        }

        let mut m = MEM_TABLE.lock().unwrap();
        m[*handle as usize].fileptr = diskfile;

        0
    }
}

/// lowest level routine to open a pre-existing memory file.
///
/// # Parameters
///
/// * `buffptr`    — (I) address of memory pointer
/// * `buffsize`   — (I) size of buffer, in bytes
/// * `deltasize`  — (I) increment for future realloc's
/// * `memrealloc` — function (may be NULL)
pub(crate) fn mem_openmem(
    buffptr: *mut *mut c_void,
    buffsize: &mut usize,
    deltasize: usize,
    memrealloc: Option<unsafe extern "C" fn(p: *mut c_void, newsize: usize) -> *mut c_void>,
    handle: &mut c_int,
) -> c_int {
    *handle = -1;

    let mut m = MEM_TABLE.lock().unwrap();
    //let m = &mut memTable;
    let mut ii = 0;
    while ii < NMAXFILES {
        /* find empty slot in handle table */

        if m[ii].memaddrptr.is_null() {
            *handle = ii as c_int;
            break;
        }

        ii += 1;
    }

    if *handle == -1 {
        return TOO_MANY_FILES; /* too many files opened */
    }

    m[ii].memaddrptr = buffptr.cast::<*mut c_char>(); /* pointer to start addres */
    m[ii].memsizeptr = buffsize; /* allocated size of memory */
    m[ii].deltasize = deltasize; /* suggested realloc increment */
    m[ii].fitsfilesize = *buffsize as LONGLONG; /* size of FITS file (upper limit) */
    m[ii].currentpos = 0; /* at beginning of the file */
    m[ii].mem_realloc = memrealloc; /* memory realloc function (may be NULL) */
    0
}

///  lowest level routine to allocate a memory file.
pub(crate) fn mem_createmem(msize: usize, handle: &mut c_int) -> c_int {
    *handle = -1;

    let mut m = MEM_TABLE.lock().unwrap();
    //let m = &mut memTable;
    let mut ii = 0;
    while ii < NMAXFILES {
        /* find empty slot in handle table */
        if m[ii].memaddrptr.is_null() {
            *handle = ii as c_int;
            break;
        }

        ii += 1;
    }

    if *handle == -1 {
        return TOO_MANY_FILES; // Too many files opened
    }

    /* Use an internally allocated address/size cell.  The C points memaddrptr
    at the struct's own memaddr field; doing that here would make the table
    slot self-referential, so the cell gets its own allocation and the
    pointers are derived from it instead. */
    let cell = Box::into_raw(Box::new(OwnedMem {
        addr: ptr::null_mut(),
        size: 0,
    }));
    m[ii].owned_cell = cell;
    // SAFETY: `cell` was just allocated and nothing else refers to it. Taking
    // the field pointers with `&raw mut` keeps them derived from `cell`
    // itself, so writes through them stay valid for as long as the cell lives.
    unsafe {
        m[ii].memaddrptr = &raw mut (*cell).addr;
        m[ii].memsizeptr = &raw mut (*cell).size;
    }

    /* allocate initial block of memory for the file */
    if msize > 0 {
        // HEAP ALLOCATION
        let mut v = Vec::new();
        if v.try_reserve_exact(msize).is_err() {
            ffpmsg_str("malloc of initial memory failed (mem_createmem)");
            release_owned_cell(&mut m[ii]);
            return FILE_NOT_OPENED;
        } else {
            v.resize(msize, 0);
            let (p, l, c) = vec_into_raw_parts(v);
            ALLOCATIONS.lock().unwrap().insert(p as usize, (l, c));
            unsafe { *m[ii].memaddrptr = p };
        }
    }

    /* set initial state of the file */
    unsafe { *m[ii].memsizeptr = msize };
    m[ii].deltasize = BL!();
    m[ii].fitsfilesize = 0;
    m[ii].currentpos = 0;
    m[ii].mem_realloc = Some(realloc);
    0
}

/// truncate the file to a new size
pub(crate) fn mem_truncate_unsafe(handle: c_int, filesize: usize) -> c_int {
    unsafe {
        let handle = handle as usize;

        let mut m = MEM_TABLE.lock().unwrap();
        //let m = &mut memTable;

        /* call the memory reallocation function, if defined */
        if !m[handle].owned_cell.is_null() || m[handle].mem_realloc.is_some() {
            let oldsize = *(m[handle].memsizeptr);
            let ptr = if m[handle].owned_cell.is_null() {
                /* caller's buffer, so the caller's realloc */
                let p = (m[handle].mem_realloc.unwrap())(
                    (*(m[handle].memaddrptr)).cast::<c_void>(),
                    filesize,
                )
                .cast::<c_char>();
                /* if allocated more memory, initialize it to zero */
                if !p.is_null() && filesize > oldsize {
                    memset(p.add(oldsize).cast::<c_void>(), 0, filesize - oldsize);
                }
                p
            } else {
                /* our own Vec: owned_realloc already zeroes any growth */
                owned_realloc(*(m[handle].memaddrptr), filesize)
            };

            if ptr.is_null() {
                ffpmsg_str("Failed to reallocate memory (mem_truncate)");
                return MEMORY_ALLOCATION;
            }

            *(m[handle].memaddrptr) = ptr;
            *(m[handle].memsizeptr) = filesize;

            m[handle].currentpos = filesize as LONGLONG;
            m[handle].fitsfilesize = filesize as LONGLONG;
        }

        0
    }
}

/// do any special case checking when opening a file on the stdin stream
pub(crate) fn stdin_checkfile(
    urltype: &mut [c_char; MAX_PREFIX_LEN],
    _infile: &mut [c_char; FLEN_FILENAME],
    outfile: &mut [c_char; FLEN_FILENAME],
) -> c_int {
    let mut s = STDIN_OUTFILE.lock().unwrap();
    //let s = &mut stdin_outfile;

    if strlen_safe(outfile) > 0 {
        s[0] = 0;
        strncat_safe(&mut s[..], outfile, FLEN_FILENAME - 1); /* an output file is specified */
        strcpy_safe(urltype, cs!(c"stdinfile://"));
    } else {
        s[0] = 0; /* no output file was specified */
    }

    0
}

/// open a FITS file from the stdin file stream by copying it into memory
/// The file name is ignored in this case.
pub(crate) fn stdin_open(filename: &mut [c_char], rwmode: c_int, handle: &mut c_int) -> c_int {
    let mut status: c_int = 0;

    let mut s = STDIN_OUTFILE.lock().unwrap();
    //let s = &mut stdin_outfile;

    if s[0] != 0 {
        /* copy the stdin stream to the specified disk file then open the file */

        /* Create the output file */
        status = file_create(&mut s, handle);

        if status != 0 {
            ffpmsg_str("Unable to create output file to copy stdin (stdin_open):");
            ffpmsg_slice(&s.clone());
            return status;
        }

        /* copy the whole stdin stream to the file */
        status = stdin2file(*handle);
        file_close(*handle);

        if status != 0 {
            ffpmsg_str("failed to copy stdin to file (stdin_open)");
            ffpmsg_slice(&s.clone());
            return status;
        }

        /* reopen file with proper rwmode attribute */
        status = file_open(&mut *s, rwmode, handle);
    } else {
        /* get the first character, then put it back */

        let mut stdin_hdl = stdin();
        let mut cbuff = [0_u8];
        stdin_hdl.read_exact(&mut cbuff).unwrap();

        // AFAIK this can't be reproduced in safe rust.
        unsafe {
            ungetc(c_int::from(cbuff[0]), STDIN!());

            /* compressed files begin with 037 or 'P' */
            if cbuff[0] == 31 || cbuff[0] == 75 {
                /* looks like the input stream is compressed */
                status = mem_compress_stdin_open(filename, rwmode, handle);
            } else {
                /* copy the stdin stream into memory then open file in memory */

                if rwmode != READONLY {
                    ffpmsg_str("cannot open stdin with WRITE access");
                    return READONLY_FILE;
                }

                status = mem_createmem(BL!(), handle);

                if status != 0 {
                    ffpmsg_str("failed to create empty memory file (stdin_open)");
                    return status;
                }

                /* copy the whole stdin stream into memory */
                status = stdin2mem(*handle);

                if status != 0 {
                    ffpmsg_str("failed to copy stdin into memory (stdin_open)");
                    // HEAP DEALLOCATION
                    let m = MEM_TABLE.lock().unwrap();
                    let ms = *m[*handle as usize].memsizeptr;
                    _ = Vec::from_raw_parts(*m[*handle as usize].memaddrptr, ms, ms);
                }
            }
        }
    }

    status
}

/// Copy the stdin stream into memory.  Fill whatever amount of memory
/// has already been allocated, then realloc more memory if necessary.
pub(crate) unsafe fn stdin2mem(hd: c_int) -> c_int {
    unsafe {
        /* handle number */

        let simple = b"SIMPLE";

        let mut m = MEM_TABLE.lock().unwrap();

        let mut memptr = *m[hd as usize].memaddrptr;
        let mut memsize = *m[hd as usize].memsizeptr;
        let delta = m[hd as usize].deltasize;

        let mut filesize = 0;
        let mut ii = 0;
        let mut jj = 0;

        let mut c = fgetc(STDIN!());
        while c != EOF && jj < 2000 {
            /* Skip over any garbage at the beginning of the stdin stream by */
            /* reading 1 char at a time, looking for 'S', 'I', 'M', 'P', 'L', 'E' */
            /* Give up if not found in the first 2000 characters */

            if c == c_int::from(simple[ii]) {
                ii += 1;
                if ii == 6
                /* found the complete string? */
                {
                    memcpy(memptr.cast::<c_void>(), simple.as_ptr().cast(), 6); /* copy "SIMPLE" to buffer */
                    filesize = 6;
                    break;
                }
            } else {
                ii = 0; /* reset search to beginning of the string */
            }
            jj += 1;
            c = fgetc(STDIN!());
        }

        if filesize == 0 {
            ffpmsg_str("Couldn't find the string 'SIMPLE' in the stdin stream.");
            ffpmsg_str("This does not look like a FITS file.");
            return FILE_NOT_OPENED;
        }

        /* fill up the remainder of the initial memory allocation */
        let mut nread = fread(memptr.add(6).cast::<c_void>(), 1, memsize - 6, STDIN!());
        nread += 6; /* add in the 6 characters in 'SIMPLE' */

        if nread < memsize
        /* reached the end? */
        {
            m[hd as usize].fitsfilesize = nread as LONGLONG;
            return 0;
        }

        filesize = nread;

        loop {
            /* allocate memory for another FITS block */
            memptr = realloc(memptr.cast::<c_void>(), memsize + delta).cast::<c_char>();

            if memptr.is_null() {
                ffpmsg_str("realloc failed while copying stdin (stdin2mem)");
                return MEMORY_ALLOCATION;
            }
            memsize += delta;

            /* read another FITS block */
            nread = fread(memptr.add(filesize).cast::<c_void>(), 1, delta, STDIN!());

            filesize += nread;

            if nread < delta {
                /* reached the end? */
                break;
            }
        }

        m[hd as usize].fitsfilesize = filesize as LONGLONG;
        *m[hd as usize].memaddrptr = memptr;
        *m[hd as usize].memsizeptr = memsize;

        0
    }
}

/// Copy the stdin stream to a file.
pub(crate) fn stdin2file(handle: c_int) -> c_int {
    /* handle number */

    let simple = b"SIMPLE";
    let mut recbuf: [c_char; RECBUFLEN] = [0; RECBUFLEN];

    let mut ii = 0;
    let mut jj = 0;

    let mut stdin_hdl = stdin().lock();
    let mut c = [0_u8];
    stdin_hdl.read_exact(&mut c).unwrap();

    while c_int::from(c[0]) != EOF && jj < 2000 {
        /* Skip over any garbage at the beginning of the stdin stream by */
        /* reading 1 char at a time, looking for 'S', 'I', 'M', 'P', 'L', 'E' */
        /* Give up if not found in the first 2000 characters */

        if c_int::from(c[0]) == c_int::from(simple[ii]) {
            ii += 1;
            if ii == 6 {
                /* found the complete string? */

                (recbuf[..6]).copy_from_slice(cast_slice(&simple[0..6])); /* copy "SIMPLE" to buffer */
                break;
            }
        } else {
            ii = 0; /* reset search to beginning of the string */
        }
        jj += 1;
        stdin_hdl.read_exact(&mut c).unwrap();
    }

    if ii != 6 {
        ffpmsg_str("Couldn't find the string 'SIMPLE' in the stdin stream");
        return FILE_NOT_OPENED;
    }

    /* fill up the remainder of the buffer.  stdin is typically a pipe, where
    short reads are the norm rather than the exception, so loop until the
    buffer is full or the stream ends. */
    let mut nread = read_fill(&mut stdin_hdl, cast_slice_mut(&mut recbuf[6..RECBUFLEN])).unwrap();

    nread += 6; /* add in the 6 characters in 'SIMPLE' */

    let mut status = file_write(handle, cast_slice(&recbuf), nread);
    if status != 0 {
        return status;
    }

    /* copy the rest of stdin stream */

    loop {
        let nread = stdin_hdl.read(cast_slice_mut(&mut recbuf[..RECBUFLEN]));

        if let Ok(nread) = nread {
            if nread == 0 {
                /* reached the end */
                break;
            }

            status = file_write(handle, cast_slice(&recbuf), nread);
            if status != 0 {
                return status;
            }
        }
    }

    status
}

/// copy the memory file to stdout, then free the memory
pub(crate) fn stdout_close_unsafe(handle: c_int) -> c_int {
    unsafe {
        let mut status: c_int = 0;

        let mut m = MEM_TABLE.lock().unwrap();

        /* Copy from memory to standard out.
        DEVIATION: the C fwrite()s to the CRT's stdout, which on windows is in
        text mode and so turns every 0x0A of the FITS stream into 0x0D 0x0A.
        Rust's stdout does no translation, and is the same handle fp_msg and
        the stream driver write through. */
        let n = m[handle as usize].fitsfilesize as usize;
        // SAFETY: memaddr owns `memsize' bytes, of which `fitsfilesize' hold
        // the file; both are set when the memory file is created.
        let bytes = slice::from_raw_parts(*m[handle as usize].memaddrptr, n);
        let mut out = std::io::stdout();
        if out.write_all(cast_slice(bytes)).is_err() || out.flush().is_err() {
            ffpmsg_str("failed to copy memory file to stdout (stdout_close)");
            status = WRITE_ERROR;
        }

        // HEAP DEALLOCATION
        let ms = *m[handle as usize].memsizeptr;
        _ = Vec::from_raw_parts(*m[handle as usize].memaddrptr, ms, ms); /* free the memory */
        release_owned_cell(&mut m[handle as usize]);

        status
    }
}

/// This routine opens the compressed diskfile and creates an empty memory
/// buffer with an appropriate size, then calls mem_uncompress2mem. It allows
/// the memory 'file' to be opened with READWRITE access.
pub(crate) fn mem_compress_openrw(
    filename: &mut [c_char],
    _rwmode: c_int,
    hdl: &mut c_int,
) -> c_int {
    mem_compress_open(filename, READONLY, hdl)
}

/// This routine opens the compressed diskfile and creates an empty memory
/// buffer with an appropriate size, then calls mem_uncompress2mem.
pub(crate) fn mem_compress_open(filename: &mut [c_char], rwmode: c_int, hdl: &mut c_int) -> c_int {
    unsafe {
        let mut status: c_int = 0;
        let mut estimated: c_int = 1;
        let mut buffer: [c_uchar; 4] = [0; 4];
        let mut finalsize: usize = 0;
        let mut filesize: usize = 0;
        let mut llsize: LONGLONG = 0;
        let mut modulosize: c_uint = 0;

        let mut diskfile = None;

        if rwmode != READONLY {
            ffpmsg_str("cannot open compressed file with WRITE access (mem_compress_open)");
            ffpmsg_slice(filename);
            return READONLY_FILE;
        }

        /* open the compressed disk file */
        status = file_openfile(filename, READONLY, &mut diskfile);
        if status != 0 {
            ffpmsg_str("failed to open compressed disk file (compress_open)");
            ffpmsg_slice(filename);
            return status;
        }

        let mut diskfile = diskfile.unwrap();

        if !matches!(read_fill(&mut diskfile, &mut buffer[..2]), Ok(2)) {
            /* read 2 bytes */
            return READ_ERROR;
        }

        if buffer[..2] == [0o37, 0o213] {
            /* GZIP */

            /* the uncompressed file size is give at the end */
            /* of the file in the ISIZE field  (modulo 2^32) */

            /* move to end of file; position = size of file */
            filesize = match diskfile.seek(SeekFrom::End(0)) {
                Ok(pos) => pos as usize,
                Err(_) => return SEEK_ERROR,
            };
            if filesize < 4 || diskfile.seek(SeekFrom::Current(-4)).is_err() {
                /* move back 4 bytes */
                ffpmsg_str("compressed file is too short to be a gzip file (mem_compress_open)");
                return READ_ERROR;
            }

            match diskfile.read_exact(&mut buffer[..4]) /* read 4 bytes */ {
                Ok(_) => (),
                Err(_) => return READ_ERROR,
            }

            /* have to worry about integer byte order */
            modulosize = c_uint::from(buffer[0]);
            modulosize |= c_uint::from(buffer[1]) << 8;
            modulosize |= c_uint::from(buffer[2]) << 16;
            modulosize |= c_uint::from(buffer[3]) << 24;

            /*
              the field ISIZE in the gzipped file header only stores 4 bytes and contains
              the uncompressed file size modulo 2^32.  If the uncompressed file size
              is less than the compressed file size (filesize), then one probably needs to
              add 2^32 = 4294967296 to the uncompressed file size, assuming that the gzip
              produces a compressed file that is smaller than the original file.

              But one must allow for the case of very small files, where the
              gzipped file may actually be larger then the original uncompressed file.
              Therefore, only perform the modulo 2^32 correction test if the compressed
              file is greater than 10,000 bytes in size.  (Note: this threhold would
              fail only if the original file was greater than 2^32 bytes in size AND gzip
              was able to compress it by more than a factor of 400,000 (!) which seems
              highly unlikely.)

              Also, obviously, this 2^32 modulo correction cannot be performed if the
              finalsize variable is only 32-bits long.  Typically, the 'size_t' integer
              type must be 8 bytes or larger in size to support data files that are
              greater than 2 GB (2^31 bytes) in size.
            */
            finalsize = modulosize as usize;

            if mem::size_of::<usize>() > 4 && filesize > 10000 {
                llsize = finalsize as LONGLONG;
                /* use LONGLONG variable to suppress compiler warning */
                while llsize < filesize as LONGLONG {
                    llsize += 4294967296;
                }

                finalsize = llsize as usize;
            }

            estimated = 0; /* file size is known, not estimated */
        } else if buffer[..2] == [0o120, 0o113] {
            /* PKZIP */

            /* the uncompressed file size is give at byte 22 the file */
            if diskfile.seek(SeekFrom::Start(22)).is_err() {
                /* move to byte 22 */
                return SEEK_ERROR;
            }
            match diskfile.read_exact(&mut buffer[..4]) /* read 4 bytes */ {
                Ok(_) => (),
                Err(_) => return READ_ERROR,
            }

            /* have to worry about integer byte order */
            modulosize = c_uint::from(buffer[0]);
            modulosize |= c_uint::from(buffer[1]) << 8;
            modulosize |= c_uint::from(buffer[2]) << 16;
            modulosize |= c_uint::from(buffer[3]) << 24;
            finalsize = modulosize as usize;

            estimated = 0; /* file size is known, not estimated */
        } else if buffer[..2] == [0o37, 0o36] {
            /* PACK */
            finalsize = 0; /* for most methods we can't determine final size */
        } else if buffer[..2] == [0o37, 0o235] {
            /* LZW */
            finalsize = 0; /* for most methods we can't determine final size */
        } else if buffer[..2] == [0o37, 0o240] {
            /* LZH */
            finalsize = 0; /* for most methods we can't determine final size */
        } else if memcmp(
            buffer.as_ptr().cast::<c_void>(),
            b"BZ".as_ptr().cast::<c_void>(),
            2,
        ) == 0
        {
            /* BZip2 */
            finalsize = 0; /* for most methods we can't determine final size */
        } else {
            /* not a compressed file; this should never happen */
            return 1;
        }

        if finalsize == 0 {
            /* estimate uncompressed file size */
            /* move to end of the compressed file; position = size of file */
            finalsize = match diskfile.seek(SeekFrom::End(0)) {
                Ok(pos) => pos as usize,
                Err(_) => return SEEK_ERROR,
            };
            finalsize *= 3; /* assume factor of 3 compression */
        }

        let _ = diskfile.rewind(); /* move back to beginning of file */

        /* create a memory file big enough (hopefully) for the uncompressed file */
        status = mem_createmem(finalsize, hdl);

        if status != 0 && estimated != 0 {
            /* memory allocation failed, so try a smaller estimated size */
            finalsize /= 3;
            status = mem_createmem(finalsize, hdl);
        }

        if status != 0 {
            ffpmsg_str("failed to create empty memory file (compress_open)");
            return status;
        }

        /* uncompress file into memory */
        status = mem_uncompress2mem(filename, &mut diskfile, *hdl);

        if status != 0 {
            mem_close_free_unsafe(*hdl); /* free up the memory */
            ffpmsg_str("failed to uncompress file into memory (compress_open)");
            return status;
        }

        let mut m = MEM_TABLE.lock().unwrap();

        /* if we allocated too much memory initially, then free it */
        if *(m[*hdl as usize].memsizeptr) > ((m[*hdl as usize].fitsfilesize as usize) + 256) {
            let ptr = realloc(
                (*(m[*hdl as usize].memaddrptr)).cast(),
                m[*hdl as usize].fitsfilesize as usize,
            )
            .cast::<c_char>();
            if ptr.is_null() {
                ffpmsg_str("Failed to reduce size of allocated memory (compress_open)");
                return MEMORY_ALLOCATION;
            }

            *(m[*hdl as usize].memaddrptr) = ptr;
            *(m[*hdl as usize].memsizeptr) = (m[*hdl as usize].fitsfilesize) as usize;
        }

        0
    }
}

/// This routine reads the compressed input stream and creates an empty memory
/// buffer, then calls mem_uncompress2mem.
pub(crate) unsafe fn mem_compress_stdin_open(
    filename: &[c_char],
    rwmode: c_int,
    hdl: &mut c_int,
) -> c_int {
    unsafe {
        let mut status = 0;

        if rwmode != READONLY {
            ffpmsg_str(
                "cannot open compressed input stream with WRITE access (mem_compress_stdin_open)",
            );
            return READONLY_FILE;
        }

        /* create a memory file for the uncompressed file */
        status = mem_createmem(28800, hdl);

        if status != 0 {
            ffpmsg_str("failed to create empty memory file (compress_stdin_open)");
            return status;
        }

        /* uncompress file into memory */

        let mut stdin_safe = std::io::stdin();
        status = mem_uncompress2mem(filename, &mut stdin_safe, *hdl); //WARNING: Urggghhh

        if status != 0 {
            mem_close_free_unsafe(*hdl); /* free up the memory */
            ffpmsg_str("failed to uncompress stdin into memory (compress_stdin_open)");
            return status;
        }

        let mut m = MEM_TABLE.lock().unwrap();

        /* if we allocated too much memory initially, then free it */
        if *(m[*hdl as usize].memsizeptr) > ((m[*hdl as usize].fitsfilesize as usize) + 256) {
            let ptr = realloc(
                (*(m[*hdl as usize].memaddrptr)).cast::<c_void>(),
                m[*hdl as usize].fitsfilesize as usize,
            )
            .cast::<c_char>();
            if ptr.is_null() {
                ffpmsg_str("Failed to reduce size of allocated memory (compress_stdin_open)");
                return MEMORY_ALLOCATION;
            }

            *(m[*hdl as usize].memaddrptr) = ptr;
            *(m[*hdl as usize].memsizeptr) = (m[*hdl as usize].fitsfilesize) as usize;
        }

        0
    }
}

/// This routine creates an empty memory buffer, then calls iraf2mem to
/// open the IRAF disk file and convert it to a FITS file in memeory.
pub(crate) fn mem_iraf_open(filename: &mut [c_char], _rwmode: c_int, hdl: &mut c_int) -> c_int {
    unsafe {
        let mut filesize = 0;

        /* create a memory file with size = 0 for the FITS converted IRAF file */
        let mut status = mem_createmem(filesize, hdl);
        if status != 0 {
            ffpmsg_str("failed to create empty memory file (mem_iraf_open)");
            return status;
        }

        /* extract the raw pointers under the lock, then release it so that the
        IRAF conversion (and any error cleanup, e.g. mem_close_free) can safely
        re-lock MEM_TABLE without deadlocking */
        let (memaddrptr, memsizeptr) = {
            let m = MEM_TABLE.lock().unwrap();
            (
                m[*hdl as usize].memaddrptr,
                core::ptr::from_mut::<usize>(m[*hdl as usize].memsizeptr.as_mut().unwrap()),
            )
        };

        /* convert the iraf file into a FITS file in memory */
        status = iraf2mem(
            filename,
            memaddrptr,
            &mut *memsizeptr,
            &mut filesize,
            &mut status,
        );

        if status != 0 {
            mem_close_free_unsafe(*hdl); /* free up the memory */
            ffpmsg_str("failed to convert IRAF file into memory (mem_iraf_open)");
            return status;
        }

        let mut m = MEM_TABLE.lock().unwrap();
        m[*hdl as usize].currentpos = 0; /* save starting position */
        m[*hdl as usize].fitsfilesize = filesize as LONGLONG; /* and initial file size  */

        0
    }
}

/// This routine creates an empty memory buffer, writes a minimal
/// image header, then copies the image data from the raw file into
/// memory.  It will byteswap the pixel values if the raw array
/// is in little endian byte order.
pub(crate) fn mem_rawfile_open(filename: &mut [c_char], rwmode: c_int, hdl: &mut c_int) -> c_int {
    unsafe {
        let mut diskfile = None;
        let mut fptr: Option<Box<fitsfile>> = None;
        let mut status: c_int = 0;
        let mut endian: c_int = 0;
        let mut datatype: c_int = 0;
        let mut bytePerPix: c_int = 0;
        let mut naxis: c_int = 0;
        let mut dim: [c_long; 5] = [1; 5];
        let mut nvals: c_long = 0;
        let mut offset: c_long = 0;
        let mut filesize: usize = 0;
        let mut datasize: usize = 0;
        let mut rootfile: [c_char; FLEN_FILENAME] = [0; FLEN_FILENAME];

        if rwmode != READONLY {
            ffpmsg_str("cannot open raw binary file with WRITE access (mem_rawfile_open)");
            ffpmsg_slice(filename);
            return READONLY_FILE;
        }

        let cptr = strchr_safe(filename, bb(b'[')); /* search for opening bracket [ */

        if cptr.is_none() {
            ffpmsg_str("binary file name missing '[' character (mem_rawfile_open)");
            ffpmsg_slice(filename);
            return URL_PARSE_ERROR;
        }

        let mut cptr = cptr.unwrap();

        rootfile[0] = 0;
        strncat_safe(&mut rootfile, filename, cptr); /* store the rootname */

        cptr += 1;

        while filename[cptr] == bb(b' ') {
            cptr += 1; /* skip leading blanks */
        }

        /* Get the Data Type of the Image */

        if filename[cptr] == bb(b'b') || filename[cptr] == bb(b'B') {
            datatype = BYTE_IMG;
            bytePerPix = 1;
        } else if filename[cptr] == bb(b'i') || filename[cptr] == bb(b'I') {
            datatype = SHORT_IMG;
            bytePerPix = 2;
        } else if filename[cptr] == bb(b'u') || filename[cptr] == bb(b'U') {
            datatype = USHORT_IMG;
            bytePerPix = 2;
        } else if filename[cptr] == bb(b'j') || filename[cptr] == bb(b'J') {
            datatype = LONG_IMG;
            bytePerPix = 4;
        } else if filename[cptr] == bb(b'r')
            || filename[cptr] == bb(b'R')
            || filename[cptr] == bb(b'f')
            || filename[cptr] == bb(b'F')
        {
            datatype = FLOAT_IMG;
            bytePerPix = 4;
        } else if filename[cptr] == bb(b'd') || filename[cptr] == bb(b'D') {
            datatype = DOUBLE_IMG;
            bytePerPix = 8;
        } else {
            ffpmsg_str("error in raw binary file datatype (mem_rawfile_open)");
            ffpmsg_slice(filename);
            return URL_PARSE_ERROR;
        }

        cptr += 1;

        /* get Endian: Big or Little; default is same as the local machine */

        if filename[cptr] == bb(b'b') || filename[cptr] == bb(b'B') {
            endian = 0;
            cptr += 1;
        } else if filename[cptr] == bb(b'l') || filename[cptr] == bb(b'L') {
            endian = 1;
            cptr += 1;
        } else {
            endian = if BYTESWAPPED { 1 } else { 0 }; /* byteswapped machines are little endian */
        }

        /* read each dimension (up to 5) */

        naxis = 1;
        let mut cptr2 = 0;
        // dim[0] = strtol_safe(&filename[cptr..], &mut cptr2, 10);
        let (r, n) = strtol_safe(&filename[cptr..]).unwrap();
        dim[0] = r;
        cptr2 = n;

        if cptr2 != 0 && filename[cptr2] == bb(b',') {
            naxis = 2;
            // dim[1] = strtol_safe(&filename[(cptr2 + 1)..], &mut cptr, 10);
            let (r, n) = strtol_safe(&filename[(cptr2 + 1)..]).unwrap();
            dim[1] = r;
            cptr = n;
            if cptr != 0 && filename[cptr] == bb(b',') {
                naxis = 3;
                //dim[2] = strtol_safe(&filename[(cptr + 1)..], &mut cptr2, 10);
                let (r, n) = strtol_safe(&filename[(cptr + 1)..]).unwrap();
                dim[2] = r;
                cptr2 = n;

                if cptr2 != 0 && filename[cptr2] == bb(b',') {
                    naxis = 4;
                    // dim[3] = strtol_safe(&filename[(cptr2 + 1)..], &mut cptr, 10);
                    let (r, n) = strtol_safe(&filename[(cptr2 + 1)..]).unwrap();
                    dim[3] = r;
                    cptr = n;

                    if cptr != 0 && filename[cptr] == bb(b',') {
                        naxis = 5;
                        //dim[4] = strtol_safe(&filename[(cptr + 1)..], &mut cptr2, 10);
                        let (r, n) = strtol_safe(&filename[(cptr + 1)..]).unwrap();
                        dim[4] = r;
                        cptr2 = n;
                    }
                }
            }
        }

        cptr = cmp::max(cptr, cptr2);

        if filename[cptr] == bb(b':') {
            /* read starting offset value */

            // offset = strtol_safe(&filename[(cptr + 1)..], &mut dummy, 10);
            let (r, _n) = strtol_safe(&filename[(cptr + 1)..]).unwrap();
            offset = r;
        }

        nvals = dim[0] * dim[1] * dim[2] * dim[3] * dim[4];
        datasize = (nvals * c_long::from(bytePerPix)) as usize;
        filesize = (nvals * c_long::from(bytePerPix)) as usize + BL!();
        filesize = ((filesize - 1) / BL!() + 1) * BL!();

        /* open the raw binary disk file */
        status = file_openfile(&rootfile, READONLY, &mut diskfile);
        if status != 0 {
            ffpmsg_str("failed to open raw  binary file (mem_rawfile_open)");
            ffpmsg_slice(&rootfile);
            return status;
        }

        let mut diskfile = diskfile.unwrap();

        /* create a memory file with corrct size for the FITS converted raw file */
        status = mem_createmem(filesize, hdl);
        if status != 0 {
            ffpmsg_str("failed to create memory file (mem_rawfile_open)");
            return status;
        }

        let mut m = MEM_TABLE.lock().unwrap();

        /* open this piece of memory as a new FITS file */

        ffimem_safe(
            &mut fptr,
            m[*hdl as usize].memaddrptr.cast::<*mut c_void>(),
            &mut filesize,
            0,
            None,
            &mut status,
        );

        let mut fptr = fptr.unwrap();

        /* write the required header keywords */
        ffcrim_safe(&mut fptr, datatype, naxis, &dim, &mut status);

        /* close the FITS file, but keep the memory allocated */
        ffclos_safe(fptr, &mut status);

        if status > 0 {
            ffpmsg_str("failed to write basic image header (mem_rawfile_open)");
            mem_close_free_unsafe(*hdl); /* free up the memory */
            return status;
        }

        if offset > 0 {
            diskfile.seek(SeekFrom::Start(offset as u64)).unwrap(); /* offset to start of the data */
        }

        /* read the raw data into memory */
        let ptr = (*m[*hdl as usize].memaddrptr).add(BL!()).cast::<u8>();
        let ptr_slice = slice::from_raw_parts_mut(ptr, datasize);

        let tmp = read_fill(&mut diskfile, &mut ptr_slice[..datasize]).unwrap();
        if tmp != datasize {
            status = READ_ERROR;
        }

        if status != 0 {
            mem_close_free_unsafe(*hdl); /* free up the memory */
            ffpmsg_str("failed to copy raw file data into memory (mem_rawfile_open)");
            return status;
        }

        if datatype == USHORT_IMG {
            /* have to subtract 32768 from each unsigned */

            /* value to conform to FITS convention. More */
            /* efficient way to do this is to just flip  */
            /* the most significant bit.                 */

            let sptr: &mut [c_ushort] = cast_slice_mut(ptr_slice);

            if endian == if BYTESWAPPED { 1 } else { 0 } {
                /* working with native format */
                for ii in 0..(nvals as usize) {
                    sptr[ii] ^= 0x8000;
                }
            } else {
                /* pixels are byteswapped WRT the native format */

                for ii in 0..(nvals as usize) {
                    sptr[ii] ^= 0x80;
                }
            }
        }

        if endian != 0 {
            /* swap the bytes if array is in little endian byte order */

            if datatype == SHORT_IMG || datatype == USHORT_IMG {
                ffswap2(cast_slice_mut(ptr_slice), nvals);
            } else if datatype == LONG_IMG || datatype == FLOAT_IMG {
                ffswap4(cast_slice_mut(ptr_slice), nvals);
            } else if datatype == DOUBLE_IMG {
                ffswap8(cast_slice_mut(ptr_slice), nvals);
            }
        }

        m[*hdl as usize].currentpos = 0; /* save starting position */
        m[*hdl as usize].fitsfilesize = filesize as LONGLONG; /* and initial file size  */

        0
    }
}

/// lower level routine to uncompress a file into memory.  The file
/// has already been opened and the memory buffer has been allocated.
#[cfg(unix)]
pub(crate) unsafe fn mem_uncompress2mem<T: Read + AsRawFd>(
    filename: &[c_char],
    diskfile: &mut T,
    hdl: c_int,
) -> c_int {
    unsafe {
        let mut finalsize = 0;
        let mut status = 0;
        /* uncompress file into memory */

        let mut m = MEM_TABLE.lock().unwrap();

        if strstr_safe(filename, cs!(c".Z")).is_some() {
            let raw_file_handle = {
                use libc::fdopen;
                let fd = diskfile.as_raw_fd();

                fdopen(fd, c"rb".as_ptr().cast::<c_char>())
            };

            zuncompress2mem(
                filename,
                raw_file_handle,
                m[hdl as usize].memaddrptr.cast::<*mut u8>(), /* pointer to memory address */
                m[hdl as usize].memsizeptr.as_mut().unwrap(), /* pointer to size of memory */
                Some(realloc),                                /* reallocation function */
                &mut finalsize,
                &mut status, /* returned file size and status*/
            );
        } else if strstr_safe(filename, cs!(c".bz2")).is_some() {
            #[cfg(feature = "bzip2")]
            {
                let raw_file_handle = {
                    use libc::fdopen;
                    let fd = diskfile.as_raw_fd();

                    fdopen(fd, c"rb".as_ptr().cast::<c_char>())
                };

                /* bzip2uncompress2mem writes into the memory file through
                mem_write_unsafe, which locks MEM_TABLE itself. The std Mutex is
                not reentrant, so we must release our guard first to avoid a
                self-deadlock, then re-acquire it for the bookkeeping below. */
                drop(m);
                bzip2uncompress2mem(filename, raw_file_handle, hdl, &mut finalsize, &mut status);
                m = MEM_TABLE.lock().unwrap();
            }
        } else {
            uncompress2mem(
                filename,
                diskfile,
                m[hdl as usize].memaddrptr.cast::<*mut u8>(), /* pointer to memory address */
                m[hdl as usize].memsizeptr.as_mut().expect(NULL_MSG), /* pointer to size of memory */
                Some(realloc),                                        /* reallocation function */
                &mut finalsize,
                &mut status,
            ); /* returned file size nd status*/
        }

        m[hdl as usize].currentpos = 0; /* save starting position */
        m[hdl as usize].fitsfilesize = finalsize as LONGLONG; /* and initial file size  */
        status
    }
}

/// lower level routine to uncompress a file into memory.  The file
/// has already been opened and the memory buffer has been allocated.
#[cfg(windows)]
pub(crate) unsafe fn mem_uncompress2mem<T: Read + AsRawHandle>(
    filename: &[c_char],
    diskfile: &mut T,
    hdl: c_int,
) -> c_int {
    unsafe {
        let mut finalsize = 0;
        let mut status = 0;
        /* uncompress file into memory */

        let mut m = MEM_TABLE.lock().unwrap();

        if strstr_safe(filename, cs!(c".Z")).is_some() {
            let raw_file_handle = unsafe {
                use libc::{fdopen, open_osfhandle};

                let handle = diskfile.as_raw_handle();
                let fd = open_osfhandle(handle as isize, 0);
                fdopen(fd, c"rb".as_ptr() as *const c_char)
            };

            zuncompress2mem(
                filename,
                raw_file_handle,
                m[hdl as usize].memaddrptr as *mut *mut u8, /* pointer to memory address */
                m[hdl as usize].memsizeptr.as_mut().unwrap(), /* pointer to size of memory */
                Some(realloc),                              /* reallocation function */
                &mut finalsize,
                &mut status, /* returned file size and status*/
            );
        } else if strstr_safe(filename, cs!(c".bz2")).is_some() {
            #[cfg(feature = "bzip2")]
            {
                let raw_file_handle = unsafe {
                    use libc::{fdopen, open_osfhandle};

                    let handle = diskfile.as_raw_handle();
                    let fd = open_osfhandle(handle as isize, 0);
                    fdopen(fd, c"rb".as_ptr() as *const c_char)
                };

                bzip2uncompress2mem(filename, raw_file_handle, hdl, &mut finalsize, &mut status);
            }
        } else {
            uncompress2mem(
                filename,
                diskfile,
                m[hdl as usize].memaddrptr as *mut *mut u8, /* pointer to memory address */
                m[hdl as usize].memsizeptr.as_mut().expect(NULL_MSG), /* pointer to size of memory */
                Some(realloc),                                        /* reallocation function */
                &mut finalsize,
                &mut status,
            ); /* returned file size nd status*/
        }

        m[hdl as usize].currentpos = 0; /* save starting position */
        m[hdl as usize].fitsfilesize = finalsize as LONGLONG; /* and initial file size  */
        status
    }
}

/// return the size of the file; only called when the file is first opened
pub(crate) fn mem_size(handle: c_int, filesize: &mut usize) -> c_int {
    let handle = handle as usize;
    let m = MEM_TABLE.lock().unwrap();
    //let m = &memTable;

    *filesize = m[handle].fitsfilesize as usize;
    0
}

/// close the file and free the memory.
pub(crate) fn mem_close_free_unsafe(handle: c_int) -> c_int {
    unsafe {
        let handle = handle as usize;
        let mut m = MEM_TABLE.lock().unwrap();
        //let m = &mut memTable;

        let memsize = *m[handle].memsizeptr;

        // HEAP DEALLOCATION
        /* free( *(m[handle].memaddrptr) );  - C free() tolerates a NULL pointer, so
        guard against it here (the buffer may never have been allocated, e.g. when
        an IRAF conversion failed before writing the buffer pointer). */
        let dataptr = *(m[handle].memaddrptr);
        if !dataptr.is_null() {
            if let Some((l, c)) = ALLOCATIONS.lock().unwrap().remove(&(dataptr as usize)) {
                _ = Vec::from_raw_parts(dataptr, l, c);
            } else {
                _ = Vec::from_raw_parts(dataptr, memsize, memsize);
            }
        }
        // free( *(m[handle].memaddrptr) );

        release_owned_cell(&mut m[handle]);
        0
    }
}
///  close the memory file but do not free the memory.
pub(crate) fn mem_close_keep(handle: c_int) -> c_int {
    let handle = handle as usize;
    let mut m = MEM_TABLE.lock().unwrap();
    //let m = &mut memTable;

    release_owned_cell(&mut m[handle]);
    0
}

/// Compress the memory file, writing it out to the fileptr (which might be stdout)
pub(crate) fn mem_close_comp_unsafe(handle: c_int) -> c_int {
    unsafe {
        let mut status = 0;
        let mut compsize: usize = 0;

        /* compress file in  memory to a .gz disk file */

        let mut m = MEM_TABLE.lock().unwrap();
        let in_mem = slice::from_raw_parts(
            *m[handle as usize].memaddrptr,
            *m[handle as usize].memsizeptr,
        );

        if compress2file_from_mem(
            in_mem,
            m[handle as usize].fitsfilesize as usize,
            &mut CFile::from(m[handle as usize].fileptr),
            Some(&mut compsize),
            &mut status,
        ) != 0
        {
            ffpmsg_str("failed to copy memory file to file (mem_close_comp)");
            status = WRITE_ERROR;
        }

        // HEAP DEALLOCATION
        let ms = *m[handle as usize].memsizeptr;
        _ = Vec::from_raw_parts(*m[handle as usize].memaddrptr, ms, ms); /* free the memory */
        release_owned_cell(&mut m[handle as usize]);

        /* close the compressed disk file (except if it is 'stdout' */
        if !core::ptr::eq(m[handle as usize].fileptr, STDOUT!()) {
            fclose(m[handle as usize].fileptr);
        }

        status
    }
}

/// seek to position relative to start of the file.
pub(crate) fn mem_seek(handle: c_int, offset: LONGLONG) -> c_int {
    let handle = handle as usize;
    let mut m = MEM_TABLE.lock().unwrap();
    //let m = &mut memTable;

    if offset > m[handle].fitsfilesize {
        return END_OF_FILE;
    }

    m[handle].currentpos = offset;
    0
}
/// read bytes from the current position in the file
pub(crate) fn mem_read_unsafe(hdl: c_int, buffer: &mut [u8], nbytes: usize) -> c_int {
    unsafe {
        let hdl = hdl as usize;
        let nbytes = nbytes as LONGLONG;
        let mut m = MEM_TABLE.lock().unwrap();
        //let m = &mut memTable;

        if m[hdl].currentpos + nbytes > m[hdl].fitsfilesize {
            return END_OF_FILE;
        }

        let m_ptr = m[hdl].memaddrptr;
        let c_pos = m[hdl].currentpos;

        memcpy(
            buffer.as_mut_ptr().cast::<c_void>(),
            (*m_ptr).add(c_pos as usize).cast::<c_void>(),
            nbytes as usize,
        );

        m[hdl].currentpos += nbytes;

        0
    }
}

///  write bytes at the current position in the file
pub(crate) fn mem_write_unsafe(hdl: c_int, buffer: &[u8], nbytes: usize) -> c_int {
    unsafe {
        let hdl: usize = hdl.try_into().unwrap();
        let nbytes = nbytes as LONGLONG;
        let mut m = MEM_TABLE.lock().unwrap();

        if (m[hdl].currentpos + nbytes) as usize > *(m[hdl].memsizeptr) {
            if m[hdl].owned_cell.is_null() && m[hdl].mem_realloc.is_none() {
                ffpmsg_str("realloc function not defined (mem_write)");
                return WRITE_ERROR;
            }

            /*
             Attempt to reallocate additional memory:
             the memory buffer size is incremented by the larger of:
                1 FITS block (2880 bytes) or
                the defined 'deltasize' parameter
            */

            let newsize = cmp::max(
                ((((m[hdl].currentpos + nbytes - 1) / BL!()) + 1) * BL!()) as usize,
                *(m[hdl].memsizeptr) + m[hdl].deltasize,
            );

            /* call the realloc function; see owned_realloc for why a
            driver-owned buffer cannot go through the C one */
            let ptr = if m[hdl].owned_cell.is_null() {
                (m[hdl].mem_realloc.unwrap())((*(m[hdl].memaddrptr)).cast::<c_void>(), newsize)
                    .cast::<c_char>()
            } else {
                owned_realloc(*(m[hdl].memaddrptr), newsize)
            };
            if ptr.is_null() {
                ffpmsg_str("Failed to reallocate memory (mem_write)");
                return MEMORY_ALLOCATION;
            }

            *(m[hdl].memaddrptr) = ptr;
            *(m[hdl].memsizeptr) = newsize;
        }

        /* now copy the bytes from the buffer into memory */
        memcpy(
            (*(m[hdl].memaddrptr))
                .add(m[hdl].currentpos as usize)
                .cast::<c_void>(),
            buffer.as_ptr() as *mut c_void,
            nbytes as usize,
        );

        m[hdl].currentpos += nbytes;
        m[hdl].fitsfilesize = cmp::max(m[hdl].fitsfilesize, m[hdl].currentpos);
        0
    }
}

/// uncompress input buffer, length nbytes and write bytes to current
/// position in file.  output buffer needs to be at position 0 to start.
pub(crate) unsafe fn mem_zuncompress_and_write(hdl: c_int, buffer: &[u8], nbytes: usize) -> c_int {
    unsafe {
        let mut newsize = 0;
        let mut status = 0;

        let mut m = MEM_TABLE.lock().unwrap();

        if m[hdl as usize].currentpos != 0 {
            ffpmsg_str("cannot append uncompressed data (mem_uncompress_and_write)");
            return WRITE_ERROR;
        }

        uncompress2mem_from_mem(
            cast_slice(buffer),
            nbytes,
            m[hdl as usize].memaddrptr.cast::<*mut u8>(),
            m[hdl as usize].memsizeptr.as_mut().expect(NULL_MSG),
            m[hdl as usize].mem_realloc,
            Some(&mut newsize),
            &mut status,
        );

        if status != 0 {
            ffpmsg_str("unabled to uncompress memory file (mem_uncompress_and_write)");
            return WRITE_ERROR;
        }

        m[hdl as usize].currentpos += newsize as LONGLONG;
        m[hdl as usize].fitsfilesize = newsize as LONGLONG;
        0
    }
}

#[cfg(feature = "bzip2")]
pub(crate) unsafe fn bzip2uncompress2mem(
    filename: &[c_char],
    diskfile: *mut FILE,
    hdl: c_int,
    filesize: &mut usize,
    status: &mut c_int,
) {
    use libbz2_rs_sys::*;

    let mut b: *mut BZFILE = ptr::null_mut();
    let mut bzerror: c_int = 0;
    let buf: [c_char; 8192] = [0; 8192];
    let mut total_read: usize = 0;
    let mut errormsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

    *filesize = 0;
    *status = 0;

    unsafe {
        b = BZ2_bzReadOpen(&raw mut bzerror, diskfile, 0, 0, ptr::null_mut(), 0);
        if bzerror != BZ_OK {
            BZ2_bzReadClose(&raw mut bzerror, b);
            if bzerror == BZ_MEM_ERROR {
                ffpmsg_str("failed to open a bzip2 file: out of memory\n");
            } else if bzerror == BZ_CONFIG_ERROR {
                ffpmsg_str("failed to open a bzip2 file: miscompiled bzip2 library\n");
            } else if bzerror == BZ_IO_ERROR {
                ffpmsg_str("failed to open a bzip2 file: I/O error");
            } else {
                ffpmsg_str("failed to open a bzip2 file");
            }
            *status = READ_ERROR;
            return;
        }
        bzerror = BZ_OK;
        while bzerror == BZ_OK {
            let mut nread = 0;
            nread = BZ2_bzRead(
                &raw mut bzerror,
                b,
                buf.as_ptr() as *mut c_void,
                mem::size_of_val(&buf) as c_int,
            );
            if bzerror == BZ_OK || bzerror == BZ_STREAM_END {
                *status = mem_write_unsafe(hdl, cast_slice(&buf), nread as usize);
                if *status != 0 {
                    BZ2_bzReadClose(&raw mut bzerror, b);
                    if *status == MEMORY_ALLOCATION {
                        ffpmsg_str("Failed to reallocate memory while uncompressing bzip2 file");
                    }
                    return;
                }
                total_read += nread as usize;
            } else if bzerror == BZ_IO_ERROR {
                strcpy_safe(&mut errormsg, cs!(c"failed to read bzip2 file: I/O error"));
            } else if bzerror == BZ_UNEXPECTED_EOF {
                strcpy_safe(
                    &mut errormsg,
                    cs!(c"failed to read bzip2 file: unexpected end-of-file"),
                );
            } else if bzerror == BZ_DATA_ERROR {
                strcpy_safe(
                    &mut errormsg,
                    cs!(c"failed to read bzip2 file: data integrity error"),
                );
            } else if bzerror == BZ_MEM_ERROR {
                strcpy_safe(
                    &mut errormsg,
                    cs!(c"failed to read bzip2 file: insufficient memory"),
                );
            }
        }
        BZ2_bzReadClose(&raw mut bzerror, b);
        if bzerror != BZ_OK {
            if errormsg[0] != 0 {
                ffpmsg_slice(&errormsg);
            } else {
                ffpmsg_str("failure closing bzip2 file after reading\n");
            }
            *status = READ_ERROR;
            return;
        }
        *filesize = total_read;
    }
}
