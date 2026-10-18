//! The public CFITSIO datatypes, symbolic constants and error status codes.
//!
//! Ported from CFITSIO's `fitsio.h`, written by William Pence at the High
//! Energy Astrophysics Science Archive Research Center (HEASARC), NASA Goddard
//! Space Flight Center. The descriptions of the error status codes are taken
//! from the "CFITSIO Error Status Codes" appendix of the CFITSIO User's
//! Reference Guide.
//!
//! Every routine in this crate returns an error status code as its `c_int`
//! return value and through its `status` parameter; `0` means success. Prefer
//! the symbolic names here over the integer values.
#![warn(missing_docs)]

use crate::{
    c_types::c_ulonglong,
    helpers::{boxed::box_try_new, vec_raw_parts::vec_into_raw_parts},
};
use bytemuck::{cast_slice, cast_slice_mut};
use core::{
    cmp,
    ffi::{CStr, c_char, c_int, c_long, c_longlong},
    ops::{Deref, DerefMut},
    ptr,
};
use std::os::raw::c_void;

use crate::{
    fitscore::{ALLOCATIONS, ffpmsg_slice},
    int_snprintf, slice_to_str,
    wrappers::strlen_safe,
};

use crate::cfileio::fitsdriver;

/// Whether the platform provides `ftruncate()`, so a file can be shortened in
/// place.
pub const HAVE_FTRUNCATE: bool = true;

/// The FITS block length, 2880 bytes, as a literal usable in a constant
/// expression such as an array length.
#[macro_export]
macro_rules! BL {
    () => {
        2880
    };
}

/// Panic message used when an `extern "C"` wrapper is handed a null pointer it
/// is not allowed to receive.
pub const NULL_MSG: &str = "Null Pointer";

/// Length of a FITS block, in bytes. A FITS file is an integral number of
/// these.
pub const BLOCK_LEN: usize = 2880;

/// The CFITSIO version this crate is a port of, as a nul-terminated string.
pub const CFITSIO_VERSION: [u8; 6] = *b"4.7.0\0";

/* Minor and micro numbers must not exceed 99 under current method of version representataion in ffvers(). */
/// Micro part of the CFITSIO version. Must not exceed 99, given how `ffvers`
/// packs the version number.
pub const CFITSIO_MICRO: u64 = 0;
/// Minor part of the CFITSIO version. Must not exceed 99, given how `ffvers`
/// packs the version number.
pub const CFITSIO_MINOR: u64 = 7;
/// Major part of the CFITSIO version.
pub const CFITSIO_MAJOR: u64 = 4;
/* the SONAME is incremented in a new release if the binary shared */
/* library (on linux and Mac systems) is not backward compatible */
/* with the previous release of CFITSIO */
/// Shared library version. Incremented in a new release when the binary shared
/// library (on Linux and macOS) is not backward compatible with the previous
/// release of CFITSIO.
pub const CFITSIO_SONAME: u64 = 10;

/// Whether 64-bit integer literals need an `LL` suffix on this platform.
pub const USE_LL_SUFFIX: u64 = 1;

/// The 64-bit signed integer datatype. CFITSIO spells it `LONGLONG` because C
/// has no portable name for it.
pub type LONGLONG = c_longlong;
/// The 64-bit unsigned integer datatype.
pub type ULONGLONG = c_ulonglong;

/*  define a default value, even if it is never used */
/// Largest value a [`LONGLONG`] can hold.
pub const LONGLONG_MAX: c_longlong = c_longlong::MAX;
/// Smallest value a [`LONGLONG`] can hold.
pub const LONGLONG_MIN: c_longlong = c_longlong::MIN;

/// Largest value a `c_long` can hold. Target-dependent: 64-bit on Linux and
/// macOS, 32-bit on Windows.
pub const LONG_MAX: c_long = c_long::MAX;
/// Smallest value a `c_long` can hold. Target-dependent: 64-bit on Linux and
/// macOS, 32-bit on Windows.
pub const LONG_MIN: c_long = c_long::MIN;

/// Number of I/O buffers to create. Significantly increasing this may *degrade*
/// performance.
pub const NIOBUF: u64 = 40;

/// Size in bytes of each I/O buffer. Do not change: it is one FITS block.
pub const IOBUFLEN: i64 = 2880;

/// Maximum length of a filename.
pub const FLEN_FILENAME: usize = 1025;
/// Maximum length of a keyword name, allowing for the HIERARCH convention.
pub const FLEN_KEYWORD: usize = 75;
/// Length of a FITS header card, plus the nul terminator.
pub const FLEN_CARD: usize = 81;
/// Maximum length of a keyword value string.
pub const FLEN_VALUE: usize = 71;
/// Maximum length of a keyword comment string.
pub const FLEN_COMMENT: usize = 73;
/// Maximum length of a FITSIO error message.
pub const FLEN_ERRMSG: usize = 81;
/// Maximum length of a FITSIO status text string.
pub const FLEN_STATUS: usize = 31;

/// Datatype code for a bit column in a FITS table.
pub const TBIT: c_int = 1;
/// Datatype code for an unsigned byte (`unsigned char`).
pub const TBYTE: c_int = 11;
/// Datatype code for a signed byte (`signed char`).
pub const TSBYTE: c_int = 12;
/// Datatype code for a logical value (`int`).
pub const TLOGICAL: c_int = 14;
/// Datatype code for a character string.
pub const TSTRING: c_int = 16;
/// Datatype code for an `unsigned short`.
pub const TUSHORT: c_int = 20;
/// Datatype code for a `short`.
pub const TSHORT: c_int = 21;
/// Datatype code for an `unsigned int`.
pub const TUINT: c_int = 30;
/// Datatype code for an `int`.
pub const TINT: c_int = 31;
/// Datatype code for an `unsigned long`.
pub const TULONG: c_int = 40;
/// Datatype code for a `long`.
pub const TLONG: c_int = 41;
/// Datatype code for a 32-bit integer. Used when returning the datatype of a
/// column; the same value as [`TLONG`].
pub const TINT32BIT: c_int = 41;
/// Datatype code for a `float`.
pub const TFLOAT: c_int = 42;
/// Datatype code for an unsigned 64-bit integer.
pub const TULONGLONG: c_int = 80;
/// Datatype code for a signed 64-bit integer.
pub const TLONGLONG: c_int = 81;
/// Datatype code for a `double`.
pub const TDOUBLE: c_int = 82;
/// Datatype code for a single-precision complex value.
pub const TCOMPLEX: c_int = 83;
/// Datatype code for a double-precision complex value.
pub const TDBLCOMPLEX: c_int = 163;

/// Keyword class: a structural keyword (`SIMPLE`, `BITPIX`, `NAXIS`, `END`, …).
pub const TYP_STRUC_KEY: c_int = 10;
/// Keyword class: a tile-compression keyword (`ZIMAGE`, `ZCMPTYPE`, …).
pub const TYP_CMPRS_KEY: c_int = 20;
/// Keyword class: a scaling keyword (`BSCALE`, `BZERO`, `TSCALn`, `TZEROn`).
pub const TYP_SCAL_KEY: c_int = 30;
/// Keyword class: a null-value keyword (`BLANK`, `TNULLn`).
pub const TYP_NULL_KEY: c_int = 40;
/// Keyword class: a dimension keyword (`TDIMn`).
pub const TYP_DIM_KEY: c_int = 50;
/// Keyword class: a range keyword (`TLMINn`, `TLMAXn`, `TDMINn`, `TDMAXn`).
pub const TYP_RANG_KEY: c_int = 60;
/// Keyword class: a units keyword (`TUNITn`).
pub const TYP_UNIT_KEY: c_int = 70;
/// Keyword class: a display-format keyword (`TDISPn`).
pub const TYP_DISP_KEY: c_int = 80;
/// Keyword class: an HDU identification keyword (`EXTNAME`, `EXTVER`, …).
pub const TYP_HDUID_KEY: c_int = 90;
/// Keyword class: a checksum keyword (`CHECKSUM`, `DATASUM`).
pub const TYP_CKSUM_KEY: c_int = 100;
/// Keyword class: a world coordinate system keyword.
pub const TYP_WCS_KEY: c_int = 110;
/// Keyword class: a reference system keyword (`EQUINOX`, `RADESYS`, …).
pub const TYP_REFSYS_KEY: c_int = 120;
/// Keyword class: a commentary keyword (`COMMENT`, `HISTORY`, blank).
pub const TYP_COMM_KEY: c_int = 130;
/// Keyword class: a `CONTINUE` keyword of the long-string convention.
pub const TYP_CONT_KEY: c_int = 140;
/// Keyword class: any other, user-defined keyword.
pub const TYP_USER_KEY: c_int = 150;

/* 32-bit integer datatype.  Currently this       */
/* datatype is an 'int' on all useful platforms   */
/* however, it is possible that that are cases    */
/* where 'int' is a 2-byte integer, in which case */
/* INT32BIT would need to be defined as 'long'.   */
/// A 32-bit integer datatype.
///
/// This is an `int` on every useful platform, but there could be cases where
/// `int` is a 2-byte integer, in which case it would have to be a `long`.
pub type INT32BIT = c_int;

/// `BITPIX` code for an 8-bit unsigned integer image.
pub const BYTE_IMG: c_int = 8;
/// `BITPIX` code for a 16-bit signed integer image.
pub const SHORT_IMG: c_int = 16;
/// `BITPIX` code for a 32-bit signed integer image.
pub const LONG_IMG: c_int = 32;
/// `BITPIX` code for a 64-bit signed integer image.
pub const LONGLONG_IMG: c_int = 64;
/// `BITPIX` code for a single-precision floating point image.
pub const FLOAT_IMG: c_int = -32;
/// `BITPIX` code for a double-precision floating point image.
pub const DOUBLE_IMG: c_int = -64;
/* The following 2 codes are not true FITS         */
/* datatypes; these codes are only used internally */
/* within cfitsio to make it easier for users      */
/* to deal with unsigned integers.                 */
/// `BITPIX` code for a signed-byte image.
///
/// Not a true FITS datatype: used only internally, to make signed bytes easier
/// to deal with. Equivalent to [`BYTE_IMG`] with a `BZERO` of -128.
pub const SBYTE_IMG: c_int = 10;
/// `BITPIX` code for an unsigned 16-bit integer image.
///
/// Not a true FITS datatype: equivalent to [`SHORT_IMG`] with a `BZERO` of
/// 32768, which is how FITS stores unsigned integers.
pub const USHORT_IMG: c_int = 20;
/// `BITPIX` code for an unsigned 32-bit integer image.
///
/// Not a true FITS datatype: equivalent to [`LONG_IMG`] with a `BZERO` of
/// 2147483648.
pub const ULONG_IMG: c_int = 40;
/// `BITPIX` code for an unsigned 64-bit integer image.
///
/// Not a true FITS datatype: equivalent to [`LONGLONG_IMG`] with a `BZERO` of
/// 9223372036854775808.
pub const ULONGLONG_IMG: c_int = 80;

/// HDU type code for a primary array or IMAGE extension.
pub const IMAGE_HDU: c_int = 0;
/// HDU type code for an ASCII table extension.
pub const ASCII_TBL: c_int = 1;
/// HDU type code for a binary table extension.
pub const BINARY_TBL: c_int = 2;
/// HDU type code matching any HDU type.
pub const ANY_HDU: c_int = -1;

/// File access mode: open for reading only.
pub const READONLY: c_int = 0;
/// File access mode: open for reading and writing.
pub const READWRITE: c_int = 1;

/* adopt a hopefully obscure number to use as a null value flag */
/// Sentinel `float` meaning "this pixel is undefined".
///
/// A hopefully obscure value, so that it does not collide with real data.
pub const FLOATNULLVALUE: f32 = -9.11912E-36;
/// Sentinel `double` meaning "this pixel is undefined".
///
/// A hopefully obscure value, so that it does not collide with real data.
pub const DOUBLENULLVALUE: f64 = -9.1191291391491E-36;

/// Quantization dithering: do not dither.
pub const NO_DITHER: c_int = -1;
/// Quantization dithering: subtractive dither, starting from a fixed offset.
pub const SUBTRACTIVE_DITHER_1: c_int = 1;
/// Quantization dithering: as [`SUBTRACTIVE_DITHER_1`], but zero-valued pixels
/// are preserved exactly.
pub const SUBTRACTIVE_DITHER_2: c_int = 2;

/// Maximum number of image dimensions the tile-compression code supports.
pub const MAX_COMPRESS_DIM: usize = 6;
/// Compression algorithm code for Rice.
pub const RICE_1: c_int = 11;
/// Compression algorithm code for GZIP.
pub const GZIP_1: c_int = 21;
/// Compression algorithm code for GZIP applied to byte-shuffled data.
pub const GZIP_2: c_int = 22;
/// Compression algorithm code for PLIO, for bit-mask images.
pub const PLIO_1: c_int = 31;
/// Compression algorithm code for Hcompress.
pub const HCOMPRESS_1: c_int = 41;
/// Compression algorithm code for BZIP2. Not publicly supported; for test
/// purposes only.
pub const BZIP2_1: c_int = 51;
/// Compression algorithm code meaning "do not compress".
pub const NOCOMPRESS: c_int = -1;

/// Boolean true, as CFITSIO spells it.
pub const TRUE: u64 = 1;

/// Boolean false, as CFITSIO spells it.
pub const FALSE: u64 = 0;

/// Do a case-sensitive string match.
pub const CASESEN: u64 = 1;
/// Do a case-insensitive string match.
pub const CASEINSEN: u64 = 0;

/// Grouping table member identification: by all of URI, position and reference.
pub const GT_ID_ALL_URI: u64 = 0;
/// Grouping table member identification: by reference.
pub const GT_ID_REF: u64 = 1;
/// Grouping table member identification: by position.
pub const GT_ID_POS: u64 = 2;
/// Grouping table member identification: by both reference and position.
pub const GT_ID_ALL: u64 = 3;
/// Grouping table member identification: by reference and URI.
pub const GT_ID_REF_URI: u64 = 11;
/// Grouping table member identification: by position and URI.
pub const GT_ID_POS_URI: u64 = 12;

/// Grouping table removal option: remove the grouping table itself.
pub const OPT_RM_GPT: u64 = 0;
/// Grouping table removal option: remove the member's entry only.
pub const OPT_RM_ENTRY: u64 = 1;
/// Grouping table removal option: remove the member HDU.
pub const OPT_RM_MBR: u64 = 2;
/// Grouping table removal option: remove the grouping table and all members.
pub const OPT_RM_ALL: u64 = 3;

/// Grouping table copy option: copy the grouping table only.
pub const OPT_GCP_GPT: u64 = 0;
/// Grouping table copy option: copy the member HDUs.
pub const OPT_GCP_MBR: u64 = 1;
/// Grouping table copy option: copy the grouping table and all its members.
pub const OPT_GCP_ALL: u64 = 2;

/// Group member copy option: add the copy to the same grouping table.
pub const OPT_MCP_ADD: u64 = 0;
/// Group member copy option: do not add the copy to a grouping table.
pub const OPT_MCP_NADD: u64 = 1;
/// Group member copy option: replace the original member with the copy.
pub const OPT_MCP_REPL: u64 = 2;
/// Group member copy option: move the member rather than copying it.
pub const OPT_MCP_MOV: u64 = 3;

/// Grouping table merge option: copy the members into the target table.
pub const OPT_MRG_COPY: u64 = 0;
/// Grouping table merge option: move the members into the target table.
pub const OPT_MRG_MOV: u64 = 1;

/// Grouping table compact option: compact member grouping tables.
pub const OPT_CMT_MBR: u64 = 1;
/// Grouping table compact option: compact member grouping tables and delete
/// them afterwards.
pub const OPT_CMT_MBR_DEL: u64 = 11;

/// Magic value stored in [`FITSfile::validcode`] to identify a valid structure.
pub const VALIDSTRUC: c_int = 555;

/// Iterator column flag: the column is read but not written.
pub const INPUT_COL: c_int = 0;
/// Iterator column flag: the column is both read and written.
pub const INPUT_OUTPUT_COL: c_int = 1;
/// Iterator column flag: the column is written but not read.
pub const OUTPUT_COL: c_int = 2;
/// Iterator column flag: a temporary column, used internally.
pub const TEMPORARY_COL: c_int = 3;

/* error status codes */

/// Create a disk file, without applying the extended filename syntax.
pub const CREATE_DISK_FILE: c_int = -106;
/// Open a disk file, without applying the extended filename syntax.
pub const OPEN_DISK_FILE: c_int = -105;
/// Move to the first image when opening the file.
pub const SKIP_TABLE: c_int = -104;
/// Move to the first table when opening the file.
pub const SKIP_IMAGE: c_int = -103;
/// Skip a null primary array when opening the file.
pub const SKIP_NULL_PRIMARY: c_int = -102;
/// Use a memory buffer when opening the file.
pub const USE_MEM_BUFF: c_int = -101;
/// Overflow during datatype conversion.
pub const OVERFLOW_ERR: c_int = -11;
/// Used in `ffiimg` to insert a new primary array.
pub const PREPEND_PRIMARY: c_int = -9;
/// Input and output files are the same.
pub const SAME_FILE: c_int = 101;
/// Tried to open too many FITS files at once.
pub const TOO_MANY_FILES: c_int = 103;
/// Could not open the named file.
pub const FILE_NOT_OPENED: c_int = 104;
/// Could not create the named file.
pub const FILE_NOT_CREATED: c_int = 105;
/// Error writing to the FITS file.
pub const WRITE_ERROR: c_int = 106;
/// Tried to move past the end of the file.
pub const END_OF_FILE: c_int = 107;
/// Error reading from the FITS file.
pub const READ_ERROR: c_int = 108;
/// Could not close the file.
pub const FILE_NOT_CLOSED: c_int = 110;
/// Array dimensions exceed an internal limit.
pub const ARRAY_TOO_BIG: c_int = 111;
/// Cannot write to a readonly file.
pub const READONLY_FILE: c_int = 112;
/// Could not allocate memory.
pub const MEMORY_ALLOCATION: c_int = 113;
/// Invalid [`fitsfile`] pointer.
pub const BAD_FILEPTR: c_int = 114;
/// Null input pointer to a routine.
pub const NULL_INPUT_PTR: c_int = 115;
/// Error seeking to a position in the file.
pub const SEEK_ERROR: c_int = 116;
/// Bad value for the file download timeout setting.
pub const BAD_NETTIMEOUT: c_int = 117;

/// Invalid URL prefix on the file name.
pub const BAD_URL_PREFIX: c_int = 121;
/// Tried to register too many I/O drivers.
pub const TOO_MANY_DRIVERS: c_int = 122;
/// Driver initialization failed.
pub const DRIVER_INIT_FAILED: c_int = 123;
/// The matching driver is not registered.
pub const NO_MATCHING_DRIVER: c_int = 124;
/// Failed to parse the input file URL.
pub const URL_PARSE_ERROR: c_int = 125;
/// Parse error in a range list.
pub const RANGE_PARSE_ERROR: c_int = 126;

/// Base value the shared memory driver's error codes are offset from.
pub const SHARED_ERRBASE: c_int = 150;
/// Bad argument in the shared memory driver.
pub const SHARED_BADARG: c_int = SHARED_ERRBASE + 1;
/// Null pointer passed as an argument to the shared memory driver.
pub const SHARED_NULPTR: c_int = SHARED_ERRBASE + 2;
/// No more free shared memory handles.
pub const SHARED_TABFULL: c_int = SHARED_ERRBASE + 3;
/// The shared memory driver is not initialized.
pub const SHARED_NOTINIT: c_int = SHARED_ERRBASE + 4;
/// IPC error returned by a system call.
pub const SHARED_IPCERR: c_int = SHARED_ERRBASE + 5;
/// No memory in the shared memory driver.
pub const SHARED_NOMEM: c_int = SHARED_ERRBASE + 6;
/// Resource deadlock would occur.
pub const SHARED_AGAIN: c_int = SHARED_ERRBASE + 7;
/// Attempt to open or create the lock file failed.
pub const SHARED_NOFILE: c_int = SHARED_ERRBASE + 8;
/// The shared memory block cannot be resized at the moment.
pub const SHARED_NORESIZE: c_int = SHARED_ERRBASE + 9;

/// The header already contains keywords.
pub const HEADER_NOT_EMPTY: c_int = 201;
/// The keyword was not found in the header.
pub const KEY_NO_EXIST: c_int = 202;
/// The keyword record number is out of bounds.
pub const KEY_OUT_BOUNDS: c_int = 203;
/// The keyword value field is blank.
pub const VALUE_UNDEFINED: c_int = 204;
/// The string is missing its closing quote.
pub const NO_QUOTE: c_int = 205;
/// Illegal indexed keyword name, e.g. `TFORM1000`.
pub const BAD_INDEX_KEY: c_int = 206;
/// Illegal character in a keyword name or card.
pub const BAD_KEYCHAR: c_int = 207;
/// Required keywords are out of order.
pub const BAD_ORDER: c_int = 208;
/// The keyword value is not a positive integer.
pub const NOT_POS_INT: c_int = 209;
/// Could not find the `END` keyword.
pub const NO_END: c_int = 210;
/// Illegal `BITPIX` keyword value.
pub const BAD_BITPIX: c_int = 211;
/// Illegal `NAXIS` keyword value.
pub const BAD_NAXIS: c_int = 212;
/// Illegal `NAXISn` keyword value.
pub const BAD_NAXES: c_int = 213;
/// Illegal `PCOUNT` keyword value.
pub const BAD_PCOUNT: c_int = 214;
/// Illegal `GCOUNT` keyword value.
pub const BAD_GCOUNT: c_int = 215;
/// Illegal `TFIELDS` keyword value.
pub const BAD_TFIELDS: c_int = 216;
/// Negative table row size.
pub const NEG_WIDTH: c_int = 217;
/// Negative number of rows in the table.
pub const NEG_ROWS: c_int = 218;
/// No column with this name was found in the table.
pub const COL_NOT_FOUND: c_int = 219;
/// Illegal value of the `SIMPLE` keyword.
pub const BAD_SIMPLE: c_int = 220;
/// The primary array does not start with `SIMPLE`.
pub const NO_SIMPLE: c_int = 221;
/// The second keyword is not `BITPIX`.
pub const NO_BITPIX: c_int = 222;
/// The third keyword is not `NAXIS`.
pub const NO_NAXIS: c_int = 223;
/// Could not find all the `NAXISn` keywords.
pub const NO_NAXES: c_int = 224;
/// The HDU does not start with the `XTENSION` keyword.
pub const NO_XTENSION: c_int = 225;
/// The current HDU is not an ASCII table extension.
pub const NOT_ATABLE: c_int = 226;
/// The current HDU is not a binary table extension.
pub const NOT_BTABLE: c_int = 227;
/// Could not find the `PCOUNT` keyword.
pub const NO_PCOUNT: c_int = 228;
/// Could not find the `GCOUNT` keyword.
pub const NO_GCOUNT: c_int = 229;
/// Could not find the `TFIELDS` keyword.
pub const NO_TFIELDS: c_int = 230;
/// Could not find the `TBCOLn` keyword.
pub const NO_TBCOL: c_int = 231;
/// Could not find the `TFORMn` keyword.
pub const NO_TFORM: c_int = 232;
/// The current HDU is not an IMAGE extension.
pub const NOT_IMAGE: c_int = 233;
/// `TBCOLn` keyword value is < 0 or > the row length.
pub const BAD_TBCOL: c_int = 234;
/// The current HDU is not a table.
pub const NOT_TABLE: c_int = 235;
/// The column is too wide to fit in the table.
pub const COL_TOO_WIDE: c_int = 236;
/// More than one column name matches the template.
pub const COL_NOT_UNIQUE: c_int = 237;
/// The sum of the column widths does not equal `NAXIS1`.
pub const BAD_ROW_WIDTH: c_int = 241;
/// Unrecognizable FITS extension type.
pub const UNKNOWN_EXT: c_int = 251;
/// Unknown record: the first keyword is neither `SIMPLE` nor `XTENSION`.
pub const UNKNOWN_REC: c_int = 252;
/// The `END` keyword record is not blank.
pub const END_JUNK: c_int = 253;
/// The header fill area contains non-blank characters.
pub const BAD_HEADER_FILL: c_int = 254;
/// Illegal data fill bytes: neither zero nor blank.
pub const BAD_DATA_FILL: c_int = 255;
/// Illegal `TFORM` format code.
pub const BAD_TFORM: c_int = 261;
/// Unrecognizable `TFORM` datatype code.
pub const BAD_TFORM_DTYPE: c_int = 262;
/// Illegal `TDIMn` keyword value.
pub const BAD_TDIM: c_int = 263;
/// The binary table heap pointer is out of range.
pub const BAD_HEAP_PTR: c_int = 264;

/// HDU number < 1 or > `MAXHDU`.
pub const BAD_HDU_NUM: c_int = 301;
/// Column number < 1 or > `tfields`.
pub const BAD_COL_NUM: c_int = 302;
/// Tried to move to a negative byte location in the file.
pub const NEG_FILE_POS: c_int = 304;
/// Tried to read or write a negative number of bytes.
pub const NEG_BYTES: c_int = 306;
/// Illegal starting row number in the table.
pub const BAD_ROW_NUM: c_int = 307;
/// Illegal starting element number in the vector.
pub const BAD_ELEM_NUM: c_int = 308;
/// This is not an ASCII string column.
pub const NOT_ASCII_COL: c_int = 309;
/// This is not a logical datatype column.
pub const NOT_LOGICAL_COL: c_int = 310;
/// The ASCII table column has the wrong format.
pub const BAD_ATABLE_FORMAT: c_int = 311;
/// The binary table column has the wrong format.
pub const BAD_BTABLE_FORMAT: c_int = 312;
/// The null value has not been defined.
pub const NO_NULL: c_int = 314;
/// This is not a variable length column.
pub const NOT_VARI_LEN: c_int = 317;
/// Illegal number of dimensions in the array.
pub const BAD_DIMEN: c_int = 320;
/// The first pixel number is greater than the last pixel.
pub const BAD_PIX_NUM: c_int = 321;
/// Illegal `BSCALE` or `TSCALn` keyword value of 0.
pub const ZERO_SCALE: c_int = 322;
/// Illegal axis length < 1.
pub const NEG_AXIS: c_int = 323;

/// Grouping function error: the HDU is not a grouping table.
pub const NOT_GROUP_TABLE: c_int = 340;
/// Grouping function error: the HDU is already a member of the group.
pub const HDU_ALREADY_MEMBER: c_int = 341;
/// Grouping function error: the group member was not found.
pub const MEMBER_NOT_FOUND: c_int = 342;
/// Grouping function error: the grouping table was not found.
pub const GROUP_NOT_FOUND: c_int = 343;
/// Grouping function error: bad group identifier.
pub const BAD_GROUP_ID: c_int = 344;
/// Grouping function error: too many HDUs are being tracked.
pub const TOO_MANY_HDUS_TRACKED: c_int = 345;
/// Grouping function error: the HDU is already being tracked.
pub const HDU_ALREADY_TRACKED: c_int = 346;
/// Grouping function error: bad option value.
pub const BAD_OPTION: c_int = 347;
/// Grouping function error: the two file pointers are identical.
pub const IDENTICAL_POINTERS: c_int = 348;
/// Grouping function error: could not attach the HDU to the group.
pub const BAD_GROUP_ATTACH: c_int = 349;
/// Grouping function error: could not detach the HDU from the group.
pub const BAD_GROUP_DETACH: c_int = 350;

/// Bad `int` to formatted string conversion.
pub const BAD_I2C: c_int = 401;
/// Bad `float` to formatted string conversion.
pub const BAD_F2C: c_int = 402;
/// Cannot interpret the keyword value as an integer.
pub const BAD_INTKEY: c_int = 403;
/// Cannot interpret the keyword value as a logical.
pub const BAD_LOGICALKEY: c_int = 404;
/// Cannot interpret the keyword value as a float.
pub const BAD_FLOATKEY: c_int = 405;
/// Cannot interpret the keyword value as a double.
pub const BAD_DOUBLEKEY: c_int = 406;
/// Bad formatted string to `int` conversion.
pub const BAD_C2I: c_int = 407;
/// Bad formatted string to `float` conversion.
pub const BAD_C2F: c_int = 408;
/// Bad formatted string to `double` conversion.
pub const BAD_C2D: c_int = 409;
/// Illegal datatype code value.
pub const BAD_DATATYPE: c_int = 410;
/// Bad number of decimal places specified.
pub const BAD_DECIM: c_int = 411;
/// Overflow during datatype conversion.
pub const NUM_OVERFLOW: c_int = 412;

/// Error compressing the image.
pub const DATA_COMPRESSION_ERR: c_int = 413;
/// Error uncompressing the image.
pub const DATA_DECOMPRESSION_ERR: c_int = 414;
/// The compressed tile does not exist.
pub const NO_COMPRESSED_TILE: c_int = 415;

/// Error in a date or time conversion.
pub const BAD_DATE: c_int = 420;

/// Syntax error in a parser expression.
pub const PARSE_SYNTAX_ERR: c_int = 431;
/// The expression did not evaluate to the desired type.
pub const PARSE_BAD_TYPE: c_int = 432;
/// The vector result is too large to return in the array.
pub const PARSE_LRG_VECTOR: c_int = 433;
/// The data parser was not sent an output column.
pub const PARSE_NO_OUTPUT: c_int = 434;
/// Bad data encountered while parsing a column.
pub const PARSE_BAD_COL: c_int = 435;
/// The output file is not of the proper type.
pub const PARSE_BAD_OUTPUT: c_int = 436;

/// Celestial angle too large for the projection.
pub const ANGLE_TOO_BIG: c_int = 501;
/// Bad celestial coordinate or pixel value.
pub const BAD_WCS_VAL: c_int = 502;
/// Error in a celestial coordinate calculation.
pub const WCS_ERROR: c_int = 503;
/// Unsupported type of celestial projection.
pub const BAD_WCS_PROJ: c_int = 504;
/// The celestial coordinate keywords were not found.
pub const NO_WCS_KEY: c_int = 505;
/// Approximate WCS keyword values were returned.
pub const APPROX_WCS_KEY: c_int = 506;

/// Special value used internally to switch off the error message from `ffclos`
/// and `ffchdu`.
pub const NO_CLOSE_ERROR: c_int = 999;

// The following error codes are used by the header template parser in
// grparser.rs.
/// Base value the template parser's error codes are offset from. Chosen so as
/// not to interfere with the CFITSIO codes.
pub const NGP_ERRBASE: c_int = 360;
/// Template parser: no error.
pub const NGP_OK: c_int = 0;
/// Template parser: memory allocation failed.
pub const NGP_NO_MEMORY: c_int = NGP_ERRBASE;
/// Template parser: read error from the file.
pub const NGP_READ_ERR: c_int = NGP_ERRBASE + 1;
/// Template parser: null pointer passed as an argument. Usually means a null
/// pointer was passed as the name of the template file.
pub const NGP_NUL_PTR: c_int = NGP_ERRBASE + 2;
/// Template parser: the line read seems to be empty. Used internally.
pub const NGP_EMPTY_CURLINE: c_int = NGP_ERRBASE + 3;
/// Template parser: cannot unread more than one line, or unread a single line
/// twice.
pub const NGP_UNREAD_QUEUE_FULL: c_int = NGP_ERRBASE + 4;
/// Template parser: include file nesting is too deep — an infinite loop, or a
/// template that includes itself.
pub const NGP_INC_NESTING: c_int = NGP_ERRBASE + 5;
/// Template parser: `fopen()` failed, cannot open the template file.
pub const NGP_ERR_FOPEN: c_int = NGP_ERRBASE + 6;
/// Template parser: end of file encountered where it was not expected.
pub const NGP_EOF: c_int = NGP_ERRBASE + 7;
/// Template parser: bad arguments passed. Usually an internal parser error that
/// should not happen.
pub const NGP_BAD_ARG: c_int = NGP_ERRBASE + 8;
/// Template parser: token not expected here.
pub const NGP_TOKEN_NOT_EXPECT: c_int = NGP_ERRBASE + 9;

/// Stores table column information.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tcolumn {
    /// Column name — the FITS `TTYPEn` keyword.
    pub ttype: [c_char; 70],
    /// Offset in the row to the first byte of the column.
    pub tbcol: LONGLONG,
    /// Datatype code of the column.
    pub tdatatype: c_int,
    /// Repeat count of the column: the number of elements.
    pub trepeat: LONGLONG,
    /// FITS `TSCALn` linear scaling factor.
    pub tscale: f64,
    /// FITS `TZEROn` linear scaling zero point.
    pub tzero: f64,
    /// FITS null value, for an integer image or binary table column.
    pub tnull: LONGLONG,
    /// FITS null value string, for an ASCII table column.
    pub strnull: [c_char; 20],
    /// FITS `TFORMn` keyword value.
    pub tform: [c_char; 10],
    /// Width of the column, for an ASCII table.
    pub twidth: c_long,
}

impl Default for tcolumn {
    fn default() -> Self {
        Self {
            ttype: [0; 70],
            tbcol: Default::default(),
            tdatatype: Default::default(),
            trepeat: Default::default(),
            tscale: Default::default(),
            tzero: Default::default(),
            tnull: Default::default(),
            strnull: Default::default(),
            tform: Default::default(),
            twidth: Default::default(),
        }
    }
}

/// Structure used to store basic FITS file information.
#[repr(C)]
pub struct FITSfile {
    /// Handle returned by the file open function.
    pub filehandle: c_int,
    /// Defines which set of I/O drivers should be used.
    pub driver: c_int,
    /// Number of opened 'fitsfiles' using this structure.
    pub open_count: c_int,
    /// File name.
    pub filename: *mut c_char,
    /// Magic value used to verify that structure is valid.
    pub validcode: c_int,
    /// Flag meaning only copy the specified extension.
    pub only_one: c_int,
    /// Flag for file opened with request to ignore extended syntax.
    pub noextsyntax: c_int,
    /// Current size of the physical disk file in bytes.
    pub filesize: LONGLONG,
    /// Logical size of file, including unflushed buffers.
    pub logfilesize: LONGLONG,
    /// Is this the last HDU in the file? 0 = no, else yes.
    pub lasthdu: c_int,
    /// Current logical I/O pointer position in file.
    pub bytepos: LONGLONG,
    /// Current I/O pointer position in the physical file.
    pub io_pos: LONGLONG,
    /// Number of I/O buffer currently in use.
    pub curbuf: c_int,
    /// Current HDU number; 0 = primary array.
    pub curhdu: c_int,
    /// 0 = primary array, 1 = ASCII table, 2 = binary table.
    pub hdutype: c_int,
    /// 0 = readonly, 1 = readwrite.
    pub writemode: c_int,
    /// Highest numbered HDU known to exist in the file.
    pub maxhdu: c_int,
    /// Dynamically allocated dimension of headstart array.
    pub MAXHDU: c_int,
    /// Byte offset in file to start of each HDU.
    pub headstart: *mut LONGLONG,
    /// Byte offset in file to end of the current HDU header.
    pub headend: LONGLONG,
    /// Byte offest to where the END keyword was last written.
    pub ENDpos: LONGLONG,
    /// Byte offset in file to beginning of next keyword.
    pub nextkey: LONGLONG,
    /// Byte offset in file to start of the current data unit.
    pub datastart: LONGLONG,
    /// Dimension of image; cached for fast access.
    pub imgdim: c_int,
    /// Length of each axis; cached for fast access.
    pub imgnaxis: [LONGLONG; 99],
    /// Number of fields in the table (primary array has 2).
    pub tfield: c_int,
    /// Used by ffgcnn to record starting column number.
    pub startcol: c_int,
    /// Original number of rows (value of NAXIS2 keyword).
    pub origrows: LONGLONG,
    /// Number of rows in the table (dynamically updated).
    pub numrows: LONGLONG,
    /// Length of a table row or image size (bytes).
    pub rowlength: LONGLONG,
    /// Pointer to the table structure.
    pub tableptr: *mut tcolumn,
    /// Heap start byte relative to start of data unit.
    pub heapstart: LONGLONG,
    /// Size of the heap, in bytes.
    pub heapsize: LONGLONG,

    // The following elements are related to compressed images

    // These record the 'requested' options to be used when the image is compressed
    /// Requested image compression algorithm.
    pub request_compress_type: c_int,
    /// Requested tiling size.
    pub request_tilesize: [c_long; 6],
    /// Requested quantize level.
    pub request_quantize_level: f32,
    /// Requested quantizing method.
    pub request_quantize_method: c_int,
    /// Starting offset into the array of random dithering.
    pub request_dither_seed: c_int,
    /// Lossy compress integer image as if float image?
    pub request_lossy_int_compress: c_int,
    /// Use '1Q' rather then '1P' variable length arrays.
    pub request_huge_hdu: c_int,
    /// Requested HCOMPRESS scale factor.
    pub request_hcomp_scale: f32,
    /// Requested HCOMPRESS smooth parameter.
    pub request_hcomp_smooth: c_int,

    // These record the actual options that were used when the image was compressed
    /// Type of compression algorithm.
    pub compress_type: c_int,
    /// Size of compression tiles.
    pub tilesize: [c_long; 6],
    /// Floating point quantization level.
    pub quantize_level: f32,
    /// Floating point pixel quantization algorithm.
    pub quantize_method: c_int,
    /// Starting offset into the array of random dithering.
    pub dither_seed: c_int,

    // Other compression parameters
    /// 1 if HDU contains a compressed image, else 0.
    pub compressimg: c_int,
    /// Compression type string.
    pub zcmptype: [c_char; 12],
    /// FITS data type of image (BITPIX).
    pub zbitpix: c_int,
    /// Dimension of image.
    pub zndim: c_int,
    /// Length of each axis.
    pub znaxis: [c_long; 6],
    /// Max number of pixels in each image tile.
    pub maxtilelen: c_long,
    /// Maximum byte length of tile compressed arrays.
    pub maxelem: c_long,

    /// Column number for COMPRESSED_DATA column.
    pub cn_compressed: c_int,
    /// Column number for UNCOMPRESSED_DATA column.
    pub cn_uncompressed: c_int,
    /// Column number for GZIP2 lossless compressed data.
    pub cn_gzip_data: c_int,
    /// Column number for ZSCALE column.
    pub cn_zscale: c_int,
    /// Column number for ZZERO column.
    pub cn_zzero: c_int,
    /// Column number for the ZBLANK column.
    pub cn_zblank: c_int,

    /// Scaling value, if same for all tiles.
    pub zscale: f64,
    /// Zero pt, if same for all tiles.
    pub zzero: f64,
    /// Value of the BSCALE keyword in header.
    pub cn_bscale: f64,
    /// Value of the BZERO keyword (may be reset).
    pub cn_bzero: f64,
    /// Actual value of the BZERO keyword.
    pub cn_actual_bzero: f64,
    /// Value for null pixels, if not a column.
    pub zblank: c_int,

    /// First compression parameter: Rice pixels/block.
    pub rice_blocksize: c_int,
    /// Second compression parameter: Rice bytes/pixel.
    pub rice_bytepix: c_int,
    /// First hcompress compression parameter.
    pub hcomp_scale: f32,
    /// Second hcompress compression parameter.
    pub hcomp_smooth: c_int,

    /// Row number of the array of uncompressed tiledata.
    pub tilerow: *mut c_int,
    /// Length of the array of tile data in bytes.
    pub tiledatasize: *mut c_long,
    /// Datatype of the array of tile (TINT, TSHORT, etc).
    pub tiletype: *mut c_int,
    /// Array of uncompressed tile of data, for row *tilerow.
    pub tiledata: *mut *mut c_void,
    /// Array of optional array of null value flags.
    pub tilenullarray: *mut *mut c_void,
    /// Anynulls in the array of tile?
    pub tileanynull: *mut c_int,

    /// Pointer to FITS file I/O buffers.
    pub iobuffer: Box<[c_char; NIOBUF as usize * IOBUFLEN as usize]>,
    /// File record number of each of the buffers.
    pub bufrecnum: [c_long; 40],
    /// Has the corresponding buffer been modified?
    pub dirty: [c_int; 40],
    /// Relative age of each buffer.
    pub ageindex: [c_int; 40],
}

impl Default for FITSfile {
    fn default() -> Self {
        Self {
            filehandle: Default::default(),
            driver: Default::default(),
            open_count: Default::default(),
            filename: core::ptr::null_mut(),
            validcode: Default::default(),
            only_one: Default::default(),
            noextsyntax: Default::default(),
            filesize: Default::default(),
            logfilesize: Default::default(),
            lasthdu: Default::default(),
            bytepos: Default::default(),
            io_pos: Default::default(),
            curbuf: Default::default(),
            curhdu: Default::default(),
            hdutype: Default::default(),
            writemode: Default::default(),
            maxhdu: Default::default(),
            MAXHDU: Default::default(),
            headstart: core::ptr::null_mut(),
            headend: Default::default(),
            ENDpos: Default::default(),
            nextkey: Default::default(),
            datastart: Default::default(),
            imgdim: Default::default(),
            imgnaxis: [0; 99],
            tfield: Default::default(),
            startcol: Default::default(),
            origrows: Default::default(),
            numrows: Default::default(),
            rowlength: Default::default(),
            tableptr: core::ptr::null_mut(),
            heapstart: Default::default(),
            heapsize: Default::default(),
            request_compress_type: Default::default(),
            request_tilesize: Default::default(),
            request_quantize_level: Default::default(),
            request_quantize_method: Default::default(),
            request_dither_seed: Default::default(),
            request_lossy_int_compress: Default::default(),
            request_huge_hdu: Default::default(),
            request_hcomp_scale: Default::default(),
            request_hcomp_smooth: Default::default(),
            compress_type: Default::default(),
            tilesize: Default::default(),
            quantize_level: Default::default(),
            quantize_method: Default::default(),
            dither_seed: Default::default(),
            compressimg: Default::default(),
            zcmptype: Default::default(),
            zbitpix: Default::default(),
            zndim: Default::default(),
            znaxis: Default::default(),
            maxtilelen: Default::default(),
            maxelem: Default::default(),
            cn_compressed: Default::default(),
            cn_uncompressed: Default::default(),
            cn_gzip_data: Default::default(),
            cn_zscale: Default::default(),
            cn_zzero: Default::default(),
            cn_zblank: Default::default(),
            zscale: Default::default(),
            zzero: Default::default(),
            cn_bscale: Default::default(),
            cn_bzero: Default::default(),
            cn_actual_bzero: Default::default(),
            zblank: Default::default(),
            rice_blocksize: Default::default(),
            rice_bytepix: Default::default(),
            hcomp_scale: Default::default(),
            hcomp_smooth: Default::default(),
            tilerow: core::ptr::null_mut(),
            tiledatasize: core::ptr::null_mut(),
            tiletype: core::ptr::null_mut(),
            tiledata: core::ptr::null_mut(),
            tilenullarray: core::ptr::null_mut(),
            tileanynull: core::ptr::null_mut(),
            iobuffer: Box::new([0; (NIOBUF as usize * IOBUFLEN as usize)]),
            bufrecnum: [0; 40],
            dirty: [0; 40],
            ageindex: [0; 40],
        }
    }
}

impl FITSfile {
    /// Allocates a `FITSfile` for an already-opened file, with its filename,
    /// `headstart` array and I/O buffers reserved.
    ///
    /// # Parameters
    ///
    /// * `driver` — (I) the I/O driver the file was opened with
    /// * `handle` — (I) the driver's handle for the open file
    /// * `url`    — (I) the file name, used for the error message
    /// * `caller` — (I) name of the calling routine, used for the error message
    /// * `status` — (IO) error status
    ///
    /// # Errors
    ///
    /// Closes `handle` and returns [`MEMORY_ALLOCATION`] if any of the three
    /// allocations fails.
    pub fn new(
        driver: &fitsdriver,
        handle: c_int,
        url: &[c_char],
        caller: &[c_char],
        status: &mut c_int,
    ) -> Result<Box<Self>, c_int> {
        let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];

        let mut slen = strlen_safe(url) + 1;
        slen = cmp::max(slen, 32); /* reserve at least 32 chars */

        // HEAP ALLOCATION
        /* mem for file name */
        let mut f_filename = Vec::new();
        if f_filename.try_reserve_exact(slen).is_err() {
            (driver.close)(handle); /* close the file */
            int_snprintf!(
                &mut errmsg,
                FLEN_ERRMSG,
                "failed to allocate memory for filename: ({}),",
                slice_to_str!(caller),
            );
            ffpmsg_slice(&errmsg);
            ffpmsg_slice(url);
            *status = MEMORY_ALLOCATION;
            return Err(*status);
        } else {
            f_filename.resize(slen, 0);
        }

        // HEAP ALLOCATION
        /* mem for headstart array */
        let mut f_headstart = Vec::new();
        if f_headstart.try_reserve_exact(1001).is_err() {
            (driver.close)(handle); /* close the file */
            int_snprintf!(
                &mut errmsg,
                FLEN_ERRMSG,
                "failed to allocate memory for headstart array: ({}),",
                slice_to_str!(caller),
            );
            ffpmsg_slice(&errmsg);
            ffpmsg_slice(url);
            *status = MEMORY_ALLOCATION;
            return Err(*status);
        } else {
            f_headstart.resize(1001, 0);
        }

        // HEAP ALLOCATION
        /* mem for file I/O buffers */
        let mut f_iobuffer = Vec::new();
        let iosize = NIOBUF as usize * IOBUFLEN as usize;
        if f_iobuffer.try_reserve_exact(iosize).is_err() {
            (driver.close)(handle);
            int_snprintf!(
                &mut errmsg,
                FLEN_ERRMSG,
                "failed to allocate memory for iobuffer array: ({}),",
                slice_to_str!(caller),
            );
            ffpmsg_slice(&errmsg);
            ffpmsg_slice(url);
            *status = MEMORY_ALLOCATION;
            return Err(*status);
        } else {
            f_iobuffer.resize(iosize, 0);
        }

        let f_iobuffer: Box<[c_char; NIOBUF as usize * IOBUFLEN as usize]> =
            f_iobuffer.into_boxed_slice().try_into().unwrap();

        let (f_headstart, l, c) = vec_into_raw_parts(f_headstart);
        crate::lock_global(&ALLOCATIONS).insert(f_headstart as usize, (l, c));

        let (f_filename, l, c) = vec_into_raw_parts(f_filename);
        crate::lock_global(&ALLOCATIONS).insert(f_filename as usize, (l, c));

        // HEAP ALLOCATION
        /* allocate FITSfile structure and initialize = 0 */
        let b = box_try_new(FITSfile {
            iobuffer: f_iobuffer,
            headstart: f_headstart,
            filename: f_filename,
            ..Default::default()
        });
        if b.is_err() {
            (driver.close)(handle); /* close the file */
            int_snprintf!(
                &mut errmsg,
                FLEN_ERRMSG,
                "failed to allocate structure for following file: ({}),",
                slice_to_str!(caller),
            );
            ffpmsg_slice(&errmsg);
            ffpmsg_slice(url);
            *status = MEMORY_ALLOCATION;
            return Err(*status);
        }
        Ok(b.unwrap())
    }

    /// The byte offset of the start of each HDU, as a slice of `MAXHDU + 1`
    /// entries.
    pub fn get_headstart_as_slice(&self) -> &[LONGLONG] {
        unsafe { core::slice::from_raw_parts(self.headstart, self.MAXHDU as usize + 1) }
    }

    /// The byte offset of the start of each HDU, mutably.
    pub fn get_headstart_as_mut_slice(&mut self) -> &mut [LONGLONG] {
        unsafe { core::slice::from_raw_parts_mut(self.headstart, self.MAXHDU as usize + 1) }
    }

    /// The [`NIOBUF`] I/O buffers of [`IOBUFLEN`] bytes each, as one flat
    /// slice, reached through a raw pointer rather than through `self`.
    pub fn get_iobuffer(iobuffer: &mut *mut c_char) -> &[c_char] {
        unsafe { core::slice::from_raw_parts(*iobuffer, NIOBUF as usize * IOBUFLEN as usize) }
    }

    /// The I/O buffers as one flat slice, mutably.
    pub fn get_iobuffer_mut(iobuffer: &mut *mut c_char) -> &mut [c_char] {
        unsafe { core::slice::from_raw_parts_mut(*iobuffer, NIOBUF as usize * IOBUFLEN as usize) }
    }

    /// The table's column descriptions, one per field.
    pub fn get_tableptr_as_slice(&self) -> &[tcolumn] {
        unsafe { core::slice::from_raw_parts(self.tableptr, self.tfield as usize) }
    }

    /// The table's column descriptions, reached through a raw pointer and a
    /// caller-supplied length rather than through `self`, so the returned
    /// slice does not borrow the `FITSfile`.
    pub fn get_tableptr_as_slice_unlinked<'a>(
        tableptr: &*mut tcolumn,
        len: usize,
    ) -> &'a [tcolumn] {
        unsafe { core::slice::from_raw_parts(*tableptr, len) }
    }

    /// The table's column descriptions, mutably.
    pub fn get_tableptr_as_mut_slice(&mut self) -> &mut [tcolumn] {
        unsafe { core::slice::from_raw_parts_mut(self.tableptr, self.tfield as usize) }
    }

    /// Number of tiles along the first axis of a compressed image, which is
    /// the length every per-tile array below is allocated to.
    pub fn get_tile_alloc_len(&self) -> usize {
        ((((self).znaxis[0] - 1) / ((self).tilesize[0])) + 1) as usize
    }

    /// The row number of each cached uncompressed tile.
    pub fn get_tilerow_as_slice(&self) -> &[c_int] {
        unsafe { core::slice::from_raw_parts(self.tilerow, self.get_tile_alloc_len()) }
    }

    /// The row number of each cached uncompressed tile, mutably.
    pub fn get_tilerow_as_mut_slice(&mut self) -> &mut [c_int] {
        unsafe { core::slice::from_raw_parts_mut(self.tilerow, self.get_tile_alloc_len()) }
    }

    /// The length in bytes of each cached tile's data.
    pub fn get_tiledatasize_as_slice(&self) -> &[c_long] {
        unsafe { core::slice::from_raw_parts(self.tiledatasize, self.get_tile_alloc_len()) }
    }

    /// The length in bytes of each cached tile's data, mutably.
    pub fn get_tiledatasize_as_mut_slice(&mut self) -> &mut [c_long] {
        unsafe { core::slice::from_raw_parts_mut(self.tiledatasize, self.get_tile_alloc_len()) }
    }

    /// The datatype code ([`TINT`], [`TSHORT`], …) of each cached tile.
    pub fn get_tiletype_as_slice(&self) -> &[c_int] {
        unsafe { core::slice::from_raw_parts(self.tiletype, self.get_tile_alloc_len()) }
    }

    /// The datatype code of each cached tile, mutably.
    pub fn get_tiletype_as_mut_slice(&mut self) -> &mut [c_int] {
        unsafe { core::slice::from_raw_parts_mut(self.tiletype, self.get_tile_alloc_len()) }
    }

    /// The uncompressed data of each cached tile.
    pub fn get_tiledata_as_slice(&self) -> &[*mut c_void] {
        unsafe { core::slice::from_raw_parts(self.tiledata, self.get_tile_alloc_len()) }
    }

    /// The uncompressed data of each cached tile, mutably.
    pub fn get_tiledata_as_mut_slice(&mut self) -> &mut [*mut c_void] {
        unsafe { core::slice::from_raw_parts_mut(self.tiledata, self.get_tile_alloc_len()) }
    }

    /// The optional null-value flags of each cached tile.
    pub fn get_tilenullarray_as_slice(&self) -> &[*mut c_void] {
        unsafe { core::slice::from_raw_parts(self.tilenullarray, self.get_tile_alloc_len()) }
    }

    /// The optional null-value flags of each cached tile, mutably.
    pub fn get_tilenullarray_as_mut_slice(&mut self) -> &mut [*mut c_void] {
        unsafe { core::slice::from_raw_parts_mut(self.tilenullarray, self.get_tile_alloc_len()) }
    }

    /*

    pub fn get_tiledata_as_slice(&self) -> &[&[c_void]] {
        let len = self.get_tile_alloc_len();
        let mut v = Vec::with_capacity(len);
        for i in 0..len {
            let start = i * self.tiledatasize[i] as usize;
            let end = start + self.tiledatasize[i] as usize;
            v.push(&self.tiledata[start..end]);
        }
        v
    }


    pub fn get_tiledata_as_mut_slice(&mut self) -> &mut [&mut [c_void]] {
        let len = self.get_tile_alloc_len();
        let mut v = Vec::with_capacity(len);
        for i in 0..len {
            let start = i * self.tiledatasize[i] as usize;
            let end = start + self.tiledatasize[i] as usize;
            v.push(&mut self.tiledata[start..end]);
        }
        v
    }

    pub fn get_tilenullarray_as_slice(&self) -> &[&[c_void]] {
        let len = self.get_tile_alloc_len();
        let mut v = Vec::with_capacity(len);
        for i in 0..len {
            let start = i * self.tiledatasize[i] as usize;
            let end = start + self.tiledatasize[i] as usize;
            v.push(&self.tilenullarray[start..end]);
        }
        v
    }

    pub fn get_tilenullarray_as_mut_slice(&mut self) -> &mut [&mut [c_void]] {
        let len = self.get_tile_alloc_len();
        let mut v = Vec::with_capacity(len);
        for i in 0..len {
            let start = i * self.tiledatasize[i] as usize;
            let end = start + self.tiledatasize[i] as usize;
            v.push(&mut self.tilenullarray[start..end]);
        }
        v
    }
    */

    /// Whether each cached tile contains any null values.
    pub fn get_tileanynull_as_slice(&self) -> &[c_int] {
        unsafe { core::slice::from_raw_parts(self.tileanynull, self.get_tile_alloc_len()) }
    }

    /// Whether each cached tile contains any null values, mutably.
    pub fn get_tileanynull_as_mut_slice(&mut self) -> &mut [c_int] {
        unsafe { core::slice::from_raw_parts_mut(self.tileanynull, self.get_tile_alloc_len()) }
    }

    /// The file name as a borrowed C string.
    pub fn get_filename_as_cstr(&self) -> &CStr {
        unsafe { CStr::from_ptr(self.filename) }
    }

    /// The heap-allocated filename buffer as a writable slice, so callers can
    /// use `strcpy_safe` instead of a raw `strcpy` into `self.filename`.
    ///
    /// The allocation's length is the one recorded in `ALLOCATIONS` when
    /// `FITSfile::new` reserved it -- the same entry `Drop` frees against. 32
    /// is the minimum `new` reserves, and the size `Drop` falls back to when
    /// the entry is missing.
    pub(crate) fn get_filename_as_mut_slice(&mut self) -> &mut [c_char] {
        let len = crate::lock_global(&ALLOCATIONS)
            .get(&(self.filename as usize))
            .map_or(32, |&(l, _)| l);

        // SAFETY: filename points at a `len`-element allocation owned by self.
        unsafe { core::slice::from_raw_parts_mut(self.filename, len) }
    }
}

impl Drop for FITSfile {
    fn drop(&mut self) {
        unsafe {
            if !self.filename.is_null() {
                // HEAP DEALLOCATION
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.filename as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.filename, l, c);
                } else {
                    let _ = Vec::from_raw_parts(self.filename, 32, 32);
                }
            }

            if !self.headstart.is_null() {
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.headstart as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.headstart, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        self.headstart,
                        (self.MAXHDU as usize) + 1,
                        (self.MAXHDU as usize) + 1,
                    );
                }
            }

            if !self.tableptr.is_null() {
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.tableptr as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.tableptr, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        self.tableptr,
                        self.tfield as usize,
                        self.tfield as usize,
                    );
                }
            }

            if !self.tilerow.is_null() {
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.tilerow as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.tilerow, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        self.tilerow,
                        self.get_tile_alloc_len(),
                        self.get_tile_alloc_len(),
                    );
                }
            }

            if !self.tiledatasize.is_null() {
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.tiledatasize as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.tiledatasize, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        self.tiledatasize,
                        self.get_tile_alloc_len(),
                        self.get_tile_alloc_len(),
                    );
                }
            }

            if !self.tiletype.is_null() {
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.tiletype as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.tiletype, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        self.tiletype,
                        self.get_tile_alloc_len(),
                        self.get_tile_alloc_len(),
                    );
                }
            }

            if !self.tiledata.is_null() {
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.tiledata as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.tiledata, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        self.tiledata,
                        self.get_tile_alloc_len(),
                        self.get_tile_alloc_len(),
                    );
                }
            }

            if !self.tilenullarray.is_null() {
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.tilenullarray as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.tilenullarray, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        self.tilenullarray,
                        self.get_tile_alloc_len(),
                        self.get_tile_alloc_len(),
                    );
                }
            }

            if !self.tileanynull.is_null() {
                let mut alloc_lock = crate::lock_global(&ALLOCATIONS);
                let alloc = alloc_lock.remove(&(self.tileanynull as usize));
                if let Some((l, c)) = alloc {
                    // HEAP DEALLOCATION
                    let _ = Vec::from_raw_parts(self.tileanynull, l, c);
                } else {
                    let _ = Vec::from_raw_parts(
                        self.tileanynull,
                        self.get_tile_alloc_len(),
                        self.get_tile_alloc_len(),
                    );
                }
            }
        }
    }
}

/// Handle to a [`FITSfile`], which several [`fitsfile`]s may share.
///
/// CFITSIO lets more than one `fitsfile` refer to the same `FITSfile`, keeping
/// a use count in its `open_count` field. `Box` cannot express that: two live
/// `Box`es to one allocation is aliasing UB, and because `Box` is `noalias`
/// every write through one invalidates pointers derived from the other, which
/// is what invalidated the entries in `FPTR_TABLE`.
///
/// This is a non-owning handle. Exactly one place frees the `FITSfile`:
/// `ffclos`, once `open_count` reaches zero. There is deliberately no `Drop`
/// impl, because closing has to flush and close the driver before the free and
/// a refcount-on-drop would fight the decrement already done there.
///
/// `#[repr(transparent)]` over a `NonNull`, so the layout is a single non-null
/// pointer exactly as `Box<FITSfile>` was and `fitsfile` keeps its C ABI.
///
/// Deliberately not `Send` (nor `Sync`; `NonNull` is neither). Opening the
/// same path read-write again reuses the already-open `FITSfile`
/// (`fits_already_open`), so two `fitsfile`s can share one `FITSfile` and both
/// deref it mutably with no locking. If a `fitsfile` could move to another
/// thread, safe code could then mutate one `FITSfile` from two threads at once.
/// Keeping the handle `!Send` makes `fitsfile` and `Box<fitsfile>` `!Send`,
/// so every handle to a `FITSfile` stays on the thread that opened it.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<rsfitsio::fitsio::fitsfile>();
/// ```
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<Box<rsfitsio::fitsio::fitsfile>>();
/// ```
#[repr(transparent)]
pub struct FptrRef(ptr::NonNull<FITSfile>);

impl FptrRef {
    /// Allocate a new `FITSfile` and take the first handle to it.
    pub fn new(f: Box<FITSfile>) -> Self {
        // SAFETY: Box::into_raw never returns null.
        FptrRef(unsafe { ptr::NonNull::new_unchecked(Box::into_raw(f)) })
    }

    /// Another handle to the same `FITSfile`.
    ///
    /// The caller is responsible for the `open_count` bookkeeping.
    pub fn share(&self) -> Self {
        FptrRef(self.0)
    }

    /// The address of the `FITSfile`.
    ///
    /// Callers that stash this (`FPTR_TABLE`) get the same pointer this handle
    /// derefs through, not a reference-derived child of it, so it stays valid
    /// across writes made through any handle.
    pub fn as_ptr(&self) -> *mut FITSfile {
        self.0.as_ptr()
    }

    /// Build a handle from a pointer obtained from [`FptrRef::as_ptr`].
    ///
    /// # Safety
    /// `p` must be non-null and point to a live `FITSfile`.
    pub unsafe fn from_ptr(p: *mut FITSfile) -> Self {
        FptrRef(unsafe { ptr::NonNull::new_unchecked(p) })
    }

    /// Free the `FITSfile`. Only valid once `open_count` has reached zero and
    /// no other handle remains.
    ///
    /// # Safety
    /// No other `FptrRef` to this `FITSfile` may be used afterwards.
    pub unsafe fn free(self) {
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

impl Deref for FptrRef {
    type Target = FITSfile;

    fn deref(&self) -> &FITSfile {
        // SAFETY: the FITSfile outlives every handle to it; see `free`.
        unsafe { self.0.as_ref() }
    }
}

impl DerefMut for FptrRef {
    fn deref_mut(&mut self) -> &mut FITSfile {
        // SAFETY: as above.
        unsafe { self.0.as_mut() }
    }
}


/// Stores basic HDU information. This is the handle every routine in the API
/// takes.
#[repr(C)]
pub struct fitsfile {
    /// HDU position in the file; 0 = the first HDU.
    pub HDUposition: c_int,

    /// The file this HDU belongs to.
    pub Fptr: FptrRef,
}

/// Describes one column to the iterator function, as input to
/// `fits_iterate_data`.
///
/// Do not change the layout: it is part of the public API.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct iteratorCol {
    /// pointer to the HDU containing the column
    pub fptr: *mut fitsfile,
    /// column number in the table (use name if < 1)
    pub colnum: c_int,
    /// name (= TTYPEn value) of the column (optional)
    pub colname: [c_char; 70],
    /// output datatype (converted if necessary
    pub datatype: c_int,
    /// output elements that may be useful for the work function: = InputCol, InputOutputCol, or OutputCol
    pub iotype: c_int,
    /// pointer to the array (and the null value)
    pub array: *mut c_void,
    /// binary table vector repeat value
    pub repeat: c_long,
    /// legal minimum data value
    pub tlmin: c_long,
    /// legal maximum data value
    pub tlmax: c_long,
    /// physical unit string
    pub tunit: [c_char; 70],
    /// suggested display format
    pub tdisp: [c_char; 70],
}

impl Default for iteratorCol {
    fn default() -> Self {
        Self {
            fptr: Default::default(),
            colnum: Default::default(),
            colname: [0; 70],
            datatype: Default::default(),
            iotype: Default::default(),
            array: Default::default(),
            repeat: Default::default(),
            tlmin: Default::default(),
            tlmax: Default::default(),
            tunit: [0; 70],
            tdisp: [0; 70],
        }
    }
}

/// Describes an array to be extracted from a binary table by
/// `fits_read_wcstab()`, for use with Mark Calabretta's WCSLIB.
///
/// In order to keep WCSLIB and CFITSIO independent of one another, neither
/// library's headers may include the other's. WCSLIB therefore defines
/// `struct wtbarr` while `fitsio.h` defines an untagged `typedef wtbarr` with
/// identical members — legal in C because structure tags and typedef names live
/// in different namespaces. Despite the two spellings these are the same
/// aggregate type, though passing a `struct wtbarr *` to `fits_read_wcstab()`
/// formally requires a cast.
///
/// See <http://www.atnf.csiro.au/~mcalabre/index.html>.
#[repr(C)]
pub struct wtbarr {
    /// Image axis number.
    pub i: c_int,
    /// Array axis number for index vectors.
    pub m: c_int,
    /// Array type: `'c'` (coord) or `'i'` (index).
    pub kind: c_int,
    /// `EXTNAME` of the binary table extension.
    pub extnam: [c_char; 72],
    /// `EXTVER` of the binary table extension.
    pub extver: c_int,
    /// `EXTLEV` of the binary table extension.
    pub extlev: c_int,
    /// `TTYPEn` of the column containing the array.
    pub ttype: [c_char; 72],
    /// Table row number.
    pub row: c_long,
    /// Expected array dimensionality.
    pub ndim: c_int,
    /// Where to write the array axis lengths.
    pub dimlen: *mut c_int,
    /// Where to write the address of the array allocated to store the array.
    pub arrayp: *mut *mut f64,
}

// `dimlen` and `arrayp` point at dynamically allocated memory, which Rust will
// not free on its own, so `wtbarr` frees them itself.
impl Drop for wtbarr {
    fn drop(&mut self) {
        unsafe {
            if !self.dimlen.is_null() {
                let _ = Vec::from_raw_parts(self.dimlen, self.ndim as usize, self.ndim as usize);
            }
            if !self.arrayp.is_null() {
                let _ = Vec::from_raw_parts(*self.arrayp, self.ndim as usize, self.ndim as usize);
            }
        }
    }
}

/// Inputs to, and output control for, the pixel filtering routines.
///
/// Do not change the layout: it is exposed through the `extern "C"` entry
/// points.
#[repr(C)]
pub struct PixelFilter {
    /// Number of input files.
    pub count: c_int,
    /// Paths of the input files.
    pub path: *mut *mut c_char,
    /// Tag naming each input file within the expression.
    pub tag: *mut *mut c_char,
    /// Open file pointer for each input file.
    pub ifptr: *mut *mut fitsfile,
    /// The filtering expression to evaluate.
    pub expression: *mut c_char,
    /// `BITPIX` of the output image.
    pub bitpix: c_int,
    /// `BLANK` value of the output image.
    pub blank: c_long,
    /// Open file pointer for the output file.
    pub ofptr: *mut fitsfile,
    /// Keyword to record the expression under in the output header.
    pub keyword: [c_char; FLEN_KEYWORD],
    /// Comment for that keyword.
    pub comment: [c_char; FLEN_COMMENT],
}
