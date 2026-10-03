# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.470.4] - 2026.10.04

### Fixed
- Decompression buffers grow as in CFITSIO instead of panicking with "Realloc function
  not implemented": GZIP tiles holding wider pixels than ZBITPIX decode, a compressed
  table VLA longer than its descriptor is written whole, `.gz` files whose ISIZE trailer
  understates the size open, and `ffimem` buffers grow through the caller's realloc
  (#139).
- Driver-allocated memory files are no longer passed to libc `realloc` when reading
  `.Z` files or stdin, or when shrinking after opening a compressed file.
- `.gz` output files can be created again; any compressed output name failed with
  `FILE_NOT_CREATED`.
- Malformed input returns a CFITSIO status instead of panicking or dividing by zero:
  non-UTF-8 keyword and TFORM values, RICE tiles whose BYTEPIX disagrees with ZBITPIX,
  tile descriptors past the end of the file, non-positive ZTILEn/ZNAXISn, and HCOMPRESS
  tiles with a short header or zero size (#139).
- Tile descriptors that run past the heap but stay inside the file are read, as in
  CFITSIO, rather than refused.

## [0.470.3] - 2026.09.28

### Changed
- The `*_safer` functions are renamed `*_safe` (29 public functions, e.g. `ffiopn_safer`
  is now `ffiopn_safe`). The `fits_*` aliases are unchanged; code calling the old names
  directly needs updating.

### Fixed
- gzip, PACK, LZW (`.Z`), LZH and PKZIP compressed files are recognised again: the magic
  numbers were ported from C's octal escapes as decimal, so a `.fits.gz` was parsed as a
  raw FITS header and failed to open (#138).
- `ffiopn`/`fftopn` return an error status instead of panicking when the open fails
  (#136, #137).
- Read and seek failures while opening a compressed file return `READ_ERROR` /
  `SEEK_ERROR` instead of panicking.

## [0.470.2] - 2026.08.23

### Changed
- `Box<FITSfile>` replaced with a shared handle; less `unsafe` outside the FFI wrappers,
  which are now marked with a `// FFI WRAPPER` comment.
- The H-compress shims transpiled.
- The transpiled C comments converted to rustdoc across the crate.
- flate2 uses its `miniz_oxide` backend, avoiding a `zlib-rs` soundness bug.

### Fixed
- Memory leaks in the expression evaluator: scratch allocations, `fffrwc` parameter
  arrays, `ffiter`, histogram iterator columns, parser and constant-node row buffers,
  and BITSTR column buffers.
- Pointer aliasing and provenance bugs found by Miri in the lexer, parser slot pointers,
  `split_names` and the expression evaluator; the last Stacked Borrows violations; the
  Miri run is now clean.
- `fits_recalloc` alignment, and driver-owned memory buffers resized with Rust's
  allocator.
- Integer exponentiation rounds instead of truncating.
- `colData[-1]` indexed when an expression names no columns.
- TSTRING output slices sized from the column rather than `FLEN_VALUE`.
- Nightly clippy lints in the `net_services` and `shared_mem` features.

## [0.470.1] - 2026.08.16

### Changed
- Large reduction in raw `libc` and unsafe pointer usage: the last raw libc string functions
  are gone from the expression engine, references and constness changes now go through
  `ptr::from_ref`/`from_mut`, `cast_mut`/`cast_const`, `pointer::cast` and `&raw`, and the five
  raw-pointer cast lints are denied package-wide.
- Pedantic clippy lints cleared across the library, tests and the transpiled binaries.
- C-ABI buffer reinterpretations now assert pointer alignment, and the tiled-image scratch
  buffers are allocated with explicit alignment.
- Miri findings documented and the remaining known issues recorded.

### Fixed
- Undefined behaviour reported by Miri: aliasing in the lexer buffer stack, the
  self-referential memory driver, and pointer provenance in `grparser` and `getcol`.
- The lexer no longer defaults `yyin`/`yyout` to the C stdio externs.
- Short-read handling in the memory, file, IRAF and region readers.
- Double free when the `headstart` array fails to grow (#100).

## [0.470.0] - 2026.08.15

### Added
- `fpack` and `funpack`, transpiled from CFITSIO.
- `fitsverify`, transpiled from CFITSIO, with unit tests and coverage.
- Differential oracle test running the expression parser against a recorded corpus.

### Changed
- Synced to CFITSIO 4.7.0 (`d6d2765`).
- Expression parser rewritten around typed enums (`NodeValue`, `ValueSort`, `funcOp`, typed
  operators) instead of unions and integer tags; GTI and region loading split out of
  `New_GTI`/`New_REG`, with regions shared behind an `Rc`.
- Large reduction in `unsafe`: 68 redundant blocks removed, raw pointers dropped from several
  `_safe` functions, error stack messages moved to a fixed-size buffer.
- Clippy clean across the crate, tests and examples; VAX/VMS/Alpha conditionals removed.
- `pliocomp` 0.6, for its worst-case buffer bound.

### Fixed
- Quickselect pivot bug breaking lossy compression reproducibility against CFITSIO (#82).
- `usize` underflow when unshuffling GZIP_2 tiles (#89).
- Image and table compression defects exposed by `fpack`/`funpack`; `BITPIX` and `BLOCKED`
  restored to the img2comp keyword table.
- `printf` conformance for `%g` and the `#` flag.
- `ffdtyp` string detection, hex literal case and bare `.` lexing, constant-folded `ANGSEP`,
  four parser bugs found while testing the expression operators, and a pending syntax error
  being discarded during error recovery.
- The stdout/stdin stream driver, which now writes its memory file through Rust rather than
  the C runtime.

## [0.464.0] - 2026.07.12

### Added
- Test harness ported from CFITSIO, and a large body of unit tests including the compression
  routines.
- Implementations for the remaining `todo!()` stubs: group and hierarchy routines (`ffgt*`,
  `ffgm*`), URL helpers (`fits_get_url`, `fits_url2path`, `fits_relurl2url`, ...), `ffimem`,
  `ffextn`, `ffexist`, `ffpcln`, `ffparsecompspec` and `fits_copy_cell2image`.

### Changed
- Use `core` instead of `std` where possible.

### Fixed
- A significant number of bugs found by the new test harness.
- Compatibility problems found by integration testing against the Python `fitsio` library.
- `iraffits` handling of long Windows filenames and `/` in place of `\`.
- Incorrect `DLONG_MAX`/`DLONG_MIN` on Windows, incorrect `c_char` usage, and mutex poisoning.

## [0.463.0] - 2026.05.30

### Added
- `grparser` (template parser) implementation.
- `histo` (histogram) implementation.

### Changed
- Minimum supported Rust version is 1.96.
- `grparser` uses indices and `Box<[T]>` in place of raw pointers.

### Fixed
- Use after free bug.
- `c_char` casting issue.
- Crash caused by bit string row filters.

## [0.462.10] - 2025.09.20

### Fixed
- Fixed compile issue with smem unexpectingly being enabled on windows.

## [0.462.9] - 2025.09.20

### Fixed
- Fixed compile error

## [0.462.8] - 2025.09.20

### Added
- Comprehensive eval* tests for table where clauses and calculator expressions
- Expanded cookbook examples for binary tables
- Unit tests for eval functions and row expressions
- Python fitsio compatibility improvements

### Changed
- Rewritten error handling with new ErrorStack and ErrorMessage structs for thread safety

### Fixed
- Bzip2 feature now fully functional
- Multiple safety issues and TODO implementations across modkey.rs, putkey.rs
- Various bugs found through Python fitsio integration testing
- Overflow handling in eval functions
- Table where clause parsing and execution

## [0.462.7] - 2025-08-04

### Fixed
- Critical bug typo

## [0.462.6] - 2025-08-04

### Added
- Shared memory code and utility functionality (previously non-functional feature)
- Expanded test coverage

### Changed
- Further reduced libc dependencies from cfileio.rs and edithdu.rs modules

### Fixed
- Bugs with long keywords and null NAXES handling
- fgets usage in region.rs

## [0.462.5] - 2025-06-29

### Added
- CI support for MacOS (Intel and ARM), Windows (x86_64 and ARM), and Linux ARM64
- Iterator examples demonstrating FITS file traversal

### Changed
- ZCompress module now uses Read/Write traits for better Rust integration

### Removed
- Dependency on nightly variadic functions - now works on stable Rust

### Fixed
- Platform-specific issues for MacOS and Windows builds
- stdin/stdout/stderr linking on MacOS
- Type compatibility issues on Windows (c_long vs LONGLONG)
- Various TODOs and documentation improvements

## [0.462.4] - 2025-06-17

### Added
- Cookbook examples (`cookbook_c` and `cookbook_rust`) demonstrating FITS file operations
- Speed utility benchmark tool (`speed` binary) for performance testing
- GitHub Actions integration with dependabot for automated dependency updates

### Removed
- Removed dependency on `errno` crate

### Fixed
- Various TODO items throughout the codebase
- Minor code improvements and optimizations

## [0.462.3] - 2025-06-06

### Added
- Split aliases into separate `c_api` and `rust_api` modules for better organization
- More safe wrapper functions for improved memory safety

### Changed
- Improved visibility controls for internal functions
- Better separation between C-compatible and Rust-native APIs

### Fixed
- Fixed visibility issues with certain functions
- Various formatting and clippy warnings

## [0.462.2] - 2025-06-01

### Added
- Full Miri support - all tests now pass under Miri memory safety checker
- Codecov integration for test coverage reporting

### Changed
- Updated from Rust edition 2021 to 2024
- Minimum supported Rust version set to 1.87
- Removed dependency on nightly features (except variadics)

### Removed
- Removed various libc function dependencies in favor of safe Rust alternatives

### Fixed
- Many Miri-detected memory safety issues
- Various unsafe code patterns replaced with safe alternatives
- Improved error handling throughout the codebase

## [0.462.1] - 2024-XX-XX

### Changed
- Initial line-by-line Rust translation of cfitsio v4.6.2
- Maintains full C API compatibility

### Added
- Dual API design: C-compatible functions and safer Rust APIs
- Pluggable I/O driver system
- Compression support (rice, plio, hcompress)
- Thread-safe error handling

### Known Issues
- Features `bzip2` and `shared_mem` not fully functional
- Limited macOS testing
- Some platform-specific code uses outdated cfg values (winnt, vms)

## Notes

This project is a direct Rust translation of the cfitsio C library (v4.7.0), maintaining
compatibility while providing Rust's memory safety guarantees. The version numbering
follows cfitsio's version with an additional patch number for Rust-specific changes.

Synchronization with upstream cfitsio is tracked in `SYNCED_COMMIT.md`.