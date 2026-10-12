#![allow(deprecated)]
// Ported from the CFITSIO C test programs; indexed loops and the
// operator-named test fns mirror the originals.
#![allow(clippy::needless_range_loop)]

use core::slice;
use std::ffi::c_void;
use std::ptr;
use std::{ffi::CStr, process::ExitCode};

use bytemuck::cast_slice;

use libc::{free, malloc, strcpy};

use rsfitsio::aliases::c_api::fits_open_file;
use rsfitsio::buffers::{ffflus, ffgtbb, ffptbb};
use rsfitsio::c_types::{c_char, c_int, c_long, c_short, c_uchar, c_ulong};
use rsfitsio::cfileio::{ffdelt, ffinit};
use rsfitsio::checksum::{ffdsum, ffesum, ffgcks, ffpcks, ffupck, ffvcks};
use rsfitsio::cs;
use rsfitsio::editcol::{ffcpcl, ffcpky, ffdcol, ffdrow, fficol, ffirow};
use rsfitsio::edithdu::{ffcopy, ffdhdu, ffibin, ffiimg, ffitab};
use rsfitsio::fitscore::{
    ffcrhd, ffgacl, ffgbcl, ffgcnn, ffgcno, ffgdes, ffghdn, ffgtcl, ffhdef, ffmahd, ffmnhd, ffmrhd,
    ffrdef, ffthdu,
};
use rsfitsio::fitsio::{
    ANY_HDU, ASCII_TBL, COL_NOT_FOUND, FLEN_CARD, FLEN_COMMENT, FLEN_ERRMSG, FLEN_KEYWORD,
    FLEN_VALUE, LONGLONG, NUM_OVERFLOW, TBYTE, TDOUBLE, TFLOAT, TINT, TLOGICAL, TLONG, TRUE,
    TSHORT, TSTRING,
};
use rsfitsio::getcol::{ffgcv, ffgpv};
use rsfitsio::getcolb::{ffgcfb, ffgcvb, ffgpfb, ffgpvb};
use rsfitsio::getcold::{ffgcfd, ffgcfm, ffgcvd, ffgcvm, ffgpfd, ffgpvd};
use rsfitsio::getcole::{ffgcfc, ffgcfe, ffgcvc, ffgcve, ffgpfe, ffgpve};
use rsfitsio::getcoli::{ffg2di, ffgcfi, ffgcvi, ffgpfi, ffgpvi, ffgsvi};
use rsfitsio::getcolj::{ffgcvj, ffgpfj, ffgpvj};
use rsfitsio::getcolk::{ffgcfk, ffgcvk};
use rsfitsio::getcoll::{ffgcfl, ffgcl, ffgcx};
use rsfitsio::getcols::{ffgcfs, ffgcvs};
use rsfitsio::getkey::{
    fffree, ffgcrd, ffghbn, ffghpr, ffghps, ffghsp, ffghtb, ffgkey, ffgkls, ffgknd, ffgkne, ffgknj,
    ffgknl, ffgkns, ffgky, ffgkyc, ffgkyd, ffgkye, ffgkyj, ffgkyl, ffgkym, ffgkyn, ffgkys, ffgkyt,
    ffgnxk, ffgrec, ffgtdm, ffgunt,
};
use rsfitsio::modkey::{
    ffdkey, ffdrec, ffikyd, ffikye, ffikyf, ffikyg, ffikyj, ffikyl, ffikys, ffirec, ffmcom, ffmcrd,
    ffmkyd, ffmkye, ffmkyf, ffmkyg, ffmkyj, ffmkyl, ffmkys, ffmnam, ffmrec, ffpunt, ffucrd, ffukyd,
    ffukye, ffukyf, ffukyg, ffukyj, ffukyl, ffukys,
};
use rsfitsio::putcol::{ffpcl, ffppr, ffppx, ffppxn};
use rsfitsio::putcolb::{ffpclb, ffpcnb};
use rsfitsio::putcold::{ffpcld, ffpclm, ffpcnd};
use rsfitsio::putcole::{ffpclc, ffpcle, ffpcne};
use rsfitsio::putcoli::{ffp2di, ffpcli, ffpcni, ffpssi};
use rsfitsio::putcolj::ffpclj;
use rsfitsio::putcolk::{ffpclk, ffpcnk};
use rsfitsio::putcoll::{ffpcll, ffpclx};
use rsfitsio::putcols::ffpcls;
use rsfitsio::putcolu::{ffpclu, ffppru};
use rsfitsio::putkey::{
    ffcrtb, ffpcom, ffpdat, ffphbn, ffphis, ffphps, ffpkfc, ffpkfm, ffpkls, ffpknd, ffpkne, ffpknf,
    ffpkng, ffpknj, ffpknl, ffpkns, ffpktp, ffpky, ffpkyc, ffpkyd, ffpkye, ffpkyf, ffpkyg, ffpkyj,
    ffpkyl, ffpkym, ffpkys, ffpkyt, ffplsw, ffprec, ffptdm,
};
use rsfitsio::scalnull::{ffsnul, fftnul, fftscl};
use rsfitsio::wcssub::ffgics;
use rsfitsio::wcsutil::{ffwldp, ffxypx};
use rsfitsio::wrappers::{strcat_safe, strncmp_safe, strncpy_safe};
use rsfitsio::{
    cfileio::ffclos,
    fitscore::{ffcmsg, ffflmd, ffflnm, ffgerr, ffgmsg},
    fitsio::{READWRITE, fitsfile},
    wrappers::strcpy_safe,
};

fn strcpy_cstr(dest: &mut [c_char], src: &CStr) {
    strcpy_safe(dest, cast_slice(src.to_bytes_with_nul()));
}

macro_rules! byte_slice_to_str {
    ($slice:expr) => {
        CStr::from_bytes_until_nul(cast_slice($slice))
            .unwrap()
            .to_str()
            .unwrap()
    };
}

pub trait AsMutPtr<T> {
    fn as_mut_ptr(&mut self) -> *mut T;
}

impl<T> AsMutPtr<T> for Option<Box<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            Some(val) => core::ptr::from_mut::<T>(val.as_mut()),
            None => ptr::null_mut(),
        }
    }
}

pub fn main() -> ExitCode {
    let mut asciisum: [c_char; 17] = [0; 17];
    let mut checksum: c_ulong;
    let mut datsum: c_ulong = 0;
    let mut datastatus: c_int = 0;
    let mut hdustatus: c_int = 0;
    let mut filemode: c_int = 0;
    let mut status: c_int = 0;
    let mut simple: c_int;
    let mut bitpix: c_int;
    let mut naxis: c_int;
    let mut extend: c_int;
    let mut hdutype: c_int = 0;
    let mut hdunum: c_int = 0;
    let mut tfields: c_int;

    let mut extvers: c_long;
    let nkeys: c_int;
    let mut nfound: c_int = 0;
    let mut colnum: c_int = 0;
    let mut typecode: c_int = 0;
    let mut signval: c_int;
    let mut nmsg: c_int;
    let mut cval: c_char;
    // 3 rather than the C's 2, the room ffgbcl takes for a `P` prefix
    let mut cvalstr: [c_char; 3] = [0; 3];
    let mut repeat: c_long = 0;
    let mut offset: c_long = 0;
    let mut width: c_long = 0;
    let mut jnulval: c_long = 0;
    let mut anynull: c_int;

    let mut xinarray: [c_uchar; 21] = [0; 21];
    let mut binarray: [c_uchar; 21] = [0; 21];
    let mut boutarray: [c_uchar; 21] = [0; 21];
    let mut bnul: c_uchar;
    let mut iinarray: [c_short; 21] = [0; 21];
    let mut ioutarray: [c_short; 21] = [0; 21];
    let mut inul: c_short;
    let mut kinarray: [c_int; 21] = [0; 21];
    let mut koutarray: [c_int; 21] = [0; 21];
    let mut knul: c_int;
    let mut jinarray: [c_long; 21] = [0; 21];
    let mut joutarray: [c_long; 21] = [0; 21];
    let mut jnul: c_long;
    let mut einarray: [f32; 21] = [0.0; 21];
    let mut eoutarray: [f32; 21] = [0.0; 21];
    let mut enul: f32;
    let mut cinarray: [f32; 42] = [0.0; 42];
    let mut dinarray: [f64; 21] = [0.0; 21];
    let mut doutarray: [f64; 21] = [0.0; 21];
    let mut dnul: f64;
    let mut minarray: [f64; 42] = [0.0; 42];
    let mut scale: f64 = 0.0;
    let mut zero: f64 = 0.0;
    let mut naxes: [c_long; 3] = [0; 3];
    let mut pcount: c_long;
    let mut gcount: c_long;
    let mut npixels: c_long;
    let mut nrows: LONGLONG;
    let mut rowlen: c_long;
    let mut firstpix: [c_long; 3] = [0; 3];
    let mut existkeys: c_int = 0;
    let mut morekeys: c_int;
    let mut keynum: c_int = 0;

    let mut larray: [c_char; 42] = [0; 42];
    let mut larray2: [c_char; 42] = [0; 42];
    let mut colname: [c_char; 70] = [0; 70];
    let mut tdisp: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut nulstr: [c_char; 40] = [0; 40];

    let mut oskey: [c_char; 13] = [0; 13];
    oskey.copy_from_slice(cast_slice(c"value_string".to_bytes_with_nul()));

    let mut iskey: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut olkey: c_int = 1;
    let mut ilkey: c_int = 0;
    let oshtkey: c_short;
    let mut ishtkey: c_short = 0;
    let mut ojkey: c_long = 11;
    let mut ijkey: c_long = 0;
    let otint: c_long = 12345678;
    let ofkey: f32 = 12.121_212;
    let mut oekey: f32 = 13.131_313;
    let mut iekey: f32 = 0.0;

    #[allow(clippy::excessive_precision)]
    let ogkey: f64 = 14.141_414_141_414_141_4;
    #[allow(clippy::excessive_precision)]
    let mut odkey: f64 = 15.151_515_151_515_151_5;

    let mut idkey: f64 = 0.0;
    let otfrac: f64 = 0.123_456_789_012_345_6;

    let mut xrval: f64;
    let mut yrval: f64;
    let mut xrpix: f64;
    let mut yrpix: f64;
    let mut xinc: f64;
    let mut yinc: f64;
    let mut rot: f64;
    let mut xpos: f64 = 0.0;
    let mut ypos: f64 = 0.0;
    let mut xpix: f64;
    let mut ypix: f64;

    let mut xcoordtype: [c_char; 9] = [0; 9];
    xcoordtype.copy_from_slice(cast_slice(c"RA---TAN".to_bytes_with_nul()));

    let mut ycoordtype: [c_char; 9] = [0; 9];
    ycoordtype.copy_from_slice(cast_slice(c"DEC--TAN".to_bytes_with_nul()));

    let mut ctype: [c_char; 5] = [0; 5];

    let mut lsptr: *mut c_char = ptr::null_mut(); // pointer to long string value
    let mut comm: [c_char; 73] = [0; 73];
    let mut comms: [*mut c_char; 3] = [ptr::null_mut(); 3];
    let mut inskey: [*mut c_char; 21] = [ptr::null_mut(); 21];
    let onskey: [*const c_char; 3] = [
        c"first string".as_ptr(),
        c"second string".as_ptr(),
        c"        ".as_ptr(),
    ];
    let inclist: [*const c_char; 2] = [c"key*".as_ptr(), c"newikys".as_ptr()];
    let exclist: [*const c_char; 2] = [c"key_pr*".as_ptr(), c"key_pkls".as_ptr()];

    let onlkey: [c_int; 3] = [1, 0, 1];
    let mut inlkey: [c_int; 3] = [0; 3];
    let onjkey: [c_long; 3] = [11, 12, 13];
    let mut injkey: [c_long; 3] = [0; 3];
    let onfkey: [f32; 3] = [12.121_212, 13.131_313, 14.141_414];
    let onekey: [f32; 3] = [13.131_313, 14.141_414, 15.151_515];
    let mut inekey: [f32; 3] = [0.0; 3];
    let ongkey: [f64; 3] = [
        #[allow(clippy::excessive_precision)]
        14.141_414_141_414_141_4,
        #[allow(clippy::excessive_precision)]
        15.151_515_151_515_151_5,
        #[allow(clippy::excessive_precision)]
        16.161_616_161_616_161_6,
    ];
    let ondkey: [f64; 3] = [
        #[allow(clippy::excessive_precision)]
        15.151_515_151_515_151_5,
        #[allow(clippy::excessive_precision)]
        16.161_616_161_616_161_6,
        #[allow(clippy::excessive_precision)]
        17.171_717_171_717_171_7,
    ];
    let mut indkey: [f64; 3] = [0.0; 3];

    let mut tbcol: [c_long; 5] = [1, 17, 28, 43, 56];

    let mut filename: [c_char; 40] = [0; 40];
    let mut card: [c_char; FLEN_CARD] = [0; FLEN_CARD];

    let mut card2: [c_char; FLEN_CARD] = [0; FLEN_CARD];
    let mut keyword: [c_char; FLEN_KEYWORD] = [0; FLEN_KEYWORD];
    let mut value: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    let mut comment: [c_char; FLEN_COMMENT] = [0; FLEN_COMMENT];
    let mut uchars: [c_uchar; 80] = [0; 80];

    // fitsfile *fptr, *tmpfptr;
    let mut fptr: Option<Box<fitsfile>> = None;
    let mut tmpfptr: Option<Box<fitsfile>> = None;

    // char *ttype[10], *tform[10], *tunit[10];
    let mut ttype: [*mut c_char; 10] = [ptr::null_mut(); 10];
    let mut tform: [*mut c_char; 10] = [ptr::null_mut(); 10];
    let mut tunit: [*mut c_char; 10] = [ptr::null_mut(); 10];

    let mut tblname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];

    let mut binname: [c_char; FLEN_VALUE] = [0; FLEN_VALUE];
    binname[..14].copy_from_slice(cast_slice(c"Test-BINTABLE".to_bytes_with_nul()));

    let mut templt: [c_char; 13] = [0; 13];
    templt.copy_from_slice(cast_slice(c"testprog.tpt".to_bytes_with_nul()));

    let mut errmsg: [c_char; FLEN_ERRMSG] = [0; FLEN_ERRMSG];
    let mut imgarray: [[c_short; 19]; 30] = [[0; 19]; 30];
    let mut imgarray2: [[c_short; 10]; 20] = [[0; 10]; 20];
    let mut fpixels: [c_long; 2] = [0; 2];
    let mut lpixels: [c_long; 2] = [0; 2];
    let mut inc: [c_long; 2] = [0; 2];

    unsafe {
        // Don't have goto statements, so emulate with a loop that always breaks
        #[allow(clippy::never_loop)]
        'mainloop: loop {
            strcpy_cstr(&mut tblname, c"Test-ASCII");

            println!("CFITSIO TESTPROG\n");
            println!("Try opening then closing a nonexistent file:");
            fits_open_file(
                &raw mut fptr,
                c"tq123x.kjl".as_ptr(),
                READWRITE,
                &raw mut status,
            );
            println!(
                "  ffopen fptr, status  = {} {} (expect an error)",
                if fptr.is_none() { 0 } else { 1 } as c_ulong,
                status
            );

            let f = fptr.take();
            ffclos(f, &raw mut status);
            println!("  ffclos status = {status}\n");
            ffcmsg();
            //status = 0;

            for ii in 0..21 {
                /* allocate space for string column value */
                inskey[ii] = malloc(FLEN_VALUE).cast::<c_char>();
            }

            for ii in 0..10 {
                ttype[ii] = malloc(FLEN_VALUE).cast::<c_char>();
                tform[ii] = malloc(FLEN_VALUE).cast::<c_char>();
                tunit[ii] = malloc(FLEN_VALUE).cast::<c_char>();
            }

            comms[0] = comm.as_mut_ptr();

            /* delete previous version of the file, if it exists (with ! prefix) */
            strcpy_safe(&mut filename, cs!(c"!testprog.fit"));

            status = 0;

            /*
              #####################
              #  create FITS file #
              #####################
            */

            ffinit(&raw mut fptr, filename.as_ptr(), &raw mut status);
            println!("ffinit create new file status = {status}");
            if status != 0 {
                break 'mainloop;
            }

            filename[0] = 0;
            ffflnm(fptr.as_mut_ptr(), filename.as_mut_ptr(), &raw mut status);

            ffflmd(fptr.as_mut_ptr(), &raw mut filemode, &raw mut status);
            println!(
                "Name of file = {}, I/O mode = {}",
                byte_slice_to_str!(&filename),
                filemode
            );

            //simple = 1;
            bitpix = 32;
            naxis = 2;
            naxes[0] = 10;
            naxes[1] = 2;
            npixels = 20;
            //pcount = 0;
            //gcount = 1;
            //extend = 1;

            /*
              ############################
              #  write single keywords   #
              ############################
            */

            if ffphps(
                fptr.as_mut_ptr(),
                bitpix,
                naxis,
                naxes.as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffphps status = {status}");
            }

            if ffprec(
                fptr.as_mut_ptr(),
                c"key_prec= 'This keyword was written by fxprec' / comment goes here".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffprec status = {status}");
            }

            print!("\ntest writing of long string keywords:\n");
            strcpy_safe(&mut card, cs!(c"1234567890123456789012345678901234567890"));
            strcat_safe(&mut card, cs!(c"12345678901234567890123456789012345"));
            ffpkys(
                fptr.as_mut_ptr(),
                c"card1".as_ptr(),
                card.as_ptr(),
                c"".as_ptr(),
                &raw mut status,
            );
            ffgkey(
                fptr.as_mut_ptr(),
                c"card1".as_ptr(),
                card2.as_mut_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            print!(
                " {}\n{}\n",
                byte_slice_to_str!(&card),
                byte_slice_to_str!(&card2)
            );

            strcpy_safe(&mut card, cs!(c"1234567890123456789012345678901234567890"));
            strcat_safe(&mut card, cs!(c"123456789012345678901234'6789012345"));
            ffpkys(
                fptr.as_mut_ptr(),
                c"card2".as_ptr(),
                card.as_ptr(),
                c"".as_ptr(),
                &raw mut status,
            );
            ffgkey(
                fptr.as_mut_ptr(),
                c"card2".as_ptr(),
                card2.as_mut_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            print!(
                " {}\n{}\n",
                byte_slice_to_str!(&card),
                byte_slice_to_str!(&card2)
            );

            strcpy_safe(&mut card, cs!(c"1234567890123456789012345678901234567890"));
            strcat_safe(&mut card, cs!(c"123456789012345678901234''789012345"));
            ffpkys(
                fptr.as_mut_ptr(),
                c"card3".as_ptr(),
                card.as_ptr(),
                c"".as_ptr(),
                &raw mut status,
            );
            ffgkey(
                fptr.as_mut_ptr(),
                c"card3".as_ptr(),
                card2.as_mut_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            print!(
                " {}\n{}\n",
                byte_slice_to_str!(&card),
                byte_slice_to_str!(&card2)
            );

            strcpy_safe(&mut card, cs!(c"1234567890123456789012345678901234567890"));
            strcat_safe(&mut card, cs!(c"123456789012345678901234567'9012345"));
            ffpkys(
                fptr.as_mut_ptr(),
                c"card4".as_ptr(),
                card.as_ptr(),
                c"".as_ptr(),
                &raw mut status,
            );
            ffgkey(
                fptr.as_mut_ptr(),
                c"card4".as_ptr(),
                card2.as_mut_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            print!(
                " {}\n{}\n",
                byte_slice_to_str!(&card),
                byte_slice_to_str!(&card2)
            );

            if ffpkys(
                fptr.as_mut_ptr(),
                c"key_pkys".as_ptr(),
                oskey.as_ptr(),
                c"fxpkys comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkys status = {status}");
            }

            if ffpkyl(
                fptr.as_mut_ptr(),
                c"key_pkyl".as_ptr(),
                olkey,
                c"fxpkyl comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkyl status = {status}");
            }

            if ffpkyj(
                fptr.as_mut_ptr(),
                c"key_pkyj".as_ptr(),
                ojkey as LONGLONG,
                c"fxpkyj comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkyj status = {status}");
            }

            if ffpkyf(
                fptr.as_mut_ptr(),
                c"key_pkyf".as_ptr(),
                ofkey,
                5,
                c"fxpkyf comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkyf status = {status}");
            }

            if ffpkye(
                fptr.as_mut_ptr(),
                c"key_pkye".as_ptr(),
                oekey,
                6,
                c"fxpkye comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkye status = {status}");
            }

            if ffpkyg(
                fptr.as_mut_ptr(),
                c"key_pkyg".as_ptr(),
                ogkey,
                14,
                c"fxpkyg comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkyg status = {status}");
            }

            if ffpkyd(
                fptr.as_mut_ptr(),
                c"key_pkyd".as_ptr(),
                odkey,
                14,
                c"fxpkyd comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkyd status = {status}");
            }

            if ffpkyc(
                fptr.as_mut_ptr(),
                c"key_pkyc".as_ptr(),
                onekey[..2].as_ptr().cast::<[f32; 2]>(),
                6,
                c"fxpkyc comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkyc status = {status}");
            }

            if ffpkym(
                fptr.as_mut_ptr(),
                c"key_pkym".as_ptr(),
                ondkey[..2].as_ptr().cast::<[f64; 2]>(),
                14,
                c"fxpkym comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkym status = {status}");
            }

            if ffpkfc(
                fptr.as_mut_ptr(),
                c"key_pkfc".as_ptr(),
                onekey[..2].as_ptr().cast::<[f32; 2]>(),
                6,
                c"fxpkfc comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkfc status = {status}");
            }

            if ffpkfm(
                fptr.as_mut_ptr(),
                c"key_pkfm".as_ptr(),
                ondkey[..2].as_ptr().cast::<[f64; 2]>(),
                14,
                c"fxpkfm comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkfm status = {status}");
            }

            if ffpkls(
                fptr.as_mut_ptr(),
                c"key_pkls".as_ptr(),
                c"This is a very long string value that is continued over more than one keyword."
                    .as_ptr(),
                c"fxpkls comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkls status = {status}");
            }

            if ffplsw(fptr.as_mut_ptr(), &raw mut status) > 0 {
                println!("ffplsw status = {status}");
            }

            if ffpkyt(
                fptr.as_mut_ptr(),
                c"key_pkyt".as_ptr(),
                otint,
                otfrac,
                c"fxpkyt comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkyt status = {status}");
            }

            if ffpcom(
                fptr.as_mut_ptr(),
                c"  This keyword was written by fxpcom.".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpcom status = {status}");
            }

            if ffphis(
                fptr.as_mut_ptr(),
                c"    This keyword written by fxphis (w/ 2 leading spaces).".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffphis status = {status}");
            }

            if ffpdat(fptr.as_mut_ptr(), &raw mut status) > 0 {
                println!("ffpdat status = {status}");
                break 'mainloop;
            }

            /*
              ###############################
              #  write arrays of keywords   #
              ###############################
            */
            nkeys = 3;

            /* use the inskey array of pointers for the comments */

            strcpy(comms[0], c"fxpkns comment&".as_ptr());
            if ffpkns(
                fptr.as_mut_ptr(),
                c"ky_pkns".as_ptr(),
                1,
                nkeys,
                onskey.as_ptr(),
                comms.as_ptr().cast::<*const c_char>(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkns status = {status}");
            }

            strcpy(comms[0], c"fxpknl comment&".as_ptr());
            if ffpknl(
                fptr.as_mut_ptr(),
                c"ky_pknl".as_ptr(),
                1,
                nkeys,
                onlkey.as_ptr(),
                comms.as_ptr().cast::<*const c_char>(),
                &raw mut status,
            ) > 0
            {
                println!("ffpknl status = {status}");
            }

            strcpy(comms[0], c"fxpknj comment&".as_ptr());
            if ffpknj(
                fptr.as_mut_ptr(),
                c"ky_pknj".as_ptr(),
                1,
                nkeys,
                onjkey.as_ptr(),
                comms.as_ptr().cast::<*const c_char>(),
                &raw mut status,
            ) > 0
            {
                println!("ffpknj status = {status}");
            }

            strcpy(comms[0], c"fxpknf comment&".as_ptr());
            if ffpknf(
                fptr.as_mut_ptr(),
                c"ky_pknf".as_ptr(),
                1,
                nkeys,
                onfkey.as_ptr(),
                5,
                comms.as_ptr().cast::<*const c_char>(),
                &raw mut status,
            ) > 0
            {
                println!("ffpknf status = {status}");
            }

            strcpy(comms[0], c"fxpkne comment&".as_ptr());
            if ffpkne(
                fptr.as_mut_ptr(),
                c"ky_pkne".as_ptr(),
                1,
                nkeys,
                onekey.as_ptr(),
                6,
                comms.as_ptr().cast::<*const c_char>(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkne status = {status}");
            }

            strcpy(comms[0], c"fxpkng comment&".as_ptr());
            if ffpkng(
                fptr.as_mut_ptr(),
                c"ky_pkng".as_ptr(),
                1,
                nkeys,
                ongkey.as_ptr(),
                13,
                comms.as_ptr().cast::<*const c_char>(),
                &raw mut status,
            ) > 0
            {
                println!("ffpkng status = {status}");
            }

            strcpy(comms[0], c"fxpknd comment&".as_ptr());
            if ffpknd(
                fptr.as_mut_ptr(),
                c"ky_pknd".as_ptr(),
                1,
                nkeys,
                ondkey.as_ptr(),
                14,
                comms.as_ptr().cast::<*const c_char>(),
                &raw mut status,
            ) > 0
            {
                println!("ffpknd status = {status}");
                break 'mainloop;
            }
            /*
              ############################
              #  write generic keywords  #
              ############################
            */

            strcpy_safe(&mut oskey, cs!(c"1"));
            if ffpky(
                fptr.as_mut_ptr(),
                TSTRING,
                c"tstring".as_ptr(),
                oskey.as_ptr().cast::<c_void>(),
                c"tstring comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpky status = {status}");
            }

            olkey = TLOGICAL;
            if ffpky(
                fptr.as_mut_ptr(),
                TLOGICAL,
                c"tlogical".as_ptr(),
                core::ptr::from_ref(&olkey).cast::<c_void>(),
                c"tlogical comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpky status = {status}");
            }

            cval = TBYTE as c_char;
            if ffpky(
                fptr.as_mut_ptr(),
                TBYTE,
                c"tbyte".as_ptr(),
                core::ptr::from_ref(&cval).cast::<c_void>(),
                c"tbyte comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpky status = {status}");
            }

            oshtkey = TSHORT as c_short;
            if ffpky(
                fptr.as_mut_ptr(),
                TSHORT,
                c"tshort".as_ptr(),
                core::ptr::from_ref(&oshtkey).cast::<c_void>(),
                c"tshort comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpky status = {status}");
            }

            olkey = TINT;
            if ffpky(
                fptr.as_mut_ptr(),
                TINT,
                c"tint".as_ptr(),
                core::ptr::from_ref(&olkey).cast::<c_void>(),
                c"tint comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpky status = {status}");
            }

            ojkey = TLONG as c_long;
            if ffpky(
                fptr.as_mut_ptr(),
                TLONG,
                c"tlong".as_ptr(),
                core::ptr::from_ref(&ojkey).cast::<c_void>(),
                c"tlong comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpky status = {status}");
            }

            oekey = TFLOAT as f32;
            if ffpky(
                fptr.as_mut_ptr(),
                TFLOAT,
                c"tfloat".as_ptr(),
                core::ptr::from_ref(&oekey).cast::<c_void>(),
                c"tfloat comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpky status = {status}");
            }

            odkey = TDOUBLE as f64;
            if ffpky(
                fptr.as_mut_ptr(),
                TDOUBLE,
                c"tdouble".as_ptr(),
                core::ptr::from_ref(&odkey).cast::<c_void>(),
                c"tdouble comment".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("ffpky status = {status}");
            }

            /*
              ############################
              #  write data              #
              ############################
            */
            /* define the null value (must do this before writing any data) */
            if ffpkyj(
                fptr.as_mut_ptr(),
                c"BLANK".as_ptr(),
                -99,
                c"value to use for undefined pixels".as_ptr(),
                &raw mut status,
            ) > 0
            {
                println!("BLANK keyword status = {status}");
            }

            /* initialize arrays of values to write to primary array */
            for ii in 0..(npixels as usize) {
                boutarray[ii] = (ii + 1) as c_uchar;
                ioutarray[ii] = (ii + 1) as c_short;
                joutarray[ii] = (ii + 1) as c_long;
                eoutarray[ii] = (ii + 1) as f32;
                doutarray[ii] = (ii + 1) as f64;
            }

            /* write a few pixels with each datatype */
            /* set the last value in each group of 4 as undefined */

            /*
                ffpprb(fptr.as_mut_ptr(), 1,  1, 2, &boutarray[0],  &mut status);
                ffppri(fptr.as_mut_ptr(), 1,  5, 2, &ioutarray[4],  &mut status);
                ffpprj(fptr.as_mut_ptr(), 1,  9, 2, &joutarray[8],  &mut status);
                ffppre(fptr.as_mut_ptr(), 1, 13, 2, &eoutarray[12], &mut status);
                ffpprd(fptr.as_mut_ptr(), 1, 17, 2, &doutarray[16], &mut status);
            */

            /*  test the newer ffpx routine, instead of the older ffppr_ routines */
            firstpix[0] = 1;
            firstpix[1] = 1;
            ffppx(
                fptr.as_mut_ptr(),
                TBYTE,
                firstpix.as_ptr(),
                2,
                boutarray[0..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            firstpix[0] = 5;
            ffppx(
                fptr.as_mut_ptr(),
                TSHORT,
                firstpix.as_ptr(),
                2,
                ioutarray[4..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            firstpix[0] = 9;
            ffppx(
                fptr.as_mut_ptr(),
                TLONG,
                firstpix.as_ptr(),
                2,
                joutarray[8..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            firstpix[0] = 3;
            firstpix[1] = 2;
            ffppx(
                fptr.as_mut_ptr(),
                TFLOAT,
                firstpix.as_ptr(),
                2,
                eoutarray[12..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            firstpix[0] = 7;
            ffppx(
                fptr.as_mut_ptr(),
                TDOUBLE,
                firstpix.as_ptr(),
                2,
                doutarray[16..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );

            /*
                ffppnb(fptr.as_mut_ptr(), 1,  3, 2, &boutarray[2],   4, &mut status);
                ffppni(fptr.as_mut_ptr(), 1,  7, 2, &ioutarray[6],   8, &mut status);
                ffppnj(fptr.as_mut_ptr(), 1, 11, 2, &joutarray[10],  12, &mut status);
                ffppne(fptr.as_mut_ptr(), 1, 15, 2, &eoutarray[14], 16., &mut status);
                ffppnd(fptr.as_mut_ptr(), 1, 19, 2, &doutarray[18], 20., &mut status);
            */
            firstpix[0] = 3;
            firstpix[1] = 1;
            bnul = 4;
            ffppxn(
                fptr.as_mut_ptr(),
                TBYTE,
                firstpix.as_mut_ptr(),
                2,
                boutarray[2..].as_ptr().cast::<c_void>(),
                core::ptr::from_ref(&bnul).cast::<c_void>(),
                &raw mut status,
            );
            firstpix[0] = 7;
            inul = 8;
            ffppxn(
                fptr.as_mut_ptr(),
                TSHORT,
                firstpix.as_mut_ptr(),
                2,
                ioutarray[6..].as_ptr().cast::<c_void>(),
                core::ptr::from_ref(&inul).cast::<c_void>(),
                &raw mut status,
            );
            firstpix[0] = 1;
            firstpix[1] = 2;
            jnul = 12;
            ffppxn(
                fptr.as_mut_ptr(),
                TLONG,
                firstpix.as_mut_ptr(),
                2,
                joutarray[10..].as_ptr().cast::<c_void>(),
                core::ptr::from_ref(&jnul).cast::<c_void>(),
                &raw mut status,
            );
            firstpix[0] = 5;
            enul = 16.0;
            ffppxn(
                fptr.as_mut_ptr(),
                TFLOAT,
                firstpix.as_mut_ptr(),
                2,
                eoutarray[14..].as_ptr().cast::<c_void>(),
                core::ptr::from_ref(&enul).cast::<c_void>(),
                &raw mut status,
            );
            firstpix[0] = 9;
            dnul = 20.0;
            ffppxn(
                fptr.as_mut_ptr(),
                TDOUBLE,
                firstpix.as_mut_ptr(),
                2,
                doutarray[18..].as_ptr().cast::<c_void>(),
                core::ptr::from_ref(&dnul).cast::<c_void>(),
                &raw mut status,
            );

            ffppru(fptr.as_mut_ptr(), 1, 1, 1, &raw mut status);

            if status > 0 {
                println!("ffppnx status = {status}");
                break 'mainloop;
            }

            ffflus(fptr.as_mut_ptr(), &raw mut status); /* flush all data to the disk file */
            println!("ffflus status = {status}");
            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            /*
              ############################
              #  read data               #
              ############################
            */
            /* read back the data, setting null values = 99 */
            print!("\nValues read back from primary array (99 = null pixel)\n");
            println!("The 1st, and every 4th pixel should be undefined:");

            anynull = 0;
            ffgpvb(
                fptr.as_mut_ptr(),
                1,
                1,
                10,
                99,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            ffgpvb(
                fptr.as_mut_ptr(),
                1,
                11,
                10,
                99,
                binarray[10..].as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for item in &binarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (ffgpvb)");

            ffgpvi(
                fptr.as_mut_ptr(),
                1,
                1,
                npixels as LONGLONG,
                99,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for item in &iinarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (ffgpvi)");

            ffgpvj(
                fptr.as_mut_ptr(),
                1,
                1,
                npixels as LONGLONG,
                99,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for item in &jinarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (ffgpvj)");

            ffgpve(
                fptr.as_mut_ptr(),
                1,
                1,
                npixels as LONGLONG,
                99.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for item in &einarray[..npixels as usize] {
                print!(" {item:2.0}");
            }
            println!("  {anynull} (ffgpve)");

            ffgpvd(
                fptr.as_mut_ptr(),
                1,
                1,
                10,
                99.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgpvd(
                fptr.as_mut_ptr(),
                1,
                11,
                10,
                99.,
                dinarray[10..].as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for item in &dinarray[..npixels as usize] {
                print!(" {item:2.0}");
            }
            println!("  {anynull} (ffgpvd)");

            if status > 0 {
                println!("ERROR: ffgpv_ status = {status}");
                break 'mainloop;
            }
            if anynull == 0 {
                println!("ERROR: ffgpv_ did not detect null values");
            }

            /* reset the output null value to the expected input value */
            for ii in (3..npixels as usize).step_by(4) {
                boutarray[ii] = 99;
                ioutarray[ii] = 99;
                joutarray[ii] = 99;
                eoutarray[ii] = 99.0;
                doutarray[ii] = 99.0;
            }

            boutarray[0] = 99;
            ioutarray[0] = 99;
            joutarray[0] = 99;
            eoutarray[0] = 99.0;
            doutarray[0] = 99.0;

            /* compare the output with the input; flag any differences */
            for ii in 0..npixels as usize {
                if boutarray[ii] != binarray[ii] {
                    println!("bout != bin = {} {} ", boutarray[ii], binarray[ii]);
                }

                if ioutarray[ii] != iinarray[ii] {
                    println!("iout != iin = {} {} ", ioutarray[ii], iinarray[ii]);
                }

                if joutarray[ii] != jinarray[ii] {
                    println!("jout != jin = {} {} ", joutarray[ii], jinarray[ii]);
                }

                if eoutarray[ii] != einarray[ii] {
                    println!("eout != ein = {:.6} {:.6} ", eoutarray[ii], einarray[ii]);
                }

                if doutarray[ii] != dinarray[ii] {
                    println!("dout != din = {:.6} {:.6} ", doutarray[ii], dinarray[ii]);
                }
            }

            for ii in 0..npixels as usize {
                binarray[ii] = 0;
                iinarray[ii] = 0;
                jinarray[ii] = 0;
                einarray[ii] = 0.0;
                dinarray[ii] = 0.0;
            }

            anynull = 0;
            ffgpfb(
                fptr.as_mut_ptr(),
                1,
                1,
                10,
                binarray.as_mut_ptr(),
                larray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgpfb(
                fptr.as_mut_ptr(),
                1,
                11,
                10,
                binarray[10..].as_mut_ptr(),
                larray[10..].as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for ii in 0..npixels as usize {
                if larray[ii] != 0 {
                    print!("  *");
                } else {
                    print!(" {:>2}", binarray[ii]);
                }
            }
            println!("  {anynull} (ffgpfb)");

            ffgpfi(
                fptr.as_mut_ptr(),
                1,
                1,
                npixels as LONGLONG,
                iinarray.as_mut_ptr(),
                larray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for ii in 0..npixels as usize {
                if larray[ii] != 0 {
                    print!("  *");
                } else {
                    print!(" {:>2}", iinarray[ii]);
                }
            }
            println!("  {anynull} (ffgpfi)");

            ffgpfj(
                fptr.as_mut_ptr(),
                1,
                1,
                npixels as LONGLONG,
                jinarray.as_mut_ptr(),
                larray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for ii in 0..npixels as usize {
                if larray[ii] != 0 {
                    print!("  *");
                } else {
                    print!(" {:>2}", jinarray[ii]);
                }
            }
            println!("  {anynull} (ffgpfj)");

            ffgpfe(
                fptr.as_mut_ptr(),
                1,
                1,
                npixels as LONGLONG,
                einarray.as_mut_ptr(),
                larray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for ii in 0..npixels as usize {
                if larray[ii] != 0 {
                    print!("  *");
                } else {
                    print!(" {:2.0}", einarray[ii]);
                }
            }
            println!("  {anynull} (ffgpfe)");

            ffgpfd(
                fptr.as_mut_ptr(),
                1,
                1,
                10,
                dinarray.as_mut_ptr(),
                larray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgpfd(
                fptr.as_mut_ptr(),
                1,
                11,
                10,
                dinarray[10..].as_mut_ptr(),
                larray[10..].as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            for ii in 0..npixels as usize {
                if larray[ii] != 0 {
                    print!("  *");
                } else {
                    print!(" {:2.0}", dinarray[ii]);
                }
            }
            println!("  {anynull} (ffgpfd)");

            if status > 0 {
                println!("ERROR: ffgpf_ status = {status}");
                break 'mainloop;
            }
            if anynull == 0 {
                println!("ERROR: ffgpf_ did not detect null values");
            }

            /*
              ##########################################
              #  close and reopen file multiple times  #
              ##########################################
            */

            for _ii in 0..10 {
                let f = fptr.take();
                if ffclos(f, &raw mut status) > 0 {
                    print!("ERROR in ftclos (1) = {status}");
                    break 'mainloop;
                }

                if fits_open_file(&raw mut fptr, filename.as_ptr(), READWRITE, &raw mut status) > 0
                {
                    println!("ERROR: ffopen open file status = {status}");
                    break 'mainloop;
                }
            }
            print!("\nClosed then reopened the FITS file 10 times.\n");
            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            filename[0] = 0;
            ffflnm(fptr.as_mut_ptr(), filename.as_mut_ptr(), &raw mut status);

            ffflmd(fptr.as_mut_ptr(), &raw mut filemode, &raw mut status);
            println!(
                "Name of file = {}, I/O mode = {}",
                byte_slice_to_str!(&filename),
                filemode
            );

            /*
              ############################
              #  read single keywords    #
              ############################
            */

            simple = 0;
            bitpix = 0;
            naxis = 0;
            naxes[0] = 0;
            naxes[1] = 0;
            pcount = -99;
            gcount = -99;
            extend = -99;
            print!("\nRead back keywords:\n");
            ffghpr(
                fptr.as_mut_ptr(),
                99,
                &raw mut simple,
                &raw mut bitpix,
                &raw mut naxis,
                naxes.as_mut_ptr(),
                &raw mut pcount,
                &raw mut gcount,
                &raw mut extend,
                &raw mut status,
            );
            println!(
                "simple = {}, bitpix = {}, naxis = {}, naxes = ({}, {})",
                simple, bitpix, naxis, naxes[0], naxes[1]
            );
            println!("  pcount = {pcount}, gcount = {gcount}, extend = {extend}");

            ffgrec(fptr.as_mut_ptr(), 9, card.as_mut_ptr(), &raw mut status);
            println!("{}", byte_slice_to_str!(&card));
            if strncmp_safe(&card, cs!(c"KEY_PREC= 'This"), 15) != 0 {
                println!("ERROR in ffgrec");
            }

            ffgkyn(
                fptr.as_mut_ptr(),
                9,
                keyword.as_mut_ptr(),
                value.as_mut_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "{} : {} : {} :",
                byte_slice_to_str!(&keyword),
                byte_slice_to_str!(&value),
                byte_slice_to_str!(&comment)
            );
            if strncmp_safe(&keyword, cs!(c"KEY_PREC"), 8) != 0 {
                println!("ERROR in ffgkyn: {}", byte_slice_to_str!(&keyword));
            }

            ffgcrd(
                fptr.as_mut_ptr(),
                keyword.as_ptr(),
                card.as_mut_ptr(),
                &raw mut status,
            );
            println!("{}", byte_slice_to_str!(&card));

            if strncmp_safe(&keyword, &card, 8) != 0 {
                println!("ERROR in ffgcrd: {}", byte_slice_to_str!(&keyword));
            }

            ffgkey(
                fptr.as_mut_ptr(),
                c"KY_PKNS1".as_ptr(),
                value.as_mut_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KY_PKNS1 : {} : {} :",
                byte_slice_to_str!(&value),
                byte_slice_to_str!(&comment)
            );

            if strncmp_safe(&value, cs!(c"'first string'"), 14) != 0 {
                println!("ERROR in ffgkey: {}", byte_slice_to_str!(&value));
            }

            ffgkys(
                fptr.as_mut_ptr(),
                c"key_pkys".as_ptr(),
                iskey.as_mut_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYS {} {} {}",
                byte_slice_to_str!(&iskey),
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyl(
                fptr.as_mut_ptr(),
                c"key_pkyl".as_ptr(),
                &raw mut ilkey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYL {} {} {}",
                ilkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyj(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                &raw mut ijkey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYJ {} {} {}",
                ijkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgkye(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                &raw mut iekey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYJ {:.6} {} {}",
                iekey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyd(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                &raw mut idkey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYJ {:.6} {} {}",
                idkey,
                byte_slice_to_str!(&comment),
                status
            );

            if ijkey != 11 || iekey != 11. || idkey != 11. {
                println!("ERROR in ffgky[jed]: {ijkey}, {iekey:.6}, {idkey:.6}");
            }

            iskey[0] = 0;
            ffgky(
                fptr.as_mut_ptr(),
                TSTRING,
                c"key_pkys".as_ptr(),
                iskey.as_mut_ptr().cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY S {} {} {}",
                byte_slice_to_str!(&iskey),
                byte_slice_to_str!(&comment),
                status
            );

            ilkey = 0;
            ffgky(
                fptr.as_mut_ptr(),
                TLOGICAL,
                c"key_pkyl".as_ptr(),
                core::ptr::from_mut(&mut ilkey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY L {} {} {}",
                ilkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgky(
                fptr.as_mut_ptr(),
                TBYTE,
                c"KEY_PKYJ".as_ptr(),
                core::ptr::from_mut(&mut cval).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY BYTE {} {} {}",
                cval,
                byte_slice_to_str!(&comment),
                status
            );

            ffgky(
                fptr.as_mut_ptr(),
                TSHORT,
                c"KEY_PKYJ".as_ptr(),
                core::ptr::from_mut(&mut ishtkey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY SHORT {} {} {}",
                ishtkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgky(
                fptr.as_mut_ptr(),
                TINT,
                c"KEY_PKYJ".as_ptr(),
                core::ptr::from_mut(&mut ilkey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY INT {} {} {}",
                ilkey,
                byte_slice_to_str!(&comment),
                status
            );

            ijkey = 0;
            ffgky(
                fptr.as_mut_ptr(),
                TLONG,
                c"KEY_PKYJ".as_ptr(),
                core::ptr::from_mut(&mut ijkey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY J {} {} {}",
                ijkey,
                byte_slice_to_str!(&comment),
                status
            );

            iekey = 0.0;
            ffgky(
                fptr.as_mut_ptr(),
                TFLOAT,
                c"KEY_PKYE".as_ptr(),
                core::ptr::from_mut(&mut iekey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY E {:.6} {} {}",
                iekey,
                byte_slice_to_str!(&comment),
                status
            );

            idkey = 0.0;
            ffgky(
                fptr.as_mut_ptr(),
                TDOUBLE,
                c"KEY_PKYD".as_ptr(),
                core::ptr::from_mut(&mut idkey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY D {:.6} {} {}",
                idkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyd(
                fptr.as_mut_ptr(),
                c"KEY_PKYF".as_ptr(),
                &raw mut idkey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYF {:.6} {} {}",
                idkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyd(
                fptr.as_mut_ptr(),
                c"KEY_PKYE".as_ptr(),
                &raw mut idkey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYE {:.6} {} {}",
                idkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyd(
                fptr.as_mut_ptr(),
                c"KEY_PKYG".as_ptr(),
                &raw mut idkey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYG {:.14} {} {}",
                idkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyd(
                fptr.as_mut_ptr(),
                c"KEY_PKYD".as_ptr(),
                &raw mut idkey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYD {:.14} {} {}",
                idkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyc(
                fptr.as_mut_ptr(),
                c"KEY_PKYC".as_ptr(),
                inekey.as_mut_ptr().cast::<[f32; 2]>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYC {:.6} {:.6} {} {}",
                inekey[0],
                inekey[1],
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyc(
                fptr.as_mut_ptr(),
                c"KEY_PKFC".as_ptr(),
                inekey.as_mut_ptr().cast::<[f32; 2]>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKFC {:.6} {:.6} {} {}",
                inekey[0],
                inekey[1],
                byte_slice_to_str!(&comment),
                status
            );

            ffgkym(
                fptr.as_mut_ptr(),
                c"KEY_PKYM".as_ptr(),
                indkey[..2].as_mut_ptr().cast::<[f64; 2]>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYM {:.6} {:.6} {} {}",
                indkey[0],
                indkey[1],
                byte_slice_to_str!(&comment),
                status
            );

            ffgkym(
                fptr.as_mut_ptr(),
                c"KEY_PKFM".as_ptr(),
                indkey[..2].as_mut_ptr().cast::<[f64; 2]>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKFM {:.6} {:.6} {} {}",
                indkey[0],
                indkey[1],
                byte_slice_to_str!(&comment),
                status
            );

            ffgkyt(
                fptr.as_mut_ptr(),
                c"KEY_PKYT".as_ptr(),
                &raw mut ijkey,
                &raw mut idkey,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKYT {} {:.14} {} {}",
                ijkey,
                idkey,
                byte_slice_to_str!(&comment),
                status
            );

            ffpunt(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                c"km/s/Mpc".as_ptr(),
                &raw mut status,
            );
            ijkey = 0;
            ffgky(
                fptr.as_mut_ptr(),
                TLONG,
                c"KEY_PKYJ".as_ptr(),
                core::ptr::from_mut(&mut ijkey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY J {} {} {}",
                ijkey,
                byte_slice_to_str!(&comment),
                status
            );
            ffgunt(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!("KEY_PKY units = {}", byte_slice_to_str!(&comment));

            ffpunt(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                c"".as_ptr(),
                &raw mut status,
            );
            ijkey = 0;
            ffgky(
                fptr.as_mut_ptr(),
                TLONG,
                c"KEY_PKYJ".as_ptr(),
                core::ptr::from_mut(&mut ijkey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY J {} {} {}",
                ijkey,
                byte_slice_to_str!(&comment),
                status
            );
            ffgunt(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!("KEY_PKY units = {}", byte_slice_to_str!(&comment));

            ffpunt(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                c"feet/second/second".as_ptr(),
                &raw mut status,
            );
            ijkey = 0;
            ffgky(
                fptr.as_mut_ptr(),
                TLONG,
                c"KEY_PKYJ".as_ptr(),
                core::ptr::from_mut(&mut ijkey).cast::<c_void>(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "KEY_PKY J {} {} {}",
                ijkey,
                byte_slice_to_str!(&comment),
                status
            );
            ffgunt(
                fptr.as_mut_ptr(),
                c"KEY_PKYJ".as_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!("KEY_PKY units = {}", byte_slice_to_str!(&comment));

            ffgkls(
                fptr.as_mut_ptr(),
                c"key_pkls".as_ptr(),
                &raw mut lsptr,
                comment.as_mut_ptr(),
                &raw mut status,
            );
            print!(
                "KEY_PKLS long string value = \n{}\n",
                CStr::from_ptr(lsptr).to_str().unwrap()
            );

            /* free the memory for the long string value */
            fffree(lsptr.cast::<c_void>(), &raw mut status);

            /* get size and position in header */
            ffghps(
                fptr.as_mut_ptr(),
                &raw mut existkeys,
                &raw mut keynum,
                &raw mut status,
            );
            println!("header contains {existkeys} keywords; located at keyword {keynum} ");

            /*
              ############################
              #  read array keywords     #
              ############################
            */
            ffgkns(
                fptr.as_mut_ptr(),
                c"ky_pkns".as_ptr(),
                1,
                3,
                inskey.as_mut_ptr(),
                &raw mut nfound,
                &raw mut status,
            );
            println!(
                "ffgkns:  {}, {}, {}",
                CStr::from_ptr(inskey[0]).to_str().unwrap(),
                CStr::from_ptr(inskey[1]).to_str().unwrap(),
                CStr::from_ptr(inskey[2]).to_str().unwrap()
            );
            if nfound != 3 || status > 0 {
                print!("\nERROR in ffgkns {nfound}, {status}\n");
            }

            ffgknl(
                fptr.as_mut_ptr(),
                c"ky_pknl".as_ptr(),
                1,
                3,
                inlkey.as_mut_ptr(),
                &raw mut nfound,
                &raw mut status,
            );
            println!("ffgknl:  {}, {}, {}", inlkey[0], inlkey[1], inlkey[2]);
            if nfound != 3 || status > 0 {
                print!("\nERROR in ffgknl {nfound}, {status}\n");
            }

            ffgknj(
                fptr.as_mut_ptr(),
                c"ky_pknj".as_ptr(),
                1,
                3,
                injkey.as_mut_ptr(),
                &raw mut nfound,
                &raw mut status,
            );
            println!("ffgknj:  {}, {}, {}", injkey[0], injkey[1], injkey[2]);
            if nfound != 3 || status > 0 {
                print!("\nERROR in ffgknj {nfound}, {status}\n");
            }

            ffgkne(
                fptr.as_mut_ptr(),
                c"ky_pkne".as_ptr(),
                1,
                3,
                inekey.as_mut_ptr(),
                &raw mut nfound,
                &raw mut status,
            );
            println!(
                "ffgkne:  {:.6}, {:.6}, {:.6}",
                inekey[0], inekey[1], inekey[2]
            );
            if nfound != 3 || status > 0 {
                print!("\nERROR in ffgkne {nfound}, {status}\n");
            }

            ffgknd(
                fptr.as_mut_ptr(),
                c"ky_pknd".as_ptr(),
                1,
                3,
                indkey.as_mut_ptr(),
                &raw mut nfound,
                &raw mut status,
            );
            println!(
                "ffgknd:  {:.6}, {:.6}, {:.6}",
                indkey[0], indkey[1], indkey[2]
            );
            if nfound != 3 || status > 0 {
                print!("\nERROR in ffgknd {nfound}, {status}\n");
            }

            /* get position of HISTORY keyword for subsequent deletes and inserts */
            ffgcrd(
                fptr.as_mut_ptr(),
                c"HISTORY".as_ptr(),
                card.as_mut_ptr(),
                &raw mut status,
            );
            ffghps(
                fptr.as_mut_ptr(),
                &raw mut existkeys,
                &raw mut keynum,
                &raw mut status,
            );
            keynum -= 2;

            print!("\nBefore deleting the HISTORY and DATE keywords...\n");
            for ii in keynum..=(keynum + 3) {
                ffgrec(fptr.as_mut_ptr(), ii, card.as_mut_ptr(), &raw mut status);
                println!("{:.8}", byte_slice_to_str!(&card)); /* don't print date value, so that */
            } /* the output will always be the same */

            /*
              ############################
              #  delete keywords         #
              ############################
            */

            ffdrec(fptr.as_mut_ptr(), keynum + 1, &raw mut status);
            ffdkey(fptr.as_mut_ptr(), c"DATE".as_ptr(), &raw mut status);

            print!("\nAfter deleting the keywords...\n");
            for ii in keynum..=(keynum + 1) {
                ffgrec(fptr.as_mut_ptr(), ii, card.as_mut_ptr(), &raw mut status);
                println!("{}", byte_slice_to_str!(&card));
            }

            if status > 0 {
                print!("\nERROR deleting keywords\n");
            }
            /*
              ############################
              #  insert keywords         #
              ############################
            */
            keynum += 4;
            ffirec(
                fptr.as_mut_ptr(),
                keynum - 3,
                c"KY_IREC = 'This keyword inserted by fxirec'".as_ptr(),
                &raw mut status,
            );
            ffikys(
                fptr.as_mut_ptr(),
                c"KY_IKYS".as_ptr(),
                c"insert_value_string".as_ptr(),
                c"ikys comment".as_ptr(),
                &raw mut status,
            );
            ffikyj(
                fptr.as_mut_ptr(),
                c"KY_IKYJ".as_ptr(),
                49,
                c"ikyj comment".as_ptr(),
                &raw mut status,
            );
            ffikyl(
                fptr.as_mut_ptr(),
                c"KY_IKYL".as_ptr(),
                1,
                c"ikyl comment".as_ptr(),
                &raw mut status,
            );
            ffikye(
                fptr.as_mut_ptr(),
                c"KY_IKYE".as_ptr(),
                12.3456,
                4,
                c"ikye comment".as_ptr(),
                &raw mut status,
            );
            ffikyd(
                fptr.as_mut_ptr(),
                c"KY_IKYD".as_ptr(),
                12.345678901234567,
                14,
                c"ikyd comment".as_ptr(),
                &raw mut status,
            );
            ffikyf(
                fptr.as_mut_ptr(),
                c"KY_IKYF".as_ptr(),
                12.3456,
                4,
                c"ikyf comment".as_ptr(),
                &raw mut status,
            );
            ffikyg(
                fptr.as_mut_ptr(),
                c"KY_IKYG".as_ptr(),
                12.345678901234567,
                13,
                c"ikyg comment".as_ptr(),
                &raw mut status,
            );

            print!("\nAfter inserting the keywords...\n");
            for ii in (keynum - 4)..=(keynum + 5) {
                ffgrec(fptr.as_mut_ptr(), ii, card.as_mut_ptr(), &raw mut status);
                println!("{}", byte_slice_to_str!(&card));
            }

            if status > 0 {
                print!("\nERROR inserting keywords\n");
            }
            /*
              ############################
              #  modify keywords         #
              ############################
            */
            ffmrec(
                fptr.as_mut_ptr(),
                keynum - 4,
                c"COMMENT   This keyword was modified by fxmrec".as_ptr(),
                &raw mut status,
            );
            ffmcrd(
                fptr.as_mut_ptr(),
                c"KY_IREC".as_ptr(),
                c"KY_MREC = 'This keyword was modified by fxmcrd'".as_ptr(),
                &raw mut status,
            );
            ffmnam(
                fptr.as_mut_ptr(),
                c"KY_IKYS".as_ptr(),
                c"NEWIKYS".as_ptr(),
                &raw mut status,
            );

            ffmcom(
                fptr.as_mut_ptr(),
                c"KY_IKYJ".as_ptr(),
                c"This is a modified comment".as_ptr(),
                &raw mut status,
            );
            ffmkyj(
                fptr.as_mut_ptr(),
                c"KY_IKYJ".as_ptr(),
                50,
                c"&".as_ptr(),
                &raw mut status,
            );
            ffmkyl(
                fptr.as_mut_ptr(),
                c"KY_IKYL".as_ptr(),
                0,
                c"&".as_ptr(),
                &raw mut status,
            );
            ffmkys(
                fptr.as_mut_ptr(),
                c"NEWIKYS".as_ptr(),
                c"modified_string".as_ptr(),
                c"&".as_ptr(),
                &raw mut status,
            );
            ffmkye(
                fptr.as_mut_ptr(),
                c"KY_IKYE".as_ptr(),
                -12.3456,
                4,
                c"&".as_ptr(),
                &raw mut status,
            );
            ffmkyd(
                fptr.as_mut_ptr(),
                c"KY_IKYD".as_ptr(),
                -12.345678901234567,
                14,
                c"modified comment".as_ptr(),
                &raw mut status,
            );
            ffmkyf(
                fptr.as_mut_ptr(),
                c"KY_IKYF".as_ptr(),
                -12.3456,
                4,
                c"&".as_ptr(),
                &raw mut status,
            );
            ffmkyg(
                fptr.as_mut_ptr(),
                c"KY_IKYG".as_ptr(),
                -12.345678901234567,
                13,
                c"&".as_ptr(),
                &raw mut status,
            );

            print!("\nAfter modifying the keywords...\n");

            for ii in (keynum - 4)..=(keynum + 5) {
                ffgrec(fptr.as_mut_ptr(), ii, card.as_mut_ptr(), &raw mut status);
                println!("{}", byte_slice_to_str!(&card));
            }
            if status > 0 {
                print!("\nERROR modifying keywords\n");
            }

            /*
              ############################
              #  update keywords         #
              ############################
            */
            ffucrd(
                fptr.as_mut_ptr(),
                c"KY_MREC".as_ptr(),
                c"KY_UCRD = 'This keyword was updated by fxucrd'".as_ptr(),
                &raw mut status,
            );

            ffukyj(
                fptr.as_mut_ptr(),
                c"KY_IKYJ".as_ptr(),
                51,
                c"&".as_ptr(),
                &raw mut status,
            );
            ffukyl(
                fptr.as_mut_ptr(),
                c"KY_IKYL".as_ptr(),
                1,
                c"&".as_ptr(),
                &raw mut status,
            );
            ffukys(
                fptr.as_mut_ptr(),
                c"NEWIKYS".as_ptr(),
                c"updated_string".as_ptr(),
                c"&".as_ptr(),
                &raw mut status,
            );
            ffukye(
                fptr.as_mut_ptr(),
                c"KY_IKYE".as_ptr(),
                -13.3456,
                4,
                c"&".as_ptr(),
                &raw mut status,
            );
            ffukyd(
                fptr.as_mut_ptr(),
                c"KY_IKYD".as_ptr(),
                -13.345678901234567,
                14,
                c"modified comment".as_ptr(),
                &raw mut status,
            );
            ffukyf(
                fptr.as_mut_ptr(),
                c"KY_IKYF".as_ptr(),
                -13.3456,
                4,
                c"&".as_ptr(),
                &raw mut status,
            );
            ffukyg(
                fptr.as_mut_ptr(),
                c"KY_IKYG".as_ptr(),
                -13.345678901234567,
                13,
                c"&".as_ptr(),
                &raw mut status,
            );

            print!("\nAfter updating the keywords...\n");
            for ii in (keynum - 4)..=(keynum + 5) {
                ffgrec(fptr.as_mut_ptr(), ii, card.as_mut_ptr(), &raw mut status);
                println!("{}", byte_slice_to_str!(&card));
            }
            if status > 0 {
                print!("\nERROR modifying keywords\n");
            }

            /* move to top of header and find keywords using wild cards */
            ffgrec(fptr.as_mut_ptr(), 0, card.as_mut_ptr(), &raw mut status);

            print!("\nKeywords found using wildcard search (should be 13)...\n");
            nfound = 0;
            while ffgnxk(
                fptr.as_mut_ptr(),
                inclist.as_ptr(),
                2,
                exclist.as_ptr(),
                2,
                card.as_mut_ptr(),
                &raw mut status,
            ) == 0
            {
                nfound += 1;
                println!("{}", byte_slice_to_str!(&card));
            }
            if nfound != 13 {
                print!("\nERROR reading keywords using wildcards (ffgnxk)\n");
                break 'mainloop;
            }
            status = 0;

            /*
              ############################
              #  copy index keyword      #
              ############################
            */

            // MIRI doesn't support this function call because it requires two mutable references to same object.
            if cfg!(not(miri)) {
                ffcpky(
                    fptr.as_mut_ptr(),
                    fptr.as_mut_ptr(),
                    1,
                    4,
                    c"KY_PKNE".as_ptr(),
                    &raw mut status,
                );
                ffgkne(
                    fptr.as_mut_ptr(),
                    c"ky_pkne".as_ptr(),
                    2,
                    3,
                    inekey.as_mut_ptr(),
                    &raw mut nfound,
                    &raw mut status,
                );
                print!(
                    "\nCopied keyword: ffgkne:  {:.6}, {:.6}, {:.6}\n",
                    inekey[0], inekey[1], inekey[2]
                );

                if status > 0 {
                    print!("\nERROR in ffgkne {nfound}, {status}\n");
                    break 'mainloop;
                }
            }

            /*
              ######################################
              #  modify header using template file #
              ######################################
            */
            if ffpktp(fptr.as_mut_ptr(), templt.as_ptr(), &raw mut status) != 0 {
                print!("\nERROR returned by ffpktp:\n");
                println!("Could not open or process the file 'testprog.tpt'.");
                println!("  This file is included with the CFITSIO distribution");
                println!("  and should be copied into the current directory");
                println!("  before running the testprog program.");
                status = 0;
            }
            println!("Updated header using template file (ffpktp)");
            /*
              ############################
              #  create binary table     #
              ############################
            */

            strcpy(tform[0], c"15A".as_ptr());
            strcpy(tform[1], c"1L".as_ptr());
            strcpy(tform[2], c"16X".as_ptr());
            strcpy(tform[3], c"1B".as_ptr());
            strcpy(tform[4], c"1I".as_ptr());
            strcpy(tform[5], c"1J".as_ptr());
            strcpy(tform[6], c"1E".as_ptr());
            strcpy(tform[7], c"1D".as_ptr());
            strcpy(tform[8], c"1C".as_ptr());
            strcpy(tform[9], c"1M".as_ptr());

            strcpy(ttype[0], c"Avalue".as_ptr());
            strcpy(ttype[1], c"Lvalue".as_ptr());
            strcpy(ttype[2], c"Xvalue".as_ptr());
            strcpy(ttype[3], c"Bvalue".as_ptr());
            strcpy(ttype[4], c"Ivalue".as_ptr());
            strcpy(ttype[5], c"Jvalue".as_ptr());
            strcpy(ttype[6], c"Evalue".as_ptr());
            strcpy(ttype[7], c"Dvalue".as_ptr());
            strcpy(ttype[8], c"Cvalue".as_ptr());
            strcpy(ttype[9], c"Mvalue".as_ptr());

            strcpy(tunit[0], c"".as_ptr());
            strcpy(tunit[1], c"m**2".as_ptr());
            strcpy(tunit[2], c"cm".as_ptr());
            strcpy(tunit[3], c"erg/s".as_ptr());
            strcpy(tunit[4], c"km/s".as_ptr());
            strcpy(tunit[5], c"".as_ptr());
            strcpy(tunit[6], c"".as_ptr());
            strcpy(tunit[7], c"".as_ptr());
            strcpy(tunit[8], c"".as_ptr());
            strcpy(tunit[9], c"".as_ptr());

            nrows = 21;
            tfields = 10;
            pcount = 0;

            /*
                ffcrtb(fptr.as_mut_ptr(), BINARY_TBL, nrows, tfields, ttype, tform, tunit, binname,
                        &mut status);
            */
            ffibin(
                fptr.as_mut_ptr(),
                nrows,
                tfields,
                ttype.as_ptr().cast::<*const c_char>(),
                tform.as_ptr().cast::<*const c_char>(),
                tunit.as_ptr().cast::<*const c_char>(),
                binname.as_ptr(),
                0,
                &raw mut status,
            );

            print!("\nffibin status = {status}\n");
            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            /* get size and position in header, and reserve space for more keywords */
            ffghps(
                fptr.as_mut_ptr(),
                &raw mut existkeys,
                &raw mut keynum,
                &raw mut status,
            );
            println!("header contains {existkeys} keywords; located at keyword {keynum} ");

            morekeys = 40;
            ffhdef(fptr.as_mut_ptr(), morekeys, &raw mut status);
            ffghsp(
                fptr.as_mut_ptr(),
                &raw mut existkeys,
                &raw mut morekeys,
                &raw mut status,
            );
            println!("header contains {existkeys} keywords with room for {morekeys} more");

            fftnul(fptr.as_mut_ptr(), 4, 99, &raw mut status); /* define null value for int cols */
            fftnul(fptr.as_mut_ptr(), 5, 99, &raw mut status);
            fftnul(fptr.as_mut_ptr(), 6, 99, &raw mut status);

            extvers = 1;
            ffpkyj(
                fptr.as_mut_ptr(),
                c"EXTVER".as_ptr(),
                extvers as LONGLONG,
                c"extension version number".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL4".as_ptr(),
                99,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL5".as_ptr(),
                99,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL6".as_ptr(),
                99,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );

            naxis = 3;
            naxes[0] = 1;
            naxes[1] = 2;
            naxes[2] = 8;
            ffptdm(fptr.as_mut_ptr(), 3, naxis, naxes.as_ptr(), &raw mut status);

            naxis = 0;
            naxes[0] = 0;
            naxes[1] = 0;
            naxes[2] = 0;
            ffgtdm(
                fptr.as_mut_ptr(),
                3,
                3,
                &raw mut naxis,
                naxes.as_mut_ptr(),
                &raw mut status,
            );
            ffgkys(
                fptr.as_mut_ptr(),
                c"TDIM3".as_ptr(),
                iskey.as_mut_ptr(),
                comment.as_mut_ptr(),
                &raw mut status,
            );
            println!(
                "TDIM3 = {}, {}, {}, {}, {}",
                byte_slice_to_str!(&iskey),
                naxis,
                naxes[0],
                naxes[1],
                naxes[2]
            );

            ffrdef(fptr.as_mut_ptr(), &raw mut status); /* force header to be scanned (not required) */

            /*
              ############################
              #  write data to columns   #
              ############################
            */

            /* initialize arrays of values to write to table */
            signval = -1;
            for ii in 0..21 as c_int {
                signval *= -1;
                boutarray[ii as usize] = (ii + 1) as c_uchar;
                ioutarray[ii as usize] = ((ii + 1) * signval) as c_short;
                joutarray[ii as usize] = ((ii + 1) * signval) as c_long;
                koutarray[ii as usize] = (ii + 1) * signval;
                eoutarray[ii as usize] = ((ii + 1) * signval) as f32;
                doutarray[ii as usize] = ((ii + 1) * signval) as f64;
            }

            ffpcls(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                3,
                onskey.as_ptr(),
                &raw mut status,
            ); /* write string values */
            ffpclu(fptr.as_mut_ptr(), 1, 4, 1, 1, &raw mut status); /* write null value */

            larray[0] = 0;
            larray[1] = 1;
            larray[2] = 0;
            larray[3] = 0;
            larray[4] = 1;
            larray[5] = 1;
            larray[6] = 0;
            larray[7] = 0;
            larray[8] = 0;
            larray[9] = 1;
            larray[10] = 1;
            larray[11] = 1;
            larray[12] = 0;
            larray[13] = 0;
            larray[14] = 0;
            larray[15] = 0;
            larray[16] = 1;
            larray[17] = 1;
            larray[18] = 1;
            larray[19] = 1;
            larray[20] = 0;
            larray[21] = 0;
            larray[22] = 0;
            larray[23] = 0;
            larray[24] = 0;
            larray[25] = 1;
            larray[26] = 1;
            larray[27] = 1;
            larray[28] = 1;
            larray[29] = 1;
            larray[30] = 0;
            larray[31] = 0;
            larray[32] = 0;
            larray[33] = 0;
            larray[34] = 0;
            larray[35] = 0;

            ffpclx(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                36,
                larray.as_ptr(),
                &raw mut status,
            ); /*write bits*/

            for ii in 4..9
            /* loop over cols 4 - 8 */
            {
                ffpclb(
                    fptr.as_mut_ptr(),
                    ii,
                    1,
                    1,
                    2,
                    boutarray.as_ptr(),
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcli(
                    fptr.as_mut_ptr(),
                    ii,
                    3,
                    1,
                    2,
                    ioutarray[2..].as_ptr(),
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpclk(
                    fptr.as_mut_ptr(),
                    ii,
                    5,
                    1,
                    2,
                    koutarray[4..].as_ptr(),
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcle(
                    fptr.as_mut_ptr(),
                    ii,
                    7,
                    1,
                    2,
                    eoutarray[6..].as_ptr(),
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcld(
                    fptr.as_mut_ptr(),
                    ii,
                    9,
                    1,
                    2,
                    doutarray[8..].as_ptr(),
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }

                ffpclu(fptr.as_mut_ptr(), ii, 11, 1, 1, &raw mut status); /* write null value */
            }

            ffpclc(
                fptr.as_mut_ptr(),
                9,
                1,
                1,
                10,
                eoutarray.as_ptr(),
                &raw mut status,
            );
            ffpclm(
                fptr.as_mut_ptr(),
                10,
                1,
                1,
                10,
                doutarray.as_ptr(),
                &raw mut status,
            );

            /* loop over cols 4 - 8 */
            for ii in 4..9 {
                ffpcnb(
                    fptr.as_mut_ptr(),
                    ii,
                    12,
                    1,
                    2,
                    boutarray[11..].as_ptr(),
                    13,
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcni(
                    fptr.as_mut_ptr(),
                    ii,
                    14,
                    1,
                    2,
                    ioutarray[13..].as_ptr(),
                    15,
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcnk(
                    fptr.as_mut_ptr(),
                    ii,
                    16,
                    1,
                    2,
                    koutarray[15..].as_ptr(),
                    17,
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcne(
                    fptr.as_mut_ptr(),
                    ii,
                    18,
                    1,
                    2,
                    eoutarray[17..].as_ptr(),
                    19.,
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcnd(
                    fptr.as_mut_ptr(),
                    ii,
                    20,
                    1,
                    2,
                    doutarray[19..].as_ptr(),
                    21.,
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    status = 0;
                }
            }
            ffpcll(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                21,
                larray.as_ptr(),
                &raw mut status,
            ); /*write logicals*/
            ffpclu(fptr.as_mut_ptr(), 2, 11, 1, 1, &raw mut status); /* write null value */
            println!("ffpcl_ status = {status}");

            /*
              #########################################
              #  get information about the columns    #
              #########################################
            */

            print!("\nFind the column numbers; a returned status value of 237 is");
            print!("\nexpected and indicates that more than one column name matches");
            print!("\nthe input column name template.  Status = 219 indicates that");
            print!("\nthere was no matching column name.");

            ffgcno(
                fptr.as_mut_ptr(),
                0,
                c"Xvalue".as_ptr(),
                &raw mut colnum,
                &raw mut status,
            );
            print!("\nColumn Xvalue is number {colnum}; status = {status}.\n");

            while status != COL_NOT_FOUND {
                ffgcnn(
                    fptr.as_mut_ptr(),
                    1,
                    c"*ue".as_ptr(),
                    colname.as_mut_ptr(),
                    &raw mut colnum,
                    &raw mut status,
                );
                println!(
                    "Column {} is number {}; status = {}.",
                    byte_slice_to_str!(&colname),
                    colnum,
                    status
                );
            }
            status = 0;

            print!("\nInformation about each column:\n");

            for ii in 0..tfields {
                ffgtcl(
                    fptr.as_mut_ptr(),
                    ii + 1,
                    &raw mut typecode,
                    &raw mut repeat,
                    &raw mut width,
                    &raw mut status,
                );
                print!(
                    "{:>4} {:>3} {:>2} {:>2}",
                    CStr::from_ptr(tform[ii as usize].cast_const())
                        .to_str()
                        .unwrap(),
                    typecode,
                    repeat,
                    width
                );
                ffgbcl(
                    fptr.as_mut_ptr(),
                    ii + 1,
                    ttype[0],
                    tunit[0],
                    cvalstr.as_mut_ptr().cast::<[c_char; 3]>(),
                    &raw mut repeat,
                    &raw mut scale,
                    &raw mut zero,
                    &raw mut jnulval,
                    tdisp.as_mut_ptr(),
                    &raw mut status,
                );
                println!(
                    " {}, {}, {}, {}, {:.6}, {:.6}, {}, {}.",
                    CStr::from_ptr(ttype[0].cast_const()).to_str().unwrap(),
                    CStr::from_ptr(tunit[0].cast_const()).to_str().unwrap(),
                    cvalstr[0] as u8 as char,
                    repeat,
                    scale,
                    zero,
                    jnulval,
                    byte_slice_to_str!(&tdisp)
                );
            }

            println!();

            /*
              ###############################################
              #  insert ASCII table before the binary table #
              ###############################################
            */

            if ffmrhd(fptr.as_mut_ptr(), -1, &raw mut hdutype, &raw mut status) > 0 {
                break 'mainloop;
            }

            strcpy(tform[0], c"A15".as_ptr());
            strcpy(tform[1], c"I10".as_ptr());
            strcpy(tform[2], c"F14.6".as_ptr());
            strcpy(tform[3], c"E12.5".as_ptr());
            strcpy(tform[4], c"D21.14".as_ptr());

            strcpy(ttype[0], c"Name".as_ptr());
            strcpy(ttype[1], c"Ivalue".as_ptr());
            strcpy(ttype[2], c"Fvalue".as_ptr());
            strcpy(ttype[3], c"Evalue".as_ptr());
            strcpy(ttype[4], c"Dvalue".as_ptr());

            strcpy(tunit[0], c"".as_ptr());
            strcpy(tunit[1], c"m**2".as_ptr());
            strcpy(tunit[2], c"cm".as_ptr());
            strcpy(tunit[3], c"erg/s".as_ptr());
            strcpy(tunit[4], c"km/s".as_ptr());

            rowlen = 76;
            nrows = 11;
            tfields = 5;

            ffitab(
                fptr.as_mut_ptr(),
                rowlen as LONGLONG,
                nrows,
                tfields,
                ttype.as_ptr().cast::<*const c_char>(),
                tbcol.as_ptr(),
                tform.as_ptr().cast::<*const c_char>(),
                tunit.as_ptr().cast::<*const c_char>(),
                tblname.as_ptr(),
                &raw mut status,
            );
            println!("ffitab status = {status}");
            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            ffsnul(fptr.as_mut_ptr(), 1, c"null1".as_ptr(), &raw mut status); /* define null value for int cols */
            ffsnul(fptr.as_mut_ptr(), 2, c"null2".as_ptr(), &raw mut status);
            ffsnul(fptr.as_mut_ptr(), 3, c"null3".as_ptr(), &raw mut status);
            ffsnul(fptr.as_mut_ptr(), 4, c"null4".as_ptr(), &raw mut status);
            ffsnul(fptr.as_mut_ptr(), 5, c"null5".as_ptr(), &raw mut status);

            extvers = 2;
            ffpkyj(
                fptr.as_mut_ptr(),
                c"EXTVER".as_ptr(),
                extvers as LONGLONG,
                c"extension version number".as_ptr(),
                &raw mut status,
            );

            ffpkys(
                fptr.as_mut_ptr(),
                c"TNULL1".as_ptr(),
                c"null1".as_ptr(),
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkys(
                fptr.as_mut_ptr(),
                c"TNULL2".as_ptr(),
                c"null2".as_ptr(),
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkys(
                fptr.as_mut_ptr(),
                c"TNULL3".as_ptr(),
                c"null3".as_ptr(),
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkys(
                fptr.as_mut_ptr(),
                c"TNULL4".as_ptr(),
                c"null4".as_ptr(),
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkys(
                fptr.as_mut_ptr(),
                c"TNULL5".as_ptr(),
                c"null5".as_ptr(),
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );

            if status > 0 {
                break 'mainloop;
            }

            /*
              ############################
              #  write data to columns   #
              ############################
            */

            /* initialize arrays of values to write to table */
            for ii in 0..21 {
                boutarray[ii] = (ii + 1) as c_uchar;
                ioutarray[ii] = (ii + 1) as c_short;
                joutarray[ii] = (ii + 1) as c_long;
                eoutarray[ii] = (ii + 1) as f32;
                doutarray[ii] = (ii + 1) as f64;
            }

            ffpcls(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                3,
                onskey.as_ptr(),
                &raw mut status,
            ); /* write string values */
            ffpclu(fptr.as_mut_ptr(), 1, 4, 1, 1, &raw mut status); /* write null value */

            for ii in 2..6
            /* loop over cols 2 - 5 */
            {
                ffpclb(
                    fptr.as_mut_ptr(),
                    ii,
                    1,
                    1,
                    2,
                    boutarray.as_ptr(),
                    &raw mut status,
                ); /* char array */
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcli(
                    fptr.as_mut_ptr(),
                    ii,
                    3,
                    1,
                    2,
                    ioutarray[2..].as_ptr(),
                    &raw mut status,
                ); /* short array */
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpclj(
                    fptr.as_mut_ptr(),
                    ii,
                    5,
                    1,
                    2,
                    joutarray[4..].as_ptr(),
                    &raw mut status,
                ); /* long array */
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcle(
                    fptr.as_mut_ptr(),
                    ii,
                    7,
                    1,
                    2,
                    eoutarray[6..].as_ptr(),
                    &raw mut status,
                ); /* float array */
                if status == NUM_OVERFLOW {
                    status = 0;
                }
                ffpcld(
                    fptr.as_mut_ptr(),
                    ii,
                    9,
                    1,
                    2,
                    doutarray[8..].as_ptr(),
                    &raw mut status,
                ); /* double array */
                if status == NUM_OVERFLOW {
                    status = 0;
                }

                ffpclu(fptr.as_mut_ptr(), ii, 11, 1, 1, &raw mut status); /* write null value */
            }
            println!("ffpcl_ status = {status}");

            /*
              ################################
              #  read data from ASCII table  #
              ################################
            */

            let mut tmp_nrows: c_long = 0;

            ffghtb(
                fptr.as_mut_ptr(),
                99,
                &raw mut rowlen,
                &raw mut tmp_nrows,
                &raw mut tfields,
                ttype.as_mut_ptr(),
                tbcol.as_mut_ptr(),
                tform.as_mut_ptr(),
                tunit.as_mut_ptr(),
                tblname.as_mut_ptr(),
                &raw mut status,
            );

            nrows = tmp_nrows as LONGLONG;

            print!(
                "\nASCII table: rowlen, nrows, tfields, extname: {} {} {} {}\n",
                rowlen,
                nrows,
                tfields,
                byte_slice_to_str!(&tblname)
            );

            for ii in 0..tfields as usize {
                println!(
                    "{:>8} {:>3} {:>8} {:>8} ",
                    CStr::from_ptr(ttype[ii]).to_str().unwrap(),
                    tbcol[ii],
                    CStr::from_ptr(tform[ii]).to_str().unwrap(),
                    CStr::from_ptr(tunit[ii]).to_str().unwrap()
                );
            }

            nrows = 11;
            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"UNDEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvj(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                nrows,
                99,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                99.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                99.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values read from ASCII table:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>2} {:>2} {:>4.1} {:>4.1}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    jinarray[ii],
                    einarray[ii],
                    dinarray[ii]
                );
            }

            ffgtbb(
                fptr.as_mut_ptr(),
                1,
                20,
                78,
                uchars.as_mut_ptr(),
                &raw mut status,
            );
            uchars[78] = 0;
            print!("\n{}\n", byte_slice_to_str!(&uchars));
            ffptbb(
                fptr.as_mut_ptr(),
                1,
                20,
                78,
                uchars.as_ptr(),
                &raw mut status,
            );

            /*
              #########################################
              #  get information about the columns    #
              #########################################
            */

            ffgcno(
                fptr.as_mut_ptr(),
                0,
                c"name".as_ptr(),
                &raw mut colnum,
                &raw mut status,
            );
            print!("\nColumn name is number {colnum}; status = {status}.\n");

            while status != COL_NOT_FOUND {
                ffgcnn(
                    fptr.as_mut_ptr(),
                    1,
                    c"*ue".as_ptr(),
                    colname.as_mut_ptr(),
                    &raw mut colnum,
                    &raw mut status,
                );
                println!(
                    "Column {} is number {}; status = {}.",
                    byte_slice_to_str!(&colname),
                    colnum,
                    status
                );
            }
            status = 0;

            for ii in 0..tfields {
                ffgtcl(
                    fptr.as_mut_ptr(),
                    ii + 1,
                    &raw mut typecode,
                    &raw mut repeat,
                    &raw mut width,
                    &raw mut status,
                );
                print!(
                    "{:>4} {:>3} {:>2} {:>2}",
                    CStr::from_ptr(tform[ii as usize]).to_str().unwrap(),
                    typecode,
                    repeat,
                    width
                );
                ffgacl(
                    fptr.as_mut_ptr(),
                    ii + 1,
                    ttype[0],
                    tbcol.as_mut_ptr(),
                    tunit[0],
                    tform[0],
                    &raw mut scale,
                    &raw mut zero,
                    nulstr.as_mut_ptr(),
                    tdisp.as_mut_ptr(),
                    &raw mut status,
                );
                println!(
                    " {}, {}, {}, {}, {:.6}, {:.6}, {}, {}.",
                    CStr::from_ptr(ttype[0]).to_str().unwrap(),
                    tbcol[0],
                    CStr::from_ptr(tunit[0]).to_str().unwrap(),
                    CStr::from_ptr(tform[0]).to_str().unwrap(),
                    scale,
                    zero,
                    byte_slice_to_str!(&nulstr),
                    byte_slice_to_str!(&tdisp)
                );
            }

            println!();

            /*
              ###############################################
              #  test the insert/delete row/column routines #
              ###############################################
            */

            if ffirow(fptr.as_mut_ptr(), 2, 3, &raw mut status) > 0 {
                break 'mainloop;
            }

            nrows = 14;
            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"UNDEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvj(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                nrows,
                99,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                99.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                99.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values after inserting 3 rows after row 2:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>2} {:>2} {:>4.1} {:>4.1}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    jinarray[ii],
                    einarray[ii],
                    dinarray[ii]
                );
            }

            if ffdrow(fptr.as_mut_ptr(), 10, 2, &raw mut status) > 0 {
                break 'mainloop;
            }

            nrows = 12;
            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"UNDEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvj(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                nrows,
                99,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                99.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                99.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values after deleting 2 rows at row 10:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>2} {:>2} {:>4.1} {:>4.1}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    jinarray[ii],
                    einarray[ii],
                    dinarray[ii]
                );
            }
            if ffdcol(fptr.as_mut_ptr(), 3, &raw mut status) > 0 {
                break 'mainloop;
            }

            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"UNDEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                nrows,
                99.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                99.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values after deleting column 3:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>2} {:>4.1} {:>4.1}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    einarray[ii],
                    dinarray[ii]
                );
            }

            if fficol(
                fptr.as_mut_ptr(),
                5,
                c"INSERT_COL".as_ptr(),
                c"F14.6".as_ptr(),
                &raw mut status,
            ) > 0
            {
                break 'mainloop;
            }

            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"UNDEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                99,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                nrows,
                99.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                99.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvj(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                99,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values after inserting column 5:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>2} {:>4.1} {:>4.1} {}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    einarray[ii],
                    dinarray[ii],
                    jinarray[ii]
                );
            }

            /*
              ############################################################
              #  create a temporary file and copy the ASCII table to it, #
              #  column by column.                                       #
              ############################################################
            */
            bitpix = 16;
            naxis = 0;

            strcpy_safe(&mut filename, cs!(c"!t1q2s3v6.tmp"));
            ffinit(&raw mut tmpfptr, filename.as_ptr(), &raw mut status);
            println!("Create temporary file: ffinit status = {status}");

            ffiimg(
                tmpfptr.as_mut_ptr(),
                bitpix,
                naxis,
                naxes.as_ptr(),
                &raw mut status,
            );
            print!("\nCreate null primary array: ffiimg status = {status}\n");

            /* create an empty table with 12 rows and 0 columns */
            nrows = 12;
            tfields = 0;
            rowlen = 0;
            ffitab(
                tmpfptr.as_mut_ptr(),
                rowlen as LONGLONG,
                nrows,
                tfields,
                ttype.as_ptr().cast::<*const c_char>(),
                tbcol.as_ptr(),
                tform.as_ptr().cast::<*const c_char>(),
                tunit.as_ptr().cast::<*const c_char>(),
                tblname.as_ptr(),
                &raw mut status,
            );
            print!("\nCreate ASCII table with 0 columns: ffitab status = {status}\n");

            /* copy columns from one table to the other */
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                4,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                3,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                2,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                1,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");

            /* now repeat by copying ASCII input to Binary output table */
            ffibin(
                tmpfptr.as_mut_ptr(),
                nrows,
                tfields,
                ttype.as_ptr().cast::<*const c_char>(),
                tform.as_ptr().cast::<*const c_char>(),
                tunit.as_ptr().cast::<*const c_char>(),
                tblname.as_ptr(),
                0,
                &raw mut status,
            );
            print!("\nCreate Binary table with 0 columns: ffibin status = {status}\n");

            /* copy columns from one table to the other */
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                4,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                3,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                2,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                1,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");

            /*
                ffclos(tmpfptr.as_mut_ptr(), &mut status);
                print!("Close the tmp file: ffclos status = {}\n", status);
            */

            let t = Box::into_raw(tmpfptr.take().unwrap());
            ffdelt(t, &raw mut status);
            println!("Delete the tmp file: ffdelt status = {status}");

            if status > 0 {
                break 'mainloop;
            }

            /*
              ################################
              #  read data from binary table #
              ################################
            */

            if ffmrhd(fptr.as_mut_ptr(), 1, &raw mut hdutype, &raw mut status) > 0 {
                break 'mainloop;
            }

            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            ffghsp(
                fptr.as_mut_ptr(),
                &raw mut existkeys,
                &raw mut morekeys,
                &raw mut status,
            );
            println!("header contains {existkeys} keywords with room for {morekeys} more");

            let mut tmp_nrows: c_long = 0;
            ffghbn(
                fptr.as_mut_ptr(),
                99,
                &raw mut tmp_nrows,
                &raw mut tfields,
                ttype.as_mut_ptr(),
                tform.as_mut_ptr(),
                tunit.as_mut_ptr(),
                binname.as_mut_ptr(),
                &raw mut pcount,
                &raw mut status,
            );

            nrows = tmp_nrows as LONGLONG;

            print!(
                "\nBinary table: nrows, tfields, extname, pcount: {} {} {} {}\n",
                nrows,
                tfields,
                byte_slice_to_str!(&binname),
                pcount
            );

            for ii in 0..tfields as usize {
                println!(
                    "{:>8} {:>8} {:>8} ",
                    CStr::from_ptr(ttype[ii]).to_str().unwrap(),
                    CStr::from_ptr(tform[ii]).to_str().unwrap(),
                    CStr::from_ptr(tunit[ii]).to_str().unwrap()
                );
            }

            for item in &mut larray[..40] {
                *item = 0;
            }

            print!("\nData values read from binary table:\n");
            print!("  Bit column (X) data values: \n\n");

            ffgcx(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                36,
                larray.as_mut_ptr(),
                &raw mut status,
            );
            for jj in 0..5 {
                for ii in 0..8 {
                    print!("{:>1}", larray[jj * 8 + ii]);
                }
                print!(" ");
            }

            for ii in 0..nrows as usize {
                larray[ii] = 0;
                xinarray[ii] = 0;
                binarray[ii] = 0;
                iinarray[ii] = 0;
                kinarray[ii] = 0;
                einarray[ii] = 0.0;
                dinarray[ii] = 0.0;
                cinarray[ii * 2] = 0.0;
                minarray[ii * 2] = 0.0;
                cinarray[ii * 2 + 1] = 0.0;
                minarray[ii * 2 + 1] = 0.0;
            }

            print!("\n\n");
            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                4,
                1,
                1,
                c"".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            println!(
                "null string column value = -{}- (should be --)",
                CStr::from_ptr(inskey[0]).to_str().unwrap()
            );

            nrows = 21;
            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"NOT DEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcl(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                larray.as_mut_ptr(),
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                nrows,
                98,
                xinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                98,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                98,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvk(
                fptr.as_mut_ptr(),
                6,
                1,
                1,
                nrows,
                98,
                kinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                7,
                1,
                1,
                nrows,
                98.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                8,
                1,
                1,
                nrows,
                98.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvc(
                fptr.as_mut_ptr(),
                9,
                1,
                1,
                nrows,
                98.,
                cinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvm(
                fptr.as_mut_ptr(),
                10,
                1,
                1,
                nrows,
                98.,
                minarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nRead columns with ffgcv_:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {} {:>3} {:>2} {:>3} {:>3} {:>5.1} {:>5.1} ({:>5.1},{:>5.1}) ({:>5.1},{:>5.1}) ",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    larray[ii],
                    xinarray[ii],
                    binarray[ii],
                    iinarray[ii],
                    kinarray[ii],
                    einarray[ii],
                    dinarray[ii],
                    cinarray[ii * 2],
                    cinarray[ii * 2 + 1],
                    minarray[ii * 2],
                    minarray[ii * 2 + 1]
                );
            }

            for ii in 0..nrows as usize {
                larray[ii] = 0;
                xinarray[ii] = 0;
                binarray[ii] = 0;
                iinarray[ii] = 0;
                kinarray[ii] = 0;
                einarray[ii] = 0.0;
                dinarray[ii] = 0.0;
                cinarray[ii * 2] = 0.0;
                minarray[ii * 2] = 0.0;
                cinarray[ii * 2 + 1] = 0.0;
                minarray[ii * 2 + 1] = 0.0;
            }

            ffgcfs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                inskey.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfl(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                nrows,
                larray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfb(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                nrows,
                xinarray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfb(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                binarray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfi(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                iinarray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfk(
                fptr.as_mut_ptr(),
                6,
                1,
                1,
                nrows,
                kinarray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfe(
                fptr.as_mut_ptr(),
                7,
                1,
                1,
                nrows,
                einarray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfd(
                fptr.as_mut_ptr(),
                8,
                1,
                1,
                nrows,
                dinarray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfc(
                fptr.as_mut_ptr(),
                9,
                1,
                1,
                nrows,
                cinarray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcfm(
                fptr.as_mut_ptr(),
                10,
                1,
                1,
                nrows,
                minarray.as_mut_ptr(),
                larray2.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nRead columns with ffgcf_:\n");
            for ii in 0..10_usize {
                println!(
                    "{:>15} {} {:>3} {:>2} {:>3} {:>3} {:>5.1} {:>5.1} ({:>5.1},{:>5.1}) ({:>5.1},{:>5.1})",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    larray[ii],
                    xinarray[ii],
                    binarray[ii],
                    iinarray[ii],
                    kinarray[ii],
                    einarray[ii],
                    dinarray[ii],
                    cinarray[ii * 2],
                    cinarray[ii * 2 + 1],
                    minarray[ii * 2],
                    minarray[ii * 2 + 1]
                );
            }
            for ii in 10..nrows as usize {
                /* don't try to print the NaN values */
                println!(
                    "{:>15} {} {:>3} {:>2} {:>3} ",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    larray[ii],
                    xinarray[ii],
                    binarray[ii],
                    iinarray[ii]
                );
            }
            ffprec(
                fptr.as_mut_ptr(),
                c"key_prec= 'This keyword was written by f_prec' / comment here".as_ptr(),
                &raw mut status,
            );

            /*
              ###############################################
              #  test the insert/delete row/column routines #
              ###############################################
            */
            if ffirow(fptr.as_mut_ptr(), 2, 3, &raw mut status) > 0 {
                break 'mainloop;
            }

            nrows = 14;
            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"NOT DEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                98,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                98,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvj(
                fptr.as_mut_ptr(),
                6,
                1,
                1,
                nrows,
                98,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                7,
                1,
                1,
                nrows,
                98.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                8,
                1,
                1,
                nrows,
                98.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values after inserting 3 rows after row 2:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>3} {:>3} {:>5.1} {:>5.1}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    jinarray[ii],
                    einarray[ii],
                    dinarray[ii]
                );
            }

            if ffdrow(fptr.as_mut_ptr(), 10, 2, &raw mut status) > 0 {
                break 'mainloop;
            }

            nrows = 12;
            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"NOT DEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                98,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                98,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvj(
                fptr.as_mut_ptr(),
                6,
                1,
                1,
                nrows,
                98,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                7,
                1,
                1,
                nrows,
                98.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                8,
                1,
                1,
                nrows,
                98.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values after deleting 2 rows at row 10:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>3} {:>3} {:>5.1} {:>5.1}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    jinarray[ii],
                    einarray[ii],
                    dinarray[ii]
                );
            }

            if ffdcol(fptr.as_mut_ptr(), 6, &raw mut status) > 0 {
                break 'mainloop;
            }

            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"NOT DEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                98,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                98,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                6,
                1,
                1,
                nrows,
                98.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                7,
                1,
                1,
                nrows,
                98.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values after deleting column 6:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>3} {:>5.1} {:>5.1}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    einarray[ii],
                    dinarray[ii]
                );
            }

            if fficol(
                fptr.as_mut_ptr(),
                8,
                c"INSERT_COL".as_ptr(),
                c"1E".as_ptr(),
                &raw mut status,
            ) > 0
            {
                break 'mainloop;
            }

            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"NOT DEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                98,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                98,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                6,
                1,
                1,
                nrows,
                98.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                7,
                1,
                1,
                nrows,
                98.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvj(
                fptr.as_mut_ptr(),
                8,
                1,
                1,
                nrows,
                98,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nData values after inserting column 8:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:>2} {:>3} {:>5.1} {:>5.1} {}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    einarray[ii],
                    dinarray[ii],
                    jinarray[ii]
                );
            }

            ffpclu(fptr.as_mut_ptr(), 8, 1, 1, 10, &raw mut status);

            ffgcvs(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                nrows,
                c"NOT DEFINED".as_ptr(),
                inskey.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvb(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                nrows,
                98,
                binarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvi(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                nrows,
                98,
                iinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcve(
                fptr.as_mut_ptr(),
                6,
                1,
                1,
                nrows,
                98.,
                einarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvd(
                fptr.as_mut_ptr(),
                7,
                1,
                1,
                nrows,
                98.,
                dinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcvj(
                fptr.as_mut_ptr(),
                8,
                1,
                1,
                nrows,
                98,
                jinarray.as_mut_ptr(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nValues after setting 1st 10 elements in column 8 = null:\n");
            for ii in 0..nrows as usize {
                println!(
                    "{:>15} {:2} {:3} {:5.1} {:5.1} {}",
                    CStr::from_ptr(inskey[ii]).to_str().unwrap(),
                    binarray[ii],
                    iinarray[ii],
                    einarray[ii],
                    dinarray[ii],
                    jinarray[ii]
                );
            }

            /*
              ############################################################
              #  create a temporary file and copy the binary table to it,#
              #  column by column.                                       #
              ############################################################
            */
            bitpix = 16;
            naxis = 0;

            strcpy_safe(&mut filename, cs!(c"!t1q2s3v5.tmp"));
            ffinit(&raw mut tmpfptr, filename.as_ptr(), &raw mut status);
            println!("Create temporary file: ffinit status = {status}");

            ffiimg(
                tmpfptr.as_mut_ptr(),
                bitpix,
                naxis,
                naxes.as_ptr(),
                &raw mut status,
            );
            print!("\nCreate null primary array: ffiimg status = {status}\n");

            /* create an empty table with 22 rows and 0 columns */
            nrows = 22;
            tfields = 0;
            ffibin(
                tmpfptr.as_mut_ptr(),
                nrows,
                tfields,
                ttype.as_ptr().cast::<*const c_char>(),
                tform.as_ptr().cast::<*const c_char>(),
                tunit.as_ptr().cast::<*const c_char>(),
                binname.as_ptr(),
                0,
                &raw mut status,
            );
            print!("\nCreate binary table with 0 columns: ffibin status = {status}\n");

            /* copy columns from one table to the other */
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                7,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                6,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                5,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                4,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                3,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                2,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");
            ffcpcl(
                fptr.as_mut_ptr(),
                tmpfptr.as_mut_ptr(),
                1,
                1,
                TRUE as c_int,
                &raw mut status,
            );
            println!("copy column, ffcpcl status = {status}");

            /*
                ffclos(tmpfptr.as_mut_ptr(), &mut status);
                print!("Close the tmp file: ffclos status = {}\n", status);
            */

            let t = Box::into_raw(tmpfptr.take().unwrap());
            ffdelt(t, &raw mut status);
            println!("Delete the tmp file: ffdelt status = {status}");
            if status > 0 {
                break 'mainloop;
            }
            /*
              ####################################################
              #  insert binary table following the primary array #
              ####################################################
            */

            ffmahd(fptr.as_mut_ptr(), 1, &raw mut hdutype, &raw mut status);

            strcpy(tform[0], c"15A".as_ptr());
            strcpy(tform[1], c"1L".as_ptr());
            strcpy(tform[2], c"16X".as_ptr());
            strcpy(tform[3], c"1B".as_ptr());
            strcpy(tform[4], c"1I".as_ptr());
            strcpy(tform[5], c"1J".as_ptr());
            strcpy(tform[6], c"1E".as_ptr());
            strcpy(tform[7], c"1D".as_ptr());
            strcpy(tform[8], c"1C".as_ptr());
            strcpy(tform[9], c"1M".as_ptr());

            strcpy(ttype[0], c"Avalue".as_ptr());
            strcpy(ttype[1], c"Lvalue".as_ptr());
            strcpy(ttype[2], c"Xvalue".as_ptr());
            strcpy(ttype[3], c"Bvalue".as_ptr());
            strcpy(ttype[4], c"Ivalue".as_ptr());
            strcpy(ttype[5], c"Jvalue".as_ptr());
            strcpy(ttype[6], c"Evalue".as_ptr());
            strcpy(ttype[7], c"Dvalue".as_ptr());
            strcpy(ttype[8], c"Cvalue".as_ptr());
            strcpy(ttype[9], c"Mvalue".as_ptr());

            strcpy(tunit[0], c"".as_ptr());
            strcpy(tunit[1], c"m**2".as_ptr());
            strcpy(tunit[2], c"cm".as_ptr());
            strcpy(tunit[3], c"erg/s".as_ptr());
            strcpy(tunit[4], c"km/s".as_ptr());
            strcpy(tunit[5], c"".as_ptr());
            strcpy(tunit[6], c"".as_ptr());
            strcpy(tunit[7], c"".as_ptr());
            strcpy(tunit[8], c"".as_ptr());
            strcpy(tunit[9], c"".as_ptr());

            nrows = 20;
            tfields = 10;
            pcount = 0;

            ffibin(
                fptr.as_mut_ptr(),
                nrows,
                tfields,
                ttype.as_ptr().cast::<*const c_char>(),
                tform.as_ptr().cast::<*const c_char>(),
                tunit.as_ptr().cast::<*const c_char>(),
                binname.as_ptr(),
                pcount as LONGLONG,
                &raw mut status,
            );
            println!("ffibin status = {status}");
            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            extvers = 3;
            ffpkyj(
                fptr.as_mut_ptr(),
                c"EXTVER".as_ptr(),
                extvers as LONGLONG,
                c"extension version number".as_ptr(),
                &raw mut status,
            );

            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL4".as_ptr(),
                77,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL5".as_ptr(),
                77,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL6".as_ptr(),
                77,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );

            ffpkyj(
                fptr.as_mut_ptr(),
                c"TSCAL4".as_ptr(),
                1000,
                c"scaling factor".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TSCAL5".as_ptr(),
                1,
                c"scaling factor".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TSCAL6".as_ptr(),
                100,
                c"scaling factor".as_ptr(),
                &raw mut status,
            );

            ffpkyj(
                fptr.as_mut_ptr(),
                c"TZERO4".as_ptr(),
                0,
                c"scaling offset".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TZERO5".as_ptr(),
                32768,
                c"scaling offset".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TZERO6".as_ptr(),
                100,
                c"scaling offset".as_ptr(),
                &raw mut status,
            );

            fftnul(fptr.as_mut_ptr(), 4, 77, &raw mut status); /* define null value for int cols */
            fftnul(fptr.as_mut_ptr(), 5, 77, &raw mut status);
            fftnul(fptr.as_mut_ptr(), 6, 77, &raw mut status);
            /* set scaling */
            fftscl(fptr.as_mut_ptr(), 4, 1000., 0., &raw mut status);
            fftscl(fptr.as_mut_ptr(), 5, 1., 32768., &raw mut status);
            fftscl(fptr.as_mut_ptr(), 6, 100., 100., &raw mut status);

            /*
              ############################
              #  write data to columns   #
              ############################
            */

            /* initialize arrays of values to write to table */

            joutarray[0] = 0;
            joutarray[1] = 1000;
            joutarray[2] = 10000;
            joutarray[3] = 32768;
            joutarray[4] = 65535;

            for ii in 4..7 {
                ffpclj(
                    fptr.as_mut_ptr(),
                    ii,
                    1,
                    1,
                    5,
                    joutarray.as_ptr(),
                    &raw mut status,
                );
                if status == NUM_OVERFLOW {
                    println!("Overflow writing to column {ii}");
                    status = 0;
                }

                ffpclu(fptr.as_mut_ptr(), ii, 6, 1, 1, &raw mut status); /* write null value */
            }

            for jj in 4..7 {
                ffgcvj(
                    fptr.as_mut_ptr(),
                    jj,
                    1,
                    1,
                    6,
                    -999,
                    jinarray.as_mut_ptr(),
                    &raw mut anynull,
                    &raw mut status,
                );

                for item in &jinarray[..6] {
                    print!(" {item:>6}");
                }
                println!();
            }

            println!();
            /* turn off scaling, and read the unscaled values */
            fftscl(fptr.as_mut_ptr(), 4, 1., 0., &raw mut status);
            fftscl(fptr.as_mut_ptr(), 5, 1., 0., &raw mut status);
            fftscl(fptr.as_mut_ptr(), 6, 1., 0., &raw mut status);

            for jj in 4..7 {
                ffgcvj(
                    fptr.as_mut_ptr(),
                    jj,
                    1,
                    1,
                    6,
                    -999,
                    jinarray.as_mut_ptr(),
                    &raw mut anynull,
                    &raw mut status,
                );
                for item in &jinarray[..6] {
                    print!(" {item:>6}");
                }
                println!();
            }
            /*
              ######################################################
              #  insert image extension following the binary table #
              ######################################################
            */

            bitpix = -32;
            naxis = 2;
            naxes[0] = 15;
            naxes[1] = 25;
            ffiimg(
                fptr.as_mut_ptr(),
                bitpix,
                naxis,
                naxes.as_ptr(),
                &raw mut status,
            );
            print!("\nCreate image extension: ffiimg status = {status}\n");
            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            for jj in 0..30 {
                for ii in 0..19 {
                    imgarray[jj][ii] = ((jj * 10) + ii) as c_short;
                }
            }

            ffp2di(
                fptr.as_mut_ptr(),
                1,
                19,
                naxes[0] as LONGLONG,
                naxes[1] as LONGLONG,
                imgarray.as_ptr().cast::<c_short>(),
                &raw mut status,
            );
            print!("\nWrote whole 2D array: ffp2di status = {status}\n");

            for jj in 0..30 {
                for ii in 0..19 {
                    imgarray[jj][ii] = 0;
                }
            }

            ffg2di(
                fptr.as_mut_ptr(),
                1,
                0,
                19,
                naxes[0] as LONGLONG,
                naxes[1] as LONGLONG,
                imgarray.as_mut_ptr().cast::<c_short>(),
                &raw mut anynull,
                &raw mut status,
            );
            print!("\nRead whole 2D array: ffg2di status = {status}\n");

            for jj in 0..30 {
                for ii in 0..19 {
                    print!(" {:>3}", imgarray[jj][ii]);
                }
                println!();
            }

            for jj in 0..30 {
                for ii in 0..19 {
                    imgarray[jj][ii] = 0;
                }
            }

            for jj in 0..20 {
                for ii in 0..10 {
                    imgarray2[jj][ii] = ((jj as c_int * -10) - ii as c_int) as c_short;
                }
            }

            fpixels[0] = 5;
            fpixels[1] = 5;
            lpixels[0] = 14;
            lpixels[1] = 14;
            ffpssi(
                fptr.as_mut_ptr(),
                1,
                naxis as c_long,
                naxes.as_ptr(),
                fpixels.as_ptr(),
                lpixels.as_ptr(),
                imgarray2.as_ptr().cast::<c_short>(),
                &raw mut status,
            );
            print!("\nWrote subset 2D array: ffpssi status = {status}\n");

            ffg2di(
                fptr.as_mut_ptr(),
                1,
                0,
                19,
                naxes[0] as LONGLONG,
                naxes[1] as LONGLONG,
                imgarray.as_mut_ptr().cast::<c_short>(),
                &raw mut anynull,
                &raw mut status,
            );
            print!("\nRead whole 2D array: ffg2di status = {status}\n");

            for jj in 0..30 {
                for ii in 0..19 {
                    print!(" {:>3}", imgarray[jj][ii]);
                }
                println!();
            }

            fpixels[0] = 2;
            fpixels[1] = 5;
            lpixels[0] = 10;
            lpixels[1] = 8;
            inc[0] = 2;
            inc[1] = 3;

            for jj in 0..30 {
                for ii in 0..19 {
                    imgarray[jj][ii] = 0;
                }
            }

            ffgsvi(
                fptr.as_mut_ptr(),
                1,
                naxis,
                naxes.as_ptr(),
                fpixels.as_ptr(),
                lpixels.as_ptr(),
                inc.as_ptr(),
                0,
                imgarray.as_mut_ptr().cast::<c_short>(),
                &raw mut anynull,
                &raw mut status,
            );
            print!("\nRead subset of 2D array: ffgsvi status = {status}\n");

            for ii in 0..10 {
                print!(" {:>3}", imgarray[0][ii]);
            }
            println!();

            /*
              ###########################################################
              #  insert another image extension                         #
              #  copy the image extension to primary array of tmp file. #
              #  then delete the tmp file, and the image extension      #
              ###########################################################
            */
            bitpix = 16;
            naxis = 2;
            naxes[0] = 15;
            naxes[1] = 25;
            ffiimg(
                fptr.as_mut_ptr(),
                bitpix,
                naxis,
                naxes.as_ptr(),
                &raw mut status,
            );
            print!("\nCreate image extension: ffiimg status = {status}\n");
            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            strcpy_safe(&mut filename, cs!(c"t1q2s3v4.tmp"));
            ffinit(&raw mut tmpfptr, filename.as_ptr(), &raw mut status);
            println!("Create temporary file: ffinit status = {status}");

            ffcopy(fptr.as_mut_ptr(), tmpfptr.as_mut_ptr(), 0, &raw mut status);
            println!("Copy image extension to primary array of tmp file.");
            println!("ffcopy status = {status}");

            ffgrec(tmpfptr.as_mut_ptr(), 1, card.as_mut_ptr(), &raw mut status);
            println!("{}", byte_slice_to_str!(&card));
            ffgrec(tmpfptr.as_mut_ptr(), 2, card.as_mut_ptr(), &raw mut status);
            println!("{}", byte_slice_to_str!(&card));
            ffgrec(tmpfptr.as_mut_ptr(), 3, card.as_mut_ptr(), &raw mut status);
            println!("{}", byte_slice_to_str!(&card));
            ffgrec(tmpfptr.as_mut_ptr(), 4, card.as_mut_ptr(), &raw mut status);
            println!("{}", byte_slice_to_str!(&card));
            ffgrec(tmpfptr.as_mut_ptr(), 5, card.as_mut_ptr(), &raw mut status);
            println!("{}", byte_slice_to_str!(&card));
            ffgrec(tmpfptr.as_mut_ptr(), 6, card.as_mut_ptr(), &raw mut status);
            println!("{}", byte_slice_to_str!(&card));

            let t = Box::into_raw(tmpfptr.take().unwrap());
            ffdelt(t, &raw mut status);
            println!("Delete the tmp file: ffdelt status = {status}");

            ffdhdu(fptr.as_mut_ptr(), &raw mut hdutype, &raw mut status);
            println!("Delete the image extension; hdutype, status = {hdutype} {status}");
            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );

            /*
              ###########################################################
              #  append bintable extension with variable length columns #
              ###########################################################
            */

            ffcrhd(fptr.as_mut_ptr(), &raw mut status);
            println!("ffcrhd status = {status}");

            strcpy(tform[0], c"1PA".as_ptr());
            strcpy(tform[1], c"1PL".as_ptr());
            strcpy(tform[2], c"1PB".as_ptr()); /* Fortran FITSIO doesn't support  1PX */
            strcpy(tform[3], c"1PB".as_ptr());
            strcpy(tform[4], c"1PI".as_ptr());
            strcpy(tform[5], c"1PJ".as_ptr());
            strcpy(tform[6], c"1PE".as_ptr());
            strcpy(tform[7], c"1PD".as_ptr());
            strcpy(tform[8], c"1PC".as_ptr());
            strcpy(tform[9], c"1PM".as_ptr());

            strcpy(ttype[0], c"Avalue".as_ptr());
            strcpy(ttype[1], c"Lvalue".as_ptr());
            strcpy(ttype[2], c"Xvalue".as_ptr());
            strcpy(ttype[3], c"Bvalue".as_ptr());
            strcpy(ttype[4], c"Ivalue".as_ptr());
            strcpy(ttype[5], c"Jvalue".as_ptr());
            strcpy(ttype[6], c"Evalue".as_ptr());
            strcpy(ttype[7], c"Dvalue".as_ptr());
            strcpy(ttype[8], c"Cvalue".as_ptr());
            strcpy(ttype[9], c"Mvalue".as_ptr());

            strcpy(tunit[0], c"".as_ptr());
            strcpy(tunit[1], c"m**2".as_ptr());
            strcpy(tunit[2], c"cm".as_ptr());
            strcpy(tunit[3], c"erg/s".as_ptr());
            strcpy(tunit[4], c"km/s".as_ptr());
            strcpy(tunit[5], c"".as_ptr());
            strcpy(tunit[6], c"".as_ptr());
            strcpy(tunit[7], c"".as_ptr());
            strcpy(tunit[8], c"".as_ptr());
            strcpy(tunit[9], c"".as_ptr());

            nrows = 20;
            tfields = 10;
            pcount = 0;

            ffphbn(
                fptr.as_mut_ptr(),
                nrows,
                tfields,
                ttype.as_ptr().cast::<*const c_char>(),
                tform.as_ptr().cast::<*const c_char>(),
                tunit.as_ptr().cast::<*const c_char>(),
                binname.as_ptr(),
                pcount as LONGLONG,
                &raw mut status,
            );
            println!("Variable length arrays: ffphbn status = {status}");

            extvers = 4;
            ffpkyj(
                fptr.as_mut_ptr(),
                c"EXTVER".as_ptr(),
                extvers as LONGLONG,
                c"extension version number".as_ptr(),
                &raw mut status,
            );

            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL4".as_ptr(),
                88,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL5".as_ptr(),
                88,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );
            ffpkyj(
                fptr.as_mut_ptr(),
                c"TNULL6".as_ptr(),
                88,
                c"value for undefined pixels".as_ptr(),
                &raw mut status,
            );

            /*
              ############################
              #  write data to columns   #
              ############################
            */

            /* initialize arrays of values to write to table */
            strcpy_safe(&mut iskey, cs!(c"abcdefghijklmnopqrst"));

            for ii in 0..20 {
                boutarray[ii] = (ii + 1) as c_uchar;
                ioutarray[ii] = (ii + 1) as c_short;
                joutarray[ii] = (ii + 1) as c_long;
                eoutarray[ii] = (ii + 1) as f32;
                doutarray[ii] = (ii + 1) as f64;
            }

            larray[0] = 0;
            larray[1] = 1;
            larray[2] = 0;
            larray[3] = 0;
            larray[4] = 1;
            larray[5] = 1;
            larray[6] = 0;
            larray[7] = 0;
            larray[8] = 0;
            larray[9] = 1;
            larray[10] = 1;
            larray[11] = 1;
            larray[12] = 0;
            larray[13] = 0;
            larray[14] = 0;
            larray[15] = 0;
            larray[16] = 1;
            larray[17] = 1;
            larray[18] = 1;
            larray[19] = 1;

            /* write values in 1st row */
            /*  strncpy_safe(&mut inskey[0], iskey, 1); */
            *inskey[0] = 0; /* write a null string (i.e., a blank) */
            ffpcls(
                fptr.as_mut_ptr(),
                1,
                1,
                1,
                1,
                inskey.as_ptr().cast::<*const c_char>(),
                &raw mut status,
            ); /* write string values */
            ffpcll(
                fptr.as_mut_ptr(),
                2,
                1,
                1,
                1,
                larray.as_ptr(),
                &raw mut status,
            ); /* write logicals */
            ffpclx(
                fptr.as_mut_ptr(),
                3,
                1,
                1,
                1,
                larray.as_ptr(),
                &raw mut status,
            ); /* write bits */
            ffpclb(
                fptr.as_mut_ptr(),
                4,
                1,
                1,
                1,
                boutarray.as_ptr(),
                &raw mut status,
            );
            ffpcli(
                fptr.as_mut_ptr(),
                5,
                1,
                1,
                1,
                ioutarray.as_ptr(),
                &raw mut status,
            );
            ffpclj(
                fptr.as_mut_ptr(),
                6,
                1,
                1,
                1,
                joutarray.as_ptr(),
                &raw mut status,
            );
            ffpcle(
                fptr.as_mut_ptr(),
                7,
                1,
                1,
                1,
                eoutarray.as_ptr(),
                &raw mut status,
            );
            ffpcld(
                fptr.as_mut_ptr(),
                8,
                1,
                1,
                1,
                doutarray.as_ptr(),
                &raw mut status,
            );

            /* loop over rows 1 - 20 */
            for ii in 2..=20 as LONGLONG {
                let inskey_item = slice::from_raw_parts_mut(inskey[0], FLEN_VALUE);
                strncpy_safe(inskey_item, &iskey, ii as usize);
                inskey_item[ii as usize] = 0;
                ffpcls(
                    fptr.as_mut_ptr(),
                    1,
                    ii,
                    1,
                    1,
                    inskey.as_ptr().cast::<*const c_char>(),
                    &raw mut status,
                ); /* write string values */

                ffpcll(
                    fptr.as_mut_ptr(),
                    2,
                    ii,
                    1,
                    ii,
                    larray.as_ptr(),
                    &raw mut status,
                ); /* write logicals */
                ffpclu(fptr.as_mut_ptr(), 2, ii, ii - 1, 1, &raw mut status);

                ffpclx(
                    fptr.as_mut_ptr(),
                    3,
                    ii,
                    1,
                    ii as c_long,
                    larray.as_ptr(),
                    &raw mut status,
                ); /* write bits */

                ffpclb(
                    fptr.as_mut_ptr(),
                    4,
                    ii,
                    1,
                    ii,
                    boutarray.as_ptr(),
                    &raw mut status,
                );
                ffpclu(fptr.as_mut_ptr(), 4, ii, ii - 1, 1, &raw mut status);

                ffpcli(
                    fptr.as_mut_ptr(),
                    5,
                    ii,
                    1,
                    ii,
                    ioutarray.as_ptr(),
                    &raw mut status,
                );
                ffpclu(fptr.as_mut_ptr(), 5, ii, ii - 1, 1, &raw mut status);

                ffpclj(
                    fptr.as_mut_ptr(),
                    6,
                    ii,
                    1,
                    ii,
                    joutarray.as_ptr(),
                    &raw mut status,
                );
                ffpclu(fptr.as_mut_ptr(), 6, ii, ii - 1, 1, &raw mut status);

                ffpcle(
                    fptr.as_mut_ptr(),
                    7,
                    ii,
                    1,
                    ii,
                    eoutarray.as_ptr(),
                    &raw mut status,
                );
                ffpclu(fptr.as_mut_ptr(), 7, ii, ii - 1, 1, &raw mut status);

                ffpcld(
                    fptr.as_mut_ptr(),
                    8,
                    ii,
                    1,
                    ii,
                    doutarray.as_ptr(),
                    &raw mut status,
                );
                ffpclu(fptr.as_mut_ptr(), 8, ii, ii - 1, 1, &raw mut status);
            }
            println!("ffpcl_ status = {status}");

            /*
              #################################
              #  close then reopen this HDU   #
              #################################
            */

            ffmrhd(fptr.as_mut_ptr(), -1, &raw mut hdutype, &raw mut status);
            ffmrhd(fptr.as_mut_ptr(), 1, &raw mut hdutype, &raw mut status);

            /*
              #############################
              #  read data from columns   #
              #############################
            */

            ffgkyj(
                fptr.as_mut_ptr(),
                c"PCOUNT".as_ptr(),
                &raw mut pcount,
                comm.as_mut_ptr(),
                &raw mut status,
            );
            println!("PCOUNT = {pcount}");

            /* initialize the variables to be read */
            strcpy(inskey[0], c" ".as_ptr());
            strcpy_safe(&mut iskey, cs!(c" "));

            println!(
                "HDU number = {}",
                ffghdn(fptr.as_mut_ptr(), &raw mut hdunum)
            );
            for ii in 1..=20 as LONGLONG
            /* loop over rows 1 - 20 */
            {
                for jj in 0..ii as usize {
                    larray[jj] = 0;
                    boutarray[jj] = 0;
                    ioutarray[jj] = 0;
                    joutarray[jj] = 0;
                    eoutarray[jj] = 0.0;
                    doutarray[jj] = 0.0;
                }

                ffgcvs(
                    fptr.as_mut_ptr(),
                    1,
                    ii,
                    1,
                    1,
                    iskey.as_ptr(),
                    inskey.as_mut_ptr(),
                    &raw mut anynull,
                    &raw mut status,
                );
                print!(
                    "A {} {}\nL",
                    CStr::from_ptr(inskey[0]).to_str().unwrap(),
                    status
                );

                ffgcl(
                    fptr.as_mut_ptr(),
                    2,
                    ii,
                    1,
                    ii,
                    larray.as_mut_ptr(),
                    &raw mut status,
                );
                for jj in 0..ii as usize {
                    print!(" {:>2}", larray[jj]);
                }
                print!(" {status}\nX");

                ffgcx(
                    fptr.as_mut_ptr(),
                    3,
                    ii,
                    1,
                    ii,
                    larray.as_mut_ptr(),
                    &raw mut status,
                );
                for jj in 0..ii as usize {
                    print!(" {:>2}", larray[jj]);
                }
                print!(" {status}\nB");

                ffgcvb(
                    fptr.as_mut_ptr(),
                    4,
                    ii,
                    1,
                    ii,
                    99,
                    boutarray.as_mut_ptr(),
                    &raw mut anynull,
                    &raw mut status,
                );
                for jj in 0..ii as usize {
                    print!(" {:>2}", boutarray[jj]);
                }
                print!(" {status}\nI");

                ffgcvi(
                    fptr.as_mut_ptr(),
                    5,
                    ii,
                    1,
                    ii,
                    99,
                    ioutarray.as_mut_ptr(),
                    &raw mut anynull,
                    &raw mut status,
                );
                for jj in 0..ii as usize {
                    print!(" {:>2}", ioutarray[jj]);
                }
                print!(" {status}\nJ");

                ffgcvj(
                    fptr.as_mut_ptr(),
                    6,
                    ii,
                    1,
                    ii,
                    99,
                    joutarray.as_mut_ptr(),
                    &raw mut anynull,
                    &raw mut status,
                );
                for jj in 0..ii as usize {
                    print!(" {:>2}", joutarray[jj]);
                }
                print!(" {status}\nE");

                ffgcve(
                    fptr.as_mut_ptr(),
                    7,
                    ii,
                    1,
                    ii,
                    99.,
                    eoutarray.as_mut_ptr(),
                    &raw mut anynull,
                    &raw mut status,
                );
                for jj in 0..ii as usize {
                    print!(" {:2.0}", eoutarray[jj]);
                }
                print!(" {status}\nD");

                ffgcvd(
                    fptr.as_mut_ptr(),
                    8,
                    ii,
                    1,
                    ii,
                    99.,
                    doutarray.as_mut_ptr(),
                    &raw mut anynull,
                    &raw mut status,
                );
                for jj in 0..ii as usize {
                    print!(" {:2.0}", doutarray[jj]);
                }
                println!(" {status}");

                ffgdes(
                    fptr.as_mut_ptr(),
                    8,
                    ii,
                    &raw mut repeat,
                    &raw mut offset,
                    &raw mut status,
                );
                println!("Column 8 repeat and offset = {repeat} {offset}");
            }

            /*
              #####################################
              #  create another image extension   #
              #####################################
            */

            bitpix = 32;
            naxis = 2;
            naxes[0] = 10;
            naxes[1] = 2;
            npixels = 20;

            /*    ffcrim(fptr.as_mut_ptr(), bitpix, naxis, naxes, &mut status); */
            ffiimg(
                fptr.as_mut_ptr(),
                bitpix,
                naxis,
                naxes.as_ptr(),
                &raw mut status,
            );
            print!("\nffcrim status = {status}\n");

            /* initialize arrays of values to write to primary array */
            for ii in 0..npixels as c_int {
                boutarray[ii as usize] = (ii * 2) as c_uchar;
                ioutarray[ii as usize] = (ii * 2) as c_short;
                joutarray[ii as usize] = (ii * 2) as c_long;
                koutarray[ii as usize] = ii * 2;
                eoutarray[ii as usize] = (ii * 2) as f32;
                doutarray[ii as usize] = (ii * 2) as f64;
            }

            /* write a few pixels with each datatype */
            ffppr(
                fptr.as_mut_ptr(),
                TBYTE,
                1,
                2,
                boutarray[0..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            ffppr(
                fptr.as_mut_ptr(),
                TSHORT,
                3,
                2,
                ioutarray[2..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            ffppr(
                fptr.as_mut_ptr(),
                TINT,
                5,
                2,
                koutarray[4..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            ffppr(
                fptr.as_mut_ptr(),
                TSHORT,
                7,
                2,
                ioutarray[6..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            ffppr(
                fptr.as_mut_ptr(),
                TLONG,
                9,
                2,
                joutarray[8..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            ffppr(
                fptr.as_mut_ptr(),
                TFLOAT,
                11,
                2,
                eoutarray[10..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            ffppr(
                fptr.as_mut_ptr(),
                TDOUBLE,
                13,
                2,
                doutarray[12..].as_ptr().cast::<c_void>(),
                &raw mut status,
            );
            println!("ffppr status = {status}");

            /* read back the pixels with each datatype */
            bnul = 0;
            inul = 0;
            knul = 0;
            jnul = 0;
            enul = 0.0;
            dnul = 0.0;

            ffgpv(
                fptr.as_mut_ptr(),
                TBYTE,
                1,
                14,
                core::ptr::from_mut(&mut bnul).cast::<c_void>(),
                binarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgpv(
                fptr.as_mut_ptr(),
                TSHORT,
                1,
                14,
                core::ptr::from_mut(&mut inul).cast::<c_void>(),
                iinarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgpv(
                fptr.as_mut_ptr(),
                TINT,
                1,
                14,
                core::ptr::from_mut(&mut knul).cast::<c_void>(),
                kinarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgpv(
                fptr.as_mut_ptr(),
                TLONG,
                1,
                14,
                core::ptr::from_mut(&mut jnul).cast::<c_void>(),
                jinarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgpv(
                fptr.as_mut_ptr(),
                TFLOAT,
                1,
                14,
                core::ptr::from_mut(&mut enul).cast::<c_void>(),
                einarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgpv(
                fptr.as_mut_ptr(),
                TDOUBLE,
                1,
                14,
                core::ptr::from_mut(&mut dnul).cast::<c_void>(),
                dinarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nImage values written with ffppr and read with ffgpv:\n");
            npixels = 14;
            for item in &binarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (byte)");
            for item in &iinarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (short)");
            for item in &kinarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (int)");
            for item in &jinarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (long)");
            for item in &einarray[..npixels as usize] {
                print!(" {item:2.0}");
            }
            println!("  {anynull} (float)");
            for item in &dinarray[..npixels as usize] {
                print!(" {item:2.0}");
            }
            println!("  {anynull} (double)");

            /*
              ##########################################
              #  test world coordinate system routines #
              ##########################################
            */

            xrval = 45.83;
            yrval = 63.57;
            xrpix = 256.0;
            yrpix = 257.0;
            xinc = -0.00277777;
            yinc = 0.00277777;

            /* write the WCS keywords */
            /* use example values from the latest WCS document */
            ffpkyd(
                fptr.as_mut_ptr(),
                c"CRVAL1".as_ptr(),
                xrval,
                10,
                c"comment".as_ptr(),
                &raw mut status,
            );
            ffpkyd(
                fptr.as_mut_ptr(),
                c"CRVAL2".as_ptr(),
                yrval,
                10,
                c"comment".as_ptr(),
                &raw mut status,
            );
            ffpkyd(
                fptr.as_mut_ptr(),
                c"CRPIX1".as_ptr(),
                xrpix,
                10,
                c"comment".as_ptr(),
                &raw mut status,
            );
            ffpkyd(
                fptr.as_mut_ptr(),
                c"CRPIX2".as_ptr(),
                yrpix,
                10,
                c"comment".as_ptr(),
                &raw mut status,
            );
            ffpkyd(
                fptr.as_mut_ptr(),
                c"CDELT1".as_ptr(),
                xinc,
                10,
                c"comment".as_ptr(),
                &raw mut status,
            );
            ffpkyd(
                fptr.as_mut_ptr(),
                c"CDELT2".as_ptr(),
                yinc,
                10,
                c"comment".as_ptr(),
                &raw mut status,
            );
            /*   ffpkyd(fptr.as_mut_ptr(), "CROTA2", rot, 10, "comment", &mut status); */
            ffpkys(
                fptr.as_mut_ptr(),
                c"CTYPE1".as_ptr(),
                xcoordtype.as_ptr(),
                c"comment".as_ptr(),
                &raw mut status,
            );
            ffpkys(
                fptr.as_mut_ptr(),
                c"CTYPE2".as_ptr(),
                ycoordtype.as_ptr(),
                c"comment".as_ptr(),
                &raw mut status,
            );
            print!("\nWrote WCS keywords status = {status}\n");

            xrval = 0.0;
            yrval = 0.0;
            xrpix = 0.0;
            yrpix = 0.0;
            xinc = 0.0;
            yinc = 0.0;
            rot = 0.0;

            ffgics(
                fptr.as_mut_ptr(),
                &raw mut xrval,
                &raw mut yrval,
                &raw mut xrpix,
                &raw mut yrpix,
                &raw mut xinc,
                &raw mut yinc,
                &raw mut rot,
                ctype.as_mut_ptr().cast::<[c_char; 5]>(),
                &raw mut status,
            );
            println!("Read WCS keywords with ffgics status = {status}");

            xpix = 0.5;
            ypix = 0.5;

            ffwldp(
                xpix,
                ypix,
                xrval,
                yrval,
                xrpix,
                yrpix,
                xinc,
                yinc,
                rot,
                ctype.as_ptr().cast::<[c_char; 5]>(),
                &raw mut xpos,
                &raw mut ypos,
                &raw mut status,
            );

            println!("  CRVAL1, CRVAL2 = {xrval:>16.12}, {yrval:>16.12}");
            println!("  CRPIX1, CRPIX2 = {xrpix:>16.12}, {yrpix:>16.12}");
            println!("  CDELT1, CDELT2 = {xinc:>16.12}, {yinc:>16.12}");
            println!(
                "  Rotation = {:>10.3}, CTYPE = {}",
                rot,
                byte_slice_to_str!(&ctype)
            );
            println!("Calculated sky coordinate with ffwldp status = {status}");
            println!("  Pixels ({xpix:>8.4},{ypix:>8.4}) --> ({xpos:>11.6}, {ypos:>11.6}) Sky");
            ffxypx(
                xpos,
                ypos,
                xrval,
                yrval,
                xrpix,
                yrpix,
                xinc,
                yinc,
                rot,
                ctype.as_ptr().cast::<[c_char; 5]>(),
                &raw mut xpix,
                &raw mut ypix,
                &raw mut status,
            );
            println!("Calculated pixel coordinate with ffxypx status = {status}");
            println!("  Sky ({xpos:>11.6}, {ypos:>11.6}) --> ({xpix:>8.4},{ypix:>8.4}) Pixels");
            /*
              ######################################
              #  append another ASCII table        #
              ######################################
            */

            strcpy(tform[0], c"A15".as_ptr());
            strcpy(tform[1], c"I11".as_ptr());
            strcpy(tform[2], c"F15.6".as_ptr());
            strcpy(tform[3], c"E13.5".as_ptr());
            strcpy(tform[4], c"D22.14".as_ptr());

            strcpy(ttype[0], c"Name".as_ptr());
            strcpy(ttype[1], c"Ivalue".as_ptr());
            strcpy(ttype[2], c"Fvalue".as_ptr());
            strcpy(ttype[3], c"Evalue".as_ptr());
            strcpy(ttype[4], c"Dvalue".as_ptr());

            strcpy(tunit[0], c"".as_ptr());
            strcpy(tunit[1], c"m**2".as_ptr());
            strcpy(tunit[2], c"cm".as_ptr());
            strcpy(tunit[3], c"erg/s".as_ptr());
            strcpy(tunit[4], c"km/s".as_ptr());

            nrows = 11;
            tfields = 5;
            strcpy_safe(&mut tblname, cs!(c"new_table"));

            ffcrtb(
                fptr.as_mut_ptr(),
                ASCII_TBL,
                nrows,
                tfields,
                ttype.as_ptr().cast::<*const c_char>(),
                tform.as_ptr().cast::<*const c_char>(),
                tunit.as_ptr().cast::<*const c_char>(),
                tblname.as_ptr(),
                &raw mut status,
            );
            print!("\nffcrtb status = {status}\n");

            extvers = 5;
            ffpkyj(
                fptr.as_mut_ptr(),
                c"EXTVER".as_ptr(),
                extvers as LONGLONG,
                c"extension version number".as_ptr(),
                &raw mut status,
            );

            ffpcl(
                fptr.as_mut_ptr(),
                TSTRING,
                1,
                1,
                1,
                3,
                onskey.as_ptr().cast::<c_void>(),
                &raw mut status,
            ); /* write string values */

            /* initialize arrays of values to write */

            for ii in 0..npixels as usize {
                boutarray[ii] = (ii * 3) as c_uchar;
                ioutarray[ii] = (ii * 3) as c_short;
                joutarray[ii] = (ii * 3) as c_long;
                koutarray[ii] = (ii * 3) as c_int;
                eoutarray[ii] = (ii * 3) as f32;
                doutarray[ii] = (ii * 3) as f64;
            }

            for ii in 2..6
            /* loop over cols 2 - 5 */
            {
                ffpcl(
                    fptr.as_mut_ptr(),
                    TBYTE,
                    ii,
                    1,
                    1,
                    2,
                    boutarray.as_ptr().cast::<c_void>(),
                    &raw mut status,
                );
                ffpcl(
                    fptr.as_mut_ptr(),
                    TSHORT,
                    ii,
                    3,
                    1,
                    2,
                    ioutarray[2..].as_ptr().cast::<c_void>(),
                    &raw mut status,
                );
                ffpcl(
                    fptr.as_mut_ptr(),
                    TLONG,
                    ii,
                    5,
                    1,
                    2,
                    joutarray[4..].as_ptr().cast::<c_void>(),
                    &raw mut status,
                );
                ffpcl(
                    fptr.as_mut_ptr(),
                    TFLOAT,
                    ii,
                    7,
                    1,
                    2,
                    eoutarray[6..].as_ptr().cast::<c_void>(),
                    &raw mut status,
                );
                ffpcl(
                    fptr.as_mut_ptr(),
                    TDOUBLE,
                    ii,
                    9,
                    1,
                    2,
                    doutarray[8..].as_ptr().cast::<c_void>(),
                    &raw mut status,
                );
            }
            println!("ffpcl status = {status}");

            /* read back the pixels with each datatype */
            ffgcv(
                fptr.as_mut_ptr(),
                TBYTE,
                2,
                1,
                1,
                10,
                core::ptr::from_mut(&mut bnul).cast::<c_void>(),
                binarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcv(
                fptr.as_mut_ptr(),
                TSHORT,
                2,
                1,
                1,
                10,
                core::ptr::from_mut(&mut inul).cast::<c_void>(),
                iinarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcv(
                fptr.as_mut_ptr(),
                TINT,
                3,
                1,
                1,
                10,
                core::ptr::from_mut(&mut knul).cast::<c_void>(),
                kinarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcv(
                fptr.as_mut_ptr(),
                TLONG,
                3,
                1,
                1,
                10,
                core::ptr::from_mut(&mut jnul).cast::<c_void>(),
                jinarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcv(
                fptr.as_mut_ptr(),
                TFLOAT,
                4,
                1,
                1,
                10,
                core::ptr::from_mut(&mut enul).cast::<c_void>(),
                einarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );
            ffgcv(
                fptr.as_mut_ptr(),
                TDOUBLE,
                5,
                1,
                1,
                10,
                core::ptr::from_mut(&mut dnul).cast::<c_void>(),
                dinarray.as_mut_ptr().cast::<c_void>(),
                &raw mut anynull,
                &raw mut status,
            );

            print!("\nColumn values written with ffpcl and read with ffgcl:\n");
            npixels = 10;
            for item in &binarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (byte)");

            for item in &iinarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (short)");

            for item in &kinarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (int)");

            for item in &jinarray[..npixels as usize] {
                print!(" {item:>2}");
            }
            println!("  {anynull} (long)");

            for item in &einarray[..npixels as usize] {
                print!(" {item:2.0}");
            }
            println!("  {anynull} (float)");

            for item in &dinarray[..npixels as usize] {
                print!(" {item:2.0}");
            }
            println!("  {anynull} (double)");

            /*
              ###########################################################
              #  perform stress test by cycling thru all the extensions #
              ###########################################################
            */
            print!("\nRepeatedly move to the 1st 4 HDUs of the file:\n");
            for _ii in 0..10 {
                ffmahd(fptr.as_mut_ptr(), 1, &raw mut hdutype, &raw mut status);
                print!("{}", ffghdn(fptr.as_mut_ptr(), &raw mut hdunum));
                ffmrhd(fptr.as_mut_ptr(), 1, &raw mut hdutype, &raw mut status);
                print!("{}", ffghdn(fptr.as_mut_ptr(), &raw mut hdunum));
                ffmrhd(fptr.as_mut_ptr(), 1, &raw mut hdutype, &raw mut status);
                print!("{}", ffghdn(fptr.as_mut_ptr(), &raw mut hdunum));
                ffmrhd(fptr.as_mut_ptr(), 1, &raw mut hdutype, &raw mut status);
                print!("{}", ffghdn(fptr.as_mut_ptr(), &raw mut hdunum));
                ffmrhd(fptr.as_mut_ptr(), -1, &raw mut hdutype, &raw mut status);
                print!("{}", ffghdn(fptr.as_mut_ptr(), &raw mut hdunum));
                if status > 0 {
                    break;
                }
            }
            println!();

            println!("Move to extensions by name and version number: (ffmnhd)");
            extvers = 1;
            ffmnhd(
                fptr.as_mut_ptr(),
                ANY_HDU,
                binname.as_ptr(),
                extvers as c_int,
                &raw mut status,
            );
            ffghdn(fptr.as_mut_ptr(), &raw mut hdunum);
            println!(
                " {}, {} = hdu {}, {}",
                byte_slice_to_str!(&binname),
                extvers,
                hdunum,
                status
            );
            extvers = 3;
            ffmnhd(
                fptr.as_mut_ptr(),
                ANY_HDU,
                binname.as_ptr(),
                extvers as c_int,
                &raw mut status,
            );
            ffghdn(fptr.as_mut_ptr(), &raw mut hdunum);
            println!(
                " {}, {} = hdu {}, {}",
                byte_slice_to_str!(&binname),
                extvers,
                hdunum,
                status
            );
            extvers = 4;
            ffmnhd(
                fptr.as_mut_ptr(),
                ANY_HDU,
                binname.as_ptr(),
                extvers as c_int,
                &raw mut status,
            );
            ffghdn(fptr.as_mut_ptr(), &raw mut hdunum);
            println!(
                " {}, {} = hdu {}, {}",
                byte_slice_to_str!(&binname),
                extvers,
                hdunum,
                status
            );

            strcpy_safe(&mut tblname, cs!(c"Test-ASCII"));
            extvers = 2;
            ffmnhd(
                fptr.as_mut_ptr(),
                ANY_HDU,
                tblname.as_ptr(),
                extvers as c_int,
                &raw mut status,
            );
            ffghdn(fptr.as_mut_ptr(), &raw mut hdunum);
            println!(
                " {}, {} = hdu {}, {}",
                byte_slice_to_str!(&tblname),
                extvers,
                hdunum,
                status
            );

            strcpy_safe(&mut tblname, cs!(c"new_table"));
            extvers = 5;
            ffmnhd(
                fptr.as_mut_ptr(),
                ANY_HDU,
                tblname.as_ptr(),
                extvers as c_int,
                &raw mut status,
            );
            ffghdn(fptr.as_mut_ptr(), &raw mut hdunum);
            println!(
                " {}, {} = hdu {}, {}",
                byte_slice_to_str!(&tblname),
                extvers,
                hdunum,
                status
            );
            extvers = 0;
            ffmnhd(
                fptr.as_mut_ptr(),
                ANY_HDU,
                binname.as_ptr(),
                extvers as c_int,
                &raw mut status,
            );
            ffghdn(fptr.as_mut_ptr(), &raw mut hdunum);
            println!(
                " {}, {} = hdu {}, {}",
                byte_slice_to_str!(&binname),
                extvers,
                hdunum,
                status
            );
            extvers = 17;
            ffmnhd(
                fptr.as_mut_ptr(),
                ANY_HDU,
                binname.as_ptr(),
                extvers as c_int,
                &raw mut status,
            );
            ffghdn(fptr.as_mut_ptr(), &raw mut hdunum);
            print!(
                " {}, {} = hdu {}, {}",
                byte_slice_to_str!(&binname),
                extvers,
                hdunum,
                status
            );
            println!(" (expect a 301 error status here)");
            status = 0;

            ffthdu(fptr.as_mut_ptr(), &raw mut hdunum, &raw mut status);
            println!("Total number of HDUs in the file = {hdunum}");
            /*
              ########################
              #  checksum tests      #
              ########################
            */
            checksum = 1234567890;
            ffesum(checksum, 0, asciisum.as_mut_ptr().cast::<[c_char; 17]>());
            print!(
                "\nEncode checksum: {} -> {}\n",
                checksum,
                byte_slice_to_str!(&asciisum)
            );
            checksum = 0;
            ffdsum(asciisum.as_ptr(), 0, &raw mut checksum);
            println!(
                "Decode checksum: {} -> {}",
                byte_slice_to_str!(&asciisum),
                checksum
            );

            ffpcks(fptr.as_mut_ptr(), &raw mut status);

            /*
               don't print the CHECKSUM value because it is different every day
               because the current date is in the comment field.

               ffgcrd(fptr.as_mut_ptr(), "CHECKSUM", card, &mut status);
               print!("{}\n", byte_slice_to_str!(&card));
            */

            ffgcrd(
                fptr.as_mut_ptr(),
                c"DATASUM".as_ptr(),
                card.as_mut_ptr(),
                &raw mut status,
            );
            println!("{:.30}", byte_slice_to_str!(&card));

            ffgcks(
                fptr.as_mut_ptr(),
                &raw mut datsum,
                &raw mut checksum,
                &raw mut status,
            );
            println!("ffgcks data checksum, status = {datsum}, {status}");

            ffvcks(
                fptr.as_mut_ptr(),
                &raw mut datastatus,
                &raw mut hdustatus,
                &raw mut status,
            );
            println!("ffvcks datastatus, hdustatus, status = {datastatus} {hdustatus} {status}");

            ffprec(
                fptr.as_mut_ptr(),
                c"new_key = 'written by fxprec' / to change checksum".as_ptr(),
                &raw mut status,
            );
            ffupck(fptr.as_mut_ptr(), &raw mut status);
            println!("ffupck status = {status}");

            ffgcrd(
                fptr.as_mut_ptr(),
                c"DATASUM".as_ptr(),
                card.as_mut_ptr(),
                &raw mut status,
            );
            println!("{:.30}", byte_slice_to_str!(&card));
            ffvcks(
                fptr.as_mut_ptr(),
                &raw mut datastatus,
                &raw mut hdustatus,
                &raw mut status,
            );
            println!("ffvcks datastatus, hdustatus, status = {datastatus} {hdustatus} {status}");

            /*
              delete the checksum keywords, so that the FITS file is always
              the same, regardless of the date of when testprog is run.
            */

            ffdkey(fptr.as_mut_ptr(), c"CHECKSUM".as_ptr(), &raw mut status);
            ffdkey(fptr.as_mut_ptr(), c"DATASUM".as_ptr(), &raw mut status);

            break;
        }

        /*
          ############################
          #  close file and quit     #
          ############################
        */

        // errstatus: Jump here on error or completion
        let f = fptr.take();
        ffclos(f, &raw mut status);
        println!("ffclos status = {status}");

        println!();
        println!("Normally, there should be 8 error messages on the stack");
        println!("all regarding 'numerical overflows':");

        ffgmsg(errmsg.as_mut_ptr());
        nmsg = 0;

        while errmsg[0] != 0 {
            println!(" {}", byte_slice_to_str!(&errmsg));
            nmsg += 1;
            ffgmsg(errmsg.as_mut_ptr());
        }

        if nmsg != 8 {
            println!();
            println!("WARNING: Did not find the expected 8 error messages!");
        }

        ffgerr(status, errmsg.as_mut_ptr());
        println!();
        println!("Status = {}: {}", status, byte_slice_to_str!(&errmsg));

        /* free the allocated memory */
        for item in inskey {
            free(item.cast::<c_void>());
        }

        for item in ttype {
            free(item.cast::<c_void>());
        }

        for item in tform {
            free(item.cast::<c_void>());
        }

        for item in tunit {
            free(item.cast::<c_void>());
        }
    }

    // Some error codes can be > 255 so this may fail
    ExitCode::from(u8::try_from(status).unwrap())
}
