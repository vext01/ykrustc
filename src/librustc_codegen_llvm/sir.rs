//! Serialised Intermideiate Representation (SIR).
//!
//! SIR is built in-memory during LLVM code-generation, and finally placed into an ELF section at
//! link time.

#![allow(dead_code, unused_imports)]

use std::default::Default;
use rustc_index::{newtype_index, vec::{Idx, IndexVec}};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::ty::TyCtxt;
use rustc::session::config::OutputType;
use crate::value::Value;
use crate::llvm::{self, BasicBlock};
use crate::{common, ModuleLlvm};
use std::ffi::CString;
use ykpack;

const SIR_SECTION: &str = ".yk_sir";

/// Writes the SIR into a buffer which will be linked in into an ELF section via LLVM.
/// This is based on write_compressed_metadata().
pub fn write_sir<'tcx>(
    tcx: TyCtxt<'tcx>,
    llvm_module: &mut ModuleLlvm,
) {
    let (sir_llcx, sir_llmod) = (&*llvm_module.llcx, llvm_module.llmod());

    // If we are dumping SIR, we are not actually generating code, so serialisation isn't
    // necessary. We also leave the sir_cx in place for the dumping code to pick up.
    if tcx.sess.opts.output_types.contains_key(&OutputType::YkSir) {
        return;
    }

    // Note that we move the sir_cx out so that we serialise without copying.
    if let Some(sir_cx) = tcx.sir_cx.replace(None) {
        let mut buf = Vec::new();
        let mut ec = ykpack::Encoder::from(&mut buf);

        for func in sir_cx.funcs {
            ec.serialise(ykpack::Pack::Body(func)).unwrap();
        }

        ec.done().unwrap();

        let llmeta = common::bytes_in_context(sir_llcx, &buf);
        let llconst = common::struct_in_context(sir_llcx, &[llmeta], false);

        //// Borrowed from exported_symbols::metadata_symbol_name().
        let sym_name = format!("yksir_{}_{}",
            tcx.original_crate_name(LOCAL_CRATE),
            tcx.crate_disambiguator(LOCAL_CRATE).to_fingerprint().to_hex());

        let buf = CString::new(sym_name).unwrap();
        let llglobal = unsafe {
            llvm::LLVMAddGlobal(sir_llmod, common::val_ty(llconst), buf.as_ptr())
        };

        unsafe {
            llvm::LLVMSetInitializer(llglobal, llconst);
            let name = SmallCStr::new(SIR_SECTION);
            llvm::LLVMSetSection(llglobal, name.as_ptr());

            // Following the precedent of write_compressed_metadata(), force empty flags so that
            // the SIR doesn't get loaded into memory.
            let directive = format!(".section {}", SIR_SECTION);
            let directive = CString::new(directive).unwrap();
            llvm::LLVMSetModuleInlineAsm(sir_llmod, directive.as_ptr())
        }
    }
}
