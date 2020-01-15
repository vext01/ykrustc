//! FIXME

#![allow(dead_code)]

use std::default::Default;
use rustc_index::{newtype_index, vec::{Idx, IndexVec}};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::ty::TyCtxt;
use crate::value::Value;
use crate::llvm::{self, BasicBlock};
use crate::{common, ModuleLlvm};
use std::ffi::CString;
use ykpack;

const SIR_SECTION: &str = ".yksir";

newtype_index! {
    pub struct SirFuncIdx {
        DEBUG_FORMAT = "SirFuncIdx({})"
    }
}

// The index of a block within a function.
// Note that these indices are not globally unique. For a globally unique block identifier, a
// (SirFuncIdx, SirBlockIdx) pair must be used.
newtype_index! {
    pub struct SirBlockIdx {
        DEBUG_FORMAT = "SirBlockIdx({})"
    }
}

#[derive(Debug)]
pub enum SirValue {
    Func(SirFuncIdx),
}

impl SirValue {
    fn func_idx(&self) -> SirFuncIdx {
        let Self::Func(idx) = self;
        *idx
    }
}

pub struct SirCx<'ll> {
    /// Maps an opaque LLVM `Value` to its SIR equivalent.
    llvm_values: FxHashMap<&'ll Value, SirValue>,
    /// Maps an opaque LLVM `BasicBlock` to the function and block index of it's SIR equivalent.
    llvm_blocks: FxHashMap<&'ll BasicBlock, (SirFuncIdx, SirBlockIdx)>,
    /// Function store. Also owns the blocks
    pub funcs: IndexVec<SirFuncIdx, ykpack::Body>,
}

impl SirCx<'ll> {
    pub fn new() -> Self {
        Self {
            llvm_values: Default::default(),
            llvm_blocks: FxHashMap::default(),
            funcs: Default::default(),
        }
    }

    pub fn add_func(&mut self, value: &'ll Value, symbol_name: String) {
        let idx = SirFuncIdx::from_usize(self.funcs.len());
        self.funcs.push(ykpack::Body{
            symbol_name,
            blocks: Default::default(),
            num_locals: 0,  // FIXME
            num_args: 0,    // FIXME
            flags: 0,       // FIXME
        });
        let existing = self.llvm_values.insert(value, SirValue::Func(idx));
        debug_assert!(existing.is_none());
    }

    pub fn add_block(&mut self, func: &'ll Value, block: &'ll BasicBlock) {
        let func_idx = self.llvm_values[func].func_idx();
        let sir_func = &mut self.funcs[func_idx];
        let block_idx = SirBlockIdx::from_usize(sir_func.blocks.len());
        sir_func.blocks.push(ykpack::BasicBlock{
            stmts: Default::default(),
        });
        let existing = self.llvm_blocks.insert(block, (func_idx, block_idx));
        debug_assert!(existing.is_none());
    }

    pub fn serialise_into_tcx(self, tcx: TyCtxt<'tcx>) {
        let mut buf = Vec::new();
        let mut ec = ykpack::Encoder::from(&mut buf);

        for func in self.funcs {
            ec.serialise(ykpack::Pack::Body(func)).unwrap();
        }

        ec.done().unwrap();
        tcx.encoded_sir.borrow_mut().push(buf);
    }
}

/// Writes the SIR into a buffer which will be linked in into an ELF section via LLVM.
/// This is based on write_compressed_metadata().
pub fn write_sir<'tcx>(
    tcx: TyCtxt<'tcx>,
    llvm_module: &mut ModuleLlvm,
) {
    let (sir_llcx, sir_llmod) = (&*llvm_module.llcx, llvm_module.llmod());

    // Steal the SIR from the tcx and move it into a buffer for LLVM to include.
    let sir = tcx.encoded_sir.replace(Default::default());
    let mut buf = Vec::new();
    for cg_sir in sir.into_iter() {
        debug!("CG sir of length: {}", cg_sir.len());
        buf.extend(cg_sir);
    }

    // FIXME number of collections needs to be encoded.
    let llmeta = common::bytes_in_context(sir_llcx, &buf);
    let llconst = common::struct_in_context(sir_llcx, &[llmeta], false);

    // Borrowed from exported_symbols::metadata_symbol_name().
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

        // Force no flags, so that the SIR doesn't get loaded into memory.
        let directive = format!(".section {}", SIR_SECTION);
        let directive = CString::new(directive).unwrap();
        llvm::LLVMSetModuleInlineAsm(sir_llmod, directive.as_ptr())
    }
}
