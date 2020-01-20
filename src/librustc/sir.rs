//! Serialised Intermediate Representation (SIR).
//!
//! SIR is built in-memory during LLVM code-generation, and finally placed into a dedicated  ELF
//! section at link time.

use std::default::Default;
use std::io::{self, Write};
use std::fs::File;
use crate::session::config::{OutputFilenames, OutputType};
use rustc_index::{newtype_index, vec::{Idx, IndexVec}};
use rustc_data_structures::fx::FxHashMap;
use ykpack;

// Duplicates of LLVM types defined elsewhere, copied to avoid cyclic dependencies. Whereas the
// LLVM backend expresses pointers to these using references, we use raw pointers so to as avoid
// introducing lifetime parameters to the SirCx (and thus into TyCtxt and every place that uses
// it).
extern { pub type Value; }
extern { pub type BasicBlock; }

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

/// Sir equivalents of LLVM values.
#[derive(Debug)]
pub enum SirValue {
    Func(SirFuncIdx),
}

impl SirValue {
    pub fn func_idx(&self) -> SirFuncIdx {
        let Self::Func(idx) = self;
        *idx
    }
}

pub struct SirCx {
    /// Maps an opaque LLVM `Value` to its SIR equivalent.
    pub llvm_values: FxHashMap<*const Value, SirValue>,
    /// Maps an opaque LLVM `BasicBlock` to the function and block index of it's SIR equivalent.
    pub llvm_blocks: FxHashMap<*const BasicBlock, (SirFuncIdx, SirBlockIdx)>,
    /// Function store. Also owns the blocks
    pub funcs: IndexVec<SirFuncIdx, ykpack::Body>,
}

impl SirCx {
    pub fn new() -> Self {
        Self {
            llvm_values: Default::default(),
            llvm_blocks: FxHashMap::default(),
            funcs: Default::default(),
        }
    }

    pub fn add_func(&mut self, value: *const Value, symbol_name: String) {
        let idx = SirFuncIdx::from_usize(self.funcs.len());

        self.funcs.push(ykpack::Body{
            symbol_name,
            blocks: Default::default(),
            num_locals: 0,  // FIXME
            num_args: 0,    // FIXME
            flags: 0,       // Set later.
        });
        let existing = self.llvm_values.insert(value, SirValue::Func(idx));
        debug_assert!(existing.is_none());
    }

    pub fn add_block(&mut self, func: *const Value, block: *const BasicBlock) {
        let func_idx = self.llvm_values[&func].func_idx();
        let sir_func = &mut self.funcs[func_idx];
        let block_idx = SirBlockIdx::from_usize(sir_func.blocks.len());
        sir_func.blocks.push(ykpack::BasicBlock{
            stmts: Default::default(),
            term: ykpack::Terminator::Unreachable, // FIXME
        });
        let existing = self.llvm_blocks.insert(block, (func_idx, block_idx));
        debug_assert!(existing.is_none());
    }

    /// Dump SIR to text file.
    /// Used in tests and for debugging.
    pub fn dump(&self, outputs: &OutputFilenames) -> Result<(), io::Error> {
        let mut file = File::create(outputs.path(OutputType::YkSir))?;

        for func in &self.funcs {
            writeln!(file, "{}", func)?;
        }

        Ok(())
    }
}
