//! FIXME

#![allow(dead_code)]

use std::default::Default;
use rustc_index::{newtype_index, vec::{Idx, IndexVec}};
use rustc_data_structures::fx::FxHashMap;
use crate::value::Value;
use crate::llvm::BasicBlock;

newtype_index! {
    pub struct SirFuncIdx {
        DEBUG_FORMAT = "SirFuncIdx({})"
    }
}

newtype_index! {
    pub struct SirBlockIdx {
        DEBUG_FORMAT = "SirBlockIdx({})"
    }
}

#[derive(Debug)]
pub enum SirValue {
    Func(SirFuncIdx),
    // more will appear...
}

impl SirValue {
    fn func_idx(&self) -> SirFuncIdx {
        let Self::Func(idx) = self;
        *idx
    }
}

// FIXME This will need to be shared via ykpack.
#[derive(Debug)]
pub struct SirFunc {
    symbol_name: String,
    blocks: Vec<SirBlockIdx>,
}

// FIXME This will need to be shared via ykpack.
#[derive(Debug)]
pub struct SirBlock {}

pub struct SirCx<'ll> {
    /// Maps an opaque LLVM `Value` to its SIR equivalent.
    llvm_values: FxHashMap<&'ll Value, SirValue>,
    llvm_blocks: FxHashMap<&'ll BasicBlock, SirBlockIdx>,

    pub funcs: IndexVec<SirFuncIdx, SirFunc>,
    pub blocks: IndexVec<SirBlockIdx, SirBlock>,
}

impl SirCx<'ll> {
    pub fn new() -> Self {
        Self {
            llvm_values: Default::default(),
            llvm_blocks: FxHashMap::default(),
            funcs: Default::default(),
            blocks: Default::default(),
        }
    }

    pub fn add_func(&mut self, value: &'ll Value, symbol_name: String) {
        let idx = SirFuncIdx::from_usize(self.funcs.len());
        self.funcs.push(SirFunc{
            symbol_name,
            blocks: Default::default(),
        });
        let existing = self.llvm_values.insert(value, SirValue::Func(idx));
        debug_assert!(existing.is_none());
    }

    pub fn add_block(&mut self, func: &'ll Value, block: &'ll BasicBlock) {
        let idx = SirBlockIdx::from_usize(self.blocks.len());
        self.blocks.push(SirBlock{});
        let sir_func = &mut self.funcs[self.llvm_values[func].func_idx()];
        sir_func.blocks.push(idx);
        let existing = self.llvm_blocks.insert(block, idx);
        debug_assert!(existing.is_none());
    }
}
