#![allow(dead_code)]

use std::default::Default;
use ykpack;

pub struct SirFuncCx {
    pub func: ykpack::Body,
}

impl SirFuncCx {
    pub fn new(symbol_name: String) -> Self {
        Self {
            func: ykpack::Body {
                symbol_name,
                blocks: Default::default(),
                flags: 0,
            }
        }
    }

    pub fn add_block(&mut self) {
        self.func.blocks.push(ykpack::BasicBlock {
            stmts: Default::default(),
            term: ykpack::Terminator::Unreachable,
        });
    }

    pub fn encode(self) {
        unimplemented!();
    }
}
