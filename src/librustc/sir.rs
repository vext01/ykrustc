#![allow(dead_code)]

use crate::ty::{Instance, TyCtxt};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_session::config::OutputType;
use rustc_span::sym;
use rustc::mir;
use std::cell::RefCell;
use std::default::Default;
use std::io;
use std::convert::TryFrom;
use ykpack;

#[derive(Default)]
pub struct Sir {
    pub funcs: RefCell<Vec<ykpack::Body>>,
}

impl Sir {
    /// Returns `true` if we should collect SIR for the current crate.
    pub fn is_required(tcx: TyCtxt<'_>) -> bool {
        (tcx.sess.opts.cg.tracer.encode_sir()
            || tcx.sess.opts.output_types.contains_key(&OutputType::YkSir))
            && tcx.crate_name(LOCAL_CRATE).as_str() != "build_script_build"
    }

    pub fn is_empty(&self) -> bool {
        self.funcs.borrow().len() == 0
    }

    pub fn update(&self, other: Self) {
        self.funcs.borrow_mut().extend(other.funcs.into_inner());
    }

    pub fn dump(&self, tcx: TyCtxt<'_>, w: &mut dyn io::Write) -> Result<(), io::Error> {
        for f in tcx.sir.funcs.borrow().iter() {
            writeln!(w, "{}", f)?;
        }
        Ok(())
    }
}

pub struct SirFuncCx {
    pub func: ykpack::Body,
}

impl SirFuncCx {
   pub fn new<'tcx>(tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>) -> Self {
       let symbol_name = format!("new__{}", tcx.symbol_name(*instance).name.as_str());

       let mut flags = 0;
       for attr in tcx.get_attrs(instance.def_id()).iter() {
           if attr.check_name(sym::trace_head) {
               flags |= ykpack::bodyflags::TRACE_HEAD;
           } else if attr.check_name(sym::trace_tail) {
               flags |= ykpack::bodyflags::TRACE_TAIL;
           }
       }

       Self { func: ykpack::Body { symbol_name, blocks: Default::default(), flags } }
   }

   pub fn add_block(&mut self) -> ykpack::BasicBlockIndex {
       dbg!("len is {}",  self.func.blocks.len());
       let idx = self.func.blocks.len();
       self.func.blocks.push(ykpack::BasicBlock {
           stmts: Default::default(),
           term: ykpack::Terminator::Unreachable,
       });
       ykpack::BasicBlockIndex::try_from(idx).unwrap()
   }

   pub fn is_empty(&self) -> bool {
       self.func.blocks.len() == 0
   }

   /// Appends a statement to the specified basic block.
   fn append_stmt(&mut self, bb: ykpack::BasicBlockIndex, stmt: ykpack::Statement) {
       self.func.blocks[usize::try_from(bb).unwrap()].stmts.push(stmt);
   }

   /// Converts a MIR statement to SIR, appending the result to `bb`.
   pub fn codegen_statement(&mut self, bb: ykpack::BasicBlockIndex, stmt: &mir::Statement<'_>) {
       // FIXME: Nothing is implemented yet.
       self.append_stmt(bb, ykpack::Statement::Unimplemented(format!("{:?}", stmt)));
   }
}
