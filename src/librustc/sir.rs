//! Serialised Intermediate Representation (SIR).
//!
//! SIR is built in-memory during LLVM code-generation (in rustc_codegen_ssa), and finally placed
//! into an ELF section at link time. Each crate has one such ELF section.

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

/// A collection of in-memory SIR data structures to be serialised.
/// Each codegen unit builds one instance of this which is then merged into a "global" instance
/// when the unit completes.
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

    /// Returns true if there is nothing inside.
    pub fn is_empty(&self) -> bool {
        self.funcs.borrow().len() == 0
    }

    /// Merges the SIR in `other` into `self`, consuming `other`.
    pub fn update(&self, other: Self) {
        self.funcs.borrow_mut().extend(other.funcs.into_inner());
    }

    /// Writes a textual representation of the SIR to `w`. Used for `--emit yk-sir`.
    pub fn dump(&self, tcx: TyCtxt<'_>, w: &mut dyn io::Write) -> Result<(), io::Error> {
        for f in tcx.sir.funcs.borrow().iter() {
            writeln!(w, "{}", f)?;
        }
        Ok(())
    }
}

/// A structure for building the SIR of a function.
pub struct SirFuncCx {
    pub func: ykpack::Body,
}

impl SirFuncCx {
   pub fn new<'tcx>(tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, num_blocks: usize) -> Self {
       let symbol_name = format!("new__{}", tcx.symbol_name(*instance).name.as_str());

       let mut flags = 0;
       for attr in tcx.get_attrs(instance.def_id()).iter() {
           if attr.check_name(sym::trace_head) {
               flags |= ykpack::bodyflags::TRACE_HEAD;
           } else if attr.check_name(sym::trace_tail) {
               flags |= ykpack::bodyflags::TRACE_TAIL;
           }
       }

       // Since there's a one-to-one mapping between MIR and SIR blocks, we know how many SIR
       // blocks we will need and can allocate empty SIR blocks ahead of time.
       let blocks = vec![ykpack::BasicBlock {
           stmts: Default::default(),
           term: ykpack::Terminator::Unreachable,
       }; num_blocks];

       Self { func: ykpack::Body { symbol_name, blocks, flags } }
   }

   /// Returns true if there are no basic blocks.
   pub fn is_empty(&self) -> bool {
       self.func.blocks.len() == 0
   }

   /// Appends a statement to the specified basic block.
   fn append_stmt(&mut self, bb: ykpack::BasicBlockIndex, stmt: ykpack::Statement) {
       self.func.blocks[usize::try_from(bb).unwrap()].stmts.push(stmt);
   }

   /// Sets the terminator of the specified block.
   pub fn codegen_terminator(&mut self, bb: ykpack::BasicBlockIndex, mir_term: &mir::Terminator<'_>) {
       let term = &mut self.func.blocks[usize::try_from(bb).unwrap()].term;
       // We should only ever replace the default unreachable terminator assigned at allocation time.
       debug_assert!(*term == ykpack::Terminator::Unreachable);

       // FIXME: Nothing is implemented yet.
       *term = ykpack::Terminator::Unimplemented(format!("{:?}", mir_term.kind));
   }

   /// Converts a MIR statement to SIR, appending the result to `bb`.
   pub fn codegen_statement(&mut self, bb: ykpack::BasicBlockIndex, stmt: &mir::Statement<'_>) {
       // FIXME: Nothing is implemented yet.
       self.append_stmt(bb, ykpack::Statement::Unimplemented(format!("{:?}", stmt)));
   }
}
