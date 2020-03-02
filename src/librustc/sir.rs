#![allow(dead_code)]

use std::default::Default;
use crate::ty::{TyCtxt, Instance};
use rustc_session::config::OutputType;
use ykpack;
use std::cell::RefCell;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_span::sym;
use std::io;

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
        };

        Self {
            func: ykpack::Body {
                symbol_name,
                blocks: Default::default(),
                flags,
            }
        }
    }

    pub fn add_block(&mut self) {
        self.func.blocks.push(ykpack::BasicBlock {
            stmts: Default::default(),
            term: ykpack::Terminator::Unreachable,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.func.blocks.len() == 0
    }
}
