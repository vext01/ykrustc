//! Serialised Intermediate Representation (SIR).
//!
//! SIR is built in-memory during code-generation (in rustc_codegen_ssa), and finally placed
//! into an ELF section at link time.

use crate::mir;
use crate::ty::{self, Instance, Ty, TyCtxt};
use indexmap::IndexMap;
use rustc_ast::ast;
use rustc_data_structures::fx::FxHasher;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_session::config::OutputType;
use rustc_span::sym;
use std::cell::RefCell;
use std::convert::TryFrom;
use std::default::Default;
use std::hash::BuildHasherDefault;
use std::io;
use ykpack;

const BUILD_SCRIPT_CRATE: &str = "build_script_build";
const CHECKABLE_BINOPS: [ykpack::BinOp; 5] = [
    ykpack::BinOp::Add,
    ykpack::BinOp::Sub,
    ykpack::BinOp::Mul,
    ykpack::BinOp::Shl,
    ykpack::BinOp::Shr,
];

// Generates a big `match` statement for the binary operation lowerings.
macro_rules! binop_lowerings {
    ( $the_op:expr, $($op:ident ),* ) => {
        match $the_op {
            $(mir::BinOp::$op => ykpack::BinOp::$op,)*
        }
    }
}

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
            && tcx.crate_name(LOCAL_CRATE).as_str() != BUILD_SCRIPT_CRATE
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
pub struct SirFuncCx<'tcx> {
    pub func: ykpack::Body,
    tcx: TyCtxt<'tcx>,
    pub trace_inputs_defid: DefId,
}

impl SirFuncCx<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &mir::Body<'_>) -> Self {
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
        let blocks = vec![
            ykpack::BasicBlock {
                stmts: Default::default(),
                term: ykpack::Terminator::Unreachable,
            };
            mir.basic_blocks().len()
        ];

        let trace_inputs_defid = tcx
            .get_lang_items(LOCAL_CRATE)
            .yk_trace_inputs()
            .expect("couldn't find trace inputs lang item");

        // FIXME get rid of the num_locals field.
        let num_locals = mir.local_decls.len();
        let local_decls = Vec::with_capacity(num_locals);

        let symbol_name = String::from(&*tcx.symbol_name(*instance).name);
        Self {
            func: ykpack::Body {
                symbol_name,
                blocks,
                flags,
                num_locals,
                trace_inputs_local: None,
                local_decls,
            },
            tcx,
            trace_inputs_defid,
        }
    }

    /// Returns true if there are no basic blocks.
    pub fn is_empty(&self) -> bool {
        self.func.blocks.len() == 0
    }

    /// Appends a statement to the specified basic block.
    fn push_stmt(&mut self, bb: ykpack::BasicBlockIndex, stmt: ykpack::Statement) {
        self.func.blocks[usize::try_from(bb).unwrap()].stmts.push(stmt);
    }

    /// Sets the terminator of the specified block.
    pub fn set_terminator(&mut self, bb: ykpack::BasicBlockIndex, new_term: ykpack::Terminator) {
        let term = &mut self.func.blocks[usize::try_from(bb).unwrap()].term;
        // We should only ever replace the default unreachable terminator assigned at allocation time.
        debug_assert!(*term == ykpack::Terminator::Unreachable);
        *term = new_term
    }

    /// Converts a MIR statement to SIR, appending the result to `bb`.
    pub fn codegen_statement(&mut self, bb: ykpack::BasicBlockIndex, stmt: &mir::Statement<'_>) {
        match stmt.kind {
            mir::StatementKind::Assign(box (ref place, ref rvalue)) => {
                let assign = self.lower_assign_stmt(place, rvalue);
                self.push_stmt(bb, assign);
            }
            mir::StatementKind::StorageLive(l) => {
                self.push_stmt(bb, ykpack::Statement::StorageLive(self.lower_local(l)))
            }
            mir::StatementKind::StorageDead(l) => {
                self.push_stmt(bb, ykpack::Statement::StorageDead(self.lower_local(l)))
            }
            _ => self.push_stmt(bb, ykpack::Statement::Unimplemented(format!("{:?}", stmt))),
        }
    }

    fn lower_assign_stmt(
        &self,
        lvalue: &mir::Place<'_>,
        rvalue: &mir::Rvalue<'_>,
    ) -> ykpack::Statement {
        let lhs = self.lower_place(lvalue);
        let rhs = self.lower_rvalue(rvalue);
        ykpack::Statement::Assign(lhs, rhs)
    }

    pub fn lower_operand(&self, operand: &mir::Operand<'_>) -> ykpack::Operand {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                ykpack::Operand::Place(self.lower_place(place))
            }
            mir::Operand::Constant(cst) => ykpack::Operand::Constant(self.lower_constant(cst)),
        }
    }

    fn lower_rvalue(&self, rvalue: &mir::Rvalue<'_>) -> ykpack::Rvalue {
        match rvalue {
            mir::Rvalue::Use(opnd) => ykpack::Rvalue::Use(self.lower_operand(opnd)),
            mir::Rvalue::BinaryOp(op, opnd1, opnd2) => self.lower_binop(*op, opnd1, opnd2, false),
            mir::Rvalue::CheckedBinaryOp(op, opnd1, opnd2) => {
                self.lower_binop(*op, opnd1, opnd2, true)
            }
            _ => ykpack::Rvalue::Unimplemented(format!("unimplemented rvalue: {:?}", rvalue)),
        }
    }

    pub fn lower_place(&self, place: &mir::Place<'_>) -> ykpack::Place {
        ykpack::Place {
            local: self.lower_local(place.local),
            // FIXME projections not yet implemented.
            projection: place.projection.iter().map(|p| self.lower_projection(&p)).collect(),
        }
    }

    pub fn lower_projection(&self, pe: &mir::PlaceElem<'_>) -> ykpack::Projection {
        match pe {
            mir::ProjectionElem::Field(field, ..) => ykpack::Projection::Field(field.as_u32()),
            _ => ykpack::Projection::Unimplemented(format!("{:?}", pe)),
        }
    }

    pub fn lower_local(&self, local: mir::Local) -> ykpack::Local {
        // For the lowering of `Local`s we currently assume a 1:1 mapping from MIR to SIR. If this
        // mapping turns out to be impossible or impractial, this is the place to change it.
        ykpack::Local(local.as_u32())
    }

    fn lower_constant(&self, constant: &mir::Constant<'_>) -> ykpack::Constant {
        match constant.literal.val {
            ty::ConstKind::Value(mir::interpret::ConstValue::Scalar(s)) => {
                self.lower_scalar(constant.literal.ty, s)
            }
            _ => ykpack::Constant::Unimplemented(format!("unimplemented constant: {:?}", constant)),
        }
    }

    fn lower_scalar(&self, ty: Ty<'_>, s: mir::interpret::Scalar) -> ykpack::Constant {
        match ty.kind {
            ty::Uint(uint) => self
                .lower_uint(uint, s)
                .map(|i| ykpack::Constant::Int(ykpack::ConstantInt::UnsignedInt(i)))
                .unwrap_or_else(|_| {
                    ykpack::Constant::Unimplemented(format!(
                        "unimplemented uint scalar: {:?}",
                        ty.kind
                    ))
                }),
            ty::Int(int) => self
                .lower_int(int, s)
                .map(|i| ykpack::Constant::Int(ykpack::ConstantInt::SignedInt(i)))
                .unwrap_or_else(|_| {
                    ykpack::Constant::Unimplemented(format!(
                        "unimplemented signed int scalar: {:?}",
                        ty.kind
                    ))
                }),
            ty::Bool => self.lower_bool(s),
            _ => ykpack::Constant::Unimplemented(format!("unimplemented scalar: {:?}", ty.kind)),
        }
    }

    /// Lower an unsigned integer.
    fn lower_uint(
        &self,
        uint: ast::UintTy,
        s: mir::interpret::Scalar,
    ) -> Result<ykpack::UnsignedInt, ()> {
        match uint {
            ast::UintTy::U8 => match s.to_u8() {
                Ok(val) => Ok(ykpack::UnsignedInt::U8(val)),
                Err(e) => panic!("Could not lower scalar to u8: {}", e),
            },
            ast::UintTy::U16 => match s.to_u16() {
                Ok(val) => Ok(ykpack::UnsignedInt::U16(val)),
                Err(e) => panic!("Could not lower scalar to u16: {}", e),
            },
            ast::UintTy::U32 => match s.to_u32() {
                Ok(val) => Ok(ykpack::UnsignedInt::U32(val)),
                Err(e) => panic!("Could not lower scalar to u32: {}", e),
            },
            ast::UintTy::U64 => match s.to_u64() {
                Ok(val) => Ok(ykpack::UnsignedInt::U64(val)),
                Err(e) => panic!("Could not lower scalar to u64: {}", e),
            },
            ast::UintTy::Usize => match s.to_machine_usize(&self.tcx) {
                Ok(val) => Ok(ykpack::UnsignedInt::Usize(val as usize)),
                Err(e) => panic!("Could not lower scalar to usize: {}", e),
            },
            _ => Err(()),
        }
    }

    /// Lower a signed integer.
    fn lower_int(
        &self,
        int: ast::IntTy,
        s: mir::interpret::Scalar,
    ) -> Result<ykpack::SignedInt, ()> {
        match int {
            ast::IntTy::I8 => match s.to_i8() {
                Ok(val) => Ok(ykpack::SignedInt::I8(val)),
                Err(e) => panic!("Could not lower scalar to i8: {}", e),
            },
            ast::IntTy::I16 => match s.to_i16() {
                Ok(val) => Ok(ykpack::SignedInt::I16(val)),
                Err(e) => panic!("Could not lower scalar to i16: {}", e),
            },
            ast::IntTy::I32 => match s.to_i32() {
                Ok(val) => Ok(ykpack::SignedInt::I32(val)),
                Err(e) => panic!("Could not lower scalar to i32: {}", e),
            },
            ast::IntTy::I64 => match s.to_i64() {
                Ok(val) => Ok(ykpack::SignedInt::I64(val)),
                Err(e) => panic!("Could not lower scalar to i64: {}", e),
            },
            ast::IntTy::Isize => match s.to_machine_isize(&self.tcx) {
                Ok(val) => Ok(ykpack::SignedInt::Isize(val as isize)),
                Err(e) => panic!("Could not lower scalar to isize: {}", e),
            },
            _ => Err(()),
        }
    }
    fn lower_binop(
        &self,
        op: mir::BinOp,
        opnd1: &mir::Operand<'_>,
        opnd2: &mir::Operand<'_>,
        checked: bool,
    ) -> ykpack::Rvalue {
        let sir_op = binop_lowerings!(
            op, Add, Sub, Mul, Div, Rem, BitXor, BitAnd, BitOr, Shl, Shr, Eq, Lt, Le, Ne, Ge, Gt,
            Offset
        );
        let sir_opnd1 = self.lower_operand(opnd1);
        let sir_opnd2 = self.lower_operand(opnd2);

        if checked {
            debug_assert!(CHECKABLE_BINOPS.contains(&sir_op));
            ykpack::Rvalue::CheckedBinaryOp(sir_op, sir_opnd1, sir_opnd2)
        } else {
            ykpack::Rvalue::BinaryOp(sir_op, sir_opnd1, sir_opnd2)
        }
    }

    fn lower_bool(&self, s: mir::interpret::Scalar) -> ykpack::Constant {
        match s.to_bool() {
            Ok(val) => ykpack::Constant::Bool(val),
            Err(e) => panic!("Could not lower scalar (bool) to u8: {}", e),
        }
    }
}

#[derive(Default)]
pub struct SirTypes {
    /// Maps types to their index. Ordered by insertion via `IndexMap`.
    pub map: IndexMap<ykpack::Ty, ykpack::TyIndex, BuildHasherDefault<FxHasher>>,
    /// The next available type index.
    next_idx: ykpack::TyIndex,
}

impl SirTypes {
    /// Get the index of a type. If this is the first time we have seen this type, a new index is
    /// allocated and returned.
    ///
    /// Note that the index is only unique within the scope of the crate currently being compiled.
    /// To make a globally unique type index, pair the index with crate hash.
    pub fn index(&mut self, t: ykpack::Ty) -> ykpack::TyIndex {
        let next_idx = &mut self.next_idx;
        *self.map.entry(t).or_insert_with(|| {
            let idx = *next_idx;
            *next_idx += 1;
            idx
        })
    }
}
