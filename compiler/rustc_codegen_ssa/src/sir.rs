//! Serialised Intermediate Representation (SIR).
//!
//! SIR is built in-memory during code-generation (in rustc_codegen_ssa), and finally placed
//! into an ELF section at link time.

#![allow(dead_code, unused_variables, unused_macros, unused_imports)]

use crate::mir::LocalRef;
use crate::traits::{BuilderMethods, SirMethods};
use indexmap::IndexMap;
use rustc_ast::ast;
use rustc_ast::ast::{IntTy, UintTy};
use rustc_data_structures::fx::{FxHashMap, FxHasher};
use rustc_hir::{self, def_id::LOCAL_CRATE};
use rustc_middle::mir;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::AdtDef;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, layout::TyAndLayout, TyCtxt};
use rustc_middle::ty::{Instance, Ty};
use rustc_span::sym;
use rustc_target::abi::FieldsShape;
use rustc_target::abi::VariantIdx;
use std::cell::RefCell;
use std::convert::TryFrom;
use std::default::Default;
use std::hash::{BuildHasherDefault, Hash, Hasher};
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
pub struct Sir {
    pub types: RefCell<SirTypes>,
    pub funcs: RefCell<Vec<ykpack::Body>>,
}

impl Sir {
    pub fn new(tcx: TyCtxt<'_>, cgu_name: &str) -> Self {
        // Build the CGU hash.
        //
        // This must be a globally unique hash for this compilation unit. It might have been
        // tempting to use the `tcx.crate_hash()` as part of the CGU hash, but this query is
        // invalidated on every source code change to the crate. In turn, that would mean lots of
        // unnecessary rebuilds.
        //
        // We settle on:
        // CGU hash = crate name + crate disambiguator + codegen unit name.
        let mut cgu_hasher = FxHasher::default();
        tcx.crate_name(LOCAL_CRATE).hash(&mut cgu_hasher);
        tcx.crate_disambiguator(LOCAL_CRATE).hash(&mut cgu_hasher);
        cgu_name.hash(&mut cgu_hasher);

        Sir {
            types: RefCell::new(SirTypes {
                cgu_hash: ykpack::CguHash(cgu_hasher.finish()),
                map: Default::default(),
                next_idx: Default::default(),
            }),
            funcs: Default::default(),
        }
    }

    /// Returns `true` if we should collect SIR for the current crate.
    pub fn is_required(tcx: TyCtxt<'_>) -> bool {
        tcx.sess.opts.cg.tracer.encode_sir()
            && tcx.crate_name(LOCAL_CRATE).as_str() != BUILD_SCRIPT_CRATE
    }

    /// Returns true if there is nothing inside.
    pub fn is_empty(&self) -> bool {
        self.funcs.borrow().len() == 0
    }

    /// Writes a textual representation of the SIR to `w`. Used for `--emit yk-sir`.
    pub fn dump(&self, w: &mut dyn io::Write) -> Result<(), io::Error> {
        for f in self.funcs.borrow().iter() {
            writeln!(w, "{}", f)?;
        }
        Ok(())
    }
}

/// A structure for building the SIR of a function.
pub struct SirFuncCx<'tcx> {
    /// The instance we are lowering.
    instance: Instance<'tcx>,
    /// The MIR body of the above instance.
    mir: &'tcx mir::Body<'tcx>,
    /// The SIR function we are building.
    pub func: ykpack::Body,
    /// Maps each MIR local to a SIR IPlace.
    var_map: FxHashMap<mir::Local, ykpack::IPlace>,
    /// The next SIR local variable index to be allocated.
    next_sir_local: ykpack::LocalIndex,
    /// The compiler's type context.
    tcx: TyCtxt<'tcx>,
}

impl SirFuncCx<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &'tcx mir::Body<'tcx>) -> Self {
        let mut flags = 0;
        for attr in tcx.get_attrs(instance.def_id()).iter() {
            if tcx.sess.check_name(attr, sym::do_not_trace) {
                flags |= ykpack::bodyflags::DO_NOT_TRACE;
            } else if tcx.sess.check_name(attr, sym::interp_step) {
                // Check various properties of the interp_step at compile time.
                if mir.args_iter().count() != 1 {
                    tcx.sess
                        .struct_err("The #[interp_step] function must accept only one argument")
                        .emit();
                }

                let arg_ok = if let ty::Ref(_, inner_ty, rustc_hir::Mutability::Mut) =
                    mir.local_decls[mir::Local::from_u32(1)].ty.kind()
                {
                    if let ty::Adt(def, _) = inner_ty.kind() { def.is_struct() } else { false }
                } else {
                    false
                };
                if !arg_ok {
                    tcx.sess
                        .struct_err(
                            "The #[interp_step] function must accept a mutable reference to a struct"
                        )
                        .emit();
                }

                if !mir.return_ty().is_unit() {
                    tcx.sess.struct_err("The #[interp_step] function must return unit").emit();
                }

                if !tcx.upvars_mentioned(instance.def_id()).is_none() {
                    tcx.sess
                        .struct_err(
                            "The #[interp_step] function must not capture from its environment",
                        )
                        .emit();
                }

                flags |= ykpack::bodyflags::INTERP_STEP;
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

        // There will be at least as many locals in the SIR as there are in the MIR.
        let local_decls = Vec::with_capacity(mir.local_decls.len());
        let symbol_name = String::from(&*tcx.symbol_name(*instance).name);

        let crate_name = tcx.crate_name(instance.def_id().krate).as_str();
        if crate_name == "core" || crate_name == "alloc" {
            flags |= ykpack::bodyflags::DO_NOT_TRACE;
        }
        let var_map: FxHashMap<mir::Local, ykpack::IPlace> = FxHashMap::default();

        Self {
            instance: instance.clone(),
            mir,
            func: ykpack::Body { symbol_name, blocks, flags, local_decls, num_args: mir.arg_count },
            var_map,
            next_sir_local: 0,
            tcx,
        }
    }

    /// Returns the SIR local corresponding with MIR local `ml`. A new SIR local is allocated if
    /// we've never seen this MIR local before.
    fn sir_local<Bx: BuilderMethods<'a, 'tcx>>(&mut self, bx: &Bx, ml: &mir::Local) -> ykpack::IPlace {
        if let Some(ip) = self.var_map.get(ml) {
            ip.clone()
        } else {
            let sirty = self.lower_ty_and_layout(bx, &self.layout_of(bx, self.mir.local_decls[*ml].ty));
            let nl = self.new_sir_local(bx, sirty, false);
            self.var_map.insert(*ml, nl.clone());
            nl
        }
    }

    /// Returns a zero-offset IPlace for a new SIR local.
    fn new_sir_local<Bx: BuilderMethods<'a, 'tcx>>(&mut self, bx: &Bx, sirty: ykpack::TypeId, is_ref: bool) -> ykpack::IPlace {
        if self.next_sir_local == 0 {
            // This is the first time we have allocated SIR locals. The return place and the
            // argument locals should come first, so let's allocate them right now.
            for idx in 0..=self.mir.arg_count {
                let ml = mir::Local::from_usize(idx);
                let sirty = self.lower_ty_and_layout(bx, &self.layout_of(bx, self.mir.local_decls[ml].ty));
                self.func.local_decls.push(ykpack::LocalDecl{ty: sirty});
                self.var_map.insert(ml, ykpack::IPlace::Val{
                    local: ykpack::Local(u32::try_from(idx).unwrap()),
                    offs: 0,
                    ty: sirty,
                });
                dbg!(ml, idx);
                self.next_sir_local += 1;
            }
        }

        let idx = self.next_sir_local;
        self.next_sir_local += 1;
        self.func.local_decls.push(ykpack::LocalDecl{ty: sirty});
        if is_ref {
            todo!();
            //ykpack::IPlace::Ref{local: ykpack::Local(idx), offs: 0, ty: sirty}
        } else {
            ykpack::IPlace::Val{local: ykpack::Local(idx), offs: 0, ty: sirty}
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
    pub fn lower_statement<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        stmt: &mir::Statement<'tcx>,
    ) {
        self.push_stmt(bb, ykpack::Statement::Debug(with_no_trimmed_paths(|| format!("{:?}", stmt))));

        match stmt.kind {
            mir::StatementKind::Assign(box (ref place, ref rvalue)) => {
                let assign = self.lower_assign_stmt(bx, bb, place, rvalue);
            }
            // We compute our own liveness in Yorick, so these are ignored.
            mir::StatementKind::StorageLive(_) | mir::StatementKind::StorageDead(_) => {}
            _ => self.push_stmt(bb, ykpack::Statement::Unimplemented(format!("{:?}", stmt))),
        }
    }

    fn lower_assign_stmt<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        lvalue: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
    ) {
        let dest_ty = lvalue.ty(self.mir, self.tcx).ty;
        let rhs = self.lower_rvalue(bx, bb, dest_ty, rvalue);

        if lvalue.projection.is_empty() && lvalue.local != mir::RETURN_PLACE {
            // No need for a store, just update the variable mapping and the corresponding IPlace
            // will appear inline the next time lvalue.local appears in a MIR statement.
            //let slot = self.var_map.get_mut(&lvalue.local).unwrap();
            let old = self.var_map.insert(lvalue.local, rhs);
            debug_assert!(old.is_none()); // Shouldn't be defined yet.
        } else {
            let lhs = self.lower_place(bx, bb, lvalue);
            self.push_stmt(bb, ykpack::Statement::IStore(lhs, rhs));
        }
    }

    pub fn lower_operand<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        operand: &mir::Operand<'tcx>,
    ) -> ykpack::IPlace {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                self.lower_place(bx, bb, place)
            }
            mir::Operand::Constant(cst) => self.lower_constant(bx, bb, cst),
        }
    }

    fn lower_rvalue<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        dest_ty: Ty<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
    ) -> ykpack::IPlace {
        match rvalue {
            mir::Rvalue::Use(opnd) => self.lower_operand(bx, bb, opnd),
            mir::Rvalue::Ref(_, _, p) => self.lower_ref(bx, bb, dest_ty, p),
            mir::Rvalue::CheckedBinaryOp(op, opnd1, opnd2) => self.lower_binop(bx, bb, dest_ty, *op, opnd1, opnd2, true),
            _ => ykpack::IPlace::Unimplemented(with_no_trimmed_paths(|| format!("unimplemented rvalue: {:?}", rvalue))),
        }
    }

    fn monomorphize<T>(&self, value: &T) -> T
    where
        T: TypeFoldable<'tcx> + Copy,
    {
        if let Some(substs) = self.instance.substs_for_mir_body() {
            self.tcx.subst_and_normalize_erasing_regions(substs, ty::ParamEnv::reveal_all(), value)
        } else {
            self.tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), *value)
        }
    }

    /// Wrapper for bx.layout_of() which ensures the type is first monomorphised.
    fn layout_of<Bx: BuilderMethods<'a, 'tcx>>(&self, bx: &Bx, t: Ty<'tcx>) -> TyAndLayout<'tcx> {
        bx.layout_of(self.monomorphize(&t))
    }

    fn offset_iplace<Bx: BuilderMethods<'a, 'tcx>>(&mut self, bx: &Bx, ip: &mut ykpack::IPlace, add: usize, mirty: Ty<'tcx>) {
        match ip {
            ykpack::IPlace::Val{local, offs, ty} => { //| ykpack::IPlace::Ref{local, offs, ty} => {
                *offs += u32::try_from(add).unwrap();
                *ty = self.lower_ty_and_layout(bx, &self.layout_of(bx, mirty));
            },
            ykpack::IPlace::Const{..} => {
                // Oddly enough this happens. It seems to be only ever with an offset of 0 and we
                // assume that the type doesn't change.
                assert_eq!(add, 0);
            }
            ykpack::IPlace::Unimplemented(_) => (),
        }
    }

    pub fn lower_place<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        place: &mir::Place<'_>,
    ) -> ykpack::IPlace {
        // We start with the base local and project away from it.
        let mut cur_iplace = self.sir_local(bx, &place.local);
        let mut cur_mirty = self.monomorphize(&self.mir.local_decls[place.local].ty);

        // Deref has some special rules if it appears at the end of the chain.
        let mut cur_proj_idx = 0;
        let num_projs = place.projection.len();

        // Loop over the projection chain, updating cur_iplace as we go.
        for pj in place.projection {
            let next_mirty = match pj {
                mir::ProjectionElem::Field(f, _) => {
                    let fi = f.as_usize();
                    match cur_mirty.kind() {
                        ty::Adt(def, _) => {
                            if def.is_struct() {
                                let ty_lay = self.layout_of(bx, cur_mirty);
                                let st_lay = ty_lay.for_variant(bx, VariantIdx::from_u32(0));
                                if let FieldsShape::Arbitrary { offsets, .. } = &st_lay.fields {
                                    let new_mirty = st_lay.field(bx, fi).ty;
                                    self.offset_iplace(bx, &mut cur_iplace, offsets[fi].bytes_usize(), new_mirty);
                                    new_mirty
                                } else {
                                    return ykpack::IPlace::Unimplemented(format!(
                                        "struct field shape: {:?}",
                                        st_lay.fields
                                    ));
                                }
                            } else if def.is_enum() {
                                    return ykpack::IPlace::Unimplemented(
                                    format!("enum_projection: {:?}", def));
                            } else {
                                return ykpack::IPlace::Unimplemented(
                                    format!("adt: {:?}", def));
                            }
                        }
                        ty::Tuple(..) => {
                            let tup_lay = self.layout_of(bx, cur_mirty);
                            match &tup_lay.fields {
                                FieldsShape::Arbitrary { offsets, .. } => {
                                    let new_mirty = tup_lay.field(bx, fi).ty;
                                    self.offset_iplace(bx, &mut cur_iplace, offsets[fi].bytes_usize(), new_mirty);
                                    new_mirty
                                }
                                _ => {
                                    return ykpack::IPlace::Unimplemented(format!("tuple field shape: {:?}", tup_lay.fields));
                                }
                            }
                        }
                        _ => return ykpack::IPlace::Unimplemented(format!("field access on: {:?}", cur_mirty)),
                    }
                },
                mir::ProjectionElem::Deref => {
                    if cur_proj_idx == num_projs {
                        // Deref is last in the projection chain, so it's a copying deref.
                        return ykpack::IPlace::Unimplemented(format!("copy deref"));
                    } else {
                        if let ty::Ref(_, ty, _) = cur_mirty.kind() {
                            // Just remove the reference from the type.
                            ty
                        } else {
                            return ykpack::IPlace::Unimplemented(format!("deref non-ref"));
                        }
                    }
                },
                _ => return ykpack::IPlace::Unimplemented(format!("projection: {:?}", pj)),
            };
            cur_mirty = self.monomorphize(&next_mirty);
            cur_proj_idx += 1;
        }
        cur_iplace
    }

    fn lower_constant<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        constant: &mir::Constant<'tcx>,
    ) -> ykpack::IPlace {
        match constant.literal.val {
            ty::ConstKind::Value(mir::interpret::ConstValue::Scalar(s)) => {
                let val = self.lower_scalar(constant.literal.ty, s);
                let ty = self.lower_ty_and_layout(bx, &self.layout_of(bx, constant.literal.ty));
                ykpack::IPlace::Const{val, ty}
            }
            _ => ykpack::IPlace::Unimplemented(with_no_trimmed_paths(|| format!("unimplemented constant: {:?}", constant))),
        }
    }

    fn lower_scalar(&self, ty: Ty<'_>, s: mir::interpret::Scalar) -> ykpack::Constant {
        match ty.kind() {
            ty::Uint(uint) => self
                .lower_uint(*uint, s)
                .map(|i| ykpack::Constant::Int(ykpack::ConstantInt::UnsignedInt(i)))
                .unwrap_or_else(|_| {
                    with_no_trimmed_paths(|| {
                        ykpack::Constant::Unimplemented(format!(
                            "unimplemented uint scalar: {:?}",
                            ty.kind()
                        ))
                    })
                }),
            ty::Int(int) => self
                .lower_int(*int, s)
                .map(|i| ykpack::Constant::Int(ykpack::ConstantInt::SignedInt(i)))
                .unwrap_or_else(|_| {
                    ykpack::Constant::Unimplemented(format!(
                        "unimplemented signed int scalar: {:?}",
                        ty.kind()
                    ))
                }),
            ty::Bool => self.lower_bool(s),
            _ => ykpack::Constant::Unimplemented(format!("unimplemented scalar: {:?}", ty.kind())),
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

    fn lower_binop<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        dest_ty: Ty<'tcx>,
        op: mir::BinOp,
        opnd1: &mir::Operand<'tcx>,
        opnd2: &mir::Operand<'tcx>,
        checked: bool,
    ) -> ykpack::IPlace {
        let op = binop_lowerings!(
            op, Add, Sub, Mul, Div, Rem, BitXor, BitAnd, BitOr, Shl, Shr, Eq, Lt, Le, Ne, Ge, Gt,
            Offset
        );
        let opnd1 = self.lower_operand(bx, bb, opnd1);
        let opnd2 = self.lower_operand(bx, bb, opnd2);

        if checked {
            debug_assert!(CHECKABLE_BINOPS.contains(&op));
        }

        let ty = self.lower_ty_and_layout(bx, &self.layout_of(bx, dest_ty));
        let dest_ip = self.new_sir_local(bx, ty, false);
        let stmt = ykpack::Statement::BinaryOp { dest: dest_ip.clone(), op, opnd1, opnd2, checked };
        self.push_stmt(bb, stmt);

        dest_ip
    }

    fn lower_bool(&self, s: mir::interpret::Scalar) -> ykpack::Constant {
        match s.to_bool() {
            Ok(val) => ykpack::Constant::Bool(val),
            Err(e) => panic!("Could not lower scalar (bool) to u8: {}", e),
        }
    }

    fn lower_ref<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        dest_ty: Ty<'tcx>,
        place: &mir::Place<'tcx>,
    ) -> ykpack::IPlace {
        let ty = self.lower_ty_and_layout(bx, &self.layout_of(bx, dest_ty));
        //let dest_ip = self.new_sir_local(ty, true);
        let dest_ip = self.new_sir_local(bx, ty, false);
        let src_ip = self.lower_place(bx, bb, place);
        let mkref = ykpack::Statement::MkRef(dest_ip.clone(), src_ip);
        self.push_stmt(bb, mkref);
        dest_ip
    }

    fn lower_ty_and_layout<'a, Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        ty_layout: &TyAndLayout<'tcx>,
    ) -> ykpack::TypeId {
        let sir_ty = match ty_layout.ty.kind() {
            ty::Int(si) => self.lower_signed_int_ty(*si),
            ty::Uint(ui) => self.lower_unsigned_int_ty(*ui),
            ty::Adt(adt_def, ..) => self.lower_adt_ty(bx, adt_def, &ty_layout),
            ty::Array(typ, _) => {
                ykpack::Ty::Array(self.lower_ty_and_layout(bx, &self.layout_of(bx, typ)))
            }
            ty::Slice(typ) => {
                ykpack::Ty::Slice(self.lower_ty_and_layout(bx, &self.layout_of(bx, typ)))
            }
            ty::Ref(_, typ, _) => {
                ykpack::Ty::Ref(self.lower_ty_and_layout(bx, &self.layout_of(bx, typ)))
            }
            ty::Bool => ykpack::Ty::Bool,
            ty::Tuple(..) => self.lower_tuple_ty(bx, ty_layout),
            _ => ykpack::Ty::Unimplemented(format!("{:?}", ty_layout)),
        };
        bx.cx().define_sir_type(sir_ty)
    }

    fn lower_signed_int_ty(&mut self, si: IntTy) -> ykpack::Ty {
        match si {
            IntTy::Isize => ykpack::Ty::SignedInt(ykpack::SignedIntTy::Isize),
            IntTy::I8 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I8),
            IntTy::I16 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I16),
            IntTy::I32 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I32),
            IntTy::I64 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I64),
            IntTy::I128 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I128),
        }
    }

    fn lower_unsigned_int_ty(&mut self, ui: UintTy) -> ykpack::Ty {
        match ui {
            UintTy::Usize => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::Usize),
            UintTy::U8 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U8),
            UintTy::U16 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U16),
            UintTy::U32 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U32),
            UintTy::U64 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U64),
            UintTy::U128 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U128),
        }
    }

    fn lower_tuple_ty<'a, Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        ty_layout: &TyAndLayout<'tcx>,
    ) -> ykpack::Ty {
        let align = i32::try_from(ty_layout.layout.align.abi.bytes()).unwrap();
        let size = i32::try_from(ty_layout.layout.size.bytes()).unwrap();

        match &ty_layout.fields {
            FieldsShape::Arbitrary { offsets, .. } => {
                let mut sir_offsets = Vec::new();
                let mut sir_tys = Vec::new();
                for (idx, offs) in offsets.iter().enumerate() {
                    sir_tys.push(self.lower_ty_and_layout(bx, &ty_layout.field(bx, idx)));
                    sir_offsets.push(offs.bytes());
                }

                ykpack::Ty::Tuple(ykpack::TupleTy {
                    fields: ykpack::Fields { offsets: sir_offsets, tys: sir_tys },
                    size_align: ykpack::SizeAndAlign { size, align },
                })
            }
            _ => ykpack::Ty::Unimplemented(format!("{:?}", ty_layout)),
        }
    }

    fn lower_adt_ty<'a, Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        adt_def: &AdtDef,
        ty_layout: &TyAndLayout<'tcx>,
    ) -> ykpack::Ty {
        let align = i32::try_from(ty_layout.layout.align.abi.bytes()).unwrap();
        let size = i32::try_from(ty_layout.layout.size.bytes()).unwrap();

        if adt_def.variants.len() == 1 {
            // Plain old struct-like thing.
            let struct_layout = ty_layout.for_variant(bx, VariantIdx::from_u32(0));

            match &ty_layout.fields {
                FieldsShape::Arbitrary { offsets, .. } => {
                    let mut sir_offsets = Vec::new();
                    let mut sir_tys = Vec::new();
                    for (idx, offs) in offsets.iter().enumerate() {
                        sir_tys.push(self.lower_ty_and_layout(
                            bx,
                            &struct_layout.field(bx, idx),
                        ));
                        sir_offsets.push(offs.bytes());
                    }

                    ykpack::Ty::Struct(ykpack::StructTy {
                        fields: ykpack::Fields { offsets: sir_offsets, tys: sir_tys },
                        size_align: ykpack::SizeAndAlign { align, size },
                    })
                }
                _ => ykpack::Ty::Unimplemented(format!("{:?}", ty_layout)),
            }
        } else {
            // An enum with variants.
            ykpack::Ty::Unimplemented(format!("{:?}", ty_layout))
        }
    }
}

pub struct SirTypes {
    /// A globally unique identifier for the codegen unit.
    pub cgu_hash: ykpack::CguHash,
    /// Maps types to their index. Ordered by insertion via `IndexMap`.
    pub map: IndexMap<ykpack::Ty, ykpack::TyIndex, BuildHasherDefault<FxHasher>>,
    /// The next available type index.
    next_idx: ykpack::TyIndex,
}

impl SirTypes {
    /// Get the index of a type. If this is the first time we have seen this type, a new index is
    /// allocated and returned.
    ///
    /// Note that the index is only unique within the scope of the current compilation unit.
    /// To make a globally unique ID, we pair the index with CGU hash (see ykpack::CguHash).
    pub fn index(&mut self, t: ykpack::Ty) -> ykpack::TyIndex {
        let next_idx = &mut self.next_idx;
        *self.map.entry(t).or_insert_with(|| {
            let idx = *next_idx;
            *next_idx += 1;
            idx
        })
    }
}
