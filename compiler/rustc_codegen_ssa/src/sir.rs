//! Serialised Intermediate Representation (SIR).
//!
//! SIR is built in-memory during code-generation (in rustc_codegen_ssa), and finally placed
//! into an ELF section at link time.

// FIXME the local decls are wrong.

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

//pub(crate) fn lower_local_ref<'a, 'l, 'tcx, Bx: BuilderMethods<'a, 'tcx>, V>(
//    tcx: TyCtxt<'tcx>,
//    bx: &Bx,
//    decl: &'l LocalRef<'tcx, V>,
//) -> ykpack::LocalDecl {
//    let ty_layout = match decl {
//        LocalRef::Place(pref) => pref.layout,
//        LocalRef::UnsizedPlace(..) => {
//            let sir_ty = ykpack::Ty::Unimplemented(format!("LocalRef::UnsizedPlace"));
//            return ykpack::LocalDecl { ty: bx.cx().define_sir_type(sir_ty) };
//        }
//        LocalRef::Operand(opt_oref) => {
//            if let Some(oref) = opt_oref {
//                oref.layout
//            } else {
//                let sir_ty = ykpack::Ty::Unimplemented(format!("LocalRef::OperandRef is None"));
//                return ykpack::LocalDecl { ty: bx.cx().define_sir_type(sir_ty) };
//            }
//        }
//    };
//
//    ykpack::LocalDecl { ty: lower_ty_and_layout(tcx, bx, &ty_layout) }
//}

fn lower_ty_and_layout<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    tcx: TyCtxt<'tcx>,
    bx: &Bx,
    ty_layout: &TyAndLayout<'tcx>,
) -> ykpack::TypeId {
    let sir_ty = match ty_layout.ty.kind() {
        ty::Int(si) => lower_signed_int(*si),
        ty::Uint(ui) => lower_unsigned_int(*ui),
        ty::Adt(adt_def, ..) => lower_adt(tcx, bx, adt_def, &ty_layout),
        ty::Array(typ, _) => ykpack::Ty::Array(lower_ty_and_layout(tcx, bx, &bx.layout_of(typ))),
        ty::Slice(typ) => ykpack::Ty::Slice(lower_ty_and_layout(tcx, bx, &bx.layout_of(typ))),
        ty::Ref(_, typ, _) => ykpack::Ty::Ref(lower_ty_and_layout(tcx, bx, &bx.layout_of(typ))),
        ty::Bool => ykpack::Ty::Bool,
        ty::Tuple(..) => lower_tuple(tcx, bx, ty_layout),
        _ => ykpack::Ty::Unimplemented(format!("{:?}", ty_layout)),
    };
    bx.cx().define_sir_type(sir_ty)
}

fn lower_signed_int(si: IntTy) -> ykpack::Ty {
    match si {
        IntTy::Isize => ykpack::Ty::SignedInt(ykpack::SignedIntTy::Isize),
        IntTy::I8 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I8),
        IntTy::I16 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I16),
        IntTy::I32 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I32),
        IntTy::I64 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I64),
        IntTy::I128 => ykpack::Ty::SignedInt(ykpack::SignedIntTy::I128),
    }
}

fn lower_unsigned_int(ui: UintTy) -> ykpack::Ty {
    match ui {
        UintTy::Usize => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::Usize),
        UintTy::U8 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U8),
        UintTy::U16 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U16),
        UintTy::U32 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U32),
        UintTy::U64 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U64),
        UintTy::U128 => ykpack::Ty::UnsignedInt(ykpack::UnsignedIntTy::U128),
    }
}

fn lower_tuple<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    tcx: TyCtxt<'tcx>,
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
                sir_tys.push(lower_ty_and_layout(tcx, bx, &ty_layout.field(bx, idx)));
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

fn lower_adt<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    tcx: TyCtxt<'tcx>,
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
                    sir_tys.push(lower_ty_and_layout(tcx, bx, &struct_layout.field(bx, idx)));
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
    /// FIXME
    instance: Instance<'tcx>,
    /// FIXME
    mir: &'tcx mir::Body<'tcx>,
    pub func: ykpack::Body,
    /// Maps a MIR local index to a SIR one. This is required because ther is not one-to-one maping
    /// between MIR and SIR locals. SIR decomposes more complicated MIR asignments (i.e.
    /// projections) to simpler assignments, resulting in more variables and assignments.
    /// FIXME optimise with a growing indexvec.
    var_map: FxHashMap<mir::Local, ykpack::Local>,
    /// The next SIR local variable index to be allocated.
    next_sir_local: ykpack::LocalIndex,
    tcx: TyCtxt<'tcx>,
    /// Local decls to be moved into the SIR body at the end of lowering.
    local_decls: Vec<Option<ykpack::TypeId>>,
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

        let local_decls = Vec::with_capacity(mir.local_decls.len());
        let symbol_name = String::from(&*tcx.symbol_name(*instance).name);

        let crate_name = tcx.crate_name(instance.def_id().krate).as_str();
        if crate_name == "core" || crate_name == "alloc" {
            flags |= ykpack::bodyflags::DO_NOT_TRACE;
        }

        Self {
            instance: instance.clone(),
            mir,
            func: ykpack::Body { symbol_name, blocks, flags, local_decls, num_args: mir.arg_count },
            var_map: Default::default(),
            local_decls: Default::default(),
            next_sir_local: 0,
            tcx,
        }
    }

    /// Returns the SIR local corresponding with MIR local `ml`. A new SIR local is allocated if
    /// we've never seen this MIR local before.
    fn sir_local(&mut self, ml: &mir::Local) -> ykpack::Local {
        if let Some(idx) = self.var_map.get(ml) { *idx } else { self.new_sir_local() }
    }

    /// Returns a fresh local variable.
    fn new_sir_local(&mut self) -> ykpack::Local {
        let idx = self.next_sir_local;
        self.next_sir_local += 1;
        self.local_decls.push(None); // Unknown type until later.
        ykpack::Local(idx)
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
        let lhs = self.lower_place(bx, bb, lvalue);
        let rhs = self.lower_rvalue(bx, bb, rvalue);
        self.push_stmt(bb, ykpack::Statement::IStore(lhs, rhs));
    }

    pub fn lower_operand<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        operand: &mir::Operand<'tcx>,
    ) -> ykpack::Local {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                self.lower_place(bx, bb, place)
            }
            mir::Operand::Constant(cst) => self.lower_constant(bx, bb, cst),
        }
    }

    fn push_unimpl_assign<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        msg: String,
    ) -> ykpack::Local {
        let l = self.new_sir_local();
        let ip = ykpack::IPlace::Unimplemented(msg);
        let sir_tyid = (ykpack::CguHash(0), 0); // Dummy value;
        self.push_assign_and_declare(bx, bb, sir_tyid, l, ip);
        l
    }

    fn lower_rvalue<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        rvalue: &mir::Rvalue<'tcx>,
    ) -> ykpack::Local {
        match rvalue {
            mir::Rvalue::Use(opnd) => self.lower_operand(bx, bb, opnd),
            mir::Rvalue::Ref(_, _, p) => self.lower_ref(bx, bb, p),
            _ => self.push_unimpl_assign(
                bx,
                bb,
                with_no_trimmed_paths(|| format!("unimplemented rvalue: {:?}", rvalue)),
            ),
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

    pub fn lower_place<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        place: &mir::Place<'_>,
    ) -> ykpack::Local {
        // Start with the base local and project off of it.
        let mut cur_sir_local = self.sir_local(&place.local);
        dbg!("mono1");
        let mut cur_mirty = self.monomorphize(&self.mir.local_decls[place.local].ty);

        // Loop over the projection chain, repeatedly emitting intermediate SIR assignments.
        for pj in place.projection {
            dbg!(pj, cur_mirty);
            let (next_mirty, next_dv): (Ty<'_>, ykpack::Derivative) = match pj {
                mir::ProjectionElem::Field(f, _) => {
                    let fi = f.as_usize();
                    match cur_mirty.kind() {
                        ty::Adt(def, _) => {
                            if def.is_struct() {
                                let ty_lay = bx.layout_of(cur_mirty);
                                let st_lay = ty_lay.for_variant(bx, VariantIdx::from_u32(0));
                                if let FieldsShape::Arbitrary { offsets, .. } = &st_lay.fields {
                                    let fld = st_lay.field(bx, fi);
                                    if fld.is_unsized() {
                                        return self.push_unimpl_assign(
                                            bx,
                                            bb,
                                            format!("struct field shape: {:?}", st_lay.fields),
                                        );
                                    } else {
                                        (fld.ty, ykpack::Derivative::ByteOffset(offsets[fi].bytes_usize()))
                                    }
                                } else {
                                    return self.push_unimpl_assign(
                                        bx,
                                        bb,
                                        format!("struct field shape: {:?}", st_lay.fields),
                                    );
                                }
                            } else if def.is_enum() {
                                return self.push_unimpl_assign(
                                    bx,
                                    bb,
                                    format!("enum_projection: {:?}", def),
                                );
                            } else {
                                return self.push_unimpl_assign(
                                    bx,
                                    bb,
                                    format!("unhandled adt: {:?}", def),
                                );
                            }
                        }
                        ty::Tuple(..) => {
                            let tup_lay = bx.layout_of(cur_mirty);
                            match &tup_lay.fields {
                                //FieldsShape::Arbitrary { offsets, .. } => (
                                //    tup_lay.field(bx, fi).ty,
                                //    ykpack::Derivative::ByteOffset(offsets[fi].bytes_usize()),
                                //),
                                _ => {
                                    return self.push_unimpl_assign(
                                        bx,
                                        bb,
                                        format!("tuple field shape: {:?}", tup_lay.fields),
                                    );
                                }
                            }
                        }
                        _ => {
                            return self.push_unimpl_assign(
                                bx,
                                bb,
                                format!("field access on: {:?}", cur_mirty),
                            );
                        }
                    }
                }
                _ => return self.push_unimpl_assign(bx, bb, format!("projection: {:?}", pj)),
            };


            // FIXME needed?
            //if !next_mirty.is_sized() {
            //    return self.push_unimpl_assign(bx, bb, format!("unsized: {:?}", next_mirty));
            //}

            dbg!("mono2");
            let next_mirty = self.monomorphize(&next_mirty); //.ty);

            // Emit an assignment to an intermediate SIR local and update the state for the next
            // iteration of the resolution loop.
            let l = self.new_sir_local();
            let ip = ykpack::IPlace::Val(cur_sir_local, next_dv);
            dbg!("layout of");
            let sir_tyid = lower_ty_and_layout(self.tcx, bx, &bx.layout_of(next_mirty));
            dbg!("/layout of");
            self.push_assign_and_declare(bx, bb, sir_tyid, l, ip);

            cur_mirty = next_mirty;
            //cur_mirty = self.monomorphize(&next_mirty);
            cur_sir_local = l;
        }
        //ykpack::IPlace::Val(cur_sir_local, ykpack::Derivative::ByteOffset(0))
        dbg!("done");
        cur_sir_local
    }

    /// Emit an assignment statement and update the type of the target local.
    pub fn push_assign_and_declare<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        sir_ty: ykpack::TypeId,
        l: ykpack::Local,
        ip: ykpack::IPlace,
    ) {
        //let sir_ty = lower_ty_and_layout(self.tcx, bx, ty);
        let slot = self.local_decls.get_mut(usize::try_from(l.0).unwrap()).unwrap();
        debug_assert!(slot.is_none()); // Check it wasn't already declared.
        *slot = Some(sir_ty);

        self.push_stmt(bb, ykpack::Statement::Assign(l, ip));
    }

    pub fn lower_projection(&self, pe: &mir::PlaceElem<'_>) -> ykpack::Projection {
        match pe {
            mir::ProjectionElem::Field(field, ..) => ykpack::Projection::Field(field.as_u32()),
            mir::ProjectionElem::Deref => ykpack::Projection::Deref,
            mir::ProjectionElem::Index(local) => {
                ykpack::Projection::Index(self.lower_local(*local))
            }
            _ => ykpack::Projection::Unimplemented(format!("{:?}", pe)),
        }
    }

    pub fn lower_local(&self, local: mir::Local) -> ykpack::Local {
        // For the lowering of `Local`s we currently assume a 1:1 mapping from MIR to SIR. If this
        // mapping turns out to be impossible or impractial, this is the place to change it.
        ykpack::Local(local.as_u32())
    }

    fn lower_constant<Bx: BuilderMethods<'a, 'tcx>>(
        &mut self,
        bx: &Bx,
        bb: ykpack::BasicBlockIndex,
        constant: &mir::Constant<'tcx>,
    ) -> ykpack::Local {
        let (c, cty) = match constant.literal.val {
            ty::ConstKind::Value(mir::interpret::ConstValue::Scalar(s)) => {
                (self.lower_scalar(constant.literal.ty, s), constant.literal.ty)
            }
            _ => return self.push_unimpl_assign(bx, bb, with_no_trimmed_paths(|| { format!("unimplemented constant: {:?}", constant)})),
        };

        let cty = self.monomorphize(&cty);

        let l = self.new_sir_local();
        let ip = ykpack::IPlace::Const(c);
        let sir_tyid = lower_ty_and_layout(self.tcx, bx, &bx.layout_of(cty));
        self.push_assign_and_declare(bx, bb, sir_tyid, l, ip);
        l
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

    fn lower_binop(
        &self,
        op: mir::BinOp,
        opnd1: &mir::Operand<'_>,
        opnd2: &mir::Operand<'_>,
        checked: bool,
    ) -> ykpack::Rvalue {
        todo!();
        //    let sir_op = binop_lowerings!(
        //        op, Add, Sub, Mul, Div, Rem, BitXor, BitAnd, BitOr, Shl, Shr, Eq, Lt, Le, Ne, Ge, Gt,
        //        Offset
        //    );
        //    let sir_opnd1 = self.lower_operand(opnd1);
        //    let sir_opnd2 = self.lower_operand(opnd2);

        //    if checked {
        //        debug_assert!(CHECKABLE_BINOPS.contains(&sir_op));
        //        ykpack::Rvalue::CheckedBinaryOp(sir_op, sir_opnd1, sir_opnd2)
        //    } else {
        //        ykpack::Rvalue::BinaryOp(sir_op, sir_opnd1, sir_opnd2)
        //    }
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
        place: &mir::Place<'_>,
    ) -> ykpack::Local {
        self.push_unimpl_assign(bx, bb, with_no_trimmed_paths(|| format!("ref: {:?}", place)))
        //let inner = self.lower_place(bx, bb, place);

        //let l = self.new_sir_local();
        //let ip = ykpack::IPlace::Ref(inner, ykpack::Derivative::ByteOffset(0));
        // FIXME unwrapping decl can fail presumably due to incomplete projections. ugh.
        //let sir_tyid = self.local_decls[usize::try_from(inner.0).unwrap()].unwrap();
        //self.push_assign_and_declare(bx, bb, sir_tyid, l, ip);
        //l
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
