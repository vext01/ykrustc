//! FIXME

// FIXME just for now, or I will go mad.
#![allow(unused_imports)]
#![allow(unused_variables)]

use rustc_index::{newtype_index, vec::{Idx, IndexVec}};
use syntax::symbol::Symbol;
use syntax::source_map::{DUMMY_SP, Span};
use syntax::ast;
use crate::mir::debuginfo::VariableKind;
use rustc::mir::interpret::{Scalar, Allocation};
use std::ops::Deref;
use std::marker::PhantomData;
use std::borrow::BorrowMut;
use crate::mir::debuginfo::FunctionDebugContext;
use rustc::hir::def_id::CrateNum;
use rustc::hir;
use rustc::mir;
use rustc::mir::mono::{Linkage, Visibility};
use std::cell::{RefCell, Cell};
use std::sync::Arc;
use rustc::session::Session;
use rustc::mir::mono::CodegenUnit;
use rustc::util::nodemap::FxHashMap;
use crate::common::{self, IntPredicate, RealPredicate, AtomicOrdering, AtomicRmwBinOp,
                    SynchronizationScope};
use crate::MemFlags;
use rustc::ty::{self, Ty, TyCtxt, Instance, PolyFnSig};
use rustc::ty::layout::{self, Align, Size, TyLayout, LayoutError, LayoutOf, FnAbiExt};
use rustc::hir::def_id::DefId;
use crate::traits::*;
use crate::mir::operand::{OperandRef, OperandValue};
use crate::mir::place::PlaceRef;
use rustc_target::spec::{HasTargetSpec, Target};
use rustc_target::abi::call::{ArgAbi, FnAbi, CastTarget, Reg};
use std::ffi::CString;
use std::ops::Range;
use std::iter::TrustedLen;

use rustc_target::abi::HasDataLayout;
pub use rustc_target::spec::abi::Abi;
use crate::traits::ArgAbiMethods;
pub use rustc_target::abi::call::PassMode;
use rustc_data_structures::base_n;

newtype_index! {
    pub struct FunctionIdx {
        DEBUG_FORMAT = "FunctionIdx[{}]"
    }
}

newtype_index! {
    pub struct InstructionIdx {
        DEBUG_FORMAT = "InstructionIdx[{}]"
    }
}

newtype_index! {
    pub struct TypeIdx {
        DEBUG_FORMAT = "TypeIdx[{}]"
    }
}

newtype_index! {
    pub struct GlobalIdx {
        DEBUG_FORMAT = "GlobalIdx[{}]"
    }
}

newtype_index! {
    /// Identifies a basic block local to a function.
    pub struct BasicBlockIdx {
        DEBUG_FORMAT = "BasicBlockIdx[{}]"
    }
}

/// Globally identifies a basic block.
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct BasicBlockPath {
    func_idx: FunctionIdx,
    block_idx: BasicBlockIdx,
}

impl BasicBlockPath {
    fn new(func_idx: FunctionIdx, block_idx: BasicBlockIdx) -> Self {
        Self {
            func_idx,
            block_idx,
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone, Hash, Eq)]
pub struct InstructionPath {
    pub func_idx: FunctionIdx,
    pub block_idx: BasicBlockIdx,
    pub instr_idx: InstructionIdx,
}

impl Default for InstructionPath {
    fn default() -> Self {
        Self {
            func_idx: FunctionIdx::from_usize(0),
            block_idx: BasicBlockIdx::from_usize(0),
            instr_idx: InstructionIdx::from_usize(0),
        }
    }
}

#[allow(dead_code)]
struct DILexicalBlock {}

/// The all-encompassing enum.
/// Each is an index into the corresponding vector in SirCodegenCx.
#[derive(Debug, PartialEq, Copy, Clone, Hash, Eq)]
pub enum Value {
    Function(FunctionIdx),
    FunctionArg(TypeIdx),
    Global(GlobalIdx),
    Instruction(InstructionPath),
    ConstUndef(TypeIdx),
    /// Unimplemented instructions currently return this.
    Dummy,
}

impl Value {
    fn ty(&self, cx: &SirCodegenCx<'ll, 'tcx>) -> TypeIdx {
        match self {
            Self::Function(idx) => {
                let funcs = cx.functions.borrow();
                funcs[*idx].ty
            },
            Self::FunctionArg(t) | Self::ConstUndef(t) => *t,
            Self::Dummy => TypeIdx::from_usize(0), // FIXME dummy types
            _ => {
                info!("{:?}", self);
                unimplemented!("Value::ty");
            }
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Instruction {
    /// An unimplemented part of the SIR codegen.
    /// The string inside should give some kind of indication where to look.
    Unimplemented(&'static str),
}

impl Value {
    fn unwrap_function(&self) -> FunctionIdx {
        if let Self::Function(idx) = self {
            *idx
        } else {
            panic!("Failed to unwrap a Value::Function");
        }
    }

    #[allow(dead_code)]
    fn unwrap_global(&self) -> GlobalIdx {
        if let Self::Global(idx) = self {
            *idx
        } else {
            panic!("Failed to unwrap a Value::Global");
        }
    }

    fn unwrap_instr(&self) -> InstructionPath {
        if let Self::Instruction(path) = self {
            *path
        } else {
            panic!("Failed to unwrap a Value::Instruction");
        }
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum SirType {
    Function {
        args: Vec<TypeIdx>,
        ret: TypeIdx,
    },
    Scalar(ScalarType),
    Dummy
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScalarType {
    I1,
    I8,
    I16,
    I32,
    I64,
    I128,
    ISize,
    // An integer value of the given size.
    //Ix(u64),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    name: String,
    ty: TypeIdx,
    //args: Vec<Value>,
    blocks: IndexVec<BasicBlockIdx, BasicBlock>,
}

impl Function {
    //fn new(name: String, ty: TypeIdx, args: Vec<Value>,
    fn new(name: String, ty: TypeIdx, blocks: IndexVec<BasicBlockIdx, BasicBlock>) -> Self {
        //Self{name, ty, args, blocks}
        Self{name, ty, blocks}
    }

    fn add_block(&mut self, parent: FunctionIdx) -> BasicBlockIdx {
        let idx = self.blocks.len();
        self.blocks.push(BasicBlock::new(parent));
        BasicBlockIdx::from_usize(idx)
    }

    fn arg(&self, cx: &SirCodegenCx<'ll, 'tcx>, idx: usize) -> Value {
        info!("arg");
        let types = cx.types.borrow();
        if let SirType::Function{args, ..} = &types[self.ty] {
            info!("arg type right");
            let ret = Value::FunctionArg(args[idx]);
            info!("got arg");
            ret
        } else {
            panic!("invalid function type");
        }

        //info!("AAA");
        ////let ret = self.args[idx];
        //let ret = self.args[idx];
        //info!("/AAA");
        //ret
    }
}

#[allow(dead_code)]
pub struct Global {
    name: String,
    ty: TypeIdx,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    instrs: IndexVec<InstructionIdx, Instruction>,
    parent: FunctionIdx,
}

impl BasicBlock {
    pub fn new(parent: FunctionIdx) -> Self {
        Self {
            instrs: IndexVec::default(),
            parent,
        }
    }
}

#[derive(Debug)]
pub struct Funclet {}

#[derive(Debug, Copy, Clone)]
pub struct DIScope {}

#[derive(Debug, Copy, Clone)]
pub struct DISubprogram {}

pub struct SirCodegenCx<'ll, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub codegen_unit: Arc<CodegenUnit<'tcx>>,

    pub functions: RefCell<IndexVec<FunctionIdx, Function>>,

    // FIXME Needed?
    pub instances: RefCell<FxHashMap<Instance<'tcx>, Value>>,

    pub types: RefCell<IndexVec<TypeIdx, SirType>>,
    /// The reverse mapping of the above.
    /// FIXME: don't store a copy of the SirType.
    pub type_cache: RefCell<FxHashMap<SirType, TypeIdx>>,
    pub dummy_type_idx: TypeIdx,

    pub globals: RefCell<IndexVec<GlobalIdx, Global>>,
    pub globals_by_name: RefCell<FxHashMap<String, GlobalIdx>>,
    pub const_globals: RefCell<FxHashMap<Value, GlobalIdx>>,

    // FIXME: almost certain this lifetime will crop up later.
    pd: PhantomData<&'ll ()>,

    pub vtables: RefCell<FxHashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), Value>>,

    local_gen_sym_counter: Cell<usize>,
}

impl SirCodegenCx<'ll, 'tcx> {
    fn add_type(&self, t: SirType) -> TypeIdx {
        let mut cache = self.type_cache.borrow_mut();
        *cache.entry(t.clone()).or_insert_with(|| { // FIXME kill clone.
            let mut types = self.types.borrow_mut();
            let new_idx = TypeIdx::from_usize(types.len());
            types.push(t);
            new_idx
        })
    }

    pub fn generate_local_symbol_name(&self, prefix: &str) -> String {
        let idx = self.local_gen_sym_counter.get();
        self.local_gen_sym_counter.set(idx + 1);
        // Include a '.' character, so there can be no accidental conflicts with
        // user defined names
        let mut name = String::with_capacity(prefix.len() + 6);
        name.push_str(prefix);
        name.push_str(".");
        base_n::push_str(idx as u128, base_n::ALPHANUMERIC_ONLY, &mut name);
        name
    }
}

pub trait ArgAbiExt<'ll, 'tcx> {
    fn memory_ty(&self, cx: &SirCodegenCx<'ll, 'tcx>) -> TypeIdx;
    fn store(
        &self,
        bx: &mut SirBuilder<'_, 'll, 'tcx>,
        val: Value,
        dst: PlaceRef<'tcx, Value>,
    );
    fn store_fn_arg(
        &self,
        bx: &mut SirBuilder<'_, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Value>,
    );
}

impl ArgAbiExt<'ll, 'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
    fn memory_ty(&self, cx: &SirCodegenCx<'ll, 'tcx>) -> TypeIdx {
        unimplemented!();
    }

    // An almost direct copy from the LLVM backend.
    fn store(
        &self,
        bx: &mut SirBuilder<'_, 'll, 'tcx>,
        val: Value,
        dst: PlaceRef<'tcx, Value>,
    ) {
        if self.is_ignore() {
            return;
        }
        if self.is_sized_indirect() {
            OperandValue::Ref(val, None, self.layout.align.abi).store(bx, dst)
        } else if self.is_unsized_indirect() {
            bug!("unsized ArgAbi must be handled through store_fn_arg");
        } else if let PassMode::Cast(cast) = self.mode {
            unimplemented!("cast store");
        } else {
            OperandValue::Immediate(val).store(bx, dst);
        }
    }

    fn store_fn_arg(
        &self,
        bx: &mut SirBuilder<'a, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Value>,
    ) {
        let mut next = || {
            info!("next arg: idx={}", idx);
            let fn_idx = bx.llfn().unwrap_function();
            let fns = bx.cx().functions.borrow();
            info!("XXX");
            let val = fns[fn_idx].arg(bx.cx(), *idx);
            info!("/XXX");
            *idx += 1;
            val
        };

        match self.mode {
            PassMode::Ignore => {
                info!(">> ignore");
            },
            PassMode::Pair(..) => {
                info!(">> pair");
                OperandValue::Pair(next(), next()).store(bx, dst);
            }
            PassMode::Indirect(_, Some(_)) => {
                info!(">> indirect");
                OperandValue::Ref(next(), Some(next()), self.layout.align.abi).store(bx, dst);
            }
            PassMode::Direct(_) | PassMode::Indirect(_, None) | PassMode::Cast(_) => {
                info!(">> direct");
                let next_arg = next();
                self.store(bx, next_arg, dst);
            }
        }
    }
}

impl SirCodegenCx<'ll, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        codegen_unit: Arc<CodegenUnit<'tcx>>,
    ) -> Self {
        let mut ty_map = FxHashMap::default();
        ty_map.insert(SirType::Dummy, TypeIdx::from_usize(0)); // FIXME dummy types

        Self {
            tcx,
            codegen_unit,
            functions: RefCell::new(IndexVec::new()),
            instances: RefCell::new(FxHashMap::default()),
            types: RefCell::new(IndexVec::from_elem_n(SirType::Dummy, 1)), // FIXME dummy types
            type_cache: RefCell::new(ty_map),
            dummy_type_idx: TypeIdx::from_usize(0),
            globals: RefCell::new(IndexVec::default()),
            globals_by_name: RefCell::new(FxHashMap::default()),
            const_globals: RefCell::new(FxHashMap::default()),
            vtables: RefCell::new(FxHashMap::default()),
            pd: PhantomData,
            local_gen_sym_counter: Cell::new(0),
        }
    }
}

impl AsmMethods for SirCodegenCx<'ll, 'tcx> {
    fn codegen_global_asm(&self, ga: &hir::GlobalAsm) {
        unimplemented!();
    }
}

impl ConstMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn const_null(&self, t: TypeIdx) -> Value {
        Value::Dummy // FIXME
    }

    fn const_undef(&self, t: TypeIdx) -> Value {
	Value::ConstUndef(t)
    }

    fn const_int(&self, t: TypeIdx, i: i64) -> Value {
        Value::Dummy // FIXME
    }

    fn const_uint(&self, t: TypeIdx, i: u64) -> Value {
        Value::Dummy // FIXME
    }

    fn const_uint_big(&self, t: TypeIdx, u: u128) -> Value {
        Value::Dummy // FIXME
    }

    fn const_bool(&self, val: bool) -> Value {
        Value::Dummy // FIXME
    }

    fn const_i32(&self, i: i32) -> Value {
        Value::Dummy // FIXME
    }

    fn const_u32(&self, i: u32) -> Value {
        Value::Dummy // FIXME
    }

    fn const_u64(&self, i: u64) -> Value {
        Value::Dummy // FIXME
    }

    fn const_usize(&self, i: u64) -> Value {
        Value::Dummy // FIXME
    }

    fn const_u8(&self, i: u8) -> Value {
        Value::Dummy // FIXME
    }

    fn const_real(&self, t: TypeIdx, val: f64) -> Value {
        Value::Dummy // FIXME
    }

    fn const_str(&self, s: Symbol) -> (Value, Value) {
        (Value::Dummy, Value::Dummy) // FIXME
    }

    fn const_struct(
        &self,
        elts: &[Value],
        packed: bool
    ) -> Value {
        Value::Dummy // FIXME
    }

    fn const_to_opt_uint(&self, v: Value) -> Option<u64> {
        Some(0) // FIXME
    }

    fn const_to_opt_u128(&self, v: Value, sign_ext: bool) -> Option<u128> {
        Some(0) // FIXME
    }

    fn scalar_to_backend(
        &self,
        cv: Scalar,
        layout: &layout::Scalar,
        llty: TypeIdx,
    ) -> Value {
        Value::Dummy // FIXME
    }

    fn from_const_alloc(
        &self,
        layout: TyLayout<'tcx>,
        alloc: &Allocation,
        offset: Size,
    ) -> PlaceRef<'tcx, Value> {
        unimplemented!();
    }

    fn const_ptrcast(&self, val: Value, ty: TypeIdx) -> Value {
        Value::Dummy // FIXME
    }
}

pub trait FnAbiSirExt<'tcx> {
    fn sir_type(&self, cx: &SirCodegenCx<'ll, 'tcx>) -> TypeIdx;
    //fn ptr_to_llvm_type(&self, cx: &SirCodegenCx<'ll, 'tcx>) -> &'ll Type;
    //fn llvm_cconv(&self) -> llvm::CallConv;
    //fn apply_attrs_llfn(&self, cx: &SirCodegenCx<'ll, 'tcx>, llfn: &'ll Value);
    //fn apply_attrs_callsite(&self, bx: &mut SirBuilder<'a, 'll, 'tcx>, callsite: &'ll Value);
}

impl<'tcx> FnAbiSirExt<'tcx> for FnAbi<'tcx, Ty<'tcx>> {
    fn sir_type(&self, cx: &SirCodegenCx<'ll, 'tcx>) -> TypeIdx {
        let args_capacity: usize = self.args.iter().map(|arg|
            if arg.pad.is_some() { 1 } else { 0 } +
            if let PassMode::Pair(_, _) = arg.mode { 2 } else { 1 }
        ).sum();

        let mut llargument_tys = Vec::with_capacity(
            if let PassMode::Indirect(..) = self.ret.mode { 1 } else { 0 } + args_capacity
        );

        let llreturn_ty = TypeIdx::from_usize(0); // FIXME
        //let llreturn_ty = match self.ret.mode {
        //    PassMode::Ignore => cx.type_void(),
        //    PassMode::Direct(_) | PassMode::Pair(..) => {
        //        self.ret.layout.immediate_llvm_type(cx)
        //    }
        //    PassMode::Cast(cast) => cast.llvm_type(cx),
        //    PassMode::Indirect(..) => {
        //        llargument_tys.push(cx.type_ptr_to(self.ret.memory_ty(cx)));
        //        cx.type_void()
        //    }
        //};

        for arg in &self.args {
            // add padding
            if let Some(ty) = arg.pad {
                //llargument_tys.push(ty.llvm_type(cx));
                llargument_tys.push(TypeIdx::from_usize(0)); // FIXME
            }

            //let llarg_ty = match arg.mode {
            //    PassMode::Ignore => continue,
            //    PassMode::Direct(_) => arg.layout.immediate_llvm_type(cx),
            //    PassMode::Pair(..) => {
            //        llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 0, true));
            //        llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 1, true));
            //        continue;
            //    }
            //    PassMode::Indirect(_, Some(_)) => {
            //        let ptr_ty = cx.tcx.mk_mut_ptr(arg.layout.ty);
            //        let ptr_layout = cx.layout_of(ptr_ty);
            //        llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 0, true));
            //        llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 1, true));
            //        continue;
            //    }
            //    PassMode::Cast(cast) => cast.llvm_type(cx),
            //    PassMode::Indirect(_, None) => cx.type_ptr_to(arg.memory_ty(cx)),
            //};
            //llargument_tys.push(llarg_ty);
            llargument_tys.push(TypeIdx::from_usize(0));
        }

        if self.c_variadic {
            unimplemented!("variadic funcs");
            //cx.type_variadic_func(&llargument_tys, llreturn_ty)
        } else {
            cx.type_func(&llargument_tys[..], llreturn_ty)
        }
    }
}

fn declare_raw_fn(
    cx: &SirCodegenCx<'ll, 'tcx>,
    name: &str,
    ty: TypeIdx,
) -> Value {
    // FIXME calling convention and ABI stuff?
    let mut fns = cx.functions.borrow_mut();
    let idx = fns.len();
    info!("declare_raw_fn: name={:?}, idx={:?}, ty={:?}", name, idx, ty);

    fns.push(Function::new(name.to_owned(), ty, IndexVec::default()));
    Value::Function(FunctionIdx::from_usize(idx))
}

impl DeclareMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn declare_global(
        &self,
        name: &str, ty: TypeIdx
    ) -> Value {
        info!("declare_global(name={:?})", name);
        let mut globals = self.globals.borrow_mut();
        let mut globals_by_name = self.globals_by_name.borrow_mut();

        let idx = globals.len();
        globals.push(Global {
            name: name.to_owned(),
            ty
        });
        let gi = GlobalIdx::from_usize(idx);
        globals_by_name.insert(name.to_owned(), gi);

        Value::Global(gi)
    }

    fn declare_cfn(
        &self,
        name: &str,
        fn_type: TypeIdx
    ) -> Value {
        declare_raw_fn(self, name, fn_type)
    }

    fn declare_fn(
        &self,
        name: &str,
        sig: PolyFnSig<'tcx>,
    ) -> Value {
        info!("declare_rust_fn(name={:?}, sig={:?})", name, sig);
        let sig = self.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
        info!("declare_rust_fn (after region erasure) sig={:?}", sig);

        //let fn_type = self.new_fn_type(sig, &[]); //.ironox_type(self);
        let fn_abi = FnAbi::new(self, sig, &[]);
        //let llfn = declare_raw_fn(self, name, fn_abi.llvm_cconv(), fn_abi.llvm_type(self));

        // declare_raw_fn(self, name, self.dummy_type_idx)
        declare_raw_fn(self, name, fn_abi.sir_type(self))
    }

    fn define_global(
        &self,
        name: &str,
        ty: TypeIdx
    ) -> Option<Value> {
        if self.get_defined_value(name).is_some() {
            None
        } else {
            Some(self.declare_global(name, ty))
        }
    }

    fn define_private_global(&self, ty: TypeIdx) -> Value {
        unimplemented!();
    }

    fn define_fn(
        &self,
        name: &str,
        fn_sig: PolyFnSig<'tcx>,
    ) -> Value {
        unimplemented!();
    }

    fn define_internal_fn(
        &self,
        name: &str,
        fn_sig: PolyFnSig<'tcx>,
    ) -> Value {
        unimplemented!();
    }

    fn get_declared_value(&self, name: &str) -> Option<Value> {
        if let Some(idx) = self.globals_by_name.borrow().get(name) {
            Some(Value::Global(*idx))
        } else {
            None
        }
    }

    fn get_defined_value(&self, name: &str) -> Option<Value> {
        self.get_declared_value(name)
    }
}

impl DebugInfoMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        sig: ty::FnSig<'tcx>,
        llfn: Value,
        mir: &mir::Body<'_>,
    ) -> Option<FunctionDebugContext<&'ll DIScope>> {
        None
    }

    fn create_vtable_metadata(
        &self,
        ty: Ty<'tcx>,
        vtable: Self::Value,
    ) {
        unimplemented!();
    }

    fn extend_scope_to_file(
         &self,
         scope_metadata: &'ll DIScope,
         file: &syntax_pos::SourceFile,
         defining_crate: CrateNum,
     ) -> &'ll DIScope {
        unimplemented!();
     }

    fn debuginfo_finalize(&self) {
        unimplemented!();
    }

    fn has_debug(&self) -> bool {
        unimplemented!();
    }
}

impl PreDefineMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn predefine_static(&self,
                                  def_id: DefId,
                                  linkage: Linkage,
                                  visibility: Visibility,
                                  symbol_name: &str) {
        let instance = Instance::mono(self.tcx, def_id);
        // FIXME layout.
        let g = self.define_global(symbol_name, self.dummy_type_idx).unwrap_or_else(|| {
            self.sess().span_fatal(self.tcx.def_span(def_id),
                &format!("symbol `{}` is already defined", symbol_name))
        });

        self.instances.borrow_mut().insert(instance, g);
    }

    fn predefine_fn(&self,
                    instance: Instance<'tcx>,
                    linkage: Linkage,
                    visibility: Visibility,
                    symbol_name: &str) {
        let mono_sig = instance.fn_sig(self.tcx);
        let lldecl = self.declare_fn(symbol_name, mono_sig);

        info!("predefine_fn: mono_sig = {:?} instance = {:?}", mono_sig, instance);
        self.instances.borrow_mut().insert(instance, lldecl);
    }
}

impl ty::layout::HasParamEnv<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        ty::ParamEnv::reveal_all()
    }
}

impl StaticMethods for SirCodegenCx<'ll, 'tcx> {
    fn static_addr_of(
        &self,
        cv: Value,
        align: Align,
        kind: Option<&str>,
    ) -> Value {
        let mut const_globals = self.const_globals.borrow_mut();
        if let Some(&gv) = const_globals.get(&cv) {
            // FIXME: Fiddle alignment, as LLVM backend does?
            return Value::Global(gv);
        }

        // In the LLVM codegen, this part is in static_addr_of_mut().
        // Inlined for borrowing reasons.
        let gv = match kind {
            Some(kind) if !self.tcx.sess.fewer_names() => {
                let name = self.generate_local_symbol_name(kind);
                let gv = self.define_global(&name[..],
                    self.val_ty(cv)).unwrap_or_else(||{
                    bug!("symbol `{}` is already defined", name);
                });
                gv
            },
            _ => self.define_private_global(self.val_ty(cv)),
        };

        // FIXME set initialiser?
        // FIXME set alignment?
        const_globals.insert(cv, gv.unwrap_global());
        gv
    }

    fn codegen_static(
        &self,
        def_id: DefId,
        is_mutable: bool,
    ) {
        unimplemented!();
    }
}

impl MiscMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn vtables(&self) -> &RefCell<FxHashMap<(Ty<'tcx>,
                                Option<ty::PolyExistentialTraitRef<'tcx>>), Value>>
    {
        &self.vtables
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> Value {
        // FIXME: do we need the pointer casting logic from the LLVM backend?
        let sym = self.tcx.symbol_name(instance).name.as_str();
        let sig = instance.fn_sig(self.tcx);
        info!("get_fn({:?}: {:?}) => {}", instance, sig, sym);

        if let Some(ref llfn) = self.instances.borrow().get(&instance) {
            **llfn
        } else {
            let llfn = self.declare_fn(&sym, sig);
            self.instances.borrow_mut().insert(instance, llfn);
            llfn
        }
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> Value {
        Value::Dummy // FIXME
    }

    fn eh_personality(&self) -> Value {
        Value::Dummy // FIXME
    }

    fn eh_unwind_resume(&self) -> Value {
        unimplemented!();
    }

    fn sess(&self) -> &Session {
	self.tcx.sess
    }

    fn check_overflow(&self) -> bool {
        false // FIXME
    }

    fn codegen_unit(&self) -> &Arc<CodegenUnit<'tcx>> {
        &self.codegen_unit
    }

    fn used_statics(&self) -> &RefCell<Vec<Value>> {
        unimplemented!();
    }

    fn set_frame_pointer_elimination(&self, llfn: Value) {
        unimplemented!();
    }

    fn apply_target_cpu_attr(&self, llfn: Value) {
        unimplemented!();
    }

    fn create_used_variable(&self) {
        unimplemented!();
    }
}

impl BaseTypeMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn type_i1(&self) -> TypeIdx {
        self.add_type(SirType::Scalar(ScalarType::I1))
    }

    fn type_i8(&self) -> TypeIdx {
        self.add_type(SirType::Scalar(ScalarType::I8))
    }

    fn type_i16(&self) -> TypeIdx {
        self.add_type(SirType::Scalar(ScalarType::I16))
    }

    fn type_i32(&self) -> TypeIdx {
        self.add_type(SirType::Scalar(ScalarType::I32))
    }

    fn type_i64(&self) -> TypeIdx {
        self.add_type(SirType::Scalar(ScalarType::I64))
    }

    fn type_i128(&self) -> TypeIdx {
        self.add_type(SirType::Scalar(ScalarType::I128))
    }

    fn type_isize(&self) -> TypeIdx {
        self.add_type(SirType::Scalar(ScalarType::ISize))
    }

    fn type_f32(&self) -> TypeIdx {
        unimplemented!("type_f32");
    }

    fn type_f64(&self) -> TypeIdx {
        unimplemented!("type_f64");
    }

    fn type_func(
        &self,
        args: &[TypeIdx],
        ret: TypeIdx
    ) -> TypeIdx {
        let fn_type = SirType::Function {
            args: Vec::from(args),
            ret: ret.clone(),
        };
        self.add_type(fn_type)
    }

    fn type_struct(
        &self,
        els: &[TypeIdx],
        packed: bool
    ) -> TypeIdx {
        TypeIdx::from_usize(0) // FIXME
    }

    fn type_kind(&self, ty: TypeIdx) -> common::TypeKind {
        common::TypeKind::Void // FIXME
    }

    fn type_ptr_to(&self, ty: TypeIdx) -> TypeIdx {
        TypeIdx::from_usize(0) // FIXME
    }

    fn element_type(&self, ty: TypeIdx) -> TypeIdx {
        TypeIdx::from_usize(0) // FIXME
    }

    fn vector_length(&self, ty: TypeIdx) -> usize {
        unimplemented!();
    }

    fn float_width(&self, ty: TypeIdx) -> usize {
        unimplemented!();
    }

    fn int_width(&self, ty: TypeIdx) -> u64 {
        unimplemented!();
    }

    fn val_ty(&self, v: Value) -> TypeIdx {
        v.ty(self)
    }
}

impl HasTargetSpec for SirCodegenCx<'ll, 'tcx> {
    fn target_spec(&self) -> &Target {
        unimplemented!();
    }
}

impl BackendTypes for SirCodegenCx<'ll, 'tcx> {
    type Value = Value;
    type Function = Value;
    type BasicBlock = BasicBlockPath;
    type Type = TypeIdx;
    type Funclet = Funclet;
    type DIScope = &'ll DIScope;
    type DISubprogram = &'ll DISubprogram;
}

impl ty::layout::HasTyCtxt<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl ty::layout::HasDataLayout for SirCodegenCx<'ll, 'tcx> {
    fn data_layout(&self) -> &ty::layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl LayoutOf for SirCodegenCx<'ll, 'tcx> {
    type Ty = Ty<'tcx>;
    type TyLayout = TyLayout<'tcx>;

    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
        self.spanned_layout_of(ty, DUMMY_SP)
    }

    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::TyLayout {
        self.tcx.layout_of(ty::ParamEnv::reveal_all().and(ty))
            .unwrap_or_else(|e| if let LayoutError::SizeOverflow(_) = e {
                self.sess().span_fatal(span, &e.to_string())
            } else {
                bug!("failed to get layout for `{}`: {}", ty, e)
            })
    }
}


impl LayoutTypeMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn backend_type(&self, layout: TyLayout<'tcx>) -> TypeIdx {
	TypeIdx::from_usize(0) // FIXME
    }

    fn immediate_backend_type(&self, layout: TyLayout<'tcx>) -> TypeIdx {
	TypeIdx::from_usize(0) // FIXME
    }

    fn is_backend_immediate(&self, ty: TyLayout<'tcx>) -> bool {
        match ty.abi {
            layout::Abi::Scalar(_) |
            layout::Abi::Vector { .. } => true,
            layout::Abi::ScalarPair(..) => false,
            layout::Abi::Uninhabited |
            layout::Abi::Aggregate { .. } => ty.is_zst()
        }
    }

    fn is_backend_scalar_pair(&self, ty: TyLayout<'tcx>) -> bool {
        match ty.abi {
            layout::Abi::ScalarPair(..) => true,
            layout::Abi::Uninhabited |
            layout::Abi::Scalar(_) |
            layout::Abi::Vector { .. } |
            layout::Abi::Aggregate { .. } => false
        }
    }

    fn backend_field_index(&self, layout: TyLayout<'tcx>, index: usize) -> u64 {
        0 // FIXME
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: TyLayout<'tcx>,
        index: usize,
        immediate: bool
    ) -> TypeIdx {
	TypeIdx::from_usize(0) // FIXME
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> TypeIdx {
	TypeIdx::from_usize(0) // FIXME
    }

    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> TypeIdx {
	TypeIdx::from_usize(0) // FIXME
    }

    fn reg_backend_type(&self, ty: &Reg) -> TypeIdx {
	TypeIdx::from_usize(0) // FIXME
    }
}

#[must_use]
pub struct SirBuilder<'a, 'll, 'tcx> {
    cx: &'a SirCodegenCx<'ll, 'tcx>,
    insertion_point: InstructionPath,
}

impl SirBuilder<'a, 'll, 'tcx> {
    fn emit_instr(&mut self, instr: Instruction) -> Value {
        let ip = self.insertion_point;

        {
            let func = &mut self.functions.borrow_mut()[ip.func_idx];
            let block = &mut func.blocks[ip.block_idx];
            block.instrs.insert(ip.instr_idx, instr);
        }

        let instr_path = self.insertion_point;
        self.insertion_point.instr_idx.increment_by(1);
        Value::Instruction(instr_path)
    }

    fn unimplemented(&mut self, msg: &'static str) -> Value {
        self.emit_instr(Instruction::Unimplemented(msg))
    }

    fn llfn(&self) -> Value {
        Value::Function(self.insertion_point.func_idx)
    }
}

impl LayoutOf for SirBuilder<'a, 'll, 'tcx> {
    type Ty = Ty<'tcx>;
    type TyLayout = TyLayout<'tcx>;

    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
        self.cx.layout_of(ty)
    }

    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::TyLayout {
        unimplemented!();
    }
}

impl ty::layout::HasDataLayout for SirBuilder<'a, 'll, 'tcx> {
    fn data_layout(&self) -> &ty::layout::TargetDataLayout {
        self.cx.data_layout()
    }
}

impl ty::layout::HasTyCtxt<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}

impl Deref for SirBuilder<'a, 'll, 'tcx> {
    type Target = SirCodegenCx<'ll, 'tcx>;

    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl HasTargetSpec for SirBuilder<'a, 'll, 'tcx> {
    fn target_spec(&self) -> &Target {
        unimplemented!();
    }
}

impl StaticBuilderMethods for SirBuilder<'a, 'll, 'tcx> {
    fn get_static(&mut self, def_id: DefId) -> Value {
        unimplemented!();
    }
}

impl AsmBuilderMethods<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        ia: &hir::InlineAsm,
        outputs: Vec<PlaceRef<'tcx, Value>>,
        mut _inputs: Vec<Value>,
        span: Span,
    ) -> bool {
        unimplemented!();
    }
}

impl AbiBuilderMethods<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn apply_attrs_callsite(
        &mut self,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        callsite: Self::Value
    ) {
        // Intentionally left blank.
    }

    fn get_param(&self, index: usize) -> Self::Value {
        Value::Dummy // FIXME
    }
}

impl DebugInfoBuilderMethods<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn declare_local(
        &mut self,
        dbg_context: &FunctionDebugContext<&'ll DIScope>,
        variable_name: ast::Name,
        variable_type: Ty<'tcx>,
        scope_metadata: &'ll DIScope,
        variable_alloca: Self::Value,
        direct_offset: Size,
        indirect_offsets: &[Size],
        variable_kind: VariableKind,
        span: Span,
    ) {
        unimplemented!();
    }

    fn set_source_location(
        &mut self,
        debug_context: &mut FunctionDebugContext<&'ll DIScope>,
        scope: &'ll DIScope,
        span: Span,
    ) {
        unimplemented!();
    }

    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {
        unimplemented!();
    }

    fn set_var_name(&mut self, value: Value, name: &str) {
        unimplemented!();
    }
}


impl IntrinsicCallMethods<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, Value>],
        llresult: Value,
        span: Span,
    ) {
        self.unimplemented("IntrinsicCallMethods::codegen_intrinsic_call");
    }

    fn abort(&mut self) {
        self.unimplemented("IntrinsicCallMethods::abort");
    }

    fn assume(&mut self, val: Self::Value) {
        self.unimplemented("IntrinsicCallMethods::assume");
    }

    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value {
        self.unimplemented("IntrinsicCallMethods::expect");
        Value::Dummy
    }

    fn sideeffect(&mut self) {
        self.unimplemented("IntrinsicCallMethods::sideeffect");
    }

    fn va_start(&mut self, va_list: Value) -> Value {
        self.unimplemented("IntrinsicCallMethods::va_start");
        Value::Dummy
    }

    fn va_end(&mut self, va_list: Value) -> Value {
        self.unimplemented("IntrinsicCallMethods::va_end");
        Value::Dummy
    }
}

impl HasCodegen<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    type CodegenCx = SirCodegenCx<'ll, 'tcx>;
}

impl ty::layout::HasParamEnv<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        ty::ParamEnv::reveal_all()
    }
}

impl ArgAbiMethods<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize, dst: PlaceRef<'tcx, Self::Value>
    ) {
        arg_abi.store_fn_arg(self, idx, dst)
    }

    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: Value,
        dst: PlaceRef<'tcx, Value>
    ) {
        arg_abi.store(self, val, dst)
    }

    fn arg_memory_ty(&self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>) -> TypeIdx {
        arg_abi.memory_ty(self)
    }
}

impl BackendTypes for SirBuilder<'a, 'll, 'tcx> {
    type Value = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::Value;
    type Function = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::Function;
    type BasicBlock = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::BasicBlock;
    type Type = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::Type;
    type Funclet = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::Funclet;
    type DIScope = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::DIScope;
    type DISubprogram = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::DISubprogram;
}

impl BuilderMethods<'a, 'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn new_block<'b>(cx: &'a Self::CodegenCx, llfn: Self::Function, name: &'b str) -> Self {
        let mut bx = SirBuilder::with_cx(cx);
        let fn_idx = llfn.unwrap_function();
        info!("{:?}", fn_idx);

        let block_path = {
            let mut fns = cx.functions.borrow_mut();
            let func = &mut fns[fn_idx];
            let block_idx = func.add_block(fn_idx);
            info!("SirBuilder::new_block: for={:?}, idx={:?}", llfn, block_idx);
            BasicBlockPath::new(llfn.unwrap_function(), block_idx)
        };

        bx.position_at_end(block_path);
        bx
    }

    fn with_cx(cx: &'a Self::CodegenCx) -> Self {
        Self {
            cx,
            insertion_point: InstructionPath::default(),
        }
    }

    fn build_sibling_block(&self, name: &str) -> Self {
        SirBuilder::new_block(self.cx,
            Value::Function(self.insertion_point.func_idx),
            name)
    }

    fn cx(&self) -> &Self::CodegenCx {
        self.cx
    }

    fn llbb(&self) -> Self::BasicBlock {
        BasicBlockPath{func_idx: self.insertion_point.func_idx,
                       block_idx: self.insertion_point.block_idx}
    }

    fn first_instruction(&mut self, llbb: Self::BasicBlock) -> Option<Self::Value> {
        unimplemented!();
    }

    fn add_yk_block_label(&mut self, di_sp: Self::DIScope, lbl_name: CString) {
        // Purposely blank.
    }

    fn add_yk_block_label_at_end(&mut self, di_sp: Self::DIScope, lbl_name: CString) {
        // Purposely blank.
    }

    fn position_before(&mut self, instr: Self::Value) {
        self.insertion_point = instr.unwrap_instr();
    }

    fn position_at_end(&mut self, llbb: Self::BasicBlock) {
        info!("position_at_end: before={:?}", llbb);
        let instr_idx = {
            let funcs = self.functions.borrow();
            let func = &funcs[llbb.func_idx];
            let block = &func.blocks[llbb.block_idx];
            block.instrs.len()
        };

        self.insertion_point.func_idx = llbb.func_idx;
        self.insertion_point.block_idx = llbb.block_idx;
        self.insertion_point.instr_idx = InstructionIdx::from_usize(instr_idx);
        info!("position_at_end: after={:?}", self.insertion_point);
    }

    fn ret_void(&mut self) {
        self.unimplemented("BuilderMethods::ret_void");
    }

    fn ret(&mut self, v: Self::Value) {
        self.unimplemented("BuilderMethods::ret");
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        self.unimplemented("BuilderMethods::br");
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        self.unimplemented("BuilderMethods::cond_br");
    }

    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock)> + TrustedLen,
    ) {
        self.unimplemented("BuilderMethods::switch");
    }

    fn invoke(
        &mut self,
        llfn: Self::Value,
        args: &[Self::Value],
        then: Self::BasicBlock,
        catch: Self::BasicBlock,
        funclet: Option<&Self::Funclet>,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::invoke");
        Value::Dummy
    }

    fn unreachable(&mut self) {
        self.unimplemented("BuilderMethods::unimplemented");
    }

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::add");
        Value::Dummy
    }

    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fadd");
        Value::Dummy
    }

    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fadd_fast");
        Value::Dummy
    }

    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::sub");
        Value::Dummy
    }

    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fsub");
        Value::Dummy
    }

    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fsub_fast");
        Value::Dummy
    }

    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::mul");
        Value::Dummy
    }

    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fmul");
        Value::Dummy
    }

    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fmul_fast");
        Value::Dummy
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::udiv");
        Value::Dummy
    }

    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::exactudiv");
        Value::Dummy
    }

    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::sdiv");
        Value::Dummy
    }

    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::exactsdiv");
        Value::Dummy
    }

    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fdiv");
        Value::Dummy
    }

    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fdiv_fast");
        Value::Dummy
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::urem");
        Value::Dummy
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::srem");
        Value::Dummy
    }

    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::frem");
        Value::Dummy
    }

    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::frem_fast");
        Value::Dummy
    }

    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::shl");
        Value::Dummy
    }

    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::lshr");
        Value::Dummy
    }

    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::ashr");
        Value::Dummy
    }

    fn unchecked_sadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::unchecked_sadd");
        Value::Dummy
    }

    fn unchecked_uadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::unchecked_uadd");
        Value::Dummy
    }

    fn unchecked_ssub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::unchecked_ssub");
        Value::Dummy
    }

    fn unchecked_usub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::unchecked_usub");
        Value::Dummy
    }

    fn unchecked_smul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::unchecked_smul");
        Value::Dummy
    }

    fn unchecked_umul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::unchecked_umul");
        Value::Dummy
    }

    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::and");
        Value::Dummy
    }

    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::or");
        Value::Dummy
    }

    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::xor");
        Value::Dummy
    }

    fn neg(&mut self, v: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::neg");
        Value::Dummy
    }

    fn fneg(&mut self, v: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fneg");
        Value::Dummy
    }

    fn not(&mut self, v: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::not");
        Value::Dummy
    }

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        self.unimplemented("BuilderMethods::checked_binop");
        (Value::Dummy, Value::Dummy)
    }

    fn alloca(&mut self, ty: Self::Type, align: Align) -> Self::Value {
        self.unimplemented("BuilderMethods::alloca");
        Value::Dummy
    }

    fn dynamic_alloca(&mut self, ty: Self::Type, align: Align) -> Self::Value {
        self.unimplemented("BuilderMethods::dynamic_alloca");
        Value::Dummy
    }

    fn array_alloca(
        &mut self,
        ty: Self::Type,
        len: Self::Value,
        align: Align,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::array_alloca");
        Value::Dummy
    }


    fn load(&mut self, ptr: Self::Value, align: Align) -> Self::Value {
        self.unimplemented("BuilderMethods::load");
        Value::Dummy
    }

    fn volatile_load(&mut self, ptr: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::volatile_load");
        Value::Dummy
    }

    fn atomic_load(&mut self, ptr: Self::Value, order: AtomicOrdering, size: Size) -> Self::Value {
        self.unimplemented("BuilderMethods::atomic_load");
        Value::Dummy
    }

    fn load_operand(&mut self, place: PlaceRef<'tcx, Self::Value>)
        -> OperandRef<'tcx, Self::Value>
    {
        OperandRef{val: OperandValue::Immediate(Value::Dummy), layout: place.layout} // FIXME
    }

    fn write_operand_repeatedly(
        self,
        elem: OperandRef<'tcx, Self::Value>,
        count: u64,
        dest: PlaceRef<'tcx, Self::Value>,
    ) -> Self {
        unimplemented!();
    }

    fn range_metadata(&mut self, load: Self::Value, range: Range<u128>) {
        unimplemented!();
    }

    fn nonnull_metadata(&mut self, load: Self::Value) {
        unimplemented!();
    }

    fn store(&mut self, val: Self::Value, ptr: Self::Value, align: Align) -> Self::Value {
        self.unimplemented("BuilderMethods::store");
        Value::Dummy
    }

    fn store_with_flags(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: Align,
        flags: MemFlags,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::store_with_flags");
        Value::Dummy
    }

    fn atomic_store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: Size,
    ) {
        self.unimplemented("BuilderMethods::atomic_store");
    }


    fn gep(&mut self, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value {
        self.unimplemented("BuilderMethods::gep");
        Value::Dummy
    }

    fn inbounds_gep(&mut self, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value {
        self.unimplemented("BuilderMethods::inbounds_gep");
        Value::Dummy
    }

    fn struct_gep(&mut self, ptr: Self::Value, idx: u64) -> Self::Value {
        self.unimplemented("BuilderMethods::struct_gep");
        Value::Dummy
    }


    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::trunc");
        Value::Dummy
    }

    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::sext");
        Value::Dummy
    }

    fn fptoui(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::fptoui");
        Value::Dummy
    }

    fn fptosi(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::fptosi");
        Value::Dummy
    }

    fn uitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::uitofp");
        Value::Dummy
    }

    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::sitofp");
        Value::Dummy
    }

    fn fptrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::fptrunc");
        Value::Dummy
    }

    fn fpext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::fpext");
        Value::Dummy
    }

    fn ptrtoint(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::ptrtoint");
        Value::Dummy
    }

    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::inttoptr");
        Value::Dummy
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::bitcast");
        Value::Dummy
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        self.unimplemented("BuilderMethods::intcast");
        Value::Dummy
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::pointercast");
        Value::Dummy
    }

    fn icmp(&mut self, op: IntPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::icmp");
        Value::Dummy
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::fcmp");
        Value::Dummy
    }

    fn memcpy(
        &mut self,
        dst: Self::Value,
        dst_align: Align,
        src: Self::Value,
        src_align: Align,
        size: Self::Value,
        flags: MemFlags,
    ) {
        self.unimplemented("BuilderMethods::memcpy");
    }

    fn memmove(
        &mut self,
        dst: Self::Value,
        dst_align: Align,
        src: Self::Value,
        src_align: Align,
        size: Self::Value,
        flags: MemFlags,
    ) {
        self.unimplemented("BuilderMethods::memmove");
    }

    fn memset(
        &mut self,
        ptr: Self::Value,
        fill_byte: Self::Value,
        size: Self::Value,
        align: Align,
        flags: MemFlags,
    ) {
        self.unimplemented("memset");
    }


    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::select");
        Value::Dummy
    }

    fn va_arg(&mut self, list: Self::Value, ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::va_arg");
        Value::Dummy
    }

    fn extract_element(&mut self, vec: Self::Value, idx: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::extract_element");
        Value::Dummy
    }

    fn vector_splat(&mut self, num_elts: usize, elt: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::vector_splat");
        Value::Dummy
    }

    fn extract_value(&mut self, agg_val: Self::Value, idx: u64) -> Self::Value {
        self.unimplemented("BuilderMethods::extract_value");
        Value::Dummy
    }

    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value {
        self.unimplemented("BuilderMethods::insert_value");
        Value::Dummy
    }

    fn landing_pad(
        &mut self,
        ty: Self::Type,
        pers_fn: Self::Value,
        num_clauses: usize,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::landing_pad");
        Value::Dummy
    }

    fn set_cleanup(&mut self, landing_pad: Self::Value) {
        self.unimplemented("BuilderMethods::set_cleanup");
    }

    fn resume(&mut self, exn: Self::Value) -> Self::Value {
        self.unimplemented("BuilderMethods::resume");
        Value::Dummy
    }

    fn cleanup_pad(&mut self, parent: Option<Self::Value>, args: &[Self::Value]) -> Self::Funclet {
        unimplemented!();
    }

    fn cleanup_ret(
        &mut self,
        funclet: &Self::Funclet,
        unwind: Option<Self::BasicBlock>,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::cleanup_ret");
        Value::Dummy
    }

    fn catch_pad(&mut self, parent: Self::Value, args: &[Self::Value]) -> Self::Funclet {
        unimplemented!();
    }

    fn catch_switch(
        &mut self,
        parent: Option<Self::Value>,
        unwind: Option<Self::BasicBlock>,
        num_handlers: usize,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::catch_switch");
        Value::Dummy
    }

    fn add_handler(&mut self, catch_switch: Self::Value, handler: Self::BasicBlock) {
        unimplemented!();
    }

    fn set_personality_fn(&mut self, personality: Self::Value) {
	// Intentionally blank.
    }

    fn atomic_cmpxchg(
        &mut self,
        dst: Self::Value,
        cmp: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::atomic_cmpxchg");
        Value::Dummy
    }

    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::atomic_rmw");
        Value::Dummy
    }

    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope) {
        self.unimplemented("BuilderMethods::atomic_fence");
    }

    fn set_invariant_load(&mut self, load: Self::Value) {
        self.unimplemented("BuilderMethods::set_invariant_load");
    }

    fn lifetime_start(&mut self, ptr: Self::Value, size: Size) {
        // Intentionally blank.
    }

    fn lifetime_end(&mut self, ptr: Self::Value, size: Size) {
        // Intentionally blank.
    }

    fn call(
        &mut self,
        llfn: Self::Value,
        args: &[Self::Value],
        funclet: Option<&Self::Funclet>,
    ) -> Self::Value {
        self.unimplemented("BuilderMethods::call");
        Value::Dummy
    }

    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.unimplemented("BuilderMethods::zext");
        Value::Dummy
    }

    unsafe fn delete_basic_block(&mut self, bb: Self::BasicBlock) {
        unimplemented!();
    }

    fn do_not_inline(&mut self, llret: Self::Value) {
        // Intentionally left blank.
    }
}
