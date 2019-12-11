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
use crate::mir::debuginfo::FunctionDebugContext;
use rustc::hir::def_id::CrateNum;
use rustc::hir;
use rustc::mir;
use rustc::mir::mono::{Linkage, Visibility};
use std::cell::RefCell;
use std::sync::Arc;
use rustc::session::Session;
use rustc::mir::mono::CodegenUnit;
use rustc::util::nodemap::FxHashMap;
use crate::common::{self, IntPredicate, RealPredicate, AtomicOrdering, AtomicRmwBinOp,
                    SynchronizationScope};
use crate::MemFlags;
use rustc::ty::{self, Ty, TyCtxt, Instance, PolyFnSig};
use rustc::ty::layout::{self, Align, Size, TyLayout, LayoutError, LayoutOf};
use rustc::hir::def_id::DefId;
use crate::traits::*;
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;
use rustc_target::spec::{HasTargetSpec, Target};
use rustc_target::abi::call::{ArgAbi, FnAbi, CastTarget, Reg};
use std::ffi::CString;
use std::ops::Range;
use std::iter::TrustedLen;

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

#[derive(Debug, PartialEq, Copy, Clone)]
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

//impl InstructionPath {
//    fn new(func_idx: FunctionIdx,
//           block_idx: BasicBlockIdx,
//           instr_idx: InstructionIdx) -> Self
//    {
//        Self {func_idx, block_idx, instr_idx}
//    }
//}

#[allow(dead_code)]
struct DILexicalBlock {}

/// The all-encompassing enum.
/// Each is an index into the corresponding vector in SirCodegenCx.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Value {
    Function(FunctionIdx),
    Global(GlobalIdx),
    Instruction(InstructionPath),
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

#[derive(Debug, PartialEq, Clone)]
pub struct Type {}

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    name: String,
    ty: TypeIdx,
    blocks: IndexVec<BasicBlockIdx, BasicBlock>,
}

impl Function {
    fn add_block(&mut self, parent: FunctionIdx) -> BasicBlockIdx {
        let idx = self.blocks.len();
        self.blocks.push(BasicBlock::new(parent));
        BasicBlockIdx::from_usize(idx)
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

    pub types: IndexVec<TypeIdx, Type>,
    pub dummy_type_idx: TypeIdx,

    pub globals: RefCell<IndexVec<GlobalIdx, Global>>,
    pub globals_cache: RefCell<FxHashMap<String, GlobalIdx>>,

    // FIXME: almost certain this lifetime will crop up later.
    pd: PhantomData<&'ll ()>,
}

impl SirCodegenCx<'ll, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        codegen_unit: Arc<CodegenUnit<'tcx>>,
    ) -> Self {
        Self {
            tcx,
            codegen_unit,
            functions: RefCell::new(IndexVec::new()),
            instances: RefCell::new(FxHashMap::default()),
            types: IndexVec::from_elem_n(Type{}, 1),
            dummy_type_idx: TypeIdx::from_usize(0),
            globals: RefCell::new(IndexVec::default()),
            globals_cache: RefCell::new(FxHashMap::default()),
            pd: PhantomData,
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
        unimplemented!();
    }

    fn const_undef(&self, t: TypeIdx) -> Value {
        unimplemented!();
    }

    fn const_int(&self, t: TypeIdx, i: i64) -> Value {
        unimplemented!();
    }

    fn const_uint(&self, t: TypeIdx, i: u64) -> Value {
        unimplemented!();
    }

    fn const_uint_big(&self, t: TypeIdx, u: u128) -> Value {
        unimplemented!();
    }

    fn const_bool(&self, val: bool) -> Value {
        unimplemented!();
    }

    fn const_i32(&self, i: i32) -> Value {
        unimplemented!();
    }

    fn const_u32(&self, i: u32) -> Value {
        unimplemented!();
    }

    fn const_u64(&self, i: u64) -> Value {
        unimplemented!();
    }

    fn const_usize(&self, i: u64) -> Value {
        unimplemented!();
    }

    fn const_u8(&self, i: u8) -> Value {
        unimplemented!();
    }

    fn const_real(&self, t: TypeIdx, val: f64) -> Value {
        unimplemented!();
    }

    fn const_str(&self, s: Symbol) -> (Value, Value) {
        unimplemented!();
    }

    fn const_struct(
        &self,
        elts: &[Value],
        packed: bool
    ) -> Value {
        unimplemented!();
    }

    fn const_to_opt_uint(&self, v: Value) -> Option<u64> {
        unimplemented!();
    }

    fn const_to_opt_u128(&self, v: Value, sign_ext: bool) -> Option<u128> {
        unimplemented!();
    }

    fn scalar_to_backend(
        &self,
        cv: Scalar,
        layout: &layout::Scalar,
        llty: TypeIdx,
    ) -> Value {
        unimplemented!();
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
        unimplemented!();
    }
}

fn declare_raw_fn(
    cx: &SirCodegenCx<'ll, 'tcx>,
    name: &str,
    ty: TypeIdx,
) -> Value {
    info!("declare_raw_fn(name={:?}, ty={:?})", name, ty);
    // FIXME calling convention and ABI stuff?
    let mut fns = cx.functions.borrow_mut();
    let idx = fns.len();

    fns.push(Function {
        name: name.to_owned(),
        ty,
        blocks: IndexVec::default(),
    });

    Value::Function(FunctionIdx::from_usize(idx))
}

impl DeclareMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn declare_global(
        &self,
        name: &str, ty: TypeIdx
    ) -> Value {
        info!("declare_global(name={:?})", name);
        let mut globals = self.globals.borrow_mut();

        let idx = globals.len();
        globals.push(Global {
            name: name.to_owned(),
            ty
        });

        Value::Global(GlobalIdx::from_usize(idx))
    }

    fn declare_cfn(
        &self,
        name: &str,
        fn_type: TypeIdx
    ) -> Value {
        unimplemented!();
    }

    fn declare_fn(
        &self,
        name: &str,
        sig: PolyFnSig<'tcx>,
    ) -> Value {
        info!("declare_rust_fn(name={:?}, sig={:?})", name, sig);
        let sig = self.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
        info!("declare_rust_fn (after region erasure) sig={:?}", sig);

        declare_raw_fn(self, name, self.dummy_type_idx)
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
        if let Some(idx) = self.globals_cache.borrow().get(name) {
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
        unimplemented!();
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
        unimplemented!();
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
        unimplemented!();
    }

    fn eh_personality(&self) -> Value {
        unimplemented!();
    }

    fn eh_unwind_resume(&self) -> Value {
        unimplemented!();
    }

    fn sess(&self) -> &Session {
        unimplemented!();
    }

    fn check_overflow(&self) -> bool {
        unimplemented!();
    }

    fn codegen_unit(&self) -> &Arc<CodegenUnit<'tcx>> {
        unimplemented!();
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
        unimplemented!();
    }

    fn type_i8(&self) -> TypeIdx {
        unimplemented!();
    }

    fn type_i16(&self) -> TypeIdx {
        unimplemented!();
    }

    fn type_i32(&self) -> TypeIdx {
        unimplemented!();
    }

    fn type_i64(&self) -> TypeIdx {
        unimplemented!();
    }

    fn type_i128(&self) -> TypeIdx {
        unimplemented!();
    }

    fn type_isize(&self) -> TypeIdx {
        unimplemented!();
    }

    fn type_f32(&self) -> TypeIdx {
        unimplemented!();
    }

    fn type_f64(&self) -> TypeIdx {
        unimplemented!();
    }

    fn type_func(
        &self,
        args: &[TypeIdx],
        ret: TypeIdx
    ) -> TypeIdx {
        unimplemented!();
    }

    fn type_struct(
        &self,
        els: &[TypeIdx],
        packed: bool
    ) -> TypeIdx {
        unimplemented!();
    }

    fn type_kind(&self, ty: TypeIdx) -> common::TypeKind {
        unimplemented!();
    }

    fn type_ptr_to(&self, ty: TypeIdx) -> TypeIdx {
        unimplemented!();
    }

    fn element_type(&self, ty: TypeIdx) -> TypeIdx {
        unimplemented!();
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
        unimplemented!();
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
        unimplemented!();
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
        unimplemented!();
    }

    fn immediate_backend_type(&self, layout: TyLayout<'tcx>) -> TypeIdx {
        unimplemented!();
    }

    fn is_backend_immediate(&self, layout: TyLayout<'tcx>) -> bool {
        unimplemented!();
    }

    fn is_backend_scalar_pair(&self, layout: TyLayout<'tcx>) -> bool {
        unimplemented!();
    }

    fn backend_field_index(&self, layout: TyLayout<'tcx>, index: usize) -> u64 {
        unimplemented!();
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: TyLayout<'tcx>,
        index: usize,
        immediate: bool
    ) -> TypeIdx {
        unimplemented!();
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> TypeIdx {
        unimplemented!();
    }

    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> TypeIdx {
        unimplemented!();
    }

    fn reg_backend_type(&self, ty: &Reg) -> TypeIdx {
        unimplemented!();
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
        self.insertion_point.block_idx.increment_by(1);
        Value::Instruction(instr_path)
    }

    fn unimplemented(&mut self, msg: &'static str) -> Value {
        self.emit_instr(Instruction::Unimplemented(msg))
    }
}

impl LayoutOf for SirBuilder<'a, 'll, 'tcx> {
    type Ty = Ty<'tcx>;
    type TyLayout = TyLayout<'tcx>;

    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
        unimplemented!();
    }

    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::TyLayout {
        unimplemented!();
    }
}

impl ty::layout::HasDataLayout for SirBuilder<'a, 'll, 'tcx> {
    fn data_layout(&self) -> &ty::layout::TargetDataLayout {
        unimplemented!();
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
        unimplemented!();
    }

    fn get_param(&self, index: usize) -> Self::Value {
        unimplemented!();
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
        self.unimplemented("IntrinsicCallMethods::expect")
    }

    fn sideeffect(&mut self) {
        self.unimplemented("IntrinsicCallMethods::sideeffect");
    }

    fn va_start(&mut self, va_list: Value) -> Value {
        self.unimplemented("IntrinsicCallMethods::va_start")
    }

    fn va_end(&mut self, va_list: Value) -> Value {
        self.unimplemented("IntrinsicCallMethods::va_end")
    }
}

impl HasCodegen<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    type CodegenCx = SirCodegenCx<'ll, 'tcx>;
}

impl ty::layout::HasParamEnv<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        unimplemented!();
    }
}

impl ArgAbiMethods<'tcx> for SirBuilder<'a, 'll, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize, dst: PlaceRef<'tcx, Self::Value>
    ) {
        unimplemented!();
    }

    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: Value,
        dst: PlaceRef<'tcx, Value>
    ) {
        unimplemented!();
    }

    fn arg_memory_ty(&self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>) -> TypeIdx {
        unimplemented!();
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

        let block_path = {
            let mut fns = cx.functions.borrow_mut();
            let func = &mut fns[fn_idx];
            let block_idx = func.add_block(fn_idx);
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
        unimplemented!();
    }

    fn cx(&self) -> &Self::CodegenCx {
        unimplemented!();
    }

    fn llbb(&self) -> Self::BasicBlock {
        unimplemented!();
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
        let instr_idx = {
            let funcs = self.functions.borrow();
            funcs[llbb.func_idx].blocks[llbb.block_idx].instrs.len()
        };

        self.insertion_point.func_idx = llbb.func_idx;
        self.insertion_point.block_idx = llbb.block_idx;
        self.insertion_point.instr_idx = InstructionIdx::from_usize(instr_idx);
    }

    fn ret_void(&mut self) {
        unimplemented!();
    }

    fn ret(&mut self, v: Self::Value) {
        unimplemented!();
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        unimplemented!();
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        unimplemented!();
    }

    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock)> + TrustedLen,
    ) {
        unimplemented!();
    }

    fn invoke(
        &mut self,
        llfn: Self::Value,
        args: &[Self::Value],
        then: Self::BasicBlock,
        catch: Self::BasicBlock,
        funclet: Option<&Self::Funclet>,
    ) -> Self::Value {
        unimplemented!();
    }

    fn unreachable(&mut self) {
        unimplemented!();
    }


    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn unchecked_sadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn unchecked_uadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn unchecked_ssub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn unchecked_usub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn unchecked_smul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn unchecked_umul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn neg(&mut self, v: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fneg(&mut self, v: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn not(&mut self, v: Self::Value) -> Self::Value {
        unimplemented!();
    }


    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        unimplemented!();
    }


    fn alloca(&mut self, ty: Self::Type, align: Align) -> Self::Value {
        unimplemented!();
    }

    fn dynamic_alloca(&mut self, ty: Self::Type, align: Align) -> Self::Value {
        unimplemented!();
    }

    fn array_alloca(
        &mut self,
        ty: Self::Type,
        len: Self::Value,
        align: Align,
    ) -> Self::Value {
        unimplemented!();
    }


    fn load(&mut self, ptr: Self::Value, align: Align) -> Self::Value {
        unimplemented!();
    }

    fn volatile_load(&mut self, ptr: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn atomic_load(&mut self, ptr: Self::Value, order: AtomicOrdering, size: Size) -> Self::Value {
        unimplemented!();
    }

    fn load_operand(&mut self, place: PlaceRef<'tcx, Self::Value>)
        -> OperandRef<'tcx, Self::Value>
    {
        unimplemented!();
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
        unimplemented!();
    }

    fn store_with_flags(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: Align,
        flags: MemFlags,
    ) -> Self::Value {
        unimplemented!();
    }

    fn atomic_store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: Size,
    ) {
        unimplemented!();
    }


    fn gep(&mut self, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value {
        unimplemented!();
    }

    fn inbounds_gep(&mut self, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value {
        unimplemented!();
    }

    fn struct_gep(&mut self, ptr: Self::Value, idx: u64) -> Self::Value {
        unimplemented!();
    }


    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn fptoui(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn fptosi(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn uitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn fptrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn fpext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn ptrtoint(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        unimplemented!();
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }


    fn icmp(&mut self, op: IntPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        unimplemented!();
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
        unimplemented!();
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
        unimplemented!();
    }

    fn memset(
        &mut self,
        ptr: Self::Value,
        fill_byte: Self::Value,
        size: Self::Value,
        align: Align,
        flags: MemFlags,
    ) {
        unimplemented!();
    }


    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value {
        unimplemented!();
    }


    fn va_arg(&mut self, list: Self::Value, ty: Self::Type) -> Self::Value {
        unimplemented!();
    }

    fn extract_element(&mut self, vec: Self::Value, idx: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn vector_splat(&mut self, num_elts: usize, elt: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn extract_value(&mut self, agg_val: Self::Value, idx: u64) -> Self::Value {
        unimplemented!();
    }

    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value {
        unimplemented!();
    }


    fn landing_pad(
        &mut self,
        ty: Self::Type,
        pers_fn: Self::Value,
        num_clauses: usize,
    ) -> Self::Value {
        unimplemented!();
    }

    fn set_cleanup(&mut self, landing_pad: Self::Value) {
        unimplemented!();
    }

    fn resume(&mut self, exn: Self::Value) -> Self::Value {
        unimplemented!();
    }

    fn cleanup_pad(&mut self, parent: Option<Self::Value>, args: &[Self::Value]) -> Self::Funclet {
        unimplemented!();
    }

    fn cleanup_ret(
        &mut self,
        funclet: &Self::Funclet,
        unwind: Option<Self::BasicBlock>,
    ) -> Self::Value {
        unimplemented!();
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
        unimplemented!();
    }

    fn add_handler(&mut self, catch_switch: Self::Value, handler: Self::BasicBlock) {
        unimplemented!();
    }

    fn set_personality_fn(&mut self, personality: Self::Value) {
        unimplemented!();
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
        unimplemented!();
    }

    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
    ) -> Self::Value {
        unimplemented!();
    }

    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope) {
        unimplemented!();
    }

    fn set_invariant_load(&mut self, load: Self::Value) {
        unimplemented!();
    }

    fn lifetime_start(&mut self, ptr: Self::Value, size: Size) {
        unimplemented!();
    }

    fn lifetime_end(&mut self, ptr: Self::Value, size: Size) {
        unimplemented!();
    }


    fn call(
        &mut self,
        llfn: Self::Value,
        args: &[Self::Value],
        funclet: Option<&Self::Funclet>,
    ) -> Self::Value {
        unimplemented!();
    }

    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        unimplemented!();
    }


    unsafe fn delete_basic_block(&mut self, bb: Self::BasicBlock) {
        unimplemented!();
    }

    fn do_not_inline(&mut self, llret: Self::Value) {
        unimplemented!();
    }

}
