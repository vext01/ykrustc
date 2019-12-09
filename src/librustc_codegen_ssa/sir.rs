//! FIXME

// FIXME jsut for now, or I will go mad.
#![allow(unused_imports)]
#![allow(unused_variables)]

use syntax::symbol::Symbol;
use syntax::source_map::Span;
use syntax::ast;
use crate::mir::debuginfo::VariableKind;
use rustc::mir::interpret::{Scalar, Allocation};
use std::ops::Deref;
use std::marker::PhantomData;
use crate::mir::debuginfo::{FunctionDebugContext};
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
use rustc::ty::layout::{self, Align, Size, TyLayout};
use rustc::ty::layout::LayoutOf;
use rustc::hir::def_id::DefId;
use crate::traits::*;
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;
use rustc_target::spec::{HasTargetSpec, Target};
use rustc_target::abi::call::{ArgAbi, FnAbi, CastTarget, Reg};
use std::ffi::CString;
use std::ops::Range;
use std::iter::TrustedLen;

#[allow(dead_code)]
struct DILexicalBlock {}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Value {}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Function {}

#[derive(Debug, Copy, Clone)]
pub struct BasicBlock {}

#[derive(Debug, Copy, PartialEq, Clone)]
pub struct Type {}

#[derive(Debug)]
pub struct Funclet {}

#[derive(Debug, Copy, Clone)]
pub struct DIScope {}

#[derive(Debug, Copy, Clone)]
pub struct DISubprogram {}

pub struct SirCodegenCx<'ll, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub codegen_unit: Arc<CodegenUnit<'tcx>>,
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
    fn const_null(&self, t: &'ll Type) -> &'ll Value {
        unimplemented!();
    }

    fn const_undef(&self, t: &'ll Type) -> &'ll Value {
        unimplemented!();
    }

    fn const_int(&self, t: &'ll Type, i: i64) -> &'ll Value {
        unimplemented!();
    }

    fn const_uint(&self, t: &'ll Type, i: u64) -> &'ll Value {
        unimplemented!();
    }

    fn const_uint_big(&self, t: &'ll Type, u: u128) -> &'ll Value {
        unimplemented!();
    }

    fn const_bool(&self, val: bool) -> &'ll Value {
        unimplemented!();
    }

    fn const_i32(&self, i: i32) -> &'ll Value {
        unimplemented!();
    }

    fn const_u32(&self, i: u32) -> &'ll Value {
        unimplemented!();
    }

    fn const_u64(&self, i: u64) -> &'ll Value {
        unimplemented!();
    }

    fn const_usize(&self, i: u64) -> &'ll Value {
        unimplemented!();
    }

    fn const_u8(&self, i: u8) -> &'ll Value {
        unimplemented!();
    }

    fn const_real(&self, t: &'ll Type, val: f64) -> &'ll Value {
        unimplemented!();
    }

    fn const_str(&self, s: Symbol) -> (&'ll Value, &'ll Value) {
        unimplemented!();
    }

    fn const_struct(
        &self,
        elts: &[&'ll Value],
        packed: bool
    ) -> &'ll Value {
        unimplemented!();
    }

    fn const_to_opt_uint(&self, v: &'ll Value) -> Option<u64> {
        unimplemented!();
    }

    fn const_to_opt_u128(&self, v: &'ll Value, sign_ext: bool) -> Option<u128> {
        unimplemented!();
    }

    fn scalar_to_backend(
        &self,
        cv: Scalar,
        layout: &layout::Scalar,
        llty: &'ll Type,
    ) -> &'ll Value {
        unimplemented!();
    }

    fn from_const_alloc(
        &self,
        layout: TyLayout<'tcx>,
        alloc: &Allocation,
        offset: Size,
    ) -> PlaceRef<'tcx, &'ll Value> {
        unimplemented!();
    }

    fn const_ptrcast(&self, val: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unimplemented!();
    }
}

impl DeclareMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn declare_global(
        &self,
        name: &str, ty: &'ll Type
    ) -> &'ll Value {
        unimplemented!();
    }

    fn declare_cfn(
        &self,
        name: &str,
        fn_type: &'ll Type
    ) -> &'ll Function {
        unimplemented!();
    }

    fn declare_fn(
        &self,
        name: &str,
        sig: PolyFnSig<'tcx>,
    ) -> &'ll Function {
        unimplemented!();
    }

    fn define_global(
        &self,
        name: &str,
        ty: &'ll Type
    ) -> Option<&'ll Value> {
        unimplemented!();
    }

    fn define_private_global(&self, ty: &'ll Type) -> &'ll Value {
        unimplemented!();
    }

    fn define_fn(
        &self,
        name: &str,
        fn_sig: PolyFnSig<'tcx>,
    ) -> &'ll Value {
        unimplemented!();
    }

    fn define_internal_fn(
        &self,
        name: &str,
        fn_sig: PolyFnSig<'tcx>,
    ) -> &'ll Value {
        unimplemented!();
    }

    fn get_declared_value(&self, name: &str) -> Option<&'ll Value> {
        unimplemented!();
    }

    fn get_defined_value(&self, name: &str) -> Option<&'ll Value> {
        unimplemented!();
    }
}

impl DebugInfoMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        sig: ty::FnSig<'tcx>,
        llfn: &'ll Function,
        mir: &mir::Body<'_>,
    ) -> Option<FunctionDebugContext<&'ll DIScope>> {
        unimplemented!();
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
        unimplemented!();
    }

    fn predefine_fn(&self,
                    instance: Instance<'tcx>,
                    linkage: Linkage,
                    visibility: Visibility,
                    symbol_name: &str) {
        unimplemented!();
    }
}

impl ty::layout::HasParamEnv<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        unimplemented!();
    }
}

impl StaticMethods for SirCodegenCx<'ll, 'tcx> {
    fn static_addr_of(
        &self,
        cv: &'ll Value,
        align: Align,
        kind: Option<&str>,
    ) -> &'ll Value {
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
                                Option<ty::PolyExistentialTraitRef<'tcx>>), &'ll Value>>
    {
        unimplemented!();
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> &'ll Function {
        unimplemented!();
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> &'ll Value {
        unimplemented!();
    }

    fn eh_personality(&self) -> &'ll Value {
        unimplemented!();
    }

    fn eh_unwind_resume(&self) -> &'ll Value {
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

    fn used_statics(&self) -> &RefCell<Vec<&'ll Value>> {
        unimplemented!();
    }

    fn set_frame_pointer_elimination(&self, llfn: &'ll Function) {
        unimplemented!();
    }

    fn apply_target_cpu_attr(&self, llfn: &'ll Function) {
        unimplemented!();
    }

    fn create_used_variable(&self) {
        unimplemented!();
    }
}

impl BaseTypeMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn type_i1(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_i8(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_i16(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_i32(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_i64(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_i128(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_isize(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_f32(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_f64(&self) -> &'ll Type {
        unimplemented!();
    }

    fn type_func(
        &self,
        args: &[&'ll Type],
        ret: &'ll Type
    ) -> &'ll Type {
        unimplemented!();
    }

    fn type_struct(
        &self,
        els: &[&'ll Type],
        packed: bool
    ) -> &'ll Type {
        unimplemented!();
    }

    fn type_kind(&self, ty: &'ll Type) -> common::TypeKind {
        unimplemented!();
    }

    fn type_ptr_to(&self, ty: &'ll Type) -> &'ll Type {
        unimplemented!();
    }

    fn element_type(&self, ty: &'ll Type) -> &'ll Type {
        unimplemented!();
    }

    fn vector_length(&self, ty: &'ll Type) -> usize {
        unimplemented!();
    }

    fn float_width(&self, ty: &'ll Type) -> usize {
        unimplemented!();
    }

    fn int_width(&self, ty: &'ll Type) -> u64 {
        unimplemented!();
    }

    fn val_ty(&self, v: &'ll Value) -> &'ll Type {
        unimplemented!();
    }
}

impl HasTargetSpec for SirCodegenCx<'ll, 'tcx> {
    fn target_spec(&self) -> &Target {
        unimplemented!();
    }
}

impl BackendTypes for SirCodegenCx<'ll, 'tcx> {
    type Value = &'ll Value;
    type Function = &'ll Function;
    type BasicBlock = &'ll BasicBlock;
    type Type = &'ll Type;
    type Funclet = &'ll Funclet;
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
        unimplemented!();
    }

    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::TyLayout {
        unimplemented!();
    }
}


impl LayoutTypeMethods<'tcx> for SirCodegenCx<'ll, 'tcx> {
    fn backend_type(&self, layout: TyLayout<'tcx>) -> &'ll Type {
        unimplemented!();
    }

    fn immediate_backend_type(&self, layout: TyLayout<'tcx>) -> &'ll Type {
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
    ) -> &'ll Type {
        unimplemented!();
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> &'ll Type {
        unimplemented!();
    }

    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> &'ll Type {
        unimplemented!();
    }

    fn reg_backend_type(&self, ty: &Reg) -> &'ll Type {
        unimplemented!();
    }
}

#[must_use]
pub struct SirBuilder<'a, 'tcx, 'll> {
    cx: &'a SirCodegenCx<'ll, 'tcx>,
}

impl LayoutOf for SirBuilder<'a, 'tcx, 'll> {
    type Ty = Ty<'tcx>;
    type TyLayout = TyLayout<'tcx>;

    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
        unimplemented!();
    }

    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::TyLayout {
        unimplemented!();
    }
}

impl ty::layout::HasDataLayout for SirBuilder<'a, 'tcx, 'll> {
    fn data_layout(&self) -> &ty::layout::TargetDataLayout {
        unimplemented!();
    }
}

impl ty::layout::HasTyCtxt<'tcx> for SirBuilder<'a, 'tcx, 'll> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}

impl Deref for SirBuilder<'a, 'tcx, 'll> {
    type Target = SirCodegenCx<'ll, 'tcx>;

    fn deref(&self) -> &Self::Target {
        unimplemented!();
    }
}

impl HasTargetSpec for SirBuilder<'a, 'tcx, 'll> {
    fn target_spec(&self) -> &Target {
        unimplemented!();
    }
}

impl StaticBuilderMethods for SirBuilder<'a, 'tcx, 'll> {
    fn get_static(&mut self, def_id: DefId) -> &'ll Value {
        unimplemented!();
    }
}

impl AsmBuilderMethods<'tcx> for SirBuilder<'a, 'tcx, 'll> {
    fn codegen_inline_asm(
        &mut self,
        ia: &hir::InlineAsm,
        outputs: Vec<PlaceRef<'tcx, &'ll Value>>,
        mut _inputs: Vec<&'ll Value>,
        span: Span,
    ) -> bool {
        unimplemented!();
    }
}

impl AbiBuilderMethods<'tcx> for SirBuilder<'a, 'tcx, 'll> {
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

impl DebugInfoBuilderMethods<'tcx> for SirBuilder<'a, 'tcx, 'll> {
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

    fn set_var_name(&mut self, value: &'ll Value, name: &str) {
        unimplemented!();
    }
}


impl IntrinsicCallMethods<'tcx> for SirBuilder<'a, 'tcx, 'll> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, &'ll Value>],
        llresult: &'ll Value,
        span: Span,
    ) {
        unimplemented!();
    }

    fn abort(&mut self) {
        unimplemented!();
    }

    fn assume(&mut self, val: Self::Value) {
        unimplemented!();
    }

    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value {
        unimplemented!();
    }

    fn sideeffect(&mut self) {
        unimplemented!();
    }

    fn va_start(&mut self, va_list: &'ll Value) -> &'ll Value {
        unimplemented!();
    }

    fn va_end(&mut self, va_list: &'ll Value) -> &'ll Value {
        unimplemented!();
    }
}

impl HasCodegen<'tcx> for SirBuilder<'a, 'tcx, 'll> {
    type CodegenCx = SirCodegenCx<'ll, 'tcx>;
}

impl ty::layout::HasParamEnv<'tcx> for SirBuilder<'a, 'tcx, 'll> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        unimplemented!();
    }
}

impl ArgAbiMethods<'tcx> for SirBuilder<'a, 'tcx, 'll> {
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
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>
    ) {
        unimplemented!();
    }

    fn arg_memory_ty(&self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>) -> &'ll Type {
        unimplemented!();
    }
}

impl BackendTypes for SirBuilder<'a, 'tcx, 'll> {
    type Value = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::Value;
    type Function = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::Function;
    type BasicBlock = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::BasicBlock;
    type Type = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::Type;
    type Funclet = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::Funclet;
    type DIScope = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::DIScope;
    type DISubprogram = <SirCodegenCx<'ll, 'tcx> as BackendTypes>::DISubprogram;
}

impl BuilderMethods<'a, 'tcx> for SirBuilder<'a, 'tcx, 'll> {
    fn new_block<'b>(cx: &'a Self::CodegenCx, llfn: Self::Function, name: &'b str) -> Self {
        unimplemented!();
    }

    fn with_cx(cx: &'a Self::CodegenCx) -> Self {
        Self { cx }
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
        unimplemented!();
    }

    fn add_yk_block_label_at_end(&mut self, di_sp: Self::DIScope, lbl_name: CString) {
        unimplemented!();
    }

    fn position_before(&mut self, instr: Self::Value) {
        unimplemented!();
    }

    fn position_at_end(&mut self, llbb: Self::BasicBlock) {
        unimplemented!();
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
