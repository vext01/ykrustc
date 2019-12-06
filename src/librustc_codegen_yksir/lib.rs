//! FIXME

// FIXME jsut for now, or I will go mad.
#![allow(unused_imports)]

#![feature(libc)]
#![feature(trusted_len)]

extern crate libc;
extern crate rustc;
extern crate rustc_target;
extern crate rustc_data_structures;

use rustc_codegen_ssa::common::{IntPredicate, RealPredicate, AtomicOrdering, AtomicRmwBinOp, SynchronizationScope};
use rustc_codegen_ssa::MemFlags;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::layout::{self, Align, Size, TyLayout};
use rustc::hir::def_id::DefId;
use rustc::session::config;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::base::to_immediate;
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_target::spec::{HasTargetSpec, Target};
use std::ffi::CString;
use std::ops::Range;
use std::iter::TrustedLen;

#[must_use]
pub struct Builder {}

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

impl BackendTypes for Builder {
    type Value = Value;
    type Function = Function;
    type BasicBlock = BasicBlock;
    type Type = Type;
    type Funclet = Funclet;
    type DIScope = DIScope;
    type DISubprogram = DISubprogram;
}

impl<'a, 'tcx> BuilderMethods<'a, 'tcx> for Builder {
    fn new_block<'b>(cx: &'a Self::CodegenCx, llfn: Self::Function, name: &'b str) -> Self {
        unimplemented!();
    }

    fn with_cx(cx: &'a Self::CodegenCx) -> Self {
        unimplemented!();
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
