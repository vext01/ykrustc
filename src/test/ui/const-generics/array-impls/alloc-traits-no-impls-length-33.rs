pub fn no_vec_partial_eq_array<A, B>() -> impl PartialEq<[B; 33]>
//~^ ERROR arrays only have std trait implementations for lengths 0..=32
where
    A: PartialEq<B>,
{
    Vec::<A>::new()
}

pub fn no_vec_partial_eq_ref_array<'a, A, B>() -> impl PartialEq<&'a [B; 33]>
//~^ ERROR arrays only have std trait implementations for lengths 0..=32
where
    A: PartialEq<B>,
{
    Vec::<A>::new()
}

use std::collections::VecDeque;

pub fn no_vecdeque_partial_eq_array<A, B>() -> impl PartialEq<[B; 33]>
//~^ ERROR arrays only have std trait implementations for lengths 0..=32
where
    A: PartialEq<B>,
{
    VecDeque::<A>::new()
}

pub fn no_vecdeque_partial_eq_ref_array<'a, A, B>() -> impl PartialEq<&'a [B; 33]>
//~^ ERROR arrays only have std trait implementations for lengths 0..=32
where
    A: PartialEq<B>,
{
    VecDeque::<A>::new()
}

pub fn no_vecdeque_partial_eq_ref_mut_array<'a, A, B>() -> impl PartialEq<&'a mut [B; 33]>
//~^ ERROR arrays only have std trait implementations for lengths 0..=32
where
    A: PartialEq<B>,
{
    VecDeque::<A>::new()
}

fn main() {}
