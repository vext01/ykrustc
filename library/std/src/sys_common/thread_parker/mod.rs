cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "linux",
        target_os = "android",
        all(target_arch = "wasm32", target_feature = "atomics"),
    ))] {
        mod futex;
        pub use futex::Parker;
    } else {
        mod generic;
        pub use generic::Parker;
    }
}
