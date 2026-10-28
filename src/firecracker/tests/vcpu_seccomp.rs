// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! The release build's vCPU seccomp filter admits the ioctls a vCPU thread issues when it saves
//! its state for a snapshot. A vCPU thread runs under that filter when it handles `SaveState`,
//! and the filter traps anything it does not list, so a missing ioctl kills Firecracker with
//! SIGSYS on every snapshot.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::tests_outside_test_module)]

use std::fs::File;

use kvm_ioctls::{Kvm, KvmNestedStateBuffer};
use vmm::seccomp::{apply_filter, deserialize_binary};

#[test]
fn test_vcpu_filter_allows_saving_nested_state() {
    let dir = std::env::temp_dir().join(format!("fc-vcpu-seccomp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bpf = dir.join("filters.bpf");
    let json = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../resources/seccomp/x86_64-unknown-linux-musl.json"
    );
    seccompiler::compile_bpf(json, "x86_64", bpf.to_str().unwrap(), false, false).unwrap();
    let filters = deserialize_binary(File::open(&bpf).unwrap()).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let vcpu_filter = filters.get("vcpu").unwrap().clone();

    // Everything that allocates happens before the fork: the child only installs the filter,
    // issues the ioctl and exits.
    let kvm = Kvm::new().unwrap();
    let vm = kvm.create_vm().unwrap();
    let vcpu = vm.create_vcpu(0).unwrap();
    let mut buffer = KvmNestedStateBuffer::empty();

    // SAFETY: the child calls only async-signal-safe functions and leaves through _exit.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
    if pid == 0 {
        if apply_filter(&vcpu_filter).is_err() {
            // SAFETY: leaves the child without running the parent's destructors.
            unsafe { libc::_exit(2) };
        }
        // The vCPU belongs to the parent's mm, so KVM answers EIO here: what the test needs is
        // that the filter lets the ioctl reach KVM at all.
        let _ = vcpu.nested_state(&mut buffer);
        // SAFETY: as above.
        unsafe { libc::_exit(0) };
    }
    let mut status = 0;
    // SAFETY: waits for the child forked above.
    assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
    assert!(
        !(libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGSYS),
        "the vCPU seccomp filter traps KVM_GET_NESTED_STATE, which every snapshot issues"
    );
    assert!(
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
        "the child did not exit cleanly: status {status:#x}"
    );
}
