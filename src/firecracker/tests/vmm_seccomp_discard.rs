// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! The release build's VMM seccomp filter admits the syscalls a guest memory discard issues on
//! a shared file mapping (a `Prefilled` restore): `fstatfs` to find whether the file is on
//! hugetlbfs, and the hole punch. Balloon inflation and virtio-mem unplug run on the VMM thread
//! under that filter, which traps anything it does not list.

#![allow(clippy::tests_outside_test_module)]

use std::fs::File;

use vmm::seccomp::{apply_filter, deserialize_binary};

#[test]
fn test_vmm_filter_allows_discarding_shared_memory() {
    let (json, arch) = if cfg!(target_arch = "x86_64") {
        (
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../resources/seccomp/x86_64-unknown-linux-musl.json"
            ),
            "x86_64",
        )
    } else {
        (
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../resources/seccomp/aarch64-unknown-linux-musl.json"
            ),
            "aarch64",
        )
    };
    let dir = std::env::temp_dir().join(format!("fc-vmm-seccomp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bpf = dir.join("filters.bpf");
    seccompiler::compile_bpf(json, arch, bpf.to_str().unwrap(), false, false).unwrap();
    let filters = deserialize_binary(File::open(&bpf).unwrap()).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let vmm_filter = filters.get("vmm").unwrap().clone();

    // SAFETY: memfd_create with a static name.
    let fd = unsafe { libc::memfd_create(c"discard".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
    // SAFETY: sizes the memfd made above.
    assert_eq!(unsafe { libc::ftruncate(fd, 0x2000) }, 0);
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };

    // SAFETY: the child calls only async-signal-safe functions and leaves through _exit.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
    if pid == 0 {
        if apply_filter(&vmm_filter).is_err() {
            // SAFETY: leaves the child without running the parent's destructors.
            unsafe { libc::_exit(2) };
        }
        // SAFETY: fstatfs and fallocate on the memfd made before the fork.
        let ok = unsafe {
            libc::fstatfs(fd, &mut stat) == 0
                && libc::fallocate(
                    fd,
                    libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                    0,
                    0x1000,
                ) == 0
        };
        // SAFETY: as above.
        unsafe { libc::_exit(if ok { 0 } else { 3 }) };
    }
    let mut status = 0;
    // SAFETY: waits for the child forked above.
    assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
    assert!(
        !(libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGSYS),
        "the VMM seccomp filter traps a syscall of a shared-memory discard"
    );
    assert!(
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
        "the child did not exit cleanly: status {status:#x}"
    );
}
