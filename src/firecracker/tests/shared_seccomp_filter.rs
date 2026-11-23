// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::tests_outside_test_module)]

//! The default policy of a build without a per-target one (`unimplemented.json`, which every
//! debug build uses) is the same program for every thread category, so `main` installs it
//! once and each thread inherits it. Before, each thread installed a copy for itself, and
//! the main thread held none until a microVM was built.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Vmm(Child, PathBuf);

impl Drop for Vmm {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
        let _ = std::fs::remove_dir_all(&self.1);
    }
}

/// The value of `key` in a thread's `/proc/<pid>/task/<tid>/status`.
fn status(pid: u32, tid: u32, key: &str) -> u32 {
    let text = std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/status")).unwrap();
    text.lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))
        .unwrap_or_else(|| panic!("no {key} in the status of thread {tid}"))
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn the_default_policy_is_installed_once_and_inherited() {
    let dir = std::env::temp_dir().join(format!("fc-shared-seccomp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let socket = dir.join("api.sock");
    let child = Command::new(env!("CARGO_BIN_EXE_firecracker"))
        .arg("--api-sock")
        .arg(&socket)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let vmm = Vmm(child, dir);
    let pid = vmm.0.id();

    // The API thread binds the socket after it has applied its category's filter.
    let deadline = Instant::now() + Duration::from_secs(30);
    while !socket.exists() {
        assert!(Instant::now() < deadline, "the API socket never appeared");
        std::thread::sleep(Duration::from_millis(10));
    }

    let threads: Vec<u32> = std::fs::read_dir(format!("/proc/{pid}/task"))
        .unwrap()
        .map(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_str()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect();
    assert!(
        threads.len() >= 2,
        "expected the main and the API thread: {threads:?}"
    );
    for tid in threads {
        // Mode 2 is SECCOMP_MODE_FILTER. No microVM is built yet, so a main thread that
        // waited for one to install its own copy would still read 0 here.
        assert_eq!(
            status(pid, tid, "Seccomp"),
            2,
            "thread {tid} runs unfiltered"
        );
        assert_eq!(
            status(pid, tid, "Seccomp_filters"),
            1,
            "thread {tid} holds more than the one inherited filter"
        );
    }
}
