// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

// This byte limit is passed to `bitcode` to guard against a potential memory
// allocation DOS caused by binary filters that are too large.
// This limit can be safely determined since the maximum length of a BPF
// filter is 4096 instructions and Firecracker has a finite number of threads.
const DESERIALIZATION_BYTES_LIMIT: usize = 100_000;

/// Each BPF instruction is 8 bytes long and 4 byte aligned.
/// This alignment needs to be satisfied in order for a BPF code to be accepted
/// by the syscalls. Using u64 here is is safe as it has same size and even bigger alignment.
pub type BpfInstruction = u64;

/// Program made up of a sequence of BPF instructions.
pub type BpfProgram = Vec<BpfInstruction>;

/// Reference to program made up of a sequence of BPF instructions.
pub type BpfProgramRef<'a> = &'a [BpfInstruction];

/// Type that associates a thread category to a BPF program.
pub type BpfThreadMap = HashMap<String, Arc<BpfProgram>>;

/// Binary filter deserialization errors.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum DeserializationError {
    /// Failed to read input: {0}
    InputRead(std::io::Error),
    /// Input size {0} exceeds limit of {1} bytes
    SizeLimitExceeded(usize, usize),
    /// Bitcode deserialization failed: {0}
    Bitcode(#[from] bitcode::Error),
}

/// Retrieve empty seccomp filters.
pub fn get_empty_filters() -> BpfThreadMap {
    let mut map = BpfThreadMap::new();
    map.insert("vmm".to_string(), Arc::new(vec![]));
    map.insert("api".to_string(), Arc::new(vec![]));
    map.insert("vcpu".to_string(), Arc::new(vec![]));
    map
}

/// Deserialize binary with bpf filters
pub fn deserialize_binary<R: Read>(reader: R) -> Result<BpfThreadMap, DeserializationError> {
    // Check size limit before reading the full file to prevent DOS attacks
    let mut buf = Vec::new();
    let bytes_read = reader
        .take((DESERIALIZATION_BYTES_LIMIT + 1) as u64)
        .read_to_end(&mut buf)
        .map_err(DeserializationError::InputRead)?;

    if bytes_read > DESERIALIZATION_BYTES_LIMIT {
        return Err(DeserializationError::SizeLimitExceeded(
            bytes_read,
            DESERIALIZATION_BYTES_LIMIT,
        ));
    }

    let result: HashMap<String, _> = bitcode::deserialize(&buf)?;

    Ok(result
        .into_iter()
        .map(|(k, v)| (k.to_lowercase(), Arc::new(v)))
        .collect())
}

/// Filter installation errors.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum InstallationError {
    /// Filter length exceeds the maximum size of {BPF_MAX_LEN:} instructions
    FilterTooLarge,
    /// prctl` syscall failed with error code: {0}
    Prctl(std::io::Error),
    /// Cannot count the threads of this process: {0}
    ThreadCount(std::io::Error),
    /// The process has {0} threads; a filter for the whole process is installed while it has one
    NotSingleThreaded(usize),
}

/// The program every thread category of `filters` holds, when all of them hold the same
/// one and it is not empty.
///
/// A thread's seccomp filter is inherited by the threads it creates, which then share the
/// one kernel program. A filter each thread installs for itself is a program of its own:
/// the kernel compiles it at the install and frees it when the thread exits, and both take
/// the BPF JIT's host-wide `pack_mutex`. A process whose categories all hold the same
/// program therefore needs it once.
pub fn shared_filter(filters: &BpfThreadMap) -> Option<Arc<BpfProgram>> {
    let mut programs = filters.values();
    let first = programs.next()?;
    if first.is_empty() || programs.any(|program| program != first) {
        return None;
    }
    Some(Arc::clone(first))
}

/// Install `bpf_filter` for the whole process: on the calling thread, which must be the
/// only one, so that every thread created afterwards inherits it.
///
/// A process that already has a second thread is refused: that thread would run without
/// the filter.
pub fn apply_filter_to_process(bpf_filter: BpfProgramRef) -> Result<(), InstallationError> {
    let threads = std::fs::read_dir("/proc/self/task")
        .map_err(InstallationError::ThreadCount)?
        .count();
    if threads != 1 {
        return Err(InstallationError::NotSingleThreaded(threads));
    }
    apply_filter(bpf_filter)
}

/// The maximum seccomp-BPF program length allowed by the linux kernel.
pub const BPF_MAX_LEN: usize = 4096;

/// BPF structure definition for filter array.
/// See /usr/include/linux/filter.h .
#[repr(C)]
#[derive(Debug)]
struct SockFprog {
    len: u16,
    filter: *const BpfInstruction,
}

/// Apply bpf filter.
pub fn apply_filter(bpf_filter: BpfProgramRef) -> Result<(), InstallationError> {
    // If the program is empty, don't install the filter.
    if bpf_filter.is_empty() {
        return Ok(());
    }

    // If the program length is greater than the limit allowed by the kernel,
    // fail quickly. Otherwise, `prctl` will give a more cryptic error code.
    if BPF_MAX_LEN < bpf_filter.len() {
        return Err(InstallationError::FilterTooLarge);
    }

    let bpf_filter_len =
        u16::try_from(bpf_filter.len()).map_err(|_| InstallationError::FilterTooLarge)?;

    // SAFETY: Safe because the parameters are valid.
    unsafe {
        {
            let rc = libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
            if rc != 0 {
                return Err(InstallationError::Prctl(std::io::Error::last_os_error()));
            }
        }

        let bpf_prog = SockFprog {
            len: bpf_filter_len,
            filter: bpf_filter.as_ptr(),
        };
        let bpf_prog_ptr = &bpf_prog as *const SockFprog;
        {
            let rc = libc::syscall(
                libc::SYS_seccomp,
                libc::SECCOMP_SET_MODE_FILTER,
                0,
                bpf_prog_ptr,
            );
            if rc != 0 {
                return Err(InstallationError::Prctl(std::io::Error::last_os_error()));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    use std::collections::HashMap;
    use std::sync::Arc;
    use std::thread;

    use super::*;

    #[test]
    fn test_deserialize_binary() {
        // Malformed bitcode binary.
        let data = "adassafvc".to_string();
        deserialize_binary(data.as_bytes()).unwrap_err();

        // Test that the binary deserialization is correct, and that the thread keys
        // have been lowercased.
        let bpf_prog = vec![0; 2];
        let mut filter_map: HashMap<String, BpfProgram> = HashMap::new();
        filter_map.insert("VcpU".to_string(), bpf_prog.clone());
        let bytes = bitcode::serialize(&filter_map).unwrap();

        let mut expected_res = BpfThreadMap::new();
        expected_res.insert("vcpu".to_string(), Arc::new(bpf_prog));
        assert_eq!(deserialize_binary(&bytes[..]).unwrap(), expected_res);

        // Test filter too large - create data that will exceed the deserialization limit
        // Create a large buffer that when serialized will exceed DESERIALIZATION_BYTES_LIMIT
        let large_data = vec![0u8; DESERIALIZATION_BYTES_LIMIT + 1000];
        deserialize_binary(&large_data[..]).unwrap_err();
    }

    fn thread_map(vmm: &[u64], api: &[u64], vcpu: &[u64]) -> BpfThreadMap {
        let mut map = BpfThreadMap::new();
        map.insert("vmm".to_string(), Arc::new(vmm.to_vec()));
        map.insert("api".to_string(), Arc::new(api.to_vec()));
        map.insert("vcpu".to_string(), Arc::new(vcpu.to_vec()));
        map
    }

    #[test]
    fn test_shared_filter() {
        let program = [1u64, 2, 3];
        assert_eq!(
            shared_filter(&thread_map(&program, &program, &program)).as_deref(),
            Some(&program.to_vec())
        );
        // Categories that differ keep a program each.
        assert_eq!(
            shared_filter(&thread_map(&program, &program, &[1, 2])),
            None
        );
        assert_eq!(
            shared_filter(&thread_map(&[9, 2, 3], &program, &program)),
            None
        );
        // --no-seccomp: nothing to install.
        assert_eq!(shared_filter(&get_empty_filters()), None);
        assert_eq!(shared_filter(&BpfThreadMap::new()), None);
    }

    /// The `Seccomp_filters` line of the calling thread's status: the number of filters
    /// attached to it, inherited ones included.
    fn seccomp_filters_of_this_thread() -> usize {
        std::fs::read_to_string("/proc/thread-self/status")
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix("Seccomp_filters:"))
            .expect("the kernel reports Seccomp_filters")
            .trim()
            .parse()
            .unwrap()
    }

    /// A thread created after the install runs under the installer's filter and adds none
    /// of its own when its category's program is empty, which is what every category's
    /// becomes once the shared one is installed.
    #[test]
    fn test_threads_inherit_the_installed_filter() {
        // BPF_RET | BPF_K, SECCOMP_RET_ALLOW: a one-instruction program that allows everything.
        let allow_all: BpfProgram = vec![0x7fff_0000_0000_0006];
        thread::spawn(move || {
            assert_eq!(seccomp_filters_of_this_thread(), 0);
            apply_filter(&allow_all).unwrap();
            assert_eq!(seccomp_filters_of_this_thread(), 1);
            thread::spawn(|| {
                assert_eq!(unsafe { libc::prctl(libc::PR_GET_SECCOMP) }, 2);
                assert_eq!(seccomp_filters_of_this_thread(), 1);
                apply_filter(&get_empty_filters()["vcpu"]).unwrap();
                assert_eq!(seccomp_filters_of_this_thread(), 1);
            })
            .join()
            .unwrap();
        })
        .join()
        .unwrap();
    }

    /// The test harness runs this on a thread beside its main one, so the process has
    /// more than one thread and the install is refused before anything is installed.
    #[test]
    fn test_apply_filter_to_process_refuses_a_second_thread() {
        thread::spawn(|| {
            let allow_all: BpfProgram = vec![0x7fff_0000_0000_0006];
            assert!(matches!(
                apply_filter_to_process(&allow_all).unwrap_err(),
                InstallationError::NotSingleThreaded(threads) if threads > 1
            ));
            assert_eq!(unsafe { libc::prctl(libc::PR_GET_SECCOMP) }, 0);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_filter_apply() {
        // Test filter too large.
        thread::spawn(|| {
            let filter: BpfProgram = vec![0; 5000];

            // Apply seccomp filter.
            assert!(matches!(
                apply_filter(&filter).unwrap_err(),
                InstallationError::FilterTooLarge
            ));
        })
        .join()
        .unwrap();

        // Test empty filter.
        thread::spawn(|| {
            let filter: BpfProgram = vec![];

            assert_eq!(filter.len(), 0);

            let seccomp_level = unsafe { libc::prctl(libc::PR_GET_SECCOMP) };
            assert_eq!(seccomp_level, 0);

            apply_filter(&filter).unwrap();

            // test that seccomp level remains 0 on failure.
            let seccomp_level = unsafe { libc::prctl(libc::PR_GET_SECCOMP) };
            assert_eq!(seccomp_level, 0);
        })
        .join()
        .unwrap();

        // Test invalid BPF code.
        thread::spawn(|| {
            let filter = vec![0xFF; 1];

            let seccomp_level = unsafe { libc::prctl(libc::PR_GET_SECCOMP) };
            assert_eq!(seccomp_level, 0);

            assert!(matches!(
                apply_filter(&filter).unwrap_err(),
                InstallationError::Prctl(_)
            ));

            // test that seccomp level remains 0 on failure.
            let seccomp_level = unsafe { libc::prctl(libc::PR_GET_SECCOMP) };
            assert_eq!(seccomp_level, 0);
        })
        .join()
        .unwrap();
    }
}
