// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use vmm::seccomp::{BpfThreadMap, DeserializationError, deserialize_binary, get_empty_filters};

const THREAD_CATEGORIES: [&str; 3] = ["vmm", "api", "vcpu"];

/// Error retrieving seccomp filters.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum FilterError {
    /// Filter deserialization failed: {0}
    Deserialization(DeserializationError),
    /// Invalid thread categories: {0}
    ThreadCategories(String),
    /// Missing thread category: {0}
    MissingThreadCategory(String),
    /// Filter file open error: {0}
    FileOpen(std::io::Error),
}

/// Seccomp filter configuration.
#[derive(Debug)]
pub enum SeccompConfig {
    /// Seccomp filtering disabled.
    None,
    /// Default, advanced filters.
    Advanced,
    /// Custom, user-provided filters.
    Custom(File),
}

impl SeccompConfig {
    /// Given the relevant command line args, return the appropriate config type.
    pub fn from_args<T: AsRef<Path> + Debug>(
        no_seccomp: bool,
        seccomp_filter: Option<T>,
    ) -> Result<Self, FilterError> {
        if no_seccomp {
            Ok(SeccompConfig::None)
        } else {
            match seccomp_filter {
                Some(path) => Ok(SeccompConfig::Custom(
                    File::open(path).map_err(FilterError::FileOpen)?,
                )),
                None => Ok(SeccompConfig::Advanced),
            }
        }
    }
}

/// Retrieve the appropriate filters, based on the SeccompConfig.
pub fn get_filters(config: SeccompConfig) -> Result<BpfThreadMap, FilterError> {
    match config {
        SeccompConfig::None => Ok(get_empty_filters()),
        SeccompConfig::Advanced => get_default_filters(),
        SeccompConfig::Custom(reader) => get_custom_filters(reader),
    }
}

/// Retrieve the default filters containing the syscall rules required by `Firecracker`
/// to function. The binary file is generated via the `build.rs` script of this crate.
fn get_default_filters() -> Result<BpfThreadMap, FilterError> {
    // Retrieve, at compile-time, the serialized binary filter generated with seccompiler.
    let bytes: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seccomp_filter.bpf"));
    let map = deserialize_binary(bytes).map_err(FilterError::Deserialization)?;
    filter_thread_categories(map)
}

/// Retrieve custom seccomp filters.
fn get_custom_filters<R: Read + Debug>(reader: R) -> Result<BpfThreadMap, FilterError> {
    let map = deserialize_binary(BufReader::new(reader)).map_err(FilterError::Deserialization)?;
    filter_thread_categories(map)
}

/// Return an error if the BpfThreadMap contains invalid thread categories.
fn filter_thread_categories(map: BpfThreadMap) -> Result<BpfThreadMap, FilterError> {
    let (filters, invalid_filters): (BpfThreadMap, BpfThreadMap) = map
        .into_iter()
        .partition(|(k, _)| THREAD_CATEGORIES.contains(&k.as_str()));
    if !invalid_filters.is_empty() {
        // build the error message
        let mut thread_categories_string =
            invalid_filters
                .keys()
                .fold("".to_string(), |mut acc, elem| {
                    acc.push_str(elem);
                    acc.push(',');
                    acc
                });
        thread_categories_string.pop();
        return Err(FilterError::ThreadCategories(thread_categories_string));
    }

    for &category in THREAD_CATEGORIES.iter() {
        let category_string = category.to_string();
        if !filters.contains_key(&category_string) {
            return Err(FilterError::MissingThreadCategory(category_string));
        }
    }

    Ok(filters)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vmm::seccomp::BpfThreadMap;
    use vmm_sys_util::tempfile::TempFile;

    use super::*;

    #[test]
    fn test_get_filters() {
        let mut filters = get_empty_filters();
        assert_eq!(filters.len(), 3);
        assert!(filters.remove("vmm").is_some());
        assert!(filters.remove("api").is_some());
        assert!(filters.remove("vcpu").is_some());

        let mut filters = get_empty_filters();
        assert_eq!(filters.len(), 3);
        assert_eq!(filters.remove("vmm").unwrap().len(), 0);
        assert_eq!(filters.remove("api").unwrap().len(), 0);
        assert_eq!(filters.remove("vcpu").unwrap().len(), 0);

        let file = TempFile::new().unwrap().into_file();

        get_filters(SeccompConfig::Custom(file)).unwrap_err();
    }

    #[test]
    fn test_filter_thread_categories() {
        // correct categories
        let mut map = BpfThreadMap::new();
        map.insert("vcpu".to_string(), Arc::new(vec![]));
        map.insert("vmm".to_string(), Arc::new(vec![]));
        map.insert("api".to_string(), Arc::new(vec![]));

        assert_eq!(filter_thread_categories(map).unwrap().len(), 3);

        // invalid categories
        let mut map = BpfThreadMap::new();
        map.insert("vcpu".to_string(), Arc::new(vec![]));
        map.insert("vmm".to_string(), Arc::new(vec![]));
        map.insert("thread1".to_string(), Arc::new(vec![]));
        map.insert("thread2".to_string(), Arc::new(vec![]));

        match filter_thread_categories(map).unwrap_err() {
            FilterError::ThreadCategories(err) => {
                assert!(err == "thread2,thread1" || err == "thread1,thread2")
            }
            _ => panic!("Expected ThreadCategories error."),
        }

        // missing category
        let mut map = BpfThreadMap::new();
        map.insert("vcpu".to_string(), Arc::new(vec![]));
        map.insert("vmm".to_string(), Arc::new(vec![]));

        match filter_thread_categories(map).unwrap_err() {
            FilterError::MissingThreadCategory(name) => assert_eq!(name, "api"),
            _ => panic!("Expected MissingThreadCategory error."),
        }
    }

    #[test]
    fn test_seccomp_config() {
        assert!(matches!(
            SeccompConfig::from_args(true, Option::<&str>::None),
            Ok(SeccompConfig::None)
        ));

        assert!(matches!(
            SeccompConfig::from_args(false, Some("/dev/null")),
            Ok(SeccompConfig::Custom(_))
        ));

        assert!(matches!(
            SeccompConfig::from_args(false, Some("invalid_path")),
            Err(FilterError::FileOpen(_))
        ));

        // test the default case, no parametes -> default advanced.
        assert!(matches!(
            SeccompConfig::from_args(false, Option::<&str>::None),
            Ok(SeccompConfig::Advanced)
        ));
    }

    /// The vmm filters shipped for musl let `mbind` through only as the call that ends the NUMA
    /// window, `mbind(addr, len, MPOL_DEFAULT, NULL, 0, 0)`: a thread under the filter cannot
    /// bind memory to another policy or have pages moved.
    #[test]
    fn the_vmm_filter_allows_mbind_only_back_to_the_default_policy() {
        let json_path = |arch: &str| {
            format!(
                "{}/../../resources/seccomp/{arch}-unknown-linux-musl.json",
                env!("CARGO_MANIFEST_DIR")
            )
        };
        // Both architectures' rules say so.
        for arch in ["aarch64", "x86_64"] {
            let json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(json_path(arch)).unwrap()).unwrap();
            for (thread, filter) in json.as_object().unwrap() {
                let rules: Vec<_> = filter["filter"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|rule| rule["syscall"] == "mbind")
                    .collect();
                if thread != "vmm" {
                    assert!(rules.is_empty(), "{arch} {thread} allows mbind");
                    continue;
                }
                assert_eq!(rules.len(), 1, "{arch} vmm");
                let conditions: Vec<_> = rules[0]["args"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{arch}: mbind is allowed with any arguments"))
                    .iter()
                    .map(|arg| {
                        (
                            arg["index"].as_u64().unwrap(),
                            arg["type"].as_str().unwrap(),
                            arg["op"].as_str().unwrap(),
                            arg["val"].as_u64().unwrap(),
                        )
                    })
                    .collect();
                assert_eq!(
                    conditions,
                    [(2, "dword", "eq", 0), (5, "dword", "eq", 0)],
                    "{arch}"
                );
            }
        }

        // And this architecture's compiled filter enforces it.
        let arch = std::env::consts::ARCH;
        let compiled = TempFile::new().unwrap();
        seccompiler::compile_bpf(
            &json_path(arch),
            arch,
            compiled.as_path().to_str().unwrap(),
            false,
            false,
        )
        .unwrap();
        let filters =
            vmm::seccomp::deserialize_binary(std::fs::File::open(compiled.as_path()).unwrap())
                .unwrap();
        let filter = Arc::clone(&filters["vmm"]);
        let len = 1usize << 16;
        // SAFETY: a new anonymous mapping, valid arguments.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(addr, libc::MAP_FAILED);
        // Not in the libc crate: <linux/mempolicy.h>.
        const MPOL_MF_MOVE: libc::c_uint = 1 << 1;
        for (mode, flags, allowed) in [
            (libc::MPOL_DEFAULT, 0, true),
            (libc::MPOL_LOCAL, 0, false),
            (libc::MPOL_DEFAULT, MPOL_MF_MOVE, false),
        ] {
            // SAFETY: the child makes system calls only: it installs the filter, calls mbind
            // and leaves through exit_group, which the filter allows.
            let pid = unsafe { libc::fork() };
            assert!(pid >= 0);
            if pid == 0 {
                if vmm::seccomp::apply_filter(&filter).is_err() {
                    // SAFETY: exit_group takes no pointers.
                    unsafe { libc::_exit(3) };
                }
                // SAFETY: mbind on this process's own mapping with no node mask.
                unsafe {
                    libc::syscall(
                        libc::SYS_mbind,
                        addr,
                        len,
                        mode,
                        std::ptr::null::<libc::c_ulong>(),
                        0 as libc::c_ulong,
                        flags,
                    );
                    libc::_exit(0);
                }
            }
            let mut status = 0;
            // SAFETY: `pid` is this process's child and `status` is writable.
            assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
            if allowed {
                assert!(
                    libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
                    "mbind({mode}, flags {flags}) was not let through: wait status {status:#x}"
                );
            } else {
                assert!(
                    libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGSYS,
                    "mbind({mode}, flags {flags}) was let through: wait status {status:#x}"
                );
            }
        }
    }
}
