// Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::undocumented_unsafe_blocks,
    // Not everything is used by both binaries
    dead_code
)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd};
use std::os::unix::net::UnixStream;
use std::ptr;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use userfaultfd::{Error, Event, Uffd};
use vmm_sys_util::sock_ctrl_msg::ScmSocket;

const UFFD_MAPPINGS_FRAGMENT_SIZE: usize = 4096;
const UFFD_MAPPINGS_MAX_SIZE: usize = 1024 * 1024;
const UFFD_MAPPINGS_TIMEOUT: Duration = Duration::from_secs(10);

// This is the same with the one used in src/vmm.
/// This describes the mapping between Firecracker base virtual address and offset in the
/// buffer or file backend for a guest memory region. It is used to tell an external
/// process/thread where to populate the guest memory data for this range.
///
/// E.g. Guest memory contents for a region of `size` bytes can be found in the backend
/// at `offset` bytes from the beginning, and should be copied/populated into `base_host_address`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuestRegionUffdMapping {
    /// Base host virtual address where the guest memory contents for this region
    /// should be copied/populated.
    pub base_host_virt_addr: u64,
    /// Region size.
    pub size: usize,
    /// Offset in the backend file/buffer where the region contents are.
    pub offset: u64,
    /// The configured page size for this memory region.
    pub page_size: usize,
}

impl GuestRegionUffdMapping {
    fn contains(&self, fault_page_addr: u64) -> bool {
        fault_page_addr >= self.base_host_virt_addr
            && fault_page_addr < self.base_host_virt_addr + self.size as u64
    }
}

#[derive(Debug)]
pub struct UffdHandler {
    pub mem_regions: Vec<GuestRegionUffdMapping>,
    pub page_size: usize,
    backing_buffer: *const u8,
    uffd: Uffd,
}

impl UffdHandler {
    fn receive_mappings_and_file(
        stream: &UnixStream,
        deadline: Instant,
    ) -> io::Result<(Vec<GuestRegionUffdMapping>, File)> {
        let mut payload = Vec::with_capacity(UFFD_MAPPINGS_FRAGMENT_SIZE);
        let mut received_file = None;

        loop {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::TimedOut, "UFFD mappings handshake timed out")
                })?;
            if remaining.is_zero() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "UFFD mappings handshake timed out",
                ));
            }
            stream.set_read_timeout(Some(remaining))?;
            let mut fragment = [0u8; UFFD_MAPPINGS_FRAGMENT_SIZE];
            let (bytes_read, file) = stream
                .recv_with_fd(&mut fragment)
                .map_err(|err| io::Error::from_raw_os_error(err.errno()))?;

            if let Some(file) = file
                && received_file.replace(file).is_some()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "received more than one userfaultfd descriptor",
                ));
            }

            if bytes_read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Firecracker closed the UFFD handshake before sending complete mappings",
                ));
            }

            let new_len = payload.len().checked_add(bytes_read).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "UFFD mappings length overflow")
            })?;
            if new_len > UFFD_MAPPINGS_MAX_SIZE {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "UFFD mappings exceed the {UFFD_MAPPINGS_MAX_SIZE}-byte protocol limit"
                    ),
                ));
            }
            payload.extend_from_slice(&fragment[..bytes_read]);

            match serde_json::from_slice::<Vec<GuestRegionUffdMapping>>(&payload) {
                Ok(mappings) => {
                    let file = received_file.ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "complete UFFD mappings arrived without a userfaultfd descriptor",
                        )
                    })?;
                    return Ok((mappings, file));
                }
                Err(err) if err.is_eof() => {}
                Err(err) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid UFFD mappings JSON: {err}"),
                    ));
                }
            }
        }
    }

    fn get_mappings_and_file(
        stream: &UnixStream,
    ) -> io::Result<(Vec<GuestRegionUffdMapping>, File)> {
        let previous_timeout = stream.read_timeout()?;
        let deadline = Instant::now() + UFFD_MAPPINGS_TIMEOUT;

        let result = Self::receive_mappings_and_file(stream, deadline);
        let restore_result = stream.set_read_timeout(previous_timeout);
        match (result, restore_result) {
            (Ok(handshake), Ok(())) => Ok(handshake),
            (Err(err), _) => Err(err),
            (Ok(_), Err(err)) => Err(err),
        }
    }

    pub fn from_unix_stream(stream: &UnixStream, backing_buffer: *const u8, size: usize) -> Self {
        let (mappings, file) = Self::get_mappings_and_file(stream)
            .unwrap_or_else(|err| panic!("Cannot receive UFFD mappings from Firecracker: {err}"));
        let memsize: usize = mappings.iter().map(|r| r.size).sum();
        // Page size is the same for all memory regions, so just grab the first one
        let first_mapping = mappings.first().unwrap_or_else(|| {
            panic!(
                "Cannot get the first mapping. Mappings size is {}.",
                mappings.len()
            )
        });
        let page_size = first_mapping.page_size;

        // Make sure memory size matches backing data size.
        assert_eq!(memsize, size);
        assert!(page_size.is_power_of_two());

        let uffd = unsafe { Uffd::from_raw_fd(file.into_raw_fd()) };

        Self {
            mem_regions: mappings,
            page_size,
            backing_buffer,
            uffd,
        }
    }

    pub fn read_event(&mut self) -> Result<Option<Event>, Error> {
        self.uffd.read_event()
    }

    /// Resolve the minor fault containing `addr` with `UFFDIO_CONTINUE`.
    ///
    /// The returned value is the number of bytes this call installed. A zero-byte result means
    /// another faulting thread populated the page first and the kernel returned `EEXIST`.
    pub fn continue_minor(&self, addr: *mut c_void) -> Result<usize, Error> {
        let page = addr as usize & !(self.page_size - 1);
        let page_end = page
            .checked_add(self.page_size)
            .expect("minor-fault page address overflow");
        assert!(
            self.mem_regions.iter().any(|region| {
                let region_start = region.base_host_virt_addr as usize;
                region_start
                    .checked_add(region.size)
                    .is_some_and(|region_end| page >= region_start && page_end <= region_end)
            }),
            "minor fault address {addr:?} is outside the registered guest-memory mappings"
        );

        let mut total_mapped = 0;
        while total_mapped < self.page_size {
            let remaining = self.page_size - total_mapped;
            let current = (page + total_mapped) as *mut c_void;
            match self.uffd.r#continue(current, remaining, true) {
                Ok(mapped) => {
                    let mapped = usize::try_from(mapped)
                        .expect("UFFDIO_CONTINUE mapped-byte count exceeds usize");
                    assert!(
                        mapped > 0 && mapped <= remaining,
                        "UFFDIO_CONTINUE returned invalid mapped-byte count {mapped} for a {remaining}-byte range"
                    );
                    total_mapped += mapped;
                }
                Err(Error::SystemError(errno)) if (errno as i32) == libc::EEXIST => {
                    return Ok(total_mapped);
                }
                Err(err) => return Err(err),
            }
        }

        Ok(total_mapped)
    }

    pub fn unregister_range(&mut self, start: *mut c_void, end: *mut c_void) {
        assert!(
            (start as usize).is_multiple_of(self.page_size)
                && (end as usize).is_multiple_of(self.page_size)
                && end > start
        );
        // SAFETY: start and end are valid and provided by UFFD
        let len = unsafe { end.offset_from_unsigned(start) };
        self.uffd
            .unregister(start, len)
            .expect("range should be valid");
    }

    pub fn serve_pf(&mut self, addr: *mut u8, len: usize) -> bool {
        // Find the start of the page that the current faulting address belongs to.
        let dst = (addr as usize & !(self.page_size - 1)) as *mut libc::c_void;
        let fault_page_addr = dst as u64;

        for region in self.mem_regions.iter() {
            if region.contains(fault_page_addr) {
                return self.populate_from_file(region, fault_page_addr, len);
            }
        }

        panic!(
            "Could not find addr: {:?} within guest region mappings.",
            addr
        );
    }

    fn populate_from_file(&self, region: &GuestRegionUffdMapping, dst: u64, len: usize) -> bool {
        let offset = dst - region.base_host_virt_addr;
        let src = self.backing_buffer as u64 + region.offset + offset;

        unsafe {
            match self.uffd.copy(src as *const _, dst as *mut _, len, true) {
                // Make sure the UFFD copied some bytes.
                Ok(value) => assert!(value > 0),
                // Catch EAGAIN errors, which occur when a `remove` event lands in the UFFD
                // queue while we're processing `pagefault` events.
                // The weird cast is because the `bytes_copied` field is based on the
                // `uffdio_copy->copy` field, which is a signed 64 bit integer, and if something
                // goes wrong, it gets set to a -errno code. However, uffd-rs always casts this
                // value to an unsigned `usize`, which scrambled the errno.
                Err(Error::PartiallyCopied(bytes_copied))
                    if bytes_copied == 0 || bytes_copied == (-libc::EAGAIN) as usize =>
                {
                    return false;
                }
                Err(Error::CopyFailed(errno))
                    if std::io::Error::from(errno).raw_os_error().unwrap() == libc::EEXIST => {}
                Err(e) => {
                    panic!("Uffd copy failed: {e:?}");
                }
            }
        };

        true
    }
}

#[derive(Debug)]
pub struct Runtime {
    stream: UnixStream,
    backing_file: File,
    backing_memory: *mut u8,
    backing_memory_size: usize,
    uffds: HashMap<i32, UffdHandler>,
}

impl Runtime {
    pub fn new(stream: UnixStream, backing_file: File) -> Self {
        let file_meta = backing_file
            .metadata()
            .expect("can not get backing file metadata");
        let backing_memory_size = file_meta.len() as usize;
        // # Safety:
        // File size and fd are valid
        let ret = unsafe {
            libc::mmap(
                ptr::null_mut(),
                backing_memory_size,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_POPULATE,
                backing_file.as_raw_fd(),
                0,
            )
        };
        if ret == libc::MAP_FAILED {
            panic!("mmap on backing file failed");
        }

        Self {
            stream,
            backing_file,
            backing_memory: ret.cast(),
            backing_memory_size,
            uffds: HashMap::default(),
        }
    }

    fn peer_process_credentials(&self) -> libc::ucred {
        let mut creds: libc::ucred = libc::ucred {
            pid: 0,
            gid: 0,
            uid: 0,
        };
        let mut creds_size = size_of::<libc::ucred>() as u32;
        let ret = unsafe {
            libc::getsockopt(
                self.stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&raw mut creds).cast::<c_void>(),
                &raw mut creds_size,
            )
        };
        if ret != 0 {
            panic!("Failed to get peer process credentials");
        }
        creds
    }

    pub fn install_panic_hook(&self) {
        let peer_creds = self.peer_process_credentials();

        let default_panic_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            let r = unsafe { libc::kill(peer_creds.pid, libc::SIGKILL) };

            if r != 0 {
                eprintln!("Failed to kill Firecracker process from panic hook");
            }

            default_panic_hook(panic_info);
        }));
    }

    /// Polls the `UnixStream` and UFFD fds in a loop.
    /// When stream is polled, new uffd is retrieved.
    /// When uffd is polled, page fault is handled by
    /// calling `pf_event_dispatch` with corresponding
    /// uffd object passed in.
    pub fn run(&mut self, pf_event_dispatch: impl Fn(&mut UffdHandler)) {
        let mut pollfds = vec![];

        // Poll the stream for incoming uffds
        pollfds.push(libc::pollfd {
            fd: self.stream.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });

        loop {
            let pollfd_ptr = pollfds.as_mut_ptr();
            let pollfd_size = pollfds.len() as u64;

            // # Safety:
            // Pollfds vector is valid
            let mut nready = unsafe { libc::poll(pollfd_ptr, pollfd_size, -1) };

            if nready == -1 {
                panic!("Could not poll for events!")
            }

            for i in 0..pollfds.len() {
                if nready == 0 {
                    break;
                }
                if pollfds[i].revents & libc::POLLIN != 0 {
                    nready -= 1;
                    if pollfds[i].fd == self.stream.as_raw_fd() {
                        // Handle new uffd from stream
                        let handler = UffdHandler::from_unix_stream(
                            &self.stream,
                            self.backing_memory,
                            self.backing_memory_size,
                        );
                        pollfds.push(libc::pollfd {
                            fd: handler.uffd.as_raw_fd(),
                            events: libc::POLLIN,
                            revents: 0,
                        });
                        self.uffds.insert(handler.uffd.as_raw_fd(), handler);
                    } else {
                        // Handle one of uffd page faults
                        pf_event_dispatch(self.uffds.get_mut(&pollfds[i].fd).unwrap());
                    }
                }
            }
            // If connection is closed, we can skip the socket from being polled.
            pollfds.retain(|pollfd| pollfd.revents & (libc::POLLRDHUP | libc::POLLHUP) == 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::mem::MaybeUninit;
    use std::os::unix::net::UnixListener;

    use vmm_sys_util::tempdir::TempDir;
    use vmm_sys_util::tempfile::TempFile;

    use super::*;

    unsafe impl Send for Runtime {}

    fn wait_until_socket_input_is_drained(socket: &UnixStream) {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let mut queued_bytes: libc::c_int = 0;
            // SAFETY: `socket` owns a valid descriptor and `queued_bytes` points to the
            // writable integer storage required by FIONREAD.
            let result = unsafe {
                libc::ioctl(
                    socket.as_raw_fd(),
                    libc::FIONREAD,
                    std::ptr::addr_of_mut!(queued_bytes),
                )
            };
            assert_eq!(result, 0, "FIONREAD failed: {}", io::Error::last_os_error());
            if queued_bytes == 0 {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "receiver did not consume the descriptor-bearing JSON fragment"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn test_receive_fragmented_mappings_reply() {
        const MAPPINGS_JSON: &[u8] =
            br#"[{"base_host_virt_addr":4096,"size":8192,"offset":0,"page_size":4096}]"#;
        let expected_mapping = GuestRegionUffdMapping {
            base_host_virt_addr: 4096,
            size: 8192,
            offset: 0,
            page_size: 4096,
        };
        let split = MAPPINGS_JSON.len() / 2;
        let (receiver, mut sender) = UnixStream::pair().unwrap();
        let observer = receiver.try_clone().unwrap();
        let previous_timeout = Duration::from_secs(7);
        receiver.set_read_timeout(Some(previous_timeout)).unwrap();
        let descriptor = TempFile::new().unwrap().into_file();
        descriptor.set_len(8192).unwrap();

        sender
            .send_with_fd(&MAPPINGS_JSON[..split], descriptor.as_raw_fd())
            .unwrap();
        let receive_thread = std::thread::spawn(move || {
            let result = UffdHandler::get_mappings_and_file(&receiver);
            (result, receiver.read_timeout().unwrap())
        });

        wait_until_socket_input_is_drained(&observer);
        sender.write_all(&MAPPINGS_JSON[split..]).unwrap();

        let (result, restored_timeout) = receive_thread.join().unwrap();
        let (mappings, received_descriptor) = result.unwrap();
        assert_eq!(mappings.len(), 1);
        assert_eq!(
            mappings[0].base_host_virt_addr,
            expected_mapping.base_host_virt_addr
        );
        assert_eq!(mappings[0].size, expected_mapping.size);
        assert_eq!(mappings[0].offset, expected_mapping.offset);
        assert_eq!(mappings[0].page_size, expected_mapping.page_size);
        assert_eq!(received_descriptor.metadata().unwrap().len(), 8192);
        assert_eq!(restored_timeout, Some(previous_timeout));
    }

    #[test]
    fn test_runtime() {
        let tmp_dir = TempDir::new().unwrap();
        let dummy_socket_path = tmp_dir.as_path().join("dummy_socket");
        let dummy_socket_path_clone = dummy_socket_path.clone();

        let mut uninit_runtime = Box::new(MaybeUninit::<Runtime>::uninit());
        // We will use this pointer to bypass a bunch of Rust Safety
        // for the sake of convenience.
        let runtime_ptr = uninit_runtime.as_ptr().cast::<Runtime>();

        let runtime_thread = std::thread::spawn(move || {
            let tmp_file = TempFile::new().unwrap();
            tmp_file.as_file().set_len(0x1000).unwrap();
            let dummy_mem_path = tmp_file.as_path();

            let file = File::open(dummy_mem_path).expect("Cannot open memfile");
            let listener =
                UnixListener::bind(dummy_socket_path).expect("Cannot bind to socket path");
            let (stream, _) = listener.accept().expect("Cannot listen on UDS socket");
            // Update runtime with actual runtime
            let runtime = uninit_runtime.write(Runtime::new(stream, file));
            runtime.run(|_: &mut UffdHandler| {});
        });

        // wait for runtime thread to initialize itself
        std::thread::sleep(std::time::Duration::from_millis(100));

        let stream =
            UnixStream::connect(dummy_socket_path_clone).expect("Cannot connect to the socket");

        let dummy_memory_region = vec![GuestRegionUffdMapping {
            base_host_virt_addr: 0,
            size: 0x1000,
            offset: 0,
            page_size: 4096,
        }];
        let dummy_memory_region_json = serde_json::to_string(&dummy_memory_region).unwrap();

        let dummy_file_1 = TempFile::new().unwrap();
        let dummy_fd_1 = dummy_file_1.as_file().as_raw_fd();
        stream
            .send_with_fd(dummy_memory_region_json.as_bytes(), dummy_fd_1)
            .unwrap();
        // wait for the runtime thread to process message
        std::thread::sleep(std::time::Duration::from_millis(100));
        unsafe {
            assert_eq!((*runtime_ptr).uffds.len(), 1);
        }

        let dummy_file_2 = TempFile::new().unwrap();
        let dummy_fd_2 = dummy_file_2.as_file().as_raw_fd();
        stream
            .send_with_fd(dummy_memory_region_json.as_bytes(), dummy_fd_2)
            .unwrap();
        // wait for the runtime thread to process message
        std::thread::sleep(std::time::Duration::from_millis(100));
        unsafe {
            assert_eq!((*runtime_ptr).uffds.len(), 2);
        }

        // there is no way to properly stop runtime, so
        // we send a message with an incorrect memory region
        // to cause runtime thread to panic
        let error_memory_region = vec![GuestRegionUffdMapping {
            base_host_virt_addr: 0,
            size: 0,
            offset: 0,
            page_size: 4096,
        }];
        let error_memory_region_json = serde_json::to_string(&error_memory_region).unwrap();
        stream
            .send_with_fd(error_memory_region_json.as_bytes(), dummy_fd_2)
            .unwrap();

        runtime_thread.join().unwrap_err();
    }
}
