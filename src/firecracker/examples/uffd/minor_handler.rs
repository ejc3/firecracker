// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Userfaultfd minor-fault handler for snapshot restore integration tests.
//!
//! The handler copies the snapshot into a sealed shmem file, gives that file to Firecracker,
//! and resolves every resulting minor fault with `UFFDIO_CONTINUE`. Firecracker maps the file
//! privately, so clean pages use the shared shmem page cache while guest writes use normal CoW.

mod uffd_utils;

use std::cell::Cell;
use std::error::Error;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};

use memfd::{FileSeal, MemfdOptions};
use uffd_utils::{Runtime, UffdHandler};
use userfaultfd::{Event, FaultKind};
use vmm_sys_util::sock_ctrl_msg::ScmSocket;

const UFFD_MINOR_BACKING_HELLO_V1: &[u8] = b"FCVM_UFFD_MINOR_BACKING";
const UFFD_MINOR_MEMFD_NAME: &str = "firecracker_uffd_minor_test";

trait MinorBackingSocket {
    fn send_with_fd_once(
        &mut self,
        payload: &[u8],
        fd: RawFd,
    ) -> Result<usize, vmm_sys_util::errno::Error>;

    fn write_remaining(&mut self, payload: &[u8]) -> io::Result<()>;
}

impl MinorBackingSocket for UnixStream {
    fn send_with_fd_once(
        &mut self,
        payload: &[u8],
        fd: RawFd,
    ) -> Result<usize, vmm_sys_util::errno::Error> {
        self.send_with_fd(payload, fd)
    }

    fn write_remaining(&mut self, payload: &[u8]) -> io::Result<()> {
        self.write_all(payload)
    }
}

fn create_minor_backing(snapshot_path: &str) -> Result<File, Box<dyn Error>> {
    let mut snapshot = File::open(snapshot_path)?;
    let snapshot_len = snapshot.metadata()?.len();
    if snapshot_len == 0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "snapshot memory file is empty").into(),
        );
    }

    let backing = MemfdOptions::default()
        .allow_sealing(true)
        .create(UFFD_MINOR_MEMFD_NAME)?;
    backing.as_file().set_len(snapshot_len)?;

    let mut destination = backing.as_file().try_clone()?;
    let copied = io::copy(
        &mut Read::by_ref(&mut snapshot).take(snapshot_len),
        &mut destination,
    )?;
    if copied != snapshot_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "snapshot shrank while copying: expected {snapshot_len} bytes, copied {copied}"
            ),
        )
        .into());
    }

    let mut trailing = [0u8; 1];
    if snapshot.read(&mut trailing)? != 0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "snapshot grew while copying").into(),
        );
    }
    destination.flush()?;
    drop(destination);

    let seals = [
        FileSeal::SealShrink,
        FileSeal::SealGrow,
        FileSeal::SealWrite,
        FileSeal::SealSeal,
    ];
    backing.add_seals(&seals)?;

    Ok(backing.into_file())
}

fn send_minor_backing<S: MinorBackingSocket>(stream: &mut S, backing: &File) -> io::Result<()> {
    let sent = stream
        .send_with_fd_once(UFFD_MINOR_BACKING_HELLO_V1, backing.as_raw_fd())
        .map_err(|err| io::Error::from_raw_os_error(err.errno()))?;
    if sent == 0 {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "descriptor-bearing UFFD minor greeting wrote zero bytes",
        ));
    }
    if sent > UFFD_MINOR_BACKING_HELLO_V1.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "descriptor-bearing UFFD minor greeting reported too many bytes",
        ));
    }
    if sent < UFFD_MINOR_BACKING_HELLO_V1.len() {
        stream.write_remaining(&UFFD_MINOR_BACKING_HELLO_V1[sent..])?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args();
    let uffd_sock_path = args.nth(1).ok_or("No socket path given")?;
    let snapshot_path = args.next().ok_or("No memory file given")?;
    if args.next().is_some() {
        return Err("Unexpected extra command-line argument".into());
    }

    let backing = create_minor_backing(&snapshot_path)?;
    let listener = UnixListener::bind(uffd_sock_path)?;
    let (mut stream, _) = listener.accept()?;
    send_minor_backing(&mut stream, &backing)?;

    let mut runtime = Runtime::new(stream, backing);
    runtime.install_panic_hook();
    let reported_success = Cell::new(false);
    runtime.run(|uffd_handler: &mut UffdHandler| {
        while let Some(event) = uffd_handler
            .read_event()
            .expect("Failed to read userfaultfd event")
        {
            match event {
                Event::Pagefault {
                    kind: FaultKind::Minor,
                    addr,
                    ..
                } => {
                    let mapped = uffd_handler
                        .continue_minor(addr)
                        .expect("UFFDIO_CONTINUE failed");
                    if mapped > 0 && !reported_success.replace(true) {
                        eprintln!("MINOR_CONTINUE_OK mapped_bytes={mapped}");
                    }
                }
                Event::Pagefault { kind, addr, .. } => {
                    panic!("Unexpected {kind:?} fault at {addr:?} in UFFD minor mode");
                }
                event => panic!("Unexpected userfaultfd event in UFFD minor mode: {event:?}"),
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use vmm_sys_util::tempfile::TempFile;

    use super::*;

    #[derive(Default)]
    struct RecordingSocket {
        first_write_len: usize,
        payload: Vec<u8>,
        sent_fds: Vec<RawFd>,
        tail_writes: usize,
        tail_error: Option<io::ErrorKind>,
    }

    impl MinorBackingSocket for RecordingSocket {
        fn send_with_fd_once(
            &mut self,
            payload: &[u8],
            fd: RawFd,
        ) -> Result<usize, vmm_sys_util::errno::Error> {
            let recorded = self.first_write_len.min(payload.len());
            self.payload.extend_from_slice(&payload[..recorded]);
            self.sent_fds.push(fd);
            Ok(self.first_write_len)
        }

        fn write_remaining(&mut self, payload: &[u8]) -> io::Result<()> {
            self.tail_writes += 1;
            if let Some(kind) = self.tail_error {
                return Err(io::Error::from(kind));
            }
            self.payload.extend_from_slice(payload);
            Ok(())
        }
    }

    #[test]
    fn test_minor_backing_greeting_completes_partial_send() {
        const EXPECTED_GREETING: &[u8] = b"FCVM_UFFD_MINOR_BACKING";
        let backing = TempFile::new().unwrap().into_file();
        let mut socket = RecordingSocket {
            first_write_len: 5,
            ..RecordingSocket::default()
        };

        send_minor_backing(&mut socket, &backing).unwrap();

        assert_eq!(socket.payload, EXPECTED_GREETING);
        assert_eq!(socket.sent_fds, [backing.as_raw_fd()]);
        assert_eq!(socket.tail_writes, 1);
    }

    #[test]
    fn test_minor_backing_greeting_full_send_has_no_tail() {
        const EXPECTED_GREETING: &[u8] = b"FCVM_UFFD_MINOR_BACKING";
        let backing = TempFile::new().unwrap().into_file();
        let mut socket = RecordingSocket {
            first_write_len: EXPECTED_GREETING.len(),
            ..RecordingSocket::default()
        };

        send_minor_backing(&mut socket, &backing).unwrap();

        assert_eq!(socket.payload, EXPECTED_GREETING);
        assert_eq!(socket.sent_fds, [backing.as_raw_fd()]);
        assert_eq!(socket.tail_writes, 0);
    }

    #[test]
    fn test_minor_backing_greeting_rejects_invalid_send_lengths() {
        const EXPECTED_GREETING: &[u8] = b"FCVM_UFFD_MINOR_BACKING";
        let backing = TempFile::new().unwrap().into_file();

        for invalid_len in [0, EXPECTED_GREETING.len() + 1] {
            let mut socket = RecordingSocket {
                first_write_len: invalid_len,
                ..RecordingSocket::default()
            };
            assert!(send_minor_backing(&mut socket, &backing).is_err());
            assert_eq!(socket.sent_fds, [backing.as_raw_fd()]);
            assert_eq!(socket.tail_writes, 0);
        }
    }

    #[test]
    fn test_minor_backing_greeting_propagates_tail_error() {
        let backing = TempFile::new().unwrap().into_file();
        let mut socket = RecordingSocket {
            first_write_len: 5,
            tail_error: Some(io::ErrorKind::BrokenPipe),
            ..RecordingSocket::default()
        };

        let error = send_minor_backing(&mut socket, &backing).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        assert_eq!(socket.sent_fds, [backing.as_raw_fd()]);
        assert_eq!(socket.tail_writes, 1);
    }
}
