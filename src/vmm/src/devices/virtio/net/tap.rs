// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::fmt::{self, Debug};
use std::fs::File;
use std::io::Error as IoError;
use std::os::raw::*;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};

use vmm_sys_util::ioctl::{ioctl_with_mut_ref, ioctl_with_ref, ioctl_with_val};
use vmm_sys_util::{ioctl_ior_nr, ioctl_iow_nr};

use crate::devices::virtio::net::generated;

// As defined in the Linux UAPI:
// https://elixir.bootlin.com/linux/v4.17/source/include/uapi/linux/if.h#L33
const IFACE_NAME_MAX_LEN: usize = 16;

/// List of errors the tap implementation can throw.
#[rustfmt::skip]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum TapError {
    /// Couldn't open /dev/net/tun: {0}
    OpenTun(IoError),
    /// Invalid interface name
    InvalidIfname,
    /// Error while creating ifreq structure: {0}. Invalid TUN/TAP Backend provided by {1}. Check our documentation on setting up the network devices.
    IfreqExecuteError(IoError, String),
    /// Error while setting the offload flags: {0}
    SetOffloadFlags(IoError),
    /// Error while setting size of the vnet header: {0}
    SetSizeOfVnetHdr(IoError),
    /// The inherited descriptor {0} is not usable: {1}
    InheritedFd(RawFd, IoError),
    /// The inherited descriptor {0} is the device {1} with flags {2:#x}, not the TAP device {3} with flags {4:#x}
    InheritedFdMismatch(RawFd, String, i16, String, i16),
}

const TUNTAP: ::std::os::raw::c_uint = 84;
ioctl_iow_nr!(TUNSETIFF, TUNTAP, 202, ::std::os::raw::c_int);
ioctl_ior_nr!(TUNGETIFF, TUNTAP, 210, ::std::os::raw::c_uint);
// `IFF_TUN` of linux/if_tun.h: a queue of a TUN device, which carries IP packets.
const IFF_TUN: u32 = 1;
ioctl_iow_nr!(TUNSETOFFLOAD, TUNTAP, 208, ::std::os::raw::c_uint);
ioctl_iow_nr!(TUNSETVNETHDRSZ, TUNTAP, 216, ::std::os::raw::c_int);

/// Handle for a network tap interface.
///
/// For now, this simply wraps the file descriptor for the tap device so methods
/// can run ioctls on the interface. The tap interface fd will be closed when
/// Tap goes out of scope, and the kernel will clean up the interface automatically.
#[derive(Debug)]
pub struct Tap {
    tap_file: File,
    pub(crate) if_name: [u8; IFACE_NAME_MAX_LEN],
}

// Returns a byte vector representing the contents of a null terminated C string which
// contains if_name.
fn build_terminated_if_name(if_name: &str) -> Result<[u8; IFACE_NAME_MAX_LEN], TapError> {
    // Convert the string slice to bytes, and shadow the variable,
    // since we no longer need the &str version.
    let if_name = if_name.as_bytes();

    if if_name.len() >= IFACE_NAME_MAX_LEN {
        return Err(TapError::InvalidIfname);
    }

    let mut terminated_if_name = [b'\0'; IFACE_NAME_MAX_LEN];
    terminated_if_name[..if_name.len()].copy_from_slice(if_name);

    Ok(terminated_if_name)
}

/// A descriptor this module does not own yet, for an ioctl on it.
#[derive(Debug)]
struct FdRef(RawFd);

impl AsRawFd for FdRef {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

#[derive(Copy, Clone)]
pub struct IfReqBuilder(generated::ifreq);

impl fmt::Debug for IfReqBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IfReqBuilder {{ .. }}")
    }
}

impl IfReqBuilder {
    pub fn new() -> Self {
        Self(Default::default())
    }

    pub fn if_name(mut self, if_name: &[u8; IFACE_NAME_MAX_LEN]) -> Self {
        // SAFETY: Since we don't call as_mut on the same union field more than once, this block is
        // safe.
        let ifrn_name = unsafe { self.0.ifr_ifrn.ifrn_name.as_mut() };
        ifrn_name.copy_from_slice(if_name.as_ref());

        self
    }

    pub(crate) fn flags(mut self, flags: i16) -> Self {
        self.0.ifr_ifru.ifru_flags = flags;
        self
    }

    pub(crate) fn mtu(mut self, mtu: i32) -> Self {
        self.0.ifr_ifru.ifru_mtu = mtu;
        self
    }

    pub(crate) fn execute<F: AsRawFd + Debug>(
        mut self,
        socket: &F,
        ioctl: u64,
    ) -> std::io::Result<generated::ifreq> {
        // SAFETY: ioctl is safe. Called with a valid socket fd, and we check the return.
        if unsafe { ioctl_with_mut_ref(socket, ioctl, &mut self.0) } < 0 {
            return Err(IoError::last_os_error());
        }

        Ok(self.0)
    }
}

impl Tap {
    /// Create a TUN/TAP device given the interface name.
    /// # Arguments
    ///
    /// * `if_name` - the name of the interface.
    pub fn open_named(if_name: &str) -> Result<Tap, TapError> {
        // SAFETY: Open calls are safe because we give a constant null-terminated
        // string and verify the result.
        let fd = unsafe {
            libc::open(
                c"/dev/net/tun".as_ptr(),
                libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(TapError::OpenTun(IoError::last_os_error()));
        }

        // SAFETY: We just checked that the fd is valid.
        let tuntap = unsafe { File::from_raw_fd(fd) };

        let terminated_if_name = build_terminated_if_name(if_name)?;
        let ifreq = IfReqBuilder::new()
            .if_name(&terminated_if_name)
            .flags(Self::device_flags())
            .execute(&tuntap, TUNSETIFF())
            .map_err(|io_error| TapError::IfreqExecuteError(io_error, if_name.to_owned()))?;

        Ok(Tap {
            tap_file: tuntap,
            // SAFETY: Safe since only the name is accessed, and it's cloned out.
            if_name: unsafe { ifreq.ifr_ifrn.ifrn_name },
        })
    }

    /// The flags a TAP device is opened with, and the flags an inherited one must have.
    fn device_flags() -> i16 {
        i16::try_from(generated::IFF_TAP | generated::IFF_NO_PI | generated::IFF_VNET_HDR).unwrap()
    }

    /// Use an open descriptor of the TAP device `if_name` that this process inherited,
    /// instead of opening the device.
    ///
    /// The descriptor is a queue someone else attached to the device and still holds. This
    /// process attaches nothing and, when it closes its copy, detaches nothing: the device
    /// keeps its queue and its carrier across the processes that use it one after another.
    /// Opening the device by name costs the host an `rtnl_mutex` hold for the attach, one
    /// for the detach at exit with a grace period under it, and two link events.
    ///
    /// The descriptor must be attached to `if_name` with the flags [`Tap::open_named`]
    /// uses, or it is refused and stays open. On success this `Tap` owns it. It is made
    /// non-blocking and close-on-exec; the first is a property of the open file, so the
    /// holder's copy becomes non-blocking with it.
    pub fn from_inherited(fd: RawFd, if_name: &str) -> Result<Tap, TapError> {
        let wanted = build_terminated_if_name(if_name)?;
        // A descriptor that is not open fails TUNGETIFF with EBADF, one that is not a
        // TUN/TAP queue with ENOTTY, and a queue attached to nothing with EBADFD.
        let borrowed = FdRef(fd);
        let ifreq = IfReqBuilder::new()
            .execute(&borrowed, TUNGETIFF())
            .map_err(|error| TapError::InheritedFd(fd, error))?;
        // SAFETY: TUNGETIFF filled both union members read here.
        let (name, flags) = unsafe { (ifreq.ifr_ifrn.ifrn_name, ifreq.ifr_ifru.ifru_flags) };
        // The kernel reports the flags a queue can be opened with and state bits beside
        // them (persistence, ownership); only the first are compared.
        let mask = i16::try_from(
            IFF_TUN
                | generated::IFF_TAP
                | generated::IFF_NO_PI
                | generated::IFF_VNET_HDR
                | generated::IFF_MULTI_QUEUE,
        )
        .unwrap();
        if name != wanted || flags & mask != Self::device_flags() {
            let len = name.iter().position(|b| *b == 0).unwrap_or(name.len());
            return Err(TapError::InheritedFdMismatch(
                fd,
                String::from_utf8_lossy(&name[..len]).into_owned(),
                flags & mask,
                if_name.to_owned(),
                Self::device_flags(),
            ));
        }
        // FIONBIO and F_SETFD, and TUNGETIFF above, are what the VMM thread's seccomp
        // policy allows on this path (`the_vmm_policy_allows_an_inherited_queue`): a net
        // device can be added to a running microVM, under that policy.
        let nonblocking: c_int = 1;
        // SAFETY: FIONBIO reads one int, which outlives the call; F_SETFD takes no pointer.
        let prepared = unsafe {
            libc::ioctl(fd, libc::FIONBIO as _, &nonblocking) == 0
                && libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) == 0
        };
        if !prepared {
            return Err(TapError::InheritedFd(fd, IoError::last_os_error()));
        }
        Ok(Tap {
            // SAFETY: the descriptor is open, is the queue checked above, and the caller
            // hands it over: nothing else in this process uses it.
            tap_file: unsafe { File::from_raw_fd(fd) },
            if_name: name,
        })
    }

    /// Retrieve the interface's name as a str.
    pub fn if_name_as_str(&self) -> &str {
        let len = self
            .if_name
            .iter()
            .position(|x| *x == 0)
            .unwrap_or(IFACE_NAME_MAX_LEN);
        std::str::from_utf8(&self.if_name[..len]).unwrap_or("")
    }

    /// Set the offload flags for the tap interface.
    pub fn set_offload(&self, flags: c_uint) -> Result<(), TapError> {
        // SAFETY: ioctl is safe. Called with a valid tap fd, and we check the return.
        if unsafe { ioctl_with_val(&self.tap_file, TUNSETOFFLOAD(), c_ulong::from(flags)) } < 0 {
            return Err(TapError::SetOffloadFlags(IoError::last_os_error()));
        }

        Ok(())
    }

    /// Set the size of the vnet hdr.
    pub fn set_vnet_hdr_size(&self, size: c_int) -> Result<(), TapError> {
        // SAFETY: ioctl is safe. Called with a valid tap fd, and we check the return.
        if unsafe { ioctl_with_ref(&self.tap_file, TUNSETVNETHDRSZ(), &size) } < 0 {
            return Err(TapError::SetSizeOfVnetHdr(IoError::last_os_error()));
        }

        Ok(())
    }
}

impl AsRawFd for Tap {
    fn as_raw_fd(&self) -> RawFd {
        self.tap_file.as_raw_fd()
    }
}

#[cfg(test)]
pub mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    use std::os::unix::ffi::OsStrExt;

    use super::*;
    use crate::devices::virtio::iovec::IoVecBuffer;
    use crate::devices::virtio::net::generated;
    use crate::devices::virtio::net::test_utils::{TapTrafficSimulator, enable, if_index};
    use crate::devices::virtio::queue::VIRTQ_DESC_F_WRITE;
    use crate::devices::virtio::test_utils::VirtQueue;
    use crate::test_utils::single_region_mem;
    use crate::vstate::memory::{Bytes, GuestAddress};

    // Redefine `IoVecBufferMut` with specific length. Otherwise
    // Rust will not know what to do.
    type IoVecBufferMut = crate::devices::virtio::iovec::IoVecBufferMut<256>;

    // The size of the virtio net header
    const VNET_HDR_SIZE: usize = 10;

    const PAYLOAD_SIZE: usize = 512;

    #[test]
    fn test_tap_name() {
        // Sanity check that the assumed max iface name length is correct.
        assert_eq!(IFACE_NAME_MAX_LEN, unsafe {
            generated::ifreq__bindgen_ty_1::default().ifrn_name.len()
        });

        // Empty name - The tap should be named "tapN" by default, where N is some number
        // assigned by the kernel (e.g. "tap0", "tap1", etc.). We cannot assert a specific
        // number because other tests running in parallel may have already created tap devices.
        let tap = Tap::open_named("").unwrap();
        let name = tap.if_name_as_str();
        assert!(
            name.starts_with("tap")
                && name.len() > 3
                && name[3..].chars().all(|c| c.is_ascii_digit()),
            "Expected tap name matching 'tapN', got '{name}'"
        );

        // Test using '%d' to have the kernel assign an unused name,
        // and that we correctly copy back that generated name
        let tap = Tap::open_named("tap%d").unwrap();
        // '%d' should be replaced with _some_ number, although we don't know what was the next
        // available one. Just assert that '%d' definitely isn't there anymore.
        assert_ne!(b"tap%d", &tap.if_name[..5]);

        // 16 characters - too long.
        let name = "a123456789abcdef";
        match Tap::open_named(name) {
            Err(TapError::InvalidIfname) => (),
            _ => panic!("Expected Error::InvalidIfname"),
        };

        // 15 characters - OK.
        let name = "a123456789abcde";
        let tap = Tap::open_named(name).unwrap();
        assert_eq!(&format!("{}\0", name).as_bytes(), &tap.if_name);
        assert_eq!(name, tap.if_name_as_str());
    }

    #[test]
    fn test_tap_exclusive_open() {
        let _tap1 = Tap::open_named("exclusivetap").unwrap();
        // Opening same tap device a second time should not be permitted.
        Tap::open_named("exclusivetap").unwrap_err();
    }

    /// An inherited queue is used as it is: the device keeps the one queue its holder
    /// attached, and a second user of the holder's descriptor takes over where the first
    /// stopped.
    #[test]
    fn test_inherited_queue_is_the_holders() {
        let holder = Tap::open_named("inherittap").unwrap();
        holder.set_vnet_hdr_size(12).unwrap();
        // What a process that inherits the queue has: a copy of the holder's descriptor,
        // without close-on-exec, on a file that may block.
        let inherited = unsafe { libc::dup(holder.as_raw_fd()) };
        assert!(inherited >= 0);
        let status = unsafe { libc::fcntl(holder.as_raw_fd(), libc::F_GETFL) };
        unsafe {
            libc::fcntl(
                holder.as_raw_fd(),
                libc::F_SETFL,
                status & !libc::O_NONBLOCK,
            )
        };

        let tap = Tap::from_inherited(inherited, "inherittap").unwrap();
        assert_eq!(tap.as_raw_fd(), inherited);
        assert_eq!(tap.if_name_as_str(), "inherittap");
        assert_ne!(
            unsafe { libc::fcntl(inherited, libc::F_GETFD) } & libc::FD_CLOEXEC,
            0
        );
        assert_ne!(
            unsafe { libc::fcntl(inherited, libc::F_GETFL) } & libc::O_NONBLOCK,
            0
        );
        // The same options as on a queue this process attached.
        tap.set_vnet_hdr_size(12).unwrap();
        tap.set_offload(0).unwrap();
        // It is the holder's queue, not a second one: the device is still attached once,
        // so it cannot be opened by name.
        Tap::open_named("inherittap").unwrap_err();
        // The user leaves and the holder still has the device: the next user gets the
        // same queue, and the device was never without one.
        drop(tap);
        let again = unsafe { libc::dup(holder.as_raw_fd()) };
        let tap = Tap::from_inherited(again, "inherittap").unwrap();
        Tap::open_named("inherittap").unwrap_err();
        drop(tap);
    }

    /// A descriptor that is not the queue asked for is refused and left open.
    /// Every syscall `Tap::from_inherited` and the device's set-up make on an inherited
    /// queue is one the VMM thread's seccomp policy allows, on both targets that have a
    /// policy: a net device hotplugged into a running microVM is made under it.
    #[test]
    fn the_vmm_policy_allows_an_inherited_queue() {
        for policy in [
            include_str!("../../../../../../resources/seccomp/aarch64-unknown-linux-musl.json"),
            include_str!("../../../../../../resources/seccomp/x86_64-unknown-linux-musl.json"),
        ] {
            let policy: serde_json::Value = serde_json::from_str(policy).unwrap();
            let rules = policy["vmm"]["filter"].as_array().unwrap();
            let allows = |syscall: &str, args: &[serde_json::Value]| {
                rules.iter().any(|rule| {
                    rule["syscall"] == syscall
                        && rule["args"].as_array().is_some_and(|rule_args| {
                            rule_args.len() == args.len()
                                && rule_args.iter().zip(args).enumerate().all(
                                    |(position, (arg, value))| {
                                        arg["index"] == position + 1
                                            && arg["op"] == "eq"
                                            && arg["val"] == *value
                                    },
                                )
                        })
                })
            };
            let ioctl = |request: serde_json::Value| allows("ioctl", &[request]);
            assert!(ioctl(serde_json::json!(TUNGETIFF())), "TUNGETIFF");
            assert!(ioctl(serde_json::json!(libc::FIONBIO)), "FIONBIO");
            assert!(
                ioctl(serde_json::json!(TUNSETVNETHDRSZ())),
                "TUNSETVNETHDRSZ"
            );
            assert!(ioctl(serde_json::json!(TUNSETOFFLOAD())), "TUNSETOFFLOAD");
            assert!(
                allows(
                    "fcntl",
                    &[
                        serde_json::json!(libc::F_SETFD),
                        serde_json::json!(libc::FD_CLOEXEC)
                    ]
                ),
                "F_SETFD"
            );
        }
    }

    #[test]
    fn test_inherited_queue_must_be_the_device_asked_for() {
        let holder = Tap::open_named("inheritone").unwrap();
        let other = unsafe { libc::dup(holder.as_raw_fd()) };
        match Tap::from_inherited(other, "inherittwo") {
            Err(TapError::InheritedFdMismatch(fd, found, _, wanted, _)) => {
                assert_eq!(
                    (fd, found.as_str(), wanted.as_str()),
                    (other, "inheritone", "inherittwo")
                );
            }
            result => panic!("{result:?}"),
        }
        assert!(
            unsafe { libc::fcntl(other, libc::F_GETFD) } >= 0,
            "the descriptor was closed"
        );
        unsafe { libc::close(other) };

        // Not a TUN/TAP queue.
        let file = File::open("/dev/null").unwrap();
        match Tap::from_inherited(file.as_raw_fd(), "inheritone") {
            Err(TapError::InheritedFd(_, error)) => {
                assert_eq!(error.raw_os_error(), Some(libc::ENOTTY));
            }
            result => panic!("{result:?}"),
        }
        // A queue attached to nothing.
        let unattached = File::options()
            .read(true)
            .write(true)
            .open("/dev/net/tun")
            .unwrap();
        match Tap::from_inherited(unattached.as_raw_fd(), "inheritone") {
            Err(TapError::InheritedFd(_, error)) => {
                assert_eq!(error.raw_os_error(), Some(libc::EBADFD));
            }
            result => panic!("{result:?}"),
        }
        // Not open at all.
        match Tap::from_inherited(1 << 20, "inheritone") {
            Err(TapError::InheritedFd(_, error)) => {
                assert_eq!(error.raw_os_error(), Some(libc::EBADF));
            }
            result => panic!("{result:?}"),
        }
        // A queue of a device with other flags: here without a virtio-net header.
        let plain = File::options()
            .read(true)
            .write(true)
            .open("/dev/net/tun")
            .unwrap();
        IfReqBuilder::new()
            .if_name(&build_terminated_if_name("inheritplain").unwrap())
            .flags(i16::try_from(generated::IFF_TAP | generated::IFF_NO_PI).unwrap())
            .execute(&plain, TUNSETIFF())
            .unwrap();
        match Tap::from_inherited(plain.as_raw_fd(), "inheritplain") {
            Err(TapError::InheritedFdMismatch(..)) => {}
            result => panic!("{result:?}"),
        }
    }

    #[test]
    fn test_set_options() {
        // This line will fail to provide an initialized FD if the test is not run as root.
        let tap = Tap::open_named("").unwrap();
        tap.set_vnet_hdr_size(16).unwrap();
        tap.set_offload(0).unwrap();
    }

    #[test]
    fn test_raw_fd() {
        let tap = Tap::open_named("").unwrap();
        assert_eq!(tap.as_raw_fd(), tap.tap_file.as_raw_fd());
    }

    #[test]
    fn test_writev_to_tap() {
        let tap = Tap::open_named("").unwrap();
        enable(&tap);
        let tap_traffic_simulator = TapTrafficSimulator::new(if_index(&tap));

        let mut fragment1 = vmm_sys_util::rand::rand_bytes(PAYLOAD_SIZE);
        fragment1.as_mut_slice()[..generated::ETH_HLEN as usize]
            .copy_from_slice(&[0; generated::ETH_HLEN as usize]);
        let fragment2 = vmm_sys_util::rand::rand_bytes(PAYLOAD_SIZE);
        let fragment3 = vmm_sys_util::rand::rand_bytes(PAYLOAD_SIZE);

        let scattered = IoVecBuffer::from(vec![
            fragment1.as_slice(),
            fragment2.as_slice(),
            fragment3.as_slice(),
        ]);

        let num_bytes = scattered.writev_to(&tap).unwrap();
        assert_eq!(num_bytes, scattered.len() as usize);

        let mut read_buf = vec![0u8; scattered.len() as usize];
        assert!(tap_traffic_simulator.pop_rx_packet(&mut read_buf));
        assert_eq!(
            &read_buf[..PAYLOAD_SIZE - VNET_HDR_SIZE],
            &fragment1[VNET_HDR_SIZE..]
        );
        assert_eq!(
            &read_buf[PAYLOAD_SIZE - VNET_HDR_SIZE..2 * PAYLOAD_SIZE - VNET_HDR_SIZE],
            fragment2
        );
        assert_eq!(
            &read_buf[2 * PAYLOAD_SIZE - VNET_HDR_SIZE..3 * PAYLOAD_SIZE - VNET_HDR_SIZE],
            fragment3
        );
    }

    #[test]
    fn test_readv_from_tap() {
        let tap = Tap::open_named("").unwrap();
        enable(&tap);
        let tap_traffic_simulator = TapTrafficSimulator::new(if_index(&tap));

        // Two write-only descriptor chains of one descriptor each, in guest memory, with the
        // first one holding exactly the virtio-net header and the first half of the payload.
        let mem = single_region_mem(0x10000);
        let vq = VirtQueue::new(GuestAddress(0), &mem, 16);
        let mut q = vq.create_queue();
        q.config.ready = true;
        let buff1_len = u32::try_from(PAYLOAD_SIZE + VNET_HDR_SIZE).unwrap();
        let buff2_len = u32::try_from(2 * PAYLOAD_SIZE).unwrap();
        let buff1_addr = vq.end().0;
        let buff2_addr = buff1_addr + u64::from(buff1_len);
        vq.dtable[0].set(buff1_addr, buff1_len, VIRTQ_DESC_F_WRITE, 0);
        vq.dtable[1].set(buff2_addr, buff2_len, VIRTQ_DESC_F_WRITE, 0);
        vq.avail.ring[0].set(0);
        vq.avail.ring[1].set(1);
        vq.avail.idx.set(2);
        let mut rx_buffers = IoVecBufferMut::new().unwrap();
        // SAFETY: the two chains do not overlap.
        unsafe {
            rx_buffers
                .append_descriptor_chain(&mem, q.pop().unwrap().unwrap(), 0)
                .unwrap();
            rx_buffers
                .append_descriptor_chain(&mem, q.pop().unwrap().unwrap(), 0)
                .unwrap();
        }

        let packet = vmm_sys_util::rand::rand_alphanumerics(2 * PAYLOAD_SIZE);
        tap_traffic_simulator.push_tx_packet(packet.as_bytes());
        assert_eq!(
            rx_buffers.readv_from(&mem, &tap, usize::MAX).unwrap(),
            2 * PAYLOAD_SIZE + VNET_HDR_SIZE
        );
        let mut buff1 = vec![0; PAYLOAD_SIZE + VNET_HDR_SIZE];
        mem.read_slice(&mut buff1, GuestAddress(buff1_addr))
            .unwrap();
        let mut buff2 = vec![0; 2 * PAYLOAD_SIZE];
        mem.read_slice(&mut buff2, GuestAddress(buff2_addr))
            .unwrap();
        assert_eq!(&buff1[VNET_HDR_SIZE..], &packet.as_bytes()[..PAYLOAD_SIZE]);
        assert_eq!(&buff2[..PAYLOAD_SIZE], &packet.as_bytes()[PAYLOAD_SIZE..]);
        assert_eq!(&buff2[PAYLOAD_SIZE..], &vec![0; PAYLOAD_SIZE])
    }
}
