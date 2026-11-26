// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Barrier, Mutex, MutexGuard};

#[cfg(target_arch = "x86_64")]
use kvm_bindings::KVM_IRQCHIP_IOAPIC;
use kvm_bindings::{
    KVM_IRQ_ROUTING_IRQCHIP, KVM_IRQ_ROUTING_MSI, KVM_MSI_VALID_DEVID, KvmIrqRouting,
    kvm_irq_routing_entry, kvm_userspace_memory_region,
};
use kvm_ioctls::VmFd;
use serde::{Deserialize, Serialize};
use userfaultfd::Uffd;
use vm_memory::{GuestMemoryBackend, WriteVolatile};
use vmm_sys_util::errno;
use vmm_sys_util::eventfd::EventFd;
use vmm_sys_util::terminal::Terminal;

use crate::arch::{GSI_MSI_END, host_page_size};
pub use crate::arch::{KvmVm, KvmVmError, VmState};
use crate::logger::{debug, info};
use crate::nitro_enclave::enclave_vm::EnclaveVm;
use crate::pci::PciSBDF;
use crate::pci::msix::MsixTableEntry;
use crate::persist::CreateSnapshotError;
use crate::vmm_config::snapshot::SnapshotType;
use crate::vstate::bus::Bus;
use crate::vstate::interrupts::{InterruptError, MsixVector, MsixVectorGroup};
use crate::vstate::kvm::Kvm;
use crate::vstate::memory::{
    AbsentPages, CleanGaps, GuestMemoryExtension, GuestMemoryMmap, GuestMemoryRegion,
    GuestMemoryState, GuestRegionMmap, GuestRegionMmapExt, MemoryError, ZeroPages,
};
use crate::vstate::resources::ResourceAllocator;
use crate::vstate::vcpu::{StartThreadedError, VcpuError, VcpuHandle};
use crate::{DirtyBitmap, Vcpu, mem_size_mib};

/// Maximum number of vCPUs supported by a KVM microVM.
///
/// The value is far below what KVM allows (1024 on x86_64, 512 on aarch64). Raising it hits, on
/// x86_64: Intel's CPUID leaf 0x4 EAX[31:26] encodes the core ID in 6 bits, so 64 cores, which
/// without SMT is 64 vCPUs; CPUID leaf 0x1 EBX[23:16] rejects counts above 128; and the MP table
/// and ACPI MADT both identify a CPU by an 8-bit APIC ID. On aarch64 none of these exist, and the
/// bound is the kernel VGIC, at 512 vCPUs.
pub const MAX_SUPPORTED_KVM_VCPUS: u16 = 64;

/// The KVM vCPU plumbing narrows the count to a `u8`, so the limit must fit in
/// one for that narrowing to be infallible.
const _: () = assert!(MAX_SUPPORTED_KVM_VCPUS <= u8::MAX as u16);

/// Error type for [`KvmVm::start_vcpus`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum StartVcpusError {
    /// Failed to set terminal mode: {0}
    SetTerminalMode(#[from] vmm_sys_util::errno::Error),
    /// Vcpu handle error: {0}
    VcpuHandle(#[from] StartThreadedError),
}

#[derive(Debug, Serialize, Deserialize)]
/// A struct representing an interrupt line used by some device of the microVM
pub struct RoutingEntry {
    entry: kvm_irq_routing_entry,
    masked: bool,
}

/// Architecture independent parts of a VM.
#[derive(Debug)]
pub struct VmCommon {
    /// The KVM file descriptor used to access this KvmVm.
    pub fd: VmFd,
    max_memslots: u32,
    /// The guest memory of this KvmVm.
    pub guest_memory: GuestMemoryMmap,
    next_kvm_slot: AtomicU32,
    /// Interrupts used by KvmVm's devices
    pub interrupts: Mutex<HashMap<u32, RoutingEntry>>,
    /// Allocator for VM resources
    pub resource_allocator: Mutex<ResourceAllocator>,
    /// MMIO bus
    pub mmio_bus: Arc<Bus>,
    /// The global KVM state (fd + capabilities).
    pub kvm: Kvm,
    /// Userfaultfd kept open for snapshot restore.
    pub uffd: Option<Uffd>,
    /// Handles to vCPU threads.
    pub vcpus_handles: Mutex<Vec<VcpuHandle>>,
    /// Event fd written to by vCPUs on exit.
    pub vcpus_exit_evt: EventFd,
    /// Whether KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2 is on for this VM (aarch64 only). With it,
    /// KVM_GET_DIRTY_LOG copies a slot's log and leaves the log and the write protection as
    /// they are, and [`KvmVm::take_dirty_bitmap`] empties the logs slot by slot afterwards.
    manual_dirty_log_protect: bool,
    /// Test-only countdown that forces the Nth-next `set_user_memory_region` call to fail, used
    /// to exercise partial-failure handling. 0 means never fail.
    #[cfg(test)]
    fail_set_user_memory_region_in: AtomicU32,
}

/// Errors associated with the wrappers over KVM ioctls.
/// Needs `rustfmt::skip` to make multiline comments work
#[rustfmt::skip]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VmError {
    /// Cannot set the memory regions: {0}
    SetUserMemoryRegion(kvm_ioctls::Error),
    /// Failed to create VM: {0}
    CreateVm(kvm_ioctls::Error),
    /// Failed to get KVM's dirty log: {0}
    GetDirtyLog(kvm_ioctls::Error),
    /// {0}
    Arch(#[from] KvmVmError),
    /// Error during eventfd operations: {0}
    EventFd(std::io::Error),
    /// Failed to create vcpu: {0}
    CreateVcpu(VcpuError),
    /// The number of configured slots is bigger than the maximum reported by KVM: {0}
    NotEnoughMemorySlots(u32),
    /// Failed to add a memory region: {0}
    InsertRegion(#[from] vm_memory::GuestRegionCollectionError),
    /// Error calling mincore: {0}
    Mincore(vmm_sys_util::errno::Error),
    /// Error reading /proc/self/pagemap: {0}
    Pagemap(std::io::Error),
    /// ResourceAllocator error: {0}
    ResourceAllocator(#[from] vm_allocator::Error),
    /// MemoryError error: {0}
    MemoryError(#[from] MemoryError),
}

/// VM abstraction: either a KVM-based VM or a Nitro Enclave.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Vm {
    /// KVM-backed virtual machine.
    Kvm(Arc<KvmVm>),
    /// Nitro Enclave-backed virtual machine.
    Enclave(EnclaveVm),
}

impl Vm {
    /// Returns the name of the VM type.
    pub fn type_name(&self) -> &'static str {
        match self {
            Vm::Kvm(_) => "Kvm",
            Vm::Enclave(_) => "Enclave",
        }
    }

    /// Returns a reference to the inner KVM VM, or `None` if this is not a KVM VM.
    pub fn as_kvm(&self) -> Option<&Arc<KvmVm>> {
        match self {
            Vm::Kvm(v) => Some(v),
            Vm::Enclave(_) => None,
        }
    }

    /// Returns a reference to the inner EnclaveVm, or `None` if this is not an Enclave VM.
    pub fn as_enclave(&self) -> Option<&EnclaveVm> {
        match self {
            Vm::Kvm(_) => None,
            Vm::Enclave(e) => Some(e),
        }
    }
}

/// Contains KvmVm functions that are usable across CPU architectures
impl KvmVm {
    /// Create a KVM VM
    pub fn create_common(kvm: Kvm) -> Result<VmCommon, VmError> {
        // It is known that KVM_CREATE_VM occasionally fails with EINTR on heavily loaded machines
        // with many VMs.
        //
        // The behavior itself that KVM_CREATE_VM can return EINTR is intentional. This is because
        // the KVM_CREATE_VM path includes mm_take_all_locks() that is CPU intensive and all CPU
        // intensive syscalls should check for pending signals and return EINTR immediately to allow
        // userland to remain interactive.
        // https://lists.nongnu.org/archive/html/qemu-devel/2014-01/msg01740.html
        //
        // However, it is empirically confirmed that, even though there is no pending signal,
        // KVM_CREATE_VM returns EINTR.
        // https://lore.kernel.org/qemu-devel/8735e0s1zw.wl-maz@kernel.org/
        //
        // To mitigate it, QEMU does an infinite retry on EINTR that greatly improves reliabiliy:
        // - https://github.com/qemu/qemu/commit/94ccff133820552a859c0fb95e33a539e0b90a75
        // - https://github.com/qemu/qemu/commit/bbde13cd14ad4eec18529ce0bf5876058464e124
        //
        // Similarly, we do retries up to 5 times. Although Firecracker clients are also able to
        // retry, they have to start Firecracker from scratch. Doing retries in Firecracker makes
        // recovery faster and improves reliability.
        const MAX_ATTEMPTS: u32 = 5;
        let mut attempt = 1;
        let fd = loop {
            match kvm.fd.create_vm() {
                Ok(fd) => break fd,
                Err(e) if e.errno() == libc::EINTR && attempt < MAX_ATTEMPTS => {
                    info!("Attempt #{attempt} of KVM_CREATE_VM returned EINTR");
                    // Exponential backoff (1us, 2us, 4us, and 8us => 15us in total)
                    std::thread::sleep(std::time::Duration::from_micros(2u64.pow(attempt - 1)));
                }
                Err(e) => return Err(VmError::CreateVm(e)),
            }

            attempt += 1;
        };

        let vcpus_exit_evt = EventFd::new(libc::EFD_NONBLOCK).map_err(VmError::EventFd)?;
        let manual_dirty_log_protect = Self::enable_manual_dirty_log_protect(&fd);

        Ok(VmCommon {
            fd,
            max_memslots: kvm.max_nr_memslots(),
            guest_memory: GuestMemoryMmap::default(),
            next_kvm_slot: AtomicU32::new(0),
            interrupts: Mutex::new(HashMap::with_capacity(GSI_MSI_END as usize + 1)),
            resource_allocator: Mutex::new(ResourceAllocator::new()),
            mmio_bus: Arc::new(Bus::new()),
            kvm,
            uffd: None,
            vcpus_handles: Mutex::new(Vec::new()),
            vcpus_exit_evt,
            manual_dirty_log_protect,
            #[cfg(test)]
            fail_set_user_memory_region_in: AtomicU32::new(0),
        })
    }

    /// Turns KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2 on for a new VM on aarch64 and says whether it
    /// is on. x86_64 keeps KVM's default, where KVM_GET_DIRTY_LOG empties the log itself.
    ///
    /// On arm64, emptying a log through KVM_GET_DIRTY_LOG write-protects each 64 pages that hold
    /// a dirty one in a call of its own, and each such call also write-protects every nested
    /// stage-2 MMU of the VM over its whole range (`kvm_arch_mmu_enable_log_dirty_pt_masked`
    /// calls `kvm_nested_s2_wp`). For a guest with a virtual EL2 that took most of a snapshot:
    /// 6.1 s of a Full snapshot of a 1 GiB guest that had written 345 MiB, against 7 ms for
    /// emptying the logs a slot at a time (Graviton3, Linux 7.0).
    #[cfg(target_arch = "aarch64")]
    fn enable_manual_dirty_log_protect(fd: &VmFd) -> bool {
        use kvm_bindings::{
            KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2, KVM_DIRTY_LOG_MANUAL_PROTECT_ENABLE, kvm_enable_cap,
        };
        let manual_protect = kvm_enable_cap {
            cap: KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2,
            args: [u64::from(KVM_DIRTY_LOG_MANUAL_PROTECT_ENABLE), 0, 0, 0],
            ..Default::default()
        };
        // _IOW(KVMIO = 0xAE, 0xa3, struct kvm_enable_cap), 104 bytes. kvm-ioctls has
        // `VmFd::enable_cap` on x86_64 only.
        const KVM_ENABLE_CAP: u32 = 0x4068_aea3;
        const _: () = assert!(std::mem::size_of::<kvm_enable_cap>() == 0x68);
        // SAFETY: `fd` is a valid KVM VM fd and `manual_protect` lives across the call.
        let ret = unsafe {
            libc::ioctl(
                std::os::fd::AsRawFd::as_raw_fd(fd),
                // The request is `c_ulong` on glibc and `c_int` on musl.
                KVM_ENABLE_CAP as _,
                &manual_protect,
            )
        };
        if ret < 0 {
            debug!(
                "KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2 not enabled: {}",
                std::io::Error::last_os_error()
            );
        }
        ret == 0
    }

    #[cfg(target_arch = "x86_64")]
    fn enable_manual_dirty_log_protect(_fd: &VmFd) -> bool {
        false
    }

    /// Creates the specified number of [`Vcpu`]s.
    ///
    /// Each vCPU gets a clone of the `vcpus_exit_evt` EventFd stored on this KvmVm.
    ///
    /// This also creates the irqchip, which KVM registers on its io bus. Since Linux 6.18 (commit
    /// 7d9a0273c459, "KVM: Avoid synchronize_srcu() in kvm_io_bus_register_dev()") that
    /// registration starts an SRCU grace period, and a memslot update issued right after it waits
    /// for the grace period to end (tens of ms), so register every memslot before calling this.
    /// Intel hosts still wait once: KVM adds its own APIC access page memslot while creating the
    /// first vCPU.
    pub fn create_vcpus(&self, vcpu_count: u8) -> Result<Vec<Vcpu>, VmError> {
        self.arch_pre_create_vcpus(vcpu_count)?;

        let mut vcpus = Vec::with_capacity(vcpu_count as usize);
        for cpu_idx in 0..vcpu_count {
            let exit_evt = self
                .vcpus_exit_evt()
                .try_clone()
                .map_err(VmError::EventFd)?;
            let vcpu = Vcpu::new(cpu_idx, self, exit_evt).map_err(VmError::CreateVcpu)?;
            vcpus.push(vcpu);
        }

        self.arch_post_create_vcpus(vcpu_count)?;

        Ok(vcpus)
    }

    /// Returns a reference to the [`Kvm`] instance.
    pub fn kvm(&self) -> &Kvm {
        &self.common.kvm
    }

    /// Returns a reference to the vCPU exit [`EventFd`].
    pub fn vcpus_exit_evt(&self) -> &EventFd {
        &self.common.vcpus_exit_evt
    }

    /// Returns a locked reference to the vCPU handles.
    pub fn vcpus_handles(&self) -> MutexGuard<'_, Vec<VcpuHandle>> {
        self.common.vcpus_handles.lock().expect("Poisoned lock")
    }

    /// Sets the userfaultfd (used during snapshot restore).
    pub fn set_uffd(&mut self, uffd: Option<Uffd>) {
        self.common.uffd = uffd;
    }

    /// Starts the microVM vCPUs.
    ///
    /// Sets the terminal to raw/non-blocking mode, then spawns a thread per vCPU
    /// and stores the resulting handles. The barrier is used to synchronize TLS
    /// initialization across all vCPU threads before returning.
    pub fn start_vcpus(
        self: &Arc<Self>,
        mut vcpus: Vec<Vcpu>,
        vcpu_seccomp_filter: Arc<crate::seccomp::BpfProgram>,
    ) -> Result<(), StartVcpusError> {
        let vcpu_count = vcpus.len();
        let barrier = Arc::new(Barrier::new(vcpu_count + 1));

        let stdin = std::io::stdin().lock();
        stdin.set_raw_mode().inspect_err(|&err| {
            crate::logger::warn!("Cannot set raw mode for the terminal. {:?}", err);
        })?;
        stdin.set_non_block(true).inspect_err(|&err| {
            crate::logger::warn!("Cannot set non block for the terminal. {:?}", err);
        })?;

        let mut handles = self.vcpus_handles();
        handles.reserve(vcpu_count);
        for mut vcpu in vcpus.drain(..) {
            vcpu.set_mmio_bus(self.common.mmio_bus.clone());
            #[cfg(target_arch = "x86_64")]
            vcpu.kvm_vcpu.set_pio_bus(self.pio_bus.clone());

            handles.push(vcpu.start_threaded(
                self,
                vcpu_seccomp_filter.clone(),
                barrier.clone(),
            )?);
        }
        drop(handles);
        barrier.wait();

        Ok(())
    }

    /// Sends a pause event to all vCPUs and waits for acknowledgement.
    pub fn pause_vcpus(&self) -> Result<(), crate::VmmError> {
        let mut handles = self.vcpus_handles();
        handles
            .iter_mut()
            .try_for_each(|handle| handle.send_event(crate::VcpuEvent::Pause))
            .map_err(|_| crate::VmmError::VcpuMessage)?;

        if handles
            .iter()
            .map(|handle| {
                handle
                    .response_receiver()
                    .recv_timeout(crate::RECV_TIMEOUT_SEC)
            })
            .any(|response| !matches!(response, Ok(crate::VcpuResponse::Paused)))
        {
            return Err(crate::VmmError::VcpuMessage);
        }
        Ok(())
    }

    /// Sends a resume event to all vCPUs and waits for acknowledgement.
    pub fn resume_vcpus(&self) -> Result<(), crate::VmmError> {
        let mut handles = self.vcpus_handles();
        handles
            .iter_mut()
            .try_for_each(|handle| handle.send_event(crate::VcpuEvent::Resume))
            .map_err(|_| crate::VmmError::VcpuMessage)?;

        if handles
            .iter()
            .map(|handle| {
                handle
                    .response_receiver()
                    .recv_timeout(crate::RECV_TIMEOUT_SEC)
            })
            .any(|response| !matches!(response, Ok(crate::VcpuResponse::Resumed)))
        {
            return Err(crate::VmmError::VcpuMessage);
        }
        Ok(())
    }

    /// Saves vCPU states by requesting each vCPU thread to serialize its state.
    pub fn save_vcpu_states(
        &self,
    ) -> Result<Vec<crate::vstate::vcpu::VcpuState>, crate::persist::MicrovmStateError> {
        use crate::persist::MicrovmStateError;

        let mut handles = self.vcpus_handles();
        for handle in handles.iter_mut() {
            handle
                .send_event(crate::VcpuEvent::SaveState)
                .map_err(MicrovmStateError::SignalVcpu)?;
        }

        let vcpu_responses = handles
            .iter()
            .map(|handle| {
                handle
                    .response_receiver()
                    .recv_timeout(crate::RECV_TIMEOUT_SEC)
            })
            .collect::<Result<Vec<crate::VcpuResponse>, _>>()
            .map_err(|_| MicrovmStateError::UnexpectedVcpuResponse)?;

        vcpu_responses
            .into_iter()
            .map(|response| match response {
                crate::VcpuResponse::SavedState(state) => Ok(*state),
                crate::VcpuResponse::Error(err) => Err(MicrovmStateError::SaveVcpuState(err)),
                crate::VcpuResponse::NotAllowed(reason) => {
                    Err(MicrovmStateError::NotAllowed(reason))
                }
                _ => Err(MicrovmStateError::UnexpectedVcpuResponse),
            })
            .collect()
    }

    /// Dumps CPU configuration from all vCPU threads.
    pub fn dump_cpu_config_states(
        &self,
    ) -> Result<Vec<crate::cpu_config::templates::CpuConfiguration>, crate::DumpCpuConfigError>
    {
        use crate::DumpCpuConfigError;

        let mut handles = self.vcpus_handles();
        for handle in handles.iter_mut() {
            handle
                .send_event(crate::VcpuEvent::DumpCpuConfig)
                .map_err(DumpCpuConfigError::SendEvent)?;
        }

        let vcpu_responses = handles
            .iter()
            .map(|handle| {
                handle
                    .response_receiver()
                    .recv_timeout(crate::RECV_TIMEOUT_SEC)
            })
            .collect::<Result<Vec<crate::VcpuResponse>, _>>()
            .map_err(|_| DumpCpuConfigError::UnexpectedResponse)?;

        vcpu_responses
            .into_iter()
            .map(|response| match response {
                crate::VcpuResponse::DumpedCpuConfig(cpu_config) => Ok(*cpu_config),
                crate::VcpuResponse::Error(err) => Err(DumpCpuConfigError::DumpCpuConfig(err)),
                crate::VcpuResponse::NotAllowed(reason) => {
                    Err(DumpCpuConfigError::NotAllowed(reason))
                }
                _ => Err(DumpCpuConfigError::UnexpectedResponse),
            })
            .collect()
    }

    /// Sends finish events to all vCPU threads and joins them.
    pub fn shutdown_vcpus(&self) {
        let mut handles = self.vcpus_handles();
        for (idx, handle) in handles.iter_mut().enumerate() {
            if let Err(err) = handle.send_event(crate::VcpuEvent::Finish) {
                crate::logger::error!("Failed to send VcpuEvent::Finish to vCPU {}: {}", idx, err);
            }
        }
        // Join the vCPU threads by running VcpuHandle::drop().
        handles.clear();
    }

    /// Reserves the next `slot_cnt` contiguous kvm slot ids and returns the first one
    pub fn next_kvm_slot(&self, slot_cnt: u32) -> Option<u32> {
        let next = self
            .common
            .next_kvm_slot
            .fetch_add(slot_cnt, Ordering::Relaxed);
        if self.common.max_memslots <= next {
            None
        } else {
            Some(next)
        }
    }

    pub(crate) fn set_user_memory_region(
        &self,
        region: kvm_userspace_memory_region,
    ) -> Result<(), VmError> {
        #[cfg(test)]
        if self.common.fail_set_user_memory_region_in.try_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |n| (n > 0).then(|| n - 1),
        ) == Ok(1)
        {
            return Err(VmError::SetUserMemoryRegion(kvm_ioctls::Error::new(
                libc::ENOMEM,
            )));
        }
        // SAFETY: Safe because the fd is a valid KVM file descriptor.
        unsafe {
            self.fd()
                .set_user_memory_region(region)
                .map_err(VmError::SetUserMemoryRegion)
        }
    }

    /// Test-only: arm the fault injector so the `n`th subsequent call to
    /// [`Self::set_user_memory_region`] fails (n == 1 fails the very next call).
    #[cfg(test)]
    pub(crate) fn fail_next_set_user_memory_region(&self, n: u32) {
        self.common
            .fail_set_user_memory_region_in
            .store(n, Ordering::Relaxed);
    }

    fn register_memory_region(&mut self, region: Arc<GuestRegionMmapExt>) -> Result<(), VmError> {
        let new_guest_memory = self
            .common
            .guest_memory
            .insert_region(Arc::clone(&region))?;

        region
            .slots()
            .try_for_each(|(ref slot, plugged)| match plugged {
                // if the slot is plugged, add it to kvm user memory regions
                true => self.set_user_memory_region(slot.into()),
                // if the slot is not plugged, protect accesses to it
                false => slot.protect(true).map_err(VmError::MemoryError),
            })?;

        self.common.guest_memory = new_guest_memory;

        Ok(())
    }

    /// Register a list of new memory regions to this [`KvmVm`].
    pub fn register_dram_memory_regions(
        &mut self,
        regions: Vec<GuestRegionMmap>,
    ) -> Result<(), VmError> {
        for region in regions {
            let next_slot = self
                .next_kvm_slot(1)
                .ok_or(VmError::NotEnoughMemorySlots(self.common.max_memslots))?;

            let arcd_region =
                Arc::new(GuestRegionMmapExt::dram_from_mmap_region(region, next_slot));

            self.register_memory_region(arcd_region)?
        }

        Ok(())
    }

    /// Register a new hotpluggable region to this [`KvmVm`].
    pub fn register_hotpluggable_memory_region(
        &mut self,
        region: GuestRegionMmap,
        slot_size: usize,
    ) -> Result<(), VmError> {
        // caller should ensure the slot size divides the region length.
        assert!(region.len().is_multiple_of(slot_size as u64));
        let slot_cnt = (region.len() / (slot_size as u64))
            .try_into()
            .map_err(|_| VmError::NotEnoughMemorySlots(self.common.max_memslots))?;
        let slot_from = self
            .next_kvm_slot(slot_cnt)
            .ok_or(VmError::NotEnoughMemorySlots(self.common.max_memslots))?;
        let arcd_region = Arc::new(GuestRegionMmapExt::hotpluggable_from_mmap_region(
            region, slot_from, slot_size,
        ));

        self.register_memory_region(arcd_region)
    }

    /// Register a list of new memory regions to this [`KvmVm`].
    ///
    /// Note: regions and state.regions need to be in the same order.
    pub fn restore_memory_regions(
        &mut self,
        regions: Vec<GuestRegionMmap>,
        state: &GuestMemoryState,
    ) -> Result<(), VmError> {
        for (region, state) in regions.into_iter().zip(state.regions.iter()) {
            let slot_cnt = state
                .plugged
                .len()
                .try_into()
                .map_err(|_| VmError::NotEnoughMemorySlots(self.common.max_memslots))?;

            let next_slot = self
                .next_kvm_slot(slot_cnt)
                .ok_or(VmError::NotEnoughMemorySlots(self.common.max_memslots))?;

            let arcd_region = Arc::new(GuestRegionMmapExt::from_state(region, state, next_slot)?);

            self.register_memory_region(arcd_region)?
        }

        Ok(())
    }

    /// Gets a reference to the kvm file descriptor owned by this VM.
    pub fn fd(&self) -> &VmFd {
        &self.common.fd
    }

    /// Gets a reference to this [`KvmVm`]'s [`GuestMemoryMmap`] object
    pub fn guest_memory(&self) -> &GuestMemoryMmap {
        &self.common.guest_memory
    }

    /// Whether a vhost-user backend can map this VM's guest memory.
    ///
    /// Guest memory is a shared memfd mapping only when a vhost-user device is
    /// configured before boot. Otherwise it is anonymous, or a private mapping
    /// of a read-only snapshot file, which a backend cannot map at all.
    pub fn vhost_user_memory_shareable(&self) -> bool {
        // Never empty after boot, so `all` cannot pass vacuously.
        self.guest_memory()
            .iter()
            .all(|r| r.is_shared() && r.file_offset().is_some())
    }

    /// Gets a mutable reference to this [`KvmVm`]'s [`ResourceAllocator`] object
    pub fn resource_allocator(&self) -> MutexGuard<'_, ResourceAllocator> {
        self.common
            .resource_allocator
            .lock()
            .expect("Poisoned lock")
    }

    /// Takes the pages dirtied since the last take, one bitmap per region, clearing Firecracker's
    /// bitmap and KVM's log. Guest memory must track dirty pages.
    ///
    /// If reading KVM's log fails, the pages taken so far are marked dirty again.
    pub fn take_dirty_bitmap(&self) -> Result<DirtyBitmap, VmError> {
        let guest_memory = self.guest_memory();
        let mut dirty_bitmap = guest_memory.take_dirty();
        let merged = guest_memory
            .merge_slot_kvm_dirty_logs(&mut dirty_bitmap, |mem_slot| {
                self.fd()
                    .get_dirty_log(mem_slot.slot, mem_slot.slice.len())
                    .map_err(VmError::GetDirtyLog)
            })
            .and_then(|()| {
                // With manual protection the reads above copied the logs. The pages are in
                // `dirty_bitmap` now, so a reset that fails part way loses none of them.
                if self.common.manual_dirty_log_protect {
                    self.reset_kvm_dirty_logs()
                } else {
                    Ok(())
                }
            });
        if let Err(err) = merged {
            guest_memory.store_dirty_bitmap(&dirty_bitmap);
            return Err(err);
        }
        Ok(dirty_bitmap)
    }

    /// Empties KVM's dirty log of every plugged slot and write-protects the slot again, by
    /// turning logging off and on for it: KVM frees the log, allocates an empty one and
    /// write-protects the slot in one pass. No vCPU may be running.
    ///
    /// A failure between the two calls leaves a slot without a log. The next read of it fails
    /// (ENOENT), and with it every later snapshot that reads the dirty log.
    fn reset_kvm_dirty_logs(&self) -> Result<(), VmError> {
        use kvm_bindings::KVM_MEM_LOG_DIRTY_PAGES;
        for region in self.guest_memory().iter() {
            for (mem_slot, plugged) in region.slots() {
                if !plugged || mem_slot.slice.bitmap().is_none() {
                    continue;
                }
                let mut kvm_region = kvm_userspace_memory_region::from(&mem_slot);
                kvm_region.flags &= !KVM_MEM_LOG_DIRTY_PAGES;
                self.set_user_memory_region(kvm_region)?;
                kvm_region.flags |= KVM_MEM_LOG_DIRTY_PAGES;
                self.set_user_memory_region(kvm_region)?;
            }
        }
        Ok(())
    }

    /// Whether guest memory tracks dirty pages. Either every region does, or none does.
    fn tracks_dirty_pages(&self) -> bool {
        let guest_memory = self.guest_memory();
        let tracked = guest_memory
            .iter()
            .filter(|region| (**region).bitmap().is_some())
            .count();
        assert!(tracked == 0 || tracked == guest_memory.num_regions());
        tracked != 0
    }

    /// The pages a Diff of guest memory without dirty tracking writes, one bitmap per region.
    /// For a region that maps a file privately (a restore from a snapshot file, or from a UFFD
    /// minor-fault handler's memfd) they are the pages the guest wrote, read from
    /// /proc/self/pagemap. For any other region they are the pages resident in memory.
    fn written_or_resident_pages(&self) -> Result<DirtyBitmap, VmError> {
        let page_size = host_page_size();
        self.guest_memory()
            .iter()
            .map(|region| {
                let mut bitmap = vec![0; (region.size() / page_size).div_ceil(64)];
                let private_file = region.maps_file_privately();
                for (i, (mem_slot, plugged)) in region.slots().enumerate() {
                    if plugged {
                        let addr = mem_slot.slice.ptr_guard_mut().as_ptr();
                        let len = mem_slot.slice.len();
                        let pages = if private_file {
                            written_private_pages(addr, len)?
                        } else {
                            mincore_bitmap(addr, len)?
                        };
                        bitmap[region.slot_words(i)].copy_from_slice(&pages);
                    }
                }
                Ok(bitmap)
            })
            .collect()
    }

    /// Takes a snapshot of the virtual machine running inside the given [`Vmm`] and saves it to
    /// `mem_file_path`.
    ///
    /// If `snapshot_type` is [`SnapshotType::Diff`], and `mem_file_path` exists and is a snapshot
    /// file of matching size, then the diff snapshot will be directly merged into the existing
    /// snapshot. Otherwise, existing files are simply overwritten. A full snapshot into a file
    /// that held no data (a new file, or one of the wrong size) leaves all-zero pages as holes.
    pub(crate) fn snapshot_memory_to_path(
        &self,
        mem_file_path: &Path,
        snapshot_type: SnapshotType,
        sync_snapshot_files: bool,
    ) -> Result<(), CreateSnapshotError> {
        use self::CreateSnapshotError::*;

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(mem_file_path)
            .map_err(|err| MemoryBackingFile("open", err))?;

        // Determine what size our total memory area is.
        let mem_size_mib = mem_size_mib(self.guest_memory());
        let expected_size = mem_size_mib * 1024 * 1024;

        // The size comes from the file just opened, not from a check before the open, so a file
        // created at the path in between is never taken for an empty one.
        let file_size = file
            .metadata()
            .map_err(|e| MemoryBackingFile("get_metadata", e))?
            .len();

        // Here we only truncate the file if the size mismatches.
        // - For full snapshots, the file's contents will be overwritten anyway, except under an
        //   unplugged slot, where the dump writes at most the pages a discard emptied. We have to
        //   avoid truncating here to deal with the edge case where it represents the snapshot file
        //   from which this very microVM was loaded (as modifying the memory file would be
        //   reflected in the mmap of the file, meaning a truncate operation would zero out guest
        //   memory, and thus corrupt the VM).
        // - For diff snapshots, we want to merge the diff layer directly into the file.
        // A file this call created is empty, so truncating it changes nothing. Either way the
        // file holds no data afterwards.
        let starts_empty = file_size != expected_size;
        if starts_empty {
            file.set_len(0)
                .map_err(|err| MemoryBackingFile("truncate", err))?;
        }

        // Set the length of the file to the full size of the memory area.
        file.set_len(expected_size)
            .map_err(|e| MemoryBackingFile("set_length", e))?;

        // A file that held no data reads as one hole after the set_len above, so a full snapshot
        // may skip zero pages. A page of anonymous memory that has no page table entry is one of
        // them and is skipped without being read, unless a userfaultfd handler supplies the
        // missing pages of guest memory, which then need not be zeros. In a file that already had
        // the right size a skipped page would keep its old bytes, so every page of a plugged slot
        // is written. An unplugged slot cannot be read. With dirty tracking the dump writes the
        // pages a discard emptied there as zeros and leaves the file's bytes under the rest of
        // the slot. Without it nothing records a discard, and the file keeps its bytes under the
        // whole slot.
        let zero_pages = match (starts_empty, &self.common.uffd) {
            (false, _) => ZeroPages::Write,
            (true, Some(_)) => ZeroPages::Hole(AbsentPages::Read),
            (true, None) => ZeroPages::Hole(AbsentPages::Skip),
        };
        // A Diff into a file that already had the right size is written in place over the
        // guest's memory as of the last snapshot, so short clean gaps between written runs are
        // written with them. A read of an untouched page there is a userfaultfd fault its server
        // answers with the snapshot's page, and elsewhere could map the shared zero page, so
        // without a userfaultfd only pages with a page table entry are. A file that held no data
        // gets only the marked pages, so it keeps its holes.
        let clean_gaps = match (starts_empty, &self.common.uffd) {
            (true, _) => CleanGaps::Seek,
            (false, Some(_)) => CleanGaps::Bridge(AbsentPages::Read),
            (false, None) => CleanGaps::Bridge(AbsentPages::Skip),
        };

        self.snapshot_memory_to_file(
            &mut file,
            snapshot_type,
            zero_pages,
            clean_gaps,
            sync_snapshot_files,
        )
    }

    /// Writes guest memory to `file` as a snapshot of `snapshot_type`, taking the dirty and the
    /// discarded pages it covers, then flushes it and, if asked, syncs it. If any of that fails,
    /// the pages taken are marked dirty and discarded again. `zero_pages` says what a full
    /// snapshot does with all-zero pages, and `clean_gaps` what a diff does with the clean pages
    /// between the runs it writes.
    fn snapshot_memory_to_file<F: MemoryFile>(
        &self,
        file: &mut F,
        snapshot_type: SnapshotType,
        zero_pages: ZeroPages,
        clean_gaps: CleanGaps,
        sync_snapshot_files: bool,
    ) -> Result<(), CreateSnapshotError> {
        use self::CreateSnapshotError::{Memory, MemoryBackingFile};

        let guest_memory = self.guest_memory();
        // A discard that failed in an unplugged slot left memory no snapshot can read in a state
        // nothing recorded. A Diff merged into its parent could keep the parent's bytes there.
        if snapshot_type == SnapshotType::Diff
            && guest_memory
                .iter()
                .any(|region| region.failed_discard_of_unplugged_memory())
        {
            return Err(Memory(MemoryError::FailedDiscardOfUnpluggedMemory));
        }
        let dirty_bitmap = match (snapshot_type, self.tracks_dirty_pages()) {
            (_, true) => self.take_dirty_bitmap()?,
            (SnapshotType::Diff, false) => self.written_or_resident_pages()?,
            (SnapshotType::Full, false) => DirtyBitmap::new(),
        };
        let discarded = guest_memory.take_discarded();
        let put_back = |_: &CreateSnapshotError| {
            guest_memory.store_dirty_bitmap(&dirty_bitmap);
            guest_memory.store_discarded(&discarded);
        };

        match snapshot_type {
            SnapshotType::Diff => {
                guest_memory.dump_dirty(file, &dirty_bitmap, &discarded, clean_gaps)
            }
            SnapshotType::Full => guest_memory.dump(file, zero_pages, &discarded),
        }
        .map_err(Memory)
        .inspect_err(put_back)?;
        file.flush()
            .map_err(|err| MemoryBackingFile("flush", err))
            .inspect_err(put_back)?;
        if sync_snapshot_files {
            file.sync_all()
                .map_err(|err| MemoryBackingFile("sync_all", err))
                .inspect_err(put_back)?;
        }
        if snapshot_type == SnapshotType::Full {
            for region in guest_memory.iter() {
                region.forget_failed_discard_of_unplugged_memory();
            }
        }
        Ok(())
    }

    /// Register a device IRQ
    pub fn register_irq(&self, fd: &EventFd, gsi: u32) -> Result<(), errno::Error> {
        self.common.fd.register_irqfd(fd, gsi)?;

        let mut entry = kvm_irq_routing_entry {
            gsi,
            type_: KVM_IRQ_ROUTING_IRQCHIP,
            ..Default::default()
        };
        #[cfg(target_arch = "x86_64")]
        {
            entry.u.irqchip.irqchip = KVM_IRQCHIP_IOAPIC;
        }
        #[cfg(target_arch = "aarch64")]
        {
            entry.u.irqchip.irqchip = 0;
        }
        entry.u.irqchip.pin = gsi;

        self.common
            .interrupts
            .lock()
            .expect("Poisoned lock")
            .insert(
                gsi,
                RoutingEntry {
                    entry,
                    masked: false,
                },
            );
        Ok(())
    }

    /// Register an MSI device interrupt
    pub fn register_msi(
        &self,
        route: &MsixVector,
        table_entry: &MsixTableEntry,
        pci_sbdf: PciSBDF,
    ) -> Result<(), errno::Error> {
        let mut entry = kvm_irq_routing_entry {
            gsi: route.gsi,
            type_: KVM_IRQ_ROUTING_MSI,
            ..Default::default()
        };
        entry.u.msi.address_lo = table_entry.msg_addr_lo;
        entry.u.msi.address_hi = table_entry.msg_addr_hi;
        entry.u.msi.data = table_entry.msg_data;

        if self.common.fd.check_extension(kvm_ioctls::Cap::MsiDevid) {
            entry.flags = KVM_MSI_VALID_DEVID;
            entry.u.msi.__bindgen_anon_1.devid = pci_sbdf.into();
        }

        self.common
            .interrupts
            .lock()
            .expect("Poisoned lock")
            .insert(
                route.gsi,
                RoutingEntry {
                    entry,
                    masked: table_entry.masked(),
                },
            );

        Ok(())
    }

    /// Create a group of MSI-X interrupts
    pub fn create_msix_group(
        vm: Arc<KvmVm>,
        count: u16,
    ) -> Result<MsixVectorGroup, InterruptError> {
        debug!("Creating new MSI group with {count} vectors");
        let mut vectors = Vec::with_capacity(count as usize);
        for gsi in vm
            .resource_allocator()
            .allocate_gsi_msi(count as u32)?
            .iter()
        {
            vectors.push(MsixVector::new(*gsi, false)?);
        }

        Ok(MsixVectorGroup { vm, vectors })
    }

    /// Set GSI routes to KVM
    pub fn set_gsi_routes(&self) -> Result<(), InterruptError> {
        let entries = self.common.interrupts.lock().expect("Poisoned lock");
        let unmasked = entries.values().filter(|e| !e.masked).count();
        let mut routes = KvmIrqRouting::new(unmasked)?;
        for (slot, entry) in entries.values().filter(|e| !e.masked).enumerate() {
            routes.as_mut_slice()[slot] = entry.entry;
        }

        self.common.fd.set_gsi_routing(&routes)?;
        Ok(())
    }
}

/// A file a memory snapshot is written to. Separate from `File` so tests can inject I/O failures
/// and guest memory writes during the copy.
trait MemoryFile: WriteVolatile + Seek + Write {
    fn sync_all(&self) -> std::io::Result<()>;
}

impl MemoryFile for File {
    fn sync_all(&self) -> std::io::Result<()> {
        File::sync_all(self)
    }
}

/// The bits of a /proc/self/pagemap entry for a page in memory (63) and a page swapped out (62).
const PAGEMAP_PRESENT: u64 = 1 << 63;
const PAGEMAP_SWAPPED: u64 = 1 << 62;
/// Bit 61: the page is a file page or shared anonymous memory.
const PAGEMAP_FILE: u64 = 1 << 61;

/// Whether a pagemap entry of a private file mapping is for a page the process wrote: a private
/// anonymous copy, in memory or swapped out. A page it only read maps the file's page cache
/// page (bit 61), and a page it never touched has no entry.
fn pagemap_written(entry: u64) -> bool {
    entry & PAGEMAP_SWAPPED != 0 || (entry & PAGEMAP_PRESENT != 0 && entry & PAGEMAP_FILE == 0)
}

/// The dirty bitmap of a memslot that maps a snapshot file privately with dirty tracking off:
/// the pages the guest wrote since the restore, from /proc/self/pagemap. A write to such a
/// mapping copies the page, so the written pages are exactly the private anonymous ones; KVM's
/// dirty log is not needed, nor the write protection that keeps it. The pages written before an
/// earlier snapshot of this VM are included too, which a diff against that snapshot writes
/// again with the bytes it already holds. Discards are recorded apart (`discarded`).
///
/// pagemap is read with lseek(2) and read(2), which Firecracker's seccomp filters allow. A
/// process that cannot open it (a jail without /proc) takes the slot's bitmap from mincore(2), as
/// it did before: every resident page of the slot, so the Diff is larger, and complete while the
/// host does not swap.
fn written_private_pages(addr: *mut u8, len: usize) -> Result<Vec<u64>, VmError> {
    use std::io::{Read, Seek, SeekFrom};
    let page_size = host_page_size();
    let pages = len / page_size;
    let mut bitmap = vec![0u64; pages.div_ceil(64)];
    let mut file = match std::fs::File::open("/proc/self/pagemap") {
        Ok(file) => file,
        Err(err) => {
            crate::logger::warn!(
                "Cannot open /proc/self/pagemap ({err}), so a Diff takes every resident page of \
                 a slot without dirty tracking (mincore)"
            );
            return mincore_bitmap(addr, len);
        }
    };
    file.seek(SeekFrom::Start((addr as usize / page_size * 8) as u64))
        .map_err(VmError::Pagemap)?;
    // 64 KiB of entries, 32 MiB of guest memory with 4 KiB pages, per read
    let mut entries = vec![0u8; 8192 * 8];
    let mut page = 0;
    while page < pages {
        let count = (pages - page).min(8192);
        let bytes = &mut entries[..count * 8];
        file.read_exact(bytes).map_err(VmError::Pagemap)?;
        for (i, entry) in bytes.chunks_exact(8).enumerate() {
            if pagemap_written(u64::from_ne_bytes(entry.try_into().unwrap())) {
                bitmap[(page + i) / 64] |= 1 << ((page + i) % 64);
            }
        }
        page += count;
    }
    Ok(bitmap)
}

/// Use `mincore(2)` to overapproximate the dirty bitmap for the given memslot. To be used
/// if a diff snapshot is requested, but dirty page tracking wasn't enabled.
fn mincore_bitmap(addr: *mut u8, len: usize) -> Result<Vec<u64>, VmError> {
    // TODO: Once Host 5.10 goes out of support, we can make this more robust and work on
    // swap-enabled systems, by doing mlock2(MLOCK_ONFAULT)/munlock() in this function (to
    // force swapped-out pages to get paged in, so that mincore will consider them incore).
    // However, on AMD (m6a/m7a) 5.10, doing so introduces a 100%/30ms regression to snapshot
    // creation, even if swap is disabled, so currently it cannot be done.

    // Mincore always works at PAGE_SIZE granularity, even if the VMA we are dealing with
    // is a hugetlbfs VMA (e.g. to report a single hugepage as "present", mincore will
    // give us 512 4k markers with the lowest bit set).
    let page_size = host_page_size();
    let mut mincore_bitmap = vec![0u8; len / page_size];
    let mut bitmap = vec![0u64; (len / page_size).div_ceil(64)];

    // SAFETY: The safety invariants of GuestRegionMmap ensure that region.as_ptr() is a valid
    // userspace mapping of size region.len() bytes. The bitmap has exactly one byte for each
    // page in this userspace mapping. Note that mincore does not operate on bitmaps like
    // KVM_MEM_LOG_DIRTY_PAGES, but rather it uses 8 bits per page (e.g. 1 byte), setting the
    // least significant bit to 1 if the page corresponding to a byte is in core (available in
    // the page cache and resolvable via just a minor page fault).
    let r = unsafe { libc::mincore(addr.cast(), len, mincore_bitmap.as_mut_ptr()) };

    if r != 0 {
        return Err(VmError::Mincore(vmm_sys_util::errno::Error::last()));
    }

    for (page_idx, b) in mincore_bitmap.iter().enumerate() {
        bitmap[page_idx / 64] |= (*b as u64 & 0x1) << (page_idx as u64 % 64);
    }

    Ok(bitmap)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::assert_matches;
    use std::sync::atomic::Ordering;

    use vm_memory::GuestAddress;
    use vmm_sys_util::tempdir::TempDir;
    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::arch;
    use crate::pci::PciSBDF;
    use crate::pci::msix::MsixTableEntry;
    use crate::snapshot::Persist;
    use crate::test_utils::single_region_mem_raw;
    use crate::utils::mib_to_bytes;
    use crate::vmm_config::machine_config::HugePageConfig;
    use crate::vstate::kvm::Kvm;
    use crate::vstate::memory::test_utils::data_runs;
    use crate::vstate::memory::{Bytes, GuestRegionType, HOLE_GRANULE, anonymous};

    #[test]
    fn test_written_private_pages_are_the_written_ones() {
        // A private mapping of a 16-page file: page 1 read, pages 3 and 7 written, page 9 read
        // and then written. Only the written pages are reported, the read ones are not.
        let page = host_page_size();
        let file = TempFile::new().unwrap();
        file.as_file().set_len((16 * page) as u64).unwrap();
        std::io::Write::write_all(&mut file.as_file(), &vec![1u8; 16 * page]).unwrap();
        // SAFETY: a private mapping of a file of this length, unmapped below.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                16 * page,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE,
                std::os::fd::AsRawFd::as_raw_fd(file.as_file()),
                0,
            )
        };
        assert_ne!(addr, libc::MAP_FAILED);
        let mem = addr.cast::<u8>();
        // SAFETY: every offset below is inside the mapping.
        unsafe {
            assert_eq!(std::ptr::read_volatile(mem.add(page)), 1);
            std::ptr::write_volatile(mem.add(3 * page), 2);
            std::ptr::write_volatile(mem.add(7 * page + 5), 2);
            assert_eq!(std::ptr::read_volatile(mem.add(9 * page)), 1);
            std::ptr::write_volatile(mem.add(9 * page), 2);
        }
        let bitmap = written_private_pages(mem, 16 * page).unwrap();
        // SAFETY: the mapping made above.
        unsafe { libc::munmap(addr, 16 * page) };
        assert_eq!(bitmap, vec![(1 << 3) | (1 << 7) | (1 << 9)]);
    }

    #[test]
    fn test_pagemap_written_reads_the_entry_bits() {
        assert!(!pagemap_written(0), "an untouched page");
        assert!(
            !pagemap_written(PAGEMAP_PRESENT | PAGEMAP_FILE),
            "a page only read maps the page cache"
        );
        assert!(pagemap_written(PAGEMAP_PRESENT), "a private copy");
        assert!(
            pagemap_written(PAGEMAP_SWAPPED),
            "a private copy swapped out"
        );
    }

    #[test]
    fn test_untracked_diff_of_a_private_file_mapping_writes_its_writes_and_discards() {
        // Guest memory mapped privately from its parent's memory file without dirty tracking,
        // as a File backend restore loaded with track_dirty_pages off. A Diff writes the pages
        // the guest wrote and the pages a discard emptied, and nothing else: the target holds
        // 0xCD so that a page written with the parent's bytes shows. Page 1 is only read, page
        // 3 written, pages 5 and 6 discarded, page 6 written after its discard.
        let size = mib_to_bytes(1);
        let page = host_page_size();
        let dir = TempDir::new().unwrap();
        let parent = dir.as_path().join("parent");
        let parent_bytes: Vec<u8> = (0..size).map(|i| (i / page) as u8 | 1).collect();
        std::fs::write(&parent, &parent_bytes).unwrap();
        let mut vm = setup_vm();
        vm.register_dram_memory_regions(
            crate::vstate::memory::snapshot_file(
                File::open(&parent).unwrap(),
                &[(GuestAddress(0), size)],
                false,
                HugePageConfig::None,
            )
            .unwrap(),
        )
        .unwrap();
        let mem = vm.guest_memory();
        let at = |page_index: usize| GuestAddress((page_index * page) as u64);
        assert_eq!(mem.read_obj::<u8>(at(1)).unwrap(), parent_bytes[page]);
        mem.write_slice(&[0xaa; 64], at(3)).unwrap();
        mem.discard_range(at(5), 2 * page).unwrap();
        mem.write_slice(&[0xbb; 8], GuestAddress((6 * page + 100) as u64))
            .unwrap();

        let mut expected = vec![0xcdu8; size];
        expected[3 * page..4 * page].copy_from_slice(&parent_bytes[3 * page..4 * page]);
        expected[3 * page..3 * page + 64].fill(0xaa);
        expected[5 * page..7 * page].fill(0);
        expected[6 * page + 100..6 * page + 108].fill(0xbb);
        let target = dir.as_path().join("diff");
        std::fs::write(&target, vec![0xcdu8; size]).unwrap();
        vm.snapshot_memory_to_path(&target, SnapshotType::Diff, false)
            .unwrap();
        assert_eq!(first_difference(&target, &expected), None);
    }

    // Auxiliary function being used throughout the tests.
    pub(crate) fn setup_vm() -> KvmVm {
        let kvm = Kvm::new(vec![]).expect("Cannot create Kvm");
        KvmVm::new(kvm).expect("Cannot create new vm")
    }

    /// What a `TestMemoryFile` does besides writing to its file.
    enum TestMemoryFileState<'a> {
        /// Fails the named step: "write", "flush" or "sync".
        Fails(&'static str),
        /// After each write of guest memory to the file, writes `value` to guest memory at `addr`.
        Races {
            memory: &'a GuestMemoryMmap,
            addr: GuestAddress,
            value: u8,
        },
    }

    /// A memory file that fails a step, or races guest memory writes with the copy.
    struct TestMemoryFile<'a> {
        file: File,
        state: TestMemoryFileState<'a>,
    }

    impl TestMemoryFile<'_> {
        fn check(&self, step: &str) -> std::io::Result<()> {
            match self.state {
                TestMemoryFileState::Fails(fails) if fails == step => {
                    Err(std::io::Error::other(format!("{step} failed")))
                }
                _ => Ok(()),
            }
        }
    }

    impl WriteVolatile for TestMemoryFile<'_> {
        fn write_volatile<B: vm_memory::bitmap::BitmapSlice>(
            &mut self,
            buf: &vm_memory::VolatileSlice<B>,
        ) -> Result<usize, vm_memory::VolatileMemoryError> {
            use crate::vstate::memory::Bytes;

            self.check("write")
                .map_err(vm_memory::VolatileMemoryError::IOError)?;
            let written = self.file.write_volatile(buf)?;
            if let TestMemoryFileState::Races {
                memory,
                addr,
                value,
            } = self.state
            {
                memory.write_obj(value, addr).unwrap();
            }
            Ok(written)
        }
    }

    impl Seek for TestMemoryFile<'_> {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            self.file.seek(pos)
        }
    }

    impl Write for TestMemoryFile<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.file.write(buf)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.check("flush")?;
            self.file.flush()
        }
    }

    impl MemoryFile for TestMemoryFile<'_> {
        fn sync_all(&self) -> std::io::Result<()> {
            self.check("sync")?;
            self.file.sync_all()
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_taking_the_dirty_log_leaves_each_slot_logging_with_an_empty_log() {
        use vmm_sys_util::tempfile::TempFile;

        // The take empties KVM's logs by turning logging off and on for each slot. A slot left
        // with logging off has no log, and the next read of it fails with ENOENT.
        let page_size = host_page_size();
        let mut vm = setup_vm();
        if !vm.common.manual_dirty_log_protect {
            println!("Skipping: this host's KVM has no KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2");
            return;
        }
        let regions = anonymous(
            &[(GuestAddress(0), 4 * page_size)],
            true,
            HugePageConfig::None,
        );
        vm.register_dram_memory_regions(regions.unwrap()).unwrap();
        let slots: Vec<(u32, usize)> = vm
            .guest_memory()
            .iter()
            .flat_map(|region| region.plugged_slots())
            .map(|slot| (slot.slot, slot.slice.len()))
            .collect();
        assert!(!slots.is_empty());

        for _ in 0..2 {
            vm.take_dirty_bitmap().unwrap();
            for (slot, len) in &slots {
                let log = vm
                    .fd()
                    .get_dirty_log(*slot, *len)
                    .expect("the take left the slot without a dirty log");
                assert!(log.iter().all(|word| *word == 0));
            }
        }

        let file = TempFile::new().unwrap();
        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Full, false)
            .unwrap();
        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Diff, false)
            .unwrap();
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_a_failed_dirty_log_reset_keeps_the_pages_taken() {
        use crate::vstate::memory::{Bitmap, Bytes};

        // The reset fails at its first call, after the logs were read. The pages already taken
        // must be dirty again, or the next Diff would leave them out.
        let page_size = host_page_size();
        let mut vm = setup_vm();
        if !vm.common.manual_dirty_log_protect {
            println!("Skipping: this host's KVM has no KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2");
            return;
        }
        let regions = anonymous(
            &[(GuestAddress(0), 4 * page_size)],
            true,
            HugePageConfig::None,
        );
        vm.register_dram_memory_regions(regions.unwrap()).unwrap();
        vm.guest_memory()
            .write(&[1], GuestAddress(page_size as u64))
            .unwrap();
        let is_dirty = || {
            vm.guest_memory()
                .iter()
                .any(|region| region.bitmap().dirty_at(page_size))
        };
        assert!(is_dirty());

        vm.fail_next_set_user_memory_region(1);
        let err = vm.take_dirty_bitmap().unwrap_err();
        assert!(matches!(err, VmError::SetUserMemoryRegion(_)), "{err}");
        assert!(is_dirty(), "the failed take lost a dirty page");

        // The slot still logs (the first call, which failed, was the one turning logging
        // off), so the next take succeeds and returns the page.
        let taken = vm.take_dirty_bitmap().unwrap();
        assert!(
            taken
                .iter()
                .any(|region| region.iter().any(|word| *word != 0))
        );
        assert!(!is_dirty());
    }

    #[test]
    fn test_failed_memory_snapshot_keeps_dirty_pages() {
        use vmm_sys_util::tempfile::TempFile;

        use crate::vstate::memory::{Bitmap, Bytes};

        let page_size = host_page_size();
        let mut vm = setup_vm();
        let regions = anonymous(
            &[(GuestAddress(0), 4 * page_size)],
            true,
            HugePageConfig::None,
        );
        vm.register_dram_memory_regions(regions.unwrap()).unwrap();
        let is_dirty = || {
            vm.guest_memory()
                .iter()
                .any(|region| region.bitmap().dirty_at(page_size))
        };

        for snapshot_type in [SnapshotType::Diff, SnapshotType::Full] {
            vm.guest_memory()
                .write(&[1], GuestAddress(page_size as u64))
                .unwrap();
            for fails in ["write", "flush", "sync"] {
                let mut file = TestMemoryFile {
                    file: TempFile::new().unwrap().into_file(),
                    state: TestMemoryFileState::Fails(fails),
                };
                vm.snapshot_memory_to_file(
                    &mut file,
                    snapshot_type,
                    ZeroPages::Write,
                    CleanGaps::Seek,
                    true,
                )
                .unwrap_err();
                assert!(is_dirty(), "{snapshot_type:?}: {fails} failed");
            }

            let mut file = TempFile::new().unwrap().into_file();
            vm.snapshot_memory_to_file(
                &mut file,
                snapshot_type,
                ZeroPages::Write,
                CleanGaps::Seek,
                true,
            )
            .unwrap();
            assert!(!is_dirty(), "{snapshot_type:?} succeeded");
        }
    }

    #[test]
    fn test_write_during_memory_snapshot_stays_dirty() {
        use std::os::unix::fs::FileExt;

        use vmm_sys_util::tempfile::TempFile;

        use crate::vstate::memory::{Bitmap, Bytes};

        let page_size = host_page_size();
        let mut vm = setup_vm();
        let regions = anonymous(
            &[(GuestAddress(0), 4 * page_size)],
            true,
            HugePageConfig::None,
        );
        vm.register_dram_memory_regions(regions.unwrap()).unwrap();
        let addr = GuestAddress(page_size as u64);
        let mut file = TempFile::new().unwrap().into_file();

        for snapshot_type in [SnapshotType::Diff, SnapshotType::Full] {
            vm.guest_memory().write_obj(1u8, addr).unwrap();
            let mut racing_file = TestMemoryFile {
                file,
                state: TestMemoryFileState::Races {
                    memory: vm.guest_memory(),
                    addr,
                    value: 2,
                },
            };
            vm.snapshot_memory_to_file(
                &mut racing_file,
                snapshot_type,
                ZeroPages::Write,
                CleanGaps::Seek,
                false,
            )
            .unwrap();
            file = racing_file.file;
            let is_dirty = vm
                .guest_memory()
                .iter()
                .any(|region| region.bitmap().dirty_at(page_size));
            assert!(is_dirty, "{snapshot_type:?}");

            file.rewind().unwrap();
            vm.snapshot_memory_to_file(
                &mut file,
                SnapshotType::Diff,
                ZeroPages::Write,
                CleanGaps::Seek,
                false,
            )
            .unwrap();
            let mut byte = [0];
            file.read_exact_at(&mut byte, page_size as u64).unwrap();
            assert_eq!(byte, [2], "{snapshot_type:?}");
            file.rewind().unwrap();
        }
    }

    #[test]
    fn test_failed_dirty_log_read_keeps_firecracker_marks() {
        use crate::vstate::memory::{Bitmap, Bytes};

        let page_size = host_page_size();
        let mut vm = setup_vm();
        let dram = anonymous(
            &[(GuestAddress(0), 4 * page_size)],
            true,
            HugePageConfig::None,
        );
        vm.register_dram_memory_regions(dram.unwrap()).unwrap();
        let hotplug = anonymous(
            &[(GuestAddress(mib_to_bytes(1) as u64), 64 * page_size)],
            true,
            HugePageConfig::None,
        );
        vm.register_hotpluggable_memory_region(
            hotplug.unwrap().into_iter().next().unwrap(),
            64 * page_size,
        )
        .unwrap();
        vm.guest_memory()
            .write(&[1], GuestAddress(page_size as u64))
            .unwrap();

        // Marking the hotplug slot plugged without adding it to KVM makes reading its dirty log
        // fail, after the DRAM slot's log was read.
        let hotplug = vm.guest_memory().iter().nth(1).unwrap();
        hotplug.plugged.lock().unwrap().set(0, true);
        vm.take_dirty_bitmap().unwrap_err();

        let dram = vm.guest_memory().iter().next().unwrap();
        assert!(dram.bitmap().dirty_at(page_size));
    }

    // Auxiliary function being used throughout the tests.
    pub(crate) fn setup_vm_with_memory(mem_size: usize) -> KvmVm {
        let mut vm = setup_vm();
        let gm = single_region_mem_raw(mem_size);
        vm.register_dram_memory_regions(gm).unwrap();
        vm
    }

    // Offset of the one all-zero page in the guest memory of setup_vm_with_one_zero_page.
    const ZERO_PAGE: usize = 2 * HOLE_GRANULE;

    // A VM with 1 MiB of guest memory holding 0x11 everywhere but one all-zero page at
    // ZERO_PAGE. Returns the VM and its memory contents.
    fn setup_vm_with_one_zero_page() -> (KvmVm, Vec<u8>) {
        let mem_size = mib_to_bytes(1);
        let vm = setup_vm_with_memory(mem_size);
        let mut contents = vec![0x11u8; mem_size];
        contents[ZERO_PAGE..ZERO_PAGE + HOLE_GRANULE].fill(0);
        vm.guest_memory()
            .write_slice(&contents, GuestAddress(0))
            .unwrap();
        (vm, contents)
    }

    // Offset of the first byte where the file differs from `expected`, or None.
    fn first_difference(path: &Path, expected: &[u8]) -> Option<usize> {
        let contents = std::fs::read(path).unwrap();
        assert_eq!(contents.len(), expected.len());
        contents.iter().zip(expected).position(|(a, b)| a != b)
    }

    #[test]
    fn test_full_snapshot_overwrites_zero_pages_in_an_existing_file() {
        let (vm, contents) = setup_vm_with_one_zero_page();
        // A file of the snapshot's size full of 0xFF stands in for an earlier snapshot.
        let file = TempFile::new().unwrap();
        std::fs::write(file.as_path(), vec![0xFFu8; contents.len()]).unwrap();

        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Full, false)
            .unwrap();

        // Seeking over the zero page would leave the old 0xFF bytes under it.
        assert_eq!(first_difference(file.as_path(), &contents), None);
    }

    #[test]
    fn test_diff_into_an_empty_target_bridges_no_clean_gap() {
        // A clean page is written with the runs around it only over a file that holds the
        // guest's memory. A Diff into a file that held none keeps it a hole, even where the page
        // has a page table entry and lies between two written pages.
        let size = mib_to_bytes(1);
        let page = host_page_size();
        let mut vm = setup_vm();
        vm.register_dram_memory_regions(
            anonymous(&[(GuestAddress(0), size)], true, HugePageConfig::None).unwrap(),
        )
        .unwrap();
        let mem = vm.guest_memory();
        // Every page gets a page table entry, and a Full snapshot takes the marks.
        mem.write_slice(&vec![0x11; size], GuestAddress(0)).unwrap();
        let dir = TempDir::new().unwrap();
        vm.snapshot_memory_to_path(&dir.as_path().join("full"), SnapshotType::Full, false)
            .unwrap();
        mem.write_slice(&vec![0x22; page], GuestAddress(0)).unwrap();
        mem.write_slice(&vec![0x22; page], GuestAddress(2 * page as u64))
            .unwrap();

        let diff = dir.as_path().join("diff");
        vm.snapshot_memory_to_path(&diff, SnapshotType::Diff, false)
            .unwrap();
        let page = page as u64;
        assert_eq!(
            data_runs(&File::open(&diff).unwrap()),
            vec![(0, page), (2 * page, page)]
        );
    }

    #[test]
    fn test_full_snapshot_leaves_zero_pages_as_holes_in_an_empty_target() {
        let (vm, contents) = setup_vm_with_one_zero_page();
        // A path with no file yet, and a file of the wrong size, which the snapshot truncates.
        let dir = TempDir::new().unwrap();
        let new_path = dir.as_path().join("new");
        let wrong_size_path = dir.as_path().join("wrong-size");
        std::fs::write(&wrong_size_path, vec![0xFFu8; 2 * contents.len()]).unwrap();

        for path in [new_path, wrong_size_path] {
            vm.snapshot_memory_to_path(&path, SnapshotType::Full, false)
                .unwrap();

            let len = contents.len() as u64;
            let (zero_page, page) = (ZERO_PAGE as u64, HOLE_GRANULE as u64);
            assert_eq!(
                data_runs(&File::open(&path).unwrap()),
                vec![(0, zero_page), (zero_page + page, len - zero_page - page)],
                "{path:?}"
            );
            assert_eq!(first_difference(&path, &contents), None, "{path:?}");
        }
    }

    /// Size of the guest memory the tests of never-written pages dump: whole MiBs, and several
    /// 2 MiB extents, so a transparent huge page that a write in the first extent populates
    /// cannot reach the pages the tests expect to stay absent.
    const DUMP_TEST_MEM: usize = 8 << 20;

    /// Whether each page of `[addr, addr + len)` is mapped or swapped out, from
    /// `/proc/self/pagemap`.
    fn populated_pages(addr: usize, len: usize) -> Vec<bool> {
        use std::io::{Read, SeekFrom};

        let page_size = host_page_size();
        let mut pagemap = File::open("/proc/self/pagemap").unwrap();
        pagemap
            .seek(SeekFrom::Start((addr / page_size * 8) as u64))
            .unwrap();
        let mut bytes = vec![0u8; len / page_size * 8];
        pagemap.read_exact(&mut bytes).unwrap();
        bytes
            .chunks_exact(8)
            .map(|e| u64::from_ne_bytes(e.try_into().unwrap()) & (0b11 << 62) != 0)
            .collect()
    }

    /// Writes two marks into the first 2 MiB of guest memory and returns them with their guest
    /// addresses.
    fn write_marks(vm: &KvmVm) -> [(u64, [u8; 16]); 2] {
        let marks = [(0, [0xab; 16]), (5 * host_page_size() as u64, [0xcd; 16])];
        for (addr, bytes) in &marks {
            vm.guest_memory()
                .write_slice(bytes, GuestAddress(*addr))
                .unwrap();
        }
        marks
    }

    /// The file must hold the marks, over each page's `filler` byte.
    fn assert_file_holds_marks_on(
        path: &Path,
        marks: &[(u64, [u8; 16])],
        filler: impl Fn(usize) -> u8,
    ) {
        let file = std::fs::read(path).unwrap();
        assert_eq!(file.len(), DUMP_TEST_MEM);
        let mut expected: Vec<u8> = (0..DUMP_TEST_MEM)
            .map(|byte| filler(byte / host_page_size()))
            .collect();
        for (addr, bytes) in marks {
            let addr = usize::try_from(*addr).unwrap();
            expected[addr..addr + bytes.len()].copy_from_slice(bytes);
        }
        // Compare page by page so a failure names the page instead of printing 8 MiB.
        for (page, (got, want)) in file
            .chunks(host_page_size())
            .zip(expected.chunks(host_page_size()))
            .enumerate()
        {
            assert!(
                got == want,
                "snapshot page {page} differs from guest memory"
            );
        }
    }

    #[test]
    fn test_full_snapshot_leaves_never_written_guest_pages_unmapped() {
        // Reading a never-written anonymous page maps the shared zero page, and the guest's first
        // write to it after the snapshot is then a copy-on-write whose MMU notifier invalidation
        // unmaps every shadow stage-2 of a nested guest. The dump must leave those pages as they
        // were: unmapped in the VMM, and holes in the file.
        let vm = setup_vm_with_memory(DUMP_TEST_MEM);
        let marks = write_marks(&vm);
        let host = vm.guest_memory().get_host_address(GuestAddress(0)).unwrap() as usize;
        let tail = (host + DUMP_TEST_MEM / 2, DUMP_TEST_MEM / 2);
        assert!(
            populated_pages(tail.0, tail.1).iter().all(|p| !p),
            "test setup: the second half of guest memory must start never written"
        );

        let file = TempFile::new().unwrap();
        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Full, false)
            .unwrap();

        let mapped = populated_pages(tail.0, tail.1)
            .iter()
            .filter(|p| **p)
            .count();
        assert_eq!(
            mapped, 0,
            "the Full snapshot mapped {mapped} never-written guest pages into the VMM"
        );
        assert_file_holds_marks_on(file.as_path(), &marks, |_| 0);
        // Only the two granules that hold a mark are data. The rest of the file is a hole, not
        // written zeros.
        assert_eq!(
            data_runs(file.as_file()),
            marks.map(|(addr, _)| (addr, HOLE_GRANULE as u64))
        );
    }

    #[test]
    fn test_full_snapshot_into_a_file_of_the_same_size_writes_every_page() {
        // A Full snapshot overwrites a memory file of the right size in place, so a page it left
        // as a hole there would keep the old file's data. Every page must be written, the
        // never-written ones included.
        let vm = setup_vm_with_memory(DUMP_TEST_MEM);
        let marks = write_marks(&vm);
        let file = TempFile::new().unwrap();
        std::fs::write(file.as_path(), vec![0xffu8; DUMP_TEST_MEM]).unwrap();

        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Full, false)
            .unwrap();

        assert_file_holds_marks_on(file.as_path(), &marks, |_| 0);
    }

    #[test]
    fn test_full_snapshot_of_userfaultfd_memory_reads_every_page() {
        // A restored VM's missing page holds what its page server will supply, not zeros, so the
        // dump must read it (and the server fill it) rather than leave a hole.
        use std::os::fd::{AsRawFd, FromRawFd};
        use std::sync::atomic::AtomicBool;

        use userfaultfd::{Event, UffdBuilder};

        const SERVED: u8 = 0x5a;
        let mut vm = setup_vm_with_memory(DUMP_TEST_MEM);
        let marks = write_marks(&vm);
        let host = vm.guest_memory().get_host_address(GuestAddress(0)).unwrap();
        // Created as for a snapshot restore: the dump faults guest memory from kernel context.
        let uffd = UffdBuilder::new()
            .close_on_exec(true)
            .non_blocking(true)
            .user_mode_only(false)
            .create()
            .expect("this test needs vm.unprivileged_userfaultfd=1 or CAP_SYS_PTRACE");
        // A page the marks' writes populated (a whole huge page, when transparent huge pages are
        // always on) holds zeros around the marks; the server fills every other page.
        let populated = populated_pages(host as usize, DUMP_TEST_MEM);
        assert!(
            populated.iter().any(|p| !p),
            "test setup: no page left to serve"
        );
        uffd.register(host.cast(), DUMP_TEST_MEM).unwrap();
        // SAFETY: dup returns a new descriptor that the server's Uffd owns.
        let server = unsafe { Uffd::from_raw_fd(libc::dup(uffd.as_raw_fd())) };
        vm.set_uffd(Some(uffd));

        let stop = Arc::new(AtomicBool::new(false));
        let page_size = host_page_size();
        let serving = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let page = vec![SERVED; page_size];
                while !stop.load(Ordering::Relaxed) {
                    let mut pollfd = libc::pollfd {
                        fd: server.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    // SAFETY: polls one valid pollfd for 50 ms.
                    if unsafe { libc::poll(&mut pollfd, 1, 50) } <= 0 {
                        continue;
                    }
                    while let Some(event) = server.read_event().unwrap() {
                        if let Event::Pagefault { addr, .. } = event {
                            let addr = (addr as usize) & !(page_size - 1);
                            // SAFETY: copies one page from a live buffer to the faulting
                            // page of the registered guest memory.
                            unsafe {
                                server
                                    .copy(page.as_ptr().cast(), addr as *mut _, page_size, true)
                                    .unwrap();
                            }
                        }
                    }
                }
            })
        };

        let file = TempFile::new().unwrap();
        let result = vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Full, false);
        stop.store(true, Ordering::Relaxed);
        serving.join().unwrap();
        result.unwrap();

        assert_file_holds_marks_on(file.as_path(), &marks, |page| {
            if populated[page] { 0 } else { SERVED }
        });
    }

    #[test]
    fn test_failed_memory_snapshot_keeps_discarded_pages() {
        use std::os::unix::fs::FileExt;

        let page_size = host_page_size();
        let len = 4 * page_size;
        let mut vm = setup_vm();
        let regions = anonymous(&[(GuestAddress(0), len)], true, HugePageConfig::None);
        vm.register_dram_memory_regions(regions.unwrap()).unwrap();
        vm.guest_memory()
            .write_slice(&vec![0xab; len], GuestAddress(0))
            .unwrap();
        // The parent of the Diff snapshots below.
        let mut file = TempFile::new().unwrap().into_file();
        vm.snapshot_memory_to_file(
            &mut file,
            SnapshotType::Full,
            ZeroPages::Write,
            CleanGaps::Seek,
            false,
        )
        .unwrap();
        vm.guest_memory()
            .discard_range(GuestAddress(page_size as u64), page_size)
            .unwrap();

        // The discarded page is the only one a Diff writes, so each step fails with it taken.
        for fails in ["write", "flush", "sync"] {
            let mut failing = TestMemoryFile {
                file: TempFile::new().unwrap().into_file(),
                state: TestMemoryFileState::Fails(fails),
            };
            vm.snapshot_memory_to_file(
                &mut failing,
                SnapshotType::Diff,
                ZeroPages::Write,
                CleanGaps::Seek,
                true,
            )
            .unwrap_err();
        }

        file.rewind().unwrap();
        vm.snapshot_memory_to_file(
            &mut file,
            SnapshotType::Diff,
            ZeroPages::Write,
            CleanGaps::Seek,
            false,
        )
        .unwrap();
        let mut expected = vec![0xab; len];
        expected[page_size..2 * page_size].fill(0);
        let mut snapshot = vec![0xff; len];
        file.read_exact_at(&mut snapshot, 0).unwrap();
        assert_eq!(
            snapshot.iter().zip(&expected).position(|(a, b)| a != b),
            None,
            "a failed snapshot lost the discard"
        );
    }

    // The size of each region of setup_vm_with_hotplug_slot: whole MiBs, and a hotpluggable slot
    // of a multiple of 64 host pages.
    fn hotplug_test_region_size() -> usize {
        mib_to_bytes(1).max(64 * host_page_size())
    }

    // A VM with one DRAM region and one hotpluggable slot of hotplug_test_region_size() bytes
    // each, both with dirty tracking, and the guest address of the slot. The slot starts
    // unplugged.
    fn setup_vm_with_hotplug_slot() -> (KvmVm, GuestAddress) {
        let size = hotplug_test_region_size();
        let hotplug = GuestAddress(1 << 30);
        let mut vm = setup_vm();
        vm.register_dram_memory_regions(
            anonymous(&[(GuestAddress(0), size)], true, HugePageConfig::None).unwrap(),
        )
        .unwrap();
        vm.register_hotpluggable_memory_region(
            anonymous(&[(hotplug, size)], true, HugePageConfig::None)
                .unwrap()
                .pop()
                .unwrap(),
            size,
        )
        .unwrap();
        (vm, hotplug)
    }

    // Plugs or unplugs the slot of setup_vm_with_hotplug_slot, and discards it when unplugging,
    // as virtio-mem does.
    fn set_hotplug_slot_plugged(vm: &KvmVm, plugged: bool) {
        let region = vm
            .guest_memory()
            .iter()
            .find(|region| region.region_type == GuestRegionType::Hotpluggable)
            .unwrap();
        let slot = region.mem_slot(region.slot_from);
        region.update_slot(vm, &slot, plugged).unwrap();
        if !plugged {
            vm.guest_memory()
                .discard_range(slot.guest_addr, slot.slice.len())
                .unwrap();
        }
    }

    // Plugs the slot of setup_vm_with_hotplug_slot, fills all guest memory with 0x11, and takes a
    // Full snapshot of it to `path`. Returns the memory file's expected contents.
    fn snapshot_vm_with_hotplug_slot(vm: &KvmVm, hotplug: GuestAddress, path: &Path) -> Vec<u8> {
        let size = hotplug_test_region_size();
        set_hotplug_slot_plugged(vm, true);
        let mem = vm.guest_memory();
        mem.write_slice(&vec![0x11; size], GuestAddress(0)).unwrap();
        mem.write_slice(&vec![0x11; size], hotplug).unwrap();
        vm.snapshot_memory_to_path(path, SnapshotType::Full, false)
            .unwrap();
        vec![0x11; 2 * size]
    }

    #[test]
    fn test_diff_snapshot_writes_the_discarded_pages_of_an_unplugged_slot() {
        // virtio-mem takes a slot out of KVM before it discards it, and no dump reads an
        // unplugged slot. The zeros still have to reach the memory file, or a VM restored from it
        // that plugs the slot back reads old bytes where this one reads zeros.
        let (vm, hotplug) = setup_vm_with_hotplug_slot();
        let file = TempFile::new().unwrap();
        let mut expected = snapshot_vm_with_hotplug_slot(&vm, hotplug, file.as_path());
        let size = hotplug_test_region_size();

        set_hotplug_slot_plugged(&vm, false);
        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Diff, false)
            .unwrap();
        expected[size..].fill(0);
        assert_eq!(
            first_difference(file.as_path(), &expected),
            None,
            "the Diff taken with the slot unplugged"
        );

        set_hotplug_slot_plugged(&vm, true);
        let page = host_page_size();
        vm.guest_memory()
            .write_slice(&vec![0x22; page], hotplug)
            .unwrap();
        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Diff, false)
            .unwrap();
        expected[size..size + page].fill(0x22);
        assert_eq!(
            first_difference(file.as_path(), &expected),
            None,
            "the Diff taken after the slot was plugged back"
        );
    }

    #[test]
    fn test_diff_snapshot_is_refused_after_a_failed_discard_of_an_unplugged_slot() {
        // The discard of an unplugged slot fails partway: its first page is freed, its second,
        // locked, is not, and nothing recorded which. No snapshot can read the slot, so a Diff,
        // which would leave the parent's bytes there, is refused until a Full snapshot is taken.
        let (vm, hotplug) = setup_vm_with_hotplug_slot();
        let file = TempFile::new().unwrap();
        snapshot_vm_with_hotplug_slot(&vm, hotplug, file.as_path());
        let page = host_page_size();
        let mem = vm.guest_memory();
        let region = mem
            .iter()
            .find(|region| region.region_type == GuestRegionType::Hotpluggable)
            .unwrap();
        let slot = region.mem_slot(region.slot_from);
        let second = mem
            .get_host_address(GuestAddress(hotplug.0 + page as u64))
            .unwrap();
        let locked = unsafe { libc::mlock(second.cast(), page) };
        assert_eq!(locked, 0, "mlock: {}", std::io::Error::last_os_error());
        region.update_slot(&vm, &slot, false).unwrap();
        mem.discard_range(slot.guest_addr, slot.slice.len())
            .unwrap_err();
        unsafe { libc::munlock(second.cast(), page) };

        for _ in 0..2 {
            let err = vm
                .snapshot_memory_to_path(file.as_path(), SnapshotType::Diff, false)
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    CreateSnapshotError::Memory(MemoryError::FailedDiscardOfUnpluggedMemory)
                ),
                "{err}"
            );
        }
        // A Full snapshot is a new base, and Diffs on it are taken again.
        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Full, false)
            .unwrap();
        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Diff, false)
            .unwrap();
    }

    #[test]
    fn test_full_snapshot_writes_the_discarded_pages_of_an_unplugged_slot() {
        // A Full snapshot into a file of the right size overwrites the file in place, so it has
        // to write the zeros an unplugged slot's discard left too. Only dirty tracking records
        // the discard, and this VM has it: without it the dump seeks over the whole unplugged
        // slot, and the file keeps its bytes there.
        let (vm, hotplug) = setup_vm_with_hotplug_slot();
        let file = TempFile::new().unwrap();
        let mut expected = snapshot_vm_with_hotplug_slot(&vm, hotplug, file.as_path());

        set_hotplug_slot_plugged(&vm, false);
        vm.snapshot_memory_to_path(file.as_path(), SnapshotType::Full, false)
            .unwrap();
        expected[hotplug_test_region_size()..].fill(0);
        assert_eq!(first_difference(file.as_path(), &expected), None);
    }

    #[test]
    fn test_vhost_user_memory_shareable() {
        // Anonymous: what a VM booted without a vhost-user device gets.
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        assert!(!vm.vhost_user_memory_shareable());

        // memfd-backed: what a VM configured with a vhost-user block before boot
        // gets.
        let mut vm = setup_vm();
        let regions = arch::arch_memory_regions(mib_to_bytes(128)).unwrap();
        let total: u64 = regions.iter().map(|&(_, size)| size as u64).sum();
        let gm = crate::vstate::memory::MemfdBacking::new(total, HugePageConfig::None)
            .unwrap()
            .allocate(&regions, false, HugePageConfig::None)
            .unwrap();
        vm.register_dram_memory_regions(gm).unwrap();
        assert!(vm.vhost_user_memory_shareable());

        // A snapshot file is mapped private: it has a descriptor and must still be
        // rejected.
        let mut vm = setup_vm();
        let file = vmm_sys_util::tempfile::TempFile::new().unwrap().into_file();
        let regions = arch::arch_memory_regions(mib_to_bytes(128)).unwrap();
        let total: u64 = regions.iter().map(|&(_, size)| size as u64).sum();
        file.set_len(total).unwrap();
        let gm = crate::vstate::memory::snapshot_file(file, &regions, false, HugePageConfig::None)
            .unwrap();
        vm.register_dram_memory_regions(gm).unwrap();
        assert!(!vm.vhost_user_memory_shareable());
    }

    #[test]
    fn test_new() {
        // Testing with a valid /dev/kvm descriptor.
        let kvm = Kvm::new(vec![]).expect("Cannot create Kvm");
        KvmVm::new(kvm).unwrap();
    }

    #[test]
    fn test_register_memory_regions() {
        let mut vm = setup_vm();

        // Trying to set a memory region with a size that is not a multiple of GUEST_PAGE_SIZE
        // will result in error.
        let gm = single_region_mem_raw(0x10);
        let res = vm.register_dram_memory_regions(gm);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Cannot set the memory regions: Invalid argument (os error 22)"
        );

        let gm = single_region_mem_raw(0x1000);
        let res = vm.register_dram_memory_regions(gm);
        res.unwrap();
    }

    #[test]
    fn test_too_many_regions() {
        let mut vm = setup_vm();
        let max_nr_regions = vm.kvm().max_nr_memslots();

        for i in 0..=max_nr_regions {
            let regions = anonymous(
                &[(GuestAddress(i as u64 * 0x1000), 0x1000)],
                false,
                HugePageConfig::None,
            )
            .unwrap();
            assert_eq!(regions.len(), 1);

            let res = vm.register_dram_memory_regions(regions);

            if max_nr_regions <= i {
                assert!(
                    matches!(res, Err(VmError::NotEnoughMemorySlots(v)) if v == max_nr_regions),
                    "{:?} at iteration {}",
                    res,
                    i
                );
            } else {
                res.unwrap_or_else(|_| {
                    panic!(
                        "to be able to insert more regions in iteration {i} - max_nr_memslots: \
                         {max_nr_regions} - num_regions: {}",
                        vm.guest_memory().num_regions()
                    )
                });
            }
        }
    }

    #[test]
    fn test_create_vcpus() {
        let vcpu_count = 2;
        let vm = setup_vm_with_memory(mib_to_bytes(128));

        let vcpu_vec = vm.create_vcpus(vcpu_count).unwrap();

        assert_eq!(vcpu_vec.len(), vcpu_count as usize);
    }

    fn enable_irqchip(vm: &KvmVm) {
        #[cfg(target_arch = "x86_64")]
        vm.setup_irqchip().unwrap();
        #[cfg(target_arch = "aarch64")]
        vm.setup_irqchip(1).unwrap();
    }

    #[test]
    fn test_msi_vector_group_new() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();
        assert_eq!(msix_group.num_vectors(), 4);
    }

    /// Enable all vectors MSI-X vector group
    pub fn enable_all_vectors(vector_group: &MsixVectorGroup) {
        for route in &vector_group.vectors {
            route.enable(&vector_group.vm.common.fd).unwrap();
        }
    }

    #[test]
    fn test_msi_vector_group_disable() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();

        // Initially all vectors are disabled
        for route in &msix_group.vectors {
            assert!(!route.enabled.load(Ordering::Acquire))
        }
        // Enabling vectors
        enable_all_vectors(&msix_group);

        // Disable works
        msix_group.disable().unwrap();
        for route in &msix_group.vectors {
            assert!(!route.enabled.load(Ordering::Acquire))
        }
        // Disabling a disabled group doesn't error out
    }

    #[test]
    fn test_msi_vector_group_trigger() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);

        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();

        // We can now trigger all vectors
        for i in 0..4 {
            msix_group.trigger(i).unwrap()
        }

        // We can't trigger an invalid vector
        msix_group.trigger(4).unwrap_err();
    }

    #[test]
    fn test_msi_vector_group_notifier() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();

        for i in 0..4 {
            assert!(msix_group.notifier(i).is_some());
        }

        assert!(msix_group.notifier(4).is_none());
    }

    #[test]
    fn test_msi_vector_group_update_invalid_vector() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();
        let sbdf = PciSBDF::from(0xafa);
        let config = MsixTableEntry {
            msg_addr_hi: 0x42,
            msg_addr_lo: 0x12,
            msg_data: 0x12,
            // Masked
            vector_ctl: 0x1,
        };
        msix_group.update(0, &config, sbdf).unwrap();
        assert_matches!(
            msix_group.update(4, &config, sbdf),
            Err(InterruptError::InvalidVectorIndex(4)),
        );
    }

    #[test]
    fn test_msi_vector_group_update() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);
        assert!(vm.common.interrupts.lock().unwrap().is_empty());
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();
        let pci_sbdf = PciSBDF::from(0xafa);

        // Set some configuration for the vectors. Initially all are masked
        let configs: [_; 4] = std::array::from_fn(|idx| MsixTableEntry {
            msg_addr_hi: 0x42,
            msg_addr_lo: 0x13,
            msg_data: 0x12 * u32::try_from(idx).unwrap(),
            vector_ctl: 0x1, // Masked
        });
        msix_group.update_batched(&configs, pci_sbdf).unwrap();

        // All vectors should be disabled
        for vector in &msix_group.vectors {
            assert!(!vector.enabled.load(Ordering::Acquire));
        }

        for i in 0..4 {
            let gsi = crate::arch::GSI_MSI_START + i;
            let interrupts = vm.common.interrupts.lock().unwrap();
            let kvm_route = interrupts.get(&gsi).unwrap();
            assert!(kvm_route.masked);
            assert_eq!(kvm_route.entry.gsi, gsi);
            assert_eq!(kvm_route.entry.type_, KVM_IRQ_ROUTING_MSI);
            // SAFETY: because we know we setup MSI routes.
            unsafe {
                assert_eq!(kvm_route.entry.u.msi.address_hi, 0x42);
                assert_eq!(kvm_route.entry.u.msi.address_lo, 0x13);
                assert_eq!(kvm_route.entry.u.msi.data, 0x12 * i);
            }
        }

        // Simply enabling the vectors should not update the registered IRQ routes
        enable_all_vectors(&msix_group);
        for i in 0..4 {
            let gsi = crate::arch::GSI_MSI_START + i;
            let interrupts = vm.common.interrupts.lock().unwrap();
            let kvm_route = interrupts.get(&gsi).unwrap();
            assert!(kvm_route.masked);
            assert_eq!(kvm_route.entry.gsi, gsi);
            assert_eq!(kvm_route.entry.type_, KVM_IRQ_ROUTING_MSI);
            // SAFETY: because we know we setup MSI routes.
            unsafe {
                assert_eq!(kvm_route.entry.u.msi.address_hi, 0x42);
                assert_eq!(kvm_route.entry.u.msi.address_lo, 0x13);
                assert_eq!(kvm_route.entry.u.msi.data, 0x12 * i);
            }
            assert!(
                msix_group.vectors[i as usize]
                    .enabled
                    .load(Ordering::Acquire)
            );
        }
        msix_group.disable().unwrap();

        let unmasked0 = MsixTableEntry {
            vector_ctl: 0x0, // un-masked
            ..configs[0].clone()
        };
        let unmasked1 = MsixTableEntry {
            vector_ctl: 0x0, // un-masked
            ..configs[1].clone()
        };
        // `update` and `register` will update the in-memory GSI config of its route (and only its route)
        msix_group.update(0, &unmasked0, pci_sbdf).unwrap();
        msix_group.register(1, &unmasked1, pci_sbdf).unwrap();
        for i in 0..4 {
            let gsi = crate::arch::GSI_MSI_START + i;
            let interrupts = vm.common.interrupts.lock().unwrap();
            let kvm_route = interrupts.get(&gsi).unwrap();
            assert_eq!(kvm_route.masked, i >= 2);
            assert_eq!(kvm_route.entry.gsi, gsi);
            assert_eq!(kvm_route.entry.type_, KVM_IRQ_ROUTING_MSI);
            // SAFETY: because we know we setup MSI routes.
            unsafe {
                assert_eq!(kvm_route.entry.u.msi.address_hi, 0x42);
                assert_eq!(kvm_route.entry.u.msi.address_lo, 0x13);
                assert_eq!(kvm_route.entry.u.msi.data, 0x12 * i);
            }
            // However, only `update` (first vector) will actually enable be enabled
            assert_eq!(
                msix_group.vectors[i as usize]
                    .enabled
                    .load(Ordering::Acquire),
                i == 0,
            );
        }
    }

    #[test]
    fn test_msi_vector_group_persistence() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();

        enable_all_vectors(&msix_group);
        let state = msix_group.save();

        // On the real restore path the allocator state omits MSI GSIs before devices replay
        // their allocations. Mimic that here so `restore` can re-claim the saved GSIs.
        let allocator_state = vm.resource_allocator().save();
        *vm.resource_allocator() = ResourceAllocator::restore((), &allocator_state).unwrap();
        let restored_group = MsixVectorGroup::restore(vm.clone(), &state).unwrap();

        assert_eq!(msix_group.num_vectors(), restored_group.num_vectors());
        // Even if an MSI group is enabled, we don't save it as such. During restoration, the PCI
        // transport will make sure the correct config is set for the vectors and enable them
        // accordingly.
        for (id, vector) in msix_group.vectors.iter().enumerate() {
            let new_vector = &restored_group.vectors[id];
            assert_eq!(vector.gsi, new_vector.gsi);
            assert!(!new_vector.enabled.load(Ordering::Acquire));
        }

        // Both groups own the same GSIs in this test so dropping both will result in panic. Resolve
        // this by simply forgetting about the restored version.
        std::mem::forget(restored_group);
    }

    #[test]
    fn test_msi_vector_group_restore_rejects_out_of_range_gsi() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);

        // A snapshot with a GSI outside the managed MSI range must be rejected rather than panic
        // when the GSI is later freed on drop.
        let tampered_state = vec![arch::GSI_MSI_END + 1];
        MsixVectorGroup::restore(vm.clone(), &tampered_state).unwrap_err();
    }

    #[test]
    fn test_msi_vector_group_drop_frees_gsis() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);

        let gsis_before = vm.resource_allocator().allocate_gsi_msi(1).unwrap();
        for id in gsis_before.iter() {
            vm.resource_allocator()
                .gsi_msi_allocator
                .free_id(*id)
                .unwrap();
        }

        // Allocating, configuring and dropping a group must leave the allocator and the routing
        // table in the same state as before.
        {
            let group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();
            let table_entry = MsixTableEntry {
                msg_addr_hi: 0x42,
                msg_addr_lo: 0x13,
                msg_data: 0x12,
                // Unmasked
                vector_ctl: 0x0,
            };
            let table_entries = vec![table_entry.clone(); 4];
            group
                .update_batched(&table_entries, PciSBDF::from(0xafa))
                .unwrap();
            assert_eq!(vm.common.interrupts.lock().unwrap().len(), 4);
        }

        assert!(vm.common.interrupts.lock().unwrap().is_empty());
        let gsis_after = vm.resource_allocator().allocate_gsi_msi(1).unwrap();
        assert_eq!(gsis_before, gsis_after);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_restore_state_resource_allocator() {
        use vm_allocator::AllocPolicy;

        let vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();

        // Allocate a GSI and some memory.
        let (gsi, range) = {
            let mut resource_allocator = vm.resource_allocator();

            let gsi = resource_allocator.allocate_gsi_msi(1).unwrap()[0];
            let range = resource_allocator
                .mmio32_memory
                .allocate(1024, 1024, AllocPolicy::FirstMatch)
                .unwrap();
            (gsi, range.start())
        };

        let state = vm.save_state().unwrap();
        let serialized_data = bitcode::serialize(&state).unwrap();

        let restored_state: VmState = bitcode::deserialize(&serialized_data).unwrap();
        vm.restore_state(&restored_state, false).unwrap();

        let mut resource_allocator = vm.resource_allocator();

        // The MSI GSI allocator is reset on restore (its allocations are replayed by the devices),
        // so the previously allocated GSI is free again and is the first one handed out.
        let gsi_new = resource_allocator.allocate_gsi_msi(1).unwrap()[0];
        assert_eq!(gsi, gsi_new);

        // Memory allocations, on the other hand, are restored from the serialized state, so the
        // range allocated before the snapshot is still reserved.
        resource_allocator
            .mmio32_memory
            .allocate(1024, 1024, AllocPolicy::ExactMatch(range))
            .unwrap_err();
        let range_new = resource_allocator
            .mmio32_memory
            .allocate(1024, 1024, AllocPolicy::FirstMatch)
            .unwrap();
        assert_eq!(range + 1024, range_new.start());
    }
}
