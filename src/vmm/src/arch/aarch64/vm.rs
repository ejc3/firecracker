// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::Kvm;
use crate::arch::aarch64::counter::{
    CounterController, CounterError, KvmCounterIo,
    normalize_snapshot_counters as normalize_saved_snapshot_counters,
};
use crate::arch::aarch64::gic::GicState;
use crate::snapshot::Persist;
use crate::vstate::memory::{GuestMemoryExtension, GuestMemoryState};
use crate::vstate::resources::{ResourceAllocator, ResourceAllocatorState};
use crate::vstate::vcpu::VcpuState;
use crate::vstate::vm::{VmCommon, VmError};

/// Structure representing the current architecture's understand of what a "virtual machine" is.
#[derive(Debug)]
pub struct KvmVm {
    /// Architecture independent parts of a vm.
    pub common: VmCommon,
    // On aarch64 we need to keep around the fd obtained by creating the VGIC device.
    irqchip_handle: Option<crate::arch::aarch64::gic::GICDevice>,
    // Runtime-only owner of the VM-wide generic-counter domain.
    counter_controller: Mutex<CounterController>,
}

/// Error type for [`KvmVm::restore_state`]
#[derive(Debug, PartialEq, Eq, thiserror::Error, displaydoc::Display)]
pub enum KvmVmError {
    /// Error creating the global interrupt controller: {0}
    VmCreateGIC(crate::arch::aarch64::gic::GicError),
    /// Failed to save the VM's GIC state: {0}
    SaveGic(crate::arch::aarch64::gic::GicError),
    /// Failed to restore the VM's GIC state: {0}
    RestoreGic(crate::arch::aarch64::gic::GicError),
    /// Failed to restore resource allocator: {0}
    ResourceAllocator(#[from] vm_allocator::Error),
}

impl KvmVm {
    /// Create a new `KvmVm` struct.
    pub fn new(kvm: Kvm) -> Result<KvmVm, VmError> {
        let common = Self::create_common(kvm)?;
        Ok(KvmVm {
            common,
            irqchip_handle: None,
            counter_controller: Mutex::new(CounterController::default()),
        })
    }

    fn counter_io<'a>(&'a self, vcpu_fd: &'a kvm_ioctls::VcpuFd) -> KvmCounterIo<'a> {
        KvmCounterIo::new(self.fd(), vcpu_fd)
    }

    /// Establish a zero-based counter domain before a fresh guest boots.
    pub(crate) fn configure_counter_for_boot(
        &self,
        vcpu_fd: &kvm_ioctls::VcpuFd,
    ) -> Result<(), CounterError> {
        let supported = self
            .kvm()
            .supports_counter_offset()
            .map_err(CounterError::CheckCapability)?;
        let io = self.counter_io(vcpu_fd);
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .configure_boot(&io, supported)
    }

    /// Continue a snapshot's counter domain before any vCPU register replay.
    pub(crate) fn configure_counter_for_restore(
        &self,
        vcpu_fd: &kvm_ioctls::VcpuFd,
        saved_counter: u64,
    ) -> Result<(), CounterError> {
        let supported = self
            .kvm()
            .supports_counter_offset()
            .map_err(CounterError::CheckCapability)?;
        let io = self.counter_io(vcpu_fd);
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .configure_restore(&io, saved_counter, supported)
    }

    /// Reject every lifecycle request after an unrecoverable transition.
    pub(crate) fn ensure_counter_healthy(&self) -> Result<(), CounterError> {
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .ensure_healthy()
    }

    /// Validate that a running VM can enter the counter pause lifecycle.
    pub(crate) fn ensure_counter_can_pause(&self) -> Result<(), CounterError> {
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .ensure_can_pause()
    }

    /// Return the physical counter captured at the exact vCPU pause point.
    ///
    /// Calling this before collecting device or KVM state also rejects
    /// snapshots from unsupported, running, uninitialized, or faulted domains.
    pub(crate) fn paused_counter_for_snapshot(&self) -> Result<u64, CounterError> {
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .paused_counter()
    }

    /// Move serialized counter samples back to the exact vCPU pause point.
    pub(crate) fn normalize_snapshot_counters(
        &self,
        states: &mut [VcpuState],
        paused_counter: u64,
    ) -> Result<(), CounterError> {
        let current_counter = self
            .counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .paused_counter()?;
        if current_counter != paused_counter {
            return Err(CounterError::SnapshotPausePointChanged(
                paused_counter,
                current_counter,
            ));
        }
        normalize_saved_snapshot_counters(states, paused_counter)
    }

    /// Capture the stable guest counter after every vCPU acknowledges pause.
    pub(crate) fn record_counter_pause(&self) -> Result<(), CounterError> {
        let handles = self.vcpus_handles();
        let vcpu_fd = &handles.first().ok_or(CounterError::NoBootVcpu)?.vcpu_fd;
        // Capability detection completed before seccomp was installed. Runtime
        // transitions use only the lifecycle state; they never issue
        // KVM_CHECK_EXTENSION through the active VMM seccomp filter.
        let io = self.counter_io(vcpu_fd);
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .record_pause(&io)
    }

    /// Advance the offset while all vCPUs remain paused.
    pub(crate) fn prepare_counter_resume(&self) -> Result<(), CounterError> {
        let handles = self.vcpus_handles();
        let vcpu_fd = &handles.first().ok_or(CounterError::NoBootVcpu)?.vcpu_fd;
        // Avoid KVM_CHECK_EXTENSION here: the VMM seccomp filter is already
        // active, and lifecycle state already records capability support.
        let io = self.counter_io(vcpu_fd);
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .prepare_resume(&io)
    }

    /// Complete a successful vCPU resume transition.
    pub(crate) fn mark_counter_running(&self) -> Result<(), CounterError> {
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .mark_running()
    }

    /// Fail closed after a counter capture could not be rolled back.
    pub(crate) fn mark_counter_faulted(&self) {
        self.counter_controller
            .lock()
            .expect("Poisoned counter controller")
            .mark_faulted();
    }

    /// Pre-vCPU creation setup.
    pub fn arch_pre_create_vcpus(&mut self, _: u8) -> Result<(), KvmVmError> {
        Ok(())
    }

    /// Post-vCPU creation setup.
    pub fn arch_post_create_vcpus(&mut self, nr_vcpus: u8) -> Result<(), KvmVmError> {
        // On aarch64, the vCPUs need to be created (i.e call KVM_CREATE_VCPU) before setting up the
        // IRQ chip because the `KVM_CREATE_VCPU` ioctl will return error if the IRQCHIP
        // was already initialized.
        // Search for `kvm_arch_vcpu_create` in arch/arm/kvm/arm.c.
        self.setup_irqchip(nr_vcpus)
    }

    /// Creates the GIC (Global Interrupt Controller).
    pub fn setup_irqchip(&mut self, vcpu_count: u8) -> Result<(), KvmVmError> {
        self.irqchip_handle = Some(
            crate::arch::aarch64::gic::create_gic(self.fd(), vcpu_count.into(), None)
                .map_err(KvmVmError::VmCreateGIC)?,
        );
        Ok(())
    }

    /// Gets a reference to the irqchip of the VM.
    pub fn get_irqchip(&self) -> &crate::arch::aarch64::gic::GICDevice {
        self.irqchip_handle.as_ref().expect("IRQ chip not set")
    }

    /// Saves and returns the KVM VM state.
    pub fn save_state(&self, mpidrs: &[u64]) -> Result<VmState, KvmVmError> {
        Ok(VmState {
            memory: self.common.guest_memory.describe(),
            gic: self
                .get_irqchip()
                .save_device(mpidrs)
                .map_err(KvmVmError::SaveGic)?,
            resource_allocator: self.resource_allocator().save(),
        })
    }

    /// Restore the KVM VM state
    ///
    /// # Errors
    ///
    /// When [`crate::arch::aarch64::gic::GICDevice::restore_device`] errors.
    pub fn restore_state(&mut self, mpidrs: &[u64], state: &VmState) -> Result<(), KvmVmError> {
        self.get_irqchip()
            .restore_device(mpidrs, &state.gic)
            .map_err(KvmVmError::RestoreGic)?;
        self.common.resource_allocator =
            Mutex::new(ResourceAllocator::restore((), &state.resource_allocator)?);

        Ok(())
    }
}

/// Structure holding an general specific VM state.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VmState {
    /// Guest memory state
    pub memory: GuestMemoryState,
    /// GIC state.
    pub gic: GicState,
    /// resource allocator
    pub resource_allocator: ResourceAllocatorState,
}
