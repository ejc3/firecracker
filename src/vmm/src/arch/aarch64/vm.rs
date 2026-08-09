// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::Kvm;
use crate::arch::aarch64::gic::GicState;
use crate::vstate::memory::{GuestMemoryExtension, GuestMemoryState};
use crate::vstate::resources::ResourceAllocator;
use crate::vstate::vm::{VmCommon, VmError};

/// Structure representing the current architecture's understand of what a "virtual machine" is.
#[derive(Debug)]
pub struct ArchVm {
    /// Architecture independent parts of a vm.
    pub common: VmCommon,
    // On aarch64 we need to keep around the fd obtained by creating the VGIC device.
    irqchip_handle: Option<crate::arch::aarch64::gic::GICDevice>,
}

/// Error type for [`Vm::restore_state`]
#[derive(Debug, PartialEq, Eq, thiserror::Error, displaydoc::Display)]
pub enum ArchVmError {
    /// Error creating the global interrupt controller: {0}
    VmCreateGIC(crate::arch::aarch64::gic::GicError),
    /// Failed to save the VM's GIC state: {0}
    SaveGic(crate::arch::aarch64::gic::GicError),
    /// Failed to restore the VM's GIC state: {0}
    RestoreGic(crate::arch::aarch64::gic::GicError),
    /// Failed to set the VM counter offset (KVM_ARM_SET_COUNTER_OFFSET), errno: {0}
    SetCounterOffset(i32),
}

impl ArchVm {
    /// Create a new `Vm` struct.
    pub fn new(kvm: &Kvm) -> Result<ArchVm, VmError> {
        let common = Self::create_common(kvm)?;
        Ok(ArchVm {
            common,
            irqchip_handle: None,
        })
    }

    /// Pre-vCPU creation setup.
    pub fn arch_pre_create_vcpus(&mut self, _: u8) -> Result<(), ArchVmError> {
        Ok(())
    }

    /// Post-vCPU creation setup.
    pub fn arch_post_create_vcpus(&mut self, nr_vcpus: u8) -> Result<(), ArchVmError> {
        // On aarch64, the vCPUs need to be created (i.e call KVM_CREATE_VCPU) before setting up the
        // IRQ chip because the `KVM_CREATE_VCPU` ioctl will return error if the IRQCHIP
        // was already initialized.
        // Search for `kvm_arch_vcpu_create` in arch/arm/kvm/arm.c.
        self.setup_irqchip(nr_vcpus)
    }

    /// Creates the GIC (Global Interrupt Controller).
    pub fn setup_irqchip(&mut self, vcpu_count: u8) -> Result<(), ArchVmError> {
        self.irqchip_handle = Some(
            crate::arch::aarch64::gic::create_gic(self.fd(), vcpu_count.into(), None)
                .map_err(ArchVmError::VmCreateGIC)?,
        );
        Ok(())
    }

    /// Gets a reference to the irqchip of the VM.
    pub fn get_irqchip(&self) -> &crate::arch::aarch64::gic::GICDevice {
        self.irqchip_handle.as_ref().expect("IRQ chip not set")
    }

    /// Set the VM-wide guest counter offset via `KVM_ARM_SET_COUNTER_OFFSET`.
    ///
    /// Applies `offset` to both the virtual and physical counter views, so the
    /// guest's CNTPCT/CNTVCT read `host_counter - offset`. Setting this before
    /// replaying saved counter registers prevents KVM from creating per-timer
    /// offsets that a later CNTVOFF_EL2 restore would overwrite.
    ///
    /// This must only be called while no vCPU is running because KVM takes all
    /// vCPU locks while changing the offset.
    pub fn set_counter_offset(&self, offset: u64) -> Result<(), ArchVmError> {
        #[repr(C)]
        struct KvmArmCounterOffset {
            counter_offset: u64,
            reserved: u64,
        }

        // _IOW(KVMIO = 0xAE, 0xb5, struct kvm_arm_counter_offset (16 bytes))
        const KVM_ARM_SET_COUNTER_OFFSET: libc::c_ulong = 0x4010_aeb5;
        let arg = KvmArmCounterOffset {
            counter_offset: offset,
            reserved: 0,
        };

        // SAFETY: self.fd() is a valid KVM VM fd and `arg` lives across the call.
        let ret = unsafe {
            libc::ioctl(
                std::os::fd::AsRawFd::as_raw_fd(self.fd()),
                KVM_ARM_SET_COUNTER_OFFSET,
                &arg,
            )
        };
        if ret < 0 {
            return Err(ArchVmError::SetCounterOffset(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            ));
        }

        Ok(())
    }

    /// Read the host's view of the generic timer counter (CNTVCT_EL0).
    pub fn host_counter() -> u64 {
        let counter: u64;
        // SAFETY: reading the virtual counter from userspace is permitted on aarch64.
        unsafe {
            core::arch::asm!("isb", "mrs {counter}, cntvct_el0", counter = out(reg) counter);
        }
        counter
    }

    /// Saves and returns the Kvm Vm state.
    pub fn save_state(&self, mpidrs: &[u64]) -> Result<VmState, ArchVmError> {
        Ok(VmState {
            memory: self.common.guest_memory.describe(),
            gic: self
                .get_irqchip()
                .save_device(mpidrs)
                .map_err(ArchVmError::SaveGic)?,
            resource_allocator: self.resource_allocator().clone(),
        })
    }

    /// Restore the KVM VM state
    ///
    /// # Errors
    ///
    /// When [`crate::arch::aarch64::gic::GICDevice::restore_device`] errors.
    pub fn restore_state(&mut self, mpidrs: &[u64], state: &VmState) -> Result<(), ArchVmError> {
        self.get_irqchip()
            .restore_device(mpidrs, &state.gic)
            .map_err(ArchVmError::RestoreGic)?;
        self.common.resource_allocator = Mutex::new(state.resource_allocator.clone());

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
    pub resource_allocator: ResourceAllocator,
}
