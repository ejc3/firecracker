// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use kvm_bindings::{KVM_CAP_ARM_WRITABLE_IMP_ID_REGS, KVMIO, kvm_enable_cap};
use serde::{Deserialize, Serialize};
use vmm_sys_util::errno;
use vmm_sys_util::ioctl::ioctl_with_ref;
use vmm_sys_util::ioctl_iow_nr;

use crate::Kvm;
use crate::arch::aarch64::gic::GicState;
use crate::logger::warn;
use crate::snapshot::Persist;
use crate::vstate::memory::{GuestMemoryExtension, GuestMemoryState};
use crate::vstate::resources::{ResourceAllocator, ResourceAllocatorState};
use crate::vstate::vm::{VmCommon, VmError};

// TODO(https://github.com/rust-vmm/kvm/pull/382): kvm-ioctls does not expose
// `VmFd::enable_cap` on aarch64 yet; this is the same definition it uses
// internally. Replace the direct ioctl with `enable_cap` once a release
// containing that PR is available.
#[allow(missing_docs)]
mod ioctls {
    use super::*;
    ioctl_iow_nr!(KVM_ENABLE_CAP, KVMIO, 0xa3, kvm_enable_cap);
}

/// Structure representing the current architecture's understand of what a "virtual machine" is.
#[derive(Debug)]
pub struct KvmVm {
    /// Architecture independent parts of a vm.
    pub common: VmCommon,
    // On aarch64 we need to keep around the fd obtained by creating the VGIC device.
    irqchip_handle: OnceLock<crate::arch::aarch64::gic::GICDevice>,
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
    /// Failed to set the VM counter offset (KVM_ARM_SET_COUNTER_OFFSET), errno: {0}
    SetCounterOffset(i32),
    /// Failed to restore resource allocator: {0}
    ResourceAllocator(#[from] vm_allocator::Error),
}

impl KvmVm {
    /// Create a new `KvmVm` struct.
    pub fn new(kvm: Kvm) -> Result<KvmVm, VmError> {
        let common = Self::create_common(kvm)?;

        // KVM gates writes to the implementation ID registers (MIDR_EL1,
        // REVIDR_EL1, AIDR_EL1) behind KVM_CAP_ARM_WRITABLE_IMP_ID_REGS,
        // which must be enabled before any vCPU is created. Without it, a
        // custom CPU template that modifies these registers fails at boot
        // with EINVAL when the template is applied. Enabling the capability
        // on its own does not change guest-visible state: the registers keep
        // their host values unless a template rewrites them, and writes of
        // unchanged values (e.g. on snapshot restore) were already accepted
        // before this capability existed.
        if common
            .fd
            .check_extension_raw(u64::from(KVM_CAP_ARM_WRITABLE_IMP_ID_REGS))
            == 1
        {
            let cap = kvm_enable_cap {
                cap: KVM_CAP_ARM_WRITABLE_IMP_ID_REGS,
                ..Default::default()
            };
            // SAFETY: The ioctl is safe because we allocated the struct and
            // the kernel will only read the size of the struct.
            let ret = unsafe { ioctl_with_ref(&common.fd, ioctls::KVM_ENABLE_CAP(), &cap) };
            if ret != 0 {
                // Not fatal: a VM whose CPU template does not touch the
                // implementation ID registers is unaffected, and one that
                // does will fail loudly when the template is applied.
                warn!(
                    "Failed to enable KVM_CAP_ARM_WRITABLE_IMP_ID_REGS: {}",
                    errno::Error::last()
                );
            }
        }

        Ok(KvmVm {
            common,
            irqchip_handle: OnceLock::new(),
        })
    }

    /// Pre-vCPU creation setup.
    pub fn arch_pre_create_vcpus(&self, _: u8) -> Result<(), KvmVmError> {
        Ok(())
    }

    /// Post-vCPU creation setup.
    pub fn arch_post_create_vcpus(&self, nr_vcpus: u8) -> Result<(), KvmVmError> {
        // On aarch64, the vCPUs need to be created (i.e call KVM_CREATE_VCPU) before setting up the
        // IRQ chip because the `KVM_CREATE_VCPU` ioctl will return error if the IRQCHIP
        // was already initialized.
        // Search for `kvm_arch_vcpu_create` in arch/arm/kvm/arm.c.
        self.setup_irqchip(nr_vcpus)
    }

    /// Creates the GIC (Global Interrupt Controller).
    pub fn setup_irqchip(&self, vcpu_count: u8) -> Result<(), KvmVmError> {
        let gic = crate::arch::aarch64::gic::create_gic(self.fd(), vcpu_count.into(), None)
            .map_err(KvmVmError::VmCreateGIC)?;
        // KVM refuses to create a second GIC, so create_gic() fails before this can be reached.
        self.irqchip_handle.set(gic).expect("IRQ chip already set");
        Ok(())
    }

    /// Gets a reference to the irqchip of the VM.
    pub fn get_irqchip(&self) -> &crate::arch::aarch64::gic::GICDevice {
        self.irqchip_handle.get().expect("IRQ chip not set")
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
            guest_counter_at_pause: None,
        })
    }

    /// Sets the VM-wide guest counter offset with `KVM_ARM_SET_COUNTER_OFFSET`.
    ///
    /// The guest's CNTPCT and CNTVCT then read `host_counter - offset`. Owning this offset is
    /// what lets the VMM move the guest's counter: without it the counter runs while the vCPUs
    /// are out of `KVM_RUN`, so a VM resumed after a pause has its clock ahead by the pause and
    /// every timer that came due in that time fires at once. The ioctl also sets KVM's
    /// `KVM_ARCH_FLAG_VM_COUNTER_OFFSET`, after which a `KVM_SET_ONE_REG` of CNTVCT or CNTPCT
    /// moves nothing, so a later register replay leaves the offset as set here.
    ///
    /// No vCPU may be running: KVM takes every vCPU's lock.
    pub fn set_counter_offset(&self, offset: u64) -> Result<(), KvmVmError> {
        #[repr(C)]
        struct KvmArmCounterOffset {
            counter_offset: u64,
            reserved: u64,
        }
        // _IOW(KVMIO = 0xAE, 0xb5, struct kvm_arm_counter_offset), 16 bytes.
        const KVM_ARM_SET_COUNTER_OFFSET: u64 = 0x4010_aeb5;
        let arg = KvmArmCounterOffset {
            counter_offset: offset,
            reserved: 0,
        };
        // SAFETY: self.fd() is a valid KVM VM fd and `arg` lives across the call.
        let ret = unsafe {
            libc::ioctl(
                std::os::fd::AsRawFd::as_raw_fd(self.fd()),
                KVM_ARM_SET_COUNTER_OFFSET as _,
                &arg,
            )
        };
        if ret < 0 {
            return Err(KvmVmError::SetCounterOffset(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            ));
        }
        Ok(())
    }

    /// Reads the host's generic timer counter (CNTVCT_EL0).
    ///
    /// On a VHE host a process's virtual counter offset is zero, so this is the counter
    /// KVM's `kvm_phys_timer_read()` reads.
    pub fn host_counter() -> u64 {
        let cnt: u64;
        // SAFETY: reading the virtual counter from EL0 is always permitted.
        unsafe {
            core::arch::asm!("isb", "mrs {cnt}, cntvct_el0", cnt = out(reg) cnt);
        }
        cnt
    }

    /// Restore the KVM VM state
    ///
    /// # Errors
    ///
    /// When [`crate::arch::aarch64::gic::GICDevice::restore_device`] errors.
    pub fn restore_state(&self, mpidrs: &[u64], state: &VmState) -> Result<(), KvmVmError> {
        self.get_irqchip()
            .restore_device(mpidrs, &state.gic)
            .map_err(KvmVmError::RestoreGic)?;
        *self.resource_allocator() = ResourceAllocator::restore((), &state.resource_allocator)?;

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
    /// The guest's counter when the VM was paused for this save, when the VMM owned the
    /// counter offset: the load continues the counter from here. A vCPU's saved CNTPCT is
    /// read at the save and is ahead of this by the time between the pause and the save.
    pub guest_counter_at_pause: Option<u64>,
}
