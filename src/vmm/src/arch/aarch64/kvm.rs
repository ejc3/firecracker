// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;

use kvm_ioctls::Kvm as KvmFd;
use vmm_sys_util::errno;

use crate::cpu_config::templates::KvmCapability;

/// ['Kvm'] initialization can't fail for Aarch64
pub type KvmArchError = Infallible;

/// Struct with kvm fd and kvm associated parameters.
#[derive(Debug)]
pub struct Kvm {
    /// KVM fd.
    pub fd: KvmFd,
    /// Additional capabilities that were specified in cpu template.
    pub kvm_cap_modifiers: Vec<KvmCapability>,
}

impl Kvm {
    pub(crate) const DEFAULT_CAPABILITIES: [u32; 7] = [
        kvm_bindings::KVM_CAP_IOEVENTFD,
        kvm_bindings::KVM_CAP_IRQFD,
        kvm_bindings::KVM_CAP_USER_MEMORY,
        kvm_bindings::KVM_CAP_ARM_PSCI_0_2,
        kvm_bindings::KVM_CAP_DEVICE_CTRL,
        kvm_bindings::KVM_CAP_MP_STATE,
        kvm_bindings::KVM_CAP_ONE_REG,
    ];

    /// Initialize [`Kvm`] type for Aarch64 architecture
    pub fn init_arch(
        fd: KvmFd,
        kvm_cap_modifiers: Vec<KvmCapability>,
    ) -> Result<Self, KvmArchError> {
        Ok(Self {
            fd,
            kvm_cap_modifiers,
        })
    }

    /// Reports whether KVM supports the VM-wide Arm counter-offset API.
    ///
    /// # Errors
    ///
    /// Returns the ioctl error instead of treating a failed capability query
    /// as either supported or unsupported.
    pub fn supports_counter_offset(&self) -> Result<bool, errno::Error> {
        checked_capability(
            self.fd
                .check_extension_raw(kvm_bindings::KVM_CAP_COUNTER_OFFSET.into()),
        )
    }
}

fn checked_capability(raw_result: i32) -> Result<bool, errno::Error> {
    if raw_result < 0 {
        Err(errno::Error::last())
    } else {
        Ok(raw_result != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::checked_capability;

    #[test]
    fn capability_result_distinguishes_unsupported_from_ioctl_failure() {
        assert!(!checked_capability(0).unwrap());
        assert!(checked_capability(1).unwrap());
        checked_capability(-1).unwrap_err();
    }
}
