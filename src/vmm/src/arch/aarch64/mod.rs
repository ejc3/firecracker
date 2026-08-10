// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod cache_info;
pub(crate) mod counter;
mod fdt;
/// Module for the global interrupt controller configuration.
pub mod gic;
/// Architecture specific KVM-related code
pub mod kvm;
/// Layout for this aarch64 system.
pub mod layout;
/// Logic for configuring aarch64 registers.
pub mod regs;
/// Architecture specific vCPU code
pub mod vcpu;
/// Architecture specific VM state code
pub mod vm;

/// Errors from the VM-wide Arm generic-counter lifecycle.
pub use counter::CounterError;

use std::cmp::min;
use std::fmt::Debug;
use std::fs::File;

use kvm_bindings::{KVM_ARM_VCPU_HAS_EL2, KVM_ARM_VCPU_HAS_EL2_E2H0, KVM_ARM_VCPU_POWER_OFF};
use linux_loader::loader::pe::PE as Loader;
use linux_loader::loader::{Cmdline, KernelLoader};
use vm_memory::{GuestMemoryBackend, GuestMemoryError, GuestMemoryRegion};

use crate::arch::{BootProtocol, EntryPoint, arch_memory_regions_with_gap};
use crate::cpu_config::aarch64::custom_cpu_template::VcpuFeatures;
use crate::cpu_config::aarch64::{CpuConfiguration, CpuConfigurationError};
use crate::cpu_config::templates::{CustomCpuTemplate, RegisterValueFilter};
use crate::initrd::InitrdConfig;
use zerocopy::IntoBytes;

use crate::logger::warn;
use crate::utils::{u64_to_usize, usize_to_u64};
use crate::vmm_config::machine_config::MachineConfig;
use crate::vstate::memory::{Address, Bytes, GuestAddress, GuestMemoryMmap, GuestRegionType};
use crate::vstate::vcpu::KvmVcpuError;
use crate::vstate::vm::KvmVm;
use crate::{DeviceManager, Kvm, Vcpu, VcpuConfig, align_up, logger};

/// Errors thrown while configuring aarch64 system.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum ConfigurationError {
    /// Failed to create a Flattened Device Tree for this aarch64 microVM: {0}
    SetupFDT(#[from] fdt::FdtError),
    /// Failed to write to guest memory.
    MemoryError(#[from] GuestMemoryError),
    /// Cannot copy kernel file fd
    KernelFile,
    /// Cannot load kernel due to invalid memory configuration or invalid kernel image: {0}
    KernelLoader(#[from] linux_loader::loader::Error),
    /// Error creating vcpu configuration: {0}
    VcpuConfig(#[from] CpuConfigurationError),
    /// Error configuring the vcpu: {0}
    VcpuConfigure(#[from] KvmVcpuError),
    /// Failed to read host cache information: {0}
    CacheInfo(#[from] cache_info::CacheInfoError),
    /// Failed to configure the VM-wide generic-counter domain: {0}
    Counter(#[from] CounterError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error, displaydoc::Display)]
enum ArmFeatureValidationError {
    /// No boot vCPU is available.
    NoBootVcpu,
    /// KVM_ARM_VCPU_HAS_EL2_E2H0 is unsupported; Arm nested virtualization requires VHE.
    UnsupportedE2h0,
    /// Effective Arm vCPU feature words disagree across vCPUs.
    InconsistentVcpuFeatures,
}

/// Exception level and host-extension mode used to boot an Arm guest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArmBootMode {
    /// Boot the guest kernel at EL1.
    El1,
    /// Boot the guest kernel at EL2 with VHE (HCR_EL2.E2H = 1).
    El2Vhe,
}

/// Returns a Vec of the valid memory addresses for aarch64.
/// See [`layout`](layout) module for a drawing of the specific memory model for this platform.
pub fn arch_memory_regions(size: usize) -> Vec<(GuestAddress, usize)> {
    assert!(size > 0, "Attempt to allocate guest memory of length 0");

    let dram_size = min(size, layout::DRAM_MEM_MAX_SIZE);

    if dram_size != size {
        logger::warn!(
            "Requested memory size {} exceeds architectural maximum (1022GiB). Size has been \
             truncated to {}",
            size,
            dram_size
        );
    }

    let mut regions = vec![];
    if let Some((offset, remaining)) = arch_memory_regions_with_gap(
        &mut regions,
        u64_to_usize(layout::DRAM_MEM_START),
        dram_size,
        u64_to_usize(layout::MMIO64_MEM_START),
        u64_to_usize(layout::MMIO64_MEM_SIZE),
    ) {
        regions.push((GuestAddress(offset as u64), remaining));
    }

    regions
}

pub(crate) fn cpu_template_with_nv2(
    cpu_template: &CustomCpuTemplate,
    nv2_enabled: bool,
) -> CustomCpuTemplate {
    let mut template = cpu_template.clone();

    if nv2_enabled {
        let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;
        let has_el2_e2h0 = 1 << KVM_ARM_VCPU_HAS_EL2_E2H0;
        template.vcpu_features.push(VcpuFeatures {
            index: 0,
            bitmap: RegisterValueFilter {
                filter: has_el2 | has_el2_e2h0,
                value: has_el2,
            },
        });
    }

    template
}

fn effective_vcpu_feature_word(initial_feature_word: u32, cpu_template: &CustomCpuTemplate) -> u32 {
    cpu_template
        .vcpu_features
        .iter()
        .filter(|feature| feature.index == 0)
        .fold(initial_feature_word, |word, feature| {
            feature.bitmap.apply(word)
        })
}

fn validate_arm_vcpu_init_feature_word(feature_word: u32) -> Result<(), ArmFeatureValidationError> {
    if feature_word & (1 << KVM_ARM_VCPU_HAS_EL2_E2H0) != 0 {
        Err(ArmFeatureValidationError::UnsupportedE2h0)
    } else {
        Ok(())
    }
}

fn arm_boot_mode(feature_word: u32) -> Result<ArmBootMode, ArmFeatureValidationError> {
    let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;
    validate_arm_vcpu_init_feature_word(feature_word)?;

    Ok(if feature_word & has_el2 != 0 {
        ArmBootMode::El2Vhe
    } else {
        ArmBootMode::El1
    })
}

pub(crate) fn validate_restored_vcpu_feature_words(
    feature_words: &[u32],
) -> Result<(), ConfigurationError> {
    // Saving strips POWER_OFF from every vCPU KVI, so restored feature words
    // must agree exactly.
    validate_vcpu_feature_words(feature_words, 0)
        .map(|_| ())
        .map_err(configuration_error_from_feature_validation)
}

fn validate_vcpu_feature_words(
    feature_words: &[u32],
    allowed_difference_mask: u32,
) -> Result<ArmBootMode, ArmFeatureValidationError> {
    let (boot_feature_word, secondary_feature_words) = feature_words
        .split_first()
        .ok_or(ArmFeatureValidationError::NoBootVcpu)?;
    let boot_mode = arm_boot_mode(*boot_feature_word)?;
    let canonical_boot_word = boot_feature_word & !allowed_difference_mask;

    for feature_word in secondary_feature_words {
        arm_boot_mode(*feature_word)?;
        if feature_word & !allowed_difference_mask != canonical_boot_word {
            return Err(ArmFeatureValidationError::InconsistentVcpuFeatures);
        }
    }

    Ok(boot_mode)
}

fn resolve_arm_boot_mode(
    cpu_template: &CustomCpuTemplate,
    initial_feature_words: &[u32],
) -> Result<ArmBootMode, ArmFeatureValidationError> {
    let effective_feature_words = initial_feature_words
        .iter()
        .map(|word| effective_vcpu_feature_word(*word, cpu_template))
        .collect::<Vec<_>>();

    // Secondary vCPUs intentionally carry POWER_OFF during boot. Every other
    // effective initialization feature must agree before any vCPU is passed to
    // KVM_ARM_VCPU_INIT, and every word must independently select supported VHE.
    validate_vcpu_feature_words(&effective_feature_words, 1 << KVM_ARM_VCPU_POWER_OFF)
}

fn configuration_error_from_feature_validation(
    error: ArmFeatureValidationError,
) -> ConfigurationError {
    warn!("Invalid Arm vCPU initialization features: {error}");
    match error {
        ArmFeatureValidationError::NoBootVcpu => {
            ConfigurationError::Counter(CounterError::NoBootVcpu)
        }
        ArmFeatureValidationError::UnsupportedE2h0
        | ArmFeatureValidationError::InconsistentVcpuFeatures => ConfigurationError::VcpuConfigure(
            KvmVcpuError::Init(kvm_ioctls::Error::new(libc::EINVAL)),
        ),
    }
}

/// Configures the system for booting Linux.
#[allow(clippy::too_many_arguments)]
pub fn configure_system_for_boot(
    _kvm: &Kvm,
    vm: &KvmVm,
    device_manager: &mut DeviceManager,
    vcpus: &mut [Vcpu],
    machine_config: &MachineConfig,
    cpu_template: &CustomCpuTemplate,
    entry_point: EntryPoint,
    initrd: &Option<InitrdConfig>,
    boot_cmdline: Cmdline,
) -> Result<(), ConfigurationError> {
    let initial_feature_words = vcpus
        .iter()
        .map(|vcpu| vcpu.kvm_vcpu.boot_feature_word())
        .collect::<Vec<_>>();
    let boot_mode = resolve_arm_boot_mode(cpu_template, &initial_feature_words)
        .map_err(configuration_error_from_feature_validation)?;

    // Construct the base CpuConfiguration to apply CPU template onto.
    let cpu_config = CpuConfiguration::new(cpu_template, vcpus)?;

    // All vCPUs are now initialized. Read KVM's exact physical-counter domain
    // from vCPU0 and install the VM-wide offset before any boot register is
    // written. This replaces the per-vCPU PTIMER SET_ONE_REG reset.
    let boot_vcpu = vcpus.first().ok_or(CounterError::NoBootVcpu)?;
    vm.configure_counter_for_boot(&boot_vcpu.kvm_vcpu.fd)?;

    // Apply CPU template to the base CpuConfiguration.
    let cpu_config = CpuConfiguration::apply_template(cpu_config, cpu_template);

    let vcpu_config = VcpuConfig {
        vcpu_count: machine_config.vcpu_count,
        smt: machine_config.smt,
        cpu_config,
    };

    // Configure vCPUs with normalizing and setting the generated CPU configuration.
    for vcpu in vcpus.iter_mut() {
        vcpu.kvm_vcpu
            .configure(vm.guest_memory(), entry_point, &vcpu_config)?;
    }

    // Override CLIDR_EL1 ctype/LoC fields on each vCPU to match the host's
    // real cache topology. See `override_clidr` for details.
    override_clidr(vcpus)?;

    let vcpu_mpidr = vcpus
        .iter_mut()
        .map(|cpu| cpu.kvm_vcpu.get_mpidr())
        .collect::<Result<Vec<_>, _>>()
        .map_err(KvmVcpuError::ConfigureRegisters)?;
    let cmdline = boot_cmdline
        .as_cstring()
        .expect("Cannot create cstring from cmdline string");

    // Enable SMC for PSCI when nested virtualization is enabled (HAS_EL2).
    // With nested virt, HVC traps to the guest's virtual EL2 which has no handler.
    // SMC goes to KVM's secure monitor emulation which handles PSCI correctly.
    let fdt = fdt::create_fdt(
        vm.guest_memory(),
        vcpu_mpidr,
        cmdline,
        device_manager,
        vm.get_irqchip(),
        initrd,
        boot_mode,
    )?;

    let fdt_address = GuestAddress(get_fdt_addr(vm.guest_memory()));
    vm.guest_memory().write_slice(fdt.as_slice(), fdt_address)?;

    Ok(())
}

/// Override CLIDR_EL1 ctype/LoC fields on each vCPU to match the host's real
/// cache topology.
///
/// Since host kernel 6.3 (commit 7af0c2534f4c), KVM fabricates CLIDR_EL1
/// instead of passing through the host's real value. This can cause the guest
/// to see fewer cache levels than actually exist. Guest kernels >= 6.1.156
/// backported `init_of_cache_level()` which counts cache leaves from the DT,
/// while `populate_cache_leaves()` uses CLIDR_EL1. If the DT (built from host
/// sysfs) describes different cache entries than CLIDR_EL1, the mismatch
/// causes cache sysfs entries to not be created.
///
/// We read the current (possibly fabricated) CLIDR_EL1, replace only the ctype
/// and LoC fields with values derived from sysfs, and preserve all other fields
/// (LoUU, LoUIS, ICB, Ttype). This is safe on pre-6.3 kernels where CLIDR
/// already matches sysfs — the write is skipped as a no-op.
fn override_clidr(vcpus: &[Vcpu]) -> Result<(), ConfigurationError> {
    let mut l1_caches = Vec::new();
    let mut non_l1_caches = Vec::new();
    cache_info::read_cache_config(&mut l1_caches, &mut non_l1_caches)?;

    // If sysfs reports no L1 caches, we cannot build a meaningful CLIDR.
    // Writing an all-zero CLIDR would tell the guest there are no caches,
    // which is worse than whatever KVM fabricated. Leave it alone.
    if l1_caches.is_empty() {
        warn!("No L1 caches found in sysfs, skipping CLIDR override");
        return Ok(());
    }

    let sysfs_clidr = cache_info::build_clidr_from_caches(&l1_caches, &non_l1_caches);

    let mut cur_clidr: u64 = 0;
    // Reading/writing CLIDR_EL1 via KVM_SET_ONE_REG may not be supported on
    // older kernels (pre-6.3). In that case KVM passes through the real host
    // CLIDR and the override is unnecessary, so we warn and continue.
    if let Err(e) = vcpus[0]
        .kvm_vcpu
        .fd
        .get_one_reg(regs::CLIDR_EL1, cur_clidr.as_mut_bytes())
    {
        warn!("Failed to read CLIDR_EL1, skipping override: {e}");
        return Ok(());
    }

    let new_clidr = cache_info::merge_clidr(cur_clidr, sysfs_clidr);

    if new_clidr != cur_clidr {
        for vcpu in vcpus.iter() {
            if let Err(e) = vcpu
                .kvm_vcpu
                .fd
                .set_one_reg(regs::CLIDR_EL1, new_clidr.as_bytes())
            {
                warn!(
                    "Failed to set CLIDR_EL1 to {:#x} on vCPU {}, skipping override: {e}",
                    new_clidr, vcpu.kvm_vcpu.index
                );
                return Ok(());
            }
        }
    }

    Ok(())
}

/// Returns the memory address where the kernel could be loaded.
pub fn get_kernel_start() -> u64 {
    layout::SYSTEM_MEM_START + layout::SYSTEM_MEM_SIZE
}

/// Returns the memory address where the initrd could be loaded.
pub fn initrd_load_addr(guest_mem: &GuestMemoryMmap, initrd_size: usize) -> Option<u64> {
    let rounded_size = align_up!(
        usize_to_u64(initrd_size),
        usize_to_u64(super::GUEST_PAGE_SIZE)
    );
    GuestAddress(get_fdt_addr(guest_mem))
        .checked_sub(rounded_size)
        .filter(|&addr| guest_mem.address_in_range(addr))
        .map(|addr| addr.raw_value())
}

// Auxiliary function to get the address where the device tree blob is loaded.
fn get_fdt_addr(mem: &GuestMemoryMmap) -> u64 {
    // Find the first (and only) DRAM region.
    let dram_region = mem
        .iter()
        .find(|region| region.region_type == GuestRegionType::Dram)
        .unwrap();

    // If the memory allocated is smaller than the size allocated for the FDT,
    // we return the start of the DRAM so that
    // we allow the code to try and load the FDT.
    dram_region
        .last_addr()
        .checked_sub(layout::FDT_MAX_SIZE as u64 - 1)
        .filter(|&addr| mem.address_in_range(addr))
        .map(|addr| addr.raw_value())
        .unwrap_or(layout::DRAM_MEM_START)
}

/// Load linux kernel into guest memory.
pub fn load_kernel(
    kernel: &File,
    guest_memory: &GuestMemoryMmap,
) -> Result<EntryPoint, ConfigurationError> {
    // Need to clone the File because reading from it
    // mutates it.
    let mut kernel_file = kernel
        .try_clone()
        .map_err(|_| ConfigurationError::KernelFile)?;

    let entry_addr = Loader::load(
        guest_memory,
        Some(GuestAddress(get_kernel_start())),
        &mut kernel_file,
        None,
    )?;

    Ok(EntryPoint {
        entry_addr: entry_addr.kernel_load,
        protocol: BootProtocol::LinuxBoot,
    })
}

#[cfg(kani)]
mod verification {
    use crate::arch::aarch64::layout::{
        DRAM_MEM_MAX_SIZE, DRAM_MEM_START, FIRST_ADDR_PAST_64BITS_MMIO, MMIO64_MEM_START,
    };
    use crate::arch::arch_memory_regions;

    #[kani::proof]
    #[kani::unwind(3)]
    fn verify_arch_memory_regions() {
        let len: usize = kani::any::<usize>();
        kani::assume(len > 0);

        let regions = arch_memory_regions(len);

        for region in &regions {
            println!(
                "region: [{:x}:{:x})",
                region.0.0,
                region.0.0 + region.1 as u64
            );
        }

        // On Arm we have one MMIO gap that might fall within addressable ranges,
        // so we can get either 1 or 2 regions.
        assert!(regions.len() >= 1);
        assert!(regions.len() <= 2);

        // The total length of all regions cannot exceed DRAM_MEM_MAX_SIZE
        let actual_len = regions.iter().map(|&(_, len)| len).sum::<usize>();
        assert!(actual_len <= DRAM_MEM_MAX_SIZE);
        // The total length is smaller or equal to the length we asked
        assert!(actual_len <= len);
        // If it's smaller, it's because we asked more than the the maximum possible.
        if (actual_len) < len {
            assert!(len > DRAM_MEM_MAX_SIZE);
        }

        // No region overlaps the 64-bit MMIO gap
        assert!(
            regions
                .iter()
                .all(|&(start, len)| start.0 >= FIRST_ADDR_PAST_64BITS_MMIO
                    || start.0 + len as u64 <= MMIO64_MEM_START)
        );

        // All regions start after our DRAM_MEM_START
        assert!(regions.iter().all(|&(start, _)| start.0 >= DRAM_MEM_START));

        // All regions have non-zero length
        assert!(regions.iter().all(|&(_, len)| len > 0));

        // If there's two regions, they perfectly snuggle up the 64bit MMIO gap
        if regions.len() == 2 {
            kani::cover!();

            // The very first address should be DRAM_MEM_START
            assert_eq!(regions[0].0.0, DRAM_MEM_START);
            // The first region ends at the beginning of the 64 bits gap.
            assert_eq!(regions[0].0.0 + regions[0].1 as u64, MMIO64_MEM_START);
            // The second region starts exactly after the 64 bits gap.
            assert_eq!(regions[1].0.0, FIRST_ADDR_PAST_64BITS_MMIO);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::aarch64::layout::{
        DRAM_MEM_MAX_SIZE, DRAM_MEM_START, FDT_MAX_SIZE, FIRST_ADDR_PAST_64BITS_MMIO,
        MMIO64_MEM_START,
    };
    use crate::test_utils::arch_mem;

    #[test]
    fn test_cpu_template_with_nv2_selects_vhe() {
        let original = CustomCpuTemplate::default();

        assert_eq!(cpu_template_with_nv2(&original, false), original);

        let enabled = cpu_template_with_nv2(&original, true);
        assert_eq!(original, CustomCpuTemplate::default());
        assert_eq!(enabled.vcpu_features.len(), 1);

        let nv2 = &enabled.vcpu_features[0];
        let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;
        let has_el2_e2h0 = 1 << KVM_ARM_VCPU_HAS_EL2_E2H0;
        assert_eq!(nv2.index, 0);
        assert_eq!(nv2.bitmap.filter, has_el2 | has_el2_e2h0);
        assert_eq!(nv2.bitmap.value, has_el2);
        assert_eq!(nv2.bitmap.value & has_el2_e2h0, 0);
    }

    fn feature_modifier(filter: u32, value: u32) -> VcpuFeatures {
        VcpuFeatures {
            index: 0,
            bitmap: RegisterValueFilter { filter, value },
        }
    }

    #[test]
    fn test_effective_has_el2_drives_boot_mode() {
        let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;
        let cpu_template = CustomCpuTemplate {
            vcpu_features: vec![feature_modifier(has_el2, has_el2)],
            ..Default::default()
        };

        let boot_mode = resolve_arm_boot_mode(&cpu_template, &[0]).unwrap();

        assert_eq!(boot_mode, ArmBootMode::El2Vhe);
    }

    #[test]
    fn test_e2h0_boot_mode_is_rejected() {
        let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;
        let has_el2_e2h0 = 1 << KVM_ARM_VCPU_HAS_EL2_E2H0;
        let cpu_template = CustomCpuTemplate {
            vcpu_features: vec![feature_modifier(
                has_el2 | has_el2_e2h0,
                has_el2 | has_el2_e2h0,
            )],
            ..Default::default()
        };

        assert!(matches!(
            resolve_arm_boot_mode(&cpu_template, &[0]),
            Err(ArmFeatureValidationError::UnsupportedE2h0)
        ));
    }

    #[test]
    fn test_primary_initial_e2h0_is_rejected() {
        let has_el2_e2h0 = 1 << KVM_ARM_VCPU_HAS_EL2_E2H0;

        assert!(matches!(
            resolve_arm_boot_mode(&CustomCpuTemplate::default(), &[has_el2_e2h0],),
            Err(ArmFeatureValidationError::UnsupportedE2h0)
        ));
    }

    #[test]
    fn test_template_can_clear_initial_has_el2() {
        let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;
        let cpu_template = CustomCpuTemplate {
            vcpu_features: vec![feature_modifier(has_el2, 0)],
            ..Default::default()
        };

        let boot_mode = resolve_arm_boot_mode(&cpu_template, &[has_el2]).unwrap();

        assert_eq!(boot_mode, ArmBootMode::El1);
    }

    #[test]
    fn test_cli_nv2_modifier_is_last_and_selects_vhe() {
        let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;
        let has_el2_e2h0 = 1 << KVM_ARM_VCPU_HAS_EL2_E2H0;
        let cpu_template = CustomCpuTemplate {
            vcpu_features: vec![feature_modifier(
                has_el2 | has_el2_e2h0,
                has_el2 | has_el2_e2h0,
            )],
            ..Default::default()
        };

        let effective_template = cpu_template_with_nv2(&cpu_template, true);
        let boot_mode = resolve_arm_boot_mode(&effective_template, &[0]).unwrap();
        let effective_word = effective_vcpu_feature_word(0, &effective_template);

        assert_eq!(boot_mode, ArmBootMode::El2Vhe);
        assert_ne!(effective_word & has_el2, 0);
        assert_eq!(effective_word & has_el2_e2h0, 0);
    }

    #[test]
    fn test_secondary_e2h0_is_rejected_before_vcpu_init() {
        let has_el2_e2h0 = 1 << KVM_ARM_VCPU_HAS_EL2_E2H0;
        let power_off = 1 << KVM_ARM_VCPU_POWER_OFF;

        assert!(matches!(
            resolve_arm_boot_mode(
                &CustomCpuTemplate::default(),
                &[0, power_off | has_el2_e2h0],
            ),
            Err(ArmFeatureValidationError::UnsupportedE2h0)
        ));
    }

    #[test]
    fn test_divergent_vcpu_features_are_rejected_before_vcpu_init() {
        let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;
        let power_off = 1 << KVM_ARM_VCPU_POWER_OFF;

        assert!(matches!(
            resolve_arm_boot_mode(&CustomCpuTemplate::default(), &[has_el2, power_off],),
            Err(ArmFeatureValidationError::InconsistentVcpuFeatures)
        ));
    }

    #[test]
    fn test_secondary_power_off_is_an_allowed_feature_difference() {
        let power_off = 1 << KVM_ARM_VCPU_POWER_OFF;

        let boot_mode =
            resolve_arm_boot_mode(&CustomCpuTemplate::default(), &[0, power_off]).unwrap();

        assert_eq!(boot_mode, ArmBootMode::El1);
    }

    #[test]
    fn test_restore_rejects_secondary_e2h0_before_any_vcpu_init() {
        let has_el2_e2h0 = 1 << KVM_ARM_VCPU_HAS_EL2_E2H0;

        assert!(matches!(
            validate_vcpu_feature_words(&[0, has_el2_e2h0], 0),
            Err(ArmFeatureValidationError::UnsupportedE2h0)
        ));
    }

    #[test]
    fn test_restore_rejects_divergent_features_before_any_vcpu_init() {
        let has_el2 = 1 << KVM_ARM_VCPU_HAS_EL2;

        assert!(matches!(
            validate_vcpu_feature_words(&[has_el2, 0], 0),
            Err(ArmFeatureValidationError::InconsistentVcpuFeatures)
        ));
    }

    #[test]
    fn test_regions_lt_1024gb() {
        let regions = arch_memory_regions(1usize << 29);
        assert_eq!(1, regions.len());
        assert_eq!(GuestAddress(DRAM_MEM_START), regions[0].0);
        assert_eq!(1usize << 29, regions[0].1);
    }

    #[test]
    fn test_regions_gt_1024gb() {
        let regions = arch_memory_regions(1usize << 41);
        assert_eq!(2, regions.len());
        assert_eq!(GuestAddress(DRAM_MEM_START), regions[0].0);
        assert_eq!(MMIO64_MEM_START - DRAM_MEM_START, regions[0].1 as u64);
        assert_eq!(GuestAddress(FIRST_ADDR_PAST_64BITS_MMIO), regions[1].0);
        assert_eq!(
            DRAM_MEM_MAX_SIZE as u64 - MMIO64_MEM_START + DRAM_MEM_START,
            regions[1].1 as u64
        );
    }

    #[test]
    fn test_get_fdt_addr() {
        let mem = arch_mem(FDT_MAX_SIZE - 0x1000);
        assert_eq!(get_fdt_addr(&mem), DRAM_MEM_START);

        let mem = arch_mem(FDT_MAX_SIZE);
        assert_eq!(get_fdt_addr(&mem), DRAM_MEM_START);

        let mem = arch_mem(FDT_MAX_SIZE + 0x1000);
        assert_eq!(get_fdt_addr(&mem), 0x1000 + DRAM_MEM_START);
    }
}
