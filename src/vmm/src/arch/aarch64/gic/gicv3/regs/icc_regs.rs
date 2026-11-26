// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use kvm_bindings::*;
use kvm_ioctls::DeviceFd;

use crate::arch::aarch64::gic::GicError;
use crate::arch::aarch64::gic::regs::{MmioReg, SimpleReg, VgicRegEngine, VgicSysRegsState};

const ICC_CTLR_EL1_PRIBITS_SHIFT: u64 = 8;
const ICC_CTLR_EL1_PRIBITS_MASK: u64 = 7 << ICC_CTLR_EL1_PRIBITS_SHIFT;

// These registers are taken from the kernel. Look for `gic_v3_icc_reg_descs`.
const SYS_ICC_SRE_EL1: SimpleReg = SimpleReg::vgic_sys_reg(3, 0, 12, 12, 5);
const SYS_ICC_CTLR_EL1: SimpleReg = SimpleReg::vgic_sys_reg(3, 0, 12, 12, 4);
const SYS_ICC_IGRPEN0_EL1: SimpleReg = SimpleReg::vgic_sys_reg(3, 0, 12, 12, 6);
const SYS_ICC_IGRPEN1_EL1: SimpleReg = SimpleReg::vgic_sys_reg(3, 0, 12, 12, 7);
const SYS_ICC_PMR_EL1: SimpleReg = SimpleReg::vgic_sys_reg(3, 0, 4, 6, 0);
const SYS_ICC_BPR0_EL1: SimpleReg = SimpleReg::vgic_sys_reg(3, 0, 12, 8, 3);
const SYS_ICC_BPR1_EL1: SimpleReg = SimpleReg::vgic_sys_reg(3, 0, 12, 12, 3);

const SYS_ICC_AP0R0_EL1: SimpleReg = SimpleReg::sys_icc_ap0rn_el1(0);
const SYS_ICC_AP0R1_EL1: SimpleReg = SimpleReg::sys_icc_ap0rn_el1(1);
const SYS_ICC_AP0R2_EL1: SimpleReg = SimpleReg::sys_icc_ap0rn_el1(2);
const SYS_ICC_AP0R3_EL1: SimpleReg = SimpleReg::sys_icc_ap0rn_el1(3);

const SYS_ICC_AP1R0_EL1: SimpleReg = SimpleReg::sys_icc_ap1rn_el1(0);
const SYS_ICC_AP1R1_EL1: SimpleReg = SimpleReg::sys_icc_ap1rn_el1(1);
const SYS_ICC_AP1R2_EL1: SimpleReg = SimpleReg::sys_icc_ap1rn_el1(2);
const SYS_ICC_AP1R3_EL1: SimpleReg = SimpleReg::sys_icc_ap1rn_el1(3);

// The interface control registers of a guest hypervisor (a vCPU with a virtual EL2). KVM keeps
// them in a page of its own per vCPU, leaves them out of KVM_GET_REG_LIST and refuses them to
// KVM_GET_ONE_REG: the vGIC device is the only way to them (`gic_v3_icc_reg_descs`, the
// `EL2_REG` entries). While such a vCPU has a nested guest's vCPU loaded, that guest's priority
// mask and group enables are in ICH_VMCR_EL2 and its pending and active interrupts are in the
// list registers. ICH_VTR_EL2 and ICC_SRE_EL2 are constants, and ICH_MISR_EL2, ICH_EISR_EL2 and
// ICH_ELRSR_EL2 are computed from the others.
const SYS_ICH_HCR_EL2: SimpleReg = SimpleReg::vgic_sys_reg(3, 4, 12, 11, 0);
const SYS_ICH_VMCR_EL2: SimpleReg = SimpleReg::vgic_sys_reg(3, 4, 12, 11, 7);
static MAIN_VGIC_ICH_REGS: [SimpleReg; 18] = [
    SYS_ICH_HCR_EL2,
    SYS_ICH_VMCR_EL2,
    SimpleReg::sys_ich_lrn_el2(0),
    SimpleReg::sys_ich_lrn_el2(1),
    SimpleReg::sys_ich_lrn_el2(2),
    SimpleReg::sys_ich_lrn_el2(3),
    SimpleReg::sys_ich_lrn_el2(4),
    SimpleReg::sys_ich_lrn_el2(5),
    SimpleReg::sys_ich_lrn_el2(6),
    SimpleReg::sys_ich_lrn_el2(7),
    SimpleReg::sys_ich_lrn_el2(8),
    SimpleReg::sys_ich_lrn_el2(9),
    SimpleReg::sys_ich_lrn_el2(10),
    SimpleReg::sys_ich_lrn_el2(11),
    SimpleReg::sys_ich_lrn_el2(12),
    SimpleReg::sys_ich_lrn_el2(13),
    SimpleReg::sys_ich_lrn_el2(14),
    SimpleReg::sys_ich_lrn_el2(15),
];

// In the order of AP_VGIC_ICC_REGS, whose availability rule they share.
static AP_VGIC_ICH_REGS: &[SimpleReg] = &[
    SimpleReg::sys_ich_ap0rn_el2(0),
    SimpleReg::sys_ich_ap0rn_el2(1),
    SimpleReg::sys_ich_ap0rn_el2(2),
    SimpleReg::sys_ich_ap0rn_el2(3),
    SimpleReg::sys_ich_ap1rn_el2(0),
    SimpleReg::sys_ich_ap1rn_el2(1),
    SimpleReg::sys_ich_ap1rn_el2(2),
    SimpleReg::sys_ich_ap1rn_el2(3),
];

static MAIN_VGIC_ICC_REGS: &[SimpleReg] = &[
    SYS_ICC_SRE_EL1,
    SYS_ICC_CTLR_EL1,
    SYS_ICC_IGRPEN0_EL1,
    SYS_ICC_IGRPEN1_EL1,
    SYS_ICC_PMR_EL1,
    SYS_ICC_BPR0_EL1,
    SYS_ICC_BPR1_EL1,
];

static AP_VGIC_ICC_REGS: &[SimpleReg] = &[
    SYS_ICC_AP0R0_EL1,
    SYS_ICC_AP0R1_EL1,
    SYS_ICC_AP0R2_EL1,
    SYS_ICC_AP0R3_EL1,
    SYS_ICC_AP1R0_EL1,
    SYS_ICC_AP1R1_EL1,
    SYS_ICC_AP1R2_EL1,
    SYS_ICC_AP1R3_EL1,
];

impl SimpleReg {
    const fn vgic_sys_reg(op0: u64, op1: u64, crn: u64, crm: u64, op2: u64) -> SimpleReg {
        let offset = ((op0 << KVM_REG_ARM64_SYSREG_OP0_SHIFT)
            & KVM_REG_ARM64_SYSREG_OP0_MASK as u64)
            | ((op1 << KVM_REG_ARM64_SYSREG_OP1_SHIFT) & KVM_REG_ARM64_SYSREG_OP1_MASK as u64)
            | ((crn << KVM_REG_ARM64_SYSREG_CRN_SHIFT) & KVM_REG_ARM64_SYSREG_CRN_MASK as u64)
            | ((crm << KVM_REG_ARM64_SYSREG_CRM_SHIFT) & KVM_REG_ARM64_SYSREG_CRM_MASK as u64)
            | ((op2 << KVM_REG_ARM64_SYSREG_OP2_SHIFT) & KVM_REG_ARM64_SYSREG_OP2_MASK as u64);

        SimpleReg::new(offset, 8)
    }

    const fn sys_icc_ap0rn_el1(n: u64) -> SimpleReg {
        Self::vgic_sys_reg(3, 0, 12, 8, 4 | n)
    }

    const fn sys_icc_ap1rn_el1(n: u64) -> SimpleReg {
        Self::vgic_sys_reg(3, 0, 12, 9, n)
    }

    const fn sys_ich_ap0rn_el2(n: u64) -> SimpleReg {
        Self::vgic_sys_reg(3, 4, 12, 8, n)
    }

    const fn sys_ich_ap1rn_el2(n: u64) -> SimpleReg {
        Self::vgic_sys_reg(3, 4, 12, 9, n)
    }

    /// ICH_LR0_EL2 to ICH_LR7_EL2 are CRm 12, ICH_LR8_EL2 to ICH_LR15_EL2 CRm 13.
    const fn sys_ich_lrn_el2(n: u64) -> SimpleReg {
        Self::vgic_sys_reg(3, 4, 12, 12 | (n >> 3), n & 7)
    }
}

struct VgicSysRegEngine {}

impl VgicRegEngine for VgicSysRegEngine {
    type Reg = SimpleReg;
    type RegChunk = u64;

    fn group() -> u32 {
        KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS
    }

    #[allow(clippy::cast_sign_loss)] // bit mask
    fn mpidr_mask() -> u64 {
        KVM_DEV_ARM_VGIC_V3_MPIDR_MASK as u64
    }
}

fn num_priority_bits(fd: &DeviceFd, mpidr: u64) -> Result<u64, GicError> {
    let reg_val = &VgicSysRegEngine::get_reg_data(fd, &SYS_ICC_CTLR_EL1, mpidr)?.chunks[0];

    Ok(((reg_val & ICC_CTLR_EL1_PRIBITS_MASK) >> ICC_CTLR_EL1_PRIBITS_SHIFT) + 1)
}

fn is_ap_reg_available(reg: &SimpleReg, num_priority_bits: u64) -> bool {
    // As per ARMv8 documentation:
    // https://static.docs.arm.com/ihi0069/c/IHI0069C_gic_architecture_specification.pdf
    // page 178,
    // ICC_AP0R1_EL1 is only implemented in implementations that support 6 or more bits of
    // priority.
    // ICC_AP0R2_EL1 and ICC_AP0R3_EL1 are only implemented in implementations that support
    // 7 bits of priority.
    if (reg == &SYS_ICC_AP0R1_EL1 || reg == &SYS_ICC_AP1R1_EL1) && num_priority_bits < 6 {
        return false;
    }
    if (reg == &SYS_ICC_AP0R2_EL1
        || reg == &SYS_ICC_AP0R3_EL1
        || reg == &SYS_ICC_AP1R2_EL1
        || reg == &SYS_ICC_AP1R3_EL1)
        && num_priority_bits != 7
    {
        return false;
    }

    true
}

pub(crate) fn get_icc_regs(fd: &DeviceFd, mpidr: u64) -> Result<VgicSysRegsState, GicError> {
    let main_icc_regs =
        VgicSysRegEngine::get_regs_data(fd, Box::new(MAIN_VGIC_ICC_REGS.iter()), mpidr)?;
    let num_priority_bits = num_priority_bits(fd, mpidr)?;

    let mut ap_icc_regs = Vec::with_capacity(AP_VGIC_ICC_REGS.len());
    for reg in AP_VGIC_ICC_REGS {
        if is_ap_reg_available(reg, num_priority_bits) {
            ap_icc_regs.push(Some(VgicSysRegEngine::get_reg_data(fd, reg, mpidr)?));
        } else {
            ap_icc_regs.push(None);
        }
    }

    let mut main_ich_regs = Vec::new();
    let mut ap_ich_regs = Vec::new();
    if has_ich_regs(fd, mpidr) {
        main_ich_regs =
            VgicSysRegEngine::get_regs_data(fd, Box::new(MAIN_VGIC_ICH_REGS.iter()), mpidr)?;
        // The index of an ICH_APxRn_EL2 is bounded as that of an ICC_APxRn_EL1 is, so the
        // register at the same place in the two lists is available or not together.
        for (reg, icc_reg) in AP_VGIC_ICH_REGS.iter().zip(AP_VGIC_ICC_REGS) {
            if is_ap_reg_available(icc_reg, num_priority_bits) {
                ap_ich_regs.push(Some(VgicSysRegEngine::get_reg_data(fd, reg, mpidr)?));
            } else {
                ap_ich_regs.push(None);
            }
        }
    }

    Ok(VgicSysRegsState {
        main_icc_regs,
        ap_icc_regs,
        main_ich_regs,
        ap_ich_regs,
    })
}

/// Whether the vGIC gives access to the ICH_*_EL2 registers of the vCPU with this MPIDR, which
/// it does for a vCPU with a virtual EL2 and for no other.
fn has_ich_regs(fd: &DeviceFd, mpidr: u64) -> bool {
    let mut val = 0_u64;
    let attr = VgicSysRegEngine::kvm_device_attr(SYS_ICH_HCR_EL2.range().start, &mut val, mpidr);
    fd.has_device_attr(&attr).is_ok()
}

pub(crate) fn set_icc_regs(
    fd: &DeviceFd,
    mpidr: u64,
    state: &VgicSysRegsState,
) -> Result<(), GicError> {
    VgicSysRegEngine::set_regs_data(
        fd,
        Box::new(MAIN_VGIC_ICC_REGS.iter()),
        &state.main_icc_regs,
        mpidr,
    )?;
    let num_priority_bits = num_priority_bits(fd, mpidr)?;

    for (reg, maybe_reg_data) in AP_VGIC_ICC_REGS.iter().zip(&state.ap_icc_regs) {
        if is_ap_reg_available(reg, num_priority_bits) != maybe_reg_data.is_some() {
            return Err(GicError::InvalidVgicSysRegState);
        }

        if let Some(reg_data) = maybe_reg_data {
            VgicSysRegEngine::set_reg_data(fd, reg, reg_data, mpidr)?;
        }
    }

    // A state with these registers was saved from a vCPU with a virtual EL2. KVM refuses them
    // for a vCPU without one, which fails the restore: a guest hypervisor restored without them
    // leaves its nested guest with its interrupts masked. The setters store the value given, so
    // the order among them does not matter.
    if state.main_ich_regs.is_empty() && state.ap_ich_regs.is_empty() {
        return Ok(());
    }
    if state.main_ich_regs.len() != MAIN_VGIC_ICH_REGS.len()
        || state.ap_ich_regs.len() != AP_VGIC_ICH_REGS.len()
    {
        return Err(GicError::InvalidVgicSysRegState);
    }
    VgicSysRegEngine::set_regs_data(
        fd,
        Box::new(MAIN_VGIC_ICH_REGS.iter()),
        &state.main_ich_regs,
        mpidr,
    )?;
    for ((reg, icc_reg), maybe_reg_data) in AP_VGIC_ICH_REGS
        .iter()
        .zip(AP_VGIC_ICC_REGS)
        .zip(&state.ap_ich_regs)
    {
        if is_ap_reg_available(icc_reg, num_priority_bits) != maybe_reg_data.is_some() {
            return Err(GicError::InvalidVgicSysRegState);
        }
        if let Some(reg_data) = maybe_reg_data {
            VgicSysRegEngine::set_reg_data(fd, reg, reg_data, mpidr)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]
    use std::os::unix::io::AsRawFd;

    use kvm_ioctls::Kvm;

    use super::*;
    use crate::arch::aarch64::gic::regs::GicRegState;
    use crate::arch::aarch64::gic::{GICVersion, create_gic};

    /// KVM_CAP_ARM_EL2 (include/uapi/linux/kvm.h), which the bindings in use predate.
    const KVM_CAP_ARM_EL2: u32 = 240;

    #[test]
    fn test_the_ich_register_encodings() {
        // op0 3, op1 4, CRn 12 for all of them (Arm ARM, D19.2: the ICH_*_EL2 registers).
        let id = |crm: u64, op2: u64| SimpleReg::vgic_sys_reg(3, 4, 12, crm, op2);
        assert!(SYS_ICH_HCR_EL2 == id(11, 0));
        assert!(SYS_ICH_VMCR_EL2 == id(11, 7));
        assert!(MAIN_VGIC_ICH_REGS[2] == id(12, 0)); // ICH_LR0_EL2
        assert!(MAIN_VGIC_ICH_REGS[9] == id(12, 7)); // ICH_LR7_EL2
        assert!(MAIN_VGIC_ICH_REGS[10] == id(13, 0)); // ICH_LR8_EL2
        assert!(MAIN_VGIC_ICH_REGS[17] == id(13, 7)); // ICH_LR15_EL2
        assert!(AP_VGIC_ICH_REGS[0] == id(8, 0)); // ICH_AP0R0_EL2
        assert!(AP_VGIC_ICH_REGS[3] == id(8, 3)); // ICH_AP0R3_EL2
        assert!(AP_VGIC_ICH_REGS[4] == id(9, 0)); // ICH_AP1R0_EL2
        assert!(AP_VGIC_ICH_REGS[7] == id(9, 3)); // ICH_AP1R3_EL2
        assert_eq!(AP_VGIC_ICH_REGS.len(), AP_VGIC_ICC_REGS.len());
    }

    #[test]
    fn test_ich_regs_in_a_state_are_refused_for_a_vcpu_without_a_virtual_el2() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _ = vm.create_vcpu(0).unwrap();
        let gic_fd = create_gic(&vm, 1, Some(GICVersion::GICV3)).expect("Cannot create gic");
        let mut state = get_icc_regs(gic_fd.device_fd(), 0).unwrap();

        // A state of the wrong shape is refused before anything is written.
        state.main_ich_regs = vec![GicRegState { chunks: vec![0] }];
        let res = set_icc_regs(gic_fd.device_fd(), 0, &state);
        assert_eq!(format!("{:?}", res.unwrap_err()), "InvalidVgicSysRegState");

        // A state saved from a guest hypervisor does not restore into a VM that is none.
        state.main_ich_regs = MAIN_VGIC_ICH_REGS
            .iter()
            .map(|_| GicRegState { chunks: vec![0] })
            .collect();
        state.ap_ich_regs = state
            .ap_icc_regs
            .iter()
            .map(|reg| reg.as_ref().map(|_| GicRegState { chunks: vec![0] }))
            .collect();
        let res = set_icc_regs(gic_fd.device_fd(), 0, &state);
        assert!(
            matches!(res, Err(GicError::DeviceAttribute(_, true, _))),
            "{res:?}"
        );
    }

    #[test]
    fn test_ich_regs_of_a_vcpu_with_a_virtual_el2_are_saved_and_restored() {
        let kvm = Kvm::new().unwrap();
        if kvm.check_extension_raw(u64::from(KVM_CAP_ARM_EL2)) <= 0 {
            println!("Skipping: this host's KVM offers no virtual EL2 (kvm-arm.mode=nested)");
            return;
        }
        // A VM whose one vCPU has a virtual EL2, with its vGIC, as a guest hypervisor's.
        let guest_hypervisor = || {
            let vm = kvm.create_vm().unwrap();
            let vcpu = vm.create_vcpu(0).unwrap();
            let mut kvi = kvm_vcpu_init::default();
            vm.get_preferred_target(&mut kvi).unwrap();
            kvi.features[0] |= 1 << KVM_ARM_VCPU_HAS_EL2;
            vcpu.vcpu_init(&kvi).unwrap();
            let gic = create_gic(&vm, 1, Some(GICVersion::GICV3)).expect("Cannot create gic");
            (vm, vcpu, gic)
        };

        let (_vm, _vcpu, gic) = guest_hypervisor();
        let mut state = get_icc_regs(gic.device_fd(), 0).unwrap();
        assert_eq!(state.main_ich_regs.len(), MAIN_VGIC_ICH_REGS.len());
        assert_eq!(state.ap_ich_regs.len(), AP_VGIC_ICH_REGS.len());
        assert!(state.ap_ich_regs[0].is_some() && state.ap_ich_regs[4].is_some());

        // What a guest hypervisor holds while its nested guest runs: virtual interrupts on
        // (ICH_HCR_EL2.En), the nested guest's priority mask open with group 1 enabled
        // (ICH_VMCR_EL2), and one pending group 1 interrupt in a list register.
        let hcr = 1_u64;
        let vmcr = (0xf0_u64 << 24) | (1 << 1);
        let lr0 = (1_u64 << 62) | (1 << 60) | 27;
        state.main_ich_regs[0].chunks[0] = hcr;
        state.main_ich_regs[1].chunks[0] = vmcr;
        state.main_ich_regs[2].chunks[0] = lr0;
        set_icc_regs(gic.device_fd(), 0, &state).unwrap();
        let saved = get_icc_regs(gic.device_fd(), 0).unwrap();
        assert_eq!(saved.main_ich_regs[0].chunks[0], hcr);
        assert_eq!(saved.main_ich_regs[1].chunks[0], vmcr);
        assert_eq!(saved.main_ich_regs[2].chunks[0], lr0);

        // A new VM starts with them clear and has them after the restore.
        let (_vm2, _vcpu2, gic2) = guest_hypervisor();
        let fresh = get_icc_regs(gic2.device_fd(), 0).unwrap();
        assert_eq!(fresh.main_ich_regs[1].chunks[0], 0);
        assert_eq!(fresh.main_ich_regs[2].chunks[0], 0);
        set_icc_regs(gic2.device_fd(), 0, &saved).unwrap();
        let restored = get_icc_regs(gic2.device_fd(), 0).unwrap();
        for (reg, (was, now)) in saved
            .main_ich_regs
            .iter()
            .zip(&restored.main_ich_regs)
            .enumerate()
        {
            assert_eq!(was.chunks, now.chunks, "ICH register {reg}");
        }
    }

    #[test]
    fn test_access_icc_regs() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _ = vm.create_vcpu(0).unwrap();
        let gic_fd = create_gic(&vm, 1, Some(GICVersion::GICV3)).expect("Cannot create gic");

        let gicr_typer = 123;
        let res = get_icc_regs(gic_fd.device_fd(), gicr_typer);
        let mut state = res.unwrap();
        assert_eq!(state.main_icc_regs.len(), 7);
        assert_eq!(state.ap_icc_regs.len(), 8);
        // This vCPU has no virtual EL2, so the vGIC offers no ICH_*_EL2 register for it.
        assert!(state.main_ich_regs.is_empty());
        assert!(state.ap_ich_regs.is_empty());

        set_icc_regs(gic_fd.device_fd(), gicr_typer, &state).unwrap();

        for reg in state.ap_icc_regs.iter_mut() {
            *reg = None;
        }
        let res = set_icc_regs(gic_fd.device_fd(), gicr_typer, &state);
        assert_eq!(format!("{:?}", res.unwrap_err()), "InvalidVgicSysRegState");

        unsafe { libc::close(gic_fd.device_fd().as_raw_fd()) };

        let res = set_icc_regs(gic_fd.device_fd(), gicr_typer, &state);
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            "DeviceAttribute(Error(9), true, 6)"
        );

        let res = get_icc_regs(gic_fd.device_fd(), gicr_typer);
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            "DeviceAttribute(Error(9), false, 6)"
        );

        // dropping gic_fd would double close the gic fd, so leak it
        std::mem::forget(gic_fd);
    }

    #[test]
    fn test_icc_constructors() {
        let sys_reg1 = SimpleReg::vgic_sys_reg(3, 0, 12, 12, 5);
        let sys_reg2 = SimpleReg::sys_icc_ap0rn_el1(1);
        let sys_reg3 = SimpleReg::sys_icc_ap1rn_el1(1);
        assert!(sys_reg1 == SimpleReg::new(50789, 8));
        assert!(sys_reg2 == SimpleReg::new(50757, 8));
        assert!(sys_reg3 == SimpleReg::new(50761, 8));
    }
}
