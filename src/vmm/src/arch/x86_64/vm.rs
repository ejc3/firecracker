// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::{Arc, Mutex};

use kvm_bindings::{
    KVM_CLOCK_REALTIME, KVM_IRQCHIP_IOAPIC, KVM_IRQCHIP_PIC_MASTER, KVM_IRQCHIP_PIC_SLAVE,
    KVM_MP_STATE_RUNNABLE, KVM_PIT_SPEAKER_DUMMY, MsrList, kvm_clock_data, kvm_irqchip,
    kvm_lapic_state, kvm_msi, kvm_pit_config, kvm_pit_state2,
};
use kvm_ioctls::Cap;
use serde::{Deserialize, Serialize};

use crate::arch::x86_64::msr::MsrError;
use crate::arch::x86_64::vcpu::{KvmVcpu, KvmVcpuError, VcpuState};
use crate::logger::{info, warn};
use crate::snapshot::Persist;
use crate::utils::u64_to_usize;
use crate::vstate::bus::Bus;
use crate::vstate::kvm::Kvm;
use crate::vstate::memory::{GuestMemoryExtension, GuestMemoryState};
use crate::vstate::resources::{ResourceAllocator, ResourceAllocatorState};
use crate::vstate::vm::{VmCommon, VmError};

/// Error type for [`KvmVm::restore_state`]
#[allow(missing_docs)]
#[cfg(target_arch = "x86_64")]
#[derive(Debug, PartialEq, Eq, thiserror::Error, displaydoc::Display)]
pub enum KvmVmError {
    /// Failed to check KVM capability (0): {1}
    CheckCapability(Cap, kvm_ioctls::Error),
    /// Set PIT2 error: {0}
    SetPit2(kvm_ioctls::Error),
    /// Set clock error: {0}
    SetClock(kvm_ioctls::Error),
    /// clock_realtime requested but not present in the snapshot state
    ClockRealtimeNotInState,
    /// Set IrqChipPicMaster error: {0}
    SetIrqChipPicMaster(kvm_ioctls::Error),
    /// Set IrqChipPicSlave error: {0}
    SetIrqChipPicSlave(kvm_ioctls::Error),
    /// Set IrqChipIoAPIC error: {0}
    SetIrqChipIoAPIC(kvm_ioctls::Error),
    /// Failed to get KVM vm pit state: {0}
    VmGetPit2(kvm_ioctls::Error),
    /// Failed to get KVM vm clock: {0}
    VmGetClock(kvm_ioctls::Error),
    /// Failed to get KVM vm irqchip: {0}
    VmGetIrqChip(kvm_ioctls::Error),
    /// Failed to set KVM vm irqchip: {0}
    VmSetIrqChip(kvm_ioctls::Error),
    /// Failed to get MSR index list to save into snapshots: {0}
    GetMsrsToSave(MsrError),
    /// Failed during KVM_SET_TSS_ADDRESS: {0}
    SetTssAddress(kvm_ioctls::Error),
    /// Failed to restore resource allocator: {0}
    ResourceAllocator(#[from] vm_allocator::Error),
    /// Failed to deliver a paravirtual kick to the restored vCPU with APIC id {0}: {1}
    KickRestoredVcpu(u8, kvm_ioctls::Error),
}

/// Error type for [`KvmVm::restore_kvm_state`]
#[derive(Debug, PartialEq, Eq, thiserror::Error, displaydoc::Display)]
pub enum RestoreKvmStateError {
    /// Failed to restore a vCPU: {0}
    Vcpu(#[from] KvmVcpuError),
    /// Failed to restore the VM: {0}
    Vm(#[from] KvmVmError),
}

/// Address of an MSI for a local APIC, without the destination (Intel SDM volume 3,
/// "Message Address Register Format"). Bit 2, the destination mode, is clear: physical.
const MSI_ADDRESS_BASE: u32 = 0xfee0_0000;
/// Position of the destination APIC id in an MSI address.
const MSI_ADDRESS_DEST_ID_SHIFT: u32 = 12;
/// MSI data whose delivery mode (bits 10:8) is 0b011, "remote read". The mode is reserved
/// on hardware. KVM gives it one meaning, `APIC_DM_REMRD` in `__apic_accept_irq()`: record
/// a paravirtual kick for the destination vCPU (`vcpu->arch.pv.pv_unhalted`), which is how
/// it delivers a guest's `KVM_HC_KICK_CPU`. The vector is not used.
const MSI_DATA_DELIVERY_REMOTE_READ: u32 = 0b011 << 8;
/// Offset of the id register in the local APIC's register page.
const APIC_ID_REGISTER: usize = 0x20;
/// The id that addresses every local APIC, not one.
const XAPIC_BROADCAST_ID: u8 = 0xff;

/// The id of a saved local APIC. `KVM_GET_LAPIC` returns it in the xAPIC layout, bits 31:24
/// of the id register, in x2APIC mode too, because this VMM does not enable
/// `KVM_X2APIC_API_USE_32BIT_IDS`.
fn saved_apic_id(lapic: &kvm_lapic_state) -> u8 {
    // The register is little endian, so bits 31:24 are its last byte.
    lapic.regs[APIC_ID_REGISTER + 3].to_ne_bytes()[0]
}

/// The MSI that records a paravirtual kick for the vCPU a snapshot holds `state` of, or
/// `None` when that vCPU gets none.
///
/// Only a runnable vCPU gets one. A halted vCPU with a kick pending is saved as runnable
/// (`KVM_GET_MP_STATE` folds the kick in), so a vCPU saved as halted has none to lose. A
/// vCPU that waits for INIT or SIPI must get none: with a kick recorded KVM takes it for
/// runnable, and its `KVM_RUN` returns at once (uninitialized) or spins in the kernel
/// (INIT received) where it would block.
fn pv_kick_msi(state: &VcpuState) -> Option<kvm_msi> {
    if state.mp_state.mp_state != KVM_MP_STATE_RUNNABLE {
        return None;
    }
    let apic_id = saved_apic_id(&state.lapic);
    if apic_id == XAPIC_BROADCAST_ID {
        // Guests here have at most 32 vCPUs with ids from 0, so only a guest that rewrote
        // its xAPIC id holds this one, and the MSI would reach every vCPU of the VM.
        warn!("A restored vCPU has the broadcast APIC id; it gets no paravirtual kick");
        return None;
    }
    Some(kvm_msi {
        address_lo: MSI_ADDRESS_BASE | (u32::from(apic_id) << MSI_ADDRESS_DEST_ID_SHIFT),
        address_hi: 0,
        data: MSI_DATA_DELIVERY_REMOTE_READ,
        ..Default::default()
    })
}

/// Structure representing the current architecture's understand of what a "virtual machine" is.
#[derive(Debug)]
pub struct KvmVm {
    /// Architecture independent parts of a vm
    pub common: VmCommon,
    msrs_to_save: MsrList,
    /// Size in bytes requiring to hold the dynamically-sized `kvm_xsave` struct.
    ///
    /// `None` if `KVM_CAP_XSAVE2` not supported.
    xsave2_size: Option<usize>,
    /// Port IO bus
    pub pio_bus: Arc<Bus>,
}

impl KvmVm {
    /// Create a new `KvmVm` struct.
    pub fn new(kvm: Kvm) -> Result<KvmVm, VmError> {
        let common = Self::create_common(kvm)?;
        let msrs_to_save = common
            .kvm
            .msrs_to_save()
            .map_err(KvmVmError::GetMsrsToSave)?;

        // `KVM_CAP_XSAVE2` was introduced to support dynamically-sized XSTATE buffer in kernel
        // v5.17. `KVM_GET_EXTENSION(KVM_CAP_XSAVE2)` returns the required size in byte if
        // supported; otherwise returns 0.
        // https://github.com/torvalds/linux/commit/be50b2065dfa3d88428fdfdc340d154d96bf6848
        //
        // Cache the value in order not to call it at each vCPU creation.
        let xsave2_size = match common.fd.check_extension_int(Cap::Xsave2) {
            // Catch all negative values just in case although the possible negative return value
            // of ioctl() is only -1.
            ..=-1 => {
                return Err(VmError::Arch(KvmVmError::CheckCapability(
                    Cap::Xsave2,
                    vmm_sys_util::errno::Error::last(),
                )));
            }
            0 => None,
            // SAFETY: Safe because negative values are handled above.
            ret => Some(usize::try_from(ret).unwrap()),
        };

        common
            .fd
            .set_tss_address(u64_to_usize(crate::arch::x86_64::layout::KVM_TSS_ADDRESS))
            .map_err(KvmVmError::SetTssAddress)?;

        let pio_bus = Arc::new(Bus::new());

        Ok(KvmVm {
            common,
            msrs_to_save,
            xsave2_size,
            pio_bus,
        })
    }

    /// Pre-vCPU creation setup.
    pub fn arch_pre_create_vcpus(&mut self, _: u8) -> Result<(), KvmVmError> {
        // For x86_64 we need to create the interrupt controller before calling `KVM_CREATE_VCPUS`
        self.setup_irqchip()
    }

    /// Post-vCPU creation setup.
    pub fn arch_post_create_vcpus(&mut self, _: u8) -> Result<(), KvmVmError> {
        Ok(())
    }

    /// Restores the KVM VM state.
    ///
    /// # Errors
    ///
    /// When:
    /// - [`kvm_ioctls::VmFd::set_pit`] errors.
    /// - [`kvm_ioctls::VmFd::set_clock`] errors.
    /// - [`kvm_ioctls::VmFd::set_irqchip`] errors.
    /// - [`kvm_ioctls::VmFd::set_irqchip`] errors.
    /// - [`kvm_ioctls::VmFd::set_irqchip`] errors.
    pub fn restore_state(
        &mut self,
        state: &VmState,
        clock_realtime: bool,
    ) -> Result<(), KvmVmError> {
        self.fd()
            .set_pit2(&state.pitstate)
            .map_err(KvmVmError::SetPit2)?;
        let mut clock = state.clock;
        clock.flags = if clock_realtime {
            // clock_realtime needs to be present in the snapshot
            if clock.flags & KVM_CLOCK_REALTIME == 0 {
                return Err(KvmVmError::ClockRealtimeNotInState);
            }
            KVM_CLOCK_REALTIME
        } else {
            0
        };
        self.fd().set_clock(&clock).map_err(KvmVmError::SetClock)?;
        self.fd()
            .set_irqchip(&state.pic_master)
            .map_err(KvmVmError::SetIrqChipPicMaster)?;
        self.fd()
            .set_irqchip(&state.pic_slave)
            .map_err(KvmVmError::SetIrqChipPicSlave)?;
        self.fd()
            .set_irqchip(&state.ioapic)
            .map_err(KvmVmError::SetIrqChipIoAPIC)?;
        self.common.resource_allocator =
            Mutex::new(ResourceAllocator::restore((), &state.resource_allocator)?);
        Ok(())
    }

    /// Restores the KVM state of every vCPU and then of the VM, as a snapshot load does, and
    /// then gives every runnable vCPU a paravirtual kick ([`Self::kick_runnable_vcpus`]).
    ///
    /// The kick comes last because `KVM_SET_MP_STATE` to runnable clears one.
    pub fn restore_kvm_state<'a>(
        &mut self,
        vcpus: impl IntoIterator<Item = &'a KvmVcpu>,
        vcpu_states: &[VcpuState],
        state: &VmState,
        clock_realtime: bool,
    ) -> Result<(), RestoreKvmStateError> {
        for (vcpu, vcpu_state) in vcpus.into_iter().zip(vcpu_states) {
            vcpu.restore_state(vcpu_state)?;
        }
        self.restore_state(state, clock_realtime)?;
        self.kick_runnable_vcpus(vcpu_states)?;
        Ok(())
    }

    /// Records a paravirtual kick in KVM for every vCPU that `states` holds as runnable, and
    /// returns how many it kicked.
    ///
    /// A guest with paravirtual spinlocks waits for a lock by checking the lock word and
    /// then executing HLT (`kvm_wait()`), with interrupts off when it took the lock in
    /// interrupt context. A kick from the lock holder (`KVM_HC_KICK_CPU`) that lands between
    /// the check and the HLT is held in `vcpu->arch.pv.pv_unhalted`, and makes that HLT
    /// return at once. No ioctl reads the flag of a runnable vCPU, so a snapshot does not
    /// hold it, and a vCPU restored from one would execute its HLT with the kick gone and,
    /// with interrupts off, wait for a wakeup that has already been sent.
    ///
    /// The kick given here stands in for the one that may have been pending. A vCPU that was
    /// not waiting for one sees its next HLT return early once, which Linux allows for: the
    /// spinlock wait and the idle loop check again and halt again.
    ///
    /// The kick is delivered as KVM delivers a guest's: an interrupt with the "remote read"
    /// delivery mode, sent here with `KVM_SIGNAL_MSI`, which needs the in-kernel irqchip.
    /// It stays recorded until the vCPU's next HLT, so a VM that is loaded paused and
    /// resumed later needs no second one.
    pub fn kick_runnable_vcpus(&self, states: &[VcpuState]) -> Result<usize, KvmVmError> {
        let mut kicked = 0;
        for msi in states.iter().filter_map(pv_kick_msi) {
            // The id is the 8 bits `pv_kick_msi` put there.
            let apic_id = (msi.address_lo >> MSI_ADDRESS_DEST_ID_SHIFT).to_le_bytes()[0];
            match self.fd().signal_msi(msi) {
                Ok(0) => warn!(
                    "No local APIC with id {apic_id} took a paravirtual kick after the restore of \
                     its vCPU"
                ),
                Ok(_) => kicked += 1,
                Err(err) => return Err(KvmVmError::KickRestoredVcpu(apic_id, err)),
            }
        }
        info!(
            "Restored the KVM state of {} vCPUs and gave {kicked} runnable ones a paravirtual kick",
            states.len()
        );
        Ok(kicked)
    }

    /// Creates the irq chip and an in-kernel device model for the PIT.
    pub fn setup_irqchip(&self) -> Result<(), KvmVmError> {
        self.fd()
            .create_irq_chip()
            .map_err(KvmVmError::VmSetIrqChip)?;
        // We need to enable the emulation of a dummy speaker port stub so that writing to port 0x61
        // (i.e. KVM_SPEAKER_BASE_ADDRESS) does not trigger an exit to user space.
        let pit_config = kvm_pit_config {
            flags: KVM_PIT_SPEAKER_DUMMY,
            ..Default::default()
        };
        self.fd()
            .create_pit2(pit_config)
            .map_err(KvmVmError::VmSetIrqChip)
    }

    /// Saves and returns the KVM VM state.
    pub fn save_state(&self) -> Result<VmState, KvmVmError> {
        let pitstate = self.fd().get_pit2().map_err(KvmVmError::VmGetPit2)?;

        let clock = self.fd().get_clock().map_err(KvmVmError::VmGetClock)?;

        let mut pic_master = kvm_irqchip {
            chip_id: KVM_IRQCHIP_PIC_MASTER,
            ..Default::default()
        };
        self.fd()
            .get_irqchip(&mut pic_master)
            .map_err(KvmVmError::VmGetIrqChip)?;

        let mut pic_slave = kvm_irqchip {
            chip_id: KVM_IRQCHIP_PIC_SLAVE,
            ..Default::default()
        };
        self.fd()
            .get_irqchip(&mut pic_slave)
            .map_err(KvmVmError::VmGetIrqChip)?;

        let mut ioapic = kvm_irqchip {
            chip_id: KVM_IRQCHIP_IOAPIC,
            ..Default::default()
        };
        self.fd()
            .get_irqchip(&mut ioapic)
            .map_err(KvmVmError::VmGetIrqChip)?;

        Ok(VmState {
            memory: self.common.guest_memory.describe(),
            resource_allocator: self.resource_allocator().save(),
            pitstate,
            clock,
            pic_master,
            pic_slave,
            ioapic,
        })
    }

    /// Gets the list of MSRs to save when creating snapshots
    pub fn msrs_to_save(&self) -> &[u32] {
        self.msrs_to_save.as_slice()
    }

    /// Gets the size (in bytes) of the `kvm_xsave` struct.
    pub fn xsave2_size(&self) -> Option<usize> {
        self.xsave2_size
    }
}

#[derive(Default, Deserialize, Serialize)]
/// Structure holding VM kvm state.
pub struct VmState {
    /// guest memory state
    pub memory: GuestMemoryState,
    /// resource allocator
    pub resource_allocator: ResourceAllocatorState,
    pitstate: kvm_pit_state2,
    clock: kvm_clock_data,
    // TODO: rename this field to adopt inclusive language once Linux updates it, too.
    pic_master: kvm_irqchip,
    // TODO: rename this field to adopt inclusive language once Linux updates it, too.
    pic_slave: kvm_irqchip,
    ioapic: kvm_irqchip,
}

impl fmt::Debug for VmState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VmState")
            .field("pitstate", &self.pitstate)
            .field("clock", &self.clock)
            .field("pic_master", &"?")
            .field("pic_slave", &"?")
            .field("ioapic", &"?")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::time::{Duration, SystemTime};

    use kvm_bindings::{
        KVM_CLOCK_REALTIME, KVM_IRQCHIP_IOAPIC, KVM_IRQCHIP_PIC_MASTER, KVM_IRQCHIP_PIC_SLAVE,
        KVM_MP_STATE_HALTED, KVM_MP_STATE_INIT_RECEIVED, KVM_MP_STATE_RUNNABLE,
        KVM_MP_STATE_SIPI_RECEIVED, KVM_MP_STATE_UNINITIALIZED, KVM_PIT_SPEAKER_DUMMY,
        kvm_mp_state,
    };
    use kvm_ioctls::{Cap, VcpuExit};
    use vm_memory::GuestAddress;

    use super::{KvmVm, pv_kick_msi};
    use crate::arch::x86_64::vcpu::{KvmVcpu, VcpuState};
    use crate::arch::{BootProtocol, EntryPoint, KvmVmError};
    use crate::cpu_config::x86_64::cpuid::Cpuid;
    use crate::utils::signal::{Killable, register_signal_handler, sigrtmin};
    use crate::vstate::memory::Bytes;
    use crate::vstate::vm::VmState;
    use crate::vstate::vm::tests::{setup_vm, setup_vm_with_memory};

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vm_save_restore_state() {
        let vm = setup_vm();
        // Irqchips, clock and pitstate are not configured so trying to save state should fail.
        vm.save_state().unwrap_err();

        let vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();

        let vm_state = vm.save_state().unwrap();
        assert_eq!(
            vm_state.pitstate.flags | KVM_PIT_SPEAKER_DUMMY,
            KVM_PIT_SPEAKER_DUMMY
        );
        assert_eq!(vm_state.pic_master.chip_id, KVM_IRQCHIP_PIC_MASTER);
        assert_eq!(vm_state.pic_slave.chip_id, KVM_IRQCHIP_PIC_SLAVE);
        assert_eq!(vm_state.ioapic.chip_id, KVM_IRQCHIP_IOAPIC);

        let mut vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();

        vm.restore_state(&vm_state, false).unwrap();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vm_save_restore_state_kvm_clock_realtime() {
        let vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();

        let clock_realtime_supported = vm
            .kvm()
            .fd
            .check_extension_int(Cap::AdjustClock)
            .cast_unsigned()
            & KVM_CLOCK_REALTIME
            != 0;

        // mock a state without realtime information
        let mut vm_state = vm.save_state().unwrap();
        vm_state.clock.flags &= !KVM_CLOCK_REALTIME;

        let mut vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();

        let res = vm.restore_state(&vm_state, true);
        assert!(res == Err(KvmVmError::ClockRealtimeNotInState));

        // mock a state with realtime information
        vm_state.clock.flags |= KVM_CLOCK_REALTIME;
        vm_state.clock.realtime = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .try_into()
            .unwrap();

        let mut vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();

        let res = vm.restore_state(&vm_state, true);
        if clock_realtime_supported {
            res.unwrap()
        } else {
            assert!(matches!(res, Err(KvmVmError::SetClock(err)) if err.errno() == libc::EINVAL))
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vm_save_restore_state_bad_irqchip() {
        use kvm_bindings::KVM_NR_IRQCHIPS;

        let vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();
        let mut vm_state = vm.save_state().unwrap();

        let mut vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();

        // Try to restore an invalid PIC Master chip ID
        let orig_master_chip_id = vm_state.pic_master.chip_id;
        vm_state.pic_master.chip_id = KVM_NR_IRQCHIPS;
        vm.restore_state(&vm_state, false).unwrap_err();
        vm_state.pic_master.chip_id = orig_master_chip_id;

        // Try to restore an invalid PIC Slave chip ID
        let orig_slave_chip_id = vm_state.pic_slave.chip_id;
        vm_state.pic_slave.chip_id = KVM_NR_IRQCHIPS;
        vm.restore_state(&vm_state, false).unwrap_err();
        vm_state.pic_slave.chip_id = orig_slave_chip_id;

        // Try to restore an invalid IOPIC chip ID
        vm_state.ioapic.chip_id = KVM_NR_IRQCHIPS;
        vm.restore_state(&vm_state, false).unwrap_err();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vmstate_serde() {
        let mut vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();
        let state = vm.save_state().unwrap();

        // Test direct bitcode serialization
        let serialized_data = bitcode::serialize(&state).unwrap();
        let restored_state: VmState = bitcode::deserialize(&serialized_data).unwrap();

        vm.restore_state(&restored_state, false).unwrap();
    }

    // The guest of the paravirtual kick tests: 64-bit code that stops with interrupts off,
    // says so on a port, executes HLT, and says on the port that it ran past the HLT.
    const GUEST_PORT: u8 = 0xe9;
    const BEFORE_HLT: u8 = 1;
    const PAST_HLT: u8 = 2;
    #[rustfmt::skip]
    const GUEST_CODE: [u8; 13] = [
        0xfa,                   // cli
        0xb0, BEFORE_HLT,       // mov al, BEFORE_HLT
        0xe6, GUEST_PORT,       // out GUEST_PORT, al
        0xf4,                   // hlt
        0xb0, PAST_HLT,         // mov al, PAST_HLT
        0xe6, GUEST_PORT,       // out GUEST_PORT, al
        0xf4,                   // hlt
        0xeb, 0xfd,             // jmp to the hlt before it
    ];
    // Offset of the first HLT in GUEST_CODE.
    const GUEST_HLT_OFFSET: u64 = 5;
    const GUEST_CODE_ADDR: u64 = 0x10000;
    const GUEST_MEM_SIZE: usize = 0x20000;
    const RFLAGS_IF: u64 = 1 << 9;

    // How long a vCPU that should run past its HLT gets to do it. It takes microseconds;
    // the whole of it is spent only when the test fails.
    const RUN_WAIT: Duration = Duration::from_secs(10);
    // How long a vCPU that should stay blocked is watched.
    const BLOCK_WAIT: Duration = Duration::from_millis(500);

    fn guest_vm() -> KvmVm {
        let vm = setup_vm_with_memory(GUEST_MEM_SIZE);
        vm.setup_irqchip().unwrap();
        vm
    }

    // A VM with GUEST_CODE in memory and `vcpu_count` vCPUs configured as for a Linux boot
    // into it. With the in-kernel irqchip the first is runnable and the others are
    // uninitialized, as application processors are until the guest starts them.
    fn guest(vcpu_count: u8) -> (KvmVm, Vec<KvmVcpu>) {
        let vm = guest_vm();
        vm.guest_memory()
            .write_slice(&GUEST_CODE, GuestAddress(GUEST_CODE_ADDR))
            .unwrap();
        let cpuid = Cpuid::try_from(vm.kvm().supported_cpuid.clone()).unwrap();
        let vcpus = (0..vcpu_count)
            .map(|index| {
                let mut vcpu = KvmVcpu::new(index, &vm).unwrap();
                let configured = vcpu.configure_cpuid(&cpuid, vcpu_count, false).unwrap();
                vcpu.configure_msrs_for_boot(&BTreeMap::new(), &configured)
                    .unwrap();
                vcpu.configure_boot_state(
                    vm.guest_memory(),
                    EntryPoint {
                        entry_addr: GuestAddress(GUEST_CODE_ADDR),
                        protocol: BootProtocol::LinuxBoot,
                        setup_header: None,
                    },
                )
                .unwrap();
                vcpu
            })
            .collect();
        (vm, vcpus)
    }

    // `guest`, with its first vCPU run up to its first HLT: runnable, interrupts off, the
    // HLT its next instruction. That is a vCPU in `kvm_wait()` between the check of the
    // lock word and the HLT.
    fn guest_stopped_before_hlt(vcpu_count: u8) -> (KvmVm, Vec<KvmVcpu>) {
        let (vm, mut vcpus) = guest(vcpu_count);
        let vcpu = &mut vcpus[0];
        match vcpu.fd.run() {
            Ok(VcpuExit::IoOut(port, data)) => {
                assert_eq!((port, data), (u16::from(GUEST_PORT), &[BEFORE_HLT][..]));
            }
            exit => panic!("the guest did not reach its first port write: {exit:?}"),
        }
        // KVM finishes the port write on the next KVM_RUN. With immediate_exit it does
        // that and runs no guest code.
        vcpu.fd.set_kvm_immediate_exit(1);
        assert_eq!(vcpu.fd.run().unwrap_err().errno(), libc::EINTR);
        vcpu.fd.set_kvm_immediate_exit(0);
        let regs = vcpu.fd.get_regs().unwrap();
        assert_eq!(regs.rip, GUEST_CODE_ADDR + GUEST_HLT_OFFSET);
        assert_eq!(regs.rflags & RFLAGS_IF, 0);
        (vm, vcpus)
    }

    // A new VM with the memory of `source`, and as many vCPUs as `vcpu_states`, their state
    // not restored yet.
    fn guest_to_restore(source: &KvmVm, vcpu_states: &[VcpuState]) -> (KvmVm, Vec<KvmVcpu>) {
        let vm = guest_vm();
        let mut memory = vec![0u8; GUEST_MEM_SIZE];
        source
            .guest_memory()
            .read_slice(&mut memory, GuestAddress(0))
            .unwrap();
        vm.guest_memory()
            .write_slice(&memory, GuestAddress(0))
            .unwrap();
        let vcpus = (0..vcpu_states.len())
            .map(|index| KvmVcpu::new(u8::try_from(index).unwrap(), &vm).unwrap())
            .collect();
        (vm, vcpus)
    }

    // How a bounded run of a test vCPU ended.
    #[derive(Debug, PartialEq, Eq)]
    enum Ran {
        // The guest wrote PAST_HLT to its port.
        PastHlt,
        // KVM_RUN failed with this errno, which is not EINTR.
        Returned(i32),
        // KVM_RUN had not come back to userspace when the wait ended. `cpu` is the CPU time
        // the vCPU thread had used by then: next to none when KVM_RUN blocked.
        StillInKvmRun { cpu: Duration },
    }

    extern "C" fn interrupt_kvm_run(_: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
    }

    fn thread_cpu_time() -> Duration {
        let mut time = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `time` is a valid timespec for the call to fill.
        let ret = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
        assert_eq!(ret, 0);
        Duration::new(
            u64::try_from(time.tv_sec).unwrap(),
            u32::try_from(time.tv_nsec).unwrap(),
        )
    }

    // Runs `vcpu` on a thread of its own until the guest writes PAST_HLT, KVM_RUN fails, or
    // `wait` has passed, when a signal takes the thread out of KVM_RUN.
    fn run_bounded(mut vcpu: KvmVcpu, wait: Duration) -> (KvmVcpu, Ran) {
        // Not the signal the vCPU threads of other tests in this process use.
        let signal = sigrtmin() + 1;
        register_signal_handler(signal, interrupt_kvm_run).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (done_tx, done_rx) = mpsc::channel();
        let thread = std::thread::spawn({
            let stop = stop.clone();
            move || {
                let ran = loop {
                    if stop.load(Ordering::SeqCst) {
                        break Ran::StillInKvmRun {
                            cpu: thread_cpu_time(),
                        };
                    }
                    match vcpu.fd.run() {
                        Ok(VcpuExit::IoOut(port, data)) => {
                            assert_eq!(port, u16::from(GUEST_PORT));
                            if data == [PAST_HLT] {
                                break Ran::PastHlt;
                            }
                        }
                        Ok(exit) => panic!("unexpected exit of the test guest: {exit:?}"),
                        Err(err) if err.errno() == libc::EINTR => (),
                        Err(err) => break Ran::Returned(err.errno()),
                    }
                };
                // The receiver is gone only if the test thread panicked.
                let _ = done_tx.send(());
                (vcpu, ran)
            }
        });
        let mut wait = wait;
        loop {
            match done_rx.recv_timeout(wait) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    // The thread may be between its check of `stop` and KVM_RUN, where the
                    // signal does nothing, so it is sent until the thread answers.
                    stop.store(true, Ordering::SeqCst);
                    thread.kill(signal).unwrap();
                    wait = Duration::from_millis(10);
                }
            }
        }
        thread.join().expect("the vCPU thread panicked")
    }

    fn assert_blocked(ran: Ran, what: &str) {
        match ran {
            Ran::StillInKvmRun { cpu } => assert!(
                cpu < BLOCK_WAIT / 5,
                "{what}: KVM_RUN used {cpu:?} of CPU in {BLOCK_WAIT:?}, so it was not blocked"
            ),
            ran => panic!("{what}: KVM_RUN did not block: {ran:?}"),
        }
    }

    // A vCPU stopped just before a HLT with interrupts off, with a paravirtual kick pending,
    // is saved and restored into a new VM. The saved state does not hold the kick, so
    // without the one the restore gives, the restored vCPU halts for ever
    // (test_a_restored_vcpu_blocks_in_its_hlt_without_a_kick).
    #[test]
    fn test_restore_kicks_a_vcpu_stopped_before_its_hlt() {
        let (vm, mut vcpus) = guest_stopped_before_hlt(1);

        // The kick of a lock holder's KVM_HC_KICK_CPU, delivered the same way.
        let kicked = vm.kick_runnable_vcpus(&[vcpus[0].save_state().unwrap()]);
        assert_eq!(kicked, Ok(1));

        // KVM reports a runnable vCPU and nothing of the kick.
        let vcpu_states = [vcpus[0].save_state().unwrap()];
        assert_eq!(vcpu_states[0].mp_state.mp_state, KVM_MP_STATE_RUNNABLE);
        let vm_state = vm.save_state().unwrap();

        let (mut restored_vm, mut restored_vcpus) = guest_to_restore(&vm, &vcpu_states);
        restored_vm
            .restore_kvm_state(&restored_vcpus, &vcpu_states, &vm_state, false)
            .unwrap();
        let (_, ran) = run_bounded(restored_vcpus.remove(0), RUN_WAIT);
        assert_eq!(
            ran,
            Ran::PastHlt,
            "the restored vCPU did not run past its HLT"
        );

        // Saving changed nothing in the source: it still has its kick.
        let (_, ran) = run_bounded(vcpus.remove(0), RUN_WAIT);
        assert_eq!(
            ran,
            Ran::PastHlt,
            "the source vCPU did not run past its HLT"
        );
    }

    // The state the test above restores, restored without the kick: the vCPU blocks in its
    // HLT, and the kick is what it waits for.
    #[test]
    fn test_a_restored_vcpu_blocks_in_its_hlt_without_a_kick() {
        let (vm, vcpus) = guest_stopped_before_hlt(1);
        assert_eq!(
            vm.kick_runnable_vcpus(&[vcpus[0].save_state().unwrap()]),
            Ok(1)
        );
        let vcpu_states = [vcpus[0].save_state().unwrap()];
        let vm_state = vm.save_state().unwrap();

        let (mut restored_vm, mut restored_vcpus) = guest_to_restore(&vm, &vcpu_states);
        restored_vcpus[0].restore_state(&vcpu_states[0]).unwrap();
        restored_vm.restore_state(&vm_state, false).unwrap();
        let (vcpu, ran) = run_bounded(restored_vcpus.remove(0), BLOCK_WAIT);
        assert_blocked(ran, "a vCPU restored without a kick");

        // It is halted now, and KVM reports a halted vCPU as such while it has no kick.
        let halted = vcpu.save_state().unwrap();
        assert_eq!(halted.mp_state.mp_state, KVM_MP_STATE_HALTED);
        assert_eq!(restored_vm.kick_runnable_vcpus(&[halted]), Ok(0));

        // A kick wakes it.
        restored_vm
            .fd()
            .signal_msi(pv_kick_msi(&vcpu_states[0]).unwrap())
            .unwrap();
        let (_, ran) = run_bounded(vcpu, RUN_WAIT);
        assert_eq!(ran, Ran::PastHlt);
    }

    // A vCPU that waits for INIT or SIPI gets no kick. With one recorded KVM takes it for
    // runnable: KVM_RUN of an uninitialized vCPU returns EAGAIN at once, and KVM_RUN of one
    // that received INIT spins in the kernel, where both would block.
    #[test]
    fn test_restore_gives_no_kick_to_a_vcpu_that_waits_for_init_or_sipi() {
        for waiting in [KVM_MP_STATE_UNINITIALIZED, KVM_MP_STATE_INIT_RECEIVED] {
            let (vm, vcpus) = guest_stopped_before_hlt(2);
            vcpus[1]
                .fd
                .set_mp_state(kvm_mp_state { mp_state: waiting })
                .unwrap();
            let vcpu_states = [
                vcpus[0].save_state().unwrap(),
                vcpus[1].save_state().unwrap(),
            ];
            assert_eq!(vcpu_states[1].mp_state.mp_state, waiting);
            let vm_state = vm.save_state().unwrap();

            let (mut restored_vm, mut restored_vcpus) = guest_to_restore(&vm, &vcpu_states);
            restored_vm
                .restore_kvm_state(&restored_vcpus, &vcpu_states, &vm_state, false)
                .unwrap();
            let (_, ran) = run_bounded(restored_vcpus.remove(1), BLOCK_WAIT);
            assert_blocked(ran, &format!("a restored vCPU in mp_state {waiting}"));
            // The runnable vCPU of the same VM got its kick.
            let (_, ran) = run_bounded(restored_vcpus.remove(0), RUN_WAIT);
            assert_eq!(ran, Ran::PastHlt);
        }
    }

    #[test]
    fn test_pv_kick_msi_is_for_runnable_vcpus_only() {
        let state_of = |mp_state, apic_id: u8| {
            let mut state = VcpuState::default();
            state.mp_state.mp_state = mp_state;
            state.lapic.regs[0x23] = i8::from_ne_bytes([apic_id]);
            state
        };

        // Remote read delivery to the physical destination 5, vector unused.
        let msi = pv_kick_msi(&state_of(KVM_MP_STATE_RUNNABLE, 5)).unwrap();
        assert_eq!(msi.address_lo, 0xfee0_5000);
        assert_eq!(msi.address_hi, 0);
        assert_eq!(msi.data, 0x300);
        assert_eq!(msi.flags, 0);
        let msi = pv_kick_msi(&state_of(KVM_MP_STATE_RUNNABLE, 254)).unwrap();
        assert_eq!(msi.address_lo, 0xfeef_e000);

        for mp_state in [
            KVM_MP_STATE_UNINITIALIZED,
            KVM_MP_STATE_INIT_RECEIVED,
            KVM_MP_STATE_SIPI_RECEIVED,
            KVM_MP_STATE_HALTED,
        ] {
            assert!(pv_kick_msi(&state_of(mp_state, 5)).is_none(), "{mp_state}");
        }
        // 0xff addresses every local APIC.
        assert!(pv_kick_msi(&state_of(KVM_MP_STATE_RUNNABLE, 0xff)).is_none());
    }
}
