// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use kvm_bindings::kvm_one_reg;
use kvm_bindings::{KVMIO, kvm_arm_counter_offset};
use kvm_ioctls::{VcpuFd, VmFd};
use vmm_sys_util::errno;
use vmm_sys_util::ioctl::ioctl_with_ref;
use vmm_sys_util::ioctl_iow_nr;

#[cfg(test)]
use crate::arch::aarch64::regs::KVM_REG_ARM_TIMER_CVAL;
use crate::arch::aarch64::regs::{KVM_REG_ARM_TIMER_CNT, SYS_CNTPCT_EL0};
use crate::arch::aarch64::vcpu::VcpuState;

// KVM ioctl that atomically establishes the VM-wide physical and virtual offsets.
ioctl_iow_nr!(
    KVM_ARM_SET_COUNTER_OFFSET,
    KVMIO,
    0xb5,
    kvm_arm_counter_offset
);

#[cfg(test)]
ioctl_iow_nr!(TEST_KVM_GET_ONE_REG, KVMIO, 0xab, kvm_one_reg);

/// Runtime state of the VM-wide Arm generic-counter domain.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CounterState {
    /// Counter configuration has not run yet.
    #[default]
    Uninitialized,
    /// The host does not implement `KVM_CAP_COUNTER_OFFSET`.
    Unsupported,
    /// The guest is running with the given VM-wide offset.
    Running { offset: u64 },
    /// The guest is paused at `guest_counter` with the given VM-wide offset.
    Paused { offset: u64, guest_counter: u64 },
    /// A lifecycle transition became ambiguous, so no further operation is safe.
    Faulted,
}

/// Failures while owning the Arm generic-counter domain.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum CounterError {
    /// KVM_CAP_COUNTER_OFFSET is required for this operation.
    Unsupported,
    /// Failed to query KVM_CAP_COUNTER_OFFSET: {0}
    CheckCapability(vmm_sys_util::errno::Error),
    /// The counter domain has not been configured.
    Uninitialized,
    /// The counter domain has already been configured.
    AlreadyConfigured,
    /// The counter domain is already paused.
    AlreadyPaused,
    /// The counter domain is not paused.
    NotPaused,
    /// The counter domain entered an unrecoverable lifecycle state.
    Faulted,
    /// Counter pause point changed from {0} to {1} while collecting snapshot state.
    SnapshotPausePointChanged(u64, u64),
    /// The snapshot contains no vCPU state.
    NoVcpuState,
    /// No boot vCPU is available for a counter operation.
    NoBootVcpu,
    /// The snapshot's vCPU {0} has no saved CNTPCT_EL0.
    MissingSavedCounter(usize),
    /// The snapshot's vCPU {0} has more than one saved CNTPCT_EL0.
    DuplicateSavedCounter(usize),
    /// The snapshot's vCPU {0} has no saved CNTVCT_EL0.
    MissingSavedVirtualCounter(usize),
    /// The snapshot's vCPU {0} has more than one saved CNTVCT_EL0.
    DuplicateSavedVirtualCounter(usize),
    /// Snapshot vCPU state count {0} does not match configured vCPU count {1}.
    VcpuStateCount(usize, usize),
    /// Failed to read CNTPCT_EL0: {0}
    ReadCounter(kvm_ioctls::Error),
    /// Failed to set KVM_ARM_SET_COUNTER_OFFSET: {0}
    SetCounterOffset(vmm_sys_util::errno::Error),
    #[cfg(test)]
    /// Injected counter backend failure: {0}
    Injected(&'static str),
}

pub(crate) trait CounterIo {
    fn read_counter(&self) -> Result<u64, CounterError>;
    fn set_offset(&self, offset: u64) -> Result<(), CounterError>;
}

/// Production counter backend backed by KVM's vCPU and VM ioctls.
pub(crate) struct KvmCounterIo<'a> {
    vm_fd: &'a VmFd,
    vcpu_fd: &'a VcpuFd,
}

impl<'a> KvmCounterIo<'a> {
    /// Construct a KVM counter backend.
    pub(crate) fn new(vm_fd: &'a VmFd, vcpu_fd: &'a VcpuFd) -> Self {
        Self { vm_fd, vcpu_fd }
    }
}

impl CounterIo for KvmCounterIo<'_> {
    fn read_counter(&self) -> Result<u64, CounterError> {
        // KVM_GET_ONE_REG(SYS_CNTPCT_EL0) reaches kvm_phys_timer_read(), the
        // same KVM counter source visible to the guest. This remains true for
        // an L1 VMM under nested virtualization and deliberately avoids any
        // assumption that userspace CNTVCT_EL0 shares KVM's counter domain.
        let mut counter = [0_u8; 8];
        self.vcpu_fd
            .get_one_reg(SYS_CNTPCT_EL0, &mut counter)
            .map_err(CounterError::ReadCounter)?;
        Ok(u64::from_le_bytes(counter))
    }

    fn set_offset(&self, offset: u64) -> Result<(), CounterError> {
        let argument = kvm_arm_counter_offset {
            counter_offset: offset,
            reserved: 0,
        };
        // The VM ioctl updates both physical and virtual offsets and sets
        // KVM_ARCH_FLAG_VM_COUNTER_OFFSET. Once set, KVM intentionally ignores
        // later CNTVCT_EL0/CNTPCT_EL0 SET_ONE_REG replay for every vCPU.
        // SAFETY: `vm_fd` is a live KVM VM descriptor, `argument` has the
        // exact UAPI layout from kvm-bindings, and the return value is checked.
        if unsafe { ioctl_with_ref(self.vm_fd, KVM_ARM_SET_COUNTER_OFFSET(), &argument) } < 0 {
            return Err(CounterError::SetCounterOffset(errno::Error::last()));
        }
        Ok(())
    }
}

/// Owns counter lifecycle state that is intentionally not part of the snapshot format.
#[derive(Debug, Default)]
pub(crate) struct CounterController {
    state: CounterState,
}

impl CounterController {
    pub(crate) fn ensure_healthy(&self) -> Result<(), CounterError> {
        match self.state {
            CounterState::Faulted => Err(CounterError::Faulted),
            CounterState::Uninitialized => Err(CounterError::Uninitialized),
            _ => Ok(()),
        }
    }

    pub(crate) fn configure_boot<I: CounterIo>(
        &mut self,
        io: &I,
        supported: bool,
    ) -> Result<(), CounterError> {
        if self.state != CounterState::Uninitialized {
            return Err(CounterError::AlreadyConfigured);
        }
        if !supported {
            self.state = CounterState::Unsupported;
            return Ok(());
        }

        let offset = io.read_counter()?;
        io.set_offset(offset)?;
        self.state = CounterState::Paused {
            offset,
            guest_counter: 0,
        };
        Ok(())
    }

    pub(crate) fn configure_restore<I: CounterIo>(
        &mut self,
        io: &I,
        saved_counter: u64,
        supported: bool,
    ) -> Result<(), CounterError> {
        if self.state != CounterState::Uninitialized {
            return Err(CounterError::AlreadyConfigured);
        }
        if !supported {
            return Err(CounterError::Unsupported);
        }

        let kvm_counter = io.read_counter()?;
        // The KVM UAPI exposes counters and offsets as u64 and kernel timer
        // arithmetic wraps in that domain.
        let offset = kvm_counter.wrapping_sub(saved_counter);
        io.set_offset(offset)?;
        self.state = CounterState::Paused {
            offset,
            guest_counter: saved_counter,
        };
        Ok(())
    }

    pub(crate) fn ensure_can_pause(&self) -> Result<(), CounterError> {
        match self.state {
            CounterState::Running { .. } => Ok(()),
            CounterState::Paused { .. } => Err(CounterError::AlreadyPaused),
            CounterState::Unsupported => Err(CounterError::Unsupported),
            CounterState::Uninitialized => Err(CounterError::Uninitialized),
            CounterState::Faulted => Err(CounterError::Faulted),
        }
    }

    /// Return the exact guest counter captured after every vCPU acknowledged pause.
    ///
    /// This is also the snapshotability check: unsupported, running,
    /// uninitialized, and faulted domains must never produce a snapshot.
    pub(crate) fn paused_counter(&self) -> Result<u64, CounterError> {
        match self.state {
            CounterState::Paused { guest_counter, .. } => Ok(guest_counter),
            CounterState::Running { .. } => Err(CounterError::NotPaused),
            CounterState::Unsupported => Err(CounterError::Unsupported),
            CounterState::Uninitialized => Err(CounterError::Uninitialized),
            CounterState::Faulted => Err(CounterError::Faulted),
        }
    }

    pub(crate) fn record_pause<I: CounterIo>(&mut self, io: &I) -> Result<(), CounterError> {
        match self.state {
            CounterState::Running { offset } => {
                let guest_counter = io.read_counter()?;
                self.state = CounterState::Paused {
                    offset,
                    guest_counter,
                };
                Ok(())
            }
            CounterState::Paused { .. } => Err(CounterError::AlreadyPaused),
            CounterState::Unsupported => Err(CounterError::Unsupported),
            CounterState::Uninitialized => Err(CounterError::Uninitialized),
            CounterState::Faulted => Err(CounterError::Faulted),
        }
    }

    pub(crate) fn prepare_resume<I: CounterIo>(&mut self, io: &I) -> Result<(), CounterError> {
        match self.state {
            CounterState::Paused {
                offset,
                guest_counter,
            } => {
                let current_counter = io.read_counter()?;
                // Preserve KVM's u64 counter arithmetic across rollover.
                let elapsed = current_counter.wrapping_sub(guest_counter);
                let new_offset = offset.wrapping_add(elapsed);
                io.set_offset(new_offset)?;
                self.state = CounterState::Paused {
                    offset: new_offset,
                    guest_counter,
                };
                Ok(())
            }
            // Fresh boot remains supported on hosts predating KVM_CAP_COUNTER_OFFSET.
            CounterState::Unsupported => Ok(()),
            CounterState::Running { .. } => Err(CounterError::NotPaused),
            CounterState::Uninitialized => Err(CounterError::Uninitialized),
            CounterState::Faulted => Err(CounterError::Faulted),
        }
    }

    pub(crate) fn mark_running(&mut self) -> Result<(), CounterError> {
        match self.state {
            CounterState::Paused { offset, .. } => {
                self.state = CounterState::Running { offset };
                Ok(())
            }
            CounterState::Unsupported => Ok(()),
            CounterState::Running { .. } => Err(CounterError::NotPaused),
            CounterState::Uninitialized => Err(CounterError::Uninitialized),
            CounterState::Faulted => Err(CounterError::Faulted),
        }
    }

    /// Prevent further operations after a lifecycle transition became ambiguous.
    pub(crate) fn mark_faulted(&mut self) {
        self.state = CounterState::Faulted;
    }
}

/// Extract the canonical counter from vCPU0.
///
/// Paused vCPU threads capture their states independently, so their counter
/// samples need not match. The VM-wide offset uses vCPU0 as the canonical
/// sample; KVM then ignores every per-vCPU CNTPCT/CNTVCT register replay for
/// the VM-wide counter domain.
pub(crate) fn canonical_saved_counter(states: &[VcpuState]) -> Result<u64, CounterError> {
    let boot_vcpu = states.first().ok_or(CounterError::NoVcpuState)?;
    unique_saved_counter(boot_vcpu, 0, SYS_CNTPCT_EL0)
}

fn unique_saved_counter(
    state: &VcpuState,
    vcpu_index: usize,
    register_id: u64,
) -> Result<u64, CounterError> {
    let mut counters = state
        .regs
        .iter()
        .filter(|register| register.id == register_id);
    let counter = counters.next().ok_or_else(|| match register_id {
        SYS_CNTPCT_EL0 => CounterError::MissingSavedCounter(vcpu_index),
        KVM_REG_ARM_TIMER_CNT => CounterError::MissingSavedVirtualCounter(vcpu_index),
        _ => unreachable!("only architectural counter registers are queried"),
    })?;
    if counters.next().is_some() {
        return Err(match register_id {
            SYS_CNTPCT_EL0 => CounterError::DuplicateSavedCounter(vcpu_index),
            KVM_REG_ARM_TIMER_CNT => CounterError::DuplicateSavedVirtualCounter(vcpu_index),
            _ => unreachable!("only architectural counter registers are queried"),
        });
    }
    Ok(counter.value::<u64, 8>())
}

/// Normalize serialized counter values to the exact point at which vCPUs paused.
///
/// KVM's counter keeps advancing after a vCPU thread acknowledges pause. Saving
/// its registers later would otherwise age every timer by the pause-to-save
/// delay. Preserve each vCPU's physical/virtual counter relationship while
/// moving both samples back by that delay. This changes only values already in
/// `VcpuState`; it does not add to or change the snapshot format.
pub(crate) fn normalize_snapshot_counters(
    states: &mut [VcpuState],
    paused_physical_counter: u64,
) -> Result<(), CounterError> {
    if states.is_empty() {
        return Err(CounterError::NoVcpuState);
    }

    for (vcpu_index, state) in states.iter_mut().enumerate() {
        let saved_physical = unique_saved_counter(state, vcpu_index, SYS_CNTPCT_EL0)?;
        unique_saved_counter(state, vcpu_index, KVM_REG_ARM_TIMER_CNT)?;
        let pause_to_save_delta = saved_physical.wrapping_sub(paused_physical_counter);

        for mut register in state.regs.iter_mut() {
            match register.id {
                SYS_CNTPCT_EL0 => register.set_value::<u64, 8>(paused_physical_counter),
                KVM_REG_ARM_TIMER_CNT => register.set_value::<u64, 8>(
                    register.value::<u64, 8>().wrapping_sub(pause_to_save_delta),
                ),
                _ => {}
            }
        }
    }
    Ok(())
}

/// Restore all vCPUs in the order required by the VM-wide counter API.
///
/// Every vCPU must be initialized before reading the KVM counter. The offset
/// must then be installed before SVE or any other saved register is replayed.
pub(crate) fn restore_vcpus_in_order<T, E>(
    vcpus: &mut [T],
    mut prepare: impl FnMut(usize, &mut T) -> Result<(), E>,
    install_counter: impl FnOnce(&[T]) -> Result<(), E>,
    mut finalize: impl FnMut(usize, &mut T) -> Result<(), E>,
    mut replay: impl FnMut(usize, &mut T) -> Result<(), E>,
) -> Result<(), E> {
    for (index, vcpu) in vcpus.iter_mut().enumerate() {
        prepare(index, vcpu)?;
    }
    install_counter(vcpus)?;
    for (index, vcpu) in vcpus.iter_mut().enumerate() {
        finalize(index, vcpu)?;
    }
    for (index, vcpu) in vcpus.iter_mut().enumerate() {
        replay(index, vcpu)?;
    }
    Ok(())
}

/// Error from the pause transition orchestrator.
pub(crate) enum PauseTransitionError<E> {
    /// Counter validation or capture failed and rollback succeeded.
    Counter(CounterError),
    /// Pausing the vCPUs failed before counter capture.
    Pause(E),
    /// Counter capture and the vCPU-resume rollback both failed.
    Rollback(CounterError, E),
}

/// Pause vCPUs before capturing their stable counter, with rollback on capture failure.
pub(crate) fn pause_in_order<E>(
    validate: impl FnOnce() -> Result<(), CounterError>,
    pause_vcpus: impl FnOnce() -> Result<(), E>,
    capture_counter: impl FnOnce() -> Result<(), CounterError>,
    rollback_vcpus: impl FnOnce() -> Result<(), E>,
    mark_faulted: impl FnOnce(),
) -> Result<(), PauseTransitionError<E>> {
    validate().map_err(PauseTransitionError::Counter)?;
    if let Err(pause_error) = pause_vcpus() {
        // A multi-vCPU pause failure can leave an unknowable mixture of
        // running and paused vCPUs. No later counter transition is safe.
        mark_faulted();
        return Err(PauseTransitionError::Pause(pause_error));
    }
    if let Err(counter_error) = capture_counter() {
        return match rollback_vcpus() {
            Ok(()) => Err(PauseTransitionError::Counter(counter_error)),
            Err(rollback_error) => {
                mark_faulted();
                Err(PauseTransitionError::Rollback(
                    counter_error,
                    rollback_error,
                ))
            }
        };
    }
    Ok(())
}

/// Error from the resume transition orchestrator.
pub(crate) enum ResumeTransitionError<E> {
    /// Preparing or completing the counter transition failed.
    Counter(CounterError),
    /// Resuming the vCPUs failed after the counter offset was installed.
    Resume(E),
}

/// Install the counter offset before any device kick or vCPU resume.
pub(crate) fn resume_in_order<E>(
    prepare_counter: impl FnOnce() -> Result<(), CounterError>,
    kick_devices: impl FnOnce(),
    resume_vcpus: impl FnOnce() -> Result<(), E>,
    mark_running: impl FnOnce() -> Result<(), CounterError>,
    mark_faulted: impl FnOnce(),
) -> Result<(), ResumeTransitionError<E>> {
    prepare_counter().map_err(ResumeTransitionError::Counter)?;
    kick_devices();
    if let Err(resume_error) = resume_vcpus() {
        // Some vCPUs may already be running. Retrying an offset transition
        // could move their counter backwards, so fail closed.
        mark_faulted();
        return Err(ResumeTransitionError::Resume(resume_error));
    }
    if let Err(counter_error) = mark_running() {
        mark_faulted();
        return Err(ResumeTransitionError::Counter(counter_error));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use serde_json::json;

    use super::*;
    use crate::arch::aarch64::regs::{Aarch64RegisterRef, Aarch64RegisterVec};

    // Independent copies of the Linux UAPI values. Test fixtures use these
    // numbers directly so a self-consistently wrong production constant does
    // not make counter normalization tests pass.
    const UAPI_SYS_CNTPCT_EL0: u64 = 0x6030_0000_0013_df01;
    const UAPI_KVM_REG_ARM_TIMER_CVAL: u64 = 0x6030_0000_0013_df02;
    const UAPI_KVM_REG_ARM_TIMER_CNT: u64 = 0x6030_0000_0013_df1a;

    #[derive(Default)]
    struct FakeCounterIo {
        reads: RefCell<VecDeque<Result<u64, &'static str>>>,
        offsets: RefCell<Vec<u64>>,
        set_error: RefCell<Option<&'static str>>,
    }

    impl FakeCounterIo {
        fn with_reads(reads: impl IntoIterator<Item = u64>) -> Self {
            Self {
                reads: RefCell::new(reads.into_iter().map(Ok).collect()),
                ..Default::default()
            }
        }

        fn push_read(&self, value: Result<u64, &'static str>) {
            self.reads.borrow_mut().push_back(value);
        }

        fn fail_next_set(&self, message: &'static str) {
            *self.set_error.borrow_mut() = Some(message);
        }
    }

    impl CounterIo for FakeCounterIo {
        fn read_counter(&self) -> Result<u64, CounterError> {
            self.reads
                .borrow_mut()
                .pop_front()
                .expect("test must provide every counter read")
                .map_err(CounterError::Injected)
        }

        fn set_offset(&self, offset: u64) -> Result<(), CounterError> {
            if let Some(message) = self.set_error.borrow_mut().take() {
                return Err(CounterError::Injected(message));
            }
            self.offsets.borrow_mut().push(offset);
            Ok(())
        }
    }

    fn state_with_counters(counters: &[u64]) -> VcpuState {
        let mut regs = Aarch64RegisterVec::default();
        for counter in counters {
            regs.push(Aarch64RegisterRef::new(
                UAPI_SYS_CNTPCT_EL0,
                &counter.to_le_bytes(),
            ));
        }
        VcpuState {
            regs,
            ..Default::default()
        }
    }

    fn state_with_counter_domain(physical: &[u64], virtual_: &[u64]) -> VcpuState {
        let mut state = state_with_counters(physical);
        for counter in virtual_ {
            state.regs.push(Aarch64RegisterRef::new(
                UAPI_KVM_REG_ARM_TIMER_CNT,
                &counter.to_le_bytes(),
            ));
        }
        state
    }

    fn saved_counter(state: &VcpuState, register_id: u64) -> u64 {
        state
            .regs
            .iter()
            .find(|register| register.id == register_id)
            .expect("test state must contain the requested counter")
            .value::<u64, 8>()
    }

    #[test]
    fn boot_and_initial_resume_freeze_pre_run_time() {
        let io = FakeCounterIo::with_reads([100]);
        let mut controller = CounterController::default();

        controller.configure_boot(&io, true).unwrap();
        assert_eq!(
            controller.state,
            CounterState::Paused {
                offset: 100,
                guest_counter: 0
            }
        );
        assert_eq!(*io.offsets.borrow(), [100]);

        io.push_read(Ok(40));
        controller.prepare_resume(&io).unwrap();
        assert_eq!(
            controller.state,
            CounterState::Paused {
                offset: 140,
                guest_counter: 0
            }
        );
        controller.mark_running().unwrap();
        assert_eq!(controller.state, CounterState::Running { offset: 140 });
        assert_eq!(*io.offsets.borrow(), [100, 140]);
    }

    #[test]
    fn restore_uses_wrapping_subtraction() {
        let io = FakeCounterIo::with_reads([5]);
        let mut controller = CounterController::default();

        controller.configure_restore(&io, 10, true).unwrap();
        assert_eq!(
            controller.state,
            CounterState::Paused {
                offset: 5_u64.wrapping_sub(10),
                guest_counter: 10
            }
        );
        assert_eq!(*io.offsets.borrow(), [5_u64.wrapping_sub(10)]);

        io.push_read(Ok(13));
        controller.prepare_resume(&io).unwrap();
        controller.mark_running().unwrap();
        assert_eq!(
            *io.offsets.borrow(),
            [
                5_u64.wrapping_sub(10),
                5_u64.wrapping_sub(10).wrapping_add(3)
            ]
        );
    }

    #[test]
    fn unsupported_host_boots_but_cannot_pause_or_restore() {
        let io = FakeCounterIo::default();
        let mut controller = CounterController::default();

        controller.configure_boot(&io, false).unwrap();
        assert_eq!(controller.state, CounterState::Unsupported);
        controller.prepare_resume(&io).unwrap();
        controller.mark_running().unwrap();
        assert!(matches!(
            controller.record_pause(&io),
            Err(CounterError::Unsupported)
        ));

        let mut restored = CounterController::default();
        assert!(matches!(
            restored.configure_restore(&io, 7, false),
            Err(CounterError::Unsupported)
        ));
    }

    #[test]
    fn pause_resume_is_repeatable_and_wraps() {
        let io = FakeCounterIo::with_reads([u64::MAX - 2]);
        let mut controller = CounterController::default();
        controller.configure_boot(&io, true).unwrap();

        io.push_read(Ok(0));
        controller.prepare_resume(&io).unwrap();
        controller.mark_running().unwrap();

        io.push_read(Ok(u64::MAX - 1));
        controller.record_pause(&io).unwrap();
        assert!(matches!(
            controller.record_pause(&io),
            Err(CounterError::AlreadyPaused)
        ));

        io.push_read(Ok(2));
        controller.prepare_resume(&io).unwrap();
        controller.mark_running().unwrap();
        assert!(matches!(
            controller.prepare_resume(&io),
            Err(CounterError::NotPaused)
        ));

        io.push_read(Ok(9));
        controller.record_pause(&io).unwrap();
        io.push_read(Ok(20));
        controller.prepare_resume(&io).unwrap();
        controller.mark_running().unwrap();

        assert_eq!(*io.offsets.borrow(), [u64::MAX - 2, u64::MAX - 2, 1, 12]);
    }

    #[test]
    fn backend_failures_do_not_commit_lifecycle_state() {
        let io = FakeCounterIo::with_reads([]);
        let mut controller = CounterController::default();
        io.push_read(Err("boot read"));
        assert!(matches!(
            controller.configure_boot(&io, true),
            Err(CounterError::Injected("boot read"))
        ));
        assert_eq!(controller.state, CounterState::Uninitialized);

        io.push_read(Ok(100));
        io.fail_next_set("boot set");
        assert!(matches!(
            controller.configure_boot(&io, true),
            Err(CounterError::Injected("boot set"))
        ));
        assert_eq!(controller.state, CounterState::Uninitialized);

        io.push_read(Ok(100));
        controller.configure_boot(&io, true).unwrap();
        io.push_read(Ok(0));
        controller.prepare_resume(&io).unwrap();
        controller.mark_running().unwrap();

        io.push_read(Err("pause read"));
        assert!(matches!(
            controller.record_pause(&io),
            Err(CounterError::Injected("pause read"))
        ));
        assert_eq!(controller.state, CounterState::Running { offset: 100 });

        io.push_read(Ok(10));
        controller.record_pause(&io).unwrap();
        io.push_read(Ok(20));
        io.fail_next_set("resume set");
        assert!(matches!(
            controller.prepare_resume(&io),
            Err(CounterError::Injected("resume set"))
        ));
        assert_eq!(
            controller.state,
            CounterState::Paused {
                offset: 100,
                guest_counter: 10
            }
        );
    }

    #[test]
    fn restore_backend_failures_do_not_commit_lifecycle_state() {
        let io = FakeCounterIo::with_reads([]);
        let mut controller = CounterController::default();

        io.push_read(Err("restore read"));
        assert!(matches!(
            controller.configure_restore(&io, 50, true),
            Err(CounterError::Injected("restore read"))
        ));
        assert_eq!(controller.state, CounterState::Uninitialized);

        io.push_read(Ok(100));
        io.fail_next_set("restore set");
        assert!(matches!(
            controller.configure_restore(&io, 50, true),
            Err(CounterError::Injected("restore set"))
        ));
        assert_eq!(controller.state, CounterState::Uninitialized);
    }

    #[test]
    fn uninitialized_and_faulted_controllers_fail_closed() {
        let io = FakeCounterIo::with_reads([100]);
        let mut controller = CounterController::default();

        assert!(matches!(
            controller.ensure_healthy(),
            Err(CounterError::Uninitialized)
        ));
        controller.configure_boot(&io, true).unwrap();
        controller.mark_faulted();
        assert!(matches!(
            controller.ensure_healthy(),
            Err(CounterError::Faulted)
        ));
        assert!(matches!(
            controller.prepare_resume(&io),
            Err(CounterError::Faulted)
        ));
        assert!(matches!(
            controller.record_pause(&io),
            Err(CounterError::Faulted)
        ));
        assert!(matches!(
            controller.paused_counter(),
            Err(CounterError::Faulted)
        ));
    }

    #[test]
    fn only_a_supported_paused_domain_is_snapshotable() {
        let io = FakeCounterIo::with_reads([100]);
        let mut controller = CounterController::default();
        assert!(matches!(
            controller.paused_counter(),
            Err(CounterError::Uninitialized)
        ));

        controller.configure_boot(&io, true).unwrap();
        assert_eq!(controller.paused_counter().unwrap(), 0);
        io.push_read(Ok(0));
        controller.prepare_resume(&io).unwrap();
        controller.mark_running().unwrap();
        assert!(matches!(
            controller.paused_counter(),
            Err(CounterError::NotPaused)
        ));

        let mut unsupported = CounterController::default();
        unsupported.configure_boot(&io, false).unwrap();
        assert!(matches!(
            unsupported.paused_counter(),
            Err(CounterError::Unsupported)
        ));
    }

    #[test]
    fn saved_counter_requires_exactly_one_vcpu0_register() {
        assert!(matches!(
            canonical_saved_counter(&[]),
            Err(CounterError::NoVcpuState)
        ));
        assert!(matches!(
            canonical_saved_counter(&[state_with_counters(&[])]),
            Err(CounterError::MissingSavedCounter(0))
        ));
        assert!(matches!(
            canonical_saved_counter(&[state_with_counters(&[1, 2])]),
            Err(CounterError::DuplicateSavedCounter(0))
        ));

        let states = [state_with_counters(&[11]), state_with_counters(&[99])];
        assert_eq!(canonical_saved_counter(&states).unwrap(), 11);
    }

    #[test]
    fn snapshot_counters_are_normalized_to_the_common_pause_point() {
        let mut states = [
            state_with_counter_domain(&[130], &[110]),
            state_with_counter_domain(&[135], &[105]),
        ];
        states[0].regs.push(Aarch64RegisterRef::new(
            UAPI_KVM_REG_ARM_TIMER_CVAL,
            &900_u64.to_le_bytes(),
        ));
        states[1].regs.push(Aarch64RegisterRef::new(
            UAPI_KVM_REG_ARM_TIMER_CVAL,
            &901_u64.to_le_bytes(),
        ));

        normalize_snapshot_counters(&mut states, 100).unwrap();

        assert_eq!(saved_counter(&states[0], UAPI_SYS_CNTPCT_EL0), 100);
        assert_eq!(saved_counter(&states[0], UAPI_KVM_REG_ARM_TIMER_CNT), 80);
        assert_eq!(saved_counter(&states[0], UAPI_KVM_REG_ARM_TIMER_CVAL), 900);
        assert_eq!(saved_counter(&states[1], UAPI_SYS_CNTPCT_EL0), 100);
        assert_eq!(saved_counter(&states[1], UAPI_KVM_REG_ARM_TIMER_CNT), 70);
        assert_eq!(saved_counter(&states[1], UAPI_KVM_REG_ARM_TIMER_CVAL), 901);
    }

    #[test]
    fn snapshot_counter_normalization_rejects_incomplete_domains() {
        assert!(matches!(
            normalize_snapshot_counters(&mut [], 0),
            Err(CounterError::NoVcpuState)
        ));

        let mut missing_virtual = [state_with_counter_domain(&[10], &[])];
        assert!(matches!(
            normalize_snapshot_counters(&mut missing_virtual, 5),
            Err(CounterError::MissingSavedVirtualCounter(0))
        ));

        let mut duplicate_virtual = [state_with_counter_domain(&[10], &[8, 9])];
        assert!(matches!(
            normalize_snapshot_counters(&mut duplicate_virtual, 5),
            Err(CounterError::DuplicateSavedVirtualCounter(0))
        ));
    }

    #[test]
    fn malformed_saved_counter_is_rejected_during_deserialization() {
        let malformed = json!([[UAPI_SYS_CNTPCT_EL0], [1, 2, 3, 4]]);
        serde_json::from_value::<Aarch64RegisterVec>(malformed).unwrap_err();
    }

    #[test]
    fn counter_offset_ioctl_matches_the_linux_uapi() {
        assert_eq!(SYS_CNTPCT_EL0, UAPI_SYS_CNTPCT_EL0);
        // The virtual timer IDs intentionally preserve KVM's swapped UAPI,
        // rather than using the registers' architectural encodings.
        assert_eq!(KVM_REG_ARM_TIMER_CVAL, UAPI_KVM_REG_ARM_TIMER_CVAL);
        assert_eq!(KVM_REG_ARM_TIMER_CNT, UAPI_KVM_REG_ARM_TIMER_CNT);
        assert_eq!(TEST_KVM_GET_ONE_REG(), 0x4010_aeab);
        assert_eq!(TEST_KVM_GET_ONE_REG(), 1_074_835_115);
        assert_eq!(KVM_ARM_SET_COUNTER_OFFSET(), 0x4010_aeb5);
        assert_eq!(KVM_ARM_SET_COUNTER_OFFSET(), 1_074_835_125);
    }

    #[test]
    fn counter_is_installed_before_any_register_replay() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let mut vcpus = [0_u8, 1_u8];

        restore_vcpus_in_order(
            &mut vcpus,
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("prepare-{index}"));
                    Ok::<_, ()>(())
                }
            },
            {
                let trace = trace.clone();
                move |_| {
                    trace.borrow_mut().push("counter".to_string());
                    Ok::<_, ()>(())
                }
            },
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("finalize-{index}"));
                    Ok::<_, ()>(())
                }
            },
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("replay-{index}"));
                    Ok::<_, ()>(())
                }
            },
        )
        .unwrap();

        assert_eq!(
            *trace.borrow(),
            [
                "prepare-0",
                "prepare-1",
                "counter",
                "finalize-0",
                "finalize-1",
                "replay-0",
                "replay-1"
            ]
        );
    }

    #[test]
    fn restore_stops_before_replay_when_counter_install_fails() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let mut vcpus = [0_u8, 1_u8];

        let result = restore_vcpus_in_order(
            &mut vcpus,
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("prepare-{index}"));
                    Ok::<_, &'static str>(())
                }
            },
            {
                let trace = trace.clone();
                move |_| {
                    trace.borrow_mut().push("counter".to_string());
                    Err("counter ioctl")
                }
            },
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("finalize-{index}"));
                    Ok::<_, &'static str>(())
                }
            },
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("replay-{index}"));
                    Ok::<_, &'static str>(())
                }
            },
        );

        assert_eq!(result.unwrap_err(), "counter ioctl");
        assert_eq!(*trace.borrow(), ["prepare-0", "prepare-1", "counter"]);
    }

    #[test]
    fn restore_stops_before_counter_when_vcpu_prepare_fails() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let mut vcpus = [0_u8, 1_u8];

        let result = restore_vcpus_in_order(
            &mut vcpus,
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("prepare-{index}"));
                    if index == 1 { Err("vcpu init") } else { Ok(()) }
                }
            },
            {
                let trace = trace.clone();
                move |_| {
                    trace.borrow_mut().push("counter".to_string());
                    Ok::<_, &'static str>(())
                }
            },
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("finalize-{index}"));
                    Ok::<_, &'static str>(())
                }
            },
            {
                let trace = trace.clone();
                move |index, _| {
                    trace.borrow_mut().push(format!("replay-{index}"));
                    Ok::<_, &'static str>(())
                }
            },
        );

        assert_eq!(result.unwrap_err(), "vcpu init");
        assert_eq!(*trace.borrow(), ["prepare-0", "prepare-1"]);
    }

    #[test]
    fn vcpu_pause_failure_never_captures_the_counter() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let result = pause_in_order(
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("validate");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("pause");
                    Err("pause failed")
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("capture");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("rollback");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || trace.borrow_mut().push("fault")
            },
        );

        assert!(matches!(
            result,
            Err(PauseTransitionError::Pause("pause failed"))
        ));
        assert_eq!(*trace.borrow(), ["validate", "pause", "fault"]);
    }

    #[test]
    fn pause_validation_failure_has_no_lifecycle_side_effects() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let result = pause_in_order(
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("validate");
                    Err(CounterError::Unsupported)
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("pause");
                    Ok::<_, ()>(())
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("capture");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("rollback");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || trace.borrow_mut().push("fault")
            },
        );

        assert!(matches!(
            result,
            Err(PauseTransitionError::Counter(CounterError::Unsupported))
        ));
        assert_eq!(*trace.borrow(), ["validate"]);
    }

    #[test]
    fn counter_capture_failure_rolls_back_or_faults() {
        let faulted = Rc::new(RefCell::new(false));
        let rollback_ok = pause_in_order(
            || Ok(()),
            || Ok::<_, &'static str>(()),
            || Err(CounterError::Injected("capture")),
            || Ok(()),
            || unreachable!("successful rollback must not fault the controller"),
        );
        assert!(matches!(
            rollback_ok,
            Err(PauseTransitionError::Counter(CounterError::Injected(
                "capture"
            )))
        ));

        let rollback_failed = pause_in_order(
            || Ok(()),
            || Ok(()),
            || Err(CounterError::Injected("capture")),
            || Err("resume failed"),
            {
                let faulted = faulted.clone();
                move || *faulted.borrow_mut() = true
            },
        );
        assert!(matches!(
            rollback_failed,
            Err(PauseTransitionError::Rollback(
                CounterError::Injected("capture"),
                "resume failed"
            ))
        ));
        assert!(*faulted.borrow());
    }

    #[test]
    fn resume_installs_offset_before_device_kick_and_vcpu_resume() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let result = resume_in_order(
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("counter");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || trace.borrow_mut().push("devices")
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("vcpus");
                    Ok::<_, ()>(())
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("running");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || trace.borrow_mut().push("fault")
            },
        );

        assert!(result.is_ok());
        assert_eq!(*trace.borrow(), ["counter", "devices", "vcpus", "running"]);
    }

    #[test]
    fn counter_prepare_failure_never_kicks_devices_or_resumes_vcpus() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let result = resume_in_order(
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("counter");
                    Err(CounterError::Injected("set offset"))
                }
            },
            {
                let trace = trace.clone();
                move || trace.borrow_mut().push("devices")
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("vcpus");
                    Ok::<_, ()>(())
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("running");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || trace.borrow_mut().push("fault")
            },
        );

        assert!(matches!(
            result,
            Err(ResumeTransitionError::Counter(CounterError::Injected(
                "set offset"
            )))
        ));
        assert_eq!(*trace.borrow(), ["counter"]);
    }

    #[test]
    fn vcpu_resume_failure_faults_after_the_counter_transition() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let result = resume_in_order(
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("counter");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || trace.borrow_mut().push("devices")
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("vcpus");
                    Err("resume failed")
                }
            },
            {
                let trace = trace.clone();
                move || {
                    trace.borrow_mut().push("running");
                    Ok(())
                }
            },
            {
                let trace = trace.clone();
                move || trace.borrow_mut().push("fault")
            },
        );

        assert!(matches!(
            result,
            Err(ResumeTransitionError::Resume("resume failed"))
        ));
        assert_eq!(*trace.borrow(), ["counter", "devices", "vcpus", "fault"]);
    }
}
