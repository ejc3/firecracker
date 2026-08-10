# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test UFFD related functionality when resuming from snapshot."""

import hashlib
import os
import re
from pathlib import Path

import pytest
import requests

from framework.artifacts import GUEST_KERNEL_DEFAULT, pin_guest_kernel
from framework.utils import Timeout, check_output

MINOR_BACKING_NAME = "memfd:firecracker_uffd_minor_test"
MINOR_MARKER = "/dev/shm/uffd-minor-marker"
MINOR_MARKER_SIZE = 8 * 1024 * 1024


def _sha256_file(path):
    """Hash a host file without loading the full VM snapshot into Python memory."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _minor_backing_smaps(pid):
    """Sum memory accounting for Firecracker's named minor-backing mappings."""
    totals = {"Shared_Clean": 0, "Shared_Dirty": 0, "Private_Dirty": 0}
    mapping_count = 0
    in_backing_mapping = False

    for line in Path(f"/proc/{pid}/smaps").read_text(encoding="utf-8").splitlines():
        if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
            in_backing_mapping = MINOR_BACKING_NAME in line
            mapping_count += int(in_backing_mapping)
            continue
        if not in_backing_mapping:
            continue
        match = re.match(
            r"^(Shared_Clean|Shared_Dirty|Private_Dirty):\s+(\d+) kB$", line
        )
        if match:
            totals[match.group(1)] += int(match.group(2))

    assert mapping_count > 0, f"no {MINOR_BACKING_NAME} mapping in Firecracker {pid}"
    return totals


@pytest.fixture(scope="function", name="snapshot")
def snapshot_fxt(microvm_factory, guest_kernel, rootfs):
    """Create a snapshot of a microVM."""

    basevm = microvm_factory.build(guest_kernel, rootfs)
    basevm.spawn()
    basevm.basic_config(vcpu_count=2, mem_size_mib=256)
    basevm.add_net_iface()

    # Add a memory balloon.
    basevm.api.balloon.put(
        amount_mib=0, deflate_on_oom=True, stats_polling_interval_s=0
    )

    basevm.start()

    # Create base snapshot.
    snapshot = basevm.snapshot_full()
    basevm.kill()

    yield snapshot


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_bad_socket_path(uvm, snapshot):
    """
    Test error scenario when socket path does not exist.
    """
    vm = uvm
    vm.spawn()
    jailed_vmstate = vm.create_jailed_resource(snapshot.vmstate)

    expected_msg = re.escape(
        "Load snapshot error: Failed to restore from snapshot: Failed to load guest "
        "memory: Error creating guest memory from uffd: Failed to connect to UDS Unix stream: No "
        "such file or directory (os error 2)"
    )
    with pytest.raises(RuntimeError, match=expected_msg):
        vm.api.snapshot_load.put(
            mem_backend={"backend_type": "Uffd", "backend_path": "inexistent"},
            snapshot_path=jailed_vmstate,
        )

    vm.mark_killed()


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_unbinded_socket(uvm, snapshot):
    """
    Test error scenario when PF handler has not yet called bind on socket.
    """
    vm = uvm
    vm.spawn()

    jailed_vmstate = vm.create_jailed_resource(snapshot.vmstate)
    socket_path = os.path.join(vm.path, "firecracker-uffd.sock")
    check_output("touch {}".format(socket_path))
    jailed_sock_path = vm.create_jailed_resource(socket_path)

    expected_msg = re.escape(
        "Load snapshot error: Failed to restore from snapshot: Failed to load guest "
        "memory: Error creating guest memory from uffd: Failed to connect to UDS Unix stream: "
        "Connection refused (os error 111)"
    )
    with pytest.raises(RuntimeError, match=expected_msg):
        vm.api.snapshot_load.put(
            mem_backend={"backend_type": "Uffd", "backend_path": jailed_sock_path},
            snapshot_path=jailed_vmstate,
        )

    vm.mark_killed()


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_valid_handler(uvm, snapshot):
    """
    Test valid uffd handler scenario.
    """
    vm = uvm
    vm.memory_monitor = None
    vm.spawn()
    vm.restore_from_snapshot(snapshot, resume=True, uffd_handler_name="on_demand")

    # Inflate balloon.
    vm.api.balloon.patch(amount_mib=200)

    # Verify if the restored guest works.
    vm.ssh.check_output("true")

    # Deflate balloon.
    vm.api.balloon.patch(amount_mib=0)

    # Verify if the restored guest works.
    vm.ssh.check_output("true")


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_minor_continue_shares_backing_and_cows(microvm_factory, guest_kernel, rootfs):
    """Resolve real MINOR faults and prove shared backing to private-dirty COW."""
    basevm = microvm_factory.build(guest_kernel, rootfs)
    basevm.spawn()
    basevm.basic_config(vcpu_count=1, mem_size_mib=256)
    basevm.add_net_iface()
    basevm.start()

    basevm.ssh.check_output(
        f"yes A | tr -d '\\n' | head -c {MINOR_MARKER_SIZE} > {MINOR_MARKER}"
    )
    expected_marker_hash = hashlib.sha256(b"A" * MINOR_MARKER_SIZE).hexdigest()
    _, marker_hash, _ = basevm.ssh.check_output(f"sha256sum {MINOR_MARKER}")
    assert marker_hash.split()[0] == expected_marker_hash
    _, marker_size, _ = basevm.ssh.check_output(f"stat -c %s {MINOR_MARKER}")
    assert int(marker_size) == MINOR_MARKER_SIZE
    snapshot = basevm.snapshot_full()
    basevm.kill()
    expected_backing_hash = _sha256_file(snapshot.mem)

    vm = microvm_factory.build()
    vm.memory_monitor = None
    vm.spawn()
    vm.restore_from_snapshot(
        snapshot,
        resume=True,
        uffd_handler_name="minor",
        uffd_backend_type="UffdMinor",
    )

    backing_memfd = vm.uffd_handler.backing_memfd_path
    assert _sha256_file(backing_memfd) == expected_backing_hash
    accounting_before_read = _minor_backing_smaps(vm.firecracker_pid)

    _, restored_marker_hash, _ = vm.ssh.check_output(f"sha256sum {MINOR_MARKER}")
    assert restored_marker_hash.split()[0] == expected_marker_hash
    assert "MINOR_CONTINUE_OK" in vm.uffd_handler.log_data

    accounting_after_read = _minor_backing_smaps(vm.firecracker_pid)
    shared_before_read = (
        accounting_before_read["Shared_Clean"] + accounting_before_read["Shared_Dirty"]
    )
    shared_after_read = (
        accounting_after_read["Shared_Clean"] + accounting_after_read["Shared_Dirty"]
    )
    marker_size_kib = MINOR_MARKER_SIZE // 1024
    page_size_kib = os.sysconf("SC_PAGE_SIZE") // 1024
    assert shared_after_read - shared_before_read >= marker_size_kib - page_size_kib, (
        accounting_before_read,
        accounting_after_read,
    )

    vm.ssh.check_output(
        f"yes B | tr -d '\\n' | head -c {MINOR_MARKER_SIZE} | "
        f"dd of={MINOR_MARKER} bs=1M conv=notrunc status=none"
    )
    expected_changed_marker_hash = hashlib.sha256(b"B" * MINOR_MARKER_SIZE).hexdigest()
    _, changed_marker_hash, _ = vm.ssh.check_output(f"sha256sum {MINOR_MARKER}")
    assert changed_marker_hash.split()[0] == expected_changed_marker_hash
    _, changed_marker_size, _ = vm.ssh.check_output(f"stat -c %s {MINOR_MARKER}")
    assert int(changed_marker_size) == MINOR_MARKER_SIZE

    accounting_after_write = _minor_backing_smaps(vm.firecracker_pid)
    assert (
        accounting_after_write["Private_Dirty"] - accounting_after_read["Private_Dirty"]
        >= marker_size_kib - page_size_kib
    ), (accounting_after_read, accounting_after_write)
    assert _sha256_file(backing_memfd) == expected_backing_hash


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_malicious_handler(uvm, snapshot):
    """
    Test malicious uffd handler scenario.

    The page fault handler panics when receiving a page fault,
    so no events are handled and snapshot memory regions cannot be
    loaded into memory. In this case, Firecracker is designed to freeze,
    instead of silently switching to having the kernel handle page
    faults, so that it becomes obvious that something went wrong.
    """

    vm = uvm
    vm.memory_monitor = None
    vm.spawn()

    # We expect Firecracker to freeze while resuming from a snapshot
    # due to the malicious handler's unavailability.
    try:
        with Timeout(seconds=30):
            vm.restore_from_snapshot(
                snapshot, resume=True, uffd_handler_name="malicious"
            )
            assert False, "Firecracker should freeze"
    except (TimeoutError, requests.exceptions.ReadTimeout):
        vm.uffd_handler.mark_killed()
