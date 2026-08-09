# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that validates that seccompiler filters work as expected"""

import json
import platform
import resource
from pathlib import Path

import pytest
import seccomp

from framework import utils

ARCH = platform.machine()
KVM_GET_ONE_REG = 1_074_835_115
KVM_ARM_SET_COUNTER_OFFSET = 1_074_835_125


def _ioctl_rules_for_request(fc_filter, request):
    """Return every thread and ioctl rule that permits the request."""

    return [
        (thread, rule)
        for thread, thread_filter in fc_filter.items()
        for rule in thread_filter["filter"]
        if rule.get("syscall") == "ioctl"
        and any(argument.get("val") == request for argument in rule.get("args", []))
    ]


@pytest.fixture
def bin_test_syscall(tmp_path):
    """Build the test_syscall binary."""
    test_syscall_bin = tmp_path / "test_syscall"
    compile_cmd = f"musl-gcc -static host_tools/test_syscalls.c -o {test_syscall_bin}"
    utils.check_output(compile_cmd)
    assert test_syscall_bin.exists()
    yield test_syscall_bin.resolve()


def test_validate_filter(seccompiler, bin_test_syscall, monkeypatch, tmp_path):
    """Assert that the seccomp filter matches the JSON description."""

    fc_filter_path = Path(f"../resources/seccomp/{ARCH}-unknown-linux-musl.json")
    fc_filter = json.loads(fc_filter_path.read_text(encoding="ascii"))

    # cd to a tmp dir because we may generate a bunch of intermediate files
    monkeypatch.chdir(tmp_path)
    # prevent coredumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    seccompiler.compile(fc_filter, split_output=True)

    # With split_output=True, individual .bpf files are created for each thread
    arch = seccomp.Arch.X86_64 if ARCH == "x86_64" else seccomp.Arch.AARCH64
    for thread, filter_data in fc_filter.items():
        filter_path = Path(f"{thread}.bpf")
        # The individual files should already exist from the split output
        assert (
            filter_path.exists()
        ), f"Expected {filter_path} to be created by seccompiler"

        # for each rule, run the helper program and execute a syscall
        for rule in filter_data["filter"]:
            print(filter_path, rule)
            syscall = rule["syscall"]
            # this one cannot be called directly
            if syscall in ["rt_sigreturn"]:
                continue
            syscall_id = seccomp.resolve_syscall(arch, syscall)
            cmd = f"{bin_test_syscall} {filter_path} {syscall_id}"
            if "args" not in rule:
                # syscall should be allowed with any arguments and exit 0
                assert utils.run_cmd(cmd).returncode == 0
            else:
                allowed_args = [0] * 4
                # if we call it with allowed args, it should exit 0
                for arg in rule["args"]:
                    allowed_args[arg["index"]] = arg["val"]
                allowed_str = " ".join(str(x) for x in allowed_args)
                assert utils.run_cmd(f"{cmd} {allowed_str}").returncode == 0
                # for each allowed arg try a different number
                for arg in rule["args"]:
                    bad_args = allowed_args.copy()
                    if isinstance(arg["op"], dict) and "masked_eq" in arg["op"]:
                        # For masked_eq, flip the mask bit to violate the check
                        bad_args[arg["index"]] = str(
                            arg["val"] ^ arg["op"]["masked_eq"]
                        )
                    else:
                        # We just add 1000000 to the allowed arg and assume it
                        # is not something we allow in another rule. While not
                        # perfect it works in practice.
                        bad_args[arg["index"]] = str(arg["val"] + 1_000_000)
                    unallowed_str = " ".join(str(x) for x in bad_args)
                    outcome = utils.run_cmd(f"{cmd} {unallowed_str}")
                    # if we call it with unallowed args, it should exit 159
                    # 159 = 128 (abnormal termination) + 31 (SIGSYS)
                    assert outcome.returncode == 159


def test_ioctl_rule_search_scans_every_thread():
    """Find matching ioctl rules outside the VMM thread."""

    test_request = 0xA5A5A5A5
    vmm_rule = {
        "syscall": "ioctl",
        "args": [{"index": 1, "type": "dword", "op": "eq", "val": test_request}],
    }
    api_rule = {
        "syscall": "ioctl",
        "args": [{"index": 1, "type": "dword", "op": "eq", "val": test_request}],
    }
    fc_filter = {
        "vmm": {"filter": [vmm_rule]},
        "api": {"filter": [api_rule]},
        "vcpu": {"filter": []},
    }

    assert _ioctl_rules_for_request(fc_filter, test_request) == [
        ("vmm", vmm_rule),
        ("api", api_rule),
    ]


def test_counter_ioctls_have_exact_vmm_rules():
    """Allow only the exact Arm counter ioctl requests required by the VMM."""

    fc_filter_path = Path(f"../resources/seccomp/{ARCH}-unknown-linux-musl.json")
    fc_filter = json.loads(fc_filter_path.read_text(encoding="ascii"))
    offset_rules = _ioctl_rules_for_request(fc_filter, KVM_ARM_SET_COUNTER_OFFSET)
    get_one_reg_rules = _ioctl_rules_for_request(fc_filter, KVM_GET_ONE_REG)
    vmm_ioctl_rules = [
        rule for rule in fc_filter["vmm"]["filter"] if rule.get("syscall") == "ioctl"
    ]

    # An unscoped or masked request rule would make the two exact additions
    # ineffective even if they were present in the JSON.
    assert all(
        any(
            argument.get("index") == 1 and argument.get("op") == "eq"
            for argument in rule.get("args", [])
        )
        for rule in vmm_ioctl_rules
    )

    if ARCH == "aarch64":
        assert offset_rules == [
            (
                "vmm",
                {
                    "syscall": "ioctl",
                    "args": [
                        {
                            "index": 1,
                            "type": "dword",
                            "op": "eq",
                            "val": KVM_ARM_SET_COUNTER_OFFSET,
                            "comment": "KVM_ARM_SET_COUNTER_OFFSET",
                        }
                    ],
                },
            )
        ]
        assert get_one_reg_rules == [
            (
                "vmm",
                {
                    "syscall": "ioctl",
                    "args": [
                        {
                            "index": 1,
                            "type": "dword",
                            "op": "eq",
                            "val": KVM_GET_ONE_REG,
                            "comment": (
                                "KVM_GET_ONE_REG, used to freeze the Arm counter while "
                                "paused"
                            ),
                        }
                    ],
                },
            ),
            (
                "vcpu",
                {
                    "syscall": "ioctl",
                    "args": [
                        {
                            "index": 1,
                            "type": "dword",
                            "op": "eq",
                            "val": KVM_GET_ONE_REG,
                            "comment": "KVM_GET_ONE_REG",
                        }
                    ],
                },
            ),
        ]
    else:
        assert offset_rules == []
        assert get_one_reg_rules == []
