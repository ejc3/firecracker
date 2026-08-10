# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests ensuring codebase style compliance for the OpenAPI specification."""

from pathlib import Path

from openapi_spec_validator import validate
from openapi_spec_validator.readers import read_from_filename


def validate_swagger(swagger_spec):
    """Fail if OpenAPI spec is not followed."""
    spec_dict, _ = read_from_filename(swagger_spec)
    validate(spec_dict)


def test_firecracker_swagger():
    """
    Test that Firecracker swagger specification is valid.
    """
    swagger_spec = Path("../src/firecracker/swagger/firecracker.yaml")
    validate_swagger(swagger_spec)


def test_memory_backend_enum_matches_runtime():
    """Ensure every runtime snapshot-memory backend is exposed by OpenAPI."""
    swagger_spec = Path("../src/firecracker/swagger/firecracker.yaml")
    spec_dict, _ = read_from_filename(swagger_spec)

    backend_types = spec_dict["definitions"]["MemoryBackend"]["properties"][
        "backend_type"
    ]["enum"]
    assert backend_types == ["File", "Uffd", "UffdMinor"]
