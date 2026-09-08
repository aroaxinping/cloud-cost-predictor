"""Tests for pricing module."""
from src.pricing import SIZE_TO_EC2, DOWNSIZE_TO


def test_all_ec2_types_have_downsize_path():
    """Every mapped EC2 type except the smallest should have a downsize target."""
    smallest = "t3.small"
    for ec2_type in SIZE_TO_EC2.values():
        if ec2_type != smallest:
            assert ec2_type in DOWNSIZE_TO, f"{ec2_type} has no downsize target"


def test_downsize_chain_terminates():
    """Following the downsize chain should always reach a type not in DOWNSIZE_TO."""
    for start in DOWNSIZE_TO:
        visited = set()
        current = start
        while current in DOWNSIZE_TO:
            assert current not in visited, f"Cycle detected at {current}"
            visited.add(current)
            current = DOWNSIZE_TO[current]


def test_size_mapping_covers_common_sizes():
    expected = {"t3.small", "t3.medium", "m5.large", "m5.xlarge", "r5.large"}
    actual = set(SIZE_TO_EC2.values())
    assert expected.issubset(actual)
