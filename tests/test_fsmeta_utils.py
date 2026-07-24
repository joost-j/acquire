from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from dissect.util.stream import RunlistStream

from acquire.fsmeta import file_runs
from acquire.fsmeta.utils import iter_records, size_of, walk_tree

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_walk_tree_visits_every_node() -> None:
    tree = {"root": ["a", "b"], "a": ["c"], "b": ["d"], "c": [], "d": []}
    seen = []

    def children(node: str) -> list[str]:
        seen.append(node)
        return tree[node]

    assert walk_tree("root", children) == 5
    assert sorted(seen) == ["a", "b", "c", "d", "root"]


def test_walk_tree_visits_shared_nodes_once() -> None:
    """Copy on write filesystems reach the same node through several parents."""
    tree = {"root": ["a", "b"], "a": ["shared"], "b": ["shared"], "shared": []}
    seen = []

    def children(node: str) -> list[str]:
        seen.append(node)
        return tree[node]

    assert walk_tree("root", children) == 4
    assert seen.count("shared") == 1


def test_walk_tree_shares_visited_between_trees() -> None:
    tree = {"first": ["shared"], "second": ["shared"], "shared": []}
    seen = []

    def children(node: str) -> list[str]:
        seen.append(node)
        return tree[node]

    visited = set()

    assert walk_tree("first", children, visited=visited) == 2
    # The second tree only adds its own root, the shared node is already collected
    assert walk_tree("second", children, visited=visited) == 1
    assert seen == ["first", "shared", "second"]


def test_walk_tree_stops_at_max_depth() -> None:
    seen = []

    def children(node: int) -> list[int]:
        seen.append(node)
        return [node + 1]

    assert walk_tree(0, children, max_depth=3) == 4
    # The node at the boundary is still collected, it is only never descended into
    assert seen == [0, 1, 2, 3]


def test_walk_tree_collects_lazy_children() -> None:
    """A generator only collects its node once it is iterated, which the depth cap must not skip."""
    seen = []

    def children(node: int) -> Iterator[int]:
        seen.append(node)
        yield node + 1

    assert walk_tree(0, children, max_depth=1) == 2
    assert seen == [0, 1]


def test_walk_tree_uses_key() -> None:
    node = MagicMock(address=1, children=[])
    same = MagicMock(address=1, children=[])
    root = MagicMock(address=0, children=[node, same])

    count = walk_tree(root, lambda node: node.children, key=lambda node: node.address)

    assert count == 2


@pytest.mark.parametrize(
    ("buf", "size", "count", "expected"),
    [
        (b"aabbcc", 2, None, [b"aa", b"bb", b"cc"]),
        # A truncated tail is dropped rather than yielded short
        (b"aabbc", 2, None, [b"aa", b"bb"]),
        (b"aabbc", 2, 3, [b"aa", b"bb"]),
        # Only the requested amount is yielded, even if more fits
        (b"aabbcc", 2, 2, [b"aa", b"bb"]),
        (b"", 2, None, []),
    ],
)
def test_iter_records(buf: bytes, size: int, count: int | None, expected: list[bytes]) -> None:
    assert list(iter_records(buf, size, count)) == expected


def test_file_runs_of_a_runlist_stream() -> None:
    volume = io.BytesIO(b"\x00" * 0x10000)
    fh = RunlistStream(volume, [(2, 1), (8, 2)], size=3 * 4096, block_size=4096)

    runs = file_runs(fh)

    assert runs is not None
    assert list(runs.items()) == [(volume, [(2 * 4096, 4096), (8 * 4096, 2 * 4096)])]


def test_file_runs_skips_sparse_runs() -> None:
    volume = io.BytesIO(b"\x00" * 0x10000)
    fh = RunlistStream(volume, [(None, 4), (2, 1)], size=5 * 4096, block_size=4096)

    assert list(file_runs(fh).items()) == [(volume, [(2 * 4096, 4096)])]


def test_file_runs_of_data_that_is_nowhere_on_disk() -> None:
    """Resident, inline and compressed data, and anything from a virtual filesystem."""
    assert file_runs(io.BytesIO(b"resident data")) is None
    assert file_runs(RunlistStream(io.BytesIO(), [], size=0, block_size=4096)) is None


def test_size_of_prefers_attribute() -> None:
    assert size_of(MagicMock(size=1337)) == 1337


def test_size_of_falls_back_to_seeking() -> None:
    fh = io.BytesIO(b"\x00" * 512)
    fh.seek(128)

    assert size_of(fh) == 512
    # The position of the stream is left alone
    assert fh.tell() == 128


def test_size_of_unknown() -> None:
    fh = MagicMock(size=None)
    fh.seek.side_effect = OSError("not seekable")

    assert size_of(fh) is None
