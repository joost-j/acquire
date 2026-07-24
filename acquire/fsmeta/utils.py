"""Helpers shared by the filesystem specific enumerators."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, BinaryIO, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator

log = logging.getLogger(__name__)

T = TypeVar("T")

# Every filesystem here keeps its trees far shallower than this, so a deeper one means the tree is
# corrupt or we lost track of where we are
MAX_TREE_DEPTH = 16


def walk_tree(
    root: T,
    children: Callable[[T], Iterable[T]],
    *,
    key: Callable[[T], Hashable] | None = None,
    visited: set[Hashable] | None = None,
    max_depth: int = MAX_TREE_DEPTH,
    name: str = "tree",
) -> int:
    """Walk a tree from ``root``, visiting every node exactly once.

    The node itself is collected by ``children``, which is called once per node and returns the nodes
    below it. Leaves simply return nothing.

    Args:
        root: The node to start at.
        children: Returns the children of a node. This is also where the node itself is collected.
        key: Maps a node onto the value that identifies it, defaults to the node itself.
        visited: A shared set of keys, so trees that share nodes are only walked once.
        max_depth: The depth at which to stop descending.
        name: Used in the warning logged when the tree turns out deeper than expected.

    Returns:
        The number of nodes that were visited.
    """
    if visited is None:
        visited = set()

    stack = [(root, 0)]
    count = 0

    while stack:
        node, depth = stack.pop()

        node_key = key(node) if key else node
        if node_key in visited:
            continue

        visited.add(node_key)
        count += 1

        # Resolved up front, and not lazily, because a node is collected by the very call that returns
        # its children - even when the depth check below gives up on descending any further
        below = list(children(node))

        if depth >= max_depth:
            log.warning("%s deeper than expected, stopping at %r", name, node_key)
            continue

        stack.extend((child, depth + 1) for child in below)

    return count


def iter_records(buf: bytes, size: int, count: int | None = None) -> Iterator[bytes]:
    """Iterate over the fixed size records in a buffer, stopping at a truncated tail.

    Args:
        buf: The buffer to read the records from.
        size: The size of a single record.
        count: The number of records to read, defaults to as many as fit in the buffer.
    """
    if count is None:
        count = len(buf) // size

    for index in range(count):
        record = buf[index * size : (index + 1) * size]
        if len(record) < size:
            return

        yield record


def size_of(fh: BinaryIO) -> int | None:
    """Determine the size of a disk or volume.

    Containers usually carry a ``size``, but a raw local disk can report ``None`` for it, in which case
    seeking to the end is the only way to find out.
    """
    size = getattr(fh, "size", None)
    if size:
        return size

    try:
        offset = fh.tell()
        size = fh.seek(0, io.SEEK_END)
        fh.seek(offset)
    except Exception:
        return None

    return size or None
