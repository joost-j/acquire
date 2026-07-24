from __future__ import annotations

import logging
from bisect import bisect_right
from typing import TYPE_CHECKING

from dissect.btrfs.c_btrfs import c_btrfs
from dissect.btrfs.stream import _get_stripe_read_info

from acquire.fsmeta.collector import MetadataRuns
from acquire.fsmeta.utils import iter_records, size_of, walk_tree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dissect.btrfs.btrfs import Btrfs
    from dissect.btrfs.stream import Chunk, ExtentStream
    from dissect.target.filesystems.btrfs import BtrfsFilesystem

log = logging.getLogger(__name__)

# Btrfs keeps up to three superblock copies, at 64KiB, 64MiB and 256GiB
SUPERBLOCK_OFFSETS = (0x10000, 0x4000000, 0x4000000000)
SUPERBLOCK_SIZE = 0x1000

# Trees that are needed to traverse the filesystem. The extent and checksum trees are deliberately
# left out: they carry no path or inode metadata and the checksum tree scales with the size of the
# data, which would dwarf everything else.
WANTED_TREES = frozenset(
    {
        c_btrfs.BTRFS_ROOT_TREE_OBJECTID,
        c_btrfs.BTRFS_CHUNK_TREE_OBJECTID,
        c_btrfs.BTRFS_DEV_TREE_OBJECTID,
        c_btrfs.BTRFS_FS_TREE_OBJECTID,
        c_btrfs.BTRFS_UUID_TREE_OBJECTID,
    }
)


def enumerate_runs(fs: BtrfsFilesystem, *, thin: bool = False) -> MetadataRuns:
    """Enumerate all metadata byte ranges of a Btrfs filesystem.

    Btrfs keeps everything in B-trees, and directory entries, inodes and inline file data are all items
    inside those trees. Collecting every node of the relevant trees is therefore enough to make the whole
    filesystem traversable - unlike ext or XFS there are no separate directory data blocks to chase.

    Args:
        fs: The Btrfs filesystem to enumerate.
        thin: Unused for Btrfs.
    """
    btrfs = fs.btrfs
    runs = MetadataRuns(btrfs.sector_size)

    _collect_superblocks(btrfs, runs)

    visited: set[int] = set()

    _collect_tree(btrfs, btrfs.sb.chunk_root, runs, visited, "chunk tree")
    _collect_tree(btrfs, btrfs.sb.root, runs, visited, "root tree")

    if btrfs.sb.log_root:
        _collect_tree(btrfs, btrfs.sb.log_root, runs, visited, "log tree")

    _collect_roots(btrfs, runs, visited)
    _collect_subvolumes(btrfs, runs, visited)

    return runs


def file_runs(fh: ExtentStream) -> MetadataRuns | None:
    """Locate the file extents of an opened Btrfs file.

    The extents are logical addresses, so they land wherever the chunk tree maps them, which can be on
    a different device than the one the file was opened through. Compressed extents are stored as they
    are on disk, and decompress again when they are read back out of the snapshot.
    """
    btrfs = fh._fh.btrfs
    runs = MetadataRuns(btrfs.sector_size)

    for extent in fh.extents:
        # An inline extent lives in the tree, and a hole has no extent on disk at all
        if extent.disk_offset:
            _add_logical(btrfs, extent.disk_offset, extent.disk_length, runs)

    return runs


def _collect_superblocks(btrfs: Btrfs, runs: MetadataRuns) -> None:
    """Collect the superblocks, pinning the primary one to the state the trees were walked from.

    Btrfs writes a new tree root on every transaction commit, so re-reading the superblock after the
    walk can yield one that points at a root we never collected - which leaves the snapshot unreadable.
    Writing back the superblock exactly as it was parsed keeps it pointing at the trees we did collect.

    The mirrors are copied straight off the device. dissect only ever reads the primary, so they are
    kept purely as evidence of what was on disk.
    """
    primary = bytes(btrfs.sb.dumps()).ljust(SUPERBLOCK_SIZE, b"\x00")[:SUPERBLOCK_SIZE]

    for fh in btrfs.devices.values():
        size = size_of(fh)

        runs.add_literal(fh, c_btrfs.BTRFS_SUPER_INFO_OFFSET, primary)

        for offset in SUPERBLOCK_OFFSETS:
            if offset == c_btrfs.BTRFS_SUPER_INFO_OFFSET:
                continue

            # The mirror at 256GiB only exists on devices that are actually that large
            if size is None or offset + SUPERBLOCK_SIZE <= size:
                runs.collector(fh).add(offset, SUPERBLOCK_SIZE)


def _collect_roots(btrfs: Btrfs, runs: MetadataRuns, visited: set[int]) -> None:
    """Collect the trees referenced by the root tree that are needed for traversal."""
    try:
        cursor = btrfs._root_tree.cursor()
        items = list(cursor.walk(type=c_btrfs.BTRFS_ROOT_ITEM_KEY))
    except Exception as e:
        log.warning("Failed to walk the root tree: %s", e)
        log.debug("", exc_info=e)
        return

    for item, data in items:
        objectid = item.key.objectid

        # Subvolume and snapshot roots get their own object IDs from 256 onwards
        if objectid not in WANTED_TREES and objectid < c_btrfs.BTRFS_FIRST_FREE_OBJECTID:
            continue

        try:
            root_item = c_btrfs.btrfs_root_item(data)
        except Exception as e:
            log.debug("Failed to parse root item %d: %s", objectid, e)
            continue

        if root_item.bytenr:
            _collect_tree(btrfs, root_item.bytenr, runs, visited, f"tree {objectid}")


def _collect_subvolumes(btrfs: Btrfs, runs: MetadataRuns, visited: set[int]) -> None:
    """Collect the filesystem tree of every subvolume."""
    try:
        subvolumes = list(btrfs.subvolumes())
    except Exception as e:
        log.warning("Failed to enumerate subvolumes: %s", e)
        log.debug("", exc_info=e)
        return

    for subvolume in subvolumes:
        try:
            _collect_tree(btrfs, subvolume.tree.root_offset, runs, visited, f"subvolume {subvolume.objectid}")
        except Exception as e:  # noqa: PERF203
            log.warning("Failed to collect subvolume %s: %s", subvolume, e)
            log.debug("", exc_info=e)


def _collect_tree(btrfs: Btrfs, address: int, runs: MetadataRuns, visited: set[int], name: str) -> None:
    """Collect every node of the B-tree rooted at the given logical address.

    Walked node by node rather than with a :class:`dissect.btrfs.tree.Cursor`, which only exposes the
    items of the leaves it lands on, never the nodes those leaves live in.
    """

    def children(address: int) -> Iterator[int]:
        # Trees are shared between snapshots, so the same node can be reached many times over
        _add_logical(btrfs, address, btrfs.node_size, runs)

        node = btrfs._read_node(address)
        header = c_btrfs.btrfs_header(node)

        if header.level == 0:
            return

        records = iter_records(node[len(c_btrfs.btrfs_header) :], len(c_btrfs.btrfs_key_ptr), header.nritems)
        for record in records:
            if blockptr := c_btrfs.btrfs_key_ptr(record).blockptr:
                yield blockptr

    try:
        count = walk_tree(address, children, visited=visited, name=name)
    except Exception as e:
        log.warning("Failed to walk %s at %#x: %s", name, address, e)
        log.debug("", exc_info=e)
        return

    log.debug("Collected %d nodes for %s", count, name)


def _add_logical(btrfs: Btrfs, offset: int, length: int, runs: MetadataRuns) -> None:
    """Map a logical byte range onto the devices it lives on and record it."""
    for fh, physical, size in _map_logical(btrfs, offset, length):
        runs.collector(fh).add(physical, size)


def _map_logical(btrfs: Btrfs, offset: int, length: int) -> Iterator[tuple[object, int, int]]:
    """Translate a logical byte range into physical ranges on the devices that hold it.

    Mirrors the mapping that :meth:`dissect.btrfs.stream.ChunkStream._read` does, but yields the
    locations instead of the data. Mirrored profiles (DUP, RAID1) resolve to the same stripe that a
    read would use, so exactly one copy of each range is stored.
    """
    stream = btrfs._logical_fh
    chunk_offsets = stream._chunk_offsets
    chunks = stream.chunks

    while length > 0:
        chunk_idx = bisect_right(chunk_offsets, offset)
        if chunk_idx == 0 or chunk_idx > len(chunks):
            # Nothing is mapped here
            return

        chunk: Chunk = chunks[chunk_idx - 1]
        if offset >= chunk.offset + chunk.length:
            return

        chunk_offset = offset - chunk.offset
        chunk_remaining = chunk.length - chunk_offset

        while length > 0 and chunk_remaining > 0:
            stripe_num, stripe_idx, stripe_offset, stripe_remaining = _get_stripe_read_info(chunk, chunk_offset)
            stripe_read = min(stripe_remaining, length)

            stripe = chunk.stripes[stripe_idx % chunk.num_stripes]
            while stripe.fh is None:
                if chunk.type & c_btrfs.BTRFS_BLOCK_GROUP.DUP:
                    stripe_idx = 1
                else:
                    stripe_idx += 1
                stripe = chunk.stripes[stripe_idx % chunk.num_stripes]

            yield stripe.fh, stripe.offset + stripe_offset + stripe_num * chunk.stripe_length, stripe_read

            offset += stripe_read
            length -= stripe_read
            chunk_offset += stripe_read
            chunk_remaining -= stripe_read
