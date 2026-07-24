from __future__ import annotations

import io
import logging
import stat
from typing import TYPE_CHECKING

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.xfs import fsb_to_bb, parse_fsblock

from acquire.fsmeta.collector import MetadataRuns
from acquire.fsmeta.utils import MAX_TREE_DEPTH, iter_records

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dissect.target.filesystems.xfs import XfsFilesystem
    from dissect.xfs.xfs import XFS, INode

    from acquire.fsmeta.collector import RunCollector

log = logging.getLogger(__name__)

# An inode chunk tracked by a single inode B+tree record always holds 64 inodes
INODES_PER_CHUNK = 64

# Superblock, AGF, AGI and AGFL, each taking up one sector at the start of an allocation group
AG_HEADER_SECTORS = 4

# Both the inode B+tree records and the block map records of an extent list are 16 bytes
RECORD_SIZE = 16


def enumerate_runs(fs: XfsFilesystem, *, thin: bool = False) -> MetadataRuns:
    """Enumerate all metadata byte ranges of an XFS filesystem.

    Collects the headers of every allocation group, the inode B+trees and the inode chunks they point at,
    and the directory blocks, remote symlink targets and attribute forks reachable from the inodes.

    Args:
        fs: The XFS filesystem to enumerate.
        thin: Unused for XFS. Inode chunks are only allocated as they are needed, so there is no
            equivalent of a sparsely used inode table to skip.
    """
    xfs = fs.xfs
    runs = MetadataRuns(xfs.block_size)
    collector = runs.collector(xfs.fh)

    for agnum in range(xfs.sb.sb_agcount):
        try:
            _collect_allocation_group(xfs, agnum, collector)
        except Exception as e:  # noqa: PERF203
            log.warning("Failed to collect allocation group %d: %s", agnum, e)
            log.debug("", exc_info=e)

    return runs


def _collect_allocation_group(xfs: XFS, agnum: int, collector: RunCollector) -> None:
    """Collect the headers, inode B+tree and inode chunks of a single allocation group."""
    ag = xfs.get_allocation_group(agnum)
    ag_offset = agnum * xfs._ag_size

    # The superblock, AGF, AGI and AGFL sit in the first sectors of the group
    collector.add(ag_offset, AG_HEADER_SECTORS * xfs.sb.sb_sectsize)

    chunk_size = INODES_PER_CHUNK * xfs.sb.sb_inodesize
    inodes = 0

    for record in _walk_inobt(xfs, agnum, ag.agi.agi_root, collector):
        chunk_offset = ag_offset + record.ir_startino * xfs.sb.sb_inodesize
        collector.add(chunk_offset, chunk_size)

        for index in range(INODES_PER_CHUNK):
            # Bits set in ir_free mark free inodes. Walk them anyway - a free inode can still hold
            # recoverable metadata, the same reason the full inode table is kept on ext.
            inum = record.ir_startino + index

            try:
                inode = xfs.get_relative_inode(agnum, inum)
                _collect_inode(xfs, inode, collector)
                inodes += 1
            except Exception as e:
                log.debug("Skipping inode %d:%d: %s", agnum, inum, e)

    log.debug("Allocation group %d: walked %d inodes", agnum, inodes)


def _collect_inode(xfs: XFS, inode: INode, collector: RunCollector) -> None:
    """Collect the blocks of an inode that hold metadata rather than file content."""
    dinode = inode.inode

    # The attribute fork can hold remote extended attribute values for any inode type
    if dinode.di_forkoff:
        try:
            _collect_fork(xfs, inode, collector, attr=True)
        except Exception as e:
            log.debug("Failed to collect attribute fork of %r: %s", inode, e)

    filetype = stat.S_IFMT(dinode.di_mode)
    if filetype not in (stat.S_IFDIR, stat.S_IFLNK):
        return

    if dinode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
        # Short form directories and symlinks live inside the inode itself
        return

    _collect_fork(xfs, inode, collector, attr=False)


def _collect_fork(xfs: XFS, inode: INode, collector: RunCollector, *, attr: bool) -> None:
    """Collect the blocks of a data or attribute fork, including the interior B+tree nodes."""
    dinode = inode.inode

    if attr:
        fork_format = dinode.di_aformat
        # The extent counts live in a union that differs per inode version, which attr_extents resolves
        extent_count = inode.attr_extents
        fork = inode.attrfork()
    else:
        fork_format = dinode.di_format
        extent_count = inode.data_extents
        fork = inode.datafork()

    if fork_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS:
        for record in iter_records(fork.read(extent_count * RECORD_SIZE), RECORD_SIZE, extent_count):
            _, block, count, _ = parse_fsblock(record)
            collector.add_block(_fsb_to_block(xfs, block), count)

    elif fork_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE:
        root = c_xfs.xfs_bmdr_block(fork)

        # Pointers start around halfway through the fork
        maxrecs = (fork.size - 4) // RECORD_SIZE
        fork.seek(4 + maxrecs * 8)

        for ptr in c_xfs.uint64[root.bb_numrecs](fork):
            for record in _walk_btree(xfs, _fsb_to_block(xfs, ptr), collector, long=True):
                _, block, count, _ = parse_fsblock(record)
                collector.add_block(_fsb_to_block(xfs, block), count)


def _fsb_to_block(xfs: XFS, fsb: int) -> int:
    """Convert a filesystem block number to a block number relative to the start of the volume."""
    agnum, blknum = fsb_to_bb(fsb, xfs.sb.sb_agblklog)
    return agnum * xfs.sb.sb_agblocks + blknum


def _walk_inobt(xfs: XFS, agnum: int, block: int, collector: RunCollector) -> Iterator[c_xfs.xfs_inobt_rec]:
    """Walk the inode B+tree of an allocation group, collecting every node it visits.

    The interior nodes are collected as well as the leaves. ``walk_agi`` reads them straight off the
    volume, so without them the tree cannot be walked again from the snapshot.
    """
    for record in _walk_btree(xfs, block, collector, base=agnum * xfs.sb.sb_agblocks):
        yield c_xfs.xfs_inobt_rec(record)


def _walk_btree(
    xfs: XFS, block: int, collector: RunCollector, *, long: bool = False, base: int = 0, depth: int = 0
) -> Iterator[bytes]:
    """Walk an XFS B+tree, collecting every node block and yielding the leaf records.

    XFS has two flavours of these. Short form trees live inside a single allocation group and address
    their nodes with 32 bit AG relative block numbers, long form trees address theirs with 64 bit
    filesystem block numbers.

    This mirrors :meth:`dissect.xfs.xfs.XFS.walk_small_tree` and :meth:`~dissect.xfs.xfs.XFS.walk_large_tree`,
    which cannot be used as they are: they only yield the leaf records and keep the blocks of the nodes
    they pass through to themselves, and those are exactly what has to end up in the snapshot.

    Args:
        xfs: The XFS filesystem the tree lives on.
        block: The block the tree is rooted at, relative to ``base``.
        collector: The collector to add the node blocks to.
        long: Whether this is a long form tree.
        base: The first block of the allocation group a short form tree is relative to.
        depth: The current depth in the tree.
    """
    if depth > MAX_TREE_DEPTH:
        log.warning("B+tree deeper than expected, stopping")
        return

    header = xfs._lblock_s if long else xfs._sblock_s
    ptr_size = 8 if long else 4

    collector.add_block(base + block)

    xfs.fh.seek((base + block) * xfs.block_size)
    node = header(xfs.fh)

    if node.bb_level == 0:
        yield from iter_records(xfs.fh.read(node.bb_numrecs * RECORD_SIZE), RECORD_SIZE, node.bb_numrecs)
        return

    # Pointers start around halfway through the node, and we are already at the end of its header
    maxrecs = (xfs.block_size - len(header)) // (2 * ptr_size)
    xfs.fh.seek(maxrecs * ptr_size, io.SEEK_CUR)

    for ptr in (c_xfs.uint64 if long else c_xfs.uint32)[node.bb_numrecs](xfs.fh):
        child = _fsb_to_block(xfs, ptr) if long else ptr
        yield from _walk_btree(xfs, child, collector, long=long, base=base, depth=depth + 1)
