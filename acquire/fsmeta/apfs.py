from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dissect.apfs.c_apfs import c_apfs
from dissect.apfs.cursor import Cursor
from dissect.apfs.objects import BTree

from acquire.fsmeta.collector import MetadataRuns
from acquire.fsmeta.utils import walk_tree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dissect.apfs.apfs import APFS
    from dissect.apfs.objects import BTreeNode, ObjectMap
    from dissect.apfs.objects.fs import FS
    from dissect.apfs.stream import FileStream
    from dissect.target.filesystems.apfs import ApfsFilesystem

    from acquire.fsmeta.collector import RunCollector

log = logging.getLogger(__name__)

# The high bit of the checkpoint area length marks it as a B-tree rather than a plain block range
CHECKPOINT_BTREE_FLAG = 0x80000000
CHECKPOINT_LENGTH_MASK = 0x7FFFFFFF


def enumerate_runs(fs: ApfsFilesystem, *, thin: bool = False) -> MetadataRuns:
    """Enumerate all metadata byte ranges of an APFS container.

    APFS keeps everything in B-trees of physically addressed objects. Collecting the superblock, the
    checkpoint areas and every node of the container and volume trees is therefore enough to make the
    whole container traversable - directory entries, inodes and extended attributes are all records in
    the filesystem tree, so there are no separate directory blocks to chase.

    The whole container is enumerated at once, covering every volume in it.

    Args:
        fs: The APFS container filesystem to enumerate.
        thin: Unused for APFS.
    """
    container = fs.container
    runs = MetadataRuns(container.block_size)
    collector = runs.collector(container.fh)

    _collect_container(container, collector)

    for volume in container.volumes:
        try:
            _collect_volume(container, volume, collector)
        except Exception as e:  # noqa: PERF203
            log.warning("Failed to collect APFS volume %r: %s", getattr(volume, "name", volume), e)
            log.debug("", exc_info=e)

    return runs


def file_runs(fh: FileStream) -> MetadataRuns | None:
    """Locate the file extents of an opened APFS file.

    APFS resolves an extent per read rather than handing out a runlist, but the records that describe
    them are the ones :meth:`dissect.apfs.objects.fs.FS.records` already returns.
    """
    volume = fh.volume
    container = volume.container

    records = volume.records(fh.oid).get(c_apfs.APFS_TYPE.FILE_EXTENT)
    if not records:
        # A sealed volume keeps its extents in the fext tree instead, which records() does not search
        return None

    runs = MetadataRuns(container.block_size)
    collector = runs.collector(container.fh)

    for _, extent in records:
        # A hole has no block behind it, and the length carries flags in its low bits
        if extent.phys_block_num:
            collector.add_block(
                extent.phys_block_num,
                (extent.len_and_flags & c_apfs.J_FILE_EXTENT_LEN_MASK) // container.block_size,
            )

    return runs


def _collect_container(container: APFS, collector: RunCollector) -> None:
    """Collect the container superblock, checkpoint areas and container object map."""
    sb = container.sb

    # Block zero holds the superblock that everything else is found through
    collector.add_block(0)
    collector.add_block(sb.address)

    # The checkpoint descriptor and data areas hold the superblock copies, checkpoint maps and the
    # ephemeral objects such as the space manager and the reaper
    for base, length in (
        (sb.object.nx_xp_desc_base, sb.object.nx_xp_desc_blocks),
        (sb.object.nx_xp_data_base, sb.object.nx_xp_data_blocks),
    ):
        if length & CHECKPOINT_BTREE_FLAG:
            # The area is described by a B-tree instead of being a plain range
            _collect_checkpoint_btree(container, base, collector)
        elif base and length:
            collector.add_block(base, length & CHECKPOINT_LENGTH_MASK)

    try:
        _collect_omap(container, sb.omap, collector, "container")
    except Exception as e:
        log.warning("Failed to collect the container object map: %s", e)
        log.debug("", exc_info=e)


def _collect_checkpoint_btree(container: APFS, base: int, collector: RunCollector) -> None:
    """Collect a checkpoint area that is described by a B-tree of block ranges."""
    try:
        btree = BTree(container, base)
        _walk_btree(container, btree, collector)

        for _, value in Cursor(btree).walk():
            prange = c_apfs.prange(value)
            collector.add_block(prange.pr_start_paddr, prange.pr_block_count)
    except Exception as e:
        log.warning("Failed to collect checkpoint B-tree at %#x: %s", base, e)
        log.debug("", exc_info=e)


def _collect_omap(container: APFS, omap: ObjectMap, collector: RunCollector, name: str) -> None:
    """Collect an object map and every node of its B-tree."""
    collector.add_block(omap.address)
    _walk_btree(container, omap.btree, collector)
    log.debug("Collected the %s object map", name)


def _collect_volume(container: APFS, volume: FS, collector: RunCollector) -> None:
    """Collect the superblock and trees of a single volume."""
    # The volume superblock itself
    collector.add_block(volume.address)

    _collect_omap(container, volume.omap, collector, f"volume {volume.name!r}")

    # The filesystem tree holds the inodes, directory entries and extended attributes, and is
    # addressed with virtual OIDs that resolve through the volume object map
    oid = volume.object.apfs_root_tree_oid if volume.is_sealed else 0
    _collect_tree(container, volume, "root_tree", collector, omap=volume.omap, oid=oid, xid=volume.xid)

    # Snapshot metadata and the extent reference tree are physically addressed
    for attr in ("snap_meta_tree", "extentref_tree", "fext_tree"):
        _collect_tree(container, volume, attr, collector)


def _collect_tree(
    container: APFS,
    volume: FS,
    attr: str,
    collector: RunCollector,
    omap: ObjectMap | None = None,
    oid: int = 0,
    xid: int | None = None,
) -> None:
    """Collect every node of one of a volume's B-trees, if it has one."""
    try:
        btree = getattr(volume, attr, None)
    except Exception as e:
        log.debug("Volume %r has no %s: %s", volume.name, attr, e)
        return

    if btree is None:
        return

    try:
        count = _walk_btree(container, btree, collector, omap=omap, oid=oid, xid=xid)
        log.debug("Collected %d nodes of %s for volume %r", count, attr, volume.name)
    except Exception as e:
        log.warning("Failed to walk %s of volume %r: %s", attr, volume.name, e)
        log.debug("", exc_info=e)


def _walk_btree(
    container: APFS,
    btree: BTree,
    collector: RunCollector,
    omap: ObjectMap | None = None,
    oid: int = 0,
    xid: int | None = None,
) -> int:
    """Walk every node of a B-tree, collecting the blocks each node occupies.

    :meth:`dissect.apfs.cursor.Cursor.walk` is no substitute: it yields the records of the leaves and
    never the nodes they were read from, and stepping over every record of a filesystem tree to find the
    handful of nodes holding them is far more work than walking the nodes themselves.
    """

    def children(node: BTreeNode) -> Iterator[BTreeNode]:
        # Nodes are not necessarily a single block
        collector.add(node.address * container.block_size, btree._node_size or container.block_size)

        if node.is_leaf:
            return

        for idx in range(node.nkeys):
            try:
                yield btree._node_child(node, idx, omap, oid, xid)
            except Exception as e:  # noqa: PERF203
                log.debug("Failed to resolve child %d of node %#x: %s", idx, node.address, e)

    return walk_tree(btree.root, children, key=lambda node: node.address, name="B-tree")
