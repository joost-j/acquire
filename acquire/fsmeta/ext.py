from __future__ import annotations

import io
import logging
import stat
from typing import TYPE_CHECKING

from dissect.extfs.c_ext import c_ext

from acquire.fsmeta.collector import MetadataRuns

if TYPE_CHECKING:
    from dissect.extfs.extfs import ExtFS
    from dissect.target.filesystems.extfs import ExtFilesystem

    from acquire.fsmeta.collector import RunCollector

log = logging.getLogger(__name__)

EXT4_EXTENT_MAGIC = 0xF30A


def enumerate_runs(fs: ExtFilesystem, *, thin: bool = False) -> MetadataRuns:
    """Enumerate all metadata byte ranges of an ext2/ext3/ext4 filesystem.

    Collects the superblock, group descriptor table, per group bitmaps and inode tables, and the directory
    data blocks, external symlink targets and extended attribute blocks reachable from the inodes.

    Directories are found with a linear scan over the inode tables rather than a recursive walk of the tree.
    That keeps acquisition cheap and, more importantly, also picks up directories that are no longer linked
    into the tree.

    Args:
        fs: The ext filesystem to enumerate.
        thin: Skip inode table regions that the group descriptors mark as never used. Saves a lot of space,
            but loses deleted inodes.
    """
    extfs = fs.extfs
    runs = MetadataRuns(extfs.block_size)
    collector = runs.collector(extfs.fh)

    _collect_superblocks(extfs, collector)
    _collect_groups(extfs, collector, thin=thin)
    _collect_inodes(extfs, collector, thin=thin)

    return runs


def _collect_superblocks(extfs: ExtFS, collector: RunCollector) -> None:
    """Collect the primary superblock and the group descriptor table."""
    sb = extfs.sb

    # The superblock lives at byte 1024, which is block 0 or 1 depending on the block size
    collector.add(0, c_ext.EXT2_SBOFF + 1024)

    gdt_blocks = _gdt_blocks(extfs)
    collector.add(extfs.groups_offset, gdt_blocks * extfs.block_size)

    if sb.s_feature_incompat & c_ext.EXT2_FEATURE_INCOMPAT_META_BG:
        log.warning("META_BG is enabled, group descriptors may be incomplete")

    # Backup superblocks and group descriptor tables, so the filesystem stays recoverable
    for group in _backup_groups(extfs):
        block = sb.s_first_data_block + group * sb.s_blocks_per_group
        collector.add_block(block, 1 + gdt_blocks)


def _gdt_blocks(extfs: ExtFS) -> int:
    """The number of blocks taken up by the group descriptor table, including reserved growth blocks."""
    gdt_size = extfs.groups_count * extfs._group_desc_size
    blocks = -(-gdt_size // extfs.block_size)
    return blocks + extfs.sb.s_reserved_gdt_blocks


def _backup_groups(extfs: ExtFS) -> list[int]:
    """Return the groups that hold a backup superblock."""
    if not extfs.sb.s_feature_ro_compat & c_ext.EXT2_FEATURE_RO_COMPAT_SPARSE_SUPER:
        return list(range(1, extfs.groups_count))

    # With sparse_super, backups live in group 1 and in the powers of 3, 5 and 7
    groups = {1}
    for base in (3, 5, 7):
        group = base
        while group < extfs.groups_count:
            groups.add(group)
            group *= base

    return sorted(group for group in groups if group < extfs.groups_count)


def _group_locations(extfs: ExtFS, group: int) -> tuple[int, int, int, int]:
    """Return the block bitmap, inode bitmap, inode table and unused inode count of a group."""
    desc = extfs._read_group_desc(group)

    if extfs._group_desc_struct == c_ext.ext4_group_desc:
        block_bitmap = (desc.bg_block_bitmap_hi << 32) | desc.bg_block_bitmap_lo
        inode_bitmap = (desc.bg_inode_bitmap_hi << 32) | desc.bg_inode_bitmap_lo
        inode_table = (desc.bg_inode_table_hi << 32) | desc.bg_inode_table_lo
        unused = (desc.bg_itable_unused_hi << 32) | desc.bg_itable_unused_lo
    else:
        block_bitmap = desc.bg_block_bitmap_lo
        inode_bitmap = desc.bg_inode_bitmap_lo
        inode_table = desc.bg_inode_table_lo
        # ext2 group descriptors do not track unused inodes
        unused = 0

    return block_bitmap, inode_bitmap, inode_table, unused


def _collect_groups(extfs: ExtFS, collector: RunCollector, *, thin: bool) -> None:
    """Collect the bitmaps and inode tables of every block group."""
    sb = extfs.sb
    table_size = sb.s_inodes_per_group * sb.s_inode_size

    for group in range(extfs.groups_count):
        try:
            block_bitmap, inode_bitmap, inode_table, unused = _group_locations(extfs, group)
        except Exception as e:
            log.warning("Failed to read group descriptor %d: %s", group, e)
            log.debug("", exc_info=e)
            continue

        collector.add_block(block_bitmap)
        collector.add_block(inode_bitmap)

        size = table_size
        if thin and 0 < unused <= sb.s_inodes_per_group:
            # The tail of the table was never used, so there is nothing to recover from it
            size = (sb.s_inodes_per_group - unused) * sb.s_inode_size

        collector.add(inode_table * extfs.block_size, size)


def _iter_inodes(extfs: ExtFS, *, thin: bool) -> tuple[int, c_ext.ext4_inode]:
    """Iterate over every inode in the filesystem, reading the inode tables block-wise.

    Reading the tables in bulk rather than seeking per inode keeps this usable on filesystems with
    millions of inodes.
    """
    sb = extfs.sb
    inode_size = sb.s_inode_size

    for group in range(extfs.groups_count):
        try:
            _, _, inode_table, unused = _group_locations(extfs, group)
        except Exception:
            continue

        count = sb.s_inodes_per_group
        if thin and 0 < unused <= count:
            count -= unused

        try:
            extfs.fh.seek(inode_table * extfs.block_size)
            # Read one inode extra so the trailing dynamic i_extra field of the last inode still has data
            buf = io.BytesIO(extfs.fh.read((count + 1) * inode_size))
        except Exception as e:
            log.warning("Failed to read inode table of group %d: %s", group, e)
            log.debug("", exc_info=e)
            continue

        for index in range(count):
            inum = group * sb.s_inodes_per_group + index + 1

            try:
                buf.seek(index * inode_size)
                yield inum, c_ext.ext4_inode(buf)
            except Exception:
                continue


def _collect_inodes(extfs: ExtFS, collector: RunCollector, *, thin: bool) -> None:
    """Collect the data blocks of every directory, external symlink and extended attribute block."""
    directories = 0
    symlinks = 0
    xattrs = 0

    for inum, inode in _iter_inodes(extfs, thin=thin):
        if not inode.i_mode and not inode.i_links_count:
            continue

        # Extended attribute blocks apply to every inode type
        if inode.i_file_acl_lo:
            collector.add_block((inode.i_file_acl_high << 32) | inode.i_file_acl_lo)
            xattrs += 1

        filetype = stat.S_IFMT(inode.i_mode)
        size = (inode.i_size_high << 32) + inode.i_size_lo

        if filetype == stat.S_IFDIR:
            directories += 1
        elif filetype == stat.S_IFLNK and size >= 60:
            # Short symlink targets are stored inside the inode itself, which the inode table already covers
            symlinks += 1
        else:
            continue

        if inode.i_flags & c_ext.EXT4_INLINE_DATA_FL:
            # The data lives in the inode, which is already covered by the inode table
            continue

        try:
            _collect_inode_blocks(extfs, inode, collector)
        except Exception as e:
            log.warning("Failed to collect data blocks of inode %d: %s", inum, e)
            log.debug("", exc_info=e)

    log.debug("Collected %d directories, %d symlinks and %d xattr blocks", directories, symlinks, xattrs)


def _collect_inode_blocks(extfs: ExtFS, inode: c_ext.ext4_inode, collector: RunCollector) -> None:
    """Collect all blocks of an inode, including the interior blocks of its block map."""
    if inode.i_flags & c_ext.EXT4_EXTENTS_FL:
        _walk_extents(extfs, bytes(inode.i_block), collector)
        return

    blocks = c_ext.uint32[15](inode.i_block)

    for block in blocks[: c_ext.EXT2_NDIR_BLOCKS]:
        if block:
            collector.add_block(block)

    for level in range(c_ext.EXT2_NIND_BLOCKS):
        _walk_indirect(extfs, blocks[c_ext.EXT2_NDIR_BLOCKS + level], level + 1, collector)


def _walk_extents(extfs: ExtFS, buf: bytes, collector: RunCollector, depth: int = 0) -> None:
    """Walk an ext4 extent tree, collecting both the interior index blocks and the leaf extents.

    The interior blocks matter as much as the leaves here: :meth:`dissect.extfs.extfs.INode.dataruns` only
    returns the leaves, but ``_parse_extents`` reads the index blocks straight off the volume, so without
    them the tree cannot be walked again from the snapshot.
    """
    if depth > 5:
        log.warning("Extent tree deeper than expected, stopping")
        return

    stream = io.BytesIO(buf)
    header = c_ext.ext4_extent_header(stream)

    if header.eh_magic != EXT4_EXTENT_MAGIC:
        return

    if header.eh_depth == 0:
        for _ in range(header.eh_entries):
            extent = c_ext.ext4_extent(stream)
            # Lengths over 0x8000 mark an uninitialized extent, which holds no directory data
            if extent.ee_len and extent.ee_len <= 0x8000:
                collector.add_block((extent.ee_start_hi << 32) | extent.ee_start_lo, extent.ee_len)
        return

    for _ in range(header.eh_entries):
        index = c_ext.ext4_extent_idx(stream)
        child = (index.ei_leaf_hi << 32) | index.ei_leaf_lo
        if not child:
            continue

        collector.add_block(child)

        extfs.fh.seek(child * extfs.block_size)
        _walk_extents(extfs, extfs.fh.read(extfs.block_size), collector, depth + 1)


def _walk_indirect(extfs: ExtFS, block: int, level: int, collector: RunCollector) -> None:
    """Walk an ext2/ext3 indirect block chain, collecting the indirect blocks and the blocks they point to."""
    if not block:
        return

    collector.add_block(block)

    extfs.fh.seek(block * extfs.block_size)
    entries = c_ext.uint32[extfs.block_size // 4](extfs.fh)

    if level == 1:
        for entry in entries:
            if entry:
                collector.add_block(entry)
        return

    for entry in entries:
        _walk_indirect(extfs, entry, level - 1, collector)
