"""Filesystem metadata enumeration.

Enumerates the on-disk structures that a filesystem needs to be traversable - superblocks, inode tables,
B-tree nodes, directory blocks - so they can be written into a sparse ASDF snapshot. The result mimics a
full disk image: the volume layout, filesystems, directory trees and inode metadata are all really there,
only the contents of files that were not collected are missing.

:func:`file_runs` answers the same question for the data of a single file, so a file that *was* collected
can be written where the filesystem expects it and stays readable through the snapshot.

Enumerators deliberately walk the on-disk structures themselves instead of relying on the tree walking
and ``dataruns()`` APIs the dissect filesystem implementations offer. Those hand out the leaves of a
tree and drop the interior nodes on the way, while the interior nodes have to be present as well or the
tree cannot be walked again from the snapshot. Everything else is left to the implementations, which is
why the enumerators here only ever read what they also collect.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dissect.target.helpers.lazy import import_lazy
from dissect.util.stream import RunlistStream

from acquire.fsmeta.collector import MetadataRuns, RunCollector

if TYPE_CHECKING:
    from typing import BinaryIO

    from dissect.target.filesystem import Filesystem

log = logging.getLogger(__name__)

__all__ = ["MetadataRuns", "RunCollector", "enumerate_runs", "file_runs", "supported"]


# Imported lazily because the base dissect.target install does not pull in every filesystem
# implementation, so importing them all up front would break acquire for anyone without the [full]
# extra. A missing one surfaces as an ImportError when it is called, which is where it belongs: only
# the filesystems that are actually on the target have to be installed.
#
# Kept under a private name, because importing acquire.fsmeta.ext binds the real module onto this
# package as "ext", which would quietly replace the handle it was reached through.
_apfs = import_lazy("acquire.fsmeta.apfs")
_btrfs = import_lazy("acquire.fsmeta.btrfs")
_ext = import_lazy("acquire.fsmeta.ext")
_ntfs = import_lazy("acquire.fsmeta.ntfs")
_xfs = import_lazy("acquire.fsmeta.xfs")

# Maps Filesystem.__type__ to the enumerator that implements it. The APFS container holds all of its
# volumes, so it is enumerated as a whole and the individual apfs volume filesystems are skipped.
ENUMERATORS = {
    "ext": _ext.enumerate_runs,
    "xfs": _xfs.enumerate_runs,
    "btrfs": _btrfs.enumerate_runs,
    "ntfs": _ntfs.enumerate_runs,
    "apfs-container": _apfs.enumerate_runs,
}

# Filesystems that are covered by another filesystem in the same target
COVERED_BY_PARENT = {"apfs"}

# Attributes that hold the actual parser, in the order they should be checked. A Btrfs subvolume
# filesystem carries both, and the subvolume is the more specific of the two.
UNDERLYING_ATTRIBUTES = ("btrfs", "extfs", "xfs", "ntfs", "container")

# Maps the module of a file data stream to the locator that can find its data on disk. Filesystems
# that hand out a plain RunlistStream carry their layout with them and need no help.
LOCATORS = {
    "dissect.apfs.stream": _apfs.file_runs,
    "dissect.btrfs.stream": _btrfs.file_runs,
}


def supported(fs: Filesystem) -> bool:
    """Return whether metadata enumeration is implemented for the given filesystem."""
    return getattr(fs, "__type__", None) in ENUMERATORS


def identity(fs: Filesystem) -> int:
    """Return a key that is shared by every filesystem backed by the same on-disk structures.

    Several :class:`dissect.target.filesystem.Filesystem` instances can wrap one on-disk filesystem:
    Btrfs exposes one per subvolume, and they all resolve to the same trees. Enumerating each of them
    would repeat the same work - on a normal Fedora install that is four passes over the same
    850 MiB of metadata.
    """
    for attr in UNDERLYING_ATTRIBUTES:
        underlying = getattr(fs, attr, None)
        if underlying is not None:
            return id(underlying)

    return id(fs)


def file_runs(fh: BinaryIO) -> MetadataRuns | None:
    """Locate the bytes of an opened file on the devices they live on.

    Data written at its real offset stays readable through the filesystem in the snapshot, which is what
    makes a collected file browsable with ``target-shell`` rather than only present in the metadata tar.

    Most filesystems hand out a :class:`dissect.util.stream.RunlistStream`, which says where its data
    lives. Btrfs and APFS resolve their extents per read instead, so those are looked up per filesystem.

    Args:
        fh: The file-like object of an opened file.

    Returns:
        The runs holding the file data, or ``None`` if it has no place on a disk.
    """
    if isinstance(fh, RunlistStream):
        if not fh.runlist:
            return None

        # RunlistStream keeps the volume it reads from in _fh, unwrapping any nested runlist stream
        runs = MetadataRuns(fh.block_size)
        runs.collector(fh._fh).add_runlist(fh.runlist, fh.block_size)
        return runs

    locator = LOCATORS.get(type(fh).__module__)
    if locator is None:
        return None

    try:
        return locator(fh)
    except Exception as e:
        log.warning("Failed to locate the data of %r: %s", fh, e)
        log.debug("", exc_info=e)
        return None


def enumerate_runs(fs: Filesystem, **kwargs) -> MetadataRuns | None:
    """Enumerate the metadata byte ranges of the given filesystem.

    Args:
        fs: The filesystem to enumerate.
        **kwargs: Passed through to the filesystem specific enumerator.

    Returns:
        The collected runs, or ``None`` if the filesystem is not supported or could not be enumerated.
    """
    fs_type = getattr(fs, "__type__", None)
    enumerator = ENUMERATORS.get(fs_type)

    if enumerator is None:
        if fs_type not in COVERED_BY_PARENT:
            log.warning("Filesystem metadata collection is not supported for %s (%s)", fs, fs_type)
        return None

    try:
        return enumerator(fs, **kwargs)
    except Exception as e:
        log.error("Failed to enumerate filesystem metadata of %s: %s", fs, e)  # noqa: TRY400
        log.debug("", exc_info=e)
        return None
