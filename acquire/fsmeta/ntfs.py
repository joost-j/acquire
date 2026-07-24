from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dissect.ntfs.c_ntfs import ATTRIBUTE_TYPE_CODE

from acquire.fsmeta.collector import MetadataRuns

if TYPE_CHECKING:
    from dissect.ntfs.mft import MftRecord
    from dissect.ntfs.ntfs import NTFS
    from dissect.target.filesystems.ntfs import NtfsFilesystem

    from acquire.fsmeta.collector import RunCollector

log = logging.getLogger(__name__)

# The boot sector and the rest of the $Boot file at the start of the volume
BOOT_SIZE = 0x2000

# System files that describe the volume itself. $LogFile is deliberately left out - it is large, it is
# not needed to traverse the filesystem, and the --ntfs module already collects it as a file.
SYSTEM_FILES = ("$MFT", "$MFTMirr", "$Bitmap", "$Boot", "$Secure", "$Volume", "$AttrDef", "$UpCase")

# Attributes that describe structure rather than file content
METADATA_ATTRIBUTES = (
    ATTRIBUTE_TYPE_CODE.INDEX_ALLOCATION,
    ATTRIBUTE_TYPE_CODE.BITMAP,
    ATTRIBUTE_TYPE_CODE.ATTRIBUTE_LIST,
)


def enumerate_runs(fs: NtfsFilesystem, *, thin: bool = False) -> MetadataRuns:
    """Enumerate all metadata byte ranges of an NTFS filesystem.

    Collects ``$Boot``, the clusters of ``$MFT`` and the other volume system files, and the
    ``$INDEX_ALLOCATION`` of every directory. The MFT on its own describes every file, but directory
    listing needs the index buffers as well, which is what makes the tree browsable rather than only
    reconstructable with the mft plugin.

    Args:
        fs: The NTFS filesystem to enumerate.
        thin: Unused for NTFS.
    """
    ntfs = fs.ntfs
    runs = MetadataRuns(ntfs.cluster_size)
    # Every runlist in this filesystem, the one of the $MFT included, is relative to the volume
    collector = runs.collector(ntfs.fh)

    # The boot sector, plus its backup in the last sector of the volume
    collector.add(0, BOOT_SIZE)

    _collect_mft(ntfs, collector)
    _collect_system_files(ntfs, collector)
    _collect_indices(ntfs, collector)

    return runs


def _collect_mft(ntfs: NTFS, collector: RunCollector) -> None:
    """Collect all clusters of the ``$MFT`` itself."""
    fh = ntfs.mft.fh

    runlist = getattr(fh, "runlist", None)
    if not runlist:
        log.warning("$MFT has no runlist, cannot collect it")
        return

    collector.add_runlist(runlist, getattr(fh, "block_size", ntfs.cluster_size))
    log.debug("Collected $MFT (%d runs)", len(runlist))


def _collect_system_files(ntfs: NTFS, collector: RunCollector) -> None:
    """Collect the clusters of the volume system files."""
    for name in SYSTEM_FILES:
        try:
            record = ntfs.mft.get(name)
        except Exception as e:
            log.debug("Skipping system file %s: %s", name, e)
            continue

        _collect_record(record, collector, all_data=True)


def _collect_indices(ntfs: NTFS, collector: RunCollector) -> None:
    """Collect the index buffers of every directory in the MFT."""
    directories = 0

    for record in ntfs.mft.segments():
        try:
            if not record.is_dir():
                continue

            _collect_record(record, collector, all_data=False)
            directories += 1
        except Exception as e:
            log.debug("Skipping MFT segment: %s", e)

    log.debug("Collected index buffers of %d directories", directories)


def _collect_record(record: MftRecord, collector: RunCollector, *, all_data: bool) -> None:
    """Collect the non-resident attributes of an MFT record that hold metadata.

    Args:
        record: The MFT record to collect.
        collector: The collector to add the runs to.
        all_data: Also collect the ``$DATA`` attributes. Used for the volume system files, whose data
            is the metadata we are after.
    """
    for attributes in record.attributes.values():
        for attribute in attributes:
            type_code = attribute.header.type
            if type_code not in METADATA_ATTRIBUTES and not (all_data and type_code == ATTRIBUTE_TYPE_CODE.DATA):
                continue

            if attribute.resident:
                # Resident attributes live inside the MFT record, which is already collected
                continue

            try:
                collector.add_runlist(attribute.dataruns(), collector.block_size)
            except Exception as e:
                log.debug("Failed to collect %s of %r: %s", type_code, record, e)
