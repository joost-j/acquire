from __future__ import annotations

import gzip
import io
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, BinaryIO

from dissect.evidence.asdf.asdf import MAX_IDX, AsdfWriter

from acquire import fsmeta
from acquire.crypt import EncryptedStream
from acquire.fsmeta.utils import size_of
from acquire.outputs.base import Output

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from dissect.target import Target
    from dissect.target.filesystem import Filesystem, FilesystemEntry

    from acquire.fsmeta import MetadataRuns

log = logging.getLogger(__name__)

# The tail of a disk, which holds the backup GPT and anchors the size of the stream, see _write_anchor
ANCHOR_SIZE = 34 * 512

# The head of a disk, which holds the MBR, the primary GPT and the alignment gap bootloaders live in
VOLUME_SYSTEM_SIZE = 1024 * 1024

# AsdfWriter gzips the whole file, so gzip is the only method it can offer
ASDF_COMPRESSION_METHODS = {"gzip": "gz"}

# Level 9 spends six times the CPU of level 6 for the same ratio on snapshot data: measured over a real
# 956 MB btrfs metadata snapshot, 8.9 MB/s at 2.61x against 54 MB/s at 2.62x
GZIP_COMPRESSION_LEVEL = 6


def volume_system_regions(volumes: list[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    """Yield the regions of a disk that describe its volume layout rather than hold volume data.

    Besides the head of the disk, that is the gap in front of every volume: an MBR extended partition
    only announces that logical partitions exist, and each one is described by an extended boot record
    sitting just in front of it, anywhere on the disk. Collecting only the head loses every filesystem
    inside the extended partition.

    Each gap is capped, and never reaches back into the volume before it, because unallocated space and
    the tail of the previous volume both hold data that was never asked for. A disk whose first volume
    starts at offset zero has no partition table at all, and gets nothing.

    Args:
        volumes: The ``(offset, size)`` of every volume on the disk.

    Yields:
        The ``(offset, length)`` of each region to collect, in order.
    """
    if not volumes:
        yield 0, VOLUME_SYSTEM_SIZE
        return

    end = 0
    for i, (offset, size) in enumerate(sorted(volumes)):
        if i:
            start = max(end, offset - VOLUME_SYSTEM_SIZE)
            length = offset - start
        else:
            start = 0
            length = min(offset, VOLUME_SYSTEM_SIZE)

        if length > 0:
            yield start, length

        end = max(end, offset + size)


class _ContinuousStream(io.RawIOBase):
    """Write into a stream continuously, and close what that stream was built on.

    :class:`dissect.evidence.asdf.stream.SubStreamBase` finalizes downwards: finalizing the CRC32 of one
    block calls ``finalize()`` and ``flush()`` on whatever sits below it. Once per block, that is fatal
    to both layers acquire puts underneath. :class:`acquire.crypt.EncryptedStream` finalizes by computing
    its GCM digest, after which it can never be written to again - writing straight through it fails on
    the second block. :class:`gzip.GzipFile` flushes with ``Z_SYNC_FLUSH``, which ends the deflate block
    and emits a fresh Huffman table, costing 1.7x the file size on 256 blocks and more as blocks get
    smaller. Stopping the cascade here leaves both to be finished exactly once, when the file is closed.
    """

    def __init__(self, fh: BinaryIO, below: BinaryIO):
        self.fh = fh
        self._below = below
        self._pos = 0

    def write(self, b: bytes) -> int:
        self._pos += len(b)
        return self.fh.write(b)

    def tell(self) -> int:
        # How much has been written to this wrapper, not how far into the file that ended up. ASDF
        # records absolute offsets in its block table, and both compressing and encrypting move the
        # file position away from the offset the reader will see once it has been undone.
        return self._pos

    def writable(self) -> bool:
        return True

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if not self.closed:
            super().close()
            self.fh.close()
            self._below.close()


class AsdfOutput(Output):
    """ASDF acquire output format.

    Writes an ASDF snapshot that mimics a full disk image. Filesystem metadata and the data of collected
    files are written at their real disk offsets, so the result can be opened as a regular disk image:
    the volume layout, filesystems and directory trees are all really there, and only the contents of
    files that were not collected read back as sparse. Anything that has no place on a disk - procfs,
    command output, acquire's own report - falls back to the metadata tar inside the snapshot.

    A compressed or encrypted snapshot is a transport format, the same way ``.tar.gz.enc`` is: reading a
    snapshot seeks to the footer for the block table and then seeks per block, so it has to be turned
    back into a plain ``.asdf`` before it can be opened.

    Args:
        path: The path to write the snapshot to.
        compress: Whether to gzip compress the snapshot.
        compression_method: Compression method to use. ASDF only supports "gzip".
        encrypt: Whether to encrypt the snapshot.
        public_key: The RSA public key to encrypt the header with.
    """

    def __init__(
        self,
        path: Path,
        compress: bool = False,
        compression_method: str = "gzip",
        encrypt: bool = False,
        public_key: bytes | None = None,
    ) -> None:
        self.compression = None

        ext = ".asdf" if ".asdf" not in path.suffixes else ""

        if compress:
            if compression_method and compression_method not in ASDF_COMPRESSION_METHODS:
                log.warning("ASDF only supports gzip compression, ignoring compression method %s", compression_method)

            self.compression = ASDF_COMPRESSION_METHODS["gzip"]
            ext += f".{self.compression}" if f".{self.compression}" not in path.suffixes else ""

        if encrypt:
            ext += ".enc"

        self.path = path.with_suffix(path.suffix + ext)

        # Both layers have to sit behind a _ContinuousStream, which is also why gzip is built here
        # instead of through AsdfWriter's own compress flag
        fh = self.path.open("wb")

        if encrypt:
            fh = _ContinuousStream(EncryptedStream(fh, public_key), fh)

        if compress:
            fh = _ContinuousStream(gzip.GzipFile(fileobj=fh, mode="wb", compresslevel=GZIP_COMPRESSION_LEVEL), fh)
            log.info("Writing a compressed snapshot, decompress it before opening it with dissect")

        self.asdf = AsdfWriter(fh, compress=False)

        # Maps id() of a disk or volume file-like object to the stream index and base offset to write at
        self._streams: dict[int, tuple[int, int]] = {}
        self._next_idx = 0

    def init(self, target: Target) -> None:
        """Map the disks and volumes of the target onto ASDF stream indices.

        Args:
            target: The target that is being acquired.
        """
        # Where the volumes of each disk live, which is what says how much of the disk describes the
        # layout rather than holds volume data
        layout: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for volume in target.volumes:
            layout[id(volume.disk)].append((volume.offset or 0, volume.size or 0))

        for disk in target.disks:
            idx = self._claim(disk)
            if idx is None:
                break

            self._write_volume_system(disk, idx, layout[id(disk)])
            self._write_anchor(disk, idx)

        for volume in target.volumes:
            if disk_stream := self._streams.get(id(volume.disk)):
                # The volume lives on a disk we already know, so write through to that disk
                self._streams[id(volume)] = (disk_stream[0], volume.offset or 0)
                continue

            # A volume without a disk we know about gets a stream of its own
            idx = self._claim(volume)
            if idx is None:
                break

            self._write_anchor(volume, idx)

        log.debug("Mapped %d disks and %d volumes onto ASDF streams", len(target.disks), len(target.volumes))

    def write_filesystem_metadata(self, fs: Filesystem, runs: MetadataRuns) -> int:
        """Write the enumerated metadata runs of a filesystem at their real disk offsets.

        Args:
            fs: The filesystem the runs belong to, used for logging.
            runs: The metadata runs to write.

        Returns:
            The amount of bytes written.
        """
        return self._write_runs(runs, fs)

    def _write_runs(self, runs: MetadataRuns, what: object, *, warn: bool = True) -> int:
        """Write a set of runs at their real disk offsets, returning the amount of bytes written.

        Args:
            runs: The runs to write.
            what: What the runs belong to, used for logging.
            warn: Whether a device that cannot be mapped onto a disk is worth warning about.
        """
        written = 0

        # Pinned bytes go first. The block table keeps whichever block claimed a region first, so
        # writing these up front stops a later copy off the live disk from overwriting them.
        for fh, offset, data in runs.literals():
            if stream := self._streams.get(id(fh)):
                idx, base = stream
                written += self._copy(io.BytesIO(data), 0, len(data), idx, base + offset, what)

        for fh, run_list in runs.items():
            stream = self._streams.get(id(fh))
            if stream is None:
                log.log(logging.WARNING if warn else logging.DEBUG, "Cannot map %s onto a disk, skipping it", what)
                continue

            idx, base = stream
            for offset, size in run_list:
                written += self._copy(fh, offset, size, idx, base, what)

        return written

    def write(
        self,
        output_path: str,
        fh: BinaryIO,
        entry: FilesystemEntry | Path | None = None,
        size: int | None = None,
    ) -> None:
        """Write a file-like object to the snapshot.

        If the data can be located on a disk it is written at its real offset, so it becomes readable
        through the filesystem itself. Otherwise it falls back to the metadata tar.

        Args:
            output_path: The path of the entry in the output.
            fh: The file-like object of the entry to write.
            entry: The optional filesystem entry to write.
            size: The optional file size in bytes of the entry to write.
        """
        # Resident, inline and compressed data has nowhere to go on a disk, and neither does anything
        # from a virtual filesystem. Those are the ones that fall through to the metadata tar below.
        try:
            # A device that is not mapped onto a disk is no cause for alarm here, unlike for filesystem
            # metadata: the file simply ends up in the metadata tar instead
            if (runs := fsmeta.file_runs(fh)) and self._write_runs(runs, output_path, warn=False):
                return
        except Exception as e:
            log.warning("Failed to write %s at its disk offsets, falling back to the metadata tar: %s", output_path, e)
            log.debug("", exc_info=e)

        # Used as handed over, the same way TarOutput does: the collector has already rooted it under
        # the filesystem it came from, and prefixing that again buries it a directory deeper
        path = output_path.lstrip("/")

        try:
            self.asdf.add_metadata_file(path, fh, size if size is not None else getattr(fh, "size", None))
        except Exception as e:
            log.warning("Failed to write %s to the metadata tar: %s", path, e)
            log.debug("", exc_info=e)

    def close(self) -> None:
        """Close the snapshot, writing the metadata tar, the block table and the footer.

        ``AsdfWriter`` closes the stream it was handed, and every :class:`_ContinuousStream` closes the
        layer below it, so the whole chain down to the file is finished from here.
        """
        self.asdf.close()

    def _claim(self, fh: BinaryIO) -> int | None:
        """Claim a stream index for a disk or volume, or return ``None`` if there are none left."""
        if self._next_idx > MAX_IDX:
            log.error("Out of ASDF stream indices, skipping remaining disks and volumes")
            return None

        idx = self._next_idx
        self._next_idx += 1
        self._streams[id(fh)] = (idx, 0)
        return idx

    def _copy(self, source: BinaryIO, offset: int, size: int, idx: int, base: int = 0, what: object = "data") -> int:
        """Copy a byte range into the snapshot and return the amount of bytes written.

        A disk that fails to read one range is still worth every other range it does have, so a failure
        is logged and skipped rather than raised.
        """
        try:
            self.asdf.copy_bytes(source, offset, size, idx=idx, base=base)
        except Exception as e:
            log.warning("Failed to write %#x+%#x of %s: %s", base + offset, size, what, e)
            log.debug("", exc_info=e)
            return 0

        return size

    def _write_volume_system(self, fh: BinaryIO, idx: int, volumes: list[tuple[int, int]]) -> None:
        """Write the parts of a disk that describe its volume layout.

        Without them the snapshot has no volumes, so none of the filesystem metadata written at volume
        offsets would ever be found.
        """
        size = size_of(fh) or 0

        for offset, length in volume_system_regions(volumes):
            # A disk smaller than the region it is asked for gets what is there and nothing beyond it
            length = min(length, max(0, size - offset)) if size else length

            if length > 0:
                self._copy(fh, offset, length, idx, what="volume system")

    def _write_anchor(self, fh: BinaryIO, idx: int) -> None:
        """Write the tail block of a stream so it reports the correct size.

        :class:`dissect.evidence.asdf.AsdfStream` derives its size from the last block in the table, so
        without this the image would appear truncated at the last block we happened to write.
        """
        size = size_of(fh)

        if size and size >= ANCHOR_SIZE:
            self._copy(fh, size - ANCHOR_SIZE, ANCHOR_SIZE, idx, what="size anchor")
