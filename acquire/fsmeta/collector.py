from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator


class RunCollector:
    """Collect byte ranges of a single device, rounded out to whole blocks and merged.

    ASDF is a block oriented format, which is why :attr:`dissect.evidence.asdf.DEFAULT_BLOCK_SIZE` exists.
    Every range added here is therefore rounded out to a block boundary before it is stored, and adjacent or
    overlapping ranges are merged. Keeping every run block aligned means the gaps between runs are always a
    multiple of the block size as well, which is the regime every other ASDF writer operates in.

    Rounding out is also useful in its own right: the tail block of a file is written whole, so file slack
    ends up in the snapshot without ever including data outside of the allocation of the file.

    Args:
        block_size: The block size to align all runs to.
    """

    def __init__(self, block_size: int):
        if block_size <= 0 or block_size % 512:
            raise ValueError(f"Invalid block size: {block_size}")

        self.block_size = block_size
        self._runs: list[tuple[int, int]] = []
        self._merged: list[tuple[int, int]] | None = None

    def __repr__(self) -> str:
        return f"<RunCollector block_size={self.block_size} runs={len(self._runs)}>"

    def add(self, offset: int, size: int) -> None:
        """Add a byte range, rounding it out to whole blocks.

        Args:
            offset: The byte offset of the range.
            size: The size of the range in bytes.
        """
        if size <= 0:
            return

        if offset < 0:
            raise ValueError(f"Negative offset: {offset}")

        start = offset - (offset % self.block_size)
        end = -(-(offset + size) // self.block_size) * self.block_size

        self._runs.append((start, end - start))
        self._merged = None

    def add_block(self, block: int, count: int = 1) -> None:
        """Add a range by block number.

        Args:
            block: The block number to add.
            count: The amount of blocks to add.
        """
        self.add(block * self.block_size, count * self.block_size)

    def add_runlist(self, runlist: list[tuple[int | None, int]], block_size: int | None = None) -> None:
        """Add a runlist, skipping sparse runs.

        Args:
            runlist: A list of ``(block_offset, num_blocks)`` tuples, where a ``None`` offset is a sparse run.
            block_size: The block size of the runlist, defaults to the block size of this collector.
        """
        block_size = block_size or self.block_size
        for run_offset, run_length in runlist:
            if run_offset is None:
                continue
            self.add(run_offset * block_size, run_length * block_size)

    def runs(self) -> list[tuple[int, int]]:
        """Return all collected runs, sorted and merged."""
        if self._merged is None:
            merged: list[tuple[int, int]] = []

            for offset, size in sorted(self._runs):
                if merged and offset <= merged[-1][0] + merged[-1][1]:
                    end = max(merged[-1][0] + merged[-1][1], offset + size)
                    merged[-1] = (merged[-1][0], end - merged[-1][0])
                else:
                    merged.append((offset, size))

            self._merged = merged

        return self._merged

    @property
    def size(self) -> int:
        """The total amount of bytes covered by all collected runs."""
        return sum(size for _, size in self.runs())


class MetadataRuns:
    """The metadata runs of a filesystem, grouped by the device they belong to.

    Most filesystems live on a single device, but Btrfs can span several, so runs are tracked per
    device file-like object.

    Args:
        block_size: The block size to align all runs to.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self._collectors: dict[int, tuple[BinaryIO, RunCollector]] = {}
        self._literals: list[tuple[BinaryIO, int, bytes]] = []

    def __repr__(self) -> str:
        return f"<MetadataRuns block_size={self.block_size} devices={len(self._collectors)} size={self.size}>"

    def add_literal(self, fh: BinaryIO, offset: int, data: bytes) -> None:
        """Pin an exact set of bytes to an offset, instead of copying whatever is on disk later on.

        Copy on write filesystems move their trees on every transaction commit. Enumeration walks the
        trees the superblock pointed at when the filesystem was opened, but the bytes for each run are
        only read afterwards - by which time a commit may have left the superblock pointing somewhere
        else entirely, at a tree that was never collected.

        Pinning the superblock as it was parsed keeps the snapshot internally consistent: it points at
        exactly the trees that were walked.

        Args:
            fh: The device the bytes belong to.
            offset: The byte offset to pin them at.
            data: The bytes to write.
        """
        if data:
            self._literals.append((fh, offset, data))

    def literals(self) -> Iterator[tuple[BinaryIO, int, bytes]]:
        """Iterate over all pinned byte ranges."""
        yield from self._literals

    def collector(self, fh: BinaryIO, block_size: int | None = None) -> RunCollector:
        """Return the :class:`RunCollector` for the given device, creating it if needed.

        Args:
            fh: The device file-like object to get the collector for.
            block_size: Optional block size override for this device.
        """
        key = id(fh)
        if key not in self._collectors:
            self._collectors[key] = (fh, RunCollector(block_size or self.block_size))
        return self._collectors[key][1]

    def items(self) -> Iterator[tuple[BinaryIO, list[tuple[int, int]]]]:
        """Iterate over all devices and their sorted, merged runs."""
        for fh, collector in self._collectors.values():
            yield fh, collector.runs()

    @property
    def size(self) -> int:
        """The total amount of bytes covered across all devices."""
        return sum(collector.size for _, collector in self._collectors.values())
