from __future__ import annotations

import pytest

from acquire.fsmeta.collector import MetadataRuns, RunCollector


def test_run_collector_rejects_bad_block_size() -> None:
    for block_size in (0, -512, 1000):
        with pytest.raises(ValueError, match="Invalid block size"):
            RunCollector(block_size)


@pytest.mark.parametrize(
    ("offset", "size", "expected"),
    [
        # Already aligned ranges are left alone
        (0, 4096, (0, 4096)),
        (4096, 8192, (4096, 8192)),
        # A range that starts mid block is rounded down to the start of that block
        (100, 10, (0, 4096)),
        (4196, 10, (4096, 4096)),
        # A range that ends mid block is rounded up to the end of that block
        (0, 1, (0, 4096)),
        (0, 4097, (0, 8192)),
        # A range that spans a boundary on both sides grows in both directions
        (4090, 12, (0, 8192)),
    ],
)
def test_run_collector_aligns(offset: int, size: int, expected: tuple[int, int]) -> None:
    collector = RunCollector(4096)
    collector.add(offset, size)

    assert collector.runs() == [expected]


def test_run_collector_ignores_empty() -> None:
    collector = RunCollector(4096)
    collector.add(0, 0)
    collector.add(4096, -1)

    assert collector.runs() == []


def test_run_collector_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="Negative offset"):
        RunCollector(4096).add(-4096, 4096)


def test_run_collector_merges_and_sorts() -> None:
    collector = RunCollector(512)

    # Out of order, overlapping, adjacent and disjoint
    collector.add(4096, 512)
    collector.add(0, 512)
    collector.add(512, 512)  # adjacent to the previous one
    collector.add(256, 1024)  # overlaps the first two
    collector.add(8192, 512)

    assert collector.runs() == [(0, 1536), (4096, 512), (8192, 512)]
    assert collector.size == 1536 + 512 + 512


def test_run_collector_add_block() -> None:
    collector = RunCollector(4096)
    collector.add_block(2)
    collector.add_block(10, 3)

    assert collector.runs() == [(8192, 4096), (40960, 12288)]


def test_run_collector_add_runlist_skips_sparse() -> None:
    collector = RunCollector(4096)
    collector.add_runlist([(0, 1), (None, 5), (10, 2)])

    assert collector.runs() == [(0, 4096), (40960, 8192)]


def test_run_collector_add_runlist_other_block_size() -> None:
    collector = RunCollector(512)
    collector.add_runlist([(1, 1)], block_size=4096)

    assert collector.runs() == [(4096, 4096)]


def test_run_collector_alignment_invariant_holds_for_arbitrary_input() -> None:
    """Every run must be block aligned, whatever is thrown at the collector.

    This is what keeps the gaps between runs an exact multiple of the block size, which is the regime
    every other ASDF writer operates in.
    """
    block_size = 4096
    collector = RunCollector(block_size)

    for offset in range(0, 100_000, 997):  # deliberately not a multiple of the block size
        collector.add(offset, offset % 5000 + 1)

    for offset, size in collector.runs():
        assert offset % block_size == 0
        assert size % block_size == 0


def test_metadata_runs_groups_per_device() -> None:
    runs = MetadataRuns(4096)

    first = object()
    second = object()

    runs.collector(first).add(0, 100)
    runs.collector(first).add(4096, 100)
    runs.collector(second).add(0, 100)

    items = dict(runs.items())
    assert len(items) == 2
    assert items[first] == [(0, 8192)]
    assert items[second] == [(0, 4096)]
    assert runs.size == 8192 + 4096


def test_metadata_runs_returns_same_collector_per_device() -> None:
    runs = MetadataRuns(4096)
    fh = object()

    assert runs.collector(fh) is runs.collector(fh)


def test_metadata_runs_block_size_override() -> None:
    runs = MetadataRuns(4096)
    fh = object()

    assert runs.collector(fh, block_size=512).block_size == 512
