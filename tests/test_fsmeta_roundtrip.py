"""Round trip tests: build a real filesystem image, snapshot only its metadata, and read it back.

These are the tests that matter most: the whole point of the feature is that a sparse snapshot walks
identically to the full image it came from. They are also the ones a CI runner is least able to run,
since they need the ``mkfs`` family installed, half a gigabyte of scratch space and about a minute. So
they are marked as regression tests and left out of a plain pytest run - use ``tox -e regression``.

Whatever tooling is missing is skipped rather than failed, so running a subset is fine.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from dissect.evidence import AsdfSnapshot
from dissect.target import Target

from acquire.fsmeta import enumerate_runs
from acquire.outputs.asdf import AsdfOutput

if TYPE_CHECKING:
    from dissect.target.filesystem import Filesystem, FilesystemEntry

pytestmark = pytest.mark.regression

IMAGE_SIZE = 128 * 1024 * 1024

# mkfs.xfs refuses to create a filesystem smaller than this
XFS_IMAGE_SIZE = 512 * 1024 * 1024

# Enough files to push directories past the inline/resident form into real index structures
FILE_COUNT = 250

# Contents of the collected file. Small files are stored inline in the metadata on several of these
# filesystems, which would make them read back correctly no matter where their data was written.
COLLECTED_CONTENT = b"collected\n" * 25_000


def _have(*tools: str) -> bool:
    return all(shutil.which(tool) for tool in tools)


def _run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


@pytest.fixture(scope="session")
def content(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory tree to populate the test filesystems with."""
    root = tmp_path_factory.mktemp("content")

    (root / "etc").mkdir()
    (root / "etc" / "hostname").write_text("acquire-test\n")
    (root / "etc" / "passwd").write_text("root:x:0:0::/root:/bin/bash\n")

    docs = root / "docs"
    docs.mkdir()
    for i in range(FILE_COUNT):
        (docs / f"a_rather_long_test_filename_{i:04d}.txt").write_text(f"contents of file {i}\n" + "A" * 200)

    # A file large enough that its data is never stored inline in the metadata
    (root / "big.bin").write_bytes(b"\x42" * (2 * 1024 * 1024))
    # The same, but collected, so its data has to be written at its real offset rather than inlined
    (root / "collected.bin").write_bytes(COLLECTED_CONTENT)

    (root / "link").symlink_to("/etc/passwd")
    # A symlink whose target is too long to fit inside the inode
    (root / "longlink").symlink_to("/" + "x" * 200)

    return root


def _make_ext4(path: Path, content: Path) -> None:
    _run("truncate", "-s", str(IMAGE_SIZE), str(path))
    _run("mkfs.ext4", "-q", "-F", "-d", str(content), str(path))


def _make_btrfs(path: Path, content: Path) -> None:
    _run("truncate", "-s", str(IMAGE_SIZE), str(path))
    _run("mkfs.btrfs", "-q", "-f", "--rootdir", str(content), str(path))


def _make_xfs(path: Path, content: Path) -> None:
    # XFS is populated through a protofile, which cannot express nested trees as conveniently
    proto = path.with_suffix(".proto")
    lines = [
        "dummy 0 0",
        "d--755 0 0",
        "etc d--755 0 0",
        f"hostname ---644 0 0 {content / 'etc' / 'hostname'}",
        f"passwd ---644 0 0 {content / 'etc' / 'passwd'}",
        "$",
        "docs d--755 0 0",
    ]
    lines.extend(f"{entry.name} ---644 0 0 {entry}" for entry in sorted((content / "docs").iterdir()))
    lines += [
        "$",
        f"big.bin ---644 0 0 {content / 'big.bin'}",
        f"collected.bin ---644 0 0 {content / 'collected.bin'}",
        "link l--777 0 0 /etc/passwd",
        "$",
    ]
    proto.write_text("\n".join(lines) + "\n")

    _run("truncate", "-s", str(XFS_IMAGE_SIZE), str(path))
    _run("mkfs.xfs", "-q", "-f", "-p", str(proto), str(path))


def _make_ntfs(path: Path, content: Path) -> None:
    _run("truncate", "-s", str(IMAGE_SIZE), str(path))
    _run("mkntfs", "-q", "-F", str(path))

    _run("ntfscp", "-q", str(path), str(content / "etc" / "hostname"), "/hostname")
    _run("ntfscp", "-q", str(path), str(content / "big.bin"), "/big.bin")
    _run("ntfscp", "-q", str(path), str(content / "collected.bin"), "/collected.bin")
    for entry in sorted((content / "docs").iterdir()):
        _run("ntfscp", "-q", str(path), str(entry), f"/{entry.name}")


BUILDERS = {
    "ext": (_make_ext4, ("truncate", "mkfs.ext4")),
    "xfs": (_make_xfs, ("truncate", "mkfs.xfs")),
    "btrfs": (_make_btrfs, ("truncate", "mkfs.btrfs")),
    "ntfs": (_make_ntfs, ("truncate", "mkntfs", "ntfscp")),
}


def _walk(fs: Filesystem, path: str = "/", depth: int = 0) -> list[str]:
    """Recursively describe every entry of a filesystem, metadata included."""
    out = []
    if depth > 32:
        return out

    for name in sorted(fs.listdir(path)):
        full = path.rstrip("/") + "/" + name

        try:
            entry = fs.get(full)
            st = entry.lstat()
            link = entry.readlink() if entry.is_symlink() else ""
            out.append(
                f"{full}\t{st.st_ino}\t{st.st_size}\t{st.st_mode}\t{st.st_uid}\t"
                f"{st.st_gid}\t{st.st_mtime}\t{st.st_atime}\t{st.st_ctime}\t{link}"
            )
        except Exception as e:
            # Some NTFS system entries with named streams raise in dissect.target itself. Record the
            # failure rather than skipping it, so a snapshot that fails differently still stands out.
            out.append(f"{full}\t<{type(e).__name__}>")
            continue

        if entry.is_dir(follow_symlinks=False):
            out.extend(_walk(fs, full, depth + 1))

    return out


def _find(fs: Filesystem, *paths: str) -> FilesystemEntry | None:
    for path in paths:
        try:
            return fs.get(path)
        except Exception:  # noqa: PERF203
            continue
    return None


@pytest.fixture(params=sorted(BUILDERS), scope="session")
def snapshot(
    request: pytest.FixtureRequest, content: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Path, Path]:
    """Build an image of the requested type and snapshot only its filesystem metadata.

    Session scoped: building a filesystem image per test would be both slow and needlessly heavy on
    disk, and none of the tests mutate what they are given.
    """
    fs_type = request.param
    builder, tools = BUILDERS[fs_type]

    if not _have(*tools):
        pytest.skip(f"{fs_type} needs {', '.join(tools)}")

    tmp_path = tmp_path_factory.mktemp(f"snapshot-{fs_type}")
    image = tmp_path / f"test.{fs_type}.img"
    builder(image, content)

    target = Target.open(str(image))
    output = AsdfOutput(tmp_path / "out")
    output.init(target)

    for fs in target.filesystems:
        runs = enumerate_runs(fs)
        if runs is not None:
            output.write_filesystem_metadata(fs, runs)

        # Collect a couple of files, the way acquire would
        for path, name in (("/etc/hostname", "/hostname"), ("/collected.bin", "/collected.bin")):
            entry = _find(fs, path, name)
            if entry is not None:
                output.write_entry(name, entry)

    output.close()

    return fs_type, image, Path(output.path)


def _filesystems(path: Path, fs_type: str) -> list[Filesystem]:
    return [fs for fs in Target.open(str(path)).filesystems if fs.__type__ == fs_type]


def test_snapshot_walks_identically_to_the_full_image(snapshot: tuple[str, Path, Path]) -> None:
    """The whole point: a metadata only snapshot must describe the tree exactly like the full image."""
    fs_type, image, asdf = snapshot

    original = _filesystems(image, fs_type)
    restored = _filesystems(asdf, fs_type)

    assert len(original) == len(restored) == 1

    expected = _walk(original[0])
    assert len(expected) > FILE_COUNT, "the test image should have a meaningful number of entries"
    assert _walk(restored[0]) == expected


def test_snapshot_is_much_smaller_than_the_image(snapshot: tuple[str, Path, Path]) -> None:
    _, image, asdf = snapshot

    assert asdf.stat().st_size < image.stat().st_size / 2


def test_block_table_is_block_aligned(snapshot: tuple[str, Path, Path]) -> None:
    """Every stored block must be sector aligned.

    Byte granular runs leave odd sized gaps between them, which older readers do not reproduce
    faithfully. Staying aligned keeps the output readable by any dissect.evidence version.
    """
    _, _, asdf = snapshot

    with asdf.open("rb") as fh:
        table = AsdfSnapshot(fh).table

        assert table, "the snapshot should contain blocks"

        for idx, entries in table.items():
            if idx > 253:  # reserved metadata/memory streams
                continue

            for offset, size, _, _ in entries:
                assert offset % 512 == 0, f"stream {idx} has a misaligned block at {offset:#x}"
                assert size % 512 == 0, f"stream {idx} has a block of unaligned size {size:#x}"


def test_collected_file_has_real_content(snapshot: tuple[str, Path, Path]) -> None:
    fs_type, _, asdf = snapshot
    fs = _filesystems(asdf, fs_type)[0]

    entry = _find(fs, "/etc/hostname", "/hostname")
    assert entry is not None
    assert entry.open().read() == b"acquire-test\n"


def test_collected_file_is_readable_through_the_filesystem(snapshot: tuple[str, Path, Path]) -> None:
    """A collected file has to be browsable, not just present in the metadata tar.

    Small files are stored inline in the metadata by btrfs and APFS, so this uses one that is big
    enough to end up in real extents. Those two resolve their extents per read instead of handing out
    a runlist, which used to leave the data in the tar and the file reading back as filler.
    """
    fs_type, _, asdf = snapshot
    fs = _filesystems(asdf, fs_type)[0]

    entry = _find(fs, "/collected.bin")
    assert entry is not None
    assert entry.open().read() == COLLECTED_CONTENT


def test_uncollected_file_reads_back_sparse(snapshot: tuple[str, Path, Path]) -> None:
    """Data that was never collected must not come back as plausible looking content."""
    fs_type, _, asdf = snapshot
    fs = _filesystems(asdf, fs_type)[0]

    entry = _find(fs, "/big.bin")
    assert entry is not None

    data = entry.open().read(64)
    assert data == b"\xa5\xdf" * 32
    # The metadata is still intact, only the contents are gone
    assert entry.lstat().st_size == 2 * 1024 * 1024
