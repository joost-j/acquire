from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dissect.evidence import AsdfSnapshot
from dissect.evidence.asdf.c_asdf import c_asdf
from dissect.evidence.exception import InvalidSnapshot

import acquire.acquire as acquire_module
from acquire.acquire import (
    PROFILES,
    BsdProfile,
    ESXiProfile,
    FilesystemMetadata,
    LinuxProfile,
    MacOSProfile,
    ProxmoxProfile,
    WindowsProfile,
)
from acquire.outputs.asdf import GZIP_COMPRESSION_LEVEL, AsdfOutput, volume_system_regions

PROFILE_CLASSES = [WindowsProfile, LinuxProfile, BsdProfile, ESXiProfile, MacOSProfile, ProxmoxProfile]


@pytest.mark.parametrize("profile_cls", PROFILE_CLASSES)
def test_not_in_minimal_profile(profile_cls: type) -> None:
    """The minimal profile stays lightweight, and filesystem metadata is not free."""
    assert FilesystemMetadata not in profile_cls.MINIMAL


@pytest.mark.parametrize("profile_cls", PROFILE_CLASSES)
def test_in_default_and_full_profiles(profile_cls: type) -> None:
    assert FilesystemMetadata in profile_cls.DEFAULT
    assert FilesystemMetadata in profile_cls.FULL


def test_registered_for_every_os() -> None:
    for profile in ("default", "full"):
        for os_name, modules in PROFILES[profile].items():
            assert FilesystemMetadata in modules, f"missing from {profile}/{os_name}"


@pytest.mark.parametrize(
    ("profile", "explicit", "expected"),
    [
        # The profile decides when the flag is not given
        ("default", None, True),
        ("full", None, False),
        # Running the module on its own, without a profile, keeps the compact default
        (None, None, True),
        # An explicit flag always wins over the profile
        ("full", True, True),
        ("default", False, False),
    ],
)
def test_thin_follows_profile(profile: str | None, explicit: bool | None, expected: bool) -> None:
    args = argparse.Namespace(profile=profile, filesystem_metadata_thin=explicit)

    assert FilesystemMetadata._thin(args) is expected


def test_skips_non_asdf_output(caplog: pytest.LogCaptureFixture) -> None:
    """Only ASDF can store metadata sparsely at its original disk offsets."""
    collector = MagicMock()
    collector.output = MagicMock()  # anything but an AsdfOutput
    target = MagicMock()

    FilesystemMetadata._run(target, argparse.Namespace(profile="default"), collector)

    assert "needs the asdf output format" in caplog.text
    target.filesystems.__iter__.assert_not_called()


def test_collects_every_supported_filesystem_once(tmp_path: Path) -> None:
    """Each filesystem is enumerated once, even when mounted in several places."""
    output = AsdfOutput(tmp_path / "out")

    fs = MagicMock()
    fs.__type__ = "ext"

    target = MagicMock()
    # The same filesystem object, mounted twice
    target.filesystems = [fs, fs]

    written = []
    output.write_filesystem_metadata = lambda _fs, runs: written.append(runs) or 0

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(acquire_module.fsmeta, "supported", lambda _: True)
            mp.setattr(acquire_module.fsmeta, "enumerate_runs", lambda _fs, **_kw: MagicMock())

            collector = MagicMock()
            collector.output = output

            FilesystemMetadata._run(target, argparse.Namespace(profile="default"), collector)
    finally:
        output.close()

    assert len(written) == 1


@pytest.mark.parametrize(
    ("compress", "encrypt", "expected"),
    [
        (False, False, ".asdf"),
        (True, False, ".asdf.gz"),
        (False, True, ".asdf.enc"),
        (True, True, ".asdf.gz.enc"),
    ],
)
def test_extension_reflects_compression_and_encryption(
    tmp_path: Path, compress: bool, encrypt: bool, expected: str
) -> None:
    """A compressed snapshot must not be named .asdf.

    It is a transport format that has to be turned back into a plain snapshot first, and naming it
    .asdf makes it look directly openable when it is not.
    """
    public_key = None
    if encrypt:
        rsa = pytest.importorskip("Crypto.PublicKey.RSA")
        public_key = rsa.generate(2048).publickey().export_key().decode()

    output = AsdfOutput(tmp_path / "out", compress=compress, encrypt=encrypt, public_key=public_key)
    output.close()

    assert Path(output.path).name.endswith(expected)


def test_uncompressed_snapshot_is_a_plain_asdf_file(tmp_path: Path) -> None:
    output = AsdfOutput(tmp_path / "out")
    output.asdf.add_bytes(b"\x00" * 512, base=0)
    output.close()

    with Path(output.path).open("rb") as fh:
        assert AsdfSnapshot(fh).table


def test_compressed_snapshot_is_gzip(tmp_path: Path) -> None:
    output = AsdfOutput(tmp_path / "out", compress=True)
    output.asdf.add_bytes(b"\x00" * 512, base=0)
    output.close()

    path = Path(output.path)
    assert path.name.endswith(".asdf.gz")
    assert path.read_bytes()[:2] == b"\x1f\x8b"

    # ...and gunzips back into something that opens
    plain = path.with_suffix("")
    plain.write_bytes(gzip.decompress(path.read_bytes()))
    with plain.open("rb") as fh:
        assert AsdfSnapshot(fh).table


def test_data_without_a_disk_goes_into_the_metadata_tar(tmp_path: Path) -> None:
    """Anything that has no place on a disk - command output, the report, an empty directory.

    The paths are used as handed over: the collector has already rooted them under the filesystem they
    came from, so keeping the leading slash would bury the whole tar a directory deeper.
    """
    entry = MagicMock()
    entry.is_dir.return_value = True

    output = AsdfOutput(tmp_path / "out")
    output.write_bytes("commands/whoami.txt", b"root\n")
    output.write_bytes("/etc/hosts", b"127.0.0.1 localhost\n")
    output.write_entry("/var/log", entry)
    output.close()

    with Path(output.path).open("rb") as fh:
        metadata = AsdfSnapshot(fh).metadata

        assert sorted(metadata.names()) == ["commands/whoami.txt", "etc/hosts", "var/log"]
        assert metadata.open("commands/whoami.txt").read() == b"root\n"
        assert metadata.open("var/log").read() == b""


def _snapshot(path: Path, **kwargs) -> Path:
    """Write a snapshot of the same content, however it is being packaged."""
    output = AsdfOutput(path, **kwargs)
    for i in range(256):
        output.asdf.add_bytes(bytes([i % 251]) * 4096, base=i * 4096)
    output.close()
    return Path(output.path)


def test_compressing_only_wraps_the_snapshot(tmp_path: Path) -> None:
    """Decompressing has to give back exactly the snapshot that would have been written uncompressed.

    The header and the footer are left out of the comparison: the header carries a timestamp and a GUID
    that are generated per writer, and the footer hashes them.
    """
    plain = _snapshot(tmp_path / "plain").read_bytes()
    packed = gzip.decompress(_snapshot(tmp_path / "packed", compress=True).read_bytes())

    body = slice(len(c_asdf.header), -len(c_asdf.footer))
    assert packed[body] == plain[body]


def test_compression_is_not_restarted_per_block(tmp_path: Path) -> None:
    """The deflate stream has to run over the whole snapshot in one go.

    ASDF finalizes every block it writes, and that cascades into a ``Z_SYNC_FLUSH`` on the gzip stream
    below it. Flushing per block ends the deflate block and emits a fresh Huffman table each time,
    which on sparse metadata costs several times the output size.
    """
    plain = _snapshot(tmp_path / "plain").read_bytes()
    packed = _snapshot(tmp_path / "packed", compress=True)

    at_once = len(gzip.compress(plain, compresslevel=GZIP_COMPRESSION_LEVEL))

    assert packed.stat().st_size < at_once * 1.05


def test_truncated_snapshot_can_still_be_recovered(tmp_path: Path) -> None:
    """An acquisition that is killed partway should still yield what it managed to write.

    The block table and footer are only written on close, so a truncated file has neither. ASDF can
    scrape the block headers back out, which is why the filesystem metadata is written first.
    """
    output = AsdfOutput(tmp_path / "out")
    for i in range(8):
        output.asdf.add_bytes(bytes([i]) * 512, base=i * 4096)
    output.close()

    path = Path(output.path)
    path.write_bytes(path.read_bytes()[: int(path.stat().st_size * 0.6)])

    with path.open("rb") as fh, pytest.raises(InvalidSnapshot):
        AsdfSnapshot(fh)

    with path.open("rb") as fh:
        assert AsdfSnapshot(fh, recover=True).table


MIB = 1024 * 1024


@pytest.mark.parametrize(
    ("volumes", "expected"),
    [
        # A GPT disk with one aligned partition: only the head is partition table
        ([(MIB, 100 * MIB)], [(0, MIB)]),
        # An MBR extended partition: the logical volume is described by an extended boot record just
        # in front of it, which is nowhere near the head of the disk
        (
            [(MIB, 512 * MIB), (538968064, 63884492800)],
            [(0, MIB), (538968064 - MIB, MIB)],
        ),
        # A gap smaller than the cap is collected whole, and never reaches into the volume before it
        ([(MIB, MIB), (3 * MIB, MIB)], [(0, MIB), (2 * MIB, MIB)]),
        # Back to back volumes leave no room for anything to describe them
        ([(MIB, MIB), (2 * MIB, MIB)], [(0, MIB)]),
        # A filesystem straight on the disk has no partition table, and its data is not ours to take
        ([(0, 100 * MIB)], []),
        # Nothing is known about the disk, so collect a head and hope for the best
        ([], [(0, MIB)]),
        # Volumes are not necessarily handed to us in order
        ([(3 * MIB, MIB), (MIB, MIB)], [(0, MIB), (2 * MIB, MIB)]),
    ],
)
def test_volume_system_regions(volumes: list[tuple[int, int]], expected: list[tuple[int, int]]) -> None:
    assert list(volume_system_regions(volumes)) == expected
