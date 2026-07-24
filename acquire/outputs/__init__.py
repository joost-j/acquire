from __future__ import annotations

from acquire.outputs.asdf import ASDF_COMPRESSION_METHODS, AsdfOutput
from acquire.outputs.dir import DirectoryOutput
from acquire.outputs.tar import TAR_COMPRESSION_METHODS, TarOutput
from acquire.outputs.zip import ZIP_COMPRESSION_METHODS, ZipOutput

__all__ = ["ASDF_COMPRESSION_METHODS", "AsdfOutput", "DirectoryOutput", "TarOutput", "ZipOutput"]

OUTPUTS = {"tar": TarOutput, "dir": DirectoryOutput, "zip": ZipOutput, "asdf": AsdfOutput}

COMPRESSION_METHODS = {*TAR_COMPRESSION_METHODS, *ZIP_COMPRESSION_METHODS, *ASDF_COMPRESSION_METHODS}
