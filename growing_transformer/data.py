"""Canonical text8 split: 90M train / 5M validation / 5M test characters."""

import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import torch

URL = "https://mattmahoney.net/dc/text8.zip"
# Pinned archive downloaded from URL; also validate size and alphabet after extraction.
ARCHIVE_MD5 = "f26f94c5209bc6159618bad4a559ff81"
VOCABULARY = " abcdefghijklmnopqrstuvwxyz"
SPLITS = {"train": (0, 90_000_000), "valid": (90_000_000, 95_000_000),
          "test": (95_000_000, 100_000_000)}


def prepare(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "text8.zip"
    if not archive.exists():
        temporary = directory / "text8.zip.tmp"
        try:
            urllib.request.urlretrieve(URL, temporary)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_MD5:
        raise ValueError(f"Invalid text8 archive checksum: {digest}")
    with zipfile.ZipFile(archive) as zipped:
        raw = zipped.read("text8")
    if len(raw) != 100_000_000:
        raise ValueError("Expected exactly 100 million text8 characters")
    lookup = np.full(256, 255, dtype=np.uint8)
    lookup[np.frombuffer(VOCABULARY.encode(), dtype=np.uint8)] = np.arange(27)
    encoded = lookup[np.frombuffer(raw, dtype=np.uint8)]
    if (encoded == 255).any():
        raise ValueError("Unexpected text8 alphabet")
    for name, (start, end) in SPLITS.items():
        encoded[start:end].tofile(directory / f"{name}.bin")
    print(f"Prepared text8 in {directory}: 90M / 5M / 5M characters")


def load_split(directory: Path, split: str):
    start, end = SPLITS[split]
    path = directory / f"{split}.bin"
    if not path.exists() or path.stat().st_size != end - start:
        raise ValueError(f"Missing or invalid {path}; run python -m growing_transformer.data")
    return np.memmap(path, dtype=np.uint8, mode="r")


def batch(data, batch_size, context, generator, device):
    if len(data) <= context:
        raise ValueError("Dataset is shorter than context plus one target")
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator)
    sequences = np.stack([data[s:s + context + 1] for s in starts.tolist()])
    tokens = torch.from_numpy(sequences.astype(np.int64)).to(device)
    return tokens[:, :-1], tokens[:, 1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("data/text8"))
    prepare(parser.parse_args().directory)
