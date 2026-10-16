#!/usr/bin/env python3
"""Verify the separately downloaded V4 ZIP and install its six inference files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA256 = 'b8ee949d2e5def582a3f6a9ed2f055e7f3ec844f2c87c74cbea3b795950137c1'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def install(archive):
    expected = json.loads((ROOT / 'model_integrity.json').read_text())
    destination = ROOT / 'best_model'
    if destination.exists():
        if all((destination / name).is_file() and digest(destination / name) == sha
               for name, sha in expected.items()):
            print('The exact V4 model is already installed and verified.')
            return
        raise ValueError('best_model exists but is incomplete or different. Rename it before installing; no files were overwritten.')
    if not archive.is_file():
        raise ValueError(f'Model ZIP not found: {archive}')
    print('Verifying the model ZIP...', flush=True)
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError('ZIP checksum mismatch. Obtain the original V4 model ZIP; nothing was installed.')
    with tempfile.TemporaryDirectory(prefix='.model-install-', dir=ROOT) as temporary:
        staging = Path(temporary) / 'best_model'
        staging.mkdir()
        with zipfile.ZipFile(archive) as saved:
            if saved.testzip() is not None:
                raise ValueError('ZIP integrity check failed.')
            for name, checksum in expected.items():
                # Select exact known filenames; ignore README/scripts/pickled training state.
                with saved.open('best_model/' + name) as source, (staging / name).open('wb') as target:
                    shutil.copyfileobj(source, target)
                if digest(staging / name) != checksum:
                    raise ValueError(f'Model file checksum mismatch: {name}')
        staging.rename(destination)
    print('Installed and verified V4 model in', destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='Path to latent_commitment_workplace_v4_model.zip')
    args = parser.parse_args()
    try:
        install(args.archive.expanduser().resolve())
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(1, f'Installation failed: {error}\n')


if __name__ == '__main__':
    main()
