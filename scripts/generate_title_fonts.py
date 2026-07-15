"""Decompress @fontsource woff2 title fonts to ttf for matplotlib (offline, build-time)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fontTools.ttLib.woff2 import decompress


def decompress_woff2(woff2_path: str, ttf_path: str) -> None:
    Path(ttf_path).parent.mkdir(parents=True, exist_ok=True)
    decompress(woff2_path, ttf_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON: [{id, roles:{bold,regular,light: woff2_path}}]")
    parser.add_argument("--out", required=True, help="Output dir root (fonts/generated)")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    for font in manifest:
        for role, woff2 in font["roles"].items():
            decompress_woff2(woff2, str(Path(args.out) / font["id"] / f"{role}.ttf"))


if __name__ == "__main__":
    main()
