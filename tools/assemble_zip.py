"""Rebuild the submitted SimulacraBench archive and verify it byte for byte.

    python assemble_zip.py --weights DIR [--out rc6-profile.zip]

DIR must contain the three public checkpoints (see README) and the three
third-party license files. Everything else comes from this repository's
`submission/` folder. The script checks every member's SHA-256 against
MANIFEST.json and then the SHA-256 of the whole archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", default="rc6-profile.zip", type=Path)
    args = parser.parse_args()
    sources = []
    for member in MANIFEST["members"]:
        name = member["name"]
        path = ROOT / "submission" / name
        if not path.exists():
            path = args.weights / name
        if not path.exists():
            raise SystemExit("missing member: %s" % name)
        digest = sha256(path)
        if digest != member["sha256"]:
            raise SystemExit("member hash mismatch: %s" % name)
        sources.append((member, path))
    with zipfile.ZipFile(args.out, "w") as archive:
        for member, path in sources:
            info = zipfile.ZipInfo(member["name"], date_time=tuple(member["date_time"]))
            info.compress_type = member["compress_type"]
            info.external_attr = member["external_attr"]
            info.create_system = member["create_system"]
            with open(path, "rb") as stream, archive.open(info, "w", force_zip64=False) as target:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    target.write(chunk)
    digest = sha256(args.out)
    print(args.out, args.out.stat().st_size, digest)
    if digest != MANIFEST["archive"]["sha256"]:
        raise SystemExit("archive hash differs from the submitted archive")
    print("byte-identical to the submitted archive")


if __name__ == "__main__":
    main()
