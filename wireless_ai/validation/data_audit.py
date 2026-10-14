"""Read-only metadata audit of CSI MAT files; never loads full arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_mat(path: Path, hash_content: bool = False) -> Dict[str, Any]:
    """Inspect metadata; reports omit host-specific parent directory names."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    variables: List[Dict[str, Any]] = []
    with path.open("rb") as handle:
        prefix = handle.read(128)
    if prefix.startswith(b"\x89HDF\r\n\x1a\n") or b"MATLAB 7.3 MAT-file" in prefix:
        import h5py

        with h5py.File(str(path), "r") as mat:
            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    variables.append({"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})

            mat.visititems(visit)
        container = "hdf5_mat"
    else:
        from scipy.io import whosmat

        variables = [
            {"name": name, "shape": list(shape), "dtype": dtype}
            for name, shape, dtype in whosmat(str(path))
        ]
        container = "mat_v5"
    result: Dict[str, Any] = {
        "path": path.name, "bytes": path.stat().st_size,
        "container": container, "variables": variables,
    }
    if hash_content:
        result["sha256"] = sha256_file(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hash", action="store_true", help="hash full contents; slower for multi-GB files")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        parser.error("output already exists; choose a new output filename")
    results = [inspect_mat(Path(name), args.hash) for name in args.paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"files": len(results), "out": output.name, "variables": sum(len(r["variables"]) for r in results)}))


if __name__ == "__main__":
    main()
