"""Check copied-source hashes and Python syntax without importing experiments."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "PROVENANCE.json").read_text())
    python_count = 0
    for entry in manifest["files"]:
        path = root / entry["destination"]
        body = path.read_bytes()
        assert len(body) == entry["bytes"], entry["destination"]
        assert hashlib.sha256(body).hexdigest() == entry["sha256"], entry["destination"]
        if path.suffix == ".py":
            ast.parse(body, filename=entry["destination"])
            python_count += 1
    print(f"Verified {len(manifest['files'])} exact source/document copies; "
          f"parsed {python_count} Python files. No experiment was imported or run.")


if __name__ == "__main__":
    main()
