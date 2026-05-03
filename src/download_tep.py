"""
Download the Tennessee Eastman Process (TEP) dataset from Harvard Dataverse.

Source: Rieth, Amsel, Tran, Cook (2017). Additional Tennessee Eastman Process
Simulation Data for Anomaly Detection Evaluation. Harvard Dataverse, V1.
https://doi.org/10.7910/DVN/6C3JR1

This script:
  1. Queries the Dataverse API for the dataset's current file metadata
  2. Resolves file names -> file IDs dynamically (resilient to ID changes)
  3. Downloads each .RData file with progress, resumable on retry
"""

from pathlib import Path
import json
import sys
import urllib.request
from tqdm import tqdm

DOI = "doi:10.7910/DVN/6C3JR1"
BASE = "https://dataverse.harvard.edu"
API_LIST = f"{BASE}/api/datasets/:persistentId/versions/:LATEST/files?persistentId={DOI}"
API_FETCH = f"{BASE}/api/access/datafile"

WANTED = {
    "TEP_FaultFree_Training.RData",
    "TEP_Faulty_Training.RData",
    "TEP_FaultFree_Testing.RData",
    "TEP_Faulty_Testing.RData",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
CHUNK_SIZE = 1024 * 256


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def list_files() -> dict[str, int]:
    """Query Dataverse API and return {filename: dataFile_id} for the wanted files."""
    print(f"Querying Dataverse API for {DOI} ...")
    with urllib.request.urlopen(_request(API_LIST), timeout=60) as resp:
        meta = json.load(resp)

    if meta.get("status") != "OK":
        raise RuntimeError(f"Dataverse API returned: {meta}")

    found: dict[str, int] = {}
    all_names: list[str] = []
    for item in meta["data"]:
        df = item.get("dataFile", {})
        name = df.get("filename")
        fid = df.get("id")
        if name and fid is not None:
            all_names.append(name)
            if name in WANTED:
                found[name] = fid

    missing = WANTED - set(found.keys())
    if missing:
        print("ERROR: expected files not found in dataset.", file=sys.stderr)
        print(f"Wanted: {sorted(WANTED)}", file=sys.stderr)
        print(f"Available: {sorted(all_names)}", file=sys.stderr)
        raise RuntimeError(f"Missing: {sorted(missing)}")

    print(f"Resolved {len(found)} file IDs.\n")
    for name, fid in sorted(found.items()):
        print(f"  {name} -> id={fid}")
    print()
    return found


def download(name: str, fid: int, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [skip] {name} already exists ({size_mb:.1f} MB)")
        return

    url = f"{API_FETCH}/{fid}"
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(_request(url), timeout=60) as resp:
        total = resp.getheader("Content-Length")
        total = int(total) if total else None
        with open(tmp, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=name,
        ) as bar:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

    tmp.rename(dest)


def main() -> None:
    print(f"Download target: {RAW_DIR}\n")
    try:
        files = list_files()
    except Exception as e:
        print(f"  [error] could not list dataset files: {e}", file=sys.stderr)
        sys.exit(1)

    for name, fid in files.items():
        dest = RAW_DIR / name
        try:
            download(name, fid, dest)
        except Exception as e:
            print(f"  [error] {name}: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll files downloaded.")
    print(f"Location: {RAW_DIR}")
    total_mb = sum(f.stat().st_size for f in RAW_DIR.glob("*.RData")) / (1024 * 1024)
    print(f"Total size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()