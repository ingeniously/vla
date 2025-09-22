#!/usr/bin/env python3
"""
Extract all .tar.gz/.tgz/.tar archives in a folder into subfolders with the same base names.

Example
  python extract_archives.py --src "C:/Users/USER/Pictures/BASEPIC/covla-mini/images"


Behavior
- For 2022-07-14--14-32-55--10_first.tar.gz → extracts into
  C:/.../images/2022-07-14--14-32-55--10_first/
- Skips already extracted folders unless --force is provided.
- Performs path safety checks to avoid writing outside the target folder.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import List

from tqdm import tqdm


def infer_output_dir(archive_path: Path) -> Path:
    name = archive_path.name
    if name.endswith('.tar.gz'):
        stem = name[:-7]
    elif name.endswith('.tgz'):
        stem = name[:-4]
    elif name.endswith('.tar'):
        stem = name[:-4]
    else:
        stem = archive_path.stem
    return archive_path.parent / stem


def is_within_dir(base: Path, target: Path) -> bool:
    try:
        base_resolved = base.resolve()
        target_resolved = target.resolve()
        return str(target_resolved).startswith(str(base_resolved))
    except Exception:
        return False


def safe_extract(tar: tarfile.TarFile, dest_dir: Path) -> None:
    members = tar.getmembers()
    for m in tqdm(members, desc=f"Extracting to {dest_dir.name}"):
        # Skip directories
        if m.isdir():
            continue
            
        # Flatten the path - extract only the filename, ignore directory structure
        original_name = m.name
        if '/' in original_name:
            # Get just the filename part
            filename = Path(original_name).name
        else:
            filename = original_name
            
        # Skip if it's not a file we want (e.g., hidden files, non-image files)
        if not filename or filename.startswith('.'):
            continue
            
        # Create a new member with flattened path
        flattened_member = tarfile.TarInfo(name=filename)
        flattened_member.size = m.size
        flattened_member.mtime = m.mtime
        flattened_member.mode = m.mode
        flattened_member.type = m.type
        flattened_member.uid = m.uid
        flattened_member.gid = m.gid
        flattened_member.uname = m.uname
        flattened_member.gname = m.gname
        
        member_path = dest_dir / filename
        if not is_within_dir(dest_dir, member_path):
            print(f"[WARN] Skipping unsafe path: {filename}")
            continue
            
        try:
            # Extract file data and write to flattened path
            if m.isfile():
                with tar.extractfile(m) as source:
                    if source:
                        with open(member_path, 'wb') as target:
                            target.write(source.read())
                        # Set file permissions and timestamps
                        member_path.chmod(m.mode)
                        import os
                        os.utime(member_path, (m.mtime, m.mtime))
        except Exception as e:
            print(f"[WARN] Failed to extract {original_name} as {filename}: {e}")


def find_archives(src: Path) -> List[Path]:
    results: List[Path] = []
    results.extend(sorted(src.glob('*.tar.gz')))
    results.extend(sorted(src.glob('*.tgz')))
    results.extend(sorted(src.glob('*.tar')))
    return results


def has_system_tar() -> bool:
    return shutil.which("tar") is not None


def extract_with_system_tar(archive: Path, out_dir: Path) -> bool:
    """Use system tar (bsdtar) to extract. Returns True on success."""
    cmd = ["tar", "-xzf", str(archive), "-C", str(out_dir)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] system tar failed for {archive}: {e.stderr or e.stdout}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract .tar.gz/.tgz/.tar archives into same-named folders")
    ap.add_argument('--src', type=str, required=True, help='Folder containing archives (e.g., images directory)')
    ap.add_argument('--force', action='store_true', help='Re-extract even if output folder exists (files may be overwritten)')
    ap.add_argument('--dry-run', action='store_true', help='List what would be extracted without doing it')
    ap.add_argument('--use-system-tar', action='store_true', help='Prefer system tar for extraction (useful if Python tarfile errors)')
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    archives = find_archives(src)
    if not archives:
        print(f"No .tar.gz/.tgz/.tar archives found under {src}")
        return

    for arch in archives:
        out_dir = infer_output_dir(arch)
        if out_dir.exists() and not args.force:
            print(f"[SKIP] {arch.name} → {out_dir} (exists, use --force to re-extract)")
            continue
        print(f"[INFO] {arch.name} → {out_dir}")
        if args.dry_run:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        # Try extraction
        used_system_tar = False
        if args.use_system_tar and has_system_tar():
            used_system_tar = True
            if extract_with_system_tar(arch, out_dir):
                continue
            else:
                print("[WARN] system tar failed, will try Python tarfile as fallback…")

        try:
            # First attempt: normal mode
            with tarfile.open(arch, mode='r:*') as tar:
                safe_extract(tar, out_dir)
        except Exception as e1:
            print(f"[WARN] Python tarfile error: {e1}")
            # Second attempt: streaming mode (sometimes tolerates odd archives)
            try:
                with tarfile.open(arch, mode='r|*') as tar:
                    safe_extract(tar, out_dir)
            except Exception as e2:
                print(f"[WARN] Python tarfile streaming error: {e2}")
                if has_system_tar() and not used_system_tar:
                    print("[INFO] Trying system tar as last resort…")
                    if extract_with_system_tar(arch, out_dir):
                        continue
                print(f"[ERROR] Failed extracting {arch}: {e2}")


if __name__ == '__main__':
    main()
