import json
import shutil
from pathlib import Path
from datetime import datetime
from moderation_pipeline_v5 import moderate_images  # your big file


def classify_and_copy(
    input_dir,
    output_dir,
    num_images: int = -1,
    move: bool = False,
    keep_structure: bool = False,
):
    """
    Recursively run moderation on images found under input_dir (including all subfolders),
    copy/move them into SAFE/ and UNSAFE/ subfolders, and save a JSONL summary report.

    Args:
        input_dir (str | Path): Root directory containing images and subfolders.
        output_dir (str | Path): Where SAFE/ and UNSAFE/ will be created.
        num_images (int): Global cap across ALL subfolders; -1 means "no limit".
        move (bool): If True, move files; else copy.
        keep_structure (bool): If True, mirror the input subfolder structure inside SAFE/ and UNSAFE/.
                               If False (default), flatten into SAFE/ and UNSAFE/ with collision-safe renames.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    safe_dir = output_dir / "SAFE"
    unsafe_dir = output_dir / "Incorrect"
    safe_dir.mkdir(parents=True, exist_ok=True)
    unsafe_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"moderation_report_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    moved_or_copied = "moved" if move else "copied"

    # Collect all directories to scan: the root + all subdirectories.
    all_dirs = [input_dir] + [d for d in input_dir.rglob("*") if d.is_dir()]

    total = safe_count = unsafe_count = 0
    remaining = None if num_images is None or num_images < 0 else int(num_images)

    with report_path.open("w", encoding="utf-8") as f:
        for d in all_dirs:
            # Respect the global cap
            if remaining is not None and remaining <= 0:
                break

            # Pass a per-call cap to your pipeline if a global cap is still in effect
            # (If remaining is None, pass -1 to indicate "no limit" per your pipeline's API.)
            per_dir_cap = -1 if remaining is None else remaining

            # Run moderation for this specific folder.
            try:
                results = moderate_images(str(d), num_images=per_dir_cap)
            except Exception as e:
                # Log folder-level errors and continue
                f.write(json.dumps({
                    "error": "moderation_exception",
                    "folder": str(d),
                    "message": repr(e),
                }, ensure_ascii=False) + "\n")
                continue

            # If your pipeline returns None/empty for non-image dirs, normalize to list
            if not results:
                continue

            for item in results:
                if remaining is not None and remaining <= 0:
                    break

                src = Path(item["image_path"])
                label = item.get("overall", "SAFE")
                dest_root = safe_dir if label == "SAFE" else unsafe_dir

                if not src.exists():
                    f.write(json.dumps({"error": "missing_source", **item}, ensure_ascii=False) + "\n")
                    continue

                # Determine destination path (either mirrored structure or flat)
                if keep_structure:
                    # Mirror path relative to input_dir
                    rel_parent = src.parent.relative_to(input_dir)
                    dest_dir = dest_root / rel_parent
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / src.name
                else:
                    dest_dir = dest_root
                    dest = dest_dir / src.name

                # Ensure we don't overwrite existing files — add suffix if needed
                if dest.exists():
                    stem, suffix = dest.stem, dest.suffix
                    i = 1
                    while True:
                        candidate = dest_dir / f"{stem}__{i}{suffix}"
                        if not candidate.exists():
                            dest = candidate
                            break
                        i += 1

                # Do the move/copy
                if move:
                    shutil.move(str(src), str(dest))
                else:
                    shutil.copy2(str(src), str(dest))

                total += 1
                safe_count += (label == "SAFE")
                unsafe_count += (label == "UNSAFE")

                # Write one line per image to the JSONL report
                f.write(json.dumps({
                    "source": str(src),
                    "destination": str(dest),
                    "overall": label,
                    "regions": item.get("regions", [])
                }, ensure_ascii=False) + "\n")

                # Decrement the global remaining count
                if remaining is not None:
                    remaining -= 1

    print("=" * 60)
    print(f"{moved_or_copied.capitalize()} {total} image(s).")
    print(f"SAFE:   {safe_count} -> {safe_dir}")
    print(f"UNSAFE: {unsafe_count} -> {unsafe_dir}")
    print(f"Report: {report_path}")
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    classify_and_copy(
        input_dir="test_run_folder_filter_data",         # root that contains nsfw/safe/test_data, etc.
        output_dir="./moderated_output",
        num_images=-1,                   # global cap across all subfolders (-1 = no cap)
        move=False,
        keep_structure=True              # set False to flatten into SAFE/ and UNSAFE/
    )
