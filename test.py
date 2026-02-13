import os
from pathlib import Path
from PIL import Image

def convert_webp_to_png(directory, recursive=False, delete_original=True):
    """
    Convert all .webp files in a directory to .png format.
    Optionally recurse into subfolders and delete the originals.
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return

    # Find .webp files
    files = directory.rglob("*.webp") if recursive else directory.glob("*.webp")

    count = 0
    for webp_path in files:
        try:
            png_path = webp_path.with_suffix(".png")
            with Image.open(webp_path) as img:
                img.save(png_path, "PNG")
            print(f"✅ Converted: {webp_path.name} → {png_path.name}")

            if delete_original:
                webp_path.unlink()
                print(f"🗑️  Deleted: {webp_path.name}")

            count += 1
        except Exception as e:
            print(f"⚠️  Failed to process {webp_path}: {e}")

    print(f"\n✨ Done! Converted {count} .webp file(s).")

# -----------------------------
# EDIT THESE SETTINGS BELOW
# -----------------------------
directory = "safe-not-finished"  # 👈 Change this
recursive = True          # Search subdirectories too
delete_original = True    # Delete the .webp after conversion
# -----------------------------

convert_webp_to_png(directory, recursive, delete_original)
