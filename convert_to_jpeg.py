#!/usr/bin/env python3
"""
Convert all PNG mnemonic images to JPEG to reduce repo size for deployment.

Run once from the project folder:
    python convert_to_jpeg.py

After verifying the JPEGs look good, you can delete the PNGs:
    del output_images\*.png   (Windows)
"""

from pathlib import Path
from PIL import Image

OUTPUT_DIR = Path("output_images")
QUALITY    = 85   # 85 is visually indistinguishable from PNG for illustrations

pngs = sorted(OUTPUT_DIR.glob("*.png"))
if not pngs:
    print("No PNG files found in output_images/")
    raise SystemExit

total_before = sum(p.stat().st_size for p in pngs)
print(f"Converting {len(pngs)} PNGs  (total {total_before / 1_048_576:.1f} MB)  →  JPEG quality={QUALITY}\n")

total_after = 0
for i, png_path in enumerate(pngs, 1):
    jpg_path = png_path.with_suffix(".jpg")
    img = Image.open(png_path).convert("RGB")   # drop alpha channel if any
    img.save(jpg_path, format="JPEG", quality=QUALITY, optimize=True)
    after = jpg_path.stat().st_size
    total_after += after
    print(f"[{i:>3}/{len(pngs)}]  {png_path.stem:<20}  "
          f"{png_path.stat().st_size // 1024:>5} KB  →  {after // 1024:>4} KB")

print(f"\nDone.")
print(f"  Before : {total_before / 1_048_576:.1f} MB")
print(f"  After  : {total_after  / 1_048_576:.1f} MB")
print(f"  Saving : {(total_before - total_after) / 1_048_576:.1f} MB  "
      f"({100 * (1 - total_after / total_before):.0f}% reduction)")
print(f"\nVerify the JPEGs look good, then delete the PNGs:")
print(f"  Windows PowerShell:  Remove-Item output_images\\*.png")
