from __future__ import annotations

# ---- path bootstrap: allow `from src...` when running as `python scripts\x.py`
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ---- end bootstrap

import argparse
from pathlib import Path
import zipfile
import subprocess
import shutil

def extract_media(docx_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted=[]
    with zipfile.ZipFile(docx_path) as z:
        media=[m for m in z.namelist() if m.startswith("word/media/")]
        for m in media:
            data=z.read(m)
            fn = out_dir / Path(m).name
            fn.write_bytes(data)
            extracted.append(fn)
    return extracted

def try_convert_with_inkscape(in_path: Path, out_png: Path, dpi: int) -> bool:
    # Inkscape CLI can convert EMF/SVG/PDF to PNG.
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        return False
    cmd = [inkscape, str(in_path), f"--export-type=png", f"--export-filename={out_png}", f"--export-dpi={dpi}"]
    try:
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=1000)
    ap.add_argument("--convert", action="store_true", help="Try to convert vector figures (EMF/SVG/PDF) to 1000-DPI PNG using Inkscape.")
    args = ap.parse_args()

    docx = Path(args.docx)
    out_dir = Path(args.out_dir)
    media_dir = out_dir / "media_raw"
    png_dir = out_dir / f"media_png_{args.dpi}dpi"
    png_dir.mkdir(parents=True, exist_ok=True)

    files = extract_media(docx, media_dir)
    print(f"Extracted {len(files)} files to {media_dir}")

    if args.convert:
        converted=0
        for f in files:
            if f.suffix.lower() in [".emf", ".svg", ".pdf"]:
                out_png = png_dir / (f.stem + ".png")
                ok = try_convert_with_inkscape(f, out_png, dpi=args.dpi)
                if ok:
                    converted += 1
        print(f"Converted {converted} vector files to PNG under {png_dir}")
        if converted == 0:
            print("No conversions succeeded. Install Inkscape and ensure it is on PATH.")

if __name__ == "__main__":
    main()
