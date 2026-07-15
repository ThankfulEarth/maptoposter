from pathlib import Path
from scripts.generate_title_fonts import decompress_woff2
from fontTools.ttLib.woff2 import compress
from fontTools.ttLib import TTFont

def test_decompress_roundtrip(tmp_path: Path):
    # Build a tiny ttf from an existing bundled Roboto, compress to woff2, then decompress.
    src_ttf = Path(__file__).parent.parent / "fonts" / "Roboto-Regular.ttf"
    woff2 = tmp_path / "in.woff2"
    out_ttf = tmp_path / "out.ttf"
    compress(str(src_ttf), str(woff2))
    decompress_woff2(str(woff2), str(out_ttf))
    assert out_ttf.exists()
    # Result is a valid sfnt matplotlib can open.
    TTFont(str(out_ttf))
