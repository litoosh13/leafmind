"""Draws bilingual-notice.png: an invented English/Persian notice for the Indra model test (tests/indra.rs).
Each row: (text, right-aligned, y). Needs Pillow with libraqm (Arabic-script shaping) and a font with Persian
letters (macOS: Arial Unicode). Run: python3 make_testdata.py"""
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
ROWS = [
    ("Equipment Loan Notice", False, 60),
    ("اطلاعیه امانت تجهیزات", True, 130),
    ("Reference: LN-4471", False, 220),
    ("شماره مرجع: ۴۴۷۱", True, 290),
    ("The workshop is closed on public holidays.", False, 380),
    ("کارگاه در روزهای تعطیل بسته است.", True, 450),
]
img = Image.new("RGB", (1240, 560), "white")
d = ImageDraw.Draw(img)
font = ImageFont.truetype(FONT, 34)
for text, right, y in ROWS:
    if right:
        w = d.textlength(text, font=font, direction="rtl", language="fa")
        d.text((1240 - 80 - w, y), text, fill="black", font=font, direction="rtl", language="fa")
    else:
        d.text((80, y), text, fill="black", font=font)
img.save("bilingual-notice.png", optimize=True)
