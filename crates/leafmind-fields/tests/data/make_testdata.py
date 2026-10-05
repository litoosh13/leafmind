"""Makes the test page and the reference result for leafmind-fields.

The page is an invented, generic library-card application (English, no real organisation). The reference
is the model card's recipe: Pillow BILINEAR letterbox to 640x640 (pad 114) + ONNX Runtime + score =
objectness x class, threshold 0.3, NMS 0.6 per class. Needs reportlab, pypdfium2, pillow, numpy,
onnxruntime. Usage: python make_testdata.py
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent
MODEL = HERE / "../../../../models/form-field-v1-nano.onnx"
W, H = A4


def make_pdf(path):
    c = canvas.Canvas(str(path), pagesize=A4)
    t = lambda x, y, s, size=10, bold=False: (c.setFont("Helvetica-Bold" if bold else "Helvetica", size), c.drawString(x, H - y, s))
    t(56, 70, "Library card application", 16, True)
    t(56, 92, "Please fill in all fields in block letters.", 9)
    rows = ["First name", "Last name", "Date of birth", "Street and number", "Postcode and town", "Email", "Phone"]
    c.setLineWidth(0.6)
    y = 120
    for label in rows:  # label cell + empty answer cell
        c.rect(56, H - y - 24, 150, 24)
        c.rect(206, H - y - 24, 333, 24)
        t(62, y + 16, label, 9)
        y += 24
    t(56, y + 30, "Card type", 10, True)
    for i, s in enumerate(["Adult", "Child (under 16)", "Student"]):
        x = 56 + i * 150
        c.rect(x, H - y - 50, 9, 9)
        t(x + 15, y + 49, s, 9)
    t(56, y + 80, "I would like to receive the newsletter:", 9)
    for i, s in enumerate(["yes", "no"]):
        x = 240 + i * 60
        c.rect(x, H - y - 81, 9, 9)
        t(x + 15, y + 80, s, 9)
    t(56, y + 110, "Anything else we should know?", 9)
    for k in range(2):
        c.line(56, H - y - 132 - 18 * k, 539, H - y - 132 - 18 * k)
    c.line(56, H - y - 210, 240, H - y - 210)
    t(56, y + 222, "Place, date", 8)
    c.line(310, H - y - 210, 539, H - y - 210)
    t(310, y + 222, "Signature", 8)
    c.save()


def reference(pil, session):
    S = 640
    r = min(S / pil.width, S / pil.height)
    board = Image.new("RGB", (S, S), (114, 114, 114))
    board.paste(pil.resize((int(pil.width * r), int(pil.height * r)), Image.BILINEAR), (0, 0))
    out = session.run(None, {"images": np.asarray(board, np.float32).transpose(2, 0, 1)[None]})[0][0]
    score = out[:, 4:5] * out[:, 5:8]
    cls, best = score.argmax(1), score.max(1)
    keep = best >= 0.3
    out, cls, best = out[keep], cls[keep], best[keep]
    boxes = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 0] + out[:, 2] / 2, out[:, 1] + out[:, 3] / 2], 1) / r
    fields = []
    for k, name in enumerate(["text", "choice", "signature"]):
        idx = list(np.where(cls == k)[0][np.argsort(-best[cls == k])])
        kept = []
        while idx:
            i = idx.pop(0)
            kept.append(i)
            def iou(a, b):
                w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h + 1e-9)
            idx = [j for j in idx if iou(boxes[i], boxes[j]) <= 0.6]
        fields += [{"kind": name, "score": round(float(best[i]), 4), "box": [round(float(v), 2) for v in boxes[i]]} for i in kept]
    return board, fields


if __name__ == "__main__":
    pdf = HERE / "library-card.pdf"
    make_pdf(pdf)
    pil = pdfium.PdfDocument(str(pdf))[0].render(scale=1.5).to_pil().convert("RGB")
    pdf.unlink()
    pil.save(HERE / "library-card.png", optimize=True)
    board, fields = reference(pil, ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"]))
    board.save(HERE / "library-card.model-input.png", optimize=True)
    json.dump({"library-card": {"width": pil.width, "height": pil.height, "fields": fields}}, open(HERE / "expected.json", "w"), indent=1)
    print(len(fields), "fields:", {k: sum(f["kind"] == k for f in fields) for k in ("text", "choice", "signature")})
