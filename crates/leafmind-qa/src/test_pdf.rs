//! Builds small invented PDFs for tests, so no binary test files are needed. Text uses the standard
//! Helvetica font with WinAnsi encoding: Latin letters incl. German umlauts, no Arabic script.

/// One page per slice. A line starting with "# " is a heading (larger font); an empty line starts a new
/// paragraph (a wider gap); other lines follow each other in the same paragraph. A tab puts the rest of the
/// line in a second column on the same row (x = 300), as in a form grid.
pub(crate) fn pdf(pages: &[&[&str]]) -> Vec<u8> {
    let n = pages.len();
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 4 + 2 * i)).collect();
    let mut objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        format!("<< /Type /Pages /Count {n} /Kids [{}] >>", kids.join(" ")).into_bytes(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    ];
    for (i, lines) in pages.iter().enumerate() {
        objects.push(
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents {} 0 R >>",
                5 + 2 * i
            )
            .into_bytes(),
        );
        let mut content = Vec::new();
        let mut y = 780;
        for line in lines.iter() {
            if line.is_empty() {
                y -= 14;
                continue;
            }
            let (size, text) = match line.strip_prefix("# ") {
                Some(heading) => (16, heading),
                None => (11, *line),
            };
            for (column, text) in text.split('\t').enumerate() {
                content.extend(format!("BT /F1 {size} Tf {} {y} Td (", 60 + 240 * column).bytes());
                for c in text.chars() {
                    let byte =
                        u8::try_from(u32::from(c)).expect("test PDFs hold Latin-1 text only");
                    if matches!(byte, b'(' | b')' | b'\\') {
                        content.push(b'\\');
                    }
                    content.push(byte);
                }
                content.extend(b") Tj ET\n");
            }
            y -= if size == 16 { 28 } else { 15 };
        }
        let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
        stream.extend(content);
        stream.extend(b"\nendstream");
        objects.push(stream);
    }
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend(format!("{} 0 obj\n", i + 1).bytes());
        out.extend(body);
        out.extend(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
    for o in offsets {
        out.extend(format!("{o:010} 00000 n \n").bytes());
    }
    out.extend(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .bytes(),
    );
    out
}
