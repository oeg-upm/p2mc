# Test fixtures

Keep fixtures small and deterministic:

- fake PDFs only need enough bytes for validation paths;
- XML fixtures should contain the minimum TEI needed by `XMLParser`;
- LightOCR JSON fixtures should contain the minimum table structure needed by `PDFHandler` and `ModelCardGenerator`.
