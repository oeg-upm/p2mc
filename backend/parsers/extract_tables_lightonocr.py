"""Extract structured tables from PDFs using LightOnOCR-2-1B.

Single folder of PDFs:
    python extract_tables_lightonocr.py -i pdfs_test -v

Custom output path:
    python extract_tables_lightonocr.py -i data/pdf_files_3 \\
        -o data/pdf_files_3/extracted_tables_lightonocr.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pypdfium2 as pdfium
import torch
from bs4 import BeautifulSoup
from PIL import Image

DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B"
DEFAULT_TARGET_LONGEST = 1540
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_OUTPUT_NAME = "extracted_tables_lightonocr.json"

log = logging.getLogger(__name__)


class LightOnOcrTableExtractor:
    """Extract and parse tables from PDFs; emit structured JSON."""

    _GREEK_MAP = {
        r"\\alpha": "α",
        r"\\beta": "β",
        r"\\gamma": "γ",
        r"\\delta": "δ",
        r"\\epsilon": "ε",
        r"\\zeta": "ζ",
        r"\\eta": "η",
        r"\\theta": "θ",
        r"\\iota": "ι",
        r"\\kappa": "κ",
        r"\\lambda": "λ",
        r"\\mu": "μ",
        r"\\nu": "ν",
        r"\\xi": "ξ",
        r"\\pi": "π",
        r"\\rho": "ρ",
        r"\\sigma": "σ",
        r"\\tau": "τ",
        r"\\phi": "φ",
        r"\\chi": "χ",
        r"\\psi": "ψ",
        r"\\omega": "ω",
        r"\\Gamma": "Γ",
        r"\\Delta": "Δ",
        r"\\Lambda": "Λ",
        r"\\Sigma": "Σ",
        r"\\Phi": "Φ",
        r"\\Omega": "Ω",
    }

    def __init__(
        self,
        pdf_dir: Path,
        output_path: Optional[Path] = None,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        target_longest: int = DEFAULT_TARGET_LONGEST,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        verbose: bool = False,
        progress_logger: Callable[[str], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.pdf_dir = Path(pdf_dir).resolve()
        self.output_path = (
            Path(output_path).resolve()
            if output_path is not None
            else self.pdf_dir / DEFAULT_OUTPUT_NAME
        )
        self.model_id = model_id
        self.target_longest = target_longest
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self._progress_logger = progress_logger
        self._progress_callback = progress_callback

        self._processor = None
        self._model = None
        self._device: str = "cpu"
        self._dtype = torch.float32

    def _progress(self, message: str) -> None:
        if self._progress_logger is not None:
            self._progress_logger(message)
        else:
            print(message, flush=True)

    def _emit_progress(self, stage: dict[str, Any]) -> None:
        if self._progress_callback is not None:
            self._progress_callback(stage)

    @staticmethod
    def _lightocr_stage(
        detail: str,
        page_num: int,
        num_pages: int,
    ) -> dict[str, Any]:
        return {
            "key": "extracting_tables",
            "label": "Extracting tables with LightOCR",
            "step": 4,
            "total": 7,
            "detail": detail,
            "item_current": page_num,
            "item_total": num_pages,
        }

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_pdf_dir(path: Optional[Path] = None) -> Path:
        """Resolve a PDF folder from an explicit path or common defaults."""
        if path is not None:
            p = Path(path).resolve()
            if p.is_dir() and any(p.glob("*.pdf")):
                return p
            raise FileNotFoundError(f"No PDFs found in {p}")

        candidates = [
            Path("pdfs_prueba"),
            Path("pdfs_test"),
            Path("table_extraction/pdfs_prueba"),
            Path("table_extraction/pdfs_test"),
        ]
        for c in candidates:
            if c.is_dir() and any(c.glob("*.pdf")):
                return c.resolve()
        raise FileNotFoundError(
            "No PDF folder found. Use -i/--input or place PDFs in pdfs_test/ or pdfs_prueba/."
        )

    def list_pdfs(self) -> List[Path]:
        return sorted(self.pdf_dir.glob("*.pdf"))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
        except ImportError as e:
            raise ImportError(
                "LightOnOcrForConditionalGeneration not found. "
                "Install transformers>=4.57.6 and restart."
            ) from e

        if torch.cuda.is_available():
            self._device = "cuda"
            self._dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float32
        else:
            self._device = "cpu"
            self._dtype = torch.float32

        self._progress(
            f"LightOCR: using device={self._device}, dtype={self._dtype}"
        )
        self._processor = LightOnOcrProcessor.from_pretrained(self.model_id)
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(self._device)
        self._model.eval()
        self._progress(f"LightOCR: model loaded: {self.model_id}")

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def render_pdf_page(self, pdf_doc, page_idx: int) -> Image.Image:
        page = pdf_doc[page_idx]
        pil_image = page.render(scale=200 / 72).to_pil()
        w, h = pil_image.size
        longest = max(w, h)
        if longest > self.target_longest:
            ratio = self.target_longest / longest
            pil_image = pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return pil_image

    def ocr_page(self, pil_image: Image.Image, max_new_tokens: Optional[int] = None) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load_models() before ocr_page().")

        tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_image.save(tmp, format="PNG")
        tmp.close()
        try:
            conversation = [{"role": "user", "content": [{"type": "image", "url": tmp.name}]}]
            inputs = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(device=self._device, dtype=self._dtype)
                if v.is_floating_point()
                else v.to(self._device)
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, max_new_tokens=tokens)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            return self._processor.decode(
                generated_ids,
                skip_special_tokens=True,
            )
        finally:
            os.unlink(tmp.name)
            try:
                del inputs
            except UnboundLocalError:
                pass
            try:
                del output_ids
            except UnboundLocalError:
                pass
            try:
                del generated_ids
            except UnboundLocalError:
                pass
            gc.collect()
            if self._device == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Markdown table parsing
    # ------------------------------------------------------------------

    @staticmethod
    def coerce_value(s: str) -> Any:
        s = s.strip()
        if not s or s in ("–", "—", "N/A", "n/a", "nan", ""):
            return "-"
        if s == "-":
            return "-"

        if re.search(r",\d{3}(\D|$)", s):
            s_clean = s.replace(",", "")
        else:
            s_clean = s.replace(",", ".")

        try:
            val = float(s_clean)
            return int(val) if val == int(val) else val
        except ValueError:
            return s

    @staticmethod
    def clean_cell_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"\$([^$]+?)\$", r"\1", text)
        text = re.sub(r"\\textbf\{(.+?)\}", r"\1", text)
        text = re.sub(r"\\textit\{(.+?)\}", r"\1", text)
        text = re.sub(r"\\text\{(.+?)\}", r"\1", text)
        text = re.sub(r"\\mathbf\{(.+?)\}", r"\1", text)
        return text.strip()

    @staticmethod
    def is_separator_line(line: str) -> bool:
        cells = [c.strip() for c in line.strip().split("|")]
        cells = [c for c in cells if c]
        if not cells:
            return False
        return all(re.match(r"^:?-{1,}:?$", c) for c in cells)

    def extract_markdown_tables(self, text: str) -> List[List[str]]:
        lines = text.split("\n")
        tables: List[List[str]] = []
        current_block: List[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                current_block.append(stripped)
                in_table = True
            else:
                if in_table and current_block:
                    has_sep = any(self.is_separator_line(l) for l in current_block)
                    if has_sep and len(current_block) >= 3:
                        tables.append(current_block[:])
                    current_block = []
                    in_table = False

        if current_block:
            has_sep = any(self.is_separator_line(l) for l in current_block)
            if has_sep and len(current_block) >= 3:
                tables.append(current_block[:])

        return tables

    def parse_row_cells(self, line: str) -> List[str]:
        cells = line.split("|")
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        return [self.clean_cell_text(c) for c in cells]

    def parse_markdown_table(self, table_lines: List[str]) -> Tuple[Optional[List[str]], Optional[List[List[Any]]]]:
        sep_idx = None
        for i, line in enumerate(table_lines):
            if self.is_separator_line(line):
                sep_idx = i
                break

        if sep_idx is None or sep_idx == 0:
            return None, None

        header_lines = table_lines[:sep_idx]
        data_lines = table_lines[sep_idx + 1 :]
        if not data_lines:
            return None, None

        header_rows = [self.parse_row_cells(line) for line in header_lines]
        all_rows_parsed = header_rows + [self.parse_row_cells(data_lines[0])]
        num_cols = max(len(r) for r in all_rows_parsed)

        if len(header_rows) == 1:
            columns = header_rows[0]
        else:
            columns = []
            for col_idx in range(num_cols):
                parts: List[str] = []
                prev = None
                for row in header_rows:
                    val = row[col_idx].strip() if col_idx < len(row) else ""
                    if val and val != prev:
                        parts.append(val)
                    prev = val
                columns.append("_".join(parts) if parts else "")

        columns = [c.replace(" ", "_") for c in columns]
        columns = [re.sub(r"@_(\d)", r"@\1", c) for c in columns]
        while len(columns) < num_cols:
            columns.append(f"col_{len(columns)}")

        data_rows: List[List[Any]] = []
        for line in data_lines:
            if not line.strip():
                continue
            cells = self.parse_row_cells(line)
            while len(cells) < num_cols:
                cells.append("")
            row = [self.coerce_value(c) for c in cells[:num_cols]]
            if not all(v in ("-", "") for v in row):
                data_rows.append(row)

        return columns, data_rows

    # ------------------------------------------------------------------
    # HTML table parsing
    # ------------------------------------------------------------------

    def strip_inline_formatting(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\$([^$]+?)\$", r"\1", text)
        for cmd in ("textbf", "textit", "text", "mathbf", "mathrm", "mathit"):
            text = re.sub(r"\\" + cmd + r"\{(.+?)\}", r"\1", text)
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
        text = re.sub(r"\\(left|right)", "", text)
        text = text.replace("\\times", "x")
        text = text.replace("\\cdot", "·")
        text = text.replace("\\pm", "±")
        text = text.replace("\\leq", "≤").replace("\\geq", "≥")
        for latex, uni in self._GREEK_MAP.items():
            text = re.sub(latex + r"(?![a-zA-Z])", uni, text)
        text = re.sub(r"\\([#&$%_])", r"\1", text)
        for _ in range(3):
            text = re.sub(r"[_^]\{([^{}]*)\}", r"\1", text)
        text = text.replace("{", "").replace("}", "")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text.strip()

    @staticmethod
    def extract_html_tables(text: str) -> List[str]:
        return re.findall(r"<table\b[^>]*>.*?</table>", text, flags=re.DOTALL | re.IGNORECASE)

    def expand_row(self, cells, col_cursor_matrix: dict, row_idx: int) -> None:
        col_idx = 0
        for cell in cells:
            while col_cursor_matrix.get(row_idx, {}).get(col_idx) is not None:
                col_idx += 1

            text = self.strip_inline_formatting(cell.get_text(separator=" "))
            try:
                rowspan = int(cell.get("rowspan", 1))
            except (TypeError, ValueError):
                rowspan = 1
            try:
                colspan = int(cell.get("colspan", 1))
            except (TypeError, ValueError):
                colspan = 1

            for dr in range(rowspan):
                for dc in range(colspan):
                    col_cursor_matrix.setdefault(row_idx + dr, {})[col_idx + dc] = text
            col_idx += colspan

    @staticmethod
    def matrix_to_rows(matrix: dict) -> List[List[str]]:
        if not matrix:
            return []
        n_rows = max(matrix.keys()) + 1
        n_cols = max((max(cols.keys()) + 1) for cols in matrix.values() if cols)
        return [[matrix.get(r, {}).get(c, "") for c in range(n_cols)] for r in range(n_rows)]

    def parse_html_table(self, html_str: str) -> Tuple[Optional[List[str]], Optional[List[List[Any]]]]:
        soup = BeautifulSoup(html_str, "html.parser")
        table = soup.find("table")
        if table is None:
            return None, None

        header_matrix: dict = {}
        body_matrix: dict = {}

        thead = table.find("thead")
        tbody = table.find("tbody")

        if thead is not None:
            for r_idx, tr in enumerate(thead.find_all("tr")):
                cells = tr.find_all(["th", "td"])
                self.expand_row(cells, header_matrix, r_idx)

        if tbody is not None:
            body_trs = tbody.find_all("tr")
        else:
            all_trs = table.find_all("tr")
            body_trs = [tr for tr in all_trs if (thead is None or tr not in thead.find_all("tr"))]

        for r_idx, tr in enumerate(body_trs):
            cells = tr.find_all(["th", "td"])
            self.expand_row(cells, body_matrix, r_idx)

        header_rows = self.matrix_to_rows(header_matrix)
        body_rows = self.matrix_to_rows(body_matrix)

        if not header_rows and body_rows:
            header_rows = [body_rows[0]]
            body_rows = body_rows[1:]

        if not header_rows or not body_rows:
            return None, None

        num_cols = max(
            max((len(r) for r in header_rows), default=0),
            max((len(r) for r in body_rows), default=0),
        )
        header_rows = [r + [""] * (num_cols - len(r)) for r in header_rows]
        body_rows = [r + [""] * (num_cols - len(r)) for r in body_rows]

        if len(header_rows) == 1:
            columns = [self.strip_inline_formatting(c) for c in header_rows[0]]
        else:
            columns = []
            for c in range(num_cols):
                parts: List[str] = []
                prev = None
                for r in header_rows:
                    val = self.strip_inline_formatting(r[c])
                    if val and val != prev:
                        parts.append(val)
                    prev = val
                columns.append("_".join(parts) if parts else "")

        columns = [re.sub(r"@_(\d)", r"@\1", c) for c in columns]
        columns = [c if c else "" for c in columns]

        data_rows: List[List[Any]] = []
        for row in body_rows:
            coerced = [self.coerce_value(v) for v in row]
            if not all(v in ("-", "") for v in coerced):
                data_rows.append(coerced)

        if not data_rows:
            return None, None

        return columns, data_rows

    # ------------------------------------------------------------------
    # Extraction pipeline
    # ------------------------------------------------------------------

    def extract_tables_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        paper_title = pdf_path.stem
        log.info("Processing: %s", paper_title)

        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        num_pages = len(pdf_doc)
        self._progress(
            f"LightOCR: processing {paper_title} with {num_pages} pages"
        )
        all_tables: List[Tuple[int, List[str], List[List[Any]], str]] = []

        try:
            for page_idx in range(num_pages):
                page_num = page_idx + 1
                self._emit_progress(
                    self._lightocr_stage(
                        f"Processing page {page_num}/{num_pages}",
                        page_num,
                        num_pages,
                    )
                )
                self._progress(
                    f"LightOCR: page {page_num}/{num_pages} render started"
                )

                pil_image = self.render_pdf_page(pdf_doc, page_idx)
                try:
                    self._progress(
                        f"LightOCR: page {page_num}/{num_pages} OCR started"
                    )
                    ocr_text = self.ocr_page(pil_image)
                finally:
                    pil_image.close()
                self._progress(
                    f"LightOCR: page {page_num}/{num_pages} OCR finished"
                )

                html_tables = self.extract_html_tables(ocr_text)
                md_tables = self.extract_markdown_tables(ocr_text) if not html_tables else []
                self._progress(
                    "LightOCR: "
                    f"page {page_num}/{num_pages} detected "
                    f"{len(html_tables)} HTML tables and {len(md_tables)} "
                    "Markdown tables"
                )

                for html_tbl in html_tables:
                    columns, data_rows = self.parse_html_table(html_tbl)
                    if columns and data_rows:
                        all_tables.append((page_num, columns, data_rows, html_tbl))
                        log.info(
                            "    HTML table → %d rows × %d cols",
                            len(data_rows),
                            len(columns),
                        )

                for md_tbl in md_tables:
                    columns, data_rows = self.parse_markdown_table(md_tbl)
                    if columns and data_rows:
                        all_tables.append((page_num, columns, data_rows, "\n".join(md_tbl)))
                        log.info(
                            "    MD table → %d rows × %d cols",
                            len(data_rows),
                            len(columns),
                        )
                self._progress(
                    f"LightOCR: page {page_num}/{num_pages} finished"
                )
        finally:
            pdf_doc.close()

        tables_json: List[Dict[str, Any]] = []
        for i, (page_num, columns, data_rows, _) in enumerate(all_tables):
            table_id = f"table_{i + 1}"
            tables_json.append(
                {
                    "table_id": table_id,
                    "page": page_num,
                    "evaluation": {
                        "expected_rows": len(data_rows),
                        "expected_cols": len(columns),
                        "columns": columns,
                    },
                    "rows": data_rows,
                }
            )

        document = {
            "paper_title": paper_title,
            "num_tables": len(tables_json),
            "tables": tables_json,
        }
        self._progress(
            f"LightOCR: finished {paper_title}; total tables={len(tables_json)}"
        )
        return document

    def extract(
        self,
        pdf_paths: Optional[Sequence[Path]] = None,
    ) -> Dict[str, Any]:
        """Process all PDFs and return the combined extraction dict."""
        self.load_models()

        paths = sorted(pdf_paths) if pdf_paths is not None else self.list_pdfs()
        if not paths:
            raise FileNotFoundError(f"No PDFs found in {self.pdf_dir}")

        documents: List[Dict[str, Any]] = []
        for pdf_path in paths:
            documents.append(self.extract_tables_from_pdf(Path(pdf_path)))

        result = {"documents": documents}
        total_tables = sum(d["num_tables"] for d in documents)
        log.info(
            "Done: %d documents, %d tables total",
            len(documents),
            total_tables,
        )
        return result

    def save_json(
        self,
        result: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Path:
        out = Path(output_path).resolve() if output_path is not None else self.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info("Extraction saved to: %s", out)
        return out

    def run(self, pdf_paths: Optional[Sequence[Path]] = None) -> Path:
        """Extract tables from PDFs and write JSON to disk."""
        result = self.extract(pdf_paths=pdf_paths)
        return self.save_json(result)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_tables_lightonocr",
        description="Extract structured tables from PDFs with LightOnOCR.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Folder with PDFs (default: auto-detect pdfs_test/ or pdfs_prueba/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON path (default: <input>/{DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"LightOnOCR model id (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    warnings.filterwarnings("ignore")
    args = _build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)

    pdf_dir = LightOnOcrTableExtractor.resolve_pdf_dir(args.input)
    extractor = LightOnOcrTableExtractor(
        pdf_dir=pdf_dir,
        output_path=args.output,
        model_id=args.model,
        verbose=args.verbose,
    )

    pdf_files = extractor.list_pdfs()
    log.info("PDF_DIR: %s", pdf_dir)
    log.info("PDFs found: %d", len(pdf_files))
    for p in pdf_files:
        log.info("  - %s", p.name)

    out_path = extractor.run()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    total_tables = sum(d["num_tables"] for d in result["documents"])
    print(f"Documents: {len(result['documents'])} | Total tables: {total_tables}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
