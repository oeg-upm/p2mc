# Backend test layout

Run backend tests with:

```powershell
cd backend
poetry run pytest
```

For a more readable output while reviewing the suite:

```powershell
poetry run pytest -vv --color=yes -ra
```

For debugging with live `print()` output and logs:

```powershell
poetry run pytest -vv -s --color=yes -ra --log-cli-level=INFO
```

## Structure

- `pipeline/`: PDF processing and ModelCard generation units with fake PDFs, XML, OCR JSON, and mocked external models/services.
- `worker/`: RabbitMQ message handling, `status.json` transitions, cached job reuse, artifact validation, and failure paths.
- `utils/`: utility-level tests, currently focused on XML parsing from GROBID-like TEI files.
- `fixtures/`: small fake input/output files used by tests. Keep fixtures tiny and deterministic.

Tests should not call real GROBID, LightOCR, Ollama, Hugging Face, arXiv, LPWC, or SemOpenAlex. Use fixtures and fakes/mocks instead.

## Test files

### `test_test_setup.py`

- `test_backend_package_is_importable`: checks that the backend package can be imported and resolves its base directory.

### `utils/test_xml_parser.py`

- `test_xml_parser_reads_metadata_and_text_from_grobid_fixture`: checks title, abstract, body text, sections, and authors from TEI XML.
- `test_xml_parser_returns_all_sections_when_no_target_filter_is_given`: checks that `XMLParser` can expose all body sections, not only target ones.
- `test_xml_parser_falls_back_to_plain_title_and_extracts_arxiv_id`: checks title fallback and arXiv ID parsing from `idno` values with versions.
- `test_xml_parser_returns_empty_values_when_optional_blocks_are_missing`: checks missing optional TEI blocks return empty defaults instead of errors.
- `test_xml_parser_raises_when_xml_file_does_not_exist`: checks that missing XML fixtures fail early with a clear file error.

### `pipeline/test_pdf_handler.py`

- `test_extract_id_from_url_reads_arxiv_id_from_pdf_url`: checks that `PDFHandler` can derive the paper ID from an arXiv PDF URL.
- `test_extract_id_from_url_rejects_urls_without_arxiv_id`: checks that non-arXiv-style URLs fail before pipeline paths are built.
- `test_emit_stage_forwards_progress_payload`: checks that pipeline stage callbacks receive step metadata and details.
- `test_process_with_scipdf_returns_generated_xml_path`: checks that SciPDF success is accepted only when the XML file exists.
- `test_process_with_scipdf_raises_when_xml_is_missing`: checks that a successful-looking SciPDF call without XML is an error.
- `test_process_with_lightocr_reuses_existing_json`: checks that existing OCR JSON avoids loading the heavy LightOCR parser.
- `test_extract_values_reads_xml_and_lightocr_json`: checks that XML and OCR artifacts are merged into extracted pipeline data.
- `test_extract_values_omits_tables_when_lightocr_json_is_missing`: checks XML extraction still works when OCR JSON has not been generated.
- `test_extract_values_keeps_full_text_when_target_sections_are_missing`: checks no-target XML still provides full text for later extractor fallback.
- `test_handle_pdf_reuses_existing_pdf_and_saves_modelcard`: checks the happy `PDFHandler` flow with SciPDF, LightOCR, and card generation mocked.
- `test_handle_pdf_downloads_missing_pdf_before_processing`: checks that `PDFHandler` downloads the PDF when it is not already stored.
- `test_handle_pdf_without_detected_model_skips_lightocr_but_returns_error`: checks the current no-model branch where LightOCR is skipped but JSON setup fails.

### `pipeline/test_model_card_generation_pipeline.py`

- `test_get_tsv_tables_converts_lightocr_tables_to_tsv`: checks that LightOCR table JSON is converted into TSV prompt context.
- `test_get_tsv_tables_returns_empty_list_for_empty_input`: checks that missing OCR table data does not break ModelCard generation.
- `test_get_tsv_tables_accepts_nested_tables_documents_shape`: checks the alternative LightOCR shape where documents live under tables.
- `test_clean_empty_fields_removes_nested_empty_values`: checks that empty JSON-LD values are removed recursively.
- `test_match_tasks_deduplicates_tasks_by_matched_uri`: checks that repeated task predictions produce one JSON-LD task object.
- `test_generate_modelcard_puts_metrics_under_has_evaluation`: checks that generated metrics use `hasEvaluation`, not `evaluatedOn`.
- `test_generate_modelcard_populates_core_fields_from_fixture_data`: checks the final ModelCard core fields without loading real models.
- `test_generate_modelcard_uses_sections_as_dataset_and_metric_context`: checks that target XML sections are preferred as extractor context.
- `test_generate_modelcard_uses_full_text_when_sections_are_empty`: checks fallback context when XML has no Experiments/Evaluation/Results.
- `test_generate_modelcard_uses_abstract_when_sections_and_full_text_are_empty`: checks the final text fallback used for dataset and metric extraction.
- `test_generate_modelcard_removes_has_evaluation_when_no_metrics`: checks that empty metric extraction removes the empty evaluation block.

### `worker/test_worker_status.py`

- `test_validate_message_accepts_valid_job_message`: checks that valid RabbitMQ payloads are normalized for processing.
- `test_validate_message_rejects_invalid_job_id`: checks that invalid job UUIDs are rejected before status files are used.
- `test_validate_message_rejects_invalid_arxiv_id`: checks that malformed arXiv IDs cannot reach `PDFHandler` paths.
- `test_validate_message_rejects_non_http_urls`: checks that worker messages cannot point to unsupported URL schemes.
- `test_write_status_atomic_and_read_status_round_trip`: checks that status JSON is written atomically and read back as a dict.
- `test_get_artifact_paths_uses_data_root`: checks that worker artifact paths are built under the configured data root.
- `test_serialize_existing_artifact_paths_keeps_only_existing_files`: checks that API artifacts expose only files present on disk.
- `test_read_existing_modelcard_returns_card_when_json_is_valid`: checks that cached ModelCards can be reused by worker jobs.
- `test_read_existing_modelcard_returns_none_for_invalid_json`: checks that corrupt cached ModelCards are ignored safely.
- `test_verify_artifacts_raises_when_expected_file_is_missing`: checks that completed jobs fail if required artifacts were not produced.
- `test_process_job_marks_status_failed_when_pdf_handler_returns_error`: checks that `PDFHandler` errors are persisted as failed job status.
