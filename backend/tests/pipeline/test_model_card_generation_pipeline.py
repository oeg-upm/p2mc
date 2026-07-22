"""Tests for ModelCardGenerator.

Planned scope:
- template population;
- TSV table extraction;
- dataset and metric extraction wiring with mocked extractors;
- empty field cleanup;
- repository and reference publication cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.model_card_generation_pipeline import ModelCardGenerator
from backend.utils.uri_builder import UriBuilder


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def build_generator_without_init() -> ModelCardGenerator:
    return ModelCardGenerator.__new__(ModelCardGenerator)


class FakeLlamaExtractor:
    def __init__(self, repositories: list[str] | None = None) -> None:
        self.repositories = repositories or [
            "https://github.com/example/fixture-model"
        ]

    def extract(self, text: str, question: str) -> list[str]:
        if "name of the model" in question:
            return ["FixtureModel"]

        if "tasks addressed" in question:
            return ["link prediction"]

        if "paper implementation" in question:
            return self.repositories

        return []


class FakeSummarizer:
    def summarize(self, text: str) -> str:
        return "Fixture summary."

    def get_keywords(self, text: str) -> list[str]:
        return ["knowledge graph embedding"]


class FakeDatasetExtractor:
    def __init__(self, datasets: list[str]) -> None:
        self.datasets = datasets
        self.calls: list[tuple[str, list[str]]] = []

    def extract(self, text: str, tables: list[str]) -> list[str]:
        self.calls.append((text, tables))
        return self.datasets


class FakeMetricExtractor:
    def __init__(self, metrics: list[str]) -> None:
        self.metrics = metrics
        self.calls: list[tuple[str, list[str]]] = []

    def extract(self, text: str, tables: list[str]) -> list[str]:
        self.calls.append((text, tables))
        return self.metrics


class FakeUriFetcher:
    def guess_dataset_uri(self, dataset: str) -> str | None:
        return None

    def guess_metric_uri(self, metric: str) -> str | None:
        if metric == "MRR":
            return "https://linkedpaperswithcode.com/metric/mrr"

        return None

    def extract_author_uri(self, author: str, arxiv_id: str) -> str | None:
        return None


class FakeCategoryMapper:
    def get_category_object(self, category: str) -> dict[str, str]:
        return {
            "@id": "https://w3id.org/p2mc/category/other",
            "name": "Other KGC technologies",
        }


class FakeTaskMatcher:
    def match_task(self, task: str) -> dict[str, str] | None:
        return {
            "uri": "https://linkedpaperswithcode.com/task/link-prediction",
            "name": "Link Prediction",
        }


def build_generator_for_generate_modelcard(
    *,
    datasets: list[str] | None = None,
    metrics: list[str] | None = None,
    repositories: list[str] | None = None,
) -> ModelCardGenerator:
    generator = build_generator_without_init()
    generator._log = lambda message: None
    generator._model_template = {
        "@context": "https://w3id.org/fair4ml",
        "@id": "",
        "@type": "MLModel",
        "name": "",
        "description": "",
        "keywords": [],
        "version": "1.0.0",
        "author": [],
        "dateCreated": "",
        "mlTask": [],
        "modelCategory": {
            "@id": "",
            "name": "",
        },
        "evaluatedOn": [],
        "hasEvaluation": {
            "@type": "fair4ml:MLModelEvaluation",
            "evaluationMetrics": [],
        },
        "referencePublication": {
            "@type": "ScholarlyArticle",
            "author": [],
        },
        "codeRepository": "",
    }
    generator._uri_builder = UriBuilder()
    generator._uri_fetcher = FakeUriFetcher()
    generator._llama = FakeLlamaExtractor(repositories)
    generator._qwen_dataset_extractor = FakeDatasetExtractor(datasets or [])
    generator._qwen_metric_extractor = FakeMetricExtractor(metrics or [])
    generator._summarizer = FakeSummarizer()
    generator._task_matcher = FakeTaskMatcher()
    generator._category_mapper = FakeCategoryMapper()
    generator._extract_classification = lambda text: "other"
    generator._extract_reference_publication = lambda arxiv_id: None
    return generator


def build_extracted_data() -> dict:
    return {
        "arxiv_id": "1802.09691",
        "title": "Fixture Model Paper",
        "abstract": "Fixture abstract.",
        "full_text": "FixtureModel reports MRR and Hits@10 on WN18RR.",
        "sections": "Experiments report MRR and Hits@10 on WN18RR.",
        "authors": ["Jane Doe"],
        "tables": {
            "documents": [
                {
                    "tables": [
                        {
                            "evaluation": {
                                "columns": ["Dataset", "MRR", "Hits@10"],
                            },
                            "rows": [["WN18RR", "0.53", "0.62"]],
                        }
                    ]
                }
            ]
        },
    }


def test_get_tsv_tables_converts_lightocr_tables_to_tsv() -> None:
    """Checks that LightOCR table JSON is converted into TSV prompt context."""
    generator = build_generator_without_init()

    raw_tables = {
        "documents": [
            {
                "tables": [
                    {
                        "evaluation": {
                            "columns": ["Dataset", "MRR"],
                        },
                        "rows": [
                            ["WN18RR", "0.53"],
                            ["FB15k-237", "0.42"],
                        ],
                    }
                ]
            }
        ]
    }

    assert generator._get_tsv_tables(raw_tables) == [
        "Dataset\tMRR\nWN18RR\t0.53\nFB15k-237\t0.42"
    ]


def test_get_tsv_tables_returns_empty_list_for_empty_input() -> None:
    """Checks that missing OCR table data does not break ModelCard generation."""
    generator = build_generator_without_init()

    assert generator._get_tsv_tables(None) == []
    assert generator._get_tsv_tables({}) == []


def test_get_tsv_tables_accepts_nested_tables_documents_shape() -> None:
    """Checks the alternative LightOCR shape where documents live under tables."""
    generator = build_generator_without_init()

    raw_tables = {
        "tables": {
            "documents": [
                {
                    "tables": [
                        {
                            "evaluation": {
                                "columns": ["Dataset", "Accuracy"],
                            },
                            "rows": [["ToySet", "0.91"]],
                        }
                    ]
                }
            ]
        }
    }

    assert generator._get_tsv_tables(raw_tables) == [
        "Dataset\tAccuracy\nToySet\t0.91"
    ]


def test_clean_empty_fields_removes_nested_empty_values() -> None:
    """Checks that empty JSON-LD values are removed recursively."""
    generator = build_generator_without_init()

    data = {
        "@id": "https://w3id.org/p2mc/model/sample",
        "name": "",
        "keywords": ["KGE", "", None],
        "referencePublication": {
            "@type": "ScholarlyArticle",
            "@id": "",
            "author": [{}],
        },
        "evaluatedOn": [],
    }

    assert generator._clean_empty_fields(data) == {
        "@id": "https://w3id.org/p2mc/model/sample",
        "keywords": ["KGE"],
        "referencePublication": {
            "@type": "ScholarlyArticle",
        },
    }


def test_match_tasks_deduplicates_tasks_by_matched_uri() -> None:
    """Checks that repeated task predictions produce one JSON-LD task object."""
    generator = build_generator_without_init()

    class FakeTaskMatcher:
        def match_task(self, task: str) -> dict[str, str] | None:
            if task == "ignored":
                return None

            return {
                "uri": "https://linkedpaperswithcode.com/task/link-prediction",
                "name": "Link Prediction",
            }

    generator._task_matcher = FakeTaskMatcher()

    assert generator._match_tasks(
        [
            "link prediction",
            "knowledge graph completion",
            "ignored",
        ]
    ) == [
        {
            "@id": "https://linkedpaperswithcode.com/task/link-prediction",
            "name": "Link Prediction",
        }
    ]


def test_clean_code_repositories_keeps_only_real_repository_urls() -> None:
    """Checks that generic domains are removed from codeRepository values."""
    generator = build_generator_without_init()

    assert generator._clean_code_repositories(
        [
            "https://github.com/",
            "https://github.com",
            "github.com",
            "https://gitlab.com/",
            "https://github.com/example/fixture-model",
            " https://github.com/example/fixture-model ",
            "",
            None,
        ]
    ) == [
        "https://github.com/example/fixture-model",
    ]


def test_generate_modelcard_puts_metrics_under_has_evaluation() -> None:
    """Checks that generated metrics use hasEvaluation, not evaluatedOn."""
    generator = build_generator_for_generate_modelcard(
        datasets=["WN18RR"],
        metrics=["MRR", "Hits@10"],
    )

    card = generator.generate_modelcard(build_extracted_data())

    assert card["evaluatedOn"] == [
        {
            "@id": "https://w3id.org/p2mc/dataset/wn18rr",
            "name": "WN18RR",
        }
    ]
    assert card["hasEvaluation"] == {
        "@type": "fair4ml:MLModelEvaluation",
        "evaluationMetrics": [
            {
                "@id": "https://linkedpaperswithcode.com/metric/mrr",
                "name": "MRR",
            },
            {
                "@id": "https://w3id.org/p2mc/metric/hits-10",
                "name": "Hits@10",
            },
        ],
    }
    assert "MRR" not in {
        item["name"]
        for item in card["evaluatedOn"]
    }


def test_generate_modelcard_populates_core_fields_from_fixture_data() -> None:
    """Checks the final ModelCard core fields without loading real models."""
    generator = build_generator_for_generate_modelcard(
        datasets=["WN18RR"],
        metrics=["MRR"],
    )
    extracted_data = build_extracted_data()
    extracted_data["tables"] = json.loads(
        (FIXTURES_DIR / "lightocr_json" / "sample.json").read_text(
            encoding="utf-8"
        )
    )

    card = generator.generate_modelcard(extracted_data)

    assert card["@context"] == "https://w3id.org/fair4ml"
    assert card["@id"] == "https://w3id.org/p2mc/model/1802.09691"
    assert card["name"] == "FixtureModel"
    assert card["description"] == "Fixture summary."
    assert card["keywords"] == ["knowledge graph embedding"]
    assert card["author"] == [
        {
            "@id": "https://w3id.org/p2mc/author/doe-991bab3fde",
            "name": "Jane Doe",
        }
    ]
    assert card["mlTask"] == [
        {
            "@id": "https://linkedpaperswithcode.com/task/link-prediction",
            "name": "Link Prediction",
        }
    ]
    assert card["codeRepository"] == ["https://github.com/example/fixture-model"]
    assert card["referencePublication"] == {
        "@type": "ScholarlyArticle",
        "@id": "1802.09691",
        "name": "Fixture Model Paper",
        "author": [
            {
                "@type": "Person",
                "name": "Jane Doe",
            }
        ],
        "url": "https://arxiv.org/abs/1802.09691",
    }


def test_generate_modelcard_omits_code_repository_when_only_generic_urls_are_found() -> None:
    """Checks generic LLM repository answers are not exported in JSON-LD."""
    generator = build_generator_for_generate_modelcard(
        datasets=["WN18RR"],
        metrics=["MRR"],
        repositories=[
            "https://github.com/",
            "github.com",
            "",
        ],
    )

    card = generator.generate_modelcard(build_extracted_data())

    assert "codeRepository" not in card


def test_generate_modelcard_adds_semopenalex_link_to_reference_publication() -> None:
    """Checks SemOpenAlex data enriches the local reference publication metadata."""
    generator = build_generator_for_generate_modelcard(
        datasets=["WN18RR"],
        metrics=["MRR"],
    )
    generator._extract_reference_publication = lambda arxiv_id: {
        "@type": "ScholarlyArticle",
        "@id": arxiv_id,
        "name": "SemOpenAlex Fixture Title",
        "author": [
            {
                "@type": "Person",
                "@id": "https://semopenalex.org/authors/A1",
                "name": "Jane Doe",
            }
        ],
        "schema:sameAs": "https://semopenalex.org/works/W1",
        "url": "https://arxiv.org/abs/1802.09691",
    }

    card = generator.generate_modelcard(build_extracted_data())

    assert card["referencePublication"] == {
        "@type": "ScholarlyArticle",
        "@id": "1802.09691",
        "name": "SemOpenAlex Fixture Title",
        "author": [
            {
                "@type": "Person",
                "@id": "https://semopenalex.org/authors/A1",
                "name": "Jane Doe",
            }
        ],
        "schema:sameAs": "https://semopenalex.org/works/W1",
        "url": "https://arxiv.org/abs/1802.09691",
    }


def test_generate_modelcard_uses_sections_as_dataset_and_metric_context() -> None:
    """Checks that target XML sections are preferred as extractor context."""
    generator = build_generator_for_generate_modelcard(
        datasets=["WN18RR"],
        metrics=["MRR"],
    )

    generator.generate_modelcard(build_extracted_data())

    assert generator._qwen_dataset_extractor.calls[0][0] == (
        "Experiments report MRR and Hits@10 on WN18RR."
    )
    assert generator._qwen_metric_extractor.calls[0][0] == (
        "Experiments report MRR and Hits@10 on WN18RR."
    )


def test_generate_modelcard_uses_full_text_when_sections_are_empty() -> None:
    """Checks fallback context when XML has no Experiments/Evaluation/Results."""
    generator = build_generator_for_generate_modelcard(
        datasets=["WN18RR"],
        metrics=["MRR"],
    )
    extracted_data = build_extracted_data()
    extracted_data["sections"] = ""
    extracted_data["full_text"] = "Full text mentions WN18RR and MRR."

    generator.generate_modelcard(extracted_data)

    assert generator._qwen_dataset_extractor.calls[0][0] == (
        "Full text mentions WN18RR and MRR."
    )
    assert generator._qwen_metric_extractor.calls[0][0] == (
        "Full text mentions WN18RR and MRR."
    )


def test_generate_modelcard_uses_abstract_when_sections_and_full_text_are_empty() -> None:
    """Checks the final text fallback used for dataset and metric extraction."""
    generator = build_generator_for_generate_modelcard(
        datasets=["WN18RR"],
        metrics=["MRR"],
    )
    extracted_data = build_extracted_data()
    extracted_data["sections"] = ""
    extracted_data["full_text"] = ""
    extracted_data["abstract"] = "Abstract mentions WN18RR and MRR."

    generator.generate_modelcard(extracted_data)

    assert generator._qwen_dataset_extractor.calls[0][0] == (
        "Abstract mentions WN18RR and MRR."
    )
    assert generator._qwen_metric_extractor.calls[0][0] == (
        "Abstract mentions WN18RR and MRR."
    )


def test_generate_modelcard_removes_has_evaluation_when_no_metrics() -> None:
    """Checks that empty metric extraction removes the empty evaluation block."""
    generator = build_generator_for_generate_modelcard(
        datasets=["WN18RR"],
        metrics=[],
    )

    card = generator.generate_modelcard(build_extracted_data())

    assert "hasEvaluation" not in card
    assert card["evaluatedOn"] == [
        {
            "@id": "https://w3id.org/p2mc/dataset/wn18rr",
            "name": "WN18RR",
        }
    ]
