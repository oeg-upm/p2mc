from __future__ import annotations


def test_backend_package_is_importable() -> None:
    import backend

    assert backend.BASE_DIR.name == "backend"
