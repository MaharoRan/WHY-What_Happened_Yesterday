from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import ingest

REPO_ROOT = Path(__file__).resolve().parents[3]

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test</title>
    <item>
      <title>Article d'hier</title>
      <link>https://example.com/hier</link>
      <description>Le resume</description>
      <pubDate>Wed, 12 Nov 2025 12:00:00 GMT</pubDate>
      <author>Jeanne Dupont</author>
    </item>
    <item>
      <title>Article d'aujourd'hui</title>
      <link>https://example.com/aujourdhui</link>
      <pubDate>Thu, 13 Nov 2025 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Sans date</title>
      <link>https://example.com/sans-date</link>
    </item>
  </channel>
</rss>
"""

DAY = date(2025, 11, 12)
FEED = {"id": "test", "name": "Test", "url": "https://example.com/rss"}


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


# --- configuration ---------------------------------------------------------


def test_repository_sources_file_is_valid():
    config = REPO_ROOT / "config" / "sources.yml"
    feeds = yaml.safe_load(config.read_text(encoding="utf-8"))["feeds"]
    by_id = {f["id"]: f for f in feeds}
    assert "lemonde" in by_id
    assert all(f["name"] for f in feeds)
    assert all(str(f["url"]).startswith("https://") for f in feeds if f.get("url"))


# --- parsing ---------------------------------------------------------------


def test_clean_journalist_normalises_names():
    assert ingest.clean_journalist("  Jeanne   Dupont ", "Le Monde") == "Jeanne Dupont"
    assert ingest.clean_journalist("A, B; A", "Le Monde") == "A, B"


def test_clean_journalist_returns_none_when_absent_or_media_name():
    assert ingest.clean_journalist(None, "Le Monde") is None
    assert ingest.clean_journalist("", "Le Monde") is None
    assert ingest.clean_journalist("Le Monde", "Le Monde") is None
    assert ingest.clean_journalist("le-monde", "Le Monde") is None


# --- fetch -----------------------------------------------------------------


def test_fetch_keeps_only_target_date(monkeypatch):
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResponse(RSS))

    rows = ingest.fetch(FEED, DAY)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "Test"
    assert row["title"] == "Article d'hier"
    assert row["link"] == "https://example.com/hier"
    assert row["date"].startswith("2025-11-12")
    assert row["journalist"] == "Jeanne Dupont"


def test_fetch_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        ingest.requests, "get", lambda *a, **k: FakeResponse(b"", status=429)
    )
    with pytest.raises(RuntimeError):
        ingest.fetch(FEED, DAY)


# --- main ------------------------------------------------------------------


def _config(tmp_path, monkeypatch):
    config = tmp_path / "sources.yml"
    config.write_text(
        "feeds:\n"
        "  - {id: test, name: Test, url: 'https://example.com/rss'}\n"
        "  - {id: other, name: Other, url: 'https://example.com/other'}\n"
        "  - {id: nourl, name: NoUrl}\n",
        encoding="utf-8",
    )
    out = tmp_path / "bronze"
    monkeypatch.setattr(ingest, "FEEDS", str(config))
    monkeypatch.setattr(ingest, "OUT", str(out))
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResponse(RSS))
    return out


def test_main_writes_bronze_partition(tmp_path, monkeypatch):
    out = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("SOURCES", "test")

    ingest.main(DAY)

    path = out / "source=test" / "date=2025-11-12" / "articles.parquet"
    assert path.exists()
    assert not (out / "source=other").exists()
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert "journalist" in df.columns


def test_main_skips_failing_source_and_feeds_without_url(tmp_path, monkeypatch):
    out = _config(tmp_path, monkeypatch)
    monkeypatch.delenv("SOURCES", raising=False)
    monkeypatch.setattr(
        ingest.requests,
        "get",
        lambda url, **k: FakeResponse(b"", 500 if url.endswith("other") else 200)
        if url.endswith("other")
        else FakeResponse(RSS),
    )

    ingest.main(DAY)

    assert (out / "source=test" / "date=2025-11-12" / "articles.parquet").exists()
    assert not (out / "source=other").exists()
    assert not (out / "source=nourl").exists()
