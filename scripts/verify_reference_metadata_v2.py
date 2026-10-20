from __future__ import annotations

import csv
import difflib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"

INPUT_CSV = PAPER_DIR / "reference_usage_map.csv"

OUT_CSV = PAPER_DIR / "bibliography_verification_v2.csv"
OUT_JSON = PAPER_DIR / "bibliography_metadata_v2.json"
OUT_BIB = PAPER_DIR / "references_recent_verified.bib"
OUT_AUDIT = PAPER_DIR / "bibliography_audit_v2.md"
OUT_CORRECTIONS = PAPER_DIR / "reference_year_corrections.csv"

USER_AGENT = (
    "ScaleFormer-Bibliography-Audit/2.0 "
    "(research bibliography metadata verification)"
)

TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
PAUSE_SECONDS = 0.15
TITLE_MATCH_THRESHOLD = 0.72

# Final issue/publication years manually checked against publisher/index records.
# These override early-access / DOI-registration years where necessary.
YEAR_OVERRIDES = {
    "49": {
        "year": 2026,
        "reason": (
            "Final IEEE Wireless Communications Letters publication is "
            "Vol. 15, pp. 880-884 (2026); DOI was registered in 2025."
        ),
    },
    "50": {
        "year": 2024,
        "reason": (
            "Final IEEE Wireless Communications Letters publication is "
            "Vol. 13, No. 12, pp. 3320-3324 (2024)."
        ),
    },
}


def request_json(
    url: str,
    accept: str = "application/json",
) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
        },
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                req,
                timeout=TIMEOUT_SECONDS,
            ) as response:
                payload = response.read()
                return json.loads(
                    payload.decode("utf-8")
                )
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                time.sleep(0.8 * attempt)

    raise RuntimeError(
        f"Request failed after {MAX_RETRIES} attempts: "
        f"{url} | {last_error}"
    )


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def expected_title(reference: str) -> str:
    text = reference.strip()

    if ", " in text:
        prefix, remainder = text.split(", ", 1)

        if (
            " et al." in prefix.lower()
            or " and " in prefix.lower()
            or len(prefix.split()) <= 5
        ):
            return remainder.strip()

    return text


def title_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    return difflib.SequenceMatcher(
        None,
        a_norm,
        b_norm,
    ).ratio()


def fetch_openalex(doi: str) -> dict:
    work_id = "https://doi.org/" + doi.lower()
    url = (
        "https://api.openalex.org/works/"
        + urllib.parse.quote(
            work_id,
            safe=":/",
        )
    )

    raw = request_json(url)

    biblio = raw.get("biblio") or {}

    source = (
        ((raw.get("primary_location") or {}).get("source"))
        or {}
    )

    authors = []

    for authorship in raw.get("authorships", []) or []:
        author = authorship.get("author") or {}
        name = str(
            author.get("display_name") or ""
        ).strip()

        if name:
            authors.append(name)

    first_page = str(
        biblio.get("first_page") or ""
    ).strip()
    last_page = str(
        biblio.get("last_page") or ""
    ).strip()

    pages = ""

    if first_page and last_page:
        pages = f"{first_page}-{last_page}"
    elif first_page:
        pages = first_page

    return {
        "provider": "OpenAlex",
        "doi": doi,
        "title": str(
            raw.get("title") or ""
        ).strip(),
        "year": raw.get("publication_year"),
        "venue": str(
            source.get("display_name") or ""
        ).strip(),
        "volume": str(
            biblio.get("volume") or ""
        ).strip(),
        "issue": str(
            biblio.get("issue") or ""
        ).strip(),
        "pages": pages,
        "article_number": "",
        "publisher": str(
            source.get("host_organization_name") or ""
        ).strip(),
        "type": str(
            raw.get("type") or ""
        ).strip(),
        "authors": authors,
        "url": str(
            (raw.get("primary_location") or {}).get("landing_page_url")
            or ""
        ).strip(),
        "raw": raw,
    }


def csl_year(raw: dict) -> int | None:
    for key in (
        "issued",
        "published-print",
        "published-online",
    ):
        block = raw.get(key)

        if not isinstance(block, dict):
            continue

        parts = block.get("date-parts")

        if (
            isinstance(parts, list)
            and parts
            and isinstance(parts[0], list)
            and parts[0]
        ):
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                pass

    return None


def fetch_doi_csl(doi: str) -> dict:
    url = "https://doi.org/" + urllib.parse.quote(
        doi,
        safe="/",
    )

    raw = request_json(
        url,
        accept="application/vnd.citationstyles.csl+json",
    )

    authors = []

    for author in raw.get("author", []) or []:
        given = str(
            author.get("given") or ""
        ).strip()
        family = str(
            author.get("family") or ""
        ).strip()

        full = " ".join(
            x for x in (given, family) if x
        ).strip()

        if full:
            authors.append(full)

    container = raw.get("container-title", "")

    if isinstance(container, list):
        venue = str(
            container[0] if container else ""
        )
    else:
        venue = str(container or "")

    page = str(
        raw.get("page") or ""
    ).strip()

    return {
        "provider": "DOI-CSL",
        "doi": str(
            raw.get("DOI") or doi
        ),
        "title": str(
            raw.get("title") or ""
        ).strip(),
        "year": csl_year(raw),
        "venue": venue.strip(),
        "volume": str(
            raw.get("volume") or ""
        ).strip(),
        "issue": str(
            raw.get("issue") or ""
        ).strip(),
        "pages": page,
        "article_number": str(
            raw.get("article-number") or ""
        ).strip(),
        "publisher": str(
            raw.get("publisher") or ""
        ).strip(),
        "type": str(
            raw.get("type") or ""
        ).strip(),
        "authors": authors,
        "url": str(
            raw.get("URL") or ""
        ).strip(),
        "raw": raw,
    }


def fetch_crossref(doi: str) -> dict:
    encoded = urllib.parse.quote(
        doi,
        safe="",
    )

    raw = request_json(
        "https://api.crossref.org/works/"
        + encoded
    ).get("message", {})

    title_value = raw.get("title") or []
    venue_value = raw.get("container-title") or []

    title = (
        str(title_value[0])
        if isinstance(title_value, list)
        and title_value
        else str(title_value or "")
    )

    venue = (
        str(venue_value[0])
        if isinstance(venue_value, list)
        and venue_value
        else str(venue_value or "")
    )

    year = None

    for key in (
        "published-print",
        "published-online",
        "published",
        "issued",
    ):
        block = raw.get(key)

        if not isinstance(block, dict):
            continue

        parts = block.get("date-parts")

        if (
            isinstance(parts, list)
            and parts
            and isinstance(parts[0], list)
            and parts[0]
        ):
            year = int(parts[0][0])
            break

    authors = []

    for author in raw.get("author", []) or []:
        given = str(
            author.get("given") or ""
        ).strip()
        family = str(
            author.get("family") or ""
        ).strip()

        full = " ".join(
            x for x in (given, family) if x
        ).strip()

        if full:
            authors.append(full)

    return {
        "provider": "Crossref",
        "doi": str(
            raw.get("DOI") or doi
        ),
        "title": title.strip(),
        "year": year,
        "venue": venue.strip(),
        "volume": str(
            raw.get("volume") or ""
        ).strip(),
        "issue": str(
            raw.get("issue") or ""
        ).strip(),
        "pages": str(
            raw.get("page") or ""
        ).strip(),
        "article_number": str(
            raw.get("article-number") or ""
        ).strip(),
        "publisher": str(
            raw.get("publisher") or ""
        ).strip(),
        "type": str(
            raw.get("type") or ""
        ).strip(),
        "authors": authors,
        "url": str(
            raw.get("URL") or ""
        ).strip(),
        "raw": raw,
    }


def fetch_metadata(doi: str) -> tuple[dict | None, list[str]]:
    attempts = [
        ("OpenAlex", fetch_openalex),
        ("DOI-CSL", fetch_doi_csl),
        ("Crossref", fetch_crossref),
    ]

    errors = []

    for label, function in attempts:
        try:
            meta = function(doi)

            if meta["title"]:
                return meta, errors

            errors.append(
                f"{label}: returned empty title"
            )
        except Exception as exc:
            errors.append(
                f"{label}: {exc}"
            )

    return None, errors


def protect_title(title: str) -> str:
    protected = title

    acronyms = [
        "CSI",
        "MIMO",
        "OFDM",
        "FDD",
        "TDD",
        "THz",
        "RIS",
        "CNN",
        "AI",
        "VQ",
        "VAE",
        "LLM",
        "BiLSTM",
        "Mamba",
    ]

    for token in acronyms:
        protected = re.sub(
            rf"\b{re.escape(token)}\b",
            "{" + token + "}",
            protected,
            flags=re.IGNORECASE,
        )

    return protected


def bib_type(meta: dict) -> str:
    type_text = str(
        meta.get("type") or ""
    ).lower()

    venue_text = str(
        meta.get("venue") or ""
    ).lower()

    if (
        "proceedings" in type_text
        or "conference" in type_text
        or "conference" in venue_text
    ):
        return "inproceedings"

    return "article"


def bib_key(
    meta: dict,
    year: int,
    used: set[str],
) -> str:
    if meta["authors"]:
        surname = normalize_text(
            meta["authors"][0].split()[-1]
        ).replace(" ", "")
    else:
        surname = "ref"

    title_words = [
        word
        for word in normalize_text(
            meta["title"]
        ).split()
        if word not in {
            "a",
            "an",
            "the",
            "of",
            "for",
            "with",
            "and",
            "in",
            "on",
            "to",
            "based",
        }
    ]

    keyword = (
        title_words[0]
        if title_words
        else "work"
    )

    base = f"{surname}{year}{keyword}"
    key = base
    suffix = ord("a")

    while key in used:
        key = base + chr(suffix)
        suffix += 1

    used.add(key)
    return key


def bib_entry(
    key: str,
    meta: dict,
    final_year: int,
) -> str:
    kind = bib_type(meta)

    fields: list[tuple[str, str]] = []

    if meta["authors"]:
        fields.append(
            ("author", " and ".join(meta["authors"]))
        )

    fields.append(
        ("title", protect_title(meta["title"]))
    )

    if meta["venue"]:
        fields.append(
            (
                "booktitle" if kind == "inproceedings"
                else "journal",
                meta["venue"],
            )
        )

    fields.append(
        ("year", str(final_year))
    )

    if meta["volume"]:
        fields.append(
            ("volume", meta["volume"])
        )

    if meta["issue"]:
        fields.append(
            ("number", meta["issue"])
        )

    if meta["pages"]:
        fields.append(
            ("pages", meta["pages"])
        )

    if meta["article_number"]:
        fields.append(
            (
                "note",
                "Article " + meta["article_number"],
            )
        )

    fields.append(
        ("doi", meta["doi"])
    )

    lines = [f"@{kind}{{{key},"]

    for idx, (name, value) in enumerate(fields):
        comma = "," if idx < len(fields) - 1 else ""
        value = (
            value
            .replace("&", r"\&")
            .replace("%", r"\%")
        )

        lines.append(
            f"  {name:<10} = {{{value}}}{comma}"
        )

    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    corrections = []

    for audit_id, item in YEAR_OVERRIDES.items():
        corrections.append({
            "audit_id": audit_id,
            "final_year": item["year"],
            "reason": item["reason"],
        })

    with OUT_CORRECTIONS.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audit_id",
                "final_year",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(corrections)

    results = []
    metadata_records = []
    bib_records = []
    used_keys: set[str] = set()

    print("=" * 100)
    print("SCALE-FORMER BIBLIOGRAPHY VERIFICATION V2")
    print("OpenAlex -> DOI CSL -> Crossref")
    print("=" * 100)

    for index, row in enumerate(
        rows,
        start=1,
    ):
        audit_id = str(
            row.get("audit_id") or index
        ).strip()

        audit_year = int(
            row["year"]
        )

        target_year = int(
            YEAR_OVERRIDES.get(
                audit_id,
                {"year": audit_year},
            )["year"]
        )

        doi = row["doi"].strip()
        audit_reference = row[
            "reference"
        ].strip()
        audit_title = expected_title(
            audit_reference
        )

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"#{audit_id} {doi}"
        )

        meta, errors = fetch_metadata(
            doi
        )

        if meta is None:
            results.append({
                "audit_id": audit_id,
                "audit_year": audit_year,
                "target_final_year": target_year,
                "doi": doi,
                "provider": "",
                "metadata_year": "",
                "title_similarity": "",
                "status": "ERROR",
                "metadata_title": "",
                "metadata_venue": "",
                "note": " | ".join(errors),
            })
            continue

        similarity = title_similarity(
            audit_title,
            meta["title"],
        )

        metadata_year = meta["year"]

        if similarity < TITLE_MATCH_THRESHOLD:
            status = "REVIEW_TITLE"
            note = (
                "DOI resolved but title similarity "
                "is below threshold."
            )
        elif (
            metadata_year is not None
            and int(metadata_year) != target_year
        ):
            # A mismatch may reflect early-access vs final issue year.
            # It remains reviewable unless the target year was explicitly
            # overridden above from final issue metadata.
            if audit_id in YEAR_OVERRIDES:
                status = "VERIFIED_OVERRIDE"
                note = (
                    "Title verified; final issue year "
                    "uses documented manual override. "
                    f"Provider year={metadata_year}."
                )
            else:
                status = "REVIEW_YEAR"
                note = (
                    "Title verified, but metadata year "
                    f"{metadata_year} differs from target "
                    f"final year {target_year}."
                )
        else:
            status = (
                "VERIFIED_OVERRIDE"
                if audit_id in YEAR_OVERRIDES
                else "VERIFIED"
            )
            note = (
                "DOI/title/year verified."
                if audit_id not in YEAR_OVERRIDES
                else "DOI/title verified and documented final-year override applied."
            )

        result = {
            "audit_id": audit_id,
            "audit_year": audit_year,
            "target_final_year": target_year,
            "doi": doi,
            "provider": meta["provider"],
            "metadata_year": (
                "" if metadata_year is None
                else metadata_year
            ),
            "title_similarity": f"{similarity:.4f}",
            "status": status,
            "metadata_title": meta["title"],
            "metadata_venue": meta["venue"],
            "note": note,
        }

        results.append(result)

        metadata_records.append({
            "audit_id": audit_id,
            "audit_year": audit_year,
            "target_final_year": target_year,
            "audit_reference": audit_reference,
            "priority": row.get("priority", ""),
            "manuscript_section": row.get(
                "manuscript_section",
                "",
            ),
            "metadata": meta,
            "status": status,
        })

        if status in {
            "VERIFIED",
            "VERIFIED_OVERRIDE",
        }:
            key = bib_key(
                meta,
                target_year,
                used_keys,
            )

            bib_records.append(
                bib_entry(
                    key,
                    meta,
                    target_year,
                )
            )

        time.sleep(
            PAUSE_SECONDS
        )

    with OUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audit_id",
                "audit_year",
                "target_final_year",
                "doi",
                "provider",
                "metadata_year",
                "title_similarity",
                "status",
                "metadata_title",
                "metadata_venue",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    OUT_JSON.write_text(
        json.dumps(
            metadata_records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    OUT_BIB.write_text(
        "\n\n".join(
            bib_records
        )
        + (
            "\n"
            if bib_records
            else ""
        ),
        encoding="utf-8",
    )

    counts: dict[str, int] = {}

    for row in results:
        counts[row["status"]] = (
            counts.get(row["status"], 0)
            + 1
        )

    unresolved = [
        row
        for row in results
        if row["status"]
        not in {
            "VERIFIED",
            "VERIFIED_OVERRIDE",
        }
    ]

    audit_lines = [
        "# SCALE-Former Bibliography Verification Audit V2",
        "",
        f"Input recent references: {len(results)}",
        "",
        "## Resolver order",
        "",
        "1. OpenAlex",
        "2. DOI content negotiation (CSL JSON)",
        "3. Crossref",
        "",
        "## Final-year corrections",
        "",
        (
            "- Audit #49: 2026 final issue year "
            "(IEEE Wireless Communications Letters, Vol. 15, pp. 880-884)."
        ),
        (
            "- Audit #50: 2024 final issue year "
            "(IEEE Wireless Communications Letters, Vol. 13, No. 12, pp. 3320-3324)."
        ),
        "",
        "## Status counts",
        "",
    ]

    for status in sorted(counts):
        audit_lines.append(
            f"- {status}: {counts[status]}"
        )

    audit_lines.extend([
        "",
        "## Unresolved records",
        "",
    ])

    if not unresolved:
        audit_lines.append(
            "- None."
        )
    else:
        for row in unresolved:
            audit_lines.append(
                f'- Audit #{row["audit_id"]}: '
                f'{row["status"]} — {row["doi"]} — '
                f'{row["note"]}'
            )

    resolved_count = (
        counts.get("VERIFIED", 0)
        + counts.get("VERIFIED_OVERRIDE", 0)
    )

    audit_lines.extend([
        "",
        f"Resolved recent references: {resolved_count}/{len(results)}",
        "",
        (
            "FINAL STATUS: "
            + (
                "ALL RECENT REFERENCES RESOLVED"
                if resolved_count == len(results)
                else "MANUAL REVIEW STILL REQUIRED"
            )
        ),
    ])

    OUT_AUDIT.write_text(
        "\n".join(audit_lines)
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Status counts:")

    for status in sorted(counts):
        print(
            f"  {status}: {counts[status]}"
        )

    print()
    print(
        f"Resolved: {resolved_count}/{len(results)}"
    )
    print(f"Created: {OUT_CSV}")
    print(f"Created: {OUT_JSON}")
    print(f"Created: {OUT_BIB}")
    print(f"Created: {OUT_CORRECTIONS}")
    print(f"Created: {OUT_AUDIT}")
    print()

    if resolved_count == len(results):
        print(
            "FINAL STATUS: ALL RECENT REFERENCES RESOLVED"
        )
    else:
        print(
            "FINAL STATUS: MANUAL REVIEW STILL REQUIRED"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
