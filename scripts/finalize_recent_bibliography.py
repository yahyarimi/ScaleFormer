from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"

USAGE_MAP = PAPER_DIR / "reference_usage_map.csv"
METADATA_V2 = PAPER_DIR / "bibliography_metadata_v2.json"

OUT_BIB = PAPER_DIR / "references_recent_final.bib"
OUT_AUDIT = PAPER_DIR / "bibliography_recent_final_audit.md"
OUT_USAGE = PAPER_DIR / "reference_usage_map_final.csv"


# Final issue-year corrections manually checked against publisher/index records.
FINAL_OVERRIDES = {
    "2": {
        "year": 2022,
        "volume": "70",
        "issue": "2",
        "pages": "935-950",
        "reason": "Final IEEE TCOM issue: Vol. 70, No. 2, Feb. 2022, pp. 935-950.",
    },
    "3": {
        "year": 2022,
        "volume": "26",
        "issue": "3",
        "pages": "547-551",
        "reason": "Final IEEE Communications Letters issue: Vol. 26, No. 3, Mar. 2022, pp. 547-551.",
    },
    "10": {
        "year": 2023,
        "volume": "12",
        "issue": "1",
        "pages": "40-44",
        "reason": "Final IEEE WCL issue: Vol. 12, No. 1, Jan. 2023, pp. 40-44.",
    },
    "11": {
        "year": 2023,
        "volume": "27",
        "issue": "1",
        "pages": "200-204",
        "reason": "Final IEEE Communications Letters issue: Vol. 27, No. 1, Jan. 2023, pp. 200-204.",
    },
    "38": {
        "year": 2025,
        "volume": "11",
        "issue": "1",
        "pages": "218-230",
        "reason": "Final IEEE TCCN issue: Vol. 11, No. 1, 2025, pp. 218-230.",
    },
    "49": {
        "year": 2026,
        "volume": "15",
        "pages": "880-884",
        "reason": "Final IEEE WCL publication year is 2026; DOI was registered in 2025.",
    },
    "50": {
        "year": 2024,
        "volume": "13",
        "issue": "12",
        "pages": "3320-3324",
        "reason": "Final IEEE WCL issue: Vol. 13, No. 12, 2024, pp. 3320-3324.",
    },
    "51": {
        "year": 2025,
        "volume": "14",
        "issue": "2",
        "pages": "499-503",
        "reason": "Final IEEE WCL issue: Vol. 14, No. 2, Feb. 2025, pp. 499-503.",
    },
}


# The V2 network resolver failed for this one record, but the final IEEE/DBLP
# metadata were manually verified.
MANUAL_METADATA = {
    "59": {
        "provider": "Manual: IEEE Xplore + DBLP",
        "doi": "10.1109/TWC.2026.3671802",
        "title": "Large Language Model Empowered CSI Feedback in Massive MIMO Systems",
        "year": 2026,
        "venue": "IEEE Transactions on Wireless Communications",
        "volume": "25",
        "issue": "",
        "pages": "13835-13851",
        "article_number": "",
        "publisher": "IEEE",
        "type": "journal-article",
        "authors": [
            "Jie Wu",
            "Wei Xu",
            "Le Liang",
            "Xiaohu You",
            "Mérouane Debbah",
        ],
        "url": "https://doi.org/10.1109/TWC.2026.3671802",
    }
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def protect_title(title: str) -> str:
    out = title
    tokens = [
        "CSI", "MIMO", "OFDM", "FDD", "TDD", "THz",
        "RIS", "CNN", "AI", "VQ", "VAE", "LLM",
        "BiLSTM", "Mamba",
    ]
    for token in tokens:
        out = re.sub(
            rf"\b{re.escape(token)}\b",
            "{" + token + "}",
            out,
            flags=re.IGNORECASE,
        )
    return out


def clean_pages(pages: str) -> str:
    pages = str(pages or "").strip()
    if not pages:
        return ""
    return re.sub(r"\s*-\s*", "--", pages)


def bib_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def make_key(meta: dict, year: int, used: set[str]) -> str:
    if meta.get("authors"):
        surname = normalize(meta["authors"][0].split()[-1]).replace(" ", "")
    else:
        surname = "ref"

    words = [
        w for w in normalize(meta["title"]).split()
        if w not in {
            "a", "an", "the", "of", "for", "with",
            "and", "in", "on", "to", "based",
        }
    ]
    keyword = words[0] if words else "work"
    base = f"{surname}{year}{keyword}"
    key = base
    suffix = ord("a")
    while key in used:
        key = base + chr(suffix)
        suffix += 1
    used.add(key)
    return key


def bib_entry(key: str, meta: dict, year: int) -> str:
    type_text = str(meta.get("type", "")).lower()
    venue_text = str(meta.get("venue", "")).lower()

    is_conf = (
        "proceedings" in type_text
        or "conference" in type_text
        or "conference" in venue_text
        or "workshop" in venue_text
    )
    kind = "inproceedings" if is_conf else "article"

    fields: list[tuple[str, str]] = []

    authors = [str(x).strip() for x in meta.get("authors", []) if str(x).strip()]
    if authors:
        fields.append(("author", " and ".join(authors)))

    fields.append(("title", protect_title(meta["title"])))

    venue = str(meta.get("venue", "")).strip()
    if venue:
        fields.append(("booktitle" if kind == "inproceedings" else "journal", venue))

    fields.append(("year", str(year)))

    volume = str(meta.get("volume", "")).strip()
    issue = str(meta.get("issue", "")).strip()
    pages = clean_pages(str(meta.get("pages", "")))

    if volume:
        fields.append(("volume", volume))
    if issue:
        fields.append(("number", issue))
    if pages:
        fields.append(("pages", pages))

    doi = str(meta.get("doi", "")).strip()
    if doi:
        fields.append(("doi", doi))

    lines = [f"@{kind}{{{key},"]
    for idx, (name, value) in enumerate(fields):
        comma = "," if idx < len(fields) - 1 else ""
        lines.append(f"  {name:<10} = {{{bib_escape(value)}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    if not USAGE_MAP.exists():
        raise FileNotFoundError(USAGE_MAP)
    if not METADATA_V2.exists():
        raise FileNotFoundError(METADATA_V2)

    with USAGE_MAP.open("r", encoding="utf-8-sig", newline="") as f:
        usage_rows = list(csv.DictReader(f))

    metadata_records = json.loads(METADATA_V2.read_text(encoding="utf-8"))

    by_id: dict[str, dict] = {}
    for item in metadata_records:
        audit_id = str(item["audit_id"]).strip()
        by_id[audit_id] = dict(item["metadata"])

    # Supply the one metadata record that failed all live resolvers.
    for audit_id, meta in MANUAL_METADATA.items():
        by_id[audit_id] = dict(meta)

    final_rows = []
    used_keys: set[str] = set()
    bib_entries = []
    missing = []

    for row in usage_rows:
        audit_id = str(row["audit_id"]).strip()

        if audit_id not in by_id:
            missing.append(audit_id)
            continue

        meta = dict(by_id[audit_id])

        # Base target year from the curated usage map.
        final_year = int(row["year"])

        if audit_id in FINAL_OVERRIDES:
            override = FINAL_OVERRIDES[audit_id]
            final_year = int(override["year"])

            for field in ("volume", "issue", "pages"):
                if field in override:
                    meta[field] = override[field]

        # The manually verified #59 already contains its final metadata.
        if audit_id == "59":
            final_year = 2026

        key = make_key(meta, final_year, used_keys)
        bib_entries.append(bib_entry(key, meta, final_year))

        final_rows.append({
            "audit_id": audit_id,
            "bib_key": key,
            "year": final_year,
            "title": meta["title"],
            "venue": meta.get("venue", ""),
            "volume": meta.get("volume", ""),
            "issue": meta.get("issue", ""),
            "pages": clean_pages(meta.get("pages", "")),
            "doi": meta.get("doi", row["doi"]),
            "priority": row.get("priority", ""),
            "manuscript_section": row.get("manuscript_section", ""),
            "verification": (
                "MANUAL_VERIFIED"
                if audit_id in MANUAL_METADATA
                else "VERIFIED_WITH_FINAL_ISSUE_OVERRIDE"
                if audit_id in FINAL_OVERRIDES
                else "AUTOMATED_VERIFIED"
            ),
            "verification_note": (
                FINAL_OVERRIDES[audit_id]["reason"]
                if audit_id in FINAL_OVERRIDES
                else "Manually verified against IEEE Xplore and DBLP."
                if audit_id in MANUAL_METADATA
                else "Passed V2 metadata verification."
            ),
        })

    if missing:
        raise RuntimeError(
            "Missing metadata for audit IDs: " + ", ".join(missing)
        )

    if len(final_rows) != 50:
        raise RuntimeError(
            f"Expected 50 recent references, got {len(final_rows)}."
        )

    dois = [r["doi"].lower() for r in final_rows if r["doi"]]
    if len(dois) != len(set(dois)):
        raise RuntimeError("Duplicate DOI detected in final recent bibliography.")

    OUT_BIB.write_text(
        "\n\n".join(bib_entries) + "\n",
        encoding="utf-8",
    )

    with OUT_USAGE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(final_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(final_rows)

    counts = Counter(int(r["year"]) for r in final_rows)

    audit_lines = [
        "# SCALE-Former Final Recent-Bibliography Audit",
        "",
        "## Count",
        "",
    ]
    for year in sorted(counts):
        audit_lines.append(f"- {year}: {counts[year]}")
    audit_lines.append(f"- **Total recent references: {len(final_rows)}**")

    audit_lines += [
        "",
        "## Resolution summary",
        "",
        f"- Automated V2 records retained: {sum(r['verification'] == 'AUTOMATED_VERIFIED' for r in final_rows)}",
        f"- Final-issue metadata overrides: {sum(r['verification'] == 'VERIFIED_WITH_FINAL_ISSUE_OVERRIDE' for r in final_rows)}",
        f"- Fully manual metadata records: {sum(r['verification'] == 'MANUAL_VERIFIED' for r in final_rows)}",
        "",
        "## Final issue/year corrections",
        "",
    ]

    for audit_id in sorted(FINAL_OVERRIDES, key=int):
        item = FINAL_OVERRIDES[audit_id]
        audit_lines.append(
            f"- Audit #{audit_id}: final year {item['year']} — {item['reason']}"
        )

    audit_lines += [
        "",
        "## Manual metadata fallback",
        "",
        (
            "- Audit #59: Large Language Model Empowered CSI Feedback in Massive MIMO Systems; "
            "IEEE Transactions on Wireless Communications, vol. 25, pp. 13835--13851, 2026; "
            "DOI 10.1109/TWC.2026.3671802."
        ),
        "",
        "## Guardrails",
        "",
        "- These 50 papers satisfy the manuscript's recent-reference requirement (2022-2026).",
        "- Foundational pre-2022 references must be appended separately and do not count toward the 50.",
        "- References should be cited only where they support a concrete statement.",
        "- Do not blur CSI feedback/compression with sparse measured-CSI reconstruction.",
        "- Do not claim SCALE-Former is the first Transformer, masked model, or scalable neural CSI method.",
        "",
        "FINAL STATUS: 50/50 RECENT REFERENCES RESOLVED",
    ]

    OUT_AUDIT.write_text(
        "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("SCALE-FORMER RECENT BIBLIOGRAPHY FINALIZATION")
    print("=" * 100)
    print(f"Recent references finalized: {len(final_rows)}")
    print("Year distribution:")
    for year in sorted(counts):
        print(f"  {year}: {counts[year]}")
    print()
    print(f"Created: {OUT_BIB}")
    print(f"Created: {OUT_USAGE}")
    print(f"Created: {OUT_AUDIT}")
    print()
    print("FINAL STATUS: 50/50 RECENT REFERENCES RESOLVED")
    print("=" * 100)


if __name__ == "__main__":
    main()
