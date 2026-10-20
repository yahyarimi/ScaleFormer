from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"

RECENT_BIB = PAPER_DIR / "references_recent_final.bib"
RECENT_USAGE = PAPER_DIR / "reference_usage_map_final.csv"

OUT_BIB = PAPER_DIR / "references.bib"
OUT_AUDIT = PAPER_DIR / "bibliography_final_audit.md"
OUT_USAGE = PAPER_DIR / "reference_usage_map_complete.csv"


FOUNDATIONAL = [
    {
        "key": "vaswani2017attention",
        "year": 2017,
        "title": "Attention Is All You Need",
        "authors": (
            "Ashish Vaswani and Noam Shazeer and Niki Parmar and "
            "Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and "
            "{\\L}ukasz Kaiser and Illia Polosukhin"
        ),
        "venue_field": "booktitle",
        "venue": "Advances in Neural Information Processing Systems",
        "volume": "30",
        "pages": "5998--6008",
        "doi": "",
        "type": "inproceedings",
        "role": "Transformer foundation",
        "section": "Related Work: Attention/Sequence Models",
    },
    {
        "key": "ye2018deepofdm",
        "year": 2018,
        "title": (
            "Power of Deep Learning for Channel Estimation and "
            "Signal Detection in {OFDM} Systems"
        ),
        "authors": "Hao Ye and Geoffrey Ye Li and Biing-Hwang Juang",
        "venue_field": "journal",
        "venue": "IEEE Wireless Communications Letters",
        "volume": "7",
        "number": "1",
        "pages": "114--117",
        "doi": "10.1109/LWC.2017.2757490",
        "type": "article",
        "role": "Foundational deep-learning OFDM channel estimation",
        "section": "Introduction; Related Work: Channel Estimation",
    },
    {
        "key": "wen2018csinet",
        "year": 2018,
        "title": "Deep Learning for Massive {MIMO} {CSI} Feedback",
        "authors": "Chao-Kai Wen and Wan-Ting Shih and Shi Jin",
        "venue_field": "journal",
        "venue": "IEEE Wireless Communications Letters",
        "volume": "7",
        "number": "5",
        "pages": "748--751",
        "doi": "10.1109/LWC.2018.2818160",
        "type": "article",
        "role": "Foundational learned CSI feedback/reconstruction",
        "section": "Introduction; Related Work: CSI Reconstruction/Feedback",
    },
    {
        "key": "yang2019deepcmc",
        "year": 2019,
        "title": (
            "Deep Convolutional Compression for Massive {MIMO} "
            "{CSI} Feedback"
        ),
        "authors": (
            "Qianqian Yang and Mahdi Boloursaz Mashhadi and "
            "Deniz G{\\\"u}nd{\\\"u}z"
        ),
        "venue_field": "booktitle",
        "venue": (
            "2019 IEEE 29th International Workshop on Machine "
            "Learning for Signal Processing (MLSP)"
        ),
        "pages": "1--6",
        "doi": "10.1109/MLSP.2019.8918798",
        "type": "inproceedings",
        "role": "Foundational CNN-based CSI compression",
        "section": "Related Work: CSI Reconstruction/Feedback",
    },
    {
        "key": "mashhadi2020survey",
        "year": 2020,
        "title": (
            "Deep Learning for Massive {MIMO} Channel State "
            "Acquisition and Feedback"
        ),
        "authors": (
            "Mahdi Boloursaz Mashhadi and Deniz G{\\\"u}nd{\\\"u}z"
        ),
        "venue_field": "journal",
        "venue": "Journal of the Indian Institute of Science",
        "volume": "100",
        "number": "2",
        "pages": "369--382",
        "doi": "10.1007/s41745-020-00169-2",
        "type": "article",
        "role": "Foundational survey of learned massive-MIMO CSI acquisition",
        "section": "Introduction; Related Work",
    },
    {
        "key": "euchner2021dichasus",
        "year": 2021,
        "title": (
            "A Distributed Massive {MIMO} Channel Sounder for "
            "``Big {CSI} Data''-Driven Machine Learning"
        ),
        "authors": (
            "Florian Euchner and Marc Gauger and Sebastian D{\\\"o}rner "
            "and Stephan ten Brink"
        ),
        "venue_field": "booktitle",
        "venue": (
            "WSA 2021; 25th International ITG Workshop on Smart Antennas"
        ),
        "pages": "289--294",
        "doi": "",
        "type": "inproceedings",
        "role": "DICHASUS measurement-system reference",
        "section": "Introduction; Experimental Methodology",
    },
    {
        "key": "euchner2021dichasus005x",
        "year": 2021,
        "title": (
            "{CSI} Dataset dichasus-005x: Indoor Non-Line of Sight, "
            "Single Room"
        ),
        "authors": "Florian Euchner and Marc Gauger",
        "venue_field": "howpublished",
        "venue": "DaRUS, Version 1.0",
        "doi": "10.18419/DARUS-2204",
        "type": "misc",
        "role": "Specific measured DICHASUS-005x dataset",
        "section": "Experimental Methodology: Dataset",
    },
]


def foundational_bib_entry(item: dict) -> str:
    lines = [f'@{item["type"]}{{{item["key"]},']
    fields = [
        ("author", item["authors"]),
        ("title", item["title"]),
        (item["venue_field"], item["venue"]),
        ("year", str(item["year"])),
    ]

    for optional in ("volume", "number", "pages", "doi"):
        value = str(item.get(optional, "")).strip()
        if value:
            fields.append((optional, value))

    for index, (name, value) in enumerate(fields):
        comma = "," if index < len(fields) - 1 else ""
        lines.append(f"  {name:<12} = {{{value}}}{comma}")

    lines.append("}")
    return "\n".join(lines)


def extract_entries(text: str) -> list[str]:
    entries = []
    current = []
    depth = 0
    in_entry = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("@") and not in_entry:
            in_entry = True
            current = [line]
            depth = line.count("{") - line.count("}")
            continue

        if in_entry:
            current.append(line)
            depth += line.count("{") - line.count("}")

            if depth <= 0:
                entries.append("\n".join(current).strip())
                current = []
                in_entry = False

    if in_entry:
        raise RuntimeError("Unclosed BibTeX entry detected.")

    return entries


def entry_key(entry: str) -> str:
    match = re.match(r"@\w+\{([^,]+),", entry)
    if not match:
        raise RuntimeError(f"Could not parse BibTeX key:\n{entry[:120]}")
    return match.group(1).strip()


def entry_doi(entry: str) -> str:
    match = re.search(
        r"(?im)^\s*doi\s*=\s*\{([^}]+)\}",
        entry,
    )
    if not match:
        return ""
    return match.group(1).strip().lower()


def entry_year(entry: str) -> int:
    match = re.search(
        r"(?im)^\s*year\s*=\s*\{(\d{4})\}",
        entry,
    )
    if not match:
        raise RuntimeError(
            f"Missing or invalid year for {entry_key(entry)}"
        )
    return int(match.group(1))


def main() -> None:
    if not RECENT_BIB.exists():
        raise FileNotFoundError(RECENT_BIB)

    recent_text = RECENT_BIB.read_text(encoding="utf-8")
    recent_entries = extract_entries(recent_text)

    if len(recent_entries) != 50:
        raise RuntimeError(
            f"Expected 50 recent BibTeX entries, got {len(recent_entries)}."
        )

    foundational_entries = [
        foundational_bib_entry(item)
        for item in FOUNDATIONAL
    ]

    all_entries = recent_entries + foundational_entries

    keys = [entry_key(entry) for entry in all_entries]
    if len(keys) != len(set(keys)):
        duplicates = [
            key for key, count in Counter(keys).items()
            if count > 1
        ]
        raise RuntimeError(
            "Duplicate BibTeX keys: " + ", ".join(duplicates)
        )

    dois = [
        entry_doi(entry)
        for entry in all_entries
        if entry_doi(entry)
    ]
    if len(dois) != len(set(dois)):
        duplicates = [
            doi for doi, count in Counter(dois).items()
            if count > 1
        ]
        raise RuntimeError(
            "Duplicate DOI(s): " + ", ".join(duplicates)
        )

    years = [entry_year(entry) for entry in all_entries]
    year_counts = Counter(years)

    recent_years = [year for year in years if 2022 <= year <= 2026]
    foundational_years = [year for year in years if year < 2022]

    if len(recent_years) != 50:
        raise RuntimeError(
            f"Expected exactly 50 references from 2022-2026, got {len(recent_years)}."
        )

    if len(foundational_years) != 7:
        raise RuntimeError(
            f"Expected 7 foundational pre-2022 references, got {len(foundational_years)}."
        )

    if len(all_entries) != 57:
        raise RuntimeError(
            f"Expected 57 total references, got {len(all_entries)}."
        )

    OUT_BIB.write_text(
        "\n\n".join(all_entries) + "\n",
        encoding="utf-8",
    )

    # Complete usage map = recent verified map + seven foundational records.
    usage_rows = []
    if RECENT_USAGE.exists():
        with RECENT_USAGE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            usage_rows = list(csv.DictReader(f))

    complete_rows = []

    for row in usage_rows:
        complete_rows.append({
            "bib_key": row.get("bib_key", ""),
            "year": row.get("year", ""),
            "title": row.get("title", ""),
            "venue": row.get("venue", ""),
            "doi": row.get("doi", ""),
            "category": "recent_2022_2026",
            "priority": row.get("priority", ""),
            "manuscript_section": row.get("manuscript_section", ""),
            "verification": row.get("verification", ""),
            "verification_note": row.get("verification_note", ""),
        })

    for item in FOUNDATIONAL:
        complete_rows.append({
            "bib_key": item["key"],
            "year": item["year"],
            "title": re.sub(r"[{}]", "", item["title"]),
            "venue": item["venue"],
            "doi": item.get("doi", ""),
            "category": "foundational_pre_2022",
            "priority": "Foundational",
            "manuscript_section": item["section"],
            "verification": "CURATED_FOUNDATIONAL",
            "verification_note": item["role"],
        })

    with OUT_USAGE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bib_key",
                "year",
                "title",
                "venue",
                "doi",
                "category",
                "priority",
                "manuscript_section",
                "verification",
                "verification_note",
            ],
        )
        writer.writeheader()
        writer.writerows(complete_rows)

    audit = [
        "# SCALE-Former Final Bibliography Audit",
        "",
        "## Counts",
        "",
        "- Recent references (2022-2026): 50",
        "- Foundational references (pre-2022): 7",
        "- **Total bibliography entries: 57**",
        "",
        "## Year distribution",
        "",
    ]

    for year in sorted(year_counts):
        audit.append(f"- {year}: {year_counts[year]}")

    audit += [
        "",
        "## Integrity checks",
        "",
        f"- Unique BibTeX keys: {len(set(keys))}/{len(keys)}",
        f"- DOI-bearing references: {len(dois)}",
        f"- Unique DOI values: {len(set(dois))}/{len(dois)}",
        "- All entries contain a four-digit publication year.",
        "- Exactly 50 references fall within the required 2022-2026 window.",
        "- Exactly 7 references are retained as pre-2022 foundational works.",
        "",
        "## Foundational references",
        "",
    ]

    for item in FOUNDATIONAL:
        audit.append(
            f'- {item["key"]} ({item["year"]}): {item["role"]}.'
        )

    audit += [
        "",
        "## Manuscript guardrails",
        "",
        "- Do not cite references solely to inflate the bibliography.",
        "- Distinguish CSI feedback/compression from sparse measured-CSI reconstruction.",
        "- Do not claim SCALE-Former is the first Transformer, first masked model, or first scalable neural CSI method.",
        "- Use “dense measured reference CSI,” not “physical ground truth channel.”",
        "- Do not describe controlled sparse masks as operational DICHASUS pilot patterns.",
        "- Claim transfer only across the tested B/F configurations without retraining.",
        "- Preserve the block-mask failure, no-global capacity confound, and accuracy-efficiency trade-off.",
        "",
        "FINAL STATUS: 57/57 BIBLIOGRAPHY ENTRIES ASSEMBLED AND INTERNALLY AUDITED",
    ]

    OUT_AUDIT.write_text(
        "\n".join(audit) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("SCALE-FORMER FINAL BIBLIOGRAPHY ASSEMBLY")
    print("=" * 100)
    print(f"Recent references:       {len(recent_entries)}")
    print(f"Foundational references: {len(foundational_entries)}")
    print(f"Total references:        {len(all_entries)}")
    print(f"Unique keys:             {len(set(keys))}")
    print(f"DOI-bearing references:  {len(dois)}")
    print(f"Unique DOI values:       {len(set(dois))}")
    print()
    print(f"Created: {OUT_BIB}")
    print(f"Created: {OUT_USAGE}")
    print(f"Created: {OUT_AUDIT}")
    print()
    print(
        "FINAL STATUS: 57/57 BIBLIOGRAPHY ENTRIES "
        "ASSEMBLED AND INTERNALLY AUDITED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
