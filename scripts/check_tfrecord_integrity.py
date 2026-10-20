from pathlib import Path

import tensorflow as tf


DATA_ROOT = Path(
    "data/dichasus_005x"
)

FILES = {
    "0053": (
        DATA_ROOT
        / "dichasus-0053.tfrecords"
    ),
    "0054": (
        DATA_ROOT
        / "dichasus-0054.tfrecords"
    ),
    "0055": (
        DATA_ROOT
        / "dichasus-0055.tfrecords"
    ),
    "0056": (
        DATA_ROOT
        / "dichasus-0056.tfrecords"
    ),
    "0057": (
        DATA_ROOT
        / "dichasus-0057.tfrecords"
    ),
}

EXPECTED_COUNTS = {
    "0053": 5698,
    "0054": 5180,
    "0055": 5062,
    "0056": 5220,
    "0057": 5001,
}


def check_file(
    run_id: str,
    path: Path,
) -> bool:

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"Checking run {run_id}"
    )

    print(
        f"Path : {path}"
    )

    if not path.exists():
        print(
            "STATUS: FILE MISSING"
        )
        return False

    size_bytes = (
        path.stat().st_size
    )

    print(
        f"Size : "
        f"{size_bytes:,} bytes"
    )

    count = 0

    try:
        dataset = (
            tf.data.TFRecordDataset(
                str(path),
                num_parallel_reads=1,
            )
        )

        for _ in dataset:
            count += 1

            if count % 1000 == 0:
                print(
                    f"  Read "
                    f"{count:,} records..."
                )

    except Exception as error:
        print(
            "\nSTATUS: FAILED"
        )

        print(
            f"Records read before "
            f"failure: {count:,}"
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error:"
            f"\n{error}"
        )

        return False

    expected = (
        EXPECTED_COUNTS[
            run_id
        ]
    )

    print(
        f"\nRecords found    : "
        f"{count:,}"
    )

    print(
        f"Expected records : "
        f"{expected:,}"
    )

    if count != expected:
        print(
            "STATUS: COUNT MISMATCH"
        )
        return False

    print(
        "STATUS: PASSED"
    )

    return True


def main():

    print("=" * 72)
    print(
        "DICHASUS TFRECORD "
        "INTEGRITY CHECK"
    )
    print("=" * 72)

    results = {}

    for run_id, path in (
        FILES.items()
    ):
        results[
            run_id
        ] = check_file(
            run_id,
            path,
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 72
    )

    for run_id, passed in (
        results.items()
    ):
        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"{run_id}: {status}"
        )

    if all(
        results.values()
    ):
        print(
            "\nALL TFRECORD FILES PASSED"
        )
    else:
        print(
            "\nONE OR MORE TFRECORD "
            "FILES FAILED"
        )


if __name__ == "__main__":
    main()