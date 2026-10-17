from pathlib import Path
import json
import time

import numpy as np
import tensorflow as tf


TRAIN_FILES = [
    Path("data/dichasus_005x/dichasus-0053.tfrecords"),
    Path("data/dichasus_005x/dichasus-0054.tfrecords"),
    Path("data/dichasus_005x/dichasus-0055.tfrecords"),
]

OUTPUT_FILE = Path("results/csv/training_normalization.json")


FEATURE_DESCRIPTION = {
    "csi": tf.io.FixedLenFeature(
        [],
        tf.string,
        default_value="",
    ),
}


def parse_csi(proto):
    record = tf.io.parse_single_example(
        proto,
        FEATURE_DESCRIPTION,
    )

    csi = tf.io.parse_tensor(
        record["csi"],
        out_type=tf.float32,
    )

    csi = tf.ensure_shape(
        csi,
        (32, 1024, 2),
    )

    return csi


def main():
    print("=" * 72)
    print("SCALE-FORMER TRAINING-SET NORMALIZATION")
    print("=" * 72)

    for file_path in TRAIN_FILES:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing training file: {file_path}"
            )

    dataset = tf.data.TFRecordDataset(
        [str(path) for path in TRAIN_FILES],
        num_parallel_reads=1,
    )

    dataset = dataset.map(
        parse_csi,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    count_records = 0
    count_values = 0

    sum_real = 0.0
    sum_imag = 0.0

    sumsq_real = 0.0
    sumsq_imag = 0.0

    min_real = np.inf
    max_real = -np.inf

    min_imag = np.inf
    max_imag = -np.inf

    start_time = time.time()

    for csi in dataset:
        x = csi.numpy()

        real = x[..., 0].astype(
            np.float64,
            copy=False,
        )

        imag = x[..., 1].astype(
            np.float64,
            copy=False,
        )

        sum_real += real.sum()
        sum_imag += imag.sum()

        sumsq_real += np.square(real).sum()
        sumsq_imag += np.square(imag).sum()

        min_real = min(
            min_real,
            float(real.min()),
        )

        max_real = max(
            max_real,
            float(real.max()),
        )

        min_imag = min(
            min_imag,
            float(imag.min()),
        )

        max_imag = max(
            max_imag,
            float(imag.max()),
        )

        count_records += 1
        count_values += real.size

        if count_records % 1000 == 0:
            print(
                f"Processed {count_records:,} "
                f"training measurements..."
            )

    mean_real = sum_real / count_values
    mean_imag = sum_imag / count_values

    variance_real = (
        sumsq_real / count_values
        - mean_real ** 2
    )

    variance_imag = (
        sumsq_imag / count_values
        - mean_imag ** 2
    )

    variance_real = max(
        variance_real,
        0.0,
    )

    variance_imag = max(
        variance_imag,
        0.0,
    )

    std_real = np.sqrt(
        variance_real
    )

    std_imag = np.sqrt(
        variance_imag
    )

    elapsed = time.time() - start_time

    stats = {
        "training_runs": [
            "0053",
            "0054",
            "0055",
        ],
        "num_records": count_records,
        "values_per_component": count_values,
        "real": {
            "mean": float(mean_real),
            "std": float(std_real),
            "min": float(min_real),
            "max": float(max_real),
        },
        "imag": {
            "mean": float(mean_imag),
            "std": float(std_imag),
            "min": float(min_imag),
            "max": float(max_imag),
        },
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            stats,
            file,
            indent=4,
        )

    print("\n--- Training statistics ---")

    print(
        f"Records        : "
        f"{count_records:,}"
    )

    print(
        f"Real mean      : "
        f"{mean_real:.8f}"
    )

    print(
        f"Real std       : "
        f"{std_real:.8f}"
    )

    print(
        f"Imag mean      : "
        f"{mean_imag:.8f}"
    )

    print(
        f"Imag std       : "
        f"{std_imag:.8f}"
    )

    print(
        f"Real range     : "
        f"[{min_real:.6f}, "
        f"{max_real:.6f}]"
    )

    print(
        f"Imag range     : "
        f"[{min_imag:.6f}, "
        f"{max_imag:.6f}]"
    )

    print(
        f"Runtime         : "
        f"{elapsed:.2f} s"
    )

    print(
        f"\nSaved statistics to:"
        f"\n{OUTPUT_FILE}"
    )

    print("\nTRAINING-ONLY NORMALIZATION: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()