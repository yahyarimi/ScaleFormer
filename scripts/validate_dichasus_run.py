from pathlib import Path
import time

import numpy as np
import tensorflow as tf


DATA_FILE = Path(
    "data/dichasus_005x/dichasus-0055.tfrecords"
)

EXPECTED_RECORDS = 5062


FEATURE_DESCRIPTION = {
    "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "pos-lidar": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "rot": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    "snr": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "time": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}


def parse_record(proto):
    record = tf.io.parse_single_example(
        proto,
        FEATURE_DESCRIPTION,
    )

    csi = tf.ensure_shape(
        tf.io.parse_tensor(
            record["csi"],
            out_type=tf.float32,
        ),
        (32, 1024, 2),
    )

    pos = tf.ensure_shape(
        tf.io.parse_tensor(
            record["pos-lidar"],
            out_type=tf.float64,
        ),
        (2,),
    )

    snr = tf.ensure_shape(
        tf.io.parse_tensor(
            record["snr"],
            out_type=tf.float32,
        ),
        (32,),
    )

    rot = tf.ensure_shape(
        record["rot"],
        (),
    )

    timestamp = tf.ensure_shape(
        record["time"],
        (),
    )

    return csi, pos, snr, rot, timestamp


def main():
    print("=" * 72)
    print("FULL DICHASUS RUN VALIDATION")
    print("=" * 72)

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    dataset = tf.data.TFRecordDataset(
        [str(DATA_FILE)],
        num_parallel_reads=1,
    )

    dataset = dataset.map(
        parse_record,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    count = 0
    invalid_csi = 0
    invalid_pos = 0
    invalid_snr = 0

    first_stored_time = None
    last_stored_time = None

    min_time = np.inf
    max_time = -np.inf

    previous_time = None
    time_decreases = 0
    time_increases = 0
    time_equal = 0

    snr_min = np.inf
    snr_max = -np.inf
    snr_sum = 0.0
    snr_elements = 0

    magnitude_min = np.inf
    magnitude_max = -np.inf
    magnitude_sum = 0.0
    magnitude_elements = 0

    start = time.time()

    for csi, pos, snr, rot, timestamp in dataset:
        csi_np = csi.numpy()
        pos_np = pos.numpy()
        snr_np = snr.numpy()
        time_value = float(timestamp.numpy())

        if csi_np.shape != (32, 1024, 2):
            raise ValueError(
                f"Unexpected CSI shape at record {count}: "
                f"{csi_np.shape}"
            )

        if pos_np.shape != (2,):
            raise ValueError(
                f"Unexpected position shape at record {count}: "
                f"{pos_np.shape}"
            )

        if snr_np.shape != (32,):
            raise ValueError(
                f"Unexpected SNR shape at record {count}: "
                f"{snr_np.shape}"
            )

        if not np.isfinite(csi_np).all():
            invalid_csi += 1

        if not np.isfinite(pos_np).all():
            invalid_pos += 1

        if not np.isfinite(snr_np).all():
            invalid_snr += 1

        h = (
            csi_np[..., 0]
            + 1j * csi_np[..., 1]
        )

        magnitude = np.abs(h)

        magnitude_min = min(
            magnitude_min,
            float(magnitude.min()),
        )

        magnitude_max = max(
            magnitude_max,
            float(magnitude.max()),
        )

        magnitude_sum += float(
            magnitude.sum(dtype=np.float64)
        )

        magnitude_elements += magnitude.size

        snr_min = min(
            snr_min,
            float(snr_np.min()),
        )

        snr_max = max(
            snr_max,
            float(snr_np.max()),
        )

        snr_sum += float(
            snr_np.sum(dtype=np.float64)
        )

        snr_elements += snr_np.size

        if first_stored_time is None:
            first_stored_time = time_value

        last_stored_time = time_value

        min_time = min(min_time, time_value)
        max_time = max(max_time, time_value)

        if previous_time is not None:
            if time_value < previous_time:
                time_decreases += 1
            elif time_value > previous_time:
                time_increases += 1
            else:
                time_equal += 1

        previous_time = time_value

        count += 1

        if count % 500 == 0:
            print(
                f"Validated {count} records..."
            )

    elapsed = time.time() - start

    print("\n--- Record integrity ---")

    print(f"Records found       : {count}")
    print(f"Records expected    : {EXPECTED_RECORDS}")

    print(
        f"Invalid CSI records : {invalid_csi}"
    )

    print(
        f"Invalid positions   : {invalid_pos}"
    )

    print(
        f"Invalid SNR records : {invalid_snr}"
    )

    print("\n--- Channel statistics ---")

    print(
        f"|H| global min      : "
        f"{magnitude_min:.6e}"
    )

    print(
        f"|H| global max      : "
        f"{magnitude_max:.6e}"
    )

    print(
        f"|H| global mean     : "
        f"{magnitude_sum / magnitude_elements:.6e}"
    )

    print("\n--- SNR statistics ---")

    print(
        f"SNR global min      : "
        f"{snr_min:.3f} dB"
    )

    print(
        f"SNR global max      : "
        f"{snr_max:.3f} dB"
    )

    print(
        f"SNR global mean     : "
        f"{snr_sum / snr_elements:.3f} dB"
    )

    print("\n--- Time metadata ---")

    print(
        f"First stored time    : "
        f"{first_stored_time:.6f}"
    )

    print(
        f"Last stored time     : "
        f"{last_stored_time:.6f}"
    )

    print(
        f"Minimum timestamp    : "
        f"{min_time:.6f}"
    )

    print(
        f"Maximum timestamp    : "
        f"{max_time:.6f}"
    )

    print(
        f"True timestamp span  : "
        f"{max_time - min_time:.3f} s"
    )

    print(
        f"Time increases       : "
        f"{time_increases}"
    )

    print(
        f"Time decreases       : "
        f"{time_decreases}"
    )

    print(
        f"Equal timestamps     : "
        f"{time_equal}"
    )

    if time_decreases > 0:
        print(
            "Record order        : NOT chronological"
        )
    else:
        print(
            "Record order        : Chronological"
        )

    print("\n--- Performance ---")

    print(
        f"Validation time      : "
        f"{elapsed:.2f} s"
    )

    print(
        f"Records / second     : "
        f"{count / elapsed:.2f}"
    )

    assert count == EXPECTED_RECORDS, (
        f"Expected {EXPECTED_RECORDS} records, "
        f"found {count}."
    )

    assert invalid_csi == 0
    assert invalid_pos == 0
    assert invalid_snr == 0

    print("\nFULL RUN VALIDATION: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()