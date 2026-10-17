from pathlib import Path

import numpy as np
import tensorflow as tf


DATA_FILE = Path(
    "data/dichasus_005x/dichasus-0055.tfrecords"
)


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

    pos_lidar = tf.ensure_shape(
        tf.io.parse_tensor(
            record["pos-lidar"],
            out_type=tf.float64,
        ),
        (2,),
    )

    rot = tf.ensure_shape(
        record["rot"],
        (),
    )

    snr = tf.ensure_shape(
        tf.io.parse_tensor(
            record["snr"],
            out_type=tf.float32,
        ),
        (32,),
    )

    time = tf.ensure_shape(
        record["time"],
        (),
    )

    return csi, pos_lidar, rot, snr, time


def main():
    print("=" * 72)
    print("DICHASUS-0055 DATASET INSPECTION")
    print("=" * 72)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE.resolve()}"
        )

    file_size_gb = DATA_FILE.stat().st_size / (1024 ** 3)

    print(f"File       : {DATA_FILE}")
    print(f"File size  : {file_size_gb:.3f} GiB")

    raw_dataset = tf.data.TFRecordDataset(
        [str(DATA_FILE)]
    )

    dataset = raw_dataset.map(
        parse_record,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    first_record = next(iter(dataset.take(1)))

    csi, pos_lidar, rot, snr, time = first_record

    csi_np = csi.numpy()
    pos_np = pos_lidar.numpy()
    snr_np = snr.numpy()

    print("\n--- Tensor shapes ---")

    print(
        f"CSI shape       : {csi_np.shape}"
    )
    print(
        f"Position shape  : {pos_np.shape}"
    )
    print(
        f"SNR shape       : {snr_np.shape}"
    )
    print(
        f"Rotation shape  : {rot.shape}"
    )
    print(
        f"Time shape      : {time.shape}"
    )

    assert csi_np.shape == (32, 1024, 2)
    assert pos_np.shape == (2,)
    assert snr_np.shape == (32,)

    print("\n--- Data types ---")

    print(
        f"CSI dtype       : {csi_np.dtype}"
    )
    print(
        f"Position dtype  : {pos_np.dtype}"
    )
    print(
        f"SNR dtype       : {snr_np.dtype}"
    )

    # Reconstruct the complex-valued CSI matrix.
    h_complex = (
        csi_np[..., 0]
        + 1j * csi_np[..., 1]
    )

    print("\n--- Complex CSI reconstruction ---")

    print(
        f"Complex shape   : {h_complex.shape}"
    )
    print(
        f"Complex dtype   : {h_complex.dtype}"
    )

    assert h_complex.shape == (32, 1024)

    # Basic numerical checks.
    finite_csi = np.isfinite(csi_np).all()
    finite_pos = np.isfinite(pos_np).all()
    finite_snr = np.isfinite(snr_np).all()

    print("\n--- Numerical integrity ---")

    print(
        f"CSI finite      : {finite_csi}"
    )
    print(
        f"Position finite : {finite_pos}"
    )
    print(
        f"SNR finite      : {finite_snr}"
    )

    if not finite_csi:
        raise ValueError(
            "Non-finite values detected in CSI."
        )

    # CSI statistics.
    magnitude = np.abs(h_complex)
    power = magnitude ** 2

    print("\n--- CSI statistics ---")

    print(
        f"|H| min         : {magnitude.min():.6e}"
    )
    print(
        f"|H| max         : {magnitude.max():.6e}"
    )
    print(
        f"|H| mean        : {magnitude.mean():.6e}"
    )
    print(
        f"|H| std         : {magnitude.std():.6e}"
    )
    print(
        f"Mean power      : {power.mean():.6e}"
    )

    print("\n--- Metadata ---")

    print(
        f"Position [x,y]  : {pos_np}"
    )
    print(
        f"Rotation [rad]  : {float(rot.numpy()):.6f}"
    )
    print(
        f"Time [s]        : {float(time.numpy()):.6f}"
    )
    print(
        f"SNR min [dB]    : {snr_np.min():.3f}"
    )
    print(
        f"SNR max [dB]    : {snr_np.max():.3f}"
    )
    print(
        f"SNR mean [dB]   : {snr_np.mean():.3f}"
    )

    print("\n--- Example CSI values ---")

    for antenna in range(2):
        for subcarrier in range(3):
            value = h_complex[
                antenna,
                subcarrier,
            ]

            print(
                f"H[{antenna},{subcarrier}] = "
                f"{value.real:.6e} "
                f"+ j{value.imag:.6e}"
            )

    print("\nDataset parsing: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()