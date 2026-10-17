from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

# Suppress TensorFlow informational messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import IterableDataset

from src.data.masking import (
    apply_observation_mask,
    make_frequency_mask,
)


NUM_ANTENNAS = 32
NUM_SUBCARRIERS = 1024


RUN_COUNTS = {
    "0053": 5698,
    "0054": 5180,
    "0055": 5062,
    "0056": 5220,
    "0057": 5001,
}


SPLIT_RUNS = {
    "train": [
        "0053",
        "0054",
        "0055",
    ],
    "val": [
        "0056",
    ],
    "test": [
        "0057",
    ],
}


FEATURE_DESCRIPTION = {
    "csi": tf.io.FixedLenFeature(
        [],
        tf.string,
        default_value="",
    ),
    "pos-lidar": tf.io.FixedLenFeature(
        [],
        tf.string,
        default_value="",
    ),
    "rot": tf.io.FixedLenFeature(
        [],
        tf.float32,
        default_value=0.0,
    ),
    "snr": tf.io.FixedLenFeature(
        [],
        tf.string,
        default_value="",
    ),
    "time": tf.io.FixedLenFeature(
        [],
        tf.float32,
        default_value=0.0,
    ),
}


def _parse_record(proto):
    record = tf.io.parse_single_example(
        proto,
        FEATURE_DESCRIPTION,
    )

    csi = tf.ensure_shape(
        tf.io.parse_tensor(
            record["csi"],
            out_type=tf.float32,
        ),
        (
            NUM_ANTENNAS,
            NUM_SUBCARRIERS,
            2,
        ),
    )

    position = tf.ensure_shape(
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
        (NUM_ANTENNAS,),
    )

    rotation = tf.ensure_shape(
        record["rot"],
        (),
    )

    timestamp = tf.ensure_shape(
        record["time"],
        (),
    )

    return (
        csi,
        position,
        snr,
        rotation,
        timestamp,
    )


class CSINormalizer:
    """
    Normalize real and imaginary CSI components using
    statistics estimated exclusively from the training set.
    """

    def __init__(
        self,
        stats_path: str | Path,
    ):
        stats_path = Path(stats_path)

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization file not found: "
                f"{stats_path}"
            )

        with stats_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            stats = json.load(file)

        self.mean_real = float(
            stats["real"]["mean"]
        )
        self.std_real = float(
            stats["real"]["std"]
        )

        self.mean_imag = float(
            stats["imag"]["mean"]
        )
        self.std_imag = float(
            stats["imag"]["std"]
        )

        if self.std_real <= 0:
            raise ValueError(
                "Real-component standard deviation "
                "must be positive."
            )

        if self.std_imag <= 0:
            raise ValueError(
                "Imaginary-component standard deviation "
                "must be positive."
            )

    def normalize(
        self,
        csi: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        csi:
            Array of shape [B, F, 2].

        Returns
        -------
        normalized:
            Float32 array of shape [2, B, F].
        """
        if csi.shape != (
            NUM_ANTENNAS,
            NUM_SUBCARRIERS,
            2,
        ):
            raise ValueError(
                f"Unexpected CSI shape: {csi.shape}"
            )

        normalized = np.empty(
            (
                2,
                NUM_ANTENNAS,
                NUM_SUBCARRIERS,
            ),
            dtype=np.float32,
        )

        normalized[0] = (
            csi[..., 0]
            - self.mean_real
        ) / self.std_real

        normalized[1] = (
            csi[..., 1]
            - self.mean_imag
        ) / self.std_imag

        return normalized

    def denormalize_tensor(
        self,
        csi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert normalized PyTorch CSI back to the
        original measured scale.

        Supports [2, B, F] and [N, 2, B, F].
        """
        output = csi.clone()

        if output.ndim == 3:
            if output.shape[0] != 2:
                raise ValueError(
                    "Expected shape [2, B, F]."
                )

            output[0] = (
                output[0] * self.std_real
                + self.mean_real
            )

            output[1] = (
                output[1] * self.std_imag
                + self.mean_imag
            )

            return output

        if output.ndim == 4:
            if output.shape[1] != 2:
                raise ValueError(
                    "Expected shape [N, 2, B, F]."
                )

            output[:, 0] = (
                output[:, 0] * self.std_real
                + self.mean_real
            )

            output[:, 1] = (
                output[:, 1] * self.std_imag
                + self.mean_imag
            )

            return output

        raise ValueError(
            "Unsupported CSI tensor shape."
        )


class DichasusDataset(IterableDataset):
    """
    Streaming DICHASUS-005x dataset.

    The raw TFRecord files remain on disk and records
    are decoded one at a time.

    The dataset returns normalized dense reference CSI,
    sparse observed CSI, the observation mask, and
    measurement metadata.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        normalization_path: str | Path,
        density: float = 0.10,
        pattern: str = "comb",
        seed: int = 0,
        shuffle_buffer: int = 512,
    ):
        super().__init__()

        if split not in SPLIT_RUNS:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Expected one of "
                f"{list(SPLIT_RUNS)}."
            )

        self.data_root = Path(
            data_root
        )

        self.split = split
        self.density = float(
            density
        )
        self.pattern = pattern
        self.seed = int(
            seed
        )
        self.shuffle_buffer = int(
            shuffle_buffer
        )

        self.normalizer = CSINormalizer(
            normalization_path
        )

        self.runs = SPLIT_RUNS[
            split
        ]

        self.files = [
            self.data_root
            / f"dichasus-{run}.tfrecords"
            for run in self.runs
        ]

        for file_path in self.files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing DICHASUS file: "
                    f"{file_path}"
                )

        self.epoch = 0

        # Comb mask is identical for every sample.
        # Random/block masks are generated per sample.
        self.fixed_mask = None

        if self.pattern == "comb":
            self.fixed_mask = (
                make_frequency_mask(
                    num_antennas=NUM_ANTENNAS,
                    num_subcarriers=NUM_SUBCARRIERS,
                    density=self.density,
                    pattern="comb",
                    seed=self.seed,
                )
            )

    def __len__(self):
        return sum(
            RUN_COUNTS[run]
            for run in self.runs
        )

    def set_epoch(
        self,
        epoch: int,
    ):
        self.epoch = int(
            epoch
        )

    def _make_tf_dataset(self):
        dataset = tf.data.TFRecordDataset(
            [
                str(file_path)
                for file_path in self.files
            ],
            num_parallel_reads=1,
        )

        if self.split == "train":
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=(
                    self.seed
                    + self.epoch
                ),
                reshuffle_each_iteration=False,
            )

        dataset = dataset.map(
            _parse_record,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset

    def __iter__(
        self,
    ) -> Iterator[dict]:
        dataset = self._make_tf_dataset()

        sample_index = 0

        for (
            csi,
            position,
            snr,
            rotation,
            timestamp,
        ) in dataset:

            csi_np = csi.numpy()

            normalized_np = (
                self.normalizer.normalize(
                    csi_np
                )
            )

            target = torch.from_numpy(
                normalized_np
            )

            if self.fixed_mask is not None:
                mask = self.fixed_mask

            else:
                # Deterministic per-sample mask.
                #
                # Validation/test remain identical
                # across repeated evaluations.
                #
                # Training masks can change when
                # set_epoch() changes the epoch.
                mask_seed = (
                    self.seed
                    + sample_index
                )

                if self.split == "train":
                    mask_seed += (
                        self.epoch
                        * 1_000_003
                    )

                mask = make_frequency_mask(
                    num_antennas=NUM_ANTENNAS,
                    num_subcarriers=NUM_SUBCARRIERS,
                    density=self.density,
                    pattern=self.pattern,
                    seed=mask_seed,
                )

            observed = (
                apply_observation_mask(
                    target,
                    mask,
                )
            )

            yield {
                "observed": observed,
                "target": target,
                "mask": mask.unsqueeze(0),
                "position": torch.from_numpy(
                    position.numpy().astype(
                        np.float32
                    )
                ),
                "snr": torch.from_numpy(
                    snr.numpy()
                ),
                "rotation": torch.tensor(
                    float(
                        rotation.numpy()
                    ),
                    dtype=torch.float32,
                ),
                "time": torch.tensor(
                    float(
                        timestamp.numpy()
                    ),
                    dtype=torch.float32,
                ),
                "sample_index": torch.tensor(
                    sample_index,
                    dtype=torch.long,
                ),
            }

            sample_index += 1