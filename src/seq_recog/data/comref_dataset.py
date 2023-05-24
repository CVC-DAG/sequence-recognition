"""Implements loading of COMREF samples."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from torch.utils import data as D

from seq_recog.data.base_dataset import (
    BaseDataset,
    BaseDataConfig,
    BaseVocab,
    DatasetSample,
)


def load_comref_splits(
    splits_file: Path,
    vocab: BaseVocab,
    config: BaseDataConfig,
) -> ComrefDataset:
    with open(splits_file, "r") as f_in:
        splits = json.load(f_in)
    root_path = splits_file.parent

    full_dataset = []

    for split in ["train", "valid", "test"]:
        datasets = []
        for work in splits[split]:
            gt_file = root_path / work / f"{work}.mtn"
            img_folder = root_path / work / "measures"

            datasets.append(
                ComrefDataset(
                    str(img_folder),
                    str(gt_file),
                    vocab,
                    config,
                    split == "train",
                )
            )
        full_dataset.append(D.ConcatDataset(datasets))

    train_dataset, valid_dataset, test_dataset = full_dataset
    return train_dataset, valid_dataset, test_dataset


class ComrefDataset(BaseDataset):
    """Load COMREF samples for inference."""

    RE_PROPERTIES = re.compile(r"\[[^]]*\]")
    RE_SEPARATOR = re.compile(r",")

    def __init__(
        self,
        image_folder: str,
        dataset_file: str,
        vocab: BaseVocab,
        config: BaseDataConfig,
        train: bool = True,
    ) -> None:
        super().__init__(image_folder, dataset_file, vocab, config, train)

    def _load_data(self) -> None:
        with open(self._dataset_file, "r") as f_gt:
            gt = json.load(f_gt)

        for fn, sample in gt.items():
            transcript = self.RE_PROPERTIES.sub("", sample)
            transcript = self.RE_SEPARATOR.split(transcript)

            gt_len = len(transcript)
            transcript = self._vocab.prepare_data(transcript, self._seqlen)

            self._samples.append(
                DatasetSample(
                    gt=transcript,
                    gt_len=gt_len,
                    segm=np.array([]),
                    fname=str(self._image_folder / fn),
                )
            )
