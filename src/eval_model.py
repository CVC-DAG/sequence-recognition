"""Generate display images for a prediction and evaluate results."""

import json
import pickle
import re

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np

from seq_alignment.data.generic_decrypt import (
    GenericDecryptDataset,
    GenericDecryptVocab
)
from seq_alignment.models.model_zoo import ModelZoo
from seq_alignment.utils.decoding import Prediction, PredictionGroup
from seq_alignment.utils.io import load_pickle_prediction


RE_NUMBER = re.compile("([0-9]+),([0-9]+)")


class AlignmentEvaluator:
    """Perform evaluation of models, ensembles of models and old predictions."""

    def __init__(self, args: Namespace) -> None:
        self._groups = {}
        
        for p in args.pickles:
            self._groups = self._load_pickle_predictions(p, self._groups)

        for p in args.fewshot:
            self._groups = self._load_few_shot_predictions(p, self._groups)

        for p in args.prediction:
            self._groups = self._load_pickle_predictions(p, self._groups)

    def _load_pickle_predictions(
        self,
        path: Path,
        groups: Dict[str, PredictionGroup],
        reverse: bool = False,
    ) -> Dict[str, PredictionGroup]:
        coord_file = path / "results_coords1d.pkl"
        confs_file = path / "results_coords1d_confidences.pkl"

        assert coord_file.exists(), "Coordinate file does not exist"
        assert confs_file.exists(), "Confidence file does not exist"

        coords = load_pickle_prediction(path)
        confs = load_pickle_prediction(path)

        files = list(set(coords.keys()) & set(confs.keys()))
        for k in files:
            file_coords = coords[k]
            file_confs = confs[k]
            gt_seq = np.zeros(len(file_coords))

            pred = Prediction(file_coords, file_confs, gt_seq)
            try:
                groups[k].add_prediction(pred, path.parent.stem)
            except KeyError:
                groups[k] = PredictionGroup([pred], gt_seq, [path.parent.stem])

        return groups

    def _load_few_shot_predictions(
        self,
        path: Path,
        groups: Dict[str, PredictionGroup],
        reverse: bool = False,
    ) -> Dict[str, PredictionGroup]:
        assert (path / "boxes").exists(), "The boxes path does not exist"
        assert (path / "text").exists(), "The text path does not exist"

        bboxes = {}

        for fbbox in (path / "boxes").glob(".txt"):
            with open(fbbox, 'r') as f_in:
                data = f_in.read()

            data = data.split("\n")[:-1]
            data = map(lambda x: x.groups(), map(RE_NUMBER.match, data))
            data = [[int(a), int(b)] for a, b in data]
            data = np.array(data)

            bboxes[fbbox.stem] = data

        pred_text = {}
        for ftext in (path / "text").glob(".txt"):
            with open(ftext, 'r') as f_in:
                data = f_in.read()
            data = data.split(" ")
            pred_text[fbbox.stem] = data

        for fn in set(bboxes.keys()) & set(pred_text.keys()):
            text = pred_text[fn]
            file_confs = np.ones(len(text))
            pred = Prediction(bboxes[fn], file_confs, text)
            try:
                groups[fn].add_prediction(pred, path.parent.stem)
            except KeyError:
                groups[fn] = PredictionGroup([pred], gt_seq, [fn])


        return groups

    def _load_json_predictions(
        self,
        path: Path,
        groups: Dict[str, PredictionGroup],
        reverse: bool = False,
    ) -> Dict[str, PredictionGroup]:
        raise NotImplementedError

    def _load_gt(self, path: Path,) -> Dict:
        raise NotImplementedError


def setup() -> Namespace:
    """Load command-line arguments and set up stuff.

    Returns
    -------
    Namespace
        Input arguments encapsulated in a namespace object.
    """
    parser = ArgumentParser()
    
    parser.add_argument(
        "groundtruth",
        help="Path to a ground truth file in json format.",
        type=Path,
    )
    parser.add_argument(
        "--prediction",
        help="Path to a prediction path or file (old json format).",
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--pickles",
        help="Path to an epoch output with pickle files.",
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--fewshot",
        help="Path to the output of a few-shot model.",
        action="append",
        type=Path,
    )

    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    """."""
    args = setup()
    evaluator = AlignmentEvaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    args = setup()
    main(args)
