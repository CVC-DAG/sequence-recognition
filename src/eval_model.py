"""Generate display images for a prediction and evaluate results."""

import json
import pickle

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np

from seq_alignment.models.model_zoo import ModelZoo
from seq_alignment.utils.decoding import Prediction, PredictionGroup


class AlignmentEvaluator:
    """Perform evaluation of models, ensembles of models and old predictions."""

    def __init__(self, args: Namespace) -> None:
        self._groups = {}
        
        for p in args.pickles:
            self._groups = self._load_pickle_predictions(p)

        for p in args.config:
            self._groups = self._load_pickle_predictions(p)
        
        for p in args.prediction:
            self._groups = self._load_pickle_predictions(p)

    def _load_pickle_predictions(
        self,
        path: Path,
        groups: Dict[str, PredictionGroup]
    ) -> Dict[str, PredictionGroup]:
        coord_file = path / "results_coords1d.pkl"
        confs_file = path / "results_coords1d_confidences.pkl"

        assert coord_file.exists(), "Coordinate file does not exist"
        assert confs_file.exists(), "Confidence file does not exist"

        with open(coord_file, 'r') as f_coords, open(confs_file, 'r') as f_confs:
            coords = json.load(f_coords)
            confs = json.load(f_confs)

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
    ) -> Dict[str, PredictionGroup]:
        raise NotImplementedError

    def _generate_model_predictions(
        self,
    ) -> Dict[str, PredictionGroup]:
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
        "dataset",              # The file to the dataset should be imported here.
                                # The dataloader config should be imported from the
                                # model config
        help="Path to the main dataset file.",
        type=Path,
    )
    parser.add_argument(
        "--config",
        help="Path to a model config file.",
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--prediction",
        help="Path to a prediction path or file.",
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
