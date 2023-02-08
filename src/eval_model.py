"""Generate display images for a prediction and evaluate results."""

import json

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from seq_alignment.models.model_zoo import ModelZoo
from seq_alignment.utils.decoding import Prediction, PredictionGroup


class AlignmentEvaluator:
    """Perform evaluation of models, ensembles of models and old predictions."""

    def __init__(self):
        raise NotImplementedError

    def _load_json_predictions(
        self,
        path: Path,
        groups: Dict[str, PredictionGroup]
    ) -> Dict[str, PredictionGroup]:
        with open(path, 'r') as f_in:
            pred_file = json.load(f_in)

        for k, v in pred_file.items():
            prediction = Prediction.from_text_coords(v["coords1d"], v["gt_seq"])
            try:
                groups[k] = groups[k].add_prediction(prediction)
            except KeyError:
                groups[k] = PredictionGroup([prediction], v["gt_seq"], path.stem)

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
    )
    parser.add_argument(
        "--config",
        help="Path to a model config file.",
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--prediction",
        help="Path to a prediction file.",
        action="append",
        type=Path,
    )

    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    """."""
    print(args)


if __name__ == "__main__":
    args = setup()
    main(args)
