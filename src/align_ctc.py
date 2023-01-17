"""Alignment using a fully convolutional CTC model."""

import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import torchvision.transforms as T
import wandb

from argparse import ArgumentParser
from data.generic_decrypt import (
    GenericDecryptDataset,
    GenericDecryptVocab,
    GenericSample,
)
from models.cnns import FullyConvCTC
from pathlib import Path
from pydantic import BaseModel
from shutil import copyfile
from torchinfo import summary
from torch import nn
from torch import optim
from torch import Tensor
from torch.optim import lr_scheduler as sched
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union
from utils.augmentations import (
    PadToMax,
    ResizeKeepingRatio,
    BinariseFixed,
    Blackout,
    StackChannels,
    KanungoNoise,
    ToFloat,
    ToNumpy,
)
from utils.ops import levenshtein, moving_average
from utils.decoding import Prediction, decode_ctc_greedy, decode_ctc
from utils.visualisation import display_prediction


class TrainConfig(BaseModel):
    augmentation: int
    batch_size: int
    checkpoint: Optional[str]
    device: str
    grad_clip: Optional[float]
    max_epochs: int
    learning_rate: float
    optimizer: str
    save_every: int
    eval_every: int
    weight_decay: float
    workers: int

    plateau_sched: bool  # Use plateau scheduler
    plateau_factor: float  # Factor by which to reduce LR on plateau
    plateau_iters: int  # Number of epochs on which to reduce w/o improvement
    plateau_thresh: float  # Threshold to measure significant changes
    plateau_min: float  # Min LR value allowed

    warmup_sched: bool  # Use warmup scheduler
    warmup_factor: float  # Factor of reduction of lr at train start
    warmup_iters: int  # Number of iterations to increase LR at the start

    cosann_sched: bool  # Use cosine annealing scheduler
    cosann_t0: int  # Iters until first restart
    cosann_factor: int  # Factor by which to increase the number of iters until restart
    cosann_min: float  # Min learning rate

    log_images: int  # Number of images to log during validation


class ModelConfig(BaseModel):
    kern_upsampling: int
    intermediate_units: int
    pretrained: bool
    resnet: int
    sequence_length: int
    target_shape: Tuple[int, int]  # Width, Height
    output_upsampling: int
    decode_beams: int


class DirectoryConfig(BaseModel):
    results_dir: str
    base_data_dir: str

    training_file: str
    training_root: str
    validation_file: str
    validation_root: str
    test_file: str
    test_root: str

    vocab_data: str


class Config(BaseModel):
    exp_name: str
    cipher: str
    description: str
    wandb_project: str
    wandb_mode: str
    train: TrainConfig
    model: ModelConfig
    dirs: DirectoryConfig


def load_configuration(config_path: str, test: bool) -> Config:
    path = Path(config_path)
    with open(path, "r") as f_config:
        cfg = json.load(f_config)
        cfg["exp_name"] = path.stem + ("_test" if test else "")
    cfg = Config(**cfg)

    return cfg


def setup_dirs(cfg: Config) -> Config:
    results_path = Path(cfg.dirs.results_dir)
    base_data_path = Path(cfg.dirs.base_data_dir)

    training_file = base_data_path / cfg.dirs.training_file
    training_root = base_data_path / cfg.dirs.training_root
    validation_file = base_data_path / cfg.dirs.validation_file
    validation_root = base_data_path / cfg.dirs.validation_root
    test_file = base_data_path / cfg.dirs.test_file
    test_root = base_data_path / cfg.dirs.test_root

    vocab_data = Path(cfg.dirs.vocab_data)

    results_path.mkdir(parents=True, exist_ok=True)
    exp_results = results_path / cfg.exp_name
    exp_results.mkdir(exist_ok=True)

    if not vocab_data.exists():
        raise FileNotFoundError

    cfg.dirs.training_file = str(training_file)
    cfg.dirs.training_root = str(training_root)
    cfg.dirs.validation_file = str(validation_file)
    cfg.dirs.validation_root = str(validation_root)
    cfg.dirs.test_file = str(test_file)
    cfg.dirs.test_root = str(test_root)

    cfg.dirs.vocab_data = str(vocab_data)
    cfg.dirs.results_dir = str(exp_results)

    return cfg


def setup() -> Config:
    """Load configuration and set up paths.

    :returns: Singleton configuration object.
    """
    parser = ArgumentParser(
        description="Sequence to Sequence model training",
    )
    parser.add_argument(
        "config_path", type=str, help="Configuration path for the experiment"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Test the input experiment with the parameter weights",
        default=None,
        required=False,
    )

    args = parser.parse_args()
    cfg = load_configuration(args.config_path, args.test is not None)
    cfg = setup_dirs(cfg)

    wandb.init(
        project=cfg.wandb_project,
        dir=cfg.dirs.results_dir,
        config=cfg.dict(),
        mode=cfg.wandb_mode,
        save_code=True,
    )

    if wandb.run.name:
        fname = f"{wandb.run.name}.json"
    else:
        fname = "unknown_run.json"

    copyfile(args.config_path, Path(cfg.dirs.results_dir) / fname)

    return cfg, args.test


def create_datasets(
    cfg: Config,
) -> Tuple[D.DataLoader, D.DataLoader, D.DataLoader, GenericDecryptVocab]:
    """Load datasets and vocab object from the experiment configuration.

    :param cfg: Experiment configuration.
    :returns: Training, validation and test partitions as dataloaders, as well
    as the vocabulary object to convert back and forth to encoded and decoded
    representations.
    """
    AUGMENTATIONS = {
        0: [],
        1: [
            T.GaussianBlur(3),
            T.RandomEqualize(),
            T.RandomPerspective(fill=255),
        ]
    }

    train_augs = AUGMENTATIONS[cfg.train.augmentation]
    valid_augs = AUGMENTATIONS[0]

    vocab = GenericDecryptVocab(cfg.dirs.vocab_data)

    train_dataset, valid_dataset, test_dataset = [
        D.DataLoader(
            GenericDecryptDataset(
                img_path, data_file, vocab, cfg.model.sequence_length, cfg.model.target_shape, aug
            ),
            batch_size=cfg.train.batch_size,
            shuffle=True if not ii else False,
            pin_memory=True if not ii and "cuda" in cfg.train.device else False,
            num_workers=cfg.train.workers,
        )
        for ii, (img_path, data_file, aug) in enumerate(
            zip(
                [cfg.dirs.training_root, cfg.dirs.validation_root, cfg.dirs.test_root],
                [cfg.dirs.training_file, cfg.dirs.validation_file, cfg.dirs.test_file],
                [train_augs, valid_augs, valid_augs],
            )
        )
    ]

    return train_dataset, valid_dataset, test_dataset, vocab


def get_lr(optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


class Experiment:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.data_train, self.data_valid, self.data_test, self.vocab = create_datasets(
            self.cfg
        )

        self.model = FullyConvCTC(
            width_upsampling=self.cfg.model.output_upsampling,
            kern_upsampling=self.cfg.model.kern_upsampling,
            intermediate_units=self.cfg.model.intermediate_units,
            output_units=len(self.vocab),
            resnet_type=self.cfg.model.resnet,
            pretrained=self.cfg.model.pretrained,
        )

        if self.cfg.train.checkpoint is not None:
            self.model.load_weights(self.cfg.train.checkpoint)

        if self.cfg.train.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.train.learning_rate,
                weight_decay=self.cfg.train.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg.train.learning_rate,
                weight_decay=self.cfg.train.weight_decay,
            )

        if self.cfg.train.plateau_sched:
            self.sched_plateau = sched.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode="min",
                factor=self.cfg.train.plateau_factor,
                patience=self.cfg.train.plateau_iters,
                threshold=self.cfg.train.plateau_thresh,
                min_lr=self.cfg.train.plateau_min,
                verbose=True,
            )
        else:
            self.sched_plateau = None

        if self.cfg.train.warmup_sched:
            self.sched_warmup = sched.LinearLR(
                optimizer=self.optimizer,
                start_factor=self.cfg.train.warmup_factor,
                total_iters=self.cfg.train.warmup_iters,
            )
        else:
            self.sched_warmup = None

        if self.cfg.train.cosann_sched:
            self.sched_cosann = sched.CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=self.cfg.train.cosann_t0,
                T_mult=self.cfg.train.cosann_factor,
                eta_min=self.cfg.train.cosann_min,
            )
        else:
            self.sched_cosann = None

        self.run_name = wandb.run.name

        self.device = torch.device(self.cfg.train.device)
        self.loss_function = nn.CTCLoss()

        self.model = self.model.to(self.device)

        self.train_iters = 0
        self.best_iou = 0.0
        self.best_epoch = -1

        self.best_name = f"weights_exp_{self.cfg.exp_name}_run_{self.run_name}_BEST.pth"
        self.curr_name = (
            lambda epoch: f"weights_exp_{self.cfg.exp_name}_run_{self.run_name}_e{epoch}.pth"
        )
        self.save_path = Path(self.cfg.dirs.results_dir)

        self.json_name = (
            lambda split, epoch: f"{split}_exp_{self.cfg.exp_name}_run_{self.run_name}_e{epoch:05d}.json"
        )
        self.csv_name = (
            lambda split, epoch: f"{split}_exp_{self.cfg.exp_name}_run_{self.run_name}_e{epoch:05d}.csv"
        )
        self.image_name = (
            lambda fname, epoch: f"demo_run_{self.run_name}_{fname}_e{epoch:05d}.png"
        )

        self.iou_thresholds = [.25, .50, .75]

    def load_model_weights(self, path: str) -> None:
        """Load weights pointed by the input path to the experiment model.

        :param path: Path to the weights to load.
        """
        self.model.load_weights(path)

    def inference_step(self, sample: GenericSample) -> Tensor:
        """Perform a single step of inference.

        :param sample: Input sample to pass through the model.
        """
        output = self.model(sample.img.to(self.device))

        return output

    def train(self) -> None:
        summary(self.model)

        for epoch in range(1, self.cfg.train.max_epochs + 1):
            self.model.train()

            sequences = []
            gt_sequences = []
            fnames = []

            for sample in tqdm(self.data_train, desc=f"Epoch {epoch} in Progress..."):
                self.train_iters += 1
                output = self.inference_step(sample)

                columns = output.shape[0]
                target_shape = cfg.model.target_shape[0]
                input_lengths = sample.curr_shape[0] * (columns / target_shape)
                input_lengths = input_lengths.numpy().astype(int).tolist()

                batch_loss = self.loss_function(
                    output,
                    sample.gt.to(self.device),
                    input_lengths,
                    tuple(sample.og_len.tolist()),
                )

                self.optimizer.zero_grad()
                batch_loss.backward()

                if self.cfg.train.grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.grad_clip
                    )

                self.optimizer.step()

                if self.sched_warmup:
                    self.sched_warmup.step()
                if self.sched_cosann:
                    self.sched_cosann.step()

                wandb.log(
                    {
                        "lr": get_lr(self.optimizer),
                        "train_loss": batch_loss,
                    },
                    step=self.train_iters,
                )

                output = output.detach().cpu().numpy()
                pred_sequences = decode_ctc_greedy(output)

                sequences += [self.vocab.decode(self.vocab.unpad(x)) for x in pred_sequences]
                gt_sequences += [self.vocab.decode(self.vocab.unpad(x)) for x in sample.gt.detach().cpu().numpy()]

                fnames += sample.filename

            self.log_results(
                "train",
                sequences,
                gt_sequences,
                fnames,
                batch_loss,
                epoch,
            )

            if not epoch % self.cfg.train.save_every:
                self.save_current_weights(epoch)

            if not epoch % self.cfg.train.eval_every:
                val_loss, val_iou = self.validate(epoch, self.data_valid)

                if val_iou > self.best_iou:
                    self.best_iou = val_iou
                    self.best_epoch = epoch

                    self.model.save_weights(str(self.save_path / self.best_name))

                # _ = self.test(epoch)

                if self.sched_plateau:
                    self.sched_plateau.step(val_iou)

    def validate(
        self,
        epoch: int,
        dataset: D.DataLoader,
        mode: str = "valid",
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0

        fnames = []
        predictions = []
        gt_coordinates = []

        with torch.no_grad():
            for sample in tqdm(
                dataset, desc=f"{mode} for epoch {epoch} in Progress..."
            ):
                output = self.inference_step(sample)

                columns = output.shape[0]
                target_shape = cfg.model.target_shape[0]
                curr_shape = sample.curr_shape[0]
                og_shape = sample.og_shape[0]

                input_lengths = sample.curr_shape[0] * (columns / target_shape)
                input_lengths = input_lengths.numpy().astype(int).tolist()

                batch_loss = self.loss_function(
                    output,
                    sample.gt.to(self.device),
                    input_lengths,
                    tuple(sample.og_len.tolist()),
                )
                total_loss += batch_loss

                output = output.detach().cpu().numpy()
                curr_gt = [np.array(self.vocab.unpad(x)) for x in sample.gt.detach().cpu().numpy()]
                segm = sample.segm.numpy()

                widths = (target_shape / columns) * (og_shape / curr_shape)

                predictions += decode_ctc(
                    output,
                    curr_gt,
                    widths,
                    cfg.model.decode_beams
                )

                fnames += sample.filename
                gt_coordinates += [x[:length]
                                   for x, length in zip(segm, sample.og_len)]

            iou = self.log_validation(
                mode,
                predictions,
                gt_coordinates,
                fnames,
                total_loss / len(dataset),
                epoch,
            )

            return total_loss / len(dataset), iou

    def test(self) -> None:
        self.validate(0, self.data_test, "test")

    def log_validation(
        self,
        split: str,
        pred_coords: List,
        gt_coords: List,
        fnames: List,
        loss: float,
        epoch: int,
    ) -> float:

        global_iou = 0.0
        global_predictions = 0
        if cfg.train.log_images is not None:
            img_output_path = self.save_path / f"demo_epoch{epoch}"
            img_output_path.mkdir(exist_ok=True, parents=False)

        results = []
        for ii, (pred, gt, fname) in enumerate(zip(pred_coords, gt_coords, fnames)):
            if cfg.train.log_images is not None and ii < cfg.train.log_images:
                display_prediction(
                    fname,
                    pred.get_coords().tolist(),
                    gt,
                    img_output_path / self.image_name(Path(fname).stem, epoch)
                )

            iou = pred.compare(gt)
            if len(iou):
                global_iou += iou.sum()
                global_predictions += len(iou)
                mean_iou = iou.mean()
                runn_avg = moving_average(iou, 5)[-1] if len(iou) > 5 else np.nan
                std_iou = iou.std()
                hit25 = np.sum(iou >= 0.25) / len(iou)
                hit50 = np.sum(iou >= 0.50) / len(iou)
                hit75 = np.sum(iou >= 0.75) / len(iou)
                misses = np.sum(iou == 0) / len(iou)
                results.append([
                    fname, mean_iou, runn_avg, std_iou, hit25, hit50, hit75,
                    misses, pred.get_coords().tolist(), gt.tolist()
                ])
            else:
                results.append([
                    fname, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, [], gt.tolist()
                ])
        table = pd.DataFrame(
            results,
            columns=["fname", "mean_iou", "run_avg", "std_iou", "hit25", "hit50", "hit75",
                     "misses", "pred_coord", "gt_coord"]
        )
        wandb.log(
            {
                f"{split}_iou": global_iou / global_predictions,
                f"{split}_sample": table.head(cfg.train.log_images),
                f"{split}_loss": loss,
            },
            step=self.train_iters,
        )

        table.to_json(str(self.save_path / self.json_name(split, epoch)))
        table.to_csv(str(self.save_path / self.csv_name(split, epoch)))

        return global_iou / global_predictions

    def log_results(
        self,
        split: str,
        sequences: List,
        gt_sequences: List,
        fnames: List,
        loss: float,
        epoch: int,
    ) -> float:

        results = [
            [" ".join(x), " ".join(y), levenshtein(x, y)[0], epoch, z]
            for x, y, z in zip(sequences, gt_sequences, fnames)
        ]
        table = pd.DataFrame(
            results,
            columns=["pred_seq", "gt_seq", "ser", "epoch", "fname"],
        )

        wandb.log(
            {
                "epoch": epoch,
                f"{split}_loss": loss,
                f"{split}_SER_mean": table["ser"].mean(),
                f"{split}_SER_std": table["ser"].std(),
                f"{split}_sequences": table.head(25),
            },
            step=self.train_iters,
        )

        table.to_json(str(self.save_path / self.json_name(split, epoch)))

        return table["ser"].mean()

    def save_current_weights(self, epoch: int) -> None:
        curr_name = self.curr_name(epoch)

        self.model.save_weights(str(self.save_path / curr_name))

        prev_name = self.curr_name(epoch - self.cfg.train.save_every)
        prev_weights = self.save_path / prev_name

        if prev_weights.exists():
            prev_weights.unlink()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    cfg, test = setup()
    exp = Experiment(cfg)

    if test is not None:
        exp.load_model_weights(test)
        exp.test()
    else:
        exp.train()
