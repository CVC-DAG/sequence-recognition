import json
import pandas as pd
import torch
import torch.utils.data as D
import torchvision.transforms as T
import wandb

from argparse import ArgumentParser
from data.generic_decrypt import GenericDecryptDataset, GenericDecryptVocab, GenericSample
from models.cnns import FullyConvCTC
from pathlib import Path
from pydantic import BaseModel
from torchinfo import summary
from torch import nn
from torch import optim
from torch import Tensor
from torch.optim import lr_scheduler as sched
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union
from utils.augmentations import PadToMax, ResizeKeepingRatio, BinariseFixed, \
    Blackout, StackChannels, KanungoNoise, ToFloat, ToNumpy
from utils.ops import levenshtein


class TrainConfig(BaseModel):
    augmentation: int
    batch_size: int
    checkpoint: str
    device: str
    grad_clip: float
    max_epochs: int
    learning_rate: float
    optimizer: str
    save_every: int
    eval_every: int
    weight_decay: float
    workers: int

    plateau_sched: bool         # Use plateau scheduler
    plateau_factor: float       # Factor by which to reduce LR on plateau
    plateau_iters: int          # Number of epochs on which to reduce w/o improvement
    plateau_thresh: float       # Threshold to measure significant changes
    plateau_min: float          # Min LR value allowed

    warmup_sched: bool          # Use warmup scheduler
    warmup_factor: float        # Factor of reduction of lr at train start
    warmup_iters: int           # Number of iterations to increase LR at the start

    cosann_sched: bool          # Use cosine annealing scheduler
    cosann_t0: int              # Iters until first restart
    cosann_factor: int          # Factor by which to increase the number of iters until restart
    cosann_min: float           # Min learning rate


class ModelConfig(BaseModel):
    kern_upsampling: int
    intermediate_units: int
    pretrained: bool
    resnet: int
    sequence_length: int
    target_shape: Tuple[int, int]   # Width, Height
    output_upsampling: int


class DirectoryConfig(BaseModel):
    results_dir: str
    base_data_dir: str

    training_data: str
    training_root: str
    validation_data: str
    validation_root: str
    test_data: str
    test_root: str

    vocab_data: str


class Config(BaseModel):
    exp_name: str
    description: str
    wandb_project: str
    wandb_mode: str
    train: TrainConfig
    model: ModelConfig
    dirs: DirectoryConfig


def load_configuration(
        config_path: str
) -> Config:
    path = Path(config_path)
    with open(path, 'r') as f_config:
        cfg = json.load(f_config)
        cfg["exp_name"] = path.stem
    cfg = Config(**cfg)

    return cfg


def setup_dirs(
        cfg: Config
) -> Config:
    results_path = Path(cfg.dirs.results_dir)
    base_data_path = Path(cfg.dirs.base_data_dir)

    training_data = base_data_path / cfg.dirs.training_data
    training_root = base_data_path / cfg.dirs.training_root
    validation_data = base_data_path / cfg.dirs.validation_data
    validation_root = base_data_path / cfg.dirs.validation_root
    test_data = base_data_path / cfg.dirs.test_data
    test_root = base_data_path / cfg.dirs.test_root

    vocab_data = Path(cfg.dirs.vocab_data)

    results_path.mkdir(parents=True, exist_ok=True)
    exp_results = results_path / cfg.exp_name
    exp_results.mkdir(exist_ok=True)

    if not vocab_data.exists():
        raise FileNotFoundError

    cfg.dirs.training_data = str(training_data)
    cfg.dirs.training_root = str(training_root)
    cfg.dirs.validation_data = str(validation_data)
    cfg.dirs.validation_root = str(validation_root)
    cfg.dirs.test_data = str(test_data)
    cfg.dirs.test_root = str(test_root)

    cfg.vocab_data = str(vocab_data)

    return cfg

def setup(
) -> Config:
    """
    """
    parser = ArgumentParser(
        description="Sequence to Sequence model training",
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Configuration path for the experiment"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Test the input experiment with the parameter weights",
        default=None,
        required=False,
    )

    args = parser.parse_args()
    cfg = load_configuration(args.config_path)
    cfg = setup_dirs(cfg)

    wandb.init(
        project=cfg.wandb_project,
        dir=Path(cfg.dirs.results_dir) / cfg.exp_name,
        config=cfg.json(),
        mode=cfg.wandb_mode,
        save_code=True,
    )

    return cfg

def create_datasets(
        cfg: Config
) -> Tuple[D.DataLoader, D.DataLoader, D.DataLoader, GenericDecryptVocab]:
    """
    """
    AUGMENTATIONS = {
        0: T.Compose([
            ResizeKeepingRatio(cfg.model.target_shape),
            PadToMax(cfg.model.target_shape),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    train_augs = AUGMENTATIONS[cfg.train.augmentation]
    valid_augs = AUGMENTATIONS[0]

    vocab = GenericDecryptVocab(cfg.dirs.vocab_data)
    
    train_dataset, valid_dataset, test_dataset = [
        D.DataLoader(
            GenericDecryptDataset(
                rootpath,
                datapath,
                vocab,
                cfg.model.sequence_length,
                aug
            ),
            batch_size=cfg.train.batch_size,
            shuffle=True if not ii else False,
            pin_memory=True if not ii and "cuda" in cfg.train.device else False,
            num_workers=cfg.train.workers,
        )
        for ii, (rootpath, datapath, aug) in enumerate(zip(
            [cfg.dirs.training_root, cfg.dirs.validation_root, cfg.dirs.test_root],
            [cfg.dirs.training_data, cfg.dirs.validation_data, cfg.dirs.test_data],
            [train_augs, valid_augs, valid_augs]
        ))
    ]

    return train_dataset, valid_dataset, test_dataset, vocab

def get_lr(
        optimizer
) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def decode_output(
        model_output: Tensor
) -> List[str]:
    ...


class Experiment:
    def __init__(
            self,
            cfg: Config
    ) -> None:
        self.cfg = cfg
        self.data_train, self.data_valid, \
            self.data_test, self.vocab = create_datasets(self.cfg)

        self.model = FullyConvCTC(
            width_upsampling=self.cfg.model.output_upsampling,
            kern_upsampling=self.cfg.model.kern_upsampling,
            intermediate_units=self.cfg.model.intermediate_units,
            output_units=len(self.vocab),
            backbone=FullyConvCTC.RESNETS[self.cfg.model.resnet],
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

        self.subsequent_mask = self.subsequent_mask.to(self.device)
        self.train_iters = 0

        self.best_ser = 99999999
        self.best_epoch = -1

        self.best_name = f"weights_run_{self.run_name}_BEST.pth"
        self.curr_name = lambda epoch: f"weights_run_{self.run_name}_e{epoch}.pth"
        self.save_path = Path(self.cfg.dirs.results_dir) / self.cfg.exp_name

        self.json_name = lambda split, epoch: f"{split}_run_{self.run_name}_e{epoch:05d}.json"
        
    def load_model_weights(
            self,
            path: str
    ) -> None:
        self.model.load_weights(path)

    def inference_step(
            self,
            sample: GenericSample,
    ) -> Tensor:
        output = self.model(sample.img.to(self.device))

        return output

    def train(
            self, 
    ) -> None:
        for epoch in range(1, self.cfg.train.max_epochs + 1):
            self.model.train()

            sequences = []
            gt_sequences = []
            fnames = []
            total_train_loss = 0.0

            for sample in tqdm(self.data_train, desc=f"Epoch {epoch} in Progress..."):
                self.train_iters += 1
                output = self.inference_step(sample)
                train_loss = self.loss_function(
                    output,
                    sample.gt.to(self.device),
                    torch.full(
                        size=sample.img.shape[0],
                        fill_value=output.shape[1],
                        dtype=torch.long
                    ),
                    sample.og_len,
                )

                self.optimizer.zero_grad()
                train_loss.backward()

                if self.cfg.train.grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.train.grad_clip
                    )

                self.optimizer.step()

                if self.sched_warmup:
                    self.sched_warmup.step()
                if self.sched_cosann:
                    self.sched_cosann.step()

                wandb.log({
                    "step": self.train_iters,
                    "lr": get_lr(self.optimizer),
                    "train_loss": train_loss,
                })

            self.log_results(
                "train",
                sequences,
                gt_sequences,
                fnames,
                train_loss,
                epoch,
                confidences,
            )

            if not epoch % self.cfg.train.save_every:
                self.save_current_weights(epoch)

            if not epoch % self.cfg.train.eval_every:
                val_loss, val_ser = self.validate(epoch)

                if val_ser < self.best_ser:
                    self.best_ser = val_ser
                    self.best_epoch = epoch

                    self.model.save_weights(
                        str(self.save_path / self.best_name)
                    )

                # _ = self.test(epoch)

                if self.sched_plateau:
                    self.sched_plateau.step(val_ser)

    def validate(
            self,
            epoch: int,
    ) -> Tuple[float, float]:
        raise NotImplementedError

    def test(
            self,
            epoch: int,
    ) -> Tuple[float, float]:
        raise NotImplementedError

    def log_results(
            self,
            split: str,
            sequences: List,
            gt_sequences: List,
            fnames: List,
            loss: float,
            epoch: int,
            confidences: List,
    ) -> float:
        results = [[" ".join(x), k, " ".join(y), levenshtein(x, y)[0], epoch, z]
                        for x, y, z, k in zip(sequences, gt_sequences, fnames, confidences)]
        table = pd.DataFrame(
            results,
            columns=["Text", "Confidences", "Ground Truth", "SER", "Epoch", "Filename"]
        )

        wandb.log({
            "step": self.train_iters,
            "epoch": epoch,
            f"{split}_loss": loss,
            f"{split}_SER_mean": table["SER"].mean(),
            f"{split}_SER_std": table["SER"].std(),
            f"{split}_sequences": table.head(25),
        })

        if split == "train" and epoch % self.cfg.train.save_every:
            return table["SER"].mean()

        table.to_json(
            str(
                self.save_path / self.json_name(split, epoch)
            )
        )

        return table["SER"].mean()


    def save_current_weights(self, epoch: int) -> None:
        curr_name = self.curr_name(epoch)
        
        self.model.save_weights(
            str(self.save_path / curr_name)
        )

        prev_name = self.curr_name(epoch - self.cfg.train.save_every)
        prev_weights = self.save_path / prev_name

        if prev_weights.exists():
            prev_weights.unlink()