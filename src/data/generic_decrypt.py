import json
import re

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms as T
from typing import AnyStr, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.utils.data as D

Coordinate = Tuple[int, int]


class GenericSample(NamedTuple):
    img: Optional[torch.Tensor]
    gt: Union[torch.Tensor, np.array]
    segm: Optional[List[Coordinate]]
    og_len: int
    curr_shape: Optional[Coordinate]
    og_shape: Optional[Coordinate]
    binthresh: int
    filename: str


class GenericDecryptVocab:
    def __init__(
            self,
            path: str,
    ) -> None:
        with open(path, "r") as f_labels:
            jlabels = json.load(f_labels)

        self.blank = "<BLANK>"
        self.go_tok = "<GO>"
        self.stop_tok = "<STOP>"
        self.pad_tok = "<PAD>"
        self.tokens = [self.blank, self.go_tok, self.stop_tok, self.pad_tok]
        self.vocab = self.tokens + jlabels["labels"]

        self.vocab2index = {
            x: ii for ii, x in enumerate(self.vocab)
        }
        self.index2vocab = {
            v: k for k, v in self.vocab2index.items()
        }

    def __len__(self):
        return len(self.vocab)

    def encode(self, labels: List[str]) -> List[int]:
        return [self.vocab2index[x] for x in labels]

    def decode(self, encoded: List[int]) -> List[str]:
        return [self.index2vocab[x] for x in encoded]

    def pad(self, encoded: List[int], pad_len: int) -> List[int]:
        assert len(encoded) + 2 <= pad_len
        return [self.vocab2index[self.go_tok]] \
               + encoded \
               + [self.vocab2index[self.stop_tok]] \
               + [self.vocab2index[self.pad_tok]
                  for _ in range(pad_len - len(encoded) - 2)]

    def unpad(self, padded) -> List[int]:
        output = []
        for x in padded:
            if x == self.vocab2index[self.stop_tok]:
                break
            if x == self.vocab2index[self.pad_tok] or \
                    x == self.vocab2index[self.go_tok]:
                continue
            output.append(x)
        return output


class GenericDecryptDataset(D.Dataset):
    DEFAULT_TRANSFORMS = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    RE_SEPARATOR = re.compile(" ")

    def __init__(
            self,
            root_path: str,
            data_path: str,
            vocab: GenericDecryptVocab,
            seqlen: int,
            aug_pipeline: Optional[Callable] = None,
    ) -> None:
        super(GenericDecryptDataset).__init__()

        self._samples = []
        self._root_path = Path(root_path)  # Images Folder
        self._data_path = Path(data_path)  # GT File
        self._seqlen = seqlen

        self._aug_pipeline = aug_pipeline if aug_pipeline is not None \
            else self.DEFAULT_TRANSFORMS

        # Open gt data file and load its contents
        with open(self._data_path, "r") as f_gt:
            gt = json.load(f_gt)

        for fn, sample in gt.items():
            transcript = self.RE_SEPARATOR.split(sample["ts"])
            if "segm" in sample and len(sample["segm"]):
                segmentation = [[-1.0, -1.0]] + sample["segm"] + \
                              ([[-1.0, -1.0]] * (self._seqlen - len(sample["segm"]) - 1))
                segmentation = np.array(segmentation)
            else:
                segmentation = None
                                                
            og_len = len(transcript)

            transcript = vocab.encode(transcript)

            if seqlen is not None:
                transcript = vocab.pad(transcript, seqlen)

            self._samples.append(GenericSample(
                img=None,
                gt=np.asarray(transcript),
                segm=segmentation,
                og_len=og_len,
                curr_shape=None,
                og_shape=None,
                binthresh=128,
                filename=str(self._root_path / fn),
            ))

    def __len__(
            self,
    ) -> int:
        return len(self._samples)

    def __getitem__(
            self,
            index: int
    ) -> GenericSample:

        sample = self._samples[index]

        img = Image.open(sample.filename).convert('RGB')
        og_shape = img.size
        img = self._aug_pipeline(img)
        new_shape = img.shape[1:]

        return GenericSample(
            img=img,
            gt=sample.gt,
            segm=sample.segm,
            og_len=sample.og_len,
            curr_shape=new_shape,
            og_shape=og_shape,
            binthresh=128,
            filename=sample.filename,
        )
