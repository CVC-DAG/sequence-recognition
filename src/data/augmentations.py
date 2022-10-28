from typing import Tuple

import numpy as np
import cv2

from torchvision import transforms as T
import torchvision.transforms.functional as F


class Blackout:
    def __call__(self, image):
        return image - image


class PadToMax:
    def __init__(self, max_size: Tuple[int, int]):
        self.max_size = max_size

    def __call__(self, image):
        w, h = image.size
        hpf = int(self.max_size[0] - w)
        hp = max(hpf // 2, 0)
        vpf = int(self.max_size[1] - h)
        vp = max(vpf // 2, 0)
        padding = [hp + int(hpf % 2), vp + int(vpf % 2), hp, vp]
        return F.pad(image, padding, 0, 'constant')


class ResizeKeepingRatio:
    def __init__(self, max_size: Tuple[int, int]):
        self.max_size = max_size

    def __call__(self, image):
        max_w, max_h = self.max_size
        w, h = image.size

        ratio = min(max_w / w, max_h / h)

        return F.resize(image, [int(h * ratio), int(w * ratio)])


class BinariseFixed:
    def __init__(self, threshold):
        self.lut = [int(x < threshold) * 255 for x in range(256)]

    def __call__(self, image):
        return image.convert('L').point(self.lut).convert('RGB')


class BinariseCustom:
    def __call__(self, image, threshold):
        return image.convert('L').point(lambda x: int(x < threshold) * 255).convert('RGB')


class KanungoNoise:
    def __init__(
            self,
            alpha=2.0,
            beta=2.0,
            alpha_0=1.0,
            beta_0=1.0,
            mu=.05,
            k=2
    ) -> None:
            """ Applies Kanungo noise model to a binary image.

            T. Kanungo, R. Haralick, H. Baird, W. Stuezle, and D. Madigan.
            A statistical, nonparametric methodology for document degradation model validation.
            IEEE Transactions Pattern Analysis and Machine Intelligence 22(11):1209 - 1223, 2000.

            Args:
            img -- 8U1C binary image either [0..1] or [0..255]
            alpha -- controls the probability of a foreground pixel flip (default = 2.0)
            alpha_0 --  controls the probability of a foreground pixel flip (default = 1.0)
            beta -- controls the probability of a background pixel flip (default = 2.0)
            beta_0 -- controls the probability of a background pixel flip (default = 1.0)
            mu -- constant probability of flipping for all pixels (default = 0.05)
            k -- diameter of the disk structuring element for the closing operation (default = 2)

            Returns:
            out -- 8U1C [0..255], binary image
            """
            self.alpha = alpha
            self.beta = beta
            self.alpha_0 = alpha_0
            self.beta_0 = beta_0
            self.mu = mu
            self.k = k

    def __call__(self, img):
        H, W = img.shape
        img = img // np.max(img)
        dist = cv2.distanceTransform(1 - img, cv2.DIST_L1, 3)
        dist2 = cv2.distanceTransform(img, cv2.DIST_L1, 3)
        P = (self.alpha_0 * np.exp(- self.alpha * dist ** 2)) + self.mu
        P2 = (self.beta_0 * np.exp(- self.beta * dist2**2)) + self.mu
        distorted = img.copy()
        distorted[((P > np.random.rand(H, W)) & (img == 0))] = 1
        distorted[((P2 > np.random.rand(H, W)) & (img == 1))] = 0
        closing = cv2.morphologyEx(
            distorted,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.k, self.k))
        )
        return closing * 255


class ToNumpy:
    def __call__(self, image):
        return np.asarray(image, np.uint8)


class ToFloat:
    def __call__(self, image):
        return image.astype(np.float32)


class StackChannels:
    def __call__(self, image):
        return np.stack([image] * 3, axis=-1)
