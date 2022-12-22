"""QOL collection of visualisation functions."""

from __future__ import annotations

from typing import (
    List,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plotimg(img: ArrayLike) -> None:
    """Display an image in a singleton plot.

    :param img: Input image in array-like fashion.
    """
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    plt.close()
