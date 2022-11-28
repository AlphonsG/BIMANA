import cv2
import numpy as np
import numpy.typing as npt
import pytest

from bimana.utils.deep_learning.dexined import detect_edges
from tests import LENA_IMG_PTH


@pytest.fixture()
def lena_img() -> npt.NDArray[np.uint8]:
    return cv2.imread(LENA_IMG_PTH, cv2.IMREAD_GRAYSCALE)


def test_detect_edges(lena_img: npt.NDArray[np.uint8]) -> None:
    edges = detect_edges(lena_img)

    assert len(edges.shape) == 2
    assert np.array_equal(range(0, 249), np.unique(edges))
