from pathlib import Path

import cv2
import pytest
import numpy as np
import numpy.typing as npt
from bimana.tight_junctions.analysis import (extract_tight_junctions,
                                             perform_ijoq, segment_cells)
from bimana.utils.image_processing import MAX_PX_VAL, ImageChannel
from tests import BIN_TGHT_JNCS_PTH, TIGHT_JUNCS_IMG_PTH


@pytest.fixture(scope='module')
def img() -> npt.NDArray[np.uint8]:
    return cv2.imread(TIGHT_JUNCS_IMG_PTH)


@pytest.fixture
def bin_tght_jncs_img() -> npt.NDArray[np.bool_]:
    return (cv2.imread(BIN_TGHT_JNCS_PTH,
                       cv2.IMREAD_GRAYSCALE) / MAX_PX_VAL).astype(bool)


def test_extract_tight_junctions(
    img: npt.NDArray[np.uint8],
    tmp_path: Path,
) -> None:
    tight_juncs, procd_imgs = extract_tight_junctions(img, ImageChannel.GREEN)

    assert len(np.unique(tight_juncs)) == 2

    for name, procd_img in procd_imgs:
        assert cv2.imwrite(str(tmp_path / f'{name}.png'), procd_img)


def test_segment_cells(
    img: npt.NDArray[np.uint8],
    tmp_path: Path,
) -> None:
    cells, procd_imgs = segment_cells(img, ImageChannel.BLUE)

    assert len(cells) == 5

    for name, procd_img in procd_imgs:
        assert cv2.imwrite(str(tmp_path / f'{name}.png'), procd_img)


def test_perform_ijoq(bin_tght_jncs_img: npt.NDArray[np.bool_]) -> None:
    assert perform_ijoq(bin_tght_jncs_img) == 0.0316
