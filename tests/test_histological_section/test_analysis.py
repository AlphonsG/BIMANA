import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from bimana.histological_section.analysis import (
    amount_cilia_above_tissue_area, amount_staining_in_tissue_area,
    tissue_boundary)
from tests import TEST_IMG_PTH, TEST_TISS_BNDY_PTH


@pytest.fixture(scope='module')
def img() -> npt.NDArray[Any]:
    return cv2.imread(TEST_IMG_PTH)


@pytest.fixture
def tiss_bndy() -> None:
    with open(TEST_TISS_BNDY_PTH, 'rb') as f:
        return pickle.load(f)


def test_amount_staining_in_tissue_areae(
    img: npt.NDArray[Any],
    tiss_bndy: list[list[npt.NDArray[Any]]],
    tmp_path: Path,
) -> None:
    upper_lower_xs, upper_lower_ys = tiss_bndy
    stain_lower_bgr = np.array([0, 0, 0])
    stain_upper_bgr = np.array([255, 255, 150])

    pct_stn, stn_amt, tiss_size, procd_imgs = amount_staining_in_tissue_area(
        img, upper_lower_xs, upper_lower_ys, stain_lower_bgr, stain_upper_bgr)

    assert 10 < pct_stn < 25
    assert 10 < (stn_amt / tiss_size) * 100 < 25
    assert tiss_size > stn_amt

    for name, procd_img in procd_imgs:
        assert cv2.imwrite(str(tmp_path / f'{name}.png'), procd_img)


def test_amount_cilia_above_tissue_area(
    img: npt.NDArray[Any],
    tiss_bndy: list[list[npt.NDArray[Any]]],
    tmp_path: Path,
) -> None:
    upper_lower_xs, upper_lower_ys = tiss_bndy
    cilia_lower_bgr = np.array([220, 220, 220])
    cilia_upper_bgr = np.array([245, 245, 245])
    thickness = 15

    pct_cil, cil_amt, reg_size, procd_imgs = amount_cilia_above_tissue_area(
        img, upper_lower_xs, upper_lower_ys, cilia_lower_bgr, cilia_upper_bgr,
        thickness)

    assert 20 < pct_cil < 35
    assert 20 < (cil_amt / reg_size) * 100 < 35
    assert reg_size > cil_amt

    for name, procd_img in procd_imgs:
        assert cv2.imwrite(str(tmp_path / f'{name}.png'), procd_img)


@pytest.mark.parametrize(
    'iso_non_tissue_sens,smooth_segmented_tissue_boundary',
    [(0.2, True), (None, False)]
)
def test_tissue_boundary(
    iso_non_tissue_sens: float | None,
    smooth_segmented_tissue_boundary: bool,
    img: npt.NDArray[Any],
    tmp_path: Path,
) -> None:
    #  TODO compare outputs of the different inputs?
    non_tissue_stain_lower_bgr = np.array([210, 210, 210])
    non_tissue_stain_upper_bgr = np.array([255, 255, 150])

    (upper_lower_xs, upper_lower_ys), procd_imgs = tissue_boundary(
        img, non_tissue_stain_lower_bgr, non_tissue_stain_upper_bgr,
        iso_non_tissue_sens, smooth_segmented_tissue_boundary)

    assert len(upper_lower_xs[0]) == len(upper_lower_ys[0])
    assert len(upper_lower_xs[1]) == len(upper_lower_ys[1])

    assert len(upper_lower_xs[0]) > 2
    assert len(upper_lower_xs[1]) > 2

    assert np.all(upper_lower_ys[0] < upper_lower_ys[1])

    for name, procd_img in procd_imgs:
        assert cv2.imwrite(str(tmp_path / f'{name}.png'), procd_img)
