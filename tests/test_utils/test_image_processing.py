from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from bimana.utils.image_processing import (MAX_PX_VAL, binary_to_bgr_image,
                                           bound_foreground,
                                           bounded_foreground_polyline,
                                           draw_polyline,
                                           remove_isolated_segmented_objects,
                                           scale_bgr_values,
                                           smooth_foreground_bounds,
                                           segment_image,
                                           segment_polyline_bounded_region,
                                           segment_region_above_polyline,
                                           segment_region_within_image)
from tests import TEST_BIN_IMG_PTH, TEST_REF_BIN_IMG_PTH


@pytest.fixture(scope='module')
def bin_img() -> npt.NDArray[bool]:
    return (cv2.imread(TEST_BIN_IMG_PTH,
                       cv2.IMREAD_GRAYSCALE) / MAX_PX_VAL).astype(bool)


@pytest.fixture
def refd_bin_img() -> npt.NDArray[bool]:
    return (cv2.imread(TEST_REF_BIN_IMG_PTH,
                       cv2.IMREAD_GRAYSCALE) / MAX_PX_VAL).astype(bool)


def test_segment_image() -> None:
    hwc = (10, 20, 3)
    img, expected_result = np.zeros(hwc), np.zeros(hwc[:2])
    img[5:, :10] = (210, 210, 210)
    expected_result[5:, :10] = 1
    bin_img = segment_image(img, (210, 210, 210), (210, 210, 210))

    assert np.array_equal(bin_img, expected_result)


def test_remove_isolated_segmented_objects(bin_img: npt.NDArray[bool]) -> None:
    threshs = [0.1, 0.2, 0.5]
    refined_bin_imgs = [remove_isolated_segmented_objects(bin_img, thresh)
                        for thresh in threshs]

    for refined_bin_img in refined_bin_imgs:
        unique_elements = np.unique(refined_bin_img)
        assert len(unique_elements) == 2
        assert unique_elements[0] == 0
        assert unique_elements[1] == 1

    assert (np.count_nonzero(refined_bin_imgs[2]) < np.count_nonzero(
            refined_bin_imgs[1]) < np.count_nonzero(refined_bin_imgs[0]))


def test_bound_foreground(bin_img: npt.NDArray[bool]) -> None:
    upper_lower_xs, upper_lower_ys = bound_foreground(bin_img)

    assert len(upper_lower_xs[0]) == len(upper_lower_ys[0])
    assert len(upper_lower_xs[1]) == len(upper_lower_ys[1])

    assert len(upper_lower_xs[0]) > 2
    assert len(upper_lower_xs[1]) > 2

    assert np.all(upper_lower_ys[0] < upper_lower_ys[1])


def test_bounded_foreground_polyline() -> None:
    upper_lower_xs = [np.array([0, 1]), np.array([0, 1])]
    upper_lower_ys = [np.array([10, 20]), np.array([50, 40])]
    polyline = bounded_foreground_polyline(upper_lower_xs, upper_lower_ys)

    assert np.array_equal(polyline, np.array([[0, 10], [1, 20], [1, 40],
                                              [0, 50]]))


def test_smooth_foreground_bounds(refd_bin_img: npt.NDArray[bool]) -> None:
    upper_ys = np.array([43, 43, 42, 43, 42, 42, 43, 43, 43, 43, 43, 43, 43,
                         43, 43, 45, 44, 44, 46, 66, 66, 45, 44, 43, 44, 44,
                         44, 45, 44, 44, 45, 45, 45, 45, 44, 44, 44, 44, 44,
                         44, 44, 44, 45, 46, 46])
    lower_ys = np.array([133, 132, 124, 125, 127, 126, 133, 134, 135, 136, 135,
                         136, 136, 136, 136, 135, 135, 138, 138, 120, 117, 117,
                         117, 117, 116, 120, 135, 137, 137, 137, 137, 137, 137,
                         136, 138, 137, 128, 126, 93, 95, 95, 136, 136, 136,
                         136])
    upper_lower_xs = [np.arange(lower_ys.size), np.arange(lower_ys.size)]
    upper_lower_ys = [upper_ys, lower_ys]

    refd_upper_lower_xs, refd_upper_lower_ys = smooth_foreground_bounds(
        refd_bin_img, upper_lower_xs, upper_lower_ys)

    assert len(refd_upper_lower_xs[0]) == len(refd_upper_lower_ys[0])
    assert len(refd_upper_lower_xs[1]) == len(refd_upper_lower_ys[1])

    assert np.all(refd_upper_lower_ys[0] < 66)
    assert np.all(refd_upper_lower_ys[1] > 127)

    upper_ys = np.array([43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
                         43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
                         43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43])
    lower_ys = np.array([133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133,
                         133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133,
                         133, 133, 133, 133, 133, 133, 133, 133, 133, 133])
    upper_lower_xs = [np.arange(lower_ys.size), np.arange(lower_ys.size)]
    upper_lower_ys = [upper_ys, lower_ys]

    refd_upper_lower_xs, refd_upper_lower_ys = smooth_foreground_bounds(
        refd_bin_img, upper_lower_xs, upper_lower_ys)

    for i in (0, 1):
        assert np.array_equal(upper_lower_xs[i], refd_upper_lower_xs[i])
        assert np.array_equal(upper_lower_ys[i], refd_upper_lower_ys[i])


def test_segment_polyline_bounded_region() -> None:
    hw = (10, 20)
    polyline = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    mask = segment_polyline_bounded_region(polyline, hw)

    assert np.all(mask[:, :11])
    assert np.all(~mask[:, 11:])


def test_segment_region_within_image() -> None:
    hwc = (10, 20, 3)

    region = np.zeros(hwc[:2], bool)
    region[5:, :] = 1

    img = np.zeros(hwc)
    img[:, :10, :] = (255, 0, 0)
    img[:, 10:20, :] = (0, 0, 255)
    lower_bgr = upper_bgr = (255, 0, 0)

    produced_result = segment_region_within_image(img, lower_bgr, upper_bgr,
                                                  region)

    expected_result = np.zeros(hwc[:2], bool)
    expected_result[5:, :10] = 1

    assert np.array_equal(produced_result, expected_result)


def test_segment_region_above_polyline() -> None:
    hw = (10, 20)
    height = 3

    upper_ys = np.array([4, 4, 4, 4, 5, 5, 5, 5])
    upper_xs = np.arange(5, 5 + upper_ys.size)
    polyline = np.array([[x, y] for x, y in zip(upper_xs, upper_ys)])

    expected_result = np.zeros(hw, bool)
    expected_result[1:5, 5:9] = 1
    expected_result[2:6, 9:13] = 1

    mask = segment_region_above_polyline(polyline, height, hw)

    assert np.array_equal(mask, expected_result)


def test_draw_polyline() -> None:
    hwc = (10, 20, 3)
    img = np.zeros(hwc)

    expected_result = img.copy()
    expected_result[0, :11] = (0, 255, 0)

    polyline = np.array([[0, 0], [10, 0]])
    img = draw_polyline(img, polyline, (0, 255, 0), 1)

    assert np.array_equal(expected_result, img)


def test_binary_to_bgr_image() -> None:
    hwc = (10, 20, 3)
    img = np.zeros(hwc[:2], bool)
    img[0, :] = 1

    bgr_img = binary_to_bgr_image(img)

    assert bgr_img.shape == (10, 20, 3)
    assert np.all(bgr_img[0, :] == 255)
    assert np.all(bgr_img[1:, :] == 0)


def test_scale_bgr_values() -> None:
    expected_result = np.array([210, 210, 210])
    bgrs = scale_bgr_values((0.823, 210, 0.823))

    assert np.array_equal(expected_result, bgrs)

