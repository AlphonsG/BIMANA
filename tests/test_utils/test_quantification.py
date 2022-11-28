from pathlib import Path

import numpy as np

from bimana.utils.quantification import (area, size_connected_components,
                                         aspect_ratio, circularity,
                                         gen_histogram, gen_stats,
                                         nearest_neighbour_distances,
                                         num_connected_components,
                                         percentage_foreground_in_region,
                                         perimeter, roundness, solidity)


def test_percentage_foreground_in_region() -> None:
    hw = (10, 20)

    img = np.zeros(hw, bool)
    img[5:, 10:] = 1

    region = np.zeros(hw, bool)
    region[5:, :] = 1

    pct, fg_ct, reg_cnt = percentage_foreground_in_region(img, region)
    exp_pct, exp_fg_cnt, exp_reg_cnt = 50, 50, 100

    assert (pct, fg_ct, reg_cnt) == (exp_pct, exp_fg_cnt, exp_reg_cnt)


def test_nearest_neighbour_distances() -> None:
    pts = np.array([[3, 4], [6, 8]])
    dists = nearest_neighbour_distances(pts)

    assert np.all(dists == 5)


def test_area() -> None:
    contours = [np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]])]
    cnt_areas = area(contours)

    assert cnt_areas[0] == 100


def test_perimeter() -> None:
    contours = [np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]])]
    cnt_perimeters = perimeter(contours)

    assert cnt_perimeters[0] == 40


def test_circularity() -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 5
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    contours = [np.array([[[x, y]] for x, y in zip(xs, ys)], dtype=np.float32)]

    cnt_circularity = circularity(contours)

    assert round(cnt_circularity[0], 3) == 1


def test_aspect_ratio() -> None:
    contours = [np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]])]
    cnt_aspect_ratios = aspect_ratio(contours)

    assert cnt_aspect_ratios[0] == 1


def test_solidity() -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 5
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    contours = [np.array([[[x, y]] for x, y in zip(xs, ys)], dtype=np.float32)]

    cnt_solidity = solidity(contours)

    assert cnt_solidity[0] == 1


def test_roundness() -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 5
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    contours = [np.array([[[x, y]] for x, y in zip(xs, ys)], dtype=np.float32)]

    cnt_roundness = roundness(contours)

    assert round(cnt_roundness[0], 2) == 1


def test_num_connected_components() -> None:
    hw = (20, 20)
    img = np.zeros(hw, bool)
    img[10, 10] = img[11, 11] = img[12, 12] = 1
    img[5, 5] = img[5, 6] = img[5, 7] = 1

    num_cnctd_comps = num_connected_components(img)

    assert num_cnctd_comps == 2


def test_size_connected_components() -> None:
    hw = (20, 20)
    img = np.zeros(hw, bool)
    img[10, 10] = img[11, 11] = img[12, 12] = 1
    img[5, 5] = img[5, 6] = img[5, 7] = 1

    size_cnctd_comps = size_connected_components(img)

    assert np.all(size_cnctd_comps == 3)


def test_gen_stats() -> None:
    data = [1, 2, 3]

    metrics = gen_stats(data)

    assert list(metrics.values()) == [1, 3, 2, 1, 1]


def test_gen_histogram(tmp_path: Path) -> None:
    data = {'test': [8, 5, 3, 5, 7, 5, 3, 5, 7, 5]}

    gen_histogram(data, tmp_path)

    files = [f for f in tmp_path.iterdir()]

    assert len(files) == 1
    assert files[0].is_file()
