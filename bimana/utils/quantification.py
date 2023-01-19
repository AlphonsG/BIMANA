import re
from math import pi, sqrt
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import describe


def percentage_foreground_in_region(
    img: npt.NDArray[np.bool_],
    region: npt.NDArray[np.bool_],
) -> tuple[float, int, int]:
    """Finds the percentage of a region occupied by the foreground in an image.

    Args:
        img: A binary image representing the primary image.
        region: A binary image specifying a region within the primary image.

    Returns:
        The percentage of the specified region within the primary image
        occupied by the foreground as well as the size of the corresponding
        foreground and region in pixels.
    """
    img = img.copy()
    img[~region] = 0
    foreground_count = np.count_nonzero(img)
    region_count = np.count_nonzero(region)
    pct = round(foreground_count / region_count * 100, 2)

    return pct, foreground_count, region_count


def nearest_neighbour_distances(
    pts: npt.NDArray[Any],
) -> npt.NDArray[np.float_]:
    """Finds the nearest neighbor distances for a given set of 2D points.

    Args:
        pts: An array of x and y coordinates for a set of 2D points.

    Returns:
        A sequence of distances, in pixels, corresponding to each input point's
        nearest neighbor.
    """
    tree = KDTree(pts)
    dists, _ = tree.query(pts, 2)

    return dists[:, 1]


def area(contours: list[npt.NDArray[Any]]) -> list[float]:
    """Calculates the area enclosed by each given contour.

    Args:
        contours: A sequence of contours each as an array of x and y
            coordinates.

    Returns:
        The areas enclosed by the given contours in pixels.
    """
    return [cv2.contourArea(c) for c in contours]


def perimeter(contours: list[npt.NDArray[Any]]) -> list[float]:
    """Calculates the perimeter of each given contour.

    Args:
        contours: A sequence of contours each as an array of x and y
            coordinates.

    Returns:
        The perimeters of the given contours in pixels.
    """
    return [cv2.arcLength(c, True) for c in contours]


def circularity(contours: list[npt.NDArray[Any]]) -> list[float]:
    """Calculates the circularity of each given contour.

    Args:
        contours: A sequence of contours each as an array of x and y
            coordinates.

    Returns:
        The circularities of the given contours.
    """
    return [(4 * pi * area([c])[0]) / perimeter([c])[0]**2 for c in contours]


def aspect_ratio(contours: list[npt.NDArray[Any]]) -> list[float]:
    """Calculates the aspect ratio of each given contour.

    Args:
        contours: A sequence of contours each as an array of x and y
            coordinates.

    Returns:
        The aspect ratios of the given contours.
    """
    aspect_ratios = []
    for c in contours:
        _, _, w, h = cv2.boundingRect(c)
        aspect_ratios.append(float(w) / h)

    return aspect_ratios


def solidity(contours: list[npt.NDArray[Any]]) -> list[float]:
    """Calculates the solidity of each given contour.

    Args:
        contours: A sequence of contours each as an array of x and y
            coordinates.

    Returns:
        The solidities of the given contours.
    """
    solidities = []
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidities.append(float(area) / hull_area)

    return solidities


def roundness(contours: list[npt.NDArray[Any]]) -> list[float]:
    """Calculates the roundness of each given contour.

    Args:
        contours: A sequence of contours each as an array of x and y
            coordinates.

    Returns:
        The roundness values of the given contours.
    """
    roundness_vals = []
    for c in contours:
        h = cv2.boundingRect(c)[3]
        area = cv2.contourArea(c)
        roundness_vals.append((4 * area) / (pi * h**2))

    return roundness_vals


def num_connected_components(img: npt.NDArray[np.bool_]) -> int:
    """Finds the number of connected foreground regions in a binary image.

    Args:
        img: A binary image.

    Returns:
        The number of connected foreground regions in the input binary image.
    """
    return cv2.connectedComponents(img.astype(np.uint8))[0] - 1


def size_connected_components(img: npt.NDArray[np.bool_]) -> npt.NDArray[Any]:
    """Finds the size of each connected foreground region in a binary image.

    Args:
        img: A binary image.

    Returns:
        The sizes, in pixels, of all connected foreground regions in the binary
        image.
    """
    stats = cv2.connectedComponentsWithStats(img.astype(np.uint8))[2]

    return stats[1:, cv2.CC_STAT_AREA]


def gen_stats(metric: list[Any] | npt.NDArray[Any]) -> dict[str, Any]:
    """Calculates statistics for a given metric.

    The statistics are the minimum value, maximum value, mean, standard
    deviation and variance.

    Args:
        metric: A sequence of values for a given metric.

    Returns:
        The statistics for the given metric.
    """
    stats = describe(metric, nan_policy='omit')

    return {'Min': stats[1][0], 'Max': stats[1][1], 'Mean (μ)': stats[2],
            'STD (σ)': sqrt(stats[3]), 'Var (σ²)': stats[3]}


def gen_histogram(
    data: dict[str, Any],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Generates a histogram of the input data in the output directory.

    Args:
        data: A dictionary of the data values and the name of the corresponding
            metric.
        output_dir: The output directory.
        dpi: The DPI of the histogram.
    """
    df = pd.DataFrame(data)
    ax = df.hist()[0][0]
    name = list(data.keys())[0]
    ax.set_title(' '.join((re.sub(
        r'\(.*\)', '', name) + ' Histogram').split()))
    ax.set_xlabel(name)
    ax.set_ylabel('Frequency')
    output_path = output_dir / (' '.join((re.sub(
        r'\(.*\)', '', name) + ' histogram').split())).lower()
    ax.get_figure().savefig(output_path, dpi=dpi)
    plt.close()
