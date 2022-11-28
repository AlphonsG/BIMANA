from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from bimana.utils.deep_learning.dexined import detect_edges
from bimana.utils.image_processing import (ImageChannel, auto_segment_image,
                                           binary_to_bgr_image,
                                           detect_circular_objects,
                                           extract_channel)


def extract_tight_junctions(
    img: npt.NDArray[Any],
    tight_junction_channel: ImageChannel,
) -> tuple[npt.NDArray[np.bool_], list[tuple[str, npt.NDArray[np.uint8]]]]:
    """Extracts stained tight junctions present in a fluorescent image.

    Args:
        img: A colour image.
        tight_junction_channel: The image channel with the stained tight
            junctions.

    Returns:
        A binary image where the stained tight junctions present in the
        fluorescent image are the foreground. Also returns a sequence of the
        intermediate images generated during processing.
    """
    grey = extract_channel(img, tight_junction_channel)
    edges = detect_edges(grey)
    bin_img = ~auto_segment_image(edges)
    tight_juncs = grey.copy()
    tight_juncs[bin_img] = 0

    procd_imgs = []
    procd_imgs.append(('tight junctions image channel', grey))
    procd_imgs.append(('tight junctions detected edges', edges))
    procd_imgs.append(('tight junctions binary mask',
                       binary_to_bgr_image(bin_img)))
    procd_imgs.append(('tight junctions', tight_juncs))

    return bin_img, procd_imgs


def segment_cells(
    img: npt.NDArray[Any],
    cell_channel: ImageChannel,
    min_dist: int = 12,
    alpha: int = 10,
    beta: int = 14,
    min_radius: int = 5,
    max_radius: int = 15,
    min_obj_area: int = 20,
    bgd_px_thresh: int = 20,
) -> tuple[list[npt.NDArray[Any]], list[tuple[str, npt.NDArray[np.uint8]]]]:
    """Segments stained circular cells present in a fluorescent image.

    Args:
        img: A colour image.
        cell_channel: The image channel with the stained cells.
        min_dist: The minimum distance between the centers of the cells in
            pixels.
        alpha: A parameter that can be fine tuned for cell segmentation
            performance. It may be decreased to identify more objects.
        beta: A parameter that can be fine tuned for cell segmentation
            performance. The smaller it is, the more false cells may be
            identified.
        min_radius: The minimum radius of cells in pixels.
        max_radius: The maximum radius of cells in pixels.
        min_obj_area: The minimum area of cells in pixels².
        bgd_px_thresh: The maximum value of background pixels in the image.

    Returns:
        A binary image where the stained tight junctions present in the
        fluorescent image are the foreground. Also returns a dictionary of the
        intermediate images generated during processing.
    """
    grey = extract_channel(img, cell_channel)
    cells = detect_circular_objects(grey, min_dist, alpha, beta, min_radius,
                                    max_radius, min_obj_area, bgd_px_thresh)

    procd_imgs = []
    procd_imgs.append(('cells_image_channel', grey))
    cell_img = img.copy()
    cv2.drawContours(cell_img, cells, -1, (0, 0, 255), 2)
    procd_imgs.append(('cells', cell_img))

    return cells, procd_imgs


NUM_LINES = 10


def perform_ijoq(bin_img: npt.NDArray[np.bool_]) -> float:
    """Perform intercellular junction organization quantification (IJOQ).

    IJOQ: https://www.frontiersin.org/articles/10.3389/fcimb.2022.865528.
    Code adapted from: https://github.com/DevonsMo/IJOQ.

    Args:
        bin_img: A binary image containing tight junctions as the foreground.

    Returns:
        The IJOQ value in the original image.
    """
    cell_border_frequency = 0
    for axis in range(2):
        for i in range(NUM_LINES):
            pixel = round((i + 0.5) * bin_img.shape[axis] / NUM_LINES)
            previous_pixel = bin_img[0, pixel] if axis else bin_img[pixel, 0]
            for j in range(1, bin_img.shape[1 - axis]):
                current_pixel = (bin_img[j, pixel] if axis else
                                 bin_img[pixel, j])
                # if the line detects a color change (i.e. black to white or
                # white to black)
                if previous_pixel != current_pixel:
                    cell_border_frequency += 0.5 / bin_img.shape[1 - axis]

                # set current pixel as the previous pixel before moving to the
                # next pixel
                previous_pixel = current_pixel

    # take average of all lines
    return round(cell_border_frequency / (2 * NUM_LINES), 4)
