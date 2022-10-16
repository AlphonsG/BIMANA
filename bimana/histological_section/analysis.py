from typing import Any

import numpy as np
import numpy.typing as npt
from bimana.utils.image_processing import (binary_to_bgr_image,
                                           bound_foreground,
                                           bounded_foreground_polyline,
                                           draw_polyline,
                                           remove_isolated_segmented_objects,
                                           segment_image,
                                           segment_polyline_bounded_region,
                                           segment_region_above_polyline,
                                           segment_region_within_image,
                                           smooth_foreground_bounds)
from bimana.utils.quantification import percentage_foreground_in_region

BIN_IMG_KEY = 'segmented_tissue'
BIN_IMG_NO_ISO_KEY = 'segmented_tissue_without_isolated_non_tissue'
POLYLINE_KEY = 'tissue_boundary'
SMOOTHED_POLYLINE_KEY = 'smoothed_tissue_boundary'
SEGD_FG_IN_RGN_KEY = 'staining_within_tissue_boundary'
SEGD_FG_ABOVE_RGN_KEY = 'cilia_above_tissue_area'


def tissue_boundary(
    img: npt.NDArray[Any],
    non_tissue_lower_bgr: npt.NDArray[np.int_],
    non_tissue_upper_bgr: npt.NDArray[np.int_],
    iso_non_tissue_sens: float | None = 0.2,
    smooth_segmented_tissue_boundary: bool = True,
) -> tuple[tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]], list[tuple[
        str, npt.NDArray[np.uint8]]]]:
    """Determines the tissue boundary in an image of a histological section.

    Args:
        img: A colour histological section image.
        non_tissue_lower_bgr: The lower bound of BGR pixel values (all between
            0 and 255) corresponding to non-tissue that will be segmented out.
        non_tissue_upper_bgr: The upper bound of BGR pixel values (all between
            0 and 255) corresponding to non-tissue that will be segmented out.
        iso_non_tissue_sens: The sensitivity, between 0 and 1, of the algorithm
            that identifies tissue to isolated non-tissue objects. Larger
            values reduce the strictness of the criteria objects must meet to
            be classified as isolated and thus ignored as non-tissue objects.
            If None, will not attempt to detect and ignore isolated non-tissue
            objects.
        smooth_segmented_tissue_boundary: If True, smooths the generated
            boundary of the tissue area.

    Returns:
        The x-coordinates of the upper tissue boundary across the width of the
        image and the corresponding y-coordinates of lower tissue boundary
        across the image. Also returns a dictionary of the intermediate images
        generated during processing.
    """
    procd_imgs = []

    # segment tissue
    bin_img = ~segment_image(img, non_tissue_lower_bgr, non_tissue_upper_bgr)
    bgr_img = binary_to_bgr_image(bin_img)
    procd_imgs.append((BIN_IMG_KEY, bgr_img))

    if iso_non_tissue_sens is not None:
        bin_img = remove_isolated_segmented_objects(
            bin_img, iso_non_tissue_sens)
        bgr_img = binary_to_bgr_image(bin_img)
        procd_imgs.append((BIN_IMG_NO_ISO_KEY, bgr_img))

    # determine tissue boundary
    upper_lower_xs, upper_lower_ys = bound_foreground(bin_img)
    polyline = bounded_foreground_polyline(upper_lower_xs, upper_lower_ys)
    procd_imgs.append((POLYLINE_KEY, draw_polyline(bgr_img, polyline)))

    if smooth_segmented_tissue_boundary:
        upper_lower_xs, upper_lower_ys = smooth_foreground_bounds(
            bin_img, upper_lower_xs, upper_lower_ys)
        polyline = bounded_foreground_polyline(upper_lower_xs, upper_lower_ys)
        procd_imgs.append((SMOOTHED_POLYLINE_KEY,
                           draw_polyline(bgr_img, polyline)))

    return (upper_lower_xs, upper_lower_ys), procd_imgs


def amount_staining_in_tissue_area(
    img: npt.NDArray[Any],
    upper_lower_xs: list[npt.NDArray[Any]],
    upper_lower_ys: list[npt.NDArray[Any]],
    stain_lower_bgr: npt.NDArray[np.int_],
    stain_upper_bgr: npt.NDArray[np.int_],
) -> tuple[float, int, int, list[tuple[str, npt.NDArray[np.uint8]]]]:
    """Finds how much of a histological section image's tissue area is stained.

    Args:
        img: A colour histological section image.
        upper_lower_xs: The x-coordinates of the upper and lower tissue
            boundary across the image.
        upper_lower_ys: The y-coordinates of the upper and lower tissue
            boundary across the image.
        stain_lower_bgr: The lower bound of BGR pixel values (all between
            0 and 255) corresponding to staining.
        stain_upper_bgr: The upper bound of BGR pixel values (all between
            0 and 255) corresponding to staining.

    Returns:
        The percentage of a histological section image's tissue area that is
        stained, along with the corresponding amount of staining and size of
        the tissue area (in pixels), followed by a sequence of the intermediate
        images generated during processing.
    """
    # calculate percentage of staining in tissue region
    polyline = bounded_foreground_polyline(upper_lower_xs, upper_lower_ys)
    tissue_mask = segment_polyline_bounded_region(polyline, img.shape[:2])
    stain_mask = segment_region_within_image(img, stain_lower_bgr,
                                             stain_upper_bgr, tissue_mask)
    pct_stain, stain_amt, tiss_size = percentage_foreground_in_region(
        stain_mask, tissue_mask)

    # visualize masked staining
    procd_imgs = []
    procd_img = img.copy()
    procd_img[~stain_mask] = (0, 0, 0)
    upper_xs, lower_xs = bound_foreground(tissue_mask)
    polyline = bounded_foreground_polyline(upper_xs, lower_xs)
    procd_imgs.append((SEGD_FG_IN_RGN_KEY, draw_polyline(procd_img, polyline)))

    return pct_stain, stain_amt, tiss_size, procd_imgs


def amount_cilia_above_tissue_area(
    img: npt.NDArray[Any],
    upper_lower_xs: list[npt.NDArray[Any]],
    upper_lower_ys: list[npt.NDArray[Any]],
    cilia_lower_bgr: npt.NDArray[np.int_],
    cilia_upper_bgr: npt.NDArray[np.int_],
    thickness: int,
) -> tuple[float, int, int, list[tuple[str, npt.NDArray[np.uint8]]]]:
    """Finds amount of cilia above a histological section image's tissue area.

    Args:
        img: A colour histological section image.
        upper_xs: The x-coordinates of the upper and lower tissue boundary
            across the image.
        upper_ys: The y-coordinates of the upper and lower tissue boundary
            across the image.
        cilia_lower_bgr: The lower bound of BGR pixel values (all between
            0 and 255) corresponding to cilia.
        cilia_upper_bgr: The upper bound of BGR pixel values (all between
            0 and 255) corresponding to corresponding to cilia.
        thickness: The expected vertical thickness, in pixels, of the
            cilia-containing region above the tissue area.

    Returns:
        The percentage of the region above the tissue area of an image of a
        histological section that is occupied by cilia, along with the
        corresponding amount of cilia and size of the cilia-containing region
        (in pixels), followed by a sequence of the intermediate images
        generated during processing.
    """
    # calculate percentage of cilia tissue region
    polyline = np.array([[x, y] for x, y in zip(upper_lower_xs[0],
                                                upper_lower_ys[0])])
    cilia_region_mask = segment_region_above_polyline(polyline, thickness,
                                                      img.shape[:2])
    cilia_mask = segment_region_within_image(
        img, cilia_lower_bgr, cilia_upper_bgr, cilia_region_mask)
    pct_cilia, cilia_amt, reg_size = percentage_foreground_in_region(
        cilia_mask, cilia_region_mask)

    # visualize masked cilia
    procd_imgs = []
    procd_img = img.copy()
    procd_img[~cilia_mask] = (0, 0, 0)

    # draw tissue boundary
    polyline = bounded_foreground_polyline(upper_lower_xs, upper_lower_ys)
    procd_img = draw_polyline(procd_img, polyline)

    # draw cilia region
    upper_lower_xs, upper_lower_ys = bound_foreground(cilia_region_mask)
    polyline = bounded_foreground_polyline(upper_lower_xs, upper_lower_ys)
    procd_img = draw_polyline(procd_img, polyline)

    procd_imgs.append((SEGD_FG_ABOVE_RGN_KEY, procd_img))

    return pct_cilia, cilia_amt, reg_size, procd_imgs


def save_metrics():
    pass
