from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from bimana.utils.file_processing import crop_and_save_images, load_imgs
from bimana.utils.image_processing import (binary_to_bgr_image,
                                           bound_foreground,
                                           bounded_foreground_polyline,
                                           draw_polyline, largest_objects,
                                           remove_isolated_segmented_objects,
                                           segment_image,
                                           segment_polyline_bounded_region,
                                           segment_region_above_polyline,
                                           segment_region_within_image,
                                           smooth_foreground_bounds,
                                           subimage_coordinates)
from bimana.utils.quantification import percentage_foreground_in_region

BIN_IMG_KEY = 'binary_image'
BIN_IMG_NO_ISO_KEY = 'binary_image_without_isolated_non_tissue'
POLYLINE_KEY = 'raw_tissue_boundary'
SMOOTHED_POLYLINE_KEY = 'smoothed_tissue_boundary'
POLYLINE_ORI_KEY = 'tissue_boundary_on_original'
SEGD_FG_IN_RGN_KEY = 'staining_within_tissue_boundary'
SEGD_FG_ABOVE_RGN_KEY = 'cilia_above_tissue_area'
SEGD_FG_ABOVE_RGN_ORI_KEY = 'cilia_region_on_original'
FIN_BIN_IMG_KEY = 'segmented_tissue'


def tissue_boundary(
    img: npt.NDArray[Any],
    non_tissue_lower_bgr: npt.NDArray[np.int_],
    non_tissue_upper_bgr: npt.NDArray[np.int_],
    selected_tissue: str = 'all',
    top: int | None = None,
    minimum: int | None = None,
    iso_non_tissue_sens: float | None = 0.2,
    smooth_segmented_tissue_boundary: bool = True,
) -> tuple[tuple[list[list[npt.NDArray[Any]]], list[list[npt.NDArray[Any]]]],
           list[tuple[str, npt.NDArray[np.uint8]]]]:
    """Determines the tissue boundary in an image of a histological section.

    Args:
        img: A colour histological section image.
        non_tissue_lower_bgr: The lower bound of BGR pixel values (all between
            0 and 255) corresponding to non-tissue that will be segmented out.
        non_tissue_upper_bgr: The upper bound of BGR pixel values (all between
            0 and 255) corresponding to non-tissue that will be segmented out.
        selected_tissue: What to treat as tissue after binarizing the image -
            if set to all, creates a bounding polyline around all foreground
            pixels in the image and treats the inner area as tissue; if set to
            largest, treats only the largest group(s) of connected foreground
            pixels (objects) in the image as tissue.
        top: The number of top largest objects in the image to treat as
            tissue, when the selected_tissue argument is set to 'largest'.
        minimum: The minimum area, in pixels, for objects in the image to be
            treated as tissue, when the selected_tissue argument is set to
            'largest'.
        iso_non_tissue_sens: The sensitivity, between 0 and 1, of the algorithm
            that identifies tissue to isolated non-tissue objects. Larger
            values reduce the strictness of the criteria objects must meet to
            be classified as isolated and thus ignored as non-tissue objects.
            If None, will not attempt to detect and ignore isolated non-tissue
            objects.
        smooth_segmented_tissue_boundary: If True, smooths the generated
            boundary of the tissue area.

    Returns:
        A sequence of x-coordinates for the upper tissue boundary(s) across the
        width of the image and the corresponding y-coordinates of lower tissue
        boundary(s) across the image. Also returns a sequence of the
        intermediate images generated during processing.
    """
    procd_imgs = []

    # segment image
    bin_img = ~segment_image(img, non_tissue_lower_bgr, non_tissue_upper_bgr)
    bgr_img = binary_to_bgr_image(bin_img)
    procd_imgs.append((BIN_IMG_KEY, bgr_img))

    if iso_non_tissue_sens is not None:
        bin_img = remove_isolated_segmented_objects(
            bin_img, iso_non_tissue_sens)
        bgr_img = binary_to_bgr_image(bin_img)
        procd_imgs.append((BIN_IMG_NO_ISO_KEY, bgr_img))

    # determine tissue boundary
    upper_lower_xs, upper_lower_ys = largest_objects(bin_img, minimum, top) \
        if selected_tissue == 'largest' else bound_foreground(bin_img)

    if not isinstance(upper_lower_xs[0], list):
        upper_lower_xs, upper_lower_ys = [upper_lower_xs], [upper_lower_ys]

    tiss_img = np.zeros(bin_img.shape, bool)
    polylines = []
    for i, (ul_xs, ul_ys) in enumerate(zip(upper_lower_xs, upper_lower_ys)):
        polyline = bounded_foreground_polyline(ul_xs, ul_ys)
        tiss_img |= segment_polyline_bounded_region(polyline, bin_img.shape)
        polylines.append(polyline)

    bgr_img = binary_to_bgr_image(tiss_img)
    procd_imgs.append((FIN_BIN_IMG_KEY, bgr_img))
    smd_polyline_img = polyline_img = bgr_img

    for i, (ul_xs, ul_ys) in enumerate(zip(upper_lower_xs, upper_lower_ys)):
        polyline = polylines[i]
        polyline_img = draw_polyline(polyline_img, polyline)

        if smooth_segmented_tissue_boundary:
            ul_xs, ul_ys = smooth_foreground_bounds(bin_img, ul_xs, ul_ys)
            polyline = bounded_foreground_polyline(ul_xs, ul_ys)
            smd_polyline_img = draw_polyline(smd_polyline_img, polyline)
            upper_lower_xs[i], upper_lower_ys[i] = ul_xs, ul_ys

        img = draw_polyline(img, polyline)

    procd_imgs.append((POLYLINE_KEY, polyline_img))
    procd_imgs.append((POLYLINE_ORI_KEY, img))
    if smooth_segmented_tissue_boundary:
        procd_imgs.append((SMOOTHED_POLYLINE_KEY, smd_polyline_img))

    return (upper_lower_xs, upper_lower_ys), procd_imgs


def amount_staining_in_tissue_area(
    img: npt.NDArray[Any],
    upper_lower_xs: list[list[npt.NDArray[Any]]],
    upper_lower_ys: list[list[npt.NDArray[Any]]],
    stain_lower_bgr: npt.NDArray[np.int_],
    stain_upper_bgr: npt.NDArray[np.int_],
) -> tuple[float, int, int, list[tuple[str, npt.NDArray[np.uint8]]]]:
    """Finds how much of a histological section image's tissue area is stained.

    Args:
        img: A colour histological section image.
        upper_lower_xs: A sequence of the x-coordinates of the upper and lower
            tissue boundary(s) across the image.
        upper_lower_ys: A sequence of the y-coordinates of the upper and lower
            tissue boundary(s) across the image.
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
    tissue_mask = np.zeros(img.shape[:2], bool)
    procd_img = img.copy()
    polylines = []
    for ul_xs, ul_ys in zip(upper_lower_xs, upper_lower_ys):
        polyline = bounded_foreground_polyline(ul_xs, ul_ys)
        tissue_mask = tissue_mask | segment_polyline_bounded_region(
            polyline, img.shape[:2])
        polylines.append(polyline)

    stain_mask = segment_region_within_image(img, stain_lower_bgr,
                                             stain_upper_bgr, tissue_mask)
    pct_stain, stain_amt, tiss_size = percentage_foreground_in_region(
        stain_mask, tissue_mask)

    procd_img[~stain_mask] = (0, 0, 0)
    for polyline in polylines:
        procd_img = draw_polyline(procd_img, polyline)

    return pct_stain, stain_amt, tiss_size, [(SEGD_FG_IN_RGN_KEY, procd_img)]


def amount_cilia_above_tissue_area(
    img: npt.NDArray[Any],
    upper_lower_xs: list[list[npt.NDArray[Any]]],
    upper_lower_ys: list[list[npt.NDArray[Any]]],
    cilia_lower_bgr: npt.NDArray[np.int_],
    cilia_upper_bgr: npt.NDArray[np.int_],
    thickness: int,
) -> tuple[float, int, int, list[tuple[str, npt.NDArray[np.uint8]]]]:
    """Finds amount of cilia above a histological section image's tissue area.

    Args:
        img: A colour histological section image.
        upper_xs: A sequence of the x-coordinates of the upper and lower tissue
            boundary(s) across the image.
        upper_ys: A sequence of the y-coordinates of the upper and lower tissue
            boundary(s) across the image.
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
    # calculate percentage of cilia above tissue region
    cilia_region_mask = np.zeros(img.shape[:2], bool)
    procd_img = img.copy()
    polylines = []
    for ul_xs, ul_ys in zip(upper_lower_xs, upper_lower_ys):
        polyline = np.array([[x, y] for x, y in zip(ul_xs[0], ul_ys[0])])
        mask = segment_region_above_polyline(
            polyline, thickness, img.shape[:2])
        cilia_region_mask |= mask

        reg_coords_xs, reg_coords_ys = largest_objects(mask)
        for rc_xs, rc_ys in zip(reg_coords_xs, reg_coords_ys):
            polyline = bounded_foreground_polyline(rc_xs, rc_ys)
            polylines.append(polyline)

    cilia_mask = segment_region_within_image(
        img, cilia_lower_bgr, cilia_upper_bgr, cilia_region_mask)
    pct_cilia, cilia_amt, reg_size = percentage_foreground_in_region(
        cilia_mask, cilia_region_mask)

    procd_img[~cilia_mask] = (0, 0, 0)

    for polyline in polylines:
        procd_img = draw_polyline(procd_img, polyline)

    return pct_cilia, cilia_amt, reg_size, [(SEGD_FG_ABOVE_RGN_KEY, procd_img)]


def crop_and_save_tissue_sections(
    input_dir: Path,
    shrink_factor: int | None = 1000,
) -> list[Path]:
    """Crops and saves tissue sections from images in the input directory.

    Args:
        input_dir: The input directory.
        shrink_factor: Factor to shrink images by before loading them into
            memory.

    Returns:
        The directories containing the saved tissue sections. Each directory
        contains one tissue section image.
    """
    imgs = load_imgs(input_dir, shrink_factor)
    output_paths = []
    for f, img in imgs:
        tissue_section_coords = subimage_coordinates(img)
        tissue_section_coords *= shrink_factor
        output_paths += crop_and_save_images(f, tissue_section_coords, True)

    return output_paths
