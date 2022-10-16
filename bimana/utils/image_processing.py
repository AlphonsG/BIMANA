from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

MAX_PX_VAL = 255


def segment_image(
    img: npt.NDArray[Any],
    lower_bgr: npt.NDArray[np.int_],
    upper_bgr: npt.NDArray[np.int_],
) -> npt.NDArray[np.bool_]:
    """Segments an image.

    Converts a colour image to a binary image where pixels with BGR values
    between (inclusive) the specified upper and lower bounds are the foreground
    (with pixel values of 1).

    Args:
        img: A colour image.
        lower_bgr: The lower bound of BGR pixel values (all between 0 and 255)
            to segment.
        upper_bgr: The upper bound of BGR pixel values (all between 0 and 255)
            to segment.

    Returns:
        A binary image where pixels with the specified values in the original
        image are the foreground.
    """
    # threshold image to binary
    return (cv2.inRange(img, lower_bgr, upper_bgr).astype(
        float) / MAX_PX_VAL).astype(bool)


def scale_bgr_values(
    bgr: tuple[int | float, int | float, int | float],
) -> npt.NDArray[np.int_]:
    """Scales normalized BGR pixels from [0.0, 1.0] to [0, 255].

    Ignored individual pixel values that are already un-normalized (int).

    Args:
        bgr: A sequence of BGR pixel values (each between 0 and 255
            or normalized between 0.0 and 1.0).

    Returns:
        A sequence of un-normalized BGR pixel values (all between 0 and 255).
    """
    return np.array([round(val * MAX_PX_VAL) if isinstance(val, float)
                     else val for val in bgr])


def remove_isolated_segmented_objects(
    bin_img: npt.NDArray[np.bool_],
    threshold: float = 0.2,
) -> npt.NDArray[np.bool_]:
    """Removes isolated objects segmented in an image.

    Removes segmented objects in a binary image that are spatially isolated. A
    segmented object is isolated if the proportion of the number of foreground
    pixels along the width of the image at the height of the object's vertical
    centre to the total number of points along the width of the image with at
    least one foreground pixel at any height is below a threshold.

    Args:
        bin_img: A binary image.
        threshold: The sensitivity, between 0 and 1, of the algorithm to
            isolated objects. Larger values reduce the strictness of the
            criteria objects must meet to be classified as isolated.

    Returns:
        A binary image without isolated segmented objects.
    """
    # find all individual segmented objects
    cnts, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_CCOMP,
                               cv2.CHAIN_APPROX_NONE)

    # filter isolated objects
    num_nonzero_cols = np.count_nonzero(np.sum(bin_img, 0))
    summed_rows = np.sum(bin_img, 1)
    dens_of_rows = summed_rows / num_nonzero_cols
    iso_cnts = [cnt for cnt in cnts if dens_of_rows[cnt[:, 0, 1].min() + round(
        (cnt[:, 0, 1].max() - cnt[:, 0, 1].min()) / 2)] < threshold]

    # remove isolated objects from binary image
    refined_bin_img = bin_img.copy()

    return cv2.drawContours(refined_bin_img.astype(np.uint8), iso_cnts, -1, 0,
                            cv2.FILLED).astype(bool)


def bound_foreground(
    bin_img: npt.NDArray[np.bool_],
) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]]:
    """Vertically bounds the foreground across the width of a binary image.

    Generates an upper and lower polyline representing the coordinates of the
    uppermost and lowermost, respectively, foreground pixels across the width
    of the binary image.

    Args:
        bin_img: A binary image.

    Returns:
        The x-coordinates of the uppermost and lowermost foreground pixels
        across the image and the corresponding y-coordinates of the uppermost
        and lowermost foreground pixels across the image.
    """
    upper_lower_xs, upper_lower_ys = [], []
    for lower, (start, step) in enumerate([[0, 1], [
            bin_img.shape[0] - 1, -1]]):
        # get all x-coordinates along the width of the binary image with at
        # least one foreground pixel at any height
        xs = np.argwhere(np.any(bin_img, axis=0))[:, 0]

        # determine the y-coordinates of the uppermost/lowermost foreground
        # pixel for each x-coordinate
        ys = (bin_img[start:None:step, :] == 1).argmax(axis=0)
        ys = np.take(ys, xs)
        if lower:
            ys = bin_img.shape[0] - ys

        upper_lower_xs.append(xs)
        upper_lower_ys.append(ys)

    return upper_lower_xs, upper_lower_ys


def bounded_foreground_polyline(
    upper_lower_xs: list[npt.NDArray[Any]],
    upper_lower_ys: list[npt.NDArray[Any]],
) -> npt.NDArray[Any]:
    """Generates a polyline of the vertically bounded foreground in an image.

    Combines the x-coordinates of the uppermost and lowermost foreground pixels
    across a binary image and the corresponding y-coordinates of the uppermost
    and lowermost foreground pixels across the image into a single closed
    polyline enclosing all foreground pixels.

    Args:
        upper_lower_xs: The x-coordinates of the uppermost and lowermost
            foreground pixels across a binary image.
        upper_lower_ys: The y-coordinates of the uppermost and lowermost
            foreground pixels across a binary image.

    Returns:
        A polyline of xy-coordinates enclosing all foreground pixels in the
        source binary image.
    """
    xs = ys = np.array([], dtype=int)
    for lower, (curr_xs, curr_ys) in enumerate(zip(
            upper_lower_xs, upper_lower_ys)):
        if lower:
            curr_xs, curr_ys = curr_xs[::-1], curr_ys[::-1]

        xs, ys = np.concatenate([xs, curr_xs]), np.concatenate([ys, curr_ys])

    return np.array([[x, y] for x, y in zip(xs, ys)])


def smooth_foreground_bounds(
    bin_img: npt.NDArray[np.bool_],
    upper_lower_xs: list[npt.NDArray[Any]],
    upper_lower_ys: list[npt.NDArray[Any]],
) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]]:
    """Smooths the vertical bounds of the foreground across a binary image.

    Smoothing is achieved by removing sharp indents from the polylines
    vertically bounding the foreground across the width of the image.

    Args:
        bin_img: A binary image.
        upper_lower_xs: The x-coordinates of the uppermost and lowermost
            foreground pixels across the image.
        upper_lower_ys: The y-coordinates of the uppermost and lowermost
            foreground pixels across the image.

    Returns:
        The smoothed x-coordinates of the uppermost and lowermost foreground
        pixels across the image and the corresponding smoothed y-coordinates of
        the uppermost and lowermost foreground pixels across the image.
    """
    smthd_upper_lower_xs, smthd_upper_lower_ys = [], []
    for lower, (xs, ys) in enumerate(zip(upper_lower_xs, upper_lower_ys)):
        # convert outline to signal with peaks and use non-peaks as new outline
        signal = ys if lower else bin_img.shape[0] - ys
        if not (smthd_xs := find_peaks(signal)[0]).size:  # no peaks
            return upper_lower_xs, upper_lower_ys
        smthd_ys = np.take(ys, smthd_xs)

        # add first and last boundary coordinates again if removed
        if smthd_xs[0] != xs[0]:
            smthd_xs = np.insert(smthd_xs, 0, xs[0])
            smthd_ys = np.insert(smthd_ys, 0, ys[0])

        if smthd_xs[-1] != xs[-1]:
            smthd_xs = np.insert(smthd_xs, smthd_xs.shape[0], xs[-1])
            smthd_ys = np.insert(smthd_ys, smthd_ys.shape[0], ys[-1])

        smthd_upper_lower_xs.append(smthd_xs)
        smthd_upper_lower_ys.append(smthd_ys)

    smthd_bin_img = bin_img.astype(np.uint8)
    cnt = bounded_foreground_polyline(smthd_upper_lower_xs,
                                      smthd_upper_lower_ys)
    cv2.drawContours(smthd_bin_img, [cnt], 0, 1, cv2.FILLED)

    return bound_foreground(smthd_bin_img.astype(bool))


def segment_polyline_bounded_region(
    polyline: npt.NDArray[Any],
    hw: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Segments all pixels bounded by a closed polyline on a source image.

    Args:
        polyline: The xy-coordinates, with respect to a source image, of a
            polyline.
        hw: The height and width of the source image.

    Returns:
        A binary image with all pixels bounded by a closed polyline on a source
        mage segmented.
    """
    bin_img = np.zeros(hw)

    return cv2.drawContours(bin_img, [polyline], 0, 1, cv2.FILLED).astype(bool)


def segment_region_within_image(
    img: npt.NDArray[Any],
    lower_bgr: npt.NDArray[np.int_],
    upper_bgr: npt.NDArray[np.int_],
    region: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Segments a region within an image.

    Converts a colour image to a binary image where pixels, in a specified
    region within the image, with BGR values between (inclusive) the specified
    upper and lower bounds are the foreground (with pixel values of 1).

    Args:
        img: A colour image.
        lower_bgr: The lower bound of BGR pixel values (all between 0 and 255)
            to segment.
        upper_bgr: The upper bound of BGR pixel values (all between 0 and 255)
            to segment.
        region A binary image specifying a region within the original image.

    Returns:
        A binary image where pixels, in a specified region, with the specified
        values in the original image are the foreground.
    """
    # segment all pixels that fall within the bgr bounds in image
    bin_img = segment_image(img, lower_bgr, upper_bgr)
    bin_img[~region] = 0  # black out regions outside region

    return bin_img


def segment_region_above_polyline(
    polyline: npt.NDArray[Any],
    thickness: int,
    hw: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Segments the region above an open polyline on a source image.

    The segmented region will have an uppermost boundary that is the same shape
    as the initial polyline and will be of the specified thickness.

    Args:
        polyline: The xy-coordinates, with respect to a source image, of a
            polyline.
        thickness: The vertical thickness, in pixels, of the segmented region.
        hw: The height and width of the source image.

    Returns:
        A binary image with the region above an open polyline on the source
        image segmented.
    """
    # create a closed polyline of the region above the input polyline
    region_polyline = polyline[::-1].copy()
    region_polyline[:, 1] -= thickness
    region_polyline = np.append(polyline, region_polyline, 0)

    # segment region in the closed polyline
    bin_img = np.zeros(hw)

    return cv2.drawContours(bin_img, [region_polyline], 0, 1,
                            cv2.FILLED).astype(bool)


def draw_polyline(
    img: npt.NDArray[np.uint8],
    polyline: npt.NDArray[Any],
    colour: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draws a closed polyline on a colour image.

    Args:
        img: A colour BGR image.
        polyline: A polyline of xy-coordinates with respect to the image.
        colour: The BGR colour of the polyline to be drawn.
        thickness: The thickness of the polyline to be drawn.
        
    Returns:
        The colour image with the closed polyline drawn on.
    """
    return cv2.polylines(img.copy(), polyline[None, ...], True, colour,
                         thickness)


def binary_to_bgr_image(
    bin_img: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    """Converts a binary image to a colour image with 3 channels (BGR).

    Args:
        bin_img: A binary image.

    Returns:
        A colour image with 3 channels (BGR).
    """
    bgr_img = bin_img.astype(np.uint8) * MAX_PX_VAL

    return cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR).astype(np.uint8)
