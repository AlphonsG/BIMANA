from enum import Enum
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks
from scipy.spatial import KDTree

MAX_PX_VAL = 255
KSIZE = 5
BLOCKSIZE = 11
C = 2
DP = 1


class ImageChannel(Enum):
    """Get_channel flags.

    BLUE: If set, extracts the blue channel.
    GREEN: If set, extracts the green channel.
    RED: If set, extracts the red channel.
    """

    BLUE = 1
    GREEN = 2
    RED = 3


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
    cnts, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_NONE)

    # filter isolated objects
    num_nonzero_cols = np.count_nonzero(np.sum(bin_img, 0))
    summed_rows = np.sum(bin_img, 1)
    dens_of_rows = summed_rows / num_nonzero_cols
    iso_cnts = [cnt for cnt in cnts if dens_of_rows[cnt[:, 0, 1].min() + round(
        (cnt[:, 0, 1].max() - cnt[:, 0, 1].min()) / 2)] < threshold]

    # remove isolated objects from binary image
    fltd_img = bin_img.astype(np.uint8)
    cv2.drawContours(fltd_img, iso_cnts, -1, 0, cv2.FILLED)

    cnts, _ = cv2.findContours(fltd_img, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    fin_cnts, pts = [], []
    area1, dist, area2 = 50 * threshold, 300 * threshold, 1062.5 * threshold
    for i, c in enumerate(cnts):
        if cv2.contourArea(c) > area1:
            x, y, w, h = cv2.boundingRect(c)
            pts.append((x + w / 2, y + h / 2))
            fin_cnts.append(c)

    tree = KDTree(pts)
    _, xs = tree.query(pts, 3, distance_upper_bound=dist)
    fin_img = np.zeros(bin_img.shape).astype(np.uint8)
    for i, x in enumerate(xs):
        if cv2.contourArea(fin_cnts[i]) >= area2:
            cv2.drawContours(fin_img, [fin_cnts[i]], -1, 1, cv2.FILLED)
        if len(pts) in x:
            continue
        cv2.drawContours(fin_img, [fin_cnts[i]], -1, 1, cv2.FILLED)

    return fin_img.astype(bool)


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
            ys = bin_img.shape[0] - ys - 1

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
    smthd_upper_lower_xs = upper_lower_xs.copy()
    smthd_upper_lower_ys = upper_lower_ys.copy()
    for lower, (xs, ys) in enumerate(zip(upper_lower_xs, upper_lower_ys)):
        # convert outline to signal with peaks and use non-peaks as new
        # outline
        if lower:
            if len(np.unique(ys)) == 1:
                continue
            polyline = np.array([[x, y] for x, y in zip(xs, ys)])
            polyline = cv2.convexHull(polyline)
            smthd_xs = polyline[..., 0].flatten()
            smthd_ys = polyline[..., 1].flatten()
        else:
            signal = ys if lower else bin_img.shape[0] - ys
            if not (smthd_xs := find_peaks(signal)[0]).size:  # no peaks
                continue
            smthd_ys = np.take(ys, smthd_xs)

            # add first and last boundary coordinates again if removed
            if smthd_xs[0] != xs[0]:
                smthd_xs = np.insert(smthd_xs, 0, xs[0])
                smthd_ys = np.insert(smthd_ys, 0, ys[0])

            if smthd_xs[-1] != xs[-1]:
                smthd_xs = np.insert(smthd_xs, smthd_xs.shape[0], xs[-1])
                smthd_ys = np.insert(smthd_ys, smthd_ys.shape[0], ys[-1])

        smthd_upper_lower_xs[lower] = smthd_xs
        smthd_upper_lower_ys[lower] = smthd_ys

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


def extract_channel(
    img: npt.NDArray[np.uint8],
    flag: ImageChannel,
) -> npt.NDArray[np.uint8]:
    """Extracts the chosen colour channel from an image.

    Args:
        img: A colour image.
        flag: A flag that can take values of ImageChannel.

    Returns:
        A 2D image of the extracted channel.
    """
    return cv2.split(img)[flag.value - 1]


def auto_segment_image(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Automatically segments the foreground in a greyscale image.

    Args:
        img: A greyscale image.

    Returns:
        A binary image of the segmented foreground from the input image.
    """
    img = cv2.medianBlur(~img, KSIZE)
    bin_img = cv2.adaptiveThreshold(img, MAX_PX_VAL,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, BLOCKSIZE, C)

    return ~(bin_img / MAX_PX_VAL).astype(bool)


def detect_circular_objects(
    img: npt.NDArray[np.uint8],
    min_dist: int = 12,
    alpha: int = 10,
    beta: int = 14,
    min_radius: int = 5,
    max_radius: int = 15,
    min_obj_area: int = 20,
    bgd_px_thresh: int = 20,
) -> list[npt.NDArray[Any]]:
    """Detects circular objects in an image.

    Args:
        img: A greyscale image.
        min_dist: The minimum distance between the centers of the circular
            objects in pixels.
        alpha: A parameter that can be fine tuned for circular object
            detection performance. It may be decreased to detect more objects.
        beta: A parameter that can be fine tuned for circular object
            detection performance. The smaller it is, the more false objects
            may be detected.
        min_radius: The minimum radii of circular objects in pixels.
        max_radius: The maximum radii of circular objects in pixels.
        min_obj_area: The minimum area of circular objects.
        bgd_px_thresh: The maximum value of background pixels in the image.

    Returns:
        A sequence of coordinates for circular objects detected in the image.
    """
    img = cv2.medianBlur(img, KSIZE)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, DP, min_dist,
                               param1=alpha, param2=beta, minRadius=min_radius,
                               maxRadius=max_radius)
    objs = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            mask = np.ones(img.shape)
            cv2.circle(mask, (c[0], c[1]), c[2], 0, cv2.FILLED)
            mask = mask.astype(bool)
            roi = img.copy()
            roi[mask] = 0
            roi[roi > bgd_px_thresh] = 255
            roi[roi <= bgd_px_thresh] = 0
            cts, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)
            objs += [ct for ct in cts if cv2.contourArea(ct) > min_obj_area]

    return objs


def centre_coordinates(contours: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    """Gets the x and y coordinates of the centres of the provided contours.

    Args:
        contours: A sequence of contours where each contour is an array of
            x and y coordinates.

    Returns:
        A sequence of the x and y coordinates of the centres of the respective
        input contours.
    """
    centres = []
    for c in contours:
        m = cv2.moments(c)
        x = int(m['m10'] / m['m00'])
        y = int(m['m01'] / m['m00'])
        centres.append((x, y))

    return np.array(centres)


def subimage_coordinates(
    img: npt.NDArray[Any],
    background_bgr: tuple[int, int, int] = (255, 255, 255),
) -> npt.NDArray[Any]:
    """Finds the coordinates of sub-images in a larger image.

    Args:
        img: A colour image.
        background_bgr: The BGR value of background pixels in the larger image.

    Returns:
        An array of the (x1, y1, x2, y2) coordinates of the bounding boxes for
        all sub-images in the larger image.
    """
    bin_img = ~segment_image(img, np.array(background_bgr),
                             np.array(background_bgr))
    contours, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    coords = []
    for c in contours:
        x1, x2 = c[:, 0, 0].min(), c[:, 0, 0].max()
        y1, y2 = c[:, 0, 1].min(), c[:, 0, 1].max()
        coords.append((x1, y1, x2, y2))

    return np.array(coords)


def largest_objects(
    bin_img: npt.NDArray[np.bool_],
    min_area: int | None = None,
    top_n: int | None = None,
) -> tuple[list[list[npt.NDArray[Any]]], list[list[npt.NDArray[Any]]]]:
    """Finds the contours of the largest foreground objects in a binary image.

    Large objects are those with at least the given area and/or those that are
    the top n largest objects in the image.

    Args:
        bin_img: A binary image.
        min_area: The minimum object area.
        top_n: The top number of largest objects to find.

    Returns:
        The contours of the largest foreground objects in the input image as a
        sequence of x-coordinates for the upper object boundary(s) across the
        width of the image and the corresponding y-coordinates of lower object
        boundary(s) across the image
    """
    cnts, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_NONE)

    if min_area is not None:
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if top_n is not None:
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c),
                      reverse=True)[:top_n]

    upper_lower_xs, upper_lower_ys = [], []
    for c in cnts:
        l, r = c[..., 0].argmin(), c[..., 0].argmax()
        if l > r:
            l, r, c = len(c) - l, len(c) - r, c[::-1]
        l_xs, l_ys = c[l:r + 1][..., 0][:, 0], c[l:r + 1][..., 1][:, 0]
        u_xs_ys = np.append(c[0:l][::-1], c[r + 1:][::-1], 0)
        u_xs, u_ys = u_xs_ys[..., 0][:, 0], u_xs_ys[..., 1][:, 0]
        upper_lower_xs.append([u_xs, l_xs])
        upper_lower_ys.append([u_ys, l_ys])

    return upper_lower_xs, upper_lower_ys
