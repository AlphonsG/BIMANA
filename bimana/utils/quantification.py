import numpy as np
import numpy.typing as npt


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
    pct = round(foreground_count / region_count, 2) * 100

    return pct, foreground_count, region_count
