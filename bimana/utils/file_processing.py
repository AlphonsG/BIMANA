import os
import platform
import struct
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '1000000000000'

import cv2
import numpy.typing as npt
import pandas as pd
from natsort import natsorted

# try to import vips
if platform.system() == 'Windows' and struct.calcsize("P") * 8 == 64:
    VIPS_HOME_DIR = (Path(__file__).parents[2] / 'third_party/'
                     'vips-dev-w64-web-8.13.3/vips-dev-8.13/bin')
    os.environ['PATH'] = str(VIPS_HOME_DIR) + ';' + os.environ['PATH']

if platform.system() == 'Windows' and '64' not in platform.machine():
    msg = ('Bimana requires 64-bit Python on Windows, some functions may fail '
           'given the current configuration.')
    warnings.warn(msg)
else:
    try:
        import pyvips
    except ImportError as e:
        msg = ('Bimana could not import Pyvips, some functions may fail. '
               'Unix-like operating system and MacOS users, please ensure '
               'vips is installed: https://www.libvips.org/install.html. '
               f'Reason for import failure: {repr(e)}')
        warnings.warn(msg)

IMAGE_FEX = '.png'
IMAGE_FILE_STEM = 'image'


class DirFormat(Enum):
    """Find_dirs flags.

    ROOT: If set, gets the path to the root directory if it contains at least
        one file.
    SUB: If set, gets the paths to file-containing subdirectories of the root
        directory.
    RECURSIVE: If set, gets the paths to all file-containing directories in
        the root directory tree.
    """

    ROOT = 1
    SUB = 2
    RECURSIVE = 3


def get_dirs(root_dir: Path, flag: DirFormat) -> list[Path]:
    """Gets the paths to directories with files from the root directory tree.

    Args:
        root_dir: The path to the root directory.
        flag: A flag that can take values of DirFormat.

    Returns:
        A sequence of paths to file-containing directories.
    """
    match flag:
        case flag.ROOT:
            dirs = ([root_dir] if len([f for f in root_dir.iterdir() if
                    f.is_file()]) != 0 else [])
        case flag.SUB:
            dirs = [curr_dir for curr_dir in root_dir.iterdir() if
                    curr_dir.is_dir() and len([f for f in curr_dir.iterdir() if
                                               f.is_file()]) != 0]
        case flag.RECURSIVE:
            dirs = []
            for curr_root_dir, _, files in os.walk(root_dir):
                if len(files) != 0:
                    dirs.append(Path(curr_root_dir))

    return dirs


def load_imgs(
    input_dir: Path,
    shrink_factor: int | None = None,
) -> list[tuple[Path, npt.NDArray[Any]]]:
    """Loads images from a directory.

    Args:
        input_dir: A directory containing image files.
        shrink_factor: Factor to shrink images by before loading them into
            memory.

    Returns:
        A sequence of images (and the corresponding image file path) for all
        successfully loaded images. The sequence is naturally sorted according
        to the image file paths.
    """
    # get paths of files in directory
    files = natsorted([f for f in input_dir.iterdir() if f.is_file()])

    # load images
    imgs = []
    if shrink_factor is None:
        imgs += [(f, img) for f in files if (img := cv2.imread(str(f))) is not
                 None]  # TODO cv2 size error?
    else:
        for f in files:
            if f.suffix == '.csv':
                continue
            try:
                img = pyvips.Image.new_from_file(str(f))
                img = img.shrink(shrink_factor, shrink_factor)
            except Exception as e:
                msg = (f'Failed to open or shrink file \'{f}\' as '
                       f'image: {repr(e)}')
                warnings.warn(msg)
                continue
            img = img.numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgs.append((f, img))

    return imgs


def save_imgs(
    imgs: list[tuple[str, npt.NDArray[Any]]],
    output_dir: Path,
) -> None:
    """Saves images to a directory.

    Args:
        imgs: A list of filename (without the extension) and image pairs.
            Images are saved with the corresponding filename in a predefined
            lossless format.
        output_dir: The directory to save images to.
    """
    for img in imgs:
        path = output_dir / (img[0] + IMAGE_FEX)
        cv2.imwrite(str(path), img[1])


def save_csv(data: dict[Any, Any], csv_path: Path) -> None:
    """Saves a csv file with the given data in the specified path.

    Args:
        data: The data to save where the keys will be the headers and the
            values will be the rows in the csv file.
        csv_path: The path to the csv file to save.
    """
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')


def crop_and_save_images(
    path: Path,
    coords: npt.NDArray[Any],
    skip_existing: bool = False,
) -> list[Path]:
    """Crops an image without loading it into memory and saves sub-images.

    Useful for dividing an image too large to load into memory into sub-images
    that can. Sub-images will be saved as .png files each in a subdirectory
    created in the same directory as the input image.

    Args:
        path: The file path to an image.
        coords: An array of the (x1, y1, x2, y2) coordinates of the bounding
            boxes for sub-images to crop in the image.
        skip_existing: If true, will not overwrite existing files with same
            names as generated subimage names.

    Returns:
        The file paths of the saved images.
    """
    output_dirs = []
    img = pyvips.Image.new_from_file(str(path))
    for x1, y1, x2, y2 in coords:
        h, w = y2 - y1, x2 - x1
        crop = img.crop(x1, y1, w, h)
        file_stem = f'{path.stem}_{x1}x1_{y1}y1_{x2}x2_{y2}y2'
        output_dir = path.parent / (file_stem + path.suffix.replace('.', '_'))
        output_dir.mkdir(exist_ok=True)
        output_path = (output_dir / file_stem).with_suffix('.png')
        if not skip_existing or not output_path.is_file():
            crop.pngsave(str(output_path))
        output_dirs.append(output_dir)

    return output_dirs
