import os
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy.typing as npt
import pandas as pd
from natsort import natsorted

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


def load_imgs(input_dir: Path) -> list[tuple[Path, npt.NDArray[Any]]]:
    """Loads images from a directory.

    Args:
        input_dir: A directory containing image files.

    Returns:
        A sequence of images (and the corresponding image file path) for all
        successfully loaded images. The sequence is naturally sorted according
        to the image file paths.
    """
    # get paths of files in directory
    files = natsorted([f for f in input_dir.iterdir() if f.is_file()])

    # load images from valid image files
    return [(f, img) for f in files if (img := cv2.imread(str(f))) is not None]


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
    df.to_csv(csv_path, index=False)
