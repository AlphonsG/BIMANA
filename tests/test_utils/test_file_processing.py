import csv
from collections import defaultdict
from pathlib import Path

import pytest
import numpy as np
from bimana.utils.file_processing import (IMAGE_FEX, DirFormat, get_dirs,
                                          load_imgs, save_imgs, save_csv)
from tests import ROOT_DIR_PTH


@pytest.mark.parametrize(
    'flag,num_dirs',
    [(DirFormat.SUB, 1), (DirFormat.ROOT, 1), (DirFormat.RECURSIVE, 3)]
)
def test_get_dirs(flag: DirFormat, num_dirs: int) -> None:
    dirs = get_dirs(ROOT_DIR_PTH, flag)

    assert len(dirs) == num_dirs
    for curr_dir in dirs:
        assert curr_dir.exists()


def test_load_imgs() -> None:
    imgs = load_imgs(ROOT_DIR_PTH)
    prev_img_num = -1
    for (path, img) in imgs:
        assert len(img.shape) == 3
        assert len(np.unique(img)) > 2
        assert int(path.stem[-1]) > prev_img_num


def test_save_imgs(tmp_path: Path) -> None:
    img1 = np.random.randint(0, 255, (50, 50))
    img2 = np.random.randint(0, 255, (50, 50))

    imgs = [('img1', img1), ('img2', img2)]
    save_imgs(imgs, tmp_path)
    for name, _ in imgs:
        assert (tmp_path / (name + IMAGE_FEX)).is_file()


def test_save_csv(tmp_path: Path) -> None:

    exp_data = {'headerA': [1, 2, 3], 'headerB': [4, 5, 6]}
    csv_path = tmp_path / 'test.csv'

    save_csv(exp_data, csv_path)
    assert csv_path.is_file()

    data = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert headers is not None
        for row in reader:
            for header in headers:
                data[header].append(int(row[header]))

    assert data == exp_data
