import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

DEXINET_DIR = Path(__file__).parents[3] / 'third_party/DexiNed'
sys.path.insert(1, str(DEXINET_DIR))
# try importing Pytorch
try:
    import torch
    from torch.utils.data import DataLoader
    from datasets import TestDataset
    from main import test
    from model import DexiNed
except ModuleNotFoundError as e:
    if e.msg == 'torch':
        msg = ('Bimana could not import PyTorch - deep learning functions '
               'will not be available. Ensure Pytorch is correctly installed '
               'to utilise these functions: '
               'https://pytorch.org/get-started/locally/. Reason for import '
               f'failure: {repr(e)}')
        warnings.warn(msg)
    else:
        raise


ARGS = SimpleNamespace()
ARGS.is_testing = True
ARGS.test_data = 'CLASSIC'
TEST_IMG_HEIGHT, TEST_IMG_WIDTH = 512, 512
MEAN_PIXEL_VALUES = [103.939, 116.779, 123.68]
CHECKPOINT_PATH = (Path(__file__).parents[3] / 'misc/checkpoints/DexiNed/'
                   'BIPED/10/10_model.pth')


def detect_edges(
    img: npt.NDArray[np.uint8],
    gpu: bool = False,
) -> npt.NDArray[np.uint8]:
    """Detects edges in an image using DexiNed.

    DexiNed: https://arxiv.org/pdf/2112.02250.pdf.

    Args:
        img: An image.
        gpu: If true, will use the GPU for edge detection.

    Returns:
        A greyscale image of the edges from the original image.
    """
    device = torch.device('cuda' if gpu and torch.cuda.device_count() > 0 else
                          'cpu')
    model = DexiNed().to(device)

    if len(img.shape) == 2:
        img = img[..., None]
        img = np.repeat(img, 3, 2)
    dataset_val = TestDataset(None, ARGS.test_data, MEAN_PIXEL_VALUES,
                              TEST_IMG_HEIGHT, TEST_IMG_WIDTH, imgs=[img])
    dataloader_val = DataLoader(dataset_val)

    avg_img, _ = test(CHECKPOINT_PATH, dataloader_val, model, device, None,
                      ARGS)[0]

    return ~avg_img
