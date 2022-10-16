from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / 'data'
TEST_IMG_PTH = str(DATA_DIR / 'image.tif')
TEST_BIN_IMG_PTH = str(DATA_DIR / 'binary_image.tif')
TEST_REF_BIN_IMG_PTH = str(DATA_DIR / 'cropped_refined_binary_image.tif')
