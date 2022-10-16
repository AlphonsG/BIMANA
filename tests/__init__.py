from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / 'data'

TEST_IMG_PTH = str(DATA_DIR / 'image.tif')
TEST_BIN_IMG_PTH = str(DATA_DIR / 'binary_image.tif')
TEST_REF_BIN_IMG_PTH = str(DATA_DIR / 'cropped_refined_binary_image.tif')
TEST_TISS_BNDY_PTH = str(DATA_DIR / 'tissue_boundary.pickle')
ROOT_DIR_PTH = DATA_DIR / str('root_directory')
HIST_SECT_IMG_DIR_PTH = str(DATA_DIR / 'histological_section/images')
