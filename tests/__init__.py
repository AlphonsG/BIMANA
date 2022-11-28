from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / 'data'

TEST_IMG_PTH = str(DATA_DIR / 'image.tif')
TEST_BIN_IMG_PTH = str(DATA_DIR / 'binary_image.tif')
TEST_REF_BIN_IMG_PTH = str(DATA_DIR / 'cropped_refined_binary_image.tif')
TEST_TISS_BNDY_PTH = str(DATA_DIR / 'tissue_boundary.pickle')
ROOT_DIR_PTH = DATA_DIR / str('root_directory')
HIST_SECT_IMG_DIR_PTH = str(DATA_DIR / 'histological_section/images')
LENA_IMG_PTH = str(DATA_DIR / 'lena.tif')
GREYSCALE_IMG_PTH = str(DATA_DIR / 'greyscale_image.jpg')
BIN_TGHT_JNCS_PTH = str(DATA_DIR / 'binary_tight_junctions.png')
CIRCLES_IMG_PTH = str(DATA_DIR / 'circles.png')
TIGHT_JUNCS_IMG_PTH = str(DATA_DIR / 'tight_junctions.tiff')
SUBIMG_PTH = str(DATA_DIR / 'subimage/subimage.png')
