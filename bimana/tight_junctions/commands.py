from collections import defaultdict
from pathlib import Path

import click
import numpy as np

from bimana.tight_junctions.analysis import (extract_tight_junctions,
                                             perform_ijoq, segment_cells)
from bimana.utils.file_processing import (DirFormat, get_dirs, load_imgs,
                                          save_csv, save_imgs)
from bimana.utils.image_processing import (MAX_PX_VAL, ImageChannel,
                                           extract_channel)
from bimana.utils.quantification import (
    area, aspect_ratio, circularity, gen_histogram, gen_stats,
    num_connected_components, perimeter, roundness, size_connected_components,
    solidity)

DIR_FORMAT = DirFormat(1).name
TIGHT_JUNC_STAIN_COL = ImageChannel(2).name
CELL_STAIN_COL = ImageChannel(1).name
MIN_DIST = 12
ALPHA = 10
BETA = 14
MIN_RADIUS = 5
MAX_RADIUS = 15
MIN_AREA = 20
BACKGROUND_PIXEL_THRESH = 20
DONT_SAVE_PROCD_IMGS = False
METRICS_CSV_FILENAME = 'metrics.csv'


@click.command()
@click.argument('root_directory', type=click.Path(
                exists=True, file_okay=False))
@click.option('--analyse-tight-junctions', is_flag=True,
              help='Analyse tight junctions in the image e.g. by performing '
              'intercellular junction organization quantification.')
@click.option('--analyse-cells', is_flag=True,
              help='Analyse cells in the image e.g. by generating cell shape '
              'metrics.')
@click.option('--directory-format', type=click.Choice([DirFormat(1).name,
              DirFormat(2).name, DirFormat(3).name], case_sensitive=False),
              default=DIR_FORMAT, show_default=True,
              help='Location of image files to process in the root directory '
                   f'tree - if set to {DirFormat(1).name}, processes images '
                   f'in the root directory; if set to {DirFormat(2).name}, '
                   'processes images in subdirectories of the root directory; '
                   f' if set to {DirFormat(3).name}, processes images '
                   'in all directories of the root directory tree.')
@click.option('--tight-junction-stain-colour', type=click.Choice(
              [ImageChannel(1).name, ImageChannel(2).name,
               ImageChannel(3).name], case_sensitive=False),
              default=TIGHT_JUNC_STAIN_COL, show_default=True,
              help='The primary colour (or one of) that the tight junction '
              'stain is composed of.')
@click.option('--cell-stain-colour', type=click.Choice(
              [ImageChannel(1).name, ImageChannel(2).name,
               ImageChannel(3).name], case_sensitive=False),
              default=CELL_STAIN_COL, show_default=True, help='The primary '
              'colour (or one of) that the cell stain is composed of.')
@click.option('--min-dist', type=click.IntRange(0),
              default=MIN_DIST, show_default=True,
              help='The minimum distance between the centers of cells in '
                   'pixels.')
@click.option('--alpha', type=click.IntRange(0), default=ALPHA,
              show_default=True,
              help='A parameter that can be fine tuned for cell segmentation '
                   'performance; it may be decreased to identify more '
                   'objects.')
@click.option('--beta', type=click.IntRange(0), default=BETA,
              show_default=True,
              help='A parameter that can be fine tuned for cell segmentation '
                   'performance; the smaller it is, the more false cells may '
                   'be identified.')
@click.option('--min-radius', type=click.IntRange(0), default=MIN_RADIUS,
              show_default=True, help='The minimum radius of cells in pixels.')
@click.option('--max-radius', type=click.IntRange(0), default=MAX_RADIUS,
              show_default=True, help='The maximum radius of cells in pixels.')
@click.option('--min-area', type=click.IntRange(0), default=MIN_AREA,
              show_default=True, help='The minimum area of cells in pixels.')
@click.option('--background-pixel-thresh', type=click.IntRange(0, MAX_PX_VAL),
              default=BACKGROUND_PIXEL_THRESH, show_default=True,
              help='The maximum intensity of background pixels in the '
                   'greyscale image.')
@click.option('--no-image-processing-visualization', is_flag=True,
              help='Do not save image files visualizing different stages of '
                   'image processing.')
def tight_junction_analysis(
    root_directory: str | Path,
    analyse_tight_junctions: bool,
    analyse_cells: bool,
    directory_format: str = DIR_FORMAT,
    tight_junction_stain_colour: str = TIGHT_JUNC_STAIN_COL,
    cell_stain_colour: str = CELL_STAIN_COL,
    min_dist: int = MIN_DIST,
    alpha: int = ALPHA,
    beta: int = BETA,
    min_radius: int = MIN_RADIUS,
    max_radius: int = MAX_RADIUS,
    min_area: int = MIN_AREA,
    background_pixel_thresh: int = BACKGROUND_PIXEL_THRESH,
    no_image_processing_visualization: bool = DONT_SAVE_PROCD_IMGS,
) -> None:
    """Analyse fluorescent tight junction images.

    Analysis is performed with respect to the tight junctions (e.g. by
    performing intercellular junction organization quantification) and/or cells
    (e.g. by generating cell shape metrics) present in the image. By default,
    saves generated data in the root directory tree.

    ROOT_DIRECTORY:

    The directory to search for images to process in. When using the web
    interface, directory must be zipped.


    IMPORTANT NOTES

    Default option values:

    The default option values were fine tuned for processing fluorescent images
    containing several tightly packed airway epithelial cells (stained blue)
    and tight junctions (stained green), for example,
    https://github.com/AlphonsG/BIMANA
    """
    # check inputs
    if not (analyse_tight_junctions or analyse_cells):
        click.echo('Invalid inputs: --analyse-tight-junctions and/or '
                   '--analyse-cells option(s) must be selected.')
        return

    root_dir = output_dir = Path(root_directory)
    output_dirs = []
    metrics = defaultdict(list)

    for curr_dir in get_dirs(root_dir, DirFormat[directory_format]):
        if curr_dir in output_dirs:  # previously generated output directory
            continue
        for filename, img in load_imgs(curr_dir):
            curr_output_dir = output_dir / curr_dir.relative_to(
                root_dir) / f'{filename.stem}_{filename.suffix[1:]}'
            output_dirs.append(curr_output_dir)

            metrics['Image'].append(filename.name)
            metrics['File path'].append(str(filename))

            procd_imgs = []
            if analyse_tight_junctions:
                tight_juncs, curr_procd_imgs = extract_tight_junctions(
                    img, ImageChannel[tight_junction_stain_colour])

                procd_imgs += curr_procd_imgs

                img_tight_junc_chn = extract_channel(
                    img, ImageChannel[tight_junction_stain_colour]).astype(
                        float)
                img_tight_junc_chn[~tight_juncs] = np.nan
                tight_junc_px = img_tight_junc_chn.flatten()
                tight_junc_intsty = (tight_junc_px / MAX_PX_VAL * 100).tolist()
                stats = gen_stats(tight_junc_intsty)
                for name, stat in stats.items():
                    metrics['Tight Junction Pixel Intensity (%) '
                            f'({name})'].append(stat)
                curr_output_dir.mkdir(parents=True, exist_ok=True)
                gen_histogram({'Tight Junction Pixel Intensity (%)':
                               tight_junc_intsty}, curr_output_dir)

                metrics['Intercellular Junction Organization '
                        'Quantification (AU)'].append(
                            perform_ijoq(tight_juncs))
                metrics['Number of Tight Junction Fragments'].append(
                    num_connected_components(tight_juncs))

                tight_junc_sizes = size_connected_components(tight_juncs)
                stats = gen_stats(tight_junc_sizes)
                for name, stat in stats.items():
                    metrics['Tight Junction Fragment Size (No. Pixels) '
                            f'({name})'].append(stat)
                gen_histogram({'Tight Junction Fragment Size (No. Pixels)':
                               tight_junc_sizes}, curr_output_dir)

            if analyse_cells:
                cells, curr_procd_imgs = segment_cells(
                    img, ImageChannel[cell_stain_colour], min_dist, alpha,
                    beta, min_radius, max_radius, min_area,
                    background_pixel_thresh)

                procd_imgs += curr_procd_imgs

                metrics['Number of Cells'].append(len(cells))
                data = {'Cell Area (No. Pixels)': area(cells),
                        'Cell Aspect Ratio (AU)': aspect_ratio(cells),
                        'Cell Circularity (AU)': circularity(cells),
                        'Cell Perimeter (No. Pixels)': perimeter(cells),
                        'Cell Roundness (AU)': roundness(cells),
                        'Cell Solidity (AU)': solidity(cells)}

                for data_name, data in data.items():
                    stats = gen_stats(data)
                    for stat_name, stat in stats.items():
                        metrics[f'{data_name} ({stat_name})'].append(stat)
                    gen_histogram({data_name: data}, curr_output_dir)

            if not no_image_processing_visualization:
                save_imgs(procd_imgs, curr_output_dir)

    # save metrics
    save_csv(metrics, output_dir / METRICS_CSV_FILENAME)
