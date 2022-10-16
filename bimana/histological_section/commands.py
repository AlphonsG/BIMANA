from collections import defaultdict
from pathlib import Path

import click

from bimana.histological_section.analysis import (
    amount_cilia_above_tissue_area, amount_staining_in_tissue_area,
    tissue_boundary)
from bimana.utils.commands import parse_input_bgr
from bimana.utils.file_processing import (DirFormat, get_dirs, load_imgs,
                                          save_csv, save_imgs)
from bimana.utils.image_processing import scale_bgr_values

DIR_FORMAT = DirFormat(1).name
NON_TISSUE_LOWER_BGR = ('210', '210', '210')
NON_TISSUE_UPPER_BGR = ('255', '255', '255')
STAIN_LOWER_BGR = ('0', '0', '0')
STAIN_UPPER_BGR = ('255', '255', '150')
CILIA_LOWER_BGR = ('220', '220', '220')
CILIA_UPPER_BGR = ('245', '245', '245')
ISO_NON_TISSUE_SENS = 0.2
NO_SMOOTH_SEGMENTED_TISSUE_BOUNDARY = False
CILIA_REGION_THICKNESS = 15
DONT_SAVE_PROCD_IMGS = False

METRICS_CSV_FILENAME = 'metrics.csv'


@click.command()
@click.argument('root_directory', type=click.Path(
                exists=True, file_okay=False))
@click.option('--staining-amount-in-tissue-area', is_flag=True,
              help='Calculate how much of a histological section image\'s '
                   'tissue area is stained.')
@click.option('--cilia-amount-above-tissue-area', is_flag=True,
              help='Calculate the amount of cilia above a histological '
                   'section image\'s tissue area.')
@click.option('--directory-format', type=click.Choice([DirFormat(1).name,
              DirFormat(2).name, DirFormat(3).name], case_sensitive=False),
              default=DIR_FORMAT, show_default=True,
              help='Location of image files to process in the root directory '
                   f'tree - if set to {DirFormat(1).name}, processes images '
                   f'in the root directory; if set to {DirFormat(2).name}, '
                   'processes images in subdirectories of the root directory; '
                   f' if set to {DirFormat(3).name}, processes images '
                   'in all directories of the root directory tree.')
@click.option('--non-tissue-lower-colour-limit', nargs=3,
              default=NON_TISSUE_LOWER_BGR, show_default=True,
              help='Control the region of the image identified as tissue - '
              'pixels in the image with blue, green and red intensity values '
              'greater than the provided values will be considered '
              'non-tissue.')
@click.option('--non-tissue-upper-colour-limit', nargs=3,
              default=NON_TISSUE_UPPER_BGR, show_default=True,
              help='Control the region of the image identified as tissue - '
              'pixels in the image with blue, green and red intensity values '
              'lesser than the provided values will be considered non-tissue.')
@click.option('--staining-lower-colour-limit', nargs=3,
              default=STAIN_LOWER_BGR, show_default=True,
              help='Control the colour of the staining to identify in the '
              'image - pixels in the image with blue, green and red intensity '
              'values greater than the provided values will be considered '
              'staining.')
@click.option('--staining-upper-colour-limit', nargs=3,
              default=STAIN_UPPER_BGR, show_default=True,
              help='Control the colour of the staining to identify in the '
              'image - pixels in the image with blue, green and red intensity '
              'values lesser than the provided values will be considered '
              'staining.')
@click.option('--cilia-lower-colour-limit', nargs=3,
              default=CILIA_LOWER_BGR, show_default=True,
              help='control the colour of the cilia to identify in the '
              'image - pixels in the image with blue, green and red intensity '
              'values greater than the provided values will be considered '
              'cilia.')
@click.option('--cilia-upper-colour-limit', nargs=3, default=CILIA_UPPER_BGR,
              show_default=True,
              help='control the colour of the cilia to identify in the '
              'image - pixels in the image with blue, green and red intensity '
              'values lesser than the provided values will be considered '
              'cilia.')
@click.option('--sensitivity-to-isolated-non-tissue',
              type=click.FloatRange(0, 1), default=ISO_NON_TISSUE_SENS,
              show_default=True,
              help='Filter out more isolated objects identified in the image '
                   'as non-tissue with greater sensitivity values.')
@click.option('--no-tissue-boundary-smoothing', is_flag=True,
              default=NO_SMOOTH_SEGMENTED_TISSUE_BOUNDARY, show_default=True,
              help='Do not smooth the identified tissue boundary as a post '
                   'processing step.')
@click.option('--cilia-region-thickness', type=click.IntRange(1),
              default=CILIA_REGION_THICKNESS, show_default=True,
              help='Vertical thickness, in pixels, of the cilia-containing '
              'region above the tissue area.')
@click.option('--no-image-processing-visualization', is_flag=True,
              help='Do not save image files visualizing different stages of '
                   'image processing.')
def histological_section_analysis(
    root_directory: str | Path,
    staining_amount_in_tissue_area: bool,
    cilia_amount_above_tissue_area: bool,
    directory_format: str = DIR_FORMAT,
    non_tissue_lower_colour_limit: tuple[str, str, str] = NON_TISSUE_LOWER_BGR,
    non_tissue_upper_colour_limit: tuple[str, str, str] = NON_TISSUE_UPPER_BGR,
    staining_lower_colour_limit: tuple[str, str, str] = STAIN_LOWER_BGR,
    staining_upper_colour_limit: tuple[str, str, str] = STAIN_UPPER_BGR,
    cilia_lower_colour_limit: tuple[str, str, str] = CILIA_LOWER_BGR,
    cilia_upper_colour_limit: tuple[str, str, str] = CILIA_UPPER_BGR,
    sensitivity_to_isolated_non_tissue: float = ISO_NON_TISSUE_SENS,
    no_tissue_boundary_smoothing: bool = NO_SMOOTH_SEGMENTED_TISSUE_BOUNDARY,
    cilia_region_thickness: int = CILIA_REGION_THICKNESS,
    no_image_processing_visualization: bool = DONT_SAVE_PROCD_IMGS,
) -> None:
    """Analyse histological section images.

    Analysis consists of calculating how much of a histological section image's
    tissue area is stained as a percentage and/or the amount of cilia in the
    region above a histological section image's tissue area as a percentage. By
    default, saves generated data in the root directory tree.

    ROOT_DIRECTORY:

    The directory to search for images to process in. When using the web
    interface, directory must be zipped.


    IMPORTANT NOTES

    Colour limit options:

    Options containing the term '...-colour-limit' accept 3 input values. The
    first, second and third input value is a blue, green and red pixel
    intensity value, respectively. Each intensity value can be either a whole
    number between 0 and 255 (raw pixel value) or a decimal number from 0.0 to
    1.0 (representing 0% to 100% intensity) which will be converted to a raw
    pixel value internally. Each pixel in any given image is composed of blue,
    green and red intensity values which result in the observable colour of the
    pixel, such as bright yellow. Specific details on what the inputs to each
    '...-colour-limit' option controls is provided in the respective help
    message.

    Default option values:

    The default option values were fine tuned for processing histological
    sections of epithelium with alcian blue stain (a mucus stain), for example,
    https://github.com/AlphonsG/BIMANA
    """
    # check inputs
    if not (staining_amount_in_tissue_area or cilia_amount_above_tissue_area):
        click.echo('Invalid inputs: --staining-amount-in-tissue-area and/or '
                   '--cilia-amount-above-tissue-area option must be selected.')
        return

    try:
        non_tissue_lower_bgr = parse_input_bgr(non_tissue_lower_colour_limit)
        non_tissue_upper_bgr = parse_input_bgr(non_tissue_upper_colour_limit)
        stain_lower_bgr = parse_input_bgr(staining_lower_colour_limit)
        stain_upper_bgr = parse_input_bgr(staining_upper_colour_limit)
        cilia_lower_bgr = parse_input_bgr(cilia_lower_colour_limit)
        cilia_upper_bgr = parse_input_bgr(cilia_upper_colour_limit)
    except ValueError:
        click.echo('Invalid value provided for \'...--colour-limit\' '
                   'option(s).')
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
            upper_lower_xs_ys, curr_procd_imgs = tissue_boundary(img,
                scale_bgr_values(non_tissue_lower_bgr),
                scale_bgr_values(non_tissue_upper_bgr),
                sensitivity_to_isolated_non_tissue or None,
                not no_tissue_boundary_smoothing)
            procd_imgs += curr_procd_imgs

            if staining_amount_in_tissue_area:
                results = amount_staining_in_tissue_area(img,
                    upper_lower_xs_ys[0], upper_lower_xs_ys[1],
                    scale_bgr_values(stain_lower_bgr),
                    scale_bgr_values(stain_upper_bgr))
                pct_stain, stain_amt, tiss_size, curr_procd_imgs = results
                procd_imgs += curr_procd_imgs

                metrics['Amount of staining (no. pixels)'].append(stain_amt)
                metrics['Tissue area size (no. pixels'].append(tiss_size)
                metrics['Percentage of staining in tissue area'].append(
                    pct_stain)

            if cilia_amount_above_tissue_area:
                results = amount_cilia_above_tissue_area(img,
                    upper_lower_xs_ys[0], upper_lower_xs_ys[1],
                    scale_bgr_values(cilia_lower_bgr),
                    scale_bgr_values(cilia_upper_bgr), cilia_region_thickness)
                pct_cilia, cilia_amt, reg_size, curr_procd_imgs = results
                procd_imgs += curr_procd_imgs

                metrics['Amount of cilia (no. pixels)'].append(cilia_amt)
                metrics['Cilia-containing region size (no. pixels'].append(
                    reg_size)
                metrics['Percentage of region occupied by cilia'].append(
                    pct_cilia)

            if not no_image_processing_visualization:
                curr_output_dir.mkdir(parents=True, exist_ok=True)
                save_imgs(procd_imgs, curr_output_dir)

    # save metrics
    save_csv(metrics, output_dir / METRICS_CSV_FILENAME)
