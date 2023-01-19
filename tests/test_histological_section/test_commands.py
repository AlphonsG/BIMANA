import shutil
from pathlib import Path

from bimana.histological_section.commands import (
    METRICS_CSV_FILENAME, histological_section_analysis)
from tests import HIST_SECT_IMG_DIR_PTH


def test_histological_section_analysis(tmp_path: Path) -> None:
    root_dir = tmp_path / 'root_dir'
    shutil.copytree(HIST_SECT_IMG_DIR_PTH, root_dir)
    img_filename = list(root_dir.iterdir())[0].name
    # TODO dirname hardcoded?
    output_dir = root_dir / str(img_filename).replace('.', '_')
    amount_staining = amount_cilia = True

    histological_section_analysis.callback(root_dir, False, False)

    assert not output_dir.is_dir()
    assert len(list(root_dir.iterdir())) == 1

    non_tissue_lower_bgr_str = ('x', '10', '10')
    histological_section_analysis.callback(
        root_dir, amount_staining, amount_cilia,
        non_tissue_lower_colour_limit=non_tissue_lower_bgr_str)

    assert not output_dir.is_dir()
    assert len(list(root_dir.iterdir())) == 1

    histological_section_analysis.callback(
        root_dir, amount_staining, amount_cilia)

    assert (output_dir.parent / img_filename).is_file()
    assert output_dir.is_dir()
    assert len([f for f in output_dir.iterdir() if f.is_file()]) > 2
    assert METRICS_CSV_FILENAME in [f.name for f in
                                    output_dir.parent.iterdir()]
