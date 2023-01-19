import shutil
from pathlib import Path

from bimana.tight_junctions.commands import (METRICS_CSV_FILENAME,
                                             tight_junction_analysis)
from tests import TIGHT_JUNCS_IMG_PTH


def test_tight_junction_analysis(tmp_path: Path) -> None:
    root_dir = tmp_path / 'root_dir'
    root_dir.mkdir()
    shutil.copy2(TIGHT_JUNCS_IMG_PTH, root_dir)
    img_filename = list(root_dir.iterdir())[0].name
    # TODO dirname hardcoded?
    output_dir = root_dir / str(img_filename).replace('.', '_')
    analyse_tight_juncs = analyse_cells = True

    tight_junction_analysis.callback(root_dir, False, False)

    assert not output_dir.is_dir()
    assert len(list(root_dir.iterdir())) == 1

    tight_junction_analysis.callback(
        root_dir, analyse_tight_juncs, analyse_cells)

    assert (output_dir.parent / img_filename).is_file()
    assert output_dir.is_dir()
    assert len([f for f in output_dir.iterdir() if f.is_file()]) > 2
    assert METRICS_CSV_FILENAME in [f.name for f in
                                    output_dir.parent.iterdir()]
