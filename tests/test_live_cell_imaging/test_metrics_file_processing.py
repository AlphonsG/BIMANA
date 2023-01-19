import pickle
from csv import DictReader
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np
import pandas as pd
import pytest
from bimana.live_cell_imaging.metrics_file_processing import (
    MetricsTxtFileProcessor, metrics_file_processing)
from tests import (COMPDS_DF, CTRL_DF, METRICS_DF_PATH, DFS_PATH, FIGURES_DIR,
                   FLD_CHG_DF, FLD_CHG_STATS_DF, GEND_EXCEL, INFO_DF_PATH,
                   METRICS_TXT_FILE_PATH, NOTES_COMPDS_DF_PATH, STATS_DF,
                   TIME_DF_PATH)


def load_csv(
    csv_path: str,
    skip_header: bool = False,
    skip_index_col: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=None if skip_index_col else 0,
                     header=None if skip_header else 'infer')
    if not skip_header:
        with open(csv_path, encoding='utf-8') as f:
            if (cols := DictReader(f).fieldnames) is not None:
                df.columns = cols if skip_index_col else cols[1:]

    if '' in df.columns:
        df[''] = ''

    return df


@pytest.fixture(scope='module')
def processor() -> MetricsTxtFileProcessor:
    return MetricsTxtFileProcessor(Path(METRICS_TXT_FILE_PATH))


@pytest.fixture(scope='module')
def metrics_df() -> pd.DataFrame:
    return load_csv(METRICS_DF_PATH)


@pytest.fixture(scope='module')
def info_df() -> pd.DataFrame:
    return load_csv(INFO_DF_PATH, True, True)


@pytest.fixture(scope='module')
def time_df() -> pd.DataFrame:
    return load_csv(TIME_DF_PATH)


@pytest.fixture(scope='module')
def stats_df() -> pd.DataFrame:
    return load_csv(STATS_DF)


@pytest.fixture(scope='module')
def fld_chg_df() -> pd.DataFrame:
    return load_csv(FLD_CHG_DF)


@pytest.fixture(scope='module')
def fld_chg_stats_df() -> pd.DataFrame:
    return load_csv(FLD_CHG_STATS_DF)


@pytest.fixture(scope='module')
def ctrl_df() -> pd.DataFrame:
    return load_csv(CTRL_DF)


@pytest.fixture(scope='module')
def notes_cmpds_df() -> pd.DataFrame:
    return load_csv(NOTES_COMPDS_DF_PATH)


@pytest.fixture(scope='module')
def compds_df() -> pd.DataFrame:
    return load_csv(COMPDS_DF)


@pytest.fixture(scope='module')
def dfs() -> pd.DataFrame:
    with open(DFS_PATH, 'rb') as f:
        return pickle.load(f)


def test_create_dataframes(
    processor: MetricsTxtFileProcessor,
    metrics_df: pd.DataFrame,
    info_df: pd.DataFrame,
    time_df: pd.DataFrame,
) -> None:
    procd_metrics_df, procd_df_info, procd_df_time = (
        processor.create_dataframes())

    assert procd_metrics_df.equals(metrics_df)
    assert procd_df_info.equals(info_df)
    assert procd_df_time.equals(time_df)


def test_get_compound_dataframe(
    processor: MetricsTxtFileProcessor,
    stats_df: pd.DataFrame,
    ctrl_df: pd.DataFrame,
) -> None:
    vehicle_control = 'DMSO 0.05%'

    procd_df_ctrl = processor.get_compound_dataframe(stats_df, vehicle_control)

    assert procd_df_ctrl.equals(ctrl_df)


def test_use_notes_compounds(
    processor: MetricsTxtFileProcessor,
    metrics_df: pd.DataFrame,
    info_df: pd.DataFrame,
    notes_cmpds_df: pd.DataFrame,
) -> None:
    placeholder_compounds = ['CA', 'CB', 'CC', 'CD']

    procd_notes_cmpd_df = processor.use_notes_compounds(metrics_df, info_df,
                                                        placeholder_compounds)

    assert procd_notes_cmpd_df.equals(notes_cmpds_df)


def test_compute_stats(
    processor: MetricsTxtFileProcessor,
    metrics_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> None:
    procd_stats_df = processor.compute_stats(metrics_df)

    assert procd_stats_df.round(5).equals(stats_df.round(5))


def test_compute_fold_changes(
    processor: MetricsTxtFileProcessor,
    metrics_df: pd.DataFrame,
    ctrl_df: pd.DataFrame,
    fld_chg_df: pd.DataFrame,
) -> None:
    procd_fld_chg_df = processor.compute_fold_changes(metrics_df, ctrl_df)

    assert procd_fld_chg_df.round(5).equals(fld_chg_df.round(5))


def test_get_timepoints_dataframe(
    processor: MetricsTxtFileProcessor,
    time_df: pd.DataFrame,
    fld_chg_df: pd.DataFrame,
    compds_df: pd.DataFrame,
) -> None:
    fig_tpts = [6]

    procd_df_compds = processor.get_timepoints_dataframe(fld_chg_df,
                                                         time_df, fig_tpts)

    assert procd_df_compds.equals(compds_df)


def test_split_dataframe(
    processor: MetricsTxtFileProcessor,
    compds_df: pd.DataFrame,
    dfs: pd.DataFrame,
) -> None:
    procd_dfs = processor.split_dataframe(compds_df)

    for (procd_compd, procd_v), (compd, v) in zip(procd_dfs.items(),
                                                  dfs.items()):
        assert procd_compd == compd
        for (procd_conc, procd_df), (conc, df) in zip(procd_v.items(),
                                                      v.items()):
            assert procd_conc == conc
            assert procd_df.round(5).equals(df.round(5))


def test_gen_analysed_excel(
    processor: MetricsTxtFileProcessor,
    stats_df: pd.DataFrame,
    fld_chg_stats_df: pd.DataFrame,
    time_df: pd.DataFrame,
    info_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    processor.stats_df = stats_df
    processor.fld_chg_stats_df = fld_chg_stats_df
    processor.time_df = time_df
    processor.info_df = info_df

    processor.gen_analysed_excel(tmp_path / 'test.xlsx')

    files = [f for f in tmp_path.iterdir() if f.is_file()]
    assert len(files) == 1

    excel = pd.read_excel(files[0])
    expected_excel = pd.read_excel(GEND_EXCEL)

    for df in [excel, expected_excel]:
        df.iloc[7:11, 1:] = df.iloc[7:11, 1:].astype(float).round(5)
        df.iloc[13:17, 1:] = df.iloc[13:17, 1:].astype(float).round(5)

    assert excel.equals(expected_excel)


def test_generate_compound_summary_figs(
    processor: MetricsTxtFileProcessor,
    dfs: pd.DataFrame,
    tmp_path: Path,
) -> None:
    vehicle_control = 'DMSO'
    positive_control = 'Pos'
    negative_control = 'UT'
    threshold = 120
    processor.dfs = dfs

    processor.generate_compound_summary_figs(
        vehicle_control, positive_control, negative_control, threshold,
        tmp_path)

    files = sorted([f for f in tmp_path.iterdir() if f.is_file()])
    assert len(files) == 4

    expected_files = sorted([f for f in Path(FIGURES_DIR).iterdir() if
                             f.is_file()])

    for f, expected_f in zip(files, expected_files):
        assert np.array_equal(cv2.imread(str(f)), cv2.imread(str(expected_f)))


def test_metrics_file_processing(tmp_path: Path) -> None:
    copy2(METRICS_TXT_FILE_PATH, tmp_path)

    fig_tpts = 'fail'
    metrics_file_processing.callback(tmp_path, figure_timepoints=fig_tpts)

    assert len(list(tmp_path.iterdir())) == 1

    metrics_file_processing.callback(tmp_path)

    dirs = [d for d in tmp_path.iterdir() if d.is_dir()]

    assert len(dirs) == 1
    assert len([f for f in dirs[0].iterdir() if f.is_file()]) == 5
