import click
from bimana.live_cell_imaging.metrics_file_processing import (
    metrics_file_processing)


@click.group(help='Process live cell imaging data.')
def live_cell_imaging() -> None:
    pass


live_cell_imaging.add_command(metrics_file_processing)
