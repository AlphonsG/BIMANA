import click

from bimana.histological_section.commands import histological_section_analysis
from bimana.tight_junctions.commands import tight_junction_analysis
from bimana.live_cell_imaging.commands import live_cell_imaging

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """ Simple command line utility called `bimana`.

    The cli is composed of commands for performing automated bio-image
    analysis tasks.
    """


cli.add_command(histological_section_analysis)
cli.add_command(tight_junction_analysis)
cli.add_command(live_cell_imaging)
