import click

from bimana.histological_section.commands import histological_section_analysis


@click.group()
def cli() -> None:
    """A collection of software tools for performing automated bio-image
    analysis tasks through the web.
    """


cli.add_command(histological_section_analysis)

if __name__ == '__main__':
    cli()
