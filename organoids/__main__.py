import click

from .commands.analyze import _analyze
from .commands.segment import _segment

cli = click.CommandCollection(sources=[
    _analyze,
    _segment,
])

if __name__ == "__main__":
    cli()