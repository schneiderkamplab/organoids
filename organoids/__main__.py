import click

from .commands.analyze import _analyze
from .commands.prune import _prune
from .commands.segment import _segment

cli = click.CommandCollection(sources=[
    _analyze,
    _prune,
    _segment,
])

if __name__ == "__main__":
    cli()