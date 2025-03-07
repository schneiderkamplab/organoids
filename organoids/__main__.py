import click

from .commands.analyze import _analyze
from .commands.ocr import _ocr
from .commands.prune import _prune
from .commands.segment import _segment
from .commands.evaluate import _evaluate

cli = click.CommandCollection(sources=[
    _analyze,
    _ocr,
    _prune,
    _segment,
    _evaluate
])

if __name__ == "__main__":
    cli()