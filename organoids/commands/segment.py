import click
import exif
import json
import os
import tqdm

from ..utils import end, start, status

@click.group()
def _segment():
    pass
@_segment.command()
@click.argument("directory", type=click.Path(exists=True), nargs=-1)
@click.option("--ext", "-e", default=".jpg", help="File extension to search for (default: .jpg)")
def segment(directory, ext):
    start("Scanning for files")
    todo = list(directory)
    found = []
    while todo:
        dir = todo.pop()
        for entry in os.listdir(dir):
            entry_path = os.path.join(dir, entry)
            if os.path.isdir(entry_path):
                todo.append(entry_path)
            elif entry.endswith(ext):
                found.append(entry_path)
    status(len(found), end='')
    end()
    meta = {}
    for entry in tqdm.tqdm(found, desc="Reading EXIF data and checking for user_comment field"):
        with open(entry, 'rb') as f:
            e = exif.Image(f)
            if e.has_exif:
                if hasattr(e, "user_comment"):
                    meta[entry] = json.loads(e.user_comment)
                else:
                    print(f"Warning: {entry} has no user_comment")
            else:
                print(f"Warning: {entry} has no EXIF data")
    status(len(meta), end='')
    end()

    