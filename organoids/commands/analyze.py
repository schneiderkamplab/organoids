import click
import exif
import json
import os
import shapely
import tqdm

from ..utils import end, start, status

@click.group()
def _analyze():
    pass
@_analyze.command()
@click.argument("directory", type=click.Path(exists=True), nargs=-1)
@click.option("--ext", default=".json", help="File extension to search for (default: .json)")
@click.option("--exif-ext", default=".jpg", help="File extension to extract EXIF from (default: .jpg)")
def analyze(directory, ext, exif_ext):
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
    data = {}
    for entry in tqdm.tqdm(found, desc="Parsin JSON and checking for shapes data"):
        with open(entry, 'rt') as f:
            d = json.load(f)
            if "shapes" in d:
                data[entry] = d
            else:
                print(f"Warning: {entry} has no shapes data")
    status(len(data), end='')
    end()
    start("Computing areas")
    for entry, d in tqdm.tqdm(data.items(), desc="Computing areas"):
        image_path = os.path.join(os.path.dirname(entry), d["imagePath"])
        if image_path.endswith(exif_ext):
            e = exif.Image(open(image_path, 'rb'))
            if e.has_exif:
                if hasattr(e, "user_comment"):
                    user_comment = json.loads(e.user_comment)
                    pix_size = user_comment["effectivePixelSize"]/10**6
                    mag = user_comment["objectiveMag"]
                    for s in d["shapes"]:
                        poly = shapely.geometry.Polygon(s["points"])
                        area = poly.area*pix_size/mag
                        s["description"] = f"{area:.2f} mm²"
                else:
                    print(f"Warning: {image_path} referenced from {entry} has no user_comment")
            else:
                print(f"Warning: {image_path} referenced from {entry} has no EXIF data")
        else:
            print(f"Skipping {image_path} referenced from {entry} as it does not end with {exif_ext}")
    end()
    start("Writing areas to disk")
    for entry, d in tqdm.tqdm(data.items(), desc="Writing areas to disk"):
        with open(entry, 'wt') as f:
            json.dump(d, f, indent=2)
    end()
    