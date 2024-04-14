import PIL
import click
import cv2
import exif
import labelme
import numpy as np
import os
import tqdm
import transformers

from ..utils import end, start, status

@click.group()
def _segment():
    pass
@_segment.command()
@click.argument("file-or-directory", type=click.Path(exists=True), nargs=-1)
@click.option("--ext", default=".jpg", help="File extension to search for (default: .jpg)")
@click.option("--json-ext", default=".json", help="File extension to save meta data to (default: .json)")
def segment(file_or_directory, ext, json_ext):
    start("Scanning for files")
    todo = list(file_or_directory)
    found = []
    while todo:
        file_or_directory = todo.pop()
        if os.path.isfile(file_or_directory) and file_or_directory.endswith(ext):
            found.append(file_or_directory)
        elif os.path.isdir(file_or_directory):
            todo.extend(os.path.join(file_or_directory, entry) for entry in os.listdir(file_or_directory))
    status(len(found), end='')
    end()
    meta = {}
    for entry in tqdm.tqdm(found, desc="Reading EXIF data and checking for user_comment field"):
        with open(entry, 'rb') as f:
            e = exif.Image(f)
            if e.has_exif:
                if hasattr(e, "pixel_x_dimension") and hasattr(e, "pixel_y_dimension"):
                    meta[entry] = (e.pixel_x_dimension, e.pixel_y_dimension)
                else:
                    print(f"Warning: {entry} has no pixel dimensions in EXIF data")
            else:
                print(f"Warning: {entry} has no EXIF data")
    status(len(meta), end='')
    end()
    start("Loading segmentation model")
    generator = transformers.pipeline("mask-generation", model="facebook/sam-vit-huge", device="cpu")
    end()
    start("Segmenting images")
    for entry, (width, height) in tqdm.tqdm(meta.items(), desc="Segmenting images"):
        print(f"Segmenting {entry}")
        image = PIL.Image.open(entry).convert("RGB")
        outputs = generator(image, points_per_batch=64, points_per_crop=8)
        polygons = []
        for mask in outputs["masks"]:
            if any(row[0] for row in mask) or any(row[-1] for row in mask):
                print("Skipping as mask touches left or right image border")
                continue
            masked = sum(1 for row in mask for pixel in row if pixel)
            total = sum(len(row) for row in mask)
            if masked/total < 0.01:
                print(f"Skipping as mask is less than 1% of total area: {masked/total*100}%")
                continue
            if masked/total > 0.50:
                print(f"Warning: mask is more than 50% of total area: {masked/total*100}%")
                continue
            print(f"Mask is {masked/total*100}% of total area")
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for obj in contours:
                poly = [
                    [int(p[0][0]), int(p[0][1])] for p in obj
                ]
                polygons.append(poly)
        lf = labelme.LabelFile()
        shapes = [dict(
            label="auto",
            points=poly,
            group_id=None,
            description=None,
            shape_type="polygon",
            flags={},
            mask=None,
        ) for poly in polygons]
        flags = {}
        imagePath = os.path.basename(entry)
        imageData = open(entry, 'rb').read()
        lf.save(
            filename=os.path.splitext(entry)[0]+json_ext,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=height,
            imageWidth=width,
            otherData=None,
            flags=flags,
        )
    end()
    