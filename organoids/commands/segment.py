import PIL
import click
import cv2
import exif
import json
import labelme
import numpy as np
import os
import shapely
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
@click.option("--eps", default=0.001, help="Epsilon for polygon approximation (default: .001)")
def segment(file_or_directory, ext, json_ext, eps):
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
    for entry in tqdm.tqdm(found, desc="Reading EXIF data and checking for user_comments and pixel_dimension fields"):
        with open(entry, 'rb') as f:
            e = exif.Image(f)
            if e.has_exif:
                if hasattr(e, "pixel_x_dimension") and hasattr(e, "pixel_y_dimension") and hasattr(e, "user_comment"):
                    user_comment = json.loads(e.user_comment)
                    meta[entry] = dict(
                        width=e.pixel_x_dimension,
                        height=e.pixel_y_dimension,
                        pix_size=user_comment["effectivePixelSize"],
                        mag=user_comment["objectiveMag"],
                    )
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
    for entry in tqdm.tqdm(meta, desc="Segmenting images"):
        print(f"Segmenting {entry}")
        image = PIL.Image.open(entry).convert("RGB")
        outputs = generator(image, points_per_batch=64, points_per_crop=8)
        masks = []
        for mask in outputs["masks"]:
            if any(row[0] for row in mask) or any(row[-1] for row in mask):
                print("Skipping as mask touches left or right image border")
                continue
            masked = sum(1 for row in mask for pixel in row if pixel)
            total = sum(len(row) for row in mask)
            if masked/total < 0.001:
                print(f"Skipping as mask is less than 1% of total area: {masked/total*100}%")
                continue
            if masked/total > 0.50:
                print(f"Skipping as mask is more than 50% of total area: {masked/total*100}%")
                continue
            print(f"Mask is {masked/total*100}% of total area")
            masks.append(mask)
        super_masks = []
        for i, m1 in enumerate(masks):
            for j, m2 in enumerate(masks):
                if i != j and m2[m1].all():
                    print("Skipping as mask is subsumed by another mask")
                    break
            else:
                super_masks.append(m1)
        polygons = []
        for mask in super_masks:
            masked_pixels = np.array(image)[mask]
            masked_pixels_mean = masked_pixels.mean()
            if masked_pixels_mean >= 64:
                print(f"Skipping as mask is too bright: {masked_pixels_mean}")
                continue
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            obj = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(obj, True)
            approx = cv2.approxPolyDP(obj, eps*peri, True)
            approx_x = approx[:,:,0]
            approx_y = approx[:,:,1]
            center, _ = cv2.minEnclosingCircle(approx)
            distances = np.sqrt((approx_x-center[0])**2 + (approx_y-center[1])**2)
            min_dist, max_dist = distances.min(), distances.max()
            if max_dist > 2*min_dist:
                print(f"Skipping as mask is too irregular: {max_dist/min_dist}")
                continue
            poly = [
                [int(p[0][0]), int(p[0][1])] for p in approx
            ]
            polygons.append(poly)
        areas = []
        for poly in polygons:
            poly = shapely.geometry.Polygon(poly)
            area = poly.area*meta[entry]["pix_size"]/10**6/meta[entry]["mag"]
            areas.append(area)
        lf = labelme.LabelFile()
        shapes = [dict(
            label="auto",
            points=poly,
            group_id=None,
            description=f"{area} mm²",
            shape_type="polygon",
            flags={},
            mask=None,
        ) for poly, area in zip(polygons, areas)]
        flags = {}
        imagePath = os.path.basename(entry)
        imageData = open(entry, 'rb').read()
        lf.save(
            filename=os.path.splitext(entry)[0]+json_ext,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=meta[entry]["height"],
            imageWidth=meta[entry]["width"],
            otherData=None,
            flags=flags,
        )
    end()
    