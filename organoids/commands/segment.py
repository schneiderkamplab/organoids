import PIL
import click
import os
import pickle
import tqdm
import transformers

from ..utils import end, start, status

@click.group()
def _segment():
    pass
@_segment.command()
@click.argument("file-or-directory", type=click.Path(exists=True), nargs=-1)
@click.option("--model", default="facebook/sam-vit-base", help="Model to use for segmentation (default: facebook/sam-vit-base)")
@click.option("--ext", default=".jpg", help="File extension to search for (default: .jpg)")
@click.option("--pickle-ext", default=".pickle", help="File extension to save masks to (default: .pickle)")
@click.option("--points-per-crop", default=60, help="Number of points per crop (default: 24)")
@click.option("--device", default="cpu", help="Device to use for segmentation (cpu, mps, cuda) (default: cpu)")
def segment(file_or_directory, model, ext, pickle_ext, points_per_crop, device):
    start("Scanning for files")
    todo = list(file_or_directory)
    found = []
    while todo:
        file_or_directory = todo.pop()
        if os.path.isfile(file_or_directory) and file_or_directory.endswith(ext):
            found.append(file_or_directory)
        elif os.path.isdir(file_or_directory):
            todo.extend(os.path.join(file_or_directory, entry) for entry in os.listdir(file_or_directory))
    found.sort()
    status(len(found), end='')
    end()
    start("Loading segmentation model")
    generator = transformers.pipeline("mask-generation", model=model, device=device)
    end()
    start("Segmenting images")
    for entry in tqdm.tqdm(found, desc="Segmenting images"):
        print(f"Segmenting {entry}")
        image = PIL.Image.open(entry).convert("RGB")
        outputs = generator(image, points_per_batch=points_per_crop, points_per_crop=points_per_crop)
        masks = []
        for mask in outputs["masks"]:
            masks.append(mask)
        pickle_path = os.path.splitext(entry)[0]+pickle_ext
        with open(pickle_path, 'wb') as f:
            pickle.dump(masks, f)
    end()
