import os
import json
import click
import exif
import tqdm
import numpy as np
import shapely

from ..utils import end, start, status

@click.group()
def _rank():
    pass

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

@_rank.command()
@click.argument("dir", type=click.Path(exists=True))
def rank(dir):
    start("Computing rank")
    todo = [f for f in os.listdir(dir) if f.endswith(".json")]
    
    for file in tqdm.tqdm(todo):
        data = load_json(os.path.join(dir, file))
        if "shapes" not in data.keys():
            print(f"Warning: {file} has no shapes data")
        
        poly_data = []
        shapes = data["shapes"]
        for shape in shapes:
            if 'label' not in shape.keys():
                print(f"Warning: {file} contains shapes without label-key")
            
            label = shape['label']
            poly = shapely.geometry.Polygon(shape['points'])
            area = poly.area
            poly_data.append({'label': label, 'area:': area})

        # sort polygons by area in decreasing order 
        poly_data.sort(key=lambda x: x['area:'], reverse=True)
        sorted_labels = [poly['label'] for poly in poly_data]    
    
        start("Writing areas to disk")
        with open(os.path.join(dir, f"{file}.csv"), "w") as f:
            f.write(','.join(sorted_labels))
        end()