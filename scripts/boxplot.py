#!/usr/bin/env python
import click
import matplotlib.pyplot as plt
import pandas as pd

FIELDS = ["Friends", "Family", "Disease history", "Healthcare specialists", "Government", "Health information", "Educational background", "Foreign lanugages", "Digital technologies", "Money for services", "Money for healthy food", "Stress handling ability"]

def clean(s):
    while s[0].isalpha():
        s = s[1:]
    return int(s)

@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("--invert", is_flag=True, default=False)
@click.option("--top-limit", type=float)
@click.option("--sorted", is_flag=True, default=False)
def boxplot(file, start, end, invert, top_limit, sorted):
    df = pd.read_excel(file)
    d = df.iloc[:,start:end]
    if sorted:
        column2field = {i+1: field for i, field in enumerate(FIELDS)}
        medians = d.median().sort_values(ascending=True)        
        d = d[medians.index]
        fields = [column2field[clean(col)] for col in medians.index.tolist()]
    else:
        fields = FIELDS
    plt.figure(figsize=(12,6))
    d.boxplot()
    if invert:
        plt.gca().invert_yaxis()
    plt.xticks(range(1, 12+1), fields, rotation=45)
    if top_limit:
        plt.ylim(top=top_limit)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    boxplot()
