#!/usr/bin/env python
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boxplot import FIELDS

@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("--output", default=None, type=click.Path())
@click.option("--format", default="pdf")
@click.option("--gender", default=None, type=int)
@click.option("--prefixes", type=str, default=None)
def corr(file, start, end, output, format, gender, prefixes):
    df = pd.read_excel(file)
    if gender is not None:
        df = df[df['gender'] == gender]
    if prefixes is not None:
        prefixes = prefixes.split(",")
        mask = df["id"].str.startswith(tuple(prefixes))
        df = df.loc[mask]
    selected_columns = df.iloc[:,start:end].columns.tolist()
    selected_columns.append("age")
    selected_columns.append("gender")
    FIELDS.append("age")
    FIELDS.append("gender")
    # Compute correlation matrix
    corr_matrix = df[selected_columns].corr()

    # Plot correlation matrixs
    fig, ax = plt.subplots(figsize=(12, 9))
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    ax.set_xticks(range(len(selected_columns)))
    ax.set_xticklabels(FIELDS, rotation=45, ha='left')
    ax.set_yticks(range(len(selected_columns)))
    ax.set_yticklabels(FIELDS)

    for (i, j), val in np.ndenumerate(corr_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    plt.title('Correlation Matrix', pad=20)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format=format)

if __name__ == "__main__":
    corr()
