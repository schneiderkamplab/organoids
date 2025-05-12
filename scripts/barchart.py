#!/usr/bin/env python
import click
import matplotlib.pyplot as plt
import pandas as pd

from boxplot import FIELDS

@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("--stacked", is_flag=True, default=False)
@click.option("--sorted", is_flag=True, default=False)
@click.option("--output", default=None, type=click.Path())
@click.option("--format", default="pdf")
@click.option("--gender", default=None, type=int)
@click.option("--prefixes", type=str, default=None)
def barchart(file, start, end, stacked, sorted, output, format, gender, prefixes):
    df = pd.read_excel(file)
    if gender is not None:
        df = df[df['gender'] == gender]
    if prefixes is not None:
        prefixes = prefixes.split(",")
        mask = df["id"].str.startswith(tuple(prefixes))
        df = df.loc[mask]
    d = df.iloc[:,start:end]
    counts_df = pd.DataFrame({col: df[col].value_counts().reindex(range(1,12+1), fill_value=0)
                            for col in d.columns.tolist()})
    if sorted:
        column2field = {i+1: field for i, field in enumerate(FIELDS)}
        counts_df['total'] = counts_df.sum(axis=1)
        counts_df = counts_df.sort_values('total', ascending=False).drop('total', axis=1)
        fields = [column2field[col] for col in counts_df.index.tolist()]
    else:
        fields = FIELDS
    ax = counts_df.plot(kind='bar', stacked=stacked, figsize=(12, 7))

    plt.title('Top health priorities')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    ax.set_xticklabels(fields)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend([f"top-{i}" for i in range(1, end-start+1)], title='Priorities')
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format=format)

if __name__ == "__main__":
    barchart()
