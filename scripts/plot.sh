#!/bin/bash
DIR=$(dirname $0)
FILE=$1
$DIR/barchart.py $FILE 25 28 --sorted --output barchart.pdf
#$DIR/barchart.py $FILE 25 31 --sorted --stacked
$DIR/corr.py $FILE 1 13 --output corr.pdf
#$DIR/boxplot.py $FILE 1 13 --top-limit 100000
$DIR/boxplot.py $FILE 13 25 --invert --sorted --output boxplot.pdf
