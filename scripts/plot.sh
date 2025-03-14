#!/bin/bash
./barchart.py ranks.xlsx 25 28 --sorted
#./barchart.py ranks.xlsx 25 31 --sorted --stacked
./corr.py ranks.xlsx 1 13
#./boxplot.py ranks.xlsx 1 13 --top-limit 100000
./boxplot.py ranks.xlsx 13 25 --invert --sorted

