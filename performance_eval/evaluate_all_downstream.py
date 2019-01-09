#!/usr/bin/env python3.5

from sys import argv, exit
import csv
from downstream_classification import main as classify

if len(argv) < 2:
    print("Usage: {} FILE_WITH_FILENAMES".format(argv[0]))
    exit()

with open(argv[1], "r") as f:
    filenames = f.readlines()

fieldnames = ['approach', 'mechanism', 'missingratio', 'n', 'accuracy']

with open("downstream_results.csv", mode='w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

for fn in filenames:
    fn = fn.rstrip()
    print("Training {}".format(fn))
    _, approach, mechanism, missing_ratio, n = fn.split('.pkl')[0].split('_')
    accuracy = classify([fn])

    with open("downstream_results.csv", mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({
            'approach': approach,
            'mechanism': mechanism,
            'missingratio': missing_ratio,
            'n': n,
            'accuracy': accuracy
            })
    print("{} done, accuracy {}".format(fn, accuracy))
