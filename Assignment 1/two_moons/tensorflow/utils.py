import csv
import numpy as np

def read_dataset():
    coordinates = []
    classes = []
    with open('../two_moon.txt') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx in range(0, 4):
                pass
            else:
                coordinates.append([float(row[0]), float(row[1])])
                classes.append(int(row[2]))
    return (np.array(coordinates), np.array(classes))
