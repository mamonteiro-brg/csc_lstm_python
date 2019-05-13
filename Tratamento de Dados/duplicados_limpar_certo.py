import csv
from collections import OrderedDict

import numpy

with open('duplicados_arranjar.csv', 'r+', encoding='ISO-8859-1') as dups, \
        open('sem_dups5.csv', 'w+', encoding='ISO-8859-1') as nops:
    csvDups = csv.reader(dups, delimiter=',')
    next(csvDups, None)
    d = OrderedDict()
    l = []
    for line in csvDups:
        key_tuple = (line[5], line[6], line[7], line[8], line[10])
        if key_tuple in d.keys():
            d[key_tuple].append((line[3], line[4], line))
        else:
            d[key_tuple] = [(line[3], line[4], line)]

    for key in d.keys():
        mean_meters = int(numpy.mean(list(map(lambda x: int(x[0]), d[key]))))
        mean_delay = int(numpy.mean(list(map(lambda x: int(x[1]), d[key]))))
        d[key][0][2][3] = mean_meters
        d[key][0][2][4] = mean_delay
        d[key][0][2][12] += '\n'
        nops.write(','.join(list(map(str, d[key][0][2]))))