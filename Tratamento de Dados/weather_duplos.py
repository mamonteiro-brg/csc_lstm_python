import csv
import numpy

with open('weather_nulos.csv', 'r+', encoding="ISO-8859-1") as nuls, \
        open('weather_sem_nulos.csv', 'r+', encoding="ISO-8859-1") as nots, \
        open('weather_means.csv', 'w+') as meds:
    csv_nuls = csv.reader(nuls, delimiter=',')
    headerSave = True
    csv_nots = csv.reader(nots, delimiter=',')
    next(csv_nots, None)
    d = {}
    stock = []
    for l in csv_nots:
        # Mes, Dia, Ano, Hora
        stock.append((int(l[2]) * 1000000 + int(l[0]) * 10000 + int(l[1]) * 100 + int(l[3])))
        d[str(int(l[2]) * 1000000 + int(l[0]) * 10000 + int(l[1]) * 100 + int(l[3]))] = l
    for nul in csv_nuls:
        if headerSave:
            meds.write((','.join(nul)) + '\n')
            headerSave = False
            continue

        nulVal = list(map(int, nul[:4]))
        nTime = nulVal[2] * 1000000 + nulVal[0] * 10000 + nulVal[1] * 100 + nulVal[3]

        targets = list(map(str, filter(lambda x: x < nTime, stock)))
        targets.sort(reverse=True)
        targets = targets[:3]

        line = list(map(str, map(numpy.mean, list(
            map(lambda x: list(map(int, x)), map(list, zip(*list(map(lambda x: d[x][8:14], targets)))))))))
        meds.write(','.join(nul[:8] + list(map(str,map(int,map(float,line)))) + nul[14:]))
        meds.write('\n')
