import csv
import numpy

LIMIT = 8


class Queue:
    def __init__(self):
        self.q = []

    def __sizeof__(self):
        return len(self.q)

    def queue(self, val):
        if self.__sizeof__() > LIMIT:
            self.q.pop()
        self.q.insert(0, val)

    def __str__(self):
        return self.q.__str__()

    def full(self):
        return self.__sizeof__() == LIMIT


with open('so_valores_nulos.csv', 'r+', encoding="ISO-8859-1") as nuls, \
        open('sem_valores_nulos.csv', 'r+', encoding="ISO-8859-1") as nots, \
        open('medias_valores.csv', 'w+') as meds:
    csv_nuls = csv.reader(nuls, delimiter=',')
    headerSave = True
    csv_nots = csv.reader(nots, delimiter=',')
    next(csv_nots, None)
    d = {}

    for l in csv_nots:
        d[(l[5] + '_' + l[3] + '_' + l[1] + '_' + l[0] + '_' + l[2] + '_' + l[8])] = l

    for nul in csv_nuls:
        if headerSave:
            meds.write((','.join(nul)) + '\n')
            headerSave = False
            continue
        nulVal = list(map(int, nul[:3]))
        nTime = nulVal[2] * 10000 + nulVal[0] * 100 + nulVal[1]
        matchWith = list(filter(lambda x: x.startswith(nul[5] + '_' + nul[3]), d.keys()))
        matched = list(map(lambda y: '_'.join(y[2::].split('_')[1::]), matchWith))

        roads = set(map(lambda s: s.split('_')[3:][0], matched))
        for road in roads:
            individualRoad = list(filter(lambda z: z.endswith(road), matched))
            q = Queue()
            for val in individualRoad:
                aday, amonth, ayear = list(map(int, val.split('_')[:3]))
                actualTime = ayear * 10000 + amonth * 100 + aday
                if actualTime > nTime and q.full():
                    break
                omega = [nul[5], nul[3], aday, amonth, ayear, road]
                q.queue('_'.join(map(str, omega)))
            matrix = []
            for i in q.q:
                matrix.append(d[i][9:])
            results = []
            for value in list(map(list, zip(*matrix))):
                results.append(int(numpy.mean(list(map(int, value)))))
            meds.write(','.join(nul[:8] + [road] + list(map(str, results)) + [nul[15] + '\n']))

