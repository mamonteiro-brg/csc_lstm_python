with open('ola.csv', 'r+', encoding="ISO-8859-1") as nuls:
    d = {}
    for line in nuls:
        line = line.strip('\n')
        print(type(line))
        if line in d.keys():
            d[line] = d[line] +1
        else:
            d[line]=0

    for key in d.keys():
        if int(d[key]) > 7:
            print(key)

    print(d['8 11 2018 9 0'])
    print(d['9 19 2018 14 0'])