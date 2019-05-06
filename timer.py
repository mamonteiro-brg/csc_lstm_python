months = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
year = {2018: [7, 8, 9, 10, 11, 12], 2019: [1, 2]}
init = 6
flag = False
with open("timer.csv", "w+") as t:
    for y in year.keys():
        for m in year[y]:
            for d in range(1, months[m] + 1):
                if d == 24:
                    flag = True
                for h in range(0, 24):
                    if flag:
                        t.write(str(m) + "," + str(d) + "," + str(y) + "," + str(h) + ",0," + str(
                            divmod(init, 7)[1] + 2) + "\n")
                init += 1