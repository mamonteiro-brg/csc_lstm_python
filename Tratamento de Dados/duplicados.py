import pandas as pd
import numpy as np

from_dupli = []
to_dupli = []
i = 0
j = 0


df = pd.read_csv('duplicados_arranjar.csv',encoding="ISO-8859-1")

col_from = df['dupl_from']
col_to = df['dupl_to']

for fr in col_from:
    from_dupli.append(fr)

for to in col_to:
    to_dupli.append(to)

print(to_dupli)

#Tratar array dos duplicados do from

for y in to_dupli:
    if i==0:
        i = i + 1
    else:
        if to_dupli[i] == 1:
            to_dupli[i-1] = 1
        i = i + 1

print(to_dupli)

# Tratar array dos duplocados do to

print('FROM##')
print(from_dupli)

for z in from_dupli:
    if j == 0:
        j = j + 1
    else:
        if from_dupli[j] == 1:
            from_dupli[j-1] = 1
        j = j + 1

print(from_dupli)


df['new_from'] = pd.Series(np.random.randn(len(df['dupl_from'])), index=df.index)
df['new_to'] = pd.Series(np.random.randn(len(df['dupl_from'])), index=df.index)

print(df)

df['new_from'] = pd.DataFrame(from_dupli)
df['new_to'] = pd.DataFrame(to_dupli)

print(df)

df.to_csv("duplic_trat.csv", sep=',')