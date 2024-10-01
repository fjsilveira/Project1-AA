import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('X_train.csv')

#print(data.head())

idx = np.hstack((0, data[data.t == 10].index.values + 1))

idx.shape, data.t.min(), data.t.max()

k = np.random.randint(idx.shape[0])
pltidx = range(idx[k], 257+idx[k])
pltsquare = idx[k]

plt.plot(data.x_1[pltidx], data.y_1[pltidx])
plt.plot(data.x_2[pltidx], data.y_2[pltidx])
plt.plot(data.x_3[pltidx], data.y_3[pltidx])

plt.plot(data.x_1[pltsquare], data.y_1[pltsquare], "s")
plt.plot(data.x_2[pltsquare], data.y_2[pltsquare], "s")
plt.plot(data.x_3[pltsquare], data.y_3[pltsquare], "s")

plt.xlabel('X')
plt.ylabel('Y')

plt.show()

