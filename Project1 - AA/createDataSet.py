import pandas as pd
import numpy as np

dataframe = pd.read_csv('X_test.csv')

dataframe.rename(columns={'x0_1': 'x_1', 'y0_1': 'y_1', 'x0_2': 'x_2', 'y0_2': 'y_2', 'x0_3': 'x_3', 'y0_3': 'y_3',
                          'Id': 'Id'}, inplace=True)

data = dataframe[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 't', 'Id']]


block_size = 257
num_blocks = len(data) // block_size

new_data_array = np.empty((len(data), data.shape[1]))

for i in range(num_blocks):

    start_idx = i * block_size
    end_idx = start_idx + block_size

    t0_values = data.iloc[start_idx].to_numpy()

    new_data_array[start_idx:end_idx, :6] = t0_values[:6]
    new_data_array[start_idx:end_idx, 6] = data.iloc[start_idx:end_idx, 6].to_numpy()

    new_data_array[start_idx:end_idx, 7] = data.iloc[start_idx:end_idx, 7].to_numpy()

new_data = pd.DataFrame(new_data_array, columns=data.columns)

new_data.to_csv('processed_x_test.csv', index=False)
