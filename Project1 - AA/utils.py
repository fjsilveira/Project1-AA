import numpy as np
import pandas as pd


def create_dataset():
    df = pd.read_csv('X_train.csv')

    zero_cols = ['x_1', 'y_1', 'v_x_1', 'v_y_1', 'x_2', 'y_2', 'v_x_2', 'v_y_2', 'x_3', 'y_3', 'v_x_3', 'v_y_3']

    df = df[~(df[zero_cols] == 0).all(axis=1)]

    df = df[['t', 'x_1', 'y_1', 'x_2', 'y_2','x_3', 'y_3']]

    df.to_csv('dataset.csv', index=False)


def create_x_train():
    df = pd.read_csv('dataset.csv')

    values = df[['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values

    t0_line = values[0, 1:7]

    for i in range(0, len(values)):
        print(i)
        if values[i, 0] == 0.0:
            t0_line = values[i, 1:7]
        else:
            values[i, 1:7] = t0_line

    df_processed = pd.DataFrame(values, columns=df.columns)

    df_processed.to_csv('processed_X_train.csv', index=False)

def split_initial(X,y):

    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

def split_final(X,y):

    test_size = 0.2
    split_index = int(len(X) * test_size)


    X_train_split = X[split_index:]
    X_test_split = X[:split_index]

    y_train_split = y[split_index:]
    y_test_split = y[:split_index]

    return X_train_split, X_test_split, y_train_split, y_test_split


def split_middle(X, y):
    test_size = 0.2
    middle_start = int(len(X) * 0.4)
    middle_end = int(len(X) * 0.6)

    X_train = np.concatenate((X[:middle_start], X[middle_end:]))
    y_train = np.concatenate((y[:middle_start], y[middle_end:]))

    X_test = X[middle_start:middle_end]
    y_test = y[middle_start:middle_end]

    return X_train, X_test, y_train, y_test

def split_data(X,y):

    initial_slip = split_initial(X,y)
    middle_slip = split_middle(X,y)
    final_slip = split_final(X,y)

    return [initial_slip, middle_slip, final_slip]


