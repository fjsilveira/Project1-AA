import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import split_data
import numpy as np


def add_variables(X):
    x1, y1, x2, y2, = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    angle1 = np.arctan2(y2 - y1, x2 - x1)

    X = np.column_stack((X, angle1))

    return X

def submit_model():
    X_train_file = pd.read_csv('processed_x_train.csv')
    y_train_file = pd.read_csv('dataset.csv')

    X_train = X_train_file[['x_1', 'y_1', 'x_2', 'y_2', 't']].values
    y_train = y_train_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values

    X_train = add_variables(X_train)

    X_test_file = pd.read_csv('X_test.csv')
    X_test_file.rename(
        columns={'x0_1': 'x_1', 'y0_1': 'y_1', 'x0_2': 'x_2', 'y0_2': 'y_2', 'x0_3': 'x_3', 'y0_3': 'y_3',
                 'Id': 'Id'}, inplace=True)
    X_test = X_test_file[['x_1', 'y_1', 'x_2', 'y_2', 't']].values

    X_test = add_variables(X_test)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', KNeighborsRegressor(n_neighbors=75))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_pred_df.insert(0, 'Id', range(len(y_pred_df)))

    y_pred_df.to_csv('knn-80-angle.csv', index=False)






def test_model():
    X_file = pd.read_csv('processed_x_train.csv')
    y_file = pd.read_csv('dataset.csv')

    X = X_file[['x_1', 'y_1', 'x_2', 'y_2', 't']].values
    y = y_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values

    X = add_variables(X)


    splits = split_data(X, y)

    RMSE = []

    i = 0
    for k in range(30, 60):

        for split in splits:
            X_train, X_test, y_train, y_test = split

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', KNeighborsRegressor(n_neighbors=k))
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            RMSE.append(mean_squared_error(y_pred, y_test))

        print(f'k {k}: Mean Squared Error: {np.mean(RMSE)}')



def main(option):
    if option == 'test':
        test_model()
    if option == 'submit':
        submit_model()


if __name__ == "__main__":
    #option = 'submit'
    option = 'test'
    main(option)
