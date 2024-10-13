import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from utils import split_initial


def plot_y_yhat(y_test, y_pred, plot_title="plot"):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    MAX = 500
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test), MAX, replace=False)
    else:
        idx = np.arange(len(y_test))
    plt.figure(figsize=(10, 10))
    for i in range(6):
        x0 = np.min(y_test[idx, i])
        x1 = np.max(y_test[idx, i])
        plt.subplot(3, 2, i + 1)
        plt.scatter(y_test[idx, i], y_pred[idx, i])
        plt.xlabel('True ' + labels[i])
        plt.ylabel('Predicted ' + labels[i])
        plt.plot([x0, x1], [x0, x1], color='red')
        plt.axis('square')
    plt.show()


def submit_base_model():
    X_train_file = pd.read_csv('processed_x_train.csv')
    y_train_file = pd.read_csv('dataset.csv')

    X_train = X_train_file[['x_1', 'y_1', 'x_2', 'y_2', 't']].values
    y_train = y_train_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values

    X_test_file = pd.read_csv('X_test.csv')
    X_test_file.rename(
        columns={'x0_1': 'x_1', 'y0_1': 'y_1', 'x0_2': 'x_2', 'y0_2': 'y_2', 'x0_3': 'x_3', 'y0_3': 'y_3',
                 'Id': 'Id'}, inplace=True)
    X_test = X_test_file[['x_1', 'y_1', 'x_2', 'y_2', 't']].values

    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=6, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_pred_df.insert(0, 'Id', range(len(y_pred_df)))

    y_pred_df.to_csv('reduced-poly-6-model.csv', index=False)


def test_base_model():
    X_file = pd.read_csv('processed_x_train.csv')
    y_file = pd.read_csv('dataset.csv')

    X = X_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 't']].values
    y = y_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values

    X_train, X_test, y_train, y_test = split_initial(X,y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    plot_y_yhat(y_test, y_pred, plot_title='baseline')

    return


def main(option):
    if option == 'test':
        test_base_model()
    if option == 'submit':
        submit_base_model()


if __name__ == "__main__":
    #option = 'submit'
    option = 'test'
    main(option)
