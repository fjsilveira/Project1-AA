from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, LinearRegression
import pandas as pd
import numpy as np


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


def validate_poly_regression(X_train, y_train, X_test, y_test):
    best_rmse = float('inf')
    best_model = None
    best_degree = None

    for degree in range(1, 15):

        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        plot_y_yhat(y_test, y_pred, plot_title='baseline')

        print(f'Degree: {degree}, RMSE: {rmse}')

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_degree = degree

        print(f'Number of features for degree {degree}: {pipeline.named_steps["poly"].n_output_features_}')
    return best_model, best_rmse, best_degree


def main():
    X_file = pd.read_csv('processed_x_train.csv')
    y_file = pd.read_csv('X_train.csv')

    X = X_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 't']].values
    y = y_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sample_size = int(len(X_train) * 0.01)
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]
    X_test_sample = X_test[:sample_size]
    y_test_sample = y_test[:sample_size]

    best_model, best_rmse, best_degree = validate_poly_regression(X_train_sample, y_train_sample, X_test_sample,
                                                                  y_test_sample)

    print(f'Best RMSE: {best_rmse} at degree {best_degree}')


if __name__ == "__main__":
    main()
