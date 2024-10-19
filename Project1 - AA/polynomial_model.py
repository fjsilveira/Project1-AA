from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, LinearRegression
import pandas as pd
import numpy as np
from utils import split_data


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


def angle_between_bodies_dataset():
    X_file = pd.read_csv('processed_x_train.csv')
    y_file = pd.read_csv('X_train.csv')

    X = X_file[['x_1', 'y_1', 'x_2', 'y_2', 't']].values
    y = y_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values

    x1, y1, x2, y2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    angle = np.arctan2(y2 - y1, x2 - x1)

    X = np.column_stack((X, angle))



def validate_poly_regression(X,y,sample_size):
    splits = split_data(X, y)

    for degree in range(1, 15):

        RMSE = []
        features = 0

        for split in splits:


            X_train, X_test, y_train, y_test = split

            size = int(len(X) * sample_size)
            X_train = X_train[:size]
            y_train = y_train[:size]
            X_test = X_test[:size]
            y_test = y_test[:size]

            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
                #('regressor', RidgeCV(alphas=np.logspace(-3, 3, 7), store_cv_results=True))
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            RMSE.append(rmse)

            features = pipeline.named_steps["poly"].n_output_features_

            # plot_y_yhat(y_test, y_pred, plot_title='baseline')


            # best_alpha = pipeline.named_steps['regressor'].alpha_
            # print(f'O melhor valor de alpha é: {best_alpha}')
            break


        final_rmse = np.mean(RMSE)

        print(f'Number of features for degree {degree}: {features}')

        print(f'Degree: {degree}, RMSE: {final_rmse}')

def main():
    X_file = pd.read_csv('processed_X_train.csv')
    y_file = pd.read_csv('dataset.csv')

    X = X_file[['x_1', 'y_1', 'x_2', 'y_2', 't']].values
    y = y_file[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values

    validate_poly_regression(X,y, 0.2)

if __name__ == "__main__":
    main()
