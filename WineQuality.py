import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

def RedWine():
    df = pd.read_csv('winequality-red.csv', sep=';')
    target = pd.DataFrame(df, columns=["quality"])

    # no categorical variables..
    # each feature is a number
    # Given the set of values for features, we have to predict the
    # quality of the wine. Finding correlation of each feature with our target variable - quality
    correlations = df.corr(method='pearson')['quality'].drop('quality')

    # print(correlations())
    # print(df)
    # print(target)
    # important = [(x, correlations[x]) for x in correlations.index if correlations[x] > 0.5]
    # print(important)

    print("Heatmap should show")
    sns.heatmap(df.corr())
    plt.show()

    # In this step, we check out different features from the dataset
    # and see how correlated with "quality" they are
    x = df[["fixed acidity", "volatile acidity"]]
    y = target["quality"]

    # Adding a constant:
    # x = sm.add_constant(x)
    # Forces our model to go through the origin

    # get model
    model = sm.OLS(y, x).fit()

    # get predictions
    predictions = model.predict(x)

    # print stats
    print("Line 45:\n",model.summary())

def WhiteWine():
    """ Gets data for white wine """
    # Get dataframe from csv
    df = pd.read_csv('winequality-white.csv', sep=";")

    # Get the target in a seperate dataframe
    target = pd.DataFrame(df, columns=["quality"])

    # Set X and Y
    X = df.drop(columns="quality")
    y = target["quality"]

    # lm.fit() function fits a linear model that we can use to make predictions
    lm = linear_model.LinearRegression()
    model = lm.fit(X,y)
    predictions = lm.predict(X)
    print("{}\n\n{}\n\nModel:\n{}\n\nPrediction:\n{}".format(X, y, model, predictions))
    print("Prediction Score: {}".format(lm.score(X,y)))

    # Next step:
    # Gradient Descent: https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f


def main():
    RedWine()
    # WhiteWine()


if __name__ == "__main__":
    main()