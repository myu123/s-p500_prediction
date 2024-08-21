import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

# Fetch the S&P 500 data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

# Clean up the data by removing unnecessary columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create the 'Target' column to indicate whether the next day's closing price is higher than today's closing price
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

# Define horizons for rolling averages and trends
horizons = [2, 5, 60, 250, 1000]
predictors = []

# Calculate rolling averages and trends for each horizon
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    predictors += [ratio_column, trend_column]

# Remove rows with missing values
sp500 = sp500.dropna()

# Initialize the RandomForest model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


def predict(train, test, predictors, model):
    """
    Train the model on the training data and make predictions on the test data.

    Parameters:
    - train: DataFrame containing training data
    - test: DataFrame containing test data
    - predictors: List of column names to be used as predictors
    - model: Initialized machine learning model

    Returns:
    - combined: DataFrame containing actual targets and predicted values
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    """
    Perform backtesting to evaluate the model performance over historical data.

    Parameters:
    - data: DataFrame containing the complete dataset
    - model: Initialized machine learning model
    - predictors: List of column names to be used as predictors
    - start: Initial number of rows to be used for training
    - step: Number of rows to be added to the training set in each iteration

    Returns:
    - all_predictions: DataFrame containing predictions for all test sets
    """
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


def get_precision_score():
    """
    Calculate and print the precision score of the model based on backtesting results.
    """
    predictions = backtest(sp500, model, new_predictors)
    print(predictions["Predictions"].value_counts())
    print(precision_score(predictions["Target"], predictions["Predictions"]))


def predict_next_day(data, model, predictors):
    """
    Train the model on the entire dataset and predict the next day's stock movement.

    Parameters:
    - data: DataFrame containing the complete dataset
    - model: Initialized machine learning model
    - predictors: List of column names to be used as predictors

    Returns:
    - prediction: Integer (1 for up, 0 for down) indicating the predicted stock movement
    """
    model.fit(data[predictors], data["Target"])
    last_data_point = data.iloc[-1:]
    prediction = model.predict_proba(last_data_point[predictors])[:, 1]
    prediction = 1 if prediction >= 0.6 else 0
    return prediction


if __name__ == "__main__":
    # Predict the next day's stock movement
    next_day_prediction = predict_next_day(sp500, model, predictors)
    print("Prediction for the next day's movement (1 = Up, 0 = Down):", next_day_prediction)
