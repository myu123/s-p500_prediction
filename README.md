# Predicting S&P 500 Market Movement

This repository contains a Python program designed to predict the next day's movement of the S&P 500 index. The model leverages historical market data and machine learning techniques to provide insights into potential market trends.

## Features

- **Data Retrieval**: Fetches historical data for the S&P 500 index using the `yfinance` library.
- **Data Preparation**: Cleans and processes data to create target variables and predictors.
- **Model Training**: Utilizes a Random Forest Classifier to train on historical data.
- **Rolling Averages and Trends**: Calculates rolling averages and trend indicators over multiple time horizons to improve prediction accuracy.
- **Backtesting**: Includes functions for backtesting the model to evaluate its performance over historical data.
- **Prediction**: Trains the model on the most recent data to predict the next day's market movement.

## Installation

To run this project, you will need Python 3.x installed along with the following libraries:
- `yfinance`
- `scikit-learn`
- `pandas`

You can install the required libraries using pip:

```bash
pip install yfinance scikit-learn pandas
