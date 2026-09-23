# 📈 Stock Price Prediction Using LSTM

A Deep Learning project that uses **Long Short-Term Memory (LSTM)** to predict stock closing prices based on historical time-series data. The project compares a baseline LSTM architecture with a modified Stacked LSTM architecture using **Facebook (FB)** and **IBM** stock data.

## 🎯 Project Overview

The objective is to build an LSTM model that learns temporal patterns from historical stock prices and predicts the next trading day's closing price.

The analysis follows a time-series forecasting workflow, including data exploration, preprocessing, windowing, model development, and evaluation.

## 📊 Dataset

The dataset contains historical daily stock data obtained from **Yahoo Finance** using `yfinance`, with data available up to April 1, 2020.

This project uses:

* **FB (Facebook, Inc.)** — 2012–2020, 1,980 records
* **IBM (International Business Machines Corporation)** — 1962–2020, 14,663 records

Only the `Date` and `Close` columns are used for prediction.

## 🔧 Methodology

### Data Preprocessing

* Convert `Date` into datetime format
* Use `Close` as the prediction target
* Split the data chronologically, with the last one year used as the test set
* Normalize prices using `MinMaxScaler`
* Create time-series sequences using **window size = 5** and **horizon = 1**
* Split the training data into **90% training** and **10% validation**

### Models

**Baseline LSTM**

* 1 LSTM layer
* 50 units
* ReLU activation
* 1 Dense output neuron
* Adam optimizer
* MSE loss

**Modified Stacked LSTM**

* LSTM with 128 units
* Dropout (0.2)
* LSTM with 64 units
* Dropout (0.2)
* Batch Normalization
* Dense layer with 32 units
* Dense output layer
* Adam optimizer with learning rate 0.001
* Early Stopping and ReduceLROnPlateau

## 📈 Results

The models were evaluated on the test set using **RMSE, MAE, and MAPE**.

| Stock | Model                 | RMSE (USD) | MAE (USD) |    MAPE |
| ----- | --------------------- | ---------: | --------: | ------: |
| FB    | Baseline LSTM         |     5.5773 |    3.9227 | 2.1159% |
| FB    | Modified Stacked LSTM |     9.6935 |    8.7027 | 4.5225% |
| IBM   | Baseline LSTM         |     2.7909 |    1.8313 | 1.4035% |
| IBM   | Modified Stacked LSTM |    12.5564 |   12.0622 | 8.7738% |

The baseline LSTM produced lower error values on both datasets. The modified architecture generated smoother predictions but had more difficulty following rapid changes in stock prices.

## 💡 Key Findings

* The baseline LSTM was able to capture the general movement of both FB and IBM stock prices.
* Increasing model complexity did not automatically improve prediction performance.
* The modified Stacked LSTM tended to produce over-smoothed predictions and was less responsive to sharp price movements.
* Larger prediction errors occurred during periods with significant changes in the stock price.

## 🛠️ Tools & Technologies

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

## 📚 Data Source

Historical stock data from Yahoo Finance, provided through the course dataset.

This project was completed as part of the **Deep Learning Final Examination**.
