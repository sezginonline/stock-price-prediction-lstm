import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import random

# Set a fixed random seed for reproducibility
seed_value = 7
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Define the ticker symbol
ticker_symbol = "ALARK.IS"

# Get the data for the last 1 year
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Handle missing data if any
data = data.dropna()

# Prepare the data for prediction
data['Date'] = data.index
data['Date_ordinal'] = data['Date'].apply(lambda x: x.toordinal())
data = data[['Date', 'Adj Close']]

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Adj Close']])

# Convert the data to sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i : (i + sequence_length)]
        target = data[i + sequence_length]
        sequences.append((sequence, target))
    return np.array([s[0] for s in sequences]), np.array([s[1] for s in sequences])

sequence_length = 10
X, y = create_sequences(data_scaled, sequence_length)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(y_test)

mse = mean_squared_error(y_test, test_predictions)
print("Mean Squared Error on test data:", mse)

# Predict x days into the future
future_days = 30

# Create a sequence for prediction
future_sequence = data_scaled[-sequence_length:]
future_sequence = future_sequence.reshape((1, sequence_length, 1))

# Predict the next x days
predicted_prices = []
predicted_dates = []

for day in range(1, future_days + 1):
    predicted_price = model.predict(future_sequence)
    future_sequence = np.append(future_sequence[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
    
    # Inverse transform the predicted price
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]
    predicted_prices.append(predicted_price)

    # Calculate the date of the predicted price
    predicted_date = data['Date'].iloc[-1] + pd.DateOffset(days=day)
    predicted_dates.append(predicted_date)

# Display day-by-day predictions with percentage difference
for i in range(future_days):
    predicted_date_str = predicted_dates[i].strftime('%Y-%m-%d')
    predicted_price = predicted_prices[i]
    actual_price = data['Adj Close'].iloc[-1]

    # Calculate percentage difference
    percentage_diff = ((predicted_price - actual_price) / actual_price) * 100

    print(f"Predicted price for {predicted_date_str}: {predicted_price}  Percentage diff: {percentage_diff:.2f}%")

# Plot the predictions along with future predictions
plt.figure(figsize=(12, 8))

# Plot actual prices
plt.plot(data['Date'], data['Adj Close'], label='Actual Prices', marker='o')

# Plot predicted prices on the testing set
test_dates = data['Date'][split:split + len(y_test)]
plt.plot(test_dates, test_predictions, label='Predicted Prices (Test Set)', marker='o')

# Plot future predictions
future_dates = data['Date'].iloc[-1] + pd.to_timedelta(np.arange(1, future_days + 1), unit='D')
plt.plot(future_dates, predicted_prices, label='Future Predictions', linestyle='dashed', marker='o')

plt.title('Stock Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Adj Closing Price')
plt.legend()
plt.show()
