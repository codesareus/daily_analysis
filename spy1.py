import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Function to fetch stock data
def fetch_stock_data(ticker):
    # Fetch minute-level data for the specified stock (including premarket)
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m", prepost=True)  # Include premarket data
    return data

# Function to perform regression analysis
def perform_regression(data, degree=1):
    # Prepare data (use only the most recent 300 points)
    data_recent = data.tail(300)  # Get the most recent 300 data points
    X = np.arange(len(data_recent)).reshape(-1, 1)  # Time as feature
    y = data_recent['Close'].values  # Closing prices as target

    # Polynomial regression
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    # Calculate R-squared
    r2 = r2_score(y, y_pred)

    return X, y, y_pred, r2, data_recent

# Function to calculate percentage change
def calculate_percentage_change(current_price, previous_close):
    return ((current_price - previous_close) / previous_close) * 100

# Streamlit app
def main():
    st.title("Stock Price Regression Analysis")
    st.write("This app fetches minute-level stock prices (including premarket data) and performs linear and polynomial regression analysis.")

    # Input box for user to enter stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., SPY, AAPL, TSLA):", value="SPY").upper()

    # Add a button to refresh data
    if st.button("Refresh Data"):
        st.cache_data.clear()  # Clear cached data to force a fresh fetch

    # Fetch data for the user-specified stock
    data = fetch_stock_data(ticker)
    if data.empty:
        st.error(f"Failed to fetch data for {ticker}. Please check the ticker and try again.")
        return

    # Get the current price (last available price in the data)
    current_price = data['Close'].iloc[-1]

    # Get the previous day's close price
    previous_close = data['Close'].iloc[0]  # First price in the dataset (previous day's close)

    # Calculate percentage change
    percentage_change = calculate_percentage_change(current_price, previous_close)

    # Display the percentage change message
    st.write("### Current Price vs Previous Close")
    if percentage_change >= 0:
        st.success(f"🟢 The current price is **{current_price:.2f}**, which is **+{percentage_change:.2f}%** higher than the previous close price of **{previous_close:.2f}**.")
    else:
        st.error(f"🔴 The current price is **{current_price:.2f}**, which is **{percentage_change:.2f}%** lower than the previous close price of **{previous_close:.2f}**.")

    # Perform linear regression (using only the most recent 300 points)
    X, y, y_pred_linear, r2_linear, data_recent = perform_regression(data, degree=1)

    # Perform polynomial regression with default degree 9 (using only the most recent 300 points)
    st.write("### Polynomial Regression Analysis")
    degree = st.slider("Select Polynomial Degree", min_value=2, max_value=10, value=9)  # Default degree set to 9
    X, y, y_pred_poly, r2_poly, _ = perform_regression(data, degree=degree)

    # Calculate residuals and standard deviation for the polynomial model
    residuals = y - y_pred_poly
    std_dev = np.std(residuals)

    # Calculate exponential moving averages
    data_recent['EMA_9'] = data_recent['Close'].ewm(span=9, adjust=False).mean()
    data_recent['EMA_20'] = data_recent['Close'].ewm(span=20, adjust=False).mean()

    # Determine the trend message
    if current_price > data_recent['EMA_9'].iloc[-1] and data_recent['EMA_9'].iloc[-1] > data_recent['EMA_20'].iloc[-1]:
        trend_message = f"{ticker} trend is UP"
        trend_color = "green"
    elif current_price < data_recent['EMA_9'].iloc[-1] and data_recent['EMA_9'].iloc[-1] < data_recent['EMA_20'].iloc[-1]:
        trend_message = f"{ticker} trend is DOWN"
        trend_color = "red"
    else:
        trend_message = f"{ticker} trend is NEUTRAL"
        trend_color = "gray"

    # Plot both linear and polynomial regression results on the same graph
    st.write("### Combined Regression Plot (Most Recent 300 Points)")
    fig, ax = plt.subplots()
    ax.plot(X, y, color="gray", label="Actual Prices")  # Actual prices as a gray line plot
    ax.plot(X, y_pred_linear, color="red", label=f"L.R. (R² = {r2_linear:.4f})")
    ax.plot(X, y_pred_poly, color="green", label=f"P.R. (Degree {degree}, R² = {r2_poly:.4f})")

    # Draw bands for 1, 2, and 3 standard deviations from the polynomial model
    ax.fill_between(X.flatten(), y_pred_poly - std_dev, y_pred_poly + std_dev, color="lightgreen", alpha=0.3, label="")
    ax.fill_between(X.flatten(), y_pred_poly - 2*std_dev, y_pred_poly + 2*std_dev, color="green", alpha=0.2, label="")
    ax.fill_between(X.flatten(), y_pred_poly - 3*std_dev, y_pred_poly + 3*std_dev, color="darkgreen", alpha=0.1, label="")

    # Draw horizontal lines from the lowest and highest points
    min_price = np.min(y)
    max_price = np.max(y)
    ax.axhline(y=min_price, color="green", linestyle="--", label="")
    ax.axhline(y=max_price, color="red", linestyle="--", label="")

    # Add price labels for the highest and lowest prices
    ax.text(X[-1], min_price, f'Low: {min_price:.2f}', color='green', verticalalignment='top')
    ax.text(X[-1], max_price, f'High: {max_price:.2f}', color='red', verticalalignment='bottom')

    # Draw exponential moving averages with dashed lines
    ax.plot(X, data_recent['EMA_9'], color="blue", linestyle="-", label="EMA 9/20_blue")
    ax.plot(X, data_recent['EMA_20'], color="navy", linestyle="-", label="")

    # Add arrows for EMA crossovers
    for i in range(1, len(data_recent)):
        if data_recent['EMA_9'].iloc[i] > data_recent['EMA_20'].iloc[i] and data_recent['EMA_9'].iloc[i-1] <= data_recent['EMA_20'].iloc[i-1]:
            ax.plot(X[i], data_recent['Close'].iloc[i], '^', markersize=5, color='blue', lw=0)
        elif data_recent['EMA_9'].iloc[i] < data_recent['EMA_20'].iloc[i] and data_recent['EMA_9'].iloc[i-1] >= data_recent['EMA_20'].iloc[i-1]:
            ax.plot(X[i], data_recent['Close'].iloc[i], 'v', markersize=5, color='red', lw=0)

    # Add trend message on top of the plot
    ax.text(0.5, 0.9, trend_message, transform=ax.transAxes, fontsize=12, color=trend_color, ha='center')

    ax.set_xlabel("Time (Minutes)")
    ax.set_ylabel(f"{ticker} Price")
    ax.set_title(f"Combined Linear and Polynomial Regression for {ticker} (Most Recent 300 Points)")
    ax.legend()
    st.pyplot(fig)

    # Display the minute-level data table at the bottom (most recent 300 points)
    #st.write(f"### {ticker} Minute-Level Data (Most Recent 300 Points)")
    #st.write(data_recent)

if __name__ == "__main__":
    main()
