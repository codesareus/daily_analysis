import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz

# Function to fetch stock data with a specified interval
def fetch_stock_data(ticker, interval="1m"):
    # Fetch data for the specified stock with the given interval (including premarket)
    stock = yf.Ticker(ticker)
    data = stock.history(period="5d", interval=interval, prepost=True)  # Include premarket data
    return data


###########
# Function to fetch the previous 5 day's close price
def fetch_daily5(ticker):
    stock = yf.Ticker(ticker)
    daily5 = stock.history(period="5d")  # Fetch last 5 days of data
    if len(daily5) >= 2:
        return daily5['Close'] 
    else:
        return None  # Handle cases where there isn't enough data

###########

def fetch_previous_close(ticker):
    close_prices = fetch_daily5(ticker)
    if close_prices is None:
        return None  # Handle cases where there isn't enough data
    
    # Get current time in NY (US Eastern Time)
    midwest = pytz.timezone("America/chicago")
    now = datetime.now(midwest)

    # Define US market hours
    market_open = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

    if now < market_open or now > market_close:  # Pre-market or post-market
        previous_close = close_prices[-1]  # Use the previous day's close
    else:
        previous_close = close_prices[-2]  # Use the most recent close

    return previous_close

def fetch_d2_close(ticker):
    close_prices = fetch_daily5(ticker)
    if close_prices is None:
        return None  # Handle cases where there isn't enough data
    
    # Get current time in NY (US Eastern Time)
    midwest = pytz.timezone("America/chicago")
    now = datetime.now(midwest)

    # Define US market hours
    market_open = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

    if now < market_open or now > market_close:  # Pre-market or post-market
        d2_close = close_prices[-2]  # Use the previous day's close
    else:
        d2_close = close_prices[-3]  # Use the most recent close

    return d2_close

##############

# Function to fetch the previous day's close price
#def fetch_previous_close(ticker):
 #   stock = yf.Ticker(ticker)
 #   previous_day_data = stock.history(period="5d")  # Fetch last 5 days of data
 #   if len(previous_day_data) >= 2:
 #       return previous_day_data['Close'].iloc[-2]  # Second-to-last close is the previous day's close
 #   else:
  #      return None  # Handle cases where there isn't enough data

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
    st.write("This app fetches stock prices at different intervals (including premarket data) and performs linear and polynomial regression analysis.")

    # Input box for user to enter stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., SPY, AAPL, TSLA):", value="SPY").upper()

    # Dropdown for interval selection
    interval = st.selectbox("Select Interval", ["1m", "5m", "30m"], index=1)

    # Add a button to refresh data
    if st.button("Refresh Data"):
        st.cache_data.clear()  # Clear cached data to force a fresh fetch

    # Fetch data for the user-specified stock and interval
    data = fetch_stock_data(ticker, interval=interval)
    if data.empty:
        st.error(f"Failed to fetch data for {ticker}. Please check the ticker and try again.")
        return

    # Get the current price (last available price in the data)
    current_price = data['Close'].iloc[-1]

    ##### fetch daily5
    daily5 = fetch_daily5(ticker)
    print("spy:::::",  daily5)
    
    # Fetch the previous day's close price
    previous_close = fetch_previous_close(ticker)
    if previous_close is None:
        st.error("Failed to fetch the previous day's close price. Please try again.")
        return

    change = current_price - previous_close
    
    # Calculate percentage change
    percentage_change = calculate_percentage_change(current_price, previous_close)

    # Display the percentage change message
    st.write("### Current Price vs Previous Close___" f"{ticker}")
    if percentage_change >= 0:
        st.success(f"🟢 {ticker}:  **{current_price:.2f}**, **{change:.2f}**  (**{percentage_change:.2f}%**, previous_close **{previous_close:.2f}**)")
    else:
        st.error(f"🔴 {ticker}:  **{current_price:.2f}**, **{change:.2f}**  (**{percentage_change:.2f}%**, prev_close **{previous_close:.2f}**)")
 
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

    # Extract time (hours and minutes) for the x-axis
    time_labels = data_recent.index.strftime('%H:%M')  # Format time as HH:MM

    # Simplify x-axis labels based on the interval
    if interval == "30m":
        # For 30-minute interval, show only every 3 hours (e.g., 09:00, 12:00, 15:00)
        simplified_time_labels = [label if label.endswith('00') and int(label.split(':')[0]) % 3 == 0 else '' for label in time_labels]
    else:
        # For 1-minute and 5-minute intervals, show only hours (e.g., 09:00, 10:00)
        simplified_time_labels = [label if label.endswith('00') else '' for label in time_labels]

    # Plot both linear and polynomial regression results on the same graph
    st.write("### Combined Regression Plot (Most Recent 300 Points)")
    fig, ax = plt.subplots()

    # Use numeric x-axis for plotting to avoid duplicate time issues
    x_values = np.arange(len(data_recent))  # Numeric x-axis

    # Plot actual prices and regression lines
    ax.plot(x_values, y, color="gray", label="Actual Prices")  # Actual prices as a gray line plot
    ax.plot(x_values, y_pred_linear, color="red", label=f"L.R. (R² = {r2_linear:.2f})")
    ax.plot(x_values, y_pred_poly, color="green", label=f"P.R. (d {degree}, R² = {r2_poly:.2f})")

    # Draw bands for 1, 2, and 3 standard deviations from the polynomial model
    ax.fill_between(x_values, y_pred_poly - std_dev, y_pred_poly + std_dev, color="blue", alpha=0.3, label="")
    ax.fill_between(x_values, y_pred_poly - 2*std_dev, y_pred_poly + 2*std_dev, color="green", alpha=0.2, label="")
    ax.fill_between(x_values, y_pred_poly - 3*std_dev, y_pred_poly + 3*std_dev, color="red", alpha=0.1, label="")

    # Draw horizontal lines from the lowest and highest points
    min_price = np.min(y)
    max_price = np.max(y)
    ax.axhline(y=min_price, color="green", linestyle="--", label="")
    ax.axhline(y=max_price, color="red", linestyle="--", label="")

    # Add price labels for the highest and lowest prices
    ax.text(x_values[-1], min_price, f'Low: {min_price:.2f}', color='green', verticalalignment='top')
    ax.text(x_values[-1], max_price, f'High: {max_price:.2f}', color='red', verticalalignment='bottom')

    ######## draw gray line for current price
    ax.axhline(y=current_price, color="gray", linestyle="--", label="")
    
    # Add price label for the current_price
    ax.text(x_values[-1], current_price, f'{current_price:.2f}', color='gray', verticalalignment='top')
    
    ## prevoius close
    ax.axhline(y=previous_close, color="navy", linestyle="--", label="")
    
    # Add price label for the previous_price
    ax.text(0, previous_close, f'{previous_close:.2f}__c1', color='navy', verticalalignment='top')

    ## d2 close
    d2_close = fetch_d2_close(ticker)
    ax.axhline(y=d2_close, color="navy", linestyle="--", label="")
    
    # Add price label for the d2_close
    ax.text(0, d2_close, f'{d2_close:.2f}__c2', color='navy', verticalalignment='top')

    ##########

    # Draw exponential moving averages with dashed lines
    ax.plot(x_values, data_recent['EMA_9'], color="blue", linestyle="--", label="EMA 9/20_orange")
    ax.plot(x_values, data_recent['EMA_20'], color="orange", linestyle="--", label="")
    ax.axhline(y=data_recent['EMA_9'].iloc[-1], color="blue", linestyle="-", label="")
    ax.axhline(y=data_recent['EMA_20'].iloc[-1], color="orange", linestyle="-", label="")
    
    # Add price label for emas
    ax.text(x_values[-1], data_recent['EMA_9'].iloc[-1], f'^^^^^^e9', color='blue', verticalalignment='top')
    ax.text(x_values[-1], data_recent['EMA_20'].iloc[-1], f'^^^^^^^^e20', color='orange', verticalalignment='top')

    # Add arrows for EMA crossovers
    for i in range(1, len(data_recent)):
        if data_recent['EMA_9'].iloc[i] > data_recent['EMA_20'].iloc[i] and data_recent['EMA_9'].iloc[i-1] <= data_recent['EMA_20'].iloc[i-1]:
            ax.plot(x_values[i], data_recent['Close'].iloc[i], '^', markersize=5, color='blue', lw=0)
        elif data_recent['EMA_9'].iloc[i] < data_recent['EMA_20'].iloc[i] and data_recent['EMA_9'].iloc[i-1] >= data_recent['EMA_20'].iloc[i-1]:
            ax.plot(x_values[i], data_recent['Close'].iloc[i], 'v', markersize=5, color='red', lw=0)

    # Add trend message on top of the plot
    ax.text(0.5, 0.9, trend_message, transform=ax.transAxes, fontsize=12, color=trend_color, ha='center')

    # Format x-axis to show only hours (or every 3 hours for 30-minute interval)
    ax.set_xticks(x_values)  # Set ticks for all time points
    ax.set_xticklabels(simplified_time_labels)  # Show only hours or every 3 hours
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel(f"{ticker} Price")
    ax.set_title(f"Combined Linear and Polynomial Regression for {ticker} (Most Recent 300 Points)")
    ax.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)

if __name__ == "__main__":
    main()
