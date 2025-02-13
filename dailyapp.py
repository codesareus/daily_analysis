import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz
from gtts import gTTS
import os
import time

#midwest = pytz.timezone("America/New")
midwest = pytz.timezone("US/Eastern")

interval = "1m"

# Function to calculate RSI
def calculate_rsi(data, window1=14, window2=25):
    # Calculate RSI with the first window (default 14)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window1).mean()
    
    # Avoid division by zero by adding a small epsilon to loss
    rs1 = gain / (loss + 1e-10)
    rsi1 = 100 - (100 / (1 + rs1))
    
    # Add RSI1 to the DataFrame
    data['RSI'] = rsi1
    
    # Calculate RSI with the second window (default 25)
    delta2 = data['Close'].diff(1)
    gain2 = (delta2.where(delta2 > 0, 0)).rolling(window=window2).mean()
    loss2 = (-delta2.where(delta2 < 0, 0)).rolling(window=window2).mean()
    
    # Avoid division by zero by adding a small epsilon to loss
    rs2 = gain2 / (loss2 + 1e-10)
    rsi2 = 100 - (100 / (1 + rs2))
    
    # Add RSI2 to the DataFrame
    data['RSI2'] = rsi2
    
    return data

def calculate_macd(data):
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to fetch stock data with a specified interval
def fetch_stock_data(ticker, interval="1m"):
    # Fetch data for the specified stock with the given interval (including premarket)
    stock = yf.Ticker(ticker)
    data = stock.history(period="5d", interval=interval, prepost=True)  # Include premarket data
    return data

# Function to fetch stock data with a specified 1h interval
def fetch_stock_data1mo(ticker, interval="1h"):
    # Fetch data for the specified stock with the given interval (including premarket)
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo", interval="1h", prepost=True)  # no doest not Include premarket data
    return data

# Function to fetch the previous 5 day's close price
def fetch_daily5(ticker):
    stock = yf.Ticker(ticker)
    daily5 = stock.history(period="5d")  # Fetch last 5 days of data
    if len(daily5) >= 2:
        return daily5['Close'] 
    else:
        return None  # Handle cases where there isn't enough data

# Function to fetch 3mo close price
def fetch_3mo(ticker):
    stock = yf.Ticker(ticker)
    daily3mo = stock.history(period="3mo")
    if len(daily3mo) >= 2:
        return daily3mo    
    else:
        return None  # Handle cases where there isn't enough data
    
# Function to fetch 6mo close price
def fetch_6mo(ticker):
    stock = yf.Ticker(ticker)
    daily6mo = stock.history(period="6mo")
    if len(daily6mo) >= 2:
        return daily6mo
    else:
        return None  # Handle cases where there isn't enough data

# Function to fetch the previous day's close price
def fetch_previous_close(ticker):
    close_prices = fetch_daily5(ticker)
    if close_prices is None:
        return None  # Handle cases where there isn't enough data
    
    # Get current time in NY (US Eastern Time)
    midwest = pytz.timezone("US/Eastern")
    now = datetime.now(midwest)

    # Define US market hours
    market_open = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

    if now < market_open or now > market_close:  # Pre-market or post-market
        previous_close = close_prices[-1]  # Use the previous day's close
    else:
        previous_close = close_prices[-2]  # Use the most recent close

    return previous_close


# Function to fetch the day before yesterday's close price
def fetch_d2_close(ticker):
    close_prices = fetch_daily5(ticker)
    if close_prices is None:
        return None  # Handle cases where there isn't enough data
    
    # Get current time in NY (US Eastern Time)
    midwest = pytz.timezone("US/Eastern")
    now = datetime.now(midwest)

    # Define US market hours
    market_open = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

    if now < market_open or now > market_close:  # Pre-market or post-market
        d2_close = close_prices[-2]  # Use the previous day's close
    else:
        d2_close = close_prices[-3]  # Use the most recent close

    return d2_close

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

# Function to calculate exponential moving averages
def calculate_emas(data):
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    return data


# Streamlit app
def main():
    st.title("Ticker Regression Analysis")
    
    # Input box for user to enter stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., SPY, AAPL, TSLA):", value="SPY").upper()

    # Add a button group for interval selection
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        if st.button("1min"):
            interval = "1m"
    with col2:
        if st.button("5min", key="5m"):
            interval = "5m"
    with col3:
        if st.button("15min", key="15m"):
            interval = "15m"

    with col4:
        if st.button("30min", key="30m"):
            interval = "30m"

    with col5:
        if st.button("1hr", key="1h"):
            interval = "1h"        
    with col6:
        if st.button("3mo", key="3mo"):
            interval = "3mo"
    with col7:
        if st.button("6mo", key="6mo"):
            interval = "6mo"
    
    # Default interval
    if 'interval' not in locals():
        interval = "5m"

    # Display current selection
    st.write(f"**Selected Interval:** {interval}")

    # Fetch data for the user-specified stock and interval
    if interval == "1h":
        data = fetch_stock_data1mo(ticker, interval="1h")
    elif interval == "3mo":
        data = fetch_3mo(ticker)
    elif interval == "6mo":
        data = fetch_6mo(ticker)
    else:
        data = fetch_stock_data(ticker, interval=interval)

    if data.empty:
        st.error(f"Failed to fetch data for {ticker}. Please check the ticker and try again.")
        return

    # Adjust the data based on the selected backtrack
    data_recent = data.tail(300)

    # Calculate EMAs
    data_recent = calculate_emas(data_recent)

    # Get the current price (last available price in the data)
    current_price = data_recent['Close'].iloc[-1]

    # Fetch the previous day's close price
    previous_close = fetch_previous_close(ticker)
    if previous_close is None:
        st.error("Failed to fetch the previous day's close price. Please try again.")
        return

    change = current_price - previous_close

    # Calculate percentage change
    percentage_change = calculate_percentage_change(current_price, previous_close)

    # Get current local time
    midwest = pytz.timezone("US/Eastern")
    #current_time = datetime.now(midwest).strftime("%H:%M:%S")
    current_time = datetime.now(midwest).strftime("%I:%M:%S %p")

    # Display the percentage change message with current local time
    #st.write("### Current Price vs Previous Close___" f"{ticker}")
    if percentage_change >= 0:
        st.success(f"🟢 {ticker}:  **{current_price:.2f}**, **{change:.2f}**  (**{percentage_change:.2f}%**, previous_close **{previous_close:.2f}**)  |  **___** {current_time} **___**")
    else:
        st.error(f"🔴 {ticker}:  **{current_price:.2f}**, **{change:.2f}**  (**{percentage_change:.2f}%**, prev_close **{previous_close:.2f}**)  |  **......** {current_time}")
        
    degree = 2  # Default to degree 2

    # Perform linear regression (using only the most recent 300 points)
    X, y, y_pred_linear, r2_linear, data_recent = perform_regression(data_recent, degree=1)

    # Perform polynomial regression with the selected degree
    X, y, y_pred_poly, r2_poly, _ = perform_regression(data_recent, degree=degree)

    # Calculate residuals and standard deviation for the polynomial model
    residuals = y - y_pred_poly
    std_dev = np.std(residuals)

    # Determine the trend message
    if current_price > data_recent['EMA_9'].iloc[-1] and data_recent['EMA_9'].iloc[-1] > data_recent['EMA_20'].iloc[-1]:
        trend_message = f"Trend UP"
        trend_color = "green"
    elif current_price < data_recent['EMA_9'].iloc[-1] and data_recent['EMA_9'].iloc[-1] < data_recent['EMA_20'].iloc[-1]:
        trend_message = f"Trend DOWN"
        trend_color = "red"
    else:
        trend_message = f"Trend NEUTRAL"
        trend_color = "gray"

    # Extract time (hours and minutes) for the x-axis
    time_labels = data_recent.index.strftime('%H:%M')  # Format time as HH:MM

    # Simplify x-axis labels based on the interval

    if interval == "15m":
        # For 15-minute interval, show only every 3 hours (e.g., 09:00, 12:00, 15:00)
        simplified_time_labels = [label if label.endswith('00') and int(label.split(':')[0]) % 3 == 0 else '' for label in time_labels]
    elif interval == "30m":
        # For 30-minute interval, show only every 3 hours (e.g., 09:00, 12:00, 15:00)
        simplified_time_labels = [label if label.endswith('00') and int(label.split(':')[0]) % 3 == 0 else '' for label in time_labels]
    elif interval == "1h":
        simplified_time_labels = [label if label.endswith('00') and int(label.split(':')[0]) % 8 == 0 else '' for label in time_labels]

    #3mo and 6mo data has only day information not hours and minute
    elif interval == "3mo":
        data3mo = fetch_3mo(ticker)
        time_labels = data3mo.index.strftime('%Y-%m-%d')  # Format to YYYY-MM-DD
        simplified_time_labels = [label if idx % 9 == 0 else '' for idx, label in enumerate(time_labels)]

    elif interval == "6mo":
        data6mo = fetch_6mo(ticker)
        time_labels = data6mo.index.strftime('%Y-%m-%d')  # Format to YYYY-MM-DD
        simplified_time_labels = [label if idx % 9 == 0 else '' for idx, label in enumerate(time_labels)]    

    else:
        # For 1-minute and 5-minute intervals, show only hours (e.g., 09:00, 10:00)
        simplified_time_labels = [label if label.endswith('00') else '' for label in time_labels]

    # Calculate the deviation of the current price from the polynomial regression model
    current_price_deviation = current_price - y_pred_poly[-1]  # Deviation from the polynomial model
    deviation_in_std = current_price_deviation / std_dev  # Deviation in terms of standard deviations

    # Add a message above the plot showing the price deviation
    if deviation_in_std >= 1:
        deviation_message = f"{ticker}_Deviation from PR: +{deviation_in_std:.2f} std_dev"
        deviation_color = "red"  # Red for >= +2 std_dev
    elif deviation_in_std <= -1:
        deviation_message = f"{ticker}_Deviation from PR: {deviation_in_std:.2f} std_dev"
        deviation_color = "green"  # Green for <= -2 std_dev
    else:
        deviation_message = f"{ticker}_Deviation from PR: {deviation_in_std:.2f} std_dev"
        deviation_color = "gray"  # Default color for other cases

    # Display the deviation message with the appropriate color
    st.markdown(f"<h3 style='color:{deviation_color};'>{deviation_message}</h3>", unsafe_allow_html=True)

    # Add a message above the plot showing the trend
    st.markdown(f"<h3 style='color:{trend_color};'>{ticker}_{trend_message}</h3>", unsafe_allow_html=True)

    col_1, col_2 = st.columns(2)
    with col_1:
        st.write(f"**Linear_Polynomial Regression Plots ({interval})**")       
    with col_2:
        # Display the current polynomial degree
        st.write(f"**PR_deg:** {degree}")

    # Calculate RSI before plotting
    data_recent = calculate_rsi(data_recent)
    data_recent = calculate_macd(data_recent)

    # Plot both linear and polynomial regression results on the same graph
    # Define a list of timeframes that support MACD
    valid_macd_timeframes = ["1m","5m","15m","30m","1h", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

    # Only plot MACD if the selected timeframe is valid
    if interval in valid_macd_timeframes:
        fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 25), gridspec_kw={'height_ratios': [3, 1, 1]})
    else:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(20, 25), gridspec_kw={'height_ratios': [3, 1]})

    #fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 20), gridspec_kw={'height_ratios': [3, 1, 1]})

    #fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

    #fig, ax = plt.subplots(figsize=(12, 12))

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

    # Draw gray line for current price
    ax.axhline(y=current_price, color="gray", linestyle="--", label="")

    # Modify the current price label to include the trend message and color
    current_price_label = f"-----{current_price:.2f} {trend_message.split()[-1]}"
    if trend_message == "Trend UP":
        current_price_color = "green"  # Green for UP trend
    elif trend_message == "Trend DOWN":
        current_price_color = "red"  # Red for DOWN trend
    else:
        current_price_color = "gray"  # Default color for NEUTRAL trend

    ax.text(x_values[-1], current_price, current_price_label, color=current_price_color, verticalalignment='top')

    # Draw gray line for previous close
    ax.axhline(y=previous_close, color="navy", linestyle="--", label="")

    # Add price label for the previous_price
    ax.text(0, previous_close, f'{previous_close:.2f}__c1', color='navy', verticalalignment='top')

    # add time intervals on bottom of chart
    ax.text(0.3, 0.05, f"Time Frame: {interval}", 
        horizontalalignment='left', verticalalignment='center', 
        transform=ax.transAxes, fontsize=12, color="blue")
    
    # Draw gray line for d2 close
    d2_close = fetch_d2_close(ticker)
    ax.axhline(y=d2_close, color="navy", linestyle="--", label="")

    # Add price label for the d2_close
    ax.text(0, d2_close, f'{d2_close:.2f}__c2', color='navy', verticalalignment='top')

    # Draw exponential moving averages with dashed lines
    ax.plot(x_values, data_recent['EMA_9'], color="orange", linestyle="--", label="EMA 9/20_blue")
    ax.plot(x_values, data_recent['EMA_20'], color="blue", linestyle="--", label="")
    ax.plot(x_values, data_recent['EMA_50'], color="gold", linestyle="--", label="EMA 50")
    ax.plot(x_values, data_recent['EMA_100'], color="gray", linestyle="--", label="EMA 100")
    ax.plot(x_values, data_recent['EMA_200'], color="purple", linestyle="--", label="EMA 200")

    # Add price labels for EMAs
    ax.text(x_values[-1], data_recent['EMA_9'].iloc[-1], f'^^^^^^e9', color='orange', verticalalignment='top')
    ax.text(x_values[-1], data_recent['EMA_20'].iloc[-1], f'^^^^^^^^e20', color='blue', verticalalignment='top')
    ax.text(x_values[-1], data_recent['EMA_50'].iloc[-1], f'^^^^^^^^e50', color='gold', verticalalignment='top')
    ax.text(x_values[-1], data_recent['EMA_100'].iloc[-1], f'^^^^^^^^e100', color='gray', verticalalignment='top')
    ax.text(x_values[-1], data_recent['EMA_200'].iloc[-1], f'^^^^^^^^e200', color='purple', verticalalignment='top')

    # Add arrows for EMA crossovers
    for i in range(1, len(data_recent)):
        if data_recent['EMA_9'].iloc[i] > data_recent['EMA_20'].iloc[i] and data_recent['EMA_9'].iloc[i-1] <= data_recent['EMA_20'].iloc[i-1]:
            ax.plot(x_values[i], data_recent['Close'].iloc[i], '^', markersize=5, color='blue', lw=0)
        elif data_recent['EMA_9'].iloc[i] < data_recent['EMA_20'].iloc[i] and data_recent['EMA_9'].iloc[i-1] >= data_recent['EMA_20'].iloc[i-1]:
            ax.plot(x_values[i], data_recent['Close'].iloc[i], 'v', markersize=5, color='red', lw=0)

    # Format x-axis to show only hours (or every 3 hours for 30-minute interval)
    ax.set_xticks(x_values)  # Set ticks for all time points
    ax.set_xticklabels(simplified_time_labels)  # Show only hours or every 3 hours
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel(f"{ticker} Price")
    ax.set_title(f"Combined Linear and Polynomial Regression for {ticker} (Most Recent 300 Points)")
    ax.legend()

    # --- RSI Plot ---
    ax2.plot(x_values, data_recent['RSI'], color="gray", label="RSI (14)")
    ax2.plot(x_values, data_recent['RSI2'], color="red", linestyle="--", label="RSI (25)")
    ax2.axhline(y=70, color="red", linestyle="--")
    ax2.axhline(y=30, color="green", linestyle="--")
    ax2.axhline(y=50, color="gray", linestyle="--")
    ax2.set_title("Relative Strength Index (RSI)")
    ax2.legend()

    # === MACD Plot (Only If Timeframe Is Valid) ===
    if interval in valid_macd_timeframes:
        # Create a sequence for the x-axis from 1 to len(data_recent)
        x_values = range(1, len(data_recent) + 1)

        # Plot the MACD and Signal lines with numeric x-values
        ax3.plot(x_values, data_recent['MACD'], color="blue", label="MACD Line")
        ax3.plot(x_values, data_recent['Signal_Line'], color="red", linestyle="--", label="Signal Line")

        # Histogram Bars (Green for Positive, Red for Negative)
        histogram_values = data_recent['MACD'] - data_recent['Signal_Line']
        ax3.bar(x_values, histogram_values, color=['green' if val > 0 else 'red' for val in histogram_values], alpha=0.5)

        ax3.set_title("MACD (Moving Average Convergence Divergence)")
        ax3.legend()

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)

   # Calculate EMAs, RSI, MACD
    data_recent = calculate_emas(data_recent)
    data_recent = calculate_rsi(data_recent)
    data_recent = calculate_macd(data_recent)

    # Get the latest EMA values and current price
    ema_values = {
        "Current Price": data_recent['Close'].iloc[-1],
        "EMA 9": data_recent['EMA_9'].iloc[-1],
        "EMA 20": data_recent['EMA_20'].iloc[-1],
        "EMA 50": data_recent['EMA_50'].iloc[-1],
        "EMA 100": data_recent['EMA_100'].iloc[-1],
        "EMA 200": data_recent['EMA_200'].iloc[-1]
    }

    #Get the latest RSI values 
    RSI_values = {
        "RSI": data_recent['RSI'].iloc[-1],
        "RSI2": data_recent['RSI2'].iloc[-1],
    }

    #Get the latest MACD values 
    MACD_values = {
        "MACD": data_recent['MACD'].iloc[-1],
        "Signal_Line": data_recent['Signal_Line'].iloc[-1],
    }

    ########## values used
    price = data_recent['Close'].iloc[-1]
    ema9= data_recent['EMA_9'].iloc[-1]
    ema20= data_recent['EMA_20'].iloc[-1]
    ema50= data_recent['EMA_50'].iloc[-1]
    ema100= data_recent['EMA_100'].iloc[-1]
    ema200= data_recent['EMA_200'].iloc[-1]
    rsi = data_recent['RSI'].iloc[-1]
    rsi2 = data_recent['RSI2'].iloc[-1]
    macd = data_recent['MACD'].iloc[-1]
    signal = data_recent['Signal_Line'].iloc[-1]

    # trend scores
    emaT_score = 0
    rsiT_score = 0
    macdT_score = 0
    
    col_1, col_2, col_3= st.columns(3)
    with col_1:

        # Create DataFrame and sort by value in descending order
        ema_df = pd.DataFrame(list(ema_values.items()), columns=["Indicator", "Value"])
        ema_df = ema_df.sort_values(by="Value", ascending=False)

        # Reset index and drop the numbers column
        ema_df = ema_df.reset_index(drop=True)

        ## message
        message = " "
        color1 = " "
        
        if price > ema9 and ema9 > ema20:
            message = "Up"
            color1 = "green"
            emaT_score = emaT_score + 1
        elif price < ema9 and ema9 < ema20:
            message = "Down"
            color1 = "red"
            emaT_score = emaT_score - 1
        else:
            message = "Neutral"
            color1 = "gray"
    
        # Display the table
        st.markdown(f"### <span style='color:{color1};'>EMA: {message}</span>", unsafe_allow_html=True)
        st.dataframe(ema_df, hide_index=True)

    with col_2:

        # Create DataFrame and sort by value in descending order
        rsi_df = pd.DataFrame(list(RSI_values.items()), columns=["Indicator", "Value"])
        rsi_df = rsi_df.sort_values(by="Value", ascending=False)

        # Reset index and drop the numbers column
        rsi_df = rsi_df.reset_index(drop=True)

        ## message
        message = " "
        color2 = " "
        if rsi > rsi2 and rsi > 50:
            message = "Up "
            color2 = "green"
            rsiT_score = rsiT_score + 1
        elif rsi < rsi2 and rsi < 50:
            message = "Down"
            color2 = "red"
            rsiT_score = rsiT_score - 1
        else:
            message = "Neutral "
            color2 = "gray"
    
        # Display the table
        st.markdown(f"### <span style='color:{color2};'>RSI: {message}</span>", unsafe_allow_html=True)
        st.dataframe(rsi_df, hide_index=True)
    

    with col_3:
        # Create DataFrame and sort by value in descending order
        macd_df = pd.DataFrame(list(MACD_values.items()), columns=["Indicator", "Value"])
        macd_df = macd_df.sort_values(by="Value", ascending=False)

        # Reset index and drop the numbers column
        macd_df = macd_df.reset_index(drop=True)

        ## message
        message = " "
        color3 = " "
        if macd > signal and macd > 0:
            message = "Up "
            color3 = "green"
            macdT_score = macdT_score + 1
        elif macd < signal and macd < 0:
            message = "Down"
            color3 = "red"
            macdT_score = macdT_score - 1
        else:
            message = "Neutral "
            color3 = "gray"
            
        # Display the table
        st.markdown(f"### <span style='color:{color3};'>MACD: {message}</span>", unsafe_allow_html=True)
        st.dataframe(macd_df, hide_index=True)

    #### calculate scores
    ema_score = (price > ema9)*0.2 + (ema9 > ema20)*0.4 + (ema20 > ema50)*0.6 + (ema50 > ema100)*0.8 +  (ema100 > ema200) - (ema200 > ema100) - (ema100 > ema50)*0.8 - (ema50 > ema20)*0.6 - (ema20 > ema9)*0.4 - (ema9 > price)*0.2

    rsi_score = 0
    if (rsi > rsi2) and (rsi > 50):
        rsi_score = 2
    elif (rsi < rsi2) and (rsi > 50):
        rsi_score = 1
    elif (rsi < rsi2) and (rsi < 50):
        rsi_score = - 2
    elif (rsi > rsi2) and (rsi < 50):
        rsi_score = -1
    else:
        rsi_score = 0
        
    macd_score = 0
    if (macd > signal) and (macd > 0):
        macd_score = 2
    elif (macd < signal) and (macd > 0):
        macd_score = 1
    elif (macd < signal) and (macd < 0):
        macd_score = - 2
    elif (macd > signal) and (macd < 0):
        macd_score = - 1
    else:
        macd_score = 0

    prm_score = (y_pred_poly[-1] > y_pred_poly[-2])*1 - (y_pred_poly[-1] < y_pred_poly[-2])*1
    std_score = - deviation_in_std
    
    score = ema_score + rsi_score + macd_score + prm_score + std_score

    

    ############## File to store historical scores


    # Define refresh interval (in seconds)
    REFRESH_INTERVAL = 60  

    # Define file name based on interval
    score_file = f"score_history_{interval}.csv"
    scoreT_file = f"scoreT.csv"

    # Ensure the file exists with proper headers
    if not os.path.exists(score_file):
        pd.DataFrame(columns=["Timestamp", "EMAs", "RSI", "MACD", "PRM", "std", "total"]).to_csv(score_file, index=False)

    # Function to update and save data
    def update_scores():
        timestamp = datetime.now(midwest).strftime("%Y-%m-%d %H:%M:%S")
        
        new_data = pd.DataFrame([{
            "Timestamp": timestamp,
            "EMAs": round(ema_score, 2),
            "RSI": round(rsi_score, 2),
            "MACD": round(macd_score, 2),
            "PRM": round(prm_score, 2),
            "std": round(std_score, 2),
            "total": round(score, 2)
        }])
        
        # Append to CSV file
        new_data.to_csv(score_file, mode="a", header=False, index=False)
        
        # Display latest score
        st.write(f"### Total Score: {score: .2f} || Interval: {interval} || Time: {datetime.now(midwest).strftime('%H:%M:%S')}")
        st.dataframe(new_data, hide_index=True)

        # Add a download button for the CSV file
        with open(score_file, "rb") as file:
            btn = st.download_button(
                label="Download data",
                data=file,
                file_name=score_file,
                mime="text/csv"
            )
            
    # Function to update and save scoreT(trend based on 3 color)
    def update_scoreT():

        ### interval is from ["1m","5m","15m","30m","1h", "1mo", "3mo", "6mo"]
        
        new_data = pd.DataFrame([{
            "tFrame": f"{interval}",
            "ema": round(emaT_score, 2),
            "rsi": round(rsiT_score, 2),
            "macd": round(macdT_score, 2),
        }])
        
        # Append to CSV file
        new_data.to_csv(scoreT_file, mode="a", header=False, index=False)
        
        # Display latest score
        st.write(f"### emaT: {emaT_score: .2f} tFrame: {interval}")
        st.dataframe(new_data, hide_index=True)

        ### do bar graph using scoreT_file
        # Load data from the CSV file
        # Load data from the CSV file
        try:
            df = pd.read_csv(scoreT_file, names=["tFrame", "ema", "rsi", "macd"])
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

        # Get unique intervals and prepare x-axis locations
        unique_intervals = df["tFrame"].unique()
        x = np.arange(len(unique_intervals))  # X locations for groups
        width = 0.25  # Width of each bar

        # Prepare values for each metric
        ema_values = [df[df["tFrame"] == interval]["ema"].mean() for interval in unique_intervals]
        rsi_values = [df[df["tFrame"] == interval]["rsi"].mean() for interval in unique_intervals]
        macd_values = [df[df["tFrame"] == interval]["macd"].mean() for interval in unique_intervals]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Function to determine color based on value (green for positive, red for negative)
        def get_color(value):
            return "green" if value >= 0 else "red"

        # Plot EMA bars
        for i, value in enumerate(ema_values):
            ax.bar(x[i] - width, value, width, color=get_color(value), hatch="//", edgecolor="black")

        # Plot RSI bars
        for i, value in enumerate(rsi_values):
            ax.bar(x[i], value, width, color=get_color(value), hatch="xx", edgecolor="black")

        # Plot MACD bars
        for i, value in enumerate(macd_values):
            ax.bar(x[i] + width, value, width, color=get_color(value), hatch="..", edgecolor="black")

        # Add labels and title
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("Score")
        ax.set_title("Trend Scores by Interval")
        ax.set_xticks(x)
        ax.set_xticklabels(unique_intervals, rotation=45)
        ax.legend(["EMA (//)", "RSI (xx)", "MACD (..)"], loc="upper left")

        # Display the chart
        st.pyplot(fig)


    # Function to perform regression analysis and plot
    def regression_analysis():
        historical_data = pd.read_csv(score_file)

        # Convert Timestamp to numerical values for regression
        historical_data["Timestamp"] = pd.to_datetime(historical_data["Timestamp"])
        historical_data["TimeIndex"] = (historical_data["Timestamp"] - historical_data["Timestamp"].min()).dt.total_seconds()/600
        historical_data["Hour"] = historical_data["Timestamp"].dt.strftime("%H:%M")  # Format as HH:MM

        # Perform Polynomial Regression (degree=2)
        X = historical_data[["TimeIndex"]].values
        y = historical_data["total"].values
        y_ema = historical_data["EMAs"].values # define Y for ema
        y_rsi = historical_data["RSI"].values # define Y for std

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        # Regression for "total"
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        y_pred_poly = poly_model.predict(X_poly)
        r2_poly = r2_score(y, y_pred_poly)

        # Regression for "EMA"
        model_ema = LinearRegression()
        model_ema.fit(X_poly, y_ema)
        y_ema_pred = model_ema.predict(X_poly)
        r2_ema = r2_score(y, y_ema_pred)

        # Regression for "rsi"
        model_rsi = LinearRegression()
        model_rsi.fit(X_poly, y_rsi)
        y_rsi_pred = model_rsi.predict(X_poly)
        r2_rsi = r2_score(y, y_rsi_pred)

        # Perform Linear Regression
        lin_model = LinearRegression()
        lin_model.fit(X, y)
        y_pred_lin = lin_model.predict(X)
        r2_lin = r2_score(y, y_pred_lin)
        
        # Plot the actual total values
        plt.figure(figsize=(10, 5))
        plt.scatter(historical_data["TimeIndex"], historical_data["total"], color="blue", label="Actual total")

        # Plot Linear Regression Line
        plt.xticks(ticks=historical_data["TimeIndex"], labels=historical_data["Hour"], rotation=45)
        plt.plot(historical_data["TimeIndex"], y_pred_lin,  color="gray", linestyle="solid", label=f"Linear ( R² = {r2_lin:.2f})")

        # Plot "tota" Polynomial Regression Line
        plt.xticks(ticks=historical_data["TimeIndex"], labels=historical_data["Hour"], rotation=45)
        plt.plot(historical_data["TimeIndex"], y_pred_poly,  color="blue", linestyle="dashed", label=f"total ( R² = {r2_poly:.2f})")

        #plot ema polynomial
        plt.scatter(X, y_ema, label="EMA (Actual)", marker="x", color="red")
        plt.plot(X, y_ema_pred, linestyle="--", color="red", label=f"EMA ( R² = {r2_ema:.2f})")

        #plot rsi polynomial
        plt.scatter(X, y_rsi, label="RSI (Actual)", marker="^", color="orange")
        plt.plot(X, y_rsi_pred, linestyle="--", color="orange", label=f"RSI ( R² = {r2_rsi:.2f})")

        
        # Labels and legend
        plt.xlabel("Time")
        plt.ylabel("Total Score")
        plt.legend()
        plt.title("Total vs. Time with Polynomial & Linear Regression")

        # Show plot in Streamlit
        st.pyplot(plt)

    ###### Call functions to update data and plot

    update_scores()
    regression_analysis()
    update_scoreT()

    # Display latest score
    #st.write(f"### Total Score: {score: .2f} || Interval: {interval} || Time: {datetime.now(midwest).strftime('%H:%M:%S')}")

    # Refresh app every minute
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

      
    
    


if __name__ == "__main__":
    main()


