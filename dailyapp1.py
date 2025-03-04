### daily analysis 02-28-25

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
from pytz import timezone
import pytz
from gtts import gTTS
import os
import time
from datetime import datetime, time
from time import sleep
from matplotlib.lines import Line2D
import pandas_market_calendars as mcal
from scipy.signal import find_peaks

#eastern = pytz.timezone("America/New")
eastern = pytz.timezone("US/Eastern")

def on_click(event):
    # event.ydata is the y-coordinate in data units where the click occurred.
    y_value = event.ydata
    if y_value is not None:
        # Add a horizontal line at the clicked y-coordinate.
        plt.axhline(y=y_value, color='red', linestyle='--', linewidth=1)
        # Redraw the plot to show the line immediately.
        plt.draw()

def get_time_now():
    eastern = timezone('US/Eastern')
    now = datetime.now(eastern)
    now_time = now.time()
    
    # Get market calendar for NYSE
    nyse = mcal.get_calendar("NYSE")
    
    # Check if today is a market holiday
    today = now.date()
    holidays = nyse.holidays().holidays
    if today in holidays:
        return "holiday"

    # Pre-market (4:00 AM - 9:30 AM)
    if datetime.strptime("04:00", "%H:%M").time() <= now_time < datetime.strptime("09:30", "%H:%M").time():
        return "pre"
    
    # Open wind (9:25 AM - 9:35 AM)
    if datetime.strptime("09:25", "%H:%M").time() <= now_time < datetime.strptime("09:35", "%H:%M").time():
        return "open_wind"
    
    # Regular market hours (9:35 AM - 3:55 PM)
    if datetime.strptime("09:35", "%H:%M").time() <= now_time < datetime.strptime("15:55", "%H:%M").time():
        return "open"
    
    # Close wind-down (3:55 PM - 4:00 PM)
    if datetime.strptime("15:55", "%H:%M").time() <= now_time < datetime.strptime("16:00", "%H:%M").time():
        return "close_wind"
    
    # After-hours (4:00 PM - 8:00 PM)
    if datetime.strptime("16:00", "%H:%M").time() <= now_time < datetime.strptime("20:00", "%H:%M").time():
        return "after_hours"
    
    # Market closed
    return "closed"

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
def fetch_stock_data(ticker, interval="5m"):
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
    daily5 = stock.history(period="3mo")  # Fetch 30 days of data
    if len(daily5) >= 2:
        return daily5['Close'][-5:]  ## return only last 5 days 
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
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    # Define US market hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

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
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    # Define US market hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

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


# Function to perform regression analysis and plot
def regression_analysis(data_recent, interval):

    # Ensure index is datetime
     
    data_recent.index = pd.to_datetime(data_recent.index)
    # Instead of using time in seconds, use a simple range index
    data_recent["TimeIndex"] = np.arange(len(data_recent))
    
    # Create TimeIndex as seconds from the first timestamp
    #data_recent["TimeIndex"] = (data_recent.index - data_recent.index.min()).total_seconds()

    # Create Timestamp column from index
    data_recent["Timestamp"] = data_recent.index  # Use index instead of non-existing 'Timestamp' column

    # Extract the Hour and Minute
    data_recent["Hour"] = data_recent["Timestamp"].dt.strftime("%H:%M")  # Format as HH:MM

    X = data_recent[["TimeIndex"]].values  # Reshape needed for sklearn
    y = data_recent["score"].fillna(0).values

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_pred_lin = lin_model.predict(X)  # Ensure this line is present
    r2_lin = r2_score(y, y_pred_lin)

    # Plot the actual total values
    plt.figure(figsize=(10, 5))
    ##assign color 
    color = "blue"

    plt.scatter(data_recent["TimeIndex"], data_recent["score"], color=color, label="Actual score")

    # Select every 10th label
    xticks = data_recent["TimeIndex"][::10]
    xlabels = data_recent["Hour"][::10]

    # Set x-ticks and labels
    plt.xticks(ticks=xticks, labels=xlabels, rotation=45)

    # Plot Linear Regression Line
    plt.plot(data_recent["TimeIndex"], y_pred_lin, color="gray", linestyle="solid", label=f"Linear ( R² = {r2_lin:.2f})")

    # Plot Polynomial Regression Line
    plt.plot(data_recent["TimeIndex"], y_pred_poly, color="blue", linestyle="dashed", label=f"Polynomial ( R² = {r2_poly:.2f})")

    # Add a horizontal line at set y_value
    current = y[-1]
    
    plt.axhline(y=current, color="gray", linestyle="--", label=f"current: {current: .2f}")
    plt.axhline(y=4, color="red", linestyle="--")
    plt.axhline(y=-4, color="green", linestyle="--")
    plt.axhline(y=0, color="gray", linestyle="-", linewidth = 3)
    
    
    # Labels and legend
    plt.xlabel("Time")
    plt.ylabel("Total Score")
    plt.legend()
    plt.title(f"T_score Regression ({interval})  || {datetime.now().strftime('%D:%H:%M')})")

    # Show plot in Streamlit
    st.pyplot(plt)

        
# Streamlit app
def main():
    st.title("Score Regression Analysis")

    #######################
        
    # Input box for user to enter stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., SPY, AAPL, TSLA):", value="SPY").upper()

    # Initialize session state for index
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'rerun_count' not in st.session_state:
        st.session_state.rerun_count = 0
        
    # Initialize rerun state
    if "stop_sleep" not in st.session_state:
        st.session_state.stop_sleep = 0

    # Initialize temp_price state
    if "temp_price" not in st.session_state:
        st.session_state.temp_price = 0

    # Initialize sb_status state
    if "sb_status" not in st.session_state:
        st.session_state.sb_status = 0

    if "sleepGap" not in st.session_state:
        st.session_state.sleepGap = 5

    # Initialize sbOK state
    #if "sbOK" not in st.session_state:
        #st.session_state.sbOK = 1

    # Store whether the music has been played
    #if 'music_played' not in st.session_state:
    #    st.session_state.music_played = False


    # Define file names
    
    scoreT_file = f"scoreT.csv"
    pe_file = f"pe.csv"

    # List of intervals
    intervals = ['1m', '5m', '15m', '30m', '1h', '3mo', '6mo']

    # Get the current interval
    interval = intervals[st.session_state.index]

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
            st.session_state.stop_sleep == 1
            
    with col6:
        if st.button("3mo", key="3mo"):
            interval = "3mo"
            
    with col7:
        if st.button("6mo", key="6mo"):
            interval = "6mo"
            
    
############################

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
    #############

# Add a slider for backtracking
    backtrack_options = [0, 2, 5, 7, 10, 20, 30, 45, 60, 90, 100, 120]
    selected_backtrack = st.slider(
        "Select number of points to backtrack:",
        min_value=min(backtrack_options),
        max_value=max(backtrack_options),
        value=0,  # Default value
        step=1,  # Step size
        key="backtrack_slider"
    )

    # Adjust the data based on the selected backtrack
    data_recent = data.tail(300 + selected_backtrack)  # Get the most recent 300 + selected_backtrack data points
    data_recent = data_recent.head(300)  # Use only the first 300 points after backtracking

    # Calculate EMAs
    data_recent = calculate_emas(data_recent)


    #######################

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
    eastern = pytz.timezone("US/Eastern")
    #current_time = datetime.now(eastern).strftime("%H:%M:%S")
    current_time = datetime.now(eastern).strftime("%I:%M:%S %p")

    # Display the percentage change message with current local time
    #st.write("### Current Price vs Previous Close___" f"{ticker}")
    if percentage_change >= 0:
        st.success(f"🟢 {ticker}:  **{current_price:.2f}**, **{change:.2f}**  (**{percentage_change:.2f}%**, previous_close **{previous_close:.2f}**)  |  **___** {current_time} **___**")
    else:
        st.error(f"🔴 {ticker}:  **{current_price:.2f}**, **{change:.2f}**  (**{percentage_change:.2f}%**, prev_close **{previous_close:.2f}**)  |  **......** {current_time}")

    ##############################
    if "poly_degree" not in st.session_state:
        st.session_state.poly_degree = 4
    degree = st.session_state.poly_degree
    
    col1, col2,col3,col4 = st.columns(4)
    
    with col1:
        if st.button("degree 3"):
            st.session_state.poly_degree = 3
            st.rerun()

    with col2:
        if st.button("degree 5"):
            st.session_state.poly_degree = 5
            st.rerun()

    with col3:
        if st.button("degree 6"):
            st.session_state.poly_degree = 6
            st.rerun()

    with col4:
        if st.button("degree 7"):
            st.session_state.poly_degree = 7
            st.rerun()

    st.write(f"selected PR degree: {degree}")
    
    ############$$##################
    # Perform linear regression (using only the most recent 300 points)
    X, y, y_pred_linear, r2_linear, data_recent = perform_regression(data_recent, degree=1)

    # Perform polynomial regression with the selected degree
    X, y, y_pred_poly, r2_poly, _ = perform_regression(data_recent, degree=degree)

    # Calculate residuals and standard deviation for the polynomial model
    #residuals = y - y_pred_poly
    #dev_from_std = np.std(residuals)

############ add std_dev and diviation from std_dev
    
    # Calculate residuals for each row
    data_recent["residuals"] = y - y_pred_poly
    # Option 1: Rolling standard deviation (e.g., over 50 time points)
    data_recent["std_dev"] = data_recent["residuals"].rolling(window=30).std()## first 50 time points no data

    std_dev = data_recent["std_dev"].iloc[-1]

####################

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

################## add price deviation score to df

###################

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
    st.markdown(f"<h3 style='color:{deviation_color};'>{deviation_message} ({interval})</h3>", unsafe_allow_html=True)

    # Add a message above the plot showing the trend
    st.markdown(f"<h3 style='color:{trend_color};'>{ticker}_{trend_message} ({interval})</h3>", unsafe_allow_html=True)
        
    # Calculate RSI before plotting
    data_recent = calculate_rsi(data_recent)
    data_recent = calculate_macd(data_recent)

#####################

    ## add p.r. model y value
    data_recent['y_pred_poly'] = y_pred_poly
    
    def calculate_scores(row):
    
        score = 0
        
        price = row['Close']
        ema9 = row['EMA_9']
        ema20 = row['EMA_20']
        ema50 = row['EMA_50']
        ema100 = row['EMA_100']
        ema200 = row['EMA_200']
        rsi = row['RSI']
        rsi2 = row['RSI2']
        macd = row['MACD']
        signal = row['Signal_Line']

        ema_trend = 0

        if (price > ema9) and (ema9 > ema20):
            ema_trend = 3
        elif (price < ema9) and (ema9 < ema20):
            ema_trend = -3
        elif (price > ema9) or (ema9 > ema20):
            ema_trend = 1
        elif (price < ema9) or (ema9 < ema20):
            ema_trend = -1
        else:
            ema_trend = 0
        
        ema_score = ((price > ema9) + (ema9 > ema20)*0.8 + (ema20 > ema50)*0.6 + (ema50 > ema100)*0.4 +  (ema100 > ema200)*0.2 - (ema200 > ema100)*0.2
                   - (ema100 > ema50)*0.4 - (ema50 > ema20)*0.6 - (ema20 > ema9)*0.8 - (ema9 > price)) * 2/3 + ((price > ema9) and (ema9 > ema20)) - ((price < ema9) and (ema9 < ema20))
            
        rsi_score = 0
        
        if (rsi > rsi2) and (rsi > 50):
            rsi_score = ((rsi - 50)/50) * 2
        elif (rsi < rsi2) and (rsi > 50):
            rsi_score = (rsi - 50)/50
        elif (rsi < rsi2) and (rsi < 50):
            rsi_score = ((rsi - 50)/50) * 2
        elif (rsi > rsi2) and (rsi < 50):
            rsi_score = ((rsi - 50)/50) 
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

        pred = row['y_pred_poly']
        
        pred_1 = data_recent["y_pred_poly"].shift(1).loc[row.name]

        pred_score = 0

        if (pred > pred_1):
            pred_score = 1
        elif (pred < pred_1):
            pred_score = - 1
        else :
            pred_score = 0
        
        std_dev = row['std_dev']

        deviation_in_std = (price - pred) / std_dev
        
        std_score = - deviation_in_std

        score = ema_score + rsi_score + macd_score  + std_score + pred_score

        return pd.Series([ema_trend, ema_score, rsi_score, macd_score, score], 
                         index=["ema_trend", "ema_score", "rsi_score", "macd_score", "score"])

    # Apply function to DataFrame
    data_recent[["ema_trend", "ema_score", "rsi_score", "macd_score", "score"]] = data_recent.apply(calculate_scores, axis=1)

######################
    # Plot both linear and polynomial regression results on the same graph
    # Define a list of timeframes that support MACD
    valid_macd_timeframes = ["1m","5m","15m","30m","1h", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

    # Only plot MACD if the selected timeframe is valid
    #if interval in valid_macd_timeframes:
    fig, (ax0, ax4, ax2, ax3, ax) = plt.subplots(5, 1, figsize=(20, 40), gridspec_kw={'height_ratios': [1.5, 1.5, 1, 1, 4 ]})
    #else:
        #fig, (ax, ax2) = plt.subplots(2, 1, figsize=(20, 25), gridspec_kw={'height_ratios': [3, 1]})

    # Use numeric x-axis for plotting to avoid duplicate time issues
    x_values = np.arange(len(data_recent))  # Numeric x-axis

    # Plot actual prices and regression lines
    ax.plot(x_values, y, color="black", label="Actual Prices")  # Actual prices as a gray line plot
    ax.plot(x_values, y_pred_linear, color="red", label=f"L.R. (R² = {r2_linear:.2f})")
    ax.plot(x_values, y_pred_poly, color="green", label=f"P.R. (d {degree}, R² = {r2_poly:.2f})")

    # Draw bands for 1, 2, and 3 standard deviations from the polynomial model
    ax.fill_between(x_values, y_pred_poly - std_dev, y_pred_poly + std_dev, color="blue", alpha=0.3, label="")
    ax.fill_between(x_values, y_pred_poly - 2*std_dev, y_pred_poly + 2*std_dev, color="green", alpha=0.2, label="")
    ax.fill_between(x_values, y_pred_poly - 3*std_dev, y_pred_poly + 3*std_dev, color="red", alpha=0.1, label="")

################# Fit linear regression model
    
# Calculate the maximum distance between the regression line and actual prices
    #channel_length = 100
    dist = np.max(np.abs(y_pred_linear - y))
    
# Upper and lower channel lines
    upper_lr = y_pred_linear + dist
    lower_lr = y_pred_linear - dist

# Plot actual prices and regression lines
    ax.plot(x_values, upper_lr, color="blue", linestyle="--", label="Upper Channel")
    ax.plot(x_values, lower_lr, color="blue", linestyle="--", label="Lower Channel")
#############

    # Detect peaks and troughs
    peaks, _ = find_peaks(y)
    troughs, _ = find_peaks(-y)  # Invert y to find troughs

    n_std = 1.5  # Set threshold (e.g., 2 standard deviations)
    significant_peaks = [p for p in peaks if abs(y[p] - y_pred_poly[p]) >= n_std * std_dev]
    significant_troughs = [t for t in troughs if abs(y[t] - y_pred_poly[t]) >= n_std * std_dev]

# Add arrows for significant peaks (pointing downward)
    for p in significant_peaks:
        ax.annotate('', xy=(x_values[p], y[p]),
                    xytext=(x_values[p], y[p] + 0.5),  # Adjust offset as needed
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))

# Add arrows for significant troughs (pointing upward)
    for t in significant_troughs:
        ax.annotate('', xy=(x_values[t], y[t]),
                    xytext=(x_values[t], y[t] - 0.5),  # Adjust offset as needed
                    arrowprops=dict(arrowstyle='->', color='blue', lw=3))


############### Draw horizontal lines from the lowest and highest points
    
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

    # add time intervals on top of chart
    ax.text(0.4, 0.98, f"Time Frame: {interval}__now: {current_price:.2f}", 
        horizontalalignment='left', verticalalignment='center', 
        transform=ax.transAxes, fontsize=18, color="blue")
    
    # Draw gray line for d2 close
    d2_close = fetch_d2_close(ticker)
    ax.axhline(y=d2_close, color="navy", linestyle="--", label="")

    # Add price label for the d2_close
    ax.text(0, d2_close, f'{d2_close:.2f}__c2', color='navy', verticalalignment='top')

    # Draw exponential moving averages with dashed lines
    ax.plot(x_values, data_recent['EMA_9'], color="red", linestyle="--", label="EMA 9/20_blue")
    ax.plot(x_values, data_recent['EMA_20'], color="blue", linestyle="--", label="")
    ax.plot(x_values, data_recent['EMA_50'], color="gold", linestyle="--", label="EMA 50")
    ax.plot(x_values, data_recent['EMA_100'], color="gray", linestyle="--", label="EMA 100")
    ax.plot(x_values, data_recent['EMA_200'], color="purple", linestyle="--", label="EMA 200")

    #fig.canvas.mpl_connect('button_press_event', on_click)
    # Add price labels for EMAs
    ax.text(x_values[-1], data_recent['EMA_9'].iloc[-1], f'^^^^^^e9', color='red', verticalalignment='top')
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
    ax.set_title(f"Combined Linear and Polynomial Regression for {ticker} (tFrame: {interval})")
    ax.legend()

    # --- RSI Plot ---
    ax2.plot(x_values, data_recent['RSI'], color="navy", linestyle="-", label="RSI (14)")
    ax2.plot(x_values, data_recent['RSI2'], color="red", linestyle="--", label="RSI (25)")
    ax2.axhline(y=70, color="red", linestyle="--")
    ax2.axhline(y=30, color="green", linestyle="--")
    ax2.axhline(y=50, color="gray", linestyle="--")
    ax2.set_title(f"RSI ({interval})")
    ax2.legend()

    # === MACD Plot (Only If Timeframe Is Valid) ===
    if interval in valid_macd_timeframes:
        # Create a sequence for the x-axis from 1 to len(data_recent)
        x_values = range(1, len(data_recent) + 1)

        # Plot the MACD and Signal lines with numeric x-values
        ax3.plot(x_values, data_recent['MACD'], color="navy", label="MACD Line")
        ax3.plot(x_values, data_recent['Signal_Line'], color="red", linestyle="--", label="Signal Line")

        # Histogram Bars (Green for Positive, Red for Negative)
        histogram_values = data_recent['MACD'] - data_recent['Signal_Line']
        ax3.bar(x_values, histogram_values, color=['green' if val > 0 else 'red' for val in histogram_values], alpha=0.5)

        ax3.set_title(f"MACD ({interval})")
        ax3.legend()
        
######################################  score

    # Ensure index is datetime
     
    data_recent.index = pd.to_datetime(data_recent.index)

    # Sort by timestamp to avoid disorder  (1hr timeframe disorganized before this)
    #data_recent = data_recent.sort_index()

    # Instead of using time in seconds, use a simple range index
    data_recent["TimeIndex"] = np.arange(len(data_recent))
    
    # Create TimeIndex as seconds from the first timestamp
    #data_recent["TimeIndex"] = (data_recent.index - data_recent.index.min()).total_seconds()

    # Create Timestamp column from index
    data_recent["Timestamp"] = data_recent.index  # Use index instead of non-existing 'Timestamp' column

    # Extract the Hour and Minute
    data_recent["Hour"] = data_recent["Timestamp"].dt.strftime("%H:%M")  # Format as HH:MM

    X = data_recent[["TimeIndex"]].values  # Reshape needed for sklearn
    y = data_recent["score"].fillna(0).values

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)  # Ensure this line is present
    r2_linear = r2_score(y, y_pred_linear)

    # Plot actual scores and regression lines
    ax4.plot(x_values, y, color="navy", marker=".", linestyle="--", markersize=15, label="Actual score")# Actual prices as a gray line plot
    
    ax4.plot(x_values, y_pred_linear, color="gray", linestyle="--", label=f"L.R. (R² = {r2_linear:.2f})")
    ax4.plot(x_values, y_pred_poly, color="blue", label=f"P.R. (d {degree}, R² = {r2_poly:.2f})")

    # Draw horizontal lines 
    ax4.axhline(y=0, color="gray", linestyle="-", label="", linewidth=3)
    ax4.axhline(y=4, color="red", linestyle="--", label="")
    ax4.axhline(y=-4, color="green", linestyle="--", label="")
    ax4.axhline(y=6, color="red", linestyle="--", label="")
    ax4.axhline(y=-6, color="green", linestyle="--", label="")

    # add time intervals on bottom of chart
    ax4.text(0.3, 0.05, f"Time Frame: {interval}", 
        horizontalalignment='left', verticalalignment='center', 
        transform=ax.transAxes, fontsize=12, color="blue")

    # Format x-axis to show only hours (or every 3 hours for 30-minute interval)
    ax4.set_xticks(x_values)  # Set ticks for all time points
    ax4.set_xticklabels(simplified_time_labels)  # Show only hours or every 3 hours
    ax4.set_xlabel("Time (HH:MM)")
    ax4.set_ylabel("total score")
    ax4.set_title(f"Combined Linear and Polynomial Regression for score ({interval})")
    ax4.legend()

    st.write("---------------------")
    st.write(data_recent.tail(5))

    ### get scores functions
    def get_scores():
        ema_score = data_recent["ema_score"].iloc[-1]
        ema_trend = data_recent["ema_trend"].iloc[-1]
        rsi_score = data_recent["rsi_score"].iloc[-1]
        macd_score = data_recent["macd_score"].iloc[-1]
        score = data_recent["score"].iloc[-1]
        
        std_dev = data_recent["std_dev"].iloc[-1]
        y_pred_poly = data_recent["y_pred_poly"].iloc[-1]
        delta = current_price - y_pred_poly
        dev_from_std = delta/std_dev
        
        return ema_score, ema_trend, rsi_score, macd_score, score, dev_from_std

    def get_scores_more():
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
        y_pred_poly = data_recent['y_pred_poly'].iloc[-1]
        y_pred_poly1 = data_recent['y_pred_poly'].iloc[-2]

        return price, ema9, ema20, ema50, ema100, ema200, rsi, rsi2, macd, signal, y_pred_poly, y_pred_poly1

    #get all scores:
    ema_score, ema_trend, rsi_score, macd_score, score, dev_from_std = get_scores()
    price, ema9, ema20, ema50, ema100, ema200, rsi, rsi2, macd, signal, y_pred_poly, y_pred_poly1 = get_scores_more()

    ##################### e_trend scoreT.csv for bar charts

    # Load existing data if there is, examine if {interval} is already there.
    #if it is, then remove it and replace with new data
    
    # File path
    file_path = 'scoreT.csv'

    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)

        # Check if any row contains {interval} in the first column
        if interval in df[0].values:
            # Remove the row where the first column is {interval}
            df = df[df[0] != interval]

            # Save the updated DataFrame back to the CSV file
            df.to_csv(file_path, index=False, header=False)
        else:
            print("No row with '{interval' found. File remains unchanged.")
    else:
        # If the file doesn't exist or is empty, create a new DataFrame
        print("File does not exist or is empty. Creating a new file.")

        df = pd.DataFrame(columns=['tFrame', 'ema_trend', 'ema', 'rsi','macd', 'total', 'dev_from_std', 'score_trend'])

        # Save the empty DataFrame to the CSV file
        df.to_csv(file_path, index=False, header=False)
        st.success(f"✅ File created successfully as `{file_path}`")

    ################### evaluate score trend and save it to scoreT.csv
    #score_prior = data_recent['score'].iloc[-2]
    #score_prior2 = data_recent['score'].iloc[-3]

    #if (score_prior > score_prior2) and score_prior >= 1 and data_recent['ema_trend'].iloc[-2] >= 1:
        #score_trend_1 = 1
    #elif (score_prior < score_prior2) and score_prior <= -1 and data_recent['ema_trend'].iloc[-2] <= -1:
       # score_trend_1 = -1
    #else:
       # score_trend_1 = 0
      
    #ema_score, ema_trend, rsi_score, macd_score, score
    y_pred_p_trend = 0
    if y_pred_poly  >= y_pred_poly1:
        y_pred_p_trend = 1
    else:
        y_pred_p_trend = -1
        
    if ema_score >=1 and rsi_score > 0 and macd_score > 0 and score > 0:
        score_trend = 1
    elif ema_score <=-1 and rsi_score < 0  and macd_score < 0 and score < 0 :
        score_trend = -1
    else:
        score_trend = 0
        
    #if (score > score_prior) and score >= 1 and data_recent['ema_trend'].iloc[-1] >= 1:
        #score_trend = 1
    #elif (score < score_prior) and score <= -1 and data_recent['ema_trend'].iloc[-1] <= -1:
        #score_trend = -1
    #else:
        #score_trend = 0

    #score4 = (ema_score>0) + (rsi_score>0) + (macd_score>0) + (score>0) + (ema_score<0) + (rsi_score<0) + (macd_score<0) + (score<0)
    # total == score (above) 
    new_data = pd.DataFrame([{
        "tFrame": f"{interval}",
        "ema_trend": round(ema_trend, 2),
        "ema": round(ema_score, 2),
        "rsi": round(rsi_score, 2),
        "macd": round(macd_score, 2),
        "total": round(score, 2),
        "dev_from_std": deviation_in_std,
        "y_pred_p_trend": y_pred_p_trend,
        "score_trend": score_trend,
    }])
    #new_data.to_csv(scoreT_file, mode="a", header=False, index=False)
    new_data.to_csv(scoreT_file, mode="a", header=False, index=False, float_format="%.2f") ## chatGPT

    # Read the updated CSV file
    df = pd.read_csv(file_path, header=None)

    # Define the custom order for the first column
    custom_order = ["1m", "5m", "15m", "30m", "1h", "3mo", "6mo"]

    # Convert the first column to a categorical type with the custom order
    df[0] = pd.Categorical(df[0], categories=custom_order, ordered=True)

    # Sort the DataFrame by the first column using the custom order
    df = df.sort_values(by=0)

    #add column names
    df.columns = ['tFrame', 'ema_trend', 'ema', 'rsi', 'macd', 'total', 'dev_from_std', "y_pred_p_trend", 'score_trend']
        
    #display table
    st.dataframe(df, hide_index=True) #original table looks neater

    ################### all control buttons ###########################################################
    ## very important use
    current_price = round(data_recent['Close'].iloc[-1], 2)
    now = datetime.now(eastern).strftime('%m-%d %I:%M:%S %p')  # Correct format
    ema_trend_1m = df[df["tFrame"] == "1m"]["ema_trend"].values[0]
    ema_trend_5m = df[df["tFrame"] == "5m"]["ema_trend"].values[0]
    ############ investigate score_trends
    
    st.write(f"interval: {interval}__rerun:{ st.session_state.rerun_count}")
    # Extract "score_trend" for "1m"  ## message
    if ema_trend_1m <=0:
        message = "B OK <= 0"
        color = "green"
    elif ema_trend_1m >=0:
        message = "S OK >= 0"
        color = "red"
    else:
        message = "Hold it"
        color = "orange"

    if ema_trend_5m <=0:
        message5 = "B OK <= 0"
        color5 = "green"
    elif ema_trend_5m >=0:
        message5 = "S OK >= 0"
        color5 = "red"
    else:
        message5 = "Hold it"
        color5 = "orange"
    #st.write(f"ema_trend_1min: ||... {ema_trend_1m: .0f} ___ {message}")
    st.markdown(f'<p style="color:{color}; font-weight:bold;">ema_trend_1min: {message}</s></p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{color5}; font-weight:bold;">ema_trend_5min: {message5}</s></p>', unsafe_allow_html=True)

    sum_score_trend_rest = df[~df["tFrame"].isin(["1m", "3mo", "6mo"])]["score_trend"].sum()
    
    if sum_score_trend_rest >=3:
        message = "___B OK >=3"
        color = "green"
    elif sum_score_trend_rest <= -3:
        message = "___S OK <=-3"
        color = "red"
    else:
        message = "Hold it"
        color = "orange"
    #st.write(f"score_trend_others: ||... {sum_score_trend_rest} ___ {message}")
    st.markdown(f'<p style="color:{color}; font-weight:bold;">score_trend_others: {message}__{sum_score_trend_rest}</s></p>', unsafe_allow_html=True)
     #display message about app status
    sleep_status = 'on' if st.session_state.stop_sleep == 0 else "off"
    updated_data = pd.read_csv(pe_file, names=["type", "B_pr", "S_pr", "pl", "total", "temp_pr"])

###########################

    b_condition =  ema_trend_1m >= -3 and ema_trend_5m >= -3 and ema_trend_1m <= 1 and y_pred_p_trend == 1 and sum_score_trend_rest >= 3
    s_condition = ((current_price - st.session_state.temp_price) >=1.0) or ((current_price - st.session_state.temp_price) <= -0.5)
    short_b = ((current_price - st.session_state.temp_price) <= -1.0) or ((current_price - st.session_state.temp_price) >= 0.5)              
    short_s = ema_trend_1m <= 3 and ema_trend_5m <= 3 and ema_trend_1m >= -1  and sum_score_trend_rest <= -3

    ########## B and S actions
    def save_pe(type="AAA", price=None, total =0): 
        updated_data = pd.read_csv(pe_file, names=["type", "B_pr", "S_pr", "pl", "total", "temp_pr"])
        pl=0
        if type == "S":
            pl = price - updated_data["temp_pr"].iloc[-1]
        elif type == "SB":
            pl = updated_data["temp_pr"].iloc[-1] - price
        else:
            pl = 0
        total = total + pl
            
        if type == "B":
            new_data = pd.DataFrame([{
                    "TimeStamp": f"{now}",
                    "type": "B",
                    "B_pr": round(price, 2),
                    "S_pr": 0,
                    "pl": pl,
                    "total": round(total, 2),
                    "temp_price": round(price, 2),
                }])

        elif type == "S":
            new_data = pd.DataFrame([{
                    "TimeStamp": f"{now}",
                    "type": "S",
                    "B_pr": 0,
                    "S_pr": round(price, 2),
                    "pl": round(pl, 2),
                    "total": round(total, 2),
                    "temp_price": 0,
                }])

        elif type == "SS":
            new_data = pd.DataFrame([{
                    "TimeStamp": f"{now}",
                    "type": "SS",
                    "B_pr": 0,
                    "S_pr": round(price, 2),
                    "pl": 0,
                    "total": total, ## for now
                    "temp_price": round(price, 2),
                }])

        elif type == "SB": 
            new_data = pd.DataFrame([{
                    "TimeStamp": f"{now}",
                    "type": "SB",
                    "B_pr": round(price, 2),
                    "S_pr": 0,
                    "pl": round(pl, 2),
                    "total": round(total, 2),
                    "temp_price": 0,
                }])
        # Append to CSV file
        new_data.to_csv(pe_file, mode="a", header=False, index=False)
            
    #####################################
    #st.write(f"### Controls:  ||______ current_price = {current_price:.2f}______")
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # delete data button
        if st.button("Ready SB: 1min"):
            #st.session_state.rerun_count = 0
            st.session_state.index = 0
           # st.session_state.stop_sleep = 0
            #st.session_state.sbOK = 1
            st.session_state.sleepGap = 6 # !=5: will keep 1min
            st. rerun()

    with col2:
        # delete data button
        if st.button("keep 1m 5m"):
            #st.session_state.rerun_count = 0
            
            st.session_state.index = 0
            #st.session_state.sbOK = 1
            st.session_state.sleepGap = 7 # !=5 or 6 will keep 5min
            st. rerun()
            
    with col3:
        if st.button("check all"):
           # if st.button("clear first"):
              #  new_data = pd.DataFrame([{
                  ##  "tFrame": "5m", 
                  #  "ema_trend": 0,
                #    "ema": 0,
                   # "rsi": 0,
                  #  "macd": 0,
                    #"total": 0,
                  #  "dev_from_std": 0,
                   # "score_trend": 0,
               # }])
                #new_data.to_csv(scoreT_file, mode="w", header=False, index=False, float_format="%.2f") ## chatGPT
              #  st.rerun()
            st.session_state.index = 0
            st.session_state.rerun_count = 0
            st.session_state.sleepGap = 5
            st.rerun()

    with col4:
        st.write(f"now: sleep__ {st.session_state.sleepGap}")

    st.write("wait for 15min, 30min, 1hr, 3mo to line up")
    st.write("wait for 1min, 5min to curve up or down")
    st.write("wait for 1min, 5min pr to push to opposit")

# Input box for setpr

# Initialize the session state variable if it doesn't exist
    if "setpr" not in st.session_state:
        st.session_state.setpr = 0.0
    if "settype" not in st.session_state:
        st.session_state.settype = "zz"
# Create a text input that displays the current session state value
    col1, col2=st.columns(2)
    with col1:
        setpr_input = st.text_input("Enter set pr: ", value=str(st.session_state.setpr))
    with col2:
        settype_input = st.text_input("Enter set type (only needed if B or SS): ", value=str(st.session_state.settype))
    #set them
    col1, col2=st.columns(2)
    with col1:
        if st.button("set"):
            try:
    # Attempt to convert the input to a float and update the session state
                st.session_state.setpr = float(setpr_input)
                st.session_state.settype = settype_input
                st.session_state.confirmation_message = f"Success!"
            except ValueError:
    # Handle invalid input (non-numeric values)
                st.error("Please enter a valid number or type")
            st.rerun()
# Display the current value of setpr from the session state
    with col2:
        st.write(f"setpr: {st.session_state.setpr}__now: {current_price}__settype:{st.session_state.settype}")
    
    col1, col2 = st.columns(2)
    total = updated_data["total"].iloc[-1]
    SB = updated_data["type"].iloc[-1]
    with col1:
        if st.button("B >>>>>>"):
            if  (SB == "AAA" or SB == "S" or SB== "SB") and interval == "1m":
                save_pe("B", current_price, total)
                st.write(f"B: Yes ||SB_status: {SB}")

            elif SB == "SS" and interval == "1m":
                save_pe("SB", current_price, total)
                st.write(f"SB: Yes ||SB_status: {SB}")
                
            else:
                st.write(f"NO, Can not ||SB_status: {SB}__interval: {interval}")
                
            st.rerun()

    with col2:
        if st.button("S >>>>>>"):
            if  SB == "B" and interval == "1m":
                save_pe("S", current_price, total)
                st.write(f"S: Yes ||SB_status: {SB}")

            elif (SB == "AAA" or SB == "S" or SB== "SB") and interval == "1m":
                save_pe("SS", current_price, total)
                st.write(f"SS: Yes ||SB_status: {SB}")
                
            else:
                st.write(f"NO, Can not ||SB_status: {SB}__interval: {interval}")
                
            st.rerun()

    st.write(f"SB_type: {updated_data["type"].iloc[-1]}")
    #show which timeframes are in bar chart:
    timeframes = ["1m", "5m", "15m", "30m", "1h", "3mo", "6mo"]
    message_here = timeframes[:st.session_state.rerun_count]

    #display pe_table
    # Read the updated CSV file ---- example
    updated_data = pd.read_csv(pe_file, names=["type", "B_pr", "S_pr", "pl", "total", "temp_pr"])
    plnow = 0
    if updated_data["type"].iloc[-1]=="B":
        plnow = current_price - updated_data["temp_pr"].iloc[-1]
    elif updated_data["type"].iloc[-1]=="SS":
        plnow = - current_price + updated_data["temp_pr"].iloc[-1]
    st.markdown(f'<p style="color:orange; font-weight:bold;">pe_table: ___now interval__{interval}___now pl:__{plnow:.2f}</s></p>', unsafe_allow_html=True)
    #st.write(f"pe_table: _______now interval: {interval}")
    #with col1:
    st.dataframe(updated_data.tail(5), hide_index=False)
    st.write(f"{len(updated_data["total"])} rows")
    #with col2:

       # st.write(":::::::::::::::::::::::::::::::::::::::::::::::::::")
    st.write(f"now: _<{now}>_{get_time_now()}")
    message1 = 1 if {b_condition} == True else 0
    message2 = 1 if {s_condition} == True else 0
    
    #st.write(f"Pre_Post_status: {st.session_state.prepo}")
    if st.button("Clear data"):
        #st.session_state.stop_sleep = 1
        new_data = pd.DataFrame([{
                    "TimeStamp": f"{now}",
                    "type": "AAA",
                    "B_pr": 0,
                    "S_pr": 0,
                    "pl": 0,
                    "total": 0, 
                    "temp_pr": 0
                }])
                # clear CSV file
        new_data.to_csv(pe_file, mode="w", header=False, index=False)
        st.write("data cleared")
        st.rerun()
            
    st.write("---------------------")
################### do bar graph using scoreT_file
    
    # Load data from the CSV file
    try:
        df = pd.read_csv(scoreT_file, names=["tFrame", "ema_trend", "ema", "rsi", "macd", "total", "score_trend"])
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

############################# status 3
   
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

############################## display ema, rsi, macd trends columns

    st.write(f"### Indicator trend ({interval})")
    
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
            
        elif price < ema9 and ema9 < ema20:
            message = "Down"
            color1 = "red"
            
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
            
        elif rsi < rsi2 and rsi < 50:
            message = "Down"
            color2 = "red"
            
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
            
        elif macd < signal and macd < 0:
            message = "Down"
            color3 = "red"
            
        else:
            message = "Neutral "
            color3 = "gray"
            
        # Display the table
        st.markdown(f"### <span style='color:{color3};'>MACD: {message}</span>", unsafe_allow_html=True)
        st.dataframe(macd_df, hide_index=True)


###################### bar chart?
    
   ## read bar data scoreT_file
    df = pd.read_csv(scoreT_file, names=["tFrame", "ema_trend", "ema", "rsi", "macd", "total", "dev_from_std", "score_trend"])
    
    # Define custom order
    timeframe_order = ["1m", "5m", "15m", "30m", "1h", "3mo", "6mo"]

    # Convert 'tFrame' column to categorical with defined order
    df["tFrame"] = pd.Categorical(df["tFrame"], categories=timeframe_order, ordered=True)

    # Sort DataFrame based on the categorical order
    df = df.sort_values("tFrame")
    
    ## plotting barchart
    ax0.set_ylim(-10, 10)  # Adjust Y-axis limits if needed

    # Get unique intervals and prepare x-axis locations
    unique_intervals = df["tFrame"].unique()
    x = np.arange(len(unique_intervals))  # X locations for groups
    width = 0.15  # Width of each bar

    # Prepare values for each metric
    ema_trend = [df[df["tFrame"] == interval]["ema_trend"].mean() for interval in unique_intervals]
    ema_values = [df[df["tFrame"] == interval]["ema"].mean() for interval in unique_intervals]
    rsi_values = [df[df["tFrame"] == interval]["rsi"].mean() for interval in unique_intervals]
    macd_values = [df[df["tFrame"] == interval]["macd"].mean() for interval in unique_intervals]
    total_values = [df[df["tFrame"] == interval]["total"].mean() for interval in unique_intervals]

    ##########
    # Define offsets for each bar group
    offsets = [-2 * width, -width, 0, width, 2 * width]

    # Plot bars with proper spacing
    ax0.bar(x + offsets[0], ema_trend, width, color="cyan", edgecolor="black", label="ema_trend")
    ax0.bar(x + offsets[1], ema_values, width, color="purple", edgecolor="black", label="EMA")
    ax0.bar(x + offsets[2], rsi_values, width, color="navy", edgecolor="black", label="RSI")
    ax0.bar(x + offsets[3], macd_values, width, color="orange", edgecolor="black", label="MACD")
    ax0.bar(x + offsets[4], total_values, width, color="gray", edgecolor="black", label="total")

    # Add value labels on top of each bar
    for i, interval in enumerate(unique_intervals):
        ax0.text(x[i] + offsets[0], ema_trend[i] + 0.2, f"{ema_trend[i]:.1f}", ha='center', fontsize=10, color="black")
        ax0.text(x[i] + offsets[1], ema_values[i] + 0.2, f"{ema_values[i]:.1f}", ha='center', fontsize=10, color="black")
        ax0.text(x[i] + offsets[2], rsi_values[i] + 0.2, f"{rsi_values[i]:.1f}", ha='center', fontsize=10, color="black")
        ax0.text(x[i] + offsets[3], macd_values[i] + 0.2, f"{macd_values[i]:.1f}", ha='center', fontsize=10, color="black")
        ax0.text(x[i] + offsets[4], total_values[i] + 0.2, f"{total_values[i]:.1f}", ha='center', fontsize=10, color="black")
    # Add horizontal lines at y = 4 and y = -4
    ax0.axhline(y=3, color="red", linestyle="--", linewidth=1, label="Threshold (4)")
    ax0.axhline(y=-3, color="green", linestyle="--", linewidth=1, label="Threshold (-4)")
    ax0.axhline(y=0, color="gray", linestyle="-", linewidth=3, label="Threshold (-4)")

    # Add labels and title

    legend_handles = [
        Line2D([0], [0], color='cyan', lw=2, label="ema_trend"),
        Line2D([0], [0], color='purple', lw=2, label="EMA"),
        Line2D([0], [0], color='navy', lw=2, label="RSI"),
        Line2D([0], [0], color='orange', lw=2, label="MACD"),
        Line2D([0], [0], color='gray', lw=2, label="total"),
    ]
    time = datetime.now(eastern).strftime('%D:%H:%M')
    ax0.set_xlabel("Time Frame")
    ax0.set_ylabel("Score")
    ax0.set_title(f"Trend Scores by Interval({time})")
    ax0.set_xticks(x)
    ax0.set_xticklabels(unique_intervals, rotation=45)
    ax0.legend(handles=legend_handles, loc="lower right")

    #################################

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)  ## finally plot all 5 figures

########################################
   # if st.session_state.stop_sleep == 0:
    # Sleep for 8 seconds (simulating some processing)
    sleep(st.session_state.sleepGap)
    if st.session_state.sleepGap == 5:
        # Update the index for the next interval
        if st.session_state.index < len(intervals) - 1:
            st.session_state.index += 1
        else:
            st.session_state.index = 0
        st.session_state.rerun_count +=1
    elif st.session_state.sleepGap == 7:
        if st.session_state.index ==0:
            st.session_state.index = 1
        else:
            st.session_state.index = 0
    elif st.session_state.sleepGap == 6:
        st.session_state.index = 0
    
    ### run automatic SB
    total = updated_data["total"].iloc[-1]
    SB = updated_data["type"].iloc[-1]
    if (b_condition or (current_price <= st.session_state.setpr and st.session_state.settype =="B")) and (SB == "AAA" or SB == "S" or SB == "SB") and interval == "1m":
        save_pe("B", current_price, total)
        st.write(f"B: Yes ||SB_status: {SB}")
              
    elif (s_condition or current_price >= st.session_state.setpr) and SB == "B" and interval == "1m":
        save_pe("S", current_price, total)
        st.write(f"S: Yes ||SB_status: {SB}")

    elif (short_s or (current_price >= st.session_state.setpr and st.session_state.settype =="SS")) and (SB == "AAA" or SB == "S" or SB== "SB") and interval == "1m":
        save_pe("SS", current_price, total)
        st.write(f"SS: Yes ||SB_status: {SB}")

    elif (short_b or current_price <= st.session_state.setpr) and SB == "SS" and interval == "1m":
        save_pe("SB", current_price, total)
        st.write(f"SB: Yes ||SB_status: {SB}")
    
   # st.empty()
    st.rerun()
        

if __name__ == "__main__":
    main()
