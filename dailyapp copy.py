### daily analysis 03-08-25

import os
import io
import time
from time import sleep
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
from pytz import timezone

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import pandas_market_calendars as mcal
#from gtts import gTTS  # Text-to-speech

eastern_tz = pytz.timezone('US/Eastern') 
current_date = datetime.now(eastern_tz).strftime("%Y-%m-%d %H:%M")

marker_position = 540
separationLen = 5
marker_width = 1 if separationLen <= 0.2 else 3
markerColor = "navy"
style = "-"

fontsize = 20

# Set page configuration to collapse the sidebar initially
st.set_page_config(initial_sidebar_state="collapsed")

#markerColor = "#4444FF"#dark blue
#eastern = pytz.timezone("America/New")
eastern = pytz.timezone("US/Eastern")
bgcolor = "lightblue"

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
    data = stock.history(period="1mo", interval=interval, prepost=True)  # no doest not Include premarket data
    return data

# Function to fetch the previous 5 day's close price
def fetch_daily5(ticker):
    stock = yf.Ticker(ticker)
    daily5 = stock.history(period="3mo")  # Fetch 30 days of data
    if len(daily5) >= 2:
        return daily5['Close'][-5:]  ## return only last 5 days 
    else:
        return None  # Handle cases where there isn't enough data

def fetch_long_interval(ticker="SPY", interval= "6mo"):
    stock = yf.Ticker(ticker)
    daily = stock.history(period=interval)
    if len(daily) >= 2:
        return daily
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

    # Create Timestamp column from index
    data_recent["Timestamp"] = data_recent.index  # Use index instead of non-existing 'Timestamp' column

    # Extract the Hour and Minute
    data_recent["Hour"] = data_recent["Timestamp"].dt.strftime("%H:%M")  # Format as HH:MM

    X = data_recent[["TimeIndex"]].values  # Reshape needed for sklearn
    y = data_recent["score"].fillna(0).values

    # Polynomial Regression
    poly = PolynomialFeatures(degree=3)
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

def plot_bars(price=0):
    # Assuming 'eastern' timezone is defined elsewhere
    eastern = 'US/Eastern'  # Example timezone, replace with actual timezone if needed

    # Read the data
    df = pd.read_csv("scoreT.csv", names=['tFrame', 'ema_trend', 'e100/200', 'pr_eAvg', 'rsi', 'macd', 'score',  'score_trend'])
    
    # Define custom order
    timeframe_order = ["1m", "5m", "15m", "30m", "1h", "3mo", "6mo", "1y"]
    
    # Convert 'tFrame' to categorical with order
    df["tFrame"] = pd.Categorical(df["tFrame"], categories=timeframe_order, ordered=True)
    df = df.sort_values("tFrame")
    
    # Prepare data
    unique_intervals = df["tFrame"].unique()
    x = np.arange(len(unique_intervals))
    width = 0.15
    
    # Calculate metric values
    ema_trend = [df[df["tFrame"] == interval]["ema_trend"].mean() for interval in unique_intervals]
    ema_values = [df[df["tFrame"] == interval]["e100/200"].mean() for interval in unique_intervals]
    premaAvg = [df[df["tFrame"] == interval]["pr_eAvg"].mean() for interval in unique_intervals]
  
    rsi_values = [df[df["tFrame"] == interval]["rsi"].mean() for interval in unique_intervals]
    macd_values = [df[df["tFrame"] == interval]["macd"].mean() for interval in unique_intervals]
    total_values = [df[df["tFrame"] == interval]["score"].mean() for interval in unique_intervals]
    
    # Define bar positions
    # Define the width of each bar
    width = 0.15

# Generate offsets for six bars (equally spaced)
    offsets = [-3 * width, -2 * width, -width, 0, width, 2 * width]

# Now you can use this `offsets` list for your six bars

    #offsets = [-2 * width, -width, 0, width, 2 * width]
    
    # Plot bars
    plt.figure(figsize=(12,5),facecolor='lightgray')
    
    plt.bar(x + offsets[0], ema_trend, width, color="red", edgecolor="black")
    plt.bar(x + offsets[1], ema_values, width, color="darkred", edgecolor="black")
    plt.bar(x + offsets[2], premaAvg, width, color="darkred", edgecolor="black")
    plt.bar(x + offsets[3], rsi_values, width, color="navy", edgecolor="black")
    plt.bar(x + offsets[4], macd_values, width, color="orange", edgecolor="black", label="MACD")
    plt.bar(x + offsets[5], total_values, width, color="gray", edgecolor="black")
    
    # Add value labels
    for i, interval in enumerate(unique_intervals):
        for offset, values in zip(offsets, [ema_trend, ema_values, premaAvg,rsi_values, macd_values, total_values]):
            plt.text(x[i] + offset, values[i] + 0.2, f"{values[i]:.1f}", ha='center', fontsize=10)
    
    # Add threshold lines
    plt.axhline(y=7, color="red", linestyle="--", linewidth=1)
    plt.axhline(y=-7, color="green", linestyle="--", linewidth=1)
    plt.axhline(y=0, color="gray", linestyle="-", linewidth=1)
    
    #current_time = datetime.now(eastern).strftime('%m/%d/%Y %H:%M')
    plt.xlabel("Time Frame")
    plt.ylabel("Score")
    plt.title(f"Trend Scores by Interval (pr now: {price:.2f})")
    
    # Format x-axis
    plt.xticks(x, unique_intervals, rotation=45)
    
    # Add legend and adjust layout
    plt.legend()
    plt.tight_layout()
    plt.gca().set_facecolor(bgcolor)

# Display the plot
    st.pyplot(plt)
    plt.close()  # Prevent memory leaks

# Streamlit app
def main():
    st.title("Score Regression Analysis")
    
    # Input box for user to enter stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., SPY, AAPL, TSLA):", value="SPY").upper()
    
    # Initialize session states
    if 'index' not in st.session_state:
        st.session_state.index = 0
        
    if 'rerun_count' not in st.session_state:
        st.session_state.rerun_count = 0
        
    if "stop_sleep" not in st.session_state:
        st.session_state.stop_sleep = 0

    if "temp_price" not in st.session_state:
        st.session_state.temp_price = 0

    if "sb_status" not in st.session_state:
        st.session_state.sb_status = 0

    if "sleepGap" not in st.session_state:
        st.session_state.sleepGap = 5

    if "setpr" not in st.session_state:
        st.session_state.setpr = 0
 
    if "setnote" not in st.session_state:
        st.session_state.setnote = "zz"
        
    if "tf" not in st.session_state:
        st.session_state.tf = 0.2

    if "marker_status" not in st.session_state:
        st.session_state.marker_status = False
        
    scoreT_file = f"scoreT.csv"
    pe_file = f"pe.csv"

    # # Add a slider for backtracking
    #backtrack_options = [50, 300]
    #selected_datanumber = st.slider(
        #"Select number of points to plot:",
       # min_value=min(backtrack_options),
        #max_value=max(backtrack_options),
        #value=192,  # Default value 5 min for 16 h  per day
        #step=10,  # Step size
        #key="backtrack_slider"
    #)
    
    # List of intervals
    intervals = ['1m', '5m', '15m', '30m', '1h', '3mo', '6mo', '1y']
    interval = intervals[st.session_state.index]

    selectedNum =[300,300,300,300,300,300,300,300]
    selected_datanumber = selectedNum[st.session_state.index]

    tfList = [0.31, 1.55, 4.65,9.3,18.6,60,120,240]
    st.session_state.tf = tfList[st.session_state.index]

    # Add a button group for interval selection
    col1, col2, col3, col4, col5, col6, col7, col8= st.columns(8)
    with col1:
        if st.button("1min"):
            st.session_state.index = 0
            st.rerun()
    with col2:
        if st.button("5min", key="5m"):
            st.session_state.index = 1
            st.rerun()
            
    with col3:
        if st.button("15min", key="15m"):
            st.session_state.index = 2
            st.rerun()
            
    with col4:
        if st.button("30min", key="30m"):
            st.session_state.index = 3
            st.rerun()
            
    with col5:
        if st.button("1hr", key="1h"):
            st.session_state.stop_sleep = 1
            st.session_state.index = 4
            st.rerun()
            
    with col6:
        if st.button("3mo", key="3mo"):
            st.session_state.index = 5
            st.rerun()
            
    with col7:
        if st.button("6mo", key="6mo"):
            st.session_state.index = 6
            st.rerun()
            
    with col8:
        if st.button("1y", key="1y"):
            st.session_state.index = 7
            st.session_state.stop_sleep = 1
            st.rerun()

    # Fetch data for the user-specified stock and interval

    if interval == "1h" or interval == "30min":
        data = fetch_stock_data1mo("SPY", interval)
       # stock = yf.Ticker("SPY")
        #st.write(stock.history_metadata)
        if data.empty:
            st.error(f"Failed to fetch data for 1h. Please check the ticker and try again.")
            
    elif interval == "3mo":
        data = fetch_long_interval("SPY", "3mo")
        if data.empty:
            st.error(f"Failed to fetch data for 3mo. Please check the ticker and try again.")
    elif interval == "6mo":
        data = fetch_long_interval("SPY", "6mo")
        if data.empty:
            st.error(f"Failed to fetch data for 6mo. Please check the ticker and try again.")
    elif interval == "1y":
        data= fetch_long_interval("SPY",  "1y")
        if data.empty:
            st.error(f"Failed to fetch data for 1y. Please check the ticker and try again.")
    else:
        data = fetch_stock_data("SPY", interval)

    #if data.empty:
        #st.error(f"Failed to fetch data for SPY. Please check the ticker and try again.")
        #return
    data_recent = data.tail(selected_datanumber)  # Use only the first 300 points after backtracking

    st.write(data_recent["Close"].isnull().sum()) 
    data_recent = data_recent.dropna(subset=['Close'])
        
##########$$$############### tap function 
    
########$$$#data_recent = data_recent.head(100)  # Use only the first 300 points after backtracking

    # Calculate EMAs
    data_recent = calculate_emas(data_recent)
    
    # Get the current price (last available price in the data)
    current_price = data_recent['Close'].iloc[-1]

    # Fetch the previous day's close price
    previous_close = 0
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
    degree = 2
    
    # Perform linear regression (using only the most recent 300 points)
    X, y, y_pred_linear, r2_linear, data_recent = perform_regression(data_recent, degree=1)

    # Perform polynomial regression with the selected degree
    X, y, y_pred_poly, r2_poly, _ = perform_regression(data_recent, degree=2)

    # Calculate residuals for each row
    data_recent["residuals"] = y - y_pred_poly
    # Option 1: Rolling standard deviation (e.g., over 50 time points)
    data_recent["std_dev"] = data_recent["residuals"].rolling(window=30).std()## first 50 time points no data

    std_dev = data_recent["std_dev"].iloc[-1]

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
        data3mo =  fetch_long_interval(ticker, interval= "3mo")
        time_labels = data3mo.index.strftime('%Y-%m-%d')  # Format to YYYY-MM-DD
        simplified_time_labels = [label if idx % 9 == 0 else '' for idx, label in enumerate(time_labels)]

    elif interval == "6mo":
        data6mo =  fetch_long_interval(ticker, interval= "6mo")
        time_labels = data6mo.index.strftime('%Y-%m-%d')  # Format to YYYY-MM-DD
        simplified_time_labels = [label if idx % 9 == 0 else '' for idx, label in enumerate(time_labels)]    

    elif interval == "1y":
        data1y =  fetch_long_interval(ticker, interval= "1y")
        time_labels = data1y.index.strftime('%Y-%m-%d')  # Format to YYYY-MM-DD
        simplified_time_labels = [label if idx % 9 == 0 else '' for idx, label in enumerate(time_labels)]    
    else:
        # For 1-minute and 5-minute intervals, show only hours (e.g., 09:00, 10:00)
        simplified_time_labels = [label if label.endswith('00') else '' for label in time_labels]

    # Calculate the deviation of the current price from the polynomial regression model
    #current_price_deviation = current_price - y_pred_poly[-1]  # Deviation from the polynomial model
    #deviation_in_std = round(current_price_deviation / std_dev ,0) # Deviation in terms of standard deviations

    # Add a message above the plot showing the price deviation
    #if deviation_in_std >= 1:
        #deviation_message = f"{ticker}_Deviation from PR: +{deviation_in_std:.2f} std_dev"
        #deviation_color = "red"  # Red for >= +2 std_dev
    #elif deviation_in_std <= -1:
        #deviation_message = f"{ticker}_Deviation from PR: {deviation_in_std:.2f} std_dev"
        #deviation_color = "green"  # Green for <= -2 std_dev
    #else:
        #deviation_message = f"{ticker}_Deviation from PR: {deviation_in_std:.2f} std_dev"
        #deviation_color = "gray"  # Default color for other cases

    # Display the deviation message with the appropriate color
   # st.markdown(f"<h3 style='color:{deviation_color};'>{deviation_message} ({interval})</h3>", unsafe_allow_html=True)
    
    dataSimple = data_recent[["Close", "Volume"]]
    dataSimple['Close'] = dataSimple['Close'].round(2)

# Verify the result
    st.write(dataSimple.tail())
    
    col1, col2, col3=st.columns([2,1,1])
    with col1:      
    # Add a message above the plot showing the trend
        st.markdown(f"<h3 style='color:{trend_color};'>{ticker}_{trend_message} ({interval})</h3>", unsafe_allow_html=True)

    with col2:
        if data_recent["Volume"].iloc[-1] > 1.91*data_recent["Volume"].iloc[-2]:
            st.markdown("### High Volume!")
        else:
            st.markdown("### …")
        
    with col3:
    #add button to toggle marker lines   
         # Align the button to the right using CSS
        st.markdown(
            """
            <style>
            div.stButton > button {
                float: right;
                #margin-top: 20px; /* Optional: Add some spacing from the top */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Markers on" if st.session_state.marker_status else "Markers off"):
            st.session_state.marker_status = not st.session_state.marker_status  # Toggle the status
            st.rerun()
        
    # Calculate RSI before plotting
    data_recent = calculate_rsi(data_recent)
    data_recent = calculate_macd(data_recent)
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

        if (ema9 >= ema20):
            ema_trend = 1
        else:
            ema_trend = -1
        
        ema_score = 0

        if (ema100>= ema200):
            ema_score = 1
        else:
            ema_score = -1
            
        rsi_score = 0
        
        if (rsi >= rsi2) and (rsi >= 50):
            rsi_score = 2
        elif (rsi < rsi2) and (rsi >= 50):
            rsi_score = 1
        elif (rsi < rsi2) and (rsi < 50):
            rsi_score = -2
        elif (rsi >= rsi2) and (rsi < 50):
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

        score = ema_trend + ema_score + rsi_score + macd_score 

        return pd.Series([ema_trend, ema_score, rsi_score, macd_score, score], 
                         index=["ema_trend", "ema_score", "rsi_score", "macd_score", "score"])

    # Apply function to DataFrame
    data_recent[["ema_trend", "ema_score", "rsi_score", "macd_score", "score"]] = data_recent.apply(calculate_scores, axis=1)

    # Define a list of timeframes that support MACD
    valid_macd_timeframes = ["1m","5m","15m","30m","1h", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

    #if interval in valid_macd_timeframes:
    fig, (ax2, ax3, ax) = plt.subplots(3, 1, figsize=(20, 25), gridspec_kw={'height_ratios': [ 1, 1, 4 ]})
   
    # Use numeric x-axis for plotting to avoid duplicate time issues
    x_values = np.arange(len(data_recent))  # Numeric x-axis

    # Plot actual prices and regression lines
    open=data_recent["Open"]
    close=data_recent["Close"]
    
   # hlLimit = 1
  #  high = data_recent["High"].copy()
   # low = data_recent["Low"].copy()
   ## close = data_recent["Close"]???
 #   open = data_recent["Open"]

# Apply conditions using NumPy vectorized operations
#    high = np.where(high - close >= hlLimit, close + hlLimit, high)
  #  low = np.where(low - close <= -hlLimit, close - hlLimit, low)

  #  if interval in ["1m", "5m", "15m"]:
     #   open = np.where(abs(open - high )>= hlLimit, high + hlLimit, open)
     #   close = np.where(abs(close - low) >= hlLimit, low + hlLimit, close)

    hlLimit = 1
    high = data_recent["High"].copy()
    low = data_recent["Low"].copy()
    close = data_recent["Close"].copy()
    openp = data_recent["Open"].copy()

    def remove_outlier(data):
        outlier = data.iloc[-1]
        for i in range(1, len(data) - 1):  # Skip the first and last elements
            prevp = data.iloc[i - 1]
            currp = data.iloc[i ]
            nextp = data.iloc[i + 1]

    # Check if the current close price deviates significantly from its neighbors
            if abs(currp - prevp) >= hlLimit :
                # Replace the current close price with the average of its neighbors
                outlier = round(data.iloc[i], 2)
                data.iloc[i] = (prevp + nextp) / 2
            else:
                outlier = 0
        return outlier 

# Apply high/low adjustments
    if interval not in ["1m", "5m", "15m"]:
        high = np.where(high - np.maximum(close,openp) >= hlLimit, np.maximum(close,openp) + hlLimit, high)
        low = np.where(low - np.minimum(close,openp) <= -hlLimit, np.minimum(close,openp) - hlLimit, low)

# Further adjustments for lower timeframes
    if interval in ["1m", "5m", "15m"]:
        outlier_openp = remove_outlier(openp)
        outlier_close = remove_outlier(close)
        outlier_high = remove_outlier(high)
        outlier_low = remove_outlier(low)
        
        high = np.where(high - np.maximum(close,openp) >= hlLimit, np.maximum(close,openp) + hlLimit, high)
        low = np.where(low - np.minimum(close,openp) <= -hlLimit, np.minimum(close,openp) - hlLimit, low)

    # Loop through each data point and plot with different colors
    
    for i in range(1, len(x_values)):  # Start from index 1 to compare with the previous value
        fill = "gray" if close[i] < openp[i] else "none"  # Filled black if low decreases, empty otherwise
        edge_color = "blue"  if close[i] >= close[i-1] else "red" #Keep the edge black for all bars
        width = 0.5 #if close[i] >= close[i-1] else 1
        
        ax.bar(
            x_values[i],  # X-position
            abs(openp[i] - close[i]),  # Bar height (difference between high and low)
            bottom=min(close[i], openp[i]), # Start bar from the low price
            color=fill,  # Fill color
            edgecolor=edge_color,# Edge color to ensure visibility
            linewidth=width
        )
    # draw high in same loop
        ax.vlines(x_values[i], 
            max(openp[i], close[i]), ### expects lower
            high[i], 
            color="black", 
            linewidth=1
        )

    #draw low   
        ax.vlines(x_values[i], 
            low[i], 
            min(openp[i], close[i]), 
            color="black", 
            linewidth=1
        )
#### std
# Parameters
    moving_avg_type = "EMA"  # Can be "SMA", "EMA", or "WMA"
    window = 20  # Lookback period for moving average & standard deviation
    std_dev_factor = 2  # Multiplier for standard deviation
    #std_dev_factor3 = 3

# Function to calculate different types of moving averages
    def moving_average(series, window, method="SMA"):
        if method == "SMA":
            return series.rolling(window=window).mean()
        elif method == "EMA":
            return series.ewm(span=window, adjust=False).mean()
        elif method == "WMA":
            weights = np.arange(1, window + 1)
            return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        else:
            raise ValueError("Invalid moving average type. Choose SMA, EMA, or WMA.")

# Calculate the middle band
    data_recent["Middle_Band"] = moving_average(data_recent["Close"], window, method=moving_avg_type)

# Calculate standard deviation over the same window
    data_recent["Std_Dev"] = data_recent["Close"].rolling(window=window).std()

# Upper and Lower Bands
    data_recent["Upper_Band"] = data_recent["Middle_Band"] + (std_dev_factor * data_recent["Std_Dev"])
    data_recent["Lower_Band"] = data_recent["Middle_Band"] - (std_dev_factor * data_recent["Std_Dev"])
    #data_recent["Upper_Band3"] = data_recent["Middle_Band"] + (std_dev_factor3 * data_recent["Std_Dev"])
    #data_recent["Lower_Band3"] = data_recent["Middle_Band"] - (std_dev_factor3 * data_recent["Std_Dev"])

# Plot Standard Deviation 

    #ax.plot(x_values, data_recent["Close"], color="gray", linewidth=1, label="Close Price")
    #ax.plot(x_values, data_recent["Middle_Band"], color="blue", linewidth=2, label="Middle Band")
    ax.plot(x_values, data_recent["Upper_Band"], color="red", linewidth=2, linestyle="dashed", label="Upper Band")
    ax.plot(x_values, data_recent["Lower_Band"], color="green", linewidth=2, linestyle="dashed", label="Lower Band")

# Fill the area between the bands
    #ax.fill_between(x_values, data_recent["Lower_Band3"], data_recent["Upper_Band3"], color="gray", alpha=0.1)
    ax.fill_between(x_values, data_recent["Lower_Band"], data_recent["Upper_Band"], color="gray", alpha=0.2)

    #channel_length = 100
    dist = np.max(np.abs(y_pred_linear - close))#####################y
    
# Upper and lower channel lines
    upper_lr = y_pred_linear + dist
    lower_lr = y_pred_linear - dist
    mid =(upper_lr +lower_lr)/2

# Plot actual prices and regression lines
    ax.plot(x_values, upper_lr, color="blue", linestyle="--", linewidth=4,label="Upper Channel")
    ax.plot(x_values, lower_lr, color="blue", linestyle="--", linewidth=4,label="Lower Channel")
    ax.plot(x_values, mid, color="red", linestyle="--", linewidth=4,label="mid line")
    ax.fill_between(x_values, upper_lr, lower_lr, color="gray", alpha=0.15)

    ############### Draw horizontal lines from the lowest and highest points    
    min_price = np.min(close)#####################y
    max_price = np.max(close)#####################y
    Percent_from_top = round(100*(max_price - current_price)/max_price, 2) 
    Percent_from_bot = round(100*(current_price - min_price)/min_price, 2) 
    tb_message = f"dropped {Percent_from_top}" if y_pred_linear[-1] < y_pred_linear[-2] else f"increased {Percent_from_bot} "
    
    ax.axhline(y=min_price, color="red", linestyle="--", lw=5,label="")
    ax.axhline(y=max_price, color="red", linestyle="--", lw=5,label="")

    # Add price labels for the highest and lowest prices
    ax.text(x_values[-1], min_price, f'Low: {min_price:.2f}', color='green', fontsize=fontsize, verticalalignment='top')
    ax.text(x_values[-1], max_price, f'High: {max_price:.2f}', color='red', fontsize=fontsize, verticalalignment='bottom')

    # Draw gray line for current price
    ax.axhline(y=current_price, color="red", linestyle="--", linewidth=2, label="")

    # Modify the current price label to include the trend message and color
    current_price_label = f"-----{current_price:.2f} {trend_message.split()[-1]}"
    if trend_message == "Trend UP":
        current_price_color = "green"  # Green for UP trend
    elif trend_message == "Trend DOWN":
        current_price_color = "red"  # Red for DOWN trend
    else:
        current_price_color = "gray"  # Default color for NEUTRAL trend

    ax.text(x_values[-1], current_price, current_price_label, color=current_price_color, fontsize=fontsize, verticalalignment='top')

    # Draw gray line for previous close
    ax.axhline(y=previous_close, color="navy", linestyle="--", label="")

    # Add price label for the previous_price
    ax.text(0, previous_close, f'{previous_close:.2f}__c1', color='navy', fontsize=fontsize, verticalalignment='top')

###### add info block on top of chart
    tfAll = [1,5,15,30,60]
    tfFactor = [60,12,4,2,1]
    
    def find_days(maxmin="max"):
        days = 0
        xm = 0
        for i in range(1, len(x_values)):
            if maxmin == "max":
                if data_recent["Close"][i] == np.max(y):
                    xm = i
            else:
                if data_recent["Close"][i] == np.min(y):
                    xm = i
        if st.session_state.index <5 :
            days = round(((len(x_values) - xm)/tfFactor[st.session_state.index])/16,1) ##16h per day
        elif st.session_state.index >=5 :
            days = len(x_values) - xm
        return days

    days = 0 
    if y_pred_linear[-1] < y_pred_linear[-2]:
        days = find_days("max")
    else:
        days = find_days("min")

    dataDays = ""
    if st.session_state.index <5:
        nowDay = round((len(x_values)/tfFactor[st.session_state.index])/16,1)
        dataDays = f"days: {nowDay}"
    else:
        dataDays = ""
        
    #outlier_message = f"outliers: o_{outlier_openp}, c_{outlier_close}, h_{outlier_high}, l_{outlier_low}"
    
    ax.text(0.4, 0.9, f"interval: {interval}__Now: {current_price:.2f}\n{tb_message}% in {days} days\ndata_length: {len(x_values)}, {dataDays}", 
         horizontalalignment='left', verticalalignment='center', 
        transform=ax.transAxes, fontsize=20, color="blue")
    
    # Draw gray line for d2 close
    d2_close = fetch_d2_close(ticker)
    ax.axhline(y=d2_close, color="navy", linestyle="--", label="")
    ax.text(0, d2_close, f'{d2_close:.2f}__c2', color='navy', fontsize=fontsize, verticalalignment='top')
    
## channel middle horizontal。 
    middle = mid[-1]
    if st.session_state.marker_status:
        ax.axhline(y=middle, color="red", linestyle="--",label="")
        ax.text(0, middle, f'{middle:.2f}__Channel_mid', color='navy', fontsize=fontsize, verticalalignment='top')

    # Draw exponential moving averages with dashed lines
    ax.plot(x_values, data_recent['EMA_9'], color="red", linestyle="--", label="EMA 9/20_blue")
    ax.plot(x_values, data_recent['EMA_20'], color="blue", lw=5,linestyle="-", label="")
    #ax.plot(x_values, data_recent['EMA_50'], color="gold", linestyle="--", label="EMA 50")
    #ax.plot(x_values, data_recent['EMA_100'], color="gray", linestyle="--", label="EMA 100")
    ax.plot(x_values, data_recent['EMA_200'], color="purple", linestyle="--", label="EMA 200")

##### ######### #######. marker line
    if st.session_state.marker_status and ((interval in ( "1m", "5m") and abs(current_price - marker_position) <= 2) or (interval not in ( "1m" , "5m", "1y") and abs(current_price - marker_position) <= 5) or interval == "1y"):
        ax.axhline(y=marker_position, color=markerColor, lw= marker_width ,linestyle=style)
        ax.axhline(y=marker_position - separationLen, color=markerColor, lw= marker_width ,linestyle=style)
    #ax.axhline(y=marker_position2, color='r', lw=5, linestyle='--')
    
########### drag lines
    # Use Streamlit sliders to adjust line positions
    h_line_pos = st.sidebar.slider('Horizontal Line Position', min_price , max_price ,(min_price +max_price )/2)
   # v_line_pos = st.sidebar.slider('Vertical Line Position', len(x_values)/2 , float(len(x_values)), (len(x_values)/2 + len(x_values)/2))

# Add lines based on slider values
    if st.session_state.marker_status:
        ax.axhline(y=h_line_pos, color='black', lw=marker_width, linestyle='--')
        ax.text(0, h_line_pos, f'{h_line_pos:.2f}_TB_mid', color='navy', fontsize=fontsize, verticalalignment='top')
    #ax.axvline(x=v_line_pos, color='b', lw=2, linestyle='--')
    
# Set the background color of the axes to light blue
    ax.set_facecolor(bgcolor)

    # Add price labels for EMAs
    ax.text(x_values[-1], data_recent['EMA_9'].iloc[-1], f'^^^^^^e9', color='orange', verticalalignment='top')
    ax.text(x_values[-1], data_recent['EMA_20'].iloc[-1], f'^^^^^^^^e20', color='blue', verticalalignment='top')
    #ax.text(x_values[-1], data_recent['EMA_50'].iloc[-1], f'^^^^^^^^e50', color='gold', verticalalignment='top')
    #ax.text(x_values[-1], data_recent['EMA_100'].iloc[-1], f'^^^^^^^^e100', color='gray', verticalalignment='top')
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
    ax.legend()

    ax.set_title(
        f"{current_date} Linear and Polynomial Regression for {ticker} (tFrame: {interval})",
        fontsize=20,          # Font size
        fontweight='bold',    # Bold text
        color='navy',         # Text color
        fontfamily='sans-serif'  # Font family (e.g., 'serif', 'monospace')
    )
##########. try cloud for today
    if st.session_state.index <5 and st.session_state.index >0:
        time_frame = tfAll[st.session_state.index]  # Change this to 1, 5, 15, etc. (minutes per data point)

# Simulated data covering from 20:00 (yesterday) to now
        start_time = datetime.now(pytz.timezone("US/Eastern")).replace(hour=4, minute=0, second=0, microsecond=0) 
        now_time = datetime.now(pytz.timezone("US/Eastern"))

# Compute total minutes from start_time to now
        total_minutes = int((now_time - start_time).total_seconds() // 60)
        dataNum= (total_minutes // time_frame)  # Simulated data
        indexNum = (dataNum-1)
    
        start_idx = x_values[-indexNum]
        end_idx = len(data_recent) - 1  # If now is beyond the last data point

        ax.axvspan(start_idx, end_idx, color='gray', alpha=0.2, label='4 AM to Now')

    # redeclare, messed up by above
    x_values = np.arange(len(data_recent))  # Numeric x-axis
    # --- RSI Plot ---
    ax2.fill_between(x_values, 0, 100, color="gray", alpha=0.15)

    ax2.plot(x_values, data_recent['RSI'], color="navy", linestyle="-", label="RSI (14)")
    ax2.plot(x_values, data_recent['RSI2'], color="red", linestyle="--", label="RSI (25)")
    ax2.set_facecolor(bgcolor)

    ax2.axhline(y=70, color="red", linestyle="--")
    ax2.axhline(y=30, color="green", linestyle="--")
    ax2.axhline(y=50, color="gray", linestyle="--")
    ax2.set_title(f"{current_date} RSI ({interval})",
        fontsize=20,          # Font size
        fontweight='bold',    # Bold text
        color='navy',         # Text color
        fontfamily='sans-serif')
    ax2.legend()

    # === MACD Plot (Only If Timeframe Is Valid) ===
    if interval in valid_macd_timeframes:
        # Create a sequence for the x-axis from 1 to len(data_recent)
        x_values = range(1, len(data_recent) + 1)

        # Plot the MACD and Signal lines with numeric x-values
        ax3.fill_between(x_values, min(data_recent['MACD']), max(data_recent['MACD']), color="gray", alpha=0.15)
        
        ax3.plot(x_values, data_recent['MACD'], color="navy", label="MACD Line")
        ax3.plot(x_values, data_recent['Signal_Line'], color="red", linestyle="--", label="Signal Line")
        
        # Histogram Bars (Green for Positive, Red for Negative)
        histogram_values = data_recent['MACD'] - data_recent['Signal_Line']
        ax3.bar(x_values, histogram_values, color=['green' if val > 0 else 'red' for val in histogram_values], alpha=0.5)
        ax3.set_facecolor(bgcolor)
        
        ax3.set_title(f"{current_date} MACD ({interval})", 
            fontsize=20,          # Font size
            fontweight='bold',    # Bold text
            color='navy',         # Text color
            fontfamily='sans-serif')
        ax3.legend()

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readabil
    st.pyplot(fig)  ## finally plot all 3 figures

    # === Add Download Button ===
# Save the figure to a BytesIO buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")  # Save as PNG (you can also use "pdf" or other formats)
    buffer.seek(0)  # Move the buffer's pointer to the beginning

# Create a download button
    st.download_button(
        label="Download Plot as PNG",
        data=buffer,
        file_name="plot.png",  # Name of the downloaded file
        mime="image/png"       # MIME type of the file
    )

    st.write("---------------------")
    #st.write(data_recent.tail(5))

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
        dev_from_std = round(delta/std_dev,0)

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

        df = pd.DataFrame(columns=['tFrame', 'ema_trend', 'e100/200', 'pr_eAvg','rsi','macd', 'score', 'score_trend'])

        # Save the empty DataFrame to the CSV file
        df.to_csv(file_path, index=False, header=False)
        st.success(f"✅ File created successfully as `{file_path}`")

    #ema_score, ema_trend, rsi_score, macd_score, score
    y_pred_p_trend = 0
    if y_pred_poly >= y_pred_poly1:
        y_pred_p_trend = 1
        #y_pred_p_trend = "↑"
    else:
       # y_pred_p_trend = "↓"
        y_pred_p_trend = -1
        
    if ema_score >= 0 and rsi_score >= 0 and macd_score >= 0 and ema_trend >= 0:
        score_trend = 1
    elif ema_score <0 and rsi_score < 0  and macd_score < 0 and ema_trend < 0:
        score_trend = -1
    else:
        score_trend = 0
        
    if current_price >= data_recent["EMA_20"].iloc[-1] :
        pr_eAvg = 1
    else:
        pr_eAvg = -1
    # total == score (above) 
    new_data = pd.DataFrame([{
        "tFrame": f"{interval}",
        "ema9/20": round(ema_trend, 2),
        "e100/200": round(ema_score, 2),
        "pr_E20": pr_eAvg,
        "rsi": round(rsi_score, 2),
        "macd": round(macd_score, 2),
        "score": (score + pr_eAvg),
        "score_trend": score_trend,
    }])
    #new_data.to_csv(scoreT_file, mode="a", header=False, index=False)
    new_data.to_csv(scoreT_file, mode="a", header=False, index=False, float_format="%.2f") ## chatGPT

    # Read the updated CSV file
    df = pd.read_csv(file_path, header=None)

    # Define the custom order for the first column
    custom_order = ["1m", "5m", "15m", "30m", "1h", "3mo", "6mo","1y"]

    # Convert the first column to a categorical type with the custom order
    df[0] = pd.Categorical(df[0], categories=custom_order, ordered=True)

    # Sort the DataFrame by the first column using the custom order
    df = df.sort_values(by=0)

    #add column names
    df.columns = ['tFrame', 'ema9/20', 'e100/200', 'pr_E20', 'rsi', 'macd', 'score',  'score_trend']
    df1 = df[['tFrame', 'ema9/20',  'pr_E20', 'rsi',   'score_trend']]
    #display table
    st.dataframe(df1, hide_index=True) #original table looks neater

    plot_bars(current_price)

    ################### all control buttons ###########################################################
    current_price = round(data_recent['Close'].iloc[-1], 2)
    now = datetime.now(eastern).strftime('%m-%d %I:%M:%S %p')  # Correct format
  #  ema_trend_1m = df[df["tFrame"] == "1m"]["ema_trend"].values[0]
 #   ema_trend_5m = df[df["tFrame"] == "5m"]["ema_trend"].values[0]
    pr1=df[df["tFrame"] == "1m"]["e100/200"].values[0]
    pr5=df[df["tFrame"] == "5m"]["e100/200"].values[0]
    
    st.write(f"interval: {interval}__rerun:{ st.session_state.rerun_count}")
    # Extract "score_trend" for "1m"  ## 
    
    if pr1 ==1:
        message = "___B OK 1"
        color = "green"
    elif pr1==-1:
        message = "___S OK -1"
        color = "red"
    else:
        message = "Hold it"
        color = "orange"
    st.markdown(f'<p style="color:{color}; font-weight:bold;">e100/200 1min: {message}</s></p>', unsafe_allow_html=True)
    if pr5 ==1:
        message = "___B OK 1"
        color = "green"
    elif pr5==-1:
        message = "___S OK -1"
        color = "red"
    else:
        message = "Hold it"
        color = "orange"
    st.markdown(f'<p style="color:{color}; font-weight:bold;">e100/200 5min: {message}</s></p>', unsafe_allow_html=True)

    sum_score_trend_rest = df[~df["tFrame"].isin(["1m", "6mo"])]["score_trend"].sum()
    
    if sum_score_trend_rest >=4:
        message = "___B OK >=4"
        color = "green"
    elif sum_score_trend_rest <= -4:
        message = "___S OK <=-4"
        color = "red"
    else:
        message = "Hold it"
        color = "orange"
    #st.write(f"score_trend_others: ||... {sum_score_trend_rest} ___ {message}")
    st.markdown(f'<p style="color:{color}; font-weight:bold;">score_trend_others: {message}__{sum_score_trend_rest}</s></p>', unsafe_allow_html=True)

    #display message about app status
    sleep_status = 'on' if st.session_state.stop_sleep == 0 else "off"
    
    col1, col2, col3, col4,col5 = st.columns(5)
    with col1:
        # delete data button
        if st.button(" 1min"):
            #st.session_state.rerun_count = 0
            st.session_state.index = 0
           # st.session_state.stop_sleep = 0
            #st.session_state.sbOK = 1
            st.session_state.sleepGap = 6 # !=5: will keep 1min
            st. rerun()

    with col2:
        # delete data button
        if st.button(" 1m 5m"):
            #st.session_state.rerun_count = 0
            
            st.session_state.index = 0
            #st.session_state.sbOK = 1
            st.session_state.sleepGap = 7 # !=5 or 6 will keep 5min
            st. rerun()
            
    with col3:
        if st.button("check all"):
            
            st.session_state.index = 0
            st.session_state.rerun_count = 0
            st.session_state.sleepGap = 5
            st.rerun()

    with col4:
        if st.button("stop slp"):
            st.session_state.stop_sleep =1
            st.session_state.index = 1
            st.rerun()

    with col5:
        if st.button(f"slp: {st.session_state.sleepGap}_stop:{st.session_state.stop_sleep}"):
        #st.write(f"slp: {st.session_state.sleepGap}_stop:{st.session_state.stop_sleep}")
            st.rerun()
########################################
    if st.session_state.stop_sleep == 0:
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
        
        ##p
        #st.write(f"B: {b_condition}__SS: {short_s}__S: {s_condition}__SB:{short_b}__Status_0: {SB == "AAA" or SB == "S" or SB == "SB"}__interval: {intervals[st.session_state.index]}")
       # st.empty()
        st.rerun()


        

if __name__ == "__main__":
    main()
