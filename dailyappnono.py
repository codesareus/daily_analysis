### daily_analysis.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pytz import timezone
import pytz
import time
import os
import pandas_market_calendars as mcal

# --------------------------
# Configuration & Constants
# --------------------------
EASTERN = timezone('US/Eastern')
INTERVALS = ['1m', '5m', '15m', '30m', '1h', '3mo', '6mo']
BACKTRACK_OPTIONS = [0, 2, 5, 7, 10, 20, 30, 45, 60, 90, 100, 120]

# --------------------------
# Session State Initialization
# --------------------------
def init_session_state():
    defaults = {
        'index': 0,
        'rerun_count': 0,
        'stop_sleep': 0,
        'temp_price': 0,
        'sb_status': 0,
        'sleepGap': 5,
        'poly_degree': 4,
        'setpr': 0.0,
        'settype': 'zz'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --------------------------
# Data Fetching Utilities
# --------------------------
class DataFetcher:
    @staticmethod
    def fetch_stock_data(ticker, interval="5m", period="5d", prepost=True):
        return yf.Ticker(ticker).history(period=period, interval=interval, prepost=prepost)
    
    @staticmethod
    def fetch_historical_data(ticker, period):
        return yf.Ticker(ticker).history(period=period)

# --------------------------
# Technical Indicators
# --------------------------
class StockAnalyzer:
    @staticmethod
    def calculate_rsi(data, window1=14, window2=25):
        delta = data['Close'].diff(1)
        
        for window in [window1, window2]:
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)
            data[f'RSI{window}' if window != window1 else 'RSI'] = 100 - (100 / (1 + rs))
        return data

    @staticmethod
    def calculate_macd(data):
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        return data

    @staticmethod
    def calculate_emas(data):
        for span in [9, 20, 50, 100, 200]:
            data[f'EMA_{span}'] = data['Close'].ewm(span=span, adjust=False).mean()
        return data

# --------------------------
# Regression Analysis
# --------------------------
class RegressionModel:
    @staticmethod
    def perform_regression(data, degree=1, points=300):
        data_recent = data.tail(points)
        X = np.arange(len(data_recent)).reshape(-1, 1)
        y = data_recent['Close'].values
        
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        
        return {
            'X': X,
            'y': y,
            'y_pred': model.predict(X_poly),
            'r2': r2_score(y, model.predict(X_poly)),
            'data': data_recent
        }

# --------------------------
# UI Components
# --------------------------
class UIComponents:
    @staticmethod
    def create_interval_buttons():
        cols = st.columns(7)
        intervals = ['1m', '5m', '15m', '30m', '1h', '3mo', '6mo']
        with cols[0]: st.button("1min", key="1m")
        with cols[1]: st.button("5min", key="5m")
        # Add remaining interval buttons...

    @staticmethod
    def create_degree_controls():
        cols = st.columns(4)
        degrees = {'3': 3, '5': 5, '6': 6, '7': 7}
        for col, (label, degree) in zip(cols, degrees.items()):
            with col:
                if st.button(f"degree {label}"):
                    st.session_state.poly_degree = degree
                    st.rerun()

    @staticmethod
    def price_change_header(ticker, current_price, previous_close):
        change = current_price - previous_close
        percentage = (change / previous_close) * 100
        current_time = datetime.now(EASTERN).strftime("%I:%M:%S %p")
        
        if percentage >= 0:
            st.success(f"🟢 {ticker}: {current_price:.2f} ({percentage:+.2f}%) | {current_time}")
        else:
            st.error(f"🔴 {ticker}: {current_price:.2f} ({percentage:+.2f}%) | {current_time}")

# --------------------------
# Main Application
# --------------------------
def main():
    init_session_state()
    st.title("Score Regression Analysis")
    
    # User Inputs
    ticker = st.text_input("Enter Stock Ticker:", value="SPY").upper()
    selected_backtrack = st.slider("Backtrack Points:", options=BACKTRACK_OPTIONS)
    
    # Data Loading
    data = DataFetcher.fetch_stock_data(ticker, interval=INTERVALS[st.session_state.index])
    if data.empty:
        st.error("Failed to fetch data")
        return
    
    # Data Processing
    data_recent = data.tail(300 + selected_backtrack).head(300)
    data_recent = StockAnalyzer.calculate_emas(data_recent)
    data_recent = StockAnalyzer.calculate_rsi(data_recent)
    data_recent = StockAnalyzer.calculate_macd(data_recent)
    
    # Regression Analysis
    regression_results = RegressionModel.perform_regression(
        data_recent, 
        degree=st.session_state.poly_degree
    )
    
    # UI Rendering
    UIComponents.price_change_header(ticker, data_recent['Close'].iloc[-1], fetch_previous_close(ticker))
    UIComponents.create_degree_controls()
    UIComponents.create_interval_buttons()
    
    # Visualization
    plot_regression_results(regression_results)
    plot_technical_indicators(data_recent)
    
    # Auto-refresh Logic
    handle_auto_refresh()

def handle_auto_refresh():
    time.sleep(st.session_state.sleepGap)
    if st.session_state.sleepGap == 5:
        st.session_state.index = (st.session_state.index + 1) % len(INTERVALS)
        st.session_state.rerun_count += 1
    st.rerun()

if __name__ == "__main__":
    main()
