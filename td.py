### daily analysis 03-03-25

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

#eastern = pytz.timezone("America/New")
eastern = pytz.timezone("US/Eastern")

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



        
# Streamlit app
def main():
    st.title("PaperTD ")

    #######################
   
    # Initialize temp_price state
    if "temp_price" not in st.session_state:
        st.session_state.temp_price = 0

    # Initialize sb_status state
    if "sb_status" not in st.session_state:
        st.session_state.sb_status = 0
  
    pe_file = f"pe.csv"

    
    
   # Get current local time
    eastern = pytz.timezone("US/Eastern")
    #current_time = datetime.now(eastern).strftime("%H:%M:%S")
    now = datetime.now(eastern).strftime("%I:%M:%S %p")
    current_time = datetime.now(eastern).strftime("%m-%d %H:%M:%S")
    
    ########## B and S actions
    def save_pe(type="AAA", price=None, total =0, note="zzz"): 
        updated_data = pd.read_csv(pe_file, names=["type", "B_pr", "S_pr", "pl", "total", "temp_pr", "note"])
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
                    "TimeStamp": f"{current_time}",
                    "type": "B",
                    "B_pr": round(price, 2),
                    "S_pr": 0,
                    "pl": pl,
                    "total": round(total, 2),
                    "temp_price": round(price, 2),
                    "note": note
                }])

        elif type == "S":
            new_data = pd.DataFrame([{
                    "TimeStamp": f"{current_time}",
                    "type": "S",
                    "B_pr": 0,
                    "S_pr": round(price, 2),
                    "pl": round(pl, 2),
                    "total": round(total, 2),
                    "temp_price": 0,
                    "note": note
                }])

        elif type == "SS":
            new_data = pd.DataFrame([{
                    "TimeStamp": f"{current_time}",
                    "type": "SS",
                    "B_pr": 0,
                    "S_pr": round(price, 2),
                    "pl": 0,
                    "total": total, ## for now
                    "temp_price": round(price, 2),
                    "note": note
                }])

        elif type == "SB": 
            new_data = pd.DataFrame([{
                    "TimeStamp": f"{current_time}",
                    "type": "SB",
                    "B_pr": round(price, 2),
                    "S_pr": 0,
                    "pl": round(pl, 2),
                    "total": round(total, 2),
                    "temp_price": 0,
                    "note": note
                }])
        # Append to CSV file
        new_data.to_csv(pe_file, mode="a", header=False, index=False)
            
   
# Initialize the session state variable if it doesn't exist
    if "setpr" not in st.session_state:
        st.session_state.setpr = 0
        
    if "setnote" not in st.session_state:
        st.session_state.setnote = "zzz"
# Create a text input that displays the current session state value
    col1, col2 = st.columns(2)
    with col1:
        setpr_input = st.text_input("Enter set pr: ", value=str(st.session_state.setpr))
    with col2:
        setnote_input = st.text_input("Enter note: ", value=str(st.session_state.setnote))
    #set them
    col1, col2=st.columns(2)
    with col1:
        if st.button("set"):
            try:
    # Attempt to convert the input to a float and update the session state
                st.session_state.setpr = float(setpr_input)
                st.session_state.setnote = str(setnote_input)
                st.session_state.confirmation_message = f"Success!"
            except ValueError:
    # Handle invalid input (non-numeric values)
                st.error("Please enter a valid number or type")
            st.rerun()
# Display the current value of setpr from the session state
    with col2:
        st.write(f"setpr: {st.session_state.setpr}__setnote:{st.session_state.setnote}")

    updated_data = pd.read_csv(pe_file, names=["type", "B_pr", "S_pr", "pl", "total", "temp_pr", "note"])
    
    col1, col2, col3, col4 = st.columns(4)
    total = updated_data["total"].iloc[-1]
    SB = updated_data["type"].iloc[-1]
    with col1:
        if st.button("B >>"):
            if  (SB == "AAA" or SB == "S" or SB== "SB") and st.session_state.setpr != 0:
                save_pe("B", st.session_state.setpr, total,st.session_state.setnote)
                st.session_state.setpr = 0
                st.write("Success B!")
                st.rerun()
            
    with col2:
        if st.button("S >>"):
            if  SB == "B" and st.session_state.setpr != 0:
                save_pe("S", st.session_state.setpr, total, st.session_state.setnote)
                st.session_state.setpr = 0
                st.write("Success S!")
                st.rerun()


    with col3:
        if st.button("Sh_S >>"):
            if (SB == "AAA" or SB == "S" or SB== "SB") and st.session_state.setpr != 0:
                save_pe("SS", st.session_state.setpr, total, st.session_state.setnote)
                st.session_state.setpr = 0
                st.write("Success Sh_S!")
                st.rerun()
                
    with col4:
        if st.button("Sh_B >>"):
            if SB == "SS" and st.session_state.setpr != 0:
                save_pe("SB", st.session_state.setpr, total, st.session_state.setnote)
                st.session_state.setpr = 0
                st.rerun()
            

    st.write(f"Price_set? __ {st.session_state.setpr != 0}")
    st.write(f"SB_type: {updated_data["type"].iloc[-1]}")
    #show which timeframes are in bar chart:

    #display pe_table
    # Read the updated CSV file ---- example
    updated_data = pd.read_csv(pe_file, names=["type", "B_pr", "S_pr", "pl", "total", "temp_pr", "note"])

    st.markdown(f'<p style="color:orange; font-weight:bold;">pe_table: </s></p>', unsafe_allow_html=True)
    
    st.dataframe(updated_data.tail(5), hide_index=False)
    st.write(f"{len(updated_data["total"])} rows")
    #with col2:

       # st.write(":::::::::::::::::::::::::::::::::::::::::::::::::::")
    st.write(f"now: _<{now}>_{get_time_now()}")
   
    #st.write(f"Pre_Post_status: {st.session_state.prepo}")
    if st.button("Clear data"):
        #st.session_state.stop_sleep = 1
        new_data = pd.DataFrame([{
                    "TimeStamp": f"{current_time}",
                    "type": "AAA",
                    "B_pr": 0,
                    "S_pr": 0,
                    "pl": 0,
                    "total": 0, 
                    "temp_pr": 0,
                    "note":"zzz"
                }])
                # clear CSV file
        new_data.to_csv(pe_file, mode="w", header=False, index=False)
        st.write("data cleared")
        st.rerun()
            
    st.write("---------------------")



if __name__ == "__main__":
    main()
