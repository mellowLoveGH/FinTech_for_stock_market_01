#!pip install yfinance
import yfinance as yf # https://pypi.org/project/yfinance/
import math
### the meaning of prediction about stock market
from numpy.core.numeric import ones_like
import random
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd

# get data by ticker-name, start-time & end-time
def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09"):
  df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
  #df_data.head()
  return df_data

# get data by ticker-name, start-time & end-time & interval
def get_df_data_update(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09", interval_len='1d'):
  df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time, interval=interval_len) 
  #df_data.head()
  return df_data

# calculate the daily return by (current_index - previous_index) / previous_index
def calculate_daily_return(df_data, OHLC_index="Close"):
  name1 = OHLC_index+"_previous"
  df_data[name1] = df_data[OHLC_index].shift(1)
  name2 = OHLC_index+"_delta"
  df_data[name2] = df_data[OHLC_index] - df_data[name1]
  name3 = OHLC_index+"_return"
  df_data[name3] = df_data[name2] / df_data[name1]
  del df_data[name1]
  del df_data[name2]
  new_feature = name3
  return df_data #, new_feature

# get the market movement (yesterday -> today) based on daily return, 
  # 1 means rise and 0 fall
def get_market_movement(df_data, signal_name="Close_return"):
  df_data["movement"] = df_data[signal_name]>0
  df_data["movement"] = df_data["movement"].astype(int)
  return df_data

# convert the time to be string type: yyyy-mm-dd
def get_ymt_date(df_data):
  df_data["ymd_time"] = df_data.index
  df_data["ymd_time"] = df_data["ymd_time"].astype(str)
  df_data["hour_minute"] = df_data["ymd_time"].str.slice(11,16)
  df_data["ymd_time"] = df_data["ymd_time"].str.slice(0,10)
  # month
  df_data["monthNumber"] = df_data["ymd_time"].str.slice(5,7)
  df_data["monthNumber"] = df_data["monthNumber"].astype(int)
  # week
  df_data['ymd_time'] = pd.to_datetime(df_data['ymd_time'], errors ='coerce')
  df_data['weekNumber'] = df_data['ymd_time'].dt.isocalendar().week
  return df_data

# 
def get_stock_info(tn, st, et):
    df_data1 = get_df_data(ticker_name=tn, start_time=st, end_time=et)
    df_data1 = calculate_daily_return(df_data1)
    #df_data1 = calculate_daily_return_update(df_data1)
    df_data1 = get_market_movement(df_data1)
    df_data1 = get_ymt_date(df_data1)
    #
    df_data1["ymd_time"] = df_data1.index
    df_data1["ymd_time"] = df_data1["ymd_time"].astype(str)
    df_data1["ymd_time"] = df_data1["ymd_time"].str.slice(0,10)
    return df_data1

# calculate percentage_for_list
def percentage_for_list(tmp_list):
    s = sum(tmp_list)
    if s<=0:
        return []
    ll = []
    for it in tmp_list:
        ll.append( it/s )
    return ll

#
def rise_fall_basic(movement1, movement2, verbose=False):
    assert len(movement1) == len(movement2)
    rise_rise, rise_fall, fall_rise, fall_fall = 0, 0, 0, 0
    for i in range(len(movement1)):
        m1, m2 = movement1[i], movement2[i]
        if m1 == 1 and m2 == 1:
            rise_rise += 1
        elif m1 == 1 and m2 == 0:
            rise_fall += 1
        elif m1 == 0 and m2 == 1:
            fall_rise += 1
        elif m1 == 0 and m2 == 0:
            fall_fall += 1     
    ll = percentage_for_list([rise_rise, rise_fall, fall_rise, fall_fall])
    dic = {}
    dic["rise_rise"] = round(ll[0]*100, 2)
    dic["rise_fall"] = round(ll[1]*100, 2)
    dic["fall_rise"] = round(ll[2]*100, 2)
    dic["fall_fall"] = round(ll[3]*100, 2)
    #
    if verbose:
      r1 = round(rise_rise/(rise_rise+rise_fall)*100, 2)
      r2 = round(rise_fall/(rise_rise+rise_fall)*100, 2)
      msg = "if the first rises, then the second will rise: " + str(r1) + "% & fall: " + str(r2) + "%"
      print(msg)
      r1 = round(fall_rise/(fall_rise+fall_fall)*100, 2)
      r2 = round(fall_fall/(fall_rise+fall_fall)*100, 2)
      msg = "if the first falls, then the second will rise: " + str(r1) + "% & fall: " + str(r2) + "%"
      print(msg)
    return dic

def previous_date(current, days_num=1):
    current = datetime.datetime.strptime(current, "%Y-%m-%d").date() # %H:%M:%S
    previous = current - datetime.timedelta(days=days_num)
    return str(previous)

def next_date(current, days_num=1):
    current = datetime.datetime.strptime(current, "%Y-%m-%d").date() # %H:%M:%S
    next_date = current + datetime.timedelta(days=days_num)
    return str(next_date)

def UST_next_HKT(ust, hkt_list):
    days_num = 1
    while days_num<10:
        hkt = next_date(ust, days_num)
        if hkt in hkt_list:
            return hkt
        days_num += 1
    return ust

def HKT_previous_UST(hkt, ust_list):
    days_num = 1
    while days_num<10:
        ust = previous_date(hkt, days_num)
        if ust in ust_list:
            return ust
        days_num += 1
    return hkt

def same_market_stocks(df_data1, df_data2):
    movement1 = list(df_data1['movement'])
    movement2 = list(df_data2['movement'])
    return movement1, movement2

# suppose the first stock is from US, the second is from HK
def different_market_stocks01(df_data_us, df_data_hk, verbose=False):
    movement_us, movement_hk = [], []
    hkt_list = list(df_data_hk['ymd_time'])
    i = 0
    while i<len(df_data_us): # 
        us_stock_row = df_data_us.iloc[i]
        ust = us_stock_row['ymd_time']
        us_close = us_stock_row['Close_return']
        hkt = UST_next_HKT(ust, hkt_list)
        i += 1
        if ust == hkt:
            if verbose:
              print(ust, "not found corresponding HKT: ", hkt)
            continue
        tmp_df = df_data_hk[df_data_hk['ymd_time']==hkt]
        assert len(tmp_df)==1
        hk_close = tmp_df.iloc[-1]['Close_return']
        #
        if us_close>0:
            movement_us.append(1)
        else:
            movement_us.append(0)
        if hk_close>0:
            movement_hk.append(1)
        else:
            movement_hk.append(0)
    return movement_us, movement_hk

# suppose the first stock is from HK, the second is from US
def different_market_stocks02(df_data_hk, df_data_us, verbose=False):
    movement_hk, movement_us = [], []
    ust_list = list(df_data_us['ymd_time'])
    i = 0
    while i<len(df_data_hk): # 
        hk_stock_row = df_data_hk.iloc[i]
        hkt = hk_stock_row['ymd_time']
        hk_close = hk_stock_row['Close_return']
        ust = HKT_previous_UST(hkt, ust_list)
        i += 1
        if hkt == ust:
            if verbose:
              print(hkt, "not found corresponding UST: ", ust)
            continue
        tmp_df = df_data_us[df_data_us['ymd_time']==ust]
        assert len(tmp_df)==1
        us_close = tmp_df.iloc[-1]['Close_return']
        #print(hkt, hk_close, ust, us_close)
        if hk_close>0:
            movement_hk.append(1)
        else:
            movement_hk.append(0)
        if us_close>0:
            movement_us.append(1)
        else:
            movement_us.append(0)        
    return movement_hk, movement_us



###
st, et = "2022-01-01", "2022-12-31"

us_stock_ticker = ["AAPL", "GOOGL", "AMZN", "MSFT", "BABA", "PDD", "JD", "NTES", "BIDU", "NIO", "TCOM", "LI", "ZTO", "TME", "XPEV"]
hk_stock_ticker = ["9988.HK", "3690.HK", "0700.HK", "9618.HK", "0981.HK", "9999.HK", "1810.HK", "1024.HK", "9888.HK", "2015.HK", "9866.HK"]

verbose = False
corelation_list1, corelation_list2 = [], []
for tn1 in hk_stock_ticker:
    df_data1 = get_stock_info(tn1, st, et)
    for tn2 in us_stock_ticker:
        df_data2 = get_stock_info(tn2, st, et)
        # US market has influence over HK market
        movement_us, movement_hk = different_market_stocks01( df_data2, df_data1, verbose )
        info = rise_fall_basic(movement_us, movement_hk, verbose)
        corelation = info["rise_rise"] + info["fall_fall"]
        #print(info, corelation)
        corelation_list1.append( (corelation, tn2, tn1) )

        # HK market has influence over US market
        movement_hk, movement_us = different_market_stocks02( df_data1, df_data2, verbose )
        info = rise_fall_basic(movement_hk, movement_us, verbose)
        corelation = info["rise_rise"] + info["fall_fall"]
        #print(info, corelation)
        corelation_list2.append( (corelation, tn1, tn2) )

#
print("US market has influence over HK market: ")
for it in sorted(corelation_list1)[-10:]:
  print("\t", it)

print("HK market has influence over US market: ")
for it in sorted(corelation_list2)[-10:]:
  print("\t", it)





