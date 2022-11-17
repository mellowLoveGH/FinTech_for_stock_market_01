#!pip install yfinance
import yfinance as yf # https://pypi.org/project/yfinance/
import math
### the meaning of prediction about stock market
from numpy.core.numeric import ones_like
import random
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# get data by ticker-name, start-time & end-time
def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09"):
  df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
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

# draw line chart, x:date, y:close point
def line_date_close(df_data, tn):
    x_axis = list( df_data.index )
    y_axis = list( df_data['Close'] ) # Close_return

    fig = plt.subplots(figsize =(12, 6))
    plt.plot(x_axis, y_axis, label = tn) # , color ='r'

    plt.title("Stock: " + tn)
    plt.xlabel('Time')
    plt.ylabel("Close points")

    plt.legend()
    plt.show()
    return 

# 
def binning_frequency(daily_returns, bin_width=0.01):
    tmp_list = sorted(daily_returns)
    mn, mx = tmp_list[0], tmp_list[-1]
    bin_dic = {} # 
    tmp_scale = int( 1/bin_width )
    for v in tmp_list:
        if math.isnan(v):
            continue
        v1 = round(v * tmp_scale)
        if v1 not in bin_dic:
            bin_dic[v1] = 1
        else:
            bin_dic[v1] += 1
    return bin_dic

# draw the distribution of close return, for example, daily close return
def distribution_close_return(df_data, tn):
    date_list = list(df_data.index)
    start_date, end_date = date_list[0], date_list[-1]
    start_date, end_date = str(start_date)[:10], str(end_date)[:10]
    daily_returns = list(df_data['Close_return'])
    x_axis, y_axis = [], []
    bin_width = 0.005
    bin_dic = binning_frequency(daily_returns, bin_width)
    for k, v in bin_dic.items():
        x_axis.append(k*bin_width * 100)
        y_axis.append(v)
    fig = plt.subplots(figsize =(12, 6))
    #plt.plot(x_axis, y_axis, label = tn) # , color ='r'
    plt.bar(x_axis, y_axis, width = 0.4, label = tn)
    #plt.xticks(x_axis)
    plt.yticks(y_axis)    
    plt.title("Stock: " + tn + ", from " + start_date + " to " + end_date)
    plt.xlabel('daily return %')
    plt.ylabel("frequency / business days")
    plt.legend()
    plt.show()
    return 

# 0005.HK:滙豐控股, 1299.HK:友邦保險, 0700.HK:騰訊控股, 9988.HK:阿⾥巴巴, 3690.HK:美團
hk_tickers = ["3690.HK", "9618.HK", "1024.HK", "9866.HK"]
tn = "3690.HK" # AAPL
st, et = "2022-01-01", "2022-12-31"
df_data = get_df_data(ticker_name=tn, start_time=st, end_time=et)
df_data = calculate_daily_return(df_data)
df_data = get_market_movement(df_data)
# rise-days vs fall-days
pos_df = df_data[df_data['Close_return']>0]
neg_df = df_data[df_data['Close_return']<0]
print( "business days: ", len(df_data) )
print( "rise vs fall: ", len(pos_df), len(neg_df) )
# draw 2 graphs
line_date_close(df_data, tn)
distribution_close_return(df_data, tn)
