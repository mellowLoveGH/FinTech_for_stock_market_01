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
  # get month number & week number
def get_ymt_date(df_data):
  df_data["ymd_time"] = df_data.index
  df_data["ymd_time"] = df_data["ymd_time"].astype(str)
  df_data["ymd_time"] = df_data["ymd_time"].str.slice(0,10)
  # month
  df_data["monthNumber"] = df_data["ymd_time"].str.slice(5,7)
  df_data["monthNumber"] = df_data["monthNumber"].astype(int)
  # week
  df_data['ymd_time'] = pd.to_datetime(df_data['ymd_time'], errors ='coerce')
  df_data['weekNumber'] = df_data['ymd_time'].dt.isocalendar().week
  return df_data

# calculate the monthly return
def month_change(df_data, feature="Close"):
    x, y = [], []
    months = set( list(df_data['monthNumber']) )
    for m in list(months)[:]:
        tmp_df = df_data[ df_data['monthNumber']==m ]
        tmp_list = tmp_df[feature]
        m_start, m_end = tmp_list[0], tmp_list[-1]
        change = (m_end - m_start)/m_start * 100
        x.append(m)
        y.append(change)
        #print(m_start, m_end)
    return x, y

# calculate the weekly return
def week_change(df_data, feature="Close"):
    x, y = [], []
    weeks = set( list(df_data['weekNumber']) )
    for w in list(weeks)[:]:
        tmp_df = df_data[ df_data['weekNumber']==w ]
        tmp_list = tmp_df[feature]
        w_start, w_end = tmp_list[0], tmp_list[-1]
        change = (w_end - w_start)/w_start * 100
        x.append(w)
        y.append(change)
    return x, y

# draw bar chart
def draw_bar(x_axis, y_axis, x_name, y_name, tn, title_name="", xt=None):
    data = {
            x_name: x_axis,
            y_name: y_axis
           }
    df = pd.DataFrame(data, columns=[x_name, y_name])
    # Defining the plot size
    # Defining the values for x-axis, y-axis
    # and from which dataframe the values are to be picked
    fig, ax = plt.subplots(figsize=(20, 6))
    plots = sns.barplot(x=x_name, y=y_name, data=df)
    if xt is not None:
        ax.set_xticks(x_axis)
        ax.set_xticklabels(xt)

    # Iterrating over the bars one-by-one
    for bar in plots.patches:
        # Using Matplotlib's annotate function and
        # passing the coordinates where the annotation shall be done
        plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=12, xytext=(0, 5),
                       textcoords='offset points')
    # Setting the title for the graph
    plt.title("Stock: " + tn + " " + title_name)
    # Finally showing the plot
    plt.show()
    return 

###
# 0005.HK:滙豐控股, 1299.HK:友邦保險, 0700.HK:騰訊控股, 9988.HK:阿⾥巴巴, 3690.HK:美團 # AAPL
hk_tickers = ["3690.HK", "9618.HK", "1024.HK", "9866.HK"]
tn = "3690.HK" 
st, et = "2022-01-01", "2022-12-31"
df_data = get_df_data(ticker_name=tn, start_time=st, end_time=et)
df_data = calculate_daily_return(df_data)
df_data = get_market_movement(df_data)
df_data = get_ymt_date(df_data)

date_list = list(df_data.index)
start_date, end_date = date_list[0], date_list[-1]
start_date, end_date = str(start_date)[:10], str(end_date)[:10]
title_name = "from " + start_date + " to " + end_date
# graph: monthly return 
x_axis, y_axis = month_change(df_data, "Close")
x_name, y_name = "Month_Number", "Return_Change"
draw_bar(x_axis, y_axis, x_name, y_name, tn, title_name)
# graph: weekly return 
x_axis, y_axis = week_change(df_data, "Close")
x_name, y_name = "Week_Number", "Return_Change"
draw_bar(x_axis, y_axis, x_name, y_name, tn, title_name)
