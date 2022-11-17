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

# get all dates from a range
def get_dates_from_range(start_date, end_dates):
    start = datetime.datetime.strptime(start_date, "%d-%m-%Y") 
    end = datetime.datetime.strptime(end_dates, "%d-%m-%Y")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    date_list = []
    for date in date_generated:
        date_ymd = date
        y, m, d = date_ymd.year, date_ymd.month, date_ymd.day
        ymd = str(y) + "-"
        if m<10:
            ymd = ymd + "0" + str(m)
        else:
            ymd = ymd + str(m)
        ymd = ymd + "-"
        if d<10:
            ymd = ymd + "0" + str(d)
        else:
            ymd = ymd + str(d)
        date_list.append( ymd  )
    return date_list

# search value from a list, return the index
def search_list(date_str, date_list):
  ii = 0
  while ii<len(date_list):
    if date_str==date_list[ii]:
      return ii
    ii += 1
  return -1

# find the index of max value in a list
def find_index_max(tmp_list):
    ind = 0
    mx = tmp_list[ind]
    i = 0
    while i<len(tmp_list):
        if mx<tmp_list[i]:
            mx = tmp_list[i]
            ind = i
        i += 1
    return ind

# find the index of min value in a list
def find_index_min(tmp_list):
    ind = 0
    mn = tmp_list[ind]
    i = 0
    while i<len(tmp_list):
        if mn>tmp_list[i]:
            mn = tmp_list[i]
            ind = i
        i += 1
    return ind

# count frequency
def add_value_to_dic(v, dic):
  if v not in dic:
    dic[v] = 1
  else:
    dic[v] += 1
  return 

# convert dict-type data to be 2 lists (x & y)
    # dict-type has keys & values, keys -> x list, values -> y list
def dic_to_xy(tmp_dic):
    index_list = list( range(12) )
    x_axis, y_axis = [], []
    for it in index_list:
        x_axis.append( it  )
        if it not in tmp_dic:
            y_axis.append(0)
        else:
            y_axis.append(tmp_dic[it])
    return x_axis, y_axis

# calculate percentage_for_list
def percentage_for_list(tmp_list):
    s = sum(tmp_list)
    if s<=0:
        return []
    ll = []
    for it in tmp_list:
        ll.append( it/s )
    return ll

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

# calculate the frequencies for high-point, low-point over different time interval
  # time interval: every 30 minutes
def high_low_frequency(tn, st, et):
  date_list = get_dates_from_range("01-01-"+st[:4], "31-12-"+et[:4])
  st_ind, ed_ind = search_list(st, date_list), search_list(et, date_list)
  print( st_ind, date_list[st_ind] )
  print( ed_ind, date_list[ed_ind] )
  #
  high_dic, low_dic = {}, {}
  for i in range(st_ind, ed_ind): # 
    st, et = date_list[i], date_list[i+1]
    print(st, et)
    # intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # df_data = get_df_data(ticker_name=tn, start_time=st, end_time=et) 
    df_data = get_df_data_update(ticker_name=tn, start_time=st, end_time=et, interval_len="30m")
    if len(df_data) != 12:
        continue
    df_data = get_ymt_date(df_data)
    #
    mx_ind = find_index_max(df_data['High'])
    row = df_data.iloc[mx_ind]
    high_point, hm_time = round(row['High'],2), row['hour_minute']
    add_value_to_dic(mx_ind, high_dic)

    mn_ind = find_index_min(df_data['Low'])
    row = df_data.iloc[mn_ind]
    low_point, lm_time = round(row['Low'],2), row['hour_minute']
    add_value_to_dic(mn_ind, low_dic)
  
  return high_dic, low_dic


### 
# 0005.HK:滙豐控股, 1299.HK:友邦保險, 0700.HK:騰訊控股, 9988.HK:阿⾥巴巴, 3690.HK:美團 # AAPL
hk_tickers = ["3690.HK", "9618.HK", "1024.HK", "9866.HK"]
tn = "3690.HK" 
st, et = "2022-09-16", "2022-11-16"
high_dic, low_dic = high_low_frequency(tn, st, et)

index_time = [
    '09:30', '10:00', '10:30', '11:00', '11:30', '13:00', '13:30', '14:00','14:30', '15:00', '15:30', '16:00'
]

title_name = " from " + st + " to " + et
x_axis, y_axis = dic_to_xy(high_dic) # high_dic, low_dic
y_axis = percentage_for_list(y_axis)
x_name, y_name = 'what time in a day High close appear', 'frequency'
draw_bar(x_axis, y_axis, x_name, y_name, tn, title_name, index_time)

x_axis, y_axis = dic_to_xy(low_dic) # high_dic, low_dic
y_axis = percentage_for_list(y_axis)
x_name, y_name = 'what time in a day Low close appear', 'frequency'
draw_bar(x_axis, y_axis, x_name, y_name, tn, title_name, index_time)
