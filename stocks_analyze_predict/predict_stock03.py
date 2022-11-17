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
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier



# predict HK stocks according to US stocks
    # there is jet-lag between US stocks and HK stocks
    # US stock markets have much influence over HK stock markets
    # exploit the information from US stock markets to predict HK stock markets



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

# calculate the daily change of points & volumes
    # by (current_index - previous_index) / previous_index
    # by (current_volume - previous_volume) / previous_volume
def calculate_daily_change(df_data):
  # , OHLC_index="Close"
  df_data = calculate_daily_return(df_data, "Close")
  df_data = calculate_daily_return(df_data, "Volume")
  return df_data #, new_feature

# convert the time to be string type: yyyy-mm-dd
def get_ymt_date(df_data):
  df_data["ymd_time"] = df_data.index
  df_data["ymd_time"] = df_data["ymd_time"].astype(str)
  df_data["ymd_time"] = df_data["ymd_time"].str.slice(0,10)
  return df_data

# get useful features and pack as dict-type
    # key: date, value:
def get_date_features(df_data, features=["Close_return", "Volume_return"]):
    time_list = list(df_data["ymd_time"])
    values_list = []
    L = len(df_data)
    for ii in range(L):
        fv = []
        for fn in features:
            v = df_data.iloc[ii][fn]
            fv.append(v)
        values_list.append( fv )
    time_return_dic = {}
    ii = 0
    while ii<L:
        time_return_dic[ time_list[ii] ] = values_list[ii]
        ii = ii + 1
    return time_return_dic

# 3 US stock indexes: Nasdaq, DJI, SP500, 1 HK index: HSI
# get their data for certain year
# close points -> moving average close points, parameter: nn
# calculate the daily return based on the MA close points
# get the date-return dic
def get_useful_data(tn, st, et, nn, features=["Close_return", "Volume_return"]):
  stock_index = get_df_data(ticker_name=tn, start_time=st, end_time=et)
  stock_index['SMA'+str(nn)] = stock_index['Close'].rolling(nn).mean() # moving average, smoothening function
  stock_index['Close'] = stock_index['SMA'+str(nn)] # moving average, smoothening function
  del stock_index['SMA'+str(nn)] # 
  stock_index = calculate_daily_change(stock_index)
  stock_index = get_ymt_date(stock_index)
  time_return = get_date_features(stock_index, features)
  return stock_index, time_return

# compare 2 string-type dates, 
  # for example: '2022-01-01' -> 20220101, '2022-10-10' -> 20221010,
def compare_date_str(date_str1, date_str2):
  num1 = date_str1[0:4]+date_str1[5:7]+date_str1[8:10]
  num1 = int(num1)
  num2 = date_str2[0:4]+date_str2[5:7]+date_str2[8:10]
  num2 = int(num2)
  if num1>num2:
    return 1
  elif num1<num2:
    return -1
  return 0

# match US stocks with HK stock by time, because there is jet-lag between HK time & US time
def stocks_signal(refer_list=["^IXIC", "^DJI", "^GSPC"], target="^HSI", st="2022-01-01", et="2022-12-31", nn=5, features=["Close_return", "Volume_return"]):
  # reference list
  refer_dic_list = []
  for it in refer_list:
    _, tmp = get_useful_data(it, st, et, nn, features)
    refer_dic_list.append( tmp )
  # target
  _, target_dic = get_useful_data(target, st, et, nn, features) # ["Close_return"]
  # integrate references & target
  hk_us_stock_signal = []
  for k, v in sorted(target_dic.items()):
    date_str = k
    rise_fall_signal = v
    for pk, pv in sorted(refer_dic_list[0].items(), reverse=True):
      if compare_date_str(k, pk)==1:
        vs = []
        for rd in refer_dic_list:
          vs.append( rd[pk] )
        #print(k, v, pk, vs)
        hk_us_stock_signal.append( (k, v, pk, vs) )
        break
  return hk_us_stock_signal

# list of lists -> dataframe
def lists_to_dataframe(hk_us_stock_signal, refer_list, target):
  tmp_list = []
  L = 0
  for it in hk_us_stock_signal[ : ]:
    hkt, hk_hsi, ust, vs = it
    # flatten list of lists to be list 
    vs = np.array(vs)
    vs = vs.reshape(-1)
    vs = list(vs)
    L = len(vs)
    sub_list = []
    sub_list.append( hkt )
    sub_list += hk_hsi
    sub_list.append( ust )
    sub_list += vs
    tmp_list.append( sub_list )
  cols = ["HKT"]
  cols.append( target+"_close" )
  cols.append( target+"_volume" )
  cols.append( "UST" )
  for it in refer_list:
    cols.append( it+"_close" )
    cols.append( it+"_volume" )
  #print( len(cols), len(tmp_list[0]) )
  assert len(cols)==len(tmp_list[0])
  df_data = pd.DataFrame(tmp_list, columns=cols)
  return df_data

# scale list
def scale_list_values(vs):
  tmp = []
  for v in vs:
    tmp.append( v*100 )
  return tmp

# get data-sets for training & testing models from dataframe
def dataframe_Xy(df_data):
    X, y = [], []
    for i in range(len(df_data)):
        row = list( df_data.iloc[i] )
        target_daily_return = row[1]
        refer_values = row[4:]
        #
        if math.isnan(target_daily_return):
            continue
        fg = False
        for v in refer_values:
            if math.isnan(v):
                fg = True
                break
        if fg:
            continue
        #
        X.append( scale_list_values(refer_values) )
        if target_daily_return>0:
            y.append(1)
        else:
            y.append(0)
    return X, y

# sample data for training & testing model
def sample_dataset(df_data, ran_seed, sample_number=-1):
    # balance the positive and negative samples
    positive_df = df_data[ df_data[target+"_close"]>0 ].copy()
    negative_df = df_data[ df_data[target+"_close"]<0 ].copy()
    positive_df = positive_df.dropna()
    negative_df = negative_df.dropna()
    print( "business days: ", len(df_data), "rise vs fall: ", len(positive_df), len(negative_df) )
    if sample_number==-1:
        sample_number = min( len(positive_df), len(negative_df) )
    X_pos_data, y_pos_data = dataframe_Xy(positive_df.sample(n=sample_number, random_state=ran_seed))
    X_neg_data, y_neg_data = dataframe_Xy(negative_df.sample(n=sample_number, random_state=ran_seed))
    X_data = X_pos_data + X_neg_data
    y_data = y_pos_data + y_neg_data
    print( "sampled business days: ", len(X_data), "rise vs fall: ", len(X_pos_data), len(X_neg_data) )
    return X_data, y_data

def split_dataset(X_data, y_data, ran_seed):
    ### data-set split and train models
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=ran_seed1, shuffle=True)
    print( "training data: ", len(X_train), len(y_train) )
    print( "testing data: ", len(X_test), len(y_test) )
    return X_train, X_test, y_train, y_test

# train RF model
def train_RF_model(X_train, X_test, y_train, y_test, ran_seed):
    # RF
    RF1=RandomForestClassifier(n_estimators=100, random_state=ran_seed)
    RF1.fit(X_train, y_train)
    y_pred = RF1.predict(X_test)
    score = RF1.score(X_test, y_test)
    print(classification_report(y_test, y_pred))
    print("RF", score)
    return RF1, score

# get latest US stock data to predict HK stock
def get_latest_features(st, et, refer_list):
    df_reference = pd.DataFrame()
    lastest_data = []
    lastest_day = -1
    for tn in refer_list:
        tmp_df = get_df_data(tn, st, et)
        tmp_df = calculate_daily_change(tmp_df)
        df_reference[tn+"_Close_return"] = tmp_df[ 'Close_return' ]
        df_reference[tn+"_Volume_return"] = tmp_df[ 'Volume_return' ]
        date_time = list(tmp_df.index)[lastest_day]
        close_point = list(tmp_df[ 'Close_return' ])[lastest_day]
        volume_point = list(tmp_df[ 'Close_return' ])[lastest_day]
        lastest_data.append( close_point )
        lastest_data.append( volume_point )
        print("date_time:", date_time, tn, close_point*100, volume_point*100)
    return df_reference, lastest_data

###
### prepare data to fit models
# ^HSI,
# 0005.HK:滙豐控股, 1299.HK:友邦保險, 0700.HK:騰訊控股, 9988.HK:阿⾥巴巴,
# 3690.HK:美團, '9888.HK':百度, 9866.HK:蔚来
nn = 1 # nn=1 means use the Close, nn>1 means use Moving Average based on Close
dataset_xy = []
refer_list=["^IXIC", "AAPL", "GOOGL", "AMZN", "BABA", "PDD", "JD", "MPNGY", "TME", "BIDU"] # "^IXIC", "^DJI", "^GSPC"
target="3690.HK" 
st, et = "2022-01-01", "2022-11-17"
features=["Close_return", "Volume_return"]

hk_us_stock_signal = stocks_signal(refer_list, target, st, et, nn, features)
df_data = lists_to_dataframe(hk_us_stock_signal, refer_list, target)
print(df_data)


fg = False # False, True
if fg:
    score_model = {'LR':[], 'SVM':[], 'RF':[]}
    for ran_seed1 in range(6, 7):
        sample_number = -1
        X_data, y_data = sample_dataset(df_data, ran_seed1, sample_number)
        for ran_seed2 in range(10):
            X_train, X_test, y_train, y_test = split_dataset(X_data, y_data, ran_seed2)
            for ran_seed3 in range(10):
              RF1, score = train_RF_model(X_train, X_test, y_train, y_test, ran_seed3)
              score_model['RF'].append( (score, ran_seed1, ran_seed2, ran_seed3) )
            
    #
    RF_list = sorted(score_model['RF'])
    print(RF_list[-3:])

sample_number = -1
ran_seed1, ran_seed2, ran_seed3 = 6, 7, 3
X_data, y_data = sample_dataset(df_data, ran_seed1, sample_number)
X_train, X_test, y_train, y_test = split_dataset(X_data, y_data, ran_seed2)
RF1, score = train_RF_model(X_train, X_test, y_train, y_test, ran_seed3)

#
st, et = "2022-01-01", "2022-12-31"
df_reference, lastest_data = get_latest_features(st, et, refer_list)
print(df_reference)
lastest_data = scale_list_values(lastest_data) # scale the data as the procedure of training data
print("lastest_data of US stock market info: ", lastest_data)
#
rise_or_fall = RF1.predict([lastest_data])
prob = RF1.predict_proba([lastest_data])
print( "RF predict today: ", rise_or_fall, prob )
