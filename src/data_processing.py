import pandas as pd
from src.utils import Read_DF, Call_Back
import os
from copy import deepcopy
from scipy.stats import hmean
import numpy as np
import math

def get_number_of_time_windows(time_window):
    if type(time_window) == str:
        time_w = float(time_window[0:-1])
    else:
        time_w = time_window
    windows = (24.0 * 60.0) / time_w
    return windows

def time_window_from_time(current_time, num_windows):
    #result = math.floor((current_time.hour * num_windows)/24.0)
    result = math.floor(((current_time.hour + (current_time.minute/60.0)) * num_windows) / 24.0)
    return result

def generate_datetime_arr(start_time, length, granularity=5, date='2019-01-01'):
    delta = pd.Timedelta(value=granularity, unit='minute')
    start_time = pd.Timestamp(f'{date} {start_time}')
    dt_col = []
    for i in range(length):
        dt_col.append(start_time)
        start_time = start_time + delta
    
    return dt_col
    
def convert_windows_to_dt_df(win_df, granularity=5, date='2019-01-01'):
    delta = pd.Timedelta(value=granularity, unit='minute')
    start_time = pd.Timestamp(f'{date} 00:00:00')
    dt_col = []
    for i in range(len(win_df)):
        dt_col.append(start_time)
        start_time = start_time + delta

    win_df['datetime'] = dt_col
    win_df = win_df.set_index('datetime')
    
#     _df = _df.groupby(pd.Grouper(freq='60T')).aggregate(hmean, axis=0)
    return win_df


def get_raw_data_df(month, half):
    m_dict = {"january":"01", "jan":"01", "february":"02", "feb":"02", "march": "03", "mar": "03", "april":"04", "apr":"04", "may":"05", "june":"06", "jun":"06", "july":"07", "jul":"07", "august":"08", "aug":"08", "september":"09", "sep":"09", "october":"10", "oct":"10", "november": "11", "nov": "11", "december":"12", "dec":"12"}
     # Load df_all for the segments with values
    MetaData={}

    #%% #Reading Combined Data
    if isinstance(month, int):
        if month > 0 and month <= 12:
            m = month
        else:
            return False
    elif isinstance(month, str):
        if month.lower() not in m_dict:
            print(f"Please enter in format of {m_dict.keys()}")
            return False
        else:
            m = int(m_dict[month.lower()])
    

    if not os.path.exists(os.path.join(os.getcwd(), 'data')):
        raise OSError("Must first download data, see README.md")
    data_dir = os.path.join(os.getcwd(), 'data')
    
    DF_All_Address = data_dir + f'/ALL_5m_DF_2019_{m}_{half}.gzip'
    DF_All = Read_DF(DF_All = DF_All_Address, Reading_Tag = 'DF_All', MetaData = MetaData)
    return DF_All
    
def get_day_from_raw(DF_All, day, time_start, time_end, cluster=None):
    days = DF_All.time_local.dt.day.unique().tolist()
    if day not in days:
        return False
    
    # Getting only 1 day within the dataframe
    # January 7, 2019 is a monday (0)
    single_day_df = DF_All[DF_All.time_local.dt.day == day]

    # Get only segments within the cluster in question
    if cluster:
        single_day_df = single_day_df.loc[single_day_df['XDSegID'].isin(cluster)]
        

    # Check if incidents exist here
    # Even without incidents, does not matter, since we are only testing the Qratios
    incident_count = len(single_day_df[single_day_df['Total_Number_Incidents']>0])
    print(f"Incident count: {incident_count}")

    # Get all data between 6am-9pm only
    single_day_df = single_day_df.set_index('time_local')
    single_day_df = single_day_df.between_time(time_start, time_end)
    return single_day_df

def compute_distributed_Q_ratio(single_day_df, cluster, granularity, speed_column='speed_mean'):
    
    delta = pd.Timedelta(value=granularity, unit='minute')

    window = 0
    segment = cluster[0]

    segment_means = {}
    for segment in cluster:
        first_time = single_day_df.index[0]
        last_time = single_day_df.index[-1]
        start_time = deepcopy(first_time)
        segment_means[f"{segment}_hm"] = []
        segment_means[f"{segment}_am"] = []
        segment_means[f"{segment}_qr"] = []
        while start_time < last_time:
            end_time = start_time + delta
            window_speeds = single_day_df[single_day_df['XDSegID'] == segment].between_time(start_time.time(), end_time.time())
            hm = hmean(window_speeds[speed_column])
            am = np.mean(window_speeds[speed_column])
            qr = hm/am
    #         print(f"{start_time}, {hm:.2f}, {am:.2f}, {qr:.2f}")
            segment_means[f"{segment}_hm"].append(hm)
            segment_means[f"{segment}_am"].append(am)
            segment_means[f"{segment}_qr"].append(qr)

            start_time = end_time

    cluster_df = pd.DataFrame(segment_means)
    cluster_df.head()
    
    return cluster_df

def get_data_slice_from_overall(overall_df, dayofweek, time_start, time_end, date='2019-01-01', segment=None, granularity=5):
    # Always has 5 min granularity
    time_windows = get_number_of_time_windows(granularity)

    start_time = pd.Timestamp(time_start)
    end_time = pd.Timestamp(time_end)
    start_window = time_window_from_time(start_time, time_windows)
    end_window = time_window_from_time(end_time, time_windows)
    
    if segment:
        _slice = overall_df.loc[(dayofweek, start_window):(dayofweek, end_window)].xs((segment,), level=['XDSegID']).droplevel(0)
        
        start_time_str = start_time.strftime(time_start)
        datetime_arr = generate_datetime_arr(start_time_str, len(_slice), granularity, date=date)
        _slice['time_local'] = datetime_arr
        _slice = _slice.reset_index()
        _slice.set_index('time_local', inplace=True)
        return _slice

    else:
        return False