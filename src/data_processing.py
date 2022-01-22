import pandas as pd

def convert_windows_to_dt_df(win_df, granularity=5, date='2019-01-01'):
    delta = pd.Timedelta(value=granularity, unit='minute')
    start_time = pd.Timestamp(f'{date} 00:00:00')
    dt_col = []
    for i in range(len(win_df)):
        dt_col.append(start_time)
        start_time = start_time + delta

    win_df['datetime'] = dt_col
    win_df = win_df.set_index('datetime')
    
    return win_df