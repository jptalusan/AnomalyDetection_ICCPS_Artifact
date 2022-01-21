import os
import pandas as pd
import numpy as np
from copy import copy
pd.options.mode.chained_assignment = None  # default='warn'

from scipy.signal import find_peaks
from scipy import signal

window = 0.5 # incident search window (hours before and hours after)
order = 1 # For finding valleys
slope_threshold_modifier = 1 # Adjustment for detecting the slope as a change point

def locate_segment_cluster(segment, clusters):
    if pd.isnull(segment):
        return -1
    for i, cluster in enumerate(list(clusters)):
        if segment in clusters[cluster]:
            return cluster
    return -1

def compute_slope_difference(data):
    return data.shift(-1)-data

# window is in hours, dt-window:dt+window
def get_window_around_incident(incident_dt, window=1):
    # Add special case if earlier than 6am
#     print(incident_dt)
    delta = pd.Timedelta(f'{window}h')
    before = incident_dt - delta
    after  = incident_dt + delta
#     print(before, after)
    return before, after

def get_peaks_valleys(dtindex, slopes, before, after, modifier=1, window=2, order=1):
    _before = slopes.index.get_loc(before, method='bfill')
    _after = slopes.index.get_loc(after, method='bfill') + 1
    valid_window = list(range(_before, _after))
    
#     print(valid_window)
    peaks = find_peaks(slopes, height=slopes.mean() + slopes.std() * modifier)
    # valleys
    valid_valleys = []
    valleys = signal.argrelextrema(slopes.to_numpy(), np.less_equal, order=order)
    for v in valleys[0]:
        if slopes.iloc[v] < (slopes.mean() - slopes.std() * modifier):
            valid_valleys.append(v)

#     print(valid_valleys, peaks[0])
    
    valid_valleys = [i for i in valid_valleys if i in valid_window]
    peaks = [i for i in peaks[0] if i in valid_window]
    
    
    return valid_valleys, peaks

# Pass a list of tuples of dips (valley and peak pairs)
def get_cma(df, min_valley, min_peak, method='cwma'):
    dip_len = min_peak - min_valley + 1
    clean_start = min_valley - dip_len
    clean_end   = min_valley
    
    # Get time indices
    cs_dt = df.index.values[clean_start]
    ce_dt = df.index.values[clean_end]
#     print(cs_dt, ce_dt)
    if method == 'cwma':
        return df[cs_dt:ce_dt].expanding().mean()
    elif method == 'ewm':
        return df[cs_dt:ce_dt].ewm(alpha=0.3, adjust=False).mean()
    elif method == 'rolling':
        return df[cs_dt:ce_dt].rolling(3, min_periods=1).mean()

# Just replace the data between window start to window end to reference_speed_mean
def clean_speed_data_basic(speed_data, incident_timestamp, window=window):
    speed_means = copy(speed_data)
    # actual incident data window
    before = incident_timestamp - pd.Timedelta(f'{window}h')
    after  = incident_timestamp + pd.Timedelta(f'{window}h')
    # reference data (window before incident)
    r_before = before - pd.Timedelta(f'{2 * window}h')
    r_after  = before
    old_vals = speed_means.loc[before:after]['speed_mean'].tolist()
    new_vals = speed_means.loc[r_before:r_after]['speed_mean'].expanding().mean().tolist()
    adj_vals = []
    for i, val in enumerate(new_vals):
        if val < old_vals[i]:
            adj_vals.append(old_vals[i])
        else:
            adj_vals.append(val)
    speed_means.loc[before:after, 'speed_mean'] = adj_vals
    return speed_means