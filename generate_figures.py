print("Generating figures...")

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import contextily as cx
import matplotlib.dates as md

from itertools import combinations
from matplotlib.lines import Line2D

from src.common_functions import *

# Confirm directorys are in place

if not os.path.exists(os.path.join(os.getcwd(), 'synthetic_data')):
    raise OSError("Must first download data, see README.md")
synth_data = os.path.join(os.getcwd(), 'synthetic_data')

if not os.path.exists(os.path.join(os.getcwd(), 'synthetic_results')):
    raise OSError("Must first download data, see README.md")
synth_results = os.path.join(os.getcwd(), 'synthetic_results')

if not os.path.exists(os.path.join(os.getcwd(), 'synthetic_figures')):
    raise OSError("Must first download data, see README.md")
synth_figures = os.path.join(os.getcwd(), 'synthetic_figures')

start_time = '06:00'
end_time   = '20:55'
training_months = 10
cross_validation_months = 11
testing_months = 12
granularity = 5
new_filename = 'synth'
kappa_L = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Only for the synthetic data, must adjust for real data
idx = pd.date_range(pd.Timestamp('12-01-2019 00:00:00'), pd.Timestamp('12-14-2019 23:59:59'), freq="5min")
detection_attempts = len(idx[idx.indexer_between_time('06:00', '20:55')])


def clustering_table_2():
    fp = os.path.join(synth_data, f'synth_clustering.pkl')
    with open(fp, 'rb') as handle:
        clusters = pickle.load(handle)

    fp = os.path.join(synth_data, 'synth_loaded_G_correlation_weekday_6am_9pm.pkl')
    SG = nx.read_gpickle(fp)

    fp = os.path.join(synth_data, f'synth_optimized_clustering_stats.pkl')
    stats_df = pd.read_pickle(fp)
    restriced_stats_df = stats_df[stats_df['clustered'] >= 4]

    fp = os.path.join(synth_results, f'optimized_results_{new_filename}.pkl')
    with open(fp, 'rb') as handle:
        results = pickle.load(handle)

    overall_correlation = []
    for head, cluster in clusters.items():
        if len(clusters[head]) < 4:
            continue
        combos = list(combinations(cluster, 2))
        correlations = []
        for c in combos:
            ed = SG.get_edge_data(*c, 0)
            if ed:
                correlation = ed['correlation']
                correlations.append(correlation)
            else:
                ed = SG.get_edge_data(c[1], c[0], 0)
                if ed:
                    correlation = ed['correlation']
                    correlations.append(correlation)
                else:
                    continue
        correlations = np.asarray(correlations)
        overall_correlation.append(np.mean(correlations) if np.mean(correlations) > 0 else 0)
        
    corr = np.mean(np.asarray(overall_correlation))

    count = restriced_stats_df.clustered.sum()
    mean = restriced_stats_df.clustered.mean()
    _min = restriced_stats_df.clustered.min()
    _max = restriced_stats_df.clustered.max()
    ave_r = restriced_stats_df.radius.mean()
        
    actual_incidents = 0
    true_positives   = 0
    false_positives  = 0
    all_attempts     = 0

    all_attempts = detection_attempts
    for j, r in enumerate(results):
        if j > 9:
            continue
        actual_incidents += results[r]['total_actual_incident']
        true_positives   += results[r]['results']['count']
        false_positives  += results[r]['results']['fa_alarm']

    tp = true_positives
    tn = all_attempts - actual_incidents - false_positives
    fp = false_positives
    fn = actual_incidents - true_positives

    precision           = tp / (tp + fp)
    true_positive_rate  = tp / (tp + fn) # recall
    false_positive_rate = fp / (fp + tn) # 
    true_negative_rate  = 1 - false_positive_rate
    accuracy            = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy   = (true_positive_rate + true_negative_rate) / 2

    beta = 20
    fbeta_measure       = ((1 + beta**2) * precision * true_positive_rate) / (beta**2 * precision + true_positive_rate)
    f_measure           = (2 * precision * true_positive_rate) / (precision + true_positive_rate)

    print(actual_incidents, true_positives, false_positives, all_attempts)
    print(f"{true_positive_rate:.2f}, {false_positive_rate:.5f}")
    print(f"{balanced_accuracy:.3f}, {accuracy:.3f}")
    print(f"{precision:.2f}, {fbeta_measure:.2f}, {f_measure:.2f}")

    data = {'fbeta_measure': fbeta_measure,
            'balanced_accuracy': balanced_accuracy,
            'false_positive_rate': false_positive_rate
        }
    print(data)
    TPR = data['balanced_accuracy']
    FPR = data['false_positive_rate']

    fp = os.path.join(synth_figures, 'synth_Table_2.txt')
    with open(fp, 'w') as f:
        print(f"count: {count}", file=f)
        print(f"mean: {mean }", file=f)
        print(f"min: {_min }", file=f)
        print(f"max: {_max }", file=f)
        print(f"ave. radius: {ave_r:.2f}", file=f)
        print(f"Overall correlation per cluster: {corr:.4f}", file=f)
        print(f"area coverage: {count/len(list(SG.nodes))}", file=f)
        print(f"TPR:{TPR:.5}", file=f)
        print(f"FPR:{FPR:.5}", file=f)

def cluster_graph():
    fp = os.path.join(synth_data, f'synth_clustering.pkl')
    with open(fp, 'rb') as handle:
        clusters = pickle.load(handle)
    print(len(clusters))
    fp = os.path.join(synth_data, 'synth_overall_means.pkl')
    df_overall_all = pd.read_pickle(fp)
    active_segments = df_overall_all.droplevel([0, 1]).reset_index()['segmentID'].unique().tolist()
    print(len(active_segments))

    fp = os.path.join(synth_data, 'synth_segments_grouped.pkl')
    segments_df = pd.read_pickle(fp)
    segments_df = segments_df[segments_df['segmentID'].isin(active_segments)]
    segments_df = segments_df.set_geometry('geometry')

    segments_df.set_crs(epsg=4326, inplace=True, allow_override=True)
    segments_df = segments_df.to_crs('EPSG:3857')

    _, ax = plt.subplots(figsize=(10, 10))
    segments_df.plot(ax=ax, figsize=(10, 10), lw=0, marker='o', markersize=200)
    cx.add_basemap(ax, crs=segments_df.crs.to_string(), zoom=10, source=cx.providers.OpenStreetMap.Mapnik)
    plt.axis('off')
    ax.set_title("Target Area: Nashville, TN")

    # Adding cluster rings
    for cluster_head in list(clusters.keys())[0:]:
        segments = segments_df[segments_df['segmentID'].isin(clusters[cluster_head])]
        poly_s = segments.unary_union.envelope
        c1 = random.uniform(0, 1)
        c2 = random.uniform(0, 1)
        c3 = random.uniform(0, 1)
        ax.fill(*poly_s.exterior.xy, edgecolor=(0,0,0,1.0), facecolor=(c1, c2, c3,.4))

    fp = os.path.join(synth_figures, f'synth_baseline_map_with_grids.png')
    plt.savefig(fp, dpi=200)

def generate_figure_5():
    fp = os.path.join(synth_data, f'synth_cluster_ground_truth.pkl')
    incident_GT_Frame = pd.read_pickle(fp)

    fp = os.path.join(synth_data, f'synth_clustering.pkl')
    with open(fp, 'rb') as handle:
        clusters = pickle.load(handle)
    
    start_time = '06:00'
    end_time   = '20:55'
    testing_incident_GT =  incident_GT_Frame.between_time(start_time, end_time)
    testing_incident_GT_Clist =  testing_incident_GT[testing_incident_GT['cluster_head'].isin (clusters)].sort_index()

    fp_standard_limit = os.path.join(synth_results, f'optimized_standard_limit_{new_filename}.pkl')
    with open(fp_standard_limit, 'rb') as handle:
        standard_limit_5C = pickle.load(handle)
    standard_limit_5C_Frame = pd.DataFrame(standard_limit_5C)

    fp_test_res = os.path.join(synth_results, f'optimized_residual_Test_QR_{new_filename}.pkl')
    with open(fp_test_res, 'rb') as handle:
        test_residual = pickle.load(handle)
        
    # Saving and backing up
    fp = os.path.join(synth_results, f'optimized_hyper_mapping_{new_filename}.pkl')
    with open(fp, 'rb') as handle:
        cross_validated_kappa_SF = pickle.load(handle)

    a = testing_incident_GT_Clist[testing_incident_GT_Clist.index.month == testing_months].sort_values(by='Total_Number_Incidents', ascending=False)
    # Highest incident cluster
    i = 0
    active_cluster = a.iloc[i]['cluster_head']
    start = a.iloc[i].name.floor('D') + pd.Timedelta('6h')
    end = a.iloc[i].name.floor('D') + pd.Timedelta('21h')

    kappa = cross_validated_kappa_SF[active_cluster]['kappa']
    SF    = cross_validated_kappa_SF[active_cluster]['SF']

    _, ax = plt.subplots(figsize=(10, 5))

    test_residual_frame = pd.DataFrame(list(test_residual[active_cluster][kappa][SF].items()),
                                    columns = ['time','RUC'])
    test_residual_frame.set_index('time', inplace=True)
    std_limit = standard_limit_5C_Frame[(standard_limit_5C_Frame['cluster_id'] == active_cluster)
                                        &(standard_limit_5C_Frame['ka ppa'] == kappa)
                                        &(standard_limit_5C_Frame['SF'] == SF)]

    index_ar = std_limit.index
    tau_max = std_limit.at[index_ar[0],'tau_max']
    tau_min = std_limit.at[index_ar[0],'tau_min']

    ax.axhline(y=tau_max, color='k', linestyle='-.')
    ax.axhline(y=tau_min, color='k', linestyle='--')

    foucsed_cluster = testing_incident_GT_Clist[testing_incident_GT_Clist['cluster_head']==active_cluster]
    foucsed_cluster = foucsed_cluster[start:end]
    plot_frame = test_residual_frame[start:end]
    plot_frame.plot(y='RUC', ax=ax)
    for k, r in foucsed_cluster.iterrows():
        ax.axvline(x=k, color='r', ls='--')
        
    plt.title("Residual: "+str(active_cluster)+" kappa: "+str(kappa)+" SF: "+str(SF))

    ax.annotate(r"$\tau_{max}$",
                xy=(start + pd.Timedelta('7h'), tau_max),
                xytext=(0, 15),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=0,angleB=-90"),
                fontsize=15)

    ax.annotate(r"$\tau_{min}$",
                xy=(start + pd.Timedelta('7h'), tau_min),
                xytext=(-10, -40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=10,angleB=-90"),
                fontsize=15)

    ax.set_ylabel('RUC')
    ax.set_xlabel('Time')
    custom_lines = [
                    Line2D([0], [0], color='blue', lw=1, ),
                    Line2D([0], [0], color='red', lw=1, ls='--'),
                    Line2D([0], [0], color='k', linestyle='-.', lw=1, ),
                    Line2D([0], [0], color='k', linestyle='--', lw=1, ),
                ]
    ax.legend(custom_lines, ['Residual', 'Incidents'])

    myFmt = md.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    plt.xticks(rotation=0, ha='right')
    fp = os.path.join(synth_figures, 'synth_Figure_5.png')
    plt.savefig(fp, bbox_inches='tight', dpi=200)

# This is required for other graphs
# Might have to put this in the training file instead
def test_all_kappas():
    fp_safe_margin = os.path.join(synth_results, f'optimized_hyper_mapping_{new_filename}.pkl')
    with open(fp_safe_margin, 'rb') as handle:
        hyper_mapping = pickle.load(handle)
    
    fp_safe_margin = os.path.join(synth_results, f'optimized_safe_margin_{new_filename}.pkl')
    with open(fp_safe_margin, 'rb') as handle:
        safe_margin = pickle.load(handle)

    fp = os.path.join(synth_results, f'used_clusters_list_{new_filename}.pkl')
    with open(fp, 'rb') as handle:
        cluster_list = pickle.load(handle)

    fp = os.path.join(synth_data, f'synth_cluster_ground_truth.pkl')
    incident_GT_Frame = pd.read_pickle(fp)

    fp_standard_limit = os.path.join(synth_results, f'optimized_standard_limit_{new_filename}.pkl')
    with open(fp_standard_limit, 'rb') as handle:
        standard_limit_5C = pickle.load(handle)
    standard_limit_5C_Frame = pd.DataFrame(standard_limit_5C)

    # Possible parameters used
    SF_List = [3,5,7,9]

    info_ratio_incidents = []
    for file in os.listdir(synth_data):
        filename = os.fsdecode(file)
        if 'incidents.pkl' in filename:
            fp = os.path.join(synth_data, filename)
            df = pd.read_pickle(fp)
            info_ratio_incidents.append(df)
            
    combined_ratio_frame_incidents = pd.concat(info_ratio_incidents)
    combined_ratio_frame_incidents = combined_ratio_frame_incidents.between_time(start_time, end_time)
    combined_ratio_frame_incidents =  combined_ratio_frame_incidents[(combined_ratio_frame_incidents.index.month == testing_months)]
    testing = combined_ratio_frame_incidents

    # Part 1
    for _kappa in kappa_L:
        for hp in hyper_mapping:
            hyper_mapping[hp]['kappa'] = _kappa
            
        cross_validated_kappa_SF = hyper_mapping

        testing_Clist = testing[list(cross_validated_kappa_SF.keys())]
        testing_Clist.columns = list(cross_validated_kappa_SF.keys())
        testing_Clist.columns

        test_residual = {}
        for column in testing_Clist.columns:
            grouped = testing_Clist[column].groupby([testing_Clist[column].index.hour,
                                                     testing_Clist[column].index.minute])

            sm_per_C = safe_margin[column]
            kappa = cross_validated_kappa_SF[column]['kappa']
            SF = cross_validated_kappa_SF[column]['SF']
            R_per_C = {}
            nabla_dict = calculate_nabla(grouped,sm_per_C[kappa])
            nabla_frame = pd.DataFrame(list(nabla_dict.items()),columns = ['time','nabla'])
            nabla_frame.set_index('time', inplace=True)
            _grouped = nabla_frame.groupby(nabla_frame.index.floor('D'))
            RUC = {}

            RUCsf = {}
            for k, group in _grouped:
                df = group.rolling(SF, min_periods=SF).sum()
                df[0:SF] = group[0:SF]
                _RUC = df.to_dict()['nabla']
                RUCsf.update(_RUC)

            RUC[SF] = RUCsf
            R_per_C[kappa] = RUC
            test_residual[column] = R_per_C

        # Saving and backing up
        fp = os.path.join(synth_results, f'{_kappa}_optimized_residual_Test_QR_{new_filename}.pkl')
        with open(fp, 'wb') as handle:
            pickle.dump(test_residual, handle)

    # Part 2
    for _kappa in kappa_L:
        for hp in hyper_mapping:
            hyper_mapping[hp]['kappa'] = _kappa
            
        fp_test_res = os.path.join(synth_results, f'{_kappa}_optimized_residual_Test_QR_{new_filename}.pkl')
        with open(fp_test_res, 'rb') as handle:
            test_residual = pickle.load(handle)

        cross_validated_kappa_SF = hyper_mapping

        testing_incident_GT = incident_GT_Frame.between_time(start_time, end_time)
        testing_incident_GT_Clist = testing_incident_GT[testing_incident_GT['cluster_head'].isin (cluster_list)]
        testing_incident_GT_Clist = testing_incident_GT_Clist[testing_incident_GT_Clist.index.month == testing_months]

        testing = combined_ratio_frame_incidents.between_time(start_time, end_time)
        testing =  testing[(testing.index.month>9) & (testing.index.month<=12) ]

        testing_Clist = testing[list(cross_validated_kappa_SF.keys())]
        testing_Clist.columns = list(cross_validated_kappa_SF.keys())

        detection_report = []
        for column in testing_Clist.columns: #per cluster 
            grouped = testing_Clist[column].groupby([testing_Clist[column].index.hour,
                                                    testing_Clist[column].index.minute])
            sm_per_C = safe_margin[column] # safe margin list for each cluster
            kappa = cross_validated_kappa_SF[column]['kappa']
            SF = cross_validated_kappa_SF[column]['SF']
            for key1, group in grouped:
                for index, item in group.iteritems():
                    if(pd.isna(item)):continue
                    if((item > sm_per_C[kappa]['upper'][key1] ) or (item < sm_per_C[kappa]['lower'][key1] )):
                        res_SF = test_residual[column][kappa]
                        std_limit = standard_limit_5C_Frame[(standard_limit_5C_Frame['cluster_id']== column) &
                                                    (standard_limit_5C_Frame['ka ppa']== kappa) &
                                                    (standard_limit_5C_Frame['SF']== SF)]

                        index_ar = std_limit.index
                        if(res_SF[SF][index] >0):
                            if(res_SF[SF][index]>std_limit.at[index_ar[0],'tau_max']):
                                temp = {'cluster_id':column,'kappa':kappa,'SF':SF,
                                        'time':index,'RUC':res_SF[SF][index],'tau_max':std_limit.at[index_ar[0],'tau_max']}
                                detection_report.append(temp)
                        else:
                            if(res_SF[SF][index]<std_limit.at[index_ar[0],'tau_min']):
                                temp = {'cluster_id':column,'kappa':kappa,'SF':SF,
                                        'time':index,'RUC':res_SF[SF][index],'tau_min':std_limit.at[index_ar[0],'tau_min']}
                                detection_report.append(temp)
        detection_report_Frame = pd.DataFrame(detection_report)
        detection_report_Frame.set_index('time',inplace = True)

        group_detection_report_by_cluster_id = detection_report_Frame.groupby('cluster_id')
        actual_detection = []
        detection_GT = []
        for key,group in group_detection_report_by_cluster_id:
            foucsed_cluster = testing_incident_GT_Clist[testing_incident_GT_Clist['cluster_head']==key]
            for index,row in group.iterrows():
                detection_type = 0
                for index1,row1 in foucsed_cluster.iterrows():
                    #iterate only incidents happend for the cluster 
                    if((index.month == index1.month) and (index.day == index1.day)):
                        #This means incident and detection are on the same day
                        if((index.hour >= (index1.hour-2)) & (index.hour <= (index1.hour+2))):
                            #this means successful detection of the incident
                            detection_type = 1
                            temp1 = {'cluster_id':key,'time':index1}
                            detection_GT.append(temp1)
                        elif((index.hour >= 6) & (index.hour <= 10) or
                            (index.hour >= 16) & (index.hour <= 18)):
                            #this means detected an incident
                            detection_type = 2
                        else:
                            detection_type =3
                        break
                temp = {'cluster_id':key,'time':index,'detection_type':detection_type}
                actual_detection.append(temp)

        actual_detection_Frame = pd.DataFrame(actual_detection)
        actual_detection_Frame.set_index('time',inplace = True)

        actual_detection_Frame['detection_number'] = 0
        group_actual_detection_Frame = actual_detection_Frame.groupby(['cluster_id'])
        for key1, group in group_actual_detection_Frame:
            prev = None
            detection = 0
            group.sort_index(inplace=True)
            for index,item in group.iterrows():
                if((prev == None)):
                    prev = index
                else:
                    if((prev.month == index.month) & (prev.day == index.day)):
                        if(prev.hour == index.hour):
                            diff = index.minute - prev.minute
                            if(diff == 5):
                                group.at[index,'detection_number'] = detection
                                prev = index
                                continue
                            else:
                                detection = detection + 1
                        else:
                            H_diff = index.hour - prev.hour
                            if(H_diff == 1):
                                if((index.minute  == 0) & (prev.minute == 55)):
                                    group.at[index,'detection_number'] = detection
                                    prev = index
                                    continue
                                else:
                                    detection = detection + 1
                            else:
                                detection = detection + 1
                    else: 
                        detection = detection + 1
                    group.at[index,'detection_number'] = detection
                    prev = index
            for index1,item in group.iterrows():
                if(actual_detection_Frame[actual_detection_Frame['cluster_id'] == key1].at[index1,'detection_number'] == 0):
                    actual_detection_Frame.at[index1,'detection_number'] = item.detection_number

        report = {}
        group_by_cluster  = actual_detection_Frame.groupby('cluster_id')
        for key, group in group_by_cluster:
            report[key] = {}
            report[key]['cluster_id'] = key
            total_actual_incident = len(testing_incident_GT_Clist[testing_incident_GT_Clist['cluster_head']==key])
            
            report[key]['total_actual_incident'] = total_actual_incident

            group = group[~group.index.duplicated(keep='first')]
            total = len(list(group['detection_number'].unique()))
            incident_frame = testing_incident_GT_Clist[testing_incident_GT_Clist['cluster_head']==key]
            count = 0
            report[key]['incident_frame'] = len(incident_frame)

            temp = group[group['detection_type'] == 1]
            for index,row in incident_frame.iterrows():
                focused_window = temp[(temp.index.month == index.month)&
                                            (temp.index.day == index.day)&
                                            (temp.index.hour >= (index.hour - 2))&
                                            (temp.index.hour <= (index.hour + 2))]
                if(len(focused_window)>0):
                    count = count + 1
            detection = len(list(group[group['detection_type'] == 1]['detection_number'].unique()))
            c_detection = len(list(group[group['detection_type'] == 2]['detection_number'].unique()))
            fa_alarm  = total - detection - c_detection
            report[key]['results'] = {'total': total, 'detection': detection, 'c_detection': c_detection, 'fa_alarm': fa_alarm, 'count': count}

        # Saving and backing up
        fp = os.path.join(synth_results, f'{_kappa}_optimized_results_{new_filename}.pkl')
        with open(fp, 'wb') as handle:
            pickle.dump(report, handle)

        fp = os.path.join(synth_results, f'{_kappa}_optimized_actual_detection_frame_{new_filename}.pkl')
        actual_detection_Frame.to_pickle(fp)

        fp = os.path.join(synth_results, f'{_kappa}_optimized_detection_report_Frame_{new_filename}.pkl')
        detection_report_Frame.to_pickle(fp)

def find_average_gap_between_alarms(cluster_list, detection_frame, granularity=5):
    average_false_alarm_gap = []
    
    for c_head in cluster_list:
        test_df = detection_frame[detection_frame['cluster_id'] == c_head].sort_index()
        if test_df.empty:
            continue
        idx = pd.date_range(start=test_df.iloc[0].name, end=test_df.iloc[-1].name, freq=f'{granularity}min')
        test_df.index = pd.DatetimeIndex(test_df.index)
        test_df = test_df.reindex(idx, fill_value=-1)
        
        test_df['detection_number2'] = -1
        detection_number = 0
        last_alarm = None
        first_alarm = None

        for i, g in test_df.groupby([(test_df.detection_type != test_df.detection_type.shift()).cumsum()]):
            if g.detection_type.mean() in [-1, 1, 2]:
                continue
            
            g.detection_number2 = detection_number
            detection_number += 1
            first_alarm = g.iloc[0].name
            if last_alarm:
                diff =  first_alarm - last_alarm
                if diff <= pd.Timedelta('15h'):
                    average_false_alarm_gap.append(diff)
                last_alarm = g.iloc[-1].name
            else:
                last_alarm = g.iloc[-1].name

    gaps = [gap.total_seconds() / 60 for gap in average_false_alarm_gap]

    return gaps

def graph_kappa_results():
    fp = os.path.join(synth_results, f'used_clusters_list_{new_filename}.pkl')
    with open(fp, 'rb') as handle:
        cluster_list = pickle.load(handle)

    kappa_results_list = []
    for _kappa in kappa_L:
        fp = os.path.join(synth_results, f'{_kappa}_optimized_results_{new_filename}.pkl')
        with open(fp, 'rb') as handle:
            report = pickle.load(handle) 
            
        for cluster in cluster_list:
            if cluster not in report:
                continue
            rep = report[cluster]
            kappa_results_list.append({'kappa': _kappa,
                                'total_actual_incidents': rep['total_actual_incident'],
                                'c_detections': rep['results']['c_detection'],
                                'counts'      : rep['results']['count'],
                                'detections'  : rep['results']['detection'],
                                'fa_alarms'   : rep['results']['fa_alarm'],
                                'totals'      : rep['results']['total'],
                                'attempts'    : detection_attempts
                                })

    kappa_df = pd.DataFrame(kappa_results_list)

    tp = kappa_df['counts']
    tn = kappa_df['attempts'] - kappa_df['total_actual_incidents'] - kappa_df['fa_alarms']
    fp = kappa_df['fa_alarms']
    fn = kappa_df['total_actual_incidents'] - kappa_df['counts']

    kappa_df['TPR'] =  tp / (tp + fn)
    kappa_df['FPR'] = fp / (fp + tn)

    beta = 0.9
    kappa_df['precision'] = tp / (tp + fp)
    kappa_df['recall'] = tp / (tp + fn)
    kappa_df['f_beta'] = (1 + beta**2) * (kappa_df['precision'] * kappa_df['recall']) / (beta**2 * kappa_df['precision'] + kappa_df['recall'])

    metrics_df = kappa_df.groupby('kappa').aggregate({'total_actual_incidents': [np.sum, np.mean],
                                                        'c_detections':         [np.sum, np.mean],
                                                        'counts':               [np.sum, np.mean],
                                                        'detections':           [np.sum, np.mean],
                                                        'fa_alarms':            [np.sum, np.mean],
                                                        'attempts':             [np.sum],
                                                        'totals'  :             np.sum})

    tp = metrics_df['counts']['sum']
    tn = metrics_df['attempts']['sum'] - metrics_df['total_actual_incidents']['sum'] - metrics_df['fa_alarms']['sum']
    fp = metrics_df['fa_alarms']['sum']
    fn = metrics_df['total_actual_incidents']['sum'] - metrics_df['counts']['sum']

    metrics_df['incident_detection_recall'] =  tp / (tp + fn)
    metrics_df['incident_detection_precision'] = tp / (tp + fp)
    metrics_df['false_positive_rate'] = fp / (fp + tn)
    metrics_df['true_positive_rate'] = tp / (tp + fn)

    beta = 20
    metrics_df['incident_f_beta_score'] = (1 + beta**2) * (metrics_df['incident_detection_precision'] * metrics_df['incident_detection_recall']) / (beta**2 * metrics_df['incident_detection_precision'] + metrics_df['incident_detection_recall'])

    gaps_dict = {}
    for kappa in kappa_L:
        fp = os.path.join(synth_results, f'{kappa}_optimized_actual_detection_frame_{new_filename}.pkl')
        actual_det_frame = pd.read_pickle(fp)
        gaps_dict[kappa] = find_average_gap_between_alarms(cluster_list=cluster_list, detection_frame=actual_det_frame)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # TODO: Not sure if this is the correct code
    gaps_df = []
    for k in list(gaps_dict.keys()):
        for l in gaps_dict[k]:
            gaps_df.append([k, l])
    gaps_df = pd.DataFrame(gaps_df, columns=['kappa', 'Gaps'])
    # gaps_df = pd.DataFrame(gaps_dict)
    gaps_df.boxplot(ax=ax, column='Gaps', by='kappa', fontsize=12, figsize=(8,10), showfliers=False, 
                    boxprops = dict(linestyle='-', linewidth=2),
                    whiskerprops = dict(linestyle='-', linewidth=2),
                    capprops = dict(linestyle='-', linewidth=2),
                    medianprops = dict(linestyle='-', linewidth=2))

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    ax.set_xticklabels(labels, rotation=0, fontsize=12)
    ax.set_xlabel('Kappa', fontsize=17)
    ax.set_ylabel('Time between false alarms (minutes)', fontsize=17)

    ax.tick_params(axis='both', which='major', labelsize=17, width=2, length=4)
    ax.tick_params(axis='both', which='minor', labelsize=17, width=2, length=4)
    fig.suptitle('')
    ax.set_title('Effect of Kappa on Performance and False Alarms on $i^{th}$ cluster', fontsize=17)
    fp = os.path.join(synth_figures, f'synth_Figure_8b.png')
    plt.savefig(fp, dpi=200, format='png', bbox_inches='tight')
    return kappa_df

def graph_ROC(kappa_df):
    fp = os.path.join(synth_results, f'optimized_hyper_mapping_{new_filename}.pkl')
    with open(fp, 'rb') as handle:
        hyper_mapping = pickle.load(handle)

    _, ax = plt.subplots()

    all_TPR_arr = []
    all_FPR_arr = []

    num_clusters = len(list(hyper_mapping.keys()))
    for c in range(num_clusters):
        TPR_arr = []
        FPR_arr = []
        for k_idx, k in enumerate(kappa_L):
            ith_cluster_kth_kappa_df = kappa_df.loc[[c + k_idx * num_clusters]]
            # All metrics including all detection attempts
            tp = ith_cluster_kth_kappa_df['counts']
            fp = ith_cluster_kth_kappa_df['fa_alarms']
            fn = ith_cluster_kth_kappa_df['total_actual_incidents'] - tp
            tn = detection_attempts - fp - tp - fn

            ith_cluster_kth_kappa_df['precision'] = tp / (tp + fp)
            ith_cluster_kth_kappa_df['recall']    = tp / (fn + tp)
            ith_cluster_kth_kappa_df['false_discovery_rate'] = fp / (fp + tp)
            ith_cluster_kth_kappa_df['false_positive_rate'] = fp / (fp + tn)
            ith_cluster_kth_kappa_df['true_negative_rate'] = tn / (tn + fp)
            ith_cluster_kth_kappa_df['balanced_accuracy'] = (ith_cluster_kth_kappa_df['precision'] + ith_cluster_kth_kappa_df['true_negative_rate']) / 2
            ith_cluster_kth_kappa_df['predictive_pcr'] = (tp + fp) / (tp + fp + tn + fn)

            beta = 20
            ith_cluster_kth_kappa_df['f_beta_score'] = (1 + beta**2) * (ith_cluster_kth_kappa_df['precision'] * ith_cluster_kth_kappa_df['recall']) / (beta**2 * ith_cluster_kth_kappa_df['precision'] + ith_cluster_kth_kappa_df['recall'])

            TPR = ith_cluster_kth_kappa_df['recall'].values[0]
            FPR = ith_cluster_kth_kappa_df['false_positive_rate'].values[0]
            TPR_arr.append(TPR)
            FPR_arr.append(FPR)

        all_TPR_arr.append(TPR_arr)
        all_FPR_arr.append(FPR_arr)

    all_tpr_np = np.asarray(all_TPR_arr)
    all_fpr_np = np.asarray(all_FPR_arr)

    ax.plot(np.mean(all_fpr_np, axis=0), np.mean(all_tpr_np, axis=0), marker='o', markersize=5, label='ROC')

    ax.set_ylabel('True Positive Rate', fontsize=17)
    ax.set_xlabel('False Positive Rate', fontsize=17)
    ax.legend(fontsize=17, loc=4)
    ax.tick_params(axis='both', which='major', labelsize=15, width=2, length=4)
    ax.tick_params(axis='both', which='minor', labelsize=15, width=2, length=4)
    ax.grid(True)

    fp = os.path.join(synth_figures, 'synth_Figure_8a.png')
    plt.savefig(fp, format='png', dpi=200, bbox_inches='tight')

def graph_ith_cluster(kappa_df):
    ith_cluster_df = kappa_df.loc[range(0, kappa_df.shape[0], kappa_df.shape[0]//len(kappa_L))].sort_values(by='kappa')

    # All metrics including all detection attempts
    tp = ith_cluster_df['counts']
    fp = ith_cluster_df['fa_alarms']
    fn = ith_cluster_df['total_actual_incidents'] - tp
    tn = detection_attempts - fp - tp - fn

    ith_cluster_df['precision'] = tp / (tp + fp)
    ith_cluster_df['recall']    = tp / (fn + tp)
    ith_cluster_df['false_discovery_rate'] = fp / (fp + tp)
    ith_cluster_df['false_positive_rate'] = fp / (fp + tn)
    ith_cluster_df['true_negative_rate'] = tn / (tn + fp)
    ith_cluster_df['balanced_accuracy'] = (ith_cluster_df['precision'] + ith_cluster_df['true_negative_rate']) / 2
    ith_cluster_df['predictive_pcr'] = (tp + fp) / (tp + fp + tn + fn)

    beta = 20
    ith_cluster_df['f_beta_score'] = (1 + beta**2) * (ith_cluster_df['precision'] * ith_cluster_df['recall']) / (beta**2 * ith_cluster_df['precision'] + ith_cluster_df['recall'])
    ith_cluster_df = ith_cluster_df[ith_cluster_df['kappa'] != 0.0]
    ith_cluster_df['missed'] = 1 - ith_cluster_df['recall']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 9), ith_cluster_df['missed'], marker='o', label='Missed Detection Rate', markersize=10)
    ax.plot(range(1, 9), ith_cluster_df['false_positive_rate'], marker='^', label='False Positive Rate', markersize=10)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel('Kappa', fontsize=17)
    ax.set_ylabel('Percent', fontsize=17)

    ax.tick_params(axis='both', which='major', labelsize=17, width=2, length=4)
    ax.tick_params(axis='both', which='minor', labelsize=17, width=2, length=4)

    ax.legend(fontsize=17)
    fig.suptitle('')
    ax.set_title('Effect of Kappa on Performance and False Alarms on $i^{th}$ cluster', fontsize=17)

    fp = os.path.join(synth_figures, f'synth_Figure_7a.png')
    plt.savefig(fp, dpi=200, format='png', bbox_inches='tight')

if __name__ == "__main__":
    print()
    print()
    print("--Anomaly based Incident Detection in Large Scale Smart Transportation Systems--")
    print("for ICCPS2022 Artifact Evaluation...")
    print()
    print()

    print("0/6:Starting to generate table and graphs...")
    print("1/6:Generating Clustering Data Table...")
    #clustering_table_2()

    print("2/6:Generating map and clusters...")
    cluster_graph()

    print("3/6:Generating Figure 5...")
    generate_figure_5()

    print("4/6:Generating and graphing Kappa tests...")
    test_all_kappas()
    kappa_df = graph_kappa_results()

    print("5/6:Graph ROC...")
    graph_ROC(kappa_df=kappa_df)

    print("6/6:Graph for ith cluster...")
    graph_ith_cluster(kappa_df=kappa_df)

    print("Finished graphing everything...")
