import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
import time
import sys
import pickle
from src.common_functions import *

print()
print()
print("--Anomaly based Incident Detection in Large Scale Smart Transportation Systems--")
print("for ICCPS2022 Artifact Evaluation...")

# Params
start_time = '06:00'
end_time   = '20:55'
training_months = 10
cross_validation_months = 11
testing_months = 12
lower_bound_correlation = 0.0
correlation_threshold = 0.0
# Adjust the number of clusters
NUMBER_OF_CLUSTERS = 10

# Confirm directories are in place
if not os.path.exists(os.path.join(os.getcwd(), 'synthetic_data')):
    raise OSError("Must first download data, see README.md")
synth_data = os.path.join(os.getcwd(), 'synthetic_data')

if not os.path.exists(os.path.join(os.getcwd(), 'synthetic_results')):
    raise OSError("Must first download data, see README.md")
synth_results = os.path.join(os.getcwd(), 'synthetic_results')

info_ratio = []
for file in os.listdir(synth_data):
    filename = os.fsdecode(file)
    if 'cleaned' in filename:
        fp = os.path.join(synth_data, filename)
        df = pd.read_pickle(fp)
        info_ratio.append(df)

combined_ratio_frame = pd.concat(info_ratio)
combined_ratio_frame = combined_ratio_frame.between_time(start_time, end_time)
combined_ratio_frame = combined_ratio_frame[combined_ratio_frame.index.month == training_months]
training = combined_ratio_frame

fp = os.path.join(synth_data, f'synth_clustering.pkl')
with open(fp, 'rb') as handle:
    clusters = pickle.load(handle)

fp = os.path.join(synth_data, f'synth_cluster_ground_truth.pkl')
incident_GT_Frame = pd.read_pickle(fp)

_df = incident_GT_Frame.groupby('cluster_head').sum()\
                       .sort_values('Total_Number_Incidents', ascending=False)

# Only pick cluster_heads which are present in the {version}_clusters.pkl
_df = _df[_df.index.isin(list(clusters.keys()))].head(NUMBER_OF_CLUSTERS)
cluster_list = _df.index.tolist()

new_filename = "synth"
fp = os.path.join(synth_results, f'used_clusters_list_{new_filename}.pkl')
with open(fp, 'wb') as handle:
    pickle.dump(cluster_list, handle)    

def reset_files():
    exception = ['used_clusters_list_synth.pkl',
                 '.gitignore']

    for file in os.listdir(synth_results):
        filename = os.fsdecode(file)
        if filename not in exception:
            fp = os.path.join(synth_results, filename)
            os.remove(fp)
            
if __name__ == "__main__":
    print()    
    print()    
    print("0/8: Starting training, this might take a few minutes...")
    print("Deleting previous results...")
    reset_files()
    
    ############ Step 1 ############
    time_start = time.time()

    training_cluster_list = training[cluster_list]
    training_cluster_list.columns = cluster_list
    Q_mean_list = {} # Qmean for each of the cluster 
    for column in training_cluster_list:
        Q_mean_list[column] = {}
        mad = training_cluster_list[column].mad()
        std = training_cluster_list[column].std()
        median = training_cluster_list[column].median()
        grouped = training_cluster_list[column].groupby([training_cluster_list[column].index.hour,
                                                         training_cluster_list[column].index.minute])
        Q_mean = {}
        for key,group in grouped:
            Q_mean[key] = group.mean()
        Q_mean_list[column]['Q_mean'] = Q_mean
        Q_mean_list[column]['mad'] = mad
        Q_mean_list[column]['std'] = std
        Q_mean_list[column]['median'] = median

    # generate safe_margin for all values of kappa
    kappa_L = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    safe_margin = {}
    for key in Q_mean_list.keys():
        safe_margin[key] = {}
        for k in kappa_L:
            safe_margin[key][k] = {'upper':{},'lower':{}}
            mad = Q_mean_list[key]['std']

            Q_mean = Q_mean_list[key]['Q_mean']
            for key1 in Q_mean.keys(): 
                safe_margin[key][k]['upper'][key1] = Q_mean[key1] + mad * k
                safe_margin[key][k]['lower'][key1] = Q_mean[key1] - mad * k

    fp = os.path.join(synth_results, f'optimized_safe_margin_{new_filename}.pkl')
    with open(fp, 'wb') as handle:
        pickle.dump(safe_margin, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'1/8:Saved optimized_safe_margin_{new_filename}.pkl')

    residual = {}

    for column in training_cluster_list.columns:
        grouped = training_cluster_list[column].groupby([training_cluster_list[column].index.hour,
                                                         training_cluster_list[column].index.minute])
        sm_per_C = safe_margin[column]
        R_per_C = {}
        for key in sm_per_C.keys():
            nabla_dict = calculate_nabla(grouped, sm_per_C[key])

            nabla_frame = pd.DataFrame(list(nabla_dict.items()),columns = ['time','nabla'])
            nabla_frame.set_index('time', inplace=True)
            SF_List = [3,5,7,9]
            RUC = {}
            for sf in SF_List:
                RUC[sf] = faster_calculate_residual(nabla_frame,sf)
            R_per_C[key] = RUC

        residual[column] = R_per_C

    fp = os.path.join(synth_results, f'optimized_residual_train_{new_filename}.pkl')
    with open(fp, 'wb') as handle:
        pickle.dump(residual, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'2/8:Saved optimized_residual_train_{new_filename}.pkl')

    ############ Step 2 ############
    print("Training residuals, this takes a bit of time...")

    fp_residual = os.path.join(synth_results, f'optimized_residual_train_{new_filename}.pkl')
    with open(fp_residual, 'rb') as handle:
        residual = pickle.load(handle)

    fp_safe_margin = os.path.join(synth_results, f'optimized_safe_margin_{new_filename}.pkl')
    with open(fp_safe_margin, 'rb') as handle:
        safe_margin = pickle.load(handle)

    residual_filtered = {}
    for key in residual.keys():
        if(key in cluster_list):
            residual_filtered[key] = residual[key]

    safe_margin_filtered = {}
    for key in safe_margin.keys():
        if(key in cluster_list):
            safe_margin_filtered[key] = safe_margin[key]

    df = pd.DataFrame.from_dict(residual_filtered, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    indices = df.index.tolist()
    sf_keys = df.columns.tolist()

    standard_limit = []

    for index in indices:
        for sf_key in sf_keys:
            _df = pd.DataFrame.from_dict(df.loc[index][sf_key].items())
            _df = _df.rename(columns={0:'time', 1: 'nabla'})
            _df.set_index('time', inplace=True)
            T_max = calculate_tmax(_df['nabla'])
            T_min = calculate_tmin(_df['nabla'])
            temp = {'cluster_id':index[0],
                    'ka ppa':index[1],
                    'SF':sf_key,
                    'tau_max':T_max, 'tau_min':T_min}
            standard_limit.append(temp)

    # Saving and backing up
    fp = os.path.join(synth_results, f'optimized_standard_limit_{new_filename}.pkl')
    with open(fp, 'wb') as handle:
        pickle.dump(standard_limit, handle)
        print(f'3/8:Saved optimized_standard_limit_{new_filename}.pkl')

    ############ Step 3 ############
    fp_safe_margin = os.path.join(synth_results, f'optimized_safe_margin_{new_filename}.pkl')
    with open(fp_safe_margin, 'rb') as handle:
        safe_margin = pickle.load(handle)

    info_ratio_incidents = []
    for file in os.listdir(synth_data):
        filename = os.fsdecode(file)
        if 'incidents.pkl' in filename:
            fp = os.path.join(synth_data, filename)
            df = pd.read_pickle(fp)
            info_ratio_incidents.append(df)

    combined_ratio_frame_incidents = pd.concat(info_ratio_incidents)

    combined_ratio_frame_incidents = combined_ratio_frame_incidents.between_time(start_time, end_time)
    combined_ratio_frame_incidents = combined_ratio_frame_incidents[combined_ratio_frame_incidents.index.month == cross_validation_months]

    testing = combined_ratio_frame_incidents
    testing_Clist = testing[cluster_list]
    testing_Clist.columns = cluster_list

    test_residual = {}

    for column in testing_Clist.columns:
        grouped = testing_Clist[column].groupby([testing_Clist[column].index.hour,
                                                 testing_Clist[column].index.minute])
        sm_per_C = safe_margin[column]
        R_per_C = {}
        for key in sm_per_C.keys():
            nabla_dict = calculate_nabla(grouped, sm_per_C[key])

            nabla_frame = pd.DataFrame(list(nabla_dict.items()),columns = ['time','nabla'])
            nabla_frame.set_index('time', inplace=True)
            SF_List = [3,5,7,9]
            RUC = {}
            for sf in SF_List:
                RUC[sf] = faster_calculate_residual(nabla_frame,sf)
            R_per_C[key] = RUC

        test_residual[column] = R_per_C

    # Saving and backing up
    fp = os.path.join(synth_results, f'optimized_test_residual_{new_filename}.pkl')
    with open(fp, 'wb') as handle:
        pickle.dump(test_residual, handle)
        print(f'4/8:Saved optimized_test_residual_{new_filename}.pkl')

    ############ Step 4 ############
    fp_safe_margin = os.path.join(synth_results, f'optimized_safe_margin_{new_filename}.pkl')
    with open(fp_safe_margin, 'rb') as handle:
        safe_margin = pickle.load(handle)

    fp_standard_limit = os.path.join(synth_results, f'optimized_standard_limit_{new_filename}.pkl')
    with open(fp_standard_limit, 'rb') as handle:
        standard_limit_5C = pickle.load(handle)
    standard_limit_5C_Frame = pd.DataFrame(standard_limit_5C)

    fp_test_res = os.path.join(synth_results, f'optimized_test_residual_{new_filename}.pkl')
    with open(fp_test_res, 'rb') as handle:
        test_residual = pickle.load(handle)

    testing = combined_ratio_frame_incidents.between_time(start_time, end_time)
    testing =  testing[testing.index.month == cross_validation_months]
    testing_Clist = testing[cluster_list]
    testing_Clist.columns = cluster_list

    detection_report = []
    for column in testing_Clist.columns:
        grouped = testing_Clist[column].groupby([testing_Clist[column].index.hour,
                                                 testing_Clist[column].index.minute])

        sm_per_C = safe_margin[column] # safe margin list for each cluster
        for key in sm_per_C.keys(): # for each safe margin
            for key1, group in grouped:
                group = group.dropna()

                groupDF = pd.DataFrame(group)
                groupDF['g_upper'] = groupDF[column] > sm_per_C[key]['upper'][key1]
                groupDF['l_lower'] = groupDF[column] < sm_per_C[key]['lower'][key1]
                groupDF['or'] = groupDF['g_upper'] | groupDF['l_lower']

                groupDF = groupDF[groupDF['or'] == True]
                res_SF = test_residual[column][key]
                for key2 in res_SF.keys():
                    std_limit = standard_limit_5C_Frame[(standard_limit_5C_Frame['cluster_id']== column) &
                                                        (standard_limit_5C_Frame['ka ppa']== key) &
                                                        (standard_limit_5C_Frame['SF']== key2)]
                    index_ar = std_limit.index
                    for index, row in groupDF.iterrows():
                        temp = None
                        if(res_SF[key2][index] >0):
                            if(res_SF[key2][index]>std_limit.at[index_ar[0],'tau_max']):
                                temp = {'cluster_id':column,'kappa':key,'SF':key2,
                                        'time':index,'RUC':res_SF[key2][index],'tau_max':std_limit.at[index_ar[0],'tau_max']}
                                detection_report.append(temp)
                        else:
                            if(res_SF[key2][index]<std_limit.at[index_ar[0],'tau_min']):
                                temp = {'cluster_id':column,'kappa':key,'SF':key2,'time':index,
                                        'RUC':res_SF[key2][index],'tau_min':std_limit.at[index_ar[0],'tau_min']}
                                detection_report.append(temp)

    detection_report_Frame = pd.DataFrame(detection_report)
    detection_report_Frame.set_index('time',inplace = True)

    # Saving and backing up
    fp = os.path.join(synth_results, f"optimized_detection_report_{new_filename}.pkl")
    detection_report_Frame.to_pickle(fp)
    print(f"5/8:Saved optimized_detection_report_{new_filename}.pkl")

    ############ Step 5 ############
    df = df.sort_index()
    idx = pd.date_range(pd.Timestamp('12-01-2019 00:00:00'), pd.Timestamp('12-14-2019 23:59:59'), freq="5min")

    detection_attempts = len(idx[idx.indexer_between_time('06:00', '20:55')])

    fp_detection_report = os.path.join(synth_results, f"optimized_detection_report_{new_filename}.pkl")
    with open(fp_detection_report, 'rb') as handle:
        detection_report_Frame = pickle.load(handle)

    testing_incident_GT = incident_GT_Frame.between_time(start_time, end_time)
    testing_incident_GT = testing_incident_GT[(testing_incident_GT.index.month == cross_validation_months)]
    testing_incident_GT_Clist = testing_incident_GT[testing_incident_GT['cluster_head'].isin(cluster_list)]

    group_detection_report_by_cluster_id = detection_report_Frame.groupby('cluster_id')
    group_gt_incident_cluster_head = testing_incident_GT_Clist.groupby('cluster_head')
    actual_detection = []
    detection_GT = []
    for key, gorup in group_detection_report_by_cluster_id:
        group_by_kappa_sf = gorup.groupby(['kappa','SF'])
        for (key1,key2), group in group_by_kappa_sf:
            for index,row in group.iterrows():
                detection_type = 0
                if key in group_gt_incident_cluster_head.groups.keys():
                    for index1,row1 in group_gt_incident_cluster_head.get_group(key).iterrows():
                        #iterate only incidents happend for the cluster 
                        if((index.month == index1.month) and (index.day == index1.day)):
                            #This means incident and detection are on the same day
                            if((index.hour >= (index1.hour-2)) & (index.hour <= (index1.hour+2))):
                                #this means successful detection of the incident
                                detection_type = 1
                                temp1 = {'cluster_id':key,'kappa':key1,'SF':key2,'time':index1}
                                detection_GT.append(temp1)
                            elif((index.hour >= 6) & (index.hour <= 10) or
                                (index.hour >= 16) & (index.hour <= 18)):
                                #this means detected an incident
                                detection_type = 2
                            else:
                                detection_type =3
                            break
                    temp = {'cluster_id':key,'kappa':key1,'SF':key2,'time':index,'detection_type':detection_type}
                    actual_detection.append(temp)

    actual_detection_Frame = pd.DataFrame(actual_detection)
    actual_detection_Frame.set_index('time',inplace = True)
    detection_GT_Frame = pd.DataFrame(detection_GT)
    detection_GT_Frame.set_index('time',inplace = True)

    actual_detection_Frame['detection_number'] = 0
    group_actual_detection_Frame = actual_detection_Frame.groupby(['cluster_id'])
    for key1, group in group_actual_detection_Frame:
        group_c = group.groupby(['kappa','SF'])
        for (key2, key3), grp in group_c:
            current = None
            detection = 0
            grp.sort_index(inplace=True)
            for index,item in grp.iterrows():
                if((current == None)):
                    current = index
                else:
                    if((current.month == index.month) & (current.day == index.day)):
                        if(current.hour == index.hour):
                            diff = index.minute - current.minute
                            if(diff == 5):
                                grp.at[index,'detection_number'] = detection
                                current = index
                                continue
                            else:
                                detection = detection + 1
                        else:
                            H_diff = index.hour - current.hour
                            if(H_diff == 1):
                                if((index.minute  == 0) & (current.minute == 55)):
                                    grp.at[index,'detection_number'] = detection
                                    current = index
                                    continue
                                else:
                                    detection = detection + 1
                            else:
                                detection = detection + 1
                    else: 
                        detection = detection + 1
                    grp.at[index,'detection_number'] = detection
                    current = index
            for index,item in grp.iterrows():
                actual_detection_Frame.at[index,'detection_number'] = item.detection_number

    hyper_mapping = {}
    group_actual_detection_Frame = actual_detection_Frame.groupby(['cluster_id'])
    for key1, group in group_actual_detection_Frame:
        group_c = group.groupby(['kappa','SF'])
        min_fa =  sys.maxsize
        min_decision_fa =  (-1.0)*sys.maxsize
        total_incident = len(testing_incident_GT_Clist[(testing_incident_GT_Clist['cluster_head']==key1)])
        min_missed = sys.maxsize
        print("CLUSTER: ",key1)
        print('total incident: ',total_incident)
        for (key2,key3),grp in group_c:
            valid_detection = len(list(grp[grp['detection_type'] == 1]['detection_number'].unique())) + len(list(grp[grp['detection_type'] == 2]['detection_number'].unique()))
            total_detection = len(list(grp['detection_number'].unique()))
            false_alarm = total_detection - valid_detection
            df3 = detection_GT_Frame[(detection_GT_Frame['cluster_id']==key1)&
                                                (detection_GT_Frame['kappa']==key2)&
                                                (detection_GT_Frame['SF']==key3)]
            df3 = df3[~df3.index.duplicated(keep='first')]
            detection = len(df3)
            fraction_of_detection = detection /total_incident
            # print("fraction_of_detection: ",fraction_of_detection)
            missed = abs(total_incident - detection)
    #         fraction_FA  = false_alarm/ total_detection
            fraction_FA = false_alarm / (false_alarm + detection_attempts)
            # print('fraction_FA: ',fraction_FA)
            decision_factor = fraction_of_detection - fraction_FA
            # print('decision_factor:', decision_factor)
            if((min_decision_fa < decision_factor)):
                min_decision_fa = decision_factor
                hyper_mapping[key1] = {'kappa':key2,'SF':key3}
                # print('false alarm: ',false_alarm)

    print(hyper_mapping)

    # Saving and backing up
    fp = os.path.join(synth_results, f"optimized_hyper_mapping_{new_filename}.pkl")
    with open(fp, 'wb') as handle:
        pickle.dump(hyper_mapping, handle)
        print(f"6/8:Saved optimized_hyper_mapping_{new_filename}.pkl")

    ############ Step 6 ############
    fp_safe_margin = os.path.join(synth_results, f'optimized_safe_margin_{new_filename}.pkl')
    with open(fp_safe_margin, 'rb') as handle:
        safe_margin = pickle.load(handle)

    cross_validated_kappa_SF = hyper_mapping

    info_ratio_incidents = []
    for file in os.listdir(synth_data):
        filename = os.fsdecode(file)
        if 'incidents.pkl' in filename:
            fp = os.path.join(synth_data, filename)
            df = pd.read_pickle(fp)
            info_ratio_incidents.append(df)
    combined_ratio_frame_incidents = pd.concat(info_ratio_incidents)

    combined_ratio_frame_incidents = combined_ratio_frame_incidents[combined_ratio_frame_incidents.index.month == testing_months]
    testing = combined_ratio_frame_incidents.between_time(start_time, end_time)
    testing_Clist = testing[list(cross_validated_kappa_SF.keys())]
    testing_Clist.columns = list(cross_validated_kappa_SF.keys())

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
        test_residual [column] = R_per_C

    # Saving and backing up
    fp = os.path.join(synth_results, f'optimized_residual_Test_QR_{new_filename}.pkl')
    with open(fp, 'wb') as handle:
        pickle.dump(test_residual, handle)
        print("7/8: Saved residual test...")

    ############ Step 7 ############
    fp_safe_margin = os.path.join(synth_results, f'optimized_safe_margin_{new_filename}.pkl')
    with open(fp_safe_margin, 'rb') as handle:
        safe_margin = pickle.load(handle)

    fp_standard_limit = os.path.join(synth_results, f'optimized_standard_limit_{new_filename}.pkl')
    with open(fp_standard_limit, 'rb') as handle:
        standard_limit_5C = pickle.load(handle)
    standard_limit_5C_Frame = pd.DataFrame(standard_limit_5C)

    fp_test_res = os.path.join(synth_results, f'optimized_residual_Test_QR_{new_filename}.pkl')
    with open(fp_test_res, 'rb') as handle:
        test_residual = pickle.load(handle)

    info_ratio_incidents = []
    for file in os.listdir(synth_data):
        filename = os.fsdecode(file)
        if 'incidents.pkl' in filename:
            fp = os.path.join(synth_data, filename)
            df = pd.read_pickle(fp)
            info_ratio_incidents.append(df)
    combined_ratio_frame_incidents = pd.concat(info_ratio_incidents)

    fp = os.path.join(synth_data, f'synth_cluster_ground_truth.pkl')
    incident_GT_Frame = pd.read_pickle(fp)

    cross_validated_kappa_SF = hyper_mapping

    testing_incident_GT = incident_GT_Frame.between_time(start_time, end_time)
    testing_incident_GT_Clist =  testing_incident_GT[testing_incident_GT['cluster_head'].isin (cluster_list)]
    testing_incident_GT_Clist =  testing_incident_GT_Clist[testing_incident_GT_Clist.index.month == testing_months]

    testing = combined_ratio_frame_incidents.between_time(start_time, end_time)
    testing =  testing[testing.index.month == testing_months]

    testing_Clist = testing[list(cross_validated_kappa_SF.keys())]
    testing_Clist.columns = list(cross_validated_kappa_SF.keys())

    detection_report = []
    for column in testing_Clist: #per cluster 
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
        print('Cluster Id: ',key)
        total_actual_incident = len(testing_incident_GT_Clist[testing_incident_GT_Clist['cluster_head']==key])
        print('Total Actual Incident: ',total_actual_incident)

        report[key]['total_actual_incident'] = total_actual_incident

        group = group[~group.index.duplicated(keep='first')]
        total = len(list(group['detection_number'].unique()))
        incident_frame = testing_incident_GT_Clist[testing_incident_GT_Clist['cluster_head']==key]
        count = 0
        print("incident length: ",len(incident_frame))
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
        print('total: ',total,' detection: ',detection,' c_detection: ',c_detection,' fa_alarm: ',fa_alarm,' count: ',count)
        report[key]['results'] = {'total': total, 'detection': detection, 'c_detection': c_detection, 'fa_alarm': fa_alarm, 'count': count}

    # Saving and backing up
    fp = os.path.join(synth_results, f'optimized_results_{new_filename}.pkl')
    with open(fp, 'wb') as handle:
        pickle.dump(report, handle)

    fp = os.path.join(synth_results, f'optimized_actual_detection_frame_{new_filename}.pkl')
    actual_detection_Frame.to_pickle(fp)

    fp = os.path.join(synth_results, f'optimized_detection_report_Frame_{new_filename}.pkl')
    detection_report_Frame.to_pickle(fp)

    elapsed_time = time.time() - time_start
    print(f"8/8: Done training in {elapsed_time:.3f} s")
