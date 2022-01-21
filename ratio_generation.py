import warnings
warnings.filterwarnings('ignore')
import os
import random
import networkx as nx
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import math
import time
import pickle

from copy import deepcopy
from scipy.stats.stats import pearsonr
from scipy.stats import hmean

from shapely.geometry import Point, Polygon, LineString

from src import network_graphing as net_graph
from src import data_processing as data_proc
from src import cleaning_data
from src.utils import Read_DF, Call_Back

if not os.path.exists(os.path.join(os.getcwd(), 'synthetic_data')):
    raise OSError("Must first download data, see README.md")
synth_data = os.path.join(os.getcwd(), 'synthetic_data')

if not os.path.exists(os.path.join(os.getcwd(), 'synthetic_results')):
    raise OSError("Must first download data, see README.md")
synth_results = os.path.join(os.getcwd(), 'synthetic_results')

# PARAMS:
time_start = '06:00:00'
time_end   = '21:00:00'
data_column = 'congestion_mean'
granularity = 5
months = [10, 11, 12]
days = [0, 1, 2, 3, 4] # weekday
# Creating a sub-graph full of only edges better than the lower bound
lower_bound_correlation = 0.0
correlation_threshold = 0.0

def list_all_segments_in_clusters(clusters):
    segments = []
    for k, v in clusters.items():
        segments = segments + v
    return segments

def locate_segment_cluster(segment, clusters):
    if pd.isnull(segment):
        return -1
    for i, cluster in enumerate(list(clusters)):
        if segment in clusters[cluster]:
            return cluster
    return -1

def get_correlation(sub_df, segment_1_ID, segment_2_ID, _method='pearson'):
    segment_1_data = sub_df[segment_1_ID]
    segment_2_data = sub_df[segment_2_ID]
    return segment_1_data.corr(segment_2_data, method=_method)

# Clustering Algorithm
# we use clusters_df since it is easier to find points
center = 'center_3310'

# Gets both positive and negative edge
def region(SG, u, r):
    clusters_df = generate_clusters_df()
    if clusters_df[clusters_df['segmentID'] == u].empty:
        return None
    
    u_pt = clusters_df[clusters_df['segmentID'] == u][center].values[0]
    
    try:
        u_buffer = u_pt.buffer(r)
    except ValueError as e:
        print("Error:", e)
        return (None, None)
        
    # this line causes nodes not in the network graph to be used as well.
    # I should have limited this to just the nodes found in the graph, 
    # But i dont think there is any problem including them.
    # Since we only check does present in the graph for hte correlation value
    nodes_in_region = clusters_df[(clusters_df[center].within(u_buffer)) |
                                  (clusters_df[center].intersects(u_buffer))]['segmentID'].tolist()
    node_partners_outside_region = []
    for n in nodes_in_region:
        try:
            node_partners_outside_region.extend(SG.neighbors(n))
        except Exception as e:
            pass
    node_partners_outside_region = np.setdiff1d(node_partners_outside_region, nodes_in_region) #list_2, list_1
    # yields the elements in `list_2` that are NOT in `list_1`
    
    return nodes_in_region, node_partners_outside_region

# Outside pos edge
def cut(SG, u, r, correlation_threshold):
    nodes_in_region, node_partners = region(SG, u, r)
    if not nodes_in_region:
        print("Error in cut")
        return -1
    
    cut_pairs = []
    for n_in in nodes_in_region:
        try:
            neighbors = list(SG.neighbors(n_in))
            for neighbor in neighbors:
                # I think this is safer
                if neighbor not in nodes_in_region:
                    cut_pairs.append((n_in, neighbor))
                    cut_pairs.append((neighbor, n_in))
        except Exception as e:
            pass
    
    correlations = []
    for cp in cut_pairs:
        if SG.get_edge_data(*cp, 0):
            correlation = SG.get_edge_data(*cp, 0)['correlation']
            if correlation >= correlation_threshold:
                correlations.append(correlation)
    correlations = np.asarray(correlations)
    return np.sum(correlations)

# should be inside pos edge + cut
def vol(SG, u, r, correlation_threshold):
    nodes_in_region, node_partners = region(SG, u, r)
    if not nodes_in_region:
        print("Error in Vol")
        return -1
    
    vol_pairs = []
    for n_in in nodes_in_region:
        try:
            neighbors = list(SG.neighbors(n_in))
            for neighbor in neighbors:
                if neighbor in nodes_in_region:
                    vol_pairs.append((n_in, neighbor))
                    vol_pairs.append((neighbor, n_in))
        except Exception as e:
            pass

    correlations = []
    for cp in vol_pairs:
        if SG.get_edge_data(*cp, 0):
            correlation = SG.get_edge_data(*cp, 0)['correlation']
            if correlation >= correlation_threshold:
                correlations.append(correlation)
    correlations = np.asarray(correlations)
    return np.sum(correlations) + cut(SG, u, r, correlation_threshold)

# Check all the vertices inside the region (instead of just one)
def nearest_node(SG, u, r):
    # Get all node partners that lie outside the region
    nodes_in_region, node_partners = region(SG, u, r)
    if not nodes_in_region:
        print(f"Error in nearest node {u}:{r}")
        return None, None
    
    min_distance = math.inf
    curr_partner = None
    
    for n_in in nodes_in_region:
        try:
            n_in_pt = Point(SG.nodes[n_in]['center_m'])
            neighbors = list(SG.neighbors(n_in))
            for neighbor in neighbors:
                if neighbor in nodes_in_region:
                    continue
                neighbor_pt = Point(SG.nodes[neighbor]['center_m'])
                partner_distance = n_in_pt.distance(neighbor_pt)
                if (partner_distance > r) and (partner_distance < min_distance):
                    min_distance = partner_distance
                    curr_partner = neighbor
        except Exception as e:
            pass
    return curr_partner, min_distance - r
# End Clustering Algorithm

def generate_clusters_df():
    fp = os.path.join(synth_data, 'synth_overall_means.pkl')
    df_overall_all = pd.read_pickle(fp)
    active_segments = df_overall_all.droplevel([0, 1]).reset_index()['segmentID'].unique().tolist()
    
    fp = os.path.join(synth_data, 'synth_segments_grouped.pkl')
    segments_df = pd.read_pickle(fp)
    segments_df = segments_df.set_geometry('geometry')    
   
    original_crs = 4326
    to_crs       = 3310
    
    segments_df.set_crs(epsg=original_crs, inplace=True, allow_override=True)
    clusters_df = segments_df.loc[segments_df['segmentID'].isin(active_segments)]

    line_to_crs = f'line_{to_crs}'
    center_to_crs = f'center_{to_crs}'

    clusters_df['center'] = clusters_df['geometry'].centroid
    clusters_df.set_crs(epsg=original_crs, inplace=True, allow_override=True)
    clusters_df[line_to_crs] = clusters_df['geometry']
    clusters_df = gpd.GeoDataFrame(clusters_df, geometry=clusters_df[line_to_crs])
    clusters_df[line_to_crs] = clusters_df[line_to_crs].to_crs(epsg=to_crs)

    clusters_df = clusters_df.set_geometry(line_to_crs)
    clusters_df.set_crs(epsg=to_crs, inplace=True, allow_override=True)
    clusters_df[center_to_crs] = clusters_df[line_to_crs].centroid.to_crs(epsg=to_crs)
    clusters_df = clusters_df.reset_index()
    clusters_df = clusters_df.drop(['index'], axis=1)
    
    return clusters_df

def cluster_correlation_optimization():
    fp = os.path.join(synth_data, 'synth_line_segment_graph.pkl')
    G = nx.read_gpickle(fp)
    
    fp = os.path.join(synth_data, 'synth_overall_means.pkl')
    df_overall_all = pd.read_pickle(fp)
    active_segments = df_overall_all.droplevel([0, 1]).reset_index()['segmentID'].unique().tolist()

    # Converting the df_overall_all to a usable dataframe
    datasets = 'congestion_mean' #congestion_mean
    days_in_a_week = range(7)
    week_df = []
    for day in days_in_a_week:
        week_df.append(df_overall_all.xs((day))[datasets].unstack("window").T)
    week_df = pd.concat(week_df, axis=0)
    # starts on a monday (dayofweek=0)
    _df = data_proc.convert_windows_to_dt_df(week_df, date='2019-01-07')
    _df = _df[active_segments]
    sub_df = _df[_df.index.dayofweek.isin(days)].between_time(time_start, time_end)

    clusters_df = generate_clusters_df()
    
    # loop through edges
    for e in G.edges:
        segment_id1 = e[0]
        segment_id2 = e[1]
        correlation = get_correlation(sub_df, segment_id1, segment_id2)
        centroid_m = clusters_df[clusters_df['segmentID'] == segment_id1]['center_3310'].values[0]
        G.nodes[segment_id1]['center_m'] = (centroid_m.x, centroid_m.y)
        centroid_m = clusters_df[clusters_df['segmentID'] == segment_id2]['center_3310'].values[0]
        G.nodes[segment_id2]['center_m'] = (centroid_m.x, centroid_m.y)
        G.edges[(segment_id1, segment_id2, 0)]['correlation'] = correlation

    positive_edges = [e for e in G.edges if G.get_edge_data(*e)['correlation'] >= lower_bound_correlation]
    positive_nodes = list(set([n[0] for n in positive_edges] + [n[1] for n in positive_edges]))

    largest_wcc = positive_nodes

    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
    if SG.is_multigraph():
        SG.add_edges_from((n, nbr, key, d)
            for n, nbrs in G.adj.items() if n in largest_wcc
            for nbr, keydict in nbrs.items() if nbr in largest_wcc
            for key, d in keydict.items())
    else:
        SG.add_edges_from((n, nbr, d)
            for n, nbrs in G.adj.items() if n in largest_wcc
            for nbr, d in nbrs.items() if nbr in largest_wcc)
    SG.graph.update(G.graph)
    
    edges_to_remove = []
    for e in SG.edges:
        corr_val = SG.get_edge_data(*e)['correlation']
        if corr_val < lower_bound_correlation:
            edges_to_remove.append(e)
    SG.remove_edges_from(edges_to_remove)

    # Clustering Approximation Algorithm
    logging = True
    random.seed(100)
    clustered_nodes = []
    clusters = {}
    _SG = deepcopy(SG)
    dead_center_nodes = []

    stats_dict = {'head': [], 
                  'clustered': [], 
                  'remaining': [], 
                  'time': [], 
                  'radius': [], 
                  'vol': [], 
                  'cut': []}

    while len(_SG.nodes) > 0:
        u = random.choice(list(_SG.nodes))
        if u in dead_center_nodes:
            continue
        i_s_time = time.time()

        # Avoid initializing a vol that is much larger than cut
        r = 1
        n = len(_SG.nodes)

        _cut = cut(_SG, u, r, correlation_threshold)
        _vol = vol(_SG, u, r, correlation_threshold)
        if logging:
            print(f"{0}:\t{r:.2f}, {_cut:.2f}, {_vol:.2f}")

        idx = 0
        while (_cut >= _vol) or (_cut < 0):
            _node, _dist = nearest_node(_SG, u, r)
            if not _node:
                # Dead center node
                break
            else:
                r = r + _dist + 1
                _cut = cut(_SG, u, r, correlation_threshold)
                _vol = vol(_SG, u, r, correlation_threshold)
                idx += 1
                if logging:
                    print(f"{idx}:\t{r:.2f}, {_cut:.2f}, {_vol:.2f}")
                i_e_time = time.time() - i_s_time

        if _vol < _cut:
            # Dead center node
            dead_center_nodes.append(u)
            i_e_time = time.time() - i_s_time
            if np.isinf(_dist):
                print(f"Center: {u} cannot be clustered, reached {r:.2f} limit, {_vol:.2f}<{_cut:.2f}. Time: {i_e_time:.2f}")
            else: 
                print(f"Center: {u} cannot be clustered, vol < cut. Time: {i_e_time:.2f}")
            continue
        else:
            ck, _ = region(_SG, u, r)
            if u not in ck:
                ck.append(u)

            ck = np.setdiff1d(ck, clustered_nodes).tolist() #list_2, list_1
            clustered_nodes.extend(ck)
            _SG.remove_nodes_from(clustered_nodes)

            i_e_time = time.time() - i_s_time
            if len(ck) > 0:
                clusters[u] = ck
                print(f"Center: {u}, Clustered: {len(ck)}, Remaining:{len(_SG.nodes)}/{len(SG.nodes)}, radius: {r:.2f}, Time: {i_e_time:.2f}")
                stats_dict['head'].append(u)
                stats_dict['clustered'].append(len(ck))
                stats_dict['remaining'].append(len(_SG.nodes))
                stats_dict['time'].append(i_e_time)
                stats_dict['radius'].append(r)
                stats_dict['vol'].append(_vol)
                stats_dict['cut'].append(_cut)
            else:
                print(f"Center: {u} cannot be clustered, No segments inside. Time: {i_e_time:.2f}")

    print("1/5:Saved clustering as: synth_clustering.pkl")
    fp = os.path.join(synth_data, f'synth_clustering.pkl')
    with open(fp, 'wb') as handle:
        pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fp = os.path.join(synth_data, 'synth_all_incidents_ground_truth.pkl')
    all_incidents_gt = pd.read_pickle(fp)
    all_incidents_gt['cluster_head'] = all_incidents_gt['segmentID'].apply(lambda x: locate_segment_cluster(x, clusters))
    all_incidents_gt = all_incidents_gt[all_incidents_gt['cluster_head'] != -1]
    
    print("2/5:Saved ground truth incidents for clusters as: synth_cluster_ground_truth.pkl")
    fp = os.path.join(synth_data, 'synth_cluster_ground_truth.pkl')
    all_incidents_gt.to_pickle(fp)
    
def cluster_ratio_generation():
    print("Starting ratio generation...")
    str_start_time = '00:00:00'
    str_end_time = '23:59:59'

    fp = os.path.join(synth_data, f'synth_clustering.pkl')
    with open(fp, 'rb') as handle:
        clusters = pickle.load(handle)
        
    for month in months:
        fp = os.path.join(synth_data, f'synth_Reduced_DF_2019_{month}_1.pkl')
        DF_All = pd.read_pickle(fp)
        DF_All['time'] = DF_All['time_local'].tolist()
        DF_All = DF_All.set_index('time_local')
        DF_All = DF_All.between_time(str_start_time, str_end_time)

        cluster_dfs = []
        for cluster_head in clusters:
            cluster = clusters[cluster_head]
            a = DF_All[DF_All['segmentID'].isin(cluster)].reset_index().dropna()

            # This is to guard from very incomplete clusters
            existing_segs = a['segmentID'].unique().tolist()
            if (len(existing_segs)/len(cluster)) < 0.5:
                a[:] = np.nan

            a = a.groupby(['time_local', 'segmentID']).agg({"speed_mean": ['mean']})
            a = a.droplevel(0, axis=1)
            a = a.groupby(['time_local']).agg({"mean": [hmean, 'mean']})

            # Added this for granularity purposes (hopefully it is right)
            if not a.empty:
                a = a.resample(f"{granularity}T").mean()

            a['qr'] = a['mean']['hmean']/a['mean']['mean']
            a = a.droplevel(0, axis=1) 
            a = a.rename(columns={'hmean':f"hm_{cluster_head}",
                                  'mean' :f"am_{cluster_head}",
                                  ''     :f"qr_{cluster_head}"})
            cluster_dfs.append(a)

        df = pd.concat(cluster_dfs, axis=1)
        print(f"Saving ratio of incidents_24h for month: {month}")
        fp = os.path.join(synth_data, f'synth_{str(month).zfill(2)}_2019_ratios_gran_{granularity}_incidents_24h.pkl')
        df.to_pickle(fp)

def cluster_ratio_cleaning():
    print("Starting cluster ratio cleaning...")

    window = 0.5 # incident search window (hours before and hours after)
    order = 1 # For finding valleys
    slope_threshold_modifier = 1 # Adjustment for detecting the slope as a change point
    
    fp = os.path.join(synth_data, f'synth_clustering.pkl')
    with open(fp, 'rb') as handle:
        clusters = pickle.load(handle)
        
    fp = os.path.join(synth_data, 'synth_cluster_ground_truth.pkl')
    df_incidents_gt = pd.read_pickle(fp)

    # Loop through by month
    for month in months:
        incidents = df_incidents_gt[df_incidents_gt.index.month == month]
        df_incidents_gt['cluster_head'] = df_incidents_gt['segmentID'].apply(lambda x: locate_segment_cluster(x, clusters))
        df_incidents_gt = df_incidents_gt[df_incidents_gt['cluster_head'] != -1]
        # df_incidents_gt.to_pickle(f"{cluster_version}_ground_truth.pkl")
        df_incidents_gt = df_incidents_gt.between_time('06:00:00', '21:00:00')

        fp = os.path.join(synth_data, f'synth_{str(month).zfill(2)}_2019_ratios_gran_{granularity}_incidents_24h.pkl')
        dirty = pd.read_pickle(fp)

        for dtindex, incident in incidents.iterrows():
            dtindex = dtindex.round(f'{granularity}min')
            cluster_head = incident['cluster_head']
            incident_month = dtindex.month
            incident_day = dtindex.day
            for_cleaning = dirty[dirty.index.day == incident_day][f"qr_{cluster_head}"]
            slope_diff = cleaning_data.compute_slope_difference(for_cleaning)
            before, after = cleaning_data.get_window_around_incident(dtindex, window=window)
            valid_valleys, peaks = cleaning_data.get_peaks_valleys(dtindex, slope_diff, 
                                                                   before, after, 
                                                                   modifier=slope_threshold_modifier, window=window, order=order)

            if valid_valleys:
                if peaks:
                    if min(valid_valleys) > max(peaks):
                        temp = peaks
                        peaks = valid_valleys
                        valid_valleys = temp

                    replacement_cma = cleaning_data.get_cma(for_cleaning, min(valid_valleys), max(peaks), method='cwma')
                    dirty.loc[slope_diff.index.values[min(valid_valleys)]:slope_diff.index.values[max(peaks) + 1], f"qr_{cluster_head}"] = replacement_cma.tolist()

        
        # Leave only the "QR" columns
        filter_col = [col for col in dirty if col.startswith('qr')]
        cluster_heads = [int(ch[3:]) for ch in filter_col]
        dirty = dirty[filter_col]
        dirty.columns = cluster_heads
        
        print(f"Saving ratio of incidents_cleaned for month: {month}")
        fp = os.path.join(synth_data, f'synth_{str(month).zfill(2)}_2019_ratios_gran_{granularity}_incidents_cleaned.pkl')
        dirty.to_pickle(fp)
        
    print("Further cleaning the incidents ratios.")
    for month in months:
        fp = os.path.join(synth_data, f'synth_{str(month).zfill(2)}_2019_ratios_gran_{granularity}_incidents_24h.pkl')
        dirty = pd.read_pickle(fp)
        dirty.between_time('06:00', '09:00').head()

        filter_col = [col for col in dirty if col.startswith('qr')]
        cluster_heads = [int(ch[3:]) for ch in filter_col]
        dirty = dirty[filter_col]
        dirty.columns = cluster_heads
        print(f"Saving ratio of incidents for month: {month}")
        fp = os.path.join(synth_data, f'synth_{str(month).zfill(2)}_2019_ratios_gran_{granularity}_incidents.pkl')
        dirty.to_pickle(fp)

def reset_files():
    exception = ['synth_overall_means.pkl',
                 'synth_segments_grouped.pkl',
                 'synth_line_segment_graph.pkl',
                 'synth_all_incidents_ground_truth.pkl',
                 'synth_Reduced_DF_2019_10_1.pkl',
                 'synth_Reduced_DF_2019_11_1.pkl',
                 'synth_Reduced_DF_2019_12_1.pkl']
    
    for file in os.listdir(synth_data):
        filename = os.fsdecode(file)
        if filename not in exception:
            fp = os.path.join(synth_data, filename)
            os.remove(fp)

if __name__ == '__main__':
    print("--Anomaly based Incident Detection in Large Scale Smart Transportation Systems--")
    print("for ICCPS2022 Artifact Evaluation...")
    print()
    print()    
    print()    
    print()    
    print("Deleting previously generated files...")
    reset_files()
    
    print("0/5: Starting file generation...")
    cluster_correlation_optimization()
    print("3/5:Finished Ratio Generation...")
    cluster_ratio_generation()
    print("4/5:Finished Ratio Generation...")
    cluster_ratio_cleaning()
    print("5/5:Finished Ratio Generation...")
    print("Done generating files for the first part...")    
    print()    
    print("Continuing to training...")
