import numpy as np
import pandas as pd
from itertools import combinations
from pprint import pprint
import random
from tqdm.notebook import tqdm
import math
import networkx as nx

class Clusters_Creator(object):
    def __init__(self, hist_data=None, segment_G=None, clusters_df=None):
        self._df = hist_data
        self._dfcols = self._df.columns.tolist()
        self.segment_G = segment_G
        self.clusters_df = clusters_df
        self.pairwise_df = None
        pass
    
    def generate_hops_df(self):
        print(f"Generating hops_df...")
        apsp = nx.all_pairs_shortest_path_length(self.segment_G, cutoff=None)

        hops = {}
        pbar = tqdm(total=len(self._dfcols))
        for sp in apsp:
            start_seg = sp[0]
            if start_seg not in self._dfcols:
                continue
        #     print(sp[0], len(sp[1]))
            _paths_dict = sp[1]
            hops[start_seg] = []
            for segment in self._dfcols:
                if segment in _paths_dict:
                    length = _paths_dict[segment]
                else:
                    length = -1
                hops[start_seg].append(length)
            pbar.update(1)
        hops_df = pd.DataFrame(hops, index=self._dfcols)
        self.hops_df = hops_df
        pbar.close()
        
    def generate_corr_df(self):
        print(f"Generating corr_df...")
        dataCorr = self._df.corr()
        np.fill_diagonal(dataCorr.values, np.nan)
        self.corr_df = dataCorr
        
    def generate_dist_df(self):
        print(f"Generating dist_df...")
        centers_df = self.clusters_df[self.clusters_df['XDSegID'].isin(self._dfcols)][['XDSegID', 'center_m']]
        src_center = centers_df.iloc[0]['center_m']
        centers_df['distance'] = centers_df['center_m'].apply(lambda x: src_center.distance(x))

        distance = {}
        pbar = tqdm(total=len(self._dfcols))
        for i, segment in enumerate(self._dfcols):
            src_center = centers_df[centers_df['XDSegID'] == segment]['center_m'].values[0]
            center_list = centers_df['center_m'].apply(lambda x: src_center.distance(x)).tolist() 
            distance[segment] = center_list
            pbar.update(1)
        pbar.close()

        self.dist_df = pd.DataFrame(distance, index=centers_df['XDSegID'].tolist())

    def generate_dataframes(self):
        self._dfcols = self._df.columns.tolist()
        self.generate_corr_df()
        self.generate_hops_df()
        self.generate_dist_df()
        self.pairwise_df = None
        print("Done generating dataframes...")
        
    def load_dataframes(self, hops_df=None, corr_df=None, dist_df=None):
        self.corr_df = corr_df
        self.hops_df = hops_df
        self.dist_df = dist_df
        print("Done loading dataframes...")
    
    def count_unused_segments(self, clusters):
        used_segments = []
        for k, v in clusters.items():
            used_segments = used_segments + v
        main_list = np.setdiff1d(self._dfcols, used_segments)
        return len(main_list)

    def get_clusters_info(self, clusters):
        info = {}
        info['number_of_clusters'] = len(clusters)
        segment_counts = []
        average_cluster_hops = []
        average_cluster_dist = []
        average_cluster_corr = []
        for k, v in clusters.items():
            segment_counts.append(len(v))
            combos = list(combinations(v, 2))
#             print(combos)
            ave_hops = []
            ave_dist = []
            ave_corr = []
            for combo in combos:
                if (combo[0] not in self.hops_df.columns) or \
                   (combo[0] not in self.dist_df.columns) or \
                   (combo[0] not in self.corr_df.columns):
                    continue
                hops = self.hops_df[combo[0]][combo[1]]
                dist = self.dist_df[combo[0]][combo[1]]
                corr = self.corr_df[combo[0]][combo[1]]
                ave_hops.append(hops)
                ave_dist.append(dist)
                ave_corr.append(corr)
            average_cluster_hops.append(np.mean(ave_hops) if np.mean(ave_hops) > 0 else 0)
            average_cluster_dist.append(np.mean(ave_dist) if np.mean(ave_dist) > 0 else 0)
            average_cluster_corr.append(np.mean(ave_corr) if np.mean(ave_corr) > 0 else 0)

        info['unused_segments'] = self.count_unused_segments(clusters)
        info['used_segments'] = len(self._dfcols) - info['unused_segments']
        
        info['average_segments_per_cluster']    = f"{np.mean(segment_counts):.2f}"
        info['average_hops_per_cluster']        = f"{np.mean(average_cluster_hops):.2f}"
        info['average_distance_per_cluster']    = f"{np.mean(average_cluster_dist):.2f}"
        info['average_correlation_per_cluster'] = f"{np.mean(average_cluster_corr):.2f}"
        return info

    # This function verifies pair wise correlation for each combination of segments within a cluster
    # It will only retain the pairs without any correlation lower than the threshold
    # This is used in conjunction with the next function if the param "strict" is set.
    def pairwise_correlation_clustering(self, main_segment, segments, params):
        combos = list(combinations(segments, 2))
        valid_segments = {'first': [], 'second': [], 'corr': []}
        under_threshold = []
        for combo in combos:
            corr = self.corr_df[combo[0]][combo[1]]
            valid_segments['first'].append(combo[0])
            valid_segments['second'].append(combo[1])
            valid_segments['corr'].append(corr)

        df = pd.DataFrame(valid_segments)
        df = df[['first', 'corr']].groupby(['first']).mean().sort_values(by='corr', ascending=False)
        df = df[df['corr'] >= params['correlation_threshold']]
        self.pairwise_df = df
        segs = [main_segment] + df.index.unique().tolist()
        return segs

    def start_clustering(self, params):
        print("Starting clustering...")
        pprint(params)
        
        if params['strict_correlations'] and (params['correlation_threshold'] > 0.90):
            print("Strict correlations is on while correlation threshold is > 0.90")
            print("Please turn off strict correlations or reduce correlation threshold.")
            return False
        
        segments = self._dfcols
        if params['random']:
            print("Randomizing segments...")
            random.seed(params['seed'])
            random.shuffle(segments)

        clusters = {}
        
        pbar = tqdm(total=len(segments))

        used_segments = []
        for segment in segments:
            if segment in used_segments:
                continue
                
            df = self.create_segment_df(segment)
            if isinstance(df, bool):
                if not df:
                    continue

            if not params['distance_threshold']:
                params['distance_threshold'] = math.inf
            if not params['hops_threshold']:
                params['hops_threshold'] = math.inf

            select = (df['distance'] < params['distance_threshold']) & \
                     (df['correlation'] > params['correlation_threshold']) & \
                     (df['hops'] < params['hops_threshold']) & \
                     (~df.index.isin(used_segments))
            
            if params['sensor_count_threshold']:
                cluster = df[select].sort_values(by='correlation', ascending=False).head(params['sensor_count_threshold']).index.tolist()
            else:
                cluster = df[select].sort_values(by='correlation', ascending=False).index.tolist()

            # Additional checking for pair-wise correlations instead of just a single one-to-all comparison
            if params['strict_correlations']:
                cluster = self.pairwise_correlation_clustering(segment, cluster, params)
            else:
                cluster.insert(0, segment)
                
            if not params['minimum_sensor_count']:
                params['minimum_sensor_count'] = 1
            if len(cluster) < params['minimum_sensor_count']:
                continue

            pbar.update(len(cluster))
            used_segments = used_segments + cluster
            clusters[segment] = cluster

        pbar.update(len(segments) - len(used_segments))
        pbar.close()
        return clusters

    def create_segment_df(self, segment):
#         print(f"Creating segment {segment} dataframe...")
        if segment not in self.hops_df.columns:
            return False
        if segment not in self.dist_df.columns:
            return False
        if segment not in self.corr_df.columns:
            return False

        _hops = self.hops_df[segment].rename("hops")
        _corr = self.corr_df[segment].rename("correlation")
        _dist = self.dist_df[segment].rename("distance")
        result = pd.concat([_hops, _corr, _dist], axis=1)
#         display(result)
        return result

    # Adding neighbors column
    def add_neighbors_column(self, clusters_df):
        clusters_df['neighbors'] = 0
        for k, segment in clusters_df.iterrows():
            seg_id = segment['XDSegID']
            if seg_id in list(segment_G.nodes):
                neighbors = list(segment_G.neighbors(seg_id))
                clusters_df.at[k, 'neighbors'] = len(neighbors)
        #     rche_df.loc[index, 'wgs1984_latitude'] = dict_temp['lat']
    #     clusters_df.sort_values(by='neighbors', ascending=False).head()
        return clusters_df