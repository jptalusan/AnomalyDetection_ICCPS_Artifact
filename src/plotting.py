import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Point
from scipy.stats import hmean
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from scipy.stats.stats import pearsonr
from scipy.stats import hmean
import random
from tqdm.notebook import tqdm
import pandas as pd

def plot_grid(target_area_df, grids, grid):
    polygon = grids[grid]['polygon']
    segments = grids[grid]['segments']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    target_area_df.plot(ax=ax, color='gray', linewidth=0.2)
    
    patch = PolygonPatch(polygon, fc='red', ec='k', alpha=0.2, zorder=2)
    ax.add_patch(patch)

    tb = polygon.bounds
    ax.set_ylim(tb[1]*0.99998, tb[3]*1.00002)
    ax.set_xlim(tb[0]*1.00002, tb[2]*0.99998)
    segs_df = target_area_df[target_area_df['XDSegID'].isin(segments)]
    segs_df.plot(ax=ax)
    # plt.savefig('mygrouping.png', dpi=200)
    
    for k, row in segs_df.iterrows():
        centroid = row['geometry'].centroid
        ax.plot(*centroid.centroid.xy, marker='o')
        
#         start_pt = Point(row['StartLong'], row['StartLat'])
#         end_pt = Point(row['EndLong'], row['EndLat'])
    
#         ax.plot(*start_pt.centroid.xy, marker='o')
#         ax.plot(*end_pt.centroid.xy, marker='^')
        
    return ax

def plot_DF_All_speeddata(single_day_df, segments, time_start=None, time_end=None, figsize=(20, 5), show_segments=True):
    fig, ax = plt.subplots(figsize=figsize)
    print(f"No. of segments: {len(segments)}")

    _temp = single_day_df[single_day_df['XDSegID'].isin(segments)]
    _temp = _temp.groupby(['window'])['speed_mean'].aggregate(hmean, axis=None).tolist()
    datetime_arr = single_day_df.index.unique()
    ax.plot(datetime_arr, _temp, marker='o', label='aggregate', color='blue', markersize=4, markevery=5)

    if show_segments:
        for segment in segments:
            _temp = single_day_df[single_day_df['XDSegID'].isin([segment])]
            ax.plot(datetime_arr, _temp['speed_mean'].tolist(), linewidth=0.5, alpha=0.5, color='gray')

    custom_lines = [
    #                 Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=1, marker='o'),
                    Line2D([0], [0], color='gray', lw=1)
                   ]

    ax.legend(custom_lines, ['aggregate', 'segments'])
    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    plt.xticks(rotation = 0)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('center')
    return ax

def plot_attacks():
    pass

def plot_centroids(gpd, cluster, highlight=[], figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    gpd.plot(ax=ax, color='gray', alpha=0.2)
    print(cluster)
    for o in cluster:
        if highlight:
            if o in highlight:
                print(o)
                ax.scatter(*center.centroid.xy, marker='*', color='yellow', edgecolor='black', s=20**2)
                
        center = gpd[gpd['XDSegID'] == o].centroid.values[0]
        ax.scatter(*center.centroid.xy, marker='o', color='r', edgecolor='black', s=10**2)
    ax.set_title(f"Cluster: {cluster[0]}: {len(cluster)} members")
    return ax

def test_cluster_correlation(hist_speeds_df, cluster, figsize=(15, 5)):
    # Running correlation of speed mean (testing)
    running_speed_mean = []
    for i, o in enumerate(cluster):
        if not running_speed_mean:
            running_speed_mean = hist_speeds_df[o].tolist()
#             print(i, "\t", 1.0)
        else:
            o_means = hist_speeds_df[o].tolist()
            correlation = pearsonr(running_speed_mean, o_means)[0]
#             print(i, "\t", correlation)
            running_speed_mean = hmean([running_speed_mean, o_means], axis=0).tolist()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hist_speeds_df.index, running_speed_mean, color='black', alpha=1.0, marker='o', markersize=2)
    hist_speeds_df[cluster].plot(ax=ax, legend=False, color='gray', alpha=0.2)
    
    custom_lines = [
    #                 Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='black', lw=1, marker='o'),
                    Line2D([0], [0], color='gray', lw=1)
                   ]

    ax.legend(custom_lines, ['aggregate', 'segments'])
    ax.set_title(f"Historical Speed data for cluster: {cluster[0]}, {len(cluster)} members")
    ax.set_ylabel("Speed")
    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    plt.xticks(rotation = 0)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('center')

    return ax

def generate_bounds_for_cluster(cluster, clusters_df, style='envelope'):
#     print(cluster)
    if style == 'envelope':
        segments = clusters_df[clusters_df['XDSegID'].isin(cluster)]['line'].unary_union.envelope
    elif style == 'mrr':
        segments = clusters_df[clusters_df['XDSegID'].isin(cluster)]['line'].unary_union.minimum_rotated_rectangle
        
#     display(segments)
    return segments

def plot_clusters(clusters, clusters_df, marked_clusters=[], show_overlap=False, show_centroids=True, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    # clusters_df['line'].plot(ax=ax, markersize=10, color='gray', alpha=0.3)
    clusters_df['line'].plot(ax=ax, color='gray', alpha=0.2)

    pbar = tqdm(total=len(clusters))
    used_bounds = []
    cluster_list = list(clusters)
    random.shuffle(cluster_list)
    for i, cluster in enumerate(cluster_list):
        s = generate_bounds_for_cluster(clusters[cluster], clusters_df, style='envelope')
        c = s.centroid
        
            
        if show_overlap:
            if cluster in marked_clusters:
                ax.plot(*s.exterior.xy, color='red', alpha=1.0)
                ax.scatter(*c.centroid.xy, marker='*', color='red', edgecolor='black', alpha=1.0, s=10**2)
            else:
                ax.plot(*s.exterior.xy, color='blue', alpha=0.1)
                ax.scatter(*c.centroid.xy, marker='o', color='blue', edgecolor='black', alpha=0.5, s=5**2)
        else:
            if i == 0:
                used_bounds.append(s)
                ax.plot(*s.exterior.xy, color='blue', alpha=1.0, lw=2)
    #             ax.scatter(*c.centroid.xy, marker='o', color='blue', edgecolor='black', alpha=0.5, s=5**2)
                continue
            tdf = pd.DataFrame({'polys': used_bounds})
            tdf['res'] = tdf['polys'].apply(lambda x: s.intersects(x))
        #     display(tdf)
            if not tdf['res'].any():
                used_bounds.append(s)
                ax.plot(*s.exterior.xy, color='blue', alpha=1.0, lw=2)
                if show_centroids:
                    geom_center = clusters_df[clusters_df['XDSegID'].isin(clusters[cluster])]['center_m'].centroid.values[0]
                    ax.scatter(*geom_center.centroid.xy, marker='o', color='blue', edgecolor='black', alpha=0.5, s=5**2)
    #             ax.scatter(*c.centroid.xy, marker='o', color='blue', edgecolor='black', alpha=0.5, s=5**2)
#             else:
#                 ax.plot(*s.exterior.xy, color='red', alpha=0.2, lw=1.0)
            

        pbar.update(1)
    pbar.update(1)
    pbar.close()

    if not show_overlap:
        print(f"Showing {len(used_bounds)}/{len(clusters)} clusters")
        ax.set_title(f"Showing {len(used_bounds)}/{len(clusters)} clusters")
    else:
        ax.set_title(f"Showing {len(clusters)}")
    plt.axis('off')

#     fp = os.path.join(img_dir, f'{clustering_version}_clustering_bounds_overlap_{show_overlap}.png')
#     plt.savefig(fp, dpi=200, bbox_inches='tight')
    return ax