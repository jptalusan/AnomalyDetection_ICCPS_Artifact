# For Artifact Evaluation
1. Say which are the parameters used and what files will all be generated at the end
2. Todo: Add the graphs notebook from the original work
    * Check again, its all over the place


# Distributed Anomaly Detection
This is a repository that will store all the code and possbily Jupyter notebooks associated with the project. I will update this `README.md` every time I encounter something and have to edit something drastic.

This consists of maybe two sets of codes. One for clustering and another for training.

I am keeping the Jupyter notebook format since it is easier to run and debug and visualize for the paper.


## Clustering
These notebooks are meant to be run in sequence except for the part where we need to generate the `overall_means.pkl`. It is faster to just download that file.

### Notebooks
1. `000_Processing_Congestion.ipynb`: Loads all the speed data into memory (requires maybe >50GB of memory). Aggregates all the data by grouping them into the days of the week, time window and segment name. Generates the `overall_means.pkl` that is required by most if not all of the other notebooks. This takes the longest.
2. `001_Creating_Segment_Network_Graph.ipynb`: This creates 2 different network graphs, one for routing and travel time analysis (DiGraph), another is simply a MultiGraph that is used for clustering. In the future we should merge the two (if it works)
3. `002_Creating_Cluster.ipynb`: This generates the clusters along with other files such as cluster params and info. I found an issue where it has a problem when using DiGraphs for the hops. Hopefully I can fix them in the future. For now, we use an older network graph created in the last parts of **2**
4. `003_Generating_Ground_Truths.ipynb`: This generates the ground truth from the speed data and labels the incident depending on which segment and cluster it belongs to. Stores to `incidents_GT`
5. `004_Cluster_Ratio_Generation.ipynb`: This generates the ratios for all clusters for all months in the dataset. It saves them into the `incident_ratio_dir` in `data` folder. Files generated are 24 hours long to help facilitate extra information needed when we clean the datasets.
6. `005_Cleaning_Cluster_Ratios.ipynb`: This generates both the `cleaned.pkl` and `incidents.pkl` dataframes of the cluster ratios. These store the files into separate folders for use in training. `incidents`, `cleaned`.

#### Sub-tasks
7. `006_Cleaning_Speed_Data.ipynb`: To generate clean speed data, meant to be used for comparing the impact of missed detections on commuter travel time. Still underconstruction, will update once finished.
8. `007_Travel_Time_Computation.ipynb`: Using the generated clean speed data, we compute random routes which pass through incident segments and we compute the travel time with and without the incident. This simulates the detection and removal of incidents and its effects on travel time. One idea is to have multiple runs with different `windows` of cleaning and simulate the effect of detection timing and cleaning.

### Data
Located inside `data/`.
* `overall_means.pkl`: Dataframe containing all the `congestion_mean` and `speed_mean` grouped by day of week, time window and segment name. No need to regenerate this, just download the data.
* `inrix_grouped.pkl`: GeopandasDataFrame containing all the Inrix segments. This is much larger than the available data so we need to filter only those ones. This is not generated, needed to download.
* `segment_network_graphs`: Directory to store the generated network graphs.
* `speed_data`: Directory to store all the `ALL_5m_DF_2019_*_*.gzip` files. Must be downloaded.
* `generated_clusters`: A place to store the different `clustering_versions` for cleanliness
* `all_incidents_ground_truth.pkl`: A DataFrame containing all the incidents throughout the dataset. Can be downloaded.

#### Others:
* `random_od_nashville.pkl`: Still subject to change once we verify the correct network graph to use
* `nashville_graph.pkl`: Same reasoning as `random_od_nashville.pkl`

## Training
This one essentially is made up of seven separate notebooks that I compiled into a single notebook so that I could run them in one sitting since the training takes a bit too long. For cleanliness, I also divided it back into 7 different notebooks listed below.

### Notebooks
1. `000_Training_Residual_and_Safe_Margin.ipynb`
2. `001_Standard_Limit_QR.ipynb`
3. `002_Cross_Validate_Residual.ipynb`
4. `003_Detection_QR.ipynb`
5. `004_Analyze_Detection_QR.ipynb`
6. `005_Test_Residual_QR.ipynb`
7. `006_Test_Analysis_QR.ipynb`

### Others:
* `ALTERNATIVE_ONE_NOTEBOOK.ipynb`

### Data Generated
* `used_clusters_list_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_safe_margin_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_residual_train_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_standard_limit_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_test_residual_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_detection_report_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_hyper_mapping_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_actual_detection_Frame_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_residual_Test_QR_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_results_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_detection_report_Frame_{clustering_version}_{no_of_clusters}C_{date}.pkl`
* `optimized_actual_detection_frame_{clustering_version}_{no_of_clusters}C_{date}.pkl`

## Visualization
### Notebooks

* `SUPPLEMENT_Graph_Visualization.ipynb`
* `SUPPLEMENT_Viewer.ipynb`

## Requirements


## Output

## TODO:

> Note: I just realized that most of my code was written with the knowledge that there is only 1 year in the dataset. It would need to be changed to accomodate multiple years so it wont overwrite on existing generated data.