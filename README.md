# ICCPS 2022
## Anomaly based Incident Detection in Large Scale Smart Transportation Systems
This is the repository for our paper that was accepted to ICCPS2022. All private and proprietary data has been censored, anonymized and then synthesized. This has been tested on Ubuntu and OSX devices.

This will run the script as `root` user inside the container and will save the files in your local directory as `root` as well.

### Tested on:
* MacBook Pro (15-inch, 2017) 10.15.7
    * 2.9 GHz Quad-Core Intel Core i7
    * Docker Desktop 2.5.0.1
    * docker-compose version 1.27.4, build 40524192
    * Clustering: TBD
    * Training: TBD
* Ubuntu 20.04.1
    * AMD Ryzen Threadripper 3970X 32-Core
    * Docker version 20.10.5, build 55c4c88
    * docker-compose version 1.27.4, build 40524192
    * Clustering: 169.528 s
    * Training: 133.135 s
* Ubuntu
    * Intel(R) Xeon(R) CPU E5-1607 v3 @ 3.10GHz
    * Total: 3 minutes

## Requirements
1. Docker or Docker Desktop
2. Docker Compose

## Steps
1. `git clone https://github.com/linusmotu/AnomalyDetection_ICCPS_Artifact.git`
2. `cd AnomalyDetection_ICCPS_Artifact`
3. `docker-compose up --build`
4. Verify the resulting figures in `synthetic_figures` directory.

## Included synthetic data
* `synthetic_data`
    * synth_overall_means.pkl
    * synth_segments_grouped.pkl
    * synth_line_segment_graph.pkl
    * synth_all_incidents_ground_truth.pkl
    * synth_Reduced_DF_2019_10_1.pkl (`93+ MB`):contains the bulk of the speed data
    * synth_Reduced_DF_2019_11_1.pkl (`93+ MB`):contains the bulk of the speed data
    * synth_Reduced_DF_2019_12_1.pkl (`93+ MB`):contains the bulk of the speed data

## Generated data and results
This will generate the following files. Note that the results are based on synthetic data and are not meant to produce any usable and actionable information.

* `synthetic_data`
    * synth_clustering.pkl
    * synth_cluster_ground_truth.pkl
    * synth_10_2019_ratios_gran_5_incidents_24h.pkl
    * synth_11_2019_ratios_gran_5_incidents_24h.pkl
    * synth_12_2019_ratios_gran_5_incidents_24h.pkl
    * synth_10_2019_ratios_gran_5_incidents_cleaned.pkl
    * synth_11_2019_ratios_gran_5_incidents_cleaned.pkl
    * synth_12_2019_ratios_gran_5_incidents_cleaned.pkl
    * synth_10_2019_ratios_gran_5_incidents.pkl
    * synth_11_2019_ratios_gran_5_incidents.pkl
    * synth_12_2019_ratios_gran_5_incidents.pkl
* `synthetic_results`
    * used_clusters_list_synth.pkl
    * optimized_safe_margin_synth.pkl
    * optimized_residual_train_synth.pkl
    * optimized_standard_limit_synth.pkl
    * optimized_test_residual_synth.pkl
    * optimized_detection_report_synth.pkl
    * optimized_hyper_mapping_synth.pkl
    * optimized_residual_Test_QR_synth.pkl
    * optimized_results_synth.pkl
    * optimized_actual_detection_frame_synth.pkl
    * optimized_detection_report_Frame_synth.pkl
    > In addition we generate the following for each κ (kappa) value
    * κ_optimized_actual_detection_frame_synth.pkl
    * κ_optimized_detection_report_Frame_synth.pkl
    * κ_optimized_residual_Test_QR_synth.pkl
    * κ_optimized_results_synth.pkl
* `synthetic_figures`
    * synth_baseline_map.png: Clustering visualiztion (not in paper)
    * synth_Table_2.txt: Cluster information
    * synth_Figure_5: Detection illustration RUC of ith cluster
    * synth_Figure_7a: Detection performance
    * synth_Figure_8a: ROC curve
    * synth_Figure_8b: Mean time between false positives
