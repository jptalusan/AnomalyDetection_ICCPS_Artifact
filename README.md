# ICCPS 2022
## Anomaly based Incident Detection in Large Scale Smart Transportation Systems
This is the repository for our paper that was accepted to ICCPS2022. All private and proprietary data has been censored, anonymized and then synthesized. This has been tested on Ubuntu and OSX devices.

### Tested on:
* MacBook Pro (15-inch, 2017) 10.15.7
    * 2.9 GHz Quad-Core Intel Core i7
    * Docker Desktop 2.5.0.1
    * docker-compose version 1.27.4, build 40524192
    > Will change
    * Clustering: 77.449 s
    * Training: 358.21 s
* Ubuntu (TBA)

## Requirements
1. Docker or Docker Desktop
2. Docker Compose

## Steps
1. `git clone https://github.com/linusmotu/AnomalyDetection_ICCPS_Artifact.git`
2. `cd AnomalyDetection_ICCPS_Artifact`
3. Give permission to the `run_script.sh` before hand this only contains the python commands to run the evaluation.
    * `chmod +x run_script.sh`
4. `docker-compose up --build`

## Results
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
    > In addition we generate the following for each kappa value
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
