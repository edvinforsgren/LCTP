#!/bin/bash

cells=("A549-WT" "U2OS-WT")
for cell in "${cells[@]}"; do
config_file="/tmp/config_${cell}.yaml"
## plate: None -> will run on all plates in dataframe. This can be specified for parallel computing.
cat <<EOL > $config_file
seed: 42
model:
    activation: LeakyReLU
    normalize: "True"
    epochs: 200
    batch_size: 64
    learning_rate: 0.0004
    loss_function: 'MSELoss'
    dropout_rate: 0.5
    layers: [512, 256]
time:
    hour: 48
data:
    cell: "$cell"
    path: "dino_feature_extraction/data/$cell/dino_vitb14_features/$cell-median_well_features.parquet.gzip"
    save_path: 'analysis/pred_data/$cell-stp/'
    compare_config_file_path: "analysis/comparison/$cell-analysis-datapaths.yaml"
EOL
echo "Running Single time point analysis for $cell with config file: $config_file"
/bin/bash -c "source venv/bin/activate && PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:16:8 python analysis/single_time_point/1.single_time_point_analysis.py $config_file"
# Clean up the temporary config file
rm $config_file
done