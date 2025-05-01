#!/bin/bash

cells=("A549-WT" "U2OS-WT")
for cell in "${cells[@]}"; do
config_file="/tmp/config_${cell}.yaml"
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
    hours1: [6, 8, 10, 12, 14]
    hours2: [46, 48, 50, 52, 54]
data:
    cell: "$cell"
    path: "dino_feature_extraction/data/$cell/dino_vitb16_features/$cell-median_well_features.parquet.gzip"
    save_path: 'analysis/pred_data/$cell-ts/'
EOL
echo "Running Time series plotting for $cell with config file: $config_file"
/bin/bash -c "source venv/bin/activate && PYTHONPATH=. python analysis/time_series/2.plot_and_save_time_series.py $config_file"
# Clean up the temporary config file
rm $config_file
done
