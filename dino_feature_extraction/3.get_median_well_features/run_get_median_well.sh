#!/bin/bash

cells=("A549-WT" "U2OS-WT")
for cell in "${cells[@]}"; do
config_file="/tmp/config_${cell}.yaml"
cat <<EOL > $config_file
channels: ['Ph']
base_dir: "dino_feature_extraction/data/$cell/dino_vitb14_features/single_cell_features"
save_path: "dino_feature_extraction/data/$cell/dino_vitb14_features/$cell-median_well_features.parquet.gzip"
EOL

/bin/bash -c "source venv/bin/activate && PYTHONPATH=. python dino_feature_extraction/3.get_median_well_features/get_median_well.py $config_file"

# Clean up the temporary config file
rm $config_file
done