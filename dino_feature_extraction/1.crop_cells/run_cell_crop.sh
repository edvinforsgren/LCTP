#!/bin/bash

cells=("U2OS-WT" "A549-WT" )
for cell in "${cells[@]}"; do
config_file="/tmp/config_${cell}.yaml"
## plate: None -> will run on all plates in dataframe. This can be specified for parallel computing.
cat <<EOL > $config_file
dataframe_path: 'dino_feature_extraction/data/metadata.parquet.gzip'
base_dir: "dino_feature_extraction/data/$cell/"
save_dir: "dino_feature_extraction/data/$cell/"
sub_dir: "crops_all_timepoints/cropped_expand_torch"
channels: ['Ph']
cell_healths: ['Live', 'Dead']
EOL
echo "Running cell crop for $cell with config file: $config_file"
# Run the singularity command with the temporary config file
# /bin/bash -c "source venv/bin/activate && python dino_feature_extraction/1.crop_cells/crop_cells.py $config_file"
/bin/bash -c "source venv/bin/activate && PYTHONPATH=. python dino_feature_extraction/1.crop_cells/crop_cells.py $config_file"
# Clean up the temporary config file
# rm $config_file
done

