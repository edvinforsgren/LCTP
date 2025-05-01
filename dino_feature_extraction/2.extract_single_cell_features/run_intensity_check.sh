#!/bin/bash

cells=("A549-WT" "U2OS-WT")
for cell in "${cells[@]}"; do
# Create a temporary configuration file for this job
config_file="/tmp/config_${cell}.yaml"
cat <<EOL > $config_file
path_incl: "dino_feature_extraction/data/$cell/crops_all_timepoints/cropped_expand_torch/incl/metadata_incl"
save_channel_intensities: "dino_feature_extraction/data/$cell/channel_intensities_$cell.yaml"
base_dir: "dino_feature_extraction/data/$cell/"
channels: ['Ph']
cell_healths: ['Live', 'Dead']
EOL

# Run the singularity command with the temporary config file
# /bin/bash -c "source venv/bin/activate && python dino_feature_extraction/2.extract_single_cell_features/check_channel_intensity_parallell.py $config_file"
/bin/bash -c "source venv/bin/activate && PYTHONPATH=. python dino_feature_extraction/2.extract_single_cell_features/check_channel_intensity_parallell.py $config_file"

# Clean up the temporary config file
rm $config_file
done
