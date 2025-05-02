#!/bin/bash

cells=("A549-WT" "U2OS-WT")
for cell in "${cells[@]}"; do
config_file="/tmp/config_${cell}.yaml"
cat <<EOL > $config_file
path_incl: "dino_feature_extraction/data/$cell/crops_all_timepoints/cropped_expand_torch/incl/metadata_incl"
base_dir: "dino_feature_extraction/data/$cell/"
save_dir: "dino_feature_extraction/data/$cell/dino_vitb14_features/single_cell_features"
channels: ['Ph']
save_channel_intensities: "dino_feature_extraction/data/$cell/channel_intensities_$cell.yaml"
batch_size: 1536
model: ["facebookresearch/dinov2", "dinov2_vitb14"]
EOL
# Run the feature extraction with the temporary config file
/bin/bash -c "source venv/bin/activate && PYTHONPATH=. python dino_feature_extraction/2.extract_single_cell_features/extract_dino_features_from_zip.py $config_file"

# Clean up the temporary config file
rm $config_file
done