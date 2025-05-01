#!/bin/bash
set -e  # Exit on error
#start timer
start_time=$(date +%s)
./dino_feature_extraction/1.crop_cells/run_cell_crop.sh
./dino_feature_extraction/2.extract_single_cell_features/run_intensity_check.sh
./dino_feature_extraction/2.extract_single_cell_features/run_dino_feat.sh
./dino_feature_extraction/3.get_median_well_features/run_get_median_well.sh
echo "Feature extraction complete."
#end timer
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Execution time: $((execution_time / 3600)) hours, $((execution_time % 3600 / 60)) minutes, and $((execution_time % 60)) seconds"