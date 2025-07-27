#!/bin/bash
#SBATCH --job-name=filetransfer
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=9:00:00
#SBATCH --partition=cpu-large
#SBATCH --output=filetransfer_%A.out
#SBATCH --error=filetransfer_%A.err
#SBATCH --mail-user=omidamiratashani@gmail.com
#SBATCH --mail-type=END,FAIL

echo "===== Starting file transfer job on $(hostname) at $(date) ====="

# Define source and destination base paths
SRC_BASE="/storage/coda1/p-fnajafi3/0/oamiratashani6/results"
DEST_BASE="/storage/coda1/p-fnajafi3/0/oamiratashani6/results/E5LG"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_BASE"

# List of directories to move
dirs=(
    "E5LG_CRBL_crux1_20250608_EBC-436"
    "E5LG_CRBL_crux1_20250611_EBC-455"
    "E5LG_CRBL_crux1_20250613_EBC-466"
    "E5LG_CRBL_crux1_20250615_EBC-478"
)

# Move each directory
for dir in "${dirs[@]}"; do
    echo "Moving $dir ..."
    rsync -avh --progress "$SRC_BASE/$dir" "$DEST_BASE/"
    # Uncomment the next line to delete the original after successful copy
    rm -rf "$SRC_BASE/$dir"
done

echo "===== File transfer job completed at $(date) ====="