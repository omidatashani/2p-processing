#!/bin/bash
#SBATCH --job-name=Preprocess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=9:00:00
#SBATCH --partition=cpu-large
#SBATCH --output=Preprocess_%A.out
#SBATCH --error=Preprocess_%A.err
#SBATCH --mail-user=omidamiratashani@gmail.com
#SBATCH --mail-type=END,FAIL

echo "===== Starting preprocessing job on $(hostname) at $(date) ====="

# Load Anaconda
module load anaconda3

# Check existing Conda environments
echo "===== Checking existing Conda environments ====="
conda info --envs

# Create environment if needed
ENV_NAME="suite2p_env"
ENV_EXISTS=$(conda info --envs | grep -w $ENV_NAME)

if [ -z "$ENV_EXISTS" ]; then
    echo "===== Environment '$ENV_NAME' not found. Creating it... ====="
    conda create -n $ENV_NAME python=3.9 -y
else
    echo "===== Environment '$ENV_NAME' already exists. ====="
fi

# Activate the environment
echo "===== Activating environment '$ENV_NAME' ====="
source activate $ENV_NAME

# Install required packages
echo "===== Checking and installing required Python packages ====="

REQUIRED_PACKAGES=(
    suite2p
    pandas
    numpy
    h5py
)

for PACKAGE in "${REQUIRED_PACKAGES[@]}"; do
    python -c "import $PACKAGE" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing missing package: $PACKAGE"
        pip install $PACKAGE --quiet
    else
        echo "Package $PACKAGE already installed."
    fi
done

# Confirm imports
echo "===== Verifying Python package imports ====="
python -c "import suite2p, pandas, numpy, h5py; print('✔️ All packages successfully imported.')"

# Change to working directory
cd /storage/coda1/p-fnajafi3/0/oamiratashani6/2p

# Run preprocessing script
echo "===== Running Suite2p pipeline ====="
python run_suite2p_pipeline.py \
  --denoise 1 \
  --spatial_scale 1 \
  --data_path '/storage/coda1/p-fnajafi3/0/shared/2P_Imaging/E5LG/E5LG_CRBL_crux2_20250506_EBC-259/' \
  --save_path '/storage/cedar/cedar0/
cedarp-fnajafi3-0 /
2p_imaging /' \
  --nchannels 1 \
  --functional_chan 2 \
  --target_structure 'dendrite'

echo "===== Preprocessing job 1 finished at $(date) ====="