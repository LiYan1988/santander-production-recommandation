#!/usr/bin/env bash
#SBATCH -A C3SE407-15-3
#SBATCH -p hebbe
#SBATCH -J cluster_1_1
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -C MEM128
#SBATCH -t 0-00:0:0
#SBATCH -o cluster_1_1.stdout
#SBATCH -e cluster_1_1.stderr
module purge 

export PATH="/c3se/NOBACKUP/users/lyaa/conda_dir/miniconda/bin:$PATH"
source activate kaggle

pdcp * $TMPDIR
pdcp -r 

cd $TMPDIR

python cluster_1_1.py

cp * $SLURM_SUBMIT_DIR
rm -rf $TMPDIR/*
# End script