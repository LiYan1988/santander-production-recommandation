#!/usr/bin/env bash
#SBATCH -A C3SE407-15-3
#SBATCH -p hebbe
#SBATCH -J cluster_1_2_all_month_validate
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -C MEM64
#SBATCH -t 0-00:10:00
#SBATCH -o cluster_1_2_all_month_validate.stdout
#SBATCH -e cluster_1_2_all_month_validate.stderr
module purge 

export PATH="/c3se/NOBACKUP/users/lyaa/conda_dir/miniconda/bin:$PATH"
source activate kaggle

pdcp * $TMPDIR

cd $TMPDIR

python cluster_1_2_all_month_validate.py

cp * $SLURM_SUBMIT_DIR
rm -rf $TMPDIR/*
# End script