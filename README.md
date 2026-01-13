# gene-stabilization-detector
Automated monitor for GENE scan runs that detects convergence from energy files using steady and oscillatory growth-rate criteria. Upon stabilization, it saves diagnostics, stops the active run, and advances to the next scan case, enabling efficient large-scale parameter scans.

Clone this repo inside `/gene/prob02/` directory, and edit `submit.cmd`:

```
#!/bin/bash -l
#SBATCH -C cpu                     # use CPU partition
#SBATCH -t 00:30:00                # wallclock limit
#SBATCH -N 1                       # total number of nodes, 2 Milan CPUs with 64 rank each
#SBATCH --ntasks-per-node=128      # 64 per CPU. Additional 2 hyperthreads disabled
#SBATCH -c 2                       # cpus per task, 2 if full CPU. Adjust accordingly
#SBATCH -J GENE
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#SBATCH -q debug
##set specific account to charge
#SBATCH -A m2116

#SBATCH --mail-user=seungjungkim@utexas.edu
#SBATCH --mail-type=END,FAIL,BEGIN


export SLURM_CPU_BIND="cores"

## set openmp threads
export OMP_NUM_THREADS=1

#do not use file locking for hdf5
export HDF5_USE_FILE_LOCKING=FALSE

# >>>>>  ADD THE LINES BELOW >>>>>>

module load python/3.11
#start monitoring scanscript
python gene-stabilization-detector/monitor_scanscript.py &

# <<<<<  TO HERE <<<<<<<

set -x
# run GENE
#srun -l -K -n $SLURM_NTASKS ./gene_perlmutter

# run scanscript
./scanscript --np $SLURM_NTASKS --ppn $SLURM_NTASKS_PER_NODE --mps 4 --syscall='srun -l -K -n $SLURM_NTASKS --cpu_bind=cores ./gene_perlmutter'

set +x
```
