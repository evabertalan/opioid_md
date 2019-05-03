#!/bin/bash

#SBATCH --job-name=4djhA_production1
#SBATCH --ntasks=64
#SBATCH --nodes=8
#SBATCH --time=0-14:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --output=step7.1.out
#SBATCH --error=step7.1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bertalae93@zedat.fu-berlin.de

module load namd
NAMD2=`which namd2`
mpirun $NAMD2 step7.1_production.inp

sleep 50
