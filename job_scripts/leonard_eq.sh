#!/bin/bash

#SBATCH --job-name=4djhA_equilibration1
#SBATCH --ntasks=32
#SBATCH --nodes=4
#SBATCH --time=0-14:00:00
#SBATCH --mem-per-cpu=800
#SBATCH --output=step6_1.out
#SBATCH --error=step6_1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bertalae93@zedat.fu-berlin.de

module load namd
NAMD2=`which namd2`
mpirun $NAMD2 step6.1_equilibration.inp

sleep 50
