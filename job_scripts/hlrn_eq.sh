#!/bin/bash
#PBS -l nodes=4:ppn=24
#PBS -N step6.1_equilibration
#PBS -A bepberta
#PBS -l feature=mpp1
#PBS -q mpp1q
#PBS -j oe
#PBS -l walltime=12:00:00

module load namd
module load craype-hugepages8M


aprun -B namd2 step6.1_equilibration.inp  > step6.1_equilibration.inp
