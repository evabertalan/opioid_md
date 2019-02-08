#!/bin/bash
#PBS -l nodes=14:ppn=24
#PBS -N step7.1_production
#PBS -A bepberta
#PBS -l feature=mpp1
#PBS -q mpp1q
#PBS -j oe
#PBS -l walltime=12:00:00

module load namd
module load craype-hugepages8M


aprun -B namd2 step7.1_production.inp  > step7.1_production.out
