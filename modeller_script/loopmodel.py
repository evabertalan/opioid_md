import sys
from modeller import *
from modeller.automodel import *    # Load the automodel class

knowns = sys.argv[1]
sequence = sys.argv[2]
num_models = int(sys.argv[3])
num_loops = int(sys.argv[4])

log.verbose()
env = environ()

# directories for input atom files
env.io.atom_files_directory = ['.']

a = dopehr_loopmodel(
  env,
  alnfile = 'alignment.ali',
  knowns = knowns, #first block, name of template, comes from pdb
  sequence = sequence, #second block. name of target, real sequens from fasta
  assess_methods=(assess.DOPE, assess.normalized_dope, assess.GA341))
a.starting_model = 1
a.ending_model = num_models #number of models to generate

a.loop.starting_model = 1
a.loop.ending_model = num_loops #number of generated loop models for each comperative model
a.loop.md_level = refine.slow

a.make()

#models: 20
#loopmodel: 5*20
#total number of models: 5*20 + 20
