from modeller import *
import sys

code = sys.argv[1]
seq = sys.argv[2]

# code = raw_input('pdb code: ')
# seq = raw_input('sequence: ')

e = environ()
m = model(e, file=code, model_segment=('FIRST:'+seq,'LAST:'+seq))
aln = alignment(e)
aln.append_model(m, align_codes=code)
aln.write(file=code+'.seq')
