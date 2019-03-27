import Bio.PDB as biop
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms
import MDAnalysis.analysis.helanal as helanal
import matplotlib.pyplot as plt

def rmsd_traj(traj, ref, title):
    R = rms.RMSD(traj.select_atoms('name CA'), ref.select_atoms('name CA'), select='all').run()
    R = R.rmsd.T
    frame = R[0]
    time = R[1]
    
    plt.subplots(figsize=(10,5))
    fig = plt.plot(time, R[2], linewidth=0.2)
    plt.ylabel('RMSD ($\AA$)')
    plt.xlabel('time (ps)')
    plt.title(title)
    plt.show()
    return R, fig

	
def secondary_structure(pdb_file):
    parser = biop.PDBParser()
    structure = parser.get_structure('6b73', pdb_file)
    model = structure[0]
    dssp = biop.DSSP(model, pdb_file)

    return {
        'helix': [np.array(np.where(np.array(dssp)[:,2] == 'H')) + 1][0][0],
        'strand': [np.array(np.where(np.array(dssp)[:,2] == 'E')) + 1][0][0],
        'pi_helix': [np.array(np.where(np.array(dssp)[:,2] == 'I')) + 1][0][0],
        'turn': [np.array(np.where(np.array(dssp)[:,2] == 'T')) + 1][0][0],
        'bend': [np.array(np.where(np.array(dssp)[:,2] == 'S')) + 1][0][0], 
    }
    
def rmsf_plot(trajectories, title, structure=False):
    fig, ax = plt.subplots(len(trajectories),1, sharex=True, figsize=(15, 4*len(trajectories)), squeeze=False)
    for i in range(len(trajectories)):
        protein = trajectories[i].select_atoms("protein")
        calphas = protein.select_atoms("name CA")
        rmsfer = rms.RMSF(calphas).run()
    
        ax[i,0].plot(calphas.resnums, rmsfer.rmsf)

        if structure:
            for j, k, in enumerate(structure['helix']):
                ax[i,0].axvline(k, color='grey', alpha=0.1)

            for j, k, in enumerate(structure['pi_helix']):
                ax[i,0].axvline(k, color='grey', alpha=0.2)     

        ax[i, 0].set_ylabel('RMSF ($\AA$)')
        ax[i, 0].set_title(title.format(i+1))
    
    ax[len(trajectories)-1, 0].set_xlabel('residue')
    plt.tight_layout()
    plt.show()
    return fig
