import Bio.PDB as biop
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms
import MDAnalysis.analysis.helanal as helanal
import matplotlib.pyplot as plt

def rmsd_traj(traj, ref, title):
    R = rms.RMSD(traj.select_atoms('name CA'), ref.select_atoms('name CA')).run()
    R = R.rmsd.T
    frame = R[0]
    time = R[1]
    
    plt.subplots(figsize=(10,5))
    fig = plt.plot(time/1000, R[2], linewidth=0.2)
    plt.ylabel('RMSD ($\AA$)')
    plt.xlabel('time (ns)')
    plt.title(title)
    plt.show()
    return R, fig

	
def secondary_structure(pdb_file, pdb_code):
    parser = biop.PDBParser()
    structure = parser.get_structure(pdb_code, pdb_file)
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
    
        ax[i,0].plot(calphas.resnums, rmsfer.rmsf, color='green')

        if structure:
            for j, k, in enumerate(structure['helix']):
                ax[i,0].axvline(k, color='#3b3b3b', alpha=0.1)

#             for j, k, in enumerate(structure['pi_helix']):
#                 ax[i,0].axvline(k, color='grey', alpha=0.2)     

        ax[i, 0].set_ylabel('RMSF ($\AA$)')
        ax[i, 0].set_title(title.format(i+1))
    
    ax[len(trajectories)-1, 0].set_xlabel('residue ID')
    plt.tight_layout()
    plt.show()
    return fig, calphas.resnums, rmsfer.rmsf

def rmsf_selected_residues(trajectories, residues, title):
    fig, ax = plt.subplots(len(trajectories),1, sharex=True, figsize=(15, 4*len(trajectories)), squeeze=False)
    for i in range(len(trajectories)):
        protein = trajectories[i].select_atoms("protein")
#         calphas = protein.select_atoms("name CA and ( resid 3:32 or resid 39:67 or resid 73:105 or resid 116:142 or resid 165:205 or resid 213:245 or resid 255:279 )")
        
        calphas = protein.select_atoms("name CA and ( resid 4:29 or resid 39:67 or resid 73:115 or resid 118:142 or resid 168:201 or resid 210:245 or resid 252:279 )")
        
        rmsfer = rms.RMSF(calphas).run()
    
        ax[i,0].plot(list(range(len(calphas))), rmsfer.rmsf)     
        ax[i, 0].set_ylabel('RMSF ($\AA$)')
        ax[i, 0].set_title(title.format(i+1))
    
    ax[len(trajectories)-1, 0].set_xlabel('residue')
    plt.tight_layout()
    plt.show()
    return fig

def waters_per_frame(water_files):
    waters = np.loadtxt(water_files[0])
    for i in range(1, len(water_files)):
        data = np.loadtxt(water_files[i])
        waters = np.concatenate((waters, data))
#     plt.plot(waters[:,1])

    plt.subplots(figsize=(10,5))
    fig = plt.plot(waters[:,1], linewidth=0.2)
    plt.ylabel('waters')
    plt.xlabel('frame')
    plt.title('Number of water within protein')
    plt.show()
    return fig