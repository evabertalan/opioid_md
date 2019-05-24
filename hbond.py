import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.hbonds
import matplotlib.pyplot as plt
import pandas as pd

class HBond:
    
    def __init__(self):
        self.h = None
        self.df = None
        self.by_type_df = None
        
    def prot_prot(self, traj):
        self.h = mda.analysis.hbonds.HydrogenBondAnalysis(traj, selection1='protein',
                                                     selection2='protein',
                                                     selection1_type='both',
                                                     distance=3.0)
        self.h.run()
        return self.h
    
    def print_table(self):
        self.h.generate_table()
        self.df = pd.DataFrame.from_records(self.h.table)
        print(self.df)
        
    def distance_histogram(self):
        self.df.hist(column=["distance"])

    def heatmap(self, bins, table=False): #bins=number of residues
        table = self.h.table
        plt.subplots(figsize=(20,20))
        plt.hist2d(table['donor_resid'], table['acceptor_resid'], bins=bins)
        
    def save_table(self, folder, file_name='hbonds.csv'):
        self.df.to_csv(folder+file_name)
        
    def plot_by_time(self):
        by_time = self.h.count_by_time()
        time = [i[0] for i in by_time]
        N = [i[1] for i in by_time]
        plt.plot(time, N)
        plt.title('Formed hydrogen bonds over time')
        
    def frequency(self):
        by_type = self.h.count_by_type()
        self.by_type_df = pd.DataFrame(data=by_type)
        self.by_type_df.columns = ['donor_index', 'acceptor_index', 'donor_resnm',
                           'donor_resid','donor_heavy_atom', 'donor_atom', 'acceptor_resnm',
                           'acceptor_resid', 'acceptor_atom', 'frequency']
        print('sajt')
#         print(self.by_type_df)
        return self.by_type_df
    
        
    def inter_frequency(self, bins, treshhold=0.5):
        inter_hbond = self.by_type_df.loc[(self.by_type_df['frequency'] > treshhold) & (self.by_type_df['donor_resid'] != self.by_type_df['acceptor_resid'])]
        plt.subplots(figsize=(20,20))
        plt.hist2d(inter_hbond['donor_resid'], inter_hbond['acceptor_resid'], bins=bins)
        
    