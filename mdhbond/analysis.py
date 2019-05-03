from . import helpfunctions as _hf
import numpy as _np
import networkx as _nx
import MDAnalysis as _MDAnalysis
from scipy import spatial as _sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from collections import OrderedDict as _odict
from scipy.optimize import curve_fit as _curve_fit
from scipy.special import gamma as _gamma
from itertools import combinations
import pickle as cPickle
from matplotlib.ticker import MaxNLocator
import copy as _cp
import matplotlib
matplotlib.use('TKAgg', warn=False)
import matplotlib.pyplot as _plt


class BasicFunctionality(object):
    
    def __init__(self, selection=None, structure=None, trajectories=None, 
                 start=None, stop=None, step=1, restore_filename=None):
        
        if restore_filename != None: 
            self.load_from_file(restore_filename)
            return
        if selection==None or structure==None: raise AssertionError('You have to specify a selection and a structure.')
        self._selection = selection
        self._structure = structure
        self._trajectories = trajectories
        if trajectories != None: self._universe = _MDAnalysis.Universe(structure, trajectories)
        else: self._universe = _MDAnalysis.Universe(structure)
        self._trajectory_slice = slice(start if isinstance(start, int) else None, stop if isinstance(stop, int) else None, step)
        
        self._mda_selection = self._universe.select_atoms(selection)
        if not self._mda_selection:  raise AssertionError('No atoms match the selection')
        
        self._water = self._universe.select_atoms('(resname TIP3 and name OH2) or (resname HOH and name O)')
        self._water_ids = _hf.MDA_info_list(self._water, detailed_info=False)
        self._water_ids_atomwise = _hf.MDA_info_list(self._water, detailed_info=True)
        
        self.initial_results = {}
        self.filtered_results = {}
        self.nb_frames = len([0 for i in self._universe.trajectory[self._trajectory_slice]])
        
    def dump_to_file(self, fname):
        self._universe=None
        self._water=None
        self._hydrogen=None
        self._mda_selection=None
        self._da_selection=None
        self._donors=None
        self._acceptors=None
        with open(fname, 'wb') as af:
            af.write(cPickle.dumps(self.__dict__))

    def load_from_file(self, fname):
        with open(fname, 'rb') as af:
            self.__dict__ = cPickle.loads(af.read())
        #if self._trajectories != None: self._universe = _MDAnalysis.Universe(self._structure, self._trajectories)
        #else: self._universe = _MDAnalysis.Universe(self._structure)

    def _set_results(self, result):
        self.initial_results = result
        self.filtered_results = result
        self._generate_graph_from_current_results()
        self._generate_filtered_graph_from_filtered_results()
        
    def _save_or_draw(self, filename):
        if filename != None:
            end = filename.split('.')[-1]
            if end == 'eps': _plt.text.usetex = True
            _plt.savefig(filename, format=end, dpi=300)
            _plt.close()
        else:
            _plt.show()
            
    def duplicate(self):
        return _cp.copy(self)


class HydrationAnalysis(BasicFunctionality):
    
    def __init__(self, selection=None, structure=None, trajectories=None,
                 start=None, stop=None, step=1, restore_filename=None):
        
        super(HydrationAnalysis, self).__init__(selection=selection, structure=structure, trajectories=trajectories, 
             start=start, stop=stop, step=step, restore_filename=restore_filename)
        
    def set_presence_in_hull(self):
        result = {}
        frame_count = 0
        frames = self.nb_frames
        
        for ts in self._universe.trajectory[self._trajectory_slice]:
            water_coordinates = self._water.positions
            select_coordinates = self._mda_selection.positions
            
            hull = _sp.Delaunay(select_coordinates)
            local_index = (hull.find_simplex(water_coordinates) != -1).nonzero()[0]
    
            frame_res = [self._water_ids[i] for i in local_index]            
            
            for water in frame_res:
                try:
                    result[water][frame_count] = True
                except:
                    result[water] = _np.zeros(frames, dtype=bool)
                    result[water][frame_count] = True
            frame_count+=1
        self.initial_results = result
        self.filtered_results = result
    
    def set_presence_around(self, around_distance):
        result = {}
        frame_count = 0
        frames = self.nb_frames
        
        for ts in self._universe.trajectory[self._trajectory_slice]:
            
            water_coordinates = self._water.positions
            selection_coordinates = self._mda_selection.positions
            
            selection_tree = _sp.cKDTree(selection_coordinates)
            water_tree = _sp.cKDTree(water_coordinates, leafsize=32)
            local_water_index = []
            [local_water_index.extend(l) for l in selection_tree.query_ball_tree(water_tree, float(around_distance))]
            local_water_index = _np.unique(local_water_index)
    
            frame_res = [self._water_ids[i] for i in local_water_index]
            
            for water in frame_res:
                try:
                    result[water][frame_count] = True
                except:
                    result[water] = _np.zeros(frames, dtype=bool)
                    result[water][frame_count] = True
            frame_count+=1
        self.initial_results = result
        self.filtered_results = result
    
    def filter_occupancy(self, min_occupancy, use_filtered=True):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if len(results) == 0: raise AssertionError('nothing to filter!')
        filtered_result = {key:results[key] for key in results if _np.mean(results[key])>min_occupancy}
        self.filtered_results = filtered_result
    
    def compute_mean_residence_time(self, filter_artifacts=True, use_filtered=True):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        
        def func(x, tau, lamda):
            return _np.exp(-(x/tau) ** lamda) 
        xdata = _np.arange(1,self.nb_frames+1)
        
        intervals = {key:_hf.intervals_binary(results[key]) for key in results}
        residence_time = _np.zeros(self.nb_frames)
        average_time = _np.array([0.,0.])
        for water in results:
            work_intervals = intervals[water]
            interval_lengths = _np.diff(work_intervals, 1).astype(_np.int).flatten()
            if filter_artifacts: interval_lengths = interval_lengths[interval_lengths<self.nb_frames]
            average_time += _np.array([interval_lengths.sum(),interval_lengths.size])
            for l in interval_lengths:
                residence_time[:l] += _np.arange(1, l+1)[::-1]
            residence_time /= _np.arange(1,self.nb_frames+1)[::-1]
        residence_time /= residence_time[0]
        average_time = average_time[0]/average_time[1]
        try: 
            (tau, lamda), pcov = _curve_fit(func, xdata, residence_time, p0=(10.0, 1.0))
            residence_time = tau/lamda * _gamma(1/lamda)
        except: 
            if average_time <= 2.0:
                residence_time = 1.0
            else:
                residence_time = _np.inf
        return residence_time, average_time
    
    def compute_water_count(self, use_filtered=True):
        if use_filtered: water_presence = self.filtered_results
        else: water_presence = self.initial_results
        presence = _np.array([val for val in water_presence.values()]).T
        return presence.sum(1)
    
    def draw_water_count(self, use_filtered=True, filename=None):
        water_count = self.compute_water_count(use_filtered)
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        _plt.xlabel('Frame' , fontsize = 16)
        _plt.ylabel('Count' , fontsize = 16)
        _plt.plot(water_count)
        self._save_or_draw(filename)
        
    def compute_influx_count(self, return_mean_and_std = False, use_filtered=True):
        if use_filtered: water_presence = self.filtered_results
        else: water_presence = self.initial_results
        presence = _np.array([val for val in water_presence.values()]).T
        results = _np.zeros((2,self.nb_frames))
        #base = presence[0].sum()
        for frame in range(self.nb_frames-1):
            results[0][frame] = (_np.logical_not(presence[frame]) & presence[frame+1]).sum()
            results[1][frame] = (presence[frame] & _np.logical_not(presence[frame+1])).sum()
        if return_mean_and_std: return results, results.mean(1), results.std(1)
        else: return results
            
    def draw_influx_count(self, interval=1, use_filtered=True, filename=None):
        results = self.compute_influx_count(use_filtered=use_filtered)
        rest = (results[0].size%interval) * -1
        if rest != 0: results = results.T[:rest].T
        influx = results[0].reshape((-1,interval)).mean(1)
        outflux = results[1].reshape((-1,interval)).mean(1)*-1
        xdata = _np.arange(len(influx))*interval
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        _plt.bar(xdata,influx, label='influx', width=interval)
        _plt.bar(xdata,outflux, label='efflux', width=interval)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _plt.legend()
        self._save_or_draw(filename)
        

class NetworkAnalysis(BasicFunctionality):
    
    def __init__(self, selection=None, structure=None, trajectories=None, distance=3.5, cut_angle=60., 
                 start=None, stop=None, step=1, additional_donors=[], 
                 additional_acceptors=[], exclude_donors=[], exclude_acceptors=[], 
                 special_naming=[], check_angle=True, residuewise=True, add_donors_without_hydrogen=False, restore_filename=None):
        
        super(NetworkAnalysis, self).__init__(selection=selection, structure=structure, trajectories=trajectories, 
             start=start, stop=stop, step=step, restore_filename=restore_filename)
    
        self.donor_names = _hf.donor_names_global.union(additional_donors)-set(exclude_donors)
        self.acceptor_names = _hf.acceptor_names_global.union(additional_acceptors)-set(exclude_acceptors)
        self.check_angle = check_angle
        self.distance = distance
        self.cut_angle = cut_angle
        
        sorted_selection = _hf.Selection(self._mda_selection, self.donor_names, self.acceptor_names, add_donors_without_hydrogen)
        if not sorted_selection.donors: da_selection = sorted_selection.acceptors
        elif not sorted_selection.acceptors: da_selection = sorted_selection.donors
        else: da_selection = _MDAnalysis.core.groups.AtomGroup(sorted_selection.donors + sorted_selection.acceptors)
        self._da_selection = da_selection
        if sorted_selection.donors: self._donors = _MDAnalysis.core.groups.AtomGroup(sorted_selection.donors)
        else: self._donors = _hf.EmptyGroup()
        self._nb_donors = len(self._donors)
        if sorted_selection.acceptors: self._acceptors = _MDAnalysis.core.groups.AtomGroup(sorted_selection.acceptors)
        else: self._acceptors = _hf.EmptyGroup()
        self._nb_acceptors = len(self._acceptors)
        da_ids = _hf.MDA_info_list(da_selection, detailed_info=not residuewise, special_naming=special_naming)
        da_ids_atomwise = _hf.MDA_info_list(da_selection, detailed_info=True, special_naming=special_naming)
        self._first_water_id = len(da_selection)
        self._first_water_hydrogen_id = len(sorted_selection.hydrogens)
        
        water_hydrogen = [h for l in self._water for h in l.residue.atoms[1:]]
        if not sorted_selection.hydrogens and not water_hydrogen: 
            if check_angle: raise AssertionError('There are no possible hbond donors in the selection and no water. Since check_angle is True, hydrogen is needed for the calculations!')
            else: hydrogen = _hf.EmptyGroup()
        elif not sorted_selection.hydrogens: hydrogen = _MDAnalysis.core.groups.AtomGroup(water_hydrogen)
        elif not water_hydrogen: hydrogen = sorted_selection.hydrogens
        else: hydrogen = sorted_selection.hydrogens + _MDAnalysis.core.groups.AtomGroup(water_hydrogen)
        self._hydrogen = hydrogen
        self.heavy2hydrogen = sorted_selection.donor2hydrogens + [[] for i in sorted_selection.acceptors] + [[self._first_water_hydrogen_id+i, self._first_water_hydrogen_id+i+1] for i in range(0, len(water_hydrogen), 2)]
        self._all_ids = da_ids+self._water_ids
        self._all_ids_atomwise = da_ids_atomwise + self._water_ids_atomwise
        self._resids = _np.array([int(ids.split('-')[2]) for ids in self._all_ids])
        self.initial_graph = _nx.Graph()
        self.filtered_graph = self.initial_graph
        self.joint_occupancy_series = None
        self.joint_occupancy_frames = None
        self.residuewise=residuewise
        self.add_missing_residues = 0
        
    def filter_occupancy(self, min_occupancy, use_filtered=True):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if len(results) == 0: raise AssertionError('nothing to filter!')
        filtered_result = {key:results[key] for key in results if _np.mean(results[key])>min_occupancy}
        self.filtered_results = filtered_result
        self._generate_filtered_graph_from_filtered_results()
    
    def filter_connected_component(self, root, atomwise_whole_residue=False, use_filtered=True):
        if use_filtered: graph = self.filtered_graph
        else: graph = self.initial_graph 
        print('NODES:', graph.nodes())
        if len(graph.nodes()) == 0: raise AssertionError('nothing to filter!')
        if (not self.residuewise) and atomwise_whole_residue:
            start_points = []
            residue = root.split('-')[:3]
            for node in graph.nodes():
                if node.split('-')[:3] == residue: start_points.append(node)
        if root not in graph.nodes(): raise AssertionError('The root node is not in the current graph')
        if (not self.residuewise) and atomwise_whole_residue:    
            components = []
            for component in _nx.connected_component_subgraphs(graph):
                for start_point in start_points:
                    if start_point in component.nodes(): components.append(component)
            component = _nx.compose_all(components)
        else:  
            for component in _nx.connected_component_subgraphs(graph):
                if root in component.nodes(): break
        self.filtered_graph = component
        self._generate_filtered_results_from_filtered_graph()
    
    def filter_all_paths(self, start, goal, only_shortest=True, use_filtered=True):
        if use_filtered: graph = self.filtered_graph
        else: graph = self.initial_graph 
        if len(graph.nodes()) == 0: raise AssertionError('nothing to filter!')
        if start not in graph.nodes(): raise AssertionError('The start node is not in the graph')
        if goal not in graph.nodes(): raise AssertionError('The goal node is not in the graph')
        for component in _nx.connected_component_subgraphs(graph):
            if start in component.nodes(): break
        try: 
            if only_shortest: paths = _nx.all_shortest_paths(component, start, goal)
            else: paths = _nx.all_simple_paths(component, start, goal)
        except: raise AssertionError('start and goal nodes are not connected')
        shortest_graph = _nx.Graph()
        for path in paths:
            shortest_graph.add_edges_from(_hf.pairwise(path))
        self.filtered_graph = shortest_graph
        self._generate_filtered_results_from_filtered_graph()

    def filter_single_path(self, *nodes, use_filtered=True):
        if use_filtered: graph = self.filtered_graph
        else: graph = self.initial_graph 
        if len(graph.nodes()) == 0: raise AssertionError('nothing to filter!')
        keep_edges = []
        for resa, resb in _hf.pairwise(nodes):
            edge, edge_check = (resa, resb), (resb, resa)
            if edge in graph.edges():
                keep_edges.append(edge)
            elif edge_check in graph.edges():
                keep_edges.append(edge_check)
            else:
                raise AssertionError('There is no connection between {} and {}'.format(resa, resb))
        self.filtered_graph = _nx.Graph()
        self.filtered_graph.add_edges_from(keep_edges)
        self._generate_filtered_results_from_filtered_graph()
    
    def filter_between_segnames(self, segna, segnb=None, use_filtered=True):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if len(results) == 0: raise AssertionError('nothing to filter!')
        segnames = [segna, segnb]
        keep_bonds = []
        for key in results:
            sa,_,_, sb,_,_ = _hf.deconst_key(key, self.residuewise)
            if ((sa in segnames) and (sb in segnames) and (sa!=sb) and (segnb!=None)) or ((sa in segnames) and (sb in segnames) and (segnb==None)):
                keep_bonds.append(key)
        self.filtered_results = {key:results[key] for key in keep_bonds}
        self._generate_filtered_graph_from_filtered_results()
    
    def filter_to_frame(self, frame, use_filtered=True):
        _ = self._compute_graph_in_frame(frame, True, use_filtered)
    
    def compute_all_pairs_shortest_path_per_frame(self, max_length=5, between_nodes=None, filter_highest_occupancy=False, use_filtered=True):
        if use_filtered: 
            graph = self.filtered_graph
            if filter_highest_occupancy: results = self.filtered_results
        else: 
            graph = self.initial_graph
            if filter_highest_occupancy: results = self.initial_results
        if between_nodes != None: nodes = between_nodes
        else: nodes = graph.nodes()
        res_length = {}
        res_path = {}
        res_occ = {}
        plot_table = _np.zeros((max_length,self.nb_frames), dtype=int)
        for keya, keyb in combinations(nodes, 2):
            key = keya+':'+keyb
            temp_len=[]
            temp_paths = []
            combined_occupancy = _np.zeros(self.nb_frames, dtype=_np.bool)
            try:
                _nx.shortest_path(graph, keya, keyb)
                for frame in range(self.nb_frames):
                    frame_graph = self._compute_graph_in_frame(frame, use_filtered=use_filtered)
                    try: 
                        path = _nx.shortest_path(frame_graph, keya, keyb)
                        combined_occupancy[frame]=True
                        temp_len.append(len(path))
                        if path not in temp_paths: temp_paths.append(path)
                        try:
                            plot_table[len(path)-2][frame] += 1 
                        except:
                            pass
                    except: continue
                if not temp_len: continue
                if filter_highest_occupancy:
                    combined_occupancy = []
                    for path in temp_paths:
                        combined_occupancy.append(_np.ones(self.nb_frames, dtype=_np.bool))
                        for nodea, nodeb in _hf.pairwise(path):
                            try: combined_occupancy[-1] &= results[':'.join((nodea, nodeb))]
                            except KeyError: combined_occupancy[-1] &= results[':'.join((nodeb, nodea))]
                    for i,c in enumerate(combined_occupancy):
                        combined_occupancy[i]=c.mean()
                    max_index = _np.argmax(combined_occupancy)
                    res_occ[key] = combined_occupancy[max_index]
                    res_path[key] = [int(node.split('-')[2]) for node in temp_paths[max_index]]
                    res_length[key] = len(res_path[key])
                else:
                    occ = combined_occupancy.mean()
                    res_occ[key]=occ
                    res_path[key]= [[int(node.split('-')[2]) for node in path] for path in temp_paths]
                    res_length[key] = _np.array(temp_len).mean()
            except: pass
        return res_length, res_path, res_occ, plot_table
    
    def compute_all_pairs_shortest_path(self, max_length=5, between_nodes=None, use_filtered=True):
        if use_filtered: 
            graph = self.filtered_graph
            results = self.filtered_results
        else: 
            graph = self.initial_graph
            results = self.initial_results
        if between_nodes != None: nodes = between_nodes
        else: nodes = graph.nodes()
        res_length = {}
        res_path = {}
        res_occ = {}
        plot_table = _np.zeros((max_length,self.nb_frames))
        for keya, keyb in combinations(nodes, 2):
            try:
                key = keya+':'+keyb
                path = _nx.shortest_path(graph, keya, keyb)
                combined_occupancy = _np.ones(self.nb_frames, dtype=_np.bool)
                for nodea, nodeb in _hf.pairwise(path):
                    try: combined_occupancy &= results[':'.join((nodea, nodeb))]
                    except KeyError: combined_occupancy &= results[':'.join((nodeb, nodea))]
                res_occ[key]=combined_occupancy.mean()
                res_path[key]=[int(node.split('-')[2]) for node in path]
                res_length[key] = len(res_path[key])
                try:
                    plot_table[res_length[key]-2] += combined_occupancy
                except:
                    pass
            except _nx.NetworkXNoPath:
                pass
        return res_length, res_path, res_occ, plot_table
    
    def compute_joint_occupancy(self, use_filtered=True):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        combined_occupancy = _np.ones(self.nb_frames, dtype=_np.bool)
        for res in results: combined_occupancy &= results[res]
        self.joint_occupancy_series = combined_occupancy
        self.joint_occupancy_frames = _np.nonzero(combined_occupancy)[0]
        return combined_occupancy.mean()
     
    def draw_multi_segment_connection_timeseries(self, segnames=None, colors=None, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if segnames == None:
            segnames = _np.unique([[_hf.deconst_key(key, self.residuewise)[0], _hf.deconst_key(key, self.residuewise)[3]] for key in results])
        segname_results = {segname:{} for segname in segnames}
        for segnamea, segnameb in combinations(segnames, 2): segname_results[segnamea+'-'+segnameb]={}
        for key in results:
            res1, res2 = key.split(':')
            sn1, sn2 = res1.split('-')[0], res2.split('-')[0]
            comb, comb_check = '-'.join((sn1,sn2)), '-'.join((sn2,sn1))
            try: segname_results[comb][key] = results[key]
            except:
                try: segname_results[comb_check][key] = results[key]
                except: 
                    try: segname_results[sn1][key] = results[key]
                    except: 
                        try: segname_results[sn2][key] = results[key]
                        except: pass

        if colors == None: 
            cmap = matplotlib.cm.get_cmap('Spectral')
            colors = [cmap(val) for val in _np.linspace(0,1,len(segname_results))]
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        _plt.xlabel('Frame' , fontsize = 16)
        _plt.ylabel('Count' , fontsize = 16)    
        for i,seg in enumerate(segname_results):
            res_temp = _np.zeros(self.nb_frames)
            for bond in segname_results[seg]: res_temp += segname_results[seg][bond]
            _plt.plot(_np.arange(self.nb_frames), res_temp, color=colors[i], label=seg)
        _plt.legend()
        self._save_or_draw(filename)
        
        
    def draw_multi_segname_occupancy_histogram(self, min_occupancy, occupancy_step, segnames=None, colors=None, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if segnames == None:
            segnames = _np.unique([[_hf.deconst_key(key, self.residuewise)[0], _hf.deconst_key(key, self.residuewise)[3]] for key in results])
        occupancies = _np.arange(min_occupancy, 1., occupancy_step)
        segname_results = {segname:{} for segname in segnames}
        temp = []
        for segnamea, segnameb in combinations(segname_results, 2): temp.append(segnamea+'-'+segnameb)
        for comb in temp: segname_results[comb]={}
        for key in results:
            res1, res2 = key.split(':')
            sn1, sn2 = res1.split('-')[0], res2.split('-')[0]
            comb, comb_check = '-'.join((sn1,sn2)), '-'.join((sn2,sn1))
            try: segname_results[comb][key] = results[key]
            except:
                try: segname_results[comb_check][key] = results[key]
                except: 
                    try: segname_results[sn1][key] = results[key]
                    except: 
                        try: segname_results[sn2][key] = results[key]
                        except: pass

        res = _np.zeros((len(segname_results), len(occupancies)))
        if colors == None: 
            cmap = matplotlib.cm.get_cmap('Spectral')
            colors = [cmap(val) for val in _np.linspace(0,1,len(segname_results))]
        labels = ['{0:.{1}f}'.format(occupancy, 2) for occupancy in occupancies]
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        _plt.xticks(range(len(occupancies)), labels)
        _plt.xlabel('Occupancy' , fontsize = 16)
        _plt.ylabel('Count' , fontsize = 16)
        handles = []
        for i,segname in enumerate(segnames):
            try:
                res[i] = _np.array([len(_hf.filter_occupancy(segname_results[segname], occupancy)) for occupancy in occupancies])
            except:
                res[i] = 0
            handles.append(_plt.bar(range(len(res[i])), res[i], width=0.8, label=segname, color=colors[i], bottom=res[:i].sum(0)))
        for segname in segname_results:
            if segname in segnames: continue
            i+=1
            try:
                res[i] = _np.array([len(_hf.filter_occupancy(segname_results[segname], occupancy)) for occupancy in occupancies])
            except:
                res[i] = 0
            handles.append(_plt.bar(range(len(res[i])), res[i], width=0.8, label=segname, color=colors[i], bottom=res[:i].sum(0)))
        _plt.legend(handles=handles)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self._save_or_draw(filename)   
    
    def draw_sum_of_connections_timeseries(self, compare_to=None,legend_text=None, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        res = _np.zeros(self.nb_frames)
        for key in results:
            res += results[key]
        if compare_to != None:
            if use_filtered: results_compare = compare_to.filtered_results
            else: results_compare = compare_to.initial_results
            res_c = _np.zeros(compare_to.nb_frames)
            for key in results_compare:
                res_c += results_compare[key]
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        _plt.xlabel('Frame' , fontsize = 16)
        _plt.ylabel('Count' , fontsize = 16)
        if legend_text != None:
            _plt.plot(_np.arange(self.nb_frames), res, label=legend_text[0])
            if compare_to != None: _plt.plot(_np.arange(compare_to.nb_frames), res_c, label=legend_text[1])
            _plt.legend()
        else:
            _plt.plot(_np.arange(self.nb_frames), res)
            if compare_to != None: _plt.plot(_np.arange(compare_to.nb_frames), res_c)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self._save_or_draw(filename)
    
    def draw_connections_per_residue(self, average=False, residues_to_plot=None, compare_to=None, xtick_resnames=True, legend_text=None, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        segnames = _np.unique([[_hf.deconst_key(key, self.residuewise)[0], _hf.deconst_key(key, self.residuewise)[3]] for key in results])
        resnames = {}
        for key in results: 
            _, resna, resia, _, resnb, resib = _hf.deconst_key(key, self.residuewise)
            resnames[resia]= resna
            resnames[resib]= resnb
        residues = _np.unique([_hf.deconst_key(key, self.residuewise)[2] for key in results]+[_hf.deconst_key(key, self.residuewise)[5] for key in results])
        if average: results_per_residue = {segname:_odict({i:_np.zeros(self.nb_frames) for i in residues}) for segname in segnames}
        else: results_per_residue = {segname:_odict({i:0 for i in residues}) for segname in segnames}
        if compare_to != None:
            if use_filtered: results_compare = compare_to.filtered_results
            else: results_compare = compare_to.initial_results
            residues_compare = _np.unique([_hf.deconst_key(key, compare_to.residuewise)[2] for key in results_compare]+[_hf.deconst_key(key, compare_to.residuewise)[5] for key in results_compare])
            for key in results_compare: 
                _, resna, resia, _, resnb, resib = _hf.deconst_key(key, compare_to.residuewise)
                resnames[resia]= resna
                resnames[resib]= resnb
            segnames_compare = _np.unique([[_hf.deconst_key(key, compare_to.residuewise)[0], _hf.deconst_key(key, compare_to.residuewise)[3]] for key in results_compare])
            if average: results_per_residue_compare = {segname:_odict({i:_np.zeros(compare_to.nb_frames) for i in residues_compare}) for segname in segnames_compare}
            else: results_per_residue_compare = {segname:_odict({i:0 for i in residues_compare}) for segname in segnames_compare}
        for key in results:
            segna, resna, resida, segnb, resnb, residb = _hf.deconst_key(key, self.residuewise)
            if resna not in ['TIP3', 'HOH']: 
                if average: results_per_residue[segna][resida] += results[key]
                else: results_per_residue[segna][resida] +=1
            if resnb not in ['TIP3', 'HOH']: 
                if average: results_per_residue[segnb][residb] += results[key]
                else: results_per_residue[segnb][residb] +=1
        if average: 
            for segn in results_per_residue:
                for key in results_per_residue[segn]:
                    results_per_residue[segn][key] = results_per_residue[segn][key].mean()
        if compare_to != None:
            for key in results_compare:
                segna, resna, resida, segnb, resnb, residb = _hf.deconst_key(key, compare_to.residuewise)
                if resna not in ['TIP3', 'HOH']: 
                    if average: results_per_residue_compare[segna][resida] += results_compare[key] 
                    else: results_per_residue_compare[segna][resida] +=1
                if resnb not in ['TIP3', 'HOH']: 
                    if average: results_per_residue_compare[segnb][residb] += results_compare[key]
                    else: results_per_residue_compare[segnb][residb] +=1
            if average: 
                for segn in results_per_residue_compare:
                    for key in results_per_residue_compare[segn]:
                        results_per_residue_compare[segn][key] = results_per_residue_compare[segn][key].mean()

        if compare_to != None:
            all_residues = _np.unique(list(residues)+list(residues_compare))
            all_segnames = _np.unique(list(segnames)+list(segnames_compare))
            for segn in all_segnames:
                if segn not in results_per_residue_compare: results_per_residue_compare[segn] = {i:0 for i in all_residues}
                if segn not in results_per_residue: results_per_residue[segn] = {i:0 for i in all_residues}
                for key in all_residues:
                    if key not in results_per_residue[segn]: results_per_residue[segn][key]=0
                    if key not in results_per_residue_compare[segn]: results_per_residue_compare[segn][key]=0

            for segn in results_per_residue: results_per_residue[segn] = _odict({key:results_per_residue[segn][key] for key in sorted(list(results_per_residue[segn].keys()))})
            for segn in results_per_residue_compare: results_per_residue_compare[segn] = _odict({key:results_per_residue_compare[segn][key] for key in sorted(list(results_per_residue_compare[segn].keys()))})
            ys_compare = {segn:[] for segn in results_per_residue}

        xs, ys = {segn:[] for segn in results_per_residue}, {segn:[] for segn in results_per_residue}
        for segn in results_per_residue:
            for key in results_per_residue[segn]:
                xs[segn].append(key)
                ys[segn].append(results_per_residue[segn][key])
                if compare_to != None:
                    ys_compare[segn].append(results_per_residue_compare[segn][key])

        for segn in xs: xs[segn] = _np.array(xs[segn])+self.add_missing_residues
        for segn in ys: ys[segn] = _np.array(ys[segn])
        if compare_to != None: 
            for segn in ys_compare: ys_compare[segn] = _np.array(ys_compare[segn])
        if residues_to_plot != None:
            r_index = {segn:[] for segn in results_per_residue}
            for segn in xs:
                for i,x in enumerate(xs[segn]):
                    if x in residues_to_plot: r_index[segn].append(i)
                xs[segn] = xs[segn][r_index[segn]]
                ys[segn] = ys[segn][r_index[segn]]
                if compare_to != None:
                    ys_compare[segn] = ys_compare[segn][r_index[segn]]

        ys_bottom, ys_compare_bottom, ss = {segn:[] for segn in results_per_residue}, {segn:[] for segn in results_per_residue}, []
        for segname in xs:
            ss.append(segname)
        ys_bottom[ss[0]] = _np.zeros(len(ys[ss[0]]))
        if compare_to != None: ys_compare_bottom[ss[0]] = _np.zeros(len(ys_compare[ss[0]]))
        for i in range(1,len(ss)):
            ys_bottom[ss[i]]=ys[ss[i-1]]
            if compare_to!=None:ys_compare_bottom[ss[i]] = ys_compare[ss[i-1]]
        fig, ax = _plt.subplots()
        for segn in xs:
            
            if compare_to != None:
                if legend_text != None:
                    if ys[segn].sum()>0:_plt.bar(_np.arange(len(ys[segn])) - 0.1, ys[segn], width=0.2, label=legend_text[0]+' - '+segn, bottom=ys_bottom[segn])
                    if ys_compare[segn].sum()>0:_plt.bar(_np.arange(len(ys_compare[segn])) + 0.1, ys_compare[segn], width=0.2, label=legend_text[1]+' - '+segn, bottom=ys_compare_bottom[segn])   
                    _plt.legend()
                else:
                    _plt.bar(_np.arange(len(ys[segn])) - 0.1, ys[segn], width=0.2, bottom=ys_bottom[segn])
                    _plt.bar(_np.arange(len(ys_compare[segn])) + 0.1, ys_compare[segn], width=0.2, bottom=ys_compare_bottom[segn])  
            else:
                if legend_text != None:
                    if ys[segn].sum()>0:_plt.bar(_np.arange(len(ys[segn])), ys[segn], width=0.4, label = legend_text[0]+' - '+segn, bottom=ys_bottom[segn])
                    _plt.legend()
                else:
                    _plt.bar(_np.arange(len(ys[segn])), ys[segn], width=0.4, bottom=ys_bottom[segn])
                    
        all_xs = _np.unique([xs[key] for key in xs])
        all_labels = [_hf.aa_three2one[resnames[resi]]+str(resi) for resi in all_xs]
        _plt.xticks(_np.arange(-1, all_xs.size, 1.0), ['']+all_labels, rotation=45)
        _plt.xlim([-0.5, all_xs.size-0.5])
        _plt.xlabel('Residue' , fontsize = 16)
        _plt.ylabel('Count' , fontsize = 16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self._save_or_draw(filename)
            
    def draw_joint_timeseries(self, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        ts = _np.ones(self.nb_frames, dtype=bool)
        for key in results:
            ts &= results[key]
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        _plt.scatter(range(len(ts)), ts, s=0.005)
        _plt.xlabel('Frame' , fontsize = 16)
        ax.set_xlim(0, self.nb_frames-1)
        _plt.yticks([-0.5,0.01,1.01,1.5], ['', 'false', 'true', ''])
        _plt.tick_params(
                axis='y', 
                which='both',
                bottom=False,   
                top=False,       
                labelbottom=False,
                length=0.0,
                width=0.0) 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._save_or_draw(filename)
        
    def draw_compare_joint_timeseries(self, other_paths, descriptor='path', use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        results = [results]
        for other in other_paths:
            if use_filtered: results.append(other.filtered_results)
            else: results.append(other.initial_results)
        ts = _np.ones((len(results), self.nb_frames), dtype=bool)
        for i, result in enumerate(results):
            for key in result:
                ts[i] &= result[key]
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        occupancies = ts.mean(1)
        sorted_index = _np.argsort(occupancies)
        for i,ind in enumerate(sorted_index):
            _plt.scatter(range(len(ts[ind])), _np.array(ts[ind], dtype=int) * (i+1), s=0.5)
        _plt.xlabel('Frame' , fontsize = 16)
        ax.set_xlim(0, self.nb_frames-1)
        ax.set_ylim(0.5, len(results)+0.5)
        #_plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, length=0.0, width=0.0) 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2 = ax.twinx()
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylim(0.5, len(results)+0.5)
        ax.set_yticks(range(1, len(ts)+1))
        ax.set_yticklabels(['{}'.format(o*100) for o in occupancies[sorted_index]])
        ax.set_ylabel('Joint Occupancy [%]', fontsize=16)
        ax2.set_yticks(range(1, len(ts)+1))
        ax2.set_yticklabels([(descriptor + ' {}').format(i+1) for i in range(len(ts))])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        self._save_or_draw(filename)
        
    def draw_occupancy_histogram(self, min_occupancy, occupancy_step, compare_to=None, legend_text=None, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if compare_to != None:
            if use_filtered: results_compare = compare_to.filtered_results
            else: results_compare = compare_to.initial_results
        occupancies = _np.arange(min_occupancy, 1., occupancy_step)
        bond_counts = [len(_hf.filter_occupancy(results, occupancy)) for occupancy in occupancies]
        diff_counts = _np.array(bond_counts)
        labels = ['{0:.{1}f}'.format(occupancy, 2) for occupancy in occupancies]
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if compare_to != None:
            diff_counts_c = _np.array([len(_hf.filter_occupancy(results_compare, occupancy)) for occupancy in occupancies])
            if legend_text!=None:
                _plt.bar(_np.array(list(range(len(diff_counts))))-0.2, diff_counts, width=0.4, label=legend_text[0])
                _plt.bar(_np.array(list(range(len(diff_counts))))+0.2, diff_counts_c, width=0.4, label=legend_text[1])
                _plt.legend()
            else:
                _plt.bar(_np.array(list(range(len(diff_counts))))-0.2, diff_counts, width=0.4)
                _plt.bar(_np.array(list(range(len(diff_counts))))+0.2, diff_counts_c, width=0.4)
        else:
            if legend_text != None:
                _plt.bar(range(len(diff_counts)), diff_counts, label=legend_text[0])
                _plt.legend()
            else:
                _plt.bar(range(len(diff_counts)), diff_counts)
        _plt.xticks(range(len(diff_counts)), labels)
        _plt.xlabel('Occupancy' , fontsize = 16)
        _plt.ylabel('Count' , fontsize = 16)
        self._save_or_draw(filename)
    
    def draw_compare_occupancy_histogram(self, compare_to, xaxis_names=None, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        results = [results]
        lengths = [self.nb_frames]
        for other in compare_to:
            lengths.append(other.nb_frames)
            if use_filtered: results.append(other.filtered_results)
            else: results.append(other.initial_results)
        if xaxis_names != None: labels = xaxis_names
        else: labels = _np.arange(1,len(results)+1)
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        occupancies = _np.zeros(len(results))
        for i,result in enumerate(results):
            temp = _np.ones(lengths[i], dtype=bool)
            for key in result:
                temp &= result[key]
            occupancies[i] = temp.mean()
        _plt.bar(_np.arange(len(occupancies)), occupancies)
        _plt.xticks(range(occupancies.size), labels)
        _plt.ylabel('Occupancy' , fontsize = 16)
        self._save_or_draw(filename)
    
    def draw_time_histogram(self, nb_blocks=10, compare_to=None, legend_text=None, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if compare_to != None:
            if use_filtered: results_compare = compare_to.filtered_results
            else: results_compare = compare_to.initial_results
        values = _np.array([results[key] for key in results])
        block_values = _np.linspace(0, self.nb_frames, nb_blocks, dtype=int)
        y_hist = []
        for i, ii in _hf.pairwise(block_values): y_hist.append(values[:,i:ii].sum())
        if compare_to != None:
            values = _np.array([results_compare[key] for key in results_compare])
            block_values = _np.linspace(0, self.nb_frames, nb_blocks, dtype=int)
            y_hist_c = []
            for i, ii in _hf.pairwise(block_values): y_hist_c.append(values[:,i:ii].sum())
        labels = ['{}'.format(block_value) for block_value in block_values]
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if compare_to != None:
            if legend_text!=None:
                _plt.bar(_np.array(list(range(len(y_hist))))-0.2, y_hist, width=0.4, label=legend_text[0])
                _plt.bar(_np.array(list(range(len(y_hist_c))))+0.2, y_hist_c, width=0.4, label=legend_text[1])
                _plt.legend()
            else:
                _plt.bar(_np.array(list(range(len(y_hist))))-0.2, y_hist, width=0.4)
                _plt.bar(_np.array(list(range(len(y_hist_c))))+0.2, y_hist_c, width=0.4)
        else:
            if legend_text != None:
                _plt.bar(range(len(y_hist)), y_hist, label=legend_text[0])
                _plt.legend()
            else:
                _plt.bar(range(len(y_hist)), y_hist)
        #_plt.bar(range(len(y_hist)), y_hist)
        _plt.xticks(range(len(y_hist)), labels)
        _plt.xlabel('Frame' , fontsize = 16)
        _plt.ylabel('Count' , fontsize = 16)
        self._save_or_draw(filename)
        
    def draw_residue_range_heatmap(self, ranges, names, average=False, label='Region', use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        residues = []
        for r in ranges: residues+=list(range(r[0],r[1]+1))
        residues = _np.array(residues)
        residue_trans = _np.empty(_np.max(ranges), dtype=int)
        for i in range(len(ranges)): residue_trans[ranges[i][0]:ranges[i][1]+1]=i
        if average: results_dict = {i:{j:_np.zeros(self.nb_frames) for j in range(len(ranges))} for i in range(len(ranges))}
        results_matrix = _np.zeros((len(ranges), len(ranges)), dtype=float)
        for key in results:
            _, resna, resida, _, resnb, residb = _hf.deconst_key(key, self.residuewise)
            if resna not in ['TIP3', 'HOH'] and resnb not in ['TIP3', 'HOH'] and resida in residues and residb in residues: 
                if average: results_dict[residue_trans[resida]][residue_trans[residb]]+=results[key]
                else: results_matrix[residue_trans[resida], residue_trans[residb]] +=1
        if average:
            for i in range(len(ranges)):
                for j in range(len(ranges)):
                    results_matrix[i, j] = results_dict[i][j].mean()
        results_matrix = results_matrix + _np.tril(results_matrix.T, -1)
        results_matrix = _np.rot90(results_matrix, 3)
        fig, ax = _plt.subplots()
        _plt.imshow(results_matrix, cmap='Reds')
        _plt.colorbar(ticks=range(int(_np.max(results_matrix))+1))
        ax.set_xticks(_np.arange(len(names)))
        ax.set_yticks(_np.arange(len(names)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticklabels(names)
        names.reverse()
        ax.set_yticklabels(names)
        _plt.setp(ax.get_yticklabels(), rotation=90, ha="center", va="center", rotation_mode="anchor")
        _plt.xlabel(label , fontsize = 16)
        _plt.ylabel(label , fontsize = 16)
        self._save_or_draw(filename)
        
    def draw_residue_residue_heatmap(self, average=False, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        residues = _np.unique([_hf.deconst_key(key, self.residuewise)[2] for key in results]+[_hf.deconst_key(key, self.residuewise)[5] for key in results])
        results_matrix = _np.zeros((_np.max(residues)+1, _np.max(residues)+1), dtype=float)
        if average: results_dict = {i:{j:_np.zeros(self.nb_frames)  for j in residues} for i in residues}
        for key in results:
            _, resna, resida, _, resnb, residb = _hf.deconst_key(key, self.residuewise)
            if resida in residues and residb in residues: 
                if average: results_dict[min(resida,residb)][max(resida,residb)]+=results[key]
                else: results_matrix[resida, residb] +=1
        if average: 
            for i in residues:
                for j in residues:
                    results_matrix[i, j] = results_dict[i][j].mean()
        results_matrix = results_matrix + _np.tril(results_matrix.T, -1)
        fig, ax = _plt.subplots()
        _plt.imshow(results_matrix, cmap='Reds', origin='lower')
        _plt.colorbar(ticks=range(int(_np.max(results_matrix))+1))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        _plt.xlabel('Residue' , fontsize = 16)
        _plt.ylabel('Residue' , fontsize = 16)
        self._save_or_draw(filename)
    
    def draw_graph(self, draw_edge_occupancy=True, compare_to=None, highlight_interregion=False, default_color='seagreen', draw_labels=True, color_dict={}, filename=None, draw_base=False, use_filtered=True):
        if compare_to != None: 
            if use_filtered: 
                graph1 = self.filtered_graph
                graph2 = compare_to.filtered_graph
                results = self.filtered_results
                compare_results = compare_to.filtered_results
            else: 
                graph1 = self.initial_graph 
                graph2 = compare_to.initial_graph
                results = self.initial_results
                compare_results = compare_to.initial_results
            composed_graph = _nx.compose(graph1, graph2)
            e_new, e_gone = [], []
            for a,b in graph1.edges():
                if not (a,b) in graph2.edges():
                    e_gone.append((a,b))
            for a,b in graph2.edges():
                if not (a,b) in graph1.edges(): 
                    e_new.append((a,b))
            edges_gone, edges_new = _nx.Graph(e_gone), _nx.Graph(e_new)
        else: 
            if use_filtered: 
                composed_graph = self.filtered_graph
                results = self.filtered_results
            else: 
                composed_graph = self.initial_graph 
                results = self.initial_results
        pos = []
        nodes = []
        for node in composed_graph.nodes():
            segid, _, resid = node.split('-')[:3]
            if hasattr(self, 'first_frame_dict'):
                if len(self.first_frame_dict) != 0: self._universe.trajectory[self.first_frame_dict[node]]
            atoms = self._universe.select_atoms("segid {} and resid {}".format(segid, resid))
            pos.append(_np.mean(atoms.positions, axis=0))
            nodes.append(node)
        pos = _np.array(pos)
        pos2d, base = _hf.pca_2d_projection(pos)
    
        _plt.figure(figsize=(_np.sqrt(len(pos))*4,_np.sqrt(len(pos))*3))
        if draw_base:
            x_min, y_max = _np.min(pos2d[:,0]), _np.max(pos2d[:,1])
            arrow_pos = base.dot(_np.eye(3).T).T * 1.4
            ax_label_pos = arrow_pos + arrow_pos / _np.linalg.norm(arrow_pos, axis=1).reshape(-1,1) * 0.5
            ax_label_pos[:,0] += x_min
            ax_label_pos[:,1] += y_max
            s = ['x', 'y', 'z']
            for i in range(3):
                _plt.arrow(x_min, y_max, arrow_pos[i,0], arrow_pos[i,1])
                _plt.text(ax_label_pos[i,0], ax_label_pos[i,1], s[i], horizontalalignment='center', verticalalignment='center')
            
        pos={}
        for p, node in zip(pos2d,nodes):
            pos[node] = p
        
        _nx.draw_networkx_edges(composed_graph, pos, width=1.5, alpha=0.5)
        
        if highlight_interregion:
            interregion_graph = _nx.Graph()
            for edge in composed_graph.edges():
                if edge[0].split('-')[0]!=edge[1].split('-')[0]:
                    interregion_graph.add_edge(*edge)
            _nx.draw_networkx_edges(interregion_graph, pos, width=10., alpha=0.7)
        if compare_to != None:
            _nx.draw_networkx_edges(edges_gone, pos, width=3.5, alpha=0.7, edge_color='red')
            _nx.draw_networkx_edges(edges_new, pos, width=3.5, alpha=0.7, edge_color='green')
        labels = {}
        color_graph = {default_color:_nx.Graph()}
    
        if color_dict:
            for color in color_dict.values(): 
                color_graph[color]=_nx.Graph()
        
        for j, node in enumerate(composed_graph.nodes()):
            segname, resname, resid = node.split('-')[:3]
            try:
                if resname == 'TIP3': labels[node] = segname+str(int(resid)+self.add_missing_residues)
                else: labels[node] = _hf.aa_three2one[resname]+str(int(resid)+self.add_missing_residues)
            except KeyError:
                labels[node] = resname+resid
            try:color = color_dict[segname]
            except KeyError: color = default_color
            color_graph[color].add_node(node)
    
        for color in color_graph:
            _nx.draw_networkx_nodes(color_graph[color], pos, node_size=_np.sqrt(len(pos))*300, alpha=0.5, node_color=color)
        f = _np.sqrt(_np.sqrt(len(pos)))*0.35
        len2fontsize = _hf.defaultdict(lambda: 6*f, {2:18*f, 3:18*f, 4:16*f, 5:14*f, 6:12*f, 7:11*f, 8:10*f, 9:8*f, 10:7*f})
        for j, node in enumerate(composed_graph.nodes()):
            tempG = _nx.Graph()
            tempG.add_node(node)
            if draw_labels: _nx.draw_networkx_labels(tempG, pos, {node:labels[node]}, font_weight='bold' , font_size=len2fontsize[len(labels[node])])
        if draw_edge_occupancy:
            edge_labels = {}
            for conn in results:
                nodea, nodeb = conn.split(':')
                edge_labels[(nodea,nodeb)] = _np.round(results[conn].mean()*100, 1)
            
            if compare_to != None:
                compare_edge_labels = {}
                edge_labels_new = {}
                edge_labels_gone = {}
                for conn in compare_results:
                    nodea, nodeb = conn.split(':')
                    compare_edge_labels[(nodea,nodeb)] = _np.round(compare_results[conn].mean()*100, 1)
                
                for edge in list(edge_labels.keys()):
                    if edge in e_gone:
                        edge_labels_gone[edge] = edge_labels[edge]
                        del edge_labels[edge]
                
                for edge in list(compare_edge_labels.keys()):
                    if edge in e_new:
                        edge_labels_new[edge] = compare_edge_labels[edge]
                        del compare_edge_labels[edge]
                    
                _nx.draw_networkx_edge_labels(composed_graph, pos, edge_labels=edge_labels, label_pos=0.33, font_color='red')
                _nx.draw_networkx_edge_labels(composed_graph, pos, edge_labels=edge_labels_gone, label_pos=0.5, font_color='red')
                _nx.draw_networkx_edge_labels(composed_graph, pos, edge_labels=compare_edge_labels, label_pos=0.66, font_color='green')
                _nx.draw_networkx_edge_labels(composed_graph, pos, edge_labels=edge_labels_new, label_pos=0.5, font_color='green')
            else:
                _nx.draw_networkx_edge_labels(composed_graph, pos, edge_labels=edge_labels, label_pos=0.5)
        _plt.axis('off')
        self._save_or_draw(filename)
      
    def _compute_graph_in_frame(self, frame, set_as_filtered_results=False, use_filtered=True):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        keep_edges=[]
        for key in results:
            if results[key][frame]:
                keep_edges.append((key.split(':')[0], key.split(':')[1]))
        graph = _nx.Graph()
        graph.add_edges_from(keep_edges)
        if set_as_filtered_results: 
            self.filtered_graph = graph
            self._generate_filtered_results_from_filtered_graph()
        return graph
    
    def _generate_current_results_from_graph(self):
        temp_res = {}
        for resa, resb in self.initial_graph.edges():
            key, key_check = ':'.join((resa, resb)), ':'.join((resb, resa))
            try: temp_res[key] = self.initial_results[key]
            except KeyError:  temp_res[key_check] = self.initial_results[key_check]
        self.initial_results = temp_res 
        
    def _generate_filtered_results_from_filtered_graph(self):
        temp_res = {}
        for resa, resb in self.filtered_graph.edges():
            key, key_check = ':'.join((resa, resb)), ':'.join((resb, resa))
            try: temp_res[key] = self.initial_results[key]
            except KeyError:  temp_res[key_check] = self.initial_results[key_check]
        self.filtered_results = temp_res 
    
    def _generate_graph_from_current_results(self):
        self.initial_graph = _hf.dict2graph(self.initial_results, self.residuewise)
        
    def _generate_filtered_graph_from_filtered_results(self):
        self.filtered_graph = _hf.dict2graph(self.filtered_results, self.residuewise)
    
class HbondAnalysis(NetworkAnalysis):
    
    def __init__(self, selection=None, structure=None, trajectories=None, distance=3.5, cut_angle=60., 
                 start=None, stop=None, step=1, additional_donors=[], 
                 additional_acceptors=[], exclude_donors=[], exclude_acceptors=[], 
                 special_naming=[], check_angle=True, residuewise=True, add_donors_without_hydrogen=False, restore_filename=None):
        
        super(HbondAnalysis, self).__init__(selection=selection, structure=structure, trajectories=trajectories, 
             distance=distance, cut_angle=cut_angle, start=start, stop=stop, step=step, 
             additional_donors=additional_donors, additional_acceptors=additional_acceptors,
             exclude_donors=exclude_donors, exclude_acceptors=exclude_acceptors,
             special_naming=special_naming, check_angle=check_angle, residuewise=residuewise,
             add_donors_without_hydrogen=add_donors_without_hydrogen, restore_filename=restore_filename)
    
    
    def set_hbonds_in_selection(self, exclude_backbone_backbone=True):
        if exclude_backbone_backbone: backbone_filter = _np.array([(ids.split('-')[3] in ['O', 'N']) for ids in self._all_ids_atomwise])
        frame_count = 0
        frames = self.nb_frames
        result = {}
        if self._nb_acceptors == 0: raise AssertionError('No acceptors in the selection')
        if self._nb_donors == 0: raise AssertionError('No donors in the selection')
        for ts in self._universe.trajectory[self._trajectory_slice]:
            selection_coordinates = self._da_selection.positions
            d_tree = _sp.cKDTree(self._donors.positions)
            a_tree = _sp.cKDTree(self._acceptors.positions)
            hydrogen_coordinates = self._hydrogen.positions
    
            da_pairs = _np.array([[i, j] for i,donors in enumerate(a_tree.query_ball_tree(d_tree, self.distance)) for j in donors])
            da_pairs[:,0] += self._nb_donors
            if exclude_backbone_backbone: da_pairs = da_pairs[_np.logical_not(_np.all(backbone_filter[da_pairs], axis=1))]

            if self.check_angle:
                all_coordinates = selection_coordinates
                local_hbonds = _hf.check_angle(da_pairs, self.heavy2hydrogen, all_coordinates, hydrogen_coordinates, self.cut_angle)
            else:
                local_hbonds = da_pairs
            
            sorted_bonds = _np.sort(local_hbonds)
            check = self._resids[sorted_bonds]
            check = check[:,0] < check[:,1]
            frame_res = [self._all_ids[i] + ':' + self._all_ids[j] if check[ii] else self._all_ids[j] + ':' + self._all_ids[i] for ii, (i, j) in enumerate(sorted_bonds)]
            
            for bond in frame_res:
                if not self.check_angle:
                    a, b = bond.split(':')
                    if a.split('-')[:3] == b.split('-')[:3]: continue
                try:
                    result[bond][frame_count] = True
                except:
                    result[bond] = _np.zeros(frames, dtype=bool)
                    result[bond][frame_count] = True
            frame_count+=1
        self._set_results(result)


    def set_hbonds_only_water_in_convex_hull(self):
        result = {}
        frame_count = 0
        frames = self.nb_frames
        
        for ts in self._universe.trajectory[self._trajectory_slice]:
            water_coordinates = self._water.positions
            select_coordinates = self._da_selection.positions
            hydrogen_coordinates = self._hydrogen.positions[self._first_water_hydrogen_id:]
            
            hull = _sp.Delaunay(select_coordinates)
            local_index = (hull.find_simplex(water_coordinates) != -1).nonzero()[0]
    
            local_water_coordinates = water_coordinates[local_index]
            local_water_tree = _sp.cKDTree(local_water_coordinates)
            water_pairs = local_index[_np.array([pair for pair in local_water_tree.query_pairs(self.distance)])]
            
            if self.check_angle:
                local_hbonds = _hf.check_angle_water(water_pairs, water_coordinates, hydrogen_coordinates, self.cut_angle)
            else:
                local_hbonds = water_pairs
            
            sorted_bonds = _np.sort(local_hbonds)
            check = self._resids[sorted_bonds]
            check = check[:,0] < check[:,1]
            frame_res = [self._all_ids[i] + ':' + self._all_ids[j] if check[ii] else self._all_ids[j] + ':' + self._all_ids[i] for ii, (i, j) in enumerate(sorted_bonds)]            
            
            for bond in frame_res:
                if not self.check_angle:
                    a, b = bond.split(':')
                    if a.split('-')[:3] == b.split('-')[:3]: continue
                try:
                    result[bond][frame_count] = True
                except:
                    result[bond] = _np.zeros(frames, dtype=bool)
                    result[bond][frame_count] = True
            frame_count+=1
        self._set_results(result)    
    
    def set_hbonds_in_selection_and_water_in_convex_hull(self, exclude_backbone_backbone=True):
        if exclude_backbone_backbone: backbone_filter = _np.array([(ids.split('-')[3] in ['O', 'N']) for ids in self._all_ids_atomwise])        
        result = {}
        frame_count = 0
        frames = self.nb_frames
    
        for ts in self._universe.trajectory[self._trajectory_slice]:
            water_coordinates = self._water.positions
            select_coordinates = self._da_selection.positions
            hydrogen_coordinates = self._hydrogen.positions
            
            selection_tree = _sp.cKDTree(select_coordinates)
            if self._nb_acceptors > 0 and self._nb_donors > 0: 
                d_tree = _sp.cKDTree(self._donors.positions)
                a_tree = _sp.cKDTree(self._acceptors.positions)
                
            hull = _sp.Delaunay(select_coordinates)
            local_water_index = (hull.find_simplex(water_coordinates) != -1).nonzero()[0]
    
            local_water_coordinates = water_coordinates[local_water_index]
            local_water_tree = _sp.cKDTree(local_water_coordinates)
            
            local_pairs = [(i, local_water_index[j]+self._first_water_id) for i, bla in enumerate(selection_tree.query_ball_tree(local_water_tree, self.distance)) for j in bla]
            water_pairs = [(local_water_index[p[0]]+self._first_water_id, local_water_index[p[1]]+self._first_water_id) for p in local_water_tree.query_pairs(self.distance)]
            if self._nb_acceptors > 0 and self._nb_donors > 0: 
                da_pairs = _np.array([[i, j] for i,donors in enumerate(a_tree.query_ball_tree(d_tree, self.distance)) for j in donors])
                da_pairs[:,0] += self._nb_donors
            else: da_pairs = []
            if exclude_backbone_backbone and da_pairs: da_pairs = da_pairs[_np.logical_not(_np.all(backbone_filter[da_pairs], axis=1))]
            
            if self.check_angle:
                all_coordinates = _np.vstack((select_coordinates, water_coordinates))
                hbonds = _hf.check_angle(list(da_pairs)+water_pairs+local_pairs, self.heavy2hydrogen, all_coordinates, hydrogen_coordinates, self.cut_angle)
            else:
                hbonds = list(da_pairs) + water_pairs + local_pairs
                
            sorted_bonds = _np.sort(hbonds)
            check = self._resids[sorted_bonds]
            check = check[:,0] < check[:,1]
            frame_res = [self._all_ids[i] + ':' + self._all_ids[j] if check[ii] else self._all_ids[j] + ':' + self._all_ids[i] for ii, (i, j) in enumerate(sorted_bonds)]
            
            for bond in frame_res:
                if not self.check_angle:
                    a, b = bond.split(':')
                    if a.split('-')[:3] == b.split('-')[:3]: continue
                try:
                    result[bond][frame_count] = True
                except:
                    result[bond] = _np.zeros(frames, dtype=bool)
                    result[bond][frame_count] = True
            frame_count+=1
        self._set_results(result)
    
    
    def set_hbonds_in_selection_and_water_around(self, around_radius, exclude_backbone_backbone=True):
        if exclude_backbone_backbone: backbone_filter = _np.array([(ids.split('-')[3] in ['O', 'N']) for ids in self._all_ids_atomwise])
        result = {}
        frame_count = 0
        frames = self.nb_frames
        
        for ts in self._universe.trajectory[self._trajectory_slice]:
            
            water_coordinates = self._water.positions
            selection_coordinates = self._da_selection.positions
            
            selection_tree = _sp.cKDTree(selection_coordinates)
            if self._nb_acceptors > 0 and self._nb_donors > 0:
                d_tree = _sp.cKDTree(self._donors.positions)
                a_tree = _sp.cKDTree(self._acceptors.positions)
            hydrogen_coordinates = self._hydrogen.positions
    
            water_tree = _sp.cKDTree(water_coordinates, leafsize=32)
            local_water_index = []
            [local_water_index.extend(l) for l in water_tree.query_ball_point(selection_coordinates, float(around_radius))]
            local_water_index = _np.unique(local_water_index)
    
            local_water_coordinates = water_coordinates[local_water_index]
            local_water_tree = _sp.cKDTree(local_water_coordinates)
            
            local_pairs = [(i, local_water_index[j]+self._first_water_id) for i, bla in enumerate(selection_tree.query_ball_tree(local_water_tree, self.distance)) for j in bla]
            water_pairs = [(local_water_index[p[0]]+self._first_water_id, local_water_index[p[1]]+self._first_water_id) for p in local_water_tree.query_pairs(self.distance)]
            if self._nb_acceptors > 0 and self._nb_donors > 0: 
                da_pairs = _np.array([[i, j] for i,donors in enumerate(a_tree.query_ball_tree(d_tree, self.distance)) for j in donors])
                da_pairs[:,0] += self._nb_donors
            else: da_pairs = []
            if exclude_backbone_backbone and da_pairs != []: da_pairs = da_pairs[_np.logical_not(_np.all(backbone_filter[da_pairs], axis=1))]
            
            if self.check_angle:
                all_coordinates = _np.vstack((selection_coordinates, water_coordinates))
                hbonds = _hf.check_angle(list(da_pairs)+water_pairs+local_pairs, self.heavy2hydrogen, all_coordinates, hydrogen_coordinates, self.cut_angle)
            else:
                hbonds = list(da_pairs) + water_pairs + local_pairs
    
            hbonds = _np.array(hbonds)
            
            sorted_bonds = _np.sort(hbonds)
            check = self._resids[sorted_bonds]
            check = check[:,0] < check[:,1]
            frame_res = [self._all_ids[i] + ':' + self._all_ids[j] if check[ii] else self._all_ids[j] + ':' + self._all_ids[i] for ii, (i, j) in enumerate(sorted_bonds)]
            
            for bond in frame_res:
                if not self.check_angle:
                    a, b = bond.split(':')
                    if a.split('-')[:3] == b.split('-')[:3]: continue
                try:
                    result[bond][frame_count] = True
                except:
                    result[bond] = _np.zeros(frames, dtype=bool)
                    result[bond][frame_count] = True
            frame_count+=1
        self._set_results(result)


    def filter_water_in_hydration_shells(self, hydration_shells=3, use_filtered=True):
        if use_filtered: 
            results = self.filtered_results
            graph = self.filtered_graph
        else: 
            results = self.initial_results
            graph = self.initial_graph
        if len(results) == 0: raise AssertionError('nothing to filter!')
        filtered_result = {}
        hysh_lengths = []
        res_nodes = [node for node in graph.nodes() if node.split('-')[1] not in ['TIP3', 'HOH']]
        for res_node in res_nodes:
            hysh_lengths += _nx.single_source_shortest_path_length(graph, res_node, hydration_shells)
        
        graph.remove_nodes_from(set(graph.nodes())-set(hysh_lengths))
        
        filtered_result = {}
        for edge in graph.edges():
            bond_name = edge[0]+':'+edge[1]
            check = edge[1]+':'+edge[0]
            try : filtered_result[bond_name] = results[bond_name]
            except KeyError: 
                filtered_result[check] = results[check]
        
        self.filtered_results = filtered_result
        self._generate_filtered_graph_from_filtered_results()

        
    def compute_i4_bonds(self, use_filtered=True, return_print_table=False, print_table=True):
        if self.residuewise: raise AssertionError('need to initialize with residuewise=False!')
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if len(results) == 0: raise AssertionError('nothing to filter!')
        
        l = _np.array([bond for bond in results])
        segname_pairs = _np.array([[bond.split(':')[0].split('-')[0],bond.split(':')[1].split('-')[0]]for bond in results])
        atomname_pairs = _np.array([[bond.split(':')[0].split('-')[3],bond.split(':')[1].split('-')[3]]for bond in results])
        resnames_pairs = _np.array([[bond.split(':')[0].split('-')[1],bond.split(':')[1].split('-')[1]]for bond in results])
        resid_pairs = _np.array([[bond.split(':')[0].split('-')[2],bond.split(':')[1].split('-')[2]]for bond in results], dtype=_np.int)
        ser_thr_filter = _np.array([(ids[0] in ['SER', 'THR']) or (ids[1] in ['SER', 'THR']) for i, ids in enumerate(resnames_pairs)])
        glu_asp_filter = _np.array([(ids[0] in ['GLU', 'ASP']) or (ids[1] in ['GLU', 'ASP']) for i, ids in enumerate(resnames_pairs)])
        backbone_filter = _np.array([not (name[0] in ['O','N'] or name[1] in ['O','N']) for name in atomname_pairs])
        backbone_backbone_filter = _np.array([not (name[0] in ['O','N'] and name[1] in ['O','N']) for name in atomname_pairs])
        both_filter = ser_thr_filter & glu_asp_filter & backbone_filter & backbone_backbone_filter
        res_thr_filter = _np.array([[pair[0] in ['SER', 'THR'], not (pair[0] in ['SER', 'THR'])] for pair in resnames_pairs], dtype=_np.bool)
        i4_filter = (((resid_pairs[:, 0] - resid_pairs[:,1] == 3) & res_thr_filter[:,0]) | ((resid_pairs[:, 0] - resid_pairs[:,1] == -3) & res_thr_filter[:,1]) | ((resid_pairs[:, 0] - resid_pairs[:,1] == 4) & res_thr_filter[:,0]) | ((resid_pairs[:, 0] - resid_pairs[:,1] == -4) & res_thr_filter[:,1]) | ((resid_pairs[:, 0] - resid_pairs[:,1] == 5) & res_thr_filter[:,0]) | ((resid_pairs[:, 0] - resid_pairs[:,1] == -5) & res_thr_filter[:,1])) & _np.logical_not(backbone_filter) & backbone_backbone_filter & ser_thr_filter
        #if i4_filter.sum() == 0: raise AssertionError('No i-4 intrahelical hbonds found for SER/THR and no sidechain hbonds between SER/THR and GLU/ASP found')
        class_dict = {}
        res3={}
        output = ''
        translate_key = {l[i]:segname_pairs[i][0]+'-'+resnames_pairs[i][0]+'-'+str(resid_pairs[i][0]+self.add_missing_residues)+'-'+atomname_pairs[i][0]+':'+segname_pairs[i][1]+'-'+resnames_pairs[i][1]+'-'+str(resid_pairs[i][1]+self.add_missing_residues)+'-'+atomname_pairs[i][1] for i in range(len(results))}
        for i in range(len(l)):
            key = l[i]
            if both_filter[i]: 
                resid = resid_pairs[i][res_thr_filter[i]]
                segname = segname_pairs[i][res_thr_filter[i]]
                resname = resnames_pairs[i][res_thr_filter[i]]
                atomname = atomname_pairs[i][res_thr_filter[i]]
                other_bond_filter = (resid_pairs == resid).any(1) & (segname_pairs == segname).any(1) & (resnames_pairs == resname).any(1) & (atomname_pairs == atomname).any(1) & i4_filter
                if other_bond_filter.sum() == 1:
                    j = _np.argmax(other_bond_filter)
                    sel_i, sel_j = _np.logical_not(res_thr_filter[i]), _np.logical_not(res_thr_filter[j])
                    if not (segname_pairs[i][sel_i] == segname_pairs[j][sel_j] and resnames_pairs[j][sel_j] == resnames_pairs[i][sel_i] and resid_pairs[j][sel_j] == resid_pairs[i][sel_i]):
                        joint_timeseries = results[l[j]] & results[l[i]]
                        if joint_timeseries.mean() != 0.0:
                            res3[l[i]] = l[j]
                class_dict[l[i]] = 2
            if i4_filter[i]:
                class_dict[l[i]] = 1
        segnames = _np.unique(segname_pairs)
        segname_results = {segname:{1:{}, 2:{}} for segname in segnames}
        temp = []
        for segnamea, segnameb in combinations(segname_results, 2): temp.append(segnamea+'-'+segnameb)
        for comb in temp: segname_results[comb]={1:{}, 2:{}}
        for key in class_dict:
            residue1, residue2 = key.split(':')
            sn1, sn2 = residue1.split('-')[0], residue2.split('-')[0]
            comb, comb_check = '-'.join((sn1,sn2)), '-'.join((sn2,sn1))
            try: segname_results[comb][class_dict[key]][key] = results[key]
            except:
                try: segname_results[comb_check][class_dict[key]][key] = results[key]
                except: 
                    try: segname_results[sn1][class_dict[key]][key] = results[key]
                    except: 
                        try: segname_results[sn2][class_dict[key]][key] = results[key]
                        except: pass
        for segname in segname_results:
            class3_text=''
            for class_id in segname_results[segname]:
                for key in segname_results[segname][class_id]:
                    output += translate_key[key]+ '.'*(45-len(translate_key[key])) + ' | '+ 'class ' + str(class_dict[key]) + ' | '+ str(_np.round(results[key].mean()*100,1))+'\n'
                    if key in res3:
                       class3_text += translate_key[key]+ '.'*(45-len(translate_key[key])) + ' | '+ 'class ' + str(3) + ' | '+ str(_np.round((results[key] & results[res3[key]]).mean()*100,1))+'\n'
                       class3_text += translate_key[res3[key]]+ '.'*(45-len(translate_key[res3[key]])) + ' | '+ 'class ' + str(3) + ' | '+ str(_np.round((results[key] & results[res3[key]]).mean()*100,1))+'\n'
            output+=class3_text+'\n'
        if print_table:
            print(output)
        if return_print_table:
            return output
        return class_dict, res3
    
    
    def draw_i4_motif_distribution(self, filename=None):
        class_dict, res3 = self.compute_i4_bonds(print_table=False)
        z = {i:[] for i in range(1,4)}
        
        for key in class_dict:
            segida, resna, resida, atoma = key.split(':')[0].split('-')
            segidb, resnb, residb, atomb = key.split(':')[1].split('-')
            atoms = self._mda_selection.select_atoms("(segid {} and resid {} and name {}) or (segid {} and resid {} and name {})".format(segida, resida, atoma, segidb, residb, atomb))
            z[class_dict[key]]+=[atom for atom in atoms]
        for key in res3:
            for i in range(2):
                segida, resna, resida, atoma = key.split(':')[0].split('-')
                segidb, resnb, residb, atomb = key.split(':')[1].split('-')
                atoms = self._mda_selection.select_atoms("(segid {} and resid {} and name {}) or (segid {} and resid {} and name {})".format(segida, resida, atoma, segidb, residb, atomb))
                z[3]+=[atom for atom in atoms]
                if i==0: key = res3[key]
        for key in z:
            if isinstance(z[key], list) and z[key] != []: 
                z[key] = _MDAnalysis.core.AtomGroup.AtomGroup(z[key])
        z_plot = {i:[] for i in range(1,4)}
        for ts in self._universe.trajectory[self._trajectory_slice]:
            for key in z:
                if not isinstance(z[key], list): 
                    if key==3: z_plot[key].append(z[key].positions[:,2].reshape(-1,4).mean(1))
                    else: z_plot[key].append(z[key].positions[:,2].reshape(-1,2).mean(1))
        z_plot = [_np.array(z_plot[key]).mean(0) for key in z_plot if z_plot[key] != []]
        mi, ma = _np.min([a.min() for a in z_plot]), _np.max([a.max() for a in z_plot])
        fig, ax = _plt.subplots()
        _plt.hist(z_plot, 
                 _np.linspace(mi, ma, 10),
                 histtype='bar',
                 orientation=u'horizontal',
                 stacked=False,  
                 fill=True,
                 label=['class 1','class 2','class 3'],
                 alpha=0.8, # opacity of the bars
                 edgecolor = "k")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        _plt.yticks(_np.round(_np.linspace(mi, ma, 10), 1))
        _plt.ylim([mi-1,ma+1])
        _plt.xlabel('Count' , fontsize = 16)
        _plt.ylabel('Adjusted Z Coordinate' , fontsize = 16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _plt.legend()
        self._save_or_draw(filename)  
        
    def draw_hbond_layer_occupancy_histogram(self, min_occupancy, occupancy_step, use_filtered=True, filename=None):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        if len(results) == 0: raise AssertionError('nothing to filter!')
        occupancies = _np.arange(min_occupancy, 1., occupancy_step)
        bonds_list = [_hf.filter_occupancy(results, occupancy) for occupancy in occupancies]
        graphs = [_hf.dict2graph(bonds, self.residuewise) for bonds in bonds_list]
        hysh_counts = _np.zeros((occupancies.size, 3))
        
        for i, occupancy in enumerate(occupancies):
            hysh_shells = [[],[],[]]
            res_nodes = [node for node in graphs[i].nodes() if node.split('-')[1] not in ['TIP3', 'HOH']]
            for res_node in res_nodes:
                hysh_lengths = _nx.single_source_shortest_path_length(graphs[i],res_node, 3)
                del hysh_lengths[res_node]
                for hydration_water in hysh_lengths:
                    if hydration_water.split('-')[1]!='TIP3': continue
                    hysh_shells[hysh_lengths[hydration_water]-1].append(hash(hydration_water))
            
            hysh_counts[i][2] = _np.unique(_np.array(hysh_shells[2])[_np.in1d(hysh_shells[2],hysh_shells[1]+hysh_shells[0], invert=True)]).size
            hysh_counts[i][1] = _np.unique(_np.array(hysh_shells[1])[_np.in1d(hysh_shells[1],hysh_shells[0], invert=True)]).size
            hysh_counts[i][0] = _np.unique(hysh_shells[0]).size
        
        labels = ['{0:.{1}f}'.format(occupancy, 2) for occupancy in occupancies]
        diff_counts = hysh_counts.T
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        second = _plt.bar(_np.arange(len(diff_counts[1]), dtype=int)*4, diff_counts[1], label='Second Layer')
        _plt.xticks(_np.arange(len(diff_counts[1]), dtype=int)*4, labels)
        first = _plt.bar(_np.arange(len(diff_counts[0]), dtype=int)*4 - 1, diff_counts[0], label='First Layer')
        third = _plt.bar(_np.arange(len(diff_counts[2]), dtype=int)*4 + 1, diff_counts[2], label='Third Layer')
        _plt.legend(handles=[first, second, third])
        _plt.xlabel('Occupancy rate' , fontsize = 16)
        _plt.ylabel('Count' , fontsize = 16)
        if filename != None:
            _plt.savefig(filename)
        else:
            _plt.show()
            
    
class WireAnalysis(NetworkAnalysis):
    
    def __init__(self, selection=None, structure=None, trajectories=None, distance=3.5, cut_angle=60., 
                 start=None, stop=None, step=1, additional_donors=[], 
                 additional_acceptors=[], exclude_donors=[], exclude_acceptors=[], 
                 special_naming=[], check_angle=True, residuewise=True, add_donors_without_hydrogen=False, restore_filename=None):
        
        super(WireAnalysis, self).__init__(selection=selection, structure=structure, trajectories=trajectories, 
             distance=distance, cut_angle=cut_angle, start=start, stop=stop, step=step, 
             additional_donors=additional_donors, additional_acceptors=additional_acceptors,
             exclude_donors=exclude_donors, exclude_acceptors=exclude_acceptors,
             special_naming=special_naming, check_angle=check_angle, residuewise=residuewise,
             add_donors_without_hydrogen=add_donors_without_hydrogen, restore_filename=restore_filename)
        
        if restore_filename != None: return
        if not self._mda_selection:  raise AssertionError('No atoms match the selection')
        sorted_selection = _hf.Selection(self._mda_selection, self.donor_names, self.acceptor_names, add_donors_without_hydrogen=add_donors_without_hydrogen)
        if not sorted_selection.donors: da_selection = sorted_selection.acceptors
        elif not sorted_selection.acceptors: da_selection = sorted_selection.donors
        else: da_selection = _MDAnalysis.core.groups.AtomGroup(sorted_selection.donors + sorted_selection.acceptors)
        da_ids = _hf.MDA_info_list(da_selection, special_naming=special_naming)
        self.hashs = {}
        self.hash_table = {}
        self.wire_lengths = {}
        da_u, da_ind, da_inv = _np.unique(da_ids, return_index=True, return_inverse=True)
        self.da_trans = da_ind[da_inv]
    
    def set_shortest_paths(self, max_len):
        
        distances = {}
        path_hashs = {}
        frame_count = 0
        hash_table = {}
        
        for ts in self._universe.trajectory[self._trajectory_slice]:
        
            water_coordinates = self._water.positions
            selection_coordinates = self._da_selection.positions
            
            selection_tree = _sp.cKDTree(selection_coordinates)
            d_tree = _sp.cKDTree(self._donors.positions)
            a_tree = _sp.cKDTree(self._acceptors.positions)
            hydrogen_coordinates = self._hydrogen.positions
    
            water_tree = _sp.cKDTree(water_coordinates, leafsize=32)
            local_water_index = []
            [local_water_index.extend(l) for l in water_tree.query_ball_point(selection_coordinates, float(max_len+1)*self.distance/2.)]
            local_water_index = _np.unique(local_water_index)
    
            local_water_coordinates = water_coordinates[local_water_index]
            local_water_tree = _sp.cKDTree(local_water_coordinates)
            
            local_water_index += self._first_water_id
            local_pairs = [(i, local_water_index[j]) for i, bla in enumerate(selection_tree.query_ball_tree(local_water_tree, self.distance)) for j in bla]
            water_pairs = [(local_water_index[p[0]], local_water_index[p[1]]) for p in local_water_tree.query_pairs(self.distance)]
            da_pairs = _np.array([[i, j] for i,donors in enumerate(a_tree.query_ball_tree(d_tree, self.distance)) for j in donors])
            da_pairs[:,0] += self._nb_donors
            
            if self.check_angle:
                all_coordinates = _np.vstack((selection_coordinates, water_coordinates))
                hbonds = _hf.check_angle(da_pairs+water_pairs+local_pairs, self.heavy2hydrogen, all_coordinates, hydrogen_coordinates, self.cut_angle)
            else:
                hbonds = da_pairs + water_pairs + local_pairs
    
            hbonds = _np.sort(_np.array(hbonds))
            hbonds[hbonds < self._first_water_id] = self.da_trans[hbonds[hbonds < self._first_water_id]]
            uniques, rowsncols = _np.unique(hbonds, return_inverse=True)
            rows, cols = rowsncols.reshape(hbonds.shape).T
            nb_nodes = uniques.size
            residues = (uniques < self._first_water_id).nonzero()[0]
            data = _np.ones(len(hbonds), dtype=_np.bool)
            g = csr_matrix((data, (rows, cols)), shape=(nb_nodes, nb_nodes))
            distance_matrix, predecessors = dijkstra(g, directed=False, indices=residues, unweighted=True, limit=max_len+1, return_predecessors=True)
            distance_matrix -= 1.0 
            a, b = _np.nonzero((distance_matrix[:, residues] + _np.tril(_np.ones((residues.size,residues.size)) * _np.inf)) != _np.inf)
            connected = _np.empty((len(a),2), dtype=int)
            connected[:,0] = residues[a]
            connected[:,1] = residues[b]
       
            for ai, bi in connected:
                d = distance_matrix[ai, bi]
                wire = uniques[[_hf.predecessor_recursive(ii,predecessors,ai,bi) for ii in range(int(d))[::-1]]]
                ai, bi = _np.sort(uniques[[ai, bi]])
                wire_hash = hash(wire.tobytes())
                hash_table[wire_hash] = wire
                try:
                    distances[self._all_ids[ai]+':'+self._all_ids[bi]][frame_count] = d
                    path_hashs[self._all_ids[ai]+':'+self._all_ids[bi]][frame_count] = wire_hash
                except KeyError:
                    distances[self._all_ids[ai]+':'+self._all_ids[bi]] = _np.ones(self.nb_frames, dtype=int) * _np.inf
                    path_hashs[self._all_ids[ai]+':'+self._all_ids[bi]] = _np.arange(self.nb_frames, dtype=int)
                    distances[self._all_ids[ai]+':'+self._all_ids[bi]][frame_count] = d
                    path_hashs[self._all_ids[ai]+':'+self._all_ids[bi]][frame_count] = wire_hash
    
            frame_count += 1
    
        self._set_results({key:distances[key]!=_np.inf for key in distances})
        self.wire_lengths = distances 
        self.hashs = path_hashs
        self.hash_table = hash_table


    def set_water_wires(self, max_water, allow_direct_bonds=True):
        
        intervals_results = {}
        results = {}
        frame_count = 0
        frames = self.nb_frames
        this_frame_table = {}
    
        for ts in self._universe.trajectory[self._trajectory_slice]:
    
            water_coordinates = self._water.positions
            selection_coordinates = self._da_selection.positions
            water_tree = _sp.cKDTree(water_coordinates, leafsize=32)
            selection_tree = _sp.cKDTree(selection_coordinates)
            d_tree = _sp.cKDTree(self._donors.positions)
            a_tree = _sp.cKDTree(self._acceptors.positions)
            hydrogen_coordinates = self._hydrogen.positions
    
            local_water_index = []
            [local_water_index.extend(l) for l in water_tree.query_ball_point(selection_coordinates, float(max_water+1)*self.distance/2.)]
            local_water_index = _np.unique(local_water_index)
            local_water_coordinates = water_coordinates[local_water_index]
            local_water_tree = _sp.cKDTree(local_water_coordinates)
            
            local_water_index += self._first_water_id
            local_pairs = [(i, local_water_index[j]) for i, bla in enumerate(selection_tree.query_ball_tree(local_water_tree, self.distance)) for j in bla]
            water_pairs = local_water_index[_np.array(list(local_water_tree.query_pairs(self.distance)))]
            da_pairs = _np.array([[i, j] for i,donors in enumerate(a_tree.query_ball_tree(d_tree, self.distance)) for j in donors])
            da_pairs[:,0] += self._nb_donors
            
            if self.check_angle:
                all_coordinates = _np.vstack((selection_coordinates, water_coordinates))
                da_hbonds = _hf.check_angle(da_pairs, self.heavy2hydrogen, all_coordinates, hydrogen_coordinates, self.cut_angle)
                water_hbonds = _hf.check_angle(water_pairs, self.heavy2hydrogen, all_coordinates, hydrogen_coordinates, self.cut_angle)
                local_hbonds = _hf.check_angle(local_pairs, self.heavy2hydrogen, all_coordinates, hydrogen_coordinates, self.cut_angle)
            else:
                da_hbonds = da_pairs
                water_hbonds = water_pairs
                local_hbonds = local_pairs
            
            da_hbonds = _np.sort(da_hbonds)
            water_hbonds = _np.sort(water_hbonds)
            local_hbonds = _np.sort(_np.array(local_hbonds))
            local_hbonds[:,0]=self.da_trans[local_hbonds[:,0]]
            
            g = _nx.Graph()
            g.add_edges_from(water_hbonds)
            
            residues = _np.unique(local_hbonds[:,0])
            already_checked=[]
            for source in residues:
                already_checked_targets = []
                source_water_index = local_hbonds[:,0]==source
    
                if not source_water_index.any(): continue
    
                g.add_edges_from(local_hbonds[source_water_index])
                paths = _nx.single_source_shortest_path(g,source,max_water)
                g.remove_node(source)
    
                idx = _np.array([self._all_ids[bl]!=self._all_ids[source] for bl in local_hbonds[:,0]])
                target_water_set = set(paths) & set(local_hbonds[:,1][idx])
                twlist = list(target_water_set)
    
                if not target_water_set: continue
                
                all_targets_index = _np.in1d(local_hbonds[:,1], _np.array(twlist))
                
                for target, last_water in local_hbonds[all_targets_index]:
    
                    if target in already_checked or target in already_checked_targets or self._all_ids[source]==self._all_ids[target]: continue
                    wire = paths[last_water] + [target]
                    wire_hash = hash(str(wire))
                    this_frame_table[wire_hash] = wire
                    wire_info = self._all_ids[source]+':'+self._all_ids[target]
                    water_in_wire = len(wire)-2
                    
                    try: 
                        water_in_wire_before = results[wire_info][frame_count-1]
                        water_already_found = results[wire_info][frame_count]
                    except: 
                        water_in_wire_before = _np.inf
                        water_already_found = _np.inf
                        
                    if (water_in_wire > water_already_found): continue    
    
                    try: 
                        last_wire_hash = intervals_results[wire_info][frame_count-1]
                        if (intervals_results[wire_info][frame_count] != 0) and (last_wire_hash == intervals_results[wire_info][frame_count]): continue
                    except: last_wire_hash=0
                    
                    if (wire_hash!=last_wire_hash):
                        try:
                            cut = water_in_wire_before + 1
                            if last_wire_hash in [hash(str(wire1)) for wire1 in _hf.all_shortest_paths(g,source,target,cut)]:
                                water_in_wire = water_in_wire_before
                                wire_hash = last_wire_hash
                                already_checked_targets.append(target)
                        except: pass
                    
                    check = self._all_ids[target]+':'+self._all_ids[source]
                    if check in results: wire_info=check
                    try:
                        results[wire_info][frame_count] = water_in_wire
                        intervals_results[wire_info][frame_count] = wire_hash
                    except:
                        results[wire_info] = _np.ones(frames)*_np.inf
                        results[wire_info][frame_count] = water_in_wire
                        intervals_results[wire_info] = _np.arange(frames, dtype=_np.int)
                        intervals_results[wire_info][frame_count] = wire_hash
    
                already_checked.append(source)  
            
            if allow_direct_bonds:
                for a, b in da_hbonds:
                    wire_info = self._all_ids[a]+':'+self._all_ids[b]
                    check = self._all_ids[b]+':'+self._all_ids[a]
                    if check in results: wire_info=check
                    try:
                        intervals_results[wire_info][frame_count] = -1
                        results[wire_info][frame_count] = 0
                    except:
                        results[wire_info] = _np.ones(frames, dtype=_np.int)*_np.inf
                        results[wire_info][frame_count] = 0
                        intervals_results[wire_info] = _np.arange(frames, dtype=_np.int)
                        intervals_results[wire_info][frame_count] = -1
            
            frame_count += 1
        
        self._set_results({key:results[key]!=_np.inf for key in results})
        self.wire_lengths = results 
        self.hashs = intervals_results
        self.hash_table = this_frame_table


    def compute_average_water_per_wire(self, use_filtered=True):
        if use_filtered: results = self.filtered_results
        else: results = self.initial_results
        return {key:self.wire_lengths[key][self.wire_lengths[key]<_np.inf].mean() for key in results}
    
    def filter_minmal_bonds_path(self, start, goal, use_filtered=True):
        if use_filtered: graph = self.filtered_graph
        else: graph = self.initial_graph 
        if len(graph.nodes()) == 0: raise AssertionError('nothing to filter!')
        if start not in graph.nodes(): raise AssertionError('The start node is not in the graph')
        if goal not in graph.nodes(): raise AssertionError('The goal node is not in the graph')
        for component in _nx.connected_component_subgraphs(graph):
            if start in component.nodes(): break
        if goal in component:
            weighted_component = _nx.Graph()
            avg = self.compute_average_water_per_wire(use_filtered=use_filtered)
            triplets = []
            for u, v in component.edges():
                try:
                    triplets.append((u,v,avg[u+':'+v]))
                except KeyError:
                    triplets.append((u,v,avg[v+':'+u]))
            weighted_component.add_weighted_edges_from(triplets)
            node_set = _nx.bellman_ford_path(weighted_component, start, goal)
            print(len(component), len(weighted_component))
        else: raise AssertionError('start and goal nodes are not connected')
        path_graph = _nx.Graph()
        path_graph.add_edges_from(_hf.pairwise(node_set))
        self.filtered_graph = path_graph
        self._generate_filtered_results_from_filtered_graph()
    
    def compute_all_shortest_paths_length(self, start, goal, use_filtered=True):
        if use_filtered: graph = self.filtered_graph
        else: graph = self.initial_graph 
        if len(graph.nodes()) == 0: raise AssertionError('nothing to filter!')
        if start not in graph.nodes(): raise AssertionError('The start node is not in the graph')
        if goal not in graph.nodes(): raise AssertionError('The goal node is not in the graph')
        try:
            paths = _nx.all_shortest_paths(graph, start, goal)
        except:
            raise AssertionError('start and goal nodes are not connected')
        res = {}
        avg = self.compute_average_water_per_wire(use_filtered=use_filtered)
        for path in paths: 
            key = start+'-'+'-'.join([node.split('-')[2] for node in path[1:-1]])+'-'+goal
            val = _np.sum([avg[a+':'+b] if a+':'+b in avg else avg[b+':'+a] for a,b in _hf.pairwise(path)])
            res[key] = val
        return res
            
    def compute_wire_projection(self):
        frame_count = -1
        projection_results = {}
        for ts in self._universe.trajectory[self._trajectory_slice]:
            frame_count += 1
            for rmsd_key in self.hashs:
                rmsd_hash = self.hashs[rmsd_key][frame_count]
                if rmsd_hash == frame_count or rmsd_hash == -1: continue
                rmsd_wire = self.hash_table[rmsd_hash]
                rmsd_water = rmsd_wire[1:-1]
                rmsd_residues = [rmsd_wire[0], rmsd_wire[-1]]
                nb_water = len(rmsd_water)
                water_coords = self._water.positions
                da_coords = self._da_selection.positions
                all_coordinates = _np.vstack((da_coords, water_coords))
                c_a, c_b, c_c = _np.repeat(all_coordinates[rmsd_residues[0]].reshape((-1,3)), nb_water, axis=0), _np.repeat(all_coordinates[rmsd_residues[1]].reshape((-1,3)), nb_water, axis=0), all_coordinates[rmsd_water].reshape((-1,3))
                angles = _hf.angle(c_a, c_b, c_c)
                dists = _np.sqrt(((c_b - c_c)**2).sum(axis=1))
                proj_sum = (_np.sin(_np.deg2rad(angles)) * dists).mean()
                try:
                    projection_results[rmsd_key][frame_count] = proj_sum
                except:
                    projection_results[rmsd_key] = _np.ones(self.nb_frames)*-1
                    projection_results[rmsd_key][frame_count] = proj_sum
        return projection_results    
    
    def draw_water_timeseries(self, resa, resb, filename=None):
        try:
            timeseries = self.wire_lengths[resa+':'+resb]
        except KeyError:
            timeseries = self.wire_lengths[resb+':'+resa]
        fig, ax = _plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        _plt.scatter(range(len(timeseries)), timeseries, s=0.005)
        _plt.xlabel('frame' , fontsize = 16)
        _plt.ylabel('# water' , fontsize = 16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if filename != None:
            _plt.text.usetex = True
            _plt.savefig(filename, format='eps', dpi=300)
        else:
            _plt.show()
    
    def _save_average_and_occupancy_to_one_file(self, filename):
        avg_water = self.compute_average_water_per_wire(use_filtered=True)
        string = ''
        for key in self.filtered_results:
            string += key + ' ' + str(self.filtered_results[key].mean()) + ' ' + str(avg_water[key]) +'\n'
        with open(filename, 'w') as file:
            file.write(string)
            