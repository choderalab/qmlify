"""
analysis utilities for post-processing
"""
#####Imports#####
from simtk import unit
import numpy as np
import os

DEFAULT_WORK_TEMPLATE = 'lig{i}to{j}.{phase}.{state}.{direction}.idx_{idx}.{annealing_steps}_steps.works.npz'
DEFAULT_WORK_QUERY_TEMPLATE = '.'.join(DEFAULT_WORK_TEMPLATE.split('.')[:4]) + '.*.' + '.'join(DEFAULT_WORK_TEMPLATE.split('.')[5:])



def aggregate_per_pair_works(ligand_indices, annealing_steps, directions = ['forward', 'backward'], phases = ['complex', 'solvent'], states = ['old', 'new'], parent_dir = os.getcwd()):
    """
    generate a dictionary of the final aggregated works of the trajectories with the specifications;
    """
    from qmlify.analysis import work_file_extractor
    import tqdm

    pair_works_dict = {
                        (i,j): {state: {phase: {direction: {} for direction in directions
                                                    }
                                           for phase in phases }
                                   for state in states }
                        for i,j in ligand_indices}

    for i,j in tqdm.tqdm(ligand_indices):
        for state in states:
            for phase in phases:
                for direction in directions:
                    file_dict = work_file_extractor(i, j, phase, state, annealing_steps, parent_dir) 
                    for idx, filename in file_dict.items():
                        try:
                            works = np.load(filename)['works']
                            assert len(works) == annealing_steps + 1
                            pair_works_dict[(i,j)][state][phase][direction][idx] = works[-1]
                        except Exception as e:
                            print(f"aggregate_per_pair_works query error: {e}")

    return pair_works_dict  


def generate_duplication_histograms(pair_works_dict, write_dir, fig_width=8.5, fig_height=7.5):
    """
    generate a seaborn histogram 
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    unique_ligands = list(set([i for sub in list(pair_work_dict.keys()) for i in sub]))
    
    
    #fig = plt.figure(figsize=(fig_width, fig_height))
    
    
    

def compute_free_energy_corrections(ligand_indices, annealing_steps, compute_EXP = False, generate_plots=True, write_dir = os.getcwd()):
    """
    generate a dictionary of BAR and EXP free energy corrections.
    """
    #run the aggregate_per_pair_works to get a dict of forward/backward works per pair, phase, etc
    from qmlify.analysis import aggregate_per_pair_works
    pair_works_dict = aggregate_per_pair_works(ligand_indices, annealing_steps, states, parent_dir = parent_dir)
    
    #save a matplotlib pdf plot of these works
    if generate_plots
        

    #aggregate the works into single dict for each ligand for each phase
    
    #save a matplotlib pdf of these works
    
    #write a dict and return it

def work_file_extractor(i, j, phase, state, annealing_steps, parent_dir):
    """
    pull the indices of all existing work files
    """
    import glob
    work_query_template = '.'.join(DEFAULT_WORK_TEMPLATE.split('.')[:4]) + '.*.' + '.'join(DEFAULT_WORK_TEMPLATE.split('.')[5:])
    work_filenames = glob.glob(work_query_template)
    index_extractions = {int(filename.split('.')[4][4:]): os.path.join(parent_dir, filename) for filename in work_filenames}
    return index_extractions


