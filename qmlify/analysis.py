"""
analysis utilities for post-processing
"""
#####Imports#####
from simtk import unit
import numpy as np
import os

DEFAULT_WORK_TEMPLATE = 'lig{i}to{j}.{phase}.{state}.{direction}.idx_{idx}.{annealing_steps}_steps.works.npz'
DEFAULT_POSITION_TEMPLATE = 'lig{i}to{j}.{phase}.{state}.{direction}.idx_{idx}.{annealing_steps}_steps.positions.npz'



def aggregate_per_pair_works(ligand_indices, annealing_steps, directions = ['forward', 'backward'], states = ['old', 'new'], parent_dir = os.getcwd()):
    """
    generate a dictionary of the final aggregated works of the trajectories with the specifications;

    arguments
        ligand_indices : list of tup
            list of tupled ligand indices (e.g. [(a,b), (a,c), ...])
        annealing_steps : dict
            dictionary of phase by annealing step;
            (e.g. ['complex': 500, 'solvent': 5000])
        directions : list of str, default ['forward', 'backward']
            list of directions
        states : list of str, default ['old', 'new']
            list of states corresponding to ligand indices
        parent_dir : str, default os.getcwd() (i.e. current directory)
            path to directory that holds the work .npz files

    returns
        pair_works_dict : dict
            nested dictionary of keys: [ligand pair (tup)] : [state ('old'/'new')] : [phase ('solvent'/'complex')] : [direction ('forward'/'backward')] : [file_index : total work (kT)]
    """
    from qmlify.analysis import work_file_extractor
    import tqdm

    pair_works_dict = {
                        (i,j): {state: {phase: {direction: {} for direction in directions
                                                    }
                                           for phase in list(annealing_steps.keys()) }
                                   for state in states }
                        for i,j in ligand_indices}

    for i,j in tqdm.tqdm(ligand_indices):
        for state in states:
            for phase in list(annealing_steps.keys()):
                for direction in directions:
                    file_dict = work_file_extractor(i, j, phase, state, direction, annealing_steps[phase], parent_dir)
                    for idx, filename in file_dict.items():
                        try:
                            works = np.load(filename)['works']
                            assert len(works) == annealing_steps[phase] + 1
                            pair_works_dict[(i,j)][state][phase][direction][idx] = works[-1]
                        except Exception as e:
                            print(f"aggregate_per_pair_works query error: {e}")

    return pair_works_dict

def fully_aggregate_work_dict(work_dict):
    """
    take a pair work dict (see aggregate_per_pair_works) and aggregate them so that the dictionary has the form
        [ligand_index (int)] : [phase ('solvent'/'complex')] : [direction ('forward'/'backward')] : np.ndarray(num_repeats, num_work_entries).
    will also generate the above dict with the final nd.ndarray aggregated to a np.ndarray(1,num_repeats*num_work_entries)

    arguments:
        work_dict : dict
            output of aggregate_per_pair_works

    returns
        agg_dict : dict
            first entry in description
        concat_dict : dict
            second entry in description
    """
    unique_ligands = sorted(list(set([i for sub in list(work_dict.keys()) for i in sub])))
    agg_dict = {ligand: {phase: {direction: [] for direction in ['forward', 'backward']} for phase in ['complex', 'solvent']} for ligand in unique_ligands}
    concat_dict = {ligand: {phase: {direction: [] for direction in ['forward', 'backward']} for phase in ['complex', 'solvent']} for ligand in unique_ligands}
    #ligand:phase:direction...

    for ligand in unique_ligands:
        for i,j in work_dict.keys():
            if i==ligand:
                state='old'
            elif j==ligand:
                state='new'
            else:
                continue
            for phase in ['complex', 'solvent']:
                for direction in ['forward', 'backward']:
                    try:
                        agg_dict[ligand][phase][direction].append(list(work_dict[(i,j)][state][phase][direction].values()))
                    except Exception as e:
                        print(f"aggregation error: {e}")

    #just make them numpy arrays
    for ligand in unique_ligands:
        for phase in ['complex', 'solvent']:
            for direction in ['forward', 'backward']:
                try:
                    agg_dict[ligand][phase][direction] = np.array(agg_dict[ligand][phase][direction])
                    concat_dict[ligand][phase][direction] = np.array([np.concatenate(agg_dict[ligand][phase][direction])])
                except:
                    pass

    return agg_dict, concat_dict

def compute_BAR(work_dict):
    """
    compute BAR free energies and uncertainties

    arguments:
        work_dict : dict
            dict of form agg_dict or concat_dict (see fully_aggregate_work_dict)

    returns
        BAR_results : dict
            dict of [ligand][phase] = list((dg, ddg)) where each entry in the list comes from an aggregated work entry
    """
    import numpy as np
    from pymbar.bar import BAR
    BAR_results = {ligand: {phase: None for phase in ['complex', 'solvent']} for ligand in work_dict.keys()}

    for ligand in work_dict.keys():
        for phase in work_dict[ligand].keys():
            forward_works = work_dict[ligand][phase]['forward']
            backward_works = work_dict[ligand][phase]['backward']
            forward_work_shape = forward_works.shape
            backward_work_shape = backward_works.shape
            if len(forward_work_shape) > 1: #these works are not aggregated in total
                outs = []
                assert forward_work_shape[0] == backward_work_shape[0], f"the number of forward work arrays is not equal to the number of backward work arrays for {ligand}: {phase}"
                for fwd_array, bkwd_array in zip(forward_works, backward_works):
                    dg_tup = BAR(fwd_array, bkwd_array)
                    outs.append(dg_tup)
                BAR_results[ligand][phase] = outs
            else: #these works _are_ aggregated in total
                dg_tup = BAR(forward_works, backward_works)
                BAR_results[ligand][phase] = [dg_tup]
    return BAR_results

def work_file_extractor(i, j, phase, state, direction, annealing_steps, parent_dir):
    """
    pull the indices of all existing work files
    """
    import glob
    import os
    work_query_template = os.path.join(parent_dir, '.'.join(DEFAULT_WORK_TEMPLATE.split('.')[:4]) + '.*.' + '.'.join(DEFAULT_WORK_TEMPLATE.split('.')[5:]))
    work_query_filename = work_query_template.format(i=i, j=j, phase=phase, state=state, direction=direction, annealing_steps=annealing_steps)
    work_filenames = glob.glob(work_query_filename)
    index_extractions = {int(filename.split('.')[4][4:]): os.path.join(parent_dir, filename) for filename in work_filenames}
    return index_extractions

def write_positions_as_pdbs(i, j, phase, state, annealing_steps, parent_dir, topology_pkl, direction='forward', output_pdb_filename=None):
    """
    extract the positions files for an array of annealing steps and write the ligand positions to a pdb;
    this is primarily used to extract and view post-annealing snapshots for sanity checks (i.e. to make sure molecules aren't exploding)
    
    arguments
        i : int
            start ligand
        j : int
            end ligand
        phase : str
            phase
        state : str
            old/new
        direction : str, default 'forward'
            direction
        annealing_steps : int
            number of annealing steps
        parent_dir : str
            parent dir where 'positions.npz's live
        topology_pkl : str
            name of pickled openmm topology file
        output_pdb_filename : str, default None
            output pdb

    will output a pdb of the form: 
    
    example:
        >>> import os
        >>> from qmlify.analysis import write_positions_to_pdbs
        >>> #let's query lig0to4, old, solvent (the more difficult transform), forward, at 500, 1000, 5000, 10000 annealing steps from cwd
        >>> annealing_list=[500, 1000, 5000, 10000]
        >>> for step in annealing_list: write_positions_to_pdbs(0,4,'solvent', 'old', step, os.getcwd(), 'lig0to4/solvent.old_topology.pkl')
            
    """
    from qmlify.analysis import work_file_extractor
    import pickle
    import mdtraj
    import glob
    import os
    import numpy as np
    import tqdm

    query_template = os.path.join(parent_dir, '.'.join(DEFAULT_POSITION_TEMPLATE.split('.')[:4]) + '.*.' + '.'.join(DEFAULT_POSITION_TEMPLATE.split('.')[5:])) 
    query_filename = query_template.format(i=i, j=j, phase=phase, state=state, direction=direction, annealing_steps=annealing_steps)
    filenames_list = glob.glob(query_filename)
    index_extractions = {int(filename.split('.')[4][4:]): os.path.join(parent_dir, filename) for filename in filenames_list}

    work_files = work_file_extractor(i, j, phase, state, direction, annealing_steps, parent_dir)
    
    from openeye import oechem
    with open(topology_pkl, 'rb') as f:
        topology = pickle.load(f)
    
    md_topology = mdtraj.Topology.from_openmm(topology)
    subset_indices = md_topology.select('resname MOL')
    
    positions = []
    snapshots = []
    counter=0
    for snapshot_index, filename in tqdm.tqdm(sorted(index_extractions.items())):
        try:
            frame = np.load(filename)['positions'][0]
            work_value = np.load(work_files[snapshot_index])['works'][-1]
            snapshots.append([counter, work_value])
            positions.append(frame[subset_indices,:])
            counter+=1
        except Exception as e:
            print(e)

    traj = mdtraj.Trajectory(xyz=np.array(positions), topology = md_topology.subset(subset_indices))

    if output_pdb_filename is None:
        output_pdb_filename = f"lig{i}to{j}.{state}.{direction}.{annealing_steps}_steps.aggregate.pdb"
    else:
        assert output_pdb_filename[-3:] == 'pdb'
    output_array_filename = output_pdb_filename[:-3] + 'npz'

    traj.save(os.path.join(parent_dir, output_pdb_filename))
    np.savez(os.path.join(parent_dir, output_array_filename), np.array(snapshots))


def extract_work_calibrations(ligand_tuple, annealing_steps, phase, direction, state, parent_dir):
    """
    extract work calibration steps
            
    arguments
        ligand_indices : list of tup
            list of tupled ligand indices (e.g. [(a,b), (a,c), ...])
        annealing_steps : list
            list of annealing steps to extract
        phases : list
            list of phases to extract
        directions : list of str, default ['forward', 'backward']
            list of directions
        states : list of str, default ['old', 'new']
            list of states corresponding to ligand indices
        parent_dir : str, default os.getcwd() (i.e. current directory)
            path to directory that holds the work .npz file
        
        nested dictionary of keys: [ligand pair (tup)] : [state ('old'/'new')] : [phase ('solvent'/'complex')] : [direction ('forward'/'backward')] : [file_index : total work (kT)]

    """
    logger_dict = {}
    for annealing_step in annealing_steps:
        step_dict = {phase: annealing_step}
        aggregation_dict = aggregate_per_pair_works([ligand_tuple], 
                                                    step_dict, 
                                                    directions = [direction], 
                                                    states = [state], 
                                                    parent_dir = parent_dir)
        logger_dict[annealing_step] = np.array(list(aggregation_dict[ligand_tuple][state][phase][direction].values()))
    
    return logger_dict
