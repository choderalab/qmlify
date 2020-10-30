"""
analysis tools for checking MM -> ML/MM absolute and relative free energies
"""

#####Imports#####
from simtk import unit
import numpy as np
import os
from openmmtools.constants import kB
import copy
DEFAULT_TEMPERATURE = 300.0 * unit.kelvin


DEFAULT_WORK_TEMPLATE = 'lig{i}to{j}.{phase}.{state}.{direction}.idx_{idx}.{annealing_steps}_steps.works.npz'
DEFAULT_POSITION_TEMPLATE = 'lig{i}to{j}.{phase}.{state}.{direction}.idx_{idx}.{annealing_steps}_steps.positions.npz'

DEFAULT_MM_POSITION_TEMPLATE = 'lig{i}to{j}/ligand{letter}lambda{_lambda}_{phase}.positions.npz'


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
            #determine if these are aggregated...
            forward_work_shape = forward_works.shape
            backward_work_shape = backward_works.shape
            first_entry_is_list = True if (type(forward_works[0]) == list or type(backward_works[0]) == list) else False

            if len(forward_work_shape) > 1 or first_entry_is_list: #these works are not aggregated in total
                outs = []
                assert forward_work_shape[0] == backward_work_shape[0], f"the number of forward work arrays is not equal to the number of backward work arrays for {ligand}: {phase}"
                for fwd_array, bkwd_array in zip(forward_works, backward_works):
                    dg_tup = BAR(np.array(fwd_array), np.array(bkwd_array))
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

def write_positions_as_pdbs(i, j, phase, state, annealing_steps, parent_dir, topology_pkl, direction='forward', output_pdb_filename=None, selection_string='resname MOL'):
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

    from openeye import oechem
    with open(topology_pkl, 'rb') as f:
        topology = pickle.load(f)
    md_topology = mdtraj.Topology.from_openmm(topology)
    subset_indices = md_topology.select(selection_string)

    if direction != 'mm_endstate':
        query_template = os.path.join(parent_dir, '.'.join(DEFAULT_POSITION_TEMPLATE.split('.')[:4]) + '.*.' + '.'.join(DEFAULT_POSITION_TEMPLATE.split('.')[5:]))
        query_filename = query_template.format(i=i, j=j, phase=phase, state=state, direction=direction, annealing_steps=annealing_steps)
        filenames_list = glob.glob(query_filename)
        index_extractions = {int(filename.split('.')[4][4:]): os.path.join(parent_dir, filename) for filename in filenames_list}

        work_files = work_file_extractor(i, j, phase, state, direction, annealing_steps, parent_dir)

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
        positions = np.array(positions)
    elif direction=='mm_endstate':
        #we are just going to pull the _before_ annealing trajectories
        letter, _lambda = ('A', 0) if state=='old' else ('B', 1)
        query_template = os.path.join(parent_dir, DEFAULT_MM_POSITION_TEMPLATE.format(i=i, j=j, _lambda=_lambda, letter=letter, phase=phase))
        positions = np.load(query_template)['positions']
    else:
        raise Exception(f"{direction} is not a supported direction")

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

    returns
        logger_dict : dict
            dict of annealing_steps : work_distribution

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

def bootstrap(original_data, statistic, num_resamples):
    """
    bootstrap some data
    """
    import numpy as np
    out_data = []
    for iteration in range(num_resamples):
        data = np.random.choice(original_data, len(original_data))
        stat = statistic(data)
        out_data.append(stat)
    return np.array(out_data)


def compute_CIs(data, alpha=0.95):
    """
    compute confidence interval for a bootstrapped set of data...
    """
    n = len(data)
    data = sorted(data)
    lower_bound, upper_bound = int(n*(1-alpha)/2), int(n*(1+alpha)/2)
    return data[lower_bound], data[upper_bound]

def _absolute_from_relative(g, experimental, experimental_error):
    """
    Use DiffNet to compute absolute free energies, from computed relative free energies and assign absolute experimental values and uncertainties.

    Parameters
    ----------
    g : nx.DiGraph()
        A graph with relative results assigned to edges
    experimental : list
        list of experimental absolute affinities in kcal/mol
    experimental_error : list
        list of experimental absolute uncertainties in kcal/mol

    Returns
    -------

    """
    from arsenic import stats
    f_i_calc, C_calc = stats.mle(g, factor='calc_DDG') #compute maximum likelihood estimates of free energies from calculated DDG attribute
    variance = np.diagonal(C_calc) #variance of estimate is diagonal of the calculation matrix
    for n, f_i, df_i in zip(g.nodes(data=True), f_i_calc, variance**0.5):
        n[1]['calc_DG'] = f_i
        n[1]['calc_dDG'] = df_i
        n[1]['exp_DG'] = experimental[n[0]]
        n[1]['exp_dDG'] = experimental_error[n[0]]

def _make_mm_graph(mm_results, expt, d_expt):
    """ Make a networkx graph from MM results

    Parameters
    ----------
    mm_results : list(perses.analysis.load_simulations.Simulation)
        List of perses simulation objects
    expt : list
        List of experimental values, in kcal/mol
    d_expt : list
        List of uncertainties in experimental values, in kcal/mol

    Returns
    -------
    nx.DiGraph
        Graph object with relative MM free energies

    """
    from qmlify.analysis import _absolute_from_relative
    import networkx as nx
    mm_g = nx.DiGraph()
    for sim in mm_results:
        ligA = int(sim.directory[3:].split('to')[0]) #define edges
        ligB = int(sim.directory[3:].split('to')[1])
        exp_dDDG = (d_expt[ligA]**2 + d_expt[ligB]**2)**0.5 #define exp error
        mm_g.add_edge(ligA,
                      ligB,
                      calc_DDG=-sim.bindingdg/sim.bindingdg.unit, #simulation reports A - B
                      calc_dDDG=sim.bindingddg/sim.bindingddg.unit,
                      exp_DDG=(expt[ligB] - expt[ligA]),
                      exp_dDDG=exp_dDDG) #add edge to the digraph
    _absolute_from_relative(mm_g, expt, d_expt)
    return mm_g

def _make_ml_graph(mm_g, corrections):
    """
    Create a ML/MM graph, using an MM graph, and corrections to that.

    Parameters
    ----------
    mm_g : nx.DiGraph()
        Graph object with MM results
    corrections : dict(int:(float,float))
        Stored per-ligand corrections in the format ligand_ID:(correction, uncertainty)

    Returns
    -------
    nx.DiGraph
        Graph object with relative MM free energies

    """
    ml_g = copy.deepcopy(mm_g)
    for node in ml_g.nodes(data=True):
        node[1]['calc_DG'] = node[1]['calc_DG'] + corrections[node[0]][0]
        node[1]['calc_dDG'] = (node[1]['calc_dDG']**2 + corrections[node[0]][1]**2)**0.5
    return ml_g

def _per_ligand_correction(ml_corrections, kT):
    """Take the output corrections for both complex and solvent phases and turn into a single, per-ligand correction

    Parameters
    ----------
    ml_corrections : dict(int:dict)
        ML/MM corrections from qmlify
    kT : unit.Quantity(float) compatible with unit.kilocalories_per_mole
        thermal energy

    Returns
    -------
    dict
        dictionary of format ligand_ID:(correction, uncertainty)

    """
    binding_corrections = {}
    for ligand in ml_corrections.keys():
        binding_corr = ((ml_corrections[ligand]['complex'][0] - ml_corrections[ligand]['solvent'][0])*kT).value_in_unit(unit.kilocalorie_per_mole)
        binding_corr_err = ((ml_corrections[ligand]['complex'][1]**2 + ml_corrections[ligand]['solvent'][1]**2)**0.5*kT).value_in_unit(unit.kilocalorie_per_mole)
        binding_corrections[ligand] = (binding_corr, binding_corr_err)
    return binding_corrections

def _relative_corrections(corrections):
    """Shift the MM->ML/MM corrections to have a mean of zero.

    Parameters
    ----------
    corrections : dict
        dictionary of corrections

    Returns
    -------
    dict
        dictionary of corrections, with mean of zero

    """
    shift = np.min(list(corrections.values()))
    for lig in corrections.keys():
        corrections[lig] = corrections[lig] - shift
    return corrections

def _plot_absolute(g, experimental, name='MM', MM_ff='', ML_ff='', color='k'):
    """ Plots calculated vs experimental results for absolute free energies in kcal/mol
    Saves a file to NAME_absolute.pdb

    Parameters
    ----------
    g : nx.DiGraph()
        Graph with absolute values assigned to nodes
    experimental : np.array()
        experimental absolute free energies in kcal/mol
    name : str, default='MM'
        name of method, used to label plot and filename
    MM_ff : str, default=''
        Name of MM method for labelling
    ML_ff : str, default=''
        Name of ML method for labelling
    color : str, default='k'
        Anything matplotlib.pyplot can understand as a color

    Returns
    -------

    """
    from arsenic import plotting
    filename = f"{name.replace('/','-')}_absolute.pdf"
    plotting.plot_DGs(g, title=f'{name}: {MM_ff}\n {ML_ff}',
                      centralizing=True, color=color,
                      shift=np.mean(experimental),
                      filename=filename)

def _plot_relative(g, name='MM', MM_ff='', ML_ff='', color='k'):
    """ Plots calculated vs experimental results for relative free energies in kcal/mol
    Saves a file to NAME_relative.pdb

    Parameters
    ----------
    g : nx.DiGraph()
        Graph with relative values assigned to nodes
    name : str, default='MM'
        name of method, used to label plot and filename
    MM_ff : str, default=''
        Name of MM method for labelling
    ML_ff : str, default=''
        Name of ML method for labelling
    color : str, default='k'
        Anything matplotlib.pyplot can understand as a color

    Returns
    -------

    """
    from arsenic import plotting
    filename = f"{name.replace('/','-')}_relative.pdf"
    for edge in g.edges(data=True):
        edge[2]['calc_DDG'] = g.nodes[edge[1]]['calc_DG'] - g.nodes[edge[0]]['calc_DG']
#        edge[2]['exp_DDG'] = g.nodes[edge[1]]['exp_DG'] - g.nodes[edge[0]]['exp_DG']
        edge[2]['exp_dDDG'] = (g.nodes[edge[1]]['exp_dDG']**2+g.nodes[edge[0]]['exp_dDG']**2)**0.5
    plotting.plot_DDGs(g, title=f'{name}: {MM_ff}\n {ML_ff}', color=color, filename=filename)



def analyze_mlmm(mm_results,
                 ml_corrections,
                 experimental=None,
                 experimental_error=0.3,
                 MM_ff='openff-1.0.0',
                 ML_ff='ANI-2x',
                 temperature=DEFAULT_TEMPERATURE):
    """
    Runs  MM-> ML/MM analysis, and generates four graphs:
    - MM relative
    - MM absolute
    - ML/MM relative
    - ML/MM absolute

    >>>  analysis.analyze_mlmm(all_sims, ani_results, experimental=experimental, experimental_error=0.18)

    Parameters
    ----------
    mm_results : list(perses.analysis.load_simulations.Simulation)
        List of perses Simulation objects of MM calculations
    ml_corrections : dict(int:dict(phase:tuple))
        MM -> ML/MM corrections and associated uncertainty
        i.e. {0:'solvent': (0.2, 0.1)} if ligand index 0 has a correction of 0.2 (0.1) kT in solvent phase
    experimental : list(float), default=None
        list of experimental binding free energies in kcal/mol
        if None function will return
    experimental_error: float or list of floats, default=0.3
        experimental uncertainty, either one value for the whole ligand set,
        or a list per ligand in kcal/mol
        Default value of 0.3 from
    MM_ff : str, default='openff-1.0.0'
        string name of MM ff for labelling plots
    ML_ff : str, default='ANI-2x'
        string name of ML ff for labelling plots

    Returns
    -------

    """
    from qmlify.analysis import _make_ml_graph, _make_mm_graph, _plot_absolute, _plot_relative, _per_ligand_correction, _relative_corrections
    import networkx as nx
    kT = temperature*kB

    #broadcast the experimental free energy to _all_ experimental results
    if isinstance(experimental_error, float):
        experimental_error = len(experimental) * [experimental_error]

    assert len(experimental) == len(experimental_error), \
        f"Need either one experimental uncertainty for ligand set, or an experimental_error per ligand. {len(experimental)} experimental DGs, with {len(experimental_error)} dDGs."

    mm_g = _make_mm_graph(mm_results, experimental, experimental_error) #compute a digraph of MM free energies
    binding_corrections = _per_ligand_correction(ml_corrections, kT) #query and compute the binding free energy corrections
    shifted_corrections = _relative_corrections(binding_corrections) #shift the binding free energy corrections off from the minimal correction (not strictly necessary)
    ml_g = _make_ml_graph(mm_g, shifted_corrections)

    _plot_absolute(mm_g, experimental, name='MM', MM_ff=MM_ff, color='#6C8EBF')
    _plot_absolute(ml_g, experimental, name='ML/MM', MM_ff=MM_ff, ML_ff=ML_ff, color='#D79B00')
    _plot_relative(mm_g, name='MM', MM_ff=MM_ff, color='#6C8EBF')
    _plot_relative(ml_g, name='ML/MM', MM_ff=MM_ff, ML_ff=ML_ff, color='#D79B00')
