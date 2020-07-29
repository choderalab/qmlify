"""
utilities specifically for perses compatibility (e.g. extracting folders from replica exchange checkpoints)
"""

###Logger###
import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def extract_sys_top(local_path, factory_npz = 'outhybrid_factory.npy.npz', phases = ['complex', 'solvent', 'vacuum']):
    """
    given a htf_factory.npz, will extract all phases, serialize systems and pickle topologies

    arguments
        local_path : str
            path that contains factory_npz
        factory_npz : str
            .npz of perses.relative.HybridTopologyFactory
        phases : list of str
            phases to extract
    """
    import os
    import numpy as np
    from qmlify.utils import serialize_xml, write_pickle
    #load the npz
    npz = np.load(os.path.join(local_path, factory_npz), allow_pickle=True)
    systems_dict = npz['arr_0'].item()
    for phase in phases:
        old_system = systems_dict[phase]._old_system
        new_system = systems_dict[phase]._new_system
        old_sys_filename = os.path.join(local_path, f"{phase}.old_system.xml")
        new_sys_filename = os.path.join(local_path, f"{phase}.new_system.xml")
        serialize_xml(old_system, old_sys_filename)
        serialize_xml(new_system, new_sys_filename)

        old_topology = systems_dict[phase]._topology_proposal.old_topology
        new_topology = systems_dict[phase]._topology_proposal.new_topology
        old_top_filename = os.path.join(local_path, f"{phase}.old_topology.pkl")
        new_top_filename = os.path.join(local_path, f"{phase}.new_topology.pkl")
        write_pickle(old_topology, old_top_filename)
        write_pickle(new_topology, new_top_filename)



def extract_perses_repex_to_local(from_dir, to_dir, phases = ['complex', 'solvent']):
    """
    extract perses data from nonlocal directory and copy to local; extract positions, topology, and system for each phase.

    arguments
        from_dir : str
            full path (including `lig{i}to{j}`) from which to extract perses results
        to_dir : str
            full_path (including `lig{i}to{j}`) to which to extract perses results
    """
    import numpy as np
    import os
    import sys
    import mdtraj as md
    from qmlify.executables import extract_sys_top
    from perses.analysis.utils import open_netcdf

    os.mkdir(to_dir)
    factory_npz = os.path.join(from_dir, 'outhybrid_factory.npy.npz')
    os.system(f"cp {factory_npz} {os.path.join(to_dir, 'outhybrid_factory.npy.npz')}")
    extract_sys_top(to_dir, phases = phases + ['vacuum'])
    npz = np.load(factory_npz, allow_pickle=True)
    htf = npz['arr_0'].item()

    #topology proposal
    top_proposal_filename = os.path.join(from_dir, f"out_topology_proposals.pkl")
    TPs = np.load(top_proposal_filename, allow_pickle=True)

    for phase in phases:
        nc_checkpoint_filename = os.path.join(from_dir, f"out-{phase}_checkpoint.nc")
        nc_checkpoint = open_netcdf(nc_checkpoint_filename) #yank the checkpoint interval
        checkpoint_interval = nc_checkpoint.CheckpointInterval
        all_positions = nc_checkpoint.variables['positions'] #pull all of the positions
        bv = nc_checkpoint.variables['box_vectors'] #pull the box vectors
        n_iter, n_replicas, n_atom, _ = np.shape(all_positions)
        nc_out_filename = os.path.join(from_dir, f"out-{phase}.nc")
        nc = open_netcdf(nc_out_filename)
        endstates = [('ligandAlambda0','old',0),('ligandBlambda1','new',n_replicas-1)]
        for endstate in endstates:
            lig, state, replica = endstate
            topology = getattr(TPs[f'{phase}_topology_proposal'], f'{state}_topology')
            molecule = [res for res in topology.residues() if res.name == 'MOL']
            molecule_indices = [a.index for a in molecule[0].atoms()]
            start_id = molecule_indices[0]

            n_atoms = topology.getNumAtoms()
            h_to_state = getattr(htf[f"{phase}"], f'_hybrid_to_{state}_map')
            positions = np.zeros(shape=(n_iter,n_atoms,3))
            lengths, angles = [], []
            bv_frames = []
            for i in range(n_iter):
                replica_id = np.where(nc.variables['states'][i*checkpoint_interval] == replica)[0]
                pos = all_positions[i,replica_id,:,:][0]
                for hybrid, index in h_to_state.items():
                    positions[i,index,:] = pos[hybrid]
                _bv_frame = bv[i, replica_id][0]
                bv_frames.append(_bv_frame)

            bv_frames = np.array(bv_frames)
            np.savez(os.path.join(to_dir, f"{lig}_{phase}.positions.npz"), positions = positions, box_vectors = bv_frames)

def extract_and_subsample_forward_works(i,j,phase,state,annealing_steps, parent_dir, num_resamples):
    """
    after forward annealing, query the output positions and work files;
    in the event that some fail, they will not be written;
    pull the index labels of the work/position files, match them accordingly, and assert that the size of the work array is appropriately defined;
    normalize the works and return a subsampled array of size num_resamples

    arguments
        i : int
            lig{i}to{j}
        j : int
            lig{i}to{j}
        phase : str
            the phase
        state : str
            'old' or 'new'
        annealing_steps : int
            number of annealing steps to extract
        parent_dir : str
            full path of the parent dir of lig{i}to{j}
        num_resamples : int
            number of resamples to pull

    returns
        resamples : np.array(num_resamples)
            resampled indices

    """
    import glob
    import os
    import numpy as np
    from qmlify.utils import exp_distribution

    #define a posiiton and work template
    positions_template = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.forward.*.{annealing_steps}_steps.positions.npz")
    works_template = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.forward.*.{annealing_steps}_steps.works.npz")

    #query the positions/work template
    positions_filenames = glob.glob(positions_template)
    position_index_extractions = {int(i.split('.')[4][4:]): os.path.join(parent_dir, i) for i in positions_filenames} #make a dict of indices
    works_filenames = glob.glob(works_template)
    corresponding_work_filenames = {int(i.split('.')[4][4:]): os.path.join(parent_dir, i) for i in works_filenames}

    #iterate through posiiton indices; if there is a work file and it has the approproate number of annealing steps, append it
    full_dict = {}; works = {}
    for index in position_index_extractions.keys():
        if index in list(corresponding_work_filenames.keys()):
            work_array = np.load(corresponding_work_filenames[index])['works']
            if len(work_array) == annealing_steps + 1:
                full_dict[index] = (position_index_extractions[index], corresponding_work_filenames[index])
                works[index] = work_array[-1]

    #normalize
    work_indices, work_values = list(works.keys()), np.array(list(works.values()))
    normalized_work_values = exp_distribution(work_values)

    assert all(len(item)>0 for item in [work_indices, work_values, normalized_work_values])

    resamples = np.random.choice(work_indices, num_resamples, p = normalized_work_values)
    return resamples

def backward_extractor(i,j,phase, state, annealing_steps, parent_dir):
    """
    pull the indices of all existing position files
    """
    import os
    import glob
    positions_template = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.ani_endstate.*.{annealing_steps}_steps.positions.npz")
    positions_filenames = glob.glob(positions_template)
    position_index_extractions = {int(filename.split('.')[4][4:]): os.path.join(parent_dir, filename) for filename in positions_filenames}
    return list(position_index_extractions.keys())


####################
###ADMINISTRATORS###
####################
def perses_extraction_admin(ligand_index_pairs,
                            from_dir_parent,
                            to_dir_parent,
                            sh_template = None,
                            phases = ['complex', 'solvent'],
                            write_log = True,
                            submission_call = 'bsub <',
                            cat_execution = True,
                            delete_execution = True):
    """
    wraps the perses extraction admin around a list of ligand pairs and submits via bsub

    arguments
        ligand_index_pairs : list of tup
            list of integer tuples (lig{i}to{j})
        from_dir_parent : str
            parent directory from which files will be extracted
        to_dir_parent : str
            parent directory to which files will be extracted
        sh_template : str
            .sh file that serves as a template (to which an execution can be written)
    """
    import os
    from pkg_resources import resource_filename
    from qmlify.executables import extract_perses_repex_to_local #local
    from qmlify.utils import write_bsub_delete
    _logger.info(f"administrating extraction of systems, positions, and topologies for {len(ligand_index_pairs)} ligand pairs...")

    if sh_template is None:
        _logger.info(f"there is no .sh template specified; using default template")
        from pkg_resources import resource_filename
        sh_template = resource_filename('qmlify', 'data/templates/cpu_daemon.sh')

    _logger.info(f"iterating through ligand pairs...")
    for i,j in ligand_index_pairs:
        _logger.info(f"lig{i}to{j}:")
        from_dir = os.path.join(from_dir_parent, f"lig{i}to{j}")
        to_dir = os.path.join(to_dir_parent, f"lig{i}to{j}")
        line_to_write = f"python -c \"from qmlify.executables import extract_perses_repex_to_local; extract_perses_repex_to_local(\'{from_dir}\', \'{to_dir}\', {phases}) \" "
        write_bsub_delete([line_to_write], sh_template, f"lig{i}to{j}_perses_extraction", write_to_dir = to_dir_parent, write_log = write_log, submission_call = submission_call, cat=cat_execution, delete = delete_execution)

def propagation_admin(ligand_index_pairs,
                              annealing_steps,
                              direction,
                              parent_dir,
                              extraction_indices = range(0,200,2),
                              backers = ['old', 'new'],
                              phases = ['complex', 'solvent'],
                              nondefault_integrator_kwarg_dict=None,
                              eq_steps = None,
                              write_log=False,
                              sh_template=None,
                              cat_outputs = False,
                              delete_outputs=True):
    """
    performs ensemble annealing in the forward direction

    arguments
        ligand_index_pairs : list of tup of str
            list of ligand tuple indices (e.g. [(0,1), (3,4)])
        annealing_steps : int
            number of annealing steps
        direction : str
            'forward', 'backward', or 'ani_endstate'
        parent_dir : str
            parent directory containing 'lig{i}to{j}'
        extraction_indices : list of int or int
            list of integers that will extract; this is only used in the 'forward' direction; otherwise, it is an int of the number of samples
        backers : list of str
            list of ['old', 'new']
        phases : list of str
            phases to run simulation
        nondefault_integrator_kwarg_dict : dict, default None
            integration kwargs
        eq_steps : int, default None
            only used for 'backward' direction to specify which set of equilibrium snapshots at the ani_endstate will be annealed backward;
            if None, will use annealing steps
        write_log : bool, default False
            whether to write the logger for the submission
        sh_template : str, default None
            path to a submission template; if None, qmlify.data.templates.cpu_daemon.sh will be used
        cat_outputs : bool, default False
            whether to cat the bsub .sh files
        delete_outputs : bool, default True
            whether to delete the bsub .sh files

    """
    import os
    import numpy as np
    import glob
    from qmlify.qmlify_data import yaml_keys #template
    from qmlify.utils import write_bsub_delete
    from pkg_resources import resource_filename

    _logger.info(f"administrating ensemble {direction} job executions...")
    _logger.debug(f"the extraction indices are: {extraction_indices}")

    if sh_template is None:
        _logger.info(f"there is no .sh template specified; using default template")
        sh_template = resource_filename('qmlify', 'data/templates/cpu_daemon.sh')

    yaml_dict = {key: None for key in yaml_keys}

    yaml_dict['direction'] = direction

    if direction =='ani_endstate':
        assert type(extraction_indices) == int, f"extraction indices must be an int; it is of type {type(extraction_indices)}"
        num_extractions = extraction_indices
        from qmlify.executables import extract_and_subsample_forward_works
        if eq_steps is None: eq_steps = annealing_steps
        yaml_dict['num_steps'] = eq_steps
    else:
        yaml_dict['num_steps'] = annealing_steps

    if nondefault_integrator_kwarg_dict is not None:
        _logger.debug(f"modifying integrator_kwargs for yaml qmlify submission...")
        yaml_dict['integrator_kwargs'] = nondefault_integration_kwarg_dict
    else:
        _logger.debug(f"there are no modifying integrator_kwargs for qmlify submission...")

    if direction == 'backward':
        from qmlify.executables import backward_extractor
        if eq_steps is None: eq_steps = annealing_steps
        _logger.debug(f"direction is backward; 'eq_steps' is not defined; extracting {annealing_steps} (annealing steps) as default")

    for i,j in ligand_index_pairs:
        _logger.debug(f"lig{i}to{j}: ")
        for phase in phases:
            _logger.debug(f"phase: {phase}")
            for state in backers:
                _logger.debug(f"{state}: ")
                system_filename = os.path.join(parent_dir, f"lig{i}to{j}", f"{phase}.{state}_system.xml")
                subset_system_filename = os.path.join(parent_dir, f"lig{i}to{j}", f"vacuum.{state}_system.xml")

                topology_filename = os.path.join(parent_dir, f"lig{i}to{j}", f"{phase}.{state}_topology.pkl")
                subset_topology_filename = os.path.join(parent_dir, f"lig{i}to{j}", f"vacuum.{state}_topology.pkl")

                yaml_dict['system'] = system_filename
                yaml_dict['subset_system'] = subset_system_filename
                yaml_dict['topology'] = topology_filename
                yaml_dict['subset_topology'] = subset_topology_filename

                if direction == 'forward':
                    if state == 'old':
                        posit_filename = os.path.join(parent_dir, f"lig{i}to{j}", f"ligandAlambda0_{phase}.positions.npz")
                    else:
                        posit_filename =os.path.join(parent_dir, f"lig{i}to{j}", f"ligandBlambda1_{phase}.positions.npz")
                    yaml_dict['positions_cache_filename'] = posit_filename

                if direction == 'ani_endstate':
                    _logger.debug(f"querying forward works and positions to resample with {num_extractions} resamples...")
                    #we need to pull works and subsample
                    extraction_indices = extract_and_subsample_forward_works(i,j,phase,state,annealing_steps, parent_dir, num_extractions) 
                    _logger.debug(f"indices extracted: {extraction_indices}")
                    #extraction_indices = range(len(extraction_indices))
                    resample_file = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.{annealing_steps}_steps.forward_resamples.npz")
                    assert not os.path.exists(resample_file), f"{resample_file} already exists; aborting"
                    np.savez(resample_file, extraction_indices)
                elif direction == 'backward':
                    _logger.debug(f"extracting positions for backward annealing...")
                    extraction_indices = backward_extractor(i,j,phase, state, eq_steps, parent_dir)

                for idx in range(len(extraction_indices)):
                    traj_work_file_prefix = f"lig{i}to{j}.{phase}.{state}.{direction}.idx_{idx}.{annealing_steps}_steps"

                    #extraction_index
                    extraction_index = extraction_indices[idx] if direction=='forward' else 0
                    yaml_dict['position_extraction_index'] = extraction_index

                    if direction == 'ani_endstate':
                        posit_filename = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.forward.idx_{extraction_indices[idx]}.{annealing_steps}_steps.positions.npz")
                        yaml_dict['positions_cache_filename'] = posit_filename
                    elif direction == 'backward':
                        posit_filename = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.ani_endstate.idx_{extraction_indices[idx]}.{eq_steps}_steps.positions.npz")
                        yaml_dict['positions_cache_filename'] = posit_filename
                    else:
                        #already chosen above
                        pass

                    yaml_dict['out_positions_npz'] = os.path.join(parent_dir, traj_work_file_prefix + f".positions.npz")
                    yaml_dict['out_works_npz'] = os.path.join(parent_dir, traj_work_file_prefix + f".works.npz")

                    line_to_write = f"python -c \"from qmlify.executor import run; run({yaml_dict})\" "
                    write_bsub_delete(lines_to_write = [line_to_write],
                                      template = sh_template,
                                      template_suffix = traj_work_file_prefix,
                                      write_to_dir = parent_dir,
                                      write_log=write_log,
                                      submission_call = 'bsub <',
                                      cat=cat_outputs,
                                      delete=delete_outputs)
