"""
utilities specifically for perses compatibility (e.g. extracting folders from replica exchange checkpoints)
"""
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
    import numpy as np
    from qmlify.utils import serialize_xml, write_pickle
    #load the npz
    npz = np.load(os.path.join(local_path, factory_npz), allow_pickle=True)
    systems_dict = npz['arr_0'].item()
    for phase in phases:
        old_system = systems_dict[phase]._old_system
        new_system = systems_dict[phase]._new_system
        old_sys_filename = os.path.join(local_path, f"{phase}.old_system.xml")
        new_sys_filename = os.path.join(local_path), f"{phase}.new_system.xml")
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
    extract_sys_top(to_dir)
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
            np.savez(os.path.join(to_dir, f"{lig}_{phase}.positions.npz"), positions = positions, box_vectors = box_vectors)


####################
###ADMINISTRATORS###
####################
def perses_extraction_admin(ligand_index_pairs,
                            from_dir_parent,
                            to_dir_parent,
                            sh_template = None,
                            phases = ['complex', 'solvent'],
                            write_log = True,
                            submission_call = 'bsub <'):
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

    if sh_template is None:
        from pkg_resources import resource_filename
        sh_template = resource_filename('qmlify', 'data/templates/cpu_daemon.sh')

    for i,j in ligand_index_pairs:
        from_dir = os.path.join(from_dir_parent, f"lig{i}to{j}")
        to_dir = os.path.join(to_dir_parent, f"lig{i}to{j}")
        line_to_write = f"python -c 'from qmlify.perses_compatibility_utils import extract_perses_repex_to_local; extract_perses_repex_to_local({from_dir}, {to_dir}, {phases})'"
        write_bsub_delete([line_to_write], sh_template, f"lig{i}to{j}_perses_extraction", write_log = write_log, submission_call = submission_call)

def forward_propagation_admin(ligand_index_pairs, phases, annealing_steps, parent_dir, extraction_indices = range(0,200,2), nondefault_kwarg_dict=None):
    """
    performs ensemble annealing in the forward direction
    """
    import os
    from qmlify.qmlify_data import yaml_keys #template

    yaml_dict = {key: None for key in yaml_keys}
    yaml_dict['num_steps'] = annealing_steps
    yaml_dict['direction'] = 'forward'

    system_template =


    for i,j in ligand_index_pairs:
