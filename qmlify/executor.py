"""
qmlify.py
feee energy calculations with ANI2x

Handles the primary functions
"""

def run(setup_dict):
    """
    execute a Propagator
    """
    import torchani
    from simtk import unit
    import sys
    import numpy as np
    import mdtraj as md
    from coddiwomple.particles import Particle
    from coddiwomple.openmm.states import OpenMMParticleState
    from qmlify.utils import load_yaml, deserialize_xml, position_extractor, generate_propagator_inputs, depickle

    #pull systems
    system = deserialize_xml(setup_dict['system'])
    system_subset = deserialize_xml(setup_dict['subset_system'])

    #load topologies
    md_topology = md.Topology.from_openmm(depickle(setup_dict['topology']))
    md_subset_topology = md.Topology.from_openmm(depickle(setup_dict['subset_topology']))

    #load positions and box vectors
    positions, box_vectors = position_extractor(positions_cache_filename = setup_dict['positions_cache_filename'], index_to_extract = setup_dict['position_extraction_index'])
    positions *= unit.nanometers
    if box_vectors is not None: box_vectors *= unit.nanometers


    #integrator integrator_kwargs
    default_integrator_kwargs = {'temperature': 300.0 * unit.kelvin,
                                 'collision_rate': 1.0 / unit.picoseconds,
                                 'timestep': 1.0 * unit.femtoseconds,
                                 'splitting': "V R O R F",
                                 'constraint_tolerance': 1e-6,
                                 'pressure': 1.0 * unit.atmosphere}
    if 'integrator_kwargs' in setup_dict.keys():
        integrator_kwargs = setup_dict['integrator_kwargs']
        if integrator_kwargs is not None:
            if 'temperature' in integrator_kwargs.keys(): integrator_kwargs['temperature'] *= unit.kelvin
            if 'collision_rate' in integrator_kwargs.keys(): integrator_kwargs['collision_rate'] /= unit.picoseconds
            if 'timestep' in integrator_kwargs.keys(): integrator_kwargs['timestep'] *= unit.femtoseconds
            if 'pressure' in integrator_kwargs.keys() and integrator_kwargs['pressure'] is not None: integrator_kwargs['pressure'] *= unit.atmosphere
            default_integrator_kwargs.update(integrator_kwargs)

    pdf_state, pdf_state_subset, integrator, ani_handler, atom_map = generate_propagator_inputs(system = system,
                                                                                                system_subset = system_subset,
                                                                                                md_topology = md_topology,
                                                                                                md_subset_topology = md_subset_topology,
                                                                                                ani_model = torchani.models.ANI2x(),
                                                                                                integrator_kwargs = default_integrator_kwargs)


    if setup_dict['direction'] == 'forward':
        from qmlify.propagation import Propagator
        prop = Propagator
    elif setup_dict['direction'] == 'backward':
        from qmlify.propagation import BackwardPropagator
        prop = BackwardPropagator
    elif setup_dict['direction'] == 'ani_endstate':
        from qmlify.propagation import ANIPropagator
        prop = ANIPropagator
    else:
        raise Exception(f"{setup_dict['direction']} is not valid. allowed directions are 'forward', 'backward', 'ani_endstate'")

    propagator = prop(openmm_pdf_state = pdf_state,
                     openmm_pdf_state_subset = pdf_state_subset,
                     subset_indices_map = atom_map,
                     integrator = integrator,
                     ani_handler = ani_handler,
                     context_cache=None,
                     reassign_velocities=True,
                     n_restart_attempts=0)

    particle = Particle(0)
    particle_state = OpenMMParticleState(positions = positions, box_vectors = box_vectors)
    particle.update_state(particle_state)
    particle_state, _return_dict = propagator.apply(particle_state, n_steps = setup_dict['num_steps'], reset_integrator=True, apply_pdf_to_context=True)
    if box_vectors is None:
        particle_state.box_vectors=None

    work_array = np.array(propagator.state_works[0])

    if particle_state.box_vectors is not None:
        np.savez(setup_dict['out_positions_npz'], positions=np.array([particle_state.positions.value_in_unit_system(unit.md_unit_system)]), box_vectors = np.array([particle_state.box_vectors.value_in_unit_system(unit.md_unit_system)]))
    else:
        np.savez(setup_dict['out_positions_npz'], positions=particle_state.positions.value_in_unit_system(unit.md_unit_system))
    np.savez(setup_dict['out_works_npz'], works = work_array)
