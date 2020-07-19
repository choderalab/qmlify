"""
utilities for pipeline and production
"""
#####Imports#####
from simtk import unit
import numpy as np
from simtk.openmm import app
import torchani #because sometimes we need appropriate default arguments for these functions that call torchani

def pull_force_by_name(system, force_name = 'NonbondedForce', **kwargs):
    """
    pull a force object from a system for querying

    arguments
        system : simtk.openmm.openmm.System
            system that will be queried
        force_name : str, default 'NonbondedForce'
            force name that will be matched

    returns
        force : simtk.openmm.openmm.Force
            matched force
    """
    forces = system.getForces()
    force_dict = {force.__class__.__name__: force for force in forces}
    try:
        force = force_dict[force_name]
    except Exception as e:
        raise Exception(f"{e}")
    return force

def generate_mol_atom_subset_map(md_topology, md_subset_topology, resname = 'MOL', **kwargs):
    """
    generate an index dictionary map from the topology to the subset topology

    arguments
        md_topology : mdtraj.topology
            topology that will be queried (often a solvated small molecule or a molecule in complex)
        md_subset_topology : mdtraj.Topology
            topology subset that will be queried (this is often a vacuum topology of a small molecule of interest)
        resname : str, default 'MOL'
            resname string that will be extracted for map

    returns
        index_map : dict
            index map of {md_top_atom_index : md_subset_top_atom_index} of the matching atoms where key is the index of the md_topology atom and the value is the index of the matching md_subset_topology atom
        species_str : list of str
            list of atom elements corresponding to the index_map

    NOTE : there are internal consistency check to ensure that the atom names are consistent between the two topologies

    """
    import mdtraj as md

    top_indices = md_topology.select(f'resname {resname}')
    subset_top_indices = md_subset_topology.select(f"resname {resname}")
    assert len(top_indices) == len(subset_top_indices)

    top_atom_names = [md_topology.atom(i).name for i in top_indices]
    subset_top_atom_names = [md_subset_topology.atom(i).name for i in subset_top_indices]
    assert all(i==j for i,j in zip(top_atom_names, subset_top_atom_names))

    index_map = {top_index : subset_top_index for top_index, subset_top_index in zip(top_indices, subset_top_indices)}
    species_str = [md_subset_topology.atom(i).element.symbol for i in subset_top_indices]
    return index_map, species_str

def generate_propagator_inputs(system,
                               system_subset,
                               md_topology,
                               md_subset_topology,
                               ani_model = torchani.models.ANI2x(),
                               integrator_kwargs = {'temperature': 300.0 * unit.kelvin,
                                                      'collision_rate': 1.0 / unit.picoseconds,
                                                      'timestep': 1.0 * unit.femtoseconds,
                                                      'splitting': "V R O R F",
                                                      'constraint_tolerance': 1e-6,
                                                      'pressure': 1.0 * unit.atmosphere},
                               **kwargs):
    """
    wrapper utility that pre-generates arguments for the Propagator and its subclasses

    arguments
        system : simtk.openmm.openmm.System
            system that will equip a PDFState
        system_subset : simtk.openmm.openmm.System
            subset system that will equip a subset PDFState
        md_topology : mdtraj.Topology
            mdtraj topology of the analogous system
        md_subset_topology : mdtraj.Topology
            mdtraj topology of the analogous system_subset
        ani_model : torchani.models.Model object, default torchani.models.ANI2x
            model that will equip the ANI handler
        integrator_kwargs : dict, default  {'temperature': 300.0 * unit.kelvin, 'collision_rate': 1.0 / unit.picoseconds, 'timestep': 1.0 * unit.femtoseconds, 'splitting': "V R O R F", 'constraint_tolerance': 1e-6, 'pressure': 1.0 * unit.atmosphere}
            kwarg argument dictionary that is passed to instantiate an Integrator (subclass of OMMLI, subclass of openmmtools.integrators.LangevinIntegrator)

    returns
        pdf_state : coddiwomple.OpenMMPDFState
            subclass of openmmtools.states.ThermodynamicState of the system object
        pdf_state_subset : coddiwomple.OpenMMPDFState
            subclass of openmmtools.states.ThermodynamicState of the system_subset object
        integrator : qmlify.propagation.Integrator
            integrator that equips the Propagator
        ani_handler : qmlify.ANI_force_and_energy
            handler of the ani components
        atom_map : dict
            index map of {md_top_atom_index : md_subset_top_atom_index} of the matching atoms where key is the index of the md_topology atom and the value is the index of the matching md_subset_topology atom
    """
    from openmmtools.states import ThermodynamicState
    from qmlify.propagation import ANI_force_and_energy
    from qmlify.propagation import Integrator

    #make an ani_handler
    atom_map, species_str = generate_mol_atom_subset_map(md_topology, md_subset_topology, **kwargs)
    ani_handler = ANI_force_and_energy(ani_model, atoms=species_str, **kwargs)

    pressure, temperature = integrator_kwargs['pressure'], integrator_kwargs['temperature']
    pdf_state = ThermodynamicState(system = system, temperature = temperature, pressure = pressure)
    pdf_state_subset = ThermodynamicState(system = system_subset, temperature = temperature)

    integrator = Integrator(**integrator_kwargs)

    return pdf_state, pdf_state_subset, integrator, ani_handler, atom_map
