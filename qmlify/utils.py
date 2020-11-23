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
                                alchemify=False,
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
    from openmmtools.states import ThermodynamicState, CompoundThermodynamicState
    from qmlify.propagation import ANI_force_and_energy
    from qmlify.propagation import Integrator

    #make an ani_handler
    atom_map, species_str = generate_mol_atom_subset_map(md_topology, md_subset_topology, **kwargs)
    ani_handler = ANI_force_and_energy(ani_model, atoms=species_str, **kwargs)

    pressure, temperature = integrator_kwargs['pressure'], integrator_kwargs['temperature']
    pdf_state = ThermodynamicState(system = system, temperature = temperature, pressure = pressure)
    pdf_state_subset = ThermodynamicState(system = system_subset, temperature = temperature)

    if alchemify:
        from openmmtools import alchemy
        subset_alch_region = alchemy.AlchemicalRegion(alchemical_atoms=list(atom_map.values()), alchemical_torsions=True)
        factory = alchemy.AbsoluteAlchemicalFactory()
        subset_alchemical_system = factory.create_alchemical_system(pdf_state_subset.system, subset_alch_region)
        pdf_state_subset_thermo = ThermodynamicState(system = subset_alchemical_system, temperature = temperature)
        pdf_state_subset = CompoundThermodynamicState(thermodynamic_state=pdf_state_subset_thermo,
                                                   composable_states=[alchemical_state])
    else:
        pass

    integrator = Integrator(**integrator_kwargs)

    return pdf_state, pdf_state_subset, integrator, ani_handler, atom_map

def position_extractor(positions_cache_filename, index_to_extract):
    """
    pull a snapshot of positions (and box_vectors) from a `.npz` file

    arguments
        positions_cache_filename : str
            full path of the .npz file to load
        index_to_extract : int
            index that will be extracted

    returns
        positions : np.ndarray(N, 3)
            position of frame to extract
        box_vectors : np.ndarray(3,3) or None
            box vectors of the frame to extract if 'box_vectors' is a variable of the .npz
    """
    if positions_cache_filename[-4:] != '.npz':
        raise Exception(f"the file mist be a .npz")

    file = np.load(positions_cache_filename)
    all_positions = file['positions']
    try:
        all_box_vectors = file['box_vectors']
        assert len(all_box_vectors) == len(all_positions)
    except Exception as e:
        print(f"box vector extractor error: {e}")
        all_box_vectors = None

    positions = all_positions[index_to_extract]
    assert positions.shape[1] == 3, f"the extracted position shapes are: {positions.shape}"
    box_vectors = all_box_vectors[index_to_extract] if all_box_vectors is not None else None
    if box_vectors is not None:
        assert box_vectors.shape == (3,3), f"the box vectors shape is: {box_vectors.shape}"

    return positions, box_vectors


def load_yaml(yml_filename):
    """
    load a yaml

    arguments
        yml_filename : str
            yaml file to load

    returns
        setup_options : dict
            dictionary of setup_options
    """
    import yaml

    yaml_file = open(yml_filename, 'r')
    setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()
    return setup_options

def write_yaml(dictionary, yml_filename):
    """
    write a yaml

    arguments
        dictionary : dict
            dictionary to write
        yml_filename : str
            yaml file to write
    """
    import yaml
    with open(yml_filename, 'w') as outfile:
        yaml.dump(dictionary, outfile, default_flow_style=False)

def deserialize_xml(xml_filename):
    """
    load and deserialize an xml

    arguments
        xml_filename : str
            full path of the xml filename

    returns
        xml_deserialized : deserialized xml object
    """
    from simtk.openmm.openmm import XmlSerializer
    with open(xml_filename, 'r') as infile:
        xml_readable = infile.read()
    xml_deserialized = XmlSerializer.deserialize(xml_readable)
    return xml_deserialized

def serialize_xml(object, xml_filename):
    """
    load and deserialize an xml

    arguments
        object : object
            serializable
        xml_filename : str
            full path of the xml filename
    """
    from simtk.openmm.openmm import XmlSerializer
    with open(xml_filename, 'w') as outfile:
        serial = XmlSerializer.serialize(object)
        outfile.write(serial)

def depickle(pickle_filename):
    """
    load a pickle

    arguments
        pickle_filename : str
            name of pickle
    returns
        pickle : loaded pickle object
    """
    import pickle

    with open(pickle_filename, 'rb') as f:
        pickle = pickle.load(f)
    return pickle

def write_pickle(object, pickle_filename):
    """
    write a pickle

    arguments
        object : object
            picklable object
        pickle_filename : str
            name of pickle
    """
    import pickle
    with open(pickle_filename, 'wb') as f:
        pickle.dump(object, f)


def generate_random_string(length):
    """
    just that...generate a random string

    arguments
        length : int
            length of random string
    returns
        _string : str
            random string
    """
    import string
    import random
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = length))
    _string = str(res)
    return _string

def write_bsub_delete(lines_to_write, template, template_suffix, write_to_dir, write_log=False, submission_call = 'bsub <', cat = True, delete = True):
    """
    write a line to a template ('r'), save template + template_prefix, submit, ask whether to write log, delete file;
    template must end in '.sh'

    arguments
        lines_to_write: list(str)
            lines to write to template
        template : str
            filepath to template
        template_suffix : str
            suffix to add to write template; separated by '.', followed by '.sh'
        write_to_dir : str
            path that the file will be written to
        write_log : bool, default False
            whether to write a log file in the submission
        submission_call : str, default 'bsub <'
            submission call to functor
    """
    import os
    assert template[-3:] == '.sh'
    with open(template, 'r') as f:
        line_template = f.readlines()

    local_template_name = template.split('/')[-1]

    write_to = os.path.join(write_to_dir, local_template_name[:-2]+template_suffix+'.sh')
    with open(write_to, 'w') as f:
        suffix = f" &> {write_to[:-2]}log" if write_log else ''
        for line in line_template: f.writelines(line) #write the lines of the template
        for line in lines_to_write:
            toline = line + suffix
            f.writelines(toline)

    os.system(f"{submission_call} {write_to}")
    if cat:
        print(f"write to file {write_to}: ")
        os.system(f"{write_to}")
    if delete: os.remove(write_to)

def exp_distribution(works):
    """
    pull a normalized weight distribution from a work distribution

    arguments
        works : np.ndarray(N)
            array of accumulated works

    returns
        weights : np.ndarray(N)
            normalized weights corresponding to works
    """
    from scipy.special import logsumexp
    import numpy as np
    xs = -works
    a = np.max(xs)
    xs_primed = xs - a
    log_normalizer = a + logsumexp(xs_primed)
    log_probs = xs - log_normalizer

    assert np.isclose(np.sum(np.exp(log_probs)), 1.)
    return np.exp(log_probs)
