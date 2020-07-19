"""
Unit and regression test for propagation.
"""

# Import package, test suite, and other packages as needed
from simtk import unit
import numpy as np
from simtk.openmm import app
import torchani

def generate_testsystem(smiles = 'CCCC',
                        forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
                        forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : None, 'hydrogenMass' : 4 * unit.amus},
                        nonperiodic_forcefield_kwargs = {'nonbondedMethod': app.NoCutoff},
                        periodic_forcefield_kwargs = {'nonbondedMethod': app.PME},
                        small_molecule_forcefield = 'gaff-2.11',
                        padding=9*unit.angstroms,
                        ionicStrength=0.0*unit.molar,
                        water_model = 'tip3p',
                        pressure = 1.0 * unit.atmosphere,
                        temperature = 300 * unit.kelvin,
                        barostat_period = 50,
                        **kwargs
                        ):
    """
    internal small molecule testsystem generator

    arguments
        smiles : str, default 'CCCC'
            smiles string of the small molecule
        forcefield_files = list, default ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
            forcefield file names
        forcefield_kwargs : dict, default {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : None, 'hydrogenMass' : 4 * unit.amus}
            forcefield kwargs
        nonperiodic_forcefield_kwargs : dict, default {'nonbondedMethod': app.NoCutoff}
            dict of nonperiodic forcefield kwargs
        small_molecule_forcefield :  str, default 'gaff-2.11'
            small molecule forcefield to parameterize smiles
        padding : simtk.unit.Quantity (compatible with unit.angstroms),default 9.0 * unit.angstroms
            solvent padding
        ionicStrength : simtk.unit.Quantity, default 0.0*unit.molar
            ionic strength of solvent
        water_model : str, default 'tip3p'
            water model for solvation
        pressure : simtk.unit.Quantity, default 1.0 * unit.atmosphere
            pressure of the barostat
        temperature : simtk.unit.Quantity, default 300 * unit.kelvin
            temperature of barostat
        barostat_period : int, default 50
            integer of the barostat period

    returns
        vac_sys_pos_top : tuple
            tuple of the vacuum openmm.System, unit.Quantity(unit.nanometers), openmm.Topology
        sol_sys_pos_top : tuple
            tuple of the solvent openmm.System, unit.Quantity(unit.nanometers), openmm.Topology
    """
    from openforcefield.topology import Molecule
    from perses.utils.openeye import smiles_to_oemol
    from openmmforcefields.generators.system_generators import SystemGenerator
    from perses.utils.openeye import OEMol_to_omm_ff
    from simtk import openmm

    oemol = smiles_to_oemol(smiles)
    off_molecules = [Molecule.from_openeye(oemol)]
    vac_system_generator = SystemGenerator(forcefields=forcefield_files,
                                       small_molecule_forcefield=small_molecule_forcefield,
                                       forcefield_kwargs=forcefield_kwargs,
                                       nonperiodic_forcefield_kwargs = nonperiodic_forcefield_kwargs, molecules = off_molecules)
    barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_period)
    sol_system_generator = SystemGenerator(forcefields=forcefield_files,
                                       small_molecule_forcefield=small_molecule_forcefield,
                                       forcefield_kwargs=forcefield_kwargs,
                                       periodic_forcefield_kwargs = periodic_forcefield_kwargs,
                                       molecules = off_molecules,
                                       barostat = barostat)


    vac_system, vac_positions, vac_topology = OEMol_to_omm_ff(oemol, vac_system_generator)

    #now i can attempt to solvate
    modeller = app.Modeller(vac_topology, vac_positions)
    modeller.addSolvent(sol_system_generator.forcefield, model=water_model, padding=padding, ionicStrength=ionicStrength)
    sol_positions, sol_topology = modeller.getPositions(), modeller.getTopology()
    sol_positions = unit.quantity.Quantity(value = np.array([list(atom_pos) for atom_pos in sol_positions.value_in_unit_system(unit.md_unit_system)]), unit = unit.nanometers)
    sol_system = sol_system_generator.create_system(sol_topology)

    vac_sys_pos_top = (vac_system, vac_positions, vac_topology)
    sol_sys_pos_top = (sol_system, sol_positions, sol_topology)

    #a quick assertion to make sure the nonbonded forces are being treated properly
    vac_nbf, sol_nbf = pull_force_by_name(vac_system, 'NonbondedForce'), pull_force_by_name(sol_system, 'NonbondedForce')
    assert not vac_nbf.usesPeriodicBoundaryConditions()
    assert sol_nbf.usesPeriodicBoundaryConditions()

    return vac_sys_pos_top, sol_sys_pos_top

def propagator_testprep():
    """
    wrapper that outputs all necessary Propagator (and subclass) inputs for testing (the test system is butane solvated in tip3p)

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
        particle : coddiwomple.particles.Particle
            particle containing a coddiwomple.openmm.OpenMMParticleState (subclass of openmmtools.states.SamplerState)
    """
    import mdtraj as md
    from coddiwomple.particles import Particle
    from coddiwomple.openmm.states import OpenMMParticleState

    vac_sys_pos_top, sol_sys_pos_top = generate_testsystem()
    vac_system, vac_positions, vac_topology = vac_sys_pos_top
    sol_system, sol_positions, sol_topology = sol_sys_pos_top

    md_topology = md.Topology.from_openmm(sol_topology)
    md_subset_topology = md.Topology.from_openmm(vac_topology)
    pdf_state, pdf_state_subset, integrator, ani_handler, atom_map = generate_propagator_inputs(system = sol_system,
                                                                                                system_subset = vac_system,
                                                                                                md_topology = md_topology,
                                                                                                md_subset_topology = md_subset_topology)
    particle = Particle(0)
    box_vectors = sol_system.getDefaultPeriodicBoxVectors()
    particle_state = OpenMMParticleState(positions = sol_positions, box_vectors = box_vectors)
    particle.update_state(particle_state)

    return pdf_state, pdf_state_subset, integrator, ani_handler, atom_map, particle



def test_ANI_force_and_energy(platform = 'cpu', temperature = 300*unit.kelvin):
    """
    use openmmtools AlanineDipeptideVacuum as a test system and execute all methods in the class to assert functionality and unit'd quantities
    WARNING : platform 'gpu' is experimental
    """
    from openmmtools.testsystems import AlanineDipeptideVacuum
    from qmlify.propagation import ANI_force_and_energy
    import torch

    ala = AlanineDipeptideVacuum()
    md_topology = md.Topology.from_openmm(ala.topology)

    model = torchani.models.ANI2x()
    atoms = [atom.element.symbol for atom in md_topology.atoms]

    #init
    ani_handler = ANI_force_and_energy(model = torchani.models.ANI2x(),
                                        atoms=atoms,
                                        platform='cpu',
                                        temperature=temperature)
    #calculate a force
    force, energy = ani_handler.calculate_force(ala.positions)
    assert force.unit.is_compatible(unit.kilojoules_per_mole/unit.nanometers)
    assert energy.unit.is_compatible(unit.kilojoules_per_mole)

    #reform energy tensors
    coordinates = torch.tensor([ala.positions.value_in_unit(unit.angstroms)],
                           requires_grad=True, device=ani_handler.device, dtype=torch.float32)
    energy_out = ani_handler._reform_as_energy_tensor(coordinates)
    assert type(energy_out) == torch.Tensor

    #calulate energy (full)
    full_energy_out = ani_handler.calculate_energy(ala.positions)
    assert full_energy_out.unit.is_compatible(unit.kilojoules_per_mole)

def test_Integrator_Propagator(annealing_steps=100):
    """
    test qmlify.propagation.Propagator on solvated butane
    """
    from qmlify.propagation import Propagator
    pdf_state, pdf_state_subset, integrator, ani_handler, atom_map, particle = propagator_testprep()

    propagator = Propagator(openmm_pdf_state = pdf_state,
                     openmm_pdf_state_subset = pdf_state_subset,
                     subset_indices_map = atom_map,
                     integrator = integrator,
                     ani_handler = ani_handler,
                     context_cache=None,
                     reassign_velocities=True,
                     n_restart_attempts=0)
    particle_state, _return_dict = propagator.apply(particle.state, n_steps = annealing_steps, reset_integrator=True, apply_pdf_to_context=True)
