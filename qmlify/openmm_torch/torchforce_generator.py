#!/usr/bin/env python
import numpy as np
from simtk import unit, openmm

#######LOGGING#############################
import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("torchforce_generator")
_logger.setLevel(logging.DEBUG)
###########################################

def make_torchforce(topology,
                             atoms,
                             model_name='ani2x',
                             save_filename = 'animodel.pt',
                             torch_scale_name='torch_scale',
                             torch_scale_default_value=0.,
                             pbc=True):
    """
    creates a scalable (via a global parameter) openmm.TorchForce.
    NOTE: this force will belong to forcegroup 1 by default.
    NOTE: there are actually two `GlobalParameters` equipped here:
        1. `torch_scale_name` goes from 0 to 1 and turns on the `TorchForce`
        2. `auxiliary_torch_scale_name` is a scaling parameter that defaults 1. and is simply multiplicative

    arguments
        topology : openmm.Topology
            topology corresponding to the openmm.System object
        atoms : list(int)
            list of particle indices that will be included in the TorchForce
        model_name : str, default `ani2x`
            the name of the model that wille build the torchforce
        save_filename : str, default `animodel.pt`
            torch module name to save
        torch_scale_name : str, default 'torch_scale'
            the name of the global parameter that scales the TorchForce
        torch_scale_default_value : float, default 1.
            the default value of the `torch_scale_name`
        pbc : bool
            whether to use periodic boundary conditions

    returns
        torch_force : openmm.TorchForce
            the generated TorchForce

    """
    import torch
    import torchani
    import openmmtorch
    if model_name == 'ani1ccx':
        model = torchani.models.ANI1ccx()
    elif model_name == 'ani2x':
        model = torchani.models.ANI2x()
    else:
        raise Exception(f"model name {model_name} is not currently supported")


    # Create the PyTorch model that will be invoked by OpenMM.
    includedAtoms = list(topology.atoms())
    if atoms is not None:
        includedAtoms = [includedAtoms[i] for i in atoms]
    elements = [atom.element.symbol for atom in includedAtoms]
    print(f"elements: {elements}")
    species = model.species_to_tensor(elements).unsqueeze(0)
    print(f"species: {species}")
    #indices = torch.tensor(atoms, dtype=torch.int64) #get the atom indices which to pull

    class ANIForce(torch.nn.Module):
        def __init__(self, indices, model, species):
            super().__init__()
            self.energyScale = torchani.units.hartree2kjoulemol(1)
            self.indices = torch.tensor(indices, dtype=torch.int64)
            self.model = model
            self.species = species

        def forward(self, positions, scale):
            positions = positions.to(torch.float32) #to float
            in_positions = positions[self.indices]
            # print(f"in_positions has the following shape: {in_positions.shape}")
            _, energy = self.model((self.species, 10.0 * in_positions.unsqueeze(0))) #get the energy
            #energy = _energy.sum()
            out = energy * scale * self.energyScale
            # print(f"energy: {energy}; out: {out}")
            return out

    class PBCANIForce(torch.nn.Module):
        def __init__(self, indices, model, species):
            super().__init__()
            self.energyScale = torchani.units.hartree2kjoulemol(1)
            self.indices = torch.tensor(indices, dtype=torch.int64)
            self.model = model
            self.species = species

        def forward(self, positions, boxvectors, scale0, scale1):
            boxsize = boxvectors.diag()
            periodicPositions = positions - torch.floor(positions/boxsize)*boxsize
            periodicPositions = periodicPositions.to(torch.float32) #to float
            in_positions = periodicPositions[self.indices]
            _, energy = self.model((self.species, 10.0 * in_positions.unsqueeze(0))) #get the energy
            #energy = _energy.sum()
            out = energy * scale0 * self.energyScale * scale1
            return out


    f_gen = ANIForce(atoms, model, species) if not pbc else PBCANIForce(atoms, model, species)
    module = torch.jit.script(f_gen)

    # Serialize the compute graph to a file
    module.save(save_filename)

    # Create the TorchForce from the serialized compute graph
    from openmmtorch import TorchForce
    torch_force = TorchForce(save_filename)
    torch_force.setForceGroup(1) #default 1st force group

    if pbc:
        torch_force.setUsesPeriodicBoundaryConditions(True)
    else:
        torch_force.setUsesPeriodicBoundaryConditions(False)

    torch_force.addGlobalParameter(torch_scale_name, torch_scale_default_value)
    torch_force.addGlobalParameter(f"auxiliary_{torch_scale_name}", 1.) #auxiliary torch scale
    return torch_force


def torch_alchemification_wrapper(
                                  topology,
                                  system,
                                  residue_indices,
                                  model_name='ani2x',
                                  save_filename = 'animodel.pt',
                                  torch_scale_name='torch_scale',
                                  torch_scale_default_value=0.
                                ):
    """
    given a topology/system, and an appropriate* residue index, call the `HybridSystemFactory` to alchemify and generate a new system object.
    Subsequently, the system will be equipped with a TorchForce that is also scalable.

    returns
        mod_system : openmm.System
            alchemified system object that is equipped with the TorchForce object
        hybrid_factory : HybridSystemFactory
            the generated hybrid system factory; this is for debugging purposes
    """
    import copy
    from qmlify.openmm_torch.force_hybridization import HybridSystemFactory
    hybrid_factory = HybridSystemFactory(topology,
                                         residue_indices,
                                         system)
    mod_system = copy.deepcopy(hybrid_factory.system)

    torchforce = make_torchforce(topology,
                             hybrid_factory._atoms,
                             model_name,
                             save_filename,
                             torch_scale_name,
                             torch_scale_default_value,
                             pbc = mod_system._system_forces['NonbondedForce'].usesPeriodicBoundaryConditions())

    mod_system.addForce(torchforce)
    return mod_system, hybrid_factory
