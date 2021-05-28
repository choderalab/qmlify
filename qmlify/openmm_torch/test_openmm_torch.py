#!/usr/bin/env python
"""
write tests for the `force_hybridization` and `torchforce_generator` `.py`s
"""
from simtk import unit, openmm
from openmmtools import testsystems
DEFAULT_TEMPERATURE = 300.0 * unit.kelvin
ENERGY_DIFFERENCE_TOLERANCE = 1e-2

def get_HostGuestExplicit():
    testsystem_class = testsystems.HostGuestExplicit(constraints = None,
                                                     hydrogenMass = 4 * unit.amus,
                                                     )

    #remove the CMMotionRemover
    num_forces = testsystem_class.system.getNumForces()
    testsystem_class.system.removeForce(num_forces - 1) # remove the CMMotionRemover force because it is unknown

    return testsystem_class


def make_HybridSystemFactory(testsystem_class = get_HostGuestExplicit(),
                             alchemical_residue_indices = [1],
                             HybridSystemFactory_kwargs = {}):
    """
    make and return a `qmlify.openmm_torch.force_hybridization.HybridSystemFactory`
    """
    from qmlify.openmm_torch.utils import get_forces
    from qmlify.openmm_torch.force_hybridization import HybridSystemFactory

    system, positions, topology = testsystem_class.system, testsystem_class.positions, testsystem_class.topology

    hsf = HybridSystemFactory(topology = topology,
                              alchemical_residue_indices = alchemical_residue_indices,
                              system = system,
                              **HybridSystemFactory_kwargs)
    return hsf, testsystem_class

def test_HybridSystemFactory():
    """
    run the `make_HybridSystemFactory`
    """
    from qmlify.openmm_torch.utils import configure_platform
    from openmmtools import utils

    hsf, testsystem_class = make_HybridSystemFactory()
    platform = configure_platform(platform_name=utils.get_fastest_platform(),
                                  fallback_platform_name='CPU',
                                  precision='mixed')

    # non/alchemical integrators
    from openmmtools.integrators import LangevinIntegrator
    nonalch_int = LangevinIntegrator(temperature=DEFAULT_TEMPERATURE)
    alch_int = LangevinIntegrator(temperature=DEFAULT_TEMPERATURE)

    system, alch_system = hsf._old_system, hsf.system

    nonalch_context, alch_context = openmm.Context(system, nonalch_int, platform), openmm.Context(alch_system, alch_int, platform)

    for context in [nonalch_context, alch_context]:
        context.setPositions(testsystem_class.positions)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

    nonalch_energy = nonalch_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
    alch_energy = alch_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)

    assert abs(alch_energy - nonalch_energy) < ENERGY_DIFFERENCE_TOLERANCE, f"the nonalchemical energy of {nonalch_energy} and the alchemical energy (at lambda=0) of {alch_energy} has a difference that is greater than {ENERGY_DIFFERENCE_TOLERANCE}"


def test_torchforce_generator():
    """
    simple test to make sure that the torchforce_generator instantiates properly and has a sufficiently low energy mismatch at the alchemical 0 endstate
    """
    from openmmtools import utils
    from qmlify.openmm_torch.torchforce_generator import torch_alchemification_wrapper
    import os

    testsystem_class = get_HostGuestExplicit()
    ml_system, hsf_mod = torch_alchemification_wrapper(testsystem_class.topology,
                                                       testsystem_class.system,
                                                       [1],
                                                       save_filename = 'test.pt')
    os.system('rm test.pt')

def test_lambda_dependent_energy_bookkeeping():
    """
    pass HostGuestExplicit testsystem through the `prepare_ml_system` generator.
    """
    from qmlify.openmm_torch.utils import prepare_ml_system
    import os
    testsystem_class = get_HostGuestExplicit()
    _, _ = prepare_ml_system(
                          positions = testsystem_class.positions,
                          topology = testsystem_class.topology,
                          system = testsystem_class.system,
                          residue_indices = [1],
                          model_name='ani2x',
                          save_filename = 'test.pt',
                          torch_scale_name='torch_scale',
                          torch_scale_default_value=0.,
                          HybridSystemFactory_kwargs = {},
                          minimizer_kwargs = {'maxIterations': 100}
                          )
    os.system('rm test.pt')
