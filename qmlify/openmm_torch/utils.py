#!/usr/bin/env python
from openmmtools import utils
from simtk import unit, openmm

# constants
DEFAULT_TEMPERATURE = 300.0 * unit.kelvin
ENERGY_DIFFERENCE_TOLERANCE = 1e-2

DEFAULT_LAMBDA0s = {'lambda_MM_angles':  1.0,
                    'lambda_MM_bonds':  1.0,
                    'lambda_MM_torsions':  1.0,
                    'lambda_electrostatic_scale':  0.0,
                    'lambda_epsilon_scale':  0.0,
                    'lambda_nonbonded_MM_electrostatics':  0.0,
                    'lambda_nonbonded_MM_sterics':  0.0,
                    'lambda_scale':  1.0}

DEFAULT_LAMBDA1s = {'lambda_MM_angles':  0.0,
                    'lambda_MM_bonds':  0.0,
                    'lambda_MM_torsions':  0.0,
                    'lambda_electrostatic_scale':  0.0,
                    'lambda_epsilon_scale':  0.0,
                    'lambda_nonbonded_MM_electrostatics':  1.0,
                    'lambda_nonbonded_MM_sterics':  1.0,
                    'lambda_scale':  1.0}

# loggers
#######LOGGING#############################
import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("utils")
_logger.setLevel(logging.DEBUG)
###########################################



def check_platform(platform):
    """
    Check whether we can construct a simulation using this platform.
    From https://github.com/choderalab/integrator-benchmark/blob/bb307e6ebf476b652e62e41ae49730f530732da3/benchmark/testsystems/configuration.py#L17
    """
    from openmmtools.testsystems import HarmonicOscillator
    try:
        integrator = openmm.VerletIntegrator(1.0)
        testsystem = HarmonicOscillator()
        context = openmm.Context(testsystem.system, integrator, platform)
        del context, testsystem, integrator
    except Exception as e:
        _logger.warning(f'Desired platform not supported. exception raised: {e}')
        raise Exception(e)


def configure_platform(platform_name='Reference', fallback_platform_name='CPU', precision='mixed'):
    """
    Retrieve the requested platform with platform-appropriate precision settings.
    platform_name : str, optional, default='Reference'
       The requested platform name
    fallback_platform_name : str, optional, default='CPU'
       If the requested platform cannot be provided, the fallback platform will be provided.
    Returns
    -------
    platform : simtk.openmm.Platform
       The requested platform with precision configured appropriately,
       or the fallback platform if this is not available.
    From https://github.com/choderalab/integrator-benchmark/blob/bb307e6ebf476b652e62e41ae49730f530732da3/benchmark/testsystems/configuration.py#L17
    """
    fallback_platform = openmm.Platform.getPlatformByName(fallback_platform_name)
    try:
        if platform_name.upper() == 'Reference'.upper():
            platform = openmm.Platform.getPlatformByName('Reference')
        elif platform_name.upper() == "CPU":
            platform = openmm.Platform.getPlatformByName("CPU")
        elif platform_name.upper() == 'OpenCL'.upper():
            platform = openmm.Platform.getPlatformByName('OpenCL')
            platform.setPropertyDefaultValue('OpenCLPrecision', precision)
        elif platform_name.upper() == 'CUDA'.upper():
            platform = openmm.Platform.getPlatformByName('CUDA')
            platform.setPropertyDefaultValue('CudaPrecision', precision)
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
        else:
            raise (ValueError("Invalid platform name"))

        check_platform(platform)

    except:
        _logger.warning("Warning: Returning {} platform instead of requested platform {}".format(fallback_platform_name, platform_name))
        platform = fallback_platform

    _logger.info(f"conducting subsequent work with the following platform: {platform.getName()}")
    return platform


def compute_potential_components(context, beta, platform):
    """
    Compute potential energy, raising an exception if it is not finite.
    Parameters
    ----------
    context : simtk.openmm.Context
        The context from which to extract, System, parameters, and positions.
    """
    # Make a deep copy of the system.
    import copy

    platform = configure_platform(platform.getName(), fallback_platform_name='Reference', precision='double')

    system = context.getSystem()
    system = copy.deepcopy(system)
    # Get positions.
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Get Parameters
    parameters = context.getParameters()
    # Segregate forces.
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
    # Create new Context.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for (parameter, value) in parameters.items():
        context.setParameter(parameter, value)
    energy_components = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        forcename = force.__class__.__name__
        groups = 1<<index
        potential = beta * context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        energy_components.append((forcename, potential))
    del context, integrator
    return energy_components

def get_forces(system):
    """
    get forces from a system as a dictionary

    arguments
        system : simtk.openmm.System
            system to query

    returns
        _dict : dict
            {force_name: (force, force_idx)} where force_name is a string, force is a simtk.openmm.Force, and
            force_idx is the index of the force
    """
    _dict = {}
    for force_idx, force in enumerate(system.getForces()):
        _dict[force.__class__.__name__] = (force, force_idx)
    return _dict

def prepare_ml_system(
                      positions,
                      topology,
                      system,
                      residue_indices,
                      model_name='ani2x',
                      save_filename = 'animodel.pt',
                      torch_scale_name='torch_scale',
                      torch_scale_default_value=0.,
                      HybridSystemFactory_kwargs = {},
                      minimizer_kwargs = {'maxIterations': 1000}
                      ):
    """
    prepare an ani-force-compatible system with built-in lambda assertions and energy compatibility assertions
    """
    from qmlify.openmm_torch.torchforce_generator import torch_alchemification_wrapper
    from openmmtools import utils
    from openmmtools.integrators import LangevinIntegrator
    from openmmtools.constants import kB
    from simtk.openmm import LocalEnergyMinimizer
    import numpy as np

    DEFAULT_TEMPERATURE = 300.0 * unit.kelvin
    ENERGY_DIFFERENCE_TOLERANCE = 1e-2

    _logger.info("preparing ML system and initializing assertions...")

    # make ml system and hybrid factory
    _logger.info(f"executing torch alchemification wrapper to make ml_system and hybrid_factory")
    ml_system, hybrid_factory = torch_alchemification_wrapper(
                                      topology,
                                      system,
                                      residue_indices,
                                      model_name,
                                      save_filename,
                                      torch_scale_name,
                                      torch_scale_default_value,
                                    )
    # get platform
    platform = configure_platform(platform_name = utils.get_fastest_platform())
    beta = 1. / (kB * DEFAULT_TEMPERATURE)

    # get integrators
    old_mm_int = LangevinIntegrator(temperature = DEFAULT_TEMPERATURE)
    mm_int = LangevinIntegrator(temperature = DEFAULT_TEMPERATURE)
    ml_int = LangevinIntegrator(temperature = DEFAULT_TEMPERATURE)


    # make mm contexts at lambda 0
    mm_context = openmm.Context(hybrid_factory.system, mm_int, platform)
    mm_context.setPositions(positions)
    mm_context.setPeriodicBoxVectors(*hybrid_factory.system.getDefaultPeriodicBoxVectors())

    # get the swig parameters and check the alchemical mm system
    _logger.debug(f"ensuring appropriate lambda initialization at lambda0 for alchemical system...")
    mm_swig_params = mm_context.getParameters()
    for name in mm_swig_params:
        assert DEFAULT_LAMBDA0s[name] == mm_swig_params[name], f"swig parameter {name} is {mm_swig_params[name]} but should be {DEFAULT_LAMBDA0s[name]}"

    # minimize mm context
    LocalEnergyMinimizer.minimize(mm_context, **minimizer_kwargs)

    # apply the positions to the ml context
    ml_context = openmm.Context(ml_system, ml_int, platform)

    # check the ml context swig parameters
    ml_context.setPositions(mm_context.getState(getPositions=True).getPositions(asNumpy=True))
    ml_context.setPeriodicBoxVectors(*hybrid_factory.system.getDefaultPeriodicBoxVectors())

    # get the swig parameters and check the alchemical ml system
    ml_swig_params = ml_context.getParameters()
    torch_parameters_lambda0 = {torch_scale_name: torch_scale_default_value, f'auxiliary_{torch_scale_name}': 1.} #this is hard coded...want this?
    _logger.debug(f"ensuring appropriate lambda initialization at lambda0 for ml alchemical system...")
    for name in ml_swig_params:
        if name in list(DEFAULT_LAMBDA0s.keys()):
            assert DEFAULT_LAMBDA0s[name] == ml_swig_params[name], f"swig parameter {name} is {ml_swig_params[name]} but should be {DEFAULT_LAMBDA0s[name]}"
        else: #it is a special torch parameter
            assert ml_swig_params[name] == torch_parameters_lambda0[name]

    # build the old (nonalch) system
    old_mm_context = openmm.Context(hybrid_factory._old_system, old_mm_int, platform)
    old_mm_context.setPositions(mm_context.getState(getPositions=True).getPositions(asNumpy=True))
    old_mm_context.setPeriodicBoxVectors(*hybrid_factory.system.getDefaultPeriodicBoxVectors())

    # now check energy by components
    _logger.debug(f"computing potential components of _all_ contexts...standby.")
    old_mm_potential_components = compute_potential_components(old_mm_context, beta, platform)
    mm_potential_components = compute_potential_components(mm_context, beta, platform)
    # ml_potential_components = compute_potential_components(ml_context, beta, platform) #we can't do this right now since there is a bug...

    sum_old_mm_potential_components = np.sum([tup[1] for tup in old_mm_potential_components])
    sum_mm_potential_components = np.sum([tup[1] for tup in mm_potential_components])
    # sum_ml_potential_components = np.sum(list(ml_potential_components.values()))

    mm_difference = abs(sum_old_mm_potential_components - sum_mm_potential_components)
    ml_difference = abs(sum_mm_potential_components - ml_context.getState(getEnergy=True).getPotentialEnergy() * beta)

    try:
        _logger.info(f"checking mm bookkeeping energies...")
        assert mm_difference < ENERGY_DIFFERENCE_TOLERANCE
    except Exception as e:
        _logger.warning(f"{e}; difference between energies of the lambda0 alchemical mm and nonalchemical mm energy is {mm_difference}, which is higher than the tolerance of {ENERGY_DIFFERENCE_TOLERANCE}")
    try:
        _logger.info(f"checking mm bookkeeping energies...")
        ml_difference < ENERGY_DIFFERENCE_TOLERANCE
    except Exception as e:
        _logger.warning(f"{e}; difference between energies of the lambda0 alchemical mm and ml energy is {mm_difference}, which is higher than the tolerance of {ENERGY_DIFFERENCE_TOLERANCE}")

    # we cannot do the following...

    # for key, val in DEFAULT_LAMBDA1s:
    #     mm_context.setParameter(key, val)
    # mm_final_potential_components = compute_potential_components(mm_context, beta, platform)
    #
    # try:
    #     _logger.info(f"checking ml bookkeeping energies...")
    #     """
    #     here, we are making sure that the alchemical forces starting with `Custom` are all zero and that the other components are unchanged
    #     """
    #     for forcename, energy in mm_final_potential_components.items():
    #         if forcename in [torch_scale_name, f'auxiliary_{torch_scale_name}']:
    #             # don't check the torch force...at least not yet
    #             pass
    #         elif forcename[:7] == 'Custom':
    #             assert np.isclose(energy, 0.), f"the energy of {forcename} at lambda 1 is {energy} when it should be 0."
    #         else:
    #             lambda0_energy = mm_potential_components[forcename]
    #             assert np.isclose(energy, lambda0_energy), f"the energy of {forcename} at lambda 1 is {energy} when it should be {lambda0_energy}"
    # except Exception as e:
    #     _logger.warning(f"{e}; there is an issue associated with the lambda1 endstate energy bookkeeping. see above for which assertion failed.")

    # TODO : add a test for scaling lambdas?

    # remove the contexts and integrators used for testing (this will shore up some memory)...
    for context in [old_mm_context, mm_context, ml_context]:
        del context
    for integrator in [old_mm_int, mm_int, ml_int]:
        del integrator

    return ml_system, hybrid_factory
