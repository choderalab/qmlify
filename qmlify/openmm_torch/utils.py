#!/usr/bin/env python
from openmmtools import utils
from simtk import unit, openmm


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
        print(f'Desired platform not supported. exception raised: {e}')
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
        print(
        "Warning: Returning {} platform instead of requested platform {}".format(fallback_platform_name, platform_name))
        platform = fallback_platform

    print(f"conducting subsequent work with the following platform: {platform.getName()}")
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
