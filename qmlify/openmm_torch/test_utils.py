#!/usr/bin/env python
"""
test some important utilities
"""
from simtk import openmm, unit
#######LOGGING#############################
import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("test_utils")
_logger.setLevel(logging.DEBUG)
###########################################



def test_reorganize_forces():
    """test the `reorganize_forces` functionality by adding a `CMMotionRemover` to a `HostGuestSystem`"""
    from qmlify.openmm_torch.test_openmm_torch import get_HostGuestExplicit, DEFAULT_TEMPERATURE
    from qmlify.openmm_torch.utils import reorganize_forces_v2, configure_platform
    from openmmtools import utils
    from copy import deepcopy
    import numpy as np
    testsystem_class = get_HostGuestExplicit()
    new_force =  openmm.CMMotionRemover()
    og_system = testsystem_class.system
    _mod_system = deepcopy(og_system)

    mod_system = reorganize_forces_v2(_mod_system, new_force)
    _logger.info(f"og_system_forces: {og_system.getForces()}")
    _logger.info(f"mod_system_forces: {mod_system.getForces()}")
    
    # create a context and get the energy...
    platform = configure_platform(platform_name=utils.get_fastest_platform().getName(),
                                  fallback_platform_name='CPU',
                                  precision='mixed')

    # non/alchemical integrators
    from openmmtools.integrators import LangevinIntegrator
    _int = LangevinIntegrator(temperature=DEFAULT_TEMPERATURE)
    mod_int = LangevinIntegrator(temperature=DEFAULT_TEMPERATURE)
    context = openmm.Context(og_system, _int, platform)
    mod_context = openmm.Context(mod_system, mod_int, platform)


    for _context in [context, mod_context]:
        _context.setPositions(testsystem_class.positions)
        _context.setPeriodicBoxVectors(*og_system.getDefaultPeriodicBoxVectors())

    og_state_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
    mod_state_e = mod_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
    print(og_state_e, mod_state_e)
    assert np.isclose(og_state_e, mod_state_e)



if __name__ == "__main__":
    test_reorganize_forces() 
