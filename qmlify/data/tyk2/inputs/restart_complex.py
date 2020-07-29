from perses.samplers.multistate import HybridRepexSampler
from openmmtools.multistate import MultiStateReporter

import logging

logging.basicConfig(level=logging.NOTSET)
_logger = logging.getLogger("utils.openeye")
_logger.setLevel(logging.DEBUG)

reporter = MultiStateReporter(storage='out-complex.nc')
simulation = HybridRepexSampler.from_storage(reporter)


total_steps = 10000
run_so_far = simulation.iteration
left_to_do = total_steps - run_so_far
_logger.info(f'{left_to_do}')
_logger.debug('debugging')
simulation.extend(n_iterations=left_to_do)
