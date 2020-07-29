from perses.samplers.multistate import HybridRepexSampler
from openmmtools.multistate import MultiStateReporter

reporter = MultiStateReporter(storage='out-solvent.nc')
simulation = HybridRepexSampler.from_storage(reporter)


total_steps = 10000
run_so_far = simulation.iteration
left_to_do = total_steps - run_so_far
simulation.extend(n_iterations=left_to_do)
