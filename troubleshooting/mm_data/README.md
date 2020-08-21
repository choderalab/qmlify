### Relative Free Energy Calculation Data
Tyk2 replica exchange relative free energy calculations from [perses](https://github.com/choderalab/perses) live here. Each `ligXtoY.pi` is a pickled [Simulation](https://github.com/choderalab/perses/blob/85aa6af259db816f46e99e9272c0ff918e808bd2/perses/analysis/load_simulations.py#L13) object. Each `X`, `Y` refers to the alchemical perturbation between ligand indices (ligand indices correspond to `./inputs/Tyk2_ligands_shifted.sdf`). 
`.pi` files be loaded and queried like:
```
import pickle
pickle_file = "simulation.pi"

with open(pickle_file,'rb') as f: simulation = pickle.load(f)
calc_DDG = -sim.bindingdg #relative free energy
calc_dDDG = sim.bindingddg #relative uncertainty of free energy
```

### Executing Perses Replica Exchange Calculations
Manual and [LSF](https://www.ibm.com/support/knowledgecenter/en/SSWRJV_10.1.0/lsf_users_guide/clusters_jobs_about.html) executions for perses calculations are detailed below.

Perses Replica Exchange calculations can be executed manually from a bash shell by first calling:
`for i in {1..25}; do python3 inputs/run.py $i`

This execution will iterate through the 24 prespecified relative ligand transformations (see the `ligand_pairs` list variable in `inputs/run.py`) and launch perses calculation setup objects in their respective `ligXtoY` subdirectories. Default replica exchage arguments are specified in `inputs/ani.yaml`. see [here](https://github.com/choderalab/perses/blob/85aa6af259db816f46e99e9272c0ff918e808bd2/perses/app/setup_relative_calculation.py#L47) for non-default input arguments.


