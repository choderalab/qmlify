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

#### Initialization
Perses Replica Exchange calculations can be executed manually from a bash shell by first calling:
```
for i in {1..25}; do python3 inputs/run.py $i; done
```

This execution will iterate through the 24 prespecified relative ligand transformations (see the `ligand_pairs` list variable in `inputs/run.py`) and launch perses calculation setup objects in their respective `ligXtoY` subdirectories. Default replica exchage arguments are specified in `inputs/ani.yaml` (see [here](https://github.com/choderalab/perses/blob/85aa6af259db816f46e99e9272c0ff918e808bd2/perses/app/setup_relative_calculation.py#L47) for non-default input arguments).

Each `python3 inputs/run.py $i` can take upwards of an hour to execute. Hence, we provide an [LSF](https://www.ibm.com/support/knowledgecenter/en/SSWRJV_10.1.0/lsf_users_guide/clusters_jobs_about.html) executable script at `inputs/submit-ligpairs.sh`. However, you will need to `source` your appropriate conda build and activate your `perses`-enabled environment.

#### Extension
From a parent directory containing all `ligXtoY` subdirectories (`X`, `Y` being generated relative transformations), replica exchange simulations can be extended manually by calling
```
cd lig{X}to{Y}
python3 restart_{phase}.py
```
for each phase of interest (e.g. `complex`, `solvent`, `vacuum`). `complex` phase requires a `.pdb` filename that references the protein of interest (in the case of Tyk2, the pdb can be found at `inputs/Tyk2_protein.pdb`). Again, extending the MCMC sampler for a total of 10000 (by default, see `inputs/restart_{phase}.py` `total_steps` `int` variable) iterations can take ~10-20 GPU hrs (depending on the system), so it may be advisable to launch
```
for a in `ls -d */`; do cd $a; echo $a; bsub < ../restart-{phase}.sh; cd ..; done
```
for each phase (`solvent`, `complex`). Again, it is necessary to modify the `restart-{phase}.sh` to your conda build and environment.

Upon completion, each ligand transformation will generate a `lig{X}to{Y}/out-{phase}.nc` file containing a [Simulation](https://github.com/choderalab/perses/blob/85aa6af259db816f46e99e9272c0ff918e808bd2/perses/analysis/load_simulations.py#L13) object which can be queried (see above) to extract relative free energy differences and errors.


