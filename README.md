# qmlify
==============================
[]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/qmlify/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/qmlify/actions?query=branch%3Amaster+workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/qmlify/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/qmlify/branch/master)

`qmlify` provides python executables for post-processing molecular mechanical (MM) receptor:ligand binding free energy calculations for the purpose of making hybrid machine learning/MM (ML/MM) free energy *corrections*. Current ML implementations are with [Torchani](https://github.com/aiqm/torchani)'s `ANI2x` model.
MM free energy calculations are conducted with [Perses](https://github.com/choderalab/perses)

The code is experimental and subject to change.

## Installation
Install `qmlify` master via `git clone`:
```
git clone https://github.com/choderalab/qmlify.git
cd qmlify
python setup.py install
```

You will also need the latest `conda` release of [Perses](https://github.com/choderalab/perses):
```
conda install -c omnia perses
```

as well as a couple of classes from [coddiwomple](https://github.com/choderalab/coddiwomple) and [Arsenic](https://github.com/openforcefield/Arsenic):

```
git clone https://github.com/choderalab/coddiwomple.git
cd coddiwomple
python setup.py install
```

While some kinks in `Arsenic` are being ironed out, checkout the `fe8724f` commit:
```
git clone https://github.com/openforcefield/Arsenic.git
cd Arsenic
git checkout fe8724f
python setup.py install
```

### Conda Environment
For the purposes of rigor and reproducibility, there is a `conda` environment `yaml` file located at `/experiments/environment_droplet.yml`, from which you can create a local environment, as detailed [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## Use
`qmlify` postprocesses `perses` free energy calculations. You will need `perses` MM relative free energy calculation data before you can post-process it with `qmlify`

### Perses
See `experiments/mm_data/README.md` for instructions on running `perses` replica exchange relative free energy calculations. A set of pre-generated `perses` simulation data is provided at `experiments/mm_data`, which are queried and analyzed in a demonstration at `experiments/free_energy_corrections.ipynb`.

### qmlify
`qmlify` uses bidirectional nonequilibrium (NEQ) free energy calculations to anneal from the MM thermodynamic states (afforded by `perses`) to the hybrid ML/MM thermodynamics states (and back).
Once `perses` free energy calculations are complete, you can extract and reformat the necessary `perses` files for the purpose of conducting a MM-to-ML/MM corrections using:
```
from qmlify.executables import perses_extraction_admin
ligand_indices = [(0,12),(0,7),(10,13),(10,6),(11,0),(11,14),(14,2),(14,8),(14,9),(15,14),(15,4),(1,13),(1,6),(2,7),(3,0),(3,13),(4,12),(4,13),(4,14),(4,5),(4,9),(5,0),(6,0),(8,0)]  # the full set of ligand index pairs for Tyk2
perses_extraction_admin(ligand_indices,
                        '<location of perses data directory to extract from>',
                        '<location of local directory to extract to>',
                        phases = ['solvent', 'complex'],
                        delete_execution=False)
```
Once you have switched to the `extract_to` directory, execute `forward` NEQ:
```
from qmlify.executables import propagation_admin
import os
propagation_admin(ligand_indices, annealing_steps=5000, direction = 'forward', parent_dir = os.getcwd(), extraction_indices = range(0, 200, 2), phases = ['solvent', 'complex'], write_log=True, cat_outputs=False, delete_outputs=False)
```
followed by resampling and decorrelation at the ML/MM endstate:
```
propagation_admin(ligand_indices, annealing_steps=5000, direction = 'ani_endstate', parent_dir = os.getcwd(), extraction_indices = 100, phases = ['solvent', 'complex'], write_log=True, cat_outputs=True, delete_outputs=False, eq_steps=5000)
```
and finally, `backward` NEQ:
```
propagation_admin(ligand_indices, annealing_steps=5000, direction = 'backward', parent_dir = os.getcwd(), extraction_indices = 100, phases = ['solvent', 'complex'], write_log=True, cat_outputs=True, delete_outputs=False, eq_steps=5000)
```
These commands will generate `.npz` files containing a `numpy` array of *cumulative* reduced work values (for each ligand in each pair, for each phase, and for each trajectory launched), the last of which is the total reduced work performed *on* the trajectory over the protocol. Each `work` file has a default form of `lig{i}to{j}.{phase}.{old/new}.{direction}.idx_{snapshot_index}.{annealing_steps}.works.npz` where `i`, `j` are ligand index pairs, `phase` is either 'complex' or 'solvent', `old/new` denotes whether the work array corresponds to ligand `i` (old) or `j` (new), `direction` is 'forward' or 'backward', `snapshot_index` is which configuration index is annealed, and `annealing_steps` denotes the number of integration steps in the NEQ protocol. Forward and backward work distributions can be extracted, and the free energy correction of each phase can be computed (in kT) with the Bennett Acceptance Ratio (BAR) to find the maximum likelihood estimate of the free energy.

The `work` files can be queried and saved for each phase with:
```
import numpy as np
from qmlify.analysis import aggregate_pair_works, fully_aggregate_work_dict
work_pair_dictionary = aggregate_pair_works(ligand_indices, annealing_steps = {'complex': 5000, 'solvent':5000}) #this may take a few minutes
aggregated_work_dictionary, concatenated_work_dictionary = fully_aggregate_work_dict(work_pair_dictionary) #process the pair dictionary into something that is easier to analyze
np.savez('work_dictionaries.npz', aggregated_work_dictionary, concatenated_work_dictionary) #save into a compressed file
```
.See the function documentation for `kwargs` in `qmlify.analysis` for a more complete description. Once the `work_dictionary.npz` is generated, see `experiments/free_energy_corrections.ipynb`, which demonstrates how to query the work dictionary, calculate BAR free energy estimates, plot the work distributions per ligand, per phase, and perform MM-to-MM/ML free energy corrections (with plots). 

## Copyright

Copyright (c) 2020, Chodera Lab

## Authors
- dominic rufa
- Hannah E. Bruce Macdonald
- Josh Fass
- Marcus Wieder


## Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.

If you found this repository helpful, consider citing:
```
@article {Rufa2020.07.29.227959,
        author = {Rufa, Dominic A. and Bruce Macdonald, Hannah E. and Fass, Josh and Wieder, Marcus and Grinaway, Patrick B. and Roitberg, Adrian E. and Isayev, Olexandr and Chodera, John D.},
        title = {Towards chemical accuracy for alchemical free energy calculations with hybrid physics-based machine learning / molecular mechanics potentials},
        elocation-id = {2020.07.29.227959},
        year = {2020},
        doi = {10.1101/2020.07.29.227959},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2020/07/30/2020.07.29.227959},
        eprint = {https://www.biorxiv.org/content/early/2020/07/30/2020.07.29.227959.full.pdf},
        journal = {bioRxiv}
}
```
