# qmlify
==============================
[//]: # (Badges)
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
as well as a couple of classes from [coddiwomple](https://github.com/choderalab/coddiwomple):
```
git clone https://github.com/choderalab/coddiwomple.git
cd coddiwomple
python setup.py install
```

## Use
`qmlify` postprocesses `perses` free energy calculations. You will need `perses` MM relative free energy calculation data before you can post-process it with `qmlify`

### Perses
You can execute `perses` relative free energy calculations on the Tyk2 congeneric inhibitors with `qmlify/data/tyk2/inputs/submit-ligpairs.sh` on your platform. Ligands `.sdf` files, system `.xml`s, and the protein `.pdb` are also found in `qmlify/data/tyk2/inputs`.

### qmlify
`qmlify` uses bidirectional nonequilibrium (NEQ) free energy calculations to anneal from the MM thermodynamic states (afforded by `perses`) to the hybrid ML/MM thermodynamics states (and back).
Once `perses` free energy calculations are complete, you can extract and reformat the necessary `perses` files using:
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
These commands will generate `.npz` files containing a `numpy` array of *cumulative* reduced work values (for each ligand in each pair, for each phase, and for each trajectory launched), the last of which is the total reduced work performed *on* the trajectory over the protocol.  Forward and backward work distributions can be extracted, and the free energy correction of each phase can be computed (in kT) with the Bennett Acceptance Ratio (BAR) to find the maximum likelihood estimate of the free energy.
For example, to compute the free energy correction of ligand `0` in the `lig0to12` transformation,

## Copyright

Copyright (c) 2020, Chodera Lab

### Authors
- dominic rufa
- Hannah E. Bruce Macdonald
- Josh Fass
- Marcus Wieder


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.

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
