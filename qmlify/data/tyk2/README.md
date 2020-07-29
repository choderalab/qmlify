This directory contains input data, executables, and outputs for the Tyk2 ML/MM binding free energy correction.

#Perses
Input data and submission scripts for perses MM relative free energy calculations are located in 'inputs' directory.

#qmlify
script to execute ML/MM (ANI-2x) correction is `executor.py`. 
forward' and 'backward' functions will write a `*works.npz` file (see documentation for a more thorough explanation) containing an np.ndarray of length (annealing_steps + 1) where the last entry is the final work done onto the system. `tyk2_works.ipynb` contains functionality to query the work files, extract works, and compute EXP/BAR free energy estimate for each phase for each ligand.
`positions.npz` files are also written for the last configuration in the 'forward', 'backward' and 'ani_endstate' directions. the torsion plots can be extracted via `torsions.py`

