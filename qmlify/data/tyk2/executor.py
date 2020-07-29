#!/usr/bin/env python

"""
see qmlify.executables for execution documentation;
NOTE : the code below cannot be run in directly from terminal. each function will bsub python executables but will not wait until completion to run the next python function. users must wait for completion to run each subsequent function; otherwise, functions will raise errors since it cannot query files that do not yet exist.

'forward' and 'backward' functions will write a `*works.npz` file (see documentation for a more thorough explanation) containing an np.ndarray of length (annealing_steps + 1) where the last entry is the final work done onto the system. `tyk2_works.ipynb` contains functionality to query the work files, extract works, and compute EXP/BAR free energy estimate for each phase for each ligand. 
"""


#this copies the appropriate perses_data and formats appropriately
from qmlify.executables import perses_extraction_admin
ligand_indices = [(0,12),
                  (0,7),
                  (10,13),
                  (10,6),
                  (11,0),
                  (11,14),
                  (14,2),
                  (14,8),
                  (14,9),
                  (15,14),
                  (15,4),
                  (1,13),
                  (1,6),
                  (2,7),
                  (3,0),
                  (3,13),
                  (4,12),
                  (4,13),
                  (4,14),
                  (4,5),
                  (4,9),
                  (5,0),
                  (6,0),
                  (8,0)]

perses_extraction_admin(ligand_indices,
                        '<location of perses data directory>',
                        '<location of local directory>', 
                        phases = ['solvent', 'complex'], 
                        delete_execution=False) 


#this anneals in the forward direction
from qmlify.executables import propagation_admin
import os
propagation_admin(ligand_indices, annealing_steps=5000, direction = 'forward', parent_dir = os.getcwd(), extraction_indices = range(0, 200, 2), phases = ['solvent', 'complex'], write_log=True, cat_outputs=False, delete_outputs=False)



#this conducts ani endstate equilibration
import os
propagation_admin(ligand_indices, annealing_steps=5000, direction = 'ani_endstate', parent_dir = os.getcwd(), extraction_indices = 100, phases = ['solvent', 'complex'], write_log=True, cat_outputs=True, delete_outputs=False, eq_steps=5000)



#this conducts the backward annealing
propagation_admin(ligand_indices, annealing_steps=5000, direction = 'backward', parent_dir = os.getcwd(), extraction_indices = 100, phases = ['solvent', 'complex'], write_log=True, cat_outputs=True, delete_outputs=False, eq_steps=5000)

