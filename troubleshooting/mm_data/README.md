Tyk2 replica exchange relative free energy calculations from [perses](https://github.com/choderalab/perses) live here. each `.pi` pickle can be loaded and queried as follows:
```
import pickle
pickle_file = "simulation.pi"

with open(pickle_file,'rb') as f: simulation = pickle.load(f)
calc_DDG = -sim.bindingdg #relative free energy
calc_dDDG = sim.bindingddg #relative uncertainty of free energy
```
