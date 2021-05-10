#!/usr/bin/env python
# coding: utf-8

# attempt 1

# In[1]:


from openmmtools.testsystems import HostGuestExplicit


# In[2]:


hge = HostGuestExplicit()


# In[3]:


system, positions, topology = hge.system, hge.positions, hge.topology


# In[4]:


from qmlify.openmm_torch.force_hybridization import HybridSystemFactory
from simtk import unit


# In[5]:


from openmmtools.testsystems import HostGuestExplicit
T = 300*unit.kelvin
system, positions, topology = hge.system, hge.positions, hge.topology
system.removeForce(system.getNumForces() - 1) # remove the CMMotionRemover force because it is unknown
_atoms = list(range(126,156)) #these atoms correspond to the guest. query these with the second residue in the topology


# In[6]:


system.getForces()


# In[7]:



hsf = HybridSystemFactory(topology = topology,
         alchemical_residue_indices = [1],
         system = system,
         softcore_alpha_sterics = 0.5,
         softcore_alpha_electrostatics = 0.5)
# grab the modified system and endstate system...
mod_system = hsf.system
endstate_system = hsf.endstate_system


# now that we have the modified system, we want to get the energy at _this_ endstate and make sure the energy is bookkeeping well with the non-alchemically-modified state.

# In[8]:


from openmmtools.integrators import LangevinIntegrator
from simtk import openmm


# In[9]:


nonalch_int = LangevinIntegrator(temperature=T)
alch_int = LangevinIntegrator(temperature=T)


# In[10]:


nonalch_context, alch_context = openmm.Context(system, nonalch_int), openmm.Context(mod_system, alch_int)


# In[11]:


for context in [nonalch_context, alch_context]:
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())


# In[12]:


nonalch_context.getState(getEnergy=True).getPotentialEnergy()


# In[13]:


alch_context.getState(getEnergy=True).getPotentialEnergy()


# we're only off by a thousandth of a kj/mol.

# if this is an artifact of the nonbonded term, we can safely ignore it.

# In[14]:


from qmlify.openmm_torch.utils import *


# In[15]:


from openmmtools.constants import kB


# In[16]:


beta = 1. / (T * kB)


# In[17]:


from openmmtools import utils


# In[18]:


platform = utils.get_fastest_platform()


# In[19]:


compute_potential_components(nonalch_context, beta, platform)


# In[20]:


compute_potential_components(alch_context, beta, platform)


# In[21]:


138.25363257328587 + 1.2408843919367571 - 139.49451696522257


# In[22]:


250.5430402178495 + 4.898738658713599 - 255.44177887656315


# In[23]:


29.64072359346019 + 127.8038989999598 - 157.44462259341995


# so it is nonbonded. can we write a function that pushed the alchemical context to the opposite endstate and asserts that all of the custom forces go to zero?

# first, let's gather the alchemical lambdas that must change...

# In[24]:


final_lambdas = {'lambda_MM_bonds' : 0.,
                 'lambda_MM_angles': 0.,
                 'lambda_MM_torsions': 0.,
                 'lambda_nonbonded_MM_sterics' : 1.,
                 'lambda_nonbonded_MM_electrostatics': 1.,
                 }


# In[25]:


for key, val in final_lambdas.items():
    alch_context.setParameter(key, val)


# In[26]:


compute_potential_components(alch_context, beta, platform)


# In[27]:


swig_params = alch_context.getParameters()


# In[28]:


for i in swig_params:
    print(i, swig_params[i])
    


# alright! now can we add the torchforce?

# In[29]:


from qmlify.openmm_torch.torchforce_generator import torch_alchemification_wrapper


# In[30]:


ml_system, hsf_mod = torch_alchemification_wrapper(topology, system, [1])


# In[ ]:




