#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qmlify.openmm_torch.utils import prepare_ml_system


# In[2]:


from qmlify.openmm_torch.utils import prepare_ml_system
from qmlify.openmm_torch.test_openmm_torch import get_HostGuestExplicit
import os
testsystem_class = get_HostGuestExplicit()


# In[3]:


torch_scale_name = 'torch_scale'
torch_scale_default_value = 0.


# In[4]:


ml_system, hybrid_factory = prepare_ml_system(
                          positions = testsystem_class.positions,
                          topology = testsystem_class.topology,
                          system = testsystem_class.system,
                          residue_indices = [1],
                          model_name='ani2x',
                          save_filename = 'repex.pt',
                          torch_scale_name=torch_scale_name,
                          torch_scale_default_value=torch_scale_default_value,
                          HybridSystemFactory_kwargs = {},
                          minimizer_kwargs = {'maxIterations': 1000}
                          )


# In[ ]:





# In[5]:


from qmlify.openmm_torch.utils import configure_platform
from openmmtools import utils
from openmmtools.integrators import LangevinIntegrator
import simtk


# In[6]:


from qmlify.openmm_torch.utils import *


# In[7]:


nonalch_system = testsystem_class.system


# In[8]:


nonalch_int = LangevinIntegrator()
ml_int = LangevinIntegrator(splitting= 'V0 V1 R O R V1 V0')


# In[9]:


platform = configure_platform(utils.get_fastest_platform().getName())


# In[10]:


nonalch_context = openmm.Context(nonalch_system, nonalch_int, platform)
ml_context = openmm.Context(ml_system, ml_int, platform)


# In[13]:


for context in [nonalch_context, ml_context]:
    context.setPositions(testsystem_class.positions)
    context.setPeriodicBoxVectors(*testsystem_class.system.getDefaultPeriodicBoxVectors())


# In[14]:


from time import time


# In[15]:


nonalch_times = []
ml_times = []

for i in range(100):
    timer1 = time()
    nonalch_int.step(1)
    nonalch_times.append(time() - timer1)

    timer2 = time()
    ml_int.step(1)
    ml_times.append(time() - timer2)




# In[18]:


import numpy as np


# In[20]:


nonalch_mean = np.mean(nonalch_times[1:])


# In[21]:


ml_mean = np.mean(ml_times[1:])


# In[22]:


print(f"nonalch_mean: {nonalch_mean}")
print(f"ml_mean: {ml_mean}")


# In[24]:


print(f"the ml context runs {100 * nonalch_mean / ml_mean}% slower than the mm context")


# In[ ]:
