#!/usr/bin/env python
# coding: utf-8

#####Imports#####
from openmmtools.states import ThermodynamicState, SamplerState
from simtk import unit
from copy import deepcopy
import os
import sys
import numpy as np
import pickle
import tqdm
import mdtraj.utils as mdtrajutils
import mdtraj as md
import torch
import torchani
from openmmtools.constants import kB
from openmmtools.utils import TrackedQuantity
from coddiwomple.openmm.integrators import OMMLI #OpenMMLangevinIntegrator
from coddiwomple.openmm.propagators import OMMBIP #OpenMMBaseIntegratorPropagator
from coddiwomple.openmm.reporters import OpenMMReporter #trajectory reporter object

from openmmtools import utils
# from perses.dispersed.utils import check_platform, configure_platform
# cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())
atomic_num_to_symbol_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 17: 'Cl', 9: 'F', 16: 'S'}
mass_dict_in_daltons = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}
_allowable_quantities = [unit.quantity.Quantity, TrackedQuantity]

import logging
from copy import deepcopy

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("hybrid_propagators")
_logger.setLevel(logging.DEBUG)


# In[ ]:


class ANI_force_and_energy(object):
    # some class attributes
    mass_unit = unit.dalton
    distance_unit = unit.nanometers
    time_unit = unit.femtoseconds
    energy_unit = unit.kilojoules_per_mole
    ani_distance_unit = unit.angstroms
    hartree_to_kJ_per_mole = 2625.499638
    ani_energy_unit = hartree_to_kJ_per_mole * energy_unit
    nm_to_angstroms = 10.
    angstroms_to_nm = 1e-1

    def __init__(self,
                 model,
                 atoms,
                 platform='cpu',
                 temperature=300 * unit.kelvin,
                 **kwargs
                 ):

        """
        Performs energy and force calculations.
        refactored code from:
            https://gist.github.com/wiederm/7ac5c29e5a0dea9d17ef16dda93fe02d#file-reweighting-py-L42

        Parameters
        ----------
        model: torchani.models object
            model from which to compute energies and forces
        atoms: str
            a string of atoms in the indexed order
        platform : str, default 'cpu',
            platform on which to initialize the model device
        temperature : float * unit.kelvin, default 300 * unit.kelvin
            temperature
        """
        self.model = model
        self.atoms = atoms

        self.platform = platform
        self.device = torch.device(self.platform)
        if self.platform == 'cpu':
            torch.set_num_threads(2)
        else:
            raise Exception(f"we don't support gpu just yet")

        self.species = self.model.species_to_tensor(atoms).to(self.device).unsqueeze(0)
        self.temperature = temperature
        self.beta = 1.0 / (kB * temperature)

        self.W_shads = []
        self.W = []

    def calculate_force(self,
                        x: unit.quantity.Quantity) -> (unit.quantity.Quantity, unit.quantity.Quantity):
        """
        Given a coordinate set the forces with respect to the coordinates are calculated.

        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration

        Returns
        -------
        F : float, unit'd
        E : float, unit'd
        """
        assert type(x) in _allowable_quantities

        coordinates = torch.tensor([x.value_in_unit(unit.angstroms)],
                                   requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_hartree = self._reform_as_energy_tensor(coordinates)

        # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad((energy_in_hartree).sum(), coordinates)[0]

        if self.platform == 'cpu':
            F = -1 * derivative[0].numpy()
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())[0]
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return (F * self.hartree_to_kJ_per_mole * (unit.kilojoule_per_mole / unit.angstrom),
                energy_in_hartree.item() * self.hartree_to_kJ_per_mole * unit.kilojoule_per_mole)

    def _reform_as_energy_tensor(self, coordinates: torch.tensor):
        """
        Helpter function to return energies as tensor.
        Given a coordinate set the energy is calculated.

        Parameters
        ----------
        coordinates : torch.tensor
            coordinates in angstroms without units attached

        Returns
        -------
        energy_in_hartree : torch.tensor

        """
        energy_in_hartree = self.model((self.species, coordinates)).energies

        return energy_in_hartree


    def calculate_energy(self, x: unit.Quantity):
        """
        Given a coordinate set (x) the energy is calculated in kJ/mol.
        Parameters
        ----------
        x : array of floats, unit'd (angstroms)
            initial configuration

        Returns
        -------
        energy : unit.quantity.Quantity
            energy in kJ/mol
        """

        assert type(x) in _allowable_quantities
        coordinates = torch.tensor([x.value_in_unit(unit.angstroms)],
                                   requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_hartrees = self._reform_as_energy_tensor(coordinates)
        energy = energy_in_hartrees.item() * self.hartree_to_kJ_per_mole * unit.kilojoule_per_mole
        return energy


class Integrator(OMMLI):
    def __init__(self,
                 temperature=300.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=1.0 * unit.femtoseconds,
                 splitting="V R O R F",
                 constraint_tolerance=1e-6,
                 **kwargs):
        """Create a Langevin integrator with the prescribed operator splitting.

        arguments
            splitting : string, default: "V R O R F"
                Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", "F" ...) substeps to be executed each timestep.
                Externally modified Forces (i.e. 'modified_f') are only used in V-step. Handle multiple force groups by appending the force group index
                to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
                Force group splitting is disabled.  "F" updates a perDofVariable 'modified_f' from the integrator 'f'
            temperature : np.unit.Quantity compatible with kelvin, default: 300.0*unit.kelvin
               Fictitious "bath" temperature
            collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 1.0/unit.picoseconds
               Collision rate
            timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
               Integration timestep
            constraint_tolerance : float, default: 1.0e-6
                Tolerance for constraint solver
        """
        #just super our previous method
        super().__init__(temperature,
                         collision_rate,
                         timestep,
                         splitting,
                         constraint_tolerance,
                         **kwargs)


    def _add_V_step(self, force_group="0"):
        """Deterministic velocity update, using only forces from force-group fg.

        arguments
            force_group : str, optional, default="0"
               Force group to use for this step
        """
        self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        if self._mts:
            self.addComputePerDof("v", "v + ((dt / {}) * modified_force / m)".format(self._force_group_nV[force_group], force_group))
        else:
            self.addComputePerDof("v", "v + (dt / {}) * modified_force / m".format(self._force_group_nV["0"]))

        self.addConstrainVelocities()


        self.addComputeSum("new_ke", self._kinetic_energy)
        self.addComputeGlobal("shadow_work", "shadow_work + (new_ke - old_ke)")

    def _add_F_step(self):
        """
        pull the mm force as a perDofVariable from the integrator to be modified
        """
        self.addComputePerDof('modified_force', 'f')


    def _add_variables(self):
        super()._add_variables()
        self.addPerDofVariable('modified_force', 0)

    def _add_integrator_steps(self):
        """
        Add the steps to the integrator--this can be overridden to place steps around the integration.
        """
        super()._add_integrator_steps()
        #self.addUpdateContextState()

    @property
    def _step_dispatch_table(self):
        dispatch_table = super()._step_dispatch_table
        dispatch_table['F'] = (self._add_F_step, False) #add a modified_force variable
        return dispatch_table


class Propagator(OMMBIP):
    """
    Propagator pseudocode:
    Step 1: initialization--
        set iteration = 0, n_iterations = n_iterations, lambda  = 0 (i.e. iteration / n_iterations); work_accumulated = 0.0
        generate sample x_0 ~ e^(-p(x))
        evaluate work_incremental = 0 (i.e. u_mm(x_0) - g(x_0), but we presume that g = u_mm(.))
        work_accumulated <- work_accumulated + work_incremental
        x' = x_0
    Step 2: sampling
        for increment in range(n_iterations):
            x = x'
            ante_perturbation_potential =  (1 - lambda) * u_mm(x) + lambda * u_ani_mm_mix(x)
            set iteration <- iteration + 1.0; lambda <- iteration / n_iterations
            evaluate work_incremental = [(1 - lambda) * u_mm(x) + lambda * u_ani_mm_mix(x)] - ante_perturbation_potential
            work_accumulated <- work_accumulated + work_incremental
            create a modified force: modified_f = (1 - lambda) * f_mm + lambda * f_ani_mm_mix (where f_. = -grad(u_.) )
            x' =  V R O R (where V deterministic update is according to modified_f defined above) w.r.t x

    NOTE: in this regime, the last x' is propagated w.r.t. a propagator whose invariant distribution respects u_ani_mm_mix;
    this should _not_ be the case.  There should be an exception in the Step 2 for loop that breaks once the final work_incremental is computed and updated
    to the work_accumulated. Regardless, the distribution of accumulated works is unaffected by this 'bug'; only expectations (as a function of x) w.r.t. these
    weights may be affected.

    See: 3.1.1. of https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf (esp. Remark 1.)




    """
    def __init__(self,
                 openmm_pdf_state,
                 openmm_pdf_state_subset,
                 subset_indices_map,
                 integrator,
                 ani_handler,
                 context_cache=None,
                 reassign_velocities=True,
                 n_restart_attempts=0,
                 reporter=None,
                 write_trajectory_interval = 1,
                 **kwargs):
        """
        arguments
            openmm_pdf_state : openmmtools.states.ThermodynamicState
                the pdf state of the propagator
            openmm_pdf_state_subset : openmmtools.states.ThermodynamicState
                the pdf state of the atom subset
            subset_indices_map : dict
                dict of {openmm_pdf_state atom_index : openmm_pdf_state_subset atom index}
            integrator : openmm.Integrator
                integrator of dynamics
            ani_handler : ANI_force_and_energy
                handler for ani forces and potential energy
            context_cache : openmmtools.cache.ContextCache, optional default:None
                The ContextCache to use for Context creation. If None, the global cache
                openmmtools.cache.global_context_cache is used.
            reassign_velocities : bool, optional default:False
                If True, the velocities will be reassigned from the Maxwell-Boltzmann
                distribution at the beginning of the move.
            n_restart_attempts : int, optional default:0
                When greater than 0, if after the integration there are NaNs in energies,
                the move will restart. When the integrator has a random component, this
                may help recovering. On the last attempt, the ``Context`` is
                re-initialized in a slower process, but better than the simulation
                crashing. An IntegratorMoveError is raised after the given number of
                attempts if there are still NaNs.
            reporter : coddiwomple.openmm.reporter.OpenMMReporter, default None
                a reporter object to write trajectories
            write_trajectory_interval : int
                frequency of writing trajectory
        """
        super().__init__(openmm_pdf_state,
                 integrator,
                 context_cache,
                 reassign_velocities,
                 n_restart_attempts)

        #create a pdf state for the subset indices (usually a vacuum system)
        self.pdf_state_subset = openmm_pdf_state_subset
        assert self.pdf_state_subset.temperature == self.pdf_state.temperature, f"the temperatures of the pdf states do not match"

        #create a dictionary for subset indices
        self._subset_indices_map = subset_indices_map

        #create an ani handler attribute that can be referenced
        self.ani_handler = ani_handler

        #create a context for the subset atoms that can be referenced
        self.context_subset, _ = self._context_cache.get_context(self.pdf_state_subset)

        #create a reporter for the accumulated works
        self._state_works = {}
        self._state_works_counter = 0

        #create a reporter
        self._write_trajectory = False if reporter is None else True
        self.reporter=reporter
        if self._write_trajectory:
            from coddiwomple.particles import Particle
            self.particle = Particle(0)
            self.write_trajectory_interval=write_trajectory_interval
        else:
            self.particle = None
            self.write_trajectory_interval=None

    def _initialize_state_works(self):
        """
        initialize an empty list and add 0.0 to it (state works)
        """
        self._current_state_works = [] #define an interim (auxiliary) list that will track the thermodynamic work of the current application
        self._current_state_works.append(0.0) #the first incremental work is always 0 since the importance function is identical to the first target distribution (i.e. fully interacting MM)

    def _initialize_iterations(self, n_iterations):
        """
        initialize the iteration counter
        """
        self._iteration = 0.0 #define the first iteration as 0
        self._n_iterations = n_iterations #the number of iterations in the protocol is equal to the number of steps in the application

    def _update_particle_state_substate(self, particle_state, new_state_subset=False):
        """
        update the particle state from the context, create a particle substate and update from context
        """
        #update the particle state and the particle state subset
        particle_state.update_from_context(self.context, ignore_velocities=True) #update the particle state from the context
        if new_state_subset:
            self.particle_state_subset = SamplerState(positions = particle_state.positions[list(self._subset_indices_map.keys())]) #create a particle state from the subset context
        else:
            self.particle_state_subset.positions = particle_state.positions[list(self._subset_indices_map.keys())] #update the particle subset positions appropriately
        self.particle_state_subset.apply_to_context(self.context_subset, ignore_velocities=True) #apply the subset particle state to its context
        self.particle_state_subset.update_from_context(self.context_subset, ignore_velocities=True) #update the subset particle state from its context to updated the potential energy

    def _update_current_state_works(self, particle_state):
        """
        update the current state and associated works
        """
        #get the reduced potential
        reduced_potential = self._compute_hybrid_potential(_lambda = self._iteration / self._n_iterations, particle_state = particle_state)
        perturbed_reduced_potential = self._compute_hybrid_potential(_lambda = (self._iteration + 1.0) / self._n_iterations, particle_state = particle_state)
        self._current_state_works.append(self._current_state_works[-1] + (perturbed_reduced_potential - reduced_potential))

    def _update_force(self, particle_state):
        """
        update the force
        """
        mm_force_matrix = self._compute_hybrid_forces(_lambda = (self._iteration + 1.0) / self._n_iterations, particle_state = particle_state).value_in_unit_system(unit.md_unit_system)
        self.integrator.setPerDofVariableByName('modified_force', mm_force_matrix)



    def _before_integration(self, *args, **kwargs):
        particle_state = args[0] #define the particle state
        n_iterations = args[1] #define the number of iterations

        self._initialize_state_works()
        self._initialize_iterations(n_iterations)

        #update the particle state and the particle state subset
        self._update_particle_state_substate(particle_state, new_state_subset=True)

        self._update_current_state_works(particle_state)

        self._update_force(particle_state)

        #report
        if self._write_trajectory: # the first state is always saved for processing purposes
            self.particle.update_state(particle_state)
            self.reporter.record([self.particle])


    def _during_integration(self, *args, **kwargs):
        particle_state = args[0]
        self._iteration += 1.0

        self._update_particle_state_substate(particle_state)

        #get the reduced potential
        if self._iteration < self._n_iterations:
            self._update_current_state_works(particle_state)
            self._update_force(particle_state)
        else:
            #we are done
            pass

        if self._write_trajectory and int(self._iteration) % self.write_trajectory_interval == 0:
            self.particle.update_state(particle_state)
            if self._iteration == self._n_iterations:
                self.reporter.record([self.particle], save_to_disk=True)
            else:
                self.reporter.record([self.particle], save_to_disk=False)



    def _after_integration(self, *args, **kwargs):
        self._state_works[self._state_works_counter] = deepcopy(self._current_state_works)
        self._state_works_counter += 1

        if self._write_trajectory:
            self.reporter.reset()
        #self._log_context_parameters()


    def _compute_hybrid_potential(self,_lambda, particle_state):
        """
        function to compute the hybrid reduced potential defined as follows:
        U(x_rec, x_lig) = u_mm,rec(x_rec) - lambda*u_mm,lig(x_lig) + lambda*u_ani,lig(x_lig)
        """
        reduced_potential = (self.pdf_state.reduced_potential(particle_state)
                             - _lambda * self.pdf_state_subset.reduced_potential(self.particle_state_subset)
                             + _lambda * self.ani_handler.calculate_energy(self.particle_state_subset.positions) * self.pdf_state.beta)
        return reduced_potential

    def _compute_hybrid_forces(self, _lambda, particle_state):
        """
        function to compute a hybrid force matrix of shape num_particles x 3
        in the spirit of the _compute_hybrid_potential, we compute the forces in the following way
            F(x_rec, x_lig) = F_mm(x_rec, x_lig) - lambda * F_mm(x_lig) + lambda * F_ani(x_lig)
        """
        # get the complex mm forces
        state = self.context.getState(getForces=True)
        mm_force_matrix = state.getForces(asNumpy=True) # returns forces in kJ/(nm mol)

        # get the ligand mm forces
        subset_state = self.context_subset.getState(getForces=True)
        mm_force_matrix_subset = subset_state.getForces(asNumpy=True)

        # get the ligand ani forces
        coords = self.particle_state_subset.positions
        subset_ani_force_matrix, energie = self.ani_handler.calculate_force(coords) # returns force in kJ/(A mol)
        #print(f"ani force matrix head: ",subset_ani_force_matrix[0])

        # now combine the ligand forces
        subset_force_matrix = _lambda * (subset_ani_force_matrix - mm_force_matrix_subset) #we are adding two Quantities with different units, but they are compatible
        #print(f"mm subset force matrix head", mm_force_matrix_subset[0])

        # and append to the complex forces...
        #print(f"mm force matrix head", mm_force_matrix[0])
        mm_force_matrix[list(self._subset_indices_map.keys()), :] += subset_force_matrix #and same, here...
        #print(f"mm force matrix head (after ani modification)", mm_force_matrix[0])

        return mm_force_matrix

    def _get_context_subset_parameters(self):
        """
        return a dictionary of the self.context_subset's parameters

        returns
            context_parameters : dict
            {parameter name <str> : parameter value value <float>}
        """
        swig_parameters = self.context_subset.getParameters()
        context_parameters = {q: swig_parameters[q] for q in swig_parameters}
        return context_parameters

    def _log_context_parameters(self):
        """
        log the context and context subset parameters
        """
        context_parameters = self._get_context_parameters()
        context_subset_parameters = self._get_context_subset_parameters()
        _logger.debug(f"\tcontext_parameters during integration:")
        for key, val in context_parameters.items():
            _logger.debug(f"\t\t{key}: {val}")

        _logger.debug(f"\tcontext subset parameters during integration:")
        for key, val in context_subset_parameters:
            _logger.debug(f"\t\t{key}: {val}")

    @property
    def state_works(self):
        return self._state_works

class BackwardPropagator(Propagator):
    """
    run the Propagator in reverse
    """
    def __init__(self,
                     openmm_pdf_state,
                     openmm_pdf_state_subset,
                     subset_indices_map,
                     integrator,
                     ani_handler,
                     context_cache=None,
                     reassign_velocities=True,
                     n_restart_attempts=0,
                     reporter=None,
                     write_trajectory_interval = 1,
                     **kwargs):
        super().__init__(
                         openmm_pdf_state,
                         openmm_pdf_state_subset,
                         subset_indices_map,
                         integrator,
                         ani_handler,
                         context_cache=context_cache,
                         reassign_velocities=reassign_velocities,
                         n_restart_attempts=n_restart_attempts,
                         reporter=reporter,
                         write_trajectory_interval = write_trajectory_interval,
                         **kwargs)

    def _update_current_state_works(self, particle_state):
        """
        update the current state and associated works
        """
        #get the reduced potential
        reduced_potential = self._compute_hybrid_potential(_lambda = 1.0 - (self._iteration / self._n_iterations), particle_state = particle_state)
        perturbed_reduced_potential = self._compute_hybrid_potential(_lambda = 1.0 - ((self._iteration + 1.0) / self._n_iterations), particle_state = particle_state)
        self._current_state_works.append(self._current_state_works[-1] + (perturbed_reduced_potential - reduced_potential))

    def _update_force(self, particle_state):
        """
        update the force
        """
        mm_force_matrix = self._compute_hybrid_forces(_lambda = 1.0 - ((self._iteration + 1.0) / self._n_iterations), particle_state = particle_state).value_in_unit_system(unit.md_unit_system)
        self.integrator.setPerDofVariableByName('modified_force', mm_force_matrix)

class ANIPropagator(Propagator):
    """
    run molecular dynamics at the ANI endstate; _current_state_works now are just the timeseries reduced potential...
    """
    def __init__(self,
                     openmm_pdf_state,
                     openmm_pdf_state_subset,
                     subset_indices_map,
                     integrator,
                     ani_handler,
                     context_cache=None,
                     reassign_velocities=True,
                     n_restart_attempts=0,
                     reporter=None,
                     write_trajectory_interval = 1,
                     **kwargs):
        super().__init__(
                         openmm_pdf_state,
                         openmm_pdf_state_subset,
                         subset_indices_map,
                         integrator,
                         ani_handler,
                         context_cache=context_cache,
                         reassign_velocities=reassign_velocities,
                         n_restart_attempts=n_restart_attempts,
                         reporter=reporter,
                         write_trajectory_interval = write_trajectory_interval,
                         **kwargs)

    def _update_current_state_works(self, particle_state):
        """
        update the current state and associated works
        """
        #get the reduced potential
        reduced_potential = self._compute_hybrid_potential(_lambda = 1.0, particle_state = particle_state)
        self._current_state_works.append(reduced_potential)

    def _update_force(self, particle_state):
        """
        update the force
        """
        mm_force_matrix = self._compute_hybrid_forces(_lambda = 1.0, particle_state = particle_state).value_in_unit_system(unit.md_unit_system)
        self.integrator.setPerDofVariableByName('modified_force', mm_force_matrix)

class RoundTripPropagator(Propagator):
    """
    run molecular dynamics in a round trip fashion, going from state A to B to A where equal time is spend on each route
    """
    def __init__(self,
                     openmm_pdf_state,
                     openmm_pdf_state_subset,
                     subset_indices_map,
                     integrator,
                     ani_handler,
                     context_cache=None,
                     reassign_velocities=True,
                     n_restart_attempts=0,
                     reporter=None,
                     write_trajectory_interval = 1,
                     **kwargs):
        super().__init__(
                         openmm_pdf_state,
                         openmm_pdf_state_subset,
                         subset_indices_map,
                         integrator,
                         ani_handler,
                         context_cache=context_cache,
                         reassign_velocities=reassign_velocities,
                         n_restart_attempts=n_restart_attempts,
                         reporter=reporter,
                         write_trajectory_interval = write_trajectory_interval,
                         **kwargs)

        def lambda_function(iter, num_iters):
            if iter <= num_iters/2.:
                return (2.*iter) / num_iters
            elif iter > num_iters/2:
                return (-2.*iter/num_iters) + 2

        self.lambda_function = lambda_function

    def _update_current_state_works(self, particle_state):
        """
        update the current state and associated works
        """
        #get the reduced potential
        reduced_potential = self._compute_hybrid_potential(_lambda = self.lambda_function(self._iteration, self._n_iterations), particle_state = particle_state)
        perturbed_reduced_potential = self._compute_hybrid_potential(_lambda = self.lambda_function(self._iteration + 1.0, self._n_iterations), particle_state = particle_state)
        self._current_state_works.append(self._current_state_works[-1] + (perturbed_reduced_potential - reduced_potential))

    def _update_force(self, particle_state):
        """
        update the force
        """
        mm_force_matrix = self._compute_hybrid_forces(_lambda = self.lambda_function(self._iteration + 1.0, self._n_iterations), particle_state = particle_state).value_in_unit_system(unit.md_unit_system)
        self.integrator.setPerDofVariableByName('modified_force', mm_force_matrix)
