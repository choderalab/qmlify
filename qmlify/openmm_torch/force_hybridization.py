#!/usr/bin/env python
import numpy as np
from simtk import unit, openmm

#######LOGGING#############################
import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("force-hybridization")
_logger.setLevel(logging.DEBUG)
###########################################

class HybridSystemFactory(object):
    """
    this class will take an existing system and make the following modifications:
    1. add a torch force that is alchemically scalable
    2. make the valence forces of the alchemical region a scalar so that we can toggle its strength
    3. make the intramolecular nonbonded forces of the alchemical region togglable exceptions and add appropriate parameter offsets
    4. add an (optional ability) to scale the parameters for rest-like enhanced sampling.

    NOTE : the `system` property is a deepcopy of the input `system`

    lambdas:
        lambda_name : default_value : description : scale
            lambda_MM_angles : 1.0 : scaling of MM angles
            lambda_MM_bonds : 1.0 : scaling of MM bonds
            lambda_MM_torsions : 1.0 : scaling of MM torsions
            lambda_nonbonded_MM_sterics : 1.0 : scaling of intra-region nonbonded steric terms
            lambda_nonbonded_MM_electrostatics : 1.0 : scaling of intra-region nonbonded electrostatic terms

            lambda_electrostatic_scale : 0.0 : scaling of nonbonded alch_residue-to-env_residue electrostatics
            lambda_epsilon_scale : 0.0 : scaling of nonbonded alch_residue-to-env_residue epsilon

            lambda_scale : 1.0 : multiplicative scaling of harmonic bonds/angles/periodic torsions/intra-region electrostatic/sterics


        an example of an N-window protocol (even) going from MM to ML (only alchemical region) with a fancy rest-like scaling goes like:
            n_replicas = 12  # Number of temperature replicas.
            T_min = 300.0 * unit.kelvin  # Minimum temperature.
            T_max = 800.0 * unit.kelvin  # Maximum temperature.
            temperatures = [T_min + (T_max - T_min) * (math.exp(float(i) / float(n_replicas-1)) - 1.0) / (math.e - 1.0)
                for i in range(N//2)]
            betas = [1. / (kB * T) for T in temperatures]
            beta0 = 1. / (kB * T_min)

            #NOTE : below, I'll scale the  MM torsions (and exceptions) differently since they will go to zero halfway through the protocol.


            lambda_MM_angles : np.linspace(1,0,N)
            lambda_MM_bonds : np.linspace(1,0,N)
            lambda_MM_torsions : np.linspace(1,0,N//2)

            lambda_electrostatic_scale : [np.sqrt(beta / beta0) - 1 for beta in betas]
            lambda_epsilon_scale : [beta / beta0 - 1 for beta in betas] #scaled slightly differently because of the ways epsilons are combined

            lambda_nonbonded_MM_sterics : np.linspace(1,0,N) #turn off intra-region sterics
            lambda_nonbonded_MM_electrostatics : np.linspace(1,0,N) # turn off intra-region electrostatics

            lambda_scale = [beta / beta0 for beta in betas]

            torch_scale : np.linspace(0,1,N) #this is a potential name
            auxiliary_torch_scale : [beta / beta0 for beta in betas]

    Example:
        >>> from openmmtools.testsystems import HostGuestExplicit
        >>> T = 300*unit.kelvin
        >>> system, positions, topology = hge.system, hge.positions, hge.topology
        >>> system.removeForce(system.getNumForces() - 1) # remove the CMMotionRemover force because it is unknown
        >>> _atoms = list(range(126:156)) #these atoms correspond to the guest. query these with the second residue in the topology
        >>> mod_system = hsf.system
        >>> endstate_system = hsf.endstate_system
        >>> hsf = HybridSystemFactory(topology = topology,
                 alchemical_residue_indices = [1],
                 system = system,
                 softcore_alpha_sterics = 0.5,
                 softcore_alpha_electrostatics = 0.5)
        >>> # grab the modified system and endstate system...
        >>> mod_system = hsf.system
        >>> endstate_system = hsf.endstate_system
    """
    _known_forces = {'HarmonicBondForce',
                     'HarmonicAngleForce',
                     'PeriodicTorsionForce',
                     'NonbondedForce',
                     'MonteCarloBarostat'}

    def __init__(self,
                 topology,
                 alchemical_residue_indices,
                 system,
                 softcore_alpha_sterics = 0.5,
                 softcore_alpha_electrostatics = 0.5,
                 **kwargs):
        """
        arguments
            topology : openmm.Topology
                topology which to modify
            alchemical_residue_indices : list(int)
                zero-indexed residues that will be treated alchemically...
        """
        import copy
        _logger.info(f"initializing system modifier...")

        self._topology = topology # set the whole topology attribute
        self._alchemical_residue_indices = alchemical_residue_indices #set the alchemical residue index
        self._old_system = system
        self._system = copy.deepcopy(system) #set the system as a deepcopy of the old system
        self._endstate_system = copy.deepcopy(system) # create a system at the opposite endstate for energy bookkeeping purposes

        #parameters for softcoring the alchemical nonbonded interactions
        self._softcore_alpha_sterics = softcore_alpha_sterics
        self._softcore_alpha_electrostatics = softcore_alpha_electrostatics

        #pull the appropriate atoms
        match_residues = [res for res in topology.residues() if res.index in self._alchemical_residue_indices]
        _logger.info(f"found {len(match_residues)} matches from {self._alchemical_residue_indices}")

        #define the atoms of the matched residue
        list_of_lists = [[atom.index for atom in match_residue.atoms()] for match_residue in match_residues]
        self._atoms = [item for sublist in list_of_lists for item in sublist]
        self._atoms_set = set(self._atoms)
        assert len(self._atoms) == len(self._atoms_set), f"there are repeated indices in `self._atoms`"
        _logger.info(f"identified {len(self._atoms)} to treat with ML.")

        #make a system forces object
        self._system_forces = {type(force).__name__ : force for force in self._system.getForces()}
        self._endstate_system_forces = {type(force).__name__ : force for force in self._endstate_system.getForces()}
        unknown_forces = [i for i in list(self._system_forces.keys()) if i not in self._known_forces]
        assert len(unknown_forces) == 0, f"there is at least one unknown force in the system: {unknown_forces}"


        _logger.info(f"modifying harmonic bonds...")
        self.modify_harmonic_bonds()

        _logger.info(f"modifying harmonic angles...")
        self.modify_harmonic_angles()

        _logger.info(f"modifying periodic torsions...")
        self.modify_periodic_torsions()

        _logger.info(f"modifying nonbondeds...")
        self.modify_nonbonded_force_v2()

    def is_in_alchemical_region(self,
                                particle_index_set):
        """
        return bool whether the given particle_index_set is a subset of the alchemical region

        arguments
            particle_index_set : set(int)
                set of particle indices

        returns
            is_subset : bool
                whether the particle set is a subset of the alchemical region

        NOTE : we do not currently support sets that straddle the alchemical/environment regions
        """
        is_subset = particle_index_set.issubset(self._atoms_set)
        if not is_subset:
            assert len(particle_index_set.union(self._atoms_set)) == len(self._atoms_set) + len(particle_index_set), f"this set straddles the alchemical region"
        return is_subset

    def modify_harmonic_bonds(self):
        """
        turn the harmonic bonds into a custom bond force

        lambda_protocol :
            - 'lambda_MM_bonds' : 1 -> 0
            - 'lambda_scale' : beta / beta0
        """
        self._alchemical_to_old_bonds = {}
        bond_expression = 'lambda_MM_bonds * lambda_scale * (k/2)*(r-r0)^2;'
        custom_bond_force = openmm.CustomBondForce(bond_expression)

        #add the global params
        custom_bond_force.addGlobalParameter('lambda_MM_bonds', 1.)
        custom_bond_force.addGlobalParameter('lambda_scale', 1.)

        #add the perbondparams
        custom_bond_force.addPerBondParameter('r0')
        custom_bond_force.addPerBondParameter('k')

        #now to iterate over the bonds.
        for idx in range(self._system_forces['HarmonicBondForce'].getNumBonds()):
            p1, p2, length, k = self._system_forces['HarmonicBondForce'].getBondParameters(idx)
            if self.is_in_alchemical_region({p1, p2}): #then this bond is in the transforming residue

                #first thing to do is to zero the force from the `HarmonicBondForce`
                self._system_forces['HarmonicBondForce'].setBondParameters(idx, p1, p2, length, k*0.0)
                self._endstate_system_forces['HarmonicBondForce'].setBondParameters(idx, p1, p2, length, k*0.0) #for bookkeeping

                #then add it to the custom bond force
                custom_bond_idx = custom_bond_force.addBond(p1, p2, [length, k])

                #add to the alchemical bonds dict for bookkeeping
                self._alchemical_to_old_bonds[custom_bond_idx] = idx

        #then add the custom bond force to the system

        if self._system_forces['HarmonicBondForce'].usesPeriodicBoundaryConditions():
            custom_bond_force.setUsesPeriodicBoundaryConditions(True)

        self._system.addForce(custom_bond_force)

    def modify_harmonic_angles(self):
        """
        turn the harmonic angles into a custom angle force

        lambda_protocol :
        - 'lambda_MM_angles' : 1 -> 0
        - 'lambda_scale' : beta / beta0
        """
        self._alchemical_to_old_angles = {}
        angle_expression = 'lambda_MM_angles * lambda_scale * (k/2)*(theta-theta0)^2;'
        custom_angle_force = openmm.CustomAngleForce(angle_expression)

        #add the global params
        custom_angle_force.addGlobalParameter('lambda_MM_angles', 1.)
        custom_angle_force.addGlobalParameter('lambda_scale', 1.)

        #add the perangleparams
        custom_angle_force.addPerAngleParameter('theta0')
        custom_angle_force.addPerAngleParameter('k')

        #now to iterate over the angles.
        for idx in range(self._system_forces['HarmonicAngleForce'].getNumAngles()):
            p1, p2, p3, theta0, k = self._system_forces['HarmonicAngleForce'].getAngleParameters(idx)
            if self.is_in_alchemical_region({p1,p2,p3}):
                #first thing to do is to zero the force from the `HarmonicAngleForce`
                self._system_forces['HarmonicAngleForce'].setAngleParameters(idx, p1, p2, p3, theta0, k*0.0)
                self._endstate_system_forces['HarmonicAngleForce'].setAngleParameters(idx, p1, p2, p3, theta0, k*0.0)

                #then add it to the custom angle force
                custom_angle_idx = custom_angle_force.addAngle(p1, p2, p3, [theta0, k])

                #add to the alchemical bonds dict for bookkeeping
                self._alchemical_to_old_angles[custom_angle_idx] = idx

        #then add the custom bond force to the system
        if self._system_forces['HarmonicAngleForce'].usesPeriodicBoundaryConditions():
            custom_angle_force.setUsesPeriodicBoundaryConditions(True)
        self._system.addForce(custom_angle_force)

    def modify_periodic_torsions(self):
        """
        turn the periodic torsions into a custom torsion force

        lambda_protocol :
        - 'lambda_MM_torsions' : 1 -> 0
        - 'lambda_scale' : beta / beta0
        """
        self._alchemical_to_old_torsions = {}
        torsion_expression = 'lambda_MM_torsions * lambda_scale * k * ( 1 + cos( periodicity * theta - phase))'
        custom_torsion_force = openmm.CustomTorsionForce(torsion_expression)

        #add the global params
        custom_torsion_force.addGlobalParameter('lambda_MM_torsions', 1.)
        custom_torsion_force.addGlobalParameter('lambda_scale', 1.)

        #add the pertorsion params
        custom_torsion_force.addPerTorsionParameter('periodicity')
        custom_torsion_force.addPerTorsionParameter('phase')
        custom_torsion_force.addPerTorsionParameter('k')

        #now to iterate over the torsions.
        for idx in range(self._system_forces['PeriodicTorsionForce'].getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = self._system_forces['PeriodicTorsionForce'].getTorsionParameters(idx)
            if self.is_in_alchemical_region({p1,p2,p3,p4}):
                #first thing to do is to zero the force from the `PeriodicTorsionForce`
                self._system_forces['PeriodicTorsionForce'].setTorsionParameters(idx, p1, p2, p3, p4, periodicity, phase, k*0.0)
                self._endstate_system_forces['PeriodicTorsionForce'].setTorsionParameters(idx, p1, p2, p3, p4, periodicity, phase, k*0.0)

                #then add it to the custom torsion force
                custom_torsion_idx = custom_torsion_force.addTorsion(p1, p2, p3, p4, [periodicity, phase, k])

                #add to the alchemical bonds dict for bookkeeping
                self._alchemical_to_old_torsions[custom_torsion_idx] = idx

        #then add the custom bond force to the system
        if self._system_forces['PeriodicTorsionForce'].usesPeriodicBoundaryConditions():
            custom_torsion_force.setUsesPeriodicBoundaryConditions(True)

        self._system.addForce(custom_torsion_force)

    def get_custom_bond_force(self):
        """
        make a custom bond force object
        """
        from openmmtools.constants import ONE_4PI_EPS0 # constant for coulomb (implicitly in md_unit_system units)

        custom_expression = "lambda_scale * (U_electrostatics + U_sterics);" #name the energy contributions
        custom_expression += "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;" #name sterics expression

        #custom_expression += "reff_sterics = r;"
        custom_expression += "reff_sterics = sigma*((softcore_alpha_sterics * lambda_nonbonded_MM_sterics + (r/sigma)^6))^(1/6);" # effective softcore distance for sterics
        custom_expression += "epsilon = (1 - lambda_nonbonded_MM_sterics) * epsilonA + lambda_nonbonded_MM_sterics * epsilonB;"
        custom_expression += "sigma = (1 - lambda_nonbonded_MM_sterics) * sigmaA + lambda_nonbonded_MM_sterics * sigmaB;"

        custom_expression += "U_electrostatics = chargeProd * ONE_4PI_EPS0 / reff_electrostatics;"
        custom_expression += f"ONE_4PI_EPS0 = {ONE_4PI_EPS0};"
        #custom_expression += f"reff_electrostatics = r;"
        custom_expression += "reff_electrostatics = ((softcore_alpha_electrostatics * lambda_nonbonded_MM_electrostatics + (r)^6))^(1/6);"
        custom_expression += "chargeProd = (1 - lambda_nonbonded_MM_electrostatics) * chargeProdA + lambda_nonbonded_MM_electrostatics * chargeProdB;"

        custom_expression += f"softcore_alpha_sterics = {self._softcore_alpha_sterics};"
        custom_expression += f"softcore_alpha_electrostatics = {self._softcore_alpha_electrostatics};"

        custom_bond_force = openmm.CustomBondForce(custom_expression)

        global_params = ['lambda_scale','lambda_nonbonded_MM_sterics', 'lambda_nonbonded_MM_electrostatics']
        per_bond_params = ['epsilonA', 'epsilonB', 'sigmaA', 'sigmaB', 'chargeProdA', 'chargeProdB']

        for global_param in global_params:
            if global_param == 'lambda_scale': # 'lambda_scale' is the exception; it defaults to 0.0
                custom_bond_force.addGlobalParameter(global_param, 1.)
            else:
                custom_bond_force.addGlobalParameter(global_param, 0.)

        for per_bond_param in per_bond_params:
            custom_bond_force.addPerBondParameter(per_bond_param)

        self._system.addForce(custom_bond_force)

        return custom_bond_force

    def add_nbf_inter_region_scaling(self):
        """
        modify the existing nonbonded force to handle inter-region REST scaling
        """
        nbf = self._system_forces['NonbondedForce'] #pull the nbf

        #third, iterate through all particles in alch region and add a parameter offset to allow for rest-like scaling
        self._particle_offsets_elec = []
        self._particle_offsets_eps = []
        _logger.debug(f"enabling rest-like scaling to alchemical particle nonbondeds")
        for atom in self._atoms:
            charge, sigma, eps = nbf.getParticleParameters(atom)
            offset_elec_idx = nbf.addParticleParameterOffset('lambda_electrostatic_scale', atom, charge, sigma * 0., eps * 0.)
            offset_eps_idx = nbf.addParticleParameterOffset('lambda_epsilon_scale', atom, charge * 0., sigma * 0., eps)
            self._particle_offsets_elec.append(offset_elec_idx)
            self._particle_offsets_eps.append(offset_eps_idx)


    def modify_nonbonded_force_v2(self):
        from itertools import combinations

        nbf = self._system_forces['NonbondedForce'] #pull the nbf

        custom_force = self.get_custom_bond_force() #get the custom bond force

        if nbf.usesPeriodicBoundaryConditions():
            custom_force.setUsesPeriodicBoundaryConditions(True)



        #add new global params

        #these are the particle parameter offsets for inter-region scaling (nonbonded only)
        nbf.addGlobalParameter('lambda_electrostatic_scale', 0.) # parameter offset for inter-region scaling
        nbf.addGlobalParameter('lambda_epsilon_scale', 0.) #parameter offset for inter-region scaling

        num_particles = nbf.getNumParticles()
        num_exceptions = nbf.getNumExceptions()

        #first, iterate through the existing exceptions in the alch region, make a list of the pairs for later, and remove the the exception so we can put it into the
        # custom bond force
        _logger.debug(f"iterating over existing exceptions and adding to custom force...")
        alch_exceptions_particles = []
        for idx in range(num_exceptions):
            p1, p2, chargeprod, sigma, eps = nbf.getExceptionParameters(idx)
            is_alch = self.is_in_alchemical_region({p1, p2})
            if is_alch: #this exception is in the appropriate exception region
                alch_exceptions_particles.append([p1, p2])
                if (chargeprod.value_in_unit_system(unit.md_unit_system) != 0.0 or eps.value_in_unit_system(unit.md_unit_system) != 0.0):
                    #then the term is not simply zeroed and must be handled
                    nbf.setExceptionParameters(idx, p1, p2, chargeprod * 0.0, sigma, eps * 0.0) #zero the exception in the nbf
                    self._endstate_system_forces['NonbondedForce'].setExceptionParameters(idx, p1, p2, chargeprod * 0.0, sigma, eps * 0.0) #zero the exception in the nbf
                    final_sigma = 1.0 * unit.angstrom if sigma.value_in_unit_system(unit.md_unit_system) <= 0. else sigma
                    custom_force.addBond(p1, p2, [eps, eps * 0.0, sigma, final_sigma, chargeprod, chargeprod * 0.0])


        #second, iterate through all pairs of particles in the alch region, if an exception doesn't exist, create a zeroed exception and add it to the custom bond force
        _logger.debug(f"iterating over alchemical particle combinations")
        alch_particle_combinations = [list(i) for i in combinations(self._atoms, 2)]
        self._auxiliary_exceptions = []
        self._auxiliary_custom_bonds = []
        for particle_pair in alch_particle_combinations:
            is_handled = True if (particle_pair in alch_exceptions_particles or particle_pair[::-1] in alch_exceptions_particles) else False
            p1, p2 = particle_pair
            if not is_handled:
                charge1, sigma1, eps1 = nbf.getParticleParameters(p1)
                charge2, sigma2, eps2 = nbf.getParticleParameters(p2)
                chargeprod, sigma, eps = charge1 * charge2, 0.5*(sigma1 + sigma2), np.sqrt(eps1.value_in_unit(unit.kilojoule_per_mole) * eps2.value_in_unit(unit.kilojoule_per_mole)) * unit.kilojoule_per_mole
                exception_idx = nbf.addException(p1,
                                                 p2,
                                                 chargeprod * 0.0,
                                                 sigma,
                                                 eps * 0.0
                                                )
                self._endstate_system_forces['NonbondedForce'].addException(p1,
                                                 p2,
                                                 chargeprod * 0.0,
                                                 sigma,
                                                 eps * 0.0
                                                )
                final_sigma = 1.0 * unit.angstrom if sigma.value_in_unit_system(unit.md_unit_system) <= 0. else sigma
                new_bond_idx = custom_force.addBond(p1, p2, [eps, eps * 0.0, sigma, final_sigma, chargeprod, chargeprod * 0.0])
                self._auxiliary_exceptions.append(exception_idx)
                self._auxiliary_custom_bonds.append(new_bond_idx)

        #handle scaling tethering for inter-region interactions
        self.add_nbf_inter_region_scaling()

    @property
    def system(self):
        return self._system

    @property
    def endstate_system(self):
        return self._endstate_system
