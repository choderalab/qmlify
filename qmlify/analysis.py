"""
analysis tools for checking MM -> ML/MM absolute and relative free energies
"""

import numpy as np
from beryllium import plotting, stats
import networkx as nx
from simtk import unit
from openmmtools.constants import kB
import copy
temperature = 300.0 * unit.kelvin
kT = kB * temperature


def analyze_mlmm(mm_results,
                 ml_corrections,
                 experimental=None,
                 experimental_error=0.3,
                 MM_ff='openff-1.0.0',
                 ML_ff='ANI-2x'):
    """ Runs  MM-> ML/MM analysis, and generates four graphs:
    - MM relative
    - MM absolute
    - ML/MM relative
    - ML/MM absolute

    >>>  analysis.analyze_mlmm(all_sims, ani_results, experimental=experimental, experimental_error=0.18)

    Parameters
    ----------
    mm_results : list(perses.analysis.load_simulations.Simulation)
        List of perses Simulation objects of MM calculations
    ml_corrections : dict(int:dict(phase:tuple))
        MM -> ML/MM corrections and associated uncertainty
        i.e. {0:'solvent': (0.2, 0.1)} if ligand index 0 has a correction of 0.2 (0.1) kT in solvent phase
    experimental : list(float), default=None
        list of experimental binding free energies in kcal/mol
        if None function will return
    experimental_error: float or list of floats, default=0.3
        experimental uncertainty, either one value for the whole ligand set,
        or a list per ligand in kcal/mol
        Default value of 0.3 from
    MM_ff : str, default='openff-1.0.0'
        string name of MM ff for labelling plots
    ML_ff : str, default='ANI-2x'
        string name of ML ff for labelling plots

    Returns
    -------

    """

    if isinstance(experimental_error, float):
        experimental_error = len(experimental) * [experimental_error]

    assert len(experimental) == len(experimental_error), \
        f"Need either one experimental uncertainty for ligand set, or an experimental_error per ligand. {len(experimental)} experimental DGs, with {len(experimental_error)} dDGs."

    def _make_mm_graph(mm_results, expt, d_expt):
        """ Make a networkx graph from some MM results

        Parameters
        ----------
        mm_results : list(perses.analysis.load_simulations.Simulation)
            List of perses simulation objects
        expt : list
            List of experimental values, in kcal/mol
        d_expt : list
            List of uncertainties in experimental values, in kcal/mol

        Returns
        -------
        nx.DiGraph
            Graph object with relative MM free energies

        """
        mm_g = nx.DiGraph()
        for sim in mm_results:
            ligA = int(sim.directory[3:].split('to')[0])
            ligB = int(sim.directory[3:].split('to')[1])
            exp_err = (d_expt[ligA]**2 + d_expt[ligB]**2)**0.5
            mm_g.add_edge(ligA,
                          ligB,
                          calc_DDG=-sim.bindingdg/sim.bindingdg.unit,
                          calc_dDDG=sim.bindingddg/sim.bindingddg.unit,
                          exp_DDG=(expt[ligB] - expt[ligA]),
                          exp_err=exp_err)
        _absolute_from_relative(mm_g, expt, d_expt)
        return mm_g

    def _make_ml_graph(mm_g, corrections):
        """ Create a ML/MM graph, using an MM graph, and corrections to that.

        Parameters
        ----------
        mm_g : nx.DiGraph()
            Graph object with MM results
        corrections : dict(int:(float,float))
            Stored per-ligand corrections in the format ligand_ID:(correction, uncertainty)

        Returns
        -------
        nx.DiGraph
            Graph object with relative MM free energies

        """
        ml_g = copy.deepcopy(mm_g)
        for node in ml_g.nodes(data=True):
            node[1]['calc_DG'] = node[1]['calc_DG'] + corrections[node[0]][0]
            node[1]['calc_dDG'] = (node[1]['calc_dDG']**2 + corrections[node[0]][1]**2)**0.5
        return ml_g

    def _per_ligand_correction(ml_corrections):
        """Take the output corrections for both complex and solvent phases and turn into a single, per-ligand correction

        Parameters
        ----------
        ml_corrections : dict(int:dict)
            ML/MM corrections from qmlify

        Returns
        -------
        dict
            dictionary of format ligand_ID:(correction, uncertainty)

        """
        binding_corrections = {}
        for ligand in ml_corrections.keys():
            binding_corr = ((ml_corrections[ligand]['complex'][0] - ml_corrections[ligand]['solvent'][0])*kT).value_in_unit(unit.kilocalorie_per_mole)
            binding_corr_err = ((ml_corrections[ligand]['complex'][1]**2 + ml_corrections[ligand]['solvent'][1]**2)**0.5*kT).value_in_unit(unit.kilocalorie_per_mole)
            binding_corrections[ligand] = (binding_corr, binding_corr_err)
        return binding_corrections

    def _relative_corrections(corrections):
        """Shift the MM->ML/MM corrections to have a mean of zero.

        Parameters
        ----------
        corrections : dict
            dictionary of corrections

        Returns
        -------
        dict
            dictionary of corrections, with mean of zero

        """
        shift = np.min(list(corrections.values()))
        for lig in corrections.keys():
            corrections[lig] = corrections[lig] - shift
        return corrections

    def _absolute_from_relative(g, experimental, experimental_error):
        """Use DiffNet to compute absolute free energies, from computed relative free energies and assign absolute experimental values and uncertainties.

        Parameters
        ----------
        g : nx.DiGraph()
            A graph with relative results assigned to edges
        experimental : list
            list of experimental absolute affinities in kcal/mol
        experimental_error : list
            list of experimental absolute uncertainties in kcal/mol

        Returns
        -------

        """
        f_i_calc, C_calc = stats.mle(g, factor='calc_DDG')
        variance = np.diagonal(C_calc)
        for i, (f_i, df_i) in enumerate(zip(f_i_calc, variance**0.5)):
            g.nodes[i]['calc_DG'] = f_i
            g.nodes[i]['calc_dDG'] = df_i
            g.nodes[i]['exp_DG'] = experimental[i]
            g.nodes[i]['exp_dDG'] = experimental_error[i]

    def _plot_absolute(g, name='MM', MM_ff='', ML_ff='', color='k'):
        """ Plots calculated vs experimental results for absolute free energies in kcal/mol
        Saves a file to NAME_absolute.pdb

        Parameters
        ----------
        g : nx.DiGraph()
            Graph with absolute values assigned to nodes
        name : str, default='MM'
            name of method, used to label plot and filename
        MM_ff : str, default=''
            Name of MM method for labelling
        ML_ff : str, default=''
            Name of ML method for labelling
        color : str, default='k'
            Anything matplotlib.pyplot can understand as a color

        Returns
        -------

        """
        filename = f"{name.replace('/','-')}_absolute.pdf"
        plotting.plot_DGs(g, title=f'{name}: {MM_ff}\n {ML_ff}',
                          centralizing=True, color=color,
                          shift=np.mean(experimental),
                          filename=filename)

    def _plot_relative(g, name='MM', MM_ff='', ML_ff='', color='k'):
        """ Plots calculated vs experimental results for relative free energies in kcal/mol
        Saves a file to NAME_relative.pdb

        Parameters
        ----------
        g : nx.DiGraph()
            Graph with relative values assigned to nodes
        name : str, default='MM'
            name of method, used to label plot and filename
        MM_ff : str, default=''
            Name of MM method for labelling
        ML_ff : str, default=''
            Name of ML method for labelling
        color : str, default='k'
            Anything matplotlib.pyplot can understand as a color

        Returns
        -------

        """
        filename = f"{name.replace('/','-')}_relative.pdf"
        for edge in g.edges(data=True):
            edge[2]['calc_DDG'] = g.nodes[edge[1]]['calc_DG'] - g.nodes[edge[0]]['calc_DG']
            edge[2]['exp_dDDG'] = (g.nodes[edge[1]]['exp_dDG']**2+g.nodes[edge[0]]['exp_dDG']**2)**0.5
        plotting.plot_DDGs(g, title=f'{name}: {MM_ff}\n {ML_ff}', color=color, filename=filename)

    mm_g = _make_mm_graph(mm_results, experimental, experimental_error)
    binding_corrections = _per_ligand_correction(ml_corrections)
    shifted_corrections = _relative_corrections(binding_corrections)
    ml_g = _make_ml_graph(mm_g, shifted_corrections)

    _plot_absolute(mm_g, name='MM', MM_ff=MM_ff, color='#6C8EBF')
    _plot_absolute(ml_g, name='ML/MM', MM_ff=MM_ff, ML_ff=ML_ff, color='#D79B00')
    _plot_relative(mm_g, name='MM', MM_ff=MM_ff, color='#6C8EBF')
    _plot_relative(ml_g, name='ML/MM', MM_ff=MM_ff, ML_ff=ML_ff, color='#D79B00')
